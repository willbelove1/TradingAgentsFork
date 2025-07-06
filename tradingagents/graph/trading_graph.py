# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.interface import set_config
from tradingagents.llm_clients import get_llm_client # Import the factory

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Config now includes merged model_settings.yaml
        # Initialize BaseLLMClient (used by Reflector, SignalProcessor, Memory, InterfaceFunctions)
        # The get_llm_client factory will use 'llm_provider', 'default_text_model',
        # 'default_embedding_model', 'openai_base_url' etc. from self.config
        self.llm_client = get_llm_client(config=self.config)

        # Initialize Langchain ChatModel instances based on model_settings.yaml
        # These are for Langchain agents that expect ChatModel objects.
        lc_provider = self.config.get("llm_provider", "google").lower()

        # Get model names for quick and deep thinkers from the YAML structure
        # (e.g., from self.config['langchain_chat_models']['quick_llm_lc']['model_name'])
        quick_llm_lc_key = self.config.get('langchain_quick_thinker_key', 'quick_llm_lc')
        deep_llm_lc_key = self.config.get('langchain_deep_thinker_key', 'deep_llm_lc')

        quick_llm_lc_model_name = self.config.get('langchain_chat_models', {}).get(quick_llm_lc_key, {}).get('model_name')
        deep_llm_lc_model_name = self.config.get('langchain_chat_models', {}).get(deep_llm_lc_key, {}).get('model_name')

        if not quick_llm_lc_model_name:
            quick_llm_lc_model_name = self.config.get('default_text_model', 'gemini-1.5-flash') # Fallback
            print(f"Warning: quick_llm_lc model name not found in model_settings.yaml, using default: {quick_llm_lc_model_name}")
        if not deep_llm_lc_model_name:
            deep_llm_lc_model_name = self.config.get('default_text_model', 'gemini-1.0-pro') # Fallback (could be same as default or a more robust one)
            print(f"Warning: deep_llm_lc model name not found in model_settings.yaml, using default: {deep_llm_lc_model_name}")

        if lc_provider == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    if not os.getenv("GOOGLE_API_KEY"):
                        raise ValueError("GOOGLE_API_KEY not found for Langchain Google models.")
                except ImportError: # pragma: no cover
                    raise ValueError("dotenv not installed. Cannot load GOOGLE_API_KEY for Langchain Google models.")
            self.quick_thinking_llm_lc = ChatGoogleGenerativeAI(model=quick_llm_lc_model_name)
            self.deep_thinking_llm_lc = ChatGoogleGenerativeAI(model=deep_llm_lc_model_name)
        elif lc_provider == "openai": # This now also covers Ollama/OpenRouter if openai_base_url is set
            openai_base_url = self.config.get("openai_base_url")
            openai_api_key = self.config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
            self.quick_thinking_llm_lc = ChatOpenAI(model=quick_llm_lc_model_name, base_url=openai_base_url, api_key=openai_api_key)
            self.deep_thinking_llm_lc = ChatOpenAI(model=deep_llm_lc_model_name, base_url=openai_base_url, api_key=openai_api_key)
        elif lc_provider == "anthropic":
            # Ensure ANTHROPIC_API_KEY is in env
            self.quick_thinking_llm_lc = ChatAnthropic(model=quick_llm_lc_model_name)
            self.deep_thinking_llm_lc = ChatAnthropic(model=deep_llm_lc_model_name)
        else:
            raise ValueError(f"Unsupported LLM provider for Langchain ChatModels: {lc_provider}")

        self.toolkit = Toolkit(config=self.config)

        # Initialize memories - Pass the llm_client for embeddings
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config, llm_client=self.llm_client)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config, llm_client=self.llm_client)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config, llm_client=self.llm_client)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config, llm_client=self.llm_client)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config, llm_client=self.llm_client)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()

        # Pass the specifically configured Langchain ChatModel instances to GraphSetup
        self.graph_setup = GraphSetup(
            quick_thinking_llm_lc=self.quick_thinking_llm_lc, # Explicitly pass the correct instance
            deep_thinking_llm_lc=self.deep_thinking_llm_lc,   # Explicitly pass the correct instance
            toolkit=self.toolkit,
            tool_nodes=self.tool_nodes,
            bull_memory=self.bull_memory,
            bear_memory=self.bear_memory,
            trader_memory=self.trader_memory,
            invest_judge_memory=self.invest_judge_memory,
            risk_manager_memory=self.risk_manager_memory,
            conditional_logic=self.conditional_logic,
            agent_model_configs=self.config.get('agent_model_configs', {}) # Pass agent configs
        )

        self.propagator = Propagator()

        # Pass the BaseLLMClient and its specific config to Reflector and SignalProcessor
        reflector_config = self.config.get('agent_model_configs', {}).get('Reflector', {})
        self.reflector = Reflector(llm_client=self.llm_client, component_config=reflector_config)

        signal_processor_config = self.config.get('agent_model_configs', {}).get('SignalProcessor', {})
        self.signal_processor = SignalProcessor(llm_client=self.llm_client, component_config=signal_processor_config)

        # Set the llm_client for the interface module to ensure it uses the same instance
        # Also pass its specific config if needed, or let it use the global client's config
        from tradingagents.dataflows import interface as dataflows_interface
        interface_functions_config = self.config.get('agent_model_configs', {}).get('InterfaceFunctions', {})
        dataflows_interface.set_interface_llm_client(self.llm_client, component_config=interface_functions_config)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources."""
        return {
            "market": ToolNode(
                [
                    # online tools
                    self.toolkit.get_YFin_data_online,
                    self.toolkit.get_stockstats_indicators_report_online,
                    # offline tools
                    self.toolkit.get_YFin_data,
                    self.toolkit.get_stockstats_indicators_report,
                ]
            ),
            "social": ToolNode(
                [
                    # online tools
                    self.toolkit.get_llm_generated_stock_news, # Renamed
                    # offline tools
                    self.toolkit.get_reddit_stock_info,
                ]
            ),
            "news": ToolNode(
                [
                    # online tools
                    self.toolkit.get_llm_generated_global_news, # Renamed
                    self.toolkit.get_google_news,
                    # offline tools
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_reddit_news,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_llm_generated_fundamentals, # Renamed
                    # offline tools
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
