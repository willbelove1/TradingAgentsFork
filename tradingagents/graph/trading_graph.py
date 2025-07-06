# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode
from tradingagents.llm_client import LLMClientFactory

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.interface import set_config

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
        profile: str = None, # Added profile argument
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            profile: Model profile to use ('fast', 'deep', or None for default).
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.profile = profile

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs using the factory
        llm_factory = LLMClientFactory(self.config)
        # Ensure your config has "deep_think_llm_key" and "quick_think_llm_key"
        # or adjust LLMClientFactory to use default keys like "deep_think_llm"
        # The factory's get_llm method expects the actual model name key.
        # The convenience methods get_deep_thinking_llm and get_quick_thinking_llm use
        # config keys "deep_think_llm_key" (defaulting to "deep_think_llm") and
        # "quick_think_llm_key" (defaulting to "quick_think_llm") to find the *actual model name*.

        # Let's adjust TradingAgentsGraph to pass the correct keys if they are different,
        # or rely on the defaults in LLMClientFactory.
        # The config structure is self.config["deep_think_llm"] = "model-name-a"
        # and self.config["quick_think_llm"] = "model-name-b".
        # The LLMClientFactory.get_llm expects a key that *points* to the model name.
        # So, we need to pass "deep_think_llm" (the key) not self.config["deep_think_llm"] (the value).

        # Determine model names based on profile
        # These are placeholders; actual model names would depend on the provider and specific SKUs
        # For example, for OpenAI: fast_model="gpt-3.5-turbo", deep_model="gpt-4"
        # For Google: fast_model="gemini-flash", deep_model="gemini-pro"
        # The config should ideally store these mappings, e.g. self.config["profiles"]["fast"]["quick_think_llm"]
        # For now, hardcoding logic based on common Gemini names for illustration.
        # User should ensure their config["llm_provider"] matches the models chosen here.

        provider = self.config.get("llm_provider", "").lower()

        if self.profile == "fast":
            # For 'google' provider
            general_deep_model_name = "gemini-1.0-pro" # Using 1.0 pro as flash might be too weak for deep
            general_quick_model_name = "gemini-1.0-pro" # Or gemini-flash if available and suitable
            trader_model_name = "gemini-1.0-pro"
            researcher_clarify_model_name = "gemini-1.0-pro"
            if provider == "openai":
                general_deep_model_name = self.config.get("quick_think_llm", "gpt-3.5-turbo") # Use quick as deep for fast
                general_quick_model_name = self.config.get("quick_think_llm", "gpt-3.5-turbo")
                trader_model_name = self.config.get("quick_think_llm", "gpt-3.5-turbo")
                researcher_clarify_model_name = self.config.get("quick_think_llm", "gpt-3.5-turbo")
        elif self.profile == "deep":
            # For 'google' provider
            general_deep_model_name = "gemini-pro"
            general_quick_model_name = "gemini-1.0-pro" # Or gemini-flash
            trader_model_name = "gemini-pro"
            researcher_clarify_model_name = "gemini-1.0-pro" # Researcher (clarifier) uses a quicker model
            if provider == "openai":
                general_deep_model_name = self.config.get("deep_think_llm", "gpt-4")
                general_quick_model_name = self.config.get("quick_think_llm", "gpt-3.5-turbo")
                trader_model_name = self.config.get("deep_think_llm", "gpt-4") # Trader uses deep model
                researcher_clarify_model_name = self.config.get("quick_think_llm", "gpt-3.5-turbo")
        else: # Default behavior: use models directly from config
            self.deep_thinking_llm = llm_factory.get_llm(model_type="deep_think_llm")
            self.quick_thinking_llm = llm_factory.get_llm(model_type="quick_think_llm")
            # For trader and researcher, if not using profiles, they might default to quick_thinking_llm in GraphSetup
            # or we need specific config keys for them.
            # With the new get_llm_by_name, it's better to be explicit.
            # So, even for default, let's define them to ensure they are created.
            # These would ideally also come from config if not for profiles.
            default_trader_model = self.config.get("trader_llm_model", self.config.get("deep_think_llm"))
            default_researcher_model = self.config.get("researcher_llm_model", self.config.get("quick_think_llm"))

            self.trader_llm = llm_factory.get_llm_by_name(default_trader_model)
            self.researcher_clarify_llm = llm_factory.get_llm_by_name(default_researcher_model)

        if self.profile in ["fast", "deep"]:
            self.deep_thinking_llm = llm_factory.get_llm_by_name(general_deep_model_name)
            self.quick_thinking_llm = llm_factory.get_llm_by_name(general_quick_model_name)
            self.trader_llm = llm_factory.get_llm_by_name(trader_model_name)
            self.researcher_clarify_llm = llm_factory.get_llm_by_name(researcher_clarify_model_name)
        
        self.toolkit = Toolkit(config=self.config)

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            trader_llm=self.trader_llm,  # Pass the specific trader LLM
            researcher_clarify_llm=self.researcher_clarify_llm # Pass the specific researcher LLM
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

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
                    self.toolkit.get_stock_news_openai,
                    # offline tools
                    self.toolkit.get_reddit_stock_info,
                ]
            ),
            "news": ToolNode(
                [
                    # online tools
                    self.toolkit.get_global_news_openai,
                    self.toolkit.get_google_news,
                    # offline tools
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_reddit_news,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_fundamentals_openai,
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
