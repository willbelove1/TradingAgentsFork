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
        self.agent_configs = self.config.get('agent_model_configs', {})
        default_provider = self.config.get("llm_provider", "google").lower()

        # --- Load Prompts ---
        # Assuming prompt_loader is accessible and loads based on default or specified provider in future
        from tradingagents.config.prompt_loader import load_prompts_from_file, get_prompt_config, format_prompt, PROMPTS_DIR, GEMINI_PROMPTS_FILENAME
        # Determine path to prompts directory (assuming it's relative to project root)
        project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # For now, always load Gemini prompts. Can be made dynamic by provider later.
        gemini_prompts_filepath = os.path.join(project_root_path, PROMPTS_DIR, GEMINI_PROMPTS_FILENAME)
        self.all_prompts = load_prompts_from_file(gemini_prompts_filepath)
        if not self.all_prompts:
            print(f"Warning: Could not load any prompts from {gemini_prompts_filepath}. Prompts will be default or hardcoded.")


        # --- Initialize a dictionary of BaseLLMClient instances, one per provider needed ---
        self.llm_clients: Dict[str, BaseLLMClient] = {}

        # Determine all unique providers mentioned in agent_model_configs and the default
        providers_needed = set([default_provider])
        for agent_cfg in self.agent_configs.values():
            if isinstance(agent_cfg, dict) and 'llm_provider' in agent_cfg:
                providers_needed.add(agent_cfg['llm_provider'].lower())
        for lc_model_cfg in self.config.get('langchain_chat_models', {}).values():
            if isinstance(lc_model_cfg, dict) and 'provider' in lc_model_cfg:
                 providers_needed.add(lc_model_cfg['provider'].lower())


        for provider_name in providers_needed:
            if provider_name not in self.llm_clients:
                # Construct specific config for this provider's client
                # Start with global config, then add provider-specific block, then set llm_provider
                provider_client_config = self.config.copy() # Base overall config

                # Merge provider-specific config block (e.g., google_config, openai_config)
                # The keys in these blocks (like default_text_model) will be used by the client's __init__
                # if they are not overridden by agent-specific model settings later.
                if f"{provider_name}_config" in self.config:
                    provider_client_config.update(self.config[f"{provider_name}_config"])

                # Ensure the client knows which provider it is, and also provide the top-level defaults
                # if its own provider_config block didn't have them.
                provider_client_config['llm_provider'] = provider_name
                if 'default_text_model' not in provider_client_config: # if not in openai_config etc.
                    provider_client_config['default_text_model'] = self.config.get('default_text_model')
                if 'default_embedding_model' not in provider_client_config:
                    provider_client_config['default_embedding_model'] = self.config.get('default_embedding_model')

                # Pass API key from provider_config if present, else client handles env vars
                api_key_for_provider = provider_client_config.get('api_key')

                self.llm_clients[provider_name] = get_llm_client(
                    provider_name=provider_name, # Explicitly tell factory
                    api_key=api_key_for_provider,
                    config=provider_client_config
                )
                print(f"Initialized LLM Client for provider: {provider_name} -> {self.llm_clients[provider_name].__class__.__name__}")


        # --- Initialize Langchain ChatModel instances based on model_settings.yaml ---
        self.langchain_chat_models_lc: Dict[str, Any] = {} # Stores ChatModel instances
        lc_model_definitions = self.config.get('langchain_chat_models', {})

        for key, lc_def in lc_model_definitions.items():
            lc_provider_name = lc_def.get('provider', default_provider).lower()
            lc_model_name = lc_def.get('model_name')

            if not lc_model_name:
                print(f"Warning: model_name not found for Langchain model key '{key}'. Skipping.")
                continue

            # Get provider-specific config (e.g., openai_config for base_url)
            provider_specific_lc_config = self.config.get(f"{lc_provider_name}_config", {})
            api_key_for_lc = provider_specific_lc_config.get('api_key') # Client SDKs will try env if None

            if lc_provider_name == "google":
                if not os.getenv("GOOGLE_API_KEY") and not api_key_for_lc: # Check before direct init
                    try:
                        from dotenv import load_dotenv; load_dotenv()
                        if not os.getenv("GOOGLE_API_KEY") and not api_key_for_lc: # Check again
                            raise ValueError(f"GOOGLE_API_KEY not found for Langchain Google model: {lc_model_name}")
                    except ImportError: raise ValueError("dotenv not installed for Langchain Google key loading.")
                self.langchain_chat_models_lc[key] = ChatGoogleGenerativeAI(model=lc_model_name, google_api_key=api_key_for_lc or os.getenv("GOOGLE_API_KEY"))
            elif lc_provider_name == "openai":
                openai_base_url = provider_specific_lc_config.get("base_url")
                # OpenAI SDK uses OPENAI_API_KEY env var by default if api_key param is None
                self.langchain_chat_models_lc[key] = ChatOpenAI(model=lc_model_name, base_url=openai_base_url, api_key=api_key_for_lc)
            elif lc_provider_name == "anthropic":
                self.langchain_chat_models_lc[key] = ChatAnthropic(model=lc_model_name, anthropic_api_key=api_key_for_lc or os.getenv("ANTHROPIC_API_KEY"))
            else:
                raise ValueError(f"Unsupported LLM provider '{lc_provider_name}' for Langchain ChatModel key '{key}'")
            print(f"Initialized Langchain ChatModel '{key}': Provider: {lc_provider_name}, Model: {lc_model_name}")

        self.toolkit = Toolkit(config=self.config)

        # --- Initialize memories ---
        # Each memory might use a different provider if specified in its agent_config, else default.
        # The llm_client passed to memory should be the one for its designated provider.
        def get_client_for_component(component_name: str) -> BaseLLMClient:
            component_cfg = self.agent_configs.get(component_name, {})
            provider = component_cfg.get('llm_provider', default_provider).lower()
            return self.llm_clients[provider]

        self.bull_memory = FinancialSituationMemory("bull_memory", self.config, llm_client=get_client_for_component("FinancialSituationMemory")) # Assuming FSM uses one client for all instances
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config, llm_client=get_client_for_component("FinancialSituationMemory"))
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config, llm_client=get_client_for_component("FinancialSituationMemory"))
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config, llm_client=get_client_for_component("FinancialSituationMemory"))
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config, llm_client=get_client_for_component("FinancialSituationMemory"))


        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()

        self.graph_setup = GraphSetup(
            # GraphSetup now needs the dictionary of Langchain models, not just two instances
            langchain_chat_models_lc_dict=self.langchain_chat_models_lc,
            toolkit=self.toolkit,
            tool_nodes=self.tool_nodes,
            bull_memory=self.bull_memory,
            bear_memory=self.bear_memory,
            trader_memory=self.trader_memory,
            invest_judge_memory=self.invest_judge_memory,
            risk_manager_memory=self.risk_manager_memory,
            conditional_logic=self.conditional_logic,
            agent_model_configs=self.agent_configs,
            all_prompts=self.all_prompts # Pass loaded prompts to GraphSetup
        )

        self.propagator = Propagator()

        # Initialize Reflector with its specific prompt config
        reflector_comp_cfg = self.agent_configs.get('Reflector', {})
        reflector_prompt_key = reflector_comp_cfg.get('prompt_key', 'ReflectorTask') # Default key
        reflector_prompt_cfg = get_prompt_config(reflector_prompt_key, prompts_config=self.all_prompts)
        reflector_client = get_client_for_component('Reflector')
        self.reflector = Reflector(
            llm_client=reflector_client,
            component_config=reflector_comp_cfg,
            prompt_config=reflector_prompt_cfg
        )

        # Initialize SignalProcessor with its specific prompt config
        signal_processor_comp_cfg = self.agent_configs.get('SignalProcessor', {})
        signal_processor_prompt_key = signal_processor_comp_cfg.get('prompt_key', 'SignalProcessorTask')
        signal_processor_prompt_cfg = get_prompt_config(signal_processor_prompt_key, prompts_config=self.all_prompts)
        signal_processor_client = get_client_for_component('SignalProcessor')
        self.signal_processor = SignalProcessor(
            llm_client=signal_processor_client,
            component_config=signal_processor_comp_cfg,
            prompt_config=signal_processor_prompt_cfg
        )

        # Configure dataflows.interface with its client and prompt config (for one default prompt)
        from tradingagents.dataflows import interface as dataflows_interface
        interface_functions_comp_cfg = self.agent_configs.get('InterfaceFunctions', {})
        interface_prompt_key = interface_functions_comp_cfg.get('prompt_key', 'NewsSummarization') # Default key for interface functions
        interface_prompt_cfg = get_prompt_config(interface_prompt_key, prompts_config=self.all_prompts)
        interface_client = get_client_for_component('InterfaceFunctions')
        dataflows_interface.set_interface_llm_client(
            client=interface_client,
            component_config=interface_functions_comp_cfg,
            default_prompt_config=interface_prompt_cfg # Pass the loaded prompt config
        )

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
