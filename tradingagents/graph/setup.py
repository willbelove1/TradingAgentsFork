# TradingAgents/graph/setup.py

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import Toolkit

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatGoogleGenerativeAI,
        deep_thinking_llm_lc: ChatGoogleGenerativeAI, # Renamed for clarity
        toolkit: Toolkit,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
        agent_model_configs: Dict[str, Any], # Added agent_model_configs
    ):
        """Initialize with required components."""
        self.quick_thinking_llm_lc = quick_thinking_llm_lc
        self.deep_thinking_llm_lc = deep_thinking_llm_lc
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.agent_model_configs = agent_model_configs # Store it
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            cfg = self.agent_model_configs.get("MarketAnalyst", {})
            llm_instance = self.quick_thinking_llm_lc if cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
            analyst_nodes["market"] = create_market_analyst(llm_instance, self.toolkit)
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            cfg = self.agent_model_configs.get("SocialMediaAnalyst", {})
            llm_instance = self.quick_thinking_llm_lc if cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
            analyst_nodes["social"] = create_social_media_analyst(llm_instance, self.toolkit)
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            cfg = self.agent_model_configs.get("NewsAnalyst", {})
            llm_instance = self.quick_thinking_llm_lc if cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
            analyst_nodes["news"] = create_news_analyst(llm_instance, self.toolkit)
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            cfg = self.agent_model_configs.get("FundamentalsAnalyst", {})
            llm_instance = self.quick_thinking_llm_lc if cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(llm_instance, self.toolkit)
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher and manager nodes
        bull_cfg = self.agent_model_configs.get("BullResearcher", {})
        bull_llm = self.quick_thinking_llm_lc if bull_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        bull_researcher_node = create_bull_researcher(bull_llm, self.bull_memory)

        bear_cfg = self.agent_model_configs.get("BearResearcher", {})
        bear_llm = self.quick_thinking_llm_lc if bear_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        bear_researcher_node = create_bear_researcher(bear_llm, self.bear_memory)

        research_mgr_cfg = self.agent_model_configs.get("ResearchManager", {})
        research_mgr_llm = self.quick_thinking_llm_lc if research_mgr_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        research_manager_node = create_research_manager(research_mgr_llm, self.invest_judge_memory)

        trader_cfg = self.agent_model_configs.get("Trader", {})
        trader_llm = self.quick_thinking_llm_lc if trader_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        trader_node = create_trader(trader_llm, self.trader_memory)

        # Create risk analysis nodes
        risky_cfg = self.agent_model_configs.get("RiskyDebator", {})
        risky_llm = self.quick_thinking_llm_lc if risky_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        risky_analyst = create_risky_debator(risky_llm)

        neutral_cfg = self.agent_model_configs.get("NeutralDebator", {})
        neutral_llm = self.quick_thinking_llm_lc if neutral_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        neutral_analyst = create_neutral_debator(neutral_llm)

        safe_cfg = self.agent_model_configs.get("SafeDebator", {})
        safe_llm = self.quick_thinking_llm_lc if safe_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        safe_analyst = create_safe_debator(safe_llm)

        risk_mgr_cfg = self.agent_model_configs.get("RiskManager", {})
        risk_mgr_llm = self.quick_thinking_llm_lc if risk_mgr_cfg.get("llm_instance_key") == self.agent_model_configs.get("langchain_quick_thinker_key") else self.deep_thinking_llm_lc
        risk_manager_node = create_risk_manager(risk_mgr_llm, self.risk_manager_memory)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()
