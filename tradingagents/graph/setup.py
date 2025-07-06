# TradingAgents/graph/setup.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI
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
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        toolkit: Toolkit,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
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
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        # The original trader_node created by create_trader is being replaced by the new logic.
        # trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Import new nodes
        from tradingagents.agents.trader.trader import trader_node as new_trader_node_logic
        from tradingagents.agents.managers.research_manager import researcher_node as new_researcher_node_logic

        # TODO: Load prompt_cfg from yaml files (model_settings.yaml and prompts/gemini_prompts.yaml)
        # This is a placeholder. In a real scenario, this would involve reading YAML files
        # and constructing the prompt_cfg dictionary.
        # For example:
        # import yaml
        # with open("prompts/gemini_prompts.yaml", 'r') as f:
        #     gemini_prompts = yaml.safe_load(f)
        # with open("model_settings.yaml", 'r') as f:
        #     model_settings = yaml.safe_load(f)
        #
        # prompt_cfg_for_trader = {
        #     "evaluate": gemini_prompts[model_settings['agents']['trader']['prompts']['evaluate']],
        #     "clarify": gemini_prompts[model_settings['agents']['trader']['prompts']['clarify']]
        # }
        # prompt_cfg_for_researcher = {
        #     "clarify": gemini_prompts[model_settings['agents']['researcher']['prompts']['clarify']]
        # }
        # This is a simplified stand-in:
        self.prompt_cfg_trader = {
            "evaluate": "Bạn là một nhà giao dịch thông minh. Hãy đọc báo cáo dưới đây và trả lời:\n\nBáo cáo có đủ cơ sở để ra quyết định chưa?\nCó cần yêu cầu làm rõ gì không?\nOutput: JSON {\n  \"decision_ready\": true/false,\n  \"clarification_needed\": true/false,\n  \"clarification_question\": \"...\"\n}",
            "clarify": "Với thông tin làm rõ (nếu có) và báo cáo ban đầu, hãy ra quyết định cuối cùng.\nOutput: JSON {\n  \"action\": \"...\",\n  \"reason\": \"...\"\n}"
        }
        self.prompt_cfg_researcher = {
            "clarify": "Một nhà giao dịch cần bạn làm rõ thông tin sau: {question}. Hãy trả lời ngắn gọn, logic."
        }

        # Instantiate new nodes with necessary llm and prompt_cfg
        # Assuming self.quick_thinking_llm can be used for both, or specific clients are configured elsewhere.
        # The trader_node expects input_data (report), state, llm_client, prompt_cfg.
        # LangGraph nodes receive the full state as input. We need to ensure 'input_data' is correctly sourced.
        # The 'input_data' for trader_node is the research report, which should be in state["investment_plan"] (output of Research Manager).

        def trader_consult_node_wrapper(state: AgentState):
            # The research report is expected to be in state["investment_plan"] from Research Manager
            report = state.get("investment_plan")
            # The new trader_node updates state internally and returns a dict that langgraph merges.
            # It needs llm_client and prompt_cfg.
            # Using quick_thinking_llm for now as the llm_client.
            return new_trader_node_logic(report, state, self.quick_thinking_llm, self.prompt_cfg_trader)

        def researcher_clarify_node_wrapper(state: AgentState):
            # researcher_node expects input_data (not strictly used if question is from state), state, llm_client, prompt_cfg
            # The question is in state["clarification_question"]
            # Using quick_thinking_llm for researcher as well (gemini-flash mapping).
            # input_data is not strictly needed by researcher_node as it reads question from state.
            return new_researcher_node_logic(None, state, self.quick_thinking_llm, self.prompt_cfg_researcher)

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        safe_analyst = create_safe_debator(self.quick_thinking_llm)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory
        )

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
        # workflow.add_node("Trader", trader_node) # Original trader node replaced
        workflow.add_node("TraderConsultNode", trader_consult_node_wrapper)
        workflow.add_node("ResearcherClarifyNode", researcher_clarify_node_wrapper)
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
        # workflow.add_edge("Research Manager", "Trader") # Old edge
        workflow.add_edge("Research Manager", "TraderConsultNode") # New: RM output goes to Trader

        # Conditional routing from TraderConsultNode
        workflow.add_conditional_edges(
            "TraderConsultNode",
            self.conditional_logic.route_trader_consultation,
            {
                "ResearcherClarifyNode": "ResearcherClarifyNode", # If clarification needed
                "Risky Analyst": "Risky Analyst" # If decision made, proceed to risk analysis
            }
        )

        # Edge from ResearcherClarifyNode back to TraderConsultNode to process clarification
        workflow.add_edge("ResearcherClarifyNode", "TraderConsultNode")

        # workflow.add_edge("Trader", "Risky Analyst") # Old edge, replaced by conditional above
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
