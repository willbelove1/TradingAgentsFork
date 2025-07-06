# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"
        return "Risky Analyst"

    def route_trader_consultation(self, state: AgentState) -> str:
        """Determine the next step after TraderConsultNode."""
        # This condition relies on state fields updated by trader_node
        if state.get("clarification_requested") is True:
            return "ResearcherClarifyNode"
        else:
            # Assuming if clarification is not requested, a decision has been made or attempted.
            # The trader_node's return structure includes a 'status'.
            # If 'status' == 'decision_made', then proceed.
            # This logic might need to be more robust based on all possible states from trader_node.
            # For now, if not requesting clarification, assume proceed to next main step.
            return "Risky Analyst" # Next main step after trader

    def route_after_researcher_clarification(self, state: AgentState) -> str:
        """Determine the next step after ResearcherClarifyNode."""
        # This edge goes back to TraderConsultNode if clarification_response is available.
        if state.get("clarification_response") is not None:
            return "TraderConsultNode"
        else:
            # This case should ideally not happen if researcher_node always provides a response.
            # Or, could be a loop exit condition if max_consultation_rounds is hit,
            # but that logic is in trader_node.
            # If no response, something is wrong, or it might mean end of this sub-flow.
            # For now, let's assume it always goes back if there's a response.
            # If no response, it's an implicit dead-end for this path by this condition,
            # which means the graph must have another way out or it's an error.
            # However, the task implies a direct edge if response is present.
            # Let's make it simple: if response, go to Trader. Otherwise, end this conditional path.
            # The graph structure should ensure this isn't a permanent dead end.
            # The original spec was: add_edge("researcher", "trader", condition="state.get('clarification_response') is not None")
            # This implies if the condition is false, the edge is not taken.
            # It might be better for route_trader_consultation to handle the "no clarification needed" case
            # and this one to only handle the loop back.
            # The example `add_conditional_edges` takes a map of outcomes.
            # So this function should return the key for the map.
            return "TraderConsultNode" # Always go back to trader to process the clarification.
            # The graph will have:
            # workflow.add_conditional_edges(
            # "ResearcherClarifyNode",
            # self.conditional_logic.route_after_researcher_clarification,
            # {"TraderConsultNode": "TraderConsultNode"} # Only one path if condition met
            # )
            # This means the condition "state.get('clarification_response') is not None" should be part of the edge definition itself,
            # not the routing function's return value.
            # Let's redefine: route_after_researcher_clarification always routes to TraderConsultNode,
            # and the graph edge itself is conditional on clarification_response.
            # This is simpler. Or the function returns where to go.

            # Simpler: if clarification_response is present, go to TraderConsultNode.
            # If not, it's an error or unexpected state. For graph, we must return a valid node name or END.
            # Let's assume researcher always sets clarification_response.
            return "TraderConsultNode"
