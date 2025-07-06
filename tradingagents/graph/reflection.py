# TradingAgents/graph/reflection.py

from typing import Dict, Any
# Remove direct ChatModel import, will use BaseLLMClient
# from langchain_google_genai import ChatGoogleGenerativeAI
from tradingagents.llm_clients import BaseLLMClient # Import BaseLLMClient

class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, llm_client: BaseLLMClient, component_config: Dict = None):
        """
        Initialize the reflector with an LLM client and component-specific configuration.
        Args:
            llm_client (BaseLLMClient): The LLM client instance.
            component_config (Dict, optional): Configuration for this component, potentially
                                               containing 'model', 'temperature', 'max_tokens'.
        """
        self.llm_client = llm_client
        self.component_config = component_config if component_config else {}
        self.reflection_system_prompt = self._get_reflection_prompt()
        # Potentially override default model from llm_client if specified in component_config
        self.model_name = self.component_config.get('model', self.llm_client.model_name)
        self.temperature = self.component_config.get('temperature', 0.6) # Default from previous hardcoding
        self.max_tokens = self.component_config.get('max_tokens', 2048)  # Default from previous hardcoding


    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
You are an expert financial analyst tasked with reviewing trading decisions/analysis and providing a comprehensive, step-by-step analysis. 
Your goal is to deliver detailed insights into investment decisions and highlight opportunities for improvement, adhering strictly to the following guidelines:

1. Reasoning:
   - For each trading decision, determine whether it was correct or incorrect. A correct decision results in an increase in returns, while an incorrect decision does the opposite.
   - Analyze the contributing factors to each success or mistake. Consider:
     - Market intelligence.
     - Technical indicators.
     - Technical signals.
     - Price movement analysis.
     - Overall market data analysis 
     - News analysis.
     - Social media and sentiment analysis.
     - Fundamental data analysis.
     - Weight the importance of each factor in the decision-making process.

2. Improvement:
   - For any incorrect decisions, propose revisions to maximize returns.
   - Provide a detailed list of corrective actions or improvements, including specific recommendations (e.g., changing a decision from HOLD to BUY on a particular date).

3. Summary:
   - Summarize the lessons learned from the successes and mistakes.
   - Highlight how these lessons can be adapted for future trading scenarios and draw connections between similar situations to apply the knowledge gained.

4. Query:
   - Extract key insights from the summary into a concise sentence of no more than 1000 tokens.
   - Ensure the condensed sentence captures the essence of the lessons and reasoning for easy reference.

Adhere strictly to these instructions, and ensure your output is detailed, accurate, and actionable. You will also be given objective descriptions of the market from a price movements, technical indicator, news, and sentiment perspective to provide more context for your analysis.
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        # Construct a single prompt string for generate_text
        # The system prompt can be prepended or incorporated into the user prompt.
        # For simplicity here, we'll prepend it.
        full_prompt = f"{self.reflection_system_prompt}\n\n" \
                      f"Component Type: {component_type}\n" \
                      f"Returns/Losses: {returns_losses}\n\n" \
                      f"Analysis/Decision Provided: {report}\n\n" \
                      f"Objective Market Reports for Reference:\n{situation}\n\n" \
                      f"Based on all the above, provide your step-by-step analysis, improvement suggestions, summary, and query."

        from tradingagents.llm_clients.base_client import logger as client_logger # Use the same logger for context
        client_logger.info(f"Reflector: Calling LLM. Model: {self.model_name}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Component: {component_type}")

        # Using the llm_client's generate_text method with model and parameters from component_config
        result = self.llm_client.generate_text(
            prompt=full_prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """Reflect on bull researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """Reflect on bear researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """Reflect on investment judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """Reflect on risk manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
