# TradingAgents/graph/reflection.py

from typing import Dict, Any
# Remove direct ChatModel import, will use BaseLLMClient
# from langchain_google_genai import ChatGoogleGenerativeAI
from tradingagents.llm_clients import BaseLLMClient # Import BaseLLMClient

from tradingagents.config.prompt_loader import format_prompt # Import formatter

class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, llm_client: BaseLLMClient, component_config: Optional[Dict] = None, prompt_config: Optional[Dict] = None):
        """
        Initialize the reflector.
        Args:
            llm_client (BaseLLMClient): The LLM client instance.
            component_config (Dict, optional): Config for model, temp, max_tokens.
            prompt_config (Dict, optional): Config for system_prompt, user_template from prompts YAML.
        """
        self.llm_client = llm_client
        self.component_config = component_config if component_config else {}
        self.prompt_config = prompt_config if prompt_config else {}

        # Get model and generation parameters
        self.model_name = self.component_config.get('model', self.llm_client.model_name)
        self.temperature = self.component_config.get('temperature', 0.6)
        self.max_tokens = self.component_config.get('max_tokens', 2048)

        # Get prompt parts from prompt_config
        # If prompt_config is empty or doesn't have the keys, it will use defaults or empty strings.
        # The 'ReflectorTask' from gemini_prompts.yaml should provide these.
        self.system_prompt = self.prompt_config.get('system_prompt', self._get_default_reflection_system_prompt())
        self.user_template = self.prompt_config.get('user_prompt_template', self._get_default_reflection_user_template())
        # Few-shot examples can be loaded if ReflectorTask in YAML defines them
        # self.few_shot_examples_string = self.prompt_config.get('few_shot_examples_string', "")


    def _get_default_reflection_system_prompt(self) -> str: # Renamed for clarity
        """Get the default system prompt for reflection if not provided in YAML."""
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

    def _get_default_reflection_user_template(self) -> str:
        """Get the default user prompt template for reflection if not provided in YAML."""
        return """Component Type: {component_type}
Returns/Losses: {returns_losses}
Analysis/Decision Provided: {report_text}
Objective Market Reports for Reference:
{situation_context}
---
Dựa trên tất cả thông tin trên, hãy cung cấp phân tích chi tiết từng bước, đề xuất cải thiện, tóm tắt bài học kinh nghiệm, và một câu truy vấn (query) ngắn gọn (tối đa 1000 token) để ghi nhớ bài học này."""


    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""

        context_vars = {
            "component_type": component_type,
            "returns_losses": returns_losses,
            "report_text": report, # Ensure placeholder in YAML matches this
            "situation_context": situation # Ensure placeholder in YAML matches this
        }

        # Use the new format_prompt function with the loaded prompt_config
        # The prompt_config for Reflector should have 'system_prompt' and 'user_prompt_template'
        # and potentially 'few_shot_examples_string'.
        # We pass self.prompt_config (which was loaded from self.all_prompts[prompt_key])
        # and the context_vars to fill the user_template.
        # The format_prompt function will combine system_prompt, few_shots (if any from prompt_config), and filled user_template.

        # Construct the full prompt using the loaded configurations
        # The self.prompt_config should contain 'system_prompt', 'user_prompt_template', etc.
        # if 'ReflectorTask' was correctly defined and loaded.
        # If self.prompt_config is empty, format_prompt might use defaults or return empty.

        # We need to ensure self.prompt_config is not empty and has the necessary keys.
        # For this, we create a temporary config dict to pass to format_prompt,
        # using the Reflector's specific system_prompt and user_template loaded in __init__.
        current_prompt_structure = {
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_template,
            "few_shot_examples_string": self.prompt_config.get('few_shot_examples_string', "") # Use if defined
            # "type" is not strictly needed by format_prompt if we handle logic here.
        }

        full_prompt = format_prompt(current_prompt_structure, context_vars)

        if not full_prompt.strip(): # Check if prompt is empty after formatting
            logger.error("Reflector: Generated prompt is empty. Check prompt_config and context_vars.")
            return "Error: Could not generate reflection prompt."

        from tradingagents.llm_clients.base_client import logger # Ensure logger is accessible
        logger.info(f"Reflector: Calling LLM. Model: {self.model_name}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Component: {component_type}")

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
