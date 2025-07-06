# TradingAgents/graph/signal_processing.py

# from langchain_google_genai import ChatGoogleGenerativeAI # Remove direct ChatModel import
from tradingagents.llm_clients import BaseLLMClient # Import BaseLLMClient

from typing import Dict, Optional # Added Optional
from tradingagents.config.prompt_loader import format_prompt # Import formatter
from tradingagents.llm_clients.base_client import logger # Import logger for consistency

class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, llm_client: BaseLLMClient, component_config: Optional[Dict] = None, prompt_config: Optional[Dict] = None):
        """
        Initialize with an LLM client, component-specific config, and prompt config.
        Args:
            llm_client (BaseLLMClient): The LLM client instance.
            component_config (Dict, optional): Config for model, temp, max_tokens.
            prompt_config (Dict, optional): Config for system_prompt, user_template from prompts YAML.
        """
        self.llm_client = llm_client
        self.component_config = component_config if component_config else {}
        self.prompt_config = prompt_config if prompt_config else {}

        self.model_name = self.component_config.get('model', self.llm_client.model_name)
        self.temperature = self.component_config.get('temperature', 0.1)
        self.max_tokens = self.component_config.get('max_tokens', 25)

        # Load prompt parts from prompt_config (e.g., from 'SignalProcessorTask' in YAML)
        self.system_prompt = self.prompt_config.get('system_prompt', self._get_default_signal_processor_system_prompt())
        self.user_template = self.prompt_config.get('user_prompt_template', self._get_default_signal_processor_user_template())
        # self.few_shot_examples_string = self.prompt_config.get('few_shot_examples_string', "") # Usually not needed for simple extraction

    def _get_default_signal_processor_system_prompt(self) -> str:
        return "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information."

    def _get_default_signal_processor_user_template(self) -> str:
        return """Financial Report/Signal:
{full_signal_text}
---
Quyết định được trích xuất (SELL, BUY, hoặc HOLD):"""


    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        context_vars = {"full_signal_text": full_signal}

        current_prompt_structure = {
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_template,
            # "few_shot_examples_string": self.few_shot_examples_string
        }
        full_prompt = format_prompt(current_prompt_structure, context_vars)

        if not full_prompt.strip():
            logger.error("SignalProcessor: Generated prompt is empty. Check prompt_config.")
            return "HOLD" # Fallback on empty prompt

        logger.info(f"SignalProcessor: Calling LLM. Model: {self.model_name}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")

        extracted_decision = self.llm_client.generate_text(
            prompt=full_prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ).strip().upper()

        # Validate the output to ensure it's one of the expected decisions
        valid_decisions = ["BUY", "SELL", "HOLD"]
        if extracted_decision not in valid_decisions:
            # Fallback or logging if the output is not as expected
            # For now, let's log a warning and return HOLD as a safe default
            # In a real scenario, this might need more robust error handling or retrying with a modified prompt.
            from tradingagents.llm_clients.base_client import logger # get logger
            logger.warning(f"SignalProcessor received an unexpected decision: '{extracted_decision}'. Full signal: '{full_signal[:100]}...'. Defaulting to HOLD.")
            return "HOLD"

        return extracted_decision
