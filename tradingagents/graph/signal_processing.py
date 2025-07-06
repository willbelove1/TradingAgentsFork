# TradingAgents/graph/signal_processing.py

# from langchain_google_genai import ChatGoogleGenerativeAI # Remove direct ChatModel import
from tradingagents.llm_clients import BaseLLMClient # Import BaseLLMClient

class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, llm_client: BaseLLMClient): # Changed parameter type
        """Initialize with an LLM client for processing."""
        self.llm_client = llm_client # Store the client

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        system_prompt = "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information."

        # Construct a single prompt string for generate_text
        full_prompt = f"{system_prompt}\n\nFinancial Report/Signal:\n{full_signal}\n\nExtracted Decision (SELL, BUY, or HOLD):"

        # Using the llm_client's generate_text method
        # A lower temperature might be better for this kind of extraction task.
        # Max tokens can be small as we expect a single word.
        extracted_decision = self.llm_client.generate_text(
            prompt=full_prompt,
            temperature=0.1, # Low temperature for deterministic output
            max_tokens=10      # Small max_tokens for a single word
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
