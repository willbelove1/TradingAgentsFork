import abc
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type #, before_sleep_log

# Configure basic logging for the module
# It's generally better to configure basicConfig at the application entry point.
# Libraries should just obtain loggers.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger name will be 'tradingagents.llm_clients.base_client'

# --- Custom Exceptions for Retry Logic and Error Handling ---
class LLMBaseException(Exception):
    """Base exception for all custom LLM client errors."""
    pass

class LLMTransientError(LLMBaseException):
    """Base class for LLM client errors that might be transient and retriable."""
    pass

class LLMRateLimitError(LLMTransientError):
    """Specific error for rate limiting."""
    pass

class LLMServiceUnavailableError(LLMTransientError):
    """Specific error for service unavailable or temporary server issues."""
    pass

class LLMAuthenticationError(LLMBaseException):
    """Specific error for authentication failures (usually not retriable with same credentials)."""
    pass

class LLMConfigurationError(LLMBaseException):
    """Specific error for configuration problems."""
    pass

class LLMResponseError(LLMBaseException):
    """Error related to the LLM's response (e.g., empty, malformed, safety blocked)."""
    pass


# --- Default Retry Configuration ---
# Define common transient exceptions from popular HTTP libraries if used directly by SDKs,
# or rely on SDKs to raise their own specific transient errors.
# For now, we'll primarily rely on our custom LLMTransientError and its children.
# SDK-specific transient errors can be caught by client implementations and re-raised as LLMTransientError.
DEFAULT_RETRY_EXCEPTIONS: Tuple[type[Exception], ...] = (
    LLMTransientError, # Includes RateLimit and ServiceUnavailable by inheritance
    # Example: Add requests.exceptions.Timeout, requests.exceptions.ConnectionError if making direct HTTP calls
)

# Helper for logging before sleep, compatible with tenacity's before_sleep argument
def log_retry_attempt(retry_state: Any) -> None:
    logger.warning(
        f"Retrying LLM API call: {retry_state.fn.__name__ if retry_state.fn else 'N/A'} "
        f"due to {retry_state.outcome.exception()!r}, "
        f"attempt number {retry_state.attempt_number} after {retry_state.seconds_since_start:.2f}s. "
        f"Waiting {retry_state.next_action.sleep:.2f}s before next attempt."
    )

DEFAULT_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(DEFAULT_RETRY_EXCEPTIONS),
    before_sleep=log_retry_attempt # Use the helper for logging
    # reraise=True # Default is False, meaning tenacity returns the result of the last attempt. If True, it re-raises the last exception. We want to handle this in client methods.
)

class BaseLLMClient(abc.ABC):
    """
    Abstract base class for LLM clients with built-in retry logic.
    Defines a standardized interface for interacting with different LLM providers.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the client.
        Args:
            api_key (str, optional): The API key for the LLM provider.
            config (Dict, optional): Additional configuration parameters.
                                     Expected: 'model' (default text model), 'embedding_model'.
        """
        self.api_key = api_key
        self.config = config if config else {}
        self.model_name = self.config.get("model")
        self.embedding_model_name = self.config.get("embedding_model")
        if not self.model_name:
            logger.warning(f"{self.__class__.__name__}: Default text model ('model') not found in config.")
        if not self.embedding_model_name:
            logger.warning(f"{self.__class__.__name__}: Default embedding model ('embedding_model') not found in config.")


    @abc.abstractmethod
    def _generate_text_impl(self, prompt: str, temperature: float, max_tokens: Optional[int], model: str) -> str:
        """Implementation specific to the LLM provider for generating text."""
        pass

    @abc.abstractmethod
    def _get_embedding_impl(self, text: str, model: str) -> List[float]:
        """Implementation specific to the LLM provider for generating embeddings."""
        pass

    # Optional chat method
    # @abc.abstractmethod
    # def _chat_impl(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int], model: str) -> Dict:
    #    """Implementation specific to the LLM provider for chat."""
    #    pass

    # Decorated methods call the internal implementations
    @DEFAULT_RETRY_DECORATOR
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = 1500, model: Optional[str] = None) -> str:
        """
        Generates text based on a given prompt with retry logic.
        """
        start_time = time.time()
        effective_model = model or self.model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for generate_text and no default text model configured.")

        logger.debug(f"Attempting generate_text with model {effective_model}. Prompt: \"{prompt[:100]}...\"")
        try:
            result = self._generate_text_impl(prompt, temperature, max_tokens, effective_model)
            # _log_usage might be called within _generate_text_impl by subclasses if they have token info
            return result
        except Exception as e: # Catch-all for non-retryable or SDK-specific errors not mapped to custom ones
            self._handle_api_error(e, operation_name="generate_text", model_name=effective_model) # Will re-raise
        finally: # This will always execute, even if retry happens or an error is re-raised
            end_time = time.time()
            logger.info(f"generate_text call completed for model {effective_model}. Duration: {end_time - start_time:.2f}s")
        return "" # Should not be reached if _handle_api_error re-raises

    @DEFAULT_RETRY_DECORATOR
    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generates an embedding for a given text with retry logic.
        """
        start_time = time.time()
        effective_model = model or self.embedding_model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for get_embedding and no default embedding model configured.")

        logger.debug(f"Attempting get_embedding for model {effective_model}. Text: \"{text[:100]}...\"")
        try:
            result = self._get_embedding_impl(text, effective_model)
            return result
        except Exception as e:
            self._handle_api_error(e, operation_name="get_embedding", model_name=effective_model)
        finally:
            end_time = time.time()
            logger.info(f"get_embedding call completed for model {effective_model}. Duration: {end_time - start_time:.2f}s")
        return [] # Should not be reached

    # @DEFAULT_RETRY_DECORATOR
    # def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = 1500, model: Optional[str] = None) -> Dict:
    #     start_time = time.time()
    #     effective_model = model or self.model_name
    #     if not effective_model:
    #         raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for chat and no default text model configured.")
    #     logger.debug(f"Attempting chat with model {effective_model}.")
    #     try:
    #         result = self._chat_impl(messages, temperature, max_tokens, effective_model)
    #         return result
    #     except Exception as e:
    #         self._handle_api_error(e, operation_name="chat", model_name=effective_model)
    #     finally:
    #         end_time = time.time()
    #         logger.info(f"chat call completed for model {effective_model}. Duration: {end_time - start_time:.2f}s")
    #     return {} # Should not be reached

    def _handle_api_error(self, error: Exception, operation_name: str, model_name: str):
        """
        Handles common API errors. Subclasses should call this or implement more specific handling,
        and then re-raise either the original error or a custom LLMBaseException.
        This base implementation will re-raise the original error if not mapped.
        Subclasses are responsible for mapping SDK-specific errors to custom LLMTransientError etc.
        """
        logger.error(f"API Error during {operation_name} with model {model_name}: {error.__class__.__name__} - {error}")

        # Example: if a subclass identified a specific SDK error as transient:
        # if isinstance(error, SomeSDKTransientError):
        #     raise LLMTransientError(f"Transient error from SDK: {error}") from error
        # if isinstance(error, SomeSDKRateLimitError):
        #     raise LLMRateLimitError(f"Rate limit from SDK: {error}") from error

        if not isinstance(error, LLMBaseException): # If it's not already one of our custom errors
            # General fallback: wrap unknown errors if needed, or just re-raise
            # For now, just re-raise to ensure it's not silently swallowed if not mapped by subclass.
            pass # Subclass should handle mapping or re-raising

        raise error


    def _log_usage(self, model_name: str, prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None, total_tokens: Optional[int] = None):
        """
        Logs token usage if available from the API response. Called by subclasses.
        """
        usage_parts = [f"Model: {model_name}"]
        if prompt_tokens is not None:
            usage_parts.append(f"Prompt Tokens: {prompt_tokens}")
        if completion_tokens is not None:
            usage_parts.append(f"Completion Tokens: {completion_tokens}")
        if total_tokens is not None:
            usage_parts.append(f"Total Tokens: {total_tokens}")

        if len(usage_parts) > 1: # Only log if there's more than just model name
            logger.info(f"Token Usage - {', '.join(usage_parts)}")
        else:
            logger.debug(f"Token usage details not available for this call with {model_name}.")
