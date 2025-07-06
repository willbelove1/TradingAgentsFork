import abc
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

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

class LLMKeyCycleError(LLMTransientError):
    """
    Special transient error used by multi-key clients to signal that the current key failed
    and the retry mechanism should cycle to the next key and try again.
    """
    def __init__(self, message, failed_key: Optional[str] = None, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.failed_key = failed_key
        self.original_exception = original_exception


# --- Default Retry Configuration ---
DEFAULT_RETRY_EXCEPTIONS: Tuple[type[Exception], ...] = (
    LLMTransientError, # Includes RateLimit, ServiceUnavailable, and KeyCycleError by inheritance
)

def log_retry_attempt(retry_state: Any) -> None:
    """Helper for logging before sleep, compatible with tenacity's before_sleep argument."""
    exc_info = retry_state.outcome.exception()
    failed_key_info = ""
    if isinstance(exc_info, LLMKeyCycleError) and exc_info.failed_key:
        failed_key_info = f" (Key: {exc_info.failed_key})"

    logger.warning(
        f"Retrying LLM API call: {retry_state.fn.__name__ if retry_state.fn else 'N/A'}{failed_key_info} "
        f"due to {exc_info!r}, "
        f"attempt number {retry_state.attempt_number} after {retry_state.seconds_since_start:.2f}s. "
        f"Waiting {retry_state.next_action.sleep:.2f}s before next attempt."
    )

class BaseLLMClient(abc.ABC):
    """
    Abstract base class for LLM clients with built-in retry logic.
    Defines a standardized interface for interacting with different LLM providers.
    """

    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_MIN_WAIT_SECONDS = 2
    DEFAULT_RETRY_MAX_WAIT_SECONDS = 10

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        self.api_key = api_key # For single API key clients. Multi-key clients will handle keys differently.
        self.config = config if config else {}

        # Default models for this client instance
        self.model_name = self.config.get("model") or self.config.get("default_text_model")
        self.embedding_model_name = self.config.get("embedding_model") or self.config.get("default_embedding_model")

        # Retry settings for this client instance
        self.retry_attempts = self.config.get("retry_attempts", self.DEFAULT_RETRY_ATTEMPTS)
        self.retry_min_wait = self.config.get("retry_min_wait_seconds", self.DEFAULT_RETRY_MIN_WAIT_SECONDS)
        self.retry_max_wait = self.config.get("retry_max_wait_seconds", self.DEFAULT_RETRY_MAX_WAIT_SECONDS)

        # Log warnings if default models are not found, as they are often essential.
        if not self.model_name and self.__class__._generate_text_impl != BaseLLMClient._generate_text_impl: # only if subclass implements it
            logger.warning(f"{self.__class__.__name__}: Default text model ('model' or 'default_text_model') not found in config.")
        if not self.embedding_model_name and self.__class__._get_embedding_impl != BaseLLMClient._get_embedding_impl: # only if subclass implements it
            logger.debug(f"{self.__class__.__name__}: Default embedding model ('embedding_model' or 'default_embedding_model') not found in config.")

    @abc.abstractmethod
    def _generate_text_impl(self, prompt: str, temperature: float, max_tokens: Optional[int], model: str, current_api_key_for_request: Optional[str]) -> str:
        pass

    @abc.abstractmethod
    def _get_embedding_impl(self, text: str, model: str, current_api_key_for_request: Optional[str]) -> List[float]:
        pass

    # @abc.abstractmethod
    # def _chat_impl(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int], model: str, current_api_key_for_request: Optional[str]) -> Dict:
    #    pass

    def _get_retry_decorator(self) -> Any:
        """Constructs a tenacity retry decorator based on client's retry settings."""
        return retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=self.retry_min_wait, max=self.retry_max_wait),
            retry=retry_if_exception_type(DEFAULT_RETRY_EXCEPTIONS),
            before_sleep=log_retry_attempt,
            reraise=True # Re-raise the last exception if all retries fail
        )

    def _get_current_api_key_for_request(self) -> Optional[str]:
        """
        Placeholder for subclasses (like multi-key clients) to override.
        Base implementation returns the single self.api_key.
        Multi-key clients will implement cycling logic here.
        This key is primarily for logging and for SDK configuration if needed per call.
        """
        return self.api_key

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = 1500, model: Optional[str] = None) -> str:
        start_time = time.time()
        effective_model = model or self.model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for generate_text and no default text model configured.")

        api_key_for_this_call = None # Will be set inside the retried function if multi-key

        # Define the function to be retried
        def _try_generate():
            nonlocal api_key_for_this_call # Allow modification of outer scope variable
            # For multi-key clients, _get_current_api_key_for_request will handle cycling & configuration.
            # For single-key, it just returns self.api_key (or the one from env).
            api_key_for_this_call = self._get_current_api_key_for_request()
            logger.debug(f"Attempting generate_text with model {effective_model} (Key: {'HIDDEN' if api_key_for_this_call else 'None'}). Prompt: \"{prompt[:100]}...\"")
            return self._generate_text_impl(prompt, temperature, max_tokens, effective_model, api_key_for_this_call)

        try:
            decorated_try_generate = self._get_retry_decorator()(_try_generate)
            result = decorated_try_generate()
            return result
        except LLMBaseException: # Re-raise our custom errors directly
            raise
        except Exception as e: # Wrap other unexpected errors
            logger.error(f"Unexpected, non-LLMBaseException error during generate_text with model {effective_model}: {e.__class__.__name__} - {e}")
            self._handle_api_error(e, operation_name="generate_text", model_name=effective_model, api_key_used=api_key_for_this_call) # Will re-raise
            return "" # Should not be reached if _handle_api_error re-raises
        finally:
            end_time = time.time()
            key_info = f"(Key: {'HIDDEN' if api_key_for_this_call else 'None'})" if api_key_for_this_call is not None else "" # Check if it was set
            logger.info(f"generate_text call for model {effective_model} {key_info} completed. Duration: {end_time - start_time:.2f}s")


    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        start_time = time.time()
        effective_model = model or self.embedding_model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for get_embedding and no default embedding model configured.")

        api_key_for_this_call = None

        def _try_embed():
            nonlocal api_key_for_this_call
            api_key_for_this_call = self._get_current_api_key_for_request()
            logger.debug(f"Attempting get_embedding for model {effective_model} (Key: {'HIDDEN' if api_key_for_this_call else 'None'}). Text: \"{text[:100]}...\"")
            return self._get_embedding_impl(text, effective_model, api_key_for_this_call)

        try:
            decorated_try_embed = self._get_retry_decorator()(_try_embed)
            result = decorated_try_embed()
            return result
        except LLMBaseException:
            raise
        except Exception as e:
            logger.error(f"Unexpected, non-LLMBaseException error during get_embedding with model {effective_model}: {e.__class__.__name__} - {e}")
            self._handle_api_error(e, operation_name="get_embedding", model_name=effective_model, api_key_used=api_key_for_this_call)
            return [] # Should not be reached
        finally:
            end_time = time.time()
            key_info = f"(Key: {'HIDDEN' if api_key_for_this_call else 'None'})" if api_key_for_this_call is not None else ""
            logger.info(f"get_embedding call for model {effective_model} {key_info} completed. Duration: {end_time - start_time:.2f}s")

    # def chat(...) would follow a similar pattern

    def _handle_api_error(self, error: Exception, operation_name: str, model_name: str, api_key_used: Optional[str]):
        key_info = f"(Key: {'HIDDEN' if api_key_used else 'None'})" if api_key_used is not None else ""
        logger.error(f"API Error during {operation_name} with model {model_name} {key_info}: {error.__class__.__name__} - {error}")
        if not isinstance(error, LLMBaseException):
            # Wrap unknown errors into a generic LLMBaseException if not already one of our custom types.
            # This helps ensure that callers can expect LLMBaseException or its children.
            raise LLMBaseException(f"Unhandled error during {operation_name}: {error}") from error
        raise error # Re-raise the (potentially wrapped) error

    def _log_usage(self, model_name: str,
                   prompt_tokens: Optional[int] = None,
                   completion_tokens: Optional[int] = None,
                   total_tokens: Optional[int] = None,
                   api_key_used: Optional[str] = None):
        usage_parts = [f"Model: {model_name}"]
        if api_key_used: # In a multi-key setup, knowing which key was used for a successful call is useful
            usage_parts.append(f"Key: {'HIDDEN'}") # Don't log actual key value for security
        if prompt_tokens is not None: usage_parts.append(f"PromptTokens: {prompt_tokens}")
        if completion_tokens is not None: usage_parts.append(f"CompletionTokens: {completion_tokens}")
        if total_tokens is not None: usage_parts.append(f"TotalTokens: {total_tokens}")

        if len(usage_parts) > (2 if api_key_used else 1) : # Log if more than just model/key
            logger.info(f"TokenUsage - {', '.join(usage_parts)}")
        else:
            logger.debug(f"Token usage details not available for this call with {model_name}.")
