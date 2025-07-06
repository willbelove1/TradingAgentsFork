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

import csv
import datetime
import os
from tradingagents.config.pricing_loader import load_llm_pricing, get_model_cost


    def _get_current_api_key_for_request(self) -> Optional[str]:
        """
        Placeholder for subclasses (like multi-key clients) to override.
        Base implementation returns the single self.api_key.
        Multi-key clients will implement cycling logic here.
        This key is primarily for logging and for SDK configuration if needed per call.
        """
        # For single-key clients, self.api_key might come from direct init, config, or env.
        # If list of keys is in self.config['api_keys'], multi-key client should use that.
        # This base method assumes a single key context; multi-key clients override it.
        if isinstance(self.config.get("api_keys"), list) and self.config["api_keys"]:
             # This case should ideally be handled by a multi-key subclass overriding this method.
             # For BaseLLMClient itself, if it encounters a list, it's ambiguous which one to use.
             # However, GeminiClient (multi-key) overrides this. OpenAIClient (single-key) does not.
             # If this BaseLLMClient is used directly with a list of keys in config (which it shouldn't),
             # it would be an issue. So, this is mostly for single key scenarios or as a fallback.
            logger.debug("BaseLLMClient._get_current_api_key_for_request found list of keys in config, but not a multi-key client. Using constructor api_key or first from list if available.")
            return self.api_key or self.config["api_keys"][0]
        return self.api_key or os.getenv("API_KEY_GENERIC_ENV_VAR") # Fallback, specific clients handle better

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = 1500, model: Optional[str] = None,
                      agent_name: Optional[str] = None) -> str: # Added agent_name
        start_time = time.time()
        effective_model = model or self.model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for generate_text and no default text model configured.")

        api_key_for_this_call = None
        result = "" # Ensure result is defined

        def _try_generate():
            nonlocal api_key_for_this_call, result # Allow modification
            api_key_for_this_call = self._get_current_api_key_for_request()
            # Log actual key hash/id if needed for tracking, not the full key. For now, 'HIDDEN'.
            key_display = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env"
            logger.debug(f"Attempting generate_text with model {effective_model} (Key: {key_display}). Prompt: \"{prompt[:100]}...\"")
            result = self._generate_text_impl(prompt, temperature, max_tokens, effective_model, api_key_for_this_call)
            return result # Tenacity expects the retried function to return the result

        try:
            decorated_try_generate = self._get_retry_decorator()(_try_generate)
            result = decorated_try_generate() # This will assign to outer 'result' if successful
            return result
        except LLMBaseException:
            raise
        except Exception as e:
            logger.error(f"Unexpected, non-LLMBaseException error during generate_text with model {effective_model}: {e.__class__.__name__} - {e}")
            self._handle_api_error(e, operation_name="generate_text", model_name=effective_model, api_key_used=api_key_for_this_call)
            return "" # Should not be reached if _handle_api_error re-raises (which it does)
        finally:
            end_time = time.time()
            duration_seconds = end_time - start_time
            # _log_usage is now called by the _impl methods in subclasses after successful call,
            # because they have access to token counts.
            # However, the overall duration and key used for the final successful/failed attempt is logged here.
            key_display_final = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env"
            logger.info(f"generate_text call for model {effective_model} (Key: {key_display_final}) by agent '{agent_name or 'Unknown'}' completed. Duration: {duration_seconds:.2f}s")
            # If _log_usage needs to be called from here with final attempt data (if _impl doesn't have all info)
            # This would be if _impl doesn't call _log_usage itself.
            # For now, assume _impl calls _log_usage with token counts.
            # If result is successfully obtained, _impl should have called _log_usage.
            # If an error occurred and was re-raised, then _log_usage for tokens might not have been called.

    def get_embedding(self, text: str, model: Optional[str] = None,
                      agent_name: Optional[str] = None) -> List[float]: # Added agent_name
        start_time = time.time()
        effective_model = model or self.embedding_model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for get_embedding and no default embedding model configured.")

        api_key_for_this_call = None
        result_embedding: List[float] = []

        def _try_embed():
            nonlocal api_key_for_this_call, result_embedding
            api_key_for_this_call = self._get_current_api_key_for_request()
            key_display = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env"
            logger.debug(f"Attempting get_embedding for model {effective_model} (Key: {key_display}). Text: \"{text[:100]}...\"")
            result_embedding = self._get_embedding_impl(text, effective_model, api_key_for_this_call)
            return result_embedding

        try:
            decorated_try_embed = self._get_retry_decorator()(_try_embed)
            result_embedding = decorated_try_embed()
            return result_embedding
        except LLMBaseException:
            raise
        except Exception as e:
            logger.error(f"Unexpected, non-LLMBaseException error during get_embedding with model {effective_model}: {e.__class__.__name__} - {e}")
            self._handle_api_error(e, operation_name="get_embedding", model_name=effective_model, api_key_used=api_key_for_this_call)
            return []
        finally:
            end_time = time.time()
            duration_seconds = end_time - start_time
            key_display_final = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env"
            logger.info(f"get_embedding call for model {effective_model} (Key: {key_display_final}) by agent '{agent_name or 'Unknown'}' completed. Duration: {duration_seconds:.2f}s")
            # As with generate_text, assume _get_embedding_impl calls _log_usage if it has token data.

    def _handle_api_error(self, error: Exception, operation_name: str, model_name: str, api_key_used: Optional[str]):
        key_display = f"...{api_key_used[-4:]}" if api_key_used and len(api_key_used) > 4 else "Default/Env"
        logger.error(f"API Error during {operation_name} with model {model_name} (Key: {key_display}): {error.__class__.__name__} - {error}")
        if not isinstance(error, LLMBaseException):
            raise LLMBaseException(f"Unhandled error during {operation_name}: {error}") from error
        raise error

    def _log_usage(self, model_name: str,
                   duration_seconds: float, # Added duration
                   prompt_tokens: Optional[int] = None,
                   completion_tokens: Optional[int] = None,
                   total_tokens: Optional[int] = None,
                   api_key_used: Optional[str] = None,
                   agent_name: Optional[str] = None, # Added agent_name
                   success: bool = True): # Added success status

        # Basic console logging (existing behavior, slightly enhanced)
        log_message_parts = [f"Model: {model_name}"]
        key_display = f"...{api_key_used[-4:]}" if api_key_used and len(api_key_used) > 4 else "Default/Env"
        log_message_parts.append(f"Key: {key_display}")
        if agent_name: log_message_parts.append(f"Agent: {agent_name}")
        log_message_parts.append(f"Duration: {duration_seconds:.2f}s")
        if prompt_tokens is not None: log_message_parts.append(f"PromptTokens: {prompt_tokens}")
        if completion_tokens is not None: log_message_parts.append(f"CompletionTokens: {completion_tokens}")
        if total_tokens is not None: log_message_parts.append(f"TotalTokens: {total_tokens}")
        log_message_parts.append(f"Success: {success}")
        logger.info(f"LLMCallLog - {', '.join(log_message_parts)}")

        # CSV Logging
        log_dir = "logs"
        csv_filepath = os.path.join(log_dir, "llm_usage.csv")

        try:
            os.makedirs(log_dir, exist_ok=True)

            # Cost Estimation
            estimated_cost_usd = 0.0
            pricing_data = load_llm_pricing() # Loads from config/llm_pricing.yaml
            # Provider name needs to be determined. Assuming it's part of self.config or class name.
            # This is a bit tricky as BaseLLMClient doesn't know its concrete provider name easily.
            # Let's assume self.config['llm_provider'] holds the provider name (e.g. "google", "openai")
            provider_name = self.config.get('llm_provider', 'unknown_provider').lower()

            cost_info = get_model_cost(provider_name, model_name, pricing_config=pricing_data)

            if cost_info:
                input_cost = cost_info.get('input_cost_per_million_tokens', 0.0)
                output_cost = cost_info.get('output_cost_per_million_tokens', 0.0)

                if prompt_tokens is not None:
                    estimated_cost_usd += (prompt_tokens / 1_000_000) * input_cost
                if completion_tokens is not None: # Text generation
                    estimated_cost_usd += (completion_tokens / 1_000_000) * output_cost
                elif total_tokens is not None and prompt_tokens is None: # Embeddings might only report total_tokens
                    # Assume total_tokens for embeddings are input tokens for costing
                    estimated_cost_usd += (total_tokens / 1_000_000) * input_cost
            else:
                logger.warning(f"No pricing info found for {provider_name}/{model_name}. Cost will be 0.")

            file_exists = os.path.isfile(csv_filepath)
            with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'provider_name', 'api_key_identifier', 'agent_name',
                    'model_name', 'prompt_tokens', 'completion_tokens', 'total_tokens',
                    'duration_seconds', 'estimated_cost_usd', 'success'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(csv_filepath) == 0:
                    writer.writeheader()

                api_key_identifier = f"...{api_key_used[-4:]}" if api_key_used and len(api_key_used) > 3 else "default"

                writer.writerow({
                    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'provider_name': provider_name,
                    'api_key_identifier': api_key_identifier,
                    'agent_name': agent_name or "Unknown",
                    'model_name': model_name,
                    'prompt_tokens': prompt_tokens if prompt_tokens is not None else 0,
                    'completion_tokens': completion_tokens if completion_tokens is not None else 0,
                    'total_tokens': total_tokens if total_tokens is not None else (prompt_tokens or 0) + (completion_tokens or 0),
                    'duration_seconds': round(duration_seconds, 3),
                    'estimated_cost_usd': round(estimated_cost_usd, 8),
                    'success': success
                })
        except Exception as e:
            logger.error(f"Failed to write to LLM usage CSV log: {e}", exc_info=True)
