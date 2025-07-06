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
    LLMTransientError,
)

def log_retry_attempt(retry_state: Any) -> None:
    """Helper for logging before sleep, compatible with tenacity's before_sleep argument."""
    exc_info = retry_state.outcome.exception()
    failed_key_info = ""
    if isinstance(exc_info, LLMKeyCycleError) and exc_info.failed_key:
        # Only show last 4 chars of key for security, if key is long enough
        key_display = f"...{exc_info.failed_key[-4:]}" if len(exc_info.failed_key) > 4 else exc_info.failed_key
        failed_key_info = f" (Key: {key_display})"

    logger.warning(
        f"Retrying LLM API call: {retry_state.fn.__name__ if retry_state.fn else 'N/A'}{failed_key_info} "
        f"due to {exc_info!r}, "
        f"attempt number {retry_state.attempt_number} after {retry_state.seconds_since_start:.2f}s. "
        f"Waiting {retry_state.next_action.sleep:.2f}s before next attempt."
    )

class BaseLLMClient(abc.ABC):
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_MIN_WAIT_SECONDS = 2
    DEFAULT_RETRY_MAX_WAIT_SECONDS = 10

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        self.api_key = api_key
        self.config = config if config else {}

        self.model_name = self.config.get("model") or self.config.get("default_text_model")
        self.embedding_model_name = self.config.get("embedding_model") or self.config.get("default_embedding_model")

        self.retry_attempts = self.config.get("retry_attempts", self.DEFAULT_RETRY_ATTEMPTS)
        self.retry_min_wait = self.config.get("retry_min_wait_seconds", self.DEFAULT_RETRY_MIN_WAIT_SECONDS)
        self.retry_max_wait = self.config.get("retry_max_wait_seconds", self.DEFAULT_RETRY_MAX_WAIT_SECONDS)

        if not self.model_name and hasattr(self.__class__, '_generate_text_impl') and self.__class__._generate_text_impl != BaseLLMClient._generate_text_impl:
            logger.warning(f"{self.__class__.__name__}: Default text model ('model' or 'default_text_model') not found in config.")
        if not self.embedding_model_name and hasattr(self.__class__, '_get_embedding_impl') and self.__class__._get_embedding_impl != BaseLLMClient._get_embedding_impl:
            logger.debug(f"{self.__class__.__name__}: Default embedding model ('embedding_model' or 'default_embedding_model') not found in config.")

    @abc.abstractmethod
    def _generate_text_impl(self, prompt: str, temperature: float, max_tokens: Optional[int], model: str, current_api_key_for_request: Optional[str], agent_name: Optional[str]) -> str:
        pass

    @abc.abstractmethod
    def _get_embedding_impl(self, text: str, model: str, current_api_key_for_request: Optional[str], agent_name: Optional[str]) -> List[float]:
        pass

    def _get_retry_decorator(self) -> Any:
        return retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=self.retry_min_wait, max=self.retry_max_wait),
            retry=retry_if_exception_type(DEFAULT_RETRY_EXCEPTIONS),
            before_sleep=log_retry_attempt,
            reraise=True
        )

    def _get_current_api_key_for_request(self) -> Optional[str]:
        # This base method assumes a single key context. Multi-key clients (GeminiClient) override this.
        # It prioritizes the key given at construction, then from config (if 'api_key' not 'api_keys'), then env.
        if self.api_key: return self.api_key
        if isinstance(self.config.get("api_key"), str): return self.config.get("api_key") # Single key in config
        # Fallback to generic env var, specific clients should handle their specific env vars (e.g. GOOGLE_API_KEY)
        # This is less likely to be used if clients are initialized properly by the factory.
        return os.getenv("TRADINGAGENTS_API_KEY")

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = 1500, model: Optional[str] = None,
                      agent_name: Optional[str] = None) -> str:
        start_time = time.time()
        effective_model = model or self.model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for generate_text and no default text model configured.")

        api_key_for_this_call = None
        result_text = ""
        success_flag = False

        def _try_generate():
            nonlocal api_key_for_this_call, result_text
            api_key_for_this_call = self._get_current_api_key_for_request()
            key_display = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env"
            logger.debug(f"Attempting generate_text with model {effective_model} (Key: {key_display}). Agent: {agent_name or 'Unknown'}. Prompt: \"{prompt[:100]}...\"")
            # The _impl method is now responsible for calling _log_usage on success/failure of SDK call
            result_text = self._generate_text_impl(prompt, temperature, max_tokens, effective_model, api_key_for_this_call, agent_name)
            return result_text

        try:
            decorated_try_generate = self._get_retry_decorator()(_try_generate)
            result_text = decorated_try_generate()
            success_flag = True # If decorated_try_generate succeeds
            return result_text
        except LLMBaseException as e:
            logger.error(f"LLM Error (final) in generate_text by agent '{agent_name or 'Unknown'}' for model {effective_model} (Key: ...{api_key_for_this_call[-4:] if api_key_for_this_call and len(api_key_for_this_call) > 4 else 'N/A'}): {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected non-LLM error (final) in generate_text by agent '{agent_name or 'Unknown'}' for model {effective_model} (Key: ...{api_key_for_this_call[-4:] if api_key_for_this_call and len(api_key_for_this_call) > 4 else 'N/A'}): {e.__class__.__name__} - {e}")
            self._handle_api_error(e, operation_name="generate_text", model_name=effective_model, api_key_used=api_key_for_this_call, agent_name=agent_name, success_status=False)
            return ""
        finally:
            # Overall duration logging (covers all retries)
            end_time = time.time()
            duration_seconds = end_time - start_time
            key_display_final = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env" # Key used in last attempt
            logger.info(f"generate_text call for model {effective_model} (Key: {key_display_final}) by agent '{agent_name or 'Unknown'}' finished. Overall Duration: {duration_seconds:.2f}s. Success: {success_flag}")
            # Note: Detailed token logging (and its associated cost) is now expected to be called
            # from within the _impl methods by subclasses, as they have direct access to the SDK response.
            # If an error occurs that prevents _impl from calling _log_usage (e.g. very early SDK error),
            # we might log a "failed attempt" entry here without token counts if needed, but _log_usage in _impl handles most cases.


    def get_embedding(self, text: str, model: Optional[str] = None,
                      agent_name: Optional[str] = None) -> List[float]:
        start_time = time.time()
        effective_model = model or self.embedding_model_name
        if not effective_model:
            raise LLMConfigurationError(f"{self.__class__.__name__}: No model specified for get_embedding and no default embedding model configured.")

        api_key_for_this_call = None
        result_embedding: List[float] = []
        success_flag = False

        def _try_embed():
            nonlocal api_key_for_this_call, result_embedding
            api_key_for_this_call = self._get_current_api_key_for_request()
            key_display = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env"
            logger.debug(f"Attempting get_embedding for model {effective_model} (Key: {key_display}). Agent: {agent_name or 'Unknown'}. Text: \"{text[:100]}...\"")
            result_embedding = self._get_embedding_impl(text, effective_model, api_key_for_this_call, agent_name)
            return result_embedding

        try:
            decorated_try_embed = self._get_retry_decorator()(_try_embed)
            result_embedding = decorated_try_embed()
            success_flag = True
            return result_embedding
        except LLMBaseException:
            raise
        except Exception as e:
            logger.error(f"Unexpected, non-LLM error (final) in get_embedding by agent '{agent_name or 'Unknown'}' for model {effective_model} (Key: ...{api_key_for_this_call[-4:] if api_key_for_this_call and len(api_key_for_this_call) > 4 else 'N/A'}): {e.__class__.__name__} - {e}")
            self._handle_api_error(e, operation_name="get_embedding", model_name=effective_model, api_key_used=api_key_for_this_call, agent_name=agent_name, success_status=False)
            return []
        finally:
            end_time = time.time()
            duration_seconds = end_time - start_time
            key_display_final = f"...{api_key_for_this_call[-4:]}" if api_key_for_this_call and len(api_key_for_this_call) > 4 else "Default/Env"
            logger.info(f"get_embedding call for model {effective_model} (Key: {key_display_final}) by agent '{agent_name or 'Unknown'}' finished. Overall Duration: {duration_seconds:.2f}s. Success: {success_flag}")


    def _handle_api_error(self, error: Exception, operation_name: str, model_name: str, api_key_used: Optional[str], agent_name: Optional[str], success_status: bool): # Added agent_name, success_status
        key_display = f"...{api_key_used[-4:]}" if api_key_used and len(api_key_used) > 4 else "Default/Env"
        # Log the error with context (this is the final error after retries if any)
        logger.error(f"API Error (final) during {operation_name} with model {model_name} (Key: {key_display}) by agent '{agent_name or 'Unknown'}': {error.__class__.__name__} - {error}")

        # Log this failed call to CSV via _log_usage if it hasn't been logged by _impl already
        # This ensures failed attempts that don't even reach _impl's own _log_usage call (e.g. very early error) get some record.
        # However, _impl methods are now designed to call _log_usage with success=False.
        # This _handle_api_error is more for unexpected errors *after* tenacity or if _impl doesn't catch something.
        # For now, we rely on _impl to call _log_usage. If an error is caught here that bypassed _impl's logging,
        # it means it's likely an error in the retry logic itself or pre-impl call.

        if not isinstance(error, LLMBaseException):
            raise LLMBaseException(f"Unhandled error during {operation_name} by agent '{agent_name or 'Unknown'}': {error}") from error
        raise error

    def _log_usage(self, model_name: str,
                   duration_seconds: float,
                   prompt_tokens: Optional[int] = None,
                   completion_tokens: Optional[int] = None,
                   total_tokens: Optional[int] = None,
                   api_key_used: Optional[str] = None,
                   agent_name: Optional[str] = None,
                   success: bool = True):

        # Console Logging
        log_message_parts = [f"Model: {model_name}"]
        key_display = f"...{api_key_used[-4:]}" if api_key_used and len(api_key_used) > 4 else "Default/Env"
        log_message_parts.append(f"KeyUsed: {key_display}") # Changed label for clarity
        if agent_name: log_message_parts.append(f"Agent: {agent_name}")
        log_message_parts.append(f"SDKDuration: {duration_seconds:.2f}s") # Clarify this is SDK call duration
        if prompt_tokens is not None: log_message_parts.append(f"PromptTokens: {prompt_tokens}")
        if completion_tokens is not None: log_message_parts.append(f"CompletionTokens: {completion_tokens}")
        if total_tokens is not None: log_message_parts.append(f"TotalTokens: {total_tokens}")
        log_message_parts.append(f"CallSuccess: {success}")
        logger.info(f"LLMCallDetails - {', '.join(log_message_parts)}")

        # CSV Logging
        log_dir = self.config.get("llm_usage_log_dir", "logs") # Get from config or default
        csv_filename = self.config.get("llm_usage_csv_name", "llm_usage.csv")
        csv_filepath = os.path.join(log_dir, csv_filename)

        try:
            os.makedirs(log_dir, exist_ok=True)

            estimated_cost_usd = 0.0
            # Pricing config is loaded once and cached by the loader if not passed explicitly
            pricing_data = load_llm_pricing(self.config.get("llm_pricing_filepath"))
            provider_name = self.config.get('llm_provider', 'unknown_provider').lower()
            cost_info = get_model_cost(provider_name, model_name, pricing_config=pricing_data)

            if cost_info:
                input_cost = cost_info.get('input_cost_per_million_tokens', 0.0)
                output_cost = cost_info.get('output_cost_per_million_tokens', 0.0)
                p_tokens = prompt_tokens or 0
                c_tokens = completion_tokens or 0
                t_tokens = total_tokens or (p_tokens + c_tokens)

                if provider_name == "openai" or "gpt" in model_name.lower() or (provider_name=="google" and "gemini" in model_name.lower() and not model_name.startswith("models/embedding")): # Text models
                    estimated_cost_usd = (p_tokens / 1_000_000 * input_cost) + \
                                         (c_tokens / 1_000_000 * output_cost)
                elif "embedding" in model_name.lower(): # Embedding models
                     # Assume total_tokens for embeddings are input tokens for costing if only total_tokens is available
                    costable_tokens = p_tokens if p_tokens > 0 else t_tokens
                    estimated_cost_usd = (costable_tokens / 1_000_000) * input_cost
            else:
                logger.debug(f"No pricing info found for {provider_name}/{model_name}. Cost will be 0 for this call.")

            file_exists = os.path.isfile(csv_filepath)
            with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'provider_name', 'api_key_identifier', 'agent_name',
                    'model_name', 'prompt_tokens', 'completion_tokens', 'total_tokens',
                    'sdk_duration_seconds', 'estimated_cost_usd', 'success'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(csv_filepath) == 0:
                    writer.writeheader()

                api_key_identifier = f"...{api_key_used[-4:]}" if api_key_used and len(api_key_used) > 3 else "default/env"

                writer.writerow({
                    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'provider_name': provider_name,
                    'api_key_identifier': api_key_identifier,
                    'agent_name': agent_name or "Unknown",
                    'model_name': model_name,
                    'prompt_tokens': prompt_tokens if prompt_tokens is not None else 0,
                    'completion_tokens': completion_tokens if completion_tokens is not None else 0,
                    'total_tokens': total_tokens if total_tokens is not None else ((prompt_tokens or 0) + (completion_tokens or 0)),
                    'sdk_duration_seconds': round(duration_seconds, 3),
                    'estimated_cost_usd': round(estimated_cost_usd, 8),
                    'success': success
                })
        except Exception as e:
            logger.error(f"Failed to write to LLM usage CSV log: {e}", exc_info=True)
