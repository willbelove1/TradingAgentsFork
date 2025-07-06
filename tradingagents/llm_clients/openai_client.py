from openai import OpenAI as OpenAIClientSDK, APIError, RateLimitError, AuthenticationError, APIConnectionError, APITimeoutError, APIStatusError, InternalServerError
import os
import time # Not strictly needed here if super() handles timing, but good for consistency
import logging
from typing import List, Dict, Optional

from tradingagents.llm_clients.base_client import (
    BaseLLMClient, logger,
    LLMTransientError, LLMRateLimitError, LLMServiceUnavailableError,
    LLMAuthenticationError, LLMConfigurationError, LLMResponseError
)

class OpenAIClient(BaseLLMClient):
    """
    LLM Client for OpenAI models.
    """

    DEFAULT_TEXT_MODEL = "gpt-3.5-turbo"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002" # Or text-embedding-3-small, text-embedding-3-large

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initializes the OpenAI client.
        Args:
            api_key (str, optional): OpenAI API key. If None, attempts to use OPENAI_API_KEY environment variable.
            config (Dict, optional): Configuration dictionary. Expected keys:
                - 'model': Name of the default text generation model.
                - 'embedding_model': Name of the default embedding model.
                - 'base_url': Optional base URL for OpenAI compatible APIs (e.g., Ollama, OpenRouter).
        """
        super().__init__(api_key, config)

        resolved_api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        # For some compatible APIs (like local Ollama), API key might not be strictly needed or might be a placeholder.
        # However, the OpenAI SDK often expects it.
        if not resolved_api_key and not self.config.get("base_url"): # Only raise if no base_url for custom server
             # Attempt to load from .env if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv()
                resolved_api_key = os.getenv("OPENAI_API_KEY")
            except ImportError:
                logger.info("python-dotenv not installed, cannot load OPENAI_API_KEY from .env file.")

            if not resolved_api_key:
                logger.warning("OPENAI_API_KEY not found. This might be an issue unless using a local/custom base_url that doesn't require it.")
                # Not raising ValueError to allow keyless local server usage, but SDK might still complain.

        self.model_name = self.config.get("model", self.DEFAULT_TEXT_MODEL)
        self.embedding_model_name = self.config.get("embedding_model", self.DEFAULT_EMBEDDING_MODEL)
        self.base_url = self.config.get("base_url")

        try:
            self.sdk_client = OpenAIClientSDK(
                api_key=resolved_api_key,
                base_url=self.base_url
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI SDK client: {e}")
            self._handle_api_error(e) # This will re-raise

        logger.info(f"OpenAIClient initialized. Default text model: {self.model_name}, Default embedding model: {self.embedding_model_name}, Base URL: {self.base_url or 'Default OpenAI'}")

    def _generate_text_impl(self, prompt: str, temperature: float, max_tokens: Optional[int], model: str, current_api_key_for_request: Optional[str], agent_name: Optional[str] = None) -> str:
        current_model = model
        messages = [{"role": "user", "content": prompt}]

        sdk_call_start_time = time.time()
        try:
            response = self.sdk_client.chat.completions.create(
                model=current_model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            sdk_call_duration = time.time() - sdk_call_start_time
            generated_text = response.choices[0].message.content.strip()

            if response.usage:
                self._log_usage(
                    model_name=current_model, duration_seconds=sdk_call_duration,
                    prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens, api_key_used=current_api_key_for_request, agent_name=agent_name, success=True
                )
            else: # Should not happen with successful OpenAI calls, but good to log if it does
                self._log_usage(model_name=current_model, duration_seconds=sdk_call_duration, api_key_used=current_api_key_for_request, agent_name=agent_name, success=True, prompt_tokens=None)

            return generated_text
        except RateLimitError as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.warning(f"OpenAI API rate limit hit for model {current_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMRateLimitError(f"OpenAI API rate limit hit: {e}") from e # Retriable by Base
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.warning(f"OpenAI API transient error (connection, timeout, or 5xx) for model {current_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMServiceUnavailableError(f"OpenAI API transient error: {e}") from e # Retriable
        except APIStatusError as e: # Need to check status code for 5xx
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            if 500 <= e.status_code < 600:
                logger.warning(f"OpenAI API transient server error (5xx) for model {current_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
                raise LLMServiceUnavailableError(f"OpenAI API server error {e.status_code}: {e}") from e # Retriable
            else:
                logger.error(f"OpenAI API non-transient status error for model {current_model}: {e.status_code} - {e}")
                raise LLMResponseError(f"OpenAI API non-transient status error {e.status_code}: {e}") from e
        except AuthenticationError as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"OpenAI API authentication error for model {current_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMAuthenticationError(f"OpenAI API authentication error: {e}") from e # Not retriable by default
        except APIError as e: # Catch other APIErrors (e.g. InvalidRequestError)
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"OpenAI API error for model {current_model}: {e}")
            if hasattr(e, 'status_code') and e.status_code == 400:
                 raise LLMConfigurationError(f"OpenAI API Bad Request (check inputs/config): {e}") from e
            raise LLMResponseError(f"OpenAI API error: {e}") from e
        except Exception as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"Unexpected error during OpenAI text generation with model {current_model}: {e}")
            raise


    def _get_embedding_impl(self, text: str, model: str, current_api_key_for_request: Optional[str], agent_name: Optional[str] = None) -> List[float]:
        current_embedding_model = model

        if not text or not text.strip():
            logger.warning(f"Agent '{agent_name or 'Unknown'}': Attempted to get embedding for empty text with model {current_embedding_model}. Returning zero vector.")
            default_dim = 1536
            if "large" in current_embedding_model: default_dim = 3072
            # No API call, so no duration or token cost to log via _log_usage here for this specific case.
            return [0.0] * default_dim

        sdk_call_start_time = time.time()
        try:
            response = self.sdk_client.embeddings.create(
                input=[text.replace("\n", " ")], model=current_embedding_model
            )
            sdk_call_duration = time.time() - sdk_call_start_time
            embedding_vector = response.data[0].embedding

            if response.usage:
                self._log_usage(
                    model_name=current_embedding_model, duration_seconds=sdk_call_duration,
                    prompt_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens,
                    api_key_used=current_api_key_for_request, agent_name=agent_name, success=True
                )
            else:
                self._log_usage(model_name=current_embedding_model, duration_seconds=sdk_call_duration, api_key_used=current_api_key_for_request, agent_name=agent_name, success=True, prompt_tokens=None)
            return embedding_vector
        except RateLimitError as e:
            self._log_usage(model_name=current_embedding_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.warning(f"OpenAI API rate limit hit for embedding model {current_embedding_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMRateLimitError(f"OpenAI API rate limit hit for embedding: {e}") from e
        except (APIConnectionError, APITimeoutError, InternalServerError, APIStatusError) as e:
            self._log_usage(model_name=current_embedding_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            if isinstance(e, APIStatusError) and not (500 <= e.status_code < 600):
                logger.error(f"OpenAI API non-transient status error for embedding model {current_embedding_model}: {e.status_code} - {e}")
                raise LLMResponseError(f"OpenAI API non-transient status error for embedding {e.status_code}: {e}") from e
            logger.warning(f"OpenAI API transient error for embedding model {current_embedding_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMServiceUnavailableError(f"OpenAI API transient error for embedding: {e}") from e
        except AuthenticationError as e:
            self._log_usage(model_name=current_embedding_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"OpenAI API authentication error for embedding model {current_embedding_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMAuthenticationError(f"OpenAI API authentication error for embedding: {e}") from e
        except APIError as e:
            self._log_usage(model_name=current_embedding_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"OpenAI API error for embedding model {current_embedding_model}: {e}")
            if hasattr(e, 'status_code') and e.status_code == 400:
                 raise LLMConfigurationError(f"OpenAI API Bad Request for embedding (check inputs/config): {e}") from e
            raise LLMResponseError(f"OpenAI API error for embedding: {e}") from e
        except Exception as e:
            self._log_usage(model_name=current_embedding_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"Unexpected error during OpenAI embedding with model {current_embedding_model}: {e}")
            raise

    def _chat_impl(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int], model: str, current_api_key_for_request: Optional[str], agent_name: Optional[str] = None) -> Dict:
        current_model = model

        sdk_call_start_time = time.time()
        try:
            response = self.sdk_client.chat.completions.create(
                model=current_model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            sdk_call_duration = time.time() - sdk_call_start_time
            assistant_response = response.choices[0].message

            if response.usage:
                self._log_usage(
                    model_name=current_model, duration_seconds=sdk_call_duration,
                    prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens, api_key_used=current_api_key_for_request, agent_name=agent_name, success=True
                )
            else:
                self._log_usage(model_name=current_model, duration_seconds=sdk_call_duration, api_key_used=current_api_key_for_request, agent_name=agent_name, success=True, prompt_tokens=None)
            return {"role": assistant_response.role, "content": assistant_response.content.strip()}
        except RateLimitError as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.warning(f"OpenAI API rate limit hit during chat for model {current_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMRateLimitError(f"OpenAI API rate limit hit during chat: {e}") from e
        except (APIConnectionError, APITimeoutError, InternalServerError, APIStatusError) as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            if isinstance(e, APIStatusError) and not (500 <= e.status_code < 600):
                logger.error(f"OpenAI API non-transient status error during chat for model {current_model}: {e.status_code} - {e}")
                raise LLMResponseError(f"OpenAI API non-transient status error during chat {e.status_code}: {e}") from e
            logger.warning(f"OpenAI API transient error during chat for model {current_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMServiceUnavailableError(f"OpenAI API transient error during chat: {e}") from e
        except AuthenticationError as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"OpenAI API authentication error during chat for model {current_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMAuthenticationError(f"OpenAI API authentication error during chat: {e}") from e
        except APIError as e: # e.g. InvalidRequestError
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"OpenAI API error during chat for model {current_model}: {e}")
            if hasattr(e, 'status_code') and e.status_code == 400:
                 raise LLMConfigurationError(f"OpenAI API Bad Request during chat (check inputs/config): {e}") from e
            raise LLMResponseError(f"OpenAI API error during chat: {e}") from e
        except Exception as e:
            self._log_usage(model_name=current_model, duration_seconds=time.time()-sdk_call_start_time, api_key_used=current_api_key_for_request, agent_name=agent_name, success=False)
            logger.error(f"Unexpected error during OpenAI chat with model {current_model}: {e}")
            raise
        # Fallback empty response is removed as errors should be raised for tenacity to handle or for caller to catch.
        # If tenacity exhausts retries, it will re-raise the last exception.

    # _handle_api_error from BaseLLMClient can be used if no further specific handling is needed here,
    # or this can be kept if more OpenAI-specific error interpretation is added later.
    # For now, the direct raising of custom exceptions in each method is more explicit for retry logic.
    # def _handle_api_error(self, error: Exception, operation_name: str, model_name: str):
    #     super()._handle_api_error(error, operation_name, model_name)
    #     # Add more specific OpenAI error mapping to custom exceptions if needed here
    #     # For example, if a generic APIError should sometimes be an LLMConfigurationError
    #     if isinstance(error, APIError) and hasattr(error, 'status_code'):
    #         if error.status_code == 400: # Bad Request
    #             raise LLMConfigurationError(f"OpenAI Bad Request (check inputs/config) during {operation_name} on {model_name}: {error}") from error
    #         elif error.status_code == 401: # Unauthorized
    #             raise LLMAuthenticationError(f"OpenAI Authentication Error during {operation_name} on {model_name}: {error}") from error
    #         elif error.status_code == 429: # Rate limit
    #             raise LLMRateLimitError(f"OpenAI Rate Limit during {operation_name} on {model_name}: {error}") from error
    #         elif 500 <= error.status_code < 600: # Server errors
    #             raise LLMServiceUnavailableError(f"OpenAI Service Unavailable during {operation_name} on {model_name}: {error}") from error
    #     # If not mapped, the original error (or the one from super) will be raised.
