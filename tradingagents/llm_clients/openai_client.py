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

    def _generate_text_impl(self, prompt: str, temperature: float, max_tokens: Optional[int], model: str) -> str:
        # 'model' here is the effective_model determined by the public method
        current_model = model

        # OpenAI uses ChatCompletion for general text generation
        messages = [{"role": "user", "content": prompt}]
        # If a system prompt is part of the config or standard usage, it should be added here:
        # messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]

        try:
            response = self.sdk_client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            generated_text = response.choices[0].message.content.strip()

            if response.usage:
                self._log_usage(
                    model_name=current_model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            return generated_text
        except RateLimitError as e:
            logger.warning(f"OpenAI API rate limit hit for model {current_model}: {e}")
            raise LLMRateLimitError(f"OpenAI API rate limit hit: {e}") from e
        except (APIConnectionError, APITimeoutError, InternalServerError, APIStatusError) as e: # APIStatusError for 5xx
             # Check if APIStatusError is indeed a 5xx type error that is transient
            if isinstance(e, APIStatusError) and not (500 <= e.status_code < 600):
                logger.error(f"OpenAI API non-transient status error for model {current_model}: {e.status_code} - {e}")
                raise LLMResponseError(f"OpenAI API non-transient status error {e.status_code}: {e}") from e # Not transient if not 5xx
            logger.warning(f"OpenAI API transient error (connection, timeout, or 5xx) for model {current_model}: {e}")
            raise LLMServiceUnavailableError(f"OpenAI API transient error: {e}") from e
        except AuthenticationError as e:
            logger.error(f"OpenAI API authentication error for model {current_model}: {e}")
            raise LLMAuthenticationError(f"OpenAI API authentication error: {e}") from e
        except APIError as e: # Catch other APIErrors (e.g. InvalidRequestError which might be config)
            logger.error(f"OpenAI API error for model {current_model}: {e}")
            # Could be LLMConfigurationError or LLMResponseError depending on status code
            if hasattr(e, 'status_code') and e.status_code == 400: # Bad Request
                 raise LLMConfigurationError(f"OpenAI API Bad Request (check inputs/config): {e}") from e
            raise LLMResponseError(f"OpenAI API error: {e}") from e # Generic response error
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during OpenAI text generation with model {current_model}: {e}")
            raise # Re-raise to be handled by BaseLLMClient's _handle_api_error or tenacity


    def _get_embedding_impl(self, text: str, model: str) -> List[float]:
        # 'model' here is the effective_model determined by the public method
        current_embedding_model = model

        if not text or not text.strip(): # Check from base client is good, but double check here is fine
            logger.warning("Attempted to get embedding for empty or whitespace-only text in OpenAIClient impl.")
            # This should ideally be caught by the public get_embedding method in BaseLLMClient if we add a check there.
            # For now, matching existing behavior.
            # Consider raising LLMConfigurationError or returning a specific error object.
            # For consistency with previous version, returning zero vector.
            # A more robust solution is for BaseLLMClient.get_embedding to validate input.
            default_dim = 1536 # Common default, e.g. for ada-002 or text-embedding-3-small
            if "large" in current_embedding_model: default_dim = 3072
            if "small" in current_embedding_model: default_dim = 1536 # often
            logger.info(f"Returning zero vector of dim {default_dim} for empty embedding input.")
            return [0.0] * default_dim


        try:
            response = self.sdk_client.embeddings.create(
                input=[text.replace("\n", " ")],
                model=current_embedding_model
            )
            embedding_vector = response.data[0].embedding

            if response.usage:
                self._log_usage(
                    model_name=current_embedding_model,
                    prompt_tokens=response.usage.prompt_tokens,
                    total_tokens=response.usage.total_tokens
                )
            return embedding_vector
        except RateLimitError as e:
            logger.warning(f"OpenAI API rate limit hit for embedding model {current_embedding_model}: {e}")
            raise LLMRateLimitError(f"OpenAI API rate limit hit for embedding: {e}") from e
        except (APIConnectionError, APITimeoutError, InternalServerError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and not (500 <= e.status_code < 600):
                logger.error(f"OpenAI API non-transient status error for embedding model {current_embedding_model}: {e.status_code} - {e}")
                raise LLMResponseError(f"OpenAI API non-transient status error for embedding {e.status_code}: {e}") from e
            logger.warning(f"OpenAI API transient error for embedding model {current_embedding_model}: {e}")
            raise LLMServiceUnavailableError(f"OpenAI API transient error for embedding: {e}") from e
        except AuthenticationError as e:
            logger.error(f"OpenAI API authentication error for embedding model {current_embedding_model}: {e}")
            raise LLMAuthenticationError(f"OpenAI API authentication error for embedding: {e}") from e
        except APIError as e: # e.g. InvalidRequestError
            logger.error(f"OpenAI API error for embedding model {current_embedding_model}: {e}")
            if hasattr(e, 'status_code') and e.status_code == 400: # Bad Request
                 raise LLMConfigurationError(f"OpenAI API Bad Request for embedding (check inputs/config): {e}") from e
            raise LLMResponseError(f"OpenAI API error for embedding: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI embedding with model {current_embedding_model}: {e}")
            raise

    def _chat_impl(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int], model: str) -> Dict:
        # 'model' here is the effective_model
        current_model = model
        try:
            response = self.sdk_client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            assistant_response = response.choices[0].message

            if response.usage:
                self._log_usage(
                    model_name=current_model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            return {"role": assistant_response.role, "content": assistant_response.content.strip()}
        except RateLimitError as e:
            logger.warning(f"OpenAI API rate limit hit during chat for model {current_model}: {e}")
            raise LLMRateLimitError(f"OpenAI API rate limit hit during chat: {e}") from e
        except (APIConnectionError, APITimeoutError, InternalServerError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and not (500 <= e.status_code < 600):
                logger.error(f"OpenAI API non-transient status error during chat for model {current_model}: {e.status_code} - {e}")
                raise LLMResponseError(f"OpenAI API non-transient status error during chat {e.status_code}: {e}") from e
            logger.warning(f"OpenAI API transient error during chat for model {current_model}: {e}")
            raise LLMServiceUnavailableError(f"OpenAI API transient error during chat: {e}") from e
        except AuthenticationError as e:
            logger.error(f"OpenAI API authentication error during chat for model {current_model}: {e}")
            raise LLMAuthenticationError(f"OpenAI API authentication error during chat: {e}") from e
        except APIError as e: # e.g. InvalidRequestError
            logger.error(f"OpenAI API error during chat for model {current_model}: {e}")
            if hasattr(e, 'status_code') and e.status_code == 400:
                 raise LLMConfigurationError(f"OpenAI API Bad Request during chat (check inputs/config): {e}") from e
            raise LLMResponseError(f"OpenAI API error during chat: {e}") from e
        except Exception as e:
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
