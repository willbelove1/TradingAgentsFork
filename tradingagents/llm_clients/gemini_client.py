import google.generativeai as genai
import os
import time
import logging
from typing import List, Dict, Optional

from tradingagents.llm_clients.base_client import BaseLLMClient, logger # Use the logger from base

import google.api_core.exceptions as google_exceptions # For specific exception types

from tradingagents.llm_clients.base_client import (
    BaseLLMClient, logger,
    LLMTransientError, LLMRateLimitError, LLMServiceUnavailableError,
    LLMAuthenticationError, LLMConfigurationError, LLMResponseError
import itertools # For API key cycling

# Import google_exceptions and custom LLM exceptions from base_client
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions

from tradingagents.llm_clients.base_client import (
    BaseLLMClient, logger,
    LLMTransientError, LLMRateLimitError, LLMServiceUnavailableError,
    LLMAuthenticationError, LLMConfigurationError, LLMResponseError,
    LLMKeyCycleError # Import the new exception
)

class GeminiClient(BaseLLMClient):
    """
    LLM Client for Google's Gemini models, with API key cycling and retry logic.
    """
    DEFAULT_TEXT_MODEL = "gemini-1.5-flash"
    DEFAULT_EMBEDDING_MODEL = "models/embedding-001"

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initializes the Gemini client.
        Args:
            api_key (str, optional): A single Google AI API key (acts as fallback or if no list in config).
            config (Dict, optional): Configuration dictionary. Expected keys under `google_config` (or top-level):
                - 'api_keys': List of Google AI API keys for cycling.
                - 'model' / 'default_text_model': Name of the default text generation model.
                - 'embedding_model' / 'default_embedding_model': Name of the default embedding model.
                - Retry settings: 'retry_attempts', 'retry_min_wait_seconds', 'retry_max_wait_seconds'.
        """
        super().__init__(api_key, config) # api_key here is the single one, config has all settings

        # API Key Management for Cycling
        self.api_keys_list: List[str] = []
        # Prioritize list of keys from config
        if isinstance(self.config.get("api_keys"), list) and self.config["api_keys"]:
            self.api_keys_list = [key for key in self.config["api_keys"] if key] # Filter out empty strings

        # Fallback to single api_key from constructor (which might be from env GOOGLE_API_KEY via get_llm_client)
        if not self.api_keys_list and self.api_key:
            self.api_keys_list.append(self.api_key)

        # Fallback to GOOGLE_API_KEY env var if still no keys
        if not self.api_keys_list:
            env_api_key = os.getenv("GOOGLE_API_KEY")
            if env_api_key:
                self.api_keys_list.append(env_api_key)
            else: # Try .env
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    env_api_key_dotenv = os.getenv("GOOGLE_API_KEY")
                    if env_api_key_dotenv:
                        self.api_keys_list.append(env_api_key_dotenv)
                except ImportError:
                    logger.debug("python-dotenv not installed, cannot load GOOGLE_API_KEY from .env for GeminiClient key list.")

        if not self.api_keys_list:
            raise LLMConfigurationError(
                "GeminiClient: No API keys found. Provide 'api_keys' in google_config (model_settings.yaml), "
                "as a constructor argument, or set GOOGLE_API_KEY environment variable."
            )

        self.api_key_cycler = itertools.cycle(self.api_keys_list)
        self.current_api_key_for_sdk: Optional[str] = None # Key currently configured with genai SDK

        # Initialize the SDK with the first key. It will be re-configured on key cycling.
        self._configure_sdk_with_next_key()

        # Default model names from config or class defaults
        self.model_name = self.config.get("model") or self.config.get("default_text_model") or self.DEFAULT_TEXT_MODEL
        self.embedding_model_name = self.config.get("embedding_model") or self.config.get("default_embedding_model") or self.DEFAULT_EMBEDDING_MODEL

        # _sdk_model instance is created per call in _generate_text_impl if model changes
        # No need to initialize self._sdk_model here for a specific default model if it's handled per-call

        logger.info(
            f"GeminiClient initialized with {len(self.api_keys_list)} API key(s). "
            f"Default text model: {self.model_name}, Default embedding model: {self.embedding_model_name}. "
            f"Retry config: {self.retry_attempts} attempts, wait {self.retry_min_wait}-{self.retry_max_wait}s."
        )

    def _configure_sdk_with_next_key(self) -> str:
        """Gets the next API key and configures the genai SDK with it."""
        new_key = next(self.api_key_cycler)
        if new_key != self.current_api_key_for_sdk: # Avoid re-configuring if key hasn't changed (e.g. only one key)
            logger.info(f"GeminiClient: Configuring SDK with new API key (ending with ...{new_key[-4:] if len(new_key) > 4 else new_key}).")
            try:
                genai.configure(api_key=new_key)
                self.current_api_key_for_sdk = new_key
            except Exception as e: # Catch potential errors during genai.configure itself
                logger.error(f"GeminiClient: Failed to configure SDK with API key ...{new_key[-4:]}: {e}")
                # This is a configuration error for this key, but we want to cycle, so raise LLMKeyCycleError
                raise LLMKeyCycleError(f"Failed to configure genai SDK with key ...{new_key[-4:]}", failed_key=new_key, original_exception=e) from e
        return new_key

    def _get_current_api_key_for_request(self) -> Optional[str]:
        """
        Overrides BaseLLMClient to provide key cycling for Gemini.
        Ensures the SDK is configured with the key to be used for the upcoming attempt.
        This is called by BaseLLMClient's retry wrapper before each attempt of _impl methods.
        """
        try:
            # This will get a key and configure the SDK with it.
            # If _configure_sdk_with_next_key itself fails (e.g., bad key format for genai.configure),
            # it raises LLMKeyCycleError, which is a LLMTransientError, so tenacity will retry.
            # During the retry, a new key will be picked.
            return self._configure_sdk_with_next_key()
        except LLMKeyCycleError: # Already an LLMKeyCycleError, let tenacity handle it
            raise
        except Exception as e: # Should not happen if _configure_sdk_with_next_key handles its errors
            logger.error(f"GeminiClient: Unexpected error in _get_current_api_key_for_request: {e}")
            raise LLMKeyCycleError("Unexpected error getting next API key", original_exception=e) from e


    def _generate_text_impl(self, prompt: str, temperature: float, max_tokens: Optional[int], model: str, current_api_key_for_request: Optional[str], agent_name: Optional[str] = None) -> str:
        current_model_name = model

        generation_config = genai.types.GenerationConfig(
            temperature=temperature
        )
        if max_tokens is not None:
            generation_config.max_output_tokens = max_tokens

        # Safety settings can be configured here if needed.
        # Example:
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]

        try:
            # Use the class's _sdk_model if model matches, else create a new one
            active_sdk_model = self._sdk_model
            if model and model != self.model_name:
                logger.info(f"Switching to model: {model} for this generate_text call.")
                active_sdk_model = genai.GenerativeModel(model)

            response = active_sdk_model.generate_content(
                prompt,
                generation_config=generation_config,
        # safety_settings=safety_settings
            )

            if not response.candidates:
                # This can happen if all candidates were filtered by safety settings or other reasons
                logger.warning(f"Gemini model {current_model_name} returned no candidates. Prompt: '{prompt[:100]}...'")
                # Check for prompt feedback if available
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                    logger.error(f"Prompt blocked for model {current_model_name} due to: {reason}")
                    raise LLMResponseError(f"Prompt blocked by Gemini safety filters: {reason}")
                raise LLMResponseError(f"No candidates returned from Gemini model {current_model_name}.")

            generated_text = response.text # Accessing .text might raise if no valid candidate

            try:
                prompt_token_count = active_sdk_model.count_tokens(prompt).total_tokens
                self._log_usage(model_name=current_model_name, prompt_tokens=prompt_token_count)
            except Exception as token_count_error: # pragma: no cover
                logger.warning(f"Could not count tokens for Gemini model {current_model_name}: {token_count_error}")

            return generated_text
        except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests) as e:
            logger.warning(f"Gemini API rate limit hit for model {current_model_name}: {e}")
            raise LLMRateLimitError(f"Gemini API rate limit hit: {e}") from e
        except (google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError) as e:
            logger.warning(f"Gemini API service unavailable for model {current_model_name}: {e}")
            raise LLMServiceUnavailableError(f"Gemini API service unavailable: {e}") from e
        except google_exceptions.InvalidArgument as e: # Often due to bad prompt/content or model name
            logger.error(f"Gemini API InvalidArgument for model {current_model_name} (check prompt or model name): {e}")
            raise LLMConfigurationError(f"Gemini API InvalidArgument (check prompt or model name): {e}") from e
        except google_exceptions.PermissionDenied as e: # API Key or access issues
            logger.error(f"Gemini API Permission Denied for model {current_model_name}: {e}")
            raise LLMAuthenticationError(f"Gemini API Permission Denied: {e}") from e
        except genai.types.generation_types.StopCandidateException as e: # Safety filter related
            logger.error(f"Gemini content generation stopped (safety/policy) for model {current_model_name}: {e}")
            raise LLMResponseError(f"Gemini content generation stopped (safety/policy): {e}") from e
        except Exception as e: # Catch-all for other google_exceptions or unexpected errors
            logger.error(f"Unexpected Gemini API error during text generation with model {current_model_name}: {e}")
            # Re-raise as a generic transient error if it seems like one, otherwise let base class's _handle_api_error deal with it
            if isinstance(e, google_exceptions.GoogleAPIError): # Base for many google API errors
                 raise LLMTransientError(f"Unhandled Google API error during text generation: {e}") from e # This will be retried by BaseClient
            raise # Re-raise other unexpected errors


    def _get_embedding_impl(self, text: str, model: str, current_api_key_for_request: Optional[str]) -> List[float]:
        # `current_api_key_for_request` is logged by BaseLLMClient.
        current_embedding_model = model

        try:
            # For embeddings, the model name is passed directly.
            # Task type can be specified, e.g., "RETRIEVAL_DOCUMENT" or "SEMANTIC_SIMILARITY"
            # Default is "UNSPECIFIED" which works for general cases.
            result = genai.embed_content(
                model=current_embedding_model,
                content=text,
                # task_type="retrieval_document" # Example
            )
            embedding_vector = result['embedding']

            # Token counting for embeddings is also not directly in response.
            # We can log the model used.
            # prompt_tokens are implicitly the tokens in 'text'.
            # self._log_usage(model_name=current_embedding_model, prompt_tokens=genai.count_tokens(text, model=current_embedding_model).total_tokens)
            # The above count_tokens might not work for embedding models, or might not be relevant.
            # Logging the fact that an embedding was generated is often sufficient here.
            # Token usage for embeddings is harder to get precisely from Gemini SDK in a simple way.
            # self._log_usage(model_name=current_embedding_model, api_key_used=current_api_key_for_request)
            return embedding_vector
        except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests) as e:
            logger.warning(f"Gemini API rate limit hit for embedding model {current_embedding_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMKeyCycleError(f"Gemini API rate limit hit for embedding", failed_key=current_api_key_for_request, original_exception=e) from e
        except (google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError) as e:
            logger.warning(f"Gemini API service unavailable for embedding model {current_embedding_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            raise LLMKeyCycleError(f"Gemini API service unavailable for embedding", failed_key=current_api_key_for_request, original_exception=e) from e
        except google_exceptions.InvalidArgument as e: # Often due to bad model name or input
            logger.error(f"Gemini API InvalidArgument for embedding model {current_embedding_model}: {e}")
            raise LLMConfigurationError(f"Gemini API InvalidArgument for embedding: {e}") from e
        except google_exceptions.PermissionDenied as e: # API Key or access issues for this specific key
            logger.error(f"Gemini API Permission Denied for embedding model {current_embedding_model} (Key: ...{current_api_key_for_request[-4:] if current_api_key_for_request else 'N/A'}): {e}")
            # This key is bad, trigger cycle. If all keys are bad, retries will exhaust.
            raise LLMKeyCycleError(f"Gemini API Permission Denied for embedding", failed_key=current_api_key_for_request, original_exception=e) from e
        except Exception as e:
            logger.error(f"Unexpected Gemini API error during embedding with model {current_embedding_model}: {e}")
            if isinstance(e, google_exceptions.GoogleAPIError):
                 raise LLMTransientError(f"Unhandled Google API error during embedding: {e}") from e
            raise


    # def _chat_impl(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int], model: str, current_api_key_for_request: Optional[str]) -> Dict:
    #     # `current_api_key_for_request` would be passed by the BaseLLMClient's public chat method.
    #     current_model_name = model
    #     # ... (similar try-except structure as _generate_text_impl, mapping SDK errors to custom LLM errors,
    #     #      and raising LLMKeyCycleError for key-specific issues like PermissionDenied or RateLimit)
    #     # Example:
    #     # try:
    #     #     response = active_sdk_model.generate_content(contents=gemini_messages, generation_config=generation_config)
    #     #     # ... process response ...
    #     #     return {"role": "assistant", "content": assistant_response_content}
    #     # except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests) as e:
    #     #     raise LLMRateLimitError(f"Gemini API rate limit hit during chat: {e}") from e
    #     # ... other specific exceptions ...
    #     # except Exception as e:
    #     #     logger.error(f"Unexpected Gemini API error during chat with model {current_model_name}: {e}")
    #     #     if isinstance(e, google_exceptions.GoogleAPIError):
    #     #          raise LLMTransientError(f"Unhandled Google API error during chat: {e}") from e
    #     #     raise
    #     return {} # Placeholder

    # def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None, model: Optional[str] = None) -> Dict:
    #     """
    #     Conducts a chat conversation using the Gemini model.
    #     Args:
    #         messages (List[Dict[str, str]]): A list of message objects.
    #                                          Gemini expects content in a specific format, often just a list of strings for simple turn-based,
    #                                          or more structured for multi-turn with history.
    #                                          Example: [{"role": "user", "parts": ["Hello!"]}, {"role": "model", "parts": ["Hi there!"]}]
    #         temperature (float): Controls randomness.
    #         max_tokens (int, optional): Maximum number of tokens for the output.
    #         model (str, optional): Specific model name to override the default.
    #     Returns:
    #         Dict: The assistant's response message (e.g., {"role": "model", "content": "..."}).
    #     """
    #     super().chat(messages, temperature, max_tokens, model) # For logging and timing

    #     current_model_name = model or self.model_name

    #     generation_config = genai.types.GenerationConfig(
    #         temperature=temperature
    #     )
    #     if max_tokens is not None:
    #         generation_config.max_output_tokens = max_tokens

    #     # Convert messages to Gemini's expected format if necessary
    #     # This might involve mapping "role": "system" to an initial user prompt,
    #     # or ensuring "parts" is a list of strings.
    #     gemini_messages = []
    #     for msg in messages:
    #         role = msg.get("role", "user")
    #         # Gemini typically uses 'user' and 'model' roles. System prompts are often handled by prepending to the user message.
    #         if role == "system": # Simple conversion: treat system as the first user message part if no user message yet.
    #             # Or prepend to the next user message. For now, let's assume direct mapping works or it's handled by user.
    #             # This part needs careful consideration based on how system prompts are used in the application.
    #             # For complex scenarios, a dedicated ChatSession (start_chat) might be better.
    #             pass # For now, assume user handles system prompts correctly in the message list

    #         gemini_messages.append({"role": role if role != "assistant" else "model", "parts": [msg.get("content", "")]})


    #     try:
    #         active_sdk_model = self._sdk_model
    #         if model and model != self.model_name:
    #             logger.info(f"Switching to model: {model} for this chat call.")
    #             active_sdk_model = genai.GenerativeModel(model)

    #         # For multi-turn chat, it's often better to use a ChatSession
    #         # chat_session = active_sdk_model.start_chat(history=gemini_messages[:-1]) # if messages include history
    #         # response = chat_session.send_message(gemini_messages[-1]['parts'][0], generation_config=generation_config)

    #         # Simpler approach for single/few turns if not managing complex history here:
    #         response = active_sdk_model.generate_content(
    #             contents=gemini_messages, # `contents` takes the list of turns
    #             generation_config=generation_config
    #         )

    #         assistant_response_content = response.text # or response.parts[0].text if structured

    #         # Token counting
    #         # prompt_token_count = active_sdk_model.count_tokens(gemini_messages).total_tokens
    #         # self._log_usage(model_name=current_model_name, prompt_tokens=prompt_token_count)

    #         return {"role": "assistant", "content": assistant_response_content}
    #     except Exception as e:
    #         logger.error(f"Gemini API error during chat with model {current_model_name}: {e}")
    #         self._handle_api_error(e)
    #     return {}
