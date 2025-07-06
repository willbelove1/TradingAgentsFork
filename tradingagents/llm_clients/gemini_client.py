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
)

class GeminiClient(BaseLLMClient):
    """
    LLM Client for Google's Gemini models.
    """

    # Default model names, can be overridden by config
    DEFAULT_TEXT_MODEL = "gemini-pro"  # Or a more specific version like "gemini-1.5-flash"
    DEFAULT_EMBEDDING_MODEL = "models/embedding-001"
    # For Gemini 1.5 Pro, vision model is usually the same as the text model.
    # For older models, it might be "gemini-pro-vision".
    # The SDK generally handles multimodal capabilities within the same model endpoint for newer models.

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initializes the Gemini client.
        Args:
            api_key (str, optional): Google AI API key. If None, attempts to use GOOGLE_API_KEY environment variable.
            config (Dict, optional): Configuration dictionary. Expected keys:
                - 'model': Name of the default text generation model (e.g., 'gemini-1.5-pro').
                - 'embedding_model': Name of the default embedding model (e.g., 'models/embedding-001').
        """
        super().__init__(api_key, config)

        resolved_api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            # Attempt to load from .env if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv()
                resolved_api_key = os.getenv("GOOGLE_API_KEY")
            except ImportError:
                logger.info("python-dotenv not installed, cannot load GOOGLE_API_KEY from .env file.")

            if not resolved_api_key:
                raise ValueError("GOOGLE_API_KEY not found. Please set it as an environment variable, pass it to the constructor, or ensure it's in a .env file.")

        genai.configure(api_key=resolved_api_key)

        self.model_name = self.config.get("model", self.DEFAULT_TEXT_MODEL)
        self.embedding_model_name = self.config.get("embedding_model", self.DEFAULT_EMBEDDING_MODEL)

        # Initialize the generative model instance.
        # We can re-initialize if a different model is passed to generate_text,
        # or create a dictionary of model instances if frequently switching.
        # For simplicity, we'll use one default and allow overriding.
        try:
            self._sdk_model = genai.GenerativeModel(self.model_name)
            # For embeddings, the model name is passed directly to the embed_content function.
        except Exception as e:
            logger.error(f"Failed to initialize Gemini GenerativeModel with {self.model_name}: {e}")
            self._handle_api_error(e)

        logger.info(f"GeminiClient initialized. Default text model: {self.model_name}, Default embedding model: {self.embedding_model_name}")

    def _generate_text_impl(self, prompt: str, temperature: float, max_tokens: Optional[int], model: str) -> str:
        # 'model' here is the effective_model determined by the public method
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
            # Re-raise as a generic transient error if it seems like one, otherwise let base class handle
            if isinstance(e, google_exceptions.GoogleAPIError): # Base for many google API errors
                 raise LLMTransientError(f"Unhandled Google API error: {e}") from e
            raise # Re-raise other unexpected errors to be caught by BaseLLMClient's _handle_api_error


    def _get_embedding_impl(self, text: str, model: str) -> List[float]:
        # 'model' here is the effective_model determined by the public method
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

            return embedding_vector
        except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests) as e:
            logger.warning(f"Gemini API rate limit hit for embedding model {current_embedding_model}: {e}")
            raise LLMRateLimitError(f"Gemini API rate limit hit for embedding: {e}") from e
        except (google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError) as e:
            logger.warning(f"Gemini API service unavailable for embedding model {current_embedding_model}: {e}")
            raise LLMServiceUnavailableError(f"Gemini API service unavailable for embedding: {e}") from e
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Gemini API InvalidArgument for embedding model {current_embedding_model}: {e}")
            raise LLMConfigurationError(f"Gemini API InvalidArgument for embedding: {e}") from e
        except google_exceptions.PermissionDenied as e:
            logger.error(f"Gemini API Permission Denied for embedding model {current_embedding_model}: {e}")
            raise LLMAuthenticationError(f"Gemini API Permission Denied for embedding: {e}") from e
        except Exception as e: # Catch-all for other google_exceptions or unexpected errors
            logger.error(f"Unexpected Gemini API error during embedding with model {current_embedding_model}: {e}")
            if isinstance(e, google_exceptions.GoogleAPIError):
                 raise LLMTransientError(f"Unhandled Google API error during embedding: {e}") from e
            raise


    # def _chat_impl(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int], model: str) -> Dict:
    #     # 'model' here is the effective_model determined by the public method
    #     current_model_name = model
    #     # ... (similar try-except structure as _generate_text_impl, mapping SDK errors to custom LLM errors)
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
