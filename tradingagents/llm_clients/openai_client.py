from openai import OpenAI as OpenAIClientSDK, APIError, RateLimitError, AuthenticationError # Renamed to avoid conflict
import os
import time
import logging
from typing import List, Dict, Optional

from tradingagents.llm_clients.base_client import BaseLLMClient, logger # Use the logger from base

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

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = 1500, model: Optional[str] = None) -> str:
        super().generate_text(prompt, temperature, max_tokens, model)
        current_model = model or self.model_name

        # OpenAI uses ChatCompletion for general text generation with newer models
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
        except APIError as e: # Catch more specific OpenAI errors
            logger.error(f"OpenAI API error during text generation with model {current_model}: {e}")
            self._handle_api_error(e)
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during OpenAI text generation with model {current_model}: {e}")
            self._handle_api_error(e)


    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        super().get_embedding(text, model)
        current_embedding_model = model or self.embedding_model_name
        # OpenAI API expects a non-empty string.
        if not text or not text.strip():
            logger.warning("Attempted to get embedding for empty or whitespace-only text. Returning zero vector.")
            # Determine the dimensionality of the model if possible, or return a fixed-size zero vector.
            # For now, let's assume a common size like 1536 for ada-002 or text-embedding-3-small
            # This should ideally be dynamically determined or handled more gracefully.
            return [0.0] * 1536 # Placeholder for zero vector

        try:
            response = self.sdk_client.embeddings.create(
                input=[text.replace("\n", " ")], # API recommendation: replace newlines
                model=current_embedding_model
            )
            embedding_vector = response.data[0].embedding

            if response.usage:
                self._log_usage(
                    model_name=current_embedding_model,
                    prompt_tokens=response.usage.prompt_tokens, # For embeddings, this is typically the input tokens
                    total_tokens=response.usage.total_tokens
                )
            return embedding_vector
        except APIError as e:
            logger.error(f"OpenAI API error during embedding generation with model {current_embedding_model}: {e}")
            self._handle_api_error(e)
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI embedding generation with model {current_embedding_model}: {e}")
            self._handle_api_error(e)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = 1500, model: Optional[str] = None) -> Dict:
        super().chat(messages, temperature, max_tokens, model)
        current_model = model or self.model_name

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
        except APIError as e:
            logger.error(f"OpenAI API error during chat with model {current_model}: {e}")
            self._handle_api_error(e)
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI chat with model {current_model}: {e}")
            self._handle_api_error(e)
        return {"role": "assistant", "content": ""} # Fallback empty response

    def _handle_api_error(self, error):
        """
        Handles common OpenAI API errors.
        """
        if isinstance(error, RateLimitError):
            logger.warning(f"OpenAI RateLimitError: {error}. Consider implementing retry with exponential backoff.")
        elif isinstance(error, AuthenticationError):
            logger.critical(f"OpenAI AuthenticationError: {error}. Check your OPENAI_API_KEY and organization if applicable.")
        elif isinstance(error, APIError): # General API error
            logger.error(f"OpenAI APIError: Status Code: {error.status_code}, Message: {error.message}")
        else: # Other unexpected errors
            logger.error(f"Unexpected OpenAI client error: {error}")
        raise error # Re-raise the error after logging
