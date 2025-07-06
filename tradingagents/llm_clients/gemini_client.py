import google.generativeai as genai
import os
import time
import logging
from typing import List, Dict, Optional

from tradingagents.llm_clients.base_client import BaseLLMClient, logger # Use the logger from base

# It's good practice to ensure API key is configured when the module is used.
# However, the actual configuration call (genai.configure) should happen
# ideally once, and can be triggered by the client's __init__.

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

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, model: Optional[str] = None) -> str:
        """
        Generates text using the Gemini model.
        Args:
            prompt (str): The input prompt.
            temperature (float): Controls randomness.
            max_tokens (int, optional): Maximum number of tokens for the output.
                                       Note: Gemini API uses 'max_output_tokens'.
            model (str, optional): Specific model name to override the default.
        Returns:
            str: The generated text.
        """
        # Call super for logging start time
        super().generate_text(prompt, temperature, max_tokens, model)

        current_model_name = model or self.model_name

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

            generated_text = response.text

            # Token counting for Gemini is not directly available in the same way as OpenAI's response object.
            # genai.count_tokens(contents=prompt, model=current_model_name) can give prompt tokens.
            # Response tokens are harder to get directly without parsing or if not provided by future API updates.
            # For now, we'll log what we can.
            try:
                prompt_token_count = active_sdk_model.count_tokens(prompt).total_tokens
                # response_token_count = active_sdk_model.count_tokens(generated_text).total_tokens # This would count tokens of the *generated* text
                # total_tokens = prompt_token_count + response_token_count
                self._log_usage(model_name=current_model_name, prompt_tokens=prompt_token_count) #, completion_tokens=response_token_count)
            except Exception as token_count_error:
                logger.warning(f"Could not count tokens for Gemini model {current_model_name}: {token_count_error}")

            return generated_text
        except Exception as e:
            logger.error(f"Gemini API error during text generation with model {current_model_name}: {e}")
            self._handle_api_error(e) # This will re-raise

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generates an embedding for a given text using Gemini embedding models.
        Args:
            text (str): The input text to embed.
            model (str, optional): Specific embedding model name to override the default.
                                   e.g., 'models/embedding-001' or 'models/text-embedding-004'.
        Returns:
            List[float]: The embedding vector.
        """
        # Call super for logging start time
        super().get_embedding(text, model)

        current_embedding_model = model or self.embedding_model_name

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
        except Exception as e:
            logger.error(f"Gemini API error during embedding generation with model {current_embedding_model}: {e}")
            self._handle_api_error(e) # This will re-raise

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
