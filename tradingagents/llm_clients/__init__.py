import logging
from typing import Dict, Optional

from .base_client import BaseLLMClient
from .gemini_client import GeminiClient
# from .openai_client import OpenAIClient # Removed
# Import other clients here if you add them, e.g.:
# from .anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {
    "gemini": GeminiClient,
    "google": GeminiClient, # Alias for gemini
    # "openai": OpenAIClient, # Removed
    # "anthropic": AnthropicClient,
}

def get_llm_client(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[Dict] = None
) -> BaseLLMClient:
    """
    Factory function to get an instance of an LLM client based on the provider name.

    Args:
        provider_name (str, optional): The name of the LLM provider (e.g., "gemini", "openai").
                                       If None, it might try to infer from config or raise error.
        api_key (str, optional): The API key for the provider. Can often be loaded from env if None.
        config (Dict, optional): A configuration dictionary that might contain the provider name,
                                 model names, API keys, base URLs, etc.
                                 The 'llm_provider' key in config can also specify the provider.

    Returns:
        BaseLLMClient: An instance of the specified LLM client.

    Raises:
        ValueError: If the provider_name is not supported or cannot be determined.
    """
    effective_config = config.copy() if config else {}

    # Determine provider name
    if provider_name:
        provider_name = provider_name.lower()
    elif 'llm_provider' in effective_config:
        provider_name = effective_config['llm_provider'].lower()
    else:
        raise ValueError("LLM provider name must be specified either via 'provider_name' argument or 'llm_provider' in config.")

    # Update config with explicit api_key if provided (overrides env vars within client)
    if api_key:
        effective_config['api_key'] = api_key # The clients can look for this key in their __init__

    client_class = SUPPORTED_PROVIDERS.get(provider_name)

    if client_class:
        logger.info(f"Creating LLM client for provider: {provider_name}")
        # The client's __init__ should handle extracting its specific needs from the effective_config
        # and also respect the direct api_key if passed.
        return client_class(api_key=api_key, config=effective_config)
    else:
        logger.error(f"Unsupported LLM provider: {provider_name}")
        raise ValueError(f"Unsupported LLM provider: {provider_name}. Supported providers are: {list(SUPPORTED_PROVIDERS.keys())}")

# Example usage (for testing or direct use):
# if __name__ == '__main__':
#     # Ensure environment variables (e.g., GOOGLE_API_KEY, OPENAI_API_KEY) are set
#     # or pass them directly or via config.
#     try:
#         print("Testing Gemini Client...")
#         gemini_conf = {"model": "gemini-pro", "embedding_model": "models/embedding-001", "llm_provider": "gemini"}
#         gemini_client = get_llm_client(config=gemini_conf)
#         # gemini_text = gemini_client.generate_text("Explain quantum computing in simple terms.")
#         # print(f"Gemini Response: {gemini_text[:100]}...")
#         # gemini_embedding = gemini_client.get_embedding("Hello world")
#         # print(f"Gemini Embedding (first 5 dims): {gemini_embedding[:5]}")
#         print("Gemini Client OK (basic init)")
#     except Exception as e:
#         print(f"Error testing Gemini Client: {e}")

#     print("\nTesting OpenAI Client...")
#     try:
#         openai_conf = {"model": "gpt-3.5-turbo", "embedding_model": "text-embedding-ada-002", "llm_provider": "openai"}
#         # To test with a local server like Ollama:
#         # openai_conf = {"model": "llama2", "llm_provider": "openai", "base_url": "http://localhost:11434/v1", "api_key":"ollama"}
#         openai_client = get_llm_client(config=openai_conf)
#         # openai_text = openai_client.generate_text("What is the capital of France?")
#         # print(f"OpenAI Response: {openai_text[:100]}...")
#         # openai_embedding = openai_client.get_embedding("Bonjour le monde")
#         # print(f"OpenAI Embedding (first 5 dims): {openai_embedding[:5]}")
#         print("OpenAI Client OK (basic init)")
#     except Exception as e:
#         print(f"Error testing OpenAI Client: {e}")
