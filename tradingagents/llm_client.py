from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Any, Dict

class LLMClientFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_llm(self, model_config_key: str):
        """
        Creates an LLM instance based on a configuration key that points to the model name.

        Args:
            model_config_key (str): The key in the config dictionary (e.g., "deep_think_llm")
                                    whose value is the actual model name string.

        Returns:
            An instance of a Langchain ChatModel.
        """
        provider = self.config.get("llm_provider", "").lower()
        actual_model_name = self.config.get(model_config_key)

        if not actual_model_name:
            raise ValueError(f"Model name for config key '{model_config_key}' not found in config.")

        backend_url = self.config.get("backend_url")
        return self._create_llm_instance(actual_model_name, provider, backend_url)

    def get_llm_by_name(self, model_name_str: str):
        """
        Creates an LLM instance for a given model name string.

        Args:
            model_name_str (str): The specific model name (e.g., "gemini-pro", "gpt-4o-mini").

        Returns:
            An instance of a Langchain ChatModel.
        """
        provider = self.config.get("llm_provider", "").lower()
        backend_url = self.config.get("backend_url")
        return self._create_llm_instance(model_name_str, provider, backend_url)

    def _create_llm_instance(self, model_name_str: str, provider: str, backend_url: str):
        """
        Internal helper to create LLM instance.
        """
        if provider == "openai" or provider == "ollama" or provider == "openrouter":
            return ChatOpenAI(model=model_name_str, base_url=backend_url)
        elif provider == "anthropic":
            return ChatAnthropic(model=model_name_str, base_url=backend_url)
        elif provider == "google":
            return ChatGoogleGenerativeAI(model=model_name_str)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def get_deep_thinking_llm(self):
        """Convenience method to get the deep thinking LLM using config key."""
        key_for_model_name = self.config.get("deep_think_llm_key", "deep_think_llm")
        return self.get_llm(model_config_key=key_for_model_name)

    def get_quick_thinking_llm(self):
        """Convenience method to get the quick thinking LLM using config key."""
        key_for_model_name = self.config.get("quick_think_llm_key", "quick_think_llm")
        return self.get_llm(model_config_key=key_for_model_name)

# Example usage (for testing or if this file is run directly):
if __name__ == '__main__':
    # This is a dummy config. In the actual application,
    # this would come from default_config.py or user settings.
    dummy_config_openai = {
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4",
        "quick_think_llm": "gpt-3.5-turbo",
        "backend_url": "http://localhost:8080/v1",
        # "deep_think_llm_key": "deep_think_llm", # Not needed if we directly use the keys
        # "quick_think_llm_key": "quick_think_llm"
    }

    # factory_openai = LLMClientFactory(dummy_config_openai)
    # deep_llm_o = factory_openai.get_llm("deep_think_llm")
    # quick_llm_o = factory_openai.get_llm("quick_think_llm")
    # print(f"OpenAI Deep LLM: {type(deep_llm_o)}, Model: {deep_llm_o.model_name}")
    # print(f"OpenAI Quick LLM: {type(quick_llm_o)}, Model: {quick_llm_o.model_name}")

    # direct_llm_o = factory_openai.get_llm_by_name("gpt-4-turbo")
    # print(f"OpenAI Direct LLM: {type(direct_llm_o)}, Model: {direct_llm_o.model_name}")

    dummy_config_google = {
        "llm_provider": "google",
        "deep_think_llm": "gemini-pro",
        "quick_think_llm": "gemini-1.0-pro", # often gemini-pro is used for quick too if flash not available or for certain capabilities
        "planner_llm": "gemini-pro", # Example for a planner role
        "trader_llm": "gemini-pro",  # Example for trader
        "researcher_clarify_llm": "gemini-1.0-pro", # example for researcher
        "backend_url": None,
    }
    # factory_google = LLMClientFactory(dummy_config_google)
    # Ensure GOOGLE_API_KEY is set in your environment to run this
    # deep_llm_g = factory_google.get_llm_by_name(dummy_config_google["deep_think_llm"])
    # quick_llm_g = factory_google.get_llm_by_name(dummy_config_google["quick_think_llm"])
    # print(f"Google Deep LLM: {type(deep_llm_g)}, Model: {deep_llm_g.model}")
    # print(f"Google Quick LLM: {type(quick_llm_g)}, Model: {quick_llm_g.model}")
    pass
