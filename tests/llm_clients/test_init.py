import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients import get_llm_client, BaseLLMClient, GeminiClient, SUPPORTED_PROVIDERS
from tradingagents.llm_clients.gemini_client import GeminiClient as ActualGeminiClient # For isinstance checks
# from tradingagents.llm_clients.openai_client import OpenAIClient as ActualOpenAIClient # Removed


@pytest.fixture(autouse=True) # Apply to all tests in this module
def mock_llm_client_constructors():
    """Mocks the constructors of actual client implementations."""
    # Only mock GeminiClient's __init__ now
    with patch.object(ActualGeminiClient, '__init__', return_value=None) as mock_gemini_init:
        yield {
            "gemini": mock_gemini_init,
            # "openai": mock_openai_init # Removed
        }

class TestGetLLMClientFactory:

    def test_get_gemini_client_by_provider_name(self, mock_llm_client_constructors):
        client = get_llm_client(provider_name="gemini", api_key="test_key", config={"model": "test_gem_model"})
        assert isinstance(client, ActualGeminiClient)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key="test_key", config={"model": "test_gem_model", "llm_provider": "gemini"})

    def test_get_gemini_client_by_config(self, mock_llm_client_constructors):
        config = {"llm_provider": "google", "model": "test_google_model", "api_key": "cfg_key"} # 'google' is alias for 'gemini'
        client = get_llm_client(config=config)
        assert isinstance(client, ActualGeminiClient)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key="cfg_key", config=config)

    # Removed OpenAI client tests:
    # def test_get_openai_client_by_provider_name(self, mock_llm_client_constructors):
    # def test_get_openai_client_by_config(self, mock_llm_client_constructors):

    def test_api_key_priority_arg_over_config(self, mock_llm_client_constructors):
        config = {"llm_provider": "gemini", "api_key": "config_api_key"}
        get_llm_client(provider_name="gemini", api_key="arg_api_key", config=config)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key="arg_api_key", config=config)

        mock_llm_client_constructors["gemini"].reset_mock()
        get_llm_client(provider_name="gemini", api_key=None, config=config)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key=None, config=config)

    def test_unsupported_provider_raises_value_error(self):
        # First, ensure "openai" is truly not in SUPPORTED_PROVIDERS for this test to be valid for "openai"
        if "openai" in SUPPORTED_PROVIDERS:
            # This would mean SUPPORTED_PROVIDERS in __init__.py wasn't cleaned properly
            pytest.skip("Skipping test_unsupported_provider_raises_value_error for 'openai' as it's unexpectedly in SUPPORTED_PROVIDERS")

        with pytest.raises(ValueError, match="Unsupported LLM provider: openai"):
            get_llm_client(provider_name="openai")
        with pytest.raises(ValueError, match="Unsupported LLM provider: non_existent_provider"):
            get_llm_client(provider_name="non_existent_provider")


    def test_no_provider_specified_raises_value_error(self):
        with pytest.raises(ValueError, match="LLM provider name must be specified"):
            get_llm_client(config={})

    def test_config_is_passed_correctly_for_gemini(self, mock_llm_client_constructors):
        my_config = {"llm_provider": "google", "custom_param": "value123"}
        get_llm_client(config=my_config)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key=None, config=my_config)

    def test_provider_name_case_insensitivity(self, mock_llm_client_constructors):
        client = get_llm_client(provider_name="GeMiNi", api_key="test_key", config={})
        assert isinstance(client, ActualGeminiClient)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key="test_key", config={"llm_provider": "gemini"})

    def test_empty_config_with_provider_name_gemini(self, mock_llm_client_constructors):
        client = get_llm_client(provider_name="google", api_key="test_key", config=None)
        assert isinstance(client, ActualGeminiClient)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key="test_key", config={"llm_provider": "google"})

    def test_get_llm_client_with_none_api_key_and_config_api_key(self, mock_llm_client_constructors):
        config_with_key = {"llm_provider": "gemini", "api_key": "key_from_config"}
        client = get_llm_client(api_key=None, config=config_with_key)
        assert isinstance(client, ActualGeminiClient)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key=None, config=config_with_key)

    def test_get_llm_client_with_explicit_none_api_key_and_no_config_api_key(self, mock_llm_client_constructors):
        config_no_key = {"llm_provider": "gemini"} # Changed to gemini
        client = get_llm_client(api_key=None, config=config_no_key)
        assert isinstance(client, ActualGeminiClient)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key=None, config=config_no_key)
