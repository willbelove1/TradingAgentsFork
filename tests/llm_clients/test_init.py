import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients import get_llm_client, BaseLLMClient, GeminiClient, OpenAIClient, SUPPORTED_PROVIDERS
from tradingagents.llm_clients.gemini_client import GeminiClient as ActualGeminiClient # For isinstance checks
from tradingagents.llm_clients.openai_client import OpenAIClient as ActualOpenAIClient # For isinstance checks


@pytest.fixture(autouse=True) # Apply to all tests in this module
def mock_llm_client_constructors():
    """Mocks the constructors of actual client implementations."""
    with patch.object(ActualGeminiClient, '__init__', return_value=None) as mock_gemini_init, \
         patch.object(ActualOpenAIClient, '__init__', return_value=None) as mock_openai_init:
        # Store mocks on a temporary object or make them accessible if needed in tests,
        # though often just checking they were called (or not) is enough.
        # For this factory, we mainly care that the correct class was CHOSEN,
        # and its __init__ would be called with right params.
        yield {
            "gemini": mock_gemini_init,
            "openai": mock_openai_init
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

    def test_get_openai_client_by_provider_name(self, mock_llm_client_constructors):
        client = get_llm_client(provider_name="openai", api_key="test_key_open", config={"model": "test_oai_model"})
        assert isinstance(client, ActualOpenAIClient)
        # The factory passes the original config, and client __init__ uses it.
        # api_key from arg takes precedence if client's __init__ logic handles it (which ours do)
        mock_llm_client_constructors["openai"].assert_called_once_with(api_key="test_key_open", config={"model": "test_oai_model", "llm_provider": "openai"})

    def test_get_openai_client_by_config(self, mock_llm_client_constructors):
        config = {"llm_provider": "openai", "model": "test_oai_cfg_model"}
        client = get_llm_client(config=config) # api_key might be picked from env by OpenAIClient.__init__
        assert isinstance(client, ActualOpenAIClient)
        mock_llm_client_constructors["openai"].assert_called_once_with(api_key=None, config=config)


    def test_api_key_priority_arg_over_config(self, mock_llm_client_constructors):
        # Test that api_key argument to get_llm_client takes precedence over api_key in config
        # when passed to the client constructor.
        config = {"llm_provider": "gemini", "api_key": "config_api_key"}
        get_llm_client(provider_name="gemini", api_key="arg_api_key", config=config)
        # The client constructor receives both. Our current client __init__ prioritizes the direct api_key arg.
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key="arg_api_key", config=config)

        mock_llm_client_constructors["gemini"].reset_mock()

        # If api_key arg is None, then client constructor might pick from config (if it's designed to) or env.
        # Our get_llm_client passes api_key=None if not provided as an arg.
        # The client's __init__ then takes over.
        get_llm_client(provider_name="gemini", api_key=None, config=config)
        # The client's __init__ will receive api_key=None and the config containing "config_api_key".
        # The client's __init__ should then use "config_api_key" if api_key arg is None.
        # This specific test is more about how client __init__ uses config['api_key'],
        # which is implicitly tested by test_get_gemini_client_by_config if config has api_key.
        # The factory passes what it gets.
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key=None, config=config)


    def test_unsupported_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider: non_existent_provider"):
            get_llm_client(provider_name="non_existent_provider")

    def test_no_provider_specified_raises_value_error(self):
        with pytest.raises(ValueError, match="LLM provider name must be specified"):
            get_llm_client(config={}) # No provider_name arg, no llm_provider in config

    def test_config_is_passed_correctly(self, mock_llm_client_constructors):
        my_config = {"llm_provider": "openai", "custom_param": "value123"}
        get_llm_client(config=my_config)
        # The factory adds 'llm_provider' to the config if it was taken from provider_name arg.
        # Here, it's already in my_config.
        mock_llm_client_constructors["openai"].assert_called_once_with(api_key=None, config=my_config)

    def test_provider_name_case_insensitivity(self, mock_llm_client_constructors):
        client = get_llm_client(provider_name="GeMiNi", api_key="test_key", config={})
        assert isinstance(client, ActualGeminiClient)
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key="test_key", config={"llm_provider": "gemini"})

    def test_empty_config_with_provider_name(self, mock_llm_client_constructors):
        client = get_llm_client(provider_name="openai", api_key="test_key", config=None) # config=None
        assert isinstance(client, ActualOpenAIClient)
        mock_llm_client_constructors["openai"].assert_called_once_with(api_key="test_key", config={"llm_provider": "openai"})

    def test_get_llm_client_with_none_api_key_and_config_api_key(self, mock_llm_client_constructors):
        """
        Test that if api_key arg to get_llm_client is None, but config contains an api_key,
        the client's __init__ receives api_key=None and the config.
        The client's __init__ is then responsible for potentially using config['api_key'].
        """
        config_with_key = {"llm_provider": "gemini", "api_key": "key_from_config"}
        client = get_llm_client(api_key=None, config=config_with_key)
        assert isinstance(client, ActualGeminiClient)
        # The factory passes api_key=None. The GeminiClient.__init__ will then look at config['api_key']
        # or os.getenv if its own api_key param is None.
        mock_llm_client_constructors["gemini"].assert_called_once_with(api_key=None, config=config_with_key)

    def test_get_llm_client_with_explicit_none_api_key_and_no_config_api_key(self, mock_llm_client_constructors):
        config_no_key = {"llm_provider": "openai"}
        client = get_llm_client(api_key=None, config=config_no_key)
        assert isinstance(client, ActualOpenAIClient)
        # OpenAIClient.__init__ will receive api_key=None and then try os.getenv.
        mock_llm_client_constructors["openai"].assert_called_once_with(api_key=None, config=config_no_key)

    # Example of how to test if other clients were added to SUPPORTED_PROVIDERS
    # @patch.dict(SUPPORTED_PROVIDERS, {"new_provider": MockClient}, clear=True)
    # def test_get_new_provider(mock_new_client_constructor):
    #     MockClient = MagicMock()
    #     mock_new_client_constructor.return_value = MockClient
    #
    #     client = get_llm_client(provider_name="new_provider")
    #     assert isinstance(client, MockClient)
    #     # Check if MockClient.__init__ was called, if it was also patched.
