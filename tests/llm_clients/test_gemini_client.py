import pytest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import google.generativeai as genai # For actual exception types

from tradingagents.llm_clients.gemini_client import GeminiClient
from tradingagents.llm_clients.base_client import logger as base_client_logger # To check log messages

# Sample API key for testing
TEST_API_KEY = "test_gemini_api_key"

@pytest.fixture
def mock_generative_model():
    """Fixture for a mocked GenerativeModel instance."""
    model_mock = MagicMock(spec=genai.GenerativeModel)
    # Mock generate_content response
    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini text response"
    model_mock.generate_content.return_value = mock_response
    # Mock count_tokens response
    mock_token_count = MagicMock()
    mock_token_count.total_tokens = 10
    model_mock.count_tokens.return_value = mock_token_count
    return model_mock

@pytest.fixture
def mock_embed_content():
    """Fixture for a mocked embed_content function."""
    with patch('tradingagents.llm_clients.gemini_client.genai.embed_content') as mock_embed:
        mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]}
        yield mock_embed

@pytest.fixture
def mock_genai_configure():
    """Fixture for mocking genai.configure."""
    with patch('tradingagents.llm_clients.gemini_client.genai.configure') as mock_config:
        yield mock_config

@pytest.fixture
def mock_genai_generativemodel_init(mock_generative_model):
    """Fixture for mocking GenerativeModel constructor."""
    with patch('tradingagents.llm_clients.gemini_client.genai.GenerativeModel', return_value=mock_generative_model) as mock_constructor:
        yield mock_constructor

@pytest.fixture
def mock_os_getenv():
    with patch('tradingagents.llm_clients.gemini_client.os.getenv') as mock_getenv:
        yield mock_getenv

@pytest.fixture
def mock_dotenv_load():
    with patch('tradingagents.llm_clients.gemini_client.load_dotenv') as mock_load:
        yield mock_load


class TestGeminiClientInitialization:
    def test_init_with_api_key_arg(self, mock_genai_configure, mock_genai_generativemodel_init, mock_os_getenv):
        mock_os_getenv.return_value = None # Ensure env var is not picked up
        client = GeminiClient(api_key=TEST_API_KEY, config={"model": "gemini-test"})
        mock_genai_configure.assert_called_once_with(api_key=TEST_API_KEY)
        mock_genai_generativemodel_init.assert_called_once_with("gemini-test")
        assert client.model_name == "gemini-test"

    def test_init_with_env_var(self, mock_genai_configure, mock_genai_generativemodel_init, mock_os_getenv):
        mock_os_getenv.return_value = TEST_API_KEY
        client = GeminiClient(config={"model": "gemini-test-env"})
        mock_genai_configure.assert_called_once_with(api_key=TEST_API_KEY)
        mock_genai_generativemodel_init.assert_called_once_with("gemini-test-env")

    def test_init_with_dotenv(self, mock_genai_configure, mock_genai_generativemodel_init, mock_os_getenv, mock_dotenv_load):
        mock_os_getenv.side_effect = [None, TEST_API_KEY] # First call no env, second call after dotenv load has key
        client = GeminiClient(config={"model": "gemini-test-dotenv"})
        mock_dotenv_load.assert_called_once()
        mock_genai_configure.assert_called_once_with(api_key=TEST_API_KEY)
        mock_genai_generativemodel_init.assert_called_once_with("gemini-test-dotenv")

    def test_init_no_api_key_raises_value_error(self, mock_os_getenv, mock_dotenv_load):
        mock_os_getenv.return_value = None
        mock_dotenv_load.side_effect = ImportError # Simulate dotenv not installed or failing
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
            GeminiClient()

    def test_init_default_models(self, mock_genai_configure, mock_genai_generativemodel_init, mock_os_getenv):
        mock_os_getenv.return_value = TEST_API_KEY
        client = GeminiClient()
        assert client.model_name == GeminiClient.DEFAULT_TEXT_MODEL
        assert client.embedding_model_name == GeminiClient.DEFAULT_EMBEDDING_MODEL
        mock_genai_generativemodel_init.assert_called_once_with(GeminiClient.DEFAULT_TEXT_MODEL)

    def test_init_models_from_config(self, mock_genai_configure, mock_genai_generativemodel_init, mock_os_getenv):
        mock_os_getenv.return_value = TEST_API_KEY
        config = {"model": "custom-text-model", "embedding_model": "custom-embedding-model"}
        client = GeminiClient(config=config)
        assert client.model_name == "custom-text-model"
        assert client.embedding_model_name == "custom-embedding-model"
        mock_genai_generativemodel_init.assert_called_once_with("custom-text-model")

    def test_init_generative_model_failure(self, mock_genai_configure, mock_os_getenv):
        mock_os_getenv.return_value = TEST_API_KEY
        with patch('tradingagents.llm_clients.gemini_client.genai.GenerativeModel', side_effect=RuntimeError("Init failed")):
            with pytest.raises(RuntimeError, match="Init failed"): # Check if it re-raises
                 GeminiClient()


class TestGeminiClientGenerateText:
    @pytest.fixture
    def client(self, mock_genai_configure, mock_genai_generativemodel_init, mock_os_getenv):
        mock_os_getenv.return_value = TEST_API_KEY
        # mock_genai_generativemodel_init is already patching the constructor to return mock_generative_model
        return GeminiClient(config={'model': 'gemini-pro'})

    def test_generate_text_success(self, client, mock_generative_model, caplog):
        prompt = "Hello Gemini!"
        expected_response = "Mocked Gemini text response"

        with patch.object(base_client_logger, 'info') as mock_log_info:
            response = client.generate_text(prompt, temperature=0.5, max_tokens=100)

        assert response == expected_response
        mock_generative_model.generate_content.assert_called_once()
        call_args = mock_generative_model.generate_content.call_args
        assert call_args[0][0] == prompt # first positional arg is prompt
        gen_config = call_args[1]['generation_config'] # generation_config is a kwarg
        assert gen_config.temperature == 0.5
        assert gen_config.max_output_tokens == 100

        # Check logging
        # Example: Check if the duration log and token usage log appeared
        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name] # Filter by logger name
        assert any("generate_text called with model gemini-pro. Duration:" in msg for msg in log_records)
        assert any("Token Usage - Model: gemini-pro, Prompt Tokens: 10" in msg for msg in log_records)

    def test_generate_text_with_override_model(self, client, mock_generative_model, mock_genai_generativemodel_init, caplog):
        # This test needs to ensure that when a new model is specified, a new GenerativeModel instance is created for it.
        new_model_instance_mock = MagicMock(spec=genai.GenerativeModel)
        new_model_response = MagicMock()
        new_model_response.text = "Response from overridden model"
        new_model_instance_mock.generate_content.return_value = new_model_response
        new_model_instance_mock.count_tokens.return_value = MagicMock(total_tokens=5)

        # Patch genai.GenerativeModel to return the new mock when called with "override-model"
        def side_effect_constructor(model_name_constructor):
            if model_name_constructor == "override-model":
                return new_model_instance_mock
            return mock_generative_model # Default client's model

        mock_genai_generativemodel_init.side_effect = side_effect_constructor

        prompt = "Test override"
        response = client.generate_text(prompt, model="override-model")

        assert response == "Response from overridden model"
        new_model_instance_mock.generate_content.assert_called_once_with(prompt, generation_config=pytest.ANY)
        mock_generative_model.generate_content.assert_not_called() # Ensure original model mock wasn't called

        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("generate_text called with model override-model. Duration:" in msg for msg in log_records)
        assert any("Token Usage - Model: override-model, Prompt Tokens: 5" in msg for msg in log_records)


    def test_generate_text_api_error(self, client, mock_generative_model):
        mock_generative_model.generate_content.side_effect = genai.types.generation_types.StopCandidateException("API error")
        with pytest.raises(genai.types.generation_types.StopCandidateException, match="API error"):
            client.generate_text("prompt")

class TestGeminiClientGetEmbedding:
    @pytest.fixture
    def client(self, mock_genai_configure, mock_genai_generativemodel_init, mock_os_getenv):
        mock_os_getenv.return_value = TEST_API_KEY
        return GeminiClient(config={'embedding_model': 'custom-embed-model'})

    def test_get_embedding_success(self, client, mock_embed_content, caplog):
        text_to_embed = "Embed this!"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        embedding = client.get_embedding(text_to_embed)

        assert embedding == expected_embedding
        mock_embed_content.assert_called_once_with(model='custom-embed-model', content=text_to_embed)

        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("get_embedding called for model custom-embed-model. Duration:" in msg for msg in log_records)

    def test_get_embedding_with_override_model(self, client, mock_embed_content, caplog):
        text_to_embed = "Embed this with override!"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] # mock_embed_content always returns this

        embedding = client.get_embedding(text_to_embed, model="override-embed-model")

        assert embedding == expected_embedding
        mock_embed_content.assert_called_once_with(model='override-embed-model', content=text_to_embed)

        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("get_embedding called for model override-embed-model. Duration:" in msg for msg in log_records)

    def test_get_embedding_api_error(self, client, mock_embed_content):
        mock_embed_content.side_effect = RuntimeError("Embedding API error")
        with pytest.raises(RuntimeError, match="Embedding API error"):
            client.get_embedding("text")

# Placeholder for Chat tests if implemented
# class TestGeminiClientChat:
#     pass
