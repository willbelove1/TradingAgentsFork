import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
from openai import APIError, RateLimitError, AuthenticationError # For actual exception types

from tradingagents.llm_clients.openai_client import OpenAIClient
from tradingagents.llm_clients.base_client import logger as base_client_logger

TEST_OPENAI_API_KEY = "test_openai_api_key"

@pytest.fixture
def mock_openai_sdk_client():
    """Fixture for a mocked OpenAI SDK client instance."""
    sdk_mock = MagicMock() # This will be the instance of OpenAIClientSDK

    # Mock for chat.completions.create
    mock_chat_completion = MagicMock()
    mock_chat_choice = MagicMock()
    mock_chat_choice.message = MagicMock(role="assistant", content="Mocked OpenAI text response")
    mock_chat_completion.choices = [mock_chat_choice]
    mock_chat_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    sdk_mock.chat.completions.create.return_value = mock_chat_completion

    # Mock for embeddings.create
    mock_embedding_response = MagicMock()
    mock_embedding_data = MagicMock(embedding=[0.5, 0.4, 0.3, 0.2, 0.1])
    mock_embedding_response.data = [mock_embedding_data]
    mock_embedding_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)
    sdk_mock.embeddings.create.return_value = mock_embedding_response

    return sdk_mock

@pytest.fixture
def mock_openai_sdk_constructor(mock_openai_sdk_client):
    """Fixture for mocking the OpenAI SDK constructor."""
    with patch('tradingagents.llm_clients.openai_client.OpenAIClientSDK', return_value=mock_openai_sdk_client) as mock_constructor:
        yield mock_constructor

@pytest.fixture
def mock_os_getenv_openai():
    with patch('tradingagents.llm_clients.openai_client.os.getenv') as mock_getenv:
        yield mock_getenv

@pytest.fixture
def mock_dotenv_load_openai():
    with patch('tradingagents.llm_clients.openai_client.load_dotenv') as mock_load:
        yield mock_load

class TestOpenAIClientInitialization:
    def test_init_with_api_key_arg(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = None
        client = OpenAIClient(api_key=TEST_OPENAI_API_KEY, config={"model": "gpt-test"})
        mock_openai_sdk_constructor.assert_called_once_with(api_key=TEST_OPENAI_API_KEY, base_url=None)
        assert client.model_name == "gpt-test"

    def test_init_with_env_var(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = TEST_OPENAI_API_KEY
        client = OpenAIClient(config={"model": "gpt-test-env"})
        mock_openai_sdk_constructor.assert_called_once_with(api_key=TEST_OPENAI_API_KEY, base_url=None)

    def test_init_with_dotenv(self, mock_openai_sdk_constructor, mock_os_getenv_openai, mock_dotenv_load_openai):
        mock_os_getenv_openai.side_effect = [None, TEST_OPENAI_API_KEY]
        client = OpenAIClient(config={"model": "gpt-test-dotenv"})
        mock_dotenv_load_openai.assert_called_once()
        mock_openai_sdk_constructor.assert_called_once_with(api_key=TEST_OPENAI_API_KEY, base_url=None)

    def test_init_no_api_key_with_no_base_url_logs_warning(self, mock_openai_sdk_constructor, mock_os_getenv_openai, mock_dotenv_load_openai, caplog):
        mock_os_getenv_openai.return_value = None
        mock_dotenv_load_openai.side_effect = ImportError # Simulate dotenv not installed
        OpenAIClient() # Should not raise ValueError, but log a warning
        assert any("OPENAI_API_KEY not found" in record.message for record in caplog.records if record.levelname == "WARNING")

    def test_init_with_base_url(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = None # API key can be None if base_url is for local server
        config = {"base_url": "http://localhost:1234/v1", "model": "local-model"}
        client = OpenAIClient(config=config)
        mock_openai_sdk_constructor.assert_called_once_with(api_key=None, base_url="http://localhost:1234/v1")
        assert client.base_url == "http://localhost:1234/v1"

    def test_init_default_models(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = TEST_OPENAI_API_KEY
        client = OpenAIClient()
        assert client.model_name == OpenAIClient.DEFAULT_TEXT_MODEL
        assert client.embedding_model_name == OpenAIClient.DEFAULT_EMBEDDING_MODEL

    def test_init_models_from_config(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = TEST_OPENAI_API_KEY
        config = {"model": "custom-gpt-model", "embedding_model": "custom-openai-embedding"}
        client = OpenAIClient(config=config)
        assert client.model_name == "custom-gpt-model"
        assert client.embedding_model_name == "custom-openai-embedding"

    def test_init_sdk_failure(self, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = TEST_OPENAI_API_KEY
        with patch('tradingagents.llm_clients.openai_client.OpenAIClientSDK', side_effect=RuntimeError("SDK Init failed")):
            with pytest.raises(RuntimeError, match="SDK Init failed"):
                 OpenAIClient()


class TestOpenAIClientGenerateText:
    @pytest.fixture
    def client(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = TEST_OPENAI_API_KEY
        return OpenAIClient(config={'model': 'gpt-3.5-turbo'})

    def test_generate_text_success(self, client, mock_openai_sdk_client, caplog):
        prompt = "Hello OpenAI!"
        expected_response = "Mocked OpenAI text response"

        response = client.generate_text(prompt, temperature=0.6, max_tokens=150)

        assert response == expected_response
        mock_openai_sdk_client.chat.completions.create.assert_called_once_with(
            model='gpt-3.5-turbo',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=150
        )
        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("generate_text called with model gpt-3.5-turbo. Duration:" in msg for msg in log_records)
        assert any("Token Usage - Model: gpt-3.5-turbo, Prompt Tokens: 10, Completion Tokens: 20, Total Tokens: 30" in msg for msg in log_records)

    def test_generate_text_with_override_model(self, client, mock_openai_sdk_client, caplog):
        prompt = "Test override"
        client.generate_text(prompt, model="gpt-4-override")
        mock_openai_sdk_client.chat.completions.create.assert_called_once_with(
            model='gpt-4-override', messages=pytest.ANY, temperature=pytest.ANY, max_tokens=pytest.ANY
        )
        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("generate_text called with model gpt-4-override. Duration:" in msg for msg in log_records)
        assert any("Token Usage - Model: gpt-4-override" in msg for msg in log_records)


    def test_generate_text_api_error(self, client, mock_openai_sdk_client):
        mock_openai_sdk_client.chat.completions.create.side_effect = APIError("msg", response=MagicMock(), body=None)
        with pytest.raises(APIError):
            client.generate_text("prompt")

    def test_generate_text_rate_limit_error(self, client, mock_openai_sdk_client):
        mock_openai_sdk_client.chat.completions.create.side_effect = RateLimitError("rate limited", response=MagicMock(), body=None)
        with pytest.raises(RateLimitError):
            client.generate_text("prompt")


class TestOpenAIClientGetEmbedding:
    @pytest.fixture
    def client(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = TEST_OPENAI_API_KEY
        return OpenAIClient(config={'embedding_model': 'text-embedding-ada-002'})

    def test_get_embedding_success(self, client, mock_openai_sdk_client, caplog):
        text_to_embed = "Embed this with OpenAI!"
        expected_embedding = [0.5, 0.4, 0.3, 0.2, 0.1]

        embedding = client.get_embedding(text_to_embed)

        assert embedding == expected_embedding
        mock_openai_sdk_client.embeddings.create.assert_called_once_with(
            input=[text_to_embed.replace("\n", " ")], model='text-embedding-ada-002'
        )
        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("get_embedding called for model text-embedding-ada-002. Duration:" in msg for msg in log_records)
        assert any("Token Usage - Model: text-embedding-ada-002, Prompt Tokens: 5, Total Tokens: 5" in msg for msg in log_records)

    def test_get_embedding_with_override_model(self, client, mock_openai_sdk_client, caplog):
        text_to_embed = "Override embedding model!"
        client.get_embedding(text_to_embed, model="text-embedding-custom")
        mock_openai_sdk_client.embeddings.create.assert_called_once_with(
            input=[text_to_embed.replace("\n", " ")], model="text-embedding-custom"
        )
        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("get_embedding called for model text-embedding-custom. Duration:" in msg for msg in log_records)

    def test_get_embedding_empty_text_returns_zero_vector(self, client, mock_openai_sdk_client, caplog):
        embedding = client.get_embedding(" ") # Empty or whitespace only
        assert embedding == [0.0] * 1536 # Default zero vector size used in client
        mock_openai_sdk_client.embeddings.create.assert_not_called()
        assert any("Attempted to get embedding for empty or whitespace-only text" in record.message for record in caplog.records if record.levelname == "WARNING")


    def test_get_embedding_api_error(self, client, mock_openai_sdk_client):
        mock_openai_sdk_client.embeddings.create.side_effect = APIError("embedding error", response=MagicMock(), body=None)
        with pytest.raises(APIError):
            client.get_embedding("text")

class TestOpenAIClientChat:
    @pytest.fixture
    def client(self, mock_openai_sdk_constructor, mock_os_getenv_openai):
        mock_os_getenv_openai.return_value = TEST_OPENAI_API_KEY
        return OpenAIClient(config={'model': 'gpt-3.5-turbo-chat'})

    def test_chat_success(self, client, mock_openai_sdk_client, caplog):
        messages = [{"role": "user", "content": "Hello from chat!"}]
        expected_response_content = "Mocked OpenAI text response"

        response = client.chat(messages, temperature=0.4, max_tokens=120)

        assert response["role"] == "assistant"
        assert response["content"] == expected_response_content
        mock_openai_sdk_client.chat.completions.create.assert_called_once_with(
            model='gpt-3.5-turbo-chat',
            messages=messages,
            temperature=0.4,
            max_tokens=120
        )
        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("chat called with model gpt-3.5-turbo-chat. Duration:" in msg for msg in log_records)
        assert any("Token Usage - Model: gpt-3.5-turbo-chat, Prompt Tokens: 10, Completion Tokens: 20, Total Tokens: 30" in msg for msg in log_records)

    def test_chat_api_error_returns_empty_fallback(self, client, mock_openai_sdk_client):
        mock_openai_sdk_client.chat.completions.create.side_effect = APIError("chat API error", response=MagicMock(), body=None)
        # The client's chat method catches the error and returns a fallback
        response = client.chat([{"role": "user", "content": "test"}])
        assert response == {"role": "assistant", "content": ""} # Check fallback
        # To assert that an error was logged, you'd use caplog if needed.
        # with pytest.raises(APIError): # This would fail as the client handles it
        #     client.chat([{"role": "user", "content": "test"}])

    def test_chat_with_override_model(self, client, mock_openai_sdk_client, caplog):
        messages = [{"role": "user", "content": "Chat override"}]
        client.chat(messages, model="gpt-4-chat-override")
        mock_openai_sdk_client.chat.completions.create.assert_called_once_with(
            model='gpt-4-chat-override', messages=pytest.ANY, temperature=pytest.ANY, max_tokens=pytest.ANY
        )
        log_records = [r.message for r in caplog.records if r.name == base_client_logger.name]
        assert any("chat called with model gpt-4-chat-override. Duration:" in msg for msg in log_records)
        assert any("Token Usage - Model: gpt-4-chat-override" in msg for msg in log_records)
