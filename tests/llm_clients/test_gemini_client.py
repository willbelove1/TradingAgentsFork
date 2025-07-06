import pytest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import itertools

import google.generativeai as genai
import google.api_core.exceptions as google_exceptions

from tradingagents.llm_clients.gemini_client import GeminiClient
from tradingagents.llm_clients.base_client import (
    logger as base_client_logger,
    LLMConfigurationError, LLMKeyCycleError, LLMAuthenticationError, LLMRateLimitError, LLMServiceUnavailableError, LLMResponseError
)

TEST_GEMINI_API_KEY_1 = "gemini_key_1"
TEST_GEMINI_API_KEY_2 = "gemini_key_2"
TEST_GEMINI_API_KEY_3 = "gemini_key_3"

@pytest.fixture
def mock_genai_configure():
    with patch('tradingagents.llm_clients.gemini_client.genai.configure') as mock_config:
        yield mock_config

@pytest.fixture
def mock_genai_generativemodel():
    # This mocks the class genai.GenerativeModel
    with patch('tradingagents.llm_clients.gemini_client.genai.GenerativeModel') as MockedGenerativeModel:
        instance_mock = MockedGenerativeModel.return_value # This is the mock for the instance

        mock_response = MagicMock()
        mock_response.text = "Mocked Gemini text response"
        mock_response.candidates = [MagicMock()] # Ensure candidates list is not empty
        mock_response.prompt_feedback = None
        instance_mock.generate_content.return_value = mock_response

        mock_token_count = MagicMock()
        mock_token_count.total_tokens = 10
        instance_mock.count_tokens.return_value = mock_token_count
        yield MockedGenerativeModel # Return the mocked class itself

@pytest.fixture
def mock_genai_embed_content():
    with patch('tradingagents.llm_clients.gemini_client.genai.embed_content') as mock_embed:
        mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}
        yield mock_embed

@pytest.fixture
def mock_os_getenv_gemini():
    with patch('tradingagents.llm_clients.gemini_client.os.getenv') as mock_getenv:
        yield mock_getenv

@pytest.fixture
def mock_dotenv_load_gemini():
    # Patch load_dotenv specifically in the gemini_client module
    with patch('tradingagents.llm_clients.gemini_client.load_dotenv', MagicMock()) as mock_load:
        # MagicMock so it doesn't raise ImportError if python-dotenv is not in test env,
        # and we can assert it was called.
        yield mock_load

# --- Initialization Tests ---
class TestGeminiClientInitialization:
    def test_init_with_api_keys_in_config(self, mock_genai_configure, mock_genai_generativemodel, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None # Ensure env var is not picked up initially
        config = {
            "api_keys": [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2],
            "model": "gemini-test",
            "retry_attempts": 2 # Custom retry for this test
        }
        client = GeminiClient(config=config)

        assert client.api_keys_list == [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2]
        assert isinstance(client.api_key_cycler, itertools.cycle)
        # genai.configure should be called with the first key
        mock_genai_configure.assert_called_once_with(api_key=TEST_GEMINI_API_KEY_1)
        assert client.current_api_key_for_sdk == TEST_GEMINI_API_KEY_1
        assert client.model_name == "gemini-test"
        assert client.retry_attempts == 2 # Check if retry config is picked up

    def test_init_fallback_to_single_constructor_api_key(self, mock_genai_configure, mock_genai_generativemodel, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {"model": "gemini-single-key-test"} # No api_keys list in config
        client = GeminiClient(api_key=TEST_GEMINI_API_KEY_1, config=config)

        assert client.api_keys_list == [TEST_GEMINI_API_KEY_1]
        mock_genai_configure.assert_called_once_with(api_key=TEST_GEMINI_API_KEY_1)

    def test_init_fallback_to_env_var_google_api_key(self, mock_genai_configure, mock_genai_generativemodel, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = TEST_GEMINI_API_KEY_1 # This is GOOGLE_API_KEY
        config = {} # No api_keys in config, no api_key in constructor
        client = GeminiClient(config=config)

        assert client.api_keys_list == [TEST_GEMINI_API_KEY_1]
        mock_genai_configure.assert_called_once_with(api_key=TEST_GEMINI_API_KEY_1)

    def test_init_fallback_to_dotenv_google_api_key(self, mock_genai_configure, mock_genai_generativemodel, mock_os_getenv_gemini, mock_dotenv_load_gemini):
        # os.getenv called multiple times: first for GOOGLE_API_KEY (None), then by load_dotenv, then again
        mock_os_getenv_gemini.side_effect = [None, TEST_GEMINI_API_KEY_1] # Simulate key found after dotenv load
        config = {}
        client = GeminiClient(config=config)

        mock_dotenv_load_gemini.assert_called_once()
        assert client.api_keys_list == [TEST_GEMINI_API_KEY_1]
        mock_genai_configure.assert_called_once_with(api_key=TEST_GEMINI_API_KEY_1)

    def test_init_no_api_keys_found_raises_error(self, mock_os_getenv_gemini, mock_dotenv_load_gemini):
        mock_os_getenv_gemini.return_value = None
        # Make load_dotenv not find the key by having getenv return None again after it's called
        mock_dotenv_load_gemini.side_effect = lambda: mock_os_getenv_gemini.return_value # os.getenv still None

        with pytest.raises(LLMConfigurationError, match="GeminiClient: No API keys found"):
            GeminiClient(config={})

    def test_init_configure_sdk_failure_raises_key_cycle_error(self, mock_genai_configure, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {"api_keys": [TEST_GEMINI_API_KEY_1]}
        mock_genai_configure.side_effect = Exception("SDK Configure failed")

        with pytest.raises(LLMKeyCycleError, match="Failed to configure genai SDK"):
            GeminiClient(config=config)

# --- Test _get_current_api_key_for_request (Key Cycling & SDK Configuration) ---
class TestGeminiClientKeyCycling:
    def test_get_current_api_key_cycles_and_configures_sdk(self, mock_genai_configure, mock_genai_generativemodel, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {"api_keys": [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2]}
        client = GeminiClient(config=config) # Initial configure with KEY_1

        # First call to _get_current_api_key_for_request (which is called by public methods)
        # The first key was already configured in __init__. If it's the same, no reconfigure.
        # To test cycling, we need to simulate it being called again.
        # Let's reset mock and call directly for test clarity, though usually called by BaseClient methods.
        mock_genai_configure.reset_mock()
        key1 = client._get_current_api_key_for_request() # Should be KEY_1 (still, or again if cycle is short)
        # If only one key, configure might not be called again if key is same.
        # If multiple keys, first call to _get_current_api_key_for_request after init *might* reconfigure with key1 if current_api_key_for_sdk was somehow reset,
        # or it might get key2. Let's check the sequence.

        # After __init__ configures with KEY_1:
        assert client.current_api_key_for_sdk == TEST_GEMINI_API_KEY_1

        # 1st explicit call for a request (simulating first attempt of a retryable operation)
        # _get_current_api_key_for_request will call _configure_sdk_with_next_key
        # which will advance the cycler.
        # If list is [K1, K2], cycler gives K1, then K2, then K1 ...
        # __init__ configures K1. current_api_key_for_sdk = K1
        # Call 1: _get_current_api_key_for_request -> _configure_sdk_with_next_key -> next(cycler) is K2. Configure K2. Returns K2.
        key_for_attempt_1 = client._get_current_api_key_for_request()
        assert key_for_attempt_1 == TEST_GEMINI_API_KEY_2
        mock_genai_configure.assert_called_with(api_key=TEST_GEMINI_API_KEY_2)

        # Call 2: (simulating second attempt)
        key_for_attempt_2 = client._get_current_api_key_for_request()
        assert key_for_attempt_2 == TEST_GEMINI_API_KEY_1 # Cycles back
        mock_genai_configure.assert_called_with(api_key=TEST_GEMINI_API_KEY_1)

    def test_configure_sdk_failure_in_get_key_raises_llm_key_cycle_error(self, mock_genai_configure, mock_genai_generativemodel, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {"api_keys": [TEST_GEMINI_API_KEY_1]}
        client = GeminiClient(config=config) # Configures with KEY_1

        mock_genai_configure.reset_mock()
        mock_genai_configure.side_effect = Exception("SDK Configure failed during get_key")

        with pytest.raises(LLMKeyCycleError, match="Failed to configure genai SDK"):
            client._get_current_api_key_for_request()


# --- Test Actual API Call Methods (via BaseLLMClient public methods) ---
# We will mock the _impl methods of GeminiClient
class TestGeminiClientApiMethods:

    @pytest.fixture
    def client_multi_key(self, mock_genai_configure, mock_genai_generativemodel, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None # Ensure keys only from config
        config = {
            "api_keys": [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2, TEST_GEMINI_API_KEY_3],
            "model": "gemini-pro",
            "retry_attempts": 3, # Allow enough retries for key cycling
            "retry_min_wait_seconds": 0.01, # Fast retries for testing
            "retry_max_wait_seconds": 0.02,
        }
        # mock_genai_generativemodel is already patching genai.GenerativeModel class
        # Its instance mock is what _generate_text_impl will use
        return GeminiClient(config=config)

    def test_generate_text_success_first_key(self, client_multi_key, mock_genai_generativemodel, caplog):
        # The _generate_text_impl will be called by BaseClient's generate_text after retry decorator
        # We mock _generate_text_impl to control its behavior for testing retry

        # Get the mocked genai.GenerativeModel instance
        mock_model_instance = mock_genai_generativemodel.return_value
        mock_model_instance.generate_content.return_value = MagicMock(text="Success with key1", candidates=[MagicMock()], prompt_feedback=None)
        mock_model_instance.count_tokens.return_value = MagicMock(total_tokens=5)

        # Patch the _generate_text_impl of the client instance
        with patch.object(client_multi_key, '_generate_text_impl',
                          wraps=client_multi_key._generate_text_impl) as mock_impl:

            response = client_multi_key.generate_text("test prompt")
            assert response == "Success with key1"

            # Check that _impl was called once
            mock_impl.assert_called_once()
            # Check that _get_current_api_key_for_request was called (implicitly configuring SDK with KEY_1)
            # The key used by _impl is passed as current_api_key_for_request
            assert mock_impl.call_args[1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_1

            # Check logging for key usage (BaseClient's finally block)
            assert any(f"generate_text call for model gemini-pro (Key: HIDDEN) completed." in record.message for record in caplog.records)


    def test_generate_text_key_cycle_on_permission_denied(self, client_multi_key, mock_genai_generativemodel, caplog):
        mock_model_instance = mock_genai_generativemodel.return_value

        # Simulate KEY_1 fails with PermissionDenied, KEY_2 succeeds
        key1_error = google_exceptions.PermissionDenied("Key 1 invalid")
        key2_success_response = MagicMock(text="Success with key2", candidates=[MagicMock()], prompt_feedback=None)

        # Mock _generate_text_impl to simulate this behavior
        # It needs to know which key is being tried. It gets current_api_key_for_request.
        def impl_side_effect(prompt, temperature, max_tokens, model, current_api_key_for_request):
            if current_api_key_for_request == TEST_GEMINI_API_KEY_1:
                # This maps to LLMKeyCycleError in the actual _generate_text_impl
                raise google_exceptions.PermissionDenied("Key 1 invalid")
            elif current_api_key_for_request == TEST_GEMINI_API_KEY_2:
                mock_model_instance.generate_content.return_value = key2_success_response # Configure success for key2
                # Call the original _generate_text_impl for success path with key2
                # This is tricky because the original _impl is what we are mocking.
                # For this test, let's simplify: if key is KEY_2, return success directly.
                return "Success with key2"
            raise AssertionError(f"Unexpected key in _impl_side_effect: {current_api_key_for_request}")

        # We need to patch the *actual* _generate_text_impl inside GeminiClient for this test
        # because the retry decorator in BaseLLMClient calls it.
        with patch.object(GeminiClient, '_generate_text_impl', side_effect=impl_side_effect) as mock_actual_impl:
            response = client_multi_key.generate_text("test prompt cycle")

        assert response == "Success with key2"
        # _generate_text_impl (the patched one) should have been called twice: once for KEY_1 (failed), once for KEY_2 (succeeded)
        assert mock_actual_impl.call_count == 2

        # Check calls with correct keys
        call_args_list = mock_actual_impl.call_args_list
        assert call_args_list[0][1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_1
        assert call_args_list[1][1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_2

        # Check retry log
        assert any(f"Retrying LLM API call: _try_generate (Key: {TEST_GEMINI_API_KEY_1})" in record.message for record in caplog.records if "LLMKeyCycleError" in record.message)
        assert any(f"generate_text call for model gemini-pro (Key: HIDDEN) completed." in record.message for record in caplog.records) # For the successful call with KEY_2


    def test_generate_text_all_keys_fail(self, client_multi_key, mock_genai_generativemodel, caplog):
        # Simulate all keys failing with a retriable (key cycle) error
        def impl_side_effect_all_fail(prompt, temperature, max_tokens, model, current_api_key_for_request):
            # Simulate an error that GeminiClient's _generate_text_impl would map to LLMKeyCycleError
            raise google_exceptions.ResourceExhausted(f"Rate limit on key {current_api_key_for_request}")

        with patch.object(GeminiClient, '_generate_text_impl', side_effect=impl_side_effect_all_fail) as mock_actual_impl:
            with pytest.raises(LLMKeyCycleError): # Tenacity re-raises the last exception after exhausting retries
                client_multi_key.generate_text("test all fail")

        # Should be called for each key (3 keys) * retry_attempts (3 for BaseClient default, but here client_multi_key has 3)
        # No, tenacity stop is on the outer call. So it will try _get_current_api_key which cycles.
        # Total attempts for generate_text is client_multi_key.retry_attempts (which is 3).
        # In each attempt, _get_current_api_key_for_request is called.
        assert mock_actual_impl.call_count == client_multi_key.retry_attempts # Should be 3 attempts

        # Check that different keys were tried
        tried_keys = {call_args[1]['current_api_key_for_request'] for call_args in mock_actual_impl.call_args_list}
        assert tried_keys == {TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2, TEST_GEMINI_API_KEY_3}

        assert any(f"Retrying LLM API call: _try_generate (Key: {TEST_GEMINI_API_KEY_1})" in record.message for record in caplog.records if "LLMKeyCycleError" in record.message)
        assert any(f"Retrying LLM API call: _try_generate (Key: {TEST_GEMINI_API_KEY_2})" in record.message for record in caplog.records if "LLMKeyCycleError" in record.message)
        # The third failure will be re-raised, not logged as "Retrying" by tenacity's before_sleep.

    # Similar tests should be written for get_embedding
    def test_get_embedding_key_cycle_on_error(self, client_multi_key, mock_genai_embed_content, caplog):
        # This test will be simpler as mock_genai_embed_content is already patching the SDK call
        # We need _get_embedding_impl to raise LLMKeyCycleError for specific keys

        def impl_side_effect_embed(text, model, current_api_key_for_request):
            if current_api_key_for_request == TEST_GEMINI_API_KEY_1:
                raise google_exceptions.PermissionDenied("Embed Key 1 invalid") # Mapped to LLMKeyCycleError by actual _impl
            elif current_api_key_for_request == TEST_GEMINI_API_KEY_2:
                # Simulate success for key 2 by not raising and letting mock_genai_embed_content work
                # The actual _get_embedding_impl would call genai.embed_content here
                return genai.embed_content(model=model, content=text) # This uses the patched version
            raise AssertionError("Unexpected key for embedding")

        with patch.object(GeminiClient, '_get_embedding_impl', side_effect=impl_side_effect_embed) as mock_actual_impl:
            embedding = client_multi_key.get_embedding("embed this")

        assert embedding == [0.1, 0.2, 0.3] # From mock_genai_embed_content
        assert mock_actual_impl.call_count == 2
        call_args_list = mock_actual_impl.call_args_list
        assert call_args_list[0][1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_1
        assert call_args_list[1][1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_2
        assert any(f"Retrying LLM API call: _try_embed (Key: {TEST_GEMINI_API_KEY_1})" in record.message for record in caplog.records if "LLMKeyCycleError" in record.message)

    def test_generate_text_no_candidates_response(self, client_multi_key, mock_genai_generativemodel, caplog):
        mock_model_instance = mock_genai_generativemodel.return_value
        # Simulate response with no candidates (e.g., safety filtered)
        mock_response_no_candidates = MagicMock()
        mock_response_no_candidates.candidates = []
        mock_response_no_candidates.prompt_feedback = MagicMock(block_reason=MagicMock(name="SAFETY"))

        def impl_side_effect_no_candidates(prompt, temperature, max_tokens, model, current_api_key_for_request):
            # This should call the actual genai.GenerativeModel().generate_content() which is mocked
            # We need the instance mock to return our special response
            mock_model_instance.generate_content.return_value = mock_response_no_candidates
            # Now, call the *original* _generate_text_impl to let it handle this response
            # This requires careful patching if we are not calling the original.
            # For simplicity, let's assume the actual _impl would raise LLMResponseError here.
            raise LLMResponseError("Prompt blocked by Gemini safety filters: SAFETY")

        with patch.object(GeminiClient, '_generate_text_impl', side_effect=impl_side_effect_no_candidates) as mock_actual_impl:
            with pytest.raises(LLMResponseError, match="Prompt blocked by Gemini safety filters: SAFETY"):
                client_multi_key.generate_text("a risky prompt")

        assert mock_actual_impl.call_count == 1 # No retry for LLMResponseError by default
        assert any("Prompt blocked for model gemini-pro due to: SAFETY" in record.message for record in caplog.records if record.levelname == "ERROR")
