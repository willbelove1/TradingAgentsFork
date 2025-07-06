import pytest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import itertools
import time # For testing delays if implemented

import google.generativeai as genai
import google.api_core.exceptions as google_exceptions

from tradingagents.llm_clients.gemini_client import GeminiClient
from tradingagents.llm_clients.base_client import (
    logger as base_client_logger,
    LLMConfigurationError, LLMKeyCycleError, LLMAuthenticationError,
    LLMRateLimitError, LLMServiceUnavailableError, LLMResponseError,
    LLMTransientError # Ensure this is imported if used directly
)

TEST_GEMINI_API_KEY_1 = "gemini_key_1"
TEST_GEMINI_API_KEY_2 = "gemini_key_2"
TEST_GEMINI_API_KEY_3 = "gemini_key_3"

# --- Fixtures ---
@pytest.fixture
def mock_genai_configure():
    with patch('tradingagents.llm_clients.gemini_client.genai.configure') as mock_config:
        yield mock_config

@pytest.fixture
def mock_generative_model_instance(): # Mocks an *instance* of GenerativeModel
    model_instance_mock = MagicMock(spec=genai.GenerativeModel)
    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini text response"
    mock_response.candidates = [MagicMock()]
    mock_response.prompt_feedback = None
    model_instance_mock.generate_content.return_value = mock_response

    mock_token_count = MagicMock(total_tokens=10)
    model_instance_mock.count_tokens.return_value = mock_token_count
    return model_instance_mock

@pytest.fixture
def mock_genai_generativemodel_class(mock_generative_model_instance): # Mocks the GenerativeModel class
    with patch('tradingagents.llm_clients.gemini_client.genai.GenerativeModel', return_value=mock_generative_model_instance) as MockedClass:
        yield MockedClass

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
    with patch('tradingagents.llm_clients.gemini_client.load_dotenv', MagicMock()) as mock_load:
        yield mock_load

# --- Initialization Tests (Largely unchanged from previous version) ---
class TestGeminiClientInitialization:
    def test_init_with_api_keys_in_config(self, mock_genai_configure, mock_genai_generativemodel_class, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {
            "api_keys": [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2],
            "model": "gemini-test",
            "retry_attempts": 2
        }
        client = GeminiClient(config=config)
        assert client.api_keys_list == [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2]
        mock_genai_configure.assert_called_once_with(api_key=TEST_GEMINI_API_KEY_1)
        assert client.retry_attempts == 2

    def test_init_no_api_keys_found_raises_error(self, mock_os_getenv_gemini, mock_dotenv_load_gemini, mock_genai_configure):
        mock_os_getenv_gemini.return_value = None
        mock_dotenv_load_gemini.side_effect = lambda: None # Simulate .env load doesn't find it

        with pytest.raises(LLMConfigurationError, match="GeminiClient: No API keys found"):
            GeminiClient(config={}) # No keys in config, env, or .env

    def test_init_configure_sdk_failure_in_init_raises_key_cycle_error(self, mock_genai_configure, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {"api_keys": [TEST_GEMINI_API_KEY_1]}
        # genai.configure is called in _configure_sdk_with_next_key, which is called by __init__
        mock_genai_configure.side_effect = Exception("SDK Configure failed in init")

        with pytest.raises(LLMKeyCycleError, match="Failed to configure genai SDK"):
            GeminiClient(config=config)

# --- Test Key Cycling & SDK Configuration (Largely unchanged) ---
class TestGeminiClientKeyCycling:
    def test_get_current_api_key_cycles_and_configures_sdk(self, mock_genai_configure, mock_genai_generativemodel_class, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {"api_keys": [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2]}
        client = GeminiClient(config=config) # Initial configure with KEY_1
        mock_genai_configure.assert_called_once_with(api_key=TEST_GEMINI_API_KEY_1) # From __init__

        # Call 1 (simulating first attempt of a retryable operation)
        key_for_attempt_1 = client._get_current_api_key_for_request()
        assert key_for_attempt_1 == TEST_GEMINI_API_KEY_2 # Advanced by _configure_sdk_with_next_key
        mock_genai_configure.assert_called_with(api_key=TEST_GEMINI_API_KEY_2) # Called again with new key

        # Call 2
        key_for_attempt_2 = client._get_current_api_key_for_request()
        assert key_for_attempt_2 == TEST_GEMINI_API_KEY_1
        mock_genai_configure.assert_called_with(api_key=TEST_GEMINI_API_KEY_1)

# --- API Method Tests (Focus of Giai Đoạn 2 review) ---
class TestGeminiClientApiMethodsWithRetryAndErrors:
    @pytest.fixture
    def client_three_keys_fast_retry(self, mock_genai_configure, mock_genai_generativemodel_class, mock_os_getenv_gemini):
        mock_os_getenv_gemini.return_value = None
        config = {
            "api_keys": [TEST_GEMINI_API_KEY_1, TEST_GEMINI_API_KEY_2, TEST_GEMINI_API_KEY_3],
            "model": "gemini-pro-test", # Ensure a model is set for client.model_name
            "retry_attempts": 3,
            "retry_min_wait_seconds": 0.01,
            "retry_max_wait_seconds": 0.02,
        }
        return GeminiClient(config=config)

    # Patch the _impl method for generate_text directly on the GeminiClient class for these tests
    @patch.object(GeminiClient, '_generate_text_impl')
    def test_generate_text_success_on_first_key(self, mock_impl, client_three_keys_fast_retry, caplog):
        mock_impl.return_value = "Success!"

        response = client_three_keys_fast_retry.generate_text("prompt")
        assert response == "Success!"
        mock_impl.assert_called_once()
        # _get_current_api_key_for_request was called by BaseClient, configured SDK with KEY_1 (from __init__)
        # then _get_current_api_key_for_request called again before _impl, cycling to KEY_2.
        # This needs adjustment: _get_current_api_key should be stable within one successful decorated call.
        # The current BaseLLMClient structure calls _get_current_api_key_for_request *inside* the _try_generate/_try_embed
        # which means for *each attempt* by tenacity, a new key is fetched. This is the desired key cycling per retry.

        # In __init__, KEY_1 is configured.
        # 1st attempt: _get_current_api_key_for_request -> _configure_sdk_with_next_key (KEY_2) -> _impl(key=KEY_2)
        assert mock_impl.call_args[1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_2
        assert not any("Retrying LLM API call" in record.message for record in caplog.records)

    @patch.object(GeminiClient, '_generate_text_impl')
    def test_generate_text_cycles_keys_on_llm_key_cycle_error(self, mock_impl, client_three_keys_fast_retry, caplog):
        # KEY_1 fails (PermissionDenied -> LLMKeyCycleError), KEY_2 fails (ResourceExhausted -> LLMKeyCycleError), KEY_3 succeeds
        def side_effect_func(prompt, temperature, max_tokens, model, current_api_key_for_request):
            if current_api_key_for_request == TEST_GEMINI_API_KEY_1: # Initial key from __init__
                 # _get_current_api_key_for_request will be called before this, so it's KEY_2
                raise LLMKeyCycleError("Key 1 fail (simulated from SDK PermissionDenied)", failed_key=TEST_GEMINI_API_KEY_1)
            elif current_api_key_for_request == TEST_GEMINI_API_KEY_2: # Next key after first failure
                raise LLMKeyCycleError("Key 2 fail (simulated from SDK ResourceExhausted)", failed_key=TEST_GEMINI_API_KEY_2)
            elif current_api_key_for_request == TEST_GEMINI_API_KEY_3: # Third key
                return "Success with Key 3"
            pytest.fail(f"Unexpected key for _impl: {current_api_key_for_request}")
        mock_impl.side_effect = side_effect_func

        response = client_three_keys_fast_retry.generate_text("prompt")
        assert response == "Success with Key 3"
        assert mock_impl.call_count == 3 # Called for KEY_1 (init), then retry with KEY_2, then retry with KEY_3

        # Check keys passed to _impl
        # Init configures with KEY_1.
        # Attempt 1: _get_key -> KEY_2. _impl(KEY_2) -> LLMKeyCycleError(KEY_2)
        # Attempt 2: _get_key -> KEY_3. _impl(KEY_3) -> LLMKeyCycleError(KEY_3)
        # Attempt 3: _get_key -> KEY_1. _impl(KEY_1) -> Success (if script was K1,K2 fail, K3 success)
        # Let's re-verify the key order from the test description above.
        # Init: configures K1. current_api_key_for_sdk = K1
        # Call generate_text():
        #   Attempt 1 (by tenacity):
        #     _get_current_api_key_for_request() -> _configure_sdk_with_next_key() -> next(cycler) is K2. current_api_key_for_sdk = K2. Returns K2.
        #     _impl(current_api_key_for_request=K2) -> Side effect for K2: LLMKeyCycleError("Key 2 fail")
        #   Attempt 2 (by tenacity):
        #     _get_current_api_key_for_request() -> _configure_sdk_with_next_key() -> next(cycler) is K3. current_api_key_for_sdk = K3. Returns K3.
        #     _impl(current_api_key_for_request=K3) -> Side effect for K3: Success "Success with Key 3"
        # So _impl is called with K2, then K3. Call count should be 2.
        # Let's adjust side_effect_func to match this sequence:
        def side_effect_func_adjusted(prompt, temperature, max_tokens, model, current_api_key_for_request):
            if current_api_key_for_request == TEST_GEMINI_API_KEY_2: # First attempt by tenacity
                raise LLMKeyCycleError("Key 2 fail", failed_key=TEST_GEMINI_API_KEY_2)
            elif current_api_key_for_request == TEST_GEMINI_API_KEY_3: # Second attempt
                return "Success with Key 3"
            pytest.fail(f"Unexpected key for _impl: {current_api_key_for_request}")
        mock_impl.side_effect = side_effect_func_adjusted

        response = client_three_keys_fast_retry.generate_text("prompt for adjusted") # Rerun with adjusted mock
        assert response == "Success with Key 3"
        assert mock_impl.call_count == 2

        call_args_list = mock_impl.call_args_list
        assert call_args_list[0][1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_2
        assert call_args_list[1][1]['current_api_key_for_request'] == TEST_GEMINI_API_KEY_3

        assert any(f"Retrying LLM API call: _try_generate (Key: {TEST_GEMINI_API_KEY_2})" in record.message for record in caplog.records if "LLMKeyCycleError" in record.message)

    @patch.object(GeminiClient, '_generate_text_impl')
    def test_generate_text_all_keys_cycle_and_fail_then_reraises(self, mock_impl, client_three_keys_fast_retry):
        # All 3 keys will raise LLMKeyCycleError. Client has 3 retry_attempts.
        # Each attempt will cycle the key.
        def side_effect_all_fail(prompt, temperature, max_tokens, model, current_api_key_for_request):
            raise LLMKeyCycleError(f"Key {current_api_key_for_request} failed", failed_key=current_api_key_for_request)
        mock_impl.side_effect = side_effect_all_fail

        with pytest.raises(LLMKeyCycleError) as excinfo: # Tenacity re-raises the last exception
            client_three_keys_fast_retry.generate_text("prompt all fail")

        assert mock_impl.call_count == client_three_keys_fast_retry.retry_attempts # 3 attempts
        # The last exception's failed_key should be the last key tried.
        # Keys tried: K2 (attempt 1), K3 (attempt 2), K1 (attempt 3)
        assert excinfo.value.failed_key == TEST_GEMINI_API_KEY_1

    @patch.object(GeminiClient, '_generate_text_impl')
    def test_generate_text_invalid_argument_no_retry(self, mock_impl, client_three_keys_fast_retry, caplog):
        # InvalidArgument -> LLMConfigurationError, which is not in DEFAULT_RETRY_EXCEPTIONS
        # The _impl method in GeminiClient would map google_exceptions.InvalidArgument to LLMConfigurationError
        mock_impl.side_effect = LLMConfigurationError("Bad prompt format")

        with pytest.raises(LLMConfigurationError, match="Bad prompt format"):
            client_three_keys_fast_retry.generate_text("invalid prompt")

        mock_impl.assert_called_once() # Should not retry
        assert not any("Retrying LLM API call" in record.message for record in caplog.records)

    @patch.object(GeminiClient, '_generate_text_impl')
    def test_generate_text_service_unavailable_causes_key_cycle(self, mock_impl, client_three_keys_fast_retry, caplog):
        # ServiceUnavailable -> LLMKeyCycleError in current GeminiClient._generate_text_impl
        def side_effect_svc_unavailable(prompt, temperature, max_tokens, model, current_api_key_for_request):
            if current_api_key_for_request == TEST_GEMINI_API_KEY_2: # First attempt key
                raise LLMKeyCycleError("Service unavailable for Key 2", failed_key=TEST_GEMINI_API_KEY_2, original_exception=google_exceptions.ServiceUnavailable("err"))
            elif current_api_key_for_request == TEST_GEMINI_API_KEY_3: # Second attempt key
                return "Success on Key 3 after service unavailable"
            pytest.fail("Should not reach here if Key 3 succeeds")
        mock_impl.side_effect = side_effect_svc_unavailable

        response = client_three_keys_fast_retry.generate_text("prompt service unavailable")
        assert response == "Success on Key 3 after service unavailable"
        assert mock_impl.call_count == 2 # K2 fails, K3 succeeds
        assert any(f"Retrying LLM API call: _try_generate (Key: {TEST_GEMINI_API_KEY_2})" in record.message for record in caplog.records if "LLMKeyCycleError" in record.message)

    # Similar tests for _get_embedding_impl
    @patch.object(GeminiClient, '_get_embedding_impl')
    def test_get_embedding_all_keys_fail_then_reraises(self, mock_impl, client_three_keys_fast_retry):
        def side_effect_embed_all_fail(text, model, current_api_key_for_request):
            raise LLMKeyCycleError(f"Embed Key {current_api_key_for_request} failed", failed_key=current_api_key_for_request)
        mock_impl.side_effect = side_effect_embed_all_fail

        with pytest.raises(LLMKeyCycleError) as excinfo:
            client_three_keys_fast_retry.get_embedding("text to embed")

        assert mock_impl.call_count == client_three_keys_fast_retry.retry_attempts
        assert excinfo.value.failed_key == TEST_GEMINI_API_KEY_1 # Last key tried after cycling

    @patch.object(GeminiClient, '_generate_text_impl')
    def test_generate_text_sdk_response_no_candidates(self, mock_impl, client_three_keys_fast_retry, caplog):
        # This tests if _generate_text_impl correctly raises LLMResponseError
        # when the SDK returns a response with no candidates (e.g., due to safety filters)
        # The actual GeminiClient._generate_text_impl contains this logic.

        # We need to call the *actual* _generate_text_impl but mock the SDK call inside it.
        # So, we patch genai.GenerativeModel().generate_content

        # Get the original _impl method before patching it for other tests
        original_impl = client_three_keys_fast_retry._generate_text_impl

        with patch.object(genai, 'GenerativeModel') as MockedGMClass:
            mock_gm_instance = MockedGMClass.return_value

            mock_response_no_candidates = MagicMock(spec=genai.types.GenerateContentResponse)
            mock_response_no_candidates.candidates = [] # Empty candidates
            mock_response_no_candidates.prompt_feedback = MagicMock(block_reason=MagicMock(name="SAFETY"))
            mock_gm_instance.generate_content.return_value = mock_response_no_candidates

            # Now, allow the actual _generate_text_impl to be called
            # It will use the mocked genai.GenerativeModel instance
            with pytest.raises(LLMResponseError, match="Prompt blocked by Gemini safety filters: SAFETY"):
                original_impl(prompt="very risky prompt", temperature=0.7, max_tokens=100, model="gemini-pro-test", current_api_key_for_request=TEST_GEMINI_API_KEY_2)

        # Check logs from the actual _impl method
        assert any("Gemini model gemini-pro-test returned no candidates." in record.message for record in caplog.records if record.levelname == "WARNING")
        assert any("Prompt blocked for model gemini-pro-test due to: SAFETY" in record.message for record in caplog.records if record.levelname == "ERROR")
        # No retry should happen for LLMResponseError by default
        assert not any("Retrying LLM API call" in record.message for record in caplog.records)
