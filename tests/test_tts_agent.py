"""Tests for the TTS Agent (pipeline stage 9).

Tests cover:
- Input validation
- Audio generation with mocked OpenAI client
- Cost calculation
- Duration estimation
- Batch processing (process_all_scenes)
- Error handling and retry logic
- Strict mode behavior
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.agents.tts_agent import TTSAgent, TTSInput, TTSOutput, TTSError
from src.agents.base import AgentExecutionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with TTS capabilities."""
    client = Mock()
    response = Mock()
    response.content = b"\x00" * 1024  # Fake MP3 bytes
    client.audio.speech.create.return_value = response
    return client


@pytest.fixture
def agent_with_client(mock_openai_client):
    """Create a TTSAgent with a mocked OpenAI client."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("src.agents.tts_agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_openai_client
            agent = TTSAgent()
    agent.client = mock_openai_client
    return agent


@pytest.fixture
def agent_no_key():
    """Create a TTSAgent without an API key."""
    with patch.dict(os.environ, {}, clear=True):
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        agent = TTSAgent()
    return agent


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_valid_input(self, agent_with_client):
        input_data = TTSInput(
            narration="Hello world",
            scene_id="scene_1",
            output_dir="/tmp/tts",
        )
        assert agent_with_client.validate_input(input_data) is True

    def test_rejects_empty_narration(self, agent_with_client):
        input_data = TTSInput(
            narration="",
            scene_id="scene_1",
            output_dir="/tmp/tts",
        )
        assert agent_with_client.validate_input(input_data) is False

    def test_rejects_wrong_type(self, agent_with_client):
        assert agent_with_client.validate_input(Mock()) is False

    def test_raises_on_invalid_input(self, agent_with_client):
        with pytest.raises(AgentExecutionError, match="INVALID_INPUT"):
            agent_with_client.execute(Mock())


# ---------------------------------------------------------------------------
# Audio Generation
# ---------------------------------------------------------------------------

class TestAudioGeneration:
    def test_generates_audio_file(self, agent_with_client, tmp_path):
        """Should generate an MP3 file and return TTSOutput."""
        input_data = TTSInput(
            narration="This is a test narration for the scene.",
            scene_id="scene_intro",
            output_dir=str(tmp_path),
        )
        result = agent_with_client.execute(input_data)

        assert isinstance(result, TTSOutput)
        assert result.scene_id == "scene_intro"
        assert result.audio_file_path.endswith(".mp3")
        assert result.duration_seconds > 0
        assert result.cost_usd >= 0
        assert os.path.exists(result.audio_file_path)

    def test_calls_openai_with_correct_params(self, agent_with_client, tmp_path):
        """Should call OpenAI TTS API with correct model and voice."""
        input_data = TTSInput(
            narration="Test",
            scene_id="s1",
            output_dir=str(tmp_path),
            quality="high",
            voice="nova",
        )
        agent_with_client.execute(input_data)

        agent_with_client.client.audio.speech.create.assert_called_once_with(
            model="tts-1-hd",
            voice="nova",
            input="Test",
            response_format="mp3",
        )

    def test_standard_quality_uses_tts1(self, agent_with_client, tmp_path):
        input_data = TTSInput(
            narration="Test",
            scene_id="s1",
            output_dir=str(tmp_path),
            quality="standard",
        )
        agent_with_client.execute(input_data)

        call_kwargs = agent_with_client.client.audio.speech.create.call_args[1]
        assert call_kwargs["model"] == "tts-1"


# ---------------------------------------------------------------------------
# No API Key Behavior
# ---------------------------------------------------------------------------

class TestNoApiKey:
    def test_returns_tts_error_without_key(self, agent_no_key, tmp_path):
        """Without API key, should return TTSError (not crash)."""
        input_data = TTSInput(
            narration="Test",
            scene_id="s1",
            output_dir=str(tmp_path),
        )
        result = agent_no_key.execute(input_data)
        assert isinstance(result, TTSError)
        assert "API key" in result.error or "not configured" in result.error


# ---------------------------------------------------------------------------
# Strict Mode
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_strict_raises_on_missing_key(self):
        """In strict mode, missing API key should raise AgentExecutionError."""
        with patch.dict(os.environ, {}, clear=True):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
                TTSAgent(strict=True)

    def test_non_strict_does_not_raise(self):
        """Without strict mode, missing API key logs warning but doesn't crash."""
        with patch.dict(os.environ, {}, clear=True):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            agent = TTSAgent(strict=False)
            assert agent.client is None


# ---------------------------------------------------------------------------
# Cost Calculation
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_cost_for_1000_chars(self, agent_with_client):
        cost = agent_with_client._calculate_cost(1000)
        expected = (1000 / 1_000_000) * 15.00
        assert abs(cost - expected) < 0.001

    def test_cost_for_empty_string(self, agent_with_client):
        cost = agent_with_client._calculate_cost(0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Duration Estimation
# ---------------------------------------------------------------------------

class TestDurationEstimation:
    def test_estimates_duration(self, agent_with_client):
        """~12.5 chars/sec, so 125 chars ≈ 10s."""
        duration = agent_with_client._estimate_duration("x" * 125)
        assert 9.5 < duration < 10.5

    def test_minimum_duration(self, agent_with_client):
        """Even for a single character, minimum is 0.1s."""
        duration = agent_with_client._estimate_duration("a")
        assert duration >= 0.1


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def test_process_multiple_scenes(self, agent_with_client, tmp_path):
        scenes = [
            {"scene_id": "s1", "narration": "First scene narration"},
            {"scene_id": "s2", "narration": "Second scene narration"},
        ]
        results = agent_with_client.process_all_scenes(
            scenes=scenes,
            output_dir=str(tmp_path),
        )
        assert len(results) == 2
        assert all(isinstance(r, TTSOutput) for r in results)

    def test_skips_scenes_with_missing_narration(self, agent_with_client, tmp_path):
        scenes = [
            {"scene_id": "s1", "narration": "Good narration"},
            {"scene_id": "s2", "narration": ""},  # empty narration
        ]
        results = agent_with_client.process_all_scenes(
            scenes=scenes,
            output_dir=str(tmp_path),
        )
        # s2 should be an error
        assert any(isinstance(r, TTSError) for r in results)

    def test_returns_empty_for_no_scenes(self, agent_with_client, tmp_path):
        results = agent_with_client.process_all_scenes(
            scenes=[],
            output_dir=str(tmp_path),
        )
        assert results == []


# ---------------------------------------------------------------------------
# Configuration Validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_rejects_invalid_quality(self, agent_with_client, tmp_path):
        input_data = TTSInput(
            narration="Test",
            scene_id="s1",
            output_dir=str(tmp_path),
            quality="ultra",  # invalid
        )
        result = agent_with_client.execute(input_data)
        assert isinstance(result, TTSError)

    def test_rejects_invalid_voice(self, agent_with_client, tmp_path):
        input_data = TTSInput(
            narration="Test",
            scene_id="s1",
            output_dir=str(tmp_path),
            voice="unknown_voice",  # invalid
        )
        result = agent_with_client.execute(input_data)
        assert isinstance(result, TTSError)


# ---------------------------------------------------------------------------
# Retry Policy
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    def test_has_exponential_backoff(self, agent_with_client):
        policy = agent_with_client.get_retry_policy()
        assert policy.max_attempts == 3
        assert "RATE_LIMIT_EXCEEDED" in policy.retryable_errors
        assert "API_TIMEOUT" in policy.retryable_errors

    def test_backoff_increases(self, agent_with_client):
        delay0 = agent_with_client._calculate_backoff(0)
        delay1 = agent_with_client._calculate_backoff(1)
        delay2 = agent_with_client._calculate_backoff(2)
        assert delay1 > delay0
        assert delay2 > delay1
