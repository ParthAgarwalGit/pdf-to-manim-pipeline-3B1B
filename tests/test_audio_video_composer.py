"""Tests for the Audio-Video Composer Agent (pipeline stage 10).

Tests cover:
- Input validation
- Per-scene composition with matched audio+video
- Silent scenes (video only, no audio)
- Timing mismatch handling strategies
- Error recovery (failed scenes are skipped)
- Final video concatenation
- Retry policy
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.agents.audio_video_composer import (
    AudioVideoComposerAgent,
    AudioVideoComposerInput,
    AudioVideoComposerOutput,
    AudioVideoComposerFailure,
    SceneCompositionResult,
    SceneMediaPair,
)
from src.agents.base import AgentExecutionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return AudioVideoComposerAgent()


@pytest.fixture
def tmp_output(tmp_path):
    """Create a temporary output directory."""
    return str(tmp_path / "output")


@pytest.fixture
def fake_video(tmp_path):
    """Create a minimal fake video file."""
    video = tmp_path / "scene_intro.mp4"
    video.write_bytes(b"\x00" * 1024)  # Fake video bytes
    return str(video)


@pytest.fixture
def fake_audio(tmp_path):
    """Create a minimal fake audio file."""
    audio = tmp_path / "scene_intro.mp3"
    audio.write_bytes(b"\x00" * 512)  # Fake audio bytes
    return str(audio)


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_validates_correct_input(self, agent, fake_video, tmp_output):
        input_data = AudioVideoComposerInput(
            scene_media=[SceneMediaPair("s1", fake_video)],
            output_dir=tmp_output,
        )
        assert agent.validate_input(input_data) is True

    def test_rejects_wrong_input_type(self, agent):
        assert agent.validate_input(Mock()) is False

    def test_empty_scene_media_returns_failure(self, agent, tmp_output):
        input_data = AudioVideoComposerInput(
            scene_media=[],
            output_dir=tmp_output,
        )
        result = agent.execute(input_data)
        assert isinstance(result, AudioVideoComposerFailure)
        assert "No scene media" in result.error

    def test_raises_on_invalid_input_type(self, agent):
        with pytest.raises(AgentExecutionError, match="INVALID_INPUT"):
            agent.execute(Mock())


# ---------------------------------------------------------------------------
# Scene Media Pairing
# ---------------------------------------------------------------------------

class TestSceneMediaPair:
    def test_with_both_video_and_audio(self, fake_video, fake_audio):
        pair = SceneMediaPair("s1", fake_video, fake_audio)
        assert pair.scene_id == "s1"
        assert pair.video_path == fake_video
        assert pair.audio_path == fake_audio

    def test_silent_scene(self, fake_video):
        pair = SceneMediaPair("s1", fake_video)
        assert pair.audio_path is None

    def test_missing_video_file(self, agent, tmp_output):
        """Composing a scene whose video file doesn't exist should fail gracefully."""
        pair = SceneMediaPair("s1", "/nonexistent/video.mp4", None)
        result = agent._compose_scene(pair, tmp_output)
        assert result.success is False
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# Composition Strategies (mocked FFmpeg)
# ---------------------------------------------------------------------------

class TestCompositionStrategies:
    @patch("src.agents.audio_video_composer.subprocess.run")
    @patch.object(AudioVideoComposerAgent, "_get_media_duration")
    def test_simple_merge_when_durations_close(
        self, mock_duration, mock_run, agent, fake_video, fake_audio, tmp_path
    ):
        """When the video is up to 0.5s longer, use simple merge."""
        mock_duration.side_effect = [10.2, 10.0, 10.2]  # video, audio, output
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        composed_dir = str(tmp_path / "composed")
        os.makedirs(composed_dir, exist_ok=True)

        # Create the expected output file so the agent finds it
        expected_out = Path(composed_dir) / "s1_composed.mp4"
        expected_out.write_bytes(b"\x00" * 100)

        pair = SceneMediaPair("s1", fake_video, fake_audio)
        result = agent._compose_scene(pair, composed_dir)

        assert result.success is True
        assert result.scene_id == "s1"
        # Verify _merge_simple was called (contains -shortest without tpad/apad)
        call_args = mock_run.call_args_list[0][0][0]
        assert "-shortest" in call_args
        joined = " ".join(call_args)
        assert "tpad" not in joined and "apad" not in joined
        assert "loudnorm" in joined  # narration normalized to the delivery target

    @patch("src.agents.audio_video_composer.subprocess.run")
    @patch.object(AudioVideoComposerAgent, "_get_media_duration")
    def test_slightly_longer_narration_is_never_trimmed(
        self, mock_duration, mock_run, agent, fake_video, fake_audio, tmp_path
    ):
        """Narration 0.2s longer than the video extends the video (the old
        simple merge cut the last 0.2s of speech with -shortest)."""
        mock_duration.side_effect = [10.0, 10.2, 10.2]
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        composed_dir = str(tmp_path / "composed")
        os.makedirs(composed_dir, exist_ok=True)
        (Path(composed_dir) / "s1_composed.mp4").write_bytes(b"\x00" * 100)

        result = agent._compose_scene(SceneMediaPair("s1", fake_video, fake_audio), composed_dir)

        assert result.success is True
        assert "tpad" in " ".join(mock_run.call_args_list[0][0][0])

    @patch("src.agents.audio_video_composer.subprocess.run")
    @patch.object(AudioVideoComposerAgent, "_get_media_duration")
    def test_freeze_video_when_audio_longer(
        self, mock_duration, mock_run, agent, fake_video, fake_audio, tmp_path
    ):
        """When audio is longer, use tpad to freeze last frame."""
        mock_duration.side_effect = [5.0, 12.0, 12.0]  # video shorter than audio
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        composed_dir = str(tmp_path / "composed")
        os.makedirs(composed_dir, exist_ok=True)
        expected_out = Path(composed_dir) / "s1_composed.mp4"
        expected_out.write_bytes(b"\x00" * 100)

        pair = SceneMediaPair("s1", fake_video, fake_audio)
        result = agent._compose_scene(pair, composed_dir)

        assert result.success is True
        # tpad filter should be in the ffmpeg command
        call_args = mock_run.call_args_list[0][0][0]
        filter_arg = " ".join(call_args)
        assert "tpad" in filter_arg

    @patch("src.agents.audio_video_composer.subprocess.run")
    @patch.object(AudioVideoComposerAgent, "_get_media_duration")
    def test_pad_audio_when_video_longer(
        self, mock_duration, mock_run, agent, fake_video, fake_audio, tmp_path
    ):
        """When video is longer, use apad to extend audio."""
        mock_duration.side_effect = [15.0, 8.0, 15.0]  # video longer than audio
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        composed_dir = str(tmp_path / "composed")
        os.makedirs(composed_dir, exist_ok=True)
        expected_out = Path(composed_dir) / "s1_composed.mp4"
        expected_out.write_bytes(b"\x00" * 100)

        pair = SceneMediaPair("s1", fake_video, fake_audio)
        result = agent._compose_scene(pair, composed_dir)

        assert result.success is True
        call_args = mock_run.call_args_list[0][0][0]
        filter_arg = " ".join(call_args)
        assert "apad" in filter_arg


# ---------------------------------------------------------------------------
# Silent Scenes
# ---------------------------------------------------------------------------

class TestSilentScenes:
    def test_copies_video_when_no_audio(self, agent, fake_video, tmp_path):
        """Scenes with no audio should just copy the video file."""
        composed_dir = str(tmp_path / "composed")
        os.makedirs(composed_dir, exist_ok=True)

        pair = SceneMediaPair("s1", fake_video, None)

        with patch.object(agent, "_get_media_duration", return_value=10.0):
            result = agent._compose_scene(pair, composed_dir)

        assert result.success is True
        assert result.audio_duration == 0.0
        assert result.final_duration == 10.0
        assert os.path.exists(result.composed_video_path)

    def test_missing_audio_file_treated_as_silent(self, agent, fake_video, tmp_path):
        """If audio path is set but file doesn't exist, treat as silent."""
        composed_dir = str(tmp_path / "composed")
        os.makedirs(composed_dir, exist_ok=True)

        pair = SceneMediaPair("s1", fake_video, "/nonexistent/audio.mp3")

        with patch.object(agent, "_get_media_duration", return_value=8.0):
            result = agent._compose_scene(pair, composed_dir)

        assert result.success is True
        assert result.audio_duration == 0.0


# ---------------------------------------------------------------------------
# Error Recovery
# ---------------------------------------------------------------------------

class TestErrorRecovery:
    @patch.object(AudioVideoComposerAgent, "_compose_scene")
    @patch.object(AudioVideoComposerAgent, "_concatenate_videos")
    def test_skips_failed_scenes(self, mock_concat, mock_compose, agent, tmp_path):
        """Failed scenes should be logged and skipped, not crash pipeline."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        composed_path = str(tmp_path / "output" / "composed" / "s2_composed.mp4")
        os.makedirs(os.path.dirname(composed_path), exist_ok=True)
        Path(composed_path).write_bytes(b"\x00" * 100)

        mock_compose.side_effect = [
            SceneCompositionResult("s1", success=False, error="FFmpeg crashed"),
            SceneCompositionResult(
                "s2", success=True, composed_video_path=composed_path,
                final_duration=10.0
            ),
        ]

        final_path = os.path.join(output_dir, "final.mp4")
        Path(final_path).write_bytes(b"\x00" * 100)

        input_data = AudioVideoComposerInput(
            scene_media=[
                SceneMediaPair("s1", "/v1.mp4"),
                SceneMediaPair("s2", "/v2.mp4"),
            ],
            output_dir=output_dir,
        )

        result = agent.execute(input_data)
        assert isinstance(result, AudioVideoComposerOutput)
        assert result.scenes_composed == 1
        assert result.scenes_skipped == 1

    @patch.object(AudioVideoComposerAgent, "_compose_scene")
    def test_all_scenes_fail(self, mock_compose, agent, tmp_path):
        """If every scene fails, return AudioVideoComposerFailure."""
        output_dir = str(tmp_path / "output")
        mock_compose.return_value = SceneCompositionResult(
            "s1", success=False, error="crash"
        )

        input_data = AudioVideoComposerInput(
            scene_media=[SceneMediaPair("s1", "/v1.mp4")],
            output_dir=output_dir,
        )

        result = agent.execute(input_data)
        assert isinstance(result, AudioVideoComposerFailure)
        assert "All scene compositions failed" in result.error

    def test_ffmpeg_timeout(self, agent, fake_video, fake_audio, tmp_path):
        """FFmpeg timeout should be caught and reported per-scene."""
        composed_dir = str(tmp_path / "composed")
        os.makedirs(composed_dir, exist_ok=True)

        pair = SceneMediaPair("s1", fake_video, fake_audio)

        with patch.object(agent, "_get_media_duration", return_value=10.0), \
             patch.object(agent, "_merge_simple", side_effect=subprocess.TimeoutExpired("ffmpeg", 120)):
            result = agent._compose_scene(pair, composed_dir)

        assert result.success is False
        assert "timed out" in result.error


# ---------------------------------------------------------------------------
# Retry Policy
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    def test_has_retry_policy(self, agent):
        policy = agent.get_retry_policy()
        assert policy.max_attempts == 2
        assert "FFMPEG_TIMEOUT" in policy.retryable_errors


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

class TestConcatenation:
    @patch("src.agents.audio_video_composer.subprocess.run")
    def test_concat_demuxer_called(self, mock_run, agent, tmp_path):
        """Final assembly should use FFmpeg concat demuxer."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        videos = [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]
        output = str(tmp_path / "final.mp4")

        agent._concatenate_videos(videos, output)

        call_args = mock_run.call_args[0][0]
        assert "-f" in call_args
        assert "concat" in call_args

    @patch("src.utils.media_probe.probe", return_value=None)  # format probe before re-encode
    @patch("src.agents.audio_video_composer.subprocess.run")
    def test_reencode_fallback_on_concat_failure(self, mock_run, _probe, agent, tmp_path):
        """If concat demuxer fails, fall back to re-encode."""
        # First call fails (concat demuxer), second succeeds (re-encode)
        mock_run.side_effect = [
            Mock(returncode=1, stderr="codec mismatch"),
            Mock(returncode=0, stdout="", stderr=""),
        ]

        videos = [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]
        output = str(tmp_path / "final.mp4")

        agent._concatenate_videos(videos, output)

        assert mock_run.call_count == 2
        # Second call should have filter_complex
        second_call = mock_run.call_args_list[1][0][0]
        assert "-filter_complex" in second_call
