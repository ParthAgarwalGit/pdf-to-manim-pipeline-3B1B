"""Audio-Video Composer Agent — merges per-scene .mp4 video with .mp3 narration.

Pipeline stage 10 (final stage):
  1. For each scene, find the matching .mp4 (from Video Renderer) and .mp3 (from TTS)
  2. Merge them into a single video-with-audio file using FFmpeg
  3. Handle timing mismatches:
     - Audio longer than video → freeze last frame to match audio duration
     - Video longer than audio → pad audio with silence
  4. Concatenate all composed scene videos into one final output
  5. Per-scene error recovery — failed compositions are logged and skipped

FFmpeg strategy:
  - Uses `-shortest` only when video is longer (audio padded with silence)
  - Uses `-filter_complex` with tpad/apad for duration-mismatch cases
  - Concat demuxer for final assembly
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SceneCompositionResult:
    """Result of composing a single scene's audio + video."""
    scene_id: str
    success: bool
    composed_video_path: Optional[str] = None
    video_duration: float = 0.0
    audio_duration: float = 0.0
    final_duration: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# I/O Contracts
# ---------------------------------------------------------------------------

@dataclass
class SceneMediaPair:
    """A matched pair of video and audio for a single scene."""
    scene_id: str
    video_path: str
    audio_path: Optional[str] = None  # None = scene has no narration


class AudioVideoComposerInput(AgentInput):
    """Input to the Audio-Video Composer Agent.

    Attributes:
        scene_media: List of SceneMediaPair objects (video + audio per scene)
        output_dir: Directory where composed videos will be written
        final_video_name: Filename for the concatenated final video
    """

    def __init__(
        self,
        scene_media: List[SceneMediaPair],
        output_dir: str,
        final_video_name: str = "final_with_narration.mp4",
    ):
        self.scene_media = scene_media
        self.output_dir = output_dir
        self.final_video_name = final_video_name


class AudioVideoComposerOutput(AgentOutput):
    """Output: composition succeeded (at least partially).

    Attributes:
        final_video_path: Path to the final composed video with narration
        scene_results: Per-scene composition results
        total_duration: Total duration of the final video in seconds
        scenes_composed: Number of scenes successfully composed
        scenes_skipped: Number of scenes that failed composition
    """

    def __init__(
        self,
        final_video_path: str,
        scene_results: List[SceneCompositionResult],
        total_duration: float,
        scenes_composed: int,
        scenes_skipped: int,
    ):
        self.final_video_path = final_video_path
        self.scene_results = scene_results
        self.total_duration = total_duration
        self.scenes_composed = scenes_composed
        self.scenes_skipped = scenes_skipped


class AudioVideoComposerFailure(AgentOutput):
    """Output: composition failed entirely."""

    def __init__(
        self,
        scene_results: List[SceneCompositionResult],
        error: str,
    ):
        self.scene_results = scene_results
        self.error = error


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AudioVideoComposerAgent(Agent):
    """Merges per-scene video (.mp4) and narration audio (.mp3) into final video.

    This is the final stage of the pipeline. It takes the outputs from
    Video Renderer (stage 8) and TTS (stage 9) and produces a single
    video file with synchronized narration audio.

    Timing mismatch handling:
    - If audio is longer than video: the video's last frame is frozen
      (using FFmpeg tpad filter) to match the audio duration.
    - If video is longer than audio: the audio track is padded with
      silence (using FFmpeg apad filter) and cut at video end.
    - If a scene has no audio: the video is used as-is (silent).
    """

    # Timeout for FFmpeg operations (seconds)
    COMPOSE_TIMEOUT = 120
    CONCAT_TIMEOUT = 180
    # Narration may end at most this much after the video before the video
    # is extended rather than the narration trimmed (AAC frame slack).
    NARRATION_CLIP_TOLERANCE = 0.05
    # Every narration track is normalized to the delivery loudness target
    # (single-pass; the delivery gate re-normalizes any clip that misses it).
    # Output is resampled to one rate so concatenated clips stay uniform.
    NARRATION_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"

    def execute(self, input_data: AudioVideoComposerInput) -> AgentOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError("INVALID_INPUT", "Expected AudioVideoComposerInput.")

        if not input_data.scene_media:
            return AudioVideoComposerFailure(
                scene_results=[],
                error="No scene media pairs provided for composition.",
            )

        # Ensure output directory exists
        composed_dir = os.path.join(input_data.output_dir, "composed")
        os.makedirs(composed_dir, exist_ok=True)

        # 1. Compose each scene
        scene_results: List[SceneCompositionResult] = []
        composed_videos: List[str] = []

        for pair in input_data.scene_media:
            result = self._compose_scene(pair, composed_dir)
            scene_results.append(result)
            if result.success and result.composed_video_path:
                composed_videos.append(result.composed_video_path)

        if not composed_videos:
            return AudioVideoComposerFailure(
                scene_results=scene_results,
                error="All scene compositions failed. Check individual scene errors.",
            )

        # 2. Concatenate into final video
        final_path = os.path.join(input_data.output_dir, input_data.final_video_name)
        try:
            if len(composed_videos) == 1:
                shutil.copy2(composed_videos[0], final_path)
            else:
                self._concatenate_videos(composed_videos, final_path)

            total_duration = sum(r.final_duration for r in scene_results if r.success)
            scenes_composed = sum(1 for r in scene_results if r.success)
            scenes_skipped = sum(1 for r in scene_results if not r.success)

            logger.info(
                f"Final composed video: {final_path} "
                f"({scenes_composed}/{len(scene_results)} scenes, {total_duration:.1f}s)"
            )

            return AudioVideoComposerOutput(
                final_video_path=final_path,
                scene_results=scene_results,
                total_duration=total_duration,
                scenes_composed=scenes_composed,
                scenes_skipped=scenes_skipped,
            )

        except Exception as e:
            logger.error(f"Final video concatenation failed: {e}")
            return AudioVideoComposerFailure(
                scene_results=scene_results,
                error=f"Concatenation failed: {e}",
            )

    def validate_input(self, input_data: AgentInput) -> bool:
        return isinstance(input_data, AudioVideoComposerInput)

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=2,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=2.0,
            max_delay_seconds=10.0,
            retryable_errors=["FFMPEG_TIMEOUT"],
        )

    # ------------------------------------------------------------------
    # Per-scene composition
    # ------------------------------------------------------------------

    def _compose_scene(
        self, pair: SceneMediaPair, composed_dir: str
    ) -> SceneCompositionResult:
        """Merge a single scene's video and audio into one file."""
        scene_id = pair.scene_id

        # Validate video file exists
        if not os.path.isfile(pair.video_path):
            return SceneCompositionResult(
                scene_id=scene_id,
                success=False,
                error=f"Video file not found: {pair.video_path}",
            )

        output_path = os.path.join(composed_dir, f"{scene_id}_composed.mp4")

        # If no audio, the video goes in silent. It still gets a silent audio
        # track: a clip without one breaks stream-copy concatenation.
        if not pair.audio_path or not os.path.isfile(pair.audio_path):
            logger.info(f"Scene {scene_id}: no audio, adding a silent track")
            try:
                try:
                    self._add_silent_track(pair.video_path, output_path)
                except Exception as exc:
                    logger.warning("Scene %s: silent track failed (%s); copying video", scene_id, exc)
                    shutil.copy2(pair.video_path, output_path)
                video_dur = self._get_media_duration(pair.video_path)
                return SceneCompositionResult(
                    scene_id=scene_id,
                    success=True,
                    composed_video_path=output_path,
                    video_duration=video_dur,
                    audio_duration=0.0,
                    final_duration=video_dur,
                )
            except Exception as e:
                return SceneCompositionResult(
                    scene_id=scene_id, success=False,
                    error=f"Failed to copy silent video: {e}",
                )

        # Both video and audio exist — probe durations
        video_dur = self._get_media_duration(pair.video_path)
        audio_dur = self._get_media_duration(pair.audio_path)

        if video_dur <= 0 and audio_dur <= 0:
            return SceneCompositionResult(
                scene_id=scene_id, success=False,
                error="Could not determine duration of video or audio.",
            )

        try:
            # Choose merge strategy based on duration comparison. Narration
            # is never cut: any audio longer than the video (beyond 50 ms of
            # encoder slack) extends the video instead of being trimmed.
            if audio_dur - video_dur <= self.NARRATION_CLIP_TOLERANCE and video_dur - audio_dur < 0.5:
                # Durations are close enough — simple merge
                self._merge_simple(pair.video_path, pair.audio_path, output_path)
                final_dur = max(video_dur, audio_dur)

            elif audio_dur > video_dur:
                # Audio is longer — freeze last frame of video
                self._merge_freeze_video(
                    pair.video_path, pair.audio_path, output_path, audio_dur,
                    video_dur=video_dur,
                )
                final_dur = audio_dur

            else:
                # Video is longer — pad audio with silence, cut at video end
                self._merge_pad_audio(pair.video_path, pair.audio_path, output_path)
                final_dur = video_dur

            # Verify output was created
            if not os.path.isfile(output_path):
                return SceneCompositionResult(
                    scene_id=scene_id, success=False,
                    error="FFmpeg completed but output file not found.",
                )

            actual_dur = self._get_media_duration(output_path)
            if actual_dur > 0:
                final_dur = actual_dur

            logger.info(
                f"Scene {scene_id} composed: video={video_dur:.1f}s, "
                f"audio={audio_dur:.1f}s, final={final_dur:.1f}s"
            )

            return SceneCompositionResult(
                scene_id=scene_id,
                success=True,
                composed_video_path=output_path,
                video_duration=video_dur,
                audio_duration=audio_dur,
                final_duration=final_dur,
            )

        except subprocess.TimeoutExpired:
            return SceneCompositionResult(
                scene_id=scene_id, success=False,
                error=f"FFmpeg timed out after {self.COMPOSE_TIMEOUT}s.",
            )
        except Exception as e:
            return SceneCompositionResult(
                scene_id=scene_id, success=False,
                error=f"Composition failed: {e}",
            )

    # ------------------------------------------------------------------
    # FFmpeg merge strategies
    # ------------------------------------------------------------------

    def _merge_simple(self, video: str, audio: str, output: str) -> None:
        """Simple merge — durations are approximately equal.

        Replaces any existing audio track in the video with the narration.
        Uses -shortest to cut at the shorter stream's end.
        """
        cmd = [
            get_ffmpeg(), "-y",
            "-i", video,
            "-i", audio,
            "-filter_complex", f"[1:a]{self.NARRATION_FILTER}[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-shortest",
            output,
        ]
        self._run_ffmpeg(cmd, timeout=self.COMPOSE_TIMEOUT)

    def _add_silent_track(self, video: str, output: str) -> None:
        """Copy the video and add a silent audio track matching narrated clips."""
        cmd = [
            get_ffmpeg(), "-y",
            "-i", video,
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output,
        ]
        self._run_ffmpeg(cmd, timeout=self.COMPOSE_TIMEOUT)

    def _merge_freeze_video(
        self, video: str, audio: str, output: str, target_duration: float,
        video_dur: float = 0.0,
    ) -> None:
        """Audio is longer than video — freeze the video's last frame.

        Uses the tpad filter to extend the video stream by repeating the
        last frame until it matches the audio duration.
        """
        if video_dur <= 0:
            video_dur = self._get_media_duration(video)
        extend_secs = max(0, target_duration - video_dur + 0.5)

        cmd = [
            get_ffmpeg(), "-y",
            "-i", video,
            "-i", audio,
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={extend_secs:.2f}[vout];"
            f"[1:a]{self.NARRATION_FILTER}[aout]",
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output,
        ]
        self._run_ffmpeg(cmd, timeout=self.COMPOSE_TIMEOUT)

    def _merge_pad_audio(self, video: str, audio: str, output: str) -> None:
        """Video is longer than audio — pad audio with silence, cut at video end.

        Uses apad to extend the audio stream, then -shortest to trim
        at the video stream's natural end.
        """
        cmd = [
            get_ffmpeg(), "-y",
            "-i", video,
            "-i", audio,
            "-filter_complex",
            f"[1:a]{self.NARRATION_FILTER},apad[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output,
        ]
        self._run_ffmpeg(cmd, timeout=self.COMPOSE_TIMEOUT)

    # ------------------------------------------------------------------
    # Video concatenation (final assembly)
    # ------------------------------------------------------------------

    def _concatenate_videos(self, video_paths: List[str], output_path: str) -> None:
        """Concatenate multiple composed scene videos into one final video.

        Uses FFmpeg's concat demuxer for lossless concatenation when codecs
        match, or re-encodes when necessary.
        """
        # First, try the fast concat demuxer (no re-encode)
        concat_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                for vp in video_paths:
                    escaped = vp.replace("\\", "/").replace("'", "'\\''")  # escape for FFmpeg concat
                    f.write(f"file '{escaped}'\n")
                concat_file = f.name

            cmd = [
                get_ffmpeg(), "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.CONCAT_TIMEOUT
            )

            if result.returncode != 0:
                # Concat demuxer failed (codec mismatch) — re-encode
                logger.warning("Concat demuxer failed, falling back to re-encode")
                self._concatenate_with_reencode(video_paths, output_path)

        finally:
            if concat_file:
                try:
                    os.unlink(concat_file)
                except OSError:
                    pass

    def _concatenate_with_reencode(
        self, video_paths: List[str], output_path: str
    ) -> None:
        """Fallback: concatenate by re-encoding all inputs to uniform format.

        The concat filter needs identical frame sizes, rates and audio
        layouts, so every clip is scaled (letterboxed) to the most common
        format first. That mismatch is exactly why this fallback runs.
        """
        from src.utils.media_probe import probe

        formats = []
        for vp in video_paths:
            info = probe(vp)
            if info and info.video and info.video.width and info.video.height:
                formats.append((info.video.width, info.video.height, info.video.fps or 30.0))
        width, height, fps = max(set(formats), key=formats.count) if formats else (1280, 720, 30.0)

        input_args = []
        filters = []
        pairs = []
        for i, vp in enumerate(video_paths):
            input_args.extend(["-i", vp])
            filters.append(
                f"[{i}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{i}]"
            )
            filters.append(f"[{i}:a:0]aformat=sample_rates=48000:channel_layouts=mono[a{i}]")
            pairs.append(f"[v{i}][a{i}]")

        n = len(video_paths)
        filter_complex = "; ".join(filters) + f"; {''.join(pairs)}concat=n={n}:v=1:a=1[vout][aout]"

        cmd = [
            get_ffmpeg(), "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        self._run_ffmpeg(cmd, timeout=self.CONCAT_TIMEOUT)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _run_ffmpeg(self, cmd: List[str], timeout: int = 120) -> None:
        """Run an FFmpeg command and raise on failure."""
        logger.debug(f"FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            stderr = result.stderr[-500:] if result.stderr else "No stderr."
            raise RuntimeError(f"FFmpeg exit code {result.returncode}: {stderr}")

    def _get_media_duration(self, file_path: str) -> float:
        """Get media duration in seconds using ffprobe. Returns -1.0 on failure."""
        try:
            result = subprocess.run(
                [
                    get_ffprobe(), "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception as exc:
            logger.warning("Duration detection failed for %s: %s", file_path, exc)
            return -1.0
