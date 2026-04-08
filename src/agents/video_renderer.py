"""Video Renderer Agent — renders Manim scenes and assembles final video.

Responsibilities:
  1. Run `manim -qm --media_dir <output>/media` on each scene Python file
  2. Collect output .mp4 paths from the known media directory
  3. Concatenate all scene videos into one final video using ffmpeg
  4. Return the path to the final assembled video
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.utils.ffmpeg_finder import ffmpeg_available, get_ffmpeg, get_ffprobe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SceneRenderResult:
    """Result of rendering a single scene."""
    scene_id: str
    success: bool
    video_path: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# I/O Contracts
# ---------------------------------------------------------------------------

class VideoRendererInput(AgentInput):
    """Input to the Video Renderer Agent."""

    def __init__(
        self,
        scene_files: List[str],
        output_dir: str,
        quality: str = "m",
        final_video_name: str = "final_output.mp4",
    ):
        self.scene_files = scene_files          # Absolute paths to scene_*.py files
        self.output_dir = output_dir            # Pipeline output directory
        self.quality = quality                  # l=low, m=medium, h=high, k=4K
        self.final_video_name = final_video_name


class VideoRendererOutput(AgentOutput):
    """Output: video rendered and assembled successfully."""

    def __init__(
        self,
        final_video_path: str,
        scene_results: List[SceneRenderResult],
        total_duration: float,
    ):
        self.final_video_path = final_video_path
        self.scene_results = scene_results
        self.total_duration = total_duration


class VideoRendererFailure(AgentOutput):
    """Output: rendering failed for some or all scenes."""

    def __init__(
        self,
        scene_results: List[SceneRenderResult],
        error: str,
    ):
        self.scene_results = scene_results
        self.error = error


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VideoRendererAgent(Agent):
    """Renders Manim scenes and assembles them into a final video."""

    QUALITY_FLAGS = {
        "l": "-ql",   # 480p15
        "m": "-qm",   # 720p30
        "h": "-qh",   # 1080p60
        "k": "-qk",   # 4K60
    }

    # Manim quality flag → subdirectory name in media/videos/
    QUALITY_SUBDIRS = {
        "-ql": "480p15",
        "-qm": "720p30",
        "-qh": "1080p60",
        "-qk": "2160p60",
    }

    def execute(self, input_data: VideoRendererInput) -> AgentOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError("INVALID_INPUT", "Expected VideoRendererInput.")

        # Preflight: ensure ffmpeg is reachable before doing any work.
        if not ffmpeg_available():
            logger.error("FFmpeg not found — cannot render or concatenate video.")
            return VideoRendererFailure(
                scene_results=[],
                error=(
                    "FFmpeg is not installed or not on PATH. "
                    "Install it (https://ffmpeg.org/download.html) and retry."
                ),
            )

        if not input_data.scene_files:
            return VideoRendererFailure(
                scene_results=[],
                error="No scene files provided for rendering.",
            )

        quality_flag = self.QUALITY_FLAGS.get(input_data.quality, "-qm")

        # Use a media directory inside the pipeline output so we always
        # know where Manim puts the rendered videos.
        media_dir = str(Path(input_data.output_dir) / "media")

        # 1. Render each scene
        scene_results: List[SceneRenderResult] = []
        rendered_videos: List[str] = []

        for scene_file in input_data.scene_files:
            result = self._render_scene(scene_file, quality_flag, media_dir)
            scene_results.append(result)
            if result.success and result.video_path is not None:
                rendered_videos.append(result.video_path)

        if not rendered_videos:
            return VideoRendererFailure(
                scene_results=scene_results,
                error="All scene renders failed. Check individual scene errors.",
            )

        # 2. Concatenate into final video
        final_path = os.path.join(input_data.output_dir, input_data.final_video_name)
        try:
            if len(rendered_videos) == 1:
                # Single scene — just copy
                import shutil
                shutil.copy2(rendered_videos[0], final_path)
            else:
                self._concatenate_videos(rendered_videos, final_path)

            total_duration = sum(r.duration_seconds for r in scene_results if r.success)

            logger.info(
                f"Final video assembled: {final_path} "
                f"({len(rendered_videos)}/{len(scene_results)} scenes, {total_duration:.1f}s)"
            )
            return VideoRendererOutput(
                final_video_path=final_path,
                scene_results=scene_results,
                total_duration=total_duration,
            )

        except Exception as e:
            logger.error(f"Video concatenation failed: {e}")
            return VideoRendererFailure(
                scene_results=scene_results,
                error=f"Concatenation failed: {e}",
            )

    def validate_input(self, input_data: AgentInput) -> bool:
        return isinstance(input_data, VideoRendererInput)

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=2,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=2.0,
            max_delay_seconds=10.0,
            retryable_errors=["RENDER_TIMEOUT"],
        )

    # ------------------------------------------------------------------
    # Scene rendering
    # ------------------------------------------------------------------

    def _render_scene(
        self, scene_file: str, quality_flag: str, media_dir: str
    ) -> SceneRenderResult:
        """Run `manim` on a single scene file and find the output video."""
        scene_stem = Path(scene_file).stem
        scene_id = scene_stem[len("scene_"):] if scene_stem.startswith("scene_scene_") else scene_stem

        if not os.path.exists(scene_file):
            return SceneRenderResult(
                scene_id=scene_id, success=False,
                error=f"Scene file not found: {scene_file}",
            )

        # Detect class name from the file
        class_name = self._detect_scene_class(scene_file)
        if not class_name:
            return SceneRenderResult(
                scene_id=scene_id, success=False,
                error="No Scene subclass found in file.",
            )

        try:
            # Build manim command with explicit --media_dir so the output
            # lands in a predictable location inside the pipeline output.
            cmd = [
                "manim", quality_flag,
                "--media_dir", media_dir,
                scene_file, class_name,
            ]
            logger.info(f"Rendering: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min per scene
            )

            if result.returncode != 0:
                stderr = result.stderr[:800] if result.stderr else "No stderr."
                stdout = result.stdout[:400] if result.stdout else ""
                logger.error(f"manim stderr: {stderr}")
                if stdout:
                    logger.error(f"manim stdout: {stdout}")
                return SceneRenderResult(
                    scene_id=scene_id, success=False,
                    error=f"manim exit code {result.returncode}: {stderr}",
                )

            # Find the rendered video
            video_path = self._find_rendered_video(
                scene_file, class_name, quality_flag, media_dir
            )
            if not video_path:
                return SceneRenderResult(
                    scene_id=scene_id, success=False,
                    error=(
                        "Render succeeded but output video not found. "
                        f"Searched in {media_dir}"
                    ),
                )

            # Get video duration via ffprobe (best-effort)
            duration = self._get_video_duration(video_path)

            return SceneRenderResult(
                scene_id=scene_id,
                success=True,
                video_path=video_path,
                duration_seconds=duration,
            )

        except FileNotFoundError:
            return SceneRenderResult(
                scene_id=scene_id, success=False,
                error=(
                    "manim CLI not found in PATH. Install with: "
                    "pip install manim"
                ),
            )
        except subprocess.TimeoutExpired:
            return SceneRenderResult(
                scene_id=scene_id, success=False,
                error="Render timed out after 300s.",
            )
        except Exception as e:
            return SceneRenderResult(
                scene_id=scene_id, success=False,
                error=f"Unexpected error: {e}",
            )

    def _detect_scene_class(self, scene_file: str) -> Optional[str]:
        """Parse the file to find the Scene subclass name."""
        try:
            with open(scene_file, "r", encoding="utf-8") as f:
                source = f.read()
            import ast
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id in (
                            "Scene", "ThreeDScene", "MovingCameraScene"
                        ):
                            return node.name
        except Exception as exc:
            logger.error(f"Failed to parse scene file {scene_file}: {exc}")
        return None

    def _find_rendered_video(
        self,
        scene_file: str,
        class_name: str,
        quality_flag: str,
        media_dir: str,
    ) -> Optional[str]:
        """Search for the rendered video in the known media directory.

        Manim writes to:
          {media_dir}/videos/{filename_stem}/{quality_subdir}/{ClassName}.mp4

        We check the expected path first, then fall back to a recursive
        glob so we never miss the file even if Manim changes its layout.
        """
        file_stem = Path(scene_file).stem
        quality_subdir = self.QUALITY_SUBDIRS.get(quality_flag, "")
        media_path = Path(media_dir)

        # 1. Check the exact expected path
        if quality_subdir:
            expected = media_path / "videos" / file_stem / quality_subdir / f"{class_name}.mp4"
            if expected.exists():
                return str(expected)

        # 2. Search all quality subdirs under the file stem
        videos_dir = media_path / "videos" / file_stem
        if videos_dir.exists():
            for qdir in videos_dir.iterdir():
                if qdir.is_dir():
                    video = qdir / f"{class_name}.mp4"
                    if video.exists():
                        return str(video)

        # 3. Recursive glob as last resort (covers unexpected layouts)
        #    SKIP partial_movie_files — those are per-animation fragments,
        #    not the final composed scene video.
        for mp4 in media_path.rglob(f"{class_name}.mp4"):
            if "partial_movie_files" not in str(mp4):
                return str(mp4)

        # 4. Scene-dir-relative media/ as last resort (older Manim versions
        #    that ignore --media_dir).  CWD-relative search is intentionally
        #    omitted to avoid picking up files from unrelated runs.
        scene_media = Path(scene_file).parent / "media" / "videos" / file_stem
        if scene_media.exists():
            for qdir in scene_media.iterdir():
                if qdir.is_dir():
                    video = qdir / f"{class_name}.mp4"
                    if video.exists():
                        return str(video)

        return None

    # ------------------------------------------------------------------
    # Video concatenation
    # ------------------------------------------------------------------

    def _concatenate_videos(self, video_paths: List[str], output_path: str) -> None:
        """Concatenate multiple .mp4 files using ffmpeg's concat demuxer."""
        # Create concat file list
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for vp in video_paths:
                # ffmpeg requires forward slashes and escaped quotes
                escaped = vp.replace("\\", "/").replace("'", "'\\''")  # escape for FFmpeg concat
                f.write(f"file '{escaped}'\n")
            concat_file = f.name

        try:
            cmd = [
                get_ffmpeg(), "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

        finally:
            try:
                os.unlink(concat_file)
            except Exception:
                pass

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    get_ffprobe(), "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0
