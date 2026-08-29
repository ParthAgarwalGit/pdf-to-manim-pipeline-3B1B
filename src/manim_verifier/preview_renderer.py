"""Low-quality raw-Manim renderer guarded by static verification."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional

from src.manim_verifier.models import VerificationIssue, VerificationReport
from src.manim_verifier.static_verifier import StaticManimVerifier
from src.utils.ffmpeg_finder import get_ffprobe


class RawManimPreviewRenderer:
    QUALITY_FLAGS = {
        "low": "-ql",
        "medium": "-qm",
        "high": "-qh",
    }

    def __init__(self, verifier: Optional[StaticManimVerifier] = None):
        self.verifier = verifier or StaticManimVerifier()

    def render(
        self,
        source: str,
        attempt_dir: Path,
        *,
        quality: str = "low",
        timeout_seconds: int = 120,
    ) -> VerificationReport:
        report = self.verifier.verify(source)
        attempt_dir = attempt_dir.resolve()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = attempt_dir / "render_stdout.txt"
        stderr_path = attempt_dir / "render_stderr.txt"
        report.stdout_path = str(stdout_path)
        report.stderr_path = str(stderr_path)

        if not report.safe or not report.scene_class:
            report.render_error = "Static safety verification failed; generated code was not executed."
            self._write_logs(stdout_path, stderr_path, "", report.render_error)
            return report

        scene_path = attempt_dir / "scene.py"
        scene_path.write_text(source, encoding="utf-8")
        media_dir = attempt_dir / "media"
        quality_flag = self.QUALITY_FLAGS.get(quality, "-ql")
        command = [
            sys.executable,
            "-m",
            "manim",
            quality_flag,
            "--media_dir",
            str(media_dir),
            str(scene_path),
            report.scene_class,
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=str(attempt_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=self._sanitized_environment(os.environ),
            )
            self._write_logs(stdout_path, stderr_path, completed.stdout, completed.stderr)
        except subprocess.TimeoutExpired as exc:
            stdout = self._decode_timeout_stream(exc.stdout)
            stderr = self._decode_timeout_stream(exc.stderr)
            self._write_logs(stdout_path, stderr_path, stdout, stderr)
            report.render_error = f"Preview render timed out after {timeout_seconds}s."
            report.issues.append(VerificationIssue(
                code="RENDER_TIMEOUT",
                severity="error",
                message=report.render_error,
            ))
            return report
        except Exception as exc:
            report.render_error = f"Preview renderer could not start Manim: {exc}"
            self._write_logs(stdout_path, stderr_path, "", report.render_error)
            report.issues.append(VerificationIssue(
                code="RENDER_START_FAILED",
                severity="error",
                message=report.render_error,
            ))
            return report

        if completed.returncode != 0:
            detail = self._tail(completed.stderr or completed.stdout)
            report.render_error = f"Manim exited with code {completed.returncode}: {detail}"
            report.issues.append(VerificationIssue(
                code="MANIM_RENDER_FAILED",
                severity="error",
                message=report.render_error,
            ))
            return report

        rendered = self._find_video(media_dir, report.scene_class)
        if rendered is None:
            report.render_error = "Manim completed but no final MP4 was found."
            report.issues.append(VerificationIssue(
                code="VIDEO_NOT_FOUND",
                severity="error",
                message=report.render_error,
            ))
            return report

        final_video = attempt_dir / "video.mp4"
        shutil.copy2(rendered, final_video)
        report.render_success = True
        report.video_path = str(final_video.resolve())
        report.duration_seconds = self._probe_duration(final_video)
        return report

    @staticmethod
    def _sanitized_environment(source: Mapping[str, str]) -> dict[str, str]:
        """Remove credentials before the generated subprocess starts."""
        blocked_fragments = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        return {
            key: value
            for key, value in source.items()
            if not any(fragment in key.upper() for fragment in blocked_fragments)
        }

    @staticmethod
    def _find_video(media_dir: Path, scene_class: str) -> Optional[Path]:
        for path in media_dir.rglob(f"{scene_class}.mp4"):
            if "partial_movie_files" not in path.parts:
                return path
        return None

    @staticmethod
    def _probe_duration(video_path: Path) -> float:
        try:
            result = subprocess.run(
                [
                    get_ffprobe(),
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return max(0.0, float(result.stdout.strip()))
        except Exception:
            return 0.0

    @staticmethod
    def _write_logs(stdout_path: Path, stderr_path: Path, stdout: str, stderr: str) -> None:
        stdout_path.write_text(stdout or "", encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")

    @staticmethod
    def _decode_timeout_stream(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""

    @staticmethod
    def _tail(text: str, limit: int = 2000) -> str:
        return (text or "No renderer output.").strip()[-limit:]
