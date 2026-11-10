"""Representative screenshot extraction for preview videos."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.manim_verifier.models import FrameExtractionResult, VerificationIssue
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe


class FrameExtractor:
    SAMPLE_FRACTIONS = (0.25, 0.50, 0.75, 1.0)

    def extract(self, video_path: Path, output_dir: Path) -> FrameExtractionResult:
        video_path = video_path.resolve()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        duration = self._probe_duration(video_path)
        if duration <= 0:
            return FrameExtractionResult(issues=[VerificationIssue(
                code="FRAME_VIDEO_INVALID",
                severity="error",
                message=f"Could not read preview duration: {video_path}",
            )])

        timestamps = self._sample_timestamps(duration)
        paths: list[str] = []
        issues: list[VerificationIssue] = []
        for index, timestamp in enumerate(timestamps):
            frame_path = output_dir / f"frame-{index:02d}-{timestamp:.3f}s.png"
            result = self._extract_one(video_path, frame_path, timestamp)
            if (
                result is None
                or result.returncode != 0
                or not frame_path.exists()
                or frame_path.stat().st_size == 0
            ):
                fallback_timestamp = max(0.0, timestamp - min(0.1, duration * 0.1))
                result = self._extract_one(video_path, frame_path, fallback_timestamp)
            if (
                result is None
                or result.returncode != 0
                or not frame_path.exists()
                or frame_path.stat().st_size == 0
            ):
                detail = "ffmpeg invocation failed"
                if result is not None:
                    detail = (result.stderr or result.stdout or "unknown ffmpeg error")[-800:]
                issues.append(VerificationIssue(
                    code="FRAME_EXTRACTION_FAILED",
                    severity="warning",
                    message=f"Frame at {timestamp:.3f}s failed: {detail}",
                    timestamp=timestamp,
                ))
                continue
            paths.append(str(frame_path.resolve()))

        return FrameExtractionResult(
            frame_paths=paths,
            timestamps=timestamps,
            issues=issues,
        )

    @staticmethod
    def _sample_timestamps(duration: float) -> list[float]:
        # Container duration can extend beyond the final decodable frame. Leave
        # at least 100 ms (or 20% for a tiny clip) before EOF for the last sample.
        final = max(0.0, duration - min(0.1, duration * 0.2))
        values = [duration * fraction for fraction in (0.25, 0.50, 0.75)] + [final]
        return [round(max(0.0, min(final, value)), 6) for value in values]

    @staticmethod
    def _extract_one(
        video_path: Path,
        frame_path: Path,
        timestamp: float,
    ) -> subprocess.CompletedProcess[str] | None:
        command = [
            get_ffmpeg(),
            "-y",
            "-i",
            str(video_path),
            "-ss",
            f"{timestamp:.6f}",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=30)
        except Exception:
            return None

    @staticmethod
    def _probe_duration(video_path: Path) -> float:
        if not video_path.exists():
            return 0.0
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
