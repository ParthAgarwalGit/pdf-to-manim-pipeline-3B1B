"""Audio/video composition, fallback clips, concatenation, and final A/V checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFont

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe

from .models import TimelineSegment


class SegmentRenderer(Protocol):
    def render(self, segment: TimelineSegment, output_path: Path) -> Path: ...


def compose_scene(video: Path, audio: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        get_ffmpeg(), "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
        "-shortest", str(output),
    ]
    _run(command, 300, "scene composition")
    return output.resolve()


def source_page_fallback(
    pdf_path: Path,
    page_number: int,
    audio_path: Path,
    output_path: Path,
) -> Path:
    image_path = output_path.with_suffix(".png")
    with pymupdf.open(pdf_path) as document:
        page = document[max(0, min(len(document) - 1, page_number - 1))]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), alpha=False)
        pixmap.save(str(image_path))
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        draw = ImageDraw.Draw(image)
        banner_height = max(40, image.height // 18)
        draw.rectangle((0, 0, image.width, banner_height), fill=(20, 20, 20))
        font = ImageFont.load_default(size=max(18, banner_height // 2))
        draw.text(
            (16, banner_height // 2),
            f"SOURCE PAGE FALLBACK - PAGE {page_number}",
            fill=(255, 220, 70),
            font=font,
            anchor="lm",
        )
        image.save(image_path)
    duration = _duration_seconds(audio_path)
    command = [
        get_ffmpeg(), "-y", "-loop", "1", "-i", str(image_path), "-i", str(audio_path),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
        "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}", "-shortest", str(output_path),
    ]
    _run(command, 300, "source-page fallback")
    return output_path.resolve()


def concatenate(videos: list[Path], output: Path) -> Path:
    if not videos:
        raise RuntimeError("No scene videos are available for final composition")
    output.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.parent / "concat.txt"
    lines = [f"file '{str(path.resolve()).replace(chr(39), chr(39) * 2)}'" for path in videos]
    list_path.write_text("\n".join(lines), encoding="utf-8")
    command = [
        get_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", str(output),
    ]
    _run(command, 900, "final concatenation")
    return output.resolve()


def inspect_av(path: Path) -> dict:
    command = [
        get_ffprobe(), "-v", "quiet", "-print_format", "json", "-show_streams",
        "-show_format", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"valid": False, "error": result.stderr[-800:], "video": False, "audio": False}
    payload = json.loads(result.stdout or "{}")
    kinds = {stream.get("codec_type") for stream in payload.get("streams", [])}
    return {
        "valid": "video" in kinds and "audio" in kinds,
        "video": "video" in kinds,
        "audio": "audio" in kinds,
        "duration_seconds": float(payload.get("format", {}).get("duration") or 0.0),
        "streams": payload.get("streams", []),
    }


def _run(command: list[str], timeout: int, label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed: {(result.stderr or result.stdout)[-1600:]}")


def _duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            get_ffprobe(), "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not determine narration duration: {result.stderr[-800:]}") from exc
    if result.returncode != 0 or duration <= 0:
        raise RuntimeError(f"Could not determine narration duration: {result.stderr[-800:]}")
    return duration
