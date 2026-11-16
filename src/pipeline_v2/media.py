"""Audio/video composition, fallback clips, concatenation, and final A/V checks."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional, Protocol

from PIL import Image, ImageDraw, ImageFont

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

from src.utils import media_probe
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe

from .models import TimelineSegment

# Delivery loudness (EBU R128 speech convention) and one sample rate everywhere.
NARRATION_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"
NARRATION_CLIP_TOLERANCE_SECONDS = 0.05
_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
# Windows caps a command line near 32k characters; long graphs go to a file.
_INLINE_FILTER_LIMIT = 6000


class SegmentRenderer(Protocol):
    def render(self, segment: TimelineSegment, output_path: Path) -> Path: ...


def compose_scene(video: Path, audio: Path, output: Path) -> Path:
    """Merge a render with its narration without ever clipping the narration.

    Narration longer than the animation holds the last frame; an animation
    longer than the narration gets trailing silence instead of being cut.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    video_seconds = _stream_seconds(video, "video")
    audio_seconds = _stream_seconds(audio, "audio")
    video_filter: Optional[str] = None
    audio_filter = NARRATION_FILTER
    if video_seconds is not None and audio_seconds is not None:
        gap = audio_seconds - video_seconds
        if gap > NARRATION_CLIP_TOLERANCE_SECONDS:
            video_filter = f"tpad=stop_mode=clone:stop_duration={gap:.3f}"
        elif gap < 0:
            audio_filter += f",apad=whole_dur={video_seconds:.3f}"
    # Unknown durations: no filter and no -shortest, so the longer stream wins.
    command = [
        get_ffmpeg(), "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        *(["-vf", video_filter] if video_filter else []),
        "-af", audio_filter, *_VIDEO_ARGS, *_AUDIO_ARGS, str(output),
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
    """Join clips into one uniform file, whatever each clip's own format is.

    Every clip is scaled and padded to the most common frame size and rate
    and resampled to 48 kHz stereo (silence when it has no audio track), so
    a 1080p fallback page or a silent clip cannot corrupt the joined video.
    """
    if not videos:
        raise RuntimeError("No scene videos are available for final composition")
    output.parent.mkdir(parents=True, exist_ok=True)
    infos = [media_probe.probe(str(path)) for path in videos]
    if any(info is None or info.video is None for info in infos):
        # Formats unknown: join as before and let ffmpeg report a bad clip.
        return _concatenate_demuxer(videos, output)

    width, height, fps = Counter(
        (info.video.width, info.video.height, info.video.fps) for info in infos
    ).most_common(1)[0][0]
    width, height = _even(width or 1280), _even(height or 720)
    rate = f"{fps or 30.0:g}"
    chains: list[str] = []
    for index, info in enumerate(infos):
        chains.append(
            f"[{index}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={rate},setsar=1,"
            f"format=yuv420p[v{index}]"
        )
        if info.audio is not None:
            chains.append(
                f"[{index}:a:0]aformat=sample_rates=48000:channel_layouts=stereo[a{index}]"
            )
        else:
            seconds = info.video.duration or info.duration or 0.0
            chains.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={max(seconds, 0.05):.3f}[a{index}]"
            )
    pads = "".join(f"[v{index}][a{index}]" for index in range(len(videos)))
    graph = ";".join(chains) + f";{pads}concat=n={len(videos)}:v=1:a=1[v][a]"

    inputs = [arg for path in videos for arg in ("-i", str(path))]
    encode = [
        "-map", "[v]", "-map", "[a]", *_VIDEO_ARGS, *_AUDIO_ARGS,
        "-movflags", "+faststart", str(output),
    ]
    if len(graph) <= _INLINE_FILTER_LIMIT:
        _run([get_ffmpeg(), "-y", *inputs, "-filter_complex", graph, *encode], 900, "final concatenation")
        return output.resolve()
    script = output.parent / "concat_filter.txt"
    script.write_text(graph, encoding="utf-8")
    try:
        _run(
            [get_ffmpeg(), "-y", *inputs, "-/filter_complex", str(script), *encode],
            900, "final concatenation",
        )
    except RuntimeError as exc:
        # ffmpeg before 7 has no "-/option" file syntax, only the script
        # option that newer builds removed. Any other failure is real.
        if "Unrecognized option" not in str(exc):
            raise
        _run(
            [get_ffmpeg(), "-y", *inputs, "-filter_complex_script", str(script), *encode],
            900, "final concatenation",
        )
    return output.resolve()


def _concatenate_demuxer(videos: list[Path], output: Path) -> Path:
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


def _stream_seconds(path: Path, kind: str) -> Optional[float]:
    info = media_probe.probe(str(path))
    if info is None:
        return None
    stream = info.video if kind == "video" else info.audio
    seconds = (stream.duration if stream else None) or info.duration
    return seconds if seconds and seconds > 0 else None


def _even(value: int) -> int:
    return value - value % 2


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
