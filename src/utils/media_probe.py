"""Measure media files with ffprobe/ffmpeg for the delivery gate.

Every function returns None (or an empty result) instead of raising when a
tool is missing or a file is unreadable: an unmeasurable check is reported
as such, it never crashes the pipeline.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe

logger = logging.getLogger(__name__)

# Delivery loudness target (EBU R128 speech/podcast convention).
TARGET_LUFS = -16.0
TARGET_TRUE_PEAK = -1.5
TARGET_LRA = 11.0


@dataclass
class StreamInfo:
    codec: str = ""
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    pix_fmt: str = ""
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


@dataclass
class MediaInfo:
    duration: Optional[float] = None
    video: Optional[StreamInfo] = None
    audio: Optional[StreamInfo] = None


@dataclass
class Loudness:
    integrated: float
    true_peak: float
    lra: Optional[float] = None
    threshold: Optional[float] = None


def _run(cmd: List[str], timeout: int = 120) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        logger.warning("Media tool failed (%s): %s", cmd[0], exc)
        return None


def _float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def _fps(rate: str) -> Optional[float]:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        num_f, den_f = _float(num), _float(den)
        return round(num_f / den_f, 3) if num_f is not None and den_f else None
    return _float(rate)


def parse_ffprobe_json(text: str) -> MediaInfo:
    """Parse ``ffprobe -show_format -show_streams -of json`` output."""
    data = json.loads(text or "{}")
    info = MediaInfo(duration=_float((data.get("format") or {}).get("duration")))
    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and info.video is None:
            info.video = StreamInfo(
                codec=stream.get("codec_name", ""),
                duration=_float(stream.get("duration")),
                width=stream.get("width"),
                height=stream.get("height"),
                fps=_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""),
                pix_fmt=stream.get("pix_fmt", ""),
            )
        elif kind == "audio" and info.audio is None:
            info.audio = StreamInfo(
                codec=stream.get("codec_name", ""),
                duration=_float(stream.get("duration")),
                sample_rate=int(stream["sample_rate"]) if str(stream.get("sample_rate", "")).isdigit() else None,
                channels=stream.get("channels"),
            )
    return info


def probe(path: str) -> Optional[MediaInfo]:
    result = _run([
        get_ffprobe(), "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ], timeout=30)
    if result is None or result.returncode != 0:
        return None
    try:
        return parse_ffprobe_json(result.stdout)
    except (ValueError, TypeError) as exc:
        logger.warning("Unparseable ffprobe output for %s: %s", path, exc)
        return None


def parse_loudnorm_json(stderr: str) -> Optional[Loudness]:
    """Extract input loudness from ffmpeg's ``loudnorm=print_format=json`` log."""
    match = None
    for match in re.finditer(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr or ""):
        pass
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    integrated, true_peak = _float(data.get("input_i")), _float(data.get("input_tp"))
    if integrated is None or true_peak is None:
        return None
    return Loudness(
        integrated=integrated,
        true_peak=true_peak,
        lra=_float(data.get("input_lra")),
        threshold=_float(data.get("input_thresh")),
    )


def measure_loudness(path: str) -> Optional[Loudness]:
    result = _run([
        get_ffmpeg(), "-hide_banner", "-nostats", "-i", str(path), "-vn",
        "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}:print_format=json",
        "-f", "null", "-",
    ])
    if result is None or result.returncode != 0:
        return None
    return parse_loudnorm_json(result.stderr)


def parse_silencedetect(stderr: str, total: Optional[float] = None) -> List[Tuple[float, float]]:
    """Silence intervals from ffmpeg's ``silencedetect`` log.

    A silence still open at end of file closes at ``total`` when given.
    """
    intervals: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for line in (stderr or "").splitlines():
        started = re.search(r"silence_start:\s*(-?[0-9.]+)", line)
        if started:
            start = max(0.0, float(started.group(1)))
            continue
        ended = re.search(r"silence_end:\s*([0-9.]+)", line)
        if ended and start is not None:
            intervals.append((start, float(ended.group(1))))
            start = None
    if start is not None and total is not None:
        intervals.append((start, total))
    return intervals


def detect_silences(
    path: str,
    noise_db: float = -40.0,
    min_duration: float = 2.0,
    total: Optional[float] = None,
) -> Optional[List[Tuple[float, float]]]:
    result = _run([
        get_ffmpeg(), "-hide_banner", "-nostats", "-i", str(path), "-vn",
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ])
    if result is None or result.returncode != 0:
        return None
    return parse_silencedetect(result.stderr, total)


def normalize_audio_track(src: str, dst: str, measured: Optional[Loudness] = None) -> bool:
    """Re-encode ``src`` with its audio normalized to the delivery target.

    Uses ffmpeg's linear (two-pass) loudnorm when a measurement is given,
    which keeps speech dynamics intact; video is stream-copied.
    """
    loudnorm = f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}"
    # Linear mode needs all four measurements; otherwise ffmpeg silently
    # falls back to dynamic normalization.
    if measured is not None and measured.lra is not None and measured.threshold is not None:
        loudnorm += (
            f":measured_I={measured.integrated}:measured_TP={measured.true_peak}"
            f":measured_LRA={measured.lra}:measured_thresh={measured.threshold}"
            ":linear=true"
        )
    result = _run([
        get_ffmpeg(), "-y", "-i", str(src),
        "-map", "0:v?", "-map", "0:a:0",
        "-c:v", "copy", "-af", loudnorm, "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
        str(dst),
    ])
    return result is not None and result.returncode == 0
