"""Joining many clips keeps every clip's picture on its own sound.

Each clip shows one solid color and beeps at its very start; its picture is a
little longer than its sound, as Manim renders and presenter clips often are.
After the join, every color change must land where its beep starts: a join
that let either stream drift would push later clips' pictures off their sound.
"""

import subprocess
from pathlib import Path

import numpy as np

from src.pipeline_v2 import media
from src.utils.ffmpeg_finder import get_ffmpeg

COLORS = ["red", "blue", "green", "yellow", "white", "purple"] * 2
FPS = 30
BEEP_SECONDS = 0.08


def _clip(path: Path, color: str, video_seconds: float, audio_seconds: float) -> Path:
    subprocess.run([
        get_ffmpeg(), "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=160x90:r={FPS}:d={video_seconds:.4f}",
        "-f", "lavfi", "-i",
        f"sine=f=1000:sample_rate=48000:d={BEEP_SECONDS},apad=whole_dur={audio_seconds:.4f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(path),
    ], check=True)
    return path


def _color_changes(video: Path) -> list[float]:
    raw = subprocess.run(
        [get_ffmpeg(), "-v", "error", "-i", str(video), "-vf", f"fps={FPS},scale=8:8",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8 * 8 * 3).astype(float)
    means = frames.reshape(len(frames), -1, 3).mean(axis=1)
    jumps = np.abs(np.diff(means, axis=0)).sum(axis=1) > 60
    return [(index + 1) / FPS for index in np.flatnonzero(jumps)]


def _beep_starts(video: Path) -> list[float]:
    rate = 8000
    raw = subprocess.run(
        [get_ffmpeg(), "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(rate),
         "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    samples = np.abs(np.frombuffer(raw, dtype=np.int16).astype(float))
    window = rate // 200  # 5 ms
    loud = samples[: len(samples) // window * window].reshape(-1, window).max(axis=1) > 2000
    starts = np.flatnonzero(loud[1:] & ~loud[:-1]) + 1
    if loud[0]:
        starts = np.concatenate([[0], starts])
    return [index * window / rate for index in starts]


def test_every_clip_stays_on_its_own_sound_after_the_join(tmp_path):
    clips = []
    for index, color in enumerate(COLORS):
        audio_seconds = 1.0 + 0.013 * index  # never a whole number of frames
        video_seconds = audio_seconds + 0.02 + 0.011 * (index % 3)  # picture a little longer
        clips.append(_clip(tmp_path / f"c{index:02d}.mp4", color, video_seconds, audio_seconds))

    final = media.concatenate(clips, tmp_path / "final.mp4")

    beeps = _beep_starts(final)
    changes = _color_changes(final)
    assert len(beeps) == len(COLORS) and len(changes) == len(COLORS) - 1
    for boundary, (change, beep) in enumerate(zip(changes, beeps[1:]), start=1):
        assert abs(change - beep) <= 1.5 / FPS, (
            f"clip {boundary}: picture changes at {change:.3f}s but its sound starts at {beep:.3f}s"
        )
