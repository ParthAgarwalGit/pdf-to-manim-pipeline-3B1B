"""FFmpeg binary discovery utility.

Finds ffmpeg and ffprobe executables via:
  1. System PATH (standard lookup)
  2. Manim's bundled / configured FFmpeg (fallback)
  3. Common installation directories on Windows (last resort)

Usage:
    from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe

    ffmpeg  = get_ffmpeg()   # returns absolute path or "ffmpeg"
    ffprobe = get_ffprobe()  # returns absolute path or "ffprobe"
"""

import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    """Return the best available path to the ffmpeg executable."""
    path = _find_binary("ffmpeg")
    if path:
        logger.info(f"FFmpeg found: {path}")
    else:
        logger.warning("FFmpeg not found — audio-video composition will fail")
        path = "ffmpeg"  # fall back to bare name (will raise FileNotFoundError)
    return path


@lru_cache(maxsize=1)
def get_ffprobe() -> str:
    """Return the best available path to the ffprobe executable."""
    path = _find_binary("ffprobe")
    if path:
        logger.info(f"ffprobe found: {path}")
    else:
        logger.warning("ffprobe not found — duration detection will be unavailable")
        path = "ffprobe"
    return path


def ffmpeg_available() -> bool:
    """Quick check: can we call ffmpeg at all?"""
    try:
        ffmpeg = get_ffmpeg()
        r = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


# ------------------------------------------------------------------
# Internal discovery
# ------------------------------------------------------------------

def _find_binary(name: str) -> Optional[str]:
    """Search for a binary (ffmpeg or ffprobe) across all known locations."""
    # 1. System PATH
    found = shutil.which(name)
    if found:
        return found

    # 2. Manim's configured / bundled FFmpeg
    manim_path = _find_via_manim(name)
    if manim_path:
        return manim_path

    # 3. Common Windows installation directories
    if sys.platform == "win32":
        win_path = _find_windows_common(name)
        if win_path:
            return win_path

    return None


def _find_via_manim(name: str) -> Optional[str]:
    """Try to locate the binary through Manim's configuration."""
    try:
        # Manim Community stores its config (including ffmpeg path) in
        # manim._config.config or manim.config
        from manim import config as manim_config

        # The config object has an `ffmpeg_executable` attribute
        if name == "ffmpeg":
            ffmpeg_path = getattr(manim_config, "ffmpeg_executable", None)
            if ffmpeg_path and _is_valid_binary(ffmpeg_path):
                return str(ffmpeg_path)

        # ffprobe is usually next to ffmpeg
        ffmpeg_path = getattr(manim_config, "ffmpeg_executable", None)
        if ffmpeg_path:
            ffmpeg_dir = Path(ffmpeg_path).parent
            candidate = ffmpeg_dir / (name + (".exe" if sys.platform == "win32" else ""))
            if candidate.is_file():
                return str(candidate)

    except ImportError:
        pass
    except Exception as exc:
        logger.debug(f"Manim config probe failed: {exc}")

    # Also check the Python Scripts directory (pip-installed manim
    # sometimes places ffmpeg there on Windows)
    try:
        scripts_dir = Path(sys.executable).parent / "Scripts"
        candidate = scripts_dir / (name + (".exe" if sys.platform == "win32" else ""))
        if candidate.is_file():
            return str(candidate)

        # Also check the same directory as the Python executable
        candidate = Path(sys.executable).parent / (
            name + (".exe" if sys.platform == "win32" else "")
        )
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass

    return None


def _find_windows_common(name: str) -> Optional[str]:
    """Check common FFmpeg install locations on Windows."""
    exe = name + ".exe"
    search_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ffmpeg" / "bin",
        Path(os.environ.get("PROGRAMFILES", "")) / "ffmpeg" / "bin",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "ffmpeg" / "bin",
        Path("C:/ffmpeg/bin"),
        Path("C:/tools/ffmpeg/bin"),
        # Chocolatey
        Path("C:/ProgramData/chocolatey/bin"),
        # Scoop
        Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims",
    ]

    for search_dir in search_paths:
        try:
            candidate = search_dir / exe
            if candidate.is_file():
                return str(candidate)
        except Exception:
            continue

    # Search PATH-adjacent directories (e.g., manim's bundled copy
    # might be in a site-packages directory)
    try:
        import importlib.util
        manim_spec = importlib.util.find_spec("manim")
        if manim_spec and manim_spec.origin:
            manim_dir = Path(manim_spec.origin).parent
            # Check sibling directories
            for subdir in ["", "bin", "ffmpeg", "lib"]:
                candidate = manim_dir / subdir / exe if subdir else manim_dir / exe
                if candidate.is_file():
                    return str(candidate)
    except Exception:
        pass

    return None


def _is_valid_binary(path) -> bool:
    """Check that a path points to an existing executable file."""
    try:
        p = Path(path)
        return p.is_file() and os.access(str(p), os.X_OK)
    except Exception:
        return False
