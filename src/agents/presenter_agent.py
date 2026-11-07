"""Presenter bridges: the professor's on-camera moments between Manim scenes.

Pipeline (Part E)::

    bridge narration audio --> AvatarProvider(audio, avatar) --> raw clip
    raw clip --> layout (full | pip) + disclosure --> presenter.mp4
                 (Manim resolution / fps / pix_fmt) --> composer, in scene order

Providers:
- ``none`` (default): no avatar. The bridge narration plays over a title
  card, so the feature degrades gracefully and tests stay hermetic.
- ``heygen``: a hosted photoreal talking head of the professor. Live-API
  only; it needs the professor's consent record and HEYGEN_API_KEY.

Cards are drawn with Pillow and composed with ffmpeg (no drawtext/font
config dependency). Every clip made with an avatar or a cloned voice
carries an on-screen "AI-generated presenter" disclosure, and the video
ends with a credits card.
"""

import json
import logging
import subprocess
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.ffmpeg_finder import get_ffmpeg

logger = logging.getLogger(__name__)

# Manim output size per render quality, so presenter clips concatenate cleanly.
RESOLUTIONS = {"low": (854, 480, 15), "medium": (1280, 720, 30), "high": (1920, 1080, 60)}
BRAND_BACKGROUND = (28, 38, 52)
TEXT_COLOR = (236, 240, 245)
DISCLOSURE = "AI-generated presenter"


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ffmpeg(args: List[str], timeout: int = 300) -> None:
    result = subprocess.run([get_ffmpeg(), "-v", "error", "-y", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-400:]}")


class AvatarProvider:
    name = "none"

    def render(self, audio_path: str, text: str, out_path: str, width: int, height: int) -> Optional[str]:
        """A talking-head clip lip-synced to ``audio_path``, or None for no avatar."""
        return None


class NoAvatarProvider(AvatarProvider):
    name = "none"


class HeyGenAvatarProvider(AvatarProvider):
    """Hosted photoreal avatar (HeyGen API: upload audio, generate, poll).

    Untested against the live service in this repository's hermetic suite;
    exercised only under the ``live_api`` marker.
    """

    name = "heygen"
    UPLOAD = "https://upload.heygen.com/v1/asset"
    GENERATE = "https://api.heygen.com/v2/video/generate"
    STATUS = "https://api.heygen.com/v1/video_status.get?video_id={video_id}"

    def __init__(self, api_key: str, avatar_id: str, poll_seconds: float = 10.0, timeout_seconds: float = 900.0):
        if not api_key:
            raise ValueError("HEYGEN_API_KEY is required for the hosted avatar.")
        self._api_key = api_key
        self.avatar_id = avatar_id
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def _request(self, url: str, data: Optional[bytes] = None, content_type: str = "application/json", method: str = "GET"):
        request = urllib.request.Request(
            url, data=data, method=method,
            headers={"X-Api-Key": self._api_key, "Content-Type": content_type, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def render(self, audio_path, text, out_path, width, height):
        asset = self._request(self.UPLOAD, Path(audio_path).read_bytes(), "audio/mpeg", "POST")
        asset_id = (asset.get("data") or {}).get("id")
        if not asset_id:
            raise RuntimeError(f"HeyGen audio upload failed: {asset}")
        body = {
            "video_inputs": [{
                "character": {"type": "avatar", "avatar_id": self.avatar_id, "avatar_style": "normal"},
                "voice": {"type": "audio", "audio_asset_id": asset_id},
            }],
            "dimension": {"width": width, "height": height},
        }
        job = self._request(self.GENERATE, json.dumps(body).encode("utf-8"), method="POST")
        video_id = (job.get("data") or {}).get("video_id")
        if not video_id:
            raise RuntimeError(f"HeyGen generation was not accepted: {job}")
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            status = (self._request(self.STATUS.format(video_id=video_id)).get("data") or {})
            if status.get("status") == "completed" and status.get("video_url"):
                with urllib.request.urlopen(status["video_url"], timeout=300) as response:
                    Path(out_path).write_bytes(response.read())
                return out_path
            if status.get("status") == "failed":
                raise RuntimeError(f"HeyGen render failed: {status.get('error')}")
            time.sleep(self.poll_seconds)
        raise TimeoutError("HeyGen render did not finish in time")


class PresenterAgent:
    def __init__(
        self,
        provider: Optional[AvatarProvider] = None,
        render_quality: str = "medium",
        layout: str = "full",
        disclose: Optional[bool] = None,
    ):
        self.provider = provider or NoAvatarProvider()
        self.width, self.height, self.fps = RESOLUTIONS.get(render_quality, RESOLUTIONS["medium"])
        self.layout = layout
        # Disclosure is required whenever a synthetic likeness is on screen.
        self.disclose = (self.provider.name != "none") if disclose is None else disclose

    # ── cards ───────────────────────────────────────────────────────────

    def _card(self, text: str, path: Path, background=None, box: bool = False) -> Path:
        from PIL import Image, ImageDraw

        image = background.copy() if background is not None else Image.new(
            "RGB", (self.width, self.height), BRAND_BACKGROUND,
        )
        draw = ImageDraw.Draw(image, "RGBA")
        size = max(18, self.height // 18)
        font = _font(size)
        lines = textwrap.wrap(text, width=max(20, self.width // max(1, size // 2 + 4)))[:8]
        line_height = int(size * 1.35)
        block = line_height * len(lines)
        top = (self.height - block) // 2 if not box else self.height - block - self.height // 10
        if box:
            draw.rectangle([self.width // 12, top - 20, self.width * 11 // 12, top + block + 20], fill=(0, 0, 0, 170))
        for i, line in enumerate(lines):
            width = draw.textlength(line, font=font)
            draw.text(((self.width - width) / 2, top + i * line_height), line, font=font, fill=TEXT_COLOR)
        image.save(path)
        return path

    def _disclosure_overlay(self, path: Path) -> Path:
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        size = max(12, self.height // 40)
        font = _font(size)
        pad = self.height // 40
        width = draw.textlength(DISCLOSURE, font=font)
        draw.rectangle([pad, pad, pad * 3 + width, pad * 3 + size], fill=(0, 0, 0, 150))
        draw.text((pad * 2, pad * 2), DISCLOSURE, font=font, fill=(255, 255, 255, 230))
        image.save(path)
        return path

    def _last_frame(self, video: str, path: Path):
        from PIL import Image, ImageEnhance

        _ffmpeg(["-sseof", "-0.2", "-i", video, "-frames:v", "1", "-vf", f"scale={self.width}:{self.height}", str(path)])
        with Image.open(path) as frame:
            return ImageEnhance.Brightness(frame.convert("RGB")).enhance(0.35)

    # ── rendering ───────────────────────────────────────────────────────

    def render(
        self,
        scene_id: str,
        text: str,
        audio_path: str,
        duration: float,
        out_dir: str,
        previous_video: Optional[str] = None,
    ) -> str:
        """Render one bridge as a silent clip exactly ``duration`` seconds long."""
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        output = out_dir_path / f"{scene_id}_presenter.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = None
            if self.provider.name != "none":
                raw = self.provider.render(audio_path, text, str(tmp_path / "raw.mp4"), self.width, self.height)
            background = None
            if self.layout == "pip" and previous_video and Path(previous_video).is_file():
                background = self._last_frame(previous_video, tmp_path / "last.png")
            card = self._card(
                text if raw is None else "", tmp_path / "card.png",
                background=background, box=background is not None and raw is None,
            )
            inputs = ["-loop", "1", "-framerate", str(self.fps), "-i", str(card)]
            filters = []
            last = "[0:v]"
            if raw is not None:
                inputs += ["-i", raw]
                if background is not None:  # picture-in-picture, bottom right
                    w = self.width // 3
                    filters.append(f"[1:v]scale={w}:-2[pip]")
                    filters.append(f"{last}[pip]overlay=W-w-{self.width // 30}:H-h-{self.height // 20}[v1]")
                else:                        # full frame, letterboxed on the brand background
                    filters.append(
                        f"[1:v]scale={self.width}:{self.height}:force_original_aspect_ratio=decrease[av]"
                    )
                    filters.append(f"{last}[av]overlay=(W-w)/2:(H-h)/2[v1]")
                last = "[v1]"
            if self.disclose:
                inputs += ["-loop", "1", "-i", str(self._disclosure_overlay(tmp_path / "disclosure.png"))]
                index = 2 if raw is not None else 1
                filters.append(f"{last}[{index}:v]overlay=0:0[v2]")
                last = "[v2]"
            filters.append(f"{last}fps={self.fps},format=yuv420p[out]")
            _ffmpeg([
                *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                "-t", f"{max(duration, 0.5):.3f}", "-c:v", "libx264", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-an", str(output),
            ])
        return str(output)

    def credits(self, lines: List[str], out_dir: str, seconds: float = 4.0) -> str:
        """Closing card naming the synthetic presenter/voice and its consent."""
        output = Path(out_dir) / "credits.mp4"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            card = self._card("   ".join(lines), Path(tmp) / "credits.png")
            _ffmpeg([
                "-loop", "1", "-framerate", str(self.fps), "-i", str(card),
                "-t", f"{seconds:.2f}", "-vf", "format=yuv420p", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-an", str(output),
            ])
        return str(output)


def presenter_settings(provider_name: str, avatar_id: str, layout: str, voice: str) -> dict:
    """What a presenter clip depends on besides its text (fingerprinting)."""
    return {"avatar": provider_name, "avatar_id": avatar_id, "layout": layout, "voice": voice}
