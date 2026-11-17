"""Gemini Omni presenter: a fictional lecturer who speaks the narration.

Hybrid backend engine ``--presenter-engine gemini``. Gemini Omni 1.2 Flash
(``client.interactions.create``) generates a talking-presenter clip from a
prompt; it cannot take our narration audio, so the presenter speaks the
lines in its own voice and **that speech becomes the scene's narration**
(the pipeline swaps the clip's audio in for the TTS track, so lips and
voice match by construction).

- Clips are at most 15 s; longer narration is split into balanced,
  sentence-aligned chunks and joined. Each clip is trimmed to its speech
  (Omni pads clips with silence), so joins carry no dead air.
- Raw clips are cached by what they say (``cache_dir``), so a revision
  re-assembles unchanged lines without paying for them again.
- The first clip's frame becomes the reference image for every later clip,
  so the same fictional lecturer appears throughout the video.
- Each clip is transcribed (Whisper) and compared with the text it had to
  say; a clip that drifts is regenerated once and the closer take is kept.
- The persona is fictional and described in words: no real person's photo
  is used. Clips carry the "AI-generated presenter" overlay downstream.
"""

import base64
import logging
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.ffmpeg_finder import get_ffmpeg

logger = logging.getLogger(__name__)

OMNI_MODEL = "gemini-omni-1.2-flash"
MIN_CLIP_SECONDS, MAX_CLIP_SECONDS = 3, 15
CHUNK_SECONDS = 13.0          # target speech per clip, leaving room to breathe
# Measured on real Omni clips: 2.8-3.2 words/s. Planning slower padded every
# clip with seconds of dead air (now also trimmed, see trim_to_speech).
SPOKEN_WORDS_PER_SECOND = 2.9
MAX_WORD_ERROR = 0.15
REQUEST_TIMEOUT_MS = 240_000  # a 15 s clip takes ~60 s to generate
# Silence around the speech in a generated clip is cut down to a breath.
SILENCE_DB = -35
LEAD_SECONDS, TAIL_SECONDS = 0.15, 0.3
DEFAULT_PERSONA = (
    "a friendly fictional university lecturer, a woman in her thirties with short dark hair "
    "and glasses, wearing a navy blazer over a plain gray t-shirt"
)
VOICE = "a warm, clear, natural female voice with a neutral accent and an unhurried, conversational pace"
# Near-black so cuts to and from Manim scenes (black background) are not jarring.
BACKGROUND = "a plain, near-black studio background (hex #0B0B0B), evenly lit, with no visible props"

Transcriber = Callable[[str], Optional[str]]


def split_for_clips(text: str, seconds_per_clip: float = CHUNK_SECONDS) -> List[str]:
    """Balanced chunks that each fit one clip, cut at sentence ends.

    The fewest clips that fit, with about the same number of words each
    (a 6-word clip followed by a 33-word one sounds choppy). A sentence
    longer than a chunk is cut at its commas.
    """
    text = (text or "").strip()
    budget = seconds_per_clip * SPOKEN_WORDS_PER_SECOND
    total = len(text.split())
    if total <= budget:
        return [text]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    for count in range(math.ceil(total / budget), math.ceil(total / budget) + 3):
        target = total / count
        units: List[str] = []
        for sentence in sentences:
            if len(sentence.split()) > target:
                units += [c.strip() for c in re.split(r"(?<=,)\s+", sentence) if c.strip()]
            else:
                units.append(sentence)
        chunks: List[str] = []
        current: List[str] = []
        for unit in units:
            size = len(" ".join(current).split())
            if current and len(chunks) < count - 1 and size + len(unit.split()) / 2 > target:
                chunks.append(" ".join(current))
                current = []
            current.append(unit)
        if current:
            chunks.append(" ".join(current))
        if max(len(c.split()) for c in chunks) <= budget:
            return chunks
    return chunks


def clip_seconds(text: str, target: Optional[float] = None) -> int:
    """Requested clip length: room for the words (and the TTS timing, if any)."""
    needed = len(text.split()) / SPOKEN_WORDS_PER_SECOND
    if target:
        needed = max(needed, float(target))
    return int(max(MIN_CLIP_SECONDS, min(MAX_CLIP_SECONDS, math.ceil(needed + 1.0))))


def _ffmpeg(args: List[str]) -> None:
    result = subprocess.run(
        [get_ffmpeg(), "-v", "error", "-y", *args],
        capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-300:]}")


def speech_bounds(path: Path) -> Optional[Tuple[float, float, float]]:
    """(speech start, speech end, clip length) from ffmpeg silencedetect, or None."""
    from src.utils.media_probe import probe

    info = probe(str(path))
    duration = float(info.duration) if info and info.duration else 0.0
    if duration <= 0:
        return None
    result = subprocess.run(
        [get_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(path),
         "-af", f"silencedetect=noise={SILENCE_DB}dB:d=0.3", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
    )
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", result.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
    start, end = 0.0, duration
    if starts and starts[0] <= 0.05 and ends:
        start = ends[0]  # leading silence
    if starts and starts[-1] > start:
        last_end = ends[len(starts) - 1] if len(ends) >= len(starts) else duration
        if last_end >= duration - 0.05:
            end = starts[-1]  # trailing silence
    if end - start < 0.5:
        return None  # no speech found: leave the clip alone
    return start, end, duration


def trim_to_speech(path: Path) -> None:
    """Cut a clip's leading and trailing silence down to a short breath.

    Omni pads its clips (it speaks faster than the requested length), and
    joined clips would otherwise carry seconds of dead air mid-sentence.
    """
    bounds = speech_bounds(path)
    if bounds is None:
        return
    speech_start, speech_end, duration = bounds
    start = max(0.0, speech_start - LEAD_SECONDS)
    end = min(duration, speech_end + TAIL_SECONDS)
    if start < 0.1 and duration - end < 0.1:
        return
    trimmed = path.with_name(f"{path.stem}_trimmed{path.suffix}")
    _ffmpeg([
        "-i", str(path), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        str(trimmed),
    ])
    trimmed.replace(path)


def extract_audio(video: str, dest: Path) -> Path:
    """The clip's speech as a standalone track (AAC, stream copy)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(["-i", video, "-vn", "-c:a", "copy", str(dest)])
    return dest


def default_transcriber() -> Optional[Transcriber]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    import openai

    client = openai.OpenAI(max_retries=3, timeout=120.0)

    def transcribe(path: str) -> Optional[str]:
        with open(path, "rb") as f:
            return client.audio.transcriptions.create(model="whisper-1", file=f).text

    return transcribe


class GeminiOmniPresenter:
    """Hybrid presenter engine backed by Gemini Omni (speaks its own lines)."""

    name = "gemini-omni"
    speaks = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Any = None,
        persona: str = DEFAULT_PERSONA,
        transcriber: Optional[Transcriber] = None,
        use_default_transcriber: bool = True,
        reference_image: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        voice: Optional[str] = None,
    ):
        if client is None:
            key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
            if not key:
                raise ValueError("GEMINI_API_KEY is required for the Gemini presenter.")
            import google.genai as genai

            from google.genai import types

            # Without a timeout one stalled request hung a whole render for hours.
            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
        self.client = client
        self.persona = persona or DEFAULT_PERSONA
        # A prebuilt Gemini voice (e.g. "Sulafat"): the presenter then speaks in
        # the narration's voice. None keeps Omni's own voice (legacy behaviour).
        self.voice = voice or None
        self.transcriber = transcriber if transcriber is not None else (
            default_transcriber() if use_default_transcriber else None
        )
        self.reference_image = Path(reference_image) if reference_image else None
        # Raw clips by what they say: a revision never pays twice for a line.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.clips: List[Dict[str, Any]] = []

    def __repr__(self) -> str:  # never show the key
        return f"GeminiOmniPresenter(model={OMNI_MODEL!r})"

    # ── public ──────────────────────────────────────────────────────────

    def speak(self, text: str, out_path: str, target_seconds: Optional[float] = None) -> Tuple[str, float]:
        """A talking clip (video + its own speech) of ``text``; returns (path, seconds)."""
        from src.utils.media_probe import probe
        from src.utils.speech_text import speakable

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        chunks = split_for_clips(speakable(text))
        parts: List[Path] = []
        for i, chunk in enumerate(chunks):
            # Sized by the words alone: the presenter's own speech becomes the
            # narration and the clip is trimmed to it, so the TTS timing
            # (target_seconds) only padded clips and destabilized the cache.
            seconds = clip_seconds(chunk)
            part = out.with_name(f"{out.stem}_part{i}.mp4") if len(chunks) > 1 else out
            self._clip(chunk, seconds, part)
            parts.append(part)
        if len(parts) > 1:
            self._join(parts, out)
        info = probe(str(out))
        return str(out), float(info.duration if info and info.duration else 0.0)

    def render(self, audio_path: str, text: str, out_path: str, width: int, height: int) -> Optional[str]:
        """AvatarProvider interface (the audio is ignored: Omni speaks itself)."""
        return self.speak(text, out_path)[0]

    # ── internals ───────────────────────────────────────────────────────

    def _prompt(self, text: str, strict: bool = False) -> str:
        who = (
            "The same woman as in the reference image" if self.reference_image else f"{self.persona[0].upper()}{self.persona[1:]}"
        )
        prompt = (
            f"{who}, in front of {BACKGROUND}. Medium close-up, static camera; she faces the camera "
            "with small, natural hand gestures. She speaks directly to the camera in "
            f"{VOICE} and says exactly these words, nothing more and nothing less: \"{text}\" "
            "Her lips match her speech. No music, no other sounds, no on-screen text or captions."
        )
        if strict:
            prompt += " Say every word of the quoted text, in order, without adding or skipping anything."
        return prompt

    def _request(self, text: str, seconds: int, strict: bool) -> bytes:
        prompt = self._prompt(text, strict)
        payload: Any = prompt
        if self.reference_image and self.reference_image.is_file():
            payload = [
                {"type": "image", "data": base64.b64encode(self.reference_image.read_bytes()).decode("ascii"),
                 "mime_type": "image/png"},
                {"type": "text", "text": prompt},
            ]
        options: Dict[str, Any] = {}
        if self.voice:
            options["generation_config"] = {"speech_config": [{"voice": self.voice, "language": "en-US"}]}
        interaction = self.client.interactions.create(
            model=OMNI_MODEL,
            input=payload,
            response_format={
                "type": "video", "duration": f"{seconds}s", "aspect_ratio": "16:9", "resolution": "720p",
            },
            **options,
        )
        return base64.b64decode(interaction.output_video.data)

    def _word_error(self, text: str, clip: Path) -> Optional[float]:
        if self.transcriber is None:
            return None
        from src.agents.delivery_gate import word_error_rate

        try:
            heard = self.transcriber(str(clip))
        except Exception as exc:
            logger.warning("Presenter clip transcription failed: %s", exc)
            return None
        return word_error_rate(text, heard or "")

    def _cache_path(self, text: str, seconds: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        import hashlib

        key = hashlib.sha256(
            f"{OMNI_MODEL}|{self.persona}|{self.voice or ''}|{text}".encode("utf-8")
        ).hexdigest()[:24]
        return self.cache_dir / f"{key}.mp4"

    def _clip(self, text: str, seconds: int, dest: Path) -> Path:
        cached = self._cache_path(text, seconds)
        if cached is not None and cached.is_file():
            dest.write_bytes(cached.read_bytes())
            self.clips.append({"seconds": seconds, "word_error": None, "attempt": 0, "cached": True})
        else:
            self._generate(text, seconds, dest)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(dest.read_bytes())
        trim_to_speech(dest)
        if self.reference_image is None:
            self._capture_reference(dest)
        return dest

    def _generate(self, text: str, seconds: int, dest: Path) -> None:
        best: Optional[Tuple[float, bytes]] = None
        for attempt in range(2):
            try:
                data = self._request(text, seconds, strict=attempt > 0)
            except Exception as exc:
                # A failed or timed-out request never hangs the run: keep the
                # take we have, or try once more, then give up (the scene
                # falls back to a narrated title card upstream).
                if best is not None or attempt == 1:
                    logger.warning("Presenter clip request failed (%s)%s.", str(exc)[:160],
                                   "; keeping the earlier take" if best is not None else "")
                    break
                logger.warning("Presenter clip request failed (%s); retrying once.", str(exc)[:160])
                continue
            dest.write_bytes(data)
            error = self._word_error(text, dest)
            score = error if error is not None else 0.0
            if best is None or score < best[0]:
                best = (score, data)
            self.clips.append({"seconds": seconds, "word_error": error, "attempt": attempt + 1})
            if error is None or error <= MAX_WORD_ERROR:
                break
            logger.warning("Presenter clip said something else (WER %.2f); regenerating once.", error)
        if best is None:
            raise RuntimeError("Gemini Omni returned no presenter clip.")
        dest.write_bytes(best[1])

    def _capture_reference(self, clip: Path) -> None:
        ref = clip.parent / "presenter_reference.png"
        try:
            _ffmpeg(["-ss", "1.0", "-i", str(clip), "-frames:v", "1", str(ref)])
            self.reference_image = ref
        except Exception as exc:  # consistency is best effort
            logger.warning("Could not capture a presenter reference frame: %s", exc)

    def _join(self, parts: List[Path], out: Path) -> None:
        """Concatenate clips with audio (re-encoded to one format)."""
        inputs: List[str] = []
        streams = []
        for i, part in enumerate(parts):
            inputs += ["-i", str(part)]
            streams.append(f"[{i}:v:0][{i}:a:0]")
        _ffmpeg([
            *inputs, "-filter_complex", f"{''.join(streams)}concat=n={len(parts)}:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(out),
        ])
