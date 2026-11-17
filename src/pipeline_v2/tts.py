"""Budgeted narration generation with an offline silence fallback."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import subprocess
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from src.agents.tts_agent import TTSAgent, TTSError, TTSInput, TTSOutput
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe
from src.utils.fingerprints import apply_lexicon, lexicon_for

from .model_config import GEMINI_TTS_TOKENS_PER_SECOND, TTS_MODEL
from .models import NarratedScene
from .observability import RunArtifacts
from .planning import apply_audio_timing
from .prompt_text import DEFAULT_VOICE, tts_style
from .providers import estimate_tokens, price
from .state import RunStore


# OpenAI voice names; a run recorded before Gemini TTS became the default
# keeps one of these in run.json, which Gemini cannot speak.
OPENAI_VOICES = frozenset({"alloy", "echo", "fable", "onyx", "nova", "shimmer"})

SPEECH_RATE = 0.94
RELAXED_SPEECH_RATE = 0.84
SCENE_END_PAUSE_SECONDS = 2.0
TTS_TIMING_VERSION = "tts-v2-6-beats"
# Gemini TTS returns raw 24 kHz, 16-bit, mono PCM: 48000 bytes per second.
GEMINI_SAMPLE_RATE = 24000
GEMINI_PCM_BYTES_PER_SECOND = GEMINI_SAMPLE_RATE * 2
# The Interactions API is the one Gemini TTS surface that takes delivery
# style as speech_metadata instead of as text the model would read aloud.
GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
# Gemini narrates near 3 words per second. A take far slower than this has
# spoken words that are not in the script; one far faster was cut short.
GEMINI_WORDS_PER_SECOND = 2.9
MIN_WORDS_PER_SECOND = 1.9
MAX_WORDS_PER_SECOND = 5.5
EXTRA_SECONDS_ALLOWED = 1.5
MAX_BEAT_TAKES = 3
# Gemini TTS allows few requests per minute per model (10 on Tier 1;
# GenerateRequestsPerMinutePerProjectPerModel). Every TTS request in the
# process waits for a slot; set GEMINI_TTS_RPM to the project's quota.
TTS_REQUESTS_PER_MINUTE = max(1, int(os.getenv("GEMINI_TTS_RPM", "10")))
# One request is tried this many times: a rate limit waits the delay the
# API asks for, a server error or dropped connection a short backoff.
TTS_REQUEST_ATTEMPTS = 4
RETRY_SECONDS = 2.0
DEFAULT_QUOTA_WAIT_SECONDS = 30.0
MAX_QUOTA_WAIT_SECONDS = 65.0
# Silence placed before the first beat and between beats (at least the
# beat's own pause_after_seconds), after each take's silence is trimmed.
LEAD_IN_SECONDS = 0.15
BEAT_GAP_SECONDS = 0.45
TRIM_MARGIN_SECONDS = 0.08
TTS_KEY_NAMES = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


def tts_key_name(provider: str) -> str:
    """The environment key the configured narration provider needs."""

    return TTS_KEY_NAMES[provider]


class NarrationAudioService:
    def __init__(
        self,
        store: RunStore,
        artifacts: RunArtifacts,
        *,
        voice: str,
        high_quality: bool,
        strict_keys: bool,
        api_limit: asyncio.Semaphore,
        speech_rate: float = SPEECH_RATE,
        provider: str = "gemini",
        audience: Optional[str] = None,
    ):
        self.store = store
        self.artifacts = artifacts
        # Gemini cannot speak an OpenAI voice (older runs resumed under Gemini TTS).
        self.voice = (
            DEFAULT_VOICE if provider == "gemini" and voice in OPENAI_VOICES else voice
        )
        self.high_quality = high_quality
        self.strict_keys = strict_keys
        self.api_limit = api_limit
        self.speech_rate = speech_rate
        self.provider = provider
        self.audience = audience

    def _rate(self, requested: float) -> float:
        """Gemini is directed to teaching pace in its prompt, so it is never time-stretched."""

        return 1.0 if self.provider == "gemini" else requested

    async def generate(
        self,
        scene: NarratedScene,
        scene_dir: Path,
        *,
        lexicon: Optional[Mapping[str, str]] = None,
        speech_rate: Optional[float] = None,
    ) -> NarratedScene:
        audio_dir = scene_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        narration_text = _narration_for_tts(scene, lexicon)
        speech_rate = self._rate(self.speech_rate if speech_rate is None else speech_rate)
        key_name = tts_key_name(self.provider)
        if not os.getenv(key_name):
            if self.strict_keys:
                raise RuntimeError(f"{key_name} is required for narration audio")
            speech_seconds = len(narration_text.split()) * 60.0 / scene.target_words_per_minute
            duration = max(2.0, speech_seconds + sum(
                beat.pause_after_seconds for beat in scene.beats
            ) + SCENE_END_PAUSE_SECONDS)
            path = audio_dir / f"{scene.scene_id}.mp3"
            await asyncio.to_thread(_silence, path, duration)
            self.artifacts.event(
                "tts_fallback", stage="tts", scene_id=scene.scene_id,
                task_id=f"tts-{scene.scene_id}",
                message=f"{key_name} missing; generated local silent timing audio.",
                duration_seconds=duration,
            )
            self.artifacts.record_provider_call(
                stage="tts", call_id=f"tts-local-{scene.scene_id}", scene_id=scene.scene_id,
                request={
                    "provider": "google" if self.provider == "gemini" else "openai",
                    "voice": self.voice,
                    "narration": narration_text,
                    "target_words_per_minute": scene.target_words_per_minute,
                },
                response={
                    "provider": "local", "model": "silent-timing-audio",
                    "audio_file_path": str(path.resolve()), "duration_seconds": duration,
                },
            )
            timed = apply_audio_timing(scene, duration)
            return timed.model_copy(update={
                "audio_path": str(path.resolve()),
                "audio_tail_seconds": SCENE_END_PAUSE_SECONDS,
                "audio_is_placeholder": True,
            })

        if self.provider == "gemini":
            original_path, source_duration, spans = await self._record_gemini(
                scene, audio_dir, lexicon,
            )
            scene = scene.model_copy(update={"beat_speech_spans": spans})
        else:
            original_path, source_duration = await self._record_openai(
                scene, audio_dir, narration_text, speech_rate,
            )
        paced_path = audio_dir / f"{scene.scene_id}_paced.mp3"
        try:
            await asyncio.to_thread(
                _pace_audio,
                original_path,
                paced_path,
                speech_rate,
                SCENE_END_PAUSE_SECONDS,
            )
            final_path = paced_path
            final_duration = await asyncio.to_thread(_probe_duration, paced_path)
            self.artifacts.event(
                "tts_paced", stage="tts", scene_id=scene.scene_id,
                task_id=f"tts-{scene.scene_id}",
                message="Applied calm educational pacing and a scene-boundary pause.",
                source_duration_seconds=source_duration,
                final_duration_seconds=round(final_duration, 3),
                speech_rate=speech_rate,
                trailing_pause_seconds=SCENE_END_PAUSE_SECONDS,
            )
        except Exception as exc:
            final_path = original_path
            final_duration = source_duration
            self.artifacts.event(
                "tts_pacing_fallback", stage="tts", scene_id=scene.scene_id,
                task_id=f"tts-{scene.scene_id}", message=str(exc),
            )
        timed = apply_audio_timing(scene, final_duration)
        return timed.model_copy(update={
            "audio_path": str(final_path.resolve()),
            "audio_tail_seconds": (
                SCENE_END_PAUSE_SECONDS if final_path == paced_path else 0.0
            ),
        })

    async def _record_gemini(
        self,
        scene: NarratedScene,
        audio_dir: Path,
        lexicon: Optional[Mapping[str, str]] = None,
    ) -> tuple[Path, float, list[list[float]]]:
        """Record a scene beat by beat with Gemini TTS.

        Each beat is its own request, so its speech is measured rather than
        estimated: the scene's beat windows are cut from these spans, which
        keeps every visual change on the words it illustrates. Returns the
        un-paced MP3, its seconds and each beat's [start, end] in it.
        """

        texts = [_beat_for_tts(beat.text, lexicon) for beat in scene.beats]
        style = tts_style(self.audience)
        spoken = "\n\n".join(texts)
        input_units = gemini_input_units(texts, self.audience)
        estimated_seconds = (
            len(spoken.split()) * 60 / scene.target_words_per_minute + 5
        )
        amount = price(
            "google", TTS_MODEL, input_units,
            estimated_seconds * GEMINI_TTS_TOKENS_PER_SECOND,
        )
        characters = len(spoken)
        call_id = f"tts-{uuid.uuid4().hex[:12]}"
        self.store.reserve_call(
            call_id, task_id=f"tts-{scene.scene_id}", scene_id=scene.scene_id,
            stage="tts", provider="google", model=TTS_MODEL, amount_usd=amount,
        )
        self.artifacts.event(
            "api_call_started", stage="tts", scene_id=scene.scene_id,
            task_id=f"tts-{scene.scene_id}", provider="google", model=TTS_MODEL,
            call_id=call_id, reserved_usd=amount, characters=characters,
        )
        request = {
            "provider": "google", "model": TTS_MODEL, "voice": self.voice,
            "narration": spoken, "characters": characters, "style": style,
            "beats": len(texts), "speech_rate": 1.0,
        }
        self.artifacts.record_provider_call(
            stage="tts", call_id=call_id, scene_id=scene.scene_id, request=request,
        )
        raw_wav = audio_dir / f"{scene.scene_id}_raw.wav"
        original_path = audio_dir / f"{scene.scene_id}.mp3"
        recorded: list[BeatTake] = []
        try:
            recorded = list(await asyncio.gather(
                *(self._gemini_beat(text, style) for text in texts)
            ))
            pcm, rate, spans = assemble_beats(
                [take.pcm for take in recorded], recorded[0].rate,
                [beat.pause_after_seconds for beat in scene.beats],
            )
            await asyncio.to_thread(_write_wav, raw_wav, pcm, rate)
            await asyncio.to_thread(_wav_to_mp3, raw_wav, original_path)
        except Exception:
            self.store.settle_call(
                call_id, actual_usd=amount, input_units=input_units,
                output_units=0, exact=False, status="FAILED_UNCERTAIN",
            )
            raise
        seconds = len(pcm) / (2 * rate)
        # Every take is billed, including the ones a retake replaced.
        output_units = int(round(
            sum(take.generated_seconds for take in recorded) * GEMINI_TTS_TOKENS_PER_SECOND
        ))
        input_units = sum(
            gemini_input_units([text], self.audience) * take.takes
            for text, take in zip(texts, recorded)
        )
        exact_cost = price("google", TTS_MODEL, input_units, output_units)
        self.store.settle_call(
            call_id, actual_usd=exact_cost, input_units=input_units,
            output_units=output_units, exact=True,
        )
        for beat, take in zip(scene.beats, recorded):
            if take.takes > 1:
                self.artifacts.event(
                    "tts_beat_retake", stage="tts", scene_id=scene.scene_id,
                    task_id=f"tts-{scene.scene_id}", beat_id=beat.beat_id,
                    takes=take.takes, plausible=take.plausible,
                    speech_seconds=round(take.speech_seconds, 3),
                    words=len(beat.text.split()),
                    message=(
                        "A take's length did not fit its words (extra or missing "
                        "speech), so the beat was recorded again."
                    ),
                )
            if take.style_fallback:
                self.artifacts.event(
                    "tts_style_fallback", stage="tts", scene_id=scene.scene_id,
                    task_id=f"tts-{scene.scene_id}", beat_id=beat.beat_id,
                    message=f"Recorded without style metadata: {take.style_fallback}",
                )
        self.artifacts.event(
            "api_call_finished", stage="tts", scene_id=scene.scene_id,
            task_id=f"tts-{scene.scene_id}", provider="google", model=TTS_MODEL,
            call_id=call_id, actual_usd=exact_cost,
            duration_seconds=seconds, characters=characters,
        )
        self.artifacts.record_provider_call(
            stage="tts", call_id=call_id, scene_id=scene.scene_id, request=request,
            response={
                "audio_file_path": str(original_path.resolve()),
                "duration_seconds": seconds,
                "beat_speech_spans": spans,
                "cost_usd": exact_cost,
            },
        )
        return original_path, seconds, spans

    async def _gemini_beat(self, text: str, style: str) -> "BeatTake":
        """Record one beat, retaking it when its length does not fit its words."""

        words = len(text.split())
        best: Optional[BeatTake] = None
        generated = 0.0
        takes = 0
        for takes in range(1, MAX_BEAT_TAKES + 1):
            pcm, rate, fallback = await self._gemini_clip(text, style)
            generated += len(pcm) / (2 * rate)
            trimmed = trim_silence(pcm, rate)
            seconds = len(trimmed) / (2 * rate)
            candidate = BeatTake(
                pcm=trimmed, rate=rate, speech_seconds=seconds, takes=takes,
                plausible=plausible_take(words, seconds), generated_seconds=0.0,
                style_fallback=fallback,
            )
            if best is None or _take_error(words, candidate) < _take_error(words, best):
                best = candidate
            if candidate.plausible:
                break
        assert best is not None
        best.takes = takes
        best.generated_seconds = generated
        return best

    async def _gemini_clip(self, text: str, style: str) -> tuple[bytes, int, str]:
        """One Gemini TTS recording of ``text`` alone; returns PCM, its rate and any fallback reason.

        Only the words to be spoken are sent as text. Delivery goes in the
        speech_metadata style annotation; if that request still fails after
        its retries, the beat is recorded from the plain text with no
        direction at all.
        """

        try:
            pcm, rate = await self._tts_request(
                lambda: _interactions_tts(text, self.voice, style),
            )
            return pcm, rate, ""
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        pcm, rate = await self._tts_request(lambda: _generate_content_tts(text, self.voice))
        return pcm, rate, reason

    async def _tts_request(self, call):
        """Run one TTS request inside the per-minute quota, retrying what is worth retrying.

        The quota slot is taken before the shared API slot, so a beat waiting
        for quota never holds a slot another stage could use.
        """

        for attempt in range(1, TTS_REQUEST_ATTEMPTS + 1):
            await TTS_PACER.acquire()
            try:
                async with self.api_limit:
                    return await asyncio.to_thread(call)
            except Exception as exc:
                if attempt == TTS_REQUEST_ATTEMPTS:
                    raise
                delay = _rate_limit_delay(exc)
                if delay is not None:
                    TTS_PACER.pause(delay)  # every waiting beat backs off together
                elif _transient(exc):
                    delay = RETRY_SECONDS * attempt
                else:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _record_openai(
        self,
        scene: NarratedScene,
        audio_dir: Path,
        narration_text: str,
        speech_rate: float,
    ) -> tuple[Path, float]:
        """Record one scene with OpenAI TTS; returns the un-paced MP3 and its seconds."""

        model = "tts-1-hd" if self.high_quality else "tts-1"
        characters = len(narration_text)
        amount = price("openai", model, characters, 0)
        call_id = f"tts-{uuid.uuid4().hex[:12]}"
        self.store.reserve_call(
            call_id, task_id=f"tts-{scene.scene_id}", scene_id=scene.scene_id,
            stage="tts", provider="openai", model=model, amount_usd=amount,
        )
        self.artifacts.event(
            "api_call_started", stage="tts", scene_id=scene.scene_id,
            task_id=f"tts-{scene.scene_id}", provider="openai", model=model,
            call_id=call_id, reserved_usd=amount, characters=characters,
        )
        self.artifacts.record_provider_call(
            stage="tts", call_id=call_id, scene_id=scene.scene_id,
            request={
                "provider": "openai", "model": model, "voice": self.voice,
                "narration": narration_text, "characters": characters,
                "speech_rate": speech_rate,
            },
        )
        agent = TTSAgent(max_retries=1, strict=True)
        try:
            async with self.api_limit:
                result = await asyncio.to_thread(agent.execute, TTSInput(
                    narration=narration_text, scene_id=scene.scene_id,
                    output_dir=str(audio_dir), quality="high" if self.high_quality else "standard",
                    voice=self.voice,
                ))
        except Exception:
            self.store.settle_call(
                call_id, actual_usd=amount, input_units=characters,
                output_units=0, exact=False, status="FAILED_UNCERTAIN",
            )
            raise
        if isinstance(result, TTSError):
            self.store.settle_call(
                call_id, actual_usd=amount, input_units=characters,
                output_units=0, exact=False, status="FAILED_UNCERTAIN",
            )
            raise RuntimeError(result.error)
        if not isinstance(result, TTSOutput):
            self.store.settle_call(
                call_id, actual_usd=amount, input_units=characters,
                output_units=0, exact=False, status="FAILED_UNCERTAIN",
            )
            raise RuntimeError("TTS provider returned an unsupported result")
        exact_cost = price("openai", model, characters, 0)
        self.store.settle_call(
            call_id, actual_usd=exact_cost, input_units=characters,
            output_units=0, exact=True,
        )
        self.artifacts.event(
            "api_call_finished", stage="tts", scene_id=scene.scene_id,
            task_id=f"tts-{scene.scene_id}", provider="openai", model=model,
            call_id=call_id, actual_usd=exact_cost,
            duration_seconds=result.duration_seconds, characters=characters,
        )
        self.artifacts.record_provider_call(
            stage="tts", call_id=call_id, scene_id=scene.scene_id,
            request={
                "provider": "openai", "model": model, "voice": self.voice,
                "narration": narration_text, "characters": characters,
                "speech_rate": speech_rate,
            },
            response={
                "audio_file_path": result.audio_file_path,
                "duration_seconds": result.duration_seconds,
                "cost_usd": exact_cost,
            },
        )
        return Path(result.audio_file_path), result.duration_seconds

    async def repace(
        self,
        scene: NarratedScene,
        raw_audio: Path,
        scene_dir: Path,
        speech_rate: float,
    ) -> NarratedScene:
        """Re-pace an existing recording locally; never calls the provider."""

        speech_rate = self._rate(speech_rate)
        audio_dir = scene_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        paced_path = audio_dir / f"{scene.scene_id}_paced.mp3"
        await asyncio.to_thread(
            _pace_audio, raw_audio, paced_path, speech_rate, SCENE_END_PAUSE_SECONDS,
        )
        duration = await asyncio.to_thread(_probe_duration, paced_path)
        self.artifacts.event(
            "tts_repaced", stage="tts", scene_id=scene.scene_id,
            task_id=f"tts-{scene.scene_id}",
            message="Re-paced the existing recording locally without a paid TTS call.",
            final_duration_seconds=round(duration, 3), speech_rate=speech_rate,
        )
        spans = [
            [round(start / speech_rate, 3), round(end / speech_rate, 3)]
            for start, end in scene.beat_speech_spans
        ]
        scene = scene.model_copy(update={"beat_speech_spans": spans})
        return apply_audio_timing(scene, duration).model_copy(update={
            "audio_path": str(paced_path.resolve()),
            "audio_tail_seconds": SCENE_END_PAUSE_SECONDS,
        })


def retime(scene: NarratedScene, previous: NarratedScene) -> NarratedScene:
    """Redistribute beat timings over an unchanged recording (pause edits).

    The recording is unchanged, so its measured beat spans still hold when
    the beats' words are the same.
    """

    if not previous.audio_duration_seconds:
        raise RuntimeError(f"{scene.scene_id} has no measured audio to re-time")
    same_words = [beat.text for beat in scene.beats] == [beat.text for beat in previous.beats]
    scene = scene.model_copy(update={
        "beat_speech_spans": previous.beat_speech_spans if same_words else [],
    })
    return apply_audio_timing(scene, previous.audio_duration_seconds).model_copy(update={
        "audio_path": previous.audio_path,
        "audio_tail_seconds": previous.audio_tail_seconds,
    })


@dataclass
class BeatTake:
    """The kept take of one beat: trimmed PCM and how it was obtained."""

    pcm: bytes
    rate: int
    speech_seconds: float
    takes: int
    plausible: bool
    generated_seconds: float
    style_fallback: str = ""


def gemini_input_units(texts: Sequence[str], audience: Optional[str] = None) -> int:
    """Input tokens of one Gemini TTS request per beat: its text plus the style."""

    style = tts_style(audience)
    return sum(estimate_tokens(f"{style}\n\n{text}") for text in texts)


def plausible_take(words: int, seconds: float) -> bool:
    """Whether a take's speech length fits its word count.

    Too long means the model spoke words that are not in the script (the
    failure that once read a delivery direction aloud); too short means it
    stopped early.
    """

    words = max(1, words)
    return words / MAX_WORDS_PER_SECOND <= seconds <= (
        words / MIN_WORDS_PER_SECOND + EXTRA_SECONDS_ALLOWED
    )


def _take_error(words: int, take: BeatTake) -> float:
    return abs(take.speech_seconds - max(1, words) / GEMINI_WORDS_PER_SECOND)


def trim_silence(pcm: bytes, rate: int, margin: float = TRIM_MARGIN_SECONDS) -> bytes:
    """Cut leading and trailing silence from 16-bit mono PCM, keeping a short margin."""

    samples = np.frombuffer(pcm, dtype=np.int16)
    window = max(1, rate // 100)  # 10 ms
    count = samples.size // window
    if count == 0:
        return pcm
    levels = np.abs(samples[: count * window].astype(np.int32)).reshape(count, window).max(axis=1)
    threshold = max(300, int(levels.max() * 0.04))
    loud = np.flatnonzero(levels > threshold)
    if loud.size == 0:
        return pcm
    keep = int(margin * rate)
    start = max(0, int(loud[0]) * window - keep)
    end = min(samples.size, (int(loud[-1]) + 1) * window + keep)
    return samples[start:end].tobytes()


def assemble_beats(
    clips: Sequence[bytes],
    rate: int,
    pauses: Sequence[float],
) -> tuple[bytes, int, list[list[float]]]:
    """Join beat clips with a lead-in and inter-beat gaps; return PCM, rate and each beat's span."""

    def silence(seconds: float) -> bytes:
        return b"\x00\x00" * int(round(seconds * rate))

    parts = [silence(LEAD_IN_SECONDS)]
    cursor = len(parts[0])
    spans: list[list[float]] = []
    for index, clip in enumerate(clips):
        start = cursor
        parts.append(clip)
        cursor += len(clip)
        spans.append([round(start / (2 * rate), 3), round(cursor / (2 * rate), 3)])
        if index < len(clips) - 1:
            gap = silence(max(BEAT_GAP_SECONDS, pauses[index]))
            parts.append(gap)
            cursor += len(gap)
    return b"".join(parts), rate, spans


def _decode_audio(raw: bytes) -> tuple[bytes, int]:
    """16-bit mono PCM and its sample rate from a WAV file or bare 24 kHz PCM."""

    if raw[:4] != b"RIFF":
        return raw, GEMINI_SAMPLE_RATE
    with wave.open(io.BytesIO(raw), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise RuntimeError(
                f"Gemini TTS returned {handle.getnchannels()}-channel, "
                f"{8 * handle.getsampwidth()}-bit audio; expected 16-bit mono"
            )
        return handle.readframes(handle.getnframes()), handle.getframerate()


class RequestPacer:
    """At most ``limit`` request starts in any ``window`` seconds, for one event loop.

    Single-threaded by construction: callers await ``acquire`` on the loop,
    so checking and recording a slot cannot interleave.
    """

    def __init__(self, limit: int, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._starts: deque[float] = deque()
        self._paused_until = 0.0

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            while self._starts and now - self._starts[0] >= self.window:
                self._starts.popleft()
            wait = self._paused_until - now
            if wait <= 0 and len(self._starts) < self.limit:
                self._starts.append(now)
                return
            if wait <= 0:
                wait = self.window - (now - self._starts[0]) + 0.05
            await asyncio.sleep(wait)

    def pause(self, seconds: float) -> None:
        """Hold every new request for ``seconds`` (the API asked us to wait)."""

        self._paused_until = max(self._paused_until, time.monotonic() + seconds)


TTS_PACER = RequestPacer(TTS_REQUESTS_PER_MINUTE)


def _rate_limit_delay(exc: Exception) -> Optional[float]:
    """Seconds to wait when ``exc`` is a rate limit (429), else ``None``.

    Uses the delay the API asks for ("Please retry in 22.2s" or RetryInfo
    "retryDelay": "22s") plus a second, or a default when none is given.
    """

    import httpx

    status = getattr(exc, "code", None)
    text = str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        try:
            text += " " + exc.response.text
        except Exception:
            pass
    if status != 429 and "RESOURCE_EXHAUSTED" not in text:
        return None
    match = (
        re.search(r"retry in ([0-9.]+)\s*s", text)
        or re.search(r"retryDelay\W+([0-9.]+)s", text)
    )
    if not match:
        return DEFAULT_QUOTA_WAIT_SECONDS
    return min(MAX_QUOTA_WAIT_SECONDS, float(match.group(1)) + 1.0)


def _transient(exc: Exception) -> bool:
    """A server error or dropped connection: worth another try after a short backoff."""

    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code >= 500:
        return True
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


def _interactions_tts(text: str, voice: str, style: str) -> tuple[bytes, int]:
    """Speak ``text`` verbatim, with delivery as a speech_metadata style annotation."""

    import httpx

    response = httpx.post(
        GEMINI_INTERACTIONS_URL,
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "model": TTS_MODEL,
            "input": [{
                "type": "text", "text": text,
                "annotations": [{"type": "speech_metadata", "style": style}],
            }],
            "generation_config": {"speech_config": [{"voice": voice, "language": "en-US"}]},
            "response_format": {"type": "audio"},
        },
        timeout=180,
    )
    response.raise_for_status()
    for step in response.json().get("steps") or []:
        for part in step.get("content") or []:
            if part.get("type") == "audio" and part.get("data"):
                return _decode_audio(base64.b64decode(part["data"]))
    raise RuntimeError("Gemini TTS returned no audio")


def _generate_content_tts(text: str, voice: str) -> tuple[bytes, int]:
    """Speak ``text`` verbatim through generate_content (no delivery style)."""

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        ),
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data
    if not pcm:
        raise RuntimeError("Gemini TTS returned no audio")
    return _decode_audio(pcm)


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def _silence(path: Path, duration: float) -> None:
    command = [
        get_ffmpeg(), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{duration:.3f}", "-q:a", "9", "-acodec", "libmp3lame", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Could not create fallback timing audio: {result.stderr[-800:]}")


def _wav_to_mp3(source: Path, output: Path) -> None:
    command = [
        get_ffmpeg(), "-y", "-i", str(source),
        "-codec:a", "libmp3lame", "-q:a", "2", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Could not encode narration audio: {result.stderr[-800:]}")


def _narration_for_tts(
    scene: NarratedScene,
    lexicon: Optional[Mapping[str, str]] = None,
) -> str:
    """Produce TTS-friendly paragraphs without provider-specific pause markup.

    ``lexicon`` respells words for the voice only; on-screen text and the
    storyboard keep the original spelling.
    """

    return "\n\n".join(_beat_for_tts(beat.text, lexicon) for beat in scene.beats)


def _beat_for_tts(text: str, lexicon: Optional[Mapping[str, str]] = None) -> str:
    """One beat as spoken: closing punctuation added, respelled for the voice only."""

    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return apply_lexicon(text, lexicon_for(text, dict(lexicon))) if lexicon else text


def _pace_audio(
    source: Path,
    output: Path,
    speech_rate: float = SPEECH_RATE,
    trailing_pause: float = SCENE_END_PAUSE_SECONDS,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        get_ffmpeg(), "-y", "-i", str(source),
        "-filter:a", f"atempo={speech_rate:.4f},apad=pad_dur={trailing_pause:.3f}",
        "-codec:a", "libmp3lame", "-q:a", "2", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Could not pace narration audio: {result.stderr[-800:]}")


def _probe_duration(path: Path) -> float:
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
        raise RuntimeError(f"Could not measure paced narration: {result.stderr[-800:]}") from exc
    if result.returncode != 0 or duration <= 0:
        raise RuntimeError(f"Could not measure paced narration: {result.stderr[-800:]}")
    return duration


def ensure_trailing_buffer(
    scene: NarratedScene,
    output: Path,
    minimum_seconds: float = SCENE_END_PAUSE_SECONDS,
) -> NarratedScene:
    """Pad legacy narration locally so retries get the current minimum tail."""

    missing = max(0.0, minimum_seconds - scene.audio_tail_seconds)
    if missing <= 0.001 or not scene.audio_path:
        return scene
    source = Path(scene.audio_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        get_ffmpeg(), "-y", "-i", str(source),
        "-filter:a", f"apad=pad_dur={missing:.3f}",
        "-codec:a", "libmp3lame", "-q:a", "2", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Could not extend narration buffer: {result.stderr[-800:]}")
    duration = _probe_duration(output)
    timed = apply_audio_timing(scene, duration)
    return timed.model_copy(update={
        "audio_path": str(output.resolve()),
        "audio_tail_seconds": minimum_seconds,
    })
