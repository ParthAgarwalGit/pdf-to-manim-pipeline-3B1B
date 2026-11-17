"""Budgeted narration generation with an offline silence fallback."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Mapping, Optional

from src.agents.tts_agent import TTSAgent, TTSError, TTSInput, TTSOutput
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe
from src.utils.fingerprints import apply_lexicon, lexicon_for

from .model_config import GEMINI_TTS_TOKENS_PER_SECOND, TTS_MODEL
from .models import NarratedScene
from .observability import RunArtifacts
from .planning import apply_audio_timing
from .prompt_text import DEFAULT_VOICE, tts_direction
from .providers import estimate_tokens, price
from .state import RunStore


# OpenAI voice names; a run recorded before Gemini TTS became the default
# keeps one of these in run.json, which Gemini cannot speak.
OPENAI_VOICES = frozenset({"alloy", "echo", "fable", "onyx", "nova", "shimmer"})

SPEECH_RATE = 0.94
RELAXED_SPEECH_RATE = 0.84
SCENE_END_PAUSE_SECONDS = 2.0
TTS_TIMING_VERSION = "tts-v2-5-gemini"
# Gemini TTS returns raw 24 kHz, 16-bit, mono PCM: 48000 bytes per second.
GEMINI_SAMPLE_RATE = 24000
GEMINI_PCM_BYTES_PER_SECOND = GEMINI_SAMPLE_RATE * 2
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
            original_path, source_duration = await self._record_gemini(
                scene, audio_dir, narration_text, speech_rate,
            )
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
        narration_text: str,
        speech_rate: float,
    ) -> tuple[Path, float]:
        """Record one scene with Gemini TTS; returns the un-paced MP3 and its seconds."""

        prompt_text = self._gemini_contents(narration_text)
        input_units = estimate_tokens(prompt_text)
        estimated_seconds = (
            len(narration_text.split()) * 60 / scene.target_words_per_minute + 5
        )
        amount = price(
            "google", TTS_MODEL, input_units,
            estimated_seconds * GEMINI_TTS_TOKENS_PER_SECOND,
        )
        characters = len(narration_text)
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
            "narration": narration_text, "characters": characters,
            "speech_rate": speech_rate,
        }
        self.artifacts.record_provider_call(
            stage="tts", call_id=call_id, scene_id=scene.scene_id, request=request,
        )
        raw_wav = audio_dir / f"{scene.scene_id}_raw.wav"
        original_path = audio_dir / f"{scene.scene_id}.mp3"
        try:
            async with self.api_limit:
                seconds = await asyncio.to_thread(self._gemini_tts, narration_text, raw_wav)
            await asyncio.to_thread(_wav_to_mp3, raw_wav, original_path)
        except Exception:
            self.store.settle_call(
                call_id, actual_usd=amount, input_units=input_units,
                output_units=0, exact=False, status="FAILED_UNCERTAIN",
            )
            raise
        output_units = int(round(seconds * GEMINI_TTS_TOKENS_PER_SECOND))
        exact_cost = price("google", TTS_MODEL, input_units, output_units)
        self.store.settle_call(
            call_id, actual_usd=exact_cost, input_units=input_units,
            output_units=output_units, exact=True,
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
                "cost_usd": exact_cost,
            },
        )
        return original_path, seconds

    def _gemini_contents(self, narration_text: str) -> str:
        return f"{tts_direction(self.audience)}\n\n{narration_text}"

    def _gemini_tts(self, narration_text: str, out_wav: Path) -> float:
        """One Gemini TTS request; writes 24 kHz mono WAV and returns its seconds."""

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=self._gemini_contents(narration_text),
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))),
            ),
        )
        pcm = response.candidates[0].content.parts[0].inline_data.data
        if not pcm:
            raise RuntimeError("Gemini TTS returned no audio")
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(GEMINI_SAMPLE_RATE)
            handle.writeframes(pcm)
        return len(pcm) / GEMINI_PCM_BYTES_PER_SECOND

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
        return apply_audio_timing(scene, duration).model_copy(update={
            "audio_path": str(paced_path.resolve()),
            "audio_tail_seconds": SCENE_END_PAUSE_SECONDS,
        })


def retime(scene: NarratedScene, previous: NarratedScene) -> NarratedScene:
    """Redistribute beat timings over an unchanged recording (pause edits)."""

    if not previous.audio_duration_seconds:
        raise RuntimeError(f"{scene.scene_id} has no measured audio to re-time")
    return apply_audio_timing(scene, previous.audio_duration_seconds).model_copy(update={
        "audio_path": previous.audio_path,
        "audio_tail_seconds": previous.audio_tail_seconds,
    })


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

    paragraphs = []
    for beat in scene.beats:
        text = beat.text.strip()
        if text and text[-1] not in ".!?":
            text += "."
        paragraphs.append(text)
    text = "\n\n".join(paragraphs)
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
