"""Budgeted narration generation with an offline silence fallback."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from pathlib import Path

from src.agents.tts_agent import TTSAgent, TTSError, TTSInput, TTSOutput
from src.utils.ffmpeg_finder import get_ffmpeg

from .models import NarratedScene
from .observability import RunArtifacts
from .planning import apply_audio_timing
from .providers import price
from .state import RunStore


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
    ):
        self.store = store
        self.artifacts = artifacts
        self.voice = voice
        self.high_quality = high_quality
        self.strict_keys = strict_keys
        self.api_limit = api_limit

    async def generate(self, scene: NarratedScene, scene_dir: Path) -> NarratedScene:
        audio_dir = scene_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        if not os.getenv("OPENAI_API_KEY"):
            if self.strict_keys:
                raise RuntimeError("OPENAI_API_KEY is required for narration audio")
            duration = max(1.5, len(scene.narration.split()) / 2.8)
            path = audio_dir / f"{scene.scene_id}.mp3"
            await asyncio.to_thread(_silence, path, duration)
            self.artifacts.event(
                "tts_fallback", stage="tts", scene_id=scene.scene_id,
                task_id=f"tts-{scene.scene_id}",
                message="OPENAI_API_KEY missing; generated local silent timing audio.",
                duration_seconds=duration,
            )
            self.artifacts.record_provider_call(
                stage="tts", call_id=f"tts-local-{scene.scene_id}", scene_id=scene.scene_id,
                request={
                    "provider": "openai", "voice": self.voice,
                    "narration": scene.narration,
                },
                response={
                    "provider": "local", "model": "silent-timing-audio",
                    "audio_file_path": str(path.resolve()), "duration_seconds": duration,
                },
            )
            timed = apply_audio_timing(scene, duration)
            return timed.model_copy(update={"audio_path": str(path.resolve())})

        model = "tts-1-hd" if self.high_quality else "tts-1"
        characters = len(scene.narration)
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
                "narration": scene.narration, "characters": characters,
            },
        )
        agent = TTSAgent(max_retries=1, strict=True)
        try:
            async with self.api_limit:
                result = await asyncio.to_thread(agent.execute, TTSInput(
                    narration=scene.narration, scene_id=scene.scene_id,
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
                "narration": scene.narration, "characters": characters,
            },
            response={
                "audio_file_path": result.audio_file_path,
                "duration_seconds": result.duration_seconds,
                "cost_usd": exact_cost,
            },
        )
        timed = apply_audio_timing(scene, result.duration_seconds)
        return timed.model_copy(update={"audio_path": str(Path(result.audio_file_path).resolve())})


def _silence(path: Path, duration: float) -> None:
    command = [
        get_ffmpeg(), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{duration:.3f}", "-q:a", "9", "-acodec", "libmp3lame", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Could not create fallback timing audio: {result.stderr[-800:]}")
