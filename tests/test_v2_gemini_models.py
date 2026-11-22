"""One Gemini key: Argon for every text stage, Gemini TTS for narration."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.agents.tts_agent import TTSOutput
from src.cli import pdf2manim
from src.evaluation.runner import MODEL_ALIASES
from src.pipeline_v2 import directives, generation, model_config, planning, revision
from src.pipeline_v2.models import NarratedScene, NarrationBeat, RunResult, RunStatus
from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.providers import PRICES, price
from src.pipeline_v2.state import RunStore
from src.pipeline_v2.tts import (
    SCENE_END_PAUSE_SECONDS,
    SPEECH_RATE,
    NarrationAudioService,
    _silence,
)


FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"
PIPELINE_V2 = Path(__file__).resolve().parents[1] / "src" / "pipeline_v2"
ARGON = "gemini-4-argon"


def _scene() -> NarratedScene:
    return NarratedScene(
        scene_id="scene_001",
        narration="A queue holds the frontier. Each step removes one node.",
        beats=[
            NarrationBeat(beat_id="b1", text="A queue holds the frontier", source_fact_ids=["f"]),
            NarrationBeat(beat_id="b2", text="Each step removes one node", source_fact_ids=["f"]),
        ],
    )


def _service(tmp_path: Path, provider: str, voice: str, speech_rate: float):
    store = RunStore(tmp_path / "run.db", 10.0)
    artifacts = RunArtifacts(tmp_path / "run", "run")
    service = NarrationAudioService(
        store, artifacts, voice=voice, high_quality=False, strict_keys=False,
        api_limit=asyncio.Semaphore(1), speech_rate=speech_rate,
        provider=provider, audience="undergraduate",
    )
    return store, artifacts, service


def _events(artifacts: RunArtifacts, name: str) -> list[dict]:
    lines = artifacts.events_path.read_text(encoding="utf-8").splitlines()
    return [record for record in map(json.loads, lines) if record["event"] == name]


def test_stage_models_are_argon():
    assert model_config.TEXT_MODEL == model_config.OCR_MODEL == model_config.CODE_MODEL == ARGON
    # Lesson, narration and visual contracts.
    assert planning.PLANNING_PROVIDER == "google"
    assert planning.PLANNING_MODEL == ARGON
    # Low-confidence page transcription.
    assert planning.OCR_MODEL == ARGON
    # Code generation, professor revision and screenshot-guided repair.
    assert generation.CODE_MODEL == ARGON
    assert revision._CODE_MODEL == ("google", ARGON)
    # Brief interpretation.
    assert (directives.BRIEF_PROVIDER, directives.BRIEF_MODEL) == ("google", ARGON)
    # Evaluation keeps both aliases; "gemini" follows the code model.
    assert MODEL_ALIASES["gemini"].model == ARGON
    assert MODEL_ALIASES["gemma"].model == "gemma-4-26b-a4b-it"


def test_no_openai_model_in_v2_stages():
    allowed = {"tts.py", "providers.py"}
    files = sorted(PIPELINE_V2.rglob("*.py"))
    assert len(files) > 10
    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if path.name not in allowed and ('"gpt-5.4' in text or 'provider="openai"' in text):
            offenders.append(path.name)
    assert offenders == []


def test_prices_cover_v2_models():
    models = {
        value for name, value in vars(model_config).items()
        if name.endswith("_MODEL") and isinstance(value, str)
    }
    assert models == {ARGON, "gemini-3.8-flash-tts", "gemini-omni-1.2-flash"}
    for model in models:
        # price() falls back to a default rate for unknown models, so check
        # the table itself as well as the computed price.
        assert ("google", model) in PRICES
        assert price("google", model, 1_000_000, 1_000_000) > 0


def test_gemini_tts_keyless_fallback_names_gemini_key(tmp_path):
    store, artifacts, service = _service(tmp_path, "gemini", "Sulafat", SPEECH_RATE)

    result = asyncio.run(service.generate(_scene(), tmp_path / "scene"))

    assert result.audio_is_placeholder
    assert Path(result.audio_path).is_file()
    fallbacks = _events(artifacts, "tts_fallback")
    assert len(fallbacks) == 1
    assert "GEMINI_API_KEY" in fallbacks[0]["message"]
    assert store.cost_rows() == []
    store.close()


def test_openai_tts_path_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    inputs = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def execute(self, tts_input):
            inputs.append(tts_input)
            path = Path(tts_input.output_dir) / f"{tts_input.scene_id}.mp3"
            _silence(path, 1.0)
            return TTSOutput(audio_file_path=str(path), duration_seconds=1.0, scene_id=tts_input.scene_id)

    def no_gemini(**_kwargs):
        raise AssertionError("the openai provider must not call Gemini")

    monkeypatch.setattr("src.pipeline_v2.tts.TTSAgent", FakeAgent)
    monkeypatch.setattr("google.genai.Client", no_gemini)
    store, artifacts, service = _service(tmp_path, "openai", "alloy", SPEECH_RATE)

    result = asyncio.run(service.generate(_scene(), tmp_path / "scene"))

    assert [item.voice for item in inputs] == ["alloy"]
    rows = store.cost_rows()
    assert [(row["provider"], row["model"], row["status"]) for row in rows] == [
        ("openai", "tts-1", "SUCCEEDED"),
    ]
    assert _events(artifacts, "tts_paced")[0]["speech_rate"] == SPEECH_RATE
    assert result.audio_duration_seconds == pytest.approx(
        1.0 / SPEECH_RATE + SCENE_END_PAUSE_SECONDS, abs=0.15,
    )
    store.close()


def test_cli_voice_and_tts_provider_defaults(tmp_path):
    def convert(*args: str):
        pipeline = MagicMock()
        pipeline.run.return_value = RunResult(
            run_id="test", run_directory=str(tmp_path), status=RunStatus.SUCCESS,
        )
        with patch("src.cli.PipelineV2", return_value=pipeline) as factory:
            result = CliRunner().invoke(pdf2manim, ["convert", str(FIXTURE), *args])
        assert result.exit_code == 0, result.output
        return factory.call_args.args[0]

    default = convert()
    assert default.voice == "Sulafat"
    assert default.tts_provider == "gemini"
    assert default.max_cost_usd == 4.0

    openai = convert("--tts-provider", "openai")
    assert (openai.tts_provider, openai.voice) == ("openai", "alloy")
    chosen = convert("--tts-provider", "openai", "--voice", "nova")
    assert chosen.voice == "nova"


def test_openai_voice_name_maps_to_gemini_default_for_gemini_tts(tmp_path):
    """A run recorded with an OpenAI voice and resumed under Gemini TTS uses the default Gemini voice."""
    import asyncio
    from src.pipeline_v2.prompt_text import DEFAULT_VOICE
    from src.pipeline_v2.tts import NarrationAudioService

    service = NarrationAudioService(
        None, None, voice="alloy", high_quality=False, strict_keys=False,
        api_limit=asyncio.Semaphore(1), provider="gemini",
    )
    assert service.voice == DEFAULT_VOICE
    openai = NarrationAudioService(
        None, None, voice="alloy", high_quality=False, strict_keys=False,
        api_limit=asyncio.Semaphore(1), provider="openai",
    )
    assert openai.voice == "alloy"

