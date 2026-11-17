"""Gemini narration is recorded beat by beat: only the script is spoken, and
every beat's window is cut from its measured speech."""

import asyncio
import json
import wave
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline_v2 import model_config, tts
from src.pipeline_v2.models import NarratedScene, NarrationBeat, NarrationPlan
from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.planning import apply_audio_timing
from src.pipeline_v2.prompt_text import tts_style
from src.pipeline_v2.revision import untimed
from src.pipeline_v2.state import RunStore
from src.pipeline_v2.tts import (
    BEAT_GAP_SECONDS,
    LEAD_IN_SECONDS,
    SCENE_END_PAUSE_SECONDS,
    TRIM_MARGIN_SECONDS,
    NarrationAudioService,
    gemini_input_units,
    plausible_take,
    retime,
    trim_silence,
)

RATE = 24000
FIRST = "A queue holds the frontier."
SECOND = "Each step removes one node."


def _speech(seconds: float, lead: float = 0.3, tail: float = 0.3) -> bytes:
    """A tone of ``seconds`` between silences: stands in for one spoken beat."""

    t = np.arange(int(seconds * RATE)) / RATE
    tone = (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    return np.concatenate([
        np.zeros(int(lead * RATE), dtype=np.int16), tone,
        np.zeros(int(tail * RATE), dtype=np.int16),
    ]).tobytes()


def _scene() -> NarratedScene:
    return NarratedScene(
        scene_id="scene_001",
        narration=f"{FIRST} {SECOND}",
        beats=[
            NarrationBeat(beat_id="b1", text="A queue holds the frontier", source_fact_ids=["f"]),
            NarrationBeat(beat_id="b2", text="Each step removes one node", source_fact_ids=["f"]),
        ],
    )


def _service(tmp_path: Path):
    store = RunStore(tmp_path / "run.db", 10.0)
    artifacts = RunArtifacts(tmp_path / "run", "run")
    service = NarrationAudioService(
        store, artifacts, voice="Sulafat", high_quality=False, strict_keys=False,
        api_limit=asyncio.Semaphore(1), provider="gemini", audience="undergraduate",
    )
    return store, artifacts, service


def _events(artifacts: RunArtifacts, name: str) -> list[dict]:
    lines = artifacts.events_path.read_text(encoding="utf-8").splitlines()
    return [record for record in map(json.loads, lines) if record["event"] == name]


def _no_plain_tts(**_kwargs):
    raise AssertionError("the styled request succeeded; the plain fallback must not run")


def test_only_the_script_is_spoken_and_each_beat_is_measured(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    calls = []

    def fake_interactions(text, voice, style):
        calls.append(SimpleNamespace(text=text, voice=voice, style=style))
        return _speech(1.5), RATE

    monkeypatch.setattr(tts, "_interactions_tts", fake_interactions)
    monkeypatch.setattr("google.genai.Client", _no_plain_tts)
    store, artifacts, service = _service(tmp_path)
    scene_dir = tmp_path / "scene"

    result = asyncio.run(service.generate(_scene(), scene_dir))

    # One request per beat, carrying exactly the words to be spoken; the
    # delivery direction travels separately as style metadata.
    assert sorted(call.text for call in calls) == sorted([FIRST, SECOND])
    assert {call.style for call in calls} == {tts_style("undergraduate")}
    assert {call.voice for call in calls} == {"Sulafat"}

    first, second = result.beat_speech_spans
    speech = 1.5 + 2 * TRIM_MARGIN_SECONDS
    assert first[0] == pytest.approx(LEAD_IN_SECONDS, abs=0.011)
    assert first[1] - first[0] == pytest.approx(speech, abs=0.021)
    assert second[0] - first[1] == pytest.approx(BEAT_GAP_SECONDS, abs=0.011)
    assert second[1] - second[0] == pytest.approx(speech, abs=0.021)

    b1, b2 = result.beats
    assert (b1.start_seconds, b1.end_seconds) == (0.0, round(second[0], 3))
    assert b2.start_seconds == round(second[0], 3)
    assert b2.end_seconds == round(result.audio_duration_seconds, 3)
    assert result.audio_duration_seconds == pytest.approx(
        second[1] + SCENE_END_PAUSE_SECONDS, abs=0.15,
    )
    assert result.audio_tail_seconds == SCENE_END_PAUSE_SECONDS

    with wave.open(str(scene_dir / "audio" / "scene_001_raw.wav"), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (RATE, 1, 2)
        assert handle.getnframes() / RATE == pytest.approx(second[1], abs=0.01)
    assert _events(artifacts, "tts_paced")[0]["speech_rate"] == 1.0
    assert _events(artifacts, "tts_beat_retake") == []

    (row,) = store.cost_rows()
    assert (row["provider"], row["model"], row["status"]) == ("google", model_config.TTS_MODEL, "SUCCEEDED")
    # Both takes are billed as generated, silence included: 2 x 2.1 s.
    assert row["output_units"] == round(4.2 * model_config.GEMINI_TTS_TOKENS_PER_SECOND)
    assert row["input_units"] == gemini_input_units([FIRST, SECOND], "undergraduate")
    store.close()


def test_a_take_with_extra_speech_is_recorded_again(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    counts: Counter = Counter()

    def fake_interactions(text, voice, style):
        counts[text] += 1
        # The first take of beat one runs 12 s for five words: the model
        # spoke something that is not in the script.
        if text == FIRST and counts[text] == 1:
            return _speech(12.0), RATE
        return _speech(1.5), RATE

    monkeypatch.setattr(tts, "_interactions_tts", fake_interactions)
    monkeypatch.setattr("google.genai.Client", _no_plain_tts)
    store, artifacts, service = _service(tmp_path)

    result = asyncio.run(service.generate(_scene(), tmp_path / "scene"))

    assert counts == {FIRST: 2, SECOND: 1}
    (retake,) = _events(artifacts, "tts_beat_retake")
    assert (retake["beat_id"], retake["takes"], retake["plausible"]) == ("b1", 2, True)
    first, _ = result.beat_speech_spans
    assert first[1] - first[0] == pytest.approx(1.5 + 2 * TRIM_MARGIN_SECONDS, abs=0.021)

    (row,) = store.cost_rows()
    # The rejected 12.6 s take is billed too.
    assert row["output_units"] == round((12.6 + 2.1 + 2.1) * model_config.GEMINI_TTS_TOKENS_PER_SECOND)
    assert row["input_units"] == (
        2 * gemini_input_units([FIRST], "undergraduate")
        + gemini_input_units([SECOND], "undergraduate")
    )
    store.close()


def test_without_style_metadata_the_plain_request_still_sends_only_the_words(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    contents = []

    def styled_unavailable(text, voice, style):
        raise RuntimeError("400 speech_metadata not supported")

    class FakeModels:
        def generate_content(self, **kwargs):
            contents.append(kwargs["contents"])
            part = SimpleNamespace(inline_data=SimpleNamespace(data=_speech(1.5)))
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setattr(tts, "_interactions_tts", styled_unavailable)
    monkeypatch.setattr("google.genai.Client", FakeClient)
    store, artifacts, service = _service(tmp_path)

    result = asyncio.run(service.generate(_scene(), tmp_path / "scene"))

    assert sorted(contents) == sorted([FIRST, SECOND])
    assert len(_events(artifacts, "tts_style_fallback")) == 2
    assert len(result.beat_speech_spans) == 2
    store.close()


def test_take_plausibility_bounds():
    assert plausible_take(10, 10 / 2.9)
    assert not plausible_take(10, 10 / 1.9 + 1.6)  # extra speech
    assert not plausible_take(10, 10 / 5.5 - 0.1)  # cut short
    assert plausible_take(1, 0.4)


def test_trim_silence_keeps_a_margin_around_speech():
    trimmed = trim_silence(_speech(1.0, lead=0.5, tail=0.7), RATE)
    assert len(trimmed) / (2 * RATE) == pytest.approx(1.0 + 2 * TRIM_MARGIN_SECONDS, abs=0.011)
    silent = np.zeros(RATE, dtype=np.int16).tobytes()
    assert trim_silence(silent, RATE) == silent


def _three_beats(spans=()) -> NarratedScene:
    return NarratedScene(
        scene_id="s", narration="one two three",
        beats=[
            NarrationBeat(beat_id="a", text="one two three four", source_fact_ids=["f"]),
            NarrationBeat(beat_id="b", text="five six", source_fact_ids=["f"]),
            NarrationBeat(beat_id="c", text="seven", source_fact_ids=["f"]),
        ],
        beat_speech_spans=[list(span) for span in spans],
    )


def test_measured_spans_cut_the_beat_windows():
    timed = apply_audio_timing(_three_beats([(0.15, 2.0), (2.5, 5.0), (5.6, 7.0)]), 9.0)
    assert [(b.start_seconds, b.end_seconds) for b in timed.beats] == [
        (0.0, 2.5), (2.5, 5.6), (5.6, 9.0),
    ]
    assert timed.audio_duration_seconds == 9.0


def test_without_spans_windows_are_shared_by_word_count():
    timed = apply_audio_timing(_three_beats(), 7.0)
    lengths = [b.end_seconds - b.start_seconds for b in timed.beats]
    assert timed.beats[0].start_seconds == 0.0 and timed.beats[-1].end_seconds == 7.0
    assert lengths[0] > lengths[1] > lengths[2]


def test_retime_keeps_spans_only_for_unchanged_words():
    spans = [(0.15, 2.0), (2.5, 5.0), (5.6, 7.0)]
    previous = apply_audio_timing(_three_beats(spans), 9.0).model_copy(
        update={"audio_path": "x.mp3", "audio_tail_seconds": 2.0},
    )
    paused = _three_beats()
    paused.beats[0].pause_after_seconds = 1.2
    kept = retime(paused, previous)
    assert kept.beat_speech_spans == [list(span) for span in spans]
    assert kept.beats[1].start_seconds == 2.5

    reworded = _three_beats()
    reworded.beats[1].text = "five six and more"
    assert retime(reworded, previous).beat_speech_spans == []


def test_spans_stay_out_of_the_planner_schema_and_are_cleared_for_new_audio():
    assert "beat_speech_spans" not in json.dumps(NarrationPlan.model_json_schema())
    scene = _three_beats([(0.1, 1.0), (1.5, 2.0), (2.5, 3.0)])
    restored = NarratedScene.model_validate(scene.model_dump(mode="json"))
    assert restored.beat_speech_spans == scene.beat_speech_spans
    assert untimed(scene).beat_speech_spans == []


def test_a_rate_limited_styled_request_is_retried_before_falling_back(monkeypatch, tmp_path):
    import httpx

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(tts, "STYLE_RETRY_SECONDS", 0.0)
    counts: Counter = Counter()

    def flaky(text, voice, style):
        counts[text] += 1
        if counts[text] == 1:
            request = httpx.Request("POST", tts.GEMINI_INTERACTIONS_URL)
            raise httpx.HTTPStatusError(
                "429", request=request, response=httpx.Response(429, request=request),
            )
        return _speech(1.5), RATE

    monkeypatch.setattr(tts, "_interactions_tts", flaky)
    monkeypatch.setattr("google.genai.Client", _no_plain_tts)
    store, artifacts, service = _service(tmp_path)

    result = asyncio.run(service.generate(_scene(), tmp_path / "scene"))

    assert counts == {FIRST: 2, SECOND: 2}
    assert _events(artifacts, "tts_style_fallback") == []
    assert len(result.beat_speech_spans) == 2
    store.close()
