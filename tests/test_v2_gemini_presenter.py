"""WPC: Gemini Omni presenter bridges orchestrated in V2, in the narration's voice.

Keyless and offline: every model call fails as if no key were set, TTS writes
a local tone, the Omni engine's ``speak`` writes a local talking clip, and
Manim rendering is replaced by a black clip. ffmpeg builds every file for real.
"""

import asyncio
import base64
import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.agents.gemini_presenter import GeminiOmniPresenter
from src.cli import pdf2manim
from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus, presenter
from src.pipeline_v2.model_config import PRESENTER_MODEL, TEXT_MODEL
from src.pipeline_v2.models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    NarrationPlan,
    RunResult,
)
from src.pipeline_v2.planning import PlanningService, apply_audio_timing
from src.pipeline_v2.presenter import (
    OMNI_USD_PER_SECOND,
    BridgeProposal,
    approve_bridges,
    GeminiPresenter,
    PresenterConfigError,
    PresenterRequest,
    PresenterSettings,
    build_presenter,
    load_bridges,
    omni_clip_seconds,
    omni_cost_usd,
    propose_bridges,
    save_bridges,
    write_bridges,
)
from src.pipeline_v2.prompt_text import DEFAULT_VOICE
from src.pipeline_v2.providers import MissingProviderKey, ProviderResponse, ProviderRouter
from src.pipeline_v2.rendering import RenderOutcome
from src.pipeline_v2.revision import parse_ops
from src.pipeline_v2.tts import NarrationAudioService
from src.utils.ffmpeg_finder import get_ffmpeg


FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"
HOOK = "Two small squares always fill one big square exactly. Why should that be true?"


# ── media helpers ───────────────────────────────────────────────────────

def _ffmpeg(*args: str) -> None:
    result = subprocess.run(
        [get_ffmpeg(), "-v", "error", "-y", *args], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr


def _tone(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds:.3f}",
            "-ac", "2", "-ar", "44100", "-c:a", "libmp3lame", str(path))


def _picture(path: Path, seconds: float) -> None:
    """A silent black render, like Manim's output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-f", "lavfi", "-i", f"color=c=black:s=1280x720:r=30:d={seconds:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))


def _talking_clip(path: Path, seconds: float) -> None:
    """What Omni returns: 720p video with its own speech."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-f", "lavfi", "-i", f"color=c=0x0B0B0B:s=1280x720:r=24:d={seconds:.3f}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path))


class FakeRenderer:
    """Stands in for TrackedManimRenderer: a black clip as long as the narration."""

    def __init__(self, *_args, **_kwargs):
        pass

    def render(self, code, contract, scene, out_dir, quality="low", timeout_seconds=None):
        path = Path(out_dir) / "render.mp4"
        seconds = scene.audio_duration_seconds or 2.0
        _picture(path, seconds)
        return RenderOutcome(
            safe=True, render_success=True, video_path=str(path), duration_seconds=seconds,
        )

    def terminate_all(self):
        return None


@pytest.fixture
def harness(monkeypatch):
    """A keyless pipeline with a Gemini key present, so the presenter can be built."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def withheld(provider):
        raise MissingProviderKey(f"{provider} key withheld by the test")

    # Every model call takes its keyless path: deterministic plan, templates.
    monkeypatch.setattr(ProviderRouter, "_require_key", staticmethod(withheld))
    fallback_lesson = PlanningService._fallback_lesson

    async def planned_lesson(self, source):
        return fallback_lesson(self, source).model_copy(update={
            "central_question": "Why do two small squares fill the big one?",
            "hook": HOOK,
            "takeaway": "Rearranging the triangles leaves the same empty area.",
        })

    monkeypatch.setattr(PlanningService, "lesson", planned_lesson)
    calls = SimpleNamespace(tts=[], speak=[])

    async def fake_tts(self, scene, scene_dir, *, lexicon=None, speech_rate=None):
        calls.tts.append(scene.scene_id)
        words = sum(len(beat.text.split()) for beat in scene.beats)
        seconds = max(4.0, round(words * 60.0 / scene.target_words_per_minute, 2))
        path = scene_dir / "audio" / f"{scene.scene_id}.mp3"
        await asyncio.to_thread(_tone, path, seconds)
        return apply_audio_timing(scene, seconds).model_copy(update={"audio_path": str(path.resolve())})

    def fake_speak(self, text, out_path, target_seconds=None):
        calls.speak.append({"text": text, "voice": self.voice})
        _talking_clip(Path(out_path), 4.0)
        return str(out_path), 4.0

    monkeypatch.setattr(NarrationAudioService, "generate", fake_tts)
    monkeypatch.setattr(GeminiOmniPresenter, "speak", fake_speak)
    monkeypatch.setattr("src.pipeline_v2.pipeline.TrackedManimRenderer", FakeRenderer)
    return calls


def _config(tmp_path, **changes) -> PipelineV2Config:
    values = dict(
        output_dir=tmp_path, max_scenes=2, repair="off", presenter="gemini",
        max_cost_usd=10.0, script_review="off",
    )
    values.update(changes)
    return PipelineV2Config(**values)


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def _named(run_dir: Path, name: str) -> list[dict]:
    return [event for event in _events(run_dir) if event["event"] == name]


def _storyboard(run_dir: Path) -> dict:
    return json.loads((run_dir / "storyboard.json").read_text(encoding="utf-8"))


def _timeline(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "timeline.json").read_text(encoding="utf-8"))


def _cost_rows(run_dir: Path, stage: str) -> list[dict]:
    connection = sqlite3.connect(str(run_dir / "run.db"))
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM api_calls WHERE stage=? ORDER BY created_at", (stage,),
        )]
    finally:
        connection.close()


def _task_ids(run_dir: Path) -> list[str]:
    connection = sqlite3.connect(str(run_dir / "run.db"))
    try:
        return [row[0] for row in connection.execute("SELECT task_id FROM tasks")]
    finally:
        connection.close()


# 1 ─────────────────────────────────────────────────────────────────────────


class FakeOmniClient:
    def __init__(self):
        self.calls = []
        self.interactions = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_video=SimpleNamespace(data=base64.b64encode(b"clip").decode()))


def test_omni_request_carries_voice(tmp_path):
    client = FakeOmniClient()
    voiced = GeminiOmniPresenter(client=client, use_default_transcriber=False, voice="Sulafat", cache_dir=tmp_path)
    assert voiced._request("Hello there.", 4, strict=False) == b"clip"
    config = client.calls[0]["generation_config"]
    assert config["speech_config"][0]["voice"] == "Sulafat"
    assert config["speech_config"][0]["language"] == "en-US"

    plain = GeminiOmniPresenter(client=client, use_default_transcriber=False, cache_dir=tmp_path)
    plain._request("Hello there.", 4, strict=False)
    assert "generation_config" not in client.calls[1]

    other = GeminiOmniPresenter(client=client, use_default_transcriber=False, voice="Kore", cache_dir=tmp_path)
    keys = {engine._cache_path("Hello there.", 4) for engine in (voiced, plain, other)}
    assert len(keys) == 3  # the cache key changes with the voice
    assert "near-black" in voiced._prompt("Hi.") and "#0B0B0B" in voiced._prompt("Hi.")


# 2 ─────────────────────────────────────────────────────────────────────────


def test_build_presenter_gemini_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = PresenterSettings(provider="gemini", cache_dir=tmp_path / "cache")
    with pytest.raises(PresenterConfigError, match="GEMINI_API_KEY is not set; the gemini presenter"):
        build_presenter(settings, environ={})
    with pytest.raises(PresenterConfigError, match="GEMINI_API_KEY"):
        build_presenter(settings)  # from os.environ

    provider = build_presenter(settings, environ={"GEMINI_API_KEY": "test-key"})
    assert isinstance(provider, GeminiPresenter)
    assert (provider.name, provider.synthetic, provider.speaks) == ("gemini", True, True)
    assert provider.voice == DEFAULT_VOICE and settings.profile_path is None  # no consent profile
    assert provider._engine is None  # checking the configuration builds no client
    assert "test-key" not in repr(provider)

    monkeypatch.setenv("GEMINI_API_KEY", "from-the-environment")
    assert isinstance(build_presenter(settings), GeminiPresenter)


# 3 ─────────────────────────────────────────────────────────────────────────


def test_cost_estimate_matches_clip_rule():
    text = " ".join(f"word{index}" for index in range(30)) + "."
    # 30 words planned at 2.5 words/s need 12 s, plus 1.5 s of headroom, rounded up: one 14 s clip.
    assert omni_clip_seconds(text) == 14
    expected = 14 * 5792 * 12.00 / 1_000_000 * 1.5
    assert expected == pytest.approx(1.459584)
    request = PresenterRequest(bridge_id="bridge_001", text=text, audio_path=None, duration_seconds=0.0)
    assert GeminiPresenter("key").estimated_cost_usd(request) == pytest.approx(expected)
    assert omni_cost_usd(text) == pytest.approx(expected)
    assert OMNI_USD_PER_SECOND == pytest.approx(5792 * 12.00 / 1_000_000)


# 4 & 5 ─────────────────────────────────────────────────────────────────────


def _spine_plan(count=4, seconds=60.0):
    scenes, narrated = [], []
    for index in range(1, count + 1):
        scene_id = f"scene_{index:03d}"
        scenes.append(LessonScene(
            scene_id=scene_id, title=f"Part {index}", description="d", learning_goal="g",
            source_fact_ids=["f1"], viewer_question=f"What happens to circle {index}?",
        ))
        narrated.append(NarratedScene(
            scene_id=scene_id,
            narration=f"Scene {index} narration.",
            beats=[
                NarrationBeat(beat_id=f"{scene_id}_b1", source_fact_ids=["f1"],
                              text=f"Circle number {index} starts out with a radius of {index} units on the board."),
                NarrationBeat(beat_id=f"{scene_id}_b2", source_fact_ids=["f1"],
                              text=f"So circle number {index} ends with an area of {index} squared times pi."),
            ],
            target_words_per_minute=150, audio_duration_seconds=seconds,
        ))
    lesson = LessonPlan(
        title="Circles", audience="undergraduate", objectives=["Compare areas"], scenes=scenes,
        central_question="Why does doubling the radius quadruple the area?",
        hook=HOOK, takeaway="Area grows with the square of the radius.",
    )
    return lesson, NarrationPlan(scenes=narrated)


def test_planned_proposals_use_hook_and_drop_recap():
    lesson, narration = _spine_plan(count=6)
    unplanned = propose_bridges(lesson, narration)
    assert unplanned[-1].kind == "recap"  # the template proposal would end on a recap

    planned = propose_bridges(lesson, narration, planned=True)
    assert planned[0].kind == "opening" and planned[0].text == lesson.hook
    assert "recap" not in [bridge.kind for bridge in planned]
    assert [b.kind for b in planned] == ["opening", "transition", "transition"]


class ScriptedRouter:
    def __init__(self, reply=None, error=None):
        self.calls = []
        self.events = []
        self.reply = reply
        self.error = error
        self.artifacts = SimpleNamespace(event=lambda name, **kwargs: self.events.append((name, kwargs)))

    async def text(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return ProviderResponse(
            text=json.dumps(self.reply), provider="google", model=TEXT_MODEL,
            input_units=1, output_units=1, exact_usage=True, raw={},
        )


def _proposals(lesson):
    kinds = ["opening"] + ["transition"] * (len(lesson.scenes) - 1)
    return [
        BridgeProposal(
            bridge_id=f"bridge_{index + 1:03d}", kind=kind, before_scene_id=scene.scene_id,
            text=HOOK if kind == "opening" else f"We move on to part {index + 1} of the lesson now, carefully.",
            estimated_seconds=5.0,
        )
        for index, (kind, scene) in enumerate(zip(kinds, lesson.scenes))
    ]


def test_write_bridges_validates_and_falls_back():
    lesson, narration = _spine_plan(count=4)
    proposals = _proposals(lesson)
    good = "So far the circle_two looks harmless, but something about its area is still unexplained."
    repeat = narration.scenes[2].beats[0].text  # what the viewer hears right after slot 2
    too_long = " ".join(["word"] * 60)
    router = ScriptedRouter(reply={"bridges": [
        {"slot": 1, "text": good}, {"slot": 2, "text": repeat}, {"slot": 3, "text": too_long},
    ]})

    written = asyncio.run(write_bridges(router, proposals, lesson, narration))

    assert len(router.calls) == 1
    call = router.calls[0]
    assert (call["stage"], call["model"], call["max_output_tokens"]) == ("bridges", TEXT_MODEL, 1500)
    assert "slot 0" not in call["prompt"]  # the opening already says the hook
    assert "slot 1 (transition)" in call["prompt"]
    assert "QUESTION THE NEXT PART ANSWERS: What happens to circle 2?" in call["prompt"]
    assert HOOK in call["prompt"]  # the spine reaches the prompt
    assert written[0] == proposals[0]
    spoken = good.replace("circle_two", "circle two")  # spoken_plain
    assert written[1].text == spoken
    assert written[1].estimated_seconds == pytest.approx(len(spoken.split()) * 60.0 / 150, abs=0.01)
    assert written[2] == proposals[2]  # repeats the next beat: template kept
    assert written[3] == proposals[3]  # 60 words: template kept
    assert [name for name, _ in router.events] == ["bridges_written"]

    keyless = ScriptedRouter(error=MissingProviderKey("GEMINI_API_KEY is not configured"))
    assert asyncio.run(write_bridges(keyless, proposals, lesson, narration)) == proposals
    assert [name for name, _ in keyless.events] == ["bridges_written_by_template"]


# 6 ─────────────────────────────────────────────────────────────────────────


def test_unapproved_opening_moves_hook_into_narration(harness, tmp_path):
    result = PipelineV2(_config(tmp_path, approve_bridges=False)).run(FIXTURE)

    assert result.final_video_path, result.message
    run_dir = Path(result.run_directory)
    first_beat = _storyboard(run_dir)["narration"]["scenes"][0]["beats"][0]["text"]
    assert first_beat.startswith(HOOK)
    assert harness.speak == []  # nothing rendered before approval
    assert len(_named(run_dir, "hook_moved_to_narration")) == 1
    bridges = load_bridges(run_dir)
    assert bridges[0].kind == "opening" and bridges[0].text == HOOK
    assert not any(bridge.approved for bridge in bridges)
    assert all(segment["mode"] == "manim" for segment in _timeline(run_dir))
    assert _cost_rows(run_dir, "presenter") == []

    # Approving the opening afterwards: the presenter says the hook and the
    # first scene is narrated again without it, so it is never said twice.
    approve_bridges(run_dir, ["bridge_001"])
    tts_before = harness.tts.count("scene_001")
    resumed = PipelineV2.resume(run_dir)
    assert resumed.final_video_path, resumed.message
    assert [call["text"] for call in harness.speak] == [HOOK]
    assert not _storyboard(run_dir)["narration"]["scenes"][0]["beats"][0]["text"].startswith(HOOK)
    assert harness.tts.count("scene_001") == tts_before + 1
    assert harness.tts.count("scene_002") == 1  # untouched scenes keep their audio
    assert len(_named(run_dir, "hook_placement_changed")) == 1
    assert _timeline(run_dir)[0]["scene_id"] == "bridge_001"


# 7 & 8 ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def two_bridges(monkeypatch):
    # The fixture lesson is short; place a bridge before every scene.
    monkeypatch.setattr(presenter, "BRIDGE_EVERY_SECONDS", 1.0)
    monkeypatch.setattr(presenter, "TYPICAL_BRIDGE_SECONDS", 1.0)


def test_approved_bridges_render_and_interleave(harness, two_bridges, tmp_path):
    result = PipelineV2(_config(tmp_path, approve_bridges=True)).run(FIXTURE)

    assert result.final_video_path, result.message
    run_dir = Path(result.run_directory)
    bridges = load_bridges(run_dir)
    assert [b.kind for b in bridges] == ["opening", "transition"] and all(b.approved for b in bridges)
    assert [call["text"] for call in harness.speak] == [b.text for b in bridges]  # once each
    assert {call["voice"] for call in harness.speak} == {DEFAULT_VOICE}
    assert bridges[0].text == HOOK
    assert not _storyboard(run_dir)["narration"]["scenes"][0]["beats"][0]["text"].startswith(HOOK)
    assert _named(run_dir, "hook_moved_to_narration") == []

    timeline = _timeline(run_dir)
    by_id = {b.bridge_id: b for b in bridges}
    assert [s["mode"] for s in timeline] == ["presenter", "manim", "presenter", "manim"]
    for index, segment in enumerate(timeline):
        if segment["mode"] == "presenter":
            assert timeline[index + 1]["scene_id"] == by_id[segment["scene_id"]].before_scene_id
            assert Path(segment["video_path"]).is_file()
    assert timeline[-1]["start_seconds"] + timeline[-1]["duration_seconds"] == pytest.approx(
        sum(segment["duration_seconds"] for segment in timeline)
    )

    rows = _cost_rows(run_dir, "presenter")
    assert len(rows) == 2
    for row, bridge in zip(rows, bridges):
        assert (row["provider"], row["model"], row["status"]) == ("google", PRESENTER_MODEL, "SUCCEEDED")
        assert row["reserved_usd"] == pytest.approx(omni_cost_usd(bridge.text))
        assert 0 < row["actual_usd"] == pytest.approx(4.0 * OMNI_USD_PER_SECOND, rel=0.1)

    review = json.loads((run_dir / "final_review.json").read_text(encoding="utf-8"))
    assert review["duration_matches"], review
    delivery = json.loads((run_dir / "delivery_report.json").read_text(encoding="utf-8"))
    assert [scene["scene_id"] for scene in delivery["scenes"]] == [s["scene_id"] for s in timeline]
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Presenter bridges" in report and DEFAULT_VOICE in report and "rendered" in report

    # A revision of one scene keeps the approved clips as they are.
    _plan, revised = PipelineV2.revise(run_dir, parse_ops([
        {"op": "fix_visual", "scene_id": "scene_002", "instruction": "Make the square larger."},
    ]))
    assert revised is not None and revised.final_video_path, revised and revised.message
    assert len(harness.speak) == 2
    assert [s["mode"] for s in _timeline(run_dir)] == ["presenter", "manim", "presenter", "manim"]
    assert len(_cost_rows(run_dir, "presenter")) == 2


def test_resume_does_not_rerender_cached_bridge(harness, two_bridges, tmp_path):
    first = PipelineV2(_config(tmp_path, approve_bridges=True)).run(FIXTURE)
    assert first.final_video_path, first.message
    run_dir = Path(first.run_directory)
    rendered = len(harness.speak)
    assert rendered == 2

    again = PipelineV2.resume(run_dir)

    assert again.final_video_path, again.message
    assert len(harness.speak) == rendered
    resumed = [event["task_id"] or "" for event in _named(run_dir, "task_resumed")]
    assert "bridges" in resumed
    assert any(task.startswith("presenter-bridge_001-") for task in resumed)
    assert any(task.startswith("presenter-bridge_002-") for task in resumed)
    assert any(task.startswith("final-compose-") for task in resumed)

    # A changed bridge text re-renders only that bridge, and the video is rebuilt.
    bridges = load_bridges(run_dir)
    edited = "Here is the part I find surprising, and it is worth a closer look."
    save_bridges(run_dir, [bridges[0], bridges[1].model_copy(update={"text": edited})])
    edited_run = PipelineV2.resume(run_dir)
    assert edited_run.final_video_path, edited_run.message
    assert [call["text"] for call in harness.speak[rendered:]] == [edited]
    clip = next(s for s in _timeline(run_dir) if s["scene_id"] == "bridge_002")["video_path"]
    composed = [e for e in _named(run_dir, "task_finished") if (e["task_id"] or "").startswith("final-compose-")]
    assert len(composed) == 2 and Path(clip).is_file()


# 9 ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("limits", "reason"), [
    ({"max_presenter_seconds": 1}, "presenter_seconds"),
    ({"max_cost_usd": 0.05}, "budget"),  # the Omni reservation alone is about $0.5
])
def test_budget_skip_keeps_hook_spoken(harness, tmp_path, limits, reason):
    result = PipelineV2(_config(tmp_path, approve_bridges=True, **limits)).run(FIXTURE)

    assert result.final_video_path, result.message
    run_dir = Path(result.run_directory)
    assert harness.speak == []  # the cap or the budget refused the Omni clip
    skipped = _named(run_dir, "presenter_skipped")
    assert [(e["scene_id"], e["reason"]) for e in skipped] == [("bridge_001", reason)]
    assert "bridge_001" in harness.tts  # the hook is voiced by TTS instead
    assert len(_named(run_dir, "presenter_title_card")) == 1
    timeline = _timeline(run_dir)
    assert timeline[0]["mode"] == "presenter" and timeline[0]["scene_id"] == "bridge_001"
    assert Path(timeline[0]["video_path"]).name.startswith("bridge_001-card-")
    assert _named(run_dir, "hook_moved_to_narration") == []  # spoken once, by the card
    render = json.loads((run_dir / "presenter" / "render.json").read_text(encoding="utf-8"))
    assert render["bridges"][0]["status"] == "title card"
    assert _cost_rows(run_dir, "presenter") == []


def test_missing_key_fails_the_run_before_any_paid_call(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = PipelineV2(_config(tmp_path)).run(FIXTURE)

    assert result.status == RunStatus.FAILED
    assert "GEMINI_API_KEY is not set; the gemini presenter cannot be used." in result.message
    assert _task_ids(Path(result.run_directory)) == []  # not even source extraction


def test_fade_through_black_at_presenter_cuts():
    from src.pipeline_v2.media import _fade_filters
    from src.pipeline_v2.models import TimelineSegment
    from src.pipeline_v2.pipeline import _presenter_fades

    timeline = [
        TimelineSegment(segment_id=f"s{index}", scene_id=f"x{index}", mode=mode,
                        start_seconds=0.0, duration_seconds=4.0)
        for index, mode in enumerate(["presenter", "manim", "manim", "presenter"])
    ]
    assert _presenter_fades(timeline) == [(0.0, 0.25), (0.25, 0.0), (0.0, 0.25), (0.25, 0.0)]
    assert _fade_filters((0.25, 0.25), 4.0) == "fade=t=in:st=0:d=0.250,fade=t=out:st=3.750:d=0.250,"
    assert _fade_filters((0.0, 0.0), 4.0) == ""


# 10 ────────────────────────────────────────────────────────────────────────


def test_cli_presenter_options_and_bridges_command(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")

    def convert(*args: str) -> PipelineV2Config:
        pipeline = MagicMock()
        pipeline.run.return_value = RunResult(
            run_id="test", run_directory=str(tmp_path), status=RunStatus.SUCCESS,
        )
        with patch("src.cli.PipelineV2", return_value=pipeline) as factory:
            result = CliRunner().invoke(pdf2manim, ["convert", str(FIXTURE), *args])
        assert result.exit_code == 0, result.output
        return factory.call_args.args[0]

    config = convert(
        "--presenter", "gemini", "--approve-bridges", "--max-presenter-seconds", "45",
        "--presenter-persona", "a calm fictional lecturer with grey hair",
    )
    assert (config.presenter, config.approve_bridges, config.max_presenter_seconds) == ("gemini", True, 45.0)
    assert config.presenter_persona == "a calm fictional lecturer with grey hair"
    did = convert("--presenter", "d-id", "--profile", str(profile),
                  "--avatar-image-url", "https://example.com/a.jpg")
    assert did.presenter_profile == profile.resolve()
    assert did.avatar_image_url == "https://example.com/a.jpg"
    default = convert()
    assert (default.presenter, default.approve_bridges, default.max_presenter_seconds) == ("off", False, 60.0)

    run_dir = tmp_path / "run"
    save_bridges(run_dir, [
        BridgeProposal(bridge_id="bridge_001", kind="opening", before_scene_id="scene_001",
                       text=HOOK, estimated_seconds=5.0),
        BridgeProposal(bridge_id="bridge_002", kind="transition", before_scene_id="scene_002",
                       text="Something here is still unexplained.", estimated_seconds=3.0),
    ])
    runner = CliRunner()
    listed = runner.invoke(pdf2manim, ["bridges", str(run_dir)])
    assert listed.exit_code == 0, listed.output
    assert "bridge_001" in listed.output and HOOK in listed.output and "not approved" in listed.output

    approved = runner.invoke(pdf2manim, ["bridges", str(run_dir), "--approve", "all"])
    assert approved.exit_code == 0, approved.output
    assert [b.approved for b in load_bridges(run_dir)] == [True, True]
    assert "pdf2manim resume" in approved.output

    rejected = runner.invoke(pdf2manim, ["bridges", str(run_dir), "--reject", "bridge_002"])
    assert rejected.exit_code == 0, rejected.output
    assert [b.approved for b in load_bridges(run_dir)] == [True, False]

    unknown = runner.invoke(pdf2manim, ["bridges", str(run_dir), "--approve", "bridge_009"])
    assert unknown.exit_code != 0 and "bridge_009" in unknown.output
    assert [b.approved for b in load_bridges(run_dir)] == [True, False]


# 11 ────────────────────────────────────────────────────────────────────────


def test_presenter_off_unchanged(harness, two_bridges, tmp_path):
    result = PipelineV2(_config(tmp_path, presenter="off")).run(FIXTURE)

    assert result.final_video_path, result.message
    run_dir = Path(result.run_directory)
    assert harness.speak == []
    assert not (run_dir / "presenter").exists()
    assert not any(task.startswith(("bridges", "presenter")) for task in _task_ids(run_dir))
    assert _named(run_dir, "hook_moved_to_narration") == []

    timeline = _timeline(run_dir)
    timed = {scene["scene_id"]: scene for scene in _storyboard(run_dir)["narration"]["scenes"]}
    assert [segment["mode"] for segment in timeline] == ["manim"] * len(timeline)
    cursor = 0.0
    for segment in timeline:
        scene = timed[segment["scene_id"]]
        assert segment["start_seconds"] == pytest.approx(cursor)
        assert segment["duration_seconds"] == scene["audio_duration_seconds"]
        assert segment["audio_path"] == scene["audio_path"]
        cursor += segment["duration_seconds"]
    started = [e for e in _named(run_dir, "task_started") if e["task_id"] == "final-compose"]
    assert started and "presenter" not in json.dumps(started)
