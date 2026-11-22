"""Frame-by-frame visual check: confirmed defects are repaired, redesigned once, or flagged.

Hermetic: the renderer, the code model and the vision model are faked.
"""

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2 import pipeline as pipeline_module
from src.pipeline_v2.generation import generation_request, repair_findings
from src.pipeline_v2.models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    RunResult,
    SourceDocument,
    SourceFact,
    VerificationFinding,
    VisualContractBatch,
)
from src.pipeline_v2.planning import PlanningService
from src.pipeline_v2.providers import MissingProviderKey
from src.pipeline_v2.rendering import RenderOutcome
from src.pipeline_v2.visual_qa import parse_defects, qa_timestamps, review_frames
from tests.test_did_avatar import needs_ffmpeg

FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"
SCENE = "scene_001"
OVERLAP = "the label 'x' is crossed by the arrow from A to B"


def _code(take: int) -> str:
    return f"from manim import *\n\nclass GeneratedScene(Scene):\n    def construct(self):\n        self.wait({take})\n"


def _take(code: str) -> int:
    return int(code.split("self.wait(", 1)[1].split(")", 1)[0])


def _scene():
    fact = SourceFact(fact_id="f1", page_number=1, text="A grounded fact.")
    source = SourceDocument(filename="fixture.pdf", sha256="a", page_count=1, facts=[fact])
    lesson = LessonPlan(
        title="Lesson", audience="general", objectives=["Explain the fact"],
        scenes=[LessonScene(
            scene_id=SCENE, title="Scene", description="Explain it",
            learning_goal="Understand it", source_fact_ids=[fact.fact_id],
        )],
    )
    narrated = NarratedScene(
        scene_id=SCENE, narration="Explain the fact.",
        beats=[NarrationBeat(
            beat_id="beat_001", text="Explain the fact.", source_fact_ids=[fact.fact_id],
            start_seconds=0.0, end_seconds=2.0,
        )],
        audio_duration_seconds=2.0,
    )
    return source, lesson, narrated, PlanningService._fallback_contract(narrated)


class Generator:
    """Fresh designs are take 0 (take 10 when told what to avoid); a repair adds one."""

    router = object()

    def __init__(self):
        self.avoid: list = []
        self.repairs: list[dict] = []

    async def generate(self, *_args, avoid_defects=None, **_kwargs):
        self.avoid.append(avoid_defects)
        return _code(10 if avoid_defects else 0)

    async def repair(self, *, code, findings, frame_paths, **_kwargs):
        self.repairs.append({"codes": [item.code for item in findings], "frames": list(frame_paths)})
        return _code(_take(code) + 1)


class Renderer:
    def __init__(self, folder: Path):
        self.folder = folder

    def render(self, code, *_args, quality="low", **_kwargs):
        video = self.folder / f"{quality}-{_take(code)}.mp4"
        video.write_bytes(f"take {_take(code)}".encode())
        return RenderOutcome(safe=True, render_success=True, video_path=str(video), duration_seconds=3.0)

    def terminate_all(self):
        return None


def _looker(defective, final=None):
    """A visual check that confirms one overlap in each defective take.

    ``defective`` names the preview takes with an overlap (None: every take);
    ``final`` names the final-render takes, defaulting to the same set.
    """

    async def review(router, *, scene_id, video_path, duration, out_dir, task_id):
        quality, take = Path(video_path).stem.split("-")
        take = int(take)
        seen = defective if quality == "low" or final is None else final
        if seen is not None and take not in seen:
            return []
        return [
            VerificationFinding(code="VISUAL_TEXT_OVERLAP", severity="error",
                                message=f"Seen at 1.5s: {OVERLAP} (take {take})",
                                timestamp=1.5, screenshot_path=f"defect-{take}.png"),
            VerificationFinding(code="VISUAL_UNCONFIRMED", severity="warning",
                                message="Seen at 0.8s: maybe a pile up"),
        ]

    return review


def _run(tmp_path, monkeypatch, review, **config):
    monkeypatch.setattr(pipeline_module, "review_frames", review)
    source, lesson, narrated, contract = _scene()
    events: list[tuple[str, dict]] = []

    class Artifacts:
        def write_json(self, path, payload):
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, default=str), encoding="utf-8")

        def event(self, name, **fields):
            events.append((name, fields))

    async def execute(_task_id, _stage, _scene_id, _inputs, operation, **_kwargs):
        value = operation()
        return await value if inspect.isawaitable(value) else value

    async def exercise():
        pipeline = PipelineV2(PipelineV2Config(**{"max_repairs_per_scene": 1, **config}))
        pipeline.renderer = Renderer(tmp_path)
        pipeline.artifacts = Artifacts()
        pipeline._task = execute
        generator = Generator()
        timed = asyncio.create_task(asyncio.sleep(0, result=narrated.model_dump(mode="json")))
        contracts = asyncio.create_task(asyncio.sleep(
            0, result=VisualContractBatch(contracts=[contract]).model_dump(mode="json"),
        ))
        result = await pipeline._process_scene(
            input_path=FIXTURE, source=source, lesson=lesson, original_scene=narrated,
            timed_scene_task=timed, contract_task=contracts, scene_dir=tmp_path / SCENE,
            generator=generator,
        )
        return pipeline, result, generator

    pipeline, result, generator = asyncio.run(exercise())
    return SimpleNamespace(pipeline=pipeline, result=result, generator=generator,
                           events=events, names=[name for name, _ in events])


def _visual(findings):
    return [item for item in findings if item.code.startswith("VISUAL_")]


def test_a_defect_seen_mid_animation_is_repaired(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _looker({0}))
    repair = run.generator.repairs[0]
    # The repair is told about the confirmed defect, shown its frame first,
    # and never sent the unconfirmed guess.
    assert repair["codes"] == ["VISUAL_TEXT_OVERLAP"] and repair["frames"][0] == "defect-0.png"
    assert run.result.status == "REPAIRED_ACCEPTED" and not _visual(run.result.findings)
    assert _take(Path(run.result.final_code_path).read_text(encoding="utf-8")) == 1
    assert run.generator.avoid == [None]  # no redesign was needed
    assert "visual_qa_unconfirmed" in run.names


def test_a_defect_repairs_cannot_clear_gets_one_redesign(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _looker({0, 1}))
    assert run.generator.avoid == [None, [f"Seen at 1.5s: {OVERLAP} (take 0)"]]
    assert "redesign_accepted" in run.names
    assert _take(Path(run.result.final_code_path).read_text(encoding="utf-8")) == 10
    assert run.result.status == "REPAIRED_ACCEPTED" and not _visual(run.result.findings)


def test_a_defect_that_survives_redesign_stays_on_the_scene_and_flags_the_run(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _looker(None))
    assert "redesign_rejected" in run.names and "visual_defects_remaining" in run.names
    assert "final_repair_rejected" in run.names
    assert _take(Path(run.result.final_code_path).read_text(encoding="utf-8")) == 0
    # The final render keeps both looks' findings: its own and the preview's.
    assert [item.code for item in _visual(run.result.findings)] == ["VISUAL_TEXT_OVERLAP"] * 2
    assert run.result.status == "ACCEPTED_WITH_FINDINGS"

    final = RunResult(run_id="r", run_directory=str(tmp_path), status=RunStatus.SUCCESS,
                      final_video_path=str(tmp_path / "final.mp4"))
    run.pipeline._flag_visual_defects(final, [run.result])
    assert final.delivery_status == "needs_review" and final.status == RunStatus.PARTIAL_SUCCESS
    assert "visual_review.json" in final.message
    review = json.loads((tmp_path / "visual_review.json").read_text(encoding="utf-8"))
    assert [row["scene_id"] for row in review] == [SCENE, SCENE]
    assert {row["frame"] for row in review} == {"defect-0.png"}


def test_a_defect_only_the_final_render_shows_is_repaired_and_rendered_again(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _looker(set(), final={0}))
    # The repair sees the final render's frame of the defect first.
    assert run.generator.repairs[0]["frames"][0] == "defect-0.png"
    assert "final_repair_accepted" in run.names
    assert _take(Path(run.result.final_code_path).read_text(encoding="utf-8")) == 1
    assert run.result.render_video_path.endswith("medium-1.mp4")
    assert run.result.status == "REPAIRED_ACCEPTED" and not _visual(run.result.findings)


def test_a_late_defect_the_repair_cannot_clear_keeps_the_first_render_and_is_flagged(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _looker(set(), final={0, 1, 2}))
    assert "final_repair_rejected" in run.names
    assert _take(Path(run.result.final_code_path).read_text(encoding="utf-8")) == 0
    assert run.result.render_video_path.endswith("medium-0.mp4")
    assert [item.code for item in _visual(run.result.findings)] == ["VISUAL_TEXT_OVERLAP"]
    assert run.result.status == "ACCEPTED_WITH_FINDINGS"


def test_a_clean_scene_is_not_flagged(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _looker(set()))
    assert run.result.status == "ACCEPTED" and run.generator.repairs == []
    final = RunResult(run_id="r", run_directory=str(tmp_path), status=RunStatus.SUCCESS,
                      final_video_path=str(tmp_path / "final.mp4"))
    run.pipeline._flag_visual_defects(final, [run.result])
    assert final.delivery_status is None and final.status == RunStatus.SUCCESS


def test_without_a_key_the_check_is_skipped_not_fatal(tmp_path, monkeypatch):
    async def no_key(*_args, **_kwargs):
        raise MissingProviderKey("GEMINI_API_KEY is not configured")

    run = _run(tmp_path, monkeypatch, no_key)
    assert "visual_qa_skipped" in run.names and run.result.status == "ACCEPTED"


def test_switching_the_check_off_skips_it(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, _looker(None), visual_qa=False)
    assert run.generator.repairs == [] and run.result.status == "ACCEPTED"
    assert not any(name.startswith("visual") for name in run.names)


# ── units ────────────────────────────────────────────────────────────────

def test_repair_lists_every_distinct_seen_defect_first():
    findings = [
        VerificationFinding(code="TEXT_FRAME_COLLISION", severity="error", message="edge", object_ids=["a"]),
        VerificationFinding(code="VISUAL_TEXT_OVERLAP", severity="error", message="label x crossed"),
        VerificationFinding(code="VISUAL_TEXT_OVERLAP", severity="error", message="labels y and z overlap"),
    ]
    assert [item.message for item in repair_findings(findings)] == [
        "label x crossed", "labels y and z overlap", "edge",
    ]


def test_redesign_request_names_the_defects_only_when_asked():
    _source, lesson, narrated, contract = _scene()
    common = dict(scene=narrated, contract=contract, source_facts=[], class_name="V2Scene001",
                  lesson_scene=lesson.scenes[0], retry_attempt=1)
    _system, plain = generation_request(**common)
    assert "avoid_defects" not in json.loads(plain)
    _system, redesign = generation_request(**common, avoid_defects=[OVERLAP])
    payload = json.loads(redesign)
    assert payload["avoid_defects"] == [OVERLAP] and "avoid_defects" in payload["regeneration_instruction"]


def test_first_look_answers_are_validated():
    data = {"defects": [
        {"kind": "text_overlap", "images": [2, 3, 99], "best": 3, "what": "a  b"},
        {"kind": "style", "images": [1]},              # not a defect kind
        {"kind": "cut_off", "images": [], "best": 4},  # best alone still places it
        {"kind": "pile_up", "images": ["x"]},          # no usable image
    ]}
    defects = parse_defects(data, image_count=5)
    assert [(item.kind, item.images, item.best, item.what) for item in defects] == [
        ("text_overlap", [2, 3], 3, "a b"), ("cut_off", [4], 4, "cut_off"),
    ]
    assert qa_timestamps(6.0) == [0.75, 2.25, 3.75, 5.25]
    long = qa_timestamps(600.0)
    assert len(long) == 48 and long[-1] > 590  # sparser, but through to the end


@needs_ffmpeg
def test_only_defects_seen_again_up_close_count(tmp_path):
    from tests.test_v2_gemini_presenter import _ffmpeg

    video = tmp_path / "scene.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=640x360:rate=15:duration=4", "-pix_fmt", "yuv420p", str(video))
    answers = iter([
        {"defects": [
            {"kind": "line_through_text", "images": [1, 2], "best": 2, "what": "arrow through label"},
            {"kind": "text_overlap", "images": [1], "best": 1, "what": "two titles overlap"},
        ]},
        {"visible": True, "where": "centre"},
        {"visible": False, "where": ""},
    ])

    class Router:
        artifacts = SimpleNamespace(event=lambda *a, **k: None)

        async def vision(self, **_kwargs):
            return SimpleNamespace(text=json.dumps(next(answers)))

    findings = asyncio.run(review_frames(
        Router(), scene_id=SCENE, video_path=video, duration=4.0, out_dir=tmp_path / "qa", task_id="qa",
    ))
    assert [(item.code, item.severity) for item in findings] == [
        ("VISUAL_TEXT_OVERLAP", "error"), ("VISUAL_UNCONFIRMED", "warning"),
    ]
    assert findings[0].timestamp == pytest.approx(2.25) and Path(findings[0].screenshot_path).is_file()


class _ScriptedRouter:
    """A vision model that gives the scripted answers in turn and records each budget."""

    artifacts = SimpleNamespace(event=lambda *a, **k: None)

    def __init__(self, answers):
        self.answers = iter(answers)
        self.budgets = []
        self.images = []

    async def vision(self, *, max_output_tokens, image_paths, **_kwargs):
        self.budgets.append(max_output_tokens)
        self.images.append([Path(path).name for path in image_paths])
        return SimpleNamespace(text=next(self.answers))


def _scene_video(tmp_path):
    from tests.test_v2_gemini_presenter import _ffmpeg

    video = tmp_path / "scene.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=640x360:rate=15:duration=4", "-pix_fmt", "yuv420p", str(video))
    return video


@needs_ffmpeg
def test_an_unreadable_first_look_fails_instead_of_passing_the_scene(tmp_path):
    from src.pipeline_v2.visual_qa import VISUAL_QA_OUTPUT_TOKENS, VisualCheckUnreadable

    router = _ScriptedRouter(['{"defects": [', "no json here"])
    with pytest.raises(VisualCheckUnreadable):
        asyncio.run(review_frames(router, scene_id=SCENE, video_path=_scene_video(tmp_path),
                                  duration=4.0, out_dir=tmp_path / "qa", task_id="qa"))
    # Asked again with twice the room (a cut-short answer), then given up.
    assert router.budgets == [VISUAL_QA_OUTPUT_TOKENS, 2 * VISUAL_QA_OUTPUT_TOKENS]


@needs_ffmpeg
def test_a_cut_short_answer_is_asked_again_with_more_room(tmp_path):
    router = _ScriptedRouter(['{"defects": [', '{"defects": []}'])
    findings = asyncio.run(review_frames(router, scene_id=SCENE, video_path=_scene_video(tmp_path),
                                         duration=4.0, out_dir=tmp_path / "qa", task_id="qa"))
    assert findings == [] and len(router.budgets) == 2


@needs_ffmpeg
def test_an_unreadable_close_look_keeps_the_defect(tmp_path):
    first = json.dumps({"defects": [
        {"kind": "text_overlap", "images": [1], "best": 1, "what": "two labels overlap"},
    ]})
    router = _ScriptedRouter([first, "{not json", "still not json"])
    findings = asyncio.run(review_frames(router, scene_id=SCENE, video_path=_scene_video(tmp_path),
                                         duration=4.0, out_dir=tmp_path / "qa", task_id="qa"))
    assert [(item.code, item.severity) for item in findings] == [("VISUAL_TEXT_OVERLAP", "error")]
    assert "could not be read" in findings[0].message


def test_a_failed_check_is_recorded_and_the_scene_goes_on(tmp_path, monkeypatch):
    from src.pipeline_v2.visual_qa import VisualCheckUnreadable

    async def unreadable(*_args, **_kwargs):
        raise VisualCheckUnreadable("The visual check's answer could not be read.")

    run = _run(tmp_path, monkeypatch, unreadable)
    assert "visual_qa_failed" in run.names and run.result.status == "ACCEPTED"


@needs_ffmpeg
def test_a_defect_is_confirmed_across_the_frames_it_was_seen_in(tmp_path):
    # A moving overlap can be plain in one frame and gone in another: the close
    # look gets the clearest frame and the first and last frames it was seen in.
    first = json.dumps({"defects": [
        {"kind": "text_overlap", "images": [1, 2], "best": 1, "what": "box through label"},
    ]})
    router = _ScriptedRouter([first, json.dumps({"visible": True, "frame": 2, "where": "top"})])
    findings = asyncio.run(review_frames(router, scene_id=SCENE, video_path=_scene_video(tmp_path),
                                         duration=4.0, out_dir=tmp_path / "qa", task_id="qa"))
    assert router.images[1] == ["confirm-00-0.png", "confirm-00-1.png"]
    assert [(item.code, item.timestamp) for item in findings] == [("VISUAL_TEXT_OVERLAP", 2.25)]
    assert Path(findings[0].screenshot_path).name == "confirm-00-1.png"  # where it was seen
