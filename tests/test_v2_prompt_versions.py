"""Prompt versions per video mode, and the planning-task hashes they key.

A planning task's input hash includes its mode's prompt version, so a resume
reuses a task only while that hash is unchanged. The digests below were
computed with the task-input formula from before the per-mode split: deep at
v2.7.0-modes / v2.8.0-voice (what deep runs had before 070d988), brief at
v2.7.1-brief-coverage / v2.8.1-voice (070d988's values). If one of these
tests fails, every existing run of that mode re-plans on resume: change a
digest only together with a deliberate version bump.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.models import LessonPlan, SourceDocument, SourceFact
from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.pipeline import _input_hash
from src.pipeline_v2.planning import (
    PROMPT_VERSION_ALIASES,
    PROMPT_VERSIONS,
    PlanningService,
    prompt_version,
    prompt_version_aliases,
)
from src.pipeline_v2.state import RunStore

FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"

DEEP_HASHES = {
    "ocr-enrichment": "4640cdb066804cdf3ae04b0ada2691b3dd0f9b721ce3b0288e905614b76e4a73",
    "lesson": "f8931b70d07276420f8e604c34ff725a05fc875f8b21065068c2cb6e7ee92c5d",
    "narration": "fc57ef57854c05d1c1348f5bd36860ce677b0d0ec27f0c66d01f936a5fa81e15",
    "narration-revised": "a32d1bcbd266d252ccb2888a68e20346a50d91a6c232318f48eaab9a101f4d6c",
}
# Deep runs planned while both modes shared one version (070d988).
DEEP_SHARED_BUMP_HASHES = {
    "ocr-enrichment": "4beecca1192a1dab2f1c7a08032f88ad76fdb7bd017f64bbf5851656804b45e2",
    "lesson": "69d6bb406bf99ce338d0d9bc74e918db0a725e5cdfd987ac5b6c9ceedc1ddd29",
    "narration": "222b18b8469dbe4a3a647f9f182e40b262c48e26b34bd0416c94c12a2e8d8437",
    "narration-revised": "fdb5c7c233ff094eafb0a6f4cd3a3e03110a688d652a1109502237b1dfa9e281",
}
BRIEF_HASHES = {
    "ocr-enrichment": "4beecca1192a1dab2f1c7a08032f88ad76fdb7bd017f64bbf5851656804b45e2",
    "lesson": "2fbbc32759474c6c570f0950ec01ee4e219d862fcab8a7418b27674acbfe8b8f",
    "narration": "4e2041fd34cf336441d776b9887e64c1a246999b359eb7fa6ebd2d99119e69fb",
    "narration-revised": "ebf619d6838f06c223d17fe629f4fca8c958cec63946b4bdf3dbcdf5e02ffe84",
}
KIND = {"ocr-enrichment": "planning", "lesson": "planning",
        "narration": "narration", "narration-revised": "narration"}


def _source() -> SourceDocument:
    return SourceDocument(
        filename="notes.pdf", sha256="0" * 64, page_count=2,
        facts=[SourceFact(fact_id="p001-f000", page_number=1,
                          text="A vector space is closed under addition.")],
        low_confidence_pages=[2],
    )


def _lesson() -> LessonPlan:
    return LessonPlan.model_validate({
        "title": "Vector spaces", "audience": "undergraduate", "objectives": ["See closure"],
        "central_question": "Why closure?", "hook": "", "takeaway": "Closure.",
        "scenes": [{"scene_id": "scene_001", "title": "Closure", "description": "d",
                    "learning_goal": "g", "source_fact_ids": ["p001-f000"]}],
    })


def _pipeline(mode: str) -> PipelineV2:
    return PipelineV2(PipelineV2Config(mode=mode, min_scenes=None, max_scenes=20,
                                       pacing="standard"))


def _inputs(pipeline: PipelineV2) -> dict[str, dict]:
    return {
        "ocr-enrichment": pipeline._ocr_inputs(_source()),
        "lesson": pipeline._lesson_inputs(_source(), {}),
        "narration": pipeline._narration_inputs(_lesson(), {}),
        "narration-revised": pipeline._narration_inputs(
            _lesson(), {}, review_feedback=["Say why."],
        ),
    }


def _hashes(pipeline: PipelineV2) -> dict[str, str]:
    return {task: _input_hash(inputs) for task, inputs in _inputs(pipeline).items()}


def _events(run_dir: Path) -> list[dict]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ── versions ────────────────────────────────────────────────────────────────


def test_each_mode_has_its_own_prompt_versions():
    assert prompt_version("deep", "planning") == "v2.7.0-modes"
    assert prompt_version("deep", "narration") == "v2.8.0-voice"
    assert prompt_version("brief", "planning") == "v2.7.1-brief-coverage"
    assert prompt_version("brief", "narration") == "v2.8.1-voice"
    assert prompt_version_aliases("deep", "planning") == ("v2.7.1-brief-coverage",)
    assert prompt_version_aliases("deep", "narration") == ("v2.8.1-voice",)
    assert prompt_version_aliases("brief", "planning") == ()


class _Router:
    strict_keys = False

    def __init__(self):
        self.calls: list[dict] = []
        self.artifacts = SimpleNamespace(event=lambda *args, **kwargs: None)

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["fallback"]()


@pytest.mark.parametrize("mode", ["brief", "deep"])
def test_planning_prompts_carry_their_modes_versions(mode, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # allow the rule-based fallback
    router = _Router()
    planner = PlanningService(router, mode=mode, min_scenes=1)
    asyncio.run(planner.lesson(_source()))
    asyncio.run(planner.narration(_lesson(), _source()))
    prompts = {call["stage"]: call["prompt"] for call in router.calls}
    assert prompts["lesson"].startswith(
        f"PROMPT_VERSION={prompt_version(mode, 'planning')}\n")
    assert prompts["narration"].startswith(
        f"PROMPT_VERSION={prompt_version(mode, 'narration')}\n")


# ── task hashes ─────────────────────────────────────────────────────────────


def test_deep_planning_task_hashes_are_unchanged():
    assert _hashes(_pipeline("deep")) == DEEP_HASHES


def test_brief_planning_task_hashes_are_unchanged():
    assert _hashes(_pipeline("brief")) == BRIEF_HASHES


def test_deep_ocr_hash_matches_an_existing_deep_run():
    # The Agreement Protocols deep run of 2026-11-20 stored this hash.
    source = _source().model_copy(update={
        "sha256": "5408e2a4916d01e03b519cb1a846b4014ef04e6530a02a389d7ed13acd7bd068",
        "low_confidence_pages": list(range(1, 31)),
    })
    assert _input_hash(_pipeline("deep")._ocr_inputs(source)) == (
        "044ebaf7e110f054fb48ddbb750d9c4438deb4012b5a02c9adbe465852794331"
    )


def test_deep_equivalent_inputs_hash_like_the_shared_bump():
    pipeline = _pipeline("deep")
    equivalent = {
        task: [_input_hash(item) for item in pipeline._equivalent_inputs(inputs, KIND[task])]
        for task, inputs in _inputs(pipeline).items()
    }
    assert equivalent == {task: [digest] for task, digest in DEEP_SHARED_BUMP_HASHES.items()}
    assert _pipeline("brief")._equivalent_inputs(_inputs(pipeline)["lesson"], "planning") == []


def test_a_brief_version_bump_leaves_deep_hashes_alone(monkeypatch):
    monkeypatch.setitem(PROMPT_VERSIONS, "brief",
                        {"planning": "v9-brief-only", "narration": "v9-brief-only"})
    assert _hashes(_pipeline("deep")) == DEEP_HASHES
    assert all(_hashes(_pipeline("brief"))[task] != BRIEF_HASHES[task] for task in BRIEF_HASHES)


# ── the task cache ──────────────────────────────────────────────────────────


def _bare_pipeline(tmp_path, mode: str = "deep") -> PipelineV2:
    pipeline = _pipeline(mode)
    pipeline.store = RunStore(tmp_path / "run.db", 10.0)
    pipeline.artifacts = RunArtifacts(tmp_path / "run", "run")
    return pipeline


def test_deep_reuses_a_lesson_planned_under_the_shared_bump(tmp_path):
    pipeline = _bare_pipeline(tmp_path)
    inputs = pipeline._lesson_inputs(_source(), {})
    calls = []

    async def operation(value):
        calls.append(value)
        return {"lesson": value}

    async def run(payload, value, **kwargs):
        return await pipeline._task("lesson", "lesson", None, payload,
                                    lambda: operation(value), **kwargs)

    # Planned by the shared-version code: stored under the brief bump's string.
    asyncio.run(run({**inputs, "prompt_version": "v2.7.1-brief-coverage"}, "old"))
    reused = asyncio.run(run(inputs, "new", equivalent_inputs=pipeline._equivalent_inputs(
        inputs, "planning")))

    assert reused == {"lesson": "old"} and calls == ["old"]
    # Recorded under the deep version now, so later resumes hit directly.
    assert pipeline.store.task_state("lesson")["input_hash"] == DEEP_HASHES["lesson"]
    assert asyncio.run(run(inputs, "newer")) == {"lesson": "old"}
    assert not [e for e in _events(tmp_path / "run") if e["event"] == "replan_warning"]


def test_rerunning_a_succeeded_planning_task_warns_with_the_old_hash(tmp_path):
    pipeline = _bare_pipeline(tmp_path)
    pipeline._status_before = RunStatus.SUCCESS.value

    async def run(task_id, payload):
        return await pipeline._task(task_id, "narration", None, payload,
                                    lambda: asyncio.sleep(0, result={"ok": payload}))

    asyncio.run(run("narration", {"v": 1}))  # first run: nothing to warn about
    asyncio.run(run("code-scene_001", {"v": 1}))
    assert not [e for e in _events(tmp_path / "run") if e["event"] == "replan_warning"]

    asyncio.run(run("narration", {"v": 2}))
    asyncio.run(run("code-scene_001", {"v": 2}))  # not a planning task
    warnings = [e for e in _events(tmp_path / "run") if e["event"] == "replan_warning"]
    assert len(warnings) == 1
    assert warnings[0]["task_id"] == "narration"
    assert warnings[0]["previous_input_hash"] == _input_hash({"v": 1})
    assert warnings[0]["input_hash"] == _input_hash({"v": 2})
    assert warnings[0]["run_status_before"] == "SUCCESS"
    assert "retry-scenes or revise" in warnings[0]["message"]


# ── end to end, offline ─────────────────────────────────────────────────────


def _dry_run_config(tmp_path) -> PipelineV2Config:
    return PipelineV2Config(output_dir=tmp_path, mode="deep", dry_run=True,
                            enable_tts=False, enable_render=False, repair="off")


def test_resuming_a_finished_deep_run_after_a_brief_bump_replans_nothing(monkeypatch, tmp_path):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    first = PipelineV2(_dry_run_config(tmp_path)).run(FIXTURE)
    assert first.status == RunStatus.SUCCESS, first.message
    run_dir = Path(first.run_directory)

    monkeypatch.setitem(PROMPT_VERSIONS, "brief",
                        {"planning": "v9-brief-only", "narration": "v9-brief-only"})
    assert PipelineV2.resume(run_dir).status == RunStatus.SUCCESS
    events = _events(run_dir)
    resumed = {e["task_id"] for e in events if e["event"] == "task_resumed"}
    assert {"ocr-enrichment", "lesson", "narration"} <= resumed
    assert not [e for e in events if e["event"] == "replan_warning"]

    # A deep bump does re-plan, and says so before each planning task re-runs.
    monkeypatch.setitem(PROMPT_VERSIONS, "deep",
                        {"planning": "v9-deep", "narration": "v9-deep"})
    monkeypatch.setitem(PROMPT_VERSION_ALIASES, "deep", {})
    assert PipelineV2.resume(run_dir).status == RunStatus.SUCCESS
    warned = [e for e in _events(run_dir)[len(events):] if e["event"] == "replan_warning"]
    assert {e["task_id"] for e in warned} == {"ocr-enrichment", "lesson", "narration"}
    assert {e["run_status_before"] for e in warned} == {"SUCCESS"}
