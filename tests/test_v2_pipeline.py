"""Offline V2 pipeline and resume acceptance tests."""

import asyncio
import json
import inspect
from pathlib import Path

import pytest

from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.media import inspect_av
from src.pipeline_v2.models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    SourceDocument,
    SourceFact,
    VerificationFinding,
    VisualContractBatch,
)
from src.pipeline_v2.planning import PlanningService
from src.pipeline_v2.rendering import RenderOutcome


FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"


def _without_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_offline_dry_run_writes_complete_forensic_artifacts_and_resumes(monkeypatch, tmp_path):
    _without_keys(monkeypatch)
    config = PipelineV2Config(
        output_dir=tmp_path, max_scenes=2, dry_run=True,
        enable_tts=False, enable_render=False, repair="off",
    )
    first = PipelineV2(config).run(FIXTURE)
    assert first.status == RunStatus.SUCCESS
    run_dir = Path(first.run_directory)
    for name in ("run.json", "run.db", "events.jsonl", "pipeline.log", "storyboard.json",
                 "storyboard.md", "costs.json", "report.html"):
        assert (run_dir / name).exists(), name
    assert first.estimated_cost_usd == 0.0

    second = PipelineV2.resume(run_dir)
    assert second.status == RunStatus.SUCCESS
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    resumed = {event.get("task_id") for event in events if event["event"] == "task_resumed"}
    assert {"source", "ocr-enrichment", "lesson", "narration"}.issubset(resumed)
    assert json.loads((run_dir / "costs.json").read_text())["total_usd"] == 0.0
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Critical path" in report
    assert "Stage activity" in report


def test_scene_can_use_three_sequential_improving_repairs(tmp_path):
    scene_id = "scene_001"
    fact = SourceFact(fact_id="f1", page_number=1, text="A grounded fact.")
    source = SourceDocument(
        filename="fixture.pdf", sha256="a", page_count=1, facts=[fact],
    )
    lesson = LessonPlan(
        title="Lesson", audience="general", objectives=["Explain the fact"],
        scenes=[LessonScene(
            scene_id=scene_id, title="Scene", description="Explain it",
            learning_goal="Understand it", source_fact_ids=[fact.fact_id],
        )],
    )
    narrated = NarratedScene(
        scene_id=scene_id,
        narration="Explain the fact.",
        beats=[NarrationBeat(
            beat_id="beat_001", text="Explain the fact.",
            source_fact_ids=[fact.fact_id], start_seconds=0.0, end_seconds=2.0,
        )],
        audio_duration_seconds=2.0,
    )
    contract = PlanningService._fallback_contract(narrated)

    class Generator:
        repair_calls = 0

        async def generate(self, *_args, **_kwargs):
            return _code(0)

        async def repair(self, *_args, **_kwargs):
            self.repair_calls += 1
            return _code(self.repair_calls)

    class Renderer:
        def render(self, code, *_args, quality="low", **_kwargs):
            attempt = int(code.split("self.wait(", 1)[1].split(")", 1)[0])
            if quality == "high":
                warning_count = 0
            else:
                warning_count = {0: 6, 1: 5, 2: 4, 3: 0}[attempt]
            return RenderOutcome(
                safe=True,
                render_success=True,
                video_path=str(tmp_path / f"{quality}-{attempt}.mp4"),
                findings=[
                    VerificationFinding(
                        code="TEXT_TEXT_OVERLAP", severity="warning",
                        message=f"overlap pair {index}",
                    )
                    for index in range(warning_count)
                ],
            )

        def terminate_all(self):
            return None

    class Artifacts:
        def write_json(self, path, payload):
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding="utf-8")

        def event(self, *_args, **_kwargs):
            return None

    async def exercise():
        pipeline = PipelineV2(PipelineV2Config(max_repairs_per_scene=3))
        pipeline.renderer = Renderer()
        pipeline.artifacts = Artifacts()

        async def execute(_task_id, _stage, _scene_id, _inputs, operation, **_kwargs):
            value = operation()
            return await value if inspect.isawaitable(value) else value

        pipeline._task = execute
        generator = Generator()
        timed_task = asyncio.create_task(asyncio.sleep(0, result=narrated.model_dump(mode="json")))
        contract_task = asyncio.create_task(asyncio.sleep(
            0,
            result=VisualContractBatch(contracts=[contract]).model_dump(mode="json"),
        ))
        result = await pipeline._process_scene(
            input_path=FIXTURE,
            source=source,
            lesson=lesson,
            original_scene=narrated,
            timed_scene_task=timed_task,
            contract_task=contract_task,
            scene_dir=tmp_path / scene_id,
            generator=generator,
        )
        return result, generator

    result, generator = asyncio.run(exercise())

    assert generator.repair_calls == 3
    assert result.repaired
    assert result.status == "REPAIRED_ACCEPTED"
    assert (tmp_path / scene_id / "code" / "repaired-03.py").exists()
    assert (
        tmp_path / "scenes" / scene_id / "attempt-03" / "diagnostics.json"
    ).exists()


def _code(attempt: int) -> str:
    return (
        "from manim import *\n\n"
        "class GeneratedScene(Scene):\n"
        "    def construct(self):\n"
        f"        self.wait({attempt})\n"
    )


@pytest.mark.integration
def test_offline_one_scene_produces_verified_audio_video(monkeypatch, tmp_path):
    _without_keys(monkeypatch)
    result = PipelineV2(PipelineV2Config(
        output_dir=tmp_path, max_scenes=1, quality="low", repair="off",
        preview_timeout_seconds=180, render_timeout_seconds=180,
    )).run(FIXTURE)
    assert result.status == RunStatus.SUCCESS
    assert result.successful_scenes == 1
    assert result.final_video_path
    review = inspect_av(Path(result.final_video_path))
    assert review["valid"]
    scene = result.scene_results[0]
    assert scene.success
    assert not any(finding.severity == "error" for finding in scene.findings)
    trace = Path(scene.preview_path).parent / "trace.jsonl"
    trace_events = {
        json.loads(line)["event"] for line in trace.read_text(encoding="utf-8").splitlines()
    }
    assert "play_sample" in trace_events

    original_code = Path(scene.original_code_path)
    retried = PipelineV2.retry_scenes(result.run_directory, [scene.scene_id])
    retried_scene = retried.scene_results[0]

    assert retried.status == RunStatus.SUCCESS
    assert retried.final_video_path
    assert original_code.exists()
    assert "retries" in Path(retried_scene.original_code_path).parts
    assert "retry-01" in Path(retried_scene.original_code_path).parts
    assert Path(retried_scene.final_video_path).exists()
