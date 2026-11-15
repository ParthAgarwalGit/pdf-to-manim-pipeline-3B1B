"""Offline coverage for frozen cases, evaluation execution, and dashboard artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.cli import pdf2manim
from src.evaluation.cases import build_benchmark, load_benchmark
from src.evaluation.dashboard import _validated_rating
from src.evaluation.models import EvaluationRunConfig, EvaluationTrack, ModelSpec
from src.evaluation.runner import EvaluationRunner
from src.pipeline_v2.models import VerificationFinding
from src.pipeline_v2.providers import ProviderResponse
from src.pipeline_v2.rendering import RenderOutcome


RAW_CODE = '''from manim import *

class Draft(Scene):
    def construct(self):
        self.next_section("scene_01_beat_01")
        dot = Dot()
        self.play(FadeIn(dot))
'''


def _production_run(tmp_path: Path) -> Path:
    run = tmp_path / "output" / "circles" / "run-001"
    scene_dir = run / "scenes" / "scene_01"
    (run / "source").mkdir(parents=True)
    (scene_dir / "code").mkdir(parents=True)
    (scene_dir / "attempt-00" / "frames").mkdir(parents=True)
    fact = {
        "fact_id": "f1", "page_number": 1,
        "text": "A radius joins the center to the circle.",
        "kind": "text", "confidence": 1.0,
    }
    lesson = {
        "scene_id": "scene_01", "title": "Radius", "description": "Introduce radius.",
        "learning_goal": "Identify a radius.", "source_fact_ids": ["f1"],
        "visual_objective": "Draw a radius.",
        "new_visual_fact": "The radius ends at the circle.",
    }
    narration = {
        "scene_id": "scene_01", "narration": "This segment is a radius.",
        "beats": [{
            "beat_id": "scene_01_beat_01", "text": "This segment is a radius.",
            "source_fact_ids": ["f1"], "start_seconds": 0.0, "end_seconds": 2.0,
        }],
        "audio_duration_seconds": 2.0,
    }
    contract = {
        "scene_id": "scene_01", "layout_template": "full_visual", "archetype": "geometry",
        "artifact_inventory": [{
            "artifact_id": "radius", "category": "primary_visual", "role": "radius",
            "purpose": "Show the radius", "source_fact_ids": ["f1"],
            "beat_ids": ["scene_01_beat_01"], "enter_beat": "scene_01_beat_01",
            "exit_beat": "scene_01_beat_01", "region": "content_full",
        }],
        "composition_plan": [{
            "beat_id": "scene_01_beat_01", "active_artifact_ids": ["radius"],
        }],
    }
    (run / "run.json").write_text(json.dumps({
        "run_id": "run-001", "result": {"scene_results": [{"scene_id": "scene_01"}]},
    }))
    (run / "source" / "document.json").write_text(json.dumps({
        "filename": "circles.pdf", "sha256": "abc", "page_count": 1, "facts": [fact],
    }))
    (run / "storyboard.json").write_text(json.dumps({
        "lesson": {"scenes": [lesson]}, "narration": {"scenes": [narration]},
    }))
    (scene_dir / "scene.json").write_text(json.dumps({
        "lesson": lesson, "narration": narration,
    }))
    (scene_dir / "visual_contract.json").write_text(json.dumps(contract))
    (scene_dir / "code" / "original.py").write_text(RAW_CODE)
    (scene_dir / "attempt-00" / "diagnostics.json").write_text(json.dumps({
        "findings": [{
            "code": "TEXT_OVERLAP", "severity": "warning", "message": "Overlap",
        }],
    }))
    (scene_dir / "attempt-00" / "frames" / "frame.png").write_bytes(b"png")
    return run


class _FakeRouter:
    async def text(self, **_kwargs):
        return ProviderResponse(
            text=RAW_CODE, provider="google", model="fake", input_units=10,
            output_units=20, exact_usage=True, raw={},
        )

    async def vision(self, **_kwargs):
        return await self.text()


class _FakeRenderer:
    def render(self, code, _contract, _scene, output_dir, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        video = output_dir / "video.mp4"
        frame = output_dir / "frame.png"
        video.write_bytes(b"video")
        frame.write_bytes(b"frame")
        return RenderOutcome(
            safe=True,
            render_success=True,
            video_path=str(video.resolve()),
            frame_paths=[str(frame.resolve())],
            duration_seconds=2.0,
        )

    def terminate_all(self):
        return None


class _ImprovingRenderer(_FakeRenderer):
    def render(self, code, contract, scene, output_dir, **kwargs):
        outcome = super().render(code, contract, scene, output_dir, **kwargs)
        if output_dir.name == "attempt-00":
            outcome.findings = [VerificationFinding(
                code="TEXT_TEXT_OVERLAP", severity="warning", message="Overlap",
            )]
        return outcome


class _FailRouter:
    async def text(self, **_kwargs):
        pytest.fail("completed candidate called provider")

    async def vision(self, **_kwargs):
        pytest.fail("completed candidate called provider")


def _router_factory(_store, _artifacts, _workers):
    return _FakeRouter()


def test_build_and_validate_frozen_benchmark(tmp_path):
    run = _production_run(tmp_path)
    benchmark = tmp_path / "evals" / "benchmarks" / "scene-v1"

    manifest = build_benchmark(run, benchmark, category="geometry")
    loaded, cases = load_benchmark(benchmark)

    assert loaded.benchmark_hash == manifest.benchmark_hash
    assert len(cases) == 1
    assert cases[0].category == "geometry"
    assert cases[0].source_facts[0].fact_id == "f1"
    assert Path(cases[0].initial_code_path).is_file()
    assert cases[0].initial_findings[0].code == "TEXT_OVERLAP"


def test_modified_case_fails_hash_validation(tmp_path):
    benchmark = tmp_path / "benchmark"
    build_benchmark(_production_run(tmp_path), benchmark)
    case_path = next(benchmark.glob("cases/*/case.json"))
    payload = json.loads(case_path.read_text())
    payload["lesson"]["title"] = "Tampered"
    case_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash mismatch"):
        load_benchmark(benchmark)


def test_generator_evaluation_runs_offline_and_builds_dashboard(tmp_path):
    benchmark = tmp_path / "benchmark"
    build_benchmark(_production_run(tmp_path), benchmark)
    config = EvaluationRunConfig(
        benchmark_path=benchmark,
        output_root=tmp_path / "runs",
        track=EvaluationTrack.GENERATOR,
        models=[ModelSpec(alias="fake", provider="google", model="fake-model")],
        max_cost_usd=1.0,
    )

    result = EvaluationRunner(
        config, router_factory=_router_factory, renderer=_FakeRenderer(),
    ).run()

    assert result.status == "SUCCESS"
    assert result.completed_candidates == 1
    assert result.candidates[0].metrics.production_ready
    dashboard = Path(result.dashboard_path)
    assert dashboard.is_file()
    text = dashboard.read_text(encoding="utf-8")
    assert "Blind review" in text
    assert "Candidate A" in text


def test_cli_eval_build_validate_and_plan_make_no_provider_calls(tmp_path):
    run = _production_run(tmp_path)
    benchmark = tmp_path / "benchmark"
    runner = CliRunner()

    built = runner.invoke(pdf2manim, [
        "eval", "build", str(run), "--benchmark", str(benchmark),
        "--category", "geometry",
    ])
    validated = runner.invoke(pdf2manim, ["eval", "validate", str(benchmark)])
    planned = runner.invoke(pdf2manim, [
        "eval", "plan", str(benchmark), "--model", "gemini",
    ])

    assert built.exit_code == 0, built.output
    assert validated.exit_code == 0, validated.output
    assert "Validation:   passed" in validated.output
    assert planned.exit_code == 0, planned.output
    assert "Paid calls made: 0" in planned.output


def test_repair_and_production_tracks_use_frozen_inputs(tmp_path):
    benchmark = tmp_path / "benchmark"
    build_benchmark(_production_run(tmp_path), benchmark)
    model = ModelSpec(alias="fake", provider="google", model="fake-model")

    repaired = EvaluationRunner(EvaluationRunConfig(
        benchmark_path=benchmark,
        output_root=tmp_path / "repair-runs",
        track=EvaluationTrack.REPAIR,
        models=[model],
        max_cost_usd=1.0,
    ), router_factory=_router_factory, renderer=_FakeRenderer()).run()
    production = EvaluationRunner(EvaluationRunConfig(
        benchmark_path=benchmark,
        output_root=tmp_path / "production-runs",
        track=EvaluationTrack.PRODUCTION,
        models=[model],
        max_cost_usd=1.0,
    ), router_factory=_router_factory, renderer=_ImprovingRenderer()).run()

    assert repaired.candidates[0].repair_accepted
    assert production.candidates[0].repaired
    assert production.candidates[0].repair_accepted
    assert production.candidates[0].metrics.issue_score == 0
    candidate_dir = Path(production.candidates[0].code_path).parent
    assert (candidate_dir / "attempt-00" / "video.mp4").is_file()
    assert (candidate_dir / "repair-01" / "video.mp4").is_file()


def test_resume_skips_completed_candidates(tmp_path):
    benchmark = tmp_path / "benchmark"
    build_benchmark(_production_run(tmp_path), benchmark)
    config = EvaluationRunConfig(
        benchmark_path=benchmark,
        output_root=tmp_path / "runs",
        track=EvaluationTrack.GENERATOR,
        models=[ModelSpec(alias="fake", provider="google", model="fake-model")],
        max_cost_usd=1.0,
    )
    first = EvaluationRunner(
        config, router_factory=_router_factory, renderer=_FakeRenderer(),
    ).run()

    resumed = EvaluationRunner(
        config,
        router_factory=lambda *_args: _FailRouter(),
        renderer=_FakeRenderer(),
    ).run(Path(first.run_directory))

    assert resumed.status == "SUCCESS"
    assert resumed.completed_candidates == 1


def test_rating_validation_rejects_incomplete_scores():
    with pytest.raises(ValueError, match="between 1 and 5"):
        _validated_rating({
            "candidate_id": "c", "case_id": "case", "verdict": "approve",
            "scores": {"correctness": 5},
        })

    result = _validated_rating({
        "candidate_id": "c", "case_id": "case", "verdict": "approve",
        "scores": {name: 4 for name in (
            "correctness", "layout", "teaching", "creativity", "timing",
        )},
    })
    assert result["scores"]["layout"] == 4
