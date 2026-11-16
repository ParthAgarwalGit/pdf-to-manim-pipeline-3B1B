"""Offline acceptance of the integration's public V2 entry points."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli import pdf2manim
from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.directives import ConceptPlan, interpret_brief
from src.pipeline_v2.models import RunResult
from src.pipeline_v2.revision import EditBeat, FixVisual, SetPause, plan_revision


FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"


def test_convert_forwards_professor_and_sandbox_options(tmp_path):
    brief = tmp_path / "brief.md"
    brief.write_text("Under 2 minutes. Use j instead of i.", encoding="utf-8")
    pipeline = MagicMock()
    pipeline.run.return_value = RunResult(
        run_id="test", run_directory=str(tmp_path), status=RunStatus.SUCCESS,
    )
    with patch("src.cli.PipelineV2", return_value=pipeline) as factory:
        result = CliRunner().invoke(pdf2manim, [
            "convert", str(FIXTURE), "--brief", str(brief),
            "--target-minutes", "3", "--render-sandbox", "docker",
            "--sandbox-image", "local:test", "--no-delivery-check",
        ])
    assert result.exit_code == 0, result.output
    config = factory.call_args.args[0]
    assert config.directives["global"]["target_minutes"] == 2
    assert config.target_minutes == 3
    assert config.render_sandbox == "docker"
    assert config.render_sandbox_image == "local:test"
    assert not config.enable_delivery_checks


def test_professor_controls_are_persisted_and_resumed(tmp_path):
    config = PipelineV2Config(
        output_dir=tmp_path, dry_run=True, max_scenes=2,
        directives=interpret_brief("Under 2 minutes.").model_dump(by_alias=True),
    )
    first = PipelineV2(config).run(FIXTURE)
    assert first.status == RunStatus.SUCCESS
    run = Path(first.run_directory)
    assert (run / "professor_controls.json").is_file()
    assert (run / "compliance.json").is_file()
    second = PipelineV2.resume(run)
    assert second.status == RunStatus.SUCCESS
    assert first.estimated_cost_usd == second.estimated_cost_usd == 0
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "task_resumed" and e.get("task_id") == "narration" for e in events)


def test_plan_command_creates_reusable_outline_without_calls(tmp_path):
    target = tmp_path / "plan.yaml"
    result = CliRunner().invoke(pdf2manim, ["plan", str(FIXTURE), "-o", str(target)])
    assert result.exit_code == 0, result.output
    assert ConceptPlan.load(target).concepts
    assert "Paid calls made: 0" in result.output


def test_revision_dry_run_and_visual_revision_reuse_unaffected_work(tmp_path):
    first = PipelineV2(PipelineV2Config(
        output_dir=tmp_path, max_scenes=2, enable_tts=False,
        enable_render=False, repair="off",
    )).run(FIXTURE)
    assert first.status == RunStatus.SUCCESS
    run = Path(first.run_directory)
    storyboard = json.loads((run / "storyboard.json").read_text())
    first_scene = storyboard["narration"]["scenes"][0]
    beat_id = first_scene["beats"][0]["beat_id"]
    before = (run / "storyboard.json").read_bytes()
    edit = EditBeat(beat_id=beat_id, text="A revised source explanation.")
    plan, result = PipelineV2.revise(run, [edit], dry_run=True)
    assert result is None
    assert len(plan.actions) == 1
    assert plan.actions[0].audio == "regenerate"
    assert (run / "storyboard.json").read_bytes() == before
    pause = plan_revision(run, [SetPause(beat_id=beat_id, pause_seconds=1.0)])
    assert pause.actions[0].audio != "regenerate"
    scene_id = first_scene["scene_id"]
    visual = FixVisual(scene_id=scene_id, instruction="Move labels further apart.")
    plan, revised = PipelineV2.revise(run, [visual])
    assert plan.actions[0].audio == "reuse"
    assert revised.status == RunStatus.SUCCESS
    untouched = first.scene_results[1]
    assert revised.scene_results[1].final_code_path == untouched.final_code_path
    assert Path(untouched.final_code_path).exists()
    resumed = PipelineV2.resume(run)
    assert resumed.status == RunStatus.SUCCESS
    assert resumed.estimated_cost_usd == 0
