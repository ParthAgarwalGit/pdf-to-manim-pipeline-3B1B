"""Offline V2 pipeline and resume acceptance tests."""

import json
from pathlib import Path

import pytest

from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.media import inspect_av


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
