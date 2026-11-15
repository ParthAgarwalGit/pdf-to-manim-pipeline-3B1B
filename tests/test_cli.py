"""V2 CLI configuration and run-management tests."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli import pdf2manim
from src.pipeline_v2.models import RunResult, RunStatus


def _result(tmp_path):
    return RunResult(
        run_id="run", run_directory=str(tmp_path), status=RunStatus.SUCCESS,
        total_scenes=1, successful_scenes=1, report_path=str(tmp_path / "report.html"),
    )


def test_convert_passes_v2_budget_policy_and_workers(tmp_path):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    pipeline = MagicMock()
    pipeline.run.return_value = _result(tmp_path)
    with patch("src.cli.PipelineV2", return_value=pipeline) as pipeline_class:
        result = CliRunner().invoke(pdf2manim, [
            "convert", str(pdf), "--policy", "strict", "--max-cost", "0.75",
            "--repair", "off", "--verifier-mode", "audit",
            "--max-repairs-per-scene", "2",
            "--min-scenes", "16",
            "--max-scenes", "18",
            "--pacing", "relaxed",
            "--api-workers", "2", "--code-workers", "1",
            "--skip-tts", "--skip-render",
        ])
    assert result.exit_code == 0, result.output
    config = pipeline_class.call_args.args[0]
    assert config.policy == "strict"
    assert config.max_cost_usd == 0.75
    assert config.repair == "off"
    assert config.max_repairs_per_scene == 2
    assert config.min_scenes == 16
    assert config.max_scenes == 18
    assert config.pacing == "relaxed"
    assert config.verifier_mode == "audit"
    assert "Verifier:    audit" in result.output
    assert "Scene range: 16-18" in result.output
    assert "Pacing:      relaxed" in result.output
    assert config.global_api_workers == 2
    assert config.code_workers == 1
    assert not config.enable_tts
    assert not config.enable_render


def test_status_and_report_commands(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({
        "run_id": "abc", "status": "SUCCESS", "input_pdf": "x.pdf",
        "result": {
            "successful_scenes": 2, "total_scenes": 2, "final_video_path": "final.mp4",
            "scene_results": [
                {"scene_id": "scene_001", "status": "ACCEPTED"},
                {"scene_id": "scene_002", "status": "REPAIRED_ACCEPTED", "repaired": True},
            ],
        },
    }))
    (tmp_path / "costs.json").write_text(json.dumps({"total_usd": 0.1234}))
    (tmp_path / "report.html").write_text("report")
    runner = CliRunner()
    status = runner.invoke(pdf2manim, ["status", str(tmp_path)])
    report = runner.invoke(pdf2manim, ["report", str(tmp_path)])
    assert status.exit_code == 0
    assert "2/2" in status.output
    assert "$0.1234" in status.output
    assert "scene_001: ACCEPTED" in status.output
    assert "scene_002: REPAIRED_ACCEPTED (repaired)" in status.output
    assert report.exit_code == 0
    assert "report.html" in report.output


def test_retry_scenes_passes_exact_run_and_scene_ids(tmp_path):
    with patch("src.cli.PipelineV2.retry_scenes", return_value=_result(tmp_path)) as retry:
        result = CliRunner().invoke(pdf2manim, [
            "retry-scenes", str(tmp_path), "scene_002", "scene_005",
            "--max-cost", "1.50",
        ])

    assert result.exit_code == 0, result.output
    retry.assert_called_once_with(
        tmp_path,
        ["scene_002", "scene_005"],
        max_cost_usd=1.5,
    )
    assert "scene_002, scene_005" in result.output
