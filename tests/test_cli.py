"""Tests for CLI flag propagation to PipelineConfig."""
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from src.cli import pdf2manim
from src.orchestrator.pipeline import PipelineConfig


def _cost_tracker_cls():
    """Build a MagicMock that behaves like the CostTracker *class*.

    Usage in CLI: ``CostTracker.get_instance()`` → instance with
    ``.get_total_cost() -> float``, ``.get_breakdown() -> dict``, ``.reset()``.
    """
    instance = MagicMock()
    instance.get_total_cost.return_value = 0.0
    instance.get_breakdown.return_value = {"by_provider": {}, "by_model": {}}
    instance.reset.return_value = None
    cls_mock = MagicMock()
    cls_mock.get_instance.return_value = instance
    return cls_mock


def _orchestrator_instance():
    """Build a MagicMock Orchestrator whose run_pipeline returns a valid manifest."""
    inst = MagicMock()
    inst.run_pipeline.return_value = MagicMock(
        output_directory="/tmp/out",
        total_scenes=1,
        successful_scenes=1,
        status="SUCCESS",
        files=[],
        scene_success_rate=1.0,
        final_video_path=None,
        run_config=None,
    )
    return inst


class TestCLIFlagPropagation:
    """Tests that CLI flags are truthfully passed to PipelineConfig."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_pdf(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 test content")
        return str(pdf)

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker", new_callable=_cost_tracker_cls)
    def test_convert_passes_skip_tts_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_orch_cls.return_value = _orchestrator_instance()
        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--skip-tts"])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        config = mock_orch_cls.call_args[1]["config"]
        assert config.enable_tts is False

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker", new_callable=_cost_tracker_cls)
    def test_convert_passes_skip_render_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_orch_cls.return_value = _orchestrator_instance()
        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--skip-render"])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        config = mock_orch_cls.call_args[1]["config"]
        assert config.enable_render is False

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker", new_callable=_cost_tracker_cls)
    def test_convert_passes_dry_run_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_orch_cls.return_value = _orchestrator_instance()
        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--dry-run"])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        config = mock_orch_cls.call_args[1]["config"]
        assert config.dry_run is True

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker", new_callable=_cost_tracker_cls)
    def test_convert_passes_voice_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_orch_cls.return_value = _orchestrator_instance()
        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--voice", "nova"])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        config = mock_orch_cls.call_args[1]["config"]
        assert config.tts_voice == "nova"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker", new_callable=_cost_tracker_cls)
    def test_convert_passes_quality_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_orch_cls.return_value = _orchestrator_instance()
        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "-q", "high"])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        config = mock_orch_cls.call_args[1]["config"]
        assert config.render_quality == "high"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker", new_callable=_cost_tracker_cls)
    def test_default_flags_are_truthful(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_orch_cls.return_value = _orchestrator_instance()
        result = runner.invoke(pdf2manim, ["convert", mock_pdf])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        config = mock_orch_cls.call_args[1]["config"]
        assert config.enable_tts is True
        assert config.enable_render is True
        assert config.dry_run is False
        assert config.tts_voice == "alloy"
        assert config.render_quality == "medium"
