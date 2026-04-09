"""Tests for CLI flag propagation to PipelineConfig."""
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from src.cli import pdf2manim
from src.orchestrator.pipeline import PipelineConfig


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
    @patch("src.cli.CostTracker")
    def test_convert_passes_skip_tts_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_instance = MagicMock()
        mock_instance.run_pipeline.return_value = MagicMock(
            output_directory="/tmp/out", total_scenes=1, successful_scenes=1,
            status="SUCCESS", files=[], scene_success_rate=1.0
        )
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--skip-tts"])

        config = mock_orch_cls.call_args[1]["config"]
        assert config.enable_tts is False, "skip-tts flag should set enable_tts=False"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker")
    def test_convert_passes_skip_render_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_instance = MagicMock()
        mock_instance.run_pipeline.return_value = MagicMock(
            output_directory="/tmp/out", total_scenes=1, successful_scenes=1,
            status="SUCCESS", files=[], scene_success_rate=1.0
        )
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--skip-render"])

        config = mock_orch_cls.call_args[1]["config"]
        assert config.enable_render is False, "skip-render flag should set enable_render=False"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker")
    def test_convert_passes_dry_run_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_instance = MagicMock()
        mock_instance.run_pipeline.return_value = MagicMock(
            output_directory="/tmp/out", total_scenes=1, successful_scenes=1,
            status="SUCCESS", files=[], scene_success_rate=1.0
        )
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--dry-run"])

        config = mock_orch_cls.call_args[1]["config"]
        assert config.dry_run is True, "dry-run flag should set dry_run=True"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker")
    def test_convert_passes_voice_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_instance = MagicMock()
        mock_instance.run_pipeline.return_value = MagicMock(
            output_directory="/tmp/out", total_scenes=1, successful_scenes=1,
            status="SUCCESS", files=[], scene_success_rate=1.0
        )
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "--voice", "nova"])

        config = mock_orch_cls.call_args[1]["config"]
        assert config.tts_voice == "nova", "voice flag should set tts_voice"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker")
    def test_convert_passes_quality_into_pipeline_config(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_instance = MagicMock()
        mock_instance.run_pipeline.return_value = MagicMock(
            output_directory="/tmp/out", total_scenes=1, successful_scenes=1,
            status="SUCCESS", files=[], scene_success_rate=1.0
        )
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(pdf2manim, ["convert", mock_pdf, "-q", "high"])

        config = mock_orch_cls.call_args[1]["config"]
        assert config.render_quality == "high", "quality flag should set render_quality"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker")
    def test_default_flags_are_truthful(self, mock_cost, mock_orch_cls, runner, mock_pdf):
        mock_instance = MagicMock()
        mock_instance.run_pipeline.return_value = MagicMock(
            output_directory="/tmp/out", total_scenes=1, successful_scenes=1,
            status="SUCCESS", files=[], scene_success_rate=1.0
        )
        mock_orch_cls.return_value = mock_instance

        result = runner.invoke(pdf2manim, ["convert", mock_pdf])

        config = mock_orch_cls.call_args[1]["config"]
        assert config.enable_tts is True
        assert config.enable_render is True
        assert config.dry_run is False
        assert config.tts_voice == "alloy"
        assert config.render_quality == "medium"
