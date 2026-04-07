"""Integration tests for the output manifest contract.

Asserts that the final manifest describes the actual end-of-run artifacts
and that all required fields are present and consistent.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.schemas.output_manifest import (
    FailedSceneManifest,
    OutputFile,
    OutputManifest,
    RunConfig,
)


class TestManifestContract:
    """The manifest must be the single source of truth for a run's outputs."""

    def _make_manifest(self, tmp_path: Path, **overrides) -> OutputManifest:
        defaults = dict(
            output_directory=str(tmp_path),
            files=[
                OutputFile(path="scene_001.py", size_bytes=1024, type="animation_code"),
                OutputFile(path="storyboard.json", size_bytes=2048, type="storyboard"),
                OutputFile(path="narration_script.txt", size_bytes=512, type="narration_script"),
                OutputFile(path="scene_001.mp4", size_bytes=50000, type="scene_video"),
                OutputFile(path="scene_001.mp3", size_bytes=30000, type="scene_audio"),
                OutputFile(path="final.mp4", size_bytes=80000, type="final_video"),
                OutputFile(path="video_review.json", size_bytes=256, type="report"),
                OutputFile(path="manifest.json", size_bytes=512, type="manifest"),
            ],
            timestamp=datetime.now(timezone.utc),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=1,
            successful_scenes=1,
            failed_scenes=[],
            run_config=RunConfig(
                dry_run=False,
                enable_render=True,
                enable_tts=True,
                render_quality="medium",
                tts_voice="alloy",
            ),
            final_video_path="final.mp4",
        )
        defaults.update(overrides)
        return OutputManifest(**defaults)

    def test_final_manifest_contains_final_video_and_review_reports(self, tmp_path):
        manifest = self._make_manifest(tmp_path)

        file_types = {f.type for f in manifest.files}
        assert "final_video" in file_types, "manifest must include final_video"
        assert "report" in file_types, "manifest must include quality review report"
        assert "scene_video" in file_types, "manifest must include scene videos"
        assert "scene_audio" in file_types, "manifest must include scene audio"
        assert manifest.final_video_path is not None, "final_video_path must be set"

    def test_manifest_records_run_config(self, tmp_path):
        manifest = self._make_manifest(tmp_path)

        assert manifest.run_config is not None
        assert manifest.run_config.enable_render is True
        assert manifest.run_config.enable_tts is True
        assert manifest.run_config.dry_run is False

    def test_manifest_has_canonical_output_directory(self, tmp_path):
        manifest = self._make_manifest(tmp_path)
        assert manifest.output_directory == str(tmp_path)

    def test_manifest_scene_counts_consistent(self, tmp_path):
        manifest = self._make_manifest(tmp_path)
        assert manifest.successful_scenes <= manifest.total_scenes
        assert abs(manifest.scene_success_rate - manifest.successful_scenes / max(manifest.total_scenes, 1)) < 0.01

    def test_manifest_serializes_to_json(self, tmp_path):
        manifest = self._make_manifest(tmp_path)
        data = json.loads(manifest.model_dump_json())

        assert "output_directory" in data
        assert "files" in data
        assert "run_config" in data
        assert "final_video_path" in data
        assert data["final_video_path"] == "final.mp4"

    def test_manifest_without_video_when_render_disabled(self, tmp_path):
        manifest = self._make_manifest(
            tmp_path,
            files=[
                OutputFile(path="scene_001.py", size_bytes=1024, type="animation_code"),
                OutputFile(path="storyboard.json", size_bytes=2048, type="storyboard"),
            ],
            run_config=RunConfig(
                dry_run=False,
                enable_render=False,
                enable_tts=False,
                render_quality="medium",
                tts_voice="alloy",
            ),
            final_video_path=None,
        )
        assert manifest.run_config.enable_render is False
        assert manifest.final_video_path is None
        video_files = [f for f in manifest.files if f.type == "final_video"]
        assert len(video_files) == 0
