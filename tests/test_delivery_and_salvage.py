"""P0.2 — honest delivery metrics + render-time salvage.

A real run reported scene_success_rate=1.0 while 13/90 scenes were missing
from the delivered video. These tests pin the new truth (delivered/lost in
the manifest) and the salvage loop that regenerates a failed scene once with
the manim error as feedback instead of dropping it.
"""

import pytest

from src.agents.video_renderer import SceneRenderResult, VideoRendererOutput
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.schemas.output_manifest import OutputManifest
from tests.test_agent_handoffs import _full_storyboard


def _orchestrator(tmp_path):
    orch = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
    orch._run_output_dir = str(tmp_path)
    return orch


def _manifest(tmp_path):
    return OutputManifest(
        output_directory=str(tmp_path),
        total_scenes=2,
        successful_scenes=2,
        files=[],
        status="SUCCESS",
        timestamp="2026-07-03T00:00:00+00:00",
        scene_success_rate=1.0,
    )


def _video_output(tmp_path, results):
    return VideoRendererOutput(
        final_video_path=str(tmp_path / "final_output.mp4"),
        scene_results=results,
        total_duration=10.0,
    )


# ---------------------------------------------------------------------------
# Honest metrics
# ---------------------------------------------------------------------------

class TestDeliveryMetrics:
    def test_delivered_and_lost_populated(self, tmp_path):
        orch = _orchestrator(tmp_path)
        results = [
            SceneRenderResult("scene_001", True, str(tmp_path / "a.mp4"), None, 5.0),
            SceneRenderResult("scene_002", False, None, "manim exit code 1: boom"),
        ]
        manifest = orch._finalize_manifest(
            _manifest(tmp_path), _video_output(tmp_path, results),
            None, None, None,
        )
        assert manifest.delivered_scenes == 1
        assert manifest.lost_scenes == ["scene_002"]

    def test_render_skipped_leaves_delivery_unset(self, tmp_path):
        orch = _orchestrator(tmp_path)
        manifest = orch._finalize_manifest(_manifest(tmp_path), None, None, None, None)
        assert manifest.delivered_scenes is None
        assert manifest.lost_scenes == []


# ---------------------------------------------------------------------------
# Render salvage
# ---------------------------------------------------------------------------

class TestRenderSalvage:
    def test_failed_scene_regenerated_and_replaced(self, tmp_path, monkeypatch):
        orch = _orchestrator(tmp_path)
        storyboard = _full_storyboard()  # contains scene_001 with a full plan

        salvage_video = tmp_path / "media" / "scene_scene_001.mp4"
        salvage_video.parent.mkdir(parents=True, exist_ok=True)
        salvage_video.write_bytes(b"fake video")

        rendered_files = []

        def fake_render(scene_file, quality_flag, media_dir):
            rendered_files.append(scene_file)
            return SceneRenderResult("scene_001", True, str(salvage_video), None, 6.0)

        monkeypatch.setattr(orch.video_renderer_agent, "_render_scene", fake_render)

        failed = SceneRenderResult(
            "scene_001", False, None,
            "manim exit code 1: ValueError: latex error converting to dvi",
        )
        out = orch._salvage_failed_renders(
            _video_output(tmp_path, [failed]), storyboard,
        )

        assert out.scene_results[0].success is True
        assert out.scene_results[0].video_path == str(salvage_video)
        # The scene file was rewritten for the re-render (mock codegen path).
        assert rendered_files and rendered_files[0].endswith("scene_scene_001.py")
        assert (tmp_path / "scene_scene_001.py").exists()
        # Single salvaged scene → concat degenerates to a copy of that video.
        assert (tmp_path / "final_output.mp4").exists()

    def test_no_failures_is_a_noop(self, tmp_path, monkeypatch):
        orch = _orchestrator(tmp_path)
        called = []
        monkeypatch.setattr(
            orch.video_renderer_agent, "_render_scene",
            lambda *a, **k: called.append(1),
        )
        ok = SceneRenderResult("scene_001", True, str(tmp_path / "a.mp4"), None, 5.0)
        out = orch._salvage_failed_renders(
            _video_output(tmp_path, [ok]), _full_storyboard(),
        )
        assert called == []
        assert out.scene_results[0] is ok

    def test_unsalvageable_scene_stays_failed(self, tmp_path, monkeypatch):
        orch = _orchestrator(tmp_path)

        def still_failing(scene_file, quality_flag, media_dir):
            return SceneRenderResult("scene_001", False, None, "still broken")

        monkeypatch.setattr(orch.video_renderer_agent, "_render_scene", still_failing)
        failed = SceneRenderResult("scene_001", False, None, "manim exit code 1: x")
        out = orch._salvage_failed_renders(
            _video_output(tmp_path, [failed]), _full_storyboard(),
        )
        assert out.scene_results[0].success is False

    def test_unknown_scene_id_skipped(self, tmp_path):
        orch = _orchestrator(tmp_path)
        failed = SceneRenderResult("scene_999", False, None, "boom")
        out = orch._salvage_failed_renders(
            _video_output(tmp_path, [failed]), _full_storyboard(),
        )
        assert out.scene_results[0].success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
