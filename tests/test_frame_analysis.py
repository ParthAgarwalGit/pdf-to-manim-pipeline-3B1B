"""Tests for frame-level visual QA (ADR-005).

Pure-function tests use synthetic PIL images; reviewer-wiring tests
monkeypatch keyframe extraction so no ffmpeg or real video is needed.
"""

import numpy as np
import pytest
from PIL import Image

from src.agents.video_quality_reviewer import VideoQualityReviewerAgent
from src.utils.frame_analysis import (
    KeyframeReport,
    analyze_keyframes,
    mean_abs_diff,
    nonblank_fraction,
)


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------

def black_frame(size=(64, 64)) -> Image.Image:
    return Image.new("RGB", size, (0, 0, 0))


def bg_frame(size=(64, 64)) -> Image.Image:
    """3B1B dark background (#1C1C1C) — must count as blank."""
    return Image.new("RGB", size, (28, 28, 28))


def circle_frame(offset=0, size=(64, 64)) -> Image.Image:
    """Dark background with a bright square 'shape' whose position can vary."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = (28, 28, 28)
    x = 10 + offset
    arr[20:40, x:x + 20] = (88, 196, 221)  # 3B1B blue block
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

class TestFrameFunctions:
    def test_black_frame_has_zero_nonblank_fraction(self):
        assert nonblank_fraction(black_frame()) == 0.0

    def test_3b1b_background_counts_as_blank(self):
        assert nonblank_fraction(bg_frame()) == 0.0

    def test_drawn_shape_is_detected(self):
        frac = nonblank_fraction(circle_frame())
        assert frac > 0.05  # 20x20 block on 64x64 ≈ 0.098

    def test_mean_abs_diff_zero_for_identical_frames(self):
        f = circle_frame()
        assert mean_abs_diff(f, f) == 0.0

    def test_mean_abs_diff_positive_for_moved_shape(self):
        assert mean_abs_diff(circle_frame(0), circle_frame(20)) > 2.0

    def test_mismatched_sizes_are_maximally_different(self):
        assert mean_abs_diff(black_frame((64, 64)), black_frame((32, 32))) == 255.0


class TestAnalyzeKeyframes:
    def test_all_blank_frames_flagged_empty_and_mostly_empty(self):
        report = analyze_keyframes([bg_frame() for _ in range(4)])
        assert report.empty_frame_indices == [0, 1, 2, 3]
        assert report.mostly_empty is True

    def test_identical_drawn_frames_are_static_not_empty(self):
        f = circle_frame()
        report = analyze_keyframes([f, f, f, f])
        assert report.empty_frame_indices == []
        assert report.is_static is True
        assert report.mostly_empty is False

    def test_moving_shape_is_neither_empty_nor_static(self):
        frames = [circle_frame(offset=i * 10) for i in range(4)]
        report = analyze_keyframes(frames)
        assert report.empty_frame_indices == []
        assert report.is_static is False

    def test_single_frame_cannot_be_static(self):
        report = analyze_keyframes([circle_frame()])
        assert report.is_static is False

    def test_empty_input_yields_empty_report(self):
        report = analyze_keyframes([])
        assert report.frame_count == 0
        assert report.mostly_empty is False
        assert report.is_static is False


# ---------------------------------------------------------------------------
# Reviewer wiring
# ---------------------------------------------------------------------------

class _FakeSceneResult:
    def __init__(self, video_path: str, success: bool = True):
        self.success = success
        self.video_path = video_path


def _render_map(tmp_path, scene_id="scene_001"):
    video = tmp_path / f"{scene_id}.mp4"
    video.write_bytes(b"fake")
    return {scene_id: _FakeSceneResult(str(video))}


class TestReviewRenderedFrames:
    def test_blank_video_produces_critical_finding(self, tmp_path, monkeypatch):
        agent = VideoQualityReviewerAgent()
        monkeypatch.setattr(
            agent, "_extract_keyframes",
            lambda _path: [bg_frame() for _ in range(4)],
        )
        findings = agent._review_rendered_frames(_render_map(tmp_path))
        assert any(
            f.severity == "critical" and "empty" in f.issue.lower()
            for f in findings
        )
        assert findings[0].scene_id == "scene_001"
        assert findings[0].suspected_stage == "animation_generator"

    def test_static_video_produces_static_finding(self, tmp_path, monkeypatch):
        agent = VideoQualityReviewerAgent()
        f = circle_frame()
        monkeypatch.setattr(agent, "_extract_keyframes", lambda _p: [f, f, f, f])
        findings = agent._review_rendered_frames(_render_map(tmp_path))
        assert any("static" in fi.issue.lower() for fi in findings)

    def test_animated_video_produces_no_findings(self, tmp_path, monkeypatch):
        agent = VideoQualityReviewerAgent()
        monkeypatch.setattr(
            agent, "_extract_keyframes",
            lambda _p: [circle_frame(offset=i * 10) for i in range(4)],
        )
        assert agent._review_rendered_frames(_render_map(tmp_path)) == []

    def test_failed_or_missing_videos_are_skipped(self, tmp_path, monkeypatch):
        agent = VideoQualityReviewerAgent()
        called = []
        monkeypatch.setattr(
            agent, "_extract_keyframes",
            lambda p: called.append(p) or [],
        )
        render_map = {
            "scene_fail": _FakeSceneResult(str(tmp_path / "x.mp4"), success=False),
            "scene_gone": _FakeSceneResult(str(tmp_path / "missing.mp4")),
        }
        assert agent._review_rendered_frames(render_map) == []
        assert called == []  # neither scene reached extraction

    def test_extraction_failure_degrades_gracefully(self, tmp_path, monkeypatch):
        """No ffmpeg / broken file => no findings, no crash."""
        agent = VideoQualityReviewerAgent()
        monkeypatch.setattr(agent, "_get_media_duration", lambda _p: 5.0)
        monkeypatch.setattr(
            "src.agents.video_quality_reviewer.get_ffmpeg",
            lambda: (_ for _ in ()).throw(RuntimeError("ffmpeg not found")),
        )
        frames = agent._extract_keyframes(str(tmp_path / "whatever.mp4"))
        assert frames == []

    def test_zero_duration_video_skips_extraction(self, tmp_path, monkeypatch):
        agent = VideoQualityReviewerAgent()
        monkeypatch.setattr(agent, "_get_media_duration", lambda _p: -1.0)
        assert agent._extract_keyframes(str(tmp_path / "v.mp4")) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
