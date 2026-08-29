from src.manim_verifier.models import FrameExtractionResult, VerificationIssue, VerificationReport
from src.manim_verifier.pipeline import RawManimVerifierPipeline, VerifierConfig
from src.manim_verifier.vision_reviewer import MockVisionReviewer


ORIGINAL = "from manim import *\nclass Original(Scene):\n    def construct(self):\n        pass\n"
REPAIRED = "from manim import *\nclass Repaired(Scene):\n    def construct(self):\n        self.add(Circle())\n"


class FakeRenderer:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = []

    def render(self, source, attempt_dir, **kwargs):
        self.calls.append(source)
        report = self.reports.pop(0)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "scene.py").write_text(source, encoding="utf-8")
        if report.render_success:
            video = attempt_dir / "video.mp4"
            video.write_bytes(b"video")
            report.video_path = str(video.resolve())
        return report


class FakeFrames:
    def extract(self, video_path, output_dir):
        return FrameExtractionResult()


class ArtifactFrames:
    def extract(self, video_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = output_dir / "frame.png"
        frame.write_bytes(b"png")
        return FrameExtractionResult(frame_paths=[str(frame.resolve())], timestamps=[1.0])


def _config(tmp_path, mode="repair", repairs=1):
    return VerifierConfig(mode=mode, run_root=tmp_path, max_repair_attempts=repairs)


def test_repair_is_rechecked_and_selected(tmp_path):
    renderer = FakeRenderer([
        VerificationReport(
            safe=True,
            syntax_valid=True,
            render_success=False,
            issues=[VerificationIssue(code="FAIL", severity="error", message="bad")],
        ),
        VerificationReport(safe=True, syntax_valid=True, render_success=True),
    ])
    reviewer = MockVisionReviewer(REPAIRED)
    pipeline = RawManimVerifierPipeline(
        reviewer=reviewer,
        config=_config(tmp_path),
        renderer=renderer,
        frame_extractor=FakeFrames(),
    )
    result = pipeline.run(brief="b", storyboard="s", initial_code=ORIGINAL)
    assert result.success
    assert result.repaired
    assert result.final_code.strip() == REPAIRED.strip()
    assert len(result.attempts) == 2
    assert reviewer.calls[0]["code"] == ORIGINAL


def test_reviewer_receives_report_and_extracted_frame_artifacts(tmp_path):
    original_report = VerificationReport(
        safe=True,
        syntax_valid=True,
        render_success=True,
        issues=[VerificationIssue(code="STATIC", severity="warning", message="still")],
    )
    repaired_report = VerificationReport(safe=True, syntax_valid=True, render_success=True)
    reviewer = MockVisionReviewer(REPAIRED)
    pipeline = RawManimVerifierPipeline(
        reviewer=reviewer,
        config=_config(tmp_path),
        renderer=FakeRenderer([original_report, repaired_report]),
        frame_extractor=ArtifactFrames(),
    )
    pipeline._visual_diagnostics = lambda paths: []
    pipeline.run(brief="brief", storyboard="storyboard", initial_code=ORIGINAL)
    call = reviewer.calls[0]
    assert call["brief"] == "brief"
    assert call["storyboard"] == "storyboard"
    assert call["report"] is original_report
    assert len(call["frame_paths"]) == 1
    assert call["frame_paths"][0].exists()


def test_failed_repair_falls_back_to_rendered_original(tmp_path):
    renderer = FakeRenderer([
        VerificationReport(
            safe=True,
            syntax_valid=True,
            render_success=True,
            issues=[VerificationIssue(code="STATIC", severity="warning", message="still")],
        ),
        VerificationReport(
            safe=False,
            syntax_valid=False,
            render_success=False,
            issues=[VerificationIssue(code="UNSAFE", severity="error", message="unsafe")],
        ),
    ])
    pipeline = RawManimVerifierPipeline(
        reviewer=MockVisionReviewer("exec('bad')"),
        config=_config(tmp_path),
        renderer=renderer,
        frame_extractor=FakeFrames(),
    )
    result = pipeline.run(brief="b", storyboard="s", initial_code=ORIGINAL)
    assert result.success
    assert result.fallback_used
    assert not result.repaired
    assert result.final_code == ORIGINAL


def test_shadow_mode_never_calls_reviewer(tmp_path):
    renderer = FakeRenderer([
        VerificationReport(safe=True, syntax_valid=True, render_success=True),
    ])
    pipeline = RawManimVerifierPipeline(
        config=_config(tmp_path, mode="shadow"),
        renderer=renderer,
        frame_extractor=FakeFrames(),
    )
    result = pipeline.run(brief="b", storyboard="s", initial_code=ORIGINAL)
    assert result.success
    assert len(result.attempts) == 1


def test_repair_loop_respects_configured_maximum(tmp_path):
    def failure():
        return VerificationReport(
            safe=True,
            syntax_valid=True,
            render_success=False,
            issues=[VerificationIssue(code="FAIL", severity="error", message="bad")],
        )
    reviewer = MockVisionReviewer(REPAIRED)
    renderer = FakeRenderer([failure(), failure(), failure()])
    pipeline = RawManimVerifierPipeline(
        reviewer=reviewer,
        config=_config(tmp_path, repairs=2),
        renderer=renderer,
        frame_extractor=FakeFrames(),
    )
    result = pipeline.run(brief="b", storyboard="s", initial_code=ORIGINAL)
    assert len(result.attempts) == 3
    assert len(reviewer.calls) == 2
