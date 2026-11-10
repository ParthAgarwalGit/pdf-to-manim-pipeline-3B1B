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


def test_advisory_warning_alone_does_not_trigger_paid_repair(tmp_path):
    reviewer = MockVisionReviewer(REPAIRED)
    renderer = FakeRenderer([
        VerificationReport(
            safe=True,
            syntax_valid=True,
            render_success=True,
            issues=[VerificationIssue(
                code="RAW_COORDINATE_WITH_AXES", severity="warning", message="lint",
            )],
        ),
    ])
    pipeline = RawManimVerifierPipeline(
        reviewer=reviewer,
        config=_config(tmp_path),
        renderer=renderer,
        frame_extractor=FakeFrames(),
    )
    result = pipeline.run(brief="b", storyboard="s", initial_code=ORIGINAL)
    assert result.success
    assert reviewer.calls == []
    assert result.final_code == ORIGINAL


def test_reviewer_provider_error_keeps_original_and_does_not_raise(tmp_path):
    class FailingReviewer(MockVisionReviewer):
        async def repair(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("429 Too Many Requests")

    reviewer = FailingReviewer(REPAIRED)
    renderer = FakeRenderer([
        VerificationReport(
            safe=True,
            syntax_valid=True,
            render_success=True,
            issues=[VerificationIssue(code="STATIC", severity="warning", message="still")],
        ),
    ])
    pipeline = RawManimVerifierPipeline(
        reviewer=reviewer,
        config=_config(tmp_path, repairs=2),
        renderer=renderer,
        frame_extractor=FakeFrames(),
    )
    result = pipeline.run(brief="b", storyboard="s", initial_code=ORIGINAL)
    assert result.success
    assert result.final_code == ORIGINAL
    assert len(reviewer.calls) == 1  # stops after the first provider error
    assert any(
        issue.code == "REPAIR_REQUEST_FAILED"
        for attempt in result.attempts
        for issue in attempt.report.issues
    )


def test_repair_key_resolves_from_environment_only_when_allowed(tmp_path, monkeypatch):
    import pytest

    from src.manim_verifier.vision_reviewer import GeminiVisionReviewer

    missing = tmp_path / "no.env"
    monkeypatch.setenv("GEMINI_API_KEY", "from-environ")
    assert GeminiVisionReviewer.resolve_api_key(missing, allow_environ=True) == "from-environ"
    with pytest.raises(ValueError):
        GeminiVisionReviewer.resolve_api_key(missing, allow_environ=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    assert GeminiVisionReviewer.resolve_api_key(env_file, allow_environ=True) == "from-file"


def test_repair_mode_without_key_aborts_before_any_stage(tmp_path, monkeypatch):
    import pytest

    from src.manim_verifier.vision_reviewer import GeminiVisionReviewer
    from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig

    def no_key(*args, **kwargs):
        raise ValueError("GEMINI_API_KEY was not found")

    monkeypatch.setattr(GeminiVisionReviewer, "resolve_api_key", staticmethod(no_key))
    orchestrator = Orchestrator(PipelineConfig(
        base_output_dir=str(tmp_path), verifier_mode="repair",
    ))
    stages = []
    monkeypatch.setattr(
        orchestrator, "_run_ingestion",
        lambda *a, **kw: stages.append("ingestion"),
    )
    with pytest.raises(PipelineAbortError) as excinfo:
        orchestrator.run_pipeline(str(tmp_path / "doc.pdf"), event_source="local")
    assert excinfo.value.error_code == "VERIFIER_REVIEWER_UNAVAILABLE"
    assert stages == []
