"""Explicit V2 repair, fallback, and renderer failure policy tests."""

import subprocess
from pathlib import Path

import pytest

from src.pipeline_v2.models import (
    ArtifactCategory,
    ArtifactSpec,
    BeatComposition,
    LayoutRegion,
    LayoutTemplate,
    NarratedScene,
    NarrationBeat,
    VerificationFinding,
    VisualContract,
)
from src.pipeline_v2.pipeline import (
    _repair_code_unchanged,
    _repair_improves,
    _repair_needed,
    _render_acceptable,
    _source_fallback_allowed,
)
from src.pipeline_v2.media import inspect_av, source_page_fallback
from src.pipeline_v2.rendering import (
    RenderOutcome,
    TrackedManimRenderer,
    _attach_issue_screenshots,
)
from src.pipeline_v2.report import _finding_html
from src.pipeline_v2.tts import _silence


FIXTURE = Path(__file__).parent / "fixtures" / "test_trig_3pages.pdf"


CODE = '''from manim import *
from src.pipeline_v2.tracked_scene import TrackedScene

class Demo(TrackedScene):
    def construct(self):
        self.set_layout("full_visual")
        dot = Dot()
        self.place_in_region(dot, "content_full")
        self.register_artifact("dot", dot, category="primary_visual", purpose="Show", beat_ids=["b1"])
        self.set_beat("b1")
        self.play(FadeIn(dot))
'''


def _scene_and_contract():
    scene = NarratedScene(
        scene_id="scene", narration="Show the dot.",
        beats=[NarrationBeat(beat_id="b1", text="Show", source_fact_ids=["f1"])],
    )
    contract = VisualContract(
        scene_id="scene", layout_template=LayoutTemplate.FULL_VISUAL,
        artifact_inventory=[ArtifactSpec(
            artifact_id="dot", category=ArtifactCategory.PRIMARY_VISUAL,
            role="dot", purpose="Show", source_fact_ids=["f1"], beat_ids=["b1"],
            enter_beat="b1", exit_beat="b1",
            region=LayoutRegion.CONTENT_FULL,
        )],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["dot"])],
    )
    return scene, contract


def test_noop_or_non_improving_repair_cannot_replace_original():
    original = RenderOutcome(
        safe=True, render_success=True,
        findings=[VerificationFinding(code="OVERLAP", severity="warning", message="x")],
    )
    worse = RenderOutcome(
        safe=True, render_success=False,
        findings=[VerificationFinding(code="FAILED", severity="error", message="x")],
    )
    better = RenderOutcome(safe=True, render_success=True, findings=[])
    assert _repair_code_unchanged(CODE, CODE)
    assert not _repair_improves(original, worse)
    assert _repair_improves(original, better)


def test_source_fallback_requires_standard_policy_and_audio():
    assert _source_fallback_allowed("standard", "narration.mp3")
    assert not _source_fallback_allowed("strict", "narration.mp3")
    assert not _source_fallback_allowed("standard", None)


def test_standard_retains_rendered_visual_errors_after_bounded_repair():
    outcome = RenderOutcome(
        safe=True,
        render_success=True,
        findings=[VerificationFinding(
            code="TEXT_TEXT_OVERLAP", severity="error", message="overlap",
        )],
    )

    assert _render_acceptable(outcome, "enforce", "standard")
    assert not _render_acceptable(outcome, "enforce", "strict")
    assert _render_acceptable(outcome, "audit")
    assert _repair_needed(outcome, "enforce")
    assert not _repair_needed(outcome, "audit")


def test_audit_mode_still_blocks_unsafe_or_failed_renders_and_allows_repair():
    unsafe = RenderOutcome(safe=False, render_success=False)
    failed = RenderOutcome(safe=True, render_success=False)

    assert not _render_acceptable(unsafe, "audit")
    assert not _render_acceptable(failed, "audit")
    assert _repair_needed(unsafe, "audit")
    assert _repair_needed(failed, "audit")


def test_advisory_warnings_do_not_spend_repairs_but_repeated_overlap_does():
    advisory = RenderOutcome(
        safe=True,
        render_success=True,
        findings=[
            VerificationFinding(
                code=f"ADVISORY_{index}", severity="warning", message="minor",
            )
            for index in range(6)
        ],
    )
    overlap = RenderOutcome(
        safe=True,
        render_success=True,
        findings=[
            VerificationFinding(
                code="TEXT_TEXT_OVERLAP", severity="warning", message=f"pair {index}",
            )
            for index in range(2)
        ],
    )

    assert not _repair_needed(advisory, "enforce")
    assert _repair_needed(overlap, "enforce")


def test_timestamped_finding_links_to_nearest_evidence_frame(tmp_path):
    first = tmp_path / "frame-000-1.000s.png"
    second = tmp_path / "frame-001-2.500s.png"
    first.touch()
    second.touch()
    finding = VerificationFinding(
        code="TEXT_TEXT_OVERLAP", severity="error", message="overlap", timestamp=2.3,
    )

    _attach_issue_screenshots([finding], [str(first), str(second)])

    assert finding.screenshot_path == str(second)


def test_report_finding_names_leaf_objects_and_links_evidence(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()
    finding = VerificationFinding(
        code="TEXT_TEXT_OVERLAP", severity="error", message="50% occupied pixels",
        timestamp=2.5, artifact_ids=["diagram"],
        object_ids=["diagram.0", "diagram.1"], screenshot_path=str(frame),
    )

    rendered = _finding_html(tmp_path, finding)

    assert "t=2.50s" in rendered
    assert "objects=diagram.0, diagram.1" in rendered
    assert 'href="frame.png"' in rendered


def test_tracked_renderer_timeout_is_structured(monkeypatch, tmp_path):
    scene, contract = _scene_and_contract()

    class TimedOutProcess:
        returncode = -9

        def __init__(self):
            self.killed = False

        def communicate(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("manim", timeout)
            return "partial stdout", "partial stderr"

        def kill(self):
            self.killed = True

        def terminate(self):
            self.killed = True

    monkeypatch.setattr(
        "src.pipeline_v2.rendering.subprocess.Popen",
        lambda *_args, **_kwargs: TimedOutProcess(),
    )
    outcome = TrackedManimRenderer().render(
        CODE, contract, scene, tmp_path / "attempt", timeout_seconds=10,
    )
    assert not outcome.render_success
    assert outcome.error == "Manim render timed out after 10s."
    assert {finding.code for finding in outcome.findings} == {"RENDER_TIMEOUT"}
    assert (tmp_path / "attempt" / "render_stdout.txt").exists()


@pytest.mark.integration
def test_source_page_fallback_is_labeled_and_capped_to_audio(tmp_path):
    audio = tmp_path / "narration.mp3"
    _silence(audio, 2.5)
    video = source_page_fallback(FIXTURE, 1, audio, tmp_path / "fallback.mp4")
    review = inspect_av(video)
    assert review["valid"]
    assert 2.4 <= review["duration_seconds"] <= 2.7
    assert video.with_suffix(".png").exists()
