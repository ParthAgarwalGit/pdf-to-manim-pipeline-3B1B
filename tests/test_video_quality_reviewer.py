"""Tests for VideoQualityReviewerAgent — heuristic quality gate.

Covers:
- Audio/video duration mismatch detection
- Overlap-style blame attribution to animation_generator
- Finding structure (suspected_stage, scene_id, code_hint)
- Pass/fail severity logic
"""

from dataclasses import dataclass
from typing import List, Optional


from src.agents.video_quality_reviewer import (
    VideoQualityReviewerAgent,
    VideoQualityReviewerInput,
)
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.output_manifest import OutputManifest
from src.schemas.storyboard import Scene, StoryboardJSON, Configuration, PDFMetadataStoryboard, VisualIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storyboard(scenes: List[Scene]) -> StoryboardJSON:
    return StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(filename="test.pdf", page_count=1),
        configuration=Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
        scenes=scenes,
    )


def _make_scene(scene_id: str, duration: float = 10.0, index: int = 0) -> Scene:
    return Scene(
        scene_id=scene_id,
        scene_index=index,
        concept_id=f"concept_{scene_id}",
        narration="Test narration text.",
        visual_intent=VisualIntent(mathematical_objects=[], transformations=[]),
        duration_estimate=duration,
    )


def _make_manifest(tmp_path) -> OutputManifest:
    from datetime import datetime, timezone
    return OutputManifest(
        output_directory=str(tmp_path),
        files=[],
        timestamp=datetime.now(timezone.utc),
        status="SUCCESS",
        scene_success_rate=1.0,
        total_scenes=1,
        successful_scenes=1,
        failed_scenes=[],
    )


@dataclass
class FakeRenderResult:
    scene_id: str
    success: bool
    video_path: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class FakeVideoOutput:
    scene_results: List[FakeRenderResult]
    final_video_path: Optional[str] = None


@dataclass
class FakeTTSOutput:
    scene_id: str
    audio_file_path: str = "/tmp/fake.mp3"
    duration_seconds: float = 0.0


@dataclass
class FakeCompositionResult:
    scene_id: str
    success: bool
    composed_video_path: Optional[str] = None
    final_duration: float = 0.0
    error: Optional[str] = None


@dataclass
class FakeCompositionOutput:
    scene_results: List[FakeCompositionResult]
    final_video_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDurationMismatch:
    """Reviewer must flag audio/video duration mismatch per scene."""

    def test_video_reviewer_flags_audio_video_duration_mismatch(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_001", duration=10.0)
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        video_output = FakeVideoOutput(
            scene_results=[FakeRenderResult("scene_001", success=True, duration_seconds=15.0)],
        )
        tts_output = [FakeTTSOutput("scene_001", duration_seconds=8.0)]

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            video_output=video_output,
            tts_output=tts_output,
        )
        result = agent.execute(input_data)

        duration_findings = [
            f for f in result.findings if "duration" in f.issue.lower() or "sync" in f.issue.lower()
        ]
        assert len(duration_findings) >= 1, "Should flag duration mismatch"
        finding = duration_findings[0]
        assert finding.scene_id == "scene_001"
        assert finding.suspected_stage == "animation_generator"
        assert finding.code_hint is not None, "Should include a code hint for pacing fix"

    def test_no_finding_when_durations_close(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_001", duration=10.0)
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        video_output = FakeVideoOutput(
            scene_results=[FakeRenderResult("scene_001", success=True, duration_seconds=10.5)],
        )
        tts_output = [FakeTTSOutput("scene_001", duration_seconds=10.0)]

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            video_output=video_output,
            tts_output=tts_output,
        )
        result = agent.execute(input_data)
        duration_findings = [
            f for f in result.findings if "duration" in f.issue.lower() or "sync" in f.issue.lower()
        ]
        assert len(duration_findings) == 0

    def test_boundary_at_tolerance_threshold(self, tmp_path):
        """Drift of exactly DURATION_TOLERANCE_SECONDS (2.0) should NOT flag; 2.01 should."""
        agent = VideoQualityReviewerAgent()

        for drift, expect_flag in [(2.0, False), (2.01, True)]:
            scene = _make_scene("scene_boundary", duration=10.0)
            storyboard = _make_storyboard([scene])
            manifest = _make_manifest(tmp_path)

            video_output = FakeVideoOutput(
                scene_results=[FakeRenderResult("scene_boundary", success=True, duration_seconds=10.0)],
            )
            tts_output = [FakeTTSOutput("scene_boundary", duration_seconds=10.0 + drift)]

            input_data = VideoQualityReviewerInput(
                storyboard=storyboard,
                manifest=manifest,
                video_output=video_output,
                tts_output=tts_output,
            )
            result = agent.execute(input_data)
            duration_findings = [
                f for f in result.findings if "duration" in f.issue.lower() or "sync" in f.issue.lower()
            ]
            if expect_flag:
                assert len(duration_findings) >= 1, f"drift={drift} should flag"
            else:
                assert len(duration_findings) == 0, f"drift={drift} should NOT flag"


class TestOverlapBlameAttribution:
    """Reviewer must blame animation_generator for overlap-style layout issues."""

    def test_video_reviewer_blames_scene_generation_for_overlap_style_issues(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_002")
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        # Code uses raw .next_to() without LayoutTracker
        code_obj = AnimationCodeObject(
            scene_id="scene_002",
            code='label = Text("x"); label.next_to(triangle, RIGHT)',
            class_name="TestScene",
            imports=["from manim import *"],
            narration="test",
            duration_estimate=10.0,
            generation_timestamp="2024-01-01T00:00:00Z",
        )

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code_obj],
            video_output=FakeVideoOutput(scene_results=[]),
            tts_output=[],
        )
        result = agent.execute(input_data)

        overlap_findings = [
            f for f in result.findings if "overlap" in f.issue.lower() or "layout" in f.issue.lower()
        ]
        assert len(overlap_findings) >= 1
        finding = overlap_findings[0]
        assert finding.suspected_stage == "animation_generator"
        assert finding.scene_id == "scene_002"
        assert "LayoutTracker" in (finding.code_hint or "")

    def test_no_overlap_finding_when_layout_tracker_used(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_003")
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        code_obj = AnimationCodeObject(
            scene_id="scene_003",
            code='layout.place_label(label, "lbl", anchor="tri", preferred=RIGHT)',
            class_name="TestScene",
            imports=["from manim import *"],
            narration="test",
            duration_estimate=10.0,
            generation_timestamp="2024-01-01T00:00:00Z",
        )

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code_obj],
            video_output=FakeVideoOutput(scene_results=[]),
            tts_output=[],
        )
        result = agent.execute(input_data)
        overlap_findings = [
            f for f in result.findings if "overlap" in f.issue.lower() or "layout" in f.issue.lower()
        ]
        assert len(overlap_findings) == 0


class TestFindingStructure:
    """Every finding must have the required diagnostic fields."""

    def test_findings_have_required_fields(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_004")
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        video_output = FakeVideoOutput(
            scene_results=[FakeRenderResult("scene_004", success=False, error="Manim crash")],
        )

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            video_output=video_output,
            tts_output=[],
        )
        result = agent.execute(input_data)

        for finding in result.findings:
            assert finding.suspected_stage, "suspected_stage must be set"
            assert finding.issue, "issue description must be set"
            assert finding.severity in ("low", "medium", "high", "critical")
            assert finding.suggested_fix, "suggested_fix must be set"
            assert finding.llm_direction, "llm_direction must be set"


class TestPassFailSeverity:
    """Review passes only when no critical/high findings exist."""

    def test_passes_when_only_medium_findings(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_005")
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        # Create a fake final video so the "no final video" critical finding is avoided
        final_video = tmp_path / "final.mp4"
        final_video.write_bytes(b"\x00" * 100)

        # Only overlap warning (medium severity)
        code_obj = AnimationCodeObject(
            scene_id="scene_005",
            code='label.next_to(obj, RIGHT)',
            class_name="TestScene",
            imports=["from manim import *"],
            narration="test",
            duration_estimate=10.0,
            generation_timestamp="2024-01-01T00:00:00Z",
        )

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code_obj],
            video_output=FakeVideoOutput(
                scene_results=[FakeRenderResult("scene_005", success=True, duration_seconds=10.0)],
                final_video_path=str(final_video),
            ),
            tts_output=[FakeTTSOutput("scene_005", duration_seconds=10.0)],
        )
        result = agent.execute(input_data)
        assert result.passed is True

    def test_fails_when_critical_finding(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_006")
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        # No final video = critical
        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            video_output=FakeVideoOutput(scene_results=[]),
            tts_output=[],
        )
        result = agent.execute(input_data)
        assert result.passed is False
        critical = [f for f in result.findings if f.severity == "critical"]
        assert len(critical) >= 1

    def test_no_spurious_critical_when_final_video_not_expected(self, tmp_path):
        """Reviewer must NOT fire 'no final video' when final_video_expected=False.

        Reproduces the --skip-tts false-positive: render succeeds but composition
        is intentionally skipped, so there is no composed final video.  The review
        should still run (checking per-scene render failures and layout risk) but
        must not mark the run as failed just because there is no final video.
        """
        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_007")
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            video_output=FakeVideoOutput(
                scene_results=[FakeRenderResult("scene_007", success=True, duration_seconds=10.0)],
            ),
            tts_output=[],
            composition_output=None,
            final_video_expected=False,   # TTS disabled — no composed video expected
        )
        result = agent.execute(input_data)

        # No "no final video" critical finding
        critical_missing_video = [
            f for f in result.findings
            if f.severity == "critical" and "no final video" in f.issue.lower()
        ]
        assert len(critical_missing_video) == 0, (
            "Reviewer must not fire 'no final video' when final_video_expected=False"
        )
        # Review as a whole should pass (all scenes rendered OK, no layout risk)
        assert result.passed is True


# ============================================================================
# Task 6 — Helper-semantic review findings
# ============================================================================

class TestHelperSemantics:
    """Reviewer must flag code that violates helper-backed visual semantics."""

    def test_flags_axes_redraw_when_helper_should_be_reused(self, tmp_path):
        """Reviewer flags code that uses Axes() when coordinate_systems is in scene_state_in."""
        from src.schemas.animation_code import AnimationCodeObject
        from datetime import datetime

        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_001")
        # Signal that a coordinate system already exists for this scene
        scene.visual_intent.scene_state_in.coordinate_systems = {"axes_main": "axes"}
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        code_obj = AnimationCodeObject(
            scene_id="scene_001",
            code="axes_main = Axes()\nself.play(Create(axes_main), run_time=1.0)",
            imports=["from manim import *"],
            class_name="Scene001",
            narration="Reuse the same axes from before.",
            duration_estimate=10.0,
            generation_timestamp=datetime.now(),
        )

        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code_obj],
            final_video_expected=False,
        ))
        cs_findings = [f for f in result.findings if "coordinate system" in f.issue.lower()]
        assert len(cs_findings) >= 1, "Should flag missing create_coordinate_system"
        assert cs_findings[0].suspected_stage == "animation_generator"
        assert cs_findings[0].severity == "medium"

    def test_flags_missing_notation_helper_when_notation_key_exported(self, tmp_path):
        """Reviewer flags code using bare MathTex when notation_bindings is in scene_state_out."""
        from src.schemas.animation_code import AnimationCodeObject
        from datetime import datetime

        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_002")
        scene.visual_intent.scene_state_out.notation_bindings = {"svd_core": "eq_main"}
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        code_obj = AnimationCodeObject(
            scene_id="scene_002",
            code=r"eq_main = MathTex(r'X = U\Sigma V^T')\nself.play(Write(eq_main), run_time=1.0)",
            imports=["from manim import *"],
            class_name="Scene002",
            narration="We keep the same notation.",
            duration_estimate=12.0,
            generation_timestamp=datetime.now(),
        )

        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code_obj],
            final_video_expected=False,
        ))
        notation_findings = [f for f in result.findings if "notation" in f.issue.lower()]
        assert len(notation_findings) >= 1, "Should flag missing create_notation"

    def test_flags_missing_character_helper_when_role_exported(self, tmp_path):
        """Reviewer flags code missing create_character_for_role when character_roles exported."""
        from src.schemas.animation_code import AnimationCodeObject
        from datetime import datetime

        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_003")
        scene.visual_intent.scene_state_out.character_roles = {"teacher": "teacher_obj"}
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        code_obj = AnimationCodeObject(
            scene_id="scene_003",
            code="teacher_obj = create_pi_teacher()\nself.play(FadeIn(teacher_obj), run_time=1.0)",
            imports=["from manim import *"],
            class_name="Scene003",
            narration="Introducing the teacher.",
            duration_estimate=10.0,
            generation_timestamp=datetime.now(),
        )

        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code_obj],
            final_video_expected=False,
        ))
        char_findings = [f for f in result.findings if "character" in f.issue.lower()]
        assert len(char_findings) >= 1, "Should flag missing create_character_for_role"

    def test_no_helper_finding_when_helper_is_used(self, tmp_path):
        """Reviewer must not fire helper findings when helper calls are present."""
        from src.schemas.animation_code import AnimationCodeObject
        from datetime import datetime

        agent = VideoQualityReviewerAgent()
        scene = _make_scene("scene_004")
        scene.visual_intent.scene_state_in.coordinate_systems = {"axes_main": "axes"}
        scene.visual_intent.scene_state_out.notation_bindings = {"eq_core": "eq_obj"}
        scene.visual_intent.scene_state_out.character_roles = {"teacher": "t_obj"}
        storyboard = _make_storyboard([scene])
        manifest = _make_manifest(tmp_path)

        code_obj = AnimationCodeObject(
            scene_id="scene_004",
            code=(
                "axes_main = create_coordinate_system(kind='axes')\n"
                "eq_obj = create_notation(tex=r'x^2', notation_key='eq_core')\n"
                "t_obj = create_character_for_role('teacher')\n"
                "self.play(Create(axes_main), run_time=1.0)"
            ),
            imports=["from manim import *"],
            class_name="Scene004",
            narration="All helpers used correctly.",
            duration_estimate=10.0,
            generation_timestamp=datetime.now(),
        )

        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code_obj],
            final_video_expected=False,
        ))
        helper_findings = [
            f for f in result.findings
            if any(kw in f.issue.lower() for kw in ("coordinate system", "notation", "character"))
        ]
        assert len(helper_findings) == 0, f"Should not flag correct helper use: {helper_findings}"


# ---------------------------------------------------------------------------
# Helper for Part 3 tests
# ---------------------------------------------------------------------------

def _make_code_obj(scene_id: str, code: str) -> AnimationCodeObject:
    from datetime import datetime
    return AnimationCodeObject(
        scene_id=scene_id,
        code=code,
        imports=["from manim import *"],
        class_name=f"Scene_{scene_id}",
        narration="Test narration.",
        duration_estimate=10.0,
        generation_timestamp=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Part 3 Task 5: Pacing & Timing Tests
# ---------------------------------------------------------------------------

class TestAnimationPacingReview:
    def test_flags_too_fast_write(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1", 'self.play(Write(eq), run_time=0.2)\nself.wait(1.0)\n')
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        timing_findings = [f for f in result.findings if "timing violation" in f.issue.lower()]
        assert len(timing_findings) >= 1

    def test_passes_correct_timing(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1", 'self.play(Write(eq), run_time=2.0)\nself.wait(1.0)\nself.play(FadeIn(label), run_time=1.0)\nself.wait(0.5)\n')
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        timing_findings = [f for f in result.findings if "timing violation" in f.issue.lower()]
        assert len(timing_findings) == 0


# ---------------------------------------------------------------------------
# Part 3 Task 6: Progressive Reveal Tests
# ---------------------------------------------------------------------------

class TestProgressiveRevealReview:
    def test_flags_complex_equation_in_one_shot(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1", 'eq = MathTex(r"a + b + c + d = e + f")\nself.play(Write(eq))\nself.wait(1.0)\n')
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        reveal_findings = [f for f in result.findings if "progressive" in f.issue.lower() or "one shot" in f.issue.lower()]
        assert len(reveal_findings) >= 1

    def test_passes_simple_equation(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1", 'eq = MathTex(r"a = b")\nself.play(Write(eq))\nself.wait(1.0)\n')
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        reveal_findings = [f for f in result.findings if "progressive" in f.issue.lower() or "one shot" in f.issue.lower()]
        assert len(reveal_findings) == 0


# ---------------------------------------------------------------------------
# Part 3 Task 8: Anti-Pattern Detection Tests
# ---------------------------------------------------------------------------

class TestAntiPatternReview:
    def test_flags_fadeout_fadein_pair(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1", 'self.play(FadeOut(old_eq))\nself.play(FadeIn(new_eq))\nself.wait(1.0)\n')
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        ap_findings = [f for f in result.findings if "FadeOut" in f.issue and "FadeIn" in f.issue]
        assert len(ap_findings) >= 1

    def test_flags_systematic_missing_waits_once(self, tmp_path):
        """Vignette pacing: a single back-to-back play pair is fine; a
        SYSTEMATIC absence of waits (>=3 gaps) yields ONE aggregated low
        finding — not one per gap (a real run produced 180 of those)."""
        agent = VideoQualityReviewerAgent()
        no_wait_run = "\n".join(
            f"self.play(Write(eq{i}), run_time=2.0)" for i in range(5)
        ) + "\nself.wait(1.0)\n"
        code = _make_code_obj("s1", no_wait_run)
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        wait_findings = [f for f in result.findings if "breathing room" in f.issue.lower()]
        assert len(wait_findings) == 1
        assert wait_findings[0].severity == "low"

    def test_single_back_to_back_play_pair_not_flagged(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1", 'self.play(Write(eq1), run_time=2.0)\nself.play(Write(eq2), run_time=2.0)\nself.wait(1.0)\n')
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        wait_findings = [f for f in result.findings if "breathing room" in f.issue.lower()]
        assert wait_findings == []

    def test_flags_text_wall(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1",
            't1 = Text("One")\nt2 = Text("Two")\nt3 = Text("Three")\n'
            't4 = Text("Four")\nself.play(Write(t1))\nself.wait(0.5)\n'
            'self.play(Write(t2))\nself.wait(0.5)\nself.play(Write(t3))\n'
            'self.wait(0.5)\nself.play(Write(t4))\nself.wait(1.0)\n'
        )
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        wall_findings = [f for f in result.findings if "text wall" in f.issue.lower()]
        assert len(wall_findings) >= 1

    def test_passes_clean_code(self, tmp_path):
        agent = VideoQualityReviewerAgent()
        code = _make_code_obj("s1",
            'eq = MathTex(r"a = b")\n'
            'self.play(Write(eq), run_time=2.0)\n'
            'self.wait(1.0)\n'
            'eq2 = MathTex(r"b = c")\n'
            'self.play(ReplacementTransform(eq, eq2), run_time=1.5)\n'
            'self.wait(1.0)\n'
        )
        storyboard = _make_storyboard([_make_scene("s1")])
        manifest = _make_manifest(tmp_path)
        result = agent.execute(VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=[code],
            final_video_expected=False,
        ))
        anti_pattern_findings = [
            f for f in result.findings
            if "FadeOut" in f.issue or "text wall" in f.issue.lower()
        ]
        assert len(anti_pattern_findings) == 0
