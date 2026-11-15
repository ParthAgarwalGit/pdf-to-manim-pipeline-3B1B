"""Tests for pi_characters — role-aware deterministic Pi character helpers."""

import pytest

manim = pytest.importorskip("manim")  # skip entire module when Manim not installed

from src.utils.pi_characters import (  # noqa: E402
    create_character_for_role,
    create_character_lineup,
)


def test_create_character_for_role_supports_teacher():
    teacher = create_character_for_role("teacher")
    assert teacher is not None
    assert teacher.__class__.__name__ == "VGroup"


def test_create_character_for_role_supports_students():
    left_student = create_character_for_role("student_left")
    center_student = create_character_for_role("student_center")
    right_student = create_character_for_role("student_right")
    assert left_student is not None
    assert center_student is not None
    assert right_student is not None


def test_create_character_for_role_unknown_falls_back_to_teacher_style():
    narrator = create_character_for_role("narrator_proxy")
    assert narrator is not None


def test_create_character_lineup_orders_left_to_right():
    # Even when given in reverse order, lineup must be sorted by ROLE_ORDER
    lineup = create_character_lineup(
        ["student_right", "teacher", "student_left"],
        spacing=1.2,
    )
    xs = [mob.get_center()[0] for mob in lineup.submobjects]
    # teacher (ROLE_ORDER 0) should be leftmost, student_right (3) rightmost
    assert xs == sorted(xs), f"Characters not sorted left-to-right: {xs}"


def test_create_character_lineup_single_role():
    lineup = create_character_lineup(["teacher"])
    assert len(lineup.submobjects) == 1


def test_create_character_for_role_custom_height():
    teacher = create_character_for_role("teacher", height=2.0)
    assert teacher is not None


# ---------------------------------------------------------------------------
# Phase 4 — blob mascot regressions
# ---------------------------------------------------------------------------

def test_mascot_contains_no_latex():
    """The blob is pure shapes — immune to the Tex render failures that broke
    13 scenes in a real run (and the old floating-eyes-over-a-glyph look)."""
    from manim import MathTex, Tex

    def flatten(mob):
        yield mob
        for sub in getattr(mob, "submobjects", []):
            yield from flatten(sub)

    from src.utils.pi_characters import create_pi_teacher
    assert not any(
        isinstance(m, (MathTex, Tex)) for m in flatten(create_pi_teacher())
    )


def test_mascot_emotions_produce_distinct_mouths():
    from src.utils.pi_characters import _make_blob
    from manim import BLUE_B, Arc, Line
    happy = _make_blob(BLUE_B, emotion="happy")
    neutral = _make_blob(BLUE_B, emotion="neutral")
    assert isinstance(happy[2], Arc)
    assert isinstance(neutral[2], Line)


def test_teacher_taller_than_student():
    from src.utils.pi_characters import create_character_for_role
    teacher = create_character_for_role("teacher")
    student = create_character_for_role("student_left")
    assert teacher.height > student.height


def test_speech_bubble_stays_in_frame():
    from src.utils.pi_characters import create_pi_teacher, create_speech_bubble
    from manim import RIGHT, UP
    speaker = create_pi_teacher().move_to(RIGHT * 6.5 + UP * 3.5)  # corner case
    bubble = create_speech_bubble("A fairly long insight that would overflow", speaker)
    assert bubble.get_right()[0] <= 6.95
    assert bubble.get_top()[1] <= 3.85


def test_codegen_prompt_uses_mascot_block(monkeypatch):
    from tests.test_animation_generator import (
        _capture_animation_prompt,
        build_test_context,
        build_test_scene_spec,
    )
    from src.agents.animation_generator import AnimationCodeGeneratorAgent
    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch,
        build_test_scene_spec(), build_test_context(),
    )
    assert "NARRATOR MASCOT" in captured["system"]
    assert "At most ONE mascot appearance" in captured["system"]


def test_visual_intent_contract_gates_mascot():
    from src.utils.manim_engine_contract import build_visual_intent_contract_prompt
    contract = build_visual_intent_contract_prompt()
    assert "at most ONE per scene" in contract
    assert "NEVER in a" in contract
