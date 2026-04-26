"""Tests for pi_characters — role-aware deterministic Pi character helpers."""

import pytest

manim = pytest.importorskip("manim")  # skip entire module when Manim not installed

from src.utils.pi_characters import (
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
