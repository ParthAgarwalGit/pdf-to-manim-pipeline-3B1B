"""Tests for scene layout schemas."""

import pytest
from pydantic import ValidationError

from src.schemas.scene_layout import (
    SCREEN_X_MAX,
    SCREEN_X_MIN,
    SCREEN_Y_MAX,
    SCREEN_Y_MIN,
    BoundingCorner,
    CharacterBoundingBox,
    OverlapViolation,
    SceneLayout,
)


# ---------------------------------------------------------------------------
# BoundingCorner
# ---------------------------------------------------------------------------

def test_bounding_corner_valid():
    c = BoundingCorner(x=0.0, y=0.0, t=10.0)
    assert c.x == 0.0
    assert c.y == 0.0
    assert c.t == 10.0


def test_bounding_corner_at_screen_edge():
    c = BoundingCorner(x=SCREEN_X_MIN, y=SCREEN_Y_MIN, t=0.0)
    assert c.x == SCREEN_X_MIN
    assert c.y == SCREEN_Y_MIN


def test_bounding_corner_x_out_of_bounds():
    with pytest.raises(ValidationError):
        BoundingCorner(x=SCREEN_X_MAX + 0.1, y=0.0, t=5.0)


def test_bounding_corner_y_out_of_bounds():
    with pytest.raises(ValidationError):
        BoundingCorner(x=0.0, y=SCREEN_Y_MIN - 0.1, t=5.0)


def test_bounding_corner_negative_t():
    with pytest.raises(ValidationError):
        BoundingCorner(x=0.0, y=0.0, t=-1.0)


# ---------------------------------------------------------------------------
# CharacterBoundingBox helpers
# ---------------------------------------------------------------------------

def _make_bbox(
    object_id: str = "eq_0",
    object_type: str = "equation",
    x1: float = -2.0,
    x2: float = 2.0,
    y1: float = 1.0,
    y2: float = 0.0,
    t: float = 30.0,
) -> CharacterBoundingBox:
    return CharacterBoundingBox(
        object_id=object_id,
        object_type=object_type,
        top_left=BoundingCorner(x=x1, y=y1, t=t),
        top_right=BoundingCorner(x=x2, y=y1, t=t),
        bottom_left=BoundingCorner(x=x1, y=y2, t=t),
        bottom_right=BoundingCorner(x=x2, y=y2, t=t),
    )


def test_bbox_geometry_properties():
    bbox = _make_bbox(x1=-2.0, x2=2.0, y1=1.5, y2=0.5)
    assert bbox.x_min == -2.0
    assert bbox.x_max == 2.0
    assert bbox.y_min == 0.5
    assert bbox.y_max == 1.5
    assert bbox.width == pytest.approx(4.0)
    assert bbox.height == pytest.approx(1.0)
    assert bbox.center_x == pytest.approx(0.0)
    assert bbox.center_y == pytest.approx(1.0)


def test_bbox_duration_is_max_t():
    bbox = CharacterBoundingBox(
        object_id="obj",
        object_type="text",
        top_left=BoundingCorner(x=-1.0, y=1.0, t=10.0),
        top_right=BoundingCorner(x=1.0, y=1.0, t=20.0),
        bottom_left=BoundingCorner(x=-1.0, y=0.0, t=15.0),
        bottom_right=BoundingCorner(x=1.0, y=0.0, t=5.0),
    )
    assert bbox.duration == 20.0


def test_bbox_invalid_top_left_x_ge_top_right_x():
    with pytest.raises(ValidationError):
        _make_bbox(x1=2.0, x2=1.0)


def test_bbox_invalid_top_y_le_bottom_y():
    with pytest.raises(ValidationError):
        _make_bbox(y1=0.0, y2=1.0)


# ---------------------------------------------------------------------------
# SceneLayout
# ---------------------------------------------------------------------------

def test_scene_layout_valid():
    layout = SceneLayout(scene_id="intro_001", objects=[_make_bbox()])
    assert layout.scene_id == "intro_001"
    assert len(layout.objects) == 1


def test_scene_layout_empty_objects():
    layout = SceneLayout(scene_id="empty_scene")
    assert layout.objects == []


def test_scene_layout_empty_scene_id():
    with pytest.raises(ValidationError):
        SceneLayout(scene_id="  ")


# ---------------------------------------------------------------------------
# OverlapViolation
# ---------------------------------------------------------------------------

def test_overlap_violation_valid():
    v = OverlapViolation(
        object_id_a="eq_0",
        object_id_b="eq_1",
        overlap_x=0.5,
        overlap_y=0.3,
        overlap_area=0.15,
        shared_duration=30.0,
        description="test overlap",
    )
    assert v.overlap_area == pytest.approx(0.15)


def test_overlap_violation_negative_area():
    with pytest.raises(ValidationError):
        OverlapViolation(
            object_id_a="a",
            object_id_b="b",
            overlap_x=0.5,
            overlap_y=0.3,
            overlap_area=-0.1,
            shared_duration=10.0,
        )
