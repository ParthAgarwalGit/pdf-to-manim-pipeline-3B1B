"""Tests for LayoutValidatorAgent."""

import pytest

from src.agents.layout_validator import LayoutValidationInput, LayoutValidatorAgent
from src.agents.base import AgentExecutionError
from src.schemas.scene_layout import (
    BoundingCorner,
    CharacterBoundingBox,
    SceneLayout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corner(x: float, y: float, t: float = 30.0) -> BoundingCorner:
    return BoundingCorner(x=x, y=y, t=t)


def _bbox(
    object_id: str,
    x1: float, x2: float,
    y1: float, y2: float,
    t: float = 30.0,
    object_type: str = "equation",
) -> CharacterBoundingBox:
    return CharacterBoundingBox(
        object_id=object_id,
        object_type=object_type,
        top_left=_corner(x1, y1, t),
        top_right=_corner(x2, y1, t),
        bottom_left=_corner(x1, y2, t),
        bottom_right=_corner(x2, y2, t),
    )


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------

def test_validator_rejects_wrong_input_type():
    agent = LayoutValidatorAgent()
    assert not agent.validate_input("not_a_layout_input")


def test_validator_valid_input():
    agent = LayoutValidatorAgent()
    layout = SceneLayout(scene_id="s1", objects=[])
    assert agent.validate_input(LayoutValidationInput(scene_layout=layout))


# ---------------------------------------------------------------------------
# No-overlap cases
# ---------------------------------------------------------------------------

def test_no_objects_is_valid():
    agent = LayoutValidatorAgent()
    layout = SceneLayout(scene_id="s1", objects=[])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid
    assert result.violations == []


def test_single_object_is_valid():
    agent = LayoutValidatorAgent()
    layout = SceneLayout(scene_id="s1", objects=[_bbox("a", -3.0, -0.5, 0.5, -0.5)])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid


def test_side_by_side_objects_no_overlap():
    # a occupies x ∈ [-3, -0.2], b occupies x ∈ [0.2, 3] — no horizontal overlap
    agent = LayoutValidatorAgent()
    layout = SceneLayout(
        scene_id="s1",
        objects=[
            _bbox("a", -3.0, -0.2, 1.0, 0.0),
            _bbox("b",  0.2,  3.0, 1.0, 0.0),
        ],
    )
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid


def test_stacked_objects_no_overlap():
    # a on top (y ∈ [1.5, 2.5]), b below (y ∈ [-1.5, -0.5]) — no vertical overlap
    agent = LayoutValidatorAgent()
    layout = SceneLayout(
        scene_id="s1",
        objects=[
            _bbox("a", -2.0, 2.0, 2.5, 1.5),
            _bbox("b", -2.0, 2.0, -0.5, -1.5),
        ],
    )
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid


def test_touching_edges_no_overlap():
    # a right edge == b left edge → overlap_x == 0 → no violation
    agent = LayoutValidatorAgent()
    layout = SceneLayout(
        scene_id="s1",
        objects=[
            _bbox("a", -2.0, 0.0, 1.0, 0.0),
            _bbox("b",  0.0, 2.0, 1.0, 0.0),
        ],
    )
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid


# ---------------------------------------------------------------------------
# Overlap cases
# ---------------------------------------------------------------------------

def test_identical_boxes_overlap():
    agent = LayoutValidatorAgent()
    bbox = _bbox("a", -1.0, 1.0, 1.0, -1.0)
    bbox2 = _bbox("b", -1.0, 1.0, 1.0, -1.0)
    layout = SceneLayout(scene_id="s1", objects=[bbox, bbox2])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert not result.is_valid
    assert len(result.violations) == 1
    v = result.violations[0]
    assert v.object_id_a == "a"
    assert v.object_id_b == "b"
    assert v.overlap_x == pytest.approx(2.0)
    assert v.overlap_y == pytest.approx(2.0)
    assert v.overlap_area == pytest.approx(4.0)


def test_partial_overlap():
    agent = LayoutValidatorAgent()
    # a: x ∈ [-2, 1], b: x ∈ [0, 3] → x_overlap = 1
    # a: y ∈ [0, 2], b: y ∈ [-1, 1] → y_overlap = 1
    layout = SceneLayout(
        scene_id="s1",
        objects=[
            _bbox("a", -2.0, 1.0, 2.0, 0.0),
            _bbox("b",  0.0, 3.0, 1.0, -1.0),
        ],
    )
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert not result.is_valid
    assert result.violations[0].overlap_x == pytest.approx(1.0)
    assert result.violations[0].overlap_y == pytest.approx(1.0)
    assert result.violations[0].overlap_area == pytest.approx(1.0)


def test_three_objects_two_pairs_overlap():
    # All three objects occupy the same region
    agent = LayoutValidatorAgent()
    layout = SceneLayout(
        scene_id="s1",
        objects=[
            _bbox("a", -1.0, 1.0, 1.0, -1.0),
            _bbox("b", -0.5, 0.5, 0.5, -0.5),
            _bbox("c",  0.0, 2.0, 0.5, -0.5),
        ],
    )
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert not result.is_valid
    # (a,b), (a,c), (b,c) → 3 pairs; at minimum a-b and a-c overlap
    assert len(result.violations) >= 2


# ---------------------------------------------------------------------------
# Temporal non-overlap: objects with t=0 are never "on screen"
# ---------------------------------------------------------------------------

def test_no_temporal_overlap_when_one_has_zero_duration():
    agent = LayoutValidatorAgent()
    layout = SceneLayout(
        scene_id="s1",
        objects=[
            _bbox("a", -1.0, 1.0, 1.0, -1.0, t=30.0),
            _bbox("b", -1.0, 1.0, 1.0, -1.0, t=0.0),   # never visible
        ],
    )
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    # shared_duration = min(30, 0) = 0 → no violation
    assert result.is_valid


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

def test_retry_policy_max_attempts_one():
    agent = LayoutValidatorAgent()
    policy = agent.get_retry_policy()
    assert policy.max_attempts == 1


# ---------------------------------------------------------------------------
# Task 13: (x, y, t) overlap detection — explicit visibility intervals
# ---------------------------------------------------------------------------

def _bbox_with_interval(
    object_id: str,
    x1: float, x2: float,
    y1: float, y2: float,
    t_in: float,
    t_out: float,
    object_type: str = "equation",
) -> CharacterBoundingBox:
    """Like ``_bbox`` but explicit about the visibility window."""
    return CharacterBoundingBox(
        object_id=object_id,
        object_type=object_type,
        top_left=_corner(x1, y1, t=t_out),
        top_right=_corner(x2, y1, t=t_out),
        bottom_left=_corner(x1, y2, t=t_out),
        bottom_right=_corner(x2, y2, t=t_out),
        t_in=t_in,
        t_out=t_out,
    )


def test_simultaneous_overlap_is_flagged():
    """Two objects at overlapping (x, y) and overlapping intervals → violation."""
    agent = LayoutValidatorAgent()
    layout = SceneLayout(scene_id="s1", objects=[
        _bbox_with_interval("a", -1.0, 1.0, 1.0, -1.0, t_in=0.0, t_out=10.0),
        _bbox_with_interval("b", -0.5, 1.5, 1.0, -1.0, t_in=2.0, t_out=8.0),  # overlaps a
    ])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert not result.is_valid
    assert any(
        getattr(v, "object_id_a", "") == "a" and getattr(v, "object_id_b", "") == "b"
        for v in result.violations
    )


def test_serial_reuse_of_position_is_allowed():
    """Two objects at identical (x, y) but disjoint intervals → no violation.

    This is the "stage carries on" case — chunk N's equation is removed,
    chunk N+1 places a new equation in the same slot. The (x, y, t) check
    must not flag this.
    """
    agent = LayoutValidatorAgent()
    layout = SceneLayout(scene_id="s1", objects=[
        _bbox_with_interval("a", -1.0, 1.0, 1.0, -1.0, t_in=0.0, t_out=5.0),
        _bbox_with_interval("b", -1.0, 1.0, 1.0, -1.0, t_in=5.0, t_out=10.0),
    ])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid, [v.description for v in result.violations]


def test_touching_intervals_are_disjoint():
    """t_out of a == t_in of b → exactly zero shared time → no violation."""
    agent = LayoutValidatorAgent()
    layout = SceneLayout(scene_id="s1", objects=[
        _bbox_with_interval("a", -1.0, 1.0, 1.0, -1.0, t_in=0.0, t_out=5.0),
        _bbox_with_interval("b", -1.0, 1.0, 1.0, -1.0, t_in=5.0, t_out=10.0),
    ])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid


def test_overlap_violation_description_names_time_window():
    """The violation's description should include the shared interval so the
    LLM retry prompt can target a specific time."""
    agent = LayoutValidatorAgent()
    layout = SceneLayout(scene_id="s1", objects=[
        _bbox_with_interval("a", -1.0, 1.0, 1.0, -1.0, t_in=0.0, t_out=10.0),
        _bbox_with_interval("b", -0.5, 1.5, 1.0, -1.0, t_in=2.0, t_out=8.0),
    ])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    overlap_descs = [
        v.description for v in result.violations if hasattr(v, "object_id_b")
    ]
    assert overlap_descs, "expected at least one overlap violation"
    desc = overlap_descs[0]
    # The time window 2.0..8.0 should be named.
    assert "2.0" in desc and "8.0" in desc


# ---------------------------------------------------------------------------
# Task 13: Safe-area / off-screen detection
# ---------------------------------------------------------------------------

def test_off_screen_object_is_flagged():
    """An object whose centre is outside the safe area produces an
    OffScreenViolation."""
    from src.schemas.scene_layout import OffScreenViolation

    agent = LayoutValidatorAgent()
    # Centre at x=6.8 — outside safe area (>6.5) but inside screen bounds (<7.1)
    layout = SceneLayout(scene_id="s1", objects=[
        _bbox_with_interval("oob", 6.6, 7.0, 0.5, -0.5, t_in=0.0, t_out=5.0),
    ])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert not result.is_valid
    off_screen = [v for v in result.violations if isinstance(v, OffScreenViolation)]
    assert len(off_screen) == 1
    assert off_screen[0].object_id == "oob"


def test_object_at_safe_area_edge_is_valid():
    """Centre exactly at the safe-area boundary is allowed (inclusive)."""
    agent = LayoutValidatorAgent()
    # Centre at x = +6.5 (SAFE_X_MAX) is still allowed (inclusive)
    layout = SceneLayout(scene_id="s1", objects=[
        _bbox_with_interval("edge", 6.0, 7.0, 0.3, -0.3, t_in=0.0, t_out=5.0),
    ])
    result = agent.execute(LayoutValidationInput(scene_layout=layout))
    assert result.is_valid


# ---------------------------------------------------------------------------
# Task 13: TimeInterval schema validation
# ---------------------------------------------------------------------------

def test_character_bbox_rejects_t_out_before_t_in():
    """t_out must be > t_in — otherwise the visibility window is empty/inverted."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CharacterBoundingBox(
            object_id="x",
            object_type="equation",
            top_left=_corner(-1.0, 1.0),
            top_right=_corner(1.0, 1.0),
            bottom_left=_corner(-1.0, -1.0),
            bottom_right=_corner(1.0, -1.0),
            t_in=10.0,
            t_out=5.0,
        )


def test_character_bbox_legacy_no_t_out_falls_back_to_duration():
    """When t_out is omitted, visible_t_out derives from corner durations
    so the legacy duration-only model keeps working."""
    bbox = CharacterBoundingBox(
        object_id="legacy",
        object_type="equation",
        top_left=_corner(-1.0, 1.0, t=12.0),
        top_right=_corner(1.0, 1.0, t=12.0),
        bottom_left=_corner(-1.0, -1.0, t=12.0),
        bottom_right=_corner(1.0, -1.0, t=12.0),
        # no t_in / t_out provided
    )
    assert bbox.visible_t_in == 0.0
    assert bbox.visible_t_out == 12.0
