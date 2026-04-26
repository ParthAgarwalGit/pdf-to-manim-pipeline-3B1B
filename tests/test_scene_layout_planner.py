"""Tests for SceneLayoutPlannerAgent."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.scene_layout_planner import (
    SceneLayoutPlannerAgent,
    SceneLayoutPlannerInput,
    SceneLayoutPlannerOutput,
)
from src.schemas.scene_layout import SceneLayout
from src.schemas.scene_processing import SceneSpec, SharedContext, VisualStyle
from src.schemas.storyboard import (
    MathematicalObject,
    ObjectStyle,
    Transformation,
    VisualIntent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene_spec(
    scene_id: str = "test_001",
    narration: str = "This is a test scene about mathematics.",
    duration: float = 30.0,
    objects: list | None = None,
) -> SceneSpec:
    if objects is None:
        objects = [
            MathematicalObject(
                object_type="equation",
                content="E = mc^2",
                style=ObjectStyle(color="blue", position="center"),
            ),
            MathematicalObject(
                object_type="text",
                content="Energy equals mass times c squared",
                style=ObjectStyle(color="white", position="top"),
            ),
        ]
    return SceneSpec(
        scene_id=scene_id,
        narration=narration,
        visual_intent=VisualIntent(
            mathematical_objects=objects,
            transformations=[],
            emphasis_points=[],
        ),
        duration_estimate=duration,
    )


def _make_context() -> SharedContext:
    return SharedContext(
        visual_style=VisualStyle(colors={"primary": "#58C4DD"}, font_size=24)
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_validate_input_correct_type():
    agent = SceneLayoutPlannerAgent()
    inp = SceneLayoutPlannerInput(
        scene_spec=_make_scene_spec(),
        scene_index=0,
        context=_make_context(),
    )
    assert agent.validate_input(inp)


def test_validate_input_wrong_type():
    agent = SceneLayoutPlannerAgent()
    assert not agent.validate_input("bad_input")


# ---------------------------------------------------------------------------
# Mock layout (no API key)
# ---------------------------------------------------------------------------

def test_mock_layout_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = SceneLayoutPlannerAgent()

    inp = SceneLayoutPlannerInput(
        scene_spec=_make_scene_spec(),
        scene_index=0,
        context=_make_context(),
    )
    output = agent.execute(inp)
    assert isinstance(output, SceneLayoutPlannerOutput)
    assert isinstance(output.scene_layout, SceneLayout)
    # Each object in spec should have a corresponding bbox
    assert len(output.scene_layout.objects) == 2


def test_mock_layout_empty_scene(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = SceneLayoutPlannerAgent()

    spec = _make_scene_spec(objects=[])
    inp = SceneLayoutPlannerInput(scene_spec=spec, scene_index=0, context=_make_context())
    output = agent.execute(inp)
    assert output.scene_layout.objects == []


def test_mock_layout_no_overlaps(monkeypatch):
    """The mock grid layout must not produce overlapping bounding boxes."""
    from src.agents.layout_validator import LayoutValidationInput, LayoutValidatorAgent

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = SceneLayoutPlannerAgent()
    validator = LayoutValidatorAgent()

    # Use enough objects to fill multiple rows
    objects = [
        MathematicalObject(
            object_type="equation",
            content=f"x_{i} = {i}",
            style=ObjectStyle(color="blue", position="center"),
        )
        for i in range(6)
    ]
    spec = _make_scene_spec(objects=objects)
    inp = SceneLayoutPlannerInput(scene_spec=spec, scene_index=0, context=_make_context())
    output = agent.execute(inp)

    val_result = validator.execute(LayoutValidationInput(scene_layout=output.scene_layout))
    assert val_result.is_valid, f"Mock layout has overlaps: {val_result.violations}"


# ---------------------------------------------------------------------------
# Bounding corners respect screen bounds
# ---------------------------------------------------------------------------

def test_mock_layout_within_screen_bounds(monkeypatch):
    from src.schemas.scene_layout import SCREEN_X_MAX, SCREEN_X_MIN, SCREEN_Y_MAX, SCREEN_Y_MIN

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = SceneLayoutPlannerAgent()

    objects = [
        MathematicalObject(
            object_type="axes",
            content="coordinate plane",
            style=ObjectStyle(color="white", position="center"),
        )
    ]
    spec = _make_scene_spec(objects=objects)
    inp = SceneLayoutPlannerInput(scene_spec=spec, scene_index=0, context=_make_context())
    output = agent.execute(inp)

    for bbox in output.scene_layout.objects:
        assert bbox.x_min >= SCREEN_X_MIN
        assert bbox.x_max <= SCREEN_X_MAX
        assert bbox.y_min >= SCREEN_Y_MIN
        assert bbox.y_max <= SCREEN_Y_MAX


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

def test_retry_policy():
    agent = SceneLayoutPlannerAgent()
    policy = agent.get_retry_policy()
    assert policy.max_attempts == 3


# ---------------------------------------------------------------------------
# Violation feedback is threaded through
# ---------------------------------------------------------------------------

def test_violation_feedback_accepted(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = SceneLayoutPlannerAgent()

    inp = SceneLayoutPlannerInput(
        scene_spec=_make_scene_spec(),
        scene_index=0,
        context=_make_context(),
        violation_feedback=["'eq_0' and 'text_1' overlap by 1.0×0.5 units for 30.0s"],
    )
    # Should not raise; mock path ignores feedback but accepts it
    output = agent.execute(inp)
    assert isinstance(output.scene_layout, SceneLayout)
