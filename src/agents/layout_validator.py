"""Layout Validator Agent — deterministic geometric overlap checker.

Given a SceneLayout, this agent checks every pair of objects that are
simultaneously on screen and reports bounding-box overlaps as
OverlapViolation records.  No LLM call is required.
"""

from typing import List, Union

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, RetryPolicy
from src.schemas.scene_layout import (
    CharacterBoundingBox,
    OffScreenViolation,
    OverlapViolation,
    SAFE_X_MAX,
    SAFE_X_MIN,
    SAFE_Y_MAX,
    SAFE_Y_MIN,
    SceneLayout,
)

LayoutViolation = Union[OverlapViolation, OffScreenViolation]


class LayoutValidationInput(AgentInput):
    """Input for the Layout Validator Agent."""

    def __init__(self, scene_layout: SceneLayout):
        self.scene_layout = scene_layout


class LayoutValidationResult(AgentOutput):
    """Result of a layout validation pass.

    ``violations`` is a heterogeneous list of ``OverlapViolation`` and
    ``OffScreenViolation`` records. Each carries a human-readable
    ``description`` so the layout planner's evaluator loop can feed them
    back into the next attempt's prompt without type discrimination.
    """

    def __init__(self, is_valid: bool, violations: List[LayoutViolation]):
        self.is_valid = is_valid
        self.violations = violations


class LayoutValidatorAgent(Agent):
    """Deterministic agent that validates a SceneLayout for layout problems.

    Two checks fire on every layout:

      1. **Spatial + temporal overlap.** A pair of objects whose
         axis-aligned bounding rectangles intersect AND whose visibility
         windows ``[t_in, t_out)`` intersect produce an
         ``OverlapViolation``. Two objects placed at the same position
         during *disjoint* intervals are NOT a violation — that is the
         "serial reuse of the stage" pattern the unified-script flow
         relies on for chunk-to-chunk continuity.
      2. **Safe area.** An object whose centre lands outside
         ``[SAFE_X_MIN..SAFE_X_MAX] × [SAFE_Y_MIN..SAFE_Y_MAX]`` produces
         an ``OffScreenViolation``. The safe area leaves a small margin
         inside the Manim default frame so labels do not crowd the edges.

    For backwards compatibility, layouts that omit ``t_in`` / ``t_out`` are
    treated as visible from ``[0, duration)`` where ``duration`` is the
    legacy ``max(corner.t)`` field.
    """

    def execute(self, input_data: LayoutValidationInput) -> LayoutValidationResult:
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be LayoutValidationInput with a SceneLayout",
                {"input_type": type(input_data).__name__},
            )

        violations: List[LayoutViolation] = []
        violations.extend(self._check_off_screen(input_data.scene_layout))
        violations.extend(self._check_overlaps(input_data.scene_layout))
        return LayoutValidationResult(is_valid=len(violations) == 0, violations=violations)

    def validate_input(self, input_data: AgentInput) -> bool:
        return (
            isinstance(input_data, LayoutValidationInput)
            and isinstance(input_data.scene_layout, SceneLayout)
        )

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(max_attempts=1)

    # ------------------------------------------------------------------
    # Internal geometry
    # ------------------------------------------------------------------

    def _check_overlaps(self, layout: SceneLayout) -> List[OverlapViolation]:
        violations: List[OverlapViolation] = []
        objects = layout.objects

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                violation = self._pair_overlap(objects[i], objects[j])
                if violation is not None:
                    violations.append(violation)

        return violations

    def _pair_overlap(
        self,
        a: CharacterBoundingBox,
        b: CharacterBoundingBox,
    ) -> "OverlapViolation | None":
        """Return an OverlapViolation if objects a and b overlap in (x, y, t).

        Temporal intersection uses ``[visible_t_in, visible_t_out)``: when
        both windows overlap, we compute the shared seconds. Two objects
        placed at the same position during disjoint windows are *not*
        flagged — the layout planner can re-use real estate as the script
        moves between concepts.
        """
        # Temporal check: visibility-window intersection (Task 13)
        shared_t_in = max(a.visible_t_in, b.visible_t_in)
        shared_t_out = min(a.visible_t_out, b.visible_t_out)
        shared_duration = shared_t_out - shared_t_in
        if shared_duration <= 0.0:
            return None

        # Spatial check: axis-aligned rectangle intersection
        x_overlap = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
        y_overlap = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)

        if x_overlap <= 0.0 or y_overlap <= 0.0:
            return None

        area = round(x_overlap * y_overlap, 4)
        return OverlapViolation(
            object_id_a=a.object_id,
            object_id_b=b.object_id,
            overlap_x=round(x_overlap, 4),
            overlap_y=round(y_overlap, 4),
            overlap_area=area,
            shared_duration=round(shared_duration, 3),
            description=(
                f"'{a.object_id}' and '{b.object_id}' overlap by "
                f"{x_overlap:.2f}×{y_overlap:.2f} units "
                f"(area={area:.3f}) for {shared_duration:.1f}s "
                f"(t={shared_t_in:.1f}..{shared_t_out:.1f})"
            ),
        )

    def _check_off_screen(self, layout: SceneLayout) -> List[OffScreenViolation]:
        """Flag objects whose centre falls outside the safe viewing area."""
        violations: List[OffScreenViolation] = []
        for obj in layout.objects:
            cx, cy = obj.center_x, obj.center_y
            if not (SAFE_X_MIN <= cx <= SAFE_X_MAX and SAFE_Y_MIN <= cy <= SAFE_Y_MAX):
                violations.append(OffScreenViolation(
                    object_id=obj.object_id,
                    center_x=round(cx, 3),
                    center_y=round(cy, 3),
                    description=(
                        f"'{obj.object_id}' centre ({cx:.2f}, {cy:.2f}) is "
                        f"outside the safe area "
                        f"x∈[{SAFE_X_MIN}, {SAFE_X_MAX}], "
                        f"y∈[{SAFE_Y_MIN}, {SAFE_Y_MAX}]"
                    ),
                ))
        return violations
