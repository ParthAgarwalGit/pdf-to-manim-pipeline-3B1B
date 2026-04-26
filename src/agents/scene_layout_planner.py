"""Scene Layout Planner Agent — LLM-based spatial placement.

Given a SceneSpec and SharedContext, this agent asks Claude to assign
4 (x, y, t) corner coordinates to every object in the scene, producing
a SceneLayout that the LayoutValidatorAgent can then check for overlaps.

Screen coordinate system (Manim):
  x ∈ [-7.1, 7.1]   (positive → right)
  y ∈ [-4.0, 4.0]   (positive → up)
  origin at screen centre
"""

import json
import logging
import os
from typing import List, Optional

import anthropic

from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.schemas.scene_layout import (
    SCREEN_X_MAX,
    SCREEN_X_MIN,
    SCREEN_Y_MAX,
    SCREEN_Y_MIN,
    BoundingCorner,
    CharacterBoundingBox,
    SceneLayout,
)
from src.schemas.scene_processing import SceneSpec, SharedContext

logger = logging.getLogger(__name__)


class SceneLayoutPlannerInput(AgentInput):
    """Input for the Scene Layout Planner Agent."""

    def __init__(
        self,
        scene_spec: SceneSpec,
        scene_index: int,
        context: SharedContext,
        violation_feedback: Optional[List[str]] = None,
    ):
        self.scene_spec = scene_spec
        self.scene_index = scene_index
        self.context = context
        self.violation_feedback = violation_feedback or []


class SceneLayoutPlannerOutput(AgentOutput):
    """Output from the Scene Layout Planner Agent."""

    def __init__(self, scene_layout: SceneLayout):
        self.scene_layout = scene_layout


class SceneLayoutPlannerAgent(Agent):
    """LLM-based agent that assigns bounding-box coordinates to scene objects.

    For each object in the scene's visual_intent, the planner assigns:
      - 4 corner coordinates: top_left, top_right, bottom_left, bottom_right
      - Each corner has (x, y, t) where t = duration the object is visible

    When overlap violations are fed back (from LayoutValidatorAgent), the
    planner retries with explicit instructions to fix those violations.
    """

    MODEL = "claude-sonnet-4-6"

    def __init__(self, strict: bool = False):
        self.strict = strict
        if os.getenv("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic()
        else:
            if strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "ANTHROPIC_API_KEY is required in --strict mode for SceneLayoutPlanner",
                )
            self.client = None

    def execute(self, input_data: SceneLayoutPlannerInput) -> SceneLayoutPlannerOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be SceneLayoutPlannerInput with a SceneSpec",
                {"input_type": type(input_data).__name__},
            )

        if self.client is None:
            return SceneLayoutPlannerOutput(
                scene_layout=self._mock_layout(input_data.scene_spec)
            )

        scene_layout = self._plan_layout(input_data)
        return SceneLayoutPlannerOutput(scene_layout=scene_layout)

    def validate_input(self, input_data: AgentInput) -> bool:
        return isinstance(input_data, SceneLayoutPlannerInput) and isinstance(
            input_data.scene_spec, SceneSpec
        )

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            retryable_errors=["API_TIMEOUT", "PARSE_ERROR"],
        )

    # ------------------------------------------------------------------
    # LLM layout planning
    # ------------------------------------------------------------------

    def _plan_layout(self, input_data: SceneLayoutPlannerInput) -> SceneLayout:
        scene_spec = input_data.scene_spec
        violation_feedback = input_data.violation_feedback
        duration = scene_spec.duration_estimate

        objects = scene_spec.visual_intent.mathematical_objects
        if not objects:
            return SceneLayout(scene_id=scene_spec.scene_id, objects=[])

        obj_list = [
            {
                "object_id": getattr(obj, "object_id", None) or f"obj_{i}",
                "object_type": obj.object_type,
                "content": obj.content,
            }
            for i, obj in enumerate(objects)
        ]

        system_prompt = f"""You are a spatial layout planner for Manim educational animations.

SCREEN BOUNDS: x ∈ [{SCREEN_X_MIN}, {SCREEN_X_MAX}], y ∈ [{SCREEN_Y_MIN}, {SCREEN_Y_MAX}]
Origin is the screen centre.  Positive x = right, positive y = up.

For each listed object, assign a rectangular bounding box using 4 corners:
  top_left, top_right, bottom_left, bottom_right
Each corner: {{"x": float, "y": float, "t": float}}
where t = duration in seconds the object is visible (scene duration = {duration:.1f}s).

PLACEMENT RULES:
1. No two objects that share screen time (both t > 0) may have overlapping bounding boxes.
   Leave at least 0.3 units of margin between neighbouring objects.
2. All corners must be within the screen bounds above.
3. Spread objects across the full screen — do not pile everything in the centre.
4. Objects visible throughout the scene all use t = {duration:.1f}.

TYPICAL OBJECT SIZES:
  equation     : width 3–5 units, height 0.8–1.5 units
  text         : width 2–4 units, height 0.6–1.0 units
  pi_creature  : width 1.5–2.5 units, height 2.0–3.0 units
  axes/graph   : width 5–6 units, height 3–4 units  (usually centred)
  geometric_shape: width 2–4 units, height 2–4 units

Respond with ONLY valid JSON — no markdown, no commentary:
{{
  "objects": [
    {{
      "object_id": "string",
      "object_type": "string",
      "top_left":    {{"x": <float>, "y": <float>, "t": <float>}},
      "top_right":   {{"x": <float>, "y": <float>, "t": <float>}},
      "bottom_left": {{"x": <float>, "y": <float>, "t": <float>}},
      "bottom_right":{{"x": <float>, "y": <float>, "t": <float>}}
    }}
  ]
}}"""

        user_content = (
            f"Scene: {scene_spec.scene_id}\n"
            f"Narration: {scene_spec.narration}\n\n"
            f"Objects to place:\n{json.dumps(obj_list, indent=2)}"
        )

        if violation_feedback:
            feedback_lines = "\n".join(f"  - {v}" for v in violation_feedback[:10])
            user_content += (
                f"\n\nOVERLAP VIOLATIONS TO FIX IN THIS REVISION:\n{feedback_lines}\n"
                "Move the overlapping objects apart so no two bounding boxes intersect."
            )

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.0,
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if the model wrapped output anyway
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)
            return self._parse_response(scene_spec.scene_id, data)

        except Exception as exc:
            logger.warning("Layout planner LLM call failed: %s — falling back to mock layout", exc)
            return self._mock_layout(scene_spec)

    def _parse_response(self, scene_id: str, data: dict) -> SceneLayout:
        objects: List[CharacterBoundingBox] = []
        for obj_data in data.get("objects", []):
            try:
                bbox = CharacterBoundingBox(
                    object_id=obj_data["object_id"],
                    object_type=obj_data["object_type"],
                    top_left=BoundingCorner(**obj_data["top_left"]),
                    top_right=BoundingCorner(**obj_data["top_right"]),
                    bottom_left=BoundingCorner(**obj_data["bottom_left"]),
                    bottom_right=BoundingCorner(**obj_data["bottom_right"]),
                )
                objects.append(bbox)
            except Exception as exc:
                logger.warning(
                    "Skipping malformed layout object '%s': %s",
                    obj_data.get("object_id", "?"),
                    exc,
                )
        return SceneLayout(scene_id=scene_id, objects=objects)

    # ------------------------------------------------------------------
    # Mock layout (used when no API key is available)
    # ------------------------------------------------------------------

    def _mock_layout(self, scene_spec: SceneSpec) -> SceneLayout:
        """Non-overlapping grid layout for testing without the API."""
        objects = scene_spec.visual_intent.mathematical_objects
        if not objects:
            return SceneLayout(scene_id=scene_spec.scene_id, objects=[])

        duration = scene_spec.duration_estimate
        # Divide screen into a 3-column grid; stack rows downward
        cols = 3
        cell_w = (SCREEN_X_MAX - SCREEN_X_MIN) / cols   # ≈ 4.73
        cell_h = 2.5
        margin = 0.4  # gap between cells

        bboxes: List[CharacterBoundingBox] = []
        for idx, obj in enumerate(objects):
            col = idx % cols
            row = idx // cols
            x_centre = SCREEN_X_MIN + (col + 0.5) * cell_w
            y_centre = SCREEN_Y_MAX - 1.0 - row * cell_h

            obj_type = obj.object_type
            if obj_type in ("pi_creature",):
                w, h = 2.0, 2.5
            elif obj_type in ("axes", "graph"):
                w, h = 5.0, 3.5
            elif obj_type == "geometric_shape":
                w, h = 2.5, 2.5
            else:
                w, h = 2.5, 1.0

            # Shrink to fit within the cell (leave margin)
            w = min(w, cell_w - margin)
            h = min(h, cell_h - margin)

            x1 = max(SCREEN_X_MIN, x_centre - w / 2)
            x2 = min(SCREEN_X_MAX, x_centre + w / 2)
            y1 = min(SCREEN_Y_MAX, y_centre + h / 2)
            y2 = max(SCREEN_Y_MIN, y_centre - h / 2)

            # Safety: ensure x1 < x2 and y2 < y1
            if x1 >= x2:
                x2 = x1 + 0.5
            if y2 >= y1:
                y1 = y2 + 0.5

            obj_id = getattr(obj, "object_id", None) or f"obj_{idx}"

            bboxes.append(
                CharacterBoundingBox(
                    object_id=obj_id,
                    object_type=obj_type,
                    top_left=BoundingCorner(x=x1, y=y1, t=duration),
                    top_right=BoundingCorner(x=x2, y=y1, t=duration),
                    bottom_left=BoundingCorner(x=x1, y=y2, t=duration),
                    bottom_right=BoundingCorner(x=x2, y=y2, t=duration),
                )
            )

        return SceneLayout(scene_id=scene_spec.scene_id, objects=bboxes)
