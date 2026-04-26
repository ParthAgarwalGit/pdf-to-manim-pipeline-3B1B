"""Scene layout schemas for spatial placement and overlap detection.

Each visual object placed in a scene is described by a bounding box with
4 corner (x, y, t) coordinates:
  - x: horizontal position in Manim screen space (range [-7.1, 7.1])
  - y: vertical position in Manim screen space (range [-4.0, 4.0])
  - t: duration in seconds the object is visible on screen
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCREEN_X_MIN = -7.1
SCREEN_X_MAX = 7.1
SCREEN_Y_MIN = -4.0
SCREEN_Y_MAX = 4.0

# Safe area — keeps a small margin so labels do not crowd the screen edges.
# Used by the layout validator to flag bboxes whose centre lands outside
# the comfortable viewing region.
SAFE_X_MIN = -6.5
SAFE_X_MAX = 6.5
SAFE_Y_MIN = -3.5
SAFE_Y_MAX = 3.5


class BoundingCorner(BaseModel):
    """A single corner of a bounding box with a visibility duration."""

    x: float = Field(..., description="Horizontal position in Manim coordinate space")
    y: float = Field(..., description="Vertical position in Manim coordinate space")
    t: float = Field(..., ge=0.0, description="Duration (seconds) this object is visible on screen")

    @field_validator("x")
    @classmethod
    def x_in_bounds(cls, v: float) -> float:
        if not (SCREEN_X_MIN <= v <= SCREEN_X_MAX):
            raise ValueError(f"x={v} is outside screen bounds [{SCREEN_X_MIN}, {SCREEN_X_MAX}]")
        return v

    @field_validator("y")
    @classmethod
    def y_in_bounds(cls, v: float) -> float:
        if not (SCREEN_Y_MIN <= v <= SCREEN_Y_MAX):
            raise ValueError(f"y={v} is outside screen bounds [{SCREEN_Y_MIN}, {SCREEN_Y_MAX}]")
        return v


class CharacterBoundingBox(BaseModel):
    """Bounding box for a single character/object in a scene.

    Defined by four (x, y, t) corner coordinates. All four corners must form
    a valid axis-aligned rectangle with top > bottom and right > left.

    Visibility window (Task 13): when ``t_in`` / ``t_out`` are set, the
    object is considered on screen during ``[t_in, t_out)``. When ``t_out``
    is None, it falls back to ``[0, max(corner.t))`` for backward
    compatibility with the original duration-only model — so any layout
    written before the interval semantics were introduced still validates.
    """

    object_id: str = Field(..., description="Identifier matching the visual_intent object_id")
    object_type: str = Field(..., description="Type of object (equation, text, pi_creature, …)")
    top_left: BoundingCorner
    top_right: BoundingCorner
    bottom_left: BoundingCorner
    bottom_right: BoundingCorner
    t_in: float = Field(
        default=0.0,
        ge=0.0,
        description="When the object first becomes visible (seconds). "
                    "Defaults to 0 so legacy layouts behave unchanged.",
    )
    t_out: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="When the object is removed (seconds). When None, falls "
                    "back to the maximum corner ``t`` so legacy layouts behave "
                    "as if every object was visible from t=0 to its duration.",
    )

    @model_validator(mode="after")
    def _validate_interval(self) -> "CharacterBoundingBox":
        if self.t_out is not None and self.t_out <= self.t_in:
            raise ValueError(
                f"t_out ({self.t_out}) must be > t_in ({self.t_in}) "
                f"for object_id={self.object_id!r}"
            )
        return self

    @model_validator(mode="after")
    def validate_corners_form_rectangle(self) -> "CharacterBoundingBox":
        if self.top_left.x >= self.top_right.x:
            raise ValueError("top_left.x must be strictly less than top_right.x")
        if self.bottom_left.x >= self.bottom_right.x:
            raise ValueError("bottom_left.x must be strictly less than bottom_right.x")
        if self.top_left.y <= self.bottom_left.y:
            raise ValueError("top_left.y must be strictly greater than bottom_left.y")
        if self.top_right.y <= self.bottom_right.y:
            raise ValueError("top_right.y must be strictly greater than bottom_right.y")
        return self

    # ------------------------------------------------------------------
    # Derived geometry helpers
    # ------------------------------------------------------------------

    @property
    def x_min(self) -> float:
        return min(self.top_left.x, self.bottom_left.x)

    @property
    def x_max(self) -> float:
        return max(self.top_right.x, self.bottom_right.x)

    @property
    def y_min(self) -> float:
        return min(self.bottom_left.y, self.bottom_right.y)

    @property
    def y_max(self) -> float:
        return max(self.top_left.y, self.top_right.y)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def duration(self) -> float:
        """How long this object is on screen (max t across all 4 corners)."""
        return max(
            self.top_left.t,
            self.top_right.t,
            self.bottom_left.t,
            self.bottom_right.t,
        )

    @property
    def visible_t_in(self) -> float:
        """When the object first becomes visible (seconds)."""
        return self.t_in

    @property
    def visible_t_out(self) -> float:
        """When the object is removed (seconds). Falls back to ``duration``
        for legacy layouts that did not specify ``t_out`` explicitly."""
        if self.t_out is not None:
            return self.t_out
        return self.duration

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0


class OverlapViolation(BaseModel):
    """Detected geometric overlap between two objects visible at the same time."""

    object_id_a: str = Field(..., description="First object in the overlapping pair")
    object_id_b: str = Field(..., description="Second object in the overlapping pair")
    overlap_x: float = Field(..., description="Horizontal overlap extent (screen units)")
    overlap_y: float = Field(..., description="Vertical overlap extent (screen units)")
    overlap_area: float = Field(..., ge=0.0, description="Area of overlap (screen units²)")
    shared_duration: float = Field(..., ge=0.0, description="Seconds both objects are visible simultaneously")
    description: str = Field(default="", description="Human-readable description of the violation")


class OffScreenViolation(BaseModel):
    """An object's centre falls outside the safe viewing area (Task 13)."""

    object_id: str = Field(..., description="Object placed outside the safe area")
    center_x: float = Field(..., description="Centre x of the offending bbox")
    center_y: float = Field(..., description="Centre y of the offending bbox")
    description: str = Field(default="", description="Human-readable description of the violation")


class SceneLayout(BaseModel):
    """Spatial layout specification for all objects in a single scene."""

    scene_id: str = Field(..., description="Scene identifier matching SceneSpec.scene_id")
    objects: List[CharacterBoundingBox] = Field(
        default_factory=list,
        description="Bounding boxes for every placed object in this scene",
    )

    @field_validator("scene_id")
    @classmethod
    def validate_scene_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("scene_id cannot be empty")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scene_id": "intro_001",
                "objects": [
                    {
                        "object_id": "equation_0",
                        "object_type": "equation",
                        "top_left":    {"x": -2.5, "y":  1.0, "t": 30.0},
                        "top_right":   {"x":  2.5, "y":  1.0, "t": 30.0},
                        "bottom_left": {"x": -2.5, "y":  0.0, "t": 30.0},
                        "bottom_right":{"x":  2.5, "y":  0.0, "t": 30.0},
                    }
                ],
            }
        }
    )
