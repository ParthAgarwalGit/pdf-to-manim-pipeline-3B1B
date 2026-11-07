"""Storyboard JSON schema for Script/Storyboard Architect."""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.schemas.video_script import Beat
from src.utils.manim_engine_contract import (
    CANONICAL_CAMERA_ACTIONS,
    CANONICAL_CHARACTER_ROLES,
    CANONICAL_CONTINUITY_ACTIONS,
    CANONICAL_OBJECT_TYPES,
    CANONICAL_RENDER_HINTS,
    normalize_object_type,
    normalize_render_hint,
)


class PDFMetadataStoryboard(BaseModel):
    """Metadata about the source PDF (simplified for storyboard)."""
    
    filename: str = Field(..., description="Name of the PDF file")
    page_count: int = Field(..., ge=1, description="Total number of pages in the PDF")


class Configuration(BaseModel):
    """Configuration parameters for storyboard generation."""
    
    verbosity: str = Field(
        ...,
        description="Narration verbosity level (low|medium|high)"
    )
    depth: str = Field(
        ...,
        description="Content depth level (introductory|intermediate|advanced)"
    )
    audience: str = Field(
        ...,
        description="Target audience level (high-school|undergraduate|graduate)"
    )
    granularity: str = Field(
        default="balanced",
        description="Source segmentation granularity (coarse|balanced|fine)"
    )
    target_minutes: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Requested finished-video length in minutes. When set, the scene "
            "budget is derived from it so length is a chosen input rather than "
            "an emergent property of document size. None = ungoverned."
        ),
    )

    @field_validator('verbosity')
    @classmethod
    def validate_verbosity(cls, v: str) -> str:
        """Validate verbosity is one of the allowed values."""
        allowed = {'low', 'medium', 'high'}
        if v not in allowed:
            raise ValueError(f"verbosity must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('depth')
    @classmethod
    def validate_depth(cls, v: str) -> str:
        """Validate depth is one of the allowed values."""
        allowed = {'introductory', 'intermediate', 'advanced'}
        if v not in allowed:
            raise ValueError(f"depth must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('audience')
    @classmethod
    def validate_audience(cls, v: str) -> str:
        """Validate audience is one of the allowed values."""
        allowed = {'high-school', 'undergraduate', 'graduate'}
        if v not in allowed:
            raise ValueError(f"audience must be one of {allowed}, got '{v}'")
        return v

    @field_validator('granularity')
    @classmethod
    def validate_granularity(cls, v: str) -> str:
        """Validate segment granularity is one of the allowed values."""
        allowed = {'coarse', 'balanced', 'fine'}
        if v not in allowed:
            raise ValueError(f"granularity must be one of {allowed}, got '{v}'")
        return v


class ConceptHierarchy(BaseModel):
    """Concept in the pedagogical hierarchy."""
    
    concept_id: str = Field(..., description="Unique identifier for the concept")
    concept_name: str = Field(..., description="Human-readable name of the concept")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Array of concept_ids that this concept depends on"
    )


class ObjectStyle(BaseModel):
    """Visual style parameters for a mathematical object."""

    color: Optional[str] = Field(
        None,
        description="Color specification (e.g., hex code, color name)"
    )
    position: Optional[str] = Field(
        None,
        description="Position on screen (e.g., 'center', 'top-left', 'bottom-right')"
    )


# ─────────────────────────────────────────────────────────────────────────
# (x, y, t) layout primitives — Task 13 of the unified-script plan.
#
# Manim's default frame is roughly x ∈ [-7.1, 7.1], y ∈ [-4.0, 4.0]. The
# safe area below leaves a small margin so labels do not crowd the edges.
# ─────────────────────────────────────────────────────────────────────────


class BBox(BaseModel):
    """Axis-aligned bounding box in Manim world units (origin at center)."""

    x: float = Field(..., description="Center x")
    y: float = Field(..., description="Center y")
    w: float = Field(..., gt=0, description="Width (positive)")
    h: float = Field(..., gt=0, description="Height (positive)")

    @property
    def x_min(self) -> float:
        return self.x - self.w / 2

    @property
    def x_max(self) -> float:
        return self.x + self.w / 2

    @property
    def y_min(self) -> float:
        return self.y - self.h / 2

    @property
    def y_max(self) -> float:
        return self.y + self.h / 2


class TimeInterval(BaseModel):
    """Visibility window for an object. ``t_in`` inclusive, ``t_out`` exclusive."""

    t_in: float = Field(..., ge=0, description="When the object first appears (s)")
    t_out: float = Field(..., gt=0, description="When the object is removed (s)")

    @model_validator(mode="after")
    def _validate_ordering(self) -> "TimeInterval":
        if self.t_out <= self.t_in:
            raise ValueError(f"t_out ({self.t_out}) must be > t_in ({self.t_in})")
        return self


class MathematicalObject(BaseModel):
    """Mathematical object to be animated in a scene."""

    object_id: Optional[str] = Field(
        None,
        description="Stable identifier for continuity-aware planning and compilation"
    )
    object_role: Optional[str] = Field(
        None,
        description="Semantic role such as main_equation, shared_axes, teacher, or label"
    )
    purpose: str = Field(
        default="",
        description="Concise explanation of why this visible object is needed",
    )
    supports_beat_ids: List[str] = Field(
        default_factory=list,
        description="Narration/visual beat ids this object directly supports",
    )
    object_type: str = Field(
        ...,
        description=(
            "Canonical object family "
            "(equation|graph|geometric_shape|axes|text|pi_creature)"
        )
    )
    content: str = Field(
        ...,
        description="LaTeX for equations, description for other types"
    )
    coordinate_system_id: Optional[str] = Field(
        None,
        description="Identifier for a shared coordinate system reused across beats/scenes"
    )
    notation_key: Optional[str] = Field(
        None,
        description="Stable symbolic identity for notation continuity"
    )
    character_role: Optional[str] = Field(
        None,
        description="Persistent character role when object_type is pi_creature"
    )
    persist_across_scenes: bool = Field(
        default=False,
        description="Whether this object should remain available to later scenes"
    )
    render_hint: Optional[str] = Field(
        None,
        description=(
            "Optional preferred Manim primitive/helper such as "
            "MathTex, Text, Axes, NumberPlane, Line, Arrow, Polygon, "
            "Circle, Arc, PiCreatureTeacher, PiCreatureStudents, or SpeechBubble"
        ),
    )
    style: ObjectStyle = Field(
        default_factory=ObjectStyle,
        description="Visual style parameters"
    )
    primitive: Optional[str] = Field(
        None,
        description=(
            "Closed-vocabulary Manim primitive name (e.g. 'Circle', 'MathTex'). "
            "When set, ``args`` carries the kwargs for direct construction. "
            "Falls back to legacy ``object_type``/``render_hint`` when None."
        )
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Kwargs for the primitive constructor (e.g. {'radius': 2.0})."
    )
    bbox: Optional[BBox] = Field(
        None,
        description="(x, y, w, h) bounding box in Manim units. Required when "
                    "the layout validator is used."
    )
    interval: Optional[TimeInterval] = Field(
        None,
        description="Visibility window (t_in, t_out) in seconds — paired with "
                    "``bbox`` for (x, y, t) overlap checks."
    )

    @field_validator('object_type')
    @classmethod
    def validate_object_type(cls, v: str) -> str:
        """Validate object_type is one of the allowed values."""
        canonical = normalize_object_type(v)
        allowed = set(CANONICAL_OBJECT_TYPES)
        if canonical not in allowed:
            raise ValueError(f"object_type must be one of {allowed}, got '{v}'")
        return canonical

    @field_validator('character_role')
    @classmethod
    def validate_character_role(cls, v: Optional[str]) -> Optional[str]:
        """Validate persistent character roles against the canonical enum."""
        if v is None:
            return None
        allowed = set(CANONICAL_CHARACTER_ROLES)
        if v not in allowed:
            raise ValueError(f"character_role must be one of {allowed}, got '{v}'")
        return v

    @field_validator('render_hint')
    @classmethod
    def validate_render_hint(cls, v: Optional[str]) -> Optional[str]:
        """Validate render_hint is a recognized Manim primitive/helper."""
        hint = normalize_render_hint(v)
        if hint is None:
            return None
        allowed = set(CANONICAL_RENDER_HINTS)
        if hint not in allowed:
            raise ValueError(f"render_hint must be one of {allowed}, got '{v}'")
        return hint

    @model_validator(mode="after")
    def _validate_primitive_against_catalog(self) -> "MathematicalObject":
        """When ``primitive`` is set it must exist in MOBJECT_CATALOG and
        every key in ``args`` must be a known parameter for that primitive
        (with ``color`` permitted for primitives that accept it)."""
        if self.primitive is None:
            return self
        # Lazy import to avoid a circular dep at module-load time
        from src.utils.manim_primitives import MOBJECT_CATALOG, allowed_args_for

        if self.primitive not in MOBJECT_CATALOG:
            raise ValueError(
                f"Unknown primitive: {self.primitive!r}. Allowed: "
                f"{sorted(MOBJECT_CATALOG.keys())}"
            )
        bad = set(self.args) - allowed_args_for(self.primitive)
        if bad:
            raise ValueError(
                f"Unknown args for primitive {self.primitive!r}: {sorted(bad)}. "
                f"Allowed: {sorted(allowed_args_for(self.primitive))}"
            )
        return self


class Transformation(BaseModel):
    """Animation transformation to apply to a mathematical object."""

    transformation_type: str = Field(
        ...,
        description="Type of transformation (morph|highlight|move|fade_in|fade_out|scale)"
    )
    source_object_id: str = Field(
        ...,
        description="Stable object id for the object the transformation starts from"
    )
    target_object_id: Optional[str] = Field(
        None,
        description="Stable object id for the object the transformation targets, when applicable"
    )
    continuity_action: str = Field(
        default="transform",
        description="Semantic lifecycle action (create|reuse|transform|remove|highlight)"
    )
    beat_id: str = Field(
        default="beat_000",
        description="Beat identifier tying this transformation to a visual beat"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transformation-specific parameters"
    )
    timing: float = Field(
        ...,
        ge=0.0,
        description="Seconds from scene start when transformation begins"
    )
    
    @field_validator('transformation_type')
    @classmethod
    def validate_transformation_type(cls, v: str) -> str:
        """Normalize and validate transformation_type.

        LLMs often return PascalCase Manim class names (e.g. 'FadeIn',
        'Write', 'GrowFromCenter').  We normalise to snake_case and map
        common Manim animation names to our canonical set.
        """
        import re

        # PascalCase → snake_case  (e.g. "FadeIn" → "fade_in")
        normalized = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', v)
        normalized = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', normalized)
        normalized = normalized.lower().strip()

        # Map common Manim animation names to canonical types
        alias_map = {
            "write": "fade_in",
            "create": "fade_in",
            "grow_from_center": "fade_in",
            "grow_arrow": "fade_in",
            "draw_border_then_fill": "fade_in",
            "show_creation": "fade_in",
            "uncreate": "fade_out",
            "shrink_to_center": "fade_out",
            "transform": "morph",
            "replacement_transform": "morph",
            "transform_from_copy": "morph",
            "clockwise_transform": "morph",
            "rotate": "move",
            "shift": "move",
            "animate": "move",
            "indicate": "highlight",
            "flash": "highlight",
            "circumscribe": "highlight",
            "wiggle": "highlight",
            "focus_on": "highlight",
            "scale_in_place": "scale",
        }

        canonical = alias_map.get(normalized, normalized)

        allowed = {'morph', 'highlight', 'move', 'fade_in', 'fade_out', 'scale'}
        if canonical not in allowed:
            raise ValueError(f"transformation_type must be one of {allowed}, got '{v}'")
        return canonical

    @field_validator('continuity_action')
    @classmethod
    def validate_continuity_action(cls, v: str) -> str:
        """Validate semantic continuity actions."""
        allowed = set(CANONICAL_CONTINUITY_ACTIONS)
        if v not in allowed:
            raise ValueError(f"continuity_action must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode='before')
    @classmethod
    def migrate_legacy_target_object(cls, data: Any) -> Any:
        """Support legacy target_object inputs while the pipeline migrates."""
        if not isinstance(data, dict):
            return data
        if "source_object_id" not in data and "target_object" in data:
            data["source_object_id"] = data["target_object"]
        if "target_object_id" not in data and "target_object" in data:
            data["target_object_id"] = data["target_object"]
        if "animation_hint" in data and "parameters" not in data:
            data["parameters"] = {"animation_hint": data["animation_hint"]}
        return data

    @property
    def target_object(self) -> str:
        """Backward-compatible alias used by older generator code."""
        return self.source_object_id


class EmphasisPoint(BaseModel):
    """Point in time where specific content should be emphasized."""
    
    timestamp: float = Field(
        ...,
        ge=0.0,
        description="Seconds from scene start"
    )
    description: str = Field(
        ...,
        description="What to emphasize at this point"
    )


class SceneState(BaseModel):
    """Explicit continuity boundary shared between scenes."""

    active_object_ids: List[str] = Field(
        default_factory=list,
        description="Object ids that remain active at this boundary"
    )
    coordinate_systems: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of coordinate_system_id to semantic kind"
    )
    notation_bindings: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of notation_key to object_id"
    )
    character_roles: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of character_role to object_id"
    )


class VisualBeat(BaseModel):
    """Beat-level visual unit aligned to the narration."""

    beat_id: str = Field(..., description="Stable beat identifier")
    narration_excerpt: str = Field(
        default="",
        description="Short narration excerpt anchoring the beat"
    )
    start_time: float = Field(..., ge=0.0, description="Beat start time in seconds")
    end_time: float = Field(..., ge=0.0, description="Beat end time in seconds")
    active_object_ids: List[str] = Field(
        default_factory=list,
        description="Objects expected to remain visible during this beat"
    )
    camera_action: str = Field(
        default="static",
        description="Camera action for this beat"
    )
    beat_goal: str = Field(
        default="",
        description="Narrative goal for the beat"
    )

    @field_validator('camera_action')
    @classmethod
    def validate_camera_action(cls, v: str) -> str:
        """Validate beat-level camera directives."""
        allowed = set(CANONICAL_CAMERA_ACTIONS)
        if v not in allowed:
            raise ValueError(f"camera_action must be one of {allowed}, got '{v}'")
        return v


class VisualIntent(BaseModel):
    """Visual intent specification for a scene."""
    
    mathematical_objects: List[MathematicalObject] = Field(
        ...,
        description="Mathematical objects to create and animate"
    )
    transformations: List[Transformation] = Field(
        ...,
        description="Transformations to apply to objects"
    )
    beats: List[VisualBeat] = Field(
        default_factory=list,
        description="Beat-by-beat structure aligned to narration timing"
    )
    scene_state_in: SceneState = Field(
        default_factory=SceneState,
        description="Continuity state entering the scene"
    )
    scene_state_out: SceneState = Field(
        default_factory=SceneState,
        description="Continuity state exiting the scene"
    )
    emphasis_points: List[EmphasisPoint] = Field(
        default_factory=list,
        description="Points in time where content should be emphasized"
    )


class Scene(BaseModel):
    """Single scene specification in the storyboard."""

    scene_id: str = Field(..., description="Unique identifier for the scene")
    scene_index: int = Field(..., ge=0, description="Sequential index in the scene array")
    concept_id: str = Field(
        ...,
        description="Reference to concept_hierarchy concept_id"
    )
    narration: str = Field(..., description="Narration text for the scene")
    visual_intent: VisualIntent = Field(
        ...,
        description="Visual intent specification"
    )
    duration_estimate: float = Field(
        ...,
        gt=0.0,
        description="Estimated scene duration in seconds"
    )
    audio_duration_seconds: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Measured TTS duration when narration-first timing is enabled",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Array of scene_ids that this scene depends on"
    )
    difficulty_level: Optional[str] = Field(
        None,
        description="Difficulty level of the scene content"
    )
    source_beats: List[Beat] = Field(
        default_factory=list,
        description="The Beats from the unified VideoScript that were grouped "
                    "into this scene by the chunker. Empty for legacy storyboards "
                    "produced before Task 4 of the chunked-render plan."
    )
    source_segment_ids: List[str] = Field(
        default_factory=list,
        description="SourceSegment ids covered by this scene."
    )
    beat_timings: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Timing track: beat_id -> [start, end] seconds within the "
                    "scene's narration audio (per-beat TTS)."
    )

    @field_validator('scene_id')
    @classmethod
    def validate_scene_id(cls, v: str) -> str:
        """Ensure scene_id is non-empty and filename-safe.

        scene_id is LLM-influenced and later embedded in file paths
        (scene_{scene_id}.py); path separators or dots would allow writes
        outside the run directory (Windows resolves '..' lexically).
        """
        if not v or not v.strip():
            raise ValueError("scene_id cannot be empty")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", v):
            raise ValueError(
                f"scene_id must contain only [A-Za-z0-9_-], got '{v}'"
            )
        return v

    @field_validator('narration')
    @classmethod
    def validate_narration(cls, v: str) -> str:
        """Ensure narration is not empty."""
        if not v or not v.strip():
            raise ValueError("narration cannot be empty")
        return v


class StoryboardJSON(BaseModel):
    """
    Complete storyboard output from Script/Storyboard Architect.

    Contains ordered scene specifications with narration and visual intent.
    """

    pdf_metadata: PDFMetadataStoryboard = Field(
        ...,
        description="Metadata about the source PDF"
    )
    configuration: Configuration = Field(
        ...,
        description="Configuration parameters used for generation"
    )
    concept_hierarchy: List[ConceptHierarchy] = Field(
        default_factory=list,
        description="Pedagogical concept hierarchy"
    )
    scenes: List[Scene] = Field(
        ...,
        description="Ordered array of scene specifications"
    )

    @field_validator('scenes')
    @classmethod
    def validate_scenes_not_empty(cls, v: List[Scene]) -> List[Scene]:
        """Ensure scenes list is not empty."""
        if not v:
            raise ValueError("scenes list cannot be empty")
        return v

    @field_validator('scenes')
    @classmethod
    def validate_scene_id_uniqueness(cls, v: List[Scene]) -> List[Scene]:
        """Ensure all scene_ids are unique."""
        scene_ids = [scene.scene_id for scene in v]
        if len(scene_ids) != len(set(scene_ids)):
            duplicates = [sid for sid in scene_ids if scene_ids.count(sid) > 1]
            raise ValueError(f"Duplicate scene_ids found: {set(duplicates)}")
        return v

    @model_validator(mode='after')
    def validate_concept_references(self) -> 'StoryboardJSON':
        """Ensure all scene concept_ids reference valid concepts in the hierarchy."""
        valid_concept_ids = {c.concept_id for c in self.concept_hierarchy}
        if valid_concept_ids:  # Only validate if hierarchy is provided
            for scene in self.scenes:
                if scene.concept_id not in valid_concept_ids:
                    raise ValueError(
                        f"Scene '{scene.scene_id}' references concept_id "
                        f"'{scene.concept_id}' which is not in concept_hierarchy. "
                        f"Valid concept_ids: {valid_concept_ids}"
                    )
        return self

    @model_validator(mode='after')
    def validate_scene_indices(self) -> 'StoryboardJSON':
        """Ensure scene_index values are sequential and match array positions."""
        for i, scene in enumerate(self.scenes):
            if scene.scene_index != i:
                raise ValueError(
                    f"Scene '{scene.scene_id}' has scene_index={scene.scene_index} "
                    f"but is at position {i} in the scenes array"
                )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pdf_metadata": {
                    "filename": "linear_algebra.pdf",
                    "page_count": 5
                },
                "configuration": {
                    "verbosity": "medium",
                    "depth": "intermediate",
                    "audience": "undergraduate"
                },
                "concept_hierarchy": [
                    {
                        "concept_id": "vector_space",
                        "concept_name": "Vector Space",
                        "dependencies": []
                    }
                ],
                "scenes": [
                    {
                        "scene_id": "intro_001",
                        "scene_index": 0,
                        "concept_id": "vector_space",
                        "narration": "Let's explore what a vector space is...",
                        "visual_intent": {
                            "mathematical_objects": [
                                {
                                    "object_type": "equation",
                                    "content": r"\mathbb{R}^n",
                                    "style": {"color": "blue", "position": "center"}
                                }
                            ],
                            "transformations": [
                                {
                                    "transformation_type": "fade_in",
                                    "target_object": "equation_0",
                                    "parameters": {},
                                    "timing": 0.5
                                }
                            ],
                            "emphasis_points": []
                        },
                        "duration_estimate": 45.0,
                        "dependencies": []
                    }
                ]
            }
        }
    )


# Sentinel that tells mypy / the old import that storyboard.py loaded cleanly.
__all__ = [
    "PDFMetadataStoryboard",
    "Configuration",
    "ConceptHierarchy",
    "ObjectStyle",
    "MathematicalObject",
    "Transformation",
    "EmphasisPoint",
    "SceneState",
    "VisualBeat",
    "VisualIntent",
    "Scene",
    "StoryboardJSON",
]
