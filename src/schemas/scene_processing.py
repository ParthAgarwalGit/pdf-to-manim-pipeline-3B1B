"""Scene processing schemas for Scene Controller."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas.storyboard import VisualIntent


class VisualStyle(BaseModel):
    """Visual style parameters for consistent appearance across scenes."""
    
    colors: Dict[str, str] = Field(
        default_factory=dict,
        description="Color mappings (e.g., primary, secondary, background)"
    )
    font_size: int = Field(
        default=24,
        ge=8,
        le=72,
        description="Font size for text elements"
    )
    animation_speed: float = Field(
        default=1.0,
        gt=0.0,
        description="Animation speed multiplier (default 1.0)"
    )


class NarrativeContext(BaseModel):
    """Tracks what the viewer has already seen so later scenes do not re-explain.

    The Scene Controller updates this after each successful scene. Per-scene
    LLM prompts read it to know which concepts to *reference* (not redefine)
    and which on-screen objects to carry forward.

    Field semantics:
      - prior_scene_summaries: one short summary per prior scene, in order.
        Bounded to keep token cost flat across long videos (see F4 mitigation
        in 2026-04-26-unified-script-chunked-render plan).
      - concepts_introduced: canonical concept labels already taught;
        first-introduction order; uncapped (small per entry, load-bearing for
        the dedup rule).
      - running_storyline: 2-3 sentence rolling synopsis.
      - live_objects_on_screen: object_id -> short description of objects
        still visible at the END of the most recent scene.
      - prior_equations: LaTeX of equations that have already appeared so the
        visual planner can mark re-appearances as continuity_action="reuse"
        instead of redefining them (F8 mitigation).
    """

    prior_scene_summaries: List[str] = Field(
        default_factory=list,
        description="One short summary per prior scene, in order"
    )
    concepts_introduced: List[str] = Field(
        default_factory=list,
        description="Canonical concept labels the viewer has already been taught"
    )
    running_storyline: str = Field(
        default="",
        description="2-3 sentence rolling synopsis of the video so far"
    )
    live_objects_on_screen: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of object_id -> short description of objects visible "
                    "at the END of the most recent scene (carried into next chunk)"
    )
    prior_equations: List[str] = Field(
        default_factory=list,
        description="LaTeX of equations already shown; used to dedupe the "
                    "visual layer when concepts are referenced again"
    )


class SharedContext(BaseModel):
    """
    Shared context propagated across scene processing.

    Contains concept definitions, visual style, and variable bindings
    accumulated from previous scenes, plus helper-friendly registries for
    coordinate systems, notation objects, and character roles.
    """

    concept_definitions: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of concept_id to definition text"
    )
    visual_style: VisualStyle = Field(
        default_factory=VisualStyle,
        description="Visual style parameters for consistent appearance"
    )
    variable_bindings: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of variable names to values or references"
    )
    coordinate_system_bindings: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of coordinate_system_id to semantic kind (e.g. 'number_plane')"
    )
    notation_registry: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of notation_key to object_id for stable notation reuse"
    )
    character_registry: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of character_role to object_id for persistent Pi creatures"
    )
    color_bindings: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of variable name to hex color for cross-scene color consistency"
    )
    narrative: NarrativeContext = Field(
        default_factory=NarrativeContext,
        description="Narrative continuity state propagated across scenes — "
                    "what the viewer has already seen, which concepts have been taught, "
                    "and which objects carry over visually"
    )


class SceneSpec(BaseModel):
    """Scene specification extracted from storyboard."""
    
    scene_id: str = Field(..., description="Unique identifier for the scene")
    narration: str = Field(..., description="Narration text for the scene")
    visual_intent: VisualIntent = Field(
        ...,
        description="Visual intent specification with mathematical objects and transformations"
    )
    duration_estimate: float = Field(
        ...,
        gt=0.0,
        description="Estimated scene duration in seconds"
    )
    audio_duration_seconds: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Measured narration duration used as the animation target",
    )
    beat_timings: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Timing track (beat_id -> [start, end] s) for the sync contract",
    )
    
    @field_validator('scene_id')
    @classmethod
    def validate_scene_id(cls, v: str) -> str:
        """Ensure scene_id is not empty."""
        if not v or not v.strip():
            raise ValueError("scene_id cannot be empty")
        return v

    @field_validator('narration')
    @classmethod
    def validate_narration(cls, v: str) -> str:
        """Ensure narration is not empty."""
        if not v or not v.strip():
            raise ValueError("narration cannot be empty")
        return v


class SceneProcessingInput(BaseModel):
    """
    Input to Animation Code Generator for a single scene.

    Contains scene specification, shared context, and scene index.
    """

    scene_spec: SceneSpec = Field(
        ...,
        description="Scene specification from storyboard"
    )
    context: SharedContext = Field(
        ...,
        description="Shared context from previous scenes"
    )
    scene_index: int = Field(
        ...,
        ge=0,
        description="Sequential index of the scene in the array"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scene_spec": {
                    "scene_id": "intro_001",
                    "narration": "Let's explore what a vector space is...",
                    "visual_intent": {
                        "mathematical_objects": [],
                        "transformations": [],
                        "emphasis_points": []
                    },
                    "duration_estimate": 45.0
                },
                "context": {
                    "concept_definitions": {},
                    "visual_style": {
                        "colors": {"primary": "#58C4DD", "secondary": "#83C167"},
                        "font_size": 24,
                        "animation_speed": 1.0
                    },
                    "variable_bindings": {}
                },
                "scene_index": 0
            }
        }
    )


class FailedScene(BaseModel):
    """Information about a scene that failed to generate."""
    
    scene_id: str = Field(..., description="Unique identifier for the failed scene")
    error: str = Field(..., description="Error message describing the failure")


class SceneProcessingReport(BaseModel):
    """
    Report generated by Scene Controller after processing all scenes.
    
    Contains success/failure statistics and details about failed scenes.
    """
    
    total_scenes: int = Field(
        ...,
        ge=0,
        description="Total number of scenes in the storyboard"
    )
    successful_scenes: int = Field(
        ...,
        ge=0,
        description="Number of scenes that generated successfully"
    )
    failed_scenes: List[FailedScene] = Field(
        default_factory=list,
        description="Array of scenes that failed to generate"
    )
    output_directory: str = Field(
        ...,
        description="Path to the output directory for generated files"
    )
    
    @field_validator('successful_scenes')
    @classmethod
    def validate_successful_scenes(cls, v: int, info) -> int:
        """Ensure successful_scenes does not exceed total_scenes."""
        total = info.data.get('total_scenes', 0)
        if v > total:
            raise ValueError(
                f"successful_scenes ({v}) cannot exceed total_scenes ({total})"
            )
        return v
    
    @property
    def scene_success_rate(self) -> float:
        """Calculate the success rate as a fraction (0-1)."""
        if self.total_scenes == 0:
            return 0.0
        return self.successful_scenes / self.total_scenes
    
    @property
    def status(self) -> str:
        """
        Determine overall status based on success rate.
        
        Returns:
            "SUCCESS" if all scenes succeeded
            "PARTIAL_SUCCESS" if some scenes succeeded
            "CRITICAL_FAILURE" if >50% of scenes failed
            "FAILURE" if all scenes failed
        """
        if self.successful_scenes == self.total_scenes:
            return "SUCCESS"
        elif self.successful_scenes == 0:
            return "FAILURE"
        elif self.scene_success_rate < 0.5:
            return "CRITICAL_FAILURE"
        else:
            return "PARTIAL_SUCCESS"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_scenes": 10,
                "successful_scenes": 9,
                "failed_scenes": [
                    {
                        "scene_id": "proof_005",
                        "error": "LaTeX syntax error in equation"
                    }
                ],
                "output_directory": "output/linear_algebra/2024-01-15T10-30-00"
            }
        }
    )
