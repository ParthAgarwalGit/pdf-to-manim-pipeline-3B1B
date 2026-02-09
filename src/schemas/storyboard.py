"""Storyboard JSON schema for Script/Storyboard Architect."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class MathematicalObject(BaseModel):
    """Mathematical object to be animated in a scene."""
    
    object_type: str = Field(
        ...,
        description="Type of object (equation|graph|geometric_shape|axes|text)"
    )
    content: str = Field(
        ...,
        description="LaTeX for equations, description for other types"
    )
    style: ObjectStyle = Field(
        default_factory=ObjectStyle,
        description="Visual style parameters"
    )
    
    @field_validator('object_type')
    @classmethod
    def validate_object_type(cls, v: str) -> str:
        """Validate object_type is one of the allowed values."""
        allowed = {'equation', 'graph', 'geometric_shape', 'axes', 'text'}
        if v not in allowed:
            raise ValueError(f"object_type must be one of {allowed}, got '{v}'")
        return v


class Transformation(BaseModel):
    """Animation transformation to apply to a mathematical object."""
    
    transformation_type: str = Field(
        ...,
        description="Type of transformation (morph|highlight|move|fade_in|fade_out|scale)"
    )
    target_object: str = Field(
        ...,
        description="Reference to the mathematical object to transform"
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
        """Validate transformation_type is one of the allowed values."""
        allowed = {'morph', 'highlight', 'move', 'fade_in', 'fade_out', 'scale'}
        if v not in allowed:
            raise ValueError(f"transformation_type must be one of {allowed}, got '{v}'")
        return v


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
    dependencies: List[str] = Field(
        default_factory=list,
        description="Array of scene_ids that this scene depends on"
    )
    difficulty_level: Optional[str] = Field(
        None,
        description="Difficulty level of the scene content"
    )
    
    @field_validator('scene_id')
    @classmethod
    def validate_scene_id(cls, v: str) -> str:
        """Ensure scene_id is not empty."""
        if not v or not v.strip():
            raise ValueError("scene_id cannot be empty")
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
                                    "content": "\\mathbb{R}^n",
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
