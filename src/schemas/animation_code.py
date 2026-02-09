"""Animation code object schema for Animation Code Generator."""

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnimationCodeObject(BaseModel):
    """
    Animation code object produced by Animation Code Generator.
    
    Contains executable Manim code for a single scene.
    """
    
    scene_id: str = Field(..., description="Unique identifier for the scene")
    code: str = Field(..., description="Python code for the Manim scene")
    imports: List[str] = Field(
        ...,
        description="Array of import statements required for the code"
    )
    class_name: str = Field(
        ...,
        description="Name of the Scene class (e.g., 'Scene_001')"
    )
    narration: str = Field(
        ...,
        description="Narration text copied from scene specification"
    )
    duration_estimate: float = Field(
        ...,
        gt=0.0,
        description="Estimated scene duration in seconds"
    )
    generation_timestamp: datetime = Field(
        ...,
        description="ISO8601 timestamp when the code was generated"
    )
    
    @field_validator('scene_id')
    @classmethod
    def validate_scene_id(cls, v: str) -> str:
        """Ensure scene_id is not empty."""
        if not v or not v.strip():
            raise ValueError("scene_id cannot be empty")
        return v
    
    @field_validator('class_name')
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        """Ensure class_name is not empty and is a valid Python identifier."""
        if not v or not v.strip():
            raise ValueError("class_name cannot be empty")
        if not v.isidentifier():
            raise ValueError(f"class_name must be a valid Python identifier, got '{v}'")
        return v
    
    @field_validator('code')
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Ensure code is not empty."""
        if not v or not v.strip():
            raise ValueError("code cannot be empty")
        return v
    
    @field_validator('imports')
    @classmethod
    def validate_imports(cls, v: List[str]) -> List[str]:
        """Ensure imports list is not empty."""
        if not v:
            raise ValueError("imports list cannot be empty")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scene_id": "intro_001",
                "code": "class Scene_001(Scene):\n    def construct(self):\n        # Scene code here\n        pass",
                "imports": ["from manim import *"],
                "class_name": "Scene_001",
                "narration": "Let's explore what a vector space is...",
                "duration_estimate": 45.0,
                "generation_timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
