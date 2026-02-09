"""Pydantic schemas for data contracts between agents."""

from src.schemas.animation_code import AnimationCodeObject
from src.schemas.file_reference import FileReference
from src.schemas.ocr_output import (
    BoundingBox,
    Diagram,
    MathExpression,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)
from src.schemas.output_manifest import (
    FailedSceneManifest,
    OutputFile,
    OutputManifest,
)
from src.schemas.scene_processing import (
    FailedScene,
    SceneProcessingInput,
    SceneProcessingReport,
    SceneSpec,
    SharedContext,
    VisualStyle,
)
from src.schemas.storyboard import (
    Configuration,
    ConceptHierarchy,
    EmphasisPoint,
    MathematicalObject,
    ObjectStyle,
    PDFMetadataStoryboard,
    Scene,
    StoryboardJSON,
    Transformation,
    VisualIntent,
)

__all__ = [
    # File Reference
    "FileReference",
    # OCR Output
    "BoundingBox",
    "TextBlock",
    "MathExpression",
    "Diagram",
    "Page",
    "PDFMetadata",
    "OCROutput",
    # Storyboard
    "PDFMetadataStoryboard",
    "Configuration",
    "ConceptHierarchy",
    "ObjectStyle",
    "MathematicalObject",
    "Transformation",
    "EmphasisPoint",
    "VisualIntent",
    "Scene",
    "StoryboardJSON",
    # Animation Code
    "AnimationCodeObject",
    # Scene Processing
    "VisualStyle",
    "SharedContext",
    "SceneSpec",
    "SceneProcessingInput",
    "FailedScene",
    "SceneProcessingReport",
    # Output Manifest
    "OutputFile",
    "FailedSceneManifest",
    "OutputManifest",
]
