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
    NarrativeContext,
    SceneProcessingInput,
    SceneProcessingReport,
    SceneSpec,
    SharedContext,
    VisualStyle,
)
from src.schemas.storyboard import (
    BBox,
    Configuration,
    ConceptHierarchy,
    EmphasisPoint,
    MathematicalObject,
    ObjectStyle,
    PDFMetadataStoryboard,
    Scene,
    SceneState,
    StoryboardJSON,
    TimeInterval,
    Transformation,
    VisualBeat,
    VisualIntent,
)
from src.schemas.video_script import Beat, VideoScript

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
    "BBox",
    "TimeInterval",
    "Transformation",
    "EmphasisPoint",
    "SceneState",
    "VisualBeat",
    "VisualIntent",
    "Scene",
    "StoryboardJSON",
    # Animation Code
    "AnimationCodeObject",
    # Scene Processing
    "VisualStyle",
    "SharedContext",
    "NarrativeContext",
    "SceneSpec",
    "SceneProcessingInput",
    "FailedScene",
    "SceneProcessingReport",
    # Output Manifest
    "OutputFile",
    "FailedSceneManifest",
    "OutputManifest",
    # Video Script (unified narration + chunker contract)
    "Beat",
    "VideoScript",
]
