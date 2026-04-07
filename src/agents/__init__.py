"""Agent implementations for the PDF to Manim pipeline.

12-agent pipeline (10 stages + 2 sub-agents):
 1. IngestAgent - Local file ingestion
 2. ValidationAgent - PDF validation
 3. VisionOCRAgent - Text/math/diagram extraction
 4. ScriptArchitectAgent - Pedagogical scene generation
 5. JSONSanitizerAgent - Schema compliance
 6. SceneControllerAgent - Deterministic scene iteration
 7. AnimationCodeGeneratorAgent - Manim code generation (sub-agent of SceneController)
 8. ManimCodeValidatorAgent - Code syntax validation (sub-agent of SceneController)
 9. PersistenceAgent - File system output (.py, storyboard, narration)
10. VideoRendererAgent - Manim rendering to video (.mp4)
11. TTSAgent - Text-to-speech narration (.mp3)
12. AudioVideoComposerAgent - Merge per-scene .mp4 + .mp3 → final video
"""

from .base import Agent, AgentInput, AgentOutput, RetryPolicy

__all__ = [
    "Agent",
    "AgentInput",
    "AgentOutput",
    "RetryPolicy",
]
