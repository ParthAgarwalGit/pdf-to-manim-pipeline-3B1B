"""Scene Controller Agent for managing deterministic scene iteration.

This agent orchestrates the processing of scenes from a storyboard, maintaining
shared context and isolating failures to individual scenes.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.scene_processing import (
    FailedScene,
    SceneProcessingInput,
    SceneProcessingReport,
    SceneSpec,
    SharedContext,
    VisualStyle,
)
from src.schemas.storyboard import Scene, StoryboardJSON


class SceneControllerInput(AgentInput):
    """Input for Scene Controller Agent.
    
    Attributes:
        storyboard: Sanitized StoryboardJSON object
        output_directory: Path to output directory for generated files
        animation_generator: Animation Generator agent instance (for scene processing)
    """
    
    def __init__(
        self,
        storyboard: StoryboardJSON,
        output_directory: str,
        animation_generator: Optional[Any] = None
    ):
        self.storyboard = storyboard
        self.output_directory = output_directory
        self.animation_generator = animation_generator


class SceneControllerOutput(AgentOutput):
    """Output from Scene Controller Agent.
    
    Attributes:
        report: Scene processing report with success/failure statistics
        animation_code_objects: List of successfully generated animation code objects
        shared_context: Final shared context after all scenes processed
    """
    
    def __init__(
        self,
        report: SceneProcessingReport,
        animation_code_objects: List[AnimationCodeObject],
        shared_context: SharedContext
    ):
        self.report = report
        self.animation_code_objects = animation_code_objects
        self.shared_context = shared_context


class SceneControllerAgent(Agent):
    """Agent responsible for managing deterministic scene iteration.
    
    The Scene Controller:
    - Extracts scene array from sanitized Storyboard JSON
    - Initializes shared context (concept_definitions, visual_style, variable_bindings)
    - Iterates over scenes in deterministic order (index 0 to N-1)
    - Passes scene specification + context to Animation Generator
    - Updates context after successful scene generation
    - Isolates failures (continues to next scene on failure)
    - Generates scene processing report with success/failure statistics
    """
    
    def __init__(self, max_retries_per_scene: int = 3):
        """Initialize Scene Controller Agent.
        
        Args:
            max_retries_per_scene: Maximum retry attempts for individual scene generation
        """
        self.max_retries_per_scene = max_retries_per_scene
    
    def execute(self, input_data: SceneControllerInput) -> SceneControllerOutput:
        """Execute scene iteration and processing.
        
        Args:
            input_data: SceneControllerInput with storyboard and output directory
            
        Returns:
            SceneControllerOutput with processing report and animation code objects
            
        Raises:
            AgentExecutionError: For unrecoverable failures
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be SceneControllerInput with storyboard and output_directory",
                {"input_type": type(input_data).__name__}
            )
        
        storyboard = input_data.storyboard
        output_directory = input_data.output_directory
        animation_generator = input_data.animation_generator
        
        # Extract scene array
        scenes = storyboard.scenes
        total_scenes = len(scenes)
        
        # Initialize shared context
        shared_context = self._initialize_context(storyboard)
        
        # Track results
        animation_code_objects: List[AnimationCodeObject] = []
        failed_scenes: List[FailedScene] = []
        
        # Iterate over scenes in deterministic order (Requirements 10.2, 10.3)
        # Process scenes by array index (0 to N-1), no shuffling or reordering
        for scene_index, scene in enumerate(scenes):
            try:
                # Construct scene processing input
                scene_spec = self._scene_to_spec(scene)
                processing_input = SceneProcessingInput(
                    scene_spec=scene_spec,
                    context=copy.deepcopy(shared_context),  # Pass read-only copy
                    scene_index=scene_index
                )
                
                # Process scene with Animation Generator
                if animation_generator is not None:
                    animation_code = self._process_scene_with_generator(
                        animation_generator,
                        processing_input,
                        scene.scene_id
                    )
                else:
                    # Mock implementation for testing without Animation Generator
                    animation_code = self._mock_process_scene(processing_input)
                
                # Success: update context and collect result
                self._update_context(shared_context, scene, scene_spec)
                animation_code_objects.append(animation_code)
                
            except Exception as e:
                # Failure: log and continue to next scene
                failed_scenes.append(FailedScene(
                    scene_id=scene.scene_id,
                    error=str(e)
                ))
        
        # Generate scene processing report
        successful_scenes = total_scenes - len(failed_scenes)
        report = SceneProcessingReport(
            total_scenes=total_scenes,
            successful_scenes=successful_scenes,
            failed_scenes=failed_scenes,
            output_directory=output_directory
        )
        
        return SceneControllerOutput(
            report=report,
            animation_code_objects=animation_code_objects,
            shared_context=shared_context
        )
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is SceneControllerInput with required fields.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, SceneControllerInput):
            return False
        if not isinstance(input_data.storyboard, StoryboardJSON):
            return False
        if not isinstance(input_data.output_directory, str):
            return False
        return True
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for Scene Controller.
        
        Returns:
            RetryPolicy with exponential backoff for transient failures
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            retryable_errors=["ANIMATION_GENERATOR_UNAVAILABLE", "TRANSIENT_FAILURE"]
        )
    
    def _initialize_context(self, storyboard: StoryboardJSON) -> SharedContext:
        """Initialize shared context from storyboard configuration.
        
        Args:
            storyboard: StoryboardJSON object
            
        Returns:
            Initialized SharedContext
        """
        # Initialize with default 3Blue1Brown-style colors
        default_colors = {
            "primary": "#58C4DD",      # Blue
            "secondary": "#83C167",    # Green
            "background": "#000000",   # Black
            "text": "#FFFFFF",         # White
            "emphasis": "#FC6255"      # Red
        }
        
        visual_style = VisualStyle(
            colors=default_colors,
            font_size=24,
            animation_speed=1.0
        )
        
        return SharedContext(
            concept_definitions={},
            visual_style=visual_style,
            variable_bindings={}
        )
    
    def _scene_to_spec(self, scene: Scene) -> SceneSpec:
        """Convert Scene from storyboard to SceneSpec.
        
        Args:
            scene: Scene object from storyboard
            
        Returns:
            SceneSpec object
        """
        return SceneSpec(
            scene_id=scene.scene_id,
            narration=scene.narration,
            visual_intent=scene.visual_intent.model_dump(),
            duration_estimate=scene.duration_estimate
        )
    
    def _process_scene_with_generator(
        self,
        animation_generator: Any,
        processing_input: SceneProcessingInput,
        scene_id: str
    ) -> AnimationCodeObject:
        """Process scene using Animation Generator with retry logic.
        
        Args:
            animation_generator: Animation Generator agent instance
            processing_input: SceneProcessingInput for the scene
            scene_id: Scene identifier for error reporting
            
        Returns:
            AnimationCodeObject from successful generation
            
        Raises:
            Exception: If generation fails after all retries
        """
        last_error = None
        
        for attempt in range(self.max_retries_per_scene):
            try:
                # Call Animation Generator's execute method
                result = animation_generator.execute(processing_input)
                
                # Check if result is an error object
                if hasattr(result, 'error'):
                    raise Exception(result.error)
                
                # Success
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries_per_scene - 1:
                    # Will retry
                    continue
        
        # All retries exhausted
        raise Exception(f"Scene generation failed after {self.max_retries_per_scene} attempts: {last_error}")
    
    def _mock_process_scene(
        self,
        processing_input: SceneProcessingInput
    ) -> AnimationCodeObject:
        """Mock scene processing for testing without Animation Generator.
        
        Args:
            processing_input: SceneProcessingInput for the scene
            
        Returns:
            Mock AnimationCodeObject
        """
        from datetime import datetime
        
        scene_spec = processing_input.scene_spec
        scene_id = scene_spec.scene_id
        
        # Generate mock Manim code
        class_name = f"Scene_{scene_id.replace('-', '_')}"
        code = f'''class {class_name}(Scene):
    def construct(self):
        # Mock animation code for {scene_id}
        text = Text("{scene_spec.narration[:50]}...")
        self.play(Write(text))
        self.wait({scene_spec.duration_estimate})
'''
        
        return AnimationCodeObject(
            scene_id=scene_id,
            code=code,
            imports=["from manim import *"],
            class_name=class_name,
            narration=scene_spec.narration,
            duration_estimate=scene_spec.duration_estimate,
            generation_timestamp=datetime.now()
        )
    
    def _update_context(
        self,
        context: SharedContext,
        scene: Scene,
        scene_spec: SceneSpec
    ) -> None:
        """Update shared context after successful scene generation.
        
        Args:
            context: SharedContext to update (modified in place)
            scene: Scene object from storyboard
            scene_spec: SceneSpec used for generation
        """
        # Update concept definitions if this scene introduces a concept
        if scene.concept_id:
            # Extract definition from narration (simplified heuristic)
            if "is defined as" in scene.narration.lower() or "we define" in scene.narration.lower():
                context.concept_definitions[scene.concept_id] = scene.narration[:200]
        
        # Update variable bindings from mathematical objects
        visual_intent = scene.visual_intent
        for math_obj in visual_intent.mathematical_objects:
            if math_obj.object_type == "equation":
                content = math_obj.content
                # Simple heuristic: if equation contains '=', extract variable name
                if '=' in content:
                    parts = content.split('=')
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        # Store simple variable bindings (e.g., "x", "y", "f(x)")
                        if len(var_name) < 20:  # Reasonable variable name length
                            context.variable_bindings[var_name] = content
