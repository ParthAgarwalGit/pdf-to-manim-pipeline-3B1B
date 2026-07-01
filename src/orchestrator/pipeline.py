"""Pipeline Orchestrator for PDF to Manim conversion.

This module implements the main orchestrator that sequences agent invocations
in a 10-stage linear pipeline:

    1. Ingestion            — Local file I/O, extract metadata
    2. Validation           — Verify PDF type and structure (abort on failure)
    3. Vision+OCR           — Extract text, math, and diagrams from PDF
    4. Script Architect     — Generate pedagogical storyboard with LLM
    5. JSON Sanitizer       — Enforce Pydantic schema compliance (abort on failure)
    6. Scene Controller     — Iterate scenes, generate Manim code with LLM
    7. Persistence          — Write .py scene files + storyboard + narration to disk
    8. Video Renderer       — Run Manim subprocess on each .py file → .mp4
    9. TTS Generation       — Generate narration audio via OpenAI TTS → .mp3
   10. Audio-Video Compose  — Merge per-scene .mp4 + .mp3, concatenate final video

Error Handling Strategy:
- **Abort**: Validation and sanitization errors terminate pipeline immediately
  (non-retryable, deterministic failures indicating invalid input)
- **Retry**: Transient failures (API timeouts, rate limits) use exponential backoff
  with configurable max attempts per agent
- **Continue**: Scene-level failures in rendering, TTS, and composition are isolated;
  the pipeline logs and skips failed scenes without crashing
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy, StepConstraints
from src.orchestrator.retry_policy import calculate_backoff_delay, is_retryable_error
from src.orchestrator.logger import StructuredJSONLogger
from src.agents.ingestion import CloudStorageEvent, IngestAgent, IngestionError, IngestionOutput
from src.agents.validation import ValidationAgent, ValidationError
from src.agents.vision_ocr import VisionOCRAgent
from src.agents.document_segmenter import DocumentSegmenterAgent, DocumentSegmenterInput
from src.agents.script_architect import ScriptArchitectAgent, ScriptArchitectInput
from src.agents.storyboard_grader import StoryboardGraderAgent, StoryboardGraderFailure, StoryboardGraderInput, StoryboardGraderOutput
from src.agents.json_sanitizer import JSONSanitizerAgent, JSONSanitizerInput, SanitizationError
from src.agents.scene_controller import SceneControllerAgent, SceneControllerInput, SceneControllerOutput
from src.agents.manim_code_validator import ManimCodeValidatorAgent, ManimValidatorFailure, ManimValidatorInput, ManimValidatorOutput
from src.agents.video_renderer import VideoRendererAgent, VideoRendererInput, VideoRendererOutput, VideoRendererFailure
from src.agents.tts_agent import TTSAgent, TTSInput, TTSOutput, TTSError
from src.agents.audio_video_composer import (
    AudioVideoComposerAgent, AudioVideoComposerInput, AudioVideoComposerOutput,
    AudioVideoComposerFailure, SceneMediaPair,
)
from src.agents.video_quality_reviewer import VideoQualityReviewerAgent, VideoQualityReviewerInput
from src.agents.persistence import PersistenceAgent, PersistenceInput
from src.schemas.file_reference import FileReference
from src.schemas.output_manifest import OutputFile, OutputManifest
from src.schemas.scene_processing import FailedScene, SceneProcessingReport
from src.schemas.storyboard import Configuration


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        verbosity: Narration verbosity level (low/medium/high)
        depth: Content depth level (introductory/intermediate/advanced)
        audience: Target audience level (high-school/undergraduate/graduate)
        base_output_dir: Base directory for output files
        run_id: Immutable identifier for this run, formatted as
                ``YYYY-MM-DDTHH-MM-SS``.  Generated once at the start of
                ``run_pipeline()`` if not supplied, then propagated to every
                stage so all artifacts land in the same directory.
        enable_ocr_debug: Whether to write OCR output for debugging
        strict: If True, agents raise errors when API keys are missing
                instead of falling back to mock/heuristic outputs
        enable_tts: If True, run TTS generation stage
        enable_render: If True, run video rendering stage
        dry_run: If True, stop after storyboard generation (skip code, render, TTS, composition)
        tts_voice: TTS voice selection (alloy/echo/fable/onyx/nova/shimmer)
        render_quality: Video render quality (low/medium/high)
    """
    verbosity: str = "medium"
    depth: str = "intermediate"
    audience: str = "undergraduate"
    base_output_dir: str = "output"
    run_id: Optional[str] = None   # set once at pipeline start; never reset
    enable_ocr_debug: bool = False
    enable_quality_loops: bool = True
    max_quality_retries: int = 2
    storyboard_quality_threshold: float = 6.0
    granularity: str = "balanced"
    enable_manim_dry_run_validation: bool = False
    enable_final_video_review: bool = True
    strict: bool = False
    enable_tts: bool = True
    enable_render: bool = True
    dry_run: bool = False
    tts_voice: str = "alloy"
    render_quality: str = "medium"
    
    def to_storyboard_config(self) -> Configuration:
        """Convert to Configuration schema for storyboard generation."""
        return Configuration(
            verbosity=self.verbosity,
            depth=self.depth,
            audience=self.audience,
            granularity=self.granularity,
        )


class PipelineAbortError(Exception):
    """Exception raised when pipeline must abort due to non-retryable error.
    
    Attributes:
        stage: Pipeline stage where abort occurred
        error_code: Machine-readable error code
        message: Human-readable error message
        context: Additional context about the failure
    """
    
    def __init__(self, stage: str, error_code: str, message: str, context: Optional[Dict[str, Any]] = None):
        self.stage = stage
        self.error_code = error_code
        self.message = message
        self.context = context or {}
        super().__init__(f"Pipeline aborted at {stage}: [{error_code}] {message}")


class Orchestrator:
    """Main orchestrator for the PDF-to-Manim pipeline.

    Sequences 10 agents in order:
      1. Ingestion → 2. Validation → 3. Vision+OCR → 4. Script Architect →
      5. JSON Sanitizer → 6. Scene Controller → 7. Persistence →
      8. Video Renderer → 9. TTS → 10. Audio-Video Compose

    Persistence runs BEFORE video rendering so that .py scene files
    exist on disk for Manim to execute.  Video rendering, TTS, and
    audio-video composition all use per-scene error recovery — failed
    scenes are logged and skipped.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        log_file: Optional[str] = None
    ):
        """Initialize the orchestrator.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
            log_file: Path to log file (if None, logs to console only)
        """
        self.config = config or PipelineConfig()
        self.log_file = log_file
        self.structured_logger: Optional[StructuredJSONLogger] = None

        # Initialize agents
        # Agents that call external APIs accept strict=True to fail fast
        # when API keys are missing, instead of falling back to mock outputs.
        strict = self.config.strict
        self.ingestion_agent = IngestAgent()
        self.validation_agent = ValidationAgent()
        self.vision_ocr_agent = VisionOCRAgent(strict=strict)
        self.document_segmenter_agent = DocumentSegmenterAgent()
        self.script_architect_agent = ScriptArchitectAgent(strict=strict)
        self.storyboard_grader_agent = StoryboardGraderAgent(strict=strict)
        self.json_sanitizer_agent = JSONSanitizerAgent()
        self.scene_controller_agent = SceneControllerAgent()
        self.manim_code_validator_agent = ManimCodeValidatorAgent()
        self.video_renderer_agent = VideoRendererAgent()
        self.tts_agent = TTSAgent(strict=strict)
        self.audio_video_composer_agent = AudioVideoComposerAgent()
        self.video_quality_reviewer_agent = VideoQualityReviewerAgent()
        self.persistence_agent = PersistenceAgent(base_output_dir=self.config.base_output_dir)
        self._quality_reports: Dict[str, Any] = {}

        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup structured logging for the pipeline."""
        # Configure root logger
        log_level = logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
        
        logger.setLevel(log_level)
    
    def run_pipeline(
        self,
        pdf_file_path: str,
        event_source: str = 's3'
    ) -> OutputManifest:
        """Run the complete PDF-to-Manim pipeline.
        
        Args:
            pdf_file_path: Path to PDF file (local or cloud storage URL)
            event_source: Cloud storage event source ('s3', 'gcs', 'azure', or 'local')
            
        Returns:
            OutputManifest with all generated files and metadata
            
        Raises:
            PipelineAbortError: If pipeline must abort due to non-retryable error
            AgentExecutionError: If an agent encounters an unrecoverable failure
        """
        logger.info(f"Starting pipeline for PDF: {pdf_file_path}")
        pipeline_start_time = time.time()
        self._quality_reports = {}

        # ── Run identity ─────────────────────────────────────────────
        # Generate run_id ONCE here, before any stage executes.
        # Every stage that needs an output path must use self.config.run_id
        # (and therefore self._run_output_dir once we know the pdf stem).
        # No stage is permitted to call datetime.now() for path decisions.
        if not self.config.run_id:
            self.config.run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        logger.info(f"Run ID: {self.config.run_id}")

        # _run_output_dir is set after ingestion gives us the pdf stem
        self._run_output_dir: Optional[str] = None

        # We'll initialize the structured logger after we know the output directory
        # For now, use None (console logging only)
        temp_logger = StructuredJSONLogger(output_directory=None)
        
        try:
            # Log pipeline start
            temp_logger.log_pipeline_start(
                pdf_file_path=pdf_file_path,
                config={
                    "verbosity": self.config.verbosity,
                    "depth": self.config.depth,
                    "audience": self.config.audience
                }
            )
            
            # Step 1: Ingestion
            file_ref = self._run_ingestion(pdf_file_path, event_source, temp_logger)
            
            # Step 2: Validation
            validated_ref = self._run_validation(file_ref, temp_logger)
            
            # Now we know the pdf stem — derive the single canonical output directory.
            # run_id was already fixed at the top of this method; we just combine it
            # with the pdf stem.  Every downstream stage uses self._run_output_dir.
            pdf_filename = Path(validated_ref.filename).stem
            self._run_output_dir = str(
                Path(self.config.base_output_dir) / pdf_filename / self.config.run_id
            )
            logger.info(f"Output directory: {self._run_output_dir}")

            # Initialize structured logger pointing at the canonical output dir
            self.structured_logger = StructuredJSONLogger(
                output_directory=self._run_output_dir
            )
            
            # Step 3: Vision+OCR
            ocr_output = self._run_vision_ocr(validated_ref, self.structured_logger)

            # Step 4: Document Segmentation
            source_segments = self._run_document_segmenter(ocr_output, self.structured_logger)
            
            # Step 5: Script/Storyboard Architect
            storyboard = self._run_script_architect(ocr_output, self.structured_logger, source_segments)
            
            # Step 6: JSON Sanitization
            sanitized_storyboard = self._run_json_sanitizer(storyboard, self.structured_logger)

            # ── dry_run gate ─────────────────────────────────────
            if self.config.dry_run:
                logger.info("Dry run: stopping after sanitized storyboard")
                # Still run persistence to write storyboard/planning artifacts to disk
                manifest = self._run_persistence(
                    None,  # No scene output in dry run
                    sanitized_storyboard,
                    ocr_output if self.config.enable_ocr_debug else None,
                    validated_ref,
                    self.structured_logger
                )
                manifest = self._finalize_manifest(
                    manifest,
                    video_output=None,
                    tts_output=None,
                    composition_output=None,
                    video_review_output=None,
                )
                pipeline_duration = time.time() - pipeline_start_time
                scene_success_rate = (
                    manifest.successful_scenes / manifest.total_scenes
                    if manifest.total_scenes > 0 else 1.0
                )
                self.structured_logger.log_pipeline_complete(
                    duration_seconds=pipeline_duration,
                    output_directory=manifest.output_directory,
                    status=manifest.status,
                    scene_success_rate=scene_success_rate
                )
                logger.info(f"Dry run completed in {pipeline_duration:.2f}s")
                return manifest

            # Step 6: Scene Processing
            scene_output = self._run_scene_controller(
                sanitized_storyboard,
                validated_ref,
                self.structured_logger
            )

            # Step 7: Persistence — write .py scene files, storyboard, narration to disk
            #         (MUST happen before video rendering so Manim has files to render)
            manifest = self._run_persistence(
                scene_output,
                sanitized_storyboard,
                ocr_output if self.config.enable_ocr_debug else None,
                validated_ref,
                self.structured_logger
            )

            # Steps 8+9: Video Rendering and TTS — parallel fan-out.
            # Both only need outputs from Persistence (step 7) so they are
            # fully independent.  The logger is thread-safe (uses a Lock).
            video_output = None
            tts_output = None

            if self.config.enable_render or self.config.enable_tts:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures: Dict[str, Any] = {}
                    if self.config.enable_render:
                        futures["render"] = executor.submit(
                            self._run_video_renderer,
                            manifest,
                            self.structured_logger,
                        )
                    else:
                        logger.info("Render skipped (enable_render=False)")
                    if self.config.enable_tts:
                        futures["tts"] = executor.submit(
                            self._run_tts_generation_logged,
                            sanitized_storyboard,
                            manifest,
                            self.structured_logger,
                        )
                    else:
                        logger.info("TTS skipped (enable_tts=False)")
                    # executor.__exit__ waits for both futures before we proceed
                    video_output = futures["render"].result() if "render" in futures else None
                    tts_output = futures["tts"].result() if "tts" in futures else None
            else:
                logger.info("Both render and TTS skipped")

            # Step 10: Audio-Video Composition (requires both render and TTS)
            composition_output = None
            if self.config.enable_render and self.config.enable_tts:
                composition_output = self._run_audio_video_composition(
                    video_output,
                    tts_output,
                    manifest,
                    self.structured_logger
                )
            else:
                logger.info("Audio-video composition skipped (missing render or TTS)")

            # Final video review (only if we have rendered output)
            video_review_output = None
            if self.config.enable_render:
                video_review_output = self._run_final_video_review(
                    sanitized_storyboard,
                    scene_output,
                    manifest,
                    video_output,
                    tts_output,
                    composition_output,
                    self.structured_logger
                )

            manifest = self._finalize_manifest(
                manifest,
                video_output,
                tts_output,
                composition_output,
                video_review_output,
            )

            # Log pipeline completion
            pipeline_duration = time.time() - pipeline_start_time
            scene_success_rate = (
                manifest.successful_scenes / manifest.total_scenes
                if manifest.total_scenes > 0 else 1.0
            )
            
            self.structured_logger.log_pipeline_complete(
                duration_seconds=pipeline_duration,
                output_directory=manifest.output_directory,
                status=manifest.status,
                scene_success_rate=scene_success_rate
            )
            
            logger.info(
                f"Pipeline completed successfully in {pipeline_duration:.2f}s",
                extra={
                    "event": "pipeline_complete",
                    "duration_seconds": pipeline_duration,
                    "output_directory": manifest.output_directory,
                    "status": manifest.status
                }
            )
            
            return manifest
            
        except PipelineAbortError as e:
            # Log pipeline error
            if self.structured_logger:
                self.structured_logger.log_pipeline_error(
                    error_type="PipelineAbortError",
                    error_message=str(e),
                    stage=e.stage
                )
            # Re-raise abort errors
            raise
        except Exception as e:
            # Log unexpected errors
            if self.structured_logger:
                self.structured_logger.log_pipeline_error(
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            
            logger.error(
                f"Pipeline failed with unexpected error: {str(e)}",
                extra={
                    "event": "pipeline_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            raise
        finally:
            # Close structured logger
            if self.structured_logger:
                self.structured_logger.close()
            temp_logger.close()
    
    def _run_ingestion(
        self,
        pdf_file_path: str,
        event_source: str,
        structured_logger: StructuredJSONLogger
    ) -> FileReference:
        """Run Ingestion Agent with retry logic.
        
        Args:
            pdf_file_path: Path to PDF file
            event_source: Cloud storage event source
            structured_logger: Logger for structured JSON logging
            
        Returns:
            FileReference object
            
        Raises:
            PipelineAbortError: If ingestion fails after retries
        """
        # Construct cloud storage event
        # For local files, create a synthetic event
        if event_source == 'local':
            event_payload = self._create_local_file_event(pdf_file_path)
        else:
            # For cloud storage, assume event_payload is constructed externally
            # This is a simplified version for demonstration
            event_payload = self._create_synthetic_event(pdf_file_path, event_source)
        
        event = CloudStorageEvent(event_payload=event_payload, event_source=event_source)
        
        # Execute with retry
        result = self._execute_with_retry(
            agent=self.ingestion_agent,
            input_data=event,
            stage_name="Ingestion",
            structured_logger=structured_logger
        )
        
        # Check for ingestion error
        if isinstance(result, IngestionError):
            raise PipelineAbortError(
                stage="Ingestion",
                error_code=result.error_code,
                message=result.reason,
                context={"payload_sample": result.event_payload_sample}
            )
        
        # Extract FileReference from IngestionOutput
        if isinstance(result, IngestionOutput):
            return result.file_reference
        
        raise PipelineAbortError(
            stage="Ingestion",
            error_code="UNEXPECTED_OUTPUT",
            message=f"Unexpected output type from Ingestion Agent: {type(result).__name__}"
        )
    
    def _run_validation(
        self,
        file_ref: FileReference,
        structured_logger: StructuredJSONLogger
    ) -> FileReference:
        """Run Validation Agent (no retry - deterministic).
        
        Args:
            file_ref: FileReference from Ingestion Agent
            structured_logger: Logger for structured JSON logging
            
        Returns:
            Validated FileReference
            
        Raises:
            PipelineAbortError: If validation fails
        """
        result = self._execute_agent(
            agent=self.validation_agent,
            input_data=file_ref,
            stage_name="Validation",
            structured_logger=structured_logger
        )
        
        # Check for validation error
        if isinstance(result, ValidationError):
            raise PipelineAbortError(
                stage="Validation",
                error_code=result.error_code,
                message=result.reason,
                context={"file_path": result.file_path}
            )
        
        # Validation succeeded, return validated FileReference
        return result
    
    def _run_vision_ocr(
        self,
        file_ref: FileReference,
        structured_logger: StructuredJSONLogger
    ):
        """Run Vision+OCR Agent with retry logic.
        
        Args:
            file_ref: Validated FileReference
            structured_logger: Logger for structured JSON logging
            
        Returns:
            OCROutput object
        """
        result = self._execute_with_retry(
            agent=self.vision_ocr_agent,
            input_data=file_ref,
            stage_name="Vision+OCR",
            structured_logger=structured_logger
        )
        
        return result

    def _run_document_segmenter(
        self,
        ocr_output,
        structured_logger: StructuredJSONLogger,
    ):
        """Run deterministic document segmentation after OCR."""
        target_words = {
            "coarse": 300,
            "balanced": 150,
            "fine": 90,
        }.get(self.config.granularity, 150)
        input_data = DocumentSegmenterInput(
            ocr_output=ocr_output,
            target_words_per_segment=target_words,
        )
        input_data.constraints = StepConstraints(timeout_seconds=30)
        result = self._execute_agent(
            agent=self.document_segmenter_agent,
            input_data=input_data,
            stage_name="DocumentSegmenter",
            structured_logger=structured_logger,
        )
        return result.segments
    
    def _run_script_architect(
        self,
        ocr_output,
        structured_logger: StructuredJSONLogger,
        source_segments=None,
    ):
        """Run Script Architect Agent with retry logic.
        
        Args:
            ocr_output: OCROutput from Vision+OCR Agent
            structured_logger: Logger for structured JSON logging
            
        Returns:
            StoryboardJSON object (as dict for sanitizer)
        """
        config = self.config.to_storyboard_config()
        source_segments = source_segments or self._run_document_segmenter(
            ocr_output,
            structured_logger,
        )
        review_feedback: List[str] = []
        grading_attempts: List[Dict[str, Any]] = []
        latest_storyboard = None
        max_attempts = max(1, self.config.max_quality_retries + 1)

        for attempt in range(max_attempts):
            input_data = ScriptArchitectInput(
                ocr_output=ocr_output,
                configuration=config,
                segments=source_segments,
                review_feedback=review_feedback,
            )
            input_data.constraints = StepConstraints(budget_tokens=16000, timeout_seconds=180)

            latest_storyboard = self._execute_with_retry(
                agent=self.script_architect_agent,
                input_data=input_data,
                stage_name="ScriptArchitect",
                structured_logger=structured_logger
            )

            if not self.config.enable_quality_loops:
                self._quality_reports["storyboard_review"] = {
                    "enabled": False,
                    "attempts": attempt + 1,
                }
                return latest_storyboard

            grader_input = StoryboardGraderInput(
                storyboard=latest_storyboard,
                source_text_summary=self._summarize_segments_for_review(source_segments),
                source_segments=source_segments,
                pass_threshold=self.config.storyboard_quality_threshold,
            )
            grader_input.constraints = StepConstraints(budget_tokens=4000, timeout_seconds=60)
            grade_result = self._execute_agent(
                agent=self.storyboard_grader_agent,
                input_data=grader_input,
                stage_name="StoryboardReview",
                structured_logger=structured_logger
            )

            report = grade_result.report
            grading_attempts.append(
                {
                    "attempt": attempt + 1,
                    "overall_score": report.overall_score,
                    "passed": report.passed,
                    "summary": report.summary,
                    "improvement_suggestions": report.improvement_suggestions,
                    "criteria": [
                        {
                            "name": criterion.name,
                            "score": criterion.score,
                            "weight": criterion.weight,
                            "feedback": criterion.feedback,
                        }
                        for criterion in report.criteria
                    ],
                }
            )

            if isinstance(grade_result, StoryboardGraderOutput):
                self._quality_reports["storyboard_review"] = {
                    "enabled": True,
                    "attempts": grading_attempts,
                }
                return latest_storyboard

            review_feedback = grade_result.report.improvement_suggestions or [
                criterion.feedback
                for criterion in grade_result.report.criteria
                if criterion.feedback
            ]
            logger.warning(
                "Storyboard quality review failed on attempt %s/%s with score %.2f. Retrying with critique.",
                attempt + 1,
                max_attempts,
                grade_result.report.overall_score,
            )

        self._quality_reports["storyboard_review"] = {
            "enabled": True,
            "attempts": grading_attempts,
            "final_status": "accepted_below_threshold",
        }
        return latest_storyboard
    
    def _run_json_sanitizer(
        self,
        storyboard,
        structured_logger: StructuredJSONLogger
    ):
        """Run JSON Sanitizer Agent (no retry - deterministic).
        
        Args:
            storyboard: StoryboardJSON object (or dict)
            structured_logger: Logger for structured JSON logging
            
        Returns:
            Sanitized StoryboardJSON object
            
        Raises:
            PipelineAbortError: If sanitization fails
        """
        # Convert storyboard to dict if it's a Pydantic model
        if hasattr(storyboard, 'model_dump'):
            storyboard_dict = storyboard.model_dump()
        else:
            storyboard_dict = storyboard
        
        input_data = JSONSanitizerInput(storyboard_dict=storyboard_dict)
        input_data.constraints = StepConstraints(budget_tokens=4000, timeout_seconds=15)

        result = self._execute_agent(
            agent=self.json_sanitizer_agent,
            input_data=input_data,
            stage_name="JSONSanitizer",
            structured_logger=structured_logger
        )
        
        # Check for sanitization error
        if isinstance(result, SanitizationError):
            error_details = "\n".join([
                f"  - {err.field_path}: {err.message}"
                for err in result.errors
            ])
            raise PipelineAbortError(
                stage="JSONSanitizer",
                error_code="SCHEMA_VIOLATION",
                message=f"Storyboard JSON failed schema validation:\n{error_details}",
                context={
                    "error_count": len(result.errors),
                    "attempted_corrections": len(result.attempted_corrections)
                }
            )
        
        # Extract sanitized storyboard
        return result.storyboard
    
    def _run_scene_controller(
        self,
        storyboard,
        file_ref: FileReference,
        structured_logger: StructuredJSONLogger
    ):
        """Run Scene Controller Agent with retry logic.
        
        Args:
            storyboard: Sanitized StoryboardJSON object
            file_ref: FileReference for output directory naming
            structured_logger: Logger for structured JSON logging
            
        Returns:
            SceneControllerOutput object
        """
        # Use the single canonical output directory established at pipeline start.
        # Never generate a new timestamp here — that was the root cause of the
        # split-directory bug.
        output_dir = self._run_output_dir

        # Create input with animation generator
        from src.agents.animation_generator import AnimationCodeGeneratorAgent
        animation_generator = AnimationCodeGeneratorAgent(strict=self.config.strict)
        scene_feedback: Dict[str, List[str]] = {}
        validation_attempts: List[Dict[str, Any]] = []
        max_attempts = max(1, self.config.max_quality_retries + 1)
        latest_result = None  # guaranteed to be assigned in first loop iteration (max_attempts >= 1)

        for attempt in range(max_attempts):
            input_data = SceneControllerInput(
                storyboard=storyboard,
                output_directory=output_dir,
                animation_generator=animation_generator,
                scene_feedback=scene_feedback,
            )
            input_data.constraints = StepConstraints(budget_tokens=32000, timeout_seconds=600)

            latest_result = self._execute_with_retry(
                agent=self.scene_controller_agent,
                input_data=input_data,
                stage_name="SceneController",
                structured_logger=structured_logger
            )

            if not self.config.enable_quality_loops or not latest_result.animation_code_objects:
                self._quality_reports["scene_code_review"] = {
                    "enabled": self.config.enable_quality_loops,
                    "attempts": attempt + 1,
                    "validated_scene_count": len(getattr(latest_result, "animation_code_objects", [])),
                }
                return latest_result

            validator_input = ManimValidatorInput(
                animation_code_objects=latest_result.animation_code_objects,
                enable_dry_run=self.config.enable_manim_dry_run_validation,
            )
            validator_input.constraints = StepConstraints(budget_tokens=8000, timeout_seconds=120)
            validation_result = self._execute_agent(
                agent=self.manim_code_validator_agent,
                input_data=validator_input,
                stage_name="SceneCodeReview",
                structured_logger=structured_logger
            )

            validation_attempts.append(
                self._serialize_scene_validation_attempt(validation_result, attempt + 1)
            )

            if isinstance(validation_result, ManimValidatorOutput):
                self._quality_reports["scene_code_review"] = {
                    "enabled": True,
                    "attempts": validation_attempts,
                }
                return latest_result

            latest_result = self._apply_validation_results_to_scene_output(
                latest_result,
                validation_result,
                storyboard=storyboard,
            )
            scene_feedback = self._build_scene_feedback_map(validation_result)

            if not scene_feedback:
                break

            logger.warning(
                "Scene code review failed on attempt %s/%s for %s scene(s). Regenerating with validator feedback.",
                attempt + 1,
                max_attempts,
                len(scene_feedback),
            )

        # Determine exit reason: if the last feedback was empty the scenes
        # passed validation; otherwise we exhausted retry attempts.
        final_status = (
            "passed" if not scene_feedback
            else "accepted_with_remaining_failures"
        )
        self._quality_reports["scene_code_review"] = {
            "enabled": self.config.enable_quality_loops,
            "attempts": validation_attempts,
            "final_status": final_status,
        }
        return latest_result
    
    def _run_video_renderer(
        self,
        manifest: OutputManifest,
        structured_logger: StructuredJSONLogger
    ):
        """Run Video Renderer Agent with per-scene error recovery.

        Finds the .py scene files written by the Persistence Agent
        and renders each through Manim. Failed renders are logged and
        skipped — the pipeline never crashes on a single scene failure.

        Args:
            manifest: OutputManifest from Persistence Agent (knows file paths)
            structured_logger: Logger for structured JSON logging

        Returns:
            VideoRendererOutput on success, VideoRendererFailure on total failure
        """
        # Collect scene .py file paths from the manifest
        output_dir = manifest.output_directory
        scene_files = []
        if hasattr(manifest, 'files'):
            for f in manifest.files:
                # OutputFile.path is relative to output_directory
                rel_path = getattr(f, 'path', '') or ''
                file_type = getattr(f, 'type', '') or ''
                if file_type == 'animation_code' or (
                    rel_path.endswith('.py') and 'scene_' in Path(rel_path).name
                ):
                    abs_path = str(Path(output_dir) / rel_path)
                    if Path(abs_path).exists():
                        scene_files.append(abs_path)

        # Fallback: glob the output directory for scene_*.py files
        if not scene_files:
            output_path = Path(output_dir)
            if output_path.exists():
                scene_files = sorted(str(p) for p in output_path.glob("scene_*.py"))

        if not scene_files:
            logger.warning(
                f"No scene .py files found for video rendering in {output_dir}. "
                "Skipping video render stage."
            )
            # List what IS in the output directory for debugging
            output_path = Path(output_dir)
            if output_path.exists():
                all_files = list(output_path.iterdir())
                logger.warning(f"  Output directory contents: {[f.name for f in all_files]}")
            return VideoRendererFailure(
                scene_results=[],
                error=f"No scene files found in output directory: {output_dir}"
            )

        logger.info(f"Found {len(scene_files)} scene files for rendering: "
                     f"{[Path(f).name for f in scene_files]}")

        input_data = VideoRendererInput(
            scene_files=scene_files,
            output_dir=output_dir,
            quality="m"
        )
        input_data.constraints = StepConstraints(timeout_seconds=1800)

        try:
            result = self._execute_with_retry(
                agent=self.video_renderer_agent,
                input_data=input_data,
                stage_name="VideoRenderer",
                structured_logger=structured_logger
            )

            # Log per-scene rendering results
            if hasattr(result, 'scene_results'):
                rendered = sum(1 for r in result.scene_results if r.success)
                failed = sum(1 for r in result.scene_results if not r.success)

                for scene_result in result.scene_results:
                    if scene_result.success:
                        logger.info(
                            f"Scene rendered: {scene_result.scene_id} -> {scene_result.video_path}",
                            extra={"event": "scene_rendered", "scene_id": scene_result.scene_id}
                        )
                    else:
                        logger.warning(
                            f"Scene render FAILED (skipped): {scene_result.scene_id} - {scene_result.error}",
                            extra={"event": "scene_render_failed", "scene_id": scene_result.scene_id}
                        )

                logger.info(f"Video rendering complete: {rendered}/{rendered + failed} scenes succeeded")

            return result

        except Exception as e:
            logger.error(f"Video rendering stage failed entirely: {e}", exc_info=True)
            return VideoRendererFailure(
                scene_results=[],
                error=f"Video rendering failed: {e}"
            )

    def _run_tts_generation_logged(
        self,
        storyboard,
        manifest: OutputManifest,
        structured_logger: StructuredJSONLogger,
    ):
        """Wrap TTS batch generation in structured logging.

        The TTS path doesn't use _execute_agent / _execute_with_retry because
        it's a batch operation, but the orchestrator-level structured log
        still needs agent_start / agent_complete events so failures show up
        in pipeline.log instead of disappearing into stderr.
        """
        scene_count = len(getattr(storyboard, "scenes", []) or [])
        structured_logger.log_agent_start(
            "TTS",
            input_summary=f"{scene_count} scenes",
            retry_attempt=0,
        )
        start = time.time()
        try:
            results = self._run_tts_generation(storyboard, manifest, structured_logger)
            success = sum(1 for r in results if isinstance(r, TTSOutput))
            failed = sum(1 for r in results if isinstance(r, TTSError))
            duration_ms = (time.time() - start) * 1000

            if failed == 0 and success > 0:
                status = "SUCCESS"
            elif success > 0:
                status = "PARTIAL"
            elif scene_count == 0:
                status = "SUCCESS"  # nothing to do
            else:
                status = "FAILURE"

            structured_logger.log_agent_complete(
                "TTS",
                duration_ms=duration_ms,
                output_summary=f"{success} ok / {failed} failed",
                status=status,
                retry_attempt=0,
            )

            # Always record the TTS batch outcome so it lands in tts.json,
            # not just on partial failure — clean runs are diagnostic too.
            self._quality_reports["tts"] = {
                "total": len(results),
                "succeeded": success,
                "failed": failed,
                "status": status,
                "errors": [
                    {"scene_id": r.scene_id, "error": r.error}
                    for r in results if isinstance(r, TTSError)
                ],
            }
            return results
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            structured_logger.log_agent_failure(
                "TTS",
                error_message=str(e),
                error_code="TTS_STAGE_FAILED",
                input_context=f"{scene_count} scenes",
                retry_attempt=0,
                duration_ms=duration_ms,
            )
            self._quality_reports["tts"] = {
                "total": scene_count,
                "succeeded": 0,
                "failed": scene_count,
                "status": "STAGE_ERROR",
                "errors": [{"scene_id": "*", "error": str(e)}],
            }
            return []

    def _run_tts_generation(
        self,
        storyboard,
        manifest: OutputManifest,
        structured_logger: StructuredJSONLogger
    ):
        """Run TTS Agent with per-scene error recovery.

        Extracts narration from the storyboard and generates MP3 audio
        for each scene. Failed scenes are logged and skipped — the
        pipeline never crashes on a single TTS failure.

        Args:
            storyboard: StoryboardJSON with scene narrations
            manifest: OutputManifest from Persistence (for output directory)
            structured_logger: Logger for structured JSON logging

        Returns:
            List of TTSOutput and TTSError results
        """
        # Extract scenes with narration from the storyboard
        scenes_to_tts = []
        if hasattr(storyboard, 'scenes'):
            for scene in storyboard.scenes:
                scene_id = getattr(scene, 'scene_id', None)
                narration = getattr(scene, 'narration', None)
                if scene_id and narration and narration.strip():
                    scenes_to_tts.append({
                        "scene_id": scene_id,
                        "narration": narration
                    })

        if not scenes_to_tts:
            logger.info("No scenes with narration found. Skipping TTS generation.")
            return []

        # Put audio files in an "audio" subdirectory of the output
        audio_output_dir = str(Path(manifest.output_directory) / "audio")
        logger.info(f"Generating TTS for {len(scenes_to_tts)} scenes -> {audio_output_dir}")

        try:
            tts_results = self.tts_agent.process_all_scenes(
                scenes=scenes_to_tts,
                output_dir=audio_output_dir,
                quality="standard",
                voice="alloy"
            )

            success_count = sum(1 for r in tts_results if isinstance(r, TTSOutput))
            error_count = sum(1 for r in tts_results if isinstance(r, TTSError))

            for result in tts_results:
                if isinstance(result, TTSOutput):
                    logger.info(f"TTS OK: {result.scene_id} ({result.duration_seconds:.1f}s, ${result.cost_usd:.4f})")
                elif isinstance(result, TTSError):
                    logger.warning(f"TTS FAILED (skipped): {result.scene_id} - {result.error}")

            logger.info(f"TTS complete: {success_count} succeeded, {error_count} failed")
            return tts_results

        except Exception as e:
            logger.error(f"TTS generation stage failed entirely: {e}", exc_info=True)
            return []

    def _run_audio_video_composition(
        self,
        video_output,
        tts_output,
        manifest: OutputManifest,
        structured_logger: StructuredJSONLogger
    ):
        """Run Audio-Video Composer Agent with per-scene error recovery.

        Matches per-scene .mp4 files (from Video Renderer) with .mp3 files
        (from TTS) and merges them into composed videos with synchronized
        narration. Handles timing mismatches automatically.

        Args:
            video_output: VideoRendererOutput or VideoRendererFailure from stage 8
            tts_output: List of TTSOutput/TTSError from stage 9
            manifest: OutputManifest from Persistence (for output directory)
            structured_logger: Logger for structured JSON logging

        Returns:
            AudioVideoComposerOutput on success,
            AudioVideoComposerFailure on total failure
        """
        # Build a lookup of scene_id → audio file path from TTS results
        audio_map: Dict[str, str] = {}
        if tts_output:
            for result in tts_output:
                if isinstance(result, TTSOutput) and result.audio_file_path:
                    audio_map[result.scene_id] = result.audio_file_path

        # Build scene media pairs from video renderer results
        scene_media: list = []

        if isinstance(video_output, VideoRendererOutput) and hasattr(video_output, 'scene_results'):
            for scene_result in video_output.scene_results:
                if scene_result.success and scene_result.video_path:
                    scene_media.append(SceneMediaPair(
                        scene_id=scene_result.scene_id,
                        video_path=scene_result.video_path,
                        audio_path=audio_map.get(scene_result.scene_id),
                    ))
        elif isinstance(video_output, VideoRendererFailure):
            # Even on partial failure, some scenes may have rendered
            if hasattr(video_output, 'scene_results'):
                for scene_result in video_output.scene_results:
                    if scene_result.success and scene_result.video_path:
                        scene_media.append(SceneMediaPair(
                            scene_id=scene_result.scene_id,
                            video_path=scene_result.video_path,
                            audio_path=audio_map.get(scene_result.scene_id),
                        ))

        if not scene_media:
            logger.warning("No scene video+audio pairs available for composition. Skipping.")
            return AudioVideoComposerFailure(
                scene_results=[],
                error="No scene videos available for composition"
            )

        # Log how many scenes have audio matches
        with_audio = sum(1 for s in scene_media if s.audio_path)
        logger.info(
            f"Composing {len(scene_media)} scenes "
            f"({with_audio} with narration, {len(scene_media) - with_audio} silent)"
        )

        input_data = AudioVideoComposerInput(
            scene_media=scene_media,
            output_dir=manifest.output_directory,
        )
        input_data.constraints = StepConstraints(timeout_seconds=600)

        try:
            result = self._execute_with_retry(
                agent=self.audio_video_composer_agent,
                input_data=input_data,
                stage_name="AudioVideoComposer",
                structured_logger=structured_logger
            )

            # Log per-scene composition results
            if hasattr(result, 'scene_results'):
                for comp_result in result.scene_results:
                    if comp_result.success:
                        logger.info(
                            f"Composed: {comp_result.scene_id} "
                            f"(video={comp_result.video_duration:.1f}s, "
                            f"audio={comp_result.audio_duration:.1f}s, "
                            f"final={comp_result.final_duration:.1f}s)"
                        )
                    else:
                        logger.warning(
                            f"Composition FAILED (skipped): {comp_result.scene_id} "
                            f"- {comp_result.error}"
                        )

            if isinstance(result, AudioVideoComposerOutput):
                logger.info(
                    f"Final video with narration: {result.final_video_path} "
                    f"({result.scenes_composed} composed, "
                    f"{result.scenes_skipped} skipped, "
                    f"{result.total_duration:.1f}s total)"
                )

            return result

        except Exception as e:
            logger.error(f"Audio-video composition stage failed entirely: {e}", exc_info=True)
            return AudioVideoComposerFailure(
                scene_results=[],
                error=f"Audio-video composition failed: {e}"
            )

    def _run_persistence(
        self,
        scene_output,
        storyboard,
        ocr_output,
        file_ref: FileReference,
        structured_logger: StructuredJSONLogger
    ) -> OutputManifest:
        """Run Persistence Agent — writes scene .py files, storyboard, and narration to disk.

        This runs BEFORE video rendering so that Manim has .py files to render.

        Args:
            scene_output: SceneControllerOutput with animation code objects
            storyboard: StoryboardJSON object
            ocr_output: Optional OCROutput for debugging
            file_ref: FileReference for PDF filename
            structured_logger: Logger for structured JSON logging

        Returns:
            OutputManifest with paths to all written files
        """
        pdf_filename = Path(file_ref.filename).stem  # Remove extension

        # In dry-run mode scene_output is None — build empty placeholders.
        if scene_output is not None:
            report = scene_output.report
            code_objects = scene_output.animation_code_objects
        else:
            from src.schemas.scene_processing import SceneProcessingReport
            report = SceneProcessingReport(
                total_scenes=0,
                successful_scenes=0,
                failed_scenes=[],
                output_directory=self._run_output_dir or "",
            )
            code_objects = []

        input_data = PersistenceInput(
            scene_processing_report=report,
            animation_code_objects=code_objects,
            storyboard=storyboard,
            ocr_output=ocr_output,
            pdf_filename=pdf_filename,
            output_dir=self._run_output_dir,  # canonical path — no new timestamp
        )
        input_data.constraints = StepConstraints(timeout_seconds=60)

        result = self._execute_with_retry(
            agent=self.persistence_agent,
            input_data=input_data,
            stage_name="Persistence",
            structured_logger=structured_logger
        )

        return result.manifest

    def _summarize_ocr_for_review(self, ocr_output) -> str:
        """Build a compact OCR summary for storyboard quality review."""
        text_chunks: List[str] = []
        for page in getattr(ocr_output, "pages", [])[:3]:
            for block in getattr(page, "text_blocks", [])[:5]:
                if getattr(block, "text", None):
                    text_chunks.append(block.text.strip())
        return "\n".join(text_chunks)[:2000]

    def _summarize_segments_for_review(self, source_segments) -> str:
        """Build a coverage-oriented source summary for storyboard review."""
        lines = []
        for segment in source_segments or []:
            preview = " ".join(segment.text.split())[:300]
            lines.append(
                f"{segment.segment_id} pages {segment.page_start}-{segment.page_end}: {preview}"
            )
        return "\n".join(lines)[:6000]

    def _serialize_scene_validation_attempt(
        self,
        validation_result: AgentOutput,
        attempt: int,
    ) -> Dict[str, Any]:
        """Convert validator output into a JSON-safe report payload."""
        return {
            "attempt": attempt,
            "status": "passed" if isinstance(validation_result, ManimValidatorOutput) else "failed",
            "results": [
                {
                    "scene_id": result.scene_id,
                    "passed": result.passed,
                    "syntax_ok": result.syntax_ok,
                    "imports_ok": result.imports_ok,
                    "dry_run_ok": result.dry_run_ok,
                    "errors": result.errors,
                    "warnings": result.warnings,
                }
                for result in getattr(validation_result, "results", [])
            ],
        }

    def _build_scene_feedback_map(
        self,
        validation_result: ManimValidatorFailure,
    ) -> Dict[str, List[str]]:
        """Turn validation failures into per-scene revision guidance."""
        feedback_map: Dict[str, List[str]] = {}
        failed_code_by_scene = {
            code_obj.scene_id: code_obj
            for code_obj in getattr(validation_result, "failed_objects", [])
        }

        for result in getattr(validation_result, "results", []):
            if result.passed:
                continue

            feedback = [
                "Fix the validation issues below while preserving the scene's narration and mathematical meaning.",
            ]
            feedback.extend(result.errors)
            feedback.extend(result.warnings[:3])

            snippet = self._extract_relevant_code_snippet(
                failed_code_by_scene.get(result.scene_id),
                result.errors,
            )
            if snippet:
                feedback.append(f"Relevant code snippet:\n{snippet}")

            feedback_map[result.scene_id] = feedback

        return feedback_map

    def _extract_relevant_code_snippet(
        self,
        code_obj,
        errors: List[str],
    ) -> Optional[str]:
        """Extract a small code window around a reported validation line."""
        if code_obj is None or not getattr(code_obj, "code", None):
            return None

        line_number = None
        for error in errors:
            if "line " not in error:
                continue
            try:
                line_number = int(error.split("line ", 1)[1].split(":", 1)[0])
                break
            except (ValueError, IndexError):
                continue

        code_lines = code_obj.code.splitlines()
        if not code_lines:
            return None

        if line_number is None:
            start_index = 0
            end_index = min(len(code_lines), 10)
        else:
            start_index = max(0, line_number - 4)
            end_index = min(len(code_lines), line_number + 2)

        snippet_lines = []
        for idx in range(start_index, end_index):
            snippet_lines.append(f"{idx + 1}: {code_lines[idx]}")
        return "\n".join(snippet_lines)

    def _apply_validation_results_to_scene_output(
        self,
        scene_output,
        validation_result: ManimValidatorFailure,
        storyboard,
    ) -> SceneControllerOutput:
        """Downgrade invalid scene code to failed scenes in the controller report."""
        passed_map = {
            code_obj.scene_id: code_obj
            for code_obj in getattr(validation_result, "passed_objects", [])
        }
        ordered_objects = [
            passed_map[scene.scene_id]
            for scene in getattr(storyboard, "scenes", [])
            if scene.scene_id in passed_map
        ]

        failed_scene_map = {
            failed.scene_id: failed.error
            for failed in getattr(scene_output.report, "failed_scenes", [])
        }
        for result in getattr(validation_result, "results", []):
            if not result.passed:
                failed_scene_map[result.scene_id] = "; ".join(result.errors) or "Scene failed code validation."

        failed_scenes = [
            FailedScene(scene_id=scene_id, error=error)
            for scene_id, error in sorted(failed_scene_map.items())
        ]
        report = SceneProcessingReport(
            total_scenes=scene_output.report.total_scenes,
            successful_scenes=len(ordered_objects),
            failed_scenes=failed_scenes,
            output_directory=scene_output.report.output_directory,
        )

        return SceneControllerOutput(
            report=report,
            animation_code_objects=ordered_objects,
            shared_context=getattr(scene_output, "shared_context", None),
        )

    def _run_final_video_review(
        self,
        storyboard,
        scene_output,
        manifest: OutputManifest,
        video_output,
        tts_output,
        composition_output,
        structured_logger: StructuredJSONLogger,
    ):
        """Run the final media reviewer and persist its report in memory."""
        if not self.config.enable_final_video_review:
            self._quality_reports["video_review"] = {"enabled": False}
            return None

        input_data = VideoQualityReviewerInput(
            storyboard=storyboard,
            manifest=manifest,
            animation_code_objects=getattr(scene_output, "animation_code_objects", []),
            video_output=video_output,
            tts_output=tts_output,
            composition_output=composition_output,
            # A composed final video only exists when both render and TTS ran.
            # When TTS is disabled, skip the "no final video" check so it
            # doesn't fire as a spurious critical finding.
            final_video_expected=self.config.enable_tts,
        )
        input_data.constraints = StepConstraints(budget_tokens=4000, timeout_seconds=60)
        result = self._execute_agent(
            agent=self.video_quality_reviewer_agent,
            input_data=input_data,
            stage_name="FinalVideoReview",
            structured_logger=structured_logger
        )
        if hasattr(result, "to_dict"):
            self._quality_reports["video_review"] = result.to_dict()
        return result

    def _finalize_manifest(
        self,
        manifest: OutputManifest,
        video_output,
        tts_output,
        composition_output,
        video_review_output,
    ) -> OutputManifest:
        """Add runtime media artifacts and quality reports to the final manifest."""
        from src.schemas.output_manifest import RunConfig

        # Record effective runtime configuration
        manifest.run_config = RunConfig(
            dry_run=self.config.dry_run,
            enable_render=self.config.enable_render,
            enable_tts=self.config.enable_tts,
            render_quality=self.config.render_quality,
            tts_voice=self.config.tts_voice,
        )

        self._register_runtime_artifact(
            manifest,
            Path(manifest.output_directory) / "pipeline.log",
            "log",
        )

        for scene_result in getattr(video_output, "scene_results", []) or []:
            if getattr(scene_result, "success", False) and getattr(scene_result, "video_path", None):
                self._register_runtime_artifact(
                    manifest,
                    scene_result.video_path,
                    "scene_video",
                )

        for tts_result in tts_output or []:
            if isinstance(tts_result, TTSOutput):
                self._register_runtime_artifact(
                    manifest,
                    tts_result.audio_file_path,
                    "scene_audio",
                )

        for composition_result in getattr(composition_output, "scene_results", []) or []:
            if getattr(composition_result, "success", False) and getattr(composition_result, "composed_video_path", None):
                self._register_runtime_artifact(
                    manifest,
                    composition_result.composed_video_path,
                    "composed_video",
                )

        final_video_path = None
        if isinstance(composition_output, AudioVideoComposerOutput):
            final_video_path = composition_output.final_video_path
        elif isinstance(video_output, VideoRendererOutput):
            final_video_path = video_output.final_video_path
        elif video_review_output is not None:
            final_video_path = getattr(video_review_output, "final_video_path", None)

        if final_video_path:
            self._register_runtime_artifact(
                manifest,
                final_video_path,
                "final_video",
            )
            # Record on manifest for direct access
            output_root = Path(manifest.output_directory)
            try:
                rel = Path(final_video_path).relative_to(output_root).as_posix()
            except ValueError:
                rel = str(Path(final_video_path).name)
            manifest.final_video_path = rel

        self._populate_cost_summary(manifest)
        self._persist_quality_reports(manifest)
        self._write_manifest_to_disk(manifest)
        return manifest

    def _populate_cost_summary(self, manifest: OutputManifest) -> None:
        """Snapshot the CostTracker singleton into manifest.cost_summary so
        operators can audit per-PDF spend from manifest.json without parsing
        a separate cost report. Silent no-op on any tracker error — cost
        accounting must never abort the pipeline."""
        try:
            from src.schemas.output_manifest import CostSummary
            from src.utils.cost_tracker import CostTracker

            tracker = CostTracker.get_instance()
            breakdown = tracker.get_breakdown()
            manifest.cost_summary = CostSummary(
                total_usd=round(float(breakdown.get("total", 0.0)), 4),
                by_provider=breakdown.get("by_provider", {}),
                by_model=breakdown.get("by_model", {}),
                by_call_type=breakdown.get("by_call_type", {}),
                call_count=len(tracker.get_calls()),
            )
        except Exception as exc:  # pragma: no cover — accounting must never break the run
            logger.warning("Cost summary not recorded (tracker error): %s", exc)

    def _persist_quality_reports(self, manifest: OutputManifest) -> None:
        """Write quality review artifacts to disk when the output directory exists."""
        output_dir = Path(manifest.output_directory)
        if not output_dir.exists():
            return

        report_filenames = {
            "storyboard_review": "storyboard_review.json",
            "scene_code_review": "scene_code_review.json",
            "video_review": "video_review.json",
            "tts": "tts.json",
        }
        for report_key, filename in report_filenames.items():
            if report_key not in self._quality_reports:
                continue
            report_path = output_dir / filename
            self._write_json_report(report_path, self._quality_reports[report_key])
            self._register_runtime_artifact(manifest, report_path, "report")

    def _write_json_report(self, file_path: Path, payload: Dict[str, Any]) -> None:
        """Write a JSON report using the pipeline output directory."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _register_runtime_artifact(
        self,
        manifest: OutputManifest,
        file_path: Union[str, Path],
        file_type: str,
    ) -> None:
        """Upsert a file into the manifest if it exists on disk."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            return

        output_root = Path(manifest.output_directory)
        try:
            relative_path = path_obj.relative_to(output_root)
        except ValueError:
            relative_path = Path(path_obj.name)

        rel_path_str = str(relative_path).replace("\\", "/")
        manifest.files = [output_file for output_file in manifest.files if output_file.path != rel_path_str]
        manifest.files.append(
            OutputFile(
                path=rel_path_str,
                size_bytes=path_obj.stat().st_size,
                type=file_type,
            )
        )

    def _write_manifest_to_disk(self, manifest: OutputManifest) -> None:
        """Rewrite manifest.json after runtime artifacts are known."""
        output_dir = Path(manifest.output_directory)
        if not output_dir.exists():
            return

        manifest_path = output_dir / "manifest.json"
        self._register_runtime_artifact(manifest, manifest_path, "manifest")
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    
    def _execute_agent(
        self,
        agent: Agent,
        input_data: AgentInput,
        stage_name: str,
        structured_logger: StructuredJSONLogger
    ) -> AgentOutput:
        """Execute an agent without retry logic.
        
        Args:
            agent: Agent to execute
            input_data: Input for the agent
            stage_name: Name of the pipeline stage (for logging)
            structured_logger: Logger for structured JSON logging
            
        Returns:
            Agent output
            
        Raises:
            AgentExecutionError: If agent execution fails
        """
        # Log agent start
        input_summary = self._summarize_input(input_data)
        structured_logger.log_agent_start(stage_name, input_summary)
        
        start_time = time.time()
        
        try:
            # Execute agent
            result = agent.execute(input_data)
            
            # Log agent completion
            duration_ms = (time.time() - start_time) * 1000
            output_summary = self._summarize_output(result)
            structured_logger.log_agent_complete(
                stage_name,
                duration_ms,
                output_summary,
                status="SUCCESS"
            )
            
            return result
            
        except Exception as e:
            # Log agent failure
            duration_ms = (time.time() - start_time) * 1000
            error_code = getattr(e, 'error_code', 'UNKNOWN')
            structured_logger.log_agent_failure(
                stage_name,
                str(e),
                error_code,
                input_summary,
                retry_attempt=0,
                duration_ms=duration_ms
            )
            raise
    
    def _execute_with_retry(
        self,
        agent: Agent,
        input_data: AgentInput,
        stage_name: str,
        structured_logger: StructuredJSONLogger
    ) -> AgentOutput:
        """Execute an agent with retry logic based on its retry policy.
        
        Args:
            agent: Agent to execute
            input_data: Input for the agent
            stage_name: Name of the pipeline stage (for logging)
            structured_logger: Logger for structured JSON logging
            
        Returns:
            Agent output
            
        Raises:
            AgentExecutionError: If agent execution fails after all retries
        """
        retry_policy = agent.get_retry_policy()
        last_error = None
        input_summary = self._summarize_input(input_data)
        
        for attempt in range(retry_policy.max_attempts):
            try:
                # Log agent start
                structured_logger.log_agent_start(stage_name, input_summary, attempt)
                
                start_time = time.time()
                
                # Execute agent
                result = agent.execute(input_data)
                
                # Log agent completion
                duration_ms = (time.time() - start_time) * 1000
                output_summary = self._summarize_output(result)
                structured_logger.log_agent_complete(
                    stage_name,
                    duration_ms,
                    output_summary,
                    status="SUCCESS",
                    retry_attempt=attempt
                )
                
                return result
                
            except AgentExecutionError as e:
                last_error = e
                duration_ms = (time.time() - start_time) * 1000
                
                # Check if error is retryable
                is_retryable = (
                    is_retryable_error(e, retry_policy)
                    and attempt < retry_policy.max_attempts - 1
                )
                
                # Log failure
                structured_logger.log_agent_failure(
                    stage_name,
                    str(e),
                    e.error_code,
                    input_summary,
                    retry_attempt=attempt,
                    duration_ms=duration_ms
                )
                
                if not is_retryable:
                    # Non-retryable error or max attempts reached
                    raise
                
                # Calculate backoff delay
                delay = calculate_backoff_delay(
                    attempt,
                    retry_policy.backoff_strategy,
                    retry_policy.base_delay_seconds,
                    retry_policy.max_delay_seconds
                )
                
                logger.info(f"Retrying {stage_name} after {delay:.2f}s delay (attempt {attempt + 2}/{retry_policy.max_attempts})")
                time.sleep(delay)
                
            except Exception as e:
                # Unexpected error
                last_error = e
                duration_ms = (time.time() - start_time) * 1000
                error_code = getattr(e, 'error_code', 'UNEXPECTED_ERROR')
                
                structured_logger.log_agent_failure(
                    stage_name,
                    str(e),
                    error_code,
                    input_summary,
                    retry_attempt=attempt,
                    duration_ms=duration_ms
                )
                
                # Wrap in AgentExecutionError and raise
                raise AgentExecutionError(
                    error_code="UNEXPECTED_ERROR",
                    message=f"Unexpected error in {stage_name}: {str(e)}",
                    context={"error_type": type(e).__name__}
                )
        
        # All retries exhausted
        raise last_error
    
    
    def _summarize_input(self, input_data: AgentInput) -> str:
        """Generate a brief summary of input data for logging."""
        if isinstance(input_data, FileReference):
            return f"FileReference(filename={input_data.filename}, size={input_data.size_bytes})"
        elif hasattr(input_data, '__class__'):
            return f"{input_data.__class__.__name__}"
        else:
            return str(type(input_data).__name__)
    
    def _summarize_output(self, output_data: AgentOutput) -> str:
        """Generate a brief summary of output data for logging."""
        if hasattr(output_data, '__class__'):
            return f"{output_data.__class__.__name__}"
        else:
            return str(type(output_data).__name__)
    
    def _create_local_file_event(self, file_path: str) -> Dict[str, Any]:
        """Create a synthetic event for local file processing.
        
        Args:
            file_path: Path to local file
            
        Returns:
            Event payload dictionary
        """
        import os
        
        file_path_obj = Path(file_path)
        filename = file_path_obj.name
        file_size = os.path.getsize(file_path)
        
        # Create synthetic S3-like event
        return {
            'Records': [{
                'eventTime': datetime.now(timezone.utc).isoformat(),
                'eventName': 'ObjectCreated:Put',
                's3': {
                    'bucket': {'name': 'local'},
                    'object': {
                        'key': str(file_path),
                        'size': file_size
                    }
                }
            }]
        }
    
    def _create_synthetic_event(
        self,
        file_path: str,
        event_source: str
    ) -> Dict[str, Any]:
        """Create a synthetic event for cloud storage processing.
        
        Args:
            file_path: Path to file (cloud storage URL)
            event_source: Event source type
            
        Returns:
            Event payload dictionary
        """
        # This is a simplified version - in production, events would come from
        # actual cloud storage notifications
        if event_source == 's3':
            return self._create_local_file_event(file_path)
        else:
            # Placeholder for other cloud storage types
            return self._create_local_file_event(file_path)
