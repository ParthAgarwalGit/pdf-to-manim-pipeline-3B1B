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
    target_minutes: Optional[float] = None  # requested finished length; None = ungoverned
    enable_manim_dry_run_validation: bool = False
    enable_final_video_review: bool = True
    strict: bool = False
    enable_tts: bool = True
    enable_render: bool = True
    dry_run: bool = False
    tts_voice: str = "alloy"
    render_quality: str = "medium"
    tts_first: bool = False
    verifier_mode: str = "off"
    verifier_max_repairs: int = 1
    # Cost ceiling for render salvage: each salvaged scene costs one codegen
    # LLM call, so a pathological run must not multiply the bill unbounded.
    # 0 disables salvage entirely.
    max_salvage_scenes: int = 10
    # Pre-delivery quality gate: measure every composed scene, apply bounded
    # deterministic repairs, write delivery_report.json (ready/needs review).
    enable_delivery_gate: bool = True
    max_gate_repair_rounds: int = 2
    # Regenerate repairs cost one codegen LLM call each; capped per run.
    max_gate_regenerations: int = 3
    # Whisper transcript vs script (paid); off by default.
    enable_asr_check: bool = False

    def to_storyboard_config(self) -> Configuration:
        """Convert to Configuration schema for storyboard generation."""
        return Configuration(
            verbosity=self.verbosity,
            depth=self.depth,
            audience=self.audience,
            granularity=self.granularity,
            target_minutes=self.target_minutes,
        )


class _WarningCaptureHandler(logging.FileHandler):
    """Persist WARNING+ records from the `src.*` logger tree to a file.

    Console warnings previously vanished when the terminal scrolled away —
    post-run triage was impossible. This handler writes them to
    ``<run>/warnings.log`` (created lazily on the first warning) and keeps a
    count that lands in the manifest.
    """

    def __init__(self, path: str):
        super().__init__(path, encoding="utf-8", delay=True)
        self.setLevel(logging.WARNING)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        super().emit(record)


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
        self._lesson_script = None
        self._scene_reuse = {}

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
            
            # Fail fast: verifier repair runs after all codegen spend, so a
            # missing key must abort before any paid stage, not after.
            if str(self.config.verifier_mode or "off").lower() == "repair":
                self._verifier_repair_key()

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
            self._attach_run_logging()

            # Step 3: Vision+OCR
            ocr_output = self._run_vision_ocr(validated_ref, self.structured_logger)

            # Step 4: Document Segmentation
            source_segments = self._run_document_segmenter(ocr_output, self.structured_logger)

            # Step 5: Script/Storyboard Architect
            storyboard = self._run_script_architect(ocr_output, self.structured_logger, source_segments)
            # Title and analogies for lesson_script.json; its beats are taken
            # from the final storyboard so the two can never disagree.
            self._lesson_script = getattr(self.script_architect_agent, "last_video_script", None)

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
                self._write_lesson_script(manifest, sanitized_storyboard)
                manifest = self._finalize_manifest(
                    manifest,
                    video_output=None,
                    tts_output=None,
                    composition_output=None,
                    video_review_output=None,
                    storyboard=sanitized_storyboard,
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
            tts_output = None
            if self.config.tts_first:
                if not self.config.enable_tts:
                    raise PipelineAbortError(
                        "TTSFirst",
                        "TTS_FIRST_REQUIRES_TTS",
                        "Narration-first timing cannot be used with TTS disabled.",
                    )
                tts_output = self._run_tts_first(
                    sanitized_storyboard,
                    self.structured_logger,
                )
                self._apply_tts_timing(sanitized_storyboard, tts_output)

            scene_output = self._run_scene_controller(
                sanitized_storyboard,
                validated_ref,
                self.structured_logger
            )

            scene_output = self._run_raw_manim_verifier(
                scene_output,
                sanitized_storyboard,
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
            self._write_lesson_script(manifest, sanitized_storyboard)

            # Steps 8+9: Video Rendering and TTS — parallel fan-out.
            # Both only need outputs from Persistence (step 7) so they are
            # fully independent.  The logger is thread-safe (uses a Lock).
            video_output = None

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
                    if self.config.enable_tts and not self.config.tts_first:
                        futures["tts"] = executor.submit(
                            self._run_tts_generation_logged,
                            sanitized_storyboard,
                            manifest,
                            self.structured_logger,
                        )
                    else:
                        logger.info(
                            "TTS generation skipped in parallel stage "
                            "(disabled or already completed by --tts-first)"
                        )
                    # executor.__exit__ waits for both futures before we proceed
                    video_output = futures["render"].result() if "render" in futures else None
                    if "tts" in futures:
                        tts_output = futures["tts"].result()
            else:
                logger.info("Both render and TTS skipped")

            # Render-time salvage: a failed render used to be retried with the
            # IDENTICAL file (same bug, same crash) and then silently dropped
            # from the video. Route the manim error into ONE codegen
            # regeneration attempt per failed scene instead.
            if video_output is not None:
                video_output = self._salvage_failed_renders(
                    video_output, sanitized_storyboard,
                )

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

            # Pre-delivery gate: measure, repair (bounded), then review the
            # repaired media.
            gate_result = self._run_delivery_gate(
                sanitized_storyboard, video_output, tts_output, composition_output,
            )

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
            self._record_delivery(manifest, gate_result, video_review_output)

            manifest = self._finalize_manifest(
                manifest,
                video_output,
                tts_output,
                composition_output,
                video_review_output,
                storyboard=sanitized_storyboard,
            )

            # Log pipeline completion. The rate means DELIVERED scenes (in the
            # rendered video) when rendering ran; codegen success otherwise.
            pipeline_duration = time.time() - pipeline_start_time
            effective_success = (
                manifest.delivered_scenes
                if manifest.delivered_scenes is not None
                else manifest.successful_scenes
            )
            scene_success_rate = (
                effective_success / manifest.total_scenes
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
            self._detach_run_logging()
            temp_logger.close()

    def _attach_run_logging(self) -> None:
        """Point the structured log and warnings.log at the run directory."""
        self.structured_logger = StructuredJSONLogger(
            output_directory=self._run_output_dir
        )
        # Capture WARNING+ records from every src.* logger into the run's
        # warnings.log so post-run triage never depends on console scrollback.
        self._warning_handler = _WarningCaptureHandler(
            str(Path(self._run_output_dir) / "warnings.log")
        )
        logging.getLogger("src").addHandler(self._warning_handler)

    def _detach_run_logging(self) -> None:
        # Detach the warnings capture so handlers don't pile up across runs
        handler = getattr(self, "_warning_handler", None)
        if handler is not None:
            logging.getLogger("src").removeHandler(handler)
            handler.close()
            self._warning_handler = None
        # Close structured logger
        if self.structured_logger:
            self.structured_logger.close()

    def run_revision(self, base_run_dir: str, script) -> OutputManifest:
        """Rebuild a finished run from an edited lesson script.

        The base run is never modified: the revision is a complete new run in
        ``<run>/rev_NNN`` plus ``revision.json``, and can itself be revised.
        Scenes whose beats are unchanged keep the parent's visual plan; their
        audio is reused when narration and voice match, and their code and
        render when the visual fingerprint matches. Everything else is
        re-planned, regenerated, rendered and narrated, and the final video is
        always re-composed. With ``dry_run`` it stops after planning, so
        ``revision.json`` shows what a real run would rebuild.

        Args:
            base_run_dir: Output directory of the run to revise.
            script: The edited lesson script (a VideoScript).
        """
        from types import SimpleNamespace
        from src.orchestrator.revision import (
            beat_diff,
            load_base_run,
            next_revision_dir,
            write_revision_report,
            REVISION_FILENAME,
        )
        from src.utils.cost_tracker import CostTracker
        from src.utils.fingerprints import narration_fingerprint, visual_fingerprint

        start = time.time()
        self._quality_reports = {}
        self._scene_reuse = {}
        cost_before = CostTracker.get_instance().get_total_cost()

        base = load_base_run(Path(base_run_dir))
        rev_dir = next_revision_dir(base.run_dir)
        rev_dir.mkdir(parents=True)
        self._run_output_dir = str(rev_dir)
        logger.info(f"Revising {base.run_dir} -> {rev_dir}")
        self._attach_run_logging()
        try:
            planned, reuse_plan = self.script_architect_agent.revise_storyboard(
                base.storyboard, script,
            )
            storyboard = self._run_json_sanitizer(planned, self.structured_logger)
            self._lesson_script = script
            file_ref = SimpleNamespace(filename=storyboard.pdf_metadata.filename)

            # What carries over. Audio is decided first: under narration-first
            # timing a new recording re-times the plan, so the scene's code
            # and render cannot be reused either.
            reuse_audio: Dict[str, Any] = {}
            reuse_video: Dict[str, Any] = {}
            for scene in storyboard.scenes:
                record = base.records.get(reuse_plan.get(scene.scene_id, ""))
                if record is None:
                    continue
                if (
                    base.artifact(record.audio_path)
                    and record.narration_fp == narration_fingerprint(scene.narration, self.config.tts_voice)
                ):
                    reuse_audio[scene.scene_id] = record
                if (
                    base.artifact(record.code_path)
                    and base.artifact(record.video_path)
                    and record.visual_fp == visual_fingerprint(
                        scene, self.config.render_quality, self.config.tts_first,
                    )
                    and (not self.config.tts_first or scene.scene_id in reuse_audio)
                ):
                    reuse_video[scene.scene_id] = record
            self._scene_reuse = {
                "video": {sid: r.scene_id for sid, r in reuse_video.items()},
                "audio": {sid: r.scene_id for sid, r in reuse_audio.items()},
            }
            # Narration-first: a scene whose beats are unchanged but whose
            # narration is re-recorded keeps its code. The new timing track is
            # injected into it and it is re-rendered, with no codegen call.
            retime_only: Dict[str, Any] = {}
            if self.config.tts_first:
                for sid, base_id in reuse_plan.items():
                    record = base.records.get(base_id)
                    if (
                        record is not None
                        and sid not in reuse_video
                        and sid not in reuse_audio
                        and base.artifact(record.code_path)
                    ):
                        retime_only[sid] = record
            regenerate = [
                s.scene_id for s in storyboard.scenes
                if s.scene_id not in reuse_video and s.scene_id not in retime_only
            ]
            narrate = [s for s in storyboard.scenes if s.scene_id not in reuse_audio]
            logger.info(
                "Revision plan: %d scenes; reusing %d plan(s), %d render(s), %d narration(s); "
                "%d re-timed without codegen",
                len(storyboard.scenes), len(reuse_plan), len(reuse_video), len(reuse_audio),
                len(retime_only),
            )

            if self.config.dry_run:
                manifest = self._run_persistence(None, storyboard, None, file_ref, self.structured_logger)
                self._write_lesson_script(manifest, storyboard)
                video_output = tts_output = composition_output = video_review_output = None
            else:
                tts_output = self._carry_over_audio(base, reuse_audio)
                if self.config.tts_first and narrate:
                    if not self.config.enable_tts:
                        raise PipelineAbortError(
                            "TTSFirst",
                            "TTS_FIRST_REQUIRES_TTS",
                            "Narration-first timing cannot be used with TTS disabled.",
                        )
                    subset = storyboard.model_copy(update={"scenes": narrate})
                    fresh_audio = self._run_tts_first(subset, self.structured_logger)
                    self._apply_tts_timing(subset, fresh_audio)
                    tts_output += fresh_audio

                if regenerate:
                    scene_output = self._run_scene_controller(
                        storyboard, file_ref, self.structured_logger, only_scene_ids=regenerate,
                    )
                    scene_output = self._run_raw_manim_verifier(scene_output, storyboard)
                else:
                    scene_output = None
                scene_output = self._merge_scene_output(
                    storyboard, scene_output, base, reuse_video, retime_only,
                )

                manifest = self._run_persistence(scene_output, storyboard, None, file_ref, self.structured_logger)
                self._write_lesson_script(manifest, storyboard)

                video_output = None
                if self.config.enable_render:
                    scene_files = [
                        str(rev_dir / f"scene_{sid}.py")
                        for sid in regenerate + list(retime_only)
                        if (rev_dir / f"scene_{sid}.py").is_file()
                    ]
                    fresh_renders = (
                        self._run_video_renderer(manifest, self.structured_logger, scene_files=scene_files)
                        if scene_files else None
                    )
                    video_output = self._merge_render_results(
                        storyboard, self._carry_over_renders(base, reuse_video), fresh_renders,
                    )
                    video_output = self._salvage_failed_renders(video_output, storyboard)

                if self.config.enable_tts and not self.config.tts_first and narrate:
                    subset = storyboard.model_copy(update={"scenes": narrate})
                    tts_output += self._run_tts_generation_logged(subset, manifest, self.structured_logger)

                composition_output = None
                if self.config.enable_render and self.config.enable_tts:
                    composition_output = self._run_audio_video_composition(
                        video_output, tts_output, manifest, self.structured_logger,
                    )
                gate_result = self._run_delivery_gate(
                    storyboard, video_output, tts_output, composition_output,
                )
                video_review_output = None
                if self.config.enable_render:
                    video_review_output = self._run_final_video_review(
                        storyboard, scene_output, manifest, video_output,
                        tts_output, composition_output, self.structured_logger,
                    )
                self._record_delivery(manifest, gate_result, video_review_output)

            manifest = self._finalize_manifest(
                manifest, video_output, tts_output, composition_output,
                video_review_output, storyboard=storyboard,
            )
            report_path = rev_dir / REVISION_FILENAME
            write_revision_report(report_path, {
                "parent_run": str(base.run_dir),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dry_run": self.config.dry_run,
                "beats": beat_diff(base.script, script),
                "scenes": {
                    "total": len(storyboard.scenes),
                    "reused_plan": dict(reuse_plan),
                    "reused_video": self._scene_reuse["video"],
                    "reused_audio": self._scene_reuse["audio"],
                    "regenerated": regenerate,
                    "retimed": list(retime_only),
                },
                "cost_usd": round(CostTracker.get_instance().get_total_cost() - cost_before, 4),
                "duration_seconds": round(time.time() - start, 2),
            })
            self._register_runtime_artifact(manifest, report_path, "report")
            self._write_manifest_to_disk(manifest)
            logger.info(
                f"Revision completed in {time.time() - start:.2f}s: "
                f"{len(regenerate)} scene(s) regenerated -> {rev_dir}"
            )
            return manifest
        finally:
            self._detach_run_logging()

    def _carry_over_audio(self, base, reuse_audio: Dict[str, Any]) -> List[Any]:
        """Copy reused narration into this run under the scenes' new ids."""
        import shutil

        audio_dir = Path(self._run_output_dir) / "audio"
        results: List[Any] = []
        for scene_id, record in reuse_audio.items():
            # Same layout TTSAgent.process_all_scenes writes.
            target = audio_dir / scene_id / f"{scene_id}.mp3"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base.artifact(record.audio_path), target)
            results.append(TTSOutput(
                audio_file_path=str(target),
                duration_seconds=float(record.audio_duration_seconds or 0.0),
                scene_id=scene_id,
                cost_usd=0.0,
            ))
        return results

    def _carry_over_renders(self, base, reuse_video: Dict[str, Any]) -> Dict[str, Any]:
        """Copy reused scene renders into this run; returns scene_id -> result."""
        import shutil
        from src.agents.video_renderer import SceneRenderResult

        media_dir = Path(self._run_output_dir) / "media" / "reused"
        results: Dict[str, Any] = {}
        for scene_id, record in reuse_video.items():
            media_dir.mkdir(parents=True, exist_ok=True)
            target = media_dir / f"scene_{scene_id}.mp4"
            shutil.copy2(base.artifact(record.video_path), target)
            results[scene_id] = SceneRenderResult(
                scene_id=scene_id,
                success=True,
                video_path=str(target),
                duration_seconds=float(record.video_duration_seconds or 0.0),
            )
        return results

    def _merge_scene_output(
        self,
        storyboard,
        scene_output,
        base,
        reuse_video: Dict[str, Any],
        retime_only: Optional[Dict[str, Any]] = None,
    ):
        """Combine fresh code with the reused scenes' code, in video order.

        ``retime_only`` scenes reuse their parent code with the new timing
        track injected (narration re-recorded, beats unchanged).
        """
        from types import SimpleNamespace
        from src.orchestrator.revision import code_object_from_file
        from src.schemas.scene_processing import SceneProcessingReport
        from src.utils.sync_contract import inject_sync_contract

        retime_only = retime_only or {}
        fresh = {c.scene_id: c for c in getattr(scene_output, "animation_code_objects", []) or []}
        failed = list(getattr(getattr(scene_output, "report", None), "failed_scenes", []) or [])
        code_objects = []
        for scene in storyboard.scenes:
            record = reuse_video.get(scene.scene_id)
            retime = retime_only.get(scene.scene_id)
            if record is not None:
                code_objects.append(code_object_from_file(base.artifact(record.code_path), scene))
            elif retime is not None:
                code_obj = code_object_from_file(base.artifact(retime.code_path), scene)
                ordered = sorted((scene.beat_timings or {}).items(), key=lambda kv: kv[1][0])
                code_obj.code = inject_sync_contract(
                    code_obj.code,
                    {beat_id: span[0] for beat_id, span in ordered},
                    scene.audio_duration_seconds,
                    class_name=code_obj.class_name,
                )
                code_objects.append(code_obj)
            elif scene.scene_id in fresh:
                code_objects.append(fresh[scene.scene_id])
        report = SceneProcessingReport(
            total_scenes=len(storyboard.scenes),
            successful_scenes=len(code_objects),
            failed_scenes=failed,
            output_directory=self._run_output_dir,
        )
        return SimpleNamespace(report=report, animation_code_objects=code_objects)

    def _merge_render_results(self, storyboard, carried: Dict[str, Any], fresh_output):
        """Order carried and fresh renders by scene and rebuild the silent concat."""
        import shutil

        fresh = {r.scene_id: r for r in getattr(fresh_output, "scene_results", []) or []}
        results = []
        for scene in storyboard.scenes:
            result = carried.get(scene.scene_id) or fresh.get(scene.scene_id)
            if result is not None:
                results.append(result)
        rendered = [r.video_path for r in results if r.success and r.video_path]
        if not rendered:
            return VideoRendererFailure(
                scene_results=results,
                error=getattr(fresh_output, "error", None) or "No scene rendered",
            )
        final_path = str(Path(self._run_output_dir) / "final_output.mp4")
        try:
            if len(rendered) == 1:
                shutil.copy2(rendered[0], final_path)
            else:
                self.video_renderer_agent._concatenate_videos(rendered, final_path)
        except Exception as exc:
            # The narrated video is composed from per-scene files; only the
            # silent preview concat is lost.
            logger.warning("Silent concat of revised scenes failed: %s", exc)
        return VideoRendererOutput(
            final_video_path=final_path,
            scene_results=results,
            total_duration=sum(r.duration_seconds for r in results if r.success),
        )

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
        from src.agents.document_segmenter import (
            MAX_SEGMENTS,
            scenes_for_minutes,
            target_words_for,
            total_ocr_words,
        )

        # Adaptive: honour the granularity dial, but cap total segments so a long
        # PDF doesn't explode into 100+ scenes (see DocumentSegmenter.MAX_SEGMENTS).
        # When the user requests a target length, tighten the cap to that scene
        # budget so the segment floor never overshoots the requested minutes.
        seg_cap = MAX_SEGMENTS
        if self.config.target_minutes:
            seg_cap = min(MAX_SEGMENTS, scenes_for_minutes(self.config.target_minutes))
        target_words = target_words_for(
            self.config.granularity, total_ocr_words(ocr_output), max_segments=seg_cap
        )
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
        structured_logger: StructuredJSONLogger,
        only_scene_ids: Optional[List[str]] = None,
    ):
        """Run Scene Controller Agent with retry logic.

        Args:
            storyboard: Sanitized StoryboardJSON object
            file_ref: FileReference for output directory naming
            structured_logger: Logger for structured JSON logging
            only_scene_ids: Generate code for these scenes only (revision
                runs); None means every scene.

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
                only_scene_ids=only_scene_ids,
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
        structured_logger: StructuredJSONLogger,
        scene_files: Optional[List[str]] = None,
    ):
        """Run Video Renderer Agent with per-scene error recovery.

        Finds the .py scene files written by the Persistence Agent
        and renders each through Manim. Failed renders are logged and
        skipped — the pipeline never crashes on a single scene failure.

        Args:
            manifest: OutputManifest from Persistence Agent (knows file paths)
            structured_logger: Logger for structured JSON logging
            scene_files: Render exactly these files (revision runs);
                None means every scene file in the manifest.

        Returns:
            VideoRendererOutput on success, VideoRendererFailure on total failure
        """
        # Collect scene .py file paths from the manifest
        output_dir = manifest.output_directory
        explicit_files = scene_files is not None
        scene_files = list(scene_files or [])
        if not explicit_files and hasattr(manifest, 'files'):
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
        if not scene_files and not explicit_files:
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
            quality=self.config.render_quality,
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

    def _regenerate_scene(self, scene, storyboard, feedback: List[str]):
        """Re-codegen one scene with ``feedback``, validate, persist, re-render.

        Shared by render salvage and the delivery gate's regenerate repair.
        Returns the new SceneRenderResult (check ``success``), or None when
        generation or validation failed. One codegen LLM call; never raises.
        """
        from src.agents.animation_generator import (
            AnimationCodeGeneratorAgent,
            AnimationGeneratorInput,
            AnimationGeneratorOutput,
        )
        from src.schemas.scene_processing import SceneProcessingInput

        controller = self.scene_controller_agent
        renderer = self.video_renderer_agent
        try:
            generator = AnimationCodeGeneratorAgent(strict=self.config.strict)
            gen_out = generator.execute(AnimationGeneratorInput(
                SceneProcessingInput(
                    scene_spec=controller._scene_to_spec(scene),
                    context=controller._initialize_context(storyboard),
                    scene_index=scene.scene_index,
                ),
                review_feedback=feedback,
            ))
            if not isinstance(gen_out, AnimationGeneratorOutput):
                return None
            code_obj = gen_out.animation_code

            # Gate the regenerated code (also applies the LaTeX auto-fix).
            validation = self.manim_code_validator_agent.execute(
                ManimValidatorInput(animation_code_objects=[code_obj])
            )
            if not isinstance(validation, ManimValidatorOutput):
                logger.warning(
                    "Regenerated code for %s failed validation; dropping.", scene.scene_id,
                )
                return None

            # Rewrite the scene file exactly as Persistence formats it.
            file_path = Path(self._run_output_dir) / f"scene_{code_obj.scene_id}.py"
            content = self.persistence_agent._generate_scene_file_content(code_obj)
            self.persistence_agent._write_file_atomically(str(file_path), content)

            # Same quality as the main render, or the stream-copy concat breaks.
            rerender = renderer._render_scene(
                str(file_path),
                renderer.quality_flag_for(self.config.render_quality),
                str(Path(self._run_output_dir) / "media"),
            )
            if not rerender.success:
                logger.warning(
                    "Re-render after regeneration failed for %s: %s",
                    scene.scene_id, (rerender.error or "")[:200],
                )
            return rerender
        except Exception as exc:
            logger.warning("Regeneration of %s errored: %s", scene.scene_id, exc)
            return None

    def _salvage_failed_renders(self, video_output, storyboard):
        """One codegen-regeneration attempt per failed render.

        The manim stderr tail (already de-noised by _extract_render_error) is
        routed into the animation generator as revision feedback, the new code
        is re-validated (which also applies the LaTeX auto-fix), the scene
        file is rewritten exactly as Persistence writes it, and the single
        scene is re-rendered. Successes replace the failed entries in place
        (preserving scene order for composition) and the silent concat is
        rebuilt. Best-effort: any exception leaves the original result alone.
        """
        scene_results = list(getattr(video_output, "scene_results", []) or [])
        failed = [r for r in scene_results if not r.success]
        if not failed or not self._run_output_dir:
            return video_output

        # Cost ceiling: one codegen LLM call per salvage attempt. A run with
        # mass failures signals a systemic bug — regenerating everything would
        # multiply the bill without fixing the cause.
        cap = max(0, int(self.config.max_salvage_scenes))
        if len(failed) > cap:
            skipped = failed[cap:]
            failed = failed[:cap]
            logger.warning(
                "Render salvage capped at %d of %d failed scene(s) "
                "(max_salvage_scenes); skipping: %s",
                cap, cap + len(skipped), [r.scene_id for r in skipped],
            )
        if not failed:
            return video_output

        import shutil

        scenes_by_id = {s.scene_id: s for s in getattr(storyboard, "scenes", [])}
        logger.info("Attempting render salvage for %d failed scene(s).", len(failed))
        salvaged = 0
        for result in failed:
            scene = scenes_by_id.get(result.scene_id)
            if scene is None:
                continue
            feedback = [
                "The previous version of this scene FAILED TO RENDER. Manim error:",
                (result.error or "unknown error")[-600:],
                "Fix the exact cause (usually a LaTeX string or an invalid mobject "
                "argument) while keeping the same visual plan and narration.",
            ]
            rerender = self._regenerate_scene(scene, storyboard, feedback)
            if rerender is not None and rerender.success:
                scene_results[scene_results.index(result)] = rerender
                salvaged += 1
                logger.info("Salvaged scene %s after render failure.", result.scene_id)

        if not salvaged:
            return video_output

        video_output.scene_results = scene_results
        # Rebuild the silent concat so final_output.mp4 includes salvaged scenes.
        rendered = [r.video_path for r in scene_results if r.success and r.video_path]
        final_path = (
            getattr(video_output, "final_video_path", None)
            or str(Path(self._run_output_dir) / "final_output.mp4")
        )
        try:
            if len(rendered) == 1:
                shutil.copy2(rendered[0], final_path)
            elif rendered:
                renderer._concatenate_videos(rendered, final_path)
            video_output.final_video_path = final_path
            if hasattr(video_output, "total_duration"):
                video_output.total_duration = sum(
                    r.duration_seconds for r in scene_results if r.success
                )
        except Exception as exc:
            logger.warning("Re-concatenation after salvage failed: %s", exc)
        logger.info("Render salvage recovered %d/%d failed scene(s).", salvaged, len(failed))
        return video_output

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

    def _run_tts_first(self, storyboard, structured_logger: StructuredJSONLogger):
        """Generate narration before codegen so measured audio owns scene timing."""
        from types import SimpleNamespace

        if not self._run_output_dir:
            raise PipelineAbortError(
                "TTSFirst",
                "OUTPUT_DIRECTORY_MISSING",
                "The run output directory must exist before narration-first TTS.",
            )
        placeholder_manifest = SimpleNamespace(output_directory=self._run_output_dir)
        results = self._run_tts_generation_logged(
            storyboard,
            placeholder_manifest,
            structured_logger,
        )
        if self.config.strict and any(isinstance(result, TTSError) for result in results):
            raise PipelineAbortError(
                "TTSFirst",
                "TTS_FIRST_FAILED",
                "At least one scene narration failed before code generation.",
            )
        return results

    def _apply_tts_timing(self, storyboard, tts_results) -> None:
        """Rescale every visual timestamp to measured per-scene narration length."""
        durations = {
            result.scene_id: result.duration_seconds
            for result in (tts_results or [])
            if isinstance(result, TTSOutput) and result.duration_seconds > 0
        }
        timing_report: List[Dict[str, Any]] = []
        for scene in getattr(storyboard, "scenes", []) or []:
            measured = durations.get(scene.scene_id)
            if measured is None:
                if self.config.strict:
                    raise PipelineAbortError(
                        "TTSFirst",
                        "SCENE_AUDIO_MISSING",
                        f"No measured narration duration for {scene.scene_id}.",
                    )
                continue
            intent = scene.visual_intent
            # Scale from the plan's own timeline, not duration_estimate: the
            # estimate is floored at MIN_SCENE_DURATION (8s) while the planner
            # times visuals to ~word_count/3s, so a short scene would otherwise
            # be compressed to a fraction of its measured audio.
            plan_extent = max(
                [obj.interval.t_out for obj in intent.mathematical_objects if obj.interval]
                + [beat.end_time for beat in intent.beats]
                + [t.timing for t in intent.transformations]
                + [e.timestamp for e in intent.emphasis_points],
                default=0.0,
            )
            old_duration = (
                plan_extent if plan_extent > 0
                else max(float(scene.duration_estimate), 0.001)
            )
            new_duration = round(float(measured), 3)
            scale = new_duration / old_duration

            for obj in intent.mathematical_objects:
                if obj.interval is not None:
                    latest_start = max(0.0, new_duration - 0.05)
                    obj.interval.t_in = min(
                        latest_start,
                        round(obj.interval.t_in * scale, 3),
                    )
                    obj.interval.t_out = min(
                        new_duration,
                        round(obj.interval.t_out * scale, 3),
                    )
                    if obj.interval.t_out <= obj.interval.t_in:
                        obj.interval.t_out = min(new_duration, obj.interval.t_in + 0.05)
            timings = getattr(scene, "beat_timings", None) or {}
            for beat in intent.beats:
                if beat.beat_id in timings:
                    # Measured per-beat narration beats proportional scaling.
                    beat.start_time, beat.end_time = (
                        min(new_duration, round(float(t), 3)) for t in timings[beat.beat_id]
                    )
                    continue
                beat.start_time = min(new_duration, round(beat.start_time * scale, 3))
                beat.end_time = min(new_duration, round(beat.end_time * scale, 3))
            for transformation in intent.transformations:
                transformation.timing = min(
                    new_duration,
                    round(transformation.timing * scale, 3),
                )
            for emphasis in intent.emphasis_points:
                emphasis.timestamp = min(
                    new_duration,
                    round(emphasis.timestamp * scale, 3),
                )

            scene.audio_duration_seconds = new_duration
            scene.duration_estimate = new_duration
            timing_report.append({
                "scene_id": scene.scene_id,
                "estimated_seconds": old_duration,
                "measured_seconds": new_duration,
                "scale": round(scale, 4),
            })
        self._quality_reports["narration_timing"] = {
            "enabled": True,
            "scenes": timing_report,
        }

    def _verifier_repair_key(self) -> str:
        """GEMINI_API_KEY for verifier repair: repo .env, then the process env
        (where src/cli.py loads keys.env). Raises PipelineAbortError if absent."""
        from src.manim_verifier.vision_reviewer import GeminiVisionReviewer

        env_path = Path(__file__).resolve().parents[2] / ".env"
        try:
            return GeminiVisionReviewer.resolve_api_key(env_path, allow_environ=True)
        except ValueError as exc:
            raise PipelineAbortError(
                "ManimVerifier",
                "VERIFIER_REVIEWER_UNAVAILABLE",
                str(exc),
            ) from exc

    def _run_raw_manim_verifier(self, scene_output, storyboard):
        """Preview generated raw Manim and optionally replace it with a repair."""
        mode = str(self.config.verifier_mode or "off").lower()
        if mode == "off" or scene_output is None:
            self._quality_reports["manim_verifier"] = {"mode": "off", "scenes": []}
            return scene_output

        from src.manim_verifier.pipeline import RawManimVerifierPipeline, VerifierConfig
        from src.manim_verifier.preview_renderer import RawManimPreviewRenderer
        from src.manim_verifier.static_verifier import StaticManimVerifier
        from src.manim_verifier.vision_reviewer import GeminiVisionReviewer

        reviewer = None
        if mode == "repair":
            try:
                reviewer = GeminiVisionReviewer(api_key=self._verifier_repair_key())
            except PipelineAbortError:
                raise
            except Exception as exc:
                raise PipelineAbortError(
                    "ManimVerifier",
                    "VERIFIER_REVIEWER_UNAVAILABLE",
                    str(exc),
                ) from exc

        static_verifier = StaticManimVerifier(
            trusted_import_roots={
                "src.utils.animation_patterns",
                "src.utils.coordinate_system_helpers",
                "src.utils.geometry_helpers",
                "src.utils.layout_tracker",
                "src.utils.notation_helpers",
                "src.utils.pi_characters",
            },
            require_purpose_comments=True,
        )
        renderer = RawManimPreviewRenderer(verifier=static_verifier)
        verifier = RawManimVerifierPipeline(
            reviewer=reviewer,
            renderer=renderer,
            config=VerifierConfig(
                mode=mode,
                run_root=Path(self._run_output_dir) / "verifier_runs",
                preview_quality="low",
                max_repair_attempts=self.config.verifier_max_repairs,
            ),
        )
        scenes_by_id = {
            scene.scene_id: scene for scene in getattr(storyboard, "scenes", []) or []
        }
        reports: List[Dict[str, Any]] = []
        for code_obj in getattr(scene_output, "animation_code_objects", []) or []:
            scene = scenes_by_id.get(code_obj.scene_id)
            existing_lines = {line.strip() for line in code_obj.code.splitlines()}
            missing_imports = [
                line for line in code_obj.imports if line.strip() not in existing_lines
            ]
            full_source = "\n".join(missing_imports + [code_obj.code])
            result = verifier.run(
                brief=code_obj.narration,
                storyboard=(
                    json.dumps(scene.model_dump(mode="json"), indent=2)
                    if scene is not None else "{}"
                ),
                initial_code=full_source,
            )
            reports.append({
                "scene_id": code_obj.scene_id,
                "success": result.success,
                "repaired": result.repaired,
                "fallback_used": result.fallback_used,
                "run_directory": result.run_directory,
                "attempts": len(result.attempts),
            })
            if mode == "repair" and result.repaired and result.success:
                code_obj.imports = []
                code_obj.code = result.final_code
                final_report = static_verifier.verify(result.final_code)
                if final_report.scene_class:
                    code_obj.class_name = final_report.scene_class
            if self.config.strict and not result.success:
                raise PipelineAbortError(
                    "ManimVerifier",
                    "STRICT_PREFLIGHT_FAILED",
                    f"Raw-Manim preflight failed for {code_obj.scene_id}.",
                    {"run_directory": result.run_directory},
                )
        self._quality_reports["manim_verifier"] = {"mode": mode, "scenes": reports}
        return scene_output

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
                    entry = {"scene_id": scene_id, "narration": narration}
                    beats = list(getattr(scene, "source_beats", None) or [])
                    # Per-beat synthesis yields the beat timing track; only
                    # when the beats are exactly the scene's narration.
                    if beats and "\n\n".join(b.narration for b in beats) == narration:
                        entry["beats"] = [(b.beat_id, b.narration) for b in beats]
                    scenes_to_tts.append(entry)

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
                voice=self.config.tts_voice,
            )

            success_count = sum(1 for r in tts_results if isinstance(r, TTSOutput))
            error_count = sum(1 for r in tts_results if isinstance(r, TTSError))

            scenes_by_id = {s.scene_id: s for s in getattr(storyboard, "scenes", []) or []}
            for result in tts_results:
                timings = getattr(result, "beat_timings", None)
                scene = scenes_by_id.get(getattr(result, "scene_id", None))
                if isinstance(result, TTSOutput) and timings and scene is not None:
                    scene.beat_timings = dict(timings)

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

    def _run_delivery_gate(self, storyboard, video_output, tts_output, composition_output):
        """Measure every composed scene and apply bounded deterministic repairs.

        Mutates ``video_output`` and ``composition_output`` in place when a
        scene is re-timed or recomposed. Returns None when there is no
        composed video to check (gate disabled, render or TTS skipped).
        Never fatal: a gate error leaves the media as composed.
        """
        if not self.config.enable_delivery_gate or not isinstance(
            composition_output, AudioVideoComposerOutput
        ):
            return None
        from src.agents.delivery_gate import DeliveryGate, _Probes

        scenes_by_id = {s.scene_id: s for s in getattr(storyboard, "scenes", []) or []}

        def regenerate(scene_id: str, feedback: List[str]):
            scene = scenes_by_id.get(scene_id)
            return self._regenerate_scene(scene, storyboard, feedback) if scene else None

        gate = DeliveryGate(
            renderer=self.video_renderer_agent,
            composer=self.audio_video_composer_agent,
            render_quality=self.config.render_quality,
            max_rounds=self.config.max_gate_repair_rounds,
            probes=_Probes(transcribe=self._asr_transcriber()),
            regenerate=regenerate,
            max_regenerations=self.config.max_gate_regenerations,
        )
        start = time.time()
        try:
            result = gate.run(
                storyboard, self._run_output_dir, video_output, tts_output or [], composition_output,
            )
        except Exception as exc:
            logger.warning("Delivery gate failed; delivering without it: %s", exc, exc_info=True)
            return None
        result.gate = gate
        repaired = sum(1 for s in result.scenes if s.repairs_applied)
        logger.info(
            "Delivery gate: %d/%d scenes passed, %d repaired in %d round(s) (%.1fs)",
            sum(1 for s in result.scenes if s.passed), len(result.scenes),
            repaired, result.repair_rounds, time.time() - start,
        )
        return result

    def _asr_transcriber(self):
        """Whisper transcription for the narration-fidelity check, if enabled.

        Off by default (paid: about $0.006 per audio minute). Returns None
        when disabled or when no OpenAI key is configured.
        """
        import os

        if not self.config.enable_asr_check or not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            import openai
        except ImportError:
            return None
        client = openai.OpenAI()

        def transcribe(path: str):
            with open(path, "rb") as audio:
                response = client.audio.transcriptions.create(model="whisper-1", file=audio)
            return getattr(response, "text", None)

        return transcribe

    def _record_delivery(self, manifest: OutputManifest, gate_result, video_review_output) -> None:
        """Turn the gate result plus blocking review findings into the delivery report."""
        if gate_result is None:
            return
        report = gate_result.gate.report(
            gate_result, getattr(video_review_output, "findings", None),
        )
        self._quality_reports["delivery"] = report.model_dump(mode="json")
        manifest.delivery_status = report.status
        log = logger.info if report.status == "ready" else logger.warning
        log("Delivery: %s", report.summary.replace("\n", " | "))

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
        storyboard=None,
    ) -> OutputManifest:
        """Add runtime media artifacts and quality reports to the final manifest.

        With ``storyboard``, also records each scene's fingerprints and
        artifact paths so a later revision run can reuse them.
        """
        from src.schemas.output_manifest import RunConfig

        # Record effective runtime configuration
        manifest.run_config = RunConfig(
            dry_run=self.config.dry_run,
            enable_render=self.config.enable_render,
            enable_tts=self.config.enable_tts,
            render_quality=self.config.render_quality,
            tts_voice=self.config.tts_voice,
            tts_first=self.config.tts_first,
        )

        self._register_runtime_artifact(
            manifest,
            Path(manifest.output_directory) / "pipeline.log",
            "log",
        )

        # Warnings observability: count + artifact (file exists only if a
        # warning actually fired — the handler creates it lazily).
        handler = getattr(self, "_warning_handler", None)
        if handler is not None:
            handler.flush()
            manifest.warning_count = handler.count
        self._register_runtime_artifact(
            manifest,
            Path(manifest.output_directory) / "warnings.log",
            "log",
        )

        # Honest delivery metrics: successful_scenes counts code generation;
        # what students actually receive is the rendered video. A real run
        # reported 1.0 success while 13/90 scenes were missing from the file.
        render_results = list(getattr(video_output, "scene_results", []) or [])
        if render_results:
            manifest.delivered_scenes = sum(1 for r in render_results if r.success)
            manifest.lost_scenes = [r.scene_id for r in render_results if not r.success]

        for scene_result in render_results:
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

        review_findings = list(getattr(video_review_output, "findings", []) or [])
        blocking_review = any(
            getattr(finding, "severity", "") in {"high", "critical"}
            for finding in review_findings
        )
        delivery_incomplete = bool(manifest.lost_scenes)
        if (
            self.config.strict
            and self.config.enable_render
            and (delivery_incomplete or blocking_review)
        ):
            lost_count = len(manifest.lost_scenes)
            if not final_video_path or (
                manifest.total_scenes > 0 and lost_count * 2 > manifest.total_scenes
            ):
                manifest.status = "CRITICAL_FAILURE"
            else:
                manifest.status = "FAILURE"

        timing_track = {
            s.scene_id: s.beat_timings
            for s in getattr(storyboard, "scenes", None) or []
            if getattr(s, "beat_timings", None)
        }
        if timing_track:
            self._quality_reports["timing_track"] = timing_track
            # Persistence wrote storyboard.json before narration existed in
            # render-parallel runs; refresh it so revisions see the track.
            try:
                self.persistence_agent._write_storyboard(manifest.output_directory, storyboard)
            except Exception as exc:  # pragma: no cover — best effort
                logger.warning("storyboard.json not refreshed with timings: %s", exc)

        if storyboard is not None and getattr(storyboard, "scenes", None):
            from src.orchestrator.revision import build_scene_records

            reuse = getattr(self, "_scene_reuse", None) or {}
            manifest.scenes = build_scene_records(
                storyboard,
                Path(manifest.output_directory),
                manifest.files,
                render_results,
                tts_output or [],
                voice=self.config.tts_voice,
                render_quality=self.config.render_quality,
                tts_first=self.config.tts_first,
                reused_video=reuse.get("video"),
                reused_audio=reuse.get("audio"),
            )

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
            "narration_timing": "narration_timing.json",
            "manim_verifier": "manim_verifier.json",
            "delivery": "delivery_report.json",
            "timing_track": "timing_track.json",
        }
        for report_key, filename in report_filenames.items():
            if report_key not in self._quality_reports:
                continue
            report_path = output_dir / filename
            self._write_json_report(report_path, self._quality_reports[report_key])
            self._register_runtime_artifact(manifest, report_path, "report")

    def _write_lesson_script(self, manifest: OutputManifest, storyboard) -> None:
        """Persist lesson_script.json, the professor-editable source of truth.

        Beats come from the final storyboard; title and analogies from the
        script the planner (or a revision) produced. Never fatal: a run
        without it simply cannot be revised from an edited script.
        """
        from src.orchestrator.revision import LESSON_SCRIPT_FILENAME, lesson_script_from_storyboard

        source = getattr(self, "_lesson_script", None)
        try:
            script = lesson_script_from_storyboard(
                storyboard,
                title=getattr(source, "title", "") or "",
                analogies=getattr(source, "analogies", None) or [],
            )
            if script is None:
                return
            path = Path(manifest.output_directory) / LESSON_SCRIPT_FILENAME
            path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
            self._register_runtime_artifact(manifest, path, "lesson_script")
        except Exception as exc:  # pragma: no cover — persistence of an extra artifact
            logger.warning("lesson_script.json not written: %s", exc)

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
