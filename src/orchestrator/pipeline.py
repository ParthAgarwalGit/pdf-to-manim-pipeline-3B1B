"""Pipeline Orchestrator for PDF to Manim conversion.

This module implements the main orchestrator that sequences agent invocations
in a linear pipeline: Ingestion → Validation → Vision+OCR → Script Architect →
JSON Sanitizer → Scene Controller → Persistence.

The orchestrator handles:
- Linear agent sequencing with explicit handoffs
- Abort logic for validation and sanitization errors
- Retry logic based on agent retry policies with exponential backoff
- Structured logging at each stage

Error Handling Strategy:
- **Abort**: Validation and sanitization errors terminate pipeline immediately
  (non-retryable, deterministic failures indicating invalid input)
- **Retry**: Transient failures (API timeouts, rate limits) use exponential backoff
  with configurable max attempts per agent
- **Continue**: Scene-level failures are isolated; pipeline continues processing
  remaining scenes (failure isolation)

Retry Policy:
- Each agent defines its own retry policy via get_retry_policy()
- Retryable errors: API_TIMEOUT, API_RATE_LIMIT, NETWORK_ERROR, etc.
- Non-retryable errors: INVALID_INPUT, SCHEMA_VIOLATION, CORRUPTED_PDF, etc.
- Backoff strategies: EXPONENTIAL (default), LINEAR, CONSTANT
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.orchestrator.retry_policy import calculate_backoff_delay, is_retryable_error
from src.orchestrator.logger import StructuredJSONLogger
from src.agents.ingestion import CloudStorageEvent, IngestAgent, IngestionError, IngestionOutput
from src.agents.validation import ValidationAgent, ValidationError
from src.agents.vision_ocr import VisionOCRAgent
from src.agents.script_architect import ScriptArchitectAgent, ScriptArchitectInput
from src.agents.json_sanitizer import JSONSanitizerAgent, JSONSanitizerInput, SanitizationError
from src.agents.scene_controller import SceneControllerAgent, SceneControllerInput
from src.agents.persistence import PersistenceAgent, PersistenceInput
from src.schemas.file_reference import FileReference
from src.schemas.output_manifest import OutputManifest
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
        enable_ocr_debug: Whether to write OCR output for debugging
    """
    verbosity: str = "medium"
    depth: str = "intermediate"
    audience: str = "undergraduate"
    base_output_dir: str = "output"
    enable_ocr_debug: bool = False
    
    def to_storyboard_config(self) -> Configuration:
        """Convert to Configuration schema for storyboard generation."""
        return Configuration(
            verbosity=self.verbosity,
            depth=self.depth,
            audience=self.audience
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
    
    The orchestrator sequences agent invocations in a linear pipeline:
    1. Ingestion Agent - Detect cloud storage uploads and extract metadata
    2. Validation Agent - Verify PDF file type and structure
    3. Vision+OCR Agent - Extract text, math, and diagrams
    4. Script Architect - Generate pedagogical storyboard
    5. JSON Sanitizer - Enforce schema compliance
    6. Scene Controller - Iterate over scenes and generate animation code
    7. Persistence Agent - Write all artifacts to file system
    
    The orchestrator implements:
    - Abort logic for validation and sanitization errors
    - Retry logic based on agent retry policies
    - Structured logging at each stage
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
        self.ingestion_agent = IngestAgent()
        self.validation_agent = ValidationAgent()
        self.vision_ocr_agent = VisionOCRAgent()
        self.script_architect_agent = ScriptArchitectAgent()
        self.json_sanitizer_agent = JSONSanitizerAgent()
        self.scene_controller_agent = SceneControllerAgent()
        self.persistence_agent = PersistenceAgent(base_output_dir=self.config.base_output_dir)
        
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
            
            # Now we can determine the output directory and create the structured logger
            pdf_filename = Path(validated_ref.filename).stem
            timestamp = datetime.now(timezone.utc)
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
            output_dir = str(Path(self.config.base_output_dir) / pdf_filename / timestamp_str)
            
            # Initialize structured logger with output directory
            self.structured_logger = StructuredJSONLogger(output_directory=output_dir)
            
            # Step 3: Vision+OCR
            ocr_output = self._run_vision_ocr(validated_ref, self.structured_logger)
            
            # Step 4: Script/Storyboard Architect
            storyboard = self._run_script_architect(ocr_output, self.structured_logger)
            
            # Step 5: JSON Sanitization
            sanitized_storyboard = self._run_json_sanitizer(storyboard, self.structured_logger)
            
            # Step 6: Scene Processing
            scene_output = self._run_scene_controller(
                sanitized_storyboard,
                validated_ref,
                self.structured_logger
            )
            
            # Step 7: Persistence
            manifest = self._run_persistence(
                scene_output,
                sanitized_storyboard,
                ocr_output if self.config.enable_ocr_debug else None,
                validated_ref,
                self.structured_logger
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
    
    def _run_script_architect(
        self,
        ocr_output,
        structured_logger: StructuredJSONLogger
    ):
        """Run Script Architect Agent with retry logic.
        
        Args:
            ocr_output: OCROutput from Vision+OCR Agent
            structured_logger: Logger for structured JSON logging
            
        Returns:
            StoryboardJSON object (as dict for sanitizer)
        """
        config = self.config.to_storyboard_config()
        input_data = ScriptArchitectInput(
            ocr_output=ocr_output,
            configuration=config
        )
        
        result = self._execute_with_retry(
            agent=self.script_architect_agent,
            input_data=input_data,
            stage_name="ScriptArchitect",
            structured_logger=structured_logger
        )
        
        return result
    
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
        # Determine output directory for this run
        pdf_filename = Path(file_ref.filename).stem  # Remove extension
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = str(Path(self.config.base_output_dir) / pdf_filename / timestamp_str)
        
        # Create input with animation generator
        from src.agents.animation_generator import AnimationCodeGeneratorAgent
        animation_generator = AnimationCodeGeneratorAgent()
        
        input_data = SceneControllerInput(
            storyboard=storyboard,
            output_directory=output_dir,
            animation_generator=animation_generator
        )
        
        result = self._execute_with_retry(
            agent=self.scene_controller_agent,
            input_data=input_data,
            stage_name="SceneController",
            structured_logger=structured_logger
        )
        
        return result
    
    def _run_persistence(
        self,
        scene_output,
        storyboard,
        ocr_output,
        file_ref: FileReference,
        structured_logger: StructuredJSONLogger
    ) -> OutputManifest:
        """Run Persistence Agent with retry logic.
        
        Args:
            scene_output: SceneControllerOutput object
            storyboard: StoryboardJSON object
            ocr_output: Optional OCROutput for debugging
            file_ref: FileReference for PDF filename
            structured_logger: Logger for structured JSON logging
            
        Returns:
            OutputManifest object
        """
        pdf_filename = Path(file_ref.filename).stem  # Remove extension
        
        input_data = PersistenceInput(
            scene_processing_report=scene_output.report,
            animation_code_objects=scene_output.animation_code_objects,
            storyboard=storyboard,
            ocr_output=ocr_output,
            pdf_filename=pdf_filename
        )
        
        result = self._execute_with_retry(
            agent=self.persistence_agent,
            input_data=input_data,
            stage_name="Persistence",
            structured_logger=structured_logger
        )
        
        return result.manifest
    
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
