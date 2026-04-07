"""Async wrapper for the synchronous pipeline with web deployment support.

This module provides async capabilities for the PDF-to-Manim pipeline, enabling
concurrent execution in web environments. It wraps the synchronous Orchestrator
class and provides:

1. AsyncPipelineRunner: Async execution with progress callbacks for WebSocket updates
2. TaskQueue: Manages multiple concurrent pipeline runs with configurable limits
3. PipelineStatus: Enum tracking pipeline execution stages
4. PipelineResult: Dataclass with execution results and cost/timing information

The async wrapper uses asyncio.to_thread() to run the synchronous pipeline in
a thread executor, preventing blocking of the event loop.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional
from uuid import uuid4

from src.orchestrator.pipeline import Orchestrator, PipelineConfig, PipelineAbortError
from src.schemas.output_manifest import OutputManifest


logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status enumeration.

    Attributes:
        QUEUED: Task waiting for execution
        INGESTING: Running ingestion agent (cloud storage event processing)
        VALIDATING: Running validation agent (PDF structure validation)
        EXTRACTING_OCR: Running vision+OCR agent (text and diagram extraction)
        GENERATING_STORYBOARD: Running script architect (pedagogical narrative)
        SANITIZING: Running JSON sanitizer (schema validation and correction)
        GENERATING_SCENES: Running scene controller (animation code generation)
        RENDERING: Running rendering (manim compilation and video generation)
        GENERATING_TTS: Running text-to-speech synthesis (narration generation)
        COMPOSITING: Running compositing (final video assembly with audio)
        COMPLETE: Pipeline completed successfully
        FAILED: Pipeline failed with error
    """
    QUEUED = "queued"
    INGESTING = "ingesting"
    VALIDATING = "validating"
    EXTRACTING_OCR = "extracting_ocr"
    GENERATING_STORYBOARD = "generating_storyboard"
    SANITIZING = "sanitizing"
    GENERATING_SCENES = "generating_scenes"
    RENDERING = "rendering"
    GENERATING_TTS = "generating_tts"
    COMPOSITING = "compositing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Result of a pipeline execution.

    Attributes:
        output_dir: Directory containing all generated outputs
        video_path: Path to the final rendered video (if successful)
        cost_summary: Dict with cost breakdown by agent/stage
        duration_seconds: Total execution time in seconds
        status: Final PipelineStatus
        error_message: Error message (if status is FAILED)
        manifest: The OutputManifest object from the pipeline (if successful)
    """
    output_dir: str
    video_path: Optional[str] = None
    cost_summary: Optional[Dict[str, float]] = None
    duration_seconds: float = 0.0
    status: PipelineStatus = PipelineStatus.QUEUED
    error_message: Optional[str] = None
    manifest: Optional[OutputManifest] = None


class AsyncPipelineRunner:
    """Async wrapper for the synchronous pipeline orchestrator.

    Runs the pipeline in a thread executor to prevent blocking, with progress
    tracking and WebSocket callback support for real-time status updates.

    Example:
        runner = AsyncPipelineRunner(config)

        def on_progress(status, percentage):
            print(f"{status.value}: {percentage}%")

        result = await runner.run(
            "path/to/document.pdf",
            on_progress=on_progress
        )
    """

    # Progress milestones for each pipeline stage (percentage complete)
    PROGRESS_MILESTONES = {
        PipelineStatus.QUEUED: 0,
        PipelineStatus.INGESTING: 5,
        PipelineStatus.VALIDATING: 10,
        PipelineStatus.EXTRACTING_OCR: 20,
        PipelineStatus.GENERATING_STORYBOARD: 35,
        PipelineStatus.SANITIZING: 45,
        PipelineStatus.GENERATING_SCENES: 60,
        PipelineStatus.RENDERING: 80,
        PipelineStatus.GENERATING_TTS: 85,
        PipelineStatus.COMPOSITING: 95,
        PipelineStatus.COMPLETE: 100,
    }

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the async pipeline runner.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or PipelineConfig()
        self.orchestrator = Orchestrator(config=self.config)
        self._progress_callback: Optional[Callable] = None
        self._current_status: PipelineStatus = PipelineStatus.QUEUED
        self._current_progress: float = 0.0

    async def run(
        self,
        pdf_path: str,
        on_progress: Optional[Callable[[PipelineStatus, float], None]] = None,
        event_source: str = "local"
    ) -> PipelineResult:
        """Run the pipeline asynchronously.

        This method runs the synchronous pipeline in a thread executor, allowing
        progress callbacks to be called from the async context without blocking
        the event loop.

        Args:
            pdf_path: Path to the PDF file to process
            on_progress: Optional callback function(status, percentage) for progress updates
            event_source: Cloud storage event source ('s3', 'gcs', 'azure', or 'local')

        Returns:
            PipelineResult with output information and status
        """
        self._progress_callback = on_progress
        start_time = time.time()

        try:
            # Set initial status
            await self._update_progress(PipelineStatus.QUEUED)

            # Run the synchronous pipeline in a thread executor
            logger.info(f"Starting async pipeline for PDF: {pdf_path}")
            manifest = await asyncio.to_thread(
                self.orchestrator.run_pipeline,
                pdf_path,
                event_source
            )

            # Determine output directory and video path from manifest
            output_dir = manifest.output_directory
            video_path = self._find_video_path(output_dir, manifest)

            # Update to complete
            await self._update_progress(PipelineStatus.COMPLETE)

            duration = time.time() - start_time

            logger.info(
                f"Pipeline completed successfully in {duration:.2f}s",
                extra={
                    "output_dir": output_dir,
                    "video_path": video_path
                }
            )

            return PipelineResult(
                output_dir=output_dir,
                video_path=video_path,
                cost_summary=self._extract_cost_summary(manifest),
                duration_seconds=duration,
                status=PipelineStatus.COMPLETE,
                error_message=None,
                manifest=manifest
            )

        except PipelineAbortError as e:
            duration = time.time() - start_time
            await self._update_progress(PipelineStatus.FAILED)

            error_msg = f"{e.stage}: [{e.error_code}] {e.message}"
            logger.error(
                error_msg,
                extra={
                    "stage": e.stage,
                    "error_code": e.error_code,
                    "context": e.context
                }
            )

            return PipelineResult(
                output_dir="",
                video_path=None,
                cost_summary=None,
                duration_seconds=duration,
                status=PipelineStatus.FAILED,
                error_message=error_msg,
                manifest=None
            )

        except Exception as e:
            duration = time.time() - start_time
            await self._update_progress(PipelineStatus.FAILED)

            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            logger.error(
                error_msg,
                extra={"error_type": type(e).__name__},
                exc_info=True
            )

            return PipelineResult(
                output_dir="",
                video_path=None,
                cost_summary=None,
                duration_seconds=duration,
                status=PipelineStatus.FAILED,
                error_message=error_msg,
                manifest=None
            )

    async def _update_progress(
        self,
        status: PipelineStatus,
        progress: Optional[float] = None
    ) -> None:
        """Update pipeline progress and trigger callback.

        Args:
            status: Current pipeline status
            progress: Optional explicit progress percentage (0-100)
        """
        self._current_status = status

        if progress is None:
            # Use milestone-based progress
            progress = self.PROGRESS_MILESTONES.get(status, self._current_progress)

        self._current_progress = min(100.0, max(0.0, progress))

        # Call the callback if provided
        if self._progress_callback:
            try:
                # Allow both sync and async callbacks
                result = self._progress_callback(status, self._current_progress)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}", exc_info=True)

    def _find_video_path(self, output_dir: str, manifest: OutputManifest) -> Optional[str]:
        """Find the final video file path from the output directory.

        Args:
            output_dir: Base output directory
            manifest: OutputManifest with file information

        Returns:
            Path to the video file, or None if not found
        """
        # Common video file extensions to look for
        video_extensions = ['.mp4', '.webm', '.mov', '.avi']

        # Try to find from manifest first
        if hasattr(manifest, 'video_file') and manifest.video_file:
            return manifest.video_file

        # Search output directory for video files
        try:
            output_path = Path(output_dir)
            if output_path.exists():
                for ext in video_extensions:
                    video_files = list(output_path.rglob(f'*{ext}'))
                    if video_files:
                        return str(video_files[0])
        except Exception as e:
            logger.warning(f"Error searching for video file: {e}")

        return None

    def _extract_cost_summary(self, manifest: OutputManifest) -> Dict[str, float]:
        """Extract cost summary from the manifest.

        Args:
            manifest: OutputManifest with execution data

        Returns:
            Dict mapping cost categories to amounts
        """
        cost_summary = {}

        if hasattr(manifest, 'cost_summary'):
            cost_summary = manifest.cost_summary or {}

        return cost_summary


class TaskQueue:
    """Queue for managing multiple concurrent pipeline executions.

    Limits the number of concurrent pipeline runs based on configuration,
    using asyncio.Queue internally. Provides methods to submit tasks, check
    status, and retrieve results.

    Example:
        queue = TaskQueue(max_concurrent=2)

        task_id = queue.submit("path/to/document.pdf", config)
        status = queue.get_status(task_id)
        result = queue.get_result(task_id)
    """

    def __init__(self, max_concurrent: int = 2):
        """Initialize the task queue.

        Args:
            max_concurrent: Maximum number of concurrent pipeline runs
        """
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, PipelineResult] = {}
        self._status: Dict[str, PipelineStatus] = {}
        self._workers_started = False

    async def start_workers(self) -> None:
        """Start the worker tasks that process the queue.

        Should be called once at application startup. Can be safely called
        multiple times (only starts workers once).
        """
        if self._workers_started:
            return

        self._workers_started = True

        # Create worker tasks
        for _ in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker())
            logger.info(f"Started pipeline worker (max_concurrent={self.max_concurrent})")

    async def _worker(self) -> None:
        """Worker coroutine that processes tasks from the queue."""
        while True:
            try:
                # Get next task from queue
                task_id, pdf_path, config = await self._queue.get()

                logger.info(f"Processing task {task_id}: {pdf_path}")

                try:
                    # Run the pipeline
                    runner = AsyncPipelineRunner(config=config)

                    # Create a progress callback that updates status
                    async def on_progress(status: PipelineStatus, progress: float):
                        self._status[task_id] = status
                        logger.debug(f"Task {task_id}: {status.value} ({progress:.0f}%)")

                    result = await runner.run(
                        pdf_path,
                        on_progress=on_progress
                    )

                    # Store result
                    self._results[task_id] = result
                    self._status[task_id] = result.status

                    logger.info(
                        f"Task {task_id} completed with status {result.status.value}",
                        extra={"duration": result.duration_seconds}
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing task {task_id}: {e}",
                        exc_info=True
                    )

                    # Store error result
                    self._results[task_id] = PipelineResult(
                        output_dir="",
                        video_path=None,
                        duration_seconds=0.0,
                        status=PipelineStatus.FAILED,
                        error_message=str(e)
                    )
                    self._status[task_id] = PipelineStatus.FAILED

                finally:
                    self._queue.task_done()

            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                # Worker should continue running despite errors
                await asyncio.sleep(0.1)

    def submit(
        self,
        pdf_path: str,
        config: Optional[PipelineConfig] = None
    ) -> str:
        """Submit a PDF for processing.

        Args:
            pdf_path: Path to the PDF file
            config: Optional pipeline configuration

        Returns:
            Task ID that can be used to check status and get results
        """
        if not self._workers_started:
            raise RuntimeError(
                "TaskQueue workers not started. Call await queue.start_workers() first."
            )

        task_id = str(uuid4())
        config = config or PipelineConfig()

        # Put task in queue
        self._queue.put_nowait((task_id, pdf_path, config))

        # Initialize status and result
        self._status[task_id] = PipelineStatus.QUEUED

        logger.info(f"Submitted task {task_id}: {pdf_path}")

        return task_id

    def get_status(self, task_id: str) -> Optional[PipelineStatus]:
        """Get the current status of a task.

        Args:
            task_id: Task ID from submit()

        Returns:
            Current PipelineStatus, or None if task not found
        """
        return self._status.get(task_id)

    def get_result(self, task_id: str) -> Optional[PipelineResult]:
        """Get the result of a completed task.

        Args:
            task_id: Task ID from submit()

        Returns:
            PipelineResult if task completed, None if still running or not found
        """
        return self._results.get(task_id)

    def is_complete(self, task_id: str) -> bool:
        """Check if a task has completed (successfully or with error).

        Args:
            task_id: Task ID from submit()

        Returns:
            True if task is complete, False if still running or not found
        """
        status = self._status.get(task_id)
        return status in (PipelineStatus.COMPLETE, PipelineStatus.FAILED)

    async def wait_for_completion(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> PipelineResult:
        """Wait for a task to complete and return its result.

        Args:
            task_id: Task ID from submit()
            timeout: Optional timeout in seconds

        Returns:
            PipelineResult

        Raises:
            TimeoutError: If timeout is exceeded
            ValueError: If task not found
        """
        start_time = time.time()

        while True:
            # Check for timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

            # Check if task is complete
            if self.is_complete(task_id):
                result = self._results.get(task_id)
                if result is None:
                    raise ValueError(f"Task {task_id} not found")
                return result

            # Wait a bit before checking again
            await asyncio.sleep(0.1)

    async def shutdown(self) -> None:
        """Shutdown the task queue and wait for all pending tasks to complete.

        This should be called during application shutdown to ensure graceful
        termination of all pipeline runs.
        """
        logger.info("Shutting down task queue")

        # Wait for all items in queue to be processed
        await self._queue.join()

        logger.info("Task queue shutdown complete")
