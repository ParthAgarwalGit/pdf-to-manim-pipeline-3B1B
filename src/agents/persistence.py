"""Persistence Agent for writing generated artifacts to file system.

This agent is responsible for:
- Creating deterministic output directory structure
- Writing animation code files atomically
- Writing narration scripts and metadata files
- Generating output manifest
- Ensuring idempotency through cleanup on retry
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.ocr_output import OCROutput
from src.schemas.output_manifest import FailedSceneManifest, OutputFile, OutputManifest
from src.schemas.scene_processing import SceneProcessingReport
from src.schemas.storyboard import StoryboardJSON


class PersistenceInput(AgentInput, BaseModel):
    """Input to Persistence Agent."""
    
    scene_processing_report: SceneProcessingReport = Field(
        ...,
        description="Report from Scene Controller with success/failure statistics"
    )
    animation_code_objects: List[AnimationCodeObject] = Field(
        ...,
        description="Array of successfully generated animation code objects"
    )
    storyboard: StoryboardJSON = Field(
        ...,
        description="Original storyboard JSON"
    )
    ocr_output: Optional[OCROutput] = Field(
        None,
        description="Optional OCR output for debugging"
    )
    pdf_filename: str = Field(
        ...,
        description="Original PDF filename (without extension)"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp for output directory (defaults to current time)"
    )


class PersistenceOutput(AgentOutput, BaseModel):
    """Output from Persistence Agent."""
    
    manifest: OutputManifest = Field(
        ...,
        description="Complete manifest of all generated files"
    )


class PersistenceAgent(Agent):
    """
    Persistence Agent writes all generated artifacts to deterministic file structure.
    
    Responsibilities:
    - Create output directory: output/{pdf_filename}/{timestamp}/
    - Write animation code files: scene_{scene_id}.py
    - Write narration script with scene markers
    - Write storyboard, processing report, and manifest
    - Ensure atomic file writes (no partial writes on failure)
    - Implement idempotency (cleanup on retry)
    """
    
    def __init__(self, base_output_dir: str = "output"):
        """
        Initialize Persistence Agent.
        
        Args:
            base_output_dir: Base directory for all output (default: "output")
        """
        self.base_output_dir = base_output_dir
    
    def execute(self, input_data: PersistenceInput) -> PersistenceOutput:
        """
        Execute persistence operations.
        
        Args:
            input_data: PersistenceInput with all artifacts to write
            
        Returns:
            PersistenceOutput with output manifest
            
        Raises:
            AgentExecutionError: For unrecoverable failures (directory creation, file write)
        """
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input validation failed for PersistenceAgent"
            )
        
        # Determine output directory path
        timestamp = input_data.timestamp or datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(
            self.base_output_dir,
            input_data.pdf_filename,
            timestamp_str
        )
        
        # Create directory structure (with cleanup for idempotency)
        try:
            self._create_output_directory(output_dir)
        except Exception as e:
            raise AgentExecutionError(
                "DIRECTORY_CREATION_FAILED",
                f"Failed to create output directory: {str(e)}",
                {"output_dir": output_dir}
            )
        
        # Track all written files
        written_files: List[OutputFile] = []
        
        # Write animation code files
        try:
            code_files = self._write_animation_code_files(
                output_dir,
                input_data.animation_code_objects
            )
            written_files.extend(code_files)
        except Exception as e:
            raise AgentExecutionError(
                "FILE_WRITE_FAILED",
                f"Failed to write animation code files: {str(e)}",
                {"output_dir": output_dir}
            )
        
        # Write narration script
        try:
            narration_file = self._write_narration_script(
                output_dir,
                input_data.storyboard
            )
            written_files.append(narration_file)
        except Exception as e:
            raise AgentExecutionError(
                "FILE_WRITE_FAILED",
                f"Failed to write narration script: {str(e)}",
                {"output_dir": output_dir}
            )
        
        # Write storyboard JSON
        try:
            storyboard_file = self._write_storyboard(
                output_dir,
                input_data.storyboard
            )
            written_files.append(storyboard_file)
        except Exception as e:
            raise AgentExecutionError(
                "FILE_WRITE_FAILED",
                f"Failed to write storyboard: {str(e)}",
                {"output_dir": output_dir}
            )
        
        # Write processing report
        try:
            report_file = self._write_processing_report(
                output_dir,
                input_data.scene_processing_report
            )
            written_files.append(report_file)
        except Exception as e:
            raise AgentExecutionError(
                "FILE_WRITE_FAILED",
                f"Failed to write processing report: {str(e)}",
                {"output_dir": output_dir}
            )
        
        # Optionally write OCR output for debugging
        if input_data.ocr_output:
            try:
                ocr_file = self._write_ocr_output(
                    output_dir,
                    input_data.ocr_output
                )
                written_files.append(ocr_file)
            except Exception as e:
                # OCR output is optional, log but don't fail
                pass
        
        # Generate and write manifest
        try:
            # Generate manifest with current files
            manifest = self._generate_manifest(
                output_dir,
                written_files,
                input_data.scene_processing_report,
                timestamp
            )
            
            # Write manifest and get actual file info
            manifest_file = self._write_manifest(output_dir, manifest)
            written_files.append(manifest_file)
            
            # Update manifest with complete file list (including manifest itself)
            manifest = self._generate_manifest(
                output_dir,
                written_files,
                input_data.scene_processing_report,
                timestamp
            )
            
            # Write final manifest with complete file list
            self._write_manifest(output_dir, manifest)
        except Exception as e:
            raise AgentExecutionError(
                "FILE_WRITE_FAILED",
                f"Failed to write manifest: {str(e)}",
                {"output_dir": output_dir}
            )
        
        return PersistenceOutput(manifest=manifest)
    
    def validate_input(self, input_data: PersistenceInput) -> bool:
        """
        Validate input conforms to expected schema.
        
        Args:
            input_data: PersistenceInput to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, PersistenceInput):
            return False
        
        # Validate required fields are present
        if not input_data.scene_processing_report:
            return False
        if not input_data.storyboard:
            return False
        if not input_data.pdf_filename:
            return False
        
        # animation_code_objects can be empty if all scenes failed
        if not isinstance(input_data.animation_code_objects, list):
            return False
        
        return True
    
    def get_retry_policy(self) -> RetryPolicy:
        """
        Return retry policy for Persistence Agent.
        
        Retry transient failures (temporary disk issues) but not
        deterministic failures (permissions, invalid paths).
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=10.0,
            retryable_errors=["TEMPORARY_DISK_ERROR", "NETWORK_STORAGE_TIMEOUT"]
        )
    
    def _create_output_directory(self, output_dir: str) -> None:
        """
        Create output directory structure with cleanup for idempotency.
        
        Args:
            output_dir: Path to output directory
            
        Raises:
            Exception: If directory creation fails
        """
        output_path = Path(output_dir)
        
        # If directory exists, clean up incomplete files for idempotency
        if output_path.exists():
            self._cleanup_incomplete_files(output_dir)
        else:
            # Create directory structure
            output_path.mkdir(parents=True, exist_ok=True)
    
    def _cleanup_incomplete_files(self, output_dir: str) -> None:
        """
        Clean up incomplete files from previous failed run.
        
        This method ensures idempotency by removing all files except
        pipeline.log from the output directory. It handles both files
        and subdirectories that might have been created.
        
        Args:
            output_dir: Path to output directory
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return
        
        # Remove all files and directories except pipeline.log (keep logs for debugging)
        for item_path in output_path.iterdir():
            if item_path.name == "pipeline.log":
                continue
                
            try:
                if item_path.is_file():
                    item_path.unlink()
                elif item_path.is_dir():
                    # Remove directory and all its contents
                    import shutil
                    shutil.rmtree(item_path)
            except Exception:
                # Best effort cleanup, don't fail if item can't be removed
                # This could happen due to permissions or file locks
                pass
    
    def _write_file_atomically(self, file_path: str, content: str) -> None:
        """
        Write file atomically using temp file and rename.
        
        This ensures no partial writes occur on failure and provides
        idempotency by completely overwriting existing files.
        
        Args:
            file_path: Destination file path
            content: File content to write
            
        Raises:
            Exception: If write fails
        """
        file_path_obj = Path(file_path)
        temp_file_path = None
        
        try:
            # Write to temporary file in same directory
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=file_path_obj.parent,
                delete=False,
                encoding='utf-8',
                suffix='.tmp'
            ) as temp_file:
                temp_file.write(content)
                temp_file.flush()  # Ensure content is written to disk
                os.fsync(temp_file.fileno())  # Force write to disk
                temp_file_path = temp_file.name
            
            # Atomically rename temp file to destination
            # This is atomic on most filesystems and overwrites existing files
            os.replace(temp_file_path, file_path)
            
        except Exception:
            # Clean up temp file if something went wrong
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    # Best effort cleanup
                    pass
            raise
    
    def _write_animation_code_files(
        self,
        output_dir: str,
        animation_code_objects: List[AnimationCodeObject]
    ) -> List[OutputFile]:
        """
        Write animation code files for each scene.
        
        Args:
            output_dir: Output directory path
            animation_code_objects: List of animation code objects
            
        Returns:
            List of OutputFile objects for written files
        """
        written_files = []
        
        for code_obj in animation_code_objects:
            # Generate filename: scene_{scene_id}.py
            filename = f"scene_{code_obj.scene_id}.py"
            file_path = os.path.join(output_dir, filename)
            
            # Generate file content with header comment
            content = self._generate_scene_file_content(code_obj)
            
            # Write atomically
            self._write_file_atomically(file_path, content)
            
            # Track written file
            file_size = os.path.getsize(file_path)
            written_files.append(OutputFile(
                path=filename,
                size_bytes=file_size,
                type="animation_code"
            ))
        
        return written_files
    
    def _generate_scene_file_content(self, code_obj: AnimationCodeObject) -> str:
        """
        Generate complete scene file content with header comment.
        
        Args:
            code_obj: Animation code object
            
        Returns:
            Complete file content as string
        """
        header = f'''"""
Scene: {code_obj.scene_id}
Generated: {code_obj.generation_timestamp.isoformat()}

Narration:
{code_obj.narration}

Duration: {code_obj.duration_estimate}s
"""

'''
        
        # Add imports
        imports = '\n'.join(code_obj.imports)
        
        # Combine header, imports, and code
        content = header + imports + '\n\n' + code_obj.code + '\n'
        
        return content
    
    def _write_narration_script(
        self,
        output_dir: str,
        storyboard: StoryboardJSON
    ) -> OutputFile:
        """
        Write narration script with scene markers.
        
        Args:
            output_dir: Output directory path
            storyboard: Storyboard JSON with scenes
            
        Returns:
            OutputFile object for written file
        """
        filename = "narration_script.txt"
        file_path = os.path.join(output_dir, filename)
        
        # Generate narration script content
        lines = []
        lines.append("=" * 80)
        lines.append("NARRATION SCRIPT")
        lines.append(f"PDF: {storyboard.pdf_metadata.filename}")
        lines.append(f"Total Scenes: {len(storyboard.scenes)}")
        lines.append("=" * 80)
        lines.append("")
        
        for scene in storyboard.scenes:
            lines.append(f"[SCENE {scene.scene_index + 1}: {scene.scene_id}]")
            lines.append(f"Duration: {scene.duration_estimate}s")
            lines.append("-" * 80)
            lines.append(scene.narration)
            lines.append("")
            lines.append("=" * 80)
            lines.append("")
        
        content = '\n'.join(lines)
        
        # Write atomically
        self._write_file_atomically(file_path, content)
        
        # Track written file
        file_size = os.path.getsize(file_path)
        return OutputFile(
            path=filename,
            size_bytes=file_size,
            type="narration_script"
        )
    
    def _write_storyboard(
        self,
        output_dir: str,
        storyboard: StoryboardJSON
    ) -> OutputFile:
        """
        Write storyboard JSON file.
        
        Args:
            output_dir: Output directory path
            storyboard: Storyboard JSON object
            
        Returns:
            OutputFile object for written file
        """
        filename = "storyboard.json"
        file_path = os.path.join(output_dir, filename)
        
        # Convert to JSON with pretty printing
        content = storyboard.model_dump_json(indent=2)
        
        # Write atomically
        self._write_file_atomically(file_path, content)
        
        # Track written file
        file_size = os.path.getsize(file_path)
        return OutputFile(
            path=filename,
            size_bytes=file_size,
            type="storyboard"
        )
    
    def _write_processing_report(
        self,
        output_dir: str,
        report: SceneProcessingReport
    ) -> OutputFile:
        """
        Write scene processing report JSON file.
        
        Args:
            output_dir: Output directory path
            report: Scene processing report
            
        Returns:
            OutputFile object for written file
        """
        filename = "processing_report.json"
        file_path = os.path.join(output_dir, filename)
        
        # Convert to JSON with pretty printing
        content = report.model_dump_json(indent=2)
        
        # Write atomically
        self._write_file_atomically(file_path, content)
        
        # Track written file
        file_size = os.path.getsize(file_path)
        return OutputFile(
            path=filename,
            size_bytes=file_size,
            type="report"
        )
    
    def _write_ocr_output(
        self,
        output_dir: str,
        ocr_output: OCROutput
    ) -> OutputFile:
        """
        Write OCR output JSON file (for debugging).
        
        Args:
            output_dir: Output directory path
            ocr_output: OCR output object
            
        Returns:
            OutputFile object for written file
        """
        filename = "ocr_output.json"
        file_path = os.path.join(output_dir, filename)
        
        # Convert to JSON with pretty printing
        content = ocr_output.model_dump_json(indent=2)
        
        # Write atomically
        self._write_file_atomically(file_path, content)
        
        # Track written file
        file_size = os.path.getsize(file_path)
        return OutputFile(
            path=filename,
            size_bytes=file_size,
            type="ocr_output"
        )
    
    def _generate_manifest(
        self,
        output_dir: str,
        written_files: List[OutputFile],
        report: SceneProcessingReport,
        timestamp: datetime
    ) -> OutputManifest:
        """
        Generate output manifest.
        
        Args:
            output_dir: Output directory path
            written_files: List of written files
            report: Scene processing report
            timestamp: Generation timestamp
            
        Returns:
            OutputManifest object
        """
        # Convert absolute path to absolute
        abs_output_dir = os.path.abspath(output_dir)
        
        # Convert failed scenes from report to manifest format
        failed_scenes = [
            FailedSceneManifest(scene_id=fs.scene_id, error=fs.error)
            for fs in report.failed_scenes
        ]
        
        manifest = OutputManifest(
            output_directory=abs_output_dir,
            files=written_files,
            timestamp=timestamp,
            status=report.status,
            scene_success_rate=report.scene_success_rate,
            total_scenes=report.total_scenes,
            successful_scenes=report.successful_scenes,
            failed_scenes=failed_scenes
        )
        
        return manifest
    
    def _write_manifest(
        self,
        output_dir: str,
        manifest: OutputManifest
    ) -> OutputFile:
        """
        Write manifest JSON file.
        
        Args:
            output_dir: Output directory path
            manifest: Output manifest object
            
        Returns:
            OutputFile object for the written manifest file
        """
        filename = "manifest.json"
        file_path = os.path.join(output_dir, filename)
        
        # Convert to JSON with pretty printing
        content = manifest.model_dump_json(indent=2)
        
        # Write atomically
        self._write_file_atomically(file_path, content)
        
        # Return file info with actual size
        file_size = os.path.getsize(file_path)
        return OutputFile(
            path=filename,
            size_bytes=file_size,
            type="manifest"
        )
