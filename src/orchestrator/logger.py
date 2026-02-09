"""Structured JSON logger for pipeline observability.

This module provides structured logging functionality that writes JSON-formatted
log entries to pipeline.log in the output directory. Each log entry is a single
JSON object on one line, making it easy to parse and analyze.

Log Event Types:
- agent_start: Agent begins execution
- agent_complete: Agent completes successfully
- agent_failure: Agent encounters an error

Requirements:
- 12.1: Log agent_start events with agent_name, timestamp, input_summary
- 12.2: Log agent_complete events with agent_name, timestamp, duration_ms, output_summary, status
- 12.3: Log agent_failure events with agent_name, timestamp, error_message, error_code, input_context, retry_attempt
- 12.4: Write all logs to pipeline.log in output directory using structured JSON format
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredJSONLogger:
    """Structured JSON logger that writes to pipeline.log.
    
    This logger writes structured JSON log entries to a file in the output directory.
    Each log entry is a single JSON object on one line, following the format:
    
    {
        "event": "agent_start|agent_complete|agent_failure",
        "agent_name": "string",
        "timestamp": "ISO8601",
        ...additional fields based on event type...
    }
    
    The logger maintains both a file handler for JSON logs and a console handler
    for human-readable logs.
    """
    
    def __init__(self, output_directory: Optional[str] = None):
        """Initialize the structured JSON logger.
        
        Args:
            output_directory: Directory where pipeline.log will be written.
                            If None, only console logging is enabled.
        """
        self.output_directory = output_directory
        self.log_file_path = None
        self.json_file_handle = None
        
        # Setup log file if output directory is provided
        if output_directory:
            self._setup_log_file(output_directory)
        
        # Setup Python logger for console output
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_log_file(self, output_directory: str) -> None:
        """Setup the pipeline.log file in the output directory.
        
        Args:
            output_directory: Directory where pipeline.log will be written
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set log file path
        self.log_file_path = output_path / "pipeline.log"
        
        # Open file handle for appending JSON logs
        self.json_file_handle = open(self.log_file_path, 'a', encoding='utf-8')
    
    def _write_json_log(self, log_entry: Dict[str, Any]) -> None:
        """Write a JSON log entry to pipeline.log.
        
        Args:
            log_entry: Dictionary containing log data
        """
        if self.json_file_handle:
            # Write JSON object on a single line
            json_line = json.dumps(log_entry, ensure_ascii=False)
            self.json_file_handle.write(json_line + '\n')
            self.json_file_handle.flush()  # Ensure immediate write
    
    def log_agent_start(
        self,
        agent_name: str,
        input_summary: str,
        retry_attempt: int = 0
    ) -> None:
        """Log agent execution start event.
        
        Requirement 12.1: Log agent_start events with agent_name, timestamp, input_summary
        
        Args:
            agent_name: Name of the agent starting execution
            input_summary: Brief summary of input data
            retry_attempt: Retry attempt number (0 for first attempt)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "event": "agent_start",
            "agent_name": agent_name,
            "timestamp": timestamp,
            "input_summary": input_summary,
            "retry_attempt": retry_attempt
        }
        
        # Write to JSON log file
        self._write_json_log(log_entry)
        
        # Write to console
        attempt_str = f" (attempt {retry_attempt + 1})" if retry_attempt > 0 else ""
        self.logger.info(f"Starting {agent_name}{attempt_str}: {input_summary}")
    
    def log_agent_complete(
        self,
        agent_name: str,
        duration_ms: float,
        output_summary: str,
        status: str = "SUCCESS",
        retry_attempt: int = 0
    ) -> None:
        """Log agent execution completion event.
        
        Requirement 12.2: Log agent_complete events with agent_name, timestamp,
        duration_ms, output_summary, status
        
        Args:
            agent_name: Name of the agent that completed
            duration_ms: Execution duration in milliseconds
            output_summary: Brief summary of output data
            status: Execution status (SUCCESS or FAILURE)
            retry_attempt: Retry attempt number (0 for first attempt)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "event": "agent_complete",
            "agent_name": agent_name,
            "timestamp": timestamp,
            "duration_ms": round(duration_ms, 2),
            "output_summary": output_summary,
            "status": status,
            "retry_attempt": retry_attempt
        }
        
        # Write to JSON log file
        self._write_json_log(log_entry)
        
        # Write to console
        self.logger.info(
            f"Completed {agent_name} in {duration_ms:.2f}ms: {output_summary}"
        )
    
    def log_agent_failure(
        self,
        agent_name: str,
        error_message: str,
        error_code: str,
        input_context: str,
        retry_attempt: int = 0,
        duration_ms: Optional[float] = None
    ) -> None:
        """Log agent execution failure event.
        
        Requirement 12.3: Log agent_failure events with agent_name, timestamp,
        error_message, error_code, input_context, retry_attempt
        
        Args:
            agent_name: Name of the agent that failed
            error_message: Human-readable error message
            error_code: Machine-readable error code
            input_context: Relevant input excerpt for debugging
            retry_attempt: Retry attempt number (0 for first attempt)
            duration_ms: Optional execution duration in milliseconds
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "event": "agent_failure",
            "agent_name": agent_name,
            "timestamp": timestamp,
            "error_message": error_message,
            "error_code": error_code,
            "input_context": input_context,
            "retry_attempt": retry_attempt
        }
        
        if duration_ms is not None:
            log_entry["duration_ms"] = round(duration_ms, 2)
        
        # Write to JSON log file
        self._write_json_log(log_entry)
        
        # Write to console
        self.logger.error(
            f"Failed {agent_name} [{error_code}]: {error_message}"
        )
    
    def log_pipeline_start(
        self,
        pdf_file_path: str,
        config: Dict[str, Any]
    ) -> None:
        """Log pipeline execution start.
        
        Args:
            pdf_file_path: Path to PDF being processed
            config: Pipeline configuration
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "event": "pipeline_start",
            "timestamp": timestamp,
            "pdf_file_path": pdf_file_path,
            "config": config
        }
        
        self._write_json_log(log_entry)
        self.logger.info(f"Starting pipeline for PDF: {pdf_file_path}")
    
    def log_pipeline_complete(
        self,
        duration_seconds: float,
        output_directory: str,
        status: str,
        scene_success_rate: Optional[float] = None
    ) -> None:
        """Log pipeline execution completion.
        
        Args:
            duration_seconds: Total pipeline duration in seconds
            output_directory: Output directory path
            status: Pipeline status (SUCCESS, PARTIAL_SUCCESS, FAILURE)
            scene_success_rate: Optional success rate for scene processing
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "event": "pipeline_complete",
            "timestamp": timestamp,
            "duration_seconds": round(duration_seconds, 2),
            "output_directory": output_directory,
            "status": status
        }
        
        if scene_success_rate is not None:
            log_entry["scene_success_rate"] = round(scene_success_rate, 3)
        
        self._write_json_log(log_entry)
        self.logger.info(
            f"Pipeline completed with status {status} in {duration_seconds:.2f}s"
        )
    
    def log_pipeline_error(
        self,
        error_type: str,
        error_message: str,
        stage: Optional[str] = None
    ) -> None:
        """Log pipeline-level error.
        
        Args:
            error_type: Type of error (exception class name)
            error_message: Error message
            stage: Optional pipeline stage where error occurred
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "event": "pipeline_error",
            "timestamp": timestamp,
            "error_type": error_type,
            "error_message": error_message
        }
        
        if stage:
            log_entry["stage"] = stage
        
        self._write_json_log(log_entry)
        self.logger.error(f"Pipeline error: {error_message}")
    
    def close(self) -> None:
        """Close the log file handle."""
        if self.json_file_handle:
            self.json_file_handle.close()
            self.json_file_handle = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures log file is closed."""
        self.close()
        return False
