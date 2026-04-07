"""Ingestion Agent for PDF to Manim pipeline

This module implements the Ingestion Agent responsible for processing local file
ingestion and routing files to validation. It extracts file metadata from
local file paths and constructs FileReference objects.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.file_reference import FileReference


logger = logging.getLogger(__name__)


class CloudStorageEvent(AgentInput):
    """Input for Ingestion Agent: local file event

    Attributes:
        event_payload: Event payload in local file format
        event_source: Source of the event (always 'local' for MVP)
    """

    def __init__(self, event_payload: Dict[str, Any], event_source: str = 'local'):
        self.event_payload = event_payload
        self.event_source = event_source


class IngestionOutput(AgentOutput):
    """Output from Ingestion Agent: file reference object
    
    Attributes:
        file_reference: FileReference object with extracted metadata
    """
    
    def __init__(self, file_reference: FileReference):
        self.file_reference = file_reference


class IngestionError(AgentOutput):
    """Error output from Ingestion Agent
    
    Attributes:
        error_code: Machine-readable error code
        reason: Human-readable error description
        event_payload_sample: Sample of the malformed payload (for debugging)
    """
    
    def __init__(self, error_code: str, reason: str, event_payload_sample: Optional[str] = None):
        self.error_code = error_code
        self.reason = reason
        self.event_payload_sample = event_payload_sample


class IngestAgent(Agent):
    """Ingestion Agent for processing local file ingestion

    Responsibilities:
    - Process local file ingestion events
    - Extract file metadata from file paths
    - Construct FileReference objects
    - Log ingestion events with timestamps

    Failure Modes:
    - Malformed event payload → Return error, skip file
    - File not found or inaccessible → Log error, skip file
    """

    def __init__(self):
        """Initialize the Ingestion Agent"""
        self.logger = logging.getLogger(f"{__name__}.IngestAgent")

    @classmethod
    def from_file_path(cls, file_path: str) -> 'CloudStorageEvent':
        """Create a CloudStorageEvent from a local file path.

        This is a convenience method for the MVP that avoids needing to construct
        the event payload manually for local file ingestion.

        Args:
            file_path: Absolute or relative path to a local file

        Returns:
            CloudStorageEvent with the file path embedded in event payload

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file path is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Get file size in bytes
        file_size = path.stat().st_size

        # Create event payload in local format
        event_payload = {
            'Records': [
                {
                    's3': {
                        'object': {
                            'key': str(path.resolve()),
                            'size': file_size
                        }
                    },
                    'eventTime': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    'eventName': 'ObjectCreated:Put'
                }
            ]
        }

        return CloudStorageEvent(event_payload=event_payload, event_source='local')
    
    def execute(self, input_data: CloudStorageEvent) -> AgentOutput:
        """Execute ingestion logic: extract metadata and create FileReference
        
        Args:
            input_data: CloudStorageEvent containing event payload
            
        Returns:
            IngestionOutput with FileReference OR IngestionError
            
        Raises:
            AgentExecutionError: For unrecoverable failures
        """
        if not self.validate_input(input_data):
            return IngestionError(
                error_code="INVALID_INPUT",
                reason="Input is not a valid CloudStorageEvent",
                event_payload_sample=None
            )
        
        # Log ingestion event start
        self.logger.info(
            "Ingestion started",
            extra={
                "event": "ingestion_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_source": input_data.event_source
            }
        )
        
        try:
            # Extract metadata from local file event
            if input_data.event_source != 'local':
                return IngestionError(
                    error_code="UNSUPPORTED_EVENT_SOURCE",
                    reason=f"Event source '{input_data.event_source}' is not supported (only 'local' is supported)",
                    event_payload_sample=json.dumps(input_data.event_payload)[:200]
                )

            file_reference = self._extract_local_metadata(input_data.event_payload)
            
            # Log successful ingestion
            # NOTE: "filename" is a reserved LogRecord attr — use "file_name" instead
            self.logger.info(
                "Ingestion completed successfully",
                extra={
                    "event": "ingestion_complete",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "file_path": file_reference.file_path,
                    "file_name": file_reference.filename,
                    "size_bytes": file_reference.size_bytes
                }
            )
            
            return IngestionOutput(file_reference=file_reference)
            
        except KeyError as e:
            # Malformed event payload - missing required fields
            error_msg = f"Malformed event payload: missing required field '{e.args[0]}'"
            self.logger.error(
                error_msg,
                extra={
                    "event": "ingestion_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_code": "MALFORMED_PAYLOAD",
                    "payload_sample": json.dumps(input_data.event_payload)[:200]
                }
            )
            return IngestionError(
                error_code="MALFORMED_PAYLOAD",
                reason=error_msg,
                event_payload_sample=json.dumps(input_data.event_payload)[:200]
            )
            
        except ValueError as e:
            # Invalid data in payload (e.g., negative file size)
            error_msg = f"Invalid data in event payload: {str(e)}"
            self.logger.error(
                error_msg,
                extra={
                    "event": "ingestion_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_code": "INVALID_PAYLOAD_DATA",
                    "payload_sample": json.dumps(input_data.event_payload)[:200]
                }
            )
            return IngestionError(
                error_code="INVALID_PAYLOAD_DATA",
                reason=error_msg,
                event_payload_sample=json.dumps(input_data.event_payload)[:200]
            )
            
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error during ingestion: {str(e)}"
            self.logger.error(
                error_msg,
                extra={
                    "event": "ingestion_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_code": "UNEXPECTED_ERROR",
                    "error_type": type(e).__name__
                }
            )
            raise AgentExecutionError(
                error_code="UNEXPECTED_ERROR",
                message=error_msg,
                context={"error_type": type(e).__name__}
            )

    def _extract_local_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Extract metadata from local file event

        Args:
            event_payload: Local file event payload (S3-like format)

        Returns:
            FileReference object with extracted metadata

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Local event structure mimics S3: Records[0].s3.object
        record = event_payload['Records'][0]
        s3_info = record['s3']

        # For local files, the object key is the actual file path
        file_path = s3_info['object']['key']
        file_size = s3_info['object']['size']

        # Extract filename from path
        filename = Path(file_path).name

        # Parse event time (ISO8601 format)
        event_time_str = record['eventTime']
        upload_timestamp = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))

        # Extract additional metadata
        metadata = {
            'source': 'local',
            'event_name': record.get('eventName', 'LocalFile')
        }

        return FileReference(
            file_path=file_path,  # Keep as local path, not S3 URL
            filename=filename,
            size_bytes=file_size,
            upload_timestamp=upload_timestamp,
            metadata=metadata
        )

    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input is a CloudStorageEvent with required fields
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, CloudStorageEvent):
            return False
        
        if not hasattr(input_data, 'event_payload') or not isinstance(input_data.event_payload, dict):
            return False
        
        if not hasattr(input_data, 'event_source') or not isinstance(input_data.event_source, str):
            return False
        
        return True
    
    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for Ingestion Agent
        
        Returns:
            RetryPolicy with exponential backoff for transient failures
            
        Note:
            Retries are appropriate for event notification system unavailability.
            Malformed payloads and missing files should not be retried.
        """
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            retryable_errors=[
                "EVENT_SYSTEM_UNAVAILABLE",
                "NETWORK_TIMEOUT",
                "TEMPORARY_SERVICE_UNAVAILABLE"
            ]
        )
