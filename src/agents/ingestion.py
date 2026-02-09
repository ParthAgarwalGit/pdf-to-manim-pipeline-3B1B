"""Ingestion Agent for PDF to Manim pipeline

This module implements the Ingestion Agent responsible for detecting cloud storage
upload events and routing files to validation. It extracts file metadata from
event payloads and constructs FileReference objects.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.file_reference import FileReference


logger = logging.getLogger(__name__)


class CloudStorageEvent(AgentInput):
    """Input for Ingestion Agent: cloud storage event notification
    
    Attributes:
        event_payload: Raw event payload from cloud storage service
        event_source: Source of the event (e.g., 's3', 'gcs', 'azure')
    """
    
    def __init__(self, event_payload: Dict[str, Any], event_source: str = 's3'):
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
    """Ingestion Agent for detecting and processing cloud storage upload events
    
    Responsibilities:
    - Subscribe to cloud storage upload events
    - Extract file metadata from event payloads
    - Construct FileReference objects
    - Log ingestion events with timestamps
    
    Failure Modes:
    - Event notification system unavailable → Retry with exponential backoff
    - Malformed event payload → Return error, skip file
    - File deleted before processing → Log warning, skip file
    """
    
    def __init__(self):
        """Initialize the Ingestion Agent"""
        self.logger = logging.getLogger(f"{__name__}.IngestAgent")
    
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
            # Extract metadata based on event source
            if input_data.event_source == 's3':
                file_reference = self._extract_s3_metadata(input_data.event_payload)
            elif input_data.event_source == 'gcs':
                file_reference = self._extract_gcs_metadata(input_data.event_payload)
            elif input_data.event_source == 'azure':
                file_reference = self._extract_azure_metadata(input_data.event_payload)
            else:
                return IngestionError(
                    error_code="UNSUPPORTED_EVENT_SOURCE",
                    reason=f"Event source '{input_data.event_source}' is not supported",
                    event_payload_sample=json.dumps(input_data.event_payload)[:200]
                )
            
            # Log successful ingestion
            self.logger.info(
                "Ingestion completed successfully",
                extra={
                    "event": "ingestion_complete",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "file_path": file_reference.file_path,
                    "filename": file_reference.filename,
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
    
    def _extract_s3_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Extract metadata from S3 event notification
        
        Args:
            event_payload: S3 event notification payload
            
        Returns:
            FileReference object with extracted metadata
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # S3 event structure: Records[0].s3.object and Records[0].s3.bucket
        record = event_payload['Records'][0]
        s3_info = record['s3']
        
        bucket_name = s3_info['bucket']['name']
        object_key = s3_info['object']['key']
        file_size = s3_info['object']['size']
        
        # Extract filename from object key
        filename = object_key.split('/')[-1]
        
        # Construct full S3 path
        file_path = f"s3://{bucket_name}/{object_key}"
        
        # Parse event time (ISO8601 format)
        event_time_str = record['eventTime']
        upload_timestamp = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
        
        # Extract additional metadata
        metadata = {
            'bucket': bucket_name,
            'object_key': object_key,
            'event_name': record.get('eventName', 'Unknown'),
            'region': record.get('awsRegion', 'Unknown')
        }
        
        # Add etag if available
        if 'eTag' in s3_info['object']:
            metadata['etag'] = s3_info['object']['eTag']
        
        return FileReference(
            file_path=file_path,
            filename=filename,
            size_bytes=file_size,
            upload_timestamp=upload_timestamp,
            metadata=metadata
        )
    
    def _extract_gcs_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Extract metadata from Google Cloud Storage event notification
        
        Args:
            event_payload: GCS event notification payload
            
        Returns:
            FileReference object with extracted metadata
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # GCS event structure
        bucket_name = event_payload['bucket']
        object_name = event_payload['name']
        file_size = int(event_payload['size'])
        
        # Extract filename from object name
        filename = object_name.split('/')[-1]
        
        # Construct full GCS path
        file_path = f"gs://{bucket_name}/{object_name}"
        
        # Parse timestamp
        time_created = event_payload['timeCreated']
        upload_timestamp = datetime.fromisoformat(time_created.replace('Z', '+00:00'))
        
        # Extract additional metadata
        metadata = {
            'bucket': bucket_name,
            'object_name': object_name,
            'content_type': event_payload.get('contentType', 'Unknown'),
            'generation': event_payload.get('generation', 'Unknown')
        }
        
        return FileReference(
            file_path=file_path,
            filename=filename,
            size_bytes=file_size,
            upload_timestamp=upload_timestamp,
            metadata=metadata
        )
    
    def _extract_azure_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Extract metadata from Azure Blob Storage event notification
        
        Args:
            event_payload: Azure event notification payload
            
        Returns:
            FileReference object with extracted metadata
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Azure event structure
        data = event_payload['data']
        url = data['url']
        content_length = int(data['contentLength'])
        
        # Extract container and blob name from URL
        # URL format: https://{account}.blob.core.windows.net/{container}/{blob}
        url_parts = url.split('/')
        container = url_parts[-2]
        blob_name = url_parts[-1]
        filename = blob_name.split('/')[-1]
        
        # Construct Azure blob path
        file_path = f"azure://{container}/{blob_name}"
        
        # Parse timestamp
        event_time = event_payload['eventTime']
        upload_timestamp = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
        
        # Extract additional metadata
        metadata = {
            'container': container,
            'blob_name': blob_name,
            'url': url,
            'content_type': data.get('contentType', 'Unknown'),
            'event_type': event_payload.get('eventType', 'Unknown')
        }
        
        return FileReference(
            file_path=file_path,
            filename=filename,
            size_bytes=content_length,
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
