"""Ingestion Agent for PDF to Manim pipeline

Processes cloud storage events (S3, GCS, Azure) and local file events,
extracts file metadata, and constructs FileReference objects.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from pydantic import ValidationError

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, BackoffStrategy, RetryPolicy
from src.schemas.file_reference import FileReference


logger = logging.getLogger(__name__)

_SUPPORTED_SOURCES = {'s3', 'gcs', 'azure', 'local'}


class CloudStorageEvent(AgentInput):
    """Input for Ingestion Agent.

    Attributes:
        event_payload: Event payload dict (S3, GCS, Azure, or local format)
        event_source: One of 's3', 'gcs', 'azure', 'local'
    """

    def __init__(self, event_payload: Dict[str, Any], event_source: str = 'local'):
        self.event_payload = event_payload
        self.event_source = event_source


class IngestionOutput(AgentOutput):
    """Successful output: extracted FileReference."""

    def __init__(self, file_reference: FileReference):
        self.file_reference = file_reference


class IngestionError(AgentOutput):
    """Error output from Ingestion Agent."""

    def __init__(self, error_code: str, reason: str, event_payload_sample: Optional[str] = None):
        self.error_code = error_code
        self.reason = reason
        self.event_payload_sample = event_payload_sample


class IngestAgent(Agent):
    """Ingestion Agent — parses cloud storage events into FileReference objects.

    Supported event sources: s3, gcs, azure, local.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.IngestAgent")

    @classmethod
    def from_file_path(cls, file_path: str) -> 'CloudStorageEvent':
        """Convenience factory: wrap a local file path as a CloudStorageEvent."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        event_payload = {
            'Records': [
                {
                    's3': {
                        'object': {
                            'key': str(path.resolve()),
                            'size': path.stat().st_size,
                        }
                    },
                    'eventTime': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    'eventName': 'ObjectCreated:Put',
                }
            ]
        }
        return CloudStorageEvent(event_payload=event_payload, event_source='local')

    def execute(self, input_data: CloudStorageEvent) -> AgentOutput:
        if not self.validate_input(input_data):
            return IngestionError(
                error_code="INVALID_INPUT",
                reason="Input is not a valid CloudStorageEvent",
            )

        self.logger.info(
            "Ingestion started",
            extra={
                "event": "ingestion_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_source": input_data.event_source,
            },
        )

        source = input_data.event_source

        if source not in _SUPPORTED_SOURCES:
            return IngestionError(
                error_code="UNSUPPORTED_EVENT_SOURCE",
                reason=(
                    f"Event source '{source}' is not supported. "
                    f"Supported sources: {sorted(_SUPPORTED_SOURCES)}"
                ),
                event_payload_sample=json.dumps(input_data.event_payload)[:200],
            )

        try:
            if source == 's3':
                file_reference = self._extract_s3_metadata(input_data.event_payload)
            elif source == 'gcs':
                file_reference = self._extract_gcs_metadata(input_data.event_payload)
            elif source == 'azure':
                file_reference = self._extract_azure_metadata(input_data.event_payload)
            else:
                file_reference = self._extract_local_metadata(input_data.event_payload)

            self.logger.info(
                "Ingestion completed successfully",
                extra={
                    "event": "ingestion_complete",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "file_path": file_reference.file_path,
                    "file_name": file_reference.filename,
                    "size_bytes": file_reference.size_bytes,
                },
            )
            return IngestionOutput(file_reference=file_reference)

        except KeyError as e:
            error_msg = f"Malformed event payload: missing required field '{e.args[0]}'"
            self.logger.error(error_msg)
            return IngestionError(
                error_code="MALFORMED_PAYLOAD",
                reason=error_msg,
                event_payload_sample=json.dumps(input_data.event_payload)[:200],
            )

        except ValidationError as e:
            error_msg = f"Invalid data in event payload: validation error - {e}"
            self.logger.error(error_msg)
            return IngestionError(
                error_code="INVALID_PAYLOAD_DATA",
                reason=error_msg,
                event_payload_sample=json.dumps(input_data.event_payload)[:200],
            )

        except ValueError as e:
            error_msg = f"Invalid data in event payload: {e}"
            self.logger.error(error_msg)
            return IngestionError(
                error_code="INVALID_PAYLOAD_DATA",
                reason=error_msg,
                event_payload_sample=json.dumps(input_data.event_payload)[:200],
            )

        except Exception as e:
            error_msg = f"Unexpected error during ingestion: {e}"
            self.logger.error(error_msg)
            raise AgentExecutionError(
                error_code="UNEXPECTED_ERROR",
                message=error_msg,
                context={"error_type": type(e).__name__},
            )

    # ------------------------------------------------------------------
    # Per-source parsers
    # ------------------------------------------------------------------

    def _extract_s3_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Parse an AWS S3 event notification."""
        record = event_payload['Records'][0]
        s3_info = record['s3']
        bucket_name = s3_info['bucket']['name']
        s3_object = s3_info['object']
        object_key = s3_object['key']
        file_size = int(s3_object['size'])

        filename = Path(object_key).name
        file_path = f"s3://{bucket_name}/{object_key}"

        event_time_str = record.get('eventTime', '')
        upload_timestamp = self._parse_timestamp(event_time_str)

        region = record.get('awsRegion', 'Unknown')
        event_name = record.get('eventName', 'Unknown')

        metadata: Dict[str, Any] = {
            'bucket': bucket_name,
            'object_key': object_key,
            'event_name': event_name,
            'region': region,
        }
        etag = s3_object.get('eTag')
        if etag is not None:
            metadata['etag'] = etag

        return FileReference(
            file_path=file_path,
            filename=filename,
            size_bytes=file_size,
            upload_timestamp=upload_timestamp,
            metadata=metadata,
        )

    def _extract_gcs_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Parse a Google Cloud Storage object notification."""
        bucket_name = event_payload['bucket']
        object_name = event_payload['name']
        file_size = int(event_payload['size'])

        filename = Path(object_name).name
        file_path = f"gs://{bucket_name}/{object_name}"

        event_time_str = event_payload.get('timeCreated', '')
        upload_timestamp = self._parse_timestamp(event_time_str)

        metadata: Dict[str, Any] = {
            'bucket': bucket_name,
            'object_name': object_name,
        }
        content_type = event_payload.get('contentType')
        if content_type is not None:
            metadata['content_type'] = content_type
        generation = event_payload.get('generation')
        if generation is not None:
            metadata['generation'] = generation

        return FileReference(
            file_path=file_path,
            filename=filename,
            size_bytes=file_size,
            upload_timestamp=upload_timestamp,
            metadata=metadata,
        )

    def _extract_azure_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Parse an Azure Blob Storage event."""
        data = event_payload['data']
        blob_url = data['url']
        file_size = int(data['contentLength'])

        parsed = urlparse(blob_url)
        # path is e.g. "/container/path/to/blob.pdf"
        path_parts = parsed.path.lstrip('/').split('/', 1)
        container = path_parts[0]
        blob_path = path_parts[1] if len(path_parts) > 1 else ''
        blob_name = Path(blob_path).name
        file_path = f"azure://{container}/{blob_path}"

        event_time_str = event_payload.get('eventTime', '')
        upload_timestamp = self._parse_timestamp(event_time_str)

        metadata: Dict[str, Any] = {
            'container': container,
            'blob_name': blob_name,
            'event_type': event_payload.get('eventType', ''),
        }
        content_type = data.get('contentType')
        if content_type is not None:
            metadata['content_type'] = content_type

        return FileReference(
            file_path=file_path,
            filename=blob_name,
            size_bytes=file_size,
            upload_timestamp=upload_timestamp,
            metadata=metadata,
        )

    def _extract_local_metadata(self, event_payload: Dict[str, Any]) -> FileReference:
        """Parse a synthetic local-file event (S3-shaped, local path as key)."""
        record = event_payload['Records'][0]
        s3_info = record['s3']
        file_path = s3_info['object']['key']
        file_size = s3_info['object']['size']
        filename = Path(file_path).name

        event_time_str = record.get('eventTime', '')
        upload_timestamp = self._parse_timestamp(event_time_str)

        metadata = {
            'source': 'local',
            'event_name': record.get('eventName', 'LocalFile'),
        }

        return FileReference(
            file_path=file_path,
            filename=filename,
            size_bytes=file_size,
            upload_timestamp=upload_timestamp,
            metadata=metadata,
        )

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse an ISO-8601 timestamp string, handling trailing Z."""
        if not ts:
            return datetime.now(timezone.utc)
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))

    def validate_input(self, input_data: AgentInput) -> bool:
        if not isinstance(input_data, CloudStorageEvent):
            return False
        if not hasattr(input_data, 'event_payload') or not isinstance(input_data.event_payload, dict):
            return False
        if not hasattr(input_data, 'event_source') or not isinstance(input_data.event_source, str):
            return False
        return True

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=3,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            retryable_errors=[
                "EVENT_SYSTEM_UNAVAILABLE",
                "NETWORK_TIMEOUT",
                "TEMPORARY_SERVICE_UNAVAILABLE",
            ],
        )
