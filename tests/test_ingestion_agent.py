"""Unit tests for Ingestion Agent

Tests cover:
- Event payload parsing with valid S3, GCS, and Azure events
- Metadata extraction accuracy
- FileReference object construction
- Error handling for malformed payloads
"""

import json
from datetime import datetime, timezone

import pytest

from src.agents.ingestion import (
    CloudStorageEvent,
    IngestAgent,
    IngestionError,
    IngestionOutput,
)
from src.schemas.file_reference import FileReference


class TestIngestAgent:
    """Test suite for IngestAgent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = IngestAgent()
    
    # Test S3 event parsing
    
    def test_s3_event_parsing_success(self):
        """Test successful parsing of valid S3 event notification"""
        # Arrange: Create a valid S3 event payload
        s3_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    'eventName': 'ObjectCreated:Put',
                    'awsRegion': 'us-east-1',
                    's3': {
                        'bucket': {
                            'name': 'my-pdf-bucket'
                        },
                        'object': {
                            'key': 'uploads/academic_paper.pdf',
                            'size': 1048576,
                            'eTag': 'abc123def456'
                        }
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=s3_event, event_source='s3')
        
        # Act: Execute the agent
        result = self.agent.execute(input_data)
        
        # Assert: Verify successful output
        assert isinstance(result, IngestionOutput)
        assert isinstance(result.file_reference, FileReference)
        
        # Verify extracted metadata
        file_ref = result.file_reference
        assert file_ref.file_path == 's3://my-pdf-bucket/uploads/academic_paper.pdf'
        assert file_ref.filename == 'academic_paper.pdf'
        assert file_ref.size_bytes == 1048576
        assert file_ref.upload_timestamp == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        
        # Verify additional metadata
        assert file_ref.metadata['bucket'] == 'my-pdf-bucket'
        assert file_ref.metadata['object_key'] == 'uploads/academic_paper.pdf'
        assert file_ref.metadata['event_name'] == 'ObjectCreated:Put'
        assert file_ref.metadata['region'] == 'us-east-1'
        assert file_ref.metadata['etag'] == 'abc123def456'
    
    def test_s3_event_with_nested_path(self):
        """Test S3 event with deeply nested object key"""
        # Arrange: S3 event with nested path
        s3_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    'eventName': 'ObjectCreated:Put',
                    's3': {
                        'bucket': {
                            'name': 'my-bucket'
                        },
                        'object': {
                            'key': 'users/john/documents/2024/paper.pdf',
                            'size': 2097152
                        }
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=s3_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: Filename should be extracted correctly
        assert isinstance(result, IngestionOutput)
        assert result.file_reference.filename == 'paper.pdf'
        assert result.file_reference.file_path == 's3://my-bucket/users/john/documents/2024/paper.pdf'
    
    def test_s3_event_without_optional_fields(self):
        """Test S3 event missing optional fields (etag, region)"""
        # Arrange: Minimal S3 event
        s3_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    's3': {
                        'bucket': {
                            'name': 'my-bucket'
                        },
                        'object': {
                            'key': 'document.pdf',
                            'size': 500000
                        }
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=s3_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: Should still succeed
        assert isinstance(result, IngestionOutput)
        assert result.file_reference.filename == 'document.pdf'
        assert 'etag' not in result.file_reference.metadata
        assert result.file_reference.metadata['region'] == 'Unknown'
        assert result.file_reference.metadata['event_name'] == 'Unknown'
    
    # Test GCS event parsing
    
    def test_gcs_event_parsing_success(self):
        """Test successful parsing of valid GCS event notification"""
        # Arrange: Create a valid GCS event payload
        gcs_event = {
            'bucket': 'my-gcs-bucket',
            'name': 'uploads/research_paper.pdf',
            'size': '2097152',
            'timeCreated': '2024-01-15T10:30:00.000Z',
            'contentType': 'application/pdf',
            'generation': '1234567890'
        }
        
        input_data = CloudStorageEvent(event_payload=gcs_event, event_source='gcs')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert
        assert isinstance(result, IngestionOutput)
        file_ref = result.file_reference
        assert file_ref.file_path == 'gs://my-gcs-bucket/uploads/research_paper.pdf'
        assert file_ref.filename == 'research_paper.pdf'
        assert file_ref.size_bytes == 2097152
        assert file_ref.upload_timestamp == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        
        # Verify GCS-specific metadata
        assert file_ref.metadata['bucket'] == 'my-gcs-bucket'
        assert file_ref.metadata['object_name'] == 'uploads/research_paper.pdf'
        assert file_ref.metadata['content_type'] == 'application/pdf'
        assert file_ref.metadata['generation'] == '1234567890'
    
    # Test Azure event parsing
    
    def test_azure_event_parsing_success(self):
        """Test successful parsing of valid Azure Blob Storage event"""
        # Arrange: Create a valid Azure event payload
        azure_event = {
            'eventTime': '2024-01-15T10:30:00.000Z',
            'eventType': 'Microsoft.Storage.BlobCreated',
            'data': {
                'url': 'https://mystorageaccount.blob.core.windows.net/pdfs/thesis.pdf',
                'contentLength': 3145728,
                'contentType': 'application/pdf'
            }
        }
        
        input_data = CloudStorageEvent(event_payload=azure_event, event_source='azure')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert
        assert isinstance(result, IngestionOutput)
        file_ref = result.file_reference
        assert file_ref.file_path == 'azure://pdfs/thesis.pdf'
        assert file_ref.filename == 'thesis.pdf'
        assert file_ref.size_bytes == 3145728
        assert file_ref.upload_timestamp == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        
        # Verify Azure-specific metadata
        assert file_ref.metadata['container'] == 'pdfs'
        assert file_ref.metadata['blob_name'] == 'thesis.pdf'
        assert file_ref.metadata['content_type'] == 'application/pdf'
        assert file_ref.metadata['event_type'] == 'Microsoft.Storage.BlobCreated'
    
    # Test metadata extraction accuracy
    
    def test_metadata_extraction_preserves_all_fields(self):
        """Test that all available metadata fields are extracted"""
        # Arrange: S3 event with all optional fields
        s3_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    'eventName': 'ObjectCreated:CompleteMultipartUpload',
                    'awsRegion': 'eu-west-1',
                    's3': {
                        'bucket': {
                            'name': 'test-bucket'
                        },
                        'object': {
                            'key': 'test.pdf',
                            'size': 1000,
                            'eTag': 'test-etag-123'
                        }
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=s3_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: All metadata fields should be present
        metadata = result.file_reference.metadata
        assert 'bucket' in metadata
        assert 'object_key' in metadata
        assert 'event_name' in metadata
        assert 'region' in metadata
        assert 'etag' in metadata
        assert metadata['event_name'] == 'ObjectCreated:CompleteMultipartUpload'
        assert metadata['region'] == 'eu-west-1'
    
    def test_timestamp_parsing_accuracy(self):
        """Test that timestamps are parsed correctly from different formats"""
        # Arrange: Test with different timestamp formats
        s3_event = {
            'Records': [
                {
                    'eventTime': '2024-03-20T15:45:30.123Z',
                    's3': {
                        'bucket': {'name': 'bucket'},
                        'object': {'key': 'file.pdf', 'size': 1000}
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=s3_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: Timestamp should be parsed correctly
        expected_time = datetime(2024, 3, 20, 15, 45, 30, 123000, tzinfo=timezone.utc)
        assert result.file_reference.upload_timestamp == expected_time
    
    # Test FileReference object construction
    
    def test_file_reference_construction_with_valid_data(self):
        """Test that FileReference objects are constructed correctly"""
        # Arrange
        s3_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    's3': {
                        'bucket': {'name': 'bucket'},
                        'object': {'key': 'path/to/file.pdf', 'size': 12345}
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=s3_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: FileReference should have all required fields
        file_ref = result.file_reference
        assert hasattr(file_ref, 'file_path')
        assert hasattr(file_ref, 'filename')
        assert hasattr(file_ref, 'size_bytes')
        assert hasattr(file_ref, 'upload_timestamp')
        assert hasattr(file_ref, 'metadata')
        
        # Verify types
        assert isinstance(file_ref.file_path, str)
        assert isinstance(file_ref.filename, str)
        assert isinstance(file_ref.size_bytes, int)
        assert isinstance(file_ref.upload_timestamp, datetime)
        assert isinstance(file_ref.metadata, dict)
    
    # Test error handling for malformed payloads
    
    def test_malformed_s3_payload_missing_records(self):
        """Test error handling when S3 event is missing Records field"""
        # Arrange: Malformed S3 event
        malformed_event = {
            'NotRecords': []  # Wrong field name
        }
        
        input_data = CloudStorageEvent(event_payload=malformed_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: Should return IngestionError
        assert isinstance(result, IngestionError)
        assert result.error_code == 'MALFORMED_PAYLOAD'
        assert 'Records' in result.reason
        assert result.event_payload_sample is not None
    
    def test_malformed_s3_payload_missing_bucket(self):
        """Test error handling when S3 event is missing bucket information"""
        # Arrange
        malformed_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    's3': {
                        'object': {'key': 'file.pdf', 'size': 1000}
                        # Missing 'bucket' field
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=malformed_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert
        assert isinstance(result, IngestionError)
        assert result.error_code == 'MALFORMED_PAYLOAD'
        assert 'bucket' in result.reason.lower()
    
    def test_malformed_s3_payload_missing_object_key(self):
        """Test error handling when S3 event is missing object key"""
        # Arrange
        malformed_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    's3': {
                        'bucket': {'name': 'bucket'},
                        'object': {'size': 1000}
                        # Missing 'key' field
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=malformed_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert
        assert isinstance(result, IngestionError)
        assert result.error_code == 'MALFORMED_PAYLOAD'
        assert 'key' in result.reason.lower()
    
    def test_malformed_gcs_payload_missing_bucket(self):
        """Test error handling for malformed GCS event"""
        # Arrange
        malformed_event = {
            'name': 'file.pdf',
            'size': '1000',
            'timeCreated': '2024-01-15T10:30:00.000Z'
            # Missing 'bucket' field
        }
        
        input_data = CloudStorageEvent(event_payload=malformed_event, event_source='gcs')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert
        assert isinstance(result, IngestionError)
        assert result.error_code == 'MALFORMED_PAYLOAD'
    
    def test_malformed_azure_payload_missing_data(self):
        """Test error handling for malformed Azure event"""
        # Arrange
        malformed_event = {
            'eventTime': '2024-01-15T10:30:00.000Z',
            'eventType': 'Microsoft.Storage.BlobCreated'
            # Missing 'data' field
        }
        
        input_data = CloudStorageEvent(event_payload=malformed_event, event_source='azure')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert
        assert isinstance(result, IngestionError)
        assert result.error_code == 'MALFORMED_PAYLOAD'
    
    def test_invalid_payload_data_negative_size(self):
        """Test error handling when file size is negative"""
        # Arrange: Event with negative file size
        invalid_event = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    's3': {
                        'bucket': {'name': 'bucket'},
                        'object': {'key': 'file.pdf', 'size': -1000}
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=invalid_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: Pydantic validation should catch negative size
        # The FileReference schema has ge=0 constraint, so this will fail at schema level
        assert isinstance(result, IngestionError)
        assert result.error_code == 'INVALID_PAYLOAD_DATA'
        assert 'validation error' in result.reason.lower()
    
    def test_unsupported_event_source(self):
        """Test error handling for unsupported event source"""
        # Arrange
        event = {'some': 'data'}
        input_data = CloudStorageEvent(event_payload=event, event_source='dropbox')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert
        assert isinstance(result, IngestionError)
        assert result.error_code == 'UNSUPPORTED_EVENT_SOURCE'
        assert 'dropbox' in result.reason
    
    def test_invalid_input_type(self):
        """Test error handling when input is not CloudStorageEvent"""
        # Arrange: Invalid input type
        invalid_input = "not a CloudStorageEvent"
        
        # Act
        result = self.agent.execute(invalid_input)
        
        # Assert
        assert isinstance(result, IngestionError)
        assert result.error_code == 'INVALID_INPUT'
    
    def test_event_payload_sample_truncation(self):
        """Test that error payload samples are truncated to reasonable length"""
        # Arrange: Create a very large payload
        large_payload = {
            'Records': [
                {
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    's3': {
                        'bucket': {'name': 'bucket'},
                        'object': {
                            'key': 'file.pdf',
                            'size': 1000,
                            'metadata': {'large_field': 'x' * 10000}  # Very large field
                        }
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=large_payload, event_source='unsupported')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: Payload sample should be truncated
        assert isinstance(result, IngestionError)
        assert len(result.event_payload_sample) <= 200
    
    # Test input validation
    
    def test_validate_input_with_valid_cloud_storage_event(self):
        """Test input validation with valid CloudStorageEvent"""
        # Arrange
        valid_input = CloudStorageEvent(event_payload={'test': 'data'}, event_source='s3')
        
        # Act
        is_valid = self.agent.validate_input(valid_input)
        
        # Assert
        assert is_valid is True
    
    def test_validate_input_with_invalid_type(self):
        """Test input validation with invalid input type"""
        # Arrange
        invalid_inputs = [
            None,
            "string",
            123,
            {'dict': 'object'},
            []
        ]
        
        # Act & Assert
        for invalid_input in invalid_inputs:
            is_valid = self.agent.validate_input(invalid_input)
            assert is_valid is False
    
    def test_validate_input_missing_event_payload(self):
        """Test input validation when event_payload is missing"""
        # Arrange: Create object without event_payload
        class InvalidEvent:
            def __init__(self):
                self.event_source = 's3'
        
        invalid_input = InvalidEvent()
        
        # Act
        is_valid = self.agent.validate_input(invalid_input)
        
        # Assert
        assert is_valid is False
    
    def test_validate_input_invalid_event_payload_type(self):
        """Test input validation when event_payload is not a dict"""
        # Arrange
        class InvalidEvent:
            def __init__(self):
                self.event_payload = "not a dict"
                self.event_source = 's3'
        
        invalid_input = InvalidEvent()
        
        # Act
        is_valid = self.agent.validate_input(invalid_input)
        
        # Assert
        assert is_valid is False
    
    # Test retry policy
    
    def test_get_retry_policy(self):
        """Test that retry policy is configured correctly"""
        # Act
        policy = self.agent.get_retry_policy()
        
        # Assert
        assert policy.max_attempts == 3
        assert policy.backoff_strategy.value == 'exponential'
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 30.0
        assert len(policy.retryable_errors) > 0
        assert 'EVENT_SYSTEM_UNAVAILABLE' in policy.retryable_errors
        assert 'NETWORK_TIMEOUT' in policy.retryable_errors
    
    def test_retry_policy_does_not_include_malformed_payload(self):
        """Test that malformed payload errors are not retryable"""
        # Act
        policy = self.agent.get_retry_policy()
        
        # Assert: Malformed payloads should not be retried
        assert 'MALFORMED_PAYLOAD' not in policy.retryable_errors
        assert 'INVALID_PAYLOAD_DATA' not in policy.retryable_errors
    
    # Integration-style tests
    
    def test_end_to_end_s3_ingestion(self):
        """Test complete S3 ingestion flow from event to FileReference"""
        # Arrange: Realistic S3 event
        s3_event = {
            'Records': [
                {
                    'eventVersion': '2.1',
                    'eventSource': 'aws:s3',
                    'eventTime': '2024-01-15T10:30:00.000Z',
                    'eventName': 'ObjectCreated:Put',
                    'awsRegion': 'us-west-2',
                    's3': {
                        'bucket': {
                            'name': 'academic-papers-bucket',
                            'arn': 'arn:aws:s3:::academic-papers-bucket'
                        },
                        'object': {
                            'key': 'uploads/2024/quantum_mechanics.pdf',
                            'size': 5242880,
                            'eTag': 'abc123def456ghi789'
                        }
                    }
                }
            ]
        }
        
        input_data = CloudStorageEvent(event_payload=s3_event, event_source='s3')
        
        # Act
        result = self.agent.execute(input_data)
        
        # Assert: Complete validation
        assert isinstance(result, IngestionOutput)
        file_ref = result.file_reference
        
        # Verify all fields are correctly populated
        assert file_ref.file_path == 's3://academic-papers-bucket/uploads/2024/quantum_mechanics.pdf'
        assert file_ref.filename == 'quantum_mechanics.pdf'
        assert file_ref.size_bytes == 5242880
        assert isinstance(file_ref.upload_timestamp, datetime)
        assert file_ref.metadata['bucket'] == 'academic-papers-bucket'
        assert file_ref.metadata['event_name'] == 'ObjectCreated:Put'
        assert file_ref.metadata['region'] == 'us-west-2'
        
        # Verify FileReference can be serialized (important for passing to next agent)
        file_ref_dict = file_ref.model_dump()
        assert 'file_path' in file_ref_dict
        assert 'filename' in file_ref_dict
        assert 'size_bytes' in file_ref_dict
        assert 'upload_timestamp' in file_ref_dict
        assert 'metadata' in file_ref_dict
