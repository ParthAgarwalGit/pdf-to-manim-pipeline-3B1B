"""Unit tests for Pipeline Orchestrator.

Tests the orchestrator's ability to:
- Sequence agents in linear pipeline
- Pass outputs from one agent as inputs to the next
- Implement abort logic for validation and sanitization errors
- Implement retry logic based on agent retry policies
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.agents.base import AgentExecutionError, BackoffStrategy, RetryPolicy
from src.agents.ingestion import IngestionError, IngestionOutput
from src.agents.json_sanitizer import SanitizationError, ValidationErrorDetail
from src.agents.validation import ValidationError
from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig
from src.schemas.file_reference import FileReference
from src.schemas.output_manifest import OutputManifest


class TestOrchestratorInitialization:
    """Test orchestrator initialization and configuration."""
    
    def test_orchestrator_initializes_with_default_config(self):
        """Test orchestrator initializes with default configuration."""
        orchestrator = Orchestrator()
        
        assert orchestrator.config is not None
        assert orchestrator.config.verbosity == "medium"
        assert orchestrator.config.depth == "intermediate"
        assert orchestrator.config.audience == "undergraduate"
        assert orchestrator.config.base_output_dir == "output"
    
    def test_orchestrator_initializes_with_custom_config(self):
        """Test orchestrator initializes with custom configuration."""
        config = PipelineConfig(
            verbosity="high",
            depth="advanced",
            audience="graduate",
            base_output_dir="custom_output"
        )
        orchestrator = Orchestrator(config=config)
        
        assert orchestrator.config.verbosity == "high"
        assert orchestrator.config.depth == "advanced"
        assert orchestrator.config.audience == "graduate"
        assert orchestrator.config.base_output_dir == "custom_output"
    
    def test_orchestrator_initializes_all_agents(self):
        """Test orchestrator initializes all required agents."""
        orchestrator = Orchestrator()
        
        assert orchestrator.ingestion_agent is not None
        assert orchestrator.validation_agent is not None
        assert orchestrator.vision_ocr_agent is not None
        assert orchestrator.script_architect_agent is not None
        assert orchestrator.json_sanitizer_agent is not None
        assert orchestrator.scene_controller_agent is not None
        assert orchestrator.persistence_agent is not None


class TestPipelineAbortLogic:
    """Test pipeline abort logic for validation and sanitization errors."""
    
    def test_pipeline_aborts_on_validation_error(self):
        """Test pipeline aborts when validation fails."""
        orchestrator = Orchestrator()
        
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test content')
            temp_path = f.name
        
        try:
            # Mock validation agent to return ValidationError
            validation_error = ValidationError(
                error_code="INVALID_MIME_TYPE",
                reason="File is not a PDF",
                file_path=temp_path
            )
            orchestrator.validation_agent.execute = Mock(return_value=validation_error)
            
            # Mock ingestion to succeed
            file_ref = FileReference(
                file_path=temp_path,
                filename="file.txt",
                size_bytes=1000,
                upload_timestamp=datetime.now(timezone.utc),
                metadata={}
            )
            orchestrator.ingestion_agent.execute = Mock(
                return_value=IngestionOutput(file_reference=file_ref)
            )
            
            # Pipeline should abort with PipelineAbortError
            with pytest.raises(PipelineAbortError) as exc_info:
                orchestrator.run_pipeline(temp_path, event_source="local")
            
            assert exc_info.value.stage == "Validation"
            assert exc_info.value.error_code == "INVALID_MIME_TYPE"
            assert "not a PDF" in exc_info.value.message
        finally:
            # Cleanup
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_pipeline_aborts_on_sanitization_error(self):
        """Test pipeline aborts when JSON sanitization fails."""
        orchestrator = Orchestrator()
        
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write('%PDF-1.4\ntest\n%%EOF\n')
            temp_path = f.name
        
        try:
            # Mock all agents up to sanitizer to succeed
            self._mock_agents_until_sanitizer(orchestrator, temp_path)
            
            # Mock sanitizer to return SanitizationError
            sanitization_error = SanitizationError(
                errors=[
                    ValidationErrorDetail(
                        field_path="scenes[0].scene_id",
                        violation_type="missing_required_field",
                        expected="string",
                        actual="None",
                        message="Field required"
                    )
                ],
                attempted_corrections=[]
            )
            orchestrator.json_sanitizer_agent.execute = Mock(return_value=sanitization_error)
            
            # Pipeline should abort with PipelineAbortError
            with pytest.raises(PipelineAbortError) as exc_info:
                orchestrator.run_pipeline(temp_path, event_source="local")
            
            assert exc_info.value.stage == "JSONSanitizer"
            assert exc_info.value.error_code == "SCHEMA_VIOLATION"
        finally:
            # Cleanup
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_pipeline_aborts_on_ingestion_error(self):
        """Test pipeline aborts when ingestion fails."""
        orchestrator = Orchestrator()
        
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write('%PDF-1.4\ntest\n%%EOF\n')
            temp_path = f.name
        
        try:
            # Mock ingestion agent to return IngestionError
            ingestion_error = IngestionError(
                error_code="MALFORMED_PAYLOAD",
                reason="Missing required field 'Records'",
                event_payload_sample="{}"
            )
            orchestrator.ingestion_agent.execute = Mock(return_value=ingestion_error)
            
            # Pipeline should abort with PipelineAbortError
            with pytest.raises(PipelineAbortError) as exc_info:
                orchestrator.run_pipeline(temp_path, event_source="local")
            
            assert exc_info.value.stage == "Ingestion"
            assert exc_info.value.error_code == "MALFORMED_PAYLOAD"
        finally:
            # Cleanup
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _mock_agents_until_sanitizer(self, orchestrator, temp_path):
        """Helper to mock all agents up to sanitizer to succeed."""
        from src.schemas.ocr_output import OCROutput, Page, PDFMetadata, TextBlock, BoundingBox
        from src.schemas.storyboard import (
            Configuration,
            PDFMetadataStoryboard,
            Scene,
            StoryboardJSON,
            VisualIntent,
        )
        
        # Mock ingestion
        file_ref = FileReference(
            file_path=temp_path,
            filename="test.pdf",
            size_bytes=100000,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        
        # Mock validation
        orchestrator.validation_agent.execute = Mock(return_value=file_ref)
        
        # Mock vision OCR - need at least one page
        ocr_output = OCROutput(
            pdf_metadata=PDFMetadata(filename="test.pdf", page_count=1, file_size_bytes=100000),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[
                        TextBlock(
                            text="Sample text",
                            bounding_box=BoundingBox(x=0, y=0, width=100, height=20),
                            reading_order=0,
                            confidence=0.95
                        )
                    ],
                    math_expressions=[],
                    diagrams=[]
                )
            ]
        )
        orchestrator.vision_ocr_agent.execute = Mock(return_value=ocr_output)
        
        # Mock script architect
        storyboard = StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="test.pdf", page_count=1),
            configuration=Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
            concept_hierarchy=[],
            scenes=[
                Scene(
                    scene_id="scene_001",
                    scene_index=0,
                    concept_id="concept_001",  # Must be a string, not None
                    narration="Test narration",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[],
                        emphasis_points=[]
                    ),
                    duration_estimate=30.0,
                    dependencies=[],
                    difficulty_level=None
                )
            ]
        )
        orchestrator.script_architect_agent.execute = Mock(return_value=storyboard)


class TestRetryLogic:
    """Test retry logic based on agent retry policies."""
    
    def test_retry_on_transient_failure(self):
        """Test orchestrator retries on transient failures."""
        orchestrator = Orchestrator()
        
        # Mock agent to fail twice then succeed
        call_count = 0
        
        def mock_execute(input_data):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentExecutionError(
                    error_code="NETWORK_TIMEOUT",
                    message="Network timeout"
                )
            # Third call succeeds
            from src.schemas.ocr_output import OCROutput, PDFMetadata, Page
            return OCROutput(
                pdf_metadata=PDFMetadata(filename="test.pdf", page_count=1, file_size_bytes=100000),
                pages=[Page(page_number=1, text_blocks=[], math_expressions=[], diagrams=[])]
            )
        
        # Mock vision OCR agent with retry policy
        orchestrator.vision_ocr_agent.execute = Mock(side_effect=mock_execute)
        orchestrator.vision_ocr_agent.get_retry_policy = Mock(
            return_value=RetryPolicy(
                max_attempts=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                base_delay_seconds=0.01,  # Short delay for testing
                max_delay_seconds=0.1,
                retryable_errors=["NETWORK_TIMEOUT"]
            )
        )
        
        # Mock other agents to succeed
        self._mock_other_agents(orchestrator)
        
        # Execute pipeline - should succeed after retries
        # Note: This will fail in full pipeline, but we're testing retry logic
        import tempfile
        from src.orchestrator.logger import StructuredJSONLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            structured_logger = StructuredJSONLogger(output_directory=tmpdir)
            
            result = orchestrator._execute_with_retry(
                agent=orchestrator.vision_ocr_agent,
                input_data=Mock(),
                stage_name="Vision+OCR",
                structured_logger=structured_logger
            )
            
            structured_logger.close()
        
        # Verify agent was called 3 times
        assert call_count == 3
    
    def test_no_retry_on_non_retryable_error(self):
        """Test orchestrator does not retry non-retryable errors."""
        orchestrator = Orchestrator()
        
        # Mock agent to fail with non-retryable error
        call_count = 0
        
        def mock_execute(input_data):
            nonlocal call_count
            call_count += 1
            raise AgentExecutionError(
                error_code="INVALID_INPUT",
                message="Invalid input"
            )
        
        orchestrator.vision_ocr_agent.execute = Mock(side_effect=mock_execute)
        orchestrator.vision_ocr_agent.get_retry_policy = Mock(
            return_value=RetryPolicy(
                max_attempts=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                base_delay_seconds=0.01,
                max_delay_seconds=0.1,
                retryable_errors=["NETWORK_TIMEOUT"]  # INVALID_INPUT not in list
            )
        )
        
        # Execute should fail immediately without retry
        import tempfile
        from src.orchestrator.logger import StructuredJSONLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            structured_logger = StructuredJSONLogger(output_directory=tmpdir)
            
            with pytest.raises(AgentExecutionError):
                orchestrator._execute_with_retry(
                    agent=orchestrator.vision_ocr_agent,
                    input_data=Mock(),
                    stage_name="Vision+OCR",
                    structured_logger=structured_logger
                )
            
            structured_logger.close()
        
        # Verify agent was called only once
        assert call_count == 1
    
    def _mock_other_agents(self, orchestrator):
        """Helper to mock other agents to succeed."""
        from src.schemas.ocr_output import OCROutput, PDFMetadata
        from src.schemas.storyboard import (
            Configuration,
            PDFMetadataStoryboard,
            Scene,
            StoryboardJSON,
            VisualIntent,
        )
        
        # Mock ingestion
        file_ref = FileReference(
            file_path="/path/to/test.pdf",
            filename="test.pdf",
            size_bytes=100000,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        
        # Mock validation
        orchestrator.validation_agent.execute = Mock(return_value=file_ref)


class TestBackoffCalculation:
    """Test backoff delay calculation for retries.
    
    Note: These tests now use the retry_policy module directly since
    the orchestrator delegates backoff calculation to that module.
    """
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        from src.orchestrator.retry_policy import calculate_backoff_delay
        
        # Test exponential backoff: base * 2^attempt
        delay0 = calculate_backoff_delay(0, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        delay1 = calculate_backoff_delay(1, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        delay2 = calculate_backoff_delay(2, BackoffStrategy.EXPONENTIAL, 1.0, 60.0)
        
        assert delay0 == 1.0  # 1 * 2^0
        assert delay1 == 2.0  # 1 * 2^1
        assert delay2 == 4.0  # 1 * 2^2
    
    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        from src.orchestrator.retry_policy import calculate_backoff_delay
        
        # Test linear backoff: base * (attempt + 1)
        delay0 = calculate_backoff_delay(0, BackoffStrategy.LINEAR, 1.0, 60.0)
        delay1 = calculate_backoff_delay(1, BackoffStrategy.LINEAR, 1.0, 60.0)
        delay2 = calculate_backoff_delay(2, BackoffStrategy.LINEAR, 1.0, 60.0)
        
        assert delay0 == 1.0  # 1 * 1
        assert delay1 == 2.0  # 1 * 2
        assert delay2 == 3.0  # 1 * 3
    
    def test_constant_backoff(self):
        """Test constant backoff calculation."""
        from src.orchestrator.retry_policy import calculate_backoff_delay
        
        # Test constant backoff: always base
        delay0 = calculate_backoff_delay(0, BackoffStrategy.CONSTANT, 1.0, 60.0)
        delay1 = calculate_backoff_delay(1, BackoffStrategy.CONSTANT, 1.0, 60.0)
        delay2 = calculate_backoff_delay(2, BackoffStrategy.CONSTANT, 1.0, 60.0)
        
        assert delay0 == 1.0
        assert delay1 == 1.0
        assert delay2 == 1.0
    
    def test_backoff_respects_max_delay(self):
        """Test backoff delay is capped at max_delay."""
        from src.orchestrator.retry_policy import calculate_backoff_delay
        
        # Exponential backoff with low max_delay
        delay = calculate_backoff_delay(10, BackoffStrategy.EXPONENTIAL, 1.0, 5.0)
        
        # 1 * 2^10 = 1024, but should be capped at 5.0
        assert delay == 5.0


class TestLocalFileProcessing:
    """Test processing of local PDF files."""
    
    def test_create_local_file_event(self):
        """Test creation of synthetic event for local file."""
        orchestrator = Orchestrator()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write('%PDF-1.4\n')
            f.write('test content\n')
            f.write('%%EOF\n')
            temp_path = f.name
        
        try:
            # Create event
            event = orchestrator._create_local_file_event(temp_path)
            
            # Verify event structure
            assert 'Records' in event
            assert len(event['Records']) == 1
            
            record = event['Records'][0]
            assert 'eventTime' in record
            assert 's3' in record
            assert record['s3']['object']['key'] == temp_path
            assert record['s3']['object']['size'] > 0
        finally:
            # Cleanup
            os.unlink(temp_path)


class TestPipelineConfig:
    """Test pipeline configuration."""
    
    def test_config_to_storyboard_config(self):
        """Test conversion of PipelineConfig to Configuration schema."""
        config = PipelineConfig(
            verbosity="high",
            depth="advanced",
            audience="graduate"
        )
        
        storyboard_config = config.to_storyboard_config()
        
        assert storyboard_config.verbosity == "high"
        assert storyboard_config.depth == "advanced"
        assert storyboard_config.audience == "graduate"
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.verbosity == "medium"
        assert config.depth == "intermediate"
        assert config.audience == "undergraduate"
        assert config.base_output_dir == "output"
        assert config.enable_ocr_debug is False


class TestLogging:
    """Test structured logging functionality."""
    
    def test_summarize_input(self):
        """Test input summarization for logging."""
        orchestrator = Orchestrator()
        
        file_ref = FileReference(
            file_path="/path/to/test.pdf",
            filename="test.pdf",
            size_bytes=100000,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        
        summary = orchestrator._summarize_input(file_ref)
        assert "FileReference" in summary
        assert "test.pdf" in summary
    
    def test_summarize_output(self):
        """Test output summarization for logging."""
        orchestrator = Orchestrator()
        
        file_ref = FileReference(
            file_path="/path/to/test.pdf",
            filename="test.pdf",
            size_bytes=100000,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        
        summary = orchestrator._summarize_output(file_ref)
        assert "FileReference" in summary
        assert "FileReference" in summary
        assert "test.pdf" in summary
        assert "100000" in summary
    
    def test_summarize_output(self):
        """Test output summarization for logging."""
        orchestrator = Orchestrator()
        
        file_ref = FileReference(
            file_path="/path/to/test.pdf",
            filename="test.pdf",
            size_bytes=100000,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        
        summary = orchestrator._summarize_output(file_ref)
        
        assert "FileReference" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
