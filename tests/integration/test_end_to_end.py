"""Integration tests for Pipeline Orchestrator.

Tests end-to-end pipeline execution including:
- Complete pipeline with sample PDF
- Abort on validation failure
- Abort on sanitization failure
- Retry logic for transient failures
- Partial success (some scenes fail, pipeline continues)
- CRITICAL_FAILURE status when >50% scenes fail

Requirements tested: 3.4, 6.5, 7.5, 14.4
"""

import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.agents.base import AgentExecutionError, BackoffStrategy, RetryPolicy
from src.agents.ingestion import IngestionOutput
from src.agents.json_sanitizer import SanitizationError, ValidationErrorDetail
from src.agents.validation import ValidationError
from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.file_reference import FileReference
from src.schemas.ocr_output import (
    BoundingBox,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)
from src.schemas.output_manifest import FailedSceneManifest, OutputFile, OutputManifest
from src.schemas.scene_processing import FailedScene, SceneProcessingReport
from src.schemas.storyboard import (
    Configuration,
    MathematicalObject,
    PDFMetadataStoryboard,
    Scene,
    StoryboardJSON,
    Transformation,
    VisualIntent,
)


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        # Create a minimal valid PDF
        f.write('%PDF-1.4\n')
        f.write('1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n')
        f.write('2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n')
        f.write('3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n')
        f.write('xref\n0 4\n0000000000 65535 f\n')
        f.write('0000000009 00000 n\n0000000056 00000 n\n0000000115 00000 n\n')
        f.write('trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline execution."""
    
    def test_successful_pipeline_execution(self, temp_pdf_file, temp_output_dir):
        """Test complete pipeline execution with all agents succeeding.
        
        Requirements: 3.4, 6.5, 7.5, 14.4
        """
        # Create orchestrator with custom output directory
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=config)
        
        # Mock all agents to succeed
        self._mock_successful_pipeline(orchestrator, temp_pdf_file)
        
        # Execute pipeline
        manifest = orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify manifest
        assert manifest is not None
        assert manifest.status == "SUCCESS"
        assert manifest.total_scenes == 2
        assert manifest.successful_scenes == 2
        assert len(manifest.failed_scenes) == 0
        assert manifest.scene_success_rate == 1.0
        
        # Verify output directory path is set (directory not actually created in mocked test)
        assert manifest.output_directory is not None
        assert len(manifest.output_directory) > 0
        
        # Verify files were created
        assert len(manifest.files) > 0
        file_types = {f.type for f in manifest.files}
        assert "animation_code" in file_types
        assert "narration_script" in file_types
        assert "storyboard" in file_types
        assert "manifest" in file_types
    
    def _mock_successful_pipeline(self, orchestrator, temp_pdf_file):
        """Helper to mock all agents for successful pipeline execution."""
        # Mock ingestion
        file_ref = FileReference(
            file_path=temp_pdf_file,
            filename="test.pdf",
            size_bytes=500,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        
        # Mock validation
        orchestrator.validation_agent.execute = Mock(return_value=file_ref)
        
        # Mock vision OCR
        ocr_output = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="test.pdf",
                page_count=1,
                file_size_bytes=500
            ),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[
                        TextBlock(
                            text="Introduction to calculus",
                            bounding_box=BoundingBox(x=0, y=0, width=200, height=30),
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
            configuration=Configuration(
                verbosity="medium",
                depth="intermediate",
                audience="undergraduate"
            ),
            concept_hierarchy=[],
            scenes=[
                Scene(
                    scene_id="scene_001",
                    scene_index=0,
                    concept_id="intro",
                    narration="Welcome to calculus",
                    visual_intent=VisualIntent(
                        mathematical_objects=[
                            MathematicalObject(
                                object_type="text",
                                content="Calculus Introduction"
                            )
                        ],
                        transformations=[],
                        emphasis_points=[]
                    ),
                    duration_estimate=30.0
                ),
                Scene(
                    scene_id="scene_002",
                    scene_index=1,
                    concept_id="derivatives",
                    narration="Understanding derivatives",
                    visual_intent=VisualIntent(
                        mathematical_objects=[
                            MathematicalObject(
                                object_type="equation",
                                content=r"\frac{dy}{dx}"
                            )
                        ],
                        transformations=[],
                        emphasis_points=[]
                    ),
                    duration_estimate=45.0
                )
            ]
        )
        orchestrator.script_architect_agent.execute = Mock(return_value=storyboard)
        
        # Mock JSON sanitizer
        orchestrator.json_sanitizer_agent.execute = Mock(
            return_value=Mock(storyboard=storyboard)
        )
        
        # Mock scene controller
        animation_codes = [
            AnimationCodeObject(
                scene_id="scene_001",
                code="from manim import *\n\nclass Scene_001(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Welcome to calculus",
                duration_estimate=30.0,
                generation_timestamp=datetime.now(timezone.utc)
            ),
            AnimationCodeObject(
                scene_id="scene_002",
                code="from manim import *\n\nclass Scene_002(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name="Scene_002",
                narration="Understanding derivatives",
                duration_estimate=45.0,
                generation_timestamp=datetime.now(timezone.utc)
            )
        ]
        
        scene_report = SceneProcessingReport(
            total_scenes=2,
            successful_scenes=2,
            failed_scenes=[],
            output_directory="output/test/timestamp"
        )
        
        orchestrator.scene_controller_agent.execute = Mock(
            return_value=Mock(
                report=scene_report,
                animation_code_objects=animation_codes
            )
        )
        
        # Mock persistence
        manifest = OutputManifest(
            output_directory="output/test/timestamp",
            files=[
                OutputFile(path="scene_001.py", size_bytes=100, type="animation_code"),
                OutputFile(path="scene_002.py", size_bytes=100, type="animation_code"),
                OutputFile(path="narration_script.txt", size_bytes=50, type="narration_script"),
                OutputFile(path="storyboard.json", size_bytes=200, type="storyboard"),
                OutputFile(path="manifest.json", size_bytes=150, type="manifest")
            ],
            timestamp=datetime.now(timezone.utc),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=2,
            successful_scenes=2,
            failed_scenes=[]
        )
        
        orchestrator.persistence_agent.execute = Mock(
            return_value=Mock(manifest=manifest)
        )


class TestPipelineAbortScenarios:
    """Test pipeline abort logic for validation and sanitization failures."""
    
    def test_abort_on_validation_failure(self, temp_pdf_file, temp_output_dir):
        """Test pipeline aborts when validation fails.
        
        Requirements: 3.4
        """
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=config)
        
        # Mock ingestion to succeed
        file_ref = FileReference(
            file_path=temp_pdf_file,
            filename="test.pdf",
            size_bytes=500,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        
        # Mock validation to fail
        validation_error = ValidationError(
            error_code="INVALID_MIME_TYPE",
            reason="File is not a valid PDF",
            file_path=temp_pdf_file
        )
        orchestrator.validation_agent.execute = Mock(return_value=validation_error)
        
        # Pipeline should abort with PipelineAbortError
        with pytest.raises(PipelineAbortError) as exc_info:
            orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify abort details
        assert exc_info.value.stage == "Validation"
        assert exc_info.value.error_code == "INVALID_MIME_TYPE"
        assert "not a valid PDF" in exc_info.value.message
    
    def test_abort_on_sanitization_failure(self, temp_pdf_file, temp_output_dir):
        """Test pipeline aborts when JSON sanitization fails.
        
        Requirements: 6.5
        """
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=config)
        
        # Mock agents up to sanitizer to succeed
        self._mock_agents_until_sanitizer(orchestrator, temp_pdf_file)
        
        # Mock sanitizer to fail
        sanitization_error = SanitizationError(
            errors=[
                ValidationErrorDetail(
                    field_path="scenes[0].scene_id",
                    violation_type="missing_required_field",
                    expected="string",
                    actual="None",
                    message="Field required"
                ),
                ValidationErrorDetail(
                    field_path="scenes[0].narration",
                    violation_type="type_error",
                    expected="string",
                    actual="int",
                    message="Input should be a valid string"
                )
            ],
            attempted_corrections=[]
        )
        orchestrator.json_sanitizer_agent.execute = Mock(return_value=sanitization_error)
        
        # Pipeline should abort with PipelineAbortError
        with pytest.raises(PipelineAbortError) as exc_info:
            orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify abort details
        assert exc_info.value.stage == "JSONSanitizer"
        assert exc_info.value.error_code == "SCHEMA_VIOLATION"
        assert "schema validation" in exc_info.value.message.lower()
    
    def _mock_agents_until_sanitizer(self, orchestrator, temp_pdf_file):
        """Helper to mock all agents up to sanitizer."""
        # Mock ingestion
        file_ref = FileReference(
            file_path=temp_pdf_file,
            filename="test.pdf",
            size_bytes=500,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        
        # Mock validation
        orchestrator.validation_agent.execute = Mock(return_value=file_ref)
        
        # Mock vision OCR
        ocr_output = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="test.pdf",
                page_count=1,
                file_size_bytes=500
            ),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[],
                    math_expressions=[],
                    diagrams=[]
                )
            ]
        )
        orchestrator.vision_ocr_agent.execute = Mock(return_value=ocr_output)
        
        # Mock script architect
        storyboard = StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="test.pdf", page_count=1),
            configuration=Configuration(
                verbosity="medium",
                depth="intermediate",
                audience="undergraduate"
            ),
            concept_hierarchy=[],
            scenes=[
                Scene(
                    scene_id="scene_001",
                    scene_index=0,
                    concept_id="test",
                    narration="Test narration",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[],
                        emphasis_points=[]
                    ),
                    duration_estimate=30.0
                )
            ]
        )
        orchestrator.script_architect_agent.execute = Mock(return_value=storyboard)


class TestRetryLogic:
    """Test retry logic for transient failures."""
    
    def test_retry_on_transient_failure(self, temp_pdf_file, temp_output_dir):
        """Test orchestrator retries on transient failures.
        
        Requirements: 14.4
        """
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=config)
        
        # Track call count
        call_count = {'count': 0}
        
        def mock_vision_ocr_execute(input_data):
            call_count['count'] += 1
            if call_count['count'] < 3:
                # Fail first two attempts with retryable error
                raise AgentExecutionError(
                    error_code="API_TIMEOUT",
                    message="API request timed out"
                )
            # Third attempt succeeds
            return OCROutput(
                pdf_metadata=PDFMetadata(
                    filename="test.pdf",
                    page_count=1,
                    file_size_bytes=500
                ),
                pages=[
                    Page(
                        page_number=1,
                        text_blocks=[],
                        math_expressions=[],
                        diagrams=[]
                    )
                ]
            )
        
        # Mock ingestion and validation to succeed
        file_ref = FileReference(
            file_path=temp_pdf_file,
            filename="test.pdf",
            size_bytes=500,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        orchestrator.validation_agent.execute = Mock(return_value=file_ref)
        
        # Mock vision OCR with retry logic
        orchestrator.vision_ocr_agent.execute = Mock(side_effect=mock_vision_ocr_execute)
        orchestrator.vision_ocr_agent.get_retry_policy = Mock(
            return_value=RetryPolicy(
                max_attempts=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                base_delay_seconds=0.01,  # Short delay for testing
                max_delay_seconds=0.1,
                retryable_errors=["API_TIMEOUT", "API_RATE_LIMIT", "NETWORK_ERROR"]
            )
        )
        
        # Mock remaining agents to succeed
        self._mock_remaining_agents(orchestrator)
        
        # Execute pipeline - should succeed after retries
        manifest = orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify agent was called 3 times
        assert call_count['count'] == 3
        assert manifest.status == "SUCCESS"
    
    def test_no_retry_on_non_retryable_error(self, temp_pdf_file, temp_output_dir):
        """Test orchestrator does not retry non-retryable errors.
        
        Requirements: 14.4
        """
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=config)
        
        # Track call count
        call_count = {'count': 0}
        
        def mock_vision_ocr_execute(input_data):
            call_count['count'] += 1
            # Always fail with non-retryable error
            raise AgentExecutionError(
                error_code="INVALID_INPUT",
                message="Invalid input format"
            )
        
        # Mock ingestion and validation to succeed
        file_ref = FileReference(
            file_path=temp_pdf_file,
            filename="test.pdf",
            size_bytes=500,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        orchestrator.validation_agent.execute = Mock(return_value=file_ref)
        
        # Mock vision OCR to fail with non-retryable error
        orchestrator.vision_ocr_agent.execute = Mock(side_effect=mock_vision_ocr_execute)
        orchestrator.vision_ocr_agent.get_retry_policy = Mock(
            return_value=RetryPolicy(
                max_attempts=3,
                backoff_strategy=BackoffStrategy.EXPONENTIAL,
                base_delay_seconds=0.01,
                max_delay_seconds=0.1,
                retryable_errors=["API_TIMEOUT", "API_RATE_LIMIT"]  # INVALID_INPUT not in list
            )
        )
        
        # Pipeline should fail immediately without retry
        with pytest.raises(AgentExecutionError) as exc_info:
            orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify agent was called only once
        assert call_count['count'] == 1
        assert exc_info.value.error_code == "INVALID_INPUT"
    
    def _mock_remaining_agents(self, orchestrator):
        """Helper to mock remaining agents after vision OCR."""
        # Mock script architect
        storyboard = StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="test.pdf", page_count=1),
            configuration=Configuration(
                verbosity="medium",
                depth="intermediate",
                audience="undergraduate"
            ),
            concept_hierarchy=[],
            scenes=[
                Scene(
                    scene_id="scene_001",
                    scene_index=0,
                    concept_id="test",
                    narration="Test narration",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[],
                        emphasis_points=[]
                    ),
                    duration_estimate=30.0
                )
            ]
        )
        orchestrator.script_architect_agent.execute = Mock(return_value=storyboard)
        
        # Mock JSON sanitizer
        orchestrator.json_sanitizer_agent.execute = Mock(
            return_value=Mock(storyboard=storyboard)
        )
        
        # Mock scene controller
        animation_code = AnimationCodeObject(
            scene_id="scene_001",
            code="from manim import *\n\nclass Scene_001(Scene):\n    def construct(self):\n        pass",
            imports=["from manim import *"],
            class_name="Scene_001",
            narration="Test narration",
            duration_estimate=30.0,
            generation_timestamp=datetime.now(timezone.utc)
        )
        
        scene_report = SceneProcessingReport(
            total_scenes=1,
            successful_scenes=1,
            failed_scenes=[],
            output_directory="output/test/timestamp"
        )
        
        orchestrator.scene_controller_agent.execute = Mock(
            return_value=Mock(
                report=scene_report,
                animation_code_objects=[animation_code]
            )
        )
        
        # Mock persistence
        manifest = OutputManifest(
            output_directory="output/test/timestamp",
            files=[
                OutputFile(path="scene_001.py", size_bytes=100, type="animation_code"),
                OutputFile(path="narration_script.txt", size_bytes=50, type="narration_script"),
                OutputFile(path="storyboard.json", size_bytes=200, type="storyboard"),
                OutputFile(path="manifest.json", size_bytes=150, type="manifest")
            ],
            timestamp=datetime.now(timezone.utc),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=1,
            successful_scenes=1,
            failed_scenes=[]
        )
        
        orchestrator.persistence_agent.execute = Mock(
            return_value=Mock(manifest=manifest)
        )


class TestPartialSuccessScenarios:
    """Test pipeline behavior with partial scene failures."""
    
    def test_partial_success_some_scenes_fail(self, temp_pdf_file, temp_output_dir):
        """Test pipeline continues when some scenes fail but <50% fail.
        
        Requirements: 7.5, 14.4
        """
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=config)
        
        # Mock agents up to scene controller
        self._mock_agents_until_scene_controller(orchestrator, temp_pdf_file, num_scenes=4)
        
        # Mock scene controller with partial success (2 out of 4 scenes succeed)
        animation_codes = [
            AnimationCodeObject(
                scene_id="scene_001",
                code="from manim import *\n\nclass Scene_001(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Scene 1",
                duration_estimate=30.0,
                generation_timestamp=datetime.now(timezone.utc)
            ),
            AnimationCodeObject(
                scene_id="scene_003",
                code="from manim import *\n\nclass Scene_003(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name="Scene_003",
                narration="Scene 3",
                duration_estimate=30.0,
                generation_timestamp=datetime.now(timezone.utc)
            )
        ]
        
        scene_report = SceneProcessingReport(
            total_scenes=4,
            successful_scenes=2,
            failed_scenes=[
                FailedScene(scene_id="scene_002", error="Syntax error in generated code"),
                FailedScene(scene_id="scene_004", error="Invalid LaTeX expression")
            ],
            output_directory="output/test/timestamp"
        )
        
        orchestrator.scene_controller_agent.execute = Mock(
            return_value=Mock(
                report=scene_report,
                animation_code_objects=animation_codes
            )
        )
        
        # Mock persistence
        manifest = OutputManifest(
            output_directory="output/test/timestamp",
            files=[
                OutputFile(path="scene_001.py", size_bytes=100, type="animation_code"),
                OutputFile(path="scene_003.py", size_bytes=100, type="animation_code"),
                OutputFile(path="narration_script.txt", size_bytes=50, type="narration_script"),
                OutputFile(path="storyboard.json", size_bytes=200, type="storyboard"),
                OutputFile(path="manifest.json", size_bytes=150, type="manifest")
            ],
            timestamp=datetime.now(timezone.utc),
            status="PARTIAL_SUCCESS",
            scene_success_rate=0.5,
            total_scenes=4,
            successful_scenes=2,
            failed_scenes=[
                FailedSceneManifest(scene_id="scene_002", error="Syntax error in generated code"),
                FailedSceneManifest(scene_id="scene_004", error="Invalid LaTeX expression")
            ]
        )
        
        orchestrator.persistence_agent.execute = Mock(
            return_value=Mock(manifest=manifest)
        )
        
        # Execute pipeline - should complete with PARTIAL_SUCCESS
        manifest = orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify partial success
        assert manifest.status == "PARTIAL_SUCCESS"
        assert manifest.total_scenes == 4
        assert manifest.successful_scenes == 2
        assert len(manifest.failed_scenes) == 2
        assert manifest.scene_success_rate == 0.5
        
        # Verify failed scenes are tracked
        failed_scene_ids = {f.scene_id for f in manifest.failed_scenes}
        assert "scene_002" in failed_scene_ids
        assert "scene_004" in failed_scene_ids
    
    def test_critical_failure_when_majority_scenes_fail(self, temp_pdf_file, temp_output_dir):
        """Test pipeline returns CRITICAL_FAILURE when >50% scenes fail.
        
        Requirements: 14.4
        """
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=config)
        
        # Mock agents up to scene controller
        self._mock_agents_until_scene_controller(orchestrator, temp_pdf_file, num_scenes=4)
        
        # Mock scene controller with critical failure (only 1 out of 4 scenes succeeds)
        animation_codes = [
            AnimationCodeObject(
                scene_id="scene_001",
                code="from manim import *\n\nclass Scene_001(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Scene 1",
                duration_estimate=30.0,
                generation_timestamp=datetime.now(timezone.utc)
            )
        ]
        
        scene_report = SceneProcessingReport(
            total_scenes=4,
            successful_scenes=1,
            failed_scenes=[
                FailedScene(scene_id="scene_002", error="Code generation failed"),
                FailedScene(scene_id="scene_003", error="Invalid visual intent"),
                FailedScene(scene_id="scene_004", error="Manim API error")
            ],
            output_directory="output/test/timestamp"
        )
        
        orchestrator.scene_controller_agent.execute = Mock(
            return_value=Mock(
                report=scene_report,
                animation_code_objects=animation_codes
            )
        )
        
        # Mock persistence
        manifest = OutputManifest(
            output_directory="output/test/timestamp",
            files=[
                OutputFile(path="scene_001.py", size_bytes=100, type="animation_code"),
                OutputFile(path="narration_script.txt", size_bytes=50, type="narration_script"),
                OutputFile(path="storyboard.json", size_bytes=200, type="storyboard"),
                OutputFile(path="manifest.json", size_bytes=150, type="manifest")
            ],
            timestamp=datetime.now(timezone.utc),
            status="CRITICAL_FAILURE",
            scene_success_rate=0.25,
            total_scenes=4,
            successful_scenes=1,
            failed_scenes=[
                FailedSceneManifest(scene_id="scene_002", error="Code generation failed"),
                FailedSceneManifest(scene_id="scene_003", error="Invalid visual intent"),
                FailedSceneManifest(scene_id="scene_004", error="Manim API error")
            ]
        )
        
        orchestrator.persistence_agent.execute = Mock(
            return_value=Mock(manifest=manifest)
        )
        
        # Execute pipeline - should complete with CRITICAL_FAILURE
        manifest = orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify critical failure
        assert manifest.status == "CRITICAL_FAILURE"
        assert manifest.total_scenes == 4
        assert manifest.successful_scenes == 1
        assert len(manifest.failed_scenes) == 3
        assert manifest.scene_success_rate == 0.25
        
        # Verify failure rate is >50%
        failure_rate = len(manifest.failed_scenes) / manifest.total_scenes
        assert failure_rate > 0.5
    
    def _mock_agents_until_scene_controller(self, orchestrator, temp_pdf_file, num_scenes):
        """Helper to mock all agents up to scene controller."""
        # Mock ingestion
        file_ref = FileReference(
            file_path=temp_pdf_file,
            filename="test.pdf",
            size_bytes=500,
            upload_timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        orchestrator.ingestion_agent.execute = Mock(
            return_value=IngestionOutput(file_reference=file_ref)
        )
        
        # Mock validation
        orchestrator.validation_agent.execute = Mock(return_value=file_ref)
        
        # Mock vision OCR
        ocr_output = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="test.pdf",
                page_count=1,
                file_size_bytes=500
            ),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[],
                    math_expressions=[],
                    diagrams=[]
                )
            ]
        )
        orchestrator.vision_ocr_agent.execute = Mock(return_value=ocr_output)
        
        # Mock script architect with multiple scenes
        scenes = [
            Scene(
                scene_id=f"scene_{i:03d}",
                scene_index=i-1,
                concept_id=f"concept_{i}",
                narration=f"Scene {i} narration",
                visual_intent=VisualIntent(
                    mathematical_objects=[],
                    transformations=[],
                    emphasis_points=[]
                ),
                duration_estimate=30.0
            )
            for i in range(1, num_scenes + 1)
        ]
        
        storyboard = StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="test.pdf", page_count=1),
            configuration=Configuration(
                verbosity="medium",
                depth="intermediate",
                audience="undergraduate"
            ),
            concept_hierarchy=[],
            scenes=scenes
        )
        orchestrator.script_architect_agent.execute = Mock(return_value=storyboard)
        
        # Mock JSON sanitizer
        orchestrator.json_sanitizer_agent.execute = Mock(
            return_value=Mock(storyboard=storyboard)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
