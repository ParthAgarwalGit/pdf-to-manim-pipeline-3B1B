"""Tests for Persistence Agent."""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from src.agents.persistence import PersistenceAgent, PersistenceInput, PersistenceOutput
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.scene_processing import FailedScene, SceneProcessingReport
from src.schemas.storyboard import (
    Configuration,
    PDFMetadataStoryboard,
    Scene,
    StoryboardJSON,
    VisualIntent,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_storyboard():
    """Create a sample storyboard for testing."""
    return StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(
            filename="test_document.pdf",
            page_count=5
        ),
        configuration=Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        ),
        concept_hierarchy=[],
        scenes=[
            Scene(
                scene_id="intro_001",
                scene_index=0,
                concept_id="vector_space",
                narration="Let's explore what a vector space is...",
                visual_intent=VisualIntent(
                    mathematical_objects=[],
                    transformations=[]
                ),
                duration_estimate=45.0
            ),
            Scene(
                scene_id="definition_002",
                scene_index=1,
                concept_id="vector_space",
                narration="A vector space is a set with two operations...",
                visual_intent=VisualIntent(
                    mathematical_objects=[],
                    transformations=[]
                ),
                duration_estimate=60.0
            )
        ]
    )


@pytest.fixture
def sample_animation_code_objects():
    """Create sample animation code objects for testing."""
    return [
        AnimationCodeObject(
            scene_id="intro_001",
            code="class Scene_001(Scene):\n    def construct(self):\n        pass",
            imports=["from manim import *"],
            class_name="Scene_001",
            narration="Let's explore what a vector space is...",
            duration_estimate=45.0,
            generation_timestamp=datetime(2024, 1, 15, 10, 30, 0)
        ),
        AnimationCodeObject(
            scene_id="definition_002",
            code="class Scene_002(Scene):\n    def construct(self):\n        pass",
            imports=["from manim import *"],
            class_name="Scene_002",
            narration="A vector space is a set with two operations...",
            duration_estimate=60.0,
            generation_timestamp=datetime(2024, 1, 15, 10, 31, 0)
        )
    ]


@pytest.fixture
def sample_scene_processing_report(temp_output_dir):
    """Create a sample scene processing report."""
    return SceneProcessingReport(
        total_scenes=2,
        successful_scenes=2,
        failed_scenes=[],
        output_directory=temp_output_dir
    )


class TestPersistenceAgentBasics:
    """Test basic Persistence Agent functionality."""
    
    def test_agent_initialization(self, temp_output_dir):
        """Test agent can be initialized with custom base directory."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        assert agent.base_output_dir == temp_output_dir
    
    def test_agent_initialization_default(self):
        """Test agent uses default output directory."""
        agent = PersistenceAgent()
        assert agent.base_output_dir == "output"
    
    def test_validate_input_valid(
        self,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test input validation with valid input."""
        agent = PersistenceAgent()
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        assert agent.validate_input(input_data) is True
    
    def test_validate_input_empty_animation_codes(
        self,
        sample_storyboard,
        sample_scene_processing_report
    ):
        """Test input validation accepts empty animation code list."""
        agent = PersistenceAgent()
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=[],
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        assert agent.validate_input(input_data) is True
    
    def test_get_retry_policy(self):
        """Test retry policy configuration."""
        agent = PersistenceAgent()
        policy = agent.get_retry_policy()
        
        assert policy.max_attempts == 3
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 10.0


class TestPersistenceAgentExecution:
    """Test Persistence Agent execution."""
    
    def test_execute_success(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test successful execution creates all expected files."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=timestamp
        )
        
        output = agent.execute(input_data)
        
        # Verify output is PersistenceOutput
        assert isinstance(output, PersistenceOutput)
        assert output.manifest is not None
        
        # Verify output directory was created
        expected_dir = os.path.join(temp_output_dir, "test_document", "2024-01-15T10-30-00")
        assert os.path.exists(expected_dir)
        
        # Verify all expected files exist
        assert os.path.exists(os.path.join(expected_dir, "scene_intro_001.py"))
        assert os.path.exists(os.path.join(expected_dir, "scene_definition_002.py"))
        assert os.path.exists(os.path.join(expected_dir, "narration_script.txt"))
        assert os.path.exists(os.path.join(expected_dir, "storyboard.json"))
        assert os.path.exists(os.path.join(expected_dir, "processing_report.json"))
        assert os.path.exists(os.path.join(expected_dir, "manifest.json"))
        
        # Verify manifest contains correct information
        assert output.manifest.total_scenes == 2
        assert output.manifest.successful_scenes == 2
        assert output.manifest.status == "SUCCESS"
        assert output.manifest.scene_success_rate == 1.0
        assert len(output.manifest.files) == 6  # 2 scenes + 4 metadata files
    
    def test_execute_with_failed_scenes(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects
    ):
        """Test execution with some failed scenes."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create report with one failed scene
        report = SceneProcessingReport(
            total_scenes=3,
            successful_scenes=2,
            failed_scenes=[
                FailedScene(scene_id="proof_003", error="LaTeX syntax error")
            ],
            output_directory=temp_output_dir
        )
        
        input_data = PersistenceInput(
            scene_processing_report=report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        output = agent.execute(input_data)
        
        # Verify manifest reflects partial success
        assert output.manifest.total_scenes == 3
        assert output.manifest.successful_scenes == 2
        assert output.manifest.status == "PARTIAL_SUCCESS"
        assert len(output.manifest.failed_scenes) == 1
        assert output.manifest.failed_scenes[0].scene_id == "proof_003"
    
    def test_execute_all_scenes_failed(
        self,
        temp_output_dir,
        sample_storyboard
    ):
        """Test execution when all scenes failed."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        report = SceneProcessingReport(
            total_scenes=2,
            successful_scenes=0,
            failed_scenes=[
                FailedScene(scene_id="intro_001", error="Error 1"),
                FailedScene(scene_id="definition_002", error="Error 2")
            ],
            output_directory=temp_output_dir
        )
        
        input_data = PersistenceInput(
            scene_processing_report=report,
            animation_code_objects=[],  # No successful scenes
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        output = agent.execute(input_data)
        
        # Verify manifest reflects failure
        assert output.manifest.total_scenes == 2
        assert output.manifest.successful_scenes == 0
        assert output.manifest.status == "FAILURE"
        assert output.manifest.scene_success_rate == 0.0


class TestPersistenceAgentFileContent:
    """Test content of generated files."""
    
    def test_animation_code_file_content(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test animation code files contain correct content."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        agent.execute(input_data)
        
        # Read first scene file
        scene_file = os.path.join(
            temp_output_dir,
            "test_document",
            "2024-01-15T10-30-00",
            "scene_intro_001.py"
        )
        
        with open(scene_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify header comment exists
        assert "Scene: intro_001" in content
        assert "Generated: 2024-01-15T10:30:00" in content
        assert "Let's explore what a vector space is..." in content
        assert "Duration: 45.0s" in content
        
        # Verify imports
        assert "from manim import *" in content
        
        # Verify code
        assert "class Scene_001(Scene):" in content
        assert "def construct(self):" in content
    
    def test_narration_script_content(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test narration script contains correct content."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        agent.execute(input_data)
        
        # Read narration script
        narration_file = os.path.join(
            temp_output_dir,
            "test_document",
            "2024-01-15T10-30-00",
            "narration_script.txt"
        )
        
        with open(narration_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify header
        assert "NARRATION SCRIPT" in content
        assert "PDF: test_document.pdf" in content
        assert "Total Scenes: 2" in content
        
        # Verify scene markers
        assert "[SCENE 1: intro_001]" in content
        assert "[SCENE 2: definition_002]" in content
        
        # Verify narration text
        assert "Let's explore what a vector space is..." in content
        assert "A vector space is a set with two operations..." in content
        
        # Verify duration
        assert "Duration: 45.0s" in content
        assert "Duration: 60.0s" in content
    
    def test_storyboard_json_content(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test storyboard JSON file is valid."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        agent.execute(input_data)
        
        # Read storyboard JSON
        storyboard_file = os.path.join(
            temp_output_dir,
            "test_document",
            "2024-01-15T10-30-00",
            "storyboard.json"
        )
        
        with open(storyboard_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify structure
        assert "pdf_metadata" in data
        assert "configuration" in data
        assert "scenes" in data
        assert len(data["scenes"]) == 2
        assert data["pdf_metadata"]["filename"] == "test_document.pdf"
    
    def test_manifest_json_content(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test manifest JSON file is valid."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        agent.execute(input_data)
        
        # Read manifest JSON
        manifest_file = os.path.join(
            temp_output_dir,
            "test_document",
            "2024-01-15T10-30-00",
            "manifest.json"
        )
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify structure
        assert "output_directory" in data
        assert "files" in data
        assert "timestamp" in data
        assert "status" in data
        assert "total_scenes" in data
        assert "successful_scenes" in data
        assert "scene_success_rate" in data
        
        # Verify values
        assert data["status"] == "SUCCESS"
        assert data["total_scenes"] == 2
        assert data["successful_scenes"] == 2
        assert data["scene_success_rate"] == 1.0
        assert len(data["files"]) == 6


class TestPersistenceAgentIdempotency:
    """Test idempotency of Persistence Agent."""
    
    def test_idempotency_retry_produces_same_output(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """
        **Property 5: Retrying persistence with same input produces identical output**
        
        For any scene processing report and animation code objects, writing twice
        to the same output directory should produce identical files.
        
        **Validates: Requirements 11.1, 11.2**
        """
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=timestamp
        )
        
        # First execution
        output1 = agent.execute(input_data)
        
        # Read all file contents
        output_dir = os.path.join(temp_output_dir, "test_document", "2024-01-15T10-30-00")
        files_content_1 = {}
        for file_info in output1.manifest.files:
            file_path = os.path.join(output_dir, file_info.path)
            with open(file_path, 'r', encoding='utf-8') as f:
                files_content_1[file_info.path] = f.read()
        
        # Second execution (retry)
        output2 = agent.execute(input_data)
        
        # Read all file contents again
        files_content_2 = {}
        for file_info in output2.manifest.files:
            file_path = os.path.join(output_dir, file_info.path)
            with open(file_path, 'r', encoding='utf-8') as f:
                files_content_2[file_info.path] = f.read()
        
        # Verify same files exist
        assert set(files_content_1.keys()) == set(files_content_2.keys())
        
        # Verify file contents are identical
        for filename in files_content_1.keys():
            assert files_content_1[filename] == files_content_2[filename], \
                f"File {filename} differs between executions"
        
        # Verify manifests are structurally equivalent
        assert output1.manifest.total_scenes == output2.manifest.total_scenes
        assert output1.manifest.successful_scenes == output2.manifest.successful_scenes
        assert output1.manifest.status == output2.manifest.status
        assert len(output1.manifest.files) == len(output2.manifest.files)
    
    def test_cleanup_incomplete_files_on_retry(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test that incomplete files are cleaned up on retry."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        output_dir = os.path.join(temp_output_dir, "test_document", "2024-01-15T10-30-00")
        
        # Create output directory with incomplete file
        os.makedirs(output_dir, exist_ok=True)
        incomplete_file = os.path.join(output_dir, "incomplete.tmp")
        with open(incomplete_file, 'w') as f:
            f.write("incomplete data")
        
        # Verify incomplete file exists
        assert os.path.exists(incomplete_file)
        
        # Execute persistence
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=timestamp
        )
        
        agent.execute(input_data)
        
        # Verify incomplete file was cleaned up
        assert not os.path.exists(incomplete_file)
        
        # Verify expected files exist
        assert os.path.exists(os.path.join(output_dir, "scene_intro_001.py"))
        assert os.path.exists(os.path.join(output_dir, "manifest.json"))


class TestPersistenceAgentAtomicWrites:
    """Test atomic file write operations."""
    
    def test_atomic_write_creates_file(self, temp_output_dir):
        """Test atomic write successfully creates file."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        file_path = os.path.join(temp_output_dir, "test_file.txt")
        content = "Test content"
        
        agent._write_file_atomically(file_path, content)
        
        # Verify file exists and has correct content
        assert os.path.exists(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            assert f.read() == content
    
    def test_atomic_write_overwrites_existing(self, temp_output_dir):
        """Test atomic write overwrites existing file."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        file_path = os.path.join(temp_output_dir, "test_file.txt")
        
        # Write initial content
        with open(file_path, 'w') as f:
            f.write("Old content")
        
        # Atomic write with new content
        new_content = "New content"
        agent._write_file_atomically(file_path, new_content)
        
        # Verify file has new content
        with open(file_path, 'r', encoding='utf-8') as f:
            assert f.read() == new_content


class TestPersistenceAgentErrorHandling:
    """Test error handling in Persistence Agent."""
    
    def test_invalid_input_raises_error(self):
        """Test that invalid input raises AgentExecutionError."""
        from src.agents.base import AgentExecutionError
        
        agent = PersistenceAgent()
        
        # Create invalid input (missing required fields)
        class InvalidInput:
            pass
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute(InvalidInput())
        
        assert exc_info.value.error_code == "INVALID_INPUT"
    
    def test_directory_creation_failure_raises_error(
        self,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test that directory creation failure raises AgentExecutionError."""
        from src.agents.base import AgentExecutionError
        import platform
        
        # Use invalid base directory that should fail on any platform
        if platform.system() == "Windows":
            # On Windows, use an invalid drive letter or reserved name
            invalid_path = "Z:\\invalid\\path\\that\\should\\not\\exist"
        else:
            # On Unix-like systems, use /dev/null/invalid
            invalid_path = "/dev/null/invalid"
        
        agent = PersistenceAgent(base_output_dir=invalid_path)
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute(input_data)
        
        assert exc_info.value.error_code == "DIRECTORY_CREATION_FAILED"
    
    def test_file_write_permission_error(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test error handling when file write fails due to permissions."""
        from src.agents.base import AgentExecutionError
        import platform
        
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create output directory first
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        output_dir = os.path.join(temp_output_dir, "test_document", "2024-01-15T10-30-00")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a read-only file that should cause write failure
        readonly_file = os.path.join(output_dir, "scene_intro_001.py")
        with open(readonly_file, 'w') as f:
            f.write("existing content")
        
        # Make file read-only (platform-specific)
        if platform.system() == "Windows":
            import stat
            os.chmod(readonly_file, stat.S_IREAD)
        else:
            os.chmod(readonly_file, 0o444)  # Read-only for all
        
        # Make directory read-only to prevent new file creation
        if platform.system() != "Windows":  # Unix-like systems
            os.chmod(output_dir, 0o555)  # Read and execute only
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document",
            timestamp=timestamp
        )
        
        try:
            with pytest.raises(AgentExecutionError) as exc_info:
                agent.execute(input_data)
            
            assert exc_info.value.error_code == "FILE_WRITE_FAILED"
            assert "Failed to write animation code files" in str(exc_info.value)
        
        finally:
            # Cleanup: restore permissions for cleanup
            if platform.system() != "Windows":
                os.chmod(output_dir, 0o755)
            if os.path.exists(readonly_file):
                if platform.system() == "Windows":
                    import stat
                    os.chmod(readonly_file, stat.S_IWRITE | stat.S_IREAD)
                else:
                    os.chmod(readonly_file, 0o644)
    
    def test_disk_full_simulation(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test error handling when disk is full (simulated)."""
        from src.agents.base import AgentExecutionError
        
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Mock the atomic write method to simulate disk full error
        original_write = agent._write_file_atomically
        
        def disk_full_write(file_path, content):
            raise OSError(28, "No space left on device")  # ENOSPC error
        
        agent._write_file_atomically = disk_full_write
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        try:
            with pytest.raises(AgentExecutionError) as exc_info:
                agent.execute(input_data)
            
            assert exc_info.value.error_code == "FILE_WRITE_FAILED"
            assert "Failed to write animation code files" in str(exc_info.value)
        
        finally:
            # Restore original method
            agent._write_file_atomically = original_write
    
    def test_invalid_filename_characters(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_scene_processing_report
    ):
        """Test error handling with invalid filename characters."""
        from src.agents.base import AgentExecutionError
        
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create animation code with invalid scene_id containing filesystem-invalid characters
        invalid_animation_codes = [
            AnimationCodeObject(
                scene_id="scene/with\\invalid:chars*",  # Invalid characters for most filesystems
                code="class Scene_001(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=45.0,
                generation_timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        ]
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=invalid_animation_codes,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute(input_data)
        
        assert exc_info.value.error_code == "FILE_WRITE_FAILED"
    
    def test_narration_script_write_failure(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test error handling when narration script write fails."""
        from src.agents.base import AgentExecutionError
        
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Mock the narration script write method to simulate failure
        original_write_narration = agent._write_narration_script
        
        def failing_write_narration(output_dir, storyboard):
            raise OSError("Simulated narration write failure")
        
        agent._write_narration_script = failing_write_narration
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        try:
            with pytest.raises(AgentExecutionError) as exc_info:
                agent.execute(input_data)
            
            assert exc_info.value.error_code == "FILE_WRITE_FAILED"
            assert "Failed to write narration script" in str(exc_info.value)
        
        finally:
            # Restore original method
            agent._write_narration_script = original_write_narration
    
    def test_storyboard_write_failure(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test error handling when storyboard write fails."""
        from src.agents.base import AgentExecutionError
        
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Mock the storyboard write method to simulate failure
        original_write_storyboard = agent._write_storyboard
        
        def failing_write_storyboard(output_dir, storyboard):
            raise OSError("Simulated storyboard write failure")
        
        agent._write_storyboard = failing_write_storyboard
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        try:
            with pytest.raises(AgentExecutionError) as exc_info:
                agent.execute(input_data)
            
            assert exc_info.value.error_code == "FILE_WRITE_FAILED"
            assert "Failed to write storyboard" in str(exc_info.value)
        
        finally:
            # Restore original method
            agent._write_storyboard = original_write_storyboard
    
    def test_manifest_write_failure(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test error handling when manifest write fails."""
        from src.agents.base import AgentExecutionError
        
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Mock the manifest write method to simulate failure
        original_write_manifest = agent._write_manifest
        
        def failing_write_manifest(output_dir, manifest):
            raise OSError("Simulated manifest write failure")
        
        agent._write_manifest = failing_write_manifest
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        try:
            with pytest.raises(AgentExecutionError) as exc_info:
                agent.execute(input_data)
            
            assert exc_info.value.error_code == "FILE_WRITE_FAILED"
            assert "Failed to write manifest" in str(exc_info.value)
        
        finally:
            # Restore original method
            agent._write_manifest = original_write_manifest


class TestPersistenceAgentEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_animation_codes_with_all_failed_scenes(
        self,
        temp_output_dir,
        sample_storyboard
    ):
        """Test handling when all scenes failed and no animation codes exist."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create report with all scenes failed
        report = SceneProcessingReport(
            total_scenes=3,
            successful_scenes=0,
            failed_scenes=[
                FailedScene(scene_id="scene_001", error="Error 1"),
                FailedScene(scene_id="scene_002", error="Error 2"),
                FailedScene(scene_id="scene_003", error="Error 3")
            ],
            output_directory=temp_output_dir
        )
        
        input_data = PersistenceInput(
            scene_processing_report=report,
            animation_code_objects=[],  # No successful scenes
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        output = agent.execute(input_data)
        
        # Verify execution succeeds even with no animation codes
        assert isinstance(output, PersistenceOutput)
        assert output.manifest.total_scenes == 3
        assert output.manifest.successful_scenes == 0
        assert output.manifest.status == "FAILURE"
        assert len(output.manifest.failed_scenes) == 3
        
        # Verify only metadata files exist (no scene files)
        expected_dir = os.path.join(temp_output_dir, "test_document")
        assert os.path.exists(expected_dir)
        
        # Should have narration, storyboard, report, and manifest files
        # but no scene_*.py files
        manifest_files = [f.path for f in output.manifest.files]
        scene_files = [f for f in manifest_files if f.startswith("scene_")]
        assert len(scene_files) == 0
        
        # Should have metadata files
        assert "narration_script.txt" in manifest_files
        assert "storyboard.json" in manifest_files
        assert "processing_report.json" in manifest_files
        assert "manifest.json" in manifest_files
    
    def test_very_long_scene_id(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_scene_processing_report
    ):
        """Test handling of very long scene IDs."""
        import platform
        
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Use a moderately long scene_id that works on all platforms
        # Windows has a 260 character path limit, so we need to be careful
        # Use 50 characters which should work on all platforms
        long_scene_id = "a" * 50  # 50 character scene ID
        long_animation_codes = [
            AnimationCodeObject(
                scene_id=long_scene_id,
                code="class Scene_001(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=45.0,
                generation_timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        ]
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=long_animation_codes,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        # Should handle long scene IDs gracefully
        output = agent.execute(input_data)
        
        # Verify file was created with long name
        expected_filename = f"scene_{long_scene_id}.py"
        manifest_files = [f.path for f in output.manifest.files]
        assert expected_filename in manifest_files
    
    def test_special_characters_in_pdf_filename(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test handling of special characters in PDF filename."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Use PDF filename with spaces and special characters
        special_filename = "My Document (2024) - Version 1.2"
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename=special_filename
        )
        
        output = agent.execute(input_data)
        
        # Verify directory was created with special filename
        expected_dir = os.path.join(temp_output_dir, special_filename)
        assert os.path.exists(expected_dir)
        
        # Verify manifest contains correct directory path
        assert special_filename in output.manifest.output_directory
    
    def test_unicode_characters_in_narration(
        self,
        temp_output_dir,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test handling of Unicode characters in narration text."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create storyboard with Unicode characters in narration
        unicode_storyboard = StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(
                filename="test_document.pdf",
                page_count=5
            ),
            configuration=Configuration(
                verbosity="medium",
                depth="intermediate",
                audience="undergraduate"
            ),
            concept_hierarchy=[],
            scenes=[
                Scene(
                    scene_id="unicode_001",
                    scene_index=0,
                    concept_id="vector_space",
                    narration="Let's explore vectors: →, ∈, ∀, ∃, ∑, ∫, α, β, γ, δ, π, θ, λ, μ, σ, φ, ψ, ω",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[]
                    ),
                    duration_estimate=45.0
                )
            ]
        )
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=unicode_storyboard,
            pdf_filename="test_document"
        )
        
        output = agent.execute(input_data)
        
        # Verify execution succeeds with Unicode characters
        assert isinstance(output, PersistenceOutput)
        
        # Verify narration script contains Unicode characters
        # Find the actual output directory from the manifest
        output_dir = output.manifest.output_directory
        narration_path = os.path.join(output_dir, "narration_script.txt")
        
        # Verify narration file exists
        assert os.path.exists(narration_path), f"Narration file not found at {narration_path}"
        
        # Read narration file and verify Unicode content
        with open(narration_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "→" in content
        assert "∈" in content
        assert "α" in content
        assert "π" in content
    
    def test_very_large_code_content(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_scene_processing_report
    ):
        """Test handling of very large animation code content."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create animation code with very large content
        large_code = "class Scene_001(Scene):\n    def construct(self):\n"
        large_code += "        # " + "A" * 10000 + "\n"  # Large comment
        large_code += "        pass\n"
        
        large_animation_codes = [
            AnimationCodeObject(
                scene_id="large_001",
                code=large_code,
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration with large code",
                duration_estimate=45.0,
                generation_timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        ]
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=large_animation_codes,
            storyboard=sample_storyboard,
            pdf_filename="test_document"
        )
        
        output = agent.execute(input_data)
        
        # Verify execution succeeds with large content
        assert isinstance(output, PersistenceOutput)
        
        # Verify file was created and has correct size
        scene_files = [f for f in output.manifest.files if f.path.startswith("scene_")]
        assert len(scene_files) == 1
        assert scene_files[0].size_bytes > 10000  # Should be large due to content
    
    def test_timestamp_formatting_edge_cases(
        self,
        temp_output_dir,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test timestamp formatting with various edge cases."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Test with different timestamp formats
        test_timestamps = [
            datetime(2024, 1, 1, 0, 0, 0),  # New Year
            datetime(2024, 12, 31, 23, 59, 59),  # End of year
            datetime(2024, 2, 29, 12, 30, 45),  # Leap year
        ]
        
        for i, timestamp in enumerate(test_timestamps):
            input_data = PersistenceInput(
                scene_processing_report=sample_scene_processing_report,
                animation_code_objects=sample_animation_code_objects,
                storyboard=sample_storyboard,
                pdf_filename=f"test_document_{i}",
                timestamp=timestamp
            )
            
            output = agent.execute(input_data)
            
            # Verify directory was created with correct timestamp format
            expected_timestamp = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
            expected_dir = os.path.join(temp_output_dir, f"test_document_{i}", expected_timestamp)
            assert os.path.exists(expected_dir)
            
            # Verify manifest timestamp matches
            assert output.manifest.timestamp == timestamp


class TestPersistenceAgentValidation:
    """Test input validation edge cases."""
    
    def test_validate_input_missing_scene_processing_report(
        self,
        sample_storyboard,
        sample_animation_code_objects
    ):
        """Test validation fails when scene_processing_report is missing."""
        agent = PersistenceAgent()
        
        # Pydantic will catch this at construction time, so we need to
        # test the validate_input method directly with a mock object
        class MockInput:
            scene_processing_report = None
            animation_code_objects = sample_animation_code_objects
            storyboard = sample_storyboard
            pdf_filename = "test_document"
        
        mock_input = MockInput()
        assert agent.validate_input(mock_input) is False
    
    def test_validate_input_missing_storyboard(
        self,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test validation fails when storyboard is missing."""
        agent = PersistenceAgent()
        
        # Pydantic will catch this at construction time, so we need to
        # test the validate_input method directly with a mock object
        class MockInput:
            scene_processing_report = sample_scene_processing_report
            animation_code_objects = sample_animation_code_objects
            storyboard = None
            pdf_filename = "test_document"
        
        mock_input = MockInput()
        assert agent.validate_input(mock_input) is False
    
    def test_validate_input_missing_pdf_filename(
        self,
        sample_storyboard,
        sample_animation_code_objects,
        sample_scene_processing_report
    ):
        """Test validation fails when pdf_filename is missing."""
        agent = PersistenceAgent()
        
        input_data = PersistenceInput(
            scene_processing_report=sample_scene_processing_report,
            animation_code_objects=sample_animation_code_objects,
            storyboard=sample_storyboard,
            pdf_filename=""
        )
        
        assert agent.validate_input(input_data) is False
    
    def test_validate_input_invalid_animation_codes_type(
        self,
        sample_storyboard,
        sample_scene_processing_report
    ):
        """Test validation fails when animation_code_objects is not a list."""
        agent = PersistenceAgent()
        
        # Pydantic will catch this at construction time, so we need to
        # test the validate_input method directly with a mock object
        class MockInput:
            scene_processing_report = sample_scene_processing_report
            animation_code_objects = "not_a_list"
            storyboard = sample_storyboard
            pdf_filename = "test_document"
        
        mock_input = MockInput()
        assert agent.validate_input(mock_input) is False
    """Property-based tests for Persistence Agent idempotency."""
    
    @given(
        scene_count=st.integers(min_value=1, max_value=10),
        pdf_filename=st.text(
            min_size=1, 
            max_size=20, 
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))
        ).filter(lambda s: s.upper() not in ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                                               'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                                               'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']),
        success_rate=st.floats(min_value=0.0, max_value=1.0),
        narration_text=st.text(min_size=10, max_size=200),
        duration=st.floats(min_value=10.0, max_value=300.0)
    )
    def test_property_persistence_idempotency(
        self,
        scene_count,
        pdf_filename,
        success_rate,
        narration_text,
        duration
    ):
        """
        **Property 5: Retrying persistence with same input produces identical output**
        
        For any scene processing report and animation code objects, writing twice
        to the same output directory should produce identical files (same content, same structure).
        
        **Validates: Requirements 11.1, 11.2**
        """
        # Create temporary directory for this test
        import tempfile
        temp_output_dir = tempfile.mkdtemp()
        
        try:
            agent = PersistenceAgent(base_output_dir=temp_output_dir)
            
            # Generate test data based on property parameters
            successful_scenes = int(scene_count * success_rate)
            failed_scenes = scene_count - successful_scenes
            
            # Create animation code objects for successful scenes
            animation_codes = []
            for i in range(successful_scenes):
                animation_codes.append(AnimationCodeObject(
                    scene_id=f"scene_{i:03d}",
                    code=f"class Scene_{i:03d}(Scene):\n    def construct(self):\n        # {narration_text[:50]}\n        pass",
                    imports=["from manim import *"],
                    class_name=f"Scene_{i:03d}",
                    narration=f"{narration_text} - Scene {i}",
                    duration_estimate=duration + i * 5,
                    generation_timestamp=datetime(2024, 1, 15, 10, 30, i)
                ))
            
            # Create failed scenes
            failed_scene_list = []
            for i in range(successful_scenes, scene_count):
                failed_scene_list.append(FailedScene(
                    scene_id=f"scene_{i:03d}",
                    error=f"Generation failed for scene {i}: {narration_text[:30]}"
                ))
            
            # Create scene processing report
            report = SceneProcessingReport(
                total_scenes=scene_count,
                successful_scenes=successful_scenes,
                failed_scenes=failed_scene_list,
                output_directory=temp_output_dir
            )
            
            # Create storyboard with scenes
            scenes = []
            for i in range(scene_count):
                scenes.append(Scene(
                    scene_id=f"scene_{i:03d}",
                    scene_index=i,
                    concept_id=f"concept_{i}",
                    narration=f"{narration_text} - Scene {i}",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[]
                    ),
                    duration_estimate=duration + i * 5
                ))
            
            storyboard = StoryboardJSON(
                pdf_metadata=PDFMetadataStoryboard(
                    filename=f"{pdf_filename}.pdf",
                    page_count=5
                ),
                configuration=Configuration(
                    verbosity="medium",
                    depth="intermediate",
                    audience="undergraduate"
                ),
                concept_hierarchy=[],
                scenes=scenes
            )
            
            # Create input
            timestamp = datetime(2024, 1, 15, 10, 30, 0)
            input_data = PersistenceInput(
                scene_processing_report=report,
                animation_code_objects=animation_codes,
                storyboard=storyboard,
                pdf_filename=pdf_filename,
                timestamp=timestamp
            )
            
            # First execution
            output1 = agent.execute(input_data)
            
            # Read all file contents after first execution
            output_dir = os.path.join(temp_output_dir, pdf_filename, "2024-01-15T10-30-00")
            files_content_1 = {}
            file_sizes_1 = {}
            
            for file_info in output1.manifest.files:
                file_path = os.path.join(output_dir, file_info.path)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files_content_1[file_info.path] = f.read()
                    file_sizes_1[file_info.path] = os.path.getsize(file_path)
            
            # Second execution (retry with same input)
            output2 = agent.execute(input_data)
            
            # Read all file contents after second execution
            files_content_2 = {}
            file_sizes_2 = {}
            
            for file_info in output2.manifest.files:
                file_path = os.path.join(output_dir, file_info.path)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files_content_2[file_info.path] = f.read()
                    file_sizes_2[file_info.path] = os.path.getsize(file_path)
            
            # Verify identical output structure
            assert set(files_content_1.keys()) == set(files_content_2.keys()), \
                "File lists should be identical between executions"
            
            # Verify identical file contents
            for filename in files_content_1.keys():
                assert files_content_1[filename] == files_content_2[filename], \
                    f"File {filename} content differs between executions"
                assert file_sizes_1[filename] == file_sizes_2[filename], \
                    f"File {filename} size differs between executions"
            
            # Verify manifests are structurally equivalent
            assert output1.manifest.total_scenes == output2.manifest.total_scenes
            assert output1.manifest.successful_scenes == output2.manifest.successful_scenes
            assert output1.manifest.status == output2.manifest.status
            assert output1.manifest.scene_success_rate == output2.manifest.scene_success_rate
            assert len(output1.manifest.files) == len(output2.manifest.files)
            
            # Verify failed scenes are identical
            assert len(output1.manifest.failed_scenes) == len(output2.manifest.failed_scenes)
            for i, (failed1, failed2) in enumerate(zip(output1.manifest.failed_scenes, output2.manifest.failed_scenes)):
                assert failed1.scene_id == failed2.scene_id, f"Failed scene {i} ID differs"
                assert failed1.error == failed2.error, f"Failed scene {i} error differs"
        
        finally:
            # Cleanup temporary directory
            import shutil
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
    
    @given(
        scene_count=st.integers(min_value=1, max_value=5),
        pdf_filename=st.text(
            min_size=1, 
            max_size=20, 
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))
        ).filter(lambda s: s.upper() not in ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                                               'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                                               'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']),
        success_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_property_idempotency_multiple_retries(
        self,
        scene_count,
        pdf_filename,
        success_rate
    ):
        """
        **Property 5 Extended: Multiple retries produce identical output**
        
        For any valid input, executing persistence multiple times should
        always produce identical files and directory structure.
        
        **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
        """
        # Create temporary directory for this test
        import tempfile
        temp_output_dir = tempfile.mkdtemp()
        
        try:
            agent = PersistenceAgent(base_output_dir=temp_output_dir)
            
            # Generate test data based on property parameters
            successful_scenes = int(scene_count * success_rate)
            failed_scenes = scene_count - successful_scenes
            
            # Create animation code objects for successful scenes
            animation_codes = []
            for i in range(successful_scenes):
                animation_codes.append(AnimationCodeObject(
                    scene_id=f"scene_{i:03d}",
                    code=f"class Scene_{i:03d}(Scene):\n    def construct(self):\n        pass",
                    imports=["from manim import *"],
                    class_name=f"Scene_{i:03d}",
                    narration=f"This is scene {i}",
                    duration_estimate=30.0 + i * 10,
                    generation_timestamp=datetime(2024, 1, 15, 10, 30, i)
                ))
            
            # Create failed scenes
            failed_scene_list = []
            for i in range(successful_scenes, scene_count):
                failed_scene_list.append(FailedScene(
                    scene_id=f"scene_{i:03d}",
                    error=f"Generation failed for scene {i}"
                ))
            
            # Create scene processing report
            report = SceneProcessingReport(
                total_scenes=scene_count,
                successful_scenes=successful_scenes,
                failed_scenes=failed_scene_list,
                output_directory=temp_output_dir
            )
            
            # Create storyboard with scenes
            scenes = []
            for i in range(scene_count):
                scenes.append(Scene(
                    scene_id=f"scene_{i:03d}",
                    scene_index=i,
                    concept_id=f"concept_{i}",
                    narration=f"This is scene {i}",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[]
                    ),
                    duration_estimate=30.0 + i * 10
                ))
            
            storyboard = StoryboardJSON(
                pdf_metadata=PDFMetadataStoryboard(
                    filename=f"{pdf_filename}.pdf",
                    page_count=5
                ),
                configuration=Configuration(
                    verbosity="medium",
                    depth="intermediate",
                    audience="undergraduate"
                ),
                concept_hierarchy=[],
                scenes=scenes
            )
            
            # Create input
            timestamp = datetime(2024, 1, 15, 10, 30, 0)
            input_data = PersistenceInput(
                scene_processing_report=report,
                animation_code_objects=animation_codes,
                storyboard=storyboard,
                pdf_filename=pdf_filename,
                timestamp=timestamp
            )
            
            # Execute multiple times and verify idempotency
            outputs = []
            file_contents = []
            
            for retry in range(3):  # Test 3 retries
                output = agent.execute(input_data)
                outputs.append(output)
                
                # Read all file contents
                output_dir = os.path.join(temp_output_dir, pdf_filename, "2024-01-15T10-30-00")
                current_files = {}
                
                for file_info in output.manifest.files:
                    file_path = os.path.join(output_dir, file_info.path)
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            current_files[file_info.path] = f.read()
                
                file_contents.append(current_files)
            
            # Verify all outputs are identical
            for i in range(1, len(outputs)):
                # Check manifest properties
                assert outputs[0].manifest.total_scenes == outputs[i].manifest.total_scenes
                assert outputs[0].manifest.successful_scenes == outputs[i].manifest.successful_scenes
                assert outputs[0].manifest.status == outputs[i].manifest.status
                assert len(outputs[0].manifest.files) == len(outputs[i].manifest.files)
                
                # Check file contents are identical
                assert set(file_contents[0].keys()) == set(file_contents[i].keys())
                for filename in file_contents[0].keys():
                    assert file_contents[0][filename] == file_contents[i][filename], \
                        f"File {filename} differs between retry 0 and retry {i}"
        
        finally:
            # Cleanup temporary directory
            import shutil
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
    
    def test_cleanup_handles_subdirectories(self, temp_output_dir):
        """Test that cleanup properly handles subdirectories."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create output directory with files and subdirectories
        output_dir = os.path.join(temp_output_dir, "test_cleanup")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create some files
        with open(os.path.join(output_dir, "file1.txt"), 'w') as f:
            f.write("content1")
        with open(os.path.join(output_dir, "file2.py"), 'w') as f:
            f.write("content2")
        
        # Create pipeline.log (should be preserved)
        with open(os.path.join(output_dir, "pipeline.log"), 'w') as f:
            f.write("log content")
        
        # Create subdirectory with files
        subdir = os.path.join(output_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "subfile.txt"), 'w') as f:
            f.write("subcontent")
        
        # Verify initial state
        assert os.path.exists(os.path.join(output_dir, "file1.txt"))
        assert os.path.exists(os.path.join(output_dir, "file2.py"))
        assert os.path.exists(os.path.join(output_dir, "pipeline.log"))
        assert os.path.exists(subdir)
        assert os.path.exists(os.path.join(subdir, "subfile.txt"))
        
        # Run cleanup
        agent._cleanup_incomplete_files(output_dir)
        
        # Verify cleanup results
        assert not os.path.exists(os.path.join(output_dir, "file1.txt"))
        assert not os.path.exists(os.path.join(output_dir, "file2.py"))
        assert os.path.exists(os.path.join(output_dir, "pipeline.log"))  # Should be preserved
        assert not os.path.exists(subdir)  # Subdirectory should be removed
    
    def test_atomic_write_handles_failure_cleanup(self, temp_output_dir):
        """Test that atomic write cleans up temp files on failure."""
        agent = PersistenceAgent(base_output_dir=temp_output_dir)
        
        # Create a scenario where write might fail
        file_path = os.path.join(temp_output_dir, "test_file.txt")
        
        # Mock a failure during the rename operation
        original_replace = os.replace
        
        def failing_replace(src, dst):
            # Let the temp file be created, but fail on rename
            raise OSError("Simulated failure")
        
        # Count temp files before
        temp_files_before = list(Path(temp_output_dir).glob("*.tmp"))
        
        try:
            # Patch os.replace to simulate failure
            os.replace = failing_replace
            
            with pytest.raises(OSError):
                agent._write_file_atomically(file_path, "test content")
            
        finally:
            # Restore original function
            os.replace = original_replace
        
        # Verify no temp files are left behind
        temp_files_after = list(Path(temp_output_dir).glob("*.tmp"))
        assert len(temp_files_after) == len(temp_files_before), \
            "Temp files should be cleaned up on failure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
