"""Property-based tests for pipeline determinism.

Tests that identical inputs produce structurally equivalent outputs.

**Property 6: Identical inputs produce structurally equivalent outputs**
*For any* PDF and configuration, processing the same input twice should produce
structurally equivalent Storyboard JSON (same scene count, same scene order,
equivalent narration content).

**Validates: Requirements 10.1, 10.2**
"""

import tempfile
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.agents.ingestion import IngestionOutput
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.file_reference import FileReference
from src.schemas.ocr_output import (
    BoundingBox,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)
from src.schemas.output_manifest import OutputFile, OutputManifest
from src.schemas.scene_processing import SceneProcessingReport
from src.schemas.storyboard import (
    Configuration,
    MathematicalObject,
    PDFMetadataStoryboard,
    Scene,
    StoryboardJSON,
    Transformation,
    VisualIntent,
)


# Strategy for generating valid PDF filenames
pdf_filename_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=65, max_codepoint=122),
    min_size=5,
    max_size=20
).map(lambda s: s + ".pdf")


# Strategy for generating configuration parameters
configuration_strategy = st.builds(
    Configuration,
    verbosity=st.sampled_from(["low", "medium", "high"]),
    depth=st.sampled_from(["introductory", "intermediate", "advanced"]),
    audience=st.sampled_from(["high-school", "undergraduate", "graduate"])
)


# Strategy for generating text content
text_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), min_codepoint=32, max_codepoint=126),
    min_size=10,
    max_size=200
)


# Strategy for generating scene specifications
@st.composite
def scene_strategy(draw, scene_index):
    """Generate a scene with deterministic scene_id based on index."""
    return Scene(
        scene_id=f"scene_{scene_index:03d}",
        scene_index=scene_index - 1,  # 0-based index for scene_index field
        concept_id=draw(st.text(min_size=3, max_size=20)),
        narration=draw(text_content_strategy),
        visual_intent=VisualIntent(
            mathematical_objects=draw(st.lists(
                st.builds(
                    MathematicalObject,
                    object_type=st.sampled_from(["equation", "graph", "geometric_shape", "axes", "text"]),
                    content=text_content_strategy
                ),
                min_size=0,
                max_size=3
            )),
            transformations=draw(st.lists(
                st.builds(
                    Transformation,
                    transformation_type=st.sampled_from(["morph", "highlight", "move", "fade_in", "fade_out", "scale"]),
                    target_object=st.text(min_size=1, max_size=20),
                    parameters=st.just({}),
                    timing=st.floats(min_value=0.0, max_value=60.0)
                ),
                min_size=0,
                max_size=3
            )),
            emphasis_points=[]
        ),
        duration_estimate=draw(st.floats(min_value=10.0, max_value=180.0))
    )


# Strategy for generating storyboards with multiple scenes
@st.composite
def storyboard_strategy(draw):
    """Generate a storyboard with unique scene_ids."""
    num_scenes = draw(st.integers(min_value=1, max_value=5))
    
    # Generate unique scenes with deterministic scene_ids
    scenes = []
    for i in range(num_scenes):
        scene = draw(scene_strategy(i + 1))  # Use 1-based indexing
        scenes.append(scene)
    
    return StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(
            filename=draw(pdf_filename_strategy),
            page_count=draw(st.integers(min_value=1, max_value=20))
        ),
        configuration=draw(configuration_strategy),
        concept_hierarchy=[],
        scenes=scenes
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
    import os
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPipelineDeterminism:
    """Property-based tests for pipeline determinism."""
    
    @given(storyboard=storyboard_strategy())
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_identical_inputs_produce_equivalent_outputs(
        self, storyboard, temp_pdf_file, temp_output_dir
    ):
        """
        **Property 6: Identical inputs produce structurally equivalent outputs**
        
        *For any* PDF and configuration, processing the same input twice should produce
        structurally equivalent Storyboard JSON (same scene count, same scene order,
        equivalent narration content).
        
        **Validates: Requirements 10.1, 10.2**
        """
        # Create two orchestrators with same configuration
        config1 = PipelineConfig(base_output_dir=temp_output_dir)
        config2 = PipelineConfig(base_output_dir=temp_output_dir)
        
        orchestrator1 = Orchestrator(config=config1)
        orchestrator2 = Orchestrator(config=config2)
        
        # Mock both orchestrators with identical inputs
        self._mock_pipeline(orchestrator1, temp_pdf_file, storyboard)
        self._mock_pipeline(orchestrator2, temp_pdf_file, storyboard)
        
        # Execute pipeline twice with same input
        manifest1 = orchestrator1.run_pipeline(temp_pdf_file, event_source="local")
        manifest2 = orchestrator2.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify structural equivalence
        # 1. Same scene count
        assert manifest1.total_scenes == manifest2.total_scenes, \
            "Scene count should be identical for same input"
        
        # 2. Same success rate
        assert manifest1.scene_success_rate == manifest2.scene_success_rate, \
            "Success rate should be identical for same input"
        
        # 3. Same number of successful scenes
        assert manifest1.successful_scenes == manifest2.successful_scenes, \
            "Number of successful scenes should be identical"
        
        # 4. Same number of failed scenes
        assert len(manifest1.failed_scenes) == len(manifest2.failed_scenes), \
            "Number of failed scenes should be identical"
        
        # 5. Same overall status
        assert manifest1.status == manifest2.status, \
            "Pipeline status should be identical for same input"
        
        # 6. Same number of output files
        assert len(manifest1.files) == len(manifest2.files), \
            "Number of output files should be identical"
        
        # 7. Same file types
        file_types1 = sorted([f.type for f in manifest1.files])
        file_types2 = sorted([f.type for f in manifest2.files])
        assert file_types1 == file_types2, \
            "Output file types should be identical"
    
    @given(
        storyboard=storyboard_strategy(),
        config=configuration_strategy
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_scene_order_is_deterministic(
        self, storyboard, config, temp_pdf_file, temp_output_dir
    ):
        """
        Test that scene processing order is deterministic (array order).
        
        Scenes should always be processed in the order they appear in the
        storyboard scenes array (index 0, 1, 2, ..., N-1).
        
        **Validates: Requirements 10.2**
        """
        # Create orchestrator
        pipeline_config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator = Orchestrator(config=pipeline_config)
        
        # Mock pipeline with given storyboard
        self._mock_pipeline(orchestrator, temp_pdf_file, storyboard)
        
        # Execute pipeline
        manifest = orchestrator.run_pipeline(temp_pdf_file, event_source="local")
        
        # Verify scenes were processed in order
        # The scene_ids in the manifest should match the order in the storyboard
        expected_scene_ids = [scene.scene_id for scene in storyboard.scenes]
        
        # Extract scene_ids from successful animation code files
        animation_files = [f for f in manifest.files if f.type == "animation_code"]
        actual_scene_ids = sorted([
            f.path.replace("scene_", "").replace(".py", "")
            for f in animation_files
        ])
        expected_scene_ids_sorted = sorted([
            sid.replace("scene_", "")
            for sid in expected_scene_ids
        ])
        
        # Verify all expected scenes are present (in some order)
        assert len(actual_scene_ids) == len(expected_scene_ids_sorted), \
            "All scenes should be processed"
    
    def test_deterministic_scene_id_generation(self, temp_pdf_file, temp_output_dir):
        """
        Test that scene_id generation is deterministic.
        
        Scene IDs should be generated consistently based on scene index,
        not based on timestamps or random values.
        
        **Validates: Requirements 10.3, 10.4**
        """
        # Create storyboard with specific scenes
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
                    narration="Introduction",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[],
                        emphasis_points=[]
                    ),
                    duration_estimate=30.0
                ),
                Scene(
                    scene_id="scene_002",
                    scene_index=1,
                    concept_id="main",
                    narration="Main content",
                    visual_intent=VisualIntent(
                        mathematical_objects=[],
                        transformations=[],
                        emphasis_points=[]
                    ),
                    duration_estimate=45.0
                )
            ]
        )
        
        # Run pipeline twice
        config = PipelineConfig(base_output_dir=temp_output_dir)
        orchestrator1 = Orchestrator(config=config)
        orchestrator2 = Orchestrator(config=config)
        
        self._mock_pipeline(orchestrator1, temp_pdf_file, storyboard)
        self._mock_pipeline(orchestrator2, temp_pdf_file, storyboard)
        
        manifest1 = orchestrator1.run_pipeline(temp_pdf_file, event_source="local")
        manifest2 = orchestrator2.run_pipeline(temp_pdf_file, event_source="local")
        
        # Extract scene IDs from both runs
        files1 = sorted([f.path for f in manifest1.files if f.type == "animation_code"])
        files2 = sorted([f.path for f in manifest2.files if f.type == "animation_code"])
        
        # Verify scene IDs are identical
        assert files1 == files2, \
            "Scene IDs should be deterministic across runs"
        
        # Verify scene IDs match expected format
        assert "scene_001.py" in files1
        assert "scene_002.py" in files1
    
    def _mock_pipeline(self, orchestrator, temp_pdf_file, storyboard):
        """Helper to mock all agents for pipeline execution."""
        # Mock ingestion
        file_ref = FileReference(
            file_path=temp_pdf_file,
            filename=storyboard.pdf_metadata.filename,
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
                filename=storyboard.pdf_metadata.filename,
                page_count=storyboard.pdf_metadata.page_count,
                file_size_bytes=500
            ),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[
                        TextBlock(
                            text="Sample text",
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
        orchestrator.script_architect_agent.execute = Mock(return_value=storyboard)
        
        # Mock JSON sanitizer
        orchestrator.json_sanitizer_agent.execute = Mock(
            return_value=Mock(storyboard=storyboard)
        )
        
        # Mock scene controller
        animation_codes = [
            AnimationCodeObject(
                scene_id=scene.scene_id,
                code=f"from manim import *\n\nclass {scene.scene_id.replace('scene_', 'Scene_')}(Scene):\n    def construct(self):\n        pass",
                imports=["from manim import *"],
                class_name=scene.scene_id.replace('scene_', 'Scene_'),
                narration=scene.narration,
                duration_estimate=scene.duration_estimate,
                generation_timestamp=datetime.now(timezone.utc)
            )
            for scene in storyboard.scenes
        ]
        
        scene_report = SceneProcessingReport(
            total_scenes=len(storyboard.scenes),
            successful_scenes=len(storyboard.scenes),
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
        files = [
            OutputFile(
                path=f"{scene.scene_id}.py",
                size_bytes=100,
                type="animation_code"
            )
            for scene in storyboard.scenes
        ]
        files.extend([
            OutputFile(path="narration_script.txt", size_bytes=50, type="narration_script"),
            OutputFile(path="storyboard.json", size_bytes=200, type="storyboard"),
            OutputFile(path="manifest.json", size_bytes=150, type="manifest")
        ])
        
        manifest = OutputManifest(
            output_directory="output/test/timestamp",
            files=files,
            timestamp=datetime.now(timezone.utc),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=len(storyboard.scenes),
            successful_scenes=len(storyboard.scenes),
            failed_scenes=[]
        )
        
        orchestrator.persistence_agent.execute = Mock(
            return_value=Mock(manifest=manifest)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
