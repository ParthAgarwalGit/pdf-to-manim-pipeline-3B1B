"""Unit tests for Pydantic schema models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas import (
    AnimationCodeObject,
    BoundingBox,
    Configuration,
    ConceptHierarchy,
    Diagram,
    EmphasisPoint,
    FileReference,
    MathematicalObject,
    MathExpression,
    ObjectStyle,
    OCROutput,
    OutputManifest,
    Page,
    PDFMetadata,
    Scene,
    SceneProcessingInput,
    SceneProcessingReport,
    SceneSpec,
    SharedContext,
    StoryboardJSON,
    TextBlock,
    Transformation,
    VisualIntent,
    VisualStyle,
)


class TestFileReference:
    """Tests for FileReference schema."""
    
    def test_valid_file_reference(self):
        """Test creating a valid FileReference."""
        ref = FileReference(
            file_path="s3://bucket/test.pdf",
            filename="test.pdf",
            size_bytes=1024,
            upload_timestamp=datetime.now(),
            metadata={"user": "test"}
        )
        assert ref.filename == "test.pdf"
        assert ref.size_bytes == 1024
    
    def test_empty_filename_raises_error(self):
        """Test that empty filename raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            FileReference(
                file_path="s3://bucket/test.pdf",
                filename="",
                size_bytes=1024,
                upload_timestamp=datetime.now()
            )
        assert "filename cannot be empty" in str(exc_info.value)
    
    def test_whitespace_filename_raises_error(self):
        """Test that whitespace-only filename raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            FileReference(
                file_path="s3://bucket/test.pdf",
                filename="   ",
                size_bytes=1024,
                upload_timestamp=datetime.now()
            )
        assert "filename cannot be empty" in str(exc_info.value)
    
    def test_empty_file_path_raises_error(self):
        """Test that empty file_path raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            FileReference(
                file_path="",
                filename="test.pdf",
                size_bytes=1024,
                upload_timestamp=datetime.now()
            )
        assert "file_path cannot be empty" in str(exc_info.value)
    
    def test_negative_size_raises_error(self):
        """Test that negative size raises validation error."""
        with pytest.raises(ValidationError):
            FileReference(
                file_path="s3://bucket/test.pdf",
                filename="test.pdf",
                size_bytes=-1,
                upload_timestamp=datetime.now()
            )
    
    def test_zero_size_allowed(self):
        """Test that zero size is allowed (edge case for empty files)."""
        ref = FileReference(
            file_path="s3://bucket/test.pdf",
            filename="test.pdf",
            size_bytes=0,
            upload_timestamp=datetime.now()
        )
        assert ref.size_bytes == 0
    
    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        ref = FileReference(
            file_path="s3://bucket/test.pdf",
            filename="test.pdf",
            size_bytes=1024,
            upload_timestamp=datetime.now()
        )
        assert ref.metadata == {}
    
    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            FileReference(
                file_path="s3://bucket/test.pdf",
                filename="test.pdf",
                size_bytes=1024
                # Missing upload_timestamp
            )
        assert "upload_timestamp" in str(exc_info.value)
    
    def test_type_coercion_size_bytes(self):
        """Test that size_bytes coerces string to int."""
        ref = FileReference(
            file_path="s3://bucket/test.pdf",
            filename="test.pdf",
            size_bytes="1024",  # String instead of int
            upload_timestamp=datetime.now()
        )
        assert ref.size_bytes == 1024
        assert isinstance(ref.size_bytes, int)


class TestOCROutput:
    """Tests for OCR output schemas."""
    
    def test_valid_bounding_box(self):
        """Test creating a valid BoundingBox."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        assert bbox.x == 10.0
        assert bbox.width == 100.0
    
    def test_negative_width_raises_error(self):
        """Test that negative width raises validation error."""
        with pytest.raises(ValidationError):
            BoundingBox(x=10.0, y=20.0, width=-100.0, height=50.0)
    
    def test_negative_height_raises_error(self):
        """Test that negative height raises validation error."""
        with pytest.raises(ValidationError):
            BoundingBox(x=10.0, y=20.0, width=100.0, height=-50.0)
    
    def test_zero_dimensions_allowed(self):
        """Test that zero width/height is allowed (edge case)."""
        bbox = BoundingBox(x=10.0, y=20.0, width=0.0, height=0.0)
        assert bbox.width == 0.0
        assert bbox.height == 0.0
    
    def test_type_coercion_bounding_box(self):
        """Test that BoundingBox coerces int to float."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert isinstance(bbox.x, float)
        assert isinstance(bbox.width, float)
    
    def test_valid_text_block(self):
        """Test creating a valid TextBlock."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        block = TextBlock(
            text="Sample text",
            bounding_box=bbox,
            reading_order=0,
            confidence=0.95
        )
        assert block.text == "Sample text"
        assert block.confidence == 0.95
    
    def test_text_block_default_confidence(self):
        """Test that TextBlock confidence defaults to 1.0."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        block = TextBlock(
            text="Sample text",
            bounding_box=bbox,
            reading_order=0
        )
        assert block.confidence == 1.0
    
    def test_text_block_confidence_out_of_range(self):
        """Test that confidence outside [0, 1] raises error."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        with pytest.raises(ValidationError):
            TextBlock(
                text="Sample text",
                bounding_box=bbox,
                reading_order=0,
                confidence=1.5
            )
    
    def test_text_block_negative_reading_order(self):
        """Test that negative reading_order raises error."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        with pytest.raises(ValidationError):
            TextBlock(
                text="Sample text",
                bounding_box=bbox,
                reading_order=-1
            )
    
    def test_valid_math_expression(self):
        """Test creating a valid MathExpression."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        expr = MathExpression(
            latex="\\frac{1}{2}",
            bounding_box=bbox,
            context="half of something",
            confidence=0.9
        )
        assert expr.latex == "\\frac{1}{2}"
        assert expr.extraction_failed is False
    
    def test_math_expression_default_context(self):
        """Test that MathExpression context defaults to empty string."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        expr = MathExpression(
            latex="\\frac{1}{2}",
            bounding_box=bbox
        )
        assert expr.context == ""
    
    def test_math_expression_default_extraction_failed(self):
        """Test that extraction_failed defaults to False."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        expr = MathExpression(
            latex="\\frac{1}{2}",
            bounding_box=bbox
        )
        assert expr.extraction_failed is False
    
    def test_diagram_type_validation(self):
        """Test that invalid diagram type raises error."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        with pytest.raises(ValidationError) as exc_info:
            Diagram(
                diagram_id="diag_001",
                bounding_box=bbox,
                diagram_type="invalid_type",
                description="A diagram"
            )
        assert "diagram_type must be one of" in str(exc_info.value)
    
    def test_diagram_valid_types(self):
        """Test that all valid diagram types are accepted."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        valid_types = ['graph', 'geometric', 'flowchart', 'plot', 'unknown']
        for dtype in valid_types:
            diagram = Diagram(
                diagram_id="diag_001",
                bounding_box=bbox,
                diagram_type=dtype,
                description="A diagram"
            )
            assert diagram.diagram_type == dtype
    
    def test_diagram_default_visual_elements(self):
        """Test that visual_elements defaults to empty list."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        diagram = Diagram(
            diagram_id="diag_001",
            bounding_box=bbox,
            diagram_type="graph",
            description="A diagram"
        )
        assert diagram.visual_elements == []
    
    def test_diagram_default_confidence(self):
        """Test that Diagram confidence defaults to 1.0."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        diagram = Diagram(
            diagram_id="diag_001",
            bounding_box=bbox,
            diagram_type="graph",
            description="A diagram"
        )
        assert diagram.confidence == 1.0
    
    def test_page_default_fields(self):
        """Test that Page optional fields default to empty lists."""
        page = Page(page_number=1)
        assert page.text_blocks == []
        assert page.math_expressions == []
        assert page.diagrams == []
    
    def test_page_invalid_page_number(self):
        """Test that page_number < 1 raises error."""
        with pytest.raises(ValidationError):
            Page(page_number=0)
    
    def test_pdf_metadata_negative_file_size(self):
        """Test that negative file_size_bytes raises error."""
        with pytest.raises(ValidationError):
            PDFMetadata(
                filename="test.pdf",
                page_count=1,
                file_size_bytes=-1
            )
    
    def test_pdf_metadata_invalid_page_count(self):
        """Test that page_count < 1 raises error."""
        with pytest.raises(ValidationError):
            PDFMetadata(
                filename="test.pdf",
                page_count=0,
                file_size_bytes=1024
            )
    
    def test_valid_ocr_output(self):
        """Test creating a valid OCROutput."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        text_block = TextBlock(
            text="Sample text",
            bounding_box=bbox,
            reading_order=0
        )
        page = Page(page_number=1, text_blocks=[text_block])
        metadata = PDFMetadata(
            filename="test.pdf",
            page_count=1,
            file_size_bytes=1024
        )
        ocr = OCROutput(pdf_metadata=metadata, pages=[page])
        assert len(ocr.pages) == 1
        assert ocr.pdf_metadata.filename == "test.pdf"
    
    def test_empty_pages_raises_error(self):
        """Test that empty pages list raises validation error."""
        metadata = PDFMetadata(
            filename="test.pdf",
            page_count=1,
            file_size_bytes=1024
        )
        with pytest.raises(ValidationError) as exc_info:
            OCROutput(pdf_metadata=metadata, pages=[])
        assert "pages list cannot be empty" in str(exc_info.value)


class TestStoryboard:
    """Tests for Storyboard schemas."""
    
    def test_valid_configuration(self):
        """Test creating a valid Configuration."""
        config = Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
        assert config.verbosity == "medium"
        assert config.depth == "intermediate"
    
    def test_invalid_verbosity_raises_error(self):
        """Test that invalid verbosity raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Configuration(
                verbosity="invalid",
                depth="intermediate",
                audience="undergraduate"
            )
        assert "verbosity must be one of" in str(exc_info.value)
    
    def test_invalid_depth_raises_error(self):
        """Test that invalid depth raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Configuration(
                verbosity="medium",
                depth="invalid",
                audience="undergraduate"
            )
        assert "depth must be one of" in str(exc_info.value)
    
    def test_invalid_audience_raises_error(self):
        """Test that invalid audience raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Configuration(
                verbosity="medium",
                depth="intermediate",
                audience="invalid"
            )
        assert "audience must be one of" in str(exc_info.value)
    
    def test_all_valid_verbosity_values(self):
        """Test all valid verbosity values are accepted."""
        for verbosity in ['low', 'medium', 'high']:
            config = Configuration(
                verbosity=verbosity,
                depth="intermediate",
                audience="undergraduate"
            )
            assert config.verbosity == verbosity
    
    def test_all_valid_depth_values(self):
        """Test all valid depth values are accepted."""
        for depth in ['introductory', 'intermediate', 'advanced']:
            config = Configuration(
                verbosity="medium",
                depth=depth,
                audience="undergraduate"
            )
            assert config.depth == depth
    
    def test_all_valid_audience_values(self):
        """Test all valid audience values are accepted."""
        for audience in ['high-school', 'undergraduate', 'graduate']:
            config = Configuration(
                verbosity="medium",
                depth="intermediate",
                audience=audience
            )
            assert config.audience == audience
    
    def test_concept_hierarchy_default_dependencies(self):
        """Test that ConceptHierarchy dependencies defaults to empty list."""
        concept = ConceptHierarchy(
            concept_id="concept_001",
            concept_name="Test Concept"
        )
        assert concept.dependencies == []
    
    def test_object_style_all_optional(self):
        """Test that ObjectStyle fields are all optional."""
        style = ObjectStyle()
        assert style.color is None
        assert style.position is None
    
    def test_mathematical_object_type_validation(self):
        """Test that invalid object_type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            MathematicalObject(
                object_type="invalid",
                content="test"
            )
        assert "object_type must be one of" in str(exc_info.value)
    
    def test_mathematical_object_valid_types(self):
        """Test all valid object types are accepted."""
        valid_types = ['equation', 'graph', 'geometric_shape', 'axes', 'text', 'pi_creature']
        for obj_type in valid_types:
            obj = MathematicalObject(
                object_type=obj_type,
                content="test content"
            )
            assert obj.object_type == obj_type

    def test_mathematical_object_aliases_normalize_to_canonical_types(self):
        """Test common LLM aliases normalize into the canonical schema."""
        obj = MathematicalObject(
            object_type="pi_character",
            content="Teacher beat",
            render_hint="pi_creature_teacher",
        )
        assert obj.object_type == "pi_creature"
        assert obj.render_hint == "PiCreatureTeacher"

    def test_engine_contract_normalizes_broader_render_hints(self):
        """Broader engine-backed render hints should normalize cleanly."""
        obj = MathematicalObject(
            object_id="axes_main",
            object_role="shared_axes",
            object_type="axes",
            render_hint="complex_plane",
            content="main coordinate frame",
            style=ObjectStyle(),
        )
        assert obj.render_hint == "ComplexPlane"
    
    def test_mathematical_object_default_style(self):
        """Test that MathematicalObject style defaults to empty ObjectStyle."""
        obj = MathematicalObject(
            object_type="equation",
            content="x^2"
        )
        assert isinstance(obj.style, ObjectStyle)
        assert obj.style.color is None
    
    def test_transformation_type_validation(self):
        """Test that invalid transformation_type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Transformation(
                transformation_type="invalid",
                target_object="obj_1",
                timing=0.5
            )
        assert "transformation_type must be one of" in str(exc_info.value)
    
    def test_transformation_valid_types(self):
        """Test all valid transformation types are accepted."""
        valid_types = ['morph', 'highlight', 'move', 'fade_in', 'fade_out', 'scale']
        for trans_type in valid_types:
            trans = Transformation(
                transformation_type=trans_type,
                target_object="obj_1",
                timing=0.5
            )
            assert trans.transformation_type == trans_type
    
    def test_transformation_negative_timing_raises_error(self):
        """Test that negative timing raises error."""
        with pytest.raises(ValidationError):
            Transformation(
                transformation_type="fade_in",
                target_object="obj_1",
                timing=-0.5
            )
    
    def test_transformation_default_parameters(self):
        """Test that Transformation parameters defaults to empty dict."""
        trans = Transformation(
            transformation_type="fade_in",
            target_object="obj_1",
            timing=0.5
        )
        assert trans.parameters == {}

    def test_transformation_accepts_engine_animation_aliases_but_normalizes_family(self):
        """Engine-level animation aliases should normalize to canonical families."""
        trans = Transformation(
            transformation_type="ReplacementTransform",
            source_object_id="eq_old",
            target_object_id="eq_new",
            continuity_action="transform",
            beat_id="beat_001",
            timing=1.0,
        )
        assert trans.transformation_type == "morph"
    
    def test_emphasis_point_negative_timestamp(self):
        """Test that negative timestamp raises error."""
        with pytest.raises(ValidationError):
            EmphasisPoint(
                timestamp=-1.0,
                description="Test emphasis"
            )
    
    def test_valid_visual_intent(self):
        """Test creating a valid VisualIntent."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        assert intent.emphasis_points == []
    
    def test_visual_intent_default_emphasis_points(self):
        """Test that emphasis_points defaults to empty list."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        assert intent.emphasis_points == []
    
    def test_valid_scene(self):
        """Test creating a valid Scene."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        scene = Scene(
            scene_id="scene_001",
            scene_index=0,
            concept_id="concept_001",
            narration="Test narration",
            visual_intent=intent,
            duration_estimate=30.0
        )
        assert scene.scene_id == "scene_001"
        assert scene.duration_estimate == 30.0
    
    def test_scene_default_dependencies(self):
        """Test that Scene dependencies defaults to empty list."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        scene = Scene(
            scene_id="scene_001",
            scene_index=0,
            concept_id="concept_001",
            narration="Test narration",
            visual_intent=intent,
            duration_estimate=30.0
        )
        assert scene.dependencies == []
    
    def test_scene_optional_difficulty_level(self):
        """Test that difficulty_level is optional."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        scene = Scene(
            scene_id="scene_001",
            scene_index=0,
            concept_id="concept_001",
            narration="Test narration",
            visual_intent=intent,
            duration_estimate=30.0
        )
        assert scene.difficulty_level is None
    
    def test_empty_scene_id_raises_error(self):
        """Test that empty scene_id raises error."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        with pytest.raises(ValidationError) as exc_info:
            Scene(
                scene_id="",
                scene_index=0,
                concept_id="concept_001",
                narration="Test narration",
                visual_intent=intent,
                duration_estimate=30.0
            )
        assert "scene_id cannot be empty" in str(exc_info.value)
    
    def test_negative_scene_index_raises_error(self):
        """Test that negative scene_index raises error."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        with pytest.raises(ValidationError):
            Scene(
                scene_id="scene_001",
                scene_index=-1,
                concept_id="concept_001",
                narration="Test narration",
                visual_intent=intent,
                duration_estimate=30.0
            )
    
    def test_zero_duration_raises_error(self):
        """Test that zero or negative duration raises error."""
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        with pytest.raises(ValidationError):
            Scene(
                scene_id="scene_001",
                scene_index=0,
                concept_id="concept_001",
                narration="Test narration",
                visual_intent=intent,
                duration_estimate=0.0
            )
    
    def test_storyboard_default_concept_hierarchy(self):
        """Test that StoryboardJSON concept_hierarchy defaults to empty list."""
        from src.schemas.storyboard import PDFMetadataStoryboard
        
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        scene = Scene(
            scene_id="scene_001",
            scene_index=0,
            concept_id="concept_001",
            narration="Test narration",
            visual_intent=intent,
            duration_estimate=30.0
        )
        metadata = PDFMetadataStoryboard(filename="test.pdf", page_count=1)
        config = Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
        storyboard = StoryboardJSON(
            pdf_metadata=metadata,
            configuration=config,
            scenes=[scene]
        )
        assert storyboard.concept_hierarchy == []
    
    def test_empty_scenes_raises_error(self):
        """Test that empty scenes list raises error."""
        from src.schemas.storyboard import PDFMetadataStoryboard
        
        metadata = PDFMetadataStoryboard(filename="test.pdf", page_count=1)
        config = Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
        with pytest.raises(ValidationError) as exc_info:
            StoryboardJSON(
                pdf_metadata=metadata,
                configuration=config,
                scenes=[]
            )
        assert "scenes list cannot be empty" in str(exc_info.value)
    
    def test_duplicate_scene_ids_raises_error(self):
        """Test that duplicate scene_ids raise validation error."""
        from src.schemas.storyboard import PDFMetadataStoryboard
        
        intent = VisualIntent(
            mathematical_objects=[],
            transformations=[]
        )
        scene1 = Scene(
            scene_id="scene_001",
            scene_index=0,
            concept_id="concept_001",
            narration="Test narration 1",
            visual_intent=intent,
            duration_estimate=30.0
        )
        scene2 = Scene(
            scene_id="scene_001",  # Duplicate ID
            scene_index=1,
            concept_id="concept_002",
            narration="Test narration 2",
            visual_intent=intent,
            duration_estimate=30.0
        )
        
        metadata = PDFMetadataStoryboard(filename="test.pdf", page_count=1)
        config = Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            StoryboardJSON(
                pdf_metadata=metadata,
                configuration=config,
                scenes=[scene1, scene2]
            )
        assert "Duplicate scene_ids found" in str(exc_info.value)


class TestAnimationCode:
    """Tests for AnimationCodeObject schema."""
    
    def test_valid_animation_code_object(self):
        """Test creating a valid AnimationCodeObject."""
        code_obj = AnimationCodeObject(
            scene_id="scene_001",
            code="class Scene_001(Scene):\n    pass",
            imports=["from manim import *"],
            class_name="Scene_001",
            narration="Test narration",
            duration_estimate=30.0,
            generation_timestamp=datetime.now()
        )
        assert code_obj.scene_id == "scene_001"
        assert code_obj.class_name == "Scene_001"
    
    def test_empty_scene_id_raises_error(self):
        """Test that empty scene_id raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AnimationCodeObject(
                scene_id="",
                code="class Scene_001(Scene):\n    pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=30.0,
                generation_timestamp=datetime.now()
            )
        assert "scene_id cannot be empty" in str(exc_info.value)
    
    def test_empty_code_raises_error(self):
        """Test that empty code raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AnimationCodeObject(
                scene_id="scene_001",
                code="",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=30.0,
                generation_timestamp=datetime.now()
            )
        assert "code cannot be empty" in str(exc_info.value)
    
    def test_whitespace_code_raises_error(self):
        """Test that whitespace-only code raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AnimationCodeObject(
                scene_id="scene_001",
                code="   \n   ",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=30.0,
                generation_timestamp=datetime.now()
            )
        assert "code cannot be empty" in str(exc_info.value)
    
    def test_invalid_class_name_raises_error(self):
        """Test that invalid Python identifier raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AnimationCodeObject(
                scene_id="scene_001",
                code="class Scene_001(Scene):\n    pass",
                imports=["from manim import *"],
                class_name="123Invalid",  # Invalid identifier
                narration="Test narration",
                duration_estimate=30.0,
                generation_timestamp=datetime.now()
            )
        assert "must be a valid Python identifier" in str(exc_info.value)
    
    def test_class_name_with_spaces_raises_error(self):
        """Test that class_name with spaces raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AnimationCodeObject(
                scene_id="scene_001",
                code="class Scene_001(Scene):\n    pass",
                imports=["from manim import *"],
                class_name="Scene 001",  # Spaces not allowed
                narration="Test narration",
                duration_estimate=30.0,
                generation_timestamp=datetime.now()
            )
        assert "must be a valid Python identifier" in str(exc_info.value)
    
    def test_empty_class_name_raises_error(self):
        """Test that empty class_name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AnimationCodeObject(
                scene_id="scene_001",
                code="class Scene_001(Scene):\n    pass",
                imports=["from manim import *"],
                class_name="",
                narration="Test narration",
                duration_estimate=30.0,
                generation_timestamp=datetime.now()
            )
        assert "class_name cannot be empty" in str(exc_info.value)
    
    def test_empty_imports_raises_error(self):
        """Test that empty imports list raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AnimationCodeObject(
                scene_id="scene_001",
                code="class Scene_001(Scene):\n    pass",
                imports=[],  # Empty imports
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=30.0,
                generation_timestamp=datetime.now()
            )
        assert "imports list cannot be empty" in str(exc_info.value)
    
    def test_zero_duration_raises_error(self):
        """Test that zero or negative duration raises error."""
        with pytest.raises(ValidationError):
            AnimationCodeObject(
                scene_id="scene_001",
                code="class Scene_001(Scene):\n    pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=0.0,
                generation_timestamp=datetime.now()
            )
    
    def test_negative_duration_raises_error(self):
        """Test that negative duration raises error."""
        with pytest.raises(ValidationError):
            AnimationCodeObject(
                scene_id="scene_001",
                code="class Scene_001(Scene):\n    pass",
                imports=["from manim import *"],
                class_name="Scene_001",
                narration="Test narration",
                duration_estimate=-5.0,
                generation_timestamp=datetime.now()
            )


class TestSceneProcessing:
    """Tests for scene processing schemas."""
    
    def test_visual_style_defaults(self):
        """Test that VisualStyle fields have correct defaults."""
        style = VisualStyle()
        assert style.colors == {}
        assert style.font_size == 24
        assert style.animation_speed == 1.0
    
    def test_visual_style_font_size_validation(self):
        """Test that font_size is validated within range."""
        with pytest.raises(ValidationError):
            VisualStyle(font_size=7)  # Too small
        
        with pytest.raises(ValidationError):
            VisualStyle(font_size=73)  # Too large
        
        # Valid values
        style = VisualStyle(font_size=8)
        assert style.font_size == 8
        style = VisualStyle(font_size=72)
        assert style.font_size == 72
    
    def test_visual_style_animation_speed_validation(self):
        """Test that animation_speed must be positive."""
        with pytest.raises(ValidationError):
            VisualStyle(animation_speed=0.0)
        
        with pytest.raises(ValidationError):
            VisualStyle(animation_speed=-1.0)
        
        # Valid value
        style = VisualStyle(animation_speed=0.5)
        assert style.animation_speed == 0.5
    
    def test_shared_context_defaults(self):
        """Test that SharedContext fields default to empty dicts."""
        context = SharedContext()
        assert context.concept_definitions == {}
        assert context.variable_bindings == {}
        assert isinstance(context.visual_style, VisualStyle)

    # ── NarrativeContext (Task 1 — additive) ─────────────────────────

    def test_narrative_context_defaults_are_empty(self):
        from src.schemas.scene_processing import NarrativeContext
        nc = NarrativeContext()
        assert nc.prior_scene_summaries == []
        assert nc.concepts_introduced == []
        assert nc.running_storyline == ""
        assert nc.live_objects_on_screen == {}
        assert nc.prior_equations == []  # F8: track prior equations for visual-layer dedup

    def test_narrative_context_appends_preserve_order(self):
        from src.schemas.scene_processing import NarrativeContext
        nc = NarrativeContext()
        nc.prior_scene_summaries.append("Introduced sine on the unit circle.")
        nc.concepts_introduced.append("sine")
        nc.live_objects_on_screen["unit_circle"] = "Circle(radius=2)"
        nc.prior_equations.append(r"\sin(\theta)")
        assert nc.prior_scene_summaries == ["Introduced sine on the unit circle."]
        assert "sine" in nc.concepts_introduced
        assert nc.live_objects_on_screen["unit_circle"] == "Circle(radius=2)"
        assert nc.prior_equations == [r"\sin(\theta)"]

    def test_shared_context_carries_narrative_context_default(self):
        from src.schemas.scene_processing import NarrativeContext
        ctx = SharedContext()
        assert isinstance(ctx.narrative, NarrativeContext)
        assert ctx.narrative.concepts_introduced == []
        assert ctx.narrative.prior_equations == []

    def test_narrative_context_roundtrips_through_pydantic_dump(self):
        """NarrativeContext must survive model_dump → model_validate roundtrip
        so the dry_run / persistence paths don't drop it on disk."""
        from src.schemas.scene_processing import NarrativeContext
        nc = NarrativeContext(
            prior_scene_summaries=["a", "b"],
            concepts_introduced=["x", "y"],
            running_storyline="brief storyline.",
            live_objects_on_screen={"obj_0": "Circle"},
            prior_equations=[r"a^2+b^2=c^2"],
        )
        dumped = nc.model_dump()
        restored = NarrativeContext.model_validate(dumped)
        assert restored == nc


class TestVideoScript:
    """Tests for the unified-script schemas (Task 2)."""

    def test_beat_validates_required_fields(self):
        from src.schemas.video_script import Beat
        b = Beat(
            beat_id="beat_001",
            narration="A sine wave traces the y-coordinate of a point on a unit circle.",
            visual_hint="Show a unit circle with a rotating radius and a graph next to it.",
            introduces_concepts=["sine", "unit_circle"],
            references_prior_beats=[],
        )
        assert b.beat_id == "beat_001"
        assert "sine" in b.introduces_concepts

    def test_video_script_concept_ledger_dedupes_and_preserves_first_intro_order(self):
        from src.schemas.video_script import VideoScript, Beat
        script = VideoScript(beats=[
            Beat(beat_id="beat_001", narration="...", visual_hint="...",
                 introduces_concepts=["sine"]),
            Beat(beat_id="beat_002", narration="...", visual_hint="...",
                 introduces_concepts=["cosine", "sine"]),  # sine repeated → ignored
        ])
        assert script.concept_ledger() == ["sine", "cosine"]

    def test_video_script_rejects_duplicate_beat_ids(self):
        from pydantic import ValidationError as _VE
        from src.schemas.video_script import VideoScript, Beat
        with pytest.raises(_VE):
            VideoScript(beats=[
                Beat(beat_id="beat_001", narration="a", visual_hint="x"),
                Beat(beat_id="beat_001", narration="b", visual_hint="y"),
            ])

    def test_video_script_drops_unknown_back_references(self):
        """F7 mitigation: invalid references_prior_beats must be dropped, not raised."""
        import logging
        from src.schemas.video_script import VideoScript, Beat
        with self._capture_warnings("src.schemas.video_script") as records:
            script = VideoScript(beats=[
                Beat(beat_id="beat_001", narration="a", visual_hint="x",
                     references_prior_beats=["beat_999"]),
            ])
        assert script.beats[0].references_prior_beats == []
        assert any("beat_999" in r.getMessage() for r in records)

    def test_video_script_drops_forward_back_references(self):
        """A beat cannot reference a later beat — drop those too."""
        from src.schemas.video_script import VideoScript, Beat
        with self._capture_warnings("src.schemas.video_script"):
            script = VideoScript(beats=[
                Beat(beat_id="beat_001", narration="a", visual_hint="x",
                     references_prior_beats=["beat_002"]),  # forward — invalid
                Beat(beat_id="beat_002", narration="b", visual_hint="y"),
            ])
        assert script.beats[0].references_prior_beats == []

    @staticmethod
    def _capture_warnings(logger_name):
        """Context manager that captures WARNING records from a logger."""
        import logging

        class _LogCapture:
            def __enter__(self_inner):
                self_inner.records = []
                self_inner.handler = logging.Handler(level=logging.WARNING)
                self_inner.handler.emit = lambda r: self_inner.records.append(r)
                self_inner.logger = logging.getLogger(logger_name)
                self_inner._old_level = self_inner.logger.level
                self_inner.logger.setLevel(logging.WARNING)
                self_inner.logger.addHandler(self_inner.handler)
                return self_inner.records

            def __exit__(self_inner, *a):
                self_inner.logger.removeHandler(self_inner.handler)
                self_inner.logger.setLevel(self_inner._old_level)
                return False

        return _LogCapture()
    
    def test_scene_spec_empty_scene_id_raises_error(self):
        """Test that empty scene_id raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SceneSpec(
                scene_id="",
                narration="Test narration",
                visual_intent={},
                duration_estimate=30.0
            )
        assert "scene_id cannot be empty" in str(exc_info.value)
    
    def test_scene_spec_zero_duration_raises_error(self):
        """Test that zero or negative duration raises error."""
        with pytest.raises(ValidationError):
            SceneSpec(
                scene_id="scene_001",
                narration="Test narration",
                visual_intent={},
                duration_estimate=0.0
            )
    
    def test_valid_scene_processing_input(self):
        """Test creating a valid SceneProcessingInput."""
        scene_spec = SceneSpec(
            scene_id="scene_001",
            narration="Test narration",
            visual_intent={
                "mathematical_objects": [],
                "transformations": []
            },
            duration_estimate=30.0
        )
        context = SharedContext()
        input_obj = SceneProcessingInput(
            scene_spec=scene_spec,
            context=context,
            scene_index=0
        )
        assert input_obj.scene_index == 0
        assert input_obj.scene_spec.scene_id == "scene_001"
    
    def test_scene_processing_input_negative_index_raises_error(self):
        """Test that negative scene_index raises error."""
        scene_spec = SceneSpec(
            scene_id="scene_001",
            narration="Test narration",
            visual_intent={
                "mathematical_objects": [],
                "transformations": []
            },
            duration_estimate=30.0
        )
        context = SharedContext()
        with pytest.raises(ValidationError):
            SceneProcessingInput(
                scene_spec=scene_spec,
                context=context,
                scene_index=-1
            )
    
    def test_failed_scene_default_fields(self):
        """Test that FailedScene has no default fields."""
        from src.schemas.scene_processing import FailedScene
        
        failed = FailedScene(
            scene_id="scene_001",
            error="Test error"
        )
        assert failed.scene_id == "scene_001"
        assert failed.error == "Test error"
    
    def test_scene_processing_report_defaults(self):
        """Test that SceneProcessingReport failed_scenes defaults to empty list."""
        report = SceneProcessingReport(
            total_scenes=10,
            successful_scenes=10,
            output_directory="/path/to/output"
        )
        assert report.failed_scenes == []
    
    def test_scene_processing_report_success_rate(self):
        """Test scene_success_rate property calculation."""
        report = SceneProcessingReport(
            total_scenes=10,
            successful_scenes=8,
            failed_scenes=[],
            output_directory="/path/to/output"
        )
        assert report.scene_success_rate == 0.8
        assert report.status == "PARTIAL_SUCCESS"
    
    def test_scene_processing_report_success_rate_zero_total(self):
        """Test scene_success_rate when total_scenes is zero."""
        report = SceneProcessingReport(
            total_scenes=0,
            successful_scenes=0,
            failed_scenes=[],
            output_directory="/path/to/output"
        )
        assert report.scene_success_rate == 0.0
    
    def test_scene_processing_report_status_success(self):
        """Test status property returns SUCCESS when all scenes succeed."""
        report = SceneProcessingReport(
            total_scenes=10,
            successful_scenes=10,
            failed_scenes=[],
            output_directory="/path/to/output"
        )
        assert report.status == "SUCCESS"
    
    def test_scene_processing_report_status_failure(self):
        """Test status property returns FAILURE when all scenes fail."""
        from src.schemas.scene_processing import FailedScene
        
        report = SceneProcessingReport(
            total_scenes=10,
            successful_scenes=0,
            failed_scenes=[
                FailedScene(scene_id=f"scene_{i}", error="Test error")
                for i in range(10)
            ],
            output_directory="/path/to/output"
        )
        assert report.status == "FAILURE"
    
    def test_scene_processing_report_status_critical_failure(self):
        """Test status property returns CRITICAL_FAILURE when >50% fail."""
        from src.schemas.scene_processing import FailedScene
        
        report = SceneProcessingReport(
            total_scenes=10,
            successful_scenes=4,
            failed_scenes=[
                FailedScene(scene_id=f"scene_{i}", error="Test error")
                for i in range(6)
            ],
            output_directory="/path/to/output"
        )
        assert report.status == "CRITICAL_FAILURE"
    
    def test_scene_processing_report_successful_exceeds_total(self):
        """Test that successful_scenes > total_scenes raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SceneProcessingReport(
                total_scenes=10,
                successful_scenes=11,
                failed_scenes=[],
                output_directory="/path/to/output"
            )
        assert "cannot exceed total_scenes" in str(exc_info.value)


class TestOutputManifest:
    """Tests for OutputManifest schema."""
    
    def test_output_file_type_validation(self):
        """Test that OutputFile type is validated."""
        from src.schemas.output_manifest import OutputFile
        
        with pytest.raises(ValidationError) as exc_info:
            OutputFile(
                path="test.txt",
                size_bytes=1024,
                type="invalid_type"
            )
        assert "type must be one of" in str(exc_info.value)
    
    def test_output_file_valid_types(self):
        """Test all valid file types are accepted."""
        from src.schemas.output_manifest import OutputFile
        
        valid_types = [
            'animation_code',
            'narration_script',
            'storyboard',
            'manifest',
            'report',
            'ocr_output',
            'log'
        ]
        for file_type in valid_types:
            file_obj = OutputFile(
                path="test.txt",
                size_bytes=1024,
                type=file_type
            )
            assert file_obj.type == file_type
    
    def test_output_file_negative_size_raises_error(self):
        """Test that negative size_bytes raises error."""
        from src.schemas.output_manifest import OutputFile
        
        with pytest.raises(ValidationError):
            OutputFile(
                path="test.txt",
                size_bytes=-1,
                type="manifest"
            )
    
    def test_valid_output_manifest(self):
        """Test creating a valid OutputManifest."""
        from src.schemas.output_manifest import OutputFile
        
        manifest = OutputManifest(
            output_directory="/path/to/output",
            files=[
                OutputFile(
                    path="scene_001.py",
                    size_bytes=1024,
                    type="animation_code"
                )
            ],
            timestamp=datetime.now(),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=1,
            successful_scenes=1,
            failed_scenes=[]
        )
        assert manifest.status == "SUCCESS"
        assert len(manifest.files) == 1
    
    def test_output_manifest_status_validation(self):
        """Test that OutputManifest status is validated."""
        from src.schemas.output_manifest import OutputFile
        
        with pytest.raises(ValidationError) as exc_info:
            OutputManifest(
                output_directory="/path/to/output",
                files=[],
                timestamp=datetime.now(),
                status="INVALID_STATUS",
                scene_success_rate=1.0,
                total_scenes=1,
                successful_scenes=1,
                failed_scenes=[]
            )
        assert "status must be one of" in str(exc_info.value)
    
    def test_output_manifest_valid_statuses(self):
        """Test all valid status values are accepted."""
        from src.schemas.output_manifest import OutputFile
        
        valid_statuses = ['SUCCESS', 'PARTIAL_SUCCESS', 'FAILURE', 'CRITICAL_FAILURE']
        for status in valid_statuses:
            manifest = OutputManifest(
                output_directory="/path/to/output",
                files=[],
                timestamp=datetime.now(),
                status=status,
                scene_success_rate=0.5,
                total_scenes=10,
                successful_scenes=5,
                failed_scenes=[]
            )
            assert manifest.status == status
    
    def test_output_manifest_empty_output_directory_raises_error(self):
        """Test that empty output_directory raises error."""
        from src.schemas.output_manifest import OutputFile
        
        with pytest.raises(ValidationError) as exc_info:
            OutputManifest(
                output_directory="",
                files=[],
                timestamp=datetime.now(),
                status="SUCCESS",
                scene_success_rate=1.0,
                total_scenes=1,
                successful_scenes=1,
                failed_scenes=[]
            )
        assert "output_directory cannot be empty" in str(exc_info.value)
    
    def test_output_manifest_success_rate_out_of_range(self):
        """Test that scene_success_rate outside [0, 1] raises error."""
        from src.schemas.output_manifest import OutputFile
        
        with pytest.raises(ValidationError):
            OutputManifest(
                output_directory="/path/to/output",
                files=[],
                timestamp=datetime.now(),
                status="SUCCESS",
                scene_success_rate=1.5,
                total_scenes=1,
                successful_scenes=1,
                failed_scenes=[]
            )
    
    def test_output_manifest_successful_exceeds_total(self):
        """Test that successful_scenes > total_scenes raises error."""
        from src.schemas.output_manifest import OutputFile
        
        with pytest.raises(ValidationError) as exc_info:
            OutputManifest(
                output_directory="/path/to/output",
                files=[],
                timestamp=datetime.now(),
                status="SUCCESS",
                scene_success_rate=1.0,
                total_scenes=10,
                successful_scenes=11,
                failed_scenes=[]
            )
        assert "cannot exceed total_scenes" in str(exc_info.value)
    
    def test_inconsistent_success_rate_raises_error(self):
        """Test that inconsistent success rate raises error."""
        from src.schemas.output_manifest import OutputFile
        
        with pytest.raises(ValidationError) as exc_info:
            OutputManifest(
                output_directory="/path/to/output",
                files=[
                    OutputFile(
                        path="scene_001.py",
                        size_bytes=1024,
                        type="animation_code"
                    )
                ],
                timestamp=datetime.now(),
                status="SUCCESS",
                scene_success_rate=0.5,  # Inconsistent with 10/10
                total_scenes=10,
                successful_scenes=10,
                failed_scenes=[]
            )
        assert "scene_success_rate" in str(exc_info.value)
        assert "inconsistent" in str(exc_info.value)
    
    def test_consistent_success_rate_accepted(self):
        """Test that consistent success rate is accepted."""
        from src.schemas.output_manifest import OutputFile

        manifest = OutputManifest(
            output_directory="/path/to/output",
            files=[],
            timestamp=datetime.now(),
            status="PARTIAL_SUCCESS",
            scene_success_rate=0.7,
            total_scenes=10,
            successful_scenes=7,
            failed_scenes=[]
        )
        assert manifest.scene_success_rate == 0.7

    def test_output_manifest_default_failed_scenes(self):
        """Test that failed_scenes defaults to empty list."""
        from src.schemas.output_manifest import OutputFile

        manifest = OutputManifest(
            output_directory="/path/to/output",
            files=[],
            timestamp=datetime.now(),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=1,
            successful_scenes=1
        )
        assert manifest.failed_scenes == []

    def test_failed_scene_manifest_fields(self):
        """Test FailedSceneManifest has required fields."""
        from src.schemas.output_manifest import FailedSceneManifest

        failed = FailedSceneManifest(
            scene_id="scene_001",
            error="Test error"
        )
        assert failed.scene_id == "scene_001"
        assert failed.error == "Test error"


# ---------------------------------------------------------------------------
# Tests for 3B1B Quality Contract constants (Part 3 Task 1)
# ---------------------------------------------------------------------------

class TestTimingContract:
    """Tests for TIMING_CONTRACT, WAIT_CONTRACT, and COLOR_PALETTE constants."""

    def test_timing_contract_covers_all_animation_hints(self):
        """Every canonical animation hint should have a timing entry."""
        from src.utils.manim_engine_contract import (
            TIMING_CONTRACT, CANONICAL_ANIMATION_HINTS,
        )
        for anim in CANONICAL_ANIMATION_HINTS:
            if anim in TIMING_CONTRACT:
                lo, hi = TIMING_CONTRACT[anim]
                assert 0 < lo <= hi, f"{anim}: min {lo} > max {hi}"

    def test_timing_contract_values_are_positive_tuples(self):
        from src.utils.manim_engine_contract import TIMING_CONTRACT
        for anim, (lo, hi) in TIMING_CONTRACT.items():
            assert isinstance(lo, (int, float))
            assert isinstance(hi, (int, float))
            assert lo > 0, f"{anim} min must be positive"
            assert hi >= lo, f"{anim} max must be >= min"

    def test_wait_contract_has_required_keys(self):
        from src.utils.manim_engine_contract import WAIT_CONTRACT
        required = {
            "between_animations",
            "after_major_reveal",
            "scene_breathing_room",
            "after_transform",
        }
        assert required.issubset(set(WAIT_CONTRACT.keys()))
        for key, val in WAIT_CONTRACT.items():
            assert val > 0, f"WAIT_CONTRACT[{key}] must be positive"

    def test_color_palette_has_semantic_aliases(self):
        from src.utils.manim_engine_contract import THREE_BLUE_ONE_BROWN_PALETTE
        for alias in ("PRIMARY", "SECONDARY", "ACCENT", "WARNING", "BACKGROUND"):
            assert alias in THREE_BLUE_ONE_BROWN_PALETTE
            assert THREE_BLUE_ONE_BROWN_PALETTE[alias].startswith("#")

    def test_color_cycle_has_enough_entries(self):
        from src.utils.manim_engine_contract import COLOR_CYCLE
        assert len(COLOR_CYCLE) >= 8, "Need enough colors for variable assignment"
        for c in COLOR_CYCLE:
            assert c.startswith("#") and len(c) == 7

    def test_build_3b1b_quality_prompt_returns_string(self):
        from src.utils.manim_engine_contract import build_3b1b_quality_prompt
        prompt = build_3b1b_quality_prompt()
        assert isinstance(prompt, str)
        assert "TIMING CONSTRAINTS" in prompt
        assert "PROGRESSIVE REVEAL" in prompt
        assert "COLOR CONSISTENCY" in prompt
        assert "SMOOTH MORPHING" in prompt
        assert "ANTI-PATTERNS" in prompt
        assert "BREATHING ROOM" in prompt


class TestCostSummary:
    """CostSummary schema and its integration into OutputManifest (Bug #5)."""

    def test_cost_summary_defaults_are_empty(self):
        from src.schemas.output_manifest import CostSummary
        cs = CostSummary(total_usd=0.0)
        assert cs.total_usd == 0.0
        assert cs.by_provider == {}
        assert cs.by_model == {}
        assert cs.by_call_type == {}
        assert cs.call_count == 0

    def test_cost_summary_round_trips_breakdown(self):
        from src.schemas.output_manifest import CostSummary
        cs = CostSummary(
            total_usd=1.2345,
            by_provider={"openai": 0.8345, "anthropic": 0.4},
            by_model={"gpt-5": 0.6, "claude-sonnet-4-6": 0.6345},
            by_call_type={"chat": 1.0, "tts": 0.2345},
            call_count=12,
        )
        dumped = cs.model_dump()
        assert dumped["total_usd"] == 1.2345
        assert dumped["by_provider"]["openai"] == 0.8345
        assert dumped["call_count"] == 12

    def test_cost_summary_total_must_be_non_negative(self):
        import pytest
        from pydantic import ValidationError
        from src.schemas.output_manifest import CostSummary
        with pytest.raises(ValidationError):
            CostSummary(total_usd=-1.0)

    def test_output_manifest_cost_summary_optional(self):
        """OutputManifest must accept None for cost_summary so legacy
        manifests stay valid."""
        from datetime import datetime, timezone
        from src.schemas.output_manifest import OutputManifest, OutputFile
        manifest = OutputManifest(
            output_directory="/tmp/out",
            files=[OutputFile(path="manifest.json", size_bytes=100, type="manifest")],
            timestamp=datetime.now(timezone.utc),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=1,
            successful_scenes=1,
        )
        assert manifest.cost_summary is None

    def test_output_manifest_carries_cost_summary_when_set(self):
        from datetime import datetime, timezone
        from src.schemas.output_manifest import CostSummary, OutputManifest, OutputFile
        manifest = OutputManifest(
            output_directory="/tmp/out",
            files=[OutputFile(path="manifest.json", size_bytes=100, type="manifest")],
            timestamp=datetime.now(timezone.utc),
            status="SUCCESS",
            scene_success_rate=1.0,
            total_scenes=1,
            successful_scenes=1,
            cost_summary=CostSummary(
                total_usd=0.42,
                by_provider={"openai": 0.42},
                call_count=3,
            ),
        )
        assert manifest.cost_summary is not None
        assert manifest.cost_summary.total_usd == 0.42