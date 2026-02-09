"""Unit tests for Script/Storyboard Architect Agent"""

import pytest
from datetime import datetime

from src.agents.base import AgentExecutionError
from src.agents.script_architect import ScriptArchitectAgent, ScriptArchitectInput
from src.schemas.ocr_output import (
    BoundingBox,
    Diagram,
    MathExpression,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)
from src.schemas.storyboard import Configuration, StoryboardJSON


class TestScriptArchitectAgent:
    """Test suite for ScriptArchitectAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a ScriptArchitectAgent instance"""
        return ScriptArchitectAgent()
    
    @pytest.fixture
    def sample_ocr_output(self):
        """Create sample OCR output for testing"""
        return OCROutput(
            pdf_metadata=PDFMetadata(
                filename="test_document.pdf",
                page_count=2,
                file_size_bytes=1024000
            ),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[
                        TextBlock(
                            text="Definition 1: A vector space is a set V with operations.",
                            bounding_box=BoundingBox(x=100, y=100, width=400, height=30),
                            reading_order=0,
                            confidence=0.95
                        ),
                        TextBlock(
                            text="Let V be a vector space over the real numbers.",
                            bounding_box=BoundingBox(x=100, y=150, width=400, height=30),
                            reading_order=1,
                            confidence=0.95
                        )
                    ],
                    math_expressions=[
                        MathExpression(
                            latex="\\mathbb{R}^n",
                            bounding_box=BoundingBox(x=150, y=200, width=50, height=20),
                            context="vector space over the real numbers",
                            confidence=0.90
                        )
                    ],
                    diagrams=[]
                ),
                Page(
                    page_number=2,
                    text_blocks=[
                        TextBlock(
                            text="Theorem 1: Every vector space has a basis.",
                            bounding_box=BoundingBox(x=100, y=100, width=400, height=30),
                            reading_order=0,
                            confidence=0.95
                        ),
                        TextBlock(
                            text="Example 1: Consider the vector space R^2.",
                            bounding_box=BoundingBox(x=100, y=150, width=400, height=30),
                            reading_order=1,
                            confidence=0.95
                        )
                    ],
                    math_expressions=[],
                    diagrams=[]
                )
            ]
        )
    
    @pytest.fixture
    def default_config(self):
        """Create default configuration"""
        return Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
    
    @pytest.fixture
    def low_verbosity_config(self):
        """Create low verbosity configuration"""
        return Configuration(
            verbosity="low",
            depth="intermediate",
            audience="undergraduate"
        )
    
    @pytest.fixture
    def high_verbosity_config(self):
        """Create high verbosity configuration"""
        return Configuration(
            verbosity="high",
            depth="advanced",
            audience="graduate"
        )
    
    @pytest.fixture
    def introductory_config(self):
        """Create introductory depth configuration"""
        return Configuration(
            verbosity="medium",
            depth="introductory",
            audience="high-school"
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert agent.retry_count == 0
        assert agent.MIN_SCENE_DURATION == 30.0
        assert agent.MAX_SCENE_DURATION == 90.0
    
    def test_validate_input_valid(self, agent, sample_ocr_output, default_config):
        """Test input validation with valid input"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        assert agent.validate_input(input_data) is True
    
    def test_validate_input_invalid_type(self, agent):
        """Test input validation with invalid type"""
        assert agent.validate_input("invalid") is False
        assert agent.validate_input(None) is False
        assert agent.validate_input(123) is False
    
    def test_validate_input_missing_attributes(self, agent):
        """Test input validation with missing attributes"""
        class InvalidInput:
            pass
        
        assert agent.validate_input(InvalidInput()) is False
    
    def test_get_retry_policy(self, agent):
        """Test retry policy configuration"""
        policy = agent.get_retry_policy()
        assert policy.max_attempts == 3
        assert "INVALID_JSON" in policy.retryable_errors
        assert "SCHEMA_VIOLATION" in policy.retryable_errors
    
    def test_execute_basic(self, agent, sample_ocr_output, default_config):
        """Test basic storyboard generation"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        assert isinstance(result, StoryboardJSON)
        assert result.pdf_metadata.filename == "test_document.pdf"
        assert result.pdf_metadata.page_count == 2
        assert result.configuration.verbosity == "medium"
        assert len(result.scenes) > 0
    
    def test_execute_invalid_input(self, agent):
        """Test execution with invalid input raises error"""
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute("invalid")
        
        assert exc_info.value.error_code == "INVALID_INPUT"
    
    def test_concept_identification(self, agent, sample_ocr_output, default_config):
        """Test concept extraction from OCR output"""
        concepts = agent._identify_concepts(sample_ocr_output, default_config)
        
        assert len(concepts) > 0
        
        # Check that definitions are identified
        definition_concepts = [c for c in concepts if c["concept_type"] == "definition"]
        assert len(definition_concepts) > 0
        
        # Check that theorems are identified
        theorem_concepts = [c for c in concepts if c["concept_type"] == "theorem"]
        assert len(theorem_concepts) > 0
        
        # Check that examples are identified
        example_concepts = [c for c in concepts if c["concept_type"] == "example"]
        assert len(example_concepts) > 0
        
        # Verify concept structure
        for concept in concepts:
            assert "concept_id" in concept
            assert "concept_name" in concept
            assert "concept_type" in concept
            assert "content" in concept
            assert "page_number" in concept
            assert "dependencies" in concept
    
    def test_concept_identification_no_proofs_for_introductory(self, agent, sample_ocr_output, introductory_config):
        """Test that proofs are skipped for introductory depth"""
        # Add a proof to the OCR output
        sample_ocr_output.pages[1].text_blocks.append(
            TextBlock(
                text="Proof: We prove this by induction.",
                bounding_box=BoundingBox(x=100, y=200, width=400, height=30),
                reading_order=2,
                confidence=0.95
            )
        )
        
        concepts = agent._identify_concepts(sample_ocr_output, introductory_config)
        
        # Proofs should not be extracted for introductory depth
        proof_concepts = [c for c in concepts if c["concept_type"] == "proof"]
        assert len(proof_concepts) == 0
    
    def test_concept_identification_includes_proofs_for_advanced(self, agent, sample_ocr_output, high_verbosity_config):
        """Test that proofs are included for advanced depth"""
        # Add a proof to the OCR output
        sample_ocr_output.pages[1].text_blocks.append(
            TextBlock(
                text="Proof: We prove this by induction.",
                bounding_box=BoundingBox(x=100, y=200, width=400, height=30),
                reading_order=2,
                confidence=0.95
            )
        )
        
        concepts = agent._identify_concepts(sample_ocr_output, high_verbosity_config)
        
        # Proofs should be extracted for advanced depth
        proof_concepts = [c for c in concepts if c["concept_type"] == "proof"]
        assert len(proof_concepts) > 0
    
    def test_concept_identification_empty_content(self, agent, default_config):
        """Test concept identification with empty content creates default concept"""
        empty_ocr = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="empty.pdf",
                page_count=1,
                file_size_bytes=1000
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
        
        concepts = agent._identify_concepts(empty_ocr, default_config)
        
        # Should create a default introduction concept
        assert len(concepts) == 1
        assert concepts[0]["concept_type"] == "introduction"
    
    def test_dependency_graph_building(self, agent, sample_ocr_output, default_config):
        """Test dependency graph construction"""
        concepts = agent._identify_concepts(sample_ocr_output, default_config)
        graph = agent._build_dependency_graph(concepts, sample_ocr_output)
        
        # Graph should have entries for all concepts
        assert len(graph) == len(concepts)
        
        # All concept IDs should be in graph
        for concept in concepts:
            assert concept["concept_id"] in graph
    
    def test_topological_sort_no_cycles(self, agent):
        """Test topological sort with acyclic graph"""
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": []
        }
        
        sorted_order = agent._topological_sort(graph)
        
        assert len(sorted_order) == 4
        # A should come before B and C
        assert sorted_order.index("A") < sorted_order.index("B")
        assert sorted_order.index("A") < sorted_order.index("C")
        # B and C should come before D
        assert sorted_order.index("B") < sorted_order.index("D")
        assert sorted_order.index("C") < sorted_order.index("D")
    
    def test_topological_sort_with_cycle(self, agent):
        """Test topological sort handles cycles gracefully"""
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"]  # Creates a cycle
        }
        
        sorted_order = agent._topological_sort(graph)
        
        # Should still return all nodes (cycle broken)
        assert len(sorted_order) == 3
        assert set(sorted_order) == {"A", "B", "C"}
    
    def test_topological_sort_empty_graph(self, agent):
        """Test topological sort with empty graph"""
        graph = {}
        sorted_order = agent._topological_sort(graph)
        assert sorted_order == []
    
    def test_scene_decomposition(self, agent, sample_ocr_output, default_config):
        """Test scene decomposition creates scenes for all concepts"""
        concepts = agent._identify_concepts(sample_ocr_output, default_config)
        graph = agent._build_dependency_graph(concepts, sample_ocr_output)
        ordered_concepts = agent._topological_sort(graph)
        
        scenes = agent._decompose_into_scenes(ordered_concepts, sample_ocr_output, default_config)
        
        # Should create one scene per concept
        assert len(scenes) == len(ordered_concepts)
        
        # Verify scene structure
        for i, scene in enumerate(scenes):
            assert scene.scene_id == f"scene_{i + 1:03d}"
            assert scene.scene_index == i
            assert scene.narration is not None
            assert len(scene.narration) > 0
            assert scene.visual_intent is not None
            assert scene.duration_estimate > 0
    
    def test_narration_generation_verbosity_low(self, agent, sample_ocr_output, low_verbosity_config):
        """Test narration generation with low verbosity"""
        narration = agent._generate_narration("concept_001", low_verbosity_config, sample_ocr_output)
        
        assert len(narration) > 0
        # Low verbosity should produce shorter narration
        assert len(narration.split()) < 50
    
    def test_narration_generation_verbosity_high(self, agent, sample_ocr_output, high_verbosity_config):
        """Test narration generation with high verbosity"""
        narration = agent._generate_narration("concept_001", high_verbosity_config, sample_ocr_output)
        
        assert len(narration) > 0
        # High verbosity should produce longer narration
        assert len(narration.split()) > 30
    
    def test_narration_generation_audience_adjustment(self, agent, sample_ocr_output):
        """Test narration adjusts based on audience level"""
        high_school_config = Configuration(
            verbosity="medium",
            depth="introductory",
            audience="high-school"
        )
        graduate_config = Configuration(
            verbosity="medium",
            depth="advanced",
            audience="graduate"
        )
        
        hs_narration = agent._generate_narration("concept_001", high_school_config, sample_ocr_output)
        grad_narration = agent._generate_narration("concept_001", graduate_config, sample_ocr_output)
        
        # Both should be non-empty
        assert len(hs_narration) > 0
        assert len(grad_narration) > 0
        
        # Content should differ based on audience
        assert "everyday life" in hs_narration.lower() or "familiar examples" in hs_narration.lower()
        assert "theoretical" in grad_narration.lower() or "generalizations" in grad_narration.lower()
    
    def test_visual_intent_generation(self, agent, sample_ocr_output, default_config):
        """Test visual intent generation"""
        visual_intent = agent._generate_visual_intent("concept_001", default_config, sample_ocr_output)
        
        assert visual_intent is not None
        assert len(visual_intent.mathematical_objects) > 0
        assert len(visual_intent.transformations) > 0
        
        # Verify mathematical object structure
        for obj in visual_intent.mathematical_objects:
            assert obj.object_type in ["equation", "graph", "geometric_shape", "axes", "text"]
            assert obj.content is not None
        
        # Verify transformation structure
        for transform in visual_intent.transformations:
            assert transform.transformation_type in ["morph", "highlight", "move", "fade_in", "fade_out", "scale"]
            assert transform.timing >= 0
    
    def test_duration_estimation(self, agent):
        """Test scene duration estimation"""
        short_narration = "This is a short narration."
        long_narration = " ".join(["word"] * 100)  # 100 words
        
        from src.schemas.storyboard import VisualIntent, MathematicalObject, Transformation, ObjectStyle
        
        simple_visual = VisualIntent(
            mathematical_objects=[
                MathematicalObject(object_type="equation", content="x=1", style=ObjectStyle())
            ],
            transformations=[
                Transformation(transformation_type="fade_in", target_object="eq", parameters={}, timing=0.5)
            ]
        )
        
        complex_visual = VisualIntent(
            mathematical_objects=[
                MathematicalObject(object_type="equation", content="x=1", style=ObjectStyle())
            ],
            transformations=[
                Transformation(transformation_type="fade_in", target_object="eq", parameters={}, timing=0.5),
                Transformation(transformation_type="morph", target_object="eq", parameters={}, timing=2.0),
                Transformation(transformation_type="highlight", target_object="eq", parameters={}, timing=4.0),
            ]
        )
        
        short_duration = agent._estimate_duration(short_narration, simple_visual)
        long_duration = agent._estimate_duration(long_narration, complex_visual)
        
        # Short narration should have shorter duration
        assert short_duration < long_duration
        
        # Durations should be within bounds
        assert agent.MIN_SCENE_DURATION <= short_duration <= agent.MAX_SCENE_DURATION
        assert agent.MIN_SCENE_DURATION <= long_duration <= agent.MAX_SCENE_DURATION
    
    def test_storyboard_assembly(self, agent, sample_ocr_output, default_config):
        """Test complete storyboard assembly"""
        concepts = agent._identify_concepts(sample_ocr_output, default_config)
        graph = agent._build_dependency_graph(concepts, sample_ocr_output)
        ordered_concepts = agent._topological_sort(graph)
        scenes = agent._decompose_into_scenes(ordered_concepts, sample_ocr_output, default_config)
        
        storyboard = agent._assemble_storyboard(sample_ocr_output, default_config, concepts, scenes)
        
        assert isinstance(storyboard, StoryboardJSON)
        assert storyboard.pdf_metadata.filename == "test_document.pdf"
        assert storyboard.configuration.verbosity == "medium"
        assert len(storyboard.concept_hierarchy) == len(concepts)
        assert len(storyboard.scenes) == len(scenes)
        
        # Verify concept hierarchy structure
        for concept_hier in storyboard.concept_hierarchy:
            assert concept_hier.concept_id is not None
            assert concept_hier.concept_name is not None
            assert isinstance(concept_hier.dependencies, list)
    
    def test_scene_id_uniqueness(self, agent, sample_ocr_output, default_config):
        """Test that all scene IDs are unique"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        scene_ids = [scene.scene_id for scene in result.scenes]
        assert len(scene_ids) == len(set(scene_ids))  # All IDs should be unique
    
    def test_scene_index_sequential(self, agent, sample_ocr_output, default_config):
        """Test that scene indices are sequential"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        for i, scene in enumerate(result.scenes):
            assert scene.scene_index == i
    
    def test_conceptual_continuity(self, agent, sample_ocr_output, default_config):
        """Test that scenes maintain conceptual continuity through dependencies"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        # First scene should have no dependencies
        if len(result.scenes) > 0:
            assert len(result.scenes[0].dependencies) == 0
        
        # Subsequent scenes should reference previous scenes
        for i in range(1, len(result.scenes)):
            assert len(result.scenes[i].dependencies) > 0
            # Should depend on previous scene
            assert f"scene_{i:03d}" in result.scenes[i].dependencies
    
    def test_configuration_defaults(self, agent, sample_ocr_output):
        """Test that default configuration values are used correctly"""
        assert agent.DEFAULT_VERBOSITY == "medium"
        assert agent.DEFAULT_DEPTH == "intermediate"
        assert agent.DEFAULT_AUDIENCE == "undergraduate"
    
    def test_multiple_pages_processing(self, agent, default_config):
        """Test processing of multi-page documents"""
        multi_page_ocr = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="multi_page.pdf",
                page_count=5,
                file_size_bytes=5000000
            ),
            pages=[
                Page(
                    page_number=i,
                    text_blocks=[
                        TextBlock(
                            text=f"Definition {i}: Content for page {i}",
                            bounding_box=BoundingBox(x=100, y=100, width=400, height=30),
                            reading_order=0,
                            confidence=0.95
                        )
                    ],
                    math_expressions=[],
                    diagrams=[]
                )
                for i in range(1, 6)
            ]
        )
        
        input_data = ScriptArchitectInput(multi_page_ocr, default_config)
        result = agent.execute(input_data)
        
        assert isinstance(result, StoryboardJSON)
        assert result.pdf_metadata.page_count == 5
        assert len(result.scenes) > 0


class TestScriptArchitectInput:
    """Test suite for ScriptArchitectInput"""
    
    def test_input_initialization(self):
        """Test ScriptArchitectInput initialization"""
        ocr_output = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="test.pdf",
                page_count=1,
                file_size_bytes=1000
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
        config = Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
        
        input_data = ScriptArchitectInput(ocr_output, config)
        
        assert input_data.ocr_output == ocr_output
        assert input_data.configuration == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
