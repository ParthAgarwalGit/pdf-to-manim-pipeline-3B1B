"""Tests for JSON Sanitizer Agent."""

import pytest
from hypothesis import given, strategies as st

from src.agents.json_sanitizer import (
    JSONSanitizerAgent,
    JSONSanitizerInput,
    JSONSanitizerOutput,
    SanitizationError,
)
from src.schemas.storyboard import StoryboardJSON


# ============================================================================
# Property-Based Tests
# ============================================================================

def build_valid_storyboard_dict():
    """Build a valid storyboard dictionary for property testing."""
    return {
        "pdf_metadata": {
            "filename": "test.pdf",
            "page_count": 5
        },
        "configuration": {
            "verbosity": "medium",
            "depth": "intermediate",
            "audience": "undergraduate"
        },
        "concept_hierarchy": [
            {
                "concept_id": "concept_1",
                "concept_name": "Test Concept",
                "dependencies": []
            }
        ],
        "scenes": [
            {
                "scene_id": "scene_001",
                "scene_index": 0,
                "concept_id": "concept_1",
                "narration": "This is a test narration.",
                "visual_intent": {
                    "mathematical_objects": [
                        {
                            "object_type": "equation",
                            "content": "x^2 + y^2 = r^2",
                            "style": {"color": "blue", "position": "center"}
                        }
                    ],
                    "transformations": [
                        {
                            "transformation_type": "fade_in",
                            "target_object": "equation_0",
                            "parameters": {},
                            "timing": 0.5
                        }
                    ],
                    "emphasis_points": []
                },
                "duration_estimate": 45.0,
                "dependencies": []
            }
        ]
    }


@st.composite
def valid_storyboard_strategy(draw):
    """Hypothesis strategy for generating valid storyboard dictionaries.
    
    **Validates: Requirements 6.1**
    """
    # Generate valid configuration values
    verbosity = draw(st.sampled_from(["low", "medium", "high"]))
    depth = draw(st.sampled_from(["introductory", "intermediate", "advanced"]))
    audience = draw(st.sampled_from(["high-school", "undergraduate", "graduate"]))
    
    # Generate valid object and transformation types
    object_type = draw(st.sampled_from(["equation", "graph", "geometric_shape", "axes", "text"]))
    transformation_type = draw(st.sampled_from(["morph", "highlight", "move", "fade_in", "fade_out", "scale"]))
    
    # Generate scene count (1-5 scenes)
    num_scenes = draw(st.integers(min_value=1, max_value=5))
    
    # Build scenes with unique scene_ids
    scenes = []
    for i in range(num_scenes):
        scene = {
            "scene_id": f"scene_{i:03d}",
            "scene_index": i,
            "concept_id": "concept_1",
            "narration": draw(st.text(min_size=10, max_size=200)),
            "visual_intent": {
                "mathematical_objects": [
                    {
                        "object_type": object_type,
                        "content": draw(st.text(min_size=1, max_size=50)),
                        "style": {"color": "blue", "position": "center"}
                    }
                ],
                "transformations": [
                    {
                        "transformation_type": transformation_type,
                        "target_object": "object_0",
                        "parameters": {},
                        "timing": draw(st.floats(min_value=0.0, max_value=10.0))
                    }
                ],
                "emphasis_points": []
            },
            "duration_estimate": draw(st.floats(min_value=10.0, max_value=180.0)),
            "dependencies": []
        }
        scenes.append(scene)
    
    return {
        "pdf_metadata": {
            "filename": draw(st.text(min_size=1, max_size=50)) + ".pdf",
            "page_count": draw(st.integers(min_value=1, max_value=100))
        },
        "configuration": {
            "verbosity": verbosity,
            "depth": depth,
            "audience": audience
        },
        "concept_hierarchy": [
            {
                "concept_id": "concept_1",
                "concept_name": "Test Concept",
                "dependencies": []
            }
        ],
        "scenes": scenes
    }


@given(storyboard_dict=valid_storyboard_strategy())
def test_property_valid_storyboards_pass_unchanged(storyboard_dict):
    """Property 2: Valid storyboards pass sanitization unchanged.
    
    **Validates: Requirements 6.1**
    
    For any Storyboard JSON that conforms to the schema, sanitization should
    return the input unchanged (no corrections applied).
    """
    agent = JSONSanitizerAgent()
    input_data = JSONSanitizerInput(storyboard_dict)
    
    result = agent.execute(input_data)
    
    # Should return JSONSanitizerOutput, not SanitizationError
    assert isinstance(result, JSONSanitizerOutput), \
        f"Valid storyboard should pass sanitization, got {type(result).__name__}"
    
    # Should not have any corrections
    assert not result.was_corrected, \
        f"Valid storyboard should not be corrected, but got {len(result.corrections)} corrections"
    
    assert len(result.corrections) == 0, \
        f"Valid storyboard should have no corrections, got: {result.corrections}"
    
    # Should return a valid StoryboardJSON object
    assert isinstance(result.storyboard, StoryboardJSON), \
        "Result should contain a StoryboardJSON object"


# ============================================================================
# Unit Tests
# ============================================================================

def test_sanitizer_accepts_valid_storyboard():
    """Test that a valid storyboard passes sanitization without changes."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    input_data = JSONSanitizerInput(storyboard_dict)
    
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert not result.was_corrected
    assert len(result.corrections) == 0
    assert isinstance(result.storyboard, StoryboardJSON)


def test_sanitizer_adds_missing_concept_hierarchy():
    """Test automatic correction for missing optional field 'concept_hierarchy'."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Remove optional field
    del storyboard_dict["concept_hierarchy"]
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    assert len(result.corrections) >= 1
    
    # Check that concept_hierarchy was added
    correction = next((c for c in result.corrections if c.field_path == "concept_hierarchy"), None)
    assert correction is not None
    assert correction.violation_type == "missing_optional_field"
    assert correction.corrected_value == []


def test_sanitizer_fixes_duplicate_scene_ids():
    """Test automatic correction for duplicate scene_ids."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Add a duplicate scene with same scene_id
    duplicate_scene = storyboard_dict["scenes"][0].copy()
    duplicate_scene["scene_index"] = 1
    storyboard_dict["scenes"].append(duplicate_scene)
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    
    # Check that duplicate was renamed
    scene_ids = [scene.scene_id for scene in result.storyboard.scenes]
    assert len(scene_ids) == len(set(scene_ids)), "Scene IDs should be unique after correction"
    
    # Check correction was logged
    duplicate_corrections = [c for c in result.corrections if c.violation_type == "duplicate_scene_id"]
    assert len(duplicate_corrections) >= 1


def test_sanitizer_coerces_page_count_string_to_int():
    """Test type coercion for page_count from string to int."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Set page_count as string
    storyboard_dict["pdf_metadata"]["page_count"] = "10"
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    
    # Check that page_count was coerced to int
    assert result.storyboard.pdf_metadata.page_count == 10
    assert isinstance(result.storyboard.pdf_metadata.page_count, int)
    
    # Check correction was logged
    correction = next((c for c in result.corrections if "page_count" in c.field_path), None)
    assert correction is not None
    assert correction.violation_type == "type_mismatch"


def test_sanitizer_coerces_scene_index_string_to_int():
    """Test type coercion for scene_index from string to int."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Set scene_index as string
    storyboard_dict["scenes"][0]["scene_index"] = "0"
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    
    # Check that scene_index was coerced to int
    assert result.storyboard.scenes[0].scene_index == 0
    assert isinstance(result.storyboard.scenes[0].scene_index, int)


def test_sanitizer_coerces_duration_estimate_string_to_float():
    """Test type coercion for duration_estimate from string to float."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Set duration_estimate as string
    storyboard_dict["scenes"][0]["duration_estimate"] = "45.5"
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    
    # Check that duration_estimate was coerced to float
    assert result.storyboard.scenes[0].duration_estimate == 45.5
    assert isinstance(result.storyboard.scenes[0].duration_estimate, float)


def test_sanitizer_adds_missing_dependencies_field():
    """Test automatic correction for missing optional 'dependencies' field in scenes."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Remove optional dependencies field
    del storyboard_dict["scenes"][0]["dependencies"]
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    
    # Check that dependencies was added
    assert result.storyboard.scenes[0].dependencies == []


def test_sanitizer_adds_missing_emphasis_points():
    """Test automatic correction for missing optional 'emphasis_points' field."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Remove optional emphasis_points field
    del storyboard_dict["scenes"][0]["visual_intent"]["emphasis_points"]
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    
    # Check that emphasis_points was added
    assert result.storyboard.scenes[0].visual_intent.emphasis_points == []


def test_sanitizer_returns_error_for_invalid_verbosity():
    """Test that invalid verbosity value cannot be corrected."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Set invalid verbosity
    storyboard_dict["configuration"]["verbosity"] = "invalid_value"
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    # Should return SanitizationError since this cannot be auto-corrected
    assert isinstance(result, SanitizationError)
    assert len(result.errors) > 0


def test_sanitizer_returns_error_for_missing_required_field():
    """Test that missing required fields cannot be corrected."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Remove required field
    del storyboard_dict["scenes"][0]["narration"]
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    # Should return SanitizationError
    assert isinstance(result, SanitizationError)
    assert len(result.errors) > 0


def test_sanitizer_returns_error_for_empty_scenes_list():
    """Test that empty scenes list cannot be corrected."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Set scenes to empty list
    storyboard_dict["scenes"] = []
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    # Should return SanitizationError
    assert isinstance(result, SanitizationError)
    assert len(result.errors) > 0


def test_sanitizer_validates_input():
    """Test input validation."""
    agent = JSONSanitizerAgent()
    
    # Valid input
    valid_input = JSONSanitizerInput(build_valid_storyboard_dict())
    assert agent.validate_input(valid_input)
    
    # Invalid input type
    assert not agent.validate_input("not an input object")
    
    # Invalid storyboard_dict type
    invalid_input = JSONSanitizerInput("not a dict")
    assert not agent.validate_input(invalid_input)


def test_sanitizer_retry_policy():
    """Test that sanitizer has no-retry policy."""
    agent = JSONSanitizerAgent()
    policy = agent.get_retry_policy()
    
    assert policy.max_attempts == 1, "Sanitizer should not retry (deterministic)"


def test_sanitizer_handles_multiple_corrections():
    """Test that sanitizer can apply multiple corrections at once."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Remove multiple optional fields and add type mismatches
    del storyboard_dict["concept_hierarchy"]
    del storyboard_dict["scenes"][0]["dependencies"]
    storyboard_dict["pdf_metadata"]["page_count"] = "5"
    storyboard_dict["scenes"][0]["scene_index"] = "0"
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.was_corrected
    assert len(result.corrections) >= 4  # At least 4 corrections should be made


def test_sanitizer_preserves_valid_optional_fields():
    """Test that sanitizer preserves valid optional fields that are present."""
    agent = JSONSanitizerAgent()
    storyboard_dict = build_valid_storyboard_dict()
    
    # Add optional difficulty_level field
    storyboard_dict["scenes"][0]["difficulty_level"] = "intermediate"
    
    input_data = JSONSanitizerInput(storyboard_dict)
    result = agent.execute(input_data)
    
    assert isinstance(result, JSONSanitizerOutput)
    assert result.storyboard.scenes[0].difficulty_level == "intermediate"
