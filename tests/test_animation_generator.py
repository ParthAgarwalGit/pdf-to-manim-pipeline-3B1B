"""Tests for Animation Code Generator Agent."""

import ast
import pytest
from datetime import datetime
from hypothesis import given, strategies as st

from src.agents.animation_generator import (
    AnimationCodeGeneratorAgent,
    AnimationGeneratorInput,
    AnimationGeneratorOutput,
    AnimationGeneratorError,
)
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.scene_processing import (
    SceneProcessingInput,
    SceneSpec,
    SharedContext,
    VisualStyle,
)


# ============================================================================
# Helper Functions
# ============================================================================

def build_test_scene_spec(
    scene_id: str = "test_001",
    narration: str = "This is a test scene.",
    duration: float = 30.0,
    math_objects: list = None,
    transformations: list = None
) -> SceneSpec:
    """Build a test scene specification."""
    if math_objects is None:
        math_objects = [
            {
                "object_type": "equation",
                "content": "x^2 + y^2 = r^2",
                "style": {"color": "blue", "position": "center"}
            }
        ]
    
    if transformations is None:
        transformations = [
            {
                "transformation_type": "fade_in",
                "target_object": "equation_0",
                "parameters": {},
                "timing": 0.5
            }
        ]
    
    return SceneSpec(
        scene_id=scene_id,
        narration=narration,
        visual_intent={
            "mathematical_objects": math_objects,
            "transformations": transformations,
            "emphasis_points": []
        },
        duration_estimate=duration
    )


def build_test_context() -> SharedContext:
    """Build a test shared context."""
    return SharedContext(
        concept_definitions={},
        visual_style=VisualStyle(
            colors={
                "primary": "#58C4DD",
                "secondary": "#83C167",
                "background": "#000000"
            },
            font_size=24,
            animation_speed=1.0
        ),
        variable_bindings={}
    )


# ============================================================================
# Property-Based Tests
# ============================================================================

@st.composite
def valid_scene_spec_strategy(draw):
    """Hypothesis strategy for generating valid scene specifications.
    
    **Validates: Requirements 8.5**
    """
    # Generate printable ASCII characters only (no null bytes or control characters)
    # Exclude 'True' and 'False' as scene_ids to avoid edge cases
    scene_id = draw(st.text(
        min_size=5,  # Minimum 5 characters to avoid reserved words
        max_size=20,
        alphabet=st.characters(min_codepoint=97, max_codepoint=122, whitelist_characters='_-')  # lowercase letters only
    ).filter(lambda s: s not in ['true', 'false', 'none']))
    
    # Exclude backslash from narration to avoid escape sequence issues in docstrings
    narration = draw(st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters='\\')
    ))
    duration = draw(st.floats(min_value=10.0, max_value=180.0))
    
    # Generate valid object types
    object_type = draw(st.sampled_from(["equation", "text", "axes", "graph", "geometric_shape"]))
    transformation_type = draw(st.sampled_from(["fade_in", "fade_out", "write", "morph", "highlight", "move", "scale"]))
    
    # Generate content without null bytes, backslashes, quotes, or problematic characters
    # Exclude backslash and quotes to avoid raw string edge cases
    content = draw(st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters='\\"\'')
    ))
    
    math_objects = [
        {
            "object_type": object_type,
            "content": content,
            "style": {"color": "blue", "position": "center"}
        }
    ]
    
    transformations = [
        {
            "transformation_type": transformation_type,
            "target_object": "equation_0",
            "parameters": {},
            "timing": draw(st.floats(min_value=0.0, max_value=5.0))
        }
    ]
    
    return build_test_scene_spec(
        scene_id=scene_id,
        narration=narration,
        duration=duration,
        math_objects=math_objects,
        transformations=transformations
    )


@given(scene_spec=valid_scene_spec_strategy())
def test_property_generated_code_is_syntactically_valid(scene_spec):
    """Property 4: Generated code is syntactically valid Python.
    
    **Validates: Requirements 8.5**
    
    For any scene specification with valid visual_intent, the generated code
    should parse successfully with Python's AST parser.
    """
    agent = AnimationCodeGeneratorAgent()
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    # Should return AnimationGeneratorOutput, not error
    assert isinstance(result, AnimationGeneratorOutput), \
        f"Expected AnimationGeneratorOutput, got {type(result).__name__}"
    
    # Generated code should be syntactically valid
    code = result.animation_code.code
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax error: {e}\nCode:\n{code}")


# ============================================================================
# Unit Tests
# ============================================================================

def test_animation_generator_generates_valid_code():
    """Test that Animation Generator generates valid Manim code."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec()
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    assert isinstance(result.animation_code, AnimationCodeObject)
    
    # Check animation code properties
    anim_code = result.animation_code
    assert anim_code.scene_id == "test_001"
    assert "class" in anim_code.code
    assert "def construct(self):" in anim_code.code
    assert "from manim import *" in anim_code.imports
    assert anim_code.narration == scene_spec.narration
    assert anim_code.duration_estimate == scene_spec.duration_estimate


def test_animation_generator_validates_syntax():
    """Test that Animation Generator validates generated code syntax."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec()
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    # Code should be syntactically valid
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should not raise SyntaxError
    ast.parse(code)


def test_animation_generator_generates_mathtex_for_equations():
    """Test that Animation Generator generates MathTex for equations."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_type": "equation",
                "content": "E = mc^2",
                "style": {"color": "blue"}
            }
        ]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain MathTex
    assert "MathTex" in code
    assert "E = mc^2" in code or "E = mc\\\\^2" in code  # May be escaped


def test_animation_generator_generates_text_objects():
    """Test that Animation Generator generates Text objects."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_type": "text",
                "content": "Hello World",
                "style": {"color": "white"}
            }
        ]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain Text
    assert "Text" in code
    assert "Hello World" in code


def test_animation_generator_generates_axes():
    """Test that Animation Generator generates Axes objects."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_type": "axes",
                "content": "coordinate axes",
                "style": {}
            }
        ]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain Axes
    assert "Axes" in code


def test_animation_generator_generates_geometric_shapes():
    """Test that Animation Generator generates geometric shapes."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_type": "geometric_shape",
                "content": "circle",
                "style": {"color": "red"}
            }
        ]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain Circle
    assert "Circle" in code


def test_animation_generator_generates_fade_in_transformation():
    """Test that Animation Generator generates FadeIn transformations."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        transformations=[
            {
                "transformation_type": "fade_in",
                "target_object": "equation_0",
                "parameters": {},
                "timing": 0.5
            }
        ]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain FadeIn
    assert "FadeIn" in code
    assert "self.play" in code


def test_animation_generator_generates_fade_out_transformation():
    """Test that Animation Generator generates FadeOut transformations."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        transformations=[
            {
                "transformation_type": "fade_out",
                "target_object": "equation_0",
                "parameters": {},
                "timing": 1.0
            }
        ]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain FadeOut
    assert "FadeOut" in code


def test_animation_generator_generates_scale_transformation():
    """Test that Animation Generator generates scale transformations."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        transformations=[
            {
                "transformation_type": "scale",
                "target_object": "equation_0",
                "parameters": {"scale_factor": 2.0},
                "timing": 0.5
            }
        ]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain scale
    assert "scale" in code
    assert "2.0" in code


def test_animation_generator_applies_visual_style():
    """Test that Animation Generator applies visual style from context."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec()
    
    # Custom visual style
    context = SharedContext(
        concept_definitions={},
        visual_style=VisualStyle(
            colors={"primary": "#FF0000"},
            font_size=32,
            animation_speed=2.0
        ),
        variable_bindings={}
    )
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should apply animation speed (run_time should be affected)
    assert "run_time" in code


def test_animation_generator_includes_header_comment():
    """Test that Animation Generator includes header comment with metadata."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        scene_id="intro_001",
        narration="This is the introduction scene."
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should contain header comment
    assert '"""' in code
    assert "Scene: intro_001" in code
    assert "Narration:" in code
    assert "Duration:" in code


def test_animation_generator_generates_valid_class_name():
    """Test that Animation Generator generates valid Python class names."""
    agent = AnimationCodeGeneratorAgent()
    
    # Test various scene_id formats
    test_cases = [
        ("scene-001", "Scene_scene_001"),
        ("intro_scene", "Intro_scene"),
        ("123_scene", "Scene_123_scene"),
        ("scene.with.dots", "Scene_with_dots"),
    ]
    
    for scene_id, expected_prefix in test_cases:
        scene_spec = build_test_scene_spec(scene_id=scene_id)
        context = build_test_context()
        
        processing_input = SceneProcessingInput(
            scene_spec=scene_spec,
            context=context,
            scene_index=0
        )
        
        input_data = AnimationGeneratorInput(processing_input)
        result = agent.execute(input_data)
        
        assert isinstance(result, AnimationGeneratorOutput)
        
        # Class name should be valid Python identifier
        class_name = result.animation_code.class_name
        assert class_name.isidentifier(), f"Class name '{class_name}' is not a valid identifier"
        
        # Code should contain the class definition
        assert f"class {class_name}(Scene):" in result.animation_code.code


def test_animation_generator_handles_empty_math_objects():
    """Test that Animation Generator handles scenes with no mathematical objects."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[],
        transformations=[]
    )
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should still generate valid code
    ast.parse(code)
    assert "def construct(self):" in code


def test_animation_generator_validates_input():
    """Test input validation."""
    agent = AnimationCodeGeneratorAgent()
    
    # Valid input
    scene_spec = build_test_scene_spec()
    context = build_test_context()
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    valid_input = AnimationGeneratorInput(processing_input)
    assert agent.validate_input(valid_input)
    
    # Invalid input type
    assert not agent.validate_input("not an input object")


def test_animation_generator_retry_policy():
    """Test that Animation Generator has retry policy with exponential backoff."""
    agent = AnimationCodeGeneratorAgent()
    policy = agent.get_retry_policy()
    
    assert policy.max_attempts == 3
    assert policy.backoff_strategy.value == "exponential"
    assert "SYNTAX_ERROR" in policy.retryable_errors


def test_animation_generator_includes_wait_for_duration():
    """Test that Animation Generator includes wait for scene duration."""
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(duration=45.5)
    context = build_test_context()
    
    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=0
    )
    
    input_data = AnimationGeneratorInput(processing_input)
    result = agent.execute(input_data)
    
    assert isinstance(result, AnimationGeneratorOutput)
    code = result.animation_code.code
    
    # Should include wait with duration
    assert "self.wait" in code
    assert "45.5" in code
