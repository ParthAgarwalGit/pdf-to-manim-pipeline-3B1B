"""Tests for Animation Code Generator Agent."""

import ast
import pytest
from datetime import datetime
from hypothesis import given, settings, strategies as st

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


@pytest.fixture(autouse=True)
def _default_to_mock_path(monkeypatch):
    """Pin every test in this module to the deterministic mock generator.

    ``src/cli.py`` calls ``load_dotenv(keys.env)`` at IMPORT time, so as soon
    as ``test_cli`` is collected the real ANTHROPIC_API_KEY leaks into the whole
    pytest process. Tests here assert on the mock emitter's exact output
    (``create_notation``, ``FadeIn``, ``scale(2.0)`` …), which is meaningless
    against live, non-deterministic LLM code. Defaulting to the mock path makes
    the suite order-independent. The prompt-capture tests re-set the key
    themselves via their own monkeypatch, which runs after this fixture.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


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


@pytest.mark.property
@settings(deadline=None)
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
    
    # Should contain create_notation helper (helper layer replaces raw MathTex)
    assert "create_notation" in code
    assert "E = mc^2" in code or "mc" in code


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
    
    # Should contain create_text_label helper (helper layer replaces raw Text)
    assert "create_text_label" in code
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
    
    # Should contain create_coordinate_system helper (helper layer replaces raw Axes)
    assert "create_coordinate_system" in code


def test_animation_generator_supports_pi_creature_helpers(monkeypatch):
    """Pi-creature storyboard hints should map to the repo helper functions."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_type": "pi_creature",
                "render_hint": "PiCreatureTeacher",
                "content": "Teacher introduces the intuition",
                "style": {"color": "blue"},
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
    assert "create_pi_teacher" in code


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


# ============================================================================
# Task 4 — Helper-backed code generation
# ============================================================================

def test_animation_generator_uses_coordinate_helper_for_axes_objects(monkeypatch):
    """Mock path must emit create_coordinate_system for axes-type objects."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_id": "axes_main",
                "object_type": "axes",
                "render_hint": "NumberPlane",
                "content": "main frame",
                "style": {},
            }
        ],
        transformations=[],
    )
    result = agent.execute(AnimationGeneratorInput(SceneProcessingInput(
        scene_spec=scene_spec,
        context=build_test_context(),
        scene_index=0,
    )))
    assert isinstance(result, AnimationGeneratorOutput)
    assert "create_coordinate_system" in result.animation_code.code


def test_animation_generator_uses_notation_helper_when_notation_key_exists(monkeypatch):
    """Mock path must emit create_notation when notation_key is provided."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_id": "eq_main",
                "object_type": "equation",
                "notation_key": "svd_core",
                "content": r"X = U\Sigma V^T",
                "style": {"color": "blue"},
            }
        ],
        transformations=[],
    )
    result = agent.execute(AnimationGeneratorInput(SceneProcessingInput(
        scene_spec=scene_spec,
        context=build_test_context(),
        scene_index=0,
    )))
    assert isinstance(result, AnimationGeneratorOutput)
    assert "create_notation" in result.animation_code.code


def test_animation_generator_uses_character_role_helper_for_pi_creature(monkeypatch):
    """Mock path must emit create_character_for_role for pi_creature objects."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_id": "teacher_obj",
                "object_type": "pi_creature",
                "character_role": "teacher",
                "content": "teacher",
                "style": {},
            }
        ],
        transformations=[],
    )
    result = agent.execute(AnimationGeneratorInput(SceneProcessingInput(
        scene_spec=scene_spec,
        context=build_test_context(),
        scene_index=0,
    )))
    assert isinstance(result, AnimationGeneratorOutput)
    assert "create_character_for_role" in result.animation_code.code


def test_animation_generator_uses_object_id_as_variable_name(monkeypatch):
    """Stable object_id should become the Python variable name."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = AnimationCodeGeneratorAgent()
    scene_spec = build_test_scene_spec(
        math_objects=[
            {
                "object_id": "sigma_matrix",
                "object_type": "equation",
                "content": r"\Sigma",
                "style": {},
            }
        ],
        transformations=[],
    )
    result = agent.execute(AnimationGeneratorInput(SceneProcessingInput(
        scene_spec=scene_spec,
        context=build_test_context(),
        scene_index=0,
    )))
    assert isinstance(result, AnimationGeneratorOutput)
    assert "sigma_matrix" in result.animation_code.code


def test_animation_generator_system_prompt_includes_quality_contract():
    """The system prompt should include the 3B1B quality contract."""
    from src.utils.manim_engine_contract import build_3b1b_quality_prompt
    prompt = build_3b1b_quality_prompt()
    # Verify the prompt is non-empty and contains key sections
    assert "TIMING CONSTRAINTS" in prompt
    assert "PROGRESSIVE REVEAL" in prompt
    assert "COLOR CONSISTENCY" in prompt
    assert "SMOOTH MORPHING" in prompt
    # Verify the import works in the animation generator module
    from src.agents.animation_generator import AnimationCodeGeneratorAgent


# ============================================================================
# Task 8: NarrativeContext briefing in the animation-generator system prompt
# ============================================================================

def _capture_animation_prompt(agent_factory, monkeypatch, scene_spec, context, scene_index=0):
    """Helper: install a fake Anthropic client that captures (system, user)
    and returns a trivial valid Manim class."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    captured: dict = {}

    class _Resp:
        content = [type("B", (), {"text": "class S(Scene):\n    def construct(self):\n        self.wait(1)"})]

    class _Messages:
        def create(self, *, system, messages, **_kw):
            captured["system_raw"] = system
            # The real call sends system as a list of blocks (cached static +
            # dynamic tail); normalize to one string for substring assertions.
            if isinstance(system, list):
                captured["system"] = "".join(b.get("text", "") for b in system)
            else:
                captured["system"] = system
            captured["user"] = messages[0]["content"]
            return _Resp()

    class _Client:
        messages = _Messages()

    agent = agent_factory()
    agent.client = _Client()

    processing_input = SceneProcessingInput(
        scene_spec=scene_spec,
        context=context,
        scene_index=scene_index,
    )
    agent.execute(AnimationGeneratorInput(processing_input))
    return captured


def test_animation_prompt_includes_concepts_already_taught(monkeypatch):
    """When NarrativeContext lists previously-taught concepts, they must
    appear in the system prompt with a 'do not re-explain' instruction."""
    from src.schemas.scene_processing import NarrativeContext

    spec = build_test_scene_spec(
        scene_id="scene_002",
        narration="Cosine traces the x-coordinate.",
        duration=20.0,
    )
    context = build_test_context()
    context.narrative = NarrativeContext(
        prior_scene_summaries=["[sine] Sine on the unit circle introduced."],
        concepts_introduced=["sine", "unit_circle"],
        running_storyline="Sine on the unit circle introduced.",
    )

    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, context, scene_index=1,
    )
    full = captured["system"] + captured["user"]
    assert "sine" in full.lower()
    assert "unit_circle" in full.lower()
    assert "do not re-explain" in full.lower() or "already" in full.lower()


def test_animation_prompt_omits_briefing_when_narrative_empty(monkeypatch):
    """First scene has empty narrative; the briefing section must NOT appear
    so we don't waste tokens on noise."""
    spec = build_test_scene_spec(scene_id="scene_001", narration="Intro.", duration=15.0)
    context = build_test_context()  # narrative is default-empty

    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, context, scene_index=0,
    )
    assert "NARRATIVE BRIEFING" not in captured["system"]


def test_animation_prompt_does_not_dump_full_pdf(monkeypatch):
    """Even with a populated narrative, the prompt must never include a
    placeholder for the raw PDF / OCR text."""
    from src.schemas.scene_processing import NarrativeContext

    spec = build_test_scene_spec(scene_id="scene_002", narration="Cosine.", duration=15.0)
    context = build_test_context()
    context.narrative = NarrativeContext(
        prior_scene_summaries=["[sine] First scene."],
        concepts_introduced=["sine"],
    )

    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, context, scene_index=1,
    )
    full = captured["system"] + captured["user"]
    assert "OCR" not in full
    # "PDF" appears in the existing CONTENT FIDELITY section ("extracted from
    # an academic PDF"). That is intentional context, not a dump. Guard
    # against the actual anti-pattern: a labelled raw-text payload.
    assert "RAW PDF TEXT" not in full
    assert "FULL PDF" not in full


# ============================================================================
# Task 9: Manim-level visual continuity at chunk boundaries (opening state)
# ============================================================================

def test_animation_prompt_instructs_carryover_when_live_objects_present(monkeypatch):
    """When the previous scene leaves objects on screen, the prompt must
    instruct the LLM to re-instantiate them at run_time=0 so the cut is
    seamless."""
    from src.schemas.scene_processing import NarrativeContext

    spec = build_test_scene_spec(scene_id="scene_002", narration="Continue.", duration=15.0)
    context = build_test_context()
    context.narrative = NarrativeContext(
        live_objects_on_screen={"unit_circle": "geometric_shape: Circle(radius=2)"},
    )

    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, context, scene_index=1,
    )
    sys_prompt = captured["system"]
    assert "OPENING STATE" in sys_prompt
    # Instruction must reference one of the no-animation reentry mechanisms.
    assert "self.add" in sys_prompt or "run_time=0" in sys_prompt
    # The live object id must be named so the LLM knows what to recreate.
    assert "unit_circle" in sys_prompt


def test_animation_prompt_omits_opening_state_when_no_live_objects(monkeypatch):
    """If nothing carries over, the OPENING STATE section must NOT appear."""
    spec = build_test_scene_spec(scene_id="scene_001", narration="Intro.", duration=15.0)
    context = build_test_context()  # narrative default — no live objects

    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, context, scene_index=0,
    )
    assert "OPENING STATE" not in captured["system"]


def test_animation_prompt_renders_color_continuity_when_bindings_present(monkeypatch):
    """When SharedContext.color_bindings is populated, the system prompt
    must surface the variable→color mapping as a tex_to_color_map hint
    (so the LLM uses the same color for the same variable across scenes)."""
    spec = build_test_scene_spec(scene_id="scene_002", narration="Continue.", duration=15.0)
    context = build_test_context()
    context.color_bindings = {"x": "#58C4DD", "y": "#83C167"}

    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, context, scene_index=1,
    )
    sys_prompt = captured["system"]
    assert "COLOR CONTINUITY" in sys_prompt
    assert "tex_to_color_map" in sys_prompt
    assert "#58C4DD" in sys_prompt
    assert "#83C167" in sys_prompt


def test_animation_prompt_omits_color_continuity_when_no_bindings(monkeypatch):
    """First scene has no color bindings; the section must NOT appear."""
    spec = build_test_scene_spec(scene_id="scene_001", narration="Intro.", duration=15.0)
    context = build_test_context()  # color_bindings default-empty

    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, context, scene_index=0,
    )
    assert "COLOR CONTINUITY" not in captured["system"]


def test_animation_generator_imports_animation_patterns_helpers():
    """Generated Manim files must import the 3B1B animation pattern helpers
    so the LLM can call progressive_write / dim_and_focus directly."""
    agent = AnimationCodeGeneratorAgent()
    imports = agent._get_imports()
    joined = "\n".join(imports)
    assert "animation_patterns" in joined
    assert "progressive_write" in joined


# ============================================================================
# Task 12: Closed-vocabulary primitive emitter
# ============================================================================

class TestPrimitiveEmitter:
    """The animation generator becomes a thin emitter for catalog primitives:
    when ``primitive`` is set on a math object, the legacy if/elif chain is
    bypassed and the emitter writes a single deterministic constructor call."""

    def _agent(self):
        return AnimationCodeGeneratorAgent.__new__(AnimationCodeGeneratorAgent)

    def test_emit_primitive_writes_circle_call_with_color(self):
        agent = self._agent()
        line = agent._emit_primitive("c0", {
            "primitive": "Circle",
            "args": {"radius": 2.5},
            "style": {"color": "blue"},
        })
        # 8-space indent (inside class > def construct), call shape, color upper.
        assert line.strip() == "c0 = Circle(radius=2.5, color=BLUE)"

    def test_emit_primitive_omits_color_when_spec_rejects_it(self):
        agent = self._agent()
        # Axes' PrimitiveSpec.accepts_color is False; color must be skipped.
        line = agent._emit_primitive("ax", {
            "primitive": "Axes",
            "args": {"x_length": 5.0},
            "style": {"color": "red"},
        })
        assert "color" not in line
        assert "Axes(x_length=5.0)" in line

    def test_emit_primitive_uses_raw_string_for_tex(self):
        agent = self._agent()
        line = agent._emit_primitive("eq", {
            "primitive": "MathTex",
            "args": {"tex": r"\sin(\theta)"},
            "style": {"color": "yellow"},
        })
        # Backslashes must survive in the emitted source.
        assert r"\sin(\theta)" in line or r"\\sin(\\theta)" in line

    def test_emit_bbox_placement_returns_move_to(self):
        agent = self._agent()
        line = agent._emit_bbox_placement("c0", {
            "bbox": {"x": 1.5, "y": -2.0, "w": 3.0, "h": 1.0},
        })
        assert line.strip() == "c0.move_to([1.5, -2.0, 0])"

    def test_emit_bbox_placement_returns_empty_when_no_bbox(self):
        agent = self._agent()
        assert agent._emit_bbox_placement("c0", {}) == ""

    def test_generate_mathematical_objects_uses_catalog_path_when_primitive_set(
        self, monkeypatch
    ):
        """If primitive is set, the generator emits the catalog form and the
        legacy heuristic chain is bypassed — no `create_notation`/`Polygon` rescue.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        agent = AnimationCodeGeneratorAgent()
        context = build_test_context()
        objects = [
            {
                "object_id": "c0",
                "object_type": "geometric_shape",
                "content": "",
                "primitive": "Circle",
                "args": {"radius": 2.0},
                "style": {"color": "blue"},
                "bbox": {"x": 1.0, "y": 0.5, "w": 4.0, "h": 4.0},
            }
        ]
        emitted = agent._generate_mathematical_objects(objects, context)
        assert "Circle(radius=2.0, color=BLUE)" in emitted
        assert "c0.move_to([1.0, 0.5, 0])" in emitted
        # Legacy fallback would have produced a `create_notation` / generic
        # Polygon; it must NOT appear when the catalog path fires.
        assert "create_notation" not in emitted


# ============================================================================
# Plan-to-render: the structured visual plan must reach the code LLM
# ============================================================================


def _agent_no_init():
    """AnimationCodeGeneratorAgent without __init__ (no API client needed)."""
    return AnimationCodeGeneratorAgent.__new__(AnimationCodeGeneratorAgent)


def test_object_plan_forwards_primitive_position_and_interval():
    """`_object_plan_for_prompt` must carry the planner's primitive, kwargs,
    (x, y) position and (t_in, t_out) window — not just type/content/style."""
    from src.schemas.storyboard import MathematicalObject, ObjectStyle, BBox, TimeInterval

    obj = MathematicalObject(
        object_id="v0",
        object_role="main_vector",
        object_type="geometric_shape",
        content="velocity vector",
        primitive="Vector",
        args={"direction": [2.0, 1.0, 0.0]},
        bbox=BBox(x=1.0, y=-0.5, w=2.0, h=2.0),
        interval=TimeInterval(t_in=0.0, t_out=6.0),
        style=ObjectStyle(color="blue"),
    )
    plan = _agent_no_init()._object_plan_for_prompt(obj)
    assert plan["primitive"] == "Vector"
    assert plan["args"] == {"direction": [2.0, 1.0, 0.0]}
    assert plan["position"] == {"x": 1.0, "y": -0.5}
    assert plan["visible"] == {"t_in": 0.0, "t_out": 6.0}
    assert plan["id"] == "v0"
    assert plan["role"] == "main_vector"


def test_object_plan_omits_absent_fields():
    """A bare legacy object must not sprout empty primitive/position keys."""
    from src.schemas.storyboard import MathematicalObject, ObjectStyle

    obj = MathematicalObject(
        object_type="text",
        content="hello",
        style=ObjectStyle(),
    )
    plan = _agent_no_init()._object_plan_for_prompt(obj)
    assert plan["type"] == "text"
    assert plan["content"] == "hello"
    for absent in ("primitive", "args", "position", "visible"):
        assert absent not in plan


def test_animation_prompt_carries_planned_primitive_and_position(monkeypatch):
    """End-to-end: a planned Vector with a bbox must appear in the code LLM's
    prompt (primitive + coordinates) and the 'HONOR IT' instruction present."""
    spec = build_test_scene_spec(
        scene_id="scene_003",
        narration="The velocity points up and to the right.",
        duration=15.0,
        math_objects=[
            {
                "object_id": "v0",
                "object_type": "geometric_shape",
                "content": "velocity vector",
                "primitive": "Vector",
                "args": {"direction": [2, 1, 0]},
                "bbox": {"x": 1.5, "y": -0.5, "w": 2.0, "h": 2.0},
                "style": {"color": "blue"},
            }
        ],
        transformations=[],
    )
    captured = _capture_animation_prompt(
        AnimationCodeGeneratorAgent, monkeypatch, spec, build_test_context(),
        scene_index=0,
    )
    assert "VISUAL PLAN (HONOR IT)" in captured["system"]
    user = captured["user"]
    assert "Vector" in user            # planned primitive survived
    assert '"x": 1.5' in user          # planned position survived
    assert "direction" in user         # planned kwargs survived


# ============================================================================
# Duration sync (P1.4): distribute slack across existing waits, no freeze-tail
# ============================================================================


def test_slack_leaves_scene_already_at_target_untouched():
    """A scene whose animation already reaches the audio length is returned
    as-is — nothing to distribute, no freeze-frame to avoid."""
    agent = _agent_no_init()
    code = (
        "class S(Scene):\n"
        "    def construct(self):\n"
        "        self.play(Create(c), run_time=5.0)\n"
        "        self.wait(2.0)\n"
        "        self.play(Transform(a, b), run_time=3.0)\n"
    )  # total = 10.0s >= target
    assert agent._distribute_duration_slack(code, 8.0) == code


def test_slack_distributed_across_existing_waits():
    """Slack is spread across the scene's waits (capped), not dumped as a
    single tail freeze."""
    agent = _agent_no_init()
    code = (
        "class S(Scene):\n"
        "    def construct(self):\n"
        "        self.play(Write(a), run_time=1.0)\n"
        "        self.wait(1.0)\n"
        "        self.play(Write(b), run_time=1.0)\n"
        "        self.wait(1.0)\n"
    )  # total = 4.0s, two waits with 3s headroom each (cap 4.0)
    out = agent._distribute_duration_slack(code, 10.0)  # gap = 6.0
    import re
    waits = [float(x) for x in re.findall(r"self\.wait\((\d+(?:\.\d+)?)\)", out)]
    run_times = [float(x) for x in re.findall(r"run_time=(\d+(?:\.\d+)?)", out)]
    total = sum(waits) + sum(run_times)
    assert abs(total - 10.0) < 0.3          # reaches the audio length
    assert all(w <= agent.MAX_SINGLE_WAIT + 0.05 for w in waits)  # capped
    assert len(waits) >= 2 and max(waits) > 1.0  # spread, not one giant tail


def test_slack_caps_then_uses_tail_for_residual():
    """When existing waits can't absorb all the slack (hit the cap), the
    remainder goes to a single bounded tail wait."""
    agent = _agent_no_init()
    code = (
        "class S(Scene):\n"
        "    def construct(self):\n"
        "        self.play(Write(a), run_time=1.0)\n"
        "        self.wait(1.0)\n"
    )  # one wait, 3s headroom; total 2s
    out = agent._distribute_duration_slack(code, 20.0)  # gap 18 >> capacity 3
    assert "duration sync tail" in out
    import re
    total = sum(float(x) for x in re.findall(r"self\.wait\((\d+(?:\.\d+)?)\)", out))
    total += 1.0  # run_time
    assert abs(total - 20.0) < 0.3


def test_slack_no_waits_appends_single_tail():
    """A scene with zero waits gets one tail wait (the only available lever)."""
    agent = _agent_no_init()
    code = (
        "class S(Scene):\n"
        "    def construct(self):\n"
        "        self.play(Write(t), run_time=2.0)\n"
    )
    out = agent._distribute_duration_slack(code, 8.0)
    assert "self.wait(6.0)" in out
    assert "run_time=2.0" in out  # original animation untouched