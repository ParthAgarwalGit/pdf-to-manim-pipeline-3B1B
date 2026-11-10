"""Tests for Scene Controller Agent."""

import pytest
from datetime import datetime
from hypothesis import given, strategies as st
from unittest.mock import Mock

from src.agents.scene_controller import (
    SceneControllerAgent,
    SceneControllerInput,
    SceneControllerOutput,
)
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.scene_processing import SharedContext, VisualStyle
from src.schemas.storyboard import (
    Configuration,
    MathematicalObject,
    ObjectStyle,
    PDFMetadataStoryboard,
    Scene,
    StoryboardJSON,
    Transformation,
    VisualIntent,
)


# ============================================================================
# Helper Functions
# ============================================================================

def build_test_storyboard(num_scenes: int = 3) -> StoryboardJSON:
    """Build a test storyboard with specified number of scenes."""
    scenes = []
    for i in range(num_scenes):
        scene = Scene(
            scene_id=f"scene_{i:03d}",
            scene_index=i,
            concept_id=f"concept_{i}",
            narration=f"This is narration for scene {i}.",
            visual_intent=VisualIntent(
                mathematical_objects=[
                    MathematicalObject(
                        object_type="equation",
                        content=f"x_{i} = {i}",
                        style=ObjectStyle(color="blue", position="center")
                    )
                ],
                transformations=[
                    Transformation(
                        transformation_type="fade_in",
                        target_object="equation_0",
                        parameters={},
                        timing=0.5
                    )
                ],
                emphasis_points=[]
            ),
            duration_estimate=30.0 + i * 5,
            dependencies=[]
        )
        scenes.append(scene)

    return StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(
            filename="test.pdf",
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


# ============================================================================
# Property-Based Tests
# ============================================================================

@st.composite
def scene_count_strategy(draw):
    """Hypothesis strategy for generating scene counts."""
    return draw(st.integers(min_value=1, max_value=10))


@pytest.mark.property
@given(num_scenes=scene_count_strategy())
def test_property_scene_iteration_is_deterministic(num_scenes):
    """Property 3: Scene iteration is deterministic.

    **Validates: Requirements 7.2, 10.2**

    For any scene array, iterating over scenes should process them in array
    order (index 0, 1, 2, ..., N-1) regardless of individual scene failures.
    """
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes)
    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None  # Use mock implementation
    )

    result = agent.execute(input_data)

    # Should return SceneControllerOutput
    assert isinstance(result, SceneControllerOutput), \
        f"Expected SceneControllerOutput, got {type(result).__name__}"

    # Should process all scenes
    assert result.report.total_scenes == num_scenes, \
        f"Expected {num_scenes} total scenes, got {result.report.total_scenes}"

    # Animation code objects should be in order by scene_index
    for i, anim_code in enumerate(result.animation_code_objects):
        expected_scene_id = f"scene_{i:03d}"
        assert anim_code.scene_id == expected_scene_id, \
            f"Scene at index {i} should have scene_id '{expected_scene_id}', got '{anim_code.scene_id}'"

    # All scenes should succeed with mock implementation
    assert result.report.successful_scenes == num_scenes, \
        f"All scenes should succeed with mock, got {result.report.successful_scenes}/{num_scenes}"


# ============================================================================
# Unit Tests
# ============================================================================

def test_scene_controller_processes_all_scenes():
    """Test that Scene Controller processes all scenes in order."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=3)
    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )

    result = agent.execute(input_data)

    assert isinstance(result, SceneControllerOutput)
    assert result.report.total_scenes == 3
    assert result.report.successful_scenes == 3
    assert len(result.animation_code_objects) == 3
    assert len(result.report.failed_scenes) == 0


def test_scene_controller_initializes_context():
    """Test that Scene Controller initializes shared context correctly."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=1)
    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )

    result = agent.execute(input_data)

    # Check that context was initialized
    assert isinstance(result.shared_context, SharedContext)
    assert isinstance(result.shared_context.visual_style, VisualStyle)
    assert result.shared_context.visual_style.font_size == 24
    assert result.shared_context.visual_style.animation_speed == 1.0

    # Check default 3Blue1Brown colors
    colors = result.shared_context.visual_style.colors
    assert "primary" in colors
    assert "secondary" in colors
    assert "background" in colors


def test_scene_controller_updates_context_with_concept_definitions():
    """Test that Scene Controller updates context with concept definitions."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=2)

    # Modify first scene to include a definition
    storyboard.scenes[0].narration = "A vector space is defined as a set with addition and scalar multiplication."
    storyboard.scenes[0].concept_id = "vector_space"

    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )

    result = agent.execute(input_data)

    # Check that concept was added to context
    assert "vector_space" in result.shared_context.concept_definitions
    assert "vector space is defined as" in result.shared_context.concept_definitions["vector_space"].lower()


def test_scene_controller_updates_context_with_variable_bindings():
    """Test that Scene Controller updates context with variable bindings."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=1)

    # Modify scene to include an equation with assignment
    storyboard.scenes[0].visual_intent.mathematical_objects[0].content = "x = 5"

    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )

    result = agent.execute(input_data)

    # Check that variable was added to context
    assert "x" in result.shared_context.variable_bindings
    assert result.shared_context.variable_bindings["x"] == "x = 5"


def test_scene_controller_isolates_failures():
    """Test that Scene Controller isolates failures and continues processing."""
    agent = SceneControllerAgent(max_retries_per_scene=1)  # Only 1 attempt, no retries
    storyboard = build_test_storyboard(num_scenes=3)

    # Create a mock animation generator that fails on second scene
    mock_generator = Mock()
    mock_generator.execute = Mock(side_effect=[
        AnimationCodeObject(
            scene_id="scene_000",
            code="# Scene 0",
            imports=["from manim import *"],
            class_name="Scene_000",
            narration="Scene 0",
            duration_estimate=30.0,
            generation_timestamp=datetime.now()
        ),
        Exception("Generation failed for scene 1"),
        AnimationCodeObject(
            scene_id="scene_002",
            code="# Scene 2",
            imports=["from manim import *"],
            class_name="Scene_002",
            narration="Scene 2",
            duration_estimate=40.0,
            generation_timestamp=datetime.now()
        )
    ])

    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=mock_generator
    )

    result = agent.execute(input_data)

    # Should have processed all 3 scenes
    assert result.report.total_scenes == 3

    # Should have 2 successful and 1 failed
    assert result.report.successful_scenes == 2
    assert len(result.report.failed_scenes) == 1

    # Failed scene should be scene_001
    assert result.report.failed_scenes[0].scene_id == "scene_001"
    assert "Generation failed" in result.report.failed_scenes[0].error


def test_scene_controller_retries_failed_scenes():
    """Test that Scene Controller retries failed scenes up to max attempts."""
    agent = SceneControllerAgent(max_retries_per_scene=3)
    storyboard = build_test_storyboard(num_scenes=1)

    # Create a mock animation generator that fails twice then succeeds
    call_count = 0
    def mock_execute(input_data):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Transient failure")
        return AnimationCodeObject(
            scene_id="scene_000",
            code="# Scene 0",
            imports=["from manim import *"],
            class_name="Scene_000",
            narration="Scene 0",
            duration_estimate=30.0,
            generation_timestamp=datetime.now()
        )

    mock_generator = Mock()
    mock_generator.execute = mock_execute

    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=mock_generator
    )

    result = agent.execute(input_data)

    # Should succeed after retries
    assert result.report.successful_scenes == 1
    assert len(result.report.failed_scenes) == 0
    assert call_count == 3  # Should have retried twice


def test_scene_controller_report_status_success():
    """Test that report status is SUCCESS when all scenes succeed."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=5)
    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )

    result = agent.execute(input_data)

    assert result.report.status == "SUCCESS"
    assert result.report.scene_success_rate == 1.0


def test_scene_controller_report_status_partial_success():
    """Test that report status is PARTIAL_SUCCESS when some scenes fail."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=5)

    # Create a mock generator that fails on 2 out of 5 scenes
    mock_generator = Mock()
    mock_generator.execute = Mock(side_effect=[
        AnimationCodeObject(
            scene_id="scene_000",
            code="# Scene 0",
            imports=["from manim import *"],
            class_name="Scene_000",
            narration="Scene 0",
            duration_estimate=30.0,
            generation_timestamp=datetime.now()
        ),
        Exception("Failed"),
        AnimationCodeObject(
            scene_id="scene_002",
            code="# Scene 2",
            imports=["from manim import *"],
            class_name="Scene_002",
            narration="Scene 2",
            duration_estimate=30.0,
            generation_timestamp=datetime.now()
        ),
        Exception("Failed"),
        AnimationCodeObject(
            scene_id="scene_004",
            code="# Scene 4",
            imports=["from manim import *"],
            class_name="Scene_004",
            narration="Scene 4",
            duration_estimate=30.0,
            generation_timestamp=datetime.now()
        )
    ])

    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=mock_generator
    )

    result = agent.execute(input_data)

    assert result.report.status == "PARTIAL_SUCCESS"
    assert result.report.successful_scenes == 3
    assert len(result.report.failed_scenes) == 2


def test_scene_controller_report_status_critical_failure():
    """Test that report status is CRITICAL_FAILURE when >50% scenes fail."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=5)

    # Create a mock generator that fails on 4 out of 5 scenes
    mock_generator = Mock()
    mock_generator.execute = Mock(side_effect=[
        AnimationCodeObject(
            scene_id="scene_000",
            code="# Scene 0",
            imports=["from manim import *"],
            class_name="Scene_000",
            narration="Scene 0",
            duration_estimate=30.0,
            generation_timestamp=datetime.now()
        ),
        Exception("Failed"),
        Exception("Failed"),
        Exception("Failed"),
        Exception("Failed")
    ])

    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=mock_generator
    )

    result = agent.execute(input_data)

    assert result.report.status == "CRITICAL_FAILURE"
    assert result.report.successful_scenes == 1
    assert len(result.report.failed_scenes) == 4


def test_scene_controller_validates_input():
    """Test input validation."""
    agent = SceneControllerAgent()

    # Valid input
    storyboard = build_test_storyboard(num_scenes=1)
    valid_input = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )
    assert agent.validate_input(valid_input)

    # Invalid input type
    assert not agent.validate_input("not an input object")

    # Invalid storyboard type
    invalid_input = SceneControllerInput(
        storyboard="not a storyboard",
        output_directory="output/test",
        animation_generator=None
    )
    assert not agent.validate_input(invalid_input)


def test_scene_controller_retry_policy():
    """Test that Scene Controller has retry policy with exponential backoff."""
    agent = SceneControllerAgent()
    policy = agent.get_retry_policy()

    assert policy.max_attempts == 3
    assert policy.backoff_strategy.value == "exponential"
    assert "ANIMATION_GENERATOR_UNAVAILABLE" in policy.retryable_errors


def test_scene_controller_generates_mock_animation_code():
    """Test that Scene Controller generates valid mock animation code."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=1)
    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )

    result = agent.execute(input_data)

    # Check mock animation code
    assert len(result.animation_code_objects) == 1
    anim_code = result.animation_code_objects[0]

    assert anim_code.scene_id == "scene_000"
    assert "class Scene_scene_000(Scene):" in anim_code.code
    assert "def construct(self):" in anim_code.code
    assert anim_code.class_name == "Scene_scene_000"
    assert "from manim import *" in anim_code.imports
    assert anim_code.duration_estimate == 30.0


def test_scene_controller_handles_empty_concept_id():
    """Test that Scene Controller handles scenes with empty concept_id."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=1)

    # Set concept_id to empty string
    storyboard.scenes[0].concept_id = ""

    input_data = SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None
    )

    result = agent.execute(input_data)

    # Should still succeed
    assert result.report.successful_scenes == 1
    assert len(result.report.failed_scenes) == 0


# ============================================================================
# Task 5 — Helper-friendly shared state across scenes
# ============================================================================

def test_scene_controller_exports_coordinate_system_bindings():
    """scene_state_out coordinate_systems must propagate into shared_context."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=1)
    storyboard.scenes[0].visual_intent.scene_state_out.coordinate_systems = {
        "axes_main": "number_plane"
    }
    result = agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))
    assert result.shared_context.coordinate_system_bindings["axes_main"] == "number_plane"


def test_scene_controller_exports_notation_bindings():
    """scene_state_out notation_bindings must propagate into shared_context."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=1)
    storyboard.scenes[0].visual_intent.scene_state_out.notation_bindings = {
        "sigma": "eq_sigma"
    }
    result = agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))
    assert result.shared_context.notation_registry["sigma"] == "eq_sigma"


def test_scene_controller_exports_character_roles():
    """scene_state_out character_roles must propagate into shared_context."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=1)
    storyboard.scenes[0].visual_intent.scene_state_out.character_roles = {
        "teacher": "teacher_obj"
    }
    result = agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))
    assert result.shared_context.character_registry["teacher"] == "teacher_obj"


def test_scene_controller_accumulates_bindings_across_scenes():
    """Bindings from scene N must still be available when scene N+1 runs."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=2)
    storyboard.scenes[0].visual_intent.scene_state_out.coordinate_systems = {
        "shared_axes": "axes"
    }
    storyboard.scenes[1].visual_intent.scene_state_out.notation_bindings = {
        "eq_main": "obj_0"
    }
    result = agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))
    # Both scenes contributed to the final shared context
    assert result.shared_context.coordinate_system_bindings.get("shared_axes") == "axes"
    assert result.shared_context.notation_registry.get("eq_main") == "obj_0"


def test_shared_context_has_registry_fields():
    """SharedContext must have coordinate, notation, and character registry fields."""
    ctx = SharedContext()
    assert hasattr(ctx, "coordinate_system_bindings")
    assert hasattr(ctx, "notation_registry")
    assert hasattr(ctx, "character_registry")
    assert isinstance(ctx.coordinate_system_bindings, dict)
    assert isinstance(ctx.notation_registry, dict)
    assert isinstance(ctx.character_registry, dict)


# ============================================================================
# Task 6: NarrativeContext propagation across scenes
# ============================================================================

def _two_scene_storyboard_with_beats() -> StoryboardJSON:
    """Two-scene storyboard whose source_beats introduce 'sine' then 'cosine'.

    Mirrors what the chunker (Task 4) attaches to each Scene so the
    SceneController can extract concepts from beats — not just from
    scene.concept_id.
    """
    from src.schemas.video_script import Beat

    storyboard = build_test_storyboard(num_scenes=2)
    storyboard.scenes[0].concept_id = "sine"
    storyboard.scenes[0].narration = (
        "Sine measures the height of a point on the unit circle. "
        "It starts at zero and rises to one."
    )
    storyboard.scenes[0].source_beats = [
        Beat(
            beat_id="beat_001",
            narration=storyboard.scenes[0].narration,
            visual_hint="Unit circle with rotating radius.",
            introduces_concepts=["sine"],
        )
    ]
    storyboard.scenes[1].concept_id = "cosine"
    storyboard.scenes[1].narration = (
        "Cosine is sine's companion: the horizontal projection on the same circle."
    )
    storyboard.scenes[1].source_beats = [
        Beat(
            beat_id="beat_002",
            narration=storyboard.scenes[1].narration,
            visual_hint="Same circle, horizontal projection highlighted.",
            introduces_concepts=["cosine"],
            references_prior_beats=["beat_001"],
        )
    ]
    return storyboard


def _stub_animation_generator():
    """Animation generator that always returns a trivial AnimationCodeObject."""
    from src.agents.animation_generator import AnimationGeneratorOutput

    gen = Mock()

    def execute(inp):
        sid = inp.scene_processing_input.scene_spec.scene_id
        return AnimationGeneratorOutput(
            animation_code=AnimationCodeObject(
                scene_id=sid,
                code=f"# {sid}",
                imports=["from manim import *"],
                class_name=sid.replace("scene", "Scene"),
                narration=inp.scene_processing_input.scene_spec.narration,
                duration_estimate=inp.scene_processing_input.scene_spec.duration_estimate,
                generation_timestamp=datetime.now(),
            )
        )

    gen.execute = Mock(side_effect=execute)
    return gen


def test_scene_controller_appends_narrative_summary_per_scene():
    """After two scenes, NarrativeContext records one summary per scene
    and concepts in first-introduction order."""
    agent = SceneControllerAgent()
    storyboard = _two_scene_storyboard_with_beats()
    gen = _stub_animation_generator()

    out = agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=gen,
    ))

    nc = out.shared_context.narrative
    assert len(nc.prior_scene_summaries) == 2
    assert "sine" in nc.concepts_introduced
    assert "cosine" in nc.concepts_introduced
    assert nc.concepts_introduced.index("sine") < nc.concepts_introduced.index("cosine")


def test_scene_controller_passes_narrative_into_next_scene():
    """Second scene's processing input must see the first scene's narrative."""
    agent = SceneControllerAgent()
    storyboard = _two_scene_storyboard_with_beats()
    gen = _stub_animation_generator()

    captured = []
    orig_side_effect = gen.execute.side_effect

    def spy(inp):
        captured.append(list(inp.scene_processing_input.context.narrative.prior_scene_summaries))
        return orig_side_effect(inp)

    gen.execute = Mock(side_effect=spy)

    agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=gen,
    ))

    assert len(captured) == 2
    assert captured[0] == []  # scene 1 sees empty narrative
    assert len(captured[1]) == 1  # scene 2 sees scene 1's summary


def test_scene_controller_running_storyline_is_bounded():
    """Running storyline keeps at most three sentences regardless of scene count."""
    agent = SceneControllerAgent()
    # Build a 5-scene storyboard so >3 summaries accumulate
    storyboard = build_test_storyboard(num_scenes=5)
    for i, scene in enumerate(storyboard.scenes):
        scene.narration = f"Scene {i} introduces topic number {i}."

    out = agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))

    storyline = out.shared_context.narrative.running_storyline
    # Count "real" sentences (split on '.', ignore empty fragments)
    sentence_count = sum(1 for s in storyline.split(".") if s.strip())
    assert sentence_count <= 3, f"Storyline had {sentence_count} sentences: {storyline!r}"


def test_scene_controller_live_objects_reflect_scene_state_out():
    """live_objects_on_screen should mirror the LAST scene's scene_state_out."""
    agent = SceneControllerAgent()
    storyboard = build_test_storyboard(num_scenes=2)
    # Scene 0 sets one active object (will be overwritten by scene 1)
    storyboard.scenes[0].visual_intent.scene_state_out.active_object_ids = ["only_in_scene_0"]
    # Scene 1 sets a different active object
    storyboard.scenes[1].visual_intent.mathematical_objects[0].object_id = "carryover_obj"
    storyboard.scenes[1].visual_intent.scene_state_out.active_object_ids = ["carryover_obj"]

    out = agent.execute(SceneControllerInput(
        storyboard=storyboard,
        output_directory="output/test",
        animation_generator=None,
    ))

    live = out.shared_context.narrative.live_objects_on_screen
    assert "carryover_obj" in live
    assert "only_in_scene_0" not in live
