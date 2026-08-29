from types import SimpleNamespace

from src.agents.animation_generator import AnimationCodeGeneratorAgent
from src.agents.script_architect import ScriptArchitectAgent
from src.agents.tts_agent import TTSOutput
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.schemas.scene_processing import SceneSpec, SharedContext
from src.schemas.storyboard import (
    BBox,
    Configuration,
    PDFMetadataStoryboard,
    MathematicalObject,
    Scene,
    StoryboardJSON,
    TimeInterval,
    Transformation,
    VisualBeat,
    VisualIntent,
)


def _storyboard(objects=None):
    objects = objects or [
        MathematicalObject(
            object_id="circle",
            object_role="concept_visual",
            purpose="Show the quantity represented in the first beat",
            supports_beat_ids=["beat_001"],
            object_type="geometric_shape",
            content="",
            primitive="Circle",
            args={"radius": 1.0},
            bbox=BBox(x=0, y=0, w=2, h=2),
            interval=TimeInterval(t_in=0, t_out=10),
        )
    ]
    intent = VisualIntent(
        mathematical_objects=objects,
        transformations=[Transformation(
            transformation_type="fade_in",
            source_object_id="circle",
            target_object_id="circle",
            beat_id="beat_001",
            timing=2,
        )],
        beats=[VisualBeat(
            beat_id="beat_001",
            start_time=0,
            end_time=10,
            active_object_ids=["circle"],
        )],
    )
    scene = Scene(
        scene_id="scene_001",
        scene_index=0,
        concept_id="concept",
        narration="Narration for this scene.",
        visual_intent=intent,
        duration_estimate=10,
    )
    return StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(filename="test.pdf", page_count=1),
        configuration=Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate",
        ),
        concept_hierarchy=[],
        scenes=[scene],
    )


def test_measured_tts_duration_rescales_scene_timeline(tmp_path):
    orchestrator = Orchestrator(PipelineConfig(base_output_dir=str(tmp_path), tts_first=True))
    storyboard = _storyboard()
    tts = TTSOutput(
        audio_file_path=str(tmp_path / "scene.mp3"),
        duration_seconds=20.0,
        cost_usd=0.0,
        scene_id="scene_001",
    )
    orchestrator._apply_tts_timing(storyboard, [tts])
    scene = storyboard.scenes[0]
    assert scene.audio_duration_seconds == 20.0
    assert scene.duration_estimate == 20.0
    assert scene.visual_intent.beats[0].end_time == 20.0
    assert scene.visual_intent.transformations[0].timing == 4.0


def test_codegen_uses_short_measured_audio_without_visual_floor(monkeypatch):
    storyboard = _storyboard()
    scene = storyboard.scenes[0]
    scene_spec = SceneSpec(
        scene_id=scene.scene_id,
        narration=scene.narration,
        visual_intent=scene.visual_intent,
        duration_estimate=3.2,
        audio_duration_seconds=3.2,
    )

    class FakeModels:
        def __init__(self):
            self.system_instruction = ""

        def generate_content(self, **kwargs):
            self.system_instruction = kwargs["config"].system_instruction
            return SimpleNamespace(text=(
                "class Scene_001(Scene):\n"
                "    def construct(self):\n"
                "        self.wait(3.2)\n"
            ))

    agent = AnimationCodeGeneratorAgent.__new__(AnimationCodeGeneratorAgent)
    agent.strict = False
    agent.client = SimpleNamespace(models=FakeModels())
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    agent._generate_code(scene_spec, SharedContext())
    prompt = agent.client.models.system_instruction
    assert "measured audio is 3.2s" in prompt
    assert "about 3.2s" in prompt


def test_object_without_purpose_is_rejected():
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    storyboard = _storyboard()
    scene = storyboard.scenes[0]
    scene.visual_intent.mathematical_objects[0].purpose = ""
    errors = agent._visual_intent_constraint_errors(scene, scene.visual_intent)
    assert any("instructional purpose" in error for error in errors)


def test_fifteen_total_but_only_seven_concurrent_is_allowed():
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    storyboard = _storyboard()
    scene = storyboard.scenes[0]
    objects = []
    for index in range(15):
        start = float(index)
        objects.append(MathematicalObject(
            object_id=f"o{index}",
            purpose=f"Explain step {index}",
            supports_beat_ids=["beat_001"],
            object_type="geometric_shape",
            content="",
            primitive="Circle",
            args={"radius": 0.1},
            interval=TimeInterval(t_in=start, t_out=start + 1),
        ))
    scene.visual_intent.mathematical_objects = objects
    scene.visual_intent.transformations = []
    assert not agent._visual_intent_constraint_errors(scene, scene.visual_intent)


def test_more_than_seven_concurrent_objects_is_rejected():
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    storyboard = _storyboard()
    scene = storyboard.scenes[0]
    scene.visual_intent.mathematical_objects = [
        MathematicalObject(
            object_id=f"o{index}",
            purpose=f"Explain item {index}",
            supports_beat_ids=["beat_001"],
            object_type="geometric_shape",
            content="",
            primitive="Circle",
            args={"radius": 0.1},
            interval=TimeInterval(t_in=0, t_out=10),
        )
        for index in range(8)
    ]
    scene.visual_intent.transformations = []
    errors = agent._visual_intent_constraint_errors(scene, scene.visual_intent)
    assert any("concurrently visible" in error for error in errors)
