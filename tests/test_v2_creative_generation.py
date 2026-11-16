"""Compact creative-generation and automatic-instrumentation tests."""

import ast
import json

import pytest
from manim import Text

from src.pipeline_v2.generation import generation_request
from src.pipeline_v2.instrumentation import (
    deinstrument_scene_code,
    instrument_scene_code,
)
from src.pipeline_v2.models import (
    ArtifactCategory,
    ArtifactSpec,
    BeatComposition,
    LayoutRegion,
    LayoutTemplate,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    SourceFact,
    VisualContract,
)
from src.pipeline_v2.static_verifier import V2StaticVerifier
from src.pipeline_v2.tracked_scene import TrackedScene
from src.pipeline_v2.rendering import TrackedManimRenderer


RAW_SCENE = '''from manim import *
import numpy as np

class Draft(Scene):
    def construct(self):
        self.next_section("beat_1")
        title = Text("A visual idea")
        wave = ParametricFunction(lambda t: np.array([t, np.sin(t), 0]), t_range=[-3, 3])
        self.play(Write(title), Create(wave), run_time=9)
        self.next_section("beat_2")
        result = MathTex(r"y=\\sin(x)")
        self.play(Transform(title, result), run_time=1)
        self.wait(4)
'''


def _scene() -> NarratedScene:
    return NarratedScene(
        scene_id="scene_01",
        narration="First see the wave. Then identify its equation.",
        audio_duration_seconds=5.0,
        beats=[
            NarrationBeat(
                beat_id="beat_1", text="First see the wave.", source_fact_ids=["f1"],
                start_seconds=0.0, end_seconds=2.0,
            ),
            NarrationBeat(
                beat_id="beat_2", text="Then identify its equation.", source_fact_ids=["f1"],
                start_seconds=2.0, end_seconds=5.0,
            ),
        ],
    )


def _contract() -> VisualContract:
    artifact = ArtifactSpec(
        artifact_id="visual", category=ArtifactCategory.PRIMARY_VISUAL,
        role="wave", purpose="Show a changing wave", source_fact_ids=["f1"],
        beat_ids=["beat_1", "beat_2"], enter_beat="beat_1", exit_beat="beat_2",
        region=LayoutRegion.CONTENT_FULL,
    )
    return VisualContract(
        scene_id="scene_01", layout_template=LayoutTemplate.FULL_VISUAL,
        archetype="graph_plot", artifact_inventory=[artifact],
        composition_plan=[
            BeatComposition(beat_id="beat_1", active_artifact_ids=["visual"]),
            BeatComposition(beat_id="beat_2", active_artifact_ids=["visual"]),
        ],
        style_notes=["Use a dark background."],
    )


def test_generation_request_is_compact_and_free_of_tracking_bookkeeping():
    scene = _scene()
    system, prompt = generation_request(
        scene=scene,
        contract=_contract(),
        source_facts=[SourceFact(fact_id="f1", page_number=1, text="y = sin(x)")],
        class_name="V2Scene01",
        lesson_scene=LessonScene(
            scene_id="scene_01", title="The sine wave", description="Build a sine wave.",
            learning_goal="Connect the graph to its equation.", source_fact_ids=["f1"],
            new_visual_fact="The equation names the wave already drawn.",
            do_not_repeat=["drawing the sine wave from scratch"],
        ),
        visual_history=[{
            "scene_id": "scene_00",
            "visual_fact_already_shown": "A point traced the sine wave.",
        }],
    )

    assert len(system) < 1800
    assert len(prompt) < 4000
    assert "register_artifact" not in system + prompt
    assert "place_in_region" not in system + prompt
    assert "TrackedScene" not in system + prompt
    assert "artifact_inventory" not in system + prompt
    assert "ordinary Manim" in system
    payload = json.loads(prompt)
    assert payload["narration_beats"][1]["beat_id"] == "beat_2"
    assert payload["visual_history"][0]["scene_id"] == "scene_00"
    assert payload["required_new_visual_fact"].startswith("The equation")
    assert "drawing the sine wave from scratch" in payload["do_not_repeat"]


def test_retry_attempt_changes_generation_request_and_demands_fresh_layout():
    _, original = generation_request(
        scene=_scene(), contract=_contract(),
        source_facts=[SourceFact(fact_id="f1", page_number=1, text="y = sin(x)")],
        class_name="V2Scene01",
    )
    _, retry = generation_request(
        scene=_scene(), contract=_contract(),
        source_facts=[SourceFact(fact_id="f1", page_number=1, text="y = sin(x)")],
        class_name="V2Scene01", retry_attempt=2,
    )

    assert original != retry
    payload = json.loads(retry)
    assert payload["regeneration_attempt"] == 2
    assert "fresh visual composition" in payload["regeneration_instruction"]


def test_instrumentation_adds_tracking_and_retimes_each_narration_section():
    result = instrument_scene_code(
        RAW_SCENE, class_name="V2Scene01", scene=_scene(), contract=_contract(),
    )

    assert result.section_timing_applied
    assert {"title", "wave", "result"}.issubset(result.registered_names)
    assert "class V2Scene01(TrackedScene)" in result.code
    assert "configure_generated_scene" in result.code
    assert "auto_register('wave', wave)" in result.code
    assert "ParametricFunction" in result.code

    # Sections move towards their beat but are never sped up more than 1.5x:
    # 9 s of animation in a 2 s beat becomes 6 s, and 5 s in a 3 s beat 3.33 s.
    durations = _section_durations(result.code)
    assert abs(durations["beat_1"] - 9.0 / 1.5) < 0.001
    assert abs(durations["beat_2"] - 5.0 / 1.5) < 0.001
    assert result.capped_beats == ("beat_1", "beat_2")

    report = V2StaticVerifier(
        {"visual"}, {"beat_1", "beat_2"}, require_contract_tracking=False,
    ).verify(result.code)
    assert report.safe


def test_deinstrumentation_returns_plain_manim_for_targeted_repair():
    instrumented = instrument_scene_code(
        RAW_SCENE, class_name="V2Scene01", scene=_scene(), contract=_contract(),
    ).code

    plain = deinstrument_scene_code(instrumented)

    assert "TrackedScene" not in plain
    assert "auto_register" not in plain
    assert "configure_generated_scene" not in plain
    assert "class V2Scene01(Scene)" in plain
    assert "next_section('beat_1')" in plain


def test_section_retiming_accounts_for_literal_animation_loops():
    loop_scene = '''from manim import *

class Draft(Scene):
    def construct(self):
        self.next_section("beat_1")
        dot = Dot()
        for _ in range(4):
            self.play(dot.animate.shift(RIGHT), run_time=2)
'''
    one_beat = _scene().model_copy(update={
        "beats": [_scene().beats[0].model_copy(update={"end_seconds": 2.0})],
        "audio_duration_seconds": 2.0,
    })

    result = instrument_scene_code(
        loop_scene, class_name="V2Scene01", scene=one_beat, contract=_contract(),
    )
    tree = ast.parse(result.code)
    play = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "play"
    )
    run_time = next(item.value.value for item in play.keywords if item.arg == "run_time")

    # The four repetitions count as 8 s, so the 2 s beat needs the full capped
    # speed-up (a single 2 s play would have needed none).
    assert abs(float(run_time) - 2.0 / 1.5) < 0.001
    assert result.capped_beats == ("beat_1",)


def test_missing_section_markers_fall_back_to_clamped_global_timing():
    raw = '''from manim import *

class Draft(Scene):
    def construct(self):
        dot = Dot()
        self.play(FadeIn(dot), run_time=8)
        self.wait(2)
'''

    result = instrument_scene_code(
        raw, class_name="V2Scene01", scene=_scene(), contract=_contract(),
    )
    tree = ast.parse(result.code)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"play", "wait"}
    ]
    total = 0.0
    for call in calls:
        if call.func.attr == "play":
            total += float(next(item.value.value for item in call.keywords if item.arg == "run_time"))
        else:
            total += float(call.args[0].value)

    # 10 s against a 5 s narration: the whole construct gets the capped scale.
    assert not result.section_timing_applied
    assert abs(total - 10.0 / 1.5) < 0.001
    assert result.timing_scales == (("scene", 1 / 1.5),)
    assert result.capped_beats == ("scene",)


def test_tracked_scene_registers_assigned_mobjects_without_model_calls():
    tracked = TrackedScene()
    tracked._artifact_registry = {}
    tracked._beat_schedule = [("beat_1", 0.0, 2.0)]
    label = Text("hello")

    returned = tracked.auto_register("label", label)
    record = tracked._artifact_record(id(label), label)

    assert returned is label
    assert record is not None
    assert record["artifact_id"] == "label"
    assert record["category"] == "semantic_text"


@pytest.mark.integration
def test_instrumented_plain_manim_renders_with_named_trace_objects(tmp_path):
    raw = '''from manim import *

class Draft(Scene):
    def construct(self):
        self.next_section("beat_1")
        dot = Dot(LEFT * 2, color=BLUE)
        label = Text("A moving point", font_size=32).next_to(dot, RIGHT, buff=0.5)
        self.play(FadeIn(dot), Write(label))
'''
    scene = _scene().model_copy(update={
        "beats": [_scene().beats[0].model_copy(update={"end_seconds": 2.0})],
        "audio_duration_seconds": 2.0,
    })
    code = instrument_scene_code(
        raw, class_name="V2Scene01", scene=scene, contract=_contract(),
    ).code

    outcome = TrackedManimRenderer().render(
        code, _contract(), scene, tmp_path / "creative-render", timeout_seconds=120,
    )

    assert outcome.safe
    assert outcome.render_success, outcome.error
    # The play is stretched to at most 1.5 s; the closing hold fills the rest.
    assert abs(outcome.duration_seconds - 2.0) < 0.2
    assert not any(item.severity == "error" for item in outcome.findings)
    trace = (tmp_path / "creative-render" / "trace.jsonl").read_text(encoding="utf-8")
    assert '"event": "sync_wait"' in trace
    assert '"artifact_id": "dot"' in trace
    assert '"artifact_id": "label"' in trace


def _section_durations(code: str) -> dict[str, float]:
    tree = ast.parse(code)
    construct = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "construct"
    )
    active = ""
    durations: dict[str, float] = {}
    for statement in construct.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "next_section":
                active = call.args[0].value
                durations[active] = 0.0
                continue
            if not active or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr == "wait":
                durations[active] += float(call.args[0].value)
            elif call.func.attr == "play":
                run_time = next(item.value.value for item in call.keywords if item.arg == "run_time")
                durations[active] += float(run_time)
    return durations
