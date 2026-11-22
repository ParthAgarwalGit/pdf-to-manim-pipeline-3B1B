"""WP2: visual plans are repaired instead of discarded; the stub plan is
harmless; the new planner and codegen prompts carry the lesson context, the
beat schedule, the script's visual hints and the video's visual world; a
planned script that comes in short is sent back once. Keyless: every LLM
call is a fake."""

import json as _json

from src.agents.animation_generator import (
    AnimationCodeGeneratorAgent,
    _codegen_static_system_prompt,
)
from src.agents.scene_controller import SceneControllerAgent
from src.agents.script_architect import ScriptArchitectAgent, _visual_intent_static_prompt
from src.schemas.storyboard import (
    BBox,
    Configuration,
    MathematicalObject,
    Scene,
    SceneState,
    TimeInterval,
    Transformation,
    VisualIntent,
)
from src.schemas.video_script import Beat, VideoScript
from src.utils.visual_plan_repair import (
    demote_stub,
    is_stub_intent,
    max_concurrent,
    repair_visual_intent,
)
from tests.test_animation_generator import (
    _capture_animation_prompt,
    build_test_context,
    build_test_scene_spec,
)
from tests.test_lesson_plan import _config, _fake_unified, _keyed_agent, _plan, _segments
from tests.test_scene_controller import build_test_storyboard


def _mob(oid, primitive, t_in, t_out, x=0.0, y=0.0, **extra):
    types = {"Text": "text", "MathTex": "equation", "Axes": "axes"}
    args = {"text": "lbl"} if primitive == "Text" else {"tex": "x"} if primitive == "MathTex" else {}
    return MathematicalObject(
        object_id=oid, object_type=types.get(primitive, "geometric_shape"),
        content="lbl" if primitive in ("Text", "MathTex") else "",
        primitive=primitive, args=args, purpose=extra.pop("purpose", f"shows {oid}"),
        supports_beat_ids=extra.pop("beats", ["beat_001"]),
        bbox=BBox(x=x, y=y, w=2.0, h=1.0), interval=TimeInterval(t_in=t_in, t_out=t_out),
    )


def _intent(objs, transformations=(), carry=()):
    return VisualIntent(
        mathematical_objects=list(objs), transformations=list(transformations), beats=[],
        scene_state_in=SceneState(), scene_state_out=SceneState(active_object_ids=list(carry)),
    )


VECTOR_NARRATION = "We rotate the vector and watch its direction change."


def _architect_config() -> Configuration:
    return Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")


def _live_agent(monkeypatch) -> ScriptArchitectAgent:
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    agent.strict = False
    agent.client = object()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return agent


def _ids(intent):
    return [o.object_id for o in intent.mathematical_objects]


# ---------------------------------------------------------------------------
# repair_visual_intent
# ---------------------------------------------------------------------------

def test_valid_plan_is_untouched():
    intent = _intent([
        _mob("axes", "Axes", 0, 10),
        _mob("label_a", "Text", 0, 10, x=-4.0),
        _mob("label_b", "Text", 0, 10, x=4.0),
    ])
    before = intent.model_dump()
    notes = repair_visual_intent(intent, ["beat_001"], max_total=15, max_concurrent_allowed=7)
    assert notes == []
    assert intent.model_dump() == before


def test_over_concurrency_cuts_labels_before_drawings():
    drawings = [
        _mob("axes", "Axes", 0, 10), _mob("circle", "Circle", 0, 10),
        _mob("line", "Line", 0, 10), _mob("dot", "Dot", 0, 10),
    ]
    labels = [_mob(f"label_{i}", "Text", 0, 10) for i in range(6)]
    intent = _intent(drawings + labels)

    repair_visual_intent(intent, ["beat_001"], max_total=15, max_concurrent_allowed=7)

    objs = intent.mathematical_objects
    assert max_concurrent(objs) <= 7
    assert {"axes", "circle", "line", "dot"} <= set(_ids(intent))
    touched = [
        label for label in labels
        if all(label is not o for o in objs)
        or (label.interval.t_in, label.interval.t_out) != (0.0, 10.0)
    ]
    assert len(touched) >= 3


def test_retire_keeps_an_object_that_had_its_time():
    label = _mob("label", "Text", 0, 10)
    intent = _intent([label] + [_mob(f"c{i}", "Circle", 5, 10) for i in range(7)])

    repair_visual_intent(intent, ["beat_001"], max_total=15, max_concurrent_allowed=7)

    kept = [o for o in intent.mathematical_objects if o.object_id == "label"]
    assert len(kept) == 1
    assert kept[0].interval.t_out == 5.0


def test_total_cap_drops_lowest_priority():
    drawings = [_mob(f"d{i}", "Circle", 2 * i, 2 * i + 2) for i in range(3)]
    labels = [_mob(f"t{i}", "Text", 6 + 2 * i, 8 + 2 * i) for i in range(5)]
    intent = _intent(drawings + labels)

    repair_visual_intent(intent, ["beat_001"], max_total=4, max_concurrent_allowed=7)

    ids = _ids(intent)
    assert len(ids) == 4
    assert {"d0", "d1", "d2"} <= set(ids)


def test_last_drawing_is_never_removed():
    intent = _intent([_mob("circle", "Circle", 0, 10)] + [_mob(f"t{i}", "Text", 0, 10) for i in range(3)])

    repair_visual_intent(intent, ["beat_001"], max_total=1, max_concurrent_allowed=7)

    assert _ids(intent) == ["circle"]


def test_protected_object_survives():
    intent = _intent([_mob(f"c{i}", "Circle", 0, 10) for i in range(8)])

    repair_visual_intent(
        intent, ["beat_001"], max_total=15, max_concurrent_allowed=7, protected_ids=["c7"],
    )

    assert "c7" in _ids(intent)
    assert max_concurrent(intent.mathematical_objects) <= 7


def test_bookkeeping_is_filled():
    early = _mob(None, "Circle", 0, 5, purpose="", beats=["nope"])
    late = _mob(None, "Text", 6, 10, purpose="", beats=["nope"])
    intent = _intent([early, late])

    repair_visual_intent(
        intent, ["beat_001", "beat_002"], max_total=15, max_concurrent_allowed=7,
        beat_windows={"beat_001": (0, 5), "beat_002": (5, 10)},
    )

    for obj in intent.mathematical_objects:
        assert obj.object_id
        assert obj.purpose.strip()
    assert late.supports_beat_ids == ["beat_002"]


def test_dangling_references_are_dropped():
    intent = _intent(
        [
            _mob("axes", "Axes", 0, 10),
            _mob("keep_label", "Text", 0, 10, x=4.0),
            _mob("doomed", "Text", 0, 2, x=-4.0),
        ],
        transformations=[
            Transformation(transformation_type="highlight", source_object_id="doomed", timing=1.0),
            Transformation(transformation_type="morph", source_object_id="axes",
                           target_object_id="doomed", timing=1.5),
            Transformation(transformation_type="scale", source_object_id="axes", timing=2.0),
        ],
        carry=["axes", "doomed"],
    )

    repair_visual_intent(intent, ["beat_001"], max_total=2, max_concurrent_allowed=7)

    ids = set(_ids(intent))
    assert "doomed" not in ids
    for t in intent.transformations:
        assert t.source_object_id in ids
        assert not t.target_object_id or t.target_object_id in ids
    assert [t.transformation_type for t in intent.transformations] == ["scale"]
    assert all(i in ids for i in intent.scene_state_out.active_object_ids)
    assert intent.scene_state_out.active_object_ids == ["axes"]


# ---------------------------------------------------------------------------
# The stub plan
# ---------------------------------------------------------------------------

def test_stub_is_detected_and_demoted():
    stub = ScriptArchitectAgent.__new__(ScriptArchitectAgent)._generate_fallback_visual_intent(
        VECTOR_NARRATION
    )
    assert is_stub_intent(stub)

    demote_stub(stub)

    ids = _ids(stub)
    assert "intro_text" not in ids
    assert "archetype_visual" in ids
    assert stub.scene_state_out.active_object_ids == []
    assert not is_stub_intent(_intent([_mob("axes_main", "Axes", 0, 10)]))


def test_stub_objects_are_never_carried():
    stub = ScriptArchitectAgent.__new__(ScriptArchitectAgent)._generate_fallback_visual_intent(
        VECTOR_NARRATION
    )
    assert stub.scene_state_out.active_object_ids  # the stub does name its objects
    assert ScriptArchitectAgent._carried_objects_from_intent(stub) == []

    axes = _mob("axes_main", "Axes", 0, 10)
    designed = _intent([axes, _mob("label", "Text", 0, 10, x=4.0)], carry=["axes_main"])
    assert ScriptArchitectAgent._carried_objects_from_intent(designed) == [axes]


# ---------------------------------------------------------------------------
# _plan_visual_intents: repair before validating, mark the source
# ---------------------------------------------------------------------------

def test_planner_output_over_budget_is_repaired_not_replaced(monkeypatch):
    agent = _live_agent(monkeypatch)
    script = VideoScript(beats=[Beat(
        beat_id="beat_001", narration=" ".join(["word"] * 30), visual_hint="h",
        introduces_concepts=["c"], source_segment_id="seg_001",
    )])
    scene = agent._chunk_script_into_scenes(script)[0]
    beat_id = scene.source_beats[0].beat_id
    calls = []

    def fake_planner(scene, video_script, config, review_feedback, carried_objects=None):
        calls.append(scene.scene_id)
        return _intent(
            [_mob("axes_main", "Axes", 0, 10, beats=[beat_id])]
            + [_mob(f"label_{i}", "Text", 0, 10, x=-5.0 + i, beats=[beat_id]) for i in range(9)]
        )

    monkeypatch.setattr(agent, "_generate_visual_intent_for_chunk", fake_planner)

    agent._plan_visual_intents([scene], script, _architect_config(), [])

    assert len(calls) == 1
    assert scene.visual_plan_source == "repaired"
    assert "axes_main" in _ids(scene.visual_intent)
    assert not is_stub_intent(scene.visual_intent)
    assert agent._visual_intent_constraint_errors(scene, scene.visual_intent) == []


def test_stub_from_planner_is_marked_and_not_carried(monkeypatch):
    agent = _live_agent(monkeypatch)
    script = VideoScript(beats=[
        Beat(beat_id="beat_001", narration=VECTOR_NARRATION, visual_hint="a vector turns",
             introduces_concepts=["rotation"], source_segment_id="seg_001"),
        Beat(beat_id="beat_002", narration="The plane underneath stays exactly where it was.",
             visual_hint="the plane stays", introduces_concepts=["plane"], source_segment_id="seg_002"),
    ])
    scenes = agent._chunk_script_into_scenes(script)
    assert len(scenes) == 2
    first_id, second_id = scenes[0].scene_id, scenes[1].scene_id
    calls = []

    def fake_planner(scene, video_script, config, review_feedback, carried_objects=None):
        calls.append((scene.scene_id, carried_objects))
        if scene.scene_id == first_id:
            return agent._generate_fallback_visual_intent(scene.narration)
        return _intent([_mob("axes_main", "Axes", 0, 10, beats=[scene.source_beats[0].beat_id])])

    monkeypatch.setattr(agent, "_generate_visual_intent_for_chunk", fake_planner)

    agent._plan_visual_intents(scenes, script, _architect_config(), [])

    first = scenes[0]
    assert first.visual_plan_source == "fallback"
    ids = _ids(first.visual_intent)
    assert "intro_text" not in ids or ids == ["intro_text"]
    assert [scene_id for scene_id, _ in calls] == [first_id, second_id]
    assert calls[1][1] == []


def test_keyless_planning_is_unchanged(monkeypatch):
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    agent.strict = False
    agent.client = None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    script = VideoScript(beats=[Beat(
        beat_id="beat_001", narration=VECTOR_NARRATION, visual_hint="a vector turns",
        introduces_concepts=["rotation"], source_segment_id="seg_001",
    )])
    scenes = agent._chunk_script_into_scenes(script)

    agent._plan_visual_intents(scenes, script, _architect_config(), [])

    assert "intro_text" in _ids(scenes[0].visual_intent)
    assert scenes[0].visual_plan_source == "planner"


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

def _capture_planner_prompt(monkeypatch, script) -> str:
    captured = {}
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    agent.strict = False

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": _json.dumps({
                "objects": [{"object_type": "geometric_shape", "content": "",
                             "primitive": "Circle", "args": {"radius": 1.0}}],
                "transformations": [], "beats": [],
                "scene_state_in": {}, "scene_state_out": {},
                "emphasis_points": [],
            })})})]

    class _Completions:
        def create(self, *, messages, **_kw):
            captured["prompt"] = messages[0]["content"]
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    agent.client = _Client()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    scene = agent._chunk_script_into_scenes(script)[0]
    agent._generate_visual_intent_for_chunk(scene, script, _architect_config(), [])
    return captured["prompt"]


def test_planner_prompt_gets_lesson_context_and_schedule(monkeypatch):
    static = _visual_intent_static_prompt()
    beat = Beat(
        beat_id="beat_001", narration="Here are eight points scattered across the plane.",
        visual_hint="data_cloud appears", act_id="act_01", source_segment_id="seg_001",
    )

    planned = _capture_planner_prompt(monkeypatch, VideoScript(beats=[beat], plan=_plan()))
    assert planned.startswith(static)
    tail = planned[len(static):]
    assert "LESSON CONTEXT" in tail
    assert "VISUAL WORLD" in tail
    assert "BEAT SCHEDULE" in tail

    unplanned = _capture_planner_prompt(monkeypatch, VideoScript(beats=[beat]))
    assert unplanned.startswith(static)
    assert "BEAT SCHEDULE" in unplanned[len(static):]
    assert "LESSON CONTEXT" not in unplanned


def test_static_planner_prompt_documents_motion_shapes():
    static = _visual_intent_static_prompt()
    for needle in ("CLOSED VOCABULARY", '"transformation_type"', '"source_object_id"',
                   '"timing"', '"beat_goal"', "at most four words"):
        assert needle in static


# ---------------------------------------------------------------------------
# Scene controller
# ---------------------------------------------------------------------------

def test_scene_spec_carries_hints_and_source():
    beats = [
        Beat(beat_id="beat_001", narration="The circle grows.", visual_hint="the circle stretches"),
        Beat(beat_id="beat_002", narration="And then it rests.", visual_hint=""),
    ]
    scene = Scene(
        scene_id="scene_001", scene_index=0, concept_id="circle",
        narration="The circle grows. And then it rests.",
        visual_intent=_intent([_mob("axes_main", "Axes", 0, 10)]),
        duration_estimate=10.0, source_beats=beats, visual_plan_source="fallback",
    )

    spec = SceneControllerAgent()._scene_to_spec(scene)

    assert spec.visual_hints == ["beat_001: the circle stretches"]
    assert spec.plan_source == "fallback"


def test_shared_context_carries_visual_world():
    storyboard = build_test_storyboard(1)
    storyboard.configuration.visual_world = ["Motif data_cloud: dots; color BLUE_B"]

    context = SceneControllerAgent()._initialize_context(storyboard)

    assert context.visual_world == ["Motif data_cloud: dots; color BLUE_B"]


# ---------------------------------------------------------------------------
# Code generator prompt
# ---------------------------------------------------------------------------

def test_codegen_prompt_carries_hints_world_and_descriptions(monkeypatch):
    spec = build_test_scene_spec(
        scene_id="scene_001", narration="Watch this shape change.",
        math_objects=[{"object_id": "circle", "object_type": "geometric_shape", "content": "",
                       "primitive": "Circle", "args": {"radius": 1.0}}],
        transformations=[{"transformation_type": "scale", "source_object_id": "circle",
                          "beat_id": "beat_001", "timing": 0.5,
                          "parameters": {"description": "the circle stretches into an ellipse"}}],
    ).model_copy(update={"visual_hints": ["beat_001: the circle stretches"], "plan_source": "fallback"})
    context = build_test_context().model_copy(update={"visual_world": ["Motif data_cloud: dots"]})

    captured = _capture_animation_prompt(AnimationCodeGeneratorAgent, monkeypatch, spec, context)

    user, system = captured["user"], captured["system_raw"]
    assert "beat_001: the circle stretches" in user
    assert '"plan_source": "fallback"' in user
    assert "the circle stretches into an ellipse" in user
    assert "VISUAL WORLD (SAME IN EVERY SCENE)" in system
    assert "Motif data_cloud: dots" in system
    assert system.startswith(_codegen_static_system_prompt())


def test_codegen_static_prompt_has_the_new_rules():
    static = _codegen_static_system_prompt()
    for needle in ("FILL THE FRAME", "MOTION AND STAGING", "ON-SCREEN TEXT", "plan_source"):
        assert needle in static
    assert "hypotenuse" not in static


# ---------------------------------------------------------------------------
# Planned script: one length retry
# ---------------------------------------------------------------------------

def _planned_response(marker, *beats):
    """A planned-script response; ``beats`` are (act_id, word_count) pairs."""
    return {
        "title": "Best fit",
        "analogies": [],
        "beats": [
            {"beat_id": f"x{i}", "act_id": act_id,
             "narration": " ".join([f"{marker}{i}"] * words),
             "visual_hint": "the data_cloud shifts", "introduces_concepts": [],
             "references_prior_beats": [], "source_segment_id": "seg_001"}
            for i, (act_id, words) in enumerate(beats, start=1)
        ],
    }


def _narrations(response):
    return [b["narration"] for b in response["beats"]]


def test_short_planned_script_is_retried_once_and_extended(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    first = _planned_response("short", ("act_01", 25), ("act_02", 25))
    second = _planned_response("long", ("act_01", 20), ("act_01", 20), ("act_02", 28), ("act_02", 28))
    prompts = _fake_unified(monkeypatch, agent, [first, second])

    script = agent._generate_planned_video_script(_plan(), _segments(), _config(target_minutes=1.0), [], None)

    assert [b.narration for b in script.beats] == _narrations(second)
    assert len(prompts) == 2
    assert "YOUR PREVIOUS DRAFT" in prompts[1]
    assert "The script is" in prompts[1]
    assert first["beats"][0]["narration"] in prompts[1]


def test_planned_script_of_full_length_is_not_retried(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    full = _planned_response("full", ("act_01", 20), ("act_01", 20), ("act_02", 28), ("act_02", 28))
    prompts = _fake_unified(monkeypatch, agent, [full])

    script = agent._generate_planned_video_script(_plan(), _segments(), _config(), [], None)

    assert [b.narration for b in script.beats] == _narrations(full)
    assert len(prompts) == 1
    assert "YOUR PREVIOUS DRAFT" not in prompts[0]


def test_shorter_retry_keeps_the_first_draft(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    first = _planned_response("first", ("act_01", 30), ("act_02", 30))
    second = _planned_response("second", ("act_01", 20), ("act_02", 20))
    prompts = _fake_unified(monkeypatch, agent, [first, second])

    script = agent._generate_planned_video_script(_plan(), _segments(), _config(target_minutes=1.0), [], None)

    assert [b.narration for b in script.beats] == _narrations(first)
    assert len(prompts) == 2


def test_short_planned_script_is_kept_when_no_length_was_requested(monkeypatch):
    """With no requested length the word budget is a ceiling: a short script
    is the better outcome and is never sent back to be made longer."""
    agent = _keyed_agent(monkeypatch)
    short = _planned_response("short", ("act_01", 25), ("act_02", 25))
    prompts = _fake_unified(monkeypatch, agent, [short])

    script = agent._generate_planned_video_script(_plan(), _segments(), _config(), [], None)

    assert [b.narration for b in script.beats] == _narrations(short)
    assert len(prompts) == 1
    assert "a ceiling and not a target" in prompts[0]
    assert "do not come in short" not in prompts[0]
