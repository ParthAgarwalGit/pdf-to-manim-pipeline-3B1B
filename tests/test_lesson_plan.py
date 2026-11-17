"""WP1: the lesson plan, the script written from it, act-aware scenes and
plan-aware bridges. Keyless: every LLM call is a fake."""

import logging
from types import SimpleNamespace

import pytest

from src.agents.bridge_planner import BridgePlannerAgent
from src.agents.script_architect import ScriptArchitectAgent, ScriptArchitectInput
from src.agents.script_editor import ScriptEditorAgent
from src.orchestrator.pipeline import PipelineConfig
from src.orchestrator.revision import lesson_script_from_storyboard
from src.schemas.lesson_plan import LessonPlan
from src.schemas.ocr_output import BoundingBox, OCROutput, Page, PDFMetadata, TextBlock
from src.schemas.source_segment import SourceSegment
from src.schemas.storyboard import Configuration, PDFMetadataStoryboard, StoryboardJSON
from src.schemas.video_script import Beat, VideoScript
from src.utils.fingerprints import beat_fingerprint

PLAN = {
    "central_question": "Why does one direction carry most of a cloud of points?",
    "hook": "Here are eight points. One line through them matters more than any other.",
    "promise": "You will see why.", "takeaway": "The long axis is the answer.", "title": "T",
    "running_example": {"description": "eight points", "values": {"A": "[[3, 1], [1, 2]]"}},
    "motifs": [{"motif_id": "data_cloud", "what": "eight dots", "color": "BLUE_B", "stands_for": "the rows"}],
    "acts": [
        {"kind": "hook", "viewer_question": "Which line?", "key_idea": "One line fits best.",
         "visual": "dots", "source_segments": ["seg_001"], "words": 40},
        {"kind": "payoff", "viewer_question": "Why that one?", "key_idea": "It is the longest stretch.",
         "visual": "ellipse", "source_segments": ["seg_002"], "words": 60},
    ],
    "coverage": [{"segment_id": "seg_003", "treatment": "omitted", "reason": "A proof."}],
}
SEGMENT_IDS = ["seg_001", "seg_002", "seg_003"]


def _plan() -> LessonPlan:
    return LessonPlan.from_llm(PLAN, SEGMENT_IDS, 100)


def _config(**kw) -> Configuration:
    return Configuration(verbosity="medium", depth="intermediate", audience="undergraduate", **kw)


def _segments():
    texts = [
        "Eight points in the plane and the line that fits them best.",
        "The best line runs along the direction of greatest spread.",
        "Proof that the maximiser is the top singular vector.",
    ]
    return [
        SourceSegment(segment_id=sid, order=n, page_start=1, page_end=1, heading="H", text=text)
        for n, (sid, text) in enumerate(zip(SEGMENT_IDS, texts), start=1)
    ]


def _ocr() -> OCROutput:
    return OCROutput(
        pdf_metadata=PDFMetadata(filename="t.pdf", page_count=1, file_size_bytes=10),
        pages=[Page(page_number=1, text_blocks=[TextBlock(
            text="hello world", reading_order=0,
            bounding_box=BoundingBox(x=0, y=0, width=1, height=1),
        )])],
    )


def _script_response():
    return {
        "title": "Best fit",
        "analogies": [],
        "beats": [
            {"beat_id": "x1", "act_id": "act_01",
             "narration": "Here are eight points scattered across the plane.",
             "visual_hint": "data_cloud appears", "introduces_concepts": ["Best Fit Line"],
             "references_prior_beats": [], "source_segment_id": "seg_001", "role": "core", "depth": 0},
            {"beat_id": "x2", "act_id": "act_09",
             "narration": "One line through them fits better than every other line.",
             "visual_hint": "a line sweeps through the data_cloud", "introduces_concepts": ["best fit line"],
             "references_prior_beats": [], "source_segment_id": "seg_999"},
            {"beat_id": "x3", "act_id": "act_02",
             "narration": "That line runs along the longest stretch of the cloud.",
             "visual_hint": "an ellipse hugs the data_cloud", "introduces_concepts": ["long axis"],
             "references_prior_beats": ["x1"], "source_segment_id": "seg_002"},
        ],
    }


def _keyed_agent(monkeypatch) -> ScriptArchitectAgent:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return ScriptArchitectAgent()


def _fake_unified(monkeypatch, agent, responses):
    """Replace the strong-model call with a queue; returns the recorded prompts."""
    prompts = []
    queue = list(responses)

    def fake(prompt, max_tokens=8000):
        prompts.append(prompt)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(agent, "_complete_unified_script_json", fake)
    return prompts


def _sentinel_segmented(monkeypatch, agent):
    sentinel = VideoScript(beats=[Beat(beat_id="beat_001", narration="segmented", visual_hint="v")])
    monkeypatch.setattr(agent, "_generate_segmented_video_script", lambda *a, **kw: sentinel)
    return sentinel


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def test_beat_and_script_defaults_are_backward_compatible():
    beat = Beat(beat_id="b", narration="n", visual_hint="v")
    assert beat.act_id == ""
    assert VideoScript(beats=[beat]).plan is None
    old = (
        '{"title": "t", "beats": [{"beat_id": "beat_001", "narration": "n", '
        '"visual_hint": "v", "source_segment_id": "seg_001"}]}'
    )
    loaded = VideoScript.model_validate_json(old)
    assert loaded.plan is None and loaded.beats[0].act_id == ""


def test_act_id_does_not_change_beat_fingerprint():
    a = Beat(beat_id="b", narration="n", visual_hint="v", act_id="act_01")
    b = Beat(beat_id="b", narration="n", visual_hint="v", act_id="act_02")
    assert beat_fingerprint(a) == beat_fingerprint(b)


def test_script_with_plan_round_trips():
    script = VideoScript(beats=[Beat(beat_id="b", narration="n", visual_hint="v")], plan=_plan())
    loaded = VideoScript.model_validate_json(script.model_dump_json())
    assert loaded.plan == script.plan


def test_from_llm_completes_coverage_and_budget():
    plan = _plan()
    assert plan.act_ids() == ["act_01", "act_02"]
    assert plan.total_words() == 100
    assert sorted(c.segment_id for c in plan.coverage) == SEGMENT_IDS
    assert plan.treatment_of("seg_003") == "omitted"
    assert plan.planned_out_segment_ids() == ["seg_003"]


def test_from_llm_rejects_unusable_plans():
    with pytest.raises(ValueError):
        LessonPlan.from_llm({})
    with pytest.raises(ValueError):
        LessonPlan.from_llm({"central_question": "q", "acts": []})


def test_planned_script_path(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    prompts = _fake_unified(monkeypatch, agent, [PLAN, _script_response()])

    script = agent._generate_video_script(_ocr(), _config(), segments=_segments())

    assert isinstance(script.plan, LessonPlan)
    assert [b.beat_id for b in script.beats] == ["beat_001", "beat_002", "beat_003"]
    assert [b.act_id for b in script.beats] == ["act_01", "act_01", "act_02"]
    assert script.beats[1].source_segment_id == "seg_001"
    assert script.beats[2].references_prior_beats == ["beat_001"]
    assert sum("best_fit_line" in b.introduces_concepts for b in script.beats) == 1
    assert "[seg_001]" in prompts[0] and "ONE CENTRAL QUESTION" in prompts[0]
    assert "THE PLAN (binding)" in prompts[1]


def test_planning_failure_falls_back_to_segments(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    prompts = _fake_unified(monkeypatch, agent, [RuntimeError("planner down")])
    sentinel = _sentinel_segmented(monkeypatch, agent)

    assert agent._generate_video_script(_ocr(), _config(), segments=_segments()) is sentinel
    assert len(prompts) == 1


def test_lesson_plan_flag_off_skips_planning(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    calls = []

    def must_not_run(prompt, max_tokens=8000):
        calls.append(prompt)
        raise AssertionError("planning ran with lesson_plan=False")

    monkeypatch.setattr(agent, "_complete_unified_script_json", must_not_run)
    sentinel = _sentinel_segmented(monkeypatch, agent)

    result = agent._generate_video_script(_ocr(), _config(lesson_plan=False), segments=_segments())
    assert result is sentinel and calls == []


def test_long_source_skips_planning(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    monkeypatch.setattr(agent, "MAX_SOURCE_CHARS", 10)
    prompts = _fake_unified(monkeypatch, agent, [])
    sentinel = _sentinel_segmented(monkeypatch, agent)

    assert agent._generate_video_script(_ocr(), _config(), segments=_segments()) is sentinel
    assert prompts == []


def test_scenes_do_not_cross_acts():
    agent = ScriptArchitectAgent()
    planned = VideoScript(beats=[
        Beat(beat_id=f"beat_00{n}", narration=_words(10), visual_hint="v",
             source_segment_id="seg_001", act_id=act)
        for n, act in ((1, "act_01"), (2, "act_01"), (3, "act_02"))
    ])
    scenes = agent._chunk_script_into_scenes(planned)
    assert len(scenes) == 2
    assert all(len({b.act_id for b in s.source_beats}) == 1 for s in scenes)

    unplanned = VideoScript(beats=[
        Beat(beat_id=f"beat_00{n}", narration=_words(10), visual_hint="v", source_segment_id=seg)
        for n, seg in ((1, "seg_a"), (2, "seg_a"), (3, "seg_b"))
    ])
    scenes = agent._chunk_script_into_scenes(unplanned)
    assert [[b.source_segment_id for b in s.source_beats] for s in scenes] == [["seg_a", "seg_a"], ["seg_b"]]


def test_beat_runs_group_by_act():
    beats = [
        Beat(beat_id=f"beat_00{n}", narration="n", visual_hint="v", source_segment_id="seg_001", act_id=act)
        for n, act in ((1, "a"), (2, "a"), (3, "b"))
    ]
    runs = ScriptArchitectAgent._beat_runs(beats)
    assert [key for key, _ in runs] == ["a", "b"]
    assert [len(run) for _, run in runs] == [2, 1]


def test_smoothing_skipped_for_planned_script_within_budget(monkeypatch):
    agent = _keyed_agent(monkeypatch)
    _fake_unified(monkeypatch, agent, [PLAN, _script_response()])
    monkeypatch.setattr(agent, "_plan_visual_intents", lambda *a, **kw: None)

    def no_smoothing(*a, **kw):
        raise AssertionError("a planned script within budget must not be smoothed")

    monkeypatch.setattr(agent, "_smooth_narration", no_smoothing)

    storyboard = agent.execute(ScriptArchitectInput(
        ocr_output=_ocr(), configuration=_config(), segments=_segments(),
    ))

    assert agent.last_flow is None
    assert isinstance(agent.last_lesson_plan, LessonPlan)
    assert agent.last_lesson_plan.central_question == PLAN["central_question"]
    assert any("data_cloud" in line for line in storyboard.configuration.visual_world)
    assert agent.last_video_script.plan is not None


def test_coverage_log_respects_planned_omissions(caplog):
    agent = ScriptArchitectAgent()
    scenes = [
        SimpleNamespace(source_segment_ids=["seg_001"], source_beats=[]),
        SimpleNamespace(source_segment_ids=["seg_002"], source_beats=[]),
    ]

    def missing_warnings():
        return [r for r in caplog.records if "missing source segments" in r.getMessage()]

    with caplog.at_level(logging.WARNING, logger="src.agents.script_architect"):
        agent._log_segment_coverage(scenes, _segments(), plan=_plan())
    assert missing_warnings() == []

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.agents.script_architect"):
        agent._log_segment_coverage(scenes, _segments())
    assert missing_warnings()


def test_bridges_follow_the_plan():
    plan = _plan()
    beats, n = [], 1
    for act_id, segment_id in (("act_01", "seg_001"), ("act_02", "seg_002")):
        for _ in range(7):  # 7 x 45 words, about 105 s per act
            beats.append(Beat(beat_id=f"beat_{n:03d}", narration=_words(45), visual_hint="v",
                              source_segment_id=segment_id, act_id=act_id))
            n += 1
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return {"bridges": []}

    out = BridgePlannerAgent(llm=fake_llm).propose(VideoScript(beats=beats, plan=plan))

    bridges = [b for b in out.beats if b.role == "bridge"]
    assert bridges and bridges[0].narration == PLAN["hook"]
    assert out.beats[-1].role != "bridge"
    assert len(prompts) == 1
    assert "QUESTION THE NEXT PART ANSWERS: Why that one?" in prompts[0]
    assert "(opening)" not in prompts[0] and "- slot 0 (" not in prompts[0]
    assert out.plan is not None


def test_bridges_without_plan_are_unchanged():
    beats = [
        Beat(beat_id=f"beat_{n:03d}", narration=_words(45), visual_hint="x", source_segment_id="seg_0")
        for n in range(1, 13)  # 12 x 15 s = 180 s
    ]
    positions = BridgePlannerAgent(use_default_llm=False).positions(beats)
    assert positions[0] == (0, "opening")
    assert positions[-1] == (len(beats), "recap")


def test_script_editor_keeps_plan():
    script = VideoScript(beats=[Beat(beat_id="beat_001", narration="n", visual_hint="v")], plan=_plan())
    result = ScriptEditorAgent(use_default_llm=False).apply(script, [])
    assert result.script.plan == script.plan


def test_lesson_script_from_storyboard_keeps_plan():
    agent = ScriptArchitectAgent()
    script = VideoScript(beats=[Beat(
        beat_id="beat_001", narration=_words(20), visual_hint="v",
        introduces_concepts=["c"], source_segment_id="seg_001",
    )])
    scenes = agent._chunk_script_into_scenes(script)
    storyboard = StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(filename="n.pdf", page_count=1),
        configuration=_config(),
        concept_hierarchy=agent._concept_hierarchy_from_ledger(script.concept_ledger()),
        scenes=scenes,
    )
    plan = _plan()
    assert lesson_script_from_storyboard(storyboard, plan=plan).plan == plan


def test_pipeline_config_carries_lesson_plan_flag():
    assert PipelineConfig().to_storyboard_config().lesson_plan is True
    assert PipelineConfig(lesson_plan=False).to_storyboard_config().lesson_plan is False


def test_planned_narration_never_contains_motif_ids(monkeypatch):
    """A motif id the writer leaked into the narration is spoken as words:
    a text-to-speech voice would read ``best_fit_line`` with its underscores."""
    agent = _keyed_agent(monkeypatch)
    response = _script_response()
    response["beats"][0]["narration"] = (
        "Every point lands on best_fit_line, and the sweep settles into second_direction."
    )
    _fake_unified(monkeypatch, agent, [PLAN, response])

    script = agent._generate_video_script(_ocr(), _config(), segments=_segments())

    assert script.beats[0].narration == (
        "Every point lands on best fit line, and the sweep settles into second direction."
    )
    assert all("_" not in beat.narration for beat in script.beats)


def test_default_budget_is_a_ceiling_not_a_target(monkeypatch):
    """No requested length: the plan keeps its own (smaller) word counts and
    the planner is told the budget is a ceiling."""
    agent = _keyed_agent(monkeypatch)
    prompts = _fake_unified(monkeypatch, agent, [PLAN])

    plan = agent._generate_lesson_plan(_segments(), _config(), [], None)

    assert plan.total_words() == 100  # the planner's 40 + 60, not scaled up to the default
    assert "nobody asked for a particular length" in prompts[0]
    assert "hard budget" not in prompts[0]


def test_requested_length_is_a_target(monkeypatch):
    """A requested length is binding: act words are rescaled to the budget."""
    agent = _keyed_agent(monkeypatch)
    prompts = _fake_unified(monkeypatch, agent, [PLAN])
    config = _config(target_minutes=1.0)

    plan = agent._generate_lesson_plan(_segments(), config, [], None)

    assert plan.total_words() == agent._plan_word_budget(config, _segments())
    assert "hard budget" in prompts[0]


def test_cap_words_only_scales_down():
    plan = LessonPlan.from_llm(PLAN, ["seg_001", "seg_002", "seg_003"])
    assert plan.total_words() == 100
    plan.cap_words(400)
    assert plan.total_words() == 100
    plan.cap_words(60)
    assert plan.total_words() == 60
