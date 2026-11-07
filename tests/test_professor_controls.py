"""P3: professor brief -> directives, feedback -> edit ops, script editor,
directive compliance, and feedback-driven revisions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.directive_compliance import (
    apply_scope, enforce_script_directives, length_check, notation_check, script_violations,
)
from src.agents.professor_input_interpreter import (
    ProfessorInputInterpreter, beat_at, build_timeline,
)
from src.agents.script_editor import ScriptEditorAgent
from src.orchestrator.revision import change_log, estimate_revision, load_base_run
from src.schemas.directives import ConceptDirective, NotationDirective, ProfessorDirectives
from src.schemas.edit_ops import (
    AddAnalogy, Condense, Expand, FixVisual, Pace, PresenterOp, Pronounce, Remove,
    ReplaceAnalogy, ReviseBeat, RewriteBeat, SetDensity, parse_ops,
)
from src.schemas.source_segment import SourceSegment
from src.schemas.video_script import Analogy, Beat, VideoScript
from tests.test_revision import _base_run, _beat, _orchestrator, _script


def _lesson() -> VideoScript:
    return VideoScript(
        title="Circuits",
        analogies=[Analogy(analogy_id="analogy_pizza", label="pizza slices")],
        beats=[
            Beat(beat_id="beat_001", narration="Voltage pushes charge around a circuit.",
                 visual_hint="battery", introduces_concepts=["voltage"]),
            Beat(beat_id="beat_002", narration="Think of a pizza cut into slices of charge.",
                 visual_hint="pizza", role="analogy", analogy_id="analogy_pizza", depth=1),
            Beat(beat_id="beat_003", narration="Current is the flow; the voltage sets its pace.",
                 visual_hint="arrows", introduces_concepts=["current"],
                 references_prior_beats=["beat_001"]),
            Beat(beat_id="beat_004", narration="Each pizza slice again stands for charge per second.",
                 visual_hint="pizza", role="analogy", analogy_id="analogy_pizza", depth=1),
            Beat(beat_id="beat_005", narration="An optional aside on history.",
                 visual_hint="portrait", depth=2),
        ],
    )


class FakeLLM:
    """Returns canned JSON by keyword; records every prompt."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


# ── edit ops ────────────────────────────────────────────────────────────

class TestEditOps:
    def test_parse_from_yaml_shaped_dicts(self):
        ops = parse_ops({"ops": [
            {"op": "rewrite_beat", "beat_id": "beat_001", "text": "Exact words."},
            {"op": "remove", "analogy_id": "analogy_pizza"},
            {"op": "pace", "beat_id": "beat_003", "pause_seconds": 2},
        ]})
        assert [type(o) for o in ops] == [RewriteBeat, Remove, Pace]

    def test_remove_needs_exactly_one_target(self):
        with pytest.raises(ValueError):
            Remove()


class TestScriptEditorDeterministic:
    def test_rewrite_locks_professor_text(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(
            _lesson(), [RewriteBeat(beat_id="beat_001", text="My own sentence.")],
        )
        beat = result.script.beats[0]
        assert (beat.narration, beat.author, beat.locked) == ("My own sentence.", "professor", True)
        assert result.diff[0].startswith("~ beat_001")

    def test_pace_pronounce_and_visual_fix(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(_lesson(), [
            Pace(beat_id="beat_003", pause_seconds=2.0),
            Pronounce(word="Euler", say_as="Oiler"),
            FixVisual(beat_id="beat_003", instruction="Move the label off the arrow."),
        ])
        assert result.script.beats[2].pause_after == 2.0
        assert result.script.pronunciations == {"Euler": "Oiler"}
        assert result.visual_fixes == {"beat_003": "Move the label off the arrow."}
        assert result.ok

    def test_remove_rejects_a_concept_later_beats_need(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(_lesson(), [Remove(beat_id="beat_001")])
        assert not result.applied
        assert "voltage" in result.errors[0] and "beat_003" in result.errors[0]
        assert [b.beat_id for b in result.script.beats][0] == "beat_001"

    def test_remove_analogy_drops_callbacks_and_records_omissions(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(_lesson(), [Remove(analogy_id="analogy_pizza")])
        ids = [b.beat_id for b in result.script.beats]
        assert ids == ["beat_001", "beat_003", "beat_005"]
        assert result.script.analogies == []
        assert {o["beat_id"] for o in result.omissions} == {"beat_002", "beat_004"}
        assert all(o["reason"] == "professor" for o in result.omissions)
        assert any("not smoothed" in w for w in result.warnings)

    def test_locked_beats_are_never_revised(self):
        lesson = _lesson()
        lesson.beats[2].locked = True
        result = ScriptEditorAgent(llm=FakeLLM({"narration": "x"})).apply(
            lesson, [ReviseBeat(beat_id="beat_003", instruction="simplify")],
        )
        assert "professor-approved" in result.errors[0]
        assert result.script.beats[2].narration == lesson.beats[2].narration

    def test_generative_ops_need_an_llm(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(
            _lesson(), [ReviseBeat(beat_id="beat_003", instruction="simplify")],
        )
        assert "needs an LLM" in result.errors[0]

    def test_density_essential_without_analogies(self):
        lesson = _lesson()
        lesson.beats[4].locked = True  # professor-approved aside survives
        result = ScriptEditorAgent(use_default_llm=False).apply(
            lesson, [SetDensity(level="essential", analogies="off")],
        )
        assert [b.beat_id for b in result.script.beats] == ["beat_001", "beat_003", "beat_005"]
        assert any("locked" in w for w in result.warnings)

    def test_presenter_bridge_lifecycle(self):
        editor = ScriptEditorAgent(use_default_llm=False)
        inserted = editor.apply(_lesson(), [PresenterOp(action="insert", beat_id="beat_001", text="Here's the real question.")])
        bridge = inserted.script.beats[1]
        assert (bridge.role, bridge.locked, bridge.beat_id) == ("bridge", True, "beat_006")
        removed = editor.apply(inserted.script, [PresenterOp(action="remove", beat_id="beat_006")])
        assert "beat_006" not in [b.beat_id for b in removed.script.beats]


class TestScriptEditorGenerative:
    def test_revise_beat_uses_neighbours_as_context(self):
        llm = FakeLLM({"narration": "Current is the flow of charge.", "visual_hint": "flow"})
        result = ScriptEditorAgent(llm=llm).apply(_lesson(), [ReviseBeat(beat_id="beat_003", instruction="simpler")])
        assert result.script.beats[2].narration == "Current is the flow of charge."
        assert "PREVIOUS BEAT: Think of a pizza" in llm.prompts[0]

    def test_replace_analogy_keeps_quoted_phrases_and_marks_professor_origin(self):
        llm = FakeLLM(
            {"label": "traffic", "beats": [  # first answer drops the quoted phrase
                {"beat_id": "beat_002", "narration": "Cars on a road."},
                {"beat_id": "beat_004", "narration": "More cars."},
            ]},
            {"label": "traffic on a highway", "beats": [
                {"beat_id": "beat_002", "narration": "Picture cars as electrons on a highway."},
                {"beat_id": "beat_004", "narration": "Lanes are wires: cars as electrons again."},
            ]},
        )
        result = ScriptEditorAgent(llm=llm).apply(_lesson(), [ReplaceAnalogy(
            analogy_id="analogy_pizza", brief='Traffic: "cars as electrons", lanes are wires.',
        )])
        assert "dropped the phrases" in llm.prompts[1]
        assert "cars as electrons" in result.script.beats[1].narration
        analogy = result.script.analogies[0]
        assert (analogy.origin, analogy.professor_brief) == ("professor", 'Traffic: "cars as electrons", lanes are wires.')

    def test_add_expand_condense(self):
        llm = FakeLLM(
            {"label": "water pipes", "narration": "Voltage is like water pressure.", "visual_hint": "pipes",
             "mapping": {"pressure": "voltage"}},
            {"beats": [{"narration": "A worked example with 9 volts.", "visual_hint": "numbers"}]},
            {"beats": [{"narration": "Both ideas in one line.", "visual_hint": "merge"}]},
        )
        result = ScriptEditorAgent(llm=llm).apply(_lesson(), [
            AddAnalogy(concept="voltage", brief="water pipes"),
            Expand(target="current", levels=1, focus="example"),
            Condense(beat_ids=["beat_004", "beat_005"]),
        ])
        ids = [b.beat_id for b in result.script.beats]
        assert ids[:2] == ["beat_001", "beat_006"]            # analogy right after voltage
        added = result.script.beats[1]
        assert (added.role, added.analogy_id) == ("analogy", "analogy_water_pipes")
        expanded = result.script.beats[ids.index("beat_003") + 1]
        assert (expanded.role, expanded.depth) == ("example", 1)
        assert "beat_005" not in ids                           # condensed away
        assert result.ok, result.errors

    def test_seam_is_repaired_around_a_cut(self):
        llm = FakeLLM({"before": "Voltage pushes charge.", "after": "So current flows at that pace."})
        result = ScriptEditorAgent(llm=llm).apply(_lesson(), [Remove(beat_id="beat_002")])
        assert result.script.beats[1].narration == "So current flows at that pace."
        assert "this beats now meet" not in llm.prompts[0] and "now meet" in llm.prompts[0]


# ── interpreter ─────────────────────────────────────────────────────────

BRIEF = (
    "My students are 2nd-year EE and struggle with phasors. Use the rotating-arm picture, "
    "not the clock analogy. Skip 4.3. Spend more time on resonance. Under 10 minutes. "
    "I write j, not i."
)


class TestBriefInterpretation:
    def test_rules_cover_the_spec_example(self):
        d = ProfessorInputInterpreter(use_default_llm=False).interpret_brief(BRIEF)
        assert d.scope.skip_sections == ["4.3"]
        assert d.global_.target_minutes == 10 and d.global_.audience == "undergraduate"
        assert [(n.use, n.instead_of) for n in d.notation] == [("j", "i")]
        assert d.avoided_analogies() == ["clock"]
        assert any(c.concept == "resonance" and c.depth == "deep" for c in d.concepts)
        assert any("struggle with phasors" in n for n in d.notes)  # never dropped
        readback = d.readback()
        assert "Leave out section(s): 4.3" in readback and "Never use the clock analogy." in readback

    def test_llm_reading_is_validated(self):
        llm = FakeLLM({"scope": {"skip_sections": ["2"]}, "global": {"target_minutes": 5}, "notes": ["Be kind."]})
        d = ProfessorInputInterpreter(llm=llm).interpret_brief("Skip 2, five minutes, be kind.")
        assert d.scope.skip_sections == ["2"] and d.global_.target_minutes == 5
        assert d.source_brief == "Skip 2, five minutes, be kind."

    def test_unmappable_brief_is_kept_verbatim(self):
        d = ProfessorInputInterpreter(use_default_llm=False).interpret_brief("Make it sparkle.")
        assert d.notes == ["Make it sparkle."]


class TestFeedbackInterpretation:
    def _timeline(self):
        scene = SimpleNamespace(
            scene_id="scene_001", audio_duration_seconds=40.0, duration_estimate=40.0,
            beat_timings={"beat_001": [0, 10], "beat_002": [10, 20], "beat_003": [20, 30], "beat_004": [30, 40]},
            source_beats=[],
        )
        return build_timeline(SimpleNamespace(scenes=[scene]))

    def test_rules_anchor_and_classify(self):
        plan = ProfessorInputInterpreter(use_default_llm=False).interpret_feedback(
            "At 0:12 the label overlaps the curve. 'Euler' is pronounced Oiler. "
            "Replace the pizza analogy with a traffic analogy: cars are electrons. "
            "At 0:31 this is too fast. Make it better.",
            _lesson(), self._timeline(),
        )
        kinds = [(type(op).__name__, getattr(op, "beat_id", None) or getattr(op, "analogy_id", None)) for op in plan.ops]
        assert kinds == [
            ("FixVisual", "beat_002"), ("Pronounce", None),
            ("ReplaceAnalogy", "analogy_pizza"), ("Pace", "beat_004"),
        ]
        assert plan.notes == ["Make it better."]
        assert plan.classes == ["audio", "content", "pacing", "visual"]

    def test_timeline_from_word_counts_and_delivery_report(self):
        beats = [Beat(beat_id="b1", narration="one two", visual_hint="x"),
                 Beat(beat_id="b2", narration="three four five six", visual_hint="y")]
        scene = SimpleNamespace(scene_id="scene_001", audio_duration_seconds=6.0, duration_estimate=6.0,
                                beat_timings={}, source_beats=beats)
        timeline = build_timeline(SimpleNamespace(scenes=[scene]),
                                  {"scenes": [{"scene_id": "scene_001", "start": 3.0, "end": 9.0}]})
        assert beat_at(timeline, 4.0) == "b1" and beat_at(timeline, 6.0) == "b2"


# ── compliance ──────────────────────────────────────────────────────────

class TestCompliance:
    def _directives(self):
        return ProfessorDirectives(
            concepts=[ConceptDirective(concept="", analogy_brief="the rotating-arm picture", avoid_analogies=["clock"])],
            notation=[NotationDirective(use="j", instead_of="i")],
        )

    def test_rejected_analogy_is_repaired_in_the_offending_beat_only(self):
        lesson = _lesson()
        lesson.beats[2].narration = "Like a clock hand, the current sweeps around."
        llm = FakeLLM({"narration": "Like a rotating arm, the current sweeps around.", "visual_hint": "arm"})
        script, checks = enforce_script_directives(lesson, self._directives(), ScriptEditorAgent(llm=llm))
        assert script.beats[2].narration.startswith("Like a rotating arm")
        assert "rotating-arm picture" in llm.prompts[0]
        assert [c.passed for c in checks] == [True]
        assert script.beats[0].narration == lesson.beats[0].narration

    def test_unrepairable_violation_is_reported(self):
        lesson = _lesson()
        lesson.beats[2].narration = "The clock again."
        script, checks = enforce_script_directives(lesson, self._directives(), ScriptEditorAgent(use_default_llm=False))
        assert checks[0].passed is False and "beat_003" in checks[0].message

    def test_scope_length_and_notation(self):
        segments = [
            SourceSegment(segment_id="s1", order=0, page_start=1, page_end=1, heading="4.2 Phasors", text="..."),
            SourceSegment(segment_id="s2", order=1, page_start=2, page_end=2, heading="4.3 Bode plots", text="..."),
        ]
        kept, omissions, checks = apply_scope(segments, ProfessorDirectives(scope={"skip_sections": ["4.3"]}))
        assert [s.segment_id for s in kept] == ["s1"] and omissions[0]["segment_id"] == "s2"
        assert checks[0].passed
        storyboard = SimpleNamespace(scenes=[SimpleNamespace(audio_duration_seconds=400.0, duration_estimate=400.0)])
        assert length_check(storyboard, 5).passed is False
        code = [SimpleNamespace(scene_id="scene_002", code='t = MathTex(r"z = a + i b")\nu = MathTex(r"\\pi")')]
        notation = notation_check(code, self._directives())
        assert notation[0].passed is False and "scene_002" in notation[0].message
        assert notation_check([SimpleNamespace(scene_id="s", code='MathTex(r"\\sin \\pi + j")')], self._directives())[0].passed


# ── end to end ──────────────────────────────────────────────────────────

class TestFeedbackRevisions:
    def test_visual_fix_regenerates_code_but_keeps_audio(self, tmp_path):
        script = _script()
        run_dir, _ = _base_run(tmp_path, script)
        base = load_base_run(run_dir)
        target = base.storyboard.scenes[1]
        beat_id = target.source_beats[0].beat_id
        result = ScriptEditorAgent(use_default_llm=False).apply(
            base.script, [FixVisual(beat_id=beat_id, instruction="Put the label above the arrow.")],
        )
        orchestrator = _orchestrator(tmp_path)
        seen = {}
        original = orchestrator._run_scene_controller

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        orchestrator._run_scene_controller = spy
        manifest = orchestrator.run_revision(str(run_dir), result.script, visual_fixes=result.visual_fixes,
                                             ops_applied=result.applied)
        assert orchestrator.media.generated == [target.scene_id]
        assert orchestrator.media.narrated == []
        assert "Put the label above the arrow." in seen["initial_feedback"][target.scene_id][0]
        report = json.loads((Path(manifest.output_directory) / "revision.json").read_text("utf-8"))
        assert report["scenes"]["visual_fixes"] == [target.scene_id]
        assert report["ops"][0]["op"] == "fix_visual"
        assert report["change_log"].startswith("Changed: scene 2.")

    def test_pronunciation_renarrates_only_scenes_with_the_word(self, tmp_path):
        script = _script()
        script.beats[0].narration = "Euler first noticed this. " + script.beats[0].narration
        run_dir, manifest = _base_run(tmp_path, script)
        base = load_base_run(run_dir)
        result = ScriptEditorAgent(use_default_llm=False).apply(base.script, [Pronounce(word="Euler", say_as="Oiler")])
        estimate = estimate_revision(base, result.script)
        assert estimate["scenes_renarrated"] == 1 and estimate["scenes_rebuilt"] == 0

        orchestrator = _orchestrator(tmp_path)
        orchestrator.run_revision(str(run_dir), result.script)
        assert orchestrator.media.generated == [] and orchestrator.media.rendered == []
        assert len(orchestrator.media.narrated) == 1
        audio = next(Path(run_dir / "rev_002" / "audio").rglob("*.mp3")).read_text()
        assert "Oiler" in audio and "Euler" not in audio

    def test_change_log_wording(self):
        storyboard = SimpleNamespace(scenes=[SimpleNamespace(scene_id=f"scene_{i:03d}") for i in range(1, 7)])
        text = change_log(storyboard, ["scene_004", "scene_005"], [], {"scene_001": "a", "scene_002": "b"})
        assert text == "Changed: scenes 4-5. Reused unchanged: 2 scenes."


class TestBriefInPipeline:
    def test_directives_reach_the_architect_and_reports(self, tmp_path):
        directives = ProfessorInputInterpreter(use_default_llm=False).interpret_brief(
            "Never mind the clock. Under 1 minutes. I write j, not i."
        )
        captured = {}
        run_dir, manifest = _base_run_with_directives(tmp_path, _script(), directives, spy_target=captured)
        assert captured["directives"] is directives
        report = json.loads((run_dir / "brief_interpretation.json").read_text("utf-8"))
        assert report["global"]["target_minutes"] == 1
        checks = json.loads((run_dir / "directive_checks.json").read_text("utf-8"))
        assert any("Under 1 minutes" in c["measured"]["directive"] and c["passed"] is False for c in checks)


def _base_run_with_directives(tmp_path, script, directives, spy_target):
    """Like tests.test_revision._base_run, with a brief's directives."""
    from src.schemas.storyboard import Configuration, PDFMetadataStoryboard, StoryboardJSON

    orchestrator = _orchestrator(tmp_path)
    ref = SimpleNamespace(filename="notes.pdf")
    orchestrator._run_ingestion = lambda *a, **k: ref
    orchestrator._run_validation = lambda *a, **k: ref
    orchestrator._run_vision_ocr = lambda *a, **k: None
    orchestrator._run_document_segmenter = lambda *a, **k: []

    def fake_architect(ocr_output, structured_logger, source_segments=None):
        spy_target["directives"] = orchestrator._directives
        agent = orchestrator.script_architect_agent
        planned = script.model_copy(deep=True)
        scenes = agent._chunk_script_into_scenes(planned)
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")
        agent._plan_visual_intents(scenes, planned, config, [])
        agent._renumber_scenes(scenes)
        agent.last_video_script = planned
        return StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="notes.pdf", page_count=1),
            configuration=config,
            concept_hierarchy=agent._concept_hierarchy_from_ledger(planned.concept_ledger()),
            scenes=scenes,
        )

    orchestrator._run_script_architect = fake_architect
    manifest = orchestrator.run_pipeline("notes.pdf", event_source="local", directives=directives)
    return Path(manifest.output_directory), manifest
