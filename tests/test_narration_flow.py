"""Whole-script narration flow: repetition metric, smoothing pass, bridge echo guard.

No external API is called: LLMs are fakes returning canned JSON.
"""

import json
from types import SimpleNamespace

from src.agents.script_architect import ScriptArchitectAgent
from src.schemas.video_script import Beat, VideoScript
from src.utils.narration_flow import apply_smoothing, overlaps, repetition, smoothing_prompt

DEFINE = "Think of a matrix as a rubber sheet. First we rotate it, then stretch it, then rotate it again."
RESTATE = "First we rotate it, then stretch it, then rotate it again. The top direction gives the best fit line."
FRESH = "Keep only the biggest stretch and you get the best rank one approximation of the data."


def _beat(n, text, concepts=(), **extra):
    return Beat(beat_id=f"beat_{n:03d}", narration=text, visual_hint=f"visual {n}",
                introduces_concepts=list(concepts), source_segment_id="seg_001", **extra)


def _script(*beats):
    return VideoScript(title="SVD", beats=list(beats))


class TestRepetition:
    def test_flags_a_sentence_said_before(self):
        rep = repetition([DEFINE, RESTATE, FRESH])
        assert [r["beat"] for r in rep["repeats"]] == [1]
        assert rep["repeats"][0]["sentence"].startswith("First we rotate it")
        assert 0 < rep["ratio"] < 0.5

    def test_fresh_script_has_no_repeats(self):
        assert repetition([DEFINE, FRESH])["repeated_words"] == 0

    def test_overlap_share(self):
        assert overlaps("First we rotate it, then stretch it, then rotate it again.", DEFINE) == 1.0
        assert 0.2 < overlaps(RESTATE, DEFINE) < 0.5  # half restated, half new
        assert overlaps(FRESH, DEFINE) == 0.0


class TestApplySmoothing:
    def test_accepts_a_rewrite_that_cuts_the_repeat(self):
        script = _script(_beat(1, DEFINE, ["svd"]), _beat(2, RESTATE, ["best_fit_line"]))
        data = {"beats": [
            {"beat_id": "beat_001", "narration": DEFINE},
            {"beat_id": "beat_002", "narration": "The direction it stretches most gives the best fit line."},
        ]}
        out, report = apply_smoothing(script, data)
        assert report["applied"] is True
        assert report["after"]["repeated_words"] == 0 < report["before"]["repeated_words"]
        assert out.beats[1].narration.startswith("The direction it stretches most")
        assert out.beats[1].visual_hint == "visual 2"

    def test_drops_a_pure_repeat_but_never_a_concepts_only_home(self):
        script = _script(
            _beat(1, DEFINE, ["svd"]),
            _beat(2, RESTATE, ["svd"]),           # only repeats: may go
            _beat(3, RESTATE + " Again.", ["low_rank"], references_prior_beats=["beat_002"]),
        )
        data = {"beats": [
            {"beat_id": "beat_001", "narration": DEFINE},
            {"beat_id": "beat_002", "drop": True},
            {"beat_id": "beat_003", "drop": True},  # sole home of low_rank: stays
        ]}
        out, report = apply_smoothing(script, data)
        assert report["applied"] and report["dropped_beats"] == ["beat_002"]
        assert [b.beat_id for b in out.beats] == ["beat_001", "beat_003"]
        assert out.beats[1].references_prior_beats == []

    def test_rejects_a_longer_rewrite(self):
        script = _script(_beat(1, DEFINE), _beat(2, FRESH))
        longer = FRESH + " And that is a lot more words than before, really quite a few more."
        out, report = apply_smoothing(script, {"beats": [{"beat_id": "beat_002", "narration": longer}]})
        assert out is script and report["applied"] is False

    def test_locked_beats_and_bad_responses_keep_the_draft(self):
        script = _script(_beat(1, DEFINE, locked=True), _beat(2, FRESH))
        out, report = apply_smoothing(script, {"beats": [{"beat_id": "beat_001", "narration": "Short new text here ok."}]})
        assert out.beats[0].narration == DEFINE
        out, report = apply_smoothing(script, {"nope": 1})
        assert out is script and report["applied"] is False

    def test_prompt_lists_beats_budget_and_rules(self):
        beats = [_beat(1, DEFINE), _beat(2, FRESH, locked=True)]
        prompt = smoothing_prompt(beats, 700, "STYLE RULES", ["Say singular values."])
        assert "[beat_001]" in prompt and "[beat_002] LOCKED" in prompt
        assert "at most 700 words" in prompt and "Say singular values." in prompt
        assert "STYLE RULES" in prompt and "json" in prompt.lower()


class FakeClient:
    def __init__(self, payload):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.payload = payload

    def create(self, model, messages, response_format):
        self.prompts.append(messages[0]["content"])
        content = json.dumps(self.payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _architect(client):
    from src.agents.script_architect import ScriptArchitectAgent

    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    agent.strict, agent.client = False, client
    return agent


class TestScriptArchitectFlow:
    def _config(self, **kw):
        from src.schemas.storyboard import Configuration

        return Configuration(verbosity="medium", depth="intermediate", audience="undergraduate", **kw)

    def test_smoothing_runs_once_over_the_whole_script(self):
        client = FakeClient({"beats": [
            {"beat_id": "beat_001", "narration": DEFINE},
            {"beat_id": "beat_002", "narration": "The direction it stretches most gives the best fit line."},
        ]})
        script = _script(_beat(1, DEFINE, ["svd"]), _beat(2, RESTATE, ["best_fit_line"]))
        out, report = _architect(client)._smooth_narration(script, self._config())
        assert len(client.prompts) == 1 and "[beat_002]" in client.prompts[0]
        assert report["applied"] and out.beats[1].narration != RESTATE

    def test_no_client_no_pass(self):
        script = _script(_beat(1, DEFINE), _beat(2, FRESH))
        out, report = _architect(None)._smooth_narration(script, self._config())
        assert out is script and report is None

    def test_segment_prompt_carries_everything_said_so_far(self):
        from src.schemas.source_segment import SourceSegment

        prompt = _architect(None)._build_segment_script_prompt(
            segment=SourceSegment(segment_id="s", order=1, page_start=1, page_end=1, text="word " * 90),
            config=self._config(professor_notes=["Explain what A = U D V^T means."]),
            covered_concepts=[], review_feedback=[], prior_narration=[DEFINE, FRESH],
        )
        assert "WHAT THE LECTURER HAS ALREADY SAID" in prompt
        assert DEFINE in prompt and FRESH in prompt
        assert "they describe the whole video" in prompt and "is met ONCE" in prompt

    def test_said_so_far_keeps_the_latest_lines(self):
        from src.agents.script_architect import ScriptArchitectAgent

        lines = [f"line {i} " + "word " * 300 for i in range(6)]
        block = ScriptArchitectAgent._said_so_far_block(lines)
        assert "(earlier lines omitted)" in block and "line 5" in block and "line 0" not in block


class TestBridgeEcho:
    def test_bridge_that_repeats_the_next_beat_falls_back_to_template(self):
        from src.agents.bridge_planner import BridgePlannerAgent

        beats = [_beat(1, DEFINE, ["svd_definition"]), _beat(2, FRESH)]
        llm = lambda prompt: {"bridges": [{"slot": 0, "text": DEFINE}]}
        texts = BridgePlannerAgent(llm=llm)._write(beats, [(0, "opening")])
        assert texts[0] != DEFINE and "svd definition" in texts[0]

    def test_fresh_bridge_is_kept(self):
        from src.agents.bridge_planner import BridgePlannerAgent

        beats = [_beat(1, DEFINE), _beat(2, FRESH)]
        hook = "Every photo on your phone is a big grid of numbers. How would you squeeze it down?"
        llm = lambda prompt: {"bridges": [{"slot": 0, "text": hook}]}
        assert BridgePlannerAgent(llm=llm)._write(beats, [(0, "opening")])[0] == hook


class TestLengthCuts:
    def test_over_budget_a_beat_may_go_with_its_concept(self):
        script = _script(_beat(1, DEFINE, ["svd"]), _beat(2, FRESH, ["proof_step"]), _beat(3, FRESH + " Yes.", ["low_rank"]))
        data = {"beats": [{"beat_id": "beat_001", "drop": True}, {"beat_id": "beat_002", "drop": True}]}
        out, report = apply_smoothing(script, data)
        assert [b.beat_id for b in out.beats] == ["beat_001", "beat_002", "beat_003"]  # concepts protected
        out, report = apply_smoothing(script, data, allow_cuts=True)
        assert [b.beat_id for b in out.beats] == ["beat_001", "beat_003"]  # never the opening

    def test_cuts_are_capped(self):
        script = _script(*[_beat(i, f"{FRESH} Number {i}.", [f"c{i}"]) for i in range(1, 6)])
        data = {"beats": [{"beat_id": f"beat_{i:03d}", "drop": True} for i in range(1, 6)]}
        out, _ = apply_smoothing(script, data, allow_cuts=True)
        assert len(out.beats) == 5 - int(5 * 0.4)

    def test_prompt_states_the_cut_when_over(self):
        prompt = smoothing_prompt([_beat(1, DEFINE * 5)], 20, "", [])
        assert "must come down to at most 20" in prompt and "binding" in prompt


class TestSmoothingBudget:
    def _config(self, **kw):
        from src.schemas.storyboard import Configuration

        return Configuration(verbosity="medium", depth="intermediate", audience="undergraduate", **kw)

    def test_bridges_reserve_part_of_the_budget(self):
        agent = _architect(None)
        plain = agent._content_word_target(self._config(target_minutes=5))
        bridged = agent._content_word_target(self._config(target_minutes=5, presenter_bridges="propose"))
        budget = round(5 * 60 * ScriptArchitectAgent.SPOKEN_WORDS_PER_SECOND)
        assert plain == budget and bridged == round(budget * 0.85)

    def test_second_pass_only_while_still_over(self):
        long_text = " ".join(f"Sentence number {i} adds a fresh new idea." for i in range(60))
        script = _script(_beat(1, DEFINE, ["svd"]), _beat(2, long_text, ["more"]))
        still_long = {"beats": [{"beat_id": "beat_002", "narration": long_text[: len(long_text) * 2 // 3]}]}
        client = FakeClient(still_long)
        agent = _architect(client)
        out, report = agent._smooth_narration(script, self._config(target_minutes=0.5))
        assert len(client.prompts) == 2 and "must come down to" in client.prompts[0]
        assert report["target_words"] == round(30 * ScriptArchitectAgent.SPOKEN_WORDS_PER_SECOND)
        assert len(report["passes"]) == 2


class TestBriefRehoming:
    def test_content_rule_filed_as_presenter_note_reaches_the_script(self):
        from src.agents.professor_input_interpreter import ProfessorInputInterpreter

        llm = lambda prompt: {"presenter": {"notes": "Skip the proofs and any theorem numbering."}, "notes": []}
        directives = ProfessorInputInterpreter(llm=llm).interpret_brief("Skip the proofs and any theorem numbering.")
        assert directives.presenter.notes == ""
        assert "Skip the proofs and any theorem numbering." in directives.script_instructions()

    def test_real_presenter_note_stays(self):
        from src.agents.professor_input_interpreter import ProfessorInputInterpreter

        llm = lambda prompt: {"presenter": {"notes": "Keep the presenter small in the corner."}}
        directives = ProfessorInputInterpreter(llm=llm).interpret_brief("Keep the presenter small in the corner.")
        assert directives.presenter.notes == "Keep the presenter small in the corner."


class TestSmoothingFloor:
    def test_an_edit_far_under_the_target_is_rejected_and_retried(self):
        from src.schemas.storyboard import Configuration

        text = " ".join(f"Idea {i} is a fresh step in the story." for i in range(9))  # 72 words each
        script = _script(_beat(1, text, ["a"]), _beat(2, text.replace("Idea", "Point"), ["b"]))
        terse = {"beats": [{"beat_id": "beat_001", "narration": "Idea one is a fresh step here, really."},
                           {"beat_id": "beat_002", "narration": "Point one is a fresh step here, really."}]}
        client = FakeClient(terse)
        out, report = _architect(client)._smooth_narration(
            script, Configuration(verbosity="medium", depth="intermediate", audience="undergraduate",
                                  target_minutes=1.0))
        assert out is script and report["applied"] is False
        assert len(client.prompts) == 2 and "far too short" in client.prompts[1]
        assert report["passes"][0]["reason"].startswith("the rewrite cut too much")
