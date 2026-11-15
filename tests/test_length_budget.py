"""A requested length (--target-minutes or a brief) budgets the narration.

Before this, the target only capped the number of scenes while the beat
count followed the source's size, so two pages asked for "one minute" came
out at ~5 minutes. Now the script writer gets a per-segment word budget and
an overshooting script is trimmed deterministically. Without a target the
prompts and scripts are exactly as before. Hermetic: no LLM is called.
"""

import json
from types import SimpleNamespace

from src.agents.script_architect import ScriptArchitectAgent, ScriptArchitectInput
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.schemas.directives import ProfessorDirectives
from src.schemas.source_segment import SourceSegment
from src.schemas.storyboard import Configuration
from src.schemas.video_script import Beat, VideoScript


def _config(target=None):
    return Configuration(
        verbosity="medium", depth="intermediate", audience="undergraduate", target_minutes=target,
    )


def _segment(segment_id, words, order=0):
    return SourceSegment(
        segment_id=segment_id, order=order, page_start=1, page_end=1, text=" ".join(["word"] * words),
    )


def _agent():
    agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
    agent.strict = False
    agent.client = None
    return agent


def _beat(n, depth=0, words=40, **extra):
    return Beat(
        beat_id=f"beat_{n:03d}", narration=" ".join([f"w{n}"] * words), visual_hint="h",
        introduces_concepts=[f"concept_{n}"] if depth == 0 else [],
        source_segment_id="seg_001", depth=depth, role="core" if depth == 0 else "example",
        **extra,
    )


def _long_script(**locked):
    """240 words: three core beats and three deeper examples."""
    return VideoScript(beats=[
        _beat(1), _beat(2, depth=1, **locked), _beat(3), _beat(4, depth=1), _beat(5), _beat(6, depth=1),
    ])


class TestBudget:
    def test_budget_is_split_by_source_size(self):
        agent = _agent()
        budgets = agent._segment_word_budgets([_segment("a", 300), _segment("b", 100, 1)], _config(1))
        assert agent._length_budget_words(_config(1)) == 150
        assert budgets == {"a": 112, "b": 38}

    def test_every_segment_keeps_one_short_beat(self):
        segments = [_segment(f"s{i}", 10, i) for i in range(10)]
        budgets = _agent()._segment_word_budgets(segments, _config(0.5))
        assert set(budgets.values()) == {ScriptArchitectAgent.MIN_SEGMENT_WORDS}

    def test_no_target_means_no_budget(self):
        agent = _agent()
        assert agent._length_budget_words(_config()) is None
        assert agent._segment_word_budgets([_segment("a", 300)], _config()) == {}
        assert agent._budgeted_beat_count(" ".join(["w"] * 1000), None) == 22


class TestPrompt:
    def test_budget_caps_beats_and_is_binding(self):
        agent = _agent()
        prompt = agent._build_segment_script_prompt(
            segment=_segment("a", 1000), config=_config(1), covered_concepts=[], review_feedback=[],
            word_budget=112,
        )
        assert "approximately 3 beat(s)" in prompt
        assert "LENGTH BUDGET" in prompt and "about 112 words" in prompt and "about 1 min" in prompt
        assert "within its length budget" in prompt

    def test_prompt_is_unchanged_without_a_budget(self):
        agent = _agent()
        common = dict(segment=_segment("a", 1000), covered_concepts=[], review_feedback=[])
        plain = agent._build_segment_script_prompt(config=_config(), **common)
        assert "approximately 22 beat(s)" in plain and "Explain THIS segment fully" in plain
        assert "LENGTH BUDGET" not in plain
        # A target set elsewhere but no budget for this call: identical text.
        assert agent._build_segment_script_prompt(config=_config(1), **common) == plain

    def test_segmented_generation_sends_each_segment_its_budget(self):
        agent = _agent()
        prompts = []

        def create(model, messages, response_format):
            prompts.append(messages[0]["content"])
            beats = [{"beat_id": "b1", "narration": "A short beat.", "introduces_concepts": [f"c{len(prompts)}"]}]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"beats": beats})))])

        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        agent._generate_segmented_video_script(
            [_segment("a", 300), _segment("b", 100, 1)], _config(1), review_feedback=[],
        )
        assert "about 112 words" in prompts[0] and "about 38 words" in prompts[1]


class TestTrim:
    def test_overshoot_drops_deeper_beats(self):
        script, report = _agent()._enforce_length_budget(_long_script(), _config(1))
        assert [b.beat_id for b in script.beats] == ["beat_001", "beat_003", "beat_005"]
        assert report["removed_beats"] == ["beat_002", "beat_004", "beat_006"]
        assert report["words_before"] == 240 and report["words_after"] == 120
        assert report["budget_words"] == 150 and report["within_budget"] is True

    def test_professor_locked_beats_survive(self):
        script, report = _agent()._enforce_length_budget(
            _long_script(locked=True, author="professor"), _config(1),
        )
        assert "beat_002" in {b.beat_id for b in script.beats}
        assert report["removed_beats"] == ["beat_004", "beat_006"]
        assert any("beat_002" in w for w in report["warnings"])

    def test_no_trim_without_target_or_within_tolerance(self):
        agent = _agent()
        script = _long_script()
        assert agent._enforce_length_budget(script, _config()) == (script, None)
        # 240 words against a 2-minute (300-word) budget: within tolerance.
        assert agent._enforce_length_budget(script, _config(2)) == (script, None)

    def test_execute_trims_and_records_the_repair(self, monkeypatch):
        from src.schemas.ocr_output import BoundingBox, OCROutput, Page, PDFMetadata, TextBlock

        agent = _agent()
        monkeypatch.setattr(agent, "_generate_video_script", lambda *a, **k: _long_script())
        ocr = OCROutput(
            pdf_metadata=PDFMetadata(filename="t.pdf", page_count=1, file_size_bytes=10),
            pages=[Page(page_number=1, text_blocks=[TextBlock(
                text="hello world", reading_order=0, bounding_box=BoundingBox(x=0, y=0, width=1, height=1),
            )])],
        )
        storyboard = agent.execute(ScriptArchitectInput(
            ocr_output=ocr, configuration=_config(1), segments=[_segment("seg_001", 50)],
            plan_visuals=False,
        ))
        beats = [b.beat_id for s in storyboard.scenes for b in s.source_beats]
        assert beats == ["beat_001", "beat_003", "beat_005"]
        assert agent.last_length_repair["removed_beats"] == ["beat_002", "beat_004", "beat_006"]


class TestPipelineRecord:
    def test_brief_length_check_reports_the_trim(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
        orch._begin_directives(ProfessorDirectives.model_validate({"global": {"target_minutes": 1}}))
        orch._length_repair = {"removed_beats": ["beat_004"], "budget_words": 150}
        orch._check_length_directive(SimpleNamespace(scenes=[
            SimpleNamespace(audio_duration_seconds=None, duration_estimate=120.0),
        ]))
        check = orch._directive_checks[0]
        assert check.passed is False and check.repair == "trim"
        assert "1 deeper beat(s) were already trimmed" in check.message
        assert check.measured["trim"]["removed_beats"] == ["beat_004"]

    def test_trim_is_saved_as_a_report(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(
            base_output_dir=str(tmp_path), target_minutes=1, enable_quality_loops=False,
        ))
        trim = {"removed_beats": ["beat_002"], "budget_words": 150}

        def architect(input_data):
            orch.script_architect_agent.last_length_repair = trim
            return SimpleNamespace()

        orch.script_architect_agent.execute = architect
        silent = SimpleNamespace(log_agent_start=lambda *a, **k: None, log_agent_complete=lambda *a, **k: None)
        orch._run_script_architect(None, silent, source_segments=[object()])
        assert orch._quality_reports["length_budget"] == trim
