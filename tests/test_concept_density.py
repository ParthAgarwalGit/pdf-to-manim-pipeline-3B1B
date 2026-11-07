"""P4: concept plan (outline step) and density control."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from src.agents.concept_planner import ConceptPlannerAgent, merge_directives, skip_directives
from src.agents.script_architect import ScriptArchitectAgent
from src.agents.script_editor import ScriptEditorAgent
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.schemas.concept_plan import ConceptPlan, ConceptPlanEntry
from src.schemas.directives import ConceptDirective, ProfessorDirectives
from src.schemas.edit_ops import SetDensity
from src.schemas.source_segment import SourceSegment
from src.schemas.storyboard import Configuration
from src.schemas.video_script import Beat, VideoScript


def _segments():
    return [
        SourceSegment(segment_id="seg_001", order=0, page_start=1, page_end=1,
                      heading="2.1 Eigenvectors", text="An eigenvector keeps its direction."),
        SourceSegment(segment_id="seg_002", order=1, page_start=2, page_end=2,
                      heading="2.2 Determinants", text="The determinant scales area."),
    ]


def _config(**kw):
    return Configuration(verbosity=kw.pop("verbosity", "medium"), depth="intermediate", audience="undergraduate", **kw)


class TestConceptPlan:
    def test_entry_instructions(self):
        assert ConceptPlanEntry(concept="eigenvector", depth="deep").instruction() == (
            "eigenvector: go deeper: add a derivation step or a worked example."
        )
        entry = ConceptPlanEntry(concept="phasor", analogy="professor",
                                 professor_analogy_brief="a rotating arm", professor_notes="students mix up i and j")
        assert "use the professor's analogy: a rotating arm" in entry.instruction()
        assert "students mix up i and j" in entry.instruction()
        assert ConceptPlanEntry(concept="x", depth="skip", professor_notes="n").instruction() == "x: do not cover it."

    def test_keyless_planner_uses_headings_and_verbosity_default(self):
        plan = ConceptPlannerAgent(use_default_llm=False).plan(_segments(), _config(verbosity="high"))
        assert [(e.segment_id, e.concept, e.depth) for e in plan.entries] == [
            ("seg_001", "Eigenvectors", "deep"), ("seg_002", "Determinants", "deep"),
        ]

    def test_llm_planner_lists_concepts_per_segment(self):
        llm = lambda prompt: {"segments": [
            {"segment_id": "seg_001", "concepts": ["eigenvector", "eigenvalue"]},
            {"segment_id": "nope", "concepts": ["ignored"]},
        ]}
        plan = ConceptPlannerAgent(llm=llm).plan(_segments(), _config())
        assert [(e.segment_id, e.concept) for e in plan.entries] == [
            ("seg_001", "eigenvector"), ("seg_001", "eigenvalue"),
        ]

    def test_brief_directives_seed_the_plan(self):
        plan = ConceptPlan(entries=[ConceptPlanEntry(concept="eigenvector", segment_id="seg_001")])
        merge_directives(plan, ProfessorDirectives(concepts=[
            ConceptDirective(concept="eigenvector", depth="deep", analogy_brief="a stretched rubber sheet",
                             avoid_analogies=["arrows on a clock"]),
            ConceptDirective(concept="trace", depth="skip"),
        ]))
        first, added = plan.entries
        assert (first.depth, first.analogy, first.professor_analogy_brief) == ("deep", "professor", "a stretched rubber sheet")
        assert "never use the arrows on a clock analogy" in first.professor_notes
        assert (added.concept, added.depth, added.segment_id) == ("trace", "skip", "")
        assert [d.concept for d in skip_directives(plan)] == ["trace"]

    def test_segment_prompt_gets_only_its_plan_entries(self):
        plan = ConceptPlan(entries=[
            ConceptPlanEntry(concept="eigenvector", segment_id="seg_001", depth="deep"),
            ConceptPlanEntry(concept="determinant", segment_id="seg_002", depth="brief"),
            ConceptPlanEntry(concept="notation", depth="standard", professor_notes="bold vectors"),
        ])
        prompts = []

        def create(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            body = json.dumps({"beats": [{"beat_id": "b1", "narration": "Words.", "visual_hint": "x"}]})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])

        agent = ScriptArchitectAgent()
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        agent._generate_segmented_video_script(_segments(), _config(), [], concept_plan=plan)
        assert "CONCEPT PLAN" in prompts[0] and "eigenvector: go deeper" in prompts[0]
        assert "determinant" not in prompts[0] and "determinant: one short beat" in prompts[1]
        assert all("bold vectors" in p for p in prompts)  # unscoped entries reach every segment


class TestConceptPlanInRun:
    def test_skipped_concepts_become_binding_and_logged(self, tmp_path):
        orchestrator = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
        caller = ProfessorDirectives(notes=["be kind"])
        orchestrator._quality_reports = {}
        orchestrator._begin_directives(caller)
        orchestrator._begin_concept_plan(ConceptPlan(entries=[
            ConceptPlanEntry(concept="trace", segment_id="seg_002", depth="skip"),
            ConceptPlanEntry(concept="eigenvector", segment_id="seg_001"),
        ]))
        assert [c.concept for c in orchestrator._directives.concepts] == ["trace"]
        assert caller.concepts == []  # the caller's object is never mutated
        assert orchestrator._quality_reports["professor_omissions"] == [
            {"concept": "trace", "segment_id": "seg_002", "reason": "professor: concept plan"},
        ]
        assert "concept_plan" in orchestrator._quality_reports


def _lesson():
    return VideoScript(beats=[
        Beat(beat_id="beat_001", narration="An eigenvector keeps its direction.", visual_hint="v",
             introduces_concepts=["eigenvector"], source_segment_id="seg_001"),
        Beat(beat_id="beat_002", narration="Like a weathervane in steady wind.", visual_hint="vane",
             role="analogy", analogy_id="analogy_vane", depth=1, source_segment_id="seg_001"),
        Beat(beat_id="beat_003", narration="A full derivation of the characteristic polynomial.", visual_hint="algebra",
             role="derivation", depth=2, source_segment_id="seg_001"),
        Beat(beat_id="beat_004", narration="The determinant scales area.", visual_hint="square",
             introduces_concepts=["determinant"], source_segment_id="seg_002"),
        Beat(beat_id="beat_005", narration="A worked 2x2 determinant example.", visual_hint="grid",
             role="example", depth=1, source_segment_id="seg_002"),
    ])


class TestDensity:
    def test_standard_keeps_depth_one(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(_lesson(), [SetDensity(level="standard")])
        assert [b.beat_id for b in result.script.beats] == ["beat_001", "beat_002", "beat_004", "beat_005"]

    def test_segment_scope_only_touches_that_segment(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(
            _lesson(), [SetDensity(scope="segment", target="seg_002", level="essential")],
        )
        assert [b.beat_id for b in result.script.beats] == ["beat_001", "beat_002", "beat_003", "beat_004"]

    def test_concept_scope(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(
            _lesson(), [SetDensity(scope="concept", target="determinant", level="essential")],
        )
        # Beats about the determinant are in scope: its optional example goes;
        # the eigenvector derivation (depth 2, other concept) stays.
        assert [b.beat_id for b in result.script.beats] == ["beat_001", "beat_002", "beat_003", "beat_004"]

    def test_rich_expands_each_concept_introduction(self):
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return {"beats": [{"narration": f"A deeper example {len(calls)}.", "visual_hint": "x"}]}

        result = ScriptEditorAgent(llm=llm).apply(_lesson(), [SetDensity(level="rich")])
        assert len(calls) == 2  # one per concept introduction (eigenvector, determinant)
        ids = [b.beat_id for b in result.script.beats]
        assert ids.index("beat_006") == ids.index("beat_001") + 1
        assert ids.index("beat_007") == ids.index("beat_004") + 1
        assert result.script.beats[1].depth == 1 and result.script.beats[1].role == "example"

    def test_rich_without_llm_is_rejected_not_degraded(self):
        result = ScriptEditorAgent(use_default_llm=False).apply(_lesson(), [SetDensity(level="rich")])
        assert not result.applied and "needs an LLM" in result.errors[0]


class TestPlanCommand:
    def test_plan_writes_editable_yaml_and_convert_accepts_it(self, tmp_path, monkeypatch):
        import src.cli as cli
        import yaml

        pdf = tmp_path / "notes.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        out = tmp_path / "plan.yaml"
        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        monkeypatch.setattr(
            Orchestrator, "plan_concepts",
            lambda self, path, directives=None, event_source="local": ConceptPlan(entries=[
                ConceptPlanEntry(concept="eigenvector", segment_id="seg_001"),
            ]),
        )
        result = CliRunner().invoke(cli.pdf2manim, ["plan", str(pdf), "-o", str(out)])
        assert result.exit_code == 0, result.output
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert data["entries"][0]["concept"] == "eigenvector"

        data["entries"][0]["depth"] = "skip"
        out.write_text(yaml.safe_dump(data), encoding="utf-8")
        captured = {}

        def fake_run(self, pdf_file_path, event_source="s3", directives=None, concept_plan=None):
            captured["plan"] = concept_plan
            return SimpleNamespace(
                status="SUCCESS", output_directory=str(tmp_path), total_scenes=0, successful_scenes=0,
                delivered_scenes=None, lost_scenes=[], warning_count=0, files=[], final_video_path=None,
            )

        monkeypatch.setattr(Orchestrator, "run_pipeline", fake_run)
        result = CliRunner().invoke(cli.pdf2manim, ["convert", str(pdf), "--plan", str(out), "-o", str(tmp_path)])
        assert captured["plan"].entries[0].depth == "skip", result.output
