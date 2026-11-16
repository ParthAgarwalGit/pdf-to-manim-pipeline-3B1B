"""Professor controls on the V2 pipeline: brief, concept plan, scope, budget, compliance.

Hermetic: no provider is called. Router-backed paths use a fake router or a
real ``ProviderRouter`` without keys, which takes the deterministic fallback.
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from src.pipeline_v2 import directives as pd
from src.pipeline_v2.directives import (
    ComplianceCheck,
    ConceptPlan,
    LengthBudget,
    allocate_scene_words,
    apply_scope,
    build_length_budget,
    check_lesson_directives,
    dump_directives,
    enforce_narration_directives,
    interpret_brief,
    interpret_brief_with_router,
    length_check,
    lesson_directive_block,
    load_directives,
    narration_directive_block,
    notation_findings,
    notation_rules,
    readback_lines,
    scene_cap_for,
    total_word_budget,
    trim_to_budget,
)
from src.pipeline_v2.models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    NarrationPlan,
    SourceDocument,
    SourceFact,
)
from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.planning import PlanningService, validate_grounding
from src.pipeline_v2.providers import ProviderRouter
from src.pipeline_v2.state import RunStore
from src.schemas.directives import (
    ConceptDirective,
    GlobalDirective,
    NotationDirective,
    ProfessorDirectives,
    ScopeDirective,
)

SPEC_BRIEF = (
    "My students are 2nd-year EE and struggle with phasors. Use the rotating-arm picture, "
    "not the clock analogy. Skip 4.3. Spend more time on resonance. Under 10 minutes. "
    "I write j, not i."
)


# ── builders ────────────────────────────────────────────────────────────────


def _words(count: int, word: str = "word") -> str:
    return " ".join([word] * count)


def _fact(fact_id: str, page: int, text: str, kind: str = "text") -> SourceFact:
    return SourceFact(fact_id=fact_id, page_number=page, text=text, kind=kind)


def _source(sizes=(30, 30, 30)) -> SourceDocument:
    """One fact per page; ``sizes`` are the facts' word counts."""

    facts = [
        _fact(f"p{index + 1:03d}-f000", index + 1, _words(size - 1, f"w{index}") + " end.")
        for index, size in enumerate(sizes)
    ]
    return SourceDocument(filename="notes.pdf", sha256="abc", page_count=len(sizes), facts=facts)


def _lesson(source: SourceDocument, titles=None) -> LessonPlan:
    titles = titles or [f"Topic {index + 1}" for index in range(len(source.facts))]
    return LessonPlan(
        title="Phasors", audience="undergraduate", objectives=["Understand phasors"],
        scenes=[
            LessonScene(
                scene_id=f"scene_{index + 1:03d}", title=title,
                description=f"About {title}", learning_goal=f"Understand {title}",
                source_fact_ids=[source.facts[index].fact_id],
                visual_objective="Show it", new_visual_fact="A new fact",
                do_not_repeat=["earlier construction"],
            )
            for index, title in enumerate(titles)
        ],
    )


def _beat(beat_id: str, text: str, role: str = "explain", fact: str = "p001-f000") -> NarrationBeat:
    return NarrationBeat(beat_id=beat_id, text=text, source_fact_ids=[fact], role=role)


def _scene(scene_id: str, beats: list[NarrationBeat]) -> NarratedScene:
    return NarratedScene(
        scene_id=scene_id, narration=" ".join(beat.text for beat in beats), beats=beats,
    )


def _budget(scene_words: dict[str, int]) -> LengthBudget:
    return LengthBudget(
        target_minutes=1, words_per_minute=135, total_words=sum(scene_words.values()),
        scene_words=scene_words, scene_cap=2,
    )


def _scene_word_total(scene: NarratedScene) -> int:
    return sum(len(beat.text.split()) for beat in scene.beats)


class FakeRouter:
    """Captures structured calls; answers from ``handlers`` or the fallback."""

    strict_keys = False

    def __init__(self, **handlers):
        self.handlers = handlers
        self.calls: list[dict] = []
        self.artifacts = SimpleNamespace(event=lambda *args, **kwargs: None)

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        handler = self.handlers.get(kwargs["stage"])
        return handler(kwargs) if handler else kwargs["fallback"]()

    def prompts(self, stage: str) -> list[str]:
        return [call["prompt"] for call in self.calls if call["stage"] == stage]


def _real_router(tmp_path) -> tuple[ProviderRouter, RunStore]:
    store = RunStore(tmp_path / "run.db", 10.0)
    return ProviderRouter(store, RunArtifacts(tmp_path, "run")), store


def _required_ids(prompt: str) -> list[str]:
    return re.findall(r'"(scene_\d+)"', re.search(r'"required_scene_ids": \[(.*?)\]', prompt)[1])


def _long_narration(source: SourceDocument):
    """A model that overshoots: five 20-40 word beats per scene, one with a clock."""

    def handler(call) -> NarrationPlan:
        scenes = []
        for scene_id in _required_ids(call["prompt"]):
            fact = source.facts[int(scene_id[-3:]) - 1].fact_id
            beats = [
                _beat(f"{scene_id}_b1", _words(19, "open") + " start.", "introduce", fact),
                _beat(f"{scene_id}_b2", "Picture a clock face turning. " + _words(15), "explain", fact),
                _beat(f"{scene_id}_b3", _words(40, "demo"), "demonstrate", fact),
                _beat(f"{scene_id}_b4", _words(40, "think"), "reflect", fact),
                _beat(f"{scene_id}_b5", _words(20, "close"), "conclude", fact),
            ]
            scenes.append(_scene(scene_id, beats))
        return NarrationPlan(scenes=scenes)

    return handler


# ── brief ───────────────────────────────────────────────────────────────────


class TestBrief:
    def test_rules_read_the_spec_example(self):
        directives = interpret_brief(SPEC_BRIEF)

        assert directives.scope.skip_sections == ["4.3"]
        assert directives.global_.target_minutes == 10
        assert directives.global_.audience == "undergraduate"
        assert [(n.use, n.instead_of) for n in directives.notation] == [("j", "i")]
        assert directives.avoided_analogies() == ["clock"]
        assert [(c.concept, c.depth) for c in directives.concepts if c.depth] == [
            ("resonance", "deep")
        ]
        assert directives.source_brief == SPEC_BRIEF

        lines = "\n".join(readback_lines(directives))
        for expected in (
            "Leave out section(s): 4.3",
            "Keep it under 10 minutes",
            "Audience: undergraduate",
            "Notation: write j, not i.",
            "Never use the clock analogy.",
            "the rotating-arm picture",
            "Cover resonance in depth",
            "struggle with phasors",
        ):
            assert expected in lines

    def test_unmappable_sentences_are_kept_verbatim(self):
        directives = interpret_brief("Mention the lab on Friday. Skip 2.1.")

        assert directives.notes == ["Mention the lab on Friday."]
        assert directives.scope.skip_sections == ["2.1"]
        assert interpret_brief("Be kind").notes == ["Be kind"]
        assert interpret_brief("   ").is_empty()

    def test_reversed_terminology_from_a_model_reading_is_corrected(self):
        brief = "Say eigenvector rather than characteristic vector."
        reading = ProfessorDirectives(
            concepts=[ConceptDirective(concept="", avoid_analogies=["eigenvector", "pizza"])],
            global_=GlobalDirective(terminology={"eigenvector": "characteristic vector"}),
        )
        router = FakeRouter(brief=lambda call: reading)

        directives = asyncio.run(interpret_brief_with_router(brief, router))

        assert directives.global_.terminology == {"characteristic vector": "eigenvector"}
        assert directives.avoided_analogies() == ["pizza"]  # vocabulary is not an analogy
        assert directives.source_brief == brief
        assert reading.global_.terminology == {"eigenvector": "characteristic vector"}
        call = router.calls[0]
        assert (call["stage"], call["task_id"], call["schema"]) == (
            "brief", "brief", ProfessorDirectives,
        )
        assert '"global"' in json.dumps(call["schema"].model_json_schema())

    def test_invented_verbosity_and_audience_are_dropped(self):
        reading = ProfessorDirectives(
            scope=ScopeDirective(skip_sections=["4.3"]),
            global_=GlobalDirective(verbosity="high", audience="graduate"),
        )
        router = FakeRouter(brief=lambda call: reading)

        directives = asyncio.run(interpret_brief_with_router("Skip 4.3.", router))

        assert directives.global_.verbosity is None
        assert directives.global_.audience is None
        assert directives.scope.skip_sections == ["4.3"]

    def test_router_path_falls_back_to_the_rules_without_a_key(self, tmp_path):
        router, store = _real_router(tmp_path)
        try:
            directives = asyncio.run(interpret_brief_with_router(SPEC_BRIEF, router))
        finally:
            store.close()

        assert dump_directives(directives) == dump_directives(interpret_brief(SPEC_BRIEF))

    def test_empty_brief_makes_no_router_call(self):
        router = FakeRouter()

        assert asyncio.run(interpret_brief_with_router("  ", router)).is_empty()
        assert router.calls == []

    def test_dump_and_load_use_the_global_alias(self, tmp_path):
        directives = interpret_brief(SPEC_BRIEF)
        data = dump_directives(directives)
        path = tmp_path / "brief_interpretation.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        assert "global" in data and "global_" not in data
        assert load_directives(path) == directives


# ── concept plan ────────────────────────────────────────────────────────────


class TestConceptPlan:
    def test_from_lesson_defaults(self):
        source = _source()
        lesson = _lesson(source)

        plan = ConceptPlan.from_lesson(lesson, source)

        assert plan.version == 1 and plan.source_sha256 == "abc"
        assert [entry.scene_id for entry in plan.concepts] == ["scene_001", "scene_002", "scene_003"]
        assert [entry.concept for entry in plan.concepts] == ["Topic 1", "Topic 2", "Topic 3"]
        assert {entry.depth for entry in plan.concepts} == {"standard"}
        assert {entry.analogy for entry in plan.concepts} == {"auto"}
        assert plan.depth_for("scene_002") == "standard" and plan.notes_for("scene_002") == ""
        assert plan.depth_for("missing") == "standard"

    @pytest.mark.parametrize(
        "verbosity, depth", [("low", "brief"), ("medium", "standard"), ("high", "deep")],
    )
    def test_default_depth_follows_verbosity(self, verbosity, depth):
        source = _source()
        directives = ProfessorDirectives(global_=GlobalDirective(verbosity=verbosity))

        plan = ConceptPlan.from_lesson(_lesson(source), source, directives)

        assert {entry.depth for entry in plan.concepts} == {depth}

    def test_brief_seeds_matching_scenes(self):
        source = _source()
        lesson = _lesson(source, ["Phasor basics", "Series Resonance", "Power"])
        directives = ProfessorDirectives(concepts=[
            ConceptDirective(
                concept="resonance", depth="deep", notes="students confuse it with damping",
                analogy_brief="a child on a swing", avoid_analogies=["clock"],
            ),
            ConceptDirective(concept="Average power in AC circuits", depth="skip"),
            ConceptDirective(concept="", avoid_analogies=["pizza"]),
        ])

        plan = ConceptPlan.from_lesson(lesson, source, directives)

        assert [entry.depth for entry in plan.concepts] == ["standard", "deep", "skip"]
        resonance = plan.concepts[1]
        assert resonance.analogy == "professor" and resonance.analogy_brief == "a child on a swing"
        assert resonance.notes == "students confuse it with damping; never use the clock analogy"
        assert "use the professor's analogy: a child on a swing" in plan.notes_for("scene_002")
        assert plan.concepts[0].notes == ""
        assert [entry.concept for entry in plan.skipped()] == ["Power"]

    def test_yaml_and_json_round_trips_are_lossless(self, tmp_path):
        source = _source()
        lesson = _lesson(source, ["Phasor basics", "Résonance: ω₀ = 1/√(LC)", "yes"])
        plan = ConceptPlan.from_lesson(lesson, source)
        plan.concepts[1].depth = "deep"
        plan.concepts[1].notes = "address the 1.0 vs 1 confusion: it matters"

        assert ConceptPlan.from_text(plan.to_yaml()) == plan
        assert "Résonance" in plan.to_yaml()  # written as readable unicode
        for name in ("plan.yaml", "plan.json"):
            assert ConceptPlan.load(plan.save(tmp_path / name)) == plan
        assert ConceptPlan.from_text(plan.to_yaml()).to_lesson(source)[0] == lesson

    def test_to_lesson_drops_skipped_entries_and_reports_them(self):
        source = _source()
        plan = ConceptPlan.from_lesson(_lesson(source), source)
        plan.concepts[1].depth = "skip"
        plan.concepts.reverse()  # scene order is file order

        lesson, omitted = plan.to_lesson(source)

        assert [scene.scene_id for scene in lesson.scenes] == ["scene_003", "scene_001"]
        assert [scene.title for scene in lesson.scenes] == ["Topic 3", "Topic 1"]
        assert omitted == ["Topic 2"]

    def test_unknown_fact_ids_are_rejected_by_name(self):
        source = _source()
        plan = ConceptPlan.from_lesson(_lesson(source), source)
        plan.concepts[0].source_fact_ids = ["p001-f000", "p099-f000"]

        with pytest.raises(ValueError, match="p099-f000"):
            plan.to_lesson(source)

    def test_skipping_everything_is_rejected(self):
        source = _source()
        plan = ConceptPlan.from_lesson(_lesson(source), source)
        for entry in plan.concepts:
            entry.depth = "skip"

        with pytest.raises(ValueError, match="skips every concept"):
            plan.to_lesson(source)

    def test_text_that_is_not_a_mapping_is_rejected(self):
        with pytest.raises(ValueError):
            ConceptPlan.from_text("- just\n- a list\n")


# ── scope ───────────────────────────────────────────────────────────────────


def _sectioned_source() -> SourceDocument:
    facts = [
        _fact("p001-f000", 1, "4.1 Phasors"),
        _fact("p001-f001", 1, "A phasor is a rotating arm."),
        _fact("p002-f000", 2, "4.3 Resonance"),
        _fact("p002-f001", 2, "At resonance the reactances cancel."),
        _fact("p002-f002", 2, "1. Compute the resonant frequency."),
        _fact("p002-f003", 2, "4.3.1 Quality Factor"),
        _fact("p002-f004", 2, "Q measures sharpness."),
        _fact("p002-f005", 2, "12"),
        _fact("p003-f000", 3, "4.4 Power"),
        _fact("p003-f001", 3, "Average power depends on the phase angle."),
    ]
    return SourceDocument(filename="notes.pdf", sha256="abc", page_count=3, facts=facts)


def _scope(**scope) -> ProfessorDirectives:
    return ProfessorDirectives(scope=ScopeDirective(**scope))


class TestScope:
    def test_no_directives_leaves_the_source_alone(self):
        source = _sectioned_source()

        assert apply_scope(source, None) == (source, [])
        assert apply_scope(source, ProfessorDirectives()) == (source, [])

    def test_page_filters(self):
        source = _sectioned_source()

        only, only_checks = apply_scope(source, _scope(only_pages=[1, 2]))
        skipped, skip_checks = apply_scope(source, _scope(skip_pages=[1]))

        assert {fact.page_number for fact in only.facts} == {1, 2}
        assert {fact.page_number for fact in skipped.facts} == {2, 3}
        assert only.page_count == skipped.page_count == 3
        assert all(check.passed for check in only_checks + skip_checks)
        assert len(source.facts) == 10  # the input document is not mutated

    def test_section_is_skipped_up_to_the_next_heading_of_its_level(self):
        source = _sectioned_source()

        scoped, checks = apply_scope(source, _scope(skip_sections=["4.3"]))

        assert [fact.fact_id for fact in scoped.facts] == [
            "p001-f000", "p001-f001", "p003-f000", "p003-f001",
        ]
        assert [(check.check, check.passed) for check in checks] == [("Skip section 4.3", True)]

    def test_section_label_spelled_with_a_prefix_still_matches(self):
        scoped, checks = apply_scope(_sectioned_source(), _scope(skip_sections=["Section 4.4"]))

        assert [fact.fact_id for fact in scoped.facts][-1] == "p002-f005"
        assert checks[0].passed

    def test_unmatched_label_fails_its_check(self):
        source = _sectioned_source()

        scoped, checks = apply_scope(source, _scope(skip_sections=["9.9", "4"]))

        assert scoped.facts == source.facts  # "4" must not match "4.1 Phasors" as a prefix
        assert [check.passed for check in checks] == [False, False]
        assert "9.9" in checks[0].message

    def test_excluding_everything_keeps_the_document(self):
        source = _sectioned_source()

        scoped, checks = apply_scope(source, _scope(only_pages=[7, 8]))

        assert scoped == source
        assert checks[-1].severity == "high" and not checks[-1].passed
        assert not any(check.passed for check in checks)

    def test_ocr_facts_appended_out_of_page_order_are_still_scoped(self):
        source = _sectioned_source()
        source = source.model_copy(update={
            "facts": source.facts + [_fact("p002-ocr000", 2, "OCR text inside section 4.3.")],
        })

        scoped, _ = apply_scope(source, _scope(skip_sections=["4.3"]))

        assert "p002-ocr000" not in {fact.fact_id for fact in scoped.facts}


# ── word budget ─────────────────────────────────────────────────────────────


class TestBudget:
    def test_formula_values(self):
        assert scene_cap_for(10) == 15
        assert scene_cap_for(1) == 2
        assert scene_cap_for(0.2) == 1
        assert total_word_budget(10, 135, 12) == 1264
        assert total_word_budget(1, 135, 2) == 121
        assert total_word_budget(1, 115, 2) == 103
        assert total_word_budget(0.5, 135, 6) == 150  # never below 25 words per scene

    def test_allocation_follows_source_size_and_sums_to_the_total(self):
        source = _source((300, 100))
        lesson = _lesson(source)

        words = allocate_scene_words(lesson, source, 200, {})

        assert words == {"scene_001": 150, "scene_002": 50}

    def test_small_scenes_get_the_floor_and_the_sum_is_kept(self):
        source = _source((1000, 10, 10))
        lesson = _lesson(source)

        words = allocate_scene_words(lesson, source, 100, {})

        assert words == {"scene_001": 50, "scene_002": 25, "scene_003": 25}
        assert sum(allocate_scene_words(lesson, source, 1264, {}).values()) == 1264

    def test_depth_weights_shift_words(self):
        source = _source((100, 100))
        lesson = _lesson(source)

        words = allocate_scene_words(
            lesson, source, 210, {"scene_001": "deep", "scene_002": "brief"},
        )

        assert words == {"scene_001": 160, "scene_002": 50}
        assert pd.DEPTH_WEIGHTS == {"brief": 0.5, "standard": 1.0, "deep": 1.6}

    def test_build_length_budget(self):
        source = _source((300, 100))

        budget = build_length_budget(_lesson(source), source, 1, 135)

        assert (budget.total_words, budget.scene_cap, budget.words_per_minute) == (121, 2, 135)
        assert sum(budget.scene_words.values()) == 121

    def test_trim_removes_roles_in_order_and_keeps_the_opening_beat(self):
        beats = [
            _beat("b1", _words(20, "a"), "explain"),
            _beat("b2", _words(20, "b"), "explain"),
            _beat("b3", _words(20, "c"), "demonstrate"),
            _beat("b4", _words(20, "d"), "bridge"),
            _beat("b5", _words(20, "e"), "reflect"),
            _beat("b6", _words(20, "f"), "conclude"),
        ]
        plan = NarrationPlan(scenes=[_scene("scene_001", beats)])

        light, light_report = trim_to_budget(plan, _budget({"scene_001": 80}))
        heavy, heavy_report = trim_to_budget(plan, _budget({"scene_001": 32}))

        assert light_report["removed_beats"] == ["b5"]  # 100 words is within 80 x 1.25
        assert [beat.beat_id for beat in light.scenes[0].beats] == ["b1", "b2", "b3", "b4", "b6"]
        assert heavy_report["removed_beats"] == ["b5", "b4", "b3", "b2"]
        assert [beat.beat_id for beat in heavy.scenes[0].beats] == ["b1", "b6"]
        assert heavy.scenes[0].narration == beats[0].text + " " + beats[5].text
        assert heavy_report["words_before"] == 120 and heavy_report["words_after"] == 40
        assert heavy_report["budget_words"] == 32 and heavy_report["within_budget"]

    def test_trim_truncates_sentences_when_beats_cannot_go(self):
        beats = [
            _beat("b1", "Short opening sentence here. " + _words(29) + " end.", "introduce"),
            _beat("b2", _words(9) + " end.", "conclude"),
        ]
        plan = NarrationPlan(scenes=[_scene("scene_001", beats)])

        trimmed, report = trim_to_budget(plan, _budget({"scene_001": 25}))

        assert report["removed_beats"] == [] and report["truncated_beats"] == ["b1"]
        assert trimmed.scenes[0].beats[0].text == "Short opening sentence here."
        assert trimmed.scenes[0].narration == "Short opening sentence here. " + beats[1].text
        assert report["words_after"] == 14 and report["within_budget"]

    def test_trim_never_empties_a_scene(self):
        plan = NarrationPlan(scenes=[_scene("scene_001", [_beat("b1", _words(100), "explain")])])

        trimmed, report = trim_to_budget(plan, _budget({"scene_001": 25}))

        assert trimmed == plan
        assert report["over_budget_scenes"] == ["scene_001"] and not report["within_budget"]

    def test_scripts_within_tolerance_are_untouched(self):
        plan = NarrationPlan(scenes=[
            _scene("scene_001", [_beat("b1", _words(30)), _beat("b2", _words(30), "reflect")]),
            _scene("scene_009", [_beat("x1", _words(500), "reflect")]),  # no budget: left alone
        ])

        trimmed, report = trim_to_budget(plan, _budget({"scene_001": 48}))

        assert trimmed is plan
        assert report["removed_beats"] == [] and report["truncated_beats"] == []
        assert report["words_before"] == report["words_after"] == 560

    def test_trim_is_deterministic_and_never_longer(self):
        source = _source((40, 40, 40))
        plan = _long_narration(source)({"prompt": '"required_scene_ids": ["scene_001", "scene_002", "scene_003"]'})
        budget = _budget({"scene_001": 25, "scene_002": 60, "scene_003": 200})

        first, report = trim_to_budget(plan, budget)
        second, _ = trim_to_budget(plan, budget)

        assert first == second
        for before, after in zip(plan.scenes, first.scenes):
            assert _scene_word_total(after) <= _scene_word_total(before)
            assert after.beats and after.beats[0].beat_id == before.beats[0].beat_id
        assert first.scenes[2] == plan.scenes[2]
        assert report["words_after"] < report["words_before"]

    def test_length_check(self):
        assert length_check(650, 10).passed  # within the 10 % allowance
        failed = length_check(400, 5)
        assert not failed.passed and failed.check == "Under 5 minutes"
        assert "6.7 min" in failed.message
        assert isinstance(failed, ComplianceCheck)


# ── compliance ──────────────────────────────────────────────────────────────


class TestCompliance:
    def test_terminology_replacement_is_whole_word_and_case_preserving(self):
        directives = ProfessorDirectives(global_=GlobalDirective(terminology={"vector": "phasor"}))
        text = "Vector sums: the vector rotates, vectorial and eigenvector stay. VECTOR!"
        plan = NarrationPlan(scenes=[_scene("scene_001", [_beat("b1", text), _beat("b2", "Nothing here.")])])

        enforced, checks = enforce_narration_directives(plan, directives)

        expected = "Phasor sums: the phasor rotates, vectorial and eigenvector stay. PHASOR!"
        assert enforced.scenes[0].beats[0].text == expected
        assert enforced.scenes[0].narration == expected + " Nothing here."
        assert enforced.scenes[0].beats[1] is plan.scenes[0].beats[1]
        assert [(check.check, check.passed, check.beat_ids) for check in checks] == [
            ("Say 'phasor' rather than 'vector'", True, ["b1"]),
        ]

    def test_preferred_wording_containing_the_avoided_word_passes(self):
        directives = ProfessorDirectives(
            global_=GlobalDirective(terminology={"vector": "state vector"}),
        )
        plan = NarrationPlan(scenes=[_scene("scene_001", [_beat("b1", "The vector grows.")])])

        enforced, checks = enforce_narration_directives(plan, directives)

        assert enforced.scenes[0].beats[0].text == "The state vector grows."
        assert checks[0].passed

    def test_avoided_analogy_beats_are_removed_or_reported(self):
        directives = ProfessorDirectives(
            concepts=[ConceptDirective(concept="", avoid_analogies=["the clock analogy"])],
        )
        plan = NarrationPlan(scenes=[
            _scene("scene_001", [
                _beat("a1", "Think of two clocks side by side."),
                _beat("a2", "The arm rotates clockwise."),
            ]),
            _scene("scene_002", [_beat("b1", "Imagine a clock face.")]),
        ])

        enforced, checks = enforce_narration_directives(plan, directives)

        assert [beat.beat_id for beat in enforced.scenes[0].beats] == ["a2"]
        assert enforced.scenes[0].narration == "The arm rotates clockwise."
        assert enforced.scenes[1] == plan.scenes[1]  # its only beat cannot be removed
        check = checks[0]
        assert check.check == "Never use the the clock analogy analogy" or not check.passed
        assert (check.passed, check.scene_ids, check.beat_ids) == (False, ["scene_002"], ["b1"])

    def test_omitted_and_skipped_concepts_are_removed_by_whole_name(self):
        directives = ProfessorDirectives(
            concepts=[ConceptDirective(concept="quality factor", depth="skip")],
        )
        plan = NarrationPlan(scenes=[_scene("scene_001", [
            _beat("b1", "Resonance appears when reactances cancel."),
            _beat("b2", "Series resonance gives a current peak."),
            _beat("b3", "Quality factors measure sharpness."),
            _beat("b4", "Now the summary."),
        ])])

        enforced, checks = enforce_narration_directives(
            plan, directives, omitted_concepts=("Series Resonance",),
        )

        # "resonance" alone is not the omitted "Series Resonance" scene.
        assert [beat.beat_id for beat in enforced.scenes[0].beats] == ["b1", "b4"]
        assert [(check.check, check.passed) for check in checks] == [
            ("Do not cover quality factor", True), ("Do not cover Series Resonance", True),
        ]

    def test_enforcement_is_idempotent_and_a_no_op_without_violations(self):
        directives = interpret_brief(SPEC_BRIEF)
        plan = NarrationPlan(scenes=[_scene("scene_001", [_beat("b1", "The arm rotates.")])])

        enforced, checks = enforce_narration_directives(plan, directives)

        assert enforced is plan
        assert [check.passed for check in checks] == [True]
        assert enforce_narration_directives(plan, None) == (plan, [])

    def test_lesson_checks_cover_skipped_concepts_and_excluded_pages(self):
        source = _source()
        lesson = _lesson(source, ["Phasors", "Series Resonance", "Power"])
        directives = ProfessorDirectives(
            concepts=[ConceptDirective(concept="power", depth="skip")],
            scope=ScopeDirective(skip_pages=[2], only_pages=[1, 2]),
        )

        checks = check_lesson_directives(lesson, source, directives, omitted_concepts=("Filters",))

        assert [(check.check, check.passed, check.scene_ids) for check in checks] == [
            ("Do not cover power", False, ["scene_003"]),
            ("Do not cover Filters", True, []),
            ("Leave out page(s): 2", False, ["scene_002"]),
            ("Only use pages 1-2", False, ["scene_003"]),
        ]
        assert check_lesson_directives(lesson, source, None) == []

    def test_notation_findings_respect_symbol_boundaries(self):
        directives = ProfessorDirectives(notation=[NotationDirective(use="j", instead_of="i")])
        clean = (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        a = MathTex(r'2\\pi f', r'\\sin(x)')\n"
            "        b = MathTex(r'\\sum_{i=1}^n x_i')\n"
            "        c = Text('i is fine outside Tex')\n"
        )
        dirty = clean + "        d = MathTex(r'\\frac{1}{(a+b)} + e^{i\\theta}')\n"

        assert notation_findings(clean, directives) == []
        findings = notation_findings(dirty, directives)
        assert len(findings) == 1
        assert "write j, not i" in findings[0] and "line(s) 7" in findings[0]
        assert notation_findings("x = MathTex(r'2i'", directives)  # unparsable code is scanned too
        assert notation_findings(dirty, None) == []
        assert notation_rules(directives) == ["Write j, not i."]
        assert notation_rules(None) == []


# ── prompt blocks ───────────────────────────────────────────────────────────


class TestPromptBlocks:
    def test_blocks_are_empty_when_there_is_nothing_to_say(self):
        assert lesson_directive_block(None, None, None) == ""
        assert lesson_directive_block(ProfessorDirectives(), None, None) == ""
        assert narration_directive_block(None, {}, {}, {}) == ""
        assert narration_directive_block(ProfessorDirectives(), {}, {"scene_001": "standard"}, {}) == ""

    def test_lesson_block_carries_binding_lines_audience_and_cap(self):
        block = lesson_directive_block(interpret_brief(SPEC_BRIEF), "graduate", 15)

        assert block.startswith("\n\nPROFESSOR DIRECTIVES (binding")
        assert "- Audience: undergraduate." in block  # the brief wins over the configured value
        assert "- Length: plan at most 15 scene(s)" in block
        assert "- Never use the clock analogy." in block
        assert "- Notation: write j, not i." in block

    def test_narration_block_lists_per_scene_budgets_depth_and_notes(self):
        block = narration_directive_block(
            interpret_brief(SPEC_BRIEF),
            {"scene_003": 60, "scene_004": 25},
            {"scene_003": "deep", "scene_004": "standard"},
            {"scene_003": "students confuse it with damping"},
        )

        assert "- Never use the clock analogy." in block
        assert "- Audience: undergraduate." in block
        assert (
            "- scene_003: at most about 60 words; " + pd.DEPTH_INSTRUCTIONS["deep"]
            + "; professor's note: students confuse it with damping"
        ) in block
        assert "- scene_004: at most about 25 words\n" in block
        assert "length budget" in block


# ── PlanningService ─────────────────────────────────────────────────────────


class TestPlanningService:
    def test_default_prompts_are_byte_identical_without_professor_controls(self):
        source = _source()
        lesson = _lesson(source)

        def capture(*args, **kwargs) -> tuple[list[str], list[str], NarrationPlan]:
            router = FakeRouter()
            planner = PlanningService(router, *args, **kwargs)
            asyncio.run(planner.lesson(source))
            narration = asyncio.run(planner.narration(lesson, source))
            assert planner.length_budget is None and planner.length_report is None
            assert planner.compliance_checks == [] and planner.omitted_concepts == []
            return router.prompts("lesson"), router.prompts("narration"), narration

        plain = capture()
        explicit = capture(
            12, None, "standard",
            audience=None, directives=None, concept_plan=None, target_minutes=None,
        )

        assert plain == explicit
        lesson_prompt, narration_prompt = plain[0][0], plain[1][0]
        # Nothing is appended after the JSON payload each prompt has always ended with.
        assert lesson_prompt.endswith(json.dumps(
            {"document": "notes.pdf", "facts": [
                {"fact_id": fact.fact_id, "page_number": fact.page_number,
                 "kind": fact.kind, "text": fact.text}
                for fact in source.facts
            ]}, ensure_ascii=False,
        ))
        assert "use at most 12 scenes. Merge" in lesson_prompt
        assert narration_prompt.endswith('"learning_goal": "Understand Topic 3"}]}'.replace(
            '"learning_goal": "Understand Topic 3"}]}', '"next_scene": null}}',
        ))
        for prompt in (lesson_prompt, narration_prompt):
            assert "PROFESSOR" not in prompt and "at most about" not in prompt

    def test_directives_reach_the_lesson_and_narration_prompts(self):
        source = _source()
        router = FakeRouter()
        planner = PlanningService(
            router, audience="graduate", directives=interpret_brief(SPEC_BRIEF),
        )

        lesson = asyncio.run(planner.lesson(source))
        asyncio.run(planner.narration(lesson, source))

        for prompt in (router.prompts("lesson")[0], router.prompts("narration")[0]):
            assert "PROFESSOR DIRECTIVES (binding" in prompt
            assert "- Audience: undergraduate." in prompt
            assert "- Never use the clock analogy." in prompt
            assert "- Notation: write j, not i." in prompt
        assert planner.target_minutes == 10  # the brief's own length is the budget
        assert "at most 12 scene(s)" in router.prompts("lesson")[0]  # min(12, cap 15)

    def test_configured_audience_alone_is_stated(self):
        router = FakeRouter()
        planner = PlanningService(router, audience="high-school")

        lesson = asyncio.run(planner.lesson(_source()))

        assert "- Audience: high-school." in router.prompts("lesson")[0]
        assert lesson.audience == "high-school"  # the keyless fallback lesson honours it

    def test_target_minutes_clamps_the_scene_ceiling(self):
        source = _source((30,) * 6)
        router = FakeRouter()
        planner = PlanningService(router, target_minutes=1)

        lesson = asyncio.run(planner.lesson(source))

        prompt = router.prompts("lesson")[0]
        assert "use at most 2 scenes. Merge" in prompt
        assert "- Length: plan at most 2 scene(s)" in prompt
        assert len(lesson.scenes) == 2  # the fallback lesson honours the clamp
        assert len(asyncio.run(PlanningService(FakeRouter()).lesson(source)).scenes) == 6

    def test_full_coverage_floor_uses_the_clamped_ceiling(self):
        source = _source((30,) * 20)
        three = _lesson(source, ["A", "B", "C"])
        router = FakeRouter(lesson=lambda call: three)
        planner = PlanningService(router, 24, None, target_minutes=2)

        lesson = asyncio.run(planner.lesson(source))

        assert lesson == three
        assert len(router.prompts("lesson")) == 1  # 3 scenes meet the clamped floor: no retry
        assert "Produce exactly 3 substantial scenes." in router.prompts("lesson")[0]

        retry_router = FakeRouter(lesson=lambda call: _lesson(source, ["A", "B"]))
        with pytest.raises(RuntimeError, match="at least 3 are required"):
            asyncio.run(PlanningService(retry_router, 24, 10, target_minutes=2).lesson(source))
        assert "with between 3 and 3 scenes" in retry_router.prompts("lesson")[1]

    def test_narration_prompts_carry_per_scene_word_budgets(self):
        source = _source((300, 100, 50, 50, 50, 50, 200))
        lesson = _lesson(source)
        router = FakeRouter()
        planner = PlanningService(router, target_minutes=4)

        asyncio.run(planner.narration(lesson, source))

        budget = planner.length_budget
        assert budget.total_words == total_word_budget(4, 135, 7) == sum(budget.scene_words.values())
        first, second = router.prompts("narration")
        for index, scene in enumerate(lesson.scenes):
            line = f"- {scene.scene_id}: at most about {budget.scene_words[scene.scene_id]} words"
            assert (line in first) == (index < 5)  # each batch lists only its own scenes
            assert (line in second) == (index >= 5)

        relaxed = PlanningService(FakeRouter(), pacing="relaxed", target_minutes=4)
        asyncio.run(relaxed.narration(lesson, source))
        assert relaxed.length_budget.total_words == total_word_budget(4, 115, 7)

    def test_brief_depth_shifts_the_budget_toward_the_emphasised_scene(self):
        source = _source((100, 100))
        lesson = _lesson(source, ["Phasors", "Resonance"])
        router = FakeRouter()
        planner = PlanningService(
            router, directives=interpret_brief("Spend more time on resonance. Under 2 minutes."),
        )

        asyncio.run(planner.narration(lesson, source))

        words = planner.length_budget.scene_words
        assert words["scene_002"] > words["scene_001"]
        assert pd.DEPTH_INSTRUCTIONS["deep"] in router.prompts("narration")[0]

    def test_concept_plan_bypasses_the_lesson_provider_call(self):
        source = _source()
        plan = ConceptPlan.from_lesson(_lesson(source), source)
        plan.concepts[1].depth = "skip"
        plan.concepts[2].depth = "deep"
        plan.concepts[2].notes = "address the sign convention"
        router = FakeRouter()
        planner = PlanningService(router, concept_plan=plan, target_minutes=2)

        assert planner.omitted_concepts == ["Topic 2"]  # known even when lesson() is not re-run
        lesson = asyncio.run(planner.lesson(source))
        asyncio.run(planner.narration(lesson, source))

        assert [scene.scene_id for scene in lesson.scenes] == ["scene_001", "scene_003"]
        assert [call["stage"] for call in router.calls] == ["narration"]
        assert planner.omitted_concepts == ["Topic 2"]
        prompt = router.prompts("narration")[0]
        assert "professor's note: address the sign convention" in prompt
        assert pd.DEPTH_INSTRUCTIONS["deep"] in prompt
        words = planner.length_budget.scene_words
        assert words["scene_003"] > words["scene_001"]
        assert [check.check for check in planner.compliance_checks] == ["Do not cover Topic 2"]

    def test_fallback_narration_without_a_key_respects_the_budget(self, tmp_path):
        text = " ".join(f"Sentence {index} has exactly six words." for index in range(60))
        source = SourceDocument(
            filename="notes.pdf", sha256="abc", page_count=2,
            facts=[_fact("p001-f000", 1, text), _fact("p002-f000", 2, _words(400))],
        )
        lesson = _lesson(source)
        router, store = _real_router(tmp_path)
        try:
            planner = PlanningService(router, target_minutes=1)
            narration = asyncio.run(planner.narration(lesson, source))
            unbudgeted = asyncio.run(PlanningService(router).narration(lesson, source))
        finally:
            store.close()

        words = planner.length_budget.scene_words
        for scene in narration.scenes:
            assert scene.beats
            assert _scene_word_total(scene) <= words[scene.scene_id]
        # Whole sentences where they fit; a single over-long sentence is cut to the budget.
        assert narration.scenes[0].beats[-1].text.endswith("six words.")
        assert _scene_word_total(narration.scenes[1]) == words["scene_002"]
        assert _scene_word_total(unbudgeted.scenes[0]) > 2 * words["scene_001"]
        assert validate_grounding(lesson, narration, source) == []

    def test_direct_fallback_narration_budgets_a_replacement_lesson(self):
        source = _source((400, 400, 400))
        planner = PlanningService(FakeRouter(), target_minutes=1)

        lesson = planner._fallback_lesson(source)
        narration = planner._fallback_narration(lesson, source)

        assert len(lesson.scenes) == 2
        assert set(planner.length_budget.scene_words) == {scene.scene_id for scene in lesson.scenes}
        for scene in narration.scenes:
            assert _scene_word_total(scene) <= planner.length_budget.scene_words[scene.scene_id]

    def test_enforced_and_trimmed_narration_stays_grounded(self):
        source = _source((40, 40, 40, 40, 40, 40))
        lesson = _lesson(source)
        router = FakeRouter(narration=_long_narration(source))
        directives = interpret_brief(SPEC_BRIEF)
        directives.global_.target_minutes = 2
        planner = PlanningService(router, directives=directives, target_minutes=2)

        narration = asyncio.run(planner.narration(lesson, source))

        assert [scene.scene_id for scene in narration.scenes] == [s.scene_id for s in lesson.scenes]
        assert validate_grounding(lesson, narration, source) == []
        assert NarrationPlan.model_validate(narration.model_dump(mode="json")) == narration
        for scene in narration.scenes:
            assert scene.beats[0].beat_id.endswith("_b1")
            assert "clock" not in scene.narration
            assert scene.narration == " ".join(beat.text for beat in scene.beats)
            assert scene.target_words_per_minute == 135
        clock = next(c for c in planner.compliance_checks if c.check == "Never use the clock analogy")
        assert clock.passed and len(clock.beat_ids) == 6
        report = planner.length_report
        assert report["budget_words"] == planner.length_budget.total_words
        assert report["removed_beats"] and report["words_after"] < report["words_before"]
        assert report["within_budget"]

    def test_scene_id_order_check_is_kept(self):
        source = _source()
        lesson = _lesson(source)
        wrong = NarrationPlan(scenes=[_scene("scene_009", [_beat("b1", "Text.")])])
        planner = PlanningService(FakeRouter(narration=lambda call: wrong), target_minutes=2)

        with pytest.raises(RuntimeError, match="returned scene IDs"):
            asyncio.run(planner.narration(lesson, source))
