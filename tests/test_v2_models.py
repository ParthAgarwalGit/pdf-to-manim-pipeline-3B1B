"""V2 contract and grounding tests."""

import asyncio

import pytest
from pydantic import ValidationError

from src.pipeline_v2.models import (
    ArtifactCategory,
    ArtifactFootprint,
    ArtifactSpec,
    BeatComposition,
    LessonPlan,
    LessonScene,
    LayoutRegion,
    LayoutTemplate,
    OverlapPolicy,
    NarratedScene,
    NarrationBeat,
    SourceDocument,
    SourceFact,
    VisualContract,
    PipelineV2Config,
)
from src.pipeline_v2.planning import (
    PlanningService,
    apply_audio_timing,
    validate_visual_contract,
)
from src.pipeline_v2.providers import _google_compatible_schema


def _scene():
    return NarratedScene(
        scene_id="scene_001", narration="Explain the idea.",
        beats=[
            NarrationBeat(beat_id="b1", text="Explain", source_fact_ids=["f1"]),
            NarrationBeat(beat_id="b2", text="Apply", source_fact_ids=["f1"]),
        ],
    )


def _artifact(**updates):
    values = {
        "artifact_id": "diagram", "category": ArtifactCategory.PRIMARY_VISUAL,
        "role": "diagram", "purpose": "Show the relationship",
        "source_fact_ids": ["f1"], "beat_ids": ["b1", "b2"],
        "enter_beat": "b1", "exit_beat": "b2",
    }
    values.update(updates)
    return ArtifactSpec(**values)


def test_visual_contract_rejects_unknown_artifact_reference():
    with pytest.raises(ValidationError):
        VisualContract(
            scene_id="scene_001", artifact_inventory=[_artifact(target_id="missing")],
            composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["diagram"])],
        )


def test_visual_contract_grounding_checks_order_and_lifecycle():
    contract = VisualContract(
        scene_id="scene_001",
        artifact_inventory=[_artifact(enter_beat="b2", exit_beat="b1")],
        composition_plan=[
            BeatComposition(beat_id="b2", active_artifact_ids=["diagram"]),
            BeatComposition(beat_id="b1", active_artifact_ids=["diagram"]),
        ],
    )
    source = SourceDocument(
        filename="x.pdf", sha256="a", page_count=1,
        facts=[SourceFact(fact_id="f1", page_number=1, text="Fact")],
    )
    errors = validate_visual_contract(contract, _scene(), source)
    assert any("once and in order" in error for error in errors)
    assert any("exits before" in error for error in errors)


def test_artifact_requires_purpose_and_beat():
    with pytest.raises(ValidationError):
        _artifact(purpose="", beat_ids=[])


def test_legacy_region_names_migrate_to_bounded_regions():
    assert _artifact(region="top").region == LayoutRegion.HEADER
    assert _artifact(region="full").region == LayoutRegion.CONTENT_FULL


def test_visual_contract_rejects_regions_outside_template_and_large_side_artifacts():
    contract = VisualContract(
        scene_id="scene_001", layout_template=LayoutTemplate.SPLIT,
        artifact_inventory=[_artifact(
            region=LayoutRegion.CONTENT_CENTER,
            footprint=ArtifactFootprint.LARGE,
        )],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["diagram"])],
    )
    source = SourceDocument(
        filename="x.pdf", sha256="a", page_count=1,
        facts=[SourceFact(fact_id="f1", page_number=1, text="Fact")],
    )
    errors = validate_visual_contract(contract, _scene(), source)
    assert any("unavailable" in error for error in errors)


def test_visual_contract_requires_one_independent_owner_per_region():
    contract = VisualContract(
        scene_id="scene_001", layout_template=LayoutTemplate.SPLIT,
        artifact_inventory=[
            _artifact(artifact_id="one", region=LayoutRegion.CONTENT_LEFT),
            _artifact(artifact_id="two", region=LayoutRegion.CONTENT_LEFT),
        ],
        composition_plan=[
            BeatComposition(beat_id="b1", active_artifact_ids=["one", "two"]),
            BeatComposition(beat_id="b2", active_artifact_ids=["one"]),
        ],
    )
    source = SourceDocument(
        filename="x.pdf", sha256="a", page_count=1,
        facts=[SourceFact(fact_id="f1", page_number=1, text="Fact")],
    )
    errors = validate_visual_contract(contract, _scene(), source)
    assert any("multiple independent artifacts" in error for error in errors)


def test_background_overlap_policy_is_reserved_for_primary_visuals():
    with pytest.raises(ValidationError):
        _artifact(
            category=ArtifactCategory.SEMANTIC_TEXT,
            overlap_policy=OverlapPolicy.BACKGROUND,
        )


def test_lesson_and_narration_use_gemini():
    calls = []

    class CapturingRouter:
        async def structured(self, **kwargs):
            calls.append(kwargs)
            return kwargs["fallback"]()

    source = SourceDocument(
        filename="x.pdf", sha256="a", page_count=1,
        facts=[SourceFact(fact_id="f1", page_number=1, text="BFS uses a queue.")],
    )
    planner = PlanningService(CapturingRouter(), max_scenes=6)

    async def exercise():
        lesson = await planner.lesson(source)
        await planner.narration(lesson, source)

    asyncio.run(exercise())

    assert [(call["provider"], call["model"]) for call in calls] == [
        ("google", "gemini-4-argon"),
        ("google", "gemini-4-argon"),
    ]
    assert [call["max_output_tokens"] for call in calls] == [7000, 7000]
    assert "Prefer 4-6 substantial scenes" in calls[0]["prompt"]
    assert "calm teaching pace" in calls[1]["prompt"]
    assert "pause_after_seconds" in calls[1]["prompt"]


def test_long_document_requests_full_coverage_scene_range():
    calls = []

    class CapturingRouter:
        async def structured(self, **kwargs):
            calls.append(kwargs)
            return kwargs["fallback"]()

    source = SourceDocument(
        filename="long.pdf", sha256="a", page_count=30,
        facts=[
            SourceFact(fact_id=f"p{page:03d}", page_number=page, text=f"Fact {page}")
            for page in range(1, 31)
        ],
    )
    planner = PlanningService(CapturingRouter(), max_scenes=18)

    lesson = asyncio.run(planner.lesson(source))

    assert len(lesson.scenes) == 18
    assert "Full-coverage mode" in calls[0]["prompt"]
    assert "exactly 18 substantial scenes" in calls[0]["prompt"]
    assert "every substantive source page" in calls[0]["prompt"]
    assert calls[0]["max_output_tokens"] == 13500
    assert '"bbox"' not in calls[0]["prompt"]


def test_explicit_scene_floor_replans_once_when_first_plan_is_too_short():
    calls = []

    class ShortThenFallbackRouter:
        async def structured(self, **kwargs):
            calls.append(kwargs)
            fallback = kwargs["fallback"]()
            if len(calls) == 1:
                return fallback.model_copy(update={"scenes": fallback.scenes[:5]})
            return fallback

    source = SourceDocument(
        filename="long.pdf", sha256="a", page_count=20,
        facts=[
            SourceFact(fact_id=f"p{page:03d}", page_number=page, text=f"Fact {page}")
            for page in range(1, 21)
        ],
    )
    planner = PlanningService(
        ShortThenFallbackRouter(), max_scenes=12, min_scenes=10,
    )

    lesson = asyncio.run(planner.lesson(source))

    assert len(lesson.scenes) == 10
    assert [call["task_id"] for call in calls] == ["lesson", "lesson-expand"]
    assert "required minimum of 10" in calls[1]["prompt"]


def test_explicit_scene_floor_rejects_a_second_short_plan():
    class AlwaysShortRouter:
        async def structured(self, **kwargs):
            fallback = kwargs["fallback"]()
            return fallback.model_copy(update={"scenes": fallback.scenes[:4]})

    source = SourceDocument(
        filename="long.pdf", sha256="a", page_count=20,
        facts=[
            SourceFact(fact_id=f"p{page:03d}", page_number=page, text=f"Fact {page}")
            for page in range(1, 21)
        ],
    )
    planner = PlanningService(AlwaysShortRouter(), max_scenes=12, min_scenes=10)

    with pytest.raises(RuntimeError, match="at least 10 are required"):
        asyncio.run(planner.lesson(source))


def test_relaxed_pacing_is_applied_even_if_provider_returns_faster_target():
    calls = []

    class CapturingRouter:
        async def structured(self, **kwargs):
            calls.append(kwargs)
            return kwargs["fallback"]()

    source = SourceDocument(
        filename="x.pdf", sha256="a", page_count=1,
        facts=[SourceFact(fact_id="f1", page_number=1, text="A circle has a radius.")],
    )
    planner = PlanningService(CapturingRouter(), max_scenes=6, pacing="relaxed")

    async def exercise():
        lesson = await planner.lesson(source)
        return await planner.narration(lesson, source)

    narration = asyncio.run(exercise())

    assert narration.scenes[0].target_words_per_minute == 115
    assert "relaxed pace of 110-120 words per minute" in calls[1]["prompt"]


def test_long_narration_is_split_into_bounded_continuity_batches():
    calls = []

    class CapturingRouter:
        async def structured(self, **kwargs):
            calls.append(kwargs)
            return kwargs["fallback"]()

    facts = [
        SourceFact(fact_id=f"f{index}", page_number=index + 1, text=f"Fact {index}")
        for index in range(11)
    ]
    source = SourceDocument(filename="x.pdf", sha256="a", page_count=11, facts=facts)
    lesson = LessonPlan(
        title="Test lesson", audience="general", objectives=["Learn"],
        scenes=[
            LessonScene(
                scene_id=f"scene_{index + 1:03d}", title=f"Part {index + 1}",
                description=f"Explain fact {index}", learning_goal=f"Learn fact {index}",
                source_fact_ids=[f"f{index}"], new_visual_fact=f"Fact {index}",
            )
            for index in range(11)
        ],
    )
    planner = PlanningService(CapturingRouter(), max_scenes=12, pacing="relaxed")

    narration = asyncio.run(planner.narration(lesson, source))

    assert len(narration.scenes) == 11
    assert [call["task_id"] for call in calls] == [
        "narration-000", "narration-001", "narration-002",
    ]
    assert all(call["max_output_tokens"] == 7000 for call in calls)
    assert '"previous_scene": null' in calls[0]["prompt"]
    assert '"previous_scene": {' in calls[1]["prompt"]


def test_pipeline_config_rejects_scene_floor_above_ceiling():
    with pytest.raises(ValidationError, match="min_scenes cannot exceed max_scenes"):
        PipelineV2Config(min_scenes=13, max_scenes=12)


def test_google_schema_keeps_local_limit_without_sending_max_items():
    original = {
        "type": "object",
        "properties": {
            "scenes": {"type": "array", "maxItems": 24, "items": {"type": "object"}},
        },
    }

    compatible = _google_compatible_schema(original)

    assert "maxItems" not in compatible["properties"]["scenes"]
    assert original["properties"]["scenes"]["maxItems"] == 24


def test_audio_timing_reserves_declared_pause_inside_beat_window():
    scene = NarratedScene(
        scene_id="scene",
        narration="One two. Three four.",
        beats=[
            NarrationBeat(
                beat_id="b1", text="One two.", source_fact_ids=["f1"],
                pause_after_seconds=1.0,
            ),
            NarrationBeat(
                beat_id="b2", text="Three four.", source_fact_ids=["f1"],
                pause_after_seconds=0.0,
            ),
        ],
    )

    timed = apply_audio_timing(scene, 10.0)

    assert timed.beats[0].end_seconds == 5.5
    assert timed.beats[1].start_seconds == 5.5
    assert timed.beats[1].end_seconds == 10.0


def test_repairs_default_to_three_per_scene_without_a_run_cap():
    config = PipelineV2Config()

    assert config.max_repairs_per_scene == 3
    assert not hasattr(config, "max_repairs_per_run")
