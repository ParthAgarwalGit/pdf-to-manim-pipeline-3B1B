"""V2 contract and grounding tests."""

import asyncio

import pytest
from pydantic import ValidationError

from src.pipeline_v2.models import (
    ArtifactCategory,
    ArtifactFootprint,
    ArtifactSpec,
    BeatComposition,
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
        ("google", "gemini-3.7-flash"),
        ("google", "gemini-3.7-flash"),
    ]
    assert [call["max_output_tokens"] for call in calls] == [7000, 9000]
    assert "Prefer 4-6 substantial scenes" in calls[0]["prompt"]
    assert "calm teaching pace" in calls[1]["prompt"]
    assert "pause_after_seconds" in calls[1]["prompt"]


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
