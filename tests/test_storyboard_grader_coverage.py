"""Coverage-aware storyboard grader tests."""

from src.agents.storyboard_grader import StoryboardGraderAgent
from src.schemas.source_segment import SourceSegment
from src.schemas.storyboard import (
    Configuration,
    PDFMetadataStoryboard,
    Scene,
    StoryboardJSON,
    VisualIntent,
)


def _scene(scene_id: str, index: int, segment_ids: list[str]) -> Scene:
    return Scene(
        scene_id=scene_id,
        scene_index=index,
        concept_id=f"concept_{index}",
        narration="Narration with enough detail to satisfy the heuristic scorer.",
        visual_intent=VisualIntent(mathematical_objects=[], transformations=[], emphasis_points=[]),
        duration_estimate=30.0,
        dependencies=[f"scene_{index:03d}"] if index > 0 else [],
        source_segment_ids=segment_ids,
    )


def _segments(count: int) -> list[SourceSegment]:
    return [
        SourceSegment(
            segment_id=f"seg_{index:03d}",
            order=index - 1,
            page_start=index,
            page_end=index,
            text=f"Segment {index}",
        )
        for index in range(1, count + 1)
    ]


def test_heuristic_completeness_scores_real_segment_coverage():
    storyboard = StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(filename="paper.pdf", page_count=4),
        configuration=Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
        concept_hierarchy=[],
        scenes=[
            _scene("scene_001", 0, ["seg_001"]),
            _scene("scene_002", 1, ["seg_002"]),
            _scene("scene_003", 2, ["seg_003"]),
            _scene("scene_004", 3, ["seg_004"]),
        ],
    )

    report = StoryboardGraderAgent()._grade_heuristic(storyboard, source_segments=_segments(4))
    completeness = next(c for c in report.criteria if c.name == "completeness")

    assert completeness.score == 10.0
    assert "4/4 source segments" in completeness.feedback


def test_heuristic_completeness_penalizes_redundant_segment_scenes():
    storyboard = StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(filename="paper.pdf", page_count=4),
        configuration=Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
        concept_hierarchy=[],
        scenes=[
            _scene("scene_001", 0, ["seg_001"]),
            _scene("scene_002", 1, ["seg_001"]),
            _scene("scene_003", 2, ["seg_001"]),
            _scene("scene_004", 3, ["seg_001"]),
        ],
    )

    report = StoryboardGraderAgent()._grade_heuristic(storyboard, source_segments=_segments(4))
    completeness = next(c for c in report.criteria if c.name == "completeness")

    assert completeness.score < 4.0
    assert "redundant" in completeness.feedback.lower()
