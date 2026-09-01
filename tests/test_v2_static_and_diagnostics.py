"""Tracked AST and runtime diagnostic tests."""

import json

from manim import Dot, Text, VGroup
from PIL import Image, ImageDraw

from src.pipeline_v2.diagnostics import analyze_trace
from src.pipeline_v2.models import (
    ArtifactCategory,
    ArtifactSpec,
    BeatComposition,
    LayoutRegion,
    LayoutTemplate,
    NarratedScene,
    NarrationBeat,
    VisualContract,
)
from src.pipeline_v2.static_verifier import V2StaticVerifier
from src.pipeline_v2.tracked_scene import TrackedScene
from src.pipeline_v2.rendering import frame_diagnostics


SAFE_CODE = '''from manim import *
from src.pipeline_v2.tracked_scene import TrackedScene

class Demo(TrackedScene):
    def construct(self):
        self.set_layout("full_visual")
        dot = Dot()
        self.place_in_region(dot, "content_full")
        self.register_artifact("dot", dot, category="primary_visual", purpose="Show point", beat_ids=["b1"])
        self.set_beat("b1")
        self.play(FadeIn(dot))
'''


def test_v2_static_accepts_fully_tracked_scene():
    report = V2StaticVerifier(
        {"dot"}, {"b1"}, expected_layout="full_visual",
        expected_regions={"dot": "content_full"},
    ).verify(SAFE_CODE)
    assert report.safe


def test_v2_static_requires_contract_layout_and_region_placement():
    code = SAFE_CODE.replace('self.set_layout("full_visual")', 'self.set_layout("split")')
    code = code.replace('self.place_in_region(dot, "content_full")', 'self.place_in_region(dot, "content_right")')
    report = V2StaticVerifier(
        {"dot"}, {"b1"}, expected_layout="full_visual",
        expected_regions={"dot": "content_full"},
    ).verify(code)
    codes = {item.code for item in report.issues}
    assert {"MISSING_LAYOUT_TEMPLATE", "REGION_PLACEMENT_MISMATCH"}.issubset(codes)
    assert not report.safe


def test_registered_group_recognizes_visible_descendants():
    scene = TrackedScene()
    scene._artifact_registry = {}
    child = Dot()
    group = VGroup(child)
    scene.register_artifact(
        "diagram", group, category="primary_visual",
        purpose="Show the diagram", beat_ids=["b1"],
    )

    record = scene._artifact_record(id(child), child)

    assert record is not None
    assert record["artifact_id"] == "diagram"


def test_registered_group_traces_individual_text_leaves():
    scene = TrackedScene()
    scene._artifact_registry = {}
    group = VGroup(Text("left"), Text("right"))
    scene.register_artifact(
        "diagram", group, category="primary_visual",
        purpose="Show the diagram", beat_ids=["b1"],
    )

    objects = scene._trace_objects([group])

    assert [item["artifact_id"] for item in objects] == ["diagram", "diagram"]
    assert [item["object_id"] for item in objects] == ["diagram.0", "diagram.1"]
    assert [item["category"] for item in objects] == ["semantic_text", "semantic_text"]


def test_registered_group_measures_occupied_pixel_overlap():
    scene = TrackedScene()
    scene._artifact_registry = {}
    group = VGroup(Text("same"), Text("same"))
    scene.register_artifact(
        "diagram", group, category="primary_visual",
        purpose="Show the diagram", beat_ids=["b1"],
    )

    overlaps = scene._pixel_overlap_records(scene._trace_objects([group]))

    assert len(overlaps) == 1
    assert overlaps[0]["ratio"] > 0.5


def test_runtime_detector_ignores_pre_animation_beat_checkpoint(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Show it.",
        beats=[NarrationBeat(beat_id="b1", text="Show", source_fact_ids=["f"])],
    )
    artifact = ArtifactSpec(
        artifact_id="diagram", category=ArtifactCategory.PRIMARY_VISUAL,
        role="diagram", purpose="Explain", source_fact_ids=["f"],
        beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
    )
    contract = VisualContract(
        scene_id="scene", artifact_inventory=[artifact],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["diagram"])],
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join([
        json.dumps({
            "event": "beat", "time": 0, "beat_id": "b1",
            "objects": [{
                "artifact_id": None, "category": "unregistered", "class_name": "VGroup",
                "bounds": {"left": -8, "right": 8, "top": 5, "bottom": -5,
                           "width": 16, "height": 10},
            }],
        }),
        json.dumps({
            "event": "play", "time": 1, "beat_id": "b1",
            "objects": [{
                "artifact_id": "diagram", "category": "primary_visual",
                "class_name": "VGroup",
                "bounds": {"left": -1, "right": 1, "top": 1, "bottom": -1,
                           "width": 2, "height": 2},
            }],
        }),
        json.dumps({
            "event": "play", "time": 2, "beat_id": "b1", "objects": [],
        }),
    ]), encoding="utf-8")

    assert analyze_trace(contract, scene, trace, 1.0) == []


def test_runtime_detector_allows_artifact_to_enter_later_in_same_beat(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Build the diagram.",
        beats=[NarrationBeat(beat_id="b1", text="Build", source_fact_ids=["f"])],
    )
    artifact = ArtifactSpec(
        artifact_id="diagram", category=ArtifactCategory.PRIMARY_VISUAL,
        role="diagram", purpose="Explain", source_fact_ids=["f"],
        beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
    )
    contract = VisualContract(
        scene_id="scene", artifact_inventory=[artifact],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["diagram"])],
    )
    bounds = {
        "left": -1, "right": 1, "top": 1, "bottom": -1,
        "width": 2, "height": 2,
    }
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join([
        json.dumps({"event": "play", "time": 1, "beat_id": "b1", "objects": []}),
        json.dumps({
            "event": "play", "time": 2, "beat_id": "b1",
            "objects": [{
                "artifact_id": "diagram", "category": "primary_visual",
                "class_name": "VGroup", "bounds": bounds,
            }],
        }),
    ]), encoding="utf-8")

    codes = {item.code for item in analyze_trace(contract, scene, trace, 2.0)}
    assert "MISSING_OBJECT" not in codes


def test_v2_static_rejects_missing_registration_and_latex():
    report = V2StaticVerifier({"dot", "label"}, {"b1", "b2"}).verify(
        SAFE_CODE.replace("self.play", 'bad = MathTex(r"\\centernot\\implies")\n        self.play')
    )
    codes = {item.code for item in report.issues}
    assert "MISSING_ARTIFACT_REGISTRATION" in codes
    assert "MISSING_BEAT_CHECKPOINT" in codes
    assert "UNSUPPORTED_LATEX" in codes
    assert not report.safe


def test_v2_static_reports_raw_coordinates_when_axes_exist():
    code = SAFE_CODE.replace(
        "dot = Dot()",
        "axes = Axes()\n        dot = Dot([8, 4, 0])",
    )
    codes = {item.code for item in V2StaticVerifier({"dot"}, {"b1"}).verify(code).issues}
    assert "OFFSCREEN_LITERAL_COORDINATE" in codes
    assert "RAW_COORDINATE_WITH_AXES" in codes


def test_runtime_detector_reports_lifecycle_bounds_overlap_and_audio(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="One two three", audio_duration_seconds=8.0,
        beats=[
            NarrationBeat(beat_id="b1", text="One", source_fact_ids=["f"]),
            NarrationBeat(beat_id="b2", text="Two", source_fact_ids=["f"]),
            NarrationBeat(beat_id="b3", text="Three", source_fact_ids=["f"]),
        ],
    )
    artifacts = [
        ArtifactSpec(
            artifact_id="title", category=ArtifactCategory.SEMANTIC_TEXT,
            role="title", purpose="Orient", content="Title", source_fact_ids=["f"],
            beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
        ),
        ArtifactSpec(
            artifact_id="label", category=ArtifactCategory.SEMANTIC_TEXT,
            role="label", purpose="Label", content="Label", source_fact_ids=["f"],
            beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
        ),
        ArtifactSpec(
            artifact_id="diagram", category=ArtifactCategory.PRIMARY_VISUAL,
            role="diagram", purpose="Explain", source_fact_ids=["f"],
            beat_ids=["b2"], enter_beat="b2", exit_beat="b2",
        ),
    ]
    contract = VisualContract(
        scene_id="scene", artifact_inventory=artifacts,
        composition_plan=[
            BeatComposition(beat_id="b1", active_artifact_ids=["title", "label"]),
            BeatComposition(beat_id="b2", active_artifact_ids=["diagram"]),
            BeatComposition(beat_id="b3", active_artifact_ids=[]),
        ],
    )
    bounds = {"left": -1, "right": 1, "top": 1, "bottom": 0, "width": 2, "height": 1}
    trace = [
        {"event": "play", "time": 1, "beat_id": "b1", "objects": [
            {"artifact_id": "title", "category": "semantic_text", "class_name": "Text", "font_size": 12,
             "bounds": {"left": -1, "right": 7.2, "top": 0.1, "bottom": 0,
                        "width": 8.2, "height": 0.1}},
            {"artifact_id": "label", "category": "semantic_text", "class_name": "Text", "font_size": 24,
             "bounds": bounds},
            {"artifact_id": "diagram", "category": "primary_visual", "class_name": "Circle", "bounds": bounds},
            {"artifact_id": None, "category": "unregistered", "class_name": "Square", "bounds": bounds},
        ]},
        {"event": "wait", "time": 6, "beat_id": "b3", "objects": [
            {"artifact_id": "title", "category": "semantic_text", "class_name": "Text", "font_size": 24,
             "bounds": bounds},
        ]},
    ]
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("\n".join(json.dumps(item) for item in trace))
    codes = {item.code for item in analyze_trace(contract, scene, trace_path, 3.0)}
    assert {"EARLY_ENTRY", "LATE_EXIT", "UNPLANNED_OBJECT", "TEXT_FRAME_COLLISION",
            "TEXT_TOO_SMALL", "TEXT_TEXT_OVERLAP", "TEXT_GEOMETRY_OVERLAP",
            "AUDIO_DURATION_MISMATCH"}.issubset(codes)


def test_overlap_detector_finds_text_children_inside_one_artifact(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Compare labels.",
        beats=[NarrationBeat(beat_id="b1", text="Compare", source_fact_ids=["f"])],
    )
    artifact = ArtifactSpec(
        artifact_id="diagram", category=ArtifactCategory.PRIMARY_VISUAL,
        role="diagram", purpose="Compare labels", source_fact_ids=["f"],
        beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
    )
    contract = VisualContract(
        scene_id="scene", artifact_inventory=[artifact],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["diagram"])],
    )
    bounds = {"left": -1, "right": 1, "top": 1, "bottom": 0, "width": 2, "height": 1}
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "event": "play", "time": 1.0, "beat_id": "b1",
        "objects": [
            {"artifact_id": "diagram", "object_id": "diagram.0", "category": "semantic_text",
             "class_name": "Text", "bounds": bounds},
            {"artifact_id": "diagram", "object_id": "diagram.1", "category": "semantic_text",
             "class_name": "Text", "bounds": bounds},
        ],
        "pixel_overlaps": [{
            "left_object_id": "diagram.0", "right_object_id": "diagram.1", "ratio": 0.8,
        }],
    }), encoding="utf-8")

    findings = analyze_trace(contract, scene, trace, 1.0)
    overlap = next(item for item in findings if item.code == "TEXT_TEXT_OVERLAP")
    assert overlap.severity == "error"
    assert overlap.artifact_ids == ["diagram"]
    assert overlap.object_ids == ["diagram.0", "diagram.1"]
    assert "80% occupied pixels" in overlap.message


def test_background_policy_does_not_hide_text_children_inside_artifact(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Compare labels.",
        beats=[NarrationBeat(beat_id="b1", text="Compare", source_fact_ids=["f"])],
    )
    artifact = ArtifactSpec(
        artifact_id="panel", category=ArtifactCategory.PRIMARY_VISUAL,
        role="panel", purpose="Show a labeled panel", source_fact_ids=["f"],
        beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
        overlap_policy="background",
    )
    contract = VisualContract(
        scene_id="scene", artifact_inventory=[artifact],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["panel"])],
    )
    bounds = {"left": -1, "right": 1, "top": 1, "bottom": 0, "width": 2, "height": 1}
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "event": "play", "time": 1.0, "beat_id": "b1",
        "objects": [
            {"artifact_id": "panel", "object_id": "panel.0", "category": "semantic_text",
             "class_name": "Text", "bounds": bounds},
            {"artifact_id": "panel", "object_id": "panel.1", "category": "semantic_text",
             "class_name": "Text", "bounds": bounds},
        ],
        "pixel_overlaps": [{
            "left_object_id": "panel.0", "right_object_id": "panel.1", "ratio": 0.8,
        }],
    }), encoding="utf-8")

    codes = {item.code for item in analyze_trace(contract, scene, trace, 1.0)}
    assert "TEXT_TEXT_OVERLAP" in codes


def test_pixel_confirmation_rejects_bbox_only_false_overlap(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Separate labels.",
        beats=[NarrationBeat(beat_id="b1", text="Separate", source_fact_ids=["f"])],
    )
    artifacts = [
        ArtifactSpec(
            artifact_id=value, category=ArtifactCategory.SEMANTIC_TEXT,
            role="label", purpose="Label", source_fact_ids=["f"],
            beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
        ) for value in ("left", "right")
    ]
    contract = VisualContract(
        scene_id="scene", artifact_inventory=artifacts,
        composition_plan=[BeatComposition(
            beat_id="b1", active_artifact_ids=["left", "right"],
        )],
    )
    bounds = {"left": -1, "right": 1, "top": 1, "bottom": 0, "width": 2, "height": 1}
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "event": "play", "time": 1.0, "beat_id": "b1",
        "objects": [
            {"artifact_id": "left", "object_id": "left", "category": "semantic_text",
             "class_name": "Text", "bounds": bounds},
            {"artifact_id": "right", "object_id": "right", "category": "semantic_text",
             "class_name": "Text", "bounds": bounds},
        ],
        "pixel_overlaps": [{
            "left_object_id": "left", "right_object_id": "right", "ratio": 0.01,
        }],
    }), encoding="utf-8")

    codes = {item.code for item in analyze_trace(contract, scene, trace, 1.0)}
    assert "TEXT_TEXT_OVERLAP" not in codes


def test_partial_text_overlap_is_warning_without_pixel_confirmation(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Labels.",
        beats=[NarrationBeat(beat_id="b1", text="Labels", source_fact_ids=["f"])],
    )
    artifacts = [
        ArtifactSpec(
            artifact_id=value, category=ArtifactCategory.SEMANTIC_TEXT,
            role="label", purpose="Label", source_fact_ids=["f"],
            beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
        ) for value in ("left", "right")
    ]
    contract = VisualContract(
        scene_id="scene", artifact_inventory=artifacts,
        composition_plan=[BeatComposition(
            beat_id="b1", active_artifact_ids=["left", "right"],
        )],
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "event": "play", "time": 1.0, "beat_id": "b1",
        "objects": [
            {"artifact_id": "left", "category": "semantic_text", "class_name": "Text",
             "bounds": {"left": 0, "right": 2, "top": 1, "bottom": 0,
                        "width": 2, "height": 1}},
            {"artifact_id": "right", "category": "semantic_text", "class_name": "Text",
             "bounds": {"left": 1.7, "right": 3.7, "top": 1, "bottom": 0,
                        "width": 2, "height": 1}},
        ],
    }), encoding="utf-8")

    overlap = next(
        item for item in analyze_trace(contract, scene, trace, 1.0)
        if item.code == "TEXT_TEXT_OVERLAP"
    )
    assert overlap.severity == "warning"
    assert "15%" in overlap.message


def test_runtime_detector_reports_missing_offscreen_and_weighted_density(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Dense scene",
        beats=[NarrationBeat(beat_id="b1", text="Dense", source_fact_ids=["f"])],
    )
    artifacts = [
        ArtifactSpec(
            artifact_id=f"item-{index}", category=ArtifactCategory.PRIMARY_VISUAL,
            role="shape", purpose="Explain one part", source_fact_ids=["f"],
            beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
        )
        for index in range(7)
    ]
    artifacts.append(ArtifactSpec(
        artifact_id="missing", category=ArtifactCategory.SEMANTIC_TEXT,
        role="label", purpose="Name result", source_fact_ids=["f"],
        beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
    ))
    contract = VisualContract(
        scene_id="scene", artifact_inventory=artifacts,
        composition_plan=[BeatComposition(
            beat_id="b1", active_artifact_ids=[item.artifact_id for item in artifacts],
        )],
    )
    visible = [{
        "artifact_id": f"item-{index}", "category": "primary_visual",
        "class_name": "Circle",
        "bounds": {
            "left": 7.2 if index == 0 else index * 0.2,
            "right": 7.6 if index == 0 else index * 0.2 + 0.1,
            "bottom": 0.0, "top": 0.1, "width": 0.1, "height": 0.1,
        },
    } for index in range(7)]
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({
        "event": "wait", "time": 1.0, "beat_id": "b1", "objects": visible,
    }))
    codes = {finding.code for finding in analyze_trace(contract, scene, path, 1.0)}
    assert {"MISSING_OBJECT", "OFFSCREEN_OBJECT", "VISUAL_DENSITY_HIGH"}.issubset(codes)


def test_frame_diagnostics_report_empty_static_and_edge_clipping(tmp_path):
    empty_paths = []
    for index in range(3):
        path = tmp_path / f"empty-{index}.png"
        Image.new("RGB", (160, 90), "black").save(path)
        empty_paths.append(str(path))
    assert "VISUAL_MOSTLY_EMPTY" in {
        finding.code for finding in frame_diagnostics(empty_paths)
    }

    static_paths = []
    for index in range(3):
        path = tmp_path / f"static-{index}.png"
        image = Image.new("RGB", (160, 90), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 159, 89), outline="white", width=8)
        image.save(path)
        static_paths.append(str(path))
    codes = {finding.code for finding in frame_diagnostics(static_paths)}
    assert "VISUAL_STATIC" in codes
    assert "VISUAL_EDGE_CLIPPED" in codes


def test_runtime_detector_distinguishes_soft_and_hard_region_overflow(tmp_path):
    scene = NarratedScene(
        scene_id="scene", narration="Compare.",
        beats=[NarrationBeat(beat_id="b1", text="Compare", source_fact_ids=["f"])],
    )
    artifact = ArtifactSpec(
        artifact_id="left", category=ArtifactCategory.PRIMARY_VISUAL,
        role="diagram", purpose="Show left case", source_fact_ids=["f"],
        beat_ids=["b1"], enter_beat="b1", exit_beat="b1",
        region=LayoutRegion.CONTENT_LEFT, region_padding=0.12,
    )
    contract = VisualContract(
        scene_id="scene", layout_template=LayoutTemplate.SPLIT,
        artifact_inventory=[artifact],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["left"])],
    )
    path = tmp_path / "trace.jsonl"
    records = [
        {
            "event": "play", "time": 1.0, "beat_id": "b1",
            "objects": [{
                "artifact_id": "left", "object_id": "left.0",
                "category": "primary_visual", "class_name": "Square",
                "bounds": {"left": -6.4, "right": -0.25, "bottom": -2.3, "top": 2.3,
                           "width": 6.15, "height": 4.6},
            }],
        },
        {
            "event": "wait", "time": 2.0, "beat_id": "b1",
            "objects": [{
                "artifact_id": "left", "object_id": "left.0",
                "category": "primary_visual", "class_name": "Square",
                "bounds": {"left": -6.4, "right": 0.0, "bottom": -2.3, "top": 2.3,
                           "width": 6.4, "height": 4.6},
            }],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    findings = analyze_trace(contract, scene, path, 2.0)
    assert any(item.code == "REGION_SOFT_OVERFLOW" for item in findings)
    hard = next(item for item in findings if item.code == "REGION_HARD_OVERFLOW")
    assert hard.severity == "error"
    assert hard.object_ids == ["left.0"]
