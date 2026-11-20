"""Text about to appear across the frame edge is brought inside before it is shown."""

from manim import RIGHT, MathTex, Square, Text, VGroup, config

from src.pipeline_v2.diagnostics import _bounds_checks
from src.pipeline_v2.generation import REPAIR_FINDINGS_SHOWN, repair_findings
from src.pipeline_v2.models import VerificationFinding
from src.pipeline_v2.tracked_scene import MIN_FIT_SCALE, SAFE_MARGIN, TrackedScene, fit_into_frame, text_only

HALF_W = float(config.frame_width) / 2 - SAFE_MARGIN
HALF_H = float(config.frame_height) / 2 - SAFE_MARGIN


def _formula(left_edge: float, width: float) -> MathTex:
    formula = MathTex(r"\text{HIGH}", "=", "2m+1", "=", "3")
    formula.scale_to_fit_width(width)
    formula.shift(RIGHT * (left_edge - formula.get_left()[0]))
    return formula


def test_text_over_one_edge_shrinks_toward_its_inner_edge():
    formula = _formula(4.6, 2.4)
    fit = fit_into_frame(formula, HALF_W, HALF_H)
    assert fit is not None and MIN_FIT_SCALE < fit["scale"] < 1.0
    assert abs(formula.get_left()[0] - 4.6) < 1e-6  # still beside what it labels
    assert formula.get_right()[0] <= HALF_W + 1e-6
    assert fit["shift_x"] == 0.0


def test_text_that_cannot_shrink_enough_is_also_shifted():
    formula = _formula(5.9, 2.4)
    fit = fit_into_frame(formula, HALF_W, HALF_H)
    assert fit["scale"] == MIN_FIT_SCALE and fit["shift_x"] < 0
    assert formula.get_right()[0] <= HALF_W + 1e-6


def test_fitting_text_and_text_staged_off_screen_are_left_alone():
    inside = Text("ok").move_to([1, 1, 0])
    assert fit_into_frame(inside, HALF_W, HALF_H) is None
    staged = Text("later").move_to([12, 0, 0])
    assert fit_into_frame(staged, HALF_W, HALF_H) is None
    assert abs(staged.get_center()[0] - 12) < 1e-6


def test_only_text_is_ever_moved():
    assert text_only(MathTex("x")) and text_only(VGroup(Text("a"), MathTex("b")))
    assert not text_only(Square()) and not text_only(VGroup(Square(), Text("P")))


def test_a_formula_written_in_pieces_is_fitted_whole_once():
    scene = TrackedScene()
    scene.setup()
    formula = _formula(4.6, 2.4)
    scene.auto_register("high_formula", formula)
    heights = [part.height for part in formula]

    scene.add(formula[:3])  # the first piece appears: the whole formula is fitted
    scale = formula[0].height / heights[0]
    assert scale < 1.0
    assert all(abs(part.height / before - scale) < 1e-6 for part, before in zip(formula, heights))
    assert formula.get_right()[0] <= HALF_W + 1e-6

    right = formula.get_right()[0]
    scene.add(formula[3:])  # the rest appears: nothing moves again
    assert abs(formula.get_right()[0] - right) < 1e-9
    fits = scene._frame_fits
    assert [item["object_id"] for item in fits] == ["high_formula"]


def test_geometry_on_the_edge_is_not_moved():
    scene = TrackedScene()
    scene.setup()
    square = Square().move_to([6.8, 0, 0])
    scene.add(square)
    assert abs(square.get_center()[0] - 6.8) < 1e-9 and scene._frame_fits == []


def test_value_trackers_are_never_reported_off_screen():
    findings, seen = [], set()
    tracker = {"artifact_id": "theta_tr", "class_name": "ValueTracker",
               "bounds": {"left": 30, "right": 30, "top": 0, "bottom": 0, "width": 0, "height": 0}}
    _bounds_checks(findings, seen, tracker, 1.0)
    assert findings == []
    label = dict(tracker, artifact_id="label", class_name="MathTex",
                 bounds={"left": 6.5, "right": 7.5, "top": 0.2, "bottom": -0.2, "width": 1, "height": 0.4})
    _bounds_checks(findings, seen, label, 1.0)
    assert [item.code for item in findings] == ["TEXT_FRAME_COLLISION"]


def test_repair_sees_the_serious_findings_first_once_each():
    density = [VerificationFinding(code="VISUAL_DENSITY_HIGH", severity="warning", message="busy",
                                   object_ids=[f"o{i}"]) for i in range(20)]
    overlap = VerificationFinding(code="TEXT_GEOMETRY_OVERLAP", severity="warning", message="label",
                                  object_ids=["lbl", "arrow"])
    edge = VerificationFinding(code="TEXT_FRAME_COLLISION", severity="error", message="cut",
                               object_ids=["high_formula"])
    chosen = repair_findings(density + [overlap, overlap, edge])
    assert [item.code for item in chosen[:2]] == ["TEXT_FRAME_COLLISION", "TEXT_GEOMETRY_OVERLAP"]
    assert len(chosen) == REPAIR_FINDINGS_SHOWN
    assert sum(item.code == "TEXT_GEOMETRY_OVERLAP" for item in chosen) == 1
