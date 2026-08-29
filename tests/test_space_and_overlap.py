"""Phase 3 — screen-space and overlap enforcement, all deterministic.

A real run averaged 16.7 objects/scene with unclamped bboxes: overlaps, text
walls, and content past the frame edge. These tests pin the clamp, the object
cap, the time-aware overlap nudge, and the border-ink frame metric.
"""

import numpy as np
import pytest
from PIL import Image

from src.agents.script_architect import (
    SAFE_X,
    SAFE_Y,
    ScriptArchitectAgent,
    _coerce_visual_object,
    _resolve_bbox_overlaps,
)
from src.schemas.storyboard import BBox, MathematicalObject, TimeInterval
from src.utils.frame_analysis import analyze_keyframes, border_ink_fraction


def _obj(oid, x, y, w=2.0, h=2.0, t_in=0.0, t_out=10.0):
    return MathematicalObject(
        object_id=oid,
        object_type="geometric_shape",
        content="",
        primitive="Circle",
        args={"radius": 1.0},
        bbox=BBox(x=x, y=y, w=w, h=h),
        interval=TimeInterval(t_in=t_in, t_out=t_out),
    )


# ---------------------------------------------------------------------------
# 3.1 — safe-area clamp in coercion
# ---------------------------------------------------------------------------

class TestBBoxClamp:
    def test_offscreen_centre_clamped(self):
        o = _coerce_visual_object({
            "primitive": "Circle", "args": {},
            "bbox": {"x": 9.0, "y": -6.0, "w": 2.0, "h": 2.0},
        })
        assert o["bbox"]["x"] == SAFE_X
        assert o["bbox"]["y"] == -SAFE_Y

    def test_oversized_box_capped(self):
        o = _coerce_visual_object({
            "primitive": "Circle", "args": {},
            "bbox": {"x": 0.0, "y": 0.0, "w": 30.0, "h": 20.0},
        })
        assert o["bbox"]["w"] == 2 * SAFE_X
        assert o["bbox"]["h"] == 2 * SAFE_Y

    def test_in_bounds_box_untouched(self):
        o = _coerce_visual_object({
            "primitive": "Circle", "args": {},
            "bbox": {"x": 1.5, "y": -2.0, "w": 3.0, "h": 1.0},
        })
        assert o["bbox"] == {"x": 1.5, "y": -2.0, "w": 3.0, "h": 1.0}


# ---------------------------------------------------------------------------
# 3.2 — time-aware overlap nudge
# ---------------------------------------------------------------------------

class TestOverlapNudge:
    def _overlap(self, a, b):
        dx = (a.bbox.w + b.bbox.w) / 2 - abs(a.bbox.x - b.bbox.x)
        dy = (a.bbox.h + b.bbox.h) / 2 - abs(a.bbox.y - b.bbox.y)
        return dx > 0 and dy > 0

    def test_concurrent_overlap_resolved(self):
        a, b = _obj("a", 0.0, 0.0), _obj("b", 0.5, 0.1)
        _resolve_bbox_overlaps([a, b])
        assert not self._overlap(a, b)

    def test_disjoint_time_windows_left_alone(self):
        a = _obj("a", 0.0, 0.0, t_in=0.0, t_out=5.0)
        b = _obj("b", 0.5, 0.1, t_in=5.0, t_out=10.0)  # appears after a leaves
        bx = b.bbox.x
        _resolve_bbox_overlaps([a, b])
        assert b.bbox.x == bx  # untouched

    def test_result_stays_in_safe_area(self):
        a = _obj("a", 6.0, 0.0),
        objs = [_obj("a", 6.0, 0.0), _obj("b", 6.3, 0.0)]
        _resolve_bbox_overlaps(objs)
        for o in objs:
            assert -SAFE_X <= o.bbox.x <= SAFE_X
            assert -SAFE_Y <= o.bbox.y <= SAFE_Y

    def test_non_overlapping_layout_unchanged(self):
        a, b = _obj("a", -3.0, 0.0), _obj("b", 3.0, 0.0)
        ax, bx = a.bbox.x, b.bbox.x
        _resolve_bbox_overlaps([a, b])
        assert (a.bbox.x, b.bbox.x) == (ax, bx)


# ---------------------------------------------------------------------------
# 3.3 — object cap in the intent parser
# ---------------------------------------------------------------------------

class TestObjectCap:
    def test_plan_under_new_limit_is_not_silently_trimmed(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        objects = [
            {
                "object_id": f"o{i:02d}", "primitive": "Circle",
                "args": {"radius": 1.0},
                "bbox": {"x": float(i % 6) - 3, "y": float(i // 6), "w": 1.0, "h": 1.0},
                "interval": {"t_in": float(i), "t_out": float(i) + 5.0},
            }
            for i in range(12)
        ]
        intent = agent._visual_intent_from_json({"objects": objects}, "n")
        assert len(intent.mathematical_objects) == 12
        kept = [o.object_id for o in intent.mathematical_objects]
        assert kept == [f"o{i:02d}" for i in range(12)]

    def test_small_plan_not_trimmed(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        objects = [{
            "object_id": "only", "primitive": "Circle", "args": {"radius": 1.0},
        }]
        intent = agent._visual_intent_from_json({"objects": objects}, "n")
        assert [o.object_id for o in intent.mathematical_objects] == ["only"]


# ---------------------------------------------------------------------------
# 3.5 — border-ink metric
# ---------------------------------------------------------------------------

def _frame_with_block(x0, x1, y0, y1, size=(100, 100)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = (20, 20, 20)  # dark background
    arr[y0:y1, x0:x1] = (200, 200, 200)
    return Image.fromarray(arr)


class TestBorderInk:
    def test_centered_content_scores_near_zero(self):
        frame = _frame_with_block(40, 60, 40, 60)
        assert border_ink_fraction(frame) == 0.0

    def test_edge_hugging_content_scores_high(self):
        frame = _frame_with_block(96, 100, 20, 80)  # strip on the right edge
        assert border_ink_fraction(frame) > 0.9

    def test_empty_frame_scores_zero(self):
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        assert border_ink_fraction(Image.fromarray(arr)) == 0.0

    def test_report_flags_clipped_frames(self):
        clipped = _frame_with_block(96, 100, 10, 90)
        centered = _frame_with_block(40, 60, 40, 60)
        report = analyze_keyframes([centered, clipped, centered, clipped])
        assert report.edge_clipped
        assert report.edge_clipped_indices == [1, 3]

    def test_report_clean_for_centered_frames(self):
        frames = [_frame_with_block(30 + i, 60 + i, 40, 60) for i in range(4)]
        report = analyze_keyframes(frames)
        assert not report.edge_clipped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
