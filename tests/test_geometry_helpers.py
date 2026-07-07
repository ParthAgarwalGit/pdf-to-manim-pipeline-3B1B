"""Phase 2 — geometric anchoring helpers + the hardcoded-endpoint lint.

The projection math is what makes "perpendicular distance" visuals land ON
the line at a true right angle instead of ending midway (the failure a real
run exhibited with hand-computed Arrow endpoints).
"""

import numpy as np
import pytest
from manim import Axes, Circle, DashedLine, Dot, Line, VGroup

from src.agents.manim_code_validator import ManimCodeValidatorAgent
from src.utils.geometry_helpers import (
    connect,
    drop_perpendicular,
    on_axes,
    perpendicular_foot,
)


# ---------------------------------------------------------------------------
# perpendicular_foot — pure math
# ---------------------------------------------------------------------------

class TestPerpendicularFoot:
    def test_foot_on_horizontal_line(self):
        foot = perpendicular_foot([2.0, 3.0, 0.0], [-5.0, 0.0, 0.0], [5.0, 0.0, 0.0])
        assert np.allclose(foot, [2.0, 0.0, 0.0])

    def test_foot_on_diagonal_line(self):
        # Project (0, 2) onto y = x — the foot must be (1, 1).
        foot = perpendicular_foot([0.0, 2.0, 0.0], [-3.0, -3.0, 0.0], [3.0, 3.0, 0.0])
        assert np.allclose(foot, [1.0, 1.0, 0.0])

    def test_residual_is_orthogonal_to_line(self):
        p = np.array([1.7, -2.3, 0.0])
        s, e = np.array([-4.0, 1.0, 0.0]), np.array([3.0, 2.5, 0.0])
        foot = perpendicular_foot(p, s, e)
        assert abs(np.dot(p - foot, e - s)) < 1e-9  # true right angle

    def test_degenerate_line_returns_start(self):
        foot = perpendicular_foot([1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0])
        assert np.allclose(foot, [2.0, 2.0, 0.0])

    def test_accepts_mobjects(self):
        dot = Dot(point=[0.0, 2.0, 0.0])
        line = Line([-3.0, 0.0, 0.0], [3.0, 0.0, 0.0])
        foot = perpendicular_foot(dot, line.get_start(), line.get_end())
        assert np.allclose(foot, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# connect / on_axes / drop_perpendicular — construction
# ---------------------------------------------------------------------------

class TestConnect:
    def test_arrow_spans_object_centers(self):
        a = Circle(radius=0.5).move_to([-2.0, 0.0, 0.0])
        b = Circle(radius=0.5).move_to([2.0, 1.0, 0.0])
        arrow = connect(a, b, buff=0.0)
        assert np.allclose(arrow.get_start(), a.get_center(), atol=1e-6)
        assert np.allclose(arrow.get_end(), b.get_center(), atol=1e-6)

    def test_accepts_raw_points(self):
        arrow = connect([0.0, 0.0], [1.0, 1.0], buff=0.0)
        assert np.allclose(arrow.get_end(), [1.0, 1.0, 0.0], atol=1e-6)


class TestOnAxes:
    def test_matches_coords_to_point(self):
        axes = Axes(x_range=[-5, 5, 1], y_range=[-3, 3, 1], x_length=5.0, y_length=3.0)
        assert np.allclose(on_axes(axes, 2.0, 1.0), axes.coords_to_point(2.0, 1.0))


class TestDropPerpendicular:
    def test_drop_ends_on_the_line(self):
        dot = Dot(point=[1.0, 2.0, 0.0])
        line = Line([-4.0, 0.0, 0.0], [4.0, 0.0, 0.0])
        group = drop_perpendicular(dot, line)
        drop = group[0]
        assert isinstance(drop, DashedLine)
        foot = drop.get_end()
        assert np.allclose(foot, [1.0, 0.0, 0.0], atol=1e-6)  # ON the line
        # And the drop is orthogonal to the line
        line_dir = line.get_end() - line.get_start()
        drop_dir = drop.get_start() - drop.get_end()
        assert abs(np.dot(line_dir, drop_dir)) < 1e-6

    def test_includes_right_angle_tick(self):
        group = drop_perpendicular(
            Dot(point=[0.0, 2.0, 0.0]), Line([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        )
        assert len(group) == 2 and isinstance(group[1], VGroup)

    def test_point_on_line_degenerates_gracefully(self):
        group = drop_perpendicular(
            Dot(point=[1.0, 0.0, 0.0]), Line([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        )
        assert len(group) >= 1  # no crash, no tick


# ---------------------------------------------------------------------------
# Validator lint: hardcoded connector endpoints
# ---------------------------------------------------------------------------

class TestAnchoringLint:
    def _warnings_for(self, body: str):
        warnings: list = []
        ManimCodeValidatorAgent()._check_geometry_anchoring(body, warnings)
        return warnings

    def test_double_literal_endpoints_flagged(self):
        code = "arrow1 = Arrow(start=[2.5, -1.7, 0], end=[3.0, -1.58, 0], buff=0)\n"
        w = self._warnings_for(code)
        assert w and "Arrow at line 1" in w[0]

    def test_negative_literals_still_flagged(self):
        code = "l = DashedLine([-1.0, -2.0, 0], [0.5, 0.25, 0])\n"
        assert self._warnings_for(code)

    def test_anchored_endpoints_not_flagged(self):
        code = (
            "arrow = Arrow(start=dot.get_center(), end=axes.c2p(2, 1), buff=0)\n"
            "link = connect(a, b)\n"
            "drop = drop_perpendicular(dot, line)\n"
        )
        assert self._warnings_for(code) == []

    def test_one_literal_one_anchor_not_flagged(self):
        code = "arrow = Arrow(start=[0, 0, 0], end=dot.get_center())\n"
        assert self._warnings_for(code) == []

    def test_imports_include_geometry_helpers(self):
        from src.agents.animation_generator import AnimationCodeGeneratorAgent
        agent = AnimationCodeGeneratorAgent.__new__(AnimationCodeGeneratorAgent)
        joined = "\n".join(agent._get_imports())
        for name in ("connect", "drop_perpendicular", "on_axes", "perpendicular_foot"):
            assert name in joined

    def test_prompt_carries_anchoring_rules(self, monkeypatch):
        from tests.test_animation_generator import (
            _capture_animation_prompt,
            build_test_context,
            build_test_scene_spec,
        )
        from src.agents.animation_generator import AnimationCodeGeneratorAgent
        captured = _capture_animation_prompt(
            AnimationCodeGeneratorAgent, monkeypatch,
            build_test_scene_spec(), build_test_context(),
        )
        assert "GEOMETRIC ANCHORING" in captured["system"]
        assert "drop_perpendicular" in captured["system"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
