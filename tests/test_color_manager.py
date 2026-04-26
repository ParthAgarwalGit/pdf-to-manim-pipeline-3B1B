"""Tests for ColorManager (Part 3 Task 3)."""

from __future__ import annotations

import pytest

from src.utils.color_manager import ColorManager
from src.utils.manim_engine_contract import COLOR_CYCLE


class TestColorManagerAssign:
    def test_explicit_assign(self):
        cm = ColorManager()
        c = cm.assign("x", "#FF0000")
        assert c == "#FF0000"
        assert cm.get("x") == "#FF0000"

    def test_auto_assign_from_cycle(self):
        cm = ColorManager()
        c = cm.assign("x")
        assert c == COLOR_CYCLE[0]

    def test_auto_assign_increments(self):
        cm = ColorManager()
        c1 = cm.assign("x")
        c2 = cm.assign("y")
        assert c1 != c2
        assert c1 == COLOR_CYCLE[0]
        assert c2 == COLOR_CYCLE[1]

    def test_first_assignment_wins(self):
        cm = ColorManager()
        cm.assign("x", "#AA0000")
        # Second assign with different color is ignored
        c = cm.assign("x", "#BB0000")
        assert c == "#AA0000"

    def test_get_unknown_returns_none(self):
        cm = ColorManager()
        assert cm.get("unknown") is None


class TestColorManagerTexMap:
    def test_build_tex_color_map_all(self):
        cm = ColorManager()
        cm.assign("x", "#FF0000")
        cm.assign("y", "#00FF00")
        m = cm.build_tex_color_map()
        assert m == {"x": "#FF0000", "y": "#00FF00"}

    def test_build_tex_color_map_subset(self):
        cm = ColorManager()
        cm.assign("x", "#FF0000")
        cm.assign("y", "#00FF00")
        cm.assign("z", "#0000FF")
        m = cm.build_tex_color_map(["x", "z"])
        assert m == {"x": "#FF0000", "z": "#0000FF"}

    def test_build_tex_color_map_skips_untracked(self):
        cm = ColorManager()
        cm.assign("x", "#FF0000")
        m = cm.build_tex_color_map(["x", "missing"])
        assert m == {"x": "#FF0000"}


class TestColorManagerPersistence:
    def test_export_and_restore(self):
        cm = ColorManager()
        cm.assign("x", "#FF0000")
        cm.assign("y")
        exported = cm.export_bindings()
        cm2 = ColorManager.from_context_bindings(exported)
        assert cm2.get("x") == "#FF0000"
        assert cm2.get("y") is not None

    def test_initial_bindings_skip_used_colors(self):
        """Auto-assign should not reuse colors from initial bindings."""
        cm = ColorManager(initial_bindings={"x": COLOR_CYCLE[0]})
        c = cm.assign("y")
        assert c != COLOR_CYCLE[0]


class TestColorBindingsInSharedContext:
    def test_shared_context_has_color_bindings(self):
        from src.schemas.scene_processing import SharedContext
        ctx = SharedContext()
        assert ctx.color_bindings == {}
        ctx.color_bindings["x"] = "#FF0000"
        assert ctx.color_bindings["x"] == "#FF0000"
