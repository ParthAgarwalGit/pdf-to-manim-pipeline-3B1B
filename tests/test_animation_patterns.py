"""Tests for animation_patterns helpers (Part 3 Task 2)."""

from __future__ import annotations

import pytest


class TestProgressiveWrite:
    """Tests for progressive_write() helper."""

    def test_empty_parts_returns_empty(self):
        from src.utils.animation_patterns import progressive_write
        assert progressive_write([]) == []

    def test_single_part_returns_write_plus_wait(self):
        from src.utils.animation_patterns import progressive_write
        steps = progressive_write(["part_0"], run_time_each=1.0, wait_between=0.5)
        # One write step + one trailing wait
        assert len(steps) == 2
        assert steps[-1] == ("wait", 0.5)

    def test_multiple_parts_have_interspersed_waits(self):
        from src.utils.animation_patterns import progressive_write
        steps = progressive_write(["a", "b", "c"], run_time_each=1.5, wait_between=0.5)
        # 3 writes + 2 between-waits + 1 trailing wait = 6 steps
        assert len(steps) == 6
        wait_steps = [s for s in steps if s[0] == "wait"]
        assert len(wait_steps) == 3  # 2 between + 1 trailing

    def test_run_time_propagated(self):
        from src.utils.animation_patterns import progressive_write
        steps = progressive_write(["x"], run_time_each=2.0)
        anim_str = steps[0][0]
        # In no-manim mode, it's a string stub containing run_time
        if isinstance(anim_str, str):
            assert "2.0" in anim_str


class TestStagedTransform:
    """Tests for staged_transform() helper."""

    def test_empty_targets_returns_empty(self):
        from src.utils.animation_patterns import staged_transform
        assert staged_transform("source", []) == []

    def test_single_target_returns_transform_plus_wait(self):
        from src.utils.animation_patterns import staged_transform
        steps = staged_transform("src", ["tgt"], run_time_each=1.0, wait_between=0.5)
        assert len(steps) == 2
        assert steps[1] == ("wait", 0.5)

    def test_multiple_targets_chain(self):
        from src.utils.animation_patterns import staged_transform
        steps = staged_transform("s", ["t1", "t2", "t3"], run_time_each=1.5)
        # 3 transforms + 3 waits = 6
        assert len(steps) == 6
        transforms = [s for s in steps if s[0] != "wait"]
        assert len(transforms) == 3


class TestDimAndFocus:
    """Tests for dim_and_focus() helper."""

    def test_empty_mobjects_returns_empty(self):
        from src.utils.animation_patterns import dim_and_focus
        assert dim_and_focus([], []) == []

    def test_all_kept_returns_only_trailing_wait(self):
        from src.utils.animation_patterns import dim_and_focus
        # If everything is in keep_ids, nothing gets dimmed
        steps = dim_and_focus(["a", "b"], keep_ids=[0, 1])
        assert steps == []

    def test_dim_correct_count(self):
        from src.utils.animation_patterns import dim_and_focus
        steps = dim_and_focus(["a", "b", "c", "d"], keep_ids=[1])
        # 3 objects dimmed + 1 trailing wait = 4
        assert len(steps) == 4
        dim_steps = [s for s in steps if s[0] != "wait"]
        assert len(dim_steps) == 3


class TestExecuteAnimationSteps:
    """Tests for execute_animation_steps() helper."""

    def test_executes_wait_steps(self):
        from src.utils.animation_patterns import execute_animation_steps

        class MockScene:
            def __init__(self):
                self.plays = []
                self.waits = []
            def play(self, anim):
                self.plays.append(anim)
            def wait(self, t):
                self.waits.append(t)

        scene = MockScene()
        steps = [("wait", 1.0), ("wait", 0.5)]
        execute_animation_steps(scene, steps)
        assert scene.waits == [1.0, 0.5]
        assert scene.plays == []

    def test_skips_string_stubs(self):
        from src.utils.animation_patterns import execute_animation_steps

        class MockScene:
            def __init__(self):
                self.plays = []
                self.waits = []
            def play(self, anim):
                self.plays.append(anim)
            def wait(self, t):
                self.waits.append(t)

        scene = MockScene()
        steps = [("Write(part_0, run_time=1.5)", 0.0)]
        execute_animation_steps(scene, steps)
        assert scene.plays == []
        assert scene.waits == []
