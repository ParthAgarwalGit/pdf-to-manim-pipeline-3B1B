"""Security/correctness fixes from the branch review.

- scene_id is LLM-influenced and embedded in file paths (scene_{id}.py):
  the schema must reject path metacharacters (Windows resolves '..'
  lexically, so 'scene_x/../../evil' escapes the run directory).
- The JSON sanitizer normalizes unsafe ids for free instead of paying for
  a regeneration round-trip.
- Requested render quality ('high') must map to the manim flag instead of
  silently falling through to the medium default.
"""

import pytest
from datetime import datetime

from src.agents.json_sanitizer import JSONSanitizerAgent
from src.agents.video_renderer import VideoRendererAgent
from src.schemas.animation_code import AnimationCodeObject


def _code_obj(scene_id):
    return AnimationCodeObject(
        scene_id=scene_id,
        code="class S(Scene):\n    def construct(self):\n        pass\n",
        imports=["from manim import *"],
        class_name="S",
        narration="n",
        duration_estimate=10.0,
        generation_timestamp=datetime.now(),
    )


class TestSceneIdPathSafety:
    def test_normal_ids_accepted(self):
        assert _code_obj("scene_001").scene_id == "scene_001"
        assert _code_obj("intro-01").scene_id == "intro-01"

    @pytest.mark.parametrize("bad", [
        "x/../../evil", "..", "a/b", "a\b", "a.b", "con:alt", "a b", "a$",
    ])
    def test_path_metacharacters_rejected(self, bad):
        with pytest.raises(Exception):
            _code_obj(bad)

    def test_storyboard_scene_id_also_guarded(self):
        from src.schemas.storyboard import Scene
        with pytest.raises(Exception):
            Scene.validate_scene_id("x/../../evil")
        assert Scene.validate_scene_id("scene_001") == "scene_001"


class TestSanitizerSceneIdNormalization:
    def _sanitize_ids(self, ids, deps=None):
        agent = JSONSanitizerAgent()
        storyboard = {
            "scenes": [
                {"scene_id": sid, "dependencies": list(deps or [])}
                for sid in ids
            ]
        }
        fixed, corrections = agent._apply_corrections(storyboard)
        return fixed, corrections

    def test_unsafe_chars_replaced(self):
        fixed, corrections = self._sanitize_ids(["x/../../evil"])
        assert fixed["scenes"][0]["scene_id"] == "x_______evil"
        assert any(c.violation_type == "unsafe_scene_id" for c in corrections)

    def test_references_follow_the_rename(self):
        fixed, _ = self._sanitize_ids(
            ["a.b", "scene_002"], deps=["a.b"]
        )
        assert fixed["scenes"][0]["scene_id"] == "a_b"
        assert fixed["scenes"][1]["dependencies"] == ["a_b"]

    def test_safe_ids_untouched(self):
        fixed, corrections = self._sanitize_ids(["scene_001"])
        assert fixed["scenes"][0]["scene_id"] == "scene_001"
        assert not any(c.violation_type == "unsafe_scene_id" for c in corrections)

    def test_normalized_collision_still_deduped(self):
        fixed, _ = self._sanitize_ids(["a.b", "a_b"])
        ids = [s["scene_id"] for s in fixed["scenes"]]
        assert len(set(ids)) == 2


class TestQualityMapping:
    def test_word_forms_map_to_flags(self):
        assert VideoRendererAgent.quality_flag_for("high") == "-qh"
        assert VideoRendererAgent.quality_flag_for("low") == "-ql"
        assert VideoRendererAgent.quality_flag_for("medium") == "-qm"

    def test_short_keys_still_work(self):
        assert VideoRendererAgent.quality_flag_for("h") == "-qh"
        assert VideoRendererAgent.quality_flag_for("m") == "-qm"

    def test_unknown_falls_back_to_medium(self):
        assert VideoRendererAgent.quality_flag_for("potato") == "-qm"

    def test_case_insensitive(self):
        assert VideoRendererAgent.quality_flag_for("High") == "-qh"
