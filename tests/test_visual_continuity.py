"""Phase 1 — scene-to-scene visual continuity.

A real run carried state on 0/90 scenes (every scene cut to black). These
tests pin the whole chain: derived scene_state_out → carried objects → next
scene's prompt → codegen carry_out contract → scene-count governance.
"""

import json as _json

import pytest

from src.agents.script_architect import ScriptArchitectAgent
from src.schemas.storyboard import (
    BBox,
    Configuration,
    MathematicalObject,
    SceneState,
    TimeInterval,
    VisualIntent,
)
from src.schemas.video_script import Beat, VideoScript


def _obj(oid, primitive, t_in, t_out, x=0.0, y=0.0, args=None):
    type_map = {"Text": "text", "MathTex": "equation", "Circle": "geometric_shape",
                "Vector": "geometric_shape", "Axes": "axes"}
    return MathematicalObject(
        object_id=oid,
        object_type=type_map.get(primitive, "geometric_shape"),
        content="x" if primitive in ("Text", "MathTex") else "",
        primitive=primitive,
        args=args if args is not None else (
            {"text": "x"} if primitive == "Text"
            else {"tex": "x"} if primitive == "MathTex" else {}
        ),
        bbox=BBox(x=x, y=y, w=2.0, h=2.0),
        interval=TimeInterval(t_in=t_in, t_out=t_out),
    )


def _intent(objs, active_out=None):
    return VisualIntent(
        mathematical_objects=objs,
        transformations=[],
        beats=[],
        scene_state_in=SceneState(),
        scene_state_out=SceneState(active_object_ids=active_out or []),
        emphasis_points=[],
    )


# ---------------------------------------------------------------------------
# 1.1 — deriving scene_state_out from intervals
# ---------------------------------------------------------------------------

class TestDeriveCarryOut:
    def test_objects_visible_at_end_are_carried(self):
        objs = [
            _obj("axes_main", "Axes", 0.0, 20.0),
            _obj("label_a", "Text", 0.0, 8.0),      # gone before the end
            _obj("eq_main", "MathTex", 5.0, 20.0),
        ]
        ids = ScriptArchitectAgent._derive_carry_out_ids(objs)
        assert "axes_main" in ids and "eq_main" in ids
        assert "label_a" not in ids

    def test_non_text_visuals_ranked_before_text(self):
        objs = [
            _obj("caption", "Text", 0.0, 10.0),
            _obj("circle", "Circle", 0.0, 10.0),
        ]
        ids = ScriptArchitectAgent._derive_carry_out_ids(objs)
        assert ids[0] == "circle"

    def test_cap_applied(self):
        objs = [_obj(f"o{i}", "Circle", 0.0, 10.0, x=float(i)) for i in range(6)]
        ids = ScriptArchitectAgent._derive_carry_out_ids(objs)
        assert len(ids) == ScriptArchitectAgent.MAX_CARRY_OVER_OBJECTS

    def test_no_intervals_returns_empty(self):
        obj = MathematicalObject(object_type="text", content="hi")
        assert ScriptArchitectAgent._derive_carry_out_ids([obj]) == []

    def test_intent_parser_populates_empty_state_out(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        data = {"objects": [{
            "object_id": "c0", "primitive": "Circle", "args": {"radius": 2.0},
            "bbox": {"x": 0, "y": 0, "w": 2, "h": 2},
            "interval": {"t_in": 0, "t_out": 9},
        }]}
        intent = agent._visual_intent_from_json(data, "narration")
        assert intent.scene_state_out.active_object_ids == ["c0"]

    def test_llm_supplied_state_out_is_respected(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        data = {
            "objects": [{
                "object_id": "c0", "primitive": "Circle", "args": {"radius": 2.0},
                "bbox": {"x": 0, "y": 0, "w": 2, "h": 2},
                "interval": {"t_in": 0, "t_out": 9},
            }],
            "scene_state_out": {"active_object_ids": ["c0"]},
        }
        intent = agent._visual_intent_from_json(data, "n")
        assert intent.scene_state_out.active_object_ids == ["c0"]


# ---------------------------------------------------------------------------
# 1.2 — carried objects reach the next scene's planner prompt
# ---------------------------------------------------------------------------

class TestCarryThreading:
    def test_carried_objects_from_intent_resolves_records(self):
        objs = [_obj("axes_main", "Axes", 0, 10), _obj("label", "Text", 0, 10)]
        intent = _intent(objs, active_out=["axes_main"])
        carried = ScriptArchitectAgent._carried_objects_from_intent(intent)
        assert [o.object_id for o in carried] == ["axes_main"]

    def test_prompt_contains_carry_block(self, monkeypatch):
        captured = {}
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {
                "content": _json.dumps({
                    "objects": [{"object_type": "geometric_shape", "content": "",
                                 "primitive": "Circle", "args": {"radius": 1.0}}],
                    "transformations": [], "beats": [],
                    "scene_state_in": {}, "scene_state_out": {},
                    "emphasis_points": [],
                })})})]

        class _Completions:
            def create(self, *, messages, **_kw):
                captured["prompt"] = messages[0]["content"]
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        agent.client = _Client()
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        script = VideoScript(beats=[Beat(
            beat_id="beat_001", narration="continue the idea", visual_hint="h",
        )])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        )
        carried = [_obj("axes_main", "Axes", 0, 10, x=1.5, y=-0.5)]
        agent._generate_visual_intent_for_chunk(scene, script, config, [], carried)

        prompt = captured["prompt"]
        assert "PREVIOUS SCENE ENDS WITH" in prompt
        assert "axes_main" in prompt
        assert "VISUAL CONTINUITY" in prompt

    def test_no_carry_block_for_first_scene(self, monkeypatch):
        captured = {}
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {
                "content": _json.dumps({
                    "objects": [{"object_type": "geometric_shape", "content": "",
                                 "primitive": "Circle", "args": {"radius": 1.0}}],
                    "transformations": [], "beats": [],
                    "scene_state_in": {}, "scene_state_out": {},
                    "emphasis_points": [],
                })})})]

        class _Completions:
            def create(self, *, messages, **_kw):
                captured["prompt"] = messages[0]["content"]
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        agent.client = _Client()
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        script = VideoScript(beats=[Beat(beat_id="beat_001", narration="intro", visual_hint="h")])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        )
        agent._generate_visual_intent_for_chunk(scene, script, config, [], [])
        assert "PREVIOUS SCENE ENDS WITH" not in captured["prompt"]


# ---------------------------------------------------------------------------
# 1.3 — codegen receives the carry_out contract
# ---------------------------------------------------------------------------

class TestCodegenCarryOut:
    def test_prompt_carries_carry_out_and_rule(self, monkeypatch):
        from tests.test_animation_generator import (
            _capture_animation_prompt,
            build_test_context,
            build_test_scene_spec,
        )
        from src.agents.animation_generator import AnimationCodeGeneratorAgent

        spec = build_test_scene_spec(
            scene_id="scene_002",
            narration="Keep the axes.",
            math_objects=[{
                "object_id": "axes_main", "object_type": "axes",
                "content": "frame", "style": {},
            }],
            transformations=[],
        )
        spec.visual_intent.scene_state_out.active_object_ids = ["axes_main"]

        captured = _capture_animation_prompt(
            AnimationCodeGeneratorAgent, monkeypatch, spec, build_test_context(),
        )
        assert "SCENE ENDING (CONTINUITY)" in captured["system"]
        assert '"carry_out"' in captured["user"]
        assert "axes_main" in captured["user"]


# ---------------------------------------------------------------------------
# 1.4 — scene-count governance
# ---------------------------------------------------------------------------

class TestSceneCap:
    def _beats(self, n, words=45, segments=1):
        per_seg = max(1, n // segments)
        return [Beat(
            beat_id=f"beat_{i:03d}",
            narration=" ".join(["w"] * words),
            visual_hint="h",
            source_segment_id=f"seg_{(i - 1) // per_seg + 1:03d}",
        ) for i in range(1, n + 1)]

    def test_many_beats_capped(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        # 200 beats x 15s in 20 segments would naively be ~100+ scenes.
        script = VideoScript(beats=self._beats(200, segments=20))
        scenes = agent._chunk_script_into_scenes(script)
        assert len(scenes) <= ScriptArchitectAgent.MAX_SCENES

    def test_small_script_unaffected(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        script = VideoScript(beats=self._beats(6))
        scenes = agent._chunk_script_into_scenes(script)
        assert 1 <= len(scenes) <= 6

    def test_segment_boundaries_still_respected(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        script = VideoScript(beats=self._beats(60, segments=30))
        scenes = agent._chunk_script_into_scenes(script)
        for scene in scenes:
            assert len(set(scene.source_segment_ids)) <= 1 or len(scene.source_segment_ids) >= 1
            # a scene never mixes beats from different segments
            segs = {b.source_segment_id for b in scene.source_beats}
            assert len(segs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
