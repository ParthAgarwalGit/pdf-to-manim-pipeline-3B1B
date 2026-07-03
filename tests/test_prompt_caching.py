"""Phase 5 — prompt caching structure + review de-noising.

Cost control for students: the code-gen static contract rides Anthropic's
prompt cache (block 1 with cache_control), and the visual-intent prompt puts
its static contract FIRST so OpenAI prefix caching fires across the ~45
planner calls of a run. Plus: vignette-friendly timing minimums and the
aggregated breathing-room finding (a real run drowned in 180 per-gap ones).
"""

import json as _json

import pytest

from src.agents.animation_generator import (
    AnimationCodeGeneratorAgent,
    _codegen_static_system_prompt,
)
from src.agents.script_architect import (
    ScriptArchitectAgent,
    _visual_intent_static_prompt,
)
from src.schemas.storyboard import Configuration
from src.schemas.video_script import Beat, VideoScript
from src.utils.manim_engine_contract import TIMING_CONTRACT
from tests.test_animation_generator import (
    _capture_animation_prompt,
    build_test_context,
    build_test_scene_spec,
)


# ---------------------------------------------------------------------------
# Codegen: cached static block + dynamic tail
# ---------------------------------------------------------------------------

class TestCodegenSystemSplit:
    def _captured(self, monkeypatch, scene_id="scene_001", narration="Hello."):
        return _capture_animation_prompt(
            AnimationCodeGeneratorAgent, monkeypatch,
            build_test_scene_spec(scene_id=scene_id, narration=narration),
            build_test_context(),
        )

    def test_system_is_two_blocks_with_cache_control(self, monkeypatch):
        captured = self._captured(monkeypatch)
        raw = captured["system_raw"]
        assert isinstance(raw, list) and len(raw) == 2
        assert raw[0].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in raw[1]

    def test_static_block_has_no_per_scene_data(self, monkeypatch):
        captured = self._captured(monkeypatch, scene_id="uniquescene")
        raw = captured["system_raw"]
        assert "CLASS NAME" not in raw[0]["text"]
        assert "Uniquescene" not in raw[0]["text"]
        assert "CLASS NAME" in raw[1]["text"]

    def test_static_block_identical_across_scenes(self, monkeypatch):
        a = self._captured(monkeypatch, "scene_a", "First narration.")
        first = a["system_raw"][0]["text"]
        b = self._captured(monkeypatch, "scene_b", "Totally different words here.")
        second = b["system_raw"][0]["text"]
        assert first == second  # byte-identical => cacheable

    def test_static_block_carries_the_contracts(self):
        s = _codegen_static_system_prompt()
        for section in ("SCREEN LAYOUT", "VISUAL PLAN (HONOR IT)",
                        "GEOMETRIC ANCHORING", "SCENE ENDING (CONTINUITY)",
                        "NARRATOR MASCOT", "OUTPUT RULES"):
            assert section in s
        # Brace un-escaping from the old f-string must have produced literals.
        assert '{"direction": [2,1,0]}' in s


# ---------------------------------------------------------------------------
# Visual intent: static-first prefix for OpenAI prefix caching
# ---------------------------------------------------------------------------

class TestVisualIntentPrefix:
    def _prompt_for(self, monkeypatch, narration):
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
        script = VideoScript(beats=[Beat(beat_id="beat_001", narration=narration, visual_hint="h")])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        )
        agent._generate_visual_intent_for_chunk(scene, script, config, [])
        return captured["prompt"]

    def test_prompts_share_static_prefix(self, monkeypatch):
        static = _visual_intent_static_prompt()
        assert len(static) > 2000  # worth caching
        p1 = self._prompt_for(monkeypatch, "the derivative is a slope")
        p2 = self._prompt_for(monkeypatch, "vectors add tip to tail in a plane")
        assert p1.startswith(static)
        assert p2.startswith(static)

    def test_narration_comes_after_the_static_prefix(self, monkeypatch):
        static = _visual_intent_static_prompt()
        p = self._prompt_for(monkeypatch, "a very distinctive narration string")
        assert p.index("a very distinctive narration string") > len(static) - 1


# ---------------------------------------------------------------------------
# Timing contract + de-noised breathing-room finding
# ---------------------------------------------------------------------------

class TestTimingDeNoise:
    def test_creation_minimums_relaxed_for_vignettes(self):
        assert TIMING_CONTRACT["Write"][0] == 0.6
        assert TIMING_CONTRACT["Create"][0] == 0.6

    def _findings_for(self, code):
        from datetime import datetime
        from src.agents.video_quality_reviewer import VideoQualityReviewerAgent
        from src.schemas.animation_code import AnimationCodeObject
        from tests.test_agent_handoffs import _full_storyboard

        code_obj = AnimationCodeObject(
            scene_id="scene_001", code=code, imports=["from manim import *"],
            class_name="S", narration="n", duration_estimate=10.0,
            generation_timestamp=datetime.now(),
        )
        agent = VideoQualityReviewerAgent()
        return agent._review_anti_patterns(
            _full_storyboard(), {"scene_001": code_obj},
        )

    def test_systematic_missing_waits_is_one_low_finding(self):
        code = "\n".join(
            f"        self.play(Write(t{i}), run_time=1.0)" for i in range(5)
        )
        findings = [f for f in self._findings_for(code)
                    if "breathing room" in f.issue.lower()]
        assert len(findings) == 1
        assert findings[0].severity == "low"

    def test_occasional_back_to_back_plays_not_flagged(self):
        code = (
            "        self.play(Write(a), run_time=1.0)\n"
            "        self.play(Write(b), run_time=1.0)\n"
            "        self.wait(1.0)\n"
            "        self.play(Write(c), run_time=1.0)\n"
        )
        findings = [f for f in self._findings_for(code)
                    if "breathing room" in f.issue.lower()]
        assert findings == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
