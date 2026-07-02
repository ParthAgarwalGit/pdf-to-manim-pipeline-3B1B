"""Tests for the concept → visual-archetype library and its wiring.

The registry is the project's distilled "how to draw this concept" knowledge;
these tests pin its integrity (never recommends anything the schema rejects),
the deterministic classifier, and both consumption points in the script
architect (planner prompt + degraded fallback intent).
"""

import pytest

from src.agents.script_architect import ScriptArchitectAgent
from src.schemas.storyboard import Configuration, MathematicalObject, ObjectStyle
from src.utils.manim_primitives import MOBJECT_CATALOG, allowed_args_for
from src.utils.visual_archetypes import (
    ARCHETYPE_REGISTRY,
    archetype_prompt_block,
    classify_concept,
)


# ---------------------------------------------------------------------------
# Registry integrity — the library must never recommend invalid vocabulary
# ---------------------------------------------------------------------------

class TestRegistryIntegrity:
    def test_registry_is_non_trivial(self):
        assert len(ARCHETYPE_REGISTRY) >= 10

    def test_every_suggested_primitive_exists_in_catalog(self):
        for archetype in ARCHETYPE_REGISTRY.values():
            for name in archetype.primitives:
                assert name in MOBJECT_CATALOG, (
                    f"{archetype.name} suggests unknown primitive {name!r}"
                )

    def test_every_fallback_primitive_validates_against_schema(self):
        """The fallback pair must survive MathematicalObject validation —
        this is what guards against catalog-arg drift breaking keyless runs."""
        for archetype in ARCHETYPE_REGISTRY.values():
            name, args, object_type = archetype.fallback_primitive
            assert name in MOBJECT_CATALOG
            assert set(args) <= allowed_args_for(name), (
                f"{archetype.name} fallback args {sorted(args)} not all legal "
                f"for {name}"
            )
            obj = MathematicalObject(
                object_id="t", object_type=object_type,
                content="x", primitive=name, args=dict(args),
                style=ObjectStyle(),
            )
            assert obj.primitive == name

    def test_every_fallback_is_a_non_text_primitive(self):
        """The whole point of the fallback enrichment is escaping text walls."""
        for archetype in ARCHETYPE_REGISTRY.values():
            name, _, _ = archetype.fallback_primitive
            assert name not in {"Text", "Tex", "MathTex"}, (
                f"{archetype.name} fallback must not be text-like"
            )

    def test_every_archetype_has_keywords_and_motion(self):
        for archetype in ARCHETYPE_REGISTRY.values():
            assert archetype.keywords
            assert len(archetype.motion) > 40  # real design prose, not a stub


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TestClassifier:
    @pytest.mark.parametrize("text,expected", [
        ("the derivative measures the slope of the tangent line", "rate_of_change"),
        ("vectors add tip to tail and scale by magnitude", "vector_space"),
        ("the probability of drawing a red ball from the urn", "probability_sampling"),
        ("the matrix rotation transforms every point of the plane", "transformation_morph"),
        ("the sequence converges to a limit as n tends to infinity", "limit_convergence"),
        ("the algorithm proceeds step by step through the pipeline", "process_flow"),
        ("a triangle inscribed in a circle with a right angle", "geometric_relation"),
        ("the partial sums of the series accumulate toward the total", "accumulation_series"),
        ("the union and intersection of two sets", "set_grouping"),
    ])
    def test_classifies_concept_families(self, text, expected):
        result = classify_concept(text)
        assert result is not None
        assert result.name == expected

    def test_unrelated_text_returns_none(self):
        assert classify_concept("the committee approved the budget yesterday") is None

    def test_empty_text_returns_none(self):
        assert classify_concept("") is None

    def test_classification_is_deterministic(self):
        text = "the graph of the function shows the slope of the tangent"
        assert classify_concept(text).name == classify_concept(text).name

    def test_word_boundary_prevents_substring_hits(self):
        # "assumption" contains "sum" but must not trigger accumulation_series;
        # nothing else in this sentence matches any archetype.
        assert classify_concept("that assumption was reasonable") is None


# ---------------------------------------------------------------------------
# Prompt block
# ---------------------------------------------------------------------------

class TestPromptBlock:
    def test_block_carries_name_primitives_and_motion(self):
        archetype = ARCHETYPE_REGISTRY["vector_space"]
        block = archetype_prompt_block(archetype)
        assert "RECOMMENDED VISUAL ARCHETYPE: vector_space" in block
        assert "Vector" in block
        assert archetype.motion in block
        assert "adapt" in block  # recommendation, not straitjacket


# ---------------------------------------------------------------------------
# Wiring: planner prompt + degraded fallback
# ---------------------------------------------------------------------------

class TestArchitectWiring:
    def _capture_planner_prompt(self, monkeypatch, narration, visual_hint="h"):
        from src.schemas.video_script import Beat, VideoScript

        captured: dict = {}
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {
                "content": '{"objects": [{"object_type": "geometric_shape", '
                           '"content": "", "primitive": "Circle", '
                           '"args": {"radius": 1.0}}], "transformations": [], '
                           '"beats": [], "scene_state_in": {}, '
                           '"scene_state_out": {}, "emphasis_points": []}',
            })})]

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
            beat_id="beat_001", narration=narration, visual_hint=visual_hint,
        )])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        )
        agent._generate_visual_intent_for_chunk(scene, script, config, [])
        return captured["prompt"]

    def test_planner_prompt_includes_matched_archetype(self, monkeypatch):
        prompt = self._capture_planner_prompt(
            monkeypatch,
            "The derivative is the slope of the tangent line at a point.",
        )
        assert "RECOMMENDED VISUAL ARCHETYPE: rate_of_change" in prompt

    def test_planner_prompt_omits_block_when_no_match(self, monkeypatch):
        prompt = self._capture_planner_prompt(
            monkeypatch,
            "The committee approved the budget yesterday afternoon.",
        )
        assert "RECOMMENDED VISUAL ARCHETYPE" not in prompt

    def test_fallback_intent_gains_non_text_visual_on_match(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        intent = agent._generate_fallback_visual_intent(
            "Vectors add tip to tail in the plane."
        )
        ids = [o.object_id for o in intent.mathematical_objects]
        assert "archetype_visual" in ids
        assert ScriptArchitectAgent._is_text_only_intent(intent) is False
        visual = next(
            o for o in intent.mathematical_objects
            if o.object_id == "archetype_visual"
        )
        assert visual.primitive == "Vector"
        assert visual.bbox is not None and visual.interval is not None

    def test_fallback_intent_still_valid_without_match(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        intent = agent._generate_fallback_visual_intent(
            "The committee approved the budget."
        )
        assert intent.mathematical_objects  # title always present
        ids = [o.object_id for o in intent.mathematical_objects]
        assert "archetype_visual" not in ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
