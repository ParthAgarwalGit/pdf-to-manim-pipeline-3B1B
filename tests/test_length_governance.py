"""P2.6: --target-minutes turns finished-video length into a chosen input.

The scene budget derived from a requested length must (a) tighten the segment
cap so the floor never overshoots, and (b) tighten the chunker ceiling so the
scene count is bounded by the request, never exceeding the static explosion
guards (MAX_SEGMENTS / MAX_SCENES)."""

import pytest

from src.agents.document_segmenter import (
    MAX_SEGMENTS,
    SECONDS_PER_SCENE_ESTIMATE,
    scenes_for_minutes,
    target_words_for,
)
from src.agents.script_architect import ScriptArchitectAgent
from src.schemas.storyboard import Configuration


class TestScenesForMinutes:
    def test_five_minutes_maps_to_expected_budget(self):
        # 5 min * 60 / 25 s per scene = 12 scenes.
        assert scenes_for_minutes(5) == round(5 * 60 / SECONDS_PER_SCENE_ESTIMATE)
        assert scenes_for_minutes(5) == 12

    def test_never_below_one(self):
        assert scenes_for_minutes(0.01) == 1
        assert scenes_for_minutes(0) == 1

    def test_longer_request_more_scenes(self):
        assert scenes_for_minutes(10) > scenes_for_minutes(3)


class TestTargetWordsCap:
    def test_tighter_cap_yields_bigger_segments(self):
        # Same document, tighter segment cap -> more words per segment.
        loose = target_words_for("fine", 6000, max_segments=MAX_SEGMENTS)
        tight = target_words_for("fine", 6000, max_segments=12)
        assert tight > loose

    def test_short_doc_unaffected_by_cap(self):
        # 300 words at balanced base (150) sits under any cap -> base wins.
        assert target_words_for("balanced", 300, max_segments=12) == 150

    def test_default_cap_is_max_segments(self):
        assert target_words_for("fine", 6000) == target_words_for(
            "fine", 6000, max_segments=MAX_SEGMENTS
        )


class TestEffectiveMaxScenes:
    def _agent(self):
        return ScriptArchitectAgent.__new__(ScriptArchitectAgent)

    def _config(self, **kw):
        base = dict(
            verbosity="medium", depth="intermediate", audience="undergraduate"
        )
        base.update(kw)
        return Configuration(**base)

    def test_none_target_uses_static_ceiling(self):
        agent = self._agent()
        assert agent._effective_max_scenes(self._config()) == agent.MAX_SCENES

    def test_target_minutes_tightens_ceiling(self):
        agent = self._agent()
        eff = agent._effective_max_scenes(self._config(target_minutes=5))
        assert eff == 12
        assert eff < agent.MAX_SCENES

    def test_target_never_exceeds_static_guard(self):
        agent = self._agent()
        # An absurdly long request still cannot exceed MAX_SCENES.
        eff = agent._effective_max_scenes(self._config(target_minutes=999))
        assert eff == agent.MAX_SCENES

    def test_ceiling_at_least_one(self):
        agent = self._agent()
        assert agent._effective_max_scenes(self._config(target_minutes=0.1)) >= 1


class TestConfigurationSchema:
    def test_target_minutes_defaults_none(self):
        c = Configuration(verbosity="low", depth="introductory", audience="high-school")
        assert c.target_minutes is None

    def test_target_minutes_must_be_positive(self):
        with pytest.raises(Exception):
            Configuration(
                verbosity="low", depth="introductory", audience="high-school",
                target_minutes=0,
            )
