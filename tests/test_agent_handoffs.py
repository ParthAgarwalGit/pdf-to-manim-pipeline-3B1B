"""Inter-agent handoff contracts + Phase-0 observability regressions.

A real run showed the agent-to-agent joins hold today but nothing guarantees
they keep holding: a sanitizer that drops a newer field would silently re-break
continuity. These tests pin the contracts, plus the warning fixes a real run's
console spam exposed (false-positive coverage warning, leaked beat refs,
progress-bar-truncated render errors, vanished warnings).
"""

import logging

import pytest

from src.agents.json_sanitizer import JSONSanitizerAgent, JSONSanitizerInput
from src.agents.script_architect import ScriptArchitectAgent
from src.agents.video_renderer import VideoRendererAgent
from src.schemas.source_segment import SourceSegment
from src.schemas.storyboard import (
    BBox,
    Configuration,
    ConceptHierarchy,
    MathematicalObject,
    ObjectStyle,
    PDFMetadataStoryboard,
    Scene,
    SceneState,
    StoryboardJSON,
    TimeInterval,
    VisualIntent,
)
from src.schemas.video_script import Beat


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _full_scene(scene_id="scene_001", index=0, concept="svd", segment="seg_001"):
    obj = MathematicalObject(
        object_id="circle_main",
        object_role="concept_visual",
        object_type="geometric_shape",
        content="",
        primitive="Circle",
        args={"radius": 2.0},
        bbox=BBox(x=1.0, y=-0.5, w=3.0, h=3.0),
        interval=TimeInterval(t_in=0.0, t_out=12.0),
        style=ObjectStyle(color="BLUE"),
    )
    return Scene(
        scene_id=scene_id,
        scene_index=index,
        concept_id=concept,
        narration="A circle representing the concept.",
        visual_intent=VisualIntent(
            mathematical_objects=[obj],
            transformations=[],
            beats=[],
            scene_state_in=SceneState(),
            scene_state_out=SceneState(active_object_ids=["circle_main"]),
            emphasis_points=[],
        ),
        duration_estimate=12.0,
        dependencies=[],
        source_beats=[Beat(
            beat_id=f"beat_{index + 1:03d}",
            narration="A circle representing the concept.",
            visual_hint="a circle",
            introduces_concepts=[concept],
            source_segment_id=segment,
        )],
        source_segment_ids=[segment],
    )


def _full_storyboard():
    return StoryboardJSON(
        pdf_metadata=PDFMetadataStoryboard(filename="t.pdf", page_count=3),
        configuration=Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        ),
        concept_hierarchy=[ConceptHierarchy(
            concept_id="svd", concept_name="Svd", dependencies=[],
        )],
        scenes=[_full_scene()],
    )


# ---------------------------------------------------------------------------
# Sanitizer round-trip: every newer field must survive the dump/validate hop
# ---------------------------------------------------------------------------

class TestSanitizerRoundTrip:
    def test_all_contract_fields_survive(self):
        sb = _full_storyboard()
        result = JSONSanitizerAgent().execute(
            JSONSanitizerInput(storyboard_dict=sb.model_dump())
        )
        out = result.storyboard
        scene = out.scenes[0]

        assert scene.source_segment_ids == ["seg_001"]
        assert [b.beat_id for b in scene.source_beats] == ["beat_001"]
        assert scene.source_beats[0].source_segment_id == "seg_001"

        obj = scene.visual_intent.mathematical_objects[0]
        assert obj.primitive == "Circle"
        assert obj.args == {"radius": 2.0}
        assert obj.bbox is not None and obj.bbox.x == 1.0
        assert obj.interval is not None and obj.interval.t_out == 12.0

        # The continuity-critical field — losing this silently kills carry-over.
        assert scene.visual_intent.scene_state_out.active_object_ids == ["circle_main"]


# ---------------------------------------------------------------------------
# Scene -> SceneSpec conversion keeps the full visual intent
# ---------------------------------------------------------------------------

class TestSceneToSpec:
    def test_visual_intent_preserved(self):
        from src.agents.scene_controller import SceneControllerAgent
        scene = _full_scene()
        spec = SceneControllerAgent()._scene_to_spec(scene)
        assert spec.scene_id == scene.scene_id
        objs = spec.visual_intent.mathematical_objects
        assert len(objs) == 1 and objs[0].primitive == "Circle"
        assert spec.visual_intent.scene_state_out.active_object_ids == ["circle_main"]


# ---------------------------------------------------------------------------
# Coverage warnings: real invariants only (Phase 0.2)
# ---------------------------------------------------------------------------

class TestCoverageInvariants:
    def _agent(self):
        return ScriptArchitectAgent.__new__(ScriptArchitectAgent)

    def _segments(self, n):
        return [SourceSegment(
            segment_id=f"seg_{i:03d}", order=i - 1, page_start=1, page_end=1, text="t",
        ) for i in range(1, n + 1)]

    def test_multi_scene_segment_is_not_a_warning(self, caplog):
        # One segment split across two scenes is BY DESIGN — the old check
        # spammed a false-positive warning per segment for this.
        scenes = [
            _full_scene("scene_001", 0, segment="seg_001"),
            _full_scene("scene_002", 1, segment="seg_001"),
        ]
        scenes[1].source_beats[0].beat_id = "beat_002"  # distinct beats
        with caplog.at_level(logging.WARNING, logger="src.agents.script_architect"):
            self._agent()._log_segment_coverage(scenes, self._segments(1))
        assert not caplog.records

    def test_missing_segment_warns(self, caplog):
        scenes = [_full_scene(segment="seg_001")]
        with caplog.at_level(logging.WARNING, logger="src.agents.script_architect"):
            self._agent()._log_segment_coverage(scenes, self._segments(2))
        assert any("missing source segments" in r.getMessage() for r in caplog.records)

    def test_duplicated_beat_warns(self, caplog):
        scenes = [
            _full_scene("scene_001", 0, segment="seg_001"),
            _full_scene("scene_002", 1, segment="seg_001"),
        ]
        scenes[1].source_beats[0].beat_id = "beat_001"  # same beat in two scenes
        with caplog.at_level(logging.WARNING, logger="src.agents.script_architect"):
            self._agent()._log_segment_coverage(scenes, self._segments(1))
        assert any("duplicated across scenes" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Beat-reference resolution (Phase 0.3): no leaked validator warnings
# ---------------------------------------------------------------------------

class TestBeatRefResolution:
    def _agent_with_beats(self, monkeypatch, beats_per_segment):
        import json as _json

        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False
        responses = iter(beats_per_segment)

        class _Msg:
            def __init__(self, content): self.content = content

        class _Choice:
            def __init__(self, content): self.message = _Msg(content)

        class _Resp:
            def __init__(self, content): self.choices = [_Choice(content)]

        class _Completions:
            def create(self, **kw):
                return _Resp(_json.dumps({"title": "T", "beats": next(responses)}))

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        agent.client = _Client()
        return agent

    def _segments(self, n):
        return [SourceSegment(
            segment_id=f"seg_{i:03d}", order=i - 1, page_start=1, page_end=1, text="t",
        ) for i in range(1, n + 1)]

    def test_refs_resolve_without_validator_warnings(self, monkeypatch, caplog):
        config = Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        )
        seg1 = [
            {"beat_id": "beat_local_001", "narration": "w " * 30, "visual_hint": "h"},
            # backward ref within segment + forward ref (must be dropped quietly)
            {"beat_id": "beat_local_002", "narration": "w " * 30, "visual_hint": "h",
             "references_prior_beats": ["beat_local_001", "beat_local_003"]},
            {"beat_id": "beat_local_003", "narration": "w " * 30, "visual_hint": "h"},
        ]
        seg2 = [
            # hallucinated local ref from another segment (must be dropped quietly)
            {"beat_id": "beat_local_001", "narration": "w " * 30, "visual_hint": "h",
             "references_prior_beats": ["beat_local_099"]},
        ]
        agent = self._agent_with_beats(monkeypatch, [seg1, seg2])

        with caplog.at_level(logging.WARNING):
            script = agent._generate_segmented_video_script(
                self._segments(2), config, [],
            )

        # No 'unknown or forward beat' warnings from the schema validator.
        assert not any(
            "references unknown or forward beat" in r.getMessage()
            for r in caplog.records
        )
        by_id = {b.beat_id: b for b in script.beats}
        assert by_id["beat_002"].references_prior_beats == ["beat_001"]
        assert by_id["beat_004"].references_prior_beats == []
        # Refs survived the VideoScript validator (nothing was late-dropped).
        assert script.beats[1].references_prior_beats == ["beat_001"]


# ---------------------------------------------------------------------------
# Render-error extraction (Phase 0.6)
# ---------------------------------------------------------------------------

class TestRenderErrorExtraction:
    def test_progress_bars_stripped_error_kept(self):
        stderr = (
            "Animation 0: Write(Tex('title')):   0%|          | 0/45 [00:00<?, ?it/s]\r"
            "Animation 0: Write(Tex('title')):  76%|#######5  | 34/45 [00:00<00:00, 155.58it/s]\n"
            "Animation 2: FadeIn(VGroup of 14 submobjects):  61%|######1   | 22/36 [00:00<00:00, 217.80it/s]\n"
            "Traceback (most recent call last):\n"
            '  File "scene.py", line 42, in construct\n'
            "ValueError: latex error converting to dvi\n"
        )
        out = VideoRendererAgent._extract_render_error(stderr)
        assert "ValueError: latex error converting to dvi" in out
        assert "it/s]" not in out

    def test_empty_stderr(self):
        assert VideoRendererAgent._extract_render_error("") == "No stderr."

    def test_pure_progress_falls_back_to_tail(self):
        stderr = "Animation 0: Write(x):  10%|#         | 1/10 [00:00<?, ?it/s]"
        out = VideoRendererAgent._extract_render_error(stderr)
        assert out  # never empty


# ---------------------------------------------------------------------------
# Warning capture handler (Phase 0.1)
# ---------------------------------------------------------------------------

class TestWarningCapture:
    def test_warnings_counted_and_persisted(self, tmp_path):
        from src.orchestrator.pipeline import _WarningCaptureHandler
        path = tmp_path / "warnings.log"
        handler = _WarningCaptureHandler(str(path))
        src_logger = logging.getLogger("src")
        src_logger.addHandler(handler)
        try:
            logging.getLogger("src.agents.script_architect").warning("boom %d", 1)
            logging.getLogger("src.agents.video_renderer").info("not captured")
            logging.getLogger("src.utils.frame_analysis").warning("boom 2")
        finally:
            src_logger.removeHandler(handler)
            handler.close()
        assert handler.count == 2
        text = path.read_text(encoding="utf-8")
        assert "boom 1" in text and "boom 2" in text
        assert "not captured" not in text

    def test_no_file_created_when_no_warnings(self, tmp_path):
        from src.orchestrator.pipeline import _WarningCaptureHandler
        path = tmp_path / "warnings.log"
        handler = _WarningCaptureHandler(str(path))
        handler.close()
        assert not path.exists()  # lazy creation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
