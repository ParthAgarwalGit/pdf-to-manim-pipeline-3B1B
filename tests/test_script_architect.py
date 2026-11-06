"""Unit tests for Script/Storyboard Architect Agent"""

import json
import pytest
from datetime import datetime

from src.agents.base import AgentExecutionError
from src.agents.script_architect import ScriptArchitectAgent, ScriptArchitectInput
from src.schemas.ocr_output import (
    BoundingBox,
    Diagram,
    MathExpression,
    OCROutput,
    Page,
    PDFMetadata,
    TextBlock,
)
from src.schemas.storyboard import Configuration, StoryboardJSON


class TestScriptArchitectAgent:
    """Test suite for ScriptArchitectAgent"""

    @pytest.fixture
    def agent(self, monkeypatch):
        """Create a ScriptArchitectAgent instance with the deterministic
        mock fallback forced on so tests do not depend on the developer's
        OPENAI_API_KEY (which can rate-limit, quota-out, or simply be absent
        in CI). Tests that exercise the real LLM path use a fake client and
        set the key explicitly via monkeypatch.setenv."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        return ScriptArchitectAgent()
    
    @pytest.fixture
    def sample_ocr_output(self):
        """Create sample OCR output for testing"""
        return OCROutput(
            pdf_metadata=PDFMetadata(
                filename="test_document.pdf",
                page_count=2,
                file_size_bytes=1024000
            ),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[
                        TextBlock(
                            text="Definition 1: A vector space is a set V with operations.",
                            bounding_box=BoundingBox(x=100, y=100, width=400, height=30),
                            reading_order=0,
                            confidence=0.95
                        ),
                        TextBlock(
                            text="Let V be a vector space over the real numbers.",
                            bounding_box=BoundingBox(x=100, y=150, width=400, height=30),
                            reading_order=1,
                            confidence=0.95
                        )
                    ],
                    math_expressions=[
                        MathExpression(
                            latex="\\mathbb{R}^n",
                            bounding_box=BoundingBox(x=150, y=200, width=50, height=20),
                            context="vector space over the real numbers",
                            confidence=0.90
                        )
                    ],
                    diagrams=[]
                ),
                Page(
                    page_number=2,
                    text_blocks=[
                        TextBlock(
                            text="Theorem 1: Every vector space has a basis.",
                            bounding_box=BoundingBox(x=100, y=100, width=400, height=30),
                            reading_order=0,
                            confidence=0.95
                        ),
                        TextBlock(
                            text="Example 1: Consider the vector space R^2.",
                            bounding_box=BoundingBox(x=100, y=150, width=400, height=30),
                            reading_order=1,
                            confidence=0.95
                        )
                    ],
                    math_expressions=[],
                    diagrams=[]
                )
            ]
        )
    
    @pytest.fixture
    def default_config(self):
        """Create default configuration"""
        return Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
    
    @pytest.fixture
    def low_verbosity_config(self):
        """Create low verbosity configuration"""
        return Configuration(
            verbosity="low",
            depth="intermediate",
            audience="undergraduate"
        )
    
    @pytest.fixture
    def high_verbosity_config(self):
        """Create high verbosity configuration"""
        return Configuration(
            verbosity="high",
            depth="advanced",
            audience="graduate"
        )
    
    @pytest.fixture
    def introductory_config(self):
        """Create introductory depth configuration"""
        return Configuration(
            verbosity="medium",
            depth="introductory",
            audience="high-school"
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert agent.retry_count == 0
        # Visual-first vignette pacing (ADR-005): short scenes, no lecture floors.
        assert agent.MIN_SCENE_DURATION == 8.0
        assert agent.MAX_SCENE_DURATION == 45.0
        assert agent.TARGET_SCENE_DURATION == 25.0
    
    def test_validate_input_valid(self, agent, sample_ocr_output, default_config):
        """Test input validation with valid input"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        assert agent.validate_input(input_data) is True
    
    def test_validate_input_invalid_type(self, agent):
        """Test input validation with invalid type"""
        assert agent.validate_input("invalid") is False
        assert agent.validate_input(None) is False
        assert agent.validate_input(123) is False
    
    def test_validate_input_missing_attributes(self, agent):
        """Test input validation with missing attributes"""
        class InvalidInput:
            pass
        
        assert agent.validate_input(InvalidInput()) is False
    
    def test_get_retry_policy(self, agent):
        """Test retry policy configuration"""
        policy = agent.get_retry_policy()
        assert policy.max_attempts == 3
        assert "INVALID_JSON" in policy.retryable_errors
        assert "SCHEMA_VIOLATION" in policy.retryable_errors
    
    def test_execute_basic(self, agent, sample_ocr_output, default_config):
        """Test basic storyboard generation"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        assert isinstance(result, StoryboardJSON)
        assert result.pdf_metadata.filename == "test_document.pdf"
        assert result.pdf_metadata.page_count == 2
        assert result.configuration.verbosity == "medium"
        assert len(result.scenes) > 0
    
    def test_execute_invalid_input(self, agent):
        """Test execution with invalid input raises error"""
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute("invalid")
        
        assert exc_info.value.error_code == "INVALID_INPUT"

    # NOTE: tests for the deleted per-concept extractor / topological sort /
    # per-concept narration & visual generation have been removed as part of
    # Task 5 in docs/superpowers/plans/2026-04-26-unified-script-chunked-render.md.
    # The unified-script flow is exercised in TestGenerateVideoScript and
    # TestChunker; end-to-end execute() behaviour is exercised by
    # test_execute_basic / test_scene_id_uniqueness / test_conceptual_continuity below.

    def test_scene_id_uniqueness(self, agent, sample_ocr_output, default_config):
        """Test that all scene IDs are unique"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        scene_ids = [scene.scene_id for scene in result.scenes]
        assert len(scene_ids) == len(set(scene_ids))  # All IDs should be unique
    
    def test_scene_index_sequential(self, agent, sample_ocr_output, default_config):
        """Test that scene indices are sequential"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        for i, scene in enumerate(result.scenes):
            assert scene.scene_index == i
    
    def test_conceptual_continuity(self, agent, sample_ocr_output, default_config):
        """Test that scenes maintain conceptual continuity through dependencies"""
        input_data = ScriptArchitectInput(sample_ocr_output, default_config)
        result = agent.execute(input_data)
        
        # First scene should have no dependencies
        if len(result.scenes) > 0:
            assert len(result.scenes[0].dependencies) == 0
        
        # Subsequent scenes should reference previous scenes
        for i in range(1, len(result.scenes)):
            assert len(result.scenes[i].dependencies) > 0
            # Should depend on previous scene
            assert f"scene_{i:03d}" in result.scenes[i].dependencies
    
    def test_configuration_defaults(self, agent, sample_ocr_output):
        """Test that default configuration values are used correctly"""
        assert agent.DEFAULT_VERBOSITY == "medium"
        assert agent.DEFAULT_DEPTH == "intermediate"
        assert agent.DEFAULT_AUDIENCE == "undergraduate"
    
    def test_multiple_pages_processing(self, agent, default_config):
        """Test processing of multi-page documents"""
        multi_page_ocr = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="multi_page.pdf",
                page_count=5,
                file_size_bytes=5000000
            ),
            pages=[
                Page(
                    page_number=i,
                    text_blocks=[
                        TextBlock(
                            text=f"Definition {i}: Content for page {i}",
                            bounding_box=BoundingBox(x=100, y=100, width=400, height=30),
                            reading_order=0,
                            confidence=0.95
                        )
                    ],
                    math_expressions=[],
                    diagrams=[]
                )
                for i in range(1, 6)
            ]
        )
        
        input_data = ScriptArchitectInput(multi_page_ocr, default_config)
        result = agent.execute(input_data)
        
        assert isinstance(result, StoryboardJSON)
        assert result.pdf_metadata.page_count == 5
        assert len(result.scenes) > 0


class TestScriptArchitectInput:
    """Test suite for ScriptArchitectInput"""
    
    def test_input_initialization(self):
        """Test ScriptArchitectInput initialization"""
        from src.schemas.source_segment import SourceSegment

        ocr_output = OCROutput(
            pdf_metadata=PDFMetadata(
                filename="test.pdf",
                page_count=1,
                file_size_bytes=1000
            ),
            pages=[
                Page(
                    page_number=1,
                    text_blocks=[],
                    math_expressions=[],
                    diagrams=[]
                )
            ]
        )
        config = Configuration(
            verbosity="medium",
            depth="intermediate",
            audience="undergraduate"
        )
        
        input_data = ScriptArchitectInput(ocr_output, config)
        
        assert input_data.ocr_output == ocr_output
        assert input_data.configuration == config
        assert input_data.segments == []

        segments = [
            SourceSegment(
                segment_id="seg_001",
                order=0,
                page_start=1,
                page_end=1,
                heading="Intro",
                text="Segment text",
            )
        ]
        input_data = ScriptArchitectInput(ocr_output, config, segments=segments)
        assert input_data.segments == segments


class TestGenerateVideoScript:
    """Tests for the unified-script LLM call (Task 3 in
    docs/superpowers/plans/2026-04-26-unified-script-chunked-render.md)."""

    def test_generate_video_script_returns_videoscript_with_ledger(self, monkeypatch):
        """One LLM call → VideoScript with concept ledger preserved."""
        from src.schemas.source_segment import SourceSegment
        from src.schemas.video_script import VideoScript
        from tests.fixtures.ocr_fixtures import minimal_trig_ocr_output

        fake_response = {
            "title": "Sine, Cosine, and the Unit Circle",
            "beats": [
                {
                    "beat_id": "beat_001",
                    "narration": "Sine measures the height of a rotating point on the unit circle.",
                    "visual_hint": "Unit circle with a rotating radius; a y-coordinate trace.",
                    "introduces_concepts": ["unit_circle", "sine"],
                    "references_prior_beats": [],
                },
                {
                    "beat_id": "beat_002",
                    "narration": "Cosine measures the same point's horizontal coordinate.",
                    "visual_hint": "Same unit circle; project the point onto the x-axis.",
                    "introduces_concepts": ["cosine"],
                    "references_prior_beats": ["beat_001"],
                },
            ],
        }

        class _FakeChoice:
            def __init__(self, content):
                self.message = type("M", (), {"content": content})

        class _FakeResp:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        class _FakeCompletions:
            @staticmethod
            def create(**kw):
                return _FakeResp(json.dumps(fake_response))

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False
        agent.client = _FakeClient()

        ocr = minimal_trig_ocr_output()
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")
        segments = [
            SourceSegment(
                segment_id="seg_001",
                order=0,
                page_start=1,
                page_end=1,
                text="Sine and cosine on the unit circle.",
            )
        ]

        # Call _generate_video_script directly (Task 3 method)
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        script = agent._generate_video_script(ocr, config, review_feedback=[], segments=segments)

        assert isinstance(script, VideoScript)
        assert script.concept_ledger() == ["unit_circle", "sine", "cosine"]
        assert len(script.beats) == 2
        assert script.beats[1].references_prior_beats == ["beat_001"]
        assert [beat.source_segment_id for beat in script.beats] == ["seg_001", "seg_001"]

    def test_generate_video_script_uses_one_prompt_per_segment(self, monkeypatch):
        """Segment-scoped generation sees later pages instead of one truncated prefix."""
        from src.schemas.source_segment import SourceSegment
        from tests.fixtures.ocr_fixtures import minimal_trig_ocr_output

        responses = [
            {
                "title": "Segment 1",
                "beats": [
                    {
                        "beat_id": "beat_local_001",
                        "narration": "Explain the first segment.",
                        "visual_hint": "Show the first idea.",
                        "introduces_concepts": ["first"],
                        "references_prior_beats": [],
                    }
                ],
            },
            {
                "title": "Segment 2",
                "beats": [
                    {
                        "beat_id": "beat_local_001",
                        "narration": "Explain the final page result.",
                        "visual_hint": "Show the final result.",
                        "introduces_concepts": ["final_result"],
                        "references_prior_beats": [],
                    }
                ],
            },
        ]
        captured_prompts = []

        class _FakeChoice:
            def __init__(self, content):
                self.message = type("M", (), {"content": content})

        class _FakeResp:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        class _FakeCompletions:
            def create(self, *, messages, **_kw):
                captured_prompts.append(messages[0]["content"])
                return _FakeResp(json.dumps(responses[len(captured_prompts) - 1]))

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False
        agent.client = _FakeClient()
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        segments = [
            SourceSegment(
                segment_id="seg_001",
                order=0,
                page_start=1,
                page_end=1,
                text="Opening material.",
            ),
            SourceSegment(
                segment_id="seg_002",
                order=1,
                page_start=18,
                page_end=18,
                text="Final page result that used to be beyond 8000 characters.",
            ),
        ]

        script = agent._generate_video_script(
            minimal_trig_ocr_output(),
            Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
            review_feedback=[],
            segments=segments,
        )

        assert len(captured_prompts) == 2
        assert "Opening material." in captured_prompts[0]
        assert "Final page result" not in captured_prompts[0]
        assert "Final page result that used to be beyond 8000 characters." in captured_prompts[1]
        assert [beat.beat_id for beat in script.beats] == ["beat_001", "beat_002"]
        assert [beat.source_segment_id for beat in script.beats] == ["seg_001", "seg_002"]

    def test_generate_video_script_falls_back_when_no_api_key(self, monkeypatch):
        from src.schemas.video_script import VideoScript
        from tests.fixtures.ocr_fixtures import minimal_trig_ocr_output

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False
        agent.client = None

        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")
        script = agent._generate_video_script(minimal_trig_ocr_output(), config, review_feedback=[])
        assert isinstance(script, VideoScript)
        assert len(script.beats) >= 1

    def test_generate_video_script_strict_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from tests.fixtures.ocr_fixtures import minimal_trig_ocr_output
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = True
        agent.client = None
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")
        with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
            agent._generate_video_script(minimal_trig_ocr_output(), config, review_feedback=[])

    def test_concept_canonicalization_collapses_aliases(self):
        """F1 mitigation — 'Sine Function', 'sin', 'sine_function' all collapse."""
        from src.agents.script_architect import _canonicalize_concept
        assert _canonicalize_concept("Sine Function") == "sine"
        assert _canonicalize_concept("sin") == "sine"
        assert _canonicalize_concept("sine_function") == "sine"
        assert _canonicalize_concept("THE  Unit-Circle") == "unit_circle"
        assert _canonicalize_concept("trig functions") == "trigonometric_function"
        # Unknown labels pass through normalized
        assert _canonicalize_concept("Some Novel Concept") == "some_novel_concept"


class TestChunker:
    """Tests for the deterministic VideoScript → List[Scene] chunker (Task 4)."""

    def _make_agent(self):
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        # Force the constants the chunker reads
        return agent

    def test_chunker_groups_beats_within_duration_window(self):
        from src.schemas.video_script import VideoScript, Beat
        # Each beat ≈ 30 words / 3 wps ≈ 10 s. Six beats ≈ 60 s.
        beats = [
            Beat(
                beat_id=f"beat_{i:03d}",
                narration=" ".join(["word"] * 30),
                visual_hint="hint",
                introduces_concepts=[f"c{i}"],
            )
            for i in range(1, 7)
        ]
        script = VideoScript(beats=beats)
        agent = self._make_agent()

        scenes = agent._chunk_script_into_scenes(script)
        assert 1 <= len(scenes) <= 3
        # Scene IDs are sequential
        assert [s.scene_id for s in scenes] == [f"scene_{i+1:03d}" for i in range(len(scenes))]
        # Every beat appears in exactly one scene's source_beats
        all_beat_ids = []
        for s in scenes:
            all_beat_ids.extend(b.beat_id for b in s.source_beats)
        assert all_beat_ids == [b.beat_id for b in beats]

    def test_chunker_never_splits_a_beat(self):
        """A beat is the atomic unit. Even if it overshoots MAX_SCENE_DURATION,
        the chunker must keep it whole and log a warning (F3)."""
        from src.schemas.video_script import VideoScript, Beat
        long_beat = Beat(
            beat_id="beat_001",
            narration=" ".join(["word"] * 600),  # ≈ 200 s — exceeds MAX_SCENE_DURATION (45s)
            visual_hint="hint",
            introduces_concepts=["x"],
        )
        agent = self._make_agent()
        scenes = agent._chunk_script_into_scenes(VideoScript(beats=[long_beat]))
        assert len(scenes) == 1
        assert scenes[0].narration.count("word") == 600
        # The atomic beat survives intact
        assert scenes[0].source_beats[0].beat_id == "beat_001"

    def test_chunker_logs_warning_for_oversized_beat(self, caplog):
        """F3 mitigation: an oversized beat triggers a WARNING naming the beat_id."""
        import logging
        from src.schemas.video_script import VideoScript, Beat
        oversized = Beat(
            beat_id="beat_huge",
            narration=" ".join(["word"] * 600),
            visual_hint="hint",
        )
        with caplog.at_level(logging.WARNING, logger="src.agents.script_architect"):
            self._make_agent()._chunk_script_into_scenes(VideoScript(beats=[oversized]))
        assert any("beat_huge" in r.getMessage() for r in caplog.records), \
            f"Expected a warning naming beat_huge, got: {[r.getMessage() for r in caplog.records]}"

    def test_chunker_dependencies_form_single_step_chain(self):
        from src.schemas.video_script import VideoScript, Beat
        beats = [
            Beat(beat_id=f"beat_{i:03d}", narration=" ".join(["word"] * 30), visual_hint="h")
            for i in range(1, 7)
        ]
        scenes = self._make_agent()._chunk_script_into_scenes(VideoScript(beats=beats))
        assert scenes[0].dependencies == []
        for prev, cur in zip(scenes, scenes[1:]):
            assert cur.dependencies == [prev.scene_id]

    def test_chunker_preserves_beat_data_via_pydantic_roundtrip(self):
        """F5 mitigation — source_beats survive Scene.model_dump()."""
        from src.schemas.storyboard import Scene
        from src.schemas.video_script import VideoScript, Beat
        beats = [
            Beat(beat_id="b1", narration=" ".join(["w"] * 30), visual_hint="h", introduces_concepts=["x"]),
            Beat(beat_id="b2", narration=" ".join(["w"] * 30), visual_hint="h", introduces_concepts=["y"]),
        ]
        scenes = self._make_agent()._chunk_script_into_scenes(VideoScript(beats=beats))
        for s in scenes:
            roundtripped = Scene.model_validate(s.model_dump())
            assert [b.beat_id for b in roundtripped.source_beats] == [b.beat_id for b in s.source_beats]

    def test_chunker_keeps_segment_boundaries_and_tags_scenes(self):
        from src.schemas.video_script import Beat, VideoScript

        beats = [
            Beat(
                beat_id="beat_001",
                narration=" ".join(["alpha"] * 30),
                visual_hint="h",
                introduces_concepts=["a"],
                source_segment_id="seg_001",
            ),
            Beat(
                beat_id="beat_002",
                narration=" ".join(["beta"] * 30),
                visual_hint="h",
                introduces_concepts=["b"],
                source_segment_id="seg_002",
            ),
        ]

        scenes = self._make_agent()._chunk_script_into_scenes(VideoScript(beats=beats))

        assert len(scenes) == 2
        assert [scene.source_segment_ids for scene in scenes] == [["seg_001"], ["seg_002"]]


class TestVisualIntentForChunk:
    """Task 7: per-chunk visual planner uses NarrativeContext, never the full PDF."""

    def _make_agent_with_fake_client(self, captured: dict, response_json: str):
        """Build an agent whose chat client captures the prompt and returns
        a canned JSON visual intent."""
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False

        class _Resp:
            def __init__(self, content: str):
                self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]

        class _Completions:
            def create(self, *, messages, **_kw):
                captured["prompt"] = messages[0]["content"]
                return _Resp(response_json)

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        agent.client = _Client()
        return agent

    def test_visual_intent_prompt_excludes_full_pdf_and_ocr(self, monkeypatch):
        """The per-chunk planner must receive only the chunk narration +
        briefing — never the raw PDF text or OCR dump."""
        from src.schemas.storyboard import Configuration
        from src.schemas.video_script import VideoScript, Beat

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        captured: dict = {}
        agent = self._make_agent_with_fake_client(
            captured,
            '{"objects": [], "transformations": [], "beats": [], '
            '"scene_state_in": {}, "scene_state_out": {}, "emphasis_points": []}',
        )

        script = VideoScript(beats=[
            Beat(
                beat_id="beat_001",
                narration="explain sine on the unit circle",
                visual_hint="unit circle with rotating radius",
                introduces_concepts=["sine"],
            ),
        ])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")

        agent._generate_visual_intent_for_chunk(scene, script, config, [])

        prompt = captured["prompt"]
        assert "explain sine on the unit circle" in prompt
        assert "unit circle with rotating radius" in prompt
        # The prompt must not name the source-material dump that the script
        # generator alone is allowed to see.
        assert "PDF" not in prompt
        assert "OCR" not in prompt

    def test_visual_intent_prompt_lists_concepts_already_introduced(self, monkeypatch):
        """For chunks past the first, the prompt must enumerate concepts
        introduced by *earlier* beats so the LLM references — not redefines —
        them."""
        from src.schemas.storyboard import Configuration
        from src.schemas.video_script import VideoScript, Beat

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        captured: dict = {}
        agent = self._make_agent_with_fake_client(
            captured,
            '{"objects": [], "transformations": [], "beats": [], '
            '"scene_state_in": {}, "scene_state_out": {}, "emphasis_points": []}',
        )

        # Two beats long enough to land in separate chunks.
        long_narration = " ".join(["word"] * 250)  # ≈ 83 s — forces own chunk
        script = VideoScript(beats=[
            Beat(
                beat_id="beat_001",
                narration=long_narration,
                visual_hint="circle",
                introduces_concepts=["sine"],
            ),
            Beat(
                beat_id="beat_002",
                narration="now we look at cosine",
                visual_hint="horizontal projection",
                introduces_concepts=["cosine"],
            ),
        ])
        scenes = agent._chunk_script_into_scenes(script)
        assert len(scenes) >= 2, "test setup expected ≥2 chunks"
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")

        agent._generate_visual_intent_for_chunk(scenes[1], script, config, [])

        prompt = captured["prompt"]
        # The cosine chunk's prompt must mention sine as already-introduced.
        assert "sine" in prompt.lower()
        assert "already" in prompt.lower() or "earlier" in prompt.lower()
        # And must NOT claim cosine has already been introduced.
        below_already = prompt.lower().split("already", 1)
        if len(below_already) > 1:
            # If we found an "already" section, cosine should not be in the
            # earlier-concepts block — only sine is.
            assert "cosine" not in below_already[1].split("\n\n")[0]

    def test_visual_intent_falls_back_when_no_api_key(self, monkeypatch):
        """Without OPENAI_API_KEY the planner uses the deterministic fallback
        (and the test does not need a fake client at all)."""
        from src.schemas.storyboard import Configuration
        from src.schemas.video_script import VideoScript, Beat

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False
        agent.client = None

        script = VideoScript(beats=[
            Beat(beat_id="beat_001", narration="some narration about sine",
                 visual_hint="hint", introduces_concepts=["sine"]),
        ])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")

        out = agent._generate_visual_intent_for_chunk(scene, script, config, [])
        # Fallback intent always produces at least one object (the title text).
        assert len(out.mathematical_objects) >= 1

    def test_visual_intent_prompt_includes_catalog_block(self, monkeypatch):
        """Task 12: the visual planner prompt must enumerate the closed
        Manim primitive catalog so the LLM cannot invent unknown classes."""
        from src.schemas.storyboard import Configuration
        from src.schemas.video_script import VideoScript, Beat

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        captured: dict = {}
        agent = self._make_agent_with_fake_client(
            captured,
            '{"objects": [], "transformations": [], "beats": [], '
            '"scene_state_in": {}, "scene_state_out": {}, "emphasis_points": []}',
        )

        script = VideoScript(beats=[
            Beat(beat_id="beat_001", narration="explain sine", visual_hint="circle",
                 introduces_concepts=["sine"]),
        ])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")

        agent._generate_visual_intent_for_chunk(scene, script, config, [])

        prompt = captured["prompt"]
        assert "CLOSED VOCABULARY" in prompt
        # A few catalog entries must appear so the LLM sees the menu.
        for name in ("MathTex", "Circle", "Axes", "Polygon"):
            assert name in prompt
        # The 'primitive' field instruction must be present.
        assert "primitive" in prompt.lower()

    def test_visual_intent_falls_back_when_llm_returns_empty(self, monkeypatch):
        """If the LLM returns an empty object list, fall back to the
        deterministic intent rather than shipping a blank scene."""
        from src.schemas.storyboard import Configuration
        from src.schemas.video_script import VideoScript, Beat

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        captured: dict = {}
        agent = self._make_agent_with_fake_client(
            captured,
            '{"objects": [], "transformations": [], "beats": [], '
            '"scene_state_in": {}, "scene_state_out": {}, "emphasis_points": []}',
        )

        script = VideoScript(beats=[
            Beat(beat_id="beat_001", narration="explain sine", visual_hint="circle",
                 introduces_concepts=["sine"]),
        ])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")

        out = agent._generate_visual_intent_for_chunk(scene, script, config, [])
        assert len(out.mathematical_objects) >= 1


class TestNonTextVisualInvariant:
    """Visual-impact invariant: a scene of only Text/Tex/MathTex objects is a
    text wall. The planner is retried once with explicit feedback; persistent
    violations are logged, never fatal."""

    def _intent(self, objects):
        from src.schemas.storyboard import SceneState, VisualIntent
        return VisualIntent(
            mathematical_objects=objects,
            transformations=[],
            beats=[],
            scene_state_in=SceneState(),
            scene_state_out=SceneState(),
            emphasis_points=[],
        )

    def _text_obj(self):
        from src.schemas.storyboard import MathematicalObject
        return MathematicalObject(
            object_id="test_text",
            purpose="State the narrated idea",
            supports_beat_ids=["beat_001"],
            object_type="text", content="hello",
            primitive="Text", args={"text": "hello"},
        )

    def _equation_obj(self):
        from src.schemas.storyboard import MathematicalObject
        return MathematicalObject(
            object_id="test_equation",
            purpose="Show the narrated equation",
            supports_beat_ids=["beat_001"],
            object_type="equation", content=r"e^{i\pi}",
            primitive="MathTex", args={"tex": r"e^{i\pi}"},
        )

    def _circle_obj(self):
        from src.schemas.storyboard import MathematicalObject
        return MathematicalObject(
            object_id="test_circle",
            purpose="Depict the narrated geometry",
            supports_beat_ids=["beat_001"],
            object_type="geometric_shape", content="unit circle",
            primitive="Circle", args={"radius": 2.0},
        )

    # ── predicate ─────────────────────────────────────────────────────

    def test_all_text_objects_is_text_only(self):
        intent = self._intent([self._text_obj(), self._equation_obj()])
        assert ScriptArchitectAgent._is_text_only_intent(intent) is True

    def test_one_shape_clears_the_invariant(self):
        intent = self._intent([self._text_obj(), self._circle_obj()])
        assert ScriptArchitectAgent._is_text_only_intent(intent) is False

    def test_empty_object_list_counts_as_text_only(self):
        assert ScriptArchitectAgent._is_text_only_intent(self._intent([])) is True

    def test_legacy_objects_without_primitive_use_object_type(self):
        from src.schemas.storyboard import MathematicalObject
        legacy_text = MathematicalObject(object_type="text", content="t")
        legacy_shape = MathematicalObject(object_type="geometric_shape", content="circle")
        assert ScriptArchitectAgent._is_text_only_intent(self._intent([legacy_text])) is True
        assert ScriptArchitectAgent._is_text_only_intent(
            self._intent([legacy_text, legacy_shape])
        ) is False

    # ── retry wiring in execute() ─────────────────────────────────────

    def _minimal_execute_input(self):
        from src.schemas.ocr_output import (
            BoundingBox, OCROutput, Page, PDFMetadata, TextBlock,
        )
        from src.schemas.source_segment import SourceSegment
        ocr = OCROutput(
            pdf_metadata=PDFMetadata(filename="t.pdf", page_count=1, file_size_bytes=10),
            pages=[Page(page_number=1, text_blocks=[TextBlock(
                text="hello world", reading_order=0,
                bounding_box=BoundingBox(x=0, y=0, width=1, height=1),
            )])],
        )
        config = Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        )
        segments = [SourceSegment(
            segment_id="seg_001", order=0, page_start=1, page_end=1, text="hello world",
        )]
        return ScriptArchitectInput(
            ocr_output=ocr, configuration=config, segments=segments,
        )

    def _agent_with_scripted_intents(self, monkeypatch, intents, key_present=True):
        """Agent whose script generation is canned and whose visual planner
        returns the given intents in order, recording each call's feedback."""
        from src.schemas.video_script import Beat, VideoScript

        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict = False
        agent.client = object() if key_present else None
        if key_present:
            monkeypatch.setenv("OPENAI_API_KEY", "test")
        else:
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        script = VideoScript(beats=[Beat(
            beat_id="beat_001", narration=" ".join(["word"] * 30),
            visual_hint="h", introduces_concepts=["c"],
            source_segment_id="seg_001",
        )])
        monkeypatch.setattr(
            agent, "_generate_video_script",
            lambda *a, **kw: script,
        )

        calls = []
        intent_iter = iter(intents)

        def fake_planner(scene, video_script, config, review_feedback, carried_objects=None):
            calls.append(list(review_feedback))
            return next(intent_iter)

        monkeypatch.setattr(agent, "_generate_visual_intent_for_chunk", fake_planner)
        return agent, calls

    def test_text_only_intent_retried_once_with_feedback(self, monkeypatch):
        text_only = self._intent([self._text_obj()])
        visual = self._intent([self._circle_obj()])
        agent, calls = self._agent_with_scripted_intents(monkeypatch, [text_only, visual])

        storyboard = agent.execute(self._minimal_execute_input())

        assert len(calls) == 2
        assert ScriptArchitectAgent.TEXT_ONLY_RETRY_FEEDBACK in calls[1]
        assert storyboard.scenes[0].visual_intent is visual

    def test_visual_intent_not_retried_when_already_visual(self, monkeypatch):
        visual = self._intent([self._circle_obj()])
        agent, calls = self._agent_with_scripted_intents(monkeypatch, [visual])

        agent.execute(self._minimal_execute_input())
        assert len(calls) == 1

    def test_persistent_text_only_logs_warning(self, monkeypatch, caplog):
        import logging
        text_only = self._intent([self._text_obj()])
        agent, calls = self._agent_with_scripted_intents(
            monkeypatch, [text_only, text_only],
        )
        with caplog.at_level(logging.WARNING, logger="src.agents.script_architect"):
            agent.execute(self._minimal_execute_input())
        assert len(calls) == 2
        assert any("Text-only visual intent" in r.getMessage() for r in caplog.records)

    def test_invalid_text_only_retry_keeps_validated_first_plan(self, monkeypatch):
        """A retry that fails purpose validation must not discard the first
        plan (which passed) in favour of the generic narration fallback."""
        from src.schemas.storyboard import MathematicalObject
        text_only = self._intent([self._text_obj()])
        invalid_retry = self._intent([MathematicalObject(
            object_id="no_purpose", object_type="geometric_shape",
            content="circle", primitive="Circle", args={"radius": 1.0},
        )])
        agent, calls = self._agent_with_scripted_intents(
            monkeypatch, [text_only, invalid_retry],
        )
        storyboard = agent.execute(self._minimal_execute_input())
        assert len(calls) == 2
        assert storyboard.scenes[0].visual_intent is text_only

    def test_overloaded_scene_split_respects_scene_ceiling(self, monkeypatch):
        """Budget-error splitting must not push the scene count past
        _effective_max_scenes; at the ceiling it falls back instead."""
        from src.schemas.storyboard import MathematicalObject, TimeInterval
        from src.schemas.video_script import Beat, VideoScript

        agent, calls = self._agent_with_scripted_intents(monkeypatch, [])
        script = VideoScript(beats=[
            Beat(
                beat_id=f"beat_00{i}", narration=" ".join(["word"] * 20),
                visual_hint="h", introduces_concepts=[f"c{i}"],
                source_segment_id="seg_001",
            )
            for i in (1, 2)
        ])
        monkeypatch.setattr(agent, "_generate_video_script", lambda *a, **kw: script)
        monkeypatch.setattr(agent, "_effective_max_scenes", lambda config: 1)
        overloaded = self._intent([
            MathematicalObject(
                object_id=f"o{i}", purpose=f"Step {i}", supports_beat_ids=["beat_001"],
                object_type="geometric_shape", content="", primitive="Circle",
                args={"radius": 0.1}, interval=TimeInterval(t_in=0, t_out=5),
            )
            for i in range(8)
        ])
        monkeypatch.setattr(
            agent, "_generate_visual_intent_for_chunk",
            lambda *a, **kw: (calls.append(1), overloaded)[1],
        )
        storyboard = agent.execute(self._minimal_execute_input())
        assert len(storyboard.scenes) == 1
        assert len(calls) == ScriptArchitectAgent.MAX_PURPOSE_RETRIES + 1

    def test_no_retry_on_keyless_fallback_path(self, monkeypatch):
        """The deterministic fallback would reproduce itself — never retried."""
        text_only = self._intent([self._text_obj()])
        agent, calls = self._agent_with_scripted_intents(
            monkeypatch, [text_only], key_present=False,
        )
        agent.execute(self._minimal_execute_input())
        assert len(calls) == 1

    def test_visual_planner_prompt_carries_visual_impact_rule(self, monkeypatch):
        """The standing anti-text-wall rule must reach every planner prompt."""
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
            beat_id="beat_001", narration="explain sine", visual_hint="circle",
        )])
        scene = agent._chunk_script_into_scenes(script)[0]
        config = Configuration(
            verbosity="medium", depth="intermediate", audience="undergraduate",
        )
        agent._generate_visual_intent_for_chunk(scene, script, config, [])
        assert "VISUAL IMPACT" in captured["prompt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
