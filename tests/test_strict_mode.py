"""Tests for --strict mode across all agents.

When strict=True, agents that depend on external API keys (OpenAI, Anthropic)
must raise AgentExecutionError at initialization instead of silently falling
back to mock/heuristic outputs.

Agents under test:
- AnimationCodeGeneratorAgent (Anthropic)
- ScriptArchitectAgent (OpenAI)
- VisionOCRAgent (OpenAI)
- TTSAgent (OpenAI)
- StoryboardGraderAgent (OpenAI)
"""

import os
from unittest.mock import patch

import pytest

from src.agents.base import AgentExecutionError


# ---------------------------------------------------------------------------
# Utility: clear all API keys from environment
# ---------------------------------------------------------------------------

def _clear_api_keys():
    """Return a dict suitable for patch.dict that removes all API keys."""
    return {k: v for k, v in os.environ.items()
            if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}


# ---------------------------------------------------------------------------
# AnimationCodeGeneratorAgent (Anthropic)
# ---------------------------------------------------------------------------

class TestAnimationGeneratorStrict:
    def test_strict_raises_without_anthropic_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.animation_generator import AnimationCodeGeneratorAgent
            with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
                AnimationCodeGeneratorAgent(strict=True)

    def test_non_strict_allows_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.animation_generator import AnimationCodeGeneratorAgent
            agent = AnimationCodeGeneratorAgent(strict=False)
            assert agent.client is None


# ---------------------------------------------------------------------------
# ScriptArchitectAgent (OpenAI)
# ---------------------------------------------------------------------------

class TestScriptArchitectStrict:
    def test_strict_raises_without_openai_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.script_architect import ScriptArchitectAgent
            with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
                ScriptArchitectAgent(strict=True)

    def test_non_strict_allows_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.script_architect import ScriptArchitectAgent
            agent = ScriptArchitectAgent(strict=False)
            assert agent.client is None


# ---------------------------------------------------------------------------
# VisionOCRAgent (OpenAI)
# ---------------------------------------------------------------------------

class TestVisionOCRStrict:
    def test_strict_raises_without_openai_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.vision_ocr import VisionOCRAgent
            with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
                VisionOCRAgent(strict=True)

    def test_non_strict_allows_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.vision_ocr import VisionOCRAgent
            agent = VisionOCRAgent(strict=False)
            assert agent.client is None


# ---------------------------------------------------------------------------
# TTSAgent (OpenAI)
# ---------------------------------------------------------------------------

class TestTTSStrict:
    def test_strict_raises_without_openai_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.tts_agent import TTSAgent
            with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
                TTSAgent(strict=True)

    def test_non_strict_allows_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.tts_agent import TTSAgent
            agent = TTSAgent(strict=False)
            assert agent.client is None


# ---------------------------------------------------------------------------
# StoryboardGraderAgent (OpenAI)
# ---------------------------------------------------------------------------

class TestStoryboardGraderStrict:
    def test_strict_raises_without_openai_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.storyboard_grader import StoryboardGraderAgent
            with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
                StoryboardGraderAgent(strict=True)

    def test_non_strict_allows_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.agents.storyboard_grader import StoryboardGraderAgent
            agent = StoryboardGraderAgent(strict=False)
            assert agent.client is None


# ---------------------------------------------------------------------------
# Pipeline-level strict mode
# ---------------------------------------------------------------------------

class TestPipelineStrictConfig:
    def test_strict_propagates_to_agents(self):
        """Strict mode in PipelineConfig should raise during Orchestrator init."""
        with patch.dict(os.environ, {}, clear=True):
            from src.orchestrator.pipeline import Orchestrator, PipelineConfig
            config = PipelineConfig(strict=True)
            # Without API keys, strict mode should raise during init
            # because VisionOCR, ScriptArchitect, or TTS will fail
            with pytest.raises(AgentExecutionError, match="MISSING_API_KEY"):
                Orchestrator(config=config)

    def test_non_strict_pipeline_initializes_without_keys(self):
        """Non-strict pipeline should initialize fine without API keys."""
        with patch.dict(os.environ, {}, clear=True):
            from src.orchestrator.pipeline import Orchestrator, PipelineConfig
            config = PipelineConfig(strict=False)
            orchestrator = Orchestrator(config=config)
            assert orchestrator is not None
