"""Guards for the per-task model tiering (cost-efficiency config).

These pin which model each agent uses and that the cost tracker knows the
authoritative rate for it, so a careless edit can't silently move a task onto a
pricier model — or bill students at a stale rate.
"""

import pytest

from src.agents.animation_generator import CODE_GEN_MODEL
from src.agents.script_architect import (
    SEGMENT_SCRIPT_MODEL,
    UNIFIED_SCRIPT_MODEL,
    VISUAL_INTENT_MODEL,
    ScriptArchitectAgent,
    _extract_json_object,
)
from src.agents.storyboard_grader import GRADER_MODEL
from src.agents.vision_ocr import OCR_MODEL
from src.agents.video_quality_reviewer import FIDELITY_MODEL
from src.utils.cost_tracker import PRICING_TABLE


# ---------------------------------------------------------------------------
# Model assignment per task
# ---------------------------------------------------------------------------

def test_task_model_assignments():
    assert CODE_GEN_MODEL == "gemini-3.7-flash"         # Manim code gen
    assert OCR_MODEL == "gpt-5.4"                        # Vision/OCR
    assert UNIFIED_SCRIPT_MODEL == "claude-sonnet-5"     # Unified script
    assert SEGMENT_SCRIPT_MODEL == "gpt-5.4-mini"        # Per-segment beats
    assert VISUAL_INTENT_MODEL == "gpt-5.4-mini"         # Visual intent
    assert GRADER_MODEL == "gpt-5.4-nano"                # Storyboard grading
    assert FIDELITY_MODEL == "gpt-5.4-nano"              # Fidelity review


def test_every_used_model_has_authoritative_pricing():
    expected = {
        ("google", "gemini-3.7-flash"): (0.75, 3.75),
        ("anthropic", "claude-sonnet-5"): (3.00, 15.00),
        ("openai", "gpt-5.4"): (2.50, 15.00),
        ("openai", "gpt-5.4-mini"): (0.75, 4.50),
        ("openai", "gpt-5.4-nano"): (0.20, 1.25),
        # Corrected stale placeholders:
        ("anthropic", "claude-haiku-4-5"): (1.00, 5.00),
        ("anthropic", "claude-opus-4-8"): (5.00, 25.00),
    }
    for key, rate in expected.items():
        assert PRICING_TABLE.get(key) == rate, f"{key} pricing wrong or missing"


# ---------------------------------------------------------------------------
# JSON extraction (Claude returns free text, not response_format json_object)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('Here is the script:\n{"a": 1}\nThanks!', '{"a": 1}'),
    ('no json here', 'no json here'),
])
def test_extract_json_object(text, expected):
    assert _extract_json_object(text) == expected


# ---------------------------------------------------------------------------
# Unified-script routing: Anthropic when available, OpenAI fallback otherwise
# ---------------------------------------------------------------------------

def _bare_agent():
    return ScriptArchitectAgent.__new__(ScriptArchitectAgent)


def test_unified_script_routes_to_anthropic_sonnet_5():
    captured = {}

    class _Block:
        type = "text"
        text = '{"title": "T", "beats": []}'

    class _Resp:
        content = [_Block()]

    class _Messages:
        def create(self, **kw):
            captured["model"] = kw["model"]
            captured["thinking"] = kw.get("thinking")
            return _Resp()

    class _Client:
        messages = _Messages()

    agent = _bare_agent()
    agent.anthropic_client = _Client()
    agent.client = None

    data = agent._complete_unified_script_json("prompt")
    assert data == {"title": "T", "beats": []}
    assert captured["model"] == "claude-sonnet-5"
    assert captured["thinking"] == {"type": "disabled"}  # cost-conscious


def test_unified_script_falls_back_to_openai_when_no_anthropic():
    captured = {}

    class _Msg:
        content = '{"title": "O", "beats": []}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kw):
            captured["model"] = kw["model"]
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        chat = _Chat()

    agent = _bare_agent()
    agent.anthropic_client = None
    agent.client = _OpenAI()

    data = agent._complete_unified_script_json("prompt")
    assert data == {"title": "O", "beats": []}
    assert captured["model"] == "gpt-5.4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
