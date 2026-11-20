"""Gemini reports a token count as None for an empty or blocked answer."""

from pathlib import Path
from types import SimpleNamespace

from src.pipeline_v2.providers import ProviderRouter, estimate_tokens


def test_vision_call_survives_missing_token_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG")

    class FakeModels:
        def generate_content(self, **kwargs):
            usage = SimpleNamespace(prompt_token_count=1200, candidates_token_count=None)
            return SimpleNamespace(text=None, usage_metadata=usage, finish_reason="MAX_TOKENS")

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setattr("google.genai.Client", FakeClient)

    response = ProviderRouter._invoke_google_vision("gemini-4-argon", "sys", "read", [Path(image)], 100)

    assert response.text == ""
    assert response.input_units == 1200
    assert response.output_units == estimate_tokens("")
