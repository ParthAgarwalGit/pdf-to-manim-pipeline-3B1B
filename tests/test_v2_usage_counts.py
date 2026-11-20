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


def test_an_unreadable_ocr_batch_is_retried_then_skipped(tmp_path):
    import asyncio
    import pymupdf

    from src.pipeline_v2.models import SourceDocument, SourceFact
    from src.pipeline_v2.planning import OCR_OUTPUT_BUDGETS, PlanningService
    from src.pipeline_v2.providers import ProviderResponse

    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as document:
        for _ in range(4):
            document.new_page()
        document.save(str(pdf))
    events, budgets = [], []

    class Router:
        strict_keys = False
        artifacts = SimpleNamespace(event=lambda name, **kw: events.append((name, kw)))

        async def vision(self, *, task_id, max_output_tokens, **kwargs):
            budgets.append((task_id, max_output_tokens))
            # Batch 0 is always empty; batch 1 answers on its first try.
            text = "" if task_id == "ocr-000" else '{"facts":[{"page_number":4,"text":"x","kind":"text"}]}'
            return ProviderResponse(text=text, provider="google", model="m", input_units=1,
                                    output_units=1, exact_usage=True, raw={})

    source = SourceDocument(
        filename="doc.pdf", sha256="a", page_count=4, low_confidence_pages=[1, 2, 3, 4],
        facts=[SourceFact(fact_id="p001-f000", page_number=1, text="t")],
    )
    result = asyncio.run(PlanningService(Router()).enrich_low_confidence_pages(source, pdf, tmp_path / "ocr"))

    assert budgets == [("ocr-000", b) for b in OCR_OUTPUT_BUDGETS] + [("ocr-001", OCR_OUTPUT_BUDGETS[0])]
    assert [name for name, _ in events] == ["ocr_batch_unreadable"] * 2
    assert [f.text for f in result.facts] == ["t", "x"]
