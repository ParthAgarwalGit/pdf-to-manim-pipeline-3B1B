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


def test_an_unreadable_ocr_batch_is_retried_then_read_page_by_page(tmp_path):
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
            # Batch 0 (pages 1-3) is always empty as a batch; read alone, page 2
            # answers and pages 1 and 3 stay empty. Batch 1 answers at once.
            answers = {
                "ocr-000-p002": '{"facts":[{"page_number":2,"text":"y","kind":"equation"}]}',
                "ocr-001": '{"facts":[{"page_number":4,"text":"x","kind":"text"}]}',
            }
            return ProviderResponse(text=answers.get(task_id, ""), provider="google", model="m",
                                    input_units=1, output_units=1, exact_usage=True, raw={})

    source = SourceDocument(
        filename="doc.pdf", sha256="a", page_count=4, low_confidence_pages=[1, 2, 3, 4],
        facts=[SourceFact(fact_id="p001-f000", page_number=1, text="t")],
    )
    result = asyncio.run(PlanningService(Router()).enrich_low_confidence_pages(source, pdf, tmp_path / "ocr"))

    largest = OCR_OUTPUT_BUDGETS[-1]
    assert budgets == (
        [("ocr-000", b) for b in OCR_OUTPUT_BUDGETS]
        + [(f"ocr-000-p00{page}", largest) for page in (1, 2, 3)]
        + [("ocr-001", OCR_OUTPUT_BUDGETS[0])]
    )
    assert [(name, kw["pages"]) for name, kw in events] == (
        [("ocr_batch_unreadable", [1, 2, 3])] * 2
        + [("ocr_batch_unreadable", [1]), ("ocr_batch_unreadable", [3])]
    )
    assert [(f.page_number, f.text, f.kind) for f in result.facts] == [
        (1, "t", "text"), (2, "y", "equation"), (4, "x", "text"),
    ]
