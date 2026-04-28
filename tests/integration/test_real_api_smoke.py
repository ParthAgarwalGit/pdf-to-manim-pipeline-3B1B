"""Real-API end-to-end smoke test.

Runs the dry-run pipeline against actual OpenAI + Anthropic APIs using
the fixture PDF. Verifies that the LLM contracts the pipeline depends on
still hold against the live providers — model availability, response
shapes, prompt-length budgets.

This test is marked ``live_api`` so it is excluded from the default
``pytest tests/`` sweep. Run explicitly when you want to verify the
real-API contract:

    pytest -m live_api

Skipped automatically when the required API keys are not set, so the
test can stay green in CI lanes that don't ship credentials.
"""

import os
from pathlib import Path

import pytest

from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.utils.cost_tracker import CostTracker


FIXTURE_PDF = (
    Path(__file__).resolve().parent.parent / "fixtures" / "test_trig_3pages.pdf"
)


_REQUIRED_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _missing_keys():
    return [k for k in _REQUIRED_KEYS if not os.getenv(k)]


@pytest.mark.live_api
@pytest.mark.skipif(
    not FIXTURE_PDF.exists(),
    reason="Fixture PDF not present",
)
@pytest.mark.skipif(
    bool(_missing_keys()),
    reason=f"Live API smoke requires {', '.join(_REQUIRED_KEYS)}",
)
def test_real_api_dry_run_smoke(tmp_path, monkeypatch):
    """End-to-end against real OpenAI + Anthropic.

    Uses ``dry_run=True`` so we stop after the storyboard stage — no Manim
    render, no TTS, no audio compose. The bill for one run is the script
    architect's `_generate_video_script` LLM call plus per-chunk visual-
    intent calls. Bounded by the fixture PDF (3 pages of trigonometry).
    """
    # Make sure cost accounting starts fresh so the test's cost summary
    # only reflects this run, not any cumulative state.
    CostTracker.reset_instance()

    config = PipelineConfig(
        base_output_dir=str(tmp_path),
        dry_run=True,
    )
    orchestrator = Orchestrator(config=config)
    manifest = orchestrator.run_pipeline(str(FIXTURE_PDF), event_source="local")

    # Manifest contract — same invariants as the mocked smoke test.
    assert manifest is not None
    out_dir = Path(manifest.output_directory)
    assert out_dir.exists(), f"output directory missing: {out_dir}"
    assert (out_dir / "manifest.json").exists()

    file_types = {f.type for f in manifest.files}
    assert "storyboard" in file_types
    assert "manifest" in file_types

    # Dry-run guarantees no media artifacts.
    assert manifest.final_video_path is None
    assert "scene_video" not in file_types
    assert "scene_audio" not in file_types

    # ScriptArchitect must produce ≥ 1 scene from a 3-page PDF.
    assert manifest.total_scenes >= 1
    assert manifest.successful_scenes >= 1

    # Cost summary must be populated and non-zero — proves the real API
    # was hit and CostTracker captured it.
    assert manifest.cost_summary is not None, "cost_summary should be set after a real run"
    assert manifest.cost_summary.call_count >= 1, "expected ≥ 1 tracked API call"
    assert manifest.cost_summary.total_usd > 0.0, (
        f"expected non-zero cost; got {manifest.cost_summary.total_usd}"
    )
    # OpenAI is on the script + visual-intent path so it must show up.
    assert "openai" in manifest.cost_summary.by_provider, (
        f"expected openai in providers, got {list(manifest.cost_summary.by_provider)}"
    )

    # RunConfig must reflect the dry_run flag.
    assert manifest.run_config is not None
    assert manifest.run_config.dry_run is True
