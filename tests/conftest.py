"""Global pytest configuration: Hypothesis profiles and marker discipline."""

import logging
import os
import pytest
from hypothesis import settings

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
# fast  — used locally and on every PR push; limits examples to keep suite quick
# ci    — used on merge-to-main and the dedicated property lane; more thorough
settings.register_profile("fast", max_examples=50, deadline=None)
settings.register_profile("ci",   max_examples=200, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "fast"))


# ---------------------------------------------------------------------------
# Cross-test state isolation
# ---------------------------------------------------------------------------
# Property tests instantiate the Orchestrator dozens of times per example;
# without reset, the CostTracker singleton accumulates entries and the root
# logger collects handlers, making each successive test slower until the
# property lane appears to hang. This fixture wipes the known global state
# before every test so each one starts from a clean slate.

@pytest.fixture(autouse=True)
def _reset_cross_test_state():
    """Reset known global singletons between tests so property-lane runs do
    not balloon. Cheap (microseconds) for non-property tests."""
    # Cost tracker singleton
    try:
        from src.utils.cost_tracker import CostTracker
        CostTracker.reset_instance()
    except Exception:
        pass
    # Root logger handlers piled up by Orchestrator._setup_logging across runs
    pipeline_logger = logging.getLogger("src.orchestrator.pipeline")
    pipeline_logger.handlers.clear()
    yield


# ---------------------------------------------------------------------------
# API-key isolation (hermetic default lane)
# ---------------------------------------------------------------------------
# src/cli.py calls load_dotenv() at IMPORT time, so the moment test_cli is
# collected the real ANTHROPIC_API_KEY / OPENAI_API_KEY leak into the whole
# pytest process. Any test that doesn't explicitly clear them then takes the
# live-provider path — slow (~11 min), costs real money, and flakes on
# non-deterministic LLM output. This fixture makes the default lane hermetic:
# every test starts keyless unless it is explicitly marked @pytest.mark.live_api
# (those legitimately need credentials and are excluded from the default sweep).
# Tests that exercise a "key present" branch set their own key via monkeypatch
# in the test body, which runs after this fixture.

@pytest.fixture(autouse=True)
def _isolate_api_keys(request, monkeypatch):
    """Strip real provider API keys for every non-live_api test."""
    if request.node.get_closest_marker("live_api"):
        return
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# @property marker discipline
# ---------------------------------------------------------------------------

def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail collection if any @given test is missing @pytest.mark.property.

    This enforces that Hypothesis property tests are always identifiable by
    marker so they can be excluded from the fast PR lane and run separately
    with HYPOTHESIS_PROFILE=ci.
    """
    violations = []
    for item in session.items:
        fn = getattr(item, "function", None)
        if fn is not None and hasattr(fn, "hypothesis"):
            if item.get_closest_marker("property") is None:
                violations.append(item.nodeid)

    if violations:
        lines = "\n".join(f"  {n}" for n in violations)
        pytest.exit(
            f"\nCollection error: the following @given tests are missing "
            f"@pytest.mark.property:\n{lines}\n"
            "Add @pytest.mark.property to each test listed above.",
            returncode=4,
        )
