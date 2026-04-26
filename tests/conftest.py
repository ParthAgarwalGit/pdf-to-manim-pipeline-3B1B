"""Global pytest configuration: Hypothesis profiles and marker discipline."""

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
