"""Pipeline orchestration components.

This package intentionally avoids importing ``src.orchestrator.pipeline`` at
module import time. Doing so can pre-load the target module before
``python -m src.orchestrator.pipeline`` executes it, which triggers runpy's
"found in sys.modules" RuntimeWarning in notebook environments.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig

__all__ = ["Orchestrator", "PipelineAbortError", "PipelineConfig"]


def __getattr__(name: str):
    """Lazily expose orchestrator symbols without eager pipeline imports."""
    if name in __all__:
        from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig

        mapping = {
            "Orchestrator": Orchestrator,
            "PipelineAbortError": PipelineAbortError,
            "PipelineConfig": PipelineConfig,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
