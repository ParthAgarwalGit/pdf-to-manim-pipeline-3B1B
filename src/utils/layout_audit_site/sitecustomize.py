"""Auto-imported by Python in audited Manim render subprocesses only.

The native renderer puts this directory on PYTHONPATH and sets
PDF2MANIM_LAYOUT_AUDIT; nothing happens without that variable.
"""
import os

if os.environ.get("PDF2MANIM_LAYOUT_AUDIT"):
    try:
        from src.utils.layout_audit import install

        install()
    except Exception:  # an audit must never break a render
        pass
