"""Helpers for normalizing complete-script repair responses."""

from __future__ import annotations

import re


def normalize_complete_script(text: str) -> str:
    """Remove an accidental single Markdown fence without accepting patches."""
    value = text.strip()
    match = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if match:
        value = match.group(1).strip()
    if value.startswith("diff ") or "@@ " in value:
        raise ValueError("Vision reviewer returned a patch instead of a complete script.")
    return value
