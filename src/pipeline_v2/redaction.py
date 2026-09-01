"""Secret redaction for persistent V2 diagnostics."""

from __future__ import annotations

import os
import re
from typing import Any


_KEY_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|token|secret|password|credential)"
    r"(\s*[:=]\s*)([^\s,}\]]+|\"[^\"]+\")"
)
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{8,}")
_LIKELY_KEY_PATTERN = re.compile(r"\b(?:sk|AIza|gsk|anthropic)[-_A-Za-z0-9]{12,}\b")


def secret_values() -> set[str]:
    fragments = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    return {
        value
        for key, value in os.environ.items()
        if value and any(fragment in key.upper() for fragment in fragments)
    }


def redact_text(value: str) -> str:
    redacted = value
    for secret in sorted(secret_values(), key=len, reverse=True):
        if len(secret) >= 6:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = _LIKELY_KEY_PATTERN.sub("[REDACTED]", redacted)
    return _KEY_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted)


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value
