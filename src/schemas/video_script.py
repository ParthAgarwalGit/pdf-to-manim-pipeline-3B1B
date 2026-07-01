"""Unified video-script schema.

The Script Architect produces a VideoScript first — one continuous narration
arc with an explicit concept-introduction ledger. A deterministic chunker
later groups beats into render-sized Scenes (see Task 4 in
docs/superpowers/plans/2026-04-26-unified-script-chunked-render.md).

Design notes:
- Beats are atomic. The chunker may group them into a Scene but never split
  one across Scene boundaries.
- `concept_ledger()` returns a deduped, first-introduction-ordered list.
  Downstream prompts use it to enforce 'reference, do not redefine' on every
  per-chunk LLM call.
- `references_prior_beats` are validated lazily: unknown / forward refs are
  dropped with a WARNING rather than raising, because the script is still
  usable with bad refs (F7 mitigation).
"""

import logging
from typing import List

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class Beat(BaseModel):
    """One narrative beat (≈ one paragraph or thought).

    Each beat is the smallest unit the script architect emits. The chunker
    groups beats into Scenes; a beat is never split across a Scene boundary,
    so a single oversized beat may produce a single oversized Scene
    (with a logged warning — see F3 mitigation).
    """

    beat_id: str = Field(..., description="Stable id, e.g. 'beat_001'")
    narration: str = Field(..., description="Spoken text for this beat")
    visual_hint: str = Field(
        ...,
        description="Short prose description of what should be on screen during "
                    "this beat. The downstream visual planner turns this into "
                    "a structured VisualIntent (Task 7)."
    )
    introduces_concepts: List[str] = Field(
        default_factory=list,
        description="Canonical concept labels first introduced in this beat. "
                    "Lowercase, underscore-separated."
    )
    references_prior_beats: List[str] = Field(
        default_factory=list,
        description="beat_ids whose visuals/concepts this beat builds on. "
                    "Forward / unknown refs are dropped at VideoScript validation."
    )
    source_segment_id: str = Field(
        default="",
        description="SourceSegment id this beat explains, if generated from segmented OCR."
    )


class VideoScript(BaseModel):
    """A single continuous narration arc covering all PDF concepts.

    The ledger property gives a deterministic ordered list of concept
    labels in first-introduction order; downstream code uses this to
    enforce the 'don't re-explain' rule.
    """

    beats: List[Beat] = Field(..., min_length=1)
    title: str = Field(default="", description="Optional title for the video")

    @model_validator(mode="after")
    def _validate_beats(self) -> "VideoScript":
        # Reject duplicate beat_ids (a structural error — never a soft drop)
        seen_ids = set()
        for b in self.beats:
            if b.beat_id in seen_ids:
                raise ValueError(f"duplicate beat_id: {b.beat_id}")
            seen_ids.add(b.beat_id)

        # Drop bad back-references (forward or unknown) with a WARNING.
        # F7 mitigation: keep the script usable if the LLM hallucinates ids.
        valid_so_far: set = set()
        for b in self.beats:
            kept: List[str] = []
            for ref in b.references_prior_beats:
                if ref in valid_so_far:
                    kept.append(ref)
                else:
                    # Either unknown (typo / hallucination) or forward (later beat)
                    logger.warning(
                        "Beat %s references unknown or forward beat %s; dropping.",
                        b.beat_id, ref,
                    )
            b.references_prior_beats = kept
            valid_so_far.add(b.beat_id)

        return self

    def concept_ledger(self) -> List[str]:
        """Concepts in first-introduction order, deduped."""
        seen, out = set(), []
        for b in self.beats:
            for c in b.introduces_concepts:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out
