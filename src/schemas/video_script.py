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
- Typed-beat fields (`role`, `analogy_id`, `depth`, `author`, `locked`) all
  default, so storyboards written before they existed still load. They let
  professor edits target "the pizza analogy" or "optional depth" instead of
  raw beat ids (docs/superpowers/specs/2026-11-07-professor-control-and-presenter-design.md).
"""

import logging
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

BeatRole = Literal["core", "analogy", "example", "derivation", "intuition", "recap", "bridge"]
MAX_BEAT_DEPTH = 3


class Analogy(BaseModel):
    """An analogy shared by its introduction beat and every callback beat."""

    analogy_id: str = Field(..., description="Stable id, e.g. 'analogy_water_pipes'")
    label: str = Field(default="", description="Short name, e.g. 'water flowing through pipes'")
    target_concepts: List[str] = Field(default_factory=list)
    mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Analogy term -> concept, e.g. {'water pressure': 'voltage'}",
    )
    origin: Literal["generated", "professor"] = "generated"
    professor_brief: str = Field(
        default="",
        description="The professor's own description, kept verbatim.",
    )


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
    role: BeatRole = Field(
        default="core",
        description="What the beat does; edit ops and density filters key off it."
    )
    analogy_id: Optional[str] = Field(
        default=None,
        description="Analogy this beat introduces or calls back (VideoScript.analogies)."
    )
    depth: int = Field(
        default=0,
        ge=0,
        le=MAX_BEAT_DEPTH,
        description="0 = essential to the argument; higher = optional elaboration."
    )
    author: Literal["generated", "professor"] = Field(
        default="generated",
        description="Provenance of the beat text."
    )
    locked: bool = Field(
        default=False,
        description="Professor-approved text: no LLM pass may rewrite it."
    )

    @field_validator("analogy_id", mode="before")
    @classmethod
    def _blank_analogy_is_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v


class VideoScript(BaseModel):
    """A single continuous narration arc covering all PDF concepts.

    The ledger property gives a deterministic ordered list of concept
    labels in first-introduction order; downstream code uses this to
    enforce the 'don't re-explain' rule.
    """

    beats: List[Beat] = Field(..., min_length=1)
    title: str = Field(default="", description="Optional title for the video")
    analogies: List[Analogy] = Field(
        default_factory=list,
        description="Analogies used by beats (Beat.analogy_id)."
    )

    @model_validator(mode="after")
    def _validate_beats(self) -> "VideoScript":
        # Reject duplicate beat_ids (a structural error — never a soft drop)
        seen_ids = set()
        for b in self.beats:
            if b.beat_id in seen_ids:
                raise ValueError(f"duplicate beat_id: {b.beat_id}")
            seen_ids.add(b.beat_id)

        analogy_ids = set()
        for analogy in self.analogies:
            if analogy.analogy_id in analogy_ids:
                raise ValueError(f"duplicate analogy_id: {analogy.analogy_id}")
            analogy_ids.add(analogy.analogy_id)
        # A beat may name an analogy the LLM forgot to declare. Keep the
        # grouping (edits target analogies by id) and declare a stub for it.
        for b in self.beats:
            if b.analogy_id and b.analogy_id not in analogy_ids:
                logger.warning(
                    "Beat %s uses undeclared analogy %s; adding a stub declaration.",
                    b.beat_id, b.analogy_id,
                )
                label = b.analogy_id.removeprefix("analogy_").replace("_", " ")
                self.analogies.append(Analogy(analogy_id=b.analogy_id, label=label))
                analogy_ids.add(b.analogy_id)

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
