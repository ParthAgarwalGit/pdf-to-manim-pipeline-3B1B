"""Concept plan: per-concept depth and analogy choices, set before scripting.

The cheapest moment to steer density (Part D). An outline step lists the
concepts of each source segment; the professor adjusts depth (skip / brief /
standard / deep) and analogy use per concept, then the segment script
prompts follow it. Skipped concepts are recorded as professor omissions.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

Depth = Literal["skip", "brief", "standard", "deep"]

DEPTH_INSTRUCTIONS = {
    "skip": "do not cover it",
    "brief": "one short beat, the essential idea only",
    "standard": "explain it normally",
    "deep": "go deeper: add a derivation step or a worked example",
}


class ConceptPlanEntry(BaseModel):
    concept: str
    segment_id: str = Field(default="", description="Empty = applies wherever the concept appears")
    depth: Depth = "standard"
    analogy: Literal["auto", "none", "professor"] = "auto"
    professor_analogy_brief: str = ""
    professor_notes: str = Field(
        default="", description="e.g. 'students always confuse X with Y; address it'"
    )

    def instruction(self) -> str:
        parts = [f"{self.concept}: {DEPTH_INSTRUCTIONS[self.depth]}"]
        if self.depth != "skip":
            if self.analogy == "none":
                parts.append("no analogy")
            elif self.analogy == "professor" and self.professor_analogy_brief:
                parts.append(f"use the professor's analogy: {self.professor_analogy_brief}")
            if self.professor_notes:
                parts.append(f"professor's note: {self.professor_notes}")
        return "; ".join(parts) + "."


class ConceptPlan(BaseModel):
    entries: List[ConceptPlanEntry] = Field(default_factory=list)

    def for_segment(self, segment_id: str) -> List[ConceptPlanEntry]:
        return [e for e in self.entries if e.segment_id in ("", segment_id)]

    def skipped(self) -> List[ConceptPlanEntry]:
        return [e for e in self.entries if e.depth == "skip"]
