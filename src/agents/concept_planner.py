"""Outline step: list each segment's concepts as an editable ConceptPlan.

Runs after document segmentation and before scripting. One LLM call lists
the concepts per segment; without a key, each segment's heading (or first
words) stands in as its concept so the plan is still usable.

Global flags seed the defaults (``--verbosity low`` -> brief, ``high`` ->
deep) and the professor's brief overrides per concept: a depth, their own
analogy, or a rejected analogy recorded as a note.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.agents.script_editor import LLM, default_llm
from src.schemas.concept_plan import ConceptPlan, ConceptPlanEntry
from src.schemas.directives import ProfessorDirectives

logger = logging.getLogger(__name__)

_VERBOSITY_DEPTH = {"low": "brief", "medium": "standard", "high": "deep"}
_MAX_EXCERPT = 1200


def _label(text: str) -> str:
    text = re.sub(r"^[\d.\s]+", "", (text or "").strip())  # drop section numbers
    return re.sub(r"\s+", " ", text)[:60].strip()


class ConceptPlannerAgent:
    def __init__(self, llm: Optional[LLM] = None, use_default_llm: bool = True):
        self.llm = llm if llm is not None else (default_llm() if use_default_llm else None)

    def plan(
        self,
        segments: List[Any],
        configuration: Any = None,
        directives: Optional[ProfessorDirectives] = None,
    ) -> ConceptPlan:
        default = _VERBOSITY_DEPTH.get(getattr(configuration, "verbosity", "medium"), "standard")
        listed = self._list_concepts(segments) if segments else {}
        entries = [
            ConceptPlanEntry(concept=concept, segment_id=segment_id, depth=default)
            for segment_id, concepts in listed.items()
            for concept in concepts
        ]
        plan = ConceptPlan(entries=entries)
        if directives is not None:
            merge_directives(plan, directives)
        return plan

    def _list_concepts(self, segments: List[Any]) -> Dict[str, List[str]]:
        if self.llm is not None:
            try:
                listing = "\n\n".join(
                    f"SEGMENT {s.segment_id} (heading: {s.heading or 'none'}):\n{(s.text or '')[:_MAX_EXCERPT]}"
                    for s in segments
                )
                data = self.llm(
                    "List the distinct concepts each segment of these lecture notes teaches, "
                    "in teaching order, 1-6 per segment, as short lowercase labels "
                    "(e.g. 'eigenvector', 'determinant as area'). A concept belongs to the "
                    "segment that first teaches it.\n\n" + listing
                    + '\n\nRespond as {"segments": [{"segment_id": "...", "concepts": ["..."]}]}'
                )
                result: Dict[str, List[str]] = {}
                known = {s.segment_id for s in segments}
                for item in data.get("segments") or []:
                    if isinstance(item, dict) and item.get("segment_id") in known:
                        concepts = [str(c).strip() for c in item.get("concepts") or [] if str(c).strip()]
                        result[item["segment_id"]] = concepts[:6]
                if result:
                    return {s.segment_id: result.get(s.segment_id, []) for s in segments}
            except Exception as exc:
                logger.warning("Concept listing by LLM failed (%s); using headings.", exc)
        return {
            s.segment_id: [_label(s.heading) or _label(" ".join((s.text or "").split()[:6])) or s.segment_id]
            for s in segments
        }


def merge_directives(plan: ConceptPlan, directives: ProfessorDirectives) -> None:
    """Apply the brief's concept directives to matching plan entries."""
    for directive in directives.concepts:
        target = directive.concept.lower().strip()
        matches = [
            e for e in plan.entries
            if target and (target in e.concept.lower() or e.concept.lower() in target)
        ] if target else []
        if not matches and target:
            entry = ConceptPlanEntry(concept=directive.concept)
            plan.entries.append(entry)
            matches = [entry]
        for entry in matches:
            if directive.depth:
                entry.depth = directive.depth
            if directive.analogy_brief:
                entry.analogy, entry.professor_analogy_brief = "professor", directive.analogy_brief
            notes = [directive.notes] if directive.notes else []
            notes += [f"never use the {a} analogy" for a in directive.avoid_analogies]
            if notes:
                entry.professor_notes = "; ".join(filter(None, [entry.professor_notes] + notes))


def skip_directives(plan: Optional[ConceptPlan]) -> List[Any]:
    """ConceptDirectives for the plan's skipped concepts (compliance checks)."""
    from src.schemas.directives import ConceptDirective

    return [ConceptDirective(concept=e.concept, depth="skip") for e in (plan.skipped() if plan else [])]
