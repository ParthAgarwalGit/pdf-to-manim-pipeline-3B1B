"""Lesson plan: the narrative spine a video is scripted from.

One planning call reads the whole source and decides, before any narration is
written: the single question the video answers, the hook, the running example,
the recurring visual motifs, the acts in viewing order (not document order) and
what happens to every source segment (covered | compressed | deferred |
omitted, each with a reason). The script writer, the visual planner, the
bridge writer and the grader all read this plan.

Coverage posture (decided 2026-07-31, "Hybrid C"): the plan may compress or
defer a segment, and may omit one with a stated reason, but never silently:
``LessonPlan.from_llm`` records a decision for every segment id it is given.

Everything here is additive: scripts and storyboards written before the plan
existed load unchanged (``VideoScript.plan`` defaults to None, ``Beat.act_id``
to "").
"""

import re
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

ActKind = Literal["hook", "build", "example", "payoff", "application", "recap"]
Treatment = Literal["covered", "compressed", "deferred", "omitted"]

_ACT_KINDS = ("hook", "build", "example", "payoff", "application", "recap")
_TREATMENTS = ("covered", "compressed", "deferred", "omitted")
# Fewer words than this cannot carry one idea plus its picture.
MIN_ACT_WORDS = 25
MAX_ACTS = 9
MAX_MOTIFS = 6


class RunningExample(BaseModel):
    """The one concrete case the whole video keeps returning to."""

    description: str = ""
    values: Dict[str, str] = Field(
        default_factory=dict,
        description="Named concrete values, e.g. {'A': '[[3, 1], [1, 2]]'}",
    )


class VisualMotif(BaseModel):
    """A recurring on-screen object with a fixed look and meaning."""

    motif_id: str
    what: str = Field(default="", description="What is drawn, concretely")
    color: str = Field(default="", description="Manim color name, e.g. BLUE_B")
    stands_for: str = Field(default="", description="The idea it represents")


class PlanAct(BaseModel):
    """One step of the argument: a question the viewer has, and its answer."""

    act_id: str
    kind: ActKind = "build"
    viewer_question: str = ""
    key_idea: str = ""
    visual: str = Field(default="", description="What the viewer watches change")
    source_segments: List[str] = Field(default_factory=list)
    words: int = Field(default=MIN_ACT_WORDS, ge=1)


class CoverageDecision(BaseModel):
    segment_id: str
    treatment: Treatment = "covered"
    reason: str = ""


class LessonPlan(BaseModel):
    title: str = ""
    central_question: str
    hook: str = ""
    promise: str = ""
    takeaway: str = ""
    running_example: RunningExample = Field(default_factory=RunningExample)
    motifs: List[VisualMotif] = Field(default_factory=list)
    acts: List[PlanAct] = Field(..., min_length=1)
    coverage: List[CoverageDecision] = Field(default_factory=list)

    # ── lookups ─────────────────────────────────────────────────────────

    def act(self, act_id: str) -> Optional[PlanAct]:
        return next((a for a in self.acts if a.act_id == act_id), None)

    def act_ids(self) -> List[str]:
        return [a.act_id for a in self.acts]

    def total_words(self) -> int:
        return sum(a.words for a in self.acts)

    def cap_words(self, budget: int) -> None:
        """Scale the acts down so their words sum to at most ``budget``; never up."""
        if budget and self.total_words() > budget:
            _rescale_words(self.acts, int(budget))

    def treatment_of(self, segment_id: str) -> Optional[str]:
        return next((c.treatment for c in self.coverage if c.segment_id == segment_id), None)

    def planned_out_segment_ids(self) -> List[str]:
        """Segments the plan deliberately leaves without narration of their own."""
        return [c.segment_id for c in self.coverage if c.treatment in ("deferred", "omitted")]

    def visual_world_lines(self) -> List[str]:
        """The recurring look of the video, one fact per line, for planner and codegen prompts."""
        lines: List[str] = []
        example = self.running_example
        if example.description:
            lines.append(f"Running example: {example.description}")
        for name, value in example.values.items():
            lines.append(f"Value: {name} = {value}")
        for motif in self.motifs:
            parts = [f"Motif {motif.motif_id}: {motif.what}" if motif.what else f"Motif {motif.motif_id}"]
            if motif.color:
                parts.append(f"color {motif.color}")
            if motif.stands_for:
                parts.append(f"stands for {motif.stands_for}")
            lines.append("; ".join(parts))
        return lines

    def coverage_report(self) -> Dict[str, Any]:
        """Audit of what happened to each source segment (reported, never a gate)."""
        counts = {t: 0 for t in _TREATMENTS}
        for decision in self.coverage:
            counts[decision.treatment] += 1
        return {
            "counts": counts,
            "decisions": [d.model_dump() for d in self.coverage],
        }

    # ── construction from an LLM response ───────────────────────────────

    @classmethod
    def from_llm(
        cls,
        data: Any,
        segment_ids: Optional[Iterable[str]] = None,
        word_budget: Optional[int] = None,
    ) -> "LessonPlan":
        """Build a plan from the planner's JSON, tolerating sloppy output.

        Raises ValueError when there is no usable plan (no central question
        or no act with a key idea); the caller then scripts without a plan.

        - acts are renumbered ``act_01``, ``act_02``, ... in the given order
          and capped at MAX_ACTS;
        - unknown ``kind`` becomes "build"; unknown segment ids are dropped;
        - each act's ``words`` is floored at MIN_ACT_WORDS and, when
          ``word_budget`` is given, all are rescaled to sum to the budget;
        - every id in ``segment_ids`` ends with exactly one coverage decision:
          the planner's own, else "covered" when an act draws on it, else
          "omitted" with a reason saying the plan did not place it.
        """
        if not isinstance(data, dict):
            raise ValueError("lesson plan response is not a JSON object")
        known = [str(s) for s in (segment_ids or [])]
        known_set = set(known)

        central_question = _text(data.get("central_question"))
        if not central_question:
            raise ValueError("lesson plan has no central_question")

        acts: List[PlanAct] = []
        for raw in data.get("acts") if isinstance(data.get("acts"), list) else []:
            if not isinstance(raw, dict):
                continue
            key_idea = _text(raw.get("key_idea"))
            if not key_idea:
                continue
            kind = _text(raw.get("kind")).lower()
            segments = [
                str(s) for s in (raw.get("source_segments") or [])
                if isinstance(s, str) and (not known_set or s in known_set)
            ]
            acts.append(PlanAct(
                act_id=f"act_{len(acts) + 1:02d}",
                kind=kind if kind in _ACT_KINDS else "build",
                viewer_question=_text(raw.get("viewer_question")),
                key_idea=key_idea,
                visual=_text(raw.get("visual")),
                source_segments=list(dict.fromkeys(segments)),
                words=max(MIN_ACT_WORDS, _int(raw.get("words"), MIN_ACT_WORDS)),
            ))
            if len(acts) >= MAX_ACTS:
                break
        if not acts:
            raise ValueError("lesson plan has no act with a key_idea")
        if word_budget:
            _rescale_words(acts, int(word_budget))

        example_raw = data.get("running_example") if isinstance(data.get("running_example"), dict) else {}
        values_raw = example_raw.get("values") if isinstance(example_raw.get("values"), dict) else {}
        example = RunningExample(
            description=_text(example_raw.get("description")),
            values={str(k): _text(v) for k, v in values_raw.items() if _text(v)},
        )

        motifs: List[VisualMotif] = []
        seen_motifs = set()
        for raw in data.get("motifs") if isinstance(data.get("motifs"), list) else []:
            if not isinstance(raw, dict):
                continue
            motif_id = _slug(raw.get("motif_id") or raw.get("what"))
            if not motif_id or motif_id in seen_motifs:
                continue
            seen_motifs.add(motif_id)
            motifs.append(VisualMotif(
                motif_id=motif_id,
                what=_text(raw.get("what")),
                color=_text(raw.get("color")).upper().replace(" ", "_"),
                stands_for=_text(raw.get("stands_for")),
            ))
            if len(motifs) >= MAX_MOTIFS:
                break

        decided: Dict[str, CoverageDecision] = {}
        for raw in data.get("coverage") if isinstance(data.get("coverage"), list) else []:
            if not isinstance(raw, dict):
                continue
            segment_id = _text(raw.get("segment_id"))
            if not segment_id or (known_set and segment_id not in known_set) or segment_id in decided:
                continue
            treatment = _text(raw.get("treatment")).lower()
            decided[segment_id] = CoverageDecision(
                segment_id=segment_id,
                treatment=treatment if treatment in _TREATMENTS else "covered",
                reason=_text(raw.get("reason")),
            )
        used = {s for act in acts for s in act.source_segments}
        coverage: List[CoverageDecision] = []
        for segment_id in known or list(decided):
            decision = decided.get(segment_id)
            if decision is None:
                decision = (
                    CoverageDecision(segment_id=segment_id, treatment="covered",
                                     reason="An act draws on this segment.")
                    if segment_id in used else
                    CoverageDecision(segment_id=segment_id, treatment="omitted",
                                     reason="The lesson plan did not place this segment.")
                )
            elif decision.treatment in ("deferred", "omitted") and not decision.reason:
                decision = decision.model_copy(update={"reason": "No reason given by the planner."})
            coverage.append(decision)

        return cls(
            title=_text(data.get("title")),
            central_question=central_question,
            hook=_text(data.get("hook")),
            promise=_text(data.get("promise")),
            takeaway=_text(data.get("takeaway")),
            running_example=example,
            motifs=motifs,
            acts=acts,
            coverage=coverage,
        )


def default_word_budget(source_words: int) -> int:
    """Narration words for a run with no requested length.

    About a third of the source, between a two-minute and a fourteen-minute
    video at the spoken pace: long enough to cover a chapter's main line,
    short enough that the plan has to choose.
    """
    return int(min(2400, max(350, round(source_words * 0.30))))


def _rescale_words(acts: List[PlanAct], budget: int) -> None:
    """Scale the acts' word counts to sum to ``budget`` (floor MIN_ACT_WORDS each)."""
    budget = max(budget, MIN_ACT_WORDS * len(acts))
    total = sum(a.words for a in acts) or 1
    scaled = [max(MIN_ACT_WORDS, round(a.words * budget / total)) for a in acts]
    # Put the rounding remainder on the largest act so the sum is exact
    # whenever the floor allows it.
    drift = budget - sum(scaled)
    largest = max(range(len(scaled)), key=lambda i: scaled[i])
    scaled[largest] = max(MIN_ACT_WORDS, scaled[largest] + drift)
    for act, words in zip(acts, scaled):
        act.words = words


def _text(value: Any) -> str:
    return str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""


def _int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _text(value).lower()).strip("_")[:40]
