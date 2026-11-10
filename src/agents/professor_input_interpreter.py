"""Turn a professor's free-form words into structured instructions.

Two modes (Parts F and G of the design):

- ``interpret_brief``: the brief given with the PDF becomes
  ``ProfessorDirectives`` (scope, depth, analogies, notation, length, notes).
- ``interpret_feedback``: feedback on a finished video becomes ``EditOp``s
  anchored to beats ("at 2:13 the label overlaps" -> the beat playing at
  2:13), classified by what they invalidate (content, visual, pacing, audio,
  presenter).

Both use one schema-constrained LLM call when a key is configured, and a
deterministic rule-based reading otherwise. Nothing is dropped: brief text
no rule understands is kept verbatim as notes, and feedback that cannot be
anchored or understood is returned as notes for the professor to rephrase.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.script_editor import LLM, default_llm
from src.schemas.directives import (
    ConceptDirective, GlobalDirective, NotationDirective, ProfessorDirectives, ScopeDirective,
)
from src.schemas.edit_ops import (
    OP_CLASSES, EditOp, Expand, FixVisual, Pace, Pronounce, Remove, ReplaceAnalogy,
    ReviseBeat, parse_ops,
)
from src.schemas.video_script import VideoScript

logger = logging.getLogger(__name__)

_TIMESTAMP = re.compile(r"\b(\d{1,2}):(\d{2})\b")


@dataclass
class TimelineBeat:
    beat_id: str
    start: float
    end: float


@dataclass
class TimelineScene:
    scene_id: str
    start: float
    end: float
    beats: List[TimelineBeat] = field(default_factory=list)


@dataclass
class FeedbackPlan:
    ops: List[EditOp] = field(default_factory=list)
    items: List[Dict[str, Any]] = field(default_factory=list)  # per op: class, anchor, source text
    notes: List[str] = field(default_factory=list)              # not understood; ask again

    @property
    def classes(self) -> List[str]:
        return sorted({OP_CLASSES.get(op.op, "content") for op in self.ops})


def build_timeline(storyboard, delivery_report: Optional[Dict[str, Any]] = None) -> List[TimelineScene]:
    """Where each scene and beat plays in the final video.

    Uses the delivery report's measured scene windows when available, else
    narration lengths. Beat windows come from the timing track, else from
    word counts.
    """
    measured = {
        s["scene_id"]: (s.get("start"), s.get("end"))
        for s in (delivery_report or {}).get("scenes", [])
        if s.get("start") is not None
    }
    timeline: List[TimelineScene] = []
    clock = 0.0
    for scene in getattr(storyboard, "scenes", []) or []:
        start, end = measured.get(scene.scene_id, (None, None))
        if start is None:
            length = float(scene.audio_duration_seconds or scene.duration_estimate or 0.0)
            start, end = clock, clock + length
        clock = end
        beats: List[TimelineBeat] = []
        timings = getattr(scene, "beat_timings", None) or {}
        source = list(getattr(scene, "source_beats", None) or [])
        if timings:
            for beat_id, (b_start, b_end) in sorted(timings.items(), key=lambda kv: kv[1][0]):
                beats.append(TimelineBeat(beat_id, start + float(b_start), start + float(b_end)))
        elif source:
            words = [max(1, len(b.narration.split())) for b in source]
            total, cursor = float(sum(words)), start
            for beat, count in zip(source, words):
                span = (end - start) * count / total
                beats.append(TimelineBeat(beat.beat_id, cursor, cursor + span))
                cursor += span
        timeline.append(TimelineScene(scene.scene_id, start, end, beats))
    return timeline


def beat_at(timeline: List[TimelineScene], seconds: float) -> Optional[str]:
    for scene in timeline:
        if scene.start <= seconds < scene.end or (scene is timeline[-1] and seconds >= scene.start):
            for beat in scene.beats:
                if beat.start <= seconds < beat.end:
                    return beat.beat_id
            return scene.beats[-1].beat_id if scene.beats else None
    return None


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


_QUOTE = "[\"'“”‘’]?"


def _prefers(brief: str, first: str, second: str) -> bool:
    """The brief says to use ``first`` rather than / instead of ``second``."""
    pattern = (
        rf"{_QUOTE}{re.escape(first)}{_QUOTE}\s*,?\s+(?:rather than|instead of|not)\s+"
        rf"{_QUOTE}{re.escape(second)}{_QUOTE}"
    )
    return re.search(pattern, brief or "", re.IGNORECASE) is not None


def _reconcile_terms(directives: ProfessorDirectives, brief: str) -> None:
    """Fix the two ways an interpreter garbles vocabulary preferences.

    1. A terminology entry {avoid: preferred} written backwards: flipped when
       the brief's own words ("A rather than B") or a notation directive
       (use A instead of B) say the opposite.
    2. A term listed as a rejected analogy: dropped, since vocabulary and
       notation are not analogies.
    """
    terminology = {}
    for avoid, preferred in (directives.global_.terminology or {}).items():
        reversed_by_notation = any(
            (n.use or "").lower() == avoid.lower() and (n.instead_of or "").lower() == preferred.lower()
            for n in directives.notation
        )
        reversed_by_brief = _prefers(brief, avoid, preferred) and not _prefers(brief, preferred, avoid)
        if reversed_by_notation or reversed_by_brief:
            logger.info("Brief terminology was reversed; using '%s' rather than '%s'.", avoid, preferred)
            avoid, preferred = preferred, avoid
        terminology[avoid] = preferred
    directives.global_.terminology = terminology

    terms = {t.lower() for pair in terminology.items() for t in pair}
    terms |= {t.lower() for n in directives.notation for t in (n.use, n.instead_of) if t}
    for concept in directives.concepts:
        concept.avoid_analogies = [a for a in concept.avoid_analogies if a.lower() not in terms]

    # Run-wide settings override the command line, so an interpreter must not
    # invent them: keep one only if the brief actually talks about it.
    if directives.global_.verbosity and not _VERBOSITY_CUE.search(brief or ""):
        logger.info("Ignoring verbosity '%s': the brief does not mention it.", directives.global_.verbosity)
        directives.global_.verbosity = None
    if directives.global_.audience and not _AUDIENCE_CUE.search(brief or ""):
        logger.info("Ignoring audience '%s': the brief does not mention it.", directives.global_.audience)
        directives.global_.audience = None


_VERBOSITY_CUE = re.compile(
    r"(?i)\b(verbos\w*|concise\w*|terse|succinct\w*|wordy|detail\w*|thorough\w*|in depth|"
    r"brief(?:ly)?\s+narration|short narration|fewer words|more words|elaborat\w*)\b"
)
_AUDIENCE_CUE = re.compile(
    r"(?i)\b(audience|students?|undergrad\w*|graduate\w*|high[- ]school\w*|freshm[ae]n|"
    r"first[- ]year|second[- ]year|\d(?:st|nd|rd|th)[- ]year|phd|masters?|beginners?|experts?)\b"
)


class ProfessorInputInterpreter:
    def __init__(self, llm: Optional[LLM] = None, use_default_llm: bool = True):
        self.llm = llm if llm is not None else (default_llm() if use_default_llm else None)

    # ── brief (pre-generation) ──────────────────────────────────────────

    def interpret_brief(self, brief: str) -> ProfessorDirectives:
        brief = (brief or "").strip()
        if not brief:
            return ProfessorDirectives()
        directives = None
        if self.llm is not None:
            try:
                data = self.llm(self._brief_prompt(brief))
                directives = ProfessorDirectives.model_validate(data)
            except Exception as exc:
                logger.warning("Brief interpretation by LLM failed (%s); using rules.", exc)
        if directives is None:
            directives = self._brief_rules(brief)
        _reconcile_terms(directives, brief)
        directives.source_brief = brief
        if directives.is_empty():
            directives.notes = [brief]  # never drop the professor's words
        return directives

    @staticmethod
    def _brief_prompt(brief: str) -> str:
        return (
            "A professor wrote this brief to steer an explainer video generated from their "
            "lecture notes. Compile it into JSON with exactly these keys:\n"
            '{"concepts": [{"concept": "...", "depth": "skip|brief|standard|deep|null", '
            '"analogy_brief": "their own analogy, verbatim", "avoid_analogies": ["..."], "notes": "..."}],\n'
            ' "scope": {"skip_sections": ["4.3"], "skip_pages": [7], "only_pages": []},\n'
            ' "notation": [{"use": "j", "instead_of": "i", "note": ""}],\n'
            ' "global": {"target_minutes": 10, "audience": "high-school|undergraduate|graduate|null", '
            '"verbosity": "low|medium|high|null", "tone": "", '
            '"terminology": {"term to avoid": "term to use instead"}},\n'
            ' "presenter": {"bridges": "on|off|null", "layout": "full|pip|null", "notes": ""},\n'
            ' "notes": ["anything that fits no field, verbatim"]}\n'
            "Rules: terminology maps the word the professor does NOT want to the word they DO "
            'want ("Say eigenvector rather than characteristic vector" -> '
            '{"characteristic vector": "eigenvector"}). avoid_analogies lists rejected analogies '
            'or metaphors (e.g. "clock", "pizza"), never vocabulary or notation.\n'
            "Use null for anything the brief does not say. Never invent directives. Copy any "
            "sentence you cannot map into notes word for word.\n\nBRIEF:\n" + brief
        )

    def _brief_rules(self, brief: str) -> ProfessorDirectives:
        scope, notation = ScopeDirective(), []
        concepts: List[ConceptDirective] = []
        global_ = GlobalDirective()
        notes: List[str] = []
        for sentence in _sentences(brief):
            low = sentence.lower()
            matched = False
            for label in re.findall(r"\bskip\s+(?:section\s+|§\s*)?(\d+(?:\.\d+)+|\d+)\b", low):
                scope.skip_sections.append(label)
                matched = True
            pages = re.search(r"\bonly\s+pages?\s+(\d+)\s*(?:-|–|to)\s*(\d+)", low)
            if pages:
                scope.only_pages = list(range(int(pages.group(1)), int(pages.group(2)) + 1))
                matched = True
            minutes = re.search(r"\b(?:under|at most|max(?:imum)?|no more than)\s+(\d+(?:\.\d+)?)\s*min", low)
            if minutes:
                global_.target_minutes = float(minutes.group(1))
                matched = True
            for use, instead in re.findall(r"\b(?:i\s+)?(?:write|use)\s+([a-z\\]+)\s*,?\s+not\s+([a-z\\]+)\b", sentence, re.I):
                if len(use) <= 3 and len(instead) <= 3:  # symbols, not whole words
                    notation.append(NotationDirective(use=use, instead_of=instead))
                    matched = True
            analogy = re.search(
                r"\buse\s+the\s+(.+?)\s+(?:picture|analogy|metaphor|model)\s*,?\s+not\s+the\s+(.+?)\s+(?:analogy|picture|metaphor|model)",
                sentence, re.I,
            )
            if analogy:
                concepts.append(ConceptDirective(
                    concept="", analogy_brief=f"the {analogy.group(1)} picture",
                    avoid_analogies=[analogy.group(2).strip()],
                ))
                matched = True
            deeper = re.search(r"\b(?:spend more time on|go deeper (?:on|into)|more depth on|emphasi[sz]e)\s+(.+?)[.,;]?$", sentence, re.I)
            if deeper:
                concepts.append(ConceptDirective(concept=deeper.group(1).strip(), depth="deep"))
                matched = True
            briefly = re.search(r"\b(?:briefly|quickly)\s+(?:cover|mention|touch on)\s+(.+?)[.,;]?$", sentence, re.I)
            if briefly:
                concepts.append(ConceptDirective(concept=briefly.group(1).strip(), depth="brief"))
                matched = True
            if re.search(r"\b(\d)(?:st|nd|rd|th)[- ]year\b|\bundergrad", low):
                global_.audience = "undergraduate"
                notes.append(sentence)  # the rest of the sentence still informs the script
                matched = True
            elif re.search(r"\bgraduate students\b|\bphd\b|\bmasters?\b", low):
                global_.audience = "graduate"
                notes.append(sentence)
                matched = True
            elif re.search(r"\bhigh[- ]school\b", low):
                global_.audience = "high-school"
                notes.append(sentence)
                matched = True
            if not matched:
                notes.append(sentence)
        return ProfessorDirectives(
            concepts=concepts, scope=scope, notation=notation,
            global_=global_, notes=notes,
        )

    # ── feedback (post-output) ──────────────────────────────────────────

    def interpret_feedback(
        self,
        feedback: str,
        script: VideoScript,
        timeline: List[TimelineScene],
    ) -> FeedbackPlan:
        feedback = (feedback or "").strip()
        if not feedback:
            return FeedbackPlan()
        anchors = {
            m.group(0): beat_at(timeline, int(m.group(1)) * 60 + int(m.group(2)))
            for m in _TIMESTAMP.finditer(feedback)
        }
        if self.llm is not None:
            try:
                return self._feedback_llm(feedback, script, timeline, anchors)
            except Exception as exc:
                logger.warning("Feedback interpretation by LLM failed (%s); using rules.", exc)
        return self._feedback_rules(feedback, script, anchors)

    def _feedback_llm(self, feedback, script, timeline, anchors) -> FeedbackPlan:
        starts = {b.beat_id: b.start for scene in timeline for b in scene.beats}
        listing = "\n".join(
            f"- {b.beat_id} (at {int(starts.get(b.beat_id, 0)) // 60}:{int(starts.get(b.beat_id, 0)) % 60:02d}, "
            f"role={b.role}{', analogy=' + b.analogy_id if b.analogy_id else ''}"
            f"{', locked' if b.locked else ''}): {b.narration}"
            for b in script.beats
        )
        analogies = "\n".join(f"- {a.analogy_id}: {a.label}" for a in script.analogies) or "(none)"
        anchor_text = "\n".join(f"- {ts} -> {beat}" for ts, beat in anchors.items() if beat) or "(none)"
        prompt = (
            "Compile a professor's feedback on a generated lecture video into edit ops.\n"
            "Op types (JSON objects):\n"
            '- {"op": "rewrite_beat", "beat_id", "text"}  exact replacement text the professor dictated\n'
            '- {"op": "revise_beat", "beat_id", "instruction"}  change one beat\'s narration\n'
            '- {"op": "replace_analogy", "analogy_id", "brief", "mapping": {}}\n'
            '- {"op": "add_analogy", "concept", "brief", "after_beat"}\n'
            '- {"op": "remove", "beat_id"} or {"op": "remove", "analogy_id"}\n'
            '- {"op": "expand", "target": beat_id or concept, "levels": 1, "focus": "derivation|example|intuition"}\n'
            '- {"op": "condense", "beat_ids": [...]}\n'
            '- {"op": "pace", "beat_id", "pause_seconds"}  "too fast here"\n'
            '- {"op": "pronounce", "word", "say_as"}  mispronunciations\n'
            '- {"op": "fix_visual", "beat_id", "instruction"}  on-screen problems; narration unchanged\n'
            '- {"op": "set_density", "scope": "global|segment|concept", "target", "level": "essential|standard|rich", "analogies": "on|off"}\n'
            '- {"op": "presenter", "action": "insert|remove|edit|approve", "beat_id", "text", "position"}\n'
            "Use the smallest ops that do what the professor asked. Anchor timestamps to beats "
            "with the table below. Never revise locked beats; use rewrite_beat only with "
            "the professor's own words.\n"
            f"TIMESTAMPS:\n{anchor_text}\nANALOGIES:\n{analogies}\nSCRIPT:\n{listing}\n\n"
            f"FEEDBACK:\n{feedback}\n\n"
            'Respond as {"ops": [...], "not_understood": ["feedback the ops do not cover, verbatim"]}'
        )
        data = self.llm(prompt)
        plan = FeedbackPlan(notes=[str(n) for n in data.get("not_understood") or []])
        for raw in data.get("ops") or []:
            try:
                op = parse_ops([raw])[0]
            except Exception as exc:
                plan.notes.append(f"Could not use an interpreted edit ({exc}): {json.dumps(raw)[:200]}")
                continue
            plan.ops.append(op)
            plan.items.append({
                "op": op.op, "class": OP_CLASSES.get(op.op, "content"),
                "anchor": getattr(op, "beat_id", None) or getattr(op, "analogy_id", None),
            })
        return plan

    def _feedback_rules(self, feedback, script, anchors) -> FeedbackPlan:
        plan = FeedbackPlan()

        def add(op: EditOp, text: str) -> None:
            plan.ops.append(op)
            plan.items.append({"op": op.op, "class": OP_CLASSES.get(op.op, "content"),
                               "anchor": getattr(op, "beat_id", None), "text": text})

        def find_analogy(words: str) -> Optional[str]:
            words = words.lower().strip()
            for analogy in script.analogies:
                if words and (words in analogy.label.lower() or words in analogy.analogy_id):
                    return analogy.analogy_id
            return None

        for sentence in _sentences(feedback):
            low = sentence.lower()
            stamp = _TIMESTAMP.search(sentence)
            beat = anchors.get(stamp.group(0)) if stamp else None
            said = re.search(
                r"['\"]?([A-Za-z][\w-]*)['\"]?\s+(?:is\s+|should be\s+)?pronounced\s+(?:like\s+|as\s+)?['\"]?([A-Za-z][\w -]*?)['\"]?[.!]?$",
                sentence,
            )
            replace = re.search(r"replace\s+(?:the\s+)?(.+?)\s+analogy\s+with\s+(.+)", sentence, re.I)
            drop = re.search(r"\b(?:cut|remove|drop|delete)\s+(?:the\s+)?(.+?)\s+analogy\b", sentence, re.I)
            if said:
                add(Pronounce(word=said.group(1), say_as=said.group(2).strip()), sentence)
            elif replace and find_analogy(replace.group(1)):
                add(ReplaceAnalogy(analogy_id=find_analogy(replace.group(1)), brief=replace.group(2).strip()), sentence)
            elif drop and find_analogy(drop.group(1)):
                add(Remove(analogy_id=find_analogy(drop.group(1))), sentence)
            elif beat and re.search(r"\b(?:cut|remove|drop|delete)\s+(?:this|that|it)\b", low):
                add(Remove(beat_id=beat), sentence)
            elif beat and re.search(r"too fast|slow (?:it )?down|more time|let it breathe|pause", low):
                add(Pace(beat_id=beat, pause_seconds=1.5), sentence)
            elif beat and re.search(
                r"overlap|label|on screen|onscreen|visual|colou?r|font|arrow|graph|diagram|animation|show the|shows the|move|blank|clipped|off screen",
                low,
            ):
                add(FixVisual(beat_id=beat, instruction=sentence), sentence)
            elif re.search(r"go deeper|more depth|explain (?:this )?more|more detail|elaborate", low) and beat:
                add(Expand(target=beat, levels=1, focus="intuition"), sentence)
            elif beat:
                add(ReviseBeat(beat_id=beat, instruction=sentence), sentence)
            else:
                plan.notes.append(sentence)
        return plan
