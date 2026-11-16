"""Propose presenter bridges: short on-camera moments between Manim scenes.

Placement policy (Part E): an opening hook, transitions at segment
boundaries, a "why this matters" before a derivation, and a recap; capped at
about one per 90 s of narration and at most 20 % of the runtime. Proposals
are ``role=bridge`` beats with ``locked=False``: nothing is rendered until
the professor approves (locks) the text.
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.agents.script_editor import LLM, default_llm
from src.prompts.narrative import bridge_prompt, bridge_slot_line
from src.schemas.video_script import Beat, VideoScript

logger = logging.getLogger(__name__)

WORDS_PER_SECOND = 3.0
BRIDGE_EVERY_SECONDS = 90.0
MAX_RUNTIME_SHARE = 0.20
TYPICAL_BRIDGE_SECONDS = 12.0


def _concepts(beat: Optional[Beat]) -> str:
    if beat is None or not beat.introduces_concepts:
        return "this idea"
    return " and ".join(c.replace("_", " ") for c in beat.introduces_concepts[:2])


def _seconds(beat: Beat) -> float:
    return len(beat.narration.split()) / WORDS_PER_SECOND


def _run_key(beat: Beat) -> str:
    return (getattr(beat, "act_id", "") or "") or (beat.source_segment_id or "")


def _next_question(beats: List[Beat], i: int, plan) -> str:
    """The viewer question of the act that starts at beat ``i`` ("" if unknown)."""
    if i < len(beats) and beats[i].act_id:
        act = plan.act(beats[i].act_id)
        if act is not None:
            return act.viewer_question
    return ""


class BridgePlannerAgent:
    def __init__(self, llm: Optional[LLM] = None, use_default_llm: bool = True):
        self.llm = llm if llm is not None else (default_llm() if use_default_llm else None)

    def positions(self, beats: List[Beat], planned: bool = False) -> List[Tuple[int, str]]:
        """(insert-before index, kind) pairs that respect the caps."""
        content = [b for b in beats if b.role != "bridge"]
        total = sum(_seconds(b) for b in content)
        if not content or total <= 0:
            return []
        budget = max(1, int(min(
            total / BRIDGE_EVERY_SECONDS + 1,
            total * MAX_RUNTIME_SHARE / TYPICAL_BRIDGE_SECONDS,
        )))
        candidates: List[Tuple[int, str]] = [(0, "opening")]
        for i in range(1, len(beats)):
            if _run_key(beats[i]) != _run_key(beats[i - 1]):
                candidates.append((i, "transition"))
            elif beats[i].role == "derivation":
                candidates.append((i, "why"))
        if not planned:
            # A planned script already ends on its payoff beat; a recap after
            # it would repeat it.
            candidates.append((len(beats), "recap"))

        chosen: List[Tuple[int, str]] = []
        since = BRIDGE_EVERY_SECONDS  # the opening is always allowed
        cursor = 0
        for index, kind in candidates:
            since += sum(_seconds(b) for b in beats[cursor:index])
            cursor = index
            if len(chosen) >= budget:
                break
            if since >= BRIDGE_EVERY_SECONDS:
                chosen.append((index, kind))
                since = 0.0
        return chosen

    def propose(self, script: VideoScript) -> VideoScript:
        beats = list(script.beats)
        plan = getattr(script, "plan", None)
        positions = self.positions(beats, planned=plan is not None)
        if not positions:
            return script
        texts = self._write(beats, positions, plan)
        next_number = max(
            (int(b.beat_id.rsplit("_", 1)[-1]) for b in beats if b.beat_id.rsplit("_", 1)[-1].isdigit()),
            default=0,
        ) + 1
        out: List[Beat] = []
        planned = dict(positions)
        for i in range(len(beats) + 1):
            if i in planned:
                neighbour = beats[i] if i < len(beats) else beats[-1]
                out.append(Beat(
                    beat_id=f"beat_{next_number:03d}",
                    narration=texts[i],
                    visual_hint=f"Presenter on camera ({planned[i]}).",
                    source_segment_id=neighbour.source_segment_id,
                    role="bridge", author="generated", locked=False,
                ))
                next_number += 1
            if i < len(beats):
                out.append(beats[i])
        logger.info("Proposed %d presenter bridge(s); they render once approved.", len(positions))
        return VideoScript(
            title=script.title, beats=out, analogies=script.analogies,
            pronunciations=script.pronunciations, plan=plan,
        )

    def _write(self, beats: List[Beat], positions: List[Tuple[int, str]], plan=None) -> Dict[int, str]:
        texts = {i: self._template(beats, i, kind, plan) for i, kind in positions}
        if plan is None:
            if self.llm is None:
                return texts
            listing = "\n".join(
                f"- slot {i} ({kind}): BEFORE: {beats[i - 1].narration if i > 0 else '(video start)'} | "
                f"AFTER: {beats[i].narration if i < len(beats) else '(video end)'}"
                for i, kind in positions
            )
            self._fill_from_llm(
                "The professor appears on camera between animated scenes of their lecture video. "
                "Write each bridge in the professor's first-person voice, 15-45 words: an opening "
                "hook, a transition that motivates what comes next, a 'why this matters' before "
                "a hard step, or a recap. Never introduce new content. The AFTER text is spoken "
                "right after your bridge, so never say what it says: no formula, definition or "
                "analogy from it. At most, raise the question it answers or say why it matters. "
                "Sound like a person talking: contractions, short sentences, no stock phrases "
                "(let's dive in, here's the key), no em dashes, math said in words.\n"
                f"{listing}\nRespond in JSON as {{\"bridges\": [{{\"slot\": 0, \"text\": \"...\"}}]}}",
                texts, beats,
            )
            return texts

        # The plan's hook was written to be said on camera: used word for word.
        slots = [
            (i, kind) for i, kind in positions
            if not (kind == "opening" and plan.hook and texts[i] == plan.hook)
        ]
        if not slots or self.llm is None:
            return texts
        lines = [
            bridge_slot_line(
                i, kind,
                before=beats[i - 1].narration if i > 0 else "",
                after=beats[i].narration if i < len(beats) else "",
                next_question=_next_question(beats, i, plan),
            )
            for i, kind in slots
        ]
        sent = {i: texts[i] for i, _ in slots}
        self._fill_from_llm(bridge_prompt(lines, plan), sent, beats)
        texts.update(sent)
        return texts

    def _fill_from_llm(self, prompt: str, texts: Dict[int, str], beats: List[Beat]) -> None:
        """Replace templates in ``texts`` with the LLM's bridges, skipping echoes of a neighbour."""
        try:
            data = self.llm(prompt)
            for item in data.get("bridges") or []:
                slot, text = item.get("slot"), str(item.get("text") or "").strip()
                if slot in texts and text:
                    if self._echoes(text, beats, slot):
                        logger.info("Bridge at slot %s repeats its neighbour; using the template.", slot)
                        continue
                    texts[slot] = text
        except Exception as exc:
            logger.warning("Bridge writing by LLM failed (%s); using templates.", exc)

    ECHO_OVERLAP = 0.3

    @classmethod
    def _echoes(cls, text: str, beats: List[Beat], slot: int) -> bool:
        """True when a bridge mostly restates the beat before or after it."""
        from src.utils.narration_flow import overlaps

        neighbours = [beats[j].narration for j in (slot - 1, slot) if 0 <= j < len(beats)]
        return any(overlaps(text, other) >= cls.ECHO_OVERLAP for other in neighbours)

    @staticmethod
    def _template(beats: List[Beat], i: int, kind: str, plan=None) -> str:
        if plan is not None:
            if kind == "opening" and plan.hook:
                return plan.hook
            if kind == "transition":
                question = _next_question(beats, i, plan)
                if question:
                    return question
        after = beats[i] if i < len(beats) else None
        before = beats[i - 1] if i > 0 else None
        if kind == "opening":
            return f"Today we're going to build up {_concepts(after)} from scratch, and see why it matters."
        if kind == "transition":
            return f"Now that we have {_concepts(before)}, here's the natural next question: {_concepts(after)}."
        if kind == "why":
            return f"This next step is the hard part, so let's slow down and see why {_concepts(after)} works."
        covered = sorted({c.replace('_', ' ') for b in beats for c in b.introduces_concepts})[:4]
        return "Let's recap what we built: " + (", ".join(covered) if covered else "the main idea") + "."
