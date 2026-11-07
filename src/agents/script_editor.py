"""Apply professor edit ops to a lesson script (Part C of the design).

``ScriptEditorAgent.apply(script, ops)`` returns the edited script plus a
diff, the rejected ops with reasons, professor omissions for the coverage
audit, and visual fixes for the revision run.

Deterministic ops (rewrite, remove, pace, pronounce, fix_visual, lowering
density, presenter edits) never call an LLM. Generative ops (revise,
replace/add analogy, expand, condense, seam repair) need one; without it
they are rejected with a clear reason instead of silently degraded.

Invariants checked after every apply:
- locked beats the ops did not explicitly target are byte-identical;
- no remaining beat depends on a concept whose introducing beat was removed
  (the remove is rejected and the error names the dependent beats).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from src.schemas.edit_ops import (
    AddAnalogy, Condense, EditOp, Expand, FixVisual, Pace, PresenterOp, Pronounce,
    Remove, ReplaceAnalogy, ReviseBeat, RewriteBeat, SetDensity,
)
from src.schemas.video_script import MAX_BEAT_DEPTH, Analogy, Beat, VideoScript

logger = logging.getLogger(__name__)

EDITOR_MODEL = "gpt-5.4-mini"
LLM = Callable[[str], Dict[str, Any]]

# Beats kept at each density level (Part D): essential keeps the argument.
DENSITY_MAX_DEPTH = {"essential": 0, "standard": 1, "rich": MAX_BEAT_DEPTH}

_STYLE = (
    "Write like a friendly, intuitive 3Blue1Brown-style narrator. Each beat is "
    "30-60 words of spoken narration plus a one-sentence visual_hint describing "
    "what is on screen. Return JSON only."
)


class EditError(Exception):
    """An op that cannot be applied; the message is shown to the professor."""


def default_llm() -> Optional[LLM]:
    """JSON-mode OpenAI call, or None when no key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    import openai

    client = openai.OpenAI(max_retries=4)

    def call(prompt: str) -> Dict[str, Any]:
        response = client.chat.completions.create(
            model=EDITOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    return call


@dataclass
class EditResult:
    script: VideoScript
    applied: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    omissions: List[Dict[str, Any]] = field(default_factory=list)
    visual_fixes: Dict[str, str] = field(default_factory=dict)
    diff: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def quoted_phrases(text: str) -> List[str]:
    return [p.strip() for p in re.findall(r'"([^"]+)"', text or "") if p.strip()]


def concept_words(concept: str) -> str:
    return concept.replace("_", " ").strip().lower()


class ScriptEditorAgent:
    def __init__(self, llm: Optional[LLM] = None, use_default_llm: bool = True):
        self.llm = llm if llm is not None else (default_llm() if use_default_llm else None)

    # ── public API ──────────────────────────────────────────────────────

    def apply(self, script: VideoScript, ops: List[EditOp]) -> EditResult:
        original = script.model_copy(deep=True)
        self._beats: List[Beat] = [b.model_copy(deep=True) for b in script.beats]
        self._analogies: Dict[str, Analogy] = {a.analogy_id: a.model_copy(deep=True) for a in script.analogies}
        self._pronunciations = dict(script.pronunciations)
        self._targeted: Set[str] = set()   # locked beats an op explicitly addressed
        self._next_id = self._max_beat_number(script) + 1
        result = EditResult(script=original)

        for op in ops:
            handler = getattr(self, f"_op_{op.op}", None)
            snapshot = (
                [b.model_copy(deep=True) for b in self._beats],
                {k: v.model_copy(deep=True) for k, v in self._analogies.items()},
                dict(self._pronunciations),
            )
            try:
                if handler is None:
                    raise EditError("this kind of edit is not supported yet")
                handler(op, result)
                result.applied.append(op.model_dump())
            except EditError as exc:
                self._beats, self._analogies, self._pronunciations = snapshot
                result.errors.append(f"{op.op}: {exc}")

        self._enforce_locked(original, result)
        used = {b.analogy_id for b in self._beats if b.analogy_id}
        result.script = VideoScript(
            title=original.title,
            beats=self._beats or original.beats,
            analogies=[a for a in self._analogies.values() if a.analogy_id in used],
            pronunciations=self._pronunciations,
        )
        result.diff = self.diff(original, result.script, result)
        return result

    @staticmethod
    def diff(before: VideoScript, after: VideoScript, result: Optional[EditResult] = None) -> List[str]:
        """Human-readable change list, in script order."""
        old = {b.beat_id: b for b in before.beats}
        new = {b.beat_id: b for b in after.beats}

        def short(text: str, n: int = 90) -> str:
            text = " ".join((text or "").split())
            return text if len(text) <= n else text[: n - 3] + "..."

        lines: List[str] = []
        for beat in after.beats:
            prior = old.get(beat.beat_id)
            if prior is None:
                lines.append(f"+ {beat.beat_id} [{beat.role}] {short(beat.narration)}")
            elif prior.narration != beat.narration:
                lines.append(f"~ {beat.beat_id} [{beat.role}] {short(prior.narration)}")
                lines.append(f"    -> {short(beat.narration)}")
            elif prior.pause_after != beat.pause_after:
                lines.append(f"~ {beat.beat_id} pause after: {prior.pause_after:g}s -> {beat.pause_after:g}s")
            elif prior.locked != beat.locked:
                lines.append(f"~ {beat.beat_id} {'locked (approved)' if beat.locked else 'unlocked'}")
        for beat in before.beats:
            if beat.beat_id not in new:
                lines.append(f"- {beat.beat_id} [{beat.role}] {short(beat.narration)}")
        for word, say in after.pronunciations.items():
            if before.pronunciations.get(word) != say:
                lines.append(f"~ pronounce '{word}' as '{say}'")
        for beat_id, instruction in (result.visual_fixes if result else {}).items():
            lines.append(f"~ {beat_id} visual: {short(instruction)}")
        return lines

    # ── deterministic ops ───────────────────────────────────────────────

    def _op_rewrite_beat(self, op: RewriteBeat, result: EditResult) -> None:
        beat = self._beat(op.beat_id)
        beat.narration = op.text.strip()
        beat.author, beat.locked = "professor", True
        self._targeted.add(beat.beat_id)

    def _op_pace(self, op: Pace, result: EditResult) -> None:
        self._beat(op.beat_id).pause_after = op.pause_seconds

    def _op_pronounce(self, op: Pronounce, result: EditResult) -> None:
        self._pronunciations[op.word.strip()] = op.say_as.strip()

    def _op_fix_visual(self, op: FixVisual, result: EditResult) -> None:
        self._beat(op.beat_id)
        result.visual_fixes[op.beat_id] = op.instruction.strip()

    def _op_remove(self, op: Remove, result: EditResult) -> None:
        if op.analogy_id:
            doomed = [b for b in self._beats if b.analogy_id == op.analogy_id]
            if not doomed:
                raise EditError(f"no beats use analogy {op.analogy_id}")
        else:
            doomed = [self._beat(op.beat_id)]
        self._remove_beats(doomed, result, reason="professor")
        if op.analogy_id:
            self._analogies.pop(op.analogy_id, None)
        self._targeted.update(b.beat_id for b in doomed)

    def _op_set_density(self, op: SetDensity, result: EditResult) -> None:
        in_scope = [b for b in self._beats if self._in_scope(b, op)]
        if not in_scope:
            raise EditError(f"nothing in scope {op.scope} {op.target or ''}".strip())
        max_depth = DENSITY_MAX_DEPTH[op.level]
        if op.level == "rich":
            # Raising density compiles to expand ops on the in-scope core beats.
            anchors = [b for b in in_scope if b.role == "core" and b.depth == 0]
            if not anchors:
                raise EditError("no core beats in scope to expand")
            for beat in anchors:
                self._op_expand(Expand(target=beat.beat_id, levels=1, focus="example"), result)
        doomed = [b for b in in_scope if b.depth > max_depth and b.role != "bridge"]
        if op.analogies == "off":
            analogy_ids = {b.analogy_id for b in in_scope if b.analogy_id}
            doomed += [
                b for b in self._beats
                if (b.role == "analogy" and b in in_scope) or (b.analogy_id and b.analogy_id in analogy_ids)
            ]
        unique = {id(b): b for b in doomed if not b.locked}
        locked = [b.beat_id for b in doomed if b.locked]
        if locked:
            result.warnings.append(
                f"set_density kept professor-locked beat(s) {', '.join(locked)}"
            )
        if unique:
            self._remove_beats(list(unique.values()), result, reason="professor: density")
            for analogy_id in [a for a in self._analogies if not any(b.analogy_id == a for b in self._beats)]:
                self._analogies.pop(analogy_id, None)

    def _op_presenter(self, op: PresenterOp, result: EditResult) -> None:
        if op.action == "insert":
            anchor = self._beat(op.beat_id) if op.beat_id else None
            if not op.text:
                raise EditError("inserting a bridge needs its text")
            bridge = Beat(
                beat_id=self._new_id(),
                narration=op.text.strip(),
                visual_hint="Presenter on camera.",
                source_segment_id=anchor.source_segment_id if anchor else "",
                role="bridge", author="professor", locked=True,
            )
            index = self._beats.index(anchor) + (1 if op.position == "after" else 0) if anchor else 0
            self._beats.insert(index, bridge)
            return
        beat = self._beat(op.beat_id)
        if beat.role != "bridge":
            raise EditError(f"{beat.beat_id} is not a presenter bridge")
        self._targeted.add(beat.beat_id)
        if op.action == "remove":
            self._beats.remove(beat)
        elif op.action == "edit":
            if not op.text:
                raise EditError("editing a bridge needs the new text")
            beat.narration, beat.author, beat.locked = op.text.strip(), "professor", True
        elif op.action == "approve":
            beat.locked = True

    # ── generative ops ──────────────────────────────────────────────────

    def _op_revise_beat(self, op: ReviseBeat, result: EditResult) -> None:
        beat = self._beat(op.beat_id)
        if beat.locked:
            raise EditError(f"{beat.beat_id} is professor-approved text; use rewrite_beat to change it")
        data = self._ask(
            f"{_STYLE}\n\nRewrite ONE beat of a lecture script.\n"
            f"{self._context(beat)}\nCURRENT BEAT ({beat.beat_id}):\n{beat.narration}\n\n"
            f"INSTRUCTION FROM THE PROFESSOR:\n{op.instruction}\n\n"
            "Keep everything the instruction does not ask to change. Respond as "
            '{"narration": "...", "visual_hint": "..."}'
        )
        beat.narration = self._text(data, "narration")
        beat.visual_hint = str(data.get("visual_hint") or beat.visual_hint)

    def _op_replace_analogy(self, op: ReplaceAnalogy, result: EditResult) -> None:
        beats = [b for b in self._beats if b.analogy_id == op.analogy_id]
        if not beats:
            raise EditError(f"no beats use analogy {op.analogy_id}")
        locked = [b.beat_id for b in beats if b.locked]
        if locked:
            raise EditError(f"analogy beats {', '.join(locked)} are professor-approved; rewrite them directly")
        phrases = quoted_phrases(op.brief)
        listing = "\n".join(f"- {b.beat_id}: {b.narration}" for b in beats)
        prompt = (
            f"{_STYLE}\n\nReplace an analogy everywhere it appears in a lecture script. "
            "Rewrite the introduction beat and every callback consistently, building on "
            "the professor's brief. Keep each beat's teaching point.\n\n"
            f"OLD ANALOGY: {self._analogies.get(op.analogy_id, Analogy(analogy_id=op.analogy_id)).label}\n"
            f"BEATS USING IT:\n{listing}\n\nPROFESSOR'S BRIEF:\n{op.brief}\n"
            + (f"MAPPING: {json.dumps(op.mapping)}\n" if op.mapping else "")
            + (f"Use these exact phrases verbatim: {json.dumps(phrases)}\n" if phrases else "")
            + 'Respond as {"label": "...", "beats": [{"beat_id": "...", "narration": "...", "visual_hint": "..."}]}'
        )
        for attempt in range(2):
            data = self._ask(prompt)
            rewritten = {str(b.get("beat_id")): b for b in data.get("beats") or [] if isinstance(b, dict)}
            missing = [b.beat_id for b in beats if b.beat_id not in rewritten]
            text = " ".join(str(r.get("narration", "")) for r in rewritten.values())
            lost = [p for p in phrases if p.lower() not in text.lower()]
            if not missing and not lost:
                break
            prompt += (
                f"\n\nYour last answer {'skipped beats ' + ', '.join(missing) if missing else ''}"
                f"{' and ' if missing and lost else ''}"
                f"{'dropped the phrases ' + json.dumps(lost) if lost else ''}. Fix that."
            )
        if missing:
            raise EditError(f"the rewrite skipped beat(s) {', '.join(missing)}")
        if lost:
            result.warnings.append(f"replace_analogy: phrases not kept verbatim: {lost}")
        for beat in beats:
            beat.narration = str(rewritten[beat.beat_id].get("narration") or beat.narration).strip()
            beat.visual_hint = str(rewritten[beat.beat_id].get("visual_hint") or beat.visual_hint)
        analogy = self._analogies.get(op.analogy_id) or Analogy(analogy_id=op.analogy_id)
        analogy.label = str(data.get("label") or analogy.label or op.brief[:60])
        analogy.mapping = dict(op.mapping) or analogy.mapping
        analogy.origin, analogy.professor_brief = "professor", op.brief
        self._analogies[op.analogy_id] = analogy

    def _op_add_analogy(self, op: AddAnalogy, result: EditResult) -> None:
        anchor = self._anchor(op.after_beat or op.concept)
        data = self._ask(
            f"{_STYLE}\n\nWrite ONE new beat that introduces the professor's analogy for "
            f"'{concept_words(op.concept)}', placed right after this beat.\n{self._context(anchor)}\n"
            f"AFTER BEAT ({anchor.beat_id}): {anchor.narration}\n\nPROFESSOR'S BRIEF:\n{op.brief}\n"
            + (f"Use these exact phrases verbatim: {json.dumps(quoted_phrases(op.brief))}\n" if quoted_phrases(op.brief) else "")
            + 'Respond as {"label": "...", "narration": "...", "visual_hint": "...", "mapping": {"analogy term": "concept"}}'
        )
        label = str(data.get("label") or op.brief[:40])
        analogy_id = self._unique_analogy_id(label)
        beat = Beat(
            beat_id=self._new_id(),
            narration=self._text(data, "narration"),
            visual_hint=str(data.get("visual_hint") or op.brief),
            references_prior_beats=[anchor.beat_id],
            source_segment_id=anchor.source_segment_id,
            role="analogy", analogy_id=analogy_id, depth=min(anchor.depth + 1, MAX_BEAT_DEPTH),
        )
        mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else {}
        self._analogies[analogy_id] = Analogy(
            analogy_id=analogy_id, label=label, target_concepts=[op.concept],
            mapping={str(k): str(v) for k, v in mapping.items()},
            origin="professor", professor_brief=op.brief,
        )
        self._beats.insert(self._beats.index(anchor) + 1, beat)

    def _op_expand(self, op: Expand, result: EditResult) -> None:
        anchor = self._anchor(op.target)
        role = op.focus if op.focus in ("derivation", "example", "intuition") else "core"
        data = self._ask(
            f"{_STYLE}\n\nGo deeper: write {op.levels} new beat(s) that continue right after "
            f"this beat, adding {op.focus}. Do not repeat what it already says.\n"
            f"{self._context(anchor)}\nTHE BEAT ({anchor.beat_id}): {anchor.narration}\n"
            'Respond as {"beats": [{"narration": "...", "visual_hint": "..."}]}'
        )
        raw = [b for b in data.get("beats") or [] if isinstance(b, dict) and b.get("narration")]
        if not raw:
            raise EditError("the expansion came back empty")
        at = self._beats.index(anchor) + 1
        for offset, item in enumerate(raw[: op.levels]):
            self._beats.insert(at + offset, Beat(
                beat_id=self._new_id(),
                narration=str(item["narration"]).strip(),
                visual_hint=str(item.get("visual_hint") or ""),
                references_prior_beats=[anchor.beat_id],
                source_segment_id=anchor.source_segment_id,
                role=role, depth=min(anchor.depth + 1, MAX_BEAT_DEPTH),
            ))

    def _op_condense(self, op: Condense, result: EditResult) -> None:
        beats = [self._beat(b) for b in op.beat_ids]
        locked = [b.beat_id for b in beats if b.locked]
        if locked:
            raise EditError(f"beat(s) {', '.join(locked)} are professor-approved")
        listing = "\n".join(f"- {b.beat_id}: {b.narration}" for b in beats)
        data = self._ask(
            f"{_STYLE}\n\nCondense these consecutive beats into fewer, shorter beats "
            f"(at most {len(beats) - 1}) that keep every teaching point.\n{listing}\n"
            'Respond as {"beats": [{"narration": "...", "visual_hint": "..."}]}'
        )
        raw = [b for b in data.get("beats") or [] if isinstance(b, dict) and b.get("narration")]
        if not raw or len(raw) >= len(beats):
            raise EditError("the condensed version is not shorter")
        concepts = [c for b in beats for c in b.introduces_concepts]
        for i, item in enumerate(raw):
            beats[i].narration = str(item["narration"]).strip()
            beats[i].visual_hint = str(item.get("visual_hint") or beats[i].visual_hint)
        beats[0].introduces_concepts = list(dict.fromkeys(concepts))
        for i in range(1, len(raw)):
            beats[i].introduces_concepts = []
        for beat in beats[len(raw):]:
            self._beats.remove(beat)
            self._drop_refs({beat.beat_id}, keep=beats[0].beat_id)

    # ── helpers ─────────────────────────────────────────────────────────

    def _remove_beats(self, doomed: List[Beat], result: EditResult, reason: str) -> None:
        ids = {b.beat_id for b in doomed}
        remaining = [b for b in self._beats if b.beat_id not in ids]
        dependents: Dict[str, List[str]] = {}
        for beat in doomed:
            for concept in beat.introduces_concepts:
                if any(concept in b.introduces_concepts for b in remaining):
                    continue  # still introduced elsewhere
                users = [
                    b.beat_id for b in remaining
                    if self._comes_after(b, beat)
                    and (beat.beat_id in b.references_prior_beats or concept_words(concept) in b.narration.lower())
                ]
                if users:
                    dependents[concept] = users
        if dependents:
            details = "; ".join(f"{c} (used by {', '.join(u)})" for c, u in dependents.items())
            raise EditError(
                f"{', '.join(sorted(ids))} introduce(s) concepts later beats still use: {details}. "
                "Remove or rewrite those beats first."
            )
        neighbours = self._neighbours(doomed)
        for beat in doomed:
            result.omissions.append({
                "beat_id": beat.beat_id, "concepts": list(beat.introduces_concepts),
                "segment_id": beat.source_segment_id, "reason": reason,
            })
        self._beats = remaining
        self._drop_refs(ids)
        self._repair_seam(neighbours, result)

    def _neighbours(self, doomed: List[Beat]):
        indices = sorted(self._beats.index(b) for b in doomed)
        before = self._beats[indices[0] - 1] if indices[0] > 0 else None
        after = self._beats[indices[-1] + 1] if indices[-1] + 1 < len(self._beats) else None
        return before, after

    def _repair_seam(self, neighbours, result: EditResult) -> None:
        """Smooth only the two sentences that now meet across the gap."""
        before, after = neighbours
        if before is None or after is None or before.locked or after.locked:
            return
        if self.llm is None:
            result.warnings.append(
                f"seam between {before.beat_id} and {after.beat_id} not smoothed (no LLM configured)"
            )
            return
        try:
            data = self.llm(
                f"{_STYLE}\n\nA passage was cut from a lecture script; these two beats now meet. "
                "Make the smallest edit (the last sentence of BEFORE and/or the first "
                "sentence of AFTER) so they flow. Change nothing else.\n"
                f"BEFORE: {before.narration}\nAFTER: {after.narration}\n"
                'Respond as {"before": "...", "after": "..."}'
            )
            before.narration = str(data.get("before") or before.narration).strip()
            after.narration = str(data.get("after") or after.narration).strip()
        except Exception as exc:
            result.warnings.append(f"seam repair failed: {exc}")

    def _drop_refs(self, ids: Set[str], keep: Optional[str] = None) -> None:
        for beat in self._beats:
            refs = [r for r in beat.references_prior_beats if r not in ids]
            if keep and len(refs) != len(beat.references_prior_beats) and keep not in refs and keep != beat.beat_id:
                refs.append(keep)
            beat.references_prior_beats = refs

    def _enforce_locked(self, original: VideoScript, result: EditResult) -> None:
        current = {b.beat_id: b for b in self._beats}
        for beat in original.beats:
            if not beat.locked or beat.beat_id in self._targeted:
                continue
            now = current.get(beat.beat_id)
            if now is not None and now.model_dump() != beat.model_dump():
                # Professor text is law: restore it and report the attempt.
                self._beats[self._beats.index(now)] = beat.model_copy(deep=True)
                result.errors.append(f"locked beat {beat.beat_id} was modified by an edit; restored")

    def _in_scope(self, beat: Beat, op: SetDensity) -> bool:
        if op.scope == "global":
            return True
        if op.scope == "segment":
            return beat.source_segment_id == op.target
        concept = (op.target or "").lower()
        return any(c.lower() == concept for c in beat.introduces_concepts) or concept_words(concept) in beat.narration.lower()

    def _ask(self, prompt: str) -> Dict[str, Any]:
        if self.llm is None:
            raise EditError("this edit needs an LLM (set OPENAI_API_KEY)")
        try:
            data = self.llm(prompt)
        except Exception as exc:
            raise EditError(f"the LLM call failed: {exc}") from exc
        if not isinstance(data, dict):
            raise EditError("the LLM returned no JSON object")
        return data

    @staticmethod
    def _text(data: Dict[str, Any], key: str) -> str:
        text = data.get(key)
        if not isinstance(text, str) or not text.strip():
            raise EditError(f"the LLM returned no {key}")
        return text.strip()

    def _context(self, beat: Beat) -> str:
        i = self._beats.index(beat)
        prev_text = self._beats[i - 1].narration if i > 0 else "(start of video)"
        next_text = self._beats[i + 1].narration if i + 1 < len(self._beats) else "(end of video)"
        return f"PREVIOUS BEAT: {prev_text}\nNEXT BEAT: {next_text}\n"

    def _beat(self, beat_id: Optional[str]) -> Beat:
        for beat in self._beats:
            if beat.beat_id == beat_id:
                return beat
        raise EditError(f"no beat {beat_id}")

    def _anchor(self, target: str) -> Beat:
        """A beat id, or the beat that introduces (else mentions) a concept."""
        for beat in self._beats:
            if beat.beat_id == target:
                return beat
        words = concept_words(target)
        for beat in self._beats:
            if any(concept_words(c) == words for c in beat.introduces_concepts):
                return beat
        for beat in reversed(self._beats):
            if words and words in beat.narration.lower():
                return beat
        raise EditError(f"no beat or concept matches '{target}'")

    def _comes_after(self, later: Beat, earlier: Beat) -> bool:
        order = {b.beat_id: i for i, b in enumerate(self._beats)}
        return order.get(later.beat_id, -1) > order.get(earlier.beat_id, -1)

    def _new_id(self) -> str:
        beat_id = f"beat_{self._next_id:03d}"
        self._next_id += 1
        return beat_id

    @staticmethod
    def _max_beat_number(script: VideoScript) -> int:
        numbers = [int(m.group(1)) for b in script.beats if (m := re.search(r"(\d+)$", b.beat_id))]
        return max(numbers, default=0)

    def _unique_analogy_id(self, label: str) -> str:
        base = "analogy_" + (re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "professor")
        analogy_id, n = base, 2
        while analogy_id in self._analogies:
            analogy_id, n = f"{base}_{n}", n + 1
        return analogy_id
