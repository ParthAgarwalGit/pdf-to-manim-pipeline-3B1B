"""Deterministic checks that professor directives were honoured (Part F).

Directives are binding. These checks run at the stage each directive can
first be verified:

- scope (skipped sections/pages): segment selection, before any spend;
- rejected analogies and skipped concepts: the lesson script, before visual
  planning; violations are repaired through the script editor (at most two
  rounds) so only the offending beats are rewritten;
- length: the storyboard;
- notation: the generated scene code.

Every check becomes a ``CheckResult`` in the delivery report; anything
still unmet keeps the run out of "ready".
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.schemas.delivery_report import CheckResult
from src.schemas.directives import ProfessorDirectives
from src.schemas.edit_ops import Remove, ReviseBeat
from src.schemas.video_script import VideoScript

_STOPWORDS = {"the", "a", "an", "of", "for", "analogy", "picture", "metaphor", "model", "with"}


def _mentions(text: str, phrase: str) -> bool:
    """Whole phrase, or its last content word, appears as a word in ``text``."""
    text = (text or "").lower()
    phrase = (phrase or "").lower().strip()
    if not phrase:
        return False
    if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", phrase) if w not in _STOPWORDS]
    return bool(words) and bool(re.search(rf"(?<!\w){re.escape(words[-1])}s?(?!\w)", text))


def script_violations(script: VideoScript, directives: ProfessorDirectives) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for concept in directives.concepts:
        for avoided in concept.avoid_analogies:
            beats = [b.beat_id for b in script.beats if _mentions(b.narration, avoided)]
            if beats:
                replacement = f" Use {concept.analogy_brief} instead." if concept.analogy_brief else ""
                violations.append({
                    "directive": f"Never use the {avoided} analogy",
                    "kind": "avoid_analogy", "beat_ids": beats,
                    "instruction": f"Remove every reference to the {avoided} analogy.{replacement}",
                })
        if concept.depth == "skip" and concept.concept:
            label = concept.concept.lower().replace("_", " ")
            beats = [
                b.beat_id for b in script.beats
                if any(c.replace("_", " ").lower() == label for c in b.introduces_concepts)
                or _mentions(b.narration, label)
            ]
            if beats:
                violations.append({
                    "directive": f"Do not cover {concept.concept}",
                    "kind": "skip_concept", "beat_ids": beats,
                })
    return violations


def enforce_script_directives(
    script: VideoScript,
    directives: ProfessorDirectives,
    editor: Any = None,
    max_rounds: int = 2,
) -> Tuple[VideoScript, List[CheckResult]]:
    """Repair script-level violations via the editor; report what remains."""
    if editor is None:
        from src.agents.script_editor import ScriptEditorAgent
        editor = ScriptEditorAgent()
    initial = script_violations(script, directives)
    for _ in range(max_rounds):
        violations = script_violations(script, directives)
        if not violations:
            break
        ops = []
        for v in violations:
            for beat_id in v["beat_ids"]:
                if v["kind"] == "avoid_analogy":
                    ops.append(ReviseBeat(beat_id=beat_id, instruction=v["instruction"]))
                else:
                    ops.append(Remove(beat_id=beat_id))
        result = editor.apply(script, ops)
        if not result.applied:
            break
        script = result.script
    remaining = script_violations(script, directives)
    checks: List[CheckResult] = []
    remaining_by = {v["directive"]: v for v in remaining}
    for v in initial + [v for v in remaining if v["directive"] not in {i["directive"] for i in initial}]:
        left = remaining_by.get(v["directive"])
        checks.append(CheckResult(
            check="directive", passed=left is None, severity="medium",
            message="" if left is None else f"{v['directive']}: still present in {', '.join(left['beat_ids'])}.",
            measured={"directive": v["directive"], "repaired_beats": v["beat_ids"] if left is None else []},
        ))
    for directive in _avoid_and_skip_directives(directives):
        if directive not in {c.measured.get("directive") for c in checks}:
            checks.append(CheckResult(check="directive", passed=True, measured={"directive": directive}))
    return script, checks


def _avoid_and_skip_directives(directives: ProfessorDirectives) -> List[str]:
    out = []
    for c in directives.concepts:
        out += [f"Never use the {a} analogy" for a in c.avoid_analogies]
        if c.depth == "skip" and c.concept:
            out.append(f"Do not cover {c.concept}")
    return out


def _heading_matches(heading: str, label: str) -> bool:
    return bool(re.search(rf"(?<![\d.]){re.escape(label)}(?![\d])", heading or ""))


def apply_scope(segments: List[Any], directives: Optional[ProfessorDirectives]):
    """Drop segments the professor excluded. Returns (kept, omissions, checks)."""
    if directives is None:
        return segments, [], []
    scope = directives.scope
    kept, omissions = [], []
    for segment in segments:
        pages = set(range(segment.page_start, segment.page_end + 1))
        reason = None
        for label in scope.skip_sections:
            if _heading_matches(segment.heading, label) or _heading_matches(segment.text[:120], label):
                reason = f"professor: skip section {label}"
        if scope.skip_pages and pages <= set(scope.skip_pages):
            reason = reason or "professor: skip pages"
        if scope.only_pages and not pages & set(scope.only_pages):
            reason = reason or "professor: outside the requested pages"
        if reason:
            omissions.append({"segment_id": segment.segment_id, "heading": segment.heading, "reason": reason})
        else:
            kept.append(segment)
    checks = []
    for label in scope.skip_sections:
        found = any(label in o["reason"] for o in omissions)
        checks.append(CheckResult(
            check="directive", passed=found, severity="medium",
            message="" if found else f"Skip section {label}: no section with that label was found in the PDF.",
            measured={"directive": f"Skip section {label}"},
        ))
    if not kept:  # never skip everything: fall back to the full document
        checks.append(CheckResult(
            check="directive", passed=False, severity="high",
            message="The requested scope excluded the whole document; all sections were kept.",
            measured={"directive": "scope"},
        ))
        return segments, [], checks
    return kept, omissions, checks


def length_check(storyboard: Any, target_minutes: Optional[float]) -> Optional[CheckResult]:
    if not target_minutes:
        return None
    seconds = sum(
        float(s.audio_duration_seconds or s.duration_estimate or 0.0)
        for s in getattr(storyboard, "scenes", []) or []
    )
    limit = float(target_minutes) * 60.0 * 1.1
    passed = seconds <= limit
    return CheckResult(
        check="directive", passed=passed, severity="medium",
        message="" if passed else (
            f"Length: about {seconds / 60:.1f} min, over the requested {target_minutes:g} min."
        ),
        measured={"directive": f"Under {target_minutes:g} minutes", "estimated_seconds": round(seconds, 1)},
    )


_TEX_CALL = re.compile(r"\b(?:MathTex|Tex)\s*\(([^)]*)\)", re.S)
_STRING = re.compile(r"""[rRbBfFuU]{0,2}(['"])(.*?)(?<!\\)\1""", re.S)


def notation_check(code_objects: List[Any], directives: Optional[ProfessorDirectives]) -> List[CheckResult]:
    checks: List[CheckResult] = []
    for rule in (directives.notation if directives else []):
        if not rule.instead_of:
            continue
        symbol = re.escape(rule.instead_of)
        offenders = []
        for code_obj in code_objects or []:
            for call in _TEX_CALL.finditer(code_obj.code or ""):
                strings = " ".join(m.group(2) for m in _STRING.finditer(call.group(1)))
                if re.search(rf"(?<![A-Za-z\\]){symbol}(?![A-Za-z])", strings):
                    offenders.append(code_obj.scene_id)
                    break
        directive = f"Write {rule.use}, not {rule.instead_of}"
        checks.append(CheckResult(
            check="directive", passed=not offenders, severity="medium",
            message="" if not offenders else f"{directive}: '{rule.instead_of}' still appears in {', '.join(offenders)}.",
            measured={"directive": directive, "scenes": offenders},
        ))
    return checks


def notation_rules(directives: Optional[ProfessorDirectives]) -> List[str]:
    return [
        f"Write {n.use}" + (f", not {n.instead_of}" if n.instead_of else "") + (f" ({n.note})" if n.note else "") + "."
        for n in (directives.notation if directives else [])
    ]
