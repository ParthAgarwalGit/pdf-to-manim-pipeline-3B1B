"""Whole-script narration flow: measure repeated explanations and smooth them out.

Segments are scripted one at a time, so a script can restate the same idea
(the definition, the formula, the analogy) in several places and every
section can start over instead of following on. ``repetition`` measures that
deterministically; ``smoothing_prompt`` + ``apply_smoothing`` turn one LLM
rewrite of the whole script into a validated new script, kept only when it is
no longer and no more repetitive than the original.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# A sentence counts as repeated when this share of its word trigrams was
# already spoken earlier in the script.
SENTENCE_REPEAT = 0.5
MIN_BEAT_WORDS = 6

_STOP = {
    "the", "an", "and", "so", "it", "is", "of", "to", "that", "this", "we", "you", "i",
    "then", "by", "in", "on", "for", "with", "as", "be", "are", "now", "just", "what",
    "how", "its", "it's", "here", "there", "our", "your", "really", "again", "all",
}
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'")) if w not in _STOP]


def _trigrams(text: str) -> set:
    w = _words(text)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE.split((text or "").strip()) if s.strip()]


def repetition(narrations: List[str]) -> Dict[str, Any]:
    """Sentences that mostly repeat something said earlier in the script.

    Returns {"repeated_words", "total_words", "ratio", "repeats": [{"beat",
    "sentence", "overlap"}]} with beat indexes into ``narrations``.
    """
    heard: set = set()
    repeats: List[Dict[str, Any]] = []
    repeated_words = total = 0
    for index, text in enumerate(narrations):
        beat_grams: set = set()
        for sentence in sentences(text):
            words = len(sentence.split())
            total += words
            grams = _trigrams(sentence)
            if len(grams) >= 2:
                overlap = len(grams & heard) / len(grams)
                if overlap >= SENTENCE_REPEAT:
                    repeated_words += words
                    repeats.append({"beat": index, "sentence": sentence, "overlap": round(overlap, 2)})
            beat_grams |= grams
        heard |= beat_grams
    return {
        "repeated_words": repeated_words,
        "total_words": total,
        "ratio": round(repeated_words / total, 3) if total else 0.0,
        "repeats": repeats,
    }


def overlaps(text: str, other: str) -> float:
    """Share of ``text``'s trigrams that also occur in ``other``."""
    grams = _trigrams(text)
    return len(grams & _trigrams(other)) / len(grams) if grams else 0.0


def smoothing_prompt(
    beats: List[Any], word_budget: Optional[int], style: str, notes: List[str], retry_note: str = "",
) -> str:
    total = sum(len(b.narration.split()) for b in beats)
    listing = "\n".join(
        f"[{b.beat_id}]{' LOCKED' if getattr(b, 'locked', False) else ''} ({len(b.narration.split())} words) "
        f"(visual: {(b.visual_hint or 'none').strip()[:160]})\n{b.narration.strip()}"
        for b in beats
    )
    if word_budget and total > word_budget * 1.05:
        budget = (
            f"- LENGTH: the narration is {total} words and must come down to at most {word_budget} "
            f"(cut about {total - word_budget} words). Cut repetition first, then drop whole beats the "
            "professor's rules don't need (proof steps and side results go first), then shorten. "
            f"This limit is binding, but land close to it: not under {round(word_budget * 0.9)} words, "
            "because the video has a set length.\n"
        )
    elif word_budget:
        budget = (
            f"- Keep the narration at most {word_budget} words (it is {total} now); only cut what "
            "repeats, the video has a set length.\n"
        )
    else:
        budget = "- Never make the narration longer.\n"
    wording = (
        "- Keep the professor's rules:\n" + "\n".join(f"  - {n}" for n in notes) + "\n"
        if notes else ""
    )
    return (
        "You are editing the narration of a short lecture video so it sounds like one person "
        "talking from start to finish. It was drafted section by section, so some beats repeat what "
        "an earlier beat already said (the same definition, formula, analogy or conclusion) and some "
        "beats start over instead of following on.\n\n"
        f"BEATS, in order (each is spoken over the visual described):\n{listing}\n\n"
        "Edit the narration as little as possible so that:\n"
        "- Every idea is explained once. When a beat repeats something already said, cut the repeated "
        "part and keep only what is new; a short callback of a few words is fine. If nothing new is "
        "left, drop the beat. Two beats making the same point in different words is a repeat too.\n"
        "- Each beat picks up where the previous one stopped, the way a lecturer moves on. Vary how "
        "beats start; don't open every beat with So or Now.\n"
        "- Each beat still matches its visual.\n"
        "- No new facts, formulas or examples.\n"
        f"{budget}{wording}"
        "- A beat that repeats nothing and follows on fine comes back word for word. A beat you edit "
        "keeps about its length unless you are cutting a repeat. Every symbol or name a beat uses "
        "(like B or w) must be introduced in that beat or earlier.\n"
        "- Beats marked LOCKED come back word for word.\n"
        f"{retry_note}\n"
        f"{style}\n\n"
        "Respond in JSON as {\"beats\": [{\"beat_id\": \"beat_001\", \"narration\": \"...\", "
        "\"drop\": false}]} listing every beat id, in order."
    )


MAX_DROP_SHARE = 0.4


def apply_smoothing(
    script: Any, data: Dict[str, Any], allow_cuts: bool = False, min_words: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Validate one LLM rewrite; return (script, report). Rejects -> original script.

    A dropped beat normally may not take a concept with it. ``allow_cuts``
    (the script is over its length budget) lets any unlocked content beat
    go except the first, up to MAX_DROP_SHARE of the beats. A rewrite under
    ``min_words`` cut too much (the video has a set length) and is rejected.
    """
    beats = list(script.beats)
    before = repetition([b.narration for b in beats])
    report: Dict[str, Any] = {"applied": False, "before": _summary(before)}
    items = data.get("beats") if isinstance(data, dict) else None
    if not isinstance(items, list):
        report["reason"] = "no beats in the response"
        return script, report
    by_id = {str(i.get("beat_id")): i for i in items if isinstance(i, dict)}

    rewritten, dropped = [], []
    for beat in beats:
        item = by_id.get(beat.beat_id)
        if item is None or getattr(beat, "locked", False):
            rewritten.append(beat)
            continue
        if item.get("drop") is True and getattr(beat, "role", "core") != "bridge":
            dropped.append(beat)
            continue
        text = str(item.get("narration") or "").strip()
        words = len(text.split())
        if words < MIN_BEAT_WORDS or words > len(beat.narration.split()) * 1.3 + 10:
            rewritten.append(beat)  # an empty or inflated rewrite keeps the draft
            continue
        rewritten.append(beat.model_copy(update={"narration": text}))

    # A dropped beat may not take a concept with it: if it is the only beat
    # introducing a concept, it stays (as rewritten text if any, else the draft).
    # Over budget, cuts win, but never the opening and never most of the video.
    introduced = {c for b in rewritten for c in b.introduces_concepts}
    kept_ids = {b.beat_id for b in rewritten}
    max_drops = int(len(beats) * MAX_DROP_SHARE) if allow_cuts else 0
    drops = 0
    for beat in dropped:
        cut_ok = allow_cuts and beat is not beats[0] and drops < max_drops
        if not cut_ok and any(c not in introduced for c in beat.introduces_concepts):
            kept_ids.add(beat.beat_id)
            introduced.update(beat.introduces_concepts)
        else:
            drops += 1
    final = [b for b in beats if b.beat_id in kept_ids]
    final = [next((r for r in rewritten if r.beat_id == b.beat_id), b) for b in final]
    if not final:
        report["reason"] = "the rewrite dropped every beat"
        return script, report
    removed = {b.beat_id for b in beats} - {b.beat_id for b in final}
    final = [
        b.model_copy(update={"references_prior_beats": [r for r in b.references_prior_beats if r not in removed]})
        if removed & set(b.references_prior_beats) else b
        for b in final
    ]

    after = repetition([b.narration for b in final])
    report["after"] = _summary(after)
    if after["total_words"] > before["total_words"]:
        report["reason"] = "the rewrite is longer than the draft"
        return script, report
    if min_words and after["total_words"] < min_words:
        report["reason"] = f"the rewrite cut too much ({after['total_words']} < {min_words} words)"
        return script, report
    if after["repeated_words"] > before["repeated_words"]:
        report["reason"] = "the rewrite repeats more than the draft"
        return script, report

    used = {b.analogy_id for b in final if getattr(b, "analogy_id", None)}
    analogies = [a for a in getattr(script, "analogies", []) or [] if a.analogy_id in used]
    report.update(applied=True, dropped_beats=sorted(removed))
    return script.model_copy(update={"beats": final, "analogies": analogies}), report


def _summary(rep: Dict[str, Any]) -> Dict[str, Any]:
    return {k: rep[k] for k in ("total_words", "repeated_words", "ratio")}
