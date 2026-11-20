"""Professor controls for V2: brief, concept plan, scope, word budget, compliance.

Everything here is deterministic and provider-free except
``interpret_brief_with_router``, which goes through the budgeted V2 router.
Per-scene professor data lives in this module's own models and travels
alongside the ``LessonPlan``: the lesson and narration contracts are unchanged
so completed tasks of older runs keep their input hashes.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, Field

from src.agents.professor_input_interpreter import (
    ProfessorInputInterpreter,
    _reconcile_terms,
)
from src.schemas.directives import ProfessorDirectives

from .model_config import TEXT_MODEL
from .models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    NarrationPlan,
    SourceDocument,
    SourceFact,
)


Depth = Literal["skip", "brief", "standard", "deep"]

BRIEF_PROVIDER = "google"
BRIEF_MODEL = TEXT_MODEL

# "deep" stays inside the source: V2 narration may not add unsupported claims.
DEPTH_INSTRUCTIONS: dict[str, str] = {
    "skip": "do not cover it",
    "brief": "one short beat, the essential idea only",
    "standard": "explain it normally",
    "deep": (
        "go deeper: give the derivation step or worked example from the source its own beat"
    ),
}
DEPTH_WEIGHTS: dict[str, float] = {"brief": 0.5, "standard": 1.0, "deep": 1.6}
_VERBOSITY_DEPTH: dict[str, Depth] = {"low": "brief", "medium": "standard", "high": "deep"}

# The 2.0 s scene-end pause plus the pauses between a scene's beats.
SCENE_OVERHEAD_SECONDS = 3.2
SECONDS_PER_SCENE = 40
MIN_SCENE_WORDS = 25
LENGTH_TOLERANCE = 1.25
LENGTH_CHECK_TOLERANCE = 1.1
WORDS_PER_BEAT = 40
HEADING_MAX_CHARS = 90

_TRIM_ROLE_ORDER = ("reflect", "bridge", "demonstrate", "explain")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {"the", "a", "an", "of", "for", "analogy", "picture", "metaphor", "model", "with"}


# ── brief ───────────────────────────────────────────────────────────────────


def interpret_brief(brief: str) -> ProfessorDirectives:
    """Read a free-form brief with the deterministic rules only (never an LLM)."""

    return ProfessorInputInterpreter(use_default_llm=False).interpret_brief(brief)


async def interpret_brief_with_router(brief: str, router: Any) -> ProfessorDirectives:
    """Read a brief through the budgeted router, then reconcile it with the brief.

    Without a provider key the router returns the rule-based reading. With
    ``strict_keys`` it raises, like every other V2 stage.
    """

    brief = (brief or "").strip()
    if not brief:
        return ProfessorDirectives()
    rules = interpret_brief(brief)
    reading = await router.structured(
        stage="brief", task_id="brief", provider=BRIEF_PROVIDER, model=BRIEF_MODEL,
        system="You compile a professor's brief into structured directives.",
        prompt=ProfessorInputInterpreter._brief_prompt(brief),
        schema=ProfessorDirectives, fallback=lambda: rules, max_output_tokens=2000,
    )
    directives = reading.model_copy(deep=True)
    _reconcile_terms(directives, brief)
    directives.source_brief = brief
    if directives.is_empty():
        directives.notes = [brief]  # never drop the professor's words
    return directives


def readback_lines(directives: ProfessorDirectives) -> list[str]:
    return directives.readback()


def load_directives(path: Path | str) -> ProfessorDirectives:
    data = _parse_mapping(Path(path).read_text(encoding="utf-8"))
    return ProfessorDirectives.model_validate(data)


def dump_directives(directives: ProfessorDirectives) -> dict[str, Any]:
    return directives.model_dump(mode="json", by_alias=True)


def effective_target_minutes(
    directives: Optional[ProfessorDirectives], configured: Optional[float],
) -> Optional[float]:
    """The brief's own length wins over the configured one."""

    if directives is not None and directives.global_.target_minutes:
        return float(directives.global_.target_minutes)
    return configured


def effective_audience(
    directives: Optional[ProfessorDirectives], configured: Optional[str],
) -> Optional[str]:
    if directives is not None and directives.global_.audience:
        return directives.global_.audience
    return configured


def _parse_mapping(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except ValueError:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("expected a YAML or JSON mapping")
    return data


# ── concept plan ────────────────────────────────────────────────────────────


class ConceptEntry(BaseModel):
    scene_id: str
    concept: str
    learning_goal: str
    depth: Depth = "standard"
    notes: str = ""
    analogy: Literal["auto", "none", "professor"] = "auto"
    analogy_brief: str = ""
    description: str = ""
    visual_objective: str = ""
    new_visual_fact: str = ""
    do_not_repeat: list[str] = Field(default_factory=list)
    example: Optional[str] = None
    source_fact_ids: list[str] = Field(default_factory=list)

    def instruction(self) -> str:
        """The binding professor notes for this scene, analogy choice included."""

        parts = [self.notes.strip()] if self.notes.strip() else []
        if self.analogy == "none":
            parts.append("use no analogy")
        elif self.analogy == "professor" and self.analogy_brief:
            parts.append(f"use the professor's analogy: {self.analogy_brief}")
        return "; ".join(parts)


class ConceptPlan(BaseModel):
    version: int = 1
    source_sha256: str = ""
    title: str
    audience: str
    objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    concept_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    concepts: list[ConceptEntry] = Field(default_factory=list)

    @classmethod
    def from_lesson(
        cls,
        lesson: LessonPlan,
        source: SourceDocument,
        directives: Optional[ProfessorDirectives] = None,
    ) -> "ConceptPlan":
        verbosity = directives.global_.verbosity if directives is not None else None
        default_depth = _VERBOSITY_DEPTH.get(verbosity or "medium", "standard")
        entries = [
            ConceptEntry(
                scene_id=scene.scene_id, concept=scene.title,
                learning_goal=scene.learning_goal, depth=default_depth,
                description=scene.description, visual_objective=scene.visual_objective,
                new_visual_fact=scene.new_visual_fact,
                do_not_repeat=list(scene.do_not_repeat), example=scene.example,
                source_fact_ids=list(scene.source_fact_ids),
            )
            for scene in lesson.scenes
        ]
        for directive in (directives.concepts if directives is not None else []):
            target = directive.concept.lower().strip()
            if not target:
                continue  # a lesson-wide directive; it reaches every prompt as a binding line
            for entry in entries:
                if not _concept_matches(target, entry):
                    continue
                if directive.depth:
                    entry.depth = directive.depth
                if directive.analogy_brief:
                    entry.analogy, entry.analogy_brief = "professor", directive.analogy_brief
                notes = [directive.notes] if directive.notes else []
                notes += [f"never use the {item} analogy" for item in directive.avoid_analogies]
                if notes:
                    entry.notes = "; ".join(filter(None, [entry.notes] + notes))
        return cls(
            source_sha256=source.sha256, title=lesson.title, audience=lesson.audience,
            objectives=list(lesson.objectives), prerequisites=list(lesson.prerequisites),
            concept_dependencies=dict(lesson.concept_dependencies), concepts=entries,
        )

    def to_lesson(self, source: SourceDocument) -> tuple[LessonPlan, list[str]]:
        """The lesson this plan describes, plus the concepts the professor skipped."""

        if self.source_sha256 and self.source_sha256 != source.sha256:
            raise ValueError("Concept plan belongs to a different source PDF")
        known = {fact.fact_id for fact in source.facts}
        unknown = sorted({
            fact_id
            for entry in self.concepts
            for fact_id in entry.source_fact_ids
            if fact_id not in known
        })
        if unknown:
            raise ValueError(f"Concept plan cites unknown source_fact_ids: {unknown}")
        kept = [entry for entry in self.concepts if entry.depth != "skip"]
        if not kept:
            raise ValueError("Concept plan skips every concept; keep at least one.")
        scenes = [
            LessonScene(
                scene_id=entry.scene_id, title=entry.concept, description=entry.description,
                learning_goal=entry.learning_goal, source_fact_ids=entry.source_fact_ids,
                example=entry.example, visual_objective=entry.visual_objective,
                new_visual_fact=entry.new_visual_fact, do_not_repeat=entry.do_not_repeat,
            )
            for entry in kept
        ]
        lesson = LessonPlan(
            title=self.title, audience=self.audience,
            objectives=self.objectives or [scene.learning_goal for scene in scenes[:4]],
            prerequisites=self.prerequisites, concept_dependencies=self.concept_dependencies,
            scenes=scenes,
        )
        return lesson, [entry.concept for entry in self.skipped()]

    def _entry(self, scene_id: str) -> Optional[ConceptEntry]:
        return next((entry for entry in self.concepts if entry.scene_id == scene_id), None)

    def depth_for(self, scene_id: str) -> str:
        entry = self._entry(scene_id)
        return entry.depth if entry is not None else "standard"

    def notes_for(self, scene_id: str) -> str:
        entry = self._entry(scene_id)
        return entry.instruction() if entry is not None else ""

    def skipped(self) -> list[ConceptEntry]:
        return [entry for entry in self.concepts if entry.depth == "skip"]

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"), sort_keys=False, allow_unicode=True,
        )

    @classmethod
    def from_text(cls, text: str) -> "ConceptPlan":
        return cls.model_validate(_parse_mapping(text))

    @classmethod
    def load(cls, path: Path | str) -> "ConceptPlan":
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        text = (
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
            if target.suffix.lower() == ".json" else self.to_yaml()
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target


def _concept_matches(target: str, entry: ConceptEntry) -> bool:
    for value in (entry.concept, entry.learning_goal):
        value = value.lower().strip()
        if value and (target in value or value in target):
            return True
    return False


# ── compliance model ────────────────────────────────────────────────────────


class ComplianceCheck(BaseModel):
    check: str
    passed: bool
    severity: Literal["low", "medium", "high"] = "medium"
    message: str = ""
    scene_ids: list[str] = Field(default_factory=list)
    beat_ids: list[str] = Field(default_factory=list)


# ── scope ───────────────────────────────────────────────────────────────────


_HEADING = re.compile(
    r"^\s*(?:(?:section|sec\.?|chapter)\s+|§\s*)?(\d+(?:\.\d+)*)\.?(?:\s+(?P<title>\S.*))?$",
    re.IGNORECASE | re.DOTALL,
)
_LABEL_PREFIX = re.compile(r"^(?:(?:section|sec\.?|chapter)\s+|§\s*)", re.IGNORECASE)


def _heading_number(fact: SourceFact) -> Optional[str]:
    """The section number a short text fact opens with, when it reads as a heading."""

    if fact.kind != "text" or len(fact.text) > HEADING_MAX_CHARS:
        return None
    match = _HEADING.match(fact.text)
    if match is None:
        return None
    number, title = match.group(1), match.group("title")
    if title is None:
        return number if "." in number else None  # a bare "12" is a page number
    title = title.strip()
    # "2 + 3 = 5", "10 minutes later ..." and list items ("1. Compute the sum.")
    # also open with a number; headings carry a capitalised, unpunctuated title.
    if not title[0].isalpha() or title[0].islower() or title[-1] in ".,;?!":
        return None
    return number


def apply_scope(
    source: SourceDocument,
    directives: Optional[ProfessorDirectives],
) -> tuple[SourceDocument, list[ComplianceCheck]]:
    """Drop the facts the professor excluded. Fact ids and page_count are preserved."""

    if directives is None:
        return source, []
    scope = directives.scope
    only_pages, skip_pages = set(scope.only_pages), set(scope.skip_pages)
    if not (only_pages or skip_pages or scope.skip_sections):
        return source, []
    facts = source.facts
    real_pages = set(range(1, source.page_count + 1))
    dropped: set[int] = set()
    checks: list[ComplianceCheck] = []

    if only_pages:
        dropped |= {i for i, fact in enumerate(facts) if fact.page_number not in only_pages}
        found = bool(only_pages & real_pages)
        checks.append(ComplianceCheck(
            check=f"Only use pages {scope.only_pages[0]}-{scope.only_pages[-1]}", passed=found,
            message="" if found else (
                f"None of the requested pages exist; the PDF has {source.page_count} page(s)."
            ),
        ))
    if skip_pages:
        dropped |= {i for i, fact in enumerate(facts) if fact.page_number in skip_pages}
        missing = sorted(skip_pages - real_pages)
        checks.append(ComplianceCheck(
            check=f"Leave out page(s): {', '.join(map(str, scope.skip_pages))}",
            passed=not missing,
            message="" if not missing else (
                f"Page(s) {', '.join(map(str, missing))} do not exist; "
                f"the PDF has {source.page_count} page(s)."
            ),
        ))

    # OCR facts are appended after extraction, so walk in page order.
    reading_order = sorted(range(len(facts)), key=lambda index: facts[index].page_number)
    headings = {index: _heading_number(facts[index]) for index in reading_order}
    for label in scope.skip_sections:
        wanted = _LABEL_PREFIX.sub("", label.strip()).rstrip(".")
        level = wanted.count(".") + 1
        found = skipping = False
        for index in reading_order:
            number = headings[index]
            if number is not None:
                if number == wanted or (found and number.startswith(wanted + ".")):
                    found = skipping = True
                elif skipping and number.count(".") + 1 <= level:
                    skipping = False
            if skipping:
                dropped.add(index)
        checks.append(ComplianceCheck(
            check=f"Skip section {label}", passed=found,
            message="" if found else (
                f"Skip section {label}: no heading with that label was found in the PDF."
            ),
        ))

    kept = [fact for index, fact in enumerate(facts) if index not in dropped]
    if not kept:  # never skip everything: fall back to the full document
        unapplied = [
            check.model_copy(update={
                "passed": False,
                "message": check.message or "Not applied: the scope excluded the whole document.",
            })
            for check in checks
        ]
        return source, unapplied + [ComplianceCheck(
            check="Scope", passed=False, severity="high",
            message="The requested scope excluded the whole document; all of it was kept.",
        )]
    return source.model_copy(update={"facts": kept}), checks


# ── word budget ─────────────────────────────────────────────────────────────


class LengthBudget(BaseModel):
    target_minutes: float
    words_per_minute: int
    total_words: int
    scene_words: dict[str, int] = Field(default_factory=dict)
    scene_cap: int


def scene_cap_for(target_minutes: float) -> int:
    return max(1, round(float(target_minutes) * 60 / SECONDS_PER_SCENE))


def total_word_budget(target_minutes: float, words_per_minute: float, scene_count: int) -> int:
    speech_seconds = float(target_minutes) * 60 - scene_count * SCENE_OVERHEAD_SECONDS
    return max(MIN_SCENE_WORDS * scene_count, round(speech_seconds * words_per_minute / 60))


def allocate_scene_words(
    lesson: LessonPlan,
    source: SourceDocument,
    total_words: int,
    depth_by_scene: Optional[Mapping[str, str]] = None,
) -> dict[str, int]:
    """Split the budget by depth weight x cited source size, never below the floor."""

    depth_by_scene = depth_by_scene or {}
    fact_words = {fact.fact_id: len(fact.text.split()) for fact in source.facts}
    weights = {
        scene.scene_id: DEPTH_WEIGHTS.get(depth_by_scene.get(scene.scene_id, "standard"), 1.0)
        * max(1, sum(fact_words.get(fact_id, 0) for fact_id in scene.source_fact_ids))
        for scene in lesson.scenes
    }
    budgets: dict[str, int] = {}
    remaining = total_words
    # Scenes whose share falls under the floor are pinned to it; the others
    # share what is left, so the allocation still sums to the total.
    while weights:
        whole = sum(weights.values())
        low = [
            scene_id for scene_id, weight in weights.items()
            if remaining * weight / whole < MIN_SCENE_WORDS
        ]
        if not low:
            break
        for scene_id in low:
            budgets[scene_id] = MIN_SCENE_WORDS
            remaining -= MIN_SCENE_WORDS
            del weights[scene_id]
    if weights:
        whole = sum(weights.values())
        shares = {scene_id: remaining * weight / whole for scene_id, weight in weights.items()}
        for scene_id, share in shares.items():
            budgets[scene_id] = int(share)
        spare = remaining - sum(int(share) for share in shares.values())
        by_remainder = sorted(shares, key=lambda scene_id: shares[scene_id] % 1, reverse=True)
        for scene_id in by_remainder[:spare]:
            budgets[scene_id] += 1
    return {scene.scene_id: budgets[scene.scene_id] for scene in lesson.scenes}


def build_length_budget(
    lesson: LessonPlan,
    source: SourceDocument,
    target_minutes: float,
    words_per_minute: int,
    depth_by_scene: Optional[Mapping[str, str]] = None,
) -> LengthBudget:
    total = total_word_budget(target_minutes, words_per_minute, len(lesson.scenes))
    return LengthBudget(
        target_minutes=target_minutes, words_per_minute=words_per_minute, total_words=total,
        scene_words=allocate_scene_words(lesson, source, total, depth_by_scene),
        scene_cap=scene_cap_for(target_minutes),
    )


def _word_count(text: str) -> int:
    return len(text.split())


def _scene_word_count(scene: NarratedScene) -> int:
    return sum(_word_count(beat.text) for beat in scene.beats)


def _with_beats(scene: NarratedScene, beats: list[NarrationBeat]) -> NarratedScene:
    return scene.model_copy(update={
        "beats": beats, "narration": " ".join(beat.text for beat in beats),
    })


def truncate_sentences(sentences: list[str], max_words: int) -> list[str]:
    """The leading sentences that fit ``max_words``; always at least some text."""

    kept: list[str] = []
    used = 0
    for sentence in sentences:
        count = _word_count(sentence)
        if used + count > max_words:
            break
        kept.append(sentence)
        used += count
    if kept or not sentences:
        return kept
    return [" ".join(sentences[0].split()[:max(1, max_words)])]


def trim_to_budget(
    narration: NarrationPlan,
    budget: LengthBudget,
    tolerance: float = LENGTH_TOLERANCE,
) -> tuple[NarrationPlan, dict[str, Any]]:
    """Cut scenes that overshoot their word budget. Deterministic, never longer."""

    words_before = sum(_scene_word_count(scene) for scene in narration.scenes)
    removed: list[str] = []
    truncated: list[str] = []
    over: list[str] = []
    scenes: list[NarratedScene] = []
    for scene in narration.scenes:
        allowed = budget.scene_words.get(scene.scene_id)
        limit = None if allowed is None else allowed * tolerance
        total = _scene_word_count(scene)
        if limit is None or total <= limit:
            scenes.append(scene)
            continue
        beats = list(scene.beats)
        for role in _TRIM_ROLE_ORDER:
            for index in range(len(beats) - 1, 0, -1):  # the opening beat always stays
                if total <= limit:
                    break
                if beats[index].role == role:
                    total -= _word_count(beats[index].text)
                    removed.append(beats[index].beat_id)
                    del beats[index]
        while total > limit:
            longest: Optional[int] = None
            for index, beat in enumerate(beats):
                if len(_SENTENCE_BREAK.split(beat.text.strip())) < 2:
                    continue
                if longest is None or _word_count(beat.text) >= _word_count(beats[longest].text):
                    longest = index
            if longest is None:
                break
            beat = beats[longest]
            text = " ".join(_SENTENCE_BREAK.split(beat.text.strip())[:-1])
            total -= _word_count(beat.text) - _word_count(text)
            beats[longest] = beat.model_copy(update={"text": text})
            if beat.beat_id not in truncated:
                truncated.append(beat.beat_id)
        if total > limit:
            over.append(scene.scene_id)
        scenes.append(_with_beats(scene, beats))
    words_after = sum(_scene_word_count(scene) for scene in scenes)
    report = {
        "target_minutes": budget.target_minutes,
        "budget_words": budget.total_words,
        "tolerance": tolerance,
        "words_before": words_before,
        "words_after": words_after,
        "removed_beats": removed,
        "truncated_beats": truncated,
        "over_budget_scenes": over,
        "within_budget": words_after <= budget.total_words * tolerance,
    }
    if not removed and not truncated:
        return narration, report
    return NarrationPlan(scenes=scenes), report


def length_check(total_audio_seconds: float, target_minutes: float) -> ComplianceCheck:
    limit = float(target_minutes) * 60 * LENGTH_CHECK_TOLERANCE
    passed = total_audio_seconds <= limit
    return ComplianceCheck(
        check=f"Under {float(target_minutes):g} minutes", passed=passed,
        message=(
            f"Narration runs about {total_audio_seconds / 60:.1f} min"
            + ("." if passed else f", over the requested {float(target_minutes):g} min.")
        ),
    )


# ── compliance ──────────────────────────────────────────────────────────────


def _mentions(text: str, phrase: str, *, last_word: bool = True) -> bool:
    """The phrase appears as words in ``text``.

    ``last_word`` also accepts the phrase's last content word, which is how an
    analogy is usually named ("the clock analogy" -> "clock"). Concept names
    are matched whole: skipping "series resonance" must not flag every beat
    that says "resonance".
    """

    text = (text or "").lower()
    phrase = (phrase or "").lower().replace("_", " ").strip()
    if not phrase:
        return False
    if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
        return True
    if last_word:
        words = [word for word in re.findall(r"[a-z0-9]+", phrase) if word not in _STOPWORDS]
        return bool(words) and bool(
            re.search(rf"(?<!\w){re.escape(words[-1])}s?(?!\w)", text)
        )
    tokens = re.findall(r"\w+", phrase)
    while len(tokens) > 1 and tokens[0] in {"the", "a", "an"}:
        tokens = tokens[1:]
    if not tokens:
        return False
    pattern = r"\W+".join(re.escape(token) for token in tokens)
    return bool(re.search(rf"(?<!\w){pattern}s?(?!\w)", text))


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def _match_case(found: str, preferred: str) -> str:
    if len(found) > 1 and found.isupper():
        return preferred.upper()
    if found[:1].isupper():
        return preferred[:1].upper() + preferred[1:]
    return preferred


def _replace_term(text: str, avoid: str, preferred: str) -> str:
    return _term_pattern(avoid).sub(lambda match: _match_case(match.group(0), preferred), text)


def _still_says(text: str, avoid: str, preferred: str) -> bool:
    # The preferred wording may itself contain the avoided word ("state vector").
    return bool(_term_pattern(avoid).search(_term_pattern(preferred).sub(" ", text)))


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for value in values:
        key = (value or "").strip().lower()
        if key and key not in seen:
            seen[key] = value.strip()
    return list(seen.values())


def _skipped_concepts(
    directives: Optional[ProfessorDirectives], omitted_concepts: Iterable[str],
) -> list[str]:
    from_brief = [
        concept.concept
        for concept in (directives.concepts if directives is not None else [])
        if concept.depth == "skip"
    ]
    return _unique([*from_brief, *omitted_concepts])


def _remove_mentions(
    scenes: list[NarratedScene], phrase: str, *, last_word: bool,
) -> tuple[list[NarratedScene], list[str], list[str], list[str]]:
    """Drop beats that mention ``phrase``; a scene always keeps one beat.

    Returns (scenes, removed beat ids, remaining beat ids, their scene ids).
    """

    updated: list[NarratedScene] = []
    removed: list[str] = []
    remaining: list[str] = []
    remaining_scenes: list[str] = []
    for scene in scenes:
        offending = [
            beat for beat in scene.beats if _mentions(beat.text, phrase, last_word=last_word)
        ]
        if not offending:
            updated.append(scene)
            continue
        offending_ids = {beat.beat_id for beat in offending}
        kept = [beat for beat in scene.beats if beat.beat_id not in offending_ids]
        if not kept:
            kept = [scene.beats[0]]
            remaining.append(scene.beats[0].beat_id)
            remaining_scenes.append(scene.scene_id)
        kept_ids = {beat.beat_id for beat in kept}
        removed.extend(beat.beat_id for beat in scene.beats if beat.beat_id not in kept_ids)
        updated.append(_with_beats(scene, kept) if len(kept) < len(scene.beats) else scene)
    return updated, removed, remaining, remaining_scenes


def enforce_narration_directives(
    narration: NarrationPlan,
    directives: Optional[ProfessorDirectives],
    omitted_concepts: Iterable[str] = (),
) -> tuple[NarrationPlan, list[ComplianceCheck]]:
    """Apply what can be enforced without a model; report one check per directive."""

    scenes = list(narration.scenes)
    checks: list[ComplianceCheck] = []

    terminology = directives.global_.terminology if directives is not None else {}
    for avoid, preferred in terminology.items():
        if not avoid.strip() or not preferred.strip():
            continue
        changed: list[str] = []
        left: list[str] = []
        left_scenes: list[str] = []
        for index, scene in enumerate(scenes):
            beats = []
            for beat in scene.beats:
                text = _replace_term(beat.text, avoid, preferred)
                if text != beat.text:
                    changed.append(beat.beat_id)
                    beat = beat.model_copy(update={"text": text})
                if _still_says(text, avoid, preferred):
                    left.append(beat.beat_id)
                    left_scenes.append(scene.scene_id)
                beats.append(beat)
            spoken = _replace_term(scene.narration, avoid, preferred)
            if spoken != scene.narration or any(
                new is not old for new, old in zip(beats, scene.beats)
            ):
                scenes[index] = scene.model_copy(update={"beats": beats, "narration": spoken})
        checks.append(ComplianceCheck(
            check=f"Say '{preferred}' rather than '{avoid}'", passed=not left,
            message=(
                f"'{avoid}' is still said in {', '.join(left)}." if left
                else f"Replaced '{avoid}' in {len(changed)} beat(s)." if changed else ""
            ),
            scene_ids=_unique(left_scenes), beat_ids=left or changed,
        ))

    analogies = _unique(directives.avoided_analogies()) if directives is not None else []
    removals = [(f"Never use the {item} analogy", item, True) for item in analogies]
    removals += [
        (f"Do not cover {item}", item, False)
        for item in _skipped_concepts(directives, omitted_concepts)
    ]
    for label, phrase, last_word in removals:
        scenes, removed, left, left_scenes = _remove_mentions(
            scenes, phrase, last_word=last_word,
        )
        checks.append(ComplianceCheck(
            check=label, passed=not left,
            message=(
                f"Still present in {', '.join(left)}: it is the only beat of its scene."
                if left else f"Removed {len(removed)} beat(s)." if removed else ""
            ),
            scene_ids=left_scenes, beat_ids=left or removed,
        ))

    if all(new is old for new, old in zip(scenes, narration.scenes)):
        return narration, checks
    return NarrationPlan(scenes=scenes), checks


def check_lesson_directives(
    lesson: LessonPlan,
    source: SourceDocument,
    directives: Optional[ProfessorDirectives],
    omitted_concepts: Iterable[str] = (),
) -> list[ComplianceCheck]:
    checks: list[ComplianceCheck] = []
    for concept in _skipped_concepts(directives, omitted_concepts):
        offenders = [
            scene.scene_id for scene in lesson.scenes
            if _mentions(scene.title, concept, last_word=False)
            or _mentions(scene.learning_goal, concept, last_word=False)
        ]
        checks.append(ComplianceCheck(
            check=f"Do not cover {concept}", passed=not offenders,
            message="" if not offenders else (
                f"The lesson still plans {concept} in {', '.join(offenders)}."
            ),
            scene_ids=offenders,
        ))
    if directives is None:
        return checks
    scope = directives.scope
    pages = {fact.fact_id: fact.page_number for fact in source.facts}

    def citing(excluded) -> list[str]:
        return [
            scene.scene_id for scene in lesson.scenes
            if any(
                fact_id in pages and excluded(pages[fact_id])
                for fact_id in scene.source_fact_ids
            )
        ]

    if scope.skip_pages:
        skipped = set(scope.skip_pages)
        offenders = citing(lambda page: page in skipped)
        checks.append(ComplianceCheck(
            check=f"Leave out page(s): {', '.join(map(str, scope.skip_pages))}",
            passed=not offenders,
            message="" if not offenders else (
                f"Scene(s) {', '.join(offenders)} cite facts from an excluded page."
            ),
            scene_ids=offenders,
        ))
    if scope.only_pages:
        allowed = set(scope.only_pages)
        offenders = citing(lambda page: page not in allowed)
        checks.append(ComplianceCheck(
            check=f"Only use pages {scope.only_pages[0]}-{scope.only_pages[-1]}",
            passed=not offenders,
            message="" if not offenders else (
                f"Scene(s) {', '.join(offenders)} cite facts from outside the requested pages."
            ),
            scene_ids=offenders,
        ))
    return checks


_TEX_NAMES = {"MathTex", "Tex"}
_TEX_CALL = re.compile(r"\b(?:MathTex|Tex)\s*\(([^)]*)\)", re.DOTALL)
_STRING = re.compile(r"""[rRbBfFuU]{0,2}(['"])(.*?)(?<!\\)\1""", re.DOTALL)


def _tex_strings(code: str) -> list[tuple[int, str]]:
    """(line, text) of every string literal passed to MathTex(...) / Tex(...)."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Code that does not parse is still scanned, as the legacy check did.
        return [
            (code.count("\n", 0, call.start()) + 1, found.group(2))
            for call in re.finditer(r"\b(?:MathTex|Tex)\s*\(([^)\n]*)", code)
            for found in _STRING.finditer(call.group(1))
        ]
    strings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in _TEX_NAMES:
            continue
        for argument in node.args:
            for part in ast.walk(argument):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    strings.append((part.lineno, part.value))
    return strings


def notation_findings(code: str, directives: Optional[ProfessorDirectives]) -> list[str]:
    """One message per notation directive the generated scene code breaks."""

    rules = [rule for rule in (directives.notation if directives else []) if rule.instead_of]
    if not rules:
        return []
    strings = _tex_strings(code or "")
    findings: list[str] = []
    for rule in rules:
        # Letters and backslashes on either side belong to another symbol
        # (\pi is not i); a leading "_" or "_{" is an index, not the symbol.
        pattern = re.compile(
            rf"(?<![A-Za-z\\])(?<!_)(?<!_\{{){re.escape(rule.instead_of)}(?![A-Za-z])"
        )
        lines = sorted({line for line, text in strings if pattern.search(text)})
        if lines:
            findings.append(
                f"Notation: write {rule.use}, not {rule.instead_of}; "
                f"'{rule.instead_of}' appears in MathTex/Tex text on line(s) "
                f"{', '.join(map(str, lines))}."
            )
    return findings


def notation_rules(directives: Optional[ProfessorDirectives]) -> list[str]:
    return [
        f"Write {rule.use}"
        + (f", not {rule.instead_of}" if rule.instead_of else "")
        + (f" ({rule.note})" if rule.note else "")
        + "."
        for rule in (directives.notation if directives else [])
    ]


# ── prompt blocks ───────────────────────────────────────────────────────────


_BLOCK_HEADER = (
    "\n\nPROFESSOR DIRECTIVES (binding; they override any conflicting guidance above):\n"
)


def lesson_directive_block(
    directives: Optional[ProfessorDirectives],
    audience: Optional[str] = None,
    scene_cap: Optional[int] = None,
) -> str:
    lines: list[str] = []
    audience = effective_audience(directives, audience)
    if audience:
        lines.append(f"Audience: {audience}. Pitch the lesson at this level.")
    if scene_cap is not None:
        lines.append(
            f"Length: plan at most {scene_cap} scene(s) so the video fits the requested "
            "length; prioritise the central ideas over exhaustive coverage."
        )
    if directives is not None:
        lines.extend(directives.script_instructions())
    if not lines:
        return ""
    return _BLOCK_HEADER + "\n".join(f"- {line}" for line in lines)


def narration_directive_block(
    directives: Optional[ProfessorDirectives],
    scene_words: Mapping[str, int],
    depth_by_scene: Mapping[str, str],
    notes_by_scene: Mapping[str, str],
    audience: Optional[str] = None,
) -> str:
    lines: list[str] = []
    audience = effective_audience(directives, audience)
    if audience:
        lines.append(f"Audience: {audience}. Pitch the narration at this level.")
    if directives is not None:
        lines.extend(directives.script_instructions())
    scene_lines: list[str] = []
    for scene_id in dict.fromkeys([*scene_words, *depth_by_scene, *notes_by_scene]):
        parts: list[str] = []
        if scene_id in scene_words:
            parts.append(f"at most about {scene_words[scene_id]} words")
        depth = depth_by_scene.get(scene_id, "standard")
        if depth != "standard" and depth in DEPTH_INSTRUCTIONS:
            parts.append(DEPTH_INSTRUCTIONS[depth])
        if notes_by_scene.get(scene_id):
            parts.append(f"professor's note: {notes_by_scene[scene_id]}")
        if parts:
            scene_lines.append(f"- {scene_id}: " + "; ".join(parts))
    if not lines and not scene_lines:
        return ""
    if scene_lines:
        lines_text = "\n".join(f"- {line}" for line in lines)
        block = _BLOCK_HEADER + (lines_text + "\n" if lines_text else "")
        block += "Per scene:\n" + "\n".join(scene_lines)
        if scene_words:
            block += (
                "\nThe word counts are a length budget for the whole video. Use about one "
                f"beat per {WORDS_PER_BEAT} words, fewer beats than usual when a budget is "
                "small, and drop secondary examples and repetition rather than exceed it."
            )
        return block
    return _BLOCK_HEADER + "\n".join(f"- {line}" for line in lines)
