"""Pre-TTS review of the planned lesson and its narration."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.prompts.review import (
    PLANNED_CRITERIA_WEIGHTS,
    REPORTED_ONLY_CRITERION,
    storyboard_review_prompt,
)

from .model_config import TEXT_MODEL
from .models import LessonPlan, NarrationPlan, SourceDocument
from .prompt_text import (
    FLOW_PASS_SCORE,
    MAX_SPOKEN_SENTENCE_WORDS,
    REVIEW_PASS_SCORE,
    STOCK_PHRASES,
)
from .providers import clean_json_text
from .state import BudgetExceeded


SCRIPT_REVIEW_PROMPT_VERSION = "script-review-3-voice"
SOURCE_SUMMARY_CHARS = 3000
# An in-depth lecture is judged against far more of the source, and its
# reviewer has more sentences to list.
DEEP_SOURCE_SUMMARY_CHARS = 14000
DEEP_REVIEW_OUTPUT_TOKENS = 8000
TREATMENTS = ("covered", "compressed", "deferred", "omitted")


class ScriptReview(BaseModel):
    overall_score: float
    passed: bool
    criteria: dict[str, dict[str, Any]]  # name -> {"score", "feedback", "weight"}
    summary: str = ""
    improvement_suggestions: list[str] = Field(default_factory=list)
    # Sentences that do not follow from what came before:
    # {"where": "hook" or a scene_id, "sentence": ..., "problem": ...}.
    flow_breaks: list[dict[str, str]] = Field(default_factory=list)
    # How many sentences the deterministic voice check flagged (with_voice_check),
    # and whether the reviewer itself passed the script before they were added.
    voice_problems: int = 0
    reviewer_passed: Optional[bool] = None
    attempt: int = 1

    def reviewer_breaks(self) -> list[dict[str, str]]:
        """Scene flow breaks the reviewer found (not the voice check's)."""

        return [item for item in self.scene_breaks() if item.get("kind") != "voice"]

    def hook_breaks(self) -> list[dict[str, str]]:
        """Flow breaks in the hook, which only a new hook can fix."""

        return [item for item in self.flow_breaks if _is_hook(item)]

    def scene_breaks(self) -> list[dict[str, str]]:
        """Flow breaks in the narration, which a rewrite of their beats can fix."""

        return [item for item in self.flow_breaks if not _is_hook(item)]

    def weakest_criterion(self) -> Optional[tuple[str, dict[str, Any]]]:
        """The lowest-scoring criterion that counts towards the overall score."""

        weighted = [
            (name, entry) for name, entry in self.criteria.items()
            if float(entry.get("weight") or 0.0) > 0.0
        ]
        return min(weighted, key=lambda item: float(item[1]["score"]), default=None)


async def review_script(
    router: Any,
    lesson: LessonPlan,
    narration: NarrationPlan,
    source: SourceDocument,
    attempt: int = 1,
    mode: str = "brief",
) -> Optional[ScriptReview]:
    """Grade the planned script; ``None`` (with an event) when the review could not run."""

    deep = mode == "deep"
    task_id = f"script-review-{attempt}"
    prompt = storyboard_review_prompt(
        _plan_summary(lesson, narration),
        _scenes(narration),
        _coverage(lesson),
        "\n".join(fact.text for fact in source.facts)[
            :DEEP_SOURCE_SUMMARY_CHARS if deep else SOURCE_SUMMARY_CHARS
        ],
        mode=mode,
    )
    try:
        response = await router.text(
            stage="script_review", task_id=task_id, provider="google",
            model=TEXT_MODEL,
            system=(
                "You are a demanding editor of in-depth educational lectures." if deep
                else "You are a demanding editor of short educational videos."
            ),
            prompt=prompt, max_output_tokens=DEEP_REVIEW_OUTPUT_TOKENS if deep else 3000,
        )
        data = json.loads(clean_json_text(response.text))
        if not isinstance(data, dict):
            raise ValueError("the review response is not a JSON object")
    except BudgetExceeded:
        raise
    except Exception as exc:
        router.artifacts.event(
            "script_review_skipped", stage="script_review", task_id=task_id,
            message=f"Script review did not run: {exc}",
            reason=type(exc).__name__, attempt=attempt,
        )
        return None
    return parse_review(data, attempt)


def parse_review(data: dict[str, Any], attempt: int = 1) -> ScriptReview:
    """Score a reviewer's JSON the way the legacy storyboard grader does."""

    def read(name: str) -> tuple[float, str]:
        entry = data.get(name)
        if not isinstance(entry, dict):
            entry = {}
        try:
            score = float(entry.get("score", 5.0))
        except (TypeError, ValueError):
            score = 5.0
        if score != score:  # NaN
            score = 5.0
        return max(0.0, min(10.0, score)), str(entry.get("feedback") or "")

    criteria: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
    for name, weight in PLANNED_CRITERIA_WEIGHTS.items():
        score, feedback = read(name)
        criteria[name] = {"score": score, "feedback": feedback, "weight": weight}
        weighted_sum += score * weight
    score, feedback = read(REPORTED_ONLY_CRITERION)
    criteria[REPORTED_ONLY_CRITERION] = {"score": score, "feedback": feedback, "weight": 0.0}

    raw = data.get("improvement_suggestions")
    suggestions = (
        [item for item in raw if isinstance(item, str) and item.strip()]
        if isinstance(raw, list) else []
    )
    breaks = _flow_breaks(data.get("flow_breaks"))
    scene_breaks = [item for item in breaks if not _is_hook(item)]
    # Each broken sentence in the narration goes to the rewrite as its own fix.
    suggestions += [
        f'In {item["where"] or "the narration"}, the sentence "{item["sentence"]}" does not '
        f'follow from what came before: {item["problem"]} Rewrite it so it does.'
        for item in scene_breaks
    ]
    overall = round(weighted_sum, 2)
    flow = criteria.get("logical_flow", {}).get("score", 10.0)
    return ScriptReview(
        overall_score=overall,
        passed=overall >= REVIEW_PASS_SCORE and flow >= FLOW_PASS_SCORE and not scene_breaks,
        criteria=criteria,
        summary=str(data.get("summary") or ""),
        improvement_suggestions=suggestions,
        flow_breaks=breaks,
        attempt=attempt,
    )


def _flow_breaks(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    breaks = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sentence = " ".join(str(item.get("sentence") or "").split())
        if not sentence:
            continue
        breaks.append({
            "where": " ".join(str(item.get("where") or "").split()),
            "sentence": sentence,
            "problem": " ".join(str(item.get("problem") or "").split()),
        })
    return breaks


def _is_hook(item: dict[str, str]) -> bool:
    return item.get("where", "").strip().lower() == "hook"


BeatKey = tuple[str, str]  # (scene_id, beat_id)


def locate_flow_breaks(
    narration: NarrationPlan, breaks: list[dict[str, str]],
) -> tuple[dict[BeatKey, list[dict[str, str]]], list[dict[str, str]]]:
    """Group flow breaks by the beat whose text holds the quoted sentence.

    The scene a break names is searched first, then every other scene, as a
    reviewer may name the scene loosely. Matching ignores case, spacing and
    the quote marks and end punctuation around the quoted sentence. Returns
    the breaks by beat, in narration order, and the breaks no beat holds.
    """

    found: dict[BeatKey, list[dict[str, str]]] = {}
    unplaced: list[dict[str, str]] = []
    for item in breaks:
        quoted = _comparable(item["sentence"])
        scenes = sorted(narration.scenes, key=lambda scene: scene.scene_id != item["where"])
        key = next(
            (
                (scene.scene_id, beat.beat_id)
                for scene in scenes for beat in scene.beats
                if quoted and quoted in _comparable(beat.text)
            ),
            None,
        )
        if key is None:
            unplaced.append(item)
        else:
            found.setdefault(key, []).append(item)
    order = [(scene.scene_id, beat.beat_id) for scene in narration.scenes for beat in scene.beats]
    return {key: found[key] for key in order if key in found}, unplaced


def splice_beats(narration: NarrationPlan, texts: dict[BeatKey, str]) -> NarrationPlan:
    """A copy of ``narration`` with the text of the beats in ``texts`` replaced.

    Every other field of a replaced beat is kept, so its id and cited facts
    cannot change. A scene with a replaced beat has its narration rebuilt from
    its beats, as the planner builds it; every other scene is left as it is.
    """

    scenes = []
    for scene in narration.scenes:
        if not any((scene.scene_id, beat.beat_id) in texts for beat in scene.beats):
            scenes.append(scene)
            continue
        beats = [
            beat.model_copy(update={"text": texts[(scene.scene_id, beat.beat_id)]})
            if (scene.scene_id, beat.beat_id) in texts else beat
            for beat in scene.beats
        ]
        scenes.append(scene.model_copy(update={
            "beats": beats, "narration": " ".join(beat.text for beat in beats),
        }))
    return narration.model_copy(update={"scenes": scenes})


def _comparable(text: str) -> str:
    return " ".join(text.lower().split()).strip(" \"'\u201c\u201d\u2018\u2019.!?,;:")


def _plan_summary(lesson: LessonPlan, narration: NarrationPlan) -> dict[str, Any]:
    words = {scene.scene_id: len(scene.narration.split()) for scene in narration.scenes}
    return {
        "central_question": lesson.central_question,
        "hook": lesson.hook,
        "promise": "",
        "takeaway": lesson.takeaway,
        "running_example": lesson.running_example.description,
        "motifs": [motif.motif_id for motif in lesson.motifs],
        "acts": [
            {
                "act_id": scene.scene_id,
                "kind": "build",
                "viewer_question": scene.viewer_question,
                "key_idea": scene.learning_goal,
                "words": words.get(scene.scene_id, 0),
            }
            for scene in lesson.scenes
        ],
    }


def _scenes(narration: NarrationPlan) -> list[dict[str, Any]]:
    scenes = []
    for scene in narration.scenes:
        words = len(scene.narration.split())
        scenes.append({
            "scene_id": scene.scene_id,
            "act_id": scene.scene_id,
            "scene_type": "manim",
            "seconds": round(words / scene.target_words_per_minute * 60, 1),
            "narration": scene.narration,
            "visual_hints": [beat.visual_action for beat in scene.beats],
            "primitives": [],
            "transformations": len(scene.beats),
            "plan_source": "planner",
        })
    return scenes


def _coverage(lesson: LessonPlan) -> dict[str, Any]:
    counts = {treatment: 0 for treatment in TREATMENTS}
    for entry in lesson.coverage:
        counts[entry.treatment] += 1
    return {
        "counts": counts,
        "decisions": [entry.model_dump(mode="json") for entry in lesson.coverage],
    }


# ── spoken voice ────────────────────────────────────────────────────────
# Patterns that make narration sound like a machine describing slides. Each
# finding becomes a flow break, so the targeted beat fix rewrites that beat.

_COLORS = r"(?:red|blue|green|gold|golden|yellow|teal|purple|orange|white|gr[ae]y|pink|maroon)"
_SHAPES = (
    r"(?:nodes?|arrows?|dots?|circles?|box(?:es)?|bars?|squares?|labels?|trees?|gauges?|"
    r"slots?|tiles?|bubbles?|triangles?|rectangles?|outlines?|badges?|cards?|tallies|tally)"
)
COLOR_SHAPE = re.compile(rf"\b{_COLORS}\b(?:[\s-]+[a-z]+){{0,2}}?[\s-]+{_SHAPES}\b", re.IGNORECASE)
POINTER = re.compile(
    r"^(?:now,?\s+|so,?\s+)?(?:look at|watch|notice|keep your eyes on|see how|observe|focus on|"
    r"compare)\b",
    re.IGNORECASE,
)
SUMMARY_CLOSE = re.compile(
    r"^(?:In [A-Z][\w-]*(?: [\w-]+){0,4},|By [a-z]+ing\b|This is (?:called|known as)\b)",
)
SUMMARY_MIN_WORDS = 15


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.?!])\s+", text.strip()) if part.strip()]


def voice_problems(narration: NarrationPlan) -> list[dict[str, str]]:
    """Sentences that sound written or machine-made rather than spoken, as flow breaks.

    Checks, per scene: describing the drawing by color and shape (the first
    one is allowed), more than one "look at"/"notice" pointer, a
    definition-shaped closing sentence, sentences too long to follow by ear,
    stock phrases, and two scenes in a row that open with a question.
    """

    found: list[dict[str, str]] = []

    def flag(scene_id: str, sentence: str, problem: str) -> None:
        found.append({"where": scene_id, "sentence": sentence, "problem": problem})

    previous_opened_with_question = False
    for scene in narration.scenes:
        sentences = [sentence for beat in scene.beats for sentence in _sentences(beat.text)]
        if not sentences:
            continue
        drawing, pointers = 0, 0
        for sentence in sentences:
            words = len(sentence.split())
            lowered = sentence.lower()
            if COLOR_SHAPE.search(sentence):
                drawing += 1
                if drawing > 1:
                    flag(scene.scene_id, sentence,
                         "It describes the drawing by color and shape; say what the thing is or "
                         "what it does instead.")
            if POINTER.search(sentence):
                pointers += 1
                if pointers > 1:
                    flag(scene.scene_id, sentence,
                         "Another 'look at / notice / watch' instruction; the animation already "
                         "shows where to look, so say what happens instead.")
            if words > MAX_SPOKEN_SENTENCE_WORDS:
                flag(scene.scene_id, sentence,
                     f"At {words} words it is too long to follow by ear; split it into shorter "
                     "sentences that follow one another.")
            phrase = next((item for item in STOCK_PHRASES if re.search(rf"\b{re.escape(item)}\b", lowered)), None)
            if phrase:
                flag(scene.scene_id, sentence,
                     f"The phrase '{phrase}' sounds machine-written; say it plainly.")
        closing = sentences[-1]
        if SUMMARY_CLOSE.search(closing) and len(closing.split()) >= SUMMARY_MIN_WORDS:
            flag(scene.scene_id, closing,
                 "It closes the scene by reciting a definition or summary, which sounds like a "
                 "textbook; land the point as a plain consequence of what was just shown.")
        opens_with_question = sentences[0].endswith("?") or (
            len(sentences) > 1 and sentences[1].endswith("?") and len(sentences[0].split()) < 12
        )
        if opens_with_question and previous_opened_with_question:
            flag(scene.scene_id, next(s for s in sentences[:2] if s.endswith("?")),
                 "The previous scene also opened with a question; open this one by picking up "
                 "the consequence of the last scene instead.")
        previous_opened_with_question = opens_with_question
    unique: list[dict[str, str]] = []
    for item in found:
        if not any(other["sentence"] == item["sentence"] for other in unique):
            unique.append(item)
        else:
            for other in unique:
                if other["sentence"] == item["sentence"] and item["problem"] not in other["problem"]:
                    other["problem"] += " " + item["problem"]
    return unique


def with_voice_check(review: ScriptReview, narration: NarrationPlan) -> ScriptReview:
    """``review`` with the narration's voice problems added as flow breaks.

    A script with any of them does not pass, so in revise mode the beats
    holding them are rewritten by the targeted beat fix.
    """

    problems = voice_problems(narration)
    if not problems:
        return review
    known = {item["sentence"] for item in review.flow_breaks}
    added = [dict(item, kind="voice") for item in problems if item["sentence"] not in known]
    return review.model_copy(update={
        "reviewer_passed": review.passed,
        "flow_breaks": review.flow_breaks + added,
        "improvement_suggestions": review.improvement_suggestions + [
            f'In {item["where"]}, "{item["sentence"]}": {item["problem"]}' for item in added
        ],
        "passed": review.passed and not added,
        "voice_problems": len(problems),
    })
