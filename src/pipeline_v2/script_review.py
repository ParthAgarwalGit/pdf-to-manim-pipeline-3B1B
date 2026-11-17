"""Pre-TTS review of the planned lesson and its narration."""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.prompts.review import (
    PLANNED_CRITERIA_WEIGHTS,
    REPORTED_ONLY_CRITERION,
    storyboard_review_prompt,
)

from .model_config import TEXT_MODEL
from .models import LessonPlan, NarrationPlan, SourceDocument
from .prompt_text import FLOW_PASS_SCORE, REVIEW_PASS_SCORE
from .providers import clean_json_text
from .state import BudgetExceeded


SCRIPT_REVIEW_PROMPT_VERSION = "script-review-2-flow"
SOURCE_SUMMARY_CHARS = 3000
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
    attempt: int = 1

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
) -> Optional[ScriptReview]:
    """Grade the planned script; ``None`` (with an event) when the review could not run."""

    task_id = f"script-review-{attempt}"
    prompt = storyboard_review_prompt(
        _plan_summary(lesson, narration),
        _scenes(narration),
        _coverage(lesson),
        "\n".join(fact.text for fact in source.facts)[:SOURCE_SUMMARY_CHARS],
    )
    try:
        response = await router.text(
            stage="script_review", task_id=task_id, provider="google",
            model=TEXT_MODEL,
            system="You are a demanding editor of short educational videos.",
            prompt=prompt, max_output_tokens=3000,
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
