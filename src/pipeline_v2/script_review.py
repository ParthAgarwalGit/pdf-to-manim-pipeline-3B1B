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
from .prompt_text import REVIEW_PASS_SCORE
from .providers import clean_json_text
from .state import BudgetExceeded


SCRIPT_REVIEW_PROMPT_VERSION = "script-review-1"
SOURCE_SUMMARY_CHARS = 3000
TREATMENTS = ("covered", "compressed", "deferred", "omitted")


class ScriptReview(BaseModel):
    overall_score: float
    passed: bool
    criteria: dict[str, dict[str, Any]]  # name -> {"score", "feedback", "weight"}
    summary: str = ""
    improvement_suggestions: list[str] = Field(default_factory=list)
    attempt: int = 1

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
    overall = round(weighted_sum, 2)
    return ScriptReview(
        overall_score=overall,
        passed=overall >= REVIEW_PASS_SCORE,
        criteria=criteria,
        summary=str(data.get("summary") or ""),
        improvement_suggestions=suggestions,
        attempt=attempt,
    )


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
