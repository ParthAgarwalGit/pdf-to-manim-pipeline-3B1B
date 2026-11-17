"""Prompts for the reviewers: the storyboard grader (before rendering) and
the frame reviewer (after rendering)."""

import json
from typing import Any, Dict, List

# Criteria the planned-video rubric scores, with their weights (sum to 1.0).
# ``source_coverage`` is reported alongside them and never gates a run
# (coverage posture "Hybrid C": coverage is an audit, not a layout driver).
PLANNED_CRITERIA_WEIGHTS: Dict[str, float] = {
    "learning_path": 0.30,
    "accuracy": 0.20,
    "narration_economy": 0.20,
    "visual_design": 0.20,
    "hook_and_payoff": 0.10,
}
REPORTED_ONLY_CRITERION = "source_coverage"

# Defects a vision reviewer may report on rendered stills.
FRAME_ISSUE_CODES = (
    "underfilled_frame",
    "text_overlap",
    "narration_as_text",
    "illegible_text",
    "off_frame",
    "mismatch",
)


def storyboard_review_prompt(
    plan_summary: Dict[str, Any],
    scenes: List[Dict[str, Any]],
    coverage: Dict[str, Any],
    source_summary: str = "",
) -> str:
    """Grade a planned video's script and visual plan before anything is rendered.

    ``scenes`` items: scene_id, act_id, narration (full text), visual_hints
    (list), primitives (list of planned primitive names), transformations
    (count), plan_source, seconds.
    """
    return f"""You are the editor who decides whether a short animated explainer is ready to be rendered. Be demanding. The bar is a video a curious viewer would watch to the end and could explain to a friend afterwards.

LESSON PLAN:
{json.dumps(plan_summary, indent=2, ensure_ascii=False)}

SCRIPT AND VISUAL PLAN, scene by scene:
{json.dumps(scenes, indent=2, ensure_ascii=False)}

WHAT THE PLAN DID WITH EACH SOURCE SEGMENT:
{json.dumps(coverage, indent=2, ensure_ascii=False)}

SOURCE SUMMARY:
{source_summary or "Not provided."}

Score each criterion from 0 to 10. 7 means good enough to render; 9 means excellent; 5 or below means it must be rewritten.

1. learning_path. The video answers ONE question. Each act answers a question the previous one raised. Ideas are seen happening to the running example before they are named or written as formulas. One new idea at a time. Take points off for: a tour of the document in its original order, definitions before any motivation, an act that walks through proof steps, a general statement with no concrete case, a concept used before it is introduced.

2. accuracy. Every statement in the narration is true as stated, for the running example and in general, and agrees with the source summary. Check each claim that connects two pictures or two quantities ("this is exactly that", "the same direction", "equals"). A single false or overstated claim caps this score at 4; quote it in the feedback and in an improvement suggestion.

3. narration_economy. Every sentence tells the viewer something new or tells them where to look. Take points off for: saying the same thing twice in different words (an ending that restates its answer two or three times counts), announcing what is about to be explained, recapping what was just shown, beats longer than 55 words, stock phrases ("let's dive in", "it turns out", "here's the key", "in this video"), sentences that read like a textbook, narration that reads out what the picture already shows.

4. visual_design. Every animated scene has a non-text main visual that changes while the narrator speaks, and reuses the plan's motifs. Take points off for: a scene whose plan_source is "fallback", a scene with no transformations, a scene whose planned objects are mostly Text, visual hints that ask for a sentence or title on screen, visual hints that cannot be drawn with simple shapes, a new unrelated picture in every scene.

5. hook_and_payoff. The first twenty seconds put a specific question in the viewer's head using a concrete picture, with no greeting and no list of topics. The last scene answers that question and stops. Take points off for a generic opening, an ending that trails off into a side topic, or a sign-off.

Also report source_coverage from 0 to 10: how much of the source's main line of argument the video conveys, given its length. This number is reported only. A short video that deliberately leaves out side results should not lose points on the other criteria for it.

Each improvement suggestion must be specific and addressed to the writer: name the scene or act and say what to change.

Return ONLY valid JSON in this shape:
{{
  "learning_path": {{"score": 0, "feedback": "one or two sentences"}},
  "accuracy": {{"score": 0, "feedback": "one or two sentences; quote any false claim"}},
  "narration_economy": {{"score": 0, "feedback": "one or two sentences"}},
  "visual_design": {{"score": 0, "feedback": "one or two sentences"}},
  "hook_and_payoff": {{"score": 0, "feedback": "one or two sentences"}},
  "source_coverage": {{"score": 0, "feedback": "one sentence"}},
  "summary": "one sentence overall",
  "improvement_suggestions": ["...", "..."]
}}
"""


def frame_issue_instructions() -> str:
    """Appended to a vision reviewer's prompt: report visible defects by code."""
    return (
        "\nAlso list the visual defects you can see in the stills, using only these codes:\n"
        "- underfilled_frame: the drawing takes up less than about a third of the frame and "
        "the rest is empty.\n"
        "- text_overlap: text overlaps other text or sits on top of a drawing so that either "
        "is hard to read.\n"
        "- narration_as_text: a full sentence, title or caption on screen repeats what the "
        "narrator is saying.\n"
        "- illegible_text: text is too small or too faint to read.\n"
        "- off_frame: something is cut off by the edge of the frame.\n"
        "- mismatch: what is drawn does not show what the narration describes.\n"
        "Add \"issues\": [\"code\", ...] to your JSON answer; use [] when there are none.\n"
    )
