# WP3 — Review gates that measure what the viewer sees

You are implementing one work package in an existing Python repository (a PDF → Manim
explainer-video pipeline). Every design decision has already been made and is written
below. Implement exactly this; do not redesign, rename, or add features.

## Ground rules (binding)

- Working directory: the repository root you were started in. Edit only the files listed
  under "Files you may edit". Create only the files listed under "Files you create".
- **Frozen files — read them, never edit them:** everything under `src/prompts/`,
  `src/schemas/lesson_plan.py`, `src/utils/visual_plan_repair.py`,
  `src/utils/manim_engine_contract.py`.
- **No git commands of any kind.**
- No network or paid API calls. Tests are keyless: `tests/conftest.py` strips API keys.
- The only shell command you run is `python -m pytest ...`.
- Several files have mixed CRLF/LF line endings. Ignore that; edit normally.
- Never weaken, skip or delete an existing test to get to green. Where this spec changes
  behaviour on purpose it says so. If any other existing test fails, stop and report it.
- Match the surrounding code style.

## Background (what this package is for)

A rendered video in this project had six scenes whose only content was a narration
sentence as a title plus one arrow, diagrams covering under a fifth of the frame, and
labels on top of drawings. Every automatic check passed it: the storyboard grader
(smallest model, graded on "completeness") scored 7.3, the render-time layout audit
reported 0 issues because it only looks for text overlapping text, and the vision
reviewer was only asked which frame type suits the scene. This package makes the three
checks look at the things that were wrong.

Prompt text already exists in `src/prompts/review.py`: `storyboard_review_prompt`,
`PLANNED_CRITERIA_WEIGHTS`, `REPORTED_ONLY_CRITERION`, `frame_issue_instructions`,
`FRAME_ISSUE_CODES`; and `plan_summary_for_review` in `src/prompts/narrative.py`.

## Files you may edit

`src/agents/storyboard_grader.py`, `src/orchestrator/pipeline.py` (one call site),
`src/utils/layout_audit.py`, `src/agents/frame_verifier.py`.
Existing test files only as allowed by the ground rules.

## Files you create

`tests/test_review_gates.py`

---

## Edit 1 — `src/agents/storyboard_grader.py`: a rubric for planned videos

1. `GRADER_MODEL = "gpt-5.4-mini"` (was `"gpt-5.4-nano"`: the smallest model cannot judge
   a script). If an existing test asserts the old model name, update that expectation.
2. Imports:
   ```python
   from src.prompts.narrative import plan_summary_for_review
   from src.prompts.review import (
       PLANNED_CRITERIA_WEIGHTS, REPORTED_ONLY_CRITERION, storyboard_review_prompt,
   )
   ```
3. `StoryboardGraderInput.__init__` gains a last keyword parameter
   `lesson_plan: Optional[Any] = None`, stored as `self.lesson_plan = lesson_plan`
   (add `Any` to the typing import if missing).
4. `execute`: the branch that picks the grader becomes
   ```python
   plan = getattr(input_data, "lesson_plan", None)
   if self.client and plan is not None:
       report = self._grade_planned_with_llm(
           storyboard, plan, input_data.source_text_summary, input_data.source_segments,
       )
   elif self.client:
       report = self._grade_with_llm(...)        # unchanged call
   else:
       report = self._grade_heuristic(...)       # unchanged call
   ```
5. `_without_visuals`: drop both visual criteria:
   `kept = [c for c in report.criteria if c.name not in ("visual_richness", "visual_design")]`.
   The `total` it divides by must ignore zero-weight criteria as it does now (it sums
   weights; a weight-0 criterion adds nothing).
6. New method `_scene_for_review` (staticmethod):
   ```python
   @staticmethod
   def _scene_for_review(scene) -> Dict[str, Any]:
       beats = list(getattr(scene, "source_beats", None) or [])
       objects = list(scene.visual_intent.mathematical_objects)
       return {
           "scene_id": scene.scene_id,
           "act_id": next((getattr(b, "act_id", "") for b in beats if getattr(b, "act_id", "")), ""),
           "scene_type": getattr(scene, "scene_type", "manim"),
           "seconds": scene.duration_estimate,
           "narration": scene.narration,
           "visual_hints": [b.visual_hint for b in beats if (b.visual_hint or "").strip()],
           "primitives": [(o.primitive or o.object_type) for o in objects][:12],
           "transformations": len(scene.visual_intent.transformations),
           "plan_source": getattr(scene, "visual_plan_source", "planner"),
       }
   ```
7. New method `_grade_planned_with_llm(self, storyboard, plan, source_summary, source_segments=None) -> GradingReport`:
   ```python
   assert self.client is not None, "LLM client not initialized"
   prompt = storyboard_review_prompt(
       plan_summary_for_review(plan),
       [self._scene_for_review(s) for s in storyboard.scenes],
       plan.coverage_report(),
       source_summary,
   )
   try:
       response = self.client.chat.completions.create(
           model=GRADER_MODEL,
           messages=[{"role": "user", "content": prompt}],
           response_format={"type": "json_object"},
       )
       data = json.loads(response.choices[0].message.content)
       return self._parse_planned_response(data)
   except Exception as e:
       logger.error(f"LLM grading failed: {e} — falling back to heuristic.")
       return self._grade_heuristic(storyboard, source_segments=source_segments)
   ```
8. New method `_parse_planned_response(self, data: Dict[str, Any]) -> GradingReport`:
   - a local helper reads one criterion: `entry = data.get(name)`; if it is not a dict use
     `{}`; `score = float(entry.get("score", 5.0))` inside `try/except (TypeError, ValueError)`
     falling back to `5.0`, then clamped to `[0.0, 10.0]`; `feedback = str(entry.get("feedback") or "No feedback.")`;
   - for each `name, weight` in `PLANNED_CRITERIA_WEIGHTS.items()` append
     `GradingCriterion(name=name, weight=weight, score=score, feedback=feedback)` and add
     `score * weight` to the weighted sum;
   - then append the reported-only criterion
     `GradingCriterion(name=REPORTED_ONLY_CRITERION, weight=0.0, score=..., feedback=...)`;
     it does not enter the weighted sum;
   - `improvement_suggestions`: `data.get("improvement_suggestions")` if it is a list,
     keeping only non-empty strings, else `[]`;
   - return `GradingReport(overall_score=round(weighted_sum, 2), passed=False, criteria=criteria,
     summary=str(data.get("summary") or ""), improvement_suggestions=suggestions)`.
   The existing `_parse_llm_response`, `_grade_with_llm`, `_grade_heuristic` and
   `CRITERIA_WEIGHTS` are unchanged.

## Edit 2 — `src/orchestrator/pipeline.py`, `_run_script_architect`

In the `StoryboardGraderInput(...)` call add the argument
`lesson_plan=getattr(self.script_architect_agent, "last_lesson_plan", None),`.
Nothing else in the file changes.

## Edit 3 — `src/utils/layout_audit.py`: catch sentences on screen and under-filled frames

The audit runs inside the Manim subprocess after every `play`/`wait`. Keep `check()` and
its return value exactly as they are (a test compares it to a two-key dict).

1. Constants, next to `OVERLAP_RATIO`:
   ```python
   LONG_TEXT_WORDS = 7    # a Text with this many words is a sentence, not a label
   MIN_FILL = 0.30        # share of the frame the picture should reach at least once
   MIN_FILL_OBSERVATIONS = 3
   _PROSE_CLASSES = ("Text", "MarkupText", "Paragraph", "Title", "BulletedList")
   ```
2. New functions (module level, after `visible_texts`):
   ```python
   def _raw_text(mob) -> str:
       raw = getattr(mob, "original_text", None) or getattr(mob, "text", None) or ""
       return " ".join(str(raw).split())


   def long_texts(mobjects) -> List[Dict[str, Any]]:
       """Visible prose mobjects (not formulas) long enough to be a sentence."""
       found = []
       for mob in visible_texts(mobjects):
           if not any(cls.__name__ in _PROSE_CLASSES for cls in type(mob).__mro__):
               continue
           words = len(_raw_text(mob).split())
           if words >= LONG_TEXT_WORDS:
               found.append({"text": _label(mob), "words": words})
       return found


   def content_fill(mobjects, frame_width: float, frame_height: float) -> float:
       """Share of the frame covered by the bounding box of everything visible (0..1)."""
       boxes = [_box(m) for m in mobjects if _visible(m)]
       boxes = [b for b in boxes if b is not None]
       if not boxes or frame_width <= 0 or frame_height <= 0:
           return 0.0
       hw, hh = frame_width / 2, frame_height / 2
       left = max(-hw, min(b[0] for b in boxes))
       right = min(hw, max(b[1] for b in boxes))
       bottom = max(-hh, min(b[2] for b in boxes))
       top = min(hh, max(b[3] for b in boxes))
       if right <= left or top <= bottom:
           return 0.0
       return round((right - left) * (top - bottom) / (frame_width * frame_height), 3)
   ```
3. `_Recorder.__init__`: add `self.long_text: Dict[str, Dict[str, Any]] = {}`,
   `self.max_fill = 0.0`, `self.observations = 0`.
4. `_Recorder.observe`: inside the existing `try:` (after `found = check(...)`) also compute
   `long = long_texts(scene.mobjects)` and
   `fill = content_fill(scene.mobjects, config.frame_width, config.frame_height)`.
   After `t` is computed: `self.observations += 1`; `self.max_fill = max(self.max_fill, fill)`;
   `for item in long: self.long_text.setdefault(item["text"], {**item, "t": t})`.
5. `_Recorder.dump`: the JSON gains two keys:
   `"long_text": list(self.long_text.values())[:MAX_FINDINGS]` and
   `"fill": {"max": round(self.max_fill, 3), "observations": self.observations}`.
6. `read_audit`: after the two existing lists are built, append
   ```python
   issues += [
       f"Text \"{o['text']}\" is a {o.get('words', '?')}-word sentence on screen at "
       f"{o.get('t', 0)}s; on-screen text must be a label of at most four words."
       for o in data.get("long_text") or []
   ]
   fill = data.get("fill")
   if (
       isinstance(fill, dict)
       and fill.get("observations", 0) >= MIN_FILL_OBSERVATIONS
       and float(fill.get("max", 1.0)) < MIN_FILL
   ):
       issues.append(
           f"The picture never covers more than {int(float(fill['max']) * 100)}% of the frame; "
           "the main visual must be much larger."
       )
   ```
   An audit file written before this change (no `long_text`, no `fill`) yields exactly
   the issues it yields today.
7. `audit_feedback`: append two more strings to the returned list, after the existing
   last one:
   ```python
   "Remove every sentence, title or caption from the screen: the narrator says it. "
   "On-screen text is a label of at most four words, or a formula.",
   "Make the main visual fill the frame: a coordinate system with x_length 9 to 11 and "
   "y_length 5.5 to 6.5, or scale the main group with scale_to_fit_height(5) and centre it.",
   ```
   If an existing test asserts the exact length or last element of this list, update
   that expectation.
8. Update the module docstring: the audit now also records sentences shown as text and
   how much of the frame the picture reaches.

## Edit 4 — `src/agents/frame_verifier.py`: report visible defects

1. Import `from src.prompts.review import FRAME_ISSUE_CODES, frame_issue_instructions`.
2. `FrameVerdict` gains a last field `issues: List[str] = field(default_factory=list)`
   (import `field` from `dataclasses` if it is not imported).
3. In `_review_one`, the single-scene branch only (the second `prompt = (...)`, the one
   whose text contains `"The images are stills of the scene as rendered"`): append
   `+ frame_issue_instructions()` to that prompt expression. After `data` is returned:
   ```python
   issues = [str(c) for c in (data.get("issues") or []) if str(c) in FRAME_ISSUE_CODES]
   ```
   and pass `issues=issues` to the `FrameVerdict(...)` that branch returns. The A/B
   comparison branch is unchanged.
4. `verdicts_report`: add the key
   `"issues": {v.scene_id: v.issues for v in verdicts if v.issues},`.

## Tests — `tests/test_review_gates.py` (new)

1. `test_planned_rubric_parses_and_weights` — `StoryboardGraderAgent.__new__(StoryboardGraderAgent)._parse_planned_response`
   on `{"learning_path": {"score": 8, "feedback": "a"}, "narration_economy": {"score": 6},
   "visual_design": {"score": "bad"}, "hook_and_payoff": {"score": 14}, "source_coverage":
   {"score": 3, "feedback": "c"}, "summary": "s", "improvement_suggestions": ["x", "", 3]}`
   (`accuracy` is missing on purpose). Expected scores: learning_path 8, narration_economy 6,
   visual_design 5 (unparseable), hook_and_payoff 10 (clamped), accuracy 5 (missing).
   Overall equals `round(sum(expected[name] * weight for name, weight in PLANNED_CRITERIA_WEIGHTS.items()), 2)`
   (import the weights; do not hard-code them); criteria names are
   `list(PLANNED_CRITERIA_WEIGHTS)` followed by `"source_coverage"` with weight `0.0` and
   score `3.0`; suggestions equal `["x"]`.
2. `test_planned_grading_uses_the_plan_prompt` — an agent with a fake `client` whose
   `chat.completions.create` records `messages` and `model` and returns a JSON string with
   all criteria at 8; input built with `lesson_plan=<LessonPlan>` (build one with
   `LessonPlan.from_llm` as in `tests/test_lesson_plan.py`) and a small storyboard (reuse
   a builder from `tests/test_storyboard_grader_coverage.py`). Assert: the output is a
   `StoryboardGraderOutput`; the recorded model is `"gpt-5.4-mini"`; the prompt contains
   `"learning_path"`, the plan's central question, and the first scene's full narration.
3. `test_unplanned_grading_is_unchanged` — same fake client, no `lesson_plan`: the
   recorded prompt contains `"completeness"` and not `"learning_path"`.
4. `test_planned_grading_falls_back_to_heuristic_on_error` — fake client raises: the
   report's criteria names are the four legacy ones.
5. `test_without_visuals_drops_visual_design` — a planned report through
   `_without_visuals` has no `"visual_design"` criterion and an overall equal to the
   weighted mean of the remaining weighted criteria.
6. `test_scene_for_review_fields` — a scene with two beats (act ids set, one empty
   visual hint) and `visual_plan_source="fallback"`: the dict has the full narration, one
   visual hint, `plan_source == "fallback"` and the right `act_id`.
7. `test_long_texts_flags_sentences_not_labels_or_formulas` — with real manim objects
   (follow the existing layout-audit tests in `tests/test_gemini_presenter_and_layout.py`
   for how they are built and skipped when LaTeX is missing): a `Text` of nine words is
   reported with `words == 9`; a `Text("top singular vector")` is not; a long `MathTex`
   (only if LaTeX is available, else leave it out of the test) is not.
8. `test_content_fill` — a single `Rectangle(width=7.11, height=4.0)` at the origin on a
   14.22 x 8.0 frame gives a fill between 0.24 and 0.26; adding a
   `Rectangle(width=14.22, height=8.0)` gives 1.0; an empty list gives 0.0; a rectangle
   placed mostly outside the frame is clipped (fill <= 1.0).
9. `test_read_audit_reports_sentences_and_underfill` — write a JSON file with
   `{"overlaps": [], "off_frame": [], "long_text": [{"text": "That is enough to
   reconstruct A", "words": 10, "t": 1.0}], "fill": {"max": 0.18, "observations": 12}}`:
   `read_audit` returns two issues, one containing `"10-word sentence"` and one containing
   `"18% of the frame"`. With `"fill": {"max": 0.18, "observations": 2}` the fill issue is
   absent. With `"fill": {"max": 0.6, "observations": 12}` it is absent.
10. `test_read_audit_old_files_unchanged` — a JSON file with only `overlaps` and
    `off_frame` (one overlap) yields exactly one issue.
11. `test_audit_feedback_mentions_labels_and_fill` — `audit_feedback(["x"])` contains a
    string with `"at most four words"` and one with `"fill the frame"`.
12. `test_frame_verifier_collects_issue_codes` — `FrameVerifierAgent(vision=fake)` where
    `fake` records the prompt and returns `{"easiest": "manim", "legible": True, "reason":
    "r", "issues": ["underfilled_frame", "made_up_code"]}`; monkeypatch
    `src.agents.frame_verifier.sample_frames` to return `["f.png"]` so no ffmpeg runs.
    Review one scene `{"scene_id": "scene_001", "frame": "manim", "video": "v.mp4",
    "narration": "n"}`: the verdict's `issues == ["underfilled_frame"]`; the prompt
    contains `"underfilled_frame"`, `"Manim is for VISUALIZATION"` and
    `"EASIEST TO UNDERSTAND"`; `verdicts_report([verdict])["issues"] == {"scene_001":
    ["underfilled_frame"]}`.

## Gates (run them, report each result in one line)

- G1 new tests — CHECK: `python -m pytest tests/test_review_gates.py -q -p no:cacheprovider` — EXPECT: `12 passed` (a LaTeX-dependent part may be left out of test 7, not skipped as a whole)
- G2 neighbouring suites — CHECK:
  `python -m pytest tests/test_storyboard_grader_coverage.py tests/test_gemini_presenter_and_layout.py tests/test_hybrid.py tests/test_lesson_plan.py tests/test_visual_plan_repair.py -q -p no:cacheprovider`
  — EXPECT: `0 failed`
- G3 full suite, once, at the end — CHECK: `python -m pytest tests -q -p no:cacheprovider`
  — EXPECT: `0 failed`

## Report (your final message, under 300 words)

1. Files changed, one line each.
2. Each gate's result line.
3. Any existing test whose expectation you changed, and why.
4. Anything in this spec that did not match the code you found, and what you did about it.
