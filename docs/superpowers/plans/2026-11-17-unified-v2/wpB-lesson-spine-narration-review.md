# WPB — Lesson spine, spoken-narration rules, script review and code-gen style in V2

You are implementing one work package in this repository. The decisions are made and all
model-facing text is already written in `src/pipeline_v2/prompt_text.py` (frozen) and
`src/prompts/review.py` (frozen). Your job is the wiring, schema fields and tests.

## Ground rules (binding)

- Work only in this checkout on its current branch. **No git commands.**
- **Frozen, read-only:** `src/pipeline_v2/prompt_text.py`, `src/pipeline_v2/model_config.py`,
  everything under `src/prompts/`.
- No paid API calls; tests are keyless with mocked routers.
- Only shell command: `python -m pytest ... -p no:cacheprovider --basetemp=.pytest_tmp/wpb...`.
- Never weaken, skip or delete a test to get green; update an expectation only when this spec
  changes the behaviour on purpose, and list it.
- Old `run.db` task payloads and `storyboard.json` files must still validate: every new model
  field has a default.
- Known pre-existing failure to leave alone:
  `tests/test_v2_delivery.py::TestRealMedia::test_mixed_clips_join_into_one_uniform_file`.

## Background

V2 plans a lesson (`LessonPlanner.lesson`), writes narration in batches
(`LessonPlanner.narration`), records TTS, then generates raw Manim per scene
(`generation.generation_request`). The user's earlier pipeline found that the strongest lever on
engagement was a *lesson spine* decided before any narration: one central question, a hook, a
running example with real numbers, 2-4 recurring visual motifs, scenes in viewing order (each
answering the question the previous scene raised), and an explicit coverage decision per page.
It also found narration needed explicit spoken-style and economy rules, and that a pre-render
script review caught overstatements and repetition. This package brings those into V2.

## B1. Schema — `src/pipeline_v2/models.py`

Add (all optional / defaulted):

```python
class RunningExample(BaseModel):
    description: str = ""
    values: dict[str, str] = Field(default_factory=dict)

class VisualMotif(BaseModel):
    motif_id: str
    what: str = ""
    color: str = ""
    stands_for: str = ""

class PageCoverage(BaseModel):
    page_number: int = Field(ge=1)
    treatment: Literal["covered", "compressed", "deferred", "omitted"] = "covered"
    reason: str = ""
```

- `LessonScene` gains `viewer_question: str = ""`.
- `LessonPlan` gains `central_question: str = ""`, `hook: str = ""`, `takeaway: str = ""`,
  `running_example: RunningExample = Field(default_factory=RunningExample)`,
  `motifs: list[VisualMotif] = Field(default_factory=list, max_length=6)`,
  `coverage: list[PageCoverage] = Field(default_factory=list)`.
- Add a method `LessonPlan.visual_world() -> list[str]` returning one line per fact, in this
  exact format (empty list when there is no running example and no motif):
  `"Running example: {description}"`, then `"Value: {name} = {value}"` per value, then per motif
  `"Motif {motif_id}: {what}; color {color}; stands for {stands_for}"` (omit a `; ...` part whose
  value is empty).
- Add `LessonPlan.spine() -> dict` returning
  `{"central_question", "hook", "takeaway", "running_example": {...}, "motifs": [...]}` (plain
  JSON-ready values).

The Google structured-output schema is generated from these models; make sure
`_google_compatible_schema` in `providers.py` still accepts them (it may need `dict[str, str]`
support — check with a test that builds the schema for `LessonPlan`).

## B2. Lesson prompt — `src/pipeline_v2/planning.py`, `LessonPlanner.lesson`

`LessonPlanner.__init__` gains `presenter_opens: bool = False` (stored). Today the prompt is
`PROMPT_VERSION line + "Create a source-grounded educational lesson plan. " + scene_guidance +
"Merge adjacent scenes ... Set delivery_mode to manim.\n\n" + facts JSON`, then the directive
block. Replace the instructional text (everything between the PROMPT_VERSION line and the facts
JSON) with:

```python
lesson_instructions(scene_guidance, self.presenter_opens) + "\n\n"
```

Keep `scene_guidance` computed exactly as today (both branches), the facts JSON payload, the
directive block, the retry-for-scene-floor logic and the retry prompt suffix unchanged.
Add `"page_count": source.page_count` to the facts JSON object so coverage can name pages.
Bump `PROMPT_VERSION` (new value `"v2.5.0-spine"`).

After the lesson returns (both the first call and the expansion retry), normalise in a small
helper `_normalise_spine(lesson, source)`:
- drop `coverage` entries whose page is outside `1..source.page_count`, keep the first entry per
  page, and add `{"page_number": p, "treatment": "covered", "reason": "Cited by a scene."}` for
  every page cited by any scene's `source_fact_ids` that has no entry, and
  `{"treatment": "omitted", "reason": "The lesson plan did not place this page."}` for any other
  page with no entry;
- give an `omitted`/`deferred`/`compressed` entry with an empty reason the reason
  `"No reason given by the planner."`;
- keep at most 4 motifs (first ones), uppercase motif colors and replace spaces with `_`.
The fallback lesson (`_fallback_lesson`) keeps empty spine fields.

## B3. Narration prompt — `LessonPlanner.narration`

- Replace the instructional text of the narration prompt (from `"Write the requested
  consecutive section..."` up to and including `"...computes them after TTS.\n\n"`) with
  `narration_instructions(pace_guidance, target_wpm, self.presenter_opens) + "\n\n"`. Keep
  `pace_guidance` and `target_wpm` as computed today.
- Add to the payload: `"lesson_spine": lesson.spine()`, and add `"viewer_question":
  item.viewer_question` to each `global_scene_outline` entry. `scenes_to_write` already carries
  full `LessonScene` dumps.
- Bump `NARRATION_PROMPT_VERSION` (new value `"v2.6.0-spoken"`).
- After a batch returns, pass every beat's `text` through `spoken_plain` from
  `src/prompts/narrative.py` (turns a leaked `best_fit_line` into `best fit line`) and rebuild
  each scene's `narration` field from its beats the same way the code already builds it (find how
  `NarratedScene.narration` relates to beats in planning/fallback code and keep it consistent).

## B4. Script review — new file `src/pipeline_v2/script_review.py`

A pre-TTS review of the planned lesson and narration, using the frozen
`src/prompts/review.py::storyboard_review_prompt` and `PLANNED_CRITERIA_WEIGHTS`,
`REPORTED_ONLY_CRITERION`, and `REVIEW_PASS_SCORE` from `prompt_text.py`.

```python
class ScriptReview(BaseModel):
    overall_score: float
    passed: bool
    criteria: dict[str, dict[str, Any]]   # name -> {"score": float, "feedback": str, "weight": float}
    summary: str = ""
    improvement_suggestions: list[str] = Field(default_factory=list)
    attempt: int = 1

async def review_script(router, lesson: LessonPlan, narration: NarrationPlan,
                        source: SourceDocument, attempt: int = 1) -> ScriptReview
```

- Build the prompt with `storyboard_review_prompt(plan_summary, scenes, coverage, source_summary)`:
  - `plan_summary` = `{"central_question", "hook", "promise": "", "takeaway",
    "running_example": lesson.running_example.description,
    "motifs": [m.motif_id ...], "acts": [{"act_id": s.scene_id, "kind": "build",
    "viewer_question": s.viewer_question, "key_idea": s.learning_goal, "words": len(words)} ...]}`
  - `scenes` = one dict per narrated scene: `scene_id`, `act_id` (= scene_id), `scene_type`:
    `"manim"`, `seconds` (words / target_wpm * 60, rounded to 1 decimal), `narration` (full),
    `visual_hints` (each beat's `visual_action`), `primitives`: `[]`,
    `transformations`: number of beats, `plan_source`: `"planner"`.
  - `coverage` = `{"counts": {...per treatment...}, "decisions": [c.model_dump() ...]}`
  - `source_summary` = the first 3000 characters of the concatenated fact texts.
- Call `router.text(...)` (or `structured` with a small pydantic model if that is the router's
  JSON path; read `providers.py` and use whichever returns parsed JSON with budget reservation)
  with `stage="script_review"`, `task_id=f"script-review-{attempt}"`, `provider="google"`,
  `model=TEXT_MODEL`, `system="You are a demanding editor of short educational videos."`,
  `max_output_tokens=3000`.
- Parse like the legacy grader: each weighted criterion's score is clamped to 0..10 (default 5.0
  when missing or unparseable); `overall_score` = sum(score*weight) over
  `PLANNED_CRITERIA_WEIGHTS`, rounded to 2; `source_coverage` recorded with weight 0;
  `passed = overall_score >= REVIEW_PASS_SCORE`; suggestions = non-empty strings only.
- Keyless or provider failure: return `None` and emit an event `script_review_skipped` with the
  reason (never fail the run because the review could not run).

Wire it in `pipeline.py` right after narration is validated and the narration directives are
enforced (after the block that ends with `compliance.extend(narration_checks)`). It runs in
dry runs too: reviewing the script before any rendering is the point of a dry run.
- New config field `script_review: Literal["off", "report", "revise"] = "revise"` and CLI option
  `--script-review [off|report|revise]` (default `revise`).
- `report`: run once, write `script_review.json` (list of attempts) via
  `self.artifacts.write_json`, add an event.
- `revise`: if the first review did not pass, re-run `planner.narration(lesson, source)` once
  with the review's `improvement_suggestions` prefixed to the narration prompt as a block:
  `"A REVIEWER REJECTED THE PREVIOUS SCRIPT. Fix every point:\n- ...\n"` (add an optional
  `review_feedback: list[str]` parameter to `LessonPlanner.narration` that inserts this block
  right after the PROMPT_VERSION line), review again, and keep whichever narration scored
  higher. Both attempts go into `script_review.json`. Run the narration-directive enforcement
  on the replacement too.
- Use the pipeline's `_task(...)` checkpointing for the review calls so a resumed run does not
  pay twice (payload keys: `{"narration": ..., "lesson": ..., "attempt": n, "model": TEXT_MODEL,
  "prompt_version": "script-review-1"}`).
- Show the final score and the weakest criterion in `report.html` (find where `report.py`
  renders run-level information and add one short section; keep it small).

## B5. Code-generation style and visual world — `src/pipeline_v2/generation.py`

- Append `CODEGEN_STYLE_RULES` (from `prompt_text.py`) to the end of the `system` string in
  `generation_request`, and also to the `system` strings of `repair` and `revise` if they restate
  the creative rules (read them; if they only describe the repair task, append it there too so a
  repair does not undo the style).
- `generation_request` gains `visual_world: list[str] | None = None`; when non-empty, the payload
  gets `"visual_world": visual_world`. The pipeline passes `lesson.visual_world()` wherever it
  builds the generation request (find the call sites; retry/revise paths too).
- Bump `CODE_PROMPT_VERSION` (append `-style`).

## B6. Presenter flag plumbing (behaviour comes in a later package)

`PipelineV2Config` gains `presenter: Literal["off", "gemini", "d-id"] = "off"`. Pass
`presenter_opens = config.presenter != "off"` into `LessonPlanner`. No CLI option yet.

## Tests — new `tests/test_v2_lesson_spine.py`

1. `test_lesson_plan_spine_fields_default_and_round_trip` — an old-shape lesson dict (no spine
   keys) validates; a full one round-trips; `visual_world()` lines match the exact format;
   `spine()` keys.
2. `test_google_schema_accepts_lesson_plan` — the router's schema conversion for `LessonPlan`
   does not raise and includes `central_question` and `motifs`.
3. `test_lesson_prompt_uses_spine_instructions` — mocked router records the prompt: contains
   `"ONE CENTRAL QUESTION"`, `"page_count"`, the directive block when directives are set, and the
   new `PROMPT_VERSION`; with `presenter_opens=True` it contains `"presenter speaks"`.
4. `test_spine_normalisation_fills_coverage` — a lesson citing facts on pages 1 and 3 of a
   4-page source with coverage only for page 3 ("omitted", empty reason) ends with 4 entries:
   1 covered, 2 omitted (not placed), 3 omitted ("No reason given by the planner."), 4 omitted.
5. `test_narration_prompt_has_spoken_rules_and_spine` — prompt contains `"SPOKEN NARRATION"`,
   `"lesson_spine"`, the running example description, and each outline entry has
   `viewer_question`.
6. `test_narration_strips_snake_case_ids` — mocked narration returns a beat with
   `"the best_fit_line moves"`; result beat text is `"the best fit line moves"`.
7. `test_script_review_scores_and_parses` — mocked router returns a JSON with criteria; overall
   equals the weighted sum; `passed` against `REVIEW_PASS_SCORE`; coverage weight 0.
8. `test_script_review_revise_keeps_better_attempt` — pipeline-level or planner-level test with a
   mocked router: first review 5.0 → narration re-run with the feedback block in its prompt →
   second review 7.5 → the second narration is used and `script_review.json` has two attempts.
   A second variant where the re-review scores lower keeps the first narration.
9. `test_script_review_keyless_is_skipped` — no key: review returns `None`, run continues, event
   `script_review_skipped`.
10. `test_codegen_request_has_style_rules_and_visual_world` — `generation_request(...,
    visual_world=["Motif m: dots"])`: system ends with the style rules, payload has
    `visual_world`; without it the payload has no `visual_world` key.
11. `test_cli_script_review_option` — `--script-review report` reaches the config (CliRunner,
    `PipelineV2` patched).

## Gates

- G1 — CHECK: `python -m pytest tests/test_v2_lesson_spine.py -q -p no:cacheprovider --basetemp=.pytest_tmp/wpb1` — EXPECT: `11 passed` (or more if you split a test)
- G2 — CHECK: `python -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp/wpb2` — EXPECT: only the one known failure.

## Report (final message, under 300 words)

Files changed; gate lines; expectations you changed and why; where the code differed from this
spec and what you did.
