# WP1 — Lesson plan, plan-driven script, act-aware scenes, plan-aware bridges

You are implementing one work package in an existing Python repository (a PDF → Manim
explainer-video pipeline). Every design decision has already been made and is written
below. Implement exactly this; do not redesign, rename, or add features.

## Ground rules (binding)

- Working directory: the repository root you were started in. Edit only the files listed
  under "Files you may edit". Create only the files listed under "Files you create".
- **Frozen files — read them, never edit them:** everything under `src/prompts/`,
  `src/schemas/lesson_plan.py`, `src/utils/visual_plan_repair.py`,
  `src/utils/manim_engine_contract.py`. They are finished and hand-written.
- **No git commands of any kind** (no commit, add, stash, checkout, branch, reset, push).
  The tree has uncommitted work that must stay exactly as it is apart from your edits.
- No network or paid API calls. Tests are keyless: `tests/conftest.py` strips API keys.
- The only shell command you run is `python -m pytest ...`.
- Several files have mixed CRLF/LF line endings. Ignore that; edit normally. The
  orchestrator restores per-line endings afterwards.
- Never weaken, skip or delete an existing test to get to green. If an existing test
  fails because of behaviour this spec changes on purpose, update that test's expectation
  and list it in your report. If one fails for any other reason, stop and report it.
- Keep every change additive for old data: new schema fields have defaults; a
  `lesson_script.json` or `storyboard.json` written before this change must still load.
- Match the surrounding code style (type hints, short docstrings that say why, no
  comments that restate the code).

## Background (what this package is for)

Today `ScriptArchitectAgent._generate_video_script` always takes the segmented path: one
small-model call per source segment, in document order, each told to "explain THIS
segment". The result is a tour of the PDF with no hook, no single question and no ending,
and proof segments get narrated even when the brief says to skip proofs.

This package adds a planning step. One strong-model call reads the whole source and
returns a `LessonPlan` (`src/schemas/lesson_plan.py`: central question, hook, running
example, motifs, acts in viewing order, and a coverage decision for every segment). A
second strong-model call writes the whole script from that plan. Scenes then follow the
plan's acts instead of the PDF's segments. When planning is off, fails, or the source is
too long for one call, the existing segmented path runs unchanged.

The prompt text for both calls already exists in `src/prompts/narrative.py`
(`lesson_plan_prompt`, `script_from_plan_prompt`, `bridge_slot_line`, `bridge_prompt`,
`SPOKEN_STYLE_RULES`, `TYPED_BEAT_PROMPT_RULES`). You only call these functions.

## Files you may edit

`src/schemas/video_script.py`, `src/schemas/storyboard.py`,
`src/agents/script_architect.py`, `src/agents/bridge_planner.py`,
`src/agents/script_editor.py`, `src/orchestrator/pipeline.py`,
`src/orchestrator/revision.py`, `src/cli.py`.
Existing test files only as allowed by the ground rules.

## Files you create

`tests/test_lesson_plan.py`

---

## Edit 1 — `src/schemas/video_script.py`

1. Add `from src.schemas.lesson_plan import LessonPlan` to the imports.
2. In `class Beat`, after `source_segment_id`, add:
   ```python
   act_id: str = Field(
       default="",
       description="Lesson-plan act this beat belongs to ('' for scripts written without a plan)."
   )
   ```
3. In `class VideoScript`, after `pronunciations`, add:
   ```python
   plan: Optional[LessonPlan] = Field(
       default=None,
       description="The lesson plan this script was written from; None for unplanned scripts."
   )
   ```
Do not touch `_BEAT_CONTENT_FIELDS` in `src/utils/fingerprints.py`: `act_id` is a tag and
must not change any fingerprint.

## Edit 2 — `src/schemas/storyboard.py`, `class Configuration`

After `presenter_bridges`, add two fields:
```python
lesson_plan: bool = Field(
    default=True,
    description="Plan the lesson (question, acts, running example) before scripting.",
)
visual_world: List[str] = Field(
    default_factory=list,
    description="The video's recurring look (running example, motifs), one fact per line, "
                "for the visual planner and the code generator. Set by the script architect.",
)
```

## Edit 3 — `src/agents/script_architect.py`

### 3a. Imports and the two prompt constants
- Add:
  ```python
  from src.prompts.narrative import (
      SPOKEN_STYLE_RULES,
      TYPED_BEAT_PROMPT_RULES,
      lesson_plan_prompt,
      script_from_plan_prompt,
  )
  from src.schemas.lesson_plan import LessonPlan, default_word_budget
  ```
- Delete the module-level definitions of `SPOKEN_STYLE_RULES = """..."""` and
  `TYPED_BEAT_PROMPT_RULES = """..."""` (the two triple-quoted constants just below
  `_LLM_BEAT_ROLES`). Every existing use of the two names keeps working through the import.

### 3b. `__init__`
After `self.last_flow = None` add `self.last_lesson_plan = None  # LessonPlan of the last run, or None`.

### 3c. Class constants (next to `MAX_SOURCE_CHARS`)
```python
PLAN_MAX_TOKENS = 6000      # the lesson plan JSON
SCRIPT_MAX_TOKENS = 16000   # a whole planned script (up to ~2400 words of narration)
```

### 3d. `_complete_unified_script_json`
Change the signature to `(self, prompt: str, max_tokens: int = 8000) -> Dict[str, Any]`.
Use `max_tokens=max_tokens` in the Anthropic call. Wrap the whole Anthropic branch
(the `client.messages.create(...)` call through the `return json.loads(...)`) in
`try/except Exception as exc:`; on an exception log
`logger.warning("Anthropic script call failed (%s); using the OpenAI client.", exc)` and
fall through to the existing OpenAI call below it. The OpenAI call is unchanged.

### 3e. New method `_run_key` (static), placed just above `_beat_runs`
```python
@staticmethod
def _run_key(beat) -> str:
    """The boundary scenes never cross: the beat's lesson-plan act, else its source segment."""
    return (getattr(beat, "act_id", "") or "") or (getattr(beat, "source_segment_id", "") or "")
```
Use it in three places, replacing the `source_segment_id` lookups there:
- `_beat_runs`: group consecutive beats by `self._run_key(beat)` (it is a staticmethod:
  call `ScriptArchitectAgent._run_key(beat)` inside the staticmethod).
- `_scene_runs`: a scene's key is `_run_key(beats[0])` when it has beats, else `""`.
- `_chunk_script_into_scenes`, inner function `pack`: replace the `beat_segment` /
  `cur_segment_ids` / `crosses_segment` block with
  ```python
  beat_key = self._run_key(beat)
  cur_keys = {self._run_key(existing) for existing in cur if self._run_key(existing)}
  crosses_boundary = bool(cur and beat_key and cur_keys and beat_key not in cur_keys)
  if cur and (crosses_boundary or cur_dur + bd > max_dur):
  ```
  Update the three docstrings to say "act (or source segment)". For beats without
  `act_id` the behaviour is identical to today.

### 3f. New method `_plan_word_budget`
```python
def _plan_word_budget(self, config, segments) -> int:
    """Narration words the lesson plan may spend: the requested length, else a default from the source size."""
    target = self._content_word_target(config)
    if target is not None:
        return target
    return default_word_budget(sum(len((s.text or "").split()) for s in segments))
```

### 3g. New method `_generate_lesson_plan`
```python
def _generate_lesson_plan(self, segments, config, review_feedback, concept_plan) -> Optional[LessonPlan]:
```
Behaviour, in order:
1. If `not getattr(config, "lesson_plan", True)`: return `None`.
2. If `sum(len(s.text or "") for s in segments) > self.MAX_SOURCE_CHARS`: log at INFO
   `"Source too long for one planning call (%d chars); scripting segment by segment."`
   and return `None`.
3. `word_budget = self._plan_word_budget(config, segments)`.
4. Build the prompt:
   ```python
   prompt = lesson_plan_prompt(
       segments,
       config.audience,
       word_budget,
       professor_notes=list(getattr(config, "professor_notes", None) or []),
       concept_plan_lines=[e.instruction() for e in concept_plan.entries] if concept_plan else None,
       presenter_opens=getattr(config, "presenter_bridges", "off") == "propose",
       review_feedback=review_feedback,
       max_source_chars=self.MAX_SOURCE_CHARS,
   )
   ```
5. In a `try`: `data = self._complete_unified_script_json(prompt, max_tokens=self.PLAN_MAX_TOKENS)`
   then `plan = LessonPlan.from_llm(data, [s.segment_id for s in segments], word_budget)`.
   On any `Exception as exc`: `logger.warning("Lesson planning failed (%s); scripting segment by segment.", exc)`
   and return `None`.
6. Log at INFO `"Lesson plan: %d acts, %d words. Question: %s"` with
   `len(plan.acts), plan.total_words(), plan.central_question` and return `plan`.

### 3h. New method `_generate_planned_video_script`
```python
def _generate_planned_video_script(self, plan: LessonPlan, segments, config, review_feedback, concept_plan):
```
Returns a `VideoScript`. Steps:
1. Prompt:
   ```python
   prompt = script_from_plan_prompt(
       plan,
       segments,
       config.audience,
       plan.total_words(),
       professor_notes=list(getattr(config, "professor_notes", None) or []),
       concept_plan_lines=[e.instruction() for e in concept_plan.entries] if concept_plan else None,
       presenter_opens=getattr(config, "presenter_bridges", "off") == "propose",
       review_feedback=review_feedback,
       max_source_chars=self.MAX_SOURCE_CHARS,
   )
   data = self._complete_unified_script_json(prompt, max_tokens=self.SCRIPT_MAX_TOKENS)
   ```
2. If `data` is not a dict or `data.get("beats")` is not a list: raise
   `ValueError("planned script response has no beats")`.
3. `known_analogies: Dict[str, Analogy] = {}` and
   `analogy_id_map = _parse_analogies(data.get("analogies"), known_analogies)`.
4. First pass over `data["beats"]`: keep items that are dicts with a non-blank string
   `narration`. For each kept item assign the global id `f"beat_{n:03d}"` (n counts kept
   items from 1) and record `local_to_global[str(raw.get("beat_id") or "")] = global_id`
   (skip the empty key).
5. Second pass, building `Beat`s in order. For each kept raw item:
   - `act_id`: `str(raw.get("act_id") or "")` if it is in `plan.act_ids()`; otherwise the
     previous built beat's `act_id`; for the first beat, `plan.acts[0].act_id`.
   - `source_segment_id`: `raw.get("source_segment_id")` if it is a string equal to one of
     the segments' ids; otherwise the first id in `plan.act(act_id).source_segments`;
     otherwise `""`.
   - `introduces_concepts`: canonicalize each string label with `_canonicalize_concept`,
     drop empties, and drop any label already introduced by an earlier beat of this
     script (keep a running `seen_concepts` set), preserving order.
   - `references_prior_beats`: map each string ref through `local_to_global`; keep it only
     if it maps to an id that sorts before this beat's id (`resolved < global_id`).
   - `visual_hint`: the raw value if it is a string, else `""`.
   - Construct `Beat(beat_id=global_id, narration=raw["narration"].strip(), visual_hint=...,
     introduces_concepts=..., references_prior_beats=..., source_segment_id=..., act_id=act_id,
     **_typed_beat_fields(raw, analogy_id_map))`. Wrap the construction in
     `try/except Exception` and skip (log at WARNING) a beat that fails validation.
6. If no beats were built: raise `ValueError("planned script has no usable beats")`.
7. `used = {b.analogy_id for b in beats if b.analogy_id}` and return
   ```python
   VideoScript(
       title=(data.get("title") or plan.title or ""),
       beats=beats,
       analogies=[a for a in known_analogies.values() if a.analogy_id in used],
       plan=plan,
   )
   ```
   `Beat` is imported inside the existing methods as
   `from src.schemas.video_script import VideoScript, Beat`; do the same here.

### 3i. `_generate_video_script`: route through the plan
Immediately after the line `segments = segments or self._segment_ocr_output(ocr_output, config)`
and before the existing `if segments:` block, insert:
```python
if segments:
    plan = self._generate_lesson_plan(segments, config, review_feedback or [], concept_plan)
    if plan is not None:
        try:
            return self._generate_planned_video_script(
                plan, segments, config, review_feedback or [], concept_plan,
            )
        except Exception as exc:
            logger.warning("Planned script failed (%s); scripting segment by segment.", exc)
```
The existing `if segments: return self._generate_segmented_video_script(...)` block and
everything after it stay as they are. The keyless early return above (`if not
os.getenv("OPENAI_API_KEY")`) also stays first, so keyless runs never plan.

### 3j. `execute`: keep the plan attached, skip the smoothing pass when it is not needed
Add a private helper on the class:
```python
@staticmethod
def _with_plan(script, plan):
    """Re-attach the lesson plan after a step that rebuilt the script without it."""
    if plan is not None and getattr(script, "plan", None) is None:
        return script.model_copy(update={"plan": plan})
    return script
```
In `execute`, right after the `logger.info("Generated video script: ...")` call:
```python
plan = getattr(video_script, "plan", None)
self.last_lesson_plan = plan
```
Replace the single line
`video_script, self.last_flow = self._smooth_narration(video_script, config)` with:
```python
# A planned script is written in one piece, so it has no section seams to
# smooth; the flow edit runs on it only to bring an over-long script down.
target_words = self._content_word_target(config)
script_words = sum(len(b.narration.split()) for b in video_script.beats)
if plan is not None and (target_words is None or script_words <= target_words * 1.05):
    self.last_flow = None
else:
    video_script, self.last_flow = self._smooth_narration(video_script, config)
video_script = self._with_plan(video_script, plan)
```
Then call `video_script = self._with_plan(video_script, plan)` again after each of these
three existing steps: the `enforce_script_directives(...)` block, the
`_enforce_length_budget(...)` line, and the `BridgePlannerAgent().propose(...)` block
(one call placed after the bridge `if` block is enough for the last two if you put it
directly before `self.last_video_script = video_script`; make sure the plan is attached
at that line and when `_chunk_script_into_scenes` is called).

### 3k. `_assemble_storyboard_from_script` and `_log_segment_coverage`
- In `_assemble_storyboard_from_script`: read `plan = getattr(video_script, "plan", None)`;
  when it is not None set `config.visual_world = plan.visual_world_lines()` before the
  `StoryboardJSON(...)` is built; pass `plan` on:
  `self._log_segment_coverage(scenes, segments or [], plan=plan)`.
- `_log_segment_coverage(self, scenes, segments, plan=None)`: when `plan` is not None,
  remove from `expected` every segment id whose `plan.treatment_of(segment_id)` is one of
  `"compressed"`, `"deferred"`, `"omitted"` (they deliberately have no beat of their own),
  and additionally log one INFO line
  `"Lesson plan coverage: %d covered, %d compressed, %d deferred, %d omitted."` from
  `plan.coverage_report()["counts"]`. Everything else in the method is unchanged.

## Edit 4 — `src/agents/bridge_planner.py`

1. Import `from src.prompts.narrative import bridge_prompt, bridge_slot_line`.
2. Add a module-level helper:
   ```python
   def _run_key(beat: Beat) -> str:
       return (getattr(beat, "act_id", "") or "") or (beat.source_segment_id or "")
   ```
3. `positions(self, beats, planned: bool = False)`:
   - the transition test becomes `if _run_key(beats[i]) != _run_key(beats[i - 1]):`;
   - append the `(len(beats), "recap")` candidate only when `not planned` (a planned
     script already ends on its payoff beat; a recap after it would repeat it).
4. `propose(self, script)`:
   - `plan = getattr(script, "plan", None)`;
   - `positions = self.positions(beats, planned=plan is not None)`;
   - `texts = self._write(beats, positions, plan)`;
   - pass `plan=plan` to the `VideoScript(...)` it returns.
5. `_write(self, beats, positions, plan=None)`:
   - `texts = {i: self._template(beats, i, kind, plan) for i, kind in positions}`;
   - when `plan is None`: the rest of the method is exactly today's code (same prompt
     string, same response handling);
   - when `plan is not None`:
     - the slots sent to the LLM are every position EXCEPT an `"opening"` slot whose
       template already equals a non-empty `plan.hook` (the hook was written to be said
       on camera and is used word for word);
     - if no slots remain or `self.llm is None`, return `texts`;
     - build one line per remaining slot with
       ```python
       bridge_slot_line(
           i, kind,
           before=beats[i - 1].narration if i > 0 else "",
           after=beats[i].narration if i < len(beats) else "",
           next_question=_next_question(beats, i, plan),
       )
       ```
       where `_next_question` (a module-level helper you add) returns
       `plan.act(beats[i].act_id).viewer_question` when `i < len(beats)`, the beat has an
       `act_id` and `plan.act(...)` is not None, else `""`;
     - call `self.llm(bridge_prompt(lines, plan))` and handle the response exactly as the
       existing code does (same `bridges` parsing, same `_echoes` check, same exception
       handling and log messages).
6. `_template(beats, i, kind, plan=None)` (stays a staticmethod): when `plan` is not None,
   return `plan.hook` for `"opening"` if it is non-empty, and for `"transition"` return
   the `_next_question(beats, i, plan)` text if it is non-empty; in every other case fall
   through to today's template strings unchanged.

## Edit 5 — `src/agents/script_editor.py`

In `apply`, the `VideoScript(...)` built into `result.script` gains one argument:
`plan=original.plan`. Nothing else changes.

## Edit 6 — `src/orchestrator/revision.py`

1. `lesson_script_from_storyboard(storyboard, title="", analogies=(), plan=None)`: pass
   `plan=plan` into the `VideoScript(...)` it returns (no copy needed).
2. `load_base_run`: where it reads `saved = load_lesson_script(script_path)`, also keep
   `plan = saved.plan` (initialise `plan = None` next to `title, analogies = "", []`) and
   pass `plan=plan` to `lesson_script_from_storyboard(...)`.

## Edit 7 — `src/orchestrator/pipeline.py`

1. `PipelineConfig`: add, after `target_minutes`:
   ```python
   # Plan the lesson (one question, acts in viewing order, a running example)
   # before scripting. False = the older segment-by-segment script.
   lesson_plan: bool = True
   ```
   and pass `lesson_plan=self.lesson_plan` in `to_storyboard_config`.
2. `_run_script_architect`: next to the block that stores `narration_flow` in
   `self._quality_reports`, add:
   ```python
   lesson_plan = getattr(self.script_architect_agent, "last_lesson_plan", None)
   if lesson_plan is not None:
       self._quality_reports["lesson_plan"] = {
           **lesson_plan.model_dump(mode="json"),
           "coverage_report": lesson_plan.coverage_report(),
       }
   else:
       self._quality_reports.pop("lesson_plan", None)
   ```
3. `_persist_quality_reports`: add `"lesson_plan": "lesson_plan.json",` to
   `report_filenames`.
4. `_write_lesson_script`: pass `plan=getattr(source, "plan", None)` to
   `lesson_script_from_storyboard(...)`.

## Edit 8 — `src/cli.py` (the `convert` command only)

1. Add an option next to `--target-minutes`:
   ```python
   @click.option(
       '--lesson-plan/--no-lesson-plan',
       default=True,
       help='Plan the lesson (one question, acts, a running example) before scripting. '
            '--no-lesson-plan uses the older segment-by-segment script.',
   )
   ```
   Add the matching `lesson_plan` parameter to the command function's signature.
2. In the "Content Configuration" printout add a line
   `click.echo(f"  Lesson Plan:  {'on' if lesson_plan else 'off'}")` after the
   target-length line.
3. Pass `lesson_plan=lesson_plan` into the `PipelineConfig(...)` built for `convert`.

## Tests — `tests/test_lesson_plan.py` (new)

Use plain pytest. Build segments with
`SourceSegment(segment_id="seg_001", order=1, page_start=1, page_end=1, heading="H", text="...")`
(check the schema file for required fields). To make the architect take its LLM path in
a keyless test: `monkeypatch.setenv("OPENAI_API_KEY", "test-key")` before constructing
`ScriptArchitectAgent()`, then replace `agent._complete_unified_script_json` with a fake
that returns queued dicts and records the prompts. Never let a test reach a real client.
Look at `tests/test_script_architect.py` and `tests/test_narration_flow.py` for how the
existing tests build `OCROutput`, `Configuration` and fake clients, and reuse those
patterns.

A valid plan dict for the fakes:
```python
PLAN = {
    "central_question": "Why does one direction carry most of a cloud of points?",
    "hook": "Here are eight points. One line through them matters more than any other.",
    "promise": "You will see why.", "takeaway": "The long axis is the answer.", "title": "T",
    "running_example": {"description": "eight points", "values": {"A": "[[3, 1], [1, 2]]"}},
    "motifs": [{"motif_id": "data_cloud", "what": "eight dots", "color": "BLUE_B", "stands_for": "the rows"}],
    "acts": [
        {"kind": "hook", "viewer_question": "Which line?", "key_idea": "One line fits best.",
         "visual": "dots", "source_segments": ["seg_001"], "words": 40},
        {"kind": "payoff", "viewer_question": "Why that one?", "key_idea": "It is the longest stretch.",
         "visual": "ellipse", "source_segments": ["seg_002"], "words": 60},
    ],
    "coverage": [{"segment_id": "seg_003", "treatment": "omitted", "reason": "A proof."}],
}
```

Write these tests (one test function each, named as given):

1. `test_beat_and_script_defaults_are_backward_compatible` — `Beat(beat_id="b", narration="n",
   visual_hint="v").act_id == ""`; `VideoScript(beats=[...]).plan is None`; a JSON string
   of a script WITHOUT `act_id`/`plan` keys loads with `VideoScript.model_validate_json`.
2. `test_act_id_does_not_change_beat_fingerprint` — `beat_fingerprint` of two beats that
   differ only in `act_id` is equal.
3. `test_script_with_plan_round_trips` — a `VideoScript` with `plan=LessonPlan.from_llm(PLAN,
   ["seg_001", "seg_002", "seg_003"], 100)` survives `model_dump_json` → `model_validate_json`
   with `plan` equal.
4. `test_from_llm_completes_coverage_and_budget` — for that plan: act ids are
   `["act_01", "act_02"]`, `total_words() == 100`, coverage has exactly the three segment
   ids, `treatment_of("seg_003") == "omitted"`, `planned_out_segment_ids() == ["seg_003"]`.
5. `test_from_llm_rejects_unusable_plans` — `LessonPlan.from_llm({})` and
   `LessonPlan.from_llm({"central_question": "q", "acts": []})` raise `ValueError`.
6. `test_planned_script_path` — fake returns `PLAN` then a script dict with three beats
   (`act_01`, an unknown act id, `act_02`; local ids `x1`, `x2`, `x3`; the third
   references `x1`; the second has `source_segment_id` `"seg_999"`; two beats both
   introduce `"best fit line"`). Call `agent._generate_video_script(ocr_output, config,
   segments=segments)`. Assert: result `.plan` is a `LessonPlan`; beat ids are
   `beat_001..beat_003`; act ids are `act_01, act_01, act_02` (unknown inherits the
   previous); the second beat's `source_segment_id` is `"seg_001"` (first of its act's
   segments); the third beat's `references_prior_beats == ["beat_001"]`;
   `"best_fit_line"` is introduced by exactly one beat; the first recorded prompt contains
   `"[seg_001]"` and `"ONE CENTRAL QUESTION"`; the second contains `"THE PLAN (binding)"`.
7. `test_planning_failure_falls_back_to_segments` — fake raises on the first call;
   `agent._generate_segmented_video_script` monkeypatched to return a sentinel script;
   the sentinel comes back.
8. `test_lesson_plan_flag_off_skips_planning` — `Configuration(..., lesson_plan=False)`;
   the fake for `_complete_unified_script_json` raises `AssertionError` if called; the
   segmented sentinel comes back.
9. `test_long_source_skips_planning` — monkeypatch `agent.MAX_SOURCE_CHARS = 10`; the
   planner fake is never called; the segmented sentinel comes back.
10. `test_scenes_do_not_cross_acts` — a script of three ~10-word beats, all with the same
    `source_segment_id`, acts `act_01, act_01, act_02`: `_chunk_script_into_scenes` gives
    scenes whose beats never mix act ids. A second script with empty `act_id`s and two
    different `source_segment_id`s still splits at the segment boundary.
11. `test_beat_runs_group_by_act` — `ScriptArchitectAgent._beat_runs` on beats with acts
    `a, a, b` and one shared segment id returns two runs keyed `"a"` and `"b"`.
12. `test_smoothing_skipped_for_planned_script_within_budget` — drive `agent.execute(...)`
    through the planned path with fakes (monkeypatch `_plan_visual_intents` to a no-op that
    leaves placeholder intents, and `_smooth_narration` to raise `AssertionError`); the
    call succeeds, `agent.last_flow is None`, `agent.last_lesson_plan` is the plan, the
    returned storyboard's `configuration.visual_world` contains a line with
    `"data_cloud"`, and `agent.last_video_script.plan` is not None.
13. `test_coverage_log_respects_planned_omissions` — with `caplog` at WARNING: calling
    `_log_segment_coverage` with scenes covering `seg_001` and `seg_002`, three segments
    and the plan above logs no "missing source segments" warning; without `plan` it does.
14. `test_bridges_follow_the_plan` — a planned script (two acts, plan attached) through
    `BridgePlannerAgent(llm=fake_llm)` where `fake_llm` records its prompt and returns
    `{"bridges": []}`: the first proposed bridge's narration equals `PLAN["hook"]`; no
    bridge is placed after the last content beat; the recorded prompt (when a transition
    slot exists; make the two acts long enough, about 300 words each, that
    `BRIDGE_EVERY_SECONDS` is reached) contains `"QUESTION THE NEXT PART ANSWERS: Why that one?"`
    and does not contain a slot line for the opening; the returned script's `plan` is not None.
15. `test_bridges_without_plan_are_unchanged` — an unplanned script still gets a recap
    candidate: `BridgePlannerAgent(use_default_llm=False).positions(beats)` ends with a
    `"recap"` entry when the budget allows (reuse the shape of the existing bridge tests
    in `tests/test_presenter.py`).
16. `test_script_editor_keeps_plan` — `ScriptEditorAgent(use_default_llm=False).apply(script_with_plan, [])`
    returns a script whose `plan` equals the input's.
17. `test_lesson_script_from_storyboard_keeps_plan` —
    `lesson_script_from_storyboard(storyboard, plan=plan).plan == plan` (build the smallest
    valid `StoryboardJSON` the way existing revision tests do).
18. `test_pipeline_config_carries_lesson_plan_flag` — `PipelineConfig().to_storyboard_config().lesson_plan is True`
    and `PipelineConfig(lesson_plan=False).to_storyboard_config().lesson_plan is False`.

## Gates (run them, report each result in one line)

- G1 new tests — CHECK: `python -m pytest tests/test_lesson_plan.py -q -p no:cacheprovider` — EXPECT: `18 passed`
- G2 architect + flow + presenter + revision suites — CHECK:
  `python -m pytest tests/test_script_architect.py tests/test_narration_flow.py tests/test_presenter.py tests/test_revision.py tests/test_length_budget.py tests/test_professor_controls.py tests/test_hybrid.py -q -p no:cacheprovider`
  — EXPECT: `0 failed`
- G3 full suite, once, at the end — CHECK: `python -m pytest tests -q -p no:cacheprovider`
  — EXPECT: `0 failed` and at least `1190 passed` (baseline before this package: 1172 passed, 10 deselected)

## Report (your final message, under 300 words)

1. Files changed, one line each.
2. Each gate's result line.
3. Any existing test whose expectation you changed, and why.
4. Anything in this spec that did not match the code you found, and what you did about it.
