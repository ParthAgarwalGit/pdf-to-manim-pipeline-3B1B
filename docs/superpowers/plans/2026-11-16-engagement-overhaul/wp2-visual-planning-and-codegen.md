# WP2 — Visual plans that are repaired instead of discarded; new planner and codegen prompts wired in

You are implementing one work package in an existing Python repository (a PDF → Manim
explainer-video pipeline). Every design decision has already been made and is written
below. Implement exactly this; do not redesign, rename, or add features.

## Ground rules (binding)

- Working directory: the repository root you were started in. Edit only the files listed
  under "Files you may edit". Create only the files listed under "Files you create".
- **Frozen files — read them, never edit them:** everything under `src/prompts/`,
  `src/schemas/lesson_plan.py`, `src/utils/visual_plan_repair.py`,
  `src/utils/manim_engine_contract.py`. They are finished and hand-written.
- **No git commands of any kind.** The tree has uncommitted work that must stay exactly
  as it is apart from your edits.
- No network or paid API calls. Tests are keyless: `tests/conftest.py` strips API keys.
- The only shell command you run is `python -m pytest ...`.
- Several files have mixed CRLF/LF line endings. Ignore that; edit normally.
- Never weaken, skip or delete an existing test to get to green. Where this spec changes
  behaviour on purpose it names the tests to update. If any other existing test fails,
  stop and report it.
- New schema fields have defaults; old `storyboard.json` files must still load.
- Match the surrounding code style.

## Background (what this package is for)

For each scene a "visual planner" LLM call returns a list of objects (Manim primitive,
position box, visibility interval). The plan is validated (ids, purposes, at most 15
objects, at most 7 visible at once). Today a plan that still fails after two retries is
**thrown away** and replaced by a stub: the narration's first sentence as an on-screen
title (`intro_text`) plus one stock arrow (`archetype_visual`). In real runs that
happened to 30-50% of scenes, and the stub's objects were then carried into the next
scene's plan. Separately, a deterministic pass (`_resolve_bbox_overlaps`) pushes any two
overlapping boxes apart, which moves drawings off the axes they belong on.

This package:
1. repairs plans deterministically (`src/utils/visual_plan_repair.py`, already written)
   so the stub becomes a rare last resort, and makes the stub harmless when it happens;
2. restricts the overlap pass to text objects;
3. swaps in the new planner and code-generator prompt text (`src/prompts/visual.py`,
   `src/prompts/codegen.py`, already written);
4. gives the code generator what it never received: the script's per-beat visual hints,
   each transformation's description, where the plan came from, and the video's
   recurring look ("visual world").

Functions you will call (read their docstrings first):
- `src/utils/visual_plan_repair.py`: `repair_visual_intent`, `resolve_text_overlaps`,
  `is_stub_intent`, `demote_stub`, `carryable_ids`.
- `src/prompts/visual.py`: `visual_plan_static_prompt`, `lesson_context_block`,
  `beat_schedule_block`.
- `src/prompts/codegen.py`: `codegen_static_system_prompt`, `visual_world_block`.

## Files you may edit

`src/schemas/storyboard.py`, `src/schemas/scene_processing.py`,
`src/agents/script_architect.py`, `src/agents/scene_controller.py`,
`src/agents/animation_generator.py`, `tests/test_space_and_overlap.py`.
Other existing test files only as allowed by the ground rules.

## Files you create

`tests/test_visual_plan_repair.py`

---

## Edit 1 — `src/schemas/storyboard.py`, `class Scene`

After `frame_reason`, add:
```python
visual_plan_source: str = Field(
    default="planner",
    description="Where the visual plan came from: 'planner' (as designed), 'repaired' "
                "(cut down to its budgets), or 'fallback' (planner failed; stub plan).",
)
```
Do not add anything to `VisualIntent` or `MathematicalObject`: their JSON dump feeds the
scene fingerprints and must not change.

## Edit 2 — `src/schemas/scene_processing.py`

1. `class SceneSpec`: after `beat_timings`, add
   ```python
   visual_hints: List[str] = Field(
       default_factory=list,
       description="The script's per-beat description of what changes on screen ('beat_id: hint').",
   )
   plan_source: str = Field(
       default="planner",
       description="Scene.visual_plan_source: planner | repaired | fallback.",
   )
   ```
2. `class SharedContext`: after `notation_rules`, add
   ```python
   visual_world: List[str] = Field(
       default_factory=list,
       description="The video's recurring look (running example, motifs), one fact per line.",
   )
   ```

## Edit 3 — `src/agents/script_architect.py`

### 3a. Imports
```python
from src.prompts.visual import beat_schedule_block, lesson_context_block, visual_plan_static_prompt
from src.utils.visual_plan_repair import (
    carryable_ids, demote_stub, is_stub_intent, repair_visual_intent, resolve_text_overlaps,
)
```

### 3b. `_visual_intent_static_prompt` (module level, keep the `@lru_cache(maxsize=1)` decorator and the docstring)
Replace the whole body below the docstring with:
```python
return visual_plan_static_prompt(build_visual_intent_contract_prompt(), catalog_for_prompt())
```

### 3c. `_resolve_bbox_overlaps` (module level)
Keep the name and signature `(objs: List[Any], iterations: int = 4) -> None`. Replace the
body with a call to `resolve_text_overlaps(objs, SAFE_X, SAFE_Y, PLAN_GAP, iterations)`
and rewrite the docstring to say: only text objects are nudged apart from each other;
drawings are never moved, because drawings that share space (a vector on its plane, the
axes of an ellipse) are meant to. `_intervals_overlap` stays (other code may use it).

### 3d. `_carried_objects_from_intent` (staticmethod)
Replace `active = set(intent.scene_state_out.active_object_ids or [])` with
`active = set(carryable_ids(intent))`. Add one sentence to the docstring: the stub plan's
objects are never handed on.

### 3e. New method `_beat_windows`
```python
def _beat_windows(self, scene: Scene) -> Dict[str, Tuple[float, float]]:
    """Estimated (start, end) seconds of each of the scene's beats, from word counts."""
    windows: Dict[str, Tuple[float, float]] = {}
    cursor = 0.0
    for beat in getattr(scene, "source_beats", None) or []:
        seconds = len(beat.narration.split()) / self.WORDS_PER_SECOND
        windows[beat.beat_id] = (cursor, cursor + seconds)
        cursor += seconds
    return windows
```

### 3f. New method `_repair_plan`
```python
def _repair_plan(self, scene: Scene, intent: VisualIntent, carried_objects) -> str:
    """Fix a plan's countable defects in place; return where the plan came from.

    'fallback' for the stub plan (made harmless here), 'repaired' when the
    plan had to be cut down or completed, else 'planner'.
    """
    if is_stub_intent(intent):
        demote_stub(intent)
        return "fallback"
    notes = repair_visual_intent(
        intent,
        valid_beat_ids=[b.beat_id for b in (getattr(scene, "source_beats", None) or [])],
        max_total=self.MAX_OBJECTS_PER_SCENE,
        max_concurrent_allowed=self.MAX_CONCURRENT_OBJECTS,
        protected_ids=[o.object_id for o in (carried_objects or []) if o.object_id],
        beat_windows=self._beat_windows(scene),
    )
    if notes:
        logger.info("Repaired visual plan for %s: %s", scene.scene_id, "; ".join(notes))
        return "repaired"
    return "planner"
```

### 3g. `_plan_visual_intents`: repair before validating, mark the source
All changes are inside the `try:` block and its `except`. Define once, at the top of the
method body (before the `while` loop):
```python
live = self.client is not None and bool(os.getenv("OPENAI_API_KEY"))
```
Repair applies only when `live` is true. Keyless runs (the mock path used by most tests)
keep their stub plans exactly as today.

1. In the `for purpose_attempt in range(...)` loop, immediately after
   `scene.visual_intent = self._generate_visual_intent_for_chunk(...)` add:
   ```python
   if live:
       scene.visual_plan_source = self._repair_plan(scene, scene.visual_intent, carried_objects)
   ```
   (the `constraint_errors = self._visual_intent_constraint_errors(...)` line that follows
   stays and now runs on the repaired plan).
2. In the text-only retry branch, immediately after
   `retry_intent = self._generate_visual_intent_for_chunk(...)` add
   `retry_source = self._repair_plan(scene, retry_intent, carried_objects)`
   (that branch already runs only on the live path). Where the branch accepts the retry
   (`scene.visual_intent = retry_intent`) also set `scene.visual_plan_source = retry_source`.
   Where it keeps `validated_intent`, leave `scene.visual_plan_source` as it was.
3. In the `except Exception as exc:` branch, after
   `scene.visual_intent = self._generate_fallback_visual_intent(scene.narration)` add:
   ```python
   if live:
       demote_stub(scene.visual_intent)
       scene.visual_plan_source = "fallback"
   ```
4. Change the final summary warning so it counts every scene that ended on the stub, not
   only the exception path: after the `while` loop compute
   ```python
   fallback_scenes = [s.scene_id for s in scenes if getattr(s, "visual_plan_source", "planner") == "fallback"]
   repaired_scenes = [s.scene_id for s in scenes if getattr(s, "visual_plan_source", "planner") == "repaired"]
   ```
   Replace the existing `if visual_failures:` warning block with:
   ```python
   if fallback_scenes or visual_failures:
       logger.warning(
           "Visual intent fallback used for %d/%d scenes",
           max(len(fallback_scenes), len(visual_failures)), len(scenes),
       )
   if repaired_scenes:
       logger.info("Visual plans repaired for %d/%d scenes: %s",
                   len(repaired_scenes), len(scenes), ", ".join(repaired_scenes))
   ```
   (the wording `"Visual intent fallback used for %d/%d scenes"` is unchanged on purpose:
   past run logs are compared on it).

`_split_scene_at_beat_boundary` builds new `Scene` objects; they get the default
`visual_plan_source="planner"` and are planned on the next loop pass. No change there.

### 3h. `_generate_visual_intent_for_chunk`: lesson context and beat schedule in the prompt
After `archetype_block` is computed, add:
```python
plan = getattr(video_script, "plan", None)
act_id = next((getattr(b, "act_id", "") for b in chunk_beats if getattr(b, "act_id", "")), "")
lesson_block = lesson_context_block(plan, act_id)
schedule_block = beat_schedule_block(chunk_beats, self.WORDS_PER_SECOND)
```
In the prompt assembly, insert `f"{lesson_block}"` as the FIRST item after
`_visual_intent_static_prompt() + (` (before `f"{archetype_block}"`), and insert
`f"{schedule_block}"` immediately after the existing
`f"TIMING: {word_count} words at ~3 wps => visuals should fill ~{target}s.\n"` line.
Nothing else in the prompt changes; in particular the static prefix stays first and
`lesson_block` is `""` when the script has no plan.

### 3i. `_generate_planned_video_script`: send a script that is too short back once
(This method was added by the previous work package. A live trial showed the writer
coming in about 30% under the requested length, which makes a "five minute" video run
three and a half.)

1. Extend the existing `from src.prompts.narrative import (...)` with `draft_listing` and
   `script_length_feedback`.
2. Change the signature to
   `(self, plan, segments, config, review_feedback, concept_plan, previous_draft: str = "", _retry: bool = False)`
   and pass `previous_draft=previous_draft` to the `script_from_plan_prompt(...)` call.
3. Bind the `VideoScript` the method builds to a local name `script` instead of
   returning it directly, then end the method with:
   ```python
   feedback = "" if _retry else script_length_feedback(plan, script.beats, plan.total_words())
   if not feedback:
       return script

   def words(s) -> int:
       return sum(len(b.narration.split()) for b in s.beats)

   try:
       retry = self._generate_planned_video_script(
           plan, segments, config, list(review_feedback or []) + [feedback], concept_plan,
           previous_draft=draft_listing(script.beats), _retry=True,
       )
   except Exception as exc:
       logger.warning("Planned script length retry failed (%s); keeping the first draft.", exc)
       return script
   if words(script) < words(retry) <= plan.total_words() * 1.1:
       logger.info(
           "Planned script extended on retry: %d -> %d words (budget %d).",
           words(script), words(retry), plan.total_words(),
       )
       return retry
   logger.info(
       "Planned script kept at %d words (budget %d); the retry gave %d.",
       words(script), plan.total_words(), words(retry),
   )
   return script
   ```
   At most one retry happens per script (`_retry=True` skips the check).

## Edit 4 — `src/agents/scene_controller.py`

1. Where `SharedContext(...)` is built (the method that reads
   `storyboard.configuration.notation_rules`), add the argument
   ```python
   visual_world=list(
       getattr(getattr(storyboard, "configuration", None), "visual_world", None) or []
   ),
   ```
2. `_scene_to_spec`: add two arguments to the `SceneSpec(...)` call:
   ```python
   visual_hints=[
       f"{b.beat_id}: {b.visual_hint.strip()}"
       for b in (getattr(scene, "source_beats", None) or [])
       if (b.visual_hint or "").strip()
   ],
   plan_source=getattr(scene, "visual_plan_source", "planner") or "planner",
   ```

## Edit 5 — `src/agents/animation_generator.py`

1. Imports: `from src.prompts.codegen import codegen_static_system_prompt, visual_world_block`.
2. `_codegen_static_system_prompt` (module level; keep `@lru_cache(maxsize=1)` and the
   docstring): replace the whole body below the docstring with
   ```python
   return codegen_static_system_prompt(
       build_animation_codegen_contract_prompt(), build_3b1b_quality_prompt()
   )
   ```
3. `_generate_code`, the `scene_data` dict:
   - add two keys after `"carry_out": carry_out,`:
     ```python
     "plan_source": getattr(scene_spec, "plan_source", "planner") or "planner",
     "visual_hints": list(getattr(scene_spec, "visual_hints", None) or []),
     ```
   - each transformation entry gains three keys (keep the existing four):
     ```python
     "beat_id": getattr(t, "beat_id", ""),
     "description": (t.parameters or {}).get("description", "") if hasattr(t, "parameters") else "",
     "animation_hint": (t.parameters or {}).get("animation_hint", "") if hasattr(t, "parameters") else "",
     ```
4. `_generate_code`, the `dynamic_system` f-string: its last line is
   `{color_block}{self._build_notation_block(context)}{self._build_sync_block(scene_spec)}"""`.
   Insert the visual world between the color block and the notation block:
   `{color_block}{visual_world_block(list(getattr(context, "visual_world", None) or []))}{self._build_notation_block(context)}{self._build_sync_block(scene_spec)}"""`.

## Edit 6 — `tests/test_space_and_overlap.py` (deliberate behaviour change)

`_resolve_bbox_overlaps` now moves only text. In `class TestOverlapNudge`:
- add a helper next to the module's `_obj`:
  ```python
  def _text(oid, x, y, w=2.0, h=2.0, t_in=0.0, t_out=10.0):
      return MathematicalObject(
          object_id=oid, object_type="text", content="label", primitive="Text",
          args={"text": "label"}, bbox=BBox(x=x, y=y, w=w, h=h),
          interval=TimeInterval(t_in=t_in, t_out=t_out),
      )
  ```
- the four existing tests in `TestOverlapNudge` build their objects with `_text` instead
  of `_obj` (same coordinates, same assertions);
- add
  ```python
  def test_overlapping_drawings_are_not_moved(self):
      a, b = _obj("a", 0.0, 0.0), _obj("b", 0.5, 0.1)
      _resolve_bbox_overlaps([a, b])
      assert (a.bbox.x, a.bbox.y, b.bbox.x, b.bbox.y) == (0.0, 0.0, 0.5, 0.1)

  def test_text_over_a_drawing_is_not_moved(self):
      a, b = _obj("a", 0.0, 0.0), _text("b", 0.5, 0.1)
      _resolve_bbox_overlaps([a, b])
      assert (b.bbox.x, b.bbox.y) == (0.5, 0.1)
  ```

## Tests — `tests/test_visual_plan_repair.py` (new)

Helpers to define at the top of the file:
```python
def _mob(oid, primitive, t_in, t_out, x=0.0, y=0.0, **extra):
    types = {"Text": "text", "MathTex": "equation", "Axes": "axes"}
    args = {"text": "lbl"} if primitive == "Text" else {"tex": "x"} if primitive == "MathTex" else {}
    return MathematicalObject(
        object_id=oid, object_type=types.get(primitive, "geometric_shape"),
        content="lbl" if primitive in ("Text", "MathTex") else "",
        primitive=primitive, args=args, purpose=extra.pop("purpose", f"shows {oid}"),
        supports_beat_ids=extra.pop("beats", ["beat_001"]),
        bbox=BBox(x=x, y=y, w=2.0, h=1.0), interval=TimeInterval(t_in=t_in, t_out=t_out),
    )

def _intent(objs, transformations=(), carry=()):
    return VisualIntent(
        mathematical_objects=list(objs), transformations=list(transformations), beats=[],
        scene_state_in=SceneState(), scene_state_out=SceneState(active_object_ids=list(carry)),
    )
```
(`Axes` takes no required args; check `MOBJECT_CATALOG` for the kwargs each primitive
allows and keep `args` within them.)

Write these tests (one function each, named as given):

1. `test_valid_plan_is_untouched` — 1 Axes + 2 Text, all visible 0-10, limits 15/7:
   `repair_visual_intent` returns `[]` and `intent.model_dump()` is unchanged.
2. `test_over_concurrency_cuts_labels_before_drawings` — 4 drawings (Axes, Circle, Line,
   Dot) visible 0-10 and 6 Text labels visible 0-10, limits 15/7: afterwards
   `max_concurrent(intent.mathematical_objects) <= 7`, all four drawings are still
   present, and at least three Text objects were removed or had their interval changed.
3. `test_retire_keeps_an_object_that_had_its_time` — 7 drawings visible 5-10 plus one Text
   visible 0-10, limits 15/7: the Text is still in the plan with `interval.t_out == 5.0`.
4. `test_total_cap_drops_lowest_priority` — 3 drawings + 5 Text with disjoint intervals,
   `max_total=4`: 4 objects remain, the 3 drawings among them.
5. `test_last_drawing_is_never_removed` — 1 Circle + 3 Text, `max_total=1`: one object
   remains and it is the Circle.
6. `test_protected_object_survives` — 8 Circles visible 0-10, limits 15/7,
   `protected_ids=["c7"]` where `c7` is the last one: `c7` is still present afterwards.
7. `test_bookkeeping_is_filled` — objects with `object_id=None`, `purpose=""` and
   `supports_beat_ids=["nope"]`, `valid_beat_ids=["beat_001", "beat_002"]`,
   `beat_windows={"beat_001": (0, 5), "beat_002": (5, 10)}`, one object appearing at
   t=6: every object ends with an id and a non-empty purpose; the t=6 object's
   `supports_beat_ids == ["beat_002"]`.
8. `test_dangling_references_are_dropped` — a plan over the total cap whose
   transformation and `scene_state_out` refer to an object that gets dropped: after
   repair no transformation and no carry-out id refers to a missing object.
9. `test_stub_is_detected_and_demoted` — `ScriptArchitectAgent.__new__(ScriptArchitectAgent)._generate_fallback_visual_intent("We rotate the vector and watch its direction change.")`:
   `is_stub_intent` is True; after `demote_stub` there is no object with id
   `"intro_text"`, `"archetype_visual"` is present, and
   `scene_state_out.active_object_ids == []`. A designed plan (`_intent([_mob("axes_main", "Axes", 0, 10)])`)
   is not a stub.
10. `test_stub_objects_are_never_carried` — `ScriptArchitectAgent._carried_objects_from_intent`
    on an un-demoted fallback intent returns `[]`; on a designed intent carrying
    `"axes_main"` it returns that object.
11. `test_planner_output_over_budget_is_repaired_not_replaced` — build an agent with
    `ScriptArchitectAgent.__new__(ScriptArchitectAgent)`, set `agent.strict = False`,
    `agent.client = object()`, `monkeypatch.setenv("OPENAI_API_KEY", "test-key")`.
    Monkeypatch `agent._generate_visual_intent_for_chunk` to return, on every call, a fresh
    intent with 1 Axes + 9 Text all visible 0-10 (all `supports_beat_ids` = the scene's
    beat id). Call `agent._plan_visual_intents([scene], script, config, [])` for one
    Manim scene with one source beat (build the scene with
    `agent._chunk_script_into_scenes(script)` from a one-beat `VideoScript`; build `config`
    as the other architect tests do). Assert: the planner fake was called exactly once;
    `scene.visual_plan_source == "repaired"`; the plan still contains the Axes; it is not a
    stub; `agent._visual_intent_constraint_errors(scene, scene.visual_intent) == []`.
12. `test_stub_from_planner_is_marked_and_not_carried` — same setup with two scenes (a
    two-beat script whose beats have different `source_segment_id`s so they chunk into
    two scenes); the planner fake returns `agent._generate_fallback_visual_intent(scene.narration)`
    for the first scene and a designed one-Axes intent for the second, and records the
    `carried_objects` argument of each call. Assert: first scene's
    `visual_plan_source == "fallback"` and it has no `"intro_text"` object unless that is
    its only object; the second call received `carried_objects == []`.
13. `test_keyless_planning_is_unchanged` — without the env var and with
    `agent.client = None`: after `_plan_visual_intents` a scene's plan still contains
    `"intro_text"` and `visual_plan_source == "planner"`.
14. `test_planner_prompt_gets_lesson_context_and_schedule` — fake OpenAI client (copy the
    capture pattern from the existing visual-planner prompt tests in
    `tests/test_script_architect.py` / `tests/test_prompt_caching.py`) returning a valid
    one-object plan; a script whose `plan` is a `LessonPlan` (build with
    `LessonPlan.from_llm` as in `tests/test_lesson_plan.py`) and whose beat has
    `act_id="act_01"`: the captured prompt starts with `_visual_intent_static_prompt()`,
    and contains `"LESSON CONTEXT"`, `"VISUAL WORLD"`, `"BEAT SCHEDULE"`. With a script
    without a plan the prompt contains `"BEAT SCHEDULE"` but not `"LESSON CONTEXT"`.
15. `test_static_planner_prompt_documents_motion_shapes` — `_visual_intent_static_prompt()`
    contains `"CLOSED VOCABULARY"`, `'"transformation_type"'`, `'"source_object_id"'`,
    `'"timing"'`, `'"beat_goal"'` and `"at most four words"`.
16. `test_scene_spec_carries_hints_and_source` — a `Scene` with two source beats (one
    with an empty `visual_hint`) and `visual_plan_source="fallback"`: the controller's
    `_scene_to_spec` gives `visual_hints == ["<beat_id>: <hint>"]` (only the non-empty
    one) and `plan_source == "fallback"`. Build the controller the way
    `tests/test_scene_controller.py` does.
17. `test_shared_context_carries_visual_world` — a storyboard whose
    `configuration.visual_world == ["Motif data_cloud: dots; color BLUE_B"]`: the
    controller's shared context has the same list in `visual_world`.
18. `test_codegen_prompt_carries_hints_world_and_descriptions` — using the fake-Gemini
    capture pattern from `tests/test_prompt_caching.py` (it captures the system
    instruction and the user contents): a `SceneSpec` with
    `visual_hints=["beat_001: the circle stretches"]`, `plan_source="fallback"`, one
    transformation with `parameters={"description": "the circle stretches into an ellipse"}`,
    and a context with `visual_world=["Motif data_cloud: dots"]`. Assert the user prompt
    contains `"the circle stretches"`, `'"plan_source": "fallback"'` and
    `"the circle stretches into an ellipse"`; the system prompt contains
    `"VISUAL WORLD (SAME IN EVERY SCENE)"` and `"Motif data_cloud: dots"`; and the system
    prompt still starts with `_codegen_static_system_prompt()`.
19. `test_codegen_static_prompt_has_the_new_rules` — `_codegen_static_system_prompt()`
    contains `"FILL THE FRAME"`, `"MOTION AND STAGING"`, `"ON-SCREEN TEXT"`,
    `"plan_source"` and does not contain `"hypotenuse"`.

The next three use the planned-script fakes from `tests/test_lesson_plan.py` (a plan of
two acts, 40 and 60 words, total 100; `agent._complete_unified_script_json` replaced by a
fake that returns queued script dicts and records prompts). Call
`agent._generate_planned_video_script(plan, segments, config, [], None)` directly.

20. `test_short_planned_script_is_retried_once_and_extended` — first response: beats
    totalling about 50 words; second: about 96 words. The result has the second
    response's beats; exactly two calls were made; the second prompt contains
    `"YOUR PREVIOUS DRAFT"`, `"The script is"` and the first draft's narration.
21. `test_planned_script_of_full_length_is_not_retried` — first response totals about 96
    words: exactly one call, and the prompt does not contain `"YOUR PREVIOUS DRAFT"`.
22. `test_shorter_retry_keeps_the_first_draft` — first response about 60 words, second
    about 40: the result has the first response's beats; exactly two calls were made.

## Gates (run them, report each result in one line)

- G1 new tests — CHECK: `python -m pytest tests/test_visual_plan_repair.py -q -p no:cacheprovider` — EXPECT: `22 passed`
- G2 neighbouring suites — CHECK:
  `python -m pytest tests/test_space_and_overlap.py tests/test_visual_continuity.py tests/test_visual_archetypes.py tests/test_prompt_caching.py tests/test_script_architect.py tests/test_animation_generator.py tests/test_scene_controller.py tests/test_lesson_plan.py tests/test_revision.py tests/test_hybrid.py -q -p no:cacheprovider`
  — EXPECT: `0 failed`
- G3 full suite, once, at the end — CHECK: `python -m pytest tests -q -p no:cacheprovider`
  — EXPECT: `0 failed`

## Report (your final message, under 300 words)

1. Files changed, one line each.
2. Each gate's result line.
3. Any existing test whose expectation you changed beyond Edit 6, and why.
4. Anything in this spec that did not match the code you found, and what you did about it.
