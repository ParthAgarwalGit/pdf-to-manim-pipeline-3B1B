# Part 3: Quality Review Against 3Blue1Brown Patterns

**Goal:** Upgrade the video quality reviewer, animation generator prompts, and helper layer so generated videos match the visual grammar of 3Blue1Brown videos — progressive reveal, color-coded variables, deliberate pacing, smooth morphing, and narration-visual fidelity.

**Design Choices (user-confirmed):**
- Full stack scope: reviewer + prompts + new helpers
- Hard timing constraints (not just outlier detection)
- Progressive reveal helper + reviewer check
- LLM-based narration-visual fidelity check

---

## Task 1: 3B1B Timing Contract + Color Palette Constants

**Files:**
- Modify: `src/utils/manim_engine_contract.py`
- Test: `tests/test_schemas.py`

Add `TIMING_CONTRACT` dict with min/max `run_time` per animation family, minimum `wait()` between animation groups, and the canonical 3B1B color palette as named constants.

## Task 2: Progressive Reveal + Animation Pattern Helpers

**Files:**
- Create: `src/utils/animation_patterns.py`
- Test: `tests/test_animation_patterns.py`

Helpers:
- `progressive_write(mobject_parts, run_time_each, wait_between)` — Write equation parts one by one
- `staged_transform(source, targets, run_time_each)` — Chain of ReplacementTransform steps
- `dim_and_focus(layout, keep_ids, dim_opacity)` — Dim old objects when introducing new content

## Task 3: Color Consistency Helpers

**Files:**
- Create: `src/utils/color_manager.py`
- Modify: `src/schemas/scene_processing.py` (add `color_bindings` to SharedContext)
- Modify: `src/agents/scene_controller.py` (propagate color bindings)
- Test: `tests/test_color_manager.py`

Helpers:
- `ColorManager.assign(variable, color=None)` — assign or auto-pick a color for a variable
- `ColorManager.get(variable)` — retrieve assigned color
- `ColorManager.build_tex_color_map(variables)` — generate `tex_to_color_map` dict for MathTex

## Task 4: Upgrade Animation Generator System Prompt

**Files:**
- Modify: `src/agents/animation_generator.py`
- Modify: `src/utils/manim_engine_contract.py` (add `build_3b1b_quality_prompt()`)
- Test: `tests/test_animation_generator.py`

Inject 3B1B-specific prompt sections:
- Timing contract (min/max run_time per animation type)
- Progressive reveal instructions (never Write a complex equation in one shot)
- Color consistency rules (one variable = one color, persist across scenes)
- Smooth morphing preference (ReplacementTransform over FadeOut→FadeIn)
- Breathing room rules (wait after every major visual change)

## Task 5: Pacing & Timing Hard-Constraint Reviewer

**Files:**
- Modify: `src/agents/video_quality_reviewer.py`
- Test: `tests/test_video_quality_reviewer.py`

New method `_review_animation_pacing()`:
- Parse all `run_time=X` and `self.wait(X)` from generated code
- Flag violations of TIMING_CONTRACT (too fast, too slow, no breathing room)
- Severity: medium for individual violations, high if >50% of animations violate

## Task 6: Progressive Reveal Reviewer Check

**Files:**
- Modify: `src/agents/video_quality_reviewer.py`
- Test: `tests/test_video_quality_reviewer.py`

New method `_review_progressive_reveal()`:
- Detect `Write(complex_equation)` where equation has >3 terms
- Flag when complex MathTex is shown all at once instead of progressively
- Suggest `progressive_write()` or isolate-and-animate pattern

## Task 7: LLM-Based Narration-Visual Fidelity Reviewer

**Files:**
- Modify: `src/agents/video_quality_reviewer.py`
- Test: `tests/test_video_quality_reviewer.py`

New method `_review_narration_fidelity()`:
- For each scene, send narration + generated code to LLM
- Ask: "Does the code show every formula, definition, and example the narration mentions?"
- Parse response into findings with specific missing items

## Task 8: Anti-Pattern Detection

**Files:**
- Modify: `src/agents/video_quality_reviewer.py`
- Test: `tests/test_video_quality_reviewer.py`

New method `_review_anti_patterns()`:
- FadeOut→FadeIn instead of ReplacementTransform for equation updates
- No `self.wait()` between consecutive `self.play()` calls
- Text walls (>3 Text objects visible simultaneously)
- Inconsistent color usage (same variable, different colors across scenes)
- Missing LayoutTracker registration after object creation

## Task 9: Verification Checkpoint

Run full test suite, syntax verification, commit all changes.
