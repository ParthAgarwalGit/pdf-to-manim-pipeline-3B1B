# Engagement overhaul: findings, recommendations and implementation plan

16 Nov 2026. Reviewed: the pipeline code and every prompt in it, 15 past runs' logs, and
the latest finished video (`output/hybrid_gemini/svd_half/2026-11-13T06-10-33/rev_007`,
4:19, 15 scenes) frame by frame against its script, plans and generated Manim code.

All code changes are in the worktree `.claude/worktrees/rag-paper-diagrams-65d6a4`
(branch `claude/autoresearch-agent-product-2ea2c9`), on top of the uncommitted hybrid /
Gemini-presenter work that already lives there. Nothing is committed.

---

## 1. What is wrong with the videos, in one paragraph

The video is a tour of the PDF, not an explanation. It has no opening question, no
running example and no ending; it narrates proof steps the brief asked to skip; and it
stops on a power-method technicality. On screen, six of thirteen animated scenes show a
narration sentence as a title and one stock arrow, the rest draw diagrams that cover a
fifth of the frame, and nothing persists from scene to scene. Every automatic check
passed it. Each of these traces to a specific structural cause, listed next.

## 2. Structural issues (ranked by how much they cost the viewer)

### S1. The "one script" is fourteen independent small-model calls in PDF order

- **Evidence.** The run log shows 14 `gpt-5.4-mini` calls, one per source segment. The
  attention-paper run from July has the same shape: scenes 2, 3 and 4 each re-introduce
  positional encoding and masked decoding.
- **Cause.** `ScriptArchitectAgent._generate_video_script` returns
  `_generate_segmented_video_script` whenever segments exist, and segments always exist.
  The whole-document path (the one that runs on the strong model) is unreachable. Each
  segment prompt says "Explain THIS segment", every segment "keeps a voice"
  (`MIN_SEGMENT_WORDS`), the length budget is split by segment size, and scenes may not
  cross a segment boundary.
- **Consequences.** PDF order becomes video order. A segment that is nothing but a proof
  must be narrated, so "Skip the proofs" cannot be honoured (scenes 10 to 12 of the SVD
  video are the proof of the best-rank-k theorem). No call ever sees the whole document,
  so nothing can choose a hook, a running example or an ending. The later "flow edit"
  can only remove repeats; it cannot reorder or restructure.
- **Fix (implemented, WP1).** A planning call over the whole source returns a
  `LessonPlan`: the one question the video answers, the hook, a running example with
  actual numbers, 2 to 4 recurring visual motifs, the acts in viewing order, and a
  decision for every source segment (covered, compressed, deferred, omitted with a
  reason). A second call writes the whole script from the plan. Scenes follow acts, not
  segments. The segmented path remains as the fallback (`--no-lesson-plan`, planning
  failure, or a source too long for one call).

### S2. Visual plans are thrown away instead of repaired

- **Evidence.** `Visual intent fallback used for N/M scenes` appears in every recent run:
  6/16, 8/17, 9/17, 7/18, 2/3. The reasons logged are countable: "9 concurrently visible
  objects; maximum is 7", "16 objects; maximum is 15".
- **Cause.** `_plan_visual_intents` validates the planner's output and, after two
  retries, replaces the entire plan with `_generate_fallback_visual_intent`: the
  narration's first sentence as a `Text` title (`intro_text`) plus one generic primitive
  (`archetype_visual`). A scene with a single beat cannot even be split.
- **It spreads.** The stub's objects are listed in `scene_state_out`, so the next scene's
  planner is told to start from `intro_text` and `archetype_visual`. In the SVD run they
  appear in six further scenes, each time using two of the seven object slots.
- **Fix (implemented, WP2).** `src/utils/visual_plan_repair.py` fixes every countable
  defect deterministically: fills in missing ids, purposes and beat references, drops
  dangling transformations, and brings a plan under its budgets by retiring, delaying or
  dropping the lowest-priority objects (labels first, then extra formulas, then minor
  drawings; the scene's main formula, its coordinate system and inherited objects go
  last, and the last drawing is never removed). The stub is now a last resort; when it does happen its title is removed, it is
  marked `visual_plan_source="fallback"`, the code generator is told to design the scene
  from the script's visual hints, and nothing from it is carried forward.

### S3. Motion is planned and then silently lost

- **Evidence.** 11 of 13 animated scenes have 0 transformations and 0 beats in
  `storyboard.json`.
- **Cause.** The planner prompt asks for `transformations` and `beats` but never states
  their fields. `Transformation` requires `source_object_id` and `timing`; `VisualBeat`
  requires `start_time` and `end_time`. Items without them fail validation and
  `_validate_items` drops them without a word. What reaches the code generator is a list
  of objects with appear and disappear times, which is a slide deck.
- **Fix (implemented, WP2).** The new planner prompt documents both shapes exactly and
  makes motion step 2 of shot design ("for each beat decide what changes on the hero").
  The code generator now also receives each transformation's description and the script
  writer's per-beat visual hints, neither of which it ever saw.

### S4. A layout pass moves drawings off the things they are drawn on

- **Evidence.** In the stored plan for scene 2 the unit square is at y = 3.4, above the
  axes it should sit on; the ellipse is at (2.3, 0.2) but its own axes are at y = 1.45
  and y = 2.65; subspace rectangles sit at y = 3.4 in three scenes.
- **Cause.** `_resolve_bbox_overlaps` pushes apart any two boxes that overlap in space
  and time, then clamps to the safe area. Drawings that share space on purpose (a vector
  on its plane, the axes of an ellipse) are treated as collisions.
- **Fix (implemented, WP2).** Only text objects are nudged apart from each other.
  Drawings are never moved. The planner prompt says overlap between a host and what
  lives on it is expected.

### S5. The code-generator prompt asks for small diagrams and on-screen sentences

- **Evidence.** Diagrams of about 5.5 by 3.8 units in a 14.2 by 8 frame. Titles such as
  "That is enough to reconstruct A from its singular directions" and "We can find that
  leading direction with the power method" are narration sentences. In the middle of
  scene 2 the code fades everything out to start a different diagram.
- **Cause.** The prompt caps axes at `x_length ≤ 5, y_length ≤ 3`. Its CONTENT FIDELITY
  section says the animation "MUST show every formula, definition, and example the
  narration mentions", with trigonometry examples in a prompt used for every subject. It
  demands a `self.wait()` between every pair of `play` calls while also saying never pad
  and while `sync_to` owns timing. Nothing asks for a persistent main visual.
- **Fix (implemented: prompt text by hand, wiring in WP2).** New sections FILL THE FRAME,
  MOTION AND STAGING and ON-SCREEN TEXT; the trig examples and the size caps are gone;
  the wait rule no longer contradicts sync; a per-run VISUAL WORLD block gives every
  scene the same example values, motif names and colors.

### S6. The reviews measure the wrong things

- **Evidence.** The storyboard grader passed the SVD storyboard at 7.3. The render-time
  layout audit reported 0 issues. The vision reviewer said "legible: true" for every
  scene. The delivery gate's only complaints were two scenes at -17.6 LUFS.
- **Cause.** The grader runs on `gpt-5.4-nano`, sees 200 characters of each scene and
  scores "completeness". The layout audit looks only for text overlapping text. The
  vision reviewer is asked only which frame type suits the scene.
- **Fix (implemented, WP3).** For planned videos the grader scores learning path, accuracy,
  narration economy, visual design and hook-and-payoff on full narration, on
  `gpt-5.4-mini`, and reports source coverage separately without gating on it (the
  "Hybrid C" coverage posture decided in July). The layout audit now also flags a
  sentence shown as text and a picture that never reaches 30% of the frame, and both
  feed the existing regeneration loop. The vision reviewer reports defect codes.

### S7. Bridges have nothing to say

- **Evidence.** The video opens with "Before we get into the details, I want you to
  picture what this matrix is really doing" (no matrix has been mentioned) and the
  mid-video bridge says "So now we have the building blocks."
- **Cause.** The bridge writer sees only the two neighbouring beats and is told never to
  introduce content. There is no hook or question for it to voice.
- **Fix (implemented, WP1).** With a plan, the opening bridge is the plan's hook, word
  for word (the planner writes it to be said on camera); transitions voice the next
  act's viewer question; there is no recap bridge, because the script already ends on
  its payoff.

### Smaller issues

- **Four speaking rates** (3.0 in the chunker and code generator, 2.8 in the length
  budget, 2.5 in the frame director, 2.9 in the Gemini presenter). Each is defensible
  where it is used and measured audio overrides estimates in hybrid runs. Not changed;
  worth one shared constants module when timing is next touched.
- **`verifier_mode` defaults to "off"**, so the per-scene vision repair loop never runs.
  Turning it on is a cost decision.
- **The presenter clip background and the Manim background differ** (grey studio
  gradient against pure black), which makes every cut between them visible.

## 3. How the good channels do it, and what that means for this pipeline

Technique-level observations, each tied to a change above. The video descriptions are
from memory, not re-watched for this report; treat them as illustrations of the technique.

| Technique | Where you see it | What the pipeline now does |
|---|---|---|
| **One question, asked early.** The first half-minute plants a specific question; everything after serves it. | 3Blue1Brown's "But what is a Fourier transform?" opens on pulling one pure tone back out of a mixed signal; Welch Labs' "Imaginary Numbers Are Real" opens on an equation that seems to have no solution. | `central_question` and `hook` in the lesson plan; act 1 is the hook; the grader scores hook-and-payoff. |
| **One picture that evolves.** A single visual device is built once and then transformed for the rest of the video. | The grid that deforms under a matrix throughout "Essence of linear algebra"; the winding machine in the Fourier video. | `motifs` and the VISUAL WORLD block; "the hero stays on screen, things change on it"; carry-over limited to designed objects. |
| **Show, then name.** The viewer watches the thing happen before it gets a name or a formula. | Standard 3Blue1Brown order: animation, then the term, then the equation as a label for what was just seen. | Planner and writer rules; the visual planner allows a formula only after the picture has given its parts meaning. |
| **A concrete case with real numbers.** | StatQuest works one tiny dataset by hand; Welch Labs works on printed paper with actual values. | `running_example.values`; "whenever a number is said, it is one of the running example's". |
| **The voice explains, the screen shows.** Almost no prose on screen; labels and formulas only; formula terms colored to match the objects they describe. | 3Blue1Brown throughout. | ON-SCREEN TEXT rules in planner and code generator; the audit flags any 7-word text. |
| **Economy.** Narration does not describe what the picture already shows; no announcements, no recaps. | Compare a transcript of any of these videos with a lecture transcript. | Beats of 15 to 45 words (was 30 to 60), an Economy section in the writer prompt, a narration-economy criterion in the grader. |
| **Proof as a picture.** One reason the result is believable, never lemma by lemma. | 3Blue1Brown's treatment of most theorems. | Coverage decisions; "never plan an act whose job is to walk through the steps of a proof". |
| **An ending that answers the opening.** | Welch Labs and 3Blue1Brown both close by returning to the opening picture. | The last act is the payoff; no recap bridge after it. |

Not yet in the pipeline, and worth doing next (see section 6): camera moves, deliberate
silence, music, and question cards between acts.

## 4. Prompt changes (all written by hand; no prompt text was delegated)

| Prompt | File | Change |
|---|---|---|
| Lesson plan (new) | `src/prompts/narrative.py::lesson_plan_prompt` | Audit, one central question, running example, motifs, acts in viewing order, coverage for every segment, self-check. |
| Script from plan (new) | `src/prompts/narrative.py::script_from_plan_prompt` | Plan is binding; beats 15 to 45 words; Economy; show before naming; proofs as one reason; hook and ending rules; visual hints must describe change and be drawable with simple shapes. |
| Spoken style | `src/prompts/narrative.py::SPOKEN_STYLE_RULES` | Adds varied openers and more banned stock phrases ("there's a catch", "it turns out", "here's the thing"). |
| Bridges (new, planned runs) | `src/prompts/narrative.py::bridge_prompt` | One job per slot; a bridge raises a question or lands a conclusion and never explains. |
| Visual planner | `src/prompts/visual.py::visual_plan_static_prompt` | Shot design (hero, motion, support, staging), budget stated with its consequence, text discipline, on-host placement, exact shapes of `transformations` and `beats`. |
| Planner context (new) | `src/prompts/visual.py` | Lesson context, visual world and a per-beat time schedule for each scene. |
| Code generator | `src/prompts/codegen.py::codegen_static_system_prompt` | FILL THE FRAME, MOTION AND STAGING, ON-SCREEN TEXT; `plan_source` and `visual_hints` semantics; trig examples and size caps removed. |
| Quality contract | `src/utils/manim_engine_contract.py::build_3b1b_quality_prompt` | Wait rule no longer conflicts with sync; three new anti-patterns. |
| Storyboard grader (new rubric) | `src/prompts/review.py::storyboard_review_prompt` | Learning path, accuracy, narration economy, visual design, hook and payoff; coverage reported only. |
| Frame reviewer | `src/prompts/review.py::frame_issue_instructions` | Defect codes: under-filled frame, text overlap, narration as text, illegible, off frame, mismatch. |

## 5. Implementation: who wrote what

Written by hand (prompts and the logic where a wrong call is expensive):
`src/prompts/*`, `src/schemas/lesson_plan.py` (the plan contract and its tolerant
parser), `src/utils/visual_plan_repair.py` (plan repair), the quality-contract edit, and
the three work-package specs in this folder.

Also by hand, after the packages landed: the audit fix that reads `tex_string`, the
one-line wiring of the spoken-text guard and its test, and the two updated test
expectations described in section 7.

Written by Opus 5.5 (extra-high effort, via the CLI) from those specs, wiring and tests:

| Package | Spec | Scope |
|---|---|---|
| WP1 | `wp1-lesson-plan-and-script.md` | Plan and script calls in the architect, act-aware chunking, plan-aware bridges, persistence (`lesson_plan.json`, plan inside `lesson_script.json`), `--lesson-plan/--no-lesson-plan`. |
| WP2 | `wp2-visual-planning-and-codegen.md` | Repair before validation, harmless stub, text-only overlap pass, new planner and code-generator prompts wired in, hints / visual world / descriptions passed to the code generator. |
| WP3 | `wp3-review-gates.md` | Planned-video grader rubric, layout audit for sentences and under-fill, frame-reviewer defect codes. |

Schema changes are additive with defaults (`Beat.act_id`, `VideoScript.plan`,
`Configuration.lesson_plan`, `Configuration.visual_world`, `Scene.visual_plan_source`,
`SceneSpec.visual_hints`, `SceneSpec.plan_source`, `SharedContext.visual_world`). None
of them enters a scene fingerprint, so older runs still revise without re-rendering.

## 6. Recommended next, not implemented here

1. **A full paid render to calibrate.** Everything below section 7 is verified on plans
   and scripts, not on rendered frames. The fill threshold (30%), the sentence threshold
   (7 words) and the grader pass mark (6.0) are first guesses.
2. **Camera.** Scenes are plain `Scene`, so nothing can zoom into the part being
   discussed. `MovingCameraScene` with two or three safe moves (frame a region, restore)
   would do more for focus than any amount of highlighting.
3. **Silence and music.** Narration runs wall to wall. A half-second of silence after a
   key reveal, and a quiet bed under the whole video, are standard on every channel
   above and cost almost nothing in the composer.
4. **Presenter and Manim on the same background** (render Manim on `#1C1C1C`, which the
   split frames already use).
5. **Semantic components** (phase D of the July retention plan): reusable builders such
   as "point cloud with best-fit line" or "circle to ellipse under a matrix" that the
   planner can name instead of assembling from primitives every time.
6. **A contact-sheet approval step before the HD render** (phase E of the July plan).
7. **One pacing-constants module** for the four speaking rates.

## 7. Results (16 Nov 2026)

### What was run

- **Tests.** `python -m pytest tests -q`: 1227 passed, 0 failed, 10 deselected (baseline
  before any change: 1172 passed). 55 new tests: `tests/test_lesson_plan.py` (19),
  `tests/test_visual_plan_repair.py` (22), `tests/test_review_gates.py` (12), plus 2 in
  `tests/test_space_and_overlap.py`.
- **Planning dry run through the real pipeline**, three times, last on the final code:
  `convert svd_half.pdf --brief output/hybrid_gemini/brief.txt --target-minutes 5
  --render-backend hybrid --presenter-engine gemini --bridges propose --approve-bridges
  --dry-run` (same PDF, brief and flags as the 13 Nov video). Output:
  `output/engagement_check/svd_half/2026-11-16T15-48-09/`; its plan, narration and review
  are copied to `evidence/`.
- **Two scenes generated and rendered at low quality** through the new planner and
  code-generator prompts (stills in `evidence/after_trial_scene_00{2,4}.jpg`, one
  generated file in `evidence/after_trial_scene_002.py`). Both rendered on the first
  attempt; lengths 30.5 s and 28.1 s against 30.7 s and 28.0 s of narration.
- **Layout audit, live**, on one of those renders: 29 observations, no overlap, no
  sentence on screen, peak frame fill 0.36.
- **Persistence and revision of a planned run** (keyless, on the final dry-run folder):
  `storyboard.json` carries the visual world (11 lines) and `lesson_script.json` carries
  the plan; revising with an unchanged script reuses all 10 scenes; removing one beat
  keeps the plan, rebuilds only the affected scenes and never mixes acts in a scene.
- **Lint.** `ruff check src tests` reports 4 errors, all `E731` in
  `tests/test_narration_flow.py`, a file from the 13 Nov session that this work did not
  touch. Everything added here is clean.

Lesson planning is on by default (`--no-lesson-plan` restores the segment-by-segment
script). The live checks above used the project's API keys: three planning dry runs, two
plan-and-script trials, one four-scene planner trial and two code-generation calls.

### Before and after, same PDF and brief

| Measure | Before (run 2026-11-13T06-10-33) | After (dry run 2026-11-16T15-48-09) |
|---|---|---|
| How the script is written | 14 calls, one per PDF segment, small model | 1 plan call + 1 script call, strong model |
| First words | "Before we get into the details, I want you to picture what this matrix is really doing." | "Picture a cloud of data points. I want one line through them that captures almost everything." |
| Running example | none | five points in the plane, used in every act |
| Scenes that walk through a proof | 3 (brief said skip proofs) | 0; those segments are marked compressed, with reasons |
| Power-method scenes (not in the brief) | 3 | 0; deferred or omitted, with reasons |
| Last words | a power-method convergence detail | the answer to the opening question |
| Scenes against the cap of 12 | 16 | 10 |
| Visual plans replaced by the stub | 6 of 16 scenes | 0 of 10 scenes (7 of the 8 animated plans repaired in place) |
| Animated scenes with planned motion | 0 of 13 (six stubs had one placeholder fade-in) | 8 of 8 (2 to 13 transformations each) |
| Animated scenes with a narration sentence planned as on-screen text | 12 of 13 | 0 of 8; every planned label is 1 to 3 words |
| Narration against the 5-minute target | 4:19 rendered | 757 words, about 4:30 at the measured pace |
| Storyboard grade | 7.3 (smallest model, "completeness" rubric) | 6.3 pass (new rubric); accuracy scored 4/10, see below |

### What the runs also showed

- **The grader's accuracy flags need a human.** In the last two dry runs it scored accuracy
  4/10 and quoted the sentences it objected to (`storyboard_review.json`). One flag in the
  first run was a real overstatement. In the final run it objected to "that gap is exactly
  sigma two", which for a rank-one approximation of a two-column matrix is correct. The
  score does not block a run; it is a list for the professor to read. Do not gate on it
  until it is calibrated or moved to a stronger model.
- **The planner over-plans; repair is what makes that harmless.** 7 of the 8 animated
  plans needed cutting. Dropped objects are mostly helper lines and labels; after the first dry run
  dropped a scene's main formula, repair was changed to keep the first formula ahead of
  minor drawings.
- **Fixed during the runs** (all in the hand-written prompts unless noted): motif ids
  leaking into spoken narration (prompt rule plus a deterministic guard with a test that
  fails without it); a placeholder "hook" act when a presenter opens; the script coming in
  30% short (one retry with the short acts named, and beat count recalibrated to the
  measured 26 words per beat: the final run reached 706 of 714 words on its first draft);
  a bridge asking the question the next scene asks; a payoff act that said its answer
  three times; on-screen symbols drifting from the brief (Σ where the brief writes D).
- **Work-package notes.** WP2's spec changed behaviour that five existing tests pinned.
  Opus stopped and reported them; three were resolved by rewording the new prompts, two
  were updated because plans are now repaired on the first attempt instead of retried or
  replaced (`test_incomplete_text_only_retry_is_repaired_and_used`,
  `test_overloaded_scene_is_repaired_without_split_or_retry`). WP3's report pointed out
  that `Title` and `Tex` sentences escaped the new audit; fixed by reading `tex_string`.
  Splitting an overloaded scene (`_split_scene_at_beat_boundary`) can no longer trigger on
  the live path and is a candidate for removal.

### Not verified

- **No full render.** No narration audio, no presenter clips, no code generation for the
  dry run's scenes, no delivery gate. The before/after table compares plans and scripts;
  only two trial scenes were rendered, and those came from an earlier plan.
- **Frame fill is still modest** in the two trial renders (about half the frame width).
  The coordinate-grid rule for the planner was added after them and has not been rendered.
- **One document.** Everything live used `svd_half.pdf`. The segment-by-segment fallback
  (long sources, `--no-lesson-plan`) is unchanged and was exercised only by tests.
- **Thresholds are first guesses:** 30% frame fill, 7-word text limit, grader pass mark 6.0.
- **The brief interpreter failed validation once in three runs** and fell back to its
  rules (the brief still reached the planner as plain lines). Existing behaviour, not
  changed here.

### Added to "recommended next" by these runs

- Apply the grader's suggestions as targeted script edits (the script editor already has
  the operations) instead of pass/fail regeneration, and calibrate the accuracy criterion.
- Teach the planner to group sets (five dots, five perpendiculars) as one object so the
  7-object limit stops cutting them.
