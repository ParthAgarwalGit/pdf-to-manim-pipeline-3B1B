# Professor Control Layer & Presenter Bridges — Design

**Date:** 2026-11-07
**Status:** Proposed — awaiting approval before implementation

## Goal

Let a professor shape the generated video without touching code:

1. **Piecewise edits** — rewrite a section, or replace a generated analogy/metaphor with their own, described their way.
2. **Concept density** — cut analogies or tangents they don't want; go deeper on concepts they care about.
3. **Presenter bridges** — short on-camera segments by an avatar of the professor (cloned voice + talking head) that bridge between Manim visualizations.
4. **Professor input at both ends** — a free-form brief alongside the PDF at the start, and feedback on the finished video afterwards. Feedback triggers a revision that changes *only* what the feedback requires.
5. **Quality gate before delivery** — audio/video sync and media quality are verified, and repaired where possible, before any video (first run or revision) reaches the professor.

## Why This Needs New Architecture

- The pipeline is one shot, PDF → mp4. `run_pipeline` has no entry point that starts from an existing run, so any change means a full, paid re-run.
- Analogies are untyped prose inside `Beat.narration`. Nothing can say "this is the pizza analogy, and beats 7 and 12 call back to it", so you can't replace it consistently or cut it cleanly.
- Depth is a set of global prompt knobs (`--depth`, `--verbosity`, `--granularity`). Nothing works per concept.
- Every scene is Manim. There is no non-Manim scene type, and the composer assumes Manim video plus TTS audio.

## Design Principles

1. **One editable source of truth: the lesson script.** Professors edit *meaning* (beats). Scenes, code, audio and video are always re-derived from it, never hand-edited.
2. **Professor text is law.** A beat the professor authored or approved is `locked`. No LLM pass (grader critique, retries, seam repair) may rewrite it.
3. **Structured ops underneath, natural language on top.** "Drop the pizza analogy" compiles to typed ops. The professor sees a diff, approves it, and only then is it applied.
4. **Incremental rebuilds.** Only changed beats are re-planned, re-generated and re-rendered. Unchanged scenes reuse their `.py`/`.mp4`/`.mp3`.
5. **Coverage stays honest.** A professor's cut is recorded as `omitted-with-reason: professor` in the coverage audit, never silently dropped. This matches the Hybrid (C) coverage posture.
6. **Flag-gated, default off.** With every flag off, output is byte-identical and the keyless suite stays green.

---

## Part A — Typed Beats (additive schema change)

`src/schemas/video_script.py`, `Beat` gains (all with defaults, so old storyboards still load):

| Field | Type | Purpose |
|---|---|---|
| `role` | `core \| analogy \| example \| derivation \| intuition \| recap \| bridge` | What the beat *does*. Ops and density filters key off this. |
| `analogy_id` | `Optional[str]` | Shared by an analogy's introduction and all its callbacks. |
| `depth` | `int` 0–3 | 0 = essential to the argument; higher = elaboration. |
| `author` | `generated \| professor` | Provenance. |
| `locked` | `bool` | Never rewritten by any LLM pass. |

`VideoScript` gains `analogies: List[Analogy]`:

```python
class Analogy(BaseModel):
    analogy_id: str              # "analogy_water_pipes"
    label: str                   # "water flowing through pipes"
    target_concepts: List[str]   # ["voltage", "current"]
    mapping: Dict[str, str]      # {"water pressure": "voltage", ...}
    origin: Literal["generated", "professor"]
    professor_brief: str = ""    # the professor's own description, kept verbatim
```

The segment script prompt (`_build_segment_script_prompt`) asks the LLM to tag `role`, `depth` and `analogy_id` and to declare analogies. The mock path tags deterministically (`role="core"`, `depth=0`).

## Part B — Persisted Lesson Script + Revision Runs

- **Persist `lesson_script.json`** (the full `VideoScript`, analogies and concept plan) next to `storyboard.json`. `ocr_output.json` and `storyboard.json` are already persisted, so a run can be rehydrated without re-OCR.
- **Component fingerprints per scene**, stored in the manifest. Each artifact is rebuilt only when its own inputs change:

  | Fingerprint | Inputs | Rebuilds |
  |---|---|---|
  | `narration_fp` | beat texts, voice, pronunciation lexicon | TTS audio + timing track |
  | `visual_fp` | visual intent, timing track, render quality | scene code + render |
  | `presenter_fp` | bridge text, avatar settings, audio | avatar clip |

  For example, fixing a mispronunciation re-runs TTS for one scene. Its render is reused if the new timing track still fits the existing beat windows, and re-timed otherwise. A visual-only fix ("the label overlaps") keeps the audio.
- **`Orchestrator.run_revision(base_run_dir, edit_plan)`** does the following:
  1. Loads the base run's script, segments and OCR.
  2. Applies the ops.
  3. Re-chunks **only the affected segments**. The chunker already never crosses segment boundaries, so an edit in segment 3 leaves every other segment's scenes and fingerprints unchanged.
  4. Re-plans visuals, generates code, renders and runs TTS only for scenes whose fingerprint changed.
  5. Reuses all other artifacts and re-composes the final video.
- Each revision writes a new sibling run dir (`<pdf>/<run_id>/rev_002/`) with `revision.json` (parent run, ops applied, beat diff, cost delta). Base runs are never mutated, which gives version history and undo for free.

This is also the "resume + per-scene regenerate" item from the product gap analysis. Part B serves both.

## Part C — Edit Operations (`ScriptEditorAgent`)

New agent `src/agents/script_editor.py`. It takes a `VideoScript` and a list of `EditOp`, and returns the edited script plus a diff and validation report.

| Op | Effect |
|---|---|
| `rewrite_beat {beat_id, text}` | Exact professor text. The beat becomes `author=professor, locked=True` and its visual is re-planned. |
| `revise_beat {beat_id, instruction}` | LLM rewrites one beat, with its neighbours as context. |
| `replace_analogy {analogy_id, brief, mapping?}` | Rewrites the introduction beat **and every callback beat** consistently, building on the professor's brief (quoted phrases are kept verbatim). |
| `add_analogy {concept, brief, after_beat?}` | Inserts a professor analogy beat. |
| `remove {beat_id \| analogy_id}` | Deletes the beats. Only the two seam sentences around the gap are repaired. |
| `expand {target, levels: 1–2, focus}` | Inserts new `depth+1` beats after the target. `focus` is `derivation \| example \| intuition` or free text. |
| `condense {beat_ids}` | Merges beats into fewer, shorter ones. |
| `set_density {scope, level, analogies}` | See Part D. |
| `presenter {insert \| remove \| edit}` | See Part E. |

**Post-edit validation** (deterministic, before anything is spent):
- **Concept ledger:** no concept is used before it is introduced. If a `remove` deletes a beat that introduces a concept still referenced later, the op is rejected and the error says which later beats depend on it.
- **Coverage audit:** professor cuts are marked `omitted-with-reason: professor`.
- **Locked beats** are byte-identical before and after (enforced by assertion).
- **Estimate:** changed-scene count, length delta and cost delta.

**Front ends:**
- `pdf2manim revise RUN_DIR --ops edits.yaml` (deterministic, scriptable, testable).
- `pdf2manim revise RUN_DIR --say "Replace the pizza analogy with a traffic analogy: cars are electrons, lanes are wires. Go deeper on eigenvectors with a shear example."` An LLM compiles this into `EditOp` JSON (schema-constrained). The CLI prints the diff and estimate and asks for confirmation before running.
- Later, a Gradio review tab (Phase 5): beats grouped by segment, with role chips and per-beat buttons for Cut, Rewrite, Go deeper, Simplify and Replace analogy, plus a diff panel and a "Render N changed scenes (~$X)" button.

## Part D — Concept Density

Density can be steered at two moments.

1. **Before scripting (cheapest).** A new outline step produces a `ConceptPlan`: one entry per concept per segment.
   ```python
   class ConceptPlanEntry(BaseModel):
       concept: str
       segment_id: str
       depth: Literal["skip", "brief", "standard", "deep"]
       analogy: Literal["auto", "none", "professor"]
       professor_analogy_brief: str = ""
       professor_notes: str = ""   # "students always confuse X with Y — address it"
   ```
   The professor adjusts it (`--plan concept_plan.yaml`, or in the UI), then the segment script prompt receives per-concept instructions. `skip` concepts are logged in the coverage audit as professor omissions.

2. **After scripting.** `set_density {scope: global|segment|concept, level: essential|standard|rich, analogies: on|off}`.
   - **Lowering density is deterministic.** Drop beats with `depth > k`. When `analogies: off`, also drop `role=analogy` beats and the callbacks that point to them. Then repair the seams. No full regeneration is needed.
   - **Raising density** compiles to `expand` ops on the in-scope concepts (this does need generation).

The existing `--depth`/`--verbosity` flags keep working as global defaults that seed the plan.

## Part E — Presenter Bridges (professor avatar)

**What it is.** Short (5–20 s) on-camera segments by the professor's avatar: an opening hook, transitions between segments ("now that we have the derivative, here's the real question…"), "why this matters" moments before a hard concept, and a recap. Each one is a `role=bridge` beat that renders as a `presenter` scene instead of a Manim scene.

**Placement policy.** The planner proposes bridges at segment boundaries and before high-difficulty concepts, capped at about 1 per 90 s of Manim and at most 20 % of runtime. The professor adds, removes or edits them like any beat. Bridge text must be professor-approved, which locks it, before it is rendered.

**Pipeline.**
```
bridge beat ─► TTSProvider (professor voice) ─► audio
                                               └─► AvatarProvider(audio, avatar_id) ─► presenter.mp4
presenter.mp4 ─► normalize (resolution / fps / pix_fmt matching Manim output) ─► composer (in scene order)
```
- **Layouts:** `full` shows the avatar over a branded background. `pip` places the avatar in a corner over the dimmed last frame of the previous Manim scene (an ffmpeg overlay), which keeps the visual thread.
- **`TTSProvider` interface:** `openai` (current) | `elevenlabs` (professor voice clone). One voice for the whole video, so bridges and Manim narration sound like the same person.
- **`AvatarProvider` interface:**
  - `none` is the default. Bridge narration plays over a title card, so the feature degrades gracefully and tests stay hermetic.
  - A hosted provider (HeyGen / D-ID / Tavus class) uses a photoreal talking head trained on a short video of the professor.
  - `local` uses an open-source lip-sync model (MuseTalk / LatentSync class) on our own GPU. This fits the AMD Developer Cloud story, but ROCm support is unverified.
- **"3D".** v1 targets a photoreal 2D talking head, which is what avatar vendors do well and cheaply. A true rigged 3D avatar is a separate, much larger build and is out of scope.
- **Timing dependency.** Lip-sync needs audio before video, which matches `--tts-first` on `vignesh/v1.0`. Merge that first rather than building a second narration-first path.

**Consent & disclosure (hard requirements, not polish):**
- A professor profile holds the consent record (a signed statement, plus the vendor's own identity verification). An avatar or voice clone can only be used by its owner's account.
- An on-screen "AI-generated presenter" disclosure, plus a credits line.
- Vendor keys come from env only and never appear in artifacts. This is already the convention for LLM keys.
- Presenter seconds are metered by the cost tracker, with a `--max-presenter-seconds` cap.

## Part F — Professor Brief at the Start

The professor gives free-form thoughts alongside the PDF (`--brief brief.md`, or a text box in the UI). For example:

> My students are 2nd-year EE and struggle with phasors. Use the rotating-arm picture, not the clock analogy. Skip 4.3. Spend more time on resonance. Under 10 minutes. I write j, not i.

**`ProfessorInputInterpreter` (pre-generation mode).** A schema-constrained LLM call compiles the brief into `ProfessorDirectives`:

| Directive kind | Feeds into |
|---|---|
| Concept depth / skip / analogy ("use the rotating-arm picture") | `ConceptPlan` entries (Part D); a professor analogy becomes an `Analogy` with `origin=professor` |
| Scope ("skip 4.3", "only pages 3–9") | Segment selection; logged as `omitted-with-reason: professor` |
| Notation ("j not i", "bold vectors") | Notation bindings in `SharedContext` → the codegen and LaTeX prompts |
| Global (length, audience, tone, terminology) | `Configuration` (`target_minutes`, audience, …) |
| Presenter preferences | Part E settings |
| Anything unmapped | Passed **verbatim** to the script prompt as professor notes. Nothing in the brief is dropped. |

- **Read-back before spending.** The interpretation is saved as `brief_interpretation.json` and shown as "Here's how I understood your brief." The UI requires a confirm; the CLI prints it, and `--yes` skips the prompt.
- **Directives are binding and checked.** After scripting, a deterministic compliance pass verifies each directive:
  - section 4.3 has no beats;
  - no beat uses the rejected analogy;
  - generated LaTeX uses `j`;
  - the estimated length is within target.

  A violation regenerates only the offending segment (at most 2 tries). Anything still unmet is reported in the delivery report, never silently ignored.

## Part G — Feedback on the Finished Video (minimal revision)

The same interpreter runs in **post-output mode**: feedback compiles into `EditOp`s (Part C) anchored to beats.

- **Timestamp anchoring.** "At 2:13 the label overlaps" maps a timestamp to a scene and beat through the composed-scene durations already recorded in the manifest.
- **Feedback is classified by what it invalidates**, so only those artifacts are rebuilt (via the component fingerprints in Part B):

  | Feedback type | Example | Rebuilds |
  |---|---|---|
  | Content | "replace this analogy", "go deeper here" | script ops → TTS + visual for the affected beats |
  | Visual | "label overlaps", "show the vector, not the text" | codegen + render for that scene; audio reused |
  | Pacing | "too fast at 3:40" | timing track stretch / pauses → re-render; text unchanged |
  | Audio | "it's 'Euler' pronounced *oiler*" | pronunciation lexicon → TTS for scenes containing the word |
  | Presenter | "rephrase the intro" | TTS + avatar clip for that bridge only |

- **Before:** the diff, the changed-scene count and the cost estimate are shown, and the professor approves.
- **After:** the delivery includes a change log ("Changed: scenes 4–5, analogy replaced. Reused unchanged: 28 scenes"). Unchanged scenes are reused byte for byte.
- **Validation:** every revision passes the full quality gate (Part H) before delivery: changed scenes in full, plus whole-video checks.

## Part H — Pre-Delivery Quality Gate

**What the code does today.** Sync is handled after the fact, and nothing blocks delivery:
- `AudioVideoComposer._compose_scene` hides a mismatch by freezing the last video frame (dead visuals) or padding silence.
- `VideoQualityReviewer` flags drift over 2.0 s only as `medium`.
- `_run_final_video_review` is report-only: its findings never trigger a repair and never hold back the video.
- Sync is also only checked per scene. Nothing checks that the visual appears *when the narration mentions it*.

The gate has three layers.

**1. Sync by construction (prevent).**
- **Narration first** (`--tts-first` from `vignesh/v1.0`), extended to a beat-level **timing track**:
  - Audio is generated per beat, or word/character timestamps come from TTS alignment or forced alignment.
  - The result is `timing_track.json`: `beat_id → [t_start, t_end]` within the scene audio.
- **Runtime sync helper** in the generated scene base class. `self.sync_to("beat_003")` waits until that beat's start time using Manim's own scene clock. Animation timing is then enforced at render time, not left to the LLM's arithmetic (today's `# Total animation time: Xs` comment convention).
- **Validator lint:** every beat in the plan has a `sync_to` call, in order.

**2. Measure (deterministic post-render checks).**

| Check | Pass criterion (initial thresholds, to be tuned on the fixed lecture test set) |
|---|---|
| Scene A/V duration | \|video − audio\| ≤ 0.3 s; frozen tail ≤ 1.0 s |
| Beat alignment | Visual activity (existing `frame_analysis` frame-diff) begins within ±0.75 s of each beat's start; no beat window is fully static |
| Narration intact | Composed audio ≥ TTS duration − 50 ms (nothing clipped); no mid-scene silence > 2 s |
| Narration fidelity | ASR transcript of the final audio vs script: word error rate ≤ 10 % (catches skipped or garbled TTS) |
| Final file integrity | Per-segment A/V stream start/duration drift ≤ 100 ms across concat; uniform resolution / fps / codec / sample rate |
| Loudness | Every clip normalized to −16 LUFS (ffmpeg `loudnorm`), true peak ≤ −1 dBTP, including presenter clips |
| Visual QA | Existing frame QA: blank, static, overlap and border-ink checks |
| Presenter | Clip duration matches its audio; lip-sync score where the provider reports one |
| Directives | Part F compliance |

**3. Repair (bounded) and deliver.**
- Each failing check maps to the **smallest repair**:
  - re-time and re-render (no LLM call);
  - re-generate one beat's TTS;
  - re-normalize audio;
  - re-codegen one scene, with the finding as feedback.
- At most 2 repair rounds per scene. Passing scenes are never touched.
- Output is `delivery_report.json` plus a plain-language summary:
  - **Ready:** "32/32 scenes passed sync and quality checks."
  - **Needs review:** "31/32 passed. Scene 14 (0:42–0:51) still shows a static frame while you describe the tangent line." The video is delivered with those scenes flagged and their timestamps.
- Critical or high findings can never produce a "Ready" status.

---

## Phasing

| Phase | Scope | Depends on |
|---|---|---|
| **P0** | Make main green (3 failing `test_prompt_caching` tests). Decide on and merge `vignesh/v1.0` (TTS-first). Remove `tmp/`. | — |
| **P1** | Typed beats (A) + `lesson_script.json` + component fingerprints + `run_revision` (B). No editing UI yet. | P0 |
| **P2** | Quality gate (H): timing track, `sync_to` helper + lint, deterministic checks, bounded repair, `delivery_report.json`. Applies to every run from here on. | P0 (parallel with P1) |
| **P3** | `ProfessorInputInterpreter` (both modes) + brief/directives with compliance checks (F) + `ScriptEditorAgent`/ops + feedback revisions via `pdf2manim revise` (C, G). | P1, P2 |
| **P4** | `ConceptPlan` outline step + `set_density` (D). | P3 |
| **P5** | Bridge role + `TTSProvider`/`AvatarProvider` (`none` + one hosted) + composer integration + consent (E). | P1, P2 (parallel with P3/P4) |
| **P6** | Gradio flow: PDF + brief → read-back → video + delivery report → feedback box → revision. | P3–P5 |

The quality gate (P2) comes before the editing features because every revision is only as trustworthy as the gate that checks it.

## Testing

- All new behaviour sits behind `PipelineConfig` flags that default to off. A flag-off regression test asserts byte-identical storyboard and manifest output.
- **Ops:** golden before/after script fixtures for each op, a locked-beat invariant test, and ledger-violation rejection tests.
- **Revision:** a fingerprint-reuse test (editing one beat in segment 2 re-renders only segment 2's scenes; the mock renderer counts calls).
- **Providers:** fake `TTSProvider`/`AvatarProvider` in the hermetic lane, with real vendors under `live_api` only.
- **Gate:** synthetic fixtures for each check, made with ffmpeg-generated clips:
  - audio longer than video;
  - a static beat window;
  - clipped narration;
  - concat drift;
  - a loudness mismatch.

  Each fixture must fail its check and then pass after its mapped repair.
- **Brief/feedback:** golden brief → directives fixtures; timestamp → beat anchoring tests; a directive-violation regeneration test (mock LLM).

## In-Scope Files

- **Schemas:** `src/schemas/video_script.py` (Beat fields, `Analogy`), `src/schemas/storyboard.py` (`scene_type`, fingerprint), new `src/schemas/edit_ops.py`, new `src/schemas/concept_plan.py`.
- **Agents:** `script_architect.py` (tagging, ConceptPlan + directive honoring), new `script_editor.py`, new `professor_input_interpreter.py`, `tts_agent.py` (provider interface, timing track, pronunciation lexicon), new `presenter_agent.py`, `audio_video_composer.py` (presenter clips + pip overlay + loudnorm), `video_quality_reviewer.py` → new `delivery_gate.py` (checks + repair routing).
- **Utils:** `manim_engine_contract.py` (`sync_to` helper), `frame_analysis.py` (per-window activity onset).
- **Schemas:** additionally new `src/schemas/directives.py` and `src/schemas/delivery_report.py`.
- **Orchestrator:** `pipeline.py` (`run_revision`, component-fingerprint reuse, gate + repair loop, presenter stage).
- **Interfaces:** `src/cli.py` (`--brief`, `revise --ops/--say`, `--plan`, `--density`, presenter flags), `app.py` (P6).
