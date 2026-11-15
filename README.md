# PDF-to-Video Pipeline V2

V2 turns an educational PDF into a source-grounded, narrated Manim video. It plans the lesson around narration, lets Gemini create ordinary Manim from a compact creative brief, injects tracking and measured-audio timing locally, verifies low-quality previews deterministically, optionally performs up to three targeted multimodal repairs per scene, and only then renders the final video.

The previous implementations remain available in Git:

- `main`: original pipeline plus Gemini Manim generation
- `vignesh/v1.0`: TTS-first and raw-verifier experiment
- `vignesh/v2.0`: this architecture

## Pipeline

```text
PDF
  -> local grounded extraction; vision OCR only for low-confidence pages
  -> lesson plan with source references
  -> continuous narration and semantic beats
  -> measured TTS timing
  -> compact per-scene creative brief
  -> unrestricted raw Manim generation
  -> local tracking registration + beat retiming
  -> AST safety gate + tracked 480p15 preview
  -> deterministic lifecycle/layout diagnostics
  -> optional screenshot-guided repair
  -> accepted high-quality scene renders
  -> audio composition, ordered concatenation, and final A/V review
```

V2 does not use SceneIR or MCP internally. Gemini writes a normal `Scene`; a local AST pass changes it to a thin `TrackedScene`, registers assigned mobjects, preserves variable names for diagnostics, and retimes marked sections to narration beats. Tracking rules, region schemas, and verifier internals are not included in the model prompt.

## Creative Generation

The code-generation prompt is intentionally close to the successful raw-Manim benchmark. It contains only the lesson brief, narration, measured duration, semantic beats, source-grounded facts, and a few optional visual ingredients. Gemini can use the full approved Manim API and is encouraged to develop one visual explanation rather than produce text-heavy slides.

The only model-visible timing requirement is:

```python
self.next_section("scene_001_beat_01")
```

The model identifies where each semantic beat begins but does not calculate seconds. After generation, local instrumentation:

- adds the trusted `TrackedScene` import and base class;
- registers assigned Manim mobjects without model-written calls;
- retains names such as `triangle`, `equation`, and `point_label` in diagnostics;
- scales each section's `play()` and `wait()` durations to the measured audio beat;
- accounts for literal `for ... in range(...)` animation loops;
- leaves geometry, transformations, visual metaphors, and composition under Gemini's control.

A small deterministic visual contract is still written to each run for local fallback rendering and report compatibility. It is no longer produced by a paid visual-planning call and is not a mandatory vocabulary for Gemini.

## Verification

Generated Python is never executed before AST validation. The gate checks imports, dangerous calls, exactly one scene, a `construct()` method, script size, suspicious coordinates, and unsupported LaTeX. Generated scenes do not fail static validation for missing artifact-registration, layout-template, or region-placement calls because those are no longer model responsibilities.

During preview rendering, `TrackedScene` records stable state after `play()`, `wait()`, `add()`, and `remove()`, plus 25/50/75 percent samples inside each animation. Registered groups retain useful generated names while visible children receive deterministic leaf IDs. The creative-mode verifier detects:

- off-screen objects and clipped text
- text-text and text-geometry overlap
- unreadably small text
- excessive active visual density
- empty or static intervals
- raw scene coordinates when axes exist
- narration/video duration mismatch

Overlap detection first measures the intersection as a fraction of the smaller leaf object's bounds. Text-related candidates are then rendered separately into small alpha masks, and occupied-pixel intersection confirms whether visible marks actually collide. Text-text collisions become warnings or errors according to severity, while geometry-geometry intersections remain allowed. Tiny-text warnings use rendered relative height rather than nominal Manim `font_size`. The HTML report shows the timestamp, generated object names, exact leaf IDs, measured ratios, and evidence frame for every finding.

Because verification uses traced leaf bounds, putting several elements in a `VGroup` does not conceal a bad internal layout. Legacy strict contract/lifecycle checks remain available as unit-tested internals, but they are not imposed on creative generated scenes.

Frames are extracted at narration boundaries, issue timestamps, 25/50/75 percent, and the final frame. A repair is accepted only when it renders, changes the checkpoint visuals, lowers the deterministic issue score, and introduces no new safety error. The original always remains available.

## Installation

Requirements:

- Python 3.10+
- Manim Community
- FFmpeg and ffprobe
- MiKTeX or TeX Live

```powershell
pip install -e ".[dev]"
```

Copy `keys.example.env` to `keys.env` or `.env`:

```text
OPENAI_API_KEY=...     # TTS
GEMINI_API_KEY=...     # lesson, narration, OCR, Manim code, and vision repair
```

Missing keys use deterministic local fallbacks unless `--strict-keys` is supplied. Tests always strip credentials and never make paid calls.
V2 does not use Claude; `ANTHROPIC_API_KEY` is only relevant to legacy V1 paths.

## Usage

Normal high-quality conversion with the default one-dollar API ceiling:

```powershell
pdf2manim convert document.pdf --quality high --max-cost 1.00 --policy standard
```

Useful controls:

```powershell
# Grounded lesson and narration only
pdf2manim convert document.pdf --dry-run

# Disable paid visual repair
pdf2manim convert document.pdf --repair off

# Override the default three improving repair attempts per scene
pdf2manim convert document.pdf --repair eligible --max-repairs-per-scene 2

# Require full coverage and relaxed pacing for a long lecture deck
pdf2manim convert document.pdf --min-scenes 18 --max-scenes 20 --pacing relaxed --quality high --repair eligible

# Audit verifier findings without letting them reject a renderable scene
pdf2manim convert document.pdf --verifier-mode audit --repair off

# Generate code without Manim rendering
pdf2manim convert document.pdf --skip-render

# Withhold the official final video if any required scene/audio/check fails
pdf2manim convert document.pdf --policy strict

# Require every configured provider key; never use local content fallbacks
pdf2manim convert document.pdf --strict-keys

# Bound external and local concurrency
pdf2manim convert document.pdf --api-workers 4 --code-workers 3 --preview-workers 2 --render-workers 1
```

The default 12-scene ceiling keeps short lessons concise. `--max-scenes` is only a ceiling;
`--min-scenes` is the hard scene floor. If the lesson model returns fewer scenes than that floor,
V2 replans once and then fails clearly instead of silently accepting a short lesson. Setting either
an explicit floor or a ceiling above 12 activates full-coverage planning: every substantive page
and distinct definition, derivation, algorithm, example family, and conclusion must be represented.
`--pacing relaxed` targets 115 spoken words per minute, slows the generated audio further, and asks
for less visual work per beat so animation calls are not squeezed into short windows. Longer plans
also receive larger lesson and narration output budgets. The maximum supported scene value
is 24; splitting by topic is still preferable when one source contains several independent courses.

The default worker limits are four global API calls, four TTS jobs, three code jobs, two previews, one vision repair, one final Manim render, and one FFmpeg composition. The creative scaffold is local, so scene code generation starts as soon as its measured audio is ready.

V2 plans short lessons as roughly four to six substantial scenes. Narration beats carry a speaking role, pause cue, transition note, and one visual action. TTS receives punctuated beat paragraphs, is slowed slightly without changing pitch, and receives a two-second trailing buffer before the next scene. The measured paced audio remains authoritative for animation timing, giving the final animation room to finish after narration stops.

Code generation receives a compact summary of the previous three scenes: their visual objectives, facts already shown, and constructions that must not be replayed. A local frame-sequence check also logs `REDUNDANT_VISUAL_REPLAY` when adjacent scenes look nearly identical; this is advisory and makes no paid call.

## Resume and Reports

Press `Ctrl+C` once to stop. V2 checkpoints running work, terminates active Manim children, and keeps every successful task and paid response.

```powershell
pdf2manim resume output\document\RUN_ID
pdf2manim resume output\document\RUN_ID --max-cost 1.50
pdf2manim status output\document\RUN_ID
pdf2manim report output\document\RUN_ID
```

Retry one or more scenes by naming the exact run directory and scene IDs shown in its report. The command keeps the original grounded plan and TTS, regenerates only the selected Manim scenes, runs preview/repair/render again, and rebuilds `final.mp4` from every scene in order.

```powershell
pdf2manim retry-scenes output\document\RUN_ID scene_002
pdf2manim retry-scenes output\document\RUN_ID scene_002 scene_005 --max-cost 1.50
```

`--max-cost` is the new total ceiling for that run, not an additional allowance. Retry attempts and costs are appended to the same `run.db`, `events.jsonl`, and report.

Each ignored `output/{document}/{run_id}/` directory contains:

- `run.json`: configuration, Git commit, environment, final status, and outputs
- `run.db`: SQLite WAL task, cache, attempt, and budget state
- `events.jsonl`: complete structured stage/scene/worker/provider timeline
- `pipeline.log`: concise chronological human log
- `storyboard.json` and `storyboard.md`: objectives, scene descriptions, narration, beats, and sources
- `costs.json`: reservations and actual usage by call, stage, provider, and model
- `report.html`: scene outcomes, failures, findings, timing, cost, and artifact links
- `prompts/`: redacted model-visible requests and responses
- `scenes/{scene_id}/`: narration, audio, local scaffold, `model_raw.py`, instrumented original/repaired/final code, traces, screenshots, diagnostics, render logs, and videos

Credentials, authorization headers, environment secrets, generated media, databases, prompts, and run artifacts are gitignored and redacted before persistence.

## Cost and Failure Policy

The default hard budget is `$1.00`. Every paid call atomically reserves its worst-case configured token cost before execution. Actual usage replaces the reservation afterward. An ambiguous provider timeout remains conservatively charged and is not retried blindly.

Schema failures receive at most one semantic retry. Confirmed rate-limit or connection failures receive at most two bounded retries; ambiguous timeouts are never retried automatically. Vision repair allows up to three sequential attempts per scene and has no separate run-wide scene cap; the atomic dollar budget remains the global limit. A scene stops early when it passes, returns unchanged code or visuals, fails to improve deterministic diagnostics, or exhausts the budget. Cache keys include provider, model, prompt version, schema, settings, and inputs, so resume does not repeat completed paid calls.

`standard` substitutes a clearly marked source-page video only when a scene cannot render safely. A successfully rendered Manim scene is retained after bounded repair even when visual findings remain, and is labeled `ACCEPTED_WITH_FINDINGS` rather than silently presented as clean. `strict` keeps visual errors blocking and withholds `final.mp4` if required content is missing. Budget exhaustion produces `PAUSED_BUDGET` and is resumable.

`--verifier-mode enforce` is the default. Errors and repeated text-overlap warnings can trigger bounded repair, while unrelated advisory warnings do not spend repair calls. `--verifier-mode audit` still runs static checks, preview rendering, trace diagnostics, screenshots, and final-render diagnostics, but visual findings are observational: a safe scene that renders continues and receives an `AUDIT_ACCEPTED` status. In audit mode, `--repair eligible` only repairs scripts that fail to render; it does not repair overlap or layout findings, preserving those findings for evaluation. Static safety and actual render failures remain blocking in every mode.

## Presenter Extension

The timeline model already supports `manim`, `presenter`, `mixed`, and `transition` segments through one `SegmentRenderer` protocol. V2 currently enables only Manim. A future avatar provider can consume the same narration audio for lip synchronization, reserve presenter-safe layout regions for mixed scenes, and return a clip to the existing compositor without changing lesson planning, verification, or run-state infrastructure.

## Scene Evaluation

The evaluation pipeline freezes scene inputs from completed V2 runs, then compares code models
without rerunning extraction, lesson planning, narration, or TTS. Generator evaluation disables
repair and deterministic source fallback, so a failed model output remains a measured failure.

Create a benchmark by adding scenes from one or more existing runs:

```powershell
pdf2manim eval build output\DOCUMENT\RUN_ID --benchmark evals\benchmarks\scene-v1 `
  --scene scene_01 --scene scene_03 --category geometry
pdf2manim eval validate evals\benchmarks\scene-v1
pdf2manim eval plan evals\benchmarks\scene-v1 --model gemini --track generator
```

Run one or more models against identical inputs. `gemini` and `gemma` are built-in aliases;
custom models use `alias=provider:model-name`.

```powershell
pdf2manim eval run evals\benchmarks\scene-v1 --track generator `
  --model gemini --model gemma --quality low --workers 2 --max-cost 2.00
```

`generator` performs exactly one generation and no repair. `repair` starts from the frozen
original code and diagnostics. `production` generates a candidate and permits one repair only
when deterministic diagnostics justify it. A production repair replaces the original only when
it renders safely and lowers the issue score.

Evaluation runs are resumable and stored under the ignored `evals/runs/` directory:

```powershell
pdf2manim eval resume evals\runs\EVALUATION_ID
pdf2manim eval status evals\runs\EVALUATION_ID
pdf2manim eval report evals\runs\EVALUATION_ID
pdf2manim eval review evals\runs\EVALUATION_ID
```

The last command starts a local dashboard at `http://127.0.0.1:8765`. It provides summary and
artifact tables, anonymized candidate videos, representative frames, diagnostics, code, prompts,
responses, and five-axis human ratings. Ratings are appended to `review/ratings.jsonl`. Use
`--retry-interrupted` only when intentionally repeating a candidate that may have reached a
provider before interruption.

## Testing

```powershell
# Entire hermetic suite, including one real local Manim/FFmpeg A/V smoke
pytest

# V2 fast tests without the renderer smoke
pytest tests\test_v2_models.py tests\test_v2_state_and_observability.py `
       tests\test_v2_providers.py tests\test_v2_static_and_diagnostics.py `
       tests\test_v2_pipeline.py -m "not integration"

# Lint the active V2 surface
ruff check src\pipeline_v2 src\cli.py tests\test_v2_*.py tests\test_cli.py
```

Production execution should still use a networkless, non-root container with CPU/memory/time limits and only the run directory writable. The AST gate is deliberately conservative but is not a complete operating-system sandbox.
