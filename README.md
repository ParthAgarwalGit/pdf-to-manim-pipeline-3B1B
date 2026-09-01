# PDF-to-Video Pipeline V2

V2 turns an educational PDF into a source-grounded, narrated Manim video. It plans the lesson around narration, defines machine-readable visual artifacts and lifecycles, generates expressive raw Manim with Gemini, verifies low-quality previews deterministically, optionally performs one targeted multimodal repair, and only then renders the final video.

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
  -> TTS timing + visual artifact contracts in parallel
  -> per-scene raw Manim generation
  -> AST safety gate + tracked 480p15 preview
  -> deterministic lifecycle/layout diagnostics
  -> optional screenshot-guided repair
  -> accepted high-quality scene renders
  -> audio composition, ordered concatenation, and final A/V review
```

V2 does not use SceneIR or MCP internally. Generated classes inherit a thin `TrackedScene` helper while otherwise using ordinary Manim.

## Visual Contracts

The visual planner separates four kinds of artifacts before arranging them:

- `primary_visual`: graphs, geometry, diagrams, algorithm state, and data demonstrations
- `semantic_text`: titles, labels, equations, and short callouts
- `annotation`: target-linked arrows, braces, and highlights
- `transient_effect`: temporary emphasis that must not remain on screen

Every artifact has a stable ID, teaching purpose, source facts, narration beats, enter/exit beat, region, layer, priority, carry policy, and intentional-overlap policy. There is no arbitrary seven-object rejection. Runtime validation uses actual active objects, bounding boxes, text occupancy, and weighted visual density.

Each scene now chooses one bounded layout template before code generation:

- `full_visual`: one large central diagram, with optional header and footer
- `split`: independent left and right content regions
- `three_column`: three compact peer regions
- `center_with_notes`: one wide central visual with narrow side-note regions

The templates use fixed Manim-coordinate boxes with gaps between regions. An artifact also declares its footprint, preferred aspect ratio, padding, minimum acceptable scale, overflow policy, and overlap policy. Only one independent artifact may own a region during a beat; related labels and geometry should be composed inside one registered `VGroup`, while target-linked annotations may share the target region explicitly. `TrackedScene.place_in_region()` scales and anchors the completed artifact into its box. Content that cannot fit above its minimum scale must be simplified or spread across beats instead of becoming unreadably small.

## Verification

Generated Python is never executed before AST validation. The gate checks imports, dangerous calls, one tracked scene, required artifact registration, beat checkpoints, script size, suspicious coordinates, and unsupported LaTeX.

During preview rendering, `TrackedScene` records stable state after `play()`, `wait()`, `add()`, and `remove()`, plus 25/50/75 percent samples inside each animation. Registered `VGroup` containers retain one logical artifact ID for lifecycle validation while their visible children receive deterministic leaf IDs for layout checks. The verifier detects:

- early entry and late exit
- missing and unplanned objects
- off-screen objects and clipped text
- preferred-region and hard-region overflow
- text-text and text-geometry overlap
- unreadably small text
- excessive active visual density
- empty or static intervals
- raw scene coordinates when axes exist
- narration/video duration mismatch

Overlap detection first measures the intersection as a fraction of the smaller leaf object's bounds. Text-related candidates are then rendered separately into small alpha masks, and occupied-pixel intersection confirms whether visible marks actually collide. Text-text collisions become warnings or errors according to severity; geometry-geometry intersections remain allowed, and intentional contract overlaps are excluded. Tiny-text warnings use rendered relative height rather than nominal Manim `font_size`. The HTML report shows the timestamp, logical artifacts, exact leaf IDs, measured ratios, and evidence frame for every finding.

Region validation has two levels. Crossing the padded preferred box creates a warning; crossing the slightly buffered hard box creates an error eligible for targeted vision repair. Because verification uses traced leaf bounds, putting several elements in a `VGroup` does not conceal a bad internal layout.

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
OPENAI_API_KEY=...     # artifact planning and TTS
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

The default worker limits are four global API calls, two visual-contract batches, four TTS jobs, three code jobs, two previews, one vision repair, one final Manim render, and one FFmpeg composition. Scene jobs advance as soon as their own audio and contract are ready.

## Resume and Reports

Press `Ctrl+C` once to stop. V2 checkpoints running work, terminates active Manim children, and keeps every successful task and paid response.

```powershell
pdf2manim resume output\document\RUN_ID
pdf2manim resume output\document\RUN_ID --max-cost 1.50
pdf2manim status output\document\RUN_ID
pdf2manim report output\document\RUN_ID
```

Each ignored `output/{document}/{run_id}/` directory contains:

- `run.json`: configuration, Git commit, environment, final status, and outputs
- `run.db`: SQLite WAL task, cache, attempt, and budget state
- `events.jsonl`: complete structured stage/scene/worker/provider timeline
- `pipeline.log`: concise chronological human log
- `storyboard.json` and `storyboard.md`: objectives, scene descriptions, narration, beats, and sources
- `costs.json`: reservations and actual usage by call, stage, provider, and model
- `report.html`: scene outcomes, failures, findings, timing, cost, and artifact links
- `prompts/`: redacted model-visible requests and responses
- `scenes/{scene_id}/`: narration, audio, contract, original/repaired/final code, traces, screenshots, diagnostics, render logs, and videos

Credentials, authorization headers, environment secrets, generated media, databases, prompts, and run artifacts are gitignored and redacted before persistence.

## Cost and Failure Policy

The default hard budget is `$1.00`. Every paid call atomically reserves its worst-case configured token cost before execution. Actual usage replaces the reservation afterward. An ambiguous provider timeout remains conservatively charged and is not retried blindly.

Schema failures receive at most one semantic retry. Confirmed rate-limit or connection failures receive at most two bounded retries; ambiguous timeouts are never retried automatically. Vision repair is limited to one attempt per scene and three per run. Cache keys include provider, model, prompt version, schema, settings, and inputs, so resume does not repeat completed paid calls.

`standard` may substitute a clearly marked source-page video when a scene cannot render. `strict` completes independent diagnostics but withholds `final.mp4` if required content is missing. Budget exhaustion produces `PAUSED_BUDGET` and is resumable.

`--verifier-mode enforce` is the default and treats deterministic visual errors as scene failures. `--verifier-mode audit` still runs static checks, preview rendering, trace diagnostics, screenshots, and final-render diagnostics, but visual findings are observational: a safe scene that renders continues and receives an `AUDIT_ACCEPTED` status. In audit mode, `--repair eligible` only repairs scripts that fail to render; it does not repair overlap or layout findings, preserving those findings for evaluation. Static safety and actual render failures remain blocking.

## Presenter Extension

The timeline model already supports `manim`, `presenter`, `mixed`, and `transition` segments through one `SegmentRenderer` protocol. V2 currently enables only Manim. A future avatar provider can consume the same narration audio for lip synchronization, reserve presenter-safe layout regions for mixed scenes, and return a clip to the existing compositor without changing lesson planning, verification, or run-state infrastructure.

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
