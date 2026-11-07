# PDF to Manim Animation Pipeline

An agent-orchestrated system that transforms academic PDF documents into short, 3Blue1Brown-style animated **visualizations** using the Manim animation engine.

## Overview

This is a **visualization-first** tool, not an end-to-end learning system. It reads study material, deeply segments it, and produces a series of short vignette videos — each one visually depicting a single concept with motion, transformation, and progressive reveal. The design goals, in order:

1. **Impactful visuals** — every scene must *draw* the concept, never narrate over a text wall.
2. **Breadth coverage** — every part of the source document is covered exactly once; nothing is compressed away or repeated across videos.
3. **Short vignettes** — scenes run ~8–45 seconds, one focused visual idea each. A 20-page paper becomes many small videos, not one long lecture.

## Architecture

A 10-stage sequential pipeline with embedded quality loops:

1. **Ingestion Agent** — detects file uploads and routes them
2. **Validation Agent** — verifies PDF type and structural integrity
3. **Vision + OCR Agent** — extracts text, mathematical notation, and diagrams from every page
4. **Script/Storyboard Architect** — the visual brain:
   - **Document Segmenter** partitions the source into ordered, non-overlapping segments (~half a page each, tunable via `--granularity`) with guaranteed full coverage
   - one narration beat script is generated **per segment**, with a concept ledger enforcing "introduce once, reference later"
   - a deterministic chunker groups beats into vignette-sized scenes that never cross segment boundaries
   - a per-scene visual planner emits a structured **visual plan**: closed-catalog Manim primitives with kwargs, an `(x, y)` position, a `(t_in, t_out)` visibility window, and a beat-linked instructional purpose
   - scenes may use up to 15 objects over time, but no more than seven concurrently; overloaded multi-beat scenes are split instead of silently truncated
   - the **visual-archetype library** classifies each scene's concept family (derivative, vector, limit, probability, …) and injects a proven visual pattern into the planner prompt
   - a **non-text-visual invariant** retries any plan that came back as a text wall
   - *quality loop:* the **Storyboard Grader** scores coverage and pedagogy; failing storyboards regenerate with critique
5. **JSON Sanitizer** — enforces schema compliance on the storyboard (abort on failure)
6. **Scene Controller** — iterates scenes deterministically, carrying narrative/color/notation continuity into code generation; the **Animation Code Generator** turns each visual plan into Manim code that honors the plan
   - *quality loop:* the **Manim Code Validator** checks syntax and imports, while the optional **Raw-Manim Vision Verifier** performs a stricter AST safety gate, low-quality preview render, four-frame visual review, and controlled full-script repair
7. **Persistence Agent** — writes scene `.py` files, storyboard, and narration to disk
8. **Video Renderer Agent** — renders each scene file with Manim → `.mp4`
9. **TTS Agent** — generates narration audio; `--tts-first` moves this before code generation and makes measured audio duration authoritative
10. **Audio-Video Composer Agent** — merges per-scene video + audio and concatenates the final video
    - *quality loop:* the **Video Quality Reviewer** inspects results — including **frame-level visual QA** that samples rendered keyframes via FFmpeg and flags empty frames and static "still-slide" scenes — and routes findings back as regeneration guidance

Each agent has a single responsibility, explicit input/output contracts, and isolated failure modes: a failing scene is logged and skipped, never fatal to the run.

## Key Features

- **Full-coverage segmentation** — the storyboard provably covers every source segment exactly once; gaps and redundant coverage are detected and reported
- **Visual-archetype library** — 11 curated concept→visual patterns (tangent-sweep for derivatives, tip-to-tail for vectors, grid-morph for matrices, …) steer every scene toward a proven design
- **Closed primitive catalog** — the LLM can only emit schema-validated Manim primitives; hallucinated classes are rejected at the boundary, not at render time
- **Narration-timed pilot** — `--tts-first` generates speech before code and rescales visual beats to measured audio duration
- **Purpose-aware visuals** — every planned and generated display object must state which narration beat it serves
- **Security gate** — generated scene code is AST-screened (import allowlist, forbidden calls, reflection escapes) before local execution, defending against prompt injection from malicious PDFs
- **Raw-Manim vision verification** — low-quality previews, representative screenshots, deterministic frame diagnostics, and at most two controlled full-script repairs
- **Frame-level visual QA** — rendered pixels are judged directly: blank or static scenes are flagged with concrete rewrite directions
- **Failure isolation & determinism** — per-scene error recovery, stable IDs, reproducible structure for the same input
- **Cost tracking** — per-run LLM/TTS spend is recorded in the manifest

## Installation

### Prerequisites

- Python 3.10+
- LaTeX (for mathematical notation rendering)
- FFmpeg (for video rendering and frame-level QA)

#### Install LaTeX

**Ubuntu/Debian:**

```bash
sudo apt-get install texlive-full
```

**macOS:**

```bash
brew install --cask mactex
```

**Windows:**
Install MiKTeX or TeX Live from their respective websites.

#### Install FFmpeg

**Ubuntu/Debian:**

```bash
sudo apt-get install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org)

### Python Dependencies

```bash
pip install -e ".[dev]"
```

If you only want runtime dependencies, `pip install -r requirements.txt` still works.

### API Keys

Copy `keys.example.env` to `keys.env` (or use `.env`) and set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`. Gemini 3.7 Flash generates the final Manim Python; without its key that stage degrades to deterministic mock output.

## Usage

### Process a Single PDF

```bash
pdf2manim convert path/to/document.pdf --output-dir ./output
```

### Process with Custom Configuration

```bash
pdf2manim convert document.pdf \
    --verbosity high \
    --depth advanced \
    --audience graduate \
    --granularity fine \
    --voice nova
```

### Execution Modes

```bash
# Planning only — segment, script, and storyboard, no code/render/TTS
pdf2manim convert document.pdf --dry-run

# Skip TTS (render video without narration audio)
pdf2manim convert document.pdf --skip-tts

# Skip rendering entirely (generate scene code only)
pdf2manim convert document.pdf --skip-render

# Set render quality: low / medium / high
pdf2manim convert document.pdf --quality high

# Fail fast when API keys are missing instead of using mock fallbacks
pdf2manim convert document.pdf --strict

# Pilot: generate narration first and animate against measured audio duration
pdf2manim convert document.pdf --tts-first

# Observe preview-verifier findings without changing generated scripts
pdf2manim convert document.pdf --verifier-mode shadow

# Allow one screenshot-guided Gemini repair per failing scene
# The verifier reads GEMINI_API_KEY from .env and never writes it to artifacts.
pdf2manim convert document.pdf --verifier-mode repair --verifier-max-repairs 1
```

`VERIFIER_MODE=off|shadow|repair` can also configure the mode. When the verifier
is enabled, strict runs fail if a preview cannot render. Strict delivery also
fails when a scene is lost or the final review contains a high/critical finding.
Preview artifacts are retained under the run's ignored `verifier_runs/`
directory.

### Verify a Raw Manim Script

```bash
# Offline safety check, preview render, four screenshots, and report
manim-verifier scene.py --mode shadow

# Optional paid multimodal repair; reads only the specified .env file
manim-verifier scene.py --mode repair --env-file .env --max-repairs 1
```

The AST verifier is a safety gate, not a complete sandbox. Production execution
should use a networkless container, non-root user, CPU/memory/time limits, and
only one writable output directory.

### Estimate Cost Before Running

```bash
pdf2manim cost document.pdf
```

### Delivery Check

Before a video is handed over, every scene is measured: animation vs.
narration length (no frozen last frame), narration not cut off, no long
silences, loudness at −16 LUFS, and uniform clip formats with no A/V drift
across the joined video. Failures are repaired automatically where a
deterministic fix exists (re-time the scene's pauses and re-render,
recompose, re-normalize, re-encode the join), at most twice per scene.
The verdict is in `delivery_report.json` and printed at the end of the run:

```
Ready: 32/32 scenes passed sync and quality checks.
```

or, when something still needs a human:

```
Needs review: 31/32 scenes passed sync and quality checks.
Scene 14 (0:42-0:51): The animation ends 2.3s before the narration, so the last frame is frozen.
```

The video is delivered either way. Disable with `PipelineConfig(enable_delivery_gate=False)`.

With `--tts-first`, narration is recorded beat by beat before any code is
written, giving a timing track (`timing_track.json`). The generated scene
calls `sync_to(self, "beat_007")` before each beat's animation and waits on
Manim's own clock, so visuals line up with the words at render time. The
gate then checks that something moves during every beat; a still beat is
regenerated with that finding as feedback (at most 3 LLM regenerations per
run). `PipelineConfig(enable_asr_check=True)` adds a paid Whisper check that
the spoken audio matches the script.

### Steer It With a Brief

Give your own instructions alongside the PDF:

```bash
pdf2manim convert notes.pdf --brief brief.md
```

> My students are 2nd-year EE and struggle with phasors. Use the rotating-arm
> picture, not the clock analogy. Skip 4.3. Spend more time on resonance.
> Under 10 minutes. I write j, not i.

The brief is read back ("Here's how I understood your brief") before anything
is spent. The instructions are binding. Section 4.3 is never scripted. A beat
that slips in the clock analogy is rewritten. Generated LaTeX is checked for
`i`, and the length is checked against your target. Anything still unmet is
listed in the delivery report.

### Plan Concept Depth First

```bash
pdf2manim plan notes.pdf -o plan.yaml          # one entry per concept per section
# edit: depth skip | brief | standard | deep; analogy auto | none | professor
pdf2manim convert notes.pdf --plan plan.yaml
```

Each section's script follows its plan entries. Skipped concepts are left
out, enforced, and listed as your omissions. After a video exists,
`pdf2manim revise RUN --density essential --analogies off` trims optional
depth and analogies without regenerating, and `--density rich` goes deeper on
every concept.

### Presenter Bridges

```bash
pdf2manim convert notes.pdf --tts-first --bridges propose            # propose, render nothing yet
pdf2manim revise output/notes/<run> --approve-bridges                # approve all proposals
pdf2manim convert notes.pdf --tts-first --bridges propose --approve-bridges \
  --avatar heygen --tts-provider elevenlabs --profile professor.json --presenter-layout pip
```

Short on-camera moments (an opening hook, transitions between sections, a
"why this matters" before a hard step, a recap) are proposed at most every
90 s and never exceed 20% of the runtime. A bridge is narrated and rendered
only after you approve its text. Without an avatar, the narration plays over
a title card. With `--avatar heygen` and/or `--tts-provider elevenlabs`, your
own likeness or voice is used. That requires a profile with your signed
consent and the vendor's identity verification. Keys come from
`HEYGEN_API_KEY` / `ELEVENLABS_API_KEY`. Every synthetic clip is labelled
"AI-generated presenter" and credited at the end.

### Feedback in Plain Words

```bash
pdf2manim revise output/notes/<run> --say "At 2:13 the label overlaps the curve. 'Euler' is pronounced Oiler. Drop the pizza analogy."
```

Each piece of feedback becomes an edit anchored to the moment you named. A
visual fix regenerates that scene's code but keeps its audio. A pronunciation
fix re-records only the scenes that say the word. A content change rewrites
the affected beats. You see the diff and an estimate before confirming
(`--yes` skips the prompt). The same edits can be scripted with `--ops
edits.yaml`, and `--density essential --analogies off` trims a video
without regenerating it.

### Revise a Finished Video

Every run saves `lesson_script.json`: the narration as beats, each tagged with
a role (`core`, `analogy`, `example`, ...), a depth, and the analogy it belongs
to. Edit the beats (reword, add, delete), then rebuild:

```bash
pdf2manim revise output/document/2026-11-07T10-00-00
pdf2manim revise output/document/2026-11-07T10-00-00 --script my_edits.json --dry-run
```

Only scenes whose beats changed are re-planned, regenerated, rendered and
narrated; the rest reuse the run's code, renders and audio (matched by
fingerprints stored in `manifest.json`). The result is a new run in
`<run>/rev_NNN/` with a `revision.json` describing what changed and what it
cost. The original run is never modified. `--dry-run` writes the plan without
spending anything; `--voice` or `--quality` re-narrates or re-renders.

## Docker & AMD Developer Cloud

The pipeline is containerized so the full workload runs on an AMD Developer
Cloud instance. The base image ships LaTeX, Cairo, and FFmpeg, so the build is
self-contained.

```bash
# 1. Provide keys (git-ignored, never baked into the image)
cp keys.example.env keys.env      # then add your OpenAI, Anthropic, and Gemini keys

# 2. Build + prove the container works WITHOUT spending credits
scripts/run_on_amd.sh smoke       # keyless --help and --dry-run

# 3. Convert a PDF
scripts/run_on_amd.sh convert path/to/paper.pdf -q high
```

Or with Compose:

```bash
docker compose run --rm convert   # ./input/document.pdf -> ./output
docker compose up demo            # web demo on http://<host>:7860
```

**What runs on AMD:** rendering and encoding — the bulk of the compute — run on
AMD Developer Cloud (EPYC CPUs, optional AMD GPU). Frontier models (gpt-5,
claude) are used only for reasoning/codegen over the API, so output quality is
unchanged. Full guide, evidence-capture tips, and the optional ROCm GPU
render/encode path: [docs/amd-deployment.md](docs/amd-deployment.md).

## Web Demo

A minimal Gradio app (`app.py`) lets you upload a PDF and watch the generated
vignettes in the browser — this is the hosted "Application URL":

```bash
pip install -e ".[demo]"
python app.py                     # http://localhost:7860

# Public URL for a submission/judging (compute stays on the host):
GRADIO_SHARE=1 python app.py      # prints https://<id>.gradio.live (~72h)
```

Note: the demo must run on a host with LaTeX/FFmpeg/Manim and minute-long
render budgets (i.e. the container on AMD Developer Cloud) — it is **not**
deployable to serverless platforms like Vercel. To use a serverless frontend,
put only a static page there and point it at this app running on AMD.

## Configuration

| Option | Values | Controls |
|---|---|---|
| `--verbosity` | `low` \| `medium` \| `high` | Narration detail level |
| `--depth` | `introductory` \| `intermediate` \| `advanced` | Content depth |
| `--audience` | `high-school` \| `undergraduate` \| `graduate` | Terminology level |
| `--granularity` | `coarse` \| `balanced` \| `fine` | Source segment size — `fine` yields more, shorter videos covering the same material |
| `--quality` | `low` \| `medium` \| `high` | Manim render quality |
| `--voice` | `alloy` \| `echo` \| `fable` \| `onyx` \| `nova` \| `shimmer` | TTS voice |

## Testing

The default lane is **hermetic**: provider API keys are stripped for every test not marked `live_api`, so it is fast, free, and deterministic.

```bash
# Fast lane (~20s, no network, no keys needed)
python -m pytest

# With coverage
python -m pytest --cov=src --cov-report=html

# Property-based lane (Hypothesis)
scripts/run_property_tests.sh

# Real-API smoke tests (requires keys; excluded from the default lane)
python -m pytest -m live_api

# Specific test file
python -m pytest tests/test_script_architect.py
```

## Project Structure

```
.
├── src/
│   ├── agents/          # Pipeline agents (segmenter, architect, codegen, validators, reviewers)
│   ├── schemas/         # Pydantic data contracts (storyboard, segments, video script, ...)
│   ├── orchestrator/    # 10-stage pipeline orchestration + retry policies
│   └── utils/           # Primitive catalog, visual archetypes, frame analysis, Manim helpers
├── tests/               # Hermetic test suite (+ property and live_api lanes)
├── docs/
│   └── adr/             # Architecture Decision Records
├── scripts/             # Dev scripts (property-test lane)
├── output/              # Generated artifacts (runtime)
├── pyproject.toml       # Packaging, CLI entry point, pytest and lint configuration
└── requirements.txt     # Runtime dependency fallback
```

## Output Structure

```
output/{pdf_filename}/{timestamp}/
├── scene_*.py              # Generated Manim code (one vignette per file)
├── narration_script.txt    # Narration with scene markers
├── storyboard.json         # Scene specifications with visual plans
├── lesson_script.json      # Editable narration beats + analogies (input to `revise`)
├── processing_report.json  # Success/failure summary
├── storyboard_review.json  # Storyboard quality review (coverage, pedagogy)
├── scene_code_review.json  # Code validation review (incl. security gate)
├── video_review.json       # Final media review (incl. frame-level visual QA)
├── delivery_report.json    # Delivery gate verdict: ready / needs review, per-scene checks
├── tts.json                # TTS batch outcome
├── manifest.json           # File listing, cost summary, per-scene fingerprints
├── audio/                  # Per-scene narration .mp3
├── pipeline.log            # Structured execution logs
└── rev_002/                # A revision (same layout + revision.json)
```

## Architecture Decision Records

| ADR | Decision |
|---|---|
| [001](docs/adr/001-sync-vs-async-orchestrator.md) | Synchronous orchestrator with targeted parallel fan-out |
| [002](docs/adr/002-unified-script-chunked-render.md) | One unified script, deterministically chunked into scenes |
| [003](docs/adr/003-closed-primitive-catalog.md) | Closed, schema-validated Manim primitive catalog |
| [004](docs/adr/004-xyt-layout-validation.md) | (Superseded) standalone (x,y,t) layout validation |
| [005](docs/adr/005-visual-first-render-path.md) | Visual-first render path: plan-informed codegen, vignette pacing |

## Development

### Agent Interface

All agents implement a common interface:

```python
class Agent(ABC):
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """Execute agent logic"""

    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input conforms to expected schema"""

    def get_retry_policy(self) -> RetryPolicy:
        """Return retry policy for this agent"""
```

### Extending the Visual Vocabulary

- **New Manim primitive** — add one entry to `MOBJECT_CATALOG` in `src/utils/manim_primitives.py`; schema validation, prompts, and the code emitter pick it up automatically. Then allow any new import roots in the validator's security allowlist if needed.
- **New visual archetype** — append a `VisualArchetype` to `ARCHETYPE_REGISTRY` in `src/utils/visual_archetypes.py` (keywords, catalog-only primitives, motion design, fallback primitive). Registry-integrity tests enforce that it can never recommend anything the schema rejects.
- **New agent** — create the class in `src/agents/`, define its schemas in `src/schemas/`, add tests, and wire it into the orchestrator sequence.

## Documentation

- [Output Contract](docs/output-contract.md) — manifest schema, file types, and inspection commands
- [Troubleshooting](docs/troubleshooting.md) — common failures and fixes
- [Determinism](docs/determinism.md) — reproducibility guarantees

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Contributors

- **Parth Agarwal**
- **Sumit Galgalkar**
- **Divyansh Bisht**
- **Vignesh KS**
