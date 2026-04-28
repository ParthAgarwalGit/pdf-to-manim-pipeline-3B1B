# PDF to Manim Professionalization Implementation Plan

> **Status:** 🟡 LARGELY SHIPPED, NOT FULLY AUDITED — most goals are in tree:
> single canonical CLI (`src/cli.py`), one canonical output directory per
> run, manifest as the canonical contract (see `docs/output-contract.md`),
> bounded quality-review loops, GitHub Actions CI, ADR-001
> (sync vs async orchestrator), `docs/troubleshooting.md`, structured
> `pipeline.log`. The 37 individual checkboxes have not been verified
> task-by-task. Treat this plan as historical scaffolding rather than
> an active backlog. Promote to `completed/` after a per-task audit.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current local PDF-to-video research pipeline into a reliable, documented, testable professional CLI tool that consistently produces inspectable outputs and can be run from a clean checkout on Windows and Linux.

**Architecture:** Keep the current agent-based local pipeline, but make the external contract truthful end-to-end: one public CLI, one canonical output directory per run, one final manifest, bounded quality-review loops, and one CI path that validates what users actually run. Prioritize reliability, output quality, and operator trust over enterprise abstractions that do not improve the resulting videos.

**Tech Stack:** Python 3.10+, Click CLI, Pydantic v2, Manim, FFmpeg, OpenAI/Anthropic integrations, pytest, Ruff, GitHub Actions

---

### Task 1: Freeze the Project Surface and Clean the Repository Boundary

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `SETUP.md`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Review and intentionally stage: `src/cli.py`
- Review and intentionally stage: `src/agents/audio_video_composer.py`
- Review and intentionally stage: `src/agents/manim_code_validator.py`
- Review and intentionally stage: `src/agents/storyboard_grader.py`
- Review and intentionally stage: `src/agents/tts_agent.py`
- Review and intentionally stage: `src/agents/video_quality_reviewer.py`
- Review and intentionally stage: `src/agents/video_renderer.py`
- Review and intentionally stage: `src/orchestrator/async_pipeline.py`
- Review and intentionally stage: `src/utils/cost_tracker.py`
- Review and intentionally stage: `src/utils/ffmpeg_finder.py`
- Review and intentionally stage: `src/utils/layout_tracker.py`
- Review and intentionally stage: `src/utils/pi_characters.py`
- Review and intentionally stage: `tests/test_audio_video_composer.py`
- Review and intentionally stage: `tests/test_strict_mode.py`
- Review and intentionally stage: `tests/test_tts_agent.py`

- [ ] **Step 1: Remove runtime artifacts and scratch files from the product surface**

Move or ignore local-only artifacts so the repo stops mixing source code with run outputs. Keep the root focused on source, config, and docs.

Run: `git status --short`
Expected: runtime directories such as `media/`, `output/`, `test_output/`, `htmlcov/`, and cache files show up separately from real source files.

- [ ] **Step 2: Tighten ignore rules for generated outputs**

Make sure `.gitignore` covers runtime outputs and local heavy artifacts that should never be treated as product source.

```gitignore
media/
output/
test_output/
htmlcov/
__pycache__/
.pytest_cache/
*.mp3
*.wav
*.jsonl
```

- [ ] **Step 3: Decide and document the canonical developer workflow**

Keep `pyproject.toml` as the canonical packaging and test config, keep `pdf2manim` as the public command, and document `requirements.txt` as fallback-only if you keep it.

Run: `python -m pytest --help`
Expected: `configfile: pyproject.toml`

- [ ] **Step 4: Stage the real product files intentionally**

Do not leave core product files untracked. Make a deliberate call on each currently untracked source file under `src/`, `tests/`, and `.github/`.

Run: `git status --short`
Expected: only intentionally ignored runtime files remain untracked.

- [ ] **Step 5: Commit the repository surface cleanup**

```bash
git add .gitignore README.md SETUP.md pyproject.toml .github/workflows/ci.yml src tests
git commit -m "chore: stabilize repository surface and developer workflow"
```

### Task 2: Make the CLI Contract Truthful

**Files:**
- Modify: `src/cli.py`
- Modify: `src/orchestrator/pipeline.py`
- Modify: `src/schemas/output_manifest.py`
- Modify: `tests/test_orchestrator.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI flag propagation**

Add tests that prove `--skip-tts`, `--skip-render`, and `--dry-run` change orchestrator behavior instead of only changing printed text.

```python
def test_convert_passes_skip_tts_into_pipeline_config():
    ...

def test_convert_passes_dry_run_into_pipeline_config():
    ...
```

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL because `PipelineConfig` does not yet capture all execution flags.

- [ ] **Step 2: Add execution flags to the runtime config**

Extend `PipelineConfig` with the execution controls the CLI already advertises.

```python
enable_tts: bool = True
enable_render: bool = True
dry_run: bool = False
tts_voice: str = "alloy"
render_quality: str = "medium"
```

- [ ] **Step 3: Wire CLI options into `PipelineConfig`**

Update `convert()` in `src/cli.py` so the CLI builds `PipelineConfig` with the actual requested behavior rather than discarding those values.

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 4: Make the orchestrator honor the flags**

Implement the actual stage skips:
- `dry_run`: stop after sanitized storyboard and persistence of planning artifacts
- `enable_render=False`: skip render and composition
- `enable_tts=False`: skip TTS and do not try to compose narrated output

Run: `python -m pytest tests/test_orchestrator.py -k "dry_run or skip_tts or skip_render" -v`
Expected: PASS

- [ ] **Step 5: Record effective execution settings in the manifest**

Extend `OutputManifest` so it records what the run really did, not just the output files.

```python
class RunConfig(BaseModel):
    dry_run: bool
    enable_render: bool
    enable_tts: bool
    render_quality: str
    tts_voice: str
```

- [ ] **Step 6: Commit the truthful CLI/runtime contract**

```bash
git add src/cli.py src/orchestrator/pipeline.py src/schemas/output_manifest.py tests/test_cli.py tests/test_orchestrator.py
git commit -m "feat: make CLI flags match runtime behavior"
```

### Task 3: Make the Output Contract Final and Inspectable

**Files:**
- Modify: `src/orchestrator/pipeline.py`
- Modify: `src/agents/persistence.py`
- Modify: `src/schemas/output_manifest.py`
- Modify: `tests/test_orchestrator.py`
- Create: `tests/integration/test_manifest_contract.py`

- [ ] **Step 1: Write a failing manifest contract test**

Add an integration-style test that asserts the final manifest describes the actual end-of-run artifacts.

```python
def test_final_manifest_contains_final_video_and_review_reports():
    ...
```

Run: `python -m pytest tests/integration/test_manifest_contract.py -v`
Expected: FAIL until final media outputs and reports are guaranteed in `manifest.json`.

- [ ] **Step 2: Treat `manifest.json` as the canonical run artifact**

Ensure the final manifest includes:
- canonical output directory
- runtime config
- scene code files
- scene video files
- scene audio files
- composed videos
- final video path
- quality review artifacts

- [ ] **Step 3: Stop relying on CLI globbing for success reporting**

Update CLI summary logic to read final output paths from the manifest first, using filesystem scanning only as a temporary fallback.

Run: `python -m pytest tests/integration/test_manifest_contract.py tests/test_orchestrator.py -k manifest -v`
Expected: PASS

- [ ] **Step 4: Commit the final manifest contract**

```bash
git add src/orchestrator/pipeline.py src/agents/persistence.py src/schemas/output_manifest.py tests/integration/test_manifest_contract.py tests/test_orchestrator.py
git commit -m "feat: finalize manifest as canonical output contract"
```

### Task 4: Turn the Quality Review Loop into a Real Quality Gate

**Files:**
- Modify: `src/agents/storyboard_grader.py`
- Modify: `src/agents/animation_generator.py`
- Modify: `src/agents/scene_controller.py`
- Modify: `src/agents/video_quality_reviewer.py`
- Modify: `src/agents/audio_video_composer.py`
- Modify: `src/utils/layout_tracker.py`
- Modify: `src/orchestrator/pipeline.py`
- Modify: `tests/test_orchestrator.py`
- Create: `tests/test_video_quality_reviewer.py`

- [ ] **Step 1: Add failing tests for the actual quality problems**

Cover the issues that matter most for this project: visual overlap, pacing drift, and audio/video duration mismatch.

```python
def test_video_reviewer_flags_audio_video_duration_mismatch():
    ...

def test_video_reviewer_blames_scene_generation_for_overlap_style_issues():
    ...
```

Run: `python -m pytest tests/test_video_quality_reviewer.py -v`
Expected: FAIL until reviewer heuristics and stage attribution are tightened.

- [ ] **Step 2: Define bounded retry policy for quality loops**

Keep the retry loop bounded and visible. Do not allow “self-healing” retries to hide instability.

```python
enable_quality_loops: bool = True
max_quality_retries: int = 2
storyboard_quality_threshold: float = 6.0
```

- [ ] **Step 3: Add concrete reviewer outputs for operator debugging**

Require the final reviewer to output:
- suspected failing stage
- scene IDs involved
- concrete diagnosis
- suggested prompt directions
- code hints when scene code appears responsible

- [ ] **Step 4: Promote review artifacts into the operator workflow**

Make the CLI print the locations of:
- `storyboard_review.json`
- `scene_code_review.json`
- `video_review.json`

Run: `python -m pytest tests/test_video_quality_reviewer.py tests/test_orchestrator.py -k QualityLoops -v`
Expected: PASS

- [ ] **Step 5: Commit the quality gate hardening**

```bash
git add src/agents/storyboard_grader.py src/agents/animation_generator.py src/agents/scene_controller.py src/agents/video_quality_reviewer.py src/agents/audio_video_composer.py src/utils/layout_tracker.py src/orchestrator/pipeline.py tests/test_video_quality_reviewer.py tests/test_orchestrator.py
git commit -m "feat: harden review loops around output quality"
```

### Task 5: Fix Cross-Platform Reliability and the Windows Logger Failure

**Files:**
- Modify: `src/orchestrator/logger.py`
- Modify: `src/utils/ffmpeg_finder.py`
- Modify: `src/agents/video_renderer.py`
- Modify: `tests/test_orchestrator.py`
- Create: `tests/test_logger.py`

- [ ] **Step 1: Write a failing regression test for the temp-directory logger bug**

Reproduce the current failure where `StructuredJSONLogger(output_directory=tmpdir)` raises `PermissionError` on Windows and leaves the temp directory undeletable.

```python
def test_structured_logger_can_write_and_close_in_temp_directory():
    ...
```

Run: `python -m pytest tests/test_logger.py tests/test_orchestrator.py -k "logger or retry" -v`
Expected: FAIL on the current code.

- [ ] **Step 2: Make logger lifecycle safe**

Guard file creation and cleanup properly:
- create parent directory safely
- open file lazily or with robust error handling
- always close on failure paths
- degrade to console logging if file logging cannot be initialized

- [ ] **Step 3: Centralize environment checks for render-time dependencies**

Use `ffmpeg_finder.py` and renderer preflight checks so failures are early and legible instead of surfacing as late-stage subprocess errors.

Run: `python -m pytest tests/test_logger.py tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 4: Commit the reliability fixes**

```bash
git add src/orchestrator/logger.py src/utils/ffmpeg_finder.py src/agents/video_renderer.py tests/test_logger.py tests/test_orchestrator.py
git commit -m "fix: stabilize logging and media preflight across platforms"
```

### Task 6: Add One Real End-to-End Gate and Ship It Professionally

**Files:**
- Create: `tests/fixtures/`
- Move or copy fixture from: `test_trig_3pages.pdf`
- Modify: `tests/integration/test_end_to_end.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `SETUP.md`
- Create: `docs/output-contract.md`
- Create: `docs/troubleshooting.md`

- [ ] **Step 1: Add a real smoke test with a tiny fixture PDF**

Use a small deterministic fixture and assert that a run produces planning artifacts, review artifacts, and a final manifest with real paths.

```python
def test_small_pdf_pipeline_smoke(tmp_path):
    ...
```

Run: `python -m pytest tests/integration/test_end_to_end.py -v`
Expected: PASS locally before enabling it in CI.

- [ ] **Step 2: Split CI into fast and slow confidence layers**

Keep unit tests on the full Python matrix, but run the slower media integration smoke test on one Python version only so CI remains affordable and reliable.

- [ ] **Step 3: Document the professional operator path**

Add docs for:
- install
- required external tools
- expected output directory layout
- common failure modes
- how to inspect quality reports

- [ ] **Step 4: Run the full verification suite before calling it professional**

Run: `python -m pytest`
Expected: PASS

Run: `python -m pytest tests/integration/test_end_to_end.py -v`
Expected: PASS

Run: `ruff check src tests`
Expected: PASS

Run: `pdf2manim --help`
Expected: CLI loads and shows the public command surface.

- [ ] **Step 5: Commit the release-grade baseline**

```bash
git add tests/integration/test_end_to_end.py tests/fixtures .github/workflows/ci.yml README.md SETUP.md docs/output-contract.md docs/troubleshooting.md
git commit -m "chore: add end-to-end gate and operator documentation"
```

## Professional Definition for This Project

This project should be considered professionally ready when all of the following are true:

- [ ] A clean clone can install dependencies and run `pdf2manim --help`
- [ ] `python -m pytest` passes without Windows-specific logger failures
- [ ] The CLI flags are truthful and reflected in the manifest
- [ ] Every run writes to one canonical output directory
- [ ] `manifest.json` names the actual produced outputs
- [ ] Quality review artifacts are persisted and easy to inspect
- [ ] At least one end-to-end PDF smoke test passes in CI
- [ ] The repo no longer mixes source files with runtime artifacts

## Scope Explicitly Deferred

These are not required to reach a professional local CLI baseline:

- Full dependency injection refactors across all agents
- Async progress granularity for a future web frontend
- Multi-user tenancy or SaaS infrastructure
- Queueing, cancellation, retention policies, and server-side rate-limit orchestration

## Self-Review

Spec coverage:
- Repository hygiene and tracked source boundary: covered by Task 1
- Packaging/docs/CI truthfulness: covered by Tasks 1 and 6
- CLI/runtime truthfulness: covered by Task 2
- Final manifest and output contract: covered by Task 3
- Output quality loops and video diagnosis: covered by Task 4
- Cross-platform reliability: covered by Task 5
- Real professional confidence gate: covered by Task 6

Placeholder scan:
- No `TODO`/`TBD` placeholders remain
- Each task names exact files and verification commands

Type consistency:
- Runtime flags use `enable_tts`, `enable_render`, `dry_run`, `tts_voice`, and `render_quality` consistently across tasks
- Manifest extension uses a single `RunConfig` concept consistently
