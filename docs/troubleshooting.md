# Troubleshooting

Common failure modes and how to resolve them.

## Installation Issues

### `ModuleNotFoundError: No module named 'src'`

You need to install the package in development mode:

```bash
pip install -e ".[dev]"
```

### `pangocairo` or `cairo` errors during `pip install manim`

Manim requires system-level C libraries. Install them first:

```bash
# Ubuntu/Debian
sudo apt-get install libcairo2-dev libpango1.0-dev ffmpeg

# macOS
brew install cairo pango ffmpeg

# Windows — use conda
conda install -c conda-forge manim
```

## Runtime Errors

### FFmpeg not found

The pipeline checks for FFmpeg before rendering. If you see:

```
FFmpeg is not installed or not on PATH.
```

Install FFmpeg and ensure it is on your PATH:

```bash
ffmpeg -version   # should print version info
```

The pipeline searches: system PATH, Manim's bundled FFmpeg, and common Windows install directories (`C:\ffmpeg\bin`, Chocolatey, Scoop).

### Logger PermissionError on Windows

Previously, the logger could crash the pipeline if the output directory was not writable. This is now fixed — the logger degrades to console-only logging. If you see a warning like:

```
Could not create log file in <dir>: [Errno 13] Permission denied
```

The pipeline will continue. Check directory permissions if you need the log file.

### `NameError: name 'openai' is not defined`

The Vision+OCR agent requires the `openai` package and a valid API key. Set:

```bash
export OPENAI_API_KEY="sk-..."
```

Use `GEMINI_API_KEY` for Manim code generation. `ANTHROPIC_API_KEY` is used by the optional unified-script provider path.

### Validation: "File is not a valid PDF"

The validation agent checks the PDF header and structure. Ensure your file is a valid PDF (not a renamed DOCX, image, etc.).

### `CRITICAL_FAILURE` status (>50% scenes failed)

Check `manifest.json` → `failed_scenes` for per-scene error messages. Common causes:

- LLM returned invalid Manim code (check scene code review)
- LaTeX expressions that Manim cannot render
- Scene dependencies on objects from previous scenes

Run with `--dry-run` first to validate the storyboard without rendering.

### Render sandbox (`--render-sandbox docker`) errors

See [amd-deployment.md](amd-deployment.md#sandboxed-execution-of-generated-code) for what the mode does.

- **"needs the docker CLI on PATH"**: install Docker and make sure the daemon is running (`docker version`).
- **`manim exit code 125 [sandbox: docker could not start the container ...]`**: usually the image isn't built locally. The sandbox never pulls (`--pull never`), so run `docker build -t pdf2manim:latest .` or pass `--sandbox-image`.
- **`manim exit code 137 [sandbox: render killed ...]`**: the scene hit the time limit or the 2 GB memory cap. Simplify the scene, or raise the limits in `RenderSandboxConfig`.
- **Permission denied writing `/work` on Linux**: the container user can't write the bind-mounted scratch dir. Run the pipeline as a regular (non-root) user so the container uses your uid:gid, or set `RenderSandboxConfig.user`.
- **A scene that reads files (`ImageMobject("...")`, `SVGMobject(...)`) fails only in docker mode**: expected. The container can only see that scene's scratch dir.

## Quality Review Issues

### Storyboard grading below threshold

The storyboard grader scores pedagogical quality. If a run retries and still scores below 6.0/10, check `storyboard_review.json` for specific feedback. You can adjust the threshold or audience/depth settings.

### Duration mismatch findings

`video_review.json` may flag scenes where audio and video durations differ by more than 2 seconds. This usually means the TTS narration is longer than the animation. Fix by shortening narration or extending scene `duration_estimate`.

### Layout overlap warnings

The reviewer flags scenes where mathematical objects may visually overlap. Check `scene_code_review.json` for `code_hint` suggestions.

## Test Failures

### Pre-existing `test_orchestrator.py` failures

Some orchestrator tests may fail with `NameError: name 'openai' is not defined` if the `openai` package is not installed. These are integration tests that require API dependencies. Unit tests should pass without API packages:

```bash
python -m pytest tests/ -k "not test_orchestrator" --no-cov
```

### `.coverage` PermissionError

If pytest-cov cannot write `.coverage`, run with:

```bash
python -m pytest --no-cov
```

## Getting More Information

- Check `pipeline.log` in the output directory for structured execution traces
- Use `--dry-run` to validate planning without rendering
- Use `pdf2manim cost <file>` to estimate API costs before a full run
