# Code Review Report — PDF-to-Manim Pipeline

**Date**: 2026-04-08  
**Scope**: Full codebase review across all files modified in Tasks 1–6, plus all agents and utilities  
**Verdict**: **Request Changes** — 4 critical, 9 major, 12 minor findings

---

## Critical Issues

### C1. Duplicate manifest write creates stale-then-updated file (pipeline.py:1447–1449)

**File**: `src/orchestrator/pipeline.py`  
**Lines**: 1447–1449

```python
manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")  # Write 1
self._register_runtime_artifact(manifest, manifest_path, "manifest")             # Mutates manifest
manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")  # Write 2
```

The manifest is written, then the manifest object is mutated by `_register_runtime_artifact` (which adds the manifest file entry to `manifest.files`), then written again. This is intentional but fragile — any reader between write 1 and write 2 sees stale data. If write 2 fails (disk full, permissions), the on-disk manifest is missing its own entry. The double I/O is also wasteful.

**Fix**: Remove line 1447. Register first, write once.

---

### C2. FFmpeg concat file path escaping is insufficient (audio_video_composer.py:416, video_renderer.py:376)

**File**: `src/agents/audio_video_composer.py` line 416, `src/agents/video_renderer.py` line 376

```python
escaped = vp.replace("\\", "/")
f.write(f"file '{escaped}'\n")
```

FFmpeg's concat demuxer uses single-quote delimited paths. If any video path contains a literal single quote (e.g., `scene_don't.mp4`), the concat file becomes malformed. Only backslash-to-forward-slash replacement is done — no quote escaping.

**Fix**: Escape single quotes per FFmpeg concat spec: `escaped = vp.replace("\\", "/").replace("'", "'\\''")`.

---

### C3. CLI tests are false positives — exit code never checked (test_cli.py)

**File**: `tests/test_cli.py`, all 6 tests

The tests mock `Orchestrator` and `CostTracker` but never assert `result.exit_code == 0`. The mocked `CostTracker` returns a `MagicMock` for `get_total_cost()`, which causes a `TypeError` when the CLI compares it with `> 0`. The CLI crashes, but the tests only check `mock_orch_cls.call_args` (which was set before the crash), so they pass anyway.

**Fix**: Add `assert result.exit_code == 0, result.output` to every test. Fix the `CostTracker` mock to return `MagicMock(get_total_cost=MagicMock(return_value=0.0), get_breakdown=MagicMock(return_value={}))`.

---

### C4. Unsafe `shutil.rmtree()` in cleanup without path validation (persistence.py:332–335)

**File**: `src/agents/persistence.py`, lines 308–339

```python
for item_path in output_path.iterdir():
    ...
    elif item_path.is_dir():
        import shutil
        shutil.rmtree(item_path)
```

`_cleanup_incomplete_files` calls `shutil.rmtree()` on any subdirectory inside the output dir without validating that paths are within expected bounds. If the output directory contains a symlink pointing outside the project (e.g., `media -> /`), this deletes the target. No TOCTOU protection either.

**Fix**: Resolve paths and verify they're children of the expected base: `assert item_path.resolve().is_relative_to(output_path.resolve())`. Consider using `shutil.rmtree(item_path, ignore_errors=False)` with explicit error handling instead of bare `except Exception: pass`.

---

## Major Issues

### M1. Inconsistent `scene_success_rate` default: 0.0 vs 1.0 (pipeline.py:302 vs 389)

In the dry-run path (line 302), when `total_scenes == 0`, success rate defaults to `0.0`. In the normal completion path (line 389), the same condition defaults to `1.0`. This means a dry-run with no scenes reports "0% success" while a normal run with no scenes reports "100% success." Confusing for operators.

**Fix**: Use `1.0` in both (no scenes = nothing failed), or `0.0` in both (no scenes = nothing succeeded). Pick one and be consistent.

### M2. `latest_result` can be None if quality loop never executes (pipeline.py:738)

In `_run_scene_controller`, `latest_result` is initialized to `None` and only assigned inside the quality loop body. If `max_quality_retries` is 0 or the loop is skipped, `latest_result` remains `None` and is returned, violating the return type contract.

**Fix**: Initialize to a default `SceneControllerOutput` or assert `max_quality_retries >= 1`.

### M3. Pydantic field validator may access unset fields (output_manifest.py:145–154)

```python
@field_validator('successful_scenes')
def validate_successful_scenes(cls, v, info):
    total = info.data.get('total_scenes', 0)
```

Field validators in Pydantic v2 run in field-declaration order. If `successful_scenes` is declared before `total_scenes`, `info.data` won't contain `total_scenes` yet. Currently the order happens to be correct (total_scenes at line ~105, successful_scenes at ~110), but this is fragile — a field reordering breaks the validator silently (defaults to 0).

**Fix**: Move this logic to the existing `@model_validator(mode='after')` at line 156, which always has all fields.

### M4. Silent exception swallowing in duration detection (video_quality_reviewer.py, audio_video_composer.py)

Both agents catch `Exception` in `_get_media_duration()` and return `0.0`. This means ffprobe failures, timeouts, and missing binaries all report "zero duration" instead of "unknown." Downstream code then flags these as duration mismatches, producing misleading quality findings.

**Fix**: Return a sentinel (e.g., `-1.0`) or `None` for failures, and handle it explicitly in callers. At minimum, log the exception.

### M5. Logger handler duplication on re-instantiation (logger.py:56)

`StructuredJSONLogger.__init__` checks `if not self.logger.handlers` to avoid adding duplicate console handlers. But `self.logger` is `logging.getLogger("pipeline.structured")` — a module-level singleton. If a second `StructuredJSONLogger` is created (common in tests), it sees handlers from the first instance and skips setup entirely. The second instance's file handler is never added to the logger.

**Fix**: Use instance-specific logger names (e.g., `f"pipeline.structured.{id(self)}"`) or track handlers per-instance.

### M6. `json.dumps()` can raise on non-serializable log data (logger.py:105)

```python
json_line = json.dumps(log_entry, ensure_ascii=False)
```

If any log entry field contains a non-JSON-serializable object (datetime, Path, bytes), `json.dumps` raises `TypeError` and the log line is silently lost (no try-except). Since agents can pass arbitrary data into `input_summary` / `output_summary`, this is plausible.

**Fix**: Add `default=str` to the `json.dumps()` call, or wrap in try-except.

### M7. Manifest double-write in persistence agent (persistence.py:214–237)

The persistence agent generates the manifest, writes it, then regenerates and overwrites. The first write is wasted I/O, and the first manifest may have stale data. If the second write fails, the on-disk file has incomplete data.

**Fix**: Generate once, write once.

### M8. TTS duration is estimated, not measured (tts_agent.py:413)

Duration is estimated from character count (`chars / 15.0`) before audio generation. The actual MP3 duration differs. The video quality reviewer then compares this estimate against video duration, producing false "sync mismatch" findings.

**Fix**: Extract actual duration from the generated MP3 using ffprobe after generation.

### M9. CWD-relative path fallback in video finder (video_renderer.py:353)

`_find_rendered_video` falls back to searching `Path("media")` relative to CWD. If the working directory isn't the project root (e.g., running from a parent directory), this finds the wrong file or a file from a different run.

**Fix**: Only search within `media_dir` (the explicitly configured Manim media directory). Remove the CWD fallback.

---

## Minor Issues

### m1. Unused imports in cli.py (lines 15, 18)

`os` and `Optional` are imported but never used. Ruff flags these.

### m2. `Path.as_posix()` should replace manual backslash replacement (pipeline.py:1381)

```python
rel = str(Path(final_video_path).relative_to(output_root)).replace("\\", "/")
```

**Fix**: `rel = Path(final_video_path).relative_to(output_root).as_posix()`

### m3. No boundary test for duration tolerance (test_video_quality_reviewer.py)

Tests use a 7s mismatch (well above 2.0s threshold) and a 1s mismatch (well below). No test at the exact boundary (2.0s vs 2.01s).

### m4. Timestamp validation is brittle (test_logger.py:328)

`assert '+' in timestamp or 'Z' in timestamp or timestamp.endswith('00:00')` would match midnight times. Use `datetime.fromisoformat(timestamp)` instead.

### m5. Missing `__init__.py` in `tests/fixtures/`

Created as empty file — correct, but should verify pytest discovers it properly.

### m6. Unused `field` import in video_renderer.py (line 14)

`from dataclasses import dataclass, field` — `field` is never used.

### m7. Cost tracker passes char count as `input_tokens` (tts_agent.py:221)

Characters ≠ tokens. The cost tracker accumulates wrong units.

### m8. No test for concurrent logger writes

Logger claims thread safety but has no concurrent write test.

### m9. `except Exception: pass` in cleanup methods (persistence.py:336, logger.py:315)

Silent exception swallowing in cleanup. Should at minimum `logger.debug()` the error.

### m10. Temp file leak if `os.replace()` raises unexpected exception (persistence.py:361–371)

`_write_file_atomically` creates a temp file with `delete=False`. If an exception other than `PermissionError` occurs after creation but before cleanup, the temp file is orphaned.

### m11. Quality loop exit reason mislabeled (pipeline.py:792–802)

Loop exits when `not scene_feedback` but logs `"accepted_with_validation_failures"`. The actual reason is "no feedback to act on."

### m12. Documentation: `output-contract.md` doesn't document invalid mode combinations

`--skip-render` with `--skip-tts=false` is valid but produces no audio (nothing to attach audio to). Not documented.

---

## What Looks Good

- **Logger graceful degradation**: The try-except around `_setup_log_file` with fallback to console-only is well designed.
- **FFmpeg preflight**: Checking `ffmpeg_available()` before rendering prevents late-stage subprocess failures.
- **Manifest as canonical contract**: `OutputManifest` with `RunConfig`, `final_video_path`, and typed `OutputFile` entries is a clean design.
- **Bounded quality loops**: `max_quality_retries=2` prevents infinite retry spirals.
- **CI split into fast/slow layers**: Unit tests on full matrix, integration on single version — good tradeoff.
- **Dry-run mode**: Clean stage-skipping logic (after the `scene_output=None` fix).
- **Structured logging**: JSON-line format with event types is well-suited for observability.

---

## Summary

| Severity | Count | Key Themes |
|----------|-------|------------|
| Critical | 4 | Double-write race, path injection, false-positive tests, unsafe rmtree |
| Major | 9 | Inconsistent defaults, uninitialized returns, silent failures, stale estimates |
| Minor | 12 | Unused imports, missing boundary tests, brittle assertions, doc gaps |

**Priority order for fixes**: C3 (tests give false confidence) → C4 (unsafe rmtree) → C1 (double-write) → C2 (path escaping) → M1–M4 (correctness) → rest.
