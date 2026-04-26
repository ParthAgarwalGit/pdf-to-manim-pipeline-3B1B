# Output Contract

Every `pdf2manim convert` run produces a single, self-contained output directory. This document describes the contract that operators and downstream tools can rely on.

## Output Directory Layout

```
output/{pdf_stem}/{ISO-timestamp}/
├── manifest.json              # Canonical run artifact (always present)
├── pipeline.log               # Structured JSON execution log
├── storyboard.json            # Scene specifications from the architect
├── narration_script.txt       # Full narration with scene markers
├── scene_001.py … scene_NNN.py  # Generated Manim animation code
├── storyboard_review.json     # Storyboard quality grading (if quality loops enabled)
├── scene_code_review.json     # Per-scene code validation (if quality loops enabled)
├── video_review.json          # Final media quality review (if render enabled)
├── media/                     # Manim render artifacts (if render enabled)
│   └── videos/…/*.mp4
├── scene_001.mp3 … scene_NNN.mp3  # TTS audio per scene (if TTS enabled)
└── final_video.mp4            # Composed narrated video (if render + TTS enabled)
```

## manifest.json

The manifest is the canonical run artifact. It is always written, even on failure.

### Key Fields

| Field | Type | Description |
|---|---|---|
| `output_directory` | `string` | Absolute path to the run directory |
| `status` | `string` | `SUCCESS`, `PARTIAL_SUCCESS`, or `CRITICAL_FAILURE` |
| `total_scenes` | `int` | Number of scenes planned |
| `successful_scenes` | `int` | Number of scenes that completed |
| `scene_success_rate` | `float` | `successful_scenes / total_scenes` |
| `final_video_path` | `string?` | Relative path to the final video, or `null` if render was skipped |
| `run_config` | `RunConfig?` | Effective execution flags for this run |
| `files` | `OutputFile[]` | Every artifact with path, size, and type |
| `failed_scenes` | `FailedScene[]` | Scene IDs and error messages |

### OutputFile Types

`animation_code`, `narration_script`, `storyboard`, `manifest`, `report`, `ocr_output`, `log`, `scene_video`, `scene_audio`, `composed_video`, `final_video`

### RunConfig

Records the effective execution settings:

```json
{
  "dry_run": false,
  "enable_render": true,
  "enable_tts": true,
  "render_quality": "medium",
  "tts_voice": "alloy"
}
```

## Execution Modes

| Mode | Artifacts Produced |
|---|---|
| `--dry-run` | storyboard, narration script, manifest, pipeline.log |
| `--skip-render` | Above + scene code files |
| `--skip-tts` | Above + rendered scene videos (no audio, no final compose) |
| Full (default) | All artifacts including final_video.mp4 |

### Invalid Mode Combinations

Some flag combinations are redundant or non-sensical. The CLI will warn or error:

| Combination | Behaviour |
|---|---|
| `--dry-run` + `--skip-render` | `--skip-render` is redundant — dry-run already skips render. Runs as `--dry-run`. |
| `--skip-tts` without render | TTS audio is only useful when composed with video. TTS is silently skipped. |
| `--skip-render` + `--skip-tts` | Valid — produces storyboard, narration, and scene code only. |

## Storyboard & Narrative Contract

Starting with the unified-script-chunked-render flow, the storyboard layer
guarantees these invariants — downstream tools may rely on them:

- **`concept_hierarchy`** is derived from the unified video script's deduped
  concept ledger. Each entry is one canonical concept in first-introduction
  order, with `dependencies` listing every concept introduced earlier (a flat
  linear chain that mirrors the narrative).
- **`Scene.dependencies`** is always a single-step chain: `scene_N` depends on
  `[scene_(N-1)]` (or `[]` for the first scene). Scenes are render chunks of
  one continuous script, never independent units.
- **`Scene.source_beats`** is the `Beat` list grouped into the scene by the
  chunker. A beat is atomic and never split across scene boundaries. Empty
  for legacy storyboards from before this contract.
- **`SharedContext.narrative`** carries cross-scene continuity:
  - `concepts_introduced` — canonical concept labels already taught (the
    "do not re-explain" set).
  - `prior_scene_summaries` — one short summary per prior scene, in order.
  - `running_storyline` — capped at 3 sentences.
  - `live_objects_on_screen` — object_id → short description for objects
    visible at the END of the most recent scene; the next scene re-creates
    them at `run_time=0` to make the cut seamless.

Per-scene LLM prompts (visual planner + animation generator) consume the
narrative briefing and the chunk's narration only — never the raw PDF text,
which is the script generator's responsibility alone.

## Quality Review Artifacts

When quality loops are enabled (default), up to three review files are produced:

- **storyboard_review.json** — Grading of scene pedagogical quality (threshold: 6.0/10)
- **scene_code_review.json** — Per-scene Manim code validation
- **video_review.json** — Final media checks: duration alignment, render failures, layout risk

Each review contains structured findings with `suspected_stage`, `scene_id`, `evidence`, `suggested_fix`, and optional `code_hint`.

## Inspecting a Run

```bash
# Check overall status
cat output/my_doc/2026-…/manifest.json | python -m json.tool | head -20

# Find the final video
jq '.final_video_path' output/my_doc/2026-…/manifest.json

# List all generated files by type
jq '.files[] | {type, path}' output/my_doc/2026-…/manifest.json

# Review quality findings
cat output/my_doc/2026-…/video_review.json | python -m json.tool
```
