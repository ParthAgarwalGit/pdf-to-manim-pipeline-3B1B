# ADR-002: Unified Video Script + Deterministic Chunked Render

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** Parth Agarwal
**Supersedes:** Per-concept narration loop in the original `ScriptArchitectAgent`
**Plan:** [`docs/superpowers/plans/2026-04-26-unified-script-chunked-render.md`](../superpowers/plans/2026-04-26-unified-script-chunked-render.md)

---

## Context

The original `ScriptArchitectAgent` generated narration **per concept**, with each
LLM call seeing the full PDF and zero context about prior scenes. This caused two
chronic problems on real PDFs:

1. **Conceptual redundancy.** Scene 4 would re-explain "sine" because the prompt
   for scene 4 had no signal that scene 1 already taught it. The viewer heard
   the same explanation 3-5 times across a 10-scene video.
2. **Cost and latency.** Every per-concept call shipped the entire PDF text to
   the LLM, multiplying token spend by the scene count.

The pipeline needed a content-contract change: each scene must be a slice of one
coherent script, not an independent re-derivation.

---

## Options Considered

### Option A — One LLM call returns all scenes (chosen)
A single `_generate_video_script` call produces a `VideoScript` (list of `Beat`s
with an explicit concept-introduction ledger). A deterministic chunker
(`_chunk_script_into_scenes`) groups beats into render-sized scenes (~30–90s
each). The full PDF only enters the script-generation call; per-scene LLM calls
receive a `NarrativeContext` briefing instead.

**TCO (3 years):** Low — one prompt to evolve, one place where the PDF is
consumed, narrative continuity is a first-class field.
**Risk:** Medium — a single LLM call must produce a coherent multi-scene script.
Mitigated by per-scene fallback paths (F6 in the plan) and concept-canonicalization
(`_canonicalize_concept`, F1).

### Option B — Cache + dedupe the per-concept calls
Keep the per-concept loop but feed each call a "concepts already explained" list.

**TCO (3 years):** Medium — every per-concept prompt grows over the run; dedup is
heuristic (LLM may still re-explain in different words).
**Risk:** Medium — does not solve the root problem (no narrative arc), just
suppresses the symptom.

### Option C — No change
Live with the redundancy.

**TCO (3 years):** High — viewers stop watching, every PDF wastes tokens.
**Risk:** Strategic — defeats the point of the pipeline.

---

## Decision

**The Script Architect emits one `VideoScript`. A deterministic chunker groups
beats into Scenes. Per-scene prompts read `NarrativeContext`, never the full PDF.**

The full implementation spans Tasks 1–10 of the plan and lives across:
- `src/schemas/video_script.py` (new) — `Beat`, `VideoScript`
- `src/schemas/scene_processing.py` — `NarrativeContext` field on `SharedContext`
- `src/agents/script_architect.py` — `_generate_video_script`,
  `_chunk_script_into_scenes`, `_generate_visual_intent_for_chunk`
- `src/agents/scene_controller.py` — `_update_context` populates `NarrativeContext`
- `src/agents/animation_generator.py` — `_build_narrative_blocks` injects
  briefing + opening-state into the system prompt

---

## Consequences

### What becomes easier
- A single concept is taught at most once across the storyboard
  (verified by `tests/integration/test_end_to_end.py::TestNoRedundantConceptIntroductions`).
- New chunks can reference earlier concepts cheaply (briefing line, not re-explanation).
- Token spend scales with PDF length, not (PDF length × scene count).
- Visual continuity at chunk boundaries (live objects re-instantiated at
  `run_time=0`) makes cuts seamless.

### What becomes harder
- Adding a new concept-extraction strategy requires touching the `VideoScript`
  schema, not a single prompt.
- The chunker's heuristics (`MIN_SCENE_DURATION=30`, `MAX_SCENE_DURATION=90`,
  `TARGET_SCENE_DURATION=60`) are now load-bearing — changing them shifts every
  storyboard.

### Standing constraints
- `concept_hierarchy` in the storyboard is **derived from** the script's
  concept ledger — it is no longer free-form input. Tools that consume the
  manifest must treat the ledger as the source of truth.
- `Scene.dependencies` is always a single-step chain (`scene_N` depends on
  `[scene_(N-1)]`).
- Per-scene LLM prompts must NEVER receive the raw PDF or OCR text — that is
  the script generator's exclusive responsibility. New downstream stages that
  need context should consume `NarrativeContext`.
