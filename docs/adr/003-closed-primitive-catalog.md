# ADR-003: Closed Manim Primitive Catalog

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** Parth Agarwal
**Plan:** [`docs/superpowers/plans/2026-04-26-unified-script-chunked-render.md`](../superpowers/plans/2026-04-26-unified-script-chunked-render.md) (Task 12)

---

## Context

Before this ADR, the visual planner emitted `MathematicalObject` records with
free-form fields (`object_type`, `render_hint`, `content`). The animation
generator's `_generate_mathematical_objects` (~lines 528–700 of
`animation_generator.py`) interpreted them through a long `if/elif` chain
that kept growing as the LLM invented new shapes.

Three concrete pains:

1. **Render-hint drift.** The LLM produced hints not in the canonical set
   (e.g. `"PiCreatureSpeechBubble"` instead of `"create_speech_bubble"`),
   forcing the heuristic chain to fall back to a generic `Polygon`.
2. **Free-form `content` ambiguity.** A single field carried LaTeX, prose,
   and shape descriptions. The animation generator had to guess which.
3. **Heuristics owned the geometry.** The animation generator inferred
   positions, sizes, and colors from prose. Two PDFs with similar narration
   produced wildly different layouts because the heuristics fired
   differently.

---

## Options Considered

### Option A — Closed catalog of primitives + args (chosen)
Define `MOBJECT_CATALOG` and `ANIMATION_CATALOG` in
`src/utils/manim_primitives.py`. Each entry declares the Manim class name,
its allowed kwargs, and whether it accepts `color`. `MathematicalObject`
gets `primitive` + `args` fields validated at the Pydantic boundary against
the catalog. The animation generator becomes a thin emitter:
`_emit_primitive(var_name, obj_data)` returns one line of Python.

**TCO (3 years):** Low — adding a new primitive is one entry in
`manim_primitives.py`. Schema validation rejects unknown primitives at the
LLM/agent boundary, so the rescue paths in the animation generator never
get exercised.
**Risk:** Low — legacy `MathematicalObject` records (no `primitive` set)
fall through to the existing `if/elif` chain, so the migration is gradual.

### Option B — Function-calling LLM with a registered tool per primitive
Use OpenAI / Anthropic function-calling to expose each Manim primitive as
a typed tool the LLM can invoke directly.

**TCO (3 years):** Medium — locks the project deeper into a specific
provider's tool-calling syntax; each primitive needs a JSON Schema dual
to its catalog entry.
**Risk:** Medium — provider-specific behavior; harder to switch models.

### Option C — Keep the heuristic chain, just expand it
Add more `elif` branches as the LLM invents new shapes.

**TCO (3 years):** High — the chain has already grown organically and is
hard to test exhaustively.
**Risk:** High — every new shape is a new bug surface.

---

## Decision

**The visual planner emits a closed-vocabulary `primitive` + `args` pair.
The schema rejects anything outside the catalog. The animation generator
emits catalog primitives with `_emit_primitive`; the legacy heuristic
chain is the fallback only for legacy intents without `primitive` set.**

Implementation:
- `src/utils/manim_primitives.py` — `MOBJECT_CATALOG`, `ANIMATION_CATALOG`,
  `catalog_for_prompt()`, `allowed_args_for()`.
- `src/schemas/storyboard.py` — `MathematicalObject.primitive`,
  `MathematicalObject.args`, plus a `model_validator` that calls
  `allowed_args_for`.
- `src/agents/script_architect.py` — `catalog_for_prompt()` is injected
  into `_generate_visual_intent_for_chunk`'s prompt with a "primitive +
  args, content only for MathTex/Tex/Text" rule.
- `src/agents/animation_generator.py` — `_emit_primitive` and
  `_emit_bbox_placement` short-circuit the heuristic chain when
  `primitive` is set.

---

## Consequences

### What becomes easier
- The LLM's output space is bounded by a known menu — fewer "what did it
  mean?" surprises.
- Adding a new Manim primitive is one entry in `MOBJECT_CATALOG`, plus
  optionally one new condition in `_emit_primitive` if the call shape is
  unusual.
- Schema-level rejection catches LLM hallucination at the agent boundary
  rather than letting it propagate to the renderer.

### What becomes harder
- Two object representations now coexist in `_generate_mathematical_objects`
  (catalog path + legacy `if/elif`). The legacy path stays until catalog
  adoption is proven on real PDFs (target: delete after 2-3 successful
  full runs).
- The visual planner prompt grows by the catalog block (~40 lines). Token
  cost is acceptable (one chunk-scoped call) but worth monitoring if the
  catalog grows >50 primitives.

### Standing constraints
- `MathematicalObject.primitive` is the canonical positioning hint going
  forward. `render_hint` is retained for legacy compatibility only and
  should not be added to new code paths.
- The catalog is **closed** — new primitives must be added to the registry
  before the LLM may emit them. Do not "sneak" new primitives in by
  catching unknown values in the emitter.
