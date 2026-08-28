# ADR-005: Visual-First Render Path (plan-informed LLM codegen)

**Status:** Accepted
**Date:** 2026-07-01
**Deciders:** Parth Agarwal
**Supersedes (in part):** the "thin emitter" framing of
[ADR-003](003-closed-primitive-catalog.md) and the wiring assumptions of
[ADR-004](004-xyt-layout-validation.md)

---

## Context

The product thesis was sharpened: this is a **visualization tool** — it must
render highly impactful, intuitive visuals covering the *breadth* of a
document. It is explicitly **not** a narrated-lecture generator, and videos
need not be long.

An audit of the real (API-key-present) render path found three gaps between
the documented architecture and what actually runs:

1. **Plan-then-discard.** The visual planner (`ScriptArchitect`,
   `_generate_visual_intent_for_chunk`, GPT‑5) emits structured
   `MathematicalObject`s with a closed-catalog `primitive`, `args`, a `bbox`
   (x, y) and an `interval` (t_in, t_out). But `AnimationCodeGenerator.
   _generate_code` built its prompt from only `type/content/render_hint/style`
   — the primitive, args, position and interval were **dropped**. The code LLM
   (the code-generation model) re-invented each visual from scratch. We paid for structured
   planning and then ignored it.
2. **The deterministic emitter is mock-only.** ADR-003 describes the animation
   generator as "a thin emitter" using `_emit_primitive`. In reality that path
   (`_generate_mathematical_objects`) is reached **only** by
   `_mock_generate_code` (no API key). On real runs, Gemini 3.7 Flash generates the code.
3. **Narration-first timing.** Duration was `word_count / 3.0` with a 15s floor,
   and `_enforce_duration` scaled `self.wait()` calls up to match that number —
   inflating visually rich scenes with dead-air waits. This produced long,
   static videos, the opposite of the thesis.

---

## Options Considered

### Option A — Deterministic-emitter render
Promote `_emit_primitive` + bbox placement to the real render path; use the
LLM only for LaTeX / expression strings.

**TCO (3 years):** Low — reproducible, cheap, layout guaranteed.
**Risk:** Medium — ceiling is the catalog's expressiveness; the signature
"impactful" moves (morphs, updaters, space transforms) are hard to template.

### Option B — LLM-first, drop structured intent
Stop generating structured `VisualIntent`/bbox/interval; hand the code LLM the
segment's concepts + a visual-design brief and let it design freely.

**TCO (3 years):** Medium — saves the GPT‑5 planning call.
**Risk:** High — no layout guarantee, low reproducibility, no structured object
we can validate or carry across scenes for continuity.

### Option C — Hybrid: the plan informs an LLM render (chosen)
The planner keeps emitting the structured plan (primitive, args, position,
interval, continuity keys). `_generate_code` forwards that **whole** plan to the
code LLM and instructs it to *honor* it (construct the planned primitive at the
planned position for the planned window), while retaining the LLM's strength for
motion, progressive reveal and helper composition. A deterministic layout /
safe-area pass will own final positioning in a later step.

**TCO (3 years):** Low–Medium — one prompt surface; the plan is a contract, not
a straitjacket. Expressiveness stays high; the plan gives reproducibility and
continuity hooks.
**Risk:** Low — additive to the existing path; degrades gracefully when a plan
field is absent (LLM falls back to type/content).

---

## Decision

**Adopt Option C (hybrid).** The structured visual plan is the source of truth
for *what* is on screen and *where/when*; the code LLM owns *how* it animates.
Concretely, in this slice:

- `AnimationCodeGenerator._object_plan_for_prompt` serializes the full plan
  (`primitive`, `args`, `id`, `role`, `position` {x,y}, `visible`
  {t_in,t_out}, `notation_key`, `coordinate_system_id`, `character_role`,
  `render_hint`, `style`), omitting empty fields.
- A "VISUAL PLAN (HONOR IT)" system-prompt section instructs the model to
  construct the planned primitive at the planned position/window and to not
  silently downgrade it (e.g. a planned `Vector` must not become a static
  `Line`).
- Timing is now **visual-first**: `MIN_SCENE_SECONDS = 8.0` is a floor only;
  `_enforce_duration` pads a scene *up to the floor* with a single tail wait and
  never scales visuals up to narration length. The pacing prompt tells the model
  to fill time with motion, not empty waits.
- The primitive catalog gained the dynamic "intuition-builder" primitives
  (`Vector`, `DoubleArrow`, `DashedLine`, `Ellipse`, `Triangle`,
  `RoundedRectangle`, `DecimalNumber`) so the plan can express them.

---

## Consequences

### What becomes easier
- The GPT‑5 plan now materially shapes the render; the two LLM stages stop
  contradicting each other.
- Continuity keys (notation / coordinate systems) reach the code LLM, so shared
  objects can be reused rather than redrawn.
- Scenes are as long as the visuals need — short, dense vignettes are now the
  default, matching the thesis.

### What becomes harder / open items
- The plan is advisory to an LLM, so honoring it is not yet *guaranteed*. A
  deterministic layout/safe-area validator (revive or delete the currently
  unwired `scene_layout_planner` + `layout_validator`) is the next step to make
  positioning enforceable, not merely requested.
- `_emit_primitive` (the deterministic path) remains mock-only. It is retained
  as the seed of a future deterministic render mode; ADR-003's "thin emitter"
  claim should be read as *aspirational for the mock path only* until that work
  lands.
- Video may now be shorter than its narration audio. The audio-video composer
  already handles per-scene timing mismatches; if narration regularly dominates,
  reduce narration (make TTS opt-in) rather than re-inflating the video.

### Standing constraints
- New primitives must be added to `MOBJECT_CATALOG` before the planner may emit
  them (the catalog stays closed — ADR-003 still holds here).
- Do not reintroduce narration-length padding. Scene length is a function of
  visual work, floored at `MIN_SCENE_SECONDS`.
