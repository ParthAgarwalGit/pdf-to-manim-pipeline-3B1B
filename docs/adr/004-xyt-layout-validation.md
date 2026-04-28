# ADR-004: Strict (x, y, t) Layout Validation with Disjoint-Interval Reuse

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** Parth Agarwal
**Plan:** [`docs/superpowers/plans/2026-04-26-unified-script-chunked-render.md`](../superpowers/plans/2026-04-26-unified-script-chunked-render.md) (Task 13)
**Related:** ADR-003 (Closed primitive catalog) — bbox is emitted alongside `primitive`/`args`.

---

## Context

The original `LayoutValidatorAgent` flagged a pair of objects as overlapping
whenever:
- their axis-aligned bounding rectangles intersected, AND
- both had a positive `duration` (`max(corner.t)` across the four corners).

This produced two failure modes:

1. **False positives on serial reuse.** When chunk N removed an equation and
   chunk N+1 placed a new equation in the same screen position, the validator
   flagged them as overlapping because both had `duration > 0` — even though
   the screen was unambiguously empty between them.
2. **Off-screen blindness.** The validator only checked pairs; an isolated
   object placed at `x = 8` (outside the Manim frame) passed.

The unified-script flow (ADR-002) makes serial position reuse the *normal*
case — chunks regularly clear the stage and re-populate it — so the false
positives became blocking.

---

## Options Considered

### Option A — Add explicit `[t_in, t_out)` intervals + safe-area check (chosen)
`CharacterBoundingBox` gains `t_in: float = 0.0` and `t_out: Optional[float] = None`
fields. `t_out` defaults to the legacy `duration` so existing layouts behave
unchanged. The validator switches to interval intersection
(`max(t_in) < min(t_out)`) instead of `min(duration, duration)`. A new
`OffScreenViolation` flags any centre outside `SAFE_X × SAFE_Y` (`[-6.5, 6.5]
× [-3.5, 3.5]` — the Manim frame minus a label margin). New `BBox` and
`TimeInterval` schemas live on `MathematicalObject` so the visual planner can
emit positions/intervals that the layout planner translates into a
`CharacterBoundingBox`.

**TCO (3 years):** Low — one validator update, one schema extension. Future
layout features (transformations, camera moves) can layer on the interval
contract without re-litigating it.
**Risk:** Low — defaults preserve legacy behavior; new tests cover both
serial-reuse-OK and simultaneous-overlap-flagged cases.

### Option B — Replace the corner schema with centre+w+h+interval
Drop `CharacterBoundingBox` entirely; everything uses `BBox` + `TimeInterval`
directly.

**TCO (3 years):** Medium — every layout-system caller has to be migrated;
the layout planner agent's prompt has to be rewritten.
**Risk:** High — breaks the layout planner's evaluator loop until all
consumers are updated.

### Option C — Heuristic suppress duplicate-position warnings
Keep `duration`-only semantics but skip violations when one object's
`disappear_at` is close to the other's `appear_at`.

**TCO (3 years):** High — heuristic boundary cases multiply; "close to" needs
tuning per chunk; doesn't address off-screen blindness.
**Risk:** High — every new transformation type needs a new heuristic.

---

## Decision

**Layout validation is `(x, y, t)`-aware via explicit intervals on
`CharacterBoundingBox`. Two objects overlap iff their bboxes intersect AND
their `[t_in, t_out)` windows intersect. Off-screen centres are an
independent violation class.**

Implementation:
- `src/schemas/scene_layout.py` — `t_in`/`t_out` on
  `CharacterBoundingBox`, `visible_t_in`/`visible_t_out` properties (legacy
  fallback to `duration`), `OffScreenViolation`, `SAFE_X_*`/`SAFE_Y_*`
  constants.
- `src/schemas/storyboard.py` — `BBox`, `TimeInterval`, and
  `MathematicalObject.bbox` / `MathematicalObject.interval` fields.
- `src/agents/layout_validator.py` — `_pair_overlap` uses interval
  intersection; new `_check_off_screen` method.
- `src/agents/script_architect.py` — visual planner prompt grows a LAYOUT
  block requiring `bbox` + `interval` per object and naming the safe-area
  + serial-reuse rules.
- `src/agents/animation_generator.py` — `_emit_bbox_placement` appends
  `obj.move_to([x, y, 0])` when bbox is present.

---

## Consequences

### What becomes easier
- The "stage carries on" pattern between chunks (clear → repopulate the same
  position) is now a first-class supported case, not a rejected layout.
- Off-screen violations get caught at validation time, not at render time.
- Violation descriptions name the shared time window
  (`"a and b overlap ... (t=2.0..8.0)"`), giving the layout-planner retry
  loop a concrete fix target.

### What becomes harder
- Two parallel position schemas now coexist: the layout system's
  `CharacterBoundingBox` (4 corners + `t_in`/`t_out`) and the visual
  planner's `MathematicalObject.bbox` + `MathematicalObject.interval`. The
  layout planner agent translates between them. This duplication is
  documented as known debt; a future ADR may collapse them once the new
  fields are exercised on real PDFs.
- Layouts authored before this ADR rely on the `t_out=None` default; if a
  legacy storyboard sets `t_in > 0` without setting `t_out`, the
  visibility window becomes `[t_in, duration)` which may be empty and
  trigger validation surprises. Mitigated by `_validate_interval` raising
  on `t_out <= t_in`.

### Standing constraints
- The safe area `[-6.5, 6.5] × [-3.5, 3.5]` is a one-line constant in
  `src/schemas/scene_layout.py`. Changes require an ADR superseding this
  one because every layout test depends on the bound.
- New layout consumers MUST handle a heterogeneous `violations` list
  (`OverlapViolation | OffScreenViolation`). Use the human-readable
  `description` field for retry-loop feedback rather than discriminating
  by type.
- Future layout system extensions (animations, camera moves, viewport
  zooms) must define their own `[t_in, t_out)` semantics; do not regress
  to the legacy `duration` model.
