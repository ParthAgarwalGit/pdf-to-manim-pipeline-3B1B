"""Deterministic repair of a planned scene, so a countable mistake never
throws the whole plan away.

The visual planner's output is validated for object budgets and bookkeeping
(ids, purposes, beat references). Before this module, any violation that
survived the retries replaced the entire plan with a stub (a narration
sentence as a title plus one stock arrow): 30-50% of scenes in real runs.
Every one of those violations is mechanical, so it is fixed here instead:

- missing ids, purposes and beat references are filled in;
- transformations and beat entries pointing at unknown objects are dropped;
- a scene over the total-object budget loses its lowest-priority objects;
- a moment with too many objects on screen has the lowest-priority one
  retired just before the crowd, delayed until after it, or dropped.

Priority (kept first): objects the scene inherits or hands on, coordinate
systems, the scene's main formula, other drawings, further formulas, then
text labels.

Also here: the only overlap pass a plan gets. It moves text objects apart
from each other and never moves a drawing, because drawings that share
space (a vector on its plane, the axes of an ellipse) are meant to.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TEXT_PRIMITIVES = frozenset({"Text", "Tex", "MathTex", "DecimalNumber", "Paragraph", "Title"})
TEXT_OBJECT_TYPES = frozenset({"text", "equation"})
COORDINATE_PRIMITIVES = frozenset({"Axes", "NumberPlane", "NumberLine", "ComplexPlane", "ThreeDAxes"})
VAGUE_PURPOSES = frozenset({"", "decoration", "decorative", "filler", "visual", "shape"})
# Ids the keyless/failed-planner stub uses; they must never be carried into
# another scene or treated as designed content.
STUB_OBJECT_IDS = frozenset({"intro_text", "archetype_visual", "main_equation"})

# An object shown for less than this is a flicker, not a visual.
MIN_VISIBLE_SECONDS = 1.5
_EPS = 1e-6


def is_text_like(obj: Any) -> bool:
    primitive = getattr(obj, "primitive", None)
    if primitive:
        return primitive in TEXT_PRIMITIVES
    return (getattr(obj, "object_type", "") or "").lower() in TEXT_OBJECT_TYPES


def _main_formula_id(objects: Sequence[Any]) -> Optional[str]:
    """The scene's first formula in plan order: the one the scene is built around."""
    for obj in objects:
        primitive = getattr(obj, "primitive", None) or ""
        is_formula = primitive in ("MathTex", "Tex") or (
            not primitive and (getattr(obj, "object_type", "") or "").lower() == "equation"
        )
        if is_formula and getattr(obj, "object_id", None):
            return obj.object_id
    return None


def _priority(obj: Any, protected: frozenset, main_formula: Optional[str] = None) -> int:
    """Higher is kept longer."""
    if getattr(obj, "object_id", None) in protected:
        return 100
    if main_formula is not None and getattr(obj, "object_id", None) == main_formula:
        # The one formula a scene is about outranks its minor drawings: a
        # crowded scene loses a helper line before it loses its equation.
        return 52
    primitive = getattr(obj, "primitive", None) or ""
    object_type = (getattr(obj, "object_type", "") or "").lower()
    if primitive in COORDINATE_PRIMITIVES or object_type == "axes":
        return 60
    if not is_text_like(obj):
        return 55 if object_type == "graph" else 50
    if primitive in ("MathTex", "Tex") or object_type == "equation":
        return 30
    return 10


def _span(obj: Any, scene_end: float) -> Tuple[float, float]:
    interval = getattr(obj, "interval", None)
    if interval is None:
        return 0.0, scene_end
    return float(interval.t_in), float(interval.t_out)


def _scene_end(objects: Sequence[Any]) -> float:
    ends = [float(o.interval.t_out) for o in objects if getattr(o, "interval", None) is not None]
    return max(ends) if ends else 1.0


def max_concurrent(objects: Sequence[Any]) -> int:
    """Most objects visible at one moment (same rule as the validator)."""
    return _peak(objects)[0]


def _peak(objects: Sequence[Any]) -> Tuple[int, float]:
    """(largest visible count, earliest time it occurs)."""
    if not objects:
        return 0, 0.0
    end = _scene_end(objects)
    times = {0.0}
    for obj in objects:
        t_in, t_out = _span(obj, end)
        times.add(t_in)
        times.add(max(t_in, t_out - _EPS))
    best, best_time = 0, 0.0
    for t in sorted(times):
        count = sum(1 for o in objects if _span(o, end)[0] <= t < _span(o, end)[1])
        if count > best:
            best, best_time = count, t
    return best, best_time


def _set_interval(obj: Any, t_in: float, t_out: float) -> None:
    """Replace the object's interval (a new validated instance, never a half-mutated one)."""
    from src.schemas.storyboard import TimeInterval

    obj.interval = TimeInterval(t_in=round(max(0.0, t_in), 3), t_out=round(t_out, 3))


def fill_bookkeeping(
    objects: List[Any],
    valid_beat_ids: Sequence[str],
    beat_windows: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[str]:
    """Give every object an id, a purpose and at least one valid beat reference.

    ``beat_windows`` (beat_id -> (start, end) seconds) lets an object with no
    usable reference take the beat during which it appears; without it the
    first valid beat is used. Returns notes describing what was filled in.
    """
    notes: List[str] = []
    valid = list(valid_beat_ids)
    valid_set = set(valid)
    taken = {o.object_id for o in objects if getattr(o, "object_id", None)}
    counter = 0
    for obj in objects:
        if not getattr(obj, "object_id", None):
            counter += 1
            while f"obj_{counter}" in taken:
                counter += 1
            obj.object_id = f"obj_{counter}"
            taken.add(obj.object_id)
            notes.append(f"assigned id {obj.object_id}")
        if (getattr(obj, "purpose", "") or "").strip().lower() in VAGUE_PURPOSES:
            label = (getattr(obj, "content", "") or "").strip()[:40] or (getattr(obj, "primitive", None) or "object")
            obj.purpose = f"Shows {label} for the narrated idea"
            notes.append(f"filled purpose of {obj.object_id}")
        if not valid:
            continue
        kept = [b for b in (getattr(obj, "supports_beat_ids", None) or []) if b in valid_set]
        if not kept:
            kept = [_beat_at(obj, valid, beat_windows)]
            notes.append(f"assigned beat {kept[0]} to {obj.object_id}")
        obj.supports_beat_ids = kept
    return notes


def _beat_at(obj: Any, valid: Sequence[str], beat_windows: Optional[Dict[str, Tuple[float, float]]]) -> str:
    interval = getattr(obj, "interval", None)
    if beat_windows and interval is not None:
        for beat_id in valid:
            window = beat_windows.get(beat_id)
            if window and window[0] <= interval.t_in < window[1]:
                return beat_id
        last = valid[-1]
        if beat_windows.get(last) and interval.t_in >= beat_windows[last][0]:
            return last
    return valid[0]


def enforce_budgets(
    objects: List[Any],
    max_total: int,
    max_concurrent_allowed: int,
    protected_ids: Iterable[str] = (),
) -> Tuple[List[Any], List[str]]:
    """Cut a plan down to its budgets. Returns (kept objects, notes).

    Order within the returned list is the plan's order. Objects in
    ``protected_ids`` are cut only when nothing else is left to cut. The last
    remaining drawing (non-text object) is never removed.
    """
    protected = frozenset(protected_ids)
    kept = list(objects)
    notes: List[str] = []
    order = {id(o): i for i, o in enumerate(kept)}
    end = _scene_end(kept)
    main_formula = _main_formula_id(kept)

    def cut_key(obj: Any):
        t_in, t_out = _span(obj, end)
        # lowest priority first; among equals the shortest-lived, then the latest planned
        return (_priority(obj, protected, main_formula), t_out - t_in, -order[id(obj)])

    def removable(candidates: Sequence[Any]) -> List[Any]:
        drawings = [o for o in kept if not is_text_like(o)]
        return [o for o in candidates if is_text_like(o) or len(drawings) > 1 or o not in drawings]

    while len(kept) > max_total:
        candidates = removable(kept)
        if not candidates:
            break
        victim = min(candidates, key=cut_key)
        kept.remove(victim)
        notes.append(f"dropped {victim.object_id} (over {max_total} objects)")

    # Each pass retires, delays or drops one object, so this terminates.
    for _ in range(4 * len(kept) + 4):
        count, at = _peak(kept)
        if count <= max_concurrent_allowed:
            break
        visible = [o for o in kept if _span(o, end)[0] <= at < _span(o, end)[1]]
        candidates = removable(visible)
        if not candidates:
            break
        victim = min(candidates, key=cut_key)
        t_in, t_out = _span(victim, end)
        if at - t_in >= MIN_VISIBLE_SECONDS:
            # It has had its time on screen: retire it as the crowd arrives.
            _set_interval(victim, t_in, at)
            notes.append(f"retired {victim.object_id} at {at:.1f}s")
            continue
        clear_at = _first_clear_time(kept, victim, at, t_out, end, max_concurrent_allowed)
        if clear_at is not None and t_out - clear_at >= MIN_VISIBLE_SECONDS:
            _set_interval(victim, clear_at, t_out)
            notes.append(f"delayed {victim.object_id} to {clear_at:.1f}s")
            continue
        kept.remove(victim)
        notes.append(f"dropped {victim.object_id} (more than {max_concurrent_allowed} visible at {at:.1f}s)")
    return kept, notes


def _first_clear_time(
    objects: Sequence[Any], victim: Any, after: float, before: float, end: float, limit: int,
) -> Optional[float]:
    """Earliest moment in (after, before) when the others leave room for one more."""
    others = [o for o in objects if o is not victim]
    times = sorted({_span(o, end)[1] for o in others if after < _span(o, end)[1] < before})
    for t in times:
        count = sum(1 for o in others if _span(o, end)[0] <= t < _span(o, end)[1])
        if count < limit:
            return t
    return None


def resolve_text_overlaps(
    objects: Sequence[Any], safe_x: float, safe_y: float, gap: float = 0.25, iterations: int = 4,
) -> int:
    """Nudge time-overlapping TEXT objects apart; drawings are never moved.

    For each pair of text-like objects visible at the same time whose boxes
    intersect, the later one in plan order moves along the axis of least
    penetration, clamped to the safe area. Returns the number of moves.
    """
    texts = [o for o in objects if getattr(o, "bbox", None) is not None and is_text_like(o)]
    moves = 0
    for _ in range(iterations):
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = texts[i], texts[j]
                if not _windows_overlap(getattr(a, "interval", None), getattr(b, "interval", None)):
                    continue
                dx = (a.bbox.w + b.bbox.w) / 2 - abs(a.bbox.x - b.bbox.x)
                dy = (a.bbox.h + b.bbox.h) / 2 - abs(a.bbox.y - b.bbox.y)
                if dx <= 0 or dy <= 0:
                    continue
                if dx <= dy:
                    direction = 1.0 if b.bbox.x >= a.bbox.x else -1.0
                    b.bbox.x = max(-safe_x, min(safe_x, b.bbox.x + direction * (dx + gap)))
                else:
                    direction = 1.0 if b.bbox.y >= a.bbox.y else -1.0
                    b.bbox.y = max(-safe_y, min(safe_y, b.bbox.y + direction * (dy + gap)))
                moved = True
                moves += 1
        if not moved:
            break
    return moves


def _windows_overlap(a: Optional[Any], b: Optional[Any]) -> bool:
    if a is None or b is None:
        return True
    return a.t_in < b.t_out and b.t_in < a.t_out


def repair_visual_intent(
    intent: Any,
    valid_beat_ids: Sequence[str],
    max_total: int,
    max_concurrent_allowed: int,
    protected_ids: Iterable[str] = (),
    beat_windows: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[str]:
    """Repair ``intent`` in place; return notes (empty when nothing changed).

    After this call the plan has ids, purposes and valid beat references on
    every object, fits both object budgets (whenever cutting text and spare
    drawings can achieve it), and no transformation, beat entry or carry-out
    id refers to an object that is no longer in the plan.
    """
    objects = list(intent.mathematical_objects)
    notes = fill_bookkeeping(objects, valid_beat_ids, beat_windows)
    objects, budget_notes = enforce_budgets(objects, max_total, max_concurrent_allowed, protected_ids)
    notes.extend(budget_notes)
    intent.mathematical_objects = objects

    ids = {o.object_id for o in objects}
    kept_transformations = [
        t for t in intent.transformations
        if t.source_object_id in ids and (not t.target_object_id or t.target_object_id in ids)
    ]
    if len(kept_transformations) != len(intent.transformations):
        notes.append(
            f"dropped {len(intent.transformations) - len(kept_transformations)} transformation(s) "
            "that referred to missing objects"
        )
        intent.transformations = kept_transformations
    for beat in intent.beats:
        beat.active_object_ids = [i for i in beat.active_object_ids if i in ids]
    for state in (intent.scene_state_in, intent.scene_state_out):
        state.active_object_ids = [i for i in state.active_object_ids if i in ids]
    return notes


def is_stub_intent(intent: Any) -> bool:
    """True when every object in the plan is one of the stub's (no designed content)."""
    ids = [getattr(o, "object_id", None) for o in intent.mathematical_objects]
    return bool(ids) and all(i in STUB_OBJECT_IDS for i in ids)


def demote_stub(intent: Any) -> None:
    """Make a stub plan harmless, in place.

    The stub's title is the narration's first sentence; on screen it is the
    exact "text that repeats the voice" the video must not have, so it is
    removed whenever another object remains. Nothing is handed to the next
    scene, so one failed plan cannot seed the scenes after it.
    """
    others = [o for o in intent.mathematical_objects if getattr(o, "object_id", None) != "intro_text"]
    if others:
        intent.mathematical_objects = others
        ids = {o.object_id for o in others}
        intent.transformations = [
            t for t in intent.transformations
            if t.source_object_id in ids and (not t.target_object_id or t.target_object_id in ids)
        ]
        for beat in intent.beats:
            beat.active_object_ids = [i for i in beat.active_object_ids if i in ids]
        intent.scene_state_in.active_object_ids = [
            i for i in intent.scene_state_in.active_object_ids if i in ids
        ]
    strip_stub_carry(intent)


def strip_stub_carry(intent: Any) -> None:
    """A stub plan hands nothing to the next scene."""
    intent.scene_state_out.active_object_ids = []


def carryable_ids(intent: Any) -> List[str]:
    """Carry-out ids that are real designed objects (never the stub's)."""
    return [i for i in (intent.scene_state_out.active_object_ids or []) if i not in STUB_OBJECT_IDS]
