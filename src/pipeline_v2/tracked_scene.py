"""Thin runtime instrumentation for raw Manim scenes."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from manim import Camera, Group, Mobject, Scene, VGroup, config
from manim.animation.animation import prepare_animation
from manim.animation.transform import MoveToTarget

from .layout import region_bounds
from .models import LayoutTemplate


TEXT_CLASS_NAMES = {
    "Code", "DecimalNumber", "Integer", "MarkupText", "MathTex", "Paragraph",
    "SingleStringMathTex", "Tex", "Text", "Title", "Variable",
}

# A beat that starts later than this after its narration is reported, and so
# is a scene that runs longer than this past the end of the narration.
LATE_TOLERANCE_SECONDS = 0.25
# Gaps below one 60 fps frame cannot be rendered and are not worth a wait.
MIN_SYNC_GAP_SECONDS = 1.0 / 60.0
# Text about to appear across the frame edge is moved this far inside it
# (and shrunk only when it is wider or taller than the safe area). The
# verifier reports text within 0.21 of the edge, so this stays clear of it.
SAFE_MARGIN = 0.4
# Text is never shrunk below this to fit; the rest of the overflow is shifted.
MIN_FIT_SCALE = 0.6


def text_only(mobject) -> bool:
    """True when ``mobject`` is text or math, or a plain group of nothing else."""

    if TrackedScene._is_text_class(type(mobject).__name__):
        return True
    if type(mobject) in (VGroup, Group):
        children = list(mobject.submobjects)
        return bool(children) and all(text_only(child) for child in children)
    return False


def fit_into_frame(mobject, half_width: float, half_height: float) -> dict[str, float] | None:
    """Bring ``mobject`` inside the safe area, disturbing its placement as little as possible.

    Generated code places text relative to its neighbours (``next_to`` a line
    or a node), so the side facing inward is where the code put it. Text that
    runs over one edge is shrunk toward that inner side, keeping it beside
    what it labels; it is shifted only for what shrinking to MIN_FIT_SCALE
    cannot fix. Returns what was done, or None when it already fits, is
    staged off screen on purpose, or has no extent.
    """

    def edges():
        return (
            float(mobject.get_left()[0]), float(mobject.get_right()[0]),
            float(mobject.get_bottom()[1]), float(mobject.get_top()[1]),
        )

    try:
        left, right, bottom, top = edges()
    except Exception:
        return None
    width, height = right - left, top - bottom
    if width <= 0.0 and height <= 0.0:
        return None
    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    if abs(center_x) > half_width + SAFE_MARGIN or abs(center_y) > half_height + SAFE_MARGIN:
        return None  # placed off screen on purpose (to fly in later)
    if left >= -half_width and right <= half_width and bottom >= -half_height and top <= half_height:
        return None
    # The largest scale at which the text fits from its inner edge, per axis.
    needed = 1.0
    anchor = np.array([center_x, center_y, 0.0])
    if right > half_width and left >= -half_width and width > 0:
        needed = min(needed, (half_width - left) / width)
        anchor[0] = left
    elif left < -half_width and right <= half_width and width > 0:
        needed = min(needed, (right + half_width) / width)
        anchor[0] = right
    elif width > 2.0 * half_width:
        needed = min(needed, 2.0 * half_width / width)
    if top > half_height and bottom >= -half_height and height > 0:
        needed = min(needed, (half_height - bottom) / height)
        anchor[1] = bottom
    elif bottom < -half_height and top <= half_height and height > 0:
        needed = min(needed, (top + half_height) / height)
        anchor[1] = top
    elif height > 2.0 * half_height:
        needed = min(needed, 2.0 * half_height / height)
    scale = max(MIN_FIT_SCALE, needed) if needed < 1.0 else 1.0
    if scale < 1.0:
        mobject.scale(scale, about_point=anchor)
        left, right, bottom, top = edges()
    shift_x = max(0.0, -half_width - left) - max(0.0, right - half_width)
    shift_y = max(0.0, -half_height - bottom) - max(0.0, top - half_height)
    if shift_x or shift_y:
        mobject.shift(np.array([shift_x, shift_y, 0.0]))
    return {"scale": round(scale, 4), "shift_x": round(shift_x, 4), "shift_y": round(shift_y, 4)}


class TrackedScene(Scene):
    """Ordinary Manim Scene that records artifact state at animation boundaries.

    It also keeps the animation on the narration's clock: each beat waits for
    its scheduled start, the scene holds until the narration ends, and beats
    that start late are written to the trace instead of being sped up.
    """

    def setup(self) -> None:
        super().setup()
        self._artifact_registry: dict[int, dict[str, Any]] = {}
        self._layout_template = LayoutTemplate.CENTER_WITH_NOTES
        self._current_beat = ""
        self._beat_schedule = self._schedule_from_environment()
        self._target_duration = self._schedule_duration(self._beat_schedule)
        self._synced_beats: set[str] = set()
        self._sync_wait_active = False
        self._narration_sync_finished = False
        self._auto_counter = 0
        self._checkpoint_index = 0
        self._animation_depth = 0
        self._play_start_time = 0.0
        self._play_sample_targets: list[float] = []
        self._frame_fits: list[dict[str, Any]] = []
        trace = os.getenv("PDF2MANIM_TRACE_PATH")
        self._trace_path = Path(trace).resolve() if trace else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_path.write_text("", encoding="utf-8")

    def set_layout(self, template: str) -> None:
        self._layout_template = LayoutTemplate(template)

    def configure_generated_scene(
        self,
        *,
        layout_template: str,
        target_duration: float,
        beat_schedule: list[tuple[str, float, float]],
        authored_timing: list[str | None] | None = None,
    ) -> None:
        """Receive deterministic metadata injected after model generation.

        ``authored_timing`` is a record for deinstrument_scene_code and is not
        used while rendering.
        """

        self.set_layout(layout_template)
        self._target_duration = max(0.0, float(target_duration))
        self._beat_schedule = [
            (str(beat_id), float(start), float(end))
            for beat_id, start, end in beat_schedule
        ]

    def auto_register(self, name: str, value):
        """Register assigned mobjects without requiring model-written bookkeeping."""

        if not isinstance(value, Mobject):
            return value
        if id(value) in self._artifact_registry:
            return value
        beat_ids = [item[0] for item in self._beat_schedule] or [self._current_beat or "scene"]
        class_name = type(value).__name__
        category = "semantic_text" if self._is_text_class(class_name) else "generated_visual"
        base = str(name).strip() or "object"
        artifact_id = base
        used = {record["artifact_id"] for record in self._artifact_registry.values()}
        suffix = 2
        while artifact_id in used:
            artifact_id = f"{base}_{suffix}"
            suffix += 1
        self._artifact_registry[id(value)] = {
            "artifact_id": artifact_id,
            "category": category,
            "purpose": f"Generated {class_name}",
            "beat_ids": beat_ids,
            "mobject": value,
        }
        return value

    def place_in_region(
        self,
        mobject,
        region: str,
        *,
        padding: float = 0.12,
        minimum_scale: float = 0.55,
        overflow_policy: str = "scale",
        anchor: str = "center",
    ):
        """Fit and anchor a mobject inside one contract-owned layout region."""
        box = region_bounds(self._layout_template, region).inset(float(padding))
        width = max(float(getattr(mobject, "width", 0.0)), 1e-9)
        height = max(float(getattr(mobject, "height", 0.0)), 1e-9)
        scale = min(1.0, box.width / width, box.height / height)
        if scale < 1.0 and overflow_policy == "reject":
            raise ValueError(f"{type(mobject).__name__} does not fit region {region!r}")
        if scale < float(minimum_scale):
            raise ValueError(
                f"{type(mobject).__name__} needs scale {scale:.3f} in region {region!r}, "
                f"below minimum {minimum_scale:.3f}"
            )
        if scale < 1.0:
            mobject.scale(scale)

        half_width = float(getattr(mobject, "width", 0.0)) / 2
        half_height = float(getattr(mobject, "height", 0.0)) / 2
        x = (box.left + box.right) / 2
        y = (box.bottom + box.top) / 2
        if "left" in anchor:
            x = box.left + half_width
        elif "right" in anchor:
            x = box.right - half_width
        if "top" in anchor:
            y = box.top - half_height
        elif "bottom" in anchor:
            y = box.bottom + half_height
        mobject.move_to(np.array([x, y, 0.0]))
        return mobject

    def register_artifact(
        self,
        artifact_id: str,
        mobject,
        *,
        category: str,
        purpose: str,
        beat_ids: list[str],
    ):
        if not artifact_id or not purpose or not beat_ids:
            raise ValueError("Tracked artifacts require id, purpose, and beat_ids")
        self._artifact_registry[id(mobject)] = {
            "artifact_id": artifact_id,
            "category": category,
            "purpose": purpose,
            "beat_ids": list(beat_ids),
            "mobject": mobject,
        }
        return mobject

    def set_beat(self, beat_id: str) -> None:
        sync = self._sync_to_beat(beat_id)
        self._current_beat = beat_id
        if sync:
            self._checkpoint(sync[0], extra={"sync": sync[1]})
        self._checkpoint("beat")

    def finish_narration_sync(self) -> float:
        """Hold until the narration ends; returns the seconds waited.

        Injected as the last statement of construct and called again from
        tear_down, so it runs once and never raises: a scene that renders
        must not fail because of its closing hold.
        """

        if getattr(self, "_narration_sync_finished", True):
            return 0.0
        self._narration_sync_finished = True
        try:
            end = max(self._target_duration, self._schedule_duration(self._beat_schedule))
            if end <= 0.0:
                return 0.0
            waited = self._hold_until(end, round_up=True)
            now = self._now()
            sync = {"beat_id": "", "scheduled_seconds": round(end, 4),
                    "actual_seconds": round(now, 4)}
            if waited:
                self._checkpoint(
                    "sync_wait", extra={"sync": {**sync, "waited_seconds": round(waited, 4)}},
                )
            elif now - end > LATE_TOLERANCE_SECONDS:
                self._checkpoint(
                    "scene_overrun", extra={"sync": {**sync, "lag_seconds": round(now - end, 4)}},
                )
            return waited
        except Exception:
            return 0.0

    def _sync_to_beat(self, beat_id: str) -> tuple[str, dict[str, Any]] | None:
        """Wait for a beat's scheduled start; returns the trace record to write."""

        start = next(
            (start for known, start, _ in self._beat_schedule if known == beat_id), None,
        )
        # A beat is synced the first time it is entered: a marker repeated in a
        # loop must not report the same beat as late over and over.
        if start is None or beat_id in self._synced_beats:
            return None
        self._synced_beats.add(beat_id)
        waited = self._hold_until(start)
        now = self._now()
        sync = {"beat_id": beat_id, "scheduled_seconds": round(start, 4),
                "actual_seconds": round(now, 4)}
        if waited:
            return "sync_wait", {**sync, "waited_seconds": round(waited, 4)}
        if now - start > LATE_TOLERANCE_SECONDS:
            return "beat_late", {**sync, "lag_seconds": round(now - start, 4)}
        return None

    def _hold_until(self, target: float, *, round_up: bool = False) -> float:
        """Wait on the scene clock until ``target``; returns the seconds waited."""

        gap = float(target) - self._now()
        if gap < MIN_SYNC_GAP_SECONDS:
            return 0.0
        # Manim renders whole frames, so ask for whole frames. Beat starts take
        # the nearest frame; the closing hold rounds up so the video is never
        # shorter than the narration.
        frame_rate = float(config.frame_rate)
        frames = math.ceil(gap * frame_rate - 1e-6) if round_up else round(gap * frame_rate)
        if frames < 1:
            return 0.0
        before = self._now()
        # Scene.wait re-enters play(); its checkpoints would look like the
        # model's own animation, so they are suppressed while the hold runs.
        self._sync_wait_active = True
        try:
            super().wait(frames / frame_rate)
        finally:
            self._sync_wait_active = False
        return self._now() - before

    def _now(self) -> float:
        return float(getattr(getattr(self, "renderer", None), "time", 0.0) or 0.0)

    def next_section(self, name: str, *args, **kwargs):
        """Use standard Manim section markers as narration beat markers."""

        self.set_beat(name)
        return super().next_section(name, *args, **kwargs)

    def _keep_text_in_frame(self, mobjects) -> None:
        """Bring text that is about to appear inside the frame before it is shown.

        Only text that is not on screen yet, has no updaters and sits at least
        partly in view is moved: moving geometry, or anything already visible,
        would leave arrows and neighbours pointing at its old place.
        """

        try:
            on_screen = {id(member) for member in self.get_mobject_family_members()}
        except Exception:
            on_screen = set()
        half_width = float(config.frame_width) / 2.0 - SAFE_MARGIN
        half_height = float(config.frame_height) / 2.0 - SAFE_MARGIN
        mobjects = [m for m in mobjects if isinstance(m, Mobject)]
        # A piece of another object being shown moves with that object, never alone.
        mobjects = [
            m for m in mobjects
            if not any(other is not m and m in other.get_family() for other in mobjects)
        ]
        fitted: set[int] = set()
        for piece in mobjects:
            if id(piece) in on_screen:
                continue
            # Code often writes one formula in pieces (formula[:3], then
            # formula[3:]); fit the whole named formula once, never a piece.
            mobject = self._text_owner(piece)
            if id(mobject) in fitted:
                continue
            fitted.add(id(mobject))
            try:
                family = mobject.get_family()
                if any(id(member) in on_screen for member in family):
                    continue  # part of it is already visible: moving it would jump
                if any(member.get_updaters() for member in family):
                    continue
            except Exception:
                continue
            if not text_only(mobject):
                continue
            fit = fit_into_frame(mobject, half_width, half_height)
            if fit is not None:
                record = self._artifact_record(id(mobject), mobject)
                self._frame_fits.append({
                    "object_id": record["artifact_id"] if record else type(mobject).__name__,
                    **fit,
                })

    def _text_owner(self, mobject):
        """The largest registered text object that ``mobject`` is part of (or itself)."""

        # A slice such as formula[:3] is a new group holding the formula's own
        # pieces, so match on the pieces, not on the group object.
        pieces = {id(member) for member in mobject.get_family() if member is not mobject}
        pieces = pieces or {id(mobject)}
        owner, size = mobject, len(mobject.get_family())
        for record in self._artifact_registry.values():
            candidate = record["mobject"]
            try:
                family = candidate.get_family()
            except Exception:
                continue
            members = {id(member) for member in family}
            if len(family) > size and pieces <= members and text_only(candidate):
                owner, size = candidate, len(family)
        return owner

    def _entering_mobjects(self, animation) -> list:
        """Objects an animation will show that may not be on screen yet."""

        whole = getattr(animation, "to_add", None)
        if isinstance(whole, Mobject):
            return [whole]  # TransformMatchingTex/Shapes: fit the whole new formula, never a piece
        found = []
        for child in getattr(animation, "animations", None) or []:
            found.extend(self._entering_mobjects(child))
        if isinstance(animation, MoveToTarget):
            return found  # a move of an object already shown: left to the prompt and repair
        mobject = getattr(animation, "mobject", None)
        if mobject is not None:
            found.append(mobject)
        target = getattr(animation, "target_mobject", None)
        if target is not None and target is not mobject:
            found.append(target)
        return found

    def add(self, *mobjects):
        self._keep_text_in_frame(mobjects)
        for mobject in mobjects:
            if self._artifact_record(id(mobject), mobject) is None:
                self._auto_counter += 1
                self.auto_register(f"auto_{self._auto_counter:03d}", mobject)
        result = super().add(*mobjects)
        if not self._animation_depth:
            self._checkpoint("add")
        return result

    def remove(self, *mobjects):
        result = super().remove(*mobjects)
        if not self._animation_depth:
            self._checkpoint("remove")
        return result

    def play(self, *args, **kwargs):
        try:
            args = tuple(prepare_animation(arg) for arg in args)
        except Exception:
            pass  # let Manim report the bad argument itself
        else:
            entering = []
            for animation in args:
                entering.extend(self._entering_mobjects(animation))
            self._keep_text_in_frame(entering)
        self._play_start_time = float(getattr(getattr(self, "renderer", None), "time", 0.0))
        self._play_sample_targets = [0.25, 0.50, 0.75]
        self._animation_depth += 1
        try:
            result = super().play(*args, **kwargs)
        finally:
            self._animation_depth -= 1
            self._play_sample_targets = []
        self._checkpoint("play")
        return result

    def update_to_time(self, t: float) -> None:
        super().update_to_time(t)
        duration = float(getattr(self, "duration", 0.0) or 0.0)
        if not self._animation_depth or duration <= 0.0:
            return
        progress = min(1.0, max(0.0, float(t) / duration))
        while self._play_sample_targets and progress >= self._play_sample_targets[0]:
            self._play_sample_targets.pop(0)
            self._checkpoint("play_sample", timestamp=self._play_start_time + float(t))

    def wait(self, duration=1.0, stop_condition=None, frozen_frame=None):
        result = super().wait(duration, stop_condition, frozen_frame)
        self._checkpoint("wait")
        return result

    def tear_down(self) -> None:
        # Covers code instrumented before the sync contract and constructs
        # that return early; a no-op when construct already finished the hold.
        self.finish_narration_sync()
        self._checkpoint("final")
        super().tear_down()

    def _checkpoint(
        self,
        event: str,
        *,
        timestamp: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._trace_path or not hasattr(self, "renderer") or self._sync_wait_active:
            return
        resolved_time = (
            float(getattr(self.renderer, "time", 0.0)) if timestamp is None else timestamp
        )
        beat_id = self._current_beat or self._beat_at(resolved_time)
        traced_objects = self._trace_objects(getattr(self, "mobjects", []))
        objects = [{key: value for key, value in item.items() if key != "_mobject"}
                   for item in traced_objects]
        payload = {
            "checkpoint": self._checkpoint_index,
            "event": event,
            "time": round(resolved_time, 4),
            "beat_id": beat_id,
            "objects": objects,
            "pixel_overlaps": self._pixel_overlap_records(traced_objects),
        }
        if extra:
            payload.update(extra)
        if self._frame_fits:
            payload["frame_fits"] = self._frame_fits
            self._frame_fits = []
        self._checkpoint_index += 1
        try:
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            pass

    def _beat_at(self, timestamp: float) -> str:
        for beat_id, start, end in self._beat_schedule:
            if start <= timestamp <= end + 1e-6:
                return beat_id
        return self._beat_schedule[-1][0] if self._beat_schedule else ""

    @staticmethod
    def _schedule_from_environment() -> list[tuple[str, float, float]]:
        raw = os.getenv("PDF2MANIM_BEAT_SCHEDULE", "")
        if not raw:
            return []
        try:
            values = json.loads(raw)
            return [
                (str(item["beat_id"]), float(item["start"]), float(item["end"]))
                for item in values
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return []

    @staticmethod
    def _schedule_duration(schedule: list[tuple[str, float, float]]) -> float:
        return max((end for _, _, end in schedule), default=0.0)

    def _trace_objects(self, mobjects) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        for top_index, top_level in enumerate(mobjects):
            for path, mobject in self._trace_members(top_level):
                record = self._artifact_record(id(mobject), mobject)
                artifact_id = record["artifact_id"] if record else None
                exact_registration = id(mobject) in self._artifact_registry
                object_id = artifact_id
                if artifact_id and path and not exact_registration:
                    object_id = f"{artifact_id}.{path}"
                class_name = type(mobject).__name__
                category = record["category"] if record else "unregistered"
                if self._is_text_class(class_name):
                    category = "semantic_text"
                objects.append({
                    "artifact_id": artifact_id,
                    "object_id": object_id,
                    "runtime_id": id(mobject),
                    "trace_path": path or str(top_index),
                    "category": category,
                    "purpose": record["purpose"] if record else "",
                    "beat_ids": record["beat_ids"] if record else [],
                    "class_name": class_name,
                    "font_size": self._number(getattr(mobject, "font_size", None)),
                    "bounds": self._bounds(mobject),
                    "_mobject": mobject,
                })
        return objects

    @classmethod
    def _trace_members(cls, mobject, path: str = ""):
        if isinstance(mobject, (Group, VGroup)) and getattr(mobject, "submobjects", None):
            for index, child in enumerate(mobject.submobjects):
                child_path = f"{path}.{index}" if path else str(index)
                yield from cls._trace_members(child, child_path)
            return
        yield path, mobject

    def _pixel_overlap_records(self, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for index, left in enumerate(objects):
            for right in objects[index + 1:]:
                if not left.get("object_id") or not right.get("object_id"):
                    continue
                left_text = self._is_text_class(left.get("class_name", ""))
                right_text = self._is_text_class(right.get("class_name", ""))
                if not (left_text or right_text):
                    continue
                if left.get("artifact_id") == right.get("artifact_id") and left_text != right_text:
                    continue
                if not self._bounds_intersect(left.get("bounds"), right.get("bounds")):
                    continue
                candidates.append((left, right))
        if not candidates:
            return []
        try:
            camera = Camera(
                pixel_width=240,
                pixel_height=135,
                frame_width=float(config.frame_width),
                frame_height=float(config.frame_height),
                background_opacity=0,
            )
        except Exception:
            return []
        masks: dict[int, np.ndarray | None] = {}
        results = []
        for left, right in candidates:
            left_key = id(left["_mobject"])
            right_key = id(right["_mobject"])
            if left_key not in masks:
                masks[left_key] = self._pixel_mask(camera, left["_mobject"])
            if right_key not in masks:
                masks[right_key] = self._pixel_mask(camera, right["_mobject"])
            ratio = self._mask_overlap_ratio(masks[left_key], masks[right_key])
            if ratio is None:
                continue
            results.append({
                "left_object_id": left["object_id"],
                "right_object_id": right["object_id"],
                "ratio": round(ratio, 4),
            })
        return results

    @staticmethod
    def _pixel_mask(camera: Camera, mobject) -> np.ndarray | None:
        try:
            camera.reset()
            camera.capture_mobjects([mobject])
            mask = camera.pixel_array[:, :, 3] > 16
            if not mask.any():
                return None
            padded = np.pad(mask, 1, mode="constant")
            height, width = mask.shape
            return np.logical_or.reduce([
                padded[y:y + height, x:x + width]
                for y in range(3) for x in range(3)
            ])
        except Exception:
            return None

    @staticmethod
    def _mask_overlap_ratio(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
        if left is None or right is None or left.shape != right.shape:
            return None
        smaller_area = min(int(left.sum()), int(right.sum()))
        if not smaller_area:
            return None
        return float(np.logical_and(left, right).sum()) / smaller_area

    @staticmethod
    def _bounds_intersect(left, right) -> bool:
        if not left or not right:
            return False
        return (
            min(left["right"], right["right"]) - max(left["left"], right["left"]) > 0.01
            and min(left["top"], right["top"]) - max(left["bottom"], right["bottom"]) > 0.01
        )

    @staticmethod
    def _is_text_class(class_name: str) -> bool:
        return class_name in TEXT_CLASS_NAMES or class_name.endswith(("Text", "Tex"))

    def _artifact_record(self, object_id: int, mobject) -> dict[str, Any] | None:
        record = self._artifact_registry.get(object_id)
        if record:
            return record
        for candidate in self._artifact_registry.values():
            registered = candidate["mobject"]
            try:
                if any(member is mobject for member in registered.get_family()):
                    return candidate
            except Exception:
                continue
        return None

    @classmethod
    def _bounds(cls, mobject) -> dict[str, float] | None:
        try:
            left = mobject.get_left()
            right = mobject.get_right()
            top = mobject.get_top()
            bottom = mobject.get_bottom()
            return {
                "left": float(left[0]), "right": float(right[0]),
                "top": float(top[1]), "bottom": float(bottom[1]),
                "width": float(right[0] - left[0]),
                "height": float(top[1] - bottom[1]),
            }
        except Exception:
            return None

    @staticmethod
    def _number(value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
