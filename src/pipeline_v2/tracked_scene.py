"""Thin runtime instrumentation for raw Manim scenes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from manim import Camera, Group, Mobject, Scene, VGroup, config

from .layout import region_bounds
from .models import LayoutTemplate


TEXT_CLASS_NAMES = {
    "Code", "DecimalNumber", "Integer", "MarkupText", "MathTex", "Paragraph",
    "SingleStringMathTex", "Tex", "Text", "Title", "Variable",
}


class TrackedScene(Scene):
    """Ordinary Manim Scene that records artifact state at animation boundaries."""

    def setup(self) -> None:
        super().setup()
        self._artifact_registry: dict[int, dict[str, Any]] = {}
        self._layout_template = LayoutTemplate.CENTER_WITH_NOTES
        self._current_beat = ""
        self._beat_schedule = self._schedule_from_environment()
        self._target_duration = self._schedule_duration(self._beat_schedule)
        self._auto_counter = 0
        self._checkpoint_index = 0
        self._animation_depth = 0
        self._play_start_time = 0.0
        self._play_sample_targets: list[float] = []
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
    ) -> None:
        """Receive deterministic metadata injected after model generation."""

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
        self._current_beat = beat_id
        self._checkpoint("beat")

    def next_section(self, name: str, *args, **kwargs):
        """Use standard Manim section markers as narration beat markers."""

        self.set_beat(name)
        return super().next_section(name, *args, **kwargs)

    def add(self, *mobjects):
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
        self._checkpoint("final")
        super().tear_down()

    def _checkpoint(self, event: str, *, timestamp: float | None = None) -> None:
        if not self._trace_path or not hasattr(self, "renderer"):
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
