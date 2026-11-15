"""Runtime layout tracker for Manim scenes — prevents text overlap and keeps
labels anchored to their parent shapes.

Usage in generated scene code:

    from src.utils.layout_tracker import LayoutTracker

    layout = LayoutTracker()

    # Register every object after creation + placement
    triangle = Polygon(...)
    layout.register("triangle", triangle)

    # Place a label near a registered object — the tracker finds a
    # non-overlapping position automatically
    opp_label = Text("opposite = 6m", font_size=24, color=GREEN)
    layout.place_label(opp_label, "opp_label", anchor="triangle",
                       preferred=RIGHT, buff=0.3)

    # For angle arcs that MUST stay at a vertex:
    arc = Arc(...)
    arc.move_to(triangle.get_vertices()[0])
    layout.register("arc", arc)
    theta = MathTex(r"\\theta")
    layout.place_label(theta, "theta", anchor="arc",
                       preferred=RIGHT, buff=0.2)

    # When you FadeOut an object, unregister it so its space is freed:
    layout.unregister("triangle")

The tracker stores each object's bounding-box center and half-extents.
When placing a label it tries the preferred direction first, then cycles
through alternatives (RIGHT, UP, LEFT, DOWN, UR, UL, DR, DL) until it
finds a spot that doesn't overlap any registered object.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

# Manim imports — these are available at runtime in generated scene code
from manim import Mobject, UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL


# Direction priority cycle — preferred direction is tried first, then these
_DIRECTION_CYCLE = [RIGHT, UP, LEFT, DOWN, UR, UL, DR, DL]

# Screen boundaries (Manim default camera: ~14.2 x 8 units)
_SCREEN_HALF_W = 7.0
_SCREEN_HALF_H = 3.8  # leave margin for edges


class _BBox:
    """Axis-aligned bounding box stored as center + half-extents."""

    __slots__ = ("cx", "cy", "hw", "hh")

    def __init__(self, mob: Mobject):
        # Manim Mobjects have .get_center(), .width, .height
        c = mob.get_center()
        self.cx: float = float(c[0])
        self.cy: float = float(c[1])
        self.hw: float = float(mob.width / 2) if mob.width > 0 else 0.1
        self.hh: float = float(mob.height / 2) if mob.height > 0 else 0.1

    def overlaps(self, other: "_BBox", margin: float = 0.05) -> bool:
        """True if the two boxes overlap (with a small margin)."""
        return (
            abs(self.cx - other.cx) < (self.hw + other.hw + margin)
            and abs(self.cy - other.cy) < (self.hh + other.hh + margin)
        )

    def is_on_screen(self) -> bool:
        return (
            abs(self.cx) + self.hw <= _SCREEN_HALF_W
            and abs(self.cy) + self.hh <= _SCREEN_HALF_H
        )


class LayoutTracker:
    """Track placed Mobjects and find non-overlapping positions for labels."""

    def __init__(self):
        self._objects: Dict[str, _BBox] = {}
        self._mobjects: Dict[str, Mobject] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, mob: Mobject) -> None:
        """Register a Mobject so the tracker knows its position and size.

        Call this AFTER the object has been positioned (shift/move_to/etc).
        """
        self._objects[name] = _BBox(mob)
        self._mobjects[name] = mob

    def unregister(self, name: str) -> None:
        """Free the space occupied by *name* (call after FadeOut)."""
        self._objects.pop(name, None)
        self._mobjects.pop(name, None)

    def clear(self) -> None:
        """Remove all tracked objects (e.g. between scene segments)."""
        self._objects.clear()
        self._mobjects.clear()

    # ------------------------------------------------------------------
    # Label placement
    # ------------------------------------------------------------------

    def place_label(
        self,
        label: Mobject,
        name: str,
        anchor: str,
        preferred: np.ndarray = RIGHT,
        buff: float = 0.3,
        max_distance: float = 1.5,
    ) -> Mobject:
        """Position *label* near the registered object *anchor* without overlap.

        Tries *preferred* direction first, then cycles through alternatives.
        After placement the label is registered under *name*.

        Args:
            label:      The Mobject to place (Text, MathTex, etc.)
            name:       Registry key for this label
            anchor:     Name of the registered object to place the label near
            preferred:  Preferred direction (e.g. RIGHT, UP)
            buff:       Gap between anchor and label
            max_distance: Maximum distance from anchor center (labels should
                         stay close to what they annotate)

        Returns:
            The label Mobject (now positioned).  The caller can animate it.
        """
        if anchor not in self._mobjects:
            # Anchor unknown — just place at preferred direction from origin
            label.shift(preferred * buff)
            self.register(name, label)
            return label

        anchor_mob = self._mobjects[anchor]

        # Build direction list starting with preferred
        directions = [preferred]
        for d in _DIRECTION_CYCLE:
            if not np.allclose(d, preferred):
                directions.append(d)

        best_pos = None

        for d in directions:
            # Candidate position: next_to anchor in direction d
            label.next_to(anchor_mob, d, buff=buff)
            candidate = _BBox(label)

            # Check: still close enough to anchor?
            anchor_bb = self._objects[anchor]
            dist = np.sqrt(
                (candidate.cx - anchor_bb.cx) ** 2
                + (candidate.cy - anchor_bb.cy) ** 2
            )
            if dist > max_distance:
                continue

            # Check: on screen?
            if not candidate.is_on_screen():
                continue

            # Check: overlaps any registered object?
            has_overlap = False
            for existing_name, existing_bb in self._objects.items():
                if existing_name == name:
                    continue  # ignore self if re-placing
                if candidate.overlaps(existing_bb):
                    has_overlap = True
                    break

            if not has_overlap:
                best_pos = label.get_center().copy()
                break
        else:
            # All directions overlapped — use preferred anyway (least bad)
            label.next_to(anchor_mob, preferred, buff=buff)
            best_pos = label.get_center().copy()

        label.move_to(best_pos)
        self.register(name, label)
        return label

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def find_free_region(
        self,
        width: float,
        height: float,
        prefer_center: bool = True,
    ) -> np.ndarray:
        """Find a screen position where a box of *width* x *height* fits
        without overlapping any registered object.

        Returns a position as np.array([x, y, 0]).
        """
        hw = width / 2
        hh = height / 2

        # Grid search from center outward
        if prefer_center:
            candidates = [(0, 0)]
        else:
            candidates = []

        for y in np.arange(3.0, -3.5, -0.5):
            for x in np.arange(-5.0, 5.5, 0.5):
                candidates.append((x, y))

        for cx, cy in candidates:
            test = _BBox.__new__(_BBox)
            test.cx, test.cy, test.hw, test.hh = cx, cy, hw, hh
            if not test.is_on_screen():
                continue
            if not any(test.overlaps(bb) for bb in self._objects.values()):
                return np.array([cx, cy, 0])

        # Fallback: center of screen
        return np.array([0, 0, 0])

    def get_position(self, name: str) -> Optional[np.ndarray]:
        """Return the center of a registered object, or None."""
        if name in self._mobjects:
            return self._mobjects[name].get_center()
        return None

    def is_registered(self, name: str) -> bool:
        return name in self._objects
