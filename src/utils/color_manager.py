"""Color consistency manager for 3Blue1Brown-style videos.

Ensures one mathematical variable = one color throughout the entire video.
Uses the canonical 3B1B palette and provides helpers for tex_to_color_map
generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.utils.manim_engine_contract import COLOR_CYCLE, THREE_BLUE_ONE_BROWN_PALETTE


class ColorManager:
    """Track and enforce consistent variable-to-color assignments.

    Usage::

        cm = ColorManager()
        cm.assign("x", "#58C4DD")   # explicit
        cm.assign("y")              # auto-pick from cycle
        color_map = cm.build_tex_color_map(["x", "y"])
    """

    def __init__(
        self,
        initial_bindings: Optional[Dict[str, str]] = None,
        palette: Optional[List[str]] = None,
    ) -> None:
        self._bindings: Dict[str, str] = dict(initial_bindings or {})
        self._palette = list(palette or COLOR_CYCLE)
        self._next_idx = 0
        # Advance _next_idx past any colors already used
        used = set(self._bindings.values())
        while (
            self._next_idx < len(self._palette)
            and self._palette[self._next_idx] in used
        ):
            self._next_idx += 1

    # ----- public API -----

    def assign(self, variable: str, color: Optional[str] = None) -> str:
        """Assign a color to a variable.

        If *color* is ``None``, the next unused color from the cycle is
        picked automatically.  If the variable already has a color and
        *color* differs, the new color is **ignored** to preserve
        consistency (first assignment wins).

        Returns the effective color for the variable.
        """
        if variable in self._bindings:
            return self._bindings[variable]

        if color is None:
            color = self._next_color()
        self._bindings[variable] = color
        return color

    def get(self, variable: str) -> Optional[str]:
        """Retrieve the assigned color for a variable, or ``None``."""
        return self._bindings.get(variable)

    def build_tex_color_map(self, variables: Optional[List[str]] = None) -> Dict[str, str]:
        """Build a ``tex_to_color_map`` dict for ``MathTex``.

        If *variables* is ``None``, returns bindings for all tracked
        variables.
        """
        if variables is None:
            return dict(self._bindings)
        return {v: self._bindings[v] for v in variables if v in self._bindings}

    def export_bindings(self) -> Dict[str, str]:
        """Export all current bindings (for propagation via SharedContext)."""
        return dict(self._bindings)

    @classmethod
    def from_context_bindings(cls, bindings: Dict[str, str]) -> "ColorManager":
        """Restore a ColorManager from SharedContext.color_bindings."""
        return cls(initial_bindings=bindings)

    # ----- internals -----

    def _next_color(self) -> str:
        if self._next_idx < len(self._palette):
            color = self._palette[self._next_idx]
            self._next_idx += 1
            return color
        # Wrap around if we exhaust the palette
        idx = (len(self._bindings)) % len(self._palette)
        return self._palette[idx]
