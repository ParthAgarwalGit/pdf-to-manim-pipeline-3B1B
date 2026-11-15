"""Geometric anchoring helpers for generated Manim scenes.

A real run showed the code LLM computing arrow endpoints by mental arithmetic
(``Arrow(start=[2.5, -1.7, 0], end=[3.0, -1.58, 0])``) — connections started
at the right object and ended midway or in the wrong direction, and
"perpendicular distance" visuals never actually met the line. These helpers
make the correct construction the easy one: endpoints come from live mobject
geometry and projections are computed, never eyeballed.

Available inside generated scene code (imported by the animation generator):

    arrow = connect(dot, plane_label)                  # object-to-object arrow
    p = on_axes(axes, 2.0, 1.5)                        # data coords -> scene point
    foot = perpendicular_foot(p, line_start, line_end) # true projection
    drop = drop_perpendicular(dot, line)               # dashed drop + right-angle tick
"""

import numpy as np
from manim import (
    Arrow,
    DashedLine,
    Line,
    VGroup,
    WHITE,
    YELLOW,
)


def _as_point(target) -> np.ndarray:
    """Accept a mobject, a 2/3-sequence, or an ndarray; return a 3D point."""
    if hasattr(target, "get_center"):
        return np.array(target.get_center(), dtype=float)
    arr = np.array(target, dtype=float).flatten()
    if arr.shape[0] == 2:
        arr = np.append(arr, 0.0)
    return arr[:3]


def connect(a, b, buff: float = 0.15, color=YELLOW, stroke_width: float = 4.0) -> Arrow:
    """Arrow from mobject/point ``a`` to mobject/point ``b``.

    Endpoints are taken from the LIVE positions of the two anchors, so the
    arrow always starts and ends exactly where the objects are — never at a
    hand-computed coordinate.
    """
    start, end = _as_point(a), _as_point(b)
    return Arrow(start=start, end=end, buff=buff, color=color, stroke_width=stroke_width)


def on_axes(axes, x: float, y: float) -> np.ndarray:
    """Scene-space point for data coordinates ``(x, y)`` on ``axes``.

    Thin, name-stable wrapper over ``axes.coords_to_point`` so prompts can
    reference one canonical helper.
    """
    return np.array(axes.coords_to_point(x, y), dtype=float)


def perpendicular_foot(point, line_start, line_end) -> np.ndarray:
    """Orthogonal projection of ``point`` onto the line through the two ends.

    Pure vector math: foot = s + ((p-s)·d / d·d) d. A degenerate (zero-length)
    line returns ``line_start``. All inputs may be mobjects or points.
    """
    p = _as_point(point)
    s = _as_point(line_start)
    e = _as_point(line_end)
    d = e - s
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        return s
    t = float(np.dot(p - s, d)) / denom
    return s + t * d


def drop_perpendicular(
    point,
    line,
    color=WHITE,
    tick_size: float = 0.18,
) -> VGroup:
    """Dashed perpendicular from ``point`` to ``line`` with a right-angle tick.

    The classic "distance from a point to a line/plane" visual, constructed
    from the actual projection so it always lands ON the line at 90°.
    Returns ``VGroup(dashed_drop, right_angle_tick)``; the foot itself is
    ``group[0].get_end()``.
    """
    p = _as_point(point)
    if hasattr(line, "get_start") and hasattr(line, "get_end"):
        s, e = np.array(line.get_start(), float), np.array(line.get_end(), float)
    else:
        s, e = _as_point(line[0]), _as_point(line[1])
    foot = perpendicular_foot(p, s, e)

    drop = DashedLine(start=p, end=foot, color=color)

    # Right-angle tick: two short segments hugging the corner at the foot,
    # one along the line direction, one along the drop direction.
    line_dir = e - s
    line_norm = float(np.linalg.norm(line_dir))
    drop_dir = p - foot
    drop_norm = float(np.linalg.norm(drop_dir))
    if line_norm < 1e-9 or drop_norm < 1e-9:
        return VGroup(drop)  # degenerate: point on the line — no tick
    u = line_dir / line_norm
    v = drop_dir / drop_norm
    corner_a = foot + u * tick_size
    corner_b = foot + u * tick_size + v * tick_size
    corner_c = foot + v * tick_size
    tick = VGroup(
        Line(corner_a, corner_b, color=color, stroke_width=2),
        Line(corner_b, corner_c, color=color, stroke_width=2),
    )
    return VGroup(drop, tick)
