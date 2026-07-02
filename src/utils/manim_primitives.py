"""Closed catalog of Manim primitives the LLM is allowed to emit.

This module is the single source of truth for which Manim Mobjects and
Animations the visual planner is permitted to choose from. Anything
outside the catalog is rejected at schema validation time so the
animation generator never has to fall back to heuristic rescue paths.

Each entry declares:
  - ``name``     — Manim class / helper name (must match the import that
                   the animation generator already pulls in).
  - ``kind``     — semantic family: ``mobject``, ``animation``, or
                   ``helper``.
  - ``args``     — tuple of (arg_name, type_hint, default) tuples. The
                   default makes calls safe when the LLM omits an arg.
  - ``accepts_color`` — whether ``color=`` is a sensible kwarg.
  - ``notes``    — optional one-line author note (rendered into the prompt).

Adding a new primitive: append it to ``MOBJECT_CATALOG`` /
``ANIMATION_CATALOG`` with a sensible default so legacy intents stay
parseable.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class PrimitiveSpec:
    """One primitive the LLM may emit."""

    name: str
    kind: str  # "mobject" | "animation" | "helper"
    args: Tuple[Tuple[str, str, Any], ...]
    accepts_color: bool = True
    notes: str = ""


# ── Mobjects ─────────────────────────────────────────────────────────────
MOBJECT_CATALOG: Dict[str, PrimitiveSpec] = {
    # Text & math
    "MathTex": PrimitiveSpec(
        "MathTex", "mobject",
        (("tex", "str", ""),),
        True,
    ),
    "Tex": PrimitiveSpec(
        "Tex", "mobject",
        (("tex", "str", ""),),
        True,
    ),
    "Text": PrimitiveSpec(
        "Text", "mobject",
        (("text", "str", ""), ("font_size", "int", 36)),
        True,
    ),
    "DecimalNumber": PrimitiveSpec(
        "DecimalNumber", "mobject",
        (("number", "float", 0.0), ("num_decimal_places", "int", 2)),
        True,
        notes="An animatable number; pair with a ValueTracker for live counters.",
    ),
    # Geometry
    "Circle": PrimitiveSpec(
        "Circle", "mobject",
        (("radius", "float", 1.0),),
        True,
    ),
    "Square": PrimitiveSpec(
        "Square", "mobject",
        (("side_length", "float", 2.0),),
        True,
    ),
    "Rectangle": PrimitiveSpec(
        "Rectangle", "mobject",
        (("width", "float", 4.0), ("height", "float", 2.0)),
        True,
    ),
    "Polygon": PrimitiveSpec(
        "Polygon", "mobject",
        (("vertices", "list[tuple]", []),),
        True,
        notes="Vertices are 3-tuples (x, y, 0).",
    ),
    "Line": PrimitiveSpec(
        "Line", "mobject",
        (("start", "tuple", (-1.0, 0.0, 0.0)), ("end", "tuple", (1.0, 0.0, 0.0))),
        True,
    ),
    "Arrow": PrimitiveSpec(
        "Arrow", "mobject",
        (("start", "tuple", (-1.0, 0.0, 0.0)), ("end", "tuple", (1.0, 0.0, 0.0))),
        True,
    ),
    "Arc": PrimitiveSpec(
        "Arc", "mobject",
        (
            ("radius", "float", 1.0),
            ("start_angle", "float", 0.0),
            ("angle", "float", 1.5708),
        ),
        True,
    ),
    "Dot": PrimitiveSpec(
        "Dot", "mobject",
        (("point", "tuple", (0.0, 0.0, 0.0)),),
        True,
    ),
    # Vectors & annotations — the "intuition-builder" primitives. These are
    # already advertised in CANONICAL_RENDER_HINTS; enrolling them here lets
    # the visual planner emit them as first-class, schema-validated objects
    # instead of collapsing everything to a static Line/Circle.
    "Vector": PrimitiveSpec(
        "Vector", "mobject",
        (("direction", "tuple", (1.0, 0.0, 0.0)),),
        True,
        notes="direction is a 3-tuple (x, y, 0) from the origin.",
    ),
    "DoubleArrow": PrimitiveSpec(
        "DoubleArrow", "mobject",
        (("start", "tuple", (-1.0, 0.0, 0.0)), ("end", "tuple", (1.0, 0.0, 0.0))),
        True,
    ),
    "DashedLine": PrimitiveSpec(
        "DashedLine", "mobject",
        (("start", "tuple", (-1.0, 0.0, 0.0)), ("end", "tuple", (1.0, 0.0, 0.0))),
        True,
    ),
    "Ellipse": PrimitiveSpec(
        "Ellipse", "mobject",
        (("width", "float", 2.0), ("height", "float", 1.0)),
        True,
    ),
    "Triangle": PrimitiveSpec(
        "Triangle", "mobject",
        (),
        True,
    ),
    "RoundedRectangle": PrimitiveSpec(
        "RoundedRectangle", "mobject",
        (("width", "float", 4.0), ("height", "float", 2.0), ("corner_radius", "float", 0.5)),
        True,
    ),
    # Coordinate systems
    "Axes": PrimitiveSpec(
        "Axes", "mobject",
        (
            ("x_range", "list", [-5, 5, 1]),
            ("y_range", "list", [-3, 3, 1]),
            ("x_length", "float", 5.0),
            ("y_length", "float", 3.0),
        ),
        False,
    ),
    "NumberPlane": PrimitiveSpec(
        "NumberPlane", "mobject",
        (),
        False,
    ),
    "NumberLine": PrimitiveSpec(
        "NumberLine", "mobject",
        (("x_range", "list", [-5, 5, 1]),),
        False,
    ),
    # Project-specific helpers (already imported by animation_generator)
    "create_pi_teacher": PrimitiveSpec(
        "create_pi_teacher", "helper",
        (),
        False,
    ),
    "create_pi_students": PrimitiveSpec(
        "create_pi_students", "helper",
        (),
        False,
    ),
    "create_speech_bubble": PrimitiveSpec(
        "create_speech_bubble", "helper",
        (("text", "str", ""), ("anchor_var", "str", "")),
        False,
        notes="anchor_var is the variable name of the registered creature.",
    ),
}


# ── Animations ───────────────────────────────────────────────────────────
ANIMATION_CATALOG: Dict[str, PrimitiveSpec] = {
    "Write": PrimitiveSpec(
        "Write", "animation",
        (("target", "var", ""), ("run_time", "float", 1.5)),
        False,
    ),
    "Create": PrimitiveSpec(
        "Create", "animation",
        (("target", "var", ""), ("run_time", "float", 1.5)),
        False,
    ),
    "FadeIn": PrimitiveSpec(
        "FadeIn", "animation",
        (("target", "var", ""), ("run_time", "float", 1.0)),
        False,
    ),
    "FadeOut": PrimitiveSpec(
        "FadeOut", "animation",
        (("target", "var", ""), ("run_time", "float", 1.0)),
        False,
    ),
    "Transform": PrimitiveSpec(
        "Transform", "animation",
        (("source", "var", ""), ("target", "var", ""), ("run_time", "float", 1.5)),
        False,
    ),
    "ReplacementTransform": PrimitiveSpec(
        "ReplacementTransform", "animation",
        (("source", "var", ""), ("target", "var", ""), ("run_time", "float", 1.5)),
        False,
    ),
    "Indicate": PrimitiveSpec(
        "Indicate", "animation",
        (("target", "var", ""), ("run_time", "float", 1.0)),
        False,
    ),
    "MoveToTarget": PrimitiveSpec(
        "MoveToTarget", "animation",
        (("target", "var", ""), ("run_time", "float", 1.5)),
        False,
    ),
}


def _format_default(value: Any) -> str:
    """Render a default into the compact form used by the prompt."""
    if isinstance(value, str):
        return f"'{value}'"
    return repr(value)


def catalog_for_prompt() -> str:
    """Render the catalog as a compact reference block for LLM prompts.

    The shape is deliberately terse — one line per primitive — so it stays
    cheap to include in every chunk's visual-planner call without dwarfing
    the rest of the prompt.
    """
    lines = ["MOBJECTS (use exactly these — any other name will be rejected):"]
    for spec in MOBJECT_CATALOG.values():
        sig = ", ".join(f"{a}: {t}={_format_default(d)}" for a, t, d in spec.args)
        suffix = f"  # {spec.notes}" if spec.notes else ""
        lines.append(f"  {spec.name}({sig}){suffix}")
    lines.append("")
    lines.append("ANIMATIONS:")
    for spec in ANIMATION_CATALOG.values():
        sig = ", ".join(f"{a}: {t}={_format_default(d)}" for a, t, d in spec.args)
        lines.append(f"  {spec.name}({sig})")
    return "\n".join(lines)


def allowed_args_for(primitive: str) -> set:
    """Return the set of legal kwarg names for a primitive (color included
    when the spec accepts it). Unknown primitive => empty set."""
    spec = MOBJECT_CATALOG.get(primitive)
    if spec is None:
        return set()
    allowed = {a for a, _, _ in spec.args}
    if spec.accepts_color:
        allowed.add("color")
    return allowed
