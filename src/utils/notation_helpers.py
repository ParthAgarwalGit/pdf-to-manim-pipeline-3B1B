"""Reusable helpers for stable MathTex notation objects in Manim scenes.

Notation helpers give every equation and symbolic expression a stable
``notation_key`` identity so the pipeline can reuse and transform them
deterministically across scenes and retries.

Usage inside generated scene code:

    from src.utils.notation_helpers import (
        create_notation,
        create_notation_transform_target,
        create_text_label,
    )

    eq = create_notation(tex=r"X = U\\Sigma V^T", notation_key="svd_core")
    eq2 = create_notation_transform_target(eq, tex=r"X = U\\Sigma V^T + E",
                                           notation_key="svd_core")
    self.play(ReplacementTransform(eq, eq2), run_time=1.5)
"""

from __future__ import annotations

from manim import MathTex, Text


def create_notation(
    tex: str,
    notation_key: str | None = None,
    color: str = "WHITE",
    substrings_to_isolate: list[str] | None = None,
    tex_to_color_map: dict[str, str] | None = None,
) -> MathTex:
    """Create a MathTex object with a stable notation_key attribute.

    Args:
        tex: LaTeX string for the expression.
        notation_key: Stable symbolic identity used for continuity tracking.
            Stored as ``mob.notation_key`` on the returned object.
        color: Base color for the expression (Manim color string).
        substrings_to_isolate: Sub-expressions to split into separate submobs
            so individual parts can be animated independently.
        tex_to_color_map: Map from sub-expression to color override.

    Returns:
        MathTex Mobject with ``notation_key`` attribute set.
    """
    mob = MathTex(
        tex,
        color=color,
        substrings_to_isolate=substrings_to_isolate or [],
        tex_to_color_map=tex_to_color_map or {},
        arg_separator="" if substrings_to_isolate else " ",
    )
    mob.notation_key = notation_key
    return mob


def create_notation_transform_target(
    source: MathTex,
    tex: str,
    notation_key: str | None = None,
    color: str = "WHITE",
    tex_to_color_map: dict[str, str] | None = None,
) -> MathTex:
    """Create a transform-target notation object centred on *source*.

    Calling ``ReplacementTransform(source, target)`` will morph one
    expression into the other while keeping the notation anchored at
    the same screen position.

    Args:
        source: The existing notation object to replace.
        tex: LaTeX string for the replacement expression.
        notation_key: Stable key for the target (defaults to source's key).
        color: Base color for the replacement.
        tex_to_color_map: Color overrides for the replacement.

    Returns:
        MathTex Mobject positioned at source's centre.
    """
    target = create_notation(
        tex=tex,
        notation_key=notation_key or getattr(source, "notation_key", None),
        color=color,
        tex_to_color_map=tex_to_color_map,
    )
    target.move_to(source.get_center())
    return target


def create_text_label(
    text: str,
    color: str = "WHITE",
    font_size: int = 28,
) -> Text:
    """Create a plain-text label with consistent styling.

    Args:
        text: Label string.
        color: Label color.
        font_size: Font size in points.

    Returns:
        Text Mobject.
    """
    return Text(text, color=color, font_size=font_size)
