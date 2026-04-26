"""Tests for notation_helpers — stable MathTex creation and transform targets."""

import pytest

manim = pytest.importorskip("manim")  # skip entire module when Manim not installed

from src.utils.notation_helpers import (
    create_notation,
    create_notation_transform_target,
    create_text_label,
)


def test_create_notation_returns_mathtex_with_notation_key():
    expr = create_notation(tex=r"x^2 + y^2 = r^2", notation_key="circle_eq")
    assert expr.__class__.__name__ == "MathTex"
    assert expr.notation_key == "circle_eq"


def test_create_notation_supports_isolated_substrings_and_color_map():
    expr = create_notation(
        tex=r"x' = \Sigma x",
        notation_key="projected_vector",
        substrings_to_isolate=[r"x'", r"\Sigma"],
        tex_to_color_map={r"\Sigma": "YELLOW"},
    )
    assert expr.tex_string == r"x' = \Sigma x"


def test_create_notation_key_none_by_default():
    expr = create_notation(tex=r"E = mc^2")
    assert expr.notation_key is None


def test_create_notation_transform_target_preserves_center():
    expr = create_notation(tex=r"x = Uy", notation_key="basis_equation")
    target = create_notation_transform_target(
        expr,
        tex=r"x = U\Sigma v",
        notation_key="basis_equation",
    )
    import numpy as np
    np.testing.assert_allclose(
        target.get_center(), expr.get_center(), atol=1e-6
    )


def test_create_notation_transform_target_inherits_source_key():
    expr = create_notation(tex=r"A", notation_key="mat_a")
    target = create_notation_transform_target(expr, tex=r"B")
    assert target.notation_key == "mat_a"


def test_create_text_label_returns_text_mobject():
    label = create_text_label("Hello", color="YELLOW", font_size=32)
    assert label.__class__.__name__ == "Text"
