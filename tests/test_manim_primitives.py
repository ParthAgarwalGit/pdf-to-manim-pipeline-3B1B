"""Tests for the closed Manim primitive catalog (Task 12)."""

import pytest
from pydantic import ValidationError

from src.schemas.storyboard import MathematicalObject, ObjectStyle
from src.utils.manim_primitives import (
    ANIMATION_CATALOG,
    MOBJECT_CATALOG,
    PrimitiveSpec,
    allowed_args_for,
    catalog_for_prompt,
)


class TestCatalog:
    def test_catalog_has_core_mobjects(self):
        for name in ("MathTex", "Tex", "Text", "Circle", "Square", "Rectangle",
                     "Polygon", "Line", "Arrow", "Arc", "Dot", "Axes",
                     "NumberPlane", "NumberLine", "create_pi_teacher"):
            assert name in MOBJECT_CATALOG, f"missing primitive: {name}"

    def test_catalog_has_dynamic_intuition_primitives(self):
        # The "visualization-first" additions: vectors, annotations and an
        # animatable number. These are advertised in CANONICAL_RENDER_HINTS and
        # must be enrollable as first-class, schema-validated primitives.
        for name in ("Vector", "DoubleArrow", "DashedLine", "Ellipse",
                     "Triangle", "RoundedRectangle", "DecimalNumber"):
            assert name in MOBJECT_CATALOG, f"missing dynamic primitive: {name}"

    def test_catalog_has_core_animations(self):
        for name in ("Write", "Create", "FadeIn", "FadeOut",
                     "Transform", "ReplacementTransform", "Indicate"):
            assert name in ANIMATION_CATALOG, f"missing animation: {name}"

    def test_catalog_entries_are_primitive_specs(self):
        for spec in MOBJECT_CATALOG.values():
            assert isinstance(spec, PrimitiveSpec)
            assert spec.kind in ("mobject", "helper")
        for spec in ANIMATION_CATALOG.values():
            assert isinstance(spec, PrimitiveSpec)
            assert spec.kind == "animation"

    def test_catalog_for_prompt_lists_mobjects_and_animations(self):
        rendered = catalog_for_prompt()
        assert "MOBJECTS" in rendered
        assert "ANIMATIONS" in rendered
        for name in ("MathTex", "Circle", "Axes", "Polygon", "create_pi_teacher",
                     "Write", "FadeIn", "Transform"):
            assert name in rendered

    def test_catalog_for_prompt_includes_arg_signatures(self):
        rendered = catalog_for_prompt()
        # Circle's signature should mention its radius default
        assert "radius:" in rendered
        # Polygon should carry its note about 3-tuple vertices
        assert "3-tuples" in rendered

    def test_allowed_args_for_known_primitive(self):
        args = allowed_args_for("Circle")
        assert "radius" in args
        # Circle accepts color
        assert "color" in args

    def test_allowed_args_for_color_excluded_when_spec_says_no(self):
        # Axes deliberately rejects color (its own colour is set per-axis)
        args = allowed_args_for("Axes")
        assert "color" not in args

    def test_allowed_args_for_unknown_primitive_returns_empty(self):
        assert allowed_args_for("NotARealPrimitive") == set()


class TestMathematicalObjectPrimitiveValidation:
    """The new model_validator hooks the schema into the catalog so the LLM
    can never sneak past with an unknown primitive or kwarg."""

    def _base(self, **overrides):
        defaults = dict(
            object_id="x",
            object_role="main_equation",
            object_type="geometric_shape",
            content="",
            style=ObjectStyle(),
        )
        defaults.update(overrides)
        return defaults

    def test_legacy_object_without_primitive_still_validates(self):
        """A storyboard from before the catalog must keep working."""
        obj = MathematicalObject(**self._base())
        assert obj.primitive is None
        assert obj.args == {}

    def test_known_primitive_with_known_args_validates(self):
        obj = MathematicalObject(**self._base(
            primitive="Circle",
            args={"radius": 2.5},
        ))
        assert obj.primitive == "Circle"
        assert obj.args == {"radius": 2.5}

    def test_known_primitive_with_color_arg_validates(self):
        # Color is permitted on primitives whose spec accepts_color is True.
        obj = MathematicalObject(**self._base(
            primitive="Circle",
            args={"radius": 2.0, "color": "BLUE"},
        ))
        assert obj.args == {"radius": 2.0, "color": "BLUE"}

    def test_unknown_primitive_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            MathematicalObject(**self._base(
                primitive="ImaginaryClass",
                args={},
            ))
        assert "Unknown primitive" in str(exc.value)

    def test_unknown_arg_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            MathematicalObject(**self._base(
                primitive="Circle",
                args={"diameter": 2.0},  # Circle takes 'radius', not 'diameter'
            ))
        assert "diameter" in str(exc.value)

    def test_color_rejected_on_primitive_that_does_not_accept_it(self):
        with pytest.raises(ValidationError) as exc:
            MathematicalObject(**self._base(
                primitive="Axes",
                args={"color": "RED"},
            ))
        assert "color" in str(exc.value)

    def test_dynamic_vector_primitive_validates(self):
        obj = MathematicalObject(**self._base(
            primitive="Vector",
            args={"direction": (2.0, 1.0, 0.0), "color": "BLUE"},
        ))
        assert obj.primitive == "Vector"
        assert obj.args["direction"] == (2.0, 1.0, 0.0)

    def test_decimal_number_args_validate(self):
        obj = MathematicalObject(**self._base(
            primitive="DecimalNumber",
            args={"number": 3.0, "num_decimal_places": 2},
        ))
        assert obj.args["num_decimal_places"] == 2
