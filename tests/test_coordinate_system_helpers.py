import pytest
manim = pytest.importorskip("manim")

from src.utils.coordinate_system_helpers import (  # noqa: E402
    attach_axis_labels,
    create_coordinate_system,
    plot_function_on_system,
)


def test_create_coordinate_system_supports_axes_and_number_plane():
    axes = create_coordinate_system(
        kind="axes",
        x_range=(-4, 4, 1),
        y_range=(-2, 2, 1),
        x_length=5,
        y_length=3,
    )
    plane = create_coordinate_system(
        kind="number_plane",
        x_range=(-4, 4, 1),
        y_range=(-2, 2, 1),
        x_length=5,
        y_length=3,
    )
    assert axes.__class__.__name__ == "Axes"
    assert plane.__class__.__name__ == "NumberPlane"


def test_attach_axis_labels_returns_group_with_labels():
    axes = create_coordinate_system(kind="axes")
    labelled = attach_axis_labels(axes, x_label="x", y_label="y")
    assert labelled.__class__.__name__ == "VGroup"
    assert len(labelled.submobjects) == 3


def test_plot_function_on_system_returns_curve():
    axes = create_coordinate_system(kind="axes")
    curve = plot_function_on_system(axes, lambda x: x**2, color="YELLOW")
    assert curve is not None
