"""Reusable helpers for constructing and using coordinate systems in Manim."""

from __future__ import annotations

from manim import Axes, ComplexPlane, MathTex, NumberLine, NumberPlane, ThreeDAxes, VGroup


def create_coordinate_system(
    kind: str = "axes",
    x_range=(-6, 6, 1),
    y_range=(-4, 4, 1),
    x_length: float = 5.5,
    y_length: float = 3.2,
):
    if kind == "number_plane":
        return NumberPlane(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
        )
    if kind == "complex_plane":
        return ComplexPlane(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
        )
    if kind == "number_line":
        return NumberLine(x_range=x_range, length=x_length)
    if kind == "three_d_axes":
        return ThreeDAxes(x_range=x_range, y_range=y_range)
    return Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length)


def attach_axis_labels(system, x_label: str = "x", y_label: str = "y"):
    labels = VGroup()
    if hasattr(system, "get_x_axis_label"):
        labels.add(system.get_x_axis_label(MathTex(x_label)))
    if hasattr(system, "get_y_axis_label"):
        labels.add(system.get_y_axis_label(MathTex(y_label)))
    return VGroup(system, *labels)


def plot_function_on_system(system, fn, color="YELLOW"):
    if hasattr(system, "plot"):
        return system.plot(fn, color=color)
    raise TypeError(f"{system.__class__.__name__} does not support plot(...)")
