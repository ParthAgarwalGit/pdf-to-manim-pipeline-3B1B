"""Deterministic canvas regions shared by planning, generation, and verification."""

from __future__ import annotations

from dataclasses import dataclass

from .models import LayoutRegion, LayoutTemplate


@dataclass(frozen=True)
class RegionBounds:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def center(self) -> tuple[float, float, float]:
        return ((self.left + self.right) / 2, (self.bottom + self.top) / 2, 0.0)

    def inset(self, value: float) -> "RegionBounds":
        if value * 2 >= min(self.width, self.height):
            raise ValueError("Region padding leaves no usable area")
        return RegionBounds(
            self.left + value,
            self.right - value,
            self.bottom + value,
            self.top - value,
        )

    def expand(self, value: float) -> "RegionBounds":
        return RegionBounds(
            self.left - value,
            self.right + value,
            self.bottom - value,
            self.top + value,
        )


COMMON_REGIONS = {
    LayoutRegion.HEADER: RegionBounds(-6.60, 6.60, 2.75, 3.55),
    LayoutRegion.FOOTER: RegionBounds(-6.60, 6.60, -3.55, -2.75),
}

TEMPLATE_REGIONS: dict[LayoutTemplate, dict[LayoutRegion, RegionBounds]] = {
    LayoutTemplate.FULL_VISUAL: {
        **COMMON_REGIONS,
        LayoutRegion.CONTENT_FULL: RegionBounds(-6.60, 6.60, -2.55, 2.55),
    },
    LayoutTemplate.SPLIT: {
        **COMMON_REGIONS,
        LayoutRegion.CONTENT_LEFT: RegionBounds(-6.60, -0.30, -2.55, 2.55),
        LayoutRegion.CONTENT_RIGHT: RegionBounds(0.30, 6.60, -2.55, 2.55),
    },
    LayoutTemplate.THREE_COLUMN: {
        **COMMON_REGIONS,
        LayoutRegion.CONTENT_LEFT: RegionBounds(-6.60, -2.35, -2.55, 2.55),
        LayoutRegion.CONTENT_CENTER: RegionBounds(-2.05, 2.05, -2.55, 2.55),
        LayoutRegion.CONTENT_RIGHT: RegionBounds(2.35, 6.60, -2.55, 2.55),
    },
    LayoutTemplate.CENTER_WITH_NOTES: {
        **COMMON_REGIONS,
        LayoutRegion.CONTENT_LEFT: RegionBounds(-6.60, -4.05, -2.55, 2.55),
        LayoutRegion.CONTENT_CENTER: RegionBounds(-3.75, 3.75, -2.55, 2.55),
        LayoutRegion.CONTENT_RIGHT: RegionBounds(4.05, 6.60, -2.55, 2.55),
    },
}

HARD_BOUNDARY_BUFFER = 0.10


def regions_for(template: LayoutTemplate | str) -> dict[LayoutRegion, RegionBounds]:
    return TEMPLATE_REGIONS[LayoutTemplate(template)]


def region_bounds(
    template: LayoutTemplate | str,
    region: LayoutRegion | str,
) -> RegionBounds:
    template_value = LayoutTemplate(template)
    region_value = LayoutRegion(region)
    try:
        return TEMPLATE_REGIONS[template_value][region_value]
    except KeyError as exc:
        raise ValueError(
            f"Region {region_value.value!r} is unavailable in {template_value.value!r}"
        ) from exc


def layout_prompt_guide() -> str:
    lines = [
        "Choose exactly one layout_template before assigning regions.",
        "All coordinates are Manim scene units inside the safe 14.22 x 8 frame.",
    ]
    for template, regions in TEMPLATE_REGIONS.items():
        values = ", ".join(
            f"{name.value}=x[{box.left:.2f},{box.right:.2f}] y[{box.bottom:.2f},{box.top:.2f}]"
            for name, box in regions.items()
        )
        lines.append(f"{template.value}: {values}")
    lines.extend([
        "Use full_visual when one large diagram needs the canvas; do not reserve side regions.",
        "Use split for two peers, three_column only for three compact peers, and "
        "center_with_notes for one central visual plus optional narrow side notes.",
        "Assign at most one independent active artifact to a region per beat. Bundle related "
        "labels and geometry into one registered VGroup artifact.",
        "Annotations may share their target's region only with target_id and overlap_policy=target_only.",
        "If content cannot fit above minimum_scale, simplify it or split the explanation across beats; "
        "never shrink text until it is technically present but unreadable.",
    ])
    return "\n".join(lines)
