"""Frame-level visual QA primitives.

Pure image-analysis functions used by the VideoQualityReviewer to judge
rendered output by what is actually ON SCREEN — the thesis metric — instead
of only reasoning about the generated code text. Deliberately deterministic
and dependency-light (numpy + PIL, both already required by manim) so the
checks run hermetically in tests with synthetic images.

Two failure modes are detected, both fatal to "impactful visualization":
  - EMPTY frame   — nothing (or almost nothing) is drawn over the dark
                    3B1B background at the sampled moment.
  - STATIC scene  — every sampled keyframe is nearly identical; the video
                    is a still slide, not an animation.

Text-wall / off-canvas detection needs OCR or a vision model and is out of
scope for this deterministic pass (see ADR-005 open items).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

# Grayscale luma above which a pixel counts as "drawn" rather than background.
# The 3B1B background (#1C1C1C ≈ luma 28) plus codec noise stays below this.
DEFAULT_LUMA_THRESHOLD = 40

# A frame with less than this fraction of drawn pixels is "empty".
DEFAULT_EMPTY_FRACTION = 0.005

# Two frames whose mean absolute grayscale difference is below this are
# "nearly identical" (codec noise on a still frame stays under ~1.5).
DEFAULT_STATIC_MEAN_DIFF = 2.0

# Outer margin band (fraction of each dimension) inspected for clipped
# content, and the share of a frame's ink inside that band above which the
# frame counts as edge-clipped. A well-composed frame keeps its margins
# almost empty; content sliding off-screen concentrates ink at the border.
DEFAULT_BORDER_BAND = 0.05
DEFAULT_BORDER_INK_SHARE = 0.35


@dataclass
class KeyframeReport:
    """Outcome of analyzing an ordered sample of keyframes from one video."""

    frame_count: int
    nonblank_fractions: List[float] = field(default_factory=list)
    empty_frame_indices: List[int] = field(default_factory=list)
    is_static: bool = False
    border_ink_fractions: List[float] = field(default_factory=list)
    edge_clipped_indices: List[int] = field(default_factory=list)

    @property
    def mostly_empty(self) -> bool:
        """True when at least half of the sampled frames are empty."""
        return (
            self.frame_count > 0
            and len(self.empty_frame_indices) * 2 >= self.frame_count
        )

    @property
    def edge_clipped(self) -> bool:
        """True when any sampled frame concentrates its ink at the border."""
        return bool(self.edge_clipped_indices)


def _to_gray_array(image) -> np.ndarray:
    """PIL image (any mode) -> float64 grayscale ndarray."""
    return np.asarray(image.convert("L"), dtype=np.float64)


def nonblank_fraction(image, luma_threshold: int = DEFAULT_LUMA_THRESHOLD) -> float:
    """Fraction of pixels meaningfully brighter than the dark background."""
    gray = _to_gray_array(image)
    if gray.size == 0:
        return 0.0
    return float((gray > luma_threshold).mean())


def border_ink_fraction(
    image,
    band_frac: float = DEFAULT_BORDER_BAND,
    luma_threshold: int = DEFAULT_LUMA_THRESHOLD,
) -> float:
    """Share of the frame's ink that sits in the outer margin band.

    "Ink" = pixels meaningfully brighter than the dark background. Returns
    0.0 for an empty frame. A centred composition scores near zero; content
    exceeding the screen concentrates ink against the border.
    """
    gray = _to_gray_array(image)
    if gray.size == 0:
        return 0.0
    ink = gray > luma_threshold
    total = int(ink.sum())
    if total == 0:
        return 0.0
    h, w = ink.shape
    bh = max(1, int(round(h * band_frac)))
    bw = max(1, int(round(w * band_frac)))
    interior = ink[bh:h - bh, bw:w - bw]
    border_total = total - int(interior.sum())
    return border_total / total


def mean_abs_diff(image_a, image_b) -> float:
    """Mean absolute grayscale difference between two same-size frames.

    Frames of different sizes are treated as maximally different (a resize
    mid-scene would itself be a defect worth surfacing).
    """
    a, b = _to_gray_array(image_a), _to_gray_array(image_b)
    if a.shape != b.shape:
        return 255.0
    return float(np.abs(a - b).mean())


def analyze_keyframes(
    frames: List,
    empty_fraction: float = DEFAULT_EMPTY_FRACTION,
    static_mean_diff: float = DEFAULT_STATIC_MEAN_DIFF,
    luma_threshold: int = DEFAULT_LUMA_THRESHOLD,
) -> KeyframeReport:
    """Analyze an ordered keyframe sample for empty frames and staticness.

    ``is_static`` requires at least two frames and every consecutive pair to
    be nearly identical — a single sampled frame proves nothing about motion.
    """
    fractions = [nonblank_fraction(f, luma_threshold) for f in frames]
    empties = [i for i, frac in enumerate(fractions) if frac < empty_fraction]

    static = False
    if len(frames) >= 2:
        static = all(
            mean_abs_diff(a, b) < static_mean_diff
            for a, b in zip(frames, frames[1:])
        )

    border_fractions = [border_ink_fraction(f, luma_threshold=luma_threshold) for f in frames]
    clipped = [
        i for i, (ink, border) in enumerate(zip(fractions, border_fractions))
        if ink >= empty_fraction and border > DEFAULT_BORDER_INK_SHARE
    ]

    return KeyframeReport(
        frame_count=len(frames),
        nonblank_fractions=fractions,
        empty_frame_indices=empties,
        is_static=static,
        border_ink_fractions=border_fractions,
        edge_clipped_indices=clipped,
    )
