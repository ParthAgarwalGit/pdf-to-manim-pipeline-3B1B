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


@dataclass
class KeyframeReport:
    """Outcome of analyzing an ordered sample of keyframes from one video."""

    frame_count: int
    nonblank_fractions: List[float] = field(default_factory=list)
    empty_frame_indices: List[int] = field(default_factory=list)
    is_static: bool = False

    @property
    def mostly_empty(self) -> bool:
        """True when at least half of the sampled frames are empty."""
        return (
            self.frame_count > 0
            and len(self.empty_frame_indices) * 2 >= self.frame_count
        )


def _to_gray_array(image) -> np.ndarray:
    """PIL image (any mode) -> float64 grayscale ndarray."""
    return np.asarray(image.convert("L"), dtype=np.float64)


def nonblank_fraction(image, luma_threshold: int = DEFAULT_LUMA_THRESHOLD) -> float:
    """Fraction of pixels meaningfully brighter than the dark background."""
    gray = _to_gray_array(image)
    if gray.size == 0:
        return 0.0
    return float((gray > luma_threshold).mean())


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

    return KeyframeReport(
        frame_count=len(frames),
        nonblank_fractions=fractions,
        empty_frame_indices=empties,
        is_static=static,
    )
