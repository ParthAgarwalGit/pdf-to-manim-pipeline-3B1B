"""Reusable 3Blue1Brown animation patterns.

These helpers encode the visual grammar of 3B1B videos:
- Progressive reveal of complex equations
- Staged transformations with breathing room
- Dim-and-focus transitions between concepts

All helpers return lists of (animation, wait) tuples that can be fed
directly to a Scene.construct() method.  When Manim is unavailable
(e.g., in CI), helpers degrade to returning descriptive string stubs.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

try:
    from manim import (
        Mobject,
        ReplacementTransform,
        Write,
    )

    MANIM_AVAILABLE = True
except ImportError:  # pragma: no cover – CI without Manim
    MANIM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# An AnimationStep is either a Manim Animation or a descriptive string stub.
AnimationStep = Tuple[Any, float]  # (animation_or_stub, wait_seconds)


def progressive_write(
    mobject_parts: Sequence[Any],
    run_time_each: float = 1.5,
    wait_between: float = 0.5,
) -> List[AnimationStep]:
    """Write equation parts one by one with pauses.

    This replicates 3B1B's signature style of revealing mathematical
    expressions incrementally — LHS first, then the relation, then RHS.

    Args:
        mobject_parts: Sequence of Mobjects (or submobjects of a VGroup)
            to reveal one at a time.
        run_time_each: Duration of each Write animation.
        wait_between: Pause between successive writes.

    Returns:
        List of (animation, wait) tuples.
    """
    if not mobject_parts:
        return []

    steps: List[AnimationStep] = []
    for i, part in enumerate(mobject_parts):
        if MANIM_AVAILABLE and isinstance(part, Mobject):
            steps.append((Write(part, run_time=run_time_each), 0.0))
        else:
            steps.append((f"Write(part_{i}, run_time={run_time_each})", 0.0))

        # Add breathing room after each part except the last
        if i < len(mobject_parts) - 1:
            steps.append(("wait", wait_between))

    # Extra breathing room after the full reveal
    steps.append(("wait", max(wait_between, 0.5)))
    return steps


def staged_transform(
    source: Any,
    targets: Sequence[Any],
    run_time_each: float = 1.5,
    wait_between: float = 0.5,
) -> List[AnimationStep]:
    """Chain of ReplacementTransform steps with breathing room.

    Useful for showing multi-step algebraic simplifications where each
    step morphs into the next.

    Args:
        source: The starting Mobject.
        targets: Sequence of target Mobjects to transform into, in order.
        run_time_each: Duration of each ReplacementTransform.
        wait_between: Pause between successive transforms.

    Returns:
        List of (animation, wait) tuples.
    """
    if not targets:
        return []

    steps: List[AnimationStep] = []
    current = source

    for i, target in enumerate(targets):
        if MANIM_AVAILABLE and isinstance(current, Mobject) and isinstance(target, Mobject):
            steps.append(
                (ReplacementTransform(current, target, run_time=run_time_each), 0.0)
            )
        else:
            steps.append(
                (f"ReplacementTransform(step_{i}, step_{i + 1}, run_time={run_time_each})", 0.0)
            )
        steps.append(("wait", wait_between))
        current = target

    return steps


def dim_and_focus(
    all_mobjects: Sequence[Any],
    keep_ids: Sequence[int],
    dim_opacity: float = 0.25,
    run_time: float = 0.8,
) -> List[AnimationStep]:
    """Dim old objects and keep focus objects at full opacity.

    Replicates 3B1B's technique of fading background elements when
    introducing a new concept, keeping the viewer's attention on
    the focal point.

    Args:
        all_mobjects: All currently visible Mobjects.
        keep_ids: Indices into all_mobjects that should stay at full opacity.
        dim_opacity: Target opacity for dimmed objects (0.0–1.0).
        run_time: Duration of the dimming animation.

    Returns:
        List of (animation, wait) tuples.
    """
    if not all_mobjects:
        return []

    steps: List[AnimationStep] = []
    keep_set = set(keep_ids)

    for i, mob in enumerate(all_mobjects):
        if i in keep_set:
            continue
        if MANIM_AVAILABLE and isinstance(mob, Mobject):
            steps.append(
                (mob.animate.set_opacity(dim_opacity), 0.0)
            )
        else:
            steps.append(
                (f"mob_{i}.animate.set_opacity({dim_opacity})", 0.0)
            )

    if steps:
        # Add a brief pause after the dimming transition
        steps.append(("wait", 0.3))

    return steps


def execute_animation_steps(scene: Any, steps: List[AnimationStep]) -> None:
    """Execute a list of AnimationSteps on a Manim Scene.

    Args:
        scene: A Manim Scene instance (must have .play() and .wait()).
        steps: Output from progressive_write, staged_transform, or dim_and_focus.
    """
    for step, wait_time in steps:
        if step == "wait":
            scene.wait(wait_time)
        elif isinstance(step, str):
            # String stub — skip in non-Manim environments
            pass
        else:
            scene.play(step)
            if wait_time > 0:
                scene.wait(wait_time)
