"""Runtime narration sync, imported by generated Manim scenes.

Narration-first runs measure when each beat starts in the scene's audio.
After code generation that timing track is injected into the scene class
(see ``src.utils.sync_contract``)::

    class Scene004(Scene):
        BEAT_TIMES = {"beat_007": 0.0, "beat_008": 7.42}
        NARRATION_END = 14.9

and the generated code calls ``sync_to(self, "beat_008")`` before each
beat's animations and ``sync_to_end(self)`` last. Each call waits on
Manim's own scene clock, so animation timing is enforced at render time
rather than left to the LLM's arithmetic. Stdlib only: this module runs
inside the render process.
"""

_MIN_WAIT = 1.0 / 60.0  # below one frame a wait is a no-op


def _now(scene) -> float:
    clock = getattr(scene, "time", None)
    if isinstance(clock, (int, float)):
        return float(clock)
    renderer = getattr(scene, "renderer", None)
    return float(getattr(renderer, "time", 0.0) or 0.0)


def _hold_until(scene, target) -> float:
    if target is None:
        return 0.0
    gap = float(target) - _now(scene)
    if gap > _MIN_WAIT:
        scene.wait(gap)
        return gap
    return 0.0


def sync_to(scene, beat_id: str) -> float:
    """Hold until the narration reaches ``beat_id``; returns the wait added.

    Already past the beat (the animation is running late) or an unknown
    beat id: no wait.
    """
    return _hold_until(scene, (getattr(scene, "BEAT_TIMES", None) or {}).get(beat_id))


def sync_to_end(scene) -> float:
    """Hold until the narration ends, so the video is never shorter than it."""
    return _hold_until(scene, getattr(scene, "NARRATION_END", None))
