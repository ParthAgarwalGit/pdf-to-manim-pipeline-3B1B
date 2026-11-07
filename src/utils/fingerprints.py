"""Content fingerprints for revision runs.

A revision rebuilds an artifact only when its own inputs changed:

- ``beats_fingerprint``: what a scene says. A scene whose beats are unchanged
  keeps its visual plan.
- ``narration_fingerprint``: spoken text and voice. Unchanged means the TTS
  audio is reused.
- ``visual_fingerprint``: visual plan, beats, timing and render quality.
  Unchanged means the scene code and render are reused.

Fingerprints never include scene ids or indices, which shift whenever scenes
are inserted or removed.
"""

import hashlib
import json
from typing import Any, Iterable

FINGERPRINT_VERSION = 1

# Beat fields that feed generation. Tags (role, depth, analogy_id) and
# provenance (author, locked) do not change what is planned or spoken, so
# re-tagging a beat never forces a rebuild.
_BEAT_CONTENT_FIELDS = (
    "beat_id",
    "narration",
    "visual_hint",
    "introduces_concepts",
    "references_prior_beats",
    "source_segment_id",
)


def _digest(payload: Any) -> str:
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def beat_fingerprint(beat: Any) -> str:
    return _digest({
        "v": FINGERPRINT_VERSION,
        **{field: getattr(beat, field, None) for field in _BEAT_CONTENT_FIELDS},
    })


def beats_fingerprint(beats: Iterable[Any]) -> str:
    return _digest([beat_fingerprint(beat) for beat in beats])


def narration_fingerprint(narration: str, voice: str) -> str:
    return _digest({"v": FINGERPRINT_VERSION, "narration": narration, "voice": voice})


def visual_fingerprint(scene: Any, render_quality: str, tts_first: bool) -> str:
    """Inputs of a scene's code and render.

    Under narration-first timing the measured audio length rescales the
    plan, so it is part of the fingerprint; otherwise audio never touches
    the visuals.
    """
    payload = {
        "v": FINGERPRINT_VERSION,
        "beats": beats_fingerprint(getattr(scene, "source_beats", None) or []),
        "narration": scene.narration,
        "intent": scene.visual_intent.model_dump(mode="json"),
        "duration": scene.duration_estimate,
        "audio_duration": scene.audio_duration_seconds if tts_first else None,
        "quality": render_quality,
    }
    timings = getattr(scene, "beat_timings", None)
    if tts_first and timings:
        # The beat timing track is injected into the scene code.
        payload["beat_timings"] = timings
    return _digest(payload)
