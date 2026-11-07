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
from typing import Any, Dict, Iterable, Optional

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


def narration_fingerprint(
    narration: str,
    voice: str,
    pauses: Optional[Dict[str, float]] = None,
    lexicon: Optional[Dict[str, str]] = None,
) -> str:
    """Inputs of a scene's TTS audio.

    Pauses and lexicon entries only enter the digest when present, so runs
    made before they existed keep matching.
    """
    payload: Dict[str, Any] = {"v": FINGERPRINT_VERSION, "narration": narration, "voice": voice}
    if pauses:
        payload["pauses"] = {k: v for k, v in sorted(pauses.items()) if v}
    if lexicon:
        payload["lexicon"] = dict(sorted(lexicon.items()))
    return _digest(payload)


def scene_narration_fingerprint(scene: Any, voice: str, lexicon: Optional[Dict[str, str]] = None) -> str:
    """Narration fingerprint of a storyboard scene, with only the pauses and
    lexicon entries that actually affect its audio."""
    beats = getattr(scene, "source_beats", None) or []
    pauses = {b.beat_id: b.pause_after for b in beats if getattr(b, "pause_after", 0)}
    return narration_fingerprint(
        scene.narration, voice, pauses, lexicon_for(scene.narration, lexicon),
    )


def lexicon_for(text: str, lexicon: Optional[Dict[str, str]]) -> Dict[str, str]:
    """The lexicon entries whose word occurs in ``text`` (case-insensitive)."""
    import re

    return {
        word: say for word, say in (lexicon or {}).items()
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text or "", re.IGNORECASE)
    }


def apply_lexicon(text: str, lexicon: Optional[Dict[str, str]]) -> str:
    """Text as TTS should read it: lexicon words replaced by their respelling."""
    import re

    for word, say in (lexicon or {}).items():
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", say, text, flags=re.IGNORECASE)
    return text


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
