"""Narration pacing and low-cost visual repetition checks."""

import pytest
from PIL import Image, ImageDraw

from src.pipeline_v2.diagnostics import frame_sequence_similarity
from src.pipeline_v2.models import NarratedScene, NarrationBeat
from src.pipeline_v2.tts import (
    SCENE_END_PAUSE_SECONDS,
    _narration_for_tts,
    _silence,
    ensure_trailing_buffer,
)


def test_tts_script_uses_punctuated_beat_paragraphs():
    scene = NarratedScene(
        scene_id="scene",
        narration="First thought Second thought",
        beats=[
            NarrationBeat(beat_id="b1", text="First thought", source_fact_ids=["f"]),
            NarrationBeat(beat_id="b2", text="Second thought?", source_fact_ids=["f"]),
        ],
    )

    assert _narration_for_tts(scene) == "First thought.\n\nSecond thought?"
    assert SCENE_END_PAUSE_SECONDS == 2.0


def test_repetition_similarity_is_high_for_matching_informative_frames(tmp_path):
    paths = []
    for name in ("left.png", "right.png"):
        path = tmp_path / name
        image = Image.new("RGB", (320, 180), "#111111")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 40, 240, 140), outline="white", width=8)
        image.save(path)
        paths.append(str(path))

    assert frame_sequence_similarity([paths[0]], [paths[1]]) > 0.99


@pytest.mark.integration
def test_legacy_audio_is_locally_extended_to_two_second_tail(tmp_path):
    source = tmp_path / "legacy.mp3"
    _silence(source, 0.5)
    scene = NarratedScene(
        scene_id="scene",
        narration="A short explanation.",
        audio_path=str(source),
        audio_duration_seconds=0.5,
        audio_tail_seconds=0.0,
        beats=[NarrationBeat(
            beat_id="b1", text="A short explanation.", source_fact_ids=["f"],
        )],
    )

    updated = ensure_trailing_buffer(scene, tmp_path / "buffered.mp3")

    assert updated.audio_tail_seconds == 2.0
    assert updated.audio_duration_seconds > 2.4
