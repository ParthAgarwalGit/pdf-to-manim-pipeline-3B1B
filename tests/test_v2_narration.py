"""Narration pacing and low-cost visual repetition checks."""

from PIL import Image, ImageDraw

from src.pipeline_v2.diagnostics import frame_sequence_similarity
from src.pipeline_v2.models import NarratedScene, NarrationBeat
from src.pipeline_v2.tts import _narration_for_tts


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
