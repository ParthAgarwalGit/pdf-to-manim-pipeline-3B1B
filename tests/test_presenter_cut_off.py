"""Presenter takes that end mid-word are caught, redone with more time, and never used.

Hermetic: Omni is faked with local ffmpeg clips (skipped without ffmpeg).
"""

import base64
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.gemini_presenter import (
    CUT_OFF_RETRY_EXTRA_SECONDS,
    MAX_CLIP_SECONDS,
    GeminiOmniPresenter,
    PresenterClipCutOff,
    clip_seconds,
    speech_bounds,
    speech_cut_off,
    split_for_clips,
    trailing_quiet_seconds,
)
from src.pipeline_v2.prompt_text import HOOK_MAX_WORDS_PRESENTER
from tests.test_did_avatar import needs_ffmpeg
from tests.test_gemini_presenter_and_layout import LINE, _padded_clip


def _speech_wav(path: Path, speech_seconds: float, tail_seconds: float, noise: float = 0.0) -> Path:
    """A voiced tone with a syllable-like envelope, then a tail of (noisy) quiet."""
    rate = 16000
    t = np.arange(int(rate * speech_seconds)) / rate
    speech = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    signal = np.concatenate([speech, np.zeros(int(rate * tail_seconds))])
    signal = signal + noise * np.random.default_rng(0).standard_normal(len(signal))
    data = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


class SequenceOmni:
    """Stands in for client.interactions.create, returning the given takes in turn."""

    def __init__(self, takes):
        self.takes = list(takes)
        self.durations = []
        self.interactions = SimpleNamespace(create=self.create)

    def create(self, model, input, response_format):
        self.durations.append(response_format["duration"])
        data = self.takes[min(len(self.durations), len(self.takes)) - 1]
        return SimpleNamespace(output_video=SimpleNamespace(data=base64.b64encode(data).decode()))


@pytest.fixture(scope="module")
def takes(tmp_path_factory):
    from tests.test_did_avatar import _ffmpeg

    if _ffmpeg() is None:
        pytest.skip("ffmpeg is not installed")
    folder = tmp_path_factory.mktemp("takes")
    return SimpleNamespace(
        # A quiet lead-in, then speech that runs to the last frame.
        cut=_padded_clip(folder / "cut.mp4", lead=0.5, speech=1.5, tail=0.0).read_bytes(),
        whole=_padded_clip(folder / "whole.mp4", lead=0.0, speech=1.6, tail=0.4).read_bytes(),
    )


# ── sizing ───────────────────────────────────────────────────────────────

def test_long_lines_get_room_to_finish():
    # A 34-word hook was requested as 13 s and ran out of time; it now gets the longest clip.
    assert clip_seconds(" ".join(["word"] * 34)) == MAX_CLIP_SECONDS
    hook = " ".join(["word"] * HOOK_MAX_WORDS_PRESENTER)
    assert len(split_for_clips(hook)) == 1 and clip_seconds(hook) < MAX_CLIP_SECONDS
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(12))
    assert all(clip_seconds(chunk) <= MAX_CLIP_SECONDS for chunk in split_for_clips(text))


# ── detection ────────────────────────────────────────────────────────────

@needs_ffmpeg
def test_speech_running_to_the_last_frame_is_cut_off(tmp_path):
    assert speech_cut_off(_speech_wav(tmp_path / "cut.wav", 2.0, 0.0))
    whole = _speech_wav(tmp_path / "whole.wav", 2.0, 0.5)
    assert not speech_cut_off(whole)
    assert trailing_quiet_seconds(whole) == pytest.approx(0.5, abs=0.05)


@needs_ffmpeg
def test_a_noisy_quiet_tail_still_counts_as_quiet(tmp_path):
    noisy = _speech_wav(tmp_path / "noisy.wav", 2.0, 0.5, noise=0.03)
    # A fixed -35 dB threshold hears the noise floor as speech to the end...
    assert speech_bounds(noisy)[1] == pytest.approx(2.5, abs=0.05)
    # ...the check measured against the clip's own floor does not.
    assert not speech_cut_off(noisy)
    assert speech_cut_off(_speech_wav(tmp_path / "noisy_cut.wav", 2.0, 0.0, noise=0.03))


@needs_ffmpeg
def test_no_speech_is_not_called_cut_off(tmp_path):
    assert not speech_cut_off(_speech_wav(tmp_path / "quiet.wav", 0.0, 2.0, noise=0.01))
    assert trailing_quiet_seconds(tmp_path / "missing.wav") is None


# ── regeneration ─────────────────────────────────────────────────────────

@needs_ffmpeg
def test_a_cut_off_take_is_redone_with_more_time(tmp_path, takes):
    fake = SequenceOmni([takes.cut, takes.whole])
    presenter = GeminiOmniPresenter(client=fake, use_default_transcriber=False)
    path, seconds = presenter.speak(LINE, str(tmp_path / "line.mp4"))
    first = clip_seconds(LINE)
    assert fake.durations == [f"{first}s", f"{first + CUT_OFF_RETRY_EXTRA_SECONDS}s"]
    assert [take["cut_off"] for take in presenter.clips] == [True, False]
    assert not speech_cut_off(Path(path)) and seconds == pytest.approx(2.0, abs=0.2)


@needs_ffmpeg
def test_every_take_cut_off_fails_and_caches_nothing(tmp_path, takes):
    fake = SequenceOmni([takes.cut, takes.cut])
    cache = tmp_path / "cache"
    presenter = GeminiOmniPresenter(client=fake, use_default_transcriber=False, cache_dir=cache)
    with pytest.raises(PresenterClipCutOff):
        presenter.speak(LINE, str(tmp_path / "line.mp4"))
    assert len(fake.durations) == 2 and not list(cache.glob("*.mp4"))


@needs_ffmpeg
def test_a_cut_off_take_in_the_cache_is_made_again(tmp_path, takes):
    fake = SequenceOmni([takes.whole])
    cache = tmp_path / "cache"
    presenter = GeminiOmniPresenter(client=fake, use_default_transcriber=False, cache_dir=cache)
    cached = presenter._cache_path(LINE, clip_seconds(LINE))
    cached.parent.mkdir(parents=True)
    cached.write_bytes(takes.cut)  # made before takes were checked
    presenter.speak(LINE, str(tmp_path / "line.mp4"))
    assert len(fake.durations) == 1 and not presenter.clips[0].get("cached")
    assert not speech_cut_off(cached)  # the complete take replaced it


@needs_ffmpeg
def test_a_complete_cached_take_is_reused(tmp_path, takes):
    fake = SequenceOmni([takes.whole])
    cache = tmp_path / "cache"
    GeminiOmniPresenter(client=fake, use_default_transcriber=False, cache_dir=cache).speak(
        LINE, str(tmp_path / "a.mp4"))
    again = GeminiOmniPresenter(client=fake, use_default_transcriber=False, cache_dir=cache)
    again.speak(LINE, str(tmp_path / "b.mp4"))
    assert len(fake.durations) == 1 and again.clips[0]["cached"] is True
