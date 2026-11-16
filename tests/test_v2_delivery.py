"""V2 delivery gate and narration-safe composition.

The gate tests drive ``run_delivery_gate`` with fake probes keyed by path, so
no ffmpeg runs. The fixture tests at the end build real clips with ffmpeg
(lavfi sources) and skip when ffmpeg is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.pipeline_v2 import delivery, media
from src.pipeline_v2.delivery import (
    THRESHOLDS,
    DeliveryCheck,
    DeliveryReport,
    MediaProbes,
    SceneClip,
    SceneDelivery,
    check_final,
    check_scene,
    format_timestamp,
    run_delivery_gate,
    summary_lines,
)
from src.pipeline_v2.tts import SCENE_END_PAUSE_SECONDS
from src.utils import media_probe
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe
from src.utils.media_probe import Loudness, MediaInfo, StreamInfo


# ── fakes ───────────────────────────────────────────────────────────────

def _info(video=None, audio=None, duration=None, size=(1280, 720), fps=30.0, channels=2):
    return MediaInfo(
        duration=duration if duration is not None else max(video or 0, audio or 0),
        video=StreamInfo(
            codec="h264", duration=video, width=size[0], height=size[1], fps=fps, pix_fmt="yuv420p",
        ) if video is not None else None,
        audio=StreamInfo(codec="aac", duration=audio, sample_rate=48000, channels=channels)
        if audio is not None else None,
    )


class FakeProbes(MediaProbes):
    """Media facts keyed by path; mutate ``facts`` to simulate a repair."""

    def __init__(self, facts, loudness=None, silences=None, fix_loudness=True):
        self.facts = facts
        self.loud = loudness or {}
        self.quiet = silences or {}
        self.fix_loudness = fix_loudness
        self.normalized: list[str] = []
        self.silence_calls: list[tuple[str, dict]] = []
        super().__init__(
            probe=lambda path: self.facts.get(str(path)),
            loudness=lambda path: self.loud.get(str(path), Loudness(-16.0, -3.0)),
            silences=self._silences,
            normalize=self._normalize,
        )

    def _silences(self, path, **kwargs):
        self.silence_calls.append((str(path), kwargs))
        return self.quiet.get(str(path), [])

    def _normalize(self, src, dst, measured=None):
        self.normalized.append(str(src))
        Path(dst).write_bytes(b"normalized")
        if self.fix_loudness:
            self.loud[str(src)] = Loudness(-16.0, -2.0)
        return True


def _clip(tmp_path, scene_id="scene_001", narration=6.0, **kwargs):
    paths = {}
    for kind in ("render", "audio", "composed"):
        path = tmp_path / f"{scene_id}_{kind}.mp4"
        path.write_bytes(b"x")
        paths[kind] = path
    return SceneClip(
        scene_id=scene_id, composed_path=paths["composed"], render_path=paths["render"],
        audio_path=paths["audio"], narration_seconds=narration, **kwargs,
    )


def _facts(clip, video, audio, composed_audio=None):
    return {
        str(clip.render_path): _info(video=video),
        str(clip.audio_path): _info(audio=audio),
        str(clip.composed_path): _info(
            video=max(video, audio), audio=audio if composed_audio is None else composed_audio,
        ),
    }


def _final(tmp_path, facts, clips, seconds=None):
    """A final video whose length is the sum of its clips unless told otherwise."""
    path = tmp_path / "final.mp4"
    path.write_bytes(b"x")
    total = sum(facts[str(clip.composed_path)].duration for clip in clips)
    seconds = total if seconds is None else seconds
    facts[str(path)] = _info(video=seconds, audio=seconds)
    return path


def _by_check(checks):
    return {check.check: check for check in checks}


def _good_run(tmp_path, count=2):
    clips = [_clip(tmp_path, f"scene_{index:03d}") for index in range(1, count + 1)]
    facts = {}
    for clip in clips:
        facts.update(_facts(clip, 6.0, 6.0))
    final = _final(tmp_path, facts, clips)
    return clips, facts, final


# ── scene checks ────────────────────────────────────────────────────────

class TestSceneChecks:
    def test_check_names_and_order(self, tmp_path):
        clip = _clip(tmp_path)
        checks = check_scene(clip, FakeProbes(_facts(clip, 6.0, 6.0)))
        assert [check.check for check in checks] == [
            "streams", "av_duration", "narration_intact", "narration_gaps", "loudness",
        ]
        assert all(check.passed is True for check in checks)

    def test_frozen_tail_fails_with_recompose(self, tmp_path):
        clip = _clip(tmp_path, narration=7.4)
        check = _by_check(check_scene(clip, FakeProbes(_facts(clip, 5.0, 7.4))))["av_duration"]
        assert check.passed is False
        assert check.severity == "high"
        assert check.repair == "recompose"
        assert "2.4s before the narration" in check.message
        assert check.measured["gap_seconds"] == pytest.approx(2.4)

    def test_short_frozen_tail_is_medium_and_small_gap_passes(self, tmp_path):
        clip = _clip(tmp_path, narration=6.6)
        medium = _by_check(check_scene(clip, FakeProbes(_facts(clip, 6.0, 6.6))))["av_duration"]
        assert medium.passed is False and medium.severity == "medium"
        clip.narration_seconds = 6.25
        assert _by_check(check_scene(clip, FakeProbes(_facts(clip, 6.0, 6.25))))["av_duration"].passed is True

    def test_animation_longer_than_narration_is_not_a_frozen_tail(self, tmp_path):
        clip = _clip(tmp_path, narration=5.0)
        checks = _by_check(check_scene(clip, FakeProbes(_facts(clip, 8.0, 5.0, composed_audio=8.0))))
        assert checks["av_duration"].passed is True
        assert checks["narration_intact"].passed is True

    def test_av_duration_is_skipped_without_a_render_or_narration_length(self, tmp_path):
        clip = _clip(tmp_path)
        facts = _facts(clip, 6.0, 6.0)
        clip.render_path = None
        skipped = _by_check(check_scene(clip, FakeProbes(facts)))["av_duration"]
        assert skipped.passed is None and skipped.severity == "info"

        unknown = SceneClip(scene_id="scene_002", composed_path=clip.composed_path)
        checks = _by_check(check_scene(unknown, FakeProbes(facts)))
        assert checks["av_duration"].severity == "info"
        assert checks["narration_intact"].severity == "info"

    def test_narration_length_is_probed_when_not_given(self, tmp_path):
        clip = _clip(tmp_path, narration=None)
        check = _by_check(check_scene(clip, FakeProbes(_facts(clip, 5.0, 7.4))))["av_duration"]
        assert check.passed is False and check.measured["narration_seconds"] == pytest.approx(7.4)

    def test_clipped_narration_fails_with_recompose(self, tmp_path):
        clip = _clip(tmp_path, narration=6.4)
        check = _by_check(check_scene(
            clip, FakeProbes(_facts(clip, 6.4, 6.4, composed_audio=6.0)),
        ))["narration_intact"]
        assert check.passed is False
        assert check.severity == "high"
        assert check.repair == "recompose"
        assert "0.40s of the narration is cut off" in check.message

    def test_tiny_narration_difference_is_not_a_cut(self, tmp_path):
        clip = _clip(tmp_path, narration=6.0)
        check = _by_check(check_scene(
            clip, FakeProbes(_facts(clip, 6.0, 6.0, composed_audio=5.96)),
        ))["narration_intact"]
        assert check.passed is True

    def test_long_silence_is_flagged(self, tmp_path):
        clip = _clip(tmp_path, narration=12.0, tail_seconds=2.0)
        probes = FakeProbes(_facts(clip, 12.0, 12.0), silences={str(clip.audio_path): [(3.0, 6.5)]})
        check = _by_check(check_scene(clip, probes))["narration_gaps"]
        assert check.passed is False
        assert check.severity == "medium"
        assert check.repair is None
        assert "3.5s silence 3.0s into the scene" in check.message
        assert check.measured["silences"] == [[3.0, 6.5]]

    def test_scene_end_pause_is_not_a_gap(self, tmp_path):
        clip = _clip(tmp_path, narration=10.0, tail_seconds=SCENE_END_PAUSE_SECONDS)
        # 1.5 s of natural trailing quiet runs into the 2 s scene-end pause.
        probes = FakeProbes(_facts(clip, 10.0, 10.0), silences={str(clip.audio_path): [(6.5, 10.0)]})
        check = _by_check(check_scene(clip, probes))["narration_gaps"]
        assert check.passed is True
        assert check.measured["silences"] == []

    def test_silence_probe_uses_the_narration_file_and_thresholds(self, tmp_path):
        clip = _clip(tmp_path, narration=9.0)
        probes = FakeProbes(_facts(clip, 9.0, 9.0))
        check_scene(clip, probes, {"silence_seconds": 4.0})
        path, kwargs = probes.silence_calls[0]
        assert path == str(clip.audio_path)
        assert kwargs == {"noise_db": -40.0, "min_duration": 4.0, "total": 9.0}

    def test_silence_threshold_stays_above_the_scene_end_pause(self):
        assert THRESHOLDS["silence_seconds"] > SCENE_END_PAUSE_SECONDS

    def test_silent_fallback_narration_skips_loudness_and_gaps(self, tmp_path):
        clip = _clip(tmp_path, narration=8.0, silent_narration=True)
        probes = FakeProbes(
            _facts(clip, 8.0, 8.0),
            loudness={str(clip.composed_path): Loudness(float("-inf"), float("-inf"))},
            silences={str(clip.audio_path): [(0.0, 8.0)]},
        )
        checks = _by_check(check_scene(clip, probes))
        for name in ("narration_gaps", "loudness"):
            assert checks[name].passed is None and checks[name].severity == "info"
        assert probes.silence_calls == []
        report = run_delivery_gate([clip], _final(tmp_path, probes.facts, [clip]), probes=probes)
        assert report.status == "ready"
        assert probes.normalized == []

    def test_quiet_and_hot_clips_fail_loudness_with_renormalize(self, tmp_path):
        clip = _clip(tmp_path)
        facts = _facts(clip, 6.0, 6.0)
        quiet = _by_check(check_scene(
            clip, FakeProbes(facts, loudness={str(clip.composed_path): Loudness(-27.0, -9.0)}),
        ))["loudness"]
        assert quiet.passed is False and quiet.severity == "medium" and quiet.repair == "renormalize"
        assert "-27.0 LUFS" in quiet.message
        hot = _by_check(check_scene(
            clip, FakeProbes(facts, loudness={str(clip.composed_path): Loudness(-16.0, -0.2)}),
        ))["loudness"]
        assert hot.passed is False and hot.repair == "renormalize"
        assert "-0.2 dBTP" in hot.message

    def test_unmeasurable_loudness_is_none_not_a_pass(self, tmp_path):
        clip = _clip(tmp_path)
        facts = _facts(clip, 6.0, 6.0)
        for reading in (None, Loudness(float("-inf"), float("-inf"))):
            probes = FakeProbes(facts)
            probes.loudness = lambda path, reading=reading: reading
            check = _by_check(check_scene(clip, probes))["loudness"]
            assert check.passed is None and check.severity == "medium"

    def test_missing_audio_stream_fails_streams(self, tmp_path):
        clip = _clip(tmp_path)
        facts = _facts(clip, 6.0, 6.0)
        facts[str(clip.composed_path)] = _info(video=6.0)
        checks = _by_check(check_scene(clip, FakeProbes(facts)))
        assert checks["streams"].passed is False and checks["streams"].severity == "high"
        assert "no audio track" in checks["streams"].message
        assert checks["narration_intact"].passed is None

    def test_unmeasurable_clip_reports_none(self, tmp_path):
        clip = _clip(tmp_path)
        probes = FakeProbes({})
        probes.loudness = lambda path: None
        probes.silences = lambda path, **kwargs: None
        checks = _by_check(check_scene(clip, probes))
        assert checks["streams"].passed is None
        assert checks["av_duration"].passed is None
        assert checks["narration_intact"].passed is None
        assert checks["narration_gaps"].passed is None
        assert checks["loudness"].passed is None


# ── whole-video checks ──────────────────────────────────────────────────

class TestFinalChecks:
    def test_check_names_and_a_clean_final(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 3)
        checks = check_final(final, clips, FakeProbes(facts))
        assert [check.check for check in checks] == [
            "concat_success", "codec_consistency", "duration_drift", "av_drift",
        ]
        assert all(check.passed is True for check in checks)

    def test_codec_mismatch_is_reported_as_info_after_a_normalising_join(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 3)
        facts[str(clips[2].composed_path)] = _info(video=6.0, audio=6.0, size=(1920, 1080), fps=25.0)
        check = _by_check(check_final(final, clips, FakeProbes(facts)))["codec_consistency"]
        assert check.passed is True and check.severity == "info"
        assert check.measured["odd_clips"] == ["scene_003"]
        assert check.measured["signature"][1:3] == [1280, 720]
        assert "scene_003" in check.message

    def test_codec_mismatch_fails_for_a_stream_copy_join(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 3)
        facts[str(clips[0].composed_path)] = _info(video=6.0, audio=6.0, channels=1)
        check = _by_check(check_final(final, clips, FakeProbes(facts), normalized=False))["codec_consistency"]
        assert check.passed is False and check.severity == "high" and check.repair == "reconcat"
        assert check.measured["odd_clips"] == ["scene_001"]

    def test_clip_without_audio_is_high(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        facts[str(clips[1].composed_path)] = _info(video=6.0)
        check = _by_check(check_final(final, clips, FakeProbes(facts)))["codec_consistency"]
        assert check.passed is False and check.severity == "high"
        assert check.measured["no_audio_clips"] == ["scene_002"]

    def test_duration_drift_tolerance_scales_with_clip_count(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        facts[str(final)] = _info(video=12.28, audio=12.28)  # tolerance 0.25 + 2 * 0.02
        assert _by_check(check_final(final, clips, FakeProbes(facts)))["duration_drift"].passed is True
        facts[str(final)] = _info(video=12.4, audio=12.4)
        check = _by_check(check_final(final, clips, FakeProbes(facts)))["duration_drift"]
        assert check.passed is False and check.severity == "high" and check.repair == "reconcat"
        assert check.measured["tolerance_seconds"] == pytest.approx(0.29)

    def test_av_drift_in_the_final_video(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        facts[str(final)] = _info(video=12.0, audio=11.2, duration=12.0)
        check = _by_check(check_final(final, clips, FakeProbes(facts)))["av_drift"]
        assert check.passed is False and check.severity == "high" and check.repair == "reconcat"
        assert "0.80s" in check.message

    def test_final_without_audio_is_critical(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        facts[str(final)] = _info(video=12.0)
        check = _by_check(check_final(final, clips, FakeProbes(facts)))["concat_success"]
        assert check.passed is False and check.severity == "critical"


# ── gate: repairs and verdict ───────────────────────────────────────────

class TestGate:
    def test_ready_only_when_everything_passes(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 3)
        report = run_delivery_gate(clips, final, probes=FakeProbes(facts))
        assert isinstance(report, DeliveryReport)
        assert report.status == "ready"
        assert report.repair_rounds == 0
        assert report.thresholds == THRESHOLDS
        assert [scene.passed for scene in report.scenes] == [True, True, True]
        assert report.summary == "Ready: 3/3 scenes passed sync and quality checks."
        # The timeline is a running clock over the composed clips.
        assert [(scene.start_seconds, scene.end_seconds) for scene in report.scenes] == [
            (0.0, 6.0), (6.0, 12.0), (12.0, 18.0),
        ]

        # One failing scene check, or one failing final check, is enough to hold it back.
        probes = FakeProbes(facts, silences={str(clips[1].audio_path): [(1.0, 5.0)]})
        assert run_delivery_gate(clips, final, probes=probes).status == "needs_review"
        facts[str(final)] = _info(video=18.0, audio=16.0, duration=18.0)
        assert run_delivery_gate(clips, final, probes=FakeProbes(facts)).status == "needs_review"

    def test_no_clips_is_never_ready(self, tmp_path):
        assert run_delivery_gate([], tmp_path / "final.mp4", probes=FakeProbes({})).status == "needs_review"

    def test_quiet_clip_is_renormalized_then_passes(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        quiet = clips[0]
        probes = FakeProbes(facts, loudness={str(quiet.composed_path): Loudness(-27.0, -9.0, 3.0, -37.0)})
        rebuilt = []

        def reconcat(paths):
            rebuilt.append(list(paths))
            return final

        report = run_delivery_gate(clips, final, probes=probes, reconcat=reconcat)
        assert probes.normalized == [str(quiet.composed_path)]
        assert quiet.composed_path.read_bytes() == b"normalized"
        assert not quiet.composed_path.with_name("scene_001_composed.norm.mp4").exists()
        assert report.scenes[0].repairs_applied == ["renormalize"]
        assert report.scenes[0].passed is True
        assert report.repair_rounds == 1
        # A repaired clip means the final video is rebuilt from the clips, once.
        assert rebuilt == [[clip.composed_path for clip in clips]]
        assert report.status == "ready"

    def test_passing_scene_is_never_touched(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        failing = clips[1]
        facts.update(_facts(failing, 6.0, 6.0, composed_audio=5.0))
        probes = FakeProbes(facts)
        recomposed = []

        def recompose(clip):
            recomposed.append(clip.scene_id)
            facts[str(clip.composed_path)] = _info(video=6.0, audio=6.0)
            return clip.composed_path

        report = run_delivery_gate(clips, final, probes=probes, recompose=recompose, reconcat=lambda paths: final)
        assert recomposed == ["scene_002"]
        assert probes.normalized == []
        assert clips[0].composed_path.read_bytes() == b"x"
        assert report.scenes[0].repairs_applied == []
        assert report.scenes[1].repairs_applied == ["recompose"]
        assert report.status == "ready"

    def test_recompose_may_return_a_new_clip_path(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 1)
        clip = clips[0]
        facts.update(_facts(clip, 6.0, 6.0, composed_audio=5.0))
        replacement = tmp_path / "recomposed.mp4"
        seen = []

        def recompose(scene):
            replacement.write_bytes(b"y")
            facts[str(replacement)] = _info(video=6.0, audio=6.0)
            return replacement

        def reconcat(paths):
            seen.append(list(paths))
            return final

        report = run_delivery_gate(clips, final, probes=FakeProbes(facts), recompose=recompose, reconcat=reconcat)
        assert clip.composed_path == replacement
        assert seen == [[replacement]]
        assert report.status == "ready"

    def test_unfixable_repair_is_not_retried_beyond_the_bound(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 1)
        clip = clips[0]
        clip.narration_seconds = 9.0
        facts.update(_facts(clip, 5.0, 9.0))  # the render itself is too short
        calls = []

        def recompose(scene):
            calls.append(scene.scene_id)
            return scene.composed_path

        _final(tmp_path, facts, clips)
        report = run_delivery_gate(
            clips, final, probes=FakeProbes(facts), max_repair_rounds=5,
            recompose=recompose, reconcat=lambda paths: final,
        )
        assert calls == ["scene_001"]
        assert report.status == "needs_review"
        assert report.repair_rounds == 1
        assert _by_check(report.scenes[0].checks)["av_duration"].passed is False

    def test_failing_normalize_is_not_retried(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 1)
        probes = FakeProbes(
            facts, loudness={str(clips[0].composed_path): Loudness(-27.0, -9.0)}, fix_loudness=False,
        )
        report = run_delivery_gate(clips, final, probes=probes, max_repair_rounds=4)
        assert probes.normalized == [str(clips[0].composed_path)]
        assert report.status == "needs_review"

    def test_each_scene_gets_at_most_max_repair_rounds(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 1)
        clip = clips[0]
        facts.update(_facts(clip, 6.0, 6.0, composed_audio=5.0))
        probes = FakeProbes(
            facts, loudness={str(clip.composed_path): Loudness(-27.0, -9.0)}, fix_loudness=False,
        )
        calls = []

        def recompose(scene):
            calls.append("recompose")
            return scene.composed_path

        report = run_delivery_gate(clips, final, probes=probes, max_repair_rounds=1, recompose=recompose)
        assert calls == ["recompose"] and probes.normalized == []
        assert report.repair_rounds == 1

        report = run_delivery_gate(clips, final, probes=probes, max_repair_rounds=0, recompose=recompose)
        assert calls == ["recompose"] and probes.normalized == []
        assert report.repair_rounds == 0 and report.status == "needs_review"

    def test_missing_recompose_callback_leaves_the_scene_for_review(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 1)
        facts.update(_facts(clips[0], 6.0, 6.0, composed_audio=5.0))
        report = run_delivery_gate(clips, final, probes=FakeProbes(facts))
        assert report.status == "needs_review"
        assert report.repair_rounds == 0
        assert report.scenes[0].repairs_applied == []

    def test_repaired_clips_without_a_rebuilt_final_need_review(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 1)
        probes = FakeProbes(facts, loudness={str(clips[0].composed_path): Loudness(-27.0, -9.0)})

        def broken_reconcat(paths):
            raise RuntimeError("ffmpeg exploded")

        for reconcat in (None, broken_reconcat):
            probes.loud[str(clips[0].composed_path)] = Loudness(-27.0, -9.0)
            report = run_delivery_gate(clips, final, probes=probes, reconcat=reconcat)
            assert report.scenes[0].passed is True
            assert report.status == "needs_review"
            stale = _by_check(report.final_checks)["final_rebuilt"]
            assert stale.passed is False and stale.severity == "high"

    def test_duration_drift_triggers_exactly_one_reconcat(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 3)
        facts[str(final)] = _info(video=16.0, audio=16.0)
        calls = []

        def reconcat(paths):
            calls.append(list(paths))
            facts[str(final)] = _info(video=18.0, audio=18.0)
            return final

        report = run_delivery_gate(clips, final, probes=FakeProbes(facts), reconcat=reconcat)
        assert calls == [[clip.composed_path for clip in clips]]
        assert report.status == "ready"
        assert report.repair_rounds == 1

    def test_reconcat_that_does_not_help_runs_only_once(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 3)
        facts[str(final)] = _info(video=16.0, audio=16.0)
        calls = []
        report = run_delivery_gate(
            clips, final, probes=FakeProbes(facts), max_repair_rounds=5,
            reconcat=lambda paths: calls.append(1) or final,
        )
        assert calls == [1]
        assert report.status == "needs_review"
        assert _by_check(report.final_checks)["duration_drift"].passed is False

    def test_missing_final_is_critical_and_needs_review(self, tmp_path):
        clips, facts, _ = _good_run(tmp_path, 2)
        missing = tmp_path / "never_written.mp4"
        report = run_delivery_gate(clips, missing, probes=FakeProbes(facts))
        concat = _by_check(report.final_checks)["concat_success"]
        assert concat.passed is False and concat.severity == "critical"
        assert report.status == "needs_review"
        assert all(scene.passed for scene in report.scenes)
        assert "Final video: The final video was not produced." in report.summary

        report = run_delivery_gate(clips, None, probes=FakeProbes(facts))
        assert _by_check(report.final_checks)["concat_success"].passed is False

    def test_missing_final_is_rebuilt_when_a_reconcat_is_available(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        final.unlink()
        calls = []

        def reconcat(paths):
            calls.append(1)
            final.write_bytes(b"x")
            return final

        report = run_delivery_gate(clips, final, probes=FakeProbes(facts), reconcat=reconcat)
        assert calls == [1]
        assert report.status == "ready"

    def test_unmeasurable_media_needs_review_and_never_raises(self, tmp_path):
        clips, _, final = _good_run(tmp_path, 2)
        probes = MediaProbes(
            probe=lambda path: None, loudness=lambda path: None,
            silences=lambda path, **kwargs: None, normalize=lambda *args: False,
        )
        report = run_delivery_gate(clips, final, probes=probes, reconcat=lambda paths: final)
        assert report.status == "needs_review"
        assert [scene.passed for scene in report.scenes] == [False, False]
        assert all(check.passed is None for check in report.scenes[0].checks)
        assert _by_check(report.final_checks)["concat_success"].passed is None
        assert report.repair_rounds == 0
        assert report.scenes[1].start_seconds == 0.0

    def test_gate_exceptions_become_a_gate_error(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)

        def explode(path):
            raise OSError("ffprobe went away")

        report = run_delivery_gate(clips, final, probes=MediaProbes(probe=explode))
        assert report.status == "needs_review"
        assert [check.check for check in report.final_checks] == ["gate_error"]
        error = report.final_checks[0]
        assert error.passed is False and error.severity == "medium"
        assert "ffprobe went away" in error.message
        assert "ffprobe went away" in report.summary

        # Garbage arguments are reported the same way.
        assert run_delivery_gate(None, final).final_checks[0].check == "gate_error"
        assert run_delivery_gate(clips, final, thresholds="strict").final_checks[0].check == "gate_error"

    def test_report_round_trips_as_json(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        probes = FakeProbes(facts, loudness={str(clips[0].composed_path): Loudness(-27.0, -9.0)})
        report = run_delivery_gate(clips, final, probes=probes, reconcat=lambda paths: final)
        assert DeliveryReport.model_validate_json(report.model_dump_json()) == report


# ── summary ─────────────────────────────────────────────────────────────

class TestSummary:
    def test_format_timestamp(self):
        assert format_timestamp(0) == "0:00"
        assert format_timestamp(42.4) == "0:42"
        assert format_timestamp(51) == "0:51"
        assert format_timestamp(125.6) == "2:06"
        assert format_timestamp(-3) == "0:00"

    def test_ready_summary_is_one_line(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 12)
        report = run_delivery_gate(clips, final, probes=FakeProbes(facts))
        assert summary_lines(report) == ["Ready: 12/12 scenes passed sync and quality checks."]

    def test_needs_review_names_the_scene_its_time_range_and_worst_failure(self):
        def scene(scene_id, start, end, checks):
            return SceneDelivery(
                scene_id=scene_id, start_seconds=start, end_seconds=end,
                passed=all(check.passed for check in checks), checks=checks,
            )

        ok = DeliveryCheck(check="streams", passed=True, severity="high", message="")
        report = DeliveryReport(
            status="needs_review", summary="",
            scenes=[
                scene("scene_003", 30.0, 42.0, [ok]),
                scene("scene_004", 42.0, 51.0, [
                    ok,
                    DeliveryCheck(check="loudness", passed=False, severity="medium", message="Too quiet."),
                    DeliveryCheck(
                        check="narration_intact", passed=False, severity="high",
                        message="The last 0.40s of the narration is cut off.",
                    ),
                    DeliveryCheck(check="av_duration", passed=None, severity="info", message="Not checked."),
                ]),
            ],
            final_checks=[
                DeliveryCheck(check="concat_success", passed=True, severity="critical", message=""),
                DeliveryCheck(
                    check="duration_drift", passed=False, severity="high",
                    message="The final video is 1.20s off from the sum of its scenes.",
                ),
            ],
        )
        assert summary_lines(report) == [
            "Needs review: 1/2 scenes passed sync and quality checks.",
            "scene_004 (0:42-0:51): The last 0.40s of the narration is cut off.",
            "Final video: The final video is 1.20s off from the sum of its scenes.",
        ]

    def test_gate_summary_matches_summary_lines(self, tmp_path):
        clips, facts, final = _good_run(tmp_path, 2)
        probes = FakeProbes(facts, silences={str(clips[1].audio_path): [(1.0, 5.5)]})
        report = run_delivery_gate(clips, final, probes=probes)
        lines = summary_lines(report)
        assert report.summary == "\n".join(lines)
        assert lines[0] == "Needs review: 1/2 scenes passed sync and quality checks."
        assert lines[1] == "scene_002 (0:06-0:12): The narration has a 4.5s silence 1.0s into the scene."


# ── composition commands (no ffmpeg) ────────────────────────────────────

class TestCompositionCommands:
    @pytest.fixture
    def commands(self, monkeypatch):
        captured: list[list[str]] = []
        monkeypatch.setattr(media, "_run", lambda command, timeout, label: captured.append(command))
        return captured

    def _probe(self, monkeypatch, facts):
        monkeypatch.setattr(media.media_probe, "probe", lambda path: facts.get(Path(path).name))

    def test_longer_narration_freezes_the_last_frame_instead_of_clipping(self, tmp_path, commands, monkeypatch):
        self._probe(monkeypatch, {"v.mp4": _info(video=5.0), "a.mp3": _info(audio=7.4)})
        media.compose_scene(tmp_path / "v.mp4", tmp_path / "a.mp3", tmp_path / "out" / "c.mp4")
        command = commands[0]
        assert "-shortest" not in command
        assert command[command.index("-vf") + 1] == "tpad=stop_mode=clone:stop_duration=2.400"
        assert command[command.index("-af") + 1] == media.NARRATION_FILTER
        assert media.NARRATION_FILTER == "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"
        for flag, value in (("-c:v", "libx264"), ("-crf", "18"), ("-pix_fmt", "yuv420p"),
                            ("-c:a", "aac"), ("-b:a", "192k"), ("-ar", "48000"), ("-ac", "2")):
            assert command[command.index(flag) + 1] == value
        assert (tmp_path / "out").is_dir()

    def test_longer_animation_pads_the_narration_with_silence(self, tmp_path, commands, monkeypatch):
        self._probe(monkeypatch, {"v.mp4": _info(video=8.0), "a.mp3": _info(audio=5.0)})
        media.compose_scene(tmp_path / "v.mp4", tmp_path / "a.mp3", tmp_path / "c.mp4")
        command = commands[0]
        assert "-shortest" not in command and "-vf" not in command
        assert command[command.index("-af") + 1] == media.NARRATION_FILTER + ",apad=whole_dur=8.000"

    def test_gap_within_tolerance_needs_no_padding(self, tmp_path, commands, monkeypatch):
        self._probe(monkeypatch, {"v.mp4": _info(video=6.0), "a.mp3": _info(audio=6.04)})
        media.compose_scene(tmp_path / "v.mp4", tmp_path / "a.mp3", tmp_path / "c.mp4")
        assert "-vf" not in commands[0]
        assert commands[0][commands[0].index("-af") + 1] == media.NARRATION_FILTER

    def test_unprobeable_inputs_keep_the_longer_stream(self, tmp_path, commands, monkeypatch):
        self._probe(monkeypatch, {})
        media.compose_scene(tmp_path / "v.mp4", tmp_path / "a.mp3", tmp_path / "c.mp4")
        assert "-shortest" not in commands[0] and "-vf" not in commands[0]

    def test_concatenate_normalises_every_clip_to_the_most_common_format(self, tmp_path, commands, monkeypatch):
        self._probe(monkeypatch, {
            "a.mp4": _info(video=6.0, audio=6.0),
            "b.mp4": _info(video=4.0, size=(1920, 1080), fps=25.0),
            "c.mp4": _info(video=5.0, audio=5.0),
        })
        media.concatenate([tmp_path / name for name in ("a.mp4", "b.mp4", "c.mp4")], tmp_path / "final.mp4")
        command = commands[0]
        graph = command[command.index("-filter_complex") + 1]
        assert graph.count("scale=1280:720:force_original_aspect_ratio=decrease") == 3
        assert graph.count("pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,fps=30,setsar=1,format=yuv420p") == 3
        assert graph.count("aformat=sample_rates=48000:channel_layouts=stereo") == 2
        assert "anullsrc=r=48000:cl=stereo,atrim=duration=4.000[a1]" in graph
        assert graph.endswith("[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]")
        assert command[command.index("-movflags") + 1] == "+faststart"
        assert command.count("-i") == 3
        assert list(tmp_path.iterdir()) == []

    def test_long_filter_graph_goes_to_a_script_next_to_the_output(self, tmp_path, commands, monkeypatch):
        self._probe(monkeypatch, {"a.mp4": _info(video=6.0, audio=6.0)})
        monkeypatch.setattr(media, "_INLINE_FILTER_LIMIT", 10)
        media.concatenate([tmp_path / "a.mp4"], tmp_path / "run" / "final.mp4")
        command = commands[0]
        script = Path(command[command.index("-/filter_complex") + 1])
        assert script.parent == tmp_path / "run"
        assert script.read_text(encoding="utf-8").endswith("[v0][a0]concat=n=1:v=1:a=1[v][a]")

    def test_filter_script_falls_back_for_older_ffmpeg_only(self, tmp_path, monkeypatch):
        self._probe(monkeypatch, {"a.mp4": _info(video=6.0, audio=6.0)})
        monkeypatch.setattr(media, "_INLINE_FILTER_LIMIT", 10)
        attempts: list[list[str]] = []

        def old_ffmpeg(command, timeout, label):
            attempts.append(command)
            if "-/filter_complex" in command:
                raise RuntimeError("final concatenation failed: Unrecognized option '/filter_complex'.")

        monkeypatch.setattr(media, "_run", old_ffmpeg)
        media.concatenate([tmp_path / "a.mp4"], tmp_path / "final.mp4")
        assert len(attempts) == 2 and "-filter_complex_script" in attempts[1]

        def broken(command, timeout, label):
            attempts.append(command)
            raise RuntimeError("final concatenation failed: Invalid data found when processing input")

        attempts.clear()
        monkeypatch.setattr(media, "_run", broken)
        with pytest.raises(RuntimeError, match="Invalid data"):
            media.concatenate([tmp_path / "a.mp4"], tmp_path / "final.mp4")
        assert len(attempts) == 1

    def test_concatenate_rejects_an_empty_list(self, tmp_path):
        with pytest.raises(RuntimeError, match="No scene videos"):
            media.concatenate([], tmp_path / "final.mp4")


# ── real media (ffmpeg) ─────────────────────────────────────────────────

def _tools_available() -> bool:
    return bool(shutil.which(get_ffmpeg()) and shutil.which(get_ffprobe()))


needs_ffmpeg = pytest.mark.skipif(not _tools_available(), reason="ffmpeg not available")


def _ffmpeg(*args):
    subprocess.run([get_ffmpeg(), "-v", "error", "-y", *args], check=True, timeout=120)


def _video(path, seconds, size="1280x720", rate=30, tone=False):
    inputs = ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate={rate}"]
    audio = []
    if tone:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}"]
        audio = ["-c:a", "aac", "-ar", "44100", "-ac", "1"]
    _ffmpeg(*inputs, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", *audio, str(path))
    return path


def _audio(path, seconds, volume=1.0):
    _ffmpeg(
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-af", f"volume={volume}", str(path),
    )
    return path


@needs_ffmpeg
@pytest.mark.integration
class TestRealMedia:
    def test_longer_narration_is_kept_whole(self, tmp_path):
        video = _video(tmp_path / "v.mp4", 2)
        audio = _audio(tmp_path / "a.wav", 4)
        composed = media.compose_scene(video, audio, tmp_path / "composed.mp4")
        info = media_probe.probe(str(composed))
        assert info.duration == pytest.approx(4.0, abs=0.2)
        assert info.audio.duration >= 4.0 - 0.05
        assert info.video.duration == pytest.approx(4.0, abs=0.2)
        assert (info.audio.codec, info.audio.sample_rate, info.audio.channels) == ("aac", 48000, 2)
        assert (info.video.codec, info.video.pix_fmt) == ("h264", "yuv420p")
        # The narration is still there at the end, not padded silence.
        silences = media_probe.detect_silences(str(composed), min_duration=0.5, total=info.duration)
        assert silences == []

    def test_longer_animation_is_kept_whole(self, tmp_path):
        video = _video(tmp_path / "v.mp4", 5)
        audio = _audio(tmp_path / "a.mp3", 2)
        composed = media.compose_scene(video, audio, tmp_path / "composed.mp4")
        info = media_probe.probe(str(composed))
        assert info.duration == pytest.approx(5.0, abs=0.2)
        assert info.video.duration == pytest.approx(5.0, abs=0.1)
        assert info.audio.duration == pytest.approx(5.0, abs=0.2)

    def test_compose_failure_raises(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        with pytest.raises(RuntimeError, match="scene composition failed"):
            media.compose_scene(broken, _audio(tmp_path / "a.wav", 1), tmp_path / "composed.mp4")

    def test_mixed_clips_join_into_one_uniform_file(self, tmp_path):
        clips = tmp_path / "clips"
        clips.mkdir()
        first = _video(clips / "a.mp4", 2, tone=True)
        second = _video(clips / "b.mp4", 3, size="1920x1080", rate=25)  # no audio
        third = _video(clips / "c.mp4", 2, tone=True)
        out = tmp_path / "run"
        final = media.concatenate([first, second, third], out / "final.mp4")
        info = media_probe.probe(str(final))
        assert info.duration == pytest.approx(7.0, abs=0.3)
        assert (info.video.width, info.video.height, info.video.fps) == (1280, 720, 30.0)
        assert (info.video.codec, info.video.pix_fmt) == ("h264", "yuv420p")
        assert (info.audio.codec, info.audio.sample_rate, info.audio.channels) == ("aac", 48000, 2)
        assert abs(info.audio.duration - info.video.duration) <= 0.3
        assert media.inspect_av(final)["valid"] is True
        assert [path.name for path in out.iterdir()] == ["final.mp4"]
        assert sorted(path.name for path in clips.iterdir()) == ["a.mp4", "b.mp4", "c.mp4"]

    def test_single_clip_and_filter_script_path(self, tmp_path, monkeypatch):
        clip = _video(tmp_path / "a.mp4", 2, size="1920x1080", rate=25)
        monkeypatch.setattr(media, "_INLINE_FILTER_LIMIT", 10)
        final = media.concatenate([clip], tmp_path / "run" / "final.mp4")
        info = media_probe.probe(str(final))
        assert info.duration == pytest.approx(2.0, abs=0.2)
        assert (info.video.width, info.video.height) == (1920, 1080)
        assert info.audio is not None and info.audio.sample_rate == 48000
        assert (tmp_path / "run" / "concat_filter.txt").is_file()

    def test_gate_on_real_clips(self, tmp_path):
        clips = []
        for index, (video_seconds, audio_seconds) in enumerate([(4, 4), (3, 5)], start=1):
            scene_dir = tmp_path / f"scene_{index:03d}"
            scene_dir.mkdir()
            render = _video(scene_dir / "render.mp4", video_seconds)
            audio = _audio(scene_dir / "narration.wav", audio_seconds, volume=0.2)
            composed = media.compose_scene(render, audio, scene_dir / "final_with_narration.mp4")
            clips.append(SceneClip(
                scene_id=f"scene_{index:03d}", composed_path=composed, render_path=render,
                audio_path=audio, narration_seconds=float(audio_seconds),
            ))
        final = media.concatenate([clip.composed_path for clip in clips], tmp_path / "final.mp4")

        report = run_delivery_gate(
            clips, final,
            recompose=lambda clip: media.compose_scene(clip.render_path, clip.audio_path, clip.composed_path),
            reconcat=lambda paths: media.concatenate(paths, tmp_path / "final.mp4"),
        )

        assert report.status in {"ready", "needs_review"}
        assert [check.check for check in report.final_checks if check.check == "gate_error"] == []
        assert [scene.scene_id for scene in report.scenes] == ["scene_001", "scene_002"]
        for scene in report.scenes:
            checks = _by_check(scene.checks)
            assert checks["streams"].passed is True
            assert checks["narration_intact"].passed is True
            assert checks["narration_gaps"].passed is True
        # Scene 2's render is 2 s shorter than its narration: a frozen tail that
        # recomposing cannot fix, so the video is held for review.
        frozen = _by_check(report.scenes[1].checks)["av_duration"]
        assert frozen.passed is False and frozen.severity == "high"
        assert report.status == "needs_review"
        assert report.scenes[1].end_seconds == pytest.approx(9.0, abs=0.3)
        final_checks = _by_check(report.final_checks)
        for name in ("concat_success", "duration_drift", "av_drift"):
            assert final_checks[name].passed is True
        assert "scene_002 (0:04-0:09)" in report.summary

    def test_quiet_real_clip_is_renormalized(self, tmp_path):
        # A clip made outside compose_scene, 20 dB too quiet.
        quiet = tmp_path / "quiet.mp4"
        _ffmpeg(
            "-f", "lavfi", "-i", "testsrc=duration=6:size=640x360:rate=30",
            "-f", "lavfi", "-i", "anoisesrc=duration=6:color=pink:amplitude=0.02",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", str(quiet),
        )
        clip = SceneClip(scene_id="scene_001", composed_path=quiet)
        before = _by_check(check_scene(clip))["loudness"]
        assert before.passed is False and before.repair == "renormalize"

        final = tmp_path / "final.mp4"
        report = run_delivery_gate(
            [clip], final, reconcat=lambda paths: media.concatenate(paths, final),
        )
        assert report.scenes[0].repairs_applied == ["renormalize"]
        assert _by_check(report.scenes[0].checks)["loudness"].passed is True
        assert not quiet.with_name("quiet.norm.mp4").exists()
        assert _by_check(report.final_checks)["concat_success"].passed is True


def test_module_does_not_import_the_legacy_gate():
    source = Path(delivery.__file__).read_text(encoding="utf-8")
    assert "src.agents" not in source and "src.orchestrator" not in source and "src.schemas" not in source
