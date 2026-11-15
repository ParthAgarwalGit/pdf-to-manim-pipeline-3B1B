"""Pre-delivery quality gate (P2a): checks, bounded repairs, delivery report.

Unit tests drive the gate with fake probes. The fixture tests build real
media with ffmpeg (and one real manim render); each fixture must fail its
check and pass after its mapped repair. They skip when the tools are absent.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.audio_video_composer import (
    AudioVideoComposerAgent,
    AudioVideoComposerInput,
    SceneMediaPair,
)
from src.agents.delivery_gate import DeliveryGate, SceneMedia, _Probes, format_timestamp
from src.agents.tts_agent import TTSOutput
from src.agents.video_renderer import SceneRenderResult, VideoRendererAgent
from src.utils import media_probe
from src.utils.ffmpeg_finder import get_ffmpeg
from src.utils.media_probe import Loudness, MediaInfo, StreamInfo
from src.utils.scene_retime import adjust_waits


# ── fakes ───────────────────────────────────────────────────────────────

def _info(video=None, audio=None, duration=None):
    return MediaInfo(
        duration=duration if duration is not None else max(video or 0, audio or 0),
        video=StreamInfo(codec="h264", duration=video, width=854, height=480, fps=30.0, pix_fmt="yuv420p")
        if video is not None else None,
        audio=StreamInfo(codec="aac", duration=audio, sample_rate=48000, channels=1)
        if audio is not None else None,
    )


class FakeProbes(_Probes):
    """Media facts keyed by path; mutate ``facts`` to simulate a repair."""

    def __init__(self, facts, loudness=None, silences=None):
        self.facts = facts
        self.loud = loudness or {}
        self.quiet = silences or {}
        super().__init__(
            probe=lambda p: self.facts.get(str(p)),
            loudness=lambda p: self.loud.get(str(p), Loudness(-16.0, -3.0)),
            silences=lambda p, **kw: self.quiet.get(str(p), []),
            normalize=self._normalize,
        )

    def _normalize(self, src, dst, measured):
        Path(dst).write_bytes(b"normalized")
        self.loud[str(src)] = Loudness(-16.0, -2.0)
        return True


def _media(tmp_path, scene_id="scene_001", index=0):
    paths = {}
    for kind in ("video", "audio", "composed"):
        path = tmp_path / f"{scene_id}_{kind}"
        path.write_bytes(b"x")
        paths[kind] = str(path)
    return SceneMedia(
        scene_id=scene_id, index=index,
        video_path=paths["video"], audio_path=paths["audio"], composed_path=paths["composed"],
    )


def _scene_facts(m, video, audio, composed_audio=None):
    return {
        m.video_path: _info(video=video),
        m.audio_path: _info(audio=audio, duration=audio),
        m.composed_path: _info(video=max(video, audio), audio=composed_audio if composed_audio is not None else audio),
    }


def _by_check(checks):
    return {c.check: c for c in checks}


# ── checks ──────────────────────────────────────────────────────────────

class TestSceneChecks:
    def test_frozen_tail_fails_and_maps_to_retime(self, tmp_path):
        m = _media(tmp_path)
        gate = DeliveryGate(probes=FakeProbes(_scene_facts(m, video=5.0, audio=7.4)))
        check = _by_check(gate.check_scene(m))["av_duration"]
        assert check.passed is False and check.severity == "high" and check.repair == "retime"
        assert "2.4s before the narration" in check.message
        assert check.measured["gap_s"] == pytest.approx(2.4)

    def test_small_gap_passes_and_long_video_is_milder(self, tmp_path):
        m = _media(tmp_path)
        assert _by_check(DeliveryGate(probes=FakeProbes(_scene_facts(m, 6.0, 6.25))).check_scene(m))[
            "av_duration"].passed is True
        long_video = _by_check(DeliveryGate(probes=FakeProbes(_scene_facts(m, 8.0, 7.2))).check_scene(m))
        assert long_video["av_duration"].passed is False
        assert long_video["av_duration"].severity == "low"

    def test_clipped_narration_maps_to_recompose(self, tmp_path):
        m = _media(tmp_path)
        check = _by_check(DeliveryGate(probes=FakeProbes(
            _scene_facts(m, 6.0, 6.4, composed_audio=6.0),
        )).check_scene(m))["narration_intact"]
        assert check.passed is False and check.repair == "recompose"
        assert "0.40s of the narration is cut off" in check.message

    def test_silence_and_loudness(self, tmp_path):
        m = _media(tmp_path)
        probes = FakeProbes(
            _scene_facts(m, 6.0, 6.0),
            loudness={m.composed_path: Loudness(-24.0, -9.0)},
            silences={m.audio_path: [(2.0, 4.5)]},
        )
        checks = _by_check(DeliveryGate(probes=probes).check_scene(m))
        assert checks["narration_gaps"].passed is False
        assert "2.5s silence 2.0s into the scene" in checks["narration_gaps"].message
        assert checks["loudness"].passed is False and checks["loudness"].repair == "renormalize"

    def test_missing_narration_and_unmeasurable_media(self, tmp_path):
        m = _media(tmp_path)
        m.audio_path = None
        assert _by_check(DeliveryGate(probes=FakeProbes({})).check_scene(m))[
            "narration_present"].passed is False
        m2 = _media(tmp_path, "scene_002")
        checks = _by_check(DeliveryGate(probes=FakeProbes({})).check_scene(m2))
        assert checks["av_duration"].passed is None  # never counted as a pass


class TestReport:
    def _result(self, tmp_path, facts_fn, media_list, review=None):
        facts = {}
        for m, (v, a) in zip(media_list, facts_fn):
            facts.update(_scene_facts(m, v, a))
        final = tmp_path / "final.mp4"
        final.write_bytes(b"x")
        total = sum(max(v, a) for v, a in facts_fn)
        facts[str(final)] = _info(video=total, audio=total, duration=total)
        gate = DeliveryGate(probes=FakeProbes(facts), max_rounds=0)
        storyboard = SimpleNamespace(scenes=[SimpleNamespace(scene_id=m.scene_id) for m in media_list])
        video_output = SimpleNamespace(scene_results=[
            SceneRenderResult(m.scene_id, True, m.video_path) for m in media_list if m.video_path
        ])
        tts = [TTSOutput(m.audio_path, 1.0, m.scene_id) for m in media_list]
        composition = SimpleNamespace(
            final_video_path=str(final),
            scene_results=[
                SimpleNamespace(scene_id=m.scene_id, success=bool(m.composed_path), composed_video_path=m.composed_path)
                for m in media_list
            ],
        )
        result = gate.run(storyboard, str(tmp_path), video_output, tts, composition)
        return gate.report(result, review)

    def test_ready_when_everything_passes(self, tmp_path):
        media = [_media(tmp_path, f"scene_00{i}", i) for i in (1, 2)]
        report = self._result(tmp_path, [(6.0, 6.1), (5.0, 5.0)], media)
        assert report.status == "ready"
        assert report.summary == "Ready: 2/2 scenes passed sync and quality checks."
        assert [(s.start, s.end) for s in report.scenes] == [(0.0, 6.1), (6.1, 11.1)]

    def test_needs_review_names_scene_and_timestamps(self, tmp_path):
        media = [_media(tmp_path, f"scene_00{i}", i) for i in (1, 2, 3)]
        media[2].composed_path = None  # never made it into the video
        report = self._result(tmp_path, [(6.0, 6.0), (4.0, 7.0), (5.0, 5.0)], media)
        assert report.status == "needs_review"
        lines = report.summary.splitlines()
        assert lines[0] == "Needs review: 1/3 scenes passed sync and quality checks."
        assert "Scene 2 (0:06-0:13): The animation ends 3.0s before the narration" in lines[1]
        assert lines[2] == "Scene 3 is missing from the video."

    def test_blocking_review_finding_prevents_ready(self, tmp_path):
        media = [_media(tmp_path)]
        finding = SimpleNamespace(
            issue="Rendered video is mostly empty.", severity="critical",
            scene_id="scene_001", suspected_stage="animation_generator",
        )
        minor = SimpleNamespace(issue="Layout risk.", severity="medium", scene_id="scene_001", suspected_stage="x")
        report = self._result(tmp_path, [(6.0, 6.0)], media, review=[finding, minor])
        assert report.status == "needs_review"
        assert "Rendered video is mostly empty." in report.summary
        assert "Layout risk" not in report.summary

    def test_timestamp_format(self):
        assert format_timestamp(0) == "0:00"
        assert format_timestamp(71.6) == "1:12"
        assert format_timestamp(None) == "?"


class TestRepairRouting:
    def _setup(self, tmp_path, video, audio):
        m = _media(tmp_path)
        scene_file = tmp_path / "scene_scene_001.py"
        scene_file.write_text(
            "from manim import *\n\nclass Scene001(Scene):\n"
            "    def construct(self):\n        self.add(Circle())\n        self.wait(1)\n",
            encoding="utf-8",
        )
        probes = FakeProbes(_scene_facts(m, video, audio))
        calls = {"render": [], "compose": []}
        new_video = tmp_path / "rerendered.mp4"
        new_video.write_bytes(b"x")

        def render(path, flag, media_dir):
            calls["render"].append(Path(path).read_text(encoding="utf-8"))
            probes.facts[str(new_video)] = _info(video=audio)  # re-timed to the narration
            return SceneRenderResult("scene_001", True, str(new_video), audio)

        def compose(pair, composed_dir):
            calls["compose"].append(pair.video_path)
            probes.facts[m.composed_path] = _info(video=audio, audio=audio)
            return SimpleNamespace(
                scene_id=pair.scene_id, success=True, composed_video_path=m.composed_path, final_duration=audio,
            )

        renderer = SimpleNamespace(
            _render_scene=render, quality_flag_for=lambda q: "-ql", _concatenate_videos=lambda *a: None,
        )
        composer = SimpleNamespace(
            _compose_scene=compose, _concatenate_videos=lambda *a: None,
            _concatenate_with_reencode=lambda *a: None,
        )
        gate = DeliveryGate(renderer=renderer, composer=composer, probes=probes)
        outputs = (
            SimpleNamespace(scene_results=[SceneRenderResult("scene_001", True, m.video_path, video)]),
            [TTSOutput(m.audio_path, audio, "scene_001")],
            SimpleNamespace(
                final_video_path=None, total_duration=audio,
                scenes_composed=1, scenes_skipped=0,
                scene_results=[SimpleNamespace(scene_id="scene_001", success=True, composed_video_path=m.composed_path, final_duration=audio)],
            ),
        )
        storyboard = SimpleNamespace(scenes=[SimpleNamespace(scene_id="scene_001")])
        return gate, storyboard, outputs, calls, scene_file

    def test_frozen_tail_is_retimed_rerendered_and_recomposed(self, tmp_path):
        gate, storyboard, (video_out, tts, comp), calls, scene_file = self._setup(tmp_path, 4.0, 6.5)
        result = gate.run(storyboard, str(tmp_path), video_out, tts, comp)
        scene = result.scenes[0]
        assert scene.repairs_applied == ["retime"] and scene.passed
        assert "self.wait(3.5)" in calls["render"][0]  # 1s hold + 2.5s gap
        assert video_out.scene_results[0].video_path.endswith("rerendered.mp4")
        assert result.repair_rounds == 1

    def test_passing_scene_is_never_touched(self, tmp_path):
        gate, storyboard, (video_out, tts, comp), calls, _ = self._setup(tmp_path, 6.0, 6.1)
        result = gate.run(storyboard, str(tmp_path), video_out, tts, comp)
        assert calls == {"render": [], "compose": []}
        assert result.repair_rounds == 0 and result.scenes[0].passed

    def test_unfixable_retime_is_not_retried(self, tmp_path):
        gate, storyboard, (video_out, tts, comp), calls, scene_file = self._setup(tmp_path, 8.0, 6.0)
        scene_file.write_text(  # no waits left to shorten
            "from manim import *\n\nclass Scene001(Scene):\n    def construct(self):\n"
            "        self.play(Create(Circle()), run_time=8)\n",
            encoding="utf-8",
        )
        result = gate.run(storyboard, str(tmp_path), video_out, tts, comp)
        assert calls["render"] == []
        assert result.scenes[0].passed is False and result.scenes[0].repairs_applied == []

    def test_renormalize_repair(self, tmp_path):
        m = _media(tmp_path)
        probes = FakeProbes(_scene_facts(m, 6.0, 6.0), loudness={m.composed_path: Loudness(-27.0, -10.0)})
        gate = DeliveryGate(probes=probes)
        comp = SimpleNamespace(final_video_path=None, scene_results=[
            SimpleNamespace(scene_id="scene_001", success=True, composed_video_path=m.composed_path),
        ])
        result = gate.run(
            SimpleNamespace(scenes=[SimpleNamespace(scene_id="scene_001")]), str(tmp_path),
            SimpleNamespace(scene_results=[SceneRenderResult("scene_001", True, m.video_path)]),
            [TTSOutput(m.audio_path, 6.0, "scene_001")], comp,
        )
        assert result.scenes[0].repairs_applied == ["renormalize"] and result.scenes[0].passed


class TestRetime:
    def test_adds_tail_wait_when_no_waits(self):
        code = "from manim import *\n\nclass S(Scene):\n    def construct(self):\n        self.add(Circle())\n"
        new, applied = adjust_waits(code, 2.0)
        assert applied == 2.0
        assert new.rstrip().endswith("self.wait(2.0)  # delivery gate: hold to match narration")

    def test_caps_each_wait_and_skips_loops(self):
        code = (
            "class S(Scene):\n    def construct(self):\n        self.wait(3)\n"
            "        for _ in range(4):\n            self.wait(1)\n"
        )
        new, applied = adjust_waits(code, 3.0)
        assert "self.wait(4.0)" in new and new.count("self.wait(1)") == 1
        assert applied == 3.0  # 1.0 into the existing hold, 2.0 as a tail wait

    def test_shrinks_but_never_removes_holds(self):
        code = "class S(Scene):\n    def construct(self):\n        self.wait(1)\n        self.wait(2)\n"
        new, applied = adjust_waits(code, -10.0)
        assert applied == pytest.approx(-2.8)
        assert "self.wait(0.1)" in new


class TestProbeParsers:
    def test_loudnorm_and_silence_parsers(self):
        stderr = (
            'noise\n{\n\t"input_i" : "-23.40",\n\t"input_tp" : "-7.10",\n\t"input_lra" : "3.0",\n'
            '\t"input_thresh" : "-33.5",\n\t"output_i" : "-16.0"\n}\n'
        )
        loud = media_probe.parse_loudnorm_json(stderr)
        assert (loud.integrated, loud.true_peak, loud.lra, loud.threshold) == (-23.4, -7.1, 3.0, -33.5)
        silences = media_probe.parse_silencedetect(
            "[x] silence_start: 1.5\n[x] silence_end: 4 | silence_duration: 2.5\n[x] silence_start: 9\n",
            total=10.0,
        )
        assert silences == [(1.5, 4.0), (9.0, 10.0)]


# ── real-media fixtures ─────────────────────────────────────────────────

def _tools_available() -> bool:
    try:
        return subprocess.run([get_ffmpeg(), "-version"], capture_output=True).returncode == 0
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _tools_available(), reason="ffmpeg not available")


def _ffmpeg(*args):
    subprocess.run([get_ffmpeg(), "-v", "error", "-y", *args], check=True)


def _video(path, seconds, size="320x240"):
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=size={size}:rate=15:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return str(path)


def _audio(path, seconds, volume=0.3):
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}:sample_rate=24000",
            "-af", f"volume={volume}", "-ac", "1", "-c:a", "libmp3lame", str(path))
    return str(path)


def _compose(tmp_path, pairs):
    return AudioVideoComposerAgent().execute(AudioVideoComposerInput(pairs, str(tmp_path)))


def _gate_run(tmp_path, scene_ids, render_results, tts, composition, renderer=None):
    gate = DeliveryGate(renderer=renderer, composer=AudioVideoComposerAgent(), render_quality="low")
    storyboard = SimpleNamespace(scenes=[SimpleNamespace(scene_id=s) for s in scene_ids])
    video_output = SimpleNamespace(scene_results=render_results, final_video_path=str(tmp_path / "silent.mp4"))
    return gate.run(storyboard, str(tmp_path), video_output, tts, composition)


@needs_ffmpeg
class TestRealMediaFixtures:
    def test_clipped_narration_is_recomposed(self, tmp_path):
        video = _video(tmp_path / "v.mp4", 5)
        audio = _audio(tmp_path / "a.mp3", 5.4)
        composition = _compose(tmp_path, [SceneMediaPair("scene_001", video, audio)])
        clipped = composition.scene_results[0].composed_video_path
        # Reproduce the old composer: `-shortest` trimmed narration up to 0.5 s.
        _ffmpeg("-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-shortest", clipped + ".tmp.mp4")
        shutil.move(clipped + ".tmp.mp4", clipped)
        gate = DeliveryGate(composer=AudioVideoComposerAgent())
        media = SceneMedia("scene_001", 0, video, audio, clipped)
        assert _by_check(gate.check_scene(media))["narration_intact"].passed is False

        result = _gate_run(
            tmp_path, ["scene_001"], [SceneRenderResult("scene_001", True, video, 5.0)],
            [TTSOutput(audio, 5.4, "scene_001")], composition,
        )
        checks = _by_check(result.scenes[0].checks)
        assert "recompose" in result.scenes[0].repairs_applied
        assert checks["narration_intact"].passed is True
        # The video is still 0.4 s short of the narration (frozen tail):
        # without a renderer that needs a re-time, so it stays flagged.
        assert checks["av_duration"].passed is False

    def test_quiet_clip_is_renormalized(self, tmp_path):
        video = _video(tmp_path / "v.mp4", 4)
        audio = _audio(tmp_path / "a.mp3", 4)
        composition = _compose(tmp_path, [SceneMediaPair("scene_001", video, audio)])
        composed = composition.scene_results[0].composed_video_path
        _ffmpeg("-i", composed, "-c:v", "copy", "-af", "volume=0.05", "-c:a", "aac", composed + ".q.mp4")
        shutil.move(composed + ".q.mp4", composed)
        assert media_probe.measure_loudness(composed).integrated < -25

        result = _gate_run(
            tmp_path, ["scene_001"], [SceneRenderResult("scene_001", True, video, 4.0)],
            [TTSOutput(audio, 4.0, "scene_001")], composition,
        )
        assert result.scenes[0].repairs_applied == ["renormalize"]
        assert result.scenes[0].passed, result.scenes[0].checks
        assert abs(media_probe.measure_loudness(composed).integrated + 16) <= 1.5

    def test_mismatched_clip_formats_are_reencoded(self, tmp_path):
        v1 = _video(tmp_path / "v1.mp4", 3, size="320x240")
        v2 = _video(tmp_path / "v2.mp4", 3, size="640x360")
        a1, a2 = _audio(tmp_path / "a1.mp3", 3), _audio(tmp_path / "a2.mp3", 3)
        composition = _compose(tmp_path, [SceneMediaPair("scene_001", v1, a1), SceneMediaPair("scene_002", v2, a2)])
        result = _gate_run(
            tmp_path, ["scene_001", "scene_002"],
            [SceneRenderResult("scene_001", True, v1, 3.0), SceneRenderResult("scene_002", True, v2, 3.0)],
            [TTSOutput(a1, 3.0, "scene_001"), TTSOutput(a2, 3.0, "scene_002")], composition,
        )
        checks = _by_check(result.final_checks)
        assert checks["stream_format"].passed is True and checks["stream_format"].measured["reencoded"]
        assert checks["concat_drift"].passed is True
        final = media_probe.probe(composition.final_video_path)
        assert (final.video.width, final.video.height) in {(320, 240), (640, 360)}

    def test_silent_scene_gets_an_audio_track(self, tmp_path):
        video = _video(tmp_path / "v.mp4", 2)
        composition = _compose(tmp_path, [SceneMediaPair("scene_001", video, None)])
        info = media_probe.probe(composition.scene_results[0].composed_video_path)
        assert info.audio is not None and info.audio.sample_rate == 48000


def _manim_available() -> bool:
    try:
        return subprocess.run(["manim", "--version"], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


@needs_ffmpeg
@pytest.mark.slow
@pytest.mark.skipif(not _manim_available(), reason="manim not available")
def test_frozen_tail_is_fixed_by_a_real_retime_and_rerender(tmp_path):
    scene_file = tmp_path / "scene_scene_001.py"
    scene_file.write_text(
        '"""\nScene: scene_001\n"""\n\nfrom manim import *\n\n'
        "class Scene001(Scene):\n    def construct(self):\n"
        "        self.play(FadeIn(Square(side_length=4, fill_opacity=0.9)), run_time=0.5)\n"
        "        self.wait(0.5)\n",
        encoding="utf-8",
    )
    renderer = VideoRendererAgent()
    first = renderer._render_scene(str(scene_file), renderer.quality_flag_for("low"), str(tmp_path / "media"))
    assert first.success, first.error
    audio = _audio(tmp_path / "a.mp3", 3.0)
    composition = _compose(tmp_path, [SceneMediaPair("scene_001", first.video_path, audio)])

    result = _gate_run(
        tmp_path, ["scene_001"], [first], [TTSOutput(audio, 3.0, "scene_001")], composition,
        renderer=renderer,
    )

    scene = result.scenes[0]
    assert scene.repairs_applied == ["retime"]
    assert scene.passed, [c for c in scene.checks if c.passed is not True]
    assert media_probe.probe(result.video_output.scene_results[0].video_path).video.duration == pytest.approx(3.0, abs=0.3)
    assert "self.wait(2.5)" in scene_file.read_text(encoding="utf-8")


@needs_ffmpeg
class TestOrchestratorWiring:
    def _run(self, tmp_path, **config):
        from src.orchestrator.pipeline import Orchestrator, PipelineConfig

        orchestrator = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path), **config))
        orchestrator._run_output_dir = str(tmp_path)
        video = _video(tmp_path / "v.mp4", 3)
        audio = _audio(tmp_path / "a.mp3", 3)
        composition = _compose(tmp_path, [SceneMediaPair("scene_001", video, audio)])
        storyboard = SimpleNamespace(scenes=[SimpleNamespace(scene_id="scene_001")])
        video_output = SimpleNamespace(scene_results=[SceneRenderResult("scene_001", True, video, 3.0)])
        result = orchestrator._run_delivery_gate(
            storyboard, video_output, [TTSOutput(audio, 3.0, "scene_001")], composition,
        )
        manifest = SimpleNamespace(delivery_status=None)
        orchestrator._record_delivery(manifest, result, None)
        return orchestrator, manifest, result

    def test_gate_report_lands_in_quality_reports_and_manifest(self, tmp_path):
        orchestrator, manifest, _ = self._run(tmp_path)
        report = orchestrator._quality_reports["delivery"]
        assert manifest.delivery_status == report["status"] == "ready"
        assert report["summary"] == "Ready: 1/1 scenes passed sync and quality checks."

    def test_gate_can_be_disabled(self, tmp_path):
        orchestrator, manifest, result = self._run(tmp_path, enable_delivery_gate=False)
        assert result is None and manifest.delivery_status is None
        assert "delivery" not in orchestrator._quality_reports
