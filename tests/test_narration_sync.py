"""P2b: beat-level timing track, render-time sync, beat alignment, regeneration."""

import subprocess
from types import SimpleNamespace

import pytest

from src.agents.animation_generator import AnimationCodeGeneratorAgent
from src.agents.delivery_gate import DeliveryGate, SceneMedia, _Probes, word_error_rate
from src.agents.tts_agent import TTSAgent, TTSOutput
from src.agents.video_renderer import SceneRenderResult, VideoRendererAgent
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.schemas.scene_processing import SceneSpec
from src.schemas.storyboard import SceneState, VisualBeat, VisualIntent
from src.schemas.video_script import Beat
from src.utils import media_probe
from src.utils.ffmpeg_finder import get_ffmpeg
from src.utils.narration_sync import sync_to, sync_to_end
from src.utils.sync_contract import SYNC_IMPORT, inject_sync_contract, lint_sync_calls


def _tool(cmd) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _tool([get_ffmpeg(), "-version"]), reason="ffmpeg not available")
needs_manim = pytest.mark.skipif(not _tool(["manim", "--version"]), reason="manim not available")

SCENE = '''class Scene001(Scene):
    """Doc."""
    def construct(self):
        sync_to(self, "beat_001")
        self.play(FadeIn(Square()))
        sync_to(self, "beat_002")
        self.wait(1)
'''


class TestSyncContract:
    def test_inject_is_idempotent_and_ends_with_sync_to_end(self):
        once = inject_sync_contract(SCENE, {"beat_001": 0.0, "beat_002": 4.25}, 9.5)
        assert "BEAT_TIMES = {'beat_001': 0.0, 'beat_002': 4.25}" in once
        assert once.index('"""Doc."""') < once.index("BEAT_TIMES")
        assert once.rstrip().endswith("sync_to_end(self)")
        again = inject_sync_contract(once, {"beat_001": 0.0, "beat_002": 5.0}, 11.0)
        assert again.count("BEAT_TIMES") == 1 and "NARRATION_END = 11.0" in again
        assert again.count("sync_to_end(self)") == 1

    def test_lint(self):
        good = inject_sync_contract(SCENE, {"beat_001": 0.0, "beat_002": 4.0}, 8.0)
        assert lint_sync_calls(good, ["beat_001", "beat_002"]) == []
        issues = lint_sync_calls(SCENE, ["beat_002", "beat_001", "beat_003"])
        assert any("beat_003" in i for i in issues)
        assert any("out of narration order" in i for i in issues)
        assert any("sync_to_end" in i for i in issues)

    def test_unparseable_code_is_left_alone(self):
        assert inject_sync_contract("class (:", {"b": 0.0}, 1.0) == "class (:"


class TestRuntimeHelper:
    class FakeScene:
        BEAT_TIMES = {"beat_001": 0.0, "beat_002": 4.0}
        NARRATION_END = 9.0

        def __init__(self):
            self.time = 0.0
            self.waits = []

        def wait(self, seconds):
            self.waits.append(round(seconds, 3))
            self.time += seconds

    def test_waits_until_beat_and_end(self):
        scene = self.FakeScene()
        scene.time = 1.5
        assert sync_to(scene, "beat_002") == pytest.approx(2.5)
        assert sync_to(scene, "beat_001") == 0.0      # already past: never rewinds
        assert sync_to(scene, "unknown") == 0.0
        assert sync_to_end(scene) == pytest.approx(5.0)
        assert scene.waits == [2.5, 5.0]


class TestGeneratorContract:
    def _spec(self, **extra):
        intent = VisualIntent(
            mathematical_objects=[], transformations=[], beats=[],
            scene_state_in=SceneState(), scene_state_out=SceneState(), emphasis_points=[],
        )
        return SceneSpec(
            scene_id="scene_004", narration="Words here.", visual_intent=intent,
            duration_estimate=8.0, **extra,
        )

    def test_timing_track_is_injected_after_generation(self):
        generator = AnimationCodeGeneratorAgent()
        spec = self._spec(audio_duration_seconds=9.0, beat_timings={"beat_007": [0, 4], "beat_008": [4, 9]})
        code = generator._apply_sync_contract(
            "class Scene004(Scene):\n    def construct(self):\n        self.wait(1)\n", spec,
        )
        assert "BEAT_TIMES = {'beat_007': 0.0, 'beat_008': 4.0}" in code
        assert "NARRATION_END = 9.0" in code and code.rstrip().endswith("sync_to_end(self)")
        assert "NARRATION SYNC" in generator._build_sync_block(spec)
        assert SYNC_IMPORT in generator._get_imports()

    def test_no_track_without_measured_narration(self):
        generator = AnimationCodeGeneratorAgent()
        spec = self._spec(beat_timings={"beat_007": [0, 4]})  # timings but no audio yet
        code = "class Scene004(Scene):\n    def construct(self):\n        self.wait(1)\n"
        assert generator._apply_sync_contract(code, spec) == code
        assert generator._build_sync_block(spec) == ""


class TestOrchestratorTimingTrack:
    def test_tts_receives_beats_and_scenes_get_the_track(self, tmp_path):
        orchestrator = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
        beats = [
            Beat(beat_id="beat_001", narration="First idea.", visual_hint="x"),
            Beat(beat_id="beat_002", narration="Second idea.", visual_hint="y"),
        ]
        scene = SimpleNamespace(
            scene_id="scene_001", narration="First idea.\n\nSecond idea.",
            source_beats=beats, beat_timings={},
        )
        seen = {}

        def tts(scenes, output_dir, quality="standard", voice="alloy"):
            seen["beats"] = scenes[0].get("beats")
            return [TTSOutput("a.mp3", 5.0, "scene_001", beat_timings={"beat_001": [0, 2], "beat_002": [2, 5]})]

        orchestrator.tts_agent.process_all_scenes = tts
        orchestrator._run_tts_generation(
            SimpleNamespace(scenes=[scene]), SimpleNamespace(output_directory=str(tmp_path)), None,
        )
        assert seen["beats"] == [("beat_001", "First idea."), ("beat_002", "Second idea.")]
        assert scene.beat_timings == {"beat_001": [0, 2], "beat_002": [2, 5]}

    def test_measured_track_sets_visual_beat_windows(self, tmp_path):
        orchestrator = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path), tts_first=True))
        intent = VisualIntent(
            mathematical_objects=[], transformations=[],
            beats=[
                VisualBeat(beat_id="beat_001", narration_excerpt="a", start_time=0, end_time=3),
                VisualBeat(beat_id="beat_002", narration_excerpt="b", start_time=3, end_time=6),
            ],
            scene_state_in=SceneState(), scene_state_out=SceneState(), emphasis_points=[],
        )
        scene = SimpleNamespace(
            scene_id="scene_001", visual_intent=intent, duration_estimate=6.0,
            audio_duration_seconds=None, beat_timings={"beat_001": [0.0, 4.2], "beat_002": [4.2, 9.0]},
        )
        orchestrator._apply_tts_timing(
            SimpleNamespace(scenes=[scene]), [TTSOutput("a.mp3", 9.0, "scene_001")],
        )
        assert [(b.start_time, b.end_time) for b in intent.beats] == [(0.0, 4.2), (4.2, 9.0)]


class TestBeatAlignmentAndRegeneration:
    def _media(self, tmp_path):
        files = {}
        for kind in ("video", "audio", "composed"):
            files[kind] = tmp_path / f"{kind}.bin"
            files[kind].write_bytes(b"x")
        return SceneMedia(
            "scene_001", 0, str(files["video"]), str(files["audio"]), str(files["composed"]),
            narration="Alpha beta. Gamma delta.",
            beats=[("beat_001", 0.0, 3.0, "Alpha beta."), ("beat_002", 3.0, 6.0, "Gamma delta.")],
        )

    def _probes(self, media, activity):
        info = media_probe.MediaInfo(
            duration=6.0,
            video=media_probe.StreamInfo(codec="h264", duration=6.0, width=8, height=8, fps=4, pix_fmt="yuv420p"),
            audio=media_probe.StreamInfo(codec="aac", duration=6.0, sample_rate=48000, channels=1),
        )
        state = {"activity": activity}
        return state, _Probes(
            probe=lambda p: info,
            loudness=lambda p: media_probe.Loudness(-16.0, -3.0),
            silences=lambda p, **kw: [],
            activity=lambda p: state["activity"],
            frame_review=lambda sid, p: [],
        )

    def test_static_beat_is_regenerated_with_feedback(self, tmp_path):
        media = self._media(tmp_path)
        moving_then_still = [0.0] + [0.01] * 11 + [0.0] * 12   # 4 fps over 6 s
        state, probes = self._probes(media, moving_then_still)
        calls = []

        def regenerate(scene_id, feedback):
            calls.append((scene_id, feedback))
            state["activity"] = [0.0] + [0.01] * 23           # the new code moves
            return SceneRenderResult(scene_id, True, media.video_path, 6.0)

        composer = SimpleNamespace(_compose_scene=lambda pair, d: SimpleNamespace(
            scene_id=pair.scene_id, success=True, composed_video_path=media.composed_path, final_duration=6.0,
        ))
        gate = DeliveryGate(composer=composer, probes=probes, regenerate=regenerate)
        before = {c.check: c for c in gate.check_scene(media)}["beat_alignment"]
        assert before.passed is False and before.measured["static_beats"] == ["beat_002"]
        assert 'Nothing moves on screen for 3.0s while the narration says "Gamma delta."' == before.message

        storyboard = SimpleNamespace(scenes=[SimpleNamespace(
            scene_id="scene_001", narration=media.narration,
            beat_timings={"beat_001": [0.0, 3.0], "beat_002": [3.0, 6.0]},
            source_beats=[SimpleNamespace(beat_id="beat_001", narration="Alpha beta."),
                          SimpleNamespace(beat_id="beat_002", narration="Gamma delta.")],
        )])
        result = gate.run(
            storyboard, str(tmp_path),
            SimpleNamespace(scene_results=[SceneRenderResult("scene_001", True, media.video_path, 6.0)]),
            [TTSOutput(media.audio_path, 6.0, "scene_001")],
            SimpleNamespace(final_video_path=None, total_duration=6.0, scene_results=[
                SimpleNamespace(scene_id="scene_001", success=True, composed_video_path=media.composed_path, final_duration=6.0),
            ]),
        )
        assert len(calls) == 1 and 'sync_to(self, "beat_002")' in calls[0][1][0]
        assert result.scenes[0].repairs_applied == ["regenerate"] and result.scenes[0].passed

    def test_regeneration_budget_is_enforced(self, tmp_path):
        media = self._media(tmp_path)
        _, probes = self._probes(media, [0.0] * 24)
        gate = DeliveryGate(probes=probes, regenerate=lambda *a: None, max_regenerations=0)
        assert gate._regenerate(media, gate.check_scene(media), str(tmp_path), None, None) is False

    def test_fidelity_check_uses_word_error_rate(self, tmp_path):
        media = self._media(tmp_path)
        _, probes = self._probes(media, [0.0] + [0.01] * 23)
        probes.transcribe = lambda path: "alpha beta gamma"
        check = {c.check: c for c in DeliveryGate(probes=probes).check_scene(media)}["narration_fidelity"]
        assert check.passed is False and check.measured["wer"] == 0.25
        assert word_error_rate("Hello, world!", "hello world") == 0.0


@needs_ffmpeg
class TestPerBeatTTS:
    def _client(self, tmp_path):
        made = []

        def create(model, voice, input, response_format):
            seconds = max(0.5, len(input.split()) * 0.4)
            path = tmp_path / f"tts_{len(made)}.mp3"
            subprocess.run([
                get_ffmpeg(), "-v", "error", "-y", "-f", "lavfi",
                "-i", f"sine=frequency=300:duration={seconds}:sample_rate=24000",
                "-ac", "1", "-c:a", "libmp3lame", str(path),
            ], check=True)
            made.append(input)
            return SimpleNamespace(content=path.read_bytes())

        return SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create))), made

    def test_timing_track_matches_the_joined_audio(self, tmp_path):
        agent = TTSAgent()
        agent.client, made = self._client(tmp_path)
        path, timings = agent._generate_beat_audio(
            [("beat_001", "one two three four five"), ("beat_002", "six seven eight")],
            "scene_001", str(tmp_path / "out"), "tts-1", "alloy",
            pauses={"beat_001": 1.0},
        )
        total = media_probe.probe(path).duration
        assert made == ["one two three four five", "six seven eight"]
        assert timings["beat_001"][0] == 0.0
        assert timings["beat_002"][1] == pytest.approx(total, abs=0.05)
        # The pause after beat 1 sits between the beats: its window ends with
        # its ~2.0s of speech, and beat 2 starts after the 1.0s of silence.
        assert timings["beat_001"][1] == pytest.approx(2.0, abs=0.15)
        assert timings["beat_002"][0] == pytest.approx(3.0, abs=0.15)


@needs_ffmpeg
@needs_manim
@pytest.mark.slow
def test_real_render_honours_the_timing_track_and_flags_a_static_beat(tmp_path):
    code = inject_sync_contract(
        "class Scene001(Scene):\n    def construct(self):\n"
        '        sync_to(self, "beat_001")\n'
        "        self.play(FadeIn(Square(side_length=3, fill_opacity=0.8)), run_time=1)\n"
        '        sync_to(self, "beat_002")\n'
        '        sync_to(self, "beat_003")\n'
        "        self.play(Rotate(self.mobjects[0], PI / 2), run_time=1)\n",
        {"beat_001": 0.0, "beat_002": 2.0, "beat_003": 4.0}, 6.0,
    )
    scene_file = tmp_path / "scene_scene_001.py"
    scene_file.write_text(f"from manim import *\n{SYNC_IMPORT}\n\n{code}", encoding="utf-8")
    renderer = VideoRendererAgent()
    result = renderer._render_scene(str(scene_file), renderer.quality_flag_for("low"), str(tmp_path / "media"))
    assert result.success, result.error
    assert media_probe.probe(result.video_path).video.duration == pytest.approx(6.0, abs=0.2)

    media = SceneMedia(
        "scene_001", 0, video_path=result.video_path,
        beats=[("beat_001", 0.0, 2.0, "Here is a square."), ("beat_002", 2.0, 4.0, "Nothing happens."),
               ("beat_003", 4.0, 6.0, "Now it turns.")],
    )
    check = DeliveryGate()._check_beat_alignment(media)
    assert check.passed is False and check.measured["static_beats"] == ["beat_002"]
    assert check.repair == "regenerate"
