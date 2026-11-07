"""P5: presenter bridges, TTS/avatar providers, consent, composer integration."""

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agents.bridge_planner import BridgePlannerAgent
from src.agents.delivery_gate import DeliveryGate, _Probes
from src.agents.presenter_agent import AvatarProvider, HeyGenAvatarProvider, PresenterAgent
from src.agents.scene_controller import SceneControllerAgent, SceneControllerInput
from src.agents.script_architect import ScriptArchitectAgent
from src.agents.tts_agent import ElevenLabsTTSProvider, TTSAgent, TTSOutput
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.schemas.professor_profile import ConsentError, ProfessorProfile, require_consent
from src.schemas.storyboard import Configuration, PDFMetadataStoryboard, StoryboardJSON
from src.schemas.video_script import Beat, VideoScript
from src.utils import media_probe
from src.utils.cost_tracker import CostTracker
from src.utils.ffmpeg_finder import get_ffmpeg
from tests.test_revision import _base_run, _beat, _orchestrator


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([get_ffmpeg(), "-version"], capture_output=True).returncode == 0
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


def _long_script(segments=3, beats_per_segment=4, words=45) -> VideoScript:
    beats, n = [], 1
    for s in range(segments):
        for b in range(beats_per_segment):
            beats.append(Beat(
                beat_id=f"beat_{n:03d}", narration=" ".join(["word"] * words), visual_hint="x",
                introduces_concepts=[f"concept_{n}"] if b == 0 else [],
                source_segment_id=f"seg_{s}",
            ))
            n += 1
    return VideoScript(beats=beats)


class TestBridgePlacement:
    def test_policy_caps_and_spacing(self):
        script = _long_script()          # 12 beats x 15 s = 180 s of narration
        positions = BridgePlannerAgent(use_default_llm=False).positions(script.beats)
        kinds = [k for _, k in positions]
        assert kinds[0] == "opening"
        assert len(positions) <= max(1, int(180 * 0.2 / 12))      # <= 20% of runtime
        indices = [i for i, _ in positions]
        gaps = [sum(len(b.narration.split()) / 3 for b in script.beats[a:b]) for a, b in zip(indices, indices[1:])]
        assert all(g >= 90 for g in gaps)                          # ~1 per 90 s

    def test_short_video_gets_only_an_opening(self):
        positions = BridgePlannerAgent(use_default_llm=False).positions(_long_script(1, 2).beats)
        assert positions == [(0, "opening")]

    def test_proposals_are_unlocked_bridge_beats(self):
        script = BridgePlannerAgent(use_default_llm=False).propose(_long_script())
        bridges = [b for b in script.beats if b.role == "bridge"]
        assert bridges and all(not b.locked and b.author == "generated" for b in bridges)
        assert "concept_1" in bridges[0].narration.replace(" ", "_")  # template names what follows
        assert len({b.beat_id for b in script.beats}) == len(script.beats)

    def test_llm_writes_the_bridge_text(self):
        llm = lambda prompt: {"bridges": [{"slot": 0, "text": "Here's why this matters to you."}]}
        script = BridgePlannerAgent(llm=llm).propose(_long_script(1, 2))
        assert script.beats[0].narration == "Here's why this matters to you."


class TestPresenterScenes:
    def _storyboard(self, locked=False):
        script = _long_script(2, 2, words=20)
        script.beats.insert(2, Beat(beat_id="beat_099", narration="Now the real question.", visual_hint="p",
                                    source_segment_id="seg_1", role="bridge", locked=locked))
        agent = ScriptArchitectAgent()
        scenes = agent._chunk_script_into_scenes(script)
        config = Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")
        agent._plan_visual_intents(scenes, script, config, [])
        agent._renumber_scenes(scenes)
        return StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="n.pdf", page_count=1),
            configuration=config,
            concept_hierarchy=agent._concept_hierarchy_from_ledger(script.concept_ledger()),
            scenes=scenes,
        )

    def test_bridge_is_its_own_presenter_scene_without_a_plan(self):
        storyboard = self._storyboard()
        presenter = [s for s in storyboard.scenes if s.is_presenter]
        assert len(presenter) == 1 and [b.beat_id for b in presenter[0].source_beats] == ["beat_099"]
        assert presenter[0].visual_intent.mathematical_objects == []
        assert not presenter[0].presenter_approved
        assert all(not s.is_presenter for s in storyboard.scenes if s is not presenter[0])

    def test_no_manim_code_for_presenter_scenes(self):
        storyboard = self._storyboard(locked=True)
        result = SceneControllerAgent().execute(SceneControllerInput(storyboard=storyboard, output_directory="o"))
        presenter_id = next(s.scene_id for s in storyboard.scenes if s.is_presenter)
        assert presenter_id not in [c.scene_id for c in result.animation_code_objects]
        assert result.report.total_scenes == len(storyboard.scenes) - 1

    def test_unapproved_bridges_are_never_narrated(self, tmp_path):
        storyboard = self._storyboard(locked=False)
        orchestrator = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
        seen = []
        orchestrator.tts_agent.process_all_scenes = lambda scenes, *a, **k: seen.extend(s["scene_id"] for s in scenes) or []
        orchestrator._run_tts_generation(storyboard, SimpleNamespace(output_directory=str(tmp_path)), None)
        presenter_id = next(s.scene_id for s in storyboard.scenes if s.is_presenter)
        assert presenter_id not in seen and len(seen) == len(storyboard.scenes) - 1


PROFILE = {
    "name": "Prof. Ada",
    "consent": {"statement": "I consent to an AI likeness of me presenting my lectures.",
                "signed_by": "Ada", "signed_at": "2026-11-01"},
    "voice": {"provider": "elevenlabs", "id": "voice_123", "verified": True},
    "avatar": {"provider": "heygen", "id": "avatar_456", "verified": False},
}


class TestConsent:
    def test_require_consent(self):
        profile = ProfessorProfile.model_validate(PROFILE)
        assert require_consent(profile, "voice", "elevenlabs").id == "voice_123"
        with pytest.raises(ConsentError, match="identity verification"):
            require_consent(profile, "avatar", "heygen")
        with pytest.raises(ConsentError, match="no openai voice"):
            require_consent(profile, "voice", "openai")
        with pytest.raises(ConsentError, match="profile"):
            require_consent(None, "voice", "elevenlabs")
        unsigned = ProfessorProfile(name="X")
        with pytest.raises(ConsentError, match="consent statement"):
            require_consent(unsigned, "voice", "elevenlabs")

    def test_voice_clone_needs_a_profile_before_any_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        with pytest.raises(ConsentError):
            Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path), tts_provider="elevenlabs"))
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(PROFILE), encoding="utf-8")
        orchestrator = Orchestrator(config=PipelineConfig(
            base_output_dir=str(tmp_path), tts_provider="elevenlabs", professor_profile=str(path),
        ))
        assert orchestrator.tts_agent.provider.voice_id == "voice_123"
        with pytest.raises(ConsentError):  # avatar not verified
            Orchestrator(config=PipelineConfig(
                base_output_dir=str(tmp_path), avatar_provider="heygen", professor_profile=str(path),
            ))._make_avatar_provider()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestVendorClients:
    def test_elevenlabs_request_and_metering(self):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            return _Response(b"mp3-bytes")

        provider = ElevenLabsTTSProvider("secret-key", "voice_123")
        agent = TTSAgent(provider=provider)
        with patch("urllib.request.urlopen", urlopen):
            assert agent._speech_bytes("Hello there.", "tts-1", "alloy") == b"mp3-bytes"
        request = calls[0]
        assert request.full_url.endswith("/v1/text-to-speech/voice_123")
        assert request.get_header("Xi-api-key") == "secret-key"
        assert json.loads(request.data)["text"] == "Hello there."
        assert agent._can_speak() and agent.client is None

    def test_heygen_upload_generate_poll_download(self, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"audio")
        replies = iter([
            {"data": {"id": "asset_1"}},
            {"data": {"video_id": "vid_1"}},
            {"data": {"status": "processing"}},
            {"data": {"status": "completed", "video_url": "https://cdn/x.mp4"}},
        ])
        seen = []

        def urlopen(request, timeout=None):
            url = request if isinstance(request, str) else request.full_url
            seen.append(url)
            if url == "https://cdn/x.mp4":
                return _Response(b"video")
            return _Response(json.dumps(next(replies)).encode())

        provider = HeyGenAvatarProvider("k", "avatar_456", poll_seconds=0)
        with patch("urllib.request.urlopen", urlopen):
            out = provider.render(str(audio), "text", str(tmp_path / "raw.mp4"), 854, 480)
        assert Path(out).read_bytes() == b"video"
        assert seen[0] == HeyGenAvatarProvider.UPLOAD and seen[1] == HeyGenAvatarProvider.GENERATE
        assert "video_id=vid_1" in seen[2]


class _FakeAvatar(AvatarProvider):
    name = "fake"

    def __init__(self, clip):
        self.clip = clip

    def render(self, audio_path, text, out_path, width, height):
        Path(out_path).write_bytes(Path(self.clip).read_bytes())
        return out_path


@needs_ffmpeg
class TestPresenterRendering:
    def test_title_card_matches_manim_format_and_length(self, tmp_path):
        clip = PresenterAgent(render_quality="low").render("scene_002", "Here is the real question.", None, 3.2, str(tmp_path))
        info = media_probe.probe(clip)
        assert (info.video.width, info.video.height, info.video.fps) == (854, 480, 15.0)
        assert info.video.duration == pytest.approx(3.2, abs=0.1) and info.audio is None

    def test_avatar_pip_with_disclosure_and_credits(self, tmp_path):
        previous = tmp_path / "prev.mp4"
        avatar = tmp_path / "avatar.mp4"
        for path, size in ((previous, "854x480"), (avatar, "640x360")):
            subprocess.run([get_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"testsrc=size={size}:rate=15:duration=3", "-pix_fmt", "yuv420p", str(path)], check=True)
        agent = PresenterAgent(_FakeAvatar(avatar), render_quality="low", layout="pip")
        assert agent.disclose
        clip = agent.render("scene_004", "Recap.", None, 2.0, str(tmp_path), previous_video=str(previous))
        assert media_probe.probe(clip).video.duration == pytest.approx(2.0, abs=0.1)
        credits = agent.credits(["Presenter: AI-generated avatar of Prof. Ada, used with consent."], str(tmp_path))
        assert media_probe.probe(credits).video.duration == pytest.approx(4.0, abs=0.1)


@needs_ffmpeg
class TestPresenterInRuns:
    def _script(self, locked):
        script = VideoScript(title="T", beats=[
            _beat(1, "seg_a"), _beat(2, "seg_a"),
            Beat(beat_id="beat_050", narration="Now that we have this, here is the real question.",
                 visual_hint="presenter", source_segment_id="seg_b", role="bridge", locked=locked),
            _beat(3, "seg_b"), _beat(4, "seg_b"),
        ])
        return script

    def test_approved_bridge_renders_in_order_and_is_reused(self, tmp_path):
        run_dir, manifest = _base_run(tmp_path, self._script(locked=True))
        records = {r.scene_id: r for r in manifest.scenes}
        storyboard = json.loads((run_dir / "storyboard.json").read_text("utf-8"))
        presenter_id = next(s["scene_id"] for s in storyboard["scenes"] if s["scene_type"] == "presenter")
        assert records[presenter_id].video_path.startswith("media/presenter/")
        assert (run_dir / records[presenter_id].video_path).is_file()
        report = json.loads((run_dir / "presenter.json").read_text("utf-8"))
        assert report["rendered"] == [presenter_id] and report["seconds"] > 0

        orchestrator = _orchestrator(tmp_path)
        revision = orchestrator.run_revision(str(run_dir), self._script(locked=True))
        rev_records = {r.scene_id: r for r in revision.scenes}
        assert rev_records[presenter_id].reused_video_from == presenter_id
        assert orchestrator.media.narrated == [] and orchestrator.media.generated == []

    def test_unapproved_bridge_is_a_free_proposal(self, tmp_path):
        run_dir, manifest = _base_run(tmp_path, self._script(locked=False))
        report = json.loads((run_dir / "presenter.json").read_text("utf-8"))
        assert report["rendered"] == [] and len(report["proposals"]) == 1
        presenter_id = report["proposals"][0]
        assert presenter_id not in [r.scene_id for r in manifest.scenes if r.audio_path]


class TestGateWithPresenter:
    def test_proposals_do_not_block_ready_and_presenter_skips_motion_checks(self, tmp_path):
        files = {}
        for name in ("v", "a", "c"):
            files[name] = tmp_path / name
            files[name].write_bytes(b"x")
        info = media_probe.MediaInfo(
            duration=4.0,
            video=media_probe.StreamInfo(codec="h264", duration=4.0, width=8, height=8, fps=15, pix_fmt="yuv420p"),
            audio=media_probe.StreamInfo(codec="aac", duration=4.0, sample_rate=48000, channels=1),
        )
        probes = _Probes(
            probe=lambda p: info, loudness=lambda p: media_probe.Loudness(-16.0, -3.0),
            silences=lambda p, **k: [], activity=lambda p: [0.0] * 16,  # a static card
            frame_review=lambda sid, p: [SimpleNamespace(severity="critical", issue="blank")],
        )
        bridge = SimpleNamespace(beat_id="beat_050", narration="Hi.", locked=True)
        proposal = SimpleNamespace(beat_id="beat_051", narration="Maybe.", locked=False)

        def scene(sid, beats, presenter, approved):
            return SimpleNamespace(scene_id=sid, narration="Hi.", source_beats=beats,
                                   beat_timings={b.beat_id: [0.0, 4.0] for b in beats},
                                   is_presenter=presenter, presenter_approved=approved)

        storyboard = SimpleNamespace(scenes=[
            scene("scene_001", [bridge], True, True),
            scene("scene_002", [proposal], True, False),
        ])
        result = DeliveryGate(probes=probes).run(
            storyboard, str(tmp_path),
            SimpleNamespace(scene_results=[SimpleNamespace(scene_id="scene_001", success=True, video_path=str(files["v"]))]),
            [TTSOutput(str(files["a"]), 4.0, "scene_001")],
            SimpleNamespace(final_video_path=None, scene_results=[
                SimpleNamespace(scene_id="scene_001", success=True, composed_video_path=str(files["c"]), final_duration=4.0),
            ]),
        )
        report = DeliveryGate(probes=probes).report(result)
        checks = {c.check for c in result.scenes[0].checks}
        assert "beat_alignment" not in checks and "visual_qa" not in checks
        assert [s.scene_id for s in result.scenes] == ["scene_001"]
        assert result.proposals == ["scene_002"]
        assert "1 presenter bridge(s) await your approval" in report.summary
