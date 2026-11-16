"""Gemini presenter, spoken math, natural-narration prompts and the
render-time layout audit. Hermetic: Gemini and Whisper are faked; clips are
made locally with ffmpeg (skipped without it)."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.frame_director import FrameDirectorAgent
from src.agents.gemini_presenter import (
    MAX_CLIP_SECONDS,
    GeminiOmniPresenter,
    clip_seconds,
    split_for_clips,
)
from src.agents.video_renderer import SceneRenderResult, VideoRendererOutput
from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig
from src.utils.layout_audit import audit_feedback, check, read_audit
from src.utils.speech_text import speakable
from tests.test_did_avatar import _clip, needs_ffmpeg
from tests.test_hybrid import WORDS, _storyboard

LINE = "Every matrix can be unpacked into three simple moves. You'll see the stretch in a moment."


# ── spoken math ─────────────────────────────────────────────────────────

class TestSpeakable:
    @pytest.mark.parametrize("text,spoken", [
        ("In symbols, A = U D V^T.", "In symbols, A equals U D V transpose."),
        ("So σ_1 = ||Av_1||.", "So sigma one equals the norm of Av one."),
        ("The error is σ_{k+1}.", "The error is sigma sub k plus one."),
        ("A^{-1} undoes A, and x^2 grows.", "A inverse undoes A, and x squared grows."),
        ("u_1 v_1^T + u_2 v_2^T", "u one v one transpose plus u two v two transpose"),
    ])
    def test_math_is_read_as_words(self, text, spoken):
        assert speakable(text) == spoken

    def test_plain_prose_is_untouched(self):
        prose = "Each singular value measures how far the sheet stretches - in one direction."
        assert speakable(prose) == prose

    def test_tts_speaks_the_speakable_text(self):
        from src.agents.tts_agent import TTSAgent

        agent = TTSAgent.__new__(TTSAgent)
        heard = []
        agent.provider = SimpleNamespace(synthesize=lambda text: heard.append(text) or b"mp3")
        agent._speech_bytes("A = U D V^T", "tts-1", "nova")
        assert heard == ["A equals U D V transpose"]


# ── natural narration prompts ───────────────────────────────────────────

class TestNarrationPrompt:
    def _prompt(self, **kwargs):
        from src.agents.script_architect import ScriptArchitectAgent
        from src.schemas.source_segment import SourceSegment
        from src.schemas.storyboard import Configuration

        agent = ScriptArchitectAgent.__new__(ScriptArchitectAgent)
        agent.strict, agent.client = False, None
        return agent._build_segment_script_prompt(
            segment=SourceSegment(segment_id="s", order=0, page_start=1, page_end=1, text="word " * 90),
            config=Configuration(verbosity="medium", depth="intermediate", audience="undergraduate"),
            covered_concepts=[], review_feedback=[], **kwargs,
        )

    def test_spoken_style_rules_are_in_every_script_prompt(self):
        prompt = self._prompt()
        assert "SPOKEN NARRATION" in prompt and "let's dive in" in prompt
        assert "Never put symbols such as ^T" in prompt
        assert "NARRATION SO FAR" not in prompt

    def test_each_section_continues_from_the_last_line(self):
        prompt = self._prompt(previous_line="So the stretch only happens along two directions.")
        assert "THE NARRATION SO FAR ENDED WITH" in prompt
        assert "So the stretch only happens along two directions." in prompt


# ── Gemini presenter ────────────────────────────────────────────────────

class FakeOmni:
    """Stands in for client.interactions.create; returns a real local clip."""

    def __init__(self, clip_bytes):
        self.calls = []
        self.interactions = SimpleNamespace(create=self.create)
        self.clip_bytes = clip_bytes

    def create(self, model, input, response_format):
        self.calls.append({"model": model, "input": input, "format": response_format})
        return SimpleNamespace(output_video=SimpleNamespace(data=base64.b64encode(self.clip_bytes).decode()))


@pytest.fixture(scope="module")
def speech_clip(tmp_path_factory):
    from tests.test_did_avatar import _ffmpeg

    if _ffmpeg() is None:
        pytest.skip("ffmpeg is not installed")
    return _clip(tmp_path_factory.mktemp("omni") / "talk.mp4", seconds=2.0).read_bytes()


class TestGeminiPresenterUnits:
    def test_long_text_splits_at_sentences_within_one_clip(self):
        text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(12))
        chunks = split_for_clips(text)
        assert len(chunks) > 1 and all(c.endswith(".") for c in chunks)
        assert all(clip_seconds(c) <= MAX_CLIP_SECONDS for c in chunks)
        assert " ".join(chunks) == text

    def test_clip_length_covers_words_and_tts_timing(self):
        assert clip_seconds("one two three") == 3
        assert clip_seconds(LINE, target=9.2) == 11
        assert clip_seconds(WORDS + WORDS) == MAX_CLIP_SECONDS

    def test_needs_a_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            GeminiOmniPresenter(use_default_transcriber=False)


@needs_ffmpeg
class TestGeminiPresenterClips:
    def test_speaks_the_line_with_a_fictional_persona(self, tmp_path, speech_clip):
        fake = FakeOmni(speech_clip)
        presenter = GeminiOmniPresenter(client=fake, use_default_transcriber=False)
        path, seconds = presenter.speak("So A = U D V^T.", str(tmp_path / "s1.mp4"), target_seconds=4.0)
        assert Path(path).is_file() and seconds == pytest.approx(2.0, abs=0.2)
        call = fake.calls[0]
        assert call["model"] == "gemini-omni-1.2-flash" and call["format"]["duration"] == "4s"  # sized by words
        assert 'says exactly these words, nothing more and nothing less: "So A equals U D V transpose."' in call["input"]
        assert "fictional" in call["input"] and "#1C1C1C" in call["input"]
        assert presenter.reference_image is not None and presenter.reference_image.is_file()

    def test_later_clips_keep_the_same_lecturer(self, tmp_path, speech_clip):
        fake = FakeOmni(speech_clip)
        presenter = GeminiOmniPresenter(client=fake, use_default_transcriber=False)
        presenter.speak(LINE, str(tmp_path / "a.mp4"))
        presenter.speak(LINE, str(tmp_path / "b.mp4"))
        second = fake.calls[1]["input"]
        assert isinstance(second, list) and second[0]["type"] == "image"
        assert second[1]["text"].startswith("The same woman as in the reference image")

    def test_long_narration_becomes_several_joined_clips(self, tmp_path, speech_clip):
        from src.utils.media_probe import probe

        fake = FakeOmni(speech_clip)
        presenter = GeminiOmniPresenter(client=fake, use_default_transcriber=False)
        text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(12))
        path, seconds = presenter.speak(text, str(tmp_path / "long.mp4"))
        assert len(fake.calls) == len(split_for_clips(text)) > 1
        info = probe(path)
        assert info.audio is not None and seconds == pytest.approx(2.0 * len(fake.calls), abs=0.5)

    def test_a_take_that_says_something_else_is_redone_once(self, tmp_path, speech_clip):
        heard = iter(["completely different words here", LINE])
        fake = FakeOmni(speech_clip)
        presenter = GeminiOmniPresenter(client=fake, transcriber=lambda path: next(heard))
        presenter.speak(LINE, str(tmp_path / "c.mp4"))
        assert len(fake.calls) == 2
        assert "Say every word of the quoted text" in fake.calls[1]["input"]
        assert [c["word_error"] for c in presenter.clips][1] == 0.0


@needs_ffmpeg
class TestGeminiInThePipeline:
    def _orch(self, tmp_path, monkeypatch, **config):
        monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
        return Orchestrator(config=PipelineConfig(
            base_output_dir=str(tmp_path), render_backend="hybrid", presenter_engine="gemini",
            render_quality="low", **config,
        ))

    def test_preflight_needs_no_portrait_but_a_gemini_key(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch)
        orch._preflight_render_backend()
        assert orch.config.avatar_provider == "gemini-omni" and orch.config.tts_first
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.setenv(name, "placeholder")
        strict = self._orch(tmp_path, monkeypatch, strict=True)
        monkeypatch.delenv("GEMINI_API_KEY")
        with pytest.raises(PipelineAbortError) as err:
            strict._preflight_render_backend()
        assert err.value.error_code == "MISSING_API_KEY"

    def test_presenter_speech_replaces_the_tts_track(self, tmp_path, monkeypatch, speech_clip):
        from src.agents.tts_agent import TTSOutput

        orch = self._orch(tmp_path, monkeypatch)
        orch._run_output_dir = str(tmp_path)
        orch.hybrid_provider = GeminiOmniPresenter(client=FakeOmni(speech_clip), use_default_transcriber=False)
        sb = _storyboard()
        bridge = sb.scenes[0]
        sb.scenes = [bridge]
        tts_file = tmp_path / "tts.mp3"
        tts_file.write_bytes(b"x")
        tts = [TTSOutput(audio_file_path=str(tts_file), duration_seconds=3.0, scene_id=bridge.scene_id)]
        out = orch._run_presenter_stage(sb, VideoRendererOutput("", [], 0.0), tts)
        assert [r.scene_id for r in out.scene_results] == [bridge.scene_id, "credits"]
        assert tts[0].audio_file_path.endswith("scene_001_presenter.m4a")
        clips = len(split_for_clips(speakable(bridge.narration)))  # 2 s per fake clip
        assert tts[0].duration_seconds == pytest.approx(2.0 * clips, abs=0.3)
        assert Path(tts[0].audio_file_path).is_file()

    def test_split_scenes_must_fit_one_clip(self):
        sb = _storyboard()
        long_intro = sb.scenes[3]
        long_intro.narration = "A singular value tells how much one direction is stretched. " + WORDS + WORDS
        plan = FrameDirectorAgent(use_default_llm=False, max_presenter_seconds=999,
                                  max_explainer_share=1.0, max_split_seconds=14.0).plan(sb)
        decision = next(d for d in plan.decisions if d.scene_id == long_intro.scene_id)
        assert decision.frame == "manim" and "clip" in decision.reason


# ── render-time layout audit ────────────────────────────────────────────

def _mob(cls_name, left, right, bottom, top, text):
    cls = type(cls_name, (), {
        "get_left": lambda self: (left, 0), "get_right": lambda self: (right, 0),
        "get_bottom": lambda self: (0, bottom), "get_top": lambda self: (0, top),
        "get_family": lambda self: [self], "get_fill_opacity": lambda self: 1.0,
        "submobjects": [],
    })
    obj = cls()
    obj.original_text = text
    return obj


class TestLayoutAudit:
    def test_overlapping_and_off_frame_text_is_found(self):
        title = _mob("Text", -7.5, 6.0, 3.2, 3.8, "A very long title")
        a = _mob("MathTex", -1.0, 1.0, -0.3, 0.3, "\\sigma_1")
        b = _mob("Text", -0.6, 1.4, -0.2, 0.4, "singular value")
        far = _mob("Text", 3.0, 5.0, -3.0, -2.5, "fine")
        found = check([title, a, b, far], 14.22, 8.0)
        assert [(o["a"], o["b"]) for o in found["overlaps"]] == [("\\sigma_1", "singular value")]
        assert [o["text"] for o in found["off_frame"]] == ["A very long title"]

    def test_touching_or_slightly_grazing_text_is_fine(self):
        a = _mob("Text", 0.0, 2.0, 0.0, 0.5, "a")
        b = _mob("Text", 1.9, 4.0, 0.45, 1.0, "b")  # ~1% of the smaller box
        assert check([a, b], 14.22, 8.0) == {"overlaps": [], "off_frame": []}

    def test_findings_become_codegen_feedback(self, tmp_path):
        path = tmp_path / "layout_audit.json"
        path.write_text(json.dumps({
            "overlaps": [{"a": "x", "b": "y", "ratio": 0.5, "t": 2.0}],
            "off_frame": [{"text": "title", "t": 0.5}],
        }), encoding="utf-8")
        issues = read_audit(path)
        assert issues == [
            'Text "x" overlaps text "y" (50% of the smaller one) at 2.0s.',
            'Text "title" runs off the frame edge at 0.5s.',
        ]
        feedback = audit_feedback(issues)
        assert "fix every one" in feedback[0] and "LayoutTracker.place_label" in feedback[-1]


class TestLayoutRepair:
    def _orch(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path), max_layout_repairs=1))
        orch._run_output_dir = str(tmp_path)
        return orch

    def _video(self, tmp_path, issues):
        (tmp_path / "scene_scene_002.py").write_text("old code", encoding="utf-8")
        result = SceneRenderResult("scene_002", True, "old.mp4", None, 5.0, layout_issues=issues)
        return VideoRendererOutput("", [result], 5.0)

    def test_a_cleaner_regeneration_replaces_the_render(self, tmp_path):
        orch = self._orch(tmp_path)

        def regenerate(scene, storyboard, feedback):
            (tmp_path / "scene_scene_002.py").write_text("new code", encoding="utf-8")
            return SceneRenderResult("scene_002", True, "new.mp4", None, 5.0, layout_issues=[])

        orch._regenerate_scene = regenerate
        out = orch._repair_layout(self._video(tmp_path, ["a", "b"]), _storyboard())
        assert out.scene_results[0].video_path == "new.mp4"
        assert orch._quality_reports["layout_audit"]["repaired"] == {"scene_002": []}

    def test_a_worse_regeneration_is_discarded_and_the_code_restored(self, tmp_path):
        orch = self._orch(tmp_path)

        def regenerate(scene, storyboard, feedback):
            assert any("a" in line for line in feedback)
            (tmp_path / "scene_scene_002.py").write_text("new code", encoding="utf-8")
            return SceneRenderResult("scene_002", True, "new.mp4", None, 5.0, layout_issues=["a", "b", "c"])

        orch._regenerate_scene = regenerate
        out = orch._repair_layout(self._video(tmp_path, ["a"]), _storyboard())
        assert out.scene_results[0].video_path == "old.mp4"
        assert (tmp_path / "scene_scene_002.py").read_text(encoding="utf-8") == "old code"
        assert "scene_002" in orch._quality_reports["layout_audit"]["kept"]

    def test_the_renderer_switch_is_wired(self, tmp_path):
        assert Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path))).video_renderer_agent.layout_audit
        off = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path), enable_layout_audit=False))
        assert not off.video_renderer_agent.layout_audit


# ── clip assembly: balanced chunks, no dead air, clip cache ─────────────

def _padded_clip(path: Path, lead: float, speech: float, tail: float) -> Path:
    """Silence, a tone standing in for speech, silence (what Omni returns)."""
    import subprocess
    from tests.test_did_avatar import _ffmpeg

    total = lead + speech + tail
    subprocess.run([
        _ffmpeg(), "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={total}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={speech}",
        "-filter_complex", f"[1:a]adelay={int(lead * 1000)}:all=1,apad=whole_dur={total}[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-t", str(total), str(path),
    ], check=True, capture_output=True)
    return path


class TestClipAssembly:
    def test_chunks_are_balanced_not_greedy(self):
        text = ("So now we can see the pattern. Once you know how each singular direction contributes, "
                "the next question is what happens when you keep only the first few of them, and why "
                "that gives the best rank k version of the matrix, which is what we want to understand next.")
        chunks = split_for_clips(text)
        sizes = [len(c.split()) for c in chunks]
        assert len(chunks) == 2 and min(sizes) >= 0.5 * max(sizes)
        assert " ".join(chunks) == text

    def test_short_text_is_one_chunk(self):
        assert split_for_clips(LINE) == [LINE]

    @needs_ffmpeg
    def test_clip_is_trimmed_to_its_speech(self, tmp_path):
        from src.agents.gemini_presenter import trim_to_speech
        from src.utils.media_probe import probe

        clip = _padded_clip(tmp_path / "padded.mp4", lead=1.2, speech=2.0, tail=3.0)
        trim_to_speech(clip)
        assert probe(str(clip)).duration == pytest.approx(2.0 + 0.15 + 0.3, abs=0.3)
        assert probe(str(clip)).audio is not None

    @needs_ffmpeg
    def test_silent_clip_is_left_alone(self, tmp_path):
        from src.agents.gemini_presenter import trim_to_speech
        from src.utils.media_probe import probe

        clip = _padded_clip(tmp_path / "quiet.mp4", lead=0.0, speech=0.2, tail=3.0)
        before = clip.read_bytes()
        trim_to_speech(clip)
        assert clip.read_bytes() == before and probe(str(clip)).duration == pytest.approx(3.2, abs=0.2)

    @needs_ffmpeg
    def test_cached_lines_are_not_generated_twice(self, tmp_path, speech_clip):
        fake = FakeOmni(speech_clip)
        cache = tmp_path / "cache"
        GeminiOmniPresenter(client=fake, use_default_transcriber=False, cache_dir=cache).speak(
            LINE, str(tmp_path / "a.mp4"))
        again = GeminiOmniPresenter(client=fake, use_default_transcriber=False, cache_dir=cache)
        path, seconds = again.speak(LINE, str(tmp_path / "b.mp4"))
        assert len(fake.calls) == 1 and Path(path).is_file() and seconds > 0
        assert again.clips[0]["cached"] is True


class TestSplitCompositesSurviveRepairs:
    def test_retime_and_regenerate_never_touch_a_presenter_composite(self, tmp_path):
        from src.agents.delivery_gate import CheckResult, DeliveryGate, SceneMedia

        gate = DeliveryGate.__new__(DeliveryGate)
        gate.renderer = gate.composer = object()  # would be used if the guard failed
        gate.regenerate = lambda *a: (_ for _ in ()).throw(AssertionError("regenerated a composite"))
        gate.regenerations, gate.max_regenerations = 0, 3
        split = SceneMedia(scene_id="scene_003", index=2, video_path="split.mp4", kind="presenter")
        (tmp_path / "scene_scene_003.py").write_text("self.wait(1)\n", encoding="utf-8")
        av = CheckResult(check="av_duration", passed=False, measured={"gap_s": 3.0}, repair="retime")
        assert gate._retime(split, [av], str(tmp_path), None, None) is False
        assert gate._regenerate(split, [av], str(tmp_path), None, None) is False
        assert (tmp_path / "scene_scene_003.py").read_text(encoding="utf-8") == "self.wait(1)\n"


class TestPresenterCacheLocation:
    def test_run_and_revisions_share_one_cache(self, tmp_path):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = SimpleNamespace(base_output_dir=str(tmp_path))
        orch._run_output_dir = str(tmp_path / "doc" / "run1")
        first = orch._presenter_cache_dir()
        orch._run_output_dir = str(tmp_path / "doc" / "run1" / "rev_002")
        assert orch._presenter_cache_dir() == first == tmp_path / "doc" / "run1" / ".presenter_cache" / "gemini-omni"


@needs_ffmpeg
class TestStalledRequests:
    def test_a_failed_retake_keeps_the_first_take(self, tmp_path, speech_clip):
        fake = FakeOmni(speech_clip)
        calls = {"n": 0}

        def create(model, input, response_format):
            calls["n"] += 1
            if calls["n"] == 2:
                raise TimeoutError("read timed out")
            return fake.create(model, input, response_format)

        client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        heard = iter(["completely different words here"])
        presenter = GeminiOmniPresenter(client=client, transcriber=lambda path: next(heard))
        path, seconds = presenter.speak(LINE, str(tmp_path / "keep.mp4"))
        assert calls["n"] == 2 and Path(path).is_file() and seconds > 0

    def test_two_failed_requests_raise_instead_of_hanging(self, tmp_path):
        def create(model, input, response_format):
            raise TimeoutError("read timed out")

        presenter = GeminiOmniPresenter(
            client=SimpleNamespace(interactions=SimpleNamespace(create=create)), use_default_transcriber=False,
        )
        with pytest.raises(RuntimeError):
            presenter.speak(LINE, str(tmp_path / "none.mp4"))


class TestGateRegenerationKeepsLayout:
    def _orch(self, tmp_path, fresh):
        orch = Orchestrator.__new__(Orchestrator)
        orch._run_output_dir = str(tmp_path)
        orch._quality_reports = {}
        orch._regenerate_scene = lambda scene, storyboard, feedback: (
            (tmp_path / "scene_scene_005.py").write_text("new code", encoding="utf-8"), fresh
        )[1]
        (tmp_path / "scene_scene_005.py").write_text("old code", encoding="utf-8")
        return orch

    def test_a_regeneration_with_a_new_overlap_is_rejected(self, tmp_path):
        fresh = SceneRenderResult("scene_005", True, "new.mp4", None, 5.0)
        fresh.layout_issues = ['Text "best k-dimensional subspace" runs off the frame edge at 2.6s.']
        orch = self._orch(tmp_path, fresh)
        current = SceneRenderResult("scene_005", True, "old.mp4", None, 5.0)
        out = orch._gate_regenerate(SimpleNamespace(scene_id="scene_005"), None,
                                    VideoRendererOutput("", [current], 5.0), ["static beat"])
        assert out is None
        assert (tmp_path / "scene_scene_005.py").read_text(encoding="utf-8") == "old code"
        assert "scene_005" in orch._quality_reports["layout_audit"]["gate_rejected"]

    def test_a_clean_regeneration_is_kept(self, tmp_path):
        fresh = SceneRenderResult("scene_005", True, "new.mp4", None, 5.0)
        orch = self._orch(tmp_path, fresh)
        current = SceneRenderResult("scene_005", True, "old.mp4", None, 5.0)
        out = orch._gate_regenerate(SimpleNamespace(scene_id="scene_005"), None,
                                    VideoRendererOutput("", [current], 5.0), ["static beat"])
        assert out is fresh
