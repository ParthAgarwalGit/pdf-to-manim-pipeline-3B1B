"""Hybrid backend: Manim for visualization, a D-ID presenter for explanation.

Covers the frame director (manim / presenter / split per scene), the D-ID
lip-sync provider (narration audio -> talk), the split layout and transition
fades (real ffmpeg, skipped without it), the comprehension verifier, and the
orchestrator's hybrid wiring. No network: D-ID and vision calls are faked.
"""

import json
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
from src.agents.did_avatar_renderer import DIDClient, DIDLipSyncProvider
from src.agents.frame_director import (
    FrameDirectorAgent,
    SPLIT_CODEGEN_HINT,
    apply_frame_plan,
    visual_need,
)
from src.agents.frame_verifier import FrameVerifierAgent, verdicts_report
from src.agents.presenter_agent import PresenterAgent
from src.agents.video_renderer import SceneRenderResult, VideoRendererOutput
from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig
from src.schemas.storyboard import MathematicalObject, Scene, Transformation, VisualIntent
from src.schemas.video_script import Beat
from tests.test_did_avatar import FAST, IMAGE, KEY, SIGNED, FakeDID, _clip, needs_ffmpeg


# ── fixtures ────────────────────────────────────────────────────────────

def _intent(pictorial=0, text=0, transforms=0):
    objects = [
        MathematicalObject(object_id=f"p{i}", object_type="geometric_shape", content="", primitive="Arrow")
        for i in range(pictorial)
    ] + [
        MathematicalObject(object_id=f"t{i}", object_type="text", content="label", primitive="Text")
        for i in range(text)
    ]
    moves = [
        Transformation(transformation_type="rotate", target_object=f"p{i}", parameters={}, timing=1.0)
        for i in range(transforms)
    ]
    return VisualIntent(mathematical_objects=objects, transformations=moves)


def _scene(n, narration, role="core", intent=None, concepts=("idea",), scene_type="manim", locked=False):
    beat = Beat(
        beat_id=f"beat_{n:03d}", narration=narration, visual_hint="h",
        introduces_concepts=list(concepts), source_segment_id="seg_001", role=role, locked=locked,
    )
    return Scene(
        scene_id=f"scene_{n:03d}", scene_index=n - 1, concept_id=f"concept_{n}",
        scene_title=f"Scene {n}", narration=narration,
        visual_intent=intent or _intent(), duration_estimate=10.0, source_beats=[beat],
        scene_type=scene_type,
    )


WORDS = " ".join(["word"] * 40)


def _storyboard():
    return SimpleNamespace(scenes=[
        _scene(1, "Welcome! Today we will meet the singular value decomposition. " + WORDS,
               role="bridge", scene_type="presenter", locked=True, concepts=()),
        _scene(2, "Watch the matrix rotate the plane, stretch it along the axes, and rotate it again. " + WORDS,
               role="derivation", intent=_intent(pictorial=5, transforms=3)),
        _scene(3, "So, to recap, we now have the whole story in words. " + WORDS,
               role="recap", intent=_intent(text=2), concepts=()),
        _scene(4, "A singular value tells how much one direction is stretched. " + WORDS,
               role="intuition", intent=_intent(pictorial=1, text=1)),
    ])


# ── frame director ──────────────────────────────────────────────────────

class TestFrameDirector:
    def test_visual_need_orders_scenes_by_how_much_they_must_be_seen(self):
        sb = _storyboard()
        needs = [visual_need(s) for s in sb.scenes]
        assert needs[1] > needs[3] > needs[2] > needs[0]

    def test_rules_pick_presenter_manim_and_split(self):
        plan = FrameDirectorAgent(use_default_llm=False, max_presenter_seconds=999,
                                  max_explainer_share=1.0).plan(_storyboard())
        assert [d.frame for d in plan.decisions] == ["presenter", "manim", "presenter", "split"]
        assert plan.decisions[0].reason.startswith("Bridge")
        assert all(d.source == "rules" for d in plan.decisions)

    def test_llm_review_wins_except_for_bridges_and_bad_answers(self):
        def llm(prompt):
            assert "Manim is for VISUALIZATION" in prompt and "EASIEST TO UNDERSTAND" in prompt
            return {"scenes": [
                {"scene_id": "scene_001", "frame": "manim", "reason": "no"},       # bridge: ignored
                {"scene_id": "scene_002", "frame": "split", "reason": "simple picture"},
                {"scene_id": "scene_003", "frame": "hologram", "reason": "?"},     # invalid: ignored
            ]}

        plan = FrameDirectorAgent(llm=llm, max_presenter_seconds=999, max_explainer_share=1.0).plan(_storyboard())
        frames = {d.scene_id: (d.frame, d.source) for d in plan.decisions}
        assert frames["scene_001"] == ("presenter", "rules")
        assert frames["scene_002"] == ("split", "llm")
        assert frames["scene_003"] == ("presenter", "rules")

    def test_budget_demotes_splits_first_and_keeps_bridges(self):
        plan = FrameDirectorAgent(use_default_llm=False, max_presenter_seconds=25.0).plan(_storyboard())
        frames = {d.scene_id: d.frame for d in plan.decisions}
        assert frames["scene_001"] == "presenter"          # bridge stays
        assert frames["scene_004"] == "manim"              # split demoted first
        assert frames["scene_003"] == "manim"
        assert any("presenter budget" in note for note in plan.notes)

    def test_applying_the_plan_sets_scene_types(self):
        sb = _storyboard()
        plan = FrameDirectorAgent(use_default_llm=False, max_presenter_seconds=999,
                                  max_explainer_share=1.0).plan(sb)
        apply_frame_plan(sb, plan)
        recap, split = sb.scenes[2], sb.scenes[3]
        assert recap.scene_type == "presenter" and recap.presenter_approved  # content, not a proposal
        assert split.scene_type == "manim" and split.frame_type == "split" and split.frame_reason

    def test_unapproved_bridge_proposal_stays_a_proposal(self):
        proposal = _scene(9, "Here is a bridge. " + WORDS, role="bridge", scene_type="presenter", locked=False)
        assert not proposal.presenter_approved


# ── D-ID lip-sync provider ──────────────────────────────────────────────

class FakeDIDAudio(FakeDID):
    def __call__(self, method, url, headers, body, timeout):
        if method == "POST" and url.endswith("/audios"):
            self.calls.append((method, url, dict(headers), body))
            return 201, json.dumps({"id": "aud_1", "url": "s3://d-id-audios/aud_1.mp3"}).encode()
        if method == "POST" and url.endswith("/talks") and json.loads(body)["script"]["type"] == "audio":
            self.calls.append((method, url, dict(headers), body))
            talk_id = f"tlk_{len(self.talks) + 1}"
            self.talks[talk_id] = {"text": "audio", "polls": 0}
            return 201, json.dumps({"id": talk_id}).encode()
        return super().__call__(method, url, headers, body, timeout)


class TestLipSyncProvider:
    def test_uploads_narration_and_lip_syncs_a_talk_to_it(self, tmp_path):
        audio = tmp_path / "scene_001.mp3"
        audio.write_bytes(b"ID3-narration-bytes")
        fake = FakeDIDAudio(polls_until_done=2)
        provider = DIDLipSyncProvider(IMAGE, client=DIDClient(KEY, transport=fake, retry_policy=FAST),
                                      poll_interval_seconds=0, sleep=lambda s: None)
        out = provider.render(str(audio), "text", str(tmp_path / "raw.mp4"), 640, 720)

        assert Path(out).read_bytes() == b"fake-mp4"
        upload = next(c for c in fake.calls if c[1].endswith("/audios"))
        assert upload[2]["Content-Type"].startswith("multipart/form-data; boundary=")
        assert b"ID3-narration-bytes" in upload[3] and b'name="audio"' in upload[3]
        talk = json.loads(next(c for c in fake.calls if c[1].endswith("/talks"))[3])
        assert talk["script"] == {"type": "audio", "audio_url": "s3://d-id-audios/aud_1.mp3"}
        assert talk["source_url"] == IMAGE
        api = [h for m, u, h, _ in fake.calls if u.startswith("https://api.d-id.com")]
        assert all(h["Authorization"] == f"Basic {KEY}" for h in api)
        assert all("Authorization" not in h for m, u, h, _ in fake.calls if u.startswith(SIGNED))
        assert provider.talks[0]["talk_id"] == "tlk_1" and provider.talks[0]["polls"] == 2
        assert KEY not in repr(provider)

    def test_rejects_a_bad_portrait_url_and_a_missing_key(self, monkeypatch):
        with pytest.raises(ValueError):
            DIDLipSyncProvider("http://example.com/a.jpg", api_key=KEY)
        with pytest.raises(ValueError):
            DIDLipSyncProvider(IMAGE, api_key="")


# ── split layout and transition fades (real ffmpeg) ─────────────────────

def _mean_rgb(video: str, at: float, tmp: Path):
    from PIL import Image

    from src.utils.ffmpeg_finder import get_ffmpeg

    frame = tmp / f"f_{at}.png"
    subprocess.run([get_ffmpeg(), "-v", "error", "-y", "-ss", str(at), "-i", video, "-frames:v", "1", str(frame)],
                   check=True, capture_output=True)
    with Image.open(frame) as img:
        data = img.convert("RGB").resize((16, 9)).tobytes()
    return tuple(sum(data[i::3]) / (len(data) / 3) for i in range(3))


@needs_ffmpeg
class TestSplitAndFades:
    def test_split_puts_presenter_left_and_animation_right(self, tmp_path):
        from src.utils.media_probe import probe

        manim = _clip(tmp_path / "manim.mp4", seconds=1.0)
        presenter = _clip(tmp_path / "talk.mp4", seconds=2.0, freq=300)

        class Provider:
            name = "d-id"

            def render(self, audio_path, text, out_path, width, height):
                shutil.copy(presenter, out_path)
                return out_path

        agent = PresenterAgent(Provider(), "low", background=Orchestrator.HYBRID_BACKGROUND)
        out = agent.render_split("scene_002", "text", str(tmp_path / "a.mp3"), 1.6, str(tmp_path / "out"),
                                 visual_video=str(manim))
        info = probe(out)
        assert (info.video.width, info.video.height) == (854, 480)
        assert info.duration == pytest.approx(1.6, abs=0.15) and info.audio is None
        # The test pattern is colourful; the background is #1C1C1C.
        left, right = _mean_rgb(out, 1.4, tmp_path), None
        assert max(left) > 40  # presenter half is not empty (and the animation held its last frame)
        del right

    def test_fades_dip_to_the_background_without_changing_length(self, tmp_path):
        from src.utils.media_probe import probe

        clip = _clip(tmp_path / "c.mp4", seconds=2.0)
        out = AudioVideoComposerAgent().execute(AudioVideoComposerInput(
            scene_media=[SceneMediaPair("s1", str(clip), embedded_audio=True, fade_in=0.4, fade_out=0.4)],
            output_dir=str(tmp_path / "o"),
        ))
        final = out.scene_results[0].composed_video_path
        info = probe(final)
        assert info.duration == pytest.approx(2.0, abs=0.15) and info.audio is not None
        first, middle = _mean_rgb(final, 0.0, tmp_path), _mean_rgb(final, 1.0, tmp_path)
        assert all(abs(c - 28) < 12 for c in first)          # starts at #1C1C1C
        assert sum(middle) > sum(first) + 60                 # fully visible mid-scene


# ── comprehension verifier ──────────────────────────────────────────────

@needs_ffmpeg
class TestFrameVerifier:
    def _scenes(self, tmp_path):
        split = _clip(tmp_path / "split.mp4")
        manim = _clip(tmp_path / "manim.mp4", freq=300)
        talk = _clip(tmp_path / "talk.mp4", freq=500)
        return [
            {"scene_id": "s1", "frame": "split", "narration": "n", "video": str(split), "manim_video": str(manim)},
            {"scene_id": "s2", "frame": "presenter", "narration": "n", "video": str(talk)},
        ]

    def test_ab_comparison_switches_a_split_when_full_screen_is_clearer(self, tmp_path):
        seen = []

        def vision(prompt, images):
            seen.append((prompt, len(images)))
            if "option A" in prompt:
                return {"easiest": "B", "legible_in_split": False, "reason": "labels too small at half size"}
            return {"easiest": "manim", "legible": True, "reason": "the narration describes a picture"}

        verdicts = FrameVerifierAgent(vision=vision).review(self._scenes(tmp_path))
        assert [(v.scene_id, v.action) for v in verdicts] == [("s1", "switch_to_manim"), ("s2", "recommend")]
        assert verdicts[0].compared and seen[0][1] == 4 and seen[1][1] == 2
        report = verdicts_report(verdicts)
        assert report["switched"] == ["s1"] and report["recommended"] == {"s2": "manim"}

    def test_split_is_kept_when_it_wins(self, tmp_path):
        verdicts = FrameVerifierAgent(
            vision=lambda p, i: {"easiest": "A", "legible_in_split": True, "reason": "ok"},
        ).review(self._scenes(tmp_path)[:1])
        assert verdicts[0].action == "keep"

    def test_no_vision_model_means_no_review(self):
        assert FrameVerifierAgent(use_default_vision=False).review([{"scene_id": "s"}]) == []


# ── orchestrator wiring ─────────────────────────────────────────────────

def _hybrid(tmp_path, **config):
    return Orchestrator(config=PipelineConfig(
        base_output_dir=str(tmp_path), render_backend="hybrid", avatar_image_url=IMAGE, **config,
    ))


class TestPipelineWiring:
    def test_preflight_forces_narration_first_and_bridges(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DID_API_KEY", KEY)
        orch = _hybrid(tmp_path)
        orch._preflight_render_backend()
        assert orch.config.tts_first and orch.config.avatar_provider == "d-id"
        assert orch.config.presenter_bridges == "propose"

    def test_preflight_rejects_disabled_tts_and_missing_key(self, tmp_path, monkeypatch):
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.setenv(name, "placeholder")  # strict mode builds every agent
        with pytest.raises(PipelineAbortError):
            _hybrid(tmp_path, enable_tts=False)._preflight_render_backend()
        with pytest.raises(PipelineAbortError) as err:
            _hybrid(tmp_path, strict=True)._preflight_render_backend()
        assert err.value.error_code == "MISSING_API_KEY"

    def test_manim_runs_get_no_hybrid_behaviour(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
        sb = _storyboard()
        assert orch._split_codegen_feedback(sb) is None
        assert orch._hybrid_fades(sb, VideoRendererOutput("", [], 0.0)) == {}

    def test_split_scenes_get_a_legibility_hint_for_codegen(self, tmp_path):
        orch = _hybrid(tmp_path)
        sb = _storyboard()
        sb.scenes[3].frame_type = "split"
        assert orch._split_codegen_feedback(sb) == {"scene_004": [SPLIT_CODEGEN_HINT]}

    def test_fades_mark_every_change_of_frame_type(self, tmp_path):
        orch = _hybrid(tmp_path)
        sb = _storyboard()
        for scene, frame in zip(sb.scenes, ["presenter", "manim", "manim", "split"]):
            scene.frame_type = frame
        results = [SceneRenderResult(s.scene_id, True, f"{s.scene_id}.mp4", None, 5.0) for s in sb.scenes]
        results.append(SceneRenderResult("credits", True, "credits.mp4", None, 4.0))
        fades = orch._hybrid_fades(sb, VideoRendererOutput("", results, 0.0))
        t = orch.TRANSITION_SECONDS
        assert fades == {
            "scene_001": (0.0, t), "scene_002": (t, 0.0), "scene_003": (0.0, t),
            "scene_004": (t, t), "credits": (t, 0.0),
        }

    def test_split_without_a_render_falls_back_to_manim(self, tmp_path):
        orch = _hybrid(tmp_path)
        scene = _storyboard().scenes[3]
        scene.frame_type = "split"
        report = {"skipped": [], "split": []}
        spent = orch._render_split_scene(scene, None, {}, {}, 0.0, tmp_path, report)
        assert spent == 0.0 and scene.frame_type == "manim"
        assert report["skipped"][0]["fallback"] == "manim"

    @needs_ffmpeg
    def test_content_presenter_scene_falls_back_to_a_title_card(self, tmp_path):
        from src.agents.tts_agent import TTSOutput

        class Broken:
            name = "d-id"

            def render(self, *args, **kwargs):
                raise RuntimeError("D-ID out of credits")

        orch = _hybrid(tmp_path, render_quality="low")
        orch._run_output_dir = str(tmp_path)
        orch.hybrid_provider = Broken()
        sb = _storyboard()
        recap = sb.scenes[2]
        recap.scene_type, recap.frame_type = "presenter", "presenter"
        sb.scenes = [recap]
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        out = orch._run_presenter_stage(
            sb, VideoRendererOutput("", [], 0.0),
            [TTSOutput(audio_file_path=str(audio), duration_seconds=1.0, scene_id=recap.scene_id)],
        )
        assert [r.scene_id for r in out.scene_results] == [recap.scene_id]
        assert Path(out.scene_results[0].video_path).is_file()
        skipped = orch._quality_reports["presenter"]["skipped"][0]
        assert skipped["fallback"] == "title_card" and "out of credits" in skipped["reason"]

        # An approved bridge in a hybrid video is lesson narration too.
        bridge = _storyboard().scenes[0]
        sb.scenes = [bridge]
        out = orch._run_presenter_stage(
            sb, VideoRendererOutput("", [], 0.0),
            [TTSOutput(audio_file_path=str(audio), duration_seconds=1.0, scene_id=bridge.scene_id)],
        )
        assert [r.scene_id for r in out.scene_results] == [bridge.scene_id]

    def test_verifier_switch_restores_the_plain_render(self, tmp_path, monkeypatch):
        from src.agents import frame_verifier

        orch = _hybrid(tmp_path)
        sb = _storyboard()
        sb.scenes[3].frame_type = "split"
        orch._split_sources = {"scene_004": "plain_manim.mp4"}
        results = [SceneRenderResult("scene_004", True, "split.mp4", None, 5.0)]
        verdict = frame_verifier.FrameVerdict("scene_004", "split", "manim", False, "too small", "switch_to_manim", True)
        monkeypatch.setattr(frame_verifier.FrameVerifierAgent, "review", lambda self, scenes: [verdict])
        orch._verify_frames(sb, VideoRendererOutput("", results, 0.0))
        assert results[0].video_path == "plain_manim.mp4" and sb.scenes[3].frame_type == "manim"
        assert orch._quality_reports["frame_review"]["switched"] == ["scene_004"]

    def test_verifier_switch_gives_the_narrator_back(self, tmp_path, monkeypatch):
        """A speaking presenter's voice must not stay on a scene it no longer appears in."""
        from src.agents import frame_verifier
        from src.agents.tts_agent import TTSOutput

        orch = _hybrid(tmp_path)
        sb = _storyboard()
        sb.scenes[3].frame_type = "split"
        narrator = TTSOutput(audio_file_path="scene_004.mp3", duration_seconds=5.0, scene_id="scene_004")
        presenter = TTSOutput(audio_file_path="scene_004_presenter.m4a", duration_seconds=4.0, scene_id="scene_004")
        tts = [presenter]
        orch._narration_before_presenter = {"scene_004": (tts, narrator)}
        orch._split_sources = {"scene_004": "plain_manim.mp4"}
        results = [SceneRenderResult("scene_004", True, "split.mp4", None, 5.0)]
        verdict = frame_verifier.FrameVerdict("scene_004", "split", "manim", True, "clearer", "switch_to_manim", True)
        monkeypatch.setattr(frame_verifier.FrameVerifierAgent, "review", lambda self, scenes: [verdict])
        orch._verify_frames(sb, VideoRendererOutput("", results, 0.0))
        assert tts == [narrator]

    def test_direct_frames_writes_the_plan(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.agents.script_editor.default_llm", lambda: None)
        orch = _hybrid(tmp_path, max_presenter_seconds=999.0)
        sb = _storyboard()
        orch._direct_frames(sb)
        plan = orch._quality_reports["frame_plan"]
        assert plan["principle"].startswith("manim for visualization")
        assert {d["scene_id"]: d["frame"] for d in plan["decisions"]}["scene_002"] == "manim"


# ── JSON-mode prompts ───────────────────────────────────────────────────

class TestJsonModePrompts:
    """OpenAI JSON mode 400s on a prompt without the word "json"; the live
    SVD dry run lost the frame director's and bridge writer's LLM passes."""

    def _capture(self, monkeypatch):
        import openai

        seen = []

        class Completions:
            def create(self, model, messages, response_format):
                seen.append(messages[0]["content"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

        class Client:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=Completions())

        monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
        monkeypatch.setattr(openai, "OpenAI", Client)
        return seen

    def test_default_llm_always_asks_for_json(self, monkeypatch):
        from src.agents.script_editor import default_llm

        seen = self._capture(monkeypatch)
        default_llm()('Respond as {"scenes": []}')
        default_llm()("Return JSON only.")
        assert "json" in seen[0].lower() and seen[1] == "Return JSON only."

    def test_bridge_and_frame_prompts_reach_the_model_with_json(self, monkeypatch):
        from src.agents.bridge_planner import BridgePlannerAgent

        seen = self._capture(monkeypatch)
        FrameDirectorAgent(max_presenter_seconds=999).plan(_storyboard())
        script = SimpleNamespace(beats=[], analogies=[], title="", pronunciations={})
        beats = [b for s in _storyboard().scenes[1:] for b in s.source_beats]
        BridgePlannerAgent()._write(beats, [(0, "opening")])
        assert len(seen) == 2 and all("json" in p.lower() for p in seen)
        del script

    def test_vision_prompt_reaches_the_model_with_json(self, monkeypatch, tmp_path):
        import openai

        from src.agents.frame_verifier import default_vision

        seen = []

        class Completions:
            def create(self, model, messages, response_format):
                seen.append(messages[0]["content"][0]["text"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

        monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
        monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
        image = tmp_path / "f.png"
        image.write_bytes(b"png")
        default_vision()('Respond as {"easiest": "A"}', [str(image)])
        assert "json" in seen[0].lower()


class TestCreditsOnRevision:
    @needs_ffmpeg
    def test_reused_presenter_clips_still_get_the_credits_card(self, tmp_path):
        from src.agents.tts_agent import TTSOutput

        class Unused:
            name = "d-id"

            def render(self, *args, **kwargs):
                raise AssertionError("a reused clip must not be re-rendered")

        orch = _hybrid(tmp_path, render_quality="low")
        orch._run_output_dir = str(tmp_path)
        orch.hybrid_provider = Unused()
        sb = _storyboard()
        bridge = sb.scenes[0]
        sb.scenes = [bridge]
        reused = _clip(tmp_path / "bridge.mp4")
        out = orch._run_presenter_stage(
            sb, VideoRendererOutput("", [SceneRenderResult(bridge.scene_id, True, str(reused), None, 1.0)], 1.0),
            [TTSOutput(audio_file_path=str(reused), duration_seconds=1.0, scene_id=bridge.scene_id)],
        )
        assert [r.scene_id for r in out.scene_results] == [bridge.scene_id, "credits"]
        assert orch._quality_reports["presenter"]["reused"] == [bridge.scene_id]
