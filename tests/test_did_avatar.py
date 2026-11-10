"""did-avatar render backend: D-ID client, agent, pipeline branch and CLI.

Hermetic: every D-ID call goes to ``FakeDID``, an in-memory stand-in for the
REST API, so the default suite never reaches the network. Tests that build
real clips with ffmpeg skip when ffmpeg is missing. The one live test is
marked ``live_api`` and needs ``DID_API_KEY``.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from src.agents.base import AgentExecutionError, BackoffStrategy, RetryPolicy
from src.agents.did_avatar_renderer import (
    AVATAR_DIRNAME,
    BAD_REQUEST,
    BUDGET_EXCEEDED,
    DEFAULT_VOICE_ID,
    DIDAPIError,
    DIDAvatarRendererAgent,
    DIDAvatarRendererInput,
    DIDClient,
    AvatarScene,
    INSUFFICIENT_CREDITS,
    INVALID_IMAGE_URL,
    MISSING_API_KEY,
    NETWORK_ERROR,
    SERVER_ERROR,
    TALK_FAILED,
    TIMEOUT,
    TRANSIENT_ERRORS,
    UNAUTHORIZED,
    redact,
    resolve_did_api_key,
    validate_avatar_image_url,
)
from src.agents.audio_video_composer import (
    AudioVideoComposerAgent,
    AudioVideoComposerInput,
    AudioVideoComposerOutput,
    SceneMediaPair,
)
from src.orchestrator.pipeline import Orchestrator, PipelineAbortError, PipelineConfig
from src.schemas.storyboard import PDFMetadataStoryboard, StoryboardJSON
from tests.test_revision import _base_run, _config, _edited, _script

# A D-ID key is "<base64 user>:<secret>"; both halves must never leak.
KEY = "cGFydGhAZXhhbXBsZS5jb20:SuperSecretValue123456"
KEY_PARTS = KEY.split(":")
IMAGE = "https://example.com/avatars/professor.jpg"
SIGNED = "https://d-id-talks.s3.amazonaws.com/clips/"
SIGNATURE = "Signature=abcSIG123"

FAST = RetryPolicy(
    max_attempts=3,
    backoff_strategy=BackoffStrategy.CONSTANT,
    base_delay_seconds=0.0,
    max_delay_seconds=0.0,
    retryable_errors=list(TRANSIENT_ERRORS),
)


class FakeDID:
    """In-memory D-ID REST API.

    ``behavior`` maps a substring of a scene's text to "error", "rejected" or
    "timeout". ``post_statuses`` is a queue of HTTP statuses returned for
    POST /talks before it succeeds; their error bodies echo the auth header,
    so redaction is exercised on hostile input.
    """

    def __init__(self, clip_bytes=b"fake-mp4", polls_until_done=1, behavior=None):
        self.calls = []
        self.talks = {}
        self.clip_bytes = clip_bytes
        self.polls_until_done = polls_until_done
        self.behavior = behavior or {}
        self.post_statuses = []
        self.network_failures = 0

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body))
        if self.network_failures:
            self.network_failures -= 1
            raise DIDAPIError(NETWORK_ERROR, f"connection reset (auth was {headers.get('Authorization')})")
        if url.endswith("/credits"):
            return 200, json.dumps({"remaining": 12, "total": 12}).encode()
        if method == "POST" and url.endswith("/talks"):
            if self.post_statuses:
                status = self.post_statuses.pop(0)
                echo = {"kind": "Err", "description": f"bad header {headers.get('Authorization')}"}
                return status, json.dumps(echo).encode()
            payload = json.loads(body)
            talk_id = f"tlk_{len(self.talks) + 1}"
            self.talks[talk_id] = {"text": payload["script"]["input"], "polls": 0}
            return 201, json.dumps({"id": talk_id, "status": "created"}).encode()
        if method == "GET" and "/talks/" in url:
            talk_id = url.rsplit("/", 1)[1]
            talk = self.talks[talk_id]
            talk["polls"] += 1
            mode = next((m for key, m in self.behavior.items() if key in talk["text"]), None)
            if mode in ("error", "rejected"):
                return 200, json.dumps({
                    "id": talk_id, "status": mode,
                    "error": {"kind": "FaceError", "description": "no face detected"},
                }).encode()
            if mode == "timeout" or talk["polls"] < self.polls_until_done:
                return 200, json.dumps({"id": talk_id, "status": "started"}).encode()
            return 200, json.dumps({
                "id": talk_id, "status": "done", "duration": 1.0,
                "result_url": f"{SIGNED}{talk_id}.mp4?AWSAccessKeyId=AKIA&{SIGNATURE}",
            }).encode()
        if method == "GET" and url.startswith(SIGNED):
            return 200, self.clip_bytes
        return 404, b"{}"


def _agent(fake, **kwargs):
    return DIDAvatarRendererAgent(
        api_key=kwargs.pop("api_key", KEY), transport=fake, retry_policy=FAST,
        sleep=kwargs.pop("sleep", lambda s: None), **kwargs,
    )


def _input(tmp_path, *scenes, **kwargs):
    return DIDAvatarRendererInput(
        scenes=[AvatarScene(sid, text) for sid, text in scenes],
        output_dir=str(tmp_path),
        image_url=kwargs.pop("image_url", IMAGE),
        poll_interval_seconds=kwargs.pop("poll_interval_seconds", 0.0),
        max_poll_attempts=kwargs.pop("max_poll_attempts", 5),
        **kwargs,
    )


def _no_network(*args, **kwargs):
    raise AssertionError("D-ID must not be called")


def _assert_no_secret(text: str):
    for secret in (KEY, *KEY_PARTS, SIGNATURE):
        assert secret not in text


# ── inputs: key and portrait URL ────────────────────────────────────────

class TestInputs:
    def test_key_comes_from_documented_name_then_alias(self):
        assert resolve_did_api_key({"DID_API_KEY": "a"}) == "a"
        assert resolve_did_api_key({"D_ID_API_KEY": "b"}) == "b"
        assert resolve_did_api_key({"DID_API_KEY": "a", "D_ID_API_KEY": "b"}) == "a"
        assert resolve_did_api_key({"DID_API_KEY": "  "}) is None
        assert resolve_did_api_key({}) is None

    @pytest.mark.parametrize("url", [
        None, "", "http://example.com/a.jpg", "ftp://example.com/a.jpg", "https:///a.jpg",
        "https://user:pw@example.com/a.jpg", "https://localhost/a.jpg", "professor.jpg",
    ])
    def test_unusable_image_urls_are_rejected(self, url):
        with pytest.raises(ValueError):
            validate_avatar_image_url(url)

    def test_public_https_url_is_accepted(self):
        assert validate_avatar_image_url(f" {IMAGE} ") == IMAGE

    def test_agent_rejects_invalid_url_before_calling_d_id(self, tmp_path):
        agent = _agent(_no_network)
        with pytest.raises(AgentExecutionError) as err:
            agent.execute(_input(tmp_path, ("s1", "Hello"), image_url="http://example.com/a.jpg"))
        assert err.value.error_code == INVALID_IMAGE_URL

    def test_missing_key_in_strict_mode_raises(self, tmp_path):
        agent = DIDAvatarRendererAgent(api_key="", strict=True, transport=_no_network)
        with pytest.raises(AgentExecutionError) as err:
            agent.execute(_input(tmp_path, ("s1", "Hello")))
        assert err.value.error_code == MISSING_API_KEY

    def test_missing_key_without_strict_fails_scenes_without_calls(self, tmp_path):
        output = DIDAvatarRendererAgent(api_key="", transport=_no_network).execute(
            _input(tmp_path, ("s1", "Hello"), ("s2", "World")),
        )
        assert [r.error_code for r in output.scene_results] == [MISSING_API_KEY, MISSING_API_KEY]
        assert output.rendered == 0

    def test_strict_pipeline_aborts_before_any_paid_stage(self, tmp_path, monkeypatch):
        # Strict mode also needs the LLM keys to build the other agents;
        # placeholders suffice since nothing is called.
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.setenv(name, "placeholder")
        orch = Orchestrator(config=PipelineConfig(
            base_output_dir=str(tmp_path), render_backend="did-avatar", avatar_image_url=IMAGE, strict=True,
        ))
        orch._run_ingestion = lambda *a, **k: pytest.fail("ingestion ran before the key check")
        with pytest.raises(PipelineAbortError) as err:
            orch.run_pipeline("notes.pdf", event_source="local")
        assert err.value.error_code == MISSING_API_KEY

    def test_pipeline_rejects_invalid_url_up_front(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(
            base_output_dir=str(tmp_path), render_backend="did-avatar", avatar_image_url="http://x.com/a.jpg",
        ))
        orch._run_ingestion = lambda *a, **k: pytest.fail("ingestion ran before the URL check")
        with pytest.raises(PipelineAbortError) as err:
            orch.run_pipeline("notes.pdf", event_source="local")
        assert err.value.error_code == INVALID_IMAGE_URL

    def test_unknown_backend_is_rejected(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path), render_backend="sora"))
        with pytest.raises(PipelineAbortError):
            orch.run_pipeline("notes.pdf", event_source="local")


# ── create -> poll -> download ──────────────────────────────────────────

class TestRender:
    def test_create_poll_download_sequence(self, tmp_path):
        fake = FakeDID(polls_until_done=3)
        sleeps = []
        output = _agent(fake, sleep=sleeps.append).execute(
            _input(tmp_path, ("scene_001", "Hello, class."), poll_interval_seconds=2.0),
        )
        result = output.scene_results[0]
        assert result.success and result.talk_id == "tlk_1" and result.polls == 3
        clip = tmp_path / AVATAR_DIRNAME / "scene_001.mp4"
        assert Path(result.video_path) == clip and clip.read_bytes() == b"fake-mp4"
        assert sleeps == [2.0, 2.0]

        sequence = [(m, u.split("?")[0].replace(SIGNED, "<clip>/")) for m, u, _, _ in fake.calls]
        assert sequence == [
            ("GET", "https://api.d-id.com/credits"),
            ("POST", "https://api.d-id.com/talks"),
            ("GET", "https://api.d-id.com/talks/tlk_1"),
            ("GET", "https://api.d-id.com/talks/tlk_1"),
            ("GET", "https://api.d-id.com/talks/tlk_1"),
            ("GET", "<clip>/tlk_1.mp4"),
            ("GET", "https://api.d-id.com/credits"),
        ]
        body = json.loads(fake.posts()[0][3])
        assert body["source_url"] == IMAGE
        assert body["script"] == {
            "type": "text", "input": "Hello, class.",
            "provider": {"type": "microsoft", "voice_id": DEFAULT_VOICE_ID},
        }
        api_headers = [h for m, u, h, _ in fake.calls if u.startswith("https://api.d-id.com")]
        assert all(h["Authorization"] == f"Basic {KEY}" for h in api_headers)
        download_headers = [h for m, u, h, _ in fake.calls if u.startswith(SIGNED)]
        assert all("Authorization" not in h for h in download_headers)

        entry = output.report["scenes"][0]
        assert entry["status"] == "done" and entry["video_path"] == "avatar/scene_001.mp4"
        assert output.report["credits_before"] == 12 and output.report["endpoint"].startswith("POST ")

    def test_rate_limits_and_server_errors_are_retried(self, tmp_path):
        fake = FakeDID()
        fake.post_statuses = [429, 503]
        output = _agent(fake).execute(_input(tmp_path, ("s1", "Hello")))
        assert output.scene_results[0].success
        assert len(fake.posts()) == 3

    def test_network_errors_are_retried(self, tmp_path):
        fake = FakeDID()
        fake.network_failures = 2  # the credit check, then the first POST
        output = _agent(fake).execute(_input(tmp_path, ("s1", "Hello")))
        assert output.scene_results[0].success

    def test_persistent_server_error_fails_only_that_scene(self, tmp_path):
        fake = FakeDID()
        fake.post_statuses = [500, 500, 500]  # all three attempts for scene one
        output = _agent(fake).execute(_input(tmp_path, ("s1", "One"), ("s2", "Two")))
        assert [r.status for r in output.scene_results] == ["failed", "done"]
        assert output.scene_results[0].error_code == SERVER_ERROR

    def test_client_error_is_not_retried(self, tmp_path):
        fake = FakeDID()
        fake.post_statuses = [400]
        output = _agent(fake).execute(_input(tmp_path, ("s1", "One"), ("s2", "Two")))
        assert output.scene_results[0].error_code == BAD_REQUEST
        assert output.scene_results[1].success and len(fake.posts()) == 2

    @pytest.mark.parametrize("mode", ["error", "rejected"])
    def test_terminal_talk_failure_is_isolated(self, tmp_path, mode):
        fake = FakeDID(behavior={"bad": mode})
        output = _agent(fake).execute(_input(tmp_path, ("s1", "a bad scene"), ("s2", "a good scene")))
        first, second = output.scene_results
        assert first.status == "failed" and first.error_code == TALK_FAILED
        assert "FaceError" in first.error and first.talk_id == "tlk_1"
        assert second.success

    def test_polling_gives_up_after_max_attempts(self, tmp_path):
        fake = FakeDID(behavior={"slow": "timeout"})
        sleeps = []
        output = _agent(fake, sleep=sleeps.append).execute(
            _input(tmp_path, ("s1", "slow scene"), ("s2", "fast scene"), max_poll_attempts=3, poll_interval_seconds=1.5),
        )
        first, second = output.scene_results
        assert first.error_code == TIMEOUT and first.polls == 3 and "3 polls" in first.error
        assert second.success
        assert sleeps[:2] == [1.5, 1.5]

    @pytest.mark.parametrize("status,code", [(401, UNAUTHORIZED), (402, INSUFFICIENT_CREDITS)])
    def test_account_errors_stop_further_calls(self, tmp_path, status, code):
        fake = FakeDID()
        fake.post_statuses = [status]
        output = _agent(fake).execute(_input(tmp_path, ("s1", "One"), ("s2", "Two"), ("s3", "Three")))
        assert [r.status for r in output.scene_results] == ["failed", "skipped", "skipped"]
        assert {r.error_code for r in output.scene_results} == {code}
        assert len(fake.posts()) == 1

    def test_narration_budget_caps_spend(self, tmp_path):
        fake = FakeDID()
        words = " ".join(["word"] * 10)  # ~4 s each at 2.5 words/s
        output = _agent(fake).execute(_input(
            tmp_path, ("s1", words), ("s2", words), ("s3", words), max_estimated_seconds=9.0,
        ))
        assert [r.status for r in output.scene_results] == ["done", "done", "skipped"]
        assert output.scene_results[2].error_code == BUDGET_EXCEEDED
        assert len(fake.posts()) == 2


# ── secrets ─────────────────────────────────────────────────────────────

class TestSecrets:
    def test_redact_removes_every_form_of_the_key(self):
        import base64
        import urllib.parse

        text = (
            f"key={KEY} b64={base64.b64encode(KEY.encode()).decode()} "
            f"q={urllib.parse.quote(KEY, safe='')} user={KEY_PARTS[0]} secret={KEY_PARTS[1]} "
            "Authorization: Basic abcdefgh12345 Bearer zzzzzzzzzzzz "
            f"url={SIGNED}x.mp4?{SIGNATURE}"
        )
        out = redact(text, [KEY])
        _assert_no_secret(out)
        assert "abcdefgh12345" not in out and "zzzzzzzzzzzz" not in out
        assert f"{SIGNED}x.mp4?[REDACTED]" in out

    def test_no_secret_in_logs_errors_or_report(self, tmp_path, caplog):
        caplog.set_level(logging.DEBUG)
        fake = FakeDID()
        fake.network_failures = 1
        fake.post_statuses = [400]  # error body echoes the auth header
        agent = _agent(fake)
        output = agent.execute(_input(tmp_path, ("s1", "One"), ("s2", "Two")))
        assert output.scene_results[0].error_code == BAD_REQUEST
        _assert_no_secret(caplog.text)
        _assert_no_secret(json.dumps(output.report))
        for result in output.scene_results:
            _assert_no_secret(str(result))
        _assert_no_secret(repr(agent) + repr(agent.client))

    def test_strict_abort_message_carries_no_key(self, tmp_path):
        agent = DIDAvatarRendererAgent(api_key="", strict=True, transport=_no_network)
        with pytest.raises(AgentExecutionError) as err:
            agent.execute(_input(tmp_path, ("s1", "Hello")))
        assert "DID_API_KEY" in str(err.value)

    def test_client_requires_a_key(self):
        with pytest.raises(ValueError):
            DIDClient("")


# ── composer keeps embedded audio ───────────────────────────────────────

def _ffmpeg():
    from src.utils.ffmpeg_finder import get_ffmpeg

    path = get_ffmpeg()
    return path if shutil.which(path) or Path(path).is_file() else None


needs_ffmpeg = pytest.mark.skipif(_ffmpeg() is None, reason="ffmpeg is not installed")


def _clip(path: Path, seconds: float = 1.0, freq: int = 440) -> Path:
    """A tiny talking-head stand-in: a test pattern with a tone."""
    subprocess.run([
        _ffmpeg(), "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return path


@pytest.fixture(scope="module")
def clip_bytes(tmp_path_factory):
    if _ffmpeg() is None:
        pytest.skip("ffmpeg is not installed")
    return _clip(tmp_path_factory.mktemp("clip") / "talk.mp4").read_bytes()


@needs_ffmpeg
class TestComposerEmbeddedAudio:
    def test_clips_keep_their_own_audio(self, tmp_path):
        from src.utils.media_probe import measure_loudness, probe

        clips = [_clip(tmp_path / f"c{i}.mp4", freq=300 + 200 * i) for i in range(2)]
        output = AudioVideoComposerAgent().execute(AudioVideoComposerInput(
            scene_media=[SceneMediaPair(f"s{i}", str(c), embedded_audio=True) for i, c in enumerate(clips)],
            output_dir=str(tmp_path / "out"),
        ))
        assert isinstance(output, AudioVideoComposerOutput) and output.scenes_composed == 2
        info = probe(output.final_video_path)
        assert info.audio is not None and info.duration == pytest.approx(2.0, abs=0.25)
        loudness = measure_loudness(output.final_video_path)
        assert loudness is not None and loudness.integrated > -40  # the tone, not silence


# ── pipeline branch ─────────────────────────────────────────────────────

def _stub_front_half(orch: Orchestrator, script):
    """Real run_pipeline from the storyboard on; ingestion..architect stubbed."""
    ref = SimpleNamespace(filename="notes.pdf")
    orch._run_ingestion = lambda *a, **k: ref
    orch._run_validation = lambda *a, **k: ref
    orch._run_vision_ocr = lambda *a, **k: None
    orch._run_document_segmenter = lambda *a, **k: []

    def fake_architect(ocr_output, structured_logger, source_segments=None):
        agent = orch.script_architect_agent
        planned = script.model_copy(deep=True)
        scenes = agent._chunk_script_into_scenes(planned)
        agent._plan_visual_intents(scenes, planned, _config(), [])
        agent._renumber_scenes(scenes)
        agent.last_video_script = planned
        return StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="notes.pdf", page_count=1),
            configuration=_config(),
            concept_hierarchy=agent._concept_hierarchy_from_ledger(planned.concept_ledger()),
            scenes=scenes,
        )

    orch._run_script_architect = fake_architect


def _forbid_manim_and_tts(orch: Orchestrator):
    def boom(*args, **kwargs):
        raise AssertionError("did-avatar mode must not run codegen, Manim or TTS")

    orch.tts_agent.execute = boom
    orch.tts_agent.process_all_scenes = boom
    orch.scene_controller_agent.execute = boom
    orch.video_renderer_agent.execute = boom


def _avatar_orchestrator(tmp_path, fake, **config) -> Orchestrator:
    orch = Orchestrator(config=PipelineConfig(
        base_output_dir=str(tmp_path),
        run_id="run1",
        enable_final_video_review=False,
        render_backend="did-avatar",
        avatar_image_url=IMAGE,
        did_poll_interval_seconds=0.0,
        **config,
    ))
    orch.did_avatar_agent = _agent(fake)
    _forbid_manim_and_tts(orch)
    return orch


def _avatar_run(tmp_path, fake, script=None, **config):
    orch = _avatar_orchestrator(tmp_path, fake, **config)
    _stub_front_half(orch, script or _script())
    manifest = orch.run_pipeline("notes.pdf", event_source="local")
    return Path(manifest.output_directory), manifest


@needs_ffmpeg
class TestPipelineBackend:
    def test_avatar_run_skips_tts_and_concatenates_clips_with_sound(self, tmp_path, clip_bytes):
        from src.utils.media_probe import probe

        fake = FakeDID(clip_bytes=clip_bytes)
        run_dir, manifest = _avatar_run(tmp_path, fake)
        scene_ids = [r.scene_id for r in manifest.scenes]
        assert len(fake.posts()) == len(scene_ids) >= 2

        assert manifest.final_video_path == "final_with_narration.mp4"
        info = probe(str(run_dir / manifest.final_video_path))
        assert info.audio is not None and info.video is not None
        assert info.duration == pytest.approx(len(scene_ids), abs=0.5)

        assert manifest.run_config.render_backend == "did-avatar"
        assert manifest.run_config.avatar_image_url == IMAGE
        types = [f.type for f in manifest.files]
        assert types.count("avatar_clip") == len(scene_ids)
        assert "scene_video" not in types and "scene_audio" not in types
        assert manifest.delivered_scenes == len(scene_ids) and manifest.delivery_status == "ready"
        assert not list(run_dir.glob("scene_*.py")) and not (run_dir / "audio").exists()

        report = json.loads((run_dir / "did_avatar.json").read_text(encoding="utf-8"))
        assert report["rendered"] == len(scene_ids) and report["final_video"]["has_audio"] is True
        # Each clip speaks exactly its scene's narration.
        spoken = [json.loads(p[3])["script"]["input"] for p in fake.posts()]
        storyboard = json.loads((run_dir / "storyboard.json").read_text(encoding="utf-8"))
        assert spoken == [s["narration"] for s in storyboard["scenes"]]

    def test_failed_scene_is_left_out_but_video_is_built(self, tmp_path, clip_bytes):
        script = _script()
        fake = FakeDID(clip_bytes=clip_bytes, behavior={"Beat 1 ": "rejected"})
        run_dir, manifest = _avatar_run(tmp_path, fake, script=script)
        assert len(manifest.lost_scenes) == 1
        assert manifest.final_video_path and manifest.delivery_status == "needs_review"
        report = json.loads((run_dir / "did_avatar.json").read_text(encoding="utf-8"))
        assert report["failed"] == 1

    def test_no_secret_in_any_run_artifact(self, tmp_path, clip_bytes):
        fake = FakeDID(clip_bytes=clip_bytes)
        run_dir, _ = _avatar_run(tmp_path, fake)
        for path in run_dir.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".log", ".txt", ".md", ".jsonl"}:
                _assert_no_secret(path.read_text(encoding="utf-8", errors="replace"))

    def test_revision_reuses_unchanged_clips(self, tmp_path, clip_bytes):
        script = _script()
        run_dir, manifest = _avatar_run(tmp_path, FakeDID(clip_bytes=clip_bytes), script=script)
        total = len(manifest.scenes)

        def revise(edited):
            fake = FakeDID(clip_bytes=clip_bytes)
            orch = _avatar_orchestrator(tmp_path, fake)

            def no_visual_planning(*args, **kwargs):
                raise AssertionError("avatar revisions must not plan Manim visuals")

            orch.script_architect_agent._plan_visual_intents = no_visual_planning
            # A relative run dir, as `pdf2manim revise output/...` passes it.
            rev = orch.run_revision(os.path.relpath(run_dir), edited)
            return fake, Path(rev.output_directory), rev

        fake, rev_dir, rev = revise(script)
        assert fake.posts() == []
        assert rev.final_video_path and len(rev.scenes) == total

        edited = _edited(script, "beat_003", "A brand new explanation of idea three, said plainly.")
        fake, rev_dir, rev = revise(edited)
        assert len(fake.posts()) == 1
        assert "brand new explanation" in json.loads(fake.posts()[0][3])["script"]["input"]
        revision = json.loads((rev_dir / "revision.json").read_text(encoding="utf-8"))
        assert len(revision["scenes"]["regenerated"]) == 1
        assert len(revision["scenes"]["reused_video"]) == total - 1
        assert (rev_dir / rev.final_video_path).is_file()
        # Clips are listed by their path inside the revision, not a bare name.
        clips = [f.path for f in rev.files if f.type == "avatar_clip"]
        assert len(clips) == total and all(c.startswith("avatar/") for c in clips)
        assert sum(c.startswith("avatar/reused/") for c in clips) == total - 1


class TestNoVisualPlanning:
    """did-avatar renders no scene visuals, so none are planned or graded."""

    def test_architect_and_grader_are_told_to_skip_visuals(self, tmp_path):
        from src.agents.storyboard_grader import StoryboardGraderOutput

        seen = {}

        def architect(input_data):
            seen["plan_visuals"] = input_data.plan_visuals
            return SimpleNamespace()

        def grade(input_data):
            seen["ignore_visuals"] = input_data.ignore_visuals
            return StoryboardGraderOutput(input_data.storyboard, SimpleNamespace(
                overall_score=9, passed=True, summary="", improvement_suggestions=[], criteria=[],
            ))

        silent = SimpleNamespace(log_agent_start=lambda *a, **k: None, log_agent_complete=lambda *a, **k: None)
        avatar = _avatar_orchestrator(tmp_path, FakeDID())
        manim = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
        for orch, is_avatar in ((avatar, True), (manim, False)):
            orch.script_architect_agent.execute = architect
            orch.storyboard_grader_agent.execute = grade
            orch._summarize_segments_for_review = lambda segments: ""
            orch._run_script_architect(None, silent, source_segments=[object()])
            assert seen == {"plan_visuals": not is_avatar, "ignore_visuals": is_avatar}

    def test_placeholder_plans_need_no_llm(self):
        from src.agents.script_architect import ScriptArchitectAgent

        agent = ScriptArchitectAgent()
        agent._generate_visual_intent_for_chunk = lambda **k: pytest.fail("LLM visual planning ran")
        scenes = agent._chunk_script_into_scenes(_script())
        for scene in scenes:
            scene.visual_intent = None
        agent._placeholder_visual_intents(scenes, keep_ids=frozenset({scenes[0].scene_id}))
        assert scenes[0].visual_intent is None
        assert all(s.visual_intent is not None for s in scenes[1:])

    def test_grader_drops_visual_richness_and_visual_critique(self):
        from src.agents.storyboard_grader import (
            GradingCriterion, GradingReport, StoryboardGraderAgent,
        )

        report = GradingReport(
            overall_score=0, passed=False, summary="",
            criteria=[
                GradingCriterion(name="pedagogical_flow", weight=0.30, score=8, feedback=""),
                GradingCriterion(name="visual_richness", weight=0.25, score=0, feedback=""),
                GradingCriterion(name="narration_quality", weight=0.25, score=8, feedback=""),
                GradingCriterion(name="completeness", weight=0.20, score=8, feedback=""),
            ],
            improvement_suggestions=["Add more animations to scene 2", "Define U before using it"],
        )
        out = StoryboardGraderAgent()._without_visuals(report)
        assert out.overall_score == 8.0
        assert [c.name for c in out.criteria] == ["pedagogical_flow", "narration_quality", "completeness"]
        assert out.improvement_suggestions == ["Define U before using it"]


# ── manim stays the default and untouched ───────────────────────────────

class TestManimDefault:
    def test_default_backend_is_manim(self):
        config = PipelineConfig()
        assert config.render_backend == "manim" and config.avatar_image_url is None
        assert Orchestrator(config=config).did_avatar_agent is None

    def test_manim_run_never_touches_d_id(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("the manim backend must not use D-ID")

        monkeypatch.setattr(Orchestrator, "_run_did_avatar_backend", boom)
        monkeypatch.setattr(DIDAvatarRendererAgent, "__init__", boom)
        run_dir, manifest = _base_run(tmp_path, _script())
        assert manifest.run_config.render_backend == "manim"
        assert manifest.run_config.avatar_image_url is None
        types = {f.type for f in manifest.files}
        assert "scene_video" in types and "avatar_clip" not in types
        assert not (run_dir / "did_avatar.json").exists() and not (run_dir / AVATAR_DIRNAME).exists()


# ── CLI ─────────────────────────────────────────────────────────────────

class TestCli:
    @pytest.fixture
    def pdf(self, tmp_path, monkeypatch):
        from src import cli

        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        path = tmp_path / "notes.pdf"
        path.write_bytes(b"%PDF-1.4")
        return path

    def _capture(self, monkeypatch):
        captured = {}

        def fake_run(self, *args, **kwargs):
            captured["config"] = self.config
            raise PipelineAbortError("Test", "STOP", "captured")

        monkeypatch.setattr(Orchestrator, "run_pipeline", fake_run)
        return captured

    def test_did_avatar_requires_an_image_url(self, pdf):
        from src import cli

        result = CliRunner().invoke(cli.pdf2manim, ["convert", str(pdf), "--render-backend", "did-avatar"])
        assert result.exit_code != 0 and "--avatar-image-url" in result.output

    def test_backend_options_reach_the_pipeline(self, pdf, tmp_path, monkeypatch):
        from src import cli

        captured = self._capture(monkeypatch)
        CliRunner().invoke(cli.pdf2manim, [
            "convert", str(pdf), "-o", str(tmp_path / "out"), "--render-backend", "did-avatar",
            "--avatar-image-url", IMAGE, "--max-avatar-seconds", "90",
        ])
        config = captured["config"]
        assert config.render_backend == "did-avatar" and config.avatar_image_url == IMAGE
        assert config.max_avatar_seconds == 90

    def test_default_cli_run_stays_manim(self, pdf, tmp_path, monkeypatch):
        from src import cli

        captured = self._capture(monkeypatch)
        CliRunner().invoke(cli.pdf2manim, ["convert", str(pdf), "-o", str(tmp_path / "out")])
        assert captured["config"].render_backend == "manim"
        assert captured["config"].avatar_image_url is None


# ── live smoke (opt-in: pytest -m live_api, needs DID_API_KEY) ──────────

@pytest.mark.live_api
def test_live_d_id_renders_one_short_clip(tmp_path):
    """Spends about one D-ID credit. Uses D-ID's own sample presenter."""
    if not resolve_did_api_key():
        pytest.skip("DID_API_KEY not set")
    output = DIDAvatarRendererAgent().execute(DIDAvatarRendererInput(
        scenes=[AvatarScene("live_001", "Hello. This is a short test of the avatar backend.")],
        output_dir=str(tmp_path),
        image_url=os.getenv(
            "DID_TEST_IMAGE_URL", "https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg",
        ),
    ))
    result = output.scene_results[0]
    assert result.success, result.error
    assert Path(result.video_path).stat().st_size > 0
