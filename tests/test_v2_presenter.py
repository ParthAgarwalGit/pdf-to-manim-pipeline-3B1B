"""V2 presenter bridges: placement, approval, consent, providers, timeline.

Hermetic: D-ID goes to an in-memory transport, sleeping is recorded instead of
done, and ffmpeg is patched except in the tests marked ``integration``, which
build real clips and skip when ffmpeg is missing.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from src.pipeline_v2 import presenter
from src.pipeline_v2.models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    NarrationPlan,
    SourceDocument,
    SourceFact,
    TimelineSegment,
)
from src.pipeline_v2.planning import validate_grounding
from src.pipeline_v2.presenter import (
    BRIDGE_EVERY_SECONDS,
    DID_MAX_POLLS,
    DID_POLL_INTERVAL_SECONDS,
    DID_USD_PER_SECOND,
    MAX_RUNTIME_SHARE,
    TYPICAL_BRIDGE_SECONDS,
    BridgeProposal,
    DIDPresenter,
    PresenterBudget,
    PresenterClip,
    PresenterConfigError,
    PresenterRenderError,
    PresenterRequest,
    PresenterSettings,
    TitleCardPresenter,
    approve_bridges,
    approved_bridges,
    bridge_fingerprint,
    bridge_lesson_scene,
    bridge_scene,
    build_presenter,
    credits_card,
    interleave_segments,
    load_bridges,
    merge_proposals,
    propose_bridges,
    render_bridge,
    save_bridges,
    synthetic_credit_lines,
)
from src.schemas.professor_profile import ProfessorProfile
from src.utils import media_probe
from src.utils.ffmpeg_finder import get_ffmpeg

# A D-ID key is "<base64 user>:<secret>"; neither half may ever leak.
KEY = "cGFydGhAZXhhbXBsZS5jb20:SuperSecretValue123456"
IMAGE = "https://example.com/avatars/professor.jpg"
SIGNED = "https://d-id-talks.s3.amazonaws.com/clips/"
SIGNATURE = "Signature=abcSIG123"
TITLES = ["Alpha limits", "Beta slopes", "Gamma areas", "Delta series", "Epsilon proofs", "Zeta bounds"]


def _plan(durations, *, titles=None, lesson_title="Calculus Foundations", words=None):
    """A lesson and narration with one scene per entry of ``durations``.

    A duration of None leaves the audio unmeasured, so the word count decides.
    """
    lesson_scenes, narrated = [], []
    for index, seconds in enumerate(durations):
        scene_id = f"scene_{index + 1:03d}"
        title = (titles or TITLES)[index % len(titles or TITLES)]
        lesson_scenes.append(LessonScene(
            scene_id=scene_id, title=title, description="d",
            learning_goal=f"Understand {title}", source_fact_ids=[f"f{index + 1}"],
        ))
        narrated.append(NarratedScene(
            scene_id=scene_id, narration=" ".join(["word"] * (words or 30)),
            beats=[
                NarrationBeat(beat_id=f"{scene_id}_b1", text="one", source_fact_ids=[f"f{index + 1}"]),
                NarrationBeat(beat_id=f"{scene_id}_b2", text="two", source_fact_ids=["f1"]),
            ],
            target_words_per_minute=150, audio_duration_seconds=seconds,
        ))
    lesson = LessonPlan(
        title=lesson_title, audience="undergraduate",
        objectives=["Explain what a limit is", "Compute a simple derivative"],
        scenes=lesson_scenes,
    )
    return lesson, NarrationPlan(scenes=narrated)


def _source(count):
    return SourceDocument(
        filename="a.pdf", sha256="0" * 64, page_count=1,
        facts=[SourceFact(fact_id=f"f{i + 1}", page_number=1, text="fact") for i in range(count)],
    )


def _bridge(bridge_id="bridge_001", before="scene_001", approved=True, kind="opening", text=None):
    return BridgeProposal(
        bridge_id=bridge_id, kind=kind, before_scene_id=before, approved=approved,
        text=text or "Hello and welcome to this short lesson about limits.", estimated_seconds=4.0,
    )


def _request(bridge_id="bridge_001", seconds=4.0, audio=None):
    return PresenterRequest(
        bridge_id=bridge_id, text="Hello and welcome to this short lesson about limits.",
        audio_path=audio, duration_seconds=seconds,
    )


def _profile(tmp_path, **changes) -> Path:
    data = {
        "name": "Prof. Ada Example",
        "consent": {
            "statement": "I consent to an AI avatar of me presenting this lecture.",
            "signed_by": "Ada Example", "signed_at": "2026-10-01",
        },
        "avatar": {"provider": "d-id", "id": "avatar-1", "verified": True},
    }
    data.update(changes)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class FakeDID:
    """In-memory D-ID REST API; error bodies echo the auth header on purpose."""

    def __init__(self, polls_until_done=2, mode=None):
        self.calls = []
        self.polls = 0
        self.polls_until_done = polls_until_done
        self.mode = mode  # None | "error" | "timeout"
        self.post_statuses = []
        self.network_failures = 0

    def count(self, method, fragment):
        return sum(1 for call in self.calls if call[0] == method and fragment in call[1])

    def __call__(self, method, url, headers, body, timeout):
        from src.agents.did_avatar_renderer import NETWORK_ERROR, DIDAPIError

        self.calls.append((method, url, dict(headers), body))
        if self.network_failures:
            self.network_failures -= 1
            raise DIDAPIError(NETWORK_ERROR, f"connection reset (auth was {headers.get('Authorization')})")
        if method == "POST" and url.endswith("/talks"):
            if self.post_statuses:
                echo = {"kind": "Err", "description": f"bad header {headers.get('Authorization')}"}
                return self.post_statuses.pop(0), json.dumps(echo).encode()
            return 201, json.dumps({"id": "tlk_1", "status": "created"}).encode()
        if method == "GET" and "/talks/" in url:
            self.polls += 1
            if self.mode == "error":
                return 200, json.dumps({
                    "id": "tlk_1", "status": "error",
                    "error": {"kind": "FaceError", "description": "no face detected"},
                }).encode()
            if self.mode == "timeout" or self.polls < self.polls_until_done:
                return 200, json.dumps({"id": "tlk_1", "status": "started"}).encode()
            return 200, json.dumps({
                "id": "tlk_1", "status": "done", "duration": 3.2,
                "result_url": f"{SIGNED}tlk_1.mp4?AWSAccessKeyId=AKIA&{SIGNATURE}",
            }).encode()
        if method == "GET" and url.startswith(SIGNED):
            return 200, b"fake-did-mp4"
        return 404, b"{}"


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Replace the ffmpeg step: record each command and write the output file."""
    commands = []

    def run(args, label, timeout=300):
        commands.append((label, list(args)))
        Path(args[-1]).write_bytes(b"encoded-mp4")

    monkeypatch.setattr(presenter, "_ffmpeg", run)
    monkeypatch.setattr(presenter, "_media_seconds", lambda path: None)
    return commands


def _did_presenter(fake, sleeps, **kwargs):
    return DIDPresenter(KEY, IMAGE, transport=fake, sleep=sleeps.append, **kwargs)


def _assert_no_secret(text):
    text = str(text)
    assert KEY not in text
    for part in KEY.split(":"):
        assert part not in text
    assert "Signature" not in text and "AWSAccessKeyId" not in text
    assert not re.search(r"(?i)(basic|bearer) (?!\[REDACTED\])", text)  # no auth header value


# ── placement ───────────────────────────────────────────────────────────

def _position(bridge, lesson):
    ids = [scene.scene_id for scene in lesson.scenes]
    return len(ids) if bridge.before_scene_id is None else ids.index(bridge.before_scene_id)


def test_placement_respects_caps_and_90_second_spacing():
    durations = [60.0] * 6
    lesson, narration = _plan(durations)
    bridges = propose_bridges(lesson, narration)

    assert [b.kind for b in bridges] == ["opening", "transition", "transition", "recap"]
    assert [b.before_scene_id for b in bridges] == ["scene_001", "scene_003", "scene_005", None]
    total = sum(durations)
    assert len(bridges) <= total / BRIDGE_EVERY_SECONDS + 1
    assert len(bridges) <= total * MAX_RUNTIME_SHARE / TYPICAL_BRIDGE_SECONDS
    positions = [_position(b, lesson) for b in bridges]
    gaps = [sum(durations[a:b]) for a, b in zip(positions, positions[1:])]
    assert all(gap >= BRIDGE_EVERY_SECONDS for gap in gaps)


def test_runtime_share_cap_stops_placement_before_the_recap():
    # 170 s allows int(170 * 0.2 / 12) = 2 bridges although three slots are 90 s apart.
    lesson, narration = _plan([90.0, 80.0])
    assert [b.kind for b in propose_bridges(lesson, narration)] == ["opening", "transition"]
    # One long scene: the budget of one is spent on the opening, so no recap.
    lesson, narration = _plan([100.0])
    assert [b.kind for b in propose_bridges(lesson, narration)] == ["opening"]


def test_short_lesson_gets_only_the_opening():
    lesson, narration = _plan([20.0, 30.0, 25.0])
    bridges = propose_bridges(lesson, narration)
    assert [(b.kind, b.before_scene_id) for b in bridges] == [("opening", "scene_001")]


def test_unmeasured_scenes_are_timed_from_their_words():
    # 150 words at the scene's 150 wpm is 60 s; at an override of 100 wpm it is 90 s.
    lesson, narration = _plan([None] * 4, words=150)
    assert [b.before_scene_id for b in propose_bridges(lesson, narration)] == [
        "scene_001", "scene_003", None,
    ]
    slower = propose_bridges(lesson, narration, words_per_minute=100)
    assert [b.before_scene_id for b in slower] == [
        "scene_001", "scene_002", "scene_003", "scene_004", None,
    ]


def test_proposals_are_unapproved_with_unique_ids():
    lesson, narration = _plan([95.0] * 6)
    bridges = propose_bridges(lesson, narration)
    ids = [b.bridge_id for b in bridges]
    assert ids == [f"bridge_{n:03d}" for n in range(1, len(ids) + 1)]
    assert len(set(ids)) == len(ids)
    assert not any(b.approved for b in bridges)
    assert all(b.estimated_seconds > 0 for b in bridges)
    assert propose_bridges(lesson, narration) == bridges  # deterministic


@pytest.mark.parametrize("titles", [
    TITLES,
    ["x"],
    [""],
    [" ".join(["verylongtitleword"] * 40)],
])
def test_templates_stay_between_15_and_45_words(titles):
    lesson, narration = _plan([95.0] * 4, titles=titles, lesson_title=" ".join(["Long"] * 30))
    lesson = lesson.model_copy(update={"objectives": [" ".join(["objective"] * 60), "second"]})
    bridges = propose_bridges(lesson, narration)
    assert {b.kind for b in bridges} == {"opening", "transition", "recap"}
    for bridge in bridges:
        assert 15 <= len(bridge.text.split()) <= 45, bridge.text


def test_templates_name_only_the_neighbouring_scenes():
    lesson, narration = _plan([95.0] * 4)
    opening, second, third, fourth, recap = propose_bridges(lesson, narration)

    assert "Calculus Foundations" in opening.text and TITLES[0] in opening.text
    assert not any(title in opening.text for title in TITLES[1:4])
    assert TITLES[1] in third.text and TITLES[2] in third.text
    assert TITLES[0] not in third.text and TITLES[3] not in third.text
    assert TITLES[0] in second.text and TITLES[2] in fourth.text
    assert "Explain what a limit is" in recap.text and "Compute a simple derivative" in recap.text
    assert not any(title in recap.text for title in TITLES[:4])
    assert all(" I " in f" {b.text} " or "we" in b.text.lower() for b in (opening, third, recap))


# ── bridge scene and grounding ──────────────────────────────────────────

def test_bridge_scene_borrows_its_neighbours_facts_and_stays_grounded():
    lesson, narration = _plan([95.0] * 3)
    bridges = propose_bridges(lesson, narration)
    transition = next(b for b in bridges if b.kind == "transition")
    recap = next(b for b in bridges if b.kind == "recap")

    scene = bridge_scene(transition, lesson, narration)
    assert scene.scene_id == transition.bridge_id and scene.narration == transition.text
    assert len(scene.beats) == 1
    beat = scene.beats[0]
    assert beat.beat_id == f"{transition.bridge_id}_beat_01" and beat.role == "bridge"
    assert beat.pause_after_seconds == 0.6
    assert beat.source_fact_ids == ["f2"]  # first beat of scene_002
    assert scene.target_words_per_minute == 150
    assert scene.audio_path is None
    assert bridge_scene(recap, lesson, narration).beats[0].source_fact_ids == ["f3"]

    grounded_lesson = lesson.model_copy(update={
        "scenes": [*lesson.scenes, *(bridge_lesson_scene(b, lesson) for b in bridges)],
    })
    grounded_narration = NarrationPlan(
        scenes=[*narration.scenes, *(bridge_scene(b, lesson, narration) for b in bridges)],
    )
    assert validate_grounding(grounded_lesson, grounded_narration, _source(3)) == []
    assert all(s.delivery_mode == "presenter" for s in grounded_lesson.scenes[3:])


def test_bridge_scene_rejects_an_unknown_anchor():
    lesson, narration = _plan([95.0] * 2)
    with pytest.raises(ValueError, match="scene_999"):
        bridge_scene(_bridge(before="scene_999"), lesson, narration)


# ── persistence and approval ────────────────────────────────────────────

def test_load_save_approve_round_trip(tmp_path):
    assert load_bridges(tmp_path) == []
    lesson, narration = _plan([95.0] * 3)
    bridges = propose_bridges(lesson, narration)
    path = save_bridges(tmp_path, bridges)
    assert path == tmp_path / "presenter" / "bridges.json"
    assert load_bridges(tmp_path) == bridges
    assert approved_bridges(load_bridges(tmp_path)) == []

    updated = approve_bridges(tmp_path, ["bridge_002"])
    assert [b.approved for b in updated] == [False, True, False, False]
    assert load_bridges(tmp_path) == updated
    assert [b.bridge_id for b in approved_bridges(updated)] == ["bridge_002"]

    everything = approve_bridges(tmp_path)
    assert all(b.approved for b in everything) and load_bridges(tmp_path) == everything


def test_approve_with_the_professors_edited_text(tmp_path):
    lesson, narration = _plan([95.0] * 3)
    save_bridges(tmp_path, propose_bridges(lesson, narration))
    edit = "Bonjour à tous — today we revisit limits, slowly and carefully."
    updated = approve_bridges(tmp_path, ["bridge_001"], texts={"bridge_001": edit})

    assert updated[0].text == edit and updated[0].approved
    assert not updated[1].approved
    assert load_bridges(tmp_path)[0].text == edit  # utf-8 survives the round trip


def test_approve_rejects_unknown_ids_and_stray_edits(tmp_path):
    lesson, narration = _plan([95.0] * 3)
    bridges = propose_bridges(lesson, narration)
    save_bridges(tmp_path, bridges)

    with pytest.raises(ValueError, match="bridge_099"):
        approve_bridges(tmp_path, ["bridge_001", "bridge_099"])
    with pytest.raises(ValueError, match="bridge_077"):
        approve_bridges(tmp_path, texts={"bridge_077": "some text"})
    with pytest.raises(ValueError, match="bridge_002"):
        approve_bridges(tmp_path, ["bridge_001"], texts={"bridge_002": "edited but not approved"})
    with pytest.raises(ValueError):
        approve_bridges(tmp_path, ["bridge_001"], texts={"bridge_001": "   "})
    assert load_bridges(tmp_path) == bridges  # nothing was approved or changed


def test_merge_keeps_approvals_and_edits_and_drops_vanished_slots():
    existing = [
        _bridge("bridge_001", "scene_001", approved=True, text="My own opening words."),
        _bridge("bridge_002", "scene_003", approved=True, kind="transition"),
        _bridge("bridge_003", "scene_009", approved=True, kind="transition"),
    ]
    proposed = [
        _bridge("bridge_001", "scene_001", approved=False, text="template opening"),
        _bridge("bridge_002", "scene_002", approved=False, kind="transition"),
        _bridge("bridge_003", "scene_003", approved=False, kind="transition"),
        _bridge("bridge_004", None, approved=False, kind="recap"),
    ]
    merged = merge_proposals(existing, proposed)

    assert [b.before_scene_id for b in merged] == ["scene_001", "scene_002", "scene_003", None]
    assert merged[0] == existing[0] and merged[2] == existing[1]
    assert [b.approved for b in merged] == [True, False, True, False]
    assert len({b.bridge_id for b in merged}) == 4
    assert merged[2].bridge_id == "bridge_002"  # an approved bridge keeps its id
    assert merge_proposals([], proposed) == proposed
    assert merge_proposals(merged, proposed) == merged  # stable when re-proposed


# ── consent and provider selection ──────────────────────────────────────

def test_default_settings_are_a_title_card_and_never_synthetic():
    settings = PresenterSettings()
    assert settings.provider == "title-card" and settings.max_presenter_seconds == 180.0
    fake = FakeDID()
    provider = build_presenter(settings, environ={}, transport=fake)

    assert isinstance(provider, TitleCardPresenter)
    assert (provider.name, provider.synthetic, provider.speaks) == ("title-card", False, False)
    assert provider.estimated_cost_usd(_request()) == 0.0
    assert fake.calls == []


def test_unknown_provider_or_layout_is_a_config_error():
    with pytest.raises(PresenterConfigError, match="heygen"):
        build_presenter(PresenterSettings(provider="heygen"))
    with pytest.raises(PresenterConfigError, match="pip"):
        build_presenter(PresenterSettings(layout="pip"))


@pytest.mark.parametrize("case, expected", [
    ("no_profile", "profile"),
    ("missing_profile_file", "could not be read"),
    ("invalid_profile_file", "not a valid profile"),
    ("no_consent", "no signed consent"),
    ("other_vendor", "no d-id avatar on record"),
    ("unverified", "identity verification"),
    ("http_url", "https"),
    ("local_url", "local address"),
    ("no_url", "URL"),
    ("no_key", "DID_API_KEY"),
])
def test_d_id_is_refused_before_any_vendor_call(tmp_path, case, expected):
    profile_path = _profile(tmp_path)
    image, environ = IMAGE, {"DID_API_KEY": KEY}
    if case == "no_profile":
        profile_path = None
    elif case == "missing_profile_file":
        profile_path = tmp_path / "absent.json"
    elif case == "invalid_profile_file":
        profile_path.write_text("{not json", encoding="utf-8")
    elif case == "no_consent":
        profile_path = _profile(tmp_path, consent=None)
    elif case == "other_vendor":
        profile_path = _profile(tmp_path, avatar={"provider": "heygen", "id": "a", "verified": True})
    elif case == "unverified":
        profile_path = _profile(tmp_path, avatar={"provider": "d-id", "id": "a", "verified": False})
    elif case == "http_url":
        image = "http://example.com/a.jpg"
    elif case == "local_url":
        image = "https://localhost/a.jpg"
    elif case == "no_url":
        image = None
    elif case == "no_key":
        environ = {}
    fake = FakeDID()
    settings = PresenterSettings(provider="d-id", profile_path=profile_path, avatar_image_url=image)

    with pytest.raises(PresenterConfigError) as error:
        build_presenter(settings, environ=environ, transport=fake)

    assert expected in str(error.value)
    assert fake.calls == []
    _assert_no_secret(error.value)


def test_d_id_checks_run_profile_first_then_consent_then_url_then_key(tmp_path):
    settings = PresenterSettings(provider="d-id", avatar_image_url="http://bad")
    with pytest.raises(PresenterConfigError, match="profile"):
        build_presenter(settings, environ={})
    settings.profile_path = _profile(tmp_path, consent=None)
    with pytest.raises(PresenterConfigError, match="consent"):
        build_presenter(settings, environ={})
    settings.profile_path = _profile(tmp_path)
    with pytest.raises(PresenterConfigError, match="https"):
        build_presenter(settings, environ={})
    settings.avatar_image_url = IMAGE
    with pytest.raises(PresenterConfigError, match="DID_API_KEY"):
        build_presenter(settings, environ={})


def test_d_id_is_built_only_with_consent_url_and_key(tmp_path):
    fake = FakeDID()
    settings = PresenterSettings(
        provider="d-id", profile_path=_profile(tmp_path), avatar_image_url=IMAGE, voice_id="en-GB-RyanNeural",
    )
    provider = build_presenter(settings, environ={"D_ID_API_KEY": KEY}, transport=fake)

    assert isinstance(provider, DIDPresenter)
    assert (provider.name, provider.synthetic, provider.speaks) == ("d-id", True, True)
    assert provider.voice_id == "en-GB-RyanNeural"
    assert fake.calls == []  # building the provider calls nothing
    _assert_no_secret(repr(provider))
    request = _request()
    assert provider.estimated_cost_usd(request) == pytest.approx(
        len(request.text.split()) / 2.5 * DID_USD_PER_SECOND
    )


# ── D-ID provider ───────────────────────────────────────────────────────

def test_d_id_happy_path_creates_polls_downloads_and_brands(tmp_path, fake_ffmpeg):
    fake, sleeps = FakeDID(polls_until_done=2), []
    output = tmp_path / "presenter" / "bridge_001.mp4"
    clip = _did_presenter(fake, sleeps).render(_request(), output)

    assert clip.path == output and output.read_bytes() == b"encoded-mp4"
    assert (clip.provider, clip.synthetic) == ("d-id", True)
    assert clip.duration_seconds == pytest.approx(3.2)  # D-ID's own figure
    assert sleeps == [DID_POLL_INTERVAL_SECONDS]  # one pending poll, never a real sleep
    assert clip.detail["talk_id"] == "tlk_1" and clip.detail["polls"] == 2
    assert clip.detail["estimated_seconds"] == pytest.approx(len(_request().text.split()) / 2.5)
    assert clip.detail["actual_seconds"] == pytest.approx(3.2)
    assert clip.detail["image_url"] == IMAGE
    _assert_no_secret(json.dumps(clip.detail))
    _assert_no_secret(clip)

    post = next(call for call in fake.calls if call[0] == "POST")
    payload = json.loads(post[3])
    assert payload["source_url"] == IMAGE
    assert payload["script"]["input"] == _request().text
    assert payload["script"]["provider"]["voice_id"] == "en-US-JennyNeural"
    download = next(call for call in fake.calls if call[1].startswith(SIGNED))
    assert "Authorization" not in download[2]  # the signed URL gets no key

    (label, args), = fake_ffmpeg
    assert label == "D-ID clip branding"
    graph = args[args.index("-filter_complex") + 1]
    assert "overlay=0:0" in graph and "scale=1280:720" in graph and "fps=30" in graph
    assert "drawtext" not in graph
    assert any(arg.endswith("label.png") for arg in args)
    assert args[args.index("-map", args.index("-map") + 1) + 1] == "0:a:0"  # the clip's own voice
    assert "aac" in args and "48000" in args and "libx264" in args and "yuv420p" in args
    assert not list(output.parent.glob("*.part*"))


def test_d_id_disclosure_label_is_a_transparent_overlay(tmp_path):
    from PIL import Image

    path = presenter._draw_label(tmp_path / "label.png", 1280, 720)
    with Image.open(path) as label:
        assert label.mode == "RGBA" and label.size == (1280, 720)
        alpha = label.getchannel("A")
        assert alpha.getpixel((1279, 719)) == 0  # frame stays visible
        left, top, right, bottom = alpha.getbbox()
        assert left < 40 and top < 40 and right < 640 and bottom < 120  # top-left only
    assert presenter.DISCLOSURE_LABEL == "AI-generated presenter"


def test_d_id_download_is_reused_after_a_local_failure(tmp_path, fake_ffmpeg, monkeypatch):
    fake, sleeps = FakeDID(polls_until_done=1), []
    provider = _did_presenter(fake, sleeps)
    output = tmp_path / "bridge_001.mp4"

    def broken(args, label, timeout=300):
        raise PresenterRenderError("D-ID clip branding failed: disk full")

    monkeypatch.setattr(presenter, "_ffmpeg", broken)
    with pytest.raises(PresenterRenderError, match="disk full"):
        provider.render(_request(), output)
    assert fake.count("POST", "/talks") == 1 and not output.exists()

    monkeypatch.setattr(presenter, "_ffmpeg", lambda args, label, timeout=300: Path(args[-1]).write_bytes(b"ok"))
    clip = provider.render(_request(), output)
    assert fake.count("POST", "/talks") == 1  # the paid talk was not repeated
    assert clip.detail["reused_download"] is True and output.read_bytes() == b"ok"

    # A different text is a different talk.
    other = PresenterRequest("bridge_001", "A completely different bridge text.", None, 3.0)
    provider.render(other, output)
    assert fake.count("POST", "/talks") == 2


def test_d_id_failure_is_redacted_and_falls_back_to_the_title_card(tmp_path, fake_ffmpeg):
    fake, sleeps = FakeDID(), []
    fake.post_statuses = [401]  # the error body echoes the Authorization header
    provider = _did_presenter(fake, sleeps)
    output = tmp_path / "bridge_001.mp4"
    budget = PresenterBudget(180.0)

    with pytest.raises(PresenterRenderError) as error:
        render_bridge(_bridge(), provider, _request(), output, budget=budget)
    assert error.value.code == "DID_UNAUTHORIZED" and "401" in str(error.value)
    _assert_no_secret(error.value)
    assert error.value.__cause__ is None and error.value.__suppress_context__
    assert budget.spent_seconds == 0.0

    fake.post_statuses = [401]
    clip = render_bridge(
        _bridge(), provider, _request(), output, budget=budget, fallback=TitleCardPresenter(),
    )
    assert (clip.provider, clip.synthetic) == ("title-card", False)
    assert "401" in clip.detail["fallback_reason"] and clip.detail["fallback_from"] == "d-id"
    _assert_no_secret(json.dumps(clip.detail))
    assert output.read_bytes() == b"encoded-mp4"
    assert budget.spent_seconds == pytest.approx(4.0)
    assert fake_ffmpeg[-1][0] == "presenter card"
    assert sleeps == []


def test_d_id_rejected_talk_and_network_errors_are_redacted(tmp_path, fake_ffmpeg):
    sleeps = []
    with pytest.raises(PresenterRenderError, match="no face detected") as error:
        _did_presenter(FakeDID(mode="error"), sleeps).render(_request(), tmp_path / "a.mp4")
    assert error.value.code == "DID_TALK_FAILED"

    # A network error while creating is ambiguous, so the paid create is not retried.
    fake = FakeDID()
    fake.network_failures = 1
    with pytest.raises(PresenterRenderError) as error:
        _did_presenter(fake, sleeps).render(_request(), tmp_path / "b.mp4")
    assert error.value.code == "DID_NETWORK_ERROR" and fake.count("POST", "/talks") == 1
    _assert_no_secret(error.value)
    assert sleeps == []


def test_d_id_retries_rate_limits_through_the_injected_sleep(tmp_path, fake_ffmpeg):
    fake, sleeps = FakeDID(polls_until_done=1), []
    fake.post_statuses = [429, 503]
    clip = _did_presenter(fake, sleeps).render(_request(), tmp_path / "a.mp4")

    assert clip.detail["talk_id"] == "tlk_1"
    assert fake.count("POST", "/talks") == 3
    assert sleeps == [2.0, 4.0]


def test_d_id_polling_gives_up_after_the_maximum_without_sleeping(tmp_path, fake_ffmpeg):
    fake, sleeps = FakeDID(mode="timeout"), []
    with pytest.raises(PresenterRenderError, match="not done after 120 polls") as error:
        _did_presenter(fake, sleeps).render(_request(), tmp_path / "a.mp4")

    assert error.value.code == "DID_TIMEOUT"
    assert fake.polls == DID_MAX_POLLS == 120
    assert sleeps == [DID_POLL_INTERVAL_SECONDS] * (DID_MAX_POLLS - 1)
    assert fake_ffmpeg == [] and not (tmp_path / "a.mp4").exists()


# ── budget and render_bridge ────────────────────────────────────────────

class _Recorder:
    name, synthetic, speaks = "recorder", False, False

    def __init__(self, seconds=4.0):
        self.requests = []
        self.seconds = seconds

    def estimated_cost_usd(self, request):
        return 0.0

    def render(self, request, output_path):
        self.requests.append(request)
        return PresenterClip(output_path, self.seconds, False, self.name, {})


def test_render_bridge_refuses_an_unapproved_bridge(tmp_path):
    provider = _Recorder()
    with pytest.raises(ValueError, match="not approved"):
        render_bridge(
            _bridge(approved=False), provider, _request(), tmp_path / "a.mp4", budget=PresenterBudget(180.0),
        )
    assert provider.requests == []


def test_render_bridge_skips_a_bridge_that_would_pass_the_budget(tmp_path):
    provider, budget = _Recorder(seconds=6.0), PresenterBudget(10.0)
    first = render_bridge(_bridge(), provider, _request(seconds=6.0), tmp_path / "a.mp4", budget=budget)
    assert first is not None and budget.spent_seconds == 6.0
    assert budget.would_exceed(4.5) and not budget.would_exceed(4.0)

    skipped = render_bridge(_bridge("bridge_002"), provider, _request("bridge_002", 6.0), tmp_path / "b.mp4", budget=budget)
    assert skipped is None and len(provider.requests) == 1 and budget.spent_seconds == 6.0

    # With no measured length the bridge's own estimate (4 s) is what is checked.
    fits = render_bridge(_bridge("bridge_003"), _Recorder(4.0), _request("bridge_003", 0.0), tmp_path / "c.mp4", budget=budget)
    assert fits is not None and budget.spent_seconds == 10.0


def test_title_card_failure_is_not_retried_on_itself(tmp_path, monkeypatch):
    def broken(args, label, timeout=300):
        raise PresenterRenderError("presenter card failed: no encoder")

    monkeypatch.setattr(presenter, "_ffmpeg", broken)
    card = TitleCardPresenter()
    with pytest.raises(PresenterRenderError, match="no encoder"):
        render_bridge(_bridge(), card, _request(), tmp_path / "a.mp4", budget=PresenterBudget(180.0), fallback=card)
    assert not list(tmp_path.iterdir())


def test_title_card_command_always_carries_an_audio_track(tmp_path, fake_ffmpeg, monkeypatch):
    card = TitleCardPresenter(854, 480, 15)
    silent = card.render(_request(seconds=0.1), tmp_path / "silent.mp4")
    args = fake_ffmpeg[-1][1]
    assert "anullsrc=r=48000:cl=stereo" in args and args[args.index("-t") + 1] == "0.500"
    assert silent.duration_seconds == 0.5 and silent.detail["narrated"] is False

    audio = tmp_path / "narration.mp3"
    audio.write_bytes(b"mp3")
    monkeypatch.setattr(presenter, "_media_seconds", lambda path: 7.25 if path == audio else None)
    narrated = card.render(_request(seconds=3.0, audio=audio), tmp_path / "narrated.mp4")
    args = fake_ffmpeg[-1][1]
    assert str(audio) in args and "anullsrc=r=48000:cl=stereo" not in args
    assert args[args.index("-t") + 1] == "7.250"  # held for the whole narration
    assert narrated.duration_seconds == 7.25 and narrated.synthetic is False
    for needle in ("libx264", "yuv420p", "aac", "48000"):
        assert needle in args

    with pytest.raises(PresenterRenderError, match="missing"):
        card.render(_request(audio=tmp_path / "absent.mp3"), tmp_path / "x.mp4")


def test_credit_lines_and_fingerprint():
    profile = ProfessorProfile(name="Prof. Ada Example")
    assert synthetic_credit_lines(profile, used_avatar=False) == []
    assert synthetic_credit_lines(profile, used_avatar=True) == [
        "Presenter: AI-generated avatar of Prof. Ada Example, used with consent."
    ]
    settings = PresenterSettings(provider="d-id", avatar_image_url=IMAGE)
    fingerprint = bridge_fingerprint(_bridge(), settings)
    assert fingerprint["text"] == _bridge().text and fingerprint["provider"] == "d-id"
    assert fingerprint != bridge_fingerprint(_bridge(text="Another approved wording."), settings)
    assert json.dumps(fingerprint)  # plain data for the task hash


# ── timeline ────────────────────────────────────────────────────────────

def _segments(*durations):
    segments, cursor = [], 0.0
    for index, seconds in enumerate(durations, start=1):
        segments.append(TimelineSegment(
            segment_id=f"segment-scene_{index:03d}", scene_id=f"scene_{index:03d}",
            start_seconds=cursor, duration_seconds=seconds,
            video_path=f"scene_{index:03d}.mp4", audio_path=f"scene_{index:03d}.mp3",
        ))
        cursor += seconds
    return segments


def _clip(name, seconds):
    return PresenterClip(Path(name), seconds, False, "title-card", {})


def test_interleave_orders_bridges_recomputes_starts_and_appends_credits():
    scenes = _segments(10.0, 20.0, 30.0)
    clips = {
        "bridge_003": (_bridge("bridge_003", None, kind="recap"), _clip("b3.mp4", 5.0)),
        "bridge_002": (_bridge("bridge_002", "scene_003", kind="transition"), _clip("b2.mp4", 4.0)),
        "bridge_001": (_bridge("bridge_001", "scene_001"), _clip("b1.mp4", 3.0)),
        "bridge_004": (_bridge("bridge_004", "scene_007", kind="transition"), _clip("b4.mp4", 9.0)),
    }
    timeline = interleave_segments(scenes, clips, credits=(Path("credits.mp4"), 4.0))

    assert [s.scene_id for s in timeline] == [
        "bridge_001", "scene_001", "scene_002", "bridge_002", "scene_003", "bridge_003", "credits",
    ]
    assert [s.mode for s in timeline] == [
        "presenter", "manim", "manim", "presenter", "manim", "presenter", "transition",
    ]
    assert [s.start_seconds for s in timeline] == [0.0, 3.0, 13.0, 33.0, 37.0, 67.0, 72.0]
    assert timeline[0].segment_id == "segment-bridge_001"
    assert timeline[0].video_path == "b1.mp4" and timeline[0].audio_path is None
    assert timeline[-1].video_path == "credits.mp4" and timeline[-1].duration_seconds == 4.0

    # Manim segments are neither removed nor altered beyond their start time.
    kept = [s for s in timeline if s.mode == "manim"]
    assert [s.model_dump(exclude={"start_seconds"}) for s in kept] == [
        s.model_dump(exclude={"start_seconds"}) for s in scenes
    ]
    assert [s.start_seconds for s in scenes] == [0.0, 10.0, 30.0]  # input untouched


def test_interleave_without_bridges_or_scenes():
    scenes = _segments(10.0, 20.0)
    assert interleave_segments(scenes, {}) == scenes
    clips = {"bridge_001": (_bridge("bridge_001", None, kind="recap"), _clip("b1.mp4", 3.0))}
    assert interleave_segments([], clips, credits=(Path("credits.mp4"), 4.0)) == []
    with pytest.raises(ValueError, match="not approved"):
        interleave_segments(scenes, {"bridge_001": (_bridge(approved=False), _clip("b1.mp4", 3.0))})


# ── real ffmpeg ─────────────────────────────────────────────────────────

def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([get_ffmpeg(), "-version"], capture_output=True).returncode == 0
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


def _assert_clip(path, width, height, fps, seconds):
    info = media_probe.probe(str(path))
    assert info is not None and info.video is not None and info.audio is not None
    assert (info.video.width, info.video.height) == (width, height)
    assert info.video.fps == pytest.approx(fps, abs=0.01)
    assert info.video.pix_fmt == "yuv420p" and info.video.codec == "h264"
    assert info.audio.codec == "aac" and info.audio.sample_rate == 48000 and info.audio.channels == 2
    assert info.duration == pytest.approx(seconds, abs=0.2)


@needs_ffmpeg
@pytest.mark.integration
def test_real_title_card_without_narration_has_video_and_silent_audio(tmp_path):
    output = tmp_path / "presenter" / "bridge_001.mp4"
    clip = TitleCardPresenter(854, 480, 15).render(_request(seconds=1.5), output)

    _assert_clip(output, 854, 480, 15, 1.5)
    assert clip.duration_seconds == pytest.approx(1.5, abs=0.2)
    assert (clip.synthetic, clip.provider) == (False, "title-card")
    assert [p.name for p in output.parent.iterdir()] == ["bridge_001.mp4"]


@needs_ffmpeg
@pytest.mark.integration
def test_real_title_card_is_held_for_the_whole_narration(tmp_path):
    audio = tmp_path / "narration.wav"
    subprocess.run(
        [get_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=2.2",
         "-ar", "24000", "-ac", "1", str(audio)],
        check=True, capture_output=True,
    )
    output = tmp_path / "bridge_002.mp4"
    clip = TitleCardPresenter(1280, 720, 30).render(_request(seconds=0.7, audio=audio), output)

    _assert_clip(output, 1280, 720, 30, 2.2)
    assert clip.duration_seconds == pytest.approx(2.2, abs=0.2) and clip.detail["narrated"] is True


@needs_ffmpeg
@pytest.mark.integration
def test_real_credits_card_has_video_and_silent_audio(tmp_path):
    lines = synthetic_credit_lines(ProfessorProfile(name="Prof. Ada Example"), used_avatar=True)
    output = credits_card(lines, tmp_path / "credits.mp4", width=854, height=480, fps=15, seconds=1.0)
    _assert_clip(output, 854, 480, 15, 1.0)


@needs_ffmpeg
@pytest.mark.integration
def test_real_d_id_clip_is_fitted_labelled_and_keeps_its_voice(tmp_path):
    raw = tmp_path / "raw.mp4"  # stands in for a downloaded square D-ID clip
    subprocess.run(
        [get_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=512x512:rate=25:duration=1.2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "44100", "-ac", "1", str(raw)],
        check=True, capture_output=True,
    )
    output = tmp_path / "bridge_001.mp4"
    DIDPresenter(KEY, IMAGE, width=854, height=480, fps=15)._brand(raw, output, tmp_path)
    _assert_clip(output, 854, 480, 15, 1.2)

    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [get_ffmpeg(), "-v", "error", "-y", "-i", str(raw), "-an", "-c:v", "copy", str(silent)],
        check=True, capture_output=True,
    )
    with pytest.raises(PresenterRenderError, match="no audio"):
        DIDPresenter(KEY, IMAGE)._brand(silent, tmp_path / "x.mp4", tmp_path)
