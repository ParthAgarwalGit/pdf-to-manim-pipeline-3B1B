"""Optional presenter bridges: short on-camera moments between Manim scenes.

A bridge is a timeline segment of its own (``mode="presenter"``); it never
replaces a Manim scene. The flow the pipeline wires together is::

    propose_bridges -> bridges.json -> professor approval -> build_presenter
        -> render_bridge (budgeted, title-card fallback) -> interleave_segments

Nothing is narrated or rendered before approval. The default provider is a
plain title card: no likeness, no vendor, no network. The Gemini presenter
is a fictional lecturer described in words who speaks in the narration's
voice. The D-ID avatar is used only after the professor's signed consent and
the vendor's identity verification have been checked, and before any vendor
object exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, Sequence

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from src.prompts.narrative import bridge_prompt, bridge_slot_line, spoken_plain
from src.schemas.professor_profile import ConsentError, ProfessorProfile, require_consent
from src.utils import media_probe
from src.utils.ffmpeg_finder import get_ffmpeg
from src.utils.narration_flow import overlaps

from .media import NARRATION_FILTER
from .model_config import TEXT_MODEL
from .prompt_text import DEFAULT_VOICE
from .providers import clean_json_text
from .state import BudgetExceeded
from .models import (
    LessonPlan,
    LessonScene,
    NarratedScene,
    NarrationBeat,
    NarrationPlan,
    TimelineSegment,
)

BRIDGE_EVERY_SECONDS = 90.0
MAX_RUNTIME_SHARE = 0.20
TYPICAL_BRIDGE_SECONDS = 12.0
BRIDGE_MIN_WORDS = 15
BRIDGE_MAX_WORDS = 45
BRIDGES_FILE = "presenter/bridges.json"

MIN_CLIP_SECONDS = 0.5
BRAND_BACKGROUND = (28, 38, 52)
TEXT_COLOR = (236, 240, 245)
DISCLOSURE_LABEL = "AI-generated presenter"
# Manim's frame per render quality, so presenter clips match the scenes.
QUALITY_FRAMES: dict[str, tuple[int, int, int]] = {
    "low": (854, 480, 15), "medium": (1280, 720, 30), "high": (1920, 1080, 60),
}

# No D-ID price exists in the legacy cost table. This is a conservative
# reservation figure (above D-ID's list price per second of video), not a quote.
DID_USD_PER_SECOND = 0.10
DID_POLL_INTERVAL_SECONDS = 5.0
DID_MAX_POLLS = 120
DID_MAX_ATTEMPTS = 4

# Gemini Omni bills video output tokens: 5792 per second at $12 per million.
OMNI_VIDEO_TOKENS_PER_SECOND = 5792
OMNI_USD_PER_MILLION_TOKENS = 12.00
OMNI_USD_PER_SECOND = OMNI_VIDEO_TOKENS_PER_SECOND * OMNI_USD_PER_MILLION_TOKENS / 1_000_000
OMNI_RETAKE_ALLOWANCE = 1.5  # the engine may regenerate one take

# Model-written bridges (planned lessons) are kept only within these bounds.
BRIDGES_PROMPT_VERSION = "bridges-1"
WRITTEN_BRIDGE_MIN_WORDS = 6
WRITTEN_BRIDGE_MAX_WORDS = 45
WRITTEN_BRIDGE_MAX_OVERLAP = 0.3

_ENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
]
_FONT_NAMES = ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "segoeui.ttf", "Helvetica.ttc")


# ── 1. bridge proposals ─────────────────────────────────────────────────

class BridgeProposal(BaseModel):
    bridge_id: str
    kind: Literal["opening", "transition", "why", "recap"]
    # The Manim scene this bridge precedes; None = after the last scene.
    before_scene_id: Optional[str] = None
    text: str = Field(min_length=1)
    approved: bool = False
    estimated_seconds: float = Field(ge=0.0)


def propose_bridges(
    lesson: LessonPlan,
    narration: NarrationPlan,
    *,
    words_per_minute: Optional[int] = None,
    planned: bool = False,
) -> list[BridgeProposal]:
    """Deterministic bridge proposals: about one per 90 s, at most 20 % of runtime.

    Every proposal is unapproved. The text only names the lesson title, the
    neighbouring scene titles and the lesson objectives; it adds no content.

    ``planned`` (the lesson has a central question): the script already ends
    on its payoff, so there is no recap slot, and the opening says the
    lesson's hook verbatim when it has one.
    """
    narrated = {scene.scene_id: scene for scene in narration.scenes}
    scenes = [
        (scene, narrated[scene.scene_id]) for scene in lesson.scenes
        if scene.scene_id in narrated and scene.delivery_mode not in ("presenter", "transition")
    ]
    durations = [_scene_seconds(timed, words_per_minute) for _, timed in scenes]
    total = sum(durations)
    if not scenes or total <= 0:
        return []
    budget = max(1, int(min(
        total / BRIDGE_EVERY_SECONDS + 1,
        total * MAX_RUNTIME_SHARE / TYPICAL_BRIDGE_SECONDS,
    )))

    slots: list[tuple[str, Optional[int]]] = []
    elapsed = BRIDGE_EVERY_SECONDS  # the opening is always allowed
    for index, seconds in enumerate(durations):
        if len(slots) >= budget:
            break
        if elapsed >= BRIDGE_EVERY_SECONDS:
            slots.append(("opening" if index == 0 else "transition", index))
            elapsed = 0.0
        elapsed += seconds
    else:
        if not planned and len(slots) < budget and elapsed >= BRIDGE_EVERY_SECONDS:
            slots.append(("recap", None))

    hook = lesson.hook if planned and (lesson.hook or "").strip() else ""
    proposals: list[BridgeProposal] = []
    for number, (kind, index) in enumerate(slots, start=1):
        if kind == "opening":
            text = hook or _opening_text(lesson, scenes[0][0])
        elif kind == "transition":
            text = _transition_text(scenes[index - 1][0], scenes[index][0])
        else:
            text = _recap_text(lesson)
        neighbour = scenes[index][1] if index is not None else scenes[-1][1]
        rate = words_per_minute or neighbour.target_words_per_minute
        proposals.append(BridgeProposal(
            bridge_id=f"bridge_{number:03d}",
            kind=kind,
            before_scene_id=scenes[index][0].scene_id if index is not None else None,
            text=text,
            estimated_seconds=round(len(text.split()) * 60.0 / rate, 3),
        ))
    return proposals


def bridge_scene(
    bridge: BridgeProposal,
    lesson: LessonPlan,
    narration: NarrationPlan,
) -> NarratedScene:
    """The bridge as a one-beat scene the V2 TTS service can narrate.

    A bridge states no fact of its own, so the beat borrows the source facts
    of its neighbouring scene's first beat to stay grounded.
    """
    neighbour = _neighbour(bridge, lesson, narration)
    return NarratedScene(
        scene_id=bridge.bridge_id,
        narration=bridge.text,
        beats=[NarrationBeat(
            beat_id=f"{bridge.bridge_id}_beat_01",
            text=bridge.text,
            role="bridge",
            source_fact_ids=list(neighbour.beats[0].source_fact_ids),
            pause_after_seconds=0.6,
        )],
        target_words_per_minute=neighbour.target_words_per_minute,
    )


def bridge_lesson_scene(bridge: BridgeProposal, lesson: LessonPlan) -> LessonScene:
    """A ``delivery_mode="presenter"`` lesson entry for validators that need one.

    The lesson plan is capped at 24 scenes, so the pipeline should keep bridges
    out of ``LessonPlan.scenes`` and use this only for per-bridge validation.
    """
    scenes = {scene.scene_id: scene for scene in lesson.scenes}
    anchor = lesson.scenes[-1] if bridge.before_scene_id is None else scenes.get(bridge.before_scene_id)
    if anchor is None:
        raise ValueError(f"{bridge.bridge_id} precedes unknown scene {bridge.before_scene_id!r}")
    return LessonScene(
        scene_id=bridge.bridge_id,
        title=f"Presenter {bridge.kind}",
        description=bridge.text,
        learning_goal=anchor.learning_goal,
        source_fact_ids=list(anchor.source_fact_ids),
        delivery_mode="presenter",
    )


def _neighbour(bridge: BridgeProposal, lesson: LessonPlan, narration: NarrationPlan) -> NarratedScene:
    narrated = {scene.scene_id: scene for scene in narration.scenes}
    if bridge.before_scene_id is not None:
        scene_id = bridge.before_scene_id
    else:
        scene_id = next(
            (scene.scene_id for scene in reversed(lesson.scenes) if scene.scene_id in narrated), "",
        )
    if scene_id not in narrated:
        raise ValueError(f"{bridge.bridge_id} has no narrated neighbour scene ({scene_id or 'none'})")
    return narrated[scene_id]


def _scene_seconds(scene: NarratedScene, words_per_minute: Optional[int]) -> float:
    if scene.audio_duration_seconds:
        return scene.audio_duration_seconds
    return len(scene.narration.split()) * 60.0 / (words_per_minute or scene.target_words_per_minute)


def _name(text: str, limit: int, default: str) -> str:
    """A title cut to ``limit`` words with no trailing punctuation."""
    words = (text or "").split()[:limit]
    return " ".join(words).rstrip(" .,;:!?") or default


# The fixed words of each template plus the clipped names stay within 15-45 words.
def _opening_text(lesson: LessonPlan, first: LessonScene) -> str:
    return (
        f"Hello, and welcome. Today I want to walk you through {_name(lesson.title, 10, 'this lesson')}. "
        f"We will take it one step at a time, and we begin with {_name(first.title, 10, 'the first idea')}."
    )


def _transition_text(previous: LessonScene, following: LessonScene) -> str:
    return (
        f"We have just looked at {_name(previous.title, 10, 'the last idea')}. "
        "Keep that picture in mind as we move on. "
        f"Next, I want to turn to {_name(following.title, 10, 'the next idea')}."
    )


def _recap_text(lesson: LessonPlan) -> str:
    room = 21
    chosen: list[str] = []
    for objective in lesson.objectives:
        clipped = _name(objective, room, "")
        if not clipped:
            continue
        if chosen and len(clipped.split()) < len(objective.split()):
            break  # only whole objectives after the first
        chosen.append(clipped)
        room -= len(clipped.split())
        if room <= 0:
            break
    summary = "; ".join(chosen) or "the main idea"
    return (
        "That brings us to the end of this lesson. To recap, here is what we set out to do: "
        f"{summary}. Thank you for following along."
    )


async def write_bridges(
    router: Any,
    proposals: Sequence[BridgeProposal],
    lesson: LessonPlan,
    narration: NarrationPlan,
) -> list[BridgeProposal]:
    """Bridge text written from the lesson spine, for a planned lesson.

    One model call covers every slot except an opening that already says the
    hook. A written text replaces its template only when it has 6-45 words
    and does not repeat the beat before or after it; otherwise, and when the
    model cannot be reached, the template stays.
    """
    narrated = {scene.scene_id: scene for scene in narration.scenes}
    planned_scenes = {scene.scene_id: scene for scene in lesson.scenes}
    order = [scene.scene_id for scene in lesson.scenes if scene.scene_id in narrated]
    updated = list(proposals)
    hook = (lesson.hook or "").strip()
    slots: list[tuple[int, str, str]] = []
    lines: list[str] = []
    for slot, bridge in enumerate(updated):
        if bridge.kind == "opening" and hook and bridge.text.strip() == hook:
            continue
        before, after, question = _bridge_context(bridge, order, narrated, planned_scenes)
        lines.append(bridge_slot_line(slot, bridge.kind, before, after, question))
        slots.append((slot, before, after))
    if not lines:
        return updated

    events = router.artifacts
    try:
        response = await router.text(
            stage="bridges", task_id="bridges", provider="google", model=TEXT_MODEL,
            system="You write what a lecturer says on camera between the scenes of an explainer video.",
            prompt=bridge_prompt(lines, lesson), max_output_tokens=1500,
        )
        data = json.loads(clean_json_text(response.text))
        items = data.get("bridges") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ValueError("the response has no bridges list")
    except BudgetExceeded:
        raise
    except Exception as exc:
        events.event(
            "bridges_written_by_template", stage="bridges", task_id="bridges",
            message=f"Bridge text was not written by the model; kept the templates: {exc}",
            reason=type(exc).__name__,
        )
        return updated

    texts: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            texts[int(item.get("slot"))] = " ".join(str(item.get("text") or "").split())
        except (TypeError, ValueError):
            continue
    written: list[str] = []
    kept: list[str] = []
    for slot, before, after in slots:
        bridge = updated[slot]
        text = texts.get(slot, "")
        if (
            WRITTEN_BRIDGE_MIN_WORDS <= len(text.split()) <= WRITTEN_BRIDGE_MAX_WORDS
            and overlaps(text, before) < WRITTEN_BRIDGE_MAX_OVERLAP
            and overlaps(text, after) < WRITTEN_BRIDGE_MAX_OVERLAP
        ):
            text = spoken_plain(text)
            rate = _bridge_neighbour(bridge, order, narrated).target_words_per_minute
            updated[slot] = bridge.model_copy(update={
                "text": text, "estimated_seconds": round(len(text.split()) * 60.0 / rate, 3),
            })
            written.append(bridge.bridge_id)
        else:
            kept.append(bridge.bridge_id)
    if written:
        events.event(
            "bridges_written", stage="bridges", task_id="bridges",
            message=f"Wrote {len(written)} bridge(s) from the lesson spine; kept {len(kept)} template(s).",
            written=written, templates=kept,
        )
    else:
        events.event(
            "bridges_written_by_template", stage="bridges", task_id="bridges",
            message="No written bridge passed the length and repetition checks; kept the templates.",
            reason="rejected", templates=kept,
        )
    return updated


def _bridge_context(
    bridge: BridgeProposal,
    order: Sequence[str],
    narrated: Mapping[str, NarratedScene],
    planned_scenes: Mapping[str, LessonScene],
) -> tuple[str, str, str]:
    """(last beat before the bridge, first beat after it, the next scene's viewer question)."""
    if bridge.before_scene_id is None or bridge.before_scene_id not in order:
        position = len(order)
    else:
        position = order.index(bridge.before_scene_id)
    before = narrated[order[position - 1]].beats[-1].text if position > 0 else ""
    if position >= len(order):
        return before, "", ""
    scene_id = order[position]
    question = planned_scenes[scene_id].viewer_question if scene_id in planned_scenes else ""
    return before, narrated[scene_id].beats[0].text, question


def _bridge_neighbour(
    bridge: BridgeProposal, order: Sequence[str], narrated: Mapping[str, NarratedScene],
) -> NarratedScene:
    scene_id = bridge.before_scene_id if bridge.before_scene_id in narrated else order[-1]
    return narrated[scene_id]


# ── 2. persistence and approval ─────────────────────────────────────────

def load_bridges(run_dir: Path) -> list[BridgeProposal]:
    path = Path(run_dir) / BRIDGES_FILE
    if not path.is_file():
        return []
    return [BridgeProposal.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def save_bridges(run_dir: Path, bridges: Sequence[BridgeProposal]) -> Path:
    path = Path(run_dir) / BRIDGES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [bridge.model_dump(mode="json") for bridge in bridges]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def approve_bridges(
    run_dir: Path,
    bridge_ids: Optional[Sequence[str]] = None,
    *,
    texts: Optional[Mapping[str, str]] = None,
) -> list[BridgeProposal]:
    """Approve every bridge (``None``) or the named ones, and save.

    ``texts`` replaces a bridge's text with the professor's edit before it is
    approved; an edit may only name a bridge that this call approves.
    """
    bridges = load_bridges(run_dir)
    known = {bridge.bridge_id for bridge in bridges}
    selected = known if bridge_ids is None else set(bridge_ids)
    edits = dict(texts or {})
    unknown = sorted((selected | set(edits)) - known)
    if unknown:
        raise ValueError(f"Unknown bridge id(s): {', '.join(unknown)}")
    unapproved_edits = sorted(set(edits) - selected)
    if unapproved_edits:
        raise ValueError(
            f"Edited text was given for bridge(s) not being approved: {', '.join(unapproved_edits)}"
        )
    updated: list[BridgeProposal] = []
    for bridge in bridges:
        values = bridge.model_dump()
        if bridge.bridge_id in edits:
            text = " ".join(edits[bridge.bridge_id].split())
            scale = len(text.split()) / max(1, len(bridge.text.split()))
            values.update(text=text, estimated_seconds=round(bridge.estimated_seconds * scale, 3))
        if bridge.bridge_id in selected:
            values["approved"] = True
        updated.append(BridgeProposal.model_validate(values))
    save_bridges(run_dir, updated)
    return updated


def merge_proposals(
    existing: Sequence[BridgeProposal],
    proposed: Sequence[BridgeProposal],
) -> list[BridgeProposal]:
    """Re-proposal that keeps the professor's work.

    The result has the slots of ``proposed`` in their order. A slot that
    already exists keeps its entry untouched (id, approval, edited text); a
    new slot gets the new proposal under an id no kept entry uses. Existing
    entries whose slot is no longer proposed (their scene is gone) are dropped.
    """
    by_slot = {bridge.before_scene_id: bridge for bridge in existing}
    slots = {bridge.before_scene_id for bridge in proposed}
    used = {bridge.bridge_id for bridge in existing if bridge.before_scene_id in slots}
    merged: list[BridgeProposal] = []
    number = 0
    for bridge in proposed:
        kept = by_slot.get(bridge.before_scene_id)
        if kept is not None:
            merged.append(kept)
            continue
        bridge_id = bridge.bridge_id
        while bridge_id in used:
            number += 1
            bridge_id = f"bridge_{number:03d}"
        used.add(bridge_id)
        merged.append(bridge.model_copy(update={"bridge_id": bridge_id}))
    return merged


def approved_bridges(bridges: Sequence[BridgeProposal]) -> list[BridgeProposal]:
    return [bridge for bridge in bridges if bridge.approved]


# ── 3. consent and provider selection ───────────────────────────────────

class PresenterConfigError(ValueError):
    """Presenter settings cannot be used. Raised before any vendor call."""


class PresenterRenderError(RuntimeError):
    """A presenter clip could not be produced. The message is already redacted."""

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


@dataclass
class PresenterSettings:
    provider: str = "title-card"  # "title-card" | "gemini" | "d-id"
    layout: str = "full"
    profile_path: Optional[Path] = None
    avatar_image_url: Optional[str] = None
    voice_id: str = "en-US-JennyNeural"  # D-ID stock voice
    max_presenter_seconds: float = 180.0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Gemini: the prebuilt voice the narration uses, a worded fictional
    # persona ("" = the default lecturer) and where raw clips are cached.
    voice: str = DEFAULT_VOICE
    persona: str = ""
    cache_dir: Optional[Path] = None


@dataclass
class PresenterRequest:
    bridge_id: str
    text: str
    audio_path: Optional[Path]
    duration_seconds: float


@dataclass
class PresenterClip:
    path: Path
    duration_seconds: float
    synthetic: bool
    provider: str
    detail: dict[str, Any] = field(default_factory=dict)


class PresenterProvider(Protocol):
    name: str
    synthetic: bool
    # True when the provider voices the text itself: the pipeline must not
    # spend TTS on that bridge.
    speaks: bool

    def estimated_cost_usd(self, request: PresenterRequest) -> float: ...

    def render(self, request: PresenterRequest, output_path: Path) -> PresenterClip: ...


def load_profile(profile_path: Optional[Path]) -> ProfessorProfile:
    if profile_path is None:
        raise PresenterConfigError(
            "The d-id presenter needs a professor profile with signed consent (no profile was given)."
        )
    try:
        return ProfessorProfile.load(str(profile_path))
    except OSError as exc:
        raise PresenterConfigError(
            f"Professor profile {profile_path} could not be read ({type(exc).__name__})."
        ) from None
    except ValueError:
        raise PresenterConfigError(
            f"Professor profile {profile_path} is not a valid profile file."
        ) from None


def build_presenter(
    settings: PresenterSettings,
    *,
    environ: Optional[Mapping[str, str]] = None,
    transport: Optional[Callable[..., tuple[int, bytes]]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PresenterProvider:
    """The configured provider, or PresenterConfigError naming what is missing.

    For ``d-id`` the checks run in a fixed order (profile, consent and vendor
    verification, image URL, API key) and all of them pass before the vendor
    client is built, so a refused configuration can never reach the network.
    ``gemini`` needs only ``GEMINI_API_KEY``: its persona is fictional and
    described in words, never a real person's likeness, so no consent profile
    is involved. Nothing here makes a network call.
    """
    if settings.layout != "full":
        raise PresenterConfigError(
            f"Unknown presenter layout {settings.layout!r}: only 'full' is supported."
        )
    if settings.provider == "title-card":
        return TitleCardPresenter(settings.width, settings.height, settings.fps)
    if settings.provider == "gemini":
        env = os.environ if environ is None else environ
        api_key = (env.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise PresenterConfigError(
                "GEMINI_API_KEY is not set; the gemini presenter cannot be used."
            )
        return GeminiPresenter(
            api_key, voice=settings.voice, persona=settings.persona, cache_dir=settings.cache_dir,
            width=settings.width, height=settings.height, fps=settings.fps,
        )
    if settings.provider != "d-id":
        raise PresenterConfigError(
            f"Unknown presenter provider {settings.provider!r}: use 'title-card', 'gemini' or 'd-id'."
        )

    did = _did()
    profile = load_profile(settings.profile_path)
    try:
        require_consent(profile, "avatar", "d-id")
    except ConsentError as exc:
        raise PresenterConfigError(str(exc)) from None
    try:
        image_url = did.validate_avatar_image_url(settings.avatar_image_url)
    except ValueError as exc:
        raise PresenterConfigError(f"Avatar image URL refused: {exc}") from None
    api_key = did.resolve_did_api_key(environ)
    if not api_key:
        raise PresenterConfigError(
            "DID_API_KEY (or D_ID_API_KEY) is not set; the d-id presenter cannot be used."
        )
    return DIDPresenter(
        api_key, image_url, voice_id=settings.voice_id,
        width=settings.width, height=settings.height, fps=settings.fps,
        transport=transport, sleep=sleep,
    )


def _did() -> Any:
    # Imported on demand: this module drags in the legacy agent stack, which
    # the default title-card path must never load.
    from src.agents import did_avatar_renderer

    return did_avatar_renderer


# ── 4. providers ────────────────────────────────────────────────────────

class TitleCardPresenter:
    """The bridge text on a brand card. Nothing synthetic, nothing paid."""

    name = "title-card"
    synthetic = False
    speaks = False

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width, self.height, self.fps = width, height, fps

    def estimated_cost_usd(self, request: PresenterRequest) -> float:
        return 0.0

    def render(self, request: PresenterRequest, output_path: Path) -> PresenterClip:
        seconds = max(request.duration_seconds, MIN_CLIP_SECONDS)
        audio = Path(request.audio_path) if request.audio_path is not None else None
        if audio is not None:
            if not audio.is_file():
                raise PresenterRenderError(f"Narration audio for {request.bridge_id} is missing.")
            # The card is held for the whole narration, whatever was estimated.
            seconds = max(_media_seconds(audio) or seconds, MIN_CLIP_SECONDS)
        with tempfile.TemporaryDirectory() as tmp:
            card = _draw_card([request.text], Path(tmp) / "card.png", self.width, self.height)
            _still_clip(card, output_path, fps=self.fps, seconds=seconds, audio=audio)
        return PresenterClip(
            path=output_path,
            duration_seconds=_media_seconds(output_path) or seconds,
            synthetic=False,
            provider=self.name,
            detail={"bridge_id": request.bridge_id, "narrated": audio is not None},
        )


class DIDPresenter:
    """A D-ID talking-head clip that speaks the bridge with a stock voice.

    Build it through ``build_presenter`` so consent is checked first. The API
    key stays inside this object; every message and detail that leaves it is
    redacted, and the signed download URL is never stored.
    """

    name = "d-id"
    synthetic = True
    speaks = True

    def __init__(
        self,
        api_key: str,
        image_url: str,
        *,
        voice_id: str = "en-US-JennyNeural",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        transport: Optional[Callable[..., tuple[int, bytes]]] = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = DID_POLL_INTERVAL_SECONDS,
        max_polls: int = DID_MAX_POLLS,
    ):
        if not api_key:
            raise PresenterConfigError("A D-ID API key is required.")
        self._api_key = api_key
        self.image_url = image_url
        self.voice_id = voice_id
        self.width, self.height, self.fps = width, height, fps
        self._transport = transport
        self._sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.max_polls = max(1, int(max_polls))

    def __repr__(self) -> str:  # never show the key
        return f"DIDPresenter(voice_id={self.voice_id!r})"

    def estimated_cost_usd(self, request: PresenterRequest) -> float:
        return _did().estimate_seconds(request.text) * DID_USD_PER_SECOND

    def render(self, request: PresenterRequest, output_path: Path) -> PresenterClip:
        did = _did()
        estimated = did.estimate_seconds(request.text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # The paid download is kept beside the output under a name tied to its
        # inputs, so a retry after a local failure never pays for the talk again.
        digest = hashlib.sha256(
            json.dumps([request.text, self.image_url, self.voice_id]).encode("utf-8")
        ).hexdigest()[:12]
        raw = output_path.with_name(f"{output_path.stem}.did-{digest}.mp4")
        talk: dict[str, Any] = {"talk_id": None, "polls": 0, "reported_seconds": None}
        try:
            reused = raw.is_file() and raw.stat().st_size > 0
            if not reused:
                talk = self._talk(request.text, raw)
            actual = _media_seconds(raw) or _positive(talk["reported_seconds"])
            with tempfile.TemporaryDirectory() as tmp:
                self._brand(raw, output_path, Path(tmp))
        except did.DIDAPIError as exc:
            raise PresenterRenderError(self._redact(exc.message), code=exc.error_code) from None
        except PresenterRenderError as exc:
            raise PresenterRenderError(self._redact(exc), code=exc.code) from None
        except Exception as exc:
            raise PresenterRenderError(self._redact(f"{type(exc).__name__}: {exc}")) from None
        detail = {
            "bridge_id": request.bridge_id,
            "talk_id": talk["talk_id"],
            "polls": talk["polls"],
            "estimated_seconds": round(estimated, 3),
            "actual_seconds": round(actual, 3) if actual else None,
            "voice_id": self.voice_id,
            "image_url": did.display_url(self.image_url),
            "reused_download": reused,
            "disclosure": DISCLOSURE_LABEL,
        }
        return PresenterClip(
            path=output_path,
            duration_seconds=_media_seconds(output_path) or actual or estimated or MIN_CLIP_SECONDS,
            synthetic=True,
            provider=self.name,
            # Belt and braces: nothing in the detail may carry the key.
            detail=json.loads(self._redact(json.dumps(detail))),
        )

    def _redact(self, text: Any) -> str:
        return _did().redact(text, [self._api_key])

    def _retry(self, call: Callable[[], Any], retry_on: Sequence[str]) -> Any:
        did = _did()
        for attempt in range(DID_MAX_ATTEMPTS):
            try:
                return call()
            except did.DIDAPIError as exc:
                if exc.error_code not in retry_on or attempt == DID_MAX_ATTEMPTS - 1:
                    raise
                self._sleep(min(30.0, 2.0 * 2 ** attempt))
        raise AssertionError("unreachable")

    def _talk(self, text: str, raw: Path) -> dict[str, Any]:
        """create -> bounded poll -> download. Returns talk id, polls, D-ID's duration."""
        did = _did()
        from src.agents.base import RetryPolicy

        # The client's own retry sleeps for real; retries happen here instead,
        # through the injected sleep.
        client = did.DIDClient(
            self._api_key, transport=self._transport,
            retry_policy=RetryPolicy(max_attempts=1, retryable_errors=[]),
        )
        # A network error on create is ambiguous (the talk may exist and be
        # billed), so only failures the server itself reported are retried.
        talk_id = self._retry(
            lambda: client.create_talk(self.image_url, text, self.voice_id),
            (did.RATE_LIMITED, did.SERVER_ERROR),
        )
        status: dict[str, Any] = {}
        for polls in range(1, self.max_polls + 1):
            status = self._retry(lambda: client.get_talk(talk_id), did.TRANSIENT_ERRORS)
            state = str(status.get("status", "")).lower()
            if state == "done":
                result_url = status.get("result_url")
                if not result_url:
                    raise did.DIDAPIError(
                        did.BAD_RESPONSE, f"D-ID reported talk {talk_id} done without a result URL."
                    )
                self._retry(lambda: client.download(str(result_url), raw), did.TRANSIENT_ERRORS)
                return {"talk_id": talk_id, "polls": polls, "reported_seconds": status.get("duration")}
            if state in did.FAILED_STATUSES:
                reason = status.get("error") or {}
                if isinstance(reason, dict):
                    reason = reason.get("description") or reason.get("kind") or ""
                raise did.DIDAPIError(
                    did.TALK_FAILED, f"D-ID talk {talk_id} {state}: {str(reason)[:200] or 'no reason given'}"
                )
            if polls < self.max_polls:
                self._sleep(self.poll_interval_seconds)
        raise did.DIDAPIError(
            did.TIMEOUT,
            f"D-ID talk {talk_id} not done after {self.max_polls} polls "
            f"({self.poll_interval_seconds:g}s apart); last status: {status.get('status', 'unknown')}.",
        )

    def _brand(self, raw: Path, output_path: Path, workdir: Path) -> None:
        """Fit the clip to the frame with the disclosure label, keeping its voice."""
        _brand_clip(raw, output_path, workdir, self.width, self.height, self.fps, "D-ID clip")


class GeminiPresenter:
    """A fictional lecturer (Gemini Omni) who says the bridge in the narration's voice.

    Build it through ``build_presenter``. The Omni engine is created on the
    first render, so checking the configuration never builds a client. Clips
    are cached by text and voice (``cache_dir``), carry the disclosure label,
    and are re-encoded to the compositor's frame, rate, audio format and
    loudness. The engine's Whisper check is off: it would be an OpenAI call
    outside this run's cost ledger.
    """

    name = "gemini"
    synthetic = True
    speaks = True

    def __init__(
        self,
        api_key: str,
        *,
        voice: str = DEFAULT_VOICE,
        persona: str = "",
        cache_dir: Optional[Path] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        client: Any = None,
    ):
        if not api_key and client is None:
            raise PresenterConfigError("A GEMINI_API_KEY is required for the gemini presenter.")
        self._api_key = api_key
        self._client = client
        self.voice = voice
        self.persona = persona
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.width, self.height, self.fps = width, height, fps
        self._engine: Any = None

    def __repr__(self) -> str:  # never show the key
        return f"GeminiPresenter(voice={self.voice!r})"

    @property
    def engine(self) -> Any:
        if self._engine is None:
            omni = _omni()
            self._engine = omni.GeminiOmniPresenter(
                api_key=self._api_key, client=self._client,
                persona=self.persona or omni.DEFAULT_PERSONA,
                use_default_transcriber=False, cache_dir=self.cache_dir, voice=self.voice,
            )
        return self._engine

    def estimated_cost_usd(self, request: PresenterRequest) -> float:
        return omni_cost_usd(request.text)

    def render(self, request: PresenterRequest, output_path: Path) -> PresenterClip:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw = output_path.with_name(f"{output_path.stem}.omni{output_path.suffix}")
        try:
            engine = self.engine
            # The engine keeps its lecturer's reference frame beside the clips;
            # a resumed run picks it up so the same person appears throughout.
            reference = output_path.parent / "presenter_reference.png"
            if getattr(engine, "reference_image", None) is None and reference.is_file():
                engine.reference_image = reference
            spoken, seconds = engine.speak(request.text, str(raw))
            with tempfile.TemporaryDirectory() as tmp:
                _brand_clip(Path(spoken), output_path, Path(tmp), self.width, self.height,
                            self.fps, "Gemini presenter clip")
        except PresenterRenderError as exc:
            raise PresenterRenderError(self._redact(exc), code=exc.code) from None
        except Exception as exc:
            raise PresenterRenderError(self._redact(f"{type(exc).__name__}: {exc}")) from None
        return PresenterClip(
            path=output_path,
            duration_seconds=_media_seconds(output_path) or _positive(seconds) or MIN_CLIP_SECONDS,
            synthetic=True,
            provider=self.name,
            detail={
                "bridge_id": request.bridge_id,
                "voice": self.voice,
                "clips": len(_omni().split_for_clips(request.text)),
                "raw_seconds": round(float(seconds or 0.0), 3),
                "disclosure": DISCLOSURE_LABEL,
            },
        )

    def _redact(self, text: Any) -> str:
        message = str(text)[:600]
        return message.replace(self._api_key, "[REDACTED]") if self._api_key else message


def omni_clip_seconds(text: str) -> int:
    """Seconds of Omni video ``text`` is requested as: each clip chunk rounded up."""
    omni = _omni()
    return sum(omni.clip_seconds(chunk) for chunk in omni.split_for_clips(text))


def omni_cost_usd(text: str) -> float:
    """Reservation for one Gemini bridge, with an allowance for one retake."""
    seconds = omni_clip_seconds(text)
    return (
        seconds * OMNI_VIDEO_TOKENS_PER_SECOND * OMNI_USD_PER_MILLION_TOKENS / 1_000_000
        * OMNI_RETAKE_ALLOWANCE
    )


def presenter_usd_per_second(provider_name: str) -> float:
    """Settlement rate for measured presenter video, by provider name."""
    return {"gemini": OMNI_USD_PER_SECOND, "d-id": DID_USD_PER_SECOND}.get(provider_name, 0.0)


def _omni() -> Any:
    # Imported on demand, like the D-ID module: the title-card path never loads it.
    from src.agents import gemini_presenter

    return gemini_presenter


def _brand_clip(
    raw: Path, output_path: Path, workdir: Path, width: int, height: int, fps: int, what: str,
) -> None:
    """Fit a talking clip to the frame under the disclosure label, keeping its voice.

    The audio is loudness-normalised exactly like narration (``NARRATION_FILTER``)
    and encoded with the compositor's codecs, so presenter and Manim segments
    join as uniform media.
    """
    info = media_probe.probe(str(raw))
    if info is not None and info.audio is None:
        raise PresenterRenderError(f"The {what} has no audio track.")
    label = _draw_label(workdir / "label.png", width, height)
    background = "0x%02X%02X%02X" % BRAND_BACKGROUND
    # The label is a transparent PNG laid over the frame: drawtext would
    # need a font file path, which is not portable to Windows.
    graph = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={background},"
        f"fps={fps},setsar=1[base];[base][1:v]overlay=0:0,format=yuv420p[v]"
    )
    _encode([
        "-i", str(raw), "-i", str(label), "-filter_complex", graph,
        "-map", "[v]", "-map", "0:a:0", "-af", NARRATION_FILTER,
    ], output_path, f"{what} branding")


def credits_card(
    lines: Sequence[str],
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    seconds: float = 4.0,
) -> Path:
    """Closing card naming the synthetic presenter, with a silent audio track."""
    with tempfile.TemporaryDirectory() as tmp:
        card = _draw_card(list(lines), Path(tmp) / "credits.png", width, height)
        _still_clip(card, output_path, fps=fps, seconds=max(seconds, MIN_CLIP_SECONDS), audio=None)
    return output_path


def synthetic_credit_lines(profile: Optional[ProfessorProfile], used_avatar: bool) -> list[str]:
    if not used_avatar:
        return []
    if profile is None:
        return ["Presenter: AI-generated avatar."]
    return [f"Presenter: AI-generated avatar of {profile.name}, used with consent."]


# ── 5. rendering one approved bridge ────────────────────────────────────

@dataclass
class PresenterBudget:
    """Seconds of presenter video allowed in one run (``max_presenter_seconds``)."""

    max_seconds: float
    spent_seconds: float = 0.0

    def would_exceed(self, seconds: float) -> bool:
        return self.spent_seconds + max(seconds, 0.0) > self.max_seconds + 1e-9

    def add(self, seconds: float) -> None:
        self.spent_seconds += max(seconds, 0.0)


def render_bridge(
    bridge: BridgeProposal,
    provider: PresenterProvider,
    request: PresenterRequest,
    output_path: Path,
    *,
    budget: PresenterBudget,
    fallback: Optional[PresenterProvider] = None,
) -> Optional[PresenterClip]:
    """Render one approved bridge, or return ``None`` when it is skipped.

    Contract:
    - an unapproved bridge raises ValueError (nothing is rendered or paid);
    - ``None`` means the bridge was skipped because its expected length
      (``request.duration_seconds``, else ``bridge.estimated_seconds``) would
      pass the presenter budget; no provider was called and nothing was spent;
    - when ``provider`` fails with PresenterRenderError and ``fallback`` is
      given, the fallback clip is returned with the redacted failure in
      ``detail["fallback_reason"]``; without a fallback the error propagates;
    - the budget is charged the rendered clip's real duration.
    """
    if not bridge.approved:
        raise ValueError(f"{bridge.bridge_id} is not approved and must not be rendered.")
    expected = request.duration_seconds if request.duration_seconds > 0 else bridge.estimated_seconds
    if budget.would_exceed(expected):
        return None
    try:
        clip = provider.render(request, output_path)
    except PresenterRenderError as exc:
        if fallback is None or fallback is provider:
            raise
        clip = fallback.render(request, output_path)
        clip.detail["fallback_reason"] = str(exc)
        clip.detail["fallback_from"] = provider.name
    budget.add(clip.duration_seconds)
    return clip


def bridge_fingerprint(bridge: BridgeProposal, settings: PresenterSettings) -> dict[str, Any]:
    """What a bridge clip depends on, for the pipeline's task input hash. No secrets."""
    return {
        "bridge_id": bridge.bridge_id,
        "kind": bridge.kind,
        "before_scene_id": bridge.before_scene_id,
        "text": bridge.text,
        "provider": settings.provider,
        "layout": settings.layout,
        "voice_id": settings.voice_id if settings.provider == "d-id" else None,
        "avatar_image_url": settings.avatar_image_url if settings.provider == "d-id" else None,
        "voice": settings.voice if settings.provider == "gemini" else None,
        "persona": settings.persona if settings.provider == "gemini" else None,
        "frame": [settings.width, settings.height, settings.fps],
    }


# ── 6. timeline ─────────────────────────────────────────────────────────

def interleave_segments(
    scene_segments: list[TimelineSegment],
    bridge_clips: Mapping[str, tuple[BridgeProposal, PresenterClip]],
    credits: Optional[tuple[Path, float]] = None,
) -> list[TimelineSegment]:
    """A new timeline with each rendered bridge placed before its scene.

    Scene segments are copied with only ``start_seconds`` recomputed; none is
    removed or replaced. A bridge whose anchor scene is absent (the scene
    failed) is dropped silently, and a timeline with no scene gets no bridge
    and no credits: presenter clips never stand in for missing Manim scenes.
    """
    if not scene_segments:
        return []
    before: dict[Optional[str], list[TimelineSegment]] = {}
    for bridge_id, (bridge, clip) in bridge_clips.items():
        if not bridge.approved:
            raise ValueError(f"{bridge_id} is not approved and cannot enter the timeline.")
        before.setdefault(bridge.before_scene_id, []).append(TimelineSegment(
            segment_id=f"segment-{bridge_id}", scene_id=bridge_id, mode="presenter",
            start_seconds=0.0, duration_seconds=clip.duration_seconds,
            video_path=str(clip.path), audio_path=None,
        ))
    ordered: list[TimelineSegment] = []
    for segment in scene_segments:
        ordered.extend(before.get(segment.scene_id, []))
        ordered.append(segment)
    ordered.extend(before.get(None, []))
    if credits is not None:
        ordered.append(TimelineSegment(
            segment_id="segment-credits", scene_id="credits", mode="transition",
            start_seconds=0.0, duration_seconds=credits[1], video_path=str(credits[0]),
        ))
    timeline: list[TimelineSegment] = []
    cursor = 0.0
    for segment in ordered:
        timeline.append(segment.model_copy(update={"start_seconds": round(cursor, 6)}))
        cursor += segment.duration_seconds
    return timeline


# ── cards and encoding ──────────────────────────────────────────────────

def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _media_seconds(path: Path) -> Optional[float]:
    info = media_probe.probe(str(path))
    if info is None:
        return None
    for seconds in (info.duration, info.video.duration if info.video else None,
                    info.audio.duration if info.audio else None):
        if seconds and seconds > 0:
            return float(seconds)
    return None


def _font(size: int) -> Any:
    for name in _FONT_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has only the fixed bitmap font
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current else lines


def _draw_card(paragraphs: Sequence[str], path: Path, width: int, height: int) -> Path:
    """Centred, wrapped text on the brand background; the font shrinks to fit."""
    image = Image.new("RGB", (width, height), BRAND_BACKGROUND)
    draw = ImageDraw.Draw(image)
    size = max(18, height // 18)
    while True:
        font = _font(size)
        line_height = int(size * 1.4)
        lines: list[str] = []
        for paragraph in paragraphs:
            if lines:
                lines.append("")
            lines.extend(_wrap(draw, paragraph, font, width * 0.8))
        if line_height * len(lines) <= height * 0.8 or size <= 12:
            break
        size = int(size * 0.85)
    top = (height - line_height * len(lines)) // 2
    for index, line in enumerate(lines):
        left = (width - draw.textlength(line, font=font)) / 2
        draw.text((left, top + index * line_height), line, font=font, fill=TEXT_COLOR)
    image.save(path)
    return path


def _draw_label(path: Path, width: int, height: int) -> Path:
    """Full-frame transparent overlay with the disclosure label top-left."""
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    size = max(12, height // 36)
    font = _font(size)
    pad = max(6, height // 60)
    left, top, right, bottom = draw.textbbox((pad * 2, pad * 2), DISCLOSURE_LABEL, font=font)
    draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=(0, 0, 0, 170))
    draw.text((pad * 2, pad * 2), DISCLOSURE_LABEL, font=font, fill=(255, 255, 255, 235))
    image.save(path)
    return path


def _still_clip(image: Path, output_path: Path, *, fps: int, seconds: float, audio: Optional[Path]) -> None:
    """A held image with an audio track: the narration, or silence."""
    inputs = ["-loop", "1", "-framerate", str(fps), "-i", str(image)]
    if audio is not None:
        # apad + -t: both streams end together even if the audio is a frame short.
        inputs += ["-i", str(audio), "-af", f"{NARRATION_FILTER},apad"]
    else:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    _encode([
        *inputs, "-map", "0:v:0", "-map", "1:a:0", "-vf", "format=yuv420p",
        "-r", str(fps), "-t", f"{seconds:.3f}", "-tune", "stillimage",
    ], output_path, "presenter card")


def _encode(args: list[str], output_path: Path, label: str) -> None:
    """Encode to a sibling file and move it into place, so no partial clip is left."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    try:
        _ffmpeg([*args, *_ENCODE_ARGS, str(partial)], label)
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)


def _ffmpeg(args: list[str], label: str, timeout: int = 300) -> None:
    try:
        result = subprocess.run(
            [get_ffmpeg(), "-v", "error", "-y", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PresenterRenderError(f"{label} failed: {type(exc).__name__}") from None
    if result.returncode != 0:
        raise PresenterRenderError(f"{label} failed: {(result.stderr or '').strip()[-600:]}")
