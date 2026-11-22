"""Incremental revisions of a finished V2 run.

A revision edits the run's storyboard in place and reruns only the affected
scenes through the existing scene-retry machinery. Each scene gets the
cheapest action that satisfies the edit:

* narration edits re-record that scene and regenerate its code;
* pronunciation changes re-record only scenes that say the word and keep
  their code, re-timed to the new audio;
* pacing changes keep the paid recording (re-paced or re-timed locally) and
  keep the code;
* visual corrections keep the audio and regenerate the code with the
  professor's instruction.

Everything else (planning, other scenes' audio, code and clips) is reused.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from src.utils.fingerprints import apply_lexicon, lexicon_for

from .model_config import CODE_MODEL, GEMINI_TTS_TOKENS_PER_SECOND, TTS_MODEL
from .models import LessonPlan, NarratedScene, NarrationBeat, NarrationPlan, SceneResult
from .providers import price
from .tts import RELAXED_SPEECH_RATE, SPEECH_RATE, gemini_input_units, tts_key_name


REVISION_DIR = "revisions"
REVISION_STATE_FILE = f"{REVISION_DIR}/state.json"
# After the first revision the storyboard, not the cached planner output, is
# the source of truth; a plain resume reads planning from this file.
FROZEN_PLANNING_FILE = f"{REVISION_DIR}/planning.json"

PACING_SPEECH_RATE = {"standard": SPEECH_RATE, "relaxed": RELAXED_SPEECH_RATE}
PACING_WORDS_PER_MINUTE = {"standard": 135, "relaxed": 115}

_AUDIO_ORDER = ("reuse", "retime", "repace", "regenerate")
_CODE_ORDER = ("reuse", "reinstrument", "regenerate")
_CODE_MODEL = ("google", CODE_MODEL)
# Used only when the run has no cost history for the stage.
_DEFAULT_CODE_USD = price(*_CODE_MODEL, 2500, 9000)
_DEFAULT_REPAIR_USD = price(*_CODE_MODEL, 4000, 9000)


class RevisionError(ValueError):
    pass


class EditBeat(BaseModel):
    """Replace one narration beat's spoken text."""

    op: Literal["edit_beat"] = "edit_beat"
    beat_id: str
    text: str = Field(min_length=1)


class AddBeat(BaseModel):
    """Insert a narration beat after an existing one, citing the same facts."""

    op: Literal["add_beat"] = "add_beat"
    after_beat_id: str
    text: str = Field(min_length=1)
    visual_action: str = ""


class RemoveBeat(BaseModel):
    op: Literal["remove_beat"] = "remove_beat"
    beat_id: str


class SetPause(BaseModel):
    """Change the pause after one beat. Keeps the recording."""

    op: Literal["set_pause"] = "set_pause"
    beat_id: str
    pause_seconds: float = Field(ge=0.0, le=1.5)


class SetPacing(BaseModel):
    """Change speech pacing for some scenes (all when scene_ids is empty)."""

    op: Literal["set_pacing"] = "set_pacing"
    pacing: Literal["standard", "relaxed"]
    scene_ids: list[str] = Field(default_factory=list)


class Pronounce(BaseModel):
    """Have TTS say ``word`` as ``say_as``. On-screen text is unchanged."""

    op: Literal["pronounce"] = "pronounce"
    word: str = Field(min_length=1)
    say_as: str = Field(min_length=1)


class FixVisual(BaseModel):
    """Regenerate one scene's visuals with an instruction. Keeps the audio."""

    op: Literal["fix_visual"] = "fix_visual"
    scene_id: Optional[str] = None
    beat_id: Optional[str] = None
    instruction: str = Field(min_length=1)

    @model_validator(mode="after")
    def one_target(self) -> "FixVisual":
        if bool(self.scene_id) == bool(self.beat_id):
            raise ValueError("fix_visual needs exactly one of scene_id or beat_id")
        return self


RevisionOp = Annotated[
    Union[EditBeat, AddBeat, RemoveBeat, SetPause, SetPacing, Pronounce, FixVisual],
    Field(discriminator="op"),
]
_OPS = TypeAdapter(list[RevisionOp])


def parse_ops(data: Any) -> list[RevisionOp]:
    """Validate ops given as a list or as ``{"ops": [...]}``."""

    if isinstance(data, dict):
        data = data.get("ops")
    if not isinstance(data, list) or not data:
        raise RevisionError("A revision needs a non-empty list of ops")
    try:
        return _OPS.validate_python(data)
    except ValueError as exc:
        raise RevisionError(f"Invalid revision ops: {exc}") from exc


def load_ops(path: str | Path) -> list[RevisionOp]:
    """Read ops from a YAML or JSON file."""

    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RevisionError(f"Could not parse {path}: {exc}") from exc
    return parse_ops(data)


class SceneAction(BaseModel):
    scene_id: str
    audio: Literal["reuse", "retime", "repace", "regenerate"] = "reuse"
    code: Literal["reuse", "reinstrument", "regenerate"] = "reuse"
    instruction: str = ""
    reasons: list[str] = Field(default_factory=list)


class RevisionEstimate(BaseModel):
    tts_calls: int = 0
    tts_usd: float = 0.0
    code_calls: int = 0
    code_usd: float = 0.0
    total_usd: float = 0.0
    # Vision repair only runs if a rebuilt scene fails verification.
    repair_usd_upper_bound: float = 0.0
    scenes_rebuilt: int = 0
    scenes_reused: int = 0
    length_delta_seconds: float = 0.0


class RevisionPlan(BaseModel):
    run_directory: str
    revision_number: int
    ops: list[dict[str, Any]]
    actions: list[SceneAction]
    reused_scene_ids: list[str]
    lexicon: dict[str, str] = Field(default_factory=dict)
    speech_rates: dict[str, float] = Field(default_factory=dict)
    changes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimate: RevisionEstimate = Field(default_factory=RevisionEstimate)

    @property
    def affected_scene_ids(self) -> list[str]:
        return [action.scene_id for action in self.actions]

    def summary_lines(self) -> list[str]:
        lines = list(self.changes)
        lines.extend(f"Warning: {item}" for item in self.warnings)
        estimate = self.estimate
        lines.append(
            f"Estimate: {estimate.scenes_rebuilt} scene(s) rebuilt, "
            f"{estimate.scenes_reused} reused; {estimate.tts_calls} narration call(s) "
            f"${estimate.tts_usd:.4f}, {estimate.code_calls} code call(s) "
            f"${estimate.code_usd:.4f}; about ${estimate.total_usd:.4f} total "
            f"(up to ${estimate.repair_usd_upper_bound:.4f} more if repairs are needed); "
            f"length {estimate.length_delta_seconds:+.1f}s."
        )
        return lines


@dataclass
class RunState:
    run_dir: Path
    config: dict[str, Any]
    lesson: LessonPlan
    narration: NarrationPlan
    scene_results: dict[str, SceneResult]
    lexicon: dict[str, str] = field(default_factory=dict)
    speech_rates: dict[str, float] = field(default_factory=dict)
    revision_count: int = 0
    pending: Optional[dict[str, Any]] = None


@dataclass
class RevisedStoryboard:
    narration: NarrationPlan
    lexicon: dict[str, str]
    speech_rates: dict[str, float]
    actions: dict[str, SceneAction]
    changes: list[str]
    warnings: list[str]


def read_revision_state(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / REVISION_STATE_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_revision_state(run_dir: Path, state: dict[str, Any]) -> None:
    path = Path(run_dir) / REVISION_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_run_state(run_dir: str | Path) -> RunState:
    path = Path(run_dir).resolve()
    run_json = path / "run.json"
    storyboard_json = path / "storyboard.json"
    if not run_json.is_file() or not storyboard_json.is_file() or not (path / "run.db").is_file():
        raise RevisionError(f"Not a complete V2 run directory: {path}")
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    storyboard = json.loads(storyboard_json.read_text(encoding="utf-8"))
    results = {
        item["scene_id"]: SceneResult.model_validate(item)
        for item in (payload.get("result") or {}).get("scene_results", [])
    }
    state = read_revision_state(path)
    return RunState(
        run_dir=path,
        config=dict(payload.get("config") or {}),
        lesson=LessonPlan.model_validate(storyboard["lesson"]),
        narration=NarrationPlan.model_validate(storyboard["narration"]),
        scene_results=results,
        lexicon=dict(state.get("lexicon") or {}),
        speech_rates={key: float(value) for key, value in (state.get("speech_rates") or {}).items()},
        revision_count=int(state.get("revision", 0)),
        pending=state.get("pending"),
    )


def untimed(scene: NarratedScene) -> NarratedScene:
    """The scene as the planner would hand it to TTS: no audio, no timings."""

    return scene.model_copy(update={
        "beats": [
            beat.model_copy(update={"start_seconds": 0.0, "end_seconds": 0.0})
            for beat in scene.beats
        ],
        "audio_path": None,
        "audio_duration_seconds": None,
        "audio_tail_seconds": 0.0,
        "beat_speech_spans": [],
    })


def tts_text(scene: NarratedScene, lexicon: Optional[dict[str, str]] = None) -> str:
    """Spoken text for one scene with pronunciations applied."""

    paragraphs = []
    for beat in scene.beats:
        text = beat.text.strip()
        if text and text[-1] not in ".!?":
            text += "."
        paragraphs.append(text)
    return apply_lexicon("\n\n".join(paragraphs), lexicon_for(scene.narration, lexicon))


def revise_storyboard(
    narration: NarrationPlan,
    ops: list[RevisionOp],
    *,
    lexicon: Optional[dict[str, str]] = None,
    speech_rates: Optional[dict[str, float]] = None,
    default_pacing: str = "standard",
) -> RevisedStoryboard:
    """Apply ops to a narration plan and decide the cheapest action per scene."""

    scenes = {scene.scene_id: scene for scene in narration.scenes}
    order = [scene.scene_id for scene in narration.scenes]
    new_lexicon = dict(lexicon or {})
    new_rates = dict(speech_rates or {})
    actions: dict[str, SceneAction] = {}
    changes: list[str] = []
    warnings: list[str] = []
    content_changed: set[str] = set()

    def action(scene_id: str) -> SceneAction:
        return actions.setdefault(scene_id, SceneAction(scene_id=scene_id))

    def raise_audio(scene_id: str, level: str, reason: str) -> None:
        item = action(scene_id)
        if _AUDIO_ORDER.index(level) > _AUDIO_ORDER.index(item.audio):
            item.audio = level
        # New audio timing always needs the code re-timed against it.
        if _CODE_ORDER.index(item.code) < _CODE_ORDER.index("reinstrument"):
            item.code = "reinstrument"
        item.reasons.append(reason)

    def raise_code(scene_id: str, reason: str) -> None:
        item = action(scene_id)
        item.code = "regenerate"
        item.reasons.append(reason)

    def locate(beat_ref: str) -> tuple[str, int]:
        wanted_scene, _, wanted_beat = beat_ref.rpartition(":")
        matches = [
            (scene_id, index)
            for scene_id in order
            for index, beat in enumerate(scenes[scene_id].beats)
            if beat.beat_id == wanted_beat and wanted_scene in ("", scene_id)
        ]
        if not matches:
            raise RevisionError(f"Unknown beat: {beat_ref}")
        if len(matches) > 1:
            options = ", ".join(f"{scene_id}:{wanted_beat}" for scene_id, _ in matches)
            raise RevisionError(f"Beat id {beat_ref!r} is ambiguous; use one of: {options}")
        return matches[0]

    def replace_beats(scene_id: str, beats: list[NarrationBeat]) -> None:
        scenes[scene_id] = scenes[scene_id].model_copy(update={"beats": beats})

    for op in ops:
        if isinstance(op, EditBeat):
            scene_id, index = locate(op.beat_id)
            beats = list(scenes[scene_id].beats)
            if beats[index].text.strip() == op.text.strip():
                warnings.append(f"{op.beat_id} already has that text; nothing to change.")
                continue
            beats[index] = beats[index].model_copy(update={"text": op.text.strip()})
            replace_beats(scene_id, beats)
            content_changed.add(scene_id)
            raise_audio(scene_id, "regenerate", f"narration of {op.beat_id} edited")
            raise_code(scene_id, "visuals follow the edited narration")
            changes.append(f"{scene_id}: rewrote {beats[index].beat_id}; re-record and regenerate.")
        elif isinstance(op, AddBeat):
            scene_id, index = locate(op.after_beat_id)
            beats = list(scenes[scene_id].beats)
            anchor = beats[index]
            used = {beat.beat_id for scene in scenes.values() for beat in scene.beats}
            suffix = 1
            while f"{anchor.beat_id}_add{suffix}" in used:
                suffix += 1
            beats.insert(index + 1, NarrationBeat(
                beat_id=f"{anchor.beat_id}_add{suffix}", text=op.text.strip(),
                source_fact_ids=list(anchor.source_fact_ids), role="explain",
                visual_action=op.visual_action,
            ))
            replace_beats(scene_id, beats)
            content_changed.add(scene_id)
            raise_audio(scene_id, "regenerate", f"beat added after {anchor.beat_id}")
            raise_code(scene_id, "visuals follow the added narration")
            changes.append(f"{scene_id}: added a beat after {anchor.beat_id}; re-record and regenerate.")
        elif isinstance(op, RemoveBeat):
            scene_id, index = locate(op.beat_id)
            beats = list(scenes[scene_id].beats)
            if len(beats) == 1:
                raise RevisionError(
                    f"{op.beat_id} is the only beat of {scene_id}; a scene needs at least one beat."
                )
            removed = beats.pop(index)
            replace_beats(scene_id, beats)
            content_changed.add(scene_id)
            raise_audio(scene_id, "regenerate", f"beat {removed.beat_id} removed")
            raise_code(scene_id, "visuals follow the shortened narration")
            changes.append(f"{scene_id}: removed {removed.beat_id}; re-record and regenerate.")
        elif isinstance(op, SetPause):
            scene_id, index = locate(op.beat_id)
            beats = list(scenes[scene_id].beats)
            if abs(beats[index].pause_after_seconds - op.pause_seconds) < 1e-6:
                warnings.append(f"{op.beat_id} already pauses {op.pause_seconds:g}s; nothing to change.")
                continue
            beats[index] = beats[index].model_copy(update={"pause_after_seconds": op.pause_seconds})
            replace_beats(scene_id, beats)
            raise_audio(scene_id, "retime", f"pause after {op.beat_id} set to {op.pause_seconds:g}s")
            changes.append(
                f"{scene_id}: pause after {beats[index].beat_id} is now {op.pause_seconds:g}s; "
                "keep the recording, re-time and re-render."
            )
        elif isinstance(op, SetPacing):
            unknown = sorted(set(op.scene_ids) - set(order))
            if unknown:
                raise RevisionError(f"Unknown scene IDs: {', '.join(unknown)}")
            rate = PACING_SPEECH_RATE[op.pacing]
            words_per_minute = PACING_WORDS_PER_MINUTE[op.pacing]
            for scene_id in op.scene_ids or order:
                current = new_rates.get(scene_id, PACING_SPEECH_RATE[default_pacing])
                if abs(current - rate) < 1e-6:
                    continue
                new_rates[scene_id] = rate
                scenes[scene_id] = scenes[scene_id].model_copy(update={
                    "target_words_per_minute": words_per_minute,
                })
                raise_audio(scene_id, "repace", f"pacing set to {op.pacing}")
                changes.append(
                    f"{scene_id}: pacing is now {op.pacing}; re-pace the recording and re-render."
                )
        elif isinstance(op, Pronounce):
            if new_lexicon.get(op.word) == op.say_as:
                warnings.append(f"{op.word!r} is already pronounced {op.say_as!r}.")
                continue
            new_lexicon[op.word] = op.say_as
            spoken_in = [
                scene_id for scene_id in order
                if lexicon_for(scenes[scene_id].narration, {op.word: op.say_as})
                or any(lexicon_for(beat.text, {op.word: op.say_as}) for beat in scenes[scene_id].beats)
            ]
            if not spoken_in:
                warnings.append(f"No scene says {op.word!r}; the pronunciation is saved for later edits.")
            for scene_id in spoken_in:
                raise_audio(scene_id, "regenerate", f"{op.word!r} is now pronounced {op.say_as!r}")
                changes.append(
                    f"{scene_id}: says {op.word!r} as {op.say_as!r}; re-record, keep the visuals."
                )
        elif isinstance(op, FixVisual):
            scene_id = op.scene_id if op.scene_id else locate(op.beat_id or "")[0]
            if scene_id not in scenes:
                raise RevisionError(f"Unknown scene ID: {scene_id}")
            item = action(scene_id)
            target = f" (at {op.beat_id})" if op.beat_id else ""
            item.instruction = " ".join(
                part for part in (item.instruction, op.instruction.strip() + target) if part
            )
            raise_code(scene_id, f"visual correction: {op.instruction.strip()}")
            changes.append(f"{scene_id}: visual fix; keep the narration audio, regenerate the visuals.")

    revised = []
    for scene_id in order:
        scene = scenes[scene_id]
        if scene_id in content_changed:
            scene = scene.model_copy(update={
                "narration": " ".join(beat.text for beat in scene.beats),
            })
        item = actions.get(scene_id)
        if item is not None and item.audio != "reuse":
            scene = untimed(scene)
        revised.append(scene)
    actions = {
        scene_id: item for scene_id, item in actions.items()
        if item.audio != "reuse" or item.code != "reuse"
    }
    return RevisedStoryboard(
        narration=NarrationPlan(scenes=revised),
        lexicon=new_lexicon,
        speech_rates=new_rates,
        actions=actions,
        changes=changes,
        warnings=warnings,
    )


def raw_audio_path(run_dir: Path, scene: NarratedScene) -> Optional[Path]:
    """The un-paced recording behind a scene's current audio, if it survives."""

    if not scene.audio_path:
        return None
    current = Path(scene.audio_path)
    name = f"{scene.scene_id}.mp3"
    for candidate in (current.parent / name, Path(run_dir) / "scenes" / scene.scene_id / "audio" / name):
        if candidate.is_file() and candidate.resolve() != current.resolve():
            return candidate
    return None


def plan_revision(run_dir: str | Path, ops: list[RevisionOp]) -> RevisionPlan:
    """Describe what a revision would change and cost. Writes nothing."""

    state = load_run_state(run_dir)
    return build_revision(state, ops)[0]


def build_revision(
    state: RunState,
    ops: list[RevisionOp],
) -> tuple[RevisionPlan, RevisedStoryboard]:
    """Plan a revision and return the revised storyboard it would write."""

    if state.pending:
        raise RevisionError(
            "A previous revision of this run did not finish. Resume the run before revising again."
        )
    missing = [
        scene.scene_id for scene in state.lesson.scenes if scene.scene_id not in state.scene_results
    ]
    if missing:
        raise RevisionError("Run does not contain reusable results for: " + ", ".join(missing))
    default_pacing = str(state.config.get("pacing", "standard"))
    revised = revise_storyboard(
        state.narration, ops, lexicon=state.lexicon, speech_rates=state.speech_rates,
        default_pacing=default_pacing,
    )
    previous = {scene.scene_id: scene for scene in state.narration.scenes}
    tts_enabled = bool(state.config.get("enable_tts", True))
    tts_key = tts_key_name(str(state.config.get("tts_provider", "gemini")))
    has_tts_key = bool(os.getenv(tts_key)) and tts_enabled
    recorded = tts_enabled and has_recorded_narration(state.run_dir)
    for scene_id, item in revised.actions.items():
        # Re-pacing needs the original un-paced recording; without it the
        # scene has to be recorded again, and the estimate must say so.
        if item.audio == "repace" and recorded and raw_audio_path(state.run_dir, previous[scene_id]) is None:
            item.audio = "regenerate"
            item.reasons.append("original recording unavailable; narration is re-recorded")
            revised.warnings.append(
                f"{scene_id}: the original recording is missing, so re-pacing re-records it."
            )
    rerecorded = [item.scene_id for item in revised.actions.values() if item.audio == "regenerate"]
    if recorded and rerecorded and not has_tts_key:
        # Falling back to silent timing audio here would mute scenes of a
        # video whose other scenes are narrated.
        raise RevisionError(
            f"This run has recorded narration; {tts_key} is required to re-record "
            + ", ".join(rerecorded) + "."
        )
    order = [scene.scene_id for scene in revised.narration.scenes]
    actions = [revised.actions[scene_id] for scene_id in order if scene_id in revised.actions]
    plan = RevisionPlan(
        run_directory=str(state.run_dir),
        revision_number=state.revision_count + 1,
        ops=[op.model_dump(mode="json") for op in ops],
        actions=actions,
        reused_scene_ids=[scene_id for scene_id in order if scene_id not in revised.actions],
        lexicon=revised.lexicon,
        speech_rates=revised.speech_rates,
        changes=revised.changes,
        warnings=revised.warnings,
    )
    plan.estimate = _estimate(state, revised, plan, has_tts_key)
    return plan, revised


def _estimate(
    state: RunState,
    revised: RevisedStoryboard,
    plan: RevisionPlan,
    has_tts_key: bool,
) -> RevisionEstimate:
    previous = {scene.scene_id: scene for scene in state.narration.scenes}
    current = {scene.scene_id: scene for scene in revised.narration.scenes}
    history = _stage_cost_history(state.run_dir / "run.db")
    has_google = bool(os.getenv("GEMINI_API_KEY"))
    gemini_tts = str(state.config.get("tts_provider", "gemini")) == "gemini"
    tts_model = "tts-1-hd" if state.config.get("quality") == "high" else "tts-1"
    default_rate = PACING_SPEECH_RATE[str(state.config.get("pacing", "standard"))]
    estimate = RevisionEstimate(
        scenes_rebuilt=len(plan.actions), scenes_reused=len(plan.reused_scene_ids),
    )
    for item in plan.actions:
        before, after = previous[item.scene_id], current[item.scene_id]
        if item.audio == "regenerate" and has_tts_key:
            estimate.tts_calls += 1
            spoken = tts_text(after, revised.lexicon)
            if gemini_tts:
                # The same reservation the Gemini TTS call makes.
                seconds = len(spoken.split()) * 60 / after.target_words_per_minute + 5
                estimate.tts_usd += price(
                    "google", TTS_MODEL,
                    gemini_input_units(spoken.split("\n\n"), state.config.get("audience")),
                    seconds * GEMINI_TTS_TOKENS_PER_SECOND,
                )
            else:
                estimate.tts_usd += price("openai", tts_model, len(spoken), 0)
        if item.code == "regenerate" and has_google:
            estimate.code_calls += 1
            estimate.code_usd += history.get("code_generation", _DEFAULT_CODE_USD)
        if has_google and state.config.get("repair", "eligible") == "eligible":
            estimate.repair_usd_upper_bound += (
                int(state.config.get("max_repairs_per_scene", 3))
                * history.get("vision_repair", _DEFAULT_REPAIR_USD)
            )
        word_delta = len(after.narration.split()) - len(before.narration.split())
        estimate.length_delta_seconds += word_delta * 60.0 / after.target_words_per_minute
        old_rate = state.speech_rates.get(item.scene_id, default_rate)
        new_rate = revised.speech_rates.get(item.scene_id, default_rate)
        if before.audio_duration_seconds and abs(old_rate - new_rate) > 1e-6:
            estimate.length_delta_seconds += before.audio_duration_seconds * (old_rate / new_rate - 1.0)
        if not has_tts_key:
            # Paid recordings carry no explicit pauses; only the local silent
            # timing track is lengthened by them.
            estimate.length_delta_seconds += sum(
                beat.pause_after_seconds for beat in after.beats
            ) - sum(beat.pause_after_seconds for beat in before.beats)
    estimate.tts_usd = round(estimate.tts_usd, 6)
    estimate.code_usd = round(estimate.code_usd, 6)
    estimate.repair_usd_upper_bound = round(estimate.repair_usd_upper_bound, 6)
    estimate.total_usd = round(estimate.tts_usd + estimate.code_usd, 6)
    estimate.length_delta_seconds = round(estimate.length_delta_seconds, 2)
    return estimate


def _stage_cost_history(database: Path) -> dict[str, float]:
    """Mean actual cost per paid call for each stage of this run."""

    try:
        connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        rows = connection.execute(
            "SELECT stage, AVG(actual_usd) FROM api_calls "
            "WHERE actual_usd IS NOT NULL AND actual_usd > 0 GROUP BY stage"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        connection.close()
    return {str(stage): float(mean) for stage, mean in rows}


def has_recorded_narration(run_dir: Path) -> bool:
    """True when the run paid for real narration (not the silent timing fallback)."""

    return "tts" in _stage_cost_history(Path(run_dir) / "run.db")


def frozen_planning(
    lesson: LessonPlan,
    revised: NarrationPlan,
    affected_audio: set[str],
    base: Optional[NarrationPlan],
) -> dict[str, Any]:
    """Planner-shaped (untimed) storyboard that a plain resume restarts from.

    Unaffected scenes keep the exact planner output when it is available, so
    their TTS task hashes are unchanged and their paid audio is reused.
    """

    base_scenes = {scene.scene_id: scene for scene in base.scenes} if base else {}
    scenes = []
    for scene in revised.scenes:
        original = base_scenes.get(scene.scene_id)
        same_script = original is not None and (
            [(beat.beat_id, beat.text, beat.pause_after_seconds) for beat in original.beats]
            == [(beat.beat_id, beat.text, beat.pause_after_seconds) for beat in scene.beats]
            and original.target_words_per_minute == scene.target_words_per_minute
        )
        if scene.scene_id not in affected_audio and same_script:
            scenes.append(original)
        else:
            scenes.append(untimed(scene))
    return {
        "lesson": lesson.model_dump(mode="json"),
        "narration": NarrationPlan(scenes=scenes).model_dump(mode="json"),
    }


def load_frozen_planning(run_dir: Path) -> Optional[tuple[LessonPlan, NarrationPlan]]:
    path = Path(run_dir) / FROZEN_PLANNING_FILE
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        LessonPlan.model_validate(payload["lesson"]),
        NarrationPlan.model_validate(payload["narration"]),
    )


def revision_directory(run_dir: Path, number: int) -> Path:
    return Path(run_dir) / REVISION_DIR / f"rev-{number:03d}"
