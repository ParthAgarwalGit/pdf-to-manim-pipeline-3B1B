"""Helpers for revision runs (Orchestrator.run_revision).

A revision rebuilds a finished run from an edited lesson script. It reuses
every artifact whose inputs are unchanged (see src/utils/fingerprints.py) and
writes a new, self-contained run directory, ``<run>/rev_NNN``, so the parent
run is never modified. These helpers are pure file and schema plumbing; the
orchestrator owns the stage sequencing.
"""

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.schemas.animation_code import AnimationCodeObject
from src.schemas.output_manifest import OutputFile, OutputManifest, RunConfig, SceneRecord
from src.schemas.storyboard import Scene, StoryboardJSON
from src.schemas.video_script import Analogy, VideoScript
from src.utils.fingerprints import (
    beat_fingerprint,
    beats_fingerprint,
    scene_narration_fingerprint,
    scene_visual_fingerprint,
    visual_fingerprint,
)

logger = logging.getLogger(__name__)

LESSON_SCRIPT_FILENAME = "lesson_script.json"
REVISION_FILENAME = "revision.json"
_REVISION_DIR_RE = re.compile(r"^rev_(\d{3,})$")
_SCENE_BASE_CLASSES = ("Scene", "ThreeDScene", "MovingCameraScene")


@dataclass
class BaseRun:
    """A finished run loaded from disk, ready to be revised."""

    run_dir: Path
    storyboard: StoryboardJSON
    script: VideoScript
    run_config: Optional[RunConfig] = None
    records: Dict[str, SceneRecord] = field(default_factory=dict)
    cost_per_scene: float = 0.0
    delivery_report: Optional[Dict[str, Any]] = None

    def artifact(self, rel_path: Optional[str]) -> Optional[Path]:
        """Absolute path of a recorded artifact, or None if it is missing."""
        if not rel_path:
            return None
        path = self.run_dir / rel_path
        return path if path.is_file() else None


def lesson_script_from_storyboard(
    storyboard: StoryboardJSON,
    title: str = "",
    analogies: Iterable[Analogy] = (),
) -> Optional[VideoScript]:
    """The lesson script is exactly the beats the storyboard's scenes carry.

    Deriving it from the storyboard (rather than keeping the planner's copy)
    guarantees the two never disagree. Returns None for legacy storyboards
    whose scenes carry no beats.
    """
    beats = [beat for scene in storyboard.scenes for beat in scene.source_beats]
    if not beats:
        return None
    used = {beat.analogy_id for beat in beats if beat.analogy_id}
    return VideoScript(
        title=title or "",
        beats=[beat.model_copy(deep=True) for beat in beats],
        analogies=[a.model_copy(deep=True) for a in analogies if a.analogy_id in used],
    )


def load_lesson_script(path: Path) -> VideoScript:
    return VideoScript.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_base_run(run_dir: Path) -> BaseRun:
    """Load the storyboard, lesson script and scene records of a finished run.

    Runs made before lesson scripts and scene records existed still load:
    the script is derived from the storyboard and the records from the
    conventional artifact locations, so they can be revised too.
    """
    run_dir = Path(run_dir)
    storyboard_path = run_dir / "storyboard.json"
    if not storyboard_path.is_file():
        raise FileNotFoundError(f"No storyboard.json in {run_dir}; not a pipeline run directory.")
    storyboard = StoryboardJSON.model_validate_json(storyboard_path.read_text(encoding="utf-8"))

    # The base script's beats always come from the storyboard, which records
    # what was actually rendered: a professor may edit lesson_script.json in
    # place before revising, so that file can already hold the new version.
    title, analogies = "", []
    script_path = run_dir / LESSON_SCRIPT_FILENAME
    if script_path.is_file():
        try:
            saved = load_lesson_script(script_path)
            title, analogies = saved.title, saved.analogies
        except Exception as exc:
            logger.warning("Could not read %s (%s); using the storyboard's beats.", script_path, exc)
    script = lesson_script_from_storyboard(storyboard, title=title, analogies=analogies)
    if script is None:
        raise ValueError(
            f"{run_dir}'s storyboard carries no beats; it predates the unified "
            "script and cannot be revised."
        )

    manifest: Optional[OutputManifest] = None
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = OutputManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read %s (%s); artifacts will not be reused.", manifest_path, exc)

    run_config = manifest.run_config if manifest else None
    if manifest and manifest.scenes:
        records = {record.scene_id: record for record in manifest.scenes}
    else:
        records = _legacy_records(run_dir, storyboard, run_config)
    cost_per_scene = 0.0
    if manifest and manifest.cost_summary and manifest.total_scenes:
        cost_per_scene = float(manifest.cost_summary.total_usd) / manifest.total_scenes
    delivery = None
    report_path = run_dir / "delivery_report.json"
    if report_path.is_file():
        try:
            delivery = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError:
            delivery = None
    return BaseRun(
        run_dir=run_dir,
        storyboard=storyboard,
        script=script,
        run_config=run_config,
        records=records,
        cost_per_scene=cost_per_scene,
        delivery_report=delivery,
    )


def _legacy_records(
    run_dir: Path,
    storyboard: StoryboardJSON,
    run_config: Optional[RunConfig],
) -> Dict[str, SceneRecord]:
    """Scene records for a run written before the manifest stored them."""
    voice = run_config.tts_voice if run_config else "alloy"
    quality = run_config.render_quality if run_config else "medium"
    tts_first = bool(getattr(run_config, "tts_first", False))
    lexicon: Dict[str, str] = {}  # legacy runs predate the pronunciation lexicon
    records: Dict[str, SceneRecord] = {}
    for scene in storyboard.scenes:
        sid = scene.scene_id
        code = run_dir / f"scene_{sid}.py"
        # TTSAgent.process_all_scenes writes audio/<id>/<id>.mp3.
        audio = next(
            (
                p for p in (run_dir / "audio" / sid / f"{sid}.mp3", run_dir / "audio" / f"{sid}.mp3")
                if p.is_file()
            ),
            None,
        )
        video = next(
            (
                p for p in sorted((run_dir / "media" / "videos" / f"scene_{sid}").glob("**/*.mp4"))
                if "partial_movie_files" not in p.parts
            ),
            None,
        )
        records[sid] = SceneRecord(
            scene_id=sid,
            beat_ids=[b.beat_id for b in scene.source_beats],
            beats_fp=beats_fingerprint(scene.source_beats),
            narration_fp=scene_narration_fingerprint(scene, voice, lexicon),
            visual_fp=visual_fingerprint(scene, quality, tts_first),
            code_path=_relative(code, run_dir) if code.is_file() else None,
            video_path=_relative(video, run_dir) if video else None,
            audio_path=_relative(audio, run_dir) if audio else None,
        )
    return records


def build_scene_records(
    storyboard: StoryboardJSON,
    output_dir: Path,
    files: Iterable[OutputFile],
    render_results: Iterable[Any],
    tts_results: Iterable[Any],
    voice: str,
    render_quality: str,
    tts_first: bool,
    reused_video: Optional[Dict[str, str]] = None,
    reused_audio: Optional[Dict[str, str]] = None,
    lexicon: Optional[Dict[str, str]] = None,
    presenter_settings: Optional[Dict[str, Any]] = None,
) -> List[SceneRecord]:
    """One record per scene: its fingerprints and where its artifacts are."""
    output_dir = Path(output_dir)
    code_paths = {
        f.path[len("scene_"):-len(".py")]: f.path
        for f in files
        if f.type == "animation_code" and f.path.startswith("scene_") and f.path.endswith(".py")
    }
    renders = {
        r.scene_id: r for r in render_results
        if getattr(r, "success", False) and getattr(r, "video_path", None)
    }
    audio = {
        r.scene_id: r for r in tts_results
        if getattr(r, "audio_file_path", None)
    }
    reused_video = reused_video or {}
    reused_audio = reused_audio or {}
    records: List[SceneRecord] = []
    for scene in storyboard.scenes:
        sid = scene.scene_id
        render = renders.get(sid)
        tts = audio.get(sid)
        records.append(SceneRecord(
            scene_id=sid,
            beat_ids=[b.beat_id for b in scene.source_beats],
            beats_fp=beats_fingerprint(scene.source_beats),
            narration_fp=scene_narration_fingerprint(scene, voice, lexicon),
            visual_fp=scene_visual_fingerprint(scene, render_quality, tts_first, presenter_settings),
            code_path=code_paths.get(sid),
            video_path=_relative(Path(render.video_path), output_dir) if render else None,
            video_duration_seconds=getattr(render, "duration_seconds", None) if render else None,
            audio_path=_relative(Path(tts.audio_file_path), output_dir) if tts else None,
            audio_duration_seconds=getattr(tts, "duration_seconds", None) if tts else None,
            reused_video_from=reused_video.get(sid),
            reused_audio_from=reused_audio.get(sid),
        ))
    return records


def code_object_from_file(path: Path, scene: Scene) -> AnimationCodeObject:
    """Rebuild a scene's code object from the .py file Persistence wrote.

    The module docstring header (scene id, narration) is dropped because
    Persistence writes a fresh one for the scene's new id.
    """
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = list(tree.body)
    if (
        nodes
        and isinstance(nodes[0], ast.Expr)
        and isinstance(nodes[0].value, ast.Constant)
        and isinstance(nodes[0].value.value, str)
    ):
        nodes.pop(0)
    # Persistence writes the leading imports separately from the code body.
    imports: List[str] = []
    while nodes and isinstance(nodes[0], (ast.Import, ast.ImportFrom)):
        imports.append(ast.get_source_segment(source, nodes.pop(0)))
    start_line = nodes[0].lineno - 1 if nodes else len(source.splitlines())
    if nodes and getattr(nodes[0], "decorator_list", None):
        start_line = min(d.lineno for d in nodes[0].decorator_list) - 1
    body = "".join(source.splitlines(keepends=True)[start_line:])
    class_name = next(
        (
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(isinstance(b, ast.Name) and b.id in _SCENE_BASE_CLASSES for b in node.bases)
        ),
        None,
    )
    if class_name is None:
        raise ValueError(f"No Scene subclass in {path}")
    return AnimationCodeObject(
        scene_id=scene.scene_id,
        code=body.strip("\n"),
        imports=imports or ["from manim import *"],
        class_name=class_name,
        narration=scene.narration,
        duration_estimate=max(float(scene.duration_estimate), 0.1),
        generation_timestamp=datetime.now(timezone.utc),
    )


def revision_root(run_dir: Path) -> Path:
    """Revisions of a revision live beside it, under the original run."""
    run_dir = Path(run_dir)
    if _REVISION_DIR_RE.match(run_dir.name) and (run_dir / REVISION_FILENAME).is_file():
        return run_dir.parent
    return run_dir


def next_revision_dir(run_dir: Path) -> Path:
    """``rev_002`` for the first revision; the original run counts as rev 1."""
    root = revision_root(run_dir)
    taken = [
        int(match.group(1))
        for child in root.iterdir()
        if child.is_dir() and (match := _REVISION_DIR_RE.match(child.name))
    ]
    return root / f"rev_{max(taken + [1]) + 1:03d}"


def beat_diff(base: VideoScript, revised: VideoScript) -> Dict[str, List[str]]:
    """Beat ids added, removed, and changed (by content fingerprint)."""
    before = {b.beat_id: beat_fingerprint(b) for b in base.beats}
    after = {b.beat_id: beat_fingerprint(b) for b in revised.beats}
    return {
        "added": [bid for bid in after if bid not in before],
        "removed": [bid for bid in before if bid not in after],
        "changed": [bid for bid in after if bid in before and after[bid] != before[bid]],
    }


def estimate_revision(
    base: BaseRun,
    script: VideoScript,
    voice: Optional[str] = None,
    visual_fixes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """What a revision would rebuild, before anything is spent.

    Uses the same segment-run matching as the revision itself (no LLM): an
    unchanged run reuses its scenes; a changed run is assumed to rebuild a
    scene per ~25 s of narration. Scenes whose narration changes only in
    pauses or pronunciation are re-narrated, not regenerated.
    """
    from src.agents.script_architect import ScriptArchitectAgent as Architect

    voice = voice or (base.run_config.tts_voice if base.run_config else "alloy")
    base_runs: Dict[Any, List[Any]] = {}
    for segment_id, scenes in Architect._scene_runs(base.storyboard.scenes):
        fps = tuple(beat_fingerprint(b) for s in scenes for b in s.source_beats)
        base_runs.setdefault((segment_id, fps), scenes)
    fixed_beats = set(visual_fixes or {})
    rebuilt = reused = renarrated = 0
    for segment_id, beats in Architect._beat_runs(script.beats):
        fps = tuple(beat_fingerprint(b) for b in beats)
        scenes = base_runs.get((segment_id, fps))
        if scenes is None:
            words = sum(len(b.narration.split()) for b in beats)
            rebuilt += max(1, round(words / Architect.WORDS_PER_SECOND / Architect.TARGET_SCENE_DURATION))
            continue
        by_id = {b.beat_id: b for b in beats}
        for scene in scenes:
            ids = [b.beat_id for b in scene.source_beats]
            if fixed_beats & set(ids):
                rebuilt += 1
                continue
            probe = scene.model_copy(update={"source_beats": [by_id.get(i, b) for i, b in zip(ids, scene.source_beats)]})
            record = base.records.get(scene.scene_id)
            if record and record.narration_fp != scene_narration_fingerprint(probe, voice, script.pronunciations):
                renarrated += 1
            else:
                reused += 1
    old_words = sum(len(b.narration.split()) for b in base.script.beats)
    new_words = sum(len(b.narration.split()) for b in script.beats)
    return {
        "scenes_rebuilt": rebuilt,
        "scenes_renarrated": renarrated,
        "scenes_reused": reused,
        "length_delta_seconds": round((new_words - old_words) / Architect.WORDS_PER_SECOND, 1),
        "estimated_cost_usd": round(base.cost_per_scene * rebuilt, 2),
    }


def change_log(storyboard: StoryboardJSON, regenerated: List[str], retimed: List[str], reused_video: Dict[str, str]) -> str:
    """'Changed: scenes 4-5. Reused unchanged: 28 scenes.' for the professor."""
    index = {s.scene_id: i + 1 for i, s in enumerate(storyboard.scenes)}

    def ranges(ids: List[str]) -> str:
        numbers = sorted(index[i] for i in ids if i in index)
        spans, start = [], None
        for n in numbers:
            if start is None:
                start = prev = n
            elif n == prev + 1:
                prev = n
            else:
                spans.append((start, prev))
                start = prev = n
        if start is not None:
            spans.append((start, prev))
        return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans)

    parts = []
    if regenerated:
        parts.append(f"Changed: scene{'s' if len(regenerated) > 1 else ''} {ranges(regenerated)}.")
    if retimed:
        parts.append(f"Re-timed to new narration: scene{'s' if len(retimed) > 1 else ''} {ranges(retimed)}.")
    parts.append(f"Reused unchanged: {len(reused_video)} scene{'s' if len(reused_video) != 1 else ''}.")
    return " ".join(parts)


def write_revision_report(path: Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
