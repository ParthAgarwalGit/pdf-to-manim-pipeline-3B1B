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
    narration_fingerprint,
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
    return BaseRun(
        run_dir=run_dir,
        storyboard=storyboard,
        script=script,
        run_config=run_config,
        records=records,
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
            narration_fp=narration_fingerprint(scene.narration, voice),
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
            narration_fp=narration_fingerprint(scene.narration, voice),
            visual_fp=visual_fingerprint(scene, render_quality, tts_first),
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


def write_revision_report(path: Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
