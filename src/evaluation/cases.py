"""Build and validate frozen evaluation cases from completed V2 runs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline_v2.models import VerificationFinding

from .models import EvaluationCase, EvaluationManifest


def build_benchmark(
    run_directory: Path,
    destination: Path,
    *,
    scene_ids: list[str] | None = None,
    category: str = "uncategorized",
) -> EvaluationManifest:
    run_directory = run_directory.resolve()
    destination = destination.resolve()
    run_payload = _read_json(run_directory / "run.json")
    storyboard = _read_json(run_directory / "storyboard.json")
    source = _read_json(run_directory / "source" / "document.json")
    lesson_by_id = {
        item["scene_id"]: item for item in storyboard.get("lesson", {}).get("scenes", [])
    }
    lesson_order = list(storyboard.get("lesson", {}).get("scenes", []))
    narration_by_id = {
        item["scene_id"]: item for item in storyboard.get("narration", {}).get("scenes", [])
    }
    result_by_id = {
        item.get("scene_id"): item
        for item in (run_payload.get("result") or {}).get("scene_results", [])
        if item.get("scene_id")
    }
    available = [path.name for path in (run_directory / "scenes").iterdir() if path.is_dir()]
    selected = scene_ids or [scene_id for scene_id in lesson_by_id if scene_id in available]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown scene IDs for this run: {unknown}")

    cases_dir = destination / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_files: list[str] = []
    for scene_id in selected:
        scene_dir = run_directory / "scenes" / scene_id
        scene_payload = _read_json(scene_dir / "scene.json")
        narration = scene_payload.get("narration") or narration_by_id.get(scene_id)
        lesson = scene_payload.get("lesson") or lesson_by_id.get(scene_id)
        if not narration or not lesson:
            raise ValueError(f"Scene {scene_id} lacks frozen narration or lesson data")
        contract = _read_json(scene_dir / "visual_contract.json")
        relevant_ids = set(lesson.get("source_fact_ids") or [])
        facts = [item for item in source.get("facts", []) if item.get("fact_id") in relevant_ids]

        case_id = _case_id(run_directory, scene_id)
        case_dir = cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        audio_path = _copy_optional(
            _resolve_existing(narration.get("audio_path")), case_dir / "audio",
        )
        narration = dict(narration)
        narration["audio_path"] = audio_path
        run_result = result_by_id.get(scene_id, {})
        selected_code = _resolve_existing(run_result.get("original_code_path"))
        initial_code = _copy_optional(
            selected_code or _preferred_code(scene_dir), case_dir / "initial",
        )
        preview = _resolve_existing(run_result.get("preview_path"))
        selected_diagnostics = preview.parent / "diagnostics.json" if preview else None
        diagnostics = (
            selected_diagnostics
            if selected_diagnostics and selected_diagnostics.is_file()
            else _preferred_diagnostics(scene_dir)
        )
        findings = _load_findings(diagnostics)
        frames = []
        if diagnostics:
            for frame in sorted((diagnostics.parent / "frames").glob("*.png"))[:12]:
                copied = _copy_optional(frame, case_dir / "initial-frames")
                if copied:
                    frames.append(copied)

        case = EvaluationCase(
            case_id=case_id,
            category=category,
            source_run=str(run_directory),
            source_scene_id=scene_id,
            source_facts=facts,
            narration=narration,
            lesson=lesson,
            visual_contract=contract,
            visual_history=_visual_history(lesson_order, scene_id),
            expected_facts=[item["text"] for item in facts],
            required_visuals=[
                lesson.get("visual_objective", ""), lesson.get("new_visual_fact", ""),
            ],
            audio_path=audio_path,
            initial_code_path=initial_code,
            initial_findings=findings,
            initial_frame_paths=frames,
        )
        case.content_hash = case_hash(case)
        case_path = case_dir / "case.json"
        case_path.write_text(
            json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        case_files.append(case_path.relative_to(destination).as_posix())

    manifest_path = destination / "manifest.json"
    existing = _read_json(manifest_path) if manifest_path.exists() else {}
    merged_files = sorted(set(existing.get("case_files", [])) | set(case_files))
    source_runs = sorted(set(existing.get("source_runs", [])) | {str(run_directory)})
    manifest = EvaluationManifest(
        benchmark_id=existing.get("benchmark_id") or destination.name,
        created_at=existing.get("created_at") or datetime.now(timezone.utc).isoformat(),
        source_runs=source_runs,
        case_files=merged_files,
    )
    manifest.benchmark_hash = manifest_hash(destination, manifest)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def load_benchmark(path: Path, *, verify_hashes: bool = True) -> tuple[EvaluationManifest, list[EvaluationCase]]:
    root = path.resolve()
    manifest = EvaluationManifest.model_validate(_read_json(root / "manifest.json"))
    cases = []
    for relative in manifest.case_files:
        case = EvaluationCase.model_validate(_read_json(root / relative))
        if verify_hashes and case.content_hash != case_hash(case):
            raise ValueError(f"Case hash mismatch: {case.case_id}")
        cases.append(case)
    if verify_hashes and manifest.benchmark_hash != manifest_hash(root, manifest):
        raise ValueError("Benchmark hash mismatch; a case or manifest entry changed")
    return manifest, cases


def case_hash(case: EvaluationCase) -> str:
    payload = case.model_dump(mode="json", exclude={"content_hash", "source_run"})
    for key in ("audio_path", "initial_code_path", "initial_frame_paths"):
        payload[key] = _asset_hashes(payload.get(key))
    payload["narration"]["audio_path"] = payload["audio_path"]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_hash(root: Path, manifest: EvaluationManifest) -> str:
    entries = []
    for relative in sorted(manifest.case_files):
        payload = _read_json(root / relative)
        entries.append((relative, payload.get("content_hash", "")))
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _asset_hashes(value: Any) -> Any:
    if isinstance(value, list):
        return [_asset_hashes(item) for item in value]
    if not value:
        return value
    path = Path(str(value))
    if not path.exists() or not path.is_file():
        return {"missing": path.name}
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _preferred_code(scene_dir: Path) -> Path | None:
    choices = [scene_dir / "code" / "original.py", scene_dir / "code" / "model_raw.py"]
    return next((path for path in choices if path.is_file()), None)


def _preferred_diagnostics(scene_dir: Path) -> Path | None:
    choices = [scene_dir / "attempt-00" / "diagnostics.json"]
    choices.extend(sorted(scene_dir.glob("retries/retry-*/attempt-00/diagnostics.json"), reverse=True))
    return next((path for path in choices if path.is_file()), None)


def _load_findings(path: Path | None) -> list[VerificationFinding]:
    if not path:
        return []
    payload = _read_json(path)
    raw = payload.get("findings", payload if isinstance(payload, list) else [])
    return [VerificationFinding.model_validate(item) for item in raw]


def _resolve_existing(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def _copy_optional(source: Path | None, destination: Path) -> str | None:
    if not source or not source.is_file():
        return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copy2(source, target)
    return str(target.resolve())


def _case_id(run_directory: Path, scene_id: str) -> str:
    document = _safe_name(run_directory.parent.name)
    return f"{document}__{_safe_name(scene_id)}"


def _visual_history(lesson_scenes: list[dict[str, Any]], scene_id: str) -> list[dict[str, Any]]:
    position = next(
        (index for index, scene in enumerate(lesson_scenes) if scene.get("scene_id") == scene_id),
        0,
    )
    return [
        {
            "scene_id": scene.get("scene_id"),
            "title": scene.get("title"),
            "learning_goal": scene.get("learning_goal"),
            "visual_objective": scene.get("visual_objective"),
            "visual_fact_already_shown": scene.get("new_visual_fact"),
            "constructions_not_to_replay": scene.get("do_not_repeat") or [],
        }
        for scene in lesson_scenes[max(0, position - 3):position]
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "scene"


def _read_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required artifact not found: {path}")
