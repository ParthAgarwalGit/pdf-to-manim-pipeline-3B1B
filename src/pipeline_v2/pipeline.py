"""Bounded, resumable narration-first PDF-to-video orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

from pydantic import BaseModel

from .diagnostics import frame_fingerprint, frame_sequence_similarity, issue_score
from .delivery import SceneClip, run_delivery_gate
from .directives import (
    ConceptPlan, apply_scope, check_lesson_directives, enforce_narration_directives,
    length_check, notation_findings, notation_rules,
)
from src.schemas.directives import ProfessorDirectives
from src.utils.render_sandbox import RenderSandboxConfig
from .generation import CODE_PROMPT_VERSION, CodeGenerationService
from .instrumentation import deinstrument_scene_code
from .media import compose_scene, concatenate, inspect_av, source_page_fallback
from .models import (
    LessonPlan,
    NarratedScene,
    NarrationPlan,
    PipelineV2Config,
    RunResult,
    RunStatus,
    SceneResult,
    SourceDocument,
    TaskStatus,
    TimelineSegment,
    VerificationFinding,
    VisualContractBatch,
)
from .observability import RunArtifacts
from .planning import (
    NARRATION_PROMPT_VERSION,
    PLANNING_MODEL,
    PLANNING_PROVIDER,
    PROMPT_VERSION,
    PlanningService,
    apply_audio_timing,
    extract_source,
    validate_grounding,
    validate_visual_contract,
)
from .providers import MissingProviderKey, ProviderRouter, STRUCTURED_OUTPUT_VERSION
from .rendering import VERIFIER_VERSION, RenderOutcome, TrackedManimRenderer
from .report import write_costs, write_html_report, write_storyboard
from .revision import (
    FROZEN_PLANNING_FILE,
    RevisionError,
    RevisionOp,
    RevisionPlan,
    SceneAction,
    build_revision,
    frozen_planning,
    load_frozen_planning,
    load_run_state,
    raw_audio_path,
    read_revision_state,
    revision_directory,
    write_revision_state,
)
from .state import BudgetExceeded, RunStore
from .tts import (
    RELAXED_SPEECH_RATE,
    SCENE_END_PAUSE_SECONDS,
    SPEECH_RATE,
    NarrationAudioService,
    TTS_TIMING_VERSION,
    ensure_trailing_buffer,
    retime,
)
from src.utils.fingerprints import lexicon_for


T = TypeVar("T")


class PipelineV2:
    def __init__(self, config: Optional[PipelineV2Config] = None):
        self.config = config or PipelineV2Config()
        self.store: Optional[RunStore] = None
        self.artifacts: Optional[RunArtifacts] = None
        self.renderer = TrackedManimRenderer(RenderSandboxConfig(
            mode=self.config.render_sandbox, image=self.config.render_sandbox_image,
        ))
        self.directives = (
            ProfessorDirectives.model_validate(self.config.directives)
            if self.config.directives is not None else None
        )
        self._visual_limit = asyncio.Semaphore(self.config.visual_workers)
        self._tts_limit = asyncio.Semaphore(self.config.tts_workers)
        self._code_limit = asyncio.Semaphore(self.config.code_workers)
        self._preview_limit = asyncio.Semaphore(self.config.preview_workers)
        self._repair_limit = asyncio.Semaphore(self.config.repair_workers)
        self._render_limit = asyncio.Semaphore(self.config.render_workers)
        self._retry_scene_ids: set[str] | None = None
        self._retry_attempts: dict[str, int] = {}
        self._previous_scene_results: dict[str, SceneResult] = {}
        self._previous_result: Optional[dict[str, Any]] = None
        # Revision state: per-scene actions, the pre-revision timed narration
        # they start from, and run-wide pronunciation/pacing overrides.
        self._revision_actions: dict[str, SceneAction] = {}
        self._revision_previous: dict[str, NarratedScene] = {}
        self._lexicon: dict[str, str] = {}
        self._speech_rates: dict[str, float] = {}

    def run(self, input_pdf: str | Path, *, run_dir: Optional[str | Path] = None) -> RunResult:
        input_path = Path(input_pdf).resolve()
        if not input_path.is_file() or input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input PDF does not exist: {input_path}")
        try:
            return asyncio.run(self._run(input_path, Path(run_dir).resolve() if run_dir else None))
        except KeyboardInterrupt:
            self.renderer.terminate_all()
            if self.store:
                self.store.interrupt_running()
                self.store.set_meta("status", RunStatus.INTERRUPTED.value)
            if self.artifacts:
                self.artifacts.event("run_interrupted", message="Interrupted by user; run is resumable.")
            run_path = str(self.artifacts.run_dir) if self.artifacts else str(run_dir or "")
            return RunResult(
                run_id=self.artifacts.run_id if self.artifacts else "interrupted",
                run_directory=run_path, status=RunStatus.INTERRUPTED,
                estimated_cost_usd=self.store.total_cost() if self.store else 0.0,
                message="Interrupted safely. Resume this run directory to continue.",
            )
        finally:
            if self.store:
                self.store.close()

    @classmethod
    def resume(cls, run_dir: str | Path, *, max_cost_usd: Optional[float] = None) -> RunResult:
        path = Path(run_dir).resolve()
        payload = json.loads((path / "run.json").read_text(encoding="utf-8"))
        config_data = payload["config"]
        if max_cost_usd is not None:
            config_data["max_cost_usd"] = max_cost_usd
        config = PipelineV2Config.model_validate(config_data)
        if read_revision_state(path).get("pending"):
            return cls._run_revision(path, config, Path(payload["input_pdf"]).resolve())
        return cls(config).run(payload["input_pdf"], run_dir=path)

    @classmethod
    def retry_scenes(
        cls,
        run_dir: str | Path,
        scene_ids: list[str],
        *,
        max_cost_usd: Optional[float] = None,
    ) -> RunResult:
        path = Path(run_dir).resolve()
        run_json = path / "run.json"
        storyboard_json = path / "storyboard.json"
        database = path / "run.db"
        if not run_json.is_file() or not storyboard_json.is_file() or not database.is_file():
            raise ValueError(f"Not a complete V2 run directory: {path}")
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        input_path = Path(payload.get("input_pdf", "")).resolve()
        if not input_path.is_file():
            raise ValueError(f"The run's input PDF is no longer available: {input_path}")
        storyboard = json.loads(storyboard_json.read_text(encoding="utf-8"))
        known_ids = {
            str(scene["scene_id"])
            for scene in storyboard.get("lesson", {}).get("scenes", [])
            if scene.get("scene_id")
        }
        requested = list(dict.fromkeys(scene_ids))
        unknown = sorted(set(requested) - known_ids)
        if not requested:
            raise ValueError("At least one scene ID is required")
        if unknown:
            raise ValueError(
                f"Unknown scene IDs: {', '.join(unknown)}. Available: {', '.join(sorted(known_ids))}"
            )
        config_data = payload["config"]
        if max_cost_usd is not None:
            config_data["max_cost_usd"] = max_cost_usd
        config = PipelineV2Config.model_validate(config_data)
        previous = {
            item["scene_id"]: SceneResult.model_validate(item)
            for item in (payload.get("result") or {}).get("scene_results", [])
        }
        missing_results = sorted(known_ids - set(previous))
        if missing_results:
            raise ValueError(
                "Run does not contain reusable results for: " + ", ".join(missing_results)
            )
        store = RunStore(database, config.max_cost_usd)
        try:
            retry_attempts = store.prepare_scene_retry(requested)
        finally:
            store.close()
        pipeline = cls(config)
        pipeline._retry_scene_ids = set(requested)
        pipeline._retry_attempts = retry_attempts
        pipeline._previous_scene_results = previous
        pipeline._previous_result = payload.get("result")
        return pipeline.run(input_path, run_dir=path)

    @classmethod
    def revise(
        cls,
        run_dir: str | Path,
        ops: list[RevisionOp],
        *,
        max_cost_usd: Optional[float] = None,
        dry_run: bool = False,
    ) -> tuple[RevisionPlan, Optional[RunResult]]:
        """Apply professor edits to a finished run, rebuilding only affected scenes.

        Returns the plan (changes and cost estimate) and, unless ``dry_run`` or
        nothing changed, the result of the rebuilt run.
        """

        path = Path(run_dir).resolve()
        state = load_run_state(path)
        plan, revised = build_revision(state, ops)
        if dry_run or not plan.actions:
            return plan, None
        payload = json.loads((path / "run.json").read_text(encoding="utf-8"))
        input_path = Path(payload.get("input_pdf", "")).resolve()
        if not input_path.is_file():
            raise RevisionError(f"The run's input PDF is no longer available: {input_path}")
        config_data = payload["config"]
        if max_cost_usd is not None:
            config_data["max_cost_usd"] = max_cost_usd
        config = PipelineV2Config.model_validate(config_data)

        revision_dir = revision_directory(path, plan.revision_number)
        revision_dir.mkdir(parents=True, exist_ok=True)
        before = {
            "storyboard": json.loads((path / "storyboard.json").read_text(encoding="utf-8")),
            "result": payload.get("result"),
        }
        _write_json(revision_dir / "before.json", before)
        _write_json(revision_dir / "plan.json", plan.model_dump(mode="json"))

        affected = plan.affected_scene_ids
        audio_ids = [item.scene_id for item in plan.actions if item.audio != "reuse"]
        store = RunStore(path / "run.db", config.max_cost_usd)
        try:
            frozen = load_frozen_planning(path)
            if frozen is not None:
                base_narration: Optional[NarrationPlan] = frozen[1]
            else:
                try:
                    base_narration = NarrationPlan.model_validate(
                        store.successful_task_output("narration")
                    )
                except ValueError:
                    base_narration = None
            retry_attempts = store.prepare_scene_revision(affected, audio_ids)
        finally:
            store.close()
        _write_json(path / FROZEN_PLANNING_FILE, frozen_planning(
            state.lesson, revised.narration, set(audio_ids), base_narration,
        ))
        _write_json(path / "storyboard.json", {
            "lesson": state.lesson.model_dump(mode="json"),
            "narration": revised.narration.model_dump(mode="json"),
        })
        write_revision_state(path, {
            "revision": plan.revision_number,
            "lexicon": revised.lexicon,
            "speech_rates": revised.speech_rates,
            "pending": {
                "revision": plan.revision_number,
                "actions": [item.model_dump(mode="json") for item in plan.actions],
                "retry_attempts": retry_attempts,
            },
        })
        return plan, cls._run_revision(path, config, input_path)

    @classmethod
    def _run_revision(cls, path: Path, config: PipelineV2Config, input_path: Path) -> RunResult:
        """Run (or re-enter after an interruption) the pending revision of a run."""

        state = read_revision_state(path)
        pending = state["pending"]
        number = int(pending["revision"])
        revision_dir = revision_directory(path, number)
        before = json.loads((revision_dir / "before.json").read_text(encoding="utf-8"))
        actions = [SceneAction.model_validate(item) for item in pending["actions"]]
        pipeline = cls(config)
        pipeline._retry_scene_ids = {item.scene_id for item in actions}
        pipeline._retry_attempts = {
            key: int(value) for key, value in pending["retry_attempts"].items()
        }
        pipeline._previous_result = before.get("result")
        pipeline._previous_scene_results = {
            item["scene_id"]: SceneResult.model_validate(item)
            for item in (before.get("result") or {}).get("scene_results", [])
        }
        pipeline._revision_actions = {item.scene_id: item for item in actions}
        pipeline._revision_previous = {
            scene.scene_id: scene
            for scene in NarrationPlan.model_validate(before["storyboard"]["narration"]).scenes
        }
        result = pipeline.run(input_path, run_dir=path)
        if result.status in (RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS, RunStatus.FAILED):
            # Interrupted and budget-paused revisions stay pending so a resume
            # continues them instead of reverting to the cached planner output.
            state["pending"] = None
            write_revision_state(path, state)
        _write_json(revision_dir / "result.json", {
            "revision": number,
            "status": result.status.value,
            "message": result.message,
            "rebuilt_scene_ids": sorted(pipeline._retry_scene_ids),
            "total_cost_usd": result.estimated_cost_usd,
            "final_video_path": result.final_video_path,
        })
        return result

    async def _run(self, input_path: Path, resume_dir: Optional[Path]) -> RunResult:
        run_dir, run_id = self._prepare_run(input_path, resume_dir)
        self.artifacts = RunArtifacts(run_dir, run_id)
        self.store = RunStore(run_dir / "run.db", self.config.max_cost_usd)
        self.store.set_meta("status", RunStatus.RUNNING.value)
        self.store.set_meta("input_pdf", str(input_path))
        self.artifacts.event(
            "run_started", message=f"Starting V2 conversion for {input_path.name}",
            max_cost_usd=self.config.max_cost_usd, quality=self.config.quality,
            policy=self.config.policy, verifier_mode=self.config.verifier_mode,
            min_scenes=self.config.min_scenes, max_scenes=self.config.max_scenes,
            pacing=self.config.pacing,
        )
        self._write_run_json(input_path, RunStatus.RUNNING)
        router = ProviderRouter(
            self.store, self.artifacts,
            max_api_workers=self.config.global_api_workers,
            strict_keys=self.config.strict_keys,
        )
        planner = PlanningService(
            router,
            self.config.max_scenes,
            self.config.min_scenes,
            self.config.pacing,
            audience=self.config.audience,
            directives=self.directives,
            concept_plan=(ConceptPlan.model_validate(self.config.concept_plan)
                          if self.config.concept_plan is not None else None),
            target_minutes=self.config.target_minutes,
        )
        controls = {
            key: value for key, value in {
                "directives": self.config.directives,
                "concept_plan": self.config.concept_plan,
                "target_minutes": self.config.target_minutes,
            }.items() if value is not None
        }
        compliance = []
        generator = CodeGenerationService(router)
        audio = NarrationAudioService(
            self.store, self.artifacts, voice=self.config.voice,
            high_quality=self.config.quality == "high", strict_keys=self.config.strict_keys,
            api_limit=router.api_limit,
            speech_rate=(
                RELAXED_SPEECH_RATE if self.config.pacing == "relaxed" else SPEECH_RATE
            ),
        )
        result = RunResult(run_id=run_id, run_directory=str(run_dir), status=RunStatus.RUNNING)
        revision_state = read_revision_state(run_dir)
        self._lexicon = dict(revision_state.get("lexicon") or {})
        self._speech_rates = {
            key: float(value)
            for key, value in (revision_state.get("speech_rates") or {}).items()
        }
        frozen_plan = load_frozen_planning(run_dir)

        try:
            if self._retry_scene_ids is not None:
                source = SourceDocument.model_validate_json(
                    (run_dir / "source" / "document.json").read_text(encoding="utf-8")
                )
                storyboard = json.loads(
                    (run_dir / "storyboard.json").read_text(encoding="utf-8")
                )
                lesson = LessonPlan.model_validate(storyboard["lesson"])
                narration = NarrationPlan.model_validate(storyboard["narration"])
                self.artifacts.event(
                    "scene_retry_started",
                    stage="retry",
                    message="Reusing grounded planning and narration for selected scene retries.",
                    scene_ids=sorted(self._retry_scene_ids),
                    retry_attempts=self._retry_attempts,
                )
            elif frozen_plan is not None:
                # A revised run restarts from its edited storyboard; rerunning
                # the cached planner output would silently undo the edits.
                source = SourceDocument.model_validate_json(
                    (run_dir / "source" / "document.json").read_text(encoding="utf-8")
                )
                lesson, narration = frozen_plan
                self.artifacts.event(
                    "revised_planning_reused", stage="narration",
                    message="Reusing the revised storyboard instead of the original plan.",
                )
            else:
                input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
                source_data = await self._task(
                    "source", "source_extraction", None,
                    {"sha256": input_hash, "stage_version": "source-v2-1"},
                    lambda: asyncio.to_thread(extract_source, input_path),
                )
                source = SourceDocument.model_validate(source_data)
                enriched_data = await self._task(
                    "ocr-enrichment", "ocr", None,
                    {"source_sha256": source.sha256,
                     "pages": source.low_confidence_pages, "prompt_version": PROMPT_VERSION},
                    lambda: planner.enrich_low_confidence_pages(
                        source, input_path, run_dir / "source" / "ocr"
                    ),
                    dependencies=["source"],
                )
                source = SourceDocument.model_validate(enriched_data)
                source, scope_checks = apply_scope(source, self.directives)
                compliance.extend(scope_checks)
                self.artifacts.write_json("source/document.json", source.model_dump(mode="json"))

                lesson_data = await self._task(
                    "lesson", "lesson", None,
                    {"source": source.model_dump(mode="json"), "prompt_version": PROMPT_VERSION,
                     "min_scenes": self.config.min_scenes,
                     "max_scenes": self.config.max_scenes,
                     "pacing": self.config.pacing,
                     **controls,
                     "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                     "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL},
                    lambda: planner.lesson(source),
                    dependencies=["ocr-enrichment"],
                )
                lesson = LessonPlan.model_validate(lesson_data)
                lesson = lesson.model_copy(update={
                    "scenes": [
                        scene.model_copy(update={"delivery_mode": "manim"})
                        for scene in lesson.scenes
                    ]
                })
                narration_data = await self._task(
                    "narration", "narration", None,
                    {"lesson": lesson.model_dump(mode="json"),
                     "prompt_version": NARRATION_PROMPT_VERSION,
                     "pacing": self.config.pacing,
                     **controls,
                     "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                     "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL},
                    lambda: planner.narration(lesson, source),
                    dependencies=["lesson"],
                )
                narration = NarrationPlan.model_validate(narration_data)
                grounding_errors = validate_grounding(lesson, narration, source)
                if grounding_errors:
                    self.artifacts.event(
                        "grounding_failed", stage="narration", message="; ".join(grounding_errors),
                    )
                    if self.config.policy == "strict":
                        raise RuntimeError(
                            "Grounding validation failed: " + "; ".join(grounding_errors)
                        )
                    lesson = planner._fallback_lesson(source)
                    narration = planner._fallback_narration(lesson, source)
                    self.artifacts.event(
                        "grounding_fallback", stage="narration",
                        message=(
                            "Replaced unsupported lesson output with a deterministic "
                            "source-grounded plan."
                        ),
                    )
            compliance.extend(check_lesson_directives(
                lesson, source, self.directives, planner.omitted_concepts,
            ))
            narration, narration_checks = enforce_narration_directives(
                narration, self.directives, planner.omitted_concepts,
            )
            compliance.extend(narration_checks)
            if controls:
                self.artifacts.write_json("professor_controls.json", controls)
                self.artifacts.write_json("compliance.json", [
                    item.model_dump(mode="json") for item in compliance
                ])
                if planner.length_report:
                    self.artifacts.write_json("length_budget.json", planner.length_report)
                if self.config.policy == "strict" and any(not item.passed for item in compliance):
                    raise RuntimeError("Professor directives failed compliance; inspect compliance.json")
            write_storyboard(self.artifacts, lesson, narration)
            result.total_scenes = len(lesson.scenes)

            if self.config.dry_run:
                result.status = RunStatus.SUCCESS
                result.message = "Dry run completed through grounded narration planning."
                return self._finalize(input_path, result)

            scene_dirs = {
                scene.scene_id: run_dir / "scenes" / scene.scene_id for scene in narration.scenes
            }
            for path in scene_dirs.values():
                path.mkdir(parents=True, exist_ok=True)

            tts_tasks: dict[str, asyncio.Task] = {}
            for scene in narration.scenes:
                persisted_audio = (
                    self.store.successful_task_output(f"tts-{scene.scene_id}")
                    if self._retry_scene_ids is not None else None
                )
                if persisted_audio is not None:
                    persisted_scene = NarratedScene.model_validate(persisted_audio)
                    if scene.scene_id in (self._retry_scene_ids or set()):
                        retry_number = self._retry_attempts[scene.scene_id]
                        padded_path = (
                            scene_dirs[scene.scene_id] / "retries"
                            / f"retry-{retry_number:02d}" / "audio"
                            / f"{scene.scene_id}_buffered.mp3"
                        )
                        original_tail = persisted_scene.audio_tail_seconds
                        persisted_scene = await asyncio.to_thread(
                            ensure_trailing_buffer,
                            persisted_scene,
                            padded_path,
                        )
                        if persisted_scene.audio_tail_seconds > original_tail:
                            self.artifacts.event(
                                "retry_audio_buffer_extended",
                                stage="tts",
                                scene_id=scene.scene_id,
                                task_id=f"tts-{scene.scene_id}",
                                message=(
                                    "Extended legacy narration locally to the current "
                                    "two-second scene buffer."
                                ),
                                previous_tail_seconds=original_tail,
                                final_tail_seconds=persisted_scene.audio_tail_seconds,
                            )
                    tts_tasks[scene.scene_id] = asyncio.create_task(
                        _return(persisted_scene.model_dump(mode="json")),
                        name=f"tts-reused-{scene.scene_id}",
                    )
                    self.artifacts.event(
                        "retry_audio_reused", stage="tts", scene_id=scene.scene_id,
                        task_id=f"tts-{scene.scene_id}",
                        message="Reused the run's existing narration audio without a paid TTS call.",
                    )
                elif self.config.enable_tts:
                    retry_number = self._retry_attempts.get(scene.scene_id)
                    tts_tasks[scene.scene_id] = asyncio.create_task(
                        self._tts_task(
                            audio, scene,
                            scene_dirs[scene.scene_id] if not retry_number
                            else scene_dirs[scene.scene_id] / "retries" / f"retry-{retry_number:02d}",
                        ),
                        name=f"tts-{scene.scene_id}",
                    )
                else:
                    duration = max(
                        2.0,
                        len(scene.narration.split()) * 60.0 / scene.target_words_per_minute
                        + sum(beat.pause_after_seconds for beat in scene.beats)
                        + SCENE_END_PAUSE_SECONDS,
                    )
                    tts_tasks[scene.scene_id] = asyncio.create_task(
                        _return(apply_audio_timing(scene, duration).model_copy(update={
                            "audio_tail_seconds": SCENE_END_PAUSE_SECONDS,
                        })),
                        name=f"tts-local-{scene.scene_id}",
                    )

            contract_tasks: dict[int, asyncio.Task] = {}
            scene_batch: dict[str, int] = {}
            for batch_index, offset in enumerate(range(0, len(narration.scenes), 3)):
                batch = narration.scenes[offset:offset + 3]
                for scene in batch:
                    scene_batch[scene.scene_id] = batch_index
                contract_tasks[batch_index] = asyncio.create_task(
                    self._contract_task(planner, batch, lesson, source, batch_index),
                    name=f"visual-{batch_index:03d}",
                )

            selected_ids = self._retry_scene_ids or {
                scene.scene_id for scene in narration.scenes
            }
            selected_scenes = [
                scene for scene in narration.scenes if scene.scene_id in selected_ids
            ]
            scene_jobs = [
                asyncio.create_task(
                    self._process_scene(
                        input_path=input_path,
                        source=source,
                        lesson=lesson,
                        original_scene=scene,
                        timed_scene_task=tts_tasks[scene.scene_id],
                        contract_task=contract_tasks[scene_batch[scene.scene_id]],
                        scene_dir=(
                            scene_dirs[scene.scene_id]
                            if not self._retry_attempts.get(scene.scene_id)
                            else scene_dirs[scene.scene_id] / "retries"
                            / f"retry-{self._retry_attempts[scene.scene_id]:02d}"
                        ),
                        generator=generator,
                        retry_attempt=self._retry_attempts.get(scene.scene_id, 0),
                    ),
                    name=f"scene-{scene.scene_id}",
                )
                for scene in selected_scenes
            ]
            selected_results = {
                item.scene_id: item for item in await asyncio.gather(*scene_jobs)
            }
            result.scene_results = [
                selected_results.get(scene.scene_id)
                or self._previous_scene_results[scene.scene_id]
                for scene in narration.scenes
            ]
            _audit_repeated_visuals(result.scene_results, self.artifacts)
            timed_scenes = [NarratedScene.model_validate(await tts_tasks[scene.scene_id])
                            for scene in narration.scenes]
            if planner.target_minutes:
                compliance.append(length_check(
                    sum(scene.audio_duration_seconds or 0.0 for scene in timed_scenes),
                    planner.target_minutes,
                ))
                self.artifacts.write_json("compliance.json", [
                    item.model_dump(mode="json") for item in compliance
                ])
            write_storyboard(self.artifacts, lesson, NarrationPlan(scenes=timed_scenes))

            successful = [scene for scene in result.scene_results if scene.success]
            result.successful_scenes = len(successful)
            result.failed_scenes = result.total_scenes - result.successful_scenes
            timeline = []
            cursor = 0.0
            ordered_videos = []
            by_id = {scene.scene_id: scene for scene in result.scene_results}
            timed_by_id = {scene.scene_id: scene for scene in timed_scenes}
            for lesson_scene in lesson.scenes:
                scene_result = by_id.get(lesson_scene.scene_id)
                timed = timed_by_id.get(lesson_scene.scene_id)
                if not scene_result or not scene_result.success or not scene_result.final_video_path:
                    continue
                path = Path(scene_result.final_video_path)
                ordered_videos.append(path)
                duration = timed.audio_duration_seconds if timed and timed.audio_duration_seconds else 1.0
                timeline.append(TimelineSegment(
                    segment_id=f"segment-{lesson_scene.scene_id}", scene_id=lesson_scene.scene_id,
                    start_seconds=cursor, duration_seconds=duration,
                    video_path=str(path), audio_path=timed.audio_path if timed else None,
                ))
                cursor += duration
            self.artifacts.write_json("timeline.json", [item.model_dump(mode="json") for item in timeline])

            strict_failure = self.config.policy == "strict" and result.failed_scenes > 0
            if self.config.enable_render and ordered_videos and not strict_failure:
                final_path = await self._task(
                    "final-compose", "composition", None,
                    {"videos": [_file_signature(path) for path in ordered_videos]},
                    lambda: asyncio.to_thread(concatenate, ordered_videos, run_dir / "final.mp4"),
                    dependencies=[
                        f"compose-{scene.scene_id}" if scene.audio_path else f"render-{scene.scene_id}"
                        for scene in successful
                    ],
                )
                result.final_video_path = str(final_path)
                av = inspect_av(Path(final_path))
                expected_duration = cursor
                duration_delta = abs(float(av.get("duration_seconds") or 0.0) - expected_duration)
                duration_tolerance = max(2.0, expected_duration * 0.05)
                av.update({
                    "expected_duration_seconds": round(expected_duration, 3),
                    "duration_delta_seconds": round(duration_delta, 3),
                    "duration_tolerance_seconds": round(duration_tolerance, 3),
                    "duration_matches": duration_delta <= duration_tolerance,
                })
                av["valid"] = bool(av.get("valid")) and bool(av["duration_matches"])
                self.artifacts.write_json("final_review.json", av)
                if self.config.enable_tts and not av.get("valid"):
                    if not av.get("video") or not av.get("audio"):
                        result.message = "Final media is missing a required audio or video stream."
                    else:
                        result.message = (
                            "Final media duration differs from the narration timeline by "
                            f"{duration_delta:.2f}s."
                        )
                    result.status = RunStatus.FAILED if self.config.policy == "strict" else RunStatus.PARTIAL_SUCCESS
                else:
                    result.status = (
                        RunStatus.SUCCESS
                        if result.failed_scenes == 0
                        and not any(scene.fallback_used for scene in result.scene_results)
                        else RunStatus.PARTIAL_SUCCESS
                    )
                    fallback_count = sum(scene.fallback_used for scene in result.scene_results)
                    if fallback_count:
                        result.message = (
                            f"{fallback_count} of {result.total_scenes} scenes used static "
                            "source-page fallback instead of accepted Manim animation."
                        )
            elif strict_failure:
                result.status = RunStatus.FAILED
                result.message = "Strict policy withheld the final video because required scenes failed."
            elif not self.config.enable_render:
                result.status = RunStatus.SUCCESS if result.failed_scenes == 0 else RunStatus.PARTIAL_SUCCESS
                result.message = "Code generation completed with rendering disabled."
            else:
                result.status = RunStatus.FAILED
                result.message = "No deliverable scene videos were produced."
            if self.config.enable_delivery_checks and result.final_video_path and self.config.enable_tts:
                clips = [SceneClip(
                    scene_id=item.scene_id, composed_path=Path(item.final_video_path),
                    render_path=Path(item.render_video_path) if item.render_video_path else None,
                    audio_path=Path(item.audio_path) if item.audio_path else None,
                    narration_seconds=timed_by_id[item.scene_id].audio_duration_seconds,
                    tail_seconds=timed_by_id[item.scene_id].audio_tail_seconds,
                    silent_narration=timed_by_id[item.scene_id].audio_is_placeholder,
                ) for item in successful if item.final_video_path]
                report = await asyncio.to_thread(
                    run_delivery_gate, clips, Path(result.final_video_path),
                    max_repair_rounds=self.config.max_delivery_repair_rounds,
                    reconcat=lambda paths: concatenate(paths, run_dir / "final.mp4"),
                )
                self.artifacts.write_json(
                    "delivery_report.json", report.model_dump(mode="json"),
                )
                result.delivery_status = report.status
                result.delivery_report_path = str(run_dir / "delivery_report.json")
                result.delivery_summary = report.summary.splitlines()
                self.artifacts.event("delivery_checked", stage="delivery", message=report.summary)
                if report.status != "ready":
                    result.status = (RunStatus.FAILED if self.config.policy == "strict"
                                     else RunStatus.PARTIAL_SUCCESS)
                    if self.config.policy == "strict":
                        result.final_video_path = None
                    result.message = "Final media needs review; inspect delivery_report.json."
            if self.config.policy == "strict" and any(not item.passed for item in compliance):
                result.status = RunStatus.FAILED
                result.final_video_path = None
                result.message = "Strict policy withheld final delivery because professor controls failed."
            if self._retry_scene_ids is not None:
                scene_list = ", ".join(sorted(self._retry_scene_ids))
                if not result.message:
                    result.message = f"Retried {scene_list} and rebuilt the complete final video."
                self.artifacts.event(
                    "scene_retry_finished",
                    stage="retry",
                    message=result.message,
                    scene_ids=sorted(self._retry_scene_ids),
                    status=result.status.value,
                    final_video_path=result.final_video_path,
                )
        except BudgetExceeded as exc:
            result.status = RunStatus.PAUSED_BUDGET
            result.message = str(exc)
            self.artifacts.event("budget_paused", message=str(exc))
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            self.store.interrupt_running()
        except asyncio.CancelledError:
            self.renderer.terminate_all()
            result.status = RunStatus.INTERRUPTED
            result.message = "Run was cancelled and can be resumed."
            self.store.interrupt_running()
            self.artifacts.event("run_interrupted", message=result.message)
        except Exception as exc:
            result.status = RunStatus.FAILED
            result.message = str(exc)
            self.artifacts.event("run_failed", message=str(exc), error_type=type(exc).__name__)
        return self._finalize(input_path, result)

    async def _tts_task(
        self,
        service: NarrationAudioService,
        scene: NarratedScene,
        scene_dir: Path,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "narration": scene.model_dump(mode="json"), "voice": self.config.voice,
            "stage_version": TTS_TIMING_VERSION, "quality": self.config.quality,
        }
        # Pronunciation and pacing overrides enter the hash only for scenes
        # they touch, so every other scene keeps its cached paid audio.
        lexicon = lexicon_for(" ".join(beat.text for beat in scene.beats), self._lexicon)
        speech_rate = self._speech_rates.get(scene.scene_id)
        if lexicon:
            payload["lexicon"] = lexicon
        if speech_rate is not None:
            payload["speech_rate"] = speech_rate

        def operation() -> Awaitable[Any]:
            action = self._revision_actions.get(scene.scene_id)
            previous = self._revision_previous.get(scene.scene_id)
            recorded = (
                action is not None and previous is not None and previous.audio_path
                and Path(previous.audio_path).is_file()
                and self.store.has_paid_call(scene.scene_id, "tts")
            )
            if recorded and action.audio == "retime":
                return _return(retime(scene, previous))
            if recorded and action.audio == "repace":
                raw = raw_audio_path(self.artifacts.run_dir, previous)
                if raw is not None:
                    return service.repace(
                        scene, raw, scene_dir,
                        speech_rate if speech_rate is not None else service.speech_rate,
                    )
            return service.generate(scene, scene_dir, lexicon=lexicon, speech_rate=speech_rate)

        async with self._tts_limit:
            return await self._task(
                f"tts-{scene.scene_id}", "tts", scene.scene_id, payload, operation,
                dependencies=["narration"],
            )

    async def _contract_task(
        self,
        planner: PlanningService,
        batch: list[NarratedScene],
        lesson: LessonPlan,
        source: SourceDocument,
        batch_index: int,
    ) -> dict[str, Any]:
        # The creative generator no longer needs a paid SceneIR-like planning
        # call. Keep a small deterministic contract for fallback rendering and
        # report compatibility; ordinary Manim receives the lesson brief.
        del lesson, source
        return await self._task(
            f"visual-{batch_index:03d}", "creative_scaffold", None,
            {
                "scenes": [scene.model_dump(mode="json") for scene in batch],
                "stage_version": "creative-scaffold-v1",
            },
            lambda: _return(VisualContractBatch(
                contracts=[planner._fallback_contract(scene) for scene in batch]
            )),
            dependencies=["narration"],
        )

    async def _process_scene(
        self,
        *,
        input_path: Path,
        source: SourceDocument,
        lesson: LessonPlan,
        original_scene: NarratedScene,
        timed_scene_task: asyncio.Task,
        contract_task: asyncio.Task,
        scene_dir: Path,
        generator: CodeGenerationService,
        retry_attempt: int = 0,
    ) -> SceneResult:
        scene_id = original_scene.scene_id
        result = SceneResult(scene_id=scene_id)
        try:
            artifact_prefix = Path("scenes") / scene_id
            if retry_attempt:
                artifact_prefix /= Path("retries") / f"retry-{retry_attempt:02d}"
            lesson_scene = next(
                item for item in lesson.scenes if item.scene_id == scene_id
            )
            visual_history = _visual_history(lesson, scene_id)
            timed_scene = NarratedScene.model_validate(await timed_scene_task)
            contract_batch = VisualContractBatch.model_validate(await contract_task)
            contract = next(
                (item for item in contract_batch.contracts if item.scene_id == scene_id), None
            )
            if contract is None:
                contract = PlanningService._fallback_contract(timed_scene)
                self.artifacts.event(
                    "contract_fallback", stage="visual_contract", scene_id=scene_id,
                    message="Batch omitted this scene; generated deterministic artifact contract.",
                )
            contract_errors = validate_visual_contract(contract, timed_scene, source)
            if contract_errors:
                self.artifacts.event(
                    "contract_invalid", stage="visual_contract", scene_id=scene_id,
                    message="; ".join(contract_errors),
                )
                contract = PlanningService._fallback_contract(timed_scene)
                self.artifacts.event(
                    "contract_fallback", stage="visual_contract", scene_id=scene_id,
                    message="Replaced invalid visual contract with a grounded deterministic contract.",
                )
            self.artifacts.write_json(
                artifact_prefix / "scene.json",
                {"narration": timed_scene.model_dump(mode="json"),
                 "lesson": next(item.model_dump(mode="json") for item in lesson.scenes if item.scene_id == scene_id)},
            )
            self.artifacts.write_json(
                artifact_prefix / "visual_contract.json",
                contract.model_dump(mode="json"),
            )
            fact_ids = {fact_id for beat in timed_scene.beats for fact_id in beat.source_fact_ids}
            facts = [fact for fact in source.facts if fact.fact_id in fact_ids]
            code_payload: dict[str, Any] = {
                "scene": timed_scene.model_dump(mode="json"), "prompt_version": CODE_PROMPT_VERSION,
                "contract": contract.model_dump(mode="json"),
                "lesson_scene": lesson_scene.model_dump(mode="json"),
                "visual_history": visual_history,
                "retry_attempt": retry_attempt,
            }
            rules = notation_rules(self.directives)
            if rules:
                code_payload["notation_rules"] = rules
            notation_kwargs = {"notation_rules": rules} if rules else {}
            action = self._revision_actions.get(scene_id)
            previous_code = self._previous_code(scene_id) if action is not None else None
            code_mode = "generate"
            if action is not None and previous_code is not None:
                if action.code == "reinstrument":
                    code_mode = "reinstrument"
                    code_payload.update({
                        "code_mode": code_mode, "base_code": _hash_text(previous_code),
                    })
                elif action.instruction:
                    code_mode = "revise"
                    code_payload.update({
                        "revision_instruction": action.instruction,
                        "base_code": _hash_text(previous_code),
                    })

            def generate_code() -> Awaitable[str]:
                if code_mode == "reinstrument":
                    # Same visuals, new narration timing: no model call.
                    return _return(generator.reinstrument(previous_code, timed_scene, contract))
                if code_mode == "revise":
                    return generator.revise(
                        scene=timed_scene, contract=contract, code=previous_code,
                        instruction=action.instruction, lesson_scene=lesson_scene,
                        visual_history=visual_history,
                        **notation_kwargs,
                    )
                return generator.generate(
                    timed_scene, contract, facts, lesson_scene=lesson_scene,
                    visual_history=visual_history,
                    retry_attempt=retry_attempt,
                    **notation_kwargs,
                )

            async with self._code_limit:
                code_data = await self._task(
                    f"code-{scene_id}", "code_generation", scene_id, code_payload,
                    generate_code,
                    dependencies=[f"tts-{scene_id}", f"visual-{_batch_index(scene_id, lesson):03d}"],
                )
            code = str(code_data)
            original_code_path = scene_dir / "code" / "original.py"
            original_code_path.parent.mkdir(parents=True, exist_ok=True)
            original_code_path.write_text(code, encoding="utf-8")
            (scene_dir / "code" / "model_raw.py").write_text(
                deinstrument_scene_code(code), encoding="utf-8",
            )
            result.original_code_path = str(original_code_path.resolve())
            result.audio_path = timed_scene.audio_path
            if not self.config.enable_render:
                result.success = True
                result.status = "CODE_GENERATED"
                result.final_code_path = result.original_code_path
                return result

            async with self._preview_limit:
                original_data = await self._task(
                    f"preview-{scene_id}-00", "preview", scene_id,
                    {"code": _hash_text(code), "contract": contract.model_dump(mode="json"),
                     "verifier_version": VERIFIER_VERSION},
                    lambda: asyncio.to_thread(
                        self.renderer.render, code, contract, timed_scene,
                        scene_dir / "attempt-00", quality="low",
                        timeout_seconds=self.config.preview_timeout_seconds,
                    ),
                    dependencies=[f"code-{scene_id}"],
                )
            original = _outcome(original_data)
            self._check_notation(code, original)
            self.artifacts.write_json(
                artifact_prefix / "attempt-00" / "diagnostics.json",
                _serialize(original),
            )
            if self.config.verifier_mode == "audit":
                self._log_audit_outcome(scene_id, "preview", original)
            selected_code = code
            selected = original
            selected_preview_task = f"preview-{scene_id}-00"
            for repair_attempt in range(1, self.config.max_repairs_per_scene + 1):
                eligible = (
                    self.config.repair == "eligible"
                    and _repair_needed(selected, self.config.verifier_mode)
                )
                if not eligible:
                    break
                attempt_label = f"{repair_attempt:02d}"
                repair_task = f"repair-{scene_id}-{attempt_label}"
                preview_task = f"preview-{scene_id}-{attempt_label}"
                try:
                    async with self._repair_limit:
                        repaired_code_data = await self._task(
                            repair_task, "vision_repair", scene_id,
                            {
                                "code": _hash_text(selected_code),
                                "findings": [
                                    item.model_dump(mode="json")
                                    for item in selected.findings
                                ],
                            },
                            lambda: generator.repair(
                                scene=timed_scene, contract=contract, code=selected_code,
                                findings=selected.findings, frame_paths=selected.frame_paths,
                                lesson_scene=lesson_scene,
                                visual_history=visual_history,
                            ),
                            dependencies=[selected_preview_task],
                        )
                    repaired_code = str(repaired_code_data)
                    if _repair_code_unchanged(selected_code, repaired_code):
                        self.artifacts.event(
                            "repair_noop", stage="vision_repair", scene_id=scene_id,
                            message="Repair returned code identical to the current candidate.",
                            repair_attempt=repair_attempt,
                        )
                        break
                    else:
                        repaired_path = scene_dir / "code" / f"repaired-{attempt_label}.py"
                        repaired_path.write_text(repaired_code, encoding="utf-8")
                        (scene_dir / "code" / f"repaired_raw-{attempt_label}.py").write_text(
                            deinstrument_scene_code(repaired_code), encoding="utf-8",
                        )
                        async with self._preview_limit:
                            repaired_data = await self._task(
                                preview_task, "preview", scene_id,
                                {"code": _hash_text(repaired_code), "verifier_version": VERIFIER_VERSION,
                                 "contract": contract.model_dump(mode="json")},
                                lambda: asyncio.to_thread(
                                    self.renderer.render, repaired_code, contract, timed_scene,
                                    scene_dir / f"attempt-{attempt_label}", quality="low",
                                    timeout_seconds=self.config.preview_timeout_seconds,
                                ),
                                dependencies=[repair_task],
                            )
                        repaired = _outcome(repaired_data)
                        self._check_notation(repaired_code, repaired)
                        self.artifacts.write_json(
                            artifact_prefix / f"attempt-{attempt_label}" / "diagnostics.json",
                            _serialize(repaired),
                        )
                        if self.config.verifier_mode == "audit":
                            self._log_audit_outcome(scene_id, "repair_preview", repaired)
                        visual_changed = _repair_visuals_changed(selected, repaired)
                        if _repair_improves(selected, repaired) and visual_changed:
                            before_score = _quality(selected)
                            selected_code, selected = repaired_code, repaired
                            selected_preview_task = preview_task
                            result.repaired = True
                            self.artifacts.event(
                                "repair_accepted", stage="vision_repair", scene_id=scene_id,
                                before_score=before_score, after_score=_quality(repaired),
                                repair_attempt=repair_attempt,
                            )
                        else:
                            self.artifacts.event(
                                "repair_rejected", stage="vision_repair", scene_id=scene_id,
                                message=("Checkpoint visuals were unchanged." if not visual_changed
                                         else "Objective diagnostics did not improve."),
                                before_score=_quality(selected), after_score=_quality(repaired),
                                repair_attempt=repair_attempt,
                            )
                            break
                except (MissingProviderKey, BudgetExceeded) as exc:
                    self.artifacts.event(
                        "repair_skipped", stage="vision_repair", scene_id=scene_id,
                        message=str(exc), repair_attempt=repair_attempt,
                    )
                    break
                except Exception as exc:
                    self.artifacts.event(
                        "repair_failed", stage="vision_repair", scene_id=scene_id,
                        message=str(exc), repair_attempt=repair_attempt,
                    )
                    break

            final_code_path = scene_dir / "code" / "final.py"
            final_code_path.write_text(selected_code, encoding="utf-8")
            result.final_code_path = str(final_code_path.resolve())
            result.preview_path = selected.video_path
            result.findings = selected.findings
            result.frame_paths = selected.frame_paths

            selected_acceptable = _render_acceptable(
                selected, self.config.verifier_mode, self.config.policy,
            )
            if selected_acceptable:
                async with self._render_limit:
                    final_data = await self._task(
                        f"render-{scene_id}", "high_render", scene_id,
                        {"code": _hash_text(selected_code), "quality": self.config.quality,
                         "verifier_version": VERIFIER_VERSION},
                        lambda: asyncio.to_thread(
                            self.renderer.render, selected_code, contract, timed_scene,
                            scene_dir / "final-render", quality=self.config.quality,
                            timeout_seconds=self.config.render_timeout_seconds,
                        ),
                        dependencies=[selected_preview_task],
                    )
                final_render = _outcome(final_data)
                self._check_notation(selected_code, final_render)
                result.findings = final_render.findings
                result.frame_paths = final_render.frame_paths or selected.frame_paths
                if self.config.verifier_mode == "audit":
                    self._log_audit_outcome(scene_id, "high_render", final_render)
                final_acceptable = _render_acceptable(
                    final_render, self.config.verifier_mode, self.config.policy,
                )
                render_path = Path(final_render.video_path) if final_acceptable and final_render.video_path else Path(selected.video_path)
                result.render_video_path = str(render_path)
                if not final_acceptable and self.config.policy == "strict":
                    raise RuntimeError(final_render.error or "High-quality render failed")
                if not final_acceptable and _source_fallback_allowed(
                    self.config.policy, timed_scene.audio_path
                ):
                    page = _fallback_page(lesson_scene.source_fact_ids, source)
                    fallback = await self._task(
                        f"compose-{scene_id}", "source_fallback", scene_id,
                        {"pdf": _file_signature(input_path), "page": page,
                         "audio": _file_signature(Path(timed_scene.audio_path))},
                        lambda: asyncio.to_thread(
                            source_page_fallback, input_path, page, Path(timed_scene.audio_path),
                            scene_dir / "source_fallback.mp4",
                        ),
                        dependencies=[f"render-{scene_id}", f"tts-{scene_id}"],
                    )
                    result.success = True
                    result.status = "SOURCE_FALLBACK"
                    result.fallback_used = True
                    result.final_video_path = str(fallback)
                    result.findings = final_render.findings
                    return result
                if timed_scene.audio_path:
                    composed = await self._task(
                        f"compose-{scene_id}", "composition", scene_id,
                        {"video": _file_signature(render_path),
                         "audio": _file_signature(Path(timed_scene.audio_path)),
                         "code": _hash_text(selected_code)},
                        lambda: asyncio.to_thread(
                            compose_scene, render_path, Path(timed_scene.audio_path),
                            scene_dir / "final_with_narration.mp4",
                        ),
                        dependencies=[f"render-{scene_id}", f"tts-{scene_id}"],
                    )
                    result.final_video_path = str(composed)
                else:
                    result.final_video_path = str(render_path.resolve())
                result.success = True
                retained_findings = bool(final_render.findings)
                if (
                    self.config.verifier_mode == "enforce"
                    and self.config.policy == "standard"
                    and retained_findings
                ):
                    result.status = (
                        "REPAIRED_ACCEPTED_WITH_FINDINGS"
                        if result.repaired else "ACCEPTED_WITH_FINDINGS"
                    )
                    self.artifacts.event(
                        "verifier_findings_retained",
                        stage="high_render",
                        scene_id=scene_id,
                        message=(
                            "Standard policy retained the successfully rendered Manim scene "
                            "after bounded repair instead of using a source-page fallback."
                        ),
                        error_count=sum(
                            item.severity == "error" for item in final_render.findings
                        ),
                        warning_count=sum(
                            item.severity == "warning" for item in final_render.findings
                        ),
                    )
                elif self.config.verifier_mode == "audit":
                    result.status = (
                        "AUDIT_REPAIRED_ACCEPTED" if result.repaired else "AUDIT_ACCEPTED"
                    )
                else:
                    result.status = "ACCEPTED" if not result.repaired else "REPAIRED_ACCEPTED"
                return result

            if _source_fallback_allowed(self.config.policy, timed_scene.audio_path):
                page = _fallback_page(lesson_scene.source_fact_ids, source)
                fallback = await self._task(
                    f"compose-{scene_id}", "source_fallback", scene_id,
                    {"pdf": _file_signature(input_path), "page": page,
                     "audio": _file_signature(Path(timed_scene.audio_path))},
                    lambda: asyncio.to_thread(
                        source_page_fallback, input_path, page, Path(timed_scene.audio_path),
                        scene_dir / "source_fallback.mp4",
                    ),
                    dependencies=[selected_preview_task, f"tts-{scene_id}"],
                )
                result.success = True
                result.status = "SOURCE_FALLBACK"
                result.fallback_used = True
                result.final_video_path = str(fallback)
                return result
            raise RuntimeError(selected.error or "Scene did not pass preview rendering")
        except BudgetExceeded:
            raise
        except Exception as exc:
            result.success = False
            result.status = "FAILED"
            result.error = str(exc)
            self.artifacts.event(
                "scene_failed", stage="scene", scene_id=scene_id, message=str(exc),
            )
            return result

    def _check_notation(self, code: str, outcome: RenderOutcome) -> None:
        for message in notation_findings(code, self.directives):
            outcome.findings.append(VerificationFinding(
                code="NOTATION_DIRECTIVE", severity="error", message=message,
            ))

    def _previous_code(self, scene_id: str) -> Optional[str]:
        previous = self._previous_scene_results.get(scene_id)
        path = previous.final_code_path if previous else None
        if not path or not Path(path).is_file():
            return None
        return Path(path).read_text(encoding="utf-8")

    def _log_audit_outcome(
        self, scene_id: str, stage: str, outcome: RenderOutcome,
    ) -> None:
        errors = sum(item.severity == "error" for item in outcome.findings)
        warnings = sum(item.severity == "warning" for item in outcome.findings)
        self.artifacts.event(
            "verifier_audit_observed",
            stage=stage,
            scene_id=scene_id,
            message=(
                f"Audit verifier recorded {errors} errors and {warnings} warnings; "
                "visual findings will not reject a renderable scene."
            ),
            render_success=outcome.render_success,
            safe=outcome.safe,
            error_count=errors,
            warning_count=warnings,
            issue_score=issue_score(outcome.findings),
            finding_codes=[item.code for item in outcome.findings],
        )

    async def _task(
        self,
        task_id: str,
        stage: str,
        scene_id: Optional[str],
        input_payload: Any,
        operation: Callable[[], Awaitable[T]],
        dependencies: Optional[list[str]] = None,
    ) -> Any:
        payload = _serialize(input_payload)
        input_hash = _hash_text(json.dumps(payload, sort_keys=True, default=str))
        self.store.upsert_task(task_id, stage, scene_id, input_hash, dependencies)
        worker = asyncio.current_task().get_name() if asyncio.current_task() else "main"
        cached = self.store.task_output(task_id, input_hash)
        if cached is not None:
            self.artifacts.event(
                "task_resumed", stage=stage, scene_id=scene_id, task_id=task_id,
                message="Reused successful task output.", input_hash=input_hash,
                dependencies=dependencies or [], worker=worker,
            )
            return cached
        attempt = self.store.start_task(task_id)
        start = time.perf_counter()
        self.artifacts.event(
            "task_started", stage=stage, scene_id=scene_id, task_id=task_id,
            attempt=attempt, input_hash=input_hash, dependencies=dependencies or [],
            worker=worker,
        )
        try:
            value = await operation()
            output = _serialize(value)
            duration_ms = (time.perf_counter() - start) * 1000
            self.store.finish_task(task_id, output, duration_ms)
            self.artifacts.event(
                "task_finished", stage=stage, scene_id=scene_id, task_id=task_id,
                attempt=attempt, duration_ms=round(duration_ms, 2), status="SUCCEEDED",
                dependencies=dependencies or [], worker=worker,
            )
            return output
        except asyncio.CancelledError:
            duration_ms = (time.perf_counter() - start) * 1000
            self.store.fail_task(task_id, "cancelled", duration_ms, TaskStatus.INTERRUPTED)
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            self.store.fail_task(task_id, str(exc), duration_ms)
            self.artifacts.event(
                "task_failed", stage=stage, scene_id=scene_id, task_id=task_id,
                attempt=attempt, duration_ms=round(duration_ms, 2),
                message=str(exc), error_type=type(exc).__name__, worker=worker,
            )
            raise

    def _prepare_run(self, input_path: Path, resume_dir: Optional[Path]) -> tuple[Path, str]:
        if resume_dir:
            return resume_dir, resume_dir.name
        slug = "".join(character if character.isalnum() else "_" for character in input_path.stem).strip("_")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = self.config.output_dir.resolve() / (slug or "document") / stamp
        suffix = 1
        candidate = run_dir
        while candidate.exists():
            candidate = run_dir.with_name(f"{run_dir.name}-{suffix:02d}")
            suffix += 1
        if os.name == "nt" and len(str(candidate)) > 180:
            raise ValueError(
                "The output directory is too deeply nested for reliable Windows media and "
                "artifact paths. Choose a shorter --output-dir."
            )
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate, candidate.name

    def _write_run_json(self, input_path: Path, status: RunStatus, result: Optional[RunResult] = None) -> None:
        payload = {
            "version": "2.0.0",
            "run_id": self.artifacts.run_id,
            "status": status.value,
            "input_pdf": str(input_path),
            "config": self.config.model_dump(mode="json"),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if result:
            payload["result"] = result.to_dict()
        elif self._previous_result:
            payload["result"] = self._previous_result
        self.artifacts.write_json("run.json", payload)

    def _finalize(self, input_path: Path, result: RunResult) -> RunResult:
        if self._retry_scene_ids is not None and not result.scene_results:
            # A retry or revision that stopped early (budget, interruption)
            # must not erase the scene results later retries build on.
            result.scene_results = list(self._previous_scene_results.values())
            result.total_scenes = len(result.scene_results)
            result.successful_scenes = sum(scene.success for scene in result.scene_results)
            result.failed_scenes = result.total_scenes - result.successful_scenes
        result.estimated_cost_usd = self.store.total_cost()
        self.store.set_meta("status", result.status.value)
        write_costs(self.artifacts, self.store)
        report = write_html_report(self.artifacts, self.store, result)
        result.report_path = str(report)
        self._write_run_json(input_path, result.status, result)
        self.artifacts.event(
            "run_finished", message=result.message or result.status.value,
            status=result.status.value, total_scenes=result.total_scenes,
            successful_scenes=result.successful_scenes, failed_scenes=result.failed_scenes,
            cost_usd=result.estimated_cost_usd,
        )
        return result


async def _return(value: T) -> T:
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _serialize(value: Any) -> Any:
    if isinstance(value, RenderOutcome):
        return {
            "safe": value.safe,
            "render_success": value.render_success,
            "scene_class": value.scene_class,
            "video_path": value.video_path,
            "trace_path": value.trace_path,
            "frame_paths": list(value.frame_paths),
            "duration_seconds": value.duration_seconds,
            "findings": [item.model_dump(mode="json") for item in value.findings],
            "error": value.error,
        }
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _outcome(value: Any) -> RenderOutcome:
    if isinstance(value, RenderOutcome):
        return value
    payload = dict(value)
    from .models import VerificationFinding
    payload["findings"] = [VerificationFinding.model_validate(item) for item in payload.get("findings", [])]
    return RenderOutcome(**payload)


def _quality(outcome: RenderOutcome) -> tuple[int, int]:
    gate = 0 if outcome.safe and outcome.render_success else 1
    return gate, issue_score(outcome.findings)


def _repair_code_unchanged(original_code: str, repaired_code: str) -> bool:
    return _hash_text(original_code) == _hash_text(repaired_code)


def _repair_improves(original: RenderOutcome, repaired: RenderOutcome) -> bool:
    return _quality(repaired) < _quality(original)


def _repair_needed(outcome: RenderOutcome, verifier_mode: str) -> bool:
    if not outcome.render_success:
        return True
    if verifier_mode == "audit":
        return False
    if any(item.severity == "error" for item in outcome.findings):
        return True
    overlap_warnings = sum(
        item.severity == "warning" and item.code == "TEXT_TEXT_OVERLAP"
        for item in outcome.findings
    )
    return overlap_warnings >= 2


def _render_acceptable(
    outcome: RenderOutcome,
    verifier_mode: str,
    policy: str = "standard",
) -> bool:
    if not outcome.safe or not outcome.render_success:
        return False
    if verifier_mode == "audit" or policy == "standard":
        return True
    return not any(
        item.severity == "error" for item in outcome.findings
    )


def _repair_visuals_changed(original: RenderOutcome, repaired: RenderOutcome) -> bool:
    return (
        not original.frame_paths
        or not repaired.frame_paths
        or frame_fingerprint(repaired.frame_paths) != frame_fingerprint(original.frame_paths)
    )


def _source_fallback_allowed(policy: str, audio_path: Optional[str]) -> bool:
    return policy == "standard" and bool(audio_path)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_signature(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "sha256": "missing"}
    return {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _batch_index(scene_id: str, lesson: LessonPlan) -> int:
    position = next(
        (index for index, scene in enumerate(lesson.scenes) if scene.scene_id == scene_id),
        0,
    )
    return position // 3


def _visual_history(lesson: LessonPlan, scene_id: str) -> list[dict[str, object]]:
    """Give generation compact continuity without serializing earlier Manim code."""

    position = next(
        (index for index, scene in enumerate(lesson.scenes) if scene.scene_id == scene_id),
        0,
    )
    history = []
    for scene in lesson.scenes[max(0, position - 3):position]:
        history.append({
            "scene_id": scene.scene_id,
            "title": scene.title,
            "learning_goal": scene.learning_goal,
            "visual_objective": scene.visual_objective,
            "visual_fact_already_shown": scene.new_visual_fact,
            "constructions_not_to_replay": scene.do_not_repeat,
        })
    return history


def _audit_repeated_visuals(
    scene_results: list[SceneResult], artifacts: RunArtifacts,
) -> None:
    previous: SceneResult | None = None
    for current in scene_results:
        if not current.success or len(current.frame_paths) < 2:
            continue
        if previous is not None and len(previous.frame_paths) >= 2:
            similarity = frame_sequence_similarity(previous.frame_paths, current.frame_paths)
            if similarity >= 0.94:
                finding = VerificationFinding(
                    code="REDUNDANT_VISUAL_REPLAY",
                    severity="warning",
                    message=(
                        f"Checkpoint sequence is {similarity:.0%} visually similar to "
                        f"{previous.scene_id}; review whether this scene adds a new visual step."
                    ),
                    screenshot_path=current.frame_paths[0],
                )
                current.findings.append(finding)
                artifacts.event(
                    "repeated_visual_sequence",
                    stage="final_review",
                    scene_id=current.scene_id,
                    message=finding.message,
                    previous_scene_id=previous.scene_id,
                    similarity=round(similarity, 4),
                )
        previous = current


def _fallback_page(fact_ids: list[str], source: SourceDocument) -> int:
    mapping = {fact.fact_id: fact.page_number for fact in source.facts}
    return next((mapping[item] for item in fact_ids if item in mapping), 1)


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None
