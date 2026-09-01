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

from .diagnostics import frame_fingerprint, issue_score
from .generation import CODE_PROMPT_VERSION, CodeGenerationService
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
    VisualContractBatch,
)
from .observability import RunArtifacts
from .planning import (
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
from .state import BudgetExceeded, RunStore
from .tts import NarrationAudioService


T = TypeVar("T")


class PipelineV2:
    def __init__(self, config: Optional[PipelineV2Config] = None):
        self.config = config or PipelineV2Config()
        self.store: Optional[RunStore] = None
        self.artifacts: Optional[RunArtifacts] = None
        self.renderer = TrackedManimRenderer()
        self._repair_count = 0
        self._repair_lock = asyncio.Lock()
        self._visual_limit = asyncio.Semaphore(self.config.visual_workers)
        self._tts_limit = asyncio.Semaphore(self.config.tts_workers)
        self._code_limit = asyncio.Semaphore(self.config.code_workers)
        self._preview_limit = asyncio.Semaphore(self.config.preview_workers)
        self._repair_limit = asyncio.Semaphore(self.config.repair_workers)
        self._render_limit = asyncio.Semaphore(self.config.render_workers)

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
        return cls(config).run(payload["input_pdf"], run_dir=path)

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
        )
        self._write_run_json(input_path, RunStatus.RUNNING)
        router = ProviderRouter(
            self.store, self.artifacts,
            max_api_workers=self.config.global_api_workers,
            strict_keys=self.config.strict_keys,
        )
        planner = PlanningService(router, self.config.max_scenes)
        generator = CodeGenerationService(router)
        audio = NarrationAudioService(
            self.store, self.artifacts, voice=self.config.voice,
            high_quality=self.config.quality == "high", strict_keys=self.config.strict_keys,
            api_limit=router.api_limit,
        )
        result = RunResult(run_id=run_id, run_directory=str(run_dir), status=RunStatus.RUNNING)

        try:
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
            self.artifacts.write_json("source/document.json", source.model_dump(mode="json"))

            lesson_data = await self._task(
                "lesson", "lesson", None,
                {"source": source.model_dump(mode="json"), "prompt_version": PROMPT_VERSION,
                 "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                 "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL},
                lambda: planner.lesson(source),
                dependencies=["ocr-enrichment"],
            )
            lesson = LessonPlan.model_validate(lesson_data)
            lesson = lesson.model_copy(update={
                "scenes": [scene.model_copy(update={"delivery_mode": "manim"}) for scene in lesson.scenes]
            })
            narration_data = await self._task(
                "narration", "narration", None,
                {"lesson": lesson.model_dump(mode="json"), "prompt_version": PROMPT_VERSION,
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
                    raise RuntimeError("Grounding validation failed: " + "; ".join(grounding_errors))
                lesson = planner._fallback_lesson(source)
                narration = planner._fallback_narration(lesson, source)
                self.artifacts.event(
                    "grounding_fallback", stage="narration",
                    message="Replaced unsupported lesson output with a deterministic source-grounded plan.",
                )
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
                if self.config.enable_tts:
                    tts_tasks[scene.scene_id] = asyncio.create_task(
                        self._tts_task(audio, scene, scene_dirs[scene.scene_id]),
                        name=f"tts-{scene.scene_id}",
                    )
                else:
                    duration = max(1.5, len(scene.narration.split()) / 2.8)
                    tts_tasks[scene.scene_id] = asyncio.create_task(
                        _return(apply_audio_timing(scene, duration)),
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

            scene_jobs = [asyncio.create_task(
                self._process_scene(
                    input_path=input_path,
                    source=source,
                    lesson=lesson,
                    original_scene=scene,
                    timed_scene_task=tts_tasks[scene.scene_id],
                    contract_task=contract_tasks[scene_batch[scene.scene_id]],
                    scene_dir=scene_dirs[scene.scene_id],
                    generator=generator,
                ),
                name=f"scene-{scene.scene_id}",
            ) for scene in narration.scenes]
            result.scene_results = await asyncio.gather(*scene_jobs)
            timed_scenes = [NarratedScene.model_validate(await tts_tasks[scene.scene_id])
                            for scene in narration.scenes]
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
        async with self._tts_limit:
            return await self._task(
                f"tts-{scene.scene_id}", "tts", scene.scene_id,
                {"narration": scene.narration, "voice": self.config.voice, "stage_version": "tts-v2-1",
                 "quality": self.config.quality},
                lambda: service.generate(scene, scene_dir),
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
        async with self._visual_limit:
            return await self._task(
                f"visual-{batch_index:03d}", "visual_contract", None,
                {"scenes": [scene.model_dump(mode="json") for scene in batch],
                 "prompt_version": PROMPT_VERSION,
                 "structured_output_version": STRUCTURED_OUTPUT_VERSION},
                lambda: planner.visual_contract_batch(batch, lesson, source, batch_index),
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
    ) -> SceneResult:
        scene_id = original_scene.scene_id
        result = SceneResult(scene_id=scene_id)
        try:
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
                Path("scenes") / scene_id / "scene.json",
                {"narration": timed_scene.model_dump(mode="json"),
                 "lesson": next(item.model_dump(mode="json") for item in lesson.scenes if item.scene_id == scene_id)},
            )
            self.artifacts.write_json(
                Path("scenes") / scene_id / "visual_contract.json",
                contract.model_dump(mode="json"),
            )
            fact_ids = {fact_id for beat in timed_scene.beats for fact_id in beat.source_fact_ids}
            facts = [fact for fact in source.facts if fact.fact_id in fact_ids]
            async with self._code_limit:
                code_data = await self._task(
                    f"code-{scene_id}", "code_generation", scene_id,
                    {"scene": timed_scene.model_dump(mode="json"), "prompt_version": CODE_PROMPT_VERSION,
                     "contract": contract.model_dump(mode="json")},
                    lambda: generator.generate(timed_scene, contract, facts),
                    dependencies=[f"tts-{scene_id}", f"visual-{_batch_index(scene_id, lesson):03d}"],
                )
            code = str(code_data)
            original_code_path = scene_dir / "code" / "original.py"
            original_code_path.parent.mkdir(parents=True, exist_ok=True)
            original_code_path.write_text(code, encoding="utf-8")
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
            self.artifacts.write_json(
                Path("scenes") / scene_id / "attempt-00" / "diagnostics.json",
                _serialize(original),
            )
            if self.config.verifier_mode == "audit":
                self._log_audit_outcome(scene_id, "preview", original)
            selected_code = code
            selected = original
            selected_preview_task = f"preview-{scene_id}-00"
            eligible = (
                self.config.repair == "eligible"
                and self.config.max_repairs_per_scene > 0
                and _repair_needed(original, self.config.verifier_mode)
            )
            if eligible and await self._claim_repair():
                try:
                    async with self._repair_limit:
                        repaired_code_data = await self._task(
                            f"repair-{scene_id}-01", "vision_repair", scene_id,
                            {"code": _hash_text(code),
                             "findings": [item.model_dump(mode="json") for item in original.findings]},
                            lambda: generator.repair(
                                scene=timed_scene, contract=contract, code=code,
                                findings=original.findings, frame_paths=original.frame_paths,
                            ),
                            dependencies=[f"preview-{scene_id}-00"],
                        )
                    repaired_code = str(repaired_code_data)
                    if _repair_code_unchanged(code, repaired_code):
                        self.artifacts.event(
                            "repair_noop", stage="vision_repair", scene_id=scene_id,
                            message="Repair returned code identical to the original.",
                        )
                    else:
                        repaired_path = scene_dir / "code" / "repaired.py"
                        repaired_path.write_text(repaired_code, encoding="utf-8")
                        async with self._preview_limit:
                            repaired_data = await self._task(
                                f"preview-{scene_id}-01", "preview", scene_id,
                                {"code": _hash_text(repaired_code), "verifier_version": VERIFIER_VERSION,
                                 "contract": contract.model_dump(mode="json")},
                                lambda: asyncio.to_thread(
                                    self.renderer.render, repaired_code, contract, timed_scene,
                                    scene_dir / "attempt-01", quality="low",
                                    timeout_seconds=self.config.preview_timeout_seconds,
                                ),
                                dependencies=[f"repair-{scene_id}-01"],
                            )
                        repaired = _outcome(repaired_data)
                        self.artifacts.write_json(
                            Path("scenes") / scene_id / "attempt-01" / "diagnostics.json",
                            _serialize(repaired),
                        )
                        if self.config.verifier_mode == "audit":
                            self._log_audit_outcome(scene_id, "repair_preview", repaired)
                        visual_changed = _repair_visuals_changed(original, repaired)
                        if _repair_improves(original, repaired) and visual_changed:
                            selected_code, selected = repaired_code, repaired
                            selected_preview_task = f"preview-{scene_id}-01"
                            result.repaired = True
                            self.artifacts.event(
                                "repair_accepted", stage="vision_repair", scene_id=scene_id,
                                before_score=_quality(original), after_score=_quality(repaired),
                            )
                        else:
                            self.artifacts.event(
                                "repair_rejected", stage="vision_repair", scene_id=scene_id,
                                message=("Checkpoint visuals were unchanged." if not visual_changed
                                         else "Objective diagnostics did not improve."),
                                before_score=_quality(original), after_score=_quality(repaired),
                            )
                except (MissingProviderKey, BudgetExceeded) as exc:
                    self.artifacts.event(
                        "repair_skipped", stage="vision_repair", scene_id=scene_id,
                        message=str(exc),
                    )
                except Exception as exc:
                    self.artifacts.event(
                        "repair_failed", stage="vision_repair", scene_id=scene_id,
                        message=str(exc),
                    )

            final_code_path = scene_dir / "code" / "final.py"
            final_code_path.write_text(selected_code, encoding="utf-8")
            result.final_code_path = str(final_code_path.resolve())
            result.preview_path = selected.video_path
            result.findings = selected.findings
            result.frame_paths = selected.frame_paths

            selected_acceptable = _render_acceptable(selected, self.config.verifier_mode)
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
                result.findings = final_render.findings
                result.frame_paths = final_render.frame_paths or selected.frame_paths
                if self.config.verifier_mode == "audit":
                    self._log_audit_outcome(scene_id, "high_render", final_render)
                final_acceptable = _render_acceptable(final_render, self.config.verifier_mode)
                render_path = Path(final_render.video_path) if final_acceptable and final_render.video_path else Path(selected.video_path)
                if not final_acceptable and self.config.policy == "strict":
                    raise RuntimeError(final_render.error or "High-quality render failed")
                if not final_acceptable and _source_fallback_allowed(
                    self.config.policy, timed_scene.audio_path
                ):
                    lesson_scene = next(item for item in lesson.scenes if item.scene_id == scene_id)
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
                if self.config.verifier_mode == "audit":
                    result.status = (
                        "AUDIT_REPAIRED_ACCEPTED" if result.repaired else "AUDIT_ACCEPTED"
                    )
                else:
                    result.status = "ACCEPTED" if not result.repaired else "REPAIRED_ACCEPTED"
                return result

            if _source_fallback_allowed(self.config.policy, timed_scene.audio_path):
                lesson_scene = next(item for item in lesson.scenes if item.scene_id == scene_id)
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

    async def _claim_repair(self) -> bool:
        async with self._repair_lock:
            if self._repair_count >= self.config.max_repairs_per_run:
                return False
            self._repair_count += 1
            return True

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
        self.artifacts.write_json("run.json", payload)

    def _finalize(self, input_path: Path, result: RunResult) -> RunResult:
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
    return (
        any(item.severity == "error" for item in outcome.findings)
        or sum(item.severity == "warning" for item in outcome.findings) >= 3
    )


def _render_acceptable(outcome: RenderOutcome, verifier_mode: str) -> bool:
    if not outcome.safe or not outcome.render_success:
        return False
    return verifier_mode == "audit" or not any(
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
