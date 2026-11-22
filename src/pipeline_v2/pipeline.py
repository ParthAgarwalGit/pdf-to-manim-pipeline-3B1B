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
import uuid
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
    BEAT_FIX_PROMPT_VERSION,
    HOOK_PROMPT_VERSION,
    PLANNING_MODEL,
    PLANNING_PROVIDER,
    PlanningService,
    apply_audio_timing,
    extract_source,
    hook_problems,
    prompt_version,
    prompt_version_aliases,
    validate_grounding,
    validate_visual_contract,
)
from .presenter import (
    BRIDGES_PROMPT_VERSION,
    BridgeProposal,
    PresenterBudget,
    PresenterClip,
    PresenterRenderError,
    PresenterRequest,
    PresenterSettings,
    TitleCardPresenter,
    approve_bridges,
    bridge_fingerprint,
    bridge_scene,
    build_presenter,
    interleave_segments,
    load_bridges,
    merge_proposals,
    presenter_usd_per_second,
    propose_bridges,
    save_bridges,
    write_bridges,
)
from .prompt_text import BEAT_FIX_MAX_EXTRA_WORDS, DEFAULT_VOICE
from .providers import MissingProviderKey, ProviderRouter, STRUCTURED_OUTPUT_VERSION
from .rendering import VERIFIER_VERSION, RenderOutcome, TrackedManimRenderer
from .report import write_costs, write_html_report, write_storyboard
from .script_review import (
    SCRIPT_REVIEW_PROMPT_VERSION,
    ScriptReview,
    locate_flow_breaks,
    review_script,
    splice_beats,
    with_voice_check,
)
from .model_config import PRESENTER_MODEL, TEXT_MODEL
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
    OPENAI_VOICES,
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
# Tasks whose re-run re-plans the video: a resume warns before redoing one
# that already succeeded.
REPLAN_TASKS = frozenset({"ocr-enrichment", "lesson", "narration", "narration-revised"})
# A hook rewrite, then at most one more that fixes only the first one's shape.
HOOK_REWRITE_ATTEMPTS = 2
# Words a beat fix may add beyond the BEAT_FIX_MAX_EXTRA_WORDS it is asked for:
# the model counts loosely, but a fix much longer than asked is rejected.
BEAT_FIX_WORD_TOLERANCE = 5

# A fade through black at every cut between a presenter clip and a Manim scene:
# the outgoing clip fades out and the incoming one fades in, each this long.
PRESENTER_FADE_SECONDS = 0.25
PRESENTER_RENDER_FILE = "presenter/render.json"


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
        # The run's status before this run started (None for a new run).
        self._status_before: Optional[str] = None
        # Revision state: per-scene actions, the pre-revision timed narration
        # they start from, and run-wide pronunciation/pacing overrides.
        self._revision_actions: dict[str, SceneAction] = {}
        self._revision_previous: dict[str, NarratedScene] = {}
        self._lexicon: dict[str, str] = {}
        self._speech_rates: dict[str, float] = {}
        # Presenter bridges: built at the start of a run with a presenter, so a
        # refused configuration fails before any paid call.
        self.presenter_settings: Optional[PresenterSettings] = None
        self.presenter_provider: Any = None

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
        self._status_before = self.store.get_meta("status")
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
            presenter_opens=self.config.presenter != "off",
            mode=self.config.mode,
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
            provider=self.config.tts_provider, audience=self.config.audience,
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
            if self.config.presenter != "off":
                # Fail fast: a refused presenter configuration (missing key,
                # consent, image) stops the run before its first paid call.
                self.presenter_settings = self._presenter_settings(run_dir)
                self.presenter_provider = build_presenter(self.presenter_settings)
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
                ocr_inputs = self._ocr_inputs(source)
                enriched_data = await self._task(
                    "ocr-enrichment", "ocr", None, ocr_inputs,
                    lambda: planner.enrich_low_confidence_pages(
                        source, input_path, run_dir / "source" / "ocr"
                    ),
                    dependencies=["source"],
                    equivalent_inputs=self._equivalent_inputs(ocr_inputs, "planning"),
                )
                source = SourceDocument.model_validate(enriched_data)
                source, scope_checks = apply_scope(source, self.directives)
                compliance.extend(scope_checks)
                self.artifacts.write_json("source/document.json", source.model_dump(mode="json"))

                lesson_inputs = self._lesson_inputs(source, controls)
                lesson_data = await self._task(
                    "lesson", "lesson", None, lesson_inputs,
                    lambda: planner.lesson(source),
                    dependencies=["ocr-enrichment"],
                    equivalent_inputs=self._equivalent_inputs(lesson_inputs, "planning"),
                )
                lesson = LessonPlan.model_validate(lesson_data)
                lesson = lesson.model_copy(update={
                    "scenes": [
                        scene.model_copy(update={"delivery_mode": "manim"})
                        for scene in lesson.scenes
                    ]
                })
                lesson = await self._settle_hook(planner, lesson)
                narration_inputs = self._narration_inputs(lesson, controls)
                narration_data = await self._task(
                    "narration", "narration", None, narration_inputs,
                    lambda: planner.narration(lesson, source),
                    dependencies=["lesson"],
                    equivalent_inputs=self._equivalent_inputs(narration_inputs, "narration"),
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
            # Only freshly planned scripts are reviewed: a scene retry or a
            # professor revision reuses a storyboard that must not be rewritten.
            if (
                self.config.script_review != "off"
                and self._retry_scene_ids is None and frozen_plan is None
            ):
                reviewed, replacement_checks = await self._review_script(
                    router, planner, lesson, narration, source, controls,
                )
                if replacement_checks is not None:
                    narration = reviewed
                    compliance = [
                        item for item in compliance
                        if not any(item is check for check in narration_checks)
                    ] + replacement_checks
            if controls:
                self.artifacts.write_json("professor_controls.json", controls)
                self.artifacts.write_json("compliance.json", [
                    item.model_dump(mode="json") for item in compliance
                ])
                if planner.length_report:
                    self.artifacts.write_json("length_budget.json", planner.length_report)
                if self.config.policy == "strict" and any(not item.passed for item in compliance):
                    raise RuntimeError("Professor directives failed compliance; inspect compliance.json")
            bridges: list[BridgeProposal] = []
            if self.presenter_provider is not None:
                bridges = await self._prepare_bridges(router, lesson, narration, run_dir)
                narration = self._keep_hook(lesson, narration, bridges)
                if self._retry_scene_ids is None:
                    self._refresh_first_scene(narration)
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
            strict_failure = self.config.policy == "strict" and result.failed_scenes > 0
            bridge_clips: dict[str, tuple[BridgeProposal, PresenterClip]] = {}
            fades: Optional[list[tuple[float, float]]] = None
            if (
                self.presenter_provider is not None and self.config.enable_render
                and ordered_videos and not strict_failure
            ):
                bridge_clips = await self._render_bridges(
                    bridges, audio, lesson, NarrationPlan(scenes=timed_scenes), run_dir,
                    {segment.scene_id for segment in timeline},
                )
                if bridge_clips:
                    # Each approved clip sits before its scene; no scene is replaced.
                    timeline = interleave_segments(timeline, bridge_clips)
                    ordered_videos = [Path(item.video_path) for item in timeline]
                    cursor = sum(item.duration_seconds for item in timeline)
                    fades = _presenter_fades(timeline)
            self.artifacts.write_json("timeline.json", [item.model_dump(mode="json") for item in timeline])
            # Fades enter the compose inputs only when there are presenter cuts,
            # so a run without bridges keeps its task hash and its join command.
            fade_kwargs: dict[str, Any] = {"fades": fades} if fades else {}

            if self.config.enable_render and ordered_videos and not strict_failure:
                compose_inputs = {
                    "videos": [_file_signature(path) for path in ordered_videos], **fade_kwargs,
                }
                # With a presenter the clip list changes as bridges are approved,
                # edited or rejected, and the store would reuse "final-compose"
                # regardless; its inputs are part of the id instead.
                compose_id = (
                    "final-compose" if self.presenter_provider is None
                    else f"final-compose-{_digest(compose_inputs)}"
                )
                final_path = await self._task(
                    compose_id, "composition", None, compose_inputs,
                    lambda: asyncio.to_thread(
                        concatenate, ordered_videos, run_dir / "final.mp4", **fade_kwargs,
                    ),
                    dependencies=[
                        f"compose-{scene.scene_id}" if scene.audio_path else f"render-{scene.scene_id}"
                        for scene in successful
                    ] + [str(clip.detail.get("task_id")) for _, clip in bridge_clips.values()],
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
                if bridge_clips:
                    # Presenter clips are checked like scenes (streams, length,
                    # loudness), in the order the final video plays them.
                    clips = _timeline_clips(clips, timeline, bridge_clips)
                report = await asyncio.to_thread(
                    run_delivery_gate, clips, Path(result.final_video_path),
                    max_repair_rounds=self.config.max_delivery_repair_rounds,
                    reconcat=lambda paths: concatenate(paths, run_dir / "final.mp4", **fade_kwargs),
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

    # ── presenter bridges ───────────────────────────────────────────────

    def _presenter_settings(self, run_dir: Path) -> PresenterSettings:
        # The presenter speaks in the narration's prebuilt Gemini voice; an
        # OpenAI narration voice has no Gemini twin, so the default is used.
        voice = DEFAULT_VOICE if self.config.voice in OPENAI_VOICES else self.config.voice
        return PresenterSettings(
            provider=self.config.presenter,
            profile_path=self.config.presenter_profile,
            avatar_image_url=self.config.avatar_image_url,
            max_presenter_seconds=self.config.max_presenter_seconds,
            voice=voice,
            persona=self.config.presenter_persona,
            cache_dir=run_dir / "presenter" / "cache",
        )

    async def _prepare_bridges(
        self,
        router: ProviderRouter,
        lesson: LessonPlan,
        narration: NarrationPlan,
        run_dir: Path,
    ) -> list[BridgeProposal]:
        """Propose (and, for a planned lesson, write) bridges; merge with the saved ones.

        The proposal is a checkpointed task, so a resume never pays for the
        written text twice. The saved file keeps the professor's approvals and
        edits. Retries and revisions reuse the saved bridges as they are.
        """
        existing = load_bridges(run_dir)
        if existing and self._retry_scene_ids is not None:
            return existing
        planned = bool(lesson.central_question.strip())

        async def propose() -> list[BridgeProposal]:
            proposals = propose_bridges(lesson, narration, planned=planned)
            if planned and proposals:
                proposals = await write_bridges(router, proposals, lesson, narration)
            return proposals

        data = await self._task(
            "bridges", "bridges", None,
            {"lesson": lesson.model_dump(mode="json"),
             "narration": narration.model_dump(mode="json"),
             "planned": planned, "prompt_version": BRIDGES_PROMPT_VERSION,
             "model": TEXT_MODEL if planned else None},
            propose,
            dependencies=["narration"],
        )
        proposed = [BridgeProposal.model_validate(item) for item in data]
        known = {bridge.bridge_id for bridge in existing}
        merged = merge_proposals(existing, proposed)
        save_bridges(run_dir, merged)
        new_ids = [bridge.bridge_id for bridge in merged if bridge.bridge_id not in known]
        if self.config.approve_bridges and new_ids:
            # Only new proposals: a bridge the professor rejected stays rejected.
            merged = approve_bridges(run_dir, new_ids)
        approved = sum(bridge.approved for bridge in merged)
        self.artifacts.event(
            "bridges_proposed", stage="bridges",
            message=(
                f"{len(merged)} presenter bridge(s) proposed, {approved} approved; "
                "approve with `pdf2manim bridges RUN_DIR --approve ...` and resume."
            ),
            bridges=len(merged), approved=approved, planned=planned,
        )
        return merged

    def _keep_hook(
        self, lesson: LessonPlan, narration: NarrationPlan, bridges: list[BridgeProposal],
    ) -> NarrationPlan:
        """The narration was written for a presenter who says the hook.

        Without an approved opening bridge, the hook becomes the first sentence
        of the first beat instead, so it is never lost. Idempotent: a reused
        storyboard that already starts with the hook is left alone.
        """
        hook = " ".join((lesson.hook or "").split())
        if not hook or any(b.kind == "opening" and b.approved for b in bridges):
            return narration
        first = narration.scenes[0]
        beat = first.beats[0]
        if beat.text.lstrip().startswith(hook):
            return narration
        sentence = hook if hook[-1] in ".!?" else f"{hook}."
        opened = first.model_copy(update={
            "narration": f"{sentence} {first.narration.lstrip()}",
            "beats": [beat.model_copy(update={"text": f"{sentence} {beat.text.lstrip()}"}),
                      *first.beats[1:]],
        })
        self.artifacts.event(
            "hook_moved_to_narration", stage="bridges", scene_id=first.scene_id,
            message="No approved opening bridge: the hook opens the first scene's narration.",
        )
        return narration.model_copy(update={"scenes": [opened, *narration.scenes[1:]]})

    def _refresh_first_scene(self, narration: NarrationPlan) -> None:
        """Rebuild the first scene when the hook moved in or out of its narration.

        Approving or rejecting the opening after a run changes what the first
        scene says; its recorded audio (and the visuals timed to it) must not
        be reused, since the store keeps a succeeded task whatever its inputs.
        """
        first = narration.scenes[0]
        recorded = self.store.successful_task_output(f"tts-{first.scene_id}")
        if recorded is None:
            return
        spoken = [beat.text for beat in NarratedScene.model_validate(recorded).beats]
        if spoken == [beat.text for beat in first.beats]:
            return
        self.store.prepare_scene_revision([first.scene_id], [first.scene_id])
        self.artifacts.event(
            "hook_placement_changed", stage="bridges", scene_id=first.scene_id,
            message="The opening bridge's approval changed; the first scene is narrated and rendered again.",
        )

    async def _render_bridges(
        self,
        bridges: list[BridgeProposal],
        audio: NarrationAudioService,
        lesson: LessonPlan,
        narration: NarrationPlan,
        run_dir: Path,
        scene_ids: set[str],
    ) -> dict[str, tuple[BridgeProposal, PresenterClip]]:
        """Render approved bridges one at a time within the presenter-seconds cap.

        A bridge that does not fit the cap or the dollar budget is skipped and
        the run continues. An approved opening that cannot be rendered becomes
        a title card over TTS of its text, so the hook is still spoken.
        """
        provider = self.presenter_provider
        settings = self.presenter_settings
        budget = PresenterBudget(self.config.max_presenter_seconds)
        clips: dict[str, tuple[BridgeProposal, PresenterClip]] = {}
        records: list[dict[str, Any]] = []
        for bridge in bridges:
            record: dict[str, Any] = {
                "bridge_id": bridge.bridge_id, "kind": bridge.kind, "text": bridge.text,
                "approved": bridge.approved, "before_scene_id": bridge.before_scene_id,
                "status": "not approved", "seconds": None, "cost_usd": 0.0, "reason": None,
            }
            records.append(record)
            if not bridge.approved:
                continue
            clip: Optional[PresenterClip] = None
            if bridge.before_scene_id is not None and bridge.before_scene_id not in scene_ids:
                # Its scene produced no video; the timeline would drop the clip.
                record.update(status="skipped", reason="scene_missing")
                self.artifacts.event(
                    "presenter_skipped", stage="presenter", scene_id=bridge.bridge_id,
                    message=f"{bridge.before_scene_id} has no video; {bridge.bridge_id} is not rendered.",
                    reason="scene_missing",
                )
                continue
            if bridge.kind == "opening" and _speaks_hook(lesson, narration):
                # A reused storyboard already opens with the hook (the opening
                # was approved after it was written): never say it twice.
                record.update(status="skipped", reason="hook_in_narration")
                self.artifacts.event(
                    "presenter_skipped", stage="presenter", scene_id=bridge.bridge_id,
                    message="The narration already opens with the hook; the opening is not rendered.",
                    reason="hook_in_narration",
                )
                continue
            if budget.would_exceed(bridge.estimated_seconds):
                record.update(status="skipped", reason="presenter_seconds")
                self.artifacts.event(
                    "presenter_skipped", stage="presenter", scene_id=bridge.bridge_id,
                    message=(
                        f"{bridge.bridge_id} ({bridge.estimated_seconds:.1f}s) would pass the "
                        f"{budget.max_seconds:g}s presenter cap; skipped."
                    ),
                    reason="presenter_seconds",
                )
            else:
                try:
                    clip = await self._presenter_clip(bridge, provider, settings, run_dir)
                    budget.add(clip.duration_seconds)
                    record.update(status="rendered", cost_usd=clip.detail.get("cost_usd", 0.0))
                except BudgetExceeded as exc:
                    record.update(status="skipped", reason="budget")
                    self.artifacts.event(
                        "presenter_skipped", stage="presenter", scene_id=bridge.bridge_id,
                        message=f"{bridge.bridge_id} does not fit the run budget; skipped. {exc}",
                        reason="budget",
                    )
                except PresenterRenderError as exc:
                    record.update(status="failed", reason=str(exc))
                    self.artifacts.event(
                        "presenter_failed", stage="presenter", scene_id=bridge.bridge_id,
                        message=f"{bridge.bridge_id} could not be rendered: {exc}",
                    )
            if clip is None and bridge.kind == "opening":
                try:
                    clip = await self._title_card_bridge(
                        bridge, audio, lesson, narration, settings, run_dir,
                    )
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    record.update(status="failed", reason=f"title card: {exc}")
                    self.artifacts.event(
                        "presenter_failed", stage="presenter", scene_id=bridge.bridge_id,
                        message=f"The opening title card could not be made: {exc}",
                    )
                else:
                    record["status"] = "title card"
                    self.artifacts.event(
                        "presenter_title_card", stage="presenter", scene_id=bridge.bridge_id,
                        message="The opening is shown as a title card over TTS of its text.",
                    )
            if clip is not None:
                record["seconds"] = round(clip.duration_seconds, 3)
                clips[bridge.bridge_id] = (bridge, clip)
        self.artifacts.write_json(PRESENTER_RENDER_FILE, {
            "provider": provider.name,
            "voice": settings.voice if provider.name == "gemini" else settings.voice_id,
            "max_presenter_seconds": budget.max_seconds,
            "presenter_seconds": round(budget.spent_seconds, 3),
            "bridges": records,
        })
        return clips

    async def _presenter_clip(
        self,
        bridge: BridgeProposal,
        provider: Any,
        settings: PresenterSettings,
        run_dir: Path,
    ) -> PresenterClip:
        """One paid presenter clip, reserved before the call and settled after.

        The task input is the bridge's fingerprint: an unchanged bridge is
        reused on resume or revision, a changed text renders again.
        """
        # The run store reuses a succeeded task id whatever its inputs, so the
        # fingerprint is also part of the id and the file name: a changed text
        # is a new task and a new clip, and an unchanged one is reused.
        fingerprint = bridge_fingerprint(bridge, settings)
        digest = _digest(fingerprint)
        task_id = f"presenter-{bridge.bridge_id}-{digest}"
        request = PresenterRequest(
            bridge_id=bridge.bridge_id, text=bridge.text, audio_path=None,
            duration_seconds=bridge.estimated_seconds,
        )
        output = run_dir / "presenter" / "clips" / f"{bridge.bridge_id}-{digest}.mp4"
        vendor = "google" if provider.name == "gemini" else provider.name
        model = PRESENTER_MODEL if provider.name == "gemini" else provider.name

        async def operation() -> dict[str, Any]:
            estimate = float(provider.estimated_cost_usd(request))
            call_id = None
            if estimate > 0:
                call_id = f"presenter-{uuid.uuid4().hex[:12]}"
                # Raises BudgetExceeded before anything is requested.
                self.store.reserve_call(
                    call_id, task_id=task_id, scene_id=bridge.bridge_id, stage="presenter",
                    provider=vendor, model=model, amount_usd=estimate,
                )
                self.artifacts.event(
                    "api_call_started", stage="presenter", scene_id=bridge.bridge_id,
                    task_id=task_id, provider=vendor, model=model, call_id=call_id,
                    reserved_usd=estimate,
                )
            try:
                clip = await asyncio.to_thread(provider.render, request, output)
            except BaseException as exc:
                if call_id is not None:
                    # The vendor may have billed a failed take: keep the reservation.
                    self.store.settle_call(
                        call_id, actual_usd=estimate, input_units=0, output_units=0,
                        exact=False, status="FAILED_UNCERTAIN",
                    )
                    self.artifacts.event(
                        "api_call_failed", stage="presenter", scene_id=bridge.bridge_id,
                        task_id=task_id, provider=vendor, model=model, call_id=call_id,
                        message=str(exc),
                    )
                raise
            actual = 0.0
            if call_id is not None:
                actual = round(clip.duration_seconds * presenter_usd_per_second(provider.name), 6)
                self.store.settle_call(
                    call_id, actual_usd=actual, input_units=0,
                    output_units=round(clip.duration_seconds), exact=False,
                )
                self.artifacts.event(
                    "api_call_finished", stage="presenter", scene_id=bridge.bridge_id,
                    task_id=task_id, provider=vendor, model=model, call_id=call_id,
                    actual_usd=actual, clip_seconds=round(clip.duration_seconds, 3),
                )
            clip.detail.update(task_id=task_id, cost_usd=actual)
            return _clip_payload(clip)

        data = await self._task(
            task_id, "presenter", bridge.bridge_id, {"bridge": fingerprint}, operation,
            dependencies=["bridges"],
        )
        return _clip_from(data)

    async def _title_card_bridge(
        self,
        bridge: BridgeProposal,
        audio: NarrationAudioService,
        lesson: LessonPlan,
        narration: NarrationPlan,
        settings: PresenterSettings,
        run_dir: Path,
    ) -> PresenterClip:
        """The opening as a title card over TTS of its text, in the narration's voice."""
        fingerprint = bridge_fingerprint(bridge, settings)
        bridge_dir = run_dir / "presenter" / f"{bridge.bridge_id}-{_digest(fingerprint)}"
        audio_path: Optional[str] = None
        seconds = bridge.estimated_seconds
        silent = True
        tts_id = None
        if self.config.enable_tts:
            scene = bridge_scene(bridge, lesson, narration)
            tts_inputs = {
                "narration": scene.model_dump(mode="json"), "voice": self.config.voice,
                "stage_version": TTS_TIMING_VERSION, "quality": self.config.quality,
            }
            tts_id = f"tts-{bridge.bridge_id}-{_digest(tts_inputs)}"
            async with self._tts_limit:
                timed_data = await self._task(
                    tts_id, "tts", bridge.bridge_id, tts_inputs,
                    lambda: audio.generate(scene, bridge_dir),
                    dependencies=["bridges"],
                )
            timed = NarratedScene.model_validate(timed_data)
            audio_path = timed.audio_path
            seconds = timed.audio_duration_seconds or seconds
            silent = timed.audio_is_placeholder
        card = TitleCardPresenter(settings.width, settings.height, settings.fps)
        request = PresenterRequest(
            bridge_id=bridge.bridge_id, text=bridge.text,
            audio_path=Path(audio_path) if audio_path else None, duration_seconds=seconds,
        )
        card_inputs = {
            "bridge": fingerprint, "card": True,
            "audio": _file_signature(Path(audio_path)) if audio_path else None,
            "seconds": round(seconds, 3),
        }
        digest = _digest(card_inputs)
        output = run_dir / "presenter" / "clips" / f"{bridge.bridge_id}-card-{digest}.mp4"
        task_id = f"presenter-card-{bridge.bridge_id}-{digest}"

        def render() -> dict[str, Any]:
            clip = card.render(request, output)
            clip.detail.update(task_id=task_id, cost_usd=0.0, silent_narration=silent)
            return _clip_payload(clip)

        data = await self._task(
            task_id, "presenter", bridge.bridge_id, card_inputs,
            lambda: asyncio.to_thread(render),
            dependencies=[tts_id] if tts_id else ["bridges"],
        )
        return _clip_from(data)

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

    def _mode_inputs(self) -> dict[str, str]:
        """The video mode, for task inputs and calls that depend on it.

        Empty for the brief visualizer, so its task hashes and calls are the
        same as before modes existed.
        """

        return {"mode": "deep"} if self._mode() == "deep" else {}

    def _mode(self) -> str:
        return getattr(self.config, "mode", "brief")

    # Planning task inputs. Their hashes key the run's task cache, so a change
    # here re-plans existing runs on resume (tests/test_v2_prompt_versions.py).
    def _ocr_inputs(self, source: SourceDocument) -> dict[str, Any]:
        return {"source_sha256": source.sha256, "pages": source.low_confidence_pages,
                "prompt_version": prompt_version(self._mode(), "planning")}

    def _lesson_inputs(self, source: SourceDocument, controls: dict[str, Any]) -> dict[str, Any]:
        return {"source": source.model_dump(mode="json"),
                "prompt_version": prompt_version(self._mode(), "planning"),
                **self._mode_inputs(),
                "min_scenes": self.config.min_scenes,
                "max_scenes": self.config.max_scenes,
                "pacing": self.config.pacing,
                **controls,
                "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL}

    def _narration_inputs(
        self, lesson: LessonPlan, controls: dict[str, Any],
        review_feedback: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        inputs = {"lesson": lesson.model_dump(mode="json"),
                  "prompt_version": prompt_version(self._mode(), "narration"),
                  **self._mode_inputs(),
                  "pacing": self.config.pacing,
                  **controls,
                  "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                  "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL}
        if review_feedback is not None:
            inputs["review_feedback"] = review_feedback
        return inputs

    def _equivalent_inputs(self, inputs: dict[str, Any], kind: str) -> list[dict[str, Any]]:
        """``inputs`` under the other names of this mode's current prompt version."""

        return [{**inputs, "prompt_version": alias}
                for alias in prompt_version_aliases(self._mode(), kind)]

    async def _review_script(
        self,
        router: ProviderRouter,
        planner: PlanningService,
        lesson: LessonPlan,
        narration: NarrationPlan,
        source: SourceDocument,
        controls: dict[str, Any],
    ) -> tuple[NarrationPlan, Optional[list]]:
        """Review the planned script before TTS; in revise mode fix a failing one once.

        A script that fails on sentences that do not follow has only the beats
        holding them rewritten (_fix_flow_breaks); one that fails with no such
        sentence is rewritten whole. The new script is reviewed once and kept
        only if it passes or scores higher than the original.

        Returns the narration to use and, when a fixed or rewritten script
        replaced the original, its narration-directive checks (otherwise
        ``None``). The hook was settled before the narration was written, so a
        hook sentence the review flags is reported, never rewritten here: a new
        hook now could no longer match the narration that follows it.
        """

        async def review(
            candidate: NarrationPlan, attempt: int, dependencies: list[str],
        ) -> Optional[ScriptReview]:
            data = await self._task(
                f"script-review-{attempt}", "script_review", None,
                {"narration": candidate.model_dump(mode="json"),
                 "lesson": lesson.model_dump(mode="json"), "attempt": attempt,
                 "model": TEXT_MODEL, "prompt_version": SCRIPT_REVIEW_PROMPT_VERSION,
                 **self._mode_inputs()},
                lambda: review_script(
                    router, lesson, candidate, source, attempt=attempt, **self._mode_inputs(),
                ),
                dependencies=dependencies,
            )
            if not data:
                return None
            # Sentences that sound written rather than spoken fail the script too,
            # and go to the targeted beat fix like the reviewer's flow breaks.
            return with_voice_check(ScriptReview.model_validate(data), candidate)

        first = await review(narration, 1, ["narration"])
        if first is None:
            return narration, None
        if first.hook_breaks():
            self.artifacts.event(
                "hook_flow_reported", stage="script_review",
                message="The script review flagged hook sentences; the hook was settled earlier.",
                breaks=first.hook_breaks(),
            )
        attempts = [first]
        kept: tuple[NarrationPlan, Optional[list]] = (narration, None)
        fix: Optional[str] = None  # how a replacement script was made: "beats" or "rewrite"
        if self.config.script_review == "revise" and not first.passed:
            # A reviewer that failed the script on its own merits without pointing
            # at sentences gets the whole rewrite (its suggestions include the
            # voice check's); otherwise only the flagged beats are fixed.
            whole_rewrite = (
                first.reviewer_passed is False and not first.reviewer_breaks()
                and bool(first.improvement_suggestions)
            )
            if first.scene_breaks() and not whole_rewrite:
                fix = "beats"
                patched, fix_tasks = await self._fix_flow_breaks(
                    planner, lesson, narration, source, first.scene_breaks(),
                )
                if patched is not None:
                    patched, patched_checks = enforce_narration_directives(
                        patched, self.directives, planner.omitted_concepts,
                    )
                    recheck = await review(patched, 2, fix_tasks)
                    if recheck is not None:
                        attempts.append(recheck)
                        # first.passed is False, so this keeps a fix that passes or scores higher.
                        if (recheck.passed, recheck.overall_score) > (
                            first.passed, first.overall_score,
                        ):
                            kept = (patched, patched_checks)
            elif not first.improvement_suggestions:
                self.artifacts.event(
                    "script_review_rewrite_skipped", stage="script_review",
                    message="The review failed without improvement suggestions; kept the script.",
                    score=first.overall_score,
                )
            else:
                fix = "rewrite"
                planner_state = (
                    planner.length_budget, planner.length_report, planner.compliance_checks,
                )
                second: Optional[ScriptReview] = None
                try:
                    revised_inputs = self._narration_inputs(
                        lesson, controls, review_feedback=first.improvement_suggestions,
                    )
                    revised_data = await self._task(
                        "narration-revised", "narration", None, revised_inputs,
                        lambda: planner.narration(
                            lesson, source, review_feedback=first.improvement_suggestions,
                        ),
                        dependencies=["script-review-1"],
                        equivalent_inputs=self._equivalent_inputs(revised_inputs, "narration"),
                    )
                    revised = NarrationPlan.model_validate(revised_data)
                    grounding_errors = validate_grounding(lesson, revised, source)
                    if grounding_errors:
                        self.artifacts.event(
                            "script_review_rewrite_rejected", stage="script_review",
                            message="; ".join(grounding_errors),
                        )
                    else:
                        revised, revised_checks = enforce_narration_directives(
                            revised, self.directives, planner.omitted_concepts,
                        )
                        second = await review(revised, 2, ["narration-revised"])
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    self.artifacts.event(
                        "script_review_rewrite_rejected", stage="script_review",
                        message=f"The rewrite failed; kept the reviewed script: {exc}",
                    )
                if second is not None:
                    attempts.append(second)
                # A rewrite that passes beats one that does not; then the higher score wins.
                if second is not None and (
                    (second.passed, second.overall_score) > (first.passed, first.overall_score)
                ):
                    kept = (revised, revised_checks)
                else:
                    (planner.length_budget, planner.length_report,
                     planner.compliance_checks) = planner_state
        final = attempts[-1] if kept[1] is not None else first
        self.artifacts.write_json(
            "script_review.json", [item.model_dump(mode="json") for item in attempts],
        )
        self.artifacts.event(
            "script_reviewed", stage="script_review",
            message=(
                f"Script review scored {final.overall_score:.2f}/10 "
                f"({'passed' if final.passed else 'below the pass mark'}) after "
                f"{len(attempts)} review(s); "
                + (("the fixed script was kept." if fix == "beats"
                    else "the rewritten script was kept.") if kept[1] is not None
                   else "the original script was kept.")
            ),
            score=final.overall_score, passed=final.passed, attempts=len(attempts),
            rewritten=kept[1] is not None, fix=fix,
        )
        return kept

    async def _fix_flow_breaks(
        self,
        planner: PlanningService,
        lesson: LessonPlan,
        narration: NarrationPlan,
        source: SourceDocument,
        breaks: list[dict[str, str]],
    ) -> tuple[Optional[NarrationPlan], list[str]]:
        """Rewrite only the beats holding the review's flow breaks, into a copy of the narration.

        Each beat is asked for once, with every break it holds, and keeps its
        id and cited facts. A fix that is empty, unchanged or longer than the
        allowance is dropped and its beat keeps its text. A beat that opens with
        the hook is never rewritten: the hook was settled before the narration.
        Returns the fixed narration (``None`` when no beat changed) and the fix
        task ids, which the re-review depends on.
        """

        placed, unplaced = locate_flow_breaks(narration, breaks)
        if unplaced:
            self.artifacts.event(
                "flow_breaks_unplaced", stage="script_review",
                message=f"{len(unplaced)} flagged sentence(s) are in no beat; left as they are.",
                breaks=unplaced,
            )
        first_scene = narration.scenes[0]
        hook_beat = (
            (first_scene.scene_id, first_scene.beats[0].beat_id)
            if _speaks_hook(lesson, narration) else None
        )
        facts = {fact.fact_id: fact.text for fact in source.facts}
        positions = {scene.scene_id: index for index, scene in enumerate(narration.scenes)}
        texts: dict[tuple[str, str], str] = {}
        task_ids: list[str] = []
        fixes: list[dict[str, Any]] = []
        for (scene_id, beat_id), items in placed.items():
            if (scene_id, beat_id) == hook_beat:
                self.artifacts.event(
                    "beat_fix_skipped", stage="script_review", scene_id=scene_id,
                    message=f"{beat_id} opens with the settled hook; it is not rewritten.",
                    breaks=items,
                )
                continue
            index = positions[scene_id]
            scene = narration.scenes[index]
            beat = next(item for item in scene.beats if item.beat_id == beat_id)
            previous = narration.scenes[index - 1].beats[-1].text if index else None
            problems = [
                f'The sentence "{item["sentence"]}" does not follow: {item["problem"]}'
                for item in items
            ]
            inputs = {
                "scene": scene.model_dump(mode="json"), "beat_id": beat_id,
                "previous_beat": previous, "problems": problems,
                "facts": {
                    fact_id: facts[fact_id] for fact_id in beat.source_fact_ids
                    if fact_id in facts
                },
                "prompt_version": BEAT_FIX_PROMPT_VERSION,
                "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL,
            }
            task_id = f"beat-fix-{beat_id}-{_digest(inputs)}"

            async def operation(
                scene: NarratedScene = scene, beat_id: str = beat_id,
                previous: Optional[str] = previous, problems: list[str] = problems,
            ) -> dict[str, str]:
                return {"text": await planner.fix_beat(scene, beat_id, previous, problems, source)}

            try:
                data = await self._task(
                    task_id, "narration", scene_id, inputs, operation,
                    dependencies=["script-review-1"],
                )
            except BudgetExceeded:
                raise
            except Exception as exc:
                self.artifacts.event(
                    "beat_fix_rejected", stage="script_review", scene_id=scene_id,
                    message=f"The fix of {beat_id} failed; it keeps its text: {exc}",
                )
                continue
            text = " ".join(str(data.get("text") or "").split())
            reason = _beat_fix_problem(beat.text, text)
            if reason:
                self.artifacts.event(
                    "beat_fix_rejected", stage="script_review", scene_id=scene_id,
                    message=f"The fix of {beat_id} was dropped: {reason}.",
                    before=beat.text, after=text,
                )
                continue
            texts[(scene_id, beat_id)] = text
            task_ids.append(task_id)
            fixes.append({
                "scene_id": scene_id, "beat_id": beat_id, "problems": problems,
                "before": beat.text, "after": text,
            })
        if not texts:
            self.artifacts.event(
                "script_review_fix_skipped", stage="script_review",
                message="No flagged beat could be fixed; kept the reviewed script.",
            )
            return None, []
        self.artifacts.event(
            "flow_breaks_fixed", stage="script_review",
            message=f"Rewrote {len(texts)} beat(s) holding sentences that did not follow.",
            fixes=fixes,
        )
        return splice_beats(narration, texts), task_ids

    async def _settle_hook(self, planner: PlanningService, lesson: LessonPlan) -> LessonPlan:
        """Check the planned hook before any narration is written; rewrite it once if needed.

        The checkable shape (length, one closing question, no dashes, a chain
        of sentences) is always enforced. Whether each sentence follows is
        checked by a model when the script review is on; in report mode those
        findings are logged and the hook is kept.
        """

        if not (lesson.hook or "").strip():
            return lesson
        problems = hook_problems(lesson.hook, planner.presenter_opens)
        flow: list[str] = []
        if self.config.script_review != "off":
            async def check() -> dict[str, list[str]]:
                return {"problems": await planner.check_hook(lesson)}

            data = await self._task(
                "hook-check", "lesson", None,
                {"hook": lesson.hook, "central_question": lesson.central_question,
                 "running_example": lesson.running_example.model_dump(mode="json"),
                 "prompt_version": HOOK_PROMPT_VERSION,
                 "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                 "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL},
                check,
                dependencies=["lesson"],
            )
            flow = [str(item) for item in data.get("problems") or []]
            self.artifacts.event(
                "hook_checked", stage="lesson", task_id="hook-check",
                message=(
                    f"{len(flow)} hook sentence(s) do not follow."
                    if flow else "Every hook sentence follows from the one before."
                ),
                problems=flow,
            )
        if self.config.script_review != "revise":
            flow = []
        if not problems and not flow:
            return lesson
        return await self._rewrite_hook(
            planner, lesson, problems + flow, "hook-rewrite", prefer_rewrite=bool(flow),
        )

    async def _rewrite_hook(
        self,
        planner: PlanningService,
        lesson: LessonPlan,
        problems: list[str],
        task_id: str,
        *,
        prefer_rewrite: bool = False,
    ) -> LessonPlan:
        """Rewrite the hook to fix ``problems``; once more if the rewrite misses the shape.

        A hook is never trimmed to fit: cutting sentences out of a chain of
        reasoning is what makes it stop following, so it is rewritten whole.
        The second request only fixes the first rewrite's checkable shape
        (length, punctuation), keeping its reasoning. ``prefer_rewrite`` (the
        planned hook does not follow): any rewrite beats it, the one with the
        fewest shape problems winning; otherwise a rewrite must not fail more
        shape checks than the planned hook.
        """

        presenter = planner.presenter_opens
        original = len(hook_problems(lesson.hook, presenter))
        best: Optional[tuple[int, str]] = None
        attempts: list[dict[str, Any]] = []
        current, asked = lesson, list(problems)
        for attempt in range(1, HOOK_REWRITE_ATTEMPTS + 1):
            attempt_id = task_id if attempt == 1 else f"{task_id}-{attempt}"

            async def operation(plan: LessonPlan = current, todo: list[str] = asked) -> dict[str, str]:
                return {"hook": await planner.rewrite_hook(plan, todo)}

            data = await self._task(
                attempt_id, "lesson", None,
                {"lesson": current.model_dump(mode="json"), "problems": asked,
                 "presenter_opens": presenter,
                 "prompt_version": HOOK_PROMPT_VERSION,
                 "structured_output_version": STRUCTURED_OUTPUT_VERSION,
                 "provider": PLANNING_PROVIDER, "model": PLANNING_MODEL},
                operation,
                dependencies=["lesson"],
            )
            hook = " ".join(str(data.get("hook") or "").split())
            if not hook or hook == current.hook:
                attempts.append({"hook": hook, "problems": None})
                break
            remaining = hook_problems(hook, presenter)
            attempts.append({"hook": hook, "problems": remaining})
            if (prefer_rewrite or len(remaining) <= original) and (
                best is None or len(remaining) <= best[0]
            ):
                best = (len(remaining), hook)
            if not remaining:
                break
            current, asked = lesson.model_copy(update={"hook": hook}), remaining
        kept = best is not None
        self.artifacts.event(
            "hook_rewritten" if kept else "hook_rewrite_rejected", stage="lesson",
            task_id=task_id,
            message=(
                "Replaced the hook so each sentence follows from the one before."
                if kept else
                "Kept the planned hook: no rewrite was usable or passed more checks."
            ),
            problems=problems, before=lesson.hook, after=best[1] if kept else lesson.hook,
            attempts=attempts,
        )
        return lesson.model_copy(update={"hook": best[1]}) if kept else lesson

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
            # Added only when the lesson has a running example or motifs, so
            # runs without a spine keep their code task hashes and requests.
            world = lesson.visual_world()
            if world:
                code_payload["visual_world"] = world
            world_kwargs = {"visual_world": world} if world else {}
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
                        **world_kwargs,
                    )
                return generator.generate(
                    timed_scene, contract, facts, lesson_scene=lesson_scene,
                    visual_history=visual_history,
                    retry_attempt=retry_attempt,
                    **notation_kwargs,
                    **world_kwargs,
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
                                **world_kwargs,
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
        equivalent_inputs: Optional[list[Any]] = None,
    ) -> Any:
        """Run ``operation`` unless this task already succeeded with the same inputs.

        ``equivalent_inputs`` are older forms of the same inputs (a prompt
        version's other names): a stored output under one of them is reused,
        and the task is then recorded under ``input_payload``'s hash.
        """

        input_hash = _input_hash(input_payload)
        # Look up the cache before recording the new hash: upsert_task stores
        # input_hash, so checking afterwards always matched and a task whose
        # inputs had changed was served its old output.
        cached = self.store.task_output(task_id, input_hash)
        for equivalent in equivalent_inputs or []:
            if cached is not None:
                break
            cached = self.store.task_output(task_id, _input_hash(equivalent))
        if cached is None and task_id in REPLAN_TASKS:
            self._warn_replanning(task_id, stage, input_hash)
        self.store.upsert_task(task_id, stage, scene_id, input_hash, dependencies)
        worker = asyncio.current_task().get_name() if asyncio.current_task() else "main"
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

    def _warn_replanning(self, task_id: str, stage: str, input_hash: str) -> None:
        """Warn when a planning task that already succeeded is about to run again.

        Its inputs changed (usually a prompt version bump in the code), so the
        run re-plans and regenerates the video after it. The stored hash is
        overwritten once the task starts, so the event keeps the old one.
        """

        stored = self.store.task_state(task_id)
        if stored is None or stored["status"] != TaskStatus.SUCCEEDED.value:
            return
        status_before = getattr(self, "_status_before", None)
        self.artifacts.event(
            "replan_warning", stage=stage, task_id=task_id,
            message=(
                f"{task_id} already succeeded in this run (run status "
                f"{status_before or 'unknown'}) but its inputs have changed, so it "
                "runs again and everything planned after it is "
                "redone. To rebuild scenes of a finished run, use retry-scenes or revise, "
                "which keep its storyboard."
            ),
            previous_input_hash=stored["input_hash"], input_hash=input_hash,
            run_status_before=status_before,
        )

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


def _speaks_hook(lesson: LessonPlan, narration: NarrationPlan) -> bool:
    hook = " ".join((lesson.hook or "").split())
    first = narration.scenes[0].beats[0].text.lstrip() if narration.scenes else ""
    return bool(hook) and first.startswith(hook)


def _beat_fix_problem(before: str, after: str) -> Optional[str]:
    """Why a beat fix cannot replace ``before``; ``None`` when it can."""

    if not after:
        return "it is empty"
    if after == " ".join(before.split()):
        return "it left the beat unchanged"
    added = len(after.split()) - len(before.split())
    limit = BEAT_FIX_MAX_EXTRA_WORDS + BEAT_FIX_WORD_TOLERANCE
    if added > limit:
        return f"it adds {added} words, more than {limit}"
    return None


def _digest(payload: Any) -> str:
    """Short, stable hash of a task's inputs, for ids the store must not reuse."""
    return _hash_text(json.dumps(_serialize(payload), sort_keys=True, default=str))[:10]


def _clip_payload(clip: PresenterClip) -> dict[str, Any]:
    return {
        "path": str(Path(clip.path).resolve()),
        "duration_seconds": clip.duration_seconds,
        "synthetic": clip.synthetic,
        "provider": clip.provider,
        "detail": clip.detail,
    }


def _clip_from(data: dict[str, Any]) -> PresenterClip:
    return PresenterClip(
        path=Path(data["path"]), duration_seconds=float(data["duration_seconds"]),
        synthetic=bool(data["synthetic"]), provider=str(data["provider"]),
        detail=dict(data.get("detail") or {}),
    )


def _presenter_fades(timeline: list[TimelineSegment]) -> list[tuple[float, float]]:
    """(fade-in, fade-out) per segment: a fade through black at each presenter/Manim cut."""
    fades = []
    for index, segment in enumerate(timeline):
        def cut(other: int) -> bool:
            if not 0 <= other < len(timeline):
                return False
            return (segment.mode == "presenter") != (timeline[other].mode == "presenter")

        fades.append((
            PRESENTER_FADE_SECONDS if cut(index - 1) else 0.0,
            PRESENTER_FADE_SECONDS if cut(index + 1) else 0.0,
        ))
    return fades


def _timeline_clips(
    scene_clips: list[SceneClip],
    timeline: list[TimelineSegment],
    bridge_clips: dict[str, tuple[BridgeProposal, PresenterClip]],
) -> list[SceneClip]:
    """Scene and presenter clips for the delivery gate, in playing order."""
    by_scene = {clip.scene_id: clip for clip in scene_clips}
    ordered: list[SceneClip] = []
    for segment in timeline:
        if segment.mode == "presenter" and segment.scene_id in bridge_clips:
            clip = bridge_clips[segment.scene_id][1]
            ordered.append(SceneClip(
                scene_id=segment.scene_id, composed_path=Path(clip.path),
                narration_seconds=clip.duration_seconds,
                silent_narration=bool(clip.detail.get("silent_narration")),
                mode="presenter",
            ))
        elif segment.scene_id in by_scene:
            ordered.append(by_scene.pop(segment.scene_id))
    return ordered + list(by_scene.values())


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


def _input_hash(input_payload: Any) -> str:
    """The task cache key of a task's inputs."""

    return _hash_text(json.dumps(_serialize(input_payload), sort_keys=True, default=str))


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
