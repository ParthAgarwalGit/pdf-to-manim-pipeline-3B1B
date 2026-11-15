"""Resumable scene-level model evaluation using the V2 runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.pipeline_v2.diagnostics import issue_score
from src.pipeline_v2.generation import (
    class_name_for,
    clean_code,
    generation_request,
)
from src.pipeline_v2.instrumentation import deinstrument_scene_code, instrument_scene_code
from src.pipeline_v2.models import VerificationFinding
from src.pipeline_v2.observability import RunArtifacts
from src.pipeline_v2.providers import ProviderRouter, estimate_tokens, price
from src.pipeline_v2.rendering import RenderOutcome, TrackedManimRenderer
from src.pipeline_v2.state import RunStore

from .cases import load_benchmark
from .dashboard import generate_dashboard
from .models import (
    CandidateMetrics,
    CandidateResult,
    EvaluationCase,
    EvaluationRunConfig,
    EvaluationRunSummary,
    EvaluationTrack,
    ModelSpec,
)


MODEL_ALIASES = {
    "gemini": ModelSpec(alias="gemini", provider="google", model="gemini-3.7-flash"),
    "gemma": ModelSpec(alias="gemma", provider="google", model="gemma-4-26b-a4b-it"),
}


class EvaluationRunner:
    def __init__(
        self,
        config: EvaluationRunConfig,
        *,
        router_factory: Callable[[RunStore, RunArtifacts, int], ProviderRouter] | None = None,
        renderer: TrackedManimRenderer | None = None,
    ):
        self.config = config
        self.router_factory = router_factory or self._default_router
        self.renderer = renderer or TrackedManimRenderer()
        self._summary_lock = asyncio.Lock()

    def run(self, run_directory: Path | None = None) -> EvaluationRunSummary:
        manifest, cases = load_benchmark(self.config.benchmark_path)
        if not cases:
            raise ValueError("The benchmark contains no cases")
        evaluation_id = run_directory.name if run_directory else _evaluation_id()
        run_dir = (
            run_directory.resolve()
            if run_directory
            else (self.config.output_root / evaluation_id).resolve()
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = RunArtifacts(run_dir, evaluation_id)
        store = RunStore(run_dir / "run.db", self.config.max_cost_usd)
        store.max_cost_usd = self.config.max_cost_usd
        store.set_meta("max_cost_usd", self.config.max_cost_usd)
        router = self.router_factory(store, artifacts, self.config.workers)
        expected = len(cases) * len(self.config.models) * self.config.replicates
        summary = self._load_or_create_summary(
            run_dir, evaluation_id, manifest.benchmark_hash, expected,
        )
        artifacts.event(
            "evaluation_started", stage="evaluation", message=manifest.benchmark_id,
            candidates=expected, track=self.config.track.value,
        )
        try:
            asyncio.run(self._run_all(cases, run_dir, router, store, artifacts, summary))
        except KeyboardInterrupt:
            self.renderer.terminate_all()
            store.interrupt_running()
            summary.status = "INTERRUPTED"
            artifacts.event("evaluation_interrupted", stage="evaluation")
        finally:
            self._refresh_summary(summary, run_dir, store)
            if summary.status == "RUNNING":
                summary.status = (
                    "SUCCESS"
                    if summary.completed_candidates == summary.total_candidates
                    else "PARTIAL_SUCCESS"
                    if summary.candidates
                    else "FAILED"
                )
            summary.dashboard_path = str(generate_dashboard(run_dir, summary).resolve())
            artifacts.write_json("costs.json", {
                "total_usd": store.total_cost(),
                "calls": store.cost_rows(),
            })
            self._write_summary(run_dir, summary)
            store.close()
        return summary

    @classmethod
    def resume(
        cls,
        run_directory: Path,
        *,
        max_cost_usd: float | None = None,
        retry_interrupted: bool = False,
    ) -> EvaluationRunSummary:
        run_directory = run_directory.resolve()
        payload = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        config = EvaluationRunConfig.model_validate(payload["config"])
        if max_cost_usd is not None:
            config.max_cost_usd = max_cost_usd
        if retry_interrupted:
            for path in run_directory.glob("candidates/*/candidate.json"):
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if candidate.get("status") == "INTERRUPTED":
                    path.unlink()
        return cls(config).run(run_directory)

    async def _run_all(
        self,
        cases: list[EvaluationCase],
        run_dir: Path,
        router: ProviderRouter,
        store: RunStore,
        artifacts: RunArtifacts,
        summary: EvaluationRunSummary,
    ) -> None:
        semaphore = asyncio.Semaphore(self.config.workers)

        async def bounded(case: EvaluationCase, model: ModelSpec, replicate: int) -> None:
            async with semaphore:
                await self._evaluate(case, model, replicate, run_dir, router, store, artifacts)
                async with self._summary_lock:
                    self._refresh_summary(summary, run_dir, store)
                    self._write_summary(run_dir, summary)

        tasks = [
            asyncio.create_task(bounded(case, model, replicate))
            for case in cases
            for model in self.config.models
            for replicate in range(1, self.config.replicates + 1)
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise

    async def _evaluate(
        self,
        case: EvaluationCase,
        model: ModelSpec,
        replicate: int,
        run_dir: Path,
        router: ProviderRouter,
        store: RunStore,
        artifacts: RunArtifacts,
    ) -> None:
        candidate_id = _candidate_id(case.case_id, model, replicate, self.config.track)
        candidate_dir = run_dir / "candidates" / candidate_id
        result_path = candidate_dir / "candidate.json"
        if result_path.exists():
            return
        candidate_dir.mkdir(parents=True, exist_ok=True)
        task_id = f"eval-{candidate_id}"
        store.upsert_task(task_id, self.config.track.value, case.source_scene_id, case.content_hash)
        store.start_task(task_id)
        started = time.perf_counter()
        artifacts.write_json(
            candidate_dir.relative_to(run_dir) / "input.json",
            case.model_dump(mode="json"),
        )
        try:
            result = await self._execute_track(
                case, model, replicate, candidate_id, candidate_dir, router, store, artifacts,
            )
            elapsed = (time.perf_counter() - started) * 1000
            store.finish_task(task_id, result.model_dump(mode="json"), elapsed)
        except asyncio.CancelledError:
            result = self._failed_result(
                case, model, replicate, candidate_id, "Evaluation was interrupted.", "INTERRUPTED",
            )
            store.fail_task(task_id, result.error or "interrupted", 0.0)
            _write_result(result_path, result)
            raise
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            result = self._failed_result(case, model, replicate, candidate_id, str(exc))
            store.fail_task(task_id, str(exc), elapsed)
            artifacts.event(
                "candidate_failed", stage=self.config.track.value,
                scene_id=case.source_scene_id, task_id=task_id, message=str(exc),
                model=model.model,
            )
        _apply_costs(result, store, candidate_id)
        _write_result(result_path, result)
        artifacts.event(
            "candidate_finished", stage=self.config.track.value,
            scene_id=case.source_scene_id, task_id=task_id,
            status=result.status, model=model.model,
            issue_score=result.metrics.issue_score,
        )

    async def _execute_track(
        self,
        case: EvaluationCase,
        model: ModelSpec,
        replicate: int,
        candidate_id: str,
        candidate_dir: Path,
        router: ProviderRouter,
        store: RunStore,
        artifacts: RunArtifacts,
    ) -> CandidateResult:
        generation_ms = 0.0
        if self.config.track == EvaluationTrack.REPAIR:
            code = _read_required(case.initial_code_path, "repair case initial code")
            (candidate_dir / "original.py").write_text(code, encoding="utf-8")
            start = time.perf_counter()
            code = await _repair(
                router, case, model, candidate_id, code,
                case.initial_findings, case.initial_frame_paths,
                candidate_dir, artifacts,
            )
            generation_ms = (time.perf_counter() - start) * 1000
            code = _instrument(code, case)
            outcome, render_ms = await self._render(code, case, candidate_dir / "repair-01")
            return self._result(
                case, model, replicate, candidate_id, code, outcome, candidate_dir,
                generation_ms, render_ms, repaired=True,
                repair_accepted=outcome.safe and outcome.render_success,
                original_code_path=str((candidate_dir / "original.py").resolve()),
            )

        start = time.perf_counter()
        code = await _generate(
            router, case, model, candidate_id, candidate_dir, artifacts,
        )
        generation_ms = (time.perf_counter() - start) * 1000
        code = _instrument(code, case)
        original_path = candidate_dir / "original.py"
        original_path.write_text(code, encoding="utf-8")
        original, render_ms = await self._render(code, case, candidate_dir / "attempt-00")
        selected_code = code
        selected = original
        repaired = False
        accepted = False
        if (
            self.config.track == EvaluationTrack.PRODUCTION
            and self.config.max_repairs
            and (not original.render_success or issue_score(original.findings) > 0)
        ):
            try:
                start = time.perf_counter()
                repair_code = await _repair(
                    router, case, model, candidate_id, code,
                    original.findings, original.frame_paths,
                    candidate_dir, artifacts,
                )
                generation_ms += (time.perf_counter() - start) * 1000
                repair_code = _instrument(repair_code, case)
                repaired_outcome, repair_render_ms = await self._render(
                    repair_code, case, candidate_dir / "repair-01",
                )
                render_ms += repair_render_ms
                repaired = True
                if _repair_improves(original, repaired_outcome):
                    selected_code = repair_code
                    selected = repaired_outcome
                    accepted = True
            except Exception as exc:
                original.findings.append(VerificationFinding(
                    code="EVAL_REPAIR_FAILED",
                    severity="warning",
                    message=f"Evaluation retained the original after repair failed: {exc}",
                ))
        return self._result(
            case, model, replicate, candidate_id, selected_code, selected, candidate_dir,
            generation_ms, render_ms, repaired=repaired, repair_accepted=accepted,
            original_code_path=str(original_path.resolve()),
        )

    async def _render(
        self, code: str, case: EvaluationCase, attempt_dir: Path,
    ) -> tuple[RenderOutcome, float]:
        start = time.perf_counter()
        outcome = await asyncio.to_thread(
            self.renderer.render,
            code,
            case.visual_contract,
            case.narration,
            attempt_dir,
            quality=self.config.quality,
            timeout_seconds=self.config.timeout_seconds,
        )
        return outcome, (time.perf_counter() - start) * 1000

    def _result(
        self,
        case: EvaluationCase,
        model: ModelSpec,
        replicate: int,
        candidate_id: str,
        code: str,
        outcome: RenderOutcome,
        candidate_dir: Path,
        generation_ms: float,
        render_ms: float,
        *,
        repaired: bool,
        repair_accepted: bool,
        original_code_path: str | None,
    ) -> CandidateResult:
        code_path = candidate_dir / "final.py"
        code_path.write_text(code, encoding="utf-8")
        diagnostics_path = candidate_dir / "diagnostics.json"
        diagnostics_path.write_text(
            json.dumps({
                "safe": outcome.safe,
                "render_success": outcome.render_success,
                "error": outcome.error,
                "findings": [item.model_dump(mode="json") for item in outcome.findings],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metrics = _metrics(case, outcome, generation_ms, render_ms)
        return CandidateResult(
            candidate_id=candidate_id,
            case_id=case.case_id,
            track=self.config.track,
            model=model,
            replicate=replicate,
            status="SUCCEEDED" if outcome.render_success else "FAILED",
            case_hash=case.content_hash,
            code_path=str(code_path.resolve()),
            original_code_path=original_code_path,
            video_path=outcome.video_path,
            frame_paths=outcome.frame_paths,
            diagnostics_path=str(diagnostics_path.resolve()),
            input_path=str((candidate_dir / "input.json").resolve()),
            request_paths=[str(path.resolve()) for path in sorted(candidate_dir.glob("*-request.json"))],
            response_paths=[str(path.resolve()) for path in sorted(candidate_dir.glob("*-response.txt"))],
            findings=outcome.findings,
            metrics=metrics,
            repaired=repaired,
            repair_accepted=repair_accepted,
            error=outcome.error,
        )

    def _failed_result(
        self,
        case: EvaluationCase,
        model: ModelSpec,
        replicate: int,
        candidate_id: str,
        error: str,
        status: str = "FAILED",
    ) -> CandidateResult:
        return CandidateResult(
            candidate_id=candidate_id,
            case_id=case.case_id,
            track=self.config.track,
            model=model,
            replicate=replicate,
            status=status,
            case_hash=case.content_hash,
            error=error,
        )

    def _load_or_create_summary(
        self, run_dir: Path, evaluation_id: str, benchmark_hash: str, expected: int,
    ) -> EvaluationRunSummary:
        path = run_dir / "run.json"
        if path.exists():
            summary = EvaluationRunSummary.model_validate_json(path.read_text(encoding="utf-8"))
            summary.config = self.config
            summary.status = "RUNNING"
            return summary
        summary = EvaluationRunSummary(
            evaluation_id=evaluation_id,
            run_directory=str(run_dir),
            status="RUNNING",
            config=self.config,
            benchmark_hash=benchmark_hash,
            total_candidates=expected,
        )
        self._write_summary(run_dir, summary)
        return summary

    @staticmethod
    def _refresh_summary(
        summary: EvaluationRunSummary, run_dir: Path, store: RunStore,
    ) -> None:
        candidates = []
        for path in sorted(run_dir.glob("candidates/*/candidate.json")):
            try:
                candidates.append(CandidateResult.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        summary.candidates = candidates
        summary.completed_candidates = sum(item.status == "SUCCEEDED" for item in candidates)
        summary.failed_candidates = sum(item.status == "FAILED" for item in candidates)
        summary.total_cost_usd = store.total_cost()

    @staticmethod
    def _write_summary(run_dir: Path, summary: EvaluationRunSummary) -> None:
        (run_dir / "run.json").write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _default_router(store: RunStore, artifacts: RunArtifacts, workers: int) -> ProviderRouter:
        return ProviderRouter(
            store, artifacts, max_api_workers=workers, strict_keys=True,
        )


async def _generate(
    router: ProviderRouter,
    case: EvaluationCase,
    model: ModelSpec,
    candidate_id: str,
    candidate_dir: Path,
    artifacts: RunArtifacts,
) -> str:
    system, prompt = generation_request(
        scene=case.narration,
        contract=case.visual_contract,
        source_facts=case.source_facts,
        class_name=class_name_for(case.source_scene_id),
        lesson_scene=case.lesson,
        visual_history=case.visual_history,
    )
    artifacts.write_json(candidate_dir.relative_to(artifacts.run_dir) / "generation-request.json", {
        "provider": model.provider, "model": model.model, "system": system, "prompt": prompt,
    })
    response = await router.text(
        stage="eval_generation",
        task_id=f"{candidate_id}-generate",
        scene_id=case.source_scene_id,
        provider=model.provider,
        model=model.model,
        system=system,
        prompt=prompt,
        max_output_tokens=9000,
    )
    artifacts.write_text(
        candidate_dir.relative_to(artifacts.run_dir) / "generation-response.txt",
        response.text,
    )
    return clean_code(response.text)


async def _repair(
    router: ProviderRouter,
    case: EvaluationCase,
    model: ModelSpec,
    candidate_id: str,
    code: str,
    findings: list[VerificationFinding],
    frame_paths: list[str],
    candidate_dir: Path,
    artifacts: RunArtifacts,
) -> str:
    if model.provider != "google":
        raise ValueError("Vision repair evaluation currently requires a Google-hosted model")
    class_name = class_name_for(case.source_scene_id)
    system = (
        "You repair one self-contained Manim Community Python animation from rendered evidence. "
        f"Return one complete executable Python script only, with exactly one Scene named {class_name}. "
        "Preserve correct mathematics and all self.next_section beat markers. Make minimal changes."
    )
    prompt = json.dumps({
        "lesson": case.lesson.model_dump(mode="json"),
        "narration": case.narration.model_dump(mode="json"),
        "expected_facts": case.expected_facts,
        "required_visuals": case.required_visuals,
        "findings": [item.model_dump(mode="json") for item in findings[:12]],
        "existing_code": deinstrument_scene_code(code),
    }, ensure_ascii=False)
    artifacts.write_json(candidate_dir.relative_to(artifacts.run_dir) / "repair-request.json", {
        "provider": model.provider, "model": model.model, "system": system,
        "prompt": prompt, "image_paths": frame_paths[:8],
    })
    response = await router.vision(
        stage="eval_repair",
        task_id=f"{candidate_id}-repair",
        scene_id=case.source_scene_id,
        model=model.model,
        system=system,
        prompt=prompt,
        image_paths=[Path(path) for path in frame_paths[:8]],
        max_output_tokens=9000,
    )
    artifacts.write_text(
        candidate_dir.relative_to(artifacts.run_dir) / "repair-response.txt",
        response.text,
    )
    return clean_code(response.text)


def _instrument(code: str, case: EvaluationCase) -> str:
    return instrument_scene_code(
        code,
        class_name=class_name_for(case.source_scene_id),
        scene=case.narration,
        contract=case.visual_contract,
    ).code


def _repair_improves(original: RenderOutcome, repaired: RenderOutcome) -> bool:
    if not repaired.render_success or not repaired.safe:
        return False
    if not original.render_success:
        return True
    return issue_score(repaired.findings) < issue_score(original.findings)


def _metrics(
    case: EvaluationCase,
    outcome: RenderOutcome,
    generation_ms: float,
    render_ms: float,
) -> CandidateMetrics:
    counts: dict[str, int] = {}
    for finding in outcome.findings:
        counts[finding.code] = counts.get(finding.code, 0) + 1
    errors = sum(item.severity == "error" for item in outcome.findings)
    warnings = sum(item.severity == "warning" for item in outcome.findings)
    target = float(case.narration.audio_duration_seconds or 0.0)
    return CandidateMetrics(
        static_safe=outcome.safe,
        render_success=outcome.render_success,
        production_ready=outcome.safe and outcome.render_success and errors == 0,
        error_count=errors,
        warning_count=warnings,
        issue_score=issue_score(outcome.findings),
        findings_by_code=counts,
        target_duration_seconds=target,
        rendered_duration_seconds=outcome.duration_seconds,
        duration_delta_seconds=abs(outcome.duration_seconds - target),
        generation_ms=generation_ms,
        render_ms=render_ms,
    )


def _apply_costs(result: CandidateResult, store: RunStore, candidate_id: str) -> None:
    rows = [
        row for row in store.cost_rows()
        if str(row.get("task_id") or "").startswith(candidate_id)
    ]
    result.metrics.cost_usd = sum(
        float(row.get("actual_usd") if row.get("actual_usd") is not None else row.get("reserved_usd") or 0)
        for row in rows
    )
    result.metrics.input_units = sum(int(row.get("input_units") or 0) for row in rows)
    result.metrics.output_units = sum(int(row.get("output_units") or 0) for row in rows)


def _candidate_id(
    case_id: str, model: ModelSpec, replicate: int, track: EvaluationTrack,
) -> str:
    digest = hashlib.sha256(
        f"{case_id}|{model.provider}|{model.model}|{replicate}|{track.value}".encode("utf-8")
    ).hexdigest()[:8]
    safe_alias = "".join(char if char.isalnum() or char in "-_" else "_" for char in model.alias)
    return f"{case_id}--{safe_alias}--r{replicate:02d}--{digest}"


def parse_model_spec(value: str) -> ModelSpec:
    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value].model_copy(deep=True)
    alias = value
    target = value
    if "=" in value:
        alias, target = value.split("=", 1)
    if ":" not in target:
        raise ValueError(
            f"Invalid model {value!r}; use an alias or alias=provider:model-name"
        )
    provider, model = target.split(":", 1)
    return ModelSpec(alias=alias, provider=provider, model=model)


def estimate_max_cost(
    cases: list[EvaluationCase],
    models: list[ModelSpec],
    track: EvaluationTrack,
    replicates: int,
) -> float:
    """Conservative reservation estimate using the active provider price table."""

    total = 0.0
    for case in cases:
        for model in models:
            if track != EvaluationTrack.REPAIR:
                system, prompt = generation_request(
                    scene=case.narration,
                    contract=case.visual_contract,
                    source_facts=case.source_facts,
                    class_name=class_name_for(case.source_scene_id),
                    lesson_scene=case.lesson,
                    visual_history=case.visual_history,
                )
                total += price(
                    model.provider, model.model, estimate_tokens(system + prompt), 9000,
                )
            if track in {EvaluationTrack.REPAIR, EvaluationTrack.PRODUCTION}:
                # Repair includes generated code, diagnostics, and up to eight preview images.
                text_units = estimate_tokens(
                    json.dumps(case.model_dump(mode="json"), ensure_ascii=False)
                ) + 9000
                image_units = 1500 * 8
                total += price(model.provider, model.model, text_units + image_units, 9000)
    return total * replicates


def _read_required(value: str | None, label: str) -> str:
    if not value or not Path(value).is_file():
        raise ValueError(f"Missing {label}")
    return Path(value).read_text(encoding="utf-8")


def _write_result(path: Path, result: CandidateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _evaluation_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
