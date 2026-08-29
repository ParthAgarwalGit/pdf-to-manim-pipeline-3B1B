"""Fixed orchestration pipeline for previewing and repairing raw Manim code."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

from src.manim_verifier.frame_extractor import FrameExtractor
from src.manim_verifier.models import (
    PipelineAttempt,
    PipelineResult,
    VerificationIssue,
    VerificationReport,
)
from src.manim_verifier.preview_renderer import RawManimPreviewRenderer
from src.manim_verifier.repair import normalize_complete_script
from src.manim_verifier.vision_reviewer import VisionReviewer
from src.utils.frame_analysis import analyze_keyframes


@dataclass
class VerifierConfig:
    mode: Literal["off", "shadow", "repair"] = "off"
    run_root: Path = Path("verifier_runs")
    preview_quality: str = "low"
    render_timeout_seconds: int = 120
    max_repair_attempts: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "repair"}:
            raise ValueError("Verifier mode must be off, shadow, or repair.")
        if not 0 <= self.max_repair_attempts <= 2:
            raise ValueError("max_repair_attempts must be between 0 and 2.")


class RawManimVerifierPipeline:
    """Preview, inspect, and optionally repair one complete Manim script."""

    def __init__(
        self,
        reviewer: Optional[VisionReviewer] = None,
        config: Optional[VerifierConfig] = None,
        renderer: Optional[RawManimPreviewRenderer] = None,
        frame_extractor: Optional[FrameExtractor] = None,
    ):
        self.config = config or VerifierConfig()
        self.reviewer = reviewer
        self.renderer = renderer or RawManimPreviewRenderer()
        self.frame_extractor = frame_extractor or FrameExtractor()
        if self.config.mode == "repair" and self.reviewer is None:
            raise ValueError("repair mode requires a VisionReviewer.")

    def run(self, *, brief: str, storyboard: str, initial_code: str) -> PipelineResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(
                brief=brief,
                storyboard=storyboard,
                initial_code=initial_code,
            ))
        raise RuntimeError("Use await run_async(...) from an active asyncio event loop.")

    async def run_async(
        self,
        *,
        brief: str,
        storyboard: str,
        initial_code: str,
    ) -> PipelineResult:
        start = time.monotonic()
        if self.config.mode == "off":
            return PipelineResult(
                success=True,
                original_code=initial_code,
                final_code=initial_code,
                elapsed_seconds=time.monotonic() - start,
            )

        run_dir = self._new_run_dir()
        original_path = run_dir / "original.py"
        original_path.write_text(initial_code, encoding="utf-8")

        attempts: list[PipelineAttempt] = []
        original_report = self._execute_attempt(initial_code, run_dir, 0)
        attempts.append(self._attempt_record(
            run_dir,
            0,
            original_report,
            code=initial_code,
        ))
        best_code = initial_code
        best_report = original_report
        total_input_tokens = 0
        total_output_tokens = 0
        repairs_called = 0

        if self.config.mode == "repair":
            while (
                repairs_called < self.config.max_repair_attempts
                and self._needs_repair(best_report)
            ):
                repairs_called += 1
                repair = await self.reviewer.repair(
                    brief=brief,
                    storyboard=storyboard,
                    code=best_code,
                    report=best_report,
                    frame_paths=[Path(path) for path in best_report.frame_paths],
                )
                total_input_tokens += repair.input_tokens or 0
                total_output_tokens += repair.output_tokens or 0
                try:
                    candidate_code = normalize_complete_script(repair.code)
                except ValueError as exc:
                    candidate_code = repair.code
                    candidate_report = VerificationReport(
                        safe=False,
                        syntax_valid=False,
                        issues=[VerificationIssue(
                            code="REPAIR_NOT_COMPLETE_SCRIPT",
                            severity="error",
                            message=str(exc),
                        )],
                        render_error=str(exc),
                    )
                    attempt_number = repairs_called
                    self._write_attempt_report(run_dir, attempt_number, candidate_report)
                else:
                    attempt_number = repairs_called
                    candidate_report = self._execute_attempt(
                        candidate_code,
                        run_dir,
                        attempt_number,
                    )
                attempts.append(self._attempt_record(
                    run_dir,
                    attempt_number,
                    candidate_report,
                    code=candidate_code,
                ))

                if self._quality_score(candidate_report) < self._quality_score(best_report):
                    best_code = candidate_code
                    best_report = candidate_report

        final_code_path = run_dir / "final.py"
        final_code_path.write_text(best_code, encoding="utf-8")
        final_video_path: Optional[str] = None
        if best_report.render_success and best_report.video_path:
            final_video = run_dir / "final.mp4"
            shutil.copy2(best_report.video_path, final_video)
            final_video_path = str(final_video.resolve())

        result = PipelineResult(
            success=bool(best_report.safe and best_report.render_success),
            original_code=initial_code,
            final_code=best_code,
            final_video_path=final_video_path,
            attempts=attempts,
            repaired=best_code != initial_code,
            fallback_used=repairs_called > 0 and best_code == initial_code,
            run_directory=str(run_dir.resolve()),
            elapsed_seconds=time.monotonic() - start,
            input_tokens=total_input_tokens or None,
            output_tokens=total_output_tokens or None,
        )
        (run_dir / "run.json").write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        return result

    def _execute_attempt(
        self,
        code: str,
        run_dir: Path,
        attempt_number: int,
    ) -> VerificationReport:
        attempt_dir = run_dir / f"attempt-{attempt_number:02d}"
        report = self.renderer.render(
            code,
            attempt_dir,
            quality=self.config.preview_quality,
            timeout_seconds=self.config.render_timeout_seconds,
        )
        if report.render_success and report.video_path:
            extraction = self.frame_extractor.extract(
                Path(report.video_path),
                attempt_dir / "frames",
            )
            report.frame_paths = extraction.frame_paths
            report.issues.extend(extraction.issues)
            report.issues.extend(self._visual_diagnostics(report.frame_paths))
        self._write_attempt_report(run_dir, attempt_number, report)
        return report

    @staticmethod
    def _visual_diagnostics(frame_paths: list[str]) -> list[VerificationIssue]:
        if not frame_paths:
            return []
        images = []
        try:
            for path in frame_paths:
                with Image.open(path) as image:
                    images.append(image.copy())
        except Exception as exc:
            return [VerificationIssue(
                code="FRAME_ANALYSIS_FAILED",
                severity="warning",
                message=f"Could not analyze preview screenshots: {exc}",
            )]

        # Thin Manim strokes occupy far less area than slides or raster media.
        # A lower threshold still catches blank backgrounds without rejecting a
        # correctly rendered graph, line, or circle.
        analysis = analyze_keyframes(
            images,
            empty_fraction=0.0002,
            static_mean_diff=0.02,
        )
        issues: list[VerificationIssue] = []
        if analysis.mostly_empty:
            index = analysis.empty_frame_indices[0]
            issues.append(VerificationIssue(
                code="VISUAL_MOSTLY_EMPTY",
                severity="error",
                message="At least half of the sampled preview frames are effectively empty.",
                screenshot_path=frame_paths[index],
            ))
        elif analysis.empty_frame_indices:
            for index in analysis.empty_frame_indices:
                issues.append(VerificationIssue(
                    code="VISUAL_EMPTY_FRAME",
                    severity="warning",
                    message=f"Sampled preview frame {index} is effectively empty.",
                    screenshot_path=frame_paths[index],
                ))
        if analysis.edge_clipped:
            for index in analysis.edge_clipped_indices:
                issues.append(VerificationIssue(
                    code="VISUAL_EDGE_CLIPPED",
                    severity="warning",
                    message=f"Sampled preview frame {index} concentrates content at the frame boundary.",
                    screenshot_path=frame_paths[index],
                ))
        if analysis.is_static and not analysis.mostly_empty:
            issues.append(VerificationIssue(
                code="VISUAL_STATIC",
                severity="warning",
                message="All sampled preview frames are nearly identical.",
                screenshot_path=frame_paths[0],
            ))
        return issues

    @staticmethod
    def _needs_repair(report: VerificationReport) -> bool:
        return (
            not report.safe
            or not report.render_success
            or bool(report.issues)
        )

    @staticmethod
    def _quality_score(report: VerificationReport) -> tuple[int, int, int]:
        return (
            0 if report.render_success and report.safe else 1,
            report.error_count,
            report.warning_count,
        )

    def _new_run_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        run_dir = self.config.run_root.resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def _write_attempt_report(
        run_dir: Path,
        attempt_number: int,
        report: VerificationReport,
    ) -> None:
        attempt_dir = run_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _attempt_record(
        run_dir: Path,
        attempt_number: int,
        report: VerificationReport,
        *,
        code: Optional[str] = None,
    ) -> PipelineAttempt:
        attempt_dir = run_dir / f"attempt-{attempt_number:02d}"
        code_path = attempt_dir / "scene.py"
        if code is not None and not code_path.exists():
            code_path.write_text(code, encoding="utf-8")
        return PipelineAttempt(
            attempt=attempt_number,
            code_path=str(code_path.resolve()),
            report_path=str((attempt_dir / "report.json").resolve()),
            report=report,
        )
