"""Isolated tracked-Manim rendering and issue-directed frame sampling."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from src.manim_verifier.pipeline import analyze_keyframes
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe

from .diagnostics import analyze_trace
from .models import NarratedScene, VerificationFinding, VisualContract
from .static_verifier import V2StaticVerifier


VERIFIER_VERSION = "v2-verifier-8-creative"


@dataclass
class RenderOutcome:
    safe: bool = False
    render_success: bool = False
    scene_class: Optional[str] = None
    video_path: Optional[str] = None
    trace_path: Optional[str] = None
    frame_paths: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    findings: list[VerificationFinding] = field(default_factory=list)
    error: Optional[str] = None


class TrackedManimRenderer:
    QUALITY_FLAGS = {"low": "-ql", "medium": "-qm", "high": "-qh"}

    def __init__(self):
        self._lock = threading.RLock()
        self._processes: set[subprocess.Popen] = set()

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            try:
                process.terminate()
            except OSError:
                continue

    def render(
        self,
        code: str,
        contract: VisualContract,
        scene: NarratedScene,
        attempt_dir: Path,
        *,
        quality: str = "low",
        timeout_seconds: int = 120,
    ) -> RenderOutcome:
        attempt_dir = attempt_dir.resolve()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        verifier = V2StaticVerifier(
            {item.artifact_id for item in contract.artifact_inventory},
            {item.beat_id for item in scene.beats},
            expected_layout=contract.layout_template.value,
            expected_regions={
                item.artifact_id: item.region.value for item in contract.artifact_inventory
            },
            require_contract_tracking=False,
        )
        report = verifier.verify(code)
        findings = [VerificationFinding(
            code=item.code, severity=item.severity, message=item.message,
            timestamp=item.timestamp, artifact_ids=item.object_ids,
            screenshot_path=item.screenshot_path, code_lines=item.code_lines,
        ) for item in report.issues]
        outcome = RenderOutcome(
            safe=report.safe, scene_class=report.scene_class, findings=findings,
        )
        (attempt_dir / "scene.py").write_text(code, encoding="utf-8")
        if not report.safe or not report.scene_class:
            outcome.error = "Static V2 verification failed; code was not executed."
            return outcome

        trace_path = attempt_dir / "trace.jsonl"
        stdout_path = attempt_dir / "render_stdout.txt"
        stderr_path = attempt_dir / "render_stderr.txt"
        environment = self._environment(trace_path, scene)
        video_path = attempt_dir / "video.mp4"
        # Manim nests partial files deeply. A short staging root avoids the
        # Windows MAX_PATH failure while durable artifacts remain in attempt_dir.
        with tempfile.TemporaryDirectory(prefix="pdf2manim-") as staging_value:
            staging_dir = Path(staging_value)
            staged_scene = staging_dir / "scene.py"
            staged_scene.write_text(code, encoding="utf-8")
            media_dir = staging_dir / "media"
            command = [
                sys.executable, "-m", "manim", self.QUALITY_FLAGS.get(quality, "-ql"),
                "--disable_caching", "--media_dir", str(media_dir),
                str(staged_scene), report.scene_class,
            ]
            process = None
            try:
                process = subprocess.Popen(
                    command, cwd=str(staging_dir), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, env=environment,
                )
                with self._lock:
                    self._processes.add(process)
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                stdout_path.write_text(stdout or "", encoding="utf-8")
                stderr_path.write_text(stderr or "", encoding="utf-8")
            except subprocess.TimeoutExpired as exc:
                if process:
                    process.kill()
                    stdout, stderr = process.communicate()
                else:
                    stdout, stderr = _decode(exc.stdout), _decode(exc.stderr)
                stdout_path.write_text(stdout or "", encoding="utf-8")
                stderr_path.write_text(stderr or "", encoding="utf-8")
                outcome.error = f"Manim render timed out after {timeout_seconds}s."
                outcome.findings.append(VerificationFinding(
                    code="RENDER_TIMEOUT", severity="error", message=outcome.error,
                ))
                return outcome
            except Exception as exc:
                outcome.error = f"Could not start Manim: {exc}"
                stderr_path.write_text(outcome.error, encoding="utf-8")
                outcome.findings.append(VerificationFinding(
                    code="RENDER_START_FAILED", severity="error", message=outcome.error,
                ))
                return outcome
            finally:
                if process:
                    with self._lock:
                        self._processes.discard(process)

            if process.returncode != 0:
                detail = (stderr or stdout or "No renderer output")[-2000:]
                outcome.error = f"Manim exited with {process.returncode}: {detail}"
                outcome.findings.append(VerificationFinding(
                    code="MANIM_RENDER_FAILED", severity="error", message=outcome.error,
                ))
                return outcome
            rendered = _find_video(media_dir, report.scene_class)
            if not rendered:
                outcome.error = "Manim completed without producing a final MP4."
                outcome.findings.append(VerificationFinding(
                    code="VIDEO_NOT_FOUND", severity="error", message=outcome.error,
                ))
                return outcome
            shutil.copy2(rendered, video_path)

        outcome.render_success = True
        outcome.video_path = str(video_path)
        outcome.trace_path = str(trace_path)
        outcome.duration_seconds = probe_duration(video_path)
        outcome.findings.extend(analyze_trace(
            contract, scene, trace_path, outcome.duration_seconds,
            enforce_contract=False,
        ))
        timestamps = sorted({
            *[beat.end_seconds for beat in scene.beats],
            *[item.timestamp for item in outcome.findings if item.timestamp is not None],
            *[outcome.duration_seconds * fraction for fraction in (0.25, 0.50, 0.75)],
            max(0.0, outcome.duration_seconds - 0.1),
        })
        outcome.frame_paths = extract_frames(video_path, attempt_dir / "frames", timestamps)
        _attach_issue_screenshots(outcome.findings, outcome.frame_paths)
        outcome.findings.extend(frame_diagnostics(outcome.frame_paths))
        return outcome

    @staticmethod
    def _environment(trace_path: Path, scene: NarratedScene) -> dict[str, str]:
        blocked = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTHORIZATION")
        environment = {
            key: value for key, value in os.environ.items()
            if not any(fragment in key.upper() for fragment in blocked)
        }
        project_root = str(Path(__file__).resolve().parents[2])
        current = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = project_root + (os.pathsep + current if current else "")
        environment["PDF2MANIM_TRACE_PATH"] = str(trace_path)
        environment["PDF2MANIM_BEAT_SCHEDULE"] = json.dumps([
            {
                "beat_id": beat.beat_id,
                "start": float(beat.start_seconds),
                "end": float(beat.end_seconds),
            }
            for beat in scene.beats
        ])
        return environment


def extract_frames(video_path: Path, output_dir: Path, timestamps: list[float]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    duration = probe_duration(video_path)
    for index, raw in enumerate(timestamps):
        timestamp = max(0.0, min(max(0.0, duration - 0.1), float(raw)))
        path = output_dir / f"frame-{index:03d}-{timestamp:.3f}s.png"
        command = [
            get_ffmpeg(), "-y", "-ss", f"{timestamp:.6f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and path.exists() and path.stat().st_size:
                paths.append(str(path.resolve()))
        except Exception:
            continue
    return paths


def _attach_issue_screenshots(
    findings: list[VerificationFinding], frame_paths: list[str],
) -> None:
    timestamped_frames: list[tuple[float, str]] = []
    for path in frame_paths:
        suffix = Path(path).stem.rsplit("-", 1)[-1]
        try:
            timestamped_frames.append((float(suffix.removesuffix("s")), path))
        except ValueError:
            continue
    for finding in findings:
        if finding.timestamp is None or finding.screenshot_path or not timestamped_frames:
            continue
        finding.screenshot_path = min(
            timestamped_frames,
            key=lambda item: abs(item[0] - finding.timestamp),
        )[1]


def frame_diagnostics(frame_paths: list[str]) -> list[VerificationFinding]:
    if not frame_paths:
        return [VerificationFinding(
            code="FRAME_EXTRACTION_FAILED", severity="warning",
            message="No checkpoint screenshots could be extracted.",
        )]
    images = []
    try:
        for path in frame_paths:
            with Image.open(path) as image:
                images.append(image.copy())
    except Exception as exc:
        return [VerificationFinding(
            code="FRAME_ANALYSIS_FAILED", severity="warning", message=str(exc),
        )]
    analysis = analyze_keyframes(images, empty_fraction=0.0002, static_mean_diff=0.02)
    findings = []
    if analysis.mostly_empty:
        findings.append(VerificationFinding(
            code="VISUAL_MOSTLY_EMPTY", severity="error",
            message="At least half of checkpoint frames are effectively empty.",
            screenshot_path=frame_paths[analysis.empty_frame_indices[0]],
        ))
    elif analysis.empty_frame_indices:
        findings.append(VerificationFinding(
            code="VISUAL_EMPTY_FRAME", severity="warning",
            message=f"Empty checkpoint frames: {analysis.empty_frame_indices}",
        ))
    if analysis.edge_clipped:
        findings.append(VerificationFinding(
            code="VISUAL_EDGE_CLIPPED", severity="warning",
            message=f"Boundary-heavy frames: {analysis.edge_clipped_indices}",
        ))
    if analysis.is_static and not analysis.mostly_empty:
        findings.append(VerificationFinding(
            code="VISUAL_STATIC", severity="warning",
            message="Checkpoint frames show almost no visible change.",
        ))
    return findings


def probe_duration(path: Path) -> float:
    try:
        result = subprocess.run([
            get_ffprobe(), "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ], capture_output=True, text=True, timeout=10)
        return max(0.0, float(result.stdout.strip()))
    except Exception:
        return 0.0


def _find_video(media_dir: Path, scene_class: str) -> Optional[Path]:
    return next((path for path in media_dir.rglob(f"{scene_class}.mp4")
                 if "partial_movie_files" not in path.parts), None)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
