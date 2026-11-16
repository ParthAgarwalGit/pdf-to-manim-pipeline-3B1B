"""Final delivery checks for V2: measure every composed scene and the joined
video, apply bounded deterministic repairs, and report ready / needs review.

Nothing here calls an LLM or a paid API. Every measurement goes through
``MediaProbes`` (ffprobe/ffmpeg by default), so tests inject fakes. The gate
never raises: a video is still delivered when it cannot be measured, it is
just reported as ``needs_review``.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from src.utils import media_probe
from src.utils.media_probe import Loudness, MediaInfo

logger = logging.getLogger(__name__)

Severity = Literal["info", "low", "medium", "high", "critical"]
Repair = Literal["recompose", "renormalize", "reconcat"]

THRESHOLDS: dict[str, float] = {
    "av_gap_seconds": 0.3,
    "narration_cut_seconds": 0.05,
    # Must stay above tts.SCENE_END_PAUSE_SECONDS (2.0): that pause is intended.
    "silence_seconds": 3.0,
    "silence_noise_db": -40.0,
    "loudness_target_lufs": -16.0,
    "loudness_tolerance_lu": 1.5,
    "true_peak_max_dbtp": -1.0,
    "concat_drift_base_seconds": 0.25,
    "concat_drift_per_clip_seconds": 0.02,
}

# A frozen last frame longer than this is a high-severity finding.
FROZEN_TAIL_HIGH_SECONDS = 1.0
# The broader repair goes first: a recompose also redoes loudness.
_REPAIR_ORDER: tuple[str, ...] = ("recompose", "renormalize")
_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class DeliveryCheck(BaseModel):
    check: str
    passed: Optional[bool]  # None: could not be measured, or not applicable (info)
    severity: Severity
    message: str = ""
    measured: dict[str, Any] = Field(default_factory=dict)
    repair: Optional[Repair] = None


class SceneDelivery(BaseModel):
    scene_id: str
    start_seconds: float
    end_seconds: float
    passed: bool
    checks: list[DeliveryCheck] = Field(default_factory=list)
    repairs_applied: list[str] = Field(default_factory=list)


class DeliveryReport(BaseModel):
    status: Literal["ready", "needs_review"]
    summary: str
    scenes: list[SceneDelivery] = Field(default_factory=list)
    final_checks: list[DeliveryCheck] = Field(default_factory=list)
    repair_rounds: int = 0
    thresholds: dict[str, float] = Field(default_factory=dict)


@dataclass
class SceneClip:
    """One scene of the final video. ``composed_path`` is updated in place
    when a recompose repair returns a different file. ``mode`` is the scene's
    delivery mode, carried for the caller; no check depends on it."""

    scene_id: str
    composed_path: Path
    render_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    narration_seconds: Optional[float] = None
    tail_seconds: float = 0.0
    silent_narration: bool = False
    mode: str = "manim"


@dataclass
class MediaProbes:
    probe: Callable[[str], Optional[MediaInfo]] = media_probe.probe
    loudness: Callable[[str], Optional[Loudness]] = media_probe.measure_loudness
    silences: Callable[..., Optional[list[tuple[float, float]]]] = media_probe.detect_silences
    normalize: Callable[..., bool] = media_probe.normalize_audio_track


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def check_scene(
    clip: SceneClip,
    probes: Optional[MediaProbes] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> list[DeliveryCheck]:
    probes = probes or MediaProbes()
    limits = {**THRESHOLDS, **(thresholds or {})}
    composed = probes.probe(str(clip.composed_path))
    narration = clip.narration_seconds
    if narration is None and clip.audio_path is not None:
        narration = _duration(probes.probe(str(clip.audio_path)), "audio")
    return [
        _check_streams(composed),
        _check_av_duration(clip, narration, probes, limits),
        _check_narration_intact(composed, narration, limits),
        _check_narration_gaps(clip, composed, narration, probes, limits),
        _check_loudness(clip, probes, limits),
    ]


def check_final(
    final_path: Optional[Path],
    clips: list[SceneClip],
    probes: Optional[MediaProbes] = None,
    thresholds: Optional[dict[str, float]] = None,
    *,
    normalized: bool = True,
) -> list[DeliveryCheck]:
    """Whole-video checks. ``normalized`` says the final file was joined by a
    normalising re-encode (``media.concatenate``), which makes differing clip
    formats harmless; pass False for a stream-copy join."""
    probes = probes or MediaProbes()
    limits = {**THRESHOLDS, **(thresholds or {})}
    infos = [probes.probe(str(clip.composed_path)) for clip in clips]
    exists = final_path is not None and Path(final_path).is_file()
    final = probes.probe(str(final_path)) if exists else None

    if not exists:
        concat = DeliveryCheck(
            check="concat_success", passed=False, severity="critical",
            message="The final video was not produced.", repair="reconcat",
        )
    elif final is None:
        concat = DeliveryCheck(
            check="concat_success", passed=None, severity="critical",
            message="The final video could not be measured.",
        )
    else:
        missing = [kind for kind, stream in (("video", final.video), ("audio", final.audio)) if stream is None]
        concat = DeliveryCheck(
            check="concat_success", passed=not missing, severity="critical",
            message=f"The final video has no {' or '.join(missing)} track." if missing else "",
            measured={"video": final.video is not None, "audio": final.audio is not None},
            repair="reconcat" if missing else None,
        )

    tolerance = (
        limits["concat_drift_base_seconds"]
        + limits["concat_drift_per_clip_seconds"] * len(clips)
    )
    return [
        concat,
        _check_codec_consistency(clips, infos, normalized),
        _check_duration_drift(final, infos, tolerance),
        _check_av_drift(final, tolerance),
    ]


def run_delivery_gate(
    clips: list[SceneClip],
    final_path: Optional[Path],
    *,
    probes: Optional[MediaProbes] = None,
    thresholds: Optional[dict[str, float]] = None,
    max_repair_rounds: int = 2,
    recompose: Optional[Callable[[SceneClip], Path]] = None,
    reconcat: Optional[Callable[[list[Path]], Path]] = None,
) -> DeliveryReport:
    """Measure, repair within bounds, re-measure, and report. Never raises."""
    try:
        return _run_gate(
            list(clips), final_path, probes or MediaProbes(),
            {**THRESHOLDS, **(thresholds or {})}, max(0, int(max_repair_rounds)),
            recompose, reconcat,
        )
    except Exception as exc:  # delivery must go ahead without the gate
        logger.warning("Delivery gate failed: %s", exc, exc_info=True)
        report = DeliveryReport(
            status="needs_review", summary="",
            final_checks=[DeliveryCheck(
                check="gate_error", passed=False, severity="medium",
                message=f"The delivery checks could not be completed: {exc}",
            )],
            thresholds=dict(THRESHOLDS),
        )
        report.summary = "\n".join(summary_lines(report))
        return report


def summary_lines(report: DeliveryReport) -> list[str]:
    total = len(report.scenes)
    passed = sum(scene.passed for scene in report.scenes)
    label = "Ready" if report.status == "ready" else "Needs review"
    lines = [f"{label}: {passed}/{total} scenes passed sync and quality checks."]
    for scene in report.scenes:
        if scene.passed:
            continue
        worst = _worst([check for check in scene.checks if not _acceptable(check)])
        reason = worst.message if worst and worst.message else "did not pass its checks."
        lines.append(
            f"{scene.scene_id} ({format_timestamp(scene.start_seconds)}-"
            f"{format_timestamp(scene.end_seconds)}): {reason}"
        )
    for check in report.final_checks:
        if not _acceptable(check):
            lines.append(f"Final video: {check.message or check.check + ' did not pass.'}")
    return lines


# ── scene checks ────────────────────────────────────────────────────────

def _check_streams(composed: Optional[MediaInfo]) -> DeliveryCheck:
    if composed is None:
        return DeliveryCheck(
            check="streams", passed=None, severity="high",
            message="The scene clip could not be measured.",
        )
    missing = [kind for kind, stream in (("video", composed.video), ("audio", composed.audio)) if stream is None]
    return DeliveryCheck(
        check="streams", passed=not missing, severity="high",
        message=f"The scene clip has no {' or '.join(missing)} track." if missing else "",
        measured={"video": composed.video is not None, "audio": composed.audio is not None},
    )


def _check_av_duration(
    clip: SceneClip, narration: Optional[float], probes: MediaProbes, limits: dict[str, float],
) -> DeliveryCheck:
    if clip.render_path is None or narration is None:
        return _not_applicable("av_duration", "No separate animation render to compare with the narration.")
    video = _duration(probes.probe(str(clip.render_path)), "video")
    if video is None:
        return DeliveryCheck(
            check="av_duration", passed=None, severity="medium",
            message="The animation length could not be measured.",
        )
    gap = narration - video
    measured = {"video_seconds": round(video, 3), "narration_seconds": round(narration, 3), "gap_seconds": round(gap, 3)}
    if gap <= limits["av_gap_seconds"]:
        return DeliveryCheck(check="av_duration", passed=True, severity="medium", measured=measured)
    return DeliveryCheck(
        check="av_duration", passed=False,
        severity="high" if gap > FROZEN_TAIL_HIGH_SECONDS else "medium",
        message=f"The animation ends {gap:.1f}s before the narration, so the last frame is frozen.",
        measured=measured, repair="recompose",
    )


def _check_narration_intact(
    composed: Optional[MediaInfo], narration: Optional[float], limits: dict[str, float],
) -> DeliveryCheck:
    if narration is None:
        return _not_applicable("narration_intact", "The narration length is unknown.")
    composed_audio = _duration(composed, "audio") if composed and composed.audio else None
    if composed_audio is None:
        return DeliveryCheck(
            check="narration_intact", passed=None, severity="high",
            message="The narration in the scene clip could not be measured.",
        )
    cut = narration - composed_audio
    passed = cut <= limits["narration_cut_seconds"]
    return DeliveryCheck(
        check="narration_intact", passed=passed, severity="high",
        message="" if passed else f"The last {cut:.2f}s of the narration is cut off.",
        measured={"narration_seconds": round(narration, 3), "composed_audio_seconds": round(composed_audio, 3)},
        repair=None if passed else "recompose",
    )


def _check_narration_gaps(
    clip: SceneClip,
    composed: Optional[MediaInfo],
    narration: Optional[float],
    probes: MediaProbes,
    limits: dict[str, float],
) -> DeliveryCheck:
    if clip.silent_narration:
        return _not_applicable("narration_gaps", "The narration is a silent placeholder.")
    # Prefer the narration file: the composed clip may end in padded silence
    # when the animation runs longer than the narration.
    source = clip.audio_path or clip.composed_path
    total = narration
    if total is None and clip.audio_path is None and composed is not None and composed.audio:
        total = _duration(composed, "audio")
    silences = probes.silences(
        str(source), noise_db=limits["silence_noise_db"],
        min_duration=limits["silence_seconds"], total=total,
    )
    if silences is None:
        return DeliveryCheck(
            check="narration_gaps", passed=None, severity="medium",
            message="Silences in the narration could not be measured.",
        )
    # The pause at the end of a scene is intentional; only speech-time gaps count.
    speech_end = narration - clip.tail_seconds if narration is not None else None
    gaps = []
    for start, end in silences:
        end = min(end, speech_end) if speech_end is not None else end
        if end - start >= limits["silence_seconds"]:
            gaps.append((start, end))
    message = ""
    if gaps:
        start, end = gaps[0]
        message = f"The narration has a {end - start:.1f}s silence {start:.1f}s into the scene."
    return DeliveryCheck(
        check="narration_gaps", passed=not gaps, severity="medium", message=message,
        measured={"silences": [[round(start, 2), round(end, 2)] for start, end in gaps]},
    )


def _check_loudness(clip: SceneClip, probes: MediaProbes, limits: dict[str, float]) -> DeliveryCheck:
    if clip.silent_narration:
        return _not_applicable("loudness", "The narration is a silent placeholder.")
    loudness = probes.loudness(str(clip.composed_path))
    if loudness is None or loudness.integrated == float("-inf"):
        return DeliveryCheck(
            check="loudness", passed=None, severity="medium",
            message="Narration loudness could not be measured.",
        )
    target = limits["loudness_target_lufs"]
    off_target = abs(loudness.integrated - target) > limits["loudness_tolerance_lu"]
    too_hot = loudness.true_peak > limits["true_peak_max_dbtp"]
    message = ""
    if off_target:
        message = f"Narration loudness is {loudness.integrated:.1f} LUFS; the target is {target:.0f}."
    elif too_hot:
        message = f"The audio peaks at {loudness.true_peak:.1f} dBTP and may distort."
    return DeliveryCheck(
        check="loudness", passed=not (off_target or too_hot), severity="medium", message=message,
        measured={"integrated_lufs": loudness.integrated, "true_peak_dbtp": loudness.true_peak},
        repair="renormalize" if off_target or too_hot else None,
    )


# ── whole-video checks ──────────────────────────────────────────────────

def _signature(info: MediaInfo) -> tuple:
    video, audio = info.video, info.audio
    return (
        video.codec, video.width, video.height, round(video.fps or 0.0, 2), video.pix_fmt,
        audio.codec if audio else None,
        audio.sample_rate if audio else None,
        audio.channels if audio else None,
    )


def _check_codec_consistency(
    clips: list[SceneClip], infos: list[Optional[MediaInfo]], normalized: bool,
) -> DeliveryCheck:
    if not clips:
        return _not_applicable("codec_consistency", "There are no scene clips to compare.")
    if any(info is None or info.video is None for info in infos):
        return DeliveryCheck(
            check="codec_consistency", passed=None, severity="high",
            message="Scene clip formats could not be measured.",
        )
    signatures = [_signature(info) for info in infos]
    common = Counter(signatures).most_common(1)[0][0]
    odd = [clip.scene_id for clip, signature in zip(clips, signatures) if signature != common]
    silent = [clip.scene_id for clip, info in zip(clips, infos) if info.audio is None]
    measured = {"signature": list(common), "odd_clips": odd, "no_audio_clips": silent, "normalized": normalized}
    if silent:
        return DeliveryCheck(
            check="codec_consistency", passed=False, severity="high",
            message=f"Scene clip(s) {', '.join(silent)} have no audio track.", measured=measured,
        )
    if not odd:
        return DeliveryCheck(check="codec_consistency", passed=True, severity="high", measured=measured)
    if normalized:
        return DeliveryCheck(
            check="codec_consistency", passed=True, severity="info",
            message=(
                f"Scene clip(s) {', '.join(odd)} used a different format; "
                "the final video was re-encoded to one format."
            ),
            measured=measured,
        )
    return DeliveryCheck(
        check="codec_consistency", passed=False, severity="high",
        message=f"Scene clip(s) {', '.join(odd)} use a different video/audio format from the rest.",
        measured=measured, repair="reconcat",
    )


def _check_duration_drift(
    final: Optional[MediaInfo], infos: list[Optional[MediaInfo]], tolerance: float,
) -> DeliveryCheck:
    durations = [_duration(info) for info in infos]
    final_seconds = _duration(final)
    if final_seconds is None or not durations or any(value is None for value in durations):
        return DeliveryCheck(
            check="duration_drift", passed=None, severity="high",
            message="The final video length could not be compared with its scenes.",
        )
    total = sum(durations)
    drift = abs(final_seconds - total)
    passed = drift <= tolerance
    return DeliveryCheck(
        check="duration_drift", passed=passed, severity="high",
        message="" if passed else f"The final video is {drift:.2f}s off from the sum of its scenes.",
        measured={
            "final_seconds": round(final_seconds, 3), "scene_total_seconds": round(total, 3),
            "tolerance_seconds": round(tolerance, 3),
        },
        repair=None if passed else "reconcat",
    )


def _check_av_drift(final: Optional[MediaInfo], tolerance: float) -> DeliveryCheck:
    video = _duration(final, "video") if final and final.video else None
    audio = _duration(final, "audio") if final and final.audio else None
    if video is None or audio is None:
        return DeliveryCheck(
            check="av_drift", passed=None, severity="high",
            message="Audio/video drift in the final video could not be measured.",
        )
    drift = abs(audio - video)
    passed = drift <= tolerance
    return DeliveryCheck(
        check="av_drift", passed=passed, severity="high",
        message="" if passed else f"Audio and video drift {drift:.2f}s apart over the final video.",
        measured={
            "final_video_seconds": round(video, 3), "final_audio_seconds": round(audio, 3),
            "tolerance_seconds": round(tolerance, 3),
        },
        repair=None if passed else "reconcat",
    )


# ── measure, repair, re-measure ─────────────────────────────────────────

def _run_gate(
    clips: list[SceneClip],
    final_path: Optional[Path],
    probes: MediaProbes,
    limits: dict[str, float],
    max_rounds: int,
    recompose: Optional[Callable[[SceneClip], Path]],
    reconcat: Optional[Callable[[list[Path]], Path]],
) -> DeliveryReport:
    checks = [check_scene(clip, probes, limits) for clip in clips]
    repairs: list[list[str]] = [[] for _ in clips]
    unavailable = {"recompose"} if recompose is None else set()
    exhausted: list[set[str]] = [set(unavailable) for _ in clips]
    rounds = 0
    clips_changed = False

    for _ in range(max_rounds):
        todo = []
        for index in range(len(clips)):
            wanted = {check.repair for check in checks[index] if check.passed is False}
            repair = next((r for r in _REPAIR_ORDER if r in wanted and r not in exhausted[index]), None)
            if repair:
                todo.append((index, repair))
        if not todo:
            break
        rounds += 1
        for index, repair in todo:
            targets = {c.check for c in checks[index] if c.passed is False and c.repair == repair}
            if not _apply_repair(repair, clips[index], probes, recompose):
                exhausted[index].add(repair)
                continue
            clips_changed = True
            repairs[index].append(repair)
            checks[index] = check_scene(clips[index], probes, limits)
            if any(c.check in targets and c.passed is not True for c in checks[index]):
                exhausted[index].add(repair)  # it ran but did not fix its check

    # The final video is rebuilt at most once, whatever asks for it.
    attempted = rebuilt = False

    def rebuild() -> None:
        nonlocal final_path, attempted, rebuilt
        attempted = True
        try:
            final_path = Path(reconcat([clip.composed_path for clip in clips]))
            rebuilt = True
        except Exception as exc:
            logger.warning("Delivery gate could not rebuild the final video: %s", exc)

    can_rebuild = reconcat is not None and max_rounds > 0 and bool(clips)
    if clips_changed and can_rebuild:
        rebuild()
    final_checks = check_final(final_path, clips, probes, limits)
    if can_rebuild and not attempted and any(
        c.passed is False and c.repair == "reconcat" for c in final_checks
    ):
        rounds = max(rounds, 1)
        rebuild()
        final_checks = check_final(final_path, clips, probes, limits)
    if clips_changed and not rebuilt:
        final_checks.append(DeliveryCheck(
            check="final_rebuilt", passed=False, severity="high",
            message="Scene clips were repaired but the final video was not rebuilt from them.",
        ))

    scenes: list[SceneDelivery] = []
    clock = 0.0
    for index, clip in enumerate(clips):
        seconds = _duration(probes.probe(str(clip.composed_path))) or 0.0
        scenes.append(SceneDelivery(
            scene_id=clip.scene_id,
            start_seconds=round(clock, 2),
            end_seconds=round(clock + seconds, 2),
            passed=all(_acceptable(check) for check in checks[index]),
            checks=checks[index],
            repairs_applied=repairs[index],
        ))
        clock += seconds

    ready = (
        bool(scenes)
        and all(scene.passed for scene in scenes)
        and all(_acceptable(check) for check in final_checks)
    )
    report = DeliveryReport(
        status="ready" if ready else "needs_review", summary="", scenes=scenes,
        final_checks=final_checks, repair_rounds=rounds, thresholds=limits,
    )
    report.summary = "\n".join(summary_lines(report))
    return report


def _apply_repair(
    repair: str,
    clip: SceneClip,
    probes: MediaProbes,
    recompose: Optional[Callable[[SceneClip], Path]],
) -> bool:
    try:
        if repair == "recompose":
            if recompose is None:
                return False
            clip.composed_path = Path(recompose(clip))
            return True
        if repair == "renormalize":
            return _renormalize(clip.composed_path, probes)
    except Exception as exc:
        logger.warning("Delivery repair %s for %s failed: %s", repair, clip.scene_id, exc)
    return False


def _renormalize(source: Path, probes: MediaProbes) -> bool:
    target = source.with_name(f"{source.stem}.norm{source.suffix}")
    try:
        if not probes.normalize(str(source), str(target), probes.loudness(str(source))):
            return False
        os.replace(target, source)
        return True
    finally:
        target.unlink(missing_ok=True)  # only still there when the swap did not happen


# ── helpers ─────────────────────────────────────────────────────────────

def _duration(info: Optional[MediaInfo], kind: Optional[str] = None) -> Optional[float]:
    if info is None:
        return None
    stream = info.video if kind == "video" else info.audio if kind == "audio" else None
    seconds = (stream.duration if stream else None) or info.duration
    return seconds if seconds and seconds > 0 else None


def _not_applicable(check: str, reason: str) -> DeliveryCheck:
    return DeliveryCheck(check=check, passed=None, severity="info", message=f"Not checked: {reason}")


def _acceptable(check: DeliveryCheck) -> bool:
    return check.passed is True or (check.passed is None and check.severity == "info")


def _worst(checks: list[DeliveryCheck]) -> Optional[DeliveryCheck]:
    # Highest severity first; a measured failure outranks an unmeasurable check.
    return max(
        checks, key=lambda c: (_SEVERITY_RANK[c.severity], c.passed is False), default=None,
    )
