"""Pre-delivery quality gate: measure, repair (bounded), report.

Runs after composition. Every composed scene is measured with ffprobe and
ffmpeg against fixed thresholds; failures are routed to the smallest repair:

- ``retime``: spread the measured audio/video gap across the scene's own
  ``self.wait`` holds, re-render, recompose (no LLM);
- ``recompose``: merge again (the composer never trims narration);
- ``renormalize``: two-pass loudness normalization of the clip;
- ``reconcat``: re-encode the final concatenation;
- ``regenerate``: re-codegen the scene with the finding as feedback, for
  visual failures (a static beat window, a blank render). The only repair
  that costs an LLM call; capped per run by ``max_regenerations``.

Each scene gets at most ``max_rounds`` repair rounds and passing scenes are
never touched. The result is a ``DeliveryReport``: ``ready`` only when every
scene is in the video and passes every check, with no blocking review
finding; otherwise ``needs_review`` with plain-language reasons and the
timestamps of each failing scene.

Design: docs/superpowers/specs/2026-11-07-professor-control-and-presenter-design.md (Part H).
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.schemas.delivery_report import CheckResult, DeliveryReport, SceneDelivery
from src.utils import media_probe
from src.utils.scene_retime import adjust_waits

logger = logging.getLogger(__name__)

# Initial thresholds (spec Part H); to be tuned on the fixed lecture set.
THRESHOLDS: Dict[str, float] = {
    "av_tolerance_s": 0.3,
    "frozen_tail_max_s": 1.0,
    "narration_clip_tolerance_s": 0.05,
    "narration_gap_max_s": 2.0,
    "silence_noise_db": -40.0,
    "loudness_target_lufs": media_probe.TARGET_LUFS,
    "loudness_tolerance_lu": 1.5,
    "true_peak_max_dbtp": -1.0,
    "concat_drift_base_s": 0.1,
    "concat_drift_per_clip_s": 0.005,
    "beat_onset_tolerance_s": 0.75,
    # Fraction of (160x90) pixels changing between samples that counts as
    # motion; a small dot sliding across a Manim frame measures ~0.0005.
    "activity_threshold": 0.0002,
    "asr_max_wer": 0.10,
}

# When several checks fail, the cheapest broad repair goes first: a retime
# re-renders and recomposes, which also redoes loudness and narration.
# Regeneration (an LLM call) comes last.
_REPAIR_PRIORITY = ("retime", "recompose", "renormalize", "regenerate")
_BLOCKING_SEVERITIES = {"high", "critical"}


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance over the reference length (case/punctuation-blind)."""
    import re

    def words(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1] / len(ref)


def format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


@dataclass
class SceneMedia:
    scene_id: str
    index: int
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    composed_path: Optional[str] = None
    narration: str = ""
    # (beat_id, start, end, text) from the timing track, in narration order.
    beats: List[tuple] = field(default_factory=list)
    kind: str = "manim"  # manim | presenter


@dataclass
class GateResult:
    scenes: List[SceneDelivery]
    final_checks: List[CheckResult]
    repair_rounds: int
    video_output: Any = None
    composition_output: Any = None
    gate: Any = None  # the DeliveryGate that produced it (builds the report)
    proposals: List[str] = field(default_factory=list)  # unapproved presenter bridges


def _frame_review(scene_id: str, video_path: str) -> List[Any]:
    """Blocking keyframe findings (blank renders) from the existing frame QA."""
    from types import SimpleNamespace
    from src.agents.video_quality_reviewer import VideoQualityReviewerAgent

    findings = VideoQualityReviewerAgent()._review_rendered_frames(
        {scene_id: SimpleNamespace(success=True, video_path=video_path)}
    )
    return [f for f in findings if getattr(f, "severity", "") in _BLOCKING_SEVERITIES]


@dataclass
class _Probes:
    probe: Callable[[str], Optional[media_probe.MediaInfo]] = media_probe.probe
    loudness: Callable[[str], Optional[media_probe.Loudness]] = media_probe.measure_loudness
    silences: Callable[..., Optional[list]] = media_probe.detect_silences
    normalize: Callable[..., bool] = media_probe.normalize_audio_track
    activity: Callable[[str], Optional[List[float]]] = media_probe.frame_activity
    frame_review: Callable[[str, str], List[Any]] = _frame_review
    # Speech-to-text of an audio file; None disables the fidelity check.
    transcribe: Optional[Callable[[str], Optional[str]]] = None


class DeliveryGate:
    def __init__(
        self,
        renderer: Any = None,
        composer: Any = None,
        render_quality: str = "medium",
        max_rounds: int = 2,
        probes: Optional[_Probes] = None,
        regenerate: Optional[Callable[[str, List[str]], Any]] = None,
        max_regenerations: int = 3,
    ):
        self.renderer = renderer
        self.composer = composer
        self.render_quality = render_quality
        self.max_rounds = max(0, int(max_rounds))
        self.probes = probes or _Probes()
        self.thresholds = dict(THRESHOLDS)
        # (scene_id, feedback) -> SceneRenderResult of fresh code, or None.
        self.regenerate = regenerate
        self.max_regenerations = max(0, int(max_regenerations))
        self.regenerations = 0

    # ── measurement ─────────────────────────────────────────────────────

    def check_scene(self, media: SceneMedia) -> List[CheckResult]:
        t = self.thresholds
        checks: List[CheckResult] = []
        if not media.audio_path or not Path(media.audio_path).is_file():
            checks.append(CheckResult(
                check="narration_present", passed=False, severity="high",
                message="This scene has no narration.",
            ))
            return checks

        video = self.probes.probe(media.video_path) if media.video_path else None
        audio = self.probes.probe(media.audio_path)
        composed = self.probes.probe(media.composed_path) if media.composed_path else None
        v = (video.video.duration if video and video.video and video.video.duration else None) or (video.duration if video else None)
        a = audio.duration if audio else None

        if v is None or a is None:
            checks.append(CheckResult(
                check="av_duration", passed=None,
                message="Scene video or narration length could not be measured.",
            ))
        else:
            gap = a - v
            measured = {"video_s": round(v, 3), "audio_s": round(a, 3), "gap_s": round(gap, 3)}
            if abs(gap) <= t["av_tolerance_s"]:
                checks.append(CheckResult(check="av_duration", passed=True, measured=measured))
            elif gap > 0:
                checks.append(CheckResult(
                    check="av_duration", passed=False,
                    severity="high" if gap > t["frozen_tail_max_s"] else "medium",
                    message=f"The animation ends {gap:.1f}s before the narration, so the last frame is frozen.",
                    measured=measured, repair="retime",
                ))
            else:
                checks.append(CheckResult(
                    check="av_duration", passed=False,
                    severity="medium" if -gap > t["frozen_tail_max_s"] else "low",
                    message=f"The narration ends {-gap:.1f}s before the animation does.",
                    measured=measured, repair="retime",
                ))

        composed_audio = composed.audio.duration if composed and composed.audio else None
        if composed_audio is None or a is None:
            checks.append(CheckResult(
                check="narration_intact", passed=None,
                message="The composed narration could not be measured.",
            ))
        else:
            cut = a - composed_audio
            passed = cut <= t["narration_clip_tolerance_s"]
            checks.append(CheckResult(
                check="narration_intact", passed=passed,
                severity="high",
                message="" if passed else f"The last {cut:.2f}s of the narration is cut off.",
                measured={"tts_s": round(a, 3), "composed_audio_s": round(composed_audio, 3)},
                repair=None if passed else "recompose",
            ))

        silences = self.probes.silences(
            media.audio_path, noise_db=t["silence_noise_db"],
            min_duration=t["narration_gap_max_s"], total=a,
        )
        if silences is None:
            checks.append(CheckResult(
                check="narration_gaps", passed=None,
                message="Silences in the narration could not be measured.",
            ))
        else:
            gaps = [(s, e) for s, e in silences if e - s >= t["narration_gap_max_s"]]
            message = ""
            if gaps:
                start, end = gaps[0]
                message = f"The narration has a {end - start:.1f}s silence {start:.1f}s into the scene."
            checks.append(CheckResult(
                check="narration_gaps", passed=not gaps, message=message,
                measured={"silences": [[round(s, 2), round(e, 2)] for s, e in gaps]},
            ))

        loudness = self.probes.loudness(media.composed_path) if media.composed_path else None
        if loudness is None or loudness.integrated == float("-inf"):
            checks.append(CheckResult(
                check="loudness", passed=None,
                message="Narration loudness could not be measured.",
            ))
        else:
            off_target = abs(loudness.integrated - t["loudness_target_lufs"]) > t["loudness_tolerance_lu"]
            too_hot = loudness.true_peak > t["true_peak_max_dbtp"]
            message = ""
            if off_target:
                message = (
                    f"Narration loudness is {loudness.integrated:.1f} LUFS; "
                    f"the target is {t['loudness_target_lufs']:.0f}."
                )
            elif too_hot:
                message = f"The audio peaks at {loudness.true_peak:.1f} dBTP and may distort."
            checks.append(CheckResult(
                check="loudness", passed=not (off_target or too_hot), message=message,
                measured={"integrated_lufs": loudness.integrated, "true_peak_dbtp": loudness.true_peak},
                repair="renormalize" if (off_target or too_hot) else None,
            ))

        # A presenter clip is a talking head or title card, not an animation:
        # motion-per-beat and blank-frame checks do not apply.
        if media.beats and media.video_path and media.kind != "presenter":
            checks.append(self._check_beat_alignment(media))
        if media.video_path and media.kind != "presenter":
            checks.append(self._check_visual_qa(media))
        if self.probes.transcribe is not None and media.narration and media.composed_path:
            checks.append(self._check_fidelity(media))
        return checks

    def _check_beat_alignment(self, media: SceneMedia) -> CheckResult:
        """Visuals must move during every beat and start when the beat starts."""
        t = self.thresholds
        activity = self.probes.activity(media.video_path)
        if not activity:
            return CheckResult(
                check="beat_alignment", passed=None,
                message="On-screen motion could not be measured.",
            )
        fps = media_probe.ACTIVITY_FPS
        threshold = t["activity_threshold"]
        tolerance = t["beat_onset_tolerance_s"]
        static, late = [], []
        for beat_id, start, end, text in media.beats:
            window = [a for i, a in enumerate(activity) if start <= i / fps < end]
            if window and max(window) < threshold:
                static.append((beat_id, start, end, text))
                continue
            onset = next(
                (i / fps for i, a in enumerate(activity)
                 if start - tolerance <= i / fps < end and a >= threshold),
                None,
            )
            if onset is not None and onset > start + tolerance:
                late.append((beat_id, start, onset, text))

        def snippet(text: str) -> str:
            words = (text or "").split()
            return " ".join(words[:8]) + ("..." if len(words) > 8 else "")

        feedback: List[str] = []
        message = ""
        for beat_id, start, end, text in static:
            feedback.append(
                f"Nothing moves on screen during {beat_id} ({start:.1f}-{end:.1f}s, "
                f"\"{snippet(text)}\"). Right after sync_to(self, \"{beat_id}\"), add a "
                "visible animation that shows what this beat says."
            )
        for beat_id, start, onset, text in late:
            feedback.append(
                f"The visual for {beat_id} starts {onset - start:.1f}s after its narration. "
                f"Begin its animation immediately after sync_to(self, \"{beat_id}\")."
            )
        if static:
            _, start, end, text = static[0]
            message = (
                f"Nothing moves on screen for {end - start:.1f}s while the narration says "
                f"\"{snippet(text)}\""
            )
        elif late:
            _, start, onset, text = late[0]
            message = f"The visual for \"{snippet(text)}\" starts {onset - start:.1f}s after the narration."
        passed = not static and not late
        return CheckResult(
            check="beat_alignment", passed=passed,
            severity="medium" if static else "low",
            message=message,
            measured={
                "static_beats": [b[0] for b in static],
                "late_beats": [b[0] for b in late],
                "feedback": feedback,
            },
            repair=None if passed else "regenerate",
        )

    def _check_visual_qa(self, media: SceneMedia) -> CheckResult:
        try:
            findings = self.probes.frame_review(media.scene_id, media.video_path)
        except Exception as exc:
            logger.warning("Frame review failed for %s: %s", media.scene_id, exc)
            return CheckResult(check="visual_qa", passed=None, message="Frames could not be reviewed.")
        if not findings:
            return CheckResult(check="visual_qa", passed=True)
        first = findings[0]
        return CheckResult(
            check="visual_qa", passed=False, severity=getattr(first, "severity", "critical"),
            message=getattr(first, "issue", "The render failed visual QA."),
            measured={"feedback": [
                getattr(f, "llm_direction", "") or getattr(f, "issue", "") for f in findings
            ]},
            repair="regenerate",
        )

    def _check_fidelity(self, media: SceneMedia) -> CheckResult:
        try:
            transcript = self.probes.transcribe(media.composed_path)
        except Exception as exc:
            logger.warning("Transcription failed for %s: %s", media.scene_id, exc)
            transcript = None
        if transcript is None:
            return CheckResult(
                check="narration_fidelity", passed=None,
                message="The narration could not be transcribed.",
            )
        wer = word_error_rate(media.narration, transcript)
        passed = wer <= self.thresholds["asr_max_wer"]
        return CheckResult(
            check="narration_fidelity", passed=passed,
            message="" if passed else (
                f"The spoken narration differs from the script ({wer:.0%} of words); "
                "words may be skipped or garbled."
            ),
            measured={"wer": round(wer, 3)},
        )

    def check_final(
        self,
        final_path: Optional[str],
        clips: List[SceneMedia],
        reencoded: bool = False,
    ) -> List[CheckResult]:
        """Whole-video checks. ``reencoded`` means the final file was joined
        by re-encoding, which makes differing clip formats harmless."""
        t = self.thresholds
        clips = [c for c in clips if c.composed_path]
        paths = [c.composed_path for c in clips]
        infos = [self.probes.probe(p) for p in paths]
        checks: List[CheckResult] = []

        def signature(info):
            if info is None or info.video is None:
                return None
            audio = info.audio
            return (
                info.video.codec, info.video.width, info.video.height,
                round(info.video.fps or 0, 2), info.video.pix_fmt,
                audio.codec if audio else None,
                audio.sample_rate if audio else None,
                audio.channels if audio else None,
            )

        signatures = [signature(info) for info in infos]
        if not paths or any(s is None for s in signatures):
            checks.append(CheckResult(
                check="stream_format", passed=None,
                message="Scene clip formats could not be measured.",
            ))
        else:
            common = max(set(signatures), key=signatures.count)
            odd = [c.index + 1 for c, s in zip(clips, signatures) if s != common]
            silent = [c.index + 1 for c, s in zip(clips, signatures) if s[5] is None]
            passed = reencoded or (not odd and not silent)
            message = ""
            if passed:
                pass
            elif silent:
                message = f"Scene(s) {', '.join(map(str, silent))} have no audio track, which can corrupt the joined video."
            elif odd:
                message = f"Scene(s) {', '.join(map(str, odd))} use a different video/audio format from the rest."
            checks.append(CheckResult(
                check="stream_format", passed=passed, severity="high", message=message,
                measured={"format": list(common), "reencoded": reencoded},
                repair=None if passed else "reconcat",
            ))

        final = self.probes.probe(final_path) if final_path and Path(final_path).is_file() else None
        clip_total = sum(info.duration for info in infos if info and info.duration) if infos else None
        if final is None or final.video is None or not clip_total:
            checks.append(CheckResult(
                check="concat_drift", passed=None,
                message="The final video could not be measured.",
            ))
        elif final.audio is None:
            checks.append(CheckResult(
                check="concat_drift", passed=False, severity="critical",
                message="The final video has no audio track.", repair="reconcat",
            ))
        else:
            tolerance = t["concat_drift_base_s"] + t["concat_drift_per_clip_s"] * len(paths)
            video_s = final.video.duration or final.duration or 0.0
            audio_s = final.audio.duration or final.duration or 0.0
            av_drift = abs(audio_s - video_s)
            length_drift = abs((final.duration or video_s) - clip_total)
            passed = av_drift <= tolerance and length_drift <= tolerance
            message = ""
            if av_drift > tolerance:
                message = f"Audio and video drift {av_drift:.2f}s apart over the full video."
            elif length_drift > tolerance:
                message = f"The joined video is {length_drift:.2f}s off from the sum of its scenes."
            checks.append(CheckResult(
                check="concat_drift", passed=passed, severity="high", message=message,
                measured={
                    "final_video_s": round(video_s, 3), "final_audio_s": round(audio_s, 3),
                    "scene_total_s": round(clip_total, 3), "tolerance_s": round(tolerance, 3),
                },
                repair=None if passed else "reconcat",
            ))
        return checks

    # ── measure → repair → re-measure ──────────────────────────────────

    def run(
        self,
        storyboard: Any,
        output_dir: str,
        video_output: Any,
        tts_output: List[Any],
        composition_output: Any,
    ) -> GateResult:
        media = self._collect_media(storyboard, video_output, tts_output, composition_output)
        in_video = [m for m in media if m.composed_path]
        checks: Dict[str, List[CheckResult]] = {m.scene_id: self.check_scene(m) for m in in_video}
        repairs: Dict[str, List[str]] = {m.scene_id: [] for m in media}
        exhausted: Dict[str, set] = {m.scene_id: set() for m in media}
        rerendered = composed_changed = False
        rounds = 0

        for _ in range(self.max_rounds):
            todo = []
            for m in in_video:
                options = {c.repair for c in checks[m.scene_id] if c.passed is False and c.repair}
                choice = next(
                    (r for r in _REPAIR_PRIORITY if r in options and r not in exhausted[m.scene_id]),
                    None,
                )
                if choice:
                    todo.append((m, choice))
            if not todo:
                break
            rounds += 1
            for m, repair in todo:
                done = self._repair(repair, m, checks[m.scene_id], output_dir, video_output, composition_output)
                if done:
                    repairs[m.scene_id].append(repair)
                    composed_changed = True
                    rerendered = rerendered or repair in ("retime", "regenerate")
                    checks[m.scene_id] = self.check_scene(m)
                else:
                    exhausted[m.scene_id].add(repair)

        final_path = getattr(composition_output, "final_video_path", None)
        # Every composed clip in playback order, including ones outside the
        # storyboard (the presenter credits card).
        by_id = {m.scene_id: m for m in in_video}
        clips = [
            by_id.get(r.scene_id) or SceneMedia(r.scene_id, -1, composed_path=r.composed_video_path)
            for r in getattr(composition_output, "scene_results", []) or []
            if getattr(r, "success", False) and getattr(r, "composed_video_path", None)
        ] or in_video
        if composed_changed and final_path:
            self._concat(clips, final_path, reencode=False)
        if rerendered:
            self._reconcat_silent(video_output, output_dir)
        final_checks = self.check_final(final_path, clips)
        if final_path and self.max_rounds and any(c.passed is False and c.repair == "reconcat" for c in final_checks):
            rounds = max(rounds, 1)
            if self._concat(clips, final_path, reencode=True):
                final_checks = self.check_final(final_path, clips, reencoded=True)

        scenes = self._timeline(media, checks, repairs)
        proposals = [
            s.scene_id for s in getattr(storyboard, "scenes", []) or []
            if getattr(s, "is_presenter", False) and not getattr(s, "presenter_approved", False)
        ]
        return GateResult(
            scenes=scenes,
            final_checks=final_checks,
            repair_rounds=rounds,
            video_output=video_output,
            composition_output=composition_output,
            proposals=proposals,
        )

    def report(
        self,
        result: GateResult,
        review_findings: Optional[List[Any]] = None,
        directive_checks: Optional[List[CheckResult]] = None,
    ) -> DeliveryReport:
        """Fold in blocking review findings and the professor's directives,
        then decide ready vs needs review."""
        directive_checks = list(directive_checks or [])
        blocking: List[Dict[str, Any]] = []
        by_scene = {s.scene_id: s for s in result.scenes}
        for finding in review_findings or []:
            severity = getattr(finding, "severity", "")
            if severity not in _BLOCKING_SEVERITIES:
                continue
            entry = {
                "issue": getattr(finding, "issue", ""),
                "severity": severity,
                "scene_id": getattr(finding, "scene_id", None),
                "suspected_stage": getattr(finding, "suspected_stage", ""),
            }
            scene = by_scene.get(entry["scene_id"])
            if scene is not None:
                scene.checks.append(CheckResult(
                    check="review", passed=False, severity=severity, message=entry["issue"],
                ))
                scene.passed = False
            else:
                blocking.append(entry)

        passed = [s for s in result.scenes if s.passed]
        ready = bool(
            result.scenes
            and len(passed) == len(result.scenes)
            and all(c.passed is True for c in result.final_checks)
            and all(c.passed is True for c in directive_checks)
            and not blocking
        )
        total = len(result.scenes)
        proposal_note = (
            f"{len(result.proposals)} presenter bridge(s) await your approval and are not in this video."
            if result.proposals else ""
        )
        if ready:
            summary = f"Ready: {total}/{total} scenes passed sync and quality checks."
            if proposal_note:
                summary += "\n" + proposal_note
        else:
            lines = [f"Needs review: {len(passed)}/{total} scenes passed sync and quality checks."]
            for index, scene in enumerate(result.scenes):
                if scene.passed:
                    continue
                label = f"Scene {index + 1}"
                if not scene.in_video:
                    lines.append(f"{label} is missing from the video.")
                    continue
                failing = [c for c in scene.checks if c.passed is False]
                unmeasured = [c for c in scene.checks if c.passed is None]
                reason = (failing or unmeasured)[0].message if (failing or unmeasured) else "did not pass."
                lines.append(
                    f"{label} ({format_timestamp(scene.start)}-{format_timestamp(scene.end)}): {reason}"
                )
            for check in result.final_checks:
                if check.passed is not True:
                    lines.append(f"Whole video: {check.message}")
            for check in directive_checks:
                if check.passed is not True:
                    lines.append(f"Your brief: {check.message}")
            for entry in blocking:
                lines.append(f"Review: {entry['issue']}")
            if proposal_note:
                lines.append(proposal_note)
            summary = "\n".join(lines)
        return DeliveryReport(
            status="ready" if ready else "needs_review",
            summary=summary,
            scenes=result.scenes,
            final_checks=result.final_checks,
            review_findings=blocking,
            directives=directive_checks,
            repair_rounds=result.repair_rounds,
            thresholds=self.thresholds,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _collect_media(storyboard, video_output, tts_output, composition_output) -> List[SceneMedia]:
        renders = {
            r.scene_id: r.video_path
            for r in getattr(video_output, "scene_results", []) or []
            if getattr(r, "success", False) and getattr(r, "video_path", None)
        }
        audio = {
            r.scene_id: r.audio_file_path
            for r in tts_output or []
            if getattr(r, "audio_file_path", None)
        }
        composed = {
            r.scene_id: r.composed_video_path
            for r in getattr(composition_output, "scene_results", []) or []
            if getattr(r, "success", False) and getattr(r, "composed_video_path", None)
        }
        media: List[SceneMedia] = []
        for index, scene in enumerate(getattr(storyboard, "scenes", []) or []):
            if getattr(scene, "is_presenter", False) and not getattr(scene, "presenter_approved", False):
                continue  # a proposal, not part of this video
            timings = getattr(scene, "beat_timings", None) or {}
            texts = {b.beat_id: b.narration for b in getattr(scene, "source_beats", None) or []}
            beats = sorted(
                (beat_id, float(span[0]), float(span[1]), texts.get(beat_id, ""))
                for beat_id, span in timings.items()
            )
            media.append(SceneMedia(
                scene_id=scene.scene_id,
                index=index,
                video_path=renders.get(scene.scene_id),
                audio_path=audio.get(scene.scene_id),
                composed_path=composed.get(scene.scene_id),
                narration=getattr(scene, "narration", "") or "",
                beats=sorted(beats, key=lambda b: b[1]),
                kind="presenter" if getattr(scene, "is_presenter", False) else "manim",
            ))
        return media

    def _timeline(self, media, checks, repairs) -> List[SceneDelivery]:
        scenes: List[SceneDelivery] = []
        clock = 0.0
        for m in media:
            if not m.composed_path:
                scenes.append(SceneDelivery(
                    scene_id=m.scene_id, in_video=False, passed=False,
                    repairs_applied=repairs.get(m.scene_id, []),
                ))
                continue
            info = self.probes.probe(m.composed_path)
            duration = info.duration if info and info.duration else 0.0
            scene_checks = checks.get(m.scene_id, [])
            scenes.append(SceneDelivery(
                scene_id=m.scene_id,
                start=round(clock, 2),
                end=round(clock + duration, 2),
                passed=bool(scene_checks) and all(c.passed is True for c in scene_checks),
                checks=scene_checks,
                repairs_applied=repairs.get(m.scene_id, []),
            ))
            clock += duration
        return scenes

    def _repair(self, repair, media: SceneMedia, checks, output_dir, video_output, composition_output) -> bool:
        try:
            if repair == "retime":
                return self._retime(media, checks, output_dir, video_output, composition_output)
            if repair == "recompose":
                return self._recompose(media, output_dir, composition_output)
            if repair == "renormalize":
                return self._renormalize(media)
            if repair == "regenerate":
                return self._regenerate(media, checks, output_dir, video_output, composition_output)
        except Exception as exc:
            logger.warning("Delivery repair %s for %s failed: %s", repair, media.scene_id, exc)
        return False

    def _retime(self, media, checks, output_dir, video_output, composition_output) -> bool:
        if self.renderer is None or self.composer is None:
            return False
        av = next((c for c in checks if c.check == "av_duration"), None)
        gap = (av.measured or {}).get("gap_s") if av else None
        scene_file = Path(output_dir) / f"scene_{media.scene_id}.py"
        if gap is None or not scene_file.is_file():
            return False
        original = scene_file.read_text(encoding="utf-8")
        new_code, applied = adjust_waits(original, gap)
        if abs(applied) < 0.05:
            return False
        scene_file.write_text(new_code, encoding="utf-8")
        result = self.renderer._render_scene(
            str(scene_file),
            self.renderer.quality_flag_for(self.render_quality),
            str(Path(output_dir) / "media"),
        )
        if not getattr(result, "success", False) or not getattr(result, "video_path", None):
            scene_file.write_text(original, encoding="utf-8")  # keep the working version
            return False
        logger.info("Delivery gate re-timed %s by %+.2fs", media.scene_id, applied)
        media.video_path = result.video_path
        results = getattr(video_output, "scene_results", None)
        if results is not None:
            for i, existing in enumerate(results):
                if existing.scene_id == media.scene_id:
                    results[i] = result
        return self._recompose(media, output_dir, composition_output)

    def _regenerate(self, media, checks, output_dir, video_output, composition_output) -> bool:
        """Re-codegen the scene with the visual findings as feedback, then recompose."""
        if self.regenerate is None or self.regenerations >= self.max_regenerations:
            return False
        feedback: List[str] = []
        for check in checks:
            if check.passed is False and check.repair == "regenerate":
                feedback.extend((check.measured or {}).get("feedback") or [check.message])
        self.regenerations += 1
        result = self.regenerate(media.scene_id, feedback)
        if not getattr(result, "success", False) or not getattr(result, "video_path", None):
            return False
        logger.info("Delivery gate regenerated %s", media.scene_id)
        media.video_path = result.video_path
        results = getattr(video_output, "scene_results", None)
        if results is not None:
            for i, existing in enumerate(results):
                if existing.scene_id == media.scene_id:
                    results[i] = result
        return self._recompose(media, output_dir, composition_output)

    def _recompose(self, media, output_dir, composition_output) -> bool:
        if self.composer is None or not media.video_path:
            return False
        from src.agents.audio_video_composer import SceneMediaPair

        composed_dir = str(Path(output_dir) / "composed")
        os.makedirs(composed_dir, exist_ok=True)
        result = self.composer._compose_scene(
            SceneMediaPair(media.scene_id, media.video_path, media.audio_path), composed_dir,
        )
        if not result.success:
            return False
        media.composed_path = result.composed_video_path
        results = getattr(composition_output, "scene_results", None)
        if results is not None:
            for i, existing in enumerate(results):
                if existing.scene_id == media.scene_id:
                    results[i] = result
            composition_output.total_duration = sum(
                r.final_duration for r in results if r.success
            )
        return True

    def _renormalize(self, media) -> bool:
        if not media.composed_path:
            return False
        source = Path(media.composed_path)
        target = source.with_name(source.stem + ".norm" + source.suffix)
        if not self.probes.normalize(str(source), str(target), self.probes.loudness(str(source))):
            return False
        os.replace(target, source)
        return True

    def _concat(self, clips: List[SceneMedia], final_path: str, reencode: bool) -> bool:
        if self.composer is None:
            return False
        paths = [c.composed_path for c in clips if c.composed_path]
        if not paths:
            return False
        try:
            if len(paths) == 1:
                import shutil
                shutil.copy2(paths[0], final_path)
            elif reencode:
                self.composer._concatenate_with_reencode(paths, final_path)
            else:
                self.composer._concatenate_videos(paths, final_path)
            return True
        except Exception as exc:
            logger.warning("Delivery gate concat failed: %s", exc)
            return False

    def _reconcat_silent(self, video_output, output_dir) -> None:
        if self.renderer is None:
            return
        rendered = [
            r.video_path for r in getattr(video_output, "scene_results", []) or []
            if getattr(r, "success", False) and getattr(r, "video_path", None)
        ]
        final = getattr(video_output, "final_video_path", None) or str(Path(output_dir) / "final_output.mp4")
        try:
            if len(rendered) == 1:
                import shutil
                shutil.copy2(rendered[0], final)
            elif rendered:
                self.renderer._concatenate_videos(rendered, final)
        except Exception as exc:
            logger.warning("Silent preview re-concat failed: %s", exc)
