"""Final video quality reviewer for pipeline diagnostics and revision hints.

This agent does not try to be a full perceptual video judge. Instead, it
combines the outputs of rendering, TTS, composition, and the generated scene
code to produce actionable findings:

- what went wrong,
- which pipeline stage most likely caused it,
- what should be changed on the next retry,
- and, when useful, a code-oriented hint snippet for the LLM.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, RetryPolicy
from src.agents.audio_video_composer import AudioVideoComposerFailure, AudioVideoComposerOutput
from src.agents.tts_agent import TTSError, TTSOutput
from src.agents.video_renderer import VideoRendererFailure, VideoRendererOutput
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.output_manifest import OutputManifest
from src.schemas.storyboard import StoryboardJSON
from src.utils.ffmpeg_finder import get_ffprobe

logger = logging.getLogger(__name__)


@dataclass
class VideoQualityFinding:
    """Single review finding for the generated media pipeline."""

    issue: str
    suspected_stage: str
    severity: str = "medium"
    scene_id: Optional[str] = None
    evidence: str = ""
    suggested_fix: str = ""
    llm_direction: str = ""
    code_hint: Optional[str] = None


class VideoQualityReviewerInput(AgentInput):
    """Input contract for final video quality review."""

    def __init__(
        self,
        storyboard: StoryboardJSON,
        manifest: OutputManifest,
        animation_code_objects: Optional[List[AnimationCodeObject]] = None,
        video_output: Optional[Any] = None,
        tts_output: Optional[List[Any]] = None,
        composition_output: Optional[Any] = None,
        final_video_path: Optional[str] = None,
        final_video_expected: bool = True,
    ):
        self.storyboard = storyboard
        self.manifest = manifest
        self.animation_code_objects = animation_code_objects or []
        self.video_output = video_output
        self.tts_output = tts_output or []
        self.composition_output = composition_output
        self.final_video_path = final_video_path
        # When False, the "no final video" check is suppressed — e.g. when
        # TTS is disabled and composition is intentionally skipped.
        self.final_video_expected = final_video_expected


class VideoQualityReviewerOutput(AgentOutput):
    """Output from the final quality review."""

    def __init__(
        self,
        passed: bool,
        summary: str,
        findings: List[VideoQualityFinding],
        final_video_path: Optional[str] = None,
    ):
        self.passed = passed
        self.summary = summary
        self.findings = findings
        self.final_video_path = final_video_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable payload for persistence."""
        return {
            "passed": self.passed,
            "summary": self.summary,
            "final_video_path": self.final_video_path,
            "findings": [asdict(finding) for finding in self.findings],
        }


class VideoQualityReviewerAgent(Agent):
    """Heuristic reviewer that turns media-stage results into revision guidance."""

    DURATION_TOLERANCE_SECONDS = 2.0

    def execute(self, input_data: VideoQualityReviewerInput) -> AgentOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Expected VideoQualityReviewerInput with storyboard and manifest.",
            )

        findings: List[VideoQualityFinding] = []

        code_by_scene = {
            code_obj.scene_id: code_obj for code_obj in input_data.animation_code_objects
        }
        render_map = self._build_render_map(input_data.video_output)
        tts_map = self._build_tts_map(input_data.tts_output)
        composition_map = self._build_composition_map(input_data.composition_output)

        final_video_path = (
            input_data.final_video_path
            or getattr(input_data.composition_output, "final_video_path", None)
            or getattr(input_data.video_output, "final_video_path", None)
        )

        findings.extend(self._review_render_failures(input_data.video_output, code_by_scene))
        findings.extend(self._review_tts_failures(input_data.tts_output))
        findings.extend(self._review_composition_failures(input_data.composition_output))
        findings.extend(
            self._review_duration_alignment(
                input_data.storyboard,
                render_map,
                tts_map,
                composition_map,
            )
        )
        findings.extend(self._review_layout_risk(code_by_scene))

        if input_data.final_video_expected and (
            not final_video_path or not Path(final_video_path).exists()
        ):
            findings.append(
                VideoQualityFinding(
                    issue="No final video artifact was produced.",
                    suspected_stage=self._guess_missing_video_stage(input_data),
                    severity="critical",
                    evidence="Neither composition nor renderer returned a usable final video path.",
                    suggested_fix="Re-run the failing media stage after fixing the reported scene/render/audio issues.",
                    llm_direction=(
                        "Regenerate only the failing scene or stage output. Preserve passing scenes and "
                        "fix the blocking media issue first."
                    ),
                )
            )
        else:
            expected_total = sum(
                getattr(scene, "duration_estimate", 0.0) for scene in input_data.storyboard.scenes
            )
            actual_total = self._get_media_duration(final_video_path)
            if (
                actual_total > 0
                and expected_total > 0
                and abs(actual_total - expected_total) > max(6.0, 0.20 * expected_total)
            ):
                findings.append(
                    VideoQualityFinding(
                        issue="The final video duration diverges materially from the storyboard timing budget.",
                        suspected_stage="script_architect",
                        severity="medium",
                        evidence=(
                            f"Storyboard total duration is ~{expected_total:.1f}s but final video is "
                            f"~{actual_total:.1f}s."
                        ),
                        suggested_fix="Split overloaded scenes or shorten narration in the scenes that drift most.",
                        llm_direction=(
                            "Revise the storyboard so narration and scene scope fit the target duration more tightly. "
                            "Prefer splitting a dense scene over compressing the explanation."
                        ),
                    )
                )

        passed = not any(finding.severity in {"critical", "high"} for finding in findings)
        if passed:
            summary = "Final video passed the heuristic review checks."
        else:
            summary = (
                f"Final video review found {len(findings)} issue(s) that should be addressed before trusting "
                "the output."
            )

        return VideoQualityReviewerOutput(
            passed=passed,
            summary=summary,
            findings=findings,
            final_video_path=final_video_path,
        )

    def validate_input(self, input_data: AgentInput) -> bool:
        return (
            isinstance(input_data, VideoQualityReviewerInput)
            and hasattr(input_data, "storyboard")
            and hasattr(input_data, "manifest")
        )

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(max_attempts=1)

    def _build_render_map(self, video_output: Optional[Any]) -> Dict[str, Any]:
        render_map: Dict[str, Any] = {}
        for scene_result in getattr(video_output, "scene_results", []) or []:
            render_map[scene_result.scene_id] = scene_result
        return render_map

    def _build_tts_map(self, tts_output: List[Any]) -> Dict[str, Any]:
        tts_map: Dict[str, Any] = {}
        for result in tts_output or []:
            if hasattr(result, "scene_id"):
                tts_map[result.scene_id] = result
        return tts_map

    def _build_composition_map(self, composition_output: Optional[Any]) -> Dict[str, Any]:
        composition_map: Dict[str, Any] = {}
        for result in getattr(composition_output, "scene_results", []) or []:
            composition_map[result.scene_id] = result
        return composition_map

    def _review_render_failures(
        self,
        video_output: Optional[Any],
        code_by_scene: Dict[str, AnimationCodeObject],
    ) -> List[VideoQualityFinding]:
        findings: List[VideoQualityFinding] = []
        for scene_result in getattr(video_output, "scene_results", []) or []:
            if scene_result.success:
                continue

            code_hint = None
            if scene_result.scene_id in code_by_scene:
                code_hint = self._duration_code_hint()

            findings.append(
                VideoQualityFinding(
                    issue="Scene rendering failed.",
                    suspected_stage="animation_generator",
                    severity="high",
                    scene_id=scene_result.scene_id,
                    evidence=scene_result.error or "Renderer reported an unspecified error.",
                    suggested_fix="Regenerate this scene's code with the render error and preserve the passing scenes.",
                    llm_direction=(
                        f"Regenerate scene {scene_result.scene_id}. Fix the Manim/runtime issue and keep the "
                        "narration timing aligned with the scene duration."
                    ),
                    code_hint=code_hint,
                )
            )
        return findings

    def _review_tts_failures(self, tts_output: List[Any]) -> List[VideoQualityFinding]:
        findings: List[VideoQualityFinding] = []
        for result in tts_output or []:
            if isinstance(result, TTSError):
                findings.append(
                    VideoQualityFinding(
                        issue="Narration audio generation failed.",
                        suspected_stage="tts_agent",
                        severity="high",
                        scene_id=result.scene_id,
                        evidence=result.error,
                        suggested_fix="Retry TTS for this scene after checking API availability and narration content.",
                        llm_direction=(
                            f"Regenerate only the narration audio for scene {result.scene_id}. "
                            "Do not change the scene visuals."
                        ),
                    )
                )
            elif isinstance(result, TTSOutput):
                audio_path = Path(result.audio_file_path)
                if not audio_path.exists() or result.duration_seconds <= 0:
                    findings.append(
                        VideoQualityFinding(
                            issue="Narration audio output is missing or has zero duration.",
                            suspected_stage="tts_agent",
                            severity="high",
                            scene_id=result.scene_id,
                            evidence=(
                                f"audio_file_path={result.audio_file_path}, "
                                f"duration_seconds={result.duration_seconds:.2f}"
                            ),
                            suggested_fix="Re-run TTS for this scene and verify the saved MP3 before composition.",
                            llm_direction=(
                                f"Regenerate TTS for scene {result.scene_id} and verify the file exists "
                                "before the composition stage."
                            ),
                        )
                    )
        return findings

    def _review_composition_failures(self, composition_output: Optional[Any]) -> List[VideoQualityFinding]:
        findings: List[VideoQualityFinding] = []
        for result in getattr(composition_output, "scene_results", []) or []:
            if result.success:
                continue
            findings.append(
                VideoQualityFinding(
                    issue="Audio-video composition failed for a scene.",
                    suspected_stage="audio_video_composer",
                    severity="high",
                    scene_id=result.scene_id,
                    evidence=result.error or "Composer reported an unspecified error.",
                    suggested_fix="Re-run composition after validating the rendered video and audio files for this scene.",
                    llm_direction=(
                        f"Retry composition for scene {result.scene_id} after checking that both video and audio "
                        "artifacts exist and have valid durations."
                    ),
                )
            )
        return findings

    def _review_duration_alignment(
        self,
        storyboard: StoryboardJSON,
        render_map: Dict[str, Any],
        tts_map: Dict[str, Any],
        composition_map: Dict[str, Any],
    ) -> List[VideoQualityFinding]:
        findings: List[VideoQualityFinding] = []

        for scene in storyboard.scenes:
            scene_id = scene.scene_id
            render_result = render_map.get(scene_id)
            tts_result = tts_map.get(scene_id)
            composition_result = composition_map.get(scene_id)

            audio_duration = getattr(tts_result, "duration_seconds", 0.0) or 0.0
            video_duration = getattr(render_result, "duration_seconds", 0.0) or 0.0
            final_duration = getattr(composition_result, "final_duration", 0.0) or video_duration

            if audio_duration > 0 and video_duration > 0:
                drift = abs(audio_duration - video_duration)
                if drift > self.DURATION_TOLERANCE_SECONDS:
                    longer_side = "audio" if audio_duration > video_duration else "video"
                    findings.append(
                        VideoQualityFinding(
                            issue="Narration and animation durations are out of sync for a scene.",
                            suspected_stage="animation_generator",
                            severity="medium",
                            scene_id=scene_id,
                            evidence=(
                                f"audio={audio_duration:.1f}s, video={video_duration:.1f}s, "
                                f"final={final_duration:.1f}s"
                            ),
                            suggested_fix=(
                                "Adjust the animation pacing so it matches the narration length. "
                                f"The {longer_side} stream is currently leading."
                            ),
                            llm_direction=(
                                f"Revise scene {scene_id} so the animation length matches ~{audio_duration:.1f}s "
                                "of narration. Keep the mathematical content the same and change only pacing."
                            ),
                            code_hint=self._duration_code_hint(target_seconds=audio_duration),
                        )
                    )

        return findings

    def _review_layout_risk(
        self,
        code_by_scene: Dict[str, AnimationCodeObject],
    ) -> List[VideoQualityFinding]:
        findings: List[VideoQualityFinding] = []
        for scene_id, code_obj in code_by_scene.items():
            code = code_obj.code
            if ".next_to(" in code and "layout.place_label" not in code:
                findings.append(
                    VideoQualityFinding(
                        issue="This scene's code is likely to create overlapping labels or cramped layouts.",
                        suspected_stage="animation_generator",
                        severity="medium",
                        scene_id=scene_id,
                        evidence="The generated code uses raw .next_to(...) placement without LayoutTracker.place_label(...).",
                        suggested_fix="Reposition labels through LayoutTracker so they can avoid collisions automatically.",
                        llm_direction=(
                            f"Regenerate scene {scene_id} so labels use LayoutTracker.place_label(...) instead of "
                            "raw next_to positioning."
                        ),
                        code_hint=self._layout_code_hint(),
                    )
                )
        return findings

    def _guess_missing_video_stage(self, input_data: VideoQualityReviewerInput) -> str:
        composition_output = input_data.composition_output
        video_output = input_data.video_output

        if isinstance(composition_output, AudioVideoComposerFailure):
            return "audio_video_composer"
        if isinstance(video_output, VideoRendererFailure):
            return "video_renderer"
        if isinstance(composition_output, AudioVideoComposerOutput):
            return "audio_video_composer"
        if isinstance(video_output, VideoRendererOutput):
            return "audio_video_composer"
        return "video_renderer"

    def _duration_code_hint(self, target_seconds: Optional[float] = None) -> str:
        target_comment = (
            f"# Aim for about {target_seconds:.1f}s total scene time.\n"
            if target_seconds is not None
            else ""
        )
        return (
            "layout = LayoutTracker()\n"
            f"{target_comment}"
            "self.play(Write(main_equation), run_time=2.0)\n"
            "self.wait(2.0)\n"
            "self.play(Transform(main_equation, next_equation), run_time=2.0)\n"
            "self.wait(2.0)  # duration padding to stay aligned with narration\n"
        )

    def _layout_code_hint(self) -> str:
        return (
            'layout = LayoutTracker()\n'
            'layout.register("triangle", triangle)\n'
            'theta = MathTex(r"\\\\theta", color=YELLOW)\n'
            'layout.place_label(theta, "theta", anchor="triangle", preferred=RIGHT, buff=0.2)\n'
        )

    def _get_media_duration(self, file_path: str) -> float:
        """Return duration in seconds, or -1.0 if detection fails."""
        try:
            result = subprocess.run(
                [
                    get_ffprobe(), "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception as exc:
            logger.warning("Duration detection failed for %s: %s", file_path, exc)
            return -1.0
