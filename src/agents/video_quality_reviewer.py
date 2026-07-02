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
import re
import subprocess
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, AgentExecutionError, AgentInput, AgentOutput, RetryPolicy
from src.agents.audio_video_composer import AudioVideoComposerFailure, AudioVideoComposerOutput
from src.agents.tts_agent import TTSError, TTSOutput
from src.agents.video_renderer import VideoRendererFailure, VideoRendererOutput
from src.schemas.animation_code import AnimationCodeObject
from src.schemas.output_manifest import OutputManifest
from src.schemas.storyboard import StoryboardJSON
from src.utils.ffmpeg_finder import get_ffmpeg, get_ffprobe
from src.utils.frame_analysis import analyze_keyframes

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
        findings.extend(self._review_helper_semantics(input_data.storyboard, code_by_scene))
        findings.extend(self._review_animation_pacing(code_by_scene))
        findings.extend(self._review_progressive_reveal(code_by_scene))
        findings.extend(self._review_anti_patterns(input_data.storyboard, code_by_scene))
        findings.extend(self._review_rendered_frames(render_map))

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

    def _review_helper_semantics(
        self, storyboard: Any, code_by_scene: Dict[str, Any]
    ) -> List[VideoQualityFinding]:
        """Check that generated code uses the helper layer where expected.

        Inspects each scene's ``scene_state_in`` / ``scene_state_out`` against
        the generated code to detect cases where the generator used raw Manim
        constructors instead of the reusable helper functions.

        Returns a list of medium-severity findings; returns an empty list when
        the storyboard or code are not available.
        """
        findings: List[VideoQualityFinding] = []
        if storyboard is None:
            return findings

        for scene in getattr(storyboard, "scenes", []):
            code_obj = code_by_scene.get(scene.scene_id)
            if code_obj is None:
                continue
            source = code_obj.code

            visual_intent = getattr(scene, "visual_intent", None)
            if visual_intent is None:
                continue

            scene_state_in = getattr(visual_intent, "scene_state_in", None)
            scene_state_out = getattr(visual_intent, "scene_state_out", None)

            # ── Coordinate-system reuse check ──────────────────────────────
            cs_in = getattr(scene_state_in, "coordinate_systems", {}) or {}
            if cs_in and "create_coordinate_system" not in source:
                findings.append(VideoQualityFinding(
                    issue="A reusable coordinate system was redrawn without the helper-backed factory.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene.scene_id,
                    evidence=(
                        "scene_state_in declares reusable coordinate systems, but generated code "
                        "does not call create_coordinate_system(...)"
                    ),
                    suggested_fix=(
                        "Reuse the prior coordinate system binding and build it through "
                        "create_coordinate_system(...) only when first introduced."
                    ),
                    llm_direction=(
                        f"Regenerate scene {scene.scene_id} so recurring axes or planes are "
                        "created or reused through create_coordinate_system(...)."
                    ),
                ))

            # ── Notation helper check ───────────────────────────────────────
            nb_out = getattr(scene_state_out, "notation_bindings", {}) or {}
            if nb_out and "create_notation" not in source:
                findings.append(VideoQualityFinding(
                    issue="Persistent notation was emitted without the notation helper.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene.scene_id,
                    evidence=(
                        "scene_state_out exports notation bindings, but generated code "
                        "does not call create_notation(...)"
                    ),
                    suggested_fix=(
                        "Build persistent equations and symbolic labels through create_notation(...) "
                        "so notation keys stay stable across retries."
                    ),
                    llm_direction=(
                        f"Regenerate scene {scene.scene_id} so notation-bearing MathTex objects "
                        "are created through create_notation(...)."
                    ),
                ))

            # ── Character-role helper check ─────────────────────────────────
            cr_out = getattr(scene_state_out, "character_roles", {}) or {}
            if cr_out and "create_character_for_role" not in source:
                findings.append(VideoQualityFinding(
                    issue="Character-role continuity was emitted without the role-aware helper.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene.scene_id,
                    evidence=(
                        "scene_state_out exports character roles, but generated code "
                        "does not call create_character_for_role(...)"
                    ),
                    suggested_fix=(
                        "Use create_character_for_role(...) so teacher and student roles "
                        "stay deterministic across scenes."
                    ),
                    llm_direction=(
                        f"Regenerate scene {scene.scene_id} so persistent character roles "
                        "use create_character_for_role(...)."
                    ),
                ))

        return findings

    # ------------------------------------------------------------------
    # Task 5: Pacing & Timing Hard-Constraint Reviewer
    # ------------------------------------------------------------------

    def _review_animation_pacing(
        self,
        code_by_scene: Dict[str, AnimationCodeObject],
    ) -> List[VideoQualityFinding]:
        """Flag run_time / wait violations against the 3B1B timing contract."""
        from src.utils.manim_engine_contract import TIMING_CONTRACT, WAIT_CONTRACT

        findings: List[VideoQualityFinding] = []
        run_time_re = re.compile(r"run_time\s*=\s*([\d.]+)")
        wait_re = re.compile(r"self\.wait\(\s*([\d.]+)\s*\)")
        play_re = re.compile(r"self\.play\(")

        for scene_id, code_obj in code_by_scene.items():
            source = code_obj.code
            violations = 0
            total_checked = 0

            # Check run_time values against contract
            for line_no, line in enumerate(source.splitlines(), 1):
                rt_match = run_time_re.search(line)
                if rt_match:
                    total_checked += 1
                    rt = float(rt_match.group(1))
                    # Try to match the animation family from the line
                    for anim, (lo, hi) in TIMING_CONTRACT.items():
                        if anim in line:
                            if rt < lo or rt > hi:
                                violations += 1
                                findings.append(VideoQualityFinding(
                                    issue=f"Animation timing violation: {anim} run_time={rt}s (allowed {lo}–{hi}s).",
                                    suspected_stage="animation_generator",
                                    severity="medium",
                                    scene_id=scene_id,
                                    evidence=f"Line {line_no}: {line.strip()}",
                                    suggested_fix=f"Set run_time between {lo} and {hi}s for {anim}.",
                                    llm_direction=f"Adjust {anim} timing in scene {scene_id} to be within {lo}–{hi}s.",
                                ))
                            break

            # Check for missing wait() between consecutive self.play() calls
            play_positions = [m.start() for m in play_re.finditer(source)]
            wait_positions = [m.start() for m in wait_re.finditer(source)]
            min_wait = WAIT_CONTRACT.get("between_animations", 0.5)

            for i in range(len(play_positions) - 1):
                start = play_positions[i]
                end = play_positions[i + 1]
                # Check if there's a wait between these two play calls
                has_wait = any(start < wp < end for wp in wait_positions)
                if not has_wait:
                    violations += 1

            if violations > 0 and total_checked > 0 and violations > total_checked * 0.5:
                findings.append(VideoQualityFinding(
                    issue="Majority of animations violate 3B1B timing contract.",
                    suspected_stage="animation_generator",
                    severity="high",
                    scene_id=scene_id,
                    evidence=f"{violations} violations out of {total_checked} checked animations.",
                    suggested_fix="Review all animation timings against the 3B1B timing contract.",
                    llm_direction=f"Regenerate scene {scene_id} with 3B1B-compliant timing.",
                ))

        return findings

    # ------------------------------------------------------------------
    # Task 6: Progressive Reveal Reviewer Check
    # ------------------------------------------------------------------

    def _review_progressive_reveal(
        self,
        code_by_scene: Dict[str, AnimationCodeObject],
    ) -> List[VideoQualityFinding]:
        """Detect complex equations written in a single shot."""
        findings: List[VideoQualityFinding] = []
        # Pattern: Write(MathTex(...)) where the MathTex has many terms
        write_mathtex_re = re.compile(
            r'Write\(\s*MathTex\((.*?)\)',
            re.DOTALL,
        )
        # Also catch Write(some_var) where some_var was defined as a complex MathTex
        complex_var_re = re.compile(
            r'(\w+)\s*=\s*MathTex\(\s*r?"([^"]*)"',
        )

        for scene_id, code_obj in code_by_scene.items():
            source = code_obj.code

            # Build map of variable → tex content
            var_tex: Dict[str, str] = {}
            for m in complex_var_re.finditer(source):
                var_tex[m.group(1)] = m.group(2)

            # Check direct Write(MathTex(...)) calls
            for m in write_mathtex_re.finditer(source):
                inner = m.group(1)
                # Count terms by separating on +, -, =, \frac, etc.
                terms = re.split(r'[+\-=]|\\frac|\\sum|\\int|\\prod', inner)
                if len(terms) > 3:
                    findings.append(VideoQualityFinding(
                        issue="Complex equation written in one shot instead of progressive reveal.",
                        suspected_stage="animation_generator",
                        severity="medium",
                        scene_id=scene_id,
                        evidence=f"Write(MathTex(...)) with {len(terms)} terms.",
                        suggested_fix="Use progressive_write() or animate sub-expressions sequentially.",
                        llm_direction=(
                            f"In scene {scene_id}, break the complex equation into parts and reveal them "
                            "progressively using progressive_write() or staged Write() calls."
                        ),
                    ))

            # Check Write(var) where var is a complex MathTex
            write_var_re = re.compile(r'Write\(\s*(\w+)')
            for m in write_var_re.finditer(source):
                var_name = m.group(1)
                if var_name in var_tex:
                    tex = var_tex[var_name]
                    terms = re.split(r'[+\-=]|\\frac|\\sum|\\int|\\prod', tex)
                    if len(terms) > 3:
                        findings.append(VideoQualityFinding(
                            issue=f"Complex equation '{var_name}' written in one shot.",
                            suspected_stage="animation_generator",
                            severity="medium",
                            scene_id=scene_id,
                            evidence=f"Write({var_name}) where {var_name} has {len(terms)} terms.",
                            suggested_fix="Use progressive_write() or isolate sub-expressions.",
                            llm_direction=(
                                f"In scene {scene_id}, reveal '{var_name}' progressively instead of "
                                "all at once."
                            ),
                        ))

        return findings

    # ------------------------------------------------------------------
    # Task 7: LLM-Based Narration-Visual Fidelity Reviewer
    # ------------------------------------------------------------------

    def _review_narration_fidelity(
        self,
        storyboard: Any,
        code_by_scene: Dict[str, AnimationCodeObject],
    ) -> List[VideoQualityFinding]:
        """Use an LLM call to check narration-visual alignment.

        For each scene, sends narration + generated code and asks whether the
        code shows every formula, definition, and example the narration mentions.
        This is an optional check — it degrades gracefully if no API key is
        available.
        """
        findings: List[VideoQualityFinding] = []
        try:
            import openai
        except ImportError:
            logger.info("openai not installed; skipping narration-fidelity check.")
            return findings

        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.info("OPENAI_API_KEY not set; skipping narration-fidelity check.")
            return findings

        client = openai.OpenAI(api_key=api_key)

        for scene in getattr(storyboard, "scenes", []):
            code_obj = code_by_scene.get(scene.scene_id)
            if code_obj is None:
                continue
            narration = getattr(scene, "narration", "")
            if not narration:
                continue

            try:
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a video production QA reviewer. Given a scene's narration "
                                "and its Manim code, identify any formulas, definitions, or examples "
                                "mentioned in the narration that are NOT visually represented in the code. "
                                "Reply with a JSON object: {\"missing\": [list of missing items], "
                                "\"aligned\": true/false}. If everything is aligned, set missing to []."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"NARRATION:\n{narration}\n\n"
                                f"MANIM CODE:\n{code_obj.code}"
                            ),
                        },
                    ],
                    max_completion_tokens=500,
                )
                content = response.choices[0].message.content or ""
                # Try to parse the response
                import json
                try:
                    result = json.loads(content)
                    missing = result.get("missing", [])
                    if missing:
                        findings.append(VideoQualityFinding(
                            issue="Narration mentions content not shown visually.",
                            suspected_stage="animation_generator",
                            severity="medium",
                            scene_id=scene.scene_id,
                            evidence=f"Missing visual elements: {', '.join(missing[:5])}",
                            suggested_fix="Add visual representations for the missing narration content.",
                            llm_direction=(
                                f"In scene {scene.scene_id}, add Manim objects to show: "
                                f"{', '.join(missing[:5])}."
                            ),
                        ))
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM fidelity response for %s", scene.scene_id)
            except Exception as exc:
                logger.warning("Narration fidelity check failed for %s: %s", scene.scene_id, exc)

        return findings

    # ------------------------------------------------------------------
    # Task 8: Anti-Pattern Detection
    # ------------------------------------------------------------------

    def _review_anti_patterns(
        self,
        storyboard: Any,
        code_by_scene: Dict[str, AnimationCodeObject],
    ) -> List[VideoQualityFinding]:
        """Detect common anti-patterns in generated Manim code."""
        findings: List[VideoQualityFinding] = []

        for scene_id, code_obj in code_by_scene.items():
            source = code_obj.code
            lines = source.splitlines()

            # 1. FadeOut immediately followed by FadeIn (should be Transform)
            for i in range(len(lines) - 1):
                if "FadeOut(" in lines[i] and "FadeIn(" in lines[i + 1]:
                    findings.append(VideoQualityFinding(
                        issue="FadeOut immediately followed by FadeIn — use ReplacementTransform instead.",
                        suspected_stage="animation_generator",
                        severity="medium",
                        scene_id=scene_id,
                        evidence=f"Lines {i + 1}-{i + 2}: {lines[i].strip()} / {lines[i + 1].strip()}",
                        suggested_fix="Replace FadeOut→FadeIn with ReplacementTransform for smoother morphing.",
                        llm_direction=(
                            f"In scene {scene_id}, replace FadeOut/FadeIn pairs with "
                            "ReplacementTransform for 3B1B-style smooth transitions."
                        ),
                    ))

            # 2. No self.wait() between consecutive self.play() calls
            play_line_nos = [
                i for i, line in enumerate(lines) if "self.play(" in line
            ]
            for j in range(len(play_line_nos) - 1):
                start_line = play_line_nos[j]
                end_line = play_line_nos[j + 1]
                has_wait = any(
                    "self.wait(" in lines[k]
                    for k in range(start_line + 1, end_line)
                )
                if not has_wait:
                    findings.append(VideoQualityFinding(
                        issue="Missing self.wait() between consecutive self.play() calls.",
                        suspected_stage="animation_generator",
                        severity="medium",
                        scene_id=scene_id,
                        evidence=f"Lines {start_line + 1} and {end_line + 1} have no wait in between.",
                        suggested_fix="Add self.wait(0.5) or more between play() calls for breathing room.",
                        llm_direction=(
                            f"In scene {scene_id}, add self.wait() calls between consecutive self.play() "
                            "calls to give viewers time to absorb changes."
                        ),
                    ))

            # 3. Text walls (>3 Text objects visible simultaneously)
            text_creates = re.findall(r'(\w+)\s*=\s*Text\(', source)
            if len(text_creates) > 3:
                # Check if they're all added without any being faded out
                fade_outs = re.findall(r'FadeOut\(\s*(\w+)', source)
                still_visible = len(text_creates) - len(
                    [t for t in text_creates if t in fade_outs]
                )
                if still_visible > 3:
                    findings.append(VideoQualityFinding(
                        issue="Potential text wall: too many Text objects visible simultaneously.",
                        suspected_stage="animation_generator",
                        severity="medium",
                        scene_id=scene_id,
                        evidence=f"{still_visible} Text objects likely visible at once.",
                        suggested_fix="Reduce simultaneous text to <=3. Use FadeOut for old text before adding new.",
                        llm_direction=(
                            f"In scene {scene_id}, limit visible Text objects to 3 or fewer. "
                            "FadeOut older text before introducing new explanatory labels."
                        ),
                    ))

            # 4. Missing LayoutTracker registration after object creation
            create_pattern = re.compile(
                r'(\w+)\s*=\s*(?:MathTex|Text|Axes|NumberPlane|Circle|Square|Triangle|Arrow|Line)\('
            )
            created_vars = [m.group(1) for m in create_pattern.finditer(source)]
            if created_vars and "layout" in source:
                for var in created_vars:
                    if f'layout.register("{var}"' not in source and f"layout.register('{var}'" not in source:
                        # Only flag if layout is being used at all in this scene
                        if "layout.register(" in source:
                            findings.append(VideoQualityFinding(
                                issue=f"Object '{var}' created but not registered with LayoutTracker.",
                                suspected_stage="animation_generator",
                                severity="low",
                                scene_id=scene_id,
                                evidence=f"'{var}' is created but no layout.register(\"{var}\", ...) found.",
                                suggested_fix=f"Add layout.register(\"{var}\", {var}) after positioning.",
                                llm_direction=(
                                    f"In scene {scene_id}, register '{var}' with LayoutTracker after creation."
                                ),
                            ))

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

    # ------------------------------------------------------------------
    # Frame-level visual QA (ADR-005) — judge what is actually ON SCREEN
    # ------------------------------------------------------------------

    KEYFRAME_SAMPLE_FRACTIONS = (0.15, 0.4, 0.65, 0.9)

    def _extract_keyframes(self, video_path: str) -> List[Any]:
        """Extract PIL keyframes at fixed fractions of the video's duration.

        Best-effort: any failure (no ffmpeg, unreadable file, missing PIL)
        returns [] with a warning so review never blocks the pipeline.
        Sampling starts at 15% to skip the black lead-in before the first
        Write/Create finishes.
        """
        import tempfile

        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL unavailable — skipping frame-level visual QA.")
            return []

        duration = self._get_media_duration(video_path)
        if duration <= 0:
            return []

        frames: List[Any] = []
        try:
            ffmpeg = get_ffmpeg()
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i, fraction in enumerate(self.KEYFRAME_SAMPLE_FRACTIONS):
                    out_path = str(Path(tmp_dir) / f"frame_{i}.png")
                    result = subprocess.run(
                        [
                            ffmpeg, "-v", "quiet",
                            "-ss", f"{duration * fraction:.3f}",
                            "-i", video_path,
                            "-frames:v", "1",
                            "-y", out_path,
                        ],
                        capture_output=True,
                        timeout=30,
                    )
                    if result.returncode == 0 and Path(out_path).exists():
                        with Image.open(out_path) as img:
                            frames.append(img.convert("RGB").copy())
        except Exception as exc:
            logger.warning("Keyframe extraction failed for %s: %s", video_path, exc)
            return []
        return frames

    def _review_rendered_frames(self, render_map: Dict[str, Any]) -> List[VideoQualityFinding]:
        """Flag scenes whose rendered pixels violate the visualization thesis.

        Empty frames mean nothing was drawn when something should be on
        screen; a static sample means the 'animation' is a still slide.
        Both route back to the animation generator, not the renderer — the
        video encoded exactly what the code drew.
        """
        findings: List[VideoQualityFinding] = []
        for scene_id, scene_result in render_map.items():
            if not getattr(scene_result, "success", False):
                continue
            video_path = getattr(scene_result, "video_path", None)
            if not video_path or not Path(video_path).exists():
                continue

            frames = self._extract_keyframes(video_path)
            if not frames:
                continue
            report = analyze_keyframes(frames)

            if report.mostly_empty:
                findings.append(VideoQualityFinding(
                    issue="Rendered video is mostly empty — almost nothing is drawn on screen.",
                    suspected_stage="animation_generator",
                    severity="critical",
                    scene_id=scene_id,
                    evidence=(
                        f"{len(report.empty_frame_indices)}/{report.frame_count} sampled "
                        f"keyframes are blank (non-background fractions: "
                        f"{[round(f, 4) for f in report.nonblank_fractions]})."
                    ),
                    suggested_fix="Regenerate the scene so its planned objects are actually created and remain visible.",
                    llm_direction=(
                        "The scene rendered to blank frames. Ensure every planned object is "
                        "instantiated, added via self.play/self.add, positioned inside the "
                        "visible frame (x in [-6.5, 6.5], y in [-3.5, 3.5]), and not faded "
                        "out immediately after creation."
                    ),
                ))
            elif report.empty_frame_indices:
                findings.append(VideoQualityFinding(
                    issue="Some sampled keyframes are blank.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene_id,
                    evidence=(
                        f"Blank keyframes at sample indices {report.empty_frame_indices} "
                        f"of {report.frame_count}."
                    ),
                    suggested_fix="Overlap object lifetimes so the screen is never empty mid-scene.",
                    llm_direction=(
                        "Part of the scene shows an empty screen. Stagger FadeOut/Create so "
                        "the next visual appears before the previous one is fully removed."
                    ),
                ))

            if report.is_static and not report.mostly_empty:
                findings.append(VideoQualityFinding(
                    issue="Scene appears static — all sampled keyframes are nearly identical.",
                    suspected_stage="animation_generator",
                    severity="medium",
                    scene_id=scene_id,
                    evidence=(
                        f"All {report.frame_count} sampled keyframes are nearly identical; "
                        "the video is a still slide, not an animation."
                    ),
                    suggested_fix="Replace long static holds with transformations, movement, or progressive reveal.",
                    llm_direction=(
                        "This is a visualization tool: fill the scene with motion. Use "
                        "Transform/ReplacementTransform, .animate moves, or progressive "
                        "reveal instead of drawing everything once and waiting."
                    ),
                ))
        return findings

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
