"""Storyboard Grader Agent — LLM-as-Judge for storyboard quality.

This agent evaluates the storyboard produced by the Script Architect
against pedagogical and visual quality criteria. If the score falls
below a configurable threshold, it provides critique feedback that
the orchestrator can use to trigger re-generation.

Scoring Criteria:
  - Pedagogical flow (30%): Do scenes build logically?
  - Visual richness  (25%): Are there enough animations/objects?
  - Narration quality (25%): 3B1B tone, not dry textbook style?
  - Completeness     (20%): Does it cover the PDF content adequately?
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.base import (
    Agent,
    AgentExecutionError,
    AgentInput,
    AgentOutput,
    BackoffStrategy,
    RetryPolicy,
)
from src.schemas.storyboard import StoryboardJSON
from src.schemas.source_segment import SourceSegment

logger = logging.getLogger(__name__)

# Per-task model tiering. Storyboard grading is cheap rubric scoring (LLM-as-
# judge) with no cascade risk, so it runs on the cheapest capable tier.
GRADER_MODEL = "gpt-5.4-nano"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GradingCriterion:
    """Single grading criterion with score and feedback."""
    name: str
    weight: float
    score: float          # 0-10
    feedback: str


@dataclass
class GradingReport:
    """Full grading report produced by the agent."""
    overall_score: float
    passed: bool
    criteria: List[GradingCriterion]
    summary: str
    improvement_suggestions: List[str]


# ---------------------------------------------------------------------------
# I/O Contracts
# ---------------------------------------------------------------------------

class StoryboardGraderInput(AgentInput):
    """Input to the Storyboard Grader Agent."""

    def __init__(
        self,
        storyboard: StoryboardJSON,
        source_text_summary: str = "",
        source_segments: Optional[List[SourceSegment]] = None,
        pass_threshold: float = 6.0,
    ):
        self.storyboard = storyboard
        self.source_text_summary = source_text_summary
        self.source_segments = source_segments or []
        self.pass_threshold = pass_threshold


class StoryboardGraderOutput(AgentOutput):
    """Output when the storyboard passes grading."""

    def __init__(self, storyboard: StoryboardJSON, report: GradingReport):
        self.storyboard = storyboard
        self.report = report


class StoryboardGraderFailure(AgentOutput):
    """Output when the storyboard fails grading — includes critique for retry."""

    def __init__(self, storyboard: StoryboardJSON, report: GradingReport):
        self.storyboard = storyboard
        self.report = report


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StoryboardGraderAgent(Agent):
    """LLM-as-Judge that scores storyboard quality.

    When an OpenAI API key is available the agent calls GPT-4o-mini to
    produce a structured grading JSON.  Without a key it falls back to
    a deterministic heuristic grader (useful for tests / mock mode).
    """

    CRITERIA_WEIGHTS = {
        "pedagogical_flow": 0.30,
        "visual_richness": 0.25,
        "narration_quality": 0.25,
        "completeness": 0.20,
    }

    def __init__(self, strict: bool = False):
        self.client = None
        self.strict = strict
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("StoryboardGraderAgent: OpenAI client initialised.")
            except ImportError:
                if self.strict:
                    raise AgentExecutionError(
                        "MISSING_DEPENDENCY",
                        "openai package is required in --strict mode for StoryboardGrader"
                    )
                logger.warning("openai package not installed — using heuristic grader.")
        else:
            if self.strict:
                raise AgentExecutionError(
                    "MISSING_API_KEY",
                    "OPENAI_API_KEY is required in --strict mode for StoryboardGrader"
                )
            logger.warning("No OPENAI_API_KEY — using heuristic grader.")

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def execute(self, input_data: StoryboardGraderInput) -> AgentOutput:
        if not self.validate_input(input_data):
            raise AgentExecutionError(
                "INVALID_INPUT",
                "Input must be a StoryboardGraderInput.",
            )

        storyboard = input_data.storyboard

        if self.client:
            report = self._grade_with_llm(
                storyboard,
                input_data.source_text_summary,
                input_data.source_segments,
            )
        else:
            report = self._grade_heuristic(
                storyboard,
                source_segments=input_data.source_segments,
            )

        report.passed = report.overall_score >= input_data.pass_threshold

        if report.passed:
            logger.info(f"Storyboard PASSED grading (score={report.overall_score:.1f}).")
            return StoryboardGraderOutput(storyboard=storyboard, report=report)
        else:
            logger.warning(
                f"Storyboard FAILED grading (score={report.overall_score:.1f} "
                f"< threshold={input_data.pass_threshold})."
            )
            return StoryboardGraderFailure(storyboard=storyboard, report=report)

    def validate_input(self, input_data: AgentInput) -> bool:
        return isinstance(input_data, StoryboardGraderInput) and hasattr(input_data, "storyboard")

    def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=2,
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            retryable_errors=["API_TIMEOUT", "RATE_LIMIT_EXCEEDED"],
        )

    # ------------------------------------------------------------------
    # LLM grading
    # ------------------------------------------------------------------

    def _grade_with_llm(
        self,
        storyboard: StoryboardJSON,
        source_summary: str,
        source_segments: Optional[List[SourceSegment]] = None,
    ) -> GradingReport:
        """Use GPT-4o-mini to produce structured grading JSON."""
        assert self.client is not None, "LLM client not initialized"

        scenes_summary = []
        for s in storyboard.scenes:
            scenes_summary.append({
                "scene_id": s.scene_id,
                "narration_preview": s.narration[:200],
                "num_objects": len(s.visual_intent.mathematical_objects),
                "num_transforms": len(s.visual_intent.transformations),
                "duration": s.duration_estimate,
                "source_segment_ids": getattr(s, "source_segment_ids", []),
            })
        segment_summary = [
            {
                "segment_id": segment.segment_id,
                "pages": f"{segment.page_start}-{segment.page_end}",
                "heading": segment.heading,
            }
            for segment in source_segments or []
        ]

        prompt = f"""
You are an expert educational content reviewer.

STORYBOARD METADATA:
- PDF: {storyboard.pdf_metadata.filename}
- Pages: {storyboard.pdf_metadata.page_count}
- Scenes: {len(storyboard.scenes)}

SCENES SUMMARY:
{json.dumps(scenes_summary, indent=2)}

SOURCE SEGMENTS:
{json.dumps(segment_summary, indent=2) if segment_summary else "Not provided."}

SOURCE TEXT SUMMARY (if available):
{source_summary or "Not provided."}

Grade this storyboard on a scale of 0-10 for each criterion:
1. pedagogical_flow — Do scenes build logically from simple to complex?
2. visual_richness — Are there enough mathematical objects and animations?
3. narration_quality — Is the narration conversational and engaging (3B1B style)?
4. completeness — Does the storyboard adequately cover the source material?

Return ONLY valid JSON:
{{
  "pedagogical_flow": {{"score": <0-10>, "feedback": "<short feedback>"}},
  "visual_richness": {{"score": <0-10>, "feedback": "<short feedback>"}},
  "narration_quality": {{"score": <0-10>, "feedback": "<short feedback>"}},
  "completeness": {{"score": <0-10>, "feedback": "<short feedback>"}},
  "summary": "<one-sentence overall summary>",
  "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=GRADER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return self._parse_llm_response(data)

        except Exception as e:
            logger.error(f"LLM grading failed: {e} — falling back to heuristic.")
            return self._grade_heuristic(storyboard, source_segments=source_segments)

    def _parse_llm_response(self, data: Dict[str, Any]) -> GradingReport:
        criteria: List[GradingCriterion] = []
        weighted_sum = 0.0

        for name, weight in self.CRITERIA_WEIGHTS.items():
            entry = data.get(name, {})
            score = float(entry.get("score", 5.0))
            feedback = entry.get("feedback", "No feedback.")
            criteria.append(GradingCriterion(name=name, weight=weight, score=score, feedback=feedback))
            weighted_sum += score * weight

        return GradingReport(
            overall_score=round(weighted_sum, 2),
            passed=False,  # caller sets this
            criteria=criteria,
            summary=data.get("summary", ""),
            improvement_suggestions=data.get("improvement_suggestions", []),
        )

    # ------------------------------------------------------------------
    # Heuristic (mock / offline) grading
    # ------------------------------------------------------------------

    def _grade_heuristic(
        self,
        storyboard: StoryboardJSON,
        source_segments: Optional[List[SourceSegment]] = None,
    ) -> GradingReport:
        """Deterministic heuristic grading — no LLM needed."""
        criteria: List[GradingCriterion] = []

        # 1. Pedagogical flow — check scene ordering and dependencies
        has_deps = any(len(s.dependencies) > 0 for s in storyboard.scenes)
        flow_score = 8.0 if has_deps else 6.0
        criteria.append(GradingCriterion(
            name="pedagogical_flow", weight=0.30, score=flow_score,
            feedback="Dependencies present." if has_deps else "No inter-scene dependencies found.",
        ))

        # 2. Visual richness — count objects and transformations
        total_objects = sum(len(s.visual_intent.mathematical_objects) for s in storyboard.scenes)
        total_transforms = sum(len(s.visual_intent.transformations) for s in storyboard.scenes)
        richness_score = min(10.0, (total_objects + total_transforms) / max(len(storyboard.scenes), 1) * 2)
        criteria.append(GradingCriterion(
            name="visual_richness", weight=0.25, score=richness_score,
            feedback=f"{total_objects} objects, {total_transforms} transformations across {len(storyboard.scenes)} scenes.",
        ))

        # 3. Narration quality — check average narration length
        avg_narration_len = sum(len(s.narration) for s in storyboard.scenes) / max(len(storyboard.scenes), 1)
        narration_score = min(10.0, avg_narration_len / 50)
        criteria.append(GradingCriterion(
            name="narration_quality", weight=0.25, score=narration_score,
            feedback=f"Average narration length: {avg_narration_len:.0f} chars.",
        ))

        # 4. Completeness - real source-segment coverage, with a legacy fallback.
        if source_segments:
            expected_segment_ids = {segment.segment_id for segment in source_segments}
            covered_sequence = [
                segment_id
                for scene in storyboard.scenes
                for segment_id in getattr(scene, "source_segment_ids", [])
                if segment_id in expected_segment_ids
            ]
            covered_segment_ids = set(covered_sequence)
            duplicate_count = sum(
                max(0, covered_sequence.count(segment_id) - 1)
                for segment_id in expected_segment_ids
            )
            coverage_ratio = len(covered_segment_ids) / max(len(expected_segment_ids), 1)
            redundancy_penalty = duplicate_count / max(len(expected_segment_ids), 1)
            completeness_score = max(
                0.0,
                min(10.0, (coverage_ratio * 10.0) - (redundancy_penalty * 3.0)),
            )
            feedback = (
                f"{len(covered_segment_ids)}/{len(expected_segment_ids)} source segments covered; "
                f"{duplicate_count} redundant segment references."
            )
        else:
            ratio = len(storyboard.scenes) / max(storyboard.pdf_metadata.page_count, 1)
            completeness_score = min(10.0, ratio * 5)
            feedback = f"{len(storyboard.scenes)} scenes for {storyboard.pdf_metadata.page_count} pages."
        criteria.append(GradingCriterion(
            name="completeness", weight=0.20, score=completeness_score,
            feedback=feedback,
        ))

        weighted = sum(c.score * c.weight for c in criteria)
        return GradingReport(
            overall_score=round(weighted, 2),
            passed=False,
            criteria=criteria,
            summary="Heuristic grading complete.",
            improvement_suggestions=[],
        )
