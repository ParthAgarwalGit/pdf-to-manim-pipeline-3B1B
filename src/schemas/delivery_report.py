"""Delivery report: what the pre-delivery quality gate measured and decided.

Written as ``delivery_report.json`` next to the final video. ``status`` is
``ready`` only when every scene is in the video and passed every check, and
no blocking (high/critical) review finding remains. Otherwise the video is
still delivered, as ``needs_review``, with the failing scenes and their
timestamps listed in plain language.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]


class CheckResult(BaseModel):
    check: str = Field(..., description="e.g. av_duration, narration_intact, loudness")
    passed: Optional[bool] = Field(
        ..., description="None when the check could not be measured"
    )
    severity: Severity = "medium"
    message: str = Field(default="", description="Plain-language description of a failure")
    measured: Dict[str, Any] = Field(default_factory=dict)
    repair: Optional[str] = Field(
        default=None, description="Repair mapped to this failure (retime, recompose, ...)"
    )


class SceneDelivery(BaseModel):
    scene_id: str
    start: Optional[float] = Field(default=None, description="Start in the final video (s)")
    end: Optional[float] = Field(default=None, description="End in the final video (s)")
    in_video: bool = True
    passed: bool = False
    checks: List[CheckResult] = Field(default_factory=list)
    repairs_applied: List[str] = Field(default_factory=list)


class DeliveryReport(BaseModel):
    status: Literal["ready", "needs_review"]
    summary: str
    scenes: List[SceneDelivery] = Field(default_factory=list)
    final_checks: List[CheckResult] = Field(default_factory=list)
    review_findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Blocking (high/critical) findings from the final video review",
    )
    directives: List[CheckResult] = Field(
        default_factory=list,
        description="Compliance with the professor's brief; unmet directives block 'ready'",
    )
    change_log: str = Field(
        default="", description="For revisions: what changed and what was reused",
    )
    repair_rounds: int = 0
    thresholds: Dict[str, float] = Field(default_factory=dict)
