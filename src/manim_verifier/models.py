"""Serializable contracts for the raw-Manim verifier."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SerializableModel(BaseModel):
    """Pydantic model with a JSON-compatible dictionary helper."""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class VerificationIssue(SerializableModel):
    code: str
    severity: Literal["error", "warning"]
    message: str
    timestamp: Optional[float] = None
    object_ids: List[str] = Field(default_factory=list)
    screenshot_path: Optional[str] = None
    code_lines: List[int] = Field(default_factory=list)


class VerificationReport(SerializableModel):
    safe: bool = False
    syntax_valid: bool = False
    render_success: bool = False
    issues: List[VerificationIssue] = Field(default_factory=list)
    video_path: Optional[str] = None
    frame_paths: List[str] = Field(default_factory=list)
    render_error: Optional[str] = None
    scene_class: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)


class RepairResult(SerializableModel):
    code: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class FrameExtractionResult(SerializableModel):
    frame_paths: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)
    issues: List[VerificationIssue] = Field(default_factory=list)


class PipelineAttempt(SerializableModel):
    attempt: int
    code_path: str
    report_path: str
    report: VerificationReport


class PipelineResult(SerializableModel):
    success: bool
    original_code: str
    final_code: str
    final_video_path: Optional[str] = None
    attempts: List[PipelineAttempt] = Field(default_factory=list)
    repaired: bool = False
    fallback_used: bool = False
    run_directory: Optional[str] = None
    elapsed_seconds: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
