"""Serializable contracts for immutable scene benchmarks and evaluation runs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.pipeline_v2.models import (
    LessonScene,
    NarratedScene,
    SourceFact,
    VerificationFinding,
    VisualContract,
)


class EvaluationTrack(str, Enum):
    GENERATOR = "generator"
    REPAIR = "repair"
    PRODUCTION = "production"


class EvaluationCase(BaseModel):
    """A frozen, model-ready scene input.

    Paths point at copied benchmark assets, never mutable production artifacts.
    The content hash covers model-visible structured inputs and frozen repair evidence.
    """

    version: str = "1.0"
    case_id: str
    category: str = "uncategorized"
    source_run: str
    source_scene_id: str
    source_facts: list[SourceFact]
    narration: NarratedScene
    lesson: LessonScene
    visual_contract: VisualContract
    visual_history: list[dict[str, Any]] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    required_visuals: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    audio_path: str | None = None
    initial_code_path: str | None = None
    initial_findings: list[VerificationFinding] = Field(default_factory=list)
    initial_frame_paths: list[str] = Field(default_factory=list)
    content_hash: str = ""

    @model_validator(mode="after")
    def scene_ids_match(self) -> "EvaluationCase":
        scene_ids = {
            self.source_scene_id,
            self.narration.scene_id,
            self.lesson.scene_id,
            self.visual_contract.scene_id,
        }
        if len(scene_ids) != 1:
            raise ValueError(f"case scene IDs do not match: {sorted(scene_ids)}")
        return self


class EvaluationManifest(BaseModel):
    version: str = "1.0"
    benchmark_id: str
    created_at: str
    source_runs: list[str] = Field(default_factory=list)
    case_files: list[str] = Field(default_factory=list)
    benchmark_hash: str = ""


class ModelSpec(BaseModel):
    alias: str
    provider: Literal["google", "openai", "anthropic"] = "google"
    model: str


class CandidateMetrics(BaseModel):
    static_safe: bool = False
    render_success: bool = False
    production_ready: bool = False
    fallback_used: bool = False
    error_count: int = 0
    warning_count: int = 0
    issue_score: int = 0
    findings_by_code: dict[str, int] = Field(default_factory=dict)
    target_duration_seconds: float = 0.0
    rendered_duration_seconds: float = 0.0
    duration_delta_seconds: float = 0.0
    generation_ms: float = 0.0
    render_ms: float = 0.0
    cost_usd: float = 0.0
    input_units: int = 0
    output_units: int = 0


class CandidateResult(BaseModel):
    candidate_id: str
    case_id: str
    track: EvaluationTrack
    model: ModelSpec
    replicate: int = Field(ge=1)
    status: Literal["SUCCEEDED", "FAILED", "INTERRUPTED"]
    case_hash: str
    code_path: str | None = None
    original_code_path: str | None = None
    video_path: str | None = None
    frame_paths: list[str] = Field(default_factory=list)
    diagnostics_path: str | None = None
    input_path: str | None = None
    request_paths: list[str] = Field(default_factory=list)
    response_paths: list[str] = Field(default_factory=list)
    findings: list[VerificationFinding] = Field(default_factory=list)
    metrics: CandidateMetrics = Field(default_factory=CandidateMetrics)
    repaired: bool = False
    repair_accepted: bool = False
    error: str | None = None


class EvaluationRunConfig(BaseModel):
    benchmark_path: Path
    output_root: Path = Path("evals/runs")
    track: EvaluationTrack = EvaluationTrack.GENERATOR
    models: list[ModelSpec]
    replicates: int = Field(default=1, ge=1, le=5)
    quality: Literal["low", "medium"] = "low"
    max_cost_usd: float = Field(default=1.0, gt=0.0)
    workers: int = Field(default=2, ge=1, le=8)
    max_repairs: int = Field(default=1, ge=0, le=1)
    timeout_seconds: int = Field(default=180, ge=10, le=900)


class EvaluationRunSummary(BaseModel):
    evaluation_id: str
    run_directory: str
    status: Literal["RUNNING", "SUCCESS", "PARTIAL_SUCCESS", "FAILED", "INTERRUPTED"]
    config: EvaluationRunConfig
    benchmark_hash: str
    total_candidates: int
    completed_candidates: int = 0
    failed_candidates: int = 0
    total_cost_usd: float = 0.0
    dashboard_path: str | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)
