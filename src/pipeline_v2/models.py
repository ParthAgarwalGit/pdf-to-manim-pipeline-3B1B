"""Decision contracts shared by every V2 stage."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .prompt_text import DEFAULT_VOICE


MAX_V2_SCENES = 24


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    PAUSED_BUDGET = "PAUSED_BUDGET"
    INTERRUPTED = "INTERRUPTED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    INTERRUPTED = "INTERRUPTED"


class SourceFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    kind: Literal["text", "equation", "image", "table"] = "text"
    bbox: Optional[tuple[float, float, float, float]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SourceDocument(BaseModel):
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    facts: list[SourceFact]
    low_confidence_pages: list[int] = Field(default_factory=list)


class OCRFact(BaseModel):
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    kind: Literal["text", "equation", "image", "table"] = "text"


class OCRBatch(BaseModel):
    facts: list[OCRFact] = Field(default_factory=list)


class LessonScene(BaseModel):
    scene_id: str
    title: str
    description: str
    learning_goal: str
    source_fact_ids: list[str] = Field(min_length=1)
    example: Optional[str] = None
    visual_objective: str = ""
    new_visual_fact: str = ""
    do_not_repeat: list[str] = Field(default_factory=list)
    delivery_mode: Literal["manim", "presenter", "mixed", "transition"] = "manim"


class LessonPlan(BaseModel):
    title: str
    audience: str
    objectives: list[str] = Field(min_length=1)
    prerequisites: list[str] = Field(default_factory=list)
    concept_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    scenes: list[LessonScene] = Field(min_length=1, max_length=MAX_V2_SCENES)

    @field_validator("scenes")
    @classmethod
    def unique_scene_ids(cls, scenes: list[LessonScene]) -> list[LessonScene]:
        ids = [scene.scene_id for scene in scenes]
        if len(ids) != len(set(ids)):
            raise ValueError("scene_id values must be unique")
        return scenes


class NarrationBeat(BaseModel):
    beat_id: str
    text: str = Field(min_length=1)
    source_fact_ids: list[str] = Field(min_length=1)
    role: Literal[
        "introduce", "explain", "demonstrate", "reflect", "bridge", "conclude",
    ] = "explain"
    pause_after_seconds: float = Field(default=0.35, ge=0.0, le=1.5)
    visual_action: str = ""
    continuity_note: str = ""
    start_seconds: float = Field(default=0.0, ge=0.0)
    end_seconds: float = Field(default=0.0, ge=0.0)


class NarratedScene(BaseModel):
    scene_id: str
    narration: str = Field(min_length=1)
    beats: list[NarrationBeat] = Field(min_length=1)
    transition_in: str = ""
    transition_out: str = ""
    target_words_per_minute: int = Field(default=135, ge=110, le=160)
    audio_path: Optional[str] = None
    audio_duration_seconds: Optional[float] = Field(default=None, gt=0.0)
    audio_tail_seconds: float = Field(default=0.0, ge=0.0, le=10.0)
    audio_is_placeholder: bool = False


class NarrationPlan(BaseModel):
    scenes: list[NarratedScene] = Field(min_length=1)


class ArtifactCategory(str, Enum):
    PRIMARY_VISUAL = "primary_visual"
    SEMANTIC_TEXT = "semantic_text"
    ANNOTATION = "annotation"
    TRANSIENT_EFFECT = "transient_effect"


class LayoutTemplate(str, Enum):
    FULL_VISUAL = "full_visual"
    SPLIT = "split"
    THREE_COLUMN = "three_column"
    CENTER_WITH_NOTES = "center_with_notes"


class LayoutRegion(str, Enum):
    HEADER = "header"
    CONTENT_FULL = "content_full"
    CONTENT_LEFT = "content_left"
    CONTENT_CENTER = "content_center"
    CONTENT_RIGHT = "content_right"
    FOOTER = "footer"


class ArtifactFootprint(str, Enum):
    COMPACT = "compact"
    STANDARD = "standard"
    LARGE = "large"


class OverflowPolicy(str, Enum):
    WRAP = "wrap"
    SCALE = "scale"
    REJECT = "reject"


class OverlapPolicy(str, Enum):
    NONE = "none"
    BACKGROUND = "background"
    TARGET_ONLY = "target_only"
    GEOMETRY = "geometry"


class ArtifactSpec(BaseModel):
    artifact_id: str = Field(min_length=1)
    category: ArtifactCategory
    role: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    content: str = ""
    source_fact_ids: list[str] = Field(min_length=1)
    beat_ids: list[str] = Field(min_length=1)
    enter_beat: str
    exit_beat: str
    region: LayoutRegion = LayoutRegion.CONTENT_CENTER
    footprint: ArtifactFootprint = ArtifactFootprint.STANDARD
    preferred_aspect_ratio: Optional[float] = Field(default=None, gt=0.0, le=10.0)
    overflow_policy: OverflowPolicy = OverflowPolicy.SCALE
    minimum_scale: float = Field(default=0.55, ge=0.25, le=1.0)
    region_padding: float = Field(default=0.12, ge=0.0, le=0.4)
    overlap_policy: OverlapPolicy = OverlapPolicy.NONE
    layer: int = Field(default=0, ge=-10, le=20)
    priority: int = Field(default=5, ge=1, le=10)
    parent_id: Optional[str] = None
    target_id: Optional[str] = None
    carry_policy: Literal["remove", "scene_end", "next_scene"] = "remove"
    allowed_overlap_with: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_regions(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        aliases = {
            "full": LayoutRegion.CONTENT_FULL.value,
            "top": LayoutRegion.HEADER.value,
            "bottom": LayoutRegion.FOOTER.value,
            "left": LayoutRegion.CONTENT_LEFT.value,
            "center": LayoutRegion.CONTENT_CENTER.value,
            "right": LayoutRegion.CONTENT_RIGHT.value,
        }
        region = value.get("region")
        if region in aliases:
            value = dict(value)
            value["region"] = aliases[region]
        return value

    @model_validator(mode="after")
    def validate_layout_policy(self) -> "ArtifactSpec":
        if self.overlap_policy == OverlapPolicy.TARGET_ONLY and not self.target_id:
            raise ValueError("target_only overlap_policy requires target_id")
        if (
            self.overlap_policy == OverlapPolicy.BACKGROUND
            and self.category != ArtifactCategory.PRIMARY_VISUAL
        ):
            raise ValueError("background overlap_policy is only valid for primary_visual artifacts")
        return self


class BeatComposition(BaseModel):
    beat_id: str
    active_artifact_ids: list[str]
    focus_artifact_ids: list[str] = Field(default_factory=list)
    camera_intent: Literal["static", "follow", "zoom", "pan"] = "static"


class VisualContract(BaseModel):
    scene_id: str
    layout_template: LayoutTemplate = LayoutTemplate.CENTER_WITH_NOTES
    archetype: Literal[
        "concept", "formula_derivation", "graph_plot", "geometry",
        "algorithm_trace", "comparison", "data_demo", "worked_example",
    ] = "concept"
    artifact_inventory: list[ArtifactSpec] = Field(min_length=1)
    composition_plan: list[BeatComposition] = Field(min_length=1)
    visual_progression: list[str] = Field(default_factory=list)
    avoid_repeating: list[str] = Field(default_factory=list)
    style_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "VisualContract":
        ids = [artifact.artifact_id for artifact in self.artifact_inventory]
        if len(ids) != len(set(ids)):
            raise ValueError("artifact_id values must be unique")
        known = set(ids)
        for artifact in self.artifact_inventory:
            for reference in (
                artifact.parent_id,
                artifact.target_id,
                *artifact.allowed_overlap_with,
            ):
                if reference and reference not in known:
                    raise ValueError(f"unknown artifact reference: {reference}")
        for beat in self.composition_plan:
            unknown = set(beat.active_artifact_ids + beat.focus_artifact_ids) - known
            if unknown:
                raise ValueError(f"composition references unknown artifacts: {sorted(unknown)}")
        return self


class VisualContractBatch(BaseModel):
    contracts: list[VisualContract] = Field(min_length=1, max_length=3)


class TimelineSegment(BaseModel):
    segment_id: str
    scene_id: str
    mode: Literal["manim", "presenter", "mixed", "transition"] = "manim"
    start_seconds: float = Field(ge=0.0)
    duration_seconds: float = Field(gt=0.0)
    video_path: Optional[str] = None
    audio_path: Optional[str] = None


class VerificationFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"]
    message: str
    timestamp: Optional[float] = None
    artifact_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    screenshot_path: Optional[str] = None
    code_lines: list[int] = Field(default_factory=list)


class SceneResult(BaseModel):
    scene_id: str
    success: bool = False
    status: str = "PENDING"
    original_code_path: Optional[str] = None
    final_code_path: Optional[str] = None
    preview_path: Optional[str] = None
    render_video_path: Optional[str] = None
    final_video_path: Optional[str] = None
    audio_path: Optional[str] = None
    repaired: bool = False
    fallback_used: bool = False
    findings: list[VerificationFinding] = Field(default_factory=list)
    frame_paths: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class PipelineV2Config(BaseModel):
    output_dir: Path = Path("output")
    quality: Literal["low", "medium", "high"] = "medium"
    audience: str = "undergraduate"
    voice: str = DEFAULT_VOICE
    tts_provider: Literal["gemini", "openai"] = "gemini"
    policy: Literal["standard", "strict"] = "standard"
    repair: Literal["off", "eligible"] = "eligible"
    verifier_mode: Literal["enforce", "audit"] = "enforce"
    max_cost_usd: float = Field(default=3.0, gt=0.0)
    min_scenes: Optional[int] = Field(default=None, ge=1, le=MAX_V2_SCENES)
    max_scenes: int = Field(default=12, ge=1, le=MAX_V2_SCENES)
    pacing: Literal["standard", "relaxed"] = "standard"
    enable_tts: bool = True
    enable_render: bool = True
    dry_run: bool = False
    strict_keys: bool = False
    global_api_workers: int = Field(default=4, ge=1, le=16)
    visual_workers: int = Field(default=2, ge=1, le=8)
    tts_workers: int = Field(default=4, ge=1, le=8)
    code_workers: int = Field(default=3, ge=1, le=8)
    preview_workers: int = Field(default=2, ge=1, le=4)
    repair_workers: int = Field(default=1, ge=1, le=2)
    render_workers: int = Field(default=1, ge=1, le=2)
    max_repairs_per_scene: int = Field(default=3, ge=0, le=3)
    preview_timeout_seconds: int = Field(default=120, ge=10)
    render_timeout_seconds: int = Field(default=600, ge=30)
    # Generated Manim always runs with scrubbed secrets in a scratch directory;
    # "docker" adds a networkless, resource-limited container.
    render_sandbox: Literal["native", "docker"] = "native"
    render_sandbox_image: str = "pdf2manim:latest"
    enable_delivery_checks: bool = True
    max_delivery_repair_rounds: int = Field(default=2, ge=0, le=2)
    # Professor controls. Stored as plain data so run.json round-trips them and
    # task hashes of runs without controls stay unchanged.
    target_minutes: Optional[float] = Field(default=None, gt=0.0)
    directives: Optional[dict[str, Any]] = None
    concept_plan: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_scene_range(self) -> "PipelineV2Config":
        if self.min_scenes is not None and self.min_scenes > self.max_scenes:
            raise ValueError("min_scenes cannot exceed max_scenes")
        return self


class RunResult(BaseModel):
    run_id: str
    run_directory: str
    status: RunStatus
    total_scenes: int = 0
    successful_scenes: int = 0
    failed_scenes: int = 0
    estimated_cost_usd: float = 0.0
    final_video_path: Optional[str] = None
    report_path: Optional[str] = None
    scene_results: list[SceneResult] = Field(default_factory=list)
    message: str = ""
    delivery_status: Optional[Literal["ready", "needs_review"]] = None
    delivery_report_path: Optional[str] = None
    delivery_summary: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
