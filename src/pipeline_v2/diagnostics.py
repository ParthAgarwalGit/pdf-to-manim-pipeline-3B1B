"""Deterministic runtime visual and lifecycle diagnostics."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from .layout import HARD_BOUNDARY_BUFFER, region_bounds
from .models import (
    ArtifactCategory,
    NarratedScene,
    OverlapPolicy,
    VerificationFinding,
    VisualContract,
)

from PIL import Image, ImageStat


ERROR_WEIGHT = 100
WARNING_WEIGHT = 3
# Never drawn: a ValueTracker keeps its value as a point's x coordinate, so an
# angle of 30 would otherwise read as an object 30 units off screen.
INVISIBLE_CLASSES = frozenset({"ValueTracker", "ComplexValueTracker"})


def issue_score(findings: list[VerificationFinding]) -> int:
    return sum(ERROR_WEIGHT if item.severity == "error" else WARNING_WEIGHT for item in findings)


def frame_fingerprint(frame_paths: list[str]) -> str:
    hashes = []
    for value in frame_paths:
        try:
            with Image.open(value) as image:
                pixels = list(image.convert("L").resize((16, 16)).getdata())
            average = sum(pixels) / len(pixels)
            hashes.append("".join("1" if pixel >= average else "0" for pixel in pixels))
        except Exception:
            continue
    return hashlib.sha256("|".join(hashes).encode("ascii")).hexdigest() if hashes else ""


def frame_sequence_similarity(left_paths: list[str], right_paths: list[str]) -> float:
    """Estimate repeated visual sequences without model calls.

    This is deliberately advisory: it compares aligned low-resolution frames and
    ignores pairs with almost no visible contrast.
    """

    similarities = []
    for left_path, right_path in zip(left_paths[:8], right_paths[:8]):
        try:
            with Image.open(left_path) as left_source, Image.open(right_path) as right_source:
                left = left_source.convert("L").resize((64, 36))
                right = right_source.convert("L").resize((64, 36))
                if max(ImageStat.Stat(left).stddev[0], ImageStat.Stat(right).stddev[0]) < 8.0:
                    continue
                difference = sum(
                    abs(a - b) for a, b in zip(left.tobytes(), right.tobytes())
                )
                similarities.append(1.0 - difference / (255.0 * 64.0 * 36.0))
        except Exception:
            continue
    return sum(similarities) / len(similarities) if similarities else 0.0


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def analyze_trace(
    contract: VisualContract,
    scene: NarratedScene,
    trace_path: Path,
    video_duration: float,
    *,
    enforce_contract: bool = True,
) -> list[VerificationFinding]:
    records = read_trace(trace_path)
    if not records:
        return [VerificationFinding(
            code="TRACE_MISSING", severity="error",
            message="The preview produced no runtime artifact checkpoints.",
        )]
    expected_by_beat = (
        {item.beat_id: set(item.active_artifact_ids) for item in contract.composition_plan}
        if enforce_contract else {}
    )
    artifacts = {item.artifact_id: item for item in contract.artifact_inventory}
    beat_order = {beat.beat_id: index for index, beat in enumerate(scene.beats)}
    findings: list[VerificationFinding] = []
    seen_keys: set[tuple] = set()
    seen_active_by_beat: dict[str, set[str]] = {}
    last_timestamp_by_beat: dict[str, float] = {}

    for record in records:
        beat_id = record.get("beat_id") or ""
        event = record.get("event") or ""
        if event in {"beat_late", "scene_overrun"}:
            _sync_finding(findings, seen_keys, event, record.get("sync"))
            continue
        if event not in {"add", "remove", "play", "play_sample", "wait", "final"}:
            continue
        expected = expected_by_beat.get(beat_id, set())
        objects = record.get("objects") or []
        active = {item.get("artifact_id") for item in objects if item.get("artifact_id")}
        seen_active = seen_active_by_beat.setdefault(beat_id, set())
        seen_active.update(active)
        timestamp = float(record.get("time") or 0.0)
        last_timestamp_by_beat[beat_id] = timestamp
        for item in objects:
            artifact_id = item.get("artifact_id")
            if not artifact_id:
                bounds = item.get("bounds") or {}
                if bounds.get("width", 0.0) <= 0.001 and bounds.get("height", 0.0) <= 0.001:
                    continue
                if enforce_contract:
                    _add(findings, seen_keys, "UNPLANNED_OBJECT", "error",
                         f"An unregistered {item.get('class_name', 'mobject')} is visible.", timestamp, [])
                _bounds_checks(findings, seen_keys, item, timestamp)
                continue
            if enforce_contract and artifact_id not in expected and beat_id in expected_by_beat:
                artifact = artifacts.get(artifact_id)
                if artifact:
                    current = beat_order.get(beat_id, -1)
                    first = min((beat_order.get(value, 10**6) for value in artifact.beat_ids), default=10**6)
                    last = max((beat_order.get(value, -1) for value in artifact.beat_ids), default=-1)
                    code = "EARLY_ENTRY" if current < first else "LATE_EXIT" if current > last else "UNEXPECTED_ARTIFACT_ACTIVE"
                else:
                    code = "UNPLANNED_OBJECT"
                _add(findings, seen_keys, code, "error",
                     f"{artifact_id} is visible during {beat_id} but is not expected.", timestamp, [artifact_id])
            _bounds_checks(findings, seen_keys, item, timestamp)
            artifact = artifacts.get(artifact_id)
            if enforce_contract and artifact:
                _region_bounds_checks(
                    findings, seen_keys, item, artifact, contract, timestamp,
                )

        _overlap_checks(
            findings,
            seen_keys,
            objects,
            artifacts,
            timestamp,
            record.get("pixel_overlaps") or [],
        )
        density = sum(_density_weight(artifacts.get(item)) for item in active)
        if density > 12:
            _add(findings, seen_keys, "VISUAL_DENSITY_HIGH", "warning",
                 f"Weighted active visual density is {density:.1f}; target is at most 12.", timestamp, sorted(active))

    # Contracts describe presence over a semantic beat, not after every
    # animation within it. An artifact may legitimately enter later in a beat.
    for beat_id, expected in expected_by_beat.items():
        missing_ids = expected - seen_active_by_beat.get(beat_id, set())
        for missing in missing_ids:
            _add(
                findings,
                seen_keys,
                "MISSING_OBJECT",
                "error",
                f"Expected artifact {missing} never appears during {beat_id}.",
                last_timestamp_by_beat.get(beat_id, 0.0),
                [missing],
            )

    if scene.audio_duration_seconds:
        delta = abs(video_duration - scene.audio_duration_seconds)
        tolerance = max(1.0, scene.audio_duration_seconds * 0.10)
        if delta > tolerance:
            _add(findings, seen_keys, "AUDIO_DURATION_MISMATCH", "error",
                 f"Preview is {video_duration:.2f}s but narration is {scene.audio_duration_seconds:.2f}s.",
                 video_duration, [])
    return findings


def _sync_finding(findings, seen, event, sync):
    """Report late narration sync recorded by TrackedScene.

    Always a warning: an error would trigger paid vision repair and fail the
    strict policy, and a late beat is something to report, not to regenerate.
    """

    try:
        scheduled = float(sync["scheduled_seconds"])
        actual = float(sync["actual_seconds"])
        lag = float(sync["lag_seconds"])
    except (KeyError, TypeError, ValueError):
        return
    if event == "beat_late":
        _add(findings, seen, "BEAT_LATE", "warning",
             f"Beat {sync.get('beat_id') or 'unknown'} starts {lag:.2f}s late: its narration "
             f"begins at {scheduled:.2f}s but its animation begins at {actual:.2f}s.",
             actual, [])
    else:
        _add(findings, seen, "SCENE_OVERRUN", "warning",
             f"The animation runs {lag:.2f}s past the end of the narration at {scheduled:.2f}s.",
             actual, [])


def _bounds_checks(findings, seen, item, timestamp):
    bounds = item.get("bounds")
    if not bounds or item.get("class_name") in INVISIBLE_CLASSES:
        return
    artifact_id = item.get("artifact_id")
    text_like = _is_text(item)
    outside_frame = (
        bounds["left"] < -7.11 or bounds["right"] > 7.11
        or bounds["bottom"] < -4.0 or bounds["top"] > 4.0
    )
    near_edge = (
        bounds["left"] < -6.9 or bounds["right"] > 6.9
        or bounds["bottom"] < -3.8 or bounds["top"] > 3.8
    )
    if outside_frame:
        _add(findings, seen, "TEXT_FRAME_COLLISION" if text_like else "OFFSCREEN_OBJECT", "error",
             f"{artifact_id} crosses the safe frame boundary.", timestamp, [artifact_id] if artifact_id else [])
    elif near_edge:
        _add(findings, seen, "FRAME_EDGE_PROXIMITY", "warning",
             f"{artifact_id} is close to the frame edge but remains visible.",
             timestamp, [artifact_id] if artifact_id else [])
    if text_like and bounds.get("height", 1) < 0.12:
        preview_pixels = bounds.get("height", 0.0) * 60.0
        _add(findings, seen, "TEXT_TOO_SMALL", "warning",
             f"{artifact_id} is only about {preview_pixels:.1f}px high in a 480p preview.",
             timestamp, [artifact_id] if artifact_id else [])


def _region_bounds_checks(findings, seen, item, artifact, contract, timestamp):
    bounds = item.get("bounds")
    if not bounds:
        return
    try:
        preferred = region_bounds(contract.layout_template, artifact.region).inset(
            artifact.region_padding
        )
        hard = region_bounds(contract.layout_template, artifact.region).expand(
            HARD_BOUNDARY_BUFFER
        )
    except ValueError:
        return
    artifact_id = artifact.artifact_id
    object_id = item.get("object_id") or artifact_id
    if _outside(bounds, hard):
        _add(
            findings, seen, "REGION_HARD_OVERFLOW", "error",
            f"{object_id} leaves hard boundary {artifact.region.value!r} in "
            f"{contract.layout_template.value!r}.",
            timestamp, [artifact_id], [object_id],
        )
    elif _outside(bounds, preferred):
        _add(
            findings, seen, "REGION_SOFT_OVERFLOW", "warning",
            f"{object_id} enters the buffer around preferred region {artifact.region.value!r}.",
            timestamp, [artifact_id], [object_id],
        )


def _outside(bounds, region) -> bool:
    tolerance = 0.015
    return (
        bounds["left"] < region.left - tolerance
        or bounds["right"] > region.right + tolerance
        or bounds["bottom"] < region.bottom - tolerance
        or bounds["top"] > region.top + tolerance
    )


def _overlap_checks(findings, seen, objects, artifacts, timestamp, pixel_overlaps):
    pixel_ratios = {
        frozenset((item.get("left_object_id"), item.get("right_object_id"))): float(item["ratio"])
        for item in pixel_overlaps
        if item.get("left_object_id") and item.get("right_object_id")
        and item.get("ratio") is not None
    }
    for index, left in enumerate(objects):
        left_id = left.get("artifact_id")
        left_object_id = left.get("object_id") or left_id
        left_bounds = left.get("bounds")
        if not left_id or not left_bounds or (
            left_bounds.get("width", 0.0) <= 0.001 and left_bounds.get("height", 0.0) <= 0.001
        ):
            continue
        for right in objects[index + 1:]:
            right_id = right.get("artifact_id")
            right_object_id = right.get("object_id") or right_id
            right_bounds = right.get("bounds")
            if not right_id or right_object_id == left_object_id or not right_bounds or (
                right_bounds.get("width", 0.0) <= 0.001 and right_bounds.get("height", 0.0) <= 0.001
            ):
                continue
            if left.get("runtime_id") and left.get("runtime_id") == right.get("runtime_id"):
                continue
            bbox_ratio = _overlap_ratio(left_bounds, right_bounds)
            if bbox_ratio <= 0.0:
                continue
            left_spec = artifacts.get(left_id)
            right_spec = artifacts.get(right_id)
            left_text = _is_text(left)
            right_text = _is_text(right)
            if (
                left_spec
                and right_spec
                and _intentional_overlap(left_spec, right_spec, left_text, right_text)
            ):
                continue
            if left_id == right_id and left_text != right_text:
                continue
            pixel_ratio = pixel_ratios.get(frozenset((left_object_id, right_object_id)))
            object_ids = [left_object_id, right_object_id]
            artifact_ids = sorted({left_id, right_id})
            if left_text and right_text:
                effective_ratio = pixel_ratio if pixel_ratio is not None else bbox_ratio
                if _same_visual_alias(left_bounds, right_bounds, pixel_ratio):
                    continue
                warning_threshold = 0.05 if pixel_ratio is not None else 0.12
                error_threshold = 0.22 if pixel_ratio is not None else 0.35
                if effective_ratio < warning_threshold:
                    continue
                severity = "error" if effective_ratio >= error_threshold else "warning"
                detail = _overlap_detail(bbox_ratio, pixel_ratio)
                _add(
                    findings, seen, "TEXT_TEXT_OVERLAP", severity,
                    f"Text objects {left_object_id} and {right_object_id} overlap ({detail}).",
                    timestamp, artifact_ids, object_ids,
                )
            elif left_text or right_text:
                effective_ratio = pixel_ratio if pixel_ratio is not None else bbox_ratio
                threshold = 0.08 if pixel_ratio is not None else 0.25
                if effective_ratio < threshold:
                    continue
                detail = _overlap_detail(bbox_ratio, pixel_ratio)
                _add(
                    findings, seen, "TEXT_GEOMETRY_OVERLAP", "warning",
                    f"Text and geometry objects {left_object_id} and {right_object_id} overlap ({detail}).",
                    timestamp, artifact_ids, object_ids,
                )


def _intentional_overlap(left, right, left_text: bool, right_text: bool) -> bool:
    if right.artifact_id in left.allowed_overlap_with:
        return True
    if left.artifact_id in right.allowed_overlap_with:
        return True
    if left.overlap_policy == OverlapPolicy.TARGET_ONLY and left.target_id == right.artifact_id:
        return True
    if right.overlap_policy == OverlapPolicy.TARGET_ONLY and right.target_id == left.artifact_id:
        return True
    if left.overlap_policy == OverlapPolicy.BACKGROUND and not left_text:
        return True
    return right.overlap_policy == OverlapPolicy.BACKGROUND and not right_text


def _overlap_ratio(a, b) -> float:
    x_overlap = min(a["right"], b["right"]) - max(a["left"], b["left"])
    y_overlap = min(a["top"], b["top"]) - max(a["bottom"], b["bottom"])
    if x_overlap <= 0.0 or y_overlap <= 0.0:
        return 0.0
    area_a = max(0.0, a["right"] - a["left"]) * max(0.0, a["top"] - a["bottom"])
    area_b = max(0.0, b["right"] - b["left"]) * max(0.0, b["top"] - b["bottom"])
    smaller_area = min(area_a, area_b)
    return (x_overlap * y_overlap) / smaller_area if smaller_area > 0.0 else 0.0


def _overlap_detail(bbox_ratio: float, pixel_ratio: float | None) -> str:
    if pixel_ratio is None:
        return f"{bbox_ratio:.0%} of the smaller bounding box"
    return f"{bbox_ratio:.0%} bounds, {pixel_ratio:.0%} occupied pixels"


def _same_visual_alias(left, right, pixel_ratio: float | None) -> bool:
    """Ignore duplicate references to the same rendered glyphs during transforms."""

    if pixel_ratio is None or pixel_ratio < 0.95:
        return False
    return all(
        abs(float(left[key]) - float(right[key])) <= 0.015
        for key in ("left", "right", "top", "bottom")
    )


def _is_text(item) -> bool:
    class_name = item.get("class_name") or ""
    return (
        item.get("category") == ArtifactCategory.SEMANTIC_TEXT.value
        or class_name in {
            "Code", "DecimalNumber", "Integer", "MarkupText", "MathTex", "Paragraph",
            "SingleStringMathTex", "Tex", "Text", "Title", "Variable",
        }
        or class_name.endswith(("Text", "Tex"))
    )


def _density_weight(artifact) -> float:
    if not artifact:
        return 1.0
    return {
        ArtifactCategory.PRIMARY_VISUAL: 2.0,
        ArtifactCategory.SEMANTIC_TEXT: 2.0,
        ArtifactCategory.ANNOTATION: 1.0,
        ArtifactCategory.TRANSIENT_EFFECT: 0.5,
    }[artifact.category]


def _add(findings, seen, code, severity, message, timestamp, artifact_ids, object_ids=None):
    object_ids = object_ids or []
    key = (code, message, tuple(sorted(artifact_ids)), tuple(sorted(object_ids)))
    if key in seen:
        return
    seen.add(key)
    findings.append(VerificationFinding(
        code=code, severity=severity, message=message,
        timestamp=round(timestamp, 3), artifact_ids=artifact_ids, object_ids=object_ids,
    ))
