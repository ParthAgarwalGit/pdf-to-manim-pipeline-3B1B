"""Run summaries, storyboard documents, cost ledger, and static HTML report."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .models import LessonPlan, NarrationPlan, RunResult, VerificationFinding
from .observability import RunArtifacts
from .providers import PRICES, PRICING_SNAPSHOT_DATE
from .state import RunStore


def write_storyboard(
    artifacts: RunArtifacts,
    lesson: LessonPlan,
    narration: NarrationPlan,
) -> None:
    payload = {
        "lesson": lesson.model_dump(mode="json"),
        "narration": narration.model_dump(mode="json"),
    }
    artifacts.write_json("storyboard.json", payload)
    narration_by_id = {scene.scene_id: scene for scene in narration.scenes}
    lines = [f"# {lesson.title}", "", "## Objectives"]
    lines.extend(f"- {item}" for item in lesson.objectives)
    lines.extend(["", "## Scenes"])
    for scene in lesson.scenes:
        spoken = narration_by_id.get(scene.scene_id)
        lines.extend([
            "", f"### {scene.scene_id}: {scene.title}", "",
            scene.description, "", f"Learning goal: {scene.learning_goal}", "",
            f"Source facts: {', '.join(scene.source_fact_ids)}", "",
            spoken.narration if spoken else "Narration unavailable.",
        ])
    artifacts.write_text("storyboard.md", "\n".join(lines) + "\n")


def write_costs(artifacts: RunArtifacts, store: RunStore) -> dict[str, Any]:
    rows = store.cost_rows()
    by_stage = defaultdict(float)
    by_provider = defaultdict(float)
    by_model = defaultdict(float)
    for row in rows:
        amount = float(row.get("actual_usd") if row.get("actual_usd") is not None else row["reserved_usd"])
        by_stage[row["stage"]] += amount
        by_provider[row["provider"]] += amount
        by_model[row["model"]] += amount
    payload = {
        "total_usd": round(sum(by_stage.values()), 6),
        "by_stage": {key: round(value, 6) for key, value in by_stage.items()},
        "by_provider": {key: round(value, 6) for key, value in by_provider.items()},
        "by_model": {key: round(value, 6) for key, value in by_model.items()},
        "calls": rows,
        "pricing_snapshot": {
            "date": PRICING_SNAPSHOT_DATE,
            "currency": "USD",
            "text_unit": "per 1M tokens",
            "tts_unit": "per 1M characters",
            "rates": {
                f"{provider}/{model}": {"input": rates[0], "output": rates[1]}
                for (provider, model), rates in sorted(PRICES.items())
            },
        },
    }
    artifacts.write_json("costs.json", payload)
    return payload


def write_html_report(
    artifacts: RunArtifacts,
    store: RunStore,
    result: RunResult,
) -> Path:
    costs = write_costs(artifacts, store)
    task_rows = store.task_rows()
    total_compute = sum(float(row.get("duration_ms") or 0.0) for row in task_rows)
    finished = [row for row in task_rows if row["status"] == "SUCCEEDED"]
    failure_count = sum(1 for row in task_rows if row["status"] == "FAILED")
    task_success_rate = 100.0 * len(finished) / len(task_rows) if task_rows else 100.0
    wall_seconds = _wall_seconds(task_rows)
    critical_path, critical_ms = _critical_path(task_rows)

    stage_stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"tasks": 0, "succeeded": 0, "failed": 0, "duration_ms": 0.0}
    )
    for row in task_rows:
        stage = stage_stats[row["stage"]]
        stage["tasks"] += 1
        stage["succeeded"] += int(row["status"] == "SUCCEEDED")
        stage["failed"] += int(row["status"] == "FAILED")
        stage["duration_ms"] += float(row.get("duration_ms") or 0.0)
    stage_rows = "".join(
        f"<tr><td>{html.escape(stage)}</td><td>{values['tasks']}</td>"
        f"<td>{values['succeeded']}</td><td>{values['failed']}</td>"
        f"<td>{float(values['duration_ms']) / 1000:.2f}s</td></tr>"
        for stage, values in sorted(stage_stats.items())
    )

    scene_rows = []
    for scene in result.scene_results:
        issues = "<br>".join(
            _finding_html(artifacts.run_dir, item)
            for item in scene.findings
        ) or "None"
        links = [
            _artifact_link(artifacts.run_dir, "Narration", artifacts.run_dir / "scenes" / scene.scene_id / "scene.json"),
            _artifact_link(artifacts.run_dir, "Contract", artifacts.run_dir / "scenes" / scene.scene_id / "visual_contract.json"),
            _artifact_link(artifacts.run_dir, "Original code", scene.original_code_path),
            _artifact_link(artifacts.run_dir, "Final code", scene.final_code_path),
            _artifact_link(artifacts.run_dir, "Preview", scene.preview_path),
            _artifact_link(artifacts.run_dir, "Audio", scene.audio_path),
            _artifact_link(artifacts.run_dir, "Final video", scene.final_video_path),
            _artifact_link(
                artifacts.run_dir,
                "Original diagnostics",
                artifacts.run_dir / "scenes" / scene.scene_id / "attempt-00" / "diagnostics.json",
            ),
            _artifact_link(
                artifacts.run_dir,
                "Repair diagnostics",
                artifacts.run_dir / "scenes" / scene.scene_id / "attempt-01" / "diagnostics.json",
            ),
        ]
        representative_frames = " ".join(
            link for link in (
                _artifact_link(artifacts.run_dir, f"Frame {index + 1}", path)
                for index, path in enumerate(scene.frame_paths[:8])
            ) if link
        )
        artifact_links = " | ".join(link for link in links if link)
        if representative_frames:
            artifact_links = f"{artifact_links}<br>{representative_frames}"
        scene_rows.append(
            f"<tr><td>{html.escape(scene.scene_id)}</td><td>{html.escape(scene.status)}</td>"
            f"<td>{'yes' if scene.repaired else 'no'}</td><td>{'yes' if scene.fallback_used else 'no'}</td>"
            f"<td>{issues}</td><td>{artifact_links or 'None'}</td></tr>"
        )
    critical_display = " &rarr; ".join(html.escape(item) for item in critical_path) or "None"
    parallel_ratio = total_compute / (wall_seconds * 1000) if wall_seconds else 0.0
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>pdf2manim V2 report</title>
<style>body{{font:14px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#202124}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #ccc;vertical-align:top;text-align:left}}
th{{background:#f3f4f6}}code{{background:#eee;padding:2px 4px}}.ok{{color:#087830}}.bad{{color:#b3261e}}
a{{color:#0759b5}}small{{color:#5f6368}}</style></head>
<body><h1>pdf2manim V2 run {html.escape(result.run_id)}</h1>
<p>Status: <strong>{html.escape(result.status.value)}</strong></p>
<ul><li>Scenes: {result.successful_scenes}/{result.total_scenes} successful</li>
<li>Tasks: {len(finished)}/{len(task_rows)} successful ({task_success_rate:.1f}%); {failure_count} failed</li>
<li>Estimated/actual API cost: ${costs['total_usd']:.4f}</li>
<li>Wall time covered by tasks: {wall_seconds:.1f}s</li>
<li>Accumulated worker time: {total_compute / 1000:.1f}s ({parallel_ratio:.2f}x wall time)</li></ul>
<p><a href="storyboard.md">Storyboard</a> | <a href="costs.json">Costs</a> |
<a href="events.jsonl">Events</a> | <a href="pipeline.log">Human log</a></p>
<h2>Critical path</h2><p>{critical_display} <small>({critical_ms / 1000:.2f}s accumulated)</small></p>
<h2>Stage activity</h2><table><thead><tr><th>Stage</th><th>Tasks</th><th>Succeeded</th>
<th>Failed</th><th>Worker time</th></tr></thead><tbody>{stage_rows}</tbody></table>
<h2>Scenes</h2><table><thead><tr><th>Scene</th><th>Status</th><th>Repaired</th>
<th>Fallback</th><th>Findings</th><th>Artifacts</th></tr></thead><tbody>{''.join(scene_rows)}</tbody></table>
</body></html>"""
    return artifacts.write_text("report.html", document)


def _artifact_link(run_dir: Path, label: str, value: str | Path | None) -> str:
    if not value:
        return ""
    path = Path(value).resolve()
    if not path.exists():
        return ""
    try:
        relative = path.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return ""
    return f'<a href="{html.escape(quote(relative))}">{html.escape(label)}</a>'


def _finding_html(run_dir: Path, finding: VerificationFinding) -> str:
    details = []
    if finding.timestamp is not None:
        details.append(f"t={finding.timestamp:.2f}s")
    if finding.artifact_ids:
        details.append("artifacts=" + ", ".join(finding.artifact_ids))
    if finding.object_ids:
        details.append("objects=" + ", ".join(finding.object_ids))
    suffix = f" [{' | '.join(details)}]" if details else ""
    text = html.escape(
        f"{finding.severity.upper()} {finding.code}: {finding.message}{suffix}"
    )
    evidence = _artifact_link(run_dir, "evidence frame", finding.screenshot_path)
    return f"{text} ({evidence})" if evidence else text


def _wall_seconds(rows: list[dict[str, Any]]) -> float:
    starts = [_parse_time(row.get("started_at")) for row in rows]
    finishes = [_parse_time(row.get("finished_at")) for row in rows]
    valid_starts = [value for value in starts if value]
    valid_finishes = [value for value in finishes if value]
    if not valid_starts or not valid_finishes:
        return 0.0
    return max(0.0, (max(valid_finishes) - min(valid_starts)).total_seconds())


def _parse_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _critical_path(rows: list[dict[str, Any]]) -> tuple[list[str], float]:
    by_id = {str(row["task_id"]): row for row in rows}
    memo: dict[str, tuple[list[str], float]] = {}
    visiting: set[str] = set()

    def visit(task_id: str) -> tuple[list[str], float]:
        if task_id in memo:
            return memo[task_id]
        if task_id in visiting or task_id not in by_id:
            return [], 0.0
        visiting.add(task_id)
        row = by_id[task_id]
        try:
            dependencies = json.loads(row.get("dependencies_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            dependencies = []
        prefix, prefix_ms = max(
            (visit(str(dependency)) for dependency in dependencies),
            key=lambda item: item[1],
            default=([], 0.0),
        )
        visiting.remove(task_id)
        result = prefix + [task_id], prefix_ms + float(row.get("duration_ms") or 0.0)
        memo[task_id] = result
        return result

    return max((visit(task_id) for task_id in by_id), key=lambda item: item[1], default=([], 0.0))
