"""Command-line interface for the narration-directed V2 pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import click

from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.models import MAX_V2_SCENES


def _load_env_files() -> None:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parent.parent
    for name in (".env", "keys.env"):
        path = project_root / name
        if path.is_file():
            load_dotenv(path)
            break


def _pdf_path(_ctx, _param, value: str) -> str:
    path = Path(value).resolve()
    if not path.is_file():
        raise click.BadParameter(f"PDF file not found: {value}")
    if path.suffix.lower() != ".pdf":
        raise click.BadParameter("Input must be a PDF file")
    return str(path)


@click.group()
@click.version_option(version="2.0.0", prog_name="pdf2manim")
def pdf2manim() -> None:
    """Convert educational PDFs into verified narrated Manim videos."""
    _load_env_files()


@pdf2manim.command()
@click.argument("input_pdf", callback=_pdf_path)
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), default=Path("output"))
@click.option("-q", "--quality", type=click.Choice(["low", "medium", "high"]), default="medium")
@click.option("--audience", default="undergraduate", show_default=True)
@click.option("--voice", type=click.Choice(["alloy", "echo", "fable", "onyx", "nova", "shimmer"]), default="alloy")
@click.option("--policy", type=click.Choice(["standard", "strict"]), default="standard")
@click.option("--strict", is_flag=True, help="Alias for --policy strict.")
@click.option("--repair", type=click.Choice(["off", "eligible"]), default="eligible")
@click.option(
    "--max-repairs-per-scene",
    type=click.IntRange(0, 3),
    default=3,
    show_default=True,
    help="Maximum sequential improving repair attempts for each scene.",
)
@click.option(
    "--verifier-mode",
    type=click.Choice(["enforce", "audit"]),
    default="enforce",
    show_default=True,
    help="Audit logs visual findings without rejecting renderable scenes.",
)
@click.option("--max-cost", type=click.FloatRange(min=0.01), default=1.0, show_default=True)
@click.option(
    "--min-scenes",
    type=click.IntRange(1, MAX_V2_SCENES),
    default=None,
    help="Required scene floor; replans once and fails instead of silently producing fewer.",
)
@click.option(
    "--max-scenes",
    type=click.IntRange(1, MAX_V2_SCENES),
    default=12,
    show_default=True,
    help="Scene ceiling; values above 12 request full coverage for long documents.",
)
@click.option(
    "--pacing",
    type=click.Choice(["standard", "relaxed"]),
    default="standard",
    show_default=True,
    help="Use less-dense narration and slower speech in relaxed mode.",
)
@click.option("--skip-tts", is_flag=True)
@click.option("--skip-render", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Stop after grounded lesson and narration planning.")
@click.option("--strict-keys", is_flag=True, help="Disallow deterministic fallbacks for missing provider keys.")
@click.option("--api-workers", type=click.IntRange(1, 16), default=4, show_default=True)
@click.option("--code-workers", type=click.IntRange(1, 8), default=3, show_default=True)
@click.option("--preview-workers", type=click.IntRange(1, 4), default=2, show_default=True)
@click.option("--render-workers", type=click.IntRange(1, 2), default=1, show_default=True)
def convert(
    input_pdf: str,
    output_dir: Path,
    quality: str,
    audience: str,
    voice: str,
    policy: str,
    strict: bool,
    repair: str,
    max_repairs_per_scene: int,
    verifier_mode: str,
    max_cost: float,
    min_scenes: Optional[int],
    max_scenes: int,
    pacing: str,
    skip_tts: bool,
    skip_render: bool,
    dry_run: bool,
    strict_keys: bool,
    api_workers: int,
    code_workers: int,
    preview_workers: int,
    render_workers: int,
) -> None:
    """Run a new V2 conversion for INPUT_PDF."""
    effective_policy = "strict" if strict else policy
    if min_scenes is not None and min_scenes > max_scenes:
        raise click.UsageError("--min-scenes cannot exceed --max-scenes")
    config = PipelineV2Config(
        output_dir=output_dir,
        quality=quality,
        audience=audience,
        voice=voice,
        policy=effective_policy,
        repair=repair,
        max_repairs_per_scene=max_repairs_per_scene,
        verifier_mode=verifier_mode,
        max_cost_usd=max_cost,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
        pacing=pacing,
        enable_tts=not skip_tts,
        enable_render=not skip_render,
        dry_run=dry_run,
        strict_keys=strict_keys,
        global_api_workers=api_workers,
        code_workers=code_workers,
        preview_workers=preview_workers,
        render_workers=render_workers,
    )
    click.echo("pdf2manim V2")
    click.echo(f"  Input:       {input_pdf}")
    click.echo(f"  Policy:      {effective_policy}")
    click.echo(f"  Quality:     {quality}")
    click.echo(f"  API budget:  ${max_cost:.2f}")
    click.echo(f"  Repair:      {repair}")
    click.echo(f"  Repair cap:  {max_repairs_per_scene} per scene")
    click.echo(f"  Verifier:    {verifier_mode}")
    click.echo(f"  Scene range: {min_scenes or 'auto'}-{max_scenes}")
    click.echo(f"  Pacing:      {pacing}")
    click.echo(f"  API workers: {api_workers} (code {code_workers}, preview {preview_workers})")
    result = PipelineV2(config).run(input_pdf)
    _print_result(result)


@pdf2manim.command("resume")
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--max-cost", type=click.FloatRange(min=0.01), default=None)
def resume_command(run_directory: Path, max_cost: Optional[float]) -> None:
    """Resume an interrupted or budget-paused run without repeating successful tasks."""
    result = PipelineV2.resume(run_directory, max_cost_usd=max_cost)
    _print_result(result)


@pdf2manim.command("retry-scenes")
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("scene_ids", nargs=-1, required=True)
@click.option(
    "--max-cost",
    type=click.FloatRange(min=0.01),
    default=None,
    help="New total run budget ceiling, including money already spent.",
)
def retry_scenes_command(
    run_directory: Path,
    scene_ids: tuple[str, ...],
    max_cost: Optional[float],
) -> None:
    """Regenerate selected SCENE_IDS in one existing run and rebuild its final video."""

    click.echo("pdf2manim V2 scene retry")
    click.echo(f"  Run:     {run_directory.resolve()}")
    click.echo(f"  Scenes:  {', '.join(scene_ids)}")
    click.echo("  Reusing: lesson, narration, TTS, and all unselected scene videos")
    try:
        result = PipelineV2.retry_scenes(
            run_directory,
            list(scene_ids),
            max_cost_usd=max_cost,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_result(result)


@pdf2manim.command()
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def status(run_directory: Path) -> None:
    """Print the persisted status and cost summary for a run."""
    run_path = run_directory / "run.json"
    if not run_path.exists():
        raise click.ClickException(f"run.json not found in {run_directory}")
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    costs_path = run_directory / "costs.json"
    costs = json.loads(costs_path.read_text(encoding="utf-8")) if costs_path.exists() else {}
    click.echo(f"Run:       {payload.get('run_id', run_directory.name)}")
    click.echo(f"Status:    {payload.get('status', 'UNKNOWN')}")
    click.echo(f"Input:     {payload.get('input_pdf', 'unknown')}")
    click.echo(f"API cost:  ${float(costs.get('total_usd', 0.0)):.4f}")
    result = payload.get("result") or {}
    if result:
        click.echo(f"Scenes:    {result.get('successful_scenes', 0)}/{result.get('total_scenes', 0)}")
        click.echo(f"Final:     {result.get('final_video_path') or 'not available'}")
        scene_results = result.get("scene_results") or []
        if scene_results:
            click.echo("Scene IDs:")
            for scene in scene_results:
                markers = []
                if scene.get("repaired"):
                    markers.append("repaired")
                if scene.get("fallback_used"):
                    markers.append("fallback")
                suffix = f" ({', '.join(markers)})" if markers else ""
                click.echo(
                    f"  {scene.get('scene_id', 'unknown')}: "
                    f"{scene.get('status', 'UNKNOWN')}{suffix}"
                )


@pdf2manim.command()
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report(run_directory: Path) -> None:
    """Print the path to a run's local HTML report."""
    path = (run_directory / "report.html").resolve()
    if not path.exists():
        raise click.ClickException(f"Report has not been generated: {path}")
    click.echo(str(path))


@pdf2manim.group("eval")
def eval_group() -> None:
    """Build and run frozen scene-level model evaluations."""


@eval_group.command("build")
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--benchmark",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("evals/benchmarks/scene-v1"),
    show_default=True,
)
@click.option("--scene", "scene_ids", multiple=True, help="Scene ID to include; repeat as needed.")
@click.option("--category", default="uncategorized", show_default=True)
def eval_build(
    run_directory: Path,
    benchmark: Path,
    scene_ids: tuple[str, ...],
    category: str,
) -> None:
    """Create immutable scene cases from an existing V2 RUN_DIRECTORY."""
    from src.evaluation.cases import build_benchmark

    try:
        manifest = build_benchmark(
            run_directory,
            benchmark,
            scene_ids=list(scene_ids) or None,
            category=category,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Benchmark: {benchmark.resolve()}")
    click.echo(f"Cases:     {len(manifest.case_files)}")
    click.echo(f"Hash:      {manifest.benchmark_hash}")


@eval_group.command("validate")
@click.argument("benchmark", type=click.Path(exists=True, file_okay=False, path_type=Path))
def eval_validate(benchmark: Path) -> None:
    """Validate schemas, copied assets, and immutable benchmark hashes."""
    from src.evaluation.cases import load_benchmark

    try:
        manifest, cases = load_benchmark(benchmark)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    repair_ready = sum(bool(case.initial_code_path) for case in cases)
    click.echo(f"Benchmark:    {manifest.benchmark_id}")
    click.echo(f"Cases:        {len(cases)}")
    click.echo(f"Repair-ready: {repair_ready}")
    click.echo(f"Hash:         {manifest.benchmark_hash}")
    click.echo("Validation:   passed")


@eval_group.command("plan")
@click.argument("benchmark", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model", "models", multiple=True, default=("gemini",), show_default=True)
@click.option("--track", type=click.Choice(["generator", "repair", "production"]), default="generator")
@click.option("--replicates", type=click.IntRange(1, 5), default=1, show_default=True)
def eval_plan(benchmark: Path, models: tuple[str, ...], track: str, replicates: int) -> None:
    """Print the exact candidate and API-call plan without making calls."""
    from src.evaluation.cases import load_benchmark
    from src.evaluation.models import EvaluationTrack
    from src.evaluation.runner import estimate_max_cost, parse_model_spec

    try:
        manifest, cases = load_benchmark(benchmark)
        specs = [parse_model_spec(value) for value in models]
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    candidates = len(cases) * len(specs) * replicates
    max_calls = candidates * (2 if track == "production" else 1)
    max_cost = estimate_max_cost(cases, specs, EvaluationTrack(track), replicates)
    click.echo(f"Benchmark:       {manifest.benchmark_id}")
    click.echo(f"Track:           {track}")
    click.echo(f"Cases:           {len(cases)}")
    click.echo(f"Models:          {', '.join(item.alias for item in specs)}")
    click.echo(f"Replicates:      {replicates}")
    click.echo(f"Candidates:      {candidates}")
    click.echo(f"Maximum calls:   {max_calls}")
    click.echo(f"Max reservation: ${max_cost:.4f}")
    click.echo("Paid calls made: 0")


@eval_group.command("run")
@click.argument("benchmark", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model", "models", multiple=True, default=("gemini",), show_default=True)
@click.option("--track", type=click.Choice(["generator", "repair", "production"]), default="generator")
@click.option("--replicates", type=click.IntRange(1, 5), default=1, show_default=True)
@click.option("--quality", type=click.Choice(["low", "medium"]), default="low", show_default=True)
@click.option("--max-cost", type=click.FloatRange(min=0.01), required=True)
@click.option("--workers", type=click.IntRange(1, 8), default=2, show_default=True)
@click.option("--max-repairs", type=click.IntRange(0, 1), default=1, show_default=True)
@click.option("--timeout", "timeout_seconds", type=click.IntRange(10, 900), default=180)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("evals/runs"),
    show_default=True,
)
def eval_run(
    benchmark: Path,
    models: tuple[str, ...],
    track: str,
    replicates: int,
    quality: str,
    max_cost: float,
    workers: int,
    max_repairs: int,
    timeout_seconds: int,
    output_root: Path,
) -> None:
    """Execute a bounded evaluation using frozen scene inputs."""
    from src.evaluation.models import EvaluationRunConfig, EvaluationTrack
    from src.evaluation.runner import EvaluationRunner, parse_model_spec

    try:
        specs = [parse_model_spec(value) for value in models]
        config = EvaluationRunConfig(
            benchmark_path=benchmark.resolve(),
            output_root=output_root.resolve(),
            track=EvaluationTrack(track),
            models=specs,
            replicates=replicates,
            quality=quality,
            max_cost_usd=max_cost,
            workers=workers,
            max_repairs=max_repairs,
            timeout_seconds=timeout_seconds,
        )
        result = EvaluationRunner(config).run()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print_evaluation_result(result)


@eval_group.command("resume")
@click.argument("evaluation_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--max-cost", type=click.FloatRange(min=0.01), default=None)
@click.option(
    "--retry-interrupted",
    is_flag=True,
    help="Explicitly retry interrupted candidates that may have reached a provider.",
)
def eval_resume(
    evaluation_directory: Path,
    max_cost: Optional[float],
    retry_interrupted: bool,
) -> None:
    """Resume unfinished candidates without repeating completed calls."""
    from src.evaluation.runner import EvaluationRunner

    try:
        result = EvaluationRunner.resume(
            evaluation_directory,
            max_cost_usd=max_cost,
            retry_interrupted=retry_interrupted,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print_evaluation_result(result)


@eval_group.command("report")
@click.argument("evaluation_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def eval_report(evaluation_directory: Path) -> None:
    """Regenerate and print the evaluation dashboard path."""
    from src.evaluation.dashboard import generate_dashboard
    from src.evaluation.models import EvaluationRunSummary

    run_path = evaluation_directory / "run.json"
    if not run_path.is_file():
        raise click.ClickException(f"run.json not found in {evaluation_directory}")
    summary = EvaluationRunSummary.model_validate_json(run_path.read_text(encoding="utf-8"))
    path = generate_dashboard(evaluation_directory, summary)
    click.echo(str(path.resolve()))


@eval_group.command("status")
@click.argument("evaluation_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def eval_status(evaluation_directory: Path) -> None:
    """Print candidate progress and cost for an evaluation run."""
    from src.evaluation.models import EvaluationRunSummary

    run_path = evaluation_directory / "run.json"
    if not run_path.is_file():
        raise click.ClickException(f"run.json not found in {evaluation_directory}")
    summary = EvaluationRunSummary.model_validate_json(run_path.read_text(encoding="utf-8"))
    _print_evaluation_result(summary)


@eval_group.command("review")
@click.argument("evaluation_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--port", type=click.IntRange(1024, 65535), default=8765, show_default=True)
@click.option("--no-open", is_flag=True, help="Do not open the dashboard in the default browser.")
def eval_review(evaluation_directory: Path, port: int, no_open: bool) -> None:
    """Serve the dashboard locally and persist human ratings."""
    from src.evaluation.dashboard import serve_dashboard

    try:
        serve_dashboard(evaluation_directory, port=port, open_browser=not no_open)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


def _print_result(result) -> None:
    click.echo()
    click.echo(f"Status:       {result.status.value}")
    click.echo(f"Run:          {result.run_directory}")
    click.echo(f"Scenes:       {result.successful_scenes}/{result.total_scenes}")
    click.echo(f"API cost:     ${result.estimated_cost_usd:.4f}")
    click.echo(f"Report:       {result.report_path or 'not available'}")
    click.echo(f"Final video:  {result.final_video_path or 'not produced'}")
    if result.message:
        click.echo(f"Message:      {result.message}")
    if result.status == RunStatus.FAILED:
        raise click.ClickException("Conversion failed; inspect the run report.")
    if result.status == RunStatus.PAUSED_BUDGET:
        raise click.ClickException("API budget exhausted; resume with a higher --max-cost.")


def _print_evaluation_result(result) -> None:
    click.echo()
    click.echo(f"Status:      {result.status}")
    click.echo(f"Evaluation:  {result.run_directory}")
    click.echo(f"Candidates:  {result.completed_candidates}/{result.total_candidates}")
    click.echo(f"Failed:      {result.failed_candidates}")
    click.echo(f"API cost:    ${result.total_cost_usd:.4f}")
    click.echo(f"Dashboard:   {result.dashboard_path or 'not available'}")


main = pdf2manim


if __name__ == "__main__":
    pdf2manim()
