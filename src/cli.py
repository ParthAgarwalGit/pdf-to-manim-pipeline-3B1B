"""Command-line interface for the narration-directed V2 pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import click

from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus


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
    "--verifier-mode",
    type=click.Choice(["enforce", "audit"]),
    default="enforce",
    show_default=True,
    help="Audit logs visual findings without rejecting renderable scenes.",
)
@click.option("--max-cost", type=click.FloatRange(min=0.01), default=1.0, show_default=True)
@click.option("--max-scenes", type=click.IntRange(1, 12), default=12, show_default=True)
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
    verifier_mode: str,
    max_cost: float,
    max_scenes: int,
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
    config = PipelineV2Config(
        output_dir=output_dir,
        quality=quality,
        audience=audience,
        voice=voice,
        policy=effective_policy,
        repair=repair,
        verifier_mode=verifier_mode,
        max_cost_usd=max_cost,
        max_scenes=max_scenes,
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
    click.echo(f"  Verifier:    {verifier_mode}")
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


@pdf2manim.command()
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report(run_directory: Path) -> None:
    """Print the path to a run's local HTML report."""
    path = (run_directory / "report.html").resolve()
    if not path.exists():
        raise click.ClickException(f"Report has not been generated: {path}")
    click.echo(str(path))


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


main = pdf2manim


if __name__ == "__main__":
    pdf2manim()
