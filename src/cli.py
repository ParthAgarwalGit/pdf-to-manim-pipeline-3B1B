"""Command-line interface for the narration-directed V2 pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import click

from src.pipeline_v2 import PipelineV2, PipelineV2Config, RunStatus
from src.pipeline_v2.models import MAX_V2_SCENES
from src.pipeline_v2.prompt_text import DEFAULT_VOICE


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
@click.option(
    "--voice",
    default=DEFAULT_VOICE,
    show_default=True,
    help="Prebuilt voice name: a Gemini voice such as Sulafat, Kore or Puck for --tts-provider "
    "gemini; alloy, nova, ... for openai.",
)
@click.option(
    "--tts-provider",
    type=click.Choice(["gemini", "openai"]),
    default="gemini",
    show_default=True,
)
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
@click.option("--max-cost", type=click.FloatRange(min=0.01), default=3.0, show_default=True)
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
@click.option(
    "--script-review",
    type=click.Choice(["off", "report", "revise"]),
    default="revise",
    show_default=True,
    help=(
        "Review the planned script before TTS; revise fixes a failing script once "
        "(only the beats with flagged sentences, else the whole script)."
    ),
)
@click.option("--skip-tts", is_flag=True)
@click.option("--skip-render", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Stop after grounded lesson and narration planning.")
@click.option("--strict-keys", is_flag=True, help="Disallow deterministic fallbacks for missing provider keys.")
@click.option("--api-workers", type=click.IntRange(1, 16), default=4, show_default=True)
@click.option("--code-workers", type=click.IntRange(1, 8), default=3, show_default=True)
@click.option("--preview-workers", type=click.IntRange(1, 4), default=2, show_default=True)
@click.option("--render-workers", type=click.IntRange(1, 2), default=1, show_default=True)
@click.option("--render-sandbox", type=click.Choice(["native", "docker"]), default="native")
@click.option("--sandbox-image", default="pdf2manim:latest")
@click.option("--delivery-check/--no-delivery-check", default=True)
@click.option("--target-minutes", type=click.FloatRange(min=0.01), default=None)
@click.option("--brief", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--directives", "directives_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--plan", "plan_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--presenter",
    type=click.Choice(["off", "gemini", "d-id"]),
    default="off",
    show_default=True,
    help="On-camera presenter bridges between scenes. gemini: a fictional lecturer in the "
    "narration's voice; d-id: the professor's consented avatar.",
)
@click.option(
    "--approve-bridges", is_flag=True,
    help="Approve every proposed bridge (otherwise approve with `pdf2manim bridges`).",
)
@click.option(
    "--max-presenter-seconds", type=click.FloatRange(min=0), default=60.0, show_default=True,
    help="Cap on presenter video in one run.",
)
@click.option("--presenter-persona", default="", help="gemini: a worded description of the fictional presenter.")
@click.option(
    "--profile", "presenter_profile", type=click.Path(dir_okay=False, path_type=Path), default=None,
    help="d-id: professor profile with signed consent.",
)
@click.option("--avatar-image-url", default=None, help="d-id: HTTPS URL of the avatar image.")
def convert(
    input_pdf: str,
    output_dir: Path,
    quality: str,
    audience: str,
    voice: str,
    tts_provider: str,
    policy: str,
    strict: bool,
    repair: str,
    max_repairs_per_scene: int,
    verifier_mode: str,
    max_cost: float,
    min_scenes: Optional[int],
    max_scenes: int,
    pacing: str,
    script_review: str,
    skip_tts: bool,
    skip_render: bool,
    dry_run: bool,
    strict_keys: bool,
    api_workers: int,
    code_workers: int,
    preview_workers: int,
    render_workers: int,
    render_sandbox: str,
    sandbox_image: str,
    delivery_check: bool,
    target_minutes: Optional[float],
    brief: Optional[Path],
    directives_path: Optional[Path],
    plan_path: Optional[Path],
    presenter: str,
    approve_bridges: bool,
    max_presenter_seconds: float,
    presenter_persona: str,
    presenter_profile: Optional[Path],
    avatar_image_url: Optional[str],
) -> None:
    """Run a new V2 conversion for INPUT_PDF."""
    effective_policy = "strict" if strict else policy
    if min_scenes is not None and min_scenes > max_scenes:
        raise click.UsageError("--min-scenes cannot exceed --max-scenes")
    if brief and directives_path:
        raise click.UsageError("Use --brief or --directives, not both.")
    from src.pipeline_v2.directives import ConceptPlan, interpret_brief, load_directives

    try:
        directives = (load_directives(directives_path) if directives_path else
                      interpret_brief(brief.read_text(encoding="utf-8")) if brief else None)
        concept_plan = ConceptPlan.load(plan_path) if plan_path else None
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if directives:
        for line in directives.readback():
            click.echo(f"  Brief: {line}")
    if tts_provider == "openai" and voice == DEFAULT_VOICE:
        voice = "alloy"  # the default is a Gemini voice; use an OpenAI one
    config = PipelineV2Config(
        output_dir=output_dir,
        quality=quality,
        audience=audience,
        voice=voice,
        tts_provider=tts_provider,
        policy=effective_policy,
        repair=repair,
        max_repairs_per_scene=max_repairs_per_scene,
        verifier_mode=verifier_mode,
        max_cost_usd=max_cost,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
        pacing=pacing,
        script_review=script_review,
        enable_tts=not skip_tts,
        enable_render=not skip_render,
        dry_run=dry_run,
        strict_keys=strict_keys,
        global_api_workers=api_workers,
        code_workers=code_workers,
        preview_workers=preview_workers,
        render_workers=render_workers,
        render_sandbox=render_sandbox,
        render_sandbox_image=sandbox_image,
        enable_delivery_checks=delivery_check,
        target_minutes=target_minutes,
        directives=directives.model_dump(mode="json", by_alias=True) if directives else None,
        concept_plan=concept_plan.model_dump(mode="json") if concept_plan else None,
        presenter=presenter,
        approve_bridges=approve_bridges,
        max_presenter_seconds=max_presenter_seconds,
        presenter_persona=presenter_persona,
        presenter_profile=presenter_profile.resolve() if presenter_profile else None,
        avatar_image_url=avatar_image_url,
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
    if presenter != "off":
        approval = "all approved" if approve_bridges else "approve with `pdf2manim bridges`"
        click.echo(f"  Presenter:   {presenter}, up to {max_presenter_seconds:g}s ({approval})")
    result = PipelineV2(config).run(input_pdf)
    _print_result(result)


@pdf2manim.command("bridges")
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--approve", "approve", multiple=True, metavar="all|BRIDGE_ID",
              help="Approve a bridge (repeat as needed), or all of them.")
@click.option("--reject", "reject", multiple=True, metavar="BRIDGE_ID",
              help="Withdraw approval of a bridge (repeat as needed).")
def bridges_command(run_directory: Path, approve: tuple[str, ...], reject: tuple[str, ...]) -> None:
    """List a run's presenter bridges and approve or reject them.

    After approving, run `pdf2manim resume RUN_DIRECTORY` to render them.
    """
    from src.pipeline_v2.presenter import approve_bridges, load_bridges, save_bridges

    bridges = load_bridges(run_directory)
    if not bridges:
        raise click.ClickException(
            f"No presenter bridges in {run_directory}; convert with --presenter gemini or d-id."
        )
    known = {bridge.bridge_id for bridge in bridges}
    named = (set(approve) - {"all"}) | set(reject)
    unknown = sorted(named - known)
    if unknown:
        raise click.ClickException(
            f"Unknown bridge id(s): {', '.join(unknown)}. Available: {', '.join(sorted(known))}"
        )
    both = sorted(set(approve) & set(reject))
    if both:
        raise click.UsageError(f"Bridge(s) both approved and rejected: {', '.join(both)}")
    if approve:
        approve_bridges(run_directory, None if "all" in approve else list(approve))
    if reject:
        save_bridges(run_directory, [
            bridge.model_copy(update={"approved": False}) if bridge.bridge_id in reject else bridge
            for bridge in load_bridges(run_directory)
        ])
    click.echo(f"Presenter bridges in {run_directory.resolve()}:")
    for bridge in load_bridges(run_directory):
        place = f"before {bridge.before_scene_id}" if bridge.before_scene_id else "at the end"
        click.echo(
            f"  {bridge.bridge_id}  {bridge.kind:<10} {bridge.estimated_seconds:5.1f}s  "
            f"{'approved' if bridge.approved else 'not approved'}  ({place})"
        )
        click.echo(f"      {bridge.text}")
    if approve or reject:
        click.echo(f"Run `pdf2manim resume {run_directory}` to render the approved bridges.")


@pdf2manim.command("plan")
@click.argument("input_pdf", callback=_pdf_path)
@click.option("-o", "--out", type=click.Path(path_type=Path), default=Path("concept_plan.yaml"))
def plan_command(input_pdf: str, out: Path) -> None:
    """Create a source-grounded editable outline locally, without API calls."""
    from src.pipeline_v2.directives import ConceptPlan
    from src.pipeline_v2.planning import PlanningService, extract_source

    source = extract_source(Path(input_pdf))
    lesson = PlanningService(None)._fallback_lesson(source)
    ConceptPlan.from_lesson(lesson, source).save(out)
    click.echo(f"Concept plan: {out.resolve()}")
    click.echo("Paid calls made: 0")


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


def _pairs(values: tuple[str, ...], option: str) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        key, separator, text = value.partition("=")
        if not separator or not key.strip() or not text.strip():
            raise click.UsageError(f"{option} expects KEY=VALUE, got {value!r}")
        pairs.append((key.strip(), text.strip()))
    return pairs


@pdf2manim.command("revise")
@click.argument("run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--ops", "ops_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML or JSON file with a list of revision ops.",
)
@click.option("--edit", "edits", multiple=True, metavar="BEAT_ID=TEXT",
              help="Replace one narration beat's text.")
@click.option("--pause", "pauses", multiple=True, metavar="BEAT_ID=SECONDS",
              help="Set the pause after one beat (0-1.5 s).")
@click.option("--pronounce", "pronunciations", multiple=True, metavar="WORD=SAY_AS",
              help="Have the voice say WORD as SAY_AS.")
@click.option("--fix-visual", "visual_fixes", multiple=True, metavar="SCENE_ID=INSTRUCTION",
              help="Regenerate one scene's visuals with an instruction; keeps its audio.")
@click.option("--pacing", type=click.Choice(["standard", "relaxed"]), default=None,
              help="Re-pace every scene's narration.")
@click.option("--dry-run", is_flag=True, help="Show the changes and estimate; spend nothing.")
@click.option("--yes", is_flag=True, help="Apply without asking for confirmation.")
@click.option(
    "--max-cost", type=click.FloatRange(min=0.01), default=None,
    help="New total run budget ceiling, including money already spent.",
)
def revise_command(
    run_directory: Path,
    ops_file: Optional[Path],
    edits: tuple[str, ...],
    pauses: tuple[str, ...],
    pronunciations: tuple[str, ...],
    visual_fixes: tuple[str, ...],
    pacing: Optional[str],
    dry_run: bool,
    yes: bool,
    max_cost: Optional[float],
) -> None:
    """Edit a finished run and rebuild only the scenes the edits affect."""
    from src.pipeline_v2.revision import RevisionError, load_ops, parse_ops, plan_revision

    try:
        raw_ops: list[dict] = (
            [op.model_dump(mode="json") for op in load_ops(ops_file)] if ops_file else []
        )
        raw_ops += [{"op": "edit_beat", "beat_id": key, "text": text}
                    for key, text in _pairs(edits, "--edit")]
        for key, text in _pairs(pauses, "--pause"):
            try:
                raw_ops.append({"op": "set_pause", "beat_id": key, "pause_seconds": float(text)})
            except ValueError as exc:
                raise click.UsageError(f"--pause expects BEAT_ID=SECONDS, got {text!r}") from exc
        raw_ops += [{"op": "pronounce", "word": key, "say_as": text}
                    for key, text in _pairs(pronunciations, "--pronounce")]
        raw_ops += [{"op": "fix_visual", "scene_id": key, "instruction": text}
                    for key, text in _pairs(visual_fixes, "--fix-visual")]
        if pacing:
            raw_ops.append({"op": "set_pacing", "pacing": pacing})
        if not raw_ops:
            raise click.UsageError(
                "Nothing to revise: pass --ops, --edit, --pause, --pronounce, --fix-visual or --pacing."
            )
        ops = parse_ops(raw_ops)
        plan = plan_revision(run_directory, ops)
    except RevisionError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("pdf2manim V2 revision")
    click.echo(f"  Run:      {run_directory.resolve()}")
    click.echo(f"  Rebuild:  {', '.join(plan.affected_scene_ids) or 'nothing'}")
    click.echo(f"  Reusing:  planning and {len(plan.reused_scene_ids)} untouched scene(s)")
    for line in plan.summary_lines():
        click.echo(f"  {line}")
    if not plan.actions:
        click.echo("No scene needs rebuilding.")
        return
    if dry_run:
        click.echo("Dry run: nothing was changed or spent.")
        return
    if not yes and not click.confirm("Apply this revision?", default=False):
        click.echo("Revision cancelled; nothing was changed.")
        return
    try:
        _plan, result = PipelineV2.revise(run_directory, ops, max_cost_usd=max_cost)
    except RevisionError as exc:
        raise click.ClickException(str(exc)) from exc
    if result is not None:
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
    if result.delivery_status:
        click.echo(f"Delivery:     {result.delivery_status}")
        click.echo(f"Delivery log: {result.delivery_report_path}")
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
