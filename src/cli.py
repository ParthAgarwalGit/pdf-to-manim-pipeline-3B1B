"""
CLI entry point for pdf2manim - PDF to Manim video conversion tool.

This module provides a command-line interface for converting PDF documents to
animated mathematical explanations using Manim. The CLI supports configuration
options for verbosity, content depth, target audience, and output formats.

Usage:
    pdf2manim convert input.pdf --output-dir ./videos --depth intermediate
    pdf2manim cost input.pdf
    pdf2manim revise output/input/<run_id> [--script edited_lesson_script.json]
    pdf2manim --version
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

from src.orchestrator.pipeline import Orchestrator, PipelineConfig, PipelineAbortError
from src.utils.cost_tracker import CostTracker


def _load_env_files() -> None:
    """Load .env / keys.env from the project root into the environment.

    Called from the CLI group callback — NOT at import time. Loading at import
    time leaked real API keys into every process that merely imported this
    module (most damagingly pytest: collecting test_cli put live credentials
    in the whole suite, turning the fast mock-path lane into slow, paid,
    non-deterministic live-API calls).
    """
    if os.getenv("PYTEST_CURRENT_TEST"):
        # Under pytest, CliRunner fires this callback inside tests; loading
        # keys.env here would re-inject real credentials after the hermetic
        # conftest fixture stripped them. live_api tests provide keys
        # explicitly via their own environment.
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # python-dotenv is optional; keys can be set manually

    project_root = Path(__file__).resolve().parent.parent
    for env_name in (".env", "keys.env"):
        env_path = project_root / env_name
        if env_path.is_file():
            load_dotenv(env_path)
            break


# Setup logging for the application
def setup_logging(verbosity: str) -> logging.Logger:
    """Configure logging based on verbosity level."""
    log_level = logging.INFO
    if verbosity == "low":
        log_level = logging.WARNING
    elif verbosity == "high":
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_pdf_path(ctx, param, value: str) -> str:
    """Validate that the PDF file exists and is a valid file."""
    if not value:
        raise click.BadParameter("PDF file path is required")

    pdf_path = Path(value)
    if not pdf_path.exists():
        raise click.BadParameter(f"PDF file not found: {value}")

    if not pdf_path.is_file():
        raise click.BadParameter(f"Path is not a file: {value}")

    if pdf_path.suffix.lower() != '.pdf':
        raise click.BadParameter(f"File must be a PDF: {value}")

    return str(pdf_path.resolve())


def validate_output_dir(ctx, param, value: str) -> str:
    """Validate and create output directory if needed."""
    if not value:
        value = "./output"

    output_path = Path(value)
    output_path.mkdir(parents=True, exist_ok=True)

    return str(output_path.resolve())


@click.group()
@click.version_option(version="1.0.0", prog_name="pdf2manim")
def pdf2manim():
    """
    pdf2manim - Convert PDF documents to animated mathematical explanations.

    Transform educational PDFs into engaging Manim video animations with
    narration. Customize content depth, audience level, and output quality.
    """
    # Runs before every subcommand — the only place env files are loaded.
    _load_env_files()


@pdf2manim.command()
@click.argument('input_pdf', callback=validate_pdf_path)
@click.option(
    '-o', '--output-dir',
    callback=validate_output_dir,
    default='./output',
    help='Output directory for generated files (default: ./output)'
)
@click.option(
    '-v', '--verbosity',
    type=click.Choice(['low', 'medium', 'high'], case_sensitive=False),
    default='medium',
    help='Narration verbosity level (default: medium)'
)
@click.option(
    '-d', '--depth',
    type=click.Choice(['introductory', 'intermediate', 'advanced'], case_sensitive=False),
    default='intermediate',
    help='Content depth level (default: intermediate)'
)
@click.option(
    '-a', '--audience',
    type=click.Choice(['high-school', 'undergraduate', 'graduate'], case_sensitive=False),
    default='undergraduate',
    help='Target audience level (default: undergraduate)'
)
@click.option(
    '--granularity',
    type=click.Choice(['coarse', 'balanced', 'fine'], case_sensitive=False),
    default='balanced',
    help='Source coverage granularity (default: balanced)'
)
@click.option(
    '--target-minutes',
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help='Requested finished-video length in minutes; caps the scene budget '
         'so length is a chosen input (default: ungoverned by length)'
)
@click.option(
    '-q', '--quality',
    type=click.Choice(['low', 'medium', 'high'], case_sensitive=False),
    default='medium',
    help='Video quality (default: medium)'
)
@click.option(
    '--voice',
    type=click.Choice(['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], case_sensitive=False),
    default='alloy',
    help='TTS voice selection (default: alloy)'
)
@click.option(
    '--skip-tts',
    is_flag=True,
    default=False,
    help='Skip TTS generation and only create animation code'
)
@click.option(
    '--skip-render',
    is_flag=True,
    default=False,
    help='Skip Manim video rendering'
)
@click.option(
    '--dry-run',
    is_flag=True,
    default=False,
    help='Generate storyboard only, skip code generation and rendering'
)
@click.option(
    '--strict',
    is_flag=True,
    default=False,
    help='Fail immediately when API keys are missing instead of falling back to mocks'
)
@click.option(
    '--tts-first',
    is_flag=True,
    default=False,
    help='Generate narration before Manim code and use measured audio durations'
)
@click.option(
    '--verifier-mode',
    type=click.Choice(['off', 'shadow', 'repair'], case_sensitive=False),
    default='off',
    envvar='VERIFIER_MODE',
    help='Raw-Manim preview verifier mode'
)
@click.option(
    '--verifier-max-repairs',
    type=click.IntRange(0, 2),
    default=1,
    envvar='VERIFIER_MAX_REPAIRS',
    help='Maximum visual repair attempts per scene (default: 1)'
)
def convert(
    input_pdf: str,
    output_dir: str,
    verbosity: str,
    depth: str,
    audience: str,
    granularity: str,
    target_minutes: Optional[float],
    quality: str,
    voice: str,
    skip_tts: bool,
    skip_render: bool,
    dry_run: bool,
    strict: bool,
    tts_first: bool,
    verifier_mode: str,
    verifier_max_repairs: int,
) -> None:
    """
    Convert a PDF document to an animated Manim video.

    INPUT_PDF is the path to the PDF file to convert.

    Examples:

        pdf2manim convert document.pdf

        pdf2manim convert math.pdf --depth advanced --audience graduate

        pdf2manim convert slides.pdf -o ./videos -q high --voice nova

        pdf2manim convert notes.pdf --dry-run
    """
    try:
        # Setup logging
        logger = setup_logging(verbosity)

        # Normalize string inputs to lowercase
        verbosity = verbosity.lower()
        depth = depth.lower()
        audience = audience.lower()
        granularity = granularity.lower()
        quality = quality.lower()
        voice = voice.lower()

        # Display conversion parameters
        click.echo(click.style("\n" + "=" * 60, fg="cyan"))
        click.echo(click.style("pdf2manim - PDF to Manim Conversion", fg="cyan", bold=True))
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo()

        click.echo(click.style("Input Configuration:", bold=True))
        click.echo(f"  PDF File:     {input_pdf}")
        click.echo(f"  Output Dir:   {output_dir}")
        click.echo()

        click.echo(click.style("Content Configuration:", bold=True))
        click.echo(f"  Verbosity:    {verbosity}")
        click.echo(f"  Depth:        {depth}")
        click.echo(f"  Audience:     {audience}")
        click.echo(f"  Granularity:  {granularity}")
        click.echo(f"  Target Length:{' ' + str(target_minutes) + ' min' if target_minutes else ' auto'}")
        click.echo()

        click.echo(click.style("Rendering Configuration:", bold=True))
        click.echo(f"  Quality:      {quality}")
        click.echo(f"  Voice:        {voice}")
        click.echo(f"  TTS Enabled:  {not skip_tts}")
        click.echo(f"  Render Video: {not skip_render}")
        click.echo(f"  Dry Run:      {dry_run}")
        click.echo(f"  Strict Mode:  {strict}")
        click.echo(f"  TTS First:    {tts_first}")
        click.echo(f"  Verifier:     {verifier_mode}")
        click.echo()

        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            verbosity=verbosity,
            depth=depth,
            audience=audience,
            granularity=granularity,
            target_minutes=target_minutes,
            base_output_dir=output_dir,
            enable_ocr_debug=False,
            strict=strict,
            enable_tts=not skip_tts,
            enable_render=not skip_render,
            dry_run=dry_run,
            tts_voice=voice,
            render_quality=quality,
            tts_first=tts_first,
            verifier_mode=verifier_mode.lower(),
            verifier_max_repairs=verifier_max_repairs,
        )

        # Initialize cost tracker
        cost_tracker = CostTracker.get_instance()
        cost_tracker.reset()

        # Run the pipeline
        click.echo(click.style("Starting pipeline...", fg="yellow"))
        click.echo()

        orchestrator = Orchestrator(config=pipeline_config)

        # Get PDF filename for logging
        pdf_path = Path(input_pdf)
        event_source = 'local'  # We're using local files in CLI mode

        # Execute pipeline
        try:
            click.echo(click.style("Executing pipeline stages:", fg="yellow"))
            with click.progressbar(
                length=7,
                label=click.style("  Pipeline Progress", bold=True),
                show_pos=True
            ) as bar:
                # In a real implementation, we would track actual progress
                # For now, we'll run the pipeline and show completion
                manifest = orchestrator.run_pipeline(
                    pdf_file_path=input_pdf,
                    event_source=event_source
                )
                bar.update(7)

            click.echo()
            if manifest.status == "SUCCESS":
                click.echo(click.style("Pipeline completed successfully!", fg="green", bold=True))
            else:
                click.echo(click.style(
                    f"Pipeline completed with status {manifest.status}.",
                    fg="yellow" if manifest.status == "PARTIAL_SUCCESS" else "red",
                    bold=True,
                ))
            click.echo()

            # Display results summary
            click.echo(click.style("Results Summary:", bold=True))
            click.echo(f"  Output Directory: {manifest.output_directory}")
            click.echo(f"  Total Scenes:     {manifest.total_scenes}")
            click.echo(f"  Successful:       {manifest.successful_scenes}")
            click.echo(f"  Failed:           {manifest.total_scenes - manifest.successful_scenes}")
            delivered = getattr(manifest, "delivered_scenes", None)
            lost = getattr(manifest, "lost_scenes", None)
            if isinstance(delivered, int):
                click.echo(f"  Delivered:        {delivered} (in the final video)")
                if isinstance(lost, list) and lost:
                    click.echo(click.style(
                        f"  Lost Scenes:      {len(lost)} "
                        f"({', '.join(lost[:5])}"
                        f"{', ...' if len(lost) > 5 else ''})",
                        fg="yellow",
                    ))
            if manifest.total_scenes > 0:
                effective = (
                    delivered if isinstance(delivered, int)
                    else manifest.successful_scenes
                )
                click.echo(f"  Success Rate:     {effective / manifest.total_scenes * 100:.1f}%")
            click.echo(f"  Status:           {manifest.status}")
            if manifest.warning_count:
                click.echo(click.style(
                    f"  Warnings:         {manifest.warning_count} "
                    f"(see {Path(manifest.output_directory) / 'warnings.log'})",
                    fg="yellow",
                ))
            click.echo()

            # Display cost information
            total_cost = cost_tracker.get_total_cost()
            if total_cost > 0:
                click.echo(click.style("Cost Summary:", bold=True))
                click.echo(f"  Total Estimated Cost: ${total_cost:.4f} USD")

                breakdown = cost_tracker.get_breakdown()
                if breakdown.get('by_provider'):
                    click.echo("  Cost by Provider:")
                    for provider, cost in breakdown['by_provider'].items():
                        click.echo(f"    {provider}: ${cost:.4f}")

                if breakdown.get('by_model'):
                    click.echo("  Cost by Model:")
                    for model, cost in sorted(
                        breakdown['by_model'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]:  # Show top 3 models
                        click.echo(f"    {model}: ${cost:.4f}")
                click.echo()

            # Output generated files
            if manifest.files:
                click.echo(click.style("Generated Files:", bold=True))
                for file_ref in manifest.files[:10]:  # Show first 10 files
                    click.echo(f"  - {file_ref.path}  ({file_ref.size_bytes:,} bytes, {file_ref.type})")
                if len(manifest.files) > 10:
                    click.echo(f"  ... and {len(manifest.files) - 10} more files")
                click.echo()

            # Report media outputs from the manifest (no filesystem globbing)
            output_path = Path(manifest.output_directory)
            video_entries = [f for f in manifest.files if f.type in ("scene_video", "composed_video", "final_video")]
            audio_entries = [f for f in manifest.files if f.type == "scene_audio"]

            if video_entries:
                click.echo(click.style("Video Output:", fg="green", bold=True))
                for vf in video_entries:
                    full = output_path / vf.path
                    size_mb = vf.size_bytes / (1024 * 1024)
                    click.echo(f"  {click.style('VIDEO', fg='green')}: {full}  ({size_mb:.1f} MB)")
                click.echo()
            else:
                # Fallback: scan filesystem in case manifest was written before render
                fs_videos = sorted(output_path.rglob("*.mp4")) if output_path.exists() else []
                if fs_videos:
                    click.echo(click.style("Video Output (from filesystem):", fg="green", bold=True))
                    for vf in fs_videos:
                        size_mb = vf.stat().st_size / (1024 * 1024)
                        click.echo(f"  {click.style('VIDEO', fg='green')}: {vf}  ({size_mb:.1f} MB)")
                    click.echo()
                else:
                    click.echo(click.style("Warning: No video files were produced.", fg="yellow", bold=True))
                    click.echo("  Check: manim installed? ffmpeg available? Scene code errors?")
                    click.echo()

            if audio_entries:
                click.echo(click.style("Audio Output:", bold=True))
                for af in audio_entries:
                    full = output_path / af.path
                    size_mb = af.size_bytes / (1024 * 1024)
                    click.echo(f"  {click.style('AUDIO', fg='cyan')}: {full}  ({size_mb:.1f} MB)")
                click.echo()

            # Quality review artifacts
            review_files = {
                "storyboard_review.json": "Storyboard Review",
                "scene_code_review.json": "Scene Code Review",
                "video_review.json": "Video Review",
            }
            found_reviews = []
            for filename, label in review_files.items():
                review_path = output_path / filename
                if review_path.exists():
                    found_reviews.append((label, review_path))
            if found_reviews:
                click.echo(click.style("Quality Review Artifacts:", bold=True))
                for label, path in found_reviews:
                    click.echo(f"  {label}: {path}")
                click.echo()

            click.echo(click.style("=" * 60, fg="cyan"))
            if manifest.status in {"FAILURE", "CRITICAL_FAILURE"}:
                click.echo(click.style("Conversion failed quality gates.", fg="red", bold=True))
                if manifest.final_video_path:
                    click.echo(f"  Diagnostic video: {output_path / manifest.final_video_path}")
            elif manifest.final_video_path:
                click.echo(click.style("Conversion completed successfully!", fg="green", bold=True))
                click.echo(f"  Final video: {output_path / manifest.final_video_path}")
            elif video_entries:
                click.echo(click.style("Conversion completed successfully!", fg="green", bold=True))
            else:
                click.echo(click.style("Pipeline finished but no video was produced.", fg="yellow", bold=True))
            click.echo(click.style("=" * 60, fg="cyan"))
            if manifest.status in {"FAILURE", "CRITICAL_FAILURE"}:
                raise click.ClickException(
                    f"Pipeline finished with status {manifest.status}; inspect the saved reports."
                )

        except PipelineAbortError as e:
            click.echo()
            click.echo(click.style(
                f"Pipeline aborted at stage '{e.stage}'",
                fg="red",
                bold=True
            ))
            click.echo(f"Error Code: {e.error_code}")
            click.echo(f"Message: {e.message}")
            if e.context:
                click.echo("Context:")
                for key, value in e.context.items():
                    click.echo(f"  {key}: {value}")
            sys.exit(1)

    except click.ClickException:
        # Re-raise Click exceptions (validation errors, etc.)
        raise
    except Exception as e:
        click.echo()
        click.echo(click.style(
            f"Error: {str(e)}",
            fg="red",
            bold=True
        ))
        logger = logging.getLogger(__name__)
        logger.exception("Unexpected error during conversion")
        sys.exit(1)


@pdf2manim.command()
@click.argument('input_pdf', callback=validate_pdf_path)
def cost(input_pdf: str) -> None:
    """
    Estimate the cost for converting a PDF to video.

    Analyzes the input PDF and provides a cost estimate for running the
    complete conversion pipeline, including API calls and processing.

    INPUT_PDF is the path to the PDF file to analyze.

    Examples:

        pdf2manim cost document.pdf
    """
    try:
        click.echo(click.style("\n" + "=" * 60, fg="cyan"))
        click.echo(click.style("pdf2manim - Cost Estimation", fg="cyan", bold=True))
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo()

        click.echo(f"Analyzing: {input_pdf}")
        click.echo()

        # Get file information
        pdf_path = Path(input_pdf)
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

        click.echo(click.style("File Information:", bold=True))
        click.echo(f"  Filename:   {pdf_path.name}")
        click.echo(f"  Size:       {file_size_mb:.2f} MB")
        click.echo()

        # Estimate costs based on file size and typical processing
        # These are rough estimates
        estimated_input_tokens = max(5000, int(file_size_mb * 50000))  # ~50k tokens per MB
        estimated_output_tokens = int(estimated_input_tokens * 0.5)

        # Estimate TTS characters (rough: ~1 char per 0.005 input tokens)
        estimated_tts_chars = int(estimated_input_tokens * 0.005)

        click.echo(click.style("Estimated Token Usage:", bold=True))
        click.echo(f"  Input Tokens:  {estimated_input_tokens:,}")
        click.echo(f"  Output Tokens: {estimated_output_tokens:,}")
        click.echo(f"  TTS Characters: {estimated_tts_chars:,}")
        click.echo()

        # Cost estimation for different configurations
        # Using typical pricing from cost_tracker
        gpt_4o_input_price = 2.50 / 1_000_000
        gpt_4o_output_price = 10.00 / 1_000_000
        tts_price = 15.00 / 1_000_000  # per character

        llm_cost = (
            (estimated_input_tokens * gpt_4o_input_price) +
            (estimated_output_tokens * gpt_4o_output_price)
        )

        tts_cost = estimated_tts_chars * tts_price

        total_cost = llm_cost + tts_cost

        click.echo(click.style("Estimated Costs:", bold=True))
        click.echo(f"  LLM Processing:  ${llm_cost:.4f} USD")
        click.echo(f"  TTS Generation:  ${tts_cost:.4f} USD")
        click.echo(f"  Total Estimate:  ${total_cost:.4f} USD")
        click.echo()

        click.echo(click.style("Notes:", bold=True))
        click.echo("  - Costs are estimates based on file size and typical processing")
        click.echo("  - Actual costs depend on content complexity and API usage")
        click.echo("  - Run 'convert' command for precise cost tracking during execution")
        click.echo()

        click.echo(click.style("=" * 60, fg="cyan"))

    except Exception as e:
        click.echo()
        click.echo(click.style(
            f"Error: {str(e)}",
            fg="red",
            bold=True
        ))
        sys.exit(1)


@pdf2manim.command()
@click.argument('run_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    '--script', 'script_path',
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help='Edited lesson script (default: RUN_DIR/lesson_script.json)'
)
@click.option(
    '-q', '--quality',
    type=click.Choice(['low', 'medium', 'high'], case_sensitive=False),
    default=None,
    help="Video quality (default: the run's own)"
)
@click.option(
    '--voice',
    type=click.Choice(['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], case_sensitive=False),
    default=None,
    help="TTS voice (default: the run's own)"
)
@click.option('--skip-tts', is_flag=True, default=False, help='Skip TTS generation')
@click.option('--skip-render', is_flag=True, default=False, help='Skip Manim video rendering')
@click.option(
    '--dry-run',
    is_flag=True,
    default=False,
    help='Plan only: write revision.json showing which scenes would be rebuilt'
)
@click.option(
    '--strict',
    is_flag=True,
    default=False,
    help='Fail immediately when API keys are missing instead of falling back to mocks'
)
@click.option(
    '--verifier-mode',
    type=click.Choice(['off', 'shadow', 'repair'], case_sensitive=False),
    default='off',
    envvar='VERIFIER_MODE',
    help='Raw-Manim preview verifier mode'
)
def revise(
    run_dir: str,
    script_path: Optional[str],
    quality: Optional[str],
    voice: Optional[str],
    skip_tts: bool,
    skip_render: bool,
    dry_run: bool,
    strict: bool,
    verifier_mode: str,
) -> None:
    """
    Rebuild a finished run from its edited lesson script.

    Edit RUN_DIR/lesson_script.json (or a copy passed with --script) and run
    this. Only scenes whose beats changed are re-planned and regenerated; the
    others reuse the run's code, renders and narration. The result is a new
    run in RUN_DIR/rev_NNN with a revision.json; RUN_DIR itself is untouched.

    Examples:

        pdf2manim revise output/notes/2026-11-07T10-00-00

        pdf2manim revise output/notes/2026-11-07T10-00-00 --script edits.json --dry-run
    """
    from src.orchestrator.revision import LESSON_SCRIPT_FILENAME, load_base_run, load_lesson_script

    setup_logging("medium")
    try:
        base = load_base_run(Path(run_dir))
        script_file = Path(script_path) if script_path else Path(run_dir) / LESSON_SCRIPT_FILENAME
        if not script_file.is_file():
            raise click.ClickException(f"No lesson script at {script_file}; pass --script.")
        script = load_lesson_script(script_file)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Cannot revise {run_dir}: {e}")

    # Keep the parent run's voice, quality and timing mode unless overridden:
    # changing any of them invalidates every scene's reusable artifacts.
    run_config = base.run_config
    pipeline_config = PipelineConfig(
        base_output_dir=str(Path(run_dir).parent),
        strict=strict,
        enable_tts=not skip_tts,
        enable_render=not skip_render,
        dry_run=dry_run,
        tts_voice=(voice or getattr(run_config, "tts_voice", None) or "alloy").lower(),
        render_quality=(quality or getattr(run_config, "render_quality", None) or "medium").lower(),
        tts_first=bool(getattr(run_config, "tts_first", False)),
        verifier_mode=verifier_mode.lower(),
    )

    cost_tracker = CostTracker.get_instance()
    cost_tracker.reset()
    click.echo(click.style(f"Revising {run_dir} from {script_file}", fg="cyan", bold=True))
    try:
        manifest = Orchestrator(config=pipeline_config).run_revision(run_dir, script)
    except PipelineAbortError as e:
        click.echo(click.style(f"Revision aborted at stage '{e.stage}'", fg="red", bold=True))
        click.echo(f"Error Code: {e.error_code}")
        click.echo(f"Message: {e.message}")
        sys.exit(1)

    output_path = Path(manifest.output_directory)
    report = json.loads((output_path / "revision.json").read_text(encoding="utf-8"))
    scenes = report["scenes"]
    beats = report["beats"]
    click.echo()
    click.echo(click.style("Revision Summary:", bold=True))
    click.echo(f"  Output Directory: {output_path}")
    click.echo(
        f"  Beats:            {len(beats['changed'])} changed, "
        f"{len(beats['added'])} added, {len(beats['removed'])} removed"
    )
    click.echo(
        f"  Scenes:           {scenes['total']} total, "
        f"{len(scenes['regenerated'])} regenerated, "
        f"{len(scenes['reused_video'])} renders reused, "
        f"{len(scenes['reused_audio'])} narrations reused"
    )
    click.echo(f"  Status:           {manifest.status}")
    click.echo(f"  Cost:             ${report['cost_usd']:.4f} USD")
    if manifest.final_video_path:
        click.echo(f"  Final video:      {output_path / manifest.final_video_path}")
    if manifest.status in {"FAILURE", "CRITICAL_FAILURE"}:
        raise click.ClickException(
            f"Revision finished with status {manifest.status}; inspect the saved reports."
        )


def main() -> None:
    """Main entry point for the CLI."""
    try:
        pdf2manim()
    except KeyboardInterrupt:
        click.echo()
        click.echo(click.style("Conversion cancelled by user", fg="yellow"))
        sys.exit(130)
    except Exception as e:
        click.echo()
        click.echo(click.style(
            f"Fatal error: {str(e)}",
            fg="red",
            bold=True
        ))
        sys.exit(1)


if __name__ == '__main__':
    main()
