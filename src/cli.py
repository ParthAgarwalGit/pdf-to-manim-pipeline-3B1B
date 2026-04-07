"""
CLI entry point for pdf2manim - PDF to Manim video conversion tool.

This module provides a command-line interface for converting PDF documents to
animated mathematical explanations using Manim. The CLI supports configuration
options for verbosity, content depth, target audience, and output formats.

Usage:
    pdf2manim convert input.pdf --output-dir ./videos --depth intermediate
    pdf2manim cost input.pdf
    pdf2manim --version
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

# Load .env file from project root (if it exists) before any agent
# tries to read API keys from the environment.
try:
    from dotenv import load_dotenv
    # Walk up from this file to find the project root .env or keys.env
    _project_root = Path(__file__).resolve().parent.parent
    for _env_name in (".env", "keys.env"):
        _env_path = _project_root / _env_name
        if _env_path.is_file():
            load_dotenv(_env_path)
            break
except ImportError:
    pass  # python-dotenv is optional; keys can be set manually

from src.orchestrator.pipeline import Orchestrator, PipelineConfig, PipelineAbortError
from src.utils.cost_tracker import CostTracker


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
    pass


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
def convert(
    input_pdf: str,
    output_dir: str,
    verbosity: str,
    depth: str,
    audience: str,
    quality: str,
    voice: str,
    skip_tts: bool,
    skip_render: bool,
    dry_run: bool,
    strict: bool
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
        click.echo()

        click.echo(click.style("Rendering Configuration:", bold=True))
        click.echo(f"  Quality:      {quality}")
        click.echo(f"  Voice:        {voice}")
        click.echo(f"  TTS Enabled:  {not skip_tts}")
        click.echo(f"  Render Video: {not skip_render}")
        click.echo(f"  Dry Run:      {dry_run}")
        click.echo(f"  Strict Mode:  {strict}")
        click.echo()

        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            verbosity=verbosity,
            depth=depth,
            audience=audience,
            base_output_dir=output_dir,
            enable_ocr_debug=False,
            strict=strict,
            enable_tts=not skip_tts,
            enable_render=not skip_render,
            dry_run=dry_run,
            tts_voice=voice,
            render_quality=quality,
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
            click.echo(click.style("Pipeline completed successfully!", fg="green", bold=True))
            click.echo()

            # Display results summary
            click.echo(click.style("Results Summary:", bold=True))
            click.echo(f"  Output Directory: {manifest.output_directory}")
            click.echo(f"  Total Scenes:     {manifest.total_scenes}")
            click.echo(f"  Successful:       {manifest.successful_scenes}")
            click.echo(f"  Failed:           {manifest.total_scenes - manifest.successful_scenes}")
            if manifest.total_scenes > 0:
                success_rate = (manifest.successful_scenes / manifest.total_scenes) * 100
                click.echo(f"  Success Rate:     {success_rate:.1f}%")
            click.echo(f"  Status:           {manifest.status}")
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

            click.echo(click.style("=" * 60, fg="cyan"))
            if manifest.final_video_path:
                click.echo(click.style("Conversion completed successfully!", fg="green", bold=True))
                click.echo(f"  Final video: {output_path / manifest.final_video_path}")
            elif video_entries:
                click.echo(click.style("Conversion completed successfully!", fg="green", bold=True))
            else:
                click.echo(click.style("Pipeline finished but no video was produced.", fg="yellow", bold=True))
            click.echo(click.style("=" * 60, fg="cyan"))

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
