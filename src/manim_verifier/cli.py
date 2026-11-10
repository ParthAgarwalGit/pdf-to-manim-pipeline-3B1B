"""Standalone CLI for the raw-Manim verifier."""

from __future__ import annotations

import json
from pathlib import Path

import click

from src.manim_verifier.pipeline import RawManimVerifierPipeline, VerifierConfig
from src.manim_verifier.vision_reviewer import GeminiVisionReviewer


@click.command(name="manim-verifier")
@click.argument("script", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--brief", default="", help="Short description of the intended animation.")
@click.option("--storyboard", default="", help="Storyboard text or JSON.")
@click.option("--mode", type=click.Choice(["shadow", "repair"]), default="shadow")
@click.option("--run-root", type=click.Path(path_type=Path), default=Path("verifier_runs"))
@click.option("--max-repairs", type=click.IntRange(0, 2), default=1)
@click.option("--env-file", type=click.Path(path_type=Path), default=Path(".env"))
def main(
    script: Path,
    brief: str,
    storyboard: str,
    mode: str,
    run_root: Path,
    max_repairs: int,
    env_file: Path,
) -> None:
    """Preview and optionally repair a complete raw Manim SCRIPT."""
    reviewer = GeminiVisionReviewer.from_env(env_file) if mode == "repair" else None
    pipeline = RawManimVerifierPipeline(
        reviewer=reviewer,
        config=VerifierConfig(
            mode=mode,
            run_root=run_root,
            max_repair_attempts=max_repairs,
        ),
    )
    result = pipeline.run(
        brief=brief,
        storyboard=storyboard,
        initial_code=script.read_text(encoding="utf-8"),
    )
    click.echo(json.dumps(result.to_dict(), indent=2))
    if not result.success:
        raise click.ClickException("No safe, rendered candidate was produced.")


if __name__ == "__main__":
    main()
