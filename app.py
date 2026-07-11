"""Minimal web demo for the PDF -> Manim pipeline (Gradio).

Purpose: give the hackathon submission a live "Application URL". Designed to be
hosted on the AMD Developer Cloud instance next to the pipeline container, so
the demo itself runs on AMD hardware.

Design choices:
- Shells out to the existing CLI (`python -m src.cli convert ...`) instead of
  importing the orchestrator, so a pipeline error surfaces as a log string and
  can never take down the web server.
- Reads frontier-model API keys from the environment (keys.env / -e), exactly
  like the CLI. No keys are handled by this file.

Run locally:
    pip install -e ".[demo]"
    python app.py                      # serves on 0.0.0.0:7860

Run in the container:
    docker run --rm -p 7860:7860 --env-file keys.env --entrypoint python IMAGE app.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr

# Frontier "brain" is unchanged (gpt-5 / claude via API); this demo only drives
# the existing pipeline. See docs/amd-deployment.md for the AMD Cloud story.
INTRO = (
    "# PDF -> animated explainer\n"
    "Upload a paper (or a few pages) and get a series of short, "
    "3Blue1Brown-style animated vignettes.\n\n"
    "_Rendering and encoding run on **AMD Developer Cloud**; reasoning uses "
    "frontier models via API._"
)


def _find_videos(output_dir: Path) -> list[str]:
    return sorted(str(p) for p in output_dir.rglob("*.mp4"))


def _pick_primary(videos: list[str]) -> str:
    """Prefer the final/composed video; fall back to the first scene."""
    for v in videos:
        name = Path(v).name.lower()
        if "final" in name or "composed" in name:
            return v
    return videos[0]


def convert(pdf_file, depth, audience, quality, skip_tts):
    if not pdf_file:
        return None, "Upload a PDF to begin."

    work = Path(tempfile.mkdtemp(prefix="pdf2manim_"))
    # Normalize the upload to a .pdf path (the CLI validates the extension).
    src_pdf = work / "input.pdf"
    shutil.copy(pdf_file, src_pdf)
    out_dir = work / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable, "-m", "src.cli", "convert", str(src_pdf),
        "--output-dir", str(out_dir),
        "--depth", depth,
        "--audience", audience,
        "--quality", quality,
    ]
    if skip_tts:
        cmd.append("--skip-tts")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent),
            timeout=int(os.getenv("DEMO_TIMEOUT_SECONDS", "3600")),
        )
    except subprocess.TimeoutExpired:
        return None, "Pipeline timed out. Try fewer pages or a lower quality."

    log = ((proc.stdout or "")[-6000:] + "\n" + (proc.stderr or "")[-2000:]).strip()
    videos = _find_videos(out_dir)
    if not videos:
        return None, "No video was produced. Pipeline log:\n\n" + log

    return _pick_primary(videos), f"Produced {len(videos)} video(s).\n\n" + log


def build_demo() -> "gr.Blocks":
    with gr.Blocks(title="PDF -> 3Blue1Brown-style video") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                pdf = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                depth = gr.Dropdown(
                    ["introductory", "intermediate", "advanced"],
                    value="intermediate", label="Depth",
                )
                audience = gr.Dropdown(
                    ["high-school", "undergraduate", "graduate"],
                    value="undergraduate", label="Audience",
                )
                quality = gr.Dropdown(
                    ["low", "medium", "high"], value="medium", label="Render quality",
                )
                skip_tts = gr.Checkbox(value=False, label="Skip narration (faster)")
                btn = gr.Button("Convert", variant="primary")
            with gr.Column():
                out_video = gr.Video(label="Result")
                out_log = gr.Textbox(label="Pipeline log", lines=18)
        btn.click(convert, [pdf, depth, audience, quality, skip_tts], [out_video, out_log])
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
