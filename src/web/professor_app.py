"""Professor studio: PDF + brief -> read-back -> video + delivery report ->
feedback -> reviewed revision (Part F/G/H flow in one page).

Every step that renders shells out to the CLI, as the original demo does,
so a pipeline error comes back as a message and never takes down the web
server. What the professor approved is passed on exactly: the confirmed
brief interpretation (``convert --directives``) and the reviewed edit
bundle (``revise --edits``), never re-interpreted.

The handlers are plain functions of their inputs (plus an injectable
command runner), so they are tested without a browser.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
Runner = Callable[[List[str]], subprocess.CompletedProcess]


def run_cli(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        timeout=int(os.getenv("DEMO_TIMEOUT_SECONDS", "3600")),
    )


def _tail(proc: subprocess.CompletedProcess, n: int = 4000) -> str:
    return ((proc.stdout or "")[-n:] + "\n" + (proc.stderr or "")[-1500:]).strip()


def _output_dir(proc: subprocess.CompletedProcess, search_root: Optional[Path] = None) -> Optional[Path]:
    """The run directory the CLI reported (or the newest manifest under root)."""
    matches = re.findall(r"Output Directory:\s+(.+)", proc.stdout or "")
    if matches:
        path = Path(matches[-1].strip())
        if path.is_dir():
            return path
    if search_root and search_root.is_dir():
        manifests = sorted(search_root.rglob("manifest.json"), key=lambda p: p.stat().st_mtime)
        if manifests:
            return manifests[-1].parent
    return None


def _final_video(run_dir: Path) -> Optional[str]:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rel = manifest.get("final_video_path")
    if rel and (run_dir / rel).is_file():
        return str(run_dir / rel)
    videos = sorted(run_dir.glob("final*.mp4"))
    return str(videos[-1]) if videos else None


def delivery_markdown(run_dir: Optional[Path]) -> str:
    if run_dir is None:
        return ""
    path = run_dir / "delivery_report.json"
    if not path.is_file():
        return "_No delivery report (rendering or narration was skipped)._"
    report = json.loads(path.read_text(encoding="utf-8"))
    icon = "✅" if report.get("status") == "ready" else "⚠️"
    lines = (report.get("summary") or "").splitlines()
    out = [f"### {icon} {lines[0] if lines else report.get('status')}"]
    out += [f"- {line}" for line in lines[1:]]
    if report.get("change_log"):
        out.append(f"\n**Change log:** {report['change_log']}")
    return "\n".join(out)


def bridges_markdown(run_dir: Optional[Path]) -> str:
    if run_dir is None or not (run_dir / "lesson_script.json").is_file():
        return ""
    script = json.loads((run_dir / "lesson_script.json").read_text(encoding="utf-8"))
    pending = [b for b in script.get("beats", []) if b.get("role") == "bridge" and not b.get("locked")]
    if not pending:
        return ""
    lines = ["**Presenter bridges awaiting your approval** (not in the video yet):"]
    lines += [f"- `{b['beat_id']}`: {b['narration']}" for b in pending]
    return "\n".join(lines)


# ── step 1: brief read-back ─────────────────────────────────────────────

def read_back_brief(brief: str, interpreter: Any = None) -> Tuple[str, str]:
    """(markdown read-back, confirmed-directives JSON state)."""
    if not (brief or "").strip():
        return "_No brief: the video follows the PDF and the options above._", ""
    if interpreter is None:
        from src.agents.professor_input_interpreter import ProfessorInputInterpreter
        interpreter = ProfessorInputInterpreter()
    directives = interpreter.interpret_brief(brief)
    lines = directives.readback() or ["(nothing specific; your brief is passed on as notes)"]
    markdown = "**Here's how I understood your brief:**\n" + "\n".join(f"- {line}" for line in lines)
    markdown += "\n\nIf that's right, press **Generate video**. Otherwise edit the brief and read it back again."
    return markdown, directives.model_dump_json(by_alias=True)


# ── step 2: generate ────────────────────────────────────────────────────

def generate(
    pdf_file: Optional[str],
    directives_json: str,
    quality: str = "medium",
    voice: str = "alloy",
    bridges: bool = False,
    layout: str = "full",
    runner: Runner = run_cli,
    work_root: Optional[Path] = None,
) -> Tuple[Optional[str], str, str, str, str]:
    """(video, delivery report, bridge proposals, log, run-dir state)."""
    if not pdf_file:
        return None, "Upload a PDF to begin.", "", "", ""
    work = Path(work_root or tempfile.mkdtemp(prefix="pdf2manim_studio_"))
    work.mkdir(parents=True, exist_ok=True)
    src_pdf = work / "input.pdf"
    shutil.copy(pdf_file, src_pdf)
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)
    args = [
        "convert", str(src_pdf), "--output-dir", str(out_dir),
        "--quality", quality, "--voice", voice, "--tts-first", "--yes",
        "--presenter-layout", layout,
    ]
    if bridges:
        args += ["--bridges", "propose"]
    if directives_json:
        directives_file = work / "directives.json"
        directives_file.write_text(directives_json, encoding="utf-8")
        args += ["--directives", str(directives_file)]
    try:
        proc = runner(args)
    except subprocess.TimeoutExpired:
        return None, "The pipeline timed out. Try fewer pages or a lower quality.", "", "", ""
    run_dir = _output_dir(proc, out_dir)
    if run_dir is None:
        return None, "No run was produced. See the log.", "", _tail(proc), ""
    return _final_video(run_dir), delivery_markdown(run_dir), bridges_markdown(run_dir), _tail(proc), str(run_dir)


# ── step 3: feedback -> reviewed revision ───────────────────────────────

def review_feedback(run_dir: str, feedback: str, interpreter: Any = None, editor: Any = None) -> Tuple[str, str]:
    """(markdown plan + diff + estimate, edit-bundle JSON state)."""
    if not run_dir:
        return "Generate a video first.", ""
    if not (feedback or "").strip():
        return "Write what you'd like to change.", ""
    from src.agents.professor_input_interpreter import ProfessorInputInterpreter, build_timeline
    from src.agents.script_editor import ScriptEditorAgent
    from src.orchestrator.revision import estimate_revision, load_base_run

    base = load_base_run(Path(run_dir))
    interpreter = interpreter or ProfessorInputInterpreter()
    plan = interpreter.interpret_feedback(feedback, base.script, build_timeline(base.storyboard, base.delivery_report))
    result = (editor or ScriptEditorAgent()).apply(base.script, plan.ops)
    estimate = estimate_revision(base, result.script, visual_fixes=result.visual_fixes)
    out = ["**Here's how I understood your feedback:**"]
    for item, op in zip(plan.items, plan.ops):
        out.append(f"- *{item['class']}*: `{op.op}` {json.dumps(op.model_dump(exclude={'op'}))[:140]}")
    out += [f"- ❓ not understood, please rephrase: {note}" for note in plan.notes]
    out.append("\n**Changes:**")
    out += [f"    {line}" for line in result.diff] or ["    (none)"]
    out += [f"- ⚠️ {w}" for w in result.warnings]
    out += [f"- ❌ rejected: {e}" for e in result.errors]
    out.append(
        f"\n**Estimate:** {estimate['scenes_rebuilt']} scene(s) rebuilt, "
        f"{estimate['scenes_renarrated']} re-narrated, {estimate['scenes_reused']} reused; "
        f"length {estimate['length_delta_seconds']:+.0f}s; about ${estimate['estimated_cost_usd']:.2f}."
    )
    if not result.applied:
        return "\n".join(out + ["\nNothing can be applied yet."]), ""
    out.append("\nPress **Apply changes** to render this revision.")
    bundle = {
        "script": result.script.model_dump(mode="json"),
        "visual_fixes": result.visual_fixes,
        "ops": result.applied,
    }
    return "\n".join(out), json.dumps(bundle)


def apply_revision(run_dir: str, bundle_json: str, runner: Runner = run_cli) -> Tuple[Optional[str], str, str, str, str]:
    """(video, delivery report, bridge proposals, log, new run-dir state)."""
    if not run_dir or not bundle_json:
        return None, "Review feedback first.", "", "", run_dir
    bundle_file = Path(run_dir) / f"edits_{os.getpid()}_{len(bundle_json)}.json"
    bundle_file.write_text(bundle_json, encoding="utf-8")
    try:
        proc = runner(["revise", run_dir, "--edits", str(bundle_file), "--yes"])
    finally:
        bundle_file.unlink(missing_ok=True)
    new_dir = _output_dir(proc)
    if new_dir is None:
        return None, "The revision did not finish. See the log.", "", _tail(proc), run_dir
    return _final_video(new_dir), delivery_markdown(new_dir), bridges_markdown(new_dir), _tail(proc), str(new_dir)


def approve_bridges(run_dir: str, runner: Runner = run_cli) -> Tuple[Optional[str], str, str, str, str]:
    if not run_dir:
        return None, "Generate a video first.", "", "", run_dir
    proc = runner(["revise", run_dir, "--approve-bridges", "--yes"])
    new_dir = _output_dir(proc)
    if new_dir is None:
        return None, "Approving bridges did not finish. See the log.", "", _tail(proc), run_dir
    return _final_video(new_dir), delivery_markdown(new_dir), bridges_markdown(new_dir), _tail(proc), str(new_dir)


# ── UI ──────────────────────────────────────────────────────────────────

def build_studio(gr) -> None:
    """Add the professor studio to the current Gradio Blocks context."""
    run_state = gr.State("")
    directives_state = gr.State("")
    edits_state = gr.State("")
    gr.Markdown(
        "### 1 · Your notes and your brief\n"
        "Upload the PDF and, optionally, tell me what you want: what to skip, what to go "
        "deeper on, analogies to use or avoid, notation, length."
    )
    with gr.Row():
        with gr.Column():
            pdf = gr.File(label="Lecture notes (PDF)", file_types=[".pdf"], type="filepath")
            brief = gr.Textbox(
                label="Brief (optional)", lines=5,
                placeholder="My students are 2nd-year EE. Use the rotating-arm picture, not the clock "
                            "analogy. Skip 4.3. Under 10 minutes. I write j, not i.",
            )
            with gr.Row():
                quality = gr.Dropdown(["low", "medium", "high"], value="medium", label="Quality")
                voice = gr.Dropdown(["alloy", "echo", "fable", "onyx", "nova", "shimmer"], value="alloy", label="Voice")
            with gr.Row():
                bridges = gr.Checkbox(value=False, label="Propose presenter bridges")
                layout = gr.Dropdown(["full", "pip"], value="full", label="Presenter layout")
            readback_btn = gr.Button("Read back my brief")
            readback = gr.Markdown()
            generate_btn = gr.Button("Generate video", variant="primary")
        with gr.Column():
            gr.Markdown("### 2 · Your video")
            video = gr.Video(label="Video")
            report = gr.Markdown()
            proposals = gr.Markdown()
            approve_btn = gr.Button("Approve presenter bridges")
    gr.Markdown("### 3 · Feedback\nSay what to change, with timestamps if you like (\"at 2:13 the label overlaps\").")
    with gr.Row():
        with gr.Column():
            feedback = gr.Textbox(label="Feedback", lines=4)
            review_btn = gr.Button("Review my feedback")
            apply_btn = gr.Button("Apply changes", variant="primary")
        with gr.Column():
            plan = gr.Markdown()
    log = gr.Textbox(label="Pipeline log", lines=10)

    readback_btn.click(read_back_brief, [brief], [readback, directives_state])
    generate_btn.click(
        lambda f, d, q, v, b, lay: generate(f, d, q, v, b, lay),
        [pdf, directives_state, quality, voice, bridges, layout],
        [video, report, proposals, log, run_state],
    )
    approve_btn.click(approve_bridges, [run_state], [video, report, proposals, log, run_state])
    review_btn.click(review_feedback, [run_state, feedback], [plan, edits_state])
    apply_btn.click(apply_revision, [run_state, edits_state], [video, report, proposals, log, run_state])
