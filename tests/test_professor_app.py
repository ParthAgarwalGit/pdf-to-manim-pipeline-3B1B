"""P6: the professor studio flow (handlers, CLI hand-offs, UI wiring)."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from src.agents.professor_input_interpreter import ProfessorInputInterpreter
from src.agents.script_editor import ScriptEditorAgent
from src.orchestrator.pipeline import Orchestrator
from src.schemas.directives import ProfessorDirectives
from src.web import professor_app as studio
from tests.test_revision import _base_run, _script


def _fake_run_dir(root: Path, status="ready", bridges=False) -> Path:
    run = root / "notes" / "run1"
    run.mkdir(parents=True)
    (run / "final_with_narration.mp4").write_bytes(b"mp4")
    (run / "manifest.json").write_text(json.dumps({"final_video_path": "final_with_narration.mp4"}), encoding="utf-8")
    summary = "Ready: 3/3 scenes passed sync and quality checks." if status == "ready" else (
        "Needs review: 2/3 scenes passed sync and quality checks.\nScene 2 (0:10-0:20): frozen frame."
    )
    (run / "delivery_report.json").write_text(json.dumps({"status": status, "summary": summary}), encoding="utf-8")
    beats = [{"beat_id": "beat_001", "narration": "Hi.", "role": "core", "locked": False}]
    if bridges:
        beats.append({"beat_id": "beat_009", "narration": "Here's the real question.", "role": "bridge", "locked": False})
    (run / "lesson_script.json").write_text(json.dumps({"beats": beats}), encoding="utf-8")
    return run


def _runner_reporting(run_dir: Path, seen: list):
    def runner(args):
        seen.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=f"  Output Directory: {run_dir}\nDone.", stderr="")
    return runner


class TestReadBack:
    def test_empty_brief(self):
        markdown, state = studio.read_back_brief("   ")
        assert state == "" and "No brief" in markdown

    def test_brief_is_read_back_and_kept_as_confirmed_state(self):
        markdown, state = studio.read_back_brief(
            "Skip 4.3. Under 10 minutes. I write j, not i.",
            interpreter=ProfessorInputInterpreter(use_default_llm=False),
        )
        assert "Leave out section(s): 4.3" in markdown and "Notation: write j, not i." in markdown
        directives = ProfessorDirectives.model_validate_json(state)
        assert directives.global_.target_minutes == 10


class TestGenerate:
    def test_passes_the_confirmed_directives_and_shows_the_report(self, tmp_path):
        run_dir = _fake_run_dir(tmp_path / "result", bridges=True)
        pdf = tmp_path / "notes.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        _, state = studio.read_back_brief("Skip 4.3.", interpreter=ProfessorInputInterpreter(use_default_llm=False))
        seen = []
        video, report, proposals, log, run_state = studio.generate(
            str(pdf), state, quality="low", bridges=True, layout="pip",
            runner=_runner_reporting(run_dir, seen), work_root=tmp_path / "work",
        )
        args = seen[0]
        assert args[:2] == ["convert", str(tmp_path / "work" / "input.pdf")]
        assert "--tts-first" in args and "--yes" in args and args[args.index("--bridges") + 1] == "propose"
        directives_file = Path(args[args.index("--directives") + 1])
        assert ProfessorDirectives.model_validate_json(directives_file.read_text()).scope.skip_sections == ["4.3"]
        assert video.endswith("final_with_narration.mp4") and run_state == str(run_dir)
        assert report.startswith("### ✅ Ready: 3/3")
        assert "beat_009" in proposals and "awaiting your approval" in proposals

    def test_missing_pdf_and_failed_run(self, tmp_path):
        assert studio.generate(None, "")[1] == "Upload a PDF to begin."
        pdf = tmp_path / "notes.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        def failing(args):
            return subprocess.CompletedProcess(args, 1, stdout="boom", stderr="trace")

        video, report, _, log, state = studio.generate(str(pdf), "", runner=failing, work_root=tmp_path / "w")
        assert video is None and "No run was produced" in report and "boom" in log and state == ""

    def test_needs_review_is_listed(self, tmp_path):
        run_dir = _fake_run_dir(tmp_path, status="needs_review")
        markdown = studio.delivery_markdown(run_dir)
        assert markdown.startswith("### ⚠️ Needs review") and "- Scene 2 (0:10-0:20): frozen frame." in markdown


class TestFeedbackLoop:
    def test_review_then_apply_hands_over_exactly_what_was_reviewed(self, tmp_path):
        run_dir, _ = _base_run(tmp_path, _script())
        plan, bundle_json = studio.review_feedback(
            str(run_dir), "At 0:05 the label overlaps the square.",
            interpreter=ProfessorInputInterpreter(use_default_llm=False),
            editor=ScriptEditorAgent(use_default_llm=False),
        )
        assert "*visual*: `fix_visual`" in plan and "Estimate:" in plan
        bundle = json.loads(bundle_json)
        assert list(bundle["visual_fixes"].values()) == ["At 0:05 the label overlaps the square."]

        new_dir = _fake_run_dir(tmp_path / "rev")
        seen, captured = [], {}

        def runner(args):
            seen.append(args)
            captured["bundle"] = json.loads(Path(args[args.index("--edits") + 1]).read_text())
            return subprocess.CompletedProcess(args, 0, stdout=f"Output Directory: {new_dir}", stderr="")

        video, report, _, _, state = studio.apply_revision(str(run_dir), bundle_json, runner=runner)
        assert seen[0][:2] == ["revise", str(run_dir)] and "--yes" in seen[0]
        assert captured["bundle"] == bundle
        assert not list(run_dir.glob("edits_*.json"))  # temp bundle cleaned up
        assert state == str(new_dir) and video.endswith(".mp4")

    def test_unintelligible_feedback_is_not_applied(self, tmp_path):
        run_dir, _ = _base_run(tmp_path, _script())
        plan, bundle = studio.review_feedback(
            str(run_dir), "Make it nicer.",
            interpreter=ProfessorInputInterpreter(use_default_llm=False),
            editor=ScriptEditorAgent(use_default_llm=False),
        )
        assert bundle == "" and "not understood" in plan

    def test_approve_bridges_hand_off(self, tmp_path):
        new_dir = _fake_run_dir(tmp_path / "rev")
        seen = []
        studio.approve_bridges("RUN", runner=_runner_reporting(new_dir, seen))
        assert seen == [["revise", "RUN", "--approve-bridges", "--yes"]]


class TestCliHandOffs:
    def test_revise_edits_bundle_is_applied_verbatim(self, tmp_path, monkeypatch):
        import src.cli as cli

        run_dir, _ = _base_run(tmp_path, _script())
        base_script = json.loads((run_dir / "lesson_script.json").read_text("utf-8"))
        bundle = tmp_path / "edits.json"
        bundle.write_text(json.dumps({
            "script": base_script, "visual_fixes": {"beat_003": "Bigger labels."},
            "ops": [{"op": "fix_visual", "beat_id": "beat_003", "instruction": "Bigger labels."}],
        }), encoding="utf-8")
        captured = {}

        def fake_run_revision(self, base_run_dir, script, visual_fixes=None, ops_applied=None):
            captured.update(visual_fixes=visual_fixes, ops=ops_applied, beats=len(script.beats))
            rev = Path(base_run_dir) / "rev_002"
            rev.mkdir()
            (rev / "revision.json").write_text(json.dumps({
                "beats": {"added": [], "removed": [], "changed": []},
                "scenes": {"total": 1, "regenerated": [], "reused_video": {}, "reused_audio": {}},
                "cost_usd": 0.0,
            }), encoding="utf-8")
            return SimpleNamespace(output_directory=str(rev), status="SUCCESS", final_video_path=None)

        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        monkeypatch.setattr(Orchestrator, "run_revision", fake_run_revision)
        result = CliRunner().invoke(cli.pdf2manim, ["revise", str(run_dir), "--edits", str(bundle), "--yes"])
        assert result.exit_code == 0, result.output
        assert captured["visual_fixes"] == {"beat_003": "Bigger labels."}
        assert captured["ops"][0]["op"] == "fix_visual" and captured["beats"] == len(base_script["beats"])

    def test_convert_uses_confirmed_directives_without_reinterpreting(self, tmp_path, monkeypatch):
        import src.cli as cli

        pdf = tmp_path / "notes.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        directives = tmp_path / "d.json"
        directives.write_text(ProfessorDirectives(scope={"skip_sections": ["2.4"]}).model_dump_json(by_alias=True))
        captured = {}

        def fake_run(self, pdf_file_path, event_source="s3", directives=None, concept_plan=None):
            captured["directives"] = directives
            return SimpleNamespace(
                status="SUCCESS", output_directory=str(tmp_path), total_scenes=0, successful_scenes=0,
                delivered_scenes=None, lost_scenes=[], warning_count=0, files=[], final_video_path=None,
            )

        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        monkeypatch.setattr(Orchestrator, "run_pipeline", fake_run)
        monkeypatch.setattr(ProfessorInputInterpreter, "interpret_brief",
                            lambda *a, **k: pytest.fail("must not re-interpret a confirmed brief"))
        CliRunner().invoke(cli.pdf2manim, ["convert", str(pdf), "--directives", str(directives), "-o", str(tmp_path)])
        assert captured["directives"].scope.skip_sections == ["2.4"]


def test_app_builds_with_studio_and_quick_convert_tabs():
    import app

    demo = app.build_demo()
    assert len(demo.fns) == 6  # read back, generate, approve, review, apply, quick convert
