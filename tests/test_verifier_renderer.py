from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

from src.manim_verifier.preview_renderer import RawManimPreviewRenderer


SAFE_SCRIPT = """from manim import *
class Demo(Scene):
    def construct(self):
        self.add(Circle())
"""


def test_unsafe_source_is_never_executed(tmp_path, monkeypatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("subprocess.run", fake_run)
    report = RawManimPreviewRenderer().render(
        "import os\n" + SAFE_SCRIPT,
        tmp_path / "attempt",
    )
    assert not report.render_success
    assert not called
    assert "not executed" in report.render_error


def test_preview_render_locates_video_and_writes_logs(tmp_path, monkeypatch):
    renderer = RawManimPreviewRenderer()
    source_video = tmp_path / "rendered.mp4"
    source_video.write_bytes(b"video")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "stdout", "stderr"),
    )
    monkeypatch.setattr(renderer, "_find_video", lambda *args: source_video)
    monkeypatch.setattr(renderer, "_probe_duration", lambda *args: 3.5)

    report = renderer.render(SAFE_SCRIPT, tmp_path / "attempt")
    assert report.render_success
    assert Path(report.video_path).name == "video.mp4"
    assert Path(report.stdout_path).read_text(encoding="utf-8") == "stdout"
    assert report.duration_seconds == 3.5


def test_preview_render_timeout_is_structured(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutExpired(args[0], timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    report = RawManimPreviewRenderer().render(
        SAFE_SCRIPT,
        tmp_path / "attempt",
        timeout_seconds=1,
    )
    assert not report.render_success
    assert any(issue.code == "RENDER_TIMEOUT" for issue in report.issues)


def test_manim_error_is_captured(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 1, "", "ValueError: bad latex"),
    )
    report = RawManimPreviewRenderer().render(SAFE_SCRIPT, tmp_path / "attempt")
    assert not report.render_success
    assert "bad latex" in report.render_error


def test_api_keys_are_removed_from_renderer_environment_and_artifacts(tmp_path, monkeypatch):
    secret = "not-a-real-secret-value"
    captured_environment = {}
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    def fake_run(*args, **kwargs):
        captured_environment.update(kwargs["env"])
        return CompletedProcess(args[0], 1, "", "render failed")

    monkeypatch.setattr("subprocess.run", fake_run)
    report = RawManimPreviewRenderer().render(SAFE_SCRIPT, tmp_path / "attempt")
    assert "GEMINI_API_KEY" not in captured_environment
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path(report.stdout_path), Path(report.stderr_path))
    )
    assert secret not in artifact_text
