"""Hermetic tests for isolating execution of LLM-generated Manim code.

No Docker daemon, Manim binary or API key is needed: ``subprocess.run`` is
replaced and the argv / env / cwd it receives are asserted.
"""

from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.agents.manim_code_validator import ManimCodeValidatorAgent
from src.agents.video_renderer import (
    VideoRendererAgent,
    VideoRendererFailure,
    VideoRendererInput,
)
from src.cli import pdf2manim
from src.manim_verifier.preview_renderer import RawManimPreviewRenderer
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.utils import render_sandbox
from src.utils.render_sandbox import (
    CONTAINER_WORKDIR,
    REPO_ROOT,
    RenderSandboxConfig,
    container_user,
    is_secret_env_name,
)
from tests.test_cli import _cost_tracker_cls, _orchestrator_instance

SCENE = (
    "from manim import *\n\n"
    "class Demo(Scene):\n"
    "    def construct(self):\n"
    "        self.add(Circle())\n"
)

SECRETS = {
    "OPENAI_API_KEY": "sk-test-openai",
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "GEMINI_API_KEY": "gemini-test",
    "GOOGLE_API_KEY": "google-test",
    "AZURE_OPENAI_KEY": "azure-test",
    "HF_TOKEN": "hf-test",
    "GITHUB_TOKEN": "ghp-test",
    "AWS_ACCESS_KEY_ID": "AKIA-test",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-test",
    "DB_PASSWORD": "hunter2-test",
    "GOOGLE_APPLICATION_CREDENTIALS": "/home/me/creds-test.json",
    "SSH_AUTH_SOCK": "/tmp/ssh-agent-test.sock",
}
BENIGN = ("PDF2MANIM_SANDBOX_TEST_BENIGN", "keep-me")


@pytest.fixture
def secret_env(monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(*BENIGN)


class _Recorder:
    """Stands in for ``subprocess.run``; records every call."""

    def __init__(self, returncode=1, on_call=None):
        self.calls = []
        self.returncode = returncode
        self.on_call = on_call

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self.on_call is not None:
            self.on_call(argv, kwargs)
        return CompletedProcess(argv, self.returncode, "", "render failed")


@pytest.fixture
def fake_run(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr("subprocess.run", recorder)
    return recorder


def _write_scene(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scene_file = run_dir / "scene_001.py"
    scene_file.write_text(SCENE, encoding="utf-8")
    return run_dir, scene_file


def _render(tmp_path, sandbox=None):
    run_dir, scene_file = _write_scene(tmp_path)
    result = VideoRendererAgent(sandbox=sandbox)._render_scene(
        str(scene_file), "-ql", str(run_dir / "media")
    )
    return run_dir, scene_file, result


def _assert_no_secrets(env):
    assert env is not None, "env must be passed explicitly, never inherited"
    assert not set(SECRETS) & set(env)
    assert not set(SECRETS.values()) & set(env.values())
    assert env.get(BENIGN[0]) == BENIGN[1]


def _option(argv, flag):
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# (1) Native mode: env passed to the render subprocess carries no secrets
# ---------------------------------------------------------------------------

class TestNativeRenderEnvironment:
    def test_render_subprocess_env_has_no_secret_like_vars(self, tmp_path, secret_env, fake_run):
        _render(tmp_path)

        assert len(fake_run.calls) == 1
        argv, kwargs = fake_run.calls[0]
        assert argv[0] == "manim"
        _assert_no_secrets(kwargs.get("env"))
        assert "PATH" in kwargs["env"]

    def test_repo_root_is_on_pythonpath_for_src_helper_imports(self, tmp_path, secret_env, fake_run):
        _render(tmp_path)

        env = fake_run.calls[0][1]["env"]
        assert env["PYTHONPATH"].split(render_sandbox.os.pathsep)[0] == str(REPO_ROOT)

    def test_preview_renderer_shares_the_same_scrubbing(self, tmp_path, secret_env, fake_run):
        # AZURE_OPENAI_KEY / SSH_AUTH_SOCK slipped past the preview renderer's
        # old private list; both renderers now use render_sandbox.
        RawManimPreviewRenderer().render(SCENE, tmp_path / "attempt")

        _assert_no_secrets(fake_run.calls[0][1].get("env"))

    def test_dry_run_validation_env_has_no_secrets(self, secret_env, fake_run):
        ManimCodeValidatorAgent()._check_dry_run(SCENE, "s1", [])

        argv, kwargs = fake_run.calls[0]
        assert "--dry_run" in argv
        _assert_no_secrets(kwargs.get("env"))

    @pytest.mark.parametrize("name", sorted(SECRETS))
    def test_secret_names_are_detected(self, name):
        assert is_secret_env_name(name)

    @pytest.mark.parametrize(
        "name",
        ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMP", "PYTHONPATH",
         "LANG", "TEXMFHOME", "LOCALAPPDATA", "DOCKER_HOST"],
    )
    def test_render_essentials_are_kept(self, name):
        assert not is_secret_env_name(name)


# ---------------------------------------------------------------------------
# (2) Render subprocesses run with cwd = a per-scene scratch dir
# ---------------------------------------------------------------------------

class TestScratchWorkingDirectory:
    def test_render_cwd_is_the_per_scene_scratch_dir(self, tmp_path, fake_run):
        seen = {}
        fake_run.on_call = lambda argv, kwargs: seen.update(
            cwd_exists=Path(kwargs["cwd"]).is_dir()
        )
        run_dir, _, _ = _render(tmp_path)

        argv, kwargs = fake_run.calls[0]
        scratch = (run_dir / "render_scratch" / "scene_001").resolve()
        assert Path(kwargs["cwd"]) == scratch
        assert seen["cwd_exists"]
        assert Path(kwargs["cwd"]) != REPO_ROOT
        assert Path(kwargs["cwd"]) != Path.cwd()

    def test_paths_handed_to_manim_are_absolute(self, tmp_path, fake_run, monkeypatch):
        # A relative output dir must still resolve once cwd moves to scratch.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "scene_001.py").write_text(SCENE, encoding="utf-8")
        VideoRendererAgent()._render_scene("run/scene_001.py", "-ql", "run/media")

        argv = fake_run.calls[0][0]
        media = Path(_option(argv, "--media_dir"))
        assert media.is_absolute() and media == (tmp_path / "run" / "media").resolve()
        assert Path(argv[-2]).is_absolute()

    def test_unused_scratch_dir_is_removed_after_native_render(self, tmp_path, fake_run):
        run_dir, _, _ = _render(tmp_path)
        assert not (run_dir / "render_scratch").exists()

    def test_files_written_by_a_scene_stay_in_scratch(self, tmp_path, fake_run):
        fake_run.on_call = lambda argv, kwargs: (
            Path(kwargs["cwd"]) / "keys.env"
        ).write_text("stolen?", encoding="utf-8")
        run_dir, _, _ = _render(tmp_path)

        assert (run_dir / "render_scratch" / "scene_001" / "keys.env").exists()
        assert not (run_dir / "keys.env").exists()

    def test_stale_scratch_from_a_previous_attempt_is_wiped(self, tmp_path, fake_run):
        run_dir, scene_file = _write_scene(tmp_path)
        stale = run_dir / "render_scratch" / "scene_001" / "media"
        stale.mkdir(parents=True)
        (stale / "Demo.mp4").write_bytes(b"old")
        seen = {}
        fake_run.on_call = lambda argv, kwargs: seen.update(
            leftovers=list(Path(kwargs["cwd"]).iterdir())
        )

        VideoRendererAgent()._render_scene(str(scene_file), "-ql", str(run_dir / "media"))
        assert seen["leftovers"] == []

    def test_dry_run_cwd_is_a_scratch_dir_holding_only_the_scene(self, fake_run):
        seen = {}
        fake_run.on_call = lambda argv, kwargs: seen.update(
            cwd=Path(kwargs["cwd"]),
            files=sorted(p.name for p in Path(kwargs["cwd"]).iterdir()),
        )
        ManimCodeValidatorAgent()._check_dry_run(SCENE, "s1", [])

        assert seen["files"] == ["scene.py"]
        assert seen["cwd"] != REPO_ROOT
        assert not seen["cwd"].exists()  # thrown away afterwards

    def test_preview_render_cwd_is_the_attempt_dir(self, tmp_path, fake_run):
        RawManimPreviewRenderer().render(SCENE, tmp_path / "attempt")
        assert Path(fake_run.calls[0][1]["cwd"]) == (tmp_path / "attempt").resolve()


# ---------------------------------------------------------------------------
# (3) Docker mode: the argv carries the isolation flags
# ---------------------------------------------------------------------------

DOCKER = RenderSandboxConfig(mode="docker")


class TestDockerSandbox:
    def test_docker_argv_carries_isolation_flags(self, tmp_path, secret_env, fake_run):
        run_dir, _, _ = _render(tmp_path, sandbox=DOCKER)

        argv, kwargs = fake_run.calls[0]
        assert argv[:3] == ["docker", "run", "--rm"]
        assert _option(argv, "--network") == "none"
        assert "--read-only" in argv
        user = _option(argv, "--user")
        assert user.split(":")[0] not in ("0", "root")
        assert _option(argv, "--cap-drop") == "ALL"
        assert _option(argv, "--security-opt") == "no-new-privileges"
        assert _option(argv, "--cpus") == DOCKER.cpus
        assert _option(argv, "--memory") == DOCKER.memory
        assert _option(argv, "--memory-swap") == DOCKER.memory
        assert _option(argv, "--pids-limit") == str(DOCKER.pids_limit)
        assert _option(argv, "--pull") == "never"
        assert "noexec" in _option(argv, "--tmpfs")
        # Time limit enforced inside the container too.
        assert _option(argv, "--entrypoint") == "timeout"
        assert "--signal=KILL" in argv
        assert argv[argv.index("--signal=KILL") + 1] == str(VideoRendererAgent.RENDER_TIMEOUT_SECONDS)

    def test_exactly_one_writable_mount_and_it_is_the_scratch_dir(self, tmp_path, fake_run):
        run_dir, _, _ = _render(tmp_path, sandbox=DOCKER)

        argv = fake_run.calls[0][0]
        mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--mount"]
        assert len(mounts) == 1
        scratch = (run_dir / "render_scratch" / "scene_001").resolve()
        assert mounts[0] == f"type=bind,source={scratch},target={CONTAINER_WORKDIR}"
        for banned in ("-v", "--volume", "--volumes-from", "--privileged", "--device"):
            assert banned not in argv

    def test_no_secrets_reach_the_container(self, tmp_path, secret_env, fake_run):
        _render(tmp_path, sandbox=DOCKER)

        argv, kwargs = fake_run.calls[0]
        assert "--env-file" not in argv
        env_flags = [argv[i + 1] for i, arg in enumerate(argv) if arg in ("--env", "-e")]
        # Explicit NAME=value only: a bare NAME would copy it from the host.
        assert env_flags and all("=" in flag for flag in env_flags)
        assert not {flag.split("=", 1)[0] for flag in env_flags} & set(SECRETS)
        assert not any(value in " ".join(argv) for value in SECRETS.values())
        # The docker client process itself gets a scrubbed env as well.
        assert not set(SECRETS) & set(kwargs["env"])

    def test_container_command_sees_only_container_paths(self, tmp_path, fake_run):
        run_dir, _, _ = _render(tmp_path, sandbox=DOCKER)

        argv = fake_run.calls[0][0]
        inner = argv[argv.index(DOCKER.image) + 1:]
        assert inner[2:5] == ["python", "-m", "manim"]
        assert _option(inner, "--media_dir") == f"{CONTAINER_WORKDIR}/media"
        assert inner[-2:] == [f"{CONTAINER_WORKDIR}/scene_001.py", "Demo"]
        assert not any(str(tmp_path) in arg for arg in inner)
        # The scene was copied into the mounted scratch dir.
        copied = run_dir / "render_scratch" / "scene_001" / "scene_001.py"
        assert copied.read_text(encoding="utf-8") == SCENE

    def test_timeout_force_removes_the_container(self, tmp_path, monkeypatch):
        calls = []

        def fake(argv, **kwargs):
            calls.append(list(argv))
            if argv[1] == "run":
                raise TimeoutExpired(argv, kwargs.get("timeout"))
            return CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr("subprocess.run", fake)
        _, _, result = _render(tmp_path, sandbox=DOCKER)

        assert not result.success and "timed out" in result.error
        assert calls[1] == ["docker", "rm", "-f", _option(calls[0], "--name")]

    def test_container_exit_codes_get_an_operator_hint(self, tmp_path, monkeypatch):
        monkeypatch.setattr("subprocess.run", _Recorder(returncode=137))
        _, _, result = _render(tmp_path, sandbox=DOCKER)
        assert "time or memory limit" in result.error

    def test_dry_run_and_preview_also_use_the_container(self, tmp_path, fake_run):
        ManimCodeValidatorAgent(sandbox=DOCKER)._check_dry_run(SCENE, "s1", [])
        RawManimPreviewRenderer(sandbox=DOCKER).render(SCENE, tmp_path / "attempt")

        dry_run_argv, preview_argv = (call[0] for call in fake_run.calls)
        assert dry_run_argv[:2] == preview_argv[:2] == ["docker", "run"]
        assert "--dry_run" in dry_run_argv
        assert _option(preview_argv, "--mount") == (
            f"type=bind,source={(tmp_path / 'attempt').resolve()},target={CONTAINER_WORKDIR}"
        )

    def test_missing_docker_fails_fast_before_rendering(self, tmp_path, fake_run, monkeypatch):
        monkeypatch.setattr("src.agents.video_renderer.ffmpeg_available", lambda: True)
        monkeypatch.setattr("src.agents.video_renderer.docker_available", lambda config: False)
        _, scene_file = _write_scene(tmp_path)

        out = VideoRendererAgent(sandbox=DOCKER).execute(
            VideoRendererInput(scene_files=[str(scene_file)], output_dir=str(tmp_path / "run"))
        )
        assert isinstance(out, VideoRendererFailure) and "docker" in out.error
        assert fake_run.calls == []

    def test_root_host_runs_the_container_as_nobody(self, monkeypatch):
        monkeypatch.setattr(render_sandbox.os, "getuid", lambda: 0, raising=False)
        monkeypatch.setattr(render_sandbox.os, "getgid", lambda: 0, raising=False)
        assert container_user(DOCKER) == "65534:65534"

    def test_non_root_host_keeps_its_uid(self, monkeypatch):
        monkeypatch.setattr(render_sandbox.os, "getuid", lambda: 1000, raising=False)
        monkeypatch.setattr(render_sandbox.os, "getgid", lambda: 1000, raising=False)
        assert container_user(DOCKER) == "1000:1000"

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            RenderSandboxConfig(mode="chroot")


# ---------------------------------------------------------------------------
# Gating: off by default, opt-in via PipelineConfig / CLI
# ---------------------------------------------------------------------------

class TestSandboxGating:
    def test_docker_mode_is_off_by_default(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(base_output_dir=str(tmp_path)))
        assert PipelineConfig().render_sandbox == "native"
        assert not orch.video_renderer_agent.sandbox.is_docker
        assert not orch.manim_code_validator_agent.sandbox.is_docker

    def test_pipeline_config_reaches_every_executor(self, tmp_path):
        orch = Orchestrator(config=PipelineConfig(
            base_output_dir=str(tmp_path),
            render_sandbox="docker",
            render_sandbox_image="render:test",
        ))
        for agent in (orch.video_renderer_agent, orch.manim_code_validator_agent):
            assert agent.sandbox.is_docker and agent.sandbox.image == "render:test"

    @patch("src.cli.Orchestrator")
    @patch("src.cli.CostTracker", new_callable=_cost_tracker_cls)
    def test_cli_flag_sets_pipeline_config(self, mock_cost, mock_orch_cls, tmp_path, monkeypatch):
        monkeypatch.delenv("RENDER_SANDBOX", raising=False)
        mock_orch_cls.return_value = _orchestrator_instance()
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        runner = CliRunner()

        result = runner.invoke(pdf2manim, ["convert", str(pdf)])
        assert result.exit_code == 0, result.output
        assert mock_orch_cls.call_args[1]["config"].render_sandbox == "native"

        result = runner.invoke(pdf2manim, [
            "convert", str(pdf), "--render-sandbox", "docker", "--sandbox-image", "render:test",
        ])
        assert result.exit_code == 0, result.output
        config = mock_orch_cls.call_args[1]["config"]
        assert config.render_sandbox == "docker"
        assert config.render_sandbox_image == "render:test"
