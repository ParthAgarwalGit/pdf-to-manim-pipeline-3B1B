"""V2 renders go through the render sandbox.

Hermetic: ``subprocess.Popen`` / ``subprocess.run`` / ``shutil.which`` are
replaced, so no Docker daemon, Manim binary or API key is needed. The one real
render at the bottom is marked ``integration``.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.pipeline_v2 import rendering
from src.pipeline_v2.models import (
    ArtifactCategory,
    ArtifactSpec,
    BeatComposition,
    LayoutRegion,
    LayoutTemplate,
    NarratedScene,
    NarrationBeat,
    VisualContract,
)
from src.pipeline_v2.rendering import TrackedManimRenderer
from src.utils import render_sandbox
from src.utils.render_sandbox import (
    CONTAINER_WORKDIR,
    DOCKER_GRACE_SECONDS,
    REPO_ROOT,
    ManimProcess,
    RenderSandboxConfig,
    build_docker_argv,
    container_path,
    host_timeout,
    kill_process_tree,
    start_manim,
)


CODE = '''from manim import *
from src.pipeline_v2.tracked_scene import TrackedScene

class Demo(TrackedScene):
    def construct(self):
        self.set_layout("full_visual")
        dot = Dot()
        self.place_in_region(dot, "content_full")
        self.register_artifact("dot", dot, category="primary_visual", purpose="Show", beat_ids=["b1"])
        self.set_beat("b1")
        self.play(FadeIn(dot))
'''

SECRETS = {
    "OPENAI_API_KEY": "sk-test-openai",
    "GEMINI_API_KEY": "gemini-test-value",
    "DID_API_KEY": "did-test-value",
    "AZURE_OPENAI_KEY": "azure-test-value",
    "AWS_ACCESS_KEY_ID": "AKIA-test-value",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-test-value",
    "GITHUB_TOKEN": "ghp-test-value",
    "SSH_AUTH_SOCK": "/tmp/ssh-agent-test.sock",
    "MY_PASSWORD": "hunter2-test-value",
}
BENIGN = ("PDF2MANIM_SANDBOX_TEST_BENIGN", "keep-me")
DOCKER = RenderSandboxConfig(mode="docker")
FAKE_PID = 424242


def _scene_and_contract():
    scene = NarratedScene(
        scene_id="scene", narration="Show the dot.",
        beats=[NarrationBeat(
            beat_id="b1", text="Show", source_fact_ids=["f1"],
            start_seconds=0.0, end_seconds=1.0,
        )],
    )
    contract = VisualContract(
        scene_id="scene", layout_template=LayoutTemplate.FULL_VISUAL,
        artifact_inventory=[ArtifactSpec(
            artifact_id="dot", category=ArtifactCategory.PRIMARY_VISUAL,
            role="dot", purpose="Show", source_fact_ids=["f1"], beat_ids=["b1"],
            enter_beat="b1", exit_beat="b1",
            region=LayoutRegion.CONTENT_FULL,
        )],
        composition_plan=[BeatComposition(beat_id="b1", active_artifact_ids=["dot"])],
    )
    return scene, contract


class FakeProcess:
    """Stands in for a ``subprocess.Popen`` object."""

    pid = FAKE_PID

    def __init__(self, argv, kwargs, spawner):
        self.args = argv
        self.kwargs = kwargs
        self.spawner = spawner
        self.returncode = None
        self.timeouts = []
        self.kill_calls = 0
        self.waiting = threading.Event()
        self._killed = threading.Event()

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self.timeouts.append(timeout)
        self.waiting.set()
        if self.spawner.mode == "block":  # runs until somebody kills it
            self._killed.wait(20)
        if self._killed.is_set():
            self.returncode = -9
            return "partial stdout", "partial stderr"
        if self.spawner.mode == "hang":
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = self.spawner.returncode
        return "", "" if self.returncode == 0 else "render failed"

    def kill(self):
        self.kill_calls += 1
        self._killed.set()


class Spawner:
    """Stands in for ``subprocess.Popen``; records every process it starts."""

    def __init__(self):
        self.calls = []
        self.returncode = 1
        self.mode = "exit"  # "exit" | "hang" (until killed) | "block" (until killed)
        self.on_start = None

    def __call__(self, argv, **kwargs):
        process = FakeProcess(list(argv), kwargs, self)
        self.calls.append(process)
        if self.on_start is not None:
            self.on_start(process)
        return process


@pytest.fixture
def sandbox(monkeypatch):
    """Replace every way the sandbox reaches the operating system."""
    spawner = Spawner()
    runs = []
    killed = []

    def fake_run(argv, **kwargs):
        runs.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fake_kill_tree(process):
        killed.append(process)
        process.kill()

    monkeypatch.setattr(render_sandbox.subprocess, "Popen", spawner)
    monkeypatch.setattr(render_sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(render_sandbox, "kill_process_tree", fake_kill_tree)
    monkeypatch.setattr(render_sandbox.shutil, "which", lambda name: f"/usr/bin/{name}")
    return SimpleNamespace(spawner=spawner, runs=runs, killed=killed)


@pytest.fixture
def secret_env(monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(*BENIGN)


def _render(tmp_path, renderer=None, *, code=CODE, name="attempt", timeout_seconds=30):
    scene, contract = _scene_and_contract()
    renderer = renderer or TrackedManimRenderer()
    return renderer.render(
        code, contract, scene, tmp_path / name, timeout_seconds=timeout_seconds,
    )


def _codes(outcome) -> set[str]:
    return {finding.code for finding in outcome.findings}


def _option(argv, flag):
    return argv[argv.index(flag) + 1]


def _env_flags(argv) -> dict[str, str]:
    flags = [argv[i + 1] for i, arg in enumerate(argv) if arg in ("--env", "-e")]
    # Explicit NAME=value only: a bare NAME would copy the value from the host.
    assert flags and all("=" in flag for flag in flags)
    return dict(flag.split("=", 1) for flag in flags)


def _assert_no_secrets(env):
    assert env is not None, "env must be passed explicitly, never inherited"
    assert not set(SECRETS) & set(env)
    assert not [value for value in env.values() if value in SECRETS.values()]
    assert env.get(BENIGN[0]) == BENIGN[1]


# ---------------------------------------------------------------------------
# (1) Native mode: the Manim child gets a scrubbed environment
# ---------------------------------------------------------------------------

class TestNativeEnvironment:
    def test_child_env_has_no_secrets_and_keeps_what_manim_needs(
        self, tmp_path, secret_env, sandbox,
    ):
        _render(tmp_path)

        assert len(sandbox.spawner.calls) == 1
        env = sandbox.spawner.calls[0].kwargs.get("env")
        _assert_no_secrets(env)
        assert "PATH" in env
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO_ROOT)

    def test_child_env_carries_the_tracking_variables(self, tmp_path, sandbox):
        _render(tmp_path)

        env = sandbox.spawner.calls[0].kwargs["env"]
        assert Path(env["PDF2MANIM_TRACE_PATH"]).name == "trace.jsonl"
        assert json.loads(env["PDF2MANIM_BEAT_SCHEDULE"]) == [
            {"beat_id": "b1", "start": 0.0, "end": 1.0},
        ]

    def test_environment_helper_matches_the_native_child_env(self, tmp_path, secret_env):
        scene, _ = _scene_and_contract()
        env = TrackedManimRenderer._environment(tmp_path / "trace.jsonl", scene)

        _assert_no_secrets(env)
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO_ROOT)
        assert env["PDF2MANIM_TRACE_PATH"] == str(tmp_path / "trace.jsonl")
        assert json.loads(env["PDF2MANIM_BEAT_SCHEDULE"])[0]["beat_id"] == "b1"

    def test_default_mode_is_native(self, tmp_path, sandbox):
        renderer = TrackedManimRenderer()
        assert renderer.sandbox == RenderSandboxConfig()
        assert renderer.sandbox.mode == "native" and not renderer.sandbox.is_docker

        _render(tmp_path, renderer)

        process = sandbox.spawner.calls[0]
        assert process.args[:3] == [sys.executable, "-m", "manim"]
        assert process.args[3:5] == ["-ql", "--disable_caching"]
        assert process.args[-1] == "Demo"
        assert process.timeouts == [30]
        assert process.kwargs["stdout"] == process.kwargs["stderr"] == subprocess.PIPE
        assert sandbox.runs == []  # no docker client involved


# ---------------------------------------------------------------------------
# (2) cwd is a fresh per-attempt scratch dir
# ---------------------------------------------------------------------------

class TestScratchWorkingDirectory:
    def test_cwd_is_a_fresh_scratch_dir_per_render(self, tmp_path, sandbox):
        seen = []
        sandbox.spawner.on_start = lambda process: seen.append(
            sorted(path.name for path in Path(process.kwargs["cwd"]).iterdir())
        )
        renderer = TrackedManimRenderer()
        _render(tmp_path, renderer, name="attempt-00")
        _render(tmp_path, renderer, name="attempt-01")

        first, second = (Path(process.kwargs["cwd"]) for process in sandbox.spawner.calls)
        assert first != second
        assert seen == [["scene.py"], ["scene.py"]]  # existed, holding only the scene
        for cwd in (first, second):
            assert cwd.is_absolute()
            assert cwd not in (REPO_ROOT, Path.cwd())
            assert REPO_ROOT not in cwd.parents
            assert tmp_path not in cwd.parents  # not the durable attempt dir either
            assert not cwd.exists()  # thrown away afterwards

    def test_paths_handed_to_manim_live_in_the_scratch_dir(self, tmp_path, sandbox):
        _render(tmp_path)

        process = sandbox.spawner.calls[0]
        cwd = Path(process.kwargs["cwd"])
        assert Path(process.args[-2]) == cwd / "scene.py"
        assert Path(_option(process.args, "--media_dir")) == cwd / "media"
        assert Path(process.kwargs["env"]["PDF2MANIM_TRACE_PATH"]) == cwd / "trace.jsonl"

    def test_files_written_by_a_scene_stay_out_of_the_repo_and_run_dir(
        self, tmp_path, sandbox, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        sandbox.spawner.on_start = lambda process: (
            Path(process.kwargs["cwd"]) / "keys.env"
        ).write_text("stolen?", encoding="utf-8")
        _render(tmp_path)

        assert not (tmp_path / "keys.env").exists()
        assert not (tmp_path / "attempt" / "keys.env").exists()


# ---------------------------------------------------------------------------
# (3) The trace is written in the scratch dir and copied to attempt_dir
# ---------------------------------------------------------------------------

def _write_trace(process):
    (Path(process.kwargs["cwd"]) / "trace.jsonl").write_text(
        '{"event": "play"}\n', encoding="utf-8",
    )


class TestTraceEvidence:
    @pytest.mark.parametrize(
        ("mode", "returncode", "expected"),
        [
            ("exit", 1, "MANIM_RENDER_FAILED"),
            ("exit", 0, "VIDEO_NOT_FOUND"),
            ("hang", 1, "RENDER_TIMEOUT"),
        ],
    )
    def test_trace_is_copied_out_when_the_render_does_not_succeed(
        self, tmp_path, sandbox, mode, returncode, expected,
    ):
        sandbox.spawner.mode = mode
        sandbox.spawner.returncode = returncode
        sandbox.spawner.on_start = _write_trace
        outcome = _render(tmp_path)

        assert expected in _codes(outcome)
        assert not outcome.render_success
        copied = tmp_path / "attempt" / "trace.jsonl"
        assert copied.read_text(encoding="utf-8") == '{"event": "play"}\n'

    def test_successful_render_reports_the_durable_trace_and_video(
        self, tmp_path, sandbox, monkeypatch,
    ):
        def finish_render(process):
            _write_trace(process)
            video = Path(_option(process.args, "--media_dir")) / "videos" / "scene" / "480p15"
            video.mkdir(parents=True)
            (video / "Demo.mp4").write_bytes(b"fake video")

        sandbox.spawner.returncode = 0
        sandbox.spawner.on_start = finish_render
        analyzed = []
        monkeypatch.setattr(rendering, "probe_duration", lambda path: 1.0)
        monkeypatch.setattr(rendering, "extract_frames", lambda *args: [])
        monkeypatch.setattr(
            rendering, "analyze_trace",
            lambda contract, scene, trace, duration, **kwargs: analyzed.append(trace) or [],
        )
        outcome = _render(tmp_path)

        attempt = (tmp_path / "attempt").resolve()
        assert outcome.render_success, outcome.error
        assert outcome.trace_path == str(attempt / "trace.jsonl")
        assert analyzed == [attempt / "trace.jsonl"]
        assert (attempt / "trace.jsonl").read_text(encoding="utf-8") == '{"event": "play"}\n'
        assert (attempt / "video.mp4").read_bytes() == b"fake video"
        assert not Path(sandbox.spawner.calls[0].kwargs["cwd"]).exists()

    def test_render_without_a_trace_does_not_invent_one(self, tmp_path, sandbox):
        _render(tmp_path)
        assert not (tmp_path / "attempt" / "trace.jsonl").exists()


# ---------------------------------------------------------------------------
# (4) Timeout kills the whole process tree
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_kills_the_process_tree_and_is_structured(self, tmp_path, sandbox):
        sandbox.spawner.mode = "hang"
        renderer = TrackedManimRenderer()
        outcome = _render(tmp_path, renderer, timeout_seconds=7)

        process = sandbox.spawner.calls[0]
        assert sandbox.killed == [process]
        assert process.timeouts[0] == 7
        assert not outcome.render_success
        assert outcome.error == "Manim render timed out after 7s."
        assert _codes(outcome) == {"RENDER_TIMEOUT"}
        attempt = tmp_path / "attempt"
        assert (attempt / "render_stdout.txt").read_text(encoding="utf-8") == "partial stdout"
        assert (attempt / "render_stderr.txt").read_text(encoding="utf-8") == "partial stderr"
        assert renderer._processes == set()

    def test_communicate_reraises_timeout_with_the_drained_output(self, sandbox):
        sandbox.spawner.mode = "hang"
        handle = ManimProcess(process=sandbox.spawner(["manim"]), sandbox=RenderSandboxConfig())

        with pytest.raises(subprocess.TimeoutExpired) as raised:
            handle.communicate(timeout=3)

        assert raised.value.timeout == 3
        assert (raised.value.stdout, raised.value.stderr) == ("partial stdout", "partial stderr")
        assert sandbox.killed == [handle.process]
        assert handle.returncode == -9

    @pytest.mark.skipif(os.name != "nt", reason="taskkill is the Windows tree kill")
    def test_kill_process_tree_uses_taskkill_on_windows(self, monkeypatch):
        runs = []
        monkeypatch.setattr(
            render_sandbox.subprocess, "run",
            lambda argv, **kwargs: runs.append((list(argv), kwargs)),
        )
        process = FakeProcess(["manim"], {}, Spawner())

        kill_process_tree(process)

        assert [argv for argv, _ in runs] == [["taskkill", "/T", "/F", "/PID", str(FAKE_PID)]]
        assert runs[0][1]["capture_output"] and runs[0][1]["timeout"]
        assert process.kill_calls == 1  # fallback for the process itself

    @pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX only")
    def test_kill_process_tree_kills_the_process_group_on_posix(self, monkeypatch):
        killed = []
        monkeypatch.setattr(
            render_sandbox.os, "getpgid", lambda pid: FAKE_PID if pid == FAKE_PID else 1,
        )
        monkeypatch.setattr(render_sandbox.os, "killpg", lambda group, sig: killed.append(group))
        process = FakeProcess(["manim"], {}, Spawner())

        kill_process_tree(process)

        assert killed == [FAKE_PID]
        assert process.kill_calls == 1

    def test_kill_process_tree_skips_the_tree_kill_once_the_process_exited(self, monkeypatch):
        runs = []
        monkeypatch.setattr(render_sandbox.subprocess, "run", lambda *a, **k: runs.append(a))
        monkeypatch.setattr(render_sandbox.os, "killpg", lambda *a: runs.append(a), raising=False)
        process = FakeProcess(["manim"], {}, Spawner())
        process.returncode = 0  # its pid may already belong to someone else

        kill_process_tree(process)

        assert runs == []

    def test_kill_process_tree_never_raises(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("no such tool")

        monkeypatch.setattr(render_sandbox.subprocess, "run", boom)
        monkeypatch.setattr(render_sandbox.os, "getpgid", boom, raising=False)
        process = FakeProcess(["manim"], {}, Spawner())
        process.kill = boom

        kill_process_tree(process)


# ---------------------------------------------------------------------------
# (5) terminate_all reaches renders that are still running
# ---------------------------------------------------------------------------

def _render_in_background(tmp_path, renderer, sandbox):
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(outcome=_render(tmp_path, renderer)), daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not sandbox.spawner.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sandbox.spawner.calls, "render never started"
    # communicate() starts only after the handle is tracked.
    assert sandbox.spawner.calls[0].waiting.wait(10)
    return thread, result


class TestTerminateAll:
    def test_terminate_all_kills_tracked_handles(self, tmp_path, sandbox):
        sandbox.spawner.mode = "block"
        renderer = TrackedManimRenderer()
        thread, result = _render_in_background(tmp_path, renderer, sandbox)

        assert [handle.process for handle in renderer._processes] == sandbox.spawner.calls
        assert all(isinstance(handle, ManimProcess) for handle in renderer._processes)
        renderer.terminate_all()
        thread.join(10)

        assert not thread.is_alive()
        assert sandbox.killed == sandbox.spawner.calls
        assert renderer._processes == set()
        assert "MANIM_RENDER_FAILED" in _codes(result["outcome"])

    def test_terminate_all_also_removes_the_container(self, tmp_path, sandbox):
        sandbox.spawner.mode = "block"
        renderer = TrackedManimRenderer(sandbox=DOCKER)
        thread, _ = _render_in_background(tmp_path, renderer, sandbox)

        renderer.terminate_all()
        thread.join(10)

        assert not thread.is_alive()
        name = _option(sandbox.spawner.calls[0].args, "--name")
        assert sandbox.runs == [["docker", "rm", "-f", name]]

    def test_terminate_all_with_nothing_running_is_a_no_op(self, sandbox):
        TrackedManimRenderer().terminate_all()
        assert sandbox.killed == [] and sandbox.runs == []

    def test_kill_is_idempotent_and_never_raises(self, sandbox, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("docker daemon is gone")

        monkeypatch.setattr(render_sandbox.subprocess, "run", boom)
        handle = ManimProcess(
            process=sandbox.spawner(["docker", "run"]), sandbox=DOCKER, container_name="c1",
        )

        handle.kill()
        handle.kill()

        assert sandbox.killed == [handle.process, handle.process]


# ---------------------------------------------------------------------------
# (6)-(8) Docker mode
# ---------------------------------------------------------------------------

class TestDockerMode:
    def test_docker_argv_carries_isolation_flags(self, tmp_path, secret_env, sandbox):
        _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER), timeout_seconds=45)

        process = sandbox.spawner.calls[0]
        argv = process.args
        assert argv[:3] == ["docker", "run", "--rm"]
        assert _option(argv, "--network") == "none"
        assert "--read-only" in argv
        assert _option(argv, "--cap-drop") == "ALL"
        assert _option(argv, "--security-opt") == "no-new-privileges"
        assert _option(argv, "--pids-limit") == str(DOCKER.pids_limit)
        assert _option(argv, "--memory") == DOCKER.memory
        assert _option(argv, "--memory-swap") == DOCKER.memory
        assert _option(argv, "--cpus") == DOCKER.cpus
        assert _option(argv, "--pull") == "never"
        assert _option(argv, "--user").split(":")[0] not in ("0", "root")
        # The time limit holds inside the container; the host waits a little longer.
        assert _option(argv, "--entrypoint") == "timeout"
        assert argv[argv.index("--signal=KILL") + 1] == "45"
        assert process.timeouts == [45 + DOCKER_GRACE_SECONDS]

    def test_exactly_one_mount_and_it_is_the_scratch_dir(self, tmp_path, sandbox):
        _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER))

        process = sandbox.spawner.calls[0]
        mounts = [process.args[i + 1] for i, arg in enumerate(process.args) if arg == "--mount"]
        scratch = process.kwargs["cwd"]
        assert mounts == [f"type=bind,source={scratch},target={CONTAINER_WORKDIR}"]
        assert tmp_path not in Path(scratch).parents  # the run dir is not exposed
        for banned in ("-v", "--volume", "--volumes-from", "--privileged", "--device"):
            assert banned not in process.args

    def test_container_env_is_explicit_and_carries_no_host_secrets(
        self, tmp_path, secret_env, sandbox,
    ):
        _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER))

        process = sandbox.spawner.calls[0]
        assert "--env-file" not in process.args
        env = _env_flags(process.args)
        assert set(env) == {
            "HOME", "PYTHONDONTWRITEBYTECODE",
            "PDF2MANIM_TRACE_PATH", "PDF2MANIM_BEAT_SCHEDULE",
        }
        assert env["PDF2MANIM_TRACE_PATH"] == f"{CONTAINER_WORKDIR}/trace.jsonl"
        assert json.loads(env["PDF2MANIM_BEAT_SCHEDULE"]) == [
            {"beat_id": "b1", "start": 0.0, "end": 1.0},
        ]
        joined = " ".join(process.args)
        assert not [name for name in SECRETS if name in joined]
        assert not [value for value in SECRETS.values() if value in joined]
        # The docker client process itself gets a scrubbed env as well.
        _assert_no_secrets(process.kwargs["env"])

    def test_container_command_sees_only_container_paths(self, tmp_path, sandbox):
        _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER))

        process = sandbox.spawner.calls[0]
        inner = process.args[process.args.index(DOCKER.image) + 1:]
        assert inner[2:] == [
            "python", "-m", "manim", "-ql", "--disable_caching",
            "--media_dir", f"{CONTAINER_WORKDIR}/media",
            f"{CONTAINER_WORKDIR}/scene.py", "Demo",
        ]
        assert not [arg for arg in inner if process.kwargs["cwd"] in arg]

    def test_trace_written_through_the_mount_is_copied_out(self, tmp_path, sandbox):
        sandbox.spawner.on_start = _write_trace
        _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER))

        assert (tmp_path / "attempt" / "trace.jsonl").exists()

    def test_timeout_force_removes_the_container(self, tmp_path, sandbox):
        sandbox.spawner.mode = "hang"
        outcome = _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER), timeout_seconds=5)

        process = sandbox.spawner.calls[0]
        assert sandbox.killed == [process]  # the docker client
        assert sandbox.runs == [["docker", "rm", "-f", _option(process.args, "--name")]]
        assert _codes(outcome) == {"RENDER_TIMEOUT"}
        assert outcome.error == "Manim render timed out after 5s."

    def test_missing_docker_fails_without_spawning(self, tmp_path, sandbox, monkeypatch):
        monkeypatch.setattr(render_sandbox.shutil, "which", lambda name: None)
        outcome = _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER))

        assert sandbox.spawner.calls == [] and sandbox.runs == []
        assert outcome.safe and not outcome.render_success
        assert _codes(outcome) == {"RENDER_START_FAILED"}
        assert "'docker'" in outcome.error
        assert "docker" in (tmp_path / "attempt" / "render_stderr.txt").read_text(encoding="utf-8")

    def test_missing_docker_names_the_configured_binary(self, tmp_path, sandbox, monkeypatch):
        monkeypatch.setattr(render_sandbox.shutil, "which", lambda name: None)
        podman = RenderSandboxConfig(mode="docker", docker_binary="podman")
        outcome = _render(tmp_path, TrackedManimRenderer(sandbox=podman))

        assert "'podman'" in outcome.error
        assert sandbox.spawner.calls == []

    @pytest.mark.parametrize(
        ("returncode", "hint"),
        [(137, "time or memory limit"), (125, "built locally")],
    )
    def test_container_exit_codes_get_an_operator_hint(
        self, tmp_path, sandbox, returncode, hint,
    ):
        sandbox.spawner.returncode = returncode
        outcome = _render(tmp_path, TrackedManimRenderer(sandbox=DOCKER))

        assert _codes(outcome) == {"MANIM_RENDER_FAILED"}
        assert f"Manim exited with {returncode}" in outcome.error
        assert hint in outcome.error

    def test_native_exit_gets_no_sandbox_hint(self, tmp_path, sandbox):
        sandbox.spawner.returncode = 137
        outcome = _render(tmp_path)
        assert "sandbox" not in outcome.error


# ---------------------------------------------------------------------------
# (9) Unsafe code is never executed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["native", "docker"])
@pytest.mark.parametrize(
    "unsafe_line",
    [
        "        text = open('keys.env').read()",
        "        icon = SVGMobject('keys.svg')",
        '        label = Tex(r"\\input{keys.env}")',
    ],
)
def test_unsafe_code_is_never_executed(tmp_path, sandbox, mode, unsafe_line):
    code = CODE.replace("        dot = Dot()", f"{unsafe_line}\n        dot = Dot()")
    renderer = TrackedManimRenderer(sandbox=RenderSandboxConfig(mode=mode))
    outcome = _render(tmp_path, renderer, code=code)

    assert sandbox.spawner.calls == [] and sandbox.runs == []
    assert not outcome.safe and not outcome.render_success
    assert outcome.error == "Static V2 verification failed; code was not executed."


# ---------------------------------------------------------------------------
# render_sandbox building blocks
# ---------------------------------------------------------------------------

class TestSandboxHelpers:
    def test_extra_env_follows_the_fixed_entries(self, tmp_path):
        argv = build_docker_argv(
            config=DOCKER, scratch_dir=tmp_path, container_command=["python"],
            timeout_seconds=10, name="c1", extra_env={"A": "1", "B": "x=y"},
        )
        env_flags = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--env"]
        assert env_flags == ["HOME=/tmp", "PYTHONDONTWRITEBYTECODE=1", "A=1", "B=x=y"]
        assert argv.index("B=x=y") < argv.index("--entrypoint")

    def test_argv_is_unchanged_without_extra_env(self, tmp_path):
        options = dict(
            config=DOCKER, scratch_dir=tmp_path, container_command=["python"],
            timeout_seconds=10, name="c1",
        )
        assert build_docker_argv(**options) == build_docker_argv(**options, extra_env={})

    def test_container_path_is_public_and_keeps_its_alias(self, tmp_path):
        assert render_sandbox._container_path is container_path
        assert container_path(tmp_path, tmp_path) == CONTAINER_WORKDIR
        assert container_path(tmp_path / "a" / "b.py", tmp_path) == f"{CONTAINER_WORKDIR}/a/b.py"
        with pytest.raises(ValueError):
            container_path(tmp_path.parent / "outside.py", tmp_path)

    def test_host_timeout_adds_grace_only_for_docker(self):
        assert host_timeout(None, 60) == 60
        assert host_timeout(RenderSandboxConfig(), 60) == 60
        assert host_timeout(DOCKER, 60) == 60 + DOCKER_GRACE_SECONDS

    def test_start_manim_native_merges_extra_env_over_a_scrubbed_env(
        self, tmp_path, secret_env, sandbox,
    ):
        handle = start_manim(
            scene_path=tmp_path / "scene.py", scene_class="Demo", flags=["-ql"],
            media_dir=tmp_path / "media", scratch_dir=tmp_path, timeout_seconds=10,
            extra_env={"PDF2MANIM_TRACE_PATH": "trace.jsonl"},
        )

        process = handle.process
        assert handle.container_name is None and handle.returncode is None
        assert process.args == [
            "manim", "-ql", "--media_dir", str((tmp_path / "media").resolve()),
            str((tmp_path / "scene.py").resolve()), "Demo",
        ]
        assert process.kwargs["cwd"] == str(tmp_path.resolve())
        assert process.kwargs["text"] and process.kwargs["encoding"] == "utf-8"
        assert process.kwargs["errors"] == "replace"
        assert process.kwargs.get("start_new_session", False) == (os.name != "nt")
        _assert_no_secrets(process.kwargs["env"])
        assert process.kwargs["env"]["PDF2MANIM_TRACE_PATH"] == "trace.jsonl"

    def test_start_manim_docker_rejects_paths_outside_the_mount(self, tmp_path, sandbox):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        with pytest.raises(ValueError):
            start_manim(
                scene_path=tmp_path / "scene.py", scene_class="Demo", flags=["-ql"],
                media_dir=scratch / "media", scratch_dir=scratch, timeout_seconds=10,
                sandbox=DOCKER,
            )
        assert sandbox.spawner.calls == []


# ---------------------------------------------------------------------------
# One real native render
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_native_sandbox_renders_a_tracked_scene_for_real(tmp_path, secret_env, monkeypatch):
    scratch_dirs = []
    real_start = rendering.start_manim

    def recording_start(**kwargs):
        scratch_dirs.append(Path(kwargs["scratch_dir"]))
        return real_start(**kwargs)

    monkeypatch.setattr(rendering, "start_manim", recording_start)
    scene, contract = _scene_and_contract()
    attempt = tmp_path / "attempt"

    outcome = TrackedManimRenderer().render(
        CODE, contract, scene, attempt, timeout_seconds=120,
    )

    assert outcome.safe
    assert outcome.render_success, outcome.error
    assert outcome.trace_path == str(attempt.resolve() / "trace.jsonl")
    trace = (attempt / "trace.jsonl").read_text(encoding="utf-8")
    assert '"artifact_id": "dot"' in trace
    assert (attempt / "video.mp4").stat().st_size > 0
    assert len(scratch_dirs) == 1 and not scratch_dirs[0].exists()
