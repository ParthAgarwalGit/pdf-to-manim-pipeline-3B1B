"""Isolation for executing LLM-generated Manim code.

The AST gates (``ManimCodeValidatorAgent._check_security`` and
``StaticManimVerifier``) reject known-dangerous constructs, but a denylist can
never be complete: Manim itself ships mobjects that read files
(``SVGMobject``, ``ImageMobject``, ``Code``) and ``import manim`` reaches
helpers the gate does not model. This module makes the *execution* safe
regardless of what slips past the screen.

Two modes, chosen by ``RenderSandboxConfig.mode``:

* ``native`` (default): the Manim subprocess gets an environment with every
  secret-like variable removed and runs with ``cwd`` set to a per-scene
  scratch directory, so relative paths such as ``keys.env`` no longer resolve
  to the repository root. The code still runs as the host user with host
  filesystem and network access.
* ``docker`` (opt-in): the subprocess is ``docker run`` with no network, a
  non-root user, a read-only root filesystem, CPU/memory/pids/time limits and
  exactly one writable bind mount (the scratch directory). The container gets
  no env file, no host environment and no other mounts, so there is nothing
  secret inside it to read or send.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

# Repository root (the directory that contains the ``src`` package). Generated
# scenes import ``src.utils.*`` helpers; once ``cwd`` moves to a scratch dir
# the native subprocess needs this on PYTHONPATH to import the same code.
REPO_ROOT = Path(__file__).resolve().parents[2]

SANDBOX_MODES = ("native", "docker")

# Path the scratch directory is mounted at inside the container.
CONTAINER_WORKDIR = "/work"

# Extra wall-clock the host allows on top of the in-container time limit for
# container start-up and teardown before it force-removes the container.
DOCKER_GRACE_SECONDS = 30

# How long a killed render gets to hand over its remaining stdout/stderr.
KILL_DRAIN_SECONDS = 10

# Uid/gid of ``nobody``: the container user when the host user is root.
NOBODY = "65534:65534"

_SECRET_NAME_FRAGMENTS = (
    "API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD",
    "CREDENTIAL", "PRIVATE_KEY", "ACCESS_KEY", "AUTH",
)
_SECRET_NAME_SUFFIXES = ("_KEY", "_PAT")


def is_secret_env_name(name: str) -> bool:
    """True for variable names that conventionally hold credentials."""
    upper = name.upper()
    return (
        any(fragment in upper for fragment in _SECRET_NAME_FRAGMENTS)
        or upper.endswith(_SECRET_NAME_SUFFIXES)
    )


def scrub_secret_env(source: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Copy of ``source`` (default ``os.environ``) without secret-like vars."""
    env = os.environ if source is None else source
    return {key: value for key, value in env.items() if not is_secret_env_name(key)}


def native_render_env(
    source: Optional[Mapping[str, str]] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Environment for a native Manim subprocess: scrubbed, repo on PYTHONPATH.

    With ``audit_path``, the layout audit's sitecustomize goes first on the
    path and writes its findings there (src/utils/layout_audit.py).
    """
    env = scrub_secret_env(source)
    parts = [str(REPO_ROOT), env.get("PYTHONPATH")]
    if audit_path is not None:
        from src.utils.layout_audit import ENV_VAR, SITE_DIR

        parts.insert(1, str(SITE_DIR))  # after the repo root, before site-packages
        env[ENV_VAR] = str(Path(audit_path).resolve())
    env["PYTHONPATH"] = os.pathsep.join(part for part in parts if part)
    return env


@dataclass(frozen=True)
class RenderSandboxConfig:
    """How generated Manim code is executed. Defaults to ``native``."""

    mode: str = "native"
    # Local image with Manim, LaTeX and the ``src`` package installed (the
    # repo's own Dockerfile builds it). Never pulled from a registry.
    image: str = "pdf2manim:latest"
    cpus: str = "2"
    memory: str = "2g"
    pids_limit: int = 256
    tmpfs_size: str = "512m"
    # ``uid:gid`` inside the container; None = host uid:gid when non-root,
    # otherwise ``nobody``.
    user: Optional[str] = None
    docker_binary: str = "docker"

    def __post_init__(self) -> None:
        if self.mode not in SANDBOX_MODES:
            raise ValueError(
                f"Unknown render sandbox mode {self.mode!r}; expected one of {SANDBOX_MODES}."
            )

    @property
    def is_docker(self) -> bool:
        return self.mode == "docker"


def docker_available(config: RenderSandboxConfig) -> bool:
    return shutil.which(config.docker_binary) is not None


def container_user(config: RenderSandboxConfig) -> str:
    """Non-root ``uid:gid`` the render container runs as."""
    if config.user:
        return config.user
    getuid = getattr(os, "getuid", None)  # absent on Windows
    if getuid is not None and getuid() != 0:
        return f"{getuid()}:{os.getgid()}"
    return NOBODY


def build_docker_argv(
    *,
    config: RenderSandboxConfig,
    scratch_dir: Path,
    container_command: Sequence[str],
    timeout_seconds: int,
    name: str,
    extra_env: Optional[Mapping[str, str]] = None,
) -> list:
    """``docker run`` argv that executes ``container_command`` in isolation.

    The only host path the container can see is ``scratch_dir`` (mounted
    read-write at ``CONTAINER_WORKDIR``). Environment variables are set
    explicitly (the two fixed ones plus ``extra_env``, whose path values must
    already be container paths); nothing is inherited from the host and no
    env file is used.
    """
    extra_env_flags = [
        item
        for key, value in (extra_env or {}).items()
        for item in ("--env", f"{key}={value}")
    ]
    return [
        config.docker_binary, "run", "--rm",
        "--name", name,
        "--pull", "never",  # a look-alike image on a registry must never be fetched
        "--network", "none",
        "--user", container_user(config),
        "--read-only",
        "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={config.tmpfs_size}",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--cpus", config.cpus,
        "--memory", config.memory,
        "--memory-swap", config.memory,  # no swap beyond the memory cap
        "--pids-limit", str(config.pids_limit),
        "--mount", f"type=bind,source={scratch_dir},target={CONTAINER_WORKDIR}",
        "--workdir", CONTAINER_WORKDIR,
        "--env", "HOME=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        *extra_env_flags,
        # In-container wall-clock limit: holds even if the host process dies.
        "--entrypoint", "timeout",
        config.image,
        "--signal=KILL", str(timeout_seconds),
        *container_command,
    ]


def fresh_scratch_dir(path: Path) -> Path:
    """Create ``path`` empty (wiping a previous attempt's leftovers)."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def remove_if_empty(path: Path) -> None:
    """Drop an unused scratch dir (and its parent when that empties too)."""
    for candidate in (Path(path), Path(path).parent):
        try:
            candidate.rmdir()
        except OSError:
            return


def describe_sandbox_exit(config: Optional[RenderSandboxConfig], returncode: int) -> str:
    """Operator hint for exit codes that come from the container, not Manim."""
    if config is None or not config.is_docker:
        return ""
    if returncode == 125:
        return (
            f" [sandbox: docker could not start the container; is the image "
            f"{config.image!r} built locally?]"
        )
    if returncode == 137:
        return " [sandbox: render killed at the time or memory limit]"
    return ""


def run_manim(
    *,
    scene_path: Path,
    scene_class: str,
    flags: Sequence[str],
    media_dir: Path,
    scratch_dir: Path,
    timeout_seconds: int,
    sandbox: Optional[RenderSandboxConfig] = None,
    native_command: Sequence[str] = ("manim",),
    audit_path: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run Manim on a generated scene file under the configured isolation.

    ``scratch_dir`` must already exist; it becomes the subprocess ``cwd``.
    In docker mode ``scene_path`` and ``media_dir`` must live inside it,
    because it is the only host directory the container can see.
    Raises ``FileNotFoundError`` (no manim/docker binary) and
    ``subprocess.TimeoutExpired`` like ``subprocess.run``.
    """
    sandbox = sandbox or RenderSandboxConfig()
    scratch_dir = Path(scratch_dir).resolve()

    if not sandbox.is_docker:
        command = [
            *native_command, *flags,
            "--media_dir", str(Path(media_dir).resolve()),
            str(Path(scene_path).resolve()), scene_class,
        ]
        return subprocess.run(
            command,
            cwd=str(scratch_dir),
            env=native_render_env(audit_path=audit_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    inner = [
        "python", "-m", "manim", *flags,
        "--media_dir", _container_path(media_dir, scratch_dir),
        _container_path(scene_path, scratch_dir), scene_class,
    ]
    name = f"pdf2manim-render-{uuid.uuid4().hex[:12]}"
    argv = build_docker_argv(
        config=sandbox,
        scratch_dir=scratch_dir,
        container_command=inner,
        timeout_seconds=timeout_seconds,
        name=name,
    )
    _grant_container_write(sandbox, scratch_dir)
    try:
        return subprocess.run(
            argv,
            cwd=str(scratch_dir),
            env=scrub_secret_env(),  # the docker client itself needs no secrets either
            capture_output=True,
            text=True,
            timeout=timeout_seconds + DOCKER_GRACE_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # Killing the docker client does not stop the container.
        _force_remove(sandbox, name)
        raise


def host_timeout(sandbox: Optional[RenderSandboxConfig], timeout_seconds: int) -> int:
    """Host-side wait for a render whose own limit is ``timeout_seconds``."""
    if sandbox is not None and sandbox.is_docker:
        return timeout_seconds + DOCKER_GRACE_SECONDS
    return timeout_seconds


def kill_process_tree(process: subprocess.Popen) -> None:
    """Kill ``process`` and its descendants. Never raises.

    Manim spawns LaTeX, dvisvgm and ffmpeg children; killing only the Python
    parent leaves them running and holding the scratch dir and the pipes.
    """
    try:
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            else:
                group = os.getpgid(pid)
                # Only a group the child leads; never the caller's own group.
                if group == pid and group != os.getpgid(0):
                    os.killpg(group, signal.SIGKILL)
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass


@dataclass(eq=False)  # identity hash: handles are tracked in sets
class ManimProcess:
    """A running render that can be waited on and cancelled."""

    process: subprocess.Popen
    sandbox: RenderSandboxConfig
    container_name: Optional[str] = None

    @property
    def returncode(self) -> Optional[int]:
        return self.process.returncode

    def communicate(self, timeout: Optional[float] = None) -> tuple[str, str]:
        """Wait for the render and return ``(stdout, stderr)``.

        On timeout the render is killed and ``subprocess.TimeoutExpired`` is
        re-raised carrying whatever output could still be collected.
        """
        try:
            stdout, stderr = self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self.kill()
            stdout, stderr = exc.stdout, exc.stderr
            try:
                stdout, stderr = self.process.communicate(timeout=KILL_DRAIN_SECONDS)
            except Exception:
                pass
            raise subprocess.TimeoutExpired(
                exc.cmd, exc.timeout, output=_text(stdout), stderr=_text(stderr),
            ) from None
        return _text(stdout), _text(stderr)

    def kill(self) -> None:
        """Stop the render: process tree, then the container. Never raises."""
        kill_process_tree(self.process)
        if self.container_name:
            # Killing the docker client does not stop the container.
            _force_remove(self.sandbox, self.container_name)


def start_manim(
    *,
    scene_path: Path,
    scene_class: str,
    flags: Sequence[str],
    media_dir: Path,
    scratch_dir: Path,
    timeout_seconds: int,
    sandbox: Optional[RenderSandboxConfig] = None,
    native_command: Sequence[str] = ("manim",),
    extra_env: Optional[Mapping[str, str]] = None,
) -> ManimProcess:
    """Start Manim like ``run_manim`` but return a cancellable handle.

    ``extra_env`` is added to the child environment; in docker mode its path
    values must be container paths (see ``container_path``). The caller waits
    with ``handle.communicate(host_timeout(sandbox, timeout_seconds))``.
    Raises ``FileNotFoundError`` when the manim/docker binary is missing.
    """
    sandbox = sandbox or RenderSandboxConfig()
    scratch_dir = Path(scratch_dir).resolve()
    popen_options: Dict[str, object] = dict(
        cwd=str(scratch_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if os.name != "nt":
        # Own session, so the whole tree can be killed as one process group.
        popen_options["start_new_session"] = True

    if not sandbox.is_docker:
        command = [
            *native_command, *flags,
            "--media_dir", str(Path(media_dir).resolve()),
            str(Path(scene_path).resolve()), scene_class,
        ]
        env = native_render_env()
        env.update(extra_env or {})
        process = subprocess.Popen(command, env=env, **popen_options)
        return ManimProcess(process=process, sandbox=sandbox)

    inner = [
        "python", "-m", "manim", *flags,
        "--media_dir", container_path(media_dir, scratch_dir),
        container_path(scene_path, scratch_dir), scene_class,
    ]
    name = f"pdf2manim-render-{uuid.uuid4().hex[:12]}"
    argv = build_docker_argv(
        config=sandbox,
        scratch_dir=scratch_dir,
        container_command=inner,
        timeout_seconds=timeout_seconds,
        name=name,
        extra_env=extra_env,
    )
    _grant_container_write(sandbox, scratch_dir)
    # The docker client itself needs no secrets either.
    process = subprocess.Popen(argv, env=scrub_secret_env(), **popen_options)
    return ManimProcess(process=process, sandbox=sandbox, container_name=name)


def container_path(path: Path, scratch_dir: Path) -> str:
    """Where a host path under ``scratch_dir`` appears inside the container."""
    scratch_dir = Path(scratch_dir).resolve()
    try:
        relative = Path(path).resolve().relative_to(scratch_dir)
    except ValueError:
        raise ValueError(
            f"{path} is outside the sandbox scratch dir {scratch_dir}; the "
            "render container cannot see it."
        ) from None
    if relative == Path("."):
        return CONTAINER_WORKDIR
    return f"{CONTAINER_WORKDIR}/{relative.as_posix()}"


_container_path = container_path


def _text(value: object) -> str:
    # TimeoutExpired can carry bytes even for a text-mode process.
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _grant_container_write(config: RenderSandboxConfig, scratch_dir: Path) -> None:
    """Let a container user that differs from the host user write the mount."""
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return  # Docker Desktop maps bind-mount ownership on Windows
    if container_user(config) != f"{getuid()}:{os.getgid()}":
        os.chmod(scratch_dir, 0o777)


def _force_remove(config: RenderSandboxConfig, name: str) -> None:
    try:
        subprocess.run(
            [config.docker_binary, "rm", "-f", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pass
