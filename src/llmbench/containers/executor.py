"""Bounded Docker/Compose client restricted to the exact argv shapes this harness owns."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..coding.sandbox import BoundedProcessExecutor
from ..config import RunMode
from ..safety import SessionLock

PROJECT = r"llmbench-[0-9a-f]{12}"
RESOURCE_ID = r"[0-9a-f]{12,64}"
IMAGE = r"(?:[a-z0-9][a-z0-9._/-]*@)?sha256:[0-9a-f]{64}"
PULLABLE = r"[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}"
# Absolute host paths for preparation-only build shapes: no control or shell metacharacters.
SAFE_PATH = r"(?:[A-Za-z]:[\\/]|/)[^\x00-\x1f;|&<>\"`$]+"
# Space separated: the argument is then quoted on Windows and holds no shell metacharacters.
STATE_FORMAT = "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}"
SCRUBBED_ENVIRONMENT = ("DOCKER_HOST", "DOCKER_CONTEXT")
_UNSAFE_PATH = re.compile(r"[\x00-\x1f;|&<>\"`$]")
BUILD_SHAPE = ("build", "--platform=linux/amd64", "--pull=false", "--no-cache", "--iidfile", SAFE_PATH,
               "--file", SAFE_PATH, SAFE_PATH)
SELF_CHECK_SHAPE = ("run", "--rm", "--pull=never", "--network=none", "--read-only", "--user=10001:10001", IMAGE,
                    "--self-check")


def compose_prefix(project: str, compose_path: str | Path) -> tuple[str, ...]:
    path = Path(compose_path)
    return ("docker", "compose", "--ansi", "never", "--progress", "plain", "-p", project,
            "--project-directory", str(path.parent), "-f", str(path))


def project_filter(project: str) -> str:
    return f"label=com.docker.compose.project={project}"


def scrubbed_environment(environment=None) -> dict[str, str]:
    source = os.environ if environment is None else environment
    return {key: value for key, value in source.items()
            if not key.upper().startswith("COMPOSE_") and key.upper() not in SCRUBBED_ENVIRONMENT}


class ComposeExecutor(BoundedProcessExecutor):
    """Inherits the bounded pipe/timeout runner; replaces only the vocabulary, executable and environment."""

    def __init__(self, *, session_lock: SessionLock | None = None, mode: RunMode = RunMode.OFFLINE,
                 popen_factory=None, clock=time.monotonic, allow_preparation: bool = False) -> None:
        super().__init__(session_lock=session_lock, mode=mode, popen_factory=self._spawn, clock=clock)
        self._process_factory = popen_factory or subprocess.Popen
        self.allow_preparation = allow_preparation

    @property
    def synthetic(self) -> bool:
        return self._process_factory is not subprocess.Popen

    def _spawn(self, command, **kwargs):
        executable = command[0]
        if self._process_factory is subprocess.Popen:
            executable = shutil.which("docker")
            if not executable:
                raise OSError("Docker CLI was not found")
        return self._process_factory((executable, *command[1:]), env=scrubbed_environment(), **kwargs)

    def _validate(self, argv: tuple[str, ...]) -> None:
        if (type(argv) is not tuple or len(argv) < 2 or argv[0] != "docker"
                or any(type(item) is not str or not item or "\x00" in item for item in argv)):
            raise ValueError("Only direct Docker argv is accepted")
        if argv[1] == "compose":
            if len(argv) < 13 or argv[:7] != compose_prefix("x", "x")[:7] or not re.fullmatch(PROJECT, argv[7]):
                raise ValueError("Compose call lacks the owned project prefix")
            directory, compose_file, tail = argv[9], argv[11], argv[12:]
            if (argv[8] != "--project-directory" or argv[10] != "-f" or _UNSAFE_PATH.search(directory + compose_file)
                    or not Path(directory).is_absolute() or Path(directory).name != "plan"
                    or Path(compose_file) != Path(directory) / "compose.json"):
                raise ValueError("Compose project directory must be an absolute run plan directory")
            if (tail in {("config", "--format", "json"), ("port", "inference", "8080"),
                         ("ps", "--all", "--format", "json")}
                    or (len(tail) == 6 and tail[:5] == ("up", "--detach", "--no-build", "--pull", "never")
                        and tail[5] in {"inference", "evaluator"})
                    or (len(tail) == 4 and tail[:3] == ("down", "--volumes", "--timeout")
                        and re.fullmatch(r"[1-9]\d{0,2}", tail[3]))):
                return
            raise ValueError("Compose subcommand is outside the bounded vocabulary")
        owned = re.escape(project_filter("")) + PROJECT
        shapes = (("version", "--format", "{{json .}}"), ("image", "inspect", "--format", "{{json .}}", IMAGE),
                  ("inspect", "--format", STATE_FORMAT, RESOURCE_ID), ("logs", RESOURCE_ID),
                  ("ps", "--all", "--filter", owned, "--format", "{{.ID}}"),
                  ("network", "ls", "--filter", owned, "--format", "{{.ID}}"),
                  ("rm", "-f", RESOURCE_ID), ("network", "rm", RESOURCE_ID))
        if self.allow_preparation:
            shapes += (("pull", PULLABLE),
                       ("run", "--rm", "--pull=never", "--network=none", "--entrypoint", "/app/llama-server",
                        IMAGE, "--help|--version"), BUILD_SHAPE, SELF_CHECK_SHAPE)
        patterns = {IMAGE, PULLABLE, RESOURCE_ID, owned, "--help|--version", SAFE_PATH}
        for shape in shapes:
            if len(shape) == len(argv) - 1 and all(
                    re.fullmatch(expected, actual) if expected in patterns else expected == actual
                    for expected, actual in zip(shape, argv[1:])):
                return
        raise ValueError("Docker subcommand is outside the bounded container vocabulary")
