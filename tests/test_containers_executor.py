import io
import subprocess

import pytest

from llmbench.config import RunMode
from llmbench.containers.executor import (STATE_FORMAT, ComposeExecutor, compose_prefix, project_filter,
                                          scrubbed_environment)
from llmbench.safety import OperationForbidden, SessionLock

PROJECT = "llmbench-0123456789ab"
CID = "0123456789abcdef" * 4
IMAGE = "ghcr.io/ggml-org/llama.cpp@sha256:" + "d" * 64


class FakeProcess:
    def __init__(self, stdout=b"ok", returncode=0):
        self.stdout, self.stderr, self.returncode = io.BytesIO(stdout), io.BytesIO(b""), returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class Spawner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return FakeProcess()


def executor(*, allowed=True, spawner=None, **options):
    return ComposeExecutor(session_lock=SessionLock(allow_container_execution=allowed), mode=RunMode.LIVE,
                           popen_factory=spawner or Spawner(), **options)


def prefix(tmp_path):
    return compose_prefix(PROJECT, tmp_path / "run" / "plan" / "compose.json")


def allowed_shapes(tmp_path):
    base = prefix(tmp_path)
    owned = project_filter(PROJECT)
    return [("docker", "version", "--format", "{{json .}}"),
            ("docker", "image", "inspect", "--format", "{{json .}}", IMAGE),
            ("docker", "image", "inspect", "--format", "{{json .}}", "sha256:" + "d" * 64),
            (*base, "config", "--format", "json"),
            (*base, "up", "--detach", "--no-build", "--pull", "never", "inference"),
            (*base, "up", "--detach", "--no-build", "--pull", "never", "evaluator"),
            (*base, "port", "inference", "8080"), (*base, "ps", "--all", "--format", "json"),
            (*base, "down", "--volumes", "--timeout", "15"),
            ("docker", "inspect", "--format", STATE_FORMAT, CID), ("docker", "logs", CID[:12]),
            ("docker", "ps", "--all", "--filter", owned, "--format", "{{.ID}}"),
            ("docker", "network", "ls", "--filter", owned, "--format", "{{.ID}}"),
            ("docker", "rm", "-f", CID), ("docker", "network", "rm", CID[:12])]


def test_every_owned_shape_runs_with_a_scrubbed_environment(tmp_path, monkeypatch):
    for name in ("COMPOSE_FILE", "compose_project_name", "DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.setenv(name, "hostile")
    monkeypatch.setenv("LLMBENCH_KEEP", "1")
    spawner = Spawner()
    runner = executor(spawner=spawner)
    for argv in allowed_shapes(tmp_path):
        result = runner.run(argv, timeout_seconds=5, max_output_bytes=1024)
        assert (result.status, result.returncode, result.stdout) == ("completed", 0, b"ok")
    assert len(spawner.calls) == len(allowed_shapes(tmp_path)) and runner.synthetic
    for command, kwargs in spawner.calls:
        environment = kwargs["env"]
        assert not any(key.upper().startswith("COMPOSE_") or key.upper() in {"DOCKER_HOST", "DOCKER_CONTEXT"}
                       for key in environment)
        assert environment["LLMBENCH_KEEP"] == "1"
        assert kwargs["shell"] is False and kwargs["stdin"] is subprocess.DEVNULL and command[0] == "docker"
    assert scrubbed_environment({"Compose_X": "1", "PATH": "p"}) == {"PATH": "p"}


def test_anything_outside_the_vocabulary_is_rejected_before_spawning(tmp_path):
    base = prefix(tmp_path)
    other = compose_prefix(PROJECT, tmp_path / "elsewhere" / "compose.json")
    forbidden = [
        ("docker", "run", "--privileged", IMAGE), ("docker", "system", "prune", "-f"), ("docker", "kill", CID),
        ("docker", "rm", "-f", "some-name"), ("docker", "rm", "-f", CID, CID), ("docker", "rm", "-f", "-a"),
        ("docker", "network", "prune"), ("docker", "volume", "rm", CID), ("docker", "exec", CID, "sh"),
        ("docker", "pull", IMAGE), ("docker", "image", "inspect", "--format", "{{json .}}", "llama.cpp:latest"),
        ("docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint", "/app/llama-server", IMAGE,
         "--help"),
        ("docker", "ps", "--all", "--filter", "label=com.docker.compose.project=other", "--format", "{{.ID}}"),
        ("docker", "ps", "--all", "--filter", project_filter(PROJECT) + "; rm -rf /", "--format", "{{.ID}}"),
        ("docker", "logs", CID + "; whoami"), ("docker", "logs", "$(whoami)"), ("docker",), ("podman", "ps"),
        (*base, "exec", "inference", "sh"), (*base, "run", "inference"), (*base, "up"),
        (*base, "up", "--detach", "--no-build", "--pull", "always", "inference"),
        (*base, "up", "--detach", "--no-build", "--pull", "never", "other"),
        (*base, "down", "--volumes", "--timeout", "15", "--rmi", "all"), (*base, "down", "--volumes"),
        (*base, "-f", "other.yml", "config", "--format", "json"), (*base, "-v"), (*base, "cp", "a", "b"),
        (*other, "config", "--format", "json"),
        (*compose_prefix("unowned", tmp_path / "plan" / "compose.json"), "config", "--format", "json"),
        (*compose_prefix(PROJECT, tmp_path / "a&b" / "plan" / "compose.json"), "config", "--format", "json"),
        ("docker", "compose", "-p", PROJECT, "down"), list(allowed_shapes(tmp_path)[0]),
        ("docker", "logs", ""), ("docker", "logs", CID + "\x00"),
    ]
    spawner = Spawner()
    runner = executor(spawner=spawner)
    for argv in forbidden:
        with pytest.raises(ValueError):
            runner.run(argv, timeout_seconds=5, max_output_bytes=1024)
    assert spawner.calls == []


def test_relative_project_directory_is_rejected():
    runner = executor()
    with pytest.raises(ValueError, match="absolute"):
        runner.run((*compose_prefix(PROJECT, "plan/compose.json"), "config", "--format", "json"),
                   timeout_seconds=5, max_output_bytes=1024)


def test_preparation_vocabulary_is_opt_in(tmp_path):
    spawner = Spawner()
    runner = executor(spawner=spawner, allow_preparation=True)
    for argv in (("docker", "pull", IMAGE),
                 ("docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint", "/app/llama-server",
                  IMAGE, "--version")):
        assert runner.run(argv, timeout_seconds=5, max_output_bytes=1024).status == "completed"
    for argv in (("docker", "pull", "sha256:" + "d" * 64), ("docker", "pull", "llama.cpp:latest"),
                 ("docker", "run", "--rm", "--pull=never", "--network=host", "--entrypoint", "/app/llama-server",
                  IMAGE, "--help"),
                 ("docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint", "/app/llama-server",
                  IMAGE, "--model")):
        with pytest.raises(ValueError):
            runner.run(argv, timeout_seconds=5, max_output_bytes=1024)


def test_container_permission_and_live_mode_are_required(tmp_path):
    spawner = Spawner()
    argv = allowed_shapes(tmp_path)[0]
    with pytest.raises(OperationForbidden):
        executor(allowed=False, spawner=spawner).run(argv, timeout_seconds=5, max_output_bytes=1024)
    offline = ComposeExecutor(session_lock=SessionLock(allow_container_execution=True), popen_factory=spawner)
    with pytest.raises(OperationForbidden):
        offline.run(argv, timeout_seconds=5, max_output_bytes=1024)
    with pytest.raises(OperationForbidden):
        ComposeExecutor(popen_factory=spawner).run(argv, timeout_seconds=5, max_output_bytes=1024)
    assert spawner.calls == []


def test_output_budget_and_missing_cli_are_reported(tmp_path, monkeypatch):
    class Loud(Spawner):
        def __call__(self, command, **kwargs):
            return FakeProcess(stdout=b"x" * 5000)
    result = executor(spawner=Loud()).run(allowed_shapes(tmp_path)[0], timeout_seconds=5, max_output_bytes=1000)
    assert result.status == "output-limit" and len(result.stdout) <= 1000
    real = ComposeExecutor(session_lock=SessionLock(allow_container_execution=True), mode=RunMode.LIVE)
    monkeypatch.setattr("llmbench.containers.executor.shutil.which", lambda name: None)
    missing = real.run(allowed_shapes(tmp_path)[0], timeout_seconds=5, max_output_bytes=1000)
    assert missing.status == "environment-error" and not real.synthetic


def build_shape(tmp_path, *, iidfile=None, dockerfile=None, context=None):
    context = str(context or tmp_path / "ctx")
    return ("docker", "build", "--platform=linux/amd64", "--pull=false", "--no-cache", "--iidfile",
            str(iidfile or tmp_path / "evaluator-image.id"), "--file", str(dockerfile or tmp_path / "ctx" / "Dockerfile"),
            context)


SELF_CHECK = ("docker", "run", "--rm", "--pull=never", "--network=none", "--read-only", "--user=10001:10001",
              "sha256:" + "e" * 64, "--self-check")


def test_build_and_self_check_shapes_are_preparation_only(tmp_path):
    spawner = Spawner()
    plain = executor(spawner=spawner)
    for argv in (build_shape(tmp_path), SELF_CHECK):
        with pytest.raises(ValueError):
            plain.run(argv, timeout_seconds=5, max_output_bytes=1024)
    assert spawner.calls == []
    prepared = executor(spawner=spawner, allow_preparation=True)
    for argv in (build_shape(tmp_path), SELF_CHECK):
        assert prepared.run(argv, timeout_seconds=5, max_output_bytes=1024).status == "completed"
    assert [call[0][1] for call in spawner.calls] == ["build", "run"]
    for argv in (build_shape(tmp_path, context="ctx"), build_shape(tmp_path, iidfile="relative.id"),
                 build_shape(tmp_path, context=str(tmp_path / "ctx") + ";rm -rf /"),
                 build_shape(tmp_path, dockerfile="C:\\x\\Dockerfile$"),
                 ("docker", "build", "--platform=linux/arm64", *build_shape(tmp_path)[3:]),
                 ("docker", "build", "--platform=linux/amd64", "--pull=true", *build_shape(tmp_path)[4:]),
                 ("docker", "build", "--platform=linux/amd64", "--pull=false", "--iidfile", *build_shape(tmp_path)[6:]),
                 build_shape(tmp_path) + ("--build-arg", "X=1"), build_shape(tmp_path) + ("--tag", "evaluator:latest")):
        with pytest.raises(ValueError):
            prepared.run(argv, timeout_seconds=5, max_output_bytes=1024)
    assert len(spawner.calls) == 2


def test_self_check_shape_requires_network_none_read_only_and_uid(tmp_path):
    spawner = Spawner()
    prepared = executor(spawner=spawner, allow_preparation=True)
    for argv in (("docker", "run", "--rm", "--pull=never", "--network=bridge", *SELF_CHECK[5:]),
                 ("docker", "run", "--rm", "--pull=never", "--network=none", "--user=10001:10001", *SELF_CHECK[7:]),
                 ("docker", "run", "--rm", "--pull=never", "--network=none", "--read-only", "--user=0:0",
                  *SELF_CHECK[7:]),
                 ("docker", "run", "--rm", "--pull=always", *SELF_CHECK[4:]),
                 (*SELF_CHECK[:7], "evaluator:latest", "--self-check"), (*SELF_CHECK[:8], "--config"),
                 (*SELF_CHECK[:8], "--self-check", "--grant-seconds", "1"), SELF_CHECK + ("-v", "/:/host"),
                 ("docker", "run", "--rm", "--pull=never", "--network=none", "--read-only", "--user=10001:10001",
                  "--privileged", SELF_CHECK[7], "--self-check")):
        with pytest.raises(ValueError):
            prepared.run(argv, timeout_seconds=5, max_output_bytes=1024)
    assert spawner.calls == []
    digest = ("docker", *SELF_CHECK[1:7], "ghcr.io/example/evaluator@sha256:" + "e" * 64, "--self-check")
    assert prepared.run(digest, timeout_seconds=5, max_output_bytes=1024).status == "completed"
    assert prepared.run(SELF_CHECK, timeout_seconds=5, max_output_bytes=1024).status == "completed"
    with pytest.raises(OperationForbidden):
        executor(allowed=False, spawner=spawner, allow_preparation=True).run(SELF_CHECK, timeout_seconds=5,
                                                                              max_output_bytes=1024)
