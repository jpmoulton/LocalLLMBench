from dataclasses import replace

import pytest

from llmbench.coding.sandbox import BoundedProcessExecutor, DockerWorker, WorkerResult
from llmbench.config import RunMode
from llmbench.safety import OperationForbidden, SessionLock


IMAGE = "local/eval@sha256:" + "a" * 64


class FakeExecutor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or WorkerResult("completed", 0, b"{}")

    def run(self, argv, **limits):
        self.calls.append((argv, limits))
        return WorkerResult("completed", 0) if argv[1] in {"rm", "ps"} else self.result


def job_for(tmp_path, *, allowed=False, result=None):
    root = tmp_path / "runs"
    candidate = root / "one"
    candidate.mkdir(parents=True)
    executor = FakeExecutor(result)
    worker = DockerWorker(allowed_root=root, executor=executor, mode=RunMode.LIVE,
                          session_lock=SessionLock(allow_container_execution=allowed))
    return worker, executor, worker.prepare(candidate, image=IMAGE, command=("python", "/runner/check.py"))


def test_no_container_or_inference_permission_by_default(tmp_path):
    worker, executor, job = job_for(tmp_path)
    with pytest.raises(OperationForbidden):
        worker.run(job)
    with pytest.raises(OperationForbidden):
        worker.check_inference_authorized()
    assert executor.calls == []


def test_command_is_resource_bounded_readonly_no_network_or_gpu(tmp_path):
    worker, executor, job = job_for(tmp_path, allowed=True)
    for argument in ("--network=none", "--read-only", "--cap-drop=ALL", "--pull=never",
                     "--memory=512m", "--memory-swap=512m", "--pids-limit=64"):
        assert argument in job.argv
    assert all("docker.sock" not in arg and "--gpus" not in arg for arg in job.argv)
    result = worker.run(job)
    assert result.cleanup_confirmed
    assert executor.calls[-2][0] == ("docker", "rm", "-f", job.name)
    assert executor.calls[-1][0][1] == "ps"
    assert executor.calls[0][1]["max_output_bytes"] == 1_048_576
    with pytest.raises(ValueError, match="already ran"):
        worker.run(job)


def test_forged_job_cannot_disable_network_or_use_shell(tmp_path):
    worker, executor, job = job_for(tmp_path, allowed=True)
    forged = replace(job, argv=tuple("--network=host" if arg == "--network=none" else arg for arg in job.argv))
    with pytest.raises(ValueError, match="changed"):
        worker.run(forged)
    assert not executor.calls


def test_timeout_retained_and_cleanup_targets_owned_name(tmp_path):
    worker, executor, job = job_for(tmp_path, allowed=True, result=WorkerResult("timeout", None))
    result = worker.run(job)
    assert result.status == "timeout"
    assert result.cleanup_confirmed
    assert len(executor.calls) == 3


def test_mounts_and_executables_and_images_are_restricted(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    candidate = root / "candidate"
    candidate.mkdir()
    worker = DockerWorker(allowed_root=root, session_lock=SessionLock())
    for path in (root, tmp_path):
        with pytest.raises(ValueError, match="dedicated"):
            worker.prepare(path, image=IMAGE, command=("python", "main.py"))
    with pytest.raises(ValueError, match="digest"):
        worker.prepare(candidate, image="python:latest", command=("python", "main.py"))
    with pytest.raises(ValueError, match="shell"):
        worker.prepare(candidate, image=IMAGE, command=("bash", "-c", "anything"))


class LifecycleExecutor:
    """Detached container whose tmpfs result is readable only by exec while it is alive."""

    def __init__(self, *, publishes=True, exits_early=False):
        self.calls, self.alive, self.removed = [], False, False
        self.publishes, self.exits_early = publishes, exits_early

    def run(self, argv, **limits):
        self.calls.append((argv, limits))
        if argv[1] == "run":
            self.alive = not self.exits_early
            return WorkerResult("completed", 0, b"container-id")
        if argv[1] == "exec":
            if not self.alive:
                return WorkerResult("completed", 1, stderr=b"Error response from daemon: container is not running")
            if not self.publishes:
                return WorkerResult("completed", 1, stderr=b"cat: /tmp/llmbench-result.json: No such file or directory")
            return WorkerResult("completed", 0, b'{"protocol":1}')
        if argv[1] == "inspect":
            return WorkerResult("completed", 0, b"true\n" if self.alive else b"false\n")
        if argv[1] == "logs":
            return WorkerResult("completed", 0, b"candidate print", b"candidate warning")
        if argv[1] == "rm":
            self.alive, self.removed = False, True
            return WorkerResult("completed", 0)
        if argv[1] == "ps":
            return WorkerResult("completed", 0, b"" if self.removed else b"still-there")
        raise AssertionError(argv)


def collecting_job(tmp_path, executor):
    root = tmp_path / "runs"
    candidate = root / "one"
    candidate.mkdir(parents=True)
    worker = DockerWorker(allowed_root=root, executor=executor, mode=RunMode.LIVE,
                          session_lock=SessionLock(allow_container_execution=True))
    return worker, worker.prepare(candidate, image=IMAGE, command=("python3", "/workspace/.llmbench-driver.py"),
                                  collect_result=True)


def test_result_is_read_by_exact_exec_then_logs_then_owned_cleanup(tmp_path):
    executor = LifecycleExecutor()
    worker, job = collecting_job(tmp_path, executor)
    result = worker.run(job)
    assert result.status == "completed" and result.returncode == 0
    assert result.result_bytes == b'{"protocol":1}'  # raw bytes, no archive framing
    assert (result.stdout, result.stderr) == (b"candidate print", b"candidate warning")
    assert result.cleanup_confirmed and result.cleanup_error is None
    assert [argv for argv, _ in executor.calls[1:]] == [
        ("docker", "exec", job.name, "cat", "/tmp/llmbench-result.json"),
        ("docker", "logs", job.name),
        ("docker", "rm", "-f", job.name),
        ("docker", "ps", "--all", "--filter", f"name=^/{job.name}$", "--format", "{{.ID}}"),
    ]
    for argv, _ in executor.calls:
        BoundedProcessExecutor._validate(argv)  # every emitted call is inside the real vocabulary


def test_container_that_exited_without_a_result_is_result_missing(tmp_path):
    executor = LifecycleExecutor(exits_early=True)
    worker, job = collecting_job(tmp_path, executor)
    result = worker.run(job)
    assert result.status == "result-missing" and result.result_bytes is None
    assert result.cleanup_confirmed
    assert [argv[1] for argv, _ in executor.calls] == ["run", "exec", "inspect", "logs", "rm", "ps"]


@pytest.mark.parametrize("status, code", [("timeout", None), ("output-limit", -1), ("completed", 7)])
def test_failed_result_read_diagnostics_survive_successful_logs_and_cleanup(tmp_path, status, code):
    class ReadFailure(LifecycleExecutor):
        def run(self, argv, **limits):
            if argv[1] == "exec":
                self.calls.append((argv, limits))
                self.alive = False
                return WorkerResult(status, code, b"PARTIAL-READ", b"RESULT-READ-ERROR")
            return super().run(argv, **limits)
    executor = ReadFailure()
    worker, job = collecting_job(tmp_path, executor)
    result = worker.run(job)
    assert result.cleanup_confirmed and result.result_bytes is None
    assert (result.stdout, result.stderr) == (b"candidate print", b"candidate warning")
    assert result.result_read_status == status and result.result_read_returncode == code
    assert result.result_read_stdout == b"PARTIAL-READ" and result.result_read_stderr == b"RESULT-READ-ERROR"
    assert result.result_read_attempts == 1 and not result.result_read_truncated


def test_result_read_diagnostics_have_one_combined_byte_limit(tmp_path):
    class LargeRead(LifecycleExecutor):
        def run(self, argv, **limits):
            if argv[1] == "exec":
                return WorkerResult("output-limit", -1, b"x" * 1048575, b"bad")
            return super().run(argv, **limits)
    worker, job = collecting_job(tmp_path, LargeRead())
    result = worker.run(job)
    assert len(result.result_read_stdout) + len(result.result_read_stderr) == job.limits.max_output_bytes
    assert result.result_read_stderr == b"b" and result.result_read_truncated
