"""Resource-bounded, no-network Docker job contract with injected execution.

The implementation prepares exact argv and ownership cleanup; no daemon is probed
implicitly. Executors must bound timeout and combined output bytes. No host-code
execution fallback is provided.
"""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import Field

from ..config import RunMode, StrictModel
from ..safety import SessionLock


# Published atomically (write .pending, then rename) by the trusted drivers.
RESULT_PATH = "/tmp/llmbench-result.json"


class SandboxLimits(StrictModel):
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    memory_mib: int = Field(default=512, ge=64, le=4096)
    cpus: float = Field(default=1.0, gt=0, le=4)
    pids: int = Field(default=64, ge=1, le=256)
    max_output_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216)
    temporary_mib: int = Field(default=128, ge=1, le=512)
    # RLIMIT_NOFILE. 128 suits a single-function fixture; a public exercise can legitimately need more: the
    # reference solution to Exercism's parallel-letter-frequency starts 50 worker threads and dies with EMFILE
    # at 128, so a harness that kept 128 would fail a correct answer.
    open_files: int = Field(default=128, ge=32, le=4096)


@dataclass(frozen=True)
class WorkerResult:
    status: str
    returncode: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    cleanup_confirmed: bool = False
    result_bytes: bytes | None = None
    cleanup_error: str | None = None
    result_read_status: str | None = None
    result_read_returncode: int | None = None
    result_read_stdout: bytes = b""
    result_read_stderr: bytes = b""
    result_read_attempts: int = 0
    result_read_truncated: bool = False


class BoundedExecutor(Protocol):
    def run(self, argv: tuple[str, ...], *, timeout_seconds: int,
            max_output_bytes: int) -> WorkerResult:
        """Use shell=False, no stdin and bounded output. Kill the client on timeout/overflow."""
        ...


@dataclass(frozen=True)
class SandboxJob:
    name: str
    argv: tuple[str, ...]
    limits: SandboxLimits
    collect_result: bool = False


class BoundedProcessExecutor:
    """Run only this module's Docker client operations with bounded pipe readers.

    It is inert until run(), which checks the central capability before Popen.
    Timeout/output overflow kills the owned client. DockerWorker independently
    removes and verifies absence of the exact container, including after timeout.
    """

    def __init__(self, *, session_lock: SessionLock | None = None, mode: RunMode = RunMode.OFFLINE,
                 popen_factory=None, clock=time.monotonic) -> None:
        self.session_lock = session_lock or SessionLock()
        self.mode = mode
        self.popen_factory = popen_factory or subprocess.Popen
        self.clock = clock

    @staticmethod
    def _validate(argv: tuple[str, ...]) -> None:
        if (type(argv) is not tuple or not argv or argv[0] != "docker"
                or any(type(item) is not str or not item or "\x00" in item for item in argv)):
            raise ValueError("Only direct Docker argv is accepted")
        name_pattern = r"llmbench-[0-9a-f]{32}"
        if len(argv) > 3 and argv[1] == "run":
            if argv[3] == "--detach":
                argv = (*argv[:3], *argv[4:])
            if argv[10:13] != ("--log-driver=local", "--log-opt=max-size=1m", "--log-opt=max-file=2"):
                raise ValueError("Docker job must have bounded daemon logging")
            argv = (*argv[:10], *argv[13:])
        if len(argv) >= 22 and argv[1:3] == ("run", "--pull=never"):
            if argv[3] != "--name" or not re.fullmatch(name_pattern, argv[4]):
                raise ValueError("Docker run requires an owned container name")
            if (argv[5:10] != ("--network=none", "--read-only", "--cap-drop=ALL",
                               "--security-opt=no-new-privileges", "--user=10001:10001")
                    or re.fullmatch(r"--ulimit=nofile=(\d+):\1", argv[14]) is None or argv[16] != "--mount"
                    or argv[18:20] != ("--workdir=/workspace", "--entrypoint")
                    or argv[20] not in {"python", "python3", "node", "tsc", "evalplus.evaluate"}
                    or not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", argv[21])
                    or not re.fullmatch(r"type=bind,source=[^,\r\n]+,target=/workspace,readonly", argv[17])):
                raise ValueError("Docker run lacks the mandatory isolation profile")
            resources = []
            for value, expression in zip(argv[10:14],
                                         (r"--memory=(\d+)m", r"--memory-swap=(\d+)m",
                                          r"--cpus=(\d+(?:\.\d+)?)", r"--pids-limit=(\d+)")):
                match = re.fullmatch(expression, value)
                if match is None:
                    raise ValueError("Invalid Docker resource option")
                resources.append(float(match[1]))
            temporary = re.fullmatch(r"--tmpfs=/tmp:rw,noexec,nosuid,size=(\d+)m,mode=1777", argv[15])
            open_files = int(re.fullmatch(r"--ulimit=nofile=(\d+):\1", argv[14])[1])
            if (not 64 <= resources[0] <= 4096 or resources[1] != resources[0]
                    or not 0 < resources[2] <= 4 or not 1 <= resources[3] <= 256
                    or not 32 <= open_files <= 4096
                    or temporary is None or not 1 <= int(temporary[1]) <= 512):
                raise ValueError("Docker resources exceed worker limits")
        elif len(argv) == 4 and argv[1:3] == ("rm", "-f") and re.fullmatch(name_pattern, argv[3]):
            return
        elif argv[1:2] == ("exec",):
            # The only exec is the fixed read of the trusted result file from an
            # owned container: no options, no shell, no other program or path.
            if (len(argv) != 5 or not re.fullmatch(name_pattern, argv[2])
                    or argv[3:] != ("cat", RESULT_PATH)):
                raise ValueError("Docker exec is limited to the exact owned result read")
            return
        elif (len(argv) == 7 and argv[1:4] == ("ps", "--all", "--filter")
              and re.fullmatch(r"name=\^/" + name_pattern + r"\$", argv[4])
              and argv[5:] == ("--format", "{{.ID}}")):
            return
        elif len(argv) == 3 and argv[1] == "logs" and re.fullmatch(name_pattern, argv[2]):
            return
        elif (len(argv) == 5 and argv[1:4] == ("inspect", "--format", "{{.State.Running}}")
              and re.fullmatch(name_pattern, argv[4])):
            return
        else:
            raise ValueError("Docker subcommand is outside the bounded worker vocabulary")

    def run(self, argv: tuple[str, ...], *, timeout_seconds: int,
            max_output_bytes: int) -> WorkerResult:
        self.session_lock.check("container", self.mode)
        self._validate(argv)
        if type(timeout_seconds) is not int or timeout_seconds < 1 or type(max_output_bytes) is not int or max_output_bytes < 1:
            raise ValueError("Positive integer timeout/output budgets are required")
        executable = argv[0]
        if self.popen_factory is subprocess.Popen:
            executable = shutil.which("docker")
            if not executable:
                return WorkerResult("environment-error", None, stderr=b"Docker CLI was not found")
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      shell=False, close_fds=True, bufsize=0)
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = self.popen_factory((executable, *argv[1:]), **kwargs)
        except OSError as exc:
            return WorkerResult("environment-error", None, stderr=str(exc).encode("utf-8")[:max_output_bytes])
        chunks = [bytearray(), bytearray()]
        mutex, exceeded, read_error = threading.Lock(), threading.Event(), threading.Event()

        def drain(pipe, index):
            try:
                while True:
                    block = pipe.read(65536)
                    if not block:
                        return
                    with mutex:
                        available = max_output_bytes - len(chunks[0]) - len(chunks[1])
                        chunks[index].extend(block[:max(0, available)])
                        if len(block) > available:
                            exceeded.set()
            except (OSError, ValueError):
                read_error.set()

        readers = [threading.Thread(target=drain, args=(pipe, index), daemon=True)
                   for index, pipe in enumerate((process.stdout, process.stderr))]
        for reader in readers:
            reader.start()
        deadline = self.clock() + timeout_seconds
        status = "completed"
        try:
            while process.poll() is None:
                if exceeded.is_set():
                    status = "output-limit"
                    break
                if self.clock() >= deadline:
                    status = "timeout"
                    break
                exceeded.wait(.01)
            if status != "completed":
                process.kill()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = None
                status = "client-cleanup-failed"
        except BaseException:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise
        finally:
            for reader in readers:
                reader.join(timeout=1)
            for pipe in (process.stdout, process.stderr):
                pipe.close()
        if exceeded.is_set():
            status = "output-limit"
        elif read_error.is_set() or any(reader.is_alive() for reader in readers):
            status = "pipe-error"
        return WorkerResult(status, returncode, bytes(chunks[0]), bytes(chunks[1]))


class DockerWorker:
    def __init__(self, *, allowed_root: str | Path, session_lock: SessionLock,
                 mode: RunMode = RunMode.OFFLINE, executor: BoundedExecutor | None = None) -> None:
        self.allowed_root = Path(allowed_root).resolve()
        self.session_lock = session_lock
        self.mode = mode
        self.executor = executor or BoundedProcessExecutor(session_lock=session_lock, mode=mode)
        self._prepared: dict[str, SandboxJob] = {}

    @property
    def synthetic(self) -> bool:
        return (type(self.executor) is not BoundedProcessExecutor
                or self.executor.popen_factory is not subprocess.Popen)

    def prepare(self, candidate: str | Path, *, image: str, command: tuple[str, ...],
                limits: SandboxLimits | None = None, collect_result: bool = False) -> SandboxJob:
        """Pure planning. Candidate mount is read-only; only bounded /tmp is writable."""
        resolved = Path(candidate).resolve(strict=True)
        if resolved == self.allowed_root or not resolved.is_relative_to(self.allowed_root) or not resolved.is_dir():
            raise ValueError("Mount must be a dedicated candidate directory strictly inside allowed_root")
        if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
               for path in resolved.rglob("*")):
            raise ValueError("Candidate tree cannot contain symlinks")
        # Docker --mount uses commas as delimiters, even when argv does not use a shell.
        if any(char in str(resolved) for char in (",", "\n", "\r")):
            raise ValueError("Candidate path cannot contain Docker mount delimiters")
        if not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", image):
            raise ValueError("A pre-staged image pinned by sha256 digest is required")
        if (not command or command[0] not in {"python", "python3", "node", "tsc", "evalplus.evaluate"}
                or any(not isinstance(item, str) or not item or "\x00" in item for item in command)):
            raise ValueError("Use an approved direct executable and a nonempty argv; shell commands are forbidden")
        resources = limits or SandboxLimits()
        name = "llmbench-" + uuid.uuid4().hex
        if type(collect_result) is not bool:
            raise ValueError("collect_result must be boolean")
        argv = ("docker", "run", "--pull=never", *(("--detach",) if collect_result else ()), "--name", name,
                "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--user=10001:10001", "--log-driver=local", "--log-opt=max-size=1m", "--log-opt=max-file=2",
                f"--memory={resources.memory_mib}m",
                f"--memory-swap={resources.memory_mib}m", f"--cpus={resources.cpus}",
                f"--pids-limit={resources.pids}", f"--ulimit=nofile={resources.open_files}:{resources.open_files}",
                f"--tmpfs=/tmp:rw,noexec,nosuid,size={resources.temporary_mib}m,mode=1777",
                "--mount", f"type=bind,source={resolved},target=/workspace,readonly",
                "--workdir=/workspace", "--entrypoint", command[0], image, *command[1:])
        job = SandboxJob(name, argv, resources, collect_result)
        self._prepared[name] = job
        return job

    def run(self, job: SandboxJob) -> WorkerResult:
        self.session_lock.check("container", self.mode)
        # Refuse a caller-created job that smuggles another executable or target.
        if self._prepared.pop(job.name, None) != job:
            raise ValueError("Job was not prepared by this worker, was changed, or already ran")
        # Recheck mount contents immediately before launch, not only at planning time.
        mount = job.argv[job.argv.index("--mount") + 1]
        candidate = Path(mount.removeprefix("type=bind,source=").removesuffix(",target=/workspace,readonly"))
        if (not candidate.resolve(strict=True).is_relative_to(self.allowed_root)
                or any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                       for path in candidate.rglob("*"))):
            raise ValueError("Candidate mount changed after planning")
        result_bytes = None
        # Keep the last actual result-read outcome apart from daemon logs. Repeated
        # unpublished-file polls replace this bounded snapshot; attempts still count.
        read_status, read_code, read_stdout, read_stderr = None, None, b"", b""
        read_attempts, read_truncated = 0, False
        cleanup_confirmed, cleanup_error = False, None
        try:
            deadline = time.monotonic() + job.limits.timeout_seconds
            result = self.executor.run(job.argv, timeout_seconds=min(10, job.limits.timeout_seconds) if job.collect_result
                                       else job.limits.timeout_seconds,
                                       max_output_bytes=job.limits.max_output_bytes)
            if len(result.stdout) + len(result.stderr) > job.limits.max_output_bytes:
                result = WorkerResult("output-limit", result.returncode)
            if job.collect_result and result.status == "completed" and result.returncode == 0:
                # The trusted driver atomically publishes the result and remains
                # alive for a bounded interval. tmpfs vanishes when PID 1 exits.
                # `docker cp` cannot read tmpfs mounts, so the file is read with one
                # exact `docker exec <owned name> cat <fixed path>`. cat runs as the
                # container user (10001) inside the same isolation profile (no network,
                # read-only root, no capabilities, no new privileges) and is a second
                # process counted against --pids-limit: a candidate that exhausts pids
                # makes the read fail and the case ends as a bounded timeout. A path the
                # candidate replaced (FIFO, symlink, huge file) only yields the client
                # timeout, output-limit or an invalid observation for that candidate.
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        result = WorkerResult("timeout", None, result.stdout, result.stderr)
                        break
                    call_timeout = max(1, min(10, int(remaining)))
                    copied = self.executor.run(("docker", "exec", job.name, "cat", RESULT_PATH),
                                               timeout_seconds=call_timeout,
                                               max_output_bytes=job.limits.max_output_bytes + 4096)
                    read_attempts += 1
                    read_status, read_code = copied.status, copied.returncode
                    read_stdout = copied.stdout[:job.limits.max_output_bytes]
                    room = max(0, job.limits.max_output_bytes - len(read_stdout))
                    read_stderr = copied.stderr[:room]
                    read_truncated = len(copied.stdout) + len(copied.stderr) > job.limits.max_output_bytes
                    if copied.status == "completed" and copied.returncode == 0:
                        # Publication is an atomic rename, so a successful read is a
                        # complete file. stdout is the raw result; stderr is never data.
                        if 1 <= len(copied.stdout) <= job.limits.max_output_bytes:
                            result_bytes = copied.stdout
                        else:
                            result = WorkerResult("protocol-error", 0, result.stdout,
                                                  b"Container result is empty or exceeds the output budget")
                        break
                    if copied.status != "completed":
                        result = WorkerResult(copied.status, copied.returncode, result.stdout, copied.stderr)
                        break
                    running = self.executor.run(("docker", "inspect", "--format", "{{.State.Running}}", job.name),
                                                timeout_seconds=call_timeout, max_output_bytes=4096)
                    if running.status != "completed" or running.returncode != 0 or running.stdout.strip() != b"true":
                        detail = (result.stderr + b"\nResult read: " + copied.stderr)[:job.limits.max_output_bytes]
                        result = WorkerResult("result-missing", 1, result.stdout, detail)
                        break
                    time.sleep(min(.05, max(0., deadline - time.monotonic())))
                logs = self.executor.run(("docker", "logs", job.name), timeout_seconds=10,
                                         max_output_bytes=job.limits.max_output_bytes)
                if logs.status == "completed" and logs.returncode == 0:
                    result = WorkerResult(result.status, result.returncode, logs.stdout, logs.stderr)
                elif logs.status == "output-limit":
                    result = WorkerResult("output-limit", result.returncode, logs.stdout, logs.stderr)
        finally:
            # This exact random name is owned by the job. Never use docker prune/kill-all.
            try:
                self.session_lock.check("container", self.mode)
                self.executor.run(("docker", "rm", "-f", job.name), timeout_seconds=10, max_output_bytes=4096)
                # Removal may race auto-removal or return NotFound. Successful
                # exact absence query is confirmation; an unavailable daemon is not.
                absent = self.executor.run(("docker", "ps", "--all", "--filter", f"name=^/{job.name}$",
                                            "--format", "{{.ID}}"), timeout_seconds=10, max_output_bytes=4096)
                cleanup_confirmed = (absent.status == "completed" and absent.returncode == 0
                                     and not absent.stdout.strip())
                if not cleanup_confirmed:
                    cleanup_error = "Owned container absence could not be verified"
            except Exception as exc:
                # Retain a collected result and diagnostics when cleanup itself
                # fails. The caller persists them and aborts the entire campaign.
                cleanup_error = type(exc).__name__ + ": " + str(exc)
        return WorkerResult(result.status, result.returncode, result.stdout, result.stderr,
                            cleanup_confirmed, result_bytes, cleanup_error,
                            result_read_status=read_status, result_read_returncode=read_code,
                            result_read_stdout=read_stdout, result_read_stderr=read_stderr,
                            result_read_attempts=read_attempts, result_read_truncated=read_truncated)

    def check_inference_authorized(self) -> None:
        """An outer coding agent must pass this independently before requesting a model."""
        self.session_lock.check("inference", self.mode)
