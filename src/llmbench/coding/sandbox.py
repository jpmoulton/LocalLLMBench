"""Resource-bounded, no-network Docker job contract with injected execution.

The implementation prepares exact argv and ownership cleanup; no daemon is probed
implicitly. Executors must bound timeout and combined output bytes. No host-code
execution fallback is provided. ``probe_docker_sandbox`` is the one explicit,
read-only question "could a worker run here now?", asked with two fixed argv.
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
from typing import Any, Protocol

from pydantic import Field

from ..config import RunMode, StrictModel
from ..safety import OperationForbidden, SessionLock


# Published atomically (write .pending, then rename) by the trusted drivers.
RESULT_PATH = "/tmp/llmbench-result.json"
# The sandbox probe's whole Docker vocabulary: the daemon's own platform, and one local image's identity and
# platform by its content-addressed id. Both are reads; neither pulls, creates or starts anything.
SERVER_PLATFORM_ARGV = ("docker", "version", "--format", "{{.Server.Os}}/{{.Server.Arch}}")
IMAGE_PLATFORM_ARGV = ("docker", "image", "inspect", "--format", "{{.Id}} {{.Os}}/{{.Architecture}}")
PROBE_TIMEOUT_SECONDS = 10
PROBE_OUTPUT_BYTES = 4096
_LOCAL_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_OS_ARCH = re.compile(r"[a-z0-9]+/[a-z0-9_]+")


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
    # False only when the Docker client itself was never spawned (CLI absent, or the spawn raised OSError): then
    # no daemon was asked anything, so no container can exist. Every result from a process that ran keeps True.
    launched: bool = True


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
        elif argv == SERVER_PLATFORM_ARGV:
            return
        elif len(argv) == 6 and argv[:5] == IMAGE_PLATFORM_ARGV and _LOCAL_IMAGE_ID.fullmatch(argv[5]):
            # Only a content-addressed local id: a name or tag could resolve to a different image tomorrow.
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
                return WorkerResult("environment-error", None, stderr=b"Docker CLI was not found", launched=False)
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      shell=False, close_fds=True, bufsize=0)
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = self.popen_factory((executable, *argv[1:]), **kwargs)
        except OSError as exc:
            return WorkerResult("environment-error", None, stderr=str(exc).encode("utf-8")[:max_output_bytes],
                                launched=False)
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
        never_launched = False
        try:
            deadline = time.monotonic() + job.limits.timeout_seconds
            result = self.executor.run(job.argv, timeout_seconds=min(10, job.limits.timeout_seconds) if job.collect_result
                                       else job.limits.timeout_seconds,
                                       max_output_bytes=job.limits.max_output_bytes)
            if result.launched is False and result.returncode is None:
                # The Docker client was never spawned, so no daemon ever received this `run`: a container of this
                # fresh random name cannot exist, and there is nothing to remove or to query (the same missing CLI
                # would make both fail, which today reads as "unverified" and aborts the whole campaign). This is
                # unavailable infrastructure, reported as such. A client that did start and then lost the daemon
                # keeps the full cleanup path below: that container's state is genuinely unknown.
                never_launched = True
                result = WorkerResult("sandbox-unavailable", None, b"", result.stderr[:job.limits.max_output_bytes],
                                      launched=False)
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
            if never_launched:
                cleanup_confirmed = True  # proven by construction above: nothing was ever created
            else:
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
                            result_read_attempts=read_attempts, result_read_truncated=read_truncated,
                            launched=not never_launched)

    def check_inference_authorized(self) -> None:
        """An outer coding agent must pass this independently before requesting a model."""
        self.session_lock.check("inference", self.mode)


def _detail(result: WorkerResult) -> str:
    """The Docker client's own words, bounded and on one line, for a blocked reason."""
    text = (result.stderr.strip() or result.stdout.strip()).decode("utf-8", "replace")
    return " ".join(text.split())[:300] or "no output"


def _pinned_worker_image(image_ref: Any) -> tuple[str | None, str | None]:
    """(local image id, pinned platform) of a worker ``ImageRef``, of a bare ``sha256:`` id, or ``(None, None)``."""
    if image_ref is None:
        return None, None
    if isinstance(image_ref, str):
        image_id, platform = image_ref, None
    else:
        if getattr(image_ref, "role", "worker") != "worker":
            raise ValueError("the sandbox probe checks the sandbox worker image, not an inference or evaluator image")
        image_id, platform = getattr(image_ref, "image_id", None), getattr(image_ref, "platform", None)
    if type(image_id) is not str or not _LOCAL_IMAGE_ID.fullmatch(image_id):
        raise ValueError("image_ref must be a worker ImageRef or a local sha256:<64 hex> image id")
    return image_id, platform


def inspect_local_image(executor: BoundedExecutor, image_id: str, *,
                        timeout_seconds: int = PROBE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """One exact ``docker image inspect`` of a LOCAL image by its content-addressed id; never pulls.

    Returns ``{"present", "image_id", "platform", "detail", "status", "returncode", "launched"}``. ``present`` is
    True only when the daemon answered for this very id with a readable ``os/arch``; the inspected platform is
    what is recorded, never the one a caller intended to build. The client's status/exit/launch are returned so a
    caller can tell "no such image" from "the daemon did not answer" from "no Docker client at all".
    """
    if type(image_id) is not str or not _LOCAL_IMAGE_ID.fullmatch(image_id):
        raise ValueError("a local sha256:<64 hex> image id is required")
    result = executor.run((*IMAGE_PLATFORM_ARGV, image_id), timeout_seconds=timeout_seconds,
                          max_output_bytes=PROBE_OUTPUT_BYTES)
    found = {"present": False, "image_id": None, "platform": None, "detail": None, "status": result.status,
             "returncode": result.returncode, "launched": result.launched}
    if result.status != "completed" or result.returncode != 0:
        return {**found, "detail": _detail(result)}
    fields = result.stdout.decode("utf-8", "replace").split()
    if len(fields) != 2 or not _LOCAL_IMAGE_ID.fullmatch(fields[0]) or not _OS_ARCH.fullmatch(fields[1]):
        return {**found, "detail": f"unreadable image inspection: {_detail(result)!r}"}
    if fields[0] != image_id:
        return {**found, "detail": f"the daemon answered for {fields[0]}, not {image_id}"}
    return {**found, "present": True, "image_id": fields[0], "platform": fields[1]}


def probe_docker_sandbox(*, session_lock: SessionLock, image_ref: Any = None,
                         executor: BoundedExecutor | None = None) -> dict[str, Any]:
    """Could a coding worker run here, now? ``{"status": "available" | "blocked", "reason", "blocked_by",
    "docker_cli", "server_os_arch", "image_id", "image_platform"}``.

    Generated code only ever runs inside the owned worker container, so where no worker can start the coding suites
    are recorded as blocked (never run on the host instead). The probe is read-only by construction: at most the two
    fixed reads ``SERVER_PLATFORM_ARGV`` and ``IMAGE_PLATFORM_ARGV + (id,)`` through the bounded executor (both
    allowlisted there and nowhere else); it pulls, builds, creates and starts nothing, and never starts a daemon
    (Colima or Docker Desktop stay as the user left them). Checks run in this order and the first failure is the
    ``blocked_by`` value, with the client's own words in ``reason``:

    1. ``policy``: the session lock allows container execution; checked before any process is spawned, so a policy
       that forbids containers never even reaches the Docker client.
    2. ``docker_cli``: ``docker`` resolves on PATH (for the real executor) and the client actually spawned.
    3. ``daemon``: ``docker version`` answered with the server's ``os/arch`` within the bound.
    4. ``worker_image``: a pinned worker image was given (None is blocked: the broker cannot run without one); the
       daemon holds exactly that id; its inspected platform is the pinned one; and it is the daemon's own
       ``os/arch``. A foreign-architecture worker is blocked rather than emulated: without binfmt emulation every
       case would exit "exec format error" and read as a failed solution, and with it the per-case limits, set for
       native execution, would time out correct answers. Either way a harness defect would score as a model 0.

    "available" therefore means all four were observed just now, not assumed. An environment that says no is a
    result, never an exception; only a malformed ``image_ref`` (a programming error) raises ``ValueError``.
    ``docker_cli`` records where PATH resolves ``docker`` on this host even when an executor is injected.
    """
    image_id, pinned_platform = _pinned_worker_image(image_ref)
    report: dict[str, Any] = {"status": "blocked", "reason": None, "blocked_by": None,
                              "docker_cli": shutil.which("docker"), "server_os_arch": None, "image_id": None,
                              "image_platform": None}

    def blocked(check: str, reason: str) -> dict[str, Any]:
        return {**report, "blocked_by": check, "reason": reason}

    try:
        session_lock.check("container", RunMode.LIVE)
    except OperationForbidden as exc:
        return blocked("policy", f"the session policy forbids container execution: {exc}")
    if executor is None:
        if report["docker_cli"] is None:
            return blocked("docker_cli", "the Docker CLI was not found on PATH")
        executor = BoundedProcessExecutor(session_lock=session_lock, mode=RunMode.LIVE)
    server = executor.run(SERVER_PLATFORM_ARGV, timeout_seconds=PROBE_TIMEOUT_SECONDS,
                          max_output_bytes=PROBE_OUTPUT_BYTES)
    if server.launched is False:
        return blocked("docker_cli", f"the Docker CLI could not be started: {_detail(server)}")
    if server.status != "completed" or server.returncode != 0:
        return blocked("daemon", f"the Docker daemon is unreachable ({server.status}, exit {server.returncode}): "
                                 f"{_detail(server)}")
    platform = server.stdout.decode("utf-8", "replace").strip()
    if not _OS_ARCH.fullmatch(platform):
        return blocked("daemon", f"the Docker daemon reported no server os/arch: {_detail(server)!r}")
    report["server_os_arch"] = platform
    if image_id is None:
        return blocked("worker_image", "no pinned sandbox worker image is configured")
    image = inspect_local_image(executor, image_id)
    if image["launched"] is False:
        return blocked("docker_cli", f"the Docker CLI could not be started: {image['detail']}")
    if image["status"] != "completed":
        return blocked("daemon", f"the Docker daemon did not answer the worker image inspection ({image['status']}): "
                                 f"{image['detail']}")
    if not image["present"]:
        return blocked("worker_image", f"the pinned worker image {image_id} is not present locally "
                                       f"(exit {image['returncode']}): {image['detail']}")
    report.update(image_id=image["image_id"], image_platform=image["platform"])
    if pinned_platform is not None and image["platform"] != pinned_platform:
        return blocked("worker_image", f"the worker image is {image['platform']}, not its pinned {pinned_platform}")
    if image["platform"] != platform:
        return blocked("worker_image", f"the {image['platform']} worker image cannot run natively on the {platform} "
                                       "Docker daemon; build the worker for the daemon's platform")
    return {**report, "status": "available"}
