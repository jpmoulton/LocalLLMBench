"""One bounded native llama.cpp candidate on an Apple Silicon host: the `metal-native` runtime.

The container runner (`runner.py`) owns the stage machine -- admit, hash, plan, start, ready, verify, broker,
quality, settings verify, cleanup, report -- and everything in it that is not Docker: bounded phases, the lease,
model hashing and re-verification, the evaluator, settings evidence, reports and the terminal result. This module
replaces only the primitives that touched Docker, so both runtimes produce the same evidence under the same rules:

* identity: instead of `docker image inspect`, the pinned executable and the llama.cpp/ggml libraries beside it are
  hashed and compared with `native_server`, and the executable's own `--version`/`--help` must still say what the
  pin says (`build_info`, `help_sha256`) before a flag is checked against it;
* admission: instead of foreign VRAM, unified memory -- the host's available memory against the model file plus a
  reserve, macOS memory pressure, and the GPU's own utilisation (a game or another server on the GPU is a conflict
  to report, never a process to stop);
* start: the server is a child process in its OWN session (so a terminal Ctrl-C reaches the harness, which stops
  it deliberately, rather than the server), bound to 127.0.0.1 on a port chosen here, with a scrubbed environment
  so no `LLAMA_ARG_*` variable can override the argv, and its combined output drained into a bounded log;
* state and logs: `Popen.poll()` and that log instead of `docker inspect`/`docker logs`;
* memory: phys_footprint of the server (what Activity Monitor calls Memory, including Metal allocations), the Metal
  buffers the server logged, host available memory, swap and pressure, sampled after load and by a watchdog while
  the evaluator runs. None of it is VRAM and none of it is reported as VRAM;
* cleanup: SIGTERM to the owned process group, SIGKILL after the grace period, then positive proof of absence --
  the child reaped, its process group empty, no process carrying this attempt's alias and port, and the port
  refusing connections. Anything short of that is `cleanup-uncertain`, exactly like a container that would not go.
  The server is stopped before anything else (the coding broker's Docker calls come after it), and a Ctrl-C that
  arrives during that bounded stop is held until the proof is recorded, then delivered: the server runs in its own
  session, so an interrupted stop is the one thing nothing else would ever finish.

Generated code is never executed here. Coding suites still go through the host broker's sandboxed Docker workers;
when that sandbox is unavailable they are recorded as blocked with the reason, and nothing is generated for them.
"""

from __future__ import annotations

import math
import os
import platform
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..backends.base import BackendError
from ..config import RunMode, canonical_json
from ..safety import OperationForbidden, SessionLock
from .capabilities import parse_server_help, require_supported
from .config import ContainerRunConfig, ContainerRunResult, NativeLimits, CleanupEvidence
from .plan import build_server_argv
from .preflight import PreflightError
from .readback import parse_startup_log
from .runner import ContainerRunner, _Attempt, _Stop

NATIVE_LOG = "logs/server.log"
LAUNCH_RECORD = "plan/native-launch.json"
LOOPBACK = "127.0.0.1"
PROBE_TIMEOUT_SECONDS = 20
LOG_CHUNK_BYTES = 65536
MIB = 1024 * 1024
MEMORY_NOTE = ("Apple Silicon unified memory: the CPU, the GPU and every other application share one physical pool. "
               "server_phys_footprint is the llama-server process's physical footprint including its Metal "
               "allocations; metal_* values are the buffers the server logged on the Metal device; available memory, "
               "swap and pressure are host-wide and include other applications. None of these is VRAM.")


def scrubbed_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The server's environment: exactly what preparation verified (`native_prep.native_environment`), so no
    LLAMA_*/GGML_* variable can override the argv, no DYLD_* can load code the pin does not cover, no Metal debug
    layer changes timing, and no proxy sits between the evaluator and a loopback server."""
    from .native_prep import native_environment
    return native_environment(base)


def free_loopback_port() -> int:
    """An ephemeral loopback port that was free a moment ago. The race with another process is real but narrow,
    and it cannot produce a wrong measurement: the server either binds it or fails to start, and readiness also
    requires the server's own `listening on http://127.0.0.1:<port>` line."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


def port_accepts(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((LOOPBACK, port), timeout=timeout):
            return True
    except OSError:
        return False


@contextmanager
def _sigint_deferred():
    """Hold SIGINT for the bounded block inside, then deliver it to whatever handled it before.

    One Ctrl-C through the sweep wrappers reaches the harness twice: the terminal's own SIGINT, and the one the
    supervisor forwards about a quarter of a second later. The first starts cleanup; the second used to land in
    the middle of it -- between SIGTERM and the SIGKILL that must follow, or before SIGTERM was ever sent -- and
    leave a llama-server that no signal from the terminal can reach (its own session). While this block runs a
    SIGINT is only recorded; on the way out the previous handler is restored and a recorded SIGINT is raised
    again, so the interrupt is deferred, never dropped. Signal handlers belong to the main thread: anywhere else,
    or when the current handler was not installed from Python (and so could not be put back), nothing is changed.
    """
    pending: list[int] = []
    previous, installed = None, False
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGINT)
        if previous is not None:
            try:
                signal.signal(signal.SIGINT, lambda signum, frame: pending.append(signum))
                installed = True
            except (ValueError, OSError):
                installed = False
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)
            if pending:
                signal.raise_signal(signal.SIGINT)


@dataclass(frozen=True)
class NativePlan:
    """The native counterpart of `ComposePlan`: what will run, and the labels that tie it to this attempt."""
    project_name: str
    server_argv: tuple[str, ...]
    labels: Mapping[str, str]
    port: int | None = None

    def compose_json(self) -> str:  # never written: a native plan has no Compose project
        raise NotImplementedError("a native plan has no Compose project")


@dataclass
class _Output:
    """Shaped like the executor's WorkerResult, so the base runner's log helpers work unchanged."""
    status: str
    returncode: int | None
    stdout: bytes
    stderr: bytes = b""


class NativeServerProcess:
    """One owned llama-server process and nothing else.

    Signals only ever go to the process group of a child this object started and has not yet reaped, so a
    recycled pid can never be hit. `remaining()` is the proof of absence cleanup needs.
    """

    def __init__(self, executable: str, argv, *, cwd: Path, env: Mapping[str, str], log_sink, log_path: Path,
                 log_cap: int, popen=subprocess.Popen) -> None:
        self.executable, self.argv = executable, tuple(argv)
        self.cwd, self.env, self.log_path, self.log_cap = Path(cwd), dict(env), Path(log_path), int(log_cap)
        self.log_sink, self.popen = log_sink, popen
        self.process = None
        self.pgid: int | None = None
        self.log_bytes = self.dropped_bytes = 0
        self.log_error: str | None = None
        self.signals_sent: list[str] = []
        self._drain: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        return None if self.process is None else self.process.pid

    def start(self) -> None:
        self.process = self.popen([self.executable, *self.argv], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, cwd=str(self.cwd), env=self.env, close_fds=True,
                                  start_new_session=True, shell=False)
        self.pgid = self.process.pid  # a new session makes the child its own process-group leader
        self._drain = threading.Thread(target=self._drain_output, name="llama-server-log", daemon=True)
        self._drain.start()

    def _drain_output(self) -> None:
        """Keep the pipe empty (a full pipe would stall the server) and keep at most `log_cap` bytes of it."""
        stream = self.process.stdout
        try:
            while True:
                chunk = stream.read1(LOG_CHUNK_BYTES) if hasattr(stream, "read1") else stream.read(LOG_CHUNK_BYTES)
                if not chunk:
                    break
                room = max(0, self.log_cap - self.log_bytes)
                kept = chunk[:room]
                if kept and self.log_sink is not None:
                    try:
                        self.log_sink.write(kept)
                        self.log_sink.flush()
                    except Exception as exc:  # the artifact budget ran out: keep draining, stop keeping
                        self.log_error = self.log_error or f"{type(exc).__name__}: {exc}"
                        self.log_cap = self.log_bytes
                        kept = b""
                self.log_bytes += len(kept)
                self.dropped_bytes += len(chunk) - len(kept)
        except Exception as exc:
            self.log_error = self.log_error or f"{type(exc).__name__}: {exc}"
        finally:
            if self.log_sink is not None:
                try:
                    self.log_sink.close()
                except Exception as exc:
                    self.log_error = self.log_error or f"{type(exc).__name__}: {exc}"

    def state(self) -> tuple[str, str, str]:
        """(status, exit, oom) like `docker inspect`. There is no OOM flag on macOS: a memory kill is a SIGKILL
        the harness did not send, which `describe_exit` reports as exactly that and no more."""
        if self.process is None:
            return ("created", "", "false")
        code = self.process.poll()
        return ("running", "", "false") if code is None else ("exited", str(code), "false")

    def read_log(self, max_bytes: int) -> _Output:
        try:
            with self.log_path.open("rb") as handle:
                data = handle.read(max(0, int(max_bytes)) + 1)
        except FileNotFoundError:
            return _Output("completed", 0, b"")
        except OSError as exc:
            return _Output("environment-error", None, b"", str(exc).encode())
        if len(data) > max_bytes:
            return _Output("output-limit", 0, data[:max_bytes])
        return _Output("output-limit" if self.dropped_bytes else "completed", 0, data)

    def describe_exit(self) -> str | None:
        if self.process is None or self.process.returncode is None:
            return None
        code = self.process.returncode
        if code >= 0:
            return f"returncode={code}"
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = str(-code)
        return f"signal={name}"

    def unexpected_kill(self) -> bool:
        """The server died of SIGKILL that this object never sent."""
        return (self.process is not None and self.process.returncode == -signal.SIGKILL
                and "SIGKILL" not in self.signals_sent)

    def _signal_group(self, sig: signal.Signals) -> None:
        if self.process is None or self.process.returncode is not None:
            return  # reaped (or never started): its pid may already belong to someone else
        try:
            os.killpg(self.pgid, sig)
            self.signals_sent.append(sig.name)
        except ProcessLookupError:
            pass

    def terminate(self) -> None:
        """Ask the whole owned group to stop (the watchdog's action); never waits, never reaps."""
        self._signal_group(signal.SIGTERM)

    def stop(self, grace_seconds: float, deadline: float, clock=time.monotonic) -> None:
        """SIGTERM to the owned group, SIGKILL once the grace period is over, then any member the group still holds.

        Whatever cuts this short -- a KeyboardInterrupt in a wait, any error -- forfeits the grace period, never the
        stop: the group is killed before the interruption propagates. The server runs in its own session, so no
        terminal signal will ever reach it, and nothing after an interrupted stop would send the SIGKILL.
        """
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self._signal_group(signal.SIGTERM)
                try:
                    self.process.wait(timeout=max(0.1, min(grace_seconds, deadline - clock())))
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    try:
                        self.process.wait(timeout=max(0.1, min(10.0, deadline - clock())))
                    except subprocess.TimeoutExpired:
                        pass
            self._reap_group_members()
        except BaseException:
            try:
                self._signal_group(signal.SIGKILL)  # still unreaped, so its pid cannot belong to anyone else
                self._reap_group_members()
            except Exception:  # the interruption is what the caller must see; remaining() reports any survivor
                pass
            raise
        if self._drain is not None:
            self._drain.join(timeout=max(0.1, min(5.0, deadline - clock())))

    def _reap_group_members(self) -> None:
        """Descendants left in the group after the leader exited are ours too; the group id stays valid (and
        cannot be reused) while any member lives, so signalling it here cannot reach a stranger."""
        if self.pgid is None:
            return
        try:
            os.killpg(self.pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        try:
            os.killpg(self.pgid, signal.SIGKILL)
            self.signals_sent.append("SIGKILL(group)")
        except ProcessLookupError:
            pass

    def remaining(self, marker: tuple[str, ...]) -> list[str]:
        """Every owned process still present: the child itself, anything in its process group, and any process
        whose command line carries this attempt's unique marker (alias and port)."""
        found = []
        if self.process is not None and self.process.poll() is None:
            found.append(f"pid:{self.process.pid}")
        if self.pgid is not None:
            try:
                os.killpg(self.pgid, 0)
                found.append(f"pgid:{self.pgid}")
            except ProcessLookupError:
                pass
            except PermissionError:
                found.append(f"pgid:{self.pgid}")
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "cmdline"]):
                cmdline = proc.info.get("cmdline") or []
                if all(any(part == item for part in cmdline) for item in marker) and proc.pid != os.getpid():
                    entry = f"pid:{proc.pid}"
                    if entry not in found:
                        found.append(entry)
        except Exception as exc:  # an unreadable process table is not proof of absence
            found.append(f"process-table-unreadable:{type(exc).__name__}")
        return found


def _default_process_factory(**kwargs) -> NativeServerProcess:
    return NativeServerProcess(**kwargs)


def _default_sampler(pid=None, **kwargs) -> dict:
    from ..apple import sample_unified_memory
    return sample_unified_memory(pid, **kwargs)


class NativeRunner(ContainerRunner):
    """Runs one `metal-native` candidate. Same public contract and run-directory contract as `ContainerRunner`.

    Any injected boundary (process factory, telemetry, HTTP, evaluator, hasher, identity check) makes the run
    synthetic, exactly as injected Docker boundaries do for containers, so a test fake is never reported as a model.
    """

    def __init__(self, *, policy_path: str | Path = "runtime-policy.json", policy=None, lease_path=None,
                 coding_hook=None, broker_factory=None, http_factory=None, evaluator=None, telemetry=None,
                 hasher=None, clock=time.monotonic, sleep=time.sleep, session_lock: SessionLock | None = None,
                 process_factory=None, identity_check=None, sandbox_probe=None, registry_validator=None) -> None:
        super().__init__(http_factory=http_factory, evaluator=evaluator, registry_validator=registry_validator,
                         capabilities_dir="artifacts/native-prep", clock=clock, sleep=sleep,
                         session_lock=session_lock, policy_path=policy_path, telemetry=telemetry, hasher=hasher,
                         broker_factory=broker_factory, lease_path=lease_path, policy=policy,
                         coding_hook=coding_hook)
        injected = (process_factory, identity_check, sandbox_probe)
        self.synthetic = self.synthetic or any(item is not None for item in injected)
        if broker_factory is None and self.synthetic:
            self.broker_factory = None  # fakes never start a Docker worker broker
        self.process_factory = process_factory or _default_process_factory
        self.identity_check, self.sandbox_probe = identity_check, sandbox_probe
        self.telemetry_injected = telemetry is not None
        self.sampler = telemetry or _default_sampler

    def run(self, config: ContainerRunConfig, output_dir: str | Path, *,
            remaining_budget_seconds: float | None = None, lease_held: bool = False) -> ContainerRunResult:
        if config.runtime != "metal-native":
            raise ValueError("NativeRunner runs metal-native candidates only")
        if remaining_budget_seconds is not None and (type(remaining_budget_seconds) not in (int, float)
                or not math.isfinite(remaining_budget_seconds) or remaining_budget_seconds <= 0):
            raise ValueError("remaining budget must be finite and positive")
        run_dir = Path(output_dir).resolve()
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise ValueError("candidate output directory must be new or empty")
        from .preflight import check_path_lengths
        check_path_lengths(run_dir)
        return _NativeAttempt(self, config, run_dir, remaining_budget_seconds, lease_held).execute()


class _NativeAttempt(_Attempt):
    LEASE_OWNER = "native-run"
    VERIFY_DETAIL = "startup log, load time and unified memory after load (server footprint, Metal buffers, swap)"
    # The deferred SIGINT (`_sigint_deferred`) is delivered right after the absence proof is recorded; that proof
    # stands (see `_Attempt._keeps_proven_cleanup`) and the run is recorded as cancelled.
    KEEPS_PROVEN_CLEANUP_ON_INTERRUPT = True

    def __init__(self, runner: NativeRunner, config: ContainerRunConfig, run_dir: Path, remaining_budget_seconds,
                 lease_held: bool) -> None:
        self.limits: NativeLimits = config.native_limits
        self.server: NativeServerProcess | None = None
        self.memory: dict[str, Any] = {"kind": "apple-unified", "note": MEMORY_NOTE}
        self.native_startup: dict[str, Any] = {}
        self.metal_budget_bytes: int | None = None
        self.watchdog = None
        self.watchdog_violation: str | None = None
        self.sandbox: dict[str, Any] | None = None
        self.admission_sample: dict | None = None
        self.launch_error: str | None = None  # the OSError with which starting the server failed, if it did
        super().__init__(runner, config, run_dir, remaining_budget_seconds, lease_held)
        self.container_mode = False

    # ---- plan ---------------------------------------------------------------------------------------------
    def _initial_plan(self):
        labels = {"llmbench.session": self.session_id, "llmbench.attempt": self.attempt_id,
                  "llmbench.fingerprint": self.config.fingerprint(), "llmbench.runtime": "metal-native"}
        argv = build_server_argv(self.config, model_path=self.config.server_model_path, host=LOOPBACK, port=0)
        return NativePlan(project_name=f"llmbench-{self.attempt_id[:12]}", server_argv=argv, labels=labels), None

    # ---- primitives replacing Docker -----------------------------------------------------------------------
    def _call(self, argv, bound: float = 30, max_output_bytes: int = 1_048_576):
        raise RuntimeError("the native runtime issues no Docker commands")

    def _container_state(self, timeout: int, container_id: str | None = None):
        return None if self.server is None else self.server.state()

    def _fetch_log(self, timeout: int, container_id: str | None = None, max_output_bytes: int | None = None):
        if self.server is None:
            return _Output("environment-error", None, b"", b"no server process")
        return self.server.read_log(self.config.limits.log_max_bytes if max_output_bytes is None
                                    else max_output_bytes)

    def _sample(self, deadline: float | None = None, *, pid: int | None = None) -> dict:
        remaining = self._remaining() if deadline is None else deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("telemetry has no remaining phase allowance")
        if self.runner.telemetry_injected:  # injected providers own their simulated timing
            return self.runner.sampler(pid)
        return self.runner.sampler(pid, timeout=min(3.0, remaining))

    def _telemetry(self, deadline: float | None = None) -> dict:
        return self._sample(deadline, pid=self.server.pid if self.server is not None else None)

    # ---- stages ---------------------------------------------------------------------------------------------
    def _admit(self) -> None:
        config, runner = self.config, self.runner
        self.artifacts.write("config.json", (canonical_json(config.model_dump(mode="json")) + "\n").encode("utf-8"))
        if self.wall < config.bounds.minimum_wall_seconds():
            raise _Stop("rejected", f"granted {self.wall:.0f}s is below the stage bounds "
                                    f"{config.bounds.minimum_wall_seconds()}s")
        from .runtime import not_enforced_settings, required_operations, unsupported_settings
        unsupported = unsupported_settings(config)
        if unsupported:
            raise _Stop("rejected", "settings the metal-native runtime cannot honour: " + "; ".join(unsupported))
        if platform.system() != "Darwin" and not runner.synthetic:
            raise _Stop("rejected", f"the metal-native runtime needs macOS on Apple Silicon, not {platform.system()}")
        try:
            lock = runner.session_lock or SessionLock.read(runner.policy_path)
            for operation in required_operations(config):
                lock.check(operation, RunMode.LIVE)
            registry_digest = runner.registry_validator(config)
        except (OperationForbidden, ValueError, OSError, ImportError) as exc:
            raise _Stop("rejected", f"{type(exc).__name__}: {exc}") from exc
        try:
            identity = self._verify_server_identity(lock)
        except (PreflightError, BackendError, ValueError, OSError) as exc:  # BackendError: help/argv refusals
            raise _Stop("rejected", f"native server identity: {exc}") from exc
        if not self.lease_held:
            from .lease import LeaseHeld
            try:
                self.lease.acquire()
            except (LeaseHeld, OSError) as exc:
                raise _Stop("rejected", f"gpu_lease_unavailable: {exc}") from exc
        try:
            memory = self._check_headroom()
        except PreflightError as exc:
            raise _Stop("rejected", str(exc)) from exc
        self.image_evidence = {"native_server": identity}
        self.memory["admission"] = memory
        preflight = {"runtime": "metal-native", "native_server": identity, "memory": memory,
                     "registry_digest": registry_digest, "help_sha256": identity.get("help_sha256"),
                     "server_build": identity.get("build_info"), "not_enforced": not_enforced_settings(config)}
        self.artifacts.write_json("preflight.json", preflight)

    def _verify_server_identity(self, lock: SessionLock) -> dict:
        """The pinned executable and libraries by hash, then the linkage proof, then its own --version and --help,
        then the argv.

        The linkage proof (`native_prep.check_native_linkage`) is what makes the libraries pin complete: hashing the
        files beside the executable says nothing about a build that loads its Metal backend from `../lib` or
        `/opt/homebrew`. Preparation proves it once, but nothing forces a config through preparation (a candidate
        needs no `--native-bundle`, and a bundle is unauthenticated JSON), so admission proves it again from the
        very files it just hashed before anything is executed. It only reads Mach-O load commands."""
        if self.runner.identity_check is not None:
            return self.runner.identity_check(self.config, self.plan.server_argv)
        from .native_prep import NativeProcessExecutor, check_native_linkage, hash_native_server
        ref = self.config.native_server
        executable = Path(ref.executable)
        hashed = hash_native_server(executable)
        problems = []
        if hashed["executable_sha256"] != ref.executable_sha256:
            problems.append(f"executable sha256 {hashed['executable_sha256']} is not the pinned {ref.executable_sha256}")
        if hashed["libraries_sha256"] != ref.libraries_sha256:
            problems.append("the llama.cpp/ggml libraries beside the executable differ from the pinned set")
        if problems:
            raise PreflightError("; ".join(problems))
        try:
            check_native_linkage(executable, hashed["libraries"])
        except ValueError as exc:
            raise PreflightError(str(exc)) from exc
        executor = NativeProcessExecutor(executable, session_lock=lock, cwd=self.run_dir)
        version = executor.run("--version", timeout_seconds=self._timeout(PROBE_TIMEOUT_SECONDS))
        helptext = executor.run("--help", timeout_seconds=self._timeout(PROBE_TIMEOUT_SECONDS))
        for name, result in (("--version", version), ("--help", helptext)):
            if result.status != "completed" or result.returncode != 0:
                raise PreflightError(f"llama-server {name} failed ({result.status}, rc={result.returncode})")
        version_text = (version.stdout + version.stderr).decode("utf-8", "replace")
        help_text = helptext.stdout.decode("utf-8", "replace")
        self.artifacts.write("capabilities/llama-server-version.txt", version_text.encode("utf-8"))
        self.artifacts.write("capabilities/llama-server-help.txt", help_text.encode("utf-8"))
        caps = parse_server_help(help_text, version_text=version_text)
        require_supported(self.plan.server_argv, caps, expected_help_sha256=ref.help_sha256)
        if caps.build != ref.build_info:
            raise PreflightError(f"the executable reports build {caps.build}, not the pinned {ref.build_info}")
        return {"executable": str(executable), "executable_sha256": hashed["executable_sha256"],
                "libraries_sha256": hashed["libraries_sha256"], "libraries": hashed["libraries"],
                "build_info": caps.build, "help_sha256": caps.help_sha256,
                "compiler": next((line.strip() for line in version_text.splitlines() if "built with" in line), None)}

    def _check_headroom(self) -> dict:
        """Three samples half a second apart: one GPU-utilisation reading is noise, three show a workload."""
        from ..apple import check_unified_memory_headroom
        samples = []
        for index in range(3):
            samples.append(self._sample(pid=None))
            if index < 2:
                self.runner.sleep(0.5)
        self.admission_sample = samples[-1]
        required = self.config.model.size_bytes + self.limits.memory_reserve_mib * MIB
        return check_unified_memory_headroom(samples, required_bytes=required, limits=self.limits)

    def _plan(self) -> None:
        port = free_loopback_port()
        argv = build_server_argv(self.config, model_path=self.config.server_model_path, host=LOOPBACK, port=port)
        self.plan = NativePlan(project_name=self.plan.project_name, server_argv=argv, labels=self.plan.labels,
                               port=port)
        env = scrubbed_environment()
        self.environment = env
        self.artifacts.write_json("plan/server-argv.json", {
            "entrypoint": [self.config.native_server.executable], "argv": list(argv),
            "environment": {}, "environment_scrubbed": sorted(set(os.environ) - set(env)),
            "project": self.plan.project_name, "labels": dict(self.plan.labels)})
        self.artifacts.write_json(LAUNCH_RECORD, {
            "runtime": "metal-native", "executable": self.config.native_server.executable, "argv": list(argv),
            "host": LOOPBACK, "port": port, "process_group": "own session (start_new_session)",
            "stop": {"signal": "SIGTERM", "grace_seconds": self.limits.stop_grace_seconds, "then": "SIGKILL"},
            "labels": dict(self.plan.labels)})

    def _start(self) -> None:
        self.up_attempted = True
        sink = self.artifacts.open_external(NATIVE_LOG, "xb")
        self.server = self.runner.process_factory(
            executable=self.config.native_server.executable, argv=self.plan.server_argv, cwd=self.run_dir,
            env=self.environment, log_sink=sink, log_path=self.run_dir / NATIVE_LOG,
            log_cap=self.config.limits.log_max_bytes)
        try:
            self.server.start()
        except OSError as exc:  # Popen reports a failed fork or exec only after reaping any child it made
            self.launch_error = f"{type(exc).__name__}: {exc}"
            raise _Stop("failed", f"llama-server could not be started: {exc}") from exc
        self.container_id = f"pid:{self.server.pid}"
        self.base_url = f"http://{LOOPBACK}:{self.plan.port}"
        if self.lease.held:
            # A standalone run's own lease names the server it now owns, so a lease left behind by a killed harness
            # tells the reader which llama-server to look for (`locks.describe_lock`). A lease taken by a campaign
            # belongs to the campaign and is never rewritten here.
            self._guard("lease_server_pid_not_recorded", lambda: self.lease.annotate(
                runtime="metal-native", server_pid=self.server.pid))

    def _ready(self) -> None:
        super()._ready()
        log = self._fetch_log(self._timeout(10)).stdout.decode("utf-8", "replace")
        listening = parse_startup_log(log).get("listening")
        if listening is not None and listening.rstrip("/") != self.base_url:
            raise _Stop("failed", f"the server reports listening on {listening}, not the planned {self.base_url}")

    def _verify(self) -> None:
        text = self._save_log("logs/startup.log", self._timeout(30))
        self.startup = parse_startup_log(text)
        self.load_seconds = self.startup.get("model_loaded_seconds")
        from .readback import parse_native_startup
        self.native_startup = parse_native_startup(text)
        budget = self.native_startup.get("metal_budget_mib")
        self.metal_budget_bytes = int(budget * MIB) if isinstance(budget, (int, float)) and budget > 0 else None
        sample = self._telemetry()
        self.memory["after_load"] = _memory_view(sample)
        self.memory["server_log"] = {key: self.native_startup.get(key) for key in (
            "devices", "model_device", "metal_device", "metal_budget_mib", "metal_resident_mib", "host_resident_mib",
            "model_buffers", "kv_buffers", "compute_buffers", "recurrent_buffers", "offloaded_layers",
            "total_layers", "system_info_backends", "metal_embed_library")}
        offloaded, total = self.startup.get("offloaded_layers"), self.startup.get("total_layers")
        metal = self.native_startup.get("metal_resident_mib")
        if not self.native_startup.get("metal_device") and not any(
                str(item.get("device", "")).startswith("MTL") for item in self.startup.get("model_buffers") or []):
            self.warnings.append("metal_device_not_observed_in_startup_log")
        if self.metal_budget_bytes and isinstance(metal, (int, float)) and metal * MIB > self.metal_budget_bytes:
            self.warnings.append(f"metal_buffers_{metal:.0f}MiB_exceed_the_{budget:.0f}MiB_metal_working_set")
        footprint = (sample.get("process") or {}).get("phys_footprint_bytes")
        self.artifacts.write_json("verify.json", {"startup": self.startup, "native_startup": self.native_startup,
                                                  "ready_seconds": self.ready_seconds, "telemetry": sample,
                                                  "gpu_offload": {"offloaded_layers": offloaded,
                                                                  "total_layers": total},
                                                  "server_phys_footprint_bytes": footprint})

    def _broker(self) -> None:
        """Coding suites need the Docker sandbox. Unavailable is recorded as blocked, and nothing is generated."""
        probe = self.runner.sandbox_probe
        if probe is None:
            from ..coding.sandbox import probe_docker_sandbox
            probe = probe_docker_sandbox
        try:
            self.sandbox = probe(session_lock=self._lock(), image_ref=self.config.worker_image)
        except Exception as exc:
            self.sandbox = {"status": "blocked", "reason": f"sandbox probe failed: {type(exc).__name__}: {exc}"}
        self.artifacts.write_json("broker/sandbox-probe.json", self.sandbox)
        if self.sandbox.get("status") != "available":
            self.warnings.append(f"coding_sandbox_blocked: {self.sandbox.get('reason')}")
            return
        self.broker = self.runner.broker_factory(self.run_dir, self.config, self._lock(),
                                                 self.artifacts.scoped("broker"), attempt_id=self.attempt_id,
                                                 session_id=self.session_id)
        for method in ("tick", "cancel_all", "direct_client"):
            if not callable(getattr(self.broker, method, None)):
                raise _Stop("failed", f"broker factory returned an object without {method}()")

    def _evaluate(self) -> None:
        from ..apple import MemoryWatchdog
        self.watchdog = MemoryWatchdog(
            self.server.pid, interval=self.limits.sample_interval_seconds, limits=self.limits,
            budget_bytes=self.metal_budget_bytes, sampler=self.runner.sampler,
            on_violation=self._on_memory_violation, clock=self.clock,
            baseline=self.admission_sample)  # swap growth counts from before the load, not from after it
        self.watchdog.start()
        try:
            super()._evaluate()
        except _Stop as stop:
            if self.watchdog_violation:  # the evaluator failed because the watchdog stopped the server: say so
                raise _Stop("failed", f"native_memory_watchdog: {self.watchdog_violation} "
                                      f"(evaluator: {stop.reason})") from stop
            raise
        finally:
            self._stop_watchdog()
        if self.watchdog_violation:
            raise _Stop("failed", f"native_memory_watchdog: {self.watchdog_violation}")
        state = self.server.state()
        if state[0] != "running":
            raise _Stop("failed", f"llama-server exited during the evaluation ({self.server.describe_exit()})")

    def _evaluation_context_extras(self) -> dict:
        blocked = self.sandbox is not None and self.sandbox.get("status") != "available"
        return {"execution_unavailable_reason": self.sandbox.get("reason") or "sandbox unavailable"} if blocked else {}

    def _on_memory_violation(self, reason: str) -> None:
        self.watchdog_violation = reason
        if self.server is not None:
            self.server.terminate()  # the evaluator's requests now fail; the stage reports the watchdog reason

    def _stop_watchdog(self) -> None:
        if self.watchdog is None:
            return
        try:
            summary = self.watchdog.stop()
        except Exception as exc:
            summary = {"error": f"{type(exc).__name__}: {exc}"}
        self.watchdog = None
        samples = summary.pop("samples", None) if isinstance(summary, dict) else None
        if samples:
            self._guard("memory_samples_not_persisted", lambda: self.artifacts.write(
                "memory-samples.jsonl", "".join(canonical_json(row) + "\n" for row in samples).encode("utf-8")))
        self.memory["during_evaluation"] = summary

    # ---- cleanup ---------------------------------------------------------------------------------------------
    def _cleanup(self) -> None:
        started = self.up_attempted and self.server is not None
        # Bounded throughout: the watchdog's join timeout, the stop's grace and deadline, the absence checks and the
        # broker's own bounds. Nothing in here may be cut short by the second SIGINT of a single Ctrl-C.
        with _sigint_deferred():
            try:
                self._stop_watchdog()
            finally:  # whatever the watchdog does, the server is stopped
                deadline = self.clock() + self.config.bounds.cleanup_reserve_seconds
                if started:
                    self._stop_and_prove_absence(deadline)
                else:
                    self.cleanup = CleanupEvidence(attempted=False, verified=True)
        if not started:
            if self.model_evidence:
                self._reverify_model()
            return
        if self.server.log_error or self.server.dropped_bytes:
            self.warnings.append(f"inference_log_truncated:{NATIVE_LOG}: kept {self.server.log_bytes} bytes, dropped "
                                 f"{self.server.dropped_bytes}" + (f" ({self.server.log_error})"
                                                                   if self.server.log_error else ""))
        final = self._guard_text("final_log_unavailable", lambda: self._save_log(
            "logs/inference-final.log", 5))
        if final:
            from .readback import parse_memory_breakdown
            breakdown = parse_memory_breakdown(final)
            if breakdown:
                self.memory["memory_breakdown_at_exit"] = breakdown
        self._reverify_model()
        try:
            self.artifacts.write_json("cleanup.json", {"cleanup": self.cleanup.model_dump(mode="json"),
                                                       "telemetry": self._sample(deadline, pid=None)})
        except Exception as exc:
            self.warnings.append(f"cleanup_telemetry_unavailable: {type(exc).__name__}: {exc}")

    def _stop_and_prove_absence(self, deadline: float) -> None:
        """Stop the server, then record positive proof that nothing it was remains; the broker's workers after it.

        The server goes first so the broker's Docker calls can neither delay the SIGTERM nor use up its grace. A
        launch that never produced a process is recorded as `not-started: <the OSError>`, which is what the launcher
        reported, and the absence checks still run: evidence, not the assumption that a failed start left nothing.
        """
        evidence: dict[str, Any] = {"attempted": True, "verified": False, "error": None}
        try:
            self.server.stop(self.limits.stop_grace_seconds, deadline, clock=self.clock)
            evidence["server_exit"] = self.server.describe_exit()
            if evidence["server_exit"] is None and self.server.pid is None and self.launch_error:
                evidence["server_exit"] = f"not-started: {self.launch_error}"
            if self.server.unexpected_kill():
                self._fail("failed", "inference_killed_by_sigkill: the server died of a SIGKILL the harness did not "
                                     "send (macOS sends that under memory pressure; not proven here)")
            elif self.server.state()[0] == "exited" and not self.server.signals_sent and self.state == "completed":
                self._fail("failed", f"llama-server exited on its own ({evidence['server_exit']})")
            marker = ("--alias", self.config.alias(), "--port", str(self.plan.port))
            remaining = self.server.remaining(marker)
            if self.plan.port is not None and port_accepts(self.plan.port):
                remaining.append(f"port:{self.plan.port}-still-accepting")
            evidence.update(processes_remaining=tuple(remaining), verified=not remaining)
            self.containers_absent = not remaining
            if remaining:
                evidence["error"] = "owned server processes remain after the stop"
        except Exception as exc:
            evidence.update(verified=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            broker_error = self._cancel_broker() if self.broker is not None else None
            if broker_error and evidence["verified"]:
                evidence.update(verified=False, error=broker_error)
            self.cleanup = CleanupEvidence(**evidence)

    def _guard_text(self, label: str, action) -> str:
        try:
            return action()
        except Exception as exc:
            self.warnings.append(f"{label}: {type(exc).__name__}: {exc}")
            return ""

    # ---- result -----------------------------------------------------------------------------------------------
    def _result(self, reports: dict) -> ContainerRunResult:
        result = super()._result(reports)
        memory = dict(self.memory)
        if self.sandbox is not None:
            memory["coding_sandbox"] = self.sandbox
        try:
            canonical_json(memory)
        except (TypeError, ValueError) as exc:
            memory = {"kind": "apple-unified", "note": MEMORY_NOTE, "error": f"memory evidence not JSON: {exc}"}
        return result.model_copy(update={"runtime": "metal-native", "memory": memory})


def _memory_view(sample: dict) -> dict:
    """A labelled, MiB-rounded copy of one unified-memory sample for the result (the raw sample stays in
    verify.json)."""
    def mib(value):
        return None if not isinstance(value, (int, float)) else round(value / MIB, 1)

    process = sample.get("process") or {}
    gpu = sample.get("gpu") or {}
    return {"server_phys_footprint_mib": mib(process.get("phys_footprint_bytes")),
            "server_rss_mib": mib(process.get("rss_bytes")),
            "host_memory_total_mib": mib(sample.get("host_memory_total_bytes")),
            "host_memory_available_mib": mib(sample.get("host_memory_available_bytes")),
            "swap_used_mib": mib(sample.get("swap_used_bytes")),
            "memory_pressure_level": sample.get("memory_pressure_level"),
            "gpu_device_utilization_percent": gpu.get("device_utilization_percent"),
            "gpu_in_use_system_memory_mib": mib(gpu.get("in_use_system_memory_bytes")),
            "power": sample.get("power"), "errors": list(sample.get("errors") or [])}


__all__ = ["NativeRunner", "NativeServerProcess", "NativePlan", "scrubbed_environment", "free_loopback_port",
           "port_accepts", "MEMORY_NOTE"]
