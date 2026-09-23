"""One bounded llama.cpp container candidate: admit, start, verify, evaluate, clean up, report.

Construction is inert. Docker runs only through the injected/owned ComposeExecutor after the persistent
session policy authorizes container, load and inference operations. Cleanup touches only resources that
carry this attempt's unique Compose project label.

Two evaluator modes share every stage up to `verify`. Host-process mode (unchanged) publishes a loopback port
and runs the evaluator in this process. Container mode publishes nothing: readiness comes from the inference
container's own startup log, the evaluator container receives a typed wall-clock grant on its argv, a host
watchdog tears it down on overrun, and its `evaluation.json` is ingested only after byte, schema and
denominator checks.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from ..backends.base import BackendError
from ..backends.http import TransportDeadlineExceeded
from ..config import RunMode, canonical_json
from ..safety import OperationForbidden, SessionLock
from ..store import utc_now
from .artifacts import RunArtifacts
from .capabilities import load_capabilities, require_supported
from .config import (EVALUATOR_ENTRYPOINT, ArtifactEntry, ChildGrant, CleanupEvidence, ContainerRunConfig,
                     ContainerRunResult, StageRecord)
from .executor import RESOURCE_ID, STATE_FORMAT, ComposeExecutor, compose_prefix, project_filter
from .lease import GpuLease, LeaseHeld
from .plan import build_compose_plan
from .preflight import (PreflightError, PreflightTimeout, check_gpu_headroom, check_path_lengths,
                        reverify_identity, verify_image, verify_model_asset)
from .readback import build_settings_evidence, parse_startup_log, settings_verified, unverified_required

STOPPED_STATES = frozenset({"exited", "dead", "removing"})
VRAM_WARNING_FRACTION = 0.97
INGEST_MARGIN_SECONDS = 30  # host time reserved after the child exits: log capture, ingestion, adoption
BROKER_TICK_MAX_SECONDS = 300
EVALUATOR_LOG_TAIL = 600


class _Stop(Exception):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(reason)
        self.state, self.reason = state, reason


def _default_registry_validator(config: ContainerRunConfig) -> str:
    from ..registry import builtin_registry
    registry = builtin_registry()
    from ..container_eval import baked_datasets
    findings = registry.validate(config.benchmarks, broker_available=config.broker is not None,
                                 datasets_available=baked_datasets(config.dataset_root))
    if findings and not isinstance(findings, str):
        raise ValueError("; ".join(str(item) for item in findings))
    digest = registry.digest()
    if config.registry_digest is not None and config.registry_digest != digest:
        raise ValueError("configured registry_digest does not match the installed benchmark registry")
    return digest


def _default_evaluator(config: ContainerRunConfig, context) -> dict:
    from ..container_eval import EvaluationContext, run_evaluation
    names = EvaluationContext.__dataclass_fields__
    fields = {key: value for key, value in vars(context).items() if key in names}
    root = getattr(config, "dataset_root", None)
    if root is not None:  # host-process runs read the host staging directory, not the image path
        fields["dataset_root"] = root
    return run_evaluation(config, EvaluationContext(**fields))


def _default_broker_factory(run_dir, config, session_lock, artifacts, **namespace):
    """The host coding broker, imported only when a run needs it. A native run passes its attempt and session
    ids (`namespace`) because it has no Compose plan for the broker to read them from."""
    from ..coding.broker import HostBroker
    return HostBroker.factory(run_dir, config, session_lock, artifacts, **namespace)


def _default_http_factory(base_url: str, *, timeout: float):
    from ..backends.http import UrllibHTTPTransport
    return UrllibHTTPTransport(base_url, timeout=timeout)


def _default_telemetry(*, timeout_seconds: float = 3.0) -> dict:
    from ..telemetry import sample_system
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise TimeoutError("telemetry has no remaining phase allowance")

    def bounded_command(argv, **kwargs):
        kwargs["timeout"] = min(float(kwargs.get("timeout", timeout_seconds)), timeout_seconds)
        return subprocess.run(argv, **kwargs)

    return sample_system(include_gpu=True, command_runner=bounded_command)


def _text(result) -> str:
    return (result.stdout + result.stderr).decode("utf-8", "replace")


class ContainerRunner:
    def __init__(self, *, executor=None, http_factory=None, evaluator=None, registry_validator=None,
                 capabilities_dir: str | Path = "artifacts/container-prep", clock=time.monotonic,
                 sleep=time.sleep, session_lock: SessionLock | None = None,
                 policy_path: str | Path = "runtime-policy.json", telemetry=None, hasher=None,
                 broker_factory=None, lease_path: str | Path | None = None, policy=None, coding_hook=None) -> None:
        # Any injected boundary makes the run synthetic: it can never be reported as a measured model.
        self.synthetic = any(item is not None for item in (executor, http_factory, evaluator, telemetry, hasher))
        self.executor, self.http_factory = executor, http_factory or _default_http_factory
        self.evaluator = evaluator or _default_evaluator
        self.registry_validator = registry_validator or _default_registry_validator
        self.capabilities_dir = Path(capabilities_dir)
        self.clock, self.sleep = clock, sleep
        self.session_lock, self.policy_path = session_lock, Path(policy_path)
        self.telemetry = telemetry or _default_telemetry
        self.hasher = hasher
        # broker_factory(run_dir, config, session_lock, artifacts) -> BrokerProtocol (plan section 2.4). A real
        # (non-synthetic) runner defaults to coding.broker.HostBroker.factory; a runner with any injected boundary
        # gets a broker only when one is injected too, so fakes never start a Docker worker broker. coding_hook is
        # container_eval's run_coding_benchmarks hook (container_eval resolves its own default).
        if broker_factory is None and not self.synthetic:
            broker_factory = _default_broker_factory
        self.broker_factory, self.coding_hook = broker_factory, coding_hook
        self.lease_path, self.policy = lease_path, policy

    def run(self, config: ContainerRunConfig, output_dir: str | Path, *,
            remaining_budget_seconds: float | None = None, lease_held: bool = False) -> ContainerRunResult:
        if remaining_budget_seconds is not None and (type(remaining_budget_seconds) not in (int, float)
                or not math.isfinite(remaining_budget_seconds) or remaining_budget_seconds <= 0):
            raise ValueError("remaining budget must be finite and positive")
        run_dir = Path(output_dir).resolve()
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise ValueError("candidate output directory must be new or empty")
        check_path_lengths(run_dir)
        attempt = _Attempt(self, config, run_dir, remaining_budget_seconds, lease_held)
        return attempt.execute()


class _Attempt:
    def __init__(self, runner: ContainerRunner, config: ContainerRunConfig, run_dir: Path,
                 remaining_budget_seconds, lease_held: bool) -> None:
        self.runner, self.config, self.run_dir, self.lease_held = runner, config, run_dir, lease_held
        self.clock = runner.clock
        self.started, self.started_utc = self.clock(), utc_now()
        self.attempt_id = uuid.uuid4().hex
        self.session_id = config.session_id or self.attempt_id[:12]
        grants = [config.bounds.candidate_wall_seconds, config.parent_grant_seconds, remaining_budget_seconds]
        self.wall = float(min(item for item in grants if item is not None))
        self.work_deadline = self.started + self.wall - config.bounds.cleanup_reserve_seconds
        self.phase_deadline = None
        self.container_mode = config.evaluator.mode == "container"
        # Provisional grant from the stage bounds alone; the plan stage re-issues it from the measured remainder.
        self.grant = self._issue_grant(self.wall - config.bounds.cleanup_reserve_seconds
                                       - config.bounds.hash_seconds, 0.0) if self.container_mode else None
        self.plan, self.prefix = self._initial_plan()
        self.artifacts = RunArtifacts(run_dir, config.limits.max_artifact_bytes,
                                      terminal_reserve_bytes=min(262144, config.limits.max_artifact_bytes // 4))
        self.executor = runner.executor
        self.lease = GpuLease(runner.lease_path, owner=f"{self.LEASE_OWNER}:{self.attempt_id}")
        self.state, self.failure_stage, self.reasons, self.warnings = "completed", None, [], []
        self.deferred_reasons: list[str] = []  # secondary failure detail recorded after the primary reason
        self.stages: list[StageRecord] = []
        self.stage_sink = None
        self.current, self.up_attempted, self.container_id = "admit", False, None
        self.evaluator_id, self.broker, self.adoption = None, None, None
        # Container-mode child bookkeeping: last observed evaluator state, whether its allocation was reserved,
        # whether adoption waits for cleanup to prove the child stopped, and whether cleanup proved it.
        self.evaluator_state, self.child_reserved, self.child_adoption_pending = None, False, False
        self.containers_absent, self.evaluator_log_attempted = False, False
        self.image_evidence, self.model_evidence, self.evaluation = {}, {}, None
        self.startup, self.settings, self.cleanup = {}, (), CleanupEvidence(verified=True)
        self.load_seconds = self.vram_after_load = self.ready_seconds = self.base_url = None

    LEASE_OWNER = "container-run"
    VERIFY_DETAIL = "startup log, load time and VRAM after load"

    def _initial_plan(self):
        """The inert plan built at construction: (plan, compose command prefix). The native runtime overrides it."""
        plan = build_compose_plan(self.config, self.run_dir, attempt_id=self.attempt_id, session_id=self.session_id,
                                  grant=self.grant)
        return plan, compose_prefix(plan.project_name, plan.compose_path)

    # ---- bookkeeping -------------------------------------------------------------------------------------
    def _issue_grant(self, available: float, offset: float) -> ChildGrant:
        """Child seconds = min(evaluation bound, host time left after startup, both verifies and ingestion)."""
        bounds = self.config.bounds
        seconds = int(min(bounds.evaluation_seconds, available - bounds.startup_seconds - 2 * bounds.verify_seconds
                          - INGEST_MARGIN_SECONDS))
        self.grant_shortfall = seconds < 1
        return ChildGrant(grant_seconds=max(1, seconds), artifact_bytes=self.config.limits.evaluator_artifact_bytes,
                          issued_utc=utc_now(), issued_host_offset_seconds=max(0.0, offset),
                          watchdog_slack_seconds=self.config.evaluator.watchdog_slack_seconds)

    def _remaining(self) -> float:
        deadline = min(self.work_deadline, self.phase_deadline) if self.phase_deadline is not None else self.work_deadline
        return deadline - self.clock()

    def _timeout(self, bound: float) -> int:
        remaining = self._remaining()
        if remaining < 1:
            raise _Stop("timeout", f"{self.current} phase or candidate wall budget exhausted")
        return int(min(bound, remaining))

    def _call(self, argv, bound: float = 30, max_output_bytes: int = 1_048_576):
        result = self.executor.run(tuple(argv), timeout_seconds=self._timeout(bound), max_output_bytes=max_output_bytes)
        if self._remaining() <= 0:
            raise _Stop("timeout", f"{self.current} phase or candidate wall budget exhausted")
        return result

    def _telemetry(self, deadline: float | None = None) -> dict:
        remaining = self._remaining() if deadline is None else deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("telemetry has no remaining phase allowance")
        if self.runner.telemetry is _default_telemetry:
            return self.runner.telemetry(timeout_seconds=min(3.0, remaining))
        return self.runner.telemetry()  # Injected test providers own their simulated timing.

    def _stage(self, name, action, *, detail: str = "", bound=None, deadline=None):
        self.current, began = name, self.clock()
        previous_deadline = self.phase_deadline
        limits = [value for value in (previous_deadline, deadline,
                                      began + bound if bound is not None else None) if value is not None]
        self.phase_deadline = min(limits) if limits else None
        status = "failed"
        try:
            if self.phase_deadline is not None and self._remaining() <= 0:
                raise _Stop("timeout", f"{name} phase budget exhausted")
            value = action()
            if self.phase_deadline is not None and self._remaining() <= 0:
                raise _Stop("timeout", f"{name} phase budget exhausted")
            status = "ok"
            return value
        except _Stop as stop:
            status, detail = ("timeout" if stop.state == "timeout" else "failed"), stop.reason
            raise
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.phase_deadline = previous_deadline
            self._record(name, status, began, detail)

    def _record(self, name, status, began, detail="") -> None:
        record = StageRecord(name=name, status=status, started_offset_seconds=max(0., began - self.started),
                             elapsed_seconds=max(0., self.clock() - began), detail=str(detail)[:2000])
        self.stages.append(record)
        if self.stage_sink is not None:
            try:
                self.stage_sink(record.model_dump(mode="json"))
            except (OSError, ValueError) as exc:
                self.warnings.append(f"stage_record_not_persisted: {exc}")

    def _guard(self, label: str, action) -> None:
        """Evidence writes must never prevent the cleanup or result that follows them."""
        try:
            action()
        except Exception as exc:
            self.warnings.append(f"{label}: {type(exc).__name__}: {exc}")

    def _fail(self, state: str, reason: str) -> None:
        if self.state == "completed":
            self.state, self.failure_stage = state, self.current
        self.reasons.append(reason)

    def _lock(self) -> SessionLock:
        return self.runner.session_lock or SessionLock.read(self.runner.policy_path)

    # ---- lifecycle ---------------------------------------------------------------------------------------
    def execute(self) -> ContainerRunResult:
        interrupted = None
        with self.artifacts.trace("stages.jsonl") as sink:
            self.stage_sink = sink
            try:
                try:
                    self._work()
                except _Stop as stop:
                    self._fail(stop.state, stop.reason)
                except BaseException as exc:
                    if not isinstance(exc, Exception):
                        interrupted = exc
                        self._fail("cancelled", f"{type(exc).__name__}: {exc}")
                    else:
                        self._fail("failed", f"{type(exc).__name__}: {exc}")
                self.reasons.extend(self.deferred_reasons)
            finally:
                # Owned resources must be reconciled even when a dependency exits the interpreter or cancels.
                before = self.cleanup  # whatever was set before this stage proves nothing the stage did
                try:
                    self._stage("cleanup", self._cleanup)
                except BaseException as exc:
                    if not isinstance(exc, Exception):
                        interrupted = exc
                    if self._keeps_proven_cleanup(exc, before):
                        self._fail("cancelled", f"{type(exc).__name__} during cleanup, after owned-resource "
                                                f"absence was proven: {exc}")
                    else:
                        self.cleanup = self.cleanup.model_copy(update={
                            "verified": False, "error": self._cleanup_error_after(exc, before)})
                if not self.cleanup.verified:
                    self.state, self.failure_stage = "cleanup-uncertain", "cleanup"
                    self.reasons.append("owned_resource_absence_unverified: " + (self.cleanup.error or "unknown"))
                    self.cleanup = self.cleanup.model_copy(update={"lease_retained": self.lease.held})
                else:
                    self.lease.release()
            self._adopt_deferred_child()  # a child alive at the end of its stage is hashed only once proven stopped
            reports = self._report()
            self.stage_sink = None
        self.artifacts.seal_external_logs()
        result = self._result(reports)
        if not self.artifacts.terminal_fits(result.model_dump(mode="json")):
            # The full artifact index lives in its separately bounded file. Keep terminal control information
            # even when verbose diagnostics alone would exhaust the reservation.
            result = result.model_copy(update={
                "warnings": tuple(str(item)[:512] for item in result.warnings[:8]) +
                            ("terminal_details_compacted_to_preserve_bounded_result_and_index",),
                "failure_reasons": tuple(str(item)[:512] for item in result.failure_reasons[:16]),
                "settings": (), "artifacts": (), "stages": (), "image_evidence": {}, "model_evidence": {},
                "speed": {}, "quality": {}, "reports": {},
                "cleanup": result.cleanup.model_copy(update={
                    "error": str(result.cleanup.error)[:512] if result.cleanup.error else None,
                    "containers_remaining": result.cleanup.containers_remaining[:16],
                    "networks_remaining": result.cleanup.networks_remaining[:16]})})
        self.artifacts.finalize_terminal(result.model_dump(mode="json"))
        if interrupted is not None:
            raise interrupted
        return result

    def _work(self) -> None:
        bounds = self.config.bounds
        self._stage("admit", self._admit)
        self._stage("hash", self._hash, bound=bounds.hash_seconds)
        self._stage("plan", self._plan)
        # Container creation and readiness share one startup grant; readiness never resets the timer.
        startup_deadline = min(self.work_deadline, self.clock() + bounds.startup_seconds)
        self._stage("start", self._start, deadline=startup_deadline)
        self._stage("ready", self._ready_from_logs if self.container_mode else self._ready, deadline=startup_deadline)
        self._stage("verify", self._verify, bound=bounds.verify_seconds, detail=self.VERIFY_DETAIL)
        if self.config.broker is not None:
            self._stage("broker", self._broker, bound=bounds.verify_seconds,
                        detail="host coding broker: spool directories and worker reconciliation")
        if self.container_mode:
            grant = self.grant
            self._stage("evaluator", self._evaluate_container,
                        bound=grant.grant_seconds + grant.watchdog_slack_seconds + INGEST_MARGIN_SECONDS,
                        detail=f"evaluator container under a {grant.grant_seconds}s grant "
                               f"(+{grant.watchdog_slack_seconds}s watchdog slack)")
        else:
            self._stage("quality", self._evaluate, bound=bounds.evaluation_seconds,
                        detail="in-process evaluator: attach, count route, overflow probe, warmup, speed, quality")
        self._stage("verify", self._settings_evidence, bound=bounds.verify_seconds,
                    detail="requested versus effective settings")

    def _admit(self) -> None:
        config, runner = self.config, self.runner
        self.artifacts.write("config.json", (canonical_json(config.model_dump(mode="json")) + "\n").encode("utf-8"))
        if self.wall < config.bounds.minimum_wall_seconds():
            raise _Stop("rejected", f"granted {self.wall:.0f}s is below the stage bounds "
                                    f"{config.bounds.minimum_wall_seconds()}s")
        if self.container_mode:
            if config.evaluator.image.entrypoint != EVALUATOR_ENTRYPOINT:
                raise _Stop("rejected", "evaluator image entrypoint must be python -m llmbench.container_eval")
            if self.grant_shortfall:
                raise _Stop("rejected", f"granted {self.wall:.0f}s leaves no evaluator grant after startup, "
                                        "verification, ingestion and cleanup reserves")
        if config.broker is not None and runner.broker_factory is None:
            raise _Stop("rejected", "config.broker is set but no coding broker factory is wired into this runner")
        try:
            lock = runner.session_lock or SessionLock.read(runner.policy_path)
            for operation in ("container", "load", "inference"):
                lock.check(operation, RunMode.LIVE)
            registry_digest = runner.registry_validator(config)
            caps = load_capabilities(runner.capabilities_dir)
            require_supported(self.plan.server_argv, caps, expected_help_sha256=config.help_sha256)
            if caps.build is not None and caps.build != config.build_info:
                raise ValueError(f"saved server version {caps.build} is not the configured build_info")
        except (OperationForbidden, BackendError, ValueError, OSError, ImportError) as exc:
            raise _Stop("rejected", f"{type(exc).__name__}: {exc}") from exc
        if self.executor is None:
            self.executor = ComposeExecutor(session_lock=lock, mode=RunMode.LIVE, clock=self.clock)
        if not self.lease_held:
            try:
                self.lease.acquire()
            except (LeaseHeld, OSError) as exc:
                raise _Stop("rejected", f"gpu_lease_unavailable: {exc}") from exc
        evaluator_evidence = None
        try:
            self.image_evidence = verify_image(self.executor, config.inference_image, timeout_seconds=self._timeout(30))
            if self.container_mode:
                evaluator_evidence = verify_image(self.executor, config.evaluator.image,
                                                  timeout_seconds=self._timeout(30))
            gpu = check_gpu_headroom(self._telemetry(), config.limits.max_foreign_vram_mib,
                                     device_id=config.limits.gpu_device_id)
        except PreflightError as exc:
            raise _Stop("rejected", str(exc)) from exc
        preflight = {"image": self.image_evidence, "gpu": gpu, "registry_digest": registry_digest,
                     "help_sha256": caps.help_sha256, "server_build": caps.build}
        if self.container_mode:
            preflight["evaluator_image"] = evaluator_evidence
            self.image_evidence = {**self.image_evidence, "evaluator": evaluator_evidence}
        self.artifacts.write_json("preflight.json", preflight)

    def _hash(self) -> None:
        deadline = self.clock() + self._timeout(self.config.bounds.hash_seconds)
        try:
            self.model_evidence = {"pre": verify_model_asset(self.config.model, hasher=self.runner.hasher,
                                                             deadline=deadline, clock=self.clock)}
        except PreflightTimeout as exc:
            raise _Stop("timeout", str(exc)) from exc
        except PreflightError as exc:
            raise _Stop("rejected", str(exc)) from exc

    def _plan(self) -> None:
        if self.container_mode:
            # Re-issue the grant from the measured remainder (hashing usually finishes well inside its bound).
            self.grant = self._issue_grant(self._remaining(), self.clock() - self.started)
            if self.grant_shortfall:
                raise _Stop("timeout", "no evaluator grant remains after the hash stage")
            self.plan = build_compose_plan(self.config, self.run_dir, attempt_id=self.attempt_id,
                                           session_id=self.session_id, grant=self.grant)
        self.artifacts.write("plan/compose.json", self.plan.compose_json().encode("utf-8"))
        self.artifacts.write_json("plan/server-argv.json", {
            "entrypoint": list(self.config.inference_image.entrypoint), "argv": list(self.plan.server_argv),
            "environment": {}, "project": self.plan.project_name, "labels": dict(self.plan.labels)})
        if self.container_mode:
            config_bytes = (canonical_json(self.config.model_dump(mode="json")) + "\n").encode("utf-8")
            # Never the native grant: the evaluator does not need it and an older evaluator image refuses it.
            policy_bytes = (canonical_json(self._lock().evaluator_json()) + "\n").encode("utf-8")
            self.artifacts.write("plan/evaluator-config.json", config_bytes)
            self.artifacts.write("plan/runtime-policy.json", policy_bytes)
            self.artifacts.write_json("plan/evaluator-grant.json", self.grant.model_dump(mode="json"))
        resolved = self._call((*self.prefix, "config", "--format", "json"))
        if resolved.status != "completed" or resolved.returncode != 0:
            raise _Stop("failed", f"compose config rejected the plan ({resolved.status}): {_text(resolved)[:1000]}")
        self.artifacts.write("plan/compose-resolved.json", resolved.stdout)

    def _start(self) -> None:
        self.up_attempted = True
        up = self._call((*self.prefix, "up", "--detach", "--no-build", "--pull", "never", "inference"), 120)
        self.artifacts.write("logs/compose-up.log", up.stdout + up.stderr)
        if up.status != "completed" or up.returncode != 0:
            raise _Stop("timeout" if up.status == "timeout" else "failed", f"compose up failed ({up.status})")
        self.container_id = self._service_container("inference")
        if self.container_mode:
            return  # No published port: the evaluator reaches the internal network by service name.
        port = self._call((*self.prefix, "port", "inference", "8080"))
        match = re.fullmatch(r"127\.0\.0\.1:(\d{1,5})", port.stdout.decode("utf-8", "replace").strip())
        if port.status != "completed" or port.returncode != 0 or match is None:
            raise _Stop("failed", "inference port is not published on the loopback interface")
        self.base_url = f"http://127.0.0.1:{int(match[1])}"

    def _service_container(self, service: str) -> str:
        listing = self._call((*self.prefix, "ps", "--all", "--format", "json"))
        identifiers = [row.get("ID") for row in _json_rows(listing.stdout) if row.get("Service") == service]
        if (listing.status != "completed" or listing.returncode != 0 or len(identifiers) != 1
                or not re.fullmatch(RESOURCE_ID, str(identifiers[0]))):
            raise _Stop("failed", f"could not identify exactly one owned {service} container")
        return identifiers[0]

    def _container_state(self, timeout: int, container_id: str | None = None) -> tuple[str, str, str] | None:
        state = self.executor.run(("docker", "inspect", "--format", STATE_FORMAT, container_id or self.container_id),
                                  timeout_seconds=timeout, max_output_bytes=4096)
        parts = state.stdout.decode("utf-8", "replace").split()
        return tuple(parts) if state.status == "completed" and state.returncode == 0 and len(parts) == 3 else None

    def _fetch_log(self, timeout: int, container_id: str | None = None, max_output_bytes: int | None = None):
        return self.executor.run(("docker", "logs", container_id or self.container_id), timeout_seconds=timeout,
                                 max_output_bytes=self.config.limits.log_max_bytes if max_output_bytes is None
                                 else max_output_bytes)

    def _save_log(self, name: str, timeout: int, container_id: str | None = None, label: str = "inference",
                  max_output_bytes: int | None = None) -> str:
        logs = self._fetch_log(timeout, container_id, max_output_bytes)
        if logs.status == "output-limit":
            self.warnings.append(f"{label}_log_truncated:{name}")
        elif logs.status != "completed" or logs.returncode != 0:
            self.warnings.append(f"{label}_log_unavailable:{name}:{logs.status}")
        self.artifacts.write(name, logs.stdout + logs.stderr)
        return _text(logs)

    def _ready(self) -> None:
        bounds = self.config.bounds
        began, limit = self.clock(), self._timeout(bounds.startup_seconds)
        http = self.runner.http_factory(self.base_url, timeout=min(5., float(bounds.request_timeout_seconds)))
        while True:
            if self._remaining() <= 0 or self.clock() - began >= limit:
                raise _Stop("timeout", f"inference was not healthy within {limit}s")
            state = self._container_state(self._timeout(10))
            if state is not None and (state[0] in STOPPED_STATES or state[2] == "true"):
                tail = self._save_log("logs/startup.log", self._timeout(30))[-600:]
                raise _Stop("failed", f"inference container stopped during startup: status={state[0]} "
                                      f"exit={state[1]} oom={state[2]}; log tail: {tail}")
            remaining = min(self._remaining(), limit - (self.clock() - began))
            if remaining <= 0:
                raise _Stop("timeout", f"inference was not healthy within {limit}s")
            # UrllibHTTPTransport exposes this request timeout; shrinking it does not reset the phase timer.
            http.timeout = min(5., float(bounds.request_timeout_seconds), remaining)
            try:
                healthy = http.request_json("GET", "/health", deadline_monotonic=min(
                    self.work_deadline, self.phase_deadline or self.work_deadline, began + limit),
                    clock=self.clock).get("status") == "ok"
            except TransportDeadlineExceeded as exc:
                raise _Stop("timeout", f"startup health request deadline exhausted: {exc}") from exc
            except (BackendError, OSError, ValueError):
                healthy = False
            if self._remaining() <= 0 or self.clock() - began >= limit:
                raise _Stop("timeout", f"inference was not healthy within {limit}s")
            if healthy:
                self.ready_seconds = self.clock() - began
                return
            self.runner.sleep(min(bounds.readiness_poll_seconds, self._remaining(),
                                  limit - (self.clock() - began)))

    def _ready_from_logs(self) -> None:
        """Container mode: readiness is the inference container's own `model loaded` + `listening on` lines."""
        bounds = self.config.bounds
        began, limit = self.clock(), self._timeout(bounds.startup_seconds)

        def left() -> float:
            return min(self._remaining(), limit - (self.clock() - began))

        while True:
            # Every Docker call needs at least one whole second; less than that is the readiness timeout itself.
            if left() < 1:
                raise _Stop("timeout", f"inference did not log readiness within {limit}s")
            state = self._container_state(self._timeout(10))
            if state is not None and (state[0] in STOPPED_STATES or state[2] == "true"):
                tail = self._save_log("logs/startup.log", self._timeout(30))[-600:]
                raise _Stop("failed", f"inference container stopped during startup: status={state[0]} "
                                      f"exit={state[1]} oom={state[2]}; log tail: {tail}")
            if left() < 1:
                raise _Stop("timeout", f"inference did not log readiness within {limit}s")
            logs = self._fetch_log(self._timeout(30))
            parsed = parse_startup_log(_text(logs)) if logs.status in {"completed", "output-limit"} else {}
            if left() <= 0:
                raise _Stop("timeout", f"inference did not log readiness within {limit}s")
            if parsed.get("model_loaded_seconds") is not None and parsed.get("listening"):
                self.ready_seconds = self.clock() - began
                return
            self.runner.sleep(min(bounds.readiness_poll_seconds, left()))

    def _verify(self) -> None:
        self.startup = parse_startup_log(self._save_log("logs/startup.log", self._timeout(30)))
        self.load_seconds = self.startup.get("model_loaded_seconds")
        sample = self._telemetry()
        device = float(self.config.limits.gpu_device_id)
        gpu = next((row for row in sample.get("gpus") or [] if row.get("index") == device), None)
        if gpu and gpu.get("memory_used_mib") is not None:
            self.vram_after_load = gpu["memory_used_mib"]
            if gpu.get("memory_total_mib") and self.vram_after_load > VRAM_WARNING_FRACTION * gpu["memory_total_mib"]:
                self.warnings.append("vram_above_97_percent_of_total_possible_shared_memory_fallback")
        else:
            self.warnings.append("vram_after_load_unavailable")
        self.artifacts.write_json("verify.json", {"startup": self.startup, "ready_seconds": self.ready_seconds,
                                                  "telemetry": sample})

    def _evaluation_context_extras(self) -> dict:
        """Extra evaluation-context fields a runtime supplies (the native runtime: why coding cannot execute)."""
        return {}

    def _broker(self) -> None:
        if self.container_mode:
            for name in ("requests", "results"):
                (self.run_dir / "spool" / name).mkdir(parents=True, exist_ok=True)
        self.broker = self.runner.broker_factory(self.run_dir, self.config, self._lock(),
                                                 self.artifacts.scoped("broker"))
        for method in ("tick", "cancel_all", "direct_client"):
            if not callable(getattr(self.broker, method, None)):
                raise _Stop("failed", f"broker factory returned an object without {method}()")

    def _evaluate(self) -> None:
        deadline = self.clock() + self._timeout(self.config.bounds.evaluation_seconds)
        lock = self.runner.session_lock or SessionLock.read(self.runner.policy_path)
        context = SimpleNamespace(base_url=self.base_url, artifacts_dir=self.run_dir / "evaluator",
                                  deadline_monotonic=deadline, session_lock=lock, clock=self.clock,
                                  policy_path=self.runner.policy_path,
                                  artifacts=self.artifacts.scoped("evaluator"),  # One shared byte budget.
                                  coding_hook=self.runner.coding_hook,
                                  coding_client=self.broker.direct_client() if self.broker is not None else None,
                                  **self._evaluation_context_extras())
        context.artifacts_dir.mkdir(parents=True, exist_ok=True)
        error = "evaluator did not return a dictionary"
        try:
            evaluation = self.runner.evaluator(self.config, context)
            if not isinstance(evaluation, dict):
                raise TypeError(error)
            self.evaluation = evaluation
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            saved = self.evaluation if self.evaluation is not None else {"error": error}
            self._guard("evaluation_not_persisted", lambda: self.artifacts.write_json("evaluation.json", saved))
        self.warnings.extend(f"evaluation_error: {json.dumps(item, default=str)[:500]}"
                             for item in self.evaluation.get("errors") or [])
        if self.evaluation.get("abort_reason"):
            raise _Stop("failed", f"evaluator aborted: {self.evaluation['abort_reason']}")
        if self.clock() > deadline:
            raise _Stop("timeout", "evaluation exceeded its stage bound")

    # ---- container-mode evaluator ---------------------------------------------------------------------------
    def _evaluate_container(self) -> None:
        try:
            self._run_evaluator_container()
        except BaseException:
            self._finish_child(failing=True)
            raise
        self._finish_child(failing=False)

    def _run_evaluator_container(self) -> None:
        config, grant, bounds = self.config, self.grant, self.config.bounds
        self.artifacts.reserve_child("evaluator", config.limits.evaluator_artifact_bytes)
        self.child_reserved = True
        (self.run_dir / "evaluator").mkdir(parents=True, exist_ok=True)
        launched = self.clock()
        watchdog = min(self.work_deadline, launched + grant.grant_seconds + grant.watchdog_slack_seconds)
        self.artifacts.write_json("plan/evaluator-launch.json", {
            "launched_offset_seconds": launched - self.started, "grant_seconds": grant.grant_seconds,
            "watchdog_offset_seconds": watchdog - self.started, "watchdog_slack_seconds": grant.watchdog_slack_seconds})
        up = self._call((*self.prefix, "up", "--detach", "--no-build", "--pull", "never", "evaluator"), 120)
        self.artifacts.write("logs/compose-up-evaluator.log", up.stdout + up.stderr)
        if up.status != "completed" or up.returncode != 0:
            raise _Stop("timeout" if up.status == "timeout" else "failed", f"compose up evaluator failed ({up.status})")
        self.evaluator_id = self._service_container("evaluator")
        state, ticks = None, []
        while True:
            # The watchdog is judged before the inspect call so its verdict, not the generic phase floor, is
            # what a candidate whose watchdog coincides with the work deadline reports.
            if self.clock() >= watchdog:
                raise _Stop("timeout", f"evaluator exceeded its {grant.grant_seconds}s grant plus "
                                       f"{grant.watchdog_slack_seconds}s watchdog slack; torn down")
            state = self.evaluator_state = self._container_state(self._timeout(10), self.evaluator_id)
            if state is not None and (state[0] in STOPPED_STATES or state[2] == "true"):
                break
            if self.broker is not None:
                tick = self.broker.tick(max(0.0, min(watchdog - self.clock(), BROKER_TICK_MAX_SECONDS)))
                ticks.append(tick)
                if not isinstance(tick, dict) or tick.get("abort_campaign"):
                    raise _Stop("failed", "coding broker requested a campaign abort during the evaluation")
            if self.clock() >= watchdog:
                raise _Stop("timeout", f"evaluator exceeded its {grant.grant_seconds}s grant plus "
                                       f"{grant.watchdog_slack_seconds}s watchdog slack; torn down")
            self.runner.sleep(max(0.0, min(bounds.readiness_poll_seconds, watchdog - self.clock())))
        if ticks:
            self._guard("broker_ticks_not_persisted", lambda: self.artifacts.write_json("broker/ticks.json", ticks))
        exit_code = int(state[1]) if re.fullmatch(r"-?\d+", state[1]) else None
        log_tail = self._capture_evaluator_log()[-EVALUATOR_LOG_TAIL:]
        if state[2] == "true":
            raise _Stop("failed", f"evaluator_oom_killed (exit={state[1]}); log tail: {log_tail}")
        if exit_code in (0, 3):
            self._ingest_evaluation()
        elif exit_code == 2:
            raise _Stop("failed", f"evaluator refused config/policy (exit 2); log tail: {log_tail}")
        else:
            raise _Stop("failed", f"evaluator exited {state[1]}; log tail: {log_tail}")

    def _capture_evaluator_log(self) -> str:
        """Bounded, once, never fatal: the fetch is capped by the parent's remaining data budget so an oversized
        child log is truncated and flagged instead of exhausting the budget and failing a valid candidate."""
        if self.evaluator_id is None or self.evaluator_log_attempted:
            return ""
        self.evaluator_log_attempted = True
        timeout = max(1, int(min(30, self.work_deadline - self.clock(), self.config.bounds.cleanup_reserve_seconds)))
        cap = max(1, min(self.config.limits.log_max_bytes, self.artifacts.remaining_bytes))
        try:
            return self._save_log("logs/evaluator.log", timeout, self.evaluator_id, label="evaluator",
                                  max_output_bytes=cap)
        except Exception as exc:
            self.warnings.append(f"evaluator_log_not_captured: {type(exc).__name__}: {exc}")
            return ""

    def _ingest_evaluation(self) -> None:
        from ..container_eval import read_evaluation
        from ..evaluations.selection import expected_task_rows
        error = "evaluation.json was not ingested"
        try:
            self.evaluation = read_evaluation(self.run_dir / "evaluator" / "evaluation.json",
                                              max_bytes=self.config.limits.evaluator_artifact_bytes,
                                              expected_rows=expected_task_rows(self.config.benchmarks))
        except ValueError as exc:
            error = f"evaluation.json rejected: {exc}"
            raise _Stop("failed", error) from exc
        finally:
            saved = self.evaluation if self.evaluation is not None else {"error": error}
            self._guard("evaluation_not_persisted", lambda: self.artifacts.write_json("evaluation.json", saved))
        self.warnings.extend(f"evaluation_error: {json.dumps(item, default=str)[:500]}"
                             for item in self.evaluation.get("errors") or [])
        if self.evaluation.get("abort_reason"):
            raise _Stop("failed", f"evaluator aborted: {self.evaluation['abort_reason']}")

    def _finish_child(self, *, failing: bool) -> None:
        """Always capture the evaluator log. Reconcile the child's tree against its allocation now only when the
        child was observed stopped; a child that may still be running (watchdog overrun, broker abort, interrupt)
        is hashed after cleanup has proven every owned container absent, never while it can still write."""
        self._capture_evaluator_log()
        if not self.child_reserved:
            return
        if not self._evaluator_stopped():
            self.child_adoption_pending = True
            return
        reasons = self._adopt_child(deferred=False)
        if failing:
            self.deferred_reasons.extend(reasons)
        elif reasons:
            self.deferred_reasons.extend(reasons[1:])
            raise _Stop("failed", reasons[0])

    def _evaluator_stopped(self) -> bool:
        state = self.evaluator_state
        return state is not None and state[0] in STOPPED_STATES

    def _adopt_child(self, *, deferred: bool) -> list[str]:
        """Index what the child wrote; returns the candidate failure reasons (never raises, never a warning)."""
        try:
            self.adoption = self.artifacts.adopt_child("evaluator")
        except Exception as exc:
            return [f"evaluator_artifacts_not_adopted: {type(exc).__name__}: {exc}"]
        record = {**self.adoption, "adopted_after_cleanup": deferred}
        self._guard("evaluator_adoption_not_persisted",
                    lambda: self.artifacts.write_json("plan/evaluator-adoption.json", record))
        reasons = []
        if self.adoption["over_allocation"]:
            reasons.append(f"evaluator_artifacts_over_allocation: {self.adoption['bytes']} bytes written against "
                           f"{self.config.limits.evaluator_artifact_bytes} allocated; "
                           f"unindexed={self.adoption['unindexed']}")
        if self.adoption["errors"]:
            errors = self.adoption["errors"]
            reasons.append(f"evaluator_artifacts_not_adopted: {len(errors)} child file(s) could not be indexed: "
                           + "; ".join(str(item)[:300] for item in errors[:4]))
        return reasons

    def _adopt_deferred_child(self) -> None:
        if not self.child_adoption_pending:
            return
        self.child_adoption_pending, self.current = False, "evaluator"  # evaluator-stage evidence, after cleanup
        if not self.containers_absent:
            self.reasons.append("evaluator_artifacts_not_adopted: the evaluator container was not verified "
                                "stopped; its artifact tree stays unindexed")
            return
        for reason in self._adopt_child(deferred=True):
            self._fail("failed", reason)  # the stage that failed first keeps the state; this is appended

    def _settings_evidence(self) -> None:
        evaluation = self.evaluation or {}
        self.settings = build_settings_evidence(
            self.config.engine, self.plan.server_argv, evaluation.get("readback"), self.startup,
            evaluation.get("overflow_probe"), alias=self.config.alias(),
            build_info=self.config.build_info)
        self.artifacts.write_json("settings-evidence.json", {
            "settings": [row.model_dump(mode="json") for row in self.settings],
            "unverified_required": unverified_required(self.settings)})
        mismatched = [row.control for row in self.settings if row.status == "mismatch"]
        if mismatched:
            raise _Stop("failed", "effective settings differ from the request: " + ", ".join(mismatched))

    # ---- cleanup -----------------------------------------------------------------------------------------
    def _cleanup(self) -> None:
        if not self.up_attempted:
            self.cleanup = CleanupEvidence(attempted=False, verified=True)
            if self.model_evidence:
                self._reverify_model()
            return
        deadline = self.clock() + self.config.bounds.cleanup_reserve_seconds
        evidence = {"attempted": True, "compose_down_returncode": None, "containers_remaining": (),
                    "networks_remaining": (), "verified": False, "error": None}
        self.cleanup = CleanupEvidence(**evidence)

        def run(argv, bound=30, max_output_bytes=65536):
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError("cleanup reserve exhausted")
            return self.executor.run(tuple(argv), timeout_seconds=max(1, int(min(bound, remaining))),
                                     max_output_bytes=max_output_bytes)

        def owned(kind) -> tuple[str, ...]:
            listing = run(("docker", *kind, "--filter", project_filter(self.plan.project_name), "--format", "{{.ID}}"))
            identifiers = tuple(listing.stdout.decode("utf-8", "replace").split())
            if (listing.status != "completed" or listing.returncode != 0
                    or not all(re.fullmatch(RESOURCE_ID, item) for item in identifiers)):
                raise RuntimeError(f"owned-resource query failed ({listing.status})")
            return identifiers

        broker_error = self._cancel_broker() if self.broker is not None else None
        try:
            if self.container_id:
                self._guard("final_state_unavailable", lambda: self._final_state(deadline))
            down = run((*self.prefix, "down", "--volumes", "--timeout", "15"), 45, 1_048_576)
            evidence["compose_down_returncode"] = down.returncode
            self._guard("compose_down_log_not_persisted",
                        lambda: self.artifacts.write("logs/compose-down.log", down.stdout + down.stderr))
            containers, networks = owned(("ps", "--all")), owned(("network", "ls"))
            if containers or networks:
                # Exact IDs returned by this attempt's unique project label; never prune or match by name.
                for identifier in containers:
                    run(("docker", "rm", "-f", identifier))
                for identifier in networks:
                    run(("docker", "network", "rm", identifier))
                containers, networks = owned(("ps", "--all")), owned(("network", "ls"))
            self.containers_absent = not containers  # exact owned-label listing: the only proof a child stopped
            evidence.update(containers_remaining=containers, networks_remaining=networks,
                            verified=not containers and not networks)
            if not evidence["verified"]:
                evidence["error"] = "owned containers or networks remain after removal"
        except Exception as exc:
            evidence.update(verified=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if broker_error and evidence["verified"]:
                evidence.update(verified=False, error=broker_error)
            self.cleanup = CleanupEvidence(**evidence)
        self._reverify_model()
        try:
            self.artifacts.write_json("cleanup.json", {"cleanup": self.cleanup.model_dump(mode="json"),
                                                       "telemetry": self._telemetry(deadline)})
        except Exception as exc:
            self.warnings.append(f"cleanup_telemetry_unavailable: {type(exc).__name__}: {exc}")

    # An interruption (not an error) that reaches the cleanup stage after this attempt's cleanup already recorded
    # verified absence leaves that evidence standing when this is True, and an unverified one keeps its own reason
    # (`_cleanup_error_after`). The container runtime keeps its original rule -- anything escaping cleanup makes it
    # uncertain, with that exception as the reason -- so its results stay exactly what they were.
    KEEPS_PROVEN_CLEANUP_ON_INTERRUPT = False

    def _keeps_proven_cleanup(self, exc: BaseException, before: CleanupEvidence) -> bool:
        """Whether `exc`, escaping the cleanup stage, arrived after this stage had already recorded verified absence.

        Only evidence the stage itself recorded counts: `before` is what `self.cleanup` held when the stage began (the
        constructor's placeholder, which proves nothing). A later interrupt cannot un-stop a server whose absence was
        already shown, so discarding that proof would retain the lease and block a resume for nothing; an interrupt
        that arrives before the proof exists still leaves the cleanup uncertain."""
        return (self.KEEPS_PROVEN_CLEANUP_ON_INTERRUPT and not isinstance(exc, Exception)
                and self.cleanup is not before and self.cleanup.verified)

    def _cleanup_error_after(self, exc: BaseException, before: CleanupEvidence) -> str:
        """The error an uncertain cleanup records when `exc` escaped the stage.

        The container runtime records `exc` alone, as it always has. Where an interrupt is held until the absence
        proof is recorded (the same flag), a proof that failed on its own -- a process or the port still there --
        already says why, and what escaped after it did not cause that: the stage's own reason comes first and `exc`
        after it, so a Ctrl-C never replaces the evidence of what is still running."""
        error = f"{type(exc).__name__}: {exc}"
        if self.KEEPS_PROVEN_CLEANUP_ON_INTERRUPT and self.cleanup is not before and self.cleanup.error:
            return f"{self.cleanup.error}; then {error}"
        return error

    def _cancel_broker(self) -> str | None:
        """Broker workers are owned resources too: unverified cancellation makes the whole cleanup uncertain."""
        try:
            cancelled = self.broker.cancel_all()
            self._guard("broker_cancel_not_persisted",
                        lambda: self.artifacts.write_json("broker/cancel-all.json", cancelled))
            if not isinstance(cancelled, dict) or cancelled.get("cleanup_verified") is not True:
                return "broker worker cleanup unverified"
            if getattr(self.broker, "abort_campaign", False):
                return "coding broker requested a campaign abort"
            return None
        except Exception as exc:
            return f"broker cancel_all failed: {type(exc).__name__}: {exc}"

    def _final_state(self, deadline: float) -> None:
        state = self._container_state(max(1, int(min(10, deadline - self.clock()))))
        if state and state[2] == "true":
            self._fail("failed", "inference_oom_killed")
        self._save_log("logs/inference-final.log", max(1, int(min(30, deadline - self.clock()))))

    def _reverify_model(self) -> None:
        post = reverify_identity(self.config.model, self.model_evidence.get("pre", {}))
        self.model_evidence = {**self.model_evidence, "post": post, "unchanged": post["unchanged"]}
        if not post["unchanged"]:
            self._fail("failed", "model_asset_changed_during_run")

    # ---- reporting ---------------------------------------------------------------------------------------
    def _report(self) -> dict:
        began = self.clock()
        try:
            from .report import write_candidate_reports
            reports = write_candidate_reports(self.config, self._result({}), self.evaluation or {},
                                              self.runner.policy, self.run_dir, artifacts=self.artifacts)
            self._record("report", "ok", began)
            return reports
        except Exception as exc:
            self.warnings.append(f"report_failed: {type(exc).__name__}: {exc}")
            self._record("report", "failed", began, f"{type(exc).__name__}: {exc}")
            return {}

    def _result(self, reports: dict) -> ContainerRunResult:
        from .report import summarize_evaluation
        try:
            summary = summarize_evaluation(self.config, self.evaluation or {}, self.runner.policy)
        except Exception as exc:  # Malformed evaluator output must not lose the terminal record.
            reason = f"evaluation_summary_failed: {type(exc).__name__}: {exc}"
            summary = {"speed": {"attempted": 0, "qualifies": False, "reasons": [reason]}, "quality": {},
                       "samples_total": 0}
            if reason not in self.warnings:
                self.warnings.append(reason)
        if reports:
            for name in ("reports", "evaluator"):
                if self.container_mode:
                    # adopt_child may have left over-cap child files unindexed on purpose; re-adoption failing
                    # is a recorded warning, never a lost terminal record. A child tree that was never adopted
                    # (child not proven stopped, or adoption itself failed) is never hashed here either.
                    # Host-process mode is unchanged.
                    if name == "evaluator" and self.adoption is None:
                        continue
                    self._guard(f"adopt_{name}", lambda name=name: self.artifacts.adopt_tree(name))
                else:
                    self.artifacts.adopt_tree(name)
        elapsed = max(0., self.clock() - self.started)
        completed = self.state == "completed"
        container_ids = tuple(item for item in (self.container_id, self.evaluator_id) if item)
        return ContainerRunResult(
            session_id=self.session_id, attempt_id=self.attempt_id, config_fingerprint=self.config.fingerprint(),
            project_name=self.plan.project_name, state=self.state, synthetic=self.runner.synthetic,
            failure_stage=self.failure_stage, failure_reasons=tuple(self.reasons), warnings=tuple(self.warnings),
            started_utc=self.started_utc, finished_utc=utc_now(), elapsed_seconds=elapsed,
            budget_charged_seconds=elapsed, stages=tuple(self.stages), container_ids=container_ids,
            image_evidence=self.image_evidence, model_evidence=self.model_evidence,
            settings=tuple(self.settings),
            effective_settings_verified=completed and settings_verified(self.settings),
            actual_context_verified=completed and (self.evaluation or {}).get("actual_context_verified") is True,
            speed=summary["speed"], quality=summary["quality"], samples_total=summary["samples_total"],
            load_seconds=self.load_seconds, vram_used_mib_after_load=self.vram_after_load, cleanup=self.cleanup,
            abort_campaign=self.state == "cleanup-uncertain",
            artifacts=tuple(ArtifactEntry(**entry) for entry in self.artifacts.index()), reports=reports)


def _json_rows(raw: bytes) -> list[dict]:
    """`compose ps --format json` prints either one array or one object per line, by Compose version."""
    text = raw.decode("utf-8", "replace").strip()
    try:
        parsed = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line]
    except ValueError:
        return []
    return [row for row in parsed if isinstance(row, dict)]
