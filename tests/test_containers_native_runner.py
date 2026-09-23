"""The metal-native runner end to end against fakes: stage order, evidence, admission, the memory watchdog,
cancellation and cleanup proof. The server log is the real b11011 Metal startup log captured on an M1
(`tests/data/startup-b11011-metal-qwen3-1.7b.log`); no process, GPU, model, Docker or network is touched. The
admission identity tests hash a synthetic llama.cpp release directory (tiny Mach-O images that are read, never run)
and script the executable's --version/--help answers; one cleanup test holds a loopback listener on the planned
port."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

from llmbench.coding.sandbox import WorkerResult
from llmbench.config import canonical_json
from llmbench.containers.capabilities import help_sha256
from llmbench.containers.config import BrokerSettings, ContainerRunConfig
from llmbench.containers.native import NativeRunner, _Output, _sigint_deferred
from llmbench.containers.native_prep import hash_native_server
from llmbench.locks import describe_lock
from llmbench.measurement import SpeedObservation
from llmbench.safety import SessionLock
from test_native_prep import HELP, VERSION, macho, release

DATA = Path(__file__).parent / "data"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.json"
METAL_LOG = (DATA / "startup-b11011-metal-qwen3-1.7b.log").read_text(encoding="utf-8")
MIB = 1024 * 1024
NATIVE_LOCK = SessionLock(allow_model_operations=True, allow_inference=True, allow_container_execution=False,
                          allow_native_execution=True)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeServer:
    """Implements the NativeServerProcess surface the runner uses; scripted per test."""

    def __init__(self, harness, *, executable, argv, cwd, env, log_sink, log_path, log_cap):
        self.harness, self.argv, self.env, self.log_sink, self.log_path = harness, list(argv), env, log_sink, log_path
        self.port = int(self.argv[self.argv.index("--port") + 1])
        self.pid, self.returncode, self.signals_sent = None, None, []  # like Popen: no pid until it started
        self.log_error, self.dropped_bytes, self.log_bytes = None, 0, 0
        self.started = self.terminated = self.stopped = False
        harness.servers.append(self)

    def start(self):
        if self.harness.start_error:
            raise self.harness.start_error
        self.started, self.pid = True, 4242
        # The captured log's port becomes the planned one; its address is whatever the log says.
        text = self.harness.log.replace(":50363", f":{self.port}")
        self.log_sink.write(text.encode("utf-8"))
        self.log_sink.close()
        self.log_bytes = len(text)
        if self.harness.log_budget_error:  # what the drain thread records when the artifact budget refuses a write
            self.log_error, self.dropped_bytes = self.harness.log_budget_error, 4096
        if self.harness.hold_port:  # a server that keeps serving its port whatever the stop did
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", self.port))
            listener.listen(1)
            self.harness.listeners.append(listener)
        if self.harness.exit_during_startup is not None:
            self.returncode = self.harness.exit_during_startup

    def state(self):
        return ("running", "", "false") if self.returncode is None else ("exited", str(self.returncode), "false")

    def read_log(self, max_bytes):
        data = self.log_path.read_bytes() if self.log_path.exists() else b""
        return _Output("completed" if len(data) <= max_bytes else "output-limit", 0, data[:max_bytes])

    def describe_exit(self):
        if self.returncode is None:
            return None
        return f"returncode={self.returncode}" if self.returncode >= 0 else "signal=SIGKILL"

    def unexpected_kill(self):
        return self.returncode == -9 and "SIGKILL" not in self.signals_sent

    def terminate(self):
        self.terminated = True
        self.signals_sent.append("SIGTERM")

    def stop(self, grace, deadline, clock=None):
        self.harness.stops.append(grace)
        self.harness.events.append("server-stop")
        if self.harness.stop_raises is not None:
            raise self.harness.stop_raises
        if self.returncode is None and self.started:
            self.signals_sent.append("SIGTERM")
            if self.harness.sigint_during_stop:  # the supervisor's forwarded SIGINT, between SIGTERM and the exit
                signal.raise_signal(signal.SIGINT)
            self.returncode = 0
        if self.harness.died_by_sigkill:
            self.returncode = -9
        self.stopped = True

    def remaining(self, marker):
        self.harness.markers.append(marker)
        return list(self.harness.leftovers)


class FakeBroker:
    def __init__(self, harness):
        self.harness = harness

    def tick(self, seconds):
        return {}

    def direct_client(self):
        return object()

    def cancel_all(self):
        self.harness.events.append("broker-cancel")
        return {"cleanup_verified": True}


def blocked_sandbox(**kwargs):
    return {"status": "blocked", "reason": "docker CLI not found"}


class FakeHTTP:
    def __call__(self, base_url, *, timeout):
        self.base_url = base_url
        return self

    def request_json(self, method, path, payload=None, **kwargs):
        assert (method, path) == ("GET", "/health")
        return {"status": "ok"}


def observation():
    return SpeedObservation(status="completed", finish_reason="length", output_tokens=512,
                            native_generation_seconds=16.0, elapsed_seconds=17.0, first_event_seconds=0.6,
                            content_event_times=tuple(0.6 + 0.03 * index for index in range(100)),
                            requested_output_tokens=512, input_tokens=4096, expected_input_tokens=4096,
                            accepted_tokens_verified=True, native_timing_source="llama.cpp.timings")


def unified(*, available=6 * 1024 * MIB, pressure=1, gpu=3, footprint=1905 * MIB, swap=6800 * MIB, pid=None):
    return {"kind": "apple-unified", "monotonic_seconds": time.monotonic(), "host_memory_total_bytes": 8192 * MIB,
            "host_memory_available_bytes": available, "host_memory_used_bytes": 8192 * MIB - available,
            "swap_total_bytes": 7168 * MIB, "swap_used_bytes": swap, "swapins": 10, "swapouts": 20,
            "memory_pressure_level": pressure,
            "process": None if pid is None else {"pid": pid, "alive": True, "phys_footprint_bytes": footprint,
                                                 "rss_bytes": footprint // 2},
            "gpu": {"device_utilization_percent": gpu, "in_use_system_memory_bytes": 1700 * MIB},
            "power": {"power_source": "battery", "battery_percent": 70, "charging": False}, "errors": []}


class Harness:
    def __init__(self, tmp_path, *, lock=NATIVE_LOCK, config_changes=None, sampler=None, evaluator=None,
                 log=METAL_LOG, limits=None, native_server=None, fake_identity=True, sandbox_probe=blocked_sandbox):
        tmp_path.mkdir(parents=True, exist_ok=True)
        model = tmp_path / "model.gguf"
        model.write_bytes(b"GGUF" * 64)
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw.pop("inference_image")
        raw["runtime"] = "metal-native"
        raw["native_server"] = native_server or {"executable": str(tmp_path / "llama-b11011" / "llama-server"),
                                                 "executable_sha256": "a" * 64, "libraries_sha256": "b" * 64,
                                                 "build_info": "b11011-aa39d7a3e", "help_sha256": "c" * 64}
        raw["native_limits"] = {"sample_interval_seconds": 0.2, **(limits or {})}
        raw["assets"][0].update(host_path=str(model), size_bytes=256,
                                sha256=hashlib.sha256(b"GGUF" * 64).hexdigest())
        raw["engine"]["ctx_size"], raw["requested_input_tokens"] = 4864, 4096
        for path, value in (config_changes or {}).items():
            target = raw
            *parents, leaf = path.split("__")
            for parent in parents:
                target = target[int(parent) if parent.isdigit() else parent]
            target[leaf] = value
        self.config = ContainerRunConfig.model_validate_json(canonical_json(raw))
        self.clock, self.http, self.contexts, self.servers = FakeClock(), FakeHTTP(), [], []
        self.stops, self.markers, self.leftovers, self.log = [], [], [], log
        self.start_error = self.exit_during_startup = self.stop_raises = self.log_budget_error = None
        self.died_by_sigkill = self.sigint_during_stop = self.hold_port = False
        self.events, self.listeners, self.leases = [], [], []
        self.identity_calls = []
        self.sampler = sampler or (lambda pid=None, **kwargs: unified(pid=pid))
        self.evaluator = evaluator or self.evaluate
        self.lease_path = tmp_path / "gpu.lock"
        self.runner = NativeRunner(
            policy_path=tmp_path / "no-policy.json", session_lock=lock, lease_path=self.lease_path,
            http_factory=self.http, evaluator=self.call_evaluator, telemetry=self.sample,
            registry_validator=lambda config: "f" * 64, clock=self.clock, sleep=self.clock.sleep,
            process_factory=lambda **kwargs: FakeServer(self, **kwargs),
            identity_check=self.identity if fake_identity else None, sandbox_probe=sandbox_probe)
        self.output = tmp_path / "out"

    def sample(self, pid=None, **kwargs):
        return self.sampler(pid, **kwargs)

    def identity(self, config, argv):
        self.identity_calls.append(list(argv))
        return {"executable": config.native_server.executable, "executable_sha256": "a" * 64,
                "libraries_sha256": "b" * 64, "build_info": "b11011-aa39d7a3e", "help_sha256": "c" * 64}

    def call_evaluator(self, config, context):
        assert self.lease_path.exists()  # the GPU lease is held for the whole evaluation
        self.leases.append((self.lease_path.read_bytes(), describe_lock(self.lease_path)))
        self.contexts.append(context)
        return self.evaluator(config, context)

    def evaluate(self, config, context):
        props = json.loads((DATA / "props-b11011-metal.json").read_text(encoding="utf-8"))
        props = props.get("body", props)
        props["model_alias"], props["model_path"] = config.alias(), config.server_model_path
        props["default_generation_settings"]["n_ctx"] = config.engine.ctx_size
        self.clock.now += 60
        return {"samples": [{"task_id": "tools/nested-exact-v1", "category": "tools", "score": 1.0,
                             "status": "completed"}],
                "speed_observations": [observation() for _ in range(config.speed.repetitions)],
                "readback": {"props": props, "models": {}, "slots": [{"id": 0, "n_ctx": config.engine.ctx_size}]},
                "count_route": {"exact": True}, "overflow_probe": {"passed": True},
                "actual_context_verified": True, "abort_reason": None, "errors": []}

    def run(self, **options):
        return self.runner.run(self.config, self.output, **options)

    def saved(self, name="result.json"):
        return json.loads((self.output / name).read_text(encoding="utf-8"))


def test_happy_path_is_the_container_stage_machine_with_native_evidence(tmp_path):
    harness = Harness(tmp_path)
    result = harness.run()
    assert result.state == "completed", result.failure_reasons
    assert result.synthetic is True and result.runtime == "metal-native"
    assert [stage.name for stage in result.stages] == ["admit", "hash", "plan", "start", "ready", "verify",
                                                       "quality", "verify", "cleanup", "report"]
    server = harness.servers[0]
    assert server.argv[server.argv.index("--host") + 1] == "127.0.0.1"  # never every interface on a laptop
    assert server.argv[server.argv.index("--model") + 1] == harness.config.model.host_path
    assert not any(key.startswith(("LLAMA_", "GGML_")) for key in server.env)
    assert harness.http.base_url == f"http://127.0.0.1:{server.port}" == harness.contexts[0].base_url
    assert result.container_ids == (f"pid:{server.pid}",)
    # Unified memory is reported as what it is; nothing lands in the VRAM field.
    assert result.vram_used_mib_after_load is None
    memory = result.memory
    assert memory["kind"] == "apple-unified" and "None of these is VRAM" in memory["note"]
    assert memory["after_load"]["server_phys_footprint_mib"] == pytest.approx(1905.0)
    assert memory["server_log"]["metal_budget_mib"] == 5461
    assert memory["admission"] and "during_evaluation" in memory
    assert result.cleanup.verified and result.cleanup.server_exit == "returncode=0" and not result.abort_campaign
    assert harness.markers == [("--alias", harness.config.alias(), "--port", str(server.port))]
    assert not harness.lease_path.exists()
    statuses = {row.control: row.status for row in result.settings}
    assert statuses["n_gpu_layers"] == "verified-log" and statuses["kv_offload"] == "verified-log"
    assert statuses["build_info"] == "verified-api" and "mismatch" not in statuses.values()
    saved = harness.saved()
    assert saved["runtime"] == "metal-native" and saved["memory"]["kind"] == "apple-unified"
    launch = harness.saved("plan/native-launch.json")
    assert launch["host"] == "127.0.0.1" and launch["port"] == server.port
    assert launch["labels"]["llmbench.attempt"] == result.attempt_id
    names = {entry.path for entry in result.artifacts}
    assert {"config.json", "preflight.json", "plan/server-argv.json", "plan/native-launch.json", "logs/server.log",
            "logs/startup.log", "verify.json", "evaluation.json", "settings-evidence.json", "cleanup.json",
            "stages.jsonl"} <= names
    assert "plan/compose.json" not in names


def test_nvidia_configs_are_refused_by_the_native_runner(tmp_path):
    harness = Harness(tmp_path)
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="metal-native candidates only"):
        harness.runner.run(ContainerRunConfig.model_validate_json(json.dumps(raw)), tmp_path / "x")


def test_admission_needs_the_native_permission_and_starts_nothing_without_it(tmp_path):
    harness = Harness(tmp_path, lock=SessionLock(True, True, True))  # a container-only policy
    result = harness.run()
    assert result.state == "rejected" and "native forbidden" in result.failure_reasons[0]
    assert harness.servers == [] and harness.identity_calls == [] and not harness.lease_path.exists()


def test_an_unsupported_setting_is_refused_by_name(tmp_path):
    harness = Harness(tmp_path, config_changes={"limits__gpu_device_id": "1"})
    result = harness.run()
    assert result.state == "rejected" and "gpu_device_id" in result.failure_reasons[0]
    assert harness.servers == []


@pytest.mark.parametrize("sample, reason", [
    (dict(available=512 * MIB), "available"),
    (dict(pressure=4), "pressure"),
    (dict(gpu=97), "utili"),
])
def test_unified_memory_admission_is_a_conflict_to_report_never_a_process_to_stop(tmp_path, sample, reason):
    harness = Harness(tmp_path, sampler=lambda pid=None, **kwargs: unified(pid=pid, **sample))
    result = harness.run()
    assert result.state == "rejected" and reason in result.failure_reasons[0].lower()
    assert harness.servers == [] and not harness.lease_path.exists()


def test_the_watchdog_stops_a_server_whose_footprint_passes_the_metal_budget(tmp_path):
    over = {"n": 0}

    def sampler(pid=None, **kwargs):
        over["n"] += pid is not None
        return unified(pid=pid, footprint=(6000 if over["n"] > 2 else 1905) * MIB)

    def slow_evaluator(config, context):
        deadline = time.monotonic() + 10
        while not harness.servers[0].terminated and time.monotonic() < deadline:
            time.sleep(0.05)
        return {"samples": [], "speed_observations": [], "readback": {}, "abort_reason": "server went away",
                "errors": [], "actual_context_verified": False}

    harness = Harness(tmp_path, sampler=sampler, evaluator=slow_evaluator)
    result = harness.run()
    assert harness.servers[0].terminated
    assert result.state == "failed" and result.failure_stage == "quality"
    assert "native_memory_watchdog" in result.failure_reasons[0] and "footprint" in result.failure_reasons[0]
    assert result.cleanup.verified and not harness.lease_path.exists()
    assert result.memory["during_evaluation"]["violations"]


def test_leftover_owned_processes_make_cleanup_uncertain_and_keep_the_lease(tmp_path):
    harness = Harness(tmp_path)
    harness.leftovers = ["pid:4242"]
    result = harness.run()
    assert result.state == "cleanup-uncertain" and result.abort_campaign
    assert result.cleanup.processes_remaining == ("pid:4242",) and not result.cleanup.verified
    assert result.cleanup.lease_retained and harness.lease_path.exists()


def test_an_interrupt_during_the_evaluation_stops_the_server_and_is_reraised(tmp_path):
    def interrupted(config, context):
        raise KeyboardInterrupt()

    harness = Harness(tmp_path, evaluator=interrupted)
    with pytest.raises(KeyboardInterrupt):
        harness.run()
    saved = harness.saved()
    assert saved["state"] == "cancelled" and saved["cleanup"]["verified"] is True
    assert harness.stops and not harness.lease_path.exists()


def test_a_server_that_exits_during_startup_fails_with_its_log(tmp_path):
    harness = Harness(tmp_path)
    harness.exit_during_startup = 1
    result = harness.run()
    assert result.state == "failed" and result.failure_stage == "ready"
    assert "stopped during startup" in result.failure_reasons[0] and "exit=1" in result.failure_reasons[0]


def test_a_sigkill_the_harness_did_not_send_is_named_but_not_overclaimed(tmp_path):
    harness = Harness(tmp_path)
    harness.died_by_sigkill = True
    result = harness.run()
    assert result.state == "failed"
    assert any("inference_killed_by_sigkill" in reason and "not proven" in reason
               for reason in result.failure_reasons)


def test_a_blocked_sandbox_reaches_the_evaluator_as_a_reason_not_a_broker(tmp_path):
    # Coding is configured and containers are authorized, but the probe finds no Docker: the candidate still runs
    # natively and the coding suites carry the reason instead of a broker.
    harness = Harness(tmp_path, lock=SessionLock(True, True, True, True))
    harness.runner.broker_factory = lambda *args, **kwargs: pytest.fail("no broker without a sandbox")
    harness.config = harness.config.model_copy(update={"broker": __import__(
        "llmbench.containers.config", fromlist=["BrokerSettings"]).BrokerSettings()})
    result = harness.run()
    assert result.state == "completed", result.failure_reasons
    assert "broker" in [stage.name for stage in result.stages]
    assert harness.contexts[0].coding_client is None
    assert harness.contexts[0].execution_unavailable_reason == "docker CLI not found"
    assert any("coding_sandbox_blocked" in warning for warning in result.warnings)
    assert result.memory["coding_sandbox"]["status"] == "blocked"


def test_coding_without_container_permission_is_refused_by_name_before_anything_starts(tmp_path):
    harness = Harness(tmp_path)  # native, load and inference only
    harness.config = harness.config.model_copy(update={"broker": __import__(
        "llmbench.containers.config", fromlist=["BrokerSettings"]).BrokerSettings()})
    result = harness.run()
    assert result.state == "rejected" and "container forbidden" in result.failure_reasons[0]
    assert harness.servers == [] and not harness.lease_path.exists()


# --------------------------------------------------------------------------------------- stop, cleanup, the lease

def test_a_server_that_could_not_be_launched_is_recorded_as_not_started(tmp_path):
    # fork/exec failing under memory pressure: nothing ran, and the evidence says so instead of leaving the exit
    # blank (which a prepared sweep reads as a started server with no proof of how it ended).
    harness = Harness(tmp_path)
    harness.start_error = OSError(12, "Cannot allocate memory")
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "start")
    assert result.cleanup.attempted and result.cleanup.verified and not result.abort_campaign
    assert result.cleanup.server_exit == "not-started: OSError: [Errno 12] Cannot allocate memory"
    assert harness.markers  # the absence checks still ran: evidence, not the assumption that nothing started
    assert not harness.lease_path.exists()


def test_the_server_is_stopped_before_the_broker_is_cancelled(tmp_path):
    # The broker's Docker calls must neither delay the SIGTERM nor use up its grace period.
    harness = Harness(tmp_path, lock=SessionLock(True, True, True, True),
                      sandbox_probe=lambda **kwargs: {"status": "available", "reason": None})
    harness.runner.broker_factory = lambda *args, **kwargs: FakeBroker(harness)
    harness.config = harness.config.model_copy(update={"broker": BrokerSettings()})
    result = harness.run()
    assert result.state == "completed", result.failure_reasons
    assert harness.events == ["server-stop", "broker-cancel"]
    assert result.cleanup.verified and not harness.lease_path.exists()


@pytest.fixture
def default_sigint():
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def test_a_second_sigint_during_the_stop_is_held_until_absence_is_proven(tmp_path, default_sigint):
    # One Ctrl-C through a sweep wrapper arrives twice; the second must not cut the stop short. It is delivered
    # once the proof is recorded, the proof stands, and the interrupt still reaches the caller.
    harness = Harness(tmp_path)
    harness.sigint_during_stop = True
    with pytest.raises(KeyboardInterrupt):
        harness.run()
    assert harness.servers[0].returncode == 0 and harness.servers[0].stopped  # the stop ran to its end
    saved = harness.saved()
    assert saved["state"] == "cancelled" and saved["failure_stage"] == "cleanup"
    assert saved["cleanup"]["verified"] is True and saved["cleanup"]["server_exit"] == "returncode=0"
    assert any("after owned-resource absence was proven" in reason for reason in saved["failure_reasons"])
    assert not harness.lease_path.exists()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # the handler is put back


def test_an_interrupt_after_absence_was_proven_keeps_the_proof(tmp_path):
    # The interrupt lands in the cleanup telemetry, after the server's absence was recorded.
    def sampler(pid=None, **kwargs):
        if harness.servers and harness.servers[0].stopped:
            raise KeyboardInterrupt()
        return unified(pid=pid)

    harness = Harness(tmp_path, sampler=sampler)
    with pytest.raises(KeyboardInterrupt):
        harness.run()
    saved = harness.saved()
    assert saved["state"] == "cancelled" and saved["cleanup"]["verified"] is True
    assert "lease_retained" not in saved["cleanup"] or saved["cleanup"]["lease_retained"] is False
    assert not harness.lease_path.exists()


def test_an_interrupt_before_absence_was_proven_leaves_cleanup_uncertain(tmp_path):
    harness = Harness(tmp_path)
    harness.stop_raises = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        harness.run()
    saved = harness.saved()
    assert saved["state"] == "cleanup-uncertain" and saved["cleanup"]["verified"] is False
    assert saved["cleanup"]["lease_retained"] is True and harness.lease_path.exists()


def test_a_held_interrupt_never_replaces_the_reason_a_failed_proof_recorded(tmp_path, default_sigint):
    # The held SIGINT is delivered right after the proof is recorded. When that proof failed on its own (the port is
    # still served), the recorded reason is what is still running, not the interrupt that followed it.
    harness = Harness(tmp_path)
    harness.sigint_during_stop = harness.hold_port = True
    try:
        with pytest.raises(KeyboardInterrupt):
            harness.run()
    finally:
        for listener in harness.listeners:
            listener.close()
    saved = harness.saved()
    port = harness.servers[0].port
    assert saved["state"] == "cleanup-uncertain" and saved["cleanup"]["verified"] is False
    assert saved["cleanup"]["processes_remaining"] == [f"port:{port}-still-accepting"]
    assert saved["cleanup"]["error"] == "owned server processes remain after the stop; then KeyboardInterrupt: "
    assert saved["failure_reasons"][-1] == ("owned_resource_absence_unverified: owned server processes remain after "
                                            "the stop; then KeyboardInterrupt: ")
    assert saved["cleanup"]["lease_retained"] is True and harness.lease_path.exists()


def test_an_interrupted_watchdog_stop_never_skips_the_server_stop(tmp_path, monkeypatch):
    class StuckWatchdog:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return self

        def stop(self):
            raise KeyboardInterrupt()
    monkeypatch.setattr("llmbench.apple.MemoryWatchdog", StuckWatchdog)
    harness = Harness(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        harness.run()
    assert harness.stops and harness.servers[0].returncode == 0
    saved = harness.saved()
    assert saved["state"] == "cancelled" and saved["cleanup"]["verified"] is True
    assert not harness.lease_path.exists()


def test_only_verified_absence_this_cleanup_recorded_survives_an_interrupt(tmp_path):
    from llmbench.containers.config import CleanupEvidence
    from llmbench.containers.native import _NativeAttempt
    from llmbench.containers.runner import _Attempt
    harness = Harness(tmp_path)
    attempt = _NativeAttempt(harness.runner, harness.config, tmp_path / "attempt", None, False)
    before = attempt.cleanup  # the constructor's placeholder says verified, but proves nothing
    assert before.verified and not attempt._keeps_proven_cleanup(KeyboardInterrupt(), before)
    attempt.cleanup = CleanupEvidence(attempted=True, verified=False)
    assert not attempt._keeps_proven_cleanup(KeyboardInterrupt(), before)
    attempt.cleanup = CleanupEvidence(attempted=True, verified=True, server_exit="returncode=0")
    assert attempt._keeps_proven_cleanup(KeyboardInterrupt(), before)
    assert not attempt._keeps_proven_cleanup(RuntimeError("an error, not an interrupt"), before)
    # An uncertain cleanup keeps the reason the stage recorded, with what escaped after it; the placeholder has none.
    attempt.cleanup = CleanupEvidence(attempted=True, verified=False, error="owned server processes remain")
    assert attempt._cleanup_error_after(KeyboardInterrupt(), before) == (
        "owned server processes remain; then KeyboardInterrupt: ")
    assert attempt._cleanup_error_after(KeyboardInterrupt(), attempt.cleanup) == "KeyboardInterrupt: "
    attempt.KEEPS_PROVEN_CLEANUP_ON_INTERRUPT = False  # the container runtime's rule: what escaped, alone
    assert attempt._cleanup_error_after(KeyboardInterrupt(), before) == "KeyboardInterrupt: "
    assert _Attempt.KEEPS_PROVEN_CLEANUP_ON_INTERRUPT is False  # the container runtime's rule is unchanged


def test_sigint_is_deferred_never_dropped_and_the_handler_is_restored(default_sigint):
    with pytest.raises(KeyboardInterrupt):
        with _sigint_deferred():
            signal.raise_signal(signal.SIGINT)
            reached = True  # the block ran to its end despite the signal
    assert reached and signal.getsignal(signal.SIGINT) is signal.default_int_handler
    with _sigint_deferred():  # no signal: nothing is raised
        pass
    ignored = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with _sigint_deferred():  # the previous handler decides: an ignored SIGINT stays ignored
            signal.raise_signal(signal.SIGINT)
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, ignored)


def test_a_standalone_runs_lease_names_its_llama_server_and_a_campaign_lease_is_left_alone(tmp_path):
    harness = Harness(tmp_path / "standalone")
    assert harness.run().state == "completed"
    holder, described = harness.leases[0]
    assert json.loads(holder)["runtime"] == "metal-native" and json.loads(holder)["server_pid"] == 4242
    assert json.loads(holder)["owner"].startswith("native-run:") and json.loads(holder)["pid"] == os.getpid()
    assert "its llama-server is pid 4242" in described
    # Inside a campaign the lock belongs to the campaign: the runner never rewrites it.
    campaign = Harness(tmp_path / "campaign")
    lock = canonical_json({"pid": os.getpid(), "created": "then"}).encode("utf-8")
    campaign.lease_path.write_bytes(lock)
    result = campaign.run(lease_held=True)
    assert result.state == "completed" and not any("lease_server_pid" in item for item in result.warnings)
    assert campaign.leases[0][0] == lock and campaign.lease_path.read_bytes() == lock


def test_a_server_listening_on_another_address_is_refused(tmp_path):
    # Loopback only: the server's own `listening on` line must name the planned address, not every interface.
    harness = Harness(tmp_path, log=METAL_LOG.replace("http://127.0.0.1:50363", "http://0.0.0.0:50363"))
    result = harness.run()
    port = harness.servers[0].port
    assert (result.state, result.failure_stage) == ("failed", "ready")
    assert result.failure_reasons[0] == (f"the server reports listening on http://0.0.0.0:{port}, not the planned "
                                         f"http://127.0.0.1:{port}")
    assert harness.contexts == [] and result.cleanup.verified and not harness.lease_path.exists()


def test_a_port_still_accepting_after_the_stop_makes_cleanup_uncertain(tmp_path):
    harness = Harness(tmp_path)
    harness.hold_port = True
    try:
        result = harness.run()
    finally:
        for listener in harness.listeners:
            listener.close()
    port = harness.servers[0].port
    assert result.state == "cleanup-uncertain" and result.abort_campaign
    assert result.cleanup.processes_remaining == (f"port:{port}-still-accepting",) and not result.cleanup.verified
    assert result.cleanup.lease_retained and harness.lease_path.exists()
    # The retained lease says which server to look for, through the same text `doctor` prints.
    holder = json.loads(harness.lease_path.read_text(encoding="utf-8"))
    assert holder["runtime"] == "metal-native" and holder["server_pid"] == 4242
    assert "its llama-server is pid 4242" in describe_lock(harness.lease_path)


# ----------------------------------------------------------------------------- admission: the real identity check

def pinned(tmp_path, **metal) -> dict:
    """A synthetic llama.cpp release where the config expects it, and the pins a prepare would record for it."""
    executable = release(tmp_path / "llama-b11011", **metal)
    hashed = hash_native_server(executable)
    return {"executable": str(executable), "executable_sha256": hashed["executable_sha256"],
            "libraries_sha256": hashed["libraries_sha256"], "build_info": "b11011-aa39d7a3e",
            "help_sha256": help_sha256(HELP.decode("utf-8"))}


def scripted_executable(monkeypatch, *, version=VERSION, helptext=HELP):
    calls = []

    def run(self, argv, *, timeout_seconds, max_output_bytes=0):
        calls.append(argv)
        return WorkerResult("completed", 0, helptext if argv == "--help" else b"", version if argv == "--version"
                            else b"")
    monkeypatch.setattr("llmbench.containers.native_prep.NativeProcessExecutor.run", run)
    return calls


def tamper_executable(ref):
    Path(ref["executable"]).write_bytes(Path(ref["executable"]).read_bytes() + b"\x00")


def tamper_library(ref):
    (Path(ref["executable"]).parent / "libggml-metal.0.24.0.dylib").write_bytes(macho(filetype=6))


def add_library(ref):
    (Path(ref["executable"]).parent / "libggml-extra.dylib").write_bytes(macho(filetype=6))


@pytest.mark.skipif(os.name == "nt", reason="POSIX links")
@pytest.mark.parametrize("tamper, answers, fragment", [
    (tamper_executable, {}, "executable sha256"),
    (tamper_library, {}, "libraries beside the executable differ"),
    (add_library, {}, "libraries beside the executable differ"),
    (None, {"version": VERSION.replace(b"build 11011, commit aa39d7a3e", b"build 11012, commit bb39d7a3e")},
     "reports build b11012-bb39d7a3e, not the pinned b11011-aa39d7a3e"),
    (None, {"helptext": HELP + b"\n--extra-flag  something new\n"}, "help does not match"),
])
def test_admission_refuses_a_server_that_is_not_the_pinned_build(tmp_path, monkeypatch, tamper, answers, fragment):
    ref = pinned(tmp_path)
    if tamper is not None:
        tamper(ref)
    calls = scripted_executable(monkeypatch, **answers)
    harness = Harness(tmp_path, native_server=ref, fake_identity=False)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("rejected", "admit"), result.failure_reasons
    assert "native server identity" in result.failure_reasons[0] and fragment in result.failure_reasons[0]
    assert harness.servers == [] and not harness.lease_path.exists()
    assert calls == ([] if tamper is not None else ["--version", "--help"])  # a changed file never runs


@pytest.mark.skipif(os.name == "nt", reason="POSIX links")
def test_admission_accepts_the_pinned_build_and_records_its_identity(tmp_path, monkeypatch):
    ref = pinned(tmp_path)
    calls = scripted_executable(monkeypatch)
    harness = Harness(tmp_path, native_server=ref, fake_identity=False)
    result = harness.run()
    assert result.state == "completed", result.failure_reasons
    assert calls == ["--version", "--help"] and len(harness.servers) == 1
    identity = harness.saved("preflight.json")["native_server"]
    assert identity["executable_sha256"] == ref["executable_sha256"] and identity["build_info"] == "b11011-aa39d7a3e"
    assert identity["libraries_sha256"] == ref["libraries_sha256"] and "libggml-metal.0.dylib" in identity["libraries"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX links")
def test_admission_reproves_the_linkage_a_config_that_skipped_prepare_never_proved(tmp_path, monkeypatch):
    # A Homebrew-style build loads its Metal backend from ../lib. Hashing the (library-free) bin directory pins
    # nothing that runs; the config's hashes still match, so only the linkage proof can refuse it.
    server = macho(dylibs=("@rpath/libllama.0.dylib", "@loader_path/../lib/libggml-metal.dylib",
                           "/usr/lib/libSystem.B.dylib"), rpaths=("@loader_path",))
    ref = pinned(tmp_path, server=server)
    calls = scripted_executable(monkeypatch)
    harness = Harness(tmp_path, native_server=ref, fake_identity=False)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("rejected", "admit")
    assert "pin does not cover" in result.failure_reasons[0] and "../lib/libggml-metal.dylib" in result.failure_reasons[0]
    assert calls == [] and harness.servers == [] and not harness.lease_path.exists()


# -------------------------------------------------------------------------------- boundaries a native run never uses

def test_without_coding_suites_the_runner_reaches_no_docker_or_process_boundary(tmp_path, monkeypatch):
    # Nothing Docker-shaped is reachable: not the sandbox probe (left unset here, so the real one would run), not
    # any bounded Docker executor (ComposeExecutor included: it inherits this run), not a process of any kind.
    def tripwire(name):
        def fail(*args, **kwargs):
            pytest.fail(f"a native run without coding suites reached {name}")
        return fail
    monkeypatch.setattr("llmbench.coding.sandbox.BoundedProcessExecutor.run", tripwire("a bounded Docker executor"))
    monkeypatch.setattr("llmbench.coding.sandbox.probe_docker_sandbox", tripwire("the Docker sandbox probe"))
    monkeypatch.setattr(subprocess, "Popen", tripwire("subprocess.Popen"))
    harness = Harness(tmp_path, sandbox_probe=None)
    result = harness.run()
    assert result.state == "completed", result.failure_reasons
    assert harness.config.broker is None and "broker" not in [stage.name for stage in result.stages]


def test_a_log_the_artifact_budget_cut_short_is_a_warning_not_a_lost_run(tmp_path):
    # The drain thread keeps draining when the budget refuses a write (test_containers_native_process covers that
    # with a real child); the run says what was kept and dropped, and why.
    harness = Harness(tmp_path)
    harness.log_budget_error = "ValueError: artifact budget exhausted"
    result = harness.run()
    assert result.state == "completed", result.failure_reasons
    kept = len(harness.log.replace(":50363", f":{harness.servers[0].port}"))
    assert f"inference_log_truncated:logs/server.log: kept {kept} bytes, dropped 4096 " \
           "(ValueError: artifact budget exhausted)" in result.warnings
