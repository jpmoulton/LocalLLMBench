"""The metal-native runner end to end against fakes: stage order, evidence, admission, the memory watchdog,
cancellation and cleanup proof. The server log is the real b11011 Metal startup log captured on an M1
(`tests/data/startup-b11011-metal-qwen3-1.7b.log`); no process, GPU, model, Docker or network is touched."""

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from llmbench.config import canonical_json
from llmbench.containers.config import ContainerRunConfig
from llmbench.containers.native import NativeRunner, _Output
from llmbench.measurement import SpeedObservation
from llmbench.safety import SessionLock

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
        self.pid, self.returncode, self.signals_sent = 4242, None, []
        self.log_error, self.dropped_bytes, self.log_bytes = None, 0, 0
        self.started = self.terminated = False
        harness.servers.append(self)

    def start(self):
        if self.harness.start_error:
            raise self.harness.start_error
        self.started = True
        text = self.harness.log.replace("http://127.0.0.1:50363", f"http://127.0.0.1:{self.port}")
        self.log_sink.write(text.encode("utf-8"))
        self.log_sink.close()
        self.log_bytes = len(text)
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
        if self.returncode is None:
            self.signals_sent.append("SIGTERM")
            self.returncode = 0
        if self.harness.died_by_sigkill:
            self.returncode = -9

    def remaining(self, marker):
        self.harness.markers.append(marker)
        return list(self.harness.leftovers)


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
                 log=METAL_LOG, limits=None):
        tmp_path.mkdir(parents=True, exist_ok=True)
        model = tmp_path / "model.gguf"
        model.write_bytes(b"GGUF" * 64)
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw.pop("inference_image")
        raw["runtime"] = "metal-native"
        raw["native_server"] = {"executable": str(tmp_path / "llama-b11011" / "llama-server"),
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
        self.start_error = self.exit_during_startup = None
        self.died_by_sigkill = False
        self.identity_calls = []
        self.sampler = sampler or (lambda pid=None, **kwargs: unified(pid=pid))
        self.evaluator = evaluator or self.evaluate
        self.lease_path = tmp_path / "gpu.lock"
        self.runner = NativeRunner(
            policy_path=tmp_path / "no-policy.json", session_lock=lock, lease_path=self.lease_path,
            http_factory=self.http, evaluator=self.call_evaluator, telemetry=self.sample,
            registry_validator=lambda config: "f" * 64, clock=self.clock, sleep=self.clock.sleep,
            process_factory=lambda **kwargs: FakeServer(self, **kwargs), identity_check=self.identity,
            sandbox_probe=lambda **kwargs: {"status": "blocked", "reason": "docker CLI not found"})
        self.output = tmp_path / "out"

    def sample(self, pid=None, **kwargs):
        return self.sampler(pid, **kwargs)

    def identity(self, config, argv):
        self.identity_calls.append(list(argv))
        return {"executable": config.native_server.executable, "executable_sha256": "a" * 64,
                "libraries_sha256": "b" * 64, "build_info": "b11011-aa39d7a3e", "help_sha256": "c" * 64}

    def call_evaluator(self, config, context):
        assert self.lease_path.exists()  # the GPU lease is held for the whole evaluation
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


def test_the_runner_never_issues_a_docker_command(tmp_path):
    harness = Harness(tmp_path)
    calls = []
    import llmbench.containers.executor as executor
    original = executor.ComposeExecutor.run
    executor.ComposeExecutor.run = lambda self, *a, **k: calls.append(a) or original(self, *a, **k)
    try:
        assert harness.run().state == "completed"
    finally:
        executor.ComposeExecutor.run = original
    assert calls == []


def test_concurrent_log_and_evidence_writes_share_one_budget(tmp_path):
    harness = Harness(tmp_path)
    result = harness.run()
    index = {entry.path: entry for entry in result.artifacts}
    assert index["logs/server.log"].size == len(harness.log.encode("utf-8"))
    assert threading.active_count() < 50
