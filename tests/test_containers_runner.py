import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from llmbench.backends.http import TransportError
from llmbench.coding.sandbox import WorkerResult
from llmbench.config import canonical_json
from llmbench.containers.config import ContainerRunConfig, read_run_config
from llmbench.containers.executor import ComposeExecutor, project_filter
from llmbench.containers.lease import GpuLease
from llmbench.containers.runner import ContainerRunner
from llmbench.evaluations.selection import expected_task_rows
from llmbench.measurement import SpeedObservation
from llmbench.safety import SessionLock

DATA = Path(__file__).parent / "data"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.json"
CID = "c0ffee" * 10 + "abcd"
NETWORK = "0123456789ab"
STARTUP = ((DATA / "startup-b11011-lv4.log").read_text(encoding="utf-8")
           .replace("prompt cache is enabled, size limit: 8192 MiB", "prompt cache is disabled - use `--cache-ram N`")
           .replace("thinking = 1", "thinking = 0"))
OK = WorkerResult("completed", 0)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeDocker:
    """Scripted executor. Every argv must belong to the real bounded vocabulary."""

    def __init__(self, clock, config, **script):
        self.clock, self.calls, self.vocabulary = clock, [], ComposeExecutor()
        image = config.inference_image
        inspection = {"Id": image.image_id, "RepoDigests": [image.reference], "Os": "linux", "Architecture": "amd64",
                      "Config": {"Entrypoint": list(image.entrypoint), "Env": ["LLAMA_ARG_HOST=0.0.0.0"]}}
        self.script = {"image-inspect": WorkerResult("completed", 0, json.dumps(inspection).encode()),
                       "config": WorkerResult("completed", 0, b"{}"), "up": OK,
                       "compose-ps": WorkerResult("completed", 0, json.dumps({"ID": CID, "Service": "inference"}).encode()),
                       "port": WorkerResult("completed", 0, b"127.0.0.1:53184\n"),
                       "inspect": WorkerResult("completed", 0, b"running 0 false\n"),
                       "logs": WorkerResult("completed", 0, b"", STARTUP.encode()), "down": OK, "ps": OK,
                       "network-ls": OK, "rm": OK, "network-rm": OK, **script}

    @staticmethod
    def key(argv):
        if argv[1] == "compose":
            return "compose-ps" if argv[12] == "ps" else argv[12]
        return {"image": "image-inspect", "network": "network-" + argv[2]}.get(argv[1], argv[1])

    def run(self, argv, *, timeout_seconds, max_output_bytes):
        self.vocabulary._validate(argv)
        assert type(timeout_seconds) is int and timeout_seconds >= 1 and max_output_bytes >= 1
        self.calls.append((argv, timeout_seconds, max_output_bytes))
        self.clock.now += 0.5
        step = self.script[self.key(argv)]
        if isinstance(step, list):
            step = step.pop(0) if len(step) > 1 else step[0]
        return step(argv) if callable(step) else step

    def keys(self):
        return [self.key(argv) for argv, _, _ in self.calls]


class FakeHTTP:
    def __init__(self, failures=2):
        self.failures, self.urls, self.requests = failures, [], 0

    def __call__(self, base_url, *, timeout):
        self.urls.append(base_url)
        return self

    def request_json(self, method, path, payload=None, **kwargs):
        assert (method, path) == ("GET", "/health")
        self.requests += 1
        if self.requests <= self.failures:
            raise TransportError("HTTP 503 from the inference server /health")
        return {"status": "ok"}


def observation():
    return SpeedObservation(status="completed", finish_reason="length", output_tokens=512,
                            native_generation_seconds=7.4, elapsed_seconds=8.0, first_event_seconds=0.4,
                            content_event_times=tuple(0.4 + 0.074 * index for index in range(100)),
                            requested_output_tokens=512, input_tokens=512, expected_input_tokens=512,
                            accepted_tokens_verified=True, native_timing_source="llama.cpp.timings")


class Harness:
    def __init__(self, tmp_path, *, evaluator=None, http=None, telemetry=None, hasher=None, lock=None,
                 registry=None, config_changes=None, **script):
        tmp_path.mkdir(parents=True, exist_ok=True)
        model = tmp_path / "model.gguf"
        model.write_bytes(b"GGUF" * 64)
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw["assets"][0].update(host_path=str(model), size_bytes=256,
                                sha256=hashlib.sha256(b"GGUF" * 64).hexdigest())
        raw["engine"]["ctx_size"], raw["requested_input_tokens"] = 131072, 512
        for path, value in (config_changes or {}).items():
            target = raw
            *parents, leaf = path.split("__")
            for parent in parents:
                target = target[int(parent) if parent.isdigit() else parent]
            target[leaf] = value
        self.config = ContainerRunConfig.model_validate_json(json.dumps(raw))
        prep = tmp_path / "prep"
        prep.mkdir()
        (prep / "llama-server-help.txt").write_bytes((DATA / "llama-server-help-b11011.txt").read_bytes())
        (prep / "llama-server-version.txt").write_bytes((DATA / "llama-server-version-b11011.txt").read_bytes())
        self.clock, self.http, self.contexts = FakeClock(), http or FakeHTTP(), []
        self.docker = FakeDocker(self.clock, self.config, **script)
        self.samples = iter([1500.0, 20000.0, 1600.0, 1600.0])
        self.evaluator = evaluator or self.evaluate
        self.lease_path = tmp_path / "gpu.lock"
        self.runner = ContainerRunner(
            executor=self.docker, http_factory=self.http, evaluator=self.call_evaluator,
            registry_validator=registry or (lambda config: "f" * 64), capabilities_dir=prep, clock=self.clock,
            sleep=self.clock.sleep, session_lock=lock or SessionLock(True, True, True),
            telemetry=telemetry or self.sample, hasher=hasher, lease_path=self.lease_path)
        self.output = tmp_path / "out"

    def sample(self):
        return {"gpus": [{"index": 0.0, "name": "RTX 5090", "memory_used_mib": next(self.samples),
                          "memory_total_mib": 32607.0}], "gpu_error": None}

    def call_evaluator(self, config, context):
        assert self.lease_path.exists()  # The GPU lease is held for the whole evaluation.
        self.contexts.append(context)
        return self.evaluator(config, context)

    def evaluate(self, config, context):
        props = json.loads((DATA / "props-b11011.json").read_text(encoding="utf-8"))["body"]
        props["model_alias"], props["default_generation_settings"]["n_ctx"] = config.alias(), config.engine.ctx_size
        (Path(context.artifacts_dir) / "inspect.log").write_bytes(b"inspect evidence")
        self.clock.now += 120
        return {"samples": [{"task_id": "tools/nested-exact-v1", "category": "tools", "score": 1.0,
                             "status": "completed"},
                            {"task_id": "niah/multi", "category": "retrieval", "score": 0.0, "status": "timeout"}],
                "speed_observations": [observation() for _ in range(config.speed.repetitions)],
                "readback": {"props": props, "models": {}, "slots": [{"id": 0, "n_ctx": config.engine.ctx_size}]},
                "count_route": {"exact": True}, "overflow_probe": {"passed": True},
                "actual_context_verified": True, "abort_reason": None, "errors": []}

    def run(self, **options):
        return self.runner.run(self.config, self.output, **options)

    def saved(self):
        return json.loads((self.output / "result.json").read_text(encoding="utf-8"))


def assert_clean(harness, result):
    assert result.cleanup.verified and not result.abort_campaign and not harness.lease_path.exists()
    assert not {"rm", "network-rm"} & set(harness.docker.keys())


def test_happy_path_ordering_artifacts_and_evidence(tmp_path):
    harness = Harness(tmp_path)
    result = harness.run()
    assert result.state == "completed" and result.synthetic is True and result.failure_reasons == ()
    assert harness.docker.keys() == ["image-inspect", "config", "up", "compose-ps", "port", "inspect", "inspect",
                                     "inspect", "logs", "inspect", "logs", "down", "ps", "network-ls"]
    assert [stage.name for stage in result.stages] == ["admit", "hash", "plan", "start", "ready", "verify",
                                                       "quality", "verify", "cleanup", "report"]
    assert all(stage.status == "ok" for stage in result.stages)
    assert harness.http.urls == ["http://127.0.0.1:53184"] and harness.contexts[0].base_url == harness.http.urls[0]
    assert Path(harness.contexts[0].artifacts_dir) == harness.output.resolve() / "evaluator"
    assert result.container_ids == (CID,) and result.project_name == "llmbench-" + result.attempt_id[:12]
    owned = project_filter(result.project_name)
    assert harness.docker.calls[-2][0] == ("docker", "ps", "--all", "--filter", owned, "--format", "{{.ID}}")
    assert harness.docker.calls[-1][0] == ("docker", "network", "ls", "--filter", owned, "--format", "{{.ID}}")
    assert harness.docker.calls[-3][0][-4:] == ("down", "--volumes", "--timeout", "15")
    assert result.load_seconds == pytest.approx(56.395523) and result.vram_used_mib_after_load == 20000.0
    assert result.speed["qualifies"] is True and result.speed["minimum_native_tps"] == pytest.approx(512 / 7.4)
    assert result.quality["tools"]["score"] == 1.0 and result.quality["retrieval"] == {
        "attempted": 1, "score": 0.0, "comparison": {"status": "unmeasured"}}
    assert result.samples_total == 2 and result.actual_context_verified is True
    assert result.effective_settings_verified is True  # --fit off is verified by behaviour: every fit-adjustable control read back equal to its request.
    statuses = {row.control: row.status for row in result.settings}
    assert statuses["ctx_size"] == "verified-api" and statuses["cache_type_k"] == "verified-log"
    assert statuses["context_shift"] == "verified-behavior" and statuses["fit"] == "verified-behavior"
    assert result.model_evidence["unchanged"] is True and result.image_evidence["id"].startswith("sha256:")
    assert_clean(harness, result)
    names = {entry.path for entry in result.artifacts}
    assert {"config.json", "preflight.json", "plan/compose.json", "plan/server-argv.json",
            "plan/compose-resolved.json", "logs/compose-up.log", "logs/startup.log", "logs/inference-final.log",
            "logs/compose-down.log", "verify.json", "evaluation.json", "settings-evidence.json", "cleanup.json",
            "stages.jsonl", "evaluator/inspect.log", "reports/campaign.json", "reports/report.md",
            "reports/report.html", "reports/candidate.md"} <= names
    for entry in result.artifacts:
        content = (harness.output / entry.path).read_bytes()
        assert (hashlib.sha256(content).hexdigest(), len(content)) == (entry.sha256, entry.size), entry.path
    saved = harness.saved()
    assert saved["state"] == "completed" and saved["attempt_id"] == result.attempt_id
    index = json.loads((harness.output / "artifact-index.json").read_text(encoding="utf-8"))
    # result.json cannot contain its own hash; the separate index seals it together with everything else.
    assert {entry["path"] for entry in index["artifacts"]} == names | {"result.json"}
    compose = json.loads((harness.output / "plan" / "compose.json").read_text(encoding="utf-8"))
    argv = json.loads((harness.output / "plan" / "server-argv.json").read_text(encoding="utf-8"))
    assert compose["services"]["inference"]["command"] == argv["argv"] and argv["environment"] == {}
    assert len((harness.output / "stages.jsonl").read_text(encoding="utf-8").splitlines()) == 10
    evaluation = json.loads((harness.output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["speed_observations"][0]["output_tokens"] == 512
    candidate = (harness.output / "reports" / "candidate.md").read_text(encoding="utf-8")
    assert "SYNTHETIC HARNESS TEST" in candidate and "Cleanup verified: **True**" in candidate
    assert result.reports["candidate"] == "reports/candidate.md"
    assert result.reports["eligibility"]["eligible"] is False


def test_startup_exit_fails_fast_with_the_log(tmp_path):
    harness = Harness(tmp_path, inspect=WorkerResult("completed", 0, b"exited 1 false\n"),
                      logs=WorkerResult("completed", 0, b"", b"error: unknown argument: --bogus\n"))
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "ready") and harness.http.requests == 0
    assert "status=exited exit=1" in result.failure_reasons[0] and "--bogus" in result.failure_reasons[0]
    assert (harness.output / "logs" / "startup.log").read_bytes() == b"error: unknown argument: --bogus\n"
    assert harness.contexts == [] and "down" in harness.docker.keys()
    assert_clean(harness, result)


def test_oom_during_startup_is_reported(tmp_path):
    result = Harness(tmp_path, inspect=WorkerResult("completed", 0, b"running 137 true\n")).run()
    assert result.state == "failed" and "oom=true" in result.failure_reasons[0]
    assert "inference_oom_killed" in result.failure_reasons


def test_readiness_timeout_is_bounded_by_the_injected_clock(tmp_path):
    harness = Harness(tmp_path, http=FakeHTTP(failures=10**9), config_changes={
        "bounds": {"candidate_wall_seconds": 600, "hash_seconds": 30, "startup_seconds": 10, "verify_seconds": 10}})
    result = harness.run()
    assert (result.state, result.failure_stage) == ("timeout", "ready")
    assert "not healthy within" in result.failure_reasons[0] and 1 <= harness.http.requests <= 10
    startup = [stage for stage in result.stages if stage.name in {"start", "ready"}]
    assert sum(stage.elapsed_seconds for stage in startup) <= 10
    assert (harness.output / "logs" / "inference-final.log").exists() and harness.contexts == []
    assert [stage.status for stage in result.stages if stage.name == "ready"] == ["timeout"]
    assert_clean(harness, result)


def test_evaluator_exception_is_recorded_and_cleaned_up(tmp_path):
    def broken(config, context):
        raise RuntimeError("inspect crashed <b>now</b>")
    harness = Harness(tmp_path, evaluator=broken)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "quality")
    assert result.failure_reasons == ("RuntimeError: inspect crashed <b>now</b>",)
    assert json.loads((harness.output / "evaluation.json").read_text(encoding="utf-8")) == {
        "error": "RuntimeError: inspect crashed <b>now</b>"}
    for name in ("report.html", "candidate.md"):
        text = (harness.output / "reports" / name).read_text(encoding="utf-8")
        assert "<b>" not in text and "inspect crashed &lt;b&gt;now" in text, name
    assert result.actual_context_verified is False and result.speed["qualifies"] is False
    assert_clean(harness, result)


def test_evaluator_abort_reason_and_setting_mismatch_fail_the_candidate(tmp_path):
    def aborting(config, context):
        return {"samples": [], "speed_observations": [], "abort_reason": "server cancellation unverified"}
    result = Harness(tmp_path / "a", evaluator=aborting).run()
    assert result.state == "failed" and "server cancellation unverified" in result.failure_reasons[0]
    harness = Harness(tmp_path / "b", config_changes={"engine__cache_type_k": "q8_0", "engine__cache_type_v": "q8_0"})
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "verify")
    assert result.failure_reasons == ("effective settings differ from the request: cache_type_k, cache_type_v",)
    assert result.effective_settings_verified is False and result.cleanup.verified


def test_log_overflow_is_flagged(tmp_path):
    harness = Harness(tmp_path, logs=WorkerResult("output-limit", None, b"", STARTUP.encode()))
    result = harness.run()
    assert result.state == "completed"
    assert result.warnings == ("inference_log_truncated:logs/startup.log",
                               "inference_log_truncated:logs/inference-final.log")
    log_calls = [call for call in harness.docker.calls if call[0][1] == "logs"]
    assert all(call[2] == harness.config.limits.log_max_bytes for call in log_calls)


def test_keyboard_interrupt_cleans_up_writes_the_result_and_reraises(tmp_path):
    def interrupted(config, context):
        raise KeyboardInterrupt()
    harness = Harness(tmp_path, evaluator=interrupted)
    with pytest.raises(KeyboardInterrupt):
        harness.run()
    saved = harness.saved()
    assert (saved["state"], saved["failure_stage"]) == ("cancelled", "quality")
    assert saved["cleanup"]["verified"] is True and saved["abort_campaign"] is False
    assert harness.docker.keys()[-3:] == ["down", "ps", "network-ls"] and not harness.lease_path.exists()


def test_budget_below_stage_bounds_is_rejected_without_docker(tmp_path):
    harness = Harness(tmp_path)
    result = harness.run(remaining_budget_seconds=500)
    assert (result.state, result.failure_stage) == ("rejected", "admit") and harness.docker.calls == []
    assert "below the stage bounds" in result.failure_reasons[0] and result.cleanup.attempted is False
    assert harness.saved()["state"] == "rejected" and not harness.lease_path.exists()
    granted = Harness(tmp_path / "grant", config_changes={"parent_grant_seconds": 100})
    assert granted.run().state == "rejected" and granted.docker.calls == []


def test_hash_time_is_charged_to_the_candidate(tmp_path):
    def slow_hasher(path):
        harness.clock.now += 250
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    harness = Harness(tmp_path, hasher=slow_hasher, config_changes={
        "bounds": {"candidate_wall_seconds": 800, "evaluation_seconds": 600}})
    result = harness.run()
    assert result.state == "completed"
    hashed = next(stage for stage in result.stages if stage.name == "hash")
    assert hashed.elapsed_seconds == pytest.approx(250) and result.budget_charged_seconds > 250
    # 800 s wall - 60 s cleanup reserve: the evaluator gets what hashing left, not its full 600 s bound.
    assert harness.contexts[0].deadline_monotonic <= 1000.0 + 800 - 60
    assert harness.contexts[0].deadline_monotonic - 1000.0 - 250 < 600
    assert result.model_evidence["pre"]["hash_seconds"] == pytest.approx(250)


def test_hash_overrun_and_wrong_hash_never_start_a_container(tmp_path):
    def too_slow(path):
        slow.clock.now += 301
        return "0" * 64
    slow = Harness(tmp_path / "slow", hasher=too_slow)
    result = slow.run()
    assert (result.state, result.failure_stage) == ("timeout", "hash") and "up" not in slow.docker.keys()
    bad = Harness(tmp_path / "bad", config_changes={"assets__0__sha256": "0" * 64})
    result = bad.run()
    assert (result.state, result.failure_stage) == ("rejected", "hash")
    assert bad.docker.keys() == ["image-inspect"] and result.cleanup.attempted is False
    assert result.cleanup.verified and not bad.lease_path.exists()
    assert "SHA256" in result.failure_reasons[0] and (bad.output / "reports" / "candidate.md").exists()


def test_failed_compose_down_falls_back_to_exact_owned_ids(tmp_path):
    listed = WorkerResult("completed", 0, (CID + "\n").encode())
    harness = Harness(tmp_path, down=WorkerResult("completed", 1, b"", b"daemon error"), ps=[listed, OK],
                      **{"network-ls": [WorkerResult("completed", 0, NETWORK.encode()), OK]})
    result = harness.run()
    assert result.state == "completed" and result.cleanup.compose_down_returncode == 1
    assert harness.docker.keys()[-7:] == ["down", "ps", "network-ls", "rm", "network-rm", "ps", "network-ls"]
    removals = [argv for argv, _, _ in harness.docker.calls if argv[1] == "rm" or argv[1:3] == ("network", "rm")]
    assert removals == [("docker", "rm", "-f", CID), ("docker", "network", "rm", NETWORK)]
    assert result.cleanup.verified and result.cleanup.containers_remaining == ()


def test_remaining_resources_make_cleanup_uncertain_and_abort_the_campaign(tmp_path):
    harness = Harness(tmp_path, ps=WorkerResult("completed", 0, CID.encode()))
    result = harness.run()
    assert (result.state, result.failure_stage, result.abort_campaign) == ("cleanup-uncertain", "cleanup", True)
    assert result.cleanup.containers_remaining == (CID,) and not result.cleanup.verified
    assert result.cleanup.lease_retained and harness.lease_path.exists()  # Needs explicit reconciliation.
    assert result.effective_settings_verified is False and result.actual_context_verified is False
    assert harness.saved()["abort_campaign"] is True
    assert "Cleanup verified: **False**" in (harness.output / "reports" / "candidate.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [{"ps": WorkerResult("timeout", None)},
                                    {"network-ls": WorkerResult("completed", 1, b"", b"daemon unreachable")},
                                    {"ps": WorkerResult("completed", 0, b"not-an-id; rm -rf /")},
                                    {"down": lambda argv: (_ for _ in ()).throw(ValueError("vocabulary"))}])
def test_unproven_absence_is_never_reported_as_clean(tmp_path, script):
    result = Harness(tmp_path, **script).run()
    assert result.state == "cleanup-uncertain" and result.abort_campaign and result.cleanup.error


def test_cleanup_never_prunes_or_touches_foreign_resources(tmp_path):
    harness = Harness(tmp_path, up=WorkerResult("completed", 1, b"", b"port is already allocated"))
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "start") and result.container_ids == ()
    assert harness.docker.keys() == ["image-inspect", "config", "up", "down", "ps", "network-ls"]
    for argv, _, _ in harness.docker.calls:
        assert not {"prune", "kill", "stop", "exec", "system", "volume"} & set(argv)
        assert all(result.project_name in item for item in argv if "llmbench-" in item)
    assert_clean(harness, result)


def test_lease_held_elsewhere_means_zero_docker_calls(tmp_path):
    harness = Harness(tmp_path)
    other = GpuLease(harness.lease_path, owner="lm-studio-campaign").acquire()
    result = harness.run()
    assert (result.state, result.failure_stage) == ("rejected", "admit") and harness.docker.calls == []
    assert "gpu_lease_unavailable" in result.failure_reasons[0]
    assert "lm-studio-campaign" in harness.lease_path.read_text(encoding="utf-8")
    caller = Harness(tmp_path / "held")
    caller.lease_path = harness.lease_path
    caller.runner.lease_path = harness.lease_path
    assert caller.run(lease_held=True).state == "completed" and harness.lease_path.exists()
    other.release()


@pytest.mark.parametrize("options, fragment", [
    ({"lock": SessionLock(allow_model_operations=True, allow_inference=True)}, "container forbidden"),
    ({"lock": SessionLock(allow_container_execution=True, allow_inference=True)}, "load forbidden"),
    ({"registry": lambda config: (_ for _ in ()).throw(ValueError("unknown benchmark revision"))}, "revision"),
    ({"config_changes": {"inference_image__help_sha256": "0" * 64}}, "help_sha256"),
    ({"config_changes": {"inference_image__build_info": "b1-deadbeef"}}, "build_info"),
    # Container mode is admitted now; an evaluator image with the wrong entrypoint is what gets rejected.
    ({"config_changes": {"evaluator": {"mode": "container", "image": {
        "role": "evaluator", "reference": "sha256:" + "a" * 64, "image_id": "sha256:" + "a" * 64,
        "entrypoint": ["python"]}}}}, "entrypoint must be python -m llmbench.container_eval"),
])
def test_admission_rejections_run_nothing(tmp_path, options, fragment):
    harness = Harness(tmp_path, **options)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("rejected", "admit") and harness.docker.calls == []
    assert fragment in result.failure_reasons[0] and harness.http.urls == [] and not harness.lease_path.exists()
    assert (harness.output / "config.json").exists() and harness.saved()["cleanup"]["attempted"] is False


def test_missing_registry_module_fails_closed(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "llmbench.registry", None)
    harness = Harness(tmp_path)
    harness.runner.registry_validator = ContainerRunner(capabilities_dir=tmp_path).registry_validator
    result = harness.run()
    assert result.state == "rejected" and harness.docker.calls == [] and "Error" in result.failure_reasons[0]


def test_image_and_gpu_conflicts_reject_before_loading(tmp_path):
    wrong = WorkerResult("completed", 0, json.dumps({"Id": "sha256:" + "e" * 64, "Os": "linux",
                                                     "Architecture": "amd64", "Config": {}}).encode())
    harness = Harness(tmp_path / "image", **{"image-inspect": wrong})
    result = harness.run()
    assert result.state == "rejected" and harness.docker.keys() == ["image-inspect"]
    assert not harness.lease_path.exists()
    busy = {"gpus": [{"index": 0.0, "memory_used_mib": 21000.0, "memory_total_mib": 32607.0}], "gpu_error": None}
    harness = Harness(tmp_path / "gpu", telemetry=lambda: busy)
    result = harness.run()
    assert result.state == "rejected" and "already has 21000 MiB" in result.failure_reasons[0]
    assert "up" not in harness.docker.keys()


def test_unexpected_port_or_container_listing_fails_without_contacting_anything(tmp_path):
    harness = Harness(tmp_path / "port", port=WorkerResult("completed", 0, b"0.0.0.0:8080\n"))
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "start") and harness.http.urls == []
    assert "loopback" in result.failure_reasons[0] and result.cleanup.verified
    rows = json.dumps([{"ID": CID, "Service": "inference"}, {"ID": CID[:12], "Service": "inference"}]).encode()
    harness = Harness(tmp_path / "ps", **{"compose-ps": WorkerResult("completed", 0, rows)})
    result = harness.run()
    assert result.state == "failed" and "exactly one" in result.failure_reasons[0] and result.cleanup.verified


def test_output_directory_must_be_new_or_empty(tmp_path):
    harness = Harness(tmp_path)
    harness.output.mkdir()
    (harness.output / "earlier.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        harness.run()
    assert harness.docker.calls == []


def test_default_evaluator_passes_only_known_context_fields(tmp_path, monkeypatch):
    container_eval = pytest.importorskip("llmbench.container_eval")
    from types import SimpleNamespace
    from llmbench.containers import runner as runner_module
    seen = {}
    monkeypatch.setattr(container_eval, "run_evaluation",
                        lambda config, ctx: seen.update(config=config, ctx=ctx) or {"samples": []})
    context = SimpleNamespace(base_url="http://127.0.0.1:53184", artifacts_dir=tmp_path, deadline_monotonic=5.0,
                              session_lock=SessionLock(), clock=FakeClock(), policy_path=tmp_path / "policy.json",
                              artifacts=None, future_field="ignored")
    assert runner_module._default_evaluator("config", context) == {"samples": []}
    assert isinstance(seen["ctx"], container_eval.EvaluationContext) and seen["ctx"].deadline_monotonic == 5.0
    assert seen["ctx"].base_url == "http://127.0.0.1:53184" and seen["ctx"].policy_path == tmp_path / "policy.json"


def test_evaluator_shares_the_artifact_budget_and_reports_its_errors(tmp_path):
    def evaluator(config, context):
        context.artifacts.write("readback.json", b"{}")
        with context.artifacts.trace("quality/transport.jsonl") as sink:
            sink({"event": "chunk"})
        return {"samples": [], "speed_observations": [], "errors": [{"stage": "count-route", "error": "inexact"}],
                "readback": None, "overflow_probe": None, "abort_reason": None}
    harness = Harness(tmp_path, evaluator=evaluator)
    result = harness.run()
    statuses = {row.control: row.status for row in result.settings}
    assert result.state == "completed" and statuses["alias"] == "unobserved"  # No readback, no API evidence.
    assert statuses["ctx_size"] == "verified-log" and statuses["context_shift"] == "argv-accepted"
    assert any("count-route" in item for item in result.warnings)
    names = {entry.path for entry in result.artifacts}
    assert {"evaluator/readback.json", "evaluator/quality/transport.jsonl"} <= names
    assert harness.contexts[0].policy_path == harness.runner.policy_path


def test_real_boundaries_are_not_synthetic_and_construction_is_inert(tmp_path):
    runner = ContainerRunner(capabilities_dir=tmp_path)
    assert runner.synthetic is False and runner.executor is None
    assert Harness(tmp_path).runner.synthetic is True


@pytest.mark.parametrize("exception", [SystemExit(9), KeyboardInterrupt()])
def test_control_exit_always_tears_down_and_persists_before_reraise(tmp_path, exception):
    def exits(config, context):
        raise exception
    harness = Harness(tmp_path, evaluator=exits)
    with pytest.raises(type(exception)):
        harness.run()
    saved = harness.saved()
    assert saved["state"] == "cancelled" and saved["cleanup"]["verified"] is True
    assert harness.docker.keys()[-3:] == ["down", "ps", "network-ls"]
    assert not harness.lease_path.exists()
    assert (harness.output / "artifact-index.json").exists()


def test_control_exit_during_cleanup_records_uncertainty_before_reraise(tmp_path):
    def stop_cleanup(argv):
        raise SystemExit(4)
    harness = Harness(tmp_path, down=stop_cleanup)
    with pytest.raises(SystemExit):
        harness.run()
    saved = harness.saved()
    assert saved["state"] == "cleanup-uncertain" and saved["abort_campaign"] is True
    assert saved["cleanup"]["verified"] is False and harness.lease_path.exists()
    assert (harness.output / "artifact-index.json").exists()


def test_full_data_allowance_preserves_bounded_terminal_result_and_index(tmp_path):
    def fills(config, context):
        context.artifacts.write("fill.bin", b"x" * context.artifacts.remaining_bytes)
        return {"samples": [], "speed_observations": [], "abort_reason": None, "errors": []}
    harness = Harness(tmp_path, evaluator=fills, config_changes={"limits": {"max_artifact_bytes": 1048576}})
    result = harness.run()
    assert result.state == "failed" and result.cleanup.verified
    assert not harness.lease_path.exists() and harness.saved()["state"] == "failed"
    index = json.loads((harness.output / "artifact-index.json").read_text())
    assert any(entry["path"] == "result.json" for entry in index["artifacts"])
    assert sum(p.stat().st_size for p in harness.output.rglob("*") if p.is_file()) <= 1048576


def test_oversized_terminal_detail_is_compacted_without_metadata_escape(tmp_path):
    def many_errors(config, context):
        context.artifacts.write("fill.bin", b"x" * context.artifacts.remaining_bytes)
        return {"samples": [], "speed_observations": [], "abort_reason": None,
                "errors": [{"error": "x" * 500} for _ in range(1000)]}
    harness = Harness(tmp_path, evaluator=many_errors,
                      config_changes={"limits": {"max_artifact_bytes": 1048576}})
    result = harness.run()
    assert any("terminal_details_compacted" in warning for warning in result.warnings)
    assert harness.saved()["state"] == "failed" and result.cleanup.verified
    assert (harness.output / "artifact-index.json").exists()
    assert sum(p.stat().st_size for p in harness.output.rglob("*") if p.is_file()) <= 1048576


def test_late_health_success_cannot_pass_shared_startup_deadline(tmp_path):
    harness = Harness(tmp_path, config_changes={
        "bounds": {"candidate_wall_seconds": 600, "hash_seconds": 30, "startup_seconds": 10, "verify_seconds": 10}})
    calls = [0]
    def delayed(method, path, payload=None, **kwargs):
        calls[0] += 1
        harness.clock.now += min(5., harness.http.timeout)
        if calls[0] == 1:
            raise TransportError("503")
        return {"status": "ok"}
    harness.http.request_json = delayed
    result = harness.run()
    assert result.state == "timeout" and result.failure_stage == "ready"
    assert harness.contexts == [] and result.cleanup.verified
    start = next(stage for stage in result.stages if stage.name == "start")
    ready = next(stage for stage in result.stages if stage.name == "ready")
    assert start.elapsed_seconds + ready.elapsed_seconds <= 10
    assert ready.status == "timeout"


def test_verify_phase_uses_its_own_allowance_and_rejects_late_success(tmp_path):
    harness = Harness(tmp_path, config_changes={"bounds": {"verify_seconds": 2}})
    sample = harness.sample
    calls = [0]
    def slow_verify_sample():
        calls[0] += 1
        if calls[0] == 2:
            harness.clock.now += 3
        return sample()
    harness.runner.telemetry = slow_verify_sample
    result = harness.run()
    assert result.state == "timeout" and result.failure_stage == "verify"
    assert harness.contexts == [] and result.cleanup.verified
    first_log = next(call for call in harness.docker.calls if call[0][1] == "logs")
    assert first_log[1] <= 2


@pytest.mark.parametrize("grant", [float("nan"), float("inf"), -1, 0, True])
def test_invalid_runtime_budget_is_rejected_before_any_execution(tmp_path, grant):
    harness = Harness(tmp_path)
    with pytest.raises(ValueError, match="finite and positive"):
        harness.run(remaining_budget_seconds=grant)
    assert harness.docker.calls == []



def test_default_gpu_probe_caps_subprocess_to_remaining_phase_time(monkeypatch):
    from types import SimpleNamespace
    import llmbench.containers.runner as runner_module
    seen = []
    def command(argv, **kwargs):
        seen.append(kwargs["timeout"])
        return SimpleNamespace(stdout="0, Test GPU, 1000, 32000, 0, 30, 40")
    monkeypatch.setattr(runner_module.subprocess, "run", command)
    sample = runner_module._default_telemetry(timeout_seconds=0.25)
    assert sample["gpus"][0]["memory_used_mib"] == 1000 and seen == [0.25]
    with pytest.raises(TimeoutError):
        runner_module._default_telemetry(timeout_seconds=0)
    assert seen == [0.25]


def test_verify_and_cleanup_default_telemetry_receive_remaining_allowance(tmp_path, monkeypatch):
    import llmbench.containers.runner as runner_module
    harness = Harness(tmp_path, config_changes={"bounds": {"verify_seconds": 1, "cleanup_reserve_seconds": 15}})
    allowances = []
    def telemetry(*, timeout_seconds=3.0):
        allowances.append(timeout_seconds)
        return harness.sample()
    monkeypatch.setattr(runner_module, "_default_telemetry", telemetry)
    harness.runner.telemetry = telemetry
    def delayed_down(argv):
        harness.clock.now += 11.25
        return OK
    harness.docker.script["down"] = delayed_down
    result = harness.run()
    assert result.state == "completed" and result.cleanup.verified
    assert allowances[0] == 3.0  # ordinary admission
    assert allowances[1] == 0.5  # verification's log collection used the other half-second
    assert 0 < allowances[2] < 3.0  # cleanup log/down/queries consumed part of the cleanup grant


# ---- container mode (plan section 2.3): scripted evaluator container, never an in-process evaluator ----------

EVAL_CID = "e5a1" * 15 + "abcd"
EVAL_IMAGE = "sha256:" + "e" * 64
EVALUATOR_IMAGE = {"role": "evaluator", "reference": EVAL_IMAGE, "image_id": EVAL_IMAGE,
                   "entrypoint": ["python", "-m", "llmbench.container_eval"]}
EVALUATOR_LOG = b'{"exit_code": 0, "abort_reason": null, "samples": 4, "errors": 0}\n'
HOST_KEYS = ["image-inspect", "config", "up", "compose-ps", "port", "inspect", "inspect", "inspect", "logs", "inspect",
             "logs", "down", "ps", "network-ls"]
CONTAINER_KEYS = ["image-inspect", "image-inspect", "config", "up", "compose-ps", "inspect", "logs", "logs", "up",
                  "compose-ps", "inspect", "logs", "inspect", "logs", "down", "ps", "network-ls"]


def child_evaluation(config, **changes):
    """What a well-behaved evaluator container leaves in /artifacts/evaluation.json."""
    props = json.loads((DATA / "props-b11011.json").read_text(encoding="utf-8"))["body"]
    props["model_alias"], props["default_generation_settings"]["n_ctx"] = config.alias(), config.engine.ctx_size
    samples = [{**row, "score": 1.0, "passed": True, "status": "completed",
                "context": {"actual_context_verified": True}} for row in expected_task_rows(config.benchmarks)]
    return {"samples": samples,
            "speed_observations": [dataclasses.asdict(observation()) for _ in range(config.speed.repetitions)],
            "readback": {"props": props, "models": {}, "slots": [{"id": 0, "n_ctx": config.engine.ctx_size}]},
            "count_route": {"exact": True}, "overflow_probe": {"passed": True}, "actual_context_verified": True,
            "abort_reason": None, "errors": [], "stages": [{"name": "wait-ready", "status": "ok"}], **changes}


class ContainerHarness(Harness):
    """The evaluator is a scripted container: `up evaluator` writes what the child would, `inspect` reports its
    state, `logs` returns its log. Every argv still passes the real bounded vocabulary."""

    def __init__(self, tmp_path, *, evaluation="default", raw_evaluation=None, extra_files=None, exit_code=0,
                 oom=False, running_polls=0, startup_log=None, startup_polls=0, evaluator_log=EVALUATOR_LOG,
                 evaluator_image=None, config_changes=None, **script):
        self.evaluation, self.raw_evaluation, self.extra_files = evaluation, raw_evaluation, extra_files or {}
        self.exit_code, self.oom, self.running_polls, self.evaluator_log = exit_code, oom, running_polls, evaluator_log
        self.startup_log, self.startup_polls = startup_log, startup_polls
        self.evaluator_polls, self.startup_log_polls, self.events, self.child_written = 0, 0, [], None
        self.brokers, self.factory_args = [], []
        image = evaluator_image or EVALUATOR_IMAGE
        changes = {"evaluator": {"mode": "container", "image": image}, **(config_changes or {})}
        rows = json.dumps([{"ID": CID, "Service": "inference"}, {"ID": EVAL_CID, "Service": "evaluator"}]).encode()
        defaults = {"image-inspect": self.inspect_image, "compose-ps": WorkerResult("completed", 0, rows),
                    "inspect": self.inspect_state, "logs": self.logs, "up": self.up, "down": self.down}
        super().__init__(tmp_path, config_changes=changes, **{**defaults, **script})

    def inspect_image(self, argv):
        inference = self.config.inference_image
        image = inference if argv[-1] == inference.reference else self.config.evaluator.image
        raw = {"Id": image.image_id, "RepoDigests": [image.reference] if "@" in image.reference else [],
               "Os": "linux", "Architecture": "amd64",
               "Config": {"Entrypoint": list(image.entrypoint),
                          "Env": ["LLAMA_ARG_HOST=0.0.0.0"] if image.role == "inference" else []}}
        return WorkerResult("completed", 0, json.dumps(raw).encode())

    def inspect_state(self, argv):
        if argv[-1] == CID:
            return WorkerResult("completed", 0, b"running 0 false\n")
        assert argv[-1] == EVAL_CID
        self.evaluator_polls += 1
        if self.evaluator_polls <= self.running_polls:
            return WorkerResult("completed", 0, b"running 0 false\n")
        return WorkerResult("completed", 0, f"exited {self.exit_code} {'true' if self.oom else 'false'}\n".encode())

    def logs(self, argv):
        if argv[-1] == EVAL_CID:
            self.events.append("evaluator-logs")
            return WorkerResult("completed", 0, self.evaluator_log)
        self.startup_log_polls += 1
        partial = self.startup_log is not None and self.startup_log_polls <= self.startup_polls
        return WorkerResult("completed", 0, b"", (self.startup_log if partial else STARTUP).encode())

    def up(self, argv):
        self.events.append(f"up-{argv[-1]}")
        if argv[-1] == "evaluator":
            run_dir = Path(argv[9]).parent
            self.child_written = run_dir / "evaluator" / "evaluation.json"
            payload = self.evaluation(self.config) if callable(self.evaluation) else self.evaluation
            if self.raw_evaluation is not None:
                self.child_written.write_bytes(self.raw_evaluation)
            elif payload == "default":
                self.child_written.write_bytes(json.dumps(child_evaluation(self.config)).encode("utf-8"))
            elif payload is not None:
                self.child_written.write_bytes(json.dumps(payload).encode("utf-8"))
            for name, content in self.extra_files.items():
                target = run_dir / "evaluator" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        return OK

    def down(self, argv):
        self.events.append("down")
        return OK


class FakeBroker:
    """Plan section 2.4 protocol; `abort_campaign` mirrors what a tick or cancel reported."""

    def __init__(self, harness, *, tick_abort=False, cleanup_verified=True, abort_after_cancel=False,
                 cancel_error=None, tick_seconds=5.0):
        self.harness, self.ticks, self.cancel_calls, self.abort_campaign = harness, [], 0, False
        self.tick_abort, self.cleanup_verified = tick_abort, cleanup_verified
        self.abort_after_cancel, self.cancel_error, self.tick_seconds = abort_after_cancel, cancel_error, tick_seconds

    def tick(self, max_seconds):
        self.ticks.append(max_seconds)
        self.harness.events.append("tick")
        self.harness.clock.now += self.tick_seconds
        self.abort_campaign = self.abort_campaign or self.tick_abort
        return {"processed": 0, "pending": 0, "abort_campaign": self.tick_abort}

    def cancel_all(self):
        self.cancel_calls += 1
        self.harness.events.append("cancel_all")
        if self.cancel_error:
            raise RuntimeError(self.cancel_error)
        self.abort_campaign = self.abort_campaign or self.abort_after_cancel
        return {"cancelled": 0, "cleanup_verified": self.cleanup_verified}

    def direct_client(self):
        return "direct-client"


def wire_broker(harness, **options):
    made = []

    def factory(run_dir, config, session_lock, artifacts):
        harness.factory_args.append((run_dir, config, session_lock, artifacts))
        made.append(FakeBroker(harness, **options))
        return made[-1]
    harness.runner.broker_factory = factory
    harness.brokers = made
    return made


def artifact_hashes_match(harness, result):
    for entry in result.artifacts:
        content = (harness.output / entry.path).read_bytes()
        assert (hashlib.sha256(content).hexdigest(), len(content)) == (entry.sha256, entry.size), entry.path


def test_container_mode_ordering_logs_readiness_grant_watchdog_and_ingestion(tmp_path):
    harness = ContainerHarness(tmp_path)
    result = harness.run()
    assert result.state == "completed" and result.failure_reasons == () and result.synthetic is True
    assert harness.contexts == [] and harness.http.urls == [] and harness.http.requests == 0
    assert harness.docker.keys() == CONTAINER_KEYS
    assert harness.events == ["up-inference", "up-evaluator", "evaluator-logs", "down"]
    assert [stage.name for stage in result.stages] == ["admit", "hash", "plan", "start", "ready", "verify",
                                                       "evaluator", "verify", "cleanup", "report"]
    assert all(stage.status == "ok" for stage in result.stages)
    assert result.container_ids == (CID, EVAL_CID) and result.load_seconds == pytest.approx(56.395523)
    grant = json.loads((harness.output / "plan" / "evaluator-grant.json").read_text(encoding="utf-8"))
    # min(evaluation bound 1200, host time left after startup, both verifies and ingestion) with a 30 s slack.
    assert grant["grant_seconds"] == 1200 and grant["watchdog_slack_seconds"] == 30
    assert grant["artifact_bytes"] == harness.config.limits.evaluator_artifact_bytes == 268_435_456
    assert grant["issued_host_offset_seconds"] == pytest.approx(1.0) and grant["issued_utc"]
    compose = json.loads((harness.output / "plan" / "compose.json").read_text(encoding="utf-8"))
    evaluator = compose["services"]["evaluator"]
    assert evaluator["command"] == ["--config", "/run/llmbench/config.json", "--policy",
                                    "/run/llmbench/runtime-policy.json", "--grant-seconds", "1200",
                                    "--artifact-bytes", str(grant["artifact_bytes"])]
    assert evaluator["image"] == EVAL_IMAGE and "ports" not in compose["services"]["inference"]
    assert set(compose["networks"]) == {"bench"} and "hostlink" not in json.dumps(compose)
    launch = json.loads((harness.output / "plan" / "evaluator-launch.json").read_text(encoding="utf-8"))
    assert launch["watchdog_offset_seconds"] - launch["launched_offset_seconds"] == pytest.approx(1230)
    stage = next(stage for stage in result.stages if stage.name == "evaluator")
    assert "1200s grant" in stage.detail and "30s watchdog slack" in stage.detail
    assert (harness.output / "logs" / "evaluator.log").read_bytes() == EVALUATOR_LOG
    assert (harness.output / "logs" / "compose-up-evaluator.log").exists()
    evaluation = json.loads((harness.output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation == json.loads(harness.child_written.read_bytes())
    adoption = json.loads((harness.output / "plan" / "evaluator-adoption.json").read_text(encoding="utf-8"))
    assert adoption == {"bytes": harness.child_written.stat().st_size, "files": 1, "over_allocation": False,
                        "unindexed": [], "errors": [], "adopted_after_cleanup": False}
    names = {entry.path for entry in result.artifacts}
    assert {"plan/evaluator-config.json", "plan/runtime-policy.json", "plan/evaluator-grant.json",
            "plan/evaluator-launch.json", "plan/evaluator-adoption.json", "logs/compose-up-evaluator.log",
            "logs/evaluator.log", "logs/startup.log", "evaluator/evaluation.json", "evaluation.json",
            "settings-evidence.json", "cleanup.json", "reports/candidate.md"} <= names
    artifact_hashes_match(harness, result)
    assert result.samples_total == 4 and result.actual_context_verified is True
    assert result.speed["qualifies"] is True and result.speed["minimum_native_tps"] == pytest.approx(512 / 7.4)
    statuses = {row.control: row.status for row in result.settings}
    assert statuses["ctx_size"] == "verified-api" and statuses["fit"] == "verified-behavior"
    assert result.effective_settings_verified is True and result.image_evidence["evaluator"]["id"] == EVAL_IMAGE
    assert harness.saved()["state"] == "completed" and harness.saved()["container_ids"] == [CID, EVAL_CID]
    assert len((harness.output / "stages.jsonl").read_text(encoding="utf-8").splitlines()) == 10
    assert_clean(harness, result)


def test_container_mode_never_calls_compose_port(tmp_path):
    harness = ContainerHarness(tmp_path)
    result = harness.run()
    assert result.state == "completed" and "port" not in harness.docker.keys()
    assert not any("port" in argv for argv, _, _ in harness.docker.calls)
    assert harness.http.urls == [] and harness.docker.keys()[3:6] == ["up", "compose-ps", "inspect"]
    argv = json.loads((harness.output / "plan" / "server-argv.json").read_text(encoding="utf-8"))
    assert argv["argv"][2:6] == ["--host", "0.0.0.0", "--port", "8080"]  # bench-internal only


def test_readiness_from_logs_requires_model_loaded_and_listening(tmp_path):
    bounds = {"bounds": {"candidate_wall_seconds": 600, "hash_seconds": 30, "startup_seconds": 10,
                         "verify_seconds": 10, "evaluation_seconds": 300}}
    no_listen = STARTUP.replace("llama_server: listening on http://0.0.0.0:8080", "llama_server: binding")
    no_load = STARTUP.replace("llama_server: model loaded", "llama_server: still loading")
    assert "model loaded" in no_listen and "listening on" in no_load
    for name, log in (("no-listen", no_listen), ("no-load", no_load)):
        harness = ContainerHarness(tmp_path / name, startup_log=log, startup_polls=10**6, config_changes=bounds)
        result = harness.run()
        assert (result.state, result.failure_stage) == ("timeout", "ready"), name
        # The start stage already used 1 s of the shared 10 s startup grant; readiness never resets the timer.
        assert "did not log readiness within 9s" in result.failure_reasons[0]
        assert harness.http.requests == 0 and harness.contexts == [] and "up-evaluator" not in harness.events
        startup = [stage for stage in result.stages if stage.name in {"start", "ready"}]
        assert sum(stage.elapsed_seconds for stage in startup) <= 10 and startup[-1].status == "timeout"
        assert (harness.output / "logs" / "inference-final.log").exists()
        assert_clean(harness, result)
    late = ContainerHarness(tmp_path / "late", startup_log=no_listen, startup_polls=2, config_changes=bounds)
    result = late.run()
    assert result.state == "completed" and late.startup_log_polls >= 3
    ready = next(stage for stage in result.stages if stage.name == "ready")
    assert ready.status == "ok" and ready.elapsed_seconds >= 2  # two partial polls, then both lines
    assert result.load_seconds == pytest.approx(56.395523)
    exited = ContainerHarness(tmp_path / "exited", config_changes=bounds,
                              inspect=WorkerResult("completed", 0, b"exited 1 false\n"))
    result = exited.run()
    assert (result.state, result.failure_stage) == ("failed", "ready")
    assert "status=exited exit=1" in result.failure_reasons[0] and "up-evaluator" not in exited.events


def test_evaluator_exit_two_fails_without_ingestion(tmp_path):
    refused = b"invalid container run config: ValidationError: extra fields\n"
    harness = ContainerHarness(tmp_path / "two", exit_code=2, evaluation=None, evaluator_log=refused)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "evaluator")
    assert "refused config/policy (exit 2)" in result.failure_reasons[0]
    assert "invalid container run config" in result.failure_reasons[0]
    assert not (harness.output / "evaluation.json").exists() and not harness.child_written.exists()
    assert (harness.output / "logs" / "evaluator.log").read_bytes() == refused
    assert result.samples_total == 0 and result.actual_context_verified is False
    assert result.speed["qualifies"] is False
    assert harness.events == ["up-inference", "up-evaluator", "evaluator-logs", "down"]
    assert_clean(harness, result)
    crashed = ContainerHarness(tmp_path / "crash", exit_code=137, evaluator_log=b"Killed\n")
    result = crashed.run()
    assert result.state == "failed" and "evaluator exited 137" in result.failure_reasons[0]
    assert "Killed" in result.failure_reasons[0] and not (crashed.output / "evaluation.json").exists()
    oom = ContainerHarness(tmp_path / "oom", exit_code=137, oom=True)
    result = oom.run()
    assert result.state == "failed" and "evaluator_oom_killed" in result.failure_reasons[0]
    assert not (oom.output / "evaluation.json").exists() and result.cleanup.verified


def test_evaluator_exit_three_ingests_and_fails_on_abort_reason(tmp_path):
    def aborted(config):
        return child_evaluation(config, abort_reason="server cancellation unverified",
                                errors=[{"stage": "speed", "error_type": "TimeoutError", "error": "probe"}])
    harness = ContainerHarness(tmp_path / "abort", exit_code=3, evaluation=aborted)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "evaluator")
    assert result.failure_reasons == ("evaluator aborted: server cancellation unverified",)
    ingested = json.loads((harness.output / "evaluation.json").read_text(encoding="utf-8"))
    assert ingested["abort_reason"] == "server cancellation unverified" and len(ingested["samples"]) == 4
    assert any(warning.startswith("evaluation_error:") and "probe" in warning for warning in result.warnings)
    assert {"evaluator/evaluation.json", "logs/evaluator.log"} <= {entry.path for entry in result.artifacts}
    assert_clean(harness, result)

    def errors_only(config):
        return child_evaluation(config, errors=[{"stage": "count-route", "error": "inexact"}])
    noisy = ContainerHarness(tmp_path / "errors", exit_code=3, evaluation=errors_only)
    result = noisy.run()
    assert result.state == "completed" and any("count-route" in warning for warning in result.warnings)
    assert result.samples_total == 4


def test_watchdog_overrun_tears_down_and_reports_timeout(tmp_path):
    harness = ContainerHarness(tmp_path, running_polls=10**6, config_changes={
        "bounds": {"candidate_wall_seconds": 900, "evaluation_seconds": 120}})
    result = harness.run()
    assert (result.state, result.failure_stage) == ("timeout", "evaluator")
    assert "120s grant" in result.failure_reasons[0] and "30s watchdog slack" in result.failure_reasons[0]
    grant = json.loads((harness.output / "plan" / "evaluator-grant.json").read_text(encoding="utf-8"))
    assert grant["grant_seconds"] == 120
    stage = next(stage for stage in result.stages if stage.name == "evaluator")
    assert stage.status == "timeout" and 150 <= stage.elapsed_seconds <= 152.5
    assert harness.events[-2:] == ["evaluator-logs", "down"]  # log captured, then the project is torn down
    assert (harness.output / "logs" / "evaluator.log").read_bytes() == EVALUATOR_LOG
    assert not (harness.output / "evaluation.json").exists()  # nothing was ingested from a running child
    assert (harness.output / "plan" / "evaluator-adoption.json").exists()
    assert result.container_ids == (CID, EVAL_CID) and harness.docker.keys()[-3:] == ["down", "ps", "network-ls"]
    assert_clean(harness, result)
    assert harness.saved()["state"] == "timeout"


def test_live_child_tree_is_adopted_only_after_owned_containers_are_verified_absent(tmp_path):
    """REV-A1-02: on watchdog overrun the child is still running when the evaluator stage ends. Its tree must be
    hashed only after `compose down` and the owned-container listing prove it stopped, so the sealed index
    matches what is on disk and late writes count against the allocation."""
    changes = {"bounds": {"candidate_wall_seconds": 900, "evaluation_seconds": 120},
               "limits": {"max_artifact_bytes": 1_048_576, "evaluator_artifact_bytes": 131072}}
    harness = ContainerHarness(tmp_path / "watchdog", running_polls=10**6, config_changes=changes)

    def down(argv):
        # The child is alive until `down` stops it: whatever it writes meanwhile is what the index must seal.
        harness.child_written.write_bytes(harness.child_written.read_bytes() + b"\n" + b"late-write " * 4000)
        (harness.child_written.parent / "late.bin").write_bytes(b"L" * 300_000)  # past the 131072 allocation
        return harness.down(argv)
    harness.docker.script["down"] = down
    result = harness.run()
    assert (result.state, result.failure_stage) == ("timeout", "evaluator")
    assert "120s grant" in result.failure_reasons[0] and "watchdog slack" in result.failure_reasons[0]
    over = [reason for reason in result.failure_reasons[1:] if reason.startswith("evaluator_artifacts_over_allocation:")]
    assert len(over) == 1 and "131072 allocated" in over[0]
    assert harness.events[-2:] == ["evaluator-logs", "down"]
    adoption = json.loads((harness.output / "plan" / "evaluator-adoption.json").read_text(encoding="utf-8"))
    assert adoption["adopted_after_cleanup"] is True and adoption["over_allocation"] is True
    assert adoption["bytes"] == harness.child_written.stat().st_size + 300_000 and adoption["errors"] == []
    entries = {entry.path: entry for entry in result.artifacts}
    assert entries["evaluator/evaluation.json"].size == harness.child_written.stat().st_size
    artifact_hashes_match(harness, result)  # every sealed hash/size equals the bytes on disk
    assert sum(entry.size for entry in result.artifacts) <= 1_048_576
    assert harness.saved()["state"] == "timeout" and not (harness.output / "evaluation.json").exists()
    assert_clean(harness, result)
    # When absence is never verified the child may still be writing: its tree is never hashed, and the result
    # says so instead of certifying stale bytes.
    stuck = ContainerHarness(tmp_path / "stuck", running_polls=10**6, config_changes=changes,
                             ps=WorkerResult("completed", 0, (EVAL_CID + "\n").encode()))
    result = stuck.run()
    assert (result.state, result.abort_campaign) == ("cleanup-uncertain", True)
    assert result.failure_reasons[0].startswith("evaluator exceeded its 120s grant")
    assert any(reason.startswith("evaluator_artifacts_not_adopted:") and "not verified" in reason
               for reason in result.failure_reasons)
    assert not (stuck.output / "plan" / "evaluator-adoption.json").exists()
    assert not any(entry.path.startswith("evaluator/") for entry in result.artifacts)
    artifact_hashes_match(stuck, result)
    # A child observed exited is still reconciled inside the evaluator stage, before cleanup.
    exited = ContainerHarness(tmp_path / "exited", config_changes=changes)
    result = exited.run()
    assert result.state == "completed"
    adoption = json.loads((exited.output / "plan" / "evaluator-adoption.json").read_text(encoding="utf-8"))
    assert adoption["adopted_after_cleanup"] is False and exited.docker.keys()[-3:] == ["down", "ps", "network-ls"]


def test_adoption_errors_fail_the_candidate_instead_of_completing(tmp_path, monkeypatch):
    """REV-A1-03: an adoption failure is never a warning on a completed candidate."""
    from llmbench.containers import artifacts as artifacts_module
    limits = {"limits": {"max_artifact_bytes": 1_048_576, "evaluator_artifact_bytes": 131072}}
    original = artifacts_module.RunArtifacts._seal_external

    def flaky_seal(self, name):
        if name == "evaluator/quality/big.bin":
            raise PermissionError(13, "simulated unreadable child file", name)
        return original(self, name)
    monkeypatch.setattr(artifacts_module.RunArtifacts, "_seal_external", flaky_seal)
    harness = ContainerHarness(tmp_path / "unreadable", extra_files={"quality/big.bin": b"q" * 200_000},
                               config_changes=limits)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "evaluator")
    assert result.failure_reasons[0].startswith("evaluator_artifacts_over_allocation:")
    assert any(reason.startswith("evaluator_artifacts_not_adopted:") and "PermissionError" in reason
               for reason in result.failure_reasons)
    adoption = json.loads((harness.output / "plan" / "evaluator-adoption.json").read_text(encoding="utf-8"))
    assert adoption["over_allocation"] is True and adoption["unindexed"] == ["evaluator/quality/big.bin"]
    assert len(adoption["errors"]) == 1 and "evaluator/quality/big.bin" not in {entry.path for entry in result.artifacts}
    assert "evaluator/evaluation.json" in {entry.path for entry in result.artifacts}
    artifact_hashes_match(harness, result)
    assert harness.saved()["state"] == "failed" and not any("not_adopted" in item for item in result.warnings)
    assert_clean(harness, result)

    def broken(self, prefix):
        raise OSError(5, "input/output error")
    monkeypatch.setattr(artifacts_module.RunArtifacts, "adopt_child", broken)
    harness = ContainerHarness(tmp_path / "broken", config_changes=limits)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "evaluator")
    assert result.failure_reasons == ("evaluator_artifacts_not_adopted: OSError: [Errno 5] input/output error",)
    assert not (harness.output / "plan" / "evaluator-adoption.json").exists()
    assert not any(entry.path.startswith("evaluator/") for entry in result.artifacts)  # never re-adopted blindly
    assert harness.saved()["state"] == "failed" and result.cleanup.verified
    assert_clean(harness, result)
    # Already failing: the primary reason stays first and the adoption failure is appended, never dropped.
    refused = ContainerHarness(tmp_path / "refused", exit_code=2, evaluation=None, config_changes=limits)
    result = refused.run()
    assert result.state == "failed" and "refused config/policy" in result.failure_reasons[0]
    assert result.failure_reasons[1] == "evaluator_artifacts_not_adopted: OSError: [Errno 5] input/output error"


def test_evaluator_log_fetch_is_capped_by_the_remaining_artifact_budget(tmp_path):
    """Review note 2: an oversized child log is truncated and flagged, never a budget error on a valid run."""
    harness = ContainerHarness(tmp_path, config_changes={
        "limits": {"max_artifact_bytes": 1_048_576, "evaluator_artifact_bytes": 131072}})
    log_max_bytes = harness.config.limits.log_max_bytes
    seen = {}

    def logs(argv):
        if argv[-1] != EVAL_CID:
            return harness.logs(argv)
        cap = harness.docker.calls[-1][2]  # the max_output_bytes the runner asked for
        seen["cap"] = cap
        harness.events.append("evaluator-logs")
        return WorkerResult("output-limit", None, b"x" * cap)  # the executor cuts at the cap
    harness.docker.script["logs"] = logs
    result = harness.run()
    assert result.state == "completed" and result.failure_reasons == ()
    assert 0 < seen["cap"] < log_max_bytes and seen["cap"] <= 1_048_576 - 131072 - 262144
    assert (harness.output / "logs" / "evaluator.log").stat().st_size == seen["cap"]
    assert "evaluator_log_truncated:logs/evaluator.log" in result.warnings
    inference_logs = [call for call in harness.docker.calls if call[0][1] == "logs" and call[0][-1] == CID]
    assert inference_logs and all(call[2] == log_max_bytes for call in inference_logs)
    assert sum(path.stat().st_size for path in harness.output.rglob("*") if path.is_file()) <= 1_048_576
    artifact_hashes_match(harness, result)


def test_oversized_or_malformed_evaluation_json_is_rejected_bounded(tmp_path):
    limits = {"limits": {"max_artifact_bytes": 1_048_576, "evaluator_artifact_bytes": 65536}}
    big = ContainerHarness(tmp_path / "big", raw_evaluation=b"{" + b" " * 70_000 + b"}", config_changes=limits)
    result = big.run()
    assert (result.state, result.failure_stage) == ("failed", "evaluator")
    assert result.failure_reasons[0].startswith("evaluation.json rejected:")
    assert "allocation is 65536" in result.failure_reasons[0]
    assert "evaluator_artifacts_over_allocation: 70002 bytes" in result.failure_reasons[1]
    assert all(len(reason) < 400 for reason in result.failure_reasons)  # never echoes the file
    saved = json.loads((big.output / "evaluation.json").read_text(encoding="utf-8"))
    assert list(saved) == ["error"] and saved["error"].startswith("evaluation.json rejected")
    assert result.samples_total == 0 and result.cleanup.verified
    for name, raw, fragment in (("bad", b'{"samples": [}', "not strict UTF-8 JSON"),
                                ("utf", b"\xff\xfe{}", "not strict UTF-8 JSON"),
                                ("list", b"[]", "must be a JSON object"),
                                ("keys", b'{"samples": []}', "lacks required keys")):
        harness = ContainerHarness(tmp_path / name, raw_evaluation=raw, config_changes=limits)
        result = harness.run()
        assert (result.state, result.failure_stage) == ("failed", "evaluator"), name
        assert fragment in result.failure_reasons[0] and len(result.failure_reasons) == 1, name
        assert_clean(harness, result)
    absent = ContainerHarness(tmp_path / "absent", evaluation=None)  # exit 0 but nothing written
    result = absent.run()
    assert result.state == "failed" and "unreadable" in result.failure_reasons[0]
    assert_clean(absent, result)


def test_missing_denominator_rows_in_evaluation_fail_the_candidate(tmp_path):
    def short(config):
        payload = child_evaluation(config)
        payload["samples"] = payload["samples"][:-1]
        return payload
    harness = ContainerHarness(tmp_path / "short", evaluation=short)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "evaluator")
    assert "denominator mismatch" in result.failure_reasons[0]
    assert "missing=['niah/multi/development/seed-42']" in result.failure_reasons[0]
    assert json.loads((harness.output / "evaluation.json").read_text(encoding="utf-8"))["error"]
    assert result.samples_total == 0 and result.actual_context_verified is False
    assert_clean(harness, result)

    def duplicated(config):
        payload = child_evaluation(config)
        payload["samples"].append(payload["samples"][0])
        return payload
    result = ContainerHarness(tmp_path / "dup", evaluation=duplicated).run()
    assert result.state == "failed" and "duplicated=['tools/nested-exact-v1']" in result.failure_reasons[0]

    def foreign(config):
        payload = child_evaluation(config)
        payload["samples"].append({**payload["samples"][0], "task_id": "tools/invented"})
        return payload
    result = ContainerHarness(tmp_path / "foreign", evaluation=foreign).run()
    assert result.state == "failed" and "unexpected=['tools/invented']" in result.failure_reasons[0]


def test_child_allocation_is_carved_from_parent_cap(tmp_path, monkeypatch):
    from llmbench.containers import artifacts as artifacts_module
    seen = []
    reserve_child, adopt_child = artifacts_module.RunArtifacts.reserve_child, artifacts_module.RunArtifacts.adopt_child

    def reserve(self, prefix, max_bytes):
        before = self.remaining_bytes
        reserve_child(self, prefix, max_bytes)
        seen.append(("reserve", prefix, max_bytes, before - self.remaining_bytes, len(harness.docker.calls)))

    def adopt(self, prefix):
        before = self.remaining_bytes
        report = adopt_child(self, prefix)
        seen.append(("adopt", prefix, report["bytes"], self.remaining_bytes - before, len(harness.docker.calls)))
        return report
    monkeypatch.setattr(artifacts_module.RunArtifacts, "reserve_child", reserve)
    monkeypatch.setattr(artifacts_module.RunArtifacts, "adopt_child", adopt)
    limits = {"limits": {"max_artifact_bytes": 1_048_576, "evaluator_artifact_bytes": 131072}}
    harness = ContainerHarness(tmp_path / "fits", config_changes=limits)
    result = harness.run()
    assert result.state == "completed" and [row[0] for row in seen] == ["reserve", "adopt"]
    _, prefix, allocation, charged, calls_before = seen[0]
    assert (prefix, allocation, charged) == ("evaluator", 131072, 131072)
    up = harness.docker.calls[calls_before][0]  # the very next Docker call starts the child
    assert up[12:] == ("up", "--detach", "--no-build", "--pull", "never", "evaluator")
    _, _, written, released, calls_after = seen[1]
    assert written == harness.child_written.stat().st_size and released == 131072 - written
    assert harness.docker.keys()[calls_after - 1] == "logs"  # adopted after the child exited and its log was read
    assert sum(path.stat().st_size for path in harness.output.rglob("*") if path.is_file()) <= 1_048_576
    artifact_hashes_match(harness, result)
    seen.clear()
    # A child that writes past its allocation fails the candidate (never the cleanup); the parent cap holds.
    harness = ContainerHarness(tmp_path / "over", extra_files={"quality/big.bin": b"q" * 200_000},
                               config_changes=limits)
    result = harness.run()
    assert (result.state, result.failure_stage) == ("failed", "evaluator")
    assert result.failure_reasons[0].startswith("evaluator_artifacts_over_allocation:")
    assert "131072 allocated" in result.failure_reasons[0]
    adoption = json.loads((harness.output / "plan" / "evaluator-adoption.json").read_text(encoding="utf-8"))
    assert adoption["over_allocation"] is True and adoption["files"] == 2 and adoption["unindexed"] == []
    assert seen[1][3] < 0  # the child kept more than it was granted; the parent charged what it actually found
    assert sum(path.stat().st_size for path in harness.output.rglob("*") if path.is_file()) <= 1_048_576
    assert_clean(harness, result)
    with pytest.raises(Exception, match="terminal reserve"):
        ContainerHarness(tmp_path / "bad", config_changes={"limits": {"max_artifact_bytes": 1_048_576,
                                                                       "evaluator_artifact_bytes": 800_000}})


def test_broker_factory_ticks_during_wait_and_cancels_at_cleanup(tmp_path):
    harness = ContainerHarness(tmp_path / "container", running_polls=3, config_changes={"broker": {}})
    wire_broker(harness)
    result = harness.run()
    broker = harness.brokers[0]
    assert result.state == "completed" and result.failure_reasons == ()
    assert [stage.name for stage in result.stages] == ["admit", "hash", "plan", "start", "ready", "verify", "broker",
                                                       "evaluator", "verify", "cleanup", "report"]
    assert len(broker.ticks) == 3 and all(0 < seconds <= 300 for seconds in broker.ticks)
    assert harness.events == ["up-inference", "up-evaluator", "tick", "tick", "tick", "evaluator-logs", "cancel_all",
                              "down"]
    assert broker.cancel_calls == 1 and harness.docker.keys()[-3:] == ["down", "ps", "network-ls"]
    run_dir, config, lock, scoped = harness.factory_args[0]
    assert run_dir == harness.output.resolve() and config is harness.config and lock.allow_container_execution
    assert scoped.prefix == "broker" and scoped.root == harness.output.resolve() / "broker"
    assert (harness.output / "spool" / "requests").is_dir() and (harness.output / "spool" / "results").is_dir()
    compose = json.loads((harness.output / "plan" / "compose.json").read_text(encoding="utf-8"))
    mounts = {mount["target"]: mount for mount in compose["services"]["evaluator"]["volumes"]}
    assert Path(mounts["/spool/requests"]["source"]) == harness.output.resolve() / "spool" / "requests"
    assert "read_only" not in mounts["/spool/requests"] and mounts["/spool/results"]["read_only"] is True
    assert len(json.loads((harness.output / "broker" / "ticks.json").read_text(encoding="utf-8"))) == 3
    assert json.loads((harness.output / "broker" / "cancel-all.json").read_text(encoding="utf-8"))["cleanup_verified"]
    assert {"broker/ticks.json", "broker/cancel-all.json"} <= {entry.path for entry in result.artifacts}
    assert_clean(harness, result)
    # Host-process mode: the same factory supplies the evaluator's direct client; the Docker sequence is unchanged.
    host = Harness(tmp_path / "host", config_changes={"broker": {}})
    host.events, host.brokers, host.factory_args = [], [], []
    wire_broker(host)
    result = host.run()
    assert result.state == "completed" and host.docker.keys() == HOST_KEYS
    assert host.contexts[0].coding_client == "direct-client" and host.brokers[0].ticks == []
    assert host.brokers[0].cancel_calls == 1 and not (host.output / "spool").exists()
    assert [stage.name for stage in result.stages] == ["admit", "hash", "plan", "start", "ready", "verify", "broker",
                                                       "quality", "verify", "cleanup", "report"]
    # config.broker without a wired factory is rejected before any Docker call.
    bare = ContainerHarness(tmp_path / "bare", config_changes={"broker": {}})
    result = bare.run()
    assert (result.state, result.failure_stage) == ("rejected", "admit") and bare.docker.calls == []
    assert "no coding broker factory" in result.failure_reasons[0]
    # A factory returning something without the protocol fails the broker stage before the evaluator starts;
    # whatever it may have started cannot be cancelled or verified absent, so the cleanup is uncertain (closed).
    partial = ContainerHarness(tmp_path / "partial", config_changes={"broker": {}})
    partial.runner.broker_factory = lambda *args: object()
    result = partial.run()
    assert result.failure_reasons[0] == "broker factory returned an object without tick()"
    assert next(stage for stage in result.stages if stage.name == "broker").status == "failed"
    assert "up-evaluator" not in partial.events and result.state == "cleanup-uncertain" and result.abort_campaign
    assert "cancel_all failed" in result.cleanup.error and "down" in partial.events


def test_broker_cleanup_uncertainty_aborts_campaign(tmp_path):
    for name, options in (("unverified", {"cleanup_verified": False}), ("abort", {"abort_after_cancel": True}),
                          ("error", {"cancel_error": "worker rm failed"})):
        harness = ContainerHarness(tmp_path / name, config_changes={"broker": {}})
        wire_broker(harness, **options)
        result = harness.run()
        assert (result.state, result.abort_campaign, result.cleanup.verified) == ("cleanup-uncertain", True, False), name
        assert result.cleanup.lease_retained and harness.lease_path.exists() and "broker" in result.cleanup.error
        assert result.failure_stage == "cleanup" and "owned_resource_absence_unverified" in result.failure_reasons[0]
        assert harness.events[-2:] == ["cancel_all", "down"]  # compose down still runs after the broker cancel
        assert harness.saved()["abort_campaign"] is True
    ticking = ContainerHarness(tmp_path / "tick", running_polls=10, config_changes={"broker": {}})
    wire_broker(ticking, tick_abort=True)
    result = ticking.run()
    assert result.failure_reasons[0] == "coding broker requested a campaign abort during the evaluation"
    assert next(stage for stage in result.stages if stage.name == "evaluator").status == "failed"
    assert len(ticking.brokers[0].ticks) == 1
    assert ticking.events[-4:] == ["tick", "evaluator-logs", "cancel_all", "down"]
    # The abort outlives the candidate: cleanup is uncertain, so the campaign stops and the lease is retained.
    assert (result.state, result.failure_stage, result.abort_campaign) == ("cleanup-uncertain", "cleanup", True)
    assert "campaign abort" in result.cleanup.error and ticking.lease_path.exists()


def normalized_calls(harness, result):
    rows = []
    for argv, timeout, max_output in harness.docker.calls:
        items = [item.replace(str(harness.output.resolve()), "RUN_DIR").replace(result.project_name, "PROJECT")
                 .replace("\\", "/") for item in argv]
        rows.append({"argv": items, "timeout_seconds": timeout, "max_output_bytes": max_output})
    return rows


def test_host_process_mode_unchanged_golden_call_sequence(tmp_path):
    harness = Harness(tmp_path)
    result = harness.run()
    golden = json.loads((DATA / "evaluator-host-process-golden.json").read_text(encoding="utf-8"))
    assert result.state == "completed" and harness.docker.keys() == HOST_KEYS
    assert normalized_calls(harness, result) == golden["docker_calls"]
    assert [stage.name for stage in result.stages] == golden["stages"]
    assert sorted(entry.path for entry in result.artifacts) == golden["artifacts"]
    assert sorted(vars(harness.contexts[0])) == golden["context_fields"]
    assert harness.contexts[0].coding_hook is None and harness.contexts[0].coding_client is None
    compose = json.loads((harness.output / "plan" / "compose.json").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"inference"} and set(compose["networks"]) == {"bench", "hostlink"}
    assert not (harness.output / "plan" / "evaluator-grant.json").exists()
    assert not (harness.output / "plan" / "runtime-policy.json").exists()


def test_evaluator_image_identity_and_entrypoint_are_verified_at_admit(tmp_path):
    wrong_entry = ContainerHarness(tmp_path / "entry", evaluator_image={**EVALUATOR_IMAGE,
                                                                        "entrypoint": ["python", "-m", "other"]})
    result = wrong_entry.run()
    assert (result.state, result.failure_stage) == ("rejected", "admit") and wrong_entry.docker.calls == []
    assert "entrypoint must be python -m llmbench.container_eval" in result.failure_reasons[0]
    assert not wrong_entry.lease_path.exists() and wrong_entry.saved()["cleanup"]["attempted"] is False

    def scripted(field, value):
        def inspect(argv):
            if argv[-1] != EVAL_IMAGE:
                return harness.inspect_image(argv)
            raw = {"Id": EVAL_IMAGE, "RepoDigests": [], "Os": "linux", "Architecture": "amd64",
                   "Config": {"Entrypoint": ["python", "-m", "llmbench.container_eval"], "Env": []}}
            raw[field] = value
            return WorkerResult("completed", 0, json.dumps(raw).encode())
        return inspect
    for name, field, value, fragment in (("id", "Id", "sha256:" + "f" * 64, "image Id"),
                                         ("arch", "Architecture", "arm64", "platform"),
                                         ("entry", "Config", {"Entrypoint": ["python", "-m", "other"], "Env": []},
                                          "entrypoint")):
        harness = ContainerHarness(tmp_path / f"image-{name}", **{"image-inspect": scripted(field, value)})
        result = harness.run()
        assert (result.state, result.failure_stage) == ("rejected", "admit"), name
        assert harness.docker.keys() == ["image-inspect", "image-inspect"] and fragment in result.failure_reasons[0]
        assert not harness.lease_path.exists() and result.cleanup.attempted is False and harness.events == []
    missing = ContainerHarness(tmp_path / "missing", **{"image-inspect": lambda argv: (
        missing.inspect_image(argv) if argv[-1] != EVAL_IMAGE
        else WorkerResult("completed", 1, b"", b"No such image"))})
    result = missing.run()
    assert result.state == "rejected" and "not available locally" in result.failure_reasons[0]
    assert missing.docker.keys() == ["image-inspect", "image-inspect"]


def test_policy_copy_is_written_before_up_and_mounted_read_only(tmp_path):
    lock = SessionLock(True, True, True, reason="stage-2 container test")
    seen = {}

    def up(argv):
        if argv[-1] == "inference":
            run_dir = Path(argv[9]).parent
            seen["policy"] = (run_dir / "plan" / "runtime-policy.json").read_bytes()
            seen["config"] = (run_dir / "plan" / "evaluator-config.json").read_bytes()
            seen["grant"] = json.loads((run_dir / "plan" / "evaluator-grant.json").read_text(encoding="utf-8"))
            seen["calls"] = len(harness.docker.calls) - 1
        return harness.up(argv)
    harness = ContainerHarness(tmp_path, lock=lock, up=up)
    result = harness.run()
    assert result.state == "completed"
    assert seen["policy"] == (canonical_json(lock.to_json()) + "\n").encode("utf-8")
    # The copy an evaluator container reads is the pre-native policy shape: an image built from an older wheel
    # rejects unknown keys, so the native permission is written only when it is granted.
    assert b"allow_native_execution" not in seen["policy"]
    assert SessionLock.read(harness.output / "plan" / "runtime-policy.json") == lock
    assert seen["config"] == (canonical_json(harness.config.model_dump(mode="json")) + "\n").encode("utf-8")
    assert read_run_config(harness.output / "plan" / "evaluator-config.json") == harness.config
    assert seen["grant"]["grant_seconds"] == 1200
    assert harness.docker.keys()[seen["calls"]] == "up" and "up" not in harness.docker.keys()[:seen["calls"]]
    compose = json.loads((harness.output / "plan" / "compose.json").read_text(encoding="utf-8"))
    evaluator = compose["services"]["evaluator"]
    mounts = {mount["target"]: mount for mount in evaluator["volumes"]}
    plan_dir = harness.output.resolve() / "plan"
    assert mounts["/run/llmbench/runtime-policy.json"] == {
        "type": "bind", "source": str(plan_dir / "runtime-policy.json"), "target": "/run/llmbench/runtime-policy.json",
        "read_only": True, "bind": {"create_host_path": False}}
    assert mounts["/run/llmbench/config.json"]["source"] == str(plan_dir / "evaluator-config.json")
    assert mounts["/run/llmbench/config.json"]["read_only"] is True and "read_only" not in mounts["/artifacts"]
    assert set(mounts) == {"/run/llmbench/config.json", "/run/llmbench/runtime-policy.json", "/artifacts"}
    assert evaluator["read_only"] is True and evaluator["user"] == "10001:10001" and "environment" not in evaluator
    assert evaluator["command"][:4] == ["--config", "/run/llmbench/config.json", "--policy",
                                        "/run/llmbench/runtime-policy.json"]
    entries = {entry.path: entry for entry in result.artifacts}
    assert entries["plan/runtime-policy.json"].sha256 == hashlib.sha256(seen["policy"]).hexdigest()
