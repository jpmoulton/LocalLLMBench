"""Docker sandbox gating for runs without a usable worker (the metal-native Mac path without Colima).

Covers the read-only sandbox probe, the never-launched worker result, the broker namespace a native run publishes,
and the evaluator's blocked execution suites. No Docker, model, GPU or network: every Docker client is a scripted
fake or the real bounded executor over a fake ``Popen``; a tripwire proves nothing real is ever spawned.
"""

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from llmbench.coding import sandbox
from llmbench.coding.broker import HostBroker
from llmbench.coding.broker_client import ExecutionUnavailable, UnavailableClient
from llmbench.coding.sandbox import (
    IMAGE_PLATFORM_ARGV, PROBE_OUTPUT_BYTES, PROBE_TIMEOUT_SECONDS, SERVER_PLATFORM_ARGV, BoundedProcessExecutor,
    DockerWorker, WorkerResult, inspect_local_image, probe_docker_sandbox,
)
from llmbench.config import RunMode
from llmbench.containers.artifacts import RunArtifacts
from llmbench.containers.config import ImageRef
from llmbench.safety import SessionLock

from test_coding_broker import FIXTURE, LOCK, broker_config, drop, ledger, make_broker, make_request, read_result

WORKER_ID = "sha256:" + "c" * 64
ARM_WORKER = ImageRef(role="worker", reference=WORKER_ID, image_id=WORKER_ID, platform="linux/arm64",
                      entrypoint=("python3",))
AMD_WORKER = ImageRef(role="worker", reference=WORKER_ID, image_id=WORKER_ID, entrypoint=("python3",))
ALLOWED = SessionLock(allow_container_execution=True)
DAEMON_DOWN = (b"Cannot connect to the Docker daemon at unix:///Users/example/.colima/default/docker.sock. "
               b"Is the docker daemon running?")


@pytest.fixture
def no_real_docker(monkeypatch):
    """Any attempt to spawn a real process fails the test."""
    def tripwire(*args, **kwargs):
        raise AssertionError(f"a real process was spawned: {args!r}")
    monkeypatch.setattr(subprocess, "Popen", tripwire)
    monkeypatch.setattr(sandbox.subprocess, "Popen", tripwire)


class ProbeExecutor:
    """Scripted Docker client that also enforces the real allowlist on every argv it is handed."""

    def __init__(self, version=None, image=None):
        self.version = version or WorkerResult("completed", 0, b"linux/arm64\n")
        self.image = image or WorkerResult("completed", 0, f"{WORKER_ID} linux/arm64\n".encode())
        self.calls = []

    def run(self, argv, *, timeout_seconds, max_output_bytes):
        BoundedProcessExecutor._validate(argv)
        self.calls.append((argv, timeout_seconds, max_output_bytes))
        if argv == SERVER_PLATFORM_ARGV:
            return self.version
        if argv[:5] == IMAGE_PLATFORM_ARGV:
            return self.image
        raise AssertionError(argv)


class FakeProcess:
    def __init__(self, stdout=b"", stderr=b"", code=0):
        self.stdout, self.stderr, self.code = io.BytesIO(stdout), io.BytesIO(stderr), code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        pass


# ---- the probe ---------------------------------------------------------------------------------------------------

def test_available_only_after_every_check_was_observed_with_two_bounded_reads():
    executor = ProbeExecutor()
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER, executor=executor)
    assert report["status"] == "available" and report["reason"] is None and report["blocked_by"] is None
    assert report["server_os_arch"] == "linux/arm64" and report["image_platform"] == "linux/arm64"
    assert report["image_id"] == WORKER_ID
    assert set(report) == {"status", "reason", "blocked_by", "docker_cli", "server_os_arch", "image_id",
                           "image_platform"}
    assert executor.calls == [(SERVER_PLATFORM_ARGV, PROBE_TIMEOUT_SECONDS, PROBE_OUTPUT_BYTES),
                              ((*IMAGE_PLATFORM_ARGV, WORKER_ID), PROBE_TIMEOUT_SECONDS, PROBE_OUTPUT_BYTES)]
    assert json.loads(json.dumps(report, allow_nan=False)) == report  # stored verbatim in bundles and run dirs
    # A bare content-addressed id is accepted too (no pinned platform to compare, the daemon's is still required).
    assert probe_docker_sandbox(session_lock=ALLOWED, image_ref=WORKER_ID,
                                executor=ProbeExecutor())["status"] == "available"


def test_policy_that_forbids_containers_blocks_before_any_client_is_spawned(no_real_docker):
    executor = ProbeExecutor()
    for lock in (SessionLock(), SessionLock(allow_inference=True, allow_native_execution=True)):
        report = probe_docker_sandbox(session_lock=lock, image_ref=ARM_WORKER, executor=executor)
        assert report["status"] == "blocked" and report["blocked_by"] == "policy"
        assert "forbids container execution" in report["reason"]
    assert executor.calls == []
    assert probe_docker_sandbox(session_lock=SessionLock(), image_ref=ARM_WORKER)["blocked_by"] == "policy"


def test_missing_docker_cli_is_blocked_without_spawning_anything(monkeypatch, no_real_docker):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER)
    assert report == {"status": "blocked", "reason": "the Docker CLI was not found on PATH", "blocked_by": "docker_cli",
                      "docker_cli": None, "server_os_arch": None, "image_id": None, "image_platform": None}


def test_a_client_that_cannot_be_started_is_a_missing_cli_not_a_daemon_problem():
    spawned = []

    def cannot_spawn(argv, **kwargs):
        spawned.append(argv)
        raise FileNotFoundError(2, "No such file or directory", "docker")

    executor = BoundedProcessExecutor(session_lock=ALLOWED, mode=RunMode.LIVE, popen_factory=cannot_spawn)
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER, executor=executor)
    assert report["blocked_by"] == "docker_cli" and "No such file or directory" in report["reason"]
    assert spawned == [SERVER_PLATFORM_ARGV]


@pytest.mark.parametrize("version, words", [
    (WorkerResult("completed", 1, b"", DAEMON_DOWN), "Is the docker daemon running?"),
    (WorkerResult("timeout", None), "timeout"),
    (WorkerResult("completed", 0, b"<no value>/<no value>\n"), "no server os/arch"),
    (WorkerResult("completed", 0, b""), "no server os/arch"),
])
def test_an_unreachable_or_mute_daemon_blocks_and_the_image_is_never_asked_about(version, words):
    executor = ProbeExecutor(version=version)
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER, executor=executor)
    assert report["status"] == "blocked" and report["blocked_by"] == "daemon" and words in report["reason"]
    assert report["server_os_arch"] is None and report["image_id"] is None
    assert [argv for argv, *_ in executor.calls] == [SERVER_PLATFORM_ARGV]


def test_worker_image_must_be_pinned_present_exact_and_native_to_the_daemon():
    executor = ProbeExecutor()
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=None, executor=executor)
    assert report["blocked_by"] == "worker_image" and "no pinned sandbox worker image" in report["reason"]
    assert report["server_os_arch"] == "linux/arm64" and len(executor.calls) == 1  # the daemon fact is still kept

    absent = ProbeExecutor(image=WorkerResult("completed", 1, b"", f"Error: No such image: {WORKER_ID}".encode()))
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER, executor=absent)
    assert report["blocked_by"] == "worker_image" and "not present locally" in report["reason"]
    assert "No such image" in report["reason"] and report["image_id"] is None

    other = ProbeExecutor(image=WorkerResult("completed", 0, f"sha256:{'d' * 64} linux/arm64".encode()))
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER, executor=other)
    assert report["blocked_by"] == "worker_image" and "answered for sha256:dddd" in report["reason"]

    # An answer that is not "<id> <os>/<arch>" is not presence, whatever the exit code said.
    for garbage in (b"", b"<no value> linux/arm64", f"{WORKER_ID} linux/arm64 extra".encode(), WORKER_ID.encode()):
        report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER,
                                      executor=ProbeExecutor(image=WorkerResult("completed", 0, garbage)))
        assert report["blocked_by"] == "worker_image" and "unreadable image inspection" in report["reason"]
        assert report["image_id"] is None and report["image_platform"] is None

    unanswered = ProbeExecutor(image=WorkerResult("timeout", None))
    assert probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER,
                                executor=unanswered)["blocked_by"] == "daemon"

    # Pinned linux/arm64 but the local image is amd64: the pin is violated, whatever the daemon could emulate.
    drifted = ProbeExecutor(image=WorkerResult("completed", 0, f"{WORKER_ID} linux/amd64".encode()))
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER, executor=drifted)
    assert report["blocked_by"] == "worker_image" and report["reason"] == (
        "the worker image is linux/amd64, not its pinned linux/arm64")
    assert report["image_platform"] == "linux/amd64"  # what was inspected is recorded, not what was intended

    # A correctly pinned amd64 worker on an Apple Silicon VM would only run emulated: blocked, never guessed.
    foreign = ProbeExecutor(image=WorkerResult("completed", 0, f"{WORKER_ID} linux/amd64".encode()))
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=AMD_WORKER, executor=foreign)
    assert report["status"] == "blocked" and report["blocked_by"] == "worker_image"
    assert "cannot run natively on the linux/arm64 Docker daemon" in report["reason"]
    # The NVIDIA host shape (amd64 worker on an amd64 daemon) stays available.
    native_amd = ProbeExecutor(version=WorkerResult("completed", 0, b"linux/amd64"),
                               image=WorkerResult("completed", 0, f"{WORKER_ID} linux/amd64".encode()))
    assert probe_docker_sandbox(session_lock=ALLOWED, image_ref=AMD_WORKER, executor=native_amd)["status"] == "available"


def test_malformed_image_references_are_programming_errors():
    inference = ImageRef(role="inference", reference=WORKER_ID, image_id=WORKER_ID, entrypoint=("llama-server",),
                         build_info="b11011-aa39d7a3e", help_sha256="e" * 64)
    for bad in (inference, "python:3.12", "local/worker@" + WORKER_ID, SimpleNamespace(image_id="latest"), 7):
        with pytest.raises(ValueError):
            probe_docker_sandbox(session_lock=ALLOWED, image_ref=bad, executor=ProbeExecutor())
    with pytest.raises(ValueError):
        inspect_local_image(ProbeExecutor(), "worker:latest")


def test_probe_argv_are_the_only_new_allowlisted_shapes():
    BoundedProcessExecutor._validate(SERVER_PLATFORM_ARGV)
    BoundedProcessExecutor._validate((*IMAGE_PLATFORM_ARGV, WORKER_ID))
    for argv in (("docker", "version"), (*SERVER_PLATFORM_ARGV, "extra"),
                 ("docker", "version", "--format", "{{json .}}"),
                 ("docker", "image", "inspect", "--format", "{{json .}}", WORKER_ID),
                 (*IMAGE_PLATFORM_ARGV, "python:latest"), (*IMAGE_PLATFORM_ARGV, "local/worker@" + WORKER_ID),
                 (*IMAGE_PLATFORM_ARGV, WORKER_ID, WORKER_ID), (*IMAGE_PLATFORM_ARGV, "sha256:" + "C" * 64),
                 ("docker", "image", "pull", WORKER_ID), ("docker", "info"), IMAGE_PLATFORM_ARGV):
        with pytest.raises(ValueError):
            BoundedProcessExecutor._validate(argv)


def test_probe_through_the_real_bounded_executor_and_its_policy_check():
    spawned = []
    answers = {"version": FakeProcess(b"linux/arm64\n"),
               "image": FakeProcess(f"{WORKER_ID} linux/arm64\n".encode())}

    def popen(argv, **kwargs):
        spawned.append((argv, kwargs["shell"], kwargs["stdin"]))
        return answers[argv[1]]

    executor = BoundedProcessExecutor(session_lock=ALLOWED, mode=RunMode.LIVE, popen_factory=popen)
    report = probe_docker_sandbox(session_lock=ALLOWED, image_ref=ARM_WORKER, executor=executor)
    assert report["status"] == "available" and report["image_platform"] == "linux/arm64"
    assert spawned == [(SERVER_PLATFORM_ARGV, False, subprocess.DEVNULL),
                       ((*IMAGE_PLATFORM_ARGV, WORKER_ID), False, subprocess.DEVNULL)]
    found = inspect_local_image(BoundedProcessExecutor(session_lock=ALLOWED, mode=RunMode.LIVE,
                                                       popen_factory=lambda argv, **kw: FakeProcess(
                                                           f"{WORKER_ID} linux/amd64".encode())), WORKER_ID)
    assert found == {"present": True, "image_id": WORKER_ID, "platform": "linux/amd64", "detail": None,
                     "status": "completed", "returncode": 0, "launched": True}


# ---- a worker whose Docker client never started -------------------------------------------------------------------

class LaunchlessExecutor:
    """The executor contract's pre-spawn answer (CLI absent) for `run`; anything else would be a cleanup call."""

    def __init__(self, result=None):
        self.result = result or WorkerResult("environment-error", None, stderr=b"Docker CLI was not found",
                                             launched=False)
        self.calls = []

    def run(self, argv, **limits):
        self.calls.append(argv)
        if argv[1] == "run":
            return self.result
        return WorkerResult("completed", 1, stderr=b"Cannot connect to the Docker daemon")


def launchless_job(tmp_path, executor, *, collect_result):
    root = tmp_path / "runs"
    (root / "case").mkdir(parents=True)
    worker = DockerWorker(allowed_root=root, session_lock=ALLOWED, mode=RunMode.LIVE, executor=executor)
    return worker, worker.prepare(root / "case", image="local/worker@" + WORKER_ID, command=("python3", "/d.py"),
                                  collect_result=collect_result)


@pytest.mark.parametrize("collect_result", [False, True])
def test_never_launched_run_is_sandbox_unavailable_with_nothing_to_clean(tmp_path, collect_result):
    executor = LaunchlessExecutor()
    worker, job = launchless_job(tmp_path, executor, collect_result=collect_result)
    result = worker.run(job)
    assert result.status == "sandbox-unavailable" and result.returncode is None and result.launched is False
    assert result.cleanup_confirmed is True and result.cleanup_error is None
    assert result.stderr == b"Docker CLI was not found" and result.result_bytes is None
    assert [argv[1] for argv in executor.calls] == ["run"]  # no rm, ps, exec, inspect or logs


def test_daemon_down_after_launch_keeps_the_unverified_cleanup(tmp_path):
    executor = LaunchlessExecutor(WorkerResult("completed", 125, stderr=DAEMON_DOWN))
    worker, job = launchless_job(tmp_path, executor, collect_result=True)
    result = worker.run(job)
    assert result.status == "completed" and result.returncode == 125 and result.launched is True
    assert result.cleanup_confirmed is False and result.cleanup_error == "Owned container absence could not be verified"
    assert [argv[1] for argv in executor.calls] == ["run", "rm", "ps"]


def test_a_launched_false_claim_with_an_exit_code_is_not_trusted(tmp_path):
    # Only a pre-spawn answer (no return code) proves nothing ran; anything else takes the full cleanup path.
    executor = LaunchlessExecutor(WorkerResult("environment-error", 1, launched=False))
    worker, job = launchless_job(tmp_path, executor, collect_result=False)
    result = worker.run(job)
    assert result.status == "environment-error" and result.launched is True and result.cleanup_confirmed is False
    assert [argv[1] for argv in executor.calls] == ["run", "rm", "ps"]


def test_real_executor_marks_only_pre_spawn_returns_as_not_launched(monkeypatch, tmp_path, no_real_docker):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    missing = BoundedProcessExecutor(session_lock=ALLOWED, mode=RunMode.LIVE)
    assert missing.run(SERVER_PLATFORM_ARGV, timeout_seconds=5, max_output_bytes=4096) == WorkerResult(
        "environment-error", None, stderr=b"Docker CLI was not found", launched=False)

    def refuse(argv, **kwargs):
        raise PermissionError(13, "Permission denied", "docker")
    denied = BoundedProcessExecutor(session_lock=ALLOWED, mode=RunMode.LIVE, popen_factory=refuse)
    result = denied.run(SERVER_PLATFORM_ARGV, timeout_seconds=5, max_output_bytes=4096)
    assert result.launched is False and result.returncode is None and b"Permission denied" in result.stderr
    ran = BoundedProcessExecutor(session_lock=ALLOWED, mode=RunMode.LIVE,
                                 popen_factory=lambda argv, **kw: FakeProcess(b"", DAEMON_DOWN, 1))
    result = ran.run(SERVER_PLATFORM_ARGV, timeout_seconds=5, max_output_bytes=4096)
    assert result.launched is True and result.returncode == 1  # a client that ran and failed DID launch

    # End to end: the default worker executor on a host without the Docker CLI spawns nothing and cleans nothing.
    root = tmp_path / "runs"
    (root / "case").mkdir(parents=True)
    worker = DockerWorker(allowed_root=root, session_lock=ALLOWED, mode=RunMode.LIVE)
    job = worker.prepare(root / "case", image="local/worker@" + WORKER_ID, command=("python3", "/d.py"))
    outcome = worker.run(job)
    assert (outcome.status, outcome.cleanup_confirmed, outcome.launched) == ("sandbox-unavailable", True, False)


def test_broker_publishes_a_missing_cli_as_environment_error_not_a_model_zero_or_an_abort(tmp_path):
    spawned = []

    def cannot_spawn(argv, **kwargs):
        spawned.append(argv)
        raise FileNotFoundError(2, "No such file or directory", "docker")

    executor = BoundedProcessExecutor(session_lock=LOCK, mode=RunMode.LIVE, popen_factory=cannot_spawn)
    broker, _, _ = make_broker(tmp_path, executor=executor)
    request = make_request(broker)  # the reference solution: it would pass if a worker existed
    drop(broker, request)
    summary = broker.tick(300)
    result = read_result(broker, request.request_id)
    assert result.status == "environment-error" and result.sample is None
    assert result.failure_reason.startswith("sandbox_unavailable: the Docker client could not be started")
    assert result.cleanup_confirmed and not result.abort_campaign and summary["abort_campaign"] is False
    assert len(spawned) == len(FIXTURE.checks) and all(argv[1] == "run" for argv in spawned)
    finished = [row for row in ledger(broker) if row["event"] == "finished"]
    assert [row["status"] for row in finished] == ["environment-error"]
    assert broker.cancel_all()["cleanup_verified"] is True  # nothing unfinished, nothing to verify


def test_public_item_with_an_unlaunched_worker_gets_the_same_named_reason(tmp_path):
    broker, _, _ = make_broker(tmp_path)
    request = make_request(broker)
    outcome = {"sample": {"status": "environment-error", "passed": False, "model_evaluated": False,
                          "failure_reason": "worker sandbox-unavailable, exit None",
                          "cases": [{"case_id": "pinned-tests", "status": "environment-error",
                                     "worker_status": "sandbox-unavailable", "cleanup_confirmed": True}]},
               "raw_bytes": b'{"traces":[]}', "abort_campaign": False}
    result = broker._outcome_result(request, request.content_sha256(), outcome)
    assert result.status == "environment-error" and not result.abort_campaign and result.cleanup_confirmed
    assert result.failure_reason == ("sandbox_unavailable: the Docker client could not be started for case(s) "
                                     "pinned-tests")
    broker.close()


# ---- broker namespace of a native run ---------------------------------------------------------------------------

def write_plan(run_dir, name, payload):
    (run_dir / "plan").mkdir(parents=True, exist_ok=True)
    (run_dir / "plan" / name).write_text(payload if isinstance(payload, str) else json.dumps(payload),
                                         encoding="utf-8")


def native_launch(session="native-sess", attempt="f" * 32):
    return {"runtime": "metal-native", "argv": ["--port", "1"],
            "labels": {"llmbench.session": session, "llmbench.attempt": attempt, "llmbench.runtime": "metal-native"}}


def test_factory_reads_the_native_launch_labels_when_there_is_no_compose_plan(tmp_path):
    run_dir = tmp_path / "run"
    write_plan(run_dir, "native-launch.json", native_launch())
    broker = HostBroker.factory(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"))
    assert (broker.session_id, broker.attempt_id) == ("native-sess", "f" * 32)
    assert ledger(broker)[0]["namespace_source"] == "native-plan"
    published = json.loads((run_dir / "spool" / "results" / "namespace.json").read_text(encoding="utf-8"))
    assert published == {"schema_version": 1, "session_id": "native-sess", "attempt_id": "f" * 32}
    broker.close()


def test_compose_plan_still_wins_so_the_container_path_is_unchanged(tmp_path):
    run_dir = tmp_path / "run"
    write_plan(run_dir, "compose.json", {"services": {"inference": {"labels": {
        "llmbench.session": "sess", "llmbench.attempt": "d" * 32}}}})
    write_plan(run_dir, "native-launch.json", native_launch())
    broker = HostBroker.factory(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"))
    assert (broker.session_id, broker.attempt_id) == ("sess", "d" * 32)
    assert ledger(broker)[0]["namespace_source"] == "plan"
    broker.close()


@pytest.mark.parametrize("payload", ["{not json", {"labels": {"llmbench.session": "s"}}, {"labels": []},
                                     {"argv": []}, ["labels"]])
def test_an_unreadable_native_launch_record_refuses_instead_of_minting(tmp_path, payload):
    run_dir = tmp_path / "run"
    write_plan(run_dir, "native-launch.json", payload)
    with pytest.raises(ValueError, match="plan/native-launch.json exists but its attempt labels are unreadable"):
        HostBroker.factory(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"))
    assert not (run_dir / "spool" / "results" / "namespace.json").exists()


def test_factory_forwards_the_runner_identity_and_checks_it_against_plan_and_ledger(tmp_path):
    bare = tmp_path / "bare"
    broker = HostBroker.factory(bare, broker_config(), LOCK, RunArtifacts(bare).scoped("broker"),
                                attempt_id="a" * 32, session_id="given")
    assert (broker.session_id, broker.attempt_id) == ("given", "a" * 32)
    assert ledger(broker)[0]["namespace_source"] == "argument"
    broker.close()

    agreeing = tmp_path / "agree"
    write_plan(agreeing, "native-launch.json", native_launch())
    broker = HostBroker.factory(agreeing, broker_config(), LOCK, RunArtifacts(agreeing).scoped("broker"),
                                attempt_id="f" * 32, session_id="native-sess")
    assert (broker.session_id, broker.attempt_id) == ("native-sess", "f" * 32)
    broker.close()
    # A restart of the same attempt reads the ledger; a different identity is refused, never adopted.
    with pytest.raises(ValueError, match="recorded ledger namespace"):
        HostBroker.factory(agreeing, broker_config(), LOCK, RunArtifacts(agreeing).scoped("broker"),
                           attempt_id="e" * 32, session_id="native-sess")

    disagreeing = tmp_path / "disagree"
    write_plan(disagreeing, "native-launch.json", native_launch())
    for kwargs in ({"attempt_id": "e" * 32, "session_id": "native-sess"},
                   {"attempt_id": "f" * 32, "session_id": "someone-else"},
                   {"session_id": "someone-else"},  # a named session alone is checked too, never replaced
                   {"attempt_id": "f" * 32}):  # session falls back to config.session_id, which is not the plan's
        with pytest.raises(ValueError, match="differ from the attempt labels"):
            HostBroker.factory(disagreeing, broker_config(), LOCK, RunArtifacts(disagreeing).scoped("broker"),
                               **kwargs)
    assert not (disagreeing / "spool" / "results" / "namespace.json").exists()  # refused before publishing
    # The same session named alone, agreeing with the plan, is the plan's namespace.
    broker = HostBroker.factory(disagreeing, broker_config(), LOCK, RunArtifacts(disagreeing).scoped("broker"),
                                session_id="native-sess")
    assert (broker.session_id, broker.attempt_id, ledger(broker)[0]["namespace_source"]) == (
        "native-sess", "f" * 32, "native-plan")
    broker.close()


# ---- the evaluator with no sandbox ------------------------------------------------------------------------------

AIDER_TASKS = ["aider-polyglot/python/word-reverse", "aider-polyglot/python/flatten-once"]
REASON = "the Docker CLI was not found on PATH"


@pytest.fixture
def eval_helpers(monkeypatch):
    import test_container_eval as helpers
    monkeypatch.setitem(helpers.NAMESPACED, "aider-polyglot", ("aider-polyglot-v1", "coding", AIDER_TASKS))
    return helpers


def blocked_config(helpers):
    from llmbench.coding.fixtures import fixtures
    fixture_ids = [item.fixture_id for item in fixtures()]
    base = helpers.public_config("evalplus", "aider-polyglot", "ruler", broker={})
    selections = [json.loads(item.model_dump_json()) for item in base.benchmarks]
    selections.insert(1, {"benchmark_id": "coding", "revision": "private-coding-v1", "task_ids": fixture_ids})
    return helpers.make_config(benchmarks=selections, broker={}), fixture_ids


class RecordingClient:
    """A broker client that must never be touched while the sandbox is blocked."""

    def __init__(self):
        self.touched = []

    def __getattr__(self, name):
        self.touched.append(name)
        raise AssertionError(f"the blocked evaluator used the broker client: {name}")


def test_blocked_sandbox_never_starts_an_execution_suite_and_names_the_reason(tmp_path, eval_helpers, monkeypatch):
    from llmbench.container_eval import read_evaluation
    from llmbench.evaluations.selection import expected_task_rows
    helpers = eval_helpers
    evalplus, aider, ruler = (helpers.FakeBenchmark("evalplus"), helpers.FakeBenchmark("aider-polyglot"),
                              helpers.FakeBenchmark("ruler"))
    helpers.register(monkeypatch, evalplus, aider, ruler)
    config, fixture_ids = blocked_config(helpers)
    hooked, client = [], RecordingClient()

    def hook(*args, **kwargs):
        hooked.append(args)
        raise AssertionError("the coding hook ran without a sandbox")

    result, _, _ = helpers.evaluate(tmp_path, config, coding_hook=hook, coding_client=client,
                                    execution_unavailable_reason=REASON, dataset_root=str(tmp_path / "data"))
    assert hooked == [] and client.touched == [] and evalplus.run_contexts == [] and aider.run_contexts == []
    assert len(ruler.run_contexts) == 1  # a suite that executes nothing still runs normally
    assert result["errors"] == [] and result["abort_reason"] is None
    assert [stage["name"] for stage in result["stages"]] == ["attach", "count-route", "overflow-probe", "speed",
                                                             "quality", "benchmarks"]
    text = "sandbox_unavailable: " + REASON
    samples = helpers.by_id(result)
    blocked_ids = fixture_ids + helpers.EVALPLUS_TASKS + AIDER_TASKS
    for task_id in blocked_ids:
        row = samples[task_id]
        assert (row["status"], row["outcome_status"], row["reason"]) == ("environment_error", "environment_error", text)
        assert row["score"] == 0.0 and row["passed"] is False
        assert row["model_evaluated"] is False and row["synthetic"] is False and row["category"] == "coding"
    assert all(samples[task_id]["status"] == "completed" for task_id in helpers.RULER_TASKS)
    assert samples["tools/no-call-v1"]["status"] == "completed"
    records = {record["benchmark_id"]: record for record in result["benchmarks"]}
    assert set(records) == {"coding", "evalplus", "aider-polyglot", "ruler"} and records["ruler"]["status"] == "ok"
    for suite, declared in (("coding", len(fixture_ids)), ("evalplus", 2), ("aider-polyglot", 2)):
        assert records[suite] == {"benchmark_id": suite, "revision": records[suite]["revision"],
                                  "split": "development", "declared": declared, "produced": 0,
                                  "status": "environment_error", "reason": text, "problems": [],
                                  "backfilled": declared, "execution_client": False}
    # The persisted verdict carries the same rows and passes the host's strict denominator check.
    persisted = tmp_path / "evaluator" / "evaluation.json"
    read_evaluation(persisted, max_bytes=10_000_000, expected_rows=expected_task_rows(config.benchmarks))
    assert json.loads(persisted.read_text(encoding="utf-8"))["benchmarks"] == result["benchmarks"]


def test_blocked_rows_keep_their_reason_when_the_run_ends_early(tmp_path, eval_helpers, monkeypatch):
    helpers = eval_helpers
    helpers.register(monkeypatch, helpers.FakeBenchmark("evalplus"), helpers.FakeBenchmark("aider-polyglot"),
                     helpers.FakeBenchmark("ruler"))
    config, fixture_ids = blocked_config(helpers)
    server = helpers.ScriptedServer(config)
    server.props["default_generation_settings"]["n_ctx"] = 4096  # attach fails: nothing else runs
    result, _, _ = helpers.evaluate(tmp_path, config, server=server, execution_unavailable_reason=REASON,
                                    dataset_root=str(tmp_path / "data"))
    assert result["abort_reason"].startswith("attach failed")
    samples = helpers.by_id(result)
    for task_id in fixture_ids + helpers.EVALPLUS_TASKS + AIDER_TASKS:
        assert samples[task_id]["reason"] == "sandbox_unavailable: " + REASON
    for task_id in helpers.RULER_TASKS + ["tools/no-call-v1"]:
        assert samples[task_id]["reason"] == "server identity was not verified"
    assert len(result["samples"]) == len(set(samples))  # nothing reported twice


@pytest.mark.parametrize("reason", ["", "   ", 7, b"docker"])
def test_a_malformed_unavailability_claim_refuses_before_any_artifact(tmp_path, reason):
    import time
    import test_container_eval as helpers
    from llmbench.container_eval import EvaluationContext, run_evaluation
    built = []
    ctx = EvaluationContext(base_url="http://127.0.0.1:18080", artifacts_dir=tmp_path / "evaluator",
                            deadline_monotonic=time.monotonic() + 60, session_lock=helpers.ALLOWED,
                            execution_unavailable_reason=reason)
    with pytest.raises(ValueError, match="execution_unavailable_reason"):
        run_evaluation(helpers.make_config(), ctx, backend_factory=lambda *a, **k: built.append(1))
    assert built == [] and not (tmp_path / "evaluator").exists()


def test_without_a_claim_the_evaluation_shape_is_unchanged(tmp_path, eval_helpers):
    from llmbench.container_eval import EvaluationContext
    helpers = eval_helpers
    assert EvaluationContext.__dataclass_fields__["execution_unavailable_reason"].default is None
    result, _, _ = helpers.evaluate(tmp_path, helpers.make_config())
    assert set(result) == {"samples", "speed_observations", "readback", "count_route", "overflow_probe",
                           "actual_context_verified", "abort_reason", "errors", "warmups", "native_timings",
                           "tokenizer_verified", "inspect_logs", "stages", "benchmarks"}
    assert result["benchmarks"] == [] and not any("sandbox_unavailable" in json.dumps(row)
                                                  for row in result["samples"])


def test_host_process_hook_without_a_broker_is_refused_and_never_falls_back_to_spool(tmp_path, eval_helpers,
                                                                                    monkeypatch):
    from llmbench.coding import broker_client
    from llmbench.coding.generation import run_coding_benchmarks
    from llmbench.container_eval import _coding_client
    helpers = eval_helpers

    def spool_tripwire(self, *args, **kwargs):
        raise AssertionError("SpoolClient was constructed on the host")

    monkeypatch.setattr(broker_client.SpoolClient, "__init__", spool_tripwire)
    config, fixture_ids = blocked_config(helpers)
    config = helpers.make_config(benchmarks=[json.loads(config.benchmarks[0].model_dump_json()),
                                             json.loads(config.benchmarks[1].model_dump_json())], broker={})
    result, _, _ = helpers.evaluate(tmp_path, config, coding_hook=run_coding_benchmarks)  # no client, host mode
    assert result["stages"][-1] == {"name": "coding", "status": "failed",
                                    "elapsed_seconds": result["stages"][-1]["elapsed_seconds"]}
    assert result["errors"][-1]["stage"] == "coding" and result["errors"][-1]["error_type"] == "ExecutionUnavailable"
    samples = helpers.by_id(result)
    assert all(samples[task_id]["status"] == "environment_error" and samples[task_id]["model_evaluated"] is False
               for task_id in fixture_ids)
    assert not (tmp_path / "evaluator" / "coding").exists() or not any(
        path.name.startswith("coding-request") for path in (tmp_path / "evaluator" / "coding").iterdir())

    # The stand-in refuses everything; the container path keeps None (it resolves its own /spool client).
    ctx = SimpleNamespace(coding_client=None, allow_remote=False)
    stand_in = _coding_client(ctx)
    assert isinstance(stand_in, UnavailableClient)
    for call in (stand_in.namespace, lambda: stand_in.submit(object()), lambda: stand_in.wait("r", deadline=1)):
        with pytest.raises(ExecutionUnavailable, match="never executed on the host"):
            call()
    assert _coding_client(SimpleNamespace(coding_client=None, allow_remote=True)) is None
    assert _coding_client(SimpleNamespace(coding_client="direct", allow_remote=False)) == "direct"
    with pytest.raises(ValueError):
        UnavailableClient(" ")


def test_host_runner_forwards_the_claim_into_the_evaluation_context(monkeypatch):
    import llmbench.container_eval as container_eval
    from llmbench.containers import runner
    seen = []
    monkeypatch.setattr(container_eval, "run_evaluation", lambda config, ctx: seen.append(ctx) or {})
    context = SimpleNamespace(base_url="http://127.0.0.1:1", artifacts_dir="x", deadline_monotonic=1.0,
                              execution_unavailable_reason=REASON, unrelated_runner_state=object())
    runner._default_evaluator(SimpleNamespace(), context)
    assert seen[0].execution_unavailable_reason == REASON
