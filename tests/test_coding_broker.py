"""Host coding broker: strict spool contracts, validation order, bounded ticks, ledger and cleanup.

No Docker, model, GPU or network: the worker executor is the scripted fake from test_coding_runner."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llmbench.coding.broker import CLEANUP_MARGIN_SECONDS, HostBroker, default_broker_settings
from llmbench.coding.broker_client import DirectClient
from llmbench.coding.fixtures import fixtures
from llmbench.coding.sandbox import DockerWorker
from llmbench.coding.spool import (
    CodingJobRequest, CodingJobResult, publish_atomic, read_bounded, request_bytes, request_name,
)
from llmbench.config import GenerationSettings, RunMode
from llmbench.containers.artifacts import RunArtifacts
from llmbench.containers.config import BenchmarkSelection
from llmbench.evaluations.tools import strict_json_loads
from llmbench.safety import SessionLock

from test_coding_runner import FixtureExecutor

IMAGE = "sha256:" + "a" * 64
ATTEMPT = "b" * 32
LOCK = SessionLock(allow_container_execution=True)
FIXTURE = fixtures()[0]


class MultiFixtureExecutor(FixtureExecutor):
    """The runner fake answers for one fixture; pick the fixture from the staged input on every launch."""

    def run(self, argv, **limits):
        if argv[1] == "run":
            mount = argv[argv.index("--mount") + 1]
            path = Path(mount.removeprefix("type=bind,source=").removesuffix(",target=/workspace,readonly"))
            source = json.loads(path.joinpath(".llmbench-input.json").read_text())["source"]
            self.fixture = next(item for item in fixtures() if any(f.path == source for f in item.initial_files))
        return super().run(argv, **limits)


def broker_config(**settings):
    return SimpleNamespace(session_id="session-1", broker=default_broker_settings(**settings),
                           worker_image=SimpleNamespace(reference=IMAGE),
                           generation=GenerationSettings(max_output_tokens=256),
                           benchmarks=(BenchmarkSelection(benchmark_id="coding", revision="private-coding-v1",
                                                          task_ids=(FIXTURE.fixture_id,)),))


def make_broker(tmp_path, *, executor=None, settings=None, artifacts=None, run_dir=None, **kwargs):
    executor = executor or MultiFixtureExecutor(FIXTURE)
    run_dir = run_dir or tmp_path / "run"
    artifacts = artifacts or RunArtifacts(run_dir)

    def worker_factory(root, lock, injected):
        return DockerWorker(allowed_root=root, session_lock=lock, mode=RunMode.LIVE, executor=injected)

    kwargs.setdefault("attempt_id", ATTEMPT)
    broker = HostBroker(run_dir, broker_config(**(settings or {})), LOCK, artifacts.scoped("broker"),
                        worker_factory=worker_factory, executor=executor, **kwargs)
    return broker, executor, artifacts


def make_request(broker, fixture=FIXTURE, patch=None, **overrides):
    data = dict(session_id=broker.session_id, attempt_id=broker.attempt_id, request_id=uuid.uuid4().hex,
                fixture_id=fixture.fixture_id, fixture_revision=fixture.revision, fixture_hash=fixture.identity(),
                attempt_index=1,
                patch=patch if patch is not None else {item.path: item.content for item in fixture.reference_files},
                submitted_utc="2026-09-18T00:00:00+00:00")
    data.update(overrides)
    return CodingJobRequest(**data)


def drop(broker, request):
    return publish_atomic(broker.requests_dir, request_name(request.request_id), request_bytes(request))


def read_result(broker, request_id):
    return CodingJobResult.model_validate(strict_json_loads(
        read_bounded(broker.results_dir / request_name(request_id), 4_194_304).decode("utf-8")))


def ledger(broker):
    rows = []
    for path in sorted(broker.ledger_dir.glob("ledger*.jsonl")):
        rows.extend(strict_json_loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def runs(executor):
    return sum(argv[1] == "run" for argv, _ in executor.calls)


def test_request_and_result_schemas_are_strict_and_hash_stable(tmp_path):
    broker, _, _ = make_broker(tmp_path)
    one = make_request(broker, request_id="c" * 32)
    other_time = make_request(broker, request_id="c" * 32, submitted_utc="2027-01-01T00:00:00+00:00")
    assert one.content_sha256() == other_time.content_sha256() and len(one.content_sha256()) == 64
    assert make_request(broker, request_id="c" * 32, patch={"batches.py": "x\n"}).content_sha256() != one.content_sha256()
    assert make_request(broker, request_id="c" * 32, attempt_index=2).content_sha256() != one.content_sha256()
    assert CodingJobRequest.model_validate_json(request_bytes(one)) == one
    for bad in ({"extra": 1}, {"request_id": "C" * 32}, {"attempt_index": 3}, {"fixture_revision": "bad rev"},
                {"fixture_revision": ""}, {"fixture_revision": "x" * 129},
                {"patch": {}}, {"patch": {"../x.py": "a"}}, {"patch": {"/abs.py": "a"}}, {"patch": {"a\\b.py": "a"}},
                {"patch": {"batches.py": "nul\x00"}}, {"patch": {"batches.py": "\ud800"}}, {"patch": {"": "a"}},
                {"patch": {"batches.py": 1}}, {"session_id": "Bad Name"}, {"schema_version": 2}):
        with pytest.raises(ValidationError):
            make_request(broker, **bad)
    # The schema admits a public benchmark's revision, so a well-formed revision the fixture does not declare is
    # refused by the BROKER instead, against its own host-side fixture: the evaluator cannot choose a revision.
    wrong = make_request(broker, fixture_revision="evalplus/mbpp-plus@v0.2.0")
    assert broker.validate(drop(broker, wrong), max_seconds=300) == (None, "fixture_revision")
    sha = one.content_sha256()
    CodingJobResult(request_id="c" * 32, request_sha256=sha, status="rejected", failure_reason="x",
                    cleanup_confirmed=True, abort_campaign=False, finished_utc="t")
    for bad in ({"status": "completed"},                                       # completed needs a sample
                {"status": "rejected", "sample": {"a": 1}},                    # only completed carries one
                {"status": "rejected", "failure_reason": None},                # every failure states a reason
                {"status": "rejected", "abort_campaign": True},                # abort only with cleanup-unverified
                {"status": "cleanup-unverified", "abort_campaign": False},
                {"status": "cleanup-unverified", "abort_campaign": True, "cleanup_confirmed": True},
                {"status": "done"}, {"trace_sha256": "abc"}, {"unknown": 1}):
        with pytest.raises(ValidationError):
            CodingJobResult(**{"request_id": "c" * 32, "request_sha256": sha, "failure_reason": "x",
                               "cleanup_confirmed": True, "abort_campaign": False, "finished_utc": "t",
                               "status": "rejected", **bad})


def test_publish_atomic_never_overwrites_and_leaves_no_temp_on_failure(tmp_path, monkeypatch):
    spool = tmp_path / "spool"
    spool.mkdir()
    target = publish_atomic(spool, "a" * 32 + ".json", b"first")
    with pytest.raises(FileExistsError):
        publish_atomic(spool, "a" * 32 + ".json", b"second")
    assert target.read_bytes() == b"first" and sorted(p.name for p in spool.iterdir()) == ["a" * 32 + ".json"]
    for name in ("../x.json", "x/y.json", ".hidden.json", "x.txt", "", "x.json\n", "a" * 70 + ".json"):
        with pytest.raises(ValueError):
            publish_atomic(spool, name, b"x")
    with pytest.raises(TypeError):
        publish_atomic(spool, "b" * 32 + ".json", "text")

    def boom(*args, **kwargs):
        raise OSError("rename failed")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        publish_atomic(spool, "b" * 32 + ".json", b"never")
    assert sorted(p.name for p in spool.iterdir()) == ["a" * 32 + ".json"]
    with pytest.raises(ValueError):
        publish_atomic(tmp_path / "missing", "c" * 32 + ".json", b"x")


def test_validate_accepts_exact_fixture_patch_and_stages_into_host_owned_root(tmp_path):
    broker, executor, artifacts = make_broker(tmp_path)
    request = make_request(broker)
    path = drop(broker, request)
    validated, reason = broker.validate(path)
    assert reason is None and validated == request
    summary = broker.tick(300)
    assert summary == {"processed": 1, "rejected": 0, "pending": 0, "abort_campaign": False}
    result = read_result(broker, request.request_id)
    assert result.status == "completed" and result.request_sha256 == request.content_sha256()
    assert result.cleanup_confirmed and not result.abort_campaign
    sample = result.sample
    assert sample["passed"] is True and sample["score"] == 1.0 and sample["first_attempt_success"] is True
    assert sample["required_checks"] == sample["attempted_checks"] == len(FIXTURE.checks)
    assert sample["request_id"] == request.request_id and sample["attempt_index"] == 1
    mounts = [argv[argv.index("--mount") + 1] for argv, _ in executor.calls if argv[1] == "run"]
    assert len(mounts) == len(FIXTURE.checks)
    for mount in mounts:
        source = Path(mount.removeprefix("type=bind,source=").removesuffix(",target=/workspace,readonly"))
        assert source.is_relative_to(broker.workers_root) and broker.workers_root.is_relative_to(broker.run_dir)
        assert source.joinpath("batches.py").read_text() == FIXTURE.reference_files[0].content
    trace = broker.run_dir / "broker" / "traces" / f"{request.request_id}.json"
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == result.trace_sha256
    kinds = [row["event"] for row in ledger(broker)]
    assert kinds[:4] == ["broker", "seen", "validated", "started"]
    assert kinds.count("worker") == len(FIXTURE.checks) and kinds[-2:] == ["finished", "published"]
    assert kinds.index("worker") < kinds.index("finished")
    namespace = json.loads((broker.results_dir / "namespace.json").read_text())
    assert namespace == {"schema_version": 1, "session_id": "session-1", "attempt_id": ATTEMPT}


def test_tick_processes_at_most_one_request_and_returns_within_max_seconds(tmp_path):
    broker, executor, _ = make_broker(tmp_path, settings={"fixture_timeout_seconds": 120})
    requests = [make_request(broker) for _ in range(3)]
    for request in requests:
        drop(broker, request)
    first = broker.tick(300)
    assert (first["processed"], first["pending"]) == (1, 2) and runs(executor) == len(FIXTURE.checks)
    second = broker.tick(90)
    assert (second["processed"], second["pending"]) == (1, 1)
    budgets = [row["budget_seconds"] for row in ledger(broker) if row["event"] == "started"]
    assert budgets == [120.0, 90.0 - CLEANUP_MARGIN_SECONDS]  # never beyond the tick grant minus cleanup
    assert broker.tick(300)["processed"] == 1 and broker.tick(300) == {"processed": 0, "rejected": 0, "pending": 0,
                                                                        "abort_campaign": False}
    assert len(list(broker.results_dir.glob("*.json"))) == 4  # three results + namespace.json
    with pytest.raises(ValueError):
        broker.tick(float("inf"))


def test_same_id_same_content_returns_existing_result(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    request = make_request(broker)
    path = drop(broker, request)
    broker.tick(300)
    original = read_result(broker, request.request_id)
    launches = runs(executor)
    result_path = broker.results_dir / request_name(request.request_id)
    result_path.unlink()
    path.unlink()
    drop(broker, make_request(broker, request_id=request.request_id, submitted_utc="2028-01-01T00:00:00+00:00"))
    assert broker.tick(300)["processed"] == 0
    assert read_result(broker, request.request_id) == original and runs(executor) == launches
    kinds = [row["event"] for row in ledger(broker)]
    assert "replay" in kinds and "republished" in kinds
    assert broker.tick(300)["processed"] == 0 and runs(executor) == launches


def test_same_id_changed_content_is_rejected_and_ledgered(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    request = make_request(broker)
    path = drop(broker, request)
    broker.tick(300)
    result_path = broker.results_dir / request_name(request.request_id)
    before, launches = result_path.read_bytes(), runs(executor)
    path.unlink()
    drop(broker, make_request(broker, request_id=request.request_id, patch={"batches.py": "def chunked(v, s): pass\n"}))
    summary = broker.tick(300)
    assert summary["processed"] == 0 and summary["rejected"] == 1 and runs(executor) == launches
    assert result_path.read_bytes() == before  # the client already holds this terminal result
    rejections = [row for row in ledger(broker) if row["event"] == "rejected"]
    assert rejections[-1]["request_id"] == request.request_id and rejections[-1]["reason"] == "replay_changed_content"
    result_path.unlink()
    broker.tick(300)
    rejected = read_result(broker, request.request_id)
    assert rejected.status == "rejected" and rejected.failure_reason == "replay_changed_content"


def test_request_count_cap_rejects_flood_without_starving_tick(tmp_path):
    broker, executor, _ = make_broker(tmp_path, settings={"max_requests": 2})
    requests = [make_request(broker) for _ in range(6)]
    for request in requests:
        drop(broker, request)
    summaries = [broker.tick(300) for _ in range(6)]
    assert all(item["processed"] <= 1 for item in summaries)
    statuses = {}
    for request in requests:
        try:
            statuses[request.request_id] = read_result(broker, request.request_id).status
        except FileNotFoundError:
            statuses[request.request_id] = None
    assert sorted(statuses.values(), key=str) == [None, None, "completed", "completed", "rejected", "rejected"]
    rejected = [read_result(broker, rid) for rid, status in statuses.items() if status == "rejected"]
    assert all(item.failure_reason == "too_many_requests" for item in rejected)
    assert runs(executor) == 2 * len(FIXTURE.checks)
    ignored = [row for row in ledger(broker) if row["event"] == "ignored"]
    assert len(ignored) == 2 and all(row["reason"] == "result_cap" for row in ignored)
    assert broker.tick(300) == {"processed": 0, "rejected": 0, "pending": 0, "abort_campaign": False}


def test_no_time_rejection_when_remaining_below_fixture_floor(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    assert broker.floor_seconds == 30 + CLEANUP_MARGIN_SECONDS
    request = make_request(broker)
    path = drop(broker, request)
    assert broker.validate(path, max_seconds=49) == (None, "no_time")
    assert broker.validate(path, max_seconds=50)[1] is None
    summary = broker.tick(10)
    assert summary["processed"] == 0 and summary["rejected"] == 1 and executor.calls == []
    result = read_result(broker, request.request_id)
    assert result.status == "rejected" and result.failure_reason == "no_time"
    assert result.request_sha256 == request.content_sha256()  # the client can match its own request
    direct = broker.execute(make_request(broker), max_seconds=5)
    assert direct.status == "rejected" and direct.failure_reason == "no_time" and executor.calls == []


def test_reconcile_marks_interrupted_requests_and_removes_recorded_workers(tmp_path):
    class Crash(MultiFixtureExecutor):
        def run(self, argv, **limits):
            if argv[1] == "run":
                self.calls.append((argv, limits))
                raise KeyboardInterrupt("operator interrupt during the first case")
            return super().run(argv, **limits)

    run_dir = tmp_path / "run"
    first, crash, artifacts = make_broker(tmp_path, executor=Crash(FIXTURE), run_dir=run_dir)
    request = make_request(broker=first)
    drop(first, request)
    with pytest.raises(KeyboardInterrupt):
        first.tick(300)
    first.close()
    rows = ledger(first)
    names = [row["name"] for row in rows if row["event"] == "worker"]
    assert len(names) == 1 and not any(row["event"] == "finished" for row in rows)
    assert not (first.results_dir / request_name(request.request_id)).exists()

    verifier = MultiFixtureExecutor(FIXTURE)
    second = HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"), executor=verifier)
    assert (second.session_id, second.attempt_id) == (first.session_id, first.attempt_id)
    summary = second.reconcile()
    assert summary == {"interrupted": 1, "workers_checked": 1, "cleanup_verified": True, "abort_campaign": False}
    assert [argv for argv, _ in verifier.calls] == [
        ("docker", "rm", "-f", names[0]),
        ("docker", "ps", "--all", "--filter", f"name=^/{names[0]}$", "--format", "{{.ID}}")]
    result = read_result(second, request.request_id)
    assert result.status == "interrupted" and result.cleanup_confirmed and not result.abort_campaign
    assert result.request_sha256 == request.content_sha256()
    assert second.tick(300) == {"processed": 0, "rejected": 0, "pending": 0, "abort_campaign": False}
    assert sorted(p.name for p in (run_dir / "broker").glob("ledger*.jsonl")) == ["ledger-1.jsonl", "ledger.jsonl"]
    with pytest.raises(ValueError, match="namespace"):
        HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"), attempt_id="f" * 32)
    second.close()

    # A worker that cannot be verified absent turns the interruption into a campaign abort.
    third = HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"),
                       executor=MultiFixtureExecutor(FIXTURE, cleanup_absent=False))
    assert third.reconcile()["interrupted"] == 0 and not third.abort_campaign  # already finished by the second broker
    third.close()


def interrupted_run(tmp_path):
    """First generation: the operator interrupts during ``docker run``; returns (run_dir, request, worker name)."""
    class Crash(MultiFixtureExecutor):
        def run(self, argv, **limits):
            if argv[1] == "run":
                self.calls.append((argv, limits))
                raise KeyboardInterrupt("operator interrupt during the first case")
            return super().run(argv, **limits)

    run_dir = tmp_path / "run"
    first, _, _ = make_broker(tmp_path, executor=Crash(FIXTURE), run_dir=run_dir)
    request = make_request(first)
    drop(first, request)
    with pytest.raises(KeyboardInterrupt):
        first.tick(300)
    first.close()
    names = [row["name"] for row in ledger(first) if row["event"] == "worker"]
    assert len(names) == 1
    return run_dir, request, names[0]


def test_reconcile_with_worker_still_present_aborts_campaign_and_is_inherited(tmp_path):
    """Absence not verified (docker ps still lists the worker): cleanup-unverified, abort, ticks refuse, inherited."""
    run_dir, request, name = interrupted_run(tmp_path)
    still_there = MultiFixtureExecutor(FIXTURE, cleanup_absent=False)
    second = HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"), executor=still_there)
    summary = second.reconcile()
    assert summary == {"interrupted": 1, "workers_checked": 1, "cleanup_verified": False, "abort_campaign": True}
    assert [argv[:2] for argv, _ in still_there.calls] == [("docker", "rm"), ("docker", "ps")]
    result = read_result(second, request.request_id)
    assert result.status == "cleanup-unverified" and result.abort_campaign and not result.cleanup_confirmed
    assert result.request_sha256 == request.content_sha256()
    tick = second.tick(300)
    assert tick["processed"] == 0 and tick["abort_campaign"] is True and "not verified absent" in tick["reason"]
    another = make_request(second)
    drop(second, another)
    assert second.tick(300)["processed"] == 0 and runs(still_there) == 0  # nothing runs after the abort
    assert second.cancel_all()["cleanup_verified"] is False
    # A third generation inherits the abort from the ledger alone, before any Docker call.
    third = HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"),
                       executor=MultiFixtureExecutor(FIXTURE))
    assert third.abort_campaign is True and third.tick(300)["abort_campaign"] is True and runs(third.executor) == 0
    third.close()


def test_direct_client_raises_on_inherited_campaign_abort_instead_of_waiting(tmp_path):
    from llmbench.coding.broker_client import BrokerAborted
    run_dir, _, _ = interrupted_run(tmp_path)
    second = HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"),
                        executor=MultiFixtureExecutor(FIXTURE, cleanup_absent=False))
    assert second.reconcile()["abort_campaign"] is True
    second.close()
    now, sleeps = [0.0], []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    third = HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"),
                       executor=MultiFixtureExecutor(FIXTURE), clock=lambda: now[0])
    client = DirectClient(third, clock=lambda: now[0], sleep=sleep)
    request = make_request(third)
    client.submit(request)
    with pytest.raises(BrokerAborted, match="not verified absent"):
        client.wait(request.request_id, deadline=now[0] + 1000.0)
    assert sleeps == [] and now[0] == 0.0  # no waiting out the budget
    assert not (third.results_dir / request_name(request.request_id)).exists()
    third.close()


def test_finished_without_published_is_republished_never_re_executed(tmp_path):
    """Crash between the ``finished`` and ``published`` ledger lines: the next generation publishes, never re-runs."""
    run_dir = tmp_path / "run"
    first, executor, _ = make_broker(tmp_path, run_dir=run_dir)
    request = make_request(first)
    drop(first, request)
    first.tick(300)
    first.close()
    ledger_path = run_dir / "broker" / "ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"] == "published"
    ledger_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # the crash window
    (run_dir / "spool" / "results" / request_name(request.request_id)).unlink()
    verifier = MultiFixtureExecutor(FIXTURE)
    second = HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"), executor=verifier)
    summary = second.reconcile()
    assert summary == {"interrupted": 1, "workers_checked": 0, "cleanup_verified": True, "abort_campaign": False}
    result = read_result(second, request.request_id)
    assert result.status == "interrupted" and result.cleanup_confirmed and not result.abort_campaign
    assert result.request_sha256 == request.content_sha256() and "before publishing" in result.failure_reason
    assert second.tick(300) == {"processed": 0, "rejected": 0, "pending": 0, "abort_campaign": False}
    assert verifier.calls == []  # no docker call at all: the first run's cleanup was already verified
    events = [row["event"] for row in ledger(second)]  # both generations; the deleted line is the only gap
    assert "interrupted" in events and events.count("published") == 1 and events.count("started") == 1
    second.close()
    # Crash after the result file was renamed into place but before its ledger line: also terminal, never re-run.
    run_dir2 = tmp_path / "run2"
    other, executor2, _ = make_broker(tmp_path, run_dir=run_dir2)
    request2 = make_request(other)
    drop(other, request2)
    other.tick(300)
    other.close()
    ledger2 = run_dir2 / "broker" / "ledger.jsonl"
    lines = ledger2.read_text(encoding="utf-8").splitlines()
    ledger2.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    verifier2 = MultiFixtureExecutor(FIXTURE)
    third = HostBroker(run_dir2, broker_config(), LOCK, RunArtifacts(run_dir2).scoped("broker"), executor=verifier2)
    assert third.reconcile()["interrupted"] == 1
    assert read_result(third, request2.request_id).status == "completed"  # the original result stays in place
    assert third.tick(300)["processed"] == 0 and verifier2.calls == []
    assert [row["event"] for row in ledger(third)].count("publish_skipped_existing") == 1
    third.close()


def test_broker_refuses_max_requests_below_two_attempts_per_selected_task(tmp_path):
    """A legitimate evaluator needs up to two results per task; a smaller cap would silently starve it (REV-B1-02)."""
    with pytest.raises(ValueError, match="max_requests"):
        make_broker(tmp_path, settings={"max_requests": 1})
    broker, _, _ = make_broker(tmp_path, settings={"max_requests": 2})
    broker.close()
    config = broker_config(max_requests=2)
    config.benchmarks = config.benchmarks + (BenchmarkSelection(benchmark_id="coding", revision="private-coding-v1",
                                                                 task_ids=("typescript/group-records",), split="holdout"),)
    with pytest.raises(ValueError, match="max_requests"):
        HostBroker(tmp_path / "two", config, LOCK, RunArtifacts(tmp_path / "two").scoped("broker"), attempt_id=ATTEMPT)
    config.broker = default_broker_settings(max_requests=4)
    HostBroker(tmp_path / "four", config, LOCK, RunArtifacts(tmp_path / "four").scoped("broker"), attempt_id=ATTEMPT).close()


def test_cleanup_unverified_sets_abort_campaign_and_halts_further_ticks(tmp_path):
    broker, executor, _ = make_broker(tmp_path, executor=MultiFixtureExecutor(FIXTURE, cleanup_absent=False))
    first, second = make_request(broker, request_id="1" * 32), make_request(broker, request_id="2" * 32)
    drop(broker, first)
    drop(broker, second)
    summary = broker.tick(300)
    assert summary["processed"] == 1 and summary["abort_campaign"] is True and broker.abort_campaign
    result = read_result(broker, first.request_id)
    assert result.status == "cleanup-unverified" and result.abort_campaign and not result.cleanup_confirmed
    assert result.sample is None and result.trace_sha256 is not None and "cleanup unverified" in result.failure_reason
    launches = runs(executor)
    again = broker.tick(300)
    assert again["processed"] == 0 and again["abort_campaign"] is True and again["pending"] == 1
    assert runs(executor) == launches
    assert not (broker.results_dir / request_name(second.request_id)).exists()
    finished = broker.cancel_all()
    assert finished["cancelled"] == 1 and finished["cleanup_verified"] is False
    assert read_result(broker, second.request_id).status == "cancelled"


def test_cancel_all_publishes_cancelled_results_and_verifies_absence(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    pending = [make_request(broker) for _ in range(2)]
    for request in pending:
        drop(broker, request)
    (broker.requests_dir / "not-a-request.txt").write_text("ignored")
    summary = broker.cancel_all()
    assert summary == {"cancelled": 2, "cleanup_verified": True, "workers_checked": 0}
    for request in pending:
        result = read_result(broker, request.request_id)
        assert result.status == "cancelled" and result.request_sha256 == request.content_sha256()
        assert result.cleanup_confirmed and not result.abort_campaign
    assert executor.calls == [] and broker.closed
    assert [row["event"] for row in ledger(broker)][-1] == "cancel_all"
    with pytest.raises(RuntimeError):
        broker.tick(300)
    assert broker.cancel_all() == {"cancelled": 0, "cleanup_verified": True, "workers_checked": 0}


def test_ledger_and_trace_bytes_are_charged_to_the_shared_artifact_budget(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run", 1_048_576)
    broker, _, _ = make_broker(tmp_path, artifacts=artifacts)
    after_open = artifacts.remaining_bytes
    assert after_open < 1_048_576  # the broker header line is already charged
    request = make_request(broker)
    drop(broker, request)
    broker.tick(300)
    broker.cancel_all()
    ledger_path = tmp_path / "run" / "broker" / "ledger.jsonl"
    trace_path = tmp_path / "run" / "broker" / "traces" / f"{request.request_id}.json"
    charged = ledger_path.stat().st_size + trace_path.stat().st_size
    assert artifacts.remaining_bytes == 1_048_576 - charged
    index = {entry["path"]: entry for entry in artifacts.index()}
    assert index[f"broker/traces/{request.request_id}.json"]["sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert index["broker/ledger.jsonl"]["size"] == ledger_path.stat().st_size

    tiny = RunArtifacts(tmp_path / "tiny", 4096)
    starved, _, _ = make_broker(tmp_path, artifacts=tiny, run_dir=tmp_path / "tiny")
    request = make_request(starved)
    drop(starved, request)
    starved.tick(300)  # the trace cannot be charged: the request fails closed instead of losing evidence silently
    result = read_result(starved, request.request_id)
    assert result.status == "environment-error" and result.failure_reason.startswith("trace_not_persisted")
    assert result.sample is None and result.trace_sha256 is None and result.cleanup_confirmed
    assert not (tmp_path / "tiny" / "broker" / "traces").exists() or not list((tmp_path / "tiny" / "broker" / "traces").iterdir())
    finished = [row for row in ledger(starved) if row["event"] == "finished"]
    assert finished[-1]["status"] == "environment-error"


def test_direct_client_uses_identical_validation_path(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    client = broker.direct_client()
    assert isinstance(client, DirectClient) and client.namespace() == {"session_id": "session-1", "attempt_id": ATTEMPT}
    wrong = make_request(broker, fixture_hash="0" * 64)
    client.submit(wrong)
    assert (broker.requests_dir / request_name(wrong.request_id)).exists()
    rejected = client.wait(wrong.request_id, deadline=broker.clock() + 30)
    assert rejected.status == "rejected" and rejected.failure_reason == "fixture_hash" and executor.calls == []
    good = make_request(broker)
    client.submit(good)
    completed = client.wait(good.request_id, deadline=broker.clock() + 60)
    assert completed.status == "completed" and completed.sample["passed"] is True
    assert runs(executor) == len(FIXTURE.checks)
    with pytest.raises(ValueError):
        client.wait("d" * 32, deadline=broker.clock() + 1)  # never submitted through this client
    kinds = [row["event"] for row in ledger(broker)]
    assert kinds.count("rejected") == 1 and kinds.count("validated") == 1


def test_factory_reads_namespace_from_plan_labels_and_rejects_conflicts(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "plan").mkdir(parents=True)
    plan = {"services": {"inference": {"labels": {"llmbench.session": "sess", "llmbench.attempt": "d" * 32}}}}
    (run_dir / "plan" / "compose.json").write_text(json.dumps(plan), encoding="utf-8")
    broker = HostBroker.factory(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"))
    assert (broker.session_id, broker.attempt_id) == ("sess", "d" * 32)
    header = ledger(broker)[0]
    assert header["namespace_source"] == "plan" and [row["event"] for row in ledger(broker)][-1] == "reconcile"
    broker.close()
    minted = HostBroker(tmp_path / "other", broker_config(), LOCK, RunArtifacts(tmp_path / "other").scoped("broker"))
    assert len(minted.attempt_id) == 32 and minted.session_id == "session-1"
    minted.close()
    (run_dir / "spool" / "results" / "namespace.json").write_text('{"attempt_id":"e","session_id":"x"}')
    with pytest.raises(ValueError, match="namespace"):
        HostBroker(run_dir, broker_config(), LOCK, RunArtifacts(run_dir).scoped("broker"))
    with pytest.raises(ValueError, match="broker"):
        HostBroker(tmp_path / "x", SimpleNamespace(broker=None, worker_image=None), LOCK,
                   RunArtifacts(tmp_path / "x").scoped("broker"))
    with pytest.raises(ValueError, match="worker_image"):
        HostBroker(tmp_path / "y", SimpleNamespace(broker=default_broker_settings(),
                                                   worker_image=SimpleNamespace(reference="python:latest")), LOCK,
                   RunArtifacts(tmp_path / "y").scoped("broker"))


def test_repaired_attempt_never_claims_first_attempt_success(tmp_path):
    broker, _, _ = make_broker(tmp_path)
    request = make_request(broker, attempt_index=2)
    drop(broker, request)
    broker.tick(300)
    sample = read_result(broker, request.request_id).sample
    assert sample["passed"] is True and sample["first_attempt_success"] is False and sample["attempt_index"] == 2


def test_environment_error_before_any_worker_does_not_abort(tmp_path):
    broker, _, _ = make_broker(tmp_path, executor=MultiFixtureExecutor(FIXTURE))
    broker.session_lock = SessionLock()  # container execution withdrawn before the worker starts
    request = make_request(broker)
    drop(broker, request)
    summary = broker.tick(300)
    result = read_result(broker, request.request_id)
    assert result.status == "environment-error" and "OperationForbidden" in result.failure_reason
    assert result.cleanup_confirmed and not result.abort_campaign and summary["abort_campaign"] is False


def load_acceptance_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "coding_broker_acceptance.py"
    spec = importlib.util.spec_from_file_location("coding_broker_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_script_refuses_synthetic_workers_and_verifies_rejections_offline(tmp_path):
    script = load_acceptance_script()
    rows = {variant: script.run_variant(tmp_path, FIXTURE, variant, IMAGE, LOCK, 12,
                                        executor=MultiFixtureExecutor(FIXTURE)) for variant in script.VARIANTS}
    for variant in ("malformed", "oversize", "traversal", "spool-oversize", "spool-traversal"):
        row = rows[variant]
        assert row["as_expected"] is True and row["reasons"] == [] and row["ledger_complete"] is True, row
        assert row["worker_names"] == [] and row["cancel_all"]["cleanup_verified"] is True
        assert row["abort_campaign"] is False
    assert rows["oversize"]["sample"]["error"] == "patch_too_large"
    assert rows["spool-oversize"]["result"]["failure_reason"] == "oversize"
    assert rows["spool-traversal"]["result"]["failure_reason"].startswith("schema: patch")
    for variant in ("reference", "initial", "hang"):  # a fake executor can never pass live acceptance
        row = rows[variant]
        assert row["as_expected"] is False and row["ledger_complete"] is True
        assert "Only actual Docker execution can pass live acceptance" in row["reasons"], row["reasons"]
        assert len(row["worker_names"]) == len(FIXTURE.checks) and row["sample"]["broker"]["status"] == "completed"
    assert rows["reference"]["sample"]["first_attempt_success"] is True
    assert rows["hang"]["reasons"] and any("Hanging source must fail" in reason for reason in rows["hang"]["reasons"])
    for variant in script.VARIANTS:
        run_dir = tmp_path / "runs" / f"python-chunks-{variant}"
        assert (run_dir / "broker" / "ledger.jsonl").exists()
        if not variant.startswith("spool-"):
            assert (run_dir / "responses" / "coding" / "python-chunks" / "response-1.json").exists()
    summary = script.summary_of(list(rows.values()), expected_rows=len(script.VARIANTS), checks=[], image=IMAGE,
                                error=None)
    assert summary["accepted"] is False and summary["complete"] is True and summary["all_ledgers_complete"] is True
    assert summary["absence_query_verified"] is False  # no absence query ran: never accepted
    accepted = script.summary_of([rows["malformed"]], expected_rows=1, image=IMAGE, error=None,
                                 checks=[{"name": "llmbench-" + "0" * 32, "verified_absent": True}])
    assert accepted["accepted"] is True
    assert script.summary_of([rows["malformed"]], expected_rows=1, image=IMAGE, error="boom",
                             checks=accepted["absence_checks"])["accepted"] is False
    complete, problems = script.ledger_complete([{"event": "broker"}, {"event": "validated", "request_id": "r"},
                                                 {"event": "cancel_all"}])
    assert complete is False and problems == ["r: validated without finished+published"]
    for fixture in fixtures():
        for variant in ("reference", "initial", "hang", "malformed", "oversize", "traversal"):
            assert type(script.response_for(fixture, variant, default_broker_settings())) is str


@pytest.mark.parametrize("reason", ["Docker startup exit 125", "driver copy failed", "test toolchain missing",
                                    "protocol-error: malformed JSON"])
def test_public_coding_infrastructure_samples_are_terminal_environment_errors(tmp_path, reason):
    broker, _, artifacts = make_broker(tmp_path)
    request = make_request(broker)
    outcome = {"sample": {"status": "environment-error", "passed": False, "model_evaluated": False,
                           "failure_reason": reason, "cases": [{"case_id": "pinned-tests",
                                                                  "status": "environment-error"}]},
               "raw_bytes": b'{"evidence":"retained"}', "abort_campaign": False}
    result = broker._outcome_result(request, request.content_sha256(), outcome)
    assert result.status == "environment-error" and result.sample is None
    assert result.failure_reason == reason and result.cleanup_confirmed and not result.abort_campaign
    assert result.trace_sha256 == hashlib.sha256(outcome["raw_bytes"]).hexdigest()
