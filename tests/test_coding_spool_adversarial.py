"""Adversarial spool content: the host broker and the evaluator client trust nothing on disk."""

import json
import os
import sys
from pathlib import Path

import pytest

from llmbench.coding import spool
from llmbench.coding.broker import MAX_REJECTIONS_PER_TICK
from llmbench.coding.broker_client import SpoolClient
from llmbench.coding.spool import (
    CodingJobResult, NotRegularFile, Oversize, SpoolIntegrityError, publish_atomic, read_bounded, request_bytes,
    request_name, result_bytes,
)
from llmbench.containers.config import ContainerRunConfig
from llmbench.evaluations.tools import strict_json_loads

from test_coding_broker import FIXTURE, drop, ledger, make_broker, make_request, read_result

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NAME = "1" * 32 + ".json"


def raw_request(broker, **overrides):
    data = json.loads(request_bytes(make_request(broker)))
    data.update(overrides)
    return data


def write_raw(broker, name, payload: bytes):
    path = broker.requests_dir / name
    path.write_bytes(payload)
    return path


def assert_ignored(broker, path, reason):
    request, why = broker.validate(path)
    assert request is None and why == reason
    summary = broker.tick(300)
    assert summary["processed"] == 0 and summary["rejected"] == 0
    assert not (broker.results_dir / path.name).exists()
    ignored = [row for row in ledger(broker) if row["event"] == "ignored"]
    assert ignored and ignored[-1] == {**ignored[-1], "name": path.name, "reason": reason}


def assert_rejected(broker, path, reason_prefix):
    request, why = broker.validate(path)
    assert request is None and why.startswith(reason_prefix), why
    summary = broker.tick(300)
    assert summary["processed"] == 0 and summary["rejected"] == 1
    result = read_result(broker, path.name[:-5])
    assert result.status == "rejected" and result.failure_reason.startswith(reason_prefix)
    return result


def test_symlink_request_is_ignored(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    real = tmp_path / "real.json"
    real.write_bytes(request_bytes(make_request(broker, request_id="1" * 32)))
    try:
        os.symlink(real, broker.requests_dir / NAME)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation needs privileges on this host")
    assert_ignored(broker, broker.requests_dir / NAME, "not_regular_file")
    with pytest.raises(NotRegularFile):
        read_bounded(broker.requests_dir / NAME, 1_000_000)
    assert executor.calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions are Windows-only reparse points")
def test_junction_or_reparse_request_is_ignored(tmp_path):
    import _winapi
    broker, executor, _ = make_broker(tmp_path)
    target = tmp_path / "target-dir"
    target.mkdir()
    _winapi.CreateJunction(str(target), str(broker.requests_dir / NAME))
    assert_ignored(broker, broker.requests_dir / NAME, "not_regular_file")
    with pytest.raises(NotRegularFile):
        read_bounded(broker.requests_dir / NAME, 1_000_000)
    assert executor.calls == []


def test_hardlinked_request_is_ignored(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    original = tmp_path / "original.json"
    original.write_bytes(request_bytes(make_request(broker, request_id="1" * 32)))
    try:
        os.link(original, broker.requests_dir / NAME)
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")
    assert_ignored(broker, broker.requests_dir / NAME, "not_regular_file")
    with pytest.raises(NotRegularFile, match="hard-linked"):
        read_bounded(broker.requests_dir / NAME, 1_000_000)
    assert executor.calls == []


def test_oversize_request_is_rejected_without_full_read(tmp_path, monkeypatch):
    broker, executor, _ = make_broker(tmp_path, settings={"max_request_bytes": 4096})
    path = write_raw(broker, NAME, b"{" + b" " * 8000 + b"}")

    def never(*args, **kwargs):
        raise AssertionError("an oversize file must be rejected from lstat alone")
    monkeypatch.setattr(spool, "_open_file", never)
    with pytest.raises(Oversize):
        read_bounded(path, 4096)
    assert broker.validate(path) == (None, "oversize")
    summary = broker.tick(300)
    assert summary["processed"] == 0 and summary["rejected"] == 1
    monkeypatch.undo()
    result = read_result(broker, path.name[:-5])
    assert result.status == "rejected" and result.failure_reason == "oversize"
    assert result.cleanup_confirmed and executor.calls == []
    exact = write_raw(broker, "2" * 32 + ".json", b"x" * 4096)
    assert read_bounded(exact, 4096) == b"x" * 4096
    with pytest.raises(Oversize):
        read_bounded(exact, 4095)


def test_request_swapped_between_lstat_and_open_is_refused(tmp_path, monkeypatch):
    """TOCTOU: a regular file replaced by another regular file after inspection fails the fstat identity recheck."""
    broker, executor, _ = make_broker(tmp_path)
    request = make_request(broker, request_id="1" * 32)
    path = drop(broker, request)
    original = request_bytes(request)
    other = request_bytes(make_request(broker, request_id="1" * 32, patch={"batches.py": "def chunked(v, s): return []\n"}))
    real_open = spool._open_file

    def swap_then_open(target, flags):
        swapped = Path(str(target) + ".swap")
        swapped.write_bytes(other.ljust(len(original), b" "))  # same size, new inode: only the identity check catches it
        os.replace(swapped, target)
        return real_open(target, flags)
    monkeypatch.setattr(spool, "_open_file", swap_then_open)
    with pytest.raises(NotRegularFile, match="changed between inspection and open"):
        read_bounded(path, 2_000_000)
    assert broker.validate(path) == (None, "not_regular_file")
    summary = broker.tick(300)
    assert summary["processed"] == 0 and summary["rejected"] == 0 and executor.calls == []
    monkeypatch.undo()
    assert not (broker.results_dir / path.name).exists() and list(broker.workers_root.iterdir()) == []
    assert [row["reason"] for row in ledger(broker) if row["event"] == "ignored"] == ["not_regular_file"]


def test_bad_name_traversal_and_absolute_paths_ignored(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    payload = request_bytes(make_request(broker, request_id="1" * 32))
    names = ["..json", "2" * 32 + ".JSON", "1" * 32, "1" * 31 + ".json", "A" * 32 + ".json", "1" * 32 + ".json.bak",
             "-" + NAME, ".1" * 16 + ".json"]
    assert broker.validate(broker.requests_dir / (NAME + " ")) == (None, "bad_name")  # never written: NTFS trims it
    for name in names:
        write_raw(broker, name, payload)
    for name in names:
        assert broker.validate(broker.requests_dir / name) == (None, "bad_name")
    # A perfectly valid request outside the owned spool directory is never admitted, however it is addressed.
    outside = broker.run_dir / "spool" / NAME
    outside.write_bytes(payload)
    assert broker.validate(outside) == (None, "bad_name")
    assert broker.validate(broker.requests_dir / ".." / NAME) == (None, "bad_name")
    assert broker.validate(Path("/") / NAME) == (None, "bad_name")
    assert broker.validate(Path(NAME)) == (None, "bad_name")
    summary = broker.tick(300)
    assert summary == {"processed": 0, "rejected": 0, "pending": 0, "abort_campaign": False}
    assert [p.name for p in broker.results_dir.iterdir()] == ["namespace.json"] and executor.calls == []
    assert broker.validate(broker.requests_dir / NAME) == (None, "not_regular_file")  # absent
    for bad in ("../x", "1" * 31, "/" + "1" * 32, "1" * 32 + "/"):
        with pytest.raises(ValueError):
            request_name(bad)


def test_patch_outside_allowlist_rejected(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    path = drop(broker, make_request(broker, request_id="1" * 32,
                                     patch={"batches.py": "x\n", "evaluator.py": "print('escape')\n"}))
    assert_rejected(broker, path, "patch_outside_allowlist")
    assert executor.calls == [] and not list((broker.workers_root).iterdir())


def test_patch_bytes_cap_and_nul_rejected(tmp_path):
    broker, executor, _ = make_broker(tmp_path, settings={"max_patch_bytes": 1024})
    large = drop(broker, make_request(broker, request_id="1" * 32, patch={"batches.py": "#" * 2000 + "\n"}))
    assert_rejected(broker, large, "patch_too_large")
    nul = write_raw(broker, "2" * 32 + ".json", json.dumps(raw_request(broker, request_id="2" * 32,
                                                                       patch={"batches.py": "x\u0000y"})).encode())
    assert_rejected(broker, nul, "schema: patch")
    surrogate = write_raw(broker, "3" * 32 + ".json",
                          json.dumps(raw_request(broker, request_id="3" * 32, patch={"batches.py": "\ud800"})).encode())
    assert_rejected(broker, surrogate, "schema: patch")
    assert executor.calls == []


def test_traversal_patch_keys_in_raw_requests_are_rejected_before_staging(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    keys = ["../escape.py", "/etc/passwd", "sub/../batches.py", "batches.py/", "a\\b.py", ".", "..", "",
            ".llmbench-driver.py", "C:batches.py", "batches.py\n"]
    for index, key in enumerate(keys, 1):
        name = f"{index:032x}.json"
        path = write_raw(broker, name, json.dumps(raw_request(broker, request_id=name[:-5],
                                                              patch={key: "print(1)\n"})).encode())
        request, reason = broker.validate(path)
        assert request is None and reason.startswith(("schema: patch", "patch_outside_allowlist")), (key, reason)
    summary = broker.tick(300)  # cheap rejections are bounded per tick (MAX_REJECTIONS_PER_TICK), never a worker
    assert summary["processed"] == 0 and summary["rejected"] == len(keys) <= MAX_REJECTIONS_PER_TICK
    assert broker.tick(300) == {"processed": 0, "rejected": 0, "pending": 0, "abort_campaign": False}
    for index, _ in enumerate(keys, 1):
        assert read_result(broker, f"{index:032x}").status == "rejected"
    assert executor.calls == [] and list(broker.workers_root.iterdir()) == []


def test_foreign_session_or_attempt_namespace_rejected(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    other_attempt = drop(broker, make_request(broker, request_id="1" * 32, attempt_id="f" * 32))
    assert_rejected(broker, other_attempt, "foreign_namespace")
    other_session = drop(broker, make_request(broker, request_id="2" * 32, session_id="someone-else"))
    assert_rejected(broker, other_session, "foreign_namespace")
    mismatch = drop(broker, make_request(broker, request_id="3" * 32))
    renamed = mismatch.with_name("4" * 32 + ".json")
    mismatch.rename(renamed)
    assert_rejected(broker, renamed, "request_id_mismatch")
    assert executor.calls == []


def test_wrong_fixture_hash_or_revision_rejected(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    assert_rejected(broker, drop(broker, make_request(broker, request_id="1" * 32, fixture_hash="9" * 64)),
                    "fixture_hash")
    revision = write_raw(broker, "2" * 32 + ".json",
                         json.dumps(raw_request(broker, request_id="2" * 32, fixture_revision="private-coding-v2")).encode())
    # A well-formed revision the host fixture does not declare is refused by the broker's own fixture check (the
    # schema now admits pinned public-benchmark revisions); a malformed one is still refused by the schema.
    assert_rejected(broker, revision, "fixture_revision")
    malformed = write_raw(broker, "4" * 32 + ".json",
                          json.dumps(raw_request(broker, request_id="4" * 32, fixture_revision="bad rev")).encode())
    assert_rejected(broker, malformed, "schema: fixture_revision")
    assert_rejected(broker, drop(broker, make_request(broker, request_id="3" * 32, fixture_id="python/unknown")),
                    "unknown_fixture")
    assert executor.calls == []


def test_invalid_utf8_or_duplicate_keys_rejected(tmp_path):
    broker, executor, _ = make_broker(tmp_path)
    assert_rejected(broker, write_raw(broker, "1" * 32 + ".json", b"\xff\xfe{}"), "invalid_utf8")
    text = json.dumps(raw_request(broker, request_id="2" * 32))
    duplicated = text[:-1] + ', "attempt_index": 1}'
    assert_rejected(broker, write_raw(broker, "2" * 32 + ".json", duplicated.encode()), "invalid_json")
    assert_rejected(broker, write_raw(broker, "3" * 32 + ".json", b"[1, 2]"), "invalid_json")
    assert_rejected(broker, write_raw(broker, "4" * 32 + ".json", b'{"a": NaN}'), "invalid_json")
    assert_rejected(broker, write_raw(broker, "5" * 32 + ".json", b"\xef\xbb\xbf{}"), "invalid_json")
    assert executor.calls == []


def test_result_directory_write_by_client_is_impossible_in_plan(tmp_path):
    from llmbench.containers.plan import build_compose_plan
    raw = json.loads((EXAMPLES / "candidate.json").read_text(encoding="utf-8"))
    raw["evaluator"] = {"mode": "container", "image": {"role": "evaluator", "reference": "sha256:" + "b" * 64,
                                                       "image_id": "sha256:" + "b" * 64,
                                                       "entrypoint": ["python", "-m", "llmbench.container_eval"]}}
    raw["worker_image"] = {"role": "worker", "reference": "sha256:" + "c" * 64, "image_id": "sha256:" + "c" * 64,
                           "entrypoint": ["python3"]}
    raw["broker"] = {}
    raw["benchmarks"].append({"benchmark_id": "coding", "revision": "private-coding-v1", "task_ids": ["python/chunks"]})
    try:
        from llmbench.containers.config import ChildGrant
        config = ContainerRunConfig.model_validate_json(json.dumps(raw))
        grant = ChildGrant(grant_seconds=600, artifact_bytes=config.limits.evaluator_artifact_bytes,
                           issued_utc="2026-09-18T00:00:00+00:00", issued_host_offset_seconds=0.0,
                           watchdog_slack_seconds=30)
        plan = build_compose_plan(config, tmp_path / "run", attempt_id="0" * 32, session_id="s", grant=grant)
    except (ImportError, ValueError, TypeError) as exc:  # Branch A's container-mode plan is landing concurrently
        pytest.skip(f"Branch A container-mode plan with a broker is not available yet: {exc}")
    evaluator = plan.compose["services"]["evaluator"]
    by_target = {volume["target"]: volume for volume in evaluator["volumes"]}
    assert by_target["/spool/results"]["read_only"] is True
    assert by_target["/spool/results"]["source"] == str(tmp_path / "run" / "spool" / "results")
    assert by_target["/spool/requests"].get("read_only") is not True
    assert by_target["/spool/requests"]["source"] == str(tmp_path / "run" / "spool" / "requests")
    assert evaluator["read_only"] is True and evaluator["user"] == "10001:10001"
    assert "docker.sock" not in plan.compose_json() and "broker" not in json.dumps(evaluator["volumes"])


def test_result_replaced_after_publish_is_detected_by_request_sha(tmp_path):
    broker, _, _ = make_broker(tmp_path)
    client = SpoolClient(broker.requests_dir, broker.results_dir, clock=lambda: 0.0, sleep=lambda s: None)
    assert client.namespace() == {"session_id": broker.session_id, "attempt_id": broker.attempt_id}
    request = make_request(broker)
    client.submit(request)
    broker.tick(300)
    genuine = client.wait(request.request_id, deadline=1.0)
    assert genuine.status == "completed"
    forged = genuine.model_copy(update={"request_sha256": "0" * 64})
    path = broker.results_dir / request_name(request.request_id)
    path.unlink()
    path.write_bytes(result_bytes(forged))
    with pytest.raises(SpoolIntegrityError):
        client.wait(request.request_id, deadline=1.0)
    other = genuine.model_copy(update={"request_id": "9" * 32})
    path.unlink()
    path.write_bytes(result_bytes(other))
    with pytest.raises(TimeoutError):
        client.wait(request.request_id, deadline=0.0)


def test_client_wait_ignores_oversize_or_malformed_results_and_times_out(tmp_path):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    broker, _, _ = make_broker(tmp_path)
    client = SpoolClient(broker.requests_dir, broker.results_dir, clock=lambda: now[0], sleep=sleep)
    request = make_request(broker)
    client.submit(request)
    stored = strict_json_loads(request_bytes(request).decode("utf-8"))
    assert stored["request_id"] == request.request_id  # the request file is the exact strict contract
    path = broker.results_dir / request_name(request.request_id)
    for payload in (b"not json", b"[]", b'{"request_id": "x"}', b"\xff\xfe",
                    json.dumps({"schema_version": 1, "request_id": request.request_id, "request_sha256": "0" * 64,
                                "status": "completed", "sample": {"passed": True}, "cleanup_confirmed": True,
                                "abort_campaign": False, "finished_utc": "t", "extra": 1}).encode()):
        path.write_bytes(payload)
        assert client.poll(request.request_id) is None
        path.unlink()
    valid = CodingJobResult(request_id=request.request_id, request_sha256=request.content_sha256(),
                            status="rejected", failure_reason="x" * 10, cleanup_confirmed=True, abort_campaign=False,
                            finished_utc="t")
    path.write_bytes(result_bytes(valid))
    assert client.poll(request.request_id, max_bytes=64) is None  # over the client's byte cap: ignored
    assert client.poll(request.request_id).status == "rejected"
    path.unlink()
    with pytest.raises(TimeoutError):
        client.wait(request.request_id, deadline=2.0, poll_seconds=0.5)
    assert sleeps == [0.5, 0.5, 0.5, 0.5] and now[0] == 2.0
    with pytest.raises(ValueError):
        client.poll("2" * 32)
    with pytest.raises(TypeError):
        client.submit({"request_id": "2" * 32})


def test_read_bounded_rejects_directories_and_reports_size_before_reading(tmp_path):
    directory = tmp_path / "dir.json"
    directory.mkdir()
    with pytest.raises(NotRegularFile):
        read_bounded(directory, 10)
    with pytest.raises(FileNotFoundError):
        read_bounded(tmp_path / "absent.json", 10)
    with pytest.raises(ValueError):
        read_bounded(tmp_path / "absent.json", 0)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    target = publish_atomic(spool_dir, "1" * 32 + ".json", b"abc")
    assert read_bounded(target, 3) == b"abc"
    assert FIXTURE.fixture_id == "python/chunks"
