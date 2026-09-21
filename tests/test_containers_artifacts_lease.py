import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from llmbench.containers.artifacts import RunArtifacts
from llmbench.containers.lease import GpuLease, LeaseHeld, default_lease_path, gpu_lease


def test_writes_are_confined_exclusive_and_indexed(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    target = artifacts.write("plan/compose.json", b"{}")
    assert target == (tmp_path / "run" / "plan" / "compose.json").resolve() and target.read_bytes() == b"{}"
    with pytest.raises(FileExistsError):
        artifacts.write("plan/compose.json", b"changed")
    assert target.read_bytes() == b"{}"
    for unsafe in ("../escape.txt", "/absolute.txt", "C:\\windows.txt", "a\\b.txt", "a/../b.txt", "a//b.txt",
                   "./a.txt", "", "a/", "stream.txt:ads", "nul\x00.txt"):
        with pytest.raises(ValueError):
            artifacts.write(unsafe, b"x")
    with pytest.raises(TypeError):
        artifacts.write("text.txt", "not bytes")
    assert not (tmp_path / "escape.txt").exists()
    artifacts.write_json("result.json", {"b": 1, "a": (1, 2), "path": Path("x")})
    assert json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8")) == {"a": [1, 2], "b": 1,
                                                                                           "path": "x"}
    index = {entry["path"]: entry for entry in artifacts.index()}
    assert list(index) == ["plan/compose.json", "result.json"]
    assert index["plan/compose.json"] == {"path": "plan/compose.json", "size": 2,
                                          "sha256": hashlib.sha256(b"{}").hexdigest()}


def test_linked_directories_cannot_redirect_writes(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, tmp_path / "run" / "logs", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable to this account")
    with pytest.raises(ValueError, match="escapes"):
        artifacts.write("logs/startup.log", b"x")
    assert not list(outside.iterdir())


def test_byte_budget_counts_existing_and_streamed_bytes(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "earlier.bin").write_bytes(b"x" * 60)
    artifacts = RunArtifacts(root, max_bytes=100)
    artifacts.write("a.bin", b"y" * 30)
    with pytest.raises(ValueError, match="budget"):
        artifacts.write("b.bin", b"z" * 11)
    assert not (root / "b.bin").exists()
    with pytest.raises(ValueError, match="budget"):
        with artifacts.trace("events.jsonl") as sink:
            sink({"n": 1})
            sink({"payload": "x" * 100})
    assert (root / "events.jsonl").read_text(encoding="utf-8") == '{"n":1}\n'


def test_trace_is_sealed_and_indexed_when_the_body_raises(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    with pytest.raises(RuntimeError):
        with artifacts.trace("quality/transport.jsonl") as sink:
            sink({"event": "first"})
            raise RuntimeError("evaluation failed")
    content = (tmp_path / "run" / "quality" / "transport.jsonl").read_bytes()
    assert content == b'{"event":"first"}\n'
    assert artifacts.index() == [{"path": "quality/transport.jsonl", "size": len(content),
                                  "sha256": hashlib.sha256(content).hexdigest()}]
    with pytest.raises(FileExistsError):
        with artifacts.trace("quality/transport.jsonl"):
            pass


def test_adopted_files_and_written_index(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.write("config.json", b"{}")
    reports = tmp_path / "run" / "reports"
    reports.mkdir()
    (reports / "report.md").write_bytes(b"# report")
    artifacts.adopt_tree("reports")
    artifacts.adopt_tree("missing")
    index_path = artifacts.write_index()
    saved = json.loads(index_path.read_text(encoding="utf-8"))
    assert [entry["path"] for entry in saved["artifacts"]] == ["config.json", "reports/report.md"]
    assert saved["artifacts"][1]["sha256"] == hashlib.sha256(b"# report").hexdigest()
    artifacts.adopt_tree()
    assert "artifact-index.json" not in [entry["path"] for entry in artifacts.index()]
    with pytest.raises(FileExistsError):
        artifacts.write_index()


def test_second_lease_raises_and_is_never_broken(tmp_path):
    path = tmp_path / "gpu.lock"
    with gpu_lease(path, owner="first") as lease:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["pid"] == os.getpid() and record["owner"] == "first" and lease.held
        with pytest.raises(LeaseHeld, match=r"held by pid \d+ \(first\).*still running"):
            with gpu_lease(path):
                pass
        second = GpuLease(path)
        with pytest.raises(LeaseHeld):
            second.acquire()
        second.release()  # A lease that was never acquired must not delete the owner's file.
        assert path.exists()
    assert not path.exists()


def test_stale_lease_needs_explicit_recovery_and_release_is_idempotent(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text('{"pid": 999999, "created": "2020-01-01T00:00:00+00:00"}', encoding="utf-8")
    with pytest.raises(LeaseHeld):
        GpuLease(path).acquire()
    assert "999999" in path.read_text(encoding="utf-8")
    path.unlink()
    lease = GpuLease(path).acquire()
    lease.release()
    lease.release()
    assert not path.exists()
    with pytest.raises(RuntimeError):
        with gpu_lease(path):
            raise RuntimeError("released on error")
    assert not path.exists()


def test_default_path_is_shared_with_lm_studio_campaigns():
    assert default_lease_path() == Path(tempfile.gettempdir()) / "llmbench-gpu-resource.lock"
    assert GpuLease().path == default_lease_path()



def test_external_writer_charges_growth_before_disk_and_allows_zip_rewrites(tmp_path):
    artifacts = RunArtifacts(tmp_path, max_bytes=100)
    with artifacts.open_external("logs/run.json", "wb+") as handle:
        handle.write(b"x" * 60)
        assert artifacts.remaining_bytes == 40
        handle.seek(0)
        handle.write(b"head")
        assert artifacts.remaining_bytes == 40
        handle.seek(60)
        with pytest.raises(ValueError, match="budget"):
            handle.write(b"y" * 41)
        assert (tmp_path / "logs/run.json").stat().st_size == 60
        handle.truncate(20)
        assert artifacts.remaining_bytes == 80
        handle.seek(0)
        assert handle.read(4) == b"head"
    with artifacts.open_external("logs/run.json", "ab") as handle:
        handle.write(b"tail")
    assert artifacts.remaining_bytes == 76
    with artifacts.open_external("logs/run.json", "rb") as handle:
        assert handle.read().endswith(b"tail")
    entry = artifacts.index()[0]
    assert entry["size"] == 24 and entry["sha256"] == hashlib.sha256((tmp_path / entry["path"]).read_bytes()).hexdigest()
    artifacts.remove_external("logs/run.json")
    assert artifacts.remaining_bytes == 100 and artifacts.index() == []
    assert not (tmp_path / "logs/run.json").exists()


def test_external_writers_cannot_replace_or_delete_sealed_raw_evidence(tmp_path):
    artifacts = RunArtifacts(tmp_path)
    artifacts.write("raw.json", b"{}")
    with pytest.raises(FileExistsError):
        artifacts.open_external("raw.json", "wb")
    with pytest.raises(FileNotFoundError):
        artifacts.remove_external("raw.json")
    with pytest.raises(ValueError):
        artifacts.open_external("../escape.json", "wb")
    assert (tmp_path / "raw.json").read_bytes() == b"{}"


def test_reserved_terminal_budget_remains_within_total_and_freezes_logging(tmp_path):
    artifacts = RunArtifacts(tmp_path, max_bytes=16384, terminal_reserve_bytes=8192)
    with artifacts.open_external("log.json", "wb") as handle:
        handle.write(b"x" * artifacts.remaining_bytes)
    assert artifacts.remaining_bytes == 0
    with pytest.raises(ValueError, match="budget"):
        artifacts.write("more", b"x")
    assert artifacts.terminal_fits({"state": "failed", "cleanup_verified": True})
    artifacts.finalize_terminal({"state": "failed", "cleanup_verified": True})
    index = json.loads((tmp_path / "artifact-index.json").read_text())
    assert {entry["path"] for entry in index["artifacts"]} == {"log.json", "result.json"}
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= 16384
    with pytest.raises(ValueError, match="finalized"):
        artifacts.open_external("new.json", "wb")


def test_many_empty_logs_cannot_exhaust_the_reserved_index_capacity(tmp_path):
    artifacts = RunArtifacts(tmp_path, max_bytes=16384, terminal_reserve_bytes=8192)
    created = 0
    with pytest.raises(ValueError, match="index metadata budget"):
        for index in range(1000):
            artifacts.write(f"logs/{index:04d}.json", b"")
            created += 1
    assert 1 <= created < 100
    artifacts.finalize_terminal({"state": "failed", "reason": "metadata budget"})
    assert (tmp_path / "result.json").exists()
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= 16384


def test_adoption_and_standalone_index_cannot_bypass_total_cap(tmp_path):
    artifacts = RunArtifacts(tmp_path, max_bytes=100)
    (tmp_path / "outside-writer.bin").write_bytes(b"x" * 101)
    with pytest.raises(ValueError, match="budget"):
        artifacts.adopt_tree()
    (tmp_path / "outside-writer.bin").unlink()
    artifacts.write("small.bin", b"x" * 90)
    with pytest.raises(ValueError, match="budget"):
        artifacts.write_index()
    assert not (tmp_path / "artifact-index.json").exists()


def test_reserve_child_charges_upfront_and_adopt_reconciles(tmp_path):
    artifacts = RunArtifacts(tmp_path, max_bytes=100_000, terminal_reserve_bytes=10_000)
    artifacts.reserve_child("evaluator", 60_000)
    assert artifacts.remaining_bytes == 30_000
    with pytest.raises(ValueError, match="budget"):
        artifacts.write("big.bin", b"x" * 30_001)
    with pytest.raises(ValueError, match="already reserved"):
        artifacts.reserve_child("evaluator", 1)
    child = tmp_path / "evaluator"
    child.mkdir(exist_ok=True)  # reserve_child already confined and created the prefix
    (child / "evaluation.json").write_bytes(b"{}")
    (child / "quality").mkdir()
    (child / "quality" / "request-0.json").write_bytes(b"x" * 1000)
    adoption = artifacts.adopt_child("evaluator")
    assert adoption == {"bytes": 1002, "files": 2, "over_allocation": False, "unindexed": [], "errors": []}
    assert artifacts.remaining_bytes == 90_000 - 1002
    paths = {entry["path"]: entry for entry in artifacts.index()}
    assert set(paths) == {"evaluator/evaluation.json", "evaluator/quality/request-0.json"}
    assert paths["evaluator/evaluation.json"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(ValueError, match="no child allocation"):
        artifacts.adopt_child("evaluator")
    artifacts.adopt_tree("evaluator")  # already indexed entries are left alone
    assert len(artifacts.index()) == 2
    with pytest.raises(ValueError, match="already holds"):
        artifacts.reserve_child("evaluator", 10)
    for bad in (0, -1, 1.5, "10"):
        with pytest.raises(ValueError):
            artifacts.reserve_child("other", bad)


def test_adopt_child_flags_over_allocation_without_escaping_cap(tmp_path):
    artifacts = RunArtifacts(tmp_path, max_bytes=20_000, terminal_reserve_bytes=8_000)
    artifacts.write("config.json", b"c" * 1_000)
    artifacts.reserve_child("evaluator", 4_000)
    child = tmp_path / "evaluator"
    child.mkdir(exist_ok=True)  # reserve_child already confined and created the prefix
    (child / "a.bin").write_bytes(b"a" * 3_000)
    (child / "b.bin").write_bytes(b"b" * 3_000)  # 6_000 written against a 4_000 allocation
    (child / "c.bin").write_bytes(b"c" * 9_000)  # would breach the data cap: stays unindexed
    adoption = artifacts.adopt_child("evaluator")
    assert adoption["over_allocation"] is True and adoption["bytes"] == 15_000
    assert adoption["files"] == 2 and adoption["unindexed"] == ["evaluator/c.bin"]
    assert artifacts.remaining_bytes == 12_000 - 1_000 - 6_000
    assert artifacts.terminal_fits({"state": "failed"})
    artifacts.finalize_terminal({"state": "failed", "reason": "over allocation"})
    index = json.loads((tmp_path / "artifact-index.json").read_text())
    assert {entry["path"] for entry in index["artifacts"]} == {"config.json", "evaluator/a.bin", "evaluator/b.bin",
                                                               "result.json"}
    indexed = sum(entry["size"] for entry in index["artifacts"]) + (tmp_path / "artifact-index.json").stat().st_size
    assert indexed <= 20_000


def test_adopt_child_records_unreadable_files_without_hiding_the_allocation_verdict(tmp_path, monkeypatch):
    """REV-A1-03: one child file that cannot be hashed must not abort adoption; its bytes still count against
    the allocation and the error is reported so the caller fails the candidate instead of completing it."""
    artifacts = RunArtifacts(tmp_path, max_bytes=100_000, terminal_reserve_bytes=10_000)
    artifacts.reserve_child("evaluator", 4_000)
    child = tmp_path / "evaluator"
    (child / "evaluation.json").write_bytes(b"{}")
    (child / "quality").mkdir()
    (child / "quality" / "big.bin").write_bytes(b"q" * 6_000)  # 6_002 written against a 4_000 allocation
    original = RunArtifacts._seal_external

    def flaky_seal(self, name):
        if name == "evaluator/quality/big.bin":
            raise PermissionError(13, "simulated unreadable child file", name)
        return original(self, name)
    monkeypatch.setattr(RunArtifacts, "_seal_external", flaky_seal)
    adoption = artifacts.adopt_child("evaluator")
    assert adoption["bytes"] == 6_002 and adoption["over_allocation"] is True and adoption["files"] == 1
    assert adoption["unindexed"] == ["evaluator/quality/big.bin"]
    assert len(adoption["errors"]) == 1 and adoption["errors"][0].startswith("evaluator/quality/big.bin: PermissionError")
    assert [entry["path"] for entry in artifacts.index()] == ["evaluator/evaluation.json"]
    assert artifacts.remaining_bytes == 90_000 - 2  # the placeholder is released; only the indexed file is charged
    assert artifacts.terminal_fits({"state": "failed"})
    artifacts.finalize_terminal({"state": "failed", "reason": "unreadable child file"})
    index = json.loads((tmp_path / "artifact-index.json").read_text())
    assert {entry["path"] for entry in index["artifacts"]} == {"evaluator/evaluation.json", "result.json"}
    for entry in index["artifacts"]:
        content = (tmp_path / entry["path"]).read_bytes()
        assert (hashlib.sha256(content).hexdigest(), len(content)) == (entry["sha256"], entry["size"])


def test_child_prefix_cannot_be_terminal_or_linked(tmp_path):
    artifacts = RunArtifacts(tmp_path, max_bytes=100_000, terminal_reserve_bytes=10_000)
    for bad in ("result.json", "artifact-index.json", "a/b", "", "../x", "evaluator.json", 5):
        with pytest.raises(ValueError):
            artifacts.reserve_child(bad, 100)
    assert artifacts.remaining_bytes == 90_000
    outside = tmp_path.parent / "outside-child"
    outside.mkdir(exist_ok=True)
    try:
        os.symlink(outside, tmp_path / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable to this account")
    with pytest.raises(ValueError, match="escapes|link"):
        artifacts.reserve_child("linked", 100)
    artifacts.reserve_child("evaluator", 1000)
    child = tmp_path / "evaluator"
    child.mkdir(exist_ok=True)  # reserve_child already confined and created the prefix
    (outside / "secret.bin").write_bytes(b"s" * 10)
    os.symlink(outside / "secret.bin", child / "link.bin")
    (child / "real.bin").write_bytes(b"r" * 10)
    adoption = artifacts.adopt_child("evaluator")
    assert adoption["files"] == 1 and [entry["path"] for entry in artifacts.index()] == ["evaluator/real.bin"]
