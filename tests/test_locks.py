"""A held lock must say who holds it and whether that process still exists - and nothing may ever break one."""

import json
import os
import subprocess
import sys

import pytest

from llmbench.containers.lease import GpuLease, LeaseHeld
from llmbench.locks import describe_lock, process_alive
from llmbench.store import Store


def finished_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_process_probe_never_signals_and_reads_liveness():
    assert process_alive(os.getpid()) is True
    assert process_alive(finished_pid()) is False
    for nonsense in (0, -1, True, "12", None):
        assert process_alive(nonsense) is None
    assert process_alive(os.getpid()) is True  # probing our own pid did not terminate us (Windows os.kill would)


def test_a_stale_lock_is_named_as_stale_and_left_in_place(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text(json.dumps({"pid": finished_pid(), "created": "2026-09-20T21:42:41+00:00", "owner": "run:abc"}))
    text = describe_lock(path)
    assert "no longer running" in text and "stale" in text and "run:abc" in text and "docker ps" in text
    with pytest.raises(LeaseHeld, match="stale"):
        GpuLease(path).acquire()
    assert path.exists()  # describing a lock never removes it


def test_a_live_holder_and_an_unreadable_lock_are_reported_differently(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text(json.dumps({"pid": os.getpid(), "created": "now"}))
    assert "still running" in describe_lock(path) and "reused pid" in describe_lock(path)
    path.write_bytes(b"\xff not json")
    assert "does not name its holder" in describe_lock(path)


def test_the_campaign_lock_reports_its_holder_instead_of_a_bare_errno(tmp_path):
    lock = tmp_path / "resource.lock"
    lock.write_text(json.dumps({"pid": finished_pid(), "created": "then"}))
    with Store(tmp_path / "store") as store:
        with pytest.raises(FileExistsError, match="another campaign holds the lock.*stale"):
            with store.campaign_lock(lock):
                pytest.fail("the lock must not be entered")
    assert lock.exists()
