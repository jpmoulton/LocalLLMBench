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


def test_lock_texts_written_before_runtimes_existed_are_unchanged(tmp_path):
    path, gone = tmp_path / "gpu.lock", finished_pid()
    path.write_text(json.dumps({"pid": gone, "created": "2026-09-20T21:42:41+00:00", "owner": "container-run:abc"}))
    assert describe_lock(path) == (
        f"{path} is held by pid {gone} (container-run:abc), created 2026-09-20T21:42:41+00:00, which is no longer "
        "running: the lock is stale. Check that no container was left behind (`docker ps -a --filter "
        "name=llmbench-`), then delete the file")
    path.write_text(json.dumps({"pid": os.getpid()}))
    assert describe_lock(path) == (
        f"{path} is held by pid {os.getpid()}, which is still running (a reused pid can belong to an unrelated "
        "process); one GPU workload runs at a time - wait for it, or stop that run first")
    # A server pid on a lock that is not a native run's is not trusted to change the advice.
    path.write_text(json.dumps({"pid": gone, "owner": "container-run:abc", "server_pid": os.getpid()}))
    assert "docker ps" in describe_lock(path) and "llama-server" not in describe_lock(path)


def test_a_native_lock_points_at_its_llama_server_never_at_docker(tmp_path):
    path, server = tmp_path / "gpu.lock", finished_pid()

    def described(**holder) -> str:
        path.write_text(json.dumps({"pid": finished_pid(), "created": "then", **holder}))
        text = describe_lock(path)
        assert "docker" not in text and "stale" in text and "no longer running" in text
        return text
    # The holder died and so did its server: say so, with the command that confirms it.
    text = described(owner="native-run:abc", runtime="metal-native", server_pid=server)
    assert f"llama-server, pid {server}, is gone too" in text and f"`ps -p {server}`" in text
    # The holder died but a process with the server's pid exists (this test process stands in for a leftover):
    # never "stop it" blindly, since a reused pid can belong to anything.
    text = described(runtime="metal-native", server_pid=os.getpid())
    assert f"llama-server pid {os.getpid()} still exists" in text and "reused pid" in text
    assert f"`ps -p {os.getpid()}`" in text and "stop it only if it is that llama-server" in text
    # The native runner's lease names its owner before any server pid is known; nonsense pids are never probed.
    assert "`pgrep -fl llama-server`" in described(owner="native-run:abc")
    for bogus in (True, "12", -1, 0, 2.5, None):
        assert "`pgrep -fl llama-server`" in described(runtime="metal-native", server_pid=bogus)
    # A live native holder is still the one GPU workload; the text also names its server.
    path.write_text(json.dumps({"pid": os.getpid(), "runtime": "metal-native", "server_pid": server}))
    text = describe_lock(path)
    assert "still running" in text and "wait for it" in text and f"its llama-server is pid {server}" in text
    # The lease refusal carries the same advice, and nothing removes the lock.
    path.write_text(json.dumps({"pid": finished_pid(), "runtime": "metal-native", "server_pid": server}))
    with pytest.raises(LeaseHeld, match=f"stale.*llama-server, pid {server}"):
        GpuLease(path).acquire()
    assert path.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_describing_a_lock_reads_a_bounded_prefix_and_never_blocks(tmp_path):
    import threading
    # `doctor` describes whatever sits at the shared lease path: a FIFO must not hang it ...
    fifo = tmp_path / "gpu.lock"
    os.mkfifo(fifo)
    described = []
    worker = threading.Thread(target=lambda: described.append(describe_lock(fifo)), daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert described == [f"{fifo} exists but does not name its holder; confirm no llmbench run is active before "
                         "deleting it"]
    assert fifo.exists()
    # ... and a valid holder followed by megabytes of padding is read only as far as the bound, as before.
    padded = tmp_path / "padded.lock"
    padded.write_text(json.dumps({"pid": os.getpid()}) + " " * (8 << 20), encoding="utf-8")
    assert "still running" in describe_lock(padded)
    padded.write_text(" " * 4096 + json.dumps({"pid": os.getpid()}), encoding="utf-8")
    assert "does not name its holder" in describe_lock(padded)
