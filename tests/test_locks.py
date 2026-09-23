"""A held lock must say who holds it and whether that process still exists - and nothing may ever break one."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from llmbench.config import canonical_json
from llmbench.containers.lease import GpuLease, LeaseHeld
from llmbench.locks import describe_lock, process_alive
from llmbench.store import Store


SRC = Path(__file__).resolve().parents[1] / "src"


def finished_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def killed_holder(script: str, *args) -> None:
    """Run `script` in a child that takes a lock through the real writer and then dies without any cleanup
    (`os._exit`, as a SIGKILL, jetsam or a closed terminal would leave it): the lock file stays behind, exactly as
    that writer wrote it, naming a pid that no longer exists."""
    subprocess.run([sys.executable, "-c", textwrap.dedent(script), *map(str, args)], check=True, timeout=60,
                   env={**os.environ, "PYTHONPATH": str(SRC)})


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


def test_a_campaign_lock_left_by_a_killed_session_names_both_runtimes_leftovers(tmp_path):
    # `tune`, `resume` and the sweeps hold the GPU through the campaign lock, which records only its pid. A killed
    # metal-native session leaves a llama-server in its own session; a killed NVIDIA one may leave a container.
    # With no runtime recorded neither is assumed: the advice covers both, and nothing removes the lock.
    lock = tmp_path / "gpu.lock"
    killed_holder("""
        import os, sys
        from pathlib import Path
        from llmbench.store import Store
        with Store(Path(sys.argv[1])) as store:
            with store.campaign_lock(Path(sys.argv[2])):
                os._exit(0)
    """, tmp_path / "store", lock)
    holder = json.loads(lock.read_text(encoding="utf-8"))
    assert set(holder) == {"pid", "created"}
    text = describe_lock(lock)
    assert "no longer running" in text and "stale" in text and "names no runtime" in text
    assert "`docker ps -a --filter name=llmbench-`" in text and "`pgrep -fl llama-server`" in text
    assert lock.exists()


def test_a_native_lease_left_by_a_killed_run_names_its_llama_server(tmp_path):
    lock, server = tmp_path / "gpu.lock", finished_pid()
    killed_holder("""
        import os, sys
        from pathlib import Path
        from llmbench.containers.lease import GpuLease
        lease = GpuLease(Path(sys.argv[1]), owner="native-run:abc").acquire()
        lease.annotate(runtime="metal-native", server_pid=int(sys.argv[2]))
        os._exit(0)
    """, lock, server)
    holder = json.loads(lock.read_text(encoding="utf-8"))
    assert holder["owner"] == "native-run:abc" and holder["runtime"] == "metal-native"
    assert holder["server_pid"] == server
    text = describe_lock(lock)
    assert f"llama-server, pid {server}, is gone too" in text and "docker" not in text
    with pytest.raises(LeaseHeld, match=f"stale.*llama-server, pid {server}"):
        GpuLease(lock).acquire()
    assert lock.exists()


def test_a_container_leases_advice_is_unchanged(tmp_path):
    lock = tmp_path / "gpu.lock"
    for owner in ("container-run", "container-run:abc"):
        killed_holder("""
            import os, sys
            from pathlib import Path
            from llmbench.containers.lease import GpuLease
            GpuLease(Path(sys.argv[1]), owner=sys.argv[2]).acquire()
            os._exit(0)
        """, lock, owner)
        text = describe_lock(lock)
        assert text.endswith("Check that no container was left behind (`docker ps -a --filter name=llmbench-`), "
                             "then delete the file") and "llama-server" not in text
        lock.unlink()


def test_annotating_a_lease_only_ever_writes_the_file_it_created(tmp_path):
    path = tmp_path / "gpu.lock"
    lease = GpuLease(path, owner="native-run:abc").acquire()
    written = json.loads(path.read_text(encoding="utf-8"))
    lease.annotate(runtime="metal-native", server_pid=4242)
    assert json.loads(path.read_text(encoding="utf-8")) == {**written, "runtime": "metal-native", "server_pid": 4242}
    for field in ("pid", "owner", "created"):  # who holds the lease is never rewritten
        with pytest.raises(ValueError, match=field):
            lease.annotate(**{field: 1})
    # Someone deleted the lease and another run took the GPU: that run's lock is left exactly as it is.
    path.unlink()
    other = canonical_json({"pid": os.getpid(), "created": "now", "owner": "container-run:def"}).encode("utf-8")
    path.write_bytes(other)
    with pytest.raises(LeaseHeld, match="no longer the lease this run created"):
        lease.annotate(server_pid=1)
    assert path.read_bytes() == other
    # A byte-identical copy swapped in over the lease is not the file this lease created either.
    swapped = GpuLease(tmp_path / "swapped.lock", owner="native-run:ghi").acquire()
    copy = tmp_path / "copy.lock"
    copy.write_bytes(swapped.path.read_bytes())
    os.replace(copy, swapped.path)
    with pytest.raises(LeaseHeld, match="no longer the lease this run created"):
        swapped.annotate(server_pid=1)
    assert "server_pid" not in json.loads(swapped.path.read_text(encoding="utf-8"))
    # A reused inode: the device and inode this lease created, holding another run's record (and a longer copy of
    # this lease's own bytes is not what it wrote either).
    reused = GpuLease(tmp_path / "reused.lock", owner="native-run:jkl").acquire()
    own = reused.path.read_bytes()
    for foreign in (other, own + b" "):
        with open(reused.path, "r+b") as handle:  # rewritten in place: the inode stays the one the lease created
            handle.truncate(0)
            handle.write(foreign)
        with pytest.raises(LeaseHeld, match="no longer the lease this run created"):
            reused.annotate(server_pid=1)
        assert reused.path.read_bytes() == foreign
    unheld = GpuLease(tmp_path / "never.lock")
    with pytest.raises(LeaseHeld):
        unheld.annotate(server_pid=1)
    assert not (tmp_path / "never.lock").exists()  # annotating never creates a lock


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
