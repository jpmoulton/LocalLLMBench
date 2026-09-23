"""The owned native server process: start in its own session, bounded log drain, stop by process group, proof
of absence. These tests start a real, harmless Python child standing in for llama-server (no model, no GPU, no
network), because process groups and signals are exactly what fakes get wrong."""

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from llmbench.containers.artifacts import RunArtifacts
from llmbench.containers.native import (NativeServerProcess, free_loopback_port, port_accepts,
                                        scrubbed_environment)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups and POSIX signals")


class Sink:
    def __init__(self, path):
        self.handle = open(path, "wb")
        self.closed = False

    def write(self, data):
        return self.handle.write(data)

    def flush(self):
        self.handle.flush()

    def close(self):
        self.closed = True
        self.handle.close()


def child(tmp_path, body: str, *, marker=("--alias", "llmbench-test", "--port", "1"), cap=1 << 20):
    script = tmp_path / "fake_server.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    log = tmp_path / "server.log"
    return NativeServerProcess(sys.executable, [str(script), *marker], cwd=tmp_path, env=scrubbed_environment(),
                               log_sink=Sink(log), log_path=log, log_cap=cap)


def wait_for(predicate, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_a_cooperative_server_stops_on_sigterm_and_leaves_nothing(tmp_path):
    server = child(tmp_path, """
        import signal, sys, time
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        print("llama_server: listening on http://127.0.0.1:1", flush=True)
        while True:
            time.sleep(0.1)
    """)
    server.start()
    assert server.state() == ("running", "", "false")
    assert wait_for(lambda: b"listening on" in server.read_log(4096).stdout)
    assert os.getpgid(server.pid) == server.pid != os.getpgid(0)  # its own session and process group
    server.stop(5, time.monotonic() + 30)
    assert server.state()[0] == "exited" and server.describe_exit() == "returncode=0"
    assert server.signals_sent == ["SIGTERM"] and not server.unexpected_kill()
    assert server.remaining(("--alias", "llmbench-test", "--port", "1")) == []


def test_a_server_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    server = child(tmp_path, """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        while True:
            time.sleep(0.1)
    """)
    server.start()
    assert wait_for(lambda: b"ready" in server.read_log(64).stdout)
    began = time.monotonic()
    server.stop(0.5, time.monotonic() + 30)
    assert time.monotonic() - began < 10
    assert server.describe_exit() == "signal=SIGKILL" and server.signals_sent[:2] == ["SIGTERM", "SIGKILL"]
    assert not server.unexpected_kill()  # the harness sent that SIGKILL itself
    assert server.remaining(("--alias", "llmbench-test", "--port", "1")) == []


def test_a_grandchild_left_in_the_process_group_is_found_and_removed(tmp_path):
    server = child(tmp_path, """
        import signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
        print("spawned", flush=True)
        while True:
            time.sleep(0.1)
    """)
    server.start()
    assert wait_for(lambda: b"spawned" in server.read_log(64).stdout)
    server.stop(5, time.monotonic() + 30)
    assert wait_for(lambda: server.remaining(("--alias", "llmbench-test", "--port", "1")) == [])
    assert "SIGKILL(group)" in server.signals_sent


def test_a_death_by_sigkill_the_harness_never_sent_is_reported_as_unexpected(tmp_path):
    server = child(tmp_path, """
        import os, signal
        print("dying", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    """)
    server.start()
    assert wait_for(lambda: server.state()[0] == "exited")
    server.stop(1, time.monotonic() + 10)
    assert server.describe_exit() == "signal=SIGKILL" and server.unexpected_kill()
    assert server.signals_sent == []


def test_the_log_is_drained_past_its_cap_without_stalling_the_server(tmp_path):
    server = child(tmp_path, """
        import sys
        for _ in range(2000):
            sys.stdout.write("x" * 1023 + "\\n")
        sys.stdout.flush()
    """, cap=4096)
    server.start()
    assert wait_for(lambda: server.state()[0] == "exited")  # a 2 MB burst never blocks on a full pipe
    server.stop(1, time.monotonic() + 10)
    assert server.log_bytes == 4096 and server.dropped_bytes == 2000 * 1024 - 4096
    assert server.read_log(1 << 20).status == "output-limit"
    assert (tmp_path / "server.log").stat().st_size == 4096 and server.log_sink.closed


def test_a_log_the_artifact_budget_refuses_is_still_drained_and_the_refusal_recorded(tmp_path):
    # The real sink: a run's artifact store with a 1 MiB budget. Once it refuses a write the drain keeps reading
    # (a full pipe would stall llama-server mid-benchmark), keeps nothing more, and says why.
    artifacts = RunArtifacts(tmp_path / "run", 1_048_576)
    server = NativeServerProcess(sys.executable, ["-c", textwrap.dedent("""
        import sys
        for _ in range(2000):
            sys.stdout.write("x" * 1023 + "\\n")
        sys.stdout.flush()
    """)], cwd=tmp_path, env=scrubbed_environment(), log_sink=artifacts.open_external("logs/server.log", "xb"),
                                 log_path=tmp_path / "run" / "logs" / "server.log", log_cap=16 << 20)
    server.start()
    try:
        assert wait_for(lambda: server.state()[0] == "exited")  # 2 MB against a 1 MiB budget never blocks
    finally:
        server.stop(1, time.monotonic() + 10)
    assert server.describe_exit() == "returncode=0"
    assert server.log_error and "budget" in server.log_error
    assert 0 < server.log_bytes <= 1_048_576 and server.log_bytes + server.dropped_bytes == 2000 * 1024
    assert server.log_sink.closed


def test_an_interrupted_stop_still_kills_the_owned_group(tmp_path):
    # A second Ctrl-C landing in the wait after SIGTERM must not leave a server that ignores SIGTERM running: it
    # is in its own session, so nothing else would ever stop it.
    server = child(tmp_path, """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        while True:
            time.sleep(0.1)
    """)
    server.start()
    try:
        assert wait_for(lambda: b"ready" in server.read_log(64).stdout)
        real_wait = server.process.wait

        def interrupted_wait(timeout=None):
            server.process.wait = real_wait
            raise KeyboardInterrupt()
        server.process.wait = interrupted_wait
        with pytest.raises(KeyboardInterrupt):
            server.stop(30, time.monotonic() + 60)
        assert server.signals_sent[:2] == ["SIGTERM", "SIGKILL"]
        assert wait_for(lambda: server.state()[0] == "exited", 5)
        assert server.describe_exit() == "signal=SIGKILL" and not server.unexpected_kill()
        assert wait_for(lambda: server.remaining(("--alias", "llmbench-test", "--port", "1")) == [], 5)
    finally:
        if server.process.poll() is None:
            os.killpg(server.pgid, signal.SIGKILL)
            server.process.wait(timeout=10)


def test_signals_are_never_sent_to_a_reaped_child(tmp_path, monkeypatch):
    server = child(tmp_path, "print('bye')\n")
    server.start()
    assert wait_for(lambda: server.state()[0] == "exited")
    sent = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append(sig) or (_ for _ in ()).throw(ProcessLookupError()))
    server.terminate()
    assert sent == []  # a reaped pid may already belong to someone else


def test_scrubbed_environment_removes_every_setting_that_could_override_the_argv():
    env = scrubbed_environment({"PATH": "/bin", "HOME": "/h", "LLAMA_ARG_CTX_SIZE": "99", "GGML_METAL_NDEBUG": "1",
                                "HF_TOKEN": "t", "https_proxy": "http://p", "NO_PROXY": "*",
                                "DYLD_INSERT_LIBRARIES": "/evil.dylib", "LLAMA_CACHE": "/c"})
    assert env == {"PATH": "/bin", "HOME": "/h"}


def test_free_loopback_port_is_bindable_and_port_accepts_reads_the_truth():
    port = free_loopback_port()
    assert 1024 <= port <= 65535 and not port_accepts(port)
    listener = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import socket, time
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", {port})); s.listen(1); print("up", flush=True); time.sleep(30)
    """)], stdout=subprocess.PIPE)
    try:
        assert listener.stdout.readline().strip() == b"up"
        assert port_accepts(port)
    finally:
        listener.send_signal(signal.SIGKILL)
        listener.wait()
    assert wait_for(lambda: not port_accepts(port))
