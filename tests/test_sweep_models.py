"""The multi-model sweep driver: what it selects and how it names things. It never starts a model here; one test
starts two tiny Python processes to show how a terminal interrupt reaches the session."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_models.py"
spec = importlib.util.spec_from_file_location("sweep_models", SCRIPT)
sweep_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep_models)


def test_slugs_are_safe_and_unique():
    taken = set()
    assert sweep_models.slug_for(Path("Qwen3.8-27B-Q4_K_M.gguf"), taken) == "qwen3-8-27b-q4-k-m"
    assert sweep_models.slug_for(Path("other/Qwen3.8-27B-Q4_K_M.gguf"), taken) == "qwen3-8-27b-q4-k-m-2"
    assert sweep_models.slug_for(Path("###.gguf"), taken) == "model"


def test_directory_discovery_skips_vision_projectors_and_missing_files_stop_the_sweep(tmp_path):
    for name in ("b.gguf", "a.GGUF", "mmproj-model-BF16.gguf", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    found = sweep_models.discover([], str(tmp_path))
    assert [item.name for item in found] == ["a.GGUF", "b.gguf"]
    with pytest.raises(SystemExit):
        sweep_models.discover([str(tmp_path / "absent.gguf")], None)
    with pytest.raises(SystemExit):
        sweep_models.discover([], None)


def test_each_model_gets_its_own_session_and_an_existing_run_is_never_overwritten(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    launched = []
    monkeypatch.setattr(sweep_models, "run_session", lambda argv, log: launched.append(argv) or 0)
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: [])
    output = tmp_path / "sweep"
    assert sweep_models.main(["--models", str(model), "--output", str(output), "--budget-seconds", "600"]) == 0
    assert len(launched) == 1 and launched[0][launched[0].index("--model") + 1] == str(model.resolve())
    assert json.loads((output / "index.json").read_text())["m"]["model"] == str(model.resolve())
    (output / "m" / "run").mkdir(parents=True)
    assert sweep_models.main(["--models", str(model), "--output", str(output)]) == 0 and len(launched) == 1


def test_a_leaked_container_stops_the_sequence(tmp_path, monkeypatch):
    models = [tmp_path / "a.gguf", tmp_path / "b.gguf"]
    for item in models:
        item.write_bytes(b"x")
    launched = []
    monkeypatch.setattr(sweep_models, "run_session", lambda argv, log: launched.append(argv) or 0)
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: ["llmbench-deadbeef-inference-1"])
    assert sweep_models.main(["--models", *map(str, models), "--output", str(tmp_path / "sweep")]) == 4
    assert len(launched) == 1  # the second model is never measured beside a leftover container


# ---- metal-native runtime -------------------------------------------------------------------------------------

def native_bundle_file(tmp_path, executable):
    from test_containers_session import make_native_bundle
    path = tmp_path / "native-bundle.json"
    path.write_text(make_native_bundle(executable=str(executable)).model_dump_json(), encoding="utf-8")
    return path


def test_native_flags_are_forwarded_with_an_absolute_bundle_and_nvidia_checks_are_not_used(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    bundle = native_bundle_file(tmp_path, tmp_path / "llama-b11011" / "llama-server")
    launched = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sweep_models, "run_session", lambda argv, log: launched.append(argv) or 0)
    monkeypatch.setattr(sweep_models, "lease_path", lambda: tmp_path / "no-lease.lock")
    monkeypatch.setattr(sweep_models, "running_servers", lambda executable, processes=None: [])
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: [])
    assert sweep_models.main(["--models", str(model), "--output", str(tmp_path / "sweep"), "--runtime",
                              "metal-native", "--native-bundle", "native-bundle.json", "--capabilities-dir",
                              "caps"]) == 0
    (argv,) = launched
    assert argv[argv.index("--runtime") + 1] == "metal-native" and "--image-bundle" not in argv
    assert argv[argv.index("--native-bundle") + 1] == str(bundle.resolve())  # sessions run with cwd=ROOT
    assert argv[argv.index("--capabilities-dir") + 1] == "caps"
    # An NVIDIA sweep never reads a lease file or the process table for a native server: its check is unchanged.
    monkeypatch.setattr(sweep_models, "native_leftovers", lambda executable: pytest.fail("NVIDIA check changed"))
    assert sweep_models.main(["--models", str(model), "--output", str(tmp_path / "cuda")]) == 0
    assert "--runtime" not in launched[-1] and "--native-bundle" not in launched[-1]


@pytest.mark.parametrize("leftover", ["lease", "server", "container"])
def test_a_native_leftover_stops_the_sequence_and_is_never_killed(tmp_path, monkeypatch, capsys, leftover):
    models = [tmp_path / "a.gguf", tmp_path / "b.gguf"]
    for item in models:
        item.write_bytes(b"x")
    executable = tmp_path / "llama-server"
    bundle = native_bundle_file(tmp_path, executable)
    lease = tmp_path / "llmbench-gpu-resource.lock"
    launched, asked = [], []

    def session(argv, log):  # the first model's session is what leaves the leftover behind
        launched.append(argv)
        if leftover == "lease":
            lease.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(sweep_models, "run_session", session)
    monkeypatch.setattr(sweep_models, "lease_path", lambda: lease)
    monkeypatch.setattr(sweep_models, "running_servers", lambda path, processes=None: asked.append(path) or (
        ["pid:4242"] if leftover == "server" and launched else []))
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: ["llmbench-" + "0" * 32]
                        if leftover == "container" and launched else [])
    assert sweep_models.main(["--models", *map(str, models), "--output", str(tmp_path / "sweep"), "--runtime",
                              "metal-native", "--native-bundle", str(bundle)]) == 4
    # Looked for before the first model and after it; the bundle's own executable is what is looked for.
    assert len(launched) == 1 and asked == [str(executable)] * 2
    err = capsys.readouterr().err
    assert "native runtime resources were left behind" in err
    assert {"lease": "GPU lease", "server": "pid:4242", "container": "llmbench-"}[leftover] in err


@pytest.mark.parametrize("leftover", [None, "lease", "server"])
def test_an_interrupted_native_sweep_still_checks_for_leftovers_before_it_exits(tmp_path, monkeypatch, capsys,
                                                                                 leftover):
    """What an interrupted session left behind is named on the way out, while the person who interrupted is still
    looking; nothing after it is launched."""
    models = [tmp_path / "a.gguf", tmp_path / "b.gguf"]
    for item in models:
        item.write_bytes(b"x")
    executable = tmp_path / "llama-server"
    bundle = native_bundle_file(tmp_path, executable)
    lease = tmp_path / "llmbench-gpu-resource.lock"
    launched, asked = [], []

    def interrupted(argv, log):  # the interrupted session is what leaves the leftover behind
        launched.append(argv)
        if leftover == "lease":
            lease.write_text("{}", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep_models, "run_session", interrupted)
    monkeypatch.setattr(sweep_models, "lease_path", lambda: lease)
    monkeypatch.setattr(sweep_models, "running_servers", lambda path, processes=None: asked.append(path) or (
        ["pid:4242"] if leftover == "server" and launched else []))
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: [])
    assert sweep_models.main(["--models", *map(str, models), "--output", str(tmp_path / "sweep"), "--runtime",
                              "metal-native", "--native-bundle", str(bundle)]) == 130
    # Looked for before the first model and after the interrupt; nothing after it was launched.
    assert len(launched) == 1 and asked == [str(executable)] * 2
    err = capsys.readouterr().err
    if leftover is None:
        assert "no native runtime resources were left behind" in err
    else:
        assert "native runtime resources are still present" in err
        assert {"lease": "GPU lease", "server": "pid:4242"}[leftover] in err


@pytest.mark.parametrize("leftover", ["lease", "server", "container"])
def test_a_native_rerun_never_starts_its_first_model_beside_a_leftover(tmp_path, monkeypatch, capsys, leftover):
    """A re-run skips the model an interrupted or killed run left and would start the next one at once. An idle
    orphaned llama-server that still holds its weights passes the session's lock check and native admission alike,
    so the sweep itself looks before its first model, and only looks: nothing is stopped or removed from here."""
    models = [tmp_path / "a.gguf", tmp_path / "b.gguf"]
    for item in models:
        item.write_bytes(b"x")
    executable = tmp_path / "llama-server"
    bundle = native_bundle_file(tmp_path, executable)
    output = tmp_path / "sweep"
    (output / "a" / "run").mkdir(parents=True)  # the model the earlier run was interrupted in
    lease = tmp_path / "llmbench-gpu-resource.lock"
    if leftover == "lease":
        lease.write_text("{}", encoding="utf-8")
    launched = []
    monkeypatch.setattr(sweep_models, "run_session", lambda argv, log: launched.append(argv) or 0)
    monkeypatch.setattr(sweep_models, "lease_path", lambda: lease)
    monkeypatch.setattr(sweep_models, "running_servers", lambda path, processes=None: (
        ["pid:4242"] if leftover == "server" else []))
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: ["llmbench-" + "0" * 32]
                        if leftover == "container" else [])
    assert sweep_models.main(["--models", *map(str, models), "--output", str(output), "--runtime", "metal-native",
                              "--native-bundle", str(bundle)]) == 4
    assert launched == [] and not (output / "b").exists()
    err = capsys.readouterr().err
    assert "stopping before b: native runtime resources are already present" in err
    assert {"lease": "GPU lease", "server": "pid:4242", "container": "llmbench-"}[leftover] in err
    assert lease.exists() == (leftover == "lease")  # read-only: the lease is reported, never removed


def test_an_interrupted_nvidia_sweep_exits_exactly_as_before(tmp_path, monkeypatch, capsys):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")

    def interrupted(argv, log):
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep_models, "run_session", interrupted)
    monkeypatch.setattr(sweep_models, "native_leftovers", lambda executable: pytest.fail("NVIDIA path changed"))
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: pytest.fail("NVIDIA path changed"))
    assert sweep_models.main(["--models", str(model), "--output", str(tmp_path / "sweep")]) == 130
    assert capsys.readouterr().err == ""


SESSION = textwrap.dedent("""
    import pathlib, signal, sys, time
    record, received = pathlib.Path(sys.argv[1]), []

    def interrupted(signum, frame):
        received.append(signum)
        record.write_text(str(len(received)))

    signal.signal(signal.SIGINT, interrupted)
    record.with_name("ready").write_text("")
    deadline = time.monotonic() + 20
    while not received and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(1.5)  # the session's cleanup: far longer than the 0.25 s Popen.wait grants before a re-send
""")

SWEEP = textwrap.dedent("""
    import importlib.util, pathlib, signal, sys
    # A terminal's foreground job raises KeyboardInterrupt on SIGINT. Python only installs that handler when SIGINT
    # is not ignored, and a test run started in the background (``pytest &``, nohup) passes on an ignored SIGINT.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    spec = importlib.util.spec_from_file_location("sweep_models", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.run_session([sys.executable, sys.argv[2], sys.argv[3]], pathlib.Path(sys.argv[4]))
    except KeyboardInterrupt:
        sys.exit(130)
""")


@pytest.mark.skipif(os.name == "nt", reason="POSIX terminal process groups")
def test_a_terminal_ctrl_c_reaches_the_session_exactly_once(tmp_path):
    """A terminal Ctrl-C signals its whole foreground process group. The session must get only the interrupt the
    sweep forwards: had it also received the terminal's, the forwarded one would land inside its cleanup."""
    (tmp_path / "session.py").write_text(SESSION, encoding="utf-8")
    (tmp_path / "sweep.py").write_text(SWEEP, encoding="utf-8")
    record = tmp_path / "interrupts"
    # The sweep leads a process group of its own, as a shell's foreground job does; os.killpg below is the Ctrl-C.
    sweep = subprocess.Popen([sys.executable, str(tmp_path / "sweep.py"), str(SCRIPT), str(tmp_path / "session.py"),
                              str(record), str(tmp_path / "tune.log")], start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready").exists():
            assert sweep.poll() is None and time.monotonic() < deadline, "the session never started"
            time.sleep(0.02)
        os.killpg(sweep.pid, signal.SIGINT)
        assert sweep.wait(timeout=30) == 130
    finally:
        if sweep.poll() is None:
            sweep.kill()
            sweep.wait()
    assert record.read_text() == "1"


def test_contradictory_runtime_flags_are_refused_before_any_model_starts(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    bundle = native_bundle_file(tmp_path, tmp_path / "llama-server")
    monkeypatch.setattr(sweep_models, "run_session", lambda argv, log: pytest.fail("nothing may start"))
    common = ["--models", str(model), "--output", str(tmp_path / "sweep")]
    (tmp_path / "not-a-bundle.json").write_text("{}", encoding="utf-8")
    for extra in (["--runtime", "metal-native"], ["--native-bundle", str(bundle)],
                  ["--runtime", "nvidia-container", "--native-bundle", str(bundle)],
                  ["--runtime", "metal-native", "--native-bundle", str(bundle), "--image-bundle", "b.json"],
                  ["--runtime", "metal-native", "--native-bundle", str(tmp_path / "not-a-bundle.json")],
                  ["--runtime", "metal-native", "--native-bundle", str(tmp_path / "absent.json")]):
        with pytest.raises(SystemExit) as refused:
            sweep_models.main(common + extra)
        assert refused.value.code == 2, extra
    assert not (tmp_path / "sweep").exists()


def test_running_servers_matches_the_resolved_executable_only_and_an_unreadable_table_is_not_clean(tmp_path):
    from types import SimpleNamespace
    real = tmp_path / "llama-b11011" / "llama-server"
    real.parent.mkdir()
    real.write_bytes(b"")
    link = tmp_path / "llama-server-link"
    link.symlink_to(real)
    processes = [SimpleNamespace(info={"pid": 11, "exe": str(link)}),  # the same file through a symlink
                 SimpleNamespace(info={"pid": 12, "exe": "/usr/bin/other"}),
                 SimpleNamespace(info={"pid": 13, "exe": None})]  # another user's process: unreadable exe
    assert sweep_models.running_servers(str(real), processes=processes) == ["pid:11"]
    assert sweep_models.running_servers(str(real), processes=[]) == []

    def unreadable():
        raise PermissionError("process table")
        yield
    (problem,) = sweep_models.running_servers(str(real), processes=unreadable())
    assert problem.startswith("process table unreadable (PermissionError")
    # The live table is read-only and this test process is not a llama-server.
    assert sweep_models.running_servers(str(real)) == []
