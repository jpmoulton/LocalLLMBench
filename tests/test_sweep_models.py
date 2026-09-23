"""The multi-model sweep driver: what it selects and how it names things. It never starts anything here."""

import importlib.util
import json
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
    if leftover == "lease":
        lease.write_text("{}", encoding="utf-8")
    launched, asked = [], []
    monkeypatch.setattr(sweep_models, "run_session", lambda argv, log: launched.append(argv) or 0)
    monkeypatch.setattr(sweep_models, "lease_path", lambda: lease)
    monkeypatch.setattr(sweep_models, "running_servers", lambda path, processes=None: asked.append(path) or (
        ["pid:4242"] if leftover == "server" else []))
    monkeypatch.setattr(sweep_models, "leaked_containers", lambda: ["llmbench-" + "0" * 32]
                        if leftover == "container" else [])
    assert sweep_models.main(["--models", *map(str, models), "--output", str(tmp_path / "sweep"), "--runtime",
                              "metal-native", "--native-bundle", str(bundle)]) == 4
    assert len(launched) == 1 and asked == [str(executable)]  # the bundle's own executable is what is looked for
    err = capsys.readouterr().err
    assert "native runtime resources were left behind" in err
    assert {"lease": "GPU lease", "server": "pid:4242", "container": "llmbench-"}[leftover] in err


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
