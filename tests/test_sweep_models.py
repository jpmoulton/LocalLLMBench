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
