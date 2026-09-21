"""Offline acceptance of orchestration; no Docker, model loading, or real subprocess launches."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from llmbench.containers.config import ContainerRunResult, read_run_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepared_sweep", ROOT / "scripts/run_prepared_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Individual path length behavior already has dedicated runner tests; pytest names make paths long.
    monkeypatch.setattr(sweep, "check_path_lengths", lambda path: None)
    monkeypatch.setattr(sweep, "GLOBAL_LOCK_DIR", tmp_path / "global-lock")
    source = json.loads((ROOT / "examples/candidate.json").read_text())
    source["registry_digest"] = None
    source["evaluator"]["mode"] = "host-process"
    source["evaluator"]["image"] = None
    write(tmp_path / "config.json", source)
    inference = source["inference_image"]
    bundle = {"schema_version": 1, "prepared_utc": "test", "inference": inference,
              "evaluator": {"role": "evaluator", "reference": "sha256:" + "a" * 64,
                            "image_id": "sha256:" + "a" * 64,
                            "entrypoint": ["python", "-m", "llmbench.container_eval"]},
              "worker": source.get("worker_image"), "help_sha256": inference["help_sha256"],
              "registry_digest": "b" * 64}
    write(tmp_path / "bundle.json", bundle)
    plan = {"schema_version": 1, "plan_id": "test-sweep", "image_bundle": "bundle.json",
            "entries": [{"id": f"e{number}", "config": "config.json", "output": f"out{number}",
                         "budget_seconds": 1200, "phase": "baseline", "model": "q4"}
                        for number in (1, 2)]}
    path = tmp_path / "plan.json"
    write(path, plan)
    return path, plan


def result_for(entry, *, state="completed", cleanup=True, synthetic=False, fingerprint=None):
    return ContainerRunResult(
        session_id="session", attempt_id="attempt", config_fingerprint=fingerprint or entry["config_fingerprint"],
        project_name="llmbench-attempt", state=state, synthetic=synthetic,
        started_utc="start", finished_utc="finish", elapsed_seconds=1, budget_charged_seconds=1,
        cleanup={"verified": cleanup}, abort_campaign=state == "cleanup-uncertain",
        failure_reasons=() if state == "completed" else ("deliberate-failure",))


class FakeProcess:
    def __init__(self, argv, record, *, outcome="completed", interrupt=False, timeout=False, **kwargs):
        self.pid = 12345
        self.record, self.outcome = record, outcome
        self.interrupt, self.timeout, self.waits, self.signals = interrupt, timeout, [], []
        self.done = False
        self.output = Path(argv[argv.index("--output") + 1])
        config = read_run_config(argv[argv.index("--config") + 1])
        self.entry = {"config_fingerprint": config.fingerprint()}
        record.append(self)
        self.argv, self.kwargs = argv, kwargs

    def wait(self, timeout):
        self.waits.append(timeout)
        if len(self.waits) == 1 and self.interrupt:
            raise KeyboardInterrupt
        if self.timeout:
            raise subprocess.TimeoutExpired("fake", timeout)
        self.output.mkdir()
        if self.outcome != "missing":
            result = result_for(self.entry, state=self.outcome, cleanup=self.outcome != "cleanup-uncertain")
            write(self.output / "result.json", result.model_dump(mode="json"))
        self.done = True
        return 0 if self.outcome == "completed" else 3

    def poll(self):
        return 0 if self.done else None

    def send_signal(self, value):
        self.signals.append(value)


def fake_factory(record, outcomes=None, **settings):
    outcomes = iter(outcomes or ["completed", "completed"])
    return lambda argv, **kwargs: FakeProcess(argv, record, outcome=next(outcomes), **settings, **kwargs)


def run(prepared, **kwargs):
    return sweep.run_plan(sweep.read_plan(prepared[0]), process_guard=lambda: None, **kwargs)


def state(prepared):
    return json.loads((prepared[0].parent / ".sweep-test-sweep/state.json").read_text())


def test_default_check_is_read_only(prepared, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "run_plan", lambda *a, **k: pytest.fail("must not launch"))
    assert sweep.main(["--plan", str(prepared[0])]) == 0
    assert not (prepared[0].parent / ".sweep-test-sweep").exists()
    assert json.loads(capsys.readouterr().out)["entries"] == 2


def test_paths_resolve_against_plan_not_cwd(prepared, monkeypatch):
    monkeypatch.chdir(ROOT / "scripts")
    plan = sweep.read_plan(prepared[0])
    assert plan["entries"][0]["config"] == str(prepared[0].parent / "config.json")
    assert plan["entries"][0]["output"] == str(prepared[0].parent / "out1")


@pytest.mark.parametrize("path", ["../outside", "C:\\outside", "C:outside", "\\outside", "out:ads",
                                  "CON", "out.", "out ", ".sweep-test-sweep/child"])
def test_reject_unsafe_output(prepared, path):
    prepared[1]["entries"][0]["output"] = path
    write(*prepared)
    with pytest.raises(ValueError):
        sweep.read_plan(prepared[0])


@pytest.mark.parametrize("field,value", [("id", "e2"), ("output", "out2"), ("output", "out2/child"),
                                         ("budget_seconds", True), ("budget_seconds", -1),
                                         ("budget_seconds", float("inf")), ("model", "")])
def test_invalid_entry_rejected(prepared, field, value):
    prepared[1]["entries"][0][field] = value
    write(*prepared)
    with pytest.raises(ValueError):
        sweep.read_plan(prepared[0])


def test_config_validation_happens_before_any_launch(prepared):
    prepared[1]["entries"][1]["config"] = "bad.json"
    write(prepared[0].parent / "bad.json", {"unsupported": True})
    write(*prepared)
    with pytest.raises(ValueError):
        sweep.read_plan(prepared[0])
    assert not (prepared[0].parent / ".sweep-test-sweep").exists()


def test_unknown_plan_fields_cannot_inject_commands(prepared):
    prepared[1]["command"] = "anything"
    write(*prepared)
    with pytest.raises(ValueError):
        sweep.read_plan(prepared[0])


def test_image_bundle_mismatch_rejected(prepared):
    path = prepared[0].parent / "bundle.json"
    raw = json.loads(path.read_text())
    raw["inference"]["image_id"] = "sha256:" + "c" * 64
    write(path, raw)
    with pytest.raises(ValueError, match="bundle mismatch"):
        sweep.read_plan(prepared[0])


def test_serial_launch_records_exact_pid_and_logs(prepared):
    children = []
    assert run(prepared, popen=fake_factory(children)) == 0
    assert len(children) == 2
    saved = state(prepared)
    assert [x["status"] for x in saved["entries"].values()] == ["completed", "completed"]
    for child in children:
        assert child.kwargs["cwd"] == str(ROOT)
        assert child.waits == [1380]
        assert "--image-bundle" in child.argv
    assert saved["entries"]["e1"]["child_pid"] == 12345
    assert Path(saved["entries"]["e1"]["stdout_file"]).exists()
    journal = (prepared[0].parent / ".sweep-test-sweep/journal.jsonl").read_text().splitlines()
    assert len(journal) == saved["event_count"]


def test_cleanup_verified_failure_does_not_stop_remaining_plan(prepared):
    children = []
    assert run(prepared, popen=fake_factory(children, ["failed", "completed"])) == 3
    assert len(children) == 2
    assert state(prepared)["entries"]["e1"]["status"] == "failed"
    assert run(prepared, resume=True, popen=lambda *a, **k: pytest.fail("never rerun terminal")) == 3


@pytest.mark.parametrize("outcome", ["missing", "cleanup-uncertain"])
def test_missing_or_uncertain_terminal_stops_and_blocks_resume(prepared, outcome):
    children = []
    assert run(prepared, popen=fake_factory(children, [outcome])) == 4
    assert len(children) == 1
    assert state(prepared)["entries"]["e2"]["status"] == "pending"
    with pytest.raises(ValueError, match="interrupted/blocked"):
        run(prepared, resume=True, popen=lambda *a, **k: pytest.fail("must not launch"))


def test_resume_completed_entries_does_not_relaunch(prepared):
    assert run(prepared, popen=fake_factory([])) == 0
    assert run(prepared, resume=True, popen=lambda *a, **k: pytest.fail("must skip")) == 0
    with pytest.raises(ValueError, match="State exists"):
        run(prepared)


def test_changed_terminal_result_blocks_resume(prepared):
    assert run(prepared, popen=fake_factory([])) == 0
    result = prepared[0].parent / "out1/result.json"
    result.write_text(result.read_text() + "\n")
    with pytest.raises(ValueError, match="Terminal evidence changed"):
        run(prepared, resume=True)


def test_changed_config_blocks_resume(prepared):
    assert run(prepared, popen=fake_factory([])) == 0
    config = prepared[0].parent / "config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(ValueError, match="runtime changed"):
        run(prepared, resume=True)


def test_existing_output_never_overwritten(prepared):
    output = prepared[0].parent / "out2"
    output.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        run(prepared, popen=lambda *a, **k: pytest.fail("validate all outputs first"))


def test_interrupted_child_gets_cleanup_signal_and_resume_continues_pending(prepared):
    children = []
    assert run(prepared, popen=fake_factory(children, ["cancelled"], interrupt=True)) == 130
    assert len(children) == 1 and len(children[0].signals) == 1
    assert children[0].waits == [1380, 180]
    assert state(prepared)["entries"]["e1"]["status"] == "failed"
    assert run(prepared, resume=True, popen=fake_factory([])) == 3


def test_unresponsive_child_retains_pid_and_stops_without_kill(prepared):
    children = []
    assert run(prepared, popen=fake_factory(children, ["completed"], timeout=True)) == 4
    assert len(children) == 1 and len(children[0].signals) == 1
    assert children[0].waits == [1380, 180]
    assert state(prepared)["entries"]["e1"]["child_stopped"] is False
    assert state(prepared)["entries"]["e1"]["child_pid"] == 12345


def test_journal_state_disagreement_blocks_resume(prepared):
    assert run(prepared, popen=fake_factory([])) == 0
    journal = prepared[0].parent / ".sweep-test-sweep/journal.jsonl"
    with journal.open("a") as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match="Journal/state disagree"):
        run(prepared, resume=True)


def test_concurrent_supervisor_refused(prepared):
    with sweep.supervisor_lock(sweep.GLOBAL_LOCK_DIR):
        with pytest.raises(ValueError, match="process lock"):
            run(prepared, popen=lambda *a, **k: pytest.fail("must not launch"))


@pytest.mark.parametrize("name", ["Game.exe", "GAME.EXE", "game.exe"])
def test_process_guard_blocks_on_windows_without_killing(name):
    def command(argv, **kwargs):
        assert argv == ["tasklist.exe", "/FO", "CSV", "/NH"]
        assert kwargs["timeout"] == 15
        return SimpleNamespace(returncode=0, stdout=f'"{name}","123","Console","1","1 K"\n')
    with pytest.raises(ValueError, match="Close these before GPU testing: game.exe"):
        sweep.guard_processes(["Game.exe"], platform="nt", runner=command)
    sweep.guard_processes(["other.exe"], platform="nt", runner=command)  # something else running is fine


def test_process_guard_blocks_on_posix_and_is_a_no_op_when_nothing_is_named():
    def command(argv, **kwargs):
        assert argv == ["ps", "-A", "-o", "comm="]
        return SimpleNamespace(returncode=0, stdout="systemd\n/usr/bin/ollama\nbash\n")
    with pytest.raises(ValueError, match="Close these before GPU testing: ollama"):
        sweep.guard_processes(["Ollama"], platform="posix", runner=command)
    sweep.guard_processes(["steam"], platform="posix", runner=command)
    sweep.guard_processes((), runner=lambda *a, **k: pytest.fail("nothing named: the process list is never read"))


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_process_query_failure_is_closed(platform):
    with pytest.raises(ValueError, match="Cannot verify"):
        sweep.guard_processes(["game.exe"], platform=platform, runner=lambda *a, **k: SimpleNamespace(returncode=1))


def test_wrong_fingerprint_or_synthetic_terminal_rejected(prepared):
    entry = sweep.read_plan(prepared[0])["entries"][0]
    output = Path(entry["output"])
    output.mkdir()
    for change in ({"fingerprint": "0" * 64}, {"synthetic": True}):
        write(output / "result.json", result_for(entry, **change).model_dump(mode="json"))
        with pytest.raises(ValueError, match="synthetic or belongs"):
            sweep.terminal(entry, 0)

def test_low_disk_pauses_pending_then_resumes(prepared):
    def low_disk(plan):
        sweep.guard_disk(plan, usage=lambda path: SimpleNamespace(free=11 * 1024 ** 3))
    assert run(prepared, popen=lambda *a, **k: pytest.fail("no launch"), disk_guard=low_disk) == 2
    assert all(entry["status"] == "pending" for entry in state(prepared)["entries"].values())
    assert run(prepared, resume=True, popen=fake_factory([])) == 0


def test_process_guard_pause_leaves_pending(prepared):
    plan = sweep.read_plan(prepared[0])
    def busy():
        raise ValueError("a blocking process is running")
    assert sweep.run_plan(plan, popen=lambda *a, **k: pytest.fail("no launch"), process_guard=busy) == 2
    assert state(prepared)["entries"]["e1"]["status"] == "pending"


def test_configurable_disk_floor(prepared):
    prepared[1]["min_free_disk_gib"] = 35
    write(*prepared)
    plan = sweep.read_plan(prepared[0])
    with pytest.raises(ValueError, match="at least 35"):
        sweep.guard_disk(plan, usage=lambda path: SimpleNamespace(free=34 * 1024 ** 3))
    sweep.guard_disk(plan, usage=lambda path: SimpleNamespace(free=35 * 1024 ** 3))


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), "12"])
def test_bad_disk_floor_refused(prepared, value):
    prepared[1]["min_free_disk_gib"] = value
    write(*prepared)
    with pytest.raises(ValueError, match="min_free_disk"):
        sweep.read_plan(prepared[0])


def test_windows_launcher_ctrl_break_uses_cleanup_interrupt(monkeypatch):
    import runpy
    import signal
    import llmbench.containers.cli
    registrations = []
    monkeypatch.setattr(signal, "SIGBREAK", 21, raising=False)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: registrations.append((sig, handler)))
    monkeypatch.setattr(llmbench.containers.cli, "main", lambda: 0)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "run.py"), run_name="__main__")
    assert exc.value.code == 0
    assert len(registrations) == 1 and registrations[0][0] == 21
    with pytest.raises(KeyboardInterrupt):
        registrations[0][1](21, None)


def test_prepared_source_snapshot_checked_and_fingerprinted(prepared):
    prepared[1]["source_snapshot"] = "source.json"
    write(prepared[0].parent / "source.json", {"files": {"run.py": sweep.digest(ROOT / "run.py")}})
    write(*prepared)
    assert sweep.read_plan(prepared[0])["identity"]["source_snapshot_sha256"]
    write(prepared[0].parent / "source.json", {"files": {"run.py": "0" * 64}})
    with pytest.raises(ValueError, match="snapshot mismatch"):
        sweep.read_plan(prepared[0])


def test_snapshot_cannot_read_outside_repo(prepared):
    prepared[1]["source_snapshot"] = "source.json"
    write(prepared[0].parent / "source.json", {"files": {"../secret": "0" * 64}})
    write(*prepared)
    with pytest.raises(ValueError, match="relative path"):
        sweep.read_plan(prepared[0])


def test_interrupt_after_spawn_before_started_checkpoint_still_requests_cleanup(prepared, monkeypatch):
    original = sweep.Ledger.record
    def interrupt_checkpoint(ledger, event, *args, **kwargs):
        if event == "started":
            raise KeyboardInterrupt
        return original(ledger, event, *args, **kwargs)
    monkeypatch.setattr(sweep.Ledger, "record", interrupt_checkpoint)
    children = []
    assert run(prepared, popen=fake_factory(children, ["cancelled"])) == 130
    assert children[0].signals
    assert state(prepared)["entries"]["e1"]["status"] == "blocked"
    assert state(prepared)["entries"]["e1"]["child_pid"] == children[0].pid
    assert state(prepared)["entries"]["e1"]["child_stopped"] is True


@pytest.mark.parametrize("stdout", ["", "query unavailable", '"Game.exe","nonnumeric","x","1","1 K"'])
def test_malformed_process_inventory_refused(stdout):
    with pytest.raises(ValueError, match="Cannot parse"):
        sweep.guard_processes(["game.exe"], platform="nt",
                              runner=lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout))


def test_result_coercion_is_not_accepted_as_terminal_evidence(prepared):
    entry = sweep.read_plan(prepared[0])["entries"][0]
    path = Path(entry["output"])
    path.mkdir()
    result = result_for(entry).model_dump(mode="json")
    result["cleanup"]["verified"] = "true"
    write(path / "result.json", result)
    with pytest.raises(ValueError):
        sweep.terminal(entry, 0)
