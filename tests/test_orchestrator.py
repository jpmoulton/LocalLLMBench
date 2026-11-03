"""CPU-only lifecycle regressions: failures, cancellation, and resource isolation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.config import resolve_config
from orchestrator.runner import execute
from orchestrator.state import save_json


class LifecycleBackend:
    def __init__(self, root, failure=None, cleanup_failure=False, interrupt_cleanup=False):
        self.root = root
        self.failure = failure
        self.cleanup_failure = cleanup_failure
        self.interrupt_cleanup = interrupt_cleanup
        self.active = {"unrelated": {"owner": "user"}}
        self.started = []
        self.current = None

    def ensure_network(self):
        pass

    def start(self, trial, plan, name, labels):
        if any("localllmbench.run" in labels for labels in self.active.values()):
            raise AssertionError("Trial containers overlapped")
        self.current = trial["id"]
        self.started.append(self.current)
        self.active[name] = labels
        if self.current == "0001" and self.failure == "start":
            raise RuntimeError("Container created, then startup failed")
        return {"container": name}

    def wait_ready(self, name, base_url, timeout):
        pass

    def probe(self, base_url, timeout):
        return {"response": {"choices": [{"message": {"content": "Hello world"}}]}}

    def run_benchmarks(self, command, log_path, tick):
        directory = log_path.parent / "benchmarks"
        if self.current == "0001" and self.failure == "interrupt":
            raise KeyboardInterrupt()
        incomplete = self.current == "0001" and self.failure == "benchmark"
        status = "incomplete" if incomplete else "completed"
        save_json(directory / "bigcodebench" / "summary.json", {
            "status": status, "pass_rate": None if incomplete else 0,
            "score_metrics": {"pass_rate": {"direction": "higher", "unit": "fraction"}},
        })
        save_json(directory / "summary.json", {"status": status})
        return 1 if incomplete else 0

    def logs(self, name, output):
        output.write_text("CPU lifecycle fixture", encoding="utf-8")

    def cleanup(self, labels):
        if self.cleanup_failure:
            raise RuntimeError("Daemon refused container removal")
        if self.interrupt_cleanup:
            signal.raise_signal(signal.SIGINT)
        self.active = {name: owned for name, owned in self.active.items() if owned != labels}


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = resolve_config({"benchmark_options": {"bigcodebench": {"dataset_path": str(self.root)}}})
        self.plan = {
            "schema_version": 1, "name": "lifecycle", "source": "fixture", "output_dir": str(self.root),
            "inference": {"bind_address": "127.0.0.1", "port": 18081},
            "benchmarks": config["benchmarks"], "benchmark_options": config["benchmark_options"],
            "generation": config["generation"], "execution": {**config["execution"], "startup_timeout": 2},
            "manifests": {"bigcodebench": {"selected_ids": ["BigCodeBench/1"]}},
            "trials": [{"id": f"{index:04d}", "model": str(self.root / "unused.gguf"), "group": "test",
                        "parameters": {"--ctx-size": index * 4096}, "server_args": []} for index in (1, 2)],
        }

    def execute(self, backend, **kwargs):
        with redirect_stdout(io.StringIO()):
            return execute(self.plan, backend=backend, **kwargs)

    def test_failure_preserves_partial_results_then_continues_without_overlap(self):
        backend = LifecycleBackend(self.root, failure="benchmark")
        output, run, report = self.execute(backend)
        self.assertEqual(run["status"], "incomplete")
        self.assertEqual([trial["status"] for trial in run["trials"]], ["incomplete", "completed"])
        self.assertEqual(backend.active, {"unrelated": {"owner": "user"}})
        self.assertEqual(report["benchmark_leaders"]["bigcodebench"]["pass_rate"]["trial_ids"], ["0002"])
        self.assertEqual(json.loads((output / "run.json").read_text())["counts"], run["counts"])
        self.assertEqual(run["counts"]["completed"], 1)  # Incorrect answers remain completed evaluations.

    def test_start_failure_after_container_creation_is_cleaned_before_next_trial(self):
        backend = LifecycleBackend(self.root, failure="start")
        _, run, _ = self.execute(backend)
        self.assertEqual([trial["status"] for trial in run["trials"]], ["incomplete", "completed"])
        self.assertEqual(run["trials"][0]["error"]["phase"], "load")
        self.assertEqual(backend.active, {"unrelated": {"owner": "user"}})

    def test_interrupt_cleans_active_trial_and_leaves_unscheduled_trial_pending(self):
        backend = LifecycleBackend(self.root, failure="interrupt")
        output, run, _ = self.execute(backend)
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual([trial["status"] for trial in run["trials"]], ["cancelled", "pending"])
        self.assertEqual(backend.started, ["0001"])
        self.assertEqual(backend.active, {"unrelated": {"owner": "user"}})
        self.assertEqual(json.loads((output / "report.json").read_text())["run"]["status"], "cancelled")

    def test_interrupt_during_cleanup_finishes_cleanup_before_stopping_schedule(self):
        backend = LifecycleBackend(self.root, interrupt_cleanup=True)
        _, run, _ = self.execute(backend)
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(backend.started, ["0001"])
        self.assertEqual(backend.active, {"unrelated": {"owner": "user"}})

    def test_cleanup_failure_stops_instead_of_launching_another_gpu_trial(self):
        backend = LifecycleBackend(self.root, cleanup_failure=True)
        _, run, _ = self.execute(backend)
        self.assertEqual(run["status"], "incomplete")
        self.assertEqual(backend.started, ["0001"])
        self.assertEqual(run["trials"][1]["status"], "pending")
        self.assertEqual(run["errors"][0]["phase"], "cleanup")

    def test_spreadsheet_failure_preserves_json_and_does_not_skip_later_trials(self):
        backend = LifecycleBackend(self.root)
        def unavailable_report(*args):
            raise PermissionError("Spreadsheet is open in Excel")
        output, run, _ = self.execute(backend, reporter=unavailable_report)
        self.assertEqual(backend.started, ["0001", "0002"])
        self.assertEqual(run["status"], "incomplete")
        self.assertEqual(len(run["errors"]), 1)
        self.assertEqual(json.loads((output / "run.json").read_text())["status"], "incomplete")

    def test_status_write_failure_during_cleanup_does_not_skip_resource_removal(self):
        backend = LifecycleBackend(self.root)
        def fail_cleanup_status(path, data):
            if path.name == "trial.json" and data.get("phase") == "cleanup":
                raise OSError("Status file unavailable")
            save_json(path, data)
        with patch("orchestrator.runner.save_json", side_effect=fail_cleanup_status):
            _, run, _ = self.execute(backend)
        self.assertEqual(backend.active, {"unrelated": {"owner": "user"}})
        self.assertEqual(run["status"], "incomplete")
        self.assertTrue(any(error["phase"] == "status_write" for error in run["errors"]))


if __name__ == "__main__":
    unittest.main()
