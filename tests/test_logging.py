"""Text-stream routing, live tailing, and follower resource ownership."""

import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from orchestrator.logging import RunLogs
from orchestrator.runner import DockerBackend


class LoggingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.live_dir = self.root / "logs"
        self.console = io.StringIO()

    def test_preview_without_execution_does_not_create_logs(self):
        with RunLogs(live_dir=self.live_dir, console=self.console) as logs:
            logs.print("plan preview")
        self.assertEqual(self.console.getvalue(), "plan preview\n")
        self.assertFalse(self.live_dir.exists())
        self.assertFalse((self.root / "logging").exists())

    def test_live_files_and_archives_keep_sources_and_trials_isolated(self):
        with RunLogs(live_dir=self.live_dir, console=self.console) as logs:
            logs.print("plan preview")
            self.assertFalse((self.root / "logging").exists())
            logs.open_run(self.root)
            for trial in ("0001", "0002"):
                logs.trial = trial
                logs.print(f"starting {trial}")
                logs.emit("llama.cpp", f"loaded {trial}\n")
                logs.emit("benchmarks", f"scored {trial}\n")
            # Read while writers remain open, as a tail window would.
            own = (self.root / "logging/localllmbench.log").read_text(encoding="utf-8")
            inference = (self.root / "logging/llamacpp.log").read_text(encoding="utf-8")
            benchmarks = (self.root / "logging/benchmarks.log").read_text(encoding="utf-8")
            self.assertIn("[localllmbench] [run] plan preview", own)
            for trial in ("0001", "0002"):
                self.assertIn(f"[llama.cpp] [trial {trial}] loaded {trial}", inference)
                self.assertIn(f"[benchmarks] [trial {trial}] scored {trial}", benchmarks)
            for filename in ("localllmbench.log", "llamacpp.log", "benchmarks.log"):
                archived = (self.root / "logging" / filename).read_text(encoding="utf-8")
                self.assertEqual((self.live_dir / filename).read_text(encoding="utf-8"), archived)
                self.assertIn(f"[run {self.root.name}]", archived)
            self.assertNotIn("loaded", own + benchmarks)
            self.assertNotIn("scored", own + inference)
            self.assertNotIn("loaded", self.console.getvalue())
            self.assertNotIn("scored", self.console.getvalue())
            self.assertIn("starting 0002", self.console.getvalue())
        self.assertEqual({p.name for p in (self.root / "logging").iterdir()},
                         {"localllmbench.log", "llamacpp.log", "benchmarks.log"})

    def test_tail_handle_survives_runs_and_archives_do_not_mix(self):
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        with RunLogs(live_dir=self.live_dir, console=self.console) as logs:
            logs.open_run(first)
            logs.emit("llama.cpp", "first model ready\n")
        first_archive = (first / "logging/llamacpp.log").read_text(encoding="utf-8")
        with (self.live_dir / "llamacpp.log").open(encoding="utf-8") as tail:
            self.assertEqual(tail.read(), first_archive)
            with RunLogs(live_dir=self.live_dir, console=self.console) as logs:
                logs.open_run(second)
                logs.emit("llama.cpp", "second model ready\n")
                second_archive = (second / "logging/llamacpp.log").read_text(encoding="utf-8")
                self.assertEqual(tail.read(), second_archive)
        self.assertIn("[run first]", first_archive)
        self.assertNotIn("second model", first_archive)
        self.assertIn("[run second]", second_archive)
        self.assertNotIn("first model", second_archive)
        self.assertEqual((first / "logging/llamacpp.log").read_text(encoding="utf-8"), first_archive)
        self.assertEqual((self.live_dir / "llamacpp.log").read_text(encoding="utf-8"),
                         first_archive + second_archive)

    def test_concurrent_sources_keep_complete_records(self):
        with RunLogs(live_dir=self.live_dir, console=self.console) as logs:
            logs.open_run(self.root)
            logs.trial = "0001"
            threads = [threading.Thread(target=lambda source=source: [
                logs.emit(source, f"record-{index}\n") for index in range(20)
            ]) for source in ("llama.cpp", "benchmarks")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            text = "".join((self.live_dir / filename).read_text(encoding="utf-8")
                           for filename in ("llamacpp.log", "benchmarks.log"))
        expected = {f"[{source}] [trial 0001] record-{index}"
                    for source in ("llama.cpp", "benchmarks") for index in range(20)}
        records = [line.split("] ", 1)[1] for line in text.splitlines()]
        self.assertCountEqual(records, expected)
        self.assertEqual(self.console.getvalue(), "")

    def test_child_output_is_saved_without_hiding_nonzero_exit(self):
        with RunLogs(live_dir=self.live_dir, console=self.console) as logs:
            logs.open_run(self.root)
            backend = DockerBackend(logs=logs)
            code = backend.run_benchmarks([
                sys.executable, "-u", "-c",
                "import sys; print('child stdout'); print('child stderr',file=sys.stderr); raise SystemExit(7)",
            ], lambda: None)
            logs.print(f"controller failed with exit {code}")
        self.assertEqual(code, 7)
        self.assertEqual(self.console.getvalue(), "controller failed with exit 7\n")
        saved = (self.live_dir / "benchmarks.log").read_text(encoding="utf-8")
        self.assertIn("child stdout", saved)
        self.assertIn("child stderr", saved)
        self.assertEqual(saved, (self.root / "logging/benchmarks.log").read_text(encoding="utf-8"))

    def test_docker_failure_still_closes_live_log_process_and_reader(self):
        received = threading.Event()

        with RunLogs(live_dir=self.live_dir, console=self.console) as logs:
            logs.open_run(self.root)
            backend = DockerBackend(logs=logs)
            emit = logs.emit

            def observe(source, text):
                emit(source, text)
                received.set()

            process = subprocess.Popen(
                [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                with patch.object(logs, "emit", side_effect=observe), \
                     patch("orchestrator.runner.subprocess.Popen", return_value=process):
                    backend._follow_logs("fixture")
                    self.assertTrue(received.wait(5), "Log stream never became active")
                reader = backend._log_reader
                with patch.object(backend, "command", side_effect=RuntimeError("Docker unavailable")):
                    with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                        backend.cleanup({"localllmbench.run": "fixture", "localllmbench.trial": "0001"})
                self.assertIsNotNone(process.poll())
                self.assertFalse(reader.is_alive())
                self.assertTrue(process.stdout.closed)
            finally:
                backend._stop_log_stream()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()
