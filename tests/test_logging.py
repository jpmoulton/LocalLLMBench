"""Log follower processes must not survive a failed container cleanup."""

from contextlib import redirect_stdout
import io
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from orchestrator.runner import DockerBackend


class LogCleanupTests(unittest.TestCase):
    def test_docker_failure_still_closes_live_log_process_and_reader(self):
        backend = DockerBackend(verbose_logging=True)
        self.addCleanup(backend._stop_log_stream)
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        def remove_process():
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
        self.addCleanup(remove_process)
        received = threading.Event()

        class Capture(io.StringIO):
            def write(self, text):
                written = super().write(text)
                if "ready" in text:
                    received.set()
                return written

        with redirect_stdout(Capture()):
            with patch("orchestrator.runner.subprocess.Popen", return_value=process):
                backend._follow_logs("fixture")
            reader = backend._log_reader
            self.assertTrue(received.wait(5), "Log stream never became active")
            with patch.object(backend, "command", side_effect=RuntimeError("Docker unavailable")):
                with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                    backend.cleanup({"localllmbench.run": "fixture", "localllmbench.trial": "0001"})
        self.assertIsNotNone(process.poll())
        self.assertFalse(reader.is_alive())
        self.assertTrue(process.stdout.closed)


if __name__ == "__main__":
    unittest.main()
