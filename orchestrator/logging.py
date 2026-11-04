"""Append live streams to stable files and mirror each run into its results."""

from pathlib import Path
import sys
import threading

from orchestrator.state import now

LIVE_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


class RunLogs:
    FILES = {
        "localllmbench": "localllmbench.log",
        "llama.cpp": "llamacpp.log",
        "benchmarks": "benchmarks.log",
    }

    def __init__(self, *, live_dir: Path | None = None, console=None):
        self.live_dir = (LIVE_LOG_DIR if live_dir is None else live_dir).resolve()
        self.console = sys.stdout if console is None else console
        self.trial = None
        self._lock = threading.Lock()
        self._files = {}
        self._pending = []
        self._opened = False
        self._run_id = None

    def open_run(self, output: Path) -> None:
        """Create files once execution starts; retain the preceding plan preview."""
        with self._lock:
            if self._opened:
                raise RuntimeError("Log router already belongs to a run")
            self._opened = True
            self._run_id = output.name
            directory = output / "logging"
            directory.mkdir(exist_ok=False)
            self.live_dir.mkdir(parents=True, exist_ok=True)
            try:
                for source, filename in self.FILES.items():
                    self._files[source] = []
                    for path, mode in ((self.live_dir / filename, "a"), (directory / filename, "x")):
                        self._files[source].append(path.open(mode, encoding="utf-8"))
                for source, record in self._pending:
                    self._save(source, record)
            except BaseException:
                for files in self._files.values():
                    for file in files:
                        file.close()
                self._files.clear()
                raise
            finally:
                self._pending.clear()

    def _save(self, source: str, text: str) -> None:
        scope = f"[trial {self.trial}]" if self.trial else "[run]"
        label = f"[run {self._run_id}] [{source}] {scope}"
        record = "".join(f"{now()} {label} {line}\n" for line in text.splitlines())
        for file in self._files[source]:
            file.write(record)
            file.flush()  # Both tail readers and archived results see entries immediately.

    def emit(self, source: str, text: str) -> None:
        if source not in self.FILES:
            raise ValueError(f"Unknown log stream: {source}")
        if not text:
            return
        with self._lock:
            if self._opened:
                self._save(source, text)
            else:
                self._pending.append((source, text))
            if source == "localllmbench":
                self.console.write(text)
                self.console.flush()

    def print(self, *values, sep: str = " ", end: str = "\n") -> None:
        self.emit("localllmbench", sep.join(str(value) for value in values) + end)

    def close(self) -> None:
        with self._lock:
            for files in self._files.values():
                for file in files:
                    file.close()
            self._files.clear()
            self._pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
