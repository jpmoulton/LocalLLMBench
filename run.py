"""Source-checkout entry point: ``python run.py <command>`` is ``llmbench <command>`` without installing."""
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llmbench.containers.cli import main  # noqa: E402

if __name__ == "__main__":
    # A supervising process on Windows can only interrupt a child process group with Ctrl-Break. Turn that into
    # the KeyboardInterrupt the candidate's cleanup path already handles, so containers are still removed.
    if hasattr(signal, "SIGBREAK"):
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK, interrupted)
    raise SystemExit(main())
