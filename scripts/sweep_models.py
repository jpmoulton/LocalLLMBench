"""Tune several models one after another, then write one cross-model report.

One ``llmbench tune`` session per GGUF (a session compares settings for ONE model, so different models cannot
share one), strictly sequential because they share a GPU. Works the same on Linux, macOS and Windows.

    python scripts/sweep_models.py --models a.gguf b.gguf --output artifacts/sweeps/today --budget-seconds 7200

Interrupting this script forwards the interrupt to the running session and WAITS for it: the session removes its
own containers on the way out. Killing the session outright is what leaks a container that keeps holding VRAM.

On an Apple Silicon Mac the same sweep runs the pinned native llama-server instead:

    python scripts/sweep_models.py --models a.gguf --output artifacts/sweeps/mac --runtime metal-native \\
        --native-bundle artifacts/native-prep/native-bundle.json

After every model the sweep checks that nothing was left behind before the next one is measured beside it:
llmbench-owned containers for NVIDIA; for the native runtime, the shared GPU lease file and any running process
whose executable is the bundle's llama-server (plus sandbox worker containers when Docker can be asked). The
checks only read; a leftover stops the sweep for a human to look at, it is never killed from here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEANUP_WAIT_SECONDS = 300
RUNTIMES = ("nvidia-container", "metal-native")


def slug_for(path: Path, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:48] or "model"
    slug, suffix = base, 2
    while slug in taken:
        slug, suffix = f"{base}-{suffix}", suffix + 1
    taken.add(slug)
    return slug


def discover(models: list[str], directory: str | None) -> list[Path]:
    found = [Path(item) for item in models]
    if directory:  # vision projectors (mmproj-*.gguf) are not standalone models
        found += sorted(item for item in Path(directory).iterdir()
                        if item.suffix.lower() == ".gguf" and not item.name.lower().startswith("mmproj"))
    missing = [str(item) for item in found if not item.is_file()]
    if missing:
        raise SystemExit("not a file: " + ", ".join(missing))
    if not found:
        raise SystemExit("no models given: use --models and/or --models-dir")
    return [item.resolve() for item in found]


def leaked_containers() -> list[str]:
    """Names of llmbench-owned containers still present. Empty when Docker cannot be asked (reported, not fatal)."""
    try:
        result = subprocess.run(["docker", "ps", "-a", "--filter", "name=llmbench-", "--format", "{{.Names}}"],
                                capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []


def _use_checkout() -> None:
    """Import this checkout's own ``llmbench`` (as ``run.py`` does); needed only by the native runtime's checks."""
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def native_executable(bundle: Path) -> str:
    """The llama-server executable a native bundle pins, read through the strict ``NativeBundle`` model."""
    _use_checkout()
    from llmbench.containers.config import read_native_bundle
    return read_native_bundle(bundle).native_server.executable


def lease_path() -> Path:
    """The shared GPU lease every live run takes (``containers.lease``), NVIDIA and native alike."""
    _use_checkout()
    from llmbench.containers.lease import default_lease_path
    return default_lease_path()


def running_servers(executable: str, *, processes=None) -> list[str]:
    """``pid:<n>`` for every running process whose executable IS ``executable``. Read-only: nothing is signalled.

    Compared by resolved path, so a symlinked install is still recognised. A process whose executable psutil cannot
    read (another user's, a zombie) reports ``None`` and is passed over: the session's own server runs as this user
    and is always readable. A process table that cannot be read at all is itself reported, because a check that
    could not be made must stop the sweep exactly like one that found something.
    """
    target = os.path.realpath(executable)
    try:
        if processes is None:
            import psutil
            processes = psutil.process_iter(["pid", "exe"])
        found = []
        for process in processes:
            exe = (process.info or {}).get("exe")
            if exe and os.path.realpath(exe) == target:
                found.append(f"pid:{process.info['pid']}")
        return found
    except Exception as exc:  # psutil missing or the process table unreadable: unverified, never "clean"
        return [f"process table unreadable ({type(exc).__name__}: {exc}), so {executable} cannot be shown stopped"]


def native_leftovers(executable: str) -> list[str]:
    """What a finished native session must not leave behind: the GPU lease, a running llama-server from the bundle,
    or a sandbox worker container (``leaked_containers``; empty when Docker is not there to ask)."""
    found = []
    lease = lease_path()
    if lease.exists():
        found.append(f"the GPU lease {lease} is still held")
    found += [f"llama-server {item} ({executable})" for item in running_servers(executable)]
    return found + leaked_containers()


def _runtime_arguments(args, parser) -> str | None:
    """The native bundle's executable for a metal-native sweep, None for NVIDIA; contradictions are refused the way
    ``llmbench tune`` refuses them, before any model is started."""
    if args.native_bundle is not None and args.runtime != "metal-native":
        parser.error("--native-bundle pins a metal-native llama-server; add --runtime metal-native")
    if args.runtime != "metal-native":
        return None
    if args.native_bundle is None:
        parser.error("--runtime metal-native requires --native-bundle (native-bundle.json from `llmbench prepare "
                     "--runtime metal-native`)")
    if args.image_bundle is not None:
        parser.error("--image-bundle pins container images; a metal-native sweep is pinned by --native-bundle")
    args.native_bundle = str(Path(args.native_bundle).resolve())  # each session runs with cwd=ROOT
    try:
        return native_executable(Path(args.native_bundle))
    except (OSError, ValueError) as exc:
        parser.error(f"--native-bundle {args.native_bundle}: {exc}")


def run_session(argv: list[str], log: Path) -> int:
    """Run one session to completion. An interrupt is forwarded so the session can clean up after itself."""
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
    with log.open("wb") as stream:
        child = subprocess.Popen(argv, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, **options)
        try:
            return child.wait()
        except KeyboardInterrupt:
            print("interrupt: asking the running session to stop and clean up ...", file=sys.stderr)
            child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            try:
                child.wait(timeout=CLEANUP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"the session (pid {child.pid}) has not stopped after {CLEANUP_WAIT_SECONDS} s; it is left "
                      "running rather than killed mid-cleanup", file=sys.stderr)
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=[], help="GGUF files, one session each")
    parser.add_argument("--models-dir", help="also sweep every *.gguf directly inside this directory")
    parser.add_argument("--output", required=True, help="a new or partly finished sweep directory")
    parser.add_argument("--budget-seconds", type=int, default=7200, help="wall budget per model (default 7200)")
    parser.add_argument("--image-bundle", help="image-bundle.json from `llmbench prepare`")
    parser.add_argument("--base-config", help="candidate template; default: the packaged one")
    parser.add_argument("--dataset-root", help="staged public-benchmark corpora")
    parser.add_argument("--context-floor", type=int)
    parser.add_argument("--context-ceiling", type=int)
    parser.add_argument("--policy", default="runtime-policy.json")
    parser.add_argument("--runtime", choices=RUNTIMES, default=None,
                        help="forwarded to `llmbench tune` (default: nvidia-container); metal-native needs "
                             "--native-bundle")
    parser.add_argument("--native-bundle", help="native-bundle.json from `llmbench prepare --runtime metal-native`")
    parser.add_argument("--capabilities-dir", help="forwarded to `llmbench tune`")
    args = parser.parse_args(argv)
    executable = _runtime_arguments(args, parser)

    models = discover(args.models, args.models_dir)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    index = {slug_for(path, taken): {"model": str(path)} for path in models}
    (output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    outcomes = []
    for slug, meta in index.items():
        run_dir = output / slug / "run"
        if run_dir.exists():
            print(f"{slug}: {run_dir} exists; skipped (use `llmbench resume --output` to continue it)")
            outcomes.append({"model": slug, "exit": None, "skipped": True})
            continue
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", str(ROOT / "run.py"), "tune", "--model", meta["model"], "--output",
                   str(run_dir), "--budget-seconds", str(args.budget_seconds), "--policy", args.policy]
        for flag, value in (("--image-bundle", args.image_bundle), ("--base-config", args.base_config),
                            ("--dataset-root", args.dataset_root), ("--context-floor", args.context_floor),
                            ("--context-ceiling", args.context_ceiling), ("--runtime", args.runtime),
                            ("--native-bundle", args.native_bundle), ("--capabilities-dir", args.capabilities_dir)):
            if value is not None:
                command += [flag, str(value)]
        started = time.monotonic()
        print(f"{slug}: tuning (log: {run_dir.parent / 'tune.log'})", flush=True)
        try:
            code = run_session(command, run_dir.parent / "tune.log")
        except KeyboardInterrupt:
            return 130
        minutes = round((time.monotonic() - started) / 60, 1)
        print(f"{slug}: exit {code} after {minutes} min", flush=True)
        outcomes.append({"model": slug, "exit": code, "minutes": minutes})
        leaked = leaked_containers() if executable is None else native_leftovers(executable)
        if leaked:  # the next model would be measured beside it; stop and let a human look
            what = "containers" if executable is None else "native runtime resources"
            print(f"stopping: {what} were left behind: " + ", ".join(leaked), file=sys.stderr)
            (output / "sweep-sequence.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
            return 4
    (output / "sweep-sequence.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    report = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "cross_model_report.py"), "--root",
                             str(output)], cwd=str(ROOT), capture_output=True, text=True, check=False)
    print(f"report: {output / 'cross-model-report.md'}" if report.returncode == 0
          else f"the cross-model report failed: {report.stderr.strip()[-400:]}")
    return 0 if all(item.get("exit") in (0, None) for item in outcomes) else 3


if __name__ == "__main__":
    raise SystemExit(main())
