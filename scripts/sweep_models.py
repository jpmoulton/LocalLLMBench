"""Tune several models one after another, then write one cross-model report.

One ``llmbench tune`` session per GGUF (a session compares settings for ONE model, so different models cannot
share one), strictly sequential because they share a GPU. Works the same on Linux, macOS and Windows.

    python scripts/sweep_models.py --models a.gguf b.gguf --output artifacts/sweeps/today --budget-seconds 7200

Interrupting this script forwards the interrupt to the running session and WAITS for it: the session removes its
own containers on the way out. Killing the session outright is what leaks a container that keeps holding VRAM.
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
    args = parser.parse_args(argv)

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
                            ("--context-ceiling", args.context_ceiling)):
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
        leaked = leaked_containers()
        if leaked:  # the next model would be measured beside it; stop and let a human look
            print("stopping: containers were left behind: " + ", ".join(leaked), file=sys.stderr)
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
