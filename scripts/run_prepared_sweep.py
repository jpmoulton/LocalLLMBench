"""Durable, explicit sequential candidates. Default is read-only --check; only --run starts inference.

A plan names the runtime its candidates run under with the optional ``inference_runtime`` (default
``nvidia-container``, pinned by ``image_bundle`` exactly as before). ``metal-native`` plans are pinned by
``native_bundle`` instead and take no ``image_bundle``; every config must run the plan's runtime and match its
bundle's pins, and each candidate is launched with ``--native-bundle``. Terminal evidence must prove cleanup either
way: no container, network or owned server process left, and for a native server that was started, how it exited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import signal
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llmbench.containers.cli import check_image_bundle, check_native_bundle  # noqa: E402
from llmbench.containers.config import (  # noqa: E402
    DEFAULT_RUNTIME, RUNTIMES, ContainerRunConfig, ContainerRunResult, read_image_bundle, read_native_bundle,
)
from llmbench.containers.preflight import check_path_lengths  # noqa: E402

NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
TOP_FIELDS = {"schema_version", "plan_id", "entries"}
OPTIONAL_FIELDS = {"min_free_disk_gib", "source_snapshot", "image_bundle", "inference_runtime", "native_bundle"}
BUNDLE_FIELD = {"nvidia-container": "image_bundle", "metal-native": "native_bundle"}
"""Which plan field pins each runtime's server. Exactly one is present: the other runtime's bundle has nothing to
say about these candidates, and accepting it would compare nothing and pass."""
ENTRY_FIELDS = {"id", "config", "output", "budget_seconds", "phase", "model"}
GLOBAL_LOCK_DIR = ROOT / "artifacts" / ".prepared-sweep-supervisor"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def resolve_path(base, value, *, output=False):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError("Paths must be nonempty strings without control characters")
    win = PureWindowsPath(value)
    raw = Path(value)
    if win.drive and not win.root:
        raise ValueError("Drive-relative paths are forbidden")
    if output:
        if raw.is_absolute() or win.anchor or ".." in win.parts or ".." in raw.parts:
            raise ValueError("Output must be a relative path inside the plan directory")
        for part in win.parts:
            if (":" in part or part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
                    or any(c in part for c in '<>"|?*')):
                raise ValueError("Output contains an unsafe Windows path component")
    elif os.name != "nt" and win.anchor:
        raise ValueError("Windows absolute paths require Windows")
    result = (base / raw).resolve()
    if output and (not result.is_relative_to(base) or result == base):
        raise ValueError("Output resolves outside the plan directory")
    return result


def read_plan(path):
    path = Path(path).resolve(strict=True)
    base = path.parent
    raw = load_json(path)
    runtime = raw.get("inference_runtime", DEFAULT_RUNTIME) if isinstance(raw, dict) else None
    if (not isinstance(raw, dict) or not TOP_FIELDS.issubset(raw) or set(raw) - TOP_FIELDS - OPTIONAL_FIELDS
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise ValueError("Plan requires exactly schema_version=1, plan_id, image_bundle, entries")
    if runtime not in RUNTIMES:
        raise ValueError(f"inference_runtime must be one of {', '.join(RUNTIMES)}")
    wanted, other = BUNDLE_FIELD[runtime], BUNDLE_FIELD[next(item for item in RUNTIMES if item != runtime)]
    if wanted not in raw or other in raw:
        raise ValueError(f"A plan for the {runtime} runtime requires {wanted} and takes no {other}")
    if not isinstance(raw["plan_id"], str) or not NAME.fullmatch(raw["plan_id"]):
        raise ValueError("Invalid plan_id")
    if not isinstance(raw["entries"], list) or not raw["entries"]:
        raise ValueError("Plan entries must be a nonempty array")
    minimum_disk = raw.get("min_free_disk_gib", 12)
    if type(minimum_disk) not in (int, float) or not math.isfinite(minimum_disk) or minimum_disk < 1:
        raise ValueError("min_free_disk_gib must be finite and at least 1")
    bundle_path = resolve_path(base, raw[wanted])
    bundle = read_image_bundle(bundle_path) if runtime == DEFAULT_RUNTIME else read_native_bundle(bundle_path)
    metadata = base / (".sweep-" + raw["plan_id"])
    if metadata.resolve() != metadata or not metadata.resolve().is_relative_to(base):
        raise ValueError("Supervisor metadata cannot be a symlink or junction")
    entries, ids, outputs = [], set(), []
    for item in raw["entries"]:
        if not isinstance(item, dict) or set(item) != ENTRY_FIELDS:
            raise ValueError("Each entry requires exactly id, config, output, budget_seconds, phase, model")
        identifier = item["id"]
        if not isinstance(identifier, str) or not NAME.fullmatch(identifier) or identifier in ids:
            raise ValueError("Entry IDs must be valid and unique")
        ids.add(identifier)
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ("phase", "model")):
            raise ValueError("Entry phase and model must be nonempty strings")
        budget = item["budget_seconds"]
        if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
            raise ValueError("budget_seconds must be finite and positive")
        config_path = resolve_path(base, item["config"])
        config = ContainerRunConfig.model_validate_json(config_path.read_bytes())
        if config.runtime != runtime:  # checked first: the other runtime's bundle check would say less
            raise ValueError(f"{identifier}: the config runs {config.runtime} but the plan's inference_runtime is "
                             f"{runtime}")
        if runtime == DEFAULT_RUNTIME:
            mismatches = check_image_bundle(config, bundle)
            if mismatches:
                raise ValueError(f"{identifier}: image bundle mismatch: {'; '.join(mismatches)}")
        else:
            mismatches = check_native_bundle(config, bundle)
            if mismatches:
                raise ValueError(f"{identifier}: native bundle mismatch: {'; '.join(mismatches)}")
        output = resolve_path(base, item["output"], output=True)
        check_path_lengths(output)
        if output.is_relative_to(metadata) or metadata.is_relative_to(output):
            raise ValueError("Output overlaps supervisor metadata")
        if any(str(output).casefold() == str(other).casefold()
               or output.is_relative_to(other) or other.is_relative_to(output) for other in outputs):
            raise ValueError("Outputs must be unique and cannot overlap")
        if config_path == output or config_path.is_relative_to(output) or path.is_relative_to(output):
            raise ValueError("Output cannot contain an input file")
        outputs.append(output)
        entries.append({**item, "config": str(config_path), "output": str(output),
                        "config_sha256": digest(config_path), "config_fingerprint": config.fingerprint()})
    snapshot_sha256 = None
    if "source_snapshot" in raw:
        snapshot_path = resolve_path(base, raw["source_snapshot"], output=True)
        snapshot = load_json(snapshot_path)
        if (not isinstance(snapshot, dict) or set(snapshot) != {"files"}
                or not isinstance(snapshot["files"], dict) or not snapshot["files"]):
            raise ValueError("source_snapshot requires exactly a nonempty files mapping")
        for relative, expected in snapshot["files"].items():
            source = resolve_path(ROOT, relative, output=True)
            if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                    or digest(source) != expected):
                raise ValueError(f"Prepared source snapshot mismatch: {relative}")
        snapshot_sha256 = digest(snapshot_path)
    source_hash = hashlib.sha256()
    for source in sorted([ROOT / "run.py", Path(__file__).resolve(), *(ROOT / "src").rglob("*.py")]):
        source_hash.update(str(source.relative_to(ROOT)).replace("\\", "/").encode())
        source_hash.update(source.read_bytes())
    identity = {"plan_sha256": digest(path), "bundle_sha256": digest(bundle_path),
                "runtime_sha256": source_hash.hexdigest(), "source_snapshot_sha256": snapshot_sha256,
                "configs": {entry["id"]: entry["config_sha256"] for entry in entries}}
    if runtime != DEFAULT_RUNTIME:  # an NVIDIA identity keeps its exact keys, so its state.json still resumes
        identity["inference_runtime"] = runtime
    return {"path": str(path), "base": str(base), "plan_id": raw["plan_id"], "entries": entries,
            "inference_runtime": runtime,
            "image_bundle": str(bundle_path) if runtime == DEFAULT_RUNTIME else None,
            "native_bundle": None if runtime == DEFAULT_RUNTIME else str(bundle_path),
            "metadata": str(metadata), "identity": identity, "min_free_disk_gib": minimum_disk}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Ledger:
    def __init__(self, plan, *, resume=False):
        self.root = Path(plan["metadata"])
        self.path, self.journal = self.root / "state.json", self.root / "journal.jsonl"
        if self.path.exists():
            if not resume:
                raise ValueError("State exists; use --run --resume to continue unchanged pending entries")
            self.state = load_json(self.path)
            if self.state.get("identity") != plan["identity"] or self.state.get("plan_id") != plan["plan_id"]:
                raise ValueError("Plan, config, image bundle or runtime changed; use a new plan/output")
            events = self.journal.read_bytes().splitlines()
            if (len(events) != self.state.get("event_count") or not events
                    or hashlib.sha256(events[-1]).hexdigest() != self.state.get("last_event_sha256")):
                raise ValueError("Journal/state disagree; reconcile interrupted persistence before a new plan")
            if set(self.state.get("entries", {})) != {entry["id"] for entry in plan["entries"]}:
                raise ValueError("State entry set differs from the plan")
        else:
            if resume:
                raise ValueError("No state to resume; use --run for a fresh plan")
            if self.journal.exists():
                raise ValueError("Orphan journal exists; reconcile before a new plan")
            self.state = {"schema_version": 1, "plan_id": plan["plan_id"], "identity": plan["identity"],
                          "event_count": 0, "entries": {e["id"]: {"status": "pending"} for e in plan["entries"]}}
            self.record("initialized")

    def record(self, event, identifier=None, **fields):
        now = datetime.now(timezone.utc).isoformat()
        row = {"event": event, "entry": identifier, "utc": now, "supervisor_pid": os.getpid(), **fields}
        if identifier:
            self.state["entries"][identifier].update(fields)
        payload = json.dumps(row, sort_keys=True, allow_nan=False).encode()
        with self.journal.open("ab") as stream:
            stream.write(payload + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.state.update(event_count=self.state["event_count"] + 1,
                          last_event_sha256=hashlib.sha256(payload).hexdigest(), updated_utc=now)
        atomic_json(self.path, self.state)


@contextmanager
def supervisor_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "supervisor.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another supervisor holds this plan's process lock") from exc
        try:
            stream.seek(0)
            stream.write(json_bytes({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}))
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def guard_processes(blocked=(), *, platform=None, runner=subprocess.run):
    """Refuse to launch while any process named in ``blocked`` is running (case-insensitive). Never kills one.

    Opt-in (``--refuse-while-running``): a game or another inference server that would share the GPU and skew
    the measurements. The candidate's own foreign-VRAM admission check is the real protection; this only pauses
    the sweep earlier, with a clearer message. A process list that cannot be read refuses the launch.
    """
    wanted = {name.casefold() for name in blocked}
    if not wanted:
        return
    windows = (platform or os.name) == "nt"
    argv = ["tasklist.exe", "/FO", "CSV", "/NH"] if windows else ["ps", "-A", "-o", "comm="]
    result = runner(argv, capture_output=True, text=True, timeout=15, check=False)
    if result.returncode != 0:
        raise ValueError("Cannot verify running processes; refusing GPU launch")
    if windows:
        rows = list(csv.reader(io.StringIO(result.stdout)))
        if not rows or any(len(row) < 5 or not row[1].isdigit() for row in rows):
            raise ValueError("Cannot parse the process inventory; refusing GPU launch")
        names = {row[0].casefold() for row in rows}
    else:
        names = {Path(line.strip()).name.casefold() for line in result.stdout.splitlines() if line.strip()}
        if not names:
            raise ValueError("Cannot parse the process inventory; refusing GPU launch")
    running = sorted(names & wanted)
    if running:
        raise ValueError("Close these before GPU testing: " + ", ".join(running))


def guard_disk(plan, *, usage=shutil.disk_usage):
    free = usage(plan["base"]).free
    if free < plan["min_free_disk_gib"] * 1024 ** 3:
        raise ValueError(f"Output drive has {free / 1024 ** 3:.2f} GiB free; "
                         f"at least {plan['min_free_disk_gib']} GiB required")


def terminal(entry, returncode):
    """The checkpointed outcome of one candidate, or a refusal when its evidence does not prove cleanup.

    A native candidate's cleanup evidence names the owned server processes still present
    (``processes_remaining``, which must be empty exactly like ``containers_remaining``) and how the server it
    started exited (``server_exit``). When the runner attempted a cleanup it started a server, so a missing exit is
    missing proof, not a clean stop. An NVIDIA result is judged exactly as before and checkpointed with the same keys.
    """
    path = Path(entry["output"]) / "result.json"
    result = ContainerRunResult.model_validate_json(path.read_bytes(), strict=True)
    if result.config_fingerprint != entry["config_fingerprint"] or result.synthetic:
        raise ValueError("Terminal evidence is synthetic or belongs to another configuration")
    cleanup = result.cleanup
    if (not cleanup.verified or cleanup.containers_remaining or cleanup.networks_remaining
            or cleanup.processes_remaining or cleanup.lease_retained or cleanup.error or result.abort_campaign):
        raise ValueError("Candidate cleanup is uncertain; preserve evidence and reconcile owned resources")
    native = result.runtime != DEFAULT_RUNTIME
    if native and cleanup.attempted and not cleanup.server_exit:
        raise ValueError("Native candidate cleanup does not record how its llama-server exited; preserve evidence "
                         "and confirm the server process is gone")
    outcome = {"status": "completed" if returncode == 0 and result.state == "completed" else "failed",
               "returncode": returncode, "result_state": result.state, "result_sha256": digest(path),
               "cleanup_verified": True, "attempt_id": result.attempt_id,
               "failure_reasons": list(result.failure_reasons)}
    if native:
        outcome.update(runtime=result.runtime, server_exit=cleanup.server_exit)
    return outcome


def verify_resume(plan, ledger):
    for entry in plan["entries"]:
        old = ledger.state["entries"][entry["id"]]
        if old["status"] in ("completed", "failed"):
            current = terminal(entry, old["returncode"])
            if any(current.get(key) != old.get(key) for key in current):
                raise ValueError("Terminal evidence changed since checkpoint; use a new plan after reconciliation")
        elif old["status"] != "pending":
            raise ValueError(f"{entry['id']}: interrupted/blocked launch; reconcile child PID {old.get('child_pid')} "
                             "and owned resources before creating a new plan/output")
        elif Path(entry["output"]).exists():
            raise ValueError(f"Refusing to overwrite existing output: {entry['output']}")


def candidate_command(plan, entry):
    # The checkout's own virtualenv when it has one, otherwise the interpreter running this supervisor.
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        python = Path(sys.executable)
    bundle = (["--image-bundle", plan["image_bundle"]] if plan["inference_runtime"] == DEFAULT_RUNTIME
              else ["--native-bundle", plan["native_bundle"]])
    return [str(python), "-B", str(ROOT / "run.py"), "candidate", "--config", entry["config"],
            "--output", entry["output"], "--budget-seconds", str(entry["budget_seconds"]), *bundle]


def request_cleanup(child, seconds):
    # A second Ctrl-C must not prevent the already requested bounded cleanup wait.
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
            if sig is not None:
                previous[sig] = signal.signal(sig, signal.SIG_IGN)
        if child.poll() is None:
            child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        return child.wait(timeout=seconds)
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_plan(plan, *, resume=False, popen=subprocess.Popen, process_guard=guard_processes,
             disk_guard=guard_disk, cleanup_wait=180):
    with supervisor_lock(GLOBAL_LOCK_DIR), supervisor_lock(plan["metadata"]):
        ledger = Ledger(plan, resume=resume)
        verify_resume(plan, ledger)
        for entry in plan["entries"]:
            identifier = entry["id"]
            if ledger.state["entries"][identifier]["status"] != "pending":
                continue
            # Revalidate on every launch so an edit during a long sweep cannot silently change later work.
            if read_plan(plan["path"]) != plan:
                raise ValueError("Sweep inputs/runtime changed after validation; no next candidate launched")
            try:
                process_guard()
                disk_guard(plan)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                ledger.record("launch_paused", identifier, reason=str(exc))
                print(f"Sweep paused before {identifier}: {exc}", file=sys.stderr)
                return 2
            log = Path(plan["metadata"]) / (identifier + ".stdout.log")
            argv = candidate_command(plan, entry)
            interrupted = False
            child = None
            cleanup_requested = False
            cleanup_returncode = None
            with log.open("xb") as stdout:
                ledger.record("launching", identifier, status="launching", child_pid=None,
                              stdout_file=str(log), argv=argv, output=entry["output"])
                try:
                    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
                    child = popen(argv, cwd=str(ROOT), stdout=stdout, stderr=subprocess.STDOUT, **options)
                    ledger.record("started", identifier, status="running", child_pid=child.pid)
                    try:
                        code = child.wait(timeout=entry["budget_seconds"] + cleanup_wait)
                    except (KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
                        interrupted = isinstance(exc, KeyboardInterrupt)
                        ledger.record("cleanup_requested", identifier, reason=type(exc).__name__)
                        cleanup_requested = True
                        code = request_cleanup(child, cleanup_wait)
                        cleanup_returncode = code
                    stdout.flush()
                    os.fsync(stdout.fileno())
                    if code is None:
                        raise ValueError("Child did not stop within cleanup allowance; PID retained; no forced kill")
                    result = terminal(entry, code)
                except BaseException as exc:
                    if child is not None and child.poll() is None and not cleanup_requested:
                        cleanup_requested = True
                        cleanup_returncode = request_cleanup(child, cleanup_wait)
                    ledger.record("blocked", identifier, status="blocked", error=f"{type(exc).__name__}: {exc}",
                                  child_pid=child.pid if child is not None else None,
                                  child_stopped=child.poll() is not None if child is not None else True,
                                  cleanup_signal_requested=cleanup_requested, cleanup_returncode=cleanup_returncode)
                    return 130 if isinstance(exc, KeyboardInterrupt) else 4
                ledger.record("finished", identifier, **result)
            print(f"{identifier}: {result['status']} (candidate {result['result_state']}; cleanup verified)", flush=True)
            if interrupted:
                ledger.record("interrupted")
                return 130
        failures = sum(value["status"] != "completed" for value in ledger.state["entries"].values())
        ledger.record("sweep_finished", failures=failures)
        return 3 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate only (the default); no runtime actions")
    mode.add_argument("--run", action="store_true", help="Explicitly launch the prepared candidates sequentially")
    parser.add_argument("--resume", action="store_true", help="Continue pending entries without rerunning terminal attempts")
    parser.add_argument("--cleanup-wait-seconds", type=int, default=180)
    parser.add_argument("--refuse-while-running", action="append", default=[], metavar="PROCESS",
                        help="pause instead of launching while this process is running (repeatable), e.g. a game "
                             "or another inference server that would share the GPU")
    args = parser.parse_args(argv)
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if not 15 <= args.cleanup_wait_seconds <= 1800:
        parser.error("--cleanup-wait-seconds must be between 15 and 1800")
    try:
        plan = read_plan(args.plan)
        if not args.run:
            report = {"mode": "check", "plan_id": plan["plan_id"], "entries": len(plan["entries"]),
                      "total_candidate_budget_seconds": sum(e["budget_seconds"] for e in plan["entries"]),
                      "identity": plan["identity"], "metadata": plan["metadata"],
                      "min_free_disk_gib": plan["min_free_disk_gib"]}
            if plan["inference_runtime"] != DEFAULT_RUNTIME:  # an NVIDIA check prints exactly what it always did
                report = {**report, "inference_runtime": plan["inference_runtime"]}
            print(json.dumps(report, indent=2))
            return 0
        def stop(_signum, _frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            return run_plan(plan, resume=args.resume, cleanup_wait=args.cleanup_wait_seconds,
                            process_guard=lambda: guard_processes(args.refuse_while_running))
        finally:
            signal.signal(signal.SIGTERM, previous)
    except KeyboardInterrupt:
        print("Interrupted; retain supervisor state and inspect candidate cleanup before resuming", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"Prepared sweep refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())