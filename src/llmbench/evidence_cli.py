"""Read-only commands over recorded evidence: nothing here contacts a server, a GPU or Docker.

``analyze`` is the useful one: it re-scores a finished campaign from its result store under a different
policy, so acceptance thresholds can be recalibrated without spending another GPU hour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

COMMANDS = ("doctor", "list", "show", "analyze", "report")
MAX_HASHED_BYTES = 1 << 30  # a llama-server build is tens of MiB; anything past this is not one, and is not read


def add_evidence_commands(sub: argparse._SubParsersAction) -> None:
    doctor = sub.add_parser("doctor", help="Show package versions, the runtime policy and static runtime facts; "
                                           "contacts nothing and runs nothing")
    doctor.add_argument("--policy", default="runtime-policy.json")
    doctor.add_argument("--native-bundle", metavar="PATH",
                        help="native-bundle.json from `llmbench prepare --runtime metal-native`: report whether its "
                             "llama-server and libraries are still the pinned files (hashes them; runs nothing)")
    for name, text in (("list", "List the attempts recorded in a result store"),
                       ("show", "Show one recorded attempt with its samples and events")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--store", required=True, help="a run directory holding results.sqlite3")
        if name == "show":
            command.add_argument("attempt_id")
    analyze = sub.add_parser("analyze", help="Re-score a finished campaign under a (different) policy; no GPU")
    analyze.add_argument("campaign", help="reports/campaign.json from a finished run")
    analyze.add_argument("--store", required=True, help="the run directory holding results.sqlite3")
    analyze.add_argument("--policy", required=True, help="a CampaignPolicy JSON file")
    analyze.add_argument("--baseline", help="attempt id to compare against; default: the campaign's own")
    analyze.add_argument("--output", required=True)
    report = sub.add_parser("report", help="Rebuild the JSON/Markdown/HTML report views from a campaign JSON")
    report.add_argument("campaign")
    report.add_argument("--output", required=True)


def _bounded_sha256(path: Path, limit: int = MAX_HASHED_BYTES) -> tuple[str | None, str | None]:
    """(sha256, None) for a regular file of at most ``limit`` bytes, else (None, why not).

    Opened non-blocking and checked with ``fstat`` AFTER opening, so a FIFO or device swapped in after the
    existence check can neither hang doctor nor be hashed."""
    try:
        if not path.is_file():
            return None, "missing" if not path.exists() else "not a regular file"
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None, "not a regular file"
            digest, total = hashlib.sha256(), 0
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                if total > limit:
                    return None, f"larger than {limit} bytes; not hashed"
                digest.update(block)
        return digest.hexdigest(), None
    except OSError as exc:
        return None, f"unreadable: {type(exc).__name__}: {exc}"


def _library_sha256(directory: Path, name: str) -> tuple[str | None, str | None]:
    """A library beside the executable, by the rule the native runner pins it by: the name may be a link, but only
    to a file in that same directory (dyld loads ``@rpath`` names, so a link elsewhere is refused, not followed)."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        return None, "invalid name; not read"
    candidate = directory / name
    try:
        if candidate.is_symlink() and candidate.resolve().parent != directory.resolve():
            return None, "a link outside the executable's directory, which the native runner refuses; not read"
    except (OSError, RuntimeError) as exc:  # RuntimeError: a link loop on older Pythons
        return None, f"unreadable: {type(exc).__name__}: {exc}"
    return _bounded_sha256(candidate)


def native_bundle_state(path: str | Path) -> dict[str, Any]:
    """Whether a native bundle's pinned server is still on this host, unchanged. Hashes files; executes nothing.

    It asks the native runner's own admission question, statically. The executable must be the pinned REGULAR
    file (the runner refuses a link, which could be repointed after the pin). The executable alone is not the
    server: the Metal backend lives in the ``libggml-*`` libraries beside it, and the pin is ``libraries_sha256``
    over EVERY ``lib*.dylib``/``lib*.so`` name in that directory -- so a library added since preparation changes
    the server exactly as a rebuilt one does, and is reported ("not in the bundle") rather than overlooked because
    the bundle never listed it. ``libraries_sha256_matches`` is the runner's comparison; the per-name states say
    which file caused a mismatch. A library name from the bundle that is not a plain file name is reported and
    never opened, so a bundle cannot direct this read outside the executable's directory. The runner repeats the
    full identity check (including the executable's own ``--version``/``--help``) at admission; this is the
    static, run-nothing half a person can use before a session.
    """
    from fnmatch import fnmatchcase

    from .config import canonical_json
    from .containers.config import read_native_bundle
    from .containers.native_prep import LIBRARY_PATTERNS, MAX_LIBRARIES
    bundle = read_native_bundle(path)
    server = bundle.native_server
    executable = Path(server.executable)
    if executable.is_symlink():
        observed, problem = None, "a link; the native runner pins only a regular file, never a link"
    else:
        observed, problem = _bounded_sha256(executable)
    directory, recorded = executable.parent, bundle.libraries
    try:
        present: set[str] | None = {name for name in os.listdir(directory)
                                    if any(fnmatchcase(name, pattern) for pattern in LIBRARY_PATTERNS)}
    except OSError:
        present = None  # the executable's own state already says why; no set digest without the set
    names = sorted(set(recorded) | (present or set()))
    libraries: dict[str, str] = {}
    hashed: dict[str, str] | None = None if present is None or len(names) > MAX_LIBRARIES else {}
    for name in names[:MAX_LIBRARIES]:
        digest, why = _library_sha256(directory, name)
        if hashed is not None and name in present:
            if digest is None:
                hashed = None
            else:
                hashed[name] = digest
        if name not in recorded:
            libraries[name] = "not in the bundle" + (f"; {why}" if why else "")
        else:
            libraries[name] = why or ("matches" if digest == recorded[name] else "differs")
    if len(names) > MAX_LIBRARIES:
        libraries["..."] = f"{len(names) - MAX_LIBRARIES} more libraries not checked"
    libraries_observed = (None if hashed is None
                          else hashlib.sha256(canonical_json(hashed).encode("utf-8")).hexdigest())
    return {"path": str(path), "executable": server.executable, "executable_exists": executable.is_file(),
            "executable_sha256_pinned": server.executable_sha256, "executable_sha256_observed": observed,
            "executable_sha256_matches": None if observed is None else observed == server.executable_sha256,
            "executable_problem": problem, "libraries": libraries,
            # None when there was no library to compare: nothing was compared, which is not a match.
            "libraries_match": all(state == "matches" for state in libraries.values()) if libraries else None,
            "libraries_sha256_pinned": server.libraries_sha256, "libraries_sha256_observed": libraries_observed,
            "libraries_sha256_matches": (None if libraries_observed is None
                                         else libraries_observed == server.libraries_sha256),
            "build_info": server.build_info, "sandbox": bundle.sandbox.get("status")}


def _total_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):  # no sysconf (Windows) or the name is unknown here
        return None


def runtime_facts(*, native_bundle: str | Path | None = None) -> dict[str, Any]:
    """Static facts for choosing and checking a runtime. Reads constants and files only: no process is started
    (``platform`` answers from ``uname`` and the macOS version plist, ``shutil.which`` searches PATH), and nothing
    here measures free memory, swap or pressure -- those change by the second and belong to a run's admission.
    """
    import platform
    import shutil
    from .containers.capabilities import HELP_FILE
    from .containers.lease import default_lease_path
    from .containers.runtime import RUNTIME_SPECS
    from .locks import describe_lock
    lease = default_lease_path()
    held = lease.exists() or lease.is_symlink()
    return {"system": platform.system(), "machine": platform.machine(), "macos": platform.mac_ver()[0] or None,
            "memory_total_bytes": _total_memory_bytes(), "docker_cli": shutil.which("docker"),
            "runtimes": {name: {"memory_kind": spec.memory_kind,
                                "capabilities_dir": spec.default_capabilities_dir,
                                "capabilities_present": Path(spec.default_capabilities_dir, HELP_FILE).is_file()}
                         for name, spec in RUNTIME_SPECS.items()},
            "lease": {"path": str(lease), "exists": held, "holder": describe_lock(lease) if held else None},
            "native_bundle": None if native_bundle is None else native_bundle_state(native_bundle),
            "execution_performed": False}


def run_evidence_command(args: argparse.Namespace) -> Any:
    from .reports import write_reports
    from .store import Store
    if args.command == "doctor":
        from .provenance import environment_record
        from .safety import SessionLock
        return {"environment": environment_record(), "session_policy": vars(SessionLock.read(args.policy)),
                "runtime": runtime_facts(native_bundle=getattr(args, "native_bundle", None)),
                "server_contacted": False, "model_operations_performed": False}
    if args.command in ("list", "show"):
        if not (Path(args.store) / "results.sqlite3").exists():
            raise ValueError(f"no result store at {args.store}")
        with Store(args.store) as store:
            return store.attempts() if args.command == "list" else store.results(args.attempt_id)
    campaign = json.loads(Path(args.campaign).read_text(encoding="utf-8"))
    if args.command == "analyze":
        from .analysis import analyze_campaign
        from .config import CampaignPolicy
        policy = CampaignPolicy.model_validate_json(Path(args.policy).read_text(encoding="utf-8"))
        with Store(args.store) as store:
            campaign = analyze_campaign(store, campaign, policy, args.baseline)
    return write_reports(campaign, args.output)
