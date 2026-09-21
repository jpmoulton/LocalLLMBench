"""Derive the hash-pinned Linux lock for the evaluator image from requirements.lock.txt.

Pure parts: `derive_pins` (drops host-only packages) and `hashed_lock_text` (turns downloaded wheels into
`name==version --hash=sha256:<hex>` lines, one per pin). The only side effect is `pip download` of the exact
pinned versions as binary wheels into a scratch directory, through the injectable `run`.

Platforms: pip does not expand a PEP 600 tag given on the command line (only the legacy `manylinux2014` and
`manylinux2010` aliases expand), so every platform the bookworm (glibc 2.36) base can run is passed as its own
`--platform`, newest first: manylinux_2_28, manylinux_2_17, manylinux2014 (which pip widens to 2010/1). The bare
`linux_x86_64` tag is deliberately absent (unvetted builds). `--only-binary=:all:` stays: a pin without any wheel
fails closed rather than being built from source.

Usage (network required for the download step):
    python scripts/lock_evaluator_requirements.py --source requirements.lock.txt \
        --output src/llmbench/containers/evaluator_image/requirements.linux.lock
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HOST_ONLY = frozenset({"lmstudio", "pytest", "ruff", "debugpy"})
PLATFORMS = ("manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64")  # preference order
PYTHON_VERSION = "3.12"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s#]+)\s*$")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def derive_pins(text: str) -> list[tuple[str, str]]:
    """Exact (name, version) pins from the host lock, excluding host-only packages; anything else is an error."""
    pins = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PIN.match(stripped)
        if match is None:
            raise ValueError(f"line {number} is not an exact `name==version` pin: {stripped!r}")
        if normalize(match[1]) in HOST_ONLY:
            continue
        pins.append((match[1], match[2]))
    if not pins:
        raise ValueError("the source lock holds no pins")
    return pins


def download_argv(python: str, requirements: Path, destination: Path) -> list[str]:
    platforms = [item for platform in PLATFORMS for item in ("--platform", platform)]
    return [python, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", *platforms,
            "--python-version", PYTHON_VERSION, "--implementation", "cp", "--dest", str(destination),
            "--requirement", str(requirements)]


def hashed_lock_text(pins: list[tuple[str, str]], wheel_dir: Path, *, source_sha256: str) -> str:
    """One hash-pinned line per pin. A pin without exactly one downloaded wheel fails closed."""
    files = {}
    for path in sorted(wheel_dir.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".whl":
            raise ValueError(f"unexpected download entry: {path.name}")
        parts = path.name[:-4].split("-")
        if len(parts) < 5:
            raise ValueError(f"not a wheel filename: {path.name}")
        files.setdefault((normalize(parts[0]), parts[1]), []).append(path)
    lines = ["# Hash-pinned Linux lock for the llmbench evaluator image.",
             f"# Derived from requirements.lock.txt (sha256 {source_sha256}) minus host-only packages "
             f"({', '.join(sorted(HOST_ONLY))}).",
             f"# Wheels: binary only, platforms {', '.join(PLATFORMS)} (or any), CPython {PYTHON_VERSION}, no deps.", ""]
    for name, version in pins:
        found = files.pop((normalize(name), version), [])
        if len(found) != 1:
            raise ValueError(f"{name}=={version}: expected exactly one downloaded wheel, found {len(found)}")
        lines.append(f"{name}=={version} --hash=sha256:{hashlib.sha256(found[0].read_bytes()).hexdigest()}")
    if files:
        raise ValueError("downloaded wheels without a pin: " + ", ".join(f"{n}=={v}" for n, v in sorted(files)))
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_lock(source: Path, output: Path, *, python: str = sys.executable, run=subprocess.run,
               scratch: Path | None = None) -> str:
    text = source.read_text(encoding="utf-8")
    pins = derive_pins(text)
    with tempfile.TemporaryDirectory(prefix="llmbench-lock-", dir=scratch) as directory:
        work = Path(directory)
        requirements, wheels = work / "pins.txt", work / "wheels"
        wheels.mkdir()
        requirements.write_text("".join(f"{name}=={version}\n" for name, version in pins), encoding="utf-8")
        done = run(download_argv(python, requirements, wheels), capture_output=True, timeout=1800)
        if done.returncode != 0:
            raise ValueError("pip download failed: " + (done.stdout + done.stderr).decode("utf-8", "replace")[-2000:])
        lock = hashed_lock_text(pins, wheels, source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    write_atomic(output, lock)
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="requirements.lock.txt")
    parser.add_argument("--output", default="src/llmbench/containers/evaluator_image/requirements.linux.lock")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        lock = build_lock(Path(args.source), Path(args.output), python=args.python)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"lock_evaluator_requirements: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}: {sum(1 for line in lock.splitlines() if '--hash=' in line)} hashed pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
