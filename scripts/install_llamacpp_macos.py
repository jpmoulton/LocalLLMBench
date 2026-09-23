"""Install the pinned llama.cpp b11011 macOS arm64 release for the metal-native runtime. No network by default.

    python scripts/install_llamacpp_macos.py --archive ~/Downloads/llama-b11011-bin-macos-arm64.tar.gz
    python scripts/install_llamacpp_macos.py --download        # explicitly allow fetching the pinned asset

The archive (a local copy, or with --download the pinned GitHub release asset, saved under <output>/downloads) must
have exactly the pinned size and SHA-256 BEFORE anything is extracted. Extraction is refused for absolute or `..`
members, members outside the release's top directory, links that leave it, hard links, devices and duplicates, and
happens in a private staging directory that is promoted to <output>/llama-b11011/ only after every check passed:
the extracted tree equals the archive, com.apple.quarantine is removed (`xattr -d`, absence ignored), and the
executable's own `llama-server --version` reports build 11011, commit aa39d7a3e. It then records
<output>/llama-b11011/install-manifest.json (asset, sha256, size, tag, commit, the upstream build flags, the
release workflow hash, the compiler line and every file's sha256), which `llmbench prepare --runtime metal-native`
checks the directory against. An existing install directory is reused only when its files match the archive
exactly; anything else is refused, never overwritten.

Running the downloaded executable (even as --version) is native execution: runtime-policy.json must set
allow_native_execution. Next step:

    llmbench prepare --runtime metal-native \\
        --llama-server artifacts/native-runtime/llama-b11011/llama-server --output artifacts/native-prep

Exit codes: 0 installed (or already installed), 2 refused (policy, archive, unsafe member, existing directory
differs), 3 the installed executable did not report the pinned build.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmbench.config import RunMode  # noqa: E402
from llmbench.containers.capabilities import parse_version  # noqa: E402
from llmbench.containers.native_prep import INSTALL_MANIFEST, NativeProcessExecutor  # noqa: E402
from llmbench.safety import OperationForbidden, SessionLock  # noqa: E402

# The pin. Values are the release asset as published and the upstream job that built it; changing any of them is a
# new runtime, not an upgrade in place.
TAG = "b11011"
ASSET = "llama-b11011-bin-macos-arm64.tar.gz"
URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{ASSET}"
SHA256 = "9f88854d8216454a883f6d970e52a888d85c1ff321086c08c3d2f69362e0154d"
SIZE = 11156605
COMMIT = "aa39d7a3e145a88202793a89462d65e94a5fc25f"
BUILD_INFO = "b11011-aa39d7a3e"
TOP_DIRECTORY = "llama-b11011"
# release.yml, macos-cpu arm64 job at COMMIT, runner macos-26. Recorded as cmake receives them (no shell quoting);
# UPSTREAM_BUILD_COMMANDS keeps the commands as upstream wrote them.
UPSTREAM_BUILD_FLAGS = ("-DGGML_METAL_EMBED_LIBRARY=ON", "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3",
                        "-DCMAKE_INSTALL_RPATH=@loader_path", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                        "-DLLAMA_FATAL_WARNINGS=ON", "-DLLAMA_BUILD_BORINGSSL=ON", "-DLLAMA_BUILD_EXAMPLES=OFF",
                        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON",
                        "-DGGML_RPC=ON")
UPSTREAM_BUILD_COMMANDS = (
    "cmake -B build -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 "
    "-DCMAKE_INSTALL_RPATH='@loader_path' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DLLAMA_FATAL_WARNINGS=ON "
    "-DLLAMA_BUILD_BORINGSSL=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON "
    "-DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON",
    "cmake --build build --config Release")
RELEASE_WORKFLOW = ".github/workflows/release.yml"
RELEASE_WORKFLOW_JOB = "macos-cpu (arm64)"
RELEASE_WORKFLOW_SHA256 = "e130ce78c37628a8adea0ce58e6de904ba0b6e5bdb85f8417374351c1f87583b"
RUNNER = "macos-26"
DEFAULT_OUTPUT = "artifacts/native-runtime"
QUARANTINE = "com.apple.quarantine"
MAX_MEMBERS = 1024
MAX_UNPACKED_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
VERSION_TIMEOUT_SECONDS = 60
# Per socket operation, and for the whole transfer: a server that trickles bytes must not hold the install open
# indefinitely while every single read stays inside the per-read timeout. 11 MB in 30 minutes is ~6 KB/s.
DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_DEADLINE_SECONDS = 1800
VERSION_LINE = re.compile(r"version: \S+ \(build \d+, commit [0-9a-f]+\)")  # capabilities.parse_version's match


@dataclass(frozen=True)
class ReleasePin:
    """Everything the installer checks, as one value, so a test can pin a tiny synthetic release instead."""

    tag: str
    asset: str
    url: str
    sha256: str
    size: int
    commit: str
    build_info: str
    top_directory: str
    upstream_build_flags: tuple[str, ...]
    upstream_build_commands: tuple[str, ...]
    release_workflow: str
    release_workflow_job: str
    release_workflow_sha256: str
    runner: str
    executable: str = "llama-server"


PINNED = ReleasePin(tag=TAG, asset=ASSET, url=URL, sha256=SHA256, size=SIZE, commit=COMMIT, build_info=BUILD_INFO,
                    top_directory=TOP_DIRECTORY, upstream_build_flags=UPSTREAM_BUILD_FLAGS,
                    upstream_build_commands=UPSTREAM_BUILD_COMMANDS, release_workflow=RELEASE_WORKFLOW,
                    release_workflow_job=RELEASE_WORKFLOW_JOB, release_workflow_sha256=RELEASE_WORKFLOW_SHA256,
                    runner=RUNNER)


class VersionCheckFailed(Exception):
    """The installed executable ran but did not report the pinned build (exit 3, not a refusal of the input)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_archive(data: bytes, pin: ReleasePin) -> None:
    """Size first (cheap, and it bounds what was read), then SHA-256. Nothing is extracted before both pass."""
    if len(data) != pin.size:
        shown = f"more than {pin.size}" if len(data) > pin.size else str(len(data))
        raise ValueError(f"the archive has {shown} bytes; the pinned {pin.asset} has {pin.size}")
    digest = _sha256(data)
    if digest != pin.sha256:
        raise ValueError(f"the archive SHA-256 is {digest}; the pinned {pin.asset} is {pin.sha256}")


def read_archive(path: Path, pin: ReleasePin) -> bytes:
    """Read at most one byte more than the pinned size, into memory: the bytes verified are the bytes extracted,
    so the file cannot change between the check and the use."""
    if not stat.S_ISREG(os.stat(path).st_mode):
        raise ValueError(f"{path} is not a regular file")
    with path.open("rb") as handle:
        data = handle.read(pin.size + 1)
    verify_archive(data, pin)
    return data


def fetch_url(url: str, max_bytes: int, *, timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
              deadline_seconds: float = DOWNLOAD_DEADLINE_SECONDS, clock: Callable[[], float] = time.monotonic,
              opener=None) -> bytes:
    """HTTPS download bounded to `max_bytes` (one byte more is an error), `timeout` per socket operation and
    `deadline_seconds` in total. Only ever called for --download. `opener` (urlopen-like) is for tests."""
    import urllib.request
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-HTTPS download URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "llmbench-install-llamacpp"})
    chunks, total, deadline = [], 0, clock() + deadline_seconds
    with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:  # noqa: S310 - https above
        if not str(response.geturl()).startswith("https://"):
            raise ValueError(f"the download was redirected off HTTPS: {response.geturl()}")
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > max_bytes:
                raise ValueError(f"the download exceeds the pinned {max_bytes} bytes")
            if clock() > deadline:
                raise ValueError(f"the download did not finish within {deadline_seconds:g} s ({total} bytes read)")
            chunks.append(block)
    return b"".join(chunks)


def obtain_download(pin: ReleasePin, output: Path, fetch: Callable[[str, int], bytes]) -> tuple[bytes, Path]:
    """--download: reuse <output>/downloads/<asset> when it verifies, otherwise fetch, verify, then save it."""
    path = output / "downloads" / pin.asset
    if path.exists() or path.is_symlink():
        return read_archive(path, pin), path  # a present file that fails verification is refused, not replaced
    data = fetch(pin.url, pin.size)
    verify_archive(data, pin)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, data)
    return data, path


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(name: str, pin: ReleasePin) -> tuple[str, ...]:
    if not name or "\\" in name or any(ord(char) < 32 for char in name):
        raise ValueError(f"archive member {name!r} has an unsafe name")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive member {name!r} is absolute or climbs out with `..`")
    if not path.parts or path.parts[0] != pin.top_directory:
        raise ValueError(f"archive member {name!r} is outside the release directory {pin.top_directory}/")
    return path.parts


def validated_members(archive: tarfile.TarFile, pin: ReleasePin) -> list[tarfile.TarInfo]:
    """Every member, checked before anything is written; any unsafe member refuses the whole archive.

    Allowed: the top directory, directories, regular files and relative links whose target has no `..` (so it
    stays inside the release). Refused: absolute or `..` names, names outside the top directory, duplicates,
    anything beneath a link, hard links, devices and FIFOs, set-id bits, and more than MAX_MEMBERS members or
    MAX_UNPACKED_BYTES bytes.
    """
    members, seen, links, total = [], set(), set(), 0
    for member in archive:
        if len(members) >= MAX_MEMBERS:
            raise ValueError(f"the archive has more than {MAX_MEMBERS} members")
        parts = _relative(member.name, pin)
        key = "/".join(parts)
        if key in seen:
            raise ValueError(f"archive member {key} appears twice")
        seen.add(key)
        if member.mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise ValueError(f"archive member {key} carries set-id or sticky bits")
        if len(parts) == 1 and not member.isdir():
            raise ValueError(f"the release's top entry {key} must be a directory")
        if member.issym():
            target = PurePosixPath(member.linkname)
            if (not member.linkname or "\\" in member.linkname or target.is_absolute() or ".." in target.parts
                    or any(ord(char) < 32 for char in member.linkname)):
                raise ValueError(f"archive link {key} -> {member.linkname!r} leaves the release directory")
            links.add(key)
        elif member.isfile():
            total += member.size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError(f"the archive unpacks to more than {MAX_UNPACKED_BYTES} bytes")
        elif not member.isdir():
            raise ValueError(f"archive member {key} is a hard link, device or FIFO; only files, directories and "
                             "links are installed")
        members.append(member)
    for member in members:
        parts = PurePosixPath(member.name).parts
        for depth in range(1, len(parts)):
            if "/".join(parts[:depth]) in links:
                raise ValueError(f"archive member {member.name} lies beneath a link")
    executable = f"{pin.top_directory}/{pin.executable}"
    if not any("/".join(PurePosixPath(item.name).parts) == executable and item.isfile() for item in members):
        raise ValueError(f"the archive has no regular file {executable}")
    return members


def archive_contents(archive: tarfile.TarFile, members: list[tarfile.TarInfo], pin: ReleasePin) -> dict:
    """{files: {relpath: sha256}, links: {relpath: target}, directories: [...]} relative to the top directory:
    what a correct extraction must reproduce exactly."""
    files, links, directories = {}, {}, set()
    for member in members:
        parts = PurePosixPath(member.name).parts[1:]
        if not parts:
            continue
        directories.update("/".join(parts[:depth]) for depth in range(1, len(parts)))
        relative = "/".join(parts)
        if member.isdir():
            directories.add(relative)
        elif member.issym():
            links[relative] = member.linkname
        else:
            files[relative] = _sha256(archive.extractfile(member).read())
    return {"files": files, "links": links, "directories": sorted(directories)}


def tree_contents(directory: Path) -> dict:
    """The same shape as `archive_contents`, read from disk without following links; the install manifest at
    the top level is the installer's own addition and is left out."""
    files, links, directories = {}, {}, []
    for root, dirnames, filenames in os.walk(directory):
        base = Path(root)
        for name in sorted(dirnames + filenames):
            path = base / name
            relative = path.relative_to(directory).as_posix()
            if relative == INSTALL_MANIFEST:
                continue
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                links[relative] = os.readlink(path)
            elif stat.S_ISDIR(mode):
                directories.append(relative)
            elif stat.S_ISREG(mode):
                files[relative] = _sha256(path.read_bytes())
            else:
                raise ValueError(f"{path} is not a regular file, directory or link")
    return {"files": files, "links": links, "directories": sorted(directories)}


def remove_quarantine(directory: Path, *, runner=subprocess.run) -> list[str]:
    """`xattr -d com.apple.quarantine` on every regular file, so Gatekeeper does not refuse to start an archive
    that a browser downloaded. A file without the attribute is fine; any other xattr failure refuses."""
    removed = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        done = runner(["xattr", "-d", QUARANTINE, str(path)], capture_output=True, timeout=10, check=False)
        if done.returncode == 0:
            removed.append(path.relative_to(directory).as_posix())
        elif b"No such xattr" not in (done.stderr or b""):
            raise ValueError(f"xattr -d {QUARANTINE} {path} failed (rc={done.returncode}): "
                             + (done.stderr or b"").decode("utf-8", "replace")[-500:])
    return removed


def check_version(executable: Path, pin: ReleasePin, *, executor) -> dict:
    """The executable's own `--version`, bounded, must name the pinned build and commit."""
    result = executor.run((str(executable), "--version"), timeout_seconds=VERSION_TIMEOUT_SECONDS,
                          max_output_bytes=1024 * 1024)
    text = (result.stdout + result.stderr).decode("utf-8", "replace")
    if result.status != "completed" or result.returncode != 0:
        raise VersionCheckFailed(f"{executable} --version failed ({result.status}, rc={result.returncode}): "
                                 + text[-500:])
    try:
        version = parse_version(text)
    except ValueError as exc:
        raise VersionCheckFailed(f"{executable} --version printed no llama.cpp version line") from exc
    if version["build_info"] != pin.build_info or len(version["commit"]) < 7 or not pin.commit.startswith(
            version["commit"]):
        raise VersionCheckFailed(f"{executable} reports build {version['build']}, commit {version['commit']}; the "
                                 f"pin is {pin.build_info} (commit {pin.commit})")
    # The recorded line is the one parse_version matched (it searches, so the line need not START with
    # "version:"); a prefixed line must not turn a verified build into an unhandled StopIteration.
    lines = [line.strip() for line in text.splitlines()]
    matched = next(line for line in lines if VERSION_LINE.search(line))
    return {"build_info": version["build_info"], "version": matched,
            "compiler": next((line for line in lines if line.startswith("built with")), None)}


def build_manifest(pin: ReleasePin, contents: dict, version: dict) -> dict:
    """Deterministic (no timestamps), so re-running an identical install compares equal to the recorded one."""
    return {"schema_version": 1, "asset": pin.asset, "url": pin.url, "sha256": pin.sha256, "size": pin.size,
            "tag": pin.tag, "commit": pin.commit, "build_info": pin.build_info,
            "upstream_build_flags": list(pin.upstream_build_flags),
            "upstream_build": {"workflow": pin.release_workflow, "job": pin.release_workflow_job,
                               "runner": pin.runner, "commit": pin.commit,
                               "commands": list(pin.upstream_build_commands)},
            "release_workflow_sha256": pin.release_workflow_sha256,
            "version": version["version"], "compiler": version["compiler"],
            "files": contents["files"], "links": contents["links"]}


def _record_manifest(root: Path, manifest: dict) -> Path:
    path = root / INSTALL_MANIFEST
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{path} is not a regular file")
        with path.open("rb") as handle:
            data = handle.read(MAX_MANIFEST_BYTES + 1)
        try:
            existing = json.loads(data.decode("utf-8")) if len(data) <= MAX_MANIFEST_BYTES else None
        except ValueError:
            existing = None
        if existing != manifest:
            raise ValueError(f"{path} already records a different install; remove the directory or choose "
                             "another --output")
        return path
    _atomic_write(path, (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return path


def _finish(root: Path, pin: ReleasePin, contents: dict, *, executor_factory, xattr_runner, darwin: bool):
    removed = remove_quarantine(root, runner=xattr_runner) if darwin else []
    executable = root / pin.executable
    version = check_version(executable, pin, executor=executor_factory(executable, root))
    manifest = build_manifest(pin, contents, version)
    return _record_manifest(root, manifest), version, removed


def install(archive_bytes: bytes, output: Path, *, pin: ReleasePin, executor_factory, xattr_runner=subprocess.run,
            darwin: bool = True) -> dict:
    """Install verified archive bytes into `<output>/<top_directory>/` (see the module docstring for the rules)."""
    output.mkdir(parents=True, exist_ok=True)
    target = output / pin.top_directory
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = validated_members(archive, pin)
        contents = archive_contents(archive, members, pin)
        reused = target.exists() or target.is_symlink()
        if reused:
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f"{target} exists and is not a directory")
            if tree_contents(target) != contents:
                raise ValueError(f"{target} exists and differs from the pinned {pin.asset}; remove it or choose "
                                 "another --output (an install is never overwritten)")
            manifest_path, version, removed = _finish(target, pin, contents, executor_factory=executor_factory,
                                                      xattr_runner=xattr_runner, darwin=darwin)
        else:
            staging = Path(tempfile.mkdtemp(prefix=".install-", dir=output))
            try:
                # The members are already validated; the stdlib `data` filter (3.11.4+) is defence in depth.
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(staging, members=members, filter="data")
                else:  # pragma: no cover - Python < 3.11.4
                    archive.extractall(staging, members=members)
                staged = staging / pin.top_directory
                if tree_contents(staged) != contents:
                    raise ValueError("the extracted tree differs from the verified archive")
                _, version, removed = _finish(staged, pin, contents, executor_factory=executor_factory,
                                              xattr_runner=xattr_runner, darwin=darwin)
                try:
                    # Atomic. rename(2) fails if a non-empty target appeared meanwhile; an EMPTY directory that
                    # appeared is replaced, which loses nothing.
                    os.rename(staged, target)
                except OSError as exc:
                    raise ValueError(f"{target} appeared during the install; nothing was overwritten") from exc
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            manifest_path = target / INSTALL_MANIFEST
    executable = target / pin.executable
    return {"install_dir": str(target), "llama_server": str(executable), "build_info": version["build_info"],
            "version": version["version"], "compiler": version["compiler"], "manifest": str(manifest_path),
            "reused_existing": reused, "quarantine_removed": removed,
            "next": f"llmbench prepare --runtime metal-native --llama-server {executable} --output artifacts/native-prep"}


def main(argv=None, *, pin: ReleasePin = PINNED, fetch: Callable[[str, int], bytes] = fetch_url,
         executor_factory=None, xattr_runner=subprocess.run, host_platform: tuple[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", help=f"local copy of {pin.asset}; no network is used")
    source.add_argument("--download", action="store_true",
                        help=f"explicitly allow fetching {pin.url} (saved under <output>/downloads)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"install root (default {DEFAULT_OUTPUT})")
    parser.add_argument("--policy", default="runtime-policy.json",
                        help="runtime policy; running the executable needs allow_native_execution")
    args = parser.parse_args(argv)
    try:
        system, machine = host_platform or (platform.system(), platform.machine())
        if (system, machine) != ("Darwin", "arm64"):
            raise ValueError(f"{pin.asset} runs on macOS arm64 only; this host is {system}/{machine}, where the "
                             "install could not be verified")
        lock = SessionLock.read(args.policy)
        lock.check("native", RunMode.LIVE)
        output = Path(args.output).resolve()
        if args.archive:
            archive_path = Path(args.archive).resolve()
            data = read_archive(archive_path, pin)
        else:
            data, archive_path = obtain_download(pin, output, fetch)
        factory = executor_factory or (lambda executable, cwd: NativeProcessExecutor(
            executable, session_lock=lock, cwd=cwd))
        summary = install(data, output, pin=pin, executor_factory=factory, xattr_runner=xattr_runner,
                          darwin=system == "Darwin")
    except VersionCheckFailed as exc:
        print(f"install_llamacpp_macos: {exc}", file=sys.stderr)
        return 3
    except (ValueError, OSError, OperationForbidden, tarfile.TarError) as exc:
        print(f"install_llamacpp_macos: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({**summary, "archive": str(archive_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
