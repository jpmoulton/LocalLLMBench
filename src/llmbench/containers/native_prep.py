"""Native (`metal-native`) preparation: pin a llama-server executable on this Mac and record what it is.

The native analogue of `image_plan.prepare_images`. Nothing here downloads, builds or serves a model.
`prepare_native` hashes the executable and every llama.cpp/ggml library (and any Metal shader library a
non-embedded build loads) beside it, proves from the Mach-O load commands that those hashed files are the only
non-system code dyld will load for it, runs the executable ONLY as `--version`, `--help` and `--list-devices`
through `NativeProcessExecutor`, checks that every flag the harness can emit exists in that build, and writes
`native-bundle.json` last and exclusively. Every refusal is a `ValueError`
(or `OperationForbidden` from the policy) naming what failed; nothing is guessed or defaulted to make a bundle
appear. Importing this module performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import posixpath
import re
import stat
import struct
import subprocess
import tempfile
import time
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable, Mapping

from ..coding.image_plan import WORKER_PLATFORMS, verified_local_image_id
from ..coding.sandbox import BoundedProcessExecutor
from ..config import RunMode, canonical_json
from ..store import utc_now
from .capabilities import (ALLOWED_FLAGS, HELP_FILE, VERSION_FILE, ServerCapabilities, check_argv, help_sha256,
                           load_capabilities, parse_server_help)
from .config import IMAGE_ID, ContainerRunConfig, ImageRef, NativeBundle, NativeServerRef
from .image_plan import BUNDLE_NAME as NVIDIA_BUNDLE_NAME
from .plan import build_server_argv

NATIVE_BUNDLE_NAME = "native-bundle.json"
EVIDENCE_NAME = "native-prepare.json"
DEVICES_FILE = "llama-server-devices.txt"
INSTALL_MANIFEST = "install-manifest.json"
BASE_CANDIDATE = Path(__file__).with_name("data") / "base-candidate.json"
CAPTURE_FLAGS = ("--version", "--help", "--list-devices")
PROBE_TIMEOUT_SECONDS = 120  # --list-devices initialises Metal; a cold first run on a busy Mac is slow, not hung.
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_LIBRARIES = 512
# A build WITHOUT GGML_METAL_EMBED_LIBRARY keeps its GPU kernels outside every dylib: ggml-metal loads
# `default.metallib` from the executable's directory, or compiles `ggml-metal.metal` found there (with the two
# headers the build copies beside it). Those files are code the Metal backend runs, so they are pinned exactly like
# a library; replacing one after prepare would otherwise change the kernels under an unchanged pin. The pinned
# b11011 release embeds its shader library, so none of these names exists beside it and its pin is unchanged.
METAL_RESOURCE_PATTERNS = ("*.metallib", "*.metal", "ggml-common.h", "ggml-metal-impl.h")
# `lib*.so` covers a GGML_BACKEND_DL build, whose backends are loadable modules named like that even on macOS.
LIBRARY_PATTERNS = ("lib*.dylib", "lib*.so") + METAL_RESOURCE_PATTERNS
# Environment that could change what the executable does behind its argv: llama.cpp reads LLAMA_ARG_* for every
# flag and GGML_* for backend switches, HF_* would point it at a hub, DYLD_* would load libraries the pin does not
# cover (DYLD_INSERT_LIBRARIES, DYLD_LIBRARY_PATH, DYLD_FRAMEWORK_PATH, ...), MTL_*/METAL_* switch on Metal debug
# layers and shader validation, and a proxy must never sit in front of anything. Everything else is inherited.
SCRUBBED_PREFIXES = ("LLAMA_", "GGML_", "HF_", "HUGGINGFACE_", "DYLD_", "MTL_", "METAL_")

# Mach-O constants (<mach-o/loader.h>, <mach-o/fat.h>). Only what the linkage proof reads.
MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC, FAT_MAGIC_64 = 0xCAFEBABE, 0xCAFEBABF
CPU_TYPE_ARM64 = 0x0100000C
# arm64 and arm64e share CPU_TYPE_ARM64 and differ only in the subtype, whose top byte carries capability bits
# (arm64e's pointer-authentication ABI version), not the subtype itself.
CPU_SUBTYPE_MASK = 0xFF000000
ARM64_SUBTYPES = {0: "arm64", 1: "arm64", 2: "arm64e"}  # CPU_SUBTYPE_ARM64_ALL, _ARM64_V8, _ARM64E
LC_ID_DYLIB, LC_RPATH = 0xD, 0x8000001C
# LC_LOAD_DYLINKER names the loader the kernel starts for the executable; LC_DYLD_ENVIRONMENT embeds DYLD_*_PATH
# variables dyld honours for a main executable, i.e. a library search path baked into the bytes. Either one could
# load code the pin does not cover, so both are read and held to `DYLD` / refused.
LC_LOAD_DYLINKER, LC_DYLD_ENVIRONMENT = 0xE, 0x27
DYLD = "/usr/lib/dyld"
# LC_LOAD_DYLIB, LC_LAZY_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_LOAD_UPWARD_DYLIB: every command
# that makes dyld load another image. A weak one is treated like a strong one: fail closed.
DYLIB_LOAD_COMMANDS = frozenset({0xC, 0x20, 0x80000018, 0x8000001F, 0x80000023})
MACHO_FILETYPES = {2: "execute", 6: "dylib", 8: "bundle"}
MAX_LOAD_COMMAND_BYTES = 1024 * 1024
MAX_LOAD_COMMANDS = 4096
MAX_FAT_ARCHS = 16
# The operating system's own libraries live on the sealed, signed system volume (or the dyld shared cache behind
# these paths); anything else a llama-server loads must be one of the hashed files beside it.
SYSTEM_LIBRARY_PREFIXES = ("/usr/lib/", "/System/Library/")
LOCAL_RPATHS = frozenset({"@loader_path", "@executable_path"})
LOCAL_PREFIXES = ("@rpath/", "@loader_path/", "@executable_path/")


def native_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a native llama-server is started with: the caller's, minus `SCRUBBED_PREFIXES` and every
    `*_proxy`/`*_PROXY`. Exposed so the native runner and the installer scrub exactly as preparation did."""
    source = os.environ if environment is None else environment
    return {key: value for key, value in source.items()
            if not key.upper().startswith(SCRUBBED_PREFIXES) and not key.upper().endswith("_PROXY")}


def _open_regular(path: Path) -> int:
    """Open without following a final link and without blocking on a FIFO, then insist on a regular file, so a
    swapped-in pipe or device can neither hang the hash nor be hashed."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{path} is not a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with os.fdopen(_open_regular(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_native_server(executable: str | Path) -> dict:
    """{executable_sha256, libraries: {name: sha256}, libraries_sha256} for a llama-server and its directory.

    `libraries` has one entry per `lib*.dylib` / `lib*.so` NAME in the executable's directory, and per Metal shader
    library or source a non-embedded build loads from there (`METAL_RESOURCE_PATTERNS`), valued with the
    SHA-256 of the regular file that name is or links to. A link must resolve to a regular file in that same
    directory, anything else is refused: dyld loads `@rpath/libggml-metal.0.dylib` by its link name, so pinning
    names rather than only the regular files means repointing a link at another (hashed) file still changes
    `libraries_sha256`. That digest is `sha256(canonical_json(libraries))`. The executable itself must be a regular
    file named by an absolute path, never a link that could be repointed after the pin.
    """
    path = Path(executable)
    if not path.is_absolute():
        raise ValueError(f"the native executable must be an absolute path: {path}")
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"the native executable must be a regular file, not a link: {path}")
    directory = path.parent.resolve(strict=True)
    executable_sha256 = _sha256_file(directory / path.name)
    names = sorted(name for name in os.listdir(directory)
                   if any(fnmatchcase(name, pattern) for pattern in LIBRARY_PATTERNS))
    if len(names) > MAX_LIBRARIES:
        raise ValueError(f"{directory} holds {len(names)} libraries; at most {MAX_LIBRARIES} are pinned")
    digests: dict[Path, str] = {}
    libraries: dict[str, str] = {}
    for name in names:
        candidate = directory / name
        mode = os.lstat(candidate).st_mode
        if stat.S_ISLNK(mode):
            target = candidate.resolve(strict=True)
            if target.parent != directory:
                raise ValueError(f"library link {name} resolves outside {directory}: {target}")
        elif stat.S_ISREG(mode):
            target = candidate
        else:
            raise ValueError(f"{candidate} is neither a regular file nor a link to one")
        if target not in digests:
            digests[target] = _sha256_file(target)
        libraries[name] = digests[target]
    return {"executable_sha256": executable_sha256, "libraries": libraries,
            "libraries_sha256": hashlib.sha256(canonical_json(libraries).encode("utf-8")).hexdigest()}


def _lc_string(body: bytes, where: str) -> str:
    """The `lc_str` of a dylib or rpath load command: an offset at byte 8, a NUL-terminated string inside the
    command."""
    if len(body) < 12:
        raise ValueError(f"{where}: truncated load command")
    (offset,) = struct.unpack_from("<I", body, 8)
    if not 12 <= offset < len(body):
        raise ValueError(f"{where}: load command string offset {offset} is outside the command")
    raw = body[offset:].split(b"\x00", 1)[0]
    if not raw:
        raise ValueError(f"{where}: empty load command string")
    return raw.decode("utf-8")  # UnicodeDecodeError is a ValueError: refused like any other malformed image


def _arm64_flavour(subtype: int, where: str) -> str:
    """"arm64" or "arm64e" for a CPU_TYPE_ARM64 subtype; any other subtype is refused rather than guessed at."""
    flavour = ARM64_SUBTYPES.get(subtype & 0xFFFFFFFF & ~CPU_SUBTYPE_MASK)
    if flavour is None:
        raise ValueError(f"{where}: an arm64 image with the unknown CPU subtype {subtype & 0xFFFFFFFF:#x}")
    return flavour


def _arm64_slice(table: bytes, count: int, wide: bool, where: str) -> tuple[str, int]:
    """The (flavour, offset) of the slice an arm64 process loads from a universal binary's fat_arch table.

    arm64 and arm64e share CPU_TYPE_ARM64, so the first entry of that type is not necessarily the one that runs:
    the kernel runs a third-party executable's plain arm64 slice (arm64e only under the developer-only arm64e
    preview ABI), and dyld loads each library's arm64 slice into that process. So the plain arm64 slice is chosen
    wherever it sits in the table. An arm64e slice is read only when it is the ONLY arm64-family slice (a platform
    binary such as /bin/ls ships x86_64 + arm64e), and the flavour returned says so. Two plain arm64 slices, or
    several arm64e ones and no arm64, are refused: which one loads would be a guess.
    """
    size = 32 if wide else 20
    slices = []
    for index in range(count):
        fields = struct.unpack_from(">iiQQII" if wide else ">iiIII", table, index * size)
        if fields[0] == CPU_TYPE_ARM64:
            slices.append((_arm64_flavour(fields[1], where), fields[2]))
    plain = [item for item in slices if item[0] == "arm64"]
    if len(plain) == 1:
        return plain[0]
    if plain:
        raise ValueError(f"{where} is a universal binary with {len(plain)} arm64 slices; which one loads is "
                         "ambiguous")
    if len(slices) == 1:
        return slices[0]
    if slices:
        raise ValueError(f"{where} is a universal binary with {len(slices)} arm64e slices and no arm64 slice")
    raise ValueError(f"{where} is a universal binary without an arm64 slice")


def read_macho_linkage(path: str | Path) -> dict:
    """The arm64 image's dynamic-library load commands, read from the file without running anything.

    Returns {"cpu", "filetype", "install_name", "dylibs": [...], "rpaths": [...], "dylinker", "environment": [...]}
    where "cpu" is the flavour of the image actually read: "arm64", or "arm64e" when that was the only arm64-family
    slice. A universal binary is read at the slice an arm64 process loads (`_arm64_slice`), and that slice's own
    header must agree with the table about its flavour. Anything that is not a 64-bit arm64 Mach-O image, or whose
    load commands are truncated, oversized or malformed, is refused: the linkage proof cannot vouch for what it
    cannot read.
    """
    source = Path(path)
    with os.fdopen(_open_regular(source), "rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            raise ValueError(f"{source.name} is not a Mach-O image")
        offset, flavour = 0, None
        (magic,) = struct.unpack(">I", head[:4])
        if magic in (FAT_MAGIC, FAT_MAGIC_64):
            (count,) = struct.unpack(">I", head[4:8])
            if not 0 < count <= MAX_FAT_ARCHS:
                raise ValueError(f"{source.name}: implausible universal binary with {count} slices")
            wide = magic == FAT_MAGIC_64
            size = 32 if wide else 20
            table = handle.read(count * size)
            if len(table) != count * size:
                raise ValueError(f"{source.name}: truncated universal binary header")
            flavour, offset = _arm64_slice(table, count, wide, source.name)
        handle.seek(offset)
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError(f"{source.name} is not a Mach-O image")
        magic, cputype, subtype, filetype, count, size, _flags, _reserved = struct.unpack("<IiiIIIII", header)
        if magic != MH_MAGIC_64:
            raise ValueError(f"{source.name} is not a 64-bit Mach-O image")
        if cputype != CPU_TYPE_ARM64:
            raise ValueError(f"{source.name} is built for CPU type {cputype:#x}, not arm64")
        cpu = _arm64_flavour(subtype, source.name)
        if flavour is not None and cpu != flavour:
            raise ValueError(f"{source.name}: the universal binary lists an {flavour} slice whose header says {cpu}")
        if count > MAX_LOAD_COMMANDS or size > MAX_LOAD_COMMAND_BYTES:
            raise ValueError(f"{source.name}: {count} load commands in {size} bytes exceeds the reader's bounds")
        commands = handle.read(size)
    if len(commands) != size:
        raise ValueError(f"{source.name}: truncated load commands")
    dylibs, rpaths, environment, install_name, dylinker, position = [], [], [], None, None, 0
    for _ in range(count):
        if position + 8 > size:
            raise ValueError(f"{source.name}: truncated load commands")
        command, length = struct.unpack_from("<II", commands, position)
        if length < 8 or position + length > size:
            raise ValueError(f"{source.name}: load command of {length} bytes overruns its table")
        body = commands[position:position + length]
        if command in DYLIB_LOAD_COMMANDS:
            dylibs.append(_lc_string(body, source.name))
        elif command == LC_ID_DYLIB:
            install_name = _lc_string(body, source.name)
        elif command == LC_RPATH:
            rpaths.append(_lc_string(body, source.name))
        elif command == LC_LOAD_DYLINKER:
            dylinker = _lc_string(body, source.name)
        elif command == LC_DYLD_ENVIRONMENT:
            environment.append(_lc_string(body, source.name))
        position += length
    return {"cpu": cpu, "filetype": MACHO_FILETYPES.get(filetype, filetype), "install_name": install_name,
            "dylibs": dylibs, "rpaths": rpaths, "dylinker": dylinker, "environment": environment}


def check_native_linkage(executable: Path, libraries: Mapping[str, str]) -> dict:
    """Prove that `libraries` (from `hash_native_server`) covers every non-system image dyld loads for it.

    Walks the load commands from the executable through each library it reaches. Every dependency must be an
    operating-system library (`SYSTEM_LIBRARY_PREFIXES`, spelled as a normalised path: `/usr/lib/../../opt/x`
    starts with `/usr/lib/` but is not under it) or `@rpath/`, `@loader_path/`, `@executable_path/` plus the
    plain name of a hashed library beside the executable, and every LC_RPATH must be that directory itself. An
    `@rpath/` dependency also needs such an LC_RPATH in scope (its own image's or the executable's): with none,
    dyld falls back to searching the leaf name in `/usr/local/lib` and `/usr/lib`, outside the pin. The executable
    must be arm64 (not arm64e), use the system `DYLD` and embed no LC_DYLD_ENVIRONMENT search path; each image
    is read at the slice an arm64 process loads (`read_macho_linkage`). A Homebrew-style layout
    (`@loader_path/../lib`, `/opt/homebrew/...`) is refused rather than pinned by the executable alone, because
    the Metal backend it would load is exactly what the pin exists to cover. Code loaded later with `dlopen` (a
    GGML_BACKEND_DL build) is not visible here; such backends beside the executable are still hashed.
    """
    directory = executable.parent
    pending, images, system, problems = [executable.name], {}, set(), []
    executable_rpath = False
    while pending:
        name = pending.pop()
        real = (directory / name).resolve(strict=True).name  # hashed links resolve inside the directory
        if real in images:
            continue
        info = read_macho_linkage(directory / real)
        images[real] = {"filetype": info["filetype"], "dylibs": info["dylibs"], "rpaths": info["rpaths"]}
        for rpath in info["rpaths"]:
            if rpath.rstrip("/") not in LOCAL_RPATHS:
                problems.append(f"{real} searches {rpath!r} for libraries, outside the pinned directory")
        local_rpath = any(rpath.rstrip("/") in LOCAL_RPATHS for rpath in info["rpaths"])
        if real == executable.name:
            executable_rpath = local_rpath
            if info["cpu"] != "arm64":
                # A third-party arm64e executable runs only under the arm64e preview ABI, and then dyld loads each
                # library's arm64e slice, not the arm64 slice this proof reads for a universal library.
                problems.append(f"{real} is an {info['cpu']} executable; the proof covers an arm64 process only")
            if info["dylinker"] not in (None, DYLD):
                problems.append(f"{real} asks the kernel for the loader {info['dylinker']!r}, not {DYLD}")
        for variable in info["environment"]:
            problems.append(f"{real} embeds the dyld environment {variable!r}")
        for dependency in info["dylibs"]:
            if dependency.startswith(SYSTEM_LIBRARY_PREFIXES) and posixpath.normpath(dependency) == dependency:
                system.add(dependency)
                continue
            prefix = next((item for item in LOCAL_PREFIXES if dependency.startswith(item)), None)
            leaf = dependency[len(prefix):] if prefix else None
            if leaf is None or "/" in leaf or leaf not in libraries:
                problems.append(f"{real} loads {dependency}, which is not a hashed library beside the executable")
                continue
            if prefix == "@rpath/" and not (local_rpath or executable_rpath):
                problems.append(f"{real} loads {dependency} with no LC_RPATH in the pinned directory, so dyld "
                                "would search /usr/local/lib and /usr/lib for it")
                continue
            pending.append(leaf)
    if problems:
        raise ValueError("the executable would load code its pin does not cover: " + "; ".join(problems))
    return {"cpu": "arm64", "images": images, "system_libraries": sorted(system)}


class _NativePermission:
    """Gives the inherited bounded runner the NATIVE permission where it checks `container`: running a host
    executable is authorized by `allow_native_execution` alone, never by a container grant."""

    def __init__(self, lock) -> None:
        self.lock = lock

    def check(self, operation: str, mode) -> None:
        self.lock.check("native", mode)


class NativeProcessExecutor(BoundedProcessExecutor):
    """Run one pinned llama-server as `--version`, `--help` or `--list-devices`, and nothing else.

    Inherits the bounded pipe readers, timeout and output cap of the sandbox executor (as `ComposeExecutor`
    does) and replaces only the vocabulary, the permission and the process start: the argv must be exactly
    `(<executable given here>, <one of CAPTURE_FLAGS>)`, the environment is `native_environment()`, the working
    directory is `cwd`, and `session_lock.check("native", mode)` runs before every start. `run` also accepts
    the bare flag. The executable is re-checked to be a regular file (not a link) immediately before it starts.
    """

    def __init__(self, executable: str | Path, *, session_lock, cwd: str | Path, mode: RunMode = RunMode.LIVE,
                 popen_factory=None, clock=time.monotonic, environment: Mapping[str, str] | None = None) -> None:
        path = Path(executable)
        if not path.is_absolute() or any(char in str(path) for char in ("\r", "\n", "\x00")):
            raise ValueError("the native executable must be an absolute path without control characters")
        if session_lock is None:
            raise ValueError("a session lock is required to run a native executable")
        super().__init__(session_lock=_NativePermission(session_lock), mode=mode, popen_factory=self._spawn,
                         clock=clock)
        self.executable, self.cwd, self.native_lock = str(path), Path(cwd), session_lock
        self._process_factory = popen_factory or subprocess.Popen
        self._environment = environment

    @property
    def synthetic(self) -> bool:
        return self._process_factory is not subprocess.Popen

    def _validate(self, argv: tuple[str, ...]) -> None:
        if type(argv) is not tuple or len(argv) != 2 or argv[0] != self.executable or argv[1] not in CAPTURE_FLAGS:
            raise ValueError("the native executor runs only `<pinned llama-server> --version|--help|--list-devices`")

    def _spawn(self, command, **kwargs):
        info = os.lstat(command[0])
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OSError(f"{command[0]} is no longer a regular file")
        return self._process_factory(tuple(command), env=native_environment(self._environment), cwd=str(self.cwd),
                                     **kwargs)

    def run(self, argv: str | tuple[str, ...], *, timeout_seconds: int, max_output_bytes: int = MAX_CAPTURE_BYTES):
        if isinstance(argv, str):
            argv = (self.executable, argv)
        return super().run(argv, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)


_DEVICE_LINE = re.compile(r"^\s*(?:-\s+)?([A-Za-z][A-Za-z0-9_]*)\s*:\s+(.+?)\s+\((\d+) MiB, (\d+) MiB free\)\s*$")


def parse_list_devices(text: str) -> list[dict]:
    """`llama-server --list-devices` rows such as `MTL0: Apple M1 (5461 MiB, 5460 MiB free)`. The MTL total is
    Metal's recommended working-set size, the budget the native runner's watchdog defaults to."""
    return [{"name": match[1], "description": match[2], "total_mib": int(match[3]), "free_mib": int(match[4])}
            for line in text.splitlines() if (match := _DEVICE_LINE.match(line))]


def _text(data: bytes) -> str:
    return data.decode("utf-8", "replace")  # the decoding the runtime identity check hashes, byte for byte


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


def _write_exclusive(path: Path, content: bytes) -> None:
    """Complete-or-absent AND never-overwrite: stage in a temporary file, then hard-link it into place, which
    fails if the name exists (a rename would silently replace a bundle written meanwhile)."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"{path.name} already exists in {path.parent}; use a new output directory") from exc
    finally:
        temporary.unlink(missing_ok=True)


def capture_native_capabilities(executable: str | Path, output_dir: str | Path, *, executor,
                                timeout_seconds: int = PROBE_TIMEOUT_SECONDS,
                                max_output_bytes: int = MAX_CAPTURE_BYTES) -> ServerCapabilities:
    """Run the pinned executable's `--version`, `--help` and `--list-devices` and save what they said.

    `llama-server-help.txt` holds `--help` stdout ONLY (stderr carries Metal initialisation logs that are not part
    of the option table and would change its hash), `llama-server-version.txt` and `llama-server-devices.txt`
    hold stdout plus stderr. Nothing is written unless all three succeeded, the help parses and the devices list
    names a Metal (`MTL*`) device: a build or host without Metal is refused as "no Metal device" rather than
    captured as if it could serve this runtime. Returns the capabilities re-read from the saved files.
    """
    argv0, output = str(executable), Path(output_dir)
    results = {}
    for flag in CAPTURE_FLAGS:
        result = executor.run((argv0, flag), timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
        if result.status != "completed" or result.returncode != 0:
            raise ValueError(f"llama-server {flag} failed ({result.status}, rc={result.returncode}): "
                             + _text(result.stdout + result.stderr)[-1500:])
        results[flag] = result
    version_text = _text(results["--version"].stdout + results["--version"].stderr)
    help_text = _text(results["--help"].stdout)
    devices_text = _text(results["--list-devices"].stdout + results["--list-devices"].stderr)
    devices = parse_list_devices(devices_text)
    if not any(device["name"].startswith("MTL") for device in devices):
        listed = ", ".join(device["name"] for device in devices) or "no devices"
        raise ValueError(f"no Metal device: `llama-server --list-devices` listed {listed}; the metal-native "
                         "runtime needs a llama.cpp build with the Metal backend on an Apple Silicon GPU")
    parse_server_help(help_text, version_text=version_text)  # unparseable output is refused before any write
    for name, text in ((HELP_FILE, help_text), (VERSION_FILE, version_text), (DEVICES_FILE, devices_text)):
        _atomic_write(output / name, text.encode("utf-8"))
    caps = load_capabilities(output)
    if caps.help_sha256 != help_sha256(help_text) or caps.build is None:
        raise ValueError("the saved llama-server capture does not re-read as it was captured")
    return caps


def example_native_config(ref: NativeServerRef) -> ContainerRunConfig:
    """The packaged base candidate moved onto `ref`: what `prepare` checks the build's flags against."""
    raw = json.loads(BASE_CANDIDATE.read_text(encoding="utf-8"))
    raw.pop("inference_image", None)
    raw.update(runtime="metal-native", native_server=ref.model_dump(mode="json"), native_limits={})
    return ContainerRunConfig.model_validate_json(json.dumps(raw))


def check_native_flags(ref: NativeServerRef, caps: ServerCapabilities) -> dict:
    """Refuse a build that lacks any flag the harness can emit, or rejects the argv built for a native config.

    The argv is exactly what the native runner passes (`build_server_argv` with the host model path and
    127.0.0.1; port 0 is a placeholder); the allowlist check covers the flags that argv does not happen to use
    (`--no-kv-offload`, `--swa-full`, `--spec-draft-n-max`, ...), so a later candidate cannot discover one missing
    in the middle of a session.
    """
    config = example_native_config(ref)
    argv = build_server_argv(config, model_path=config.server_model_path, host="127.0.0.1", port=0)
    problems = check_argv(argv, caps)
    problems += [f"{flag} is in the harness allowlist but unknown to this llama-server build"
                 for flag in sorted(ALLOWED_FLAGS - set(caps.flags)) if flag not in argv]
    if problems:
        raise ValueError("this llama-server build cannot run the harness argv: " + "; ".join(problems))
    return {"argv": list(argv), "allowed_flags_checked": len(ALLOWED_FLAGS)}


def _native_executable(executable: str | Path) -> Path:
    path = Path(os.path.abspath(executable))
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"--llama-server must name the executable itself, not a link or a directory: {path}")
    if not os.access(path, os.X_OK):
        raise ValueError(f"{path} is not executable")
    return path.resolve(strict=True)


def read_install_manifest(path: str | Path) -> dict:
    """`install-manifest.json` from `scripts/install_llamacpp_macos.py`, bounded and refused when malformed: a
    present-but-unreadable manifest is not the same thing as no manifest."""
    source = Path(path)
    with os.fdopen(_open_regular(source), "rb") as handle:
        data = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError(f"{source} exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"{source} is not a JSON install manifest: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict) or not isinstance(raw.get("links", {}), dict):
        raise ValueError(f"{source} is not an install manifest (no files map)")
    for key in ("asset", "sha256", "tag", "build_info"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"{source} does not record {key}")
    canonical_json(raw)  # JSON-safe, or refused here rather than inside the bundle
    return raw


def verify_install_manifest(manifest: Mapping[str, Any], executable: Path, hashed: Mapping[str, Any]) -> None:
    """The executable, every hashed library and every library link must be exactly what the install recorded.

    A library that is not in the recorded install (dropped in afterwards) is refused as well as a changed one:
    it would be pinned, but not as part of the release the manifest describes.
    """
    files, links, directory, problems = manifest["files"], manifest.get("links", {}), executable.parent, []
    if files.get(executable.name) != hashed["executable_sha256"]:
        problems.append(f"{executable.name} is not the recorded install's executable")
    for name, digest in hashed["libraries"].items():
        entry, real = directory / name, name
        if entry.is_symlink():
            target = os.readlink(entry)
            if links.get(name) != target:
                problems.append(f"{name} links to {target!r}; the install recorded {links.get(name)!r}")
            real = entry.resolve(strict=True).name
        if real not in files:
            problems.append(f"{real} is not part of the recorded install")
        elif files[real] != digest:
            problems.append(f"{real} differs from the recorded install")
    if problems:
        raise ValueError("the llama-server directory does not match its install-manifest.json: "
                         + "; ".join(sorted(set(problems))))


def _install_build(executable: Path, install_manifest: str | Path | None) -> tuple[Path | None, dict]:
    if install_manifest is not None:
        path = Path(install_manifest)
    else:
        path = executable.parent / INSTALL_MANIFEST
        if not (path.exists() or path.is_symlink()):
            return None, {"provenance": "unrecorded"}
    return path, read_install_manifest(path)


def inspect_worker_image(iidfile: str | Path, *, executor) -> ImageRef:
    """The sandbox worker image named by a verified iidfile, inspected on the daemon the coding sandbox uses.

    The platform is the image's own inspected `Os/Architecture` (linux/arm64 for a worker built for an Apple
    Silicon Colima VM), never the platform the build was asked for, and the entrypoint is its own Config
    entrypoint, exactly as `prepare_images` derives a worker `ImageRef`.
    """
    image_id = verified_local_image_id(iidfile)
    result = executor.run(("docker", "image", "inspect", "--format", "{{json .}}", image_id), timeout_seconds=60,
                          max_output_bytes=MAX_CAPTURE_BYTES)
    if result.status != "completed" or result.returncode != 0:
        raise ValueError(f"worker image {image_id} is not available locally ({result.status}): "
                         + _text(result.stderr)[-500:])
    try:
        raw = json.loads(result.stdout)
        platform = f"{raw['Os']}/{raw['Architecture']}"
        entrypoint = tuple((raw.get("Config") or {}).get("Entrypoint") or ())
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"unreadable worker image inspection: {exc}") from exc
    if raw.get("Id") != image_id:
        raise ValueError(f"worker image inspection returned Id {raw.get('Id')!r}, not {image_id}")
    if platform not in WORKER_PLATFORMS:
        raise ValueError(f"worker image platform {platform} is not one of {', '.join(WORKER_PLATFORMS)}")
    return ImageRef(role="worker", reference=image_id, image_id=image_id, platform=platform, entrypoint=entrypoint,
                    source=f"prepare: verified iidfile {Path(iidfile).name}")


class WorkerInspectExecutor(BoundedProcessExecutor):
    """The coding sandbox's own Docker client (same environment, so the same daemon `DockerWorker` will use),
    narrowed to one read-only shape: `docker image inspect --format {{json .}} <sha256 image id>`. It still
    requires the `container` permission, checked by the inherited runner before every call."""

    @staticmethod
    def _validate(argv: tuple[str, ...]) -> None:
        if (type(argv) is not tuple or len(argv) != 6
                or argv[:5] != ("docker", "image", "inspect", "--format", "{{json .}}")
                or type(argv[5]) is not str or not re.fullmatch(IMAGE_ID, argv[5])):
            raise ValueError("the worker inspection executor runs only `docker image inspect` of a sha256 image ID")


def sandbox_verdict(probe: Mapping[str, Any], worker: ImageRef | None) -> dict:
    """The bundle's sandbox record: the probe's own result, made stricter where the bundle knows more.

    The probe (`coding.sandbox.probe_docker_sandbox`) answers whether Docker can run the pinned worker, and
    already blocks both cases below itself; they are re-asserted here because the probe is injectable and the
    bundle must not depend on every probe getting them right. A sandbox is `blocked` when no worker image was
    prepared at all, and when the worker's platform is not the Docker server's: an emulated worker would make
    every coding case's timeout measure the emulator instead of the answer. An overridden verdict keeps the
    probe's status as `probe_status`. The coding suites read `status` and are recorded as blocked, never run on
    the host, whenever it is not `available`.
    """
    if not isinstance(probe, Mapping) or probe.get("status") not in ("available", "blocked"):
        raise ValueError("the Docker sandbox probe returned no available/blocked status")
    record = dict(probe)
    canonical_json(record)
    if record["status"] == "available":
        server = record.get("server_os_arch")
        if worker is None:
            record.update(status="blocked", probe_status="available",
                          reason="no sandbox worker image was prepared: build one with scripts/build_worker_image.py "
                                 "and pass its worker-image.id with --worker-iidfile")
        elif isinstance(server, str) and server != worker.platform:
            record.update(status="blocked", probe_status="available",
                          reason=f"the worker image is {worker.platform} but the Docker server is {server}; an "
                                 "emulated worker would make coding timeouts measure the emulator. Rebuild it with "
                                 f"scripts/build_worker_image.py --platform {server}")
    return record


class _Recorded:
    """Pass-through that keeps an argv/status/offset log of every native call for `native-prepare.json`."""

    def __init__(self, executor, log: list, clock: Callable[[], float], started: float) -> None:
        self.executor, self.log, self.clock, self.started = executor, log, clock, started

    def run(self, argv, **kwargs):
        result = self.executor.run(argv, **kwargs)
        self.log.append({"argv": list(argv), "status": result.status, "returncode": result.returncode,
                         "offset_seconds": max(0.0, self.clock() - self.started)})
        return result


def _default_host_facts() -> dict:
    from ..apple import host_facts
    return host_facts()


def _default_docker_probe(**kwargs) -> dict:
    from ..coding.sandbox import probe_docker_sandbox
    return probe_docker_sandbox(**kwargs)


def _host_registry_digest() -> str:
    from ..registry import builtin_registry
    return builtin_registry().digest()


def prepare_native(executable: str | Path, output: str | Path, *, session_lock, worker_iidfile: str | Path | None = None,
                   install_manifest: str | Path | None = None, host_facts: Callable[[], Mapping] | None = None,
                   docker_probe: Callable[..., Mapping] | None = None, clock: Callable[[], float] = time.monotonic,
                   executor=None, docker_executor=None, host_platform: tuple[str, str] | None = None,
                   registry_digest: str | None = None,
                   timeout_seconds: int = PROBE_TIMEOUT_SECONDS) -> NativeBundle:
    """Pin one llama-server on this Mac and write `<output>/native-bundle.json`. Any failed check refuses it.

    Order matters and is deliberate: the host must be darwin/arm64 (`host_platform` for tests) and the policy
    must allow `native` before anything else happens; the bundle must not exist yet, and the output must not be
    an NVIDIA preparation whose capture files share these names; the executable and its
    libraries are hashed and their Mach-O linkage proven BEFORE the executable is run; an `install-manifest.json`
    beside the executable (or `install_manifest`) must match those hashes and, after capture, the build the
    executable reports; the executable is then run only as --version/--help/--list-devices (`executor` for
    tests); every allowlisted flag must exist; and the files are hashed AGAIN, so a directory that changed while
    it ran is refused rather than pinned half-old. The worker image (optional) needs the `container`
    permission and is inspected, not assumed. `docker_probe(session_lock=..., image_ref=worker)` defaults to the
    side-effect-free `coding.sandbox.probe_docker_sandbox`; a probe that fails or answers nonsense is recorded as
    a blocked sandbox with its cause (see `sandbox_verdict`). `host_facts()` defaults to `llmbench.apple.host_facts`.
    Evidence of the run goes to `native-prepare.json`; the bundle is written last, atomically and exclusively.
    """
    system, machine = host_platform or (_platform.system(), _platform.machine())
    if (system, machine) != ("Darwin", "arm64"):
        hint = " (an x86_64 Python under Rosetta reports this too; use an arm64 Python)" if system == "Darwin" else ""
        raise ValueError(f"native preparation needs macOS on Apple Silicon (Darwin/arm64); this host is "
                         f"{system}/{machine}{hint}")
    session_lock.check("native", RunMode.LIVE)
    artifacts = Path(output).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle_path = artifacts / NATIVE_BUNDLE_NAME
    if bundle_path.exists() or bundle_path.is_symlink():
        raise ValueError(f"{NATIVE_BUNDLE_NAME} already exists in {artifacts}; use a new output directory")
    nvidia = artifacts / NVIDIA_BUNDLE_NAME
    if nvidia.exists() or nvidia.is_symlink():
        # The container prepare keeps the CUDA image's llama-server-help.txt/-version.txt in the same file names
        # the capture below writes, and ContainerRunner checks them against the NVIDIA configs' help_sha256.
        raise ValueError(f"{artifacts} is an NVIDIA preparation ({NVIDIA_BUNDLE_NAME}); its llama-server capture "
                         "would be overwritten. Use a separate output directory such as artifacts/native-prep")
    started, calls = clock(), []

    # 1. Identity before execution: bytes, linkage, recorded install.
    path = _native_executable(executable)
    hashed = hash_native_server(path)
    linkage = check_native_linkage(path, hashed["libraries"])
    manifest_path, build = _install_build(path, install_manifest)
    if manifest_path is not None:
        verify_install_manifest(build, path, hashed)
    hash_seconds = max(0.0, clock() - started)

    # 2. What the pinned build itself says, then whether it can run everything the harness emits.
    runner = _Recorded(executor or NativeProcessExecutor(path, session_lock=session_lock, cwd=artifacts),
                       calls, clock, started)
    caps = capture_native_capabilities(path, artifacts, executor=runner, timeout_seconds=timeout_seconds)
    if manifest_path is not None and build["build_info"] != caps.build:
        raise ValueError(f"the executable reports build {caps.build}; its install manifest records "
                         f"{build['build_info']}")
    devices = parse_list_devices((artifacts / DEVICES_FILE).read_text(encoding="utf-8"))
    source = (f"llama.cpp {build['tag']} release asset {build['asset']} (sha256 {build['sha256'][:12]})"
              if manifest_path is not None else "local")
    ref = NativeServerRef(executable=str(path), executable_sha256=hashed["executable_sha256"],
                          libraries_sha256=hashed["libraries_sha256"], build_info=caps.build,
                          help_sha256=caps.help_sha256, source=source)
    flags = check_native_flags(ref, caps)
    if hash_native_server(path) != hashed:
        raise ValueError("the executable or its libraries changed while they were being prepared; nothing was pinned")

    # 3. The coding sandbox: an optional worker image, then whether Docker can run it here.
    worker = None
    if worker_iidfile is not None:
        session_lock.check("container", RunMode.LIVE)
        worker = inspect_worker_image(worker_iidfile, executor=docker_executor or WorkerInspectExecutor(
            session_lock=session_lock, mode=RunMode.LIVE))
    try:
        sandbox = sandbox_verdict((docker_probe or _default_docker_probe)(session_lock=session_lock,
                                                                          image_ref=worker), worker)
    except Exception as exc:  # as the native runner does: a failed probe blocks the coding suites, recorded with
        # its cause; it does not unpin a server that was verified above.
        sandbox = {"status": "blocked", "reason": f"sandbox probe failed: {type(exc).__name__}: {exc}"}

    # 4. Stable host facts (never free memory or swap), plus the Metal devices the build itself listed.
    facts = (host_facts or _default_host_facts)()
    if not isinstance(facts, Mapping):
        raise ValueError("host facts must be a JSON object")
    host = {**facts, "metal_devices": [{"name": item["name"], "description": item["description"],
                                        "total_mib": item["total_mib"]}
                                       for item in devices if item["name"].startswith("MTL")]}
    evidence = {"schema_version": 1, "executable": str(path), "install_manifest": str(manifest_path) if manifest_path
                else None, "linkage": linkage, "devices": devices, "flags": flags, "calls": calls,
                "hash_seconds": hash_seconds, "elapsed_seconds": max(0.0, clock() - started)}
    _atomic_write(artifacts / EVIDENCE_NAME,
                  (json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    bundle = NativeBundle(prepared_utc=utc_now(), native_server=ref, libraries=hashed["libraries"], worker=worker,
                          help_sha256=ref.help_sha256, registry_digest=registry_digest or _host_registry_digest(),
                          host=host, build=build, sandbox=sandbox)
    _write_exclusive(bundle_path, (canonical_json(bundle.model_dump(mode="json")) + "\n").encode("utf-8"))
    return bundle


__all__ = ["NATIVE_BUNDLE_NAME", "NativeProcessExecutor", "WorkerInspectExecutor", "capture_native_capabilities",
           "check_native_flags", "check_native_linkage", "example_native_config", "hash_native_server",
           "inspect_worker_image", "native_environment", "parse_list_devices", "prepare_native",
           "read_install_manifest", "read_macho_linkage", "sandbox_verdict", "verify_install_manifest"]
