"""Native (metal-native) preparation and the arm64 worker image option.

Nothing here starts llama-server, Docker or a GPU workload. Executables are tiny synthetic Mach-O images (the
linkage proof reads load commands, it never runs them), the native executor is exercised with fake process
factories plus one POSIX shell script, and Docker is a scripted executor held to the real bounded vocabulary.
"""

import hashlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from llmbench.coding.image_plan import WORKER_PLATFORMS, worker_image_build_plan
from llmbench.coding.sandbox import BoundedProcessExecutor, WorkerResult
from llmbench.config import RunMode, canonical_json
from llmbench.containers.capabilities import ALLOWED_FLAGS, help_sha256
from llmbench.containers.config import NativeBundle, read_native_bundle
from llmbench.containers.native_prep import (CAPTURE_FLAGS, DEVICES_FILE, EVIDENCE_NAME, NATIVE_BUNDLE_NAME,
                                             NativeProcessExecutor, WorkerInspectExecutor,
                                             capture_native_capabilities, check_native_linkage,
                                             example_native_config, hash_native_server, native_environment,
                                             parse_list_devices, prepare_native, read_macho_linkage,
                                             sandbox_verdict, verify_install_manifest)
from llmbench.safety import OperationForbidden, SessionLock

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
HELP = (DATA / "llama-server-help-b11011.txt").read_bytes()
VERSION = (b"version: 0.4.1-dev (build 11011, commit aa39d7a3e)\n"
           b"built with AppleClang 21.0.0.21000101 for Darwin arm64\n")
DEVICES = b"Available devices:\n  MTL0: Apple M1 (5461 MiB, 5460 MiB free)\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n"
NATIVE = SessionLock(allow_native_execution=True, allow_container_execution=True)
NATIVE_ONLY = SessionLock(allow_native_execution=True)
CONTAINER_ONLY = SessionLock(allow_container_execution=True, allow_model_operations=True, allow_inference=True)
HOST = {"machine": "arm64", "chip": "Apple M1", "memory_bytes": 8589934592, "macos": "14.2.1",
        "performance_cores": 4, "efficiency_cores": 4, "gpu_core_count": 8}
WORKER_ID = "sha256:" + "c" * 64
BASE = "node:24-bookworm-slim@sha256:" + "a" * 64
ARM64, X86_64 = 0x0100000C, 0x01000007
POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX links, FIFOs and shell scripts")


# ----------------------------------------------------------------------------------------------- synthetic Mach-O

def _command(cmd: int, fixed: bytes, text: str) -> bytes:
    payload = text.encode("utf-8") + b"\x00"
    size = 8 + len(fixed) + len(payload)
    size += (-size) % 8
    return struct.pack("<II", cmd, size) + fixed + payload + b"\x00" * (size - 8 - len(fixed) - len(payload))


def macho(*, filetype=2, dylibs=(), rpaths=(), install_name=None, cputype=ARM64, weak=(), dylinker=None,
          environment=(), subtype=0) -> bytes:
    """A minimal 64-bit Mach-O header plus the load commands the linkage proof reads (`subtype` 2 is arm64e)."""
    commands = []
    if dylinker:
        commands.append(_command(0xE, struct.pack("<I", 12), dylinker))
    if install_name:
        commands.append(_command(0xD, struct.pack("<IIII", 24, 0, 0, 0), install_name))
    commands += [_command(0xC, struct.pack("<IIII", 24, 0, 0, 0), name) for name in dylibs]
    commands += [_command(0x80000018, struct.pack("<IIII", 24, 0, 0, 0), name) for name in weak]
    commands += [_command(0x8000001C, struct.pack("<I", 12), path) for path in rpaths]
    commands += [_command(0x27, struct.pack("<I", 12), variable) for variable in environment]
    body = b"".join(commands)
    header = struct.pack("<IiIIIIII", 0xFEEDFACF, cputype, subtype, filetype, len(commands), len(body), 0, 0)
    return header + body + b"\x00" * 64


def universal(slices: list[tuple]) -> bytes:
    """A fat (universal) binary: big-endian fat_arch table, slices aligned to 4 KiB. A slice is (cputype, image) or
    (cputype, subtype, image); the subtype is what the table claims, independent of the slice's own header."""
    offset, table, payload = 4096, b"", b""
    for *kind, image in slices:
        cputype, subtype = kind if len(kind) == 2 else (kind[0], 0)
        table += struct.pack(">iIIII", cputype, subtype, offset + len(payload), len(image), 12)
        payload += image + b"\x00" * ((-len(image)) % 4096)
    header = struct.pack(">II", 0xCAFEBABE, len(slices)) + table
    return header + b"\x00" * (offset - len(header)) + payload


def write(path: Path, data: bytes, mode=0o644) -> Path:
    path.write_bytes(data)
    path.chmod(mode)
    return path


def release(root: Path, *, server=None, metal=None) -> Path:
    """A llama.cpp-release-shaped directory: versioned libraries, SONAME links, an unrelated tool, a licence."""
    root.mkdir(parents=True)
    write(root / "llama-server", server or macho(
        dylibs=("@rpath/libllama.0.dylib", "@rpath/libggml-metal.0.dylib", "/usr/lib/libSystem.B.dylib"),
        rpaths=("@loader_path",)), 0o755)
    write(root / "libllama.0.4.1.dylib", macho(filetype=6, install_name="@rpath/libllama.0.dylib",
                                               dylibs=("@rpath/libggml-metal.0.dylib", "/usr/lib/libc++.1.dylib")))
    write(root / "libggml-metal.0.24.0.dylib", metal or macho(
        filetype=6, install_name="@rpath/libggml-metal.0.dylib",
        dylibs=("/System/Library/Frameworks/Metal.framework/Versions/A/Metal",)))
    write(root / "libllama-cli-impl.dylib", macho(filetype=6, install_name="@rpath/libllama-cli-impl.dylib"))
    for link, target in (("libllama.0.dylib", "libllama.0.4.1.dylib"), ("libllama.dylib", "libllama.0.dylib"),
                         ("libggml-metal.0.dylib", "libggml-metal.0.24.0.dylib")):
        (root / link).symlink_to(target)
    write(root / "llama-cli", macho(), 0o755)
    write(root / "LICENSE", b"MIT")
    return root.resolve() / "llama-server"


def install_manifest(executable: Path, **changes) -> Path:
    """What scripts/install_llamacpp_macos.py records, computed independently here."""
    root = executable.parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in root.iterdir() if path.is_file() and not path.is_symlink()}
    links = {path.name: os.readlink(path) for path in root.iterdir() if path.is_symlink()}
    manifest = {"schema_version": 1, "asset": "llama-b11011-bin-macos-arm64.tar.gz", "sha256": "9f" * 32,
                "size": 11156605, "tag": "b11011", "commit": "aa39d7a3e145a88202793a89462d65e94a5fc25f",
                "build_info": "b11011-aa39d7a3e", "upstream_build_flags": ["-DGGML_RPC=ON"],
                "release_workflow_sha256": "e1" * 32, "compiler": "built with AppleClang 21.0.0.21000101",
                "files": files, "links": links, **changes}
    path = root / "install-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------- scripted fakes

class FakeServer:
    """Stands in for NativeProcessExecutor: exact argv only, scripted answers, optional side effects."""

    def __init__(self, executable: Path, *, outputs=None, during=None):
        self.executable, self.calls, self.during = str(executable), [], during or {}
        self.outputs = {"--version": WorkerResult("completed", 0, b"", VERSION),
                        "--help": WorkerResult("completed", 0, HELP, b"ggml_metal_library_init: loaded\n"),
                        "--list-devices": WorkerResult("completed", 0, DEVICES, b""), **(outputs or {})}

    def run(self, argv, *, timeout_seconds, max_output_bytes):
        assert type(argv) is tuple and len(argv) == 2 and argv[0] == self.executable and argv[1] in CAPTURE_FLAGS
        assert type(timeout_seconds) is int and timeout_seconds >= 1 and max_output_bytes >= 1
        self.calls.append(argv[1])
        if argv[1] in self.during:
            self.during[argv[1]]()
        return self.outputs[argv[1]]


class FakeDocker:
    """Worker image inspection held to WorkerInspectExecutor's one allowed shape."""

    def __init__(self, *, architecture="arm64", os_name="linux", returned_id=WORKER_ID, entrypoint=("python3",)):
        self.calls, self.raw = [], {"Id": returned_id, "Os": os_name, "Architecture": architecture,
                                    "Config": {"Entrypoint": list(entrypoint)}}

    def run(self, argv, *, timeout_seconds, max_output_bytes):
        WorkerInspectExecutor._validate(argv)
        self.calls.append(argv)
        return WorkerResult("completed", 0, json.dumps(self.raw).encode("utf-8"))


class Probe:
    def __init__(self, **result):
        self.calls = []
        self.result = {"status": "available", "reason": None, "server_os_arch": "linux/arm64",
                       "image_platform": "linux/arm64", "docker_cli": "/usr/local/bin/docker", **result}

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


def iidfile(tmp_path: Path) -> Path:
    path = tmp_path / "worker-image.id"
    path.write_text(WORKER_ID + "\n", encoding="ascii")
    return path


def prepare(tmp_path, executable, **overrides):
    kwargs = dict(session_lock=NATIVE, host_platform=("Darwin", "arm64"), executor=FakeServer(executable),
                  host_facts=lambda: dict(HOST), docker_probe=Probe(), registry_digest="e" * 64)
    kwargs.update(overrides)
    return prepare_native(executable, tmp_path / "prep", **kwargs)


# --------------------------------------------------------------------------------------------- hash_native_server

@POSIX
def test_hash_pins_every_library_name_by_the_bytes_it_resolves_to(tmp_path):
    executable = release(tmp_path / "rel")
    hashed = hash_native_server(executable)
    root = executable.parent
    digest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("libllama.0.4.1.dylib", "libggml-metal.0.24.0.dylib", "libllama-cli-impl.dylib")}
    assert hashed["executable_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    # every lib name dyld could load by, links included, valued with the file it resolves to; tools and the
    # licence are not libraries
    assert hashed["libraries"] == {
        "libggml-metal.0.24.0.dylib": digest["libggml-metal.0.24.0.dylib"],
        "libggml-metal.0.dylib": digest["libggml-metal.0.24.0.dylib"],
        "libllama-cli-impl.dylib": digest["libllama-cli-impl.dylib"],
        "libllama.0.4.1.dylib": digest["libllama.0.4.1.dylib"],
        "libllama.0.dylib": digest["libllama.0.4.1.dylib"], "libllama.dylib": digest["libllama.0.4.1.dylib"]}
    assert hashed["libraries_sha256"] == hashlib.sha256(canonical_json(hashed["libraries"]).encode()).hexdigest()
    assert hash_native_server(executable) == hashed  # deterministic


@POSIX
def test_repointing_a_library_link_changes_the_pin_without_changing_any_bytes(tmp_path):
    executable = release(tmp_path / "rel")
    root = executable.parent
    write(root / "libggml-metal.0.23.0.dylib", macho(filetype=6, install_name="@rpath/libggml-metal.0.dylib"))
    before = hash_native_server(executable)
    (root / "libggml-metal.0.dylib").unlink()
    (root / "libggml-metal.0.dylib").symlink_to("libggml-metal.0.23.0.dylib")
    after = hash_native_server(executable)
    assert set(before["libraries"].values()) == set(after["libraries"].values())  # the same regular files
    assert after["libraries_sha256"] != before["libraries_sha256"]  # but a different library is loaded


@POSIX
def test_hash_refuses_links_that_escape_and_executables_that_are_links(tmp_path):
    executable = release(tmp_path / "rel")
    write(tmp_path / "libevil.dylib", b"elsewhere")
    (executable.parent / "libggml-rpc.dylib").symlink_to(tmp_path / "libevil.dylib")
    with pytest.raises(ValueError, match="outside"):
        hash_native_server(executable)
    (executable.parent / "libggml-rpc.dylib").unlink()
    alias = tmp_path / "llama-server"
    alias.symlink_to(executable)
    with pytest.raises(ValueError, match="regular file, not a link"):
        hash_native_server(alias)
    with pytest.raises(ValueError, match="absolute"):
        hash_native_server(Path("rel/llama-server"))


@POSIX
@pytest.mark.parametrize("name", ["default.metallib", "ggml-metal.metal", "ggml-common.h", "ggml-metal-impl.h"])
def test_a_metal_shader_library_beside_the_executable_is_pinned_like_a_library(tmp_path, name):
    # a build without GGML_METAL_EMBED_LIBRARY loads (or compiles) its GPU kernels from these files at run time:
    # rebuilding them in place must change the pin even though no dylib and not the executable changed
    executable = release(tmp_path / "rel")
    shader = write(executable.parent / name, b"kernels v1")
    before = hash_native_server(executable)
    assert before["libraries"][name] == hashlib.sha256(b"kernels v1").hexdigest()
    shader.write_bytes(b"kernels v2")
    after = hash_native_server(executable)
    assert after["executable_sha256"] == before["executable_sha256"]
    assert after["libraries_sha256"] != before["libraries_sha256"]
    # ...and an embedded build (no such file) pins exactly the dylibs, as before
    shader.unlink()
    assert set(hash_native_server(executable)["libraries"]) == {
        "libggml-metal.0.24.0.dylib", "libggml-metal.0.dylib", "libllama-cli-impl.dylib", "libllama.0.4.1.dylib",
        "libllama.0.dylib", "libllama.dylib"}


@POSIX
def test_doctor_and_prepare_agree_on_a_pinned_metal_shader_library(tmp_path):
    from llmbench.evidence_cli import native_bundle_state
    executable = release(tmp_path / "rel")
    shader = write(executable.parent / "default.metallib", b"kernels v1")
    prepare(tmp_path, executable)
    state = native_bundle_state(tmp_path / "prep" / NATIVE_BUNDLE_NAME)
    assert state["libraries"]["default.metallib"] == "matches" and state["libraries_sha256_matches"] is True
    shader.write_bytes(b"kernels v2")
    state = native_bundle_state(tmp_path / "prep" / NATIVE_BUNDLE_NAME)
    assert state["libraries"]["default.metallib"] == "differs" and state["libraries_sha256_matches"] is False


@POSIX
def test_a_metal_shader_library_the_install_did_not_record_is_refused(tmp_path):
    executable = release(tmp_path / "rel")
    install_manifest(executable)
    write(executable.parent / "default.metallib", b"dropped in after the install")
    server = FakeServer(executable)
    with pytest.raises(ValueError, match="default.metallib is not part of the recorded install"):
        prepare(tmp_path, executable, executor=server)
    assert server.calls == []


@POSIX
def test_hash_refuses_a_fifo_library_without_blocking(tmp_path):
    executable = release(tmp_path / "rel")
    os.mkfifo(executable.parent / "libggml-fifo.dylib")
    with pytest.raises(ValueError, match="neither a regular file"):
        hash_native_server(executable)
    (executable.parent / "libggml-fifo.dylib").unlink()
    os.mkfifo(executable.parent / "pipe")
    (executable.parent / "libggml-pipe.dylib").symlink_to("pipe")  # a link inside the directory, to a FIFO
    with pytest.raises(ValueError, match="not a regular file"):
        hash_native_server(executable)


# -------------------------------------------------------------------------------------------------- linkage proof

@POSIX
def test_linkage_proof_walks_exactly_what_the_executable_loads(tmp_path):
    executable = release(tmp_path / "rel")
    linkage = check_native_linkage(executable, hash_native_server(executable)["libraries"])
    # the unrelated libllama-cli-impl is pinned (hashed) but not loaded, so it is not part of the walk
    assert sorted(linkage["images"]) == ["libggml-metal.0.24.0.dylib", "libllama.0.4.1.dylib", "llama-server"]
    assert linkage["images"]["llama-server"]["rpaths"] == ["@loader_path"]
    assert linkage["system_libraries"] == ["/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
                                           "/usr/lib/libSystem.B.dylib", "/usr/lib/libc++.1.dylib"]


@POSIX
@pytest.mark.parametrize("server, fragment", [
    (macho(dylibs=("@rpath/libllama.0.dylib",), rpaths=("@loader_path/../lib",)), "searches '@loader_path/../lib'"),
    (macho(dylibs=("@rpath/libllama.0.dylib",), rpaths=("/opt/homebrew/lib",)), "searches '/opt/homebrew/lib'"),
    (macho(dylibs=("/opt/homebrew/opt/ggml/lib/libggml.dylib",)), "loads /opt/homebrew/opt/ggml/lib/libggml.dylib"),
    (macho(dylibs=("/usr/local/lib/libggml.dylib",)), "loads /usr/local/lib/libggml.dylib"),
    (macho(dylibs=("@rpath/libmissing.dylib",), rpaths=("@loader_path",)), "loads @rpath/libmissing.dylib"),
    (macho(dylibs=("@rpath/sub/libllama.0.dylib",), rpaths=("@loader_path",)), "loads @rpath/sub/libllama"),
    (macho(weak=("@rpath/libweak.dylib",), rpaths=("@loader_path",)), "loads @rpath/libweak.dylib"),
    # starts with /usr/lib/ but dyld opens /opt/homebrew/lib/libggml.dylib: a system prefix is not a system path
    (macho(dylibs=("/usr/lib/../../opt/homebrew/lib/libggml.dylib",)), "loads /usr/lib/../../opt/homebrew"),
    (macho(dylibs=("/System/Library/../../tmp/libggml.dylib",)), "loads /System/Library/../../tmp"),
    # a hashed leaf, but no LC_RPATH in scope: dyld would try /usr/local/lib/libllama.0.dylib and /usr/lib first
    (macho(dylibs=("@rpath/libllama.0.dylib",)), "loads @rpath/libllama.0.dylib with no LC_RPATH"),
    (macho(dylibs=("@rpath/libllama.0.dylib",), rpaths=("@loader_path",),
           environment=("DYLD_LIBRARY_PATH=/opt/homebrew/lib",)), "embeds the dyld environment 'DYLD_LIBRARY_PATH"),
    (macho(dylibs=("@rpath/libllama.0.dylib",), rpaths=("@loader_path",), dylinker="/tmp/dyld"),
     "asks the kernel for the loader '/tmp/dyld'"),
    (macho(cputype=X86_64), "not arm64"),
    (b"#!/bin/sh\necho a shell script is not a mach-o image at all\n", "not a 64-bit Mach-O"),
    (b"MZ", "not a Mach-O image"),
])
def test_linkage_proof_refuses_code_the_pin_does_not_cover(tmp_path, server, fragment):
    executable = release(tmp_path / "rel", server=server)
    with pytest.raises(ValueError, match="not a|pin does not cover") as caught:
        check_native_linkage(executable, hash_native_server(executable)["libraries"])
    assert fragment in str(caught.value)


@POSIX
def test_an_rpath_dependency_resolves_through_its_own_image_or_the_executable(tmp_path):
    # no LC_RPATH on the executable: libllama (loaded by @loader_path) carries its own, so its @rpath load of the
    # Metal backend is found beside it; the executable's system dyld is accepted
    server = macho(dylibs=("@loader_path/libllama.0.dylib", "/usr/lib/libSystem.B.dylib"), dylinker="/usr/lib/dyld")
    executable = release(tmp_path / "own", server=server)
    write(executable.parent / "libllama.0.4.1.dylib", macho(filetype=6, install_name="@rpath/libllama.0.dylib",
                                                            dylibs=("@rpath/libggml-metal.0.dylib",),
                                                            rpaths=("@loader_path",)))
    linkage = check_native_linkage(executable, hash_native_server(executable)["libraries"])
    assert sorted(linkage["images"]) == ["libggml-metal.0.24.0.dylib", "libllama.0.4.1.dylib", "llama-server"]
    # neither the library nor the executable has an LC_RPATH: dyld would fall back to /usr/local/lib and /usr/lib
    executable = release(tmp_path / "none", server=server)
    with pytest.raises(ValueError, match="libllama.0.4.1.dylib loads @rpath/libggml-metal.0.dylib with no LC_RPATH"):
        check_native_linkage(executable, hash_native_server(executable)["libraries"])


@POSIX
def test_linkage_proof_follows_libraries_into_their_own_dependencies(tmp_path):
    metal = macho(filetype=6, dylibs=("@loader_path/../Frameworks/libggml-base.dylib",))
    executable = release(tmp_path / "rel", metal=metal)
    with pytest.raises(ValueError, match="libggml-metal.0.24.0.dylib loads @loader_path/../Frameworks"):
        check_native_linkage(executable, hash_native_server(executable)["libraries"])


def test_macho_reader_selects_the_arm64_slice_and_refuses_malformed_images(tmp_path):
    fat = write(tmp_path / "fat", universal([(X86_64, macho(cputype=X86_64, dylibs=("/usr/lib/x86.dylib",))),
                                             (ARM64, macho(dylibs=("/usr/lib/arm.dylib",), rpaths=("@loader_path",)))]))
    info = read_macho_linkage(fat)
    assert info == {"cpu": "arm64", "filetype": "execute", "install_name": None, "dylibs": ["/usr/lib/arm.dylib"],
                    "rpaths": ["@loader_path"], "dylinker": None, "environment": []}
    with pytest.raises(ValueError, match="without an arm64 slice"):
        read_macho_linkage(write(tmp_path / "intel", universal([(X86_64, macho(cputype=X86_64))])))
    image = macho(dylibs=("/usr/lib/libSystem.B.dylib",))
    with pytest.raises(ValueError, match="truncated"):
        read_macho_linkage(write(tmp_path / "short", image[:40]))
    bad_offset = bytearray(image)
    struct.pack_into("<I", bad_offset, 32 + 8, 4000)  # the dylib name offset points outside its command
    with pytest.raises(ValueError, match="outside the command"):
        read_macho_linkage(write(tmp_path / "offset", bytes(bad_offset)))
    oversized = bytearray(image)
    struct.pack_into("<I", oversized, 20, 64 * 1024 * 1024)  # sizeofcmds
    with pytest.raises(ValueError, match="bounds"):
        read_macho_linkage(write(tmp_path / "huge", bytes(oversized)))


@pytest.mark.skipif(sys.platform != "darwin", reason="reads a real system Mach-O image")
def test_macho_reader_reads_a_real_system_binary():
    info = read_macho_linkage("/bin/ls")  # a universal binary with an arm64(e) slice on every supported macOS
    # a platform binary ships x86_64 + arm64e: its only arm64-family slice is read, and the result says which
    assert info["cpu"] in ("arm64", "arm64e") and info["filetype"] == "execute"
    assert "/usr/lib/libSystem.B.dylib" in info["dylibs"]
    assert info["dylinker"] == "/usr/lib/dyld" and info["environment"] == []


ARM64E = 2
ARM64E_PTRAUTH = 0x80000002  # arm64e with the pointer-authentication ABI capability bit, as Apple's toolchain writes it
HOMEBREW = "/opt/homebrew/lib/libggml-metal.dylib"
METAL_FRAMEWORK = "/System/Library/Frameworks/Metal.framework/Versions/A/Metal"


@pytest.mark.parametrize("arm64e_first", [True, False])
def test_macho_reader_reads_the_arm64_slice_an_arm64_process_loads_not_the_first_arm64_type(tmp_path, arm64e_first):
    # arm64 and arm64e share CPU_TYPE_ARM64; the arm64 process that runs llama-server loads the arm64 slice
    arm64e = (ARM64, ARM64E_PTRAUTH, macho(filetype=6, subtype=ARM64E_PTRAUTH, dylibs=(METAL_FRAMEWORK,)))
    arm64 = (ARM64, 0, macho(filetype=6, dylibs=(HOMEBREW,)))
    fat = write(tmp_path / "fat", universal([arm64e, arm64] if arm64e_first else [arm64, arm64e]))
    info = read_macho_linkage(fat)
    assert info["cpu"] == "arm64" and info["dylibs"] == [HOMEBREW]


@POSIX
def test_linkage_proof_is_not_fooled_by_an_arm64e_slice_listed_before_the_arm64_one(tmp_path):
    # the arm64e slice only loads Metal; the arm64 slice dyld really loads pulls in an unpinned Homebrew library
    metal = universal([(ARM64, ARM64E, macho(filetype=6, subtype=ARM64E, install_name="@rpath/libggml-metal.0.dylib",
                                             dylibs=(METAL_FRAMEWORK,))),
                       (ARM64, 0, macho(filetype=6, install_name="@rpath/libggml-metal.0.dylib",
                                        dylibs=(METAL_FRAMEWORK, HOMEBREW)))])
    executable = release(tmp_path / "rel", metal=metal)
    with pytest.raises(ValueError, match=f"libggml-metal.0.24.0.dylib loads {HOMEBREW}, which is not a hashed"):
        check_native_linkage(executable, hash_native_server(executable)["libraries"])


def test_an_arm64e_slice_is_read_only_when_it_is_the_only_one_and_is_reported_as_arm64e(tmp_path):
    only = write(tmp_path / "only", universal([(X86_64, macho(cputype=X86_64)),
                                               (ARM64, ARM64E_PTRAUTH, macho(subtype=ARM64E_PTRAUTH))]))
    assert read_macho_linkage(only)["cpu"] == "arm64e"
    assert read_macho_linkage(write(tmp_path / "thin", macho(subtype=ARM64E_PTRAUTH)))["cpu"] == "arm64e"
    assert read_macho_linkage(write(tmp_path / "v8", macho(subtype=1)))["cpu"] == "arm64"  # CPU_SUBTYPE_ARM64_V8


@pytest.mark.parametrize("slices, fragment", [
    ([(ARM64, 0, macho()), (ARM64, 1, macho(subtype=1))], "2 arm64 slices; which one loads is ambiguous"),
    ([(ARM64, ARM64E, macho(subtype=ARM64E)), (ARM64, ARM64E, macho(subtype=ARM64E))],
     "2 arm64e slices and no arm64 slice"),
    ([(ARM64, 0, macho(subtype=ARM64E))], "lists an arm64 slice whose header says arm64e"),
    ([(ARM64, 7, macho(subtype=7))], "unknown CPU subtype 0x7"),
])
def test_an_ambiguous_or_inconsistent_universal_binary_is_refused(tmp_path, slices, fragment):
    with pytest.raises(ValueError, match="universal binary|unknown CPU subtype") as caught:
        read_macho_linkage(write(tmp_path / "fat", universal(slices)))
    assert fragment in str(caught.value)


@POSIX
def test_linkage_proof_refuses_an_arm64e_executable(tmp_path):
    # it would run only under the arm64e preview ABI, where each library's arm64e slice loads instead
    server = macho(subtype=ARM64E, dylibs=("@rpath/libllama.0.dylib",), rpaths=("@loader_path",))
    executable = release(tmp_path / "rel", server=server)
    with pytest.raises(ValueError, match="llama-server is an arm64e executable; the proof covers an arm64 process"):
        check_native_linkage(executable, hash_native_server(executable)["libraries"])


# ------------------------------------------------------------------------------------------ NativeProcessExecutor

class FakeProcess:
    def __init__(self, stdout=b"", stderr=b"", returncode=0, finishes=True):
        self.stdout, self.stderr = io.BytesIO(stdout), io.BytesIO(stderr)
        self._code, self._finishes, self.killed = returncode, finishes, False

    def poll(self):
        return self._code if self._finishes or self.killed else None

    def kill(self):
        self.killed = True
        self._code = -9

    def wait(self, timeout=None):
        return self._code


class FakePopen:
    def __init__(self, **process):
        self.calls, self.process = [], process

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        self.last = FakeProcess(**self.process)
        return self.last


@POSIX
def test_native_executor_needs_the_native_permission_not_a_container_grant(tmp_path):
    executable = release(tmp_path / "rel")
    for lock in (CONTAINER_ONLY, SessionLock()):
        popen = FakePopen(stdout=VERSION)
        with pytest.raises(OperationForbidden, match="native forbidden"):
            NativeProcessExecutor(executable, session_lock=lock, cwd=tmp_path, popen_factory=popen).run(
                "--version", timeout_seconds=5)
        assert popen.calls == []
    popen = FakePopen(stdout=VERSION)
    with pytest.raises(OperationForbidden, match="live mode"):
        NativeProcessExecutor(executable, session_lock=NATIVE, cwd=tmp_path, mode=RunMode.DRY_RUN,
                              popen_factory=popen).run("--version", timeout_seconds=5)
    assert popen.calls == []


@POSIX
@pytest.mark.parametrize("argv", ["--port", "--model", ("--version",), ("/bin/sh", "--version"),
                                  (None, "--version", "--verbose"), (None, "--help=x"), [None, "--version"]])
def test_native_executor_runs_only_the_three_capture_argv(tmp_path, argv):
    executable = release(tmp_path / "rel")
    if isinstance(argv, (tuple, list)):
        argv = type(argv)(str(executable) if item is None else item for item in argv)
    popen = FakePopen(stdout=VERSION)
    executor = NativeProcessExecutor(executable, session_lock=NATIVE, cwd=tmp_path, popen_factory=popen)
    with pytest.raises(ValueError, match="runs only"):
        executor.run(argv, timeout_seconds=5)
    assert popen.calls == []
    with pytest.raises(ValueError, match="absolute"):
        NativeProcessExecutor("llama-server", session_lock=NATIVE, cwd=tmp_path)


@POSIX
def test_native_executor_scrubs_the_environment_and_sets_the_working_directory(tmp_path):
    executable = release(tmp_path / "rel")
    environment = {"PATH": "/usr/bin:/bin", "HOME": "/Users/example", "KEEP_ME": "1", "LLAMA_ARG_MODEL": "/x.gguf",
                   "LLAMA_CACHE": "/c", "GGML_METAL_NDEBUG": "1", "HF_TOKEN": "t", "HUGGINGFACE_HUB_CACHE": "h",
                   "https_proxy": "http://p", "HTTP_PROXY": "http://p", "NO_PROXY": "*", "all_proxy": "socks://p",
                   "DYLD_INSERT_LIBRARIES": "/evil.dylib", "DYLD_LIBRARY_PATH": "/opt/lib",
                   "MTL_DEBUG_LAYER": "1", "METAL_DEVICE_WRAPPER_TYPE": "1"}
    popen = FakePopen(stdout=VERSION)
    result = NativeProcessExecutor(executable, session_lock=NATIVE, cwd=tmp_path / "work", popen_factory=popen,
                                   environment=environment).run("--version", timeout_seconds=5)
    assert (result.status, result.returncode, result.stdout) == ("completed", 0, VERSION)
    (argv, kwargs), = popen.calls
    assert argv == (str(executable), "--version")
    assert kwargs["env"] == {"PATH": "/usr/bin:/bin", "HOME": "/Users/example", "KEEP_ME": "1"}
    assert kwargs["cwd"] == str(tmp_path / "work") and kwargs["shell"] is False
    assert native_environment(environment) == kwargs["env"]


@POSIX
def test_native_executor_bounds_time_and_output(tmp_path):
    executable = release(tmp_path / "rel")
    ticks = iter(range(0, 1000, 10))
    hung = FakePopen(finishes=False)
    result = NativeProcessExecutor(executable, session_lock=NATIVE, cwd=tmp_path, popen_factory=hung,
                                   clock=lambda: next(ticks)).run("--list-devices", timeout_seconds=5)
    assert result.status == "timeout" and hung.last.killed
    flood = FakePopen(stdout=b"x" * 4096, finishes=False)
    result = NativeProcessExecutor(executable, session_lock=NATIVE, cwd=tmp_path, popen_factory=flood).run(
        "--help", timeout_seconds=5, max_output_bytes=100)
    assert result.status == "output-limit" and flood.last.killed and len(result.stdout) <= 100


@POSIX
def test_native_executor_refuses_an_executable_swapped_for_a_link(tmp_path):
    executable = release(tmp_path / "rel")
    popen = FakePopen(stdout=VERSION)
    executor = NativeProcessExecutor(executable, session_lock=NATIVE, cwd=tmp_path, popen_factory=popen)
    executable.rename(executable.with_name("llama-server.real"))
    executable.symlink_to("llama-server.real")
    result = executor.run("--version", timeout_seconds=5)
    assert result.status == "environment-error" and b"no longer a regular file" in result.stderr
    assert popen.calls == []


@POSIX
def test_native_executor_runs_a_real_process_with_its_bounded_readers(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    script = write(tmp_path / "llama-server", b"#!/bin/sh\necho \"cwd=$(pwd -P)\"\necho \"keep=$KEEP_ME\"\n"
                                              b"echo \"llama=$LLAMA_ARG_PORT\"\necho err >&2\nexit 7\n", 0o755)
    executor = NativeProcessExecutor(script, session_lock=NATIVE, cwd=work,
                                     environment={"PATH": "/usr/bin:/bin", "KEEP_ME": "yes", "LLAMA_ARG_PORT": "1"})
    assert executor.synthetic is False
    result = executor.run((str(script), "--help"), timeout_seconds=30)
    assert (result.status, result.returncode) == ("completed", 7)
    lines = result.stdout.decode().splitlines()
    assert lines == [f"cwd={work.resolve()}", "keep=yes", "llama="] and result.stderr == b"err\n"


# ---------------------------------------------------------------------------------- capture_native_capabilities

@POSIX
def test_capture_writes_help_stdout_only_and_version_and_devices_with_stderr(tmp_path):
    executable = release(tmp_path / "rel")
    output = tmp_path / "caps"
    output.mkdir()
    server = FakeServer(executable, outputs={"--list-devices": WorkerResult("completed", 0, DEVICES, b"ggml: init\n")})
    caps = capture_native_capabilities(executable, output, executor=server)
    assert server.calls == ["--version", "--help", "--list-devices"]
    assert (output / "llama-server-help.txt").read_bytes() == HELP  # the Metal init log on stderr is excluded
    assert (output / "llama-server-version.txt").read_bytes() == VERSION
    assert (output / DEVICES_FILE).read_bytes() == DEVICES + b"ggml: init\n"
    assert caps.help_sha256 == help_sha256(HELP.decode()) and caps.build == "b11011-aa39d7a3e"
    assert parse_list_devices(DEVICES.decode()) == [
        {"name": "MTL0", "description": "Apple M1", "total_mib": 5461, "free_mib": 5460},
        {"name": "BLAS", "description": "Accelerate", "total_mib": 0, "free_mib": 0}]


@POSIX
@pytest.mark.parametrize("outputs, fragment", [
    ({"--list-devices": WorkerResult("completed", 0, b"Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n")},
     "no Metal device: `llama-server --list-devices` listed BLAS"),
    ({"--list-devices": WorkerResult("completed", 0, b"Available devices:\n")}, "no Metal device"),
    ({"--help": WorkerResult("timeout", None, b"", b"")}, "--help failed (timeout"),
    ({"--version": WorkerResult("completed", 1, b"", b"dyld: Library not loaded")}, "--version failed"),
    ({"--version": WorkerResult("completed", 0, b"llama-server (unknown build)\n")}, "unrecognized"),
])
def test_capture_refuses_and_writes_nothing(tmp_path, outputs, fragment):
    executable = release(tmp_path / "rel")
    output = tmp_path / "caps"
    output.mkdir()
    with pytest.raises(ValueError, match="failed|Metal|unrecognized") as caught:
        capture_native_capabilities(executable, output, executor=FakeServer(executable, outputs=outputs))
    assert fragment in str(caught.value)
    assert list(output.iterdir()) == []


# ------------------------------------------------------------------------------------------------ prepare_native

@POSIX
def test_prepare_native_pins_the_server_and_records_everything(tmp_path):
    executable = release(tmp_path / "rel")
    manifest = install_manifest(executable)
    server, docker, probe = FakeServer(executable), FakeDocker(), Probe()
    bundle = prepare(tmp_path, executable, executor=server, docker_executor=docker, docker_probe=probe,
                     worker_iidfile=iidfile(tmp_path))
    hashed = hash_native_server(executable)
    prep = (tmp_path / "prep").resolve()
    ref = bundle.native_server
    assert (ref.executable, ref.executable_sha256, ref.libraries_sha256) == (
        str(executable), hashed["executable_sha256"], hashed["libraries_sha256"])
    assert (ref.build_info, ref.help_sha256, ref.backend, ref.platform) == (
        "b11011-aa39d7a3e", help_sha256(HELP.decode()), "metal", "darwin/arm64")
    assert ref.source == "llama.cpp b11011 release asset llama-b11011-bin-macos-arm64.tar.gz (sha256 9f9f9f9f9f9f)"
    assert bundle.help_sha256 == ref.help_sha256 and bundle.libraries == hashed["libraries"]
    assert bundle.registry_digest == "e" * 64 and bundle.runtime == "metal-native"
    assert bundle.worker.platform == "linux/arm64" and bundle.worker.image_id == WORKER_ID
    assert bundle.worker.entrypoint == ("python3",) and bundle.worker.role == "worker"
    assert bundle.sandbox["status"] == "available" and "probe_status" not in bundle.sandbox
    assert probe.calls == [{"session_lock": NATIVE, "image_ref": bundle.worker}]
    assert bundle.model_dump(mode="json")["host"] == {
        **HOST, "metal_devices": [{"name": "MTL0", "description": "Apple M1", "total_mib": 5461}]}
    assert bundle.model_dump(mode="json")["build"] == json.loads(manifest.read_text())
    assert server.calls == ["--version", "--help", "--list-devices"]
    assert docker.calls == [("docker", "image", "inspect", "--format", "{{json .}}", WORKER_ID)]
    # the bundle on disk is exactly the returned one; the capture beside it is what help_sha256 pins
    assert read_native_bundle(prep / NATIVE_BUNDLE_NAME) == bundle
    assert help_sha256((prep / "llama-server-help.txt").read_text()) == bundle.help_sha256
    evidence = json.loads((prep / EVIDENCE_NAME).read_text())
    assert [call["argv"][1] for call in evidence["calls"]] == list(CAPTURE_FLAGS)
    assert sorted(evidence["linkage"]["images"]) == ["libggml-metal.0.24.0.dylib", "libllama.0.4.1.dylib",
                                                     "llama-server"]
    assert "--host" in evidence["flags"]["argv"] and "127.0.0.1" in evidence["flags"]["argv"]
    assert not (prep / "image-bundle.json").exists()  # the NVIDIA bundle is never written by a native prepare


@POSIX
def test_prepared_server_drops_into_a_native_config_whose_fingerprint_ignores_where_it_lives(tmp_path):
    first = prepare(tmp_path / "a", release(tmp_path / "a" / "rel"))
    second = prepare(tmp_path / "b", release(tmp_path / "b" / "rel"))
    assert first.native_server.executable != second.native_server.executable
    one, two = example_native_config(first.native_server), example_native_config(second.native_server)
    assert one.runtime == "metal-native" and one.inference_image is None and one.evaluator.mode == "host-process"
    assert one.fingerprint() == two.fingerprint()  # same bytes, same build: the same candidate on any path


@POSIX
@pytest.mark.parametrize("platform", [("Linux", "x86_64"), ("Darwin", "x86_64"), ("Linux", "aarch64")])
def test_prepare_refuses_off_apple_silicon_before_anything_runs(tmp_path, platform):
    executable = release(tmp_path / "rel")
    server = FakeServer(executable)
    with pytest.raises(ValueError, match="Apple Silicon") as caught:
        prepare(tmp_path, executable, executor=server, host_platform=platform)
    assert server.calls == [] and not (tmp_path / "prep").exists()
    if platform == ("Darwin", "x86_64"):
        assert "Rosetta" in str(caught.value)


@POSIX
def test_prepare_needs_the_native_policy_even_with_every_container_permission(tmp_path):
    executable = release(tmp_path / "rel")
    server = FakeServer(executable)
    with pytest.raises(OperationForbidden, match="native forbidden"):
        prepare(tmp_path, executable, executor=server, session_lock=CONTAINER_ONLY)
    assert server.calls == [] and not (tmp_path / "prep").exists()


@POSIX
def test_prepare_never_overwrites_an_existing_bundle(tmp_path):
    executable = release(tmp_path / "rel")
    (tmp_path / "prep").mkdir()
    (tmp_path / "prep" / NATIVE_BUNDLE_NAME).write_text("earlier", encoding="utf-8")
    server = FakeServer(executable)
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path, executable, executor=server)
    assert (tmp_path / "prep" / NATIVE_BUNDLE_NAME).read_text() == "earlier" and server.calls == []


@POSIX
def test_a_bundle_written_while_prepare_ran_is_not_overwritten(tmp_path):
    # the up-front check passed; another prepare finished into the same directory while this one ran
    executable = release(tmp_path / "rel")
    raced = lambda: (tmp_path / "prep" / NATIVE_BUNDLE_NAME).write_text("raced", encoding="utf-8")  # noqa: E731
    server = FakeServer(executable, during={"--list-devices": raced})
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path, executable, executor=server)
    assert (tmp_path / "prep" / NATIVE_BUNDLE_NAME).read_text() == "raced"
    assert not [path.name for path in (tmp_path / "prep").iterdir() if path.name.endswith(".tmp")]


@POSIX
def test_prepare_refuses_an_nvidia_preparation_directory_and_leaves_its_capture_alone(tmp_path):
    executable = release(tmp_path / "rel")
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "image-bundle.json").write_text("{}", encoding="utf-8")
    (prep / "llama-server-help.txt").write_bytes(b"the CUDA image's help")
    server = FakeServer(executable)
    with pytest.raises(ValueError, match="NVIDIA preparation"):
        prepare(tmp_path, executable, executor=server)
    assert (prep / "llama-server-help.txt").read_bytes() == b"the CUDA image's help" and server.calls == []
    assert sorted(path.name for path in prep.iterdir()) == ["image-bundle.json", "llama-server-help.txt"]


@POSIX
def test_prepare_without_a_manifest_records_unrecorded_provenance(tmp_path):
    bundle = prepare(tmp_path, release(tmp_path / "rel"))
    assert bundle.build == {"provenance": "unrecorded"} and bundle.native_server.source == "local"


@POSIX
@pytest.mark.parametrize("tamper, fragment", [
    (lambda root: (root / "libggml-metal.0.24.0.dylib").write_bytes(macho(filetype=6) + b"patched"),
     "libggml-metal.0.24.0.dylib differs from the recorded install"),
    (lambda root: write(root / "libggml-extra.dylib", macho(filetype=6)), "libggml-extra.dylib is not part of"),
    (lambda root: ((root / "libllama.dylib").unlink(), (root / "libllama.dylib").symlink_to("libllama.0.4.1.dylib")),
     "libllama.dylib links to 'libllama.0.4.1.dylib'; the install recorded 'libllama.0.dylib'"),
    (lambda root: write(root / "llama-server", macho(dylibs=("/usr/lib/libSystem.B.dylib",)), 0o755),
     "llama-server is not the recorded install's executable"),
])
def test_prepare_refuses_a_directory_that_no_longer_matches_its_install_manifest(tmp_path, tamper, fragment):
    executable = release(tmp_path / "rel")
    install_manifest(executable)
    tamper(executable.parent)
    server = FakeServer(executable)
    with pytest.raises(ValueError, match="does not match its install-manifest.json") as caught:
        prepare(tmp_path, executable, executor=server)
    assert fragment in str(caught.value) and server.calls == []  # refused before the executable ever ran


@POSIX
def test_prepare_refuses_a_build_other_than_the_manifest_records(tmp_path):
    executable = release(tmp_path / "rel")
    install_manifest(executable, build_info="b11010-0123456")
    with pytest.raises(ValueError, match="reports build b11011-aa39d7a3e; its install manifest records b11010"):
        prepare(tmp_path, executable)
    assert not (tmp_path / "prep" / NATIVE_BUNDLE_NAME).exists()


@POSIX
@pytest.mark.parametrize("content", [b"not json", b"[]", json.dumps({"files": {}}).encode(),
                                     b"{" + b" " * (1024 * 1024) + b"}"])
def test_prepare_refuses_a_present_but_unreadable_manifest(tmp_path, content):
    executable = release(tmp_path / "rel")
    (executable.parent / "install-manifest.json").write_bytes(content)
    with pytest.raises(ValueError, match="manifest|exceeds"):
        prepare(tmp_path, executable)


@POSIX
def test_prepare_refuses_a_build_missing_an_allowlisted_flag(tmp_path):
    lines = HELP.decode().split("\n")
    start = next(index for index, line in enumerate(lines) if line.startswith("--swa-full"))
    end = next(index for index in range(start + 1, len(lines)) if not lines[index].startswith(" "))
    trimmed = "\n".join(lines[:start] + lines[end:]).encode()
    executable = release(tmp_path / "rel")
    server = FakeServer(executable, outputs={"--help": WorkerResult("completed", 0, trimmed, b"")})
    with pytest.raises(ValueError, match="--swa-full is in the harness allowlist but unknown"):
        prepare(tmp_path, executable, executor=server)
    assert "--swa-full" in ALLOWED_FLAGS and not (tmp_path / "prep" / NATIVE_BUNDLE_NAME).exists()


@POSIX
def test_prepare_refuses_a_directory_that_changed_while_the_executable_ran(tmp_path):
    executable = release(tmp_path / "rel")
    swap = lambda: write(executable.parent / "libggml-metal.0.24.0.dylib", macho(filetype=6) + b"swapped")  # noqa: E731
    server = FakeServer(executable, during={"--list-devices": swap})
    with pytest.raises(ValueError, match="changed while they were being prepared"):
        prepare(tmp_path, executable, executor=server)
    assert not (tmp_path / "prep" / NATIVE_BUNDLE_NAME).exists()


@POSIX
def test_prepare_refuses_a_homebrew_style_layout_before_running_it(tmp_path):
    executable = release(tmp_path / "rel", server=macho(dylibs=("@rpath/libllama.dylib",),
                                                        rpaths=("@loader_path/../lib",)))
    server = FakeServer(executable)
    with pytest.raises(ValueError, match="pin does not cover"):
        prepare(tmp_path, executable, executor=server)
    assert server.calls == []


@POSIX
def test_without_a_worker_image_the_sandbox_is_blocked_even_when_docker_is_up(tmp_path):
    probe = Probe()
    bundle = prepare(tmp_path, release(tmp_path / "rel"), docker_probe=probe)
    assert bundle.worker is None and probe.calls == [{"session_lock": NATIVE, "image_ref": None}]
    assert bundle.sandbox["status"] == "blocked" and bundle.sandbox["probe_status"] == "available"
    assert "--worker-iidfile" in bundle.sandbox["reason"]


@POSIX
def test_an_amd64_worker_on_an_arm64_docker_server_is_blocked_not_emulated(tmp_path):
    bundle = prepare(tmp_path, release(tmp_path / "rel"), worker_iidfile=iidfile(tmp_path),
                     docker_executor=FakeDocker(architecture="amd64"), docker_probe=Probe())
    assert bundle.worker.platform == "linux/amd64"
    assert bundle.sandbox["status"] == "blocked" and "--platform linux/arm64" in bundle.sandbox["reason"]


@POSIX
def test_a_blocked_probe_is_recorded_as_it_was_reported(tmp_path):
    probe = Probe(status="blocked", reason="the Docker daemon is not reachable", server_os_arch=None)
    bundle = prepare(tmp_path, release(tmp_path / "rel"), docker_probe=probe)
    assert bundle.sandbox == {**probe.result}


@POSIX
def test_a_worker_image_needs_the_container_permission(tmp_path):
    docker = FakeDocker()
    with pytest.raises(OperationForbidden, match="container forbidden"):
        prepare(tmp_path, release(tmp_path / "rel"), session_lock=NATIVE_ONLY, worker_iidfile=iidfile(tmp_path),
                docker_executor=docker)
    assert docker.calls == []


@POSIX
@pytest.mark.parametrize("docker, fragment", [(FakeDocker(returned_id="sha256:" + "d" * 64), "returned Id"),
                                              (FakeDocker(os_name="windows", architecture="amd64"), "windows/amd64"),
                                              (FakeDocker(entrypoint=()), "entrypoint")])
def test_worker_inspection_must_describe_the_iidfile_image(tmp_path, docker, fragment):
    with pytest.raises(ValueError, match="worker|entrypoint") as caught:
        prepare(tmp_path, release(tmp_path / "rel"), worker_iidfile=iidfile(tmp_path), docker_executor=docker)
    assert fragment in str(caught.value)


@pytest.mark.parametrize("probe", [None, {}, {"status": "maybe"}, {"status": "available", "x": float("nan")}])
def test_sandbox_verdict_refuses_an_unusable_probe_answer(probe):
    with pytest.raises(ValueError):
        sandbox_verdict(probe, None)


@POSIX
@pytest.mark.parametrize("probe, fragment", [
    (lambda **kwargs: {"status": "maybe"}, "ValueError: the Docker sandbox probe returned no available/blocked status"),
    (lambda **kwargs: (_ for _ in ()).throw(OperationForbidden("container forbidden: test")),
     "OperationForbidden: container forbidden: test"),
])
def test_a_failed_sandbox_probe_blocks_coding_but_still_pins_the_server(tmp_path, probe, fragment):
    bundle = prepare(tmp_path, release(tmp_path / "rel"), docker_probe=probe)
    assert bundle.sandbox == {"status": "blocked", "reason": "sandbox probe failed: " + fragment}
    assert (tmp_path / "prep" / NATIVE_BUNDLE_NAME).exists()


@POSIX
def test_an_exact_install_passes_the_manifest_check_and_a_link_chain_is_checked_hop_by_hop(tmp_path):
    executable = release(tmp_path / "rel")
    install_manifest(executable)
    manifest = json.loads((executable.parent / "install-manifest.json").read_text())
    verify_install_manifest(manifest, executable, hash_native_server(executable))  # an exact install passes
    # libllama.dylib -> libllama.0.dylib -> libllama.0.4.1.dylib: the recorded FIRST hop is what is compared,
    # even though both spellings end at the same bytes
    changed = {**manifest, "links": {**manifest["links"], "libllama.dylib": "libllama.0.4.1.dylib"}}
    with pytest.raises(ValueError, match="libllama.dylib links to 'libllama.0.dylib'"):
        verify_install_manifest(changed, executable, hash_native_server(executable))


# ------------------------------------------------------------------ NVIDIA guarantees: vocabularies do not widen

@pytest.mark.parametrize("argv", [("/opt/llama/llama-server", "--version"), ("docker", "image", "inspect",
                                                                              "--format", "{{json .}}", WORKER_ID)])
def test_the_sandbox_docker_vocabulary_is_unchanged_by_native_preparation(argv):
    with pytest.raises(ValueError):
        BoundedProcessExecutor._validate(argv)


@pytest.mark.parametrize("argv", [("docker", "image", "inspect", "--format", "{{json .}}", "llmbench-worker:staged"),
                                  ("docker", "image", "inspect", WORKER_ID),
                                  ("docker", "run", "--rm", WORKER_ID), ("docker", "image", "rm", WORKER_ID)])
def test_the_worker_inspection_executor_allows_one_read_only_shape(argv):
    with pytest.raises(ValueError, match="only `docker image inspect`"):
        WorkerInspectExecutor._validate(argv)


# ------------------------------------------------------------------------------------ arm64 worker image option

def test_default_worker_build_plan_is_byte_identical(tmp_path):
    plan = worker_image_build_plan(base_image=BASE, artifact_dir=tmp_path)
    context = ROOT / "src" / "llmbench" / "coding" / "worker_image"
    assert plan["platform"] == "linux/amd64"
    assert plan["build_argv"] == ["docker", "build", "--platform=linux/amd64", "--pull=false", "--no-cache",
                                  "--build-arg", "BASE_IMAGE=" + BASE, "--iidfile",
                                  str(tmp_path.resolve() / "worker-image.id"), "--tag", "llmbench-worker:staged",
                                  "--file", str(context.resolve() / "Dockerfile"), str(context.resolve())]
    assert worker_image_build_plan(base_image=BASE, artifact_dir=tmp_path, platform="linux/amd64") == plan


def test_arm64_worker_build_plan_changes_only_the_platform(tmp_path):
    amd64 = worker_image_build_plan(base_image=BASE, artifact_dir=tmp_path)
    arm64 = worker_image_build_plan(base_image=BASE, artifact_dir=tmp_path, platform="linux/arm64")
    assert arm64["platform"] == "linux/arm64" and arm64["build_argv"][2] == "--platform=linux/arm64"
    assert {key: value for key, value in arm64.items() if key not in ("platform", "build_argv")} == {
        key: value for key, value in amd64.items() if key not in ("platform", "build_argv")}
    assert arm64["build_argv"][:2] + arm64["build_argv"][3:] == amd64["build_argv"][:2] + amd64["build_argv"][3:]
    assert WORKER_PLATFORMS == ("linux/amd64", "linux/arm64")
    for platform in ("linux/riscv64", "darwin/arm64", "linux/arm64/v8", "", None):
        with pytest.raises(ValueError, match="worker platform"):
            worker_image_build_plan(base_image=BASE, artifact_dir=tmp_path, platform=platform)


def _build_script():
    path = ROOT / "scripts" / "build_worker_image.py"
    spec = importlib.util.spec_from_file_location("build_worker_image_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("extra, platform", [([], "linux/amd64"), (["--platform", "linux/arm64"], "linux/arm64")])
def test_build_script_dry_run_passes_the_platform_through(tmp_path, capsys, extra, platform):
    script = _build_script()
    assert script.main(["--base-image", BASE, "--output", str(tmp_path / "w"), "--dry-run", *extra]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["platform"] == platform and f"--platform={platform}" in plan["build_argv"]
    assert not (tmp_path / "w").exists()
    with pytest.raises(SystemExit):
        script.main(["--base-image", BASE, "--output", str(tmp_path / "w"), "--platform", "linux/s390x"])


class ScriptedDockerCli:
    """Replaces subprocess.run inside the build script: the build writes an iidfile, inspect answers JSON."""

    def __init__(self, architecture):
        self.architecture, self.calls = architecture, []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[1] == "build":
            Path(argv[argv.index("--iidfile") + 1]).write_text(WORKER_ID, encoding="ascii")
            return subprocess.CompletedProcess(argv, 0)
        assert argv == ["docker", "image", "inspect", WORKER_ID] and kwargs["timeout"] >= 1
        raw = [{"Id": WORKER_ID, "Os": "linux", "Architecture": self.architecture}]
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(raw), stderr="")


@pytest.mark.parametrize("requested, built, code", [("linux/arm64", "arm64", 0), ("linux/amd64", "amd64", 0),
                                                   ("linux/arm64", "amd64", 3)])
def test_build_script_records_and_verifies_the_built_platform(tmp_path, monkeypatch, capsys, requested, built, code):
    script = _build_script()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"allow_container_execution": True}), encoding="utf-8")
    docker = ScriptedDockerCli(built)
    monkeypatch.setattr(script.subprocess, "run", docker)
    output = tmp_path / "w"
    assert script.main(["--base-image", BASE, "--output", str(output), "--policy", str(policy),
                        "--platform", requested]) == code
    assert docker.calls[0][2] == f"--platform={requested}"
    assert json.loads((output / "build-plan.json").read_text())["platform"] == requested
    captured = capsys.readouterr()
    if code == 0:
        assert json.loads(captured.out) == {"image_id": WORKER_ID, "iidfile": str(output.resolve() / "worker-image.id"),
                                            "platform": requested}
    else:
        assert f"the built image is linux/{built}, not the requested {requested}" in captured.err


def test_build_script_still_requires_the_container_policy(tmp_path, monkeypatch):
    script = _build_script()
    docker = ScriptedDockerCli("arm64")
    monkeypatch.setattr(script.subprocess, "run", docker)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"allow_native_execution": True}), encoding="utf-8")
    with pytest.raises(OperationForbidden, match="container forbidden"):
        script.main(["--base-image", BASE, "--output", str(tmp_path / "w"), "--policy", str(policy),
                     "--platform", "linux/arm64"])
    assert docker.calls == []


def test_native_bundle_accepts_what_prepare_writes_and_nothing_looser():
    # the lead's contract: help_sha256 must equal the server's; a worker must carry the worker role
    raw = {"prepared_utc": "2026-09-23T00:00:00+00:00", "help_sha256": "a" * 64, "registry_digest": "b" * 64,
           "native_server": {"executable": "/Users/example/llama-b11011/llama-server", "executable_sha256": "c" * 64,
                             "libraries_sha256": "d" * 64, "build_info": "b11011-aa39d7a3e", "help_sha256": "a" * 64}}
    NativeBundle.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError):
        NativeBundle.model_validate_json(json.dumps({**raw, "help_sha256": "f" * 64}))
