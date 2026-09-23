"""The pinned llama.cpp macOS installer, driven with tiny synthetic release tarballs and injected pins.

No network, no real llama-server and no dependency on the developer's downloaded archive: `--version` is a scripted
executor, `xattr` a scripted runner, and a download a fetch function that fails the test if it is ever called
without --download.
"""

import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from llmbench.coding.sandbox import WorkerResult
from llmbench.containers.native_prep import hash_native_server, read_install_manifest, verify_install_manifest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("install_llamacpp_macos", ROOT / "scripts" / "install_llamacpp_macos.py")
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer  # dataclasses resolve their module while the script executes
SPEC.loader.exec_module(installer)

VERSION = (b"version: 0.4.1-dev (build 11011, commit aa39d7a3e)\n"
           b"built with AppleClang 21.0.0.21000101 for Darwin arm64\n")
MAC = ("Darwin", "arm64")
POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX links and modes")
RELEASE = [("llama-test", "dir", None), ("llama-test/llama-server", "file", b"\xcf\xfa\xed\xfe server", 0o755),
           ("llama-test/libllama.0.1.dylib", "file", b"\xcf\xfa\xed\xfe libllama"),
           ("llama-test/libllama.0.dylib", "sym", "libllama.0.1.dylib"),
           ("llama-test/libllama.dylib", "sym", "libllama.0.dylib"),
           ("llama-test/LICENSE", "file", b"MIT License")]


def tarball(entries) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, kind, payload, *mode in entries:
            info = tarfile.TarInfo(name)
            info.mode = mode[0] if mode else (0o755 if kind == "dir" else 0o644)
            if kind == "file":
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                continue
            info.type = {"dir": tarfile.DIRTYPE, "sym": tarfile.SYMTYPE, "hard": tarfile.LNKTYPE,
                         "fifo": tarfile.FIFOTYPE, "chr": tarfile.CHRTYPE}[kind]
            if kind in ("sym", "hard"):
                info.linkname = payload
            archive.addfile(info)
    return buffer.getvalue()


def pin_for(data: bytes, **changes):
    return dataclasses.replace(installer.PINNED, asset="llama-test.tar.gz", url="https://example.invalid/llama-test.tar.gz",
                               sha256=hashlib.sha256(data).hexdigest(), size=len(data), top_directory="llama-test",
                               **changes)


class FakeVersion:
    """executor_factory + executor in one: records where it was created and exactly what it ran."""

    def __init__(self, stdout=b"", stderr=VERSION, status="completed", returncode=0):
        self.result, self.created, self.calls = WorkerResult(status, returncode, stdout, stderr), [], []

    def __call__(self, executable, cwd):
        self.created.append((Path(executable), Path(cwd)))
        return self

    def run(self, argv, *, timeout_seconds, max_output_bytes):
        executable = self.created[-1][0]
        assert argv == (str(executable), "--version") and executable.is_file()
        assert type(timeout_seconds) is int and timeout_seconds >= 1 and max_output_bytes >= 1
        self.calls.append(argv)
        return self.result


class FakeXattr:
    def __init__(self, quarantined=(), failure=None):
        self.calls, self.quarantined, self.failure = [], set(quarantined), failure

    def __call__(self, argv, **kwargs):
        assert kwargs["timeout"] >= 1 and kwargs["check"] is False and kwargs["capture_output"] is True
        self.calls.append(argv)
        if self.failure:
            return subprocess.CompletedProcess(argv, 1, b"", self.failure)
        if Path(argv[-1]).name in self.quarantined:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return subprocess.CompletedProcess(argv, 1, b"", f"xattr: {argv[-1]}: No such xattr: {argv[2]}".encode())


def no_network(url, max_bytes):
    raise AssertionError("the installer fetched without --download")


@pytest.fixture
def policy(tmp_path):
    path = tmp_path / "runtime-policy.json"
    path.write_text(json.dumps({"allow_native_execution": True, "reason": "test"}), encoding="utf-8")
    return path


def run(tmp_path, policy, data, *, pin=None, version=None, xattr=None, fetch=no_network, extra=None,
        host_platform=MAC, write_archive=True):
    archive = tmp_path / "llama-test.tar.gz"
    if write_archive:
        archive.write_bytes(data)
    args = extra if extra is not None else ["--archive", str(archive)]
    code = installer.main([*args, "--output", str(tmp_path / "out"), "--policy", str(policy)],
                          pin=pin or pin_for(data), fetch=fetch, executor_factory=version or FakeVersion(),
                          xattr_runner=xattr or FakeXattr(), host_platform=host_platform)
    return code


def leftovers(output: Path) -> list[str]:
    return sorted(path.name for path in output.iterdir()) if output.exists() else []


def test_pinned_release_is_the_audited_b11011_asset():
    pin = installer.PINNED
    assert pin.url == ("https://github.com/ggml-org/llama.cpp/releases/download/b11011/"
                       "llama-b11011-bin-macos-arm64.tar.gz")
    assert (pin.sha256, pin.size) == ("9f88854d8216454a883f6d970e52a888d85c1ff321086c08c3d2f69362e0154d", 11156605)
    assert (pin.tag, pin.commit, pin.build_info) == ("b11011", "aa39d7a3e145a88202793a89462d65e94a5fc25f",
                                                     "b11011-aa39d7a3e")
    assert pin.top_directory == "llama-b11011" and pin.runner == "macos-26"
    assert pin.release_workflow_sha256 == "e130ce78c37628a8adea0ce58e6de904ba0b6e5bdb85f8417374351c1f87583b"
    assert pin.upstream_build_flags == (
        "-DGGML_METAL_EMBED_LIBRARY=ON", "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3", "-DCMAKE_INSTALL_RPATH=@loader_path",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DLLAMA_FATAL_WARNINGS=ON", "-DLLAMA_BUILD_BORINGSSL=ON",
        "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON",
        "-DGGML_RPC=ON")
    assert "-DCMAKE_INSTALL_RPATH='@loader_path'" in pin.upstream_build_commands[0]
    assert pin.upstream_build_commands[1] == "cmake --build build --config Release"
    assert installer.DEFAULT_OUTPUT == "artifacts/native-runtime"


@POSIX
def test_install_extracts_verifies_and_records_a_manifest_prepare_accepts(tmp_path, policy, capsys):
    data = tarball(RELEASE)
    version, xattr = FakeVersion(), FakeXattr(quarantined={"llama-server"})
    assert run(tmp_path, policy, data, version=version, xattr=xattr) == 0
    target = (tmp_path / "out" / "llama-test").resolve()
    assert leftovers(tmp_path / "out") == ["llama-test"]  # the staging directory is gone
    assert (target / "llama-server").read_bytes() == b"\xcf\xfa\xed\xfe server"
    assert os.access(target / "llama-server", os.X_OK)
    assert os.readlink(target / "libllama.dylib") == "libllama.0.dylib"  # links stay links
    # quarantine: every regular file, never a link; only the quarantined one is reported as removed
    assert sorted(Path(argv[-1]).name for argv in xattr.calls) == ["LICENSE", "libllama.0.1.dylib", "llama-server"]
    assert all(argv[:3] == ["xattr", "-d", "com.apple.quarantine"] for argv in xattr.calls)
    assert len(version.calls) == 1 and version.created[0][0].name == "llama-server"
    manifest = json.loads((target / "install-manifest.json").read_text())
    assert manifest["files"] == {name: hashlib.sha256(payload).hexdigest()
                                 for name, kind, payload, *_ in RELEASE if kind == "file"
                                 for name in [name.split("/", 1)[1]]}
    assert manifest["links"] == {"libllama.0.dylib": "libllama.0.1.dylib", "libllama.dylib": "libllama.0.dylib"}
    assert (manifest["size"], manifest["sha256"], manifest["tag"], manifest["build_info"]) == (
        len(data), hashlib.sha256(data).hexdigest(), "b11011", "b11011-aa39d7a3e")
    assert manifest["compiler"] == "built with AppleClang 21.0.0.21000101 for Darwin arm64"
    assert manifest["version"] == "version: 0.4.1-dev (build 11011, commit aa39d7a3e)"
    assert manifest["upstream_build_flags"] == list(installer.UPSTREAM_BUILD_FLAGS)
    assert manifest["release_workflow_sha256"] == installer.RELEASE_WORKFLOW_SHA256
    summary = json.loads(capsys.readouterr().out)
    assert summary["reused_existing"] is False and summary["quarantine_removed"] == ["llama-server"]
    assert summary["llama_server"] == str(target / "llama-server")
    assert f"--llama-server {target / 'llama-server'}" in summary["next"]
    # the handoff: native preparation reads this manifest and finds the directory exactly as recorded
    executable = target / "llama-server"
    verify_install_manifest(read_install_manifest(target / "install-manifest.json"), executable,
                            hash_native_server(executable))


@POSIX
@pytest.mark.parametrize("tamper, fragment", [(lambda data: data + b"\x00", "has more than"),
                                              (lambda data: data[:-1], "bytes; the pinned"),
                                              (lambda data: data[:-1] + bytes([data[-1] ^ 1]), "SHA-256")])
def test_the_archive_is_verified_before_anything_is_extracted(tmp_path, policy, capsys, tamper, fragment):
    data = tarball(RELEASE)
    version = FakeVersion()
    archive = tmp_path / "llama-test.tar.gz"
    archive.write_bytes(tamper(data))
    assert run(tmp_path, policy, data, version=version, extra=["--archive", str(archive)], write_archive=False) == 2
    assert fragment in capsys.readouterr().err
    assert leftovers(tmp_path / "out") == [] and version.calls == []


@POSIX
@pytest.mark.parametrize("entry, fragment", [
    (("/etc/llama-evil", "file", b"x"), "absolute"),
    (("llama-test/../llama-evil", "file", b"x"), "absolute or climbs"),
    (("elsewhere/llama-server", "file", b"x"), "outside the release directory"),
    (("llama-test/libevil.dylib", "sym", "/usr/lib/libSystem.B.dylib"), "leaves the release directory"),
    (("llama-test/libevil.dylib", "sym", "../../outside.dylib"), "leaves the release directory"),
    (("llama-test/libhard.dylib", "hard", "llama-test/LICENSE"), "hard link, device or FIFO"),
    (("llama-test/fifo", "fifo", None), "hard link, device or FIFO"),
    (("llama-test/tty", "chr", None), "hard link, device or FIFO"),
    (("llama-test/LICENSE", "file", b"second copy"), "appears twice"),
    (("llama-test/setuid", "file", b"x", 0o4755), "set-id"),
])
def test_an_unsafe_member_refuses_the_whole_archive(tmp_path, policy, capsys, entry, fragment):
    data = tarball([*RELEASE, entry])
    version = FakeVersion()
    assert run(tmp_path, policy, data, version=version) == 2
    assert fragment in capsys.readouterr().err
    assert leftovers(tmp_path / "out") == [] and version.calls == []
    assert not (tmp_path / "llama-evil").exists() and not Path("/etc/llama-evil").exists()


@POSIX
@pytest.mark.parametrize("entries, fragment", [
    ([*RELEASE, ("llama-test/sub", "sym", "."), ("llama-test/sub/libx.dylib", "file", b"x")], "beneath a link"),
    ([("llama-test", "file", b"not a directory")], "must be a directory"),
    ([entry for entry in RELEASE if entry[0] != "llama-test/llama-server"], "no regular file llama-test/llama-server"),
    ([*[entry for entry in RELEASE if entry[0] != "llama-test/llama-server"],
      ("llama-test/llama-server", "sym", "libllama.dylib")], "no regular file llama-test/llama-server"),
])
def test_structural_refusals(tmp_path, policy, capsys, entries, fragment):
    data = tarball(entries)
    assert run(tmp_path, policy, data) == 2
    assert fragment in capsys.readouterr().err
    assert leftovers(tmp_path / "out") == []


@POSIX
@pytest.mark.parametrize("version, fragment", [
    (FakeVersion(stderr=VERSION.replace(b"11011", b"11010")), "reports build 11010"),
    (FakeVersion(stderr=VERSION.replace(b"aa39d7a3e", b"0123456789")), "commit 0123456789"),
    (FakeVersion(stderr=b"dyld: Library not loaded", returncode=6), "--version failed"),
    (FakeVersion(stderr=b"", status="timeout", returncode=None), "--version failed (timeout"),
    (FakeVersion(stderr=b"llama-server\n"), "no llama.cpp version line"),
])
def test_a_wrong_build_is_exit_3_and_nothing_is_promoted(tmp_path, policy, capsys, version, fragment):
    assert run(tmp_path, policy, tarball(RELEASE), version=version) == 3
    assert fragment in capsys.readouterr().err
    assert leftovers(tmp_path / "out") == []  # neither the install nor its staging directory survives


@POSIX
def test_a_version_line_behind_a_log_prefix_is_recorded_not_a_crash(tmp_path, policy, capsys):
    prefixed = b"main: version: 0.4.1-dev (build 11011, commit aa39d7a3e)\nbuilt with AppleClang 21 for Darwin arm64\n"
    assert run(tmp_path, policy, tarball(RELEASE), version=FakeVersion(stderr=prefixed)) == 0
    manifest = json.loads((tmp_path / "out" / "llama-test" / "install-manifest.json").read_text())
    assert manifest["version"] == "main: version: 0.4.1-dev (build 11011, commit aa39d7a3e)"
    assert manifest["build_info"] == "b11011-aa39d7a3e"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeResponse:
    """An HTTP response as the stdlib really behaves: each `recv` delivers what the server sent, `recv_seconds`
    after the last one; `read(n)` (a BufferedReader) keeps receiving until it has n bytes or EOF, while `read1(n)`
    returns after a single receive."""

    def __init__(self, sends, url="https://objects.example.invalid/asset", *, clock=None, recv_seconds=0.0):
        self.sends, self.url, self.clock, self.recv_seconds, self.recvs = list(sends), url, clock, recv_seconds, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def geturl(self):
        return self.url

    def _recv(self, size):
        self.recvs += 1
        if self.clock is not None:
            self.clock.now += self.recv_seconds
        if not self.sends:
            return b""
        block = self.sends.pop(0)
        if len(block) > size:
            self.sends.insert(0, block[size:])
        return block[:size]

    def read1(self, size):
        assert size >= 1
        return self._recv(size)

    def read(self, size):
        assert size >= 1
        data = b""
        while len(data) < size and (block := self._recv(size - len(data))):
            data += block
        return data


def opener_for(response, opened=None):
    def urlopen(request, *, timeout, context=None):
        if opened is not None:
            opened.append((request.full_url, timeout, context))
        return response
    return urlopen


def test_the_download_itself_is_bounded_in_bytes_time_and_scheme():
    url = "https://example.invalid/llama.tar.gz"
    opened = []
    assert installer.fetch_url(url, 6, opener=opener_for(FakeResponse([b"abc", b"def"]), opened)) == b"abcdef"
    assert [(item[0], item[1]) for item in opened] == [(url, installer.DOWNLOAD_TIMEOUT_SECONDS)]
    with pytest.raises(ValueError, match="exceeds the pinned 5 bytes"):
        installer.fetch_url(url, 5, opener=opener_for(FakeResponse([b"abc", b"def"])))
    with pytest.raises(ValueError, match="redirected off HTTPS"):
        installer.fetch_url(url, 6, opener=opener_for(FakeResponse([b"abc"], url="http://mirror.example.invalid/x")))


def test_a_trickling_server_cannot_hold_the_download_past_its_deadline():
    # one byte every 100 s: every receive is well inside the 120 s socket timeout, so only the overall deadline can
    # end it, and it must end it on time, not after a whole 1 MiB block has trickled in
    clock = Clock()
    response = FakeResponse([b"x"] * 100_000, clock=clock, recv_seconds=100.0)
    with pytest.raises(ValueError, match=r"did not finish within 1800 s \(18 bytes read\)"):
        installer.fetch_url("https://example.invalid/llama.tar.gz", 1_000_000, opener=opener_for(response),
                            clock=clock)
    assert clock.now <= installer.DOWNLOAD_DEADLINE_SECONDS + 100.0 and response.recvs == 18


def test_every_socket_read_takes_its_timeout_from_the_whole_download_deadline():
    import socket
    import ssl
    clock = Clock()
    deadline = installer.DownloadDeadline(1800, 120, clock)
    assert deadline.remaining() == 120  # the per-operation cap while plenty is left
    clock.now = 1750.0
    assert deadline.remaining() == 50.0  # then only what is left of the deadline
    # the TLS sockets the download opens arm themselves before every read and handshake (urllib reads the status
    # line and headers of every response, redirects included, through them)
    opened = []
    installer.fetch_url("https://example.invalid/a", 1, opener=opener_for(FakeResponse([]), opened), clock=clock,
                        deadline_seconds=50)
    context = opened[0][2]
    assert opened[0][1] == 50.0 and isinstance(context, ssl.SSLContext)  # the connect gets at most what is left
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname  # still the verifying default
    assert issubclass(context.sslsocket_class, ssl.SSLSocket) and ssl.SSLContext.sslsocket_class is ssl.SSLSocket
    clock.now = 0.0
    context = installer.DownloadDeadline(1800, 120, clock).context()
    with socket.socket() as raw, context.wrap_socket(raw, server_hostname="example.invalid",
                                                     do_handshake_on_connect=False) as tls:
        with pytest.raises(ValueError, match="closed or unwrapped"):  # armed first, then the (unconnected) read
            tls.read(1)
        assert tls.gettimeout() == 120
        clock.now = 1790.0
        with pytest.raises(OSError):
            tls.do_handshake()
        assert tls.gettimeout() == 10.0
        clock.now = 1800.0
        with pytest.raises(TimeoutError, match="did not finish within 1800 s"):
            tls.read(1)
        with pytest.raises(TimeoutError):
            tls.do_handshake()


@POSIX
def test_an_identical_existing_install_is_reused_and_the_run_is_idempotent(tmp_path, policy, capsys):
    data = tarball(RELEASE)
    assert run(tmp_path, policy, data) == 0
    target = tmp_path / "out" / "llama-test"
    recorded = (target / "install-manifest.json").read_bytes()
    capsys.readouterr()
    version = FakeVersion()
    assert run(tmp_path, policy, data, version=version) == 0
    assert json.loads(capsys.readouterr().out)["reused_existing"] is True
    assert (target / "install-manifest.json").read_bytes() == recorded
    assert version.created[0][0] == (target / "llama-server").resolve()  # re-verified in place
    (target / "install-manifest.json").unlink()  # an identical tree without a manifest gets one
    assert run(tmp_path, policy, data) == 0
    assert (target / "install-manifest.json").read_bytes() == recorded


@POSIX
@pytest.mark.parametrize("tamper, fragment", [
    (lambda target: (target / "libllama.0.1.dylib").write_bytes(b"patched"), "differs from the pinned"),
    (lambda target: (target / "libextra.dylib").write_bytes(b"new"), "differs from the pinned"),
    (lambda target: (target / "libllama.dylib").unlink(), "differs from the pinned"),
    (lambda target: (target / "install-manifest.json").write_text(
        json.dumps({**json.loads((target / "install-manifest.json").read_text()), "compiler": "gcc"})),
     "already records a different install"),
])
def test_a_different_existing_install_is_refused_and_left_untouched(tmp_path, policy, capsys, tamper, fragment):
    data = tarball(RELEASE)
    assert run(tmp_path, policy, data) == 0
    target = tmp_path / "out" / "llama-test"
    tamper(target)
    before = {path.name: (path.read_bytes() if path.is_file() and not path.is_symlink() else os.readlink(path)
                          if path.is_symlink() else None) for path in target.iterdir()}
    capsys.readouterr()
    assert run(tmp_path, policy, data) == 2
    assert fragment in capsys.readouterr().err
    after = {path.name: (path.read_bytes() if path.is_file() and not path.is_symlink() else os.readlink(path)
                         if path.is_symlink() else None) for path in target.iterdir()}
    assert after == before


UNREADABLE = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                reason="needs a file its owner cannot read")


@POSIX
@UNREADABLE
def test_a_stray_file_in_an_existing_install_is_refused_without_being_read(tmp_path, policy, capsys):
    # a model kept beside llama-server: the names already differ, so none of its bytes may be read (the file is
    # unreadable here, so a read would surface as "Permission denied" instead of the refusal)
    data = tarball(RELEASE)
    assert run(tmp_path, policy, data) == 0
    stray = tmp_path / "out" / "llama-test" / "model.gguf"
    stray.write_bytes(b"GGUF" + b"\x00" * 4096)
    stray.chmod(0)
    capsys.readouterr()
    version = FakeVersion()
    assert run(tmp_path, policy, data, version=version) == 2
    assert "exists and differs from the pinned llama-test.tar.gz" in capsys.readouterr().err
    assert version.calls == [] and stray.stat().st_mode & 0o777 == 0


@POSIX
@UNREADABLE
def test_an_existing_install_too_large_to_be_the_release_is_refused_before_anything_is_hashed(
        tmp_path, policy, capsys, monkeypatch):
    data = tarball(RELEASE)
    assert run(tmp_path, policy, data) == 0
    target = tmp_path / "out" / "llama-test"
    # the same names, but one file has grown past what any accepted archive unpacks to
    monkeypatch.setattr(installer, "MAX_UNPACKED_BYTES", 4096)
    grown = target / "libllama.0.1.dylib"
    grown.write_bytes(b"\x00" * 8192)
    grown.chmod(0)
    capsys.readouterr()
    assert run(tmp_path, policy, data) == 2
    assert "exists and differs from the pinned" in capsys.readouterr().err
    with pytest.raises(installer.TreeTooLarge, match="more than 4096 bytes of files"):
        installer.tree_contents(target)
    monkeypatch.setattr(installer, "MAX_UNPACKED_BYTES", 512 * 1024 * 1024)
    monkeypatch.setattr(installer, "MAX_MEMBERS", 3)  # the tree holds 5 entries besides its manifest
    with pytest.raises(installer.TreeTooLarge, match="more than 3 entries"):
        installer.tree_contents(target)


@POSIX
def test_tree_hashes_stream_through_the_no_follow_reader_never_a_whole_file_read(tmp_path, monkeypatch):
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "big.bin").write_bytes(b"\x01" * (3 * 1024 * 1024 + 7))
    (root / "link").symlink_to("sub/big.bin")
    expected = hashlib.sha256((root / "sub" / "big.bin").read_bytes()).hexdigest()

    def whole_file_read(self):
        raise AssertionError(f"{self} was read whole into memory")
    monkeypatch.setattr(Path, "read_bytes", whole_file_read)
    assert installer.tree_contents(root) == {"files": {"sub/big.bin": expected}, "links": {"link": "sub/big.bin"},
                                             "directories": ["sub"]}


@POSIX
@UNREADABLE
def test_an_unlistable_directory_in_an_existing_install_is_an_error_not_skipped(tmp_path):
    # os.walk skips a directory it cannot list unless told otherwise: an unlistable `sub/` holding a stray model
    # would then read as the release's empty `sub/`, and a tree that is not the release would compare equal
    root = tmp_path / "tree"
    hidden = root / "sub"
    hidden.mkdir(parents=True)
    (hidden / "model.gguf").write_bytes(b"GGUF")
    hidden.chmod(0)
    try:
        with pytest.raises(PermissionError):
            installer.tree_contents(root)
        with pytest.raises(PermissionError):
            installer.same_tree(root, {"files": {}, "links": {}, "directories": ["sub"]})
    finally:
        hidden.chmod(0o755)


def test_there_is_no_network_without_download(tmp_path, policy):
    with pytest.raises(SystemExit) as caught:  # a source is required; argparse refuses before anything runs
        installer.main(["--output", str(tmp_path / "out"), "--policy", str(policy)], fetch=no_network,
                       host_platform=MAC)
    assert caught.value.code == 2
    with pytest.raises(SystemExit):
        installer.main(["--archive", "x.tar.gz", "--download"], fetch=no_network, host_platform=MAC)
    with pytest.raises(ValueError, match="non-HTTPS"):
        installer.fetch_url("http://example.invalid/llama.tar.gz", 10)


@POSIX
def test_download_is_explicit_bounded_verified_and_saved(tmp_path, policy, capsys):
    data = tarball(RELEASE)
    pin, calls = pin_for(data), []

    def fetch(url, max_bytes):
        calls.append((url, max_bytes))
        return data

    assert run(tmp_path, policy, data, pin=pin, fetch=fetch, extra=["--download"], write_archive=False) == 0
    saved = tmp_path / "out" / "downloads" / "llama-test.tar.gz"
    assert calls == [(pin.url, pin.size)] and saved.read_bytes() == data
    assert json.loads(capsys.readouterr().out)["archive"] == str(saved.resolve())
    # a verified saved archive is reused: the second --download does not fetch again
    assert run(tmp_path, policy, data, pin=pin, fetch=no_network, extra=["--download"], write_archive=False) == 0


@POSIX
def test_a_tampered_download_is_refused_and_not_saved(tmp_path, policy, capsys):
    data = tarball(RELEASE)
    code = run(tmp_path, policy, data, fetch=lambda url, size: data[:-1] + b"\x01", extra=["--download"],
               write_archive=False)
    assert code == 2 and "SHA-256" in capsys.readouterr().err
    assert not (tmp_path / "out" / "downloads" / "llama-test.tar.gz").exists()
    assert not (tmp_path / "out" / "llama-test").exists()


@POSIX
@pytest.mark.parametrize("content", [None, {"allow_container_execution": True, "allow_inference": True}])
def test_running_the_executable_needs_the_native_policy_first(tmp_path, capsys, content):
    policy = tmp_path / "runtime-policy.json"
    if content is not None:
        policy.write_text(json.dumps(content), encoding="utf-8")
    version = FakeVersion()
    assert run(tmp_path, policy, tarball(RELEASE), version=version) == 2
    assert "native forbidden" in capsys.readouterr().err
    assert version.calls == [] and leftovers(tmp_path / "out") == []


@POSIX
def test_an_xattr_failure_other_than_absence_refuses(tmp_path, policy, capsys):
    xattr = FakeXattr(failure=b"xattr: [Errno 1] Operation not permitted")
    assert run(tmp_path, policy, tarball(RELEASE), xattr=xattr) == 2
    assert "Operation not permitted" in capsys.readouterr().err
    assert leftovers(tmp_path / "out") == []


@pytest.mark.parametrize("host", [("Linux", "x86_64"), ("Darwin", "x86_64")])
def test_other_hosts_are_refused(tmp_path, policy, capsys, host):
    version = FakeVersion()
    assert run(tmp_path, policy, tarball(RELEASE), version=version, host_platform=host) == 2
    assert "macOS arm64 only" in capsys.readouterr().err and version.calls == []
