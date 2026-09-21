"""Confined, byte-budgeted artifacts with bounded terminal reserve and external-writer support."""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath

from ..config import canonical_json

INDEX_NAME = "artifact-index.json"
MAX_ARTIFACT_PATH = 512


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_plain) + "\n").encode()


class RunArtifacts:
    def __init__(self, root: str | Path, max_bytes: int = 1_073_741_824, *, terminal_reserve_bytes: int = 0) -> None:
        if (type(max_bytes) is not int or max_bytes < 1 or type(terminal_reserve_bytes) is not int
                or not 0 <= terminal_reserve_bytes < max_bytes):
            raise ValueError("artifact and terminal byte limits must be valid positive integers")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.terminal_reserve_bytes = max_bytes, terminal_reserve_bytes
        self._entries: dict[str, dict] = {}
        self._paths: set[str] = set()
        self._charged_paths = {p.relative_to(self.root).as_posix(): p.stat().st_size
                               for p in self.root.rglob("*") if p.is_file()}
        self._bytes = sum(self._charged_paths.values())
        self._external: set[str] = set()
        self._handles: dict[str, _BudgetedFile] = {}
        self._mutex = threading.RLock()
        self._sealed = False
        if self._bytes > max_bytes - terminal_reserve_bytes:
            raise ValueError("existing files exceed artifact data budget")

    @property
    def remaining_bytes(self) -> int:
        with self._mutex:
            return max(0, self.max_bytes - self.terminal_reserve_bytes - self._bytes)

    def _target(self, relative: str) -> tuple[str, Path]:
        pure = PurePosixPath(relative) if type(relative) is str else None
        if (pure is None or not relative or len(relative) > MAX_ARTIFACT_PATH or "\\" in relative
                or ":" in relative or any(ord(c) < 32 for c in relative) or pure.is_absolute()
                or any(part in {"", ".", ".."} for part in pure.parts) or str(pure) != relative):
            raise ValueError("artifact path must be a safe relative POSIX path")
        target = self.root.joinpath(*pure.parts)
        for parent in (target.parent, *target.parent.parents):
            if parent == self.root:
                break
            if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                raise ValueError("artifact path escapes through a linked directory")
        if not target.parent.resolve().is_relative_to(self.root):
            raise ValueError("artifact path escapes the run directory")
        if target.is_symlink() or getattr(target, "is_junction", lambda: False)():
            raise ValueError("artifact target cannot be a link")
        target.parent.mkdir(parents=True, exist_ok=True)
        return relative, target

    def _check_open(self) -> None:
        if self._sealed:
            raise ValueError("artifact store is finalized")

    def _reserve_name(self, name: str) -> None:
        self._check_open()
        if self.terminal_reserve_bytes and name in {"result.json", INDEX_NAME}:
            raise ValueError("terminal artifact names are reserved")
        names = self._paths | {name}
        if self.terminal_reserve_bytes:
            # Reserve an index entry before creating even an empty file. Many tiny logs cannot exhaust metadata.
            prospective = [{"path": n, "sha256": "0" * 64, "size": self.max_bytes}
                           for n in sorted(names | {"result.json"})]
            if len(_json_bytes({"schema_version": 1, "artifacts": prospective})) > self.terminal_reserve_bytes // 2:
                raise ValueError("artifact index metadata budget exhausted")
        self._paths.add(name)

    def _resize(self, name: str, size: int) -> None:
        self._check_open()
        before = self._charged_paths.get(name, 0)
        if self._bytes + size - before > self.max_bytes - self.terminal_reserve_bytes:
            raise ValueError("artifact budget exhausted")
        self._bytes += size - before
        self._charged_paths[name] = size

    def write(self, relative: str, content: bytes) -> Path:
        if type(content) is not bytes:
            raise TypeError("artifact content must be bytes")
        with self._mutex:
            name, target = self._target(relative)
            if target.exists():
                raise FileExistsError(str(target))
            self._reserve_name(name)
            self._resize(name, len(content))
            try:
                with target.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                actual = target.stat().st_size if target.exists() else 0
                self._resize(name, actual)
                raise
            self._entries[name] = {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
            return target

    def write_json(self, relative: str, value) -> Path:
        return self.write(relative, _json_bytes(value))

    @contextmanager
    def trace(self, relative: str):
        name, target = self._target(relative)
        with self._mutex:
            self._reserve_name(name)
            handle = target.open("xb")
        digest, count = hashlib.sha256(), 0
        with handle:
            def write(event) -> None:
                nonlocal count
                line = (canonical_json(_plain(event) if is_dataclass(event) else event) + "\n").encode("utf-8")
                with self._mutex:
                    self._resize(name, count + len(line))
                    handle.write(line)
                    handle.flush()
                    digest.update(line)
                    count += len(line)
            try:
                yield write
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                self._entries[name] = {"path": name, "sha256": digest.hexdigest(), "size": count}

    def scoped(self, prefix: str) -> "ScopedArtifacts":
        self._target(prefix + "/x")
        return ScopedArtifacts(self, prefix)

    # ---- child allocations (evaluator container writes its own tree) -----------------------------------
    def _child_key(self, prefix: str) -> str:
        if (type(prefix) is not str or "/" in prefix or prefix in {"result.json", INDEX_NAME}
                or prefix.endswith(".json")):
            raise ValueError("child prefix must be one plain directory name, never a terminal artifact")
        self._target(prefix + "/x")  # same confinement and link checks as every artifact path
        return prefix + "//reserved"  # a double slash can never name a real artifact

    def reserve_child(self, prefix: str, max_bytes: int) -> None:
        """Charge a child's full allocation now; the child process writes `<root>/<prefix>` unsupervised."""
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("child allocation must be a positive integer")
        with self._mutex:
            key = self._child_key(prefix)
            if key in self._charged_paths:
                raise ValueError("child allocation already reserved")
            if any(name.startswith(prefix + "/") for name in self._entries):
                raise ValueError("child prefix already holds indexed artifacts")
            self._resize(key, max_bytes)

    def adopt_child(self, prefix: str) -> dict:
        """Release the placeholder, then index what the child actually wrote, never beyond the cap.

        Returns {"bytes", "files", "over_allocation", "unindexed", "errors"}; over-allocation and errors are the
        caller's failure to report. Files that would breach the cap stay unindexed rather than escaping it; a file
        that cannot be read (OSError) is counted by its stat size, left unindexed and listed under "errors", so one
        bad file never hides the allocation verdict. Only call this once the child is known to have stopped.
        """
        with self._mutex:
            self._check_open()
            key = self._child_key(prefix)
            if key not in self._charged_paths:
                raise ValueError("no child allocation to adopt")
            allocation = self._charged_paths[key]
            self._resize(key, 0)
            self._charged_paths.pop(key, None)
            base = self.root / prefix
            total, files, unindexed, errors = 0, 0, [], []
            for path in sorted(base.rglob("*")) if base.is_dir() else []:
                name = path.relative_to(self.root).as_posix()
                if name in self._entries:
                    continue
                try:
                    if path.is_symlink() or getattr(path, "is_junction", lambda: False)() or not path.is_file():
                        continue
                    size = path.stat().st_size
                except OSError as exc:
                    unindexed.append(name)
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
                    continue
                total += size
                try:
                    self._target(name)
                    self._reserve_name(name)
                    if self._bytes + size > self.max_bytes - self.terminal_reserve_bytes:
                        raise ValueError("artifact budget exhausted")
                    self._seal_external(name)
                    files += 1
                except ValueError:
                    self._paths.discard(name)
                    unindexed.append(name)
                except OSError as exc:  # open/hash/stat failed before anything was charged or indexed
                    self._paths.discard(name)
                    unindexed.append(name)
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return {"bytes": total, "files": files, "over_allocation": total > allocation,
                    "unindexed": unindexed, "errors": errors}

    def open_external(self, relative: str, mode: str = "wb") -> "_BudgetedFile":
        """Binary file for an integrated external logger. Growth is charged before every disk write.

        Only files first created through this API may be reopened/truncated; ordinary sealed evidence is immutable.
        No raw descriptor is exposed. Seeking and ZIP header rewrites work without bypassing the byte cap.
        """
        if mode not in {"wb", "w+b", "wb+", "xb", "x+b", "xb+", "ab", "a+b", "ab+", "rb", "r+b", "rb+"}:
            raise ValueError("external logs require a supported binary mode")
        with self._mutex:
            name, target = self._target(relative)
            self._reserve_name(name)
            if name in self._handles and not self._handles[name].closed:
                raise ValueError("external artifact already has an active writer")
            exists = target.exists()
            if exists and name not in self._external:
                raise FileExistsError("external writer cannot replace sealed evidence")
            if exists and (not target.is_file() or target.stat().st_nlink != 1):
                raise ValueError("external artifact must be a regular unlinked file")
            if "x" in mode and exists:
                raise FileExistsError(str(target))
            if mode.startswith("r") and not exists:
                raise FileNotFoundError(str(target))
            self._external.add(name)
            raw = target.open("r+b" if exists else "x+b")
            if mode.startswith("w"):
                raw.truncate(0)
                self._resize(name, 0)
            elif mode.startswith("a"):
                raw.seek(0, io.SEEK_END)
            self._resize(name, target.stat().st_size)
            stream = _BudgetedFile(self, name, raw, mode)
            self._handles[name] = stream
            return stream

    def remove_external(self, relative: str) -> None:
        """Delete only a log registered through this API, releasing its charged data and index capacity."""
        with self._mutex:
            self._check_open()
            name, target = self._target(relative)
            if name not in self._external:
                raise FileNotFoundError("not a registered external log")
            handle = self._handles.get(name)
            if handle is not None:
                handle.close()
            if target.exists() and (not target.is_file() or target.stat().st_nlink != 1):
                raise ValueError("external artifact must be a regular unlinked file")
            target.unlink(missing_ok=True)
            self._resize(name, 0)
            self._entries.pop(name, None)
            self._paths.discard(name)
            self._charged_paths.pop(name, None)
            self._external.discard(name)
            self._handles.pop(name, None)

    def _seal_external(self, name: str) -> None:
        target = self._target(name)[1]
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        size = target.stat().st_size
        self._resize(name, size)
        self._entries[name] = {"path": name, "sha256": digest.hexdigest(), "size": size}

    def adopt_tree(self, relative: str = "") -> None:
        """Reconcile trusted files; admission enforces the budget rather than silently adopting overflow."""
        with self._mutex:
            base = self._target(relative)[1] if relative else self.root
            for path in sorted(base.rglob("*")) if base.is_dir() else []:
                name = path.relative_to(self.root).as_posix()
                if name in self._entries or name == INDEX_NAME or not path.is_file():
                    continue
                self._target(name)
                self._reserve_name(name)
                self._seal_external(name)

    def index(self) -> list[dict]:
        with self._mutex:
            return [dict(self._entries[name]) for name in sorted(self._entries)]

    def write_index(self) -> Path:
        """Unreserved standalone writer; the index itself still consumes the configured total limit."""
        if self.terminal_reserve_bytes:
            raise ValueError("reserved terminal metadata must be written with finalize_terminal")
        content = _json_bytes({"schema_version": 1, "artifacts": self.index()})
        target = self.write(INDEX_NAME, content)
        self._entries.pop(INDEX_NAME, None)  # An index cannot contain its own hash.
        return target

    def seal_external_logs(self) -> None:
        with self._mutex:
            for handle in self._handles.values():
                handle.close()

    def _terminal_content(self, result) -> tuple[bytes, bytes]:
        payload = _json_bytes(result)
        entry = {"path": "result.json", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        index = _json_bytes({"schema_version": 1, "artifacts": sorted([*self.index(), entry], key=lambda e: e["path"])})
        return payload, index

    def terminal_fits(self, result) -> bool:
        payload, index = self._terminal_content(result)
        return self._bytes + len(payload) + len(index) <= self.max_bytes

    def finalize_terminal(self, result) -> None:
        """Use only the pre-reserved bounded terminal capacity; neither result nor index may escape the cap."""
        with self._mutex:
            self._check_open()
            for handle in self._handles.values():
                handle.close()
            payload, index = self._terminal_content(result)
            if self._bytes + len(payload) + len(index) > self.max_bytes:
                raise ValueError("terminal metadata exceeds its reserved byte budget")
            for name, content in (("result.json", payload), (INDEX_NAME, index)):
                target = self._target(name)[1]
                with target.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._bytes += len(content)
            self._sealed = True


class _BudgetedFile(io.BufferedIOBase):
    def __init__(self, parent, name, raw, mode):
        self._parent, self._name, self._raw, self._mode = parent, name, raw, mode

    @property
    def name(self):
        return self._name

    def writable(self):
        return self._mode[0] != "r" or "+" in self._mode

    def readable(self):
        return self._mode[0] == "r" or "+" in self._mode

    def seekable(self):
        return True

    def tell(self):
        return self._raw.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        return self._raw.seek(offset, whence)

    def read(self, size=-1):
        if not self.readable():
            raise io.UnsupportedOperation("not readable")
        return self._raw.read(size)

    def readinto(self, buffer):
        return self._raw.readinto(buffer)

    def write(self, data):
        if not self.writable():
            raise io.UnsupportedOperation("not writable")
        content = bytes(data)
        with self._parent._mutex:
            if self._mode.startswith("a"):
                self._raw.seek(0, io.SEEK_END)
            size = max(self._parent._charged_paths.get(self._name, 0), self._raw.tell() + len(content))
            self._parent._resize(self._name, size)
            try:
                written = self._raw.write(content)
                self._raw.flush()
                return written
            except BaseException:
                self._parent._resize(self._name, os.fstat(self._raw.fileno()).st_size)
                raise

    def truncate(self, size=None):
        if not self.writable():
            raise io.UnsupportedOperation("not writable")
        size = self._raw.tell() if size is None else size
        if type(size) is not int or size < 0:
            raise ValueError("invalid truncate size")
        with self._parent._mutex:
            self._parent._resize(self._name, size)
            try:
                return self._raw.truncate(size)
            except BaseException:
                self._parent._resize(self._name, os.fstat(self._raw.fileno()).st_size)
                raise

    def flush(self):
        if not self._raw.closed:
            self._raw.flush()

    def close(self):
        if self.closed:
            return
        with self._parent._mutex:
            try:
                self._raw.flush()
                os.fsync(self._raw.fileno())
                self._parent._seal_external(self._name)
            finally:
                self._raw.close()
                super().close()


class ScopedArtifacts:
    """A subdirectory view sharing the parent index, terminal reservation and byte budget."""
    def __init__(self, parent: RunArtifacts, prefix: str) -> None:
        self.parent, self.prefix = parent, prefix
        self.root = parent.root / prefix

    @property
    def remaining_bytes(self):
        return self.parent.remaining_bytes

    def write(self, relative: str, content: bytes) -> Path:
        return self.parent.write(f"{self.prefix}/{relative}", content)

    def trace(self, relative: str):
        return self.parent.trace(f"{self.prefix}/{relative}")

    def open_external(self, relative: str, mode: str = "wb"):
        return self.parent.open_external(f"{self.prefix}/{relative}", mode)

    def remove_external(self, relative: str):
        return self.parent.remove_external(f"{self.prefix}/{relative}")


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)
