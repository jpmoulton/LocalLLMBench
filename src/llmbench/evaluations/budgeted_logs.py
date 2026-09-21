"""Quota-backed Inspect JSON logs and diagnostics.

The public fsspec log-directory hook meters every persistent log write. JSON is
selected deliberately: Inspect's .eval recorder first spools an unmetered temporary
ZIP. The auxiliary logger/view hooks below are a narrow, serial, version-tested
compatibility boundary; they do not alter tasks, model responses, or scoring.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import io
import json
import logging
import sys
from pathlib import Path, PurePosixPath
import threading
from typing import Any, Callable, Iterator
from uuid import uuid4

from fsspec import register_implementation
from fsspec.spec import AbstractFileSystem

from .capture import UnsafeCaptureRuntimeState

_PROTOCOL = "llmbenchlog"
_sessions: dict[str, "InspectLogSession"] = {}
_sessions_lock = threading.RLock()


class InspectLogSession:
    def __init__(self, artifacts: Any, on_failure: Callable[[BaseException], None]) -> None:
        if not callable(getattr(artifacts, "open_external", None)) or not callable(
                getattr(artifacts, "remove_external", None)):
            raise ValueError("Inspect requires an artifact writer with metered external logs")
        self.artifacts, self.on_failure = artifacts, on_failure
        self.token = uuid4().hex
        self.url = f"{_PROTOCOL}://{self.token}/inspect"
        self.failure: str | None = None
        self.handles: list[Any] = []
        self._lock = threading.RLock()

    def check(self) -> None:
        if self.failure:
            raise UnsafeCaptureRuntimeState(self.failure)

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            first = self.failure is None
            self.failure = self.failure or f"Inspect log persistence failed: {type(exc).__name__}: {exc}"
        if first:
            self.on_failure(exc)

    def open(self, relative: str, mode: str) -> Any:
        self.check()
        try:
            handle = _GuardedLogFile(self.artifacts.open_external(relative, mode), self)
        except BaseException as exc:
            self.fail(exc)
            raise
        self.handles.append(handle)
        return handle

    def local_path(self, uri: str) -> str:
        prefix = f"{_PROTOCOL}://{self.token}/"
        if not uri.startswith(prefix):
            raise ValueError("Inspect returned a log outside its assigned artifact namespace")
        return str(Path(self.artifacts.root) / uri[len(prefix):])

    def close(self) -> None:
        for handle in self.handles:
            try:
                handle.close()
            except BaseException as exc:
                self.fail(exc)


class _GuardedLogFile(io.BufferedIOBase):
    def __init__(self, inner: Any, session: InspectLogSession) -> None:
        self.inner, self.session = inner, session

    def _call(self, method: str, *args: Any) -> Any:
        if method not in {"close", "flush"}:
            self.session.check()
        try:
            return getattr(self.inner, method)(*args)
        except BaseException as exc:
            self.session.fail(exc)
            raise

    def writable(self) -> bool:
        return self.inner.writable()

    def readable(self) -> bool:
        return self.inner.readable()

    def seekable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        return self._call("write", data)

    def read(self, size: int = -1) -> bytes:
        return self._call("read", size)

    def readinto(self, buffer: Any) -> int:
        return self._call("readinto", buffer)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._call("seek", offset, whence)

    def tell(self) -> int:
        return self._call("tell")

    def truncate(self, size: int | None = None) -> int:
        return self._call("truncate", size)

    def flush(self) -> None:
        if not self.inner.closed:
            self._call("flush")

    def close(self) -> None:
        if not self.closed:
            try:
                self._call("close")
            finally:
                super().close()


class InspectArtifactFileSystem(AbstractFileSystem):
    protocol = _PROTOCOL
    cachable = False

    @staticmethod
    def _resolve(path: str) -> tuple[InspectLogSession, str, Path]:
        path = InspectArtifactFileSystem._strip_protocol(path)
        token, separator, relative = path.partition("/")
        with _sessions_lock:
            session = _sessions.get(token)
        if session is None:
            raise FileNotFoundError("expired or unknown Inspect log namespace")
        pure = PurePosixPath(relative)
        if (not separator or not relative or str(pure) != relative or pure.is_absolute()
                or "\\" in relative or ":" in relative or any(ord(c) < 32 for c in relative)
                or any(part in {"", ".", ".."} for part in pure.parts) or pure.parts[0] != "inspect"):
            raise ValueError("Inspect log path must stay inside its assigned prefix")
        root = Path(session.artifacts.root).resolve()
        local = root.joinpath(*pure.parts)
        if not local.resolve().is_relative_to(root):
            raise ValueError("Inspect log path escapes its artifact directory")
        for parent in (local, *local.parents):
            if parent == root:
                break
            if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                raise ValueError("Inspect log path traverses a linked entry")
        return session, relative, local

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        session, relative, _ = self._resolve(path)
        return session.open(relative, mode)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        _, _, local = self._resolve(path)
        local.mkdir(parents=True, exist_ok=exist_ok)

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        _, _, local = self._resolve(path)
        stat = local.stat()
        return {"name": self._strip_protocol(path), "size": stat.st_size,
                "type": "directory" if local.is_dir() else "file", "mtime": stat.st_mtime}

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        _, _, local = self._resolve(path)
        entries = [self.info(self._strip_protocol(path).rstrip("/") + "/" + item.name)
                   for item in local.iterdir()]
        return entries if detail else [item["name"] for item in entries]

    def rm_file(self, path: str) -> None:
        session, relative, _ = self._resolve(path)
        try:
            session.artifacts.remove_external(relative)
        except BaseException as exc:
            session.fail(exc)
            raise


class _DiagnosticHandler(logging.Handler):
    def __init__(self, session: InspectLogSession, relative: str) -> None:
        super().__init__()
        self.session, self.relative = session, relative
        self.stream: Any = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.session.failure:
            return  # admission is already poisoned; avoid recursive logging failures
        try:
            if self.stream is None:
                self.stream = self.session.open(self.relative, "ab")
            self.stream.write((self.format(record) + "\n").encode("utf-8"))
            self.stream.flush()
        except BaseException as exc:
            self.session.fail(exc)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
        super().close()


@contextmanager
def bounded_inspect_logs(artifacts: Any, on_failure: Callable[[BaseException], None]) -> Iterator[InspectLogSession]:
    """Call only inside inspect_runtime_directory's serial compatibility lock."""
    from unittest.mock import patch
    import inspect_ai._util.logger as inspect_logger
    import inspect_ai._eval.task.run as task_run
    from inspect_ai._util.trace import TraceFormatter

    session = InspectLogSession(artifacts, on_failure)
    register_implementation(_PROTOCOL, InspectArtifactFileSystem, clobber=True)
    with _sessions_lock:
        _sessions[session.token] = session
    old_handler = inspect_logger._logHandler["handler"]
    diagnostics: list[_DiagnosticHandler] = []

    def handler_factory(*args: Any, **kwargs: Any) -> _DiagnosticHandler:
        handler = _DiagnosticHandler(session, f"inspect/runtime/diagnostics-{len(diagnostics)}.log")
        diagnostics.append(handler)
        return handler

    def notify(location: str) -> None:
        with session.open("inspect/runtime/last-eval.json", "wb") as handle:
            handle.write(json.dumps({"location": session.local_path(location)}).encode("utf-8"))

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(inspect_logger, "FileHandler", handler_factory))
            stack.enter_context(patch.object(inspect_logger, "rotate_trace_files", lambda *a, **k: None))
            stack.enter_context(patch.object(inspect_logger, "compress_trace_log", lambda *a, **k: lambda: None))
            stack.enter_context(patch.object(task_run, "view_notify_eval", notify))
            if old_handler is not None:
                trace = handler_factory()
                trace.setFormatter(TraceFormatter())
                stack.enter_context(patch.object(old_handler, "trace_logger", trace))
                if old_handler.file_logger is not None:
                    stack.enter_context(patch.object(old_handler, "file_logger", handler_factory()))
            yield session
            session.check()
    finally:
        if old_handler is None:
            created = inspect_logger._logHandler["handler"]
            if created is not None:
                for logger in [logging.getLogger(), *list(logging.Logger.manager.loggerDict.values())]:
                    if isinstance(logger, logging.Logger):
                        logger.removeHandler(created)
                created.close()
            inspect_logger._logHandler["handler"] = None
        session.close()
        with _sessions_lock:
            _sessions.pop(session.token, None)
        if sys.exc_info()[0] is None:
            session.check()  # close/fsync failures are evidence failures too
