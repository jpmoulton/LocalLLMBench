"""Bounded HTTP and SSE transport, with no retry or redirect behavior."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .base import BackendError


MAX_ERROR_BODY_BYTES = 64 * 1024


class TransportError(BackendError):
    """``status``/``body`` are evidence attributes; bodies never enter the message text."""

    def __init__(self, message: str, *, status: int | None = None, body: bytes | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = None if body is None else bytes(body[:MAX_ERROR_BODY_BYTES])


class TransportDeadlineExceeded(TimeoutError):
    """An absolute request deadline expired; server idleness is not established."""

    abort_campaign = True


class _RequestDeadline:
    """Own one HTTP response and close it if its waiting caller abandons it.

    urllib may be blocked before it exposes a response (for example in headers).
    The caller still returns at its deadline; a late response is immediately closed.
    The transport is poisoned, so that worker can never authorize another request.
    """

    def __init__(self, deadline: float, clock: Callable[[], float]) -> None:
        self.deadline, self.clock = deadline, clock
        self._response: Any = None
        self._expired = False
        self._lock = threading.Lock()

    def remaining(self) -> float:
        remaining = self.deadline - self.clock()
        if self._expired or remaining <= 0:
            raise TransportDeadlineExceeded("HTTP absolute request deadline exhausted; server idleness is unverified")
        return remaining

    def own(self, response: Any) -> None:
        with self._lock:
            self._response = response
            expired = self._expired or self.clock() >= self.deadline
        if expired:
            self._close(response)
            raise TransportDeadlineExceeded("HTTP response arrived after its absolute deadline")

    @staticmethod
    def _close(response: Any) -> None:
        # Interrupt the owned socket before close(), which can otherwise wait for
        # a buffered read's lock. Never discover or touch another request's socket.
        try:
            raw = getattr(getattr(response, "fp", None), "raw", None)
            owned_socket = getattr(raw, "_sock", None)
            if owned_socket is not None:
                owned_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            response.close()
        except Exception:
            pass

    def expire(self) -> None:
        with self._lock:
            self._expired = True
            response = self._response
        if response is not None:
            threading.Thread(target=self._close, args=(response,), daemon=True,
                             name="llmbench-http-deadline-close").start()


class EventStream(Protocol):
    def __iter__(self) -> Iterator[tuple[str, dict[str, Any] | str]]: ...
    def close(self) -> None: ...


class HTTPTransport(Protocol):
    def request_json(self, method: str, path: str,
                     payload: Mapping[str, Any] | None = None) -> dict[str, Any]: ...
    def stream(self, path: str, payload: Mapping[str, Any]) -> EventStream: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


class SSEStream:
    """An owned response; closing it does not certify server-side cancellation."""

    def __init__(self, response: Any, *, max_event_bytes: int = 4 * 1024 * 1024) -> None:
        self.response = response
        self.max_event_bytes = max_event_bytes
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.response.close()

    @staticmethod
    def _decode(event: str, parts: list[str]) -> tuple[str, dict[str, Any] | str]:
        data = "\n".join(parts)
        if data == "[DONE]":
            return "done", data
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise TransportError("Malformed JSON in streaming response") from exc
        if not isinstance(parsed, dict):
            raise TransportError("Streaming event JSON must be an object")
        return event or "message", parsed

    def __iter__(self) -> Iterator[tuple[str, dict[str, Any] | str]]:
        event, parts, size = "", [], 0
        try:
            while not self.closed:
                line = self.response.readline(self.max_event_bytes + 1)
                if not line:
                    if parts:
                        yield self._decode(event, parts)
                    break
                size += len(line)
                if size > self.max_event_bytes:
                    raise TransportError("Streaming event exceeds the configured byte limit")
                try:
                    text = line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise TransportError("Non-UTF-8 streaming response") from exc
                if not text:
                    if parts:
                        decoded = self._decode(event, parts)
                        yield decoded
                        if decoded[0] == "done":
                            return
                    event, parts, size = "", [], 0
                elif text.startswith(":"):
                    continue
                else:
                    field, _, value = text.partition(":")
                    value = value.removeprefix(" ")
                    if field == "data":
                        parts.append(value)
                    elif field == "event":
                        event = value
        except (OSError, TimeoutError) as exc:
            raise TransportError("Streaming connection failed or timed out") from exc
        finally:
            self.close()


class UrllibHTTPTransport:
    """A local-first transport. Constructor performs no network activity."""

    def __init__(self, base_url: str = "http://127.0.0.1:1234", *,
                 api_key: str | None = None, timeout: float = 30.0,
                 max_response_bytes: int = 16 * 1024 * 1024,
                 allow_remote: bool = False) -> None:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("base_url must be an HTTP(S) origin without credentials or a path")
        if not allow_remote and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote inference requires explicit allow_remote=True")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self._opener = build_opener(_NoRedirect())
        self._deadline_failed = False

    def _open(self, method: str, path: str, payload: Mapping[str, Any] | None,
              *, streaming: bool = False, deadline: _RequestDeadline | None = None) -> Any:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("HTTP paths must be origin-relative")
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {"Accept": "text/event-stream" if streaming else "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            timeout = min(self.timeout, deadline.remaining()) if deadline is not None else self.timeout
            response = self._opener.open(request, timeout=timeout)
            if deadline is not None:
                deadline.own(response)
            return response
        except HTTPError as exc:
            # Deliberately omit raw response bodies, which can contain prompts or tokens,
            # from the message. A bounded copy is kept as an attribute for explicit probes.
            try:
                if deadline is not None:
                    deadline.own(exc)
                body = exc.read(MAX_ERROR_BODY_BYTES)
            except Exception:
                body = None
            finally:
                exc.close()
            raise TransportError(f"HTTP {exc.code} from the inference server {path}", status=exc.code, body=body) from exc
        except TransportDeadlineExceeded:
            raise
        except (URLError, OSError, TimeoutError) as exc:
            reason = exc.reason if isinstance(exc, URLError) else exc
            if deadline is not None and isinstance(reason, TimeoutError):
                raise TransportDeadlineExceeded("HTTP socket timed out; server idleness is unverified") from exc
            raise TransportError(f"Cannot reach the inference server {path}") from exc

    def _bounded(self, operation: Callable[[_RequestDeadline | None], Any], *,
                 deadline_monotonic: float | None, clock: Callable[[], float]) -> Any:
        if self._deadline_failed:
            raise TransportDeadlineExceeded("HTTP transport is poisoned by a prior deadline; use a new owned server run")
        if deadline_monotonic is None:
            return operation(None)
        if not math.isfinite(deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite")
        scope = _RequestDeadline(min(deadline_monotonic, clock() + self.timeout), clock)
        wait = scope.remaining()
        finished = threading.Event()
        result: list[tuple[bool, Any]] = []

        def work() -> None:
            try:
                scope.remaining()
                result.append((True, operation(scope)))
            except BaseException as exc:
                result.append((False, exc))
            finally:
                finished.set()

        worker = threading.Thread(target=work, name="llmbench-http-request", daemon=True)
        worker.start()
        try:
            completed = finished.wait(wait)
        except BaseException:
            self._deadline_failed = True
            scope.expire()
            raise
        if not completed or clock() >= scope.deadline:
            self._deadline_failed = True
            scope.expire()
            raise TransportDeadlineExceeded("HTTP absolute request deadline exhausted; server idleness is unverified")
        succeeded, value = result[0]
        if not succeeded:
            reason = value.reason if isinstance(value, URLError) else value
            if isinstance(reason, TimeoutError):
                self._deadline_failed = True
                scope.expire()
                if not isinstance(value, TransportDeadlineExceeded):
                    raise TransportDeadlineExceeded("HTTP response timed out; server idleness is unverified") from value
            raise value
        return value

    def request_any(self, method: str, path: str, payload: Mapping[str, Any] | None = None, *,
                    deadline_monotonic: float | None = None,
                    clock: Callable[[], float] = time.monotonic) -> Any:
        """Bound one complete JSON request, including headers and a trickling body."""
        def request(deadline: _RequestDeadline | None) -> Any:
            with self._open(method, path, payload, deadline=deadline) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise TransportError("JSON response exceeds the configured byte limit")
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TransportError("Malformed JSON response") from exc
        return self._bounded(request, deadline_monotonic=deadline_monotonic, clock=clock)

    def request_json(self, method: str, path: str,
                     payload: Mapping[str, Any] | None = None, *,
                     deadline_monotonic: float | None = None,
                     clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
        result = self.request_any(method, path, payload, deadline_monotonic=deadline_monotonic, clock=clock)
        if not isinstance(result, dict):
            raise TransportError("JSON response must be an object")
        return result

    def stream(self, path: str, payload: Mapping[str, Any], *, deadline_monotonic: float | None = None,
               clock: Callable[[], float] = time.monotonic) -> SSEStream:
        def open_stream(deadline: _RequestDeadline | None) -> SSEStream:
            response = self._open("POST", path, payload, streaming=True, deadline=deadline)
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                response.close()
                raise TransportError("Server did not return text/event-stream")
            return SSEStream(response)
        # The measurement/capture bridge owns the stream body deadline and ticket;
        # this bound covers opening, before a stream object exists to cancel.
        return self._bounded(open_stream, deadline_monotonic=deadline_monotonic, clock=clock)
