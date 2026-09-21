"""Convert preserved protocol events into conservative speed observations."""

from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from ..measurement import SpeedObservation
from ..safety import OperationForbidden
from .base import Backend, OperationDenied, StreamEvent, VerifiedModel


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _number(value: Any) -> float | None:
    if type(value) in {int, float} and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def observation_from_events(events: Iterable[StreamEvent], *, started_at: float, finished_at: float,
                            requested_output_tokens: int, expected_input_tokens: int | None = None,
                            terminal_status: str | None = None, synthetic: bool = False) -> SpeedObservation:
    """Pure conversion: no networking, retokenization, or duration reconstruction.

    OpenAI usage counts alone cannot supply a native generation speed: the llama.cpp adapter adds the
    server's own timings afterwards (``llamacpp.observation_with_native_timings``).
    """
    rows = list(events)
    if not math.isfinite(started_at) or not math.isfinite(finished_at) or finished_at < started_at:
        raise ValueError("Invalid observation start/end times")
    if len({row.request_id for row in rows}) > 1:
        raise ValueError("Cannot combine different requests in one speed observation")
    times = [row.monotonic_seconds for row in rows]
    if times != sorted(times) or any(not math.isfinite(value) or value < started_at or value > finished_at for value in times):
        raise ValueError("Event timestamps must be finite, ordered, and inside the observed request")
    request_starts = [row.monotonic_seconds for row in rows if row.event == "request_start"]
    if len(request_starts) > 1:
        raise ValueError("A speed observation cannot contain multiple request starts")
    origin = request_starts[0] if request_starts else started_at
    request_ends = [row.monotonic_seconds for row in rows if row.event == "request_end"]
    if len(request_ends) > 1 or (request_ends and request_ends[0] < origin):
        raise ValueError("Invalid or duplicate request-end marker")
    request_finished = request_ends[0] if request_ends else finished_at
    content_times = []
    first = None
    output_tokens = input_tokens = None
    accepted_verified = False
    native_rate = native_source = None
    finish = None
    complete = cancelled = cancellation_ack = invalid = False
    for row in rows:
        if row.event == "done":
            complete = True
        elif row.event == "cancelled":
            cancelled = True
            cancellation_ack = isinstance(row.data, dict) and row.data.get("server_cancellation_verified") is True
        elif row.event == "error":
            invalid = True
        if not isinstance(row.data, dict):
            continue
        data = row.data
        delivered = False
        if row.event not in {"request_start", "cancelled"}:
            choices = data.get("choices", [])
            if not isinstance(choices, list):
                invalid = True
                choices = []
            if len(choices) > 1:
                invalid = True  # Multi-completion throughput is not a single-user probe.
            for choice in choices:
                if not isinstance(choice, dict):
                    invalid = True
                    continue
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    delivered |= any(bool(delta.get(key)) for key in ("content", "reasoning_content", "reasoning", "tool_calls"))
                if choice.get("finish_reason") is not None:
                    next_finish = choice["finish_reason"]
                    if not isinstance(next_finish, str) or (finish is not None and finish != next_finish):
                        invalid = True
                    else:
                        finish = next_finish
            usage = data.get("usage")
            if isinstance(usage, dict):
                count = _count(usage.get("completion_tokens"))
                if count is not None:
                    if output_tokens is not None and output_tokens != count:
                        invalid = True
                    output_tokens, accepted_verified = count, True
                count = _count(usage.get("prompt_tokens"))
                if count is not None:
                    if input_tokens is not None and input_tokens != count:
                        invalid = True
                    input_tokens = count
            if data.get("error"):
                invalid = True
        if delivered:
            elapsed = row.monotonic_seconds - origin
            if elapsed < 0:
                raise ValueError("A content event preceded the request start")
            content_times.append(elapsed)
            if first is None:
                first = elapsed
    status = terminal_status
    if status is None:
        if cancelled or finish == "cancelled":
            status = "cancelled"
        elif invalid or finish in {"model_unloaded", "error"}:
            status = "failed"
        elif finish == "context_length":
            status = "context_limited"
        elif complete and finish is not None:
            status = "completed"
        else:
            status = "interrupted"
    if invalid:
        accepted_verified = False
        if status == "completed":
            status = "failed"
    return SpeedObservation(
        status=status, finish_reason=finish, output_tokens=output_tokens, native_generation_seconds=None,
        elapsed_seconds=request_finished - origin, first_event_seconds=first, content_event_times=tuple(content_times),
        requested_output_tokens=requested_output_tokens, input_tokens=input_tokens,
        expected_input_tokens=expected_input_tokens, accepted_tokens_verified=accepted_verified,
        synthetic=synthetic, cancellation_acknowledged=cancellation_ack,
        native_tokens_per_second_reported=native_rate, native_timing_source=native_source,
    )


@dataclass(frozen=True)
class SpeedProbeResult:
    observation: SpeedObservation
    events: tuple[StreamEvent, ...]
    error: str | None
    transport: str
    deadline_exceeded: bool = False
    worker_terminated: bool = True
    cancellation_requested: bool = False


def run_speed_probe(backend: Backend, verified: VerifiedModel, payload: Mapping[str, Any], *,
                    expected_input_tokens: int | None = None,
                    event_sink: Callable[[StreamEvent], None] | None = None,
                    max_events: int = 100_000, timeout_seconds: float = 300.0,
                    cancellation_grace_seconds: float = 0.5) -> SpeedProbeResult:
    """Collect a bounded OpenAI stream; backend permissions remain mandatory.

    The wall deadline requests cancellation and permits only a bounded grace
    interval. A blocked transport worker may outlive this call; that fact
    remains explicit and its reservation blocks subsequent model operations.
    """
    requested = payload.get("max_tokens")
    if type(requested) is not int or requested < 1:
        raise ValueError("Speed probes require an explicit positive max_tokens")
    if payload.get("n", 1) != 1:
        raise ValueError("Speed probes require n=1")
    request_id = str(uuid4())
    return _collect(backend, backend.stream(verified.model.instance_id, payload, request_id=request_id),
                    request_id=request_id, requested=requested, expected_input_tokens=expected_input_tokens,
                    event_sink=event_sink, max_events=max_events, transport="openai-compatible-sse",
                    timeout_seconds=timeout_seconds, cancellation_grace_seconds=cancellation_grace_seconds)


def _collect(backend: Backend, iterator: Iterable[StreamEvent], *, request_id: str,
             requested: int, expected_input_tokens: int | None,
             event_sink: Callable[[StreamEvent], None] | None, max_events: int,
             transport: str, timeout_seconds: float,
             cancellation_grace_seconds: float) -> SpeedProbeResult:
    if type(max_events) is not int or max_events < 1:
        raise ValueError("max_events must be positive")
    for name, value in (("timeout_seconds", timeout_seconds), ("cancellation_grace_seconds", cancellation_grace_seconds)):
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    started = time.monotonic()
    rows = []
    error = terminal = None
    messages: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=256)
    stop_collecting = threading.Event()
    deadline_fired = threading.Event()
    cancellation_requested = threading.Event()
    worker_finished = threading.Event()
    backend.reserve_request(request_id)

    def publish(kind: str, value: Any) -> None:
        while not stop_collecting.is_set():
            try:
                messages.put((kind, value), timeout=0.02)
                return
            except queue.Full:
                continue

    def cancel_request() -> None:
        cancellation_requested.set()
        try:
            backend.cancel(request_id)
        except Exception:
            # A cancellation transport error is not an acknowledgement. The
            # result records pending worker state and remains unsuccessful.
            pass

    def deadline() -> None:
        deadline_fired.set()
        cancel_request()

    def work() -> None:
        try:
            for event in iterator:
                publish("event", event)
        except BaseException as exc:
            publish("error", exc)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            except BaseException as exc:
                publish("error", exc)
            finally:
                backend.release_request(request_id)
                worker_finished.set()
                publish("done", None)

    worker = threading.Thread(target=work, name=f"llmbench-probe-{request_id}", daemon=True)
    timer = threading.Timer(timeout_seconds, deadline)
    timer.daemon = True
    worker.start()
    timer.start()
    grace_end: float | None = None
    try:
        while True:
            now = time.monotonic()
            if deadline_fired.is_set() or now - started >= timeout_seconds:
                if not deadline_fired.is_set():
                    deadline_fired.set()
                    threading.Thread(target=cancel_request, daemon=True).start()
                terminal = "timed_out"
                if error is None:
                    error = f"Probe exceeded its {timeout_seconds:g}-second wall deadline"
                if grace_end is None:
                    grace_end = now + cancellation_grace_seconds
            if grace_end is not None and now >= grace_end:
                break
            wait = min(0.02, max(0.001, (grace_end or started + timeout_seconds) - now))
            try:
                kind, value = messages.get(timeout=wait)
            except queue.Empty:
                if worker_finished.is_set() and messages.empty():
                    break
                continue
            if kind == "done":
                break
            if kind == "error":
                if isinstance(value, (OperationDenied, OperationForbidden, KeyboardInterrupt, SystemExit)):
                    raise value
                error = f"{type(value).__name__}: {value}"
                cause = value
                timed_out = False
                while cause is not None:
                    timed_out |= isinstance(cause, TimeoutError)
                    cause = cause.__cause__
                if terminal != "timed_out":
                    terminal = "timed_out" if timed_out else "failed"
                continue
            if len(rows) >= max_events:
                error, terminal = "Speed probe exceeded the event storage budget", "failed"
                threading.Thread(target=cancel_request, daemon=True).start()
                break
            rows.append(value)
            if event_sink is not None:
                # Persistence remains on the calling thread (e.g. a SQLite
                # connection). Callbacks should be bounded local writes.
                event_sink(value)
    except BaseException:
        threading.Thread(target=cancel_request, daemon=True).start()
        raise
    finally:
        timer.cancel()
        stop_collecting.set()
        if worker_finished.is_set():
            worker.join(timeout=0.02)
    finished = time.monotonic()
    observation = observation_from_events(rows, started_at=started, finished_at=finished,
                                          requested_output_tokens=requested,
                                          expected_input_tokens=expected_input_tokens,
                                          terminal_status=terminal)
    return SpeedProbeResult(observation, tuple(rows), error, transport,
                            deadline_exceeded=deadline_fired.is_set(), worker_terminated=not worker.is_alive(),
                            cancellation_requested=cancellation_requested.is_set())
