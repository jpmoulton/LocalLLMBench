"""Cancellation/persistence regressions; all backend traffic is in-memory fake IO."""

import asyncio
import threading
import time

import pytest

from llmbench.backends import BackendError, StreamEvent
from llmbench.config import RunMode, TaskSelection
from llmbench.evaluations.capture import (
    UnsafeCaptureRuntimeState, backend_completion_callback, collect_chat_stream, create_capture_model,
)
from llmbench.evaluations.inspect_tasks import run_inspect_suite
from llmbench.safety import SessionLock
from test_backends_llamacpp import ALIAS, FakeLlamaTransport, FakeStream, backend_for, chat_chunks


async def until(predicate, timeout=1):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError("fake worker condition not reached")
        await asyncio.sleep(0.005)


def prepared_backend():
    """An attached llama.cpp backend over the recorded-fixture transport. No server is touched."""
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    backend.attach()
    transport.calls.clear()
    return backend, transport


def never_streamed(transport):
    """Cancelled before or during the preflight: inference is never started - no stream is even requested."""
    return not any(call[0] == "STREAM" for call in transport.calls)


def test_cancel_during_lazy_preflight_never_starts_inference_after_cancel():
    backend, transport = prepared_backend()
    entered, release = threading.Event(), threading.Event()
    check_deadline = backend._check_deadline

    def blocked_preflight():
        entered.set()
        assert release.wait(2)
        return check_deadline()

    backend._check_deadline = blocked_preflight
    rows, sink_threads = [], []
    caller_thread = threading.get_ident()

    def sink(event):
        rows.append(event)
        sink_threads.append(threading.get_ident())

    callback = backend_completion_callback(backend, ALIAS, session_lock=SessionLock(allow_inference=True),
                                           mode=RunMode.LIVE, event_sink=sink, cancellation_grace_seconds=0.5)

    async def exercise():
        task = asyncio.create_task(callback({"messages": []}))
        await until(entered.is_set)
        assert len(backend.pending_request_ids) == 1
        task.cancel()
        await until(lambda: bool(backend._cancelled))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not backend.pending_request_ids
        with pytest.raises(UnsafeCaptureRuntimeState):
            await callback({"messages": []})

    try:
        asyncio.run(exercise())
    finally:
        release.set()
    assert never_streamed(transport)
    assert sink_threads and set(sink_threads) == {caller_thread}
    assert rows[0].event == "capture.request"
    assert any(row.event == "capture.cancelled" for row in rows)
    assert any(row.event == "capture.termination" and row.data["worker_terminated"] for row in rows)


def test_cancel_before_worker_body_keeps_ticket_and_skips_backend_stream(monkeypatch):
    backend, transport = prepared_backend()
    entered, release = threading.Event(), threading.Event()
    original_run = threading.Thread.run

    def gated_run(thread):
        if thread.name.startswith("llmbench-quality-") and "-cancel-" not in thread.name:
            entered.set()
            assert release.wait(2)
        original_run(thread)

    monkeypatch.setattr(threading.Thread, "run", gated_run)
    callback = backend_completion_callback(backend, ALIAS, session_lock=SessionLock(allow_inference=True),
                                           mode=RunMode.LIVE)

    async def exercise():
        task = asyncio.create_task(callback({"messages": []}))
        await until(entered.is_set)
        task.cancel()
        await until(lambda: bool(backend._cancelled))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not backend.pending_request_ids

    try:
        asyncio.run(exercise())
    finally:
        release.set()
    assert not any(call[0] == "STREAM" for call in transport.calls)


def test_timeout_keeps_unfinished_ticket_and_aborts_future_dispatch():
    backend, transport = prepared_backend()
    release = threading.Event()
    check_deadline = backend._check_deadline

    def blocked_preflight():
        release.wait(2)
        return check_deadline()

    backend._check_deadline = blocked_preflight
    rows = []
    callback = backend_completion_callback(backend, ALIAS, session_lock=SessionLock(allow_inference=True),
                                           mode=RunMode.LIVE, event_sink=rows.append,
                                           timeout_seconds=0.02, cancellation_grace_seconds=0.02)

    async def exercise():
        with pytest.raises(UnsafeCaptureRuntimeState) as caught:
            await callback({"messages": []})
        assert caught.value.abort_campaign
        assert backend.pending_request_ids
        with pytest.raises(BackendError):
            backend.reserve_request("unrelated-next-request")
        with pytest.raises(UnsafeCaptureRuntimeState):
            await callback({"messages": []})
        release.set()
        await until(lambda: not backend.pending_request_ids)

    try:
        asyncio.run(exercise())
    finally:
        release.set()
    assert never_streamed(transport)
    assert any(row.event == "capture.termination" and row.data["worker_terminated"] is False for row in rows)


def test_cancellation_during_connection_startup_closes_owned_connection():
    backend, transport = prepared_backend()
    entered, release = threading.Event(), threading.Event()
    stream = FakeStream(chat_chunks())

    def blocked_stream(path, body):
        entered.set()
        assert release.wait(2)
        return stream

    transport.stream = blocked_stream
    callback = backend_completion_callback(backend, ALIAS, session_lock=SessionLock(allow_inference=True),
                                           mode=RunMode.LIVE, cancellation_grace_seconds=0.5)

    async def exercise():
        task = asyncio.create_task(callback({"messages": []}))
        await until(entered.is_set)
        task.cancel()
        await until(lambda: bool(backend._cancelled))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed and not backend.pending_request_ids

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_cancel_after_partial_delta_retains_exact_partial_and_releases_worker():
    backend, transport = prepared_backend()
    partial = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "first", "type": "function",
                 "function": {"name": "x", "arguments": '{"key":'}}]}}]}

    class BlockingStream:
        def __init__(self):
            self.release = threading.Event()
            self.closed = False

        def __iter__(self):
            yield "message", partial
            assert self.release.wait(2)

        def close(self):
            self.closed = True
            self.release.set()

    stream = BlockingStream()
    transport.stream = lambda path, body: stream
    rows = []
    callback = backend_completion_callback(backend, ALIAS, session_lock=SessionLock(allow_inference=True),
                                           mode=RunMode.LIVE, event_sink=rows.append)

    async def exercise():
        task = asyncio.create_task(callback({"messages": []}))
        await until(lambda: any(row.event == "message" for row in rows))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed and not backend.pending_request_ids

    try:
        asyncio.run(exercise())
    finally:
        stream.release.set()
    assert next(row for row in rows if row.event == "message").data == partial
    assert callback.abort_reason


def test_sink_failure_cannot_skip_worker_cleanup():
    backend, transport = prepared_backend()

    def sink(event):
        if event.event == "message":
            raise OSError("synthetic persistence failure")

    callback = backend_completion_callback(backend, ALIAS, session_lock=SessionLock(allow_inference=True),
                                           mode=RunMode.LIVE, event_sink=sink)
    with pytest.raises(UnsafeCaptureRuntimeState):
        asyncio.run(callback({"messages": []}))
    # The worker releases its ticket in a ``finally`` that runs after the stream is closed and the server has
    # been polled for an idle slot, i.e. moments after the callback raises. Cleanup must COMPLETE, and soon.
    deadline = time.monotonic() + 2
    while backend.pending_request_ids and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not backend.pending_request_ids
    assert callback.abort_reason and transport.last_stream.closed


def test_partial_disconnect_and_overflow_persist_raw_before_error():
    partial = StreamEvent("native-request", 123.25, "message", {"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": "c", "type": "function", "function": {"name": "x", "arguments": '{"id":'},
    }]}}]})

    def disconnected():
        yield partial
        raise ConnectionError("synthetic disconnect")

    rows = []
    with pytest.raises(ConnectionError) as caught:
        collect_chat_stream(disconnected(), event_sink=rows.append)
    assert rows[0] == partial
    assert rows[-1].event == "capture.error"
    assert caught.value.raw_events[0]["monotonic_seconds"] == 123.25
    assert caught.value.raw_events[0]["data"] == partial.data
    rows.clear()
    with pytest.raises(ValueError, match="byte budget"):
        collect_chat_stream([partial], event_sink=rows.append, max_response_bytes=1)
    assert rows[0] == partial and rows[-1].event == "capture.error"
    rows.clear()
    with pytest.raises(RuntimeError, match="incomplete") as caught:
        collect_chat_stream([partial], event_sink=rows.append)
    assert caught.value.raw_events[0]["data"] == partial.data
    assert rows[-1].event == "capture.error"


def test_fatal_callback_stops_inspect_suite_and_preserves_no_second_dispatch(tmp_path):
    pytest.importorskip("inspect_ai")
    calls = []
    lock = SessionLock(allow_inference=True)

    async def callback(request):
        calls.append(request)
        raise UnsafeCaptureRuntimeState("synthetic unacknowledged worker")

    model = create_capture_model(callback, model_name="fake-only", session_lock=lock, mode=RunMode.LIVE)
    selection = TaskSelection(suite="tool-probes", revision="local-tools-v1",
                              task_ids=("tools/nested-exact-v1", "tools/no-call-v1"))
    with pytest.raises(UnsafeCaptureRuntimeState) as caught:
        run_inspect_suite(model, tmp_path, [selection], lock, mode=RunMode.LIVE)
    assert caught.value.abort_campaign
    assert len(calls) == 1
    assert caught.value.logs
