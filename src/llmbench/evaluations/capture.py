"""Guarded Inspect ModelAPI bridge preserving native tool-call argument bytes.

Construction is inert. Every callback invocation checks the inference session
lock. The optional backend callback consumes an already verified loaded alias;
this adapter never calls load, unload, or model discovery itself.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import queue
import threading
import time
from typing import Any, Awaitable, Callable, Iterable
from uuid import uuid4

from ..config import GenerationSettings, RunMode
from ..backends.base import StreamEvent
from ..safety import SessionLock
from .tools import parse_tool_response

CompletionCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class UnsafeCaptureRuntimeState(RuntimeError):
    """The callback cannot safely dispatch another inference request."""

    abort_campaign = True


class _ChatStreamCollector:
    def __init__(self, *, request_id: str = "capture", event_sink=None,
                 max_events: int = 100_000, max_response_bytes: int = 16 * 1024 * 1024):
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        self.request_id, self.event_sink = request_id, event_sink
        self.max_events, self.max_response_bytes = max_events, max_response_bytes
        self.content, self.reasoning, self.raw_events = [], [], []
        self.tool_calls = {}
        self.usage = self.finish_reason = None
        self.byte_count = 0
        self.done = self.failure_recorded = False

    def publish(self, kind: str, data: Any, *, timestamp: float | None = None,
                request_id: str | None = None) -> None:
        if self.event_sink is not None:
            event = StreamEvent(request_id or self.request_id, time.monotonic() if timestamp is None else timestamp,
                                kind, copy.deepcopy(data))
            try:
                self.event_sink(event)
            except BaseException as exc:
                raise UnsafeCaptureRuntimeState("Raw capture persistence failed; further inference is unsafe") from exc

    def record(self, event: Any) -> tuple[str, Any]:
        event_fields = event if isinstance(event, dict) else {}
        kind = event.event if hasattr(event, "event") else event.get("event", "message")
        data = event.data if hasattr(event, "data") else event.get("data", event)
        timestamp = (event.monotonic_seconds if hasattr(event, "monotonic_seconds")
                     else event_fields.get("monotonic_seconds", time.monotonic()))
        request_id = event.request_id if hasattr(event, "request_id") else event_fields.get("request_id", self.request_id)
        row = {"request_id": request_id, "monotonic_seconds": timestamp,
               "event": kind, "data": copy.deepcopy(data)}
        # Persistence happens on the collector's caller before parsing. The last
        # event triggering an aggregate limit is retained by the sink, even when
        # it cannot be included in the bounded in-memory response bundle.
        self.publish(kind, data, timestamp=timestamp, request_id=request_id)
        if len(self.raw_events) >= self.max_events:
            raise ValueError("response exceeded stream event budget")
        self.byte_count += len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        if self.byte_count > self.max_response_bytes:
            raise ValueError("response exceeded stream byte budget")
        self.raw_events.append(row)
        return kind, data

    def consume(self, event: Any) -> None:
        kind, data = self.record(event)
        if kind in {"cancelled", "error"}:
            raise RuntimeError(f"stream ended with {kind}: {data}")
        if kind == "done" or data == "[DONE]":
            self.done = True
            return
        if self.done:
            raise ValueError("received output after native stream completion")
        if type(data) is not dict:
            raise ValueError("expected an OpenAI streaming object")
        if data.get("error"):
            raise RuntimeError(f"API error: {data['error']}")
        if data.get("usage") is not None:
            self.usage = copy.deepcopy(data["usage"])
        choices = data.get("choices", [])
        if type(choices) is not list:
            raise ValueError("choices must be a list")
        if len(choices) > 1:
            raise ValueError("benchmark capture supports exactly one generated choice")
        for choice in choices:
            if choice.get("index", 0) != 0:
                raise ValueError("unexpected completion choice index")
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            if type(delta) is not dict:
                raise ValueError("delta must be an object")
            for name, chunks in (("content", self.content), ("reasoning_content", self.reasoning)):
                value = delta.get(name)
                if value is not None:
                    if type(value) is not str:
                        raise ValueError(f"{name} delta must be text")
                    chunks.append(value)
            for call_delta in delta.get("tool_calls") or []:
                index = call_delta.get("index")
                if type(index) is not int or index < 0:
                    raise ValueError("tool-call delta requires a nonnegative integer index")
                call = self.tool_calls.setdefault(index, {"id": "", "type": None,
                                                          "function": {"name": "", "arguments": ""}})
                if call_delta.get("id"):
                    if type(call_delta["id"]) is not str:
                        raise ValueError("tool-call id must be a string")
                    if call["id"] and call["id"] != call_delta["id"]:
                        raise ValueError("tool-call id changed during streaming")
                    call["id"] = call_delta["id"]
                if "type" in call_delta:
                    call["type"] = call_delta["type"]
                function = call_delta.get("function", {})
                for key in ("name", "arguments"):
                    if function.get(key) is not None:
                        if type(function[key]) is not str:
                            raise ValueError("native function delta fields must be raw strings")
                        call["function"][key] += function[key]

    def fail(self, exc: BaseException, *, kind: str = "capture.error") -> None:
        terminal = {"error_type": type(exc).__name__, "detail": str(exc), "events_retained": len(self.raw_events)}
        # Even users without a sink receive partial evidence on the exception.
        exc.raw_events = copy.deepcopy(self.raw_events)
        exc.capture_terminal = terminal
        if not self.failure_recorded:
            self.failure_recorded = True
            self.publish(kind, terminal)

    def finish(self) -> dict[str, Any]:
        if self.finish_reason is None:
            raise RuntimeError("incomplete stream: no native finish reason")
        if self.tool_calls and sorted(self.tool_calls) != list(range(len(self.tool_calls))):
            raise ValueError("non-contiguous tool-call indexes")
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self.content) or None}
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[index] for index in sorted(self.tool_calls)]
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        self.publish("capture.completed", {"finish_reason": self.finish_reason, "events_retained": len(self.raw_events)})
        return {"choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
                "usage": self.usage, "llmbench_raw_stream": self.raw_events}


def collect_chat_stream(events: Iterable[Any], *, max_events: int = 100_000,
                        max_response_bytes: int = 16 * 1024 * 1024,
                        event_sink: Callable[[StreamEvent], None] | None = None,
                        request_id: str = "capture") -> dict[str, Any]:
    """Assemble deltas and expose exact partial events on errors and premature EOF.

    The sink is called synchronously before parsing each event. Exceptions retain
    ``raw_events`` and ``capture_terminal`` even when no sink was supplied.
    """
    collector = _ChatStreamCollector(request_id=request_id, event_sink=event_sink,
                                     max_events=max_events, max_response_bytes=max_response_bytes)
    try:
        for event in events:
            collector.consume(event)
            if collector.done:
                break
        return collector.finish()
    except BaseException as exc:
        collector.fail(exc)
        raise


def backend_completion_callback(
    backend: Any,
    instance_id: str,
    *,
    session_lock: SessionLock,
    mode: RunMode | str = RunMode.OFFLINE,
    event_sink: Callable[[StreamEvent], None] | None = None,
    timeout_seconds: float = 120.0,
    cancellation_grace_seconds: float = 0.5,
    max_events: int = 100_000,
    max_response_bytes: int = 16 * 1024 * 1024,
    deadline_monotonic: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CompletionCallback:
    """Guarded callback with reserved cancellation tickets and bounded termination.

    Sink callbacks run on the async calling thread, never the stream worker.
    Every cancellation/deadline poisons this callback: ``abort_reason`` becomes
    nonempty and every subsequent call fails before touching the backend.
    A worker retains its ticket until its own cleanup actually finishes.
    """
    if not instance_id:
        raise ValueError("a verified loaded instance alias is required")
    for name, value in (("timeout_seconds", timeout_seconds), ("cancellation_grace_seconds", cancellation_grace_seconds)):
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    if deadline_monotonic is not None and not math.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite")

    async def complete(request: dict[str, Any]) -> dict[str, Any]:
        session_lock.check("inference", mode)
        if complete.abort_reason:
            raise UnsafeCaptureRuntimeState(complete.abort_reason)
        request_timeout = timeout_seconds
        if deadline_monotonic is not None:
            request_timeout = min(request_timeout, deadline_monotonic - clock())
        if request_timeout <= 0:
            complete.abort_reason = "Quality evaluation wall deadline exhausted before request admission"
            raise UnsafeCaptureRuntimeState(complete.abort_reason)
        start = time.monotonic()
        request_id = str(uuid4())
        collector = _ChatStreamCollector(request_id=request_id, event_sink=event_sink,
                                         max_events=max_events, max_response_bytes=max_response_bytes)
        collector.publish("capture.request", {"request": copy.deepcopy(request), "instance_id": instance_id})
        if time.monotonic() - start >= request_timeout or (
                deadline_monotonic is not None and clock() >= deadline_monotonic):
            complete.abort_reason = "Quality evaluation wall deadline exhausted while persisting request evidence"
            raise UnsafeCaptureRuntimeState(complete.abort_reason)
        messages: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=32)
        abandon = threading.Event()
        finished = threading.Event()
        cancel_requested = threading.Event()
        deadline_fired = threading.Event()
        backend.reserve_request(request_id)

        def publish(kind: str, value: Any) -> None:
            while not abandon.is_set():
                try:
                    messages.put((kind, value), timeout=0.01)
                    return
                except queue.Full:
                    continue

        def work() -> None:
            stream = None
            try:
                if not cancel_requested.is_set():
                    stream = backend.stream(instance_id, copy.deepcopy(request), request_id=request_id)
                    for event in stream:
                        if abandon.is_set():
                            break
                        publish("event", event)
            except BaseException as exc:
                publish("error", exc)
            finally:
                try:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                except BaseException as exc:
                    publish("error", exc)
                finally:
                    try:
                        backend.release_request(request_id)
                    except BaseException as exc:
                        unsafe = UnsafeCaptureRuntimeState(f"Could not release quality request ticket: {exc}")
                        publish("error", unsafe)
                    finally:
                        finished.set()

        cancel_started = threading.Event()

        def request_cancel() -> None:
            cancel_requested.set()
            if cancel_started.is_set():
                return
            cancel_started.set()

            def cancel() -> None:
                try:
                    backend.cancel(request_id)
                except BaseException as exc:
                    publish("cancel_error", exc)

            threading.Thread(target=cancel, name=f"llmbench-quality-cancel-{request_id}", daemon=True).start()

        def deadline() -> None:
            deadline_fired.set()
            request_cancel()

        worker = threading.Thread(target=work, name=f"llmbench-quality-{request_id}", daemon=True)
        timer = threading.Timer(max(0.0, request_timeout - (time.monotonic() - start)), deadline)
        timer.daemon = True
        try:
            worker.start()
        except BaseException:
            backend.release_request(request_id)
            raise
        timer.start()

        async def drain_until_finished() -> bool:
            end = time.monotonic() + cancellation_grace_seconds
            omitted = 0
            while True:
                for _ in range(32):
                    if messages.empty() or time.monotonic() >= end:
                        break
                    kind, value = messages.get_nowait()
                    if kind == "event":
                        if (len(collector.raw_events) >= collector.max_events
                                or collector.byte_count >= collector.max_response_bytes):
                            omitted += 1
                            continue
                        try:
                            collector.record(value)
                        except ValueError:
                            # Budget-limit evidence was already sent to the sink.
                            pass
                    elif kind in {"error", "cancel_error"}:
                        collector.publish("capture.worker_error", {"error_type": type(value).__name__, "detail": str(value)})
                if finished.is_set():
                    if omitted:
                        collector.publish("capture.events_omitted", {"count": omitted, "reason": "capture_budget"})
                    worker.join(timeout=0)
                    return not worker.is_alive()
                if time.monotonic() >= end:
                    if omitted:
                        collector.publish("capture.events_omitted", {"count": omitted, "reason": "capture_budget"})
                    return False
                await asyncio.sleep(0.005)

        try:
            while True:
                if (deadline_fired.is_set() or time.monotonic() - start >= request_timeout
                        or (deadline_monotonic is not None and clock() >= deadline_monotonic)):
                    raise TimeoutError(f"Quality request exceeded {request_timeout:g}-second wall deadline")
                try:
                    kind, value = messages.get_nowait()
                except queue.Empty:
                    if finished.is_set():
                        break
                    await asyncio.sleep(0.005)
                    continue
                if kind == "event":
                    collector.consume(value)
                elif kind in {"error", "cancel_error"}:
                    raise value
            worker.join(timeout=0)
            return collector.finish()
        except BaseException as exc:
            need_cancel = not finished.is_set() or isinstance(exc, (asyncio.CancelledError, TimeoutError))
            if need_cancel:
                complete.abort_reason = f"Quality request {request_id} cancelled or interrupted; server idleness is unverified"
                request_cancel()
            if getattr(exc, "abort_campaign", False):
                complete.abort_reason = str(exc)
            try:
                collector.fail(exc, kind="capture.cancelled" if isinstance(exc, asyncio.CancelledError) else "capture.error")
            except BaseException as persistence_error:
                complete.abort_reason = str(persistence_error)
                # Failed persistence must not bypass request cancellation or
                # release a ticket while its worker remains active.
                exc = persistence_error
                collector.event_sink = None
            join_task = asyncio.create_task(drain_until_finished())
            # Repeated outer cancellation must not discard the bounded cleanup
            # check. The separate join task cannot wait longer than its grace.
            while not join_task.done():
                try:
                    await asyncio.shield(join_task)
                except asyncio.CancelledError:
                    continue
            terminated = join_task.result()
            collector.publish("capture.termination", {"worker_terminated": terminated,
                              "cancellation_requested": cancel_requested.is_set(),
                              "queued_events_not_retained": messages.qsize(),
                              "server_cancellation_verified": False})
            if not terminated:
                complete.abort_reason = f"Quality worker {request_id} did not terminate within cancellation grace"
                unsafe = UnsafeCaptureRuntimeState(complete.abort_reason)
                unsafe.raw_events = copy.deepcopy(collector.raw_events)
                raise unsafe from exc
            exc.raw_events = copy.deepcopy(collector.raw_events)
            if getattr(exc, "abort_campaign", False):
                raise exc
            raise
        finally:
            timer.cancel()
            abandon.set()

    complete.abort_reason = None
    return complete


def create_capture_model(
    completion: CompletionCallback,
    *,
    model_name: str,
    session_lock: SessionLock,
    mode: RunMode | str = RunMode.OFFLINE,
    generation: GenerationSettings | None = None,
) -> Any:
    """Create a real Inspect Model with a guarded caller-supplied completion API.

    The callback accepts an OpenAI chat-completion request and returns a complete
    response (or the output of collect_chat_stream). Even a fake callback requires
    the explicit inference lock to test this boundary; safe demos use MockLLM.
    """
    from inspect_ai.model import (
        ChatCompletionChoice, ChatMessageAssistant, ContentReasoning, ContentText,
        Model, ModelAPI, ModelCall, ModelOutput, ModelUsage, modelapi,
    )
    from inspect_ai.tool import ToolCall

    from .inspect_tasks import evaluation_config
    settings = generation or GenerationSettings(max_output_tokens=256)
    configured = evaluation_config(generation=settings)

    @modelapi("llmbench_capture")
    class RawCaptureAPI(ModelAPI):
        def __init__(self) -> None:
            super().__init__(model_name=model_name, config=configured)
            self._abort_reason = None

        @property
        def abort_reason(self) -> str | None:
            return self._abort_reason or getattr(completion, "abort_reason", None)

        async def generate(self, input: list[Any], tools: list[Any], tool_choice: Any, config: Any) -> Any:
            session_lock.check("inference", mode)
            if self.abort_reason:
                raise UnsafeCaptureRuntimeState(self.abort_reason)
            messages = []
            for item in input:
                message: dict[str, Any] = {"role": item.role, "content": item.text}
                native_history = (item.metadata or {}).get("llmbench_raw_message")
                if item.role == "assistant" and type(native_history) is dict:
                    # Preserve prior model reasoning and original raw argument
                    # formatting; rebuilding parsed args would alter the history.
                    messages.append({key: copy.deepcopy(value) for key, value in native_history.items()
                                     if key in {"role", "content", "tool_calls", "reasoning_content"}})
                    continue
                if item.role == "assistant" and item.tool_calls:
                    message["tool_calls"] = [{"id": call.id, "type": call.type, "function": {
                        "name": call.function, "arguments": json.dumps(call.arguments, allow_nan=False),
                    }} for call in item.tool_calls]
                if item.role == "tool":
                    message["tool_call_id"] = item.tool_call_id
                messages.append(message)
            request: dict[str, Any] = {
                "model": model_name, "messages": messages, "temperature": config.temperature,
                "max_tokens": config.max_tokens, "seed": config.seed,
            }
            if tools:
                request["tools"] = [{"type": "function", "function": {
                    "name": tool.name, "description": tool.description,
                    "parameters": tool.parameters.model_dump(exclude_none=True, by_alias=True),
                    "strict": settings.strict_tools,
                }} for tool in tools]
                request["tool_choice"] = tool_choice if type(tool_choice) is str else {
                    "type": "function", "function": {"name": tool_choice.name},
                }
                request["parallel_tool_calls"] = False
            for option in ("top_p", "top_k", "stop_seqs"):
                value = getattr(config, option, None)
                if value is not None:
                    request["stop" if option == "stop_seqs" else option] = value
            if config.reasoning_effort is not None:
                request["reasoning_effort"] = config.reasoning_effort
            try:
                raw = await completion(copy.deepcopy(request))
            except BaseException as exc:
                if getattr(exc, "abort_campaign", False) or isinstance(exc, (asyncio.CancelledError, TimeoutError)):
                    self._abort_reason = f"Quality callback aborted: {type(exc).__name__}: {exc}"
                raise
            if type(raw) is not dict or raw.get("error"):
                raise ValueError("completion callback returned an API error or invalid response")
            choices = raw.get("choices")
            if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
                raise ValueError("completion callback must return one native choice")
            message = choices[0].get("message")
            if type(message) is not dict:
                raise ValueError("completion response has no native message")
            calls, parse_errors = parse_tool_response(message)
            parsed_calls = [ToolCall(id=call.call_id, function=call.name, arguments=call.arguments) for call in calls]
            usage = None
            native_usage = raw.get("usage")
            if type(native_usage) is dict:
                prompt_tokens = native_usage.get("prompt_tokens")
                completion_tokens = native_usage.get("completion_tokens")
                if all(type(value) is int and value >= 0 for value in (prompt_tokens, completion_tokens)):
                    details = native_usage.get("completion_tokens_details") or {}
                    reasoning_tokens = details.get("reasoning_tokens")
                    if type(reasoning_tokens) is not int or reasoning_tokens < 0:
                        reasoning_tokens = None
                    usage = ModelUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens,
                                       total_tokens=prompt_tokens + completion_tokens, reasoning_tokens=reasoning_tokens)
            reason = choices[0].get("finish_reason")
            reason = {"length": "max_tokens"}.get(reason, reason)
            if reason not in {"stop", "max_tokens", "model_length", "tool_calls", "content_filter"}:
                reason = "unknown"
            reasoning_content = message.get("reasoning_content")
            assistant_content: Any = message.get("content") or ""
            if type(reasoning_content) is str and reasoning_content:
                assistant_content = [ContentReasoning(reasoning=reasoning_content),
                                     ContentText(text=message.get("content") or "")]
            output = ModelOutput(model=model_name, choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=assistant_content, tool_calls=parsed_calls or None,
                                             metadata={"llmbench_raw_message": copy.deepcopy(message)}),
                stop_reason=reason,
            )], usage=usage, metadata={
                "llmbench_raw_message": copy.deepcopy(message), "llmbench_raw_response": copy.deepcopy(raw),
                "raw_parse_errors": parse_errors, "synthetic": False,
                "input_truncated": raw.get("input_truncated"),
                "llmbench_context_policy": raw.get("llmbench_context_policy", "unknown"),
            })
            return output, ModelCall.create(request, raw)

    return Model(RawCaptureAPI(), config=configured)
