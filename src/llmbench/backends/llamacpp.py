"""Attach-only adapter for an already running, pinned llama.cpp ``llama-server``.

The host container runner owns process start, model load and teardown. This
adapter never loads or unloads anything: it reads the server identity back,
counts prompts through the server's own template and tokenizer, and streams
the OpenAI-compatible chat route. Construction performs no network activity.

Route behaviour verified on build b11011 (tests/data/route-checks-b11011-mtp.json):
``/apply-template`` returns ``{"prompt": str}``; ``/tokenize`` of that prompt equals chat
``usage.prompt_tokens``; the final streamed chat chunk carries ``timings`` and ``usage`` with empty
``choices``; an oversized request returns HTTP 400 ``exceed_context_size_error``. Anything else
fails loudly instead of being repaired.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from ..config import RunMode
from ..measurement import SpeedObservation
from ..safety import SessionLock
from .base import BackendError, LivePermission, LoadedModel, StreamEvent, VerificationError, VerifiedModel
from .http import TransportError, UrllibHTTPTransport

INFERENCE_URL = "http://inference:8080"
CHAT_PATH = "/v1/chat/completions"
COUNT_METHOD = "llama.cpp:/apply-template+/tokenize(add_special,parse_special)"
OVERFLOW_ERROR_TYPE = "exceed_context_size_error"
OVERFLOW_MARGIN_TOKENS = 64
NATIVE_TIMING_SOURCE = "llama.cpp.timings"
CONTEXT_POLICIES = ("reject-overflow-no-shift", "unknown")
_TIMING_COUNTS = ("cache_n", "prompt_n", "predicted_n")
_TIMING_NUMBERS = ("prompt_ms", "predicted_ms", "predicted_per_second")
_DRAFT_COUNTS = ("draft_n", "draft_n_accepted")
_COUNT_PROBE_MESSAGES = (
    {"role": "system", "content": "You are a terse assistant used to verify prompt token accounting."},
    {"role": "user", "content": "Reply with the single word: ready."},
)
_COUNT_PROBE_TOOL = {"type": "function", "function": {
    "name": "record_status", "description": "Record a status string.",
    "parameters": {"type": "object", "required": ["status"], "additionalProperties": False,
                   "properties": {"status": {"type": "string"}}},
}}
# Byte-for-byte the same schema as _COUNT_PROBE_TOOL with the object keys permuted into the order the
# Inspect serving path emits. llama.cpp keeps request key order when a chat template renders a tool with
# ``{{ tool | tojson }}``, so if these two count differently the counting body must be the serving body
# exactly, not merely the same schema (LIVE-008).
_COUNT_PROBE_TOOL_REORDERED = {"type": "function", "function": {
    "name": "record_status", "description": "Record a status string.",
    "parameters": {"type": "object", "properties": {"status": {"type": "string"}},
                   "required": ["status"], "additionalProperties": False},
}}


@dataclass(frozen=True)
class ExpectedServer:
    alias: str
    n_ctx: int
    build_info: str | None
    model_path: str = "/models/model.gguf"
    total_slots: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not self.alias:
            raise ValueError("alias must be a nonempty string")
        for name in ("n_ctx", "total_slots"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _mismatch(result: dict[str, Any], key: str, requested: Any, effective: Any, *, missing: bool = False) -> None:
    if missing:
        result[key] = {"requested": requested, "effective": None, "reason": "missing_readback"}
    elif type(requested) is not type(effective) or requested != effective:
        result[key] = {"requested": requested, "effective": effective, "reason": "different"}


def _messages_body(messages: Iterable[Mapping[str, Any]], tools: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The template-affecting fields exactly as the capture model sends them (capture.py)."""
    body: dict[str, Any] = {"messages": [deepcopy(dict(item)) for item in messages]}
    tool_list = [deepcopy(dict(item)) for item in tools or ()]
    if tool_list:
        body.update(tools=tool_list, tool_choice="auto", parallel_tool_calls=False)
    return body


class _CancelledBeforeOpen(Exception):
    """Internal: leave the stream-opening block without opening anything."""


class LlamaCppBackend:
    """One verified server, one alias, one request at a time."""

    def __init__(self, base_url: str, *, expected: ExpectedServer,
                 permissions: LivePermission | None = None, transport: Any = None,
                 session_lock: SessionLock | None = None,
                 policy_path: str | Path = "runtime-policy.json", timeout: float = 30.0,
                 allow_remote: bool = False, cache_prompt: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 deadline_monotonic: float | None = None) -> None:
        if not isinstance(expected, ExpectedServer):
            raise ValueError("expected must be an ExpectedServer")
        if type(allow_remote) is not bool or type(cache_prompt) is not bool:
            raise ValueError("allow_remote and cache_prompt must be booleans")
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("base_url must be an HTTP(S) origin without credentials or a path")
        # The compose-internal name "inference" is remote by this rule on purpose: only the
        # evaluator container path may opt in, and it has to say so.
        if not allow_remote and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote inference requires explicit allow_remote=True")
        self.base_url = base_url.rstrip("/")
        self.expected = expected
        self.permissions = permissions or LivePermission()
        self.cache_prompt = cache_prompt
        self._session_lock = session_lock
        self._policy_path = Path(policy_path)
        self.transport = transport or UrllibHTTPTransport(base_url, timeout=timeout, allow_remote=allow_remote)
        self._clock, self._sleep = clock, sleep
        if deadline_monotonic is not None and not math.isfinite(deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite")
        self._deadline_monotonic = deadline_monotonic
        self._abort_reason: str | None = None
        self._verified: VerifiedModel | None = None
        self._streams: dict[str, Any] = {}
        self._pending_requests: set[str] = set()
        self._opening_requests: set[str] = set()
        self._cancelled: set[str] = set()
        self._cancel_done: dict[str, threading.Event] = {}
        self._idle_verified: dict[str, bool] = {}
        self._lock = threading.RLock()
        self._closed = False

    # -- guards -----------------------------------------------------------------------------
    def _require(self, operation: str) -> None:
        if self._closed:
            raise BackendError("llama.cpp adapter is closed")
        self.permissions.require(operation)
        lock = self._session_lock or SessionLock.read(self._policy_path)  # reread on every operation
        lock.check(operation, RunMode.LIVE)

    def _require_attached(self) -> VerifiedModel:
        if self._verified is None:
            raise VerificationError("attach() must verify the server before this operation",
                                    instance_id=self.expected.alias)
        return self._verified

    def _require_no_active_request(self) -> None:
        if self.pending_request_ids:
            raise BackendError("A prior inference worker is still active; wait for its termination before continuing")

    @property
    def pending_request_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._pending_requests | self._opening_requests | set(self._streams)))

    def reserve_request(self, request_id: str) -> None:
        """Reserve a cancellation ticket before a worker can block in preflight."""
        with self._lock:
            if self.pending_request_ids:
                raise BackendError("A prior inference worker is still active; wait for its termination before continuing")
            self._pending_requests.add(request_id)

    def release_request(self, request_id: str) -> None:
        with self._lock:
            self._pending_requests.discard(request_id)
            if request_id not in self._streams and request_id not in self._opening_requests:
                self._forget(request_id)

    def _forget(self, request_id: str) -> None:
        self._cancelled.discard(request_id)
        self._cancel_done.pop(request_id, None)
        self._idle_verified.pop(request_id, None)

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def _check_deadline(self, deadline: float | None = None) -> None:
        if self._abort_reason:
            error = TimeoutError(self._abort_reason)
            error.abort_campaign = True
            raise error
        effective = self._deadline_monotonic if deadline is None else deadline
        if effective is not None and self._clock() >= effective:
            self._abort_reason = "llama.cpp evaluation wall deadline exhausted"
            error = TimeoutError(self._abort_reason)
            error.abort_campaign = True
            raise error

    def _request(self, method: str, path: str, payload: Any = None, *, any_json: bool = False,
                 cleanup_deadline: float | None = None) -> Any:
        # Each route is admitted independently: template/tokenizer/count probes must
        # never carry an old stage-level timeout into a later generation request.
        if cleanup_deadline is None:
            self._check_deadline()
            deadline = self._deadline_monotonic
        else:
            deadline = cleanup_deadline  # cleanup may outlive inference, but is bounded itself
            if self._clock() >= deadline:
                raise TimeoutError("server-idle verification deadline exhausted")
        call = self.transport.request_any if any_json else self.transport.request_json
        try:
            if isinstance(self.transport, UrllibHTTPTransport):
                value = call(method, path, payload, deadline_monotonic=deadline, clock=self._clock)
            else:
                # Injected offline transports own their blocking behavior. Admission
                # and post-response checks remain identical to the production path.
                value = call(method, path, payload)
            if cleanup_deadline is None:
                self._check_deadline()
            elif self._clock() >= deadline:
                raise TimeoutError("server-idle verification deadline exhausted")
            return value
        except TimeoutError as exc:
            if cleanup_deadline is None:
                self._abort_reason = f"llama.cpp request deadline failure at {path}; server idleness is unverified"
                exc.abort_campaign = True
            raise

    # -- identity ---------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def readback(self) -> dict[str, Any]:
        props = self._request("GET", "/props")
        models = self._request("GET", "/v1/models")
        slots = self._request("GET", "/slots", any_json=True)
        if not isinstance(slots, list) or any(not isinstance(item, dict) for item in slots):
            raise BackendError("/slots must return a JSON list of slot objects")
        return {"props": props, "models": models, "slots": slots}

    def attach(self) -> VerifiedModel:
        """GET-only identity readback. No ownership is acquired; nothing can be unloaded here."""
        expected = self.expected
        raw = self.readback()
        props, slots = raw["props"], raw["slots"]
        settings = props.get("default_generation_settings")
        n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        requested: dict[str, Any] = {"alias": expected.alias, "model_path": expected.model_path,
                                     "n_ctx": expected.n_ctx, "total_slots": expected.total_slots}
        effective: dict[str, Any] = {"alias": props.get("model_alias"), "model_path": props.get("model_path"),
                                     "n_ctx": n_ctx, "total_slots": props.get("total_slots")}
        if expected.build_info is not None:
            requested["build_info"], effective["build_info"] = expected.build_info, props.get("build_info")
        mismatches: dict[str, Any] = {}
        for key, value in requested.items():
            _mismatch(mismatches, key, value, effective[key], missing=effective[key] is None)
        if props.get("is_sleeping") not in {None, False}:
            _mismatch(mismatches, "is_sleeping", False, props.get("is_sleeping"))
        _mismatch(mismatches, "slots.count", expected.total_slots, len(slots))
        for index, slot in enumerate(slots):
            _mismatch(mismatches, f"slots[{index}].n_ctx", expected.n_ctx, slot.get("n_ctx"),
                      missing=slot.get("n_ctx") is None)
        data = raw["models"].get("data")
        served = [item.get("id") for item in data if isinstance(item, dict)] if isinstance(data, list) else None
        _mismatch(mismatches, "models.ids", [expected.alias], served, missing=served is None)
        if served == [expected.alias] and isinstance(data[0].get("meta"), dict) and "n_ctx" in data[0]["meta"]:
            _mismatch(mismatches, "models.meta.n_ctx", expected.n_ctx, data[0]["meta"]["n_ctx"])
        effective["slots_n_ctx"] = [slot.get("n_ctx") for slot in slots]
        effective["speculative"] = [slot.get("speculative") for slot in slots]
        if mismatches:
            self._verified = None
            raise VerificationError("llama.cpp server identity differs from the expected candidate",
                                    mismatches=mismatches, instance_id=expected.alias)
        model = LoadedModel(expected.alias, expected.model_path, deepcopy(effective), False, deepcopy(raw))
        self._verified = VerifiedModel(model, requested, deepcopy(effective),
                                       "llama.cpp /props + /v1/models + /slots readback (attach-only)")
        return deepcopy(self._verified)

    # -- prompt accounting ------------------------------------------------------------------
    def apply_template(self, messages: Iterable[Mapping[str, Any]], tools: Iterable[Mapping[str, Any]]) -> str:
        self._require("inference")
        response = self._request("POST", "/apply-template", _messages_body(messages, tools))
        prompt = response.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise BackendError("/apply-template did not return a nonempty 'prompt' string; "
                               "prompt accounting is unavailable for this server build")
        return prompt

    def tokenize(self, content: str, *, add_special: bool = True, parse_special: bool = True) -> list[int]:
        self._require("inference")
        if not isinstance(content, str):
            raise ValueError("content must be text")
        response = self._request("POST", "/tokenize", {
            "content": content, "add_special": add_special, "parse_special": parse_special})
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or any(type(item) is not int or item < 0 for item in tokens):
            raise BackendError("/tokenize did not return a list of token IDs")
        return tokens

    def count_chat_tokens(self, messages: Iterable[Mapping[str, Any]],
                          tools: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        body = _messages_body(messages, tools)
        rendered = self.apply_template(body["messages"], body.get("tools", ()))
        return {"tokens": len(self.tokenize(rendered)), "rendered_prompt": rendered,
                "messages": body["messages"], "method": COUNT_METHOD}

    def verify_count_route(self) -> dict[str, Any]:
        """Compare the counter with the chat route's own prompt count, with and without tools.

        One generated token per case. ``exact`` is true only if every case matches; nothing is
        rounded or tolerated, because NIAH context claims depend on this equality.

        Both legs of each case are built from one dict, so this check can only prove that the counter
        agrees with the chat route for the body it was given. It cannot see a caller that counts one
        serialisation and sends another, which is how LIVE-008 stayed hidden behind ``exact: true``.
        ``tool_schema_key_order_sensitive`` reports the missing half: the same tool schema is rendered
        and tokenized a second time with permuted object keys (template/tokenize only, no generation).
        True means the served prompt depends on request key order, so a counting body that is not the
        serving body byte-for-byte cannot produce an exact count.
        """
        self._require("inference")
        self._require_attached()
        self._require_no_active_request()
        cases = []
        for name, tools in (("no-tools", ()), ("tools", (_COUNT_PROBE_TOOL,))):
            counted = self.count_chat_tokens(_COUNT_PROBE_MESSAGES, tools)
            body = {**_messages_body(_COUNT_PROBE_MESSAGES, tools), "model": self.expected.alias,
                    "max_tokens": 1, "temperature": 0, "stream": False, "cache_prompt": self.cache_prompt}
            response = self._request("POST", CHAT_PATH, body)
            usage = response.get("usage")
            observed = _count(usage.get("prompt_tokens")) if isinstance(usage, dict) else None
            cases.append({"name": name, "counted_tokens": counted["tokens"], "usage_prompt_tokens": observed,
                          "match": observed is not None and observed == counted["tokens"],
                          "served_model": response.get("model"), "timings": native_timings([response])})
        reordered = self.count_chat_tokens(_COUNT_PROBE_MESSAGES, (_COUNT_PROBE_TOOL_REORDERED,))
        with_tools = next(case["counted_tokens"] for case in cases if case["name"] == "tools")
        return {"exact": all(case["match"] for case in cases), "method": COUNT_METHOD, "cases": cases,
                "tool_schema_reordered_tokens": reordered["tokens"],
                "tool_schema_key_order_sensitive": reordered["tokens"] != with_tools}

    def probe_overflow_rejection(self) -> dict[str, Any]:
        """Behavioural evidence that the slot rejects (not shifts/truncates) an oversized prompt."""
        self._require("inference")
        self._require_attached()
        self._require_no_active_request()
        n_ctx = self.expected.n_ctx
        seed_tokens = self.tokenize("hello", add_special=False)
        if not seed_tokens:
            raise BackendError("/tokenize returned no token for the overflow probe")
        sent = n_ctx + OVERFLOW_MARGIN_TOKENS
        result: dict[str, Any] = {"passed": False, "sent_tokens": sent, "expected_n_ctx": n_ctx,
                                  "expected_error_type": OVERFLOW_ERROR_TYPE, "status": None,
                                  "error_type": None, "reported_n_ctx": None, "reported_n_prompt_tokens": None,
                                  "detail": None}
        try:
            response = self._request("POST", "/completion", {
                "prompt": [seed_tokens[0]] * sent, "n_predict": 1, "cache_prompt": False, "stream": False})
        except TransportError as exc:
            result["status"] = exc.status
            if exc.status is None or exc.body is None:
                result["detail"] = f"no HTTP error status/body to inspect: {exc}"
                return result
            try:
                parsed = json.loads(exc.body)
                error = parsed.get("error") if isinstance(parsed, dict) else None
            except (ValueError, UnicodeDecodeError):
                error = None
            if not isinstance(error, dict):
                result["detail"] = "HTTP error body is not a JSON object with an 'error' object"
                return result
            result.update(error_type=error.get("type"), reported_n_ctx=error.get("n_ctx"),
                          reported_n_prompt_tokens=error.get("n_prompt_tokens"))
            context_agrees = error.get("n_ctx") is None or (type(error["n_ctx"]) is int and error["n_ctx"] == n_ctx)
            result["passed"] = exc.status == 400 and error.get("type") == OVERFLOW_ERROR_TYPE and context_agrees
            if not result["passed"]:
                result["detail"] = "rejection differs from HTTP 400 exceed_context_size_error for the expected n_ctx"
            return result
        result.update(status=200, detail="server accepted a prompt larger than the slot context",
                      truncated=response.get("truncated"), tokens_evaluated=response.get("tokens_evaluated"))
        return result

    # -- inference --------------------------------------------------------------------------
    def _cancelled_event(self, identifier: str, *, started: bool) -> StreamEvent:
        data: dict[str, Any] = {"server_cancellation_verified": self._cancel_outcome(identifier) if started else False}
        if not started:
            data["request_started"] = False
        return StreamEvent(identifier, time.monotonic(), "cancelled", data)

    def _cancel_outcome(self, identifier: str) -> bool:
        with self._lock:
            done = self._cancel_done.get(identifier)
        if done is not None:
            done.wait(30.0)  # cancel() itself is bounded by wait_idle(); this only orders the two threads.
        with self._lock:
            return self._idle_verified.get(identifier, False)

    def stream(self, instance_id: str, payload: Mapping[str, Any], *,
               request_id: str | None = None) -> Iterator[StreamEvent]:
        """Stream the verified alias; API events are preserved unmodified.

        Event timestamps always come from ``time.monotonic`` because the measurement bridge
        compares them with its own monotonic request window.
        """
        self._require("inference")
        identifier = request_id or str(uuid4())
        if identifier in self._cancelled:
            yield self._cancelled_event(identifier, started=False)
            return
        self._require_attached()
        if instance_id != self.expected.alias:
            raise VerificationError("Inference is limited to the verified server alias", instance_id=instance_id)
        body = deepcopy(dict(payload))
        if "model" in body and body["model"] != instance_id:
            raise ValueError("Payload model must match the verified server alias")
        if body.get("n", 1) != 1:
            raise ValueError("Single-user measurement requires n=1")
        body.update(model=instance_id, stream=True, cache_prompt=self.cache_prompt, timings_per_token=False)
        body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
        with self._lock:
            if self._streams or self._opening_requests or self._pending_requests - {identifier}:
                raise BackendError("Single-user measurement permits only one active inference stream")
            cancelled_before_open = identifier in self._cancelled
            if not cancelled_before_open:
                self._opening_requests.add(identifier)
        if cancelled_before_open:
            yield self._cancelled_event(identifier, started=False)
            return
        try:
            self._check_deadline()
            if identifier in self._cancelled:  # cancelled while the preflight ran: never start inference
                cancelled_before_open = True
                raise _CancelledBeforeOpen()
            if isinstance(self.transport, UrllibHTTPTransport):
                stream = self.transport.stream(CHAT_PATH, body, deadline_monotonic=self._deadline_monotonic,
                                               clock=self._clock)
            else:
                stream = self.transport.stream(CHAT_PATH, body)
            try:
                self._check_deadline()
            except BaseException:
                stream.close()
                raise
            with self._lock:
                self._streams[identifier] = stream
        except _CancelledBeforeOpen:
            pass
        except TimeoutError as exc:
            self._abort_reason = "llama.cpp stream opening exceeded its deadline; server idleness is unverified"
            exc.abort_campaign = True
            raise
        finally:
            with self._lock:
                self._opening_requests.discard(identifier)
        if cancelled_before_open:
            yield self._cancelled_event(identifier, started=False)
            return
        try:
            if identifier in self._cancelled:
                stream.close()
                yield self._cancelled_event(identifier, started=True)
                return
            for event, data in stream:
                if identifier in self._cancelled:
                    yield self._cancelled_event(identifier, started=True)
                    return
                if isinstance(data, dict) and isinstance(data.get("model"), str) and data["model"] != instance_id:
                    raise VerificationError("Stream was served by a different model alias", instance_id=instance_id,
                                            mismatches={"model": {"requested": instance_id,
                                                                  "effective": data["model"], "reason": "different"}})
                yield StreamEvent(identifier, time.monotonic(), event, data)
            if identifier in self._cancelled:
                yield self._cancelled_event(identifier, started=True)
        except TransportError:
            if identifier not in self._cancelled:
                raise
            yield self._cancelled_event(identifier, started=True)
        finally:
            stream.close()
            with self._lock:
                self._streams.pop(identifier, None)
                if identifier not in self._pending_requests:
                    self._forget(identifier)

    def cancel(self, request_id: str) -> bool:
        """Close only this adapter's stream, then look for an idle slot. Never a global cancel API.

        The return value says whether the request was known. Whether the server was afterwards
        observed idle is reported on the stream's ``cancelled`` event.
        """
        with self._lock:
            stream = self._streams.get(request_id)
            if stream is None and request_id not in self._pending_requests and request_id not in self._opening_requests:
                return False
            self._cancelled.add(request_id)
            done = self._cancel_done.setdefault(request_id, threading.Event())
        verified = False
        try:
            if stream is not None:
                stream.close()
                verified = self.wait_idle()
        finally:
            with self._lock:
                if request_id in self._cancel_done:
                    self._idle_verified[request_id] = verified
            done.set()
        return True

    def wait_idle(self, timeout_seconds: float = 5.0) -> bool:
        """True only when ``/slots`` was read and no slot reports ``is_processing``."""
        if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        deadline = self._clock() + timeout_seconds
        while True:
            try:
                slots = self._request("GET", "/slots", any_json=True, cleanup_deadline=deadline)
            except (BackendError, OSError):
                slots = None
            if slots is not None:
                if (not isinstance(slots, list) or not slots
                        or any(not isinstance(item, dict) or type(item.get("is_processing")) is not bool
                               for item in slots)):
                    return False  # Unreadable state is not idleness.
                if not any(item["is_processing"] for item in slots):
                    return True
            if self._clock() >= deadline:
                return False
            self._sleep(min(0.1, max(0.0, deadline - self._clock())))

    def close(self) -> None:
        """Close local streams only. The server and its model belong to the host runner."""
        with self._lock:
            streams = list(self._streams.values())
            self._closed = True
        for stream in streams:
            try:
                stream.close()
            except Exception:
                pass


def _event_data(event: Any) -> Any:
    if isinstance(event, StreamEvent):
        return event.data
    if isinstance(event, Mapping) and "data" in event and "event" in event:
        return event["data"]  # a capture ``llmbench_raw_stream`` row
    return event


def native_timings(events: Iterable[Any]) -> dict[str, Any] | None:
    """The last well-formed llama.cpp ``timings`` object, or None. Nothing is derived or filled in.

    ``draft_n``/``draft_n_accepted`` are proposal statistics. They are carried as separate
    fields and never added to ``predicted_n``, which already is the accepted output count.
    """
    found = None
    for event in events:
        data = _event_data(event)
        timings = data.get("timings") if isinstance(data, Mapping) else None
        if not isinstance(timings, Mapping):
            continue
        row: dict[str, Any] = {}
        for key in _TIMING_COUNTS:
            row[key] = _count(timings.get(key))
        for key in _TIMING_NUMBERS:
            value = timings.get(key)
            row[key] = (float(value) if type(value) in {int, float} and math.isfinite(value) and value >= 0
                        else None)
        if any(value is None for value in row.values()):
            found = None
            continue
        for key in _DRAFT_COUNTS:
            if key in timings:
                row[key] = _count(timings[key])
        found = row
    return found


def observation_with_native_timings(probe: Any, *, require_cold_prompt: bool = True) -> SpeedObservation:
    """Merge server timings into ``probe.observation``; any inconsistency yields a failed copy."""
    observation: SpeedObservation = probe.observation
    timings = native_timings(probe.events)

    def failed() -> SpeedObservation:
        return replace(observation, accepted_tokens_verified=False,
                       status="failed" if observation.status == "completed" else observation.status)

    if timings is None or observation.status != "completed" or not observation.accepted_tokens_verified:
        return failed()
    if timings["predicted_n"] != observation.output_tokens:
        return failed()
    if observation.input_tokens is None or timings["prompt_n"] + timings["cache_n"] != observation.input_tokens:
        return failed()
    if require_cold_prompt and timings["cache_n"] != 0:
        return failed()
    generation_seconds = timings["predicted_ms"] / 1000.0
    if generation_seconds <= 0 or generation_seconds > observation.elapsed_seconds:
        return failed()  # A server duration longer than the observed request is not usable evidence.
    return replace(observation, native_generation_seconds=generation_seconds,
                   prompt_processing_seconds=timings["prompt_ms"] / 1000.0,
                   native_tokens_per_second_reported=timings["predicted_per_second"] or None,
                   native_timing_source=NATIVE_TIMING_SOURCE)


class _ContextEvidenceCallback:
    def __init__(self, callback: Any, n_ctx_slot: int, overflow_policy: str, save: Any) -> None:
        self._callback, self._n_ctx_slot = callback, n_ctx_slot
        self._policy, self._save = overflow_policy, save

    @property
    def abort_reason(self) -> str | None:
        return getattr(self, "_abort_reason", None) or getattr(self._callback, "abort_reason", None)

    def _persist(self, record: dict[str, Any]) -> None:
        if self._save is None:
            return
        try:
            self._save(record)
        except BaseException as exc:
            from ..evaluations.capture import UnsafeCaptureRuntimeState
            self._abort_reason = f"Quality evidence persistence failed: {type(exc).__name__}: {exc}"
            raise UnsafeCaptureRuntimeState(self._abort_reason) from exc

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._callback(request)
        except BaseException as exc:
            self._persist({"llmbench_error": f"{type(exc).__name__}: {exc}",
                           "llmbench_raw_stream": deepcopy(getattr(exc, "raw_events", []))})
            raise
        if type(response) is dict:
            usage = response.get("usage")
            timings = native_timings(response.get("llmbench_raw_stream") or [response])
            prompt = _count(usage.get("prompt_tokens")) if isinstance(usage, dict) else None
            completion = _count(usage.get("completion_tokens")) if isinstance(usage, dict) else None
            untruncated = (prompt is not None and completion is not None and timings is not None
                           and timings["prompt_n"] + timings["cache_n"] == prompt
                           and prompt + completion <= self._n_ctx_slot)
            # Only positive evidence is asserted. Anything else stays unknown (None), never True/False by guess.
            response["input_truncated"] = False if untruncated else None
            response["llmbench_context_policy"] = self._policy
            response["llmbench_native_timings"] = timings
        self._persist(response)
        return response


def context_evidence_callback(callback: Any, *, n_ctx_slot: int, overflow_policy: str,
                              save: Callable[[dict[str, Any]], None] | None = None) -> Any:
    """Wrap a capture completion callback with the runtime context evidence capture.py reads.

    ``overflow_policy`` must be "reject-overflow-no-shift" only when the overflow probe passed.
    """
    if type(n_ctx_slot) is not int or n_ctx_slot < 1:
        raise ValueError("n_ctx_slot must be a positive integer")
    if overflow_policy not in CONTEXT_POLICIES:
        raise ValueError("unknown context policy evidence")
    return _ContextEvidenceCallback(callback, n_ctx_slot, overflow_policy, save)


def speed_record(probe: Any, observation: SpeedObservation, *, label: str, index: int) -> dict[str, Any]:
    """JSON-ready raw record of one probe; draft statistics stay separate from accepted output."""
    timings = native_timings(probe.events)
    return {"label": label, "index": index, "observation": asdict(observation),
            "bridge_observation": asdict(probe.observation), "native_timings": timings,
            "draft": {key: timings.get(key) for key in _DRAFT_COUNTS} if timings else None,
            "error": probe.error, "transport": probe.transport, "deadline_exceeded": probe.deadline_exceeded,
            "worker_terminated": probe.worker_terminated, "cancellation_requested": probe.cancellation_requested,
            "events": [asdict(event) for event in probe.events]}
