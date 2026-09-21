"""llama.cpp adapter contract tests. Every response is an in-memory fake modelled on the
real b11011 captures in tests/data; no socket, server, container or GPU is touched."""

from copy import deepcopy
import asyncio
import json
from pathlib import Path
import threading
import time

import pytest

from llmbench.backends.base import BackendError, LivePermission, OperationDenied, VerificationError
from llmbench.backends.http import TransportError
from llmbench.backends.llamacpp import (
    INFERENCE_URL, ExpectedServer, LlamaCppBackend, context_evidence_callback, native_timings,
    observation_with_native_timings, speed_record,
)
from llmbench.backends.measurement_bridge import run_speed_probe
from llmbench.safety import OperationForbidden, SessionLock

DATA = Path(__file__).parent / "data"
ALIAS = "qwen3.8-27b-q4_k_m"
BUILD = "b11011-aa39d7a3e"
ROUTES = json.loads((DATA / "route-checks-b11011-mtp.json").read_text(encoding="utf-8"))
OVERFLOW_BODY = json.dumps({"error": {"code": 400, "type": "exceed_context_size_error", "n_ctx": 8192,
                                      "message": "request (8256 tokens) exceeds the available context size "
                                                 "(8192 tokens), try increasing it",
                                      "n_prompt_tokens": 8256}}).encode()


def fixture_body(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))["body"]


def render(payload):
    """Stand-in for the server template: deterministic and sensitive to messages and tools."""
    return "<|im_start|>" + json.dumps({"m": payload["messages"], "t": payload.get("tools", [])}, sort_keys=True)


def chat_chunks(*, content=("Hello", " there", " friend", "."), finish="length", prompt_tokens=60,
                completion_tokens=None, timings=None, tool_call=None, reasoning=None, model=ALIAS):
    """Chunk order and shapes copied from route-checks-b11011-mtp.json ``stream_chat``."""
    base = {key: value for key, value in ROUTES["stream_chat"]["first_chunk"].items() if key != "choices"}
    base["model"] = model
    def chunk(delta, reason=None):
        return ("message", {**deepcopy(base), "choices": [{"finish_reason": reason, "index": 0, "delta": delta}]})
    rows = [chunk({"role": "assistant", "content": None})]
    if reasoning:
        rows.append(chunk({"reasoning_content": reasoning}))
    rows.extend(chunk({"content": text}) for text in content)
    if tool_call:
        rows.append(chunk({"tool_calls": [{"index": 0, **deepcopy(tool_call)}]}))
    rows.append(chunk({}, finish))
    completion = len(content) if completion_tokens is None else completion_tokens
    final = deepcopy(ROUTES["stream_chat"]["last_chunk"])
    final["model"] = model
    final["usage"] = {"completion_tokens": completion, "prompt_tokens": prompt_tokens,
                      "total_tokens": completion + prompt_tokens, "prompt_tokens_details": {"cached_tokens": 0}}
    final["timings"] = ({"cache_n": 0, "prompt_n": prompt_tokens, "prompt_ms": 0.5, "predicted_n": completion,
                         "predicted_ms": 1.0, "predicted_per_second": completion * 1000.0}
                        if timings is None else timings)
    if final["timings"] is False:
        del final["timings"]
    return rows + [("message", final), ("done", "[DONE]")]


class FakeStream:
    def __init__(self, events, *, delay=0.0, block=None):
        self.events, self.delay, self.block = events, delay, block
        self.closed = False

    def __iter__(self):
        for event in self.events:
            if self.block is not None:
                self.block.wait(5)
            if self.closed:
                raise TransportError("Streaming connection failed or timed out")
            if self.delay:
                time.sleep(self.delay)
            yield event

    def close(self):
        self.closed = True
        if self.block is not None:
            self.block.set()


class FakeLlamaTransport:
    def __init__(self):
        self.props = fixture_body("props-b11011.json")
        self.models = fixture_body("v1-models-b11011.json")
        self.slots = fixture_body("slots-b11011.json")
        self.calls = []
        self.count_offset = {"no-tools": 0, "tools": 0}
        self.apply_template_response = None
        self.overflow = TransportError("HTTP 400 from the inference server /completion", status=400, body=OVERFLOW_BODY)
        self.processing_script = []  # successive is_processing answers; afterwards False
        self.stream_factory = lambda payload: FakeStream(chat_chunks(), delay=0.002)
        self.last_stream = None

    def request_json(self, method, path, payload=None):
        result = self.request_any(method, path, payload)
        if not isinstance(result, dict):
            raise TransportError("JSON response must be an object")
        return result

    def request_any(self, method, path, payload=None):
        self.calls.append((method, path, deepcopy(payload)))
        if (method, path) == ("GET", "/health"):
            return {"status": "ok"}
        if (method, path) == ("GET", "/props"):
            return deepcopy(self.props)
        if (method, path) == ("GET", "/v1/models"):
            return deepcopy(self.models)
        if (method, path) == ("GET", "/slots"):
            slots = deepcopy(self.slots)
            if self.processing_script:
                answer = self.processing_script.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                slots[0]["is_processing"] = answer
            return slots
        if (method, path) == ("POST", "/apply-template"):
            return ({"prompt": render(payload)} if self.apply_template_response is None
                    else deepcopy(self.apply_template_response))
        if (method, path) == ("POST", "/tokenize"):
            assert payload["add_special"] in {True, False} and payload["parse_special"] is True
            return {"tokens": list(range(len(payload["content"]) // 4 + 1))}
        if (method, path) == ("POST", "/v1/chat/completions"):
            assert payload["stream"] is False and payload["max_tokens"] == 1
            name = "tools" if payload.get("tools") else "no-tools"
            prompt = len(render(payload)) // 4 + 1 + self.count_offset[name]
            return {"model": ALIAS, "choices": [{"index": 0, "finish_reason": "length",
                                                 "message": {"role": "assistant", "content": "r"}}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": 1, "total_tokens": prompt + 1},
                    "timings": {**ROUTES["count_route"]["no_tools"]["timings"], "prompt_n": prompt}}
        if (method, path) == ("POST", "/completion"):
            if self.overflow is not None:
                raise self.overflow
            return {"content": "x", "truncated": True, "tokens_evaluated": 8191, "tokens_predicted": 1}
        raise AssertionError(f"Unexpected fake request: {method} {path}")

    def stream(self, path, payload):
        self.calls.append(("STREAM", path, deepcopy(payload)))
        self.last_stream = self.stream_factory(payload)
        return self.last_stream

    def paths(self):
        return [(method, path) for method, path, _ in self.calls]


def expected(**changes):
    return ExpectedServer(**{"alias": ALIAS, "n_ctx": 8192, "build_info": BUILD, **changes})


def backend_for(transport=None, *, server=None, **kwargs):
    kwargs.setdefault("permissions", LivePermission(False, True, False))
    kwargs.setdefault("session_lock", SessionLock(False, True, False))
    return LlamaCppBackend("http://127.0.0.1:8080", expected=server or expected(),
                           transport=transport or FakeLlamaTransport(), **kwargs)


# -- construction and guards ----------------------------------------------------------------

def test_constructor_is_inert_and_in_container_url_needs_explicit_remote_opt_in():
    transport = FakeLlamaTransport()
    backend_for(transport)
    assert transport.calls == []
    with pytest.raises(ValueError, match="allow_remote"):
        LlamaCppBackend(INFERENCE_URL, expected=expected(), transport=transport)
    with pytest.raises(ValueError, match="allow_remote"):
        LlamaCppBackend(INFERENCE_URL, expected=expected())  # the default transport is no loophole either
    LlamaCppBackend(INFERENCE_URL, expected=expected(), transport=transport, allow_remote=True)
    for origin in ("http://127.0.0.1:8080/v1", "http://user@127.0.0.1:8080", "ftp://127.0.0.1"):
        with pytest.raises(ValueError):
            LlamaCppBackend(origin, expected=expected(), transport=transport)
    assert transport.calls == []


def test_inference_operations_need_permission_and_session_lock(tmp_path):
    transport = FakeLlamaTransport()
    denied = LlamaCppBackend("http://127.0.0.1:8080", expected=expected(), transport=transport,
                             session_lock=SessionLock(True, True, True))
    denied.attach()
    reads = len(transport.calls)
    operations = (lambda b: b.tokenize("x"), lambda b: b.apply_template([], []), lambda b: b.count_chat_tokens([], []),
                  lambda b: b.verify_count_route(), lambda b: b.probe_overflow_rejection(),
                  lambda b: list(b.stream(ALIAS, {"messages": []})))
    for operation in operations:
        with pytest.raises(OperationDenied):
            operation(denied)
    locked = backend_for(transport, session_lock=SessionLock())
    locked.attach()
    reads = len(transport.calls)
    for operation in operations:
        with pytest.raises(OperationForbidden):
            operation(locked)
    assert len(transport.calls) == reads
    policy = tmp_path / "runtime-policy.json"
    reread = LlamaCppBackend("http://127.0.0.1:8080", expected=expected(), transport=transport,
                             permissions=LivePermission(False, True, False), policy_path=policy)
    with pytest.raises(OperationForbidden):
        reread.tokenize("x")
    policy.write_text(json.dumps({"allow_inference": True}), encoding="utf-8")
    assert reread.tokenize("abcdefgh") == [0, 1, 2]
    policy.write_text(json.dumps({"allow_inference": False}), encoding="utf-8")
    with pytest.raises(OperationForbidden):
        reread.tokenize("x")


# -- attach / readback ----------------------------------------------------------------------

def test_attach_verifies_real_fixture_identity_with_get_requests_only():
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    assert backend.health() == {"status": "ok"}
    verified = backend.attach()
    assert verified.model.instance_id == ALIAS and verified.model.owned is False
    assert verified.effective["n_ctx"] == 8192 and verified.effective["build_info"] == BUILD
    assert verified.effective["slots_n_ctx"] == [8192]
    assert verified.model.raw["slots"] == fixture_body("slots-b11011.json")
    assert {method for method, _ in transport.paths()} == {"GET"}
    json.dumps(verified.effective)


def test_slots_list_goes_through_request_any_and_non_list_is_rejected():
    transport = FakeLlamaTransport()
    assert isinstance(backend_for(transport).readback()["slots"], list)
    transport.slots = {"error": "slots endpoint disabled"}
    with pytest.raises(BackendError, match="/slots"):
        backend_for(transport).readback()


@pytest.mark.parametrize("server,key", [
    (expected(alias="llmbench-other"), "alias"), (expected(n_ctx=131072), "n_ctx"),
    (expected(build_info="b1-deadbeef"), "build_info"), (expected(model_path="/models/other.gguf"), "model_path"),
    (expected(total_slots=2), "total_slots"),
])
def test_attach_reports_each_mismatch_and_blocks_inference(server, key):
    backend = backend_for(server=server)
    with pytest.raises(VerificationError) as failure:
        backend.attach()
    assert failure.value.mismatches[key]["reason"] == "different"
    with pytest.raises(VerificationError):
        list(backend.stream(server.alias, {"messages": []}))


def test_attach_never_fills_missing_or_loosely_typed_readback():
    transport = FakeLlamaTransport()
    del transport.props["default_generation_settings"]["n_ctx"]
    with pytest.raises(VerificationError) as failure:
        backend_for(transport).attach()
    assert failure.value.mismatches["n_ctx"]["reason"] == "missing_readback"
    transport = FakeLlamaTransport()
    transport.props["total_slots"] = True  # bool is not the integer 1
    transport.slots[0]["n_ctx"] = 4096
    transport.models["data"].append({"id": "second-model"})
    with pytest.raises(VerificationError) as failure:
        backend_for(transport).attach()
    assert {"total_slots", "slots[0].n_ctx", "models.ids"} <= set(failure.value.mismatches)


def test_build_info_none_skips_only_that_comparison():
    transport = FakeLlamaTransport()
    transport.props["build_info"] = "anything"
    assert "build_info" not in backend_for(transport, server=expected(build_info=None)).attach().requested


# -- prompt accounting ----------------------------------------------------------------------

def test_count_chat_tokens_renders_then_tokenizes_with_capture_shaped_tool_fields():
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    messages = [{"role": "user", "content": "hello"}]
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    plain = backend.count_chat_tokens(messages, [])
    with_tool = backend.count_chat_tokens(messages, [tool])
    assert plain["rendered_prompt"] == render({"messages": messages}) and plain["messages"] == messages
    assert plain["tokens"] == len(plain["rendered_prompt"]) // 4 + 1 < with_tool["tokens"]
    assert "apply-template" in plain["method"]
    bodies = [payload for method, path, payload in transport.calls if path == "/apply-template"]
    assert "tools" not in bodies[0] and "tool_choice" not in bodies[0]
    assert bodies[1]["tool_choice"] == "auto" and bodies[1]["parallel_tool_calls"] is False
    tokenize = [payload for _, path, payload in transport.calls if path == "/tokenize"]
    assert all(item["add_special"] is True and item["parse_special"] is True for item in tokenize)


def test_unexpected_apply_template_or_tokenize_shape_fails_loudly():
    transport = FakeLlamaTransport()
    transport.apply_template_response = {"rendered": "different key"}
    with pytest.raises(BackendError, match="apply-template"):
        backend_for(transport).count_chat_tokens([{"role": "user", "content": "x"}], [])
    class BadTokens(FakeLlamaTransport):
        def request_any(self, method, path, payload=None):
            return {"tokens": ["a"]} if path == "/tokenize" else super().request_any(method, path, payload)
    with pytest.raises(BackendError, match="tokenize"):
        backend_for(BadTokens()).tokenize("x")


def test_count_route_exact_and_inexact():
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    with pytest.raises(VerificationError):
        backend.verify_count_route()
    backend.attach()
    result = backend.verify_count_route()
    assert result["exact"] is True and [case["name"] for case in result["cases"]] == ["no-tools", "tools"]
    assert all(case["counted_tokens"] == case["usage_prompt_tokens"] for case in result["cases"])
    chats = [payload for _, path, payload in transport.calls if path == "/v1/chat/completions"]
    assert all(item["model"] == ALIAS and item["max_tokens"] == 1 and item["cache_prompt"] is False for item in chats)
    assert "tools" not in chats[0] and chats[1]["tool_choice"] == "auto"
    json.dumps(result)
    transport.count_offset["tools"] = 3
    result = backend.verify_count_route()
    assert result["exact"] is False
    assert [case["match"] for case in result["cases"]] == [True, False]


def test_overflow_probe_requires_http_400_json_error_for_the_expected_context():
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    backend.attach()
    result = backend.probe_overflow_rejection()
    assert result["passed"] is True and result["status"] == 400 and result["reported_n_ctx"] == 8192
    sent = [payload for _, path, payload in transport.calls if path == "/completion"][0]
    assert len(sent["prompt"]) == 8192 + 64 == result["sent_tokens"] and sent["n_predict"] == 1
    assert all(type(item) is int for item in sent["prompt"]) and sent["cache_prompt"] is False
    json.dumps(result)
    failures = {
        "no body": TransportError("HTTP 400 from the inference server /completion", status=400),
        "unreachable": TransportError("Cannot reach the inference server /completion"),
        "not json": TransportError("HTTP 400", status=400, body=b"<html>bad request</html>"),
        "other status": TransportError("HTTP 500", status=500, body=OVERFLOW_BODY),
        "other type": TransportError("HTTP 400", status=400, body=b'{"error":{"type":"invalid_request_error"}}'),
        "other ctx": TransportError("HTTP 400", status=400,
                                    body=OVERFLOW_BODY.replace(b'"n_ctx": 8192', b'"n_ctx": 4096')),
        "accepted": None,
    }
    for name, failure in failures.items():
        transport.overflow = failure
        result = backend.probe_overflow_rejection()
        assert result["passed"] is False and result["detail"], name
    assert result["status"] == 200 and result["truncated"] is True


# -- streaming ------------------------------------------------------------------------------

def test_stream_forces_alias_usage_cache_prompt_and_preserves_events():
    transport = FakeLlamaTransport()
    backend = backend_for(transport, cache_prompt=False)
    backend.attach()
    payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "cache_prompt": True,
               "stream": False, "timings_per_token": True, "stream_options": {"include_usage": False, "x": 1}}
    events = list(backend.stream(ALIAS, payload, request_id="r1"))
    body = transport.calls[-1][2]
    assert transport.calls[-1][:2] == ("STREAM", "/v1/chat/completions")
    assert body["model"] == ALIAS and body["stream"] is True and body["cache_prompt"] is False
    assert body["timings_per_token"] is False and body["stream_options"] == {"include_usage": True, "x": 1}
    assert payload["cache_prompt"] is True  # caller payload is not mutated
    assert events[-1].event == "done" and events[-2].data["choices"] == [] and "timings" in events[-2].data
    assert {event.request_id for event in events} == {"r1"}
    assert transport.last_stream.closed and backend.pending_request_ids == ()
    assert backend_for(FakeLlamaTransport(), cache_prompt=True).cache_prompt is True


def test_stream_refuses_other_alias_payload_model_and_foreign_server_chunks():
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    backend.attach()
    with pytest.raises(VerificationError):
        list(backend.stream("some-other-alias", {"messages": []}))
    with pytest.raises(ValueError):
        list(backend.stream(ALIAS, {"model": "would-route-elsewhere", "messages": []}))
    with pytest.raises(ValueError):
        list(backend.stream(ALIAS, {"messages": [], "n": 2}))
    assert not any(method == "STREAM" for method, _ in transport.paths())
    transport.stream_factory = lambda payload: FakeStream(chat_chunks(model="a-different-model"))
    with pytest.raises(VerificationError, match="different model"):
        list(backend.stream(ALIAS, {"messages": []}))
    assert transport.last_stream.closed and backend.pending_request_ids == ()


def test_second_concurrent_reservation_or_stream_is_refused():
    backend = backend_for()
    backend.attach()
    backend.reserve_request("first")
    with pytest.raises(BackendError, match="still active"):
        backend.reserve_request("second")
    with pytest.raises(BackendError, match="one active"):
        list(backend.stream(ALIAS, {"messages": []}, request_id="second"))
    with pytest.raises(BackendError, match="still active"):
        backend.verify_count_route()
    backend.release_request("first")
    running = backend.stream(ALIAS, {"messages": []}, request_id="third")
    next(running)
    with pytest.raises(BackendError):
        backend.reserve_request("fourth")
    running.close()
    backend.reserve_request("fourth")
    assert backend.pending_request_ids == ("fourth",)


@pytest.mark.parametrize("script,verified", [([True, False], True), ([True] * 200, False),
                                             ([TransportError("Cannot reach the inference server /slots")] * 200, False)])
def test_cancel_closes_stream_and_reports_idle_verification(script, verified):
    transport = FakeLlamaTransport()
    now = [0.0]
    backend = backend_for(transport, clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    backend.attach()
    stream = backend.stream(ALIAS, {"messages": []}, request_id="r1")
    assert next(stream).event == "message"
    assert backend.cancel("someone-elses-request") is False
    transport.processing_script = list(script)
    assert backend.cancel("r1") is True
    assert transport.last_stream.closed
    remaining = list(stream)
    assert [event.event for event in remaining] == ["cancelled"]
    assert remaining[0].data == {"server_cancellation_verified": verified}
    assert backend.pending_request_ids == ()
    assert now[0] <= 5.2


def test_cancel_from_another_thread_while_the_stream_is_blocked():
    transport = FakeLlamaTransport()
    gate = threading.Event()
    transport.stream_factory = lambda payload: FakeStream(chat_chunks(), block=gate)
    backend = backend_for(transport)
    backend.attach()
    rows = []
    worker = threading.Thread(target=lambda: rows.extend(backend.stream(ALIAS, {"messages": []}, request_id="r1")))
    worker.start()
    for _ in range(500):
        if backend.pending_request_ids == ("r1",) and transport.last_stream is not None:
            break
        time.sleep(0.002)
    assert backend.cancel("r1") is True
    worker.join(5)
    assert not worker.is_alive()
    assert rows[-1].event == "cancelled" and rows[-1].data["server_cancellation_verified"] is True


def test_cancel_before_open_never_posts():
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    backend.attach()
    backend.reserve_request("pre-open")
    assert backend.cancel("pre-open") is True
    events = list(backend.stream(ALIAS, {"messages": []}, request_id="pre-open"))
    assert [event.event for event in events] == ["cancelled"]
    assert events[0].data == {"server_cancellation_verified": False, "request_started": False}
    assert not any(method == "STREAM" for method, _ in transport.paths())
    backend.release_request("pre-open")
    assert backend.pending_request_ids == ()
    assert [event.event for event in backend.stream(ALIAS, {"messages": []}, request_id="pre-open")][-1] == "done"


def test_wait_idle_rejects_unreadable_slot_state_and_close_blocks_further_use():
    transport = FakeLlamaTransport()
    backend = backend_for(transport)
    assert backend.wait_idle() is True
    transport.slots = [{"id": 0}]
    assert backend.wait_idle() is False
    transport.slots = []
    assert backend.wait_idle() is False
    with pytest.raises(ValueError):
        backend.wait_idle(0)
    backend.close()
    with pytest.raises(BackendError, match="closed"):
        backend.tokenize("x")


# -- native timings -------------------------------------------------------------------------

def probe_with(transport_events, *, max_tokens=4, expected_input_tokens=60):
    transport = FakeLlamaTransport()
    transport.stream_factory = lambda payload: FakeStream(transport_events, delay=0.01)  # > clock tick
    backend = backend_for(transport)
    verified = backend.attach()
    return run_speed_probe(backend, verified, {"messages": [{"role": "user", "content": "x"}],
                                               "max_tokens": max_tokens, "ignore_eos": True},
                           expected_input_tokens=expected_input_tokens, timeout_seconds=10)


def test_native_timings_reads_real_fixtures_and_rejects_malformed_values():
    real = ROUTES["stream_chat"]["last_chunk"]
    timings = native_timings([{"choices": []}, real])
    assert timings["predicted_n"] == 96 and timings["draft_n"] == 90 and timings["draft_n_accepted"] == 65
    nonstream = fixture_body("chat-tool-call-b11011.json")
    assert native_timings([nonstream])["prompt_n"] == 446 and "draft_n" not in native_timings([nonstream])
    assert native_timings([{"event": "message", "data": real, "request_id": "r", "monotonic_seconds": 1.0}]) == timings
    assert native_timings([]) is None and native_timings([{"timings": "fast"}]) is None
    for key, value in (("predicted_n", True), ("predicted_n", -1), ("predicted_ms", float("nan")), ("cache_n", 1.5)):
        assert native_timings([{"timings": {**real["timings"], key: value}}]) is None
    partial = dict(real["timings"])
    del partial["predicted_ms"]
    assert native_timings([{"timings": partial}]) is None


def test_timings_merge_into_the_bridge_observation():
    probe = probe_with(chat_chunks())
    assert probe.error is None and probe.transport == "openai-compatible-sse"
    assert probe.observation.native_generation_seconds is None  # the bridge alone never reports a native duration
    merged = observation_with_native_timings(probe)
    assert merged.status == "completed" and merged.accepted_tokens_verified is True
    assert merged.output_tokens == 4 and merged.input_tokens == 60
    assert merged.native_generation_seconds == pytest.approx(0.001)
    assert merged.prompt_processing_seconds == pytest.approx(0.0005)
    assert merged.native_tokens_per_second_reported == 4000.0
    assert merged.native_timing_source == "llama.cpp.timings"
    assert merged.metrics()["native_tokens_per_second"] == pytest.approx(4000.0)
    assert merged.metrics()["actual_context_verified"] is True
    record = speed_record(probe, merged, label="speed", index=0)
    assert record["label"] == "speed" and record["draft"] == {"draft_n": None, "draft_n_accepted": None}
    json.dumps(record)


@pytest.mark.parametrize("timings", [
    False,  # final chunk without timings: no silent fallback to wall-clock rates
    {"cache_n": 0, "prompt_n": 60, "prompt_ms": 0.5, "predicted_n": 5, "predicted_ms": 1.0, "predicted_per_second": 5e3},
    {"cache_n": 12, "prompt_n": 48, "prompt_ms": 0.5, "predicted_n": 4, "predicted_ms": 1.0, "predicted_per_second": 4e3},
    {"cache_n": 0, "prompt_n": 59, "prompt_ms": 0.5, "predicted_n": 4, "predicted_ms": 1.0, "predicted_per_second": 4e3},
    {"cache_n": 0, "prompt_n": 60, "prompt_ms": 0.5, "predicted_n": 4, "predicted_ms": 9e6, "predicted_per_second": 4e3},
    {"cache_n": 0, "prompt_n": 60, "prompt_ms": 0.5, "predicted_n": 4, "predicted_ms": 0.0, "predicted_per_second": 0.0},
])
def test_missing_or_inconsistent_timings_give_a_failed_unverified_copy(timings):
    probe = probe_with(chat_chunks(timings=timings))
    assert probe.observation.status == "completed"
    merged = observation_with_native_timings(probe)
    assert merged.status == "failed" and merged.accepted_tokens_verified is False
    assert merged.native_generation_seconds is None and merged.native_timing_source is None
    assert merged.metrics()["native_tokens_per_second"] is None


def test_warm_cache_is_acceptable_only_when_cold_prompt_is_not_required():
    warm = {"cache_n": 12, "prompt_n": 48, "prompt_ms": 0.5, "predicted_n": 4, "predicted_ms": 1.0,
            "predicted_per_second": 4e3}
    probe = probe_with(chat_chunks(timings=warm))
    assert observation_with_native_timings(probe, require_cold_prompt=False).status == "completed"
    assert observation_with_native_timings(probe).status == "failed"


def test_draft_counts_are_never_added_to_accepted_tokens():
    speculative = {"cache_n": 0, "prompt_n": 60, "prompt_ms": 0.5, "predicted_n": 4, "predicted_ms": 1.0,
                   "predicted_per_second": 4e3, "draft_n": 9, "draft_n_accepted": 3}
    probe = probe_with(chat_chunks(timings=speculative))
    merged = observation_with_native_timings(probe)
    assert merged.status == "completed" and merged.output_tokens == 4
    assert merged.metrics()["native_tokens_per_second"] == pytest.approx(4 / 0.001)
    record = speed_record(probe, merged, label="speed", index=1)
    assert record["draft"] == {"draft_n": 9, "draft_n_accepted": 3}
    assert record["observation"]["output_tokens"] == record["native_timings"]["predicted_n"] == 4
    # A server that counted proposals as output would disagree with usage and fail instead of inflating speed.
    inflated = {**speculative, "predicted_n": 4 + 9}
    assert observation_with_native_timings(probe_with(chat_chunks(timings=inflated))).status == "failed"


def test_interrupted_probe_stays_interrupted_and_unverified():
    probe = probe_with(chat_chunks()[:-3])  # no finish reason, usage, timings or [DONE]
    merged = observation_with_native_timings(probe)
    assert merged.status == "interrupted" and merged.accepted_tokens_verified is False


# -- context evidence wrapper ---------------------------------------------------------------

def response_with(*, prompt=446, completion=159, timings="fixture", raw_stream=True):
    body = fixture_body("chat-tool-call-b11011.json")
    final = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
    if timings == "fixture":
        final["timings"] = body["timings"]
    elif timings is not None:
        final["timings"] = timings
    response = {"choices": body["choices"], "usage": final["usage"]}
    if raw_stream:
        response["llmbench_raw_stream"] = [{"request_id": "r", "monotonic_seconds": 1.0, "event": "message", "data": final}]
    return response


def wrapped_call(response, **kwargs):
    async def callback(request):
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)
    callback.abort_reason = None
    wrapper = context_evidence_callback(callback, **{"n_ctx_slot": 8192, "overflow_policy": "reject-overflow-no-shift",
                                                     **kwargs})
    return wrapper, asyncio.run(wrapper({"messages": []}))


def test_context_evidence_sets_untruncated_only_on_positive_evidence():
    saved = []
    _, result = wrapped_call(response_with(), save=saved.append)
    assert result["input_truncated"] is False and result["llmbench_context_policy"] == "reject-overflow-no-shift"
    assert result["llmbench_native_timings"]["prompt_n"] == 446 and saved == [result]
    unknown = [response_with(timings=None), response_with(prompt=445), response_with(prompt=8100, completion=159,
               timings={**fixture_body("chat-tool-call-b11011.json")["timings"], "prompt_n": 8100}),
               {"choices": [], "usage": None}]
    for response in unknown:
        assert wrapped_call(response)[1]["input_truncated"] is None
    _, result = wrapped_call(response_with(), overflow_policy="unknown")
    assert result["input_truncated"] is False and result["llmbench_context_policy"] == "unknown"
    with pytest.raises(ValueError):
        context_evidence_callback(lambda request: None, n_ctx_slot=8192, overflow_policy="probably-fine")
    with pytest.raises(ValueError):
        context_evidence_callback(lambda request: None, n_ctx_slot=0, overflow_policy="unknown")


def test_context_evidence_wrapper_exposes_abort_reason_and_records_failures():
    saved = []
    failure = RuntimeError("stream ended with error")
    failure.raw_events = [{"event": "message", "data": {"choices": []}}]
    async def failing(request):
        raise failure
    failing.abort_reason = None
    wrapper = context_evidence_callback(failing, n_ctx_slot=8192, overflow_policy="unknown", save=saved.append)
    with pytest.raises(RuntimeError):
        asyncio.run(wrapper({"messages": []}))
    assert saved[0]["llmbench_error"].startswith("RuntimeError") and saved[0]["llmbench_raw_stream"] == failure.raw_events
    failing.abort_reason = "worker did not terminate"
    assert wrapper.abort_reason == "worker did not terminate"
    def broken_save(record):
        raise OSError("artifact budget exhausted")
    wrapper, _ = None, None
    async def fine(request):
        return response_with()
    wrapper = context_evidence_callback(fine, n_ctx_slot=8192, overflow_policy="unknown", save=broken_save)
    with pytest.raises(Exception) as unsafe:
        asyncio.run(wrapper({"messages": []}))
    assert getattr(unsafe.value, "abort_campaign", False) is True and "persistence" in wrapper.abort_reason
