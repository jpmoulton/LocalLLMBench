"""Review regression tests: fake HTTP/model responses only; no sockets, GPU, or Docker."""
import asyncio
import io
import threading
import time

import pytest

from llmbench.backends.http import TransportDeadlineExceeded, UrllibHTTPTransport
from llmbench.container_eval import EvaluationContext, PlainArtifacts, run_evaluation
from llmbench.containers.artifacts import RunArtifacts
from llmbench.evaluations.capture import UnsafeCaptureRuntimeState, backend_completion_callback
from llmbench.registry import RegistryError

from test_container_eval import ALLOWED, ScriptedServer, evaluate, make_config
from test_backends_llamacpp import FakeLlamaTransport, backend_for


def test_rev045_expired_template_never_starts_next_route_or_inference(tmp_path):
    now = [0.0]
    config = make_config()
    class SlowTemplate(ScriptedServer):
        def request_any(self, method, path, payload=None):
            result = super().request_any(method, path, payload)
            if path == "/apply-template":
                now[0] += 2.0
            return result
    server = SlowTemplate(config)
    result, _, _ = evaluate(tmp_path, config, server=server, deadline_monotonic=1.0, clock=lambda: now[0])
    assert server.paths() == [("GET", "/props"), ("GET", "/v1/models"), ("GET", "/slots"),
                              ("POST", "/apply-template")]
    assert result["abort_reason"] and result["count_route"]["exact"] is False
    assert len(result["samples"]) == 3 and all(row["passed"] is False for row in result["samples"])


def test_rev045_deadline_between_tokenizer_and_count_probe_rejects_generation(tmp_path):
    now = [0.0]
    config = make_config()
    class SlowTokenizer(ScriptedServer):
        def request_any(self, method, path, payload=None):
            result = super().request_any(method, path, payload)
            if path == "/tokenize":
                now[0] = 2.0
            return result
    server = SlowTokenizer(config)
    result, _, _ = evaluate(tmp_path, config, server=server, deadline_monotonic=1.0, clock=lambda: now[0])
    assert ("POST", "/v1/chat/completions") not in server.paths()
    assert ("POST", "/completion") not in server.paths()
    assert result["abort_reason"]


def test_rev049_failed_count_evidence_stops_before_overflow(tmp_path):
    class BrokenCountEvidence(PlainArtifacts):
        def write(self, name, data):
            if name == "count-route.json":
                raise OSError("simulated disk exhaustion")
            return super().write(name, data)
    artifacts = BrokenCountEvidence(tmp_path / "evidence")
    result, server, _ = evaluate(tmp_path, artifacts=artifacts)
    assert result["abort_reason"] and "count-route.json" in result["abort_reason"]
    assert ("POST", "/completion") not in server.paths()
    assert not any(method == "STREAM" for method, _ in server.paths())
    assert len(result["samples"]) == 3


def test_rev047_seed_cannot_disguise_repeated_static_task_before_backend_creation(tmp_path):
    config = make_config(benchmarks=[
        {"benchmark_id": "tool-probes", "revision": "local-tools-v1", "task_ids": ["tools/no-call-v1"], "seed": 42},
        {"benchmark_id": "tool-probes", "revision": "local-tools-v1", "task_ids": ["tools/no-call-v1"], "seed": 43},
    ])
    def forbidden_factory(*args, **kwargs):
        pytest.fail("a backend was constructed before duplicate static task validation")
    with pytest.raises(RegistryError, match="selected twice"):
        run_evaluation(config, EvaluationContext("http://127.0.0.1:18080", tmp_path / "out",
                       time.monotonic() + 60, ALLOWED), backend_factory=forbidden_factory)
    assert not (tmp_path / "out").exists()


def test_rev045_quality_callback_recomputes_remaining_budget_for_each_request():
    backend = backend_for(FakeLlamaTransport())
    backend.attach()
    now = [0.0]
    callback = backend_completion_callback(backend, backend.expected.alias, session_lock=ALLOWED,
        mode="live", timeout_seconds=60, deadline_monotonic=1.0, clock=lambda: now[0])
    now[0] = 2.0
    with pytest.raises(UnsafeCaptureRuntimeState, match="deadline"):
        asyncio.run(callback({"messages": [{"role": "user", "content": "hello"}]}))
    assert not backend.pending_request_ids
    assert not any(method == "STREAM" for method, _ in backend.transport.paths())


class _Response(io.BytesIO):
    headers = {"Content-Type": "application/json"}


class _BlockedBody(_Response):
    def __init__(self):
        super().__init__(b"{}")
        self.started, self.released, self.closed_event = threading.Event(), threading.Event(), threading.Event()
    def read(self, size=-1):
        self.started.set()
        self.released.wait(2)
        return b"{}"
    def close(self):
        self.released.set()
        super().close()
        self.closed_event.set()


def test_rev045_http_bounds_blocked_body_and_poison_prevents_another_request():
    response = _BlockedBody()
    class Opener:
        calls = []
        def open(self, request, timeout):
            self.calls.append(timeout)
            return response
    transport = UrllibHTTPTransport(timeout=30)
    transport._opener = Opener()
    start = time.monotonic()
    with pytest.raises(TransportDeadlineExceeded):
        transport.request_any("POST", "/v1/chat/completions", {}, deadline_monotonic=start + 0.03)
    assert time.monotonic() - start < 0.5
    assert response.closed_event.wait(0.5)
    assert 0 < transport._opener.calls[0] <= 0.030001
    with pytest.raises(TransportDeadlineExceeded, match="poisoned"):
        transport.request_any("POST", "/v1/chat/completions", {}, deadline_monotonic=time.monotonic() + 1)
    assert len(transport._opener.calls) == 1


def test_rev045_late_opened_response_is_closed_without_body_read():
    release, returned = threading.Event(), threading.Event()
    response = _Response(b"{}")
    class Opener:
        def open(self, request, timeout):
            release.wait(1)
            returned.set()
            return response
    transport = UrllibHTTPTransport(timeout=30)
    transport._opener = Opener()
    with pytest.raises(TransportDeadlineExceeded):
        transport.request_json("GET", "/health", deadline_monotonic=time.monotonic() + 0.03)
    assert not returned.is_set()
    release.set()
    assert returned.wait(0.5)
    for _ in range(100):
        if response.closed:
            break
        time.sleep(0.001)
    assert response.closed


def test_rev046_inspect_growth_is_metered_and_every_disk_file_is_indexed(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_VERSION", raising=False)  # include the real view notification path
    artifacts = RunArtifacts(tmp_path / "all", max_bytes=2_000_000)
    result, _, _ = evaluate(tmp_path, artifacts=artifacts)
    assert result["errors"] == [] and result["abort_reason"] is None
    assert len(result["inspect_logs"]) == 2 and all(path.endswith(".json") for path in result["inspect_logs"])
    indexed = {row["path"]: row["size"] for row in artifacts.index()}
    actual = {p.relative_to(artifacts.root).as_posix(): p.stat().st_size
              for p in artifacts.root.rglob("*") if p.is_file()}
    assert indexed == actual
    assert "inspect/runtime/last-eval.json" in indexed
    assert any(name.startswith("inspect/runtime/diagnostics-") for name in indexed)
    assert sum(actual.values()) <= artifacts.max_bytes


def test_rev046_quota_failure_poison_blocks_later_quality_requests(tmp_path):
    class TightInspectBudget(RunArtifacts):
        tightened = False
        def open_external(self, relative, mode="wb"):
            if relative.startswith("inspect/") and not self.tightened:
                self.max_bytes = self._bytes + 100
                self.tightened = True
            return super().open_external(relative, mode)
    artifacts = TightInspectBudget(tmp_path / "quota", max_bytes=2_000_000)
    result, server, _ = evaluate(tmp_path, artifacts=artifacts)
    assert artifacts.tightened
    assert result["abort_reason"] and "Inspect" in result["abort_reason"]
    assert len(result["samples"]) == 3
    assert not any(method == "STREAM" and not body.get("ignore_eos") for method, _, body in server.calls)
    assert sum(p.stat().st_size for p in artifacts.root.rglob("*") if p.is_file()) <= artifacts.max_bytes


def test_rev045_container_threads_absolute_deadline_into_quality_callback(tmp_path, monkeypatch):
    import llmbench.evaluations.capture as capture
    original = capture.backend_completion_callback
    received = []
    def tracked(*args, **kwargs):
        received.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(capture, "backend_completion_callback", tracked)
    deadline = time.monotonic() + 30
    result, _, _ = evaluate(tmp_path, deadline_monotonic=deadline)
    assert result["errors"] == [] and received[0]["deadline_monotonic"] == deadline
    assert received[0]["timeout_seconds"] == make_config().bounds.request_timeout_seconds


def test_rev045_quality_request_timer_shrinks_with_remaining_budget(monkeypatch):
    import llmbench.evaluations.capture as capture
    original = capture.threading.Timer
    intervals = []
    def timer(interval, function):
        intervals.append(interval)
        return original(interval, function)
    monkeypatch.setattr(capture.threading, "Timer", timer)
    backend = backend_for(FakeLlamaTransport())
    backend.attach()
    now = [0.0]
    callback = backend_completion_callback(backend, backend.expected.alias, session_lock=ALLOWED, mode="live",
        timeout_seconds=60, deadline_monotonic=10.0, clock=lambda: now[0])
    request = {"messages": [{"role": "user", "content": "hello"}]}
    asyncio.run(callback(request))
    now[0] = 9.0
    asyncio.run(callback(request))
    assert 9.5 < intervals[0] <= 10 and 0.5 < intervals[1] <= 1


def test_rev045_request_persistence_cannot_extend_admission_past_deadline():
    backend = backend_for(FakeLlamaTransport())
    backend.attach()
    now = [0.0]
    def sink(event):
        now[0] = 2.0
    callback = backend_completion_callback(backend, backend.expected.alias, session_lock=ALLOWED, mode="live",
        event_sink=sink, timeout_seconds=60, deadline_monotonic=1.0, clock=lambda: now[0])
    with pytest.raises(UnsafeCaptureRuntimeState, match="persisting request evidence"):
        asyncio.run(callback({"messages": [{"role": "user", "content": "hello"}]}))
    assert not backend.pending_request_ids and not any(method == "STREAM" for method, _ in backend.transport.paths())


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_rev045_interrupted_http_wait_closes_late_response_and_poisons(monkeypatch, control):
    response = _Response(b"{}")
    release = threading.Event()
    class Opener:
        def open(self, request, timeout):
            release.wait(1)
            return response
    transport = UrllibHTTPTransport(timeout=30)
    transport._opener = Opener()
    real_wait = threading.Event.wait
    def wait(event, timeout=None):
        if threading.current_thread() is threading.main_thread() and timeout is not None and timeout > 0.5:
            raise control()
        return real_wait(event, timeout)
    with monkeypatch.context() as scoped:
        scoped.setattr(threading.Event, "wait", wait)
        with pytest.raises(control):
            transport.request_any("GET", "/health", deadline_monotonic=time.monotonic() + 1)
    release.set()
    for _ in range(100):
        if response.closed:
            break
        time.sleep(0.001)
    assert response.closed
    with pytest.raises(TransportDeadlineExceeded, match="poisoned"):
        transport.request_any("GET", "/health")


@pytest.mark.parametrize("failure_route", ["open-timeout", "url-timeout", "body-timeout"])
def test_rev045_socket_timeout_poisons_transport_before_another_post(failure_route):
    from urllib.error import URLError
    class Response(_Response):
        def read(self, size=-1):
            raise TimeoutError("socket body timeout")
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            if failure_route == "body-timeout":
                return Response(b"{}")
            failure = TimeoutError("socket header timeout")
            raise URLError(failure) if failure_route == "url-timeout" else failure
    transport = UrllibHTTPTransport(timeout=30)
    transport._opener = Opener()
    with pytest.raises(TransportDeadlineExceeded):
        transport.request_json("POST", "/v1/chat/completions", {}, deadline_monotonic=time.monotonic() + 1)
    with pytest.raises(TransportDeadlineExceeded, match="poisoned"):
        transport.request_json("POST", "/v1/chat/completions", {}, deadline_monotonic=time.monotonic() + 1)
    assert transport._opener.calls == 1
