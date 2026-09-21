"""Evaluator wiring tests. The backend is the real adapter over an in-memory fake transport
modelled on the b11011 captures; Inspect really schedules and scores. No network, Docker or GPU."""

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from llmbench.backends.http import TransportError
from llmbench.backends.llamacpp import INFERENCE_URL, LlamaCppBackend
from llmbench.config import AcceptancePolicy
from llmbench.container_eval import (
    RESULT_KEYS, EvaluationContext, PlainArtifacts, main, run_evaluation, self_check, speed_observations_from,
)
from llmbench.containers.config import ContainerRunConfig
from llmbench.evaluations.tools import strict_argument_fixture
from llmbench.measurement import summarize_speed
from llmbench.registry import builtin_registry
from llmbench.safety import OperationForbidden, SessionLock

from test_backends_llamacpp import FakeLlamaTransport, FakeStream, chat_chunks, fixture_body, render

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples" / "candidate.json").read_text(encoding="utf-8"))
ALLOWED = SessionLock(False, True, False)


def make_config(**changes):
    data = json.loads(json.dumps(EXAMPLE))
    data.update({"requested_input_tokens": 700, "speed": {"repetitions": 2, "warmup_repetitions": 1,
                                                          "output_tokens": 16, "ignore_eos": True},
                 "benchmarks": [
                     {"benchmark_id": "tool-probes", "revision": "local-tools-v1",
                      "task_ids": ["tools/nested-exact-v1", "tools/no-call-v1"]},
                     {"benchmark_id": "niah", "revision": "local-niah-v1", "task_ids": ["single-middle"]}]})
    data["generation"]["max_output_tokens"] = 64
    data.update(changes)
    return ContainerRunConfig.model_validate_json(json.dumps(data))


class ScriptedServer(FakeLlamaTransport):
    """Answers like the pinned server: speed requests run to the limit, probes get tool/text replies."""

    def __init__(self, config, *, nested="pilot"):
        super().__init__()
        self.alias = config.alias()
        self.props["model_alias"] = self.alias
        self.models["data"][0]["id"] = self.alias
        self.nested = nested
        self.quality_failure = None
        self.stream_factory = self.respond

    def request_any(self, method, path, payload=None):
        result = super().request_any(method, path, payload)
        if path == "/v1/chat/completions":
            result["model"] = self.alias
        return result

    def respond(self, payload):
        prompt_tokens = len(render(payload)) // 4 + 1
        common = {"prompt_tokens": prompt_tokens, "model": self.alias}
        if payload.get("ignore_eos"):
            return FakeStream(chat_chunks(content=tuple(f" w{index}" for index in range(payload["max_tokens"])),
                                          **common), delay=0.002)
        if self.quality_failure is not None:
            raise self.quality_failure
        user = payload["messages"][-1]["content"]
        if "Requested keys:" in user:
            found = dict(re.findall(r"CURRENT (key_\w+): (pass_\w+)", user))
            keys = json.loads(user.split("Requested keys: ", 1)[1].strip())
            return FakeStream(chat_chunks(content=(json.dumps({key: found.get(key) for key in keys}),),
                                          finish="stop", **common))
        if "configure_job" in user:
            pilot = fixture_body("chat-tool-call-b11011.json")["choices"][0]["message"]
            call = json.loads(json.dumps(pilot["tool_calls"][0]))
            if self.nested == "correct":
                expected = strict_argument_fixture().calls[0]
                call["function"] = {"name": expected.name, "arguments": json.dumps(expected.arguments)}
            return FakeStream(chat_chunks(content=(), reasoning=pilot["reasoning_content"], tool_call=call,
                                          finish="tool_calls", completion_tokens=159, **common))
        return FakeStream(chat_chunks(content=("No tool is needed.",), finish="stop", **common))


def evaluate(tmp_path, config=None, *, server=None, lock=ALLOWED, **context):
    config = config or make_config()
    server = server or ScriptedServer(config)
    built = []
    def factory(base_url, **kwargs):
        built.append(kwargs)
        return LlamaCppBackend(base_url, transport=server, **kwargs)
    import time
    ctx = EvaluationContext(**{"base_url": "http://127.0.0.1:18080", "artifacts_dir": tmp_path / "evaluator",
                               "deadline_monotonic": time.monotonic() + 120, "session_lock": lock, **context})
    return run_evaluation(config, ctx, backend_factory=factory), server, built


def first_index(paths, item):
    return paths.index(item)


def test_full_order_artifacts_and_real_inspect_scoring(tmp_path):
    pytest.importorskip("inspect_ai")  # the optional `eval` extra; without it the quality stage has no scorer
    config = make_config()
    result, server, built = evaluate(tmp_path, config)
    assert set(RESULT_KEYS) <= set(result) and result == json.loads(json.dumps(result))
    assert result["errors"] == [] and result["abort_reason"] is None
    assert [stage["name"] for stage in result["stages"]] == ["attach", "count-route", "overflow-probe", "speed", "quality"]
    assert all(stage["status"] == "ok" for stage in result["stages"])

    paths = server.paths()
    streams = [index for index, item in enumerate(paths) if item[0] == "STREAM"]
    order = [first_index(paths, ("GET", "/props")), first_index(paths, ("POST", "/v1/chat/completions")),
             first_index(paths, ("POST", "/completion")), streams[0]]
    assert order == sorted(order) and len(streams) == 3 + 3  # 1 warmup + 2 speed, then three quality requests
    bodies = [payload for method, _, payload in server.calls if method == "STREAM"]
    assert [bool(body.get("ignore_eos")) for body in bodies] == [True, True, True, False, False, False]
    assert all(body["model"] == config.alias() and body["cache_prompt"] is False
               and body["stream_options"]["include_usage"] is True for body in bodies)
    assert bodies[0]["max_tokens"] == 16 and bodies[0]["seed"] == 42 and bodies[0]["temperature"] == 0
    assert built[0]["allow_remote"] is False and built[0]["permissions"].allow_model_load is False
    assert built[0]["expected"].n_ctx == 8192 and built[0]["expected"].build_info == "b11011-aa39d7a3e"
    assert built[0]["timeout"] == config.bounds.request_timeout_seconds

    assert result["count_route"]["exact"] is True and result["tokenizer_verified"] is True
    assert result["overflow_probe"]["passed"] is True
    assert result["readback"]["props"]["model_alias"] == config.alias()
    assert len(result["speed_observations"]) == 2 and len(result["warmups"]) == 1
    observations = speed_observations_from(result)
    assert all(item.status == "completed" and item.output_tokens == 16 and item.accepted_tokens_verified
               and item.native_timing_source == "llama.cpp.timings" and item.input_tokens == item.expected_input_tokens
               for item in observations)
    assert 600 < observations[0].input_tokens <= 700
    summary = summarize_speed(observations, AcceptancePolicy(speed_repetitions=2))
    assert summary["attempted"] == 2 and not any("native_token_evidence_missing" in item for item in summary["reasons"])

    samples = {sample["task_id"]: sample for sample in result["samples"]}
    assert list(samples) == ["tools/nested-exact-v1", "tools/no-call-v1", "niah/single-middle/development/seed-42"]
    # The pilot capture called create_ticket, not the fixture's tool: scored as a failure, kept in the denominator.
    assert samples["tools/nested-exact-v1"]["passed"] is False and samples["tools/nested-exact-v1"]["status"] == "completed"
    assert samples["tools/nested-exact-v1"]["raw_response"]["reasoning_content"].startswith("The user is asking")
    assert samples["tools/no-call-v1"]["passed"] is True
    needle = samples["niah/single-middle/development/seed-42"]
    assert needle["passed"] is True and needle["category"] == "retrieval"
    assert needle["context"]["actual_context_verified"] is True
    assert needle["context"]["counting_method"] == "exact-post-template"
    assert needle["context"]["context_policy"] == "reject-overflow-no-shift"
    assert needle["context"]["tokenizer_id"] == "gguf-sha256:" + config.assets[0].sha256
    assert needle["context"]["template_id"].startswith("sha256:")
    assert all(sample["model_evaluated"] and not sample["synthetic"] for sample in samples.values())
    assert result["actual_context_verified"] is True

    root = tmp_path / "evaluator"
    names = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and "inspect" not in path.parts}
    assert {"readback.json", "count-route.json", "overflow-probe.json", "evaluation.json", "speed/input.json",
            "speed/transport.jsonl", "speed/warmup-0.json", "speed/speed-0.json", "speed/speed-1.json",
            "quality/transport.jsonl", "quality/quality-request-0.json", "quality/quality-request-1.json",
            "quality/quality-request-2.json"} <= names
    assert json.loads((root / "evaluation.json").read_text(encoding="utf-8")) == result
    warmup = json.loads((root / "speed/warmup-0.json").read_text(encoding="utf-8"))
    assert warmup["label"] == "warmup" and warmup["native_timings"]["predicted_n"] == 16
    trace = [json.loads(line) for line in (root / "quality/transport.jsonl").read_text(encoding="utf-8").splitlines()]
    assert trace[0]["event"] == "capture.request" and any(row["event"] == "capture.completed" for row in trace)
    saved = json.loads((root / "quality/quality-request-2.json").read_text(encoding="utf-8"))
    assert saved["input_truncated"] is False and saved["llmbench_native_timings"]["cache_n"] == 0
    from inspect_ai.log import read_eval_log
    log_files = list((root / "inspect").glob("*.json"))
    assert log_files and not list((root / "inspect").glob("*.eval"))
    assert all(read_eval_log(str(path)).status == "success" for path in log_files)


def test_correct_native_tool_call_passes_through_the_scripted_stream(tmp_path):
    config = make_config(benchmarks=[{"benchmark_id": "tool-probes", "revision": "local-tools-v1",
                                      "task_ids": ["tools/nested-exact-v1"]}])
    result, _, _ = evaluate(tmp_path, config, server=ScriptedServer(config, nested="correct"))
    assert [sample["passed"] for sample in result["samples"]] == [True]
    assert result["actual_context_verified"] is False  # no retrieval sample, so no context claim


def test_denied_policy_or_bad_selection_never_constructs_a_backend(tmp_path):
    for lock in (SessionLock(), SessionLock(True, False, True)):
        with pytest.raises(OperationForbidden):
            evaluate(tmp_path, lock=lock)
    calls = []
    import time
    ctx = EvaluationContext("http://127.0.0.1:18080", tmp_path / "e", time.monotonic() + 60, ALLOWED)
    with pytest.raises(OperationForbidden):
        run_evaluation(make_config(), EvaluationContext("http://127.0.0.1:18080", tmp_path / "e", time.monotonic() + 60,
                                                        policy_path=tmp_path / "absent-policy.json"),
                       backend_factory=lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="registry_digest"):
        run_evaluation(make_config(registry_digest="0" * 64), ctx, backend_factory=lambda *a, **k: calls.append(1))
    run_config = make_config(registry_digest=builtin_registry().digest())
    assert run_config.registry_digest == builtin_registry().digest()
    # tool-eval-bench has no adapter, so it is still genuinely import-only.
    bad = make_config(benchmarks=[{"benchmark_id": "tool-eval-bench", "revision": "upstream-pin-required",
                                   "task_ids": ["HumanEval/0"]}])
    with pytest.raises(ValueError, match="import-only"):
        run_evaluation(bad, ctx, backend_factory=lambda *a, **k: calls.append(1))
    # A wired suite has code but no baked dataset here: refused for that reason, still no backend.
    unbaked = make_config(benchmarks=[{"benchmark_id": "evalplus", "revision": "evalplus/mbpp-plus@v0.2.0",
                                       "task_ids": ["evalplus/mbpp-plus/Mbpp/2"]}])
    with pytest.raises(ValueError, match="requires"):
        run_evaluation(unbaked, ctx, backend_factory=lambda *a, **k: calls.append(1))
    assert calls == [] and not (tmp_path / "e").exists()


def test_attach_mismatch_aborts_before_any_inference_and_keeps_the_denominator(tmp_path):
    config = make_config()
    server = ScriptedServer(config)
    server.props["default_generation_settings"]["n_ctx"] = 4096
    result, server, _ = evaluate(tmp_path, config, server=server)
    assert {method for method, _ in server.paths()} == {"GET"}
    assert result["abort_reason"].startswith("attach failed") and result["errors"][0]["stage"] == "attach"
    assert [sample["status"] for sample in result["samples"]] == ["environment_error"] * 3
    assert all(sample["score"] == 0.0 and sample["passed"] is False for sample in result["samples"])
    assert result["speed_observations"] == [] and result["actual_context_verified"] is False
    assert (tmp_path / "evaluator" / "evaluation.json").exists()


def test_inexact_count_route_and_failed_overflow_probe_withhold_the_context_claim(tmp_path):
    config = make_config()
    server = ScriptedServer(config)
    server.count_offset["tools"] = 2
    result, _, _ = evaluate(tmp_path / "a", config, server=server)
    needle = result["samples"][-1]
    assert result["tokenizer_verified"] is False and needle["passed"] is True
    assert needle["context"]["counting_method"] == "estimated" and result["actual_context_verified"] is False
    server = ScriptedServer(config)
    server.overflow = TransportError("HTTP 500 from the inference server /completion", status=500, body=b"{}")
    result, _, _ = evaluate(tmp_path / "b", config, server=server)
    needle = result["samples"][-1]
    assert result["overflow_probe"]["passed"] is False and needle["context"]["context_policy"] == "unknown"
    assert needle["context"]["verification_status"] == "context_policy_unverified"
    assert result["actual_context_verified"] is False


def test_unsupported_count_route_is_recorded_not_papered_over(tmp_path):
    config = make_config()
    server = ScriptedServer(config)
    server.apply_template_response = {"unexpected": "shape"}
    result, _, _ = evaluate(tmp_path, config, server=server)
    stages = {stage["name"]: stage["status"] for stage in result["stages"]}
    assert stages["count-route"] == "failed" and stages["speed"] == "failed" and stages["quality"] == "failed"
    assert result["count_route"]["exact"] is False and result["speed_observations"] == []
    assert {error["stage"] for error in result["errors"]} == {"count-route", "speed", "quality"}
    assert len(result["samples"]) == 3 and all(sample["status"] == "environment_error" for sample in result["samples"])


def test_quality_transport_errors_stay_in_the_denominator(tmp_path):
    config = make_config()
    server = ScriptedServer(config)
    server.quality_failure = TransportError("Cannot reach the inference server /v1/chat/completions")
    result, _, _ = evaluate(tmp_path, config, server=server)
    assert len(result["samples"]) == 3 and len(result["speed_observations"]) == 2
    assert [sample["status"] for sample in result["samples"]] == ["transport_error"] * 3
    assert all(sample["score"] == 0.0 for sample in result["samples"])
    failed = json.loads((tmp_path / "evaluator/quality/quality-request-0.json").read_text(encoding="utf-8"))
    assert failed["llmbench_error"].startswith("TransportError")


def test_missing_native_timings_fail_speed_without_a_wall_clock_fallback(tmp_path):
    config = make_config()
    server = ScriptedServer(config)
    respond = server.respond
    def without_timings(payload):
        stream = respond(payload)
        if payload.get("ignore_eos"):
            del stream.events[-2][1]["timings"]
        return stream
    server.stream_factory = without_timings
    result, _, _ = evaluate(tmp_path, config, server=server)
    assert [row["status"] for row in result["speed_observations"]] == ["failed", "failed"]
    assert all(row["native_generation_seconds"] is None and row["accepted_tokens_verified"] is False
               for row in result["speed_observations"])
    assert result["native_timings"] == [None, None]


def test_timed_out_speed_probe_aborts_and_skips_quality(tmp_path):
    import threading
    data = json.loads(json.dumps(EXAMPLE))
    data["bounds"] = {"request_timeout_seconds": 1}
    config = make_config(bounds=data["bounds"])
    server = ScriptedServer(config)
    server.stream_factory = lambda payload: FakeStream(chat_chunks(model=server.alias), block=threading.Event())
    result, server, _ = evaluate(tmp_path, config, server=server)
    assert "unverified" in result["abort_reason"]
    assert [method for method, _ in server.paths()].count("STREAM") == 1  # no further request after the abort
    assert result["warmups"][0]["status"] == "timed_out" and result["speed_observations"] == []
    assert result["warmups"][0]["cancellation_acknowledged"] is True  # /slots showed an idle slot after the close
    assert [sample["status"] for sample in result["samples"]] == ["environment_error"] * 3
    assert [stage["name"] for stage in result["stages"]][-1] == "speed"


def test_exhausted_wall_budget_marks_every_task_as_timeout(tmp_path):
    config = make_config()
    now = [0.0]
    def clock():
        now[0] += 1.0
        return now[0]
    result, server, _ = evaluate(tmp_path, config, deadline_monotonic=12.0, clock=clock)
    assert len(result["samples"]) == 3
    assert {sample["status"] for sample in result["samples"]} <= {"timeout", "environment_error"}
    assert any(error["error_type"] == "TimeoutError" for error in result["errors"])
    assert result["stages"][-1]["status"] in {"timeout", "failed"}


def test_shared_budgeted_writer_is_used_and_exhaustion_aborts_instead_of_dropping_evidence(tmp_path):
    from llmbench.containers.artifacts import RunArtifacts
    shared = RunArtifacts(tmp_path / "shared", 1_000_000_000)
    result, _, _ = evaluate(tmp_path, artifacts=shared, artifacts_dir=tmp_path / "shared")
    assert result["errors"] == [] and "speed/speed-1.json" in {entry["path"] for entry in shared.index()}
    tiny = RunArtifacts(tmp_path / "tiny", 2_000)
    result, server, _ = evaluate(tmp_path, artifacts=tiny, artifacts_dir=tmp_path / "tiny")
    assert "budget" in result["abort_reason"] and len(result["samples"]) == 3
    assert not any(method == "STREAM" for method, _ in server.paths())


def test_plain_artifacts_are_confined_and_exclusive(tmp_path):
    artifacts = PlainArtifacts(tmp_path / "out")
    artifacts.write("a/b.json", b"{}")
    with pytest.raises(FileExistsError):
        artifacts.write("a/b.json", b"{}")
    for rel in ("../escape.json", "/abs.json", "a/../../x", "C:/x.json", ""):
        with pytest.raises(ValueError):
            artifacts.write(rel, b"x")
    with artifacts.trace("t.jsonl") as sink:
        sink({"event": "x"})
    assert json.loads((tmp_path / "out/t.jsonl").read_text(encoding="utf-8")) == {"event": "x"}
    assert not (tmp_path / "escape.json").exists()


def test_main_accepts_only_config_or_self_check(tmp_path, capsys):
    for argv in ([], ["--config"], ["--config", "x.json", "--self-check"], ["--config", "x.json", "--base-url", "http://x"],
                 ["--policy", "p.json"], ["--self"], ["positional"]):
        assert main(argv) == 2
    assert main(["--config", str(tmp_path / "missing.json")]) == 2
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({**EXAMPLE, "unknown_field": 1}), encoding="utf-8")
    assert main(["--config", str(broken)]) == 2
    capsys.readouterr()


def test_main_runs_the_container_path_with_explicit_remote_opt_in(tmp_path, capsys):
    config = make_config()
    path = tmp_path / "run" / "config.json"
    path.parent.mkdir()
    path.write_text(config.model_dump_json(), encoding="utf-8")
    server = ScriptedServer(config)
    built = []
    def factory(base_url, **kwargs):
        built.append((base_url, kwargs))
        return LlamaCppBackend(base_url, transport=server, **kwargs)
    argv = ["--config", str(path), "--grant-seconds", "600"]
    assert main(argv, artifacts_dir=tmp_path / "artifacts", backend_factory=factory) == 2
    assert built == []  # no policy beside the config: refused before any client exists
    (path.parent / "runtime-policy.json").write_text(json.dumps({"allow_inference": True}), encoding="utf-8")
    assert main(["--config", str(path)], artifacts_dir=tmp_path / "artifacts", backend_factory=factory) == 2
    assert built == []  # the container path needs the host's typed grant
    assert main(argv, artifacts_dir=tmp_path / "artifacts", backend_factory=factory) == 0
    assert built[0][0] == INFERENCE_URL and built[0][1]["allow_remote"] is True
    written = json.loads((tmp_path / "artifacts" / "evaluation.json").read_text(encoding="utf-8"))
    assert len(written["samples"]) == 3
    built.clear()
    assert main(["--config", str(path)], base_url="http://127.0.0.1:18080", artifacts_dir=tmp_path / "loopback",
                backend_factory=factory) == 0
    assert built[0][1]["allow_remote"] is False
    capsys.readouterr()


def test_self_check_reports_registry_and_tasks_without_network(monkeypatch, capsys):
    import socket
    def refuse(*args, **kwargs):
        raise AssertionError("self-check must not open sockets")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    report = self_check()
    assert report["ok"] is True and report["registry_digest"] == builtin_registry().digest()
    assert set(report["tasks"]) == {"tool-probes", "tool-episodes", "niah"}
    assert all(not item["missing"] for item in report["tasks"].values())
    assert main(["--self-check"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_import_has_no_side_effects(tmp_path):
    code = ("import sys, socket\n"
            "def refuse(*a, **k): raise SystemExit('network touched on import')\n"
            "socket.socket.connect = refuse\n"
            "import llmbench.container_eval, llmbench.backends.llamacpp, llmbench.registry\n"
            "import llmbench.evaluations.selection\n"
            "heavy = [name for name in ('inspect_ai', 'lmstudio') if name in sys.modules]\n"
            "print(heavy)\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr
    assert list(tmp_path.iterdir()) == []


def test_selection_builds_named_niah_cases_and_the_same_denominator():
    from llmbench.config import GenerationSettings, TaskSelection
    from llmbench.containers.config import BenchmarkSelection
    from llmbench.evaluations.selection import build_selected_cases, expected_task_rows
    benchmarks = [
        BenchmarkSelection(benchmark_id="tool-episodes", revision="local-tools-v1", task_ids=("tools/read-edit-test-v1",)),
        BenchmarkSelection(benchmark_id="niah", revision="local-niah-v1", task_ids=("multi", "tool-multi"),
                           split="holdout", seed=7),
        TaskSelection(suite="coding", revision="private-coding-v1", task_ids=("python/chunks",)),
    ]
    calls = []
    def counter(messages, tools):
        calls.append(bool(tools))
        return {"tokens": len(json.dumps([messages, tools])) // 4}
    common = {"requested_input_tokens": 900, "ctx_size": 2048, "generation": GenerationSettings(max_output_tokens=128),
              "counter": counter, "tokenizer_id": "gguf-sha256:" + "0" * 64, "template_id": "sha256:" + "1" * 64}
    cases, selections = build_selected_cases(benchmarks, **common)
    assert [case.task_id for case in cases] == ["niah/multi/holdout/seed-7", "niah/tool-multi/holdout/seed-7"]
    assert [case.mode for case in cases] == ["text", "tool"] and any(calls) and not all(calls)
    assert all(case.accounting.counting_method == "estimated" and 800 < case.accounting.actual_input_tokens <= 900
               and case.accounting.reserved_output_tokens == 128 and case.split == "holdout" and case.seed == 7
               for case in cases)
    assert [item.suite for item in selections] == ["tool-episodes", "niah"]  # coding is never an Inspect task
    assert selections[1].task_ids == tuple(case.task_id for case in cases) and selections[1].fixture_seed == 7
    verified, _ = build_selected_cases(benchmarks[1:2], **common, tokenizer_verified=True)
    assert verified[0].accounting.counting_method == "exact-post-template"
    assert verified[0].accounting.tokenizer_id == common["tokenizer_id"]
    rows = expected_task_rows(benchmarks)
    assert [row["task_id"] for row in rows] == ["tools/read-edit-test-v1", "niah/multi/holdout/seed-7",
                                                "niah/tool-multi/holdout/seed-7", "python/chunks"]
    assert [row["category"] for row in rows] == ["tools", "retrieval", "retrieval", "coding"]
    with pytest.raises(ValueError):
        build_selected_cases([TaskSelection(suite="niah", revision="local-niah-v1", task_ids=("unknown",))], **common)
    with pytest.raises(ValueError):
        build_selected_cases(benchmarks, **common, tokenizer_verified="yes")


def container_main_setup(tmp_path, config=None):
    config = config or make_config()
    path = tmp_path / "run" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(), encoding="utf-8")
    (path.parent / "runtime-policy.json").write_text(json.dumps({"allow_inference": True}), encoding="utf-8")
    server = ScriptedServer(config)
    built = []

    def factory(base_url, **kwargs):
        built.append((base_url, kwargs))
        return LlamaCppBackend(base_url, transport=server, **kwargs)
    return config, path, server, built, factory


def test_main_grant_seconds_bounds_the_deadline_below_evaluation_seconds(tmp_path, capsys):
    import time
    config, path, server, built, factory = container_main_setup(tmp_path)
    assert config.bounds.evaluation_seconds == 1200
    before = time.monotonic()
    assert main(["--config", str(path), "--grant-seconds", "7"], artifacts_dir=tmp_path / "a", backend_factory=factory) == 0
    deadline = built[0][1]["deadline_monotonic"]
    assert before + 6 <= deadline <= before + 7.5 and built[0][1]["allow_remote"] is True
    written = json.loads((tmp_path / "a" / "evaluation.json").read_text(encoding="utf-8"))
    assert [stage["name"] for stage in written["stages"]][:2] == ["wait-ready", "attach"]
    assert written["readiness"]["ready"] is True and len(written["samples"]) == 3
    built.clear()
    before = time.monotonic()
    assert main(["--config", str(path), "--grant-seconds", "99999", "--artifact-bytes", "50000000"],
                artifacts_dir=tmp_path / "b", backend_factory=factory) == 0
    assert built[0][1]["deadline_monotonic"] <= before + 1200 + 1  # bounded by bounds.evaluation_seconds
    assert (tmp_path / "b" / "evaluation.json").exists()
    built.clear()
    assert main(["--config", str(path), "--grant-seconds", "600", "--artifact-bytes", "3000"],
                artifacts_dir=tmp_path / "c", backend_factory=factory) == 3  # the child cap aborts honestly
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["exit_code"] == 3 and "budget" in summary["abort_reason"] and summary["samples"] == 3
    assert summary["persist_errors"] >= 1 and summary["stages"][:2] == ["wait-ready", "attach"]
    # 3000 bytes cannot hold an evaluation that embeds the readback: the file is absent, never truncated, and
    # the host then fails closed on ingestion while the summary line above survives in the captured log.
    assert not (tmp_path / "c" / "evaluation.json").exists()


def test_finish_falls_back_to_a_compact_evaluation_when_the_full_dump_exceeds_the_cap():
    from llmbench.container_eval import _finish
    from llmbench.evaluations.selection import expected_task_rows
    rows = expected_task_rows(make_config().benchmarks)

    class Capped:
        def __init__(self, failures):
            self.failures, self.written = failures, {}

        def write(self, rel, data):
            if rel in self.written:
                raise FileExistsError(rel)
            if self.failures:
                self.failures -= 1
                raise ValueError("artifact budget exhausted")
            self.written[rel] = data

    def fresh():
        return {"samples": [], "speed_observations": [], "readback": {"props": {"chat_template": "x" * 300}},
                "count_route": None, "overflow_probe": None, "actual_context_verified": False,
                "abort_reason": None, "errors": [], "stages": []}

    artifacts = Capped(failures=1)
    result = _finish(fresh(), rows, artifacts, "environment_error", "count-route failed")
    assert [item["stage"] for item in result["errors"]] == ["persist"]
    assert result["abort_reason"].startswith("evaluation.json was not written: ValueError: artifact budget")
    data = artifacts.written["evaluation.json"]
    assert b"\n" not in data and json.loads(data) == result  # compact, byte-identical content to the return value
    assert [sample["status"] for sample in result["samples"]] == ["environment_error"] * 3
    assert all(sample["reason"] == "count-route failed" for sample in result["samples"])
    both = Capped(failures=2)
    result = _finish(fresh(), rows, both, "timeout", "wall budget exhausted")
    assert both.written == {} and [item["stage"] for item in result["errors"]] == ["persist", "persist-compact"]
    assert len(result["samples"]) == 3 and result["abort_reason"].startswith("evaluation.json was not written")
    clean = Capped(failures=0)
    result = _finish(fresh(), rows, clean, "environment_error", "unused")
    assert result["errors"] == [] and result["abort_reason"] is None and b"\n " in clean.written["evaluation.json"]


def test_main_rejects_missing_or_invalid_grant_and_policy(tmp_path, capsys):
    config, path, server, built, factory = container_main_setup(tmp_path)
    for argv in (["--config", str(path)], ["--config", str(path), "--grant-seconds", "0"],
                 ["--config", str(path), "--grant-seconds", "-5"], ["--config", str(path), "--grant-seconds", "ten"],
                 ["--config", str(path), "--grant-seconds"], ["--config", str(path), "--grant-seconds", "60", "--artifact-bytes", "0"],
                 ["--config", str(path), "--grant-seconds", "60", "--policy", str(tmp_path / "absent.json")],
                 ["--self-check", "--policy", str(path)], ["--self-check", "--artifact-bytes", "5"]):
        assert main(argv, artifacts_dir=tmp_path / "never", backend_factory=factory) == 2, argv
    assert built == [] and not (tmp_path / "never").exists()
    denied = tmp_path / "denied-policy.json"
    denied.write_text(json.dumps({"allow_inference": False}), encoding="utf-8")
    assert main(["--config", str(path), "--grant-seconds", "60", "--policy", str(denied), "--artifact-bytes", "5000000"],
                artifacts_dir=tmp_path / "never", backend_factory=factory) == 2
    assert built == [] and not (tmp_path / "never").exists()
    explicit = tmp_path / "explicit-policy.json"
    explicit.write_text(json.dumps({"allow_inference": True}), encoding="utf-8")
    assert main(["--config", str(path), "--grant-seconds", "60", "--policy", str(explicit)],
                artifacts_dir=tmp_path / "explicit", backend_factory=factory) == 0
    assert built[0][1]["policy_path"] == explicit
    # Loopback (host-process style) invocations keep working without a grant.
    built.clear()
    assert main(["--config", str(path)], base_url="http://127.0.0.1:18080", artifacts_dir=tmp_path / "loop",
                backend_factory=factory) == 0
    assert built[0][1]["allow_remote"] is False
    capsys.readouterr()


def test_wait_ready_polls_within_grant_and_times_out_honestly():
    from llmbench.container_eval import wait_ready

    class Transport:
        def __init__(self, failures, answer=None):
            self.failures, self.answer, self.requests, self.timeout = failures, answer, 0, 30.0

        def request_json(self, method, path, payload=None):
            assert (method, path, payload) == ("GET", "/health", None)
            self.requests += 1
            if self.requests <= self.failures:
                raise TransportError("Cannot reach the inference server /health")
            return self.answer or {"status": "ok"}

    class Backend:
        def __init__(self, transport):
            self.transport = transport

    now = [100.0]
    slept = []

    def clock():
        return now[0]

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    transport = Transport(failures=3)
    report = wait_ready(Backend(transport), deadline=110.0, clock=clock, sleep=sleep, poll_seconds=2.0)
    assert report["ready"] is True and report["attempts"] == 4 and report["elapsed_seconds"] == 6.0
    # The per-request timeout is min(5 s, time left): 4 s remained at the fourth, successful attempt.
    assert slept == [2.0, 2.0, 2.0] and transport.timeout == 4.0 and "TransportError" in report["last_error"]
    now[0], slept[:] = 100.0, []
    never = Transport(failures=10**6)
    report = wait_ready(never, deadline=105.0, clock=clock, sleep=sleep, poll_seconds=2.0)
    assert report["ready"] is False and report["reason"] and never.requests == 3 and now[0] == 105.0
    assert slept == [2.0, 2.0, 1.0] and never.timeout == 1.0
    now[0] = 100.0
    loading = Transport(failures=0, answer={"status": "loading model"})
    assert wait_ready(loading, deadline=101.0, clock=clock, sleep=sleep, poll_seconds=0.5)["ready"] is False
    assert wait_ready(Transport(0), deadline=99.0, clock=clock, sleep=sleep) == {
        "ready": False, "attempts": 0, "elapsed_seconds": 0.0, "last_error": None,
        "reason": "inference did not report healthy before the deadline"}
    for bad in ({"deadline": float("nan")}, {"deadline": 200.0, "poll_seconds": 0}):
        with pytest.raises(ValueError):
            wait_ready(Transport(0), clock=clock, sleep=sleep, **bad)


def test_read_evaluation_enforces_bytes_schema_and_denominator(tmp_path):
    from llmbench.container_eval import read_evaluation
    from llmbench.evaluations.selection import expected_task_rows
    config = make_config()
    rows = expected_task_rows(config.benchmarks)
    ids = [row["task_id"] for row in rows]

    def sample(task_id, category="tools", **extra):
        return {"task_id": task_id, "category": category, "score": 1.0, "status": "completed", **extra}

    good = {key: None for key in RESULT_KEYS}
    good.update(samples=[sample(ids[0]), sample(ids[1]), sample(ids[2], "retrieval")], speed_observations=[],
                actual_context_verified=False, errors=[])
    path = tmp_path / "evaluation.json"

    def write(value, raw=None):
        path.write_bytes(raw if raw is not None else json.dumps(value).encode("utf-8"))
        return path

    assert read_evaluation(write(good), max_bytes=100_000, expected_rows=rows) == good
    size = path.stat().st_size
    with pytest.raises(ValueError, match="allocation"):
        read_evaluation(path, max_bytes=size - 1, expected_rows=rows)
    assert read_evaluation(path, max_bytes=size, expected_rows=rows)["samples"][0]["task_id"] == ids[0]
    for raw in (b"", b"[1, 2]", b"{", b'"text"', b"\xff\xfe", json.dumps({"samples": []}).encode()):
        with pytest.raises(ValueError):
            read_evaluation(write(None, raw), max_bytes=100_000, expected_rows=rows)
    bad_shapes = [{**good, "samples": {}}, {**good, "errors": "x"}, {**good, "abort_reason": 5},
                  {**good, "actual_context_verified": "yes"}, {**good, "speed_observations": None},
                  {**good, "samples": [{"task_id": ids[0]}] + good["samples"][1:]},
                  {**good, "samples": good["samples"][:2]},
                  {**good, "samples": good["samples"] + [sample(ids[0])]},
                  {**good, "samples": good["samples"] + [sample("tools/unknown")]}]
    for value in bad_shapes:
        with pytest.raises(ValueError):
            read_evaluation(write(value), max_bytes=100_000, expected_rows=rows)
    with pytest.raises(ValueError, match="missing"):
        read_evaluation(write({**good, "samples": good["samples"][:2]}), max_bytes=100_000, expected_rows=rows)
    with pytest.raises(ValueError, match="unreadable"):
        read_evaluation(tmp_path / "absent.json", max_bytes=100_000, expected_rows=rows)
    with pytest.raises(ValueError):
        read_evaluation(write(good), max_bytes=0, expected_rows=rows)
    try:
        import os
        os.symlink(path, tmp_path / "link.json")
    except OSError:
        return
    with pytest.raises(ValueError, match="regular"):
        read_evaluation(tmp_path / "link.json", max_bytes=100_000, expected_rows=rows)


def test_coding_hook_is_called_only_for_coding_selections_and_rows_stay_in_denominator(tmp_path):
    from llmbench.coding.fixtures import fixtures
    fixture_ids = [item.fixture_id for item in fixtures()]
    coding = {"benchmark_id": "coding", "revision": "private-coding-v1", "task_ids": fixture_ids}
    with pytest.raises(Exception):
        make_config(benchmarks=[*make_config().benchmarks[0:1], coding])  # coding needs config.broker
    base = make_config()
    tools = json.loads(base.benchmarks[0].model_dump_json())
    config = make_config(benchmarks=[tools, coding], broker={})
    calls = []

    def hook(cfg, callback, remaining, artifacts, lock, *, client=None):
        calls.append((cfg, callback, remaining(), artifacts, lock, client))
        return [{"task_id": fixture_ids[0], "category": "coding", "score": 1.0, "status": "completed",
                 "model_evaluated": True, "synthetic": False}]

    result, server, _ = evaluate(tmp_path / "hooked", config, coding_hook=hook, coding_client="direct-client")
    assert len(calls) == 1 and calls[0][0] is config and calls[0][5] == "direct-client"
    assert calls[0][2] > 0 and calls[0][4] is ALLOWED and hasattr(calls[0][1], "abort_reason")
    assert [stage["name"] for stage in result["stages"]] == ["attach", "count-route", "overflow-probe", "speed",
                                                             "quality", "coding"]
    samples = {sample["task_id"]: sample for sample in result["samples"]}
    assert set(samples) == set(tools["task_ids"]) | set(fixture_ids)
    assert samples[fixture_ids[0]]["status"] == "completed" and samples[fixture_ids[0]]["score"] == 1.0
    for missing in fixture_ids[1:]:  # the hook returned one row; the other fixtures stay in the denominator
        assert samples[missing]["status"] == "environment_error" and samples[missing]["score"] == 0.0
    assert not any(sample["category"] == "coding" and sample["status"] == "completed"
                   for sample in samples.values() if sample["task_id"] != fixture_ids[0])

    calls.clear()
    result, _, _ = evaluate(tmp_path / "no-coding", base, coding_hook=hook)  # no coding selection: never called
    assert calls == [] and "coding" not in [stage["name"] for stage in result["stages"]]
    assert len(result["samples"]) == 3

    result, _, _ = evaluate(tmp_path / "no-hook", config)  # selection without a hook: rows fail, never vanish
    assert [sample["status"] for sample in result["samples"] if sample["category"] == "coding"] == ["environment_error"] * 3

    def broken(cfg, callback, remaining, artifacts, lock, *, client=None):
        raise RuntimeError("broker unreachable")
    result, _, _ = evaluate(tmp_path / "broken", config, coding_hook=broken)
    assert result["errors"][-1]["stage"] == "coding" and result["stages"][-1]["status"] == "failed"
    assert len([s for s in result["samples"] if s["category"] == "coding"]) == 3

    def wrong_shape(cfg, callback, remaining, artifacts, lock, *, client=None):
        return "rows"
    result, _, _ = evaluate(tmp_path / "shape", config, coding_hook=wrong_shape)
    assert result["errors"][-1]["error_type"] == "TypeError"


def test_coding_stage_publishes_into_its_own_live_transport_sink(tmp_path):
    """LIVE-007: the coding hook's real requests reached a trace sink the quality stage had already closed."""
    import asyncio
    from llmbench.coding.fixtures import fixtures
    fixture_ids = [item.fixture_id for item in fixtures()]
    tools = json.loads(make_config().benchmarks[0].model_dump_json())
    config = make_config(broker={}, benchmarks=[tools, {"benchmark_id": "coding", "revision": "private-coding-v1",
                                                        "task_ids": fixture_ids}])
    marker = "CODING-PROMPT-MARKER"

    def hook(cfg, callback, remaining, artifacts, lock, *, client=None):
        # Mirrors coding.generation.ask: every fixture really awaits the supplied callback, so each request
        # publishes through the sink the evaluator bound it to instead of being answered by a recorded stub.
        rows = []
        for task_id in fixture_ids:
            request = {"messages": [{"role": "user", "content": f"{marker} {task_id}"}], "max_tokens": 32,
                       "temperature": 0, "seed": 42, "top_p": 1.0}
            response = asyncio.run(asyncio.wait_for(callback(request), timeout=remaining()))
            assert response["choices"][0]["message"]["content"] == "No tool is needed."
            rows.append({"task_id": task_id, "category": "coding", "score": 1.0, "passed": True,
                         "status": "completed", "model_evaluated": True, "synthetic": False})
        return rows

    result, _, _ = evaluate(tmp_path, config, coding_hook=hook, coding_client="direct-client")
    assert result["abort_reason"] is None and result["errors"] == []
    assert [stage["name"] for stage in result["stages"]] == ["attach", "count-route", "overflow-probe", "speed",
                                                             "quality", "coding"]
    assert all(stage["status"] == "ok" for stage in result["stages"])
    samples = {sample["task_id"]: sample for sample in result["samples"]}
    assert set(fixture_ids) <= set(samples)
    assert all(samples[task_id]["status"] == "completed" and samples[task_id]["score"] == 1.0
               for task_id in fixture_ids)

    root = tmp_path / "evaluator"
    trace = [json.loads(line) for line in (root / "coding/transport.jsonl").read_text(encoding="utf-8").splitlines()]
    requests = [row for row in trace if row["event"] == "capture.request"]
    assert len(requests) == len(fixture_ids) and all(marker in json.dumps(row["data"]) for row in requests)
    assert sum(1 for row in trace if row["event"] == "capture.completed") == len(fixture_ids)
    # The two stages keep separately identifiable raw evidence: no coding request lands in the quality trace.
    assert marker not in (root / "quality/transport.jsonl").read_text(encoding="utf-8")
    saved = [json.loads((root / f"coding/coding-request-{index}.json").read_text(encoding="utf-8"))
             for index in range(len(fixture_ids))]
    assert all(row["llmbench_context_policy"] == "reject-overflow-no-shift" for row in saved)
    assert [row["choices"][0]["message"]["content"] for row in saved] == ["No tool is needed."] * len(fixture_ids)


# ---------------------------------------------------------------------------------------------------
# Public benchmark dispatch (the `benchmarks` stage)
# ---------------------------------------------------------------------------------------------------

RULER_TASKS = ["ruler/niah_multikey_3/4096", "ruler/vt/4096"]
BFCL_TASKS = ["bfcl/native/simple_0", "bfcl/native/simple_1"]
EVALPLUS_TASKS = ["evalplus/mbpp-plus/Mbpp/2", "evalplus/mbpp-plus/Mbpp/3"]
NAMESPACED = {"ruler": ("ruler-vendored-v1", "retrieval", RULER_TASKS),
              "bfcl": ("bfcl-v4-ast-subset-v1", "tools", BFCL_TASKS),
              "evalplus": ("evalplus/mbpp-plus@v0.2.0", "coding", EVALPLUS_TASKS)}


def fake_row(task_id, context, benchmark_id="ruler", **extra):
    revision, category, _ = NAMESPACED[benchmark_id]
    row = {"task_id": task_id, "suite": benchmark_id, "suite_revision": revision, "category": category,
           "split": context.split, "status": "completed", "score": 1.0, "passed": True,
           "model_evaluated": True, "synthetic": False}
    row.update(extra)
    return row


class FakeBenchmark:
    """A ``BenchmarkAdapter`` stand-in registered through ``container_eval.BENCHMARK_ADAPTERS``.

    ``answers`` is consumed one entry per ``available()`` call, so a test can let the pre-load
    registry preflight succeed and the dispatch stage's own preflight fail.
    """

    def __init__(self, benchmark_id="ruler", *, answers=None, run=None, rows=None):
        self.benchmark_id = benchmark_id
        self.revision, self.category, self.declared = NAMESPACED[benchmark_id]
        self.answers, self.run_hook, self.rows = list(answers or []), run, rows
        self.contexts, self.run_contexts = [], []

    def available(self, context):
        self.contexts.append(context)
        return self.answers.pop(0) if self.answers else (True, f"{self.benchmark_id} fixture is present")

    def task_ids(self, context):
        return tuple(self.declared)

    def run(self, context):
        self.run_contexts.append(context)
        if self.run_hook is not None:
            return self.run_hook(context)
        if self.rows is not None:
            return list(self.rows)
        return [fake_row(task_id, context, self.benchmark_id) for task_id in context.task_ids]


def register(monkeypatch, *adapters):
    from llmbench import container_eval
    for adapter in adapters:
        monkeypatch.setitem(container_eval.BENCHMARK_ADAPTERS, adapter.benchmark_id, lambda a=adapter: a)


def public_config(*benchmark_ids, tools=True, **changes):
    selections = []
    if tools:
        selections.append(json.loads(make_config().benchmarks[0].model_dump_json()))
    for benchmark_id in benchmark_ids:
        revision, _, task_ids = NAMESPACED[benchmark_id]
        selections.append({"benchmark_id": benchmark_id, "revision": revision, "task_ids": list(task_ids)})
    return make_config(benchmarks=selections, **changes)


def by_id(result):
    return {sample["task_id"]: sample for sample in result["samples"]}


def test_benchmarks_stage_runs_last_only_for_a_namespaced_selection(tmp_path, monkeypatch):
    from llmbench.benchmarks import REQUIRED_SAMPLE_KEYS
    counted = []

    def run(context):  # the counting route is live while the adapter runs, not after
        counted.append(context.options["token_counter"]([{"role": "user", "content": "hi"}], []))
        return [fake_row(task_id, context) for task_id in context.task_ids]

    adapter = FakeBenchmark(run=run)
    register(monkeypatch, adapter)
    config = public_config("ruler")
    result, _, _ = evaluate(tmp_path / "with", config, dataset_root=str(tmp_path / "datasets"))
    assert [stage["name"] for stage in result["stages"]] == ["attach", "count-route", "overflow-probe",
                                                             "speed", "quality", "benchmarks"]
    assert all(stage["status"] == "ok" for stage in result["stages"])
    assert result["abort_reason"] is None and result["errors"] == []
    samples = by_id(result)
    assert set(samples) == set(RULER_TASKS) | set(config.benchmarks[0].task_ids)
    for task_id in RULER_TASKS:
        assert set(REQUIRED_SAMPLE_KEYS) <= set(samples[task_id])
        assert samples[task_id]["status"] == "completed" and samples[task_id]["score"] == 1.0
    assert result["benchmarks"] == [{"benchmark_id": "ruler", "revision": "ruler-vendored-v1",
                                     "split": "development", "declared": 2, "produced": 2, "status": "ok",
                                     "reason": None, "problems": [], "backfilled": 0,
                                     "execution_client": None}]

    # The context carries the real endpoint, the served alias, the selection and the live budget.
    assert len(adapter.contexts) == 2 and len(adapter.run_contexts) == 1  # preflight, then the stage
    context = adapter.run_contexts[0]
    assert context.base_url == "http://127.0.0.1:18080" and context.model_alias == config.alias()
    assert context.task_ids == tuple(RULER_TASKS) and context.split == "development" and context.seed == 42
    assert context.generation is config.generation and context.session_lock is ALLOWED
    assert context.dataset_root == str(tmp_path / "datasets") and context.remaining_seconds() > 0
    assert callable(context.artifacts.write) and callable(context.artifacts.trace)
    # RULER's own option names, plus the evaluator-supplied counting route. No execution client.
    assert callable(context.options["completion"]) and callable(context.options["token_counter"])
    assert context.options["context_capacity"] == config.engine.ctx_size
    assert context.options["tokenizer_id"] == "gguf-sha256:" + config.assets[0].sha256
    assert context.options["template_id"].startswith("sha256:")
    assert context.options["tokenizer_verified"] is True and "execution_client" not in context.options
    assert counted[0]["tokens"] > 0  # the serving-template counter, reached through the live backend

    result, _, _ = evaluate(tmp_path / "without", make_config())
    assert "benchmarks" not in [stage["name"] for stage in result["stages"]]
    assert result["benchmarks"] == [] and len(result["samples"]) == 3


def test_unavailable_benchmark_becomes_one_environment_error_row_per_declared_task(tmp_path, monkeypatch):
    adapter = FakeBenchmark(answers=[(True, "baked"), (False, "RULER corpus missing - haystack absent")])
    register(monkeypatch, adapter)
    result, _, _ = evaluate(tmp_path, public_config("ruler"))
    assert adapter.run_contexts == []  # unavailable means not run, never means skipped
    samples = by_id(result)
    assert [samples[task_id]["status"] for task_id in RULER_TASKS] == ["environment_error"] * 2
    assert all("haystack absent" in samples[task_id]["error"] for task_id in RULER_TASKS)
    assert all(samples[task_id]["score"] == 0.0 and samples[task_id]["suite"] == "ruler"
               and samples[task_id]["model_evaluated"] is False for task_id in RULER_TASKS)
    assert len(result["samples"]) == 4 and result["abort_reason"] is None
    assert result["benchmarks"][0]["status"] == "environment_error"
    assert result["benchmarks"][0]["backfilled"] == 2 and result["benchmarks"][0]["produced"] == 0
    assert [stage["name"] for stage in result["stages"]][-1] == "benchmarks"


def test_raising_adapter_is_backfilled_and_does_not_kill_the_run(tmp_path, monkeypatch):
    def explode(context):
        raise RuntimeError("the pinned bundle is corrupt")

    register(monkeypatch, FakeBenchmark(run=explode))
    result, _, _ = evaluate(tmp_path, public_config("ruler"))
    samples = by_id(result)
    assert len(result["samples"]) == 4 and result["abort_reason"] is None
    assert [samples[task_id]["status"] for task_id in RULER_TASKS] == ["environment_error"] * 2
    assert all("RuntimeError: the pinned bundle is corrupt" in samples[task_id]["error"]
               for task_id in RULER_TASKS)
    assert samples["tools/no-call-v1"]["status"] == "completed"  # the rest of the run is untouched
    assert [error["stage"] for error in result["errors"]] == ["benchmarks:ruler"]
    assert result["errors"][0]["error_type"] == "RuntimeError"
    assert result["stages"][-1] == {"name": "benchmarks", "status": "ok",
                                    "elapsed_seconds": result["stages"][-1]["elapsed_seconds"]}


def test_benchmark_abort_sets_abort_reason_and_keeps_the_rows_already_produced(tmp_path, monkeypatch):
    from llmbench.benchmarks import BenchmarkAborted

    def abort(context):
        exc = BenchmarkAborted("BFCL response evidence could not be persisted")
        exc.rows = [fake_row(BFCL_TASKS[0], context, "bfcl", status="completed")]
        raise exc

    ruler, bfcl = FakeBenchmark(), FakeBenchmark("bfcl", run=abort)
    register(monkeypatch, ruler, bfcl)
    result, _, _ = evaluate(tmp_path, public_config("ruler", "bfcl"))
    assert result["abort_reason"] == "bfcl aborted: BFCL response evidence could not be persisted"
    samples = by_id(result)
    assert len(result["samples"]) == 6  # nothing dropped: 2 tool probes + 2 ruler + 2 bfcl
    assert all(samples[task_id]["status"] == "completed" for task_id in RULER_TASKS)  # kept
    assert samples[BFCL_TASKS[0]]["status"] == "completed"  # the row the abort carried, kept
    assert samples[BFCL_TASKS[1]]["status"] == "environment_error"
    assert "could not be persisted" in samples[BFCL_TASKS[1]]["error"]
    assert result["stages"][-1] == {"name": "benchmarks", "status": "failed",
                                    "elapsed_seconds": result["stages"][-1]["elapsed_seconds"]}
    assert result["errors"][-1]["stage"] == "benchmarks"
    assert result["errors"][-1]["error_type"] == "BenchmarkAborted"
    assert [row["status"] for row in result["benchmarks"]] == ["ok", "aborted"]


def test_a_row_for_an_undeclared_task_is_rejected_and_the_declared_one_is_backfilled(tmp_path, monkeypatch):
    def sneaky(context):
        return [fake_row(RULER_TASKS[0], context),
                fake_row("ruler/vt/131072", context),                     # never selected
                {k: v for k, v in fake_row(RULER_TASKS[1], context).items() if k != "score"}]

    register(monkeypatch, FakeBenchmark(run=sneaky))
    result, _, _ = evaluate(tmp_path, public_config("ruler"))
    samples = by_id(result)
    assert "ruler/vt/131072" not in samples and len(result["samples"]) == 4
    assert samples[RULER_TASKS[0]]["status"] == "completed"
    assert samples[RULER_TASKS[1]]["status"] == "environment_error"  # dropped for a missing key
    record = result["benchmarks"][0]
    assert record["status"] == "invalid_rows" and record["produced"] == 1 and record["backfilled"] == 1
    assert any("undeclared task 'ruler/vt/131072'" in item for item in record["problems"])
    assert any("lacks required key(s): score" in item for item in record["problems"])


def test_benchmark_stage_publishes_into_its_own_open_transport_sink(tmp_path, monkeypatch):
    """LIVE-007, in the new stage: the adapter's requests must not reach an already-closed sink."""
    import asyncio
    marker = "BENCHMARK-PROMPT-MARKER"

    def talk(context):
        rows = []
        for task_id in context.task_ids:
            request = {"messages": [{"role": "user", "content": f"{marker} {task_id}"}], "max_tokens": 32,
                       "temperature": 0, "seed": 42, "top_p": 1.0}
            response = asyncio.run(asyncio.wait_for(context.options["completion"](request),
                                                    timeout=context.remaining_seconds()))
            assert response["choices"][0]["message"]["content"] == "No tool is needed."
            rows.append(fake_row(task_id, context))
        return rows

    register(monkeypatch, FakeBenchmark(run=talk))
    result, _, _ = evaluate(tmp_path, public_config("ruler"))
    assert result["abort_reason"] is None and result["errors"] == []
    assert all(by_id(result)[task_id]["status"] == "completed" for task_id in RULER_TASKS)

    root = tmp_path / "evaluator"
    raw = (root / "benchmarks/ruler/transport.jsonl").read_text(encoding="utf-8")
    trace = [json.loads(line) for line in raw.splitlines()]
    requests = [row for row in trace if row["event"] == "capture.request"]
    assert len(requests) == 2 and all(marker in json.dumps(row["data"]) for row in requests)
    assert sum(1 for row in trace if row["event"] == "capture.completed") == 2
    # Separate, independently identifiable evidence: no benchmark request lands in another stage's trace.
    assert marker not in (root / "quality/transport.jsonl").read_text(encoding="utf-8")
    saved = [json.loads((root / f"benchmarks/ruler/request-{index}.json").read_text(encoding="utf-8"))
             for index in range(2)]
    assert all(row["llmbench_context_policy"] == "reject-overflow-no-shift" for row in saved)


def test_bfcl_style_byte_transport_returns_the_served_alias_from_a_real_stream(tmp_path, monkeypatch):
    """BFCL parses bytes and checks the served model itself; both must survive the evaluator's bridge."""
    bodies = []

    def post(context):
        for task_id in context.task_ids:
            payload = {"messages": [{"role": "user", "content": task_id}], "max_tokens": 16,
                       "temperature": 0, "seed": 42, "top_p": 1.0, "stream": False}
            bodies.append(context.options["transport"].post_chat(payload, timeout=30))
        return [fake_row(task_id, context, "bfcl") for task_id in context.task_ids]

    register(monkeypatch, FakeBenchmark("bfcl", run=post))
    config = public_config("bfcl")
    result, _, _ = evaluate(tmp_path, config)
    assert result["abort_reason"] is None and result["errors"] == []
    assert len(bodies) == 2 and all(type(body) is bytes for body in bodies)
    parsed = [json.loads(body) for body in bodies]
    # The alias comes out of the chunks the pinned b11011 capture really carries, not from the request.
    assert all(row["model"] == config.alias() for row in parsed)
    assert all(row["choices"][0]["message"]["content"] == "No tool is needed." for row in parsed)
    trace = (tmp_path / "evaluator" / "benchmarks/bfcl/transport.jsonl").read_text(encoding="utf-8")
    assert trace.count('"capture.request"') == 2 and trace.count('"capture.completed"') == 2


def test_only_coding_capable_adapters_receive_the_isolated_execution_client(tmp_path, monkeypatch):
    coding = FakeBenchmark("evalplus")
    register(monkeypatch, coding, FakeBenchmark())
    config = public_config("evalplus", "ruler", broker={})
    result, _, _ = evaluate(tmp_path / "client", config, coding_client="direct-client")
    assert result["abort_reason"] is None and len(result["samples"]) == 6
    assert coding.run_contexts[0].options["execution_client"] == "direct-client"
    ruler_record = [record for record in result["benchmarks"] if record["benchmark_id"] == "ruler"][0]
    assert ruler_record["execution_client"] is None  # not a coding-capable adapter at all
    assert [record["execution_client"] for record in result["benchmarks"]
            if record["benchmark_id"] == "evalplus"] == [True]

    # No broker client anywhere: the coding-capable adapter is handed None, never a host fallback.
    absent = FakeBenchmark("evalplus")
    register(monkeypatch, absent)
    result, _, _ = evaluate(tmp_path / "no-client", public_config("evalplus", tools=False, broker={}))
    assert absent.run_contexts[0].options["execution_client"] is None
    assert [record["execution_client"] for record in result["benchmarks"]] == [False]


def test_budget_exhaustion_marks_the_remaining_benchmark_tasks_instead_of_dropping_them(tmp_path,
                                                                                        monkeypatch):
    now = [100.0]

    def burn(context):
        row = fake_row(context.task_ids[0], context)
        now[0] = 10_000.0  # the wall budget runs out while the first benchmark is still running
        return [row]

    ruler, bfcl = FakeBenchmark(run=burn), FakeBenchmark("bfcl")
    register(monkeypatch, ruler, bfcl)
    result, _, _ = evaluate(tmp_path, public_config("ruler", "bfcl"), clock=lambda: now[0],
                            deadline_monotonic=700.0)
    assert bfcl.run_contexts == []  # never started, and never silently dropped either
    samples = by_id(result)
    assert len(result["samples"]) == 6
    assert samples[RULER_TASKS[0]]["status"] == "completed"
    assert [samples[task_id]["status"] for task_id in RULER_TASKS[1:] + BFCL_TASKS] == ["timeout"] * 3
    assert all(samples[task_id]["score"] == 0.0 and samples[task_id]["passed"] is False
               for task_id in RULER_TASKS[1:] + BFCL_TASKS)
    assert [record["status"] for record in result["benchmarks"]] == ["timeout", "timeout"]
    assert "wall budget ended before bfcl ran" in result["benchmarks"][1]["reason"]


def test_public_benchmark_tasks_stay_in_the_denominator_when_the_stage_never_runs(tmp_path, monkeypatch):
    register(monkeypatch, FakeBenchmark())
    local = [json.loads(item.model_dump_json()) for item in make_config().benchmarks]
    config = make_config(benchmarks=local + [{"benchmark_id": "ruler", "revision": "ruler-vendored-v1",
                                              "task_ids": list(RULER_TASKS)}])
    server = ScriptedServer(config)
    server.apply_template_response = {"unexpected": "shape"}  # count-route and quality both fail
    result, _, _ = evaluate(tmp_path, config, server=server)
    stages = {stage["name"]: stage["status"] for stage in result["stages"]}
    assert stages["quality"] == "failed" and "benchmarks" not in stages
    samples = by_id(result)
    assert len(result["samples"]) == 5 and set(RULER_TASKS) <= set(samples)
    assert all(sample["status"] == "environment_error" and sample["score"] == 0.0
               for sample in result["samples"])
    assert all(samples[task_id]["suite"] == "ruler" for task_id in RULER_TASKS)


def test_option_wiring_translates_or_refuses_per_adapter():
    from llmbench.container_eval import BENCHMARK_WIRING, _benchmark_options, _BytesChatTransport

    supplied = {"token_counter": len, "context_capacity": 8192, "tokenizer_id": "t", "template_id": "x",
                "tokenizer_verified": True}
    common = {"route": lambda request: request, "client": "worker", "supplied": supplied}
    options, problem = _benchmark_options("ruler", {"tasks": ("vt",), "lengths": (4096,)}, **common)
    assert problem is None and options["ruler_tasks"] == ("vt",) and options["ruler_lengths"] == (4096,)
    assert "execution_client" not in options and callable(options["completion"])
    assert options["context_capacity"] == 8192 and options["tokenizer_verified"] is True
    # Aider reads development_fraction, never the registry's split_fraction spelling.
    options, problem = _benchmark_options("aider-polyglot", {"split_fraction": 0.5, "languages": "python"},
                                          **common)
    assert problem is None and options["development_fraction"] == 0.5 and options["languages"] == "python"
    assert options["execution_client"] == "worker" and callable(options["transport"])
    # BFCL wants raw bytes back, so it gets the byte transport, not the callback itself.
    options, problem = _benchmark_options("bfcl", {"modes": ("native",)}, **common)
    assert problem is None and isinstance(options["transport"], _BytesChatTransport)
    assert options["modes"] == ("native",) and "execution_client" not in options
    # An option the registry declares but no adapter reads is a wiring gap, reported, never forwarded.
    for benchmark_id, gap in (("ruler", "samples_per_task"), ("aider-polyglot", "item_limit")):
        assert gap not in BENCHMARK_WIRING[benchmark_id].options
        options, problem = _benchmark_options(benchmark_id, {gap: 3}, **common)
        assert options == {} and problem == f"the evaluator has no wiring for selected option(s): {gap}"
    assert _benchmark_options("not-wired", {}, **common) == ({}, "no dispatch wiring is declared for not-wired")


def test_bytes_transport_carries_only_the_model_the_stream_reported():
    from llmbench.container_eval import _BytesChatTransport

    streamed = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "llmbench_raw_stream": [{"event": "message", "data": {"model": "served-alias"}}]}
    body = _BytesChatTransport(lambda request: streamed).post_chat({"messages": []}, timeout=5)
    assert json.loads(body)["model"] == "served-alias" and type(body) is bytes
    disagreeing = {**streamed, "llmbench_raw_stream": [{"event": "m", "data": {"model": "a"}},
                                                       {"event": "m", "data": {"model": "b"}}]}
    assert "model" not in json.loads(_BytesChatTransport(lambda r: disagreeing).post_chat({}, timeout=5))
    silent = {"choices": [], "llmbench_raw_stream": []}
    assert "model" not in json.loads(_BytesChatTransport(lambda r: silent).post_chat({}, timeout=5))
    explicit = {"model": "already-there", "llmbench_raw_stream": [{"data": {"model": "other"}}]}
    assert json.loads(_BytesChatTransport(lambda r: explicit).post_chat({}, timeout=5))["model"] == \
        "already-there"


def test_self_check_reports_image_provenance_when_present(tmp_path):
    report = self_check(provenance_dir=tmp_path / "absent")
    assert report["ok"] is True and report["python_version"] is None and report["pip_freeze_sha256"] is None
    assert report["coding"]["fixtures_present"] is True and set(report["coding"]["fixtures"]) == set(report["coding"]["task_ids"])
    assert all(len(value) == 64 for value in report["coding"]["fixtures"].values())
    assert report["runtime_python"] == sys.version.split()[0]
    provenance = tmp_path / "opt"
    provenance.mkdir()
    (provenance / "python-version.txt").write_text("Python 3.12.14\n", encoding="utf-8")
    (provenance / "pip-freeze.txt").write_bytes(b"pydantic==2.13.5\n")
    report = self_check(provenance_dir=provenance)
    assert report["python_version"] == "Python 3.12.14"
    assert report["pip_freeze_sha256"] == __import__("hashlib").sha256(b"pydantic==2.13.5\n").hexdigest()
    assert report["ok"] is True and report["provenance_dir"] == str(provenance)
