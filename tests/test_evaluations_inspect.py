import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage, get_model

from llmbench.config import GenerationSettings, RunMode, TaskSelection
from llmbench.evaluations.capture import (
    backend_completion_callback, collect_chat_stream, create_capture_model,
)
from llmbench.evaluations.inspect_tasks import (
    inspect_runtime_directory, mock_model_from_messages, retrieval_task,
    run_inspect_mock_demo, run_inspect_suite, stateful_tool_task, tool_probe_task,
)
from llmbench.evaluations.retrieval import build_niah_case
from llmbench.evaluations.tools import ToolEpisode, strict_argument_fixture, tool_message
from llmbench.safety import OperationForbidden, SessionLock


def run_task(task, model, directory):
    with inspect_runtime_directory(directory):
        logs = inspect_eval(task, model=model, log_dir=str(directory), display="none", max_samples=1,
                            max_tasks=1, retry_on_error=0, fail_on_error=False, score_on_error=True,
                            ctl_server=False, acp_server=False)
    assert len(logs) == 1 and logs[0].status == "success", str(logs[0].error)
    return [sample.metadata["llmbench_result"] for sample in logs[0].samples]


def test_real_inspect_mock_eval_logs_success_and_failure(tmp_path):
    good = run_inspect_mock_demo(tmp_path / "pass")
    bad = run_inspect_mock_demo(tmp_path / "fail", inject_failure=True)
    assert good["attempted"] == 2 and good["passed"] == 2
    assert bad["attempted"] == 2 and bad["passed"] == 1
    assert good["synthetic"] and not good["model_evaluated"]
    assert all(case["synthetic"] for case in good["cases"])
    assert list((tmp_path / "pass").glob("*.eval"))


def test_mock_eval_does_not_initialize_or_download_a_tokenizer(tmp_path, monkeypatch):
    import tiktoken

    def forbidden(*args, **kwargs):
        raise AssertionError("offline mock must not initialize any tokenizer")

    monkeypatch.setattr(tiktoken, "get_encoding", forbidden)
    result = run_inspect_mock_demo(tmp_path)
    assert result["passed"] == 2


def test_raw_json_survives_inspect_parsed_representation(tmp_path):
    fixture = strict_argument_fixture()
    raw = tool_message("configure_job", '{"path":"x","path":"y"}')
    results = run_task(tool_probe_task([("Call configure_job.", fixture)]),
                       mock_model_from_messages([raw]), tmp_path)
    assert results[0]["status"] == "protocol_error"
    assert results[0]["raw_response"] == raw


def test_parsed_only_provider_output_cannot_claim_raw_protocol_pass(tmp_path):
    output = ModelOutput.from_content("mockllm", "ok")
    output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    model = get_model("mockllm/no-raw", custom_outputs=[output], memoize=False)
    fixture = strict_argument_fixture()
    result = run_task(tool_probe_task([("Call a tool.", fixture)]), model, tmp_path)[0]
    assert result["status"] == "environment_error"
    assert "raw_response_unavailable" in result["errors"][0]


def test_inspect_retrieval_uses_mock_without_context_claim(tmp_path):
    case = build_niah_case(depths=(0.1, 0.9))
    model = mock_model_from_messages([{"role": "assistant", "content": json.dumps(case.expected)}])
    result = run_task(retrieval_task([case]), model, tmp_path)[0]
    assert result["passed"] and result["synthetic"]
    assert not result["eligible_for_context_claim"] and result["context"]["observed_input_tokens"] is None


def test_inspect_stateful_chain_and_recovery(tmp_path):
    episode = ToolEpisode()
    outputs = [
        tool_message("read_file", {"path": episode.path}, "read"),
        tool_message("apply_patch", {"path": episode.path, "expected_revision": 0,
                                     "content": episode.expected_content}, "bad"),
        tool_message("apply_patch", {"path": episode.path, "expected_revision": 1,
                                     "content": episode.expected_content}, "fix"),
        tool_message("run_tests", {"path": episode.path}, "tests"),
    ]
    result = run_task(stateful_tool_task(), mock_model_from_messages(outputs), tmp_path)[0]
    assert result["passed"] and result["success_after_repairs"] and not result["first_attempt_success"]
    assert result["calls"] == 4


def event(data, kind="message"):
    return SimpleNamespace(event=kind, data=data)


def fragment(arguments, *, index=0, first=False, finish=None):
    call = {"index": index, "function": {"arguments": arguments}}
    if first:
        call.update({"id": "native_1", "type": "function"})
        call["function"]["name"] = "configure_job"
    return event({"choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": finish}]})


def test_stream_capture_keeps_malformed_arguments_exactly():
    response = collect_chat_stream([
        fragment('{"path":', first=True), fragment('"a",}'),
        event({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        event({"choices": [], "usage": {"prompt_tokens": 321, "completion_tokens": 17}}),
        event("[DONE]", "done"),
    ])
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"path":"a",}'
    assert response["usage"]["prompt_tokens"] == 321
    assert len(response["llmbench_raw_stream"]) == 5


def test_stream_capture_does_not_invent_missing_call_type():
    response = collect_chat_stream([
        event({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c",
               "function": {"name": "x", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}),
    ])
    assert response["choices"][0]["message"]["tool_calls"][0]["type"] is None


def test_incomplete_cancelled_and_oversized_streams_fail():
    with pytest.raises(RuntimeError, match="incomplete"):
        collect_chat_stream([fragment("{}", first=True)])
    with pytest.raises(RuntimeError, match="cancelled"):
        collect_chat_stream([event({}, "cancelled")])
    with pytest.raises(ValueError, match="byte budget"):
        collect_chat_stream([fragment("{}", first=True)], max_response_bytes=1)
    with pytest.raises(ValueError, match="index"):
        collect_chat_stream([fragment("{}", first=True, index=True)])


def test_capture_model_default_lock_blocks_callback_before_transport():
    called = []

    async def callback(request):
        called.append(request)
        raise AssertionError("must not run")

    model = create_capture_model(callback, model_name="verified-alias", session_lock=SessionLock())
    with pytest.raises(OperationForbidden):
        asyncio.run(model.api.generate([ChatMessageUser(content="hi")], [], "auto", model.config))
    assert not called


def test_capture_model_fake_callback_preserves_native_usage_and_raw_args():
    captured = []
    fixture = strict_argument_fixture()
    raw_message = tool_message(fixture.calls[0].name, fixture.calls[0].arguments)

    async def callback(request):
        captured.append(request)
        return {"choices": [{"message": raw_message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 321, "completion_tokens": 19}}

    # Explicit in-memory lock capability only; this test has no network callback.
    model = create_capture_model(callback, model_name="verified-alias",
                                  session_lock=SessionLock(allow_inference=True), mode=RunMode.LIVE)
    output, call = asyncio.run(model.api.generate([ChatMessageUser(content="hi")], [], "auto", model.config))
    assert len(captured) == 1
    assert output.usage.input_tokens == 321 and output.usage.output_tokens == 19
    assert output.metadata["llmbench_raw_message"] == raw_message
    assert call.response["choices"][0]["message"] == raw_message
    assert output.metadata["input_truncated"] is None
    assert output.metadata["llmbench_context_policy"] == "unknown"


def test_capture_preserves_reasoning_and_raw_arguments_in_followup_history():
    requests = []
    raw = tool_message("x", '{ "identifier" : "A" }')
    raw["reasoning_content"] = "Keep the exact identifier."

    async def callback(request):
        requests.append(request)
        return {"choices": [{"message": raw, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3}}

    model = create_capture_model(callback, model_name="fixture", session_lock=SessionLock(allow_inference=True),
                                  mode=RunMode.LIVE)

    async def two_requests():
        output, _ = await model.api.generate([ChatMessageUser(content="hi")], [], "auto", model.config)
        assert output.completion == ""
        await model.api.generate([output.message], [], "auto", model.config)

    asyncio.run(two_requests())
    assert requests[1]["messages"][0] == raw


def test_capture_provider_runs_through_inspect_with_fake_callback(tmp_path):
    fixture = strict_argument_fixture()
    raw = tool_message(fixture.calls[0].name, fixture.calls[0].arguments)
    lock = SessionLock(allow_inference=True)
    requests = []

    async def callback(request):
        requests.append(request)
        return {"choices": [{"message": raw, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 12}}

    model = create_capture_model(callback, model_name="fixture-alias", session_lock=lock, mode=RunMode.LIVE)
    task = tool_probe_task([("Call configure_job.", fixture)], session_lock=lock, mode=RunMode.LIVE)
    results = run_task(task, model, tmp_path)
    assert results[0]["passed"]
    assert len(requests) == 1 and requests[0]["parallel_tool_calls"] is False
    assert requests[0]["tools"][0]["function"]["name"] == "configure_job"


def test_backend_callback_checks_lock_before_stream_or_cancel():
    class NeverBackend:
        def stream(self, *args, **kwargs):
            raise AssertionError("forbidden transport")

    callback = backend_completion_callback(NeverBackend(), "alias", session_lock=SessionLock())
    with pytest.raises(OperationForbidden):
        asyncio.run(callback({"messages": []}))


def test_selected_suite_applies_generation_and_preserves_task_provenance(tmp_path):
    fixture = strict_argument_fixture()
    requests = []
    settings = GenerationSettings(temperature=0.4, top_p=0.9, top_k=17, seed=59, max_output_tokens=321,
                                  reasoning="low", strict_tools=True)
    lock = SessionLock(allow_inference=True)

    async def callback(request):
        requests.append(request)
        return {"choices": [{"message": tool_message(fixture.calls[0].name, fixture.calls[0].arguments),
                              "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}

    model = create_capture_model(callback, model_name="fixture-only", session_lock=lock,
                                  mode=RunMode.LIVE, generation=settings)
    selection = TaskSelection(suite="tool-probes", revision="local-tools-v1", task_ids=(fixture.task_id,))
    result = run_inspect_suite(model, tmp_path, [selection], lock, mode=RunMode.LIVE, generation=settings)
    assert result["attempted"] == 1 and result["passed"] == 1
    sample = result["samples"][0]
    assert sample["task_id"] == fixture.task_id and sample["category"] == "tools"
    assert sample["status"] == "completed" and sample["outcome_status"] == "passed"
    assert sample["split"] == "development" and len(sample["fixture_hash"]) == 64
    request = requests[0]
    assert request["temperature"] == 0.4 and request["top_p"] == 0.9 and request["top_k"] == 17
    assert request["seed"] == 59 and request["max_tokens"] == 321 and request["reasoning_effort"] == "low"
    assert request["tools"][0]["function"]["strict"] is True


@pytest.mark.parametrize("selection", [
    TaskSelection(suite="bfcl", revision="local-tools-v1", task_ids=("tools/nested-exact-v1",)),
    TaskSelection(suite="tool-probes", revision="untrusted-revision", task_ids=("tools/nested-exact-v1",)),
    TaskSelection(suite="tool-probes", revision="local-tools-v1", task_ids=("invented",)),
    TaskSelection(suite="tool-probes", revision="local-tools-v1", task_ids=("tools/nested-exact-v1",), split="holdout"),
])
def test_unknown_suite_task_revision_and_pretend_holdout_rejected_before_evaluation(tmp_path, selection):
    model = mock_model_from_messages([])
    with pytest.raises(ValueError):
        run_inspect_suite(model, tmp_path, [selection], SessionLock())
    assert not list(tmp_path.glob("*.eval"))


def test_runner_uses_exact_supplied_niah_case(tmp_path):
    case = build_niah_case(seed=123, split="holdout")
    selection = TaskSelection(suite="niah", revision="local-niah-v1", task_ids=(case.task_id,),
                              fixture_seed=123, split="holdout")
    model = mock_model_from_messages([{"content": json.dumps(case.expected)}])
    result = run_inspect_suite(model, tmp_path, [selection], SessionLock(), niah_cases=[case])
    sample = result["samples"][0]
    assert sample["passed"] and sample["split"] == "holdout"
    assert sample["context"]["prompt_sha256"] == case.accounting.prompt_sha256
    assert not sample["eligible_for_context_claim"]


def test_mutated_fixture_is_rejected_at_inspect_task_construction():
    fixture = strict_argument_fixture()
    fixture.calls[0].arguments["path"] = "changed"
    with pytest.raises(ValueError, match="stable ID"):
        tool_probe_task([("unchanged prompt", fixture)])
    case = build_niah_case()
    case.messages[1]["content"] = "changed"
    with pytest.raises(ValueError, match="stable prompt hash"):
        retrieval_task([case])
