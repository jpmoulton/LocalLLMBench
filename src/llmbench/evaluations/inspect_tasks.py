"""Inspect task factories, strict raw-response scoring, and CPU-only mock runs.

Inspect imports are lazy so normal configuration/report commands do not require
the evaluation extra. Native tool tests require an adapter to retain the actual
API message in ModelOutput.metadata['llmbench_raw_message']. Re-serializing an
already parsed argument dictionary would hide malformed JSON and is prohibited.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import GenerationSettings, RunMode, TaskSelection, canonical_json
from ..safety import SessionLock
from .retrieval import NiahCase, build_niah_case, score_niah
from .tools import (
    ToolEpisode,
    ToolExpectation,
    failed_tool_score,
    score_tool_response,
    strict_argument_fixture,
    tool_message,
)

_inspect_run_lock = threading.RLock()


@contextmanager
def inspect_runtime_directory(directory: str | Path) -> Any:
    """Keep Inspect's auxiliary caches/traces beside the requested evaluation logs.

    Inspect 0.3 currently exposes no public auxiliary-data-directory option. This
    narrow compatibility adapter replaces its platform path resolvers only while
    our serial evaluation runs, then restores them. It does not change user env
    variables or suppress permission errors. Upgrade tests cover this boundary.
    """
    from unittest.mock import patch

    root = Path(directory).resolve() / ".inspect-runtime"
    root.mkdir(parents=True, exist_ok=True)
    with _inspect_run_lock:
        with patch("inspect_ai._util.appdirs.user_data_path", lambda package: root / "data" / package):
            with patch("inspect_ai._util.appdirs.user_cache_path", lambda package: root / "cache" / package):
                yield


def _imports() -> dict[str, Any]:
    try:
        from inspect_ai import Task
        from inspect_ai.dataset import Sample
        from inspect_ai.model import ChatMessageSystem, ChatMessageTool, ChatMessageUser
        from inspect_ai.model import GenerateConfig, get_model
        from inspect_ai.scorer import Score, mean, scorer
        from inspect_ai.solver import solver
        from inspect_ai.tool import ToolInfo, ToolParams
    except ImportError as exc:
        raise RuntimeError("Install the evaluation extra: pip install -e '.[eval]'") from exc
    return {"Task": Task, "Sample": Sample, "ChatMessageSystem": ChatMessageSystem,
            "ChatMessageTool": ChatMessageTool, "ChatMessageUser": ChatMessageUser,
            "GenerateConfig": GenerateConfig, "get_model": get_model, "Score": Score,
            "mean": mean, "scorer": scorer, "solver": solver, "ToolInfo": ToolInfo, "ToolParams": ToolParams}


def evaluation_config(
    max_output_tokens: int = 256, *, generation: GenerationSettings | None = None, timeout_seconds: int = 30,
) -> Any:
    """Deliberately serial, uncached and without automatic transport retries."""
    GenerateConfig = _imports()["GenerateConfig"]
    settings = generation or GenerationSettings(max_output_tokens=max_output_tokens)
    if settings.tool_emulation:
        raise ValueError("native tool evaluation does not support tool emulation")
    if settings.reasoning not in {"model-default", "none", "minimal", "low", "medium", "high"}:
        raise ValueError("reasoning policy must be model-default or an explicit supported reasoning effort")
    return GenerateConfig(
        max_retries=0, timeout=timeout_seconds, attempt_timeout=timeout_seconds, max_connections=1,
        max_tokens=settings.max_output_tokens, temperature=settings.temperature,
        seed=settings.seed, top_p=settings.top_p, top_k=settings.top_k,
        reasoning_effort=None if settings.reasoning == "model-default" else settings.reasoning, cache=False,
        cache_prompt=False, parallel_tool_calls=False,
    )


def _guard(model: Any, session_lock: SessionLock | None, mode: RunMode | str) -> bool:
    # This exact installed provider class has no network implementation. A model
    # whose display name happens to contain 'mock' does not bypass the lock.
    from inspect_ai.model._providers.mockllm import MockLLM

    synthetic = type(model.api) is MockLLM
    if getattr(model.api, "abort_reason", None):
        from .capture import UnsafeCaptureRuntimeState
        raise UnsafeCaptureRuntimeState(model.api.abort_reason)
    if not synthetic:
        (session_lock or SessionLock()).check("inference", mode)
    return synthetic


def _tool_infos(definitions: Iterable[Any]) -> list[Any]:
    api = _imports()
    return [api["ToolInfo"](name=item.name, description=item.description,
                             parameters=api["ToolParams"].model_validate(item.parameters))
            for item in definitions]


def _output_raw(output: Any) -> Any:
    return (output.metadata or {}).get("llmbench_raw_message")


def _result_scorer() -> Any:
    api = _imports()

    @api["scorer"](metrics=[api["mean"]()])
    def llmbench_result() -> Any:
        async def score(state: Any, target: Any) -> Any:
            result = state.metadata.get("llmbench_result")
            if result is None:
                result = {"score": 0.0, "passed": False, "status": "environment_error",
                          "reason": "sample ended without a recorded result"}
            return api["Score"](value=result["score"], answer=str(result["status"]),
                                 explanation=json.dumps(result, ensure_ascii=False), metadata=result)
        return score

    return llmbench_result()


def _task_settings(name: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "name": name, "version": 1, "config": evaluation_config(), "message_limit": 32,
        "time_limit": 120, "fail_on_error": False, "score_on_error": True,
        "metadata": {"suite_provenance": "local-regression-fixtures", "raw_arguments_required": True,
                     "model_operations_default": "forbidden", "response_cache": False},
        **kwargs,
    }


def tool_probe_task(
    cases: Sequence[tuple[str, ToolExpectation]] | None = None,
    *,
    session_lock: SessionLock | None = None,
    mode: RunMode | str = RunMode.OFFLINE,
    generation: GenerationSettings | None = None,
    timeout_seconds: int = 30,
) -> Any:
    """Create an Inspect single-response native tool task without running it."""
    api = _imports()
    config = evaluation_config(generation=generation, timeout_seconds=timeout_seconds)
    if cases is None:
        nested = strict_argument_fixture()
        no_call = ToolExpectation("tools/no-call-v1", nested.tools, expected_text="No tool is needed.")
        cases = [
            ("Call configure_job for src/CacheConfig.ts with retries 3, enabled true, "
             "and ordered tags [cpu, gpu].", nested),
            ("Do not change anything or call a tool. Reply exactly: No tool is needed.", no_call),
        ]
    lookup = {expectation.task_id: expectation for _, expectation in cases}
    if not cases or len(lookup) != len(cases):
        raise ValueError("provide nonempty uniquely identified tool cases")
    for _, expectation in cases:
        expectation.validate_integrity()
    samples = [api["Sample"](id=expectation.task_id, input=prompt,
                              metadata={"fixture_id": expectation.task_id})
               for prompt, expectation in cases]

    @api["solver"]
    def llmbench_native_probe() -> Any:
        async def solve(state: Any, generate: Any) -> Any:
            model = api["get_model"]()
            synthetic = _guard(model, session_lock, mode)
            expectation = lookup[state.metadata["fixture_id"]]
            expectation.validate_integrity()
            try:
                output = await model.generate(state.messages, tools=_tool_infos(expectation.tools),
                                              tool_choice="auto", config=config, cache=False)
                state.output = output
                state.messages.append(output.message)
                raw = _output_raw(output)
                if raw is None:
                    result = failed_tool_score(expectation.task_id, "environment_error",
                                               "raw_response_unavailable: parsed arguments cannot prove raw validity").to_dict()
                else:
                    result = score_tool_response(raw, expectation).to_dict()
            except TimeoutError as exc:
                result = failed_tool_score(expectation.task_id, "timeout", str(exc)).to_dict()
            except Exception as exc:
                if getattr(exc, "abort_campaign", False):
                    raise
                result = failed_tool_score(expectation.task_id, "transport_error", f"{type(exc).__name__}: {exc}").to_dict()
            result.update({"synthetic": synthetic, "model_evaluated": not synthetic})
            state.metadata["llmbench_result"] = result
            state.completed = True
            return state
        return solve

    return api["Task"](dataset=samples, solver=llmbench_native_probe(), scorer=_result_scorer(),
                        **_task_settings("llmbench_native_tools", config=config, time_limit=timeout_seconds))


def retrieval_task(
    cases: Sequence[NiahCase] | None = None,
    *,
    session_lock: SessionLock | None = None,
    mode: RunMode | str = RunMode.OFFLINE,
    generation: GenerationSettings | None = None,
    timeout_seconds: int = 30,
) -> Any:
    """Create NIAH text/tool cases, preserving exact prompts and counting evidence."""
    api = _imports()
    cases = list(cases) if cases is not None else [build_niah_case()]
    lookup = {case.task_id: case for case in cases}
    if not cases or len(lookup) != len(cases):
        raise ValueError("provide nonempty uniquely identified retrieval cases")
    for case in cases:
        case.validate_integrity()
    if generation and any(generation.max_output_tokens > case.accounting.reserved_output_tokens for case in cases):
        raise ValueError("generation output budget exceeds the retrieval fixture output reserve")
    message_types = {"system": api["ChatMessageSystem"], "user": api["ChatMessageUser"]}
    samples = [api["Sample"](
        id=case.task_id,
        input=[message_types[item["role"]](content=item["content"]) for item in case.messages],
        metadata={"fixture_id": case.task_id, "context": case.accounting.verify_runtime(None)},
    ) for case in cases]

    @api["solver"]
    def llmbench_retrieval_probe() -> Any:
        async def solve(state: Any, generate: Any) -> Any:
            model = api["get_model"]()
            synthetic = _guard(model, session_lock, mode)
            case = lookup[state.metadata["fixture_id"]]
            case.validate_integrity()
            try:
                definitions = case.tool_expectation.tools if case.tool_expectation else ()
                output = await model.generate(state.messages, tools=_tool_infos(definitions), tool_choice="auto",
                                              config=evaluation_config(case.accounting.reserved_output_tokens,
                                                                       generation=generation,
                                                                       timeout_seconds=timeout_seconds), cache=False)
                state.output = output
                state.messages.append(output.message)
                response = _output_raw(output) if case.mode == "tool" else output.completion
                # Mock provider usage is simulated, not evidence about a serving tokenizer.
                observed = output.usage.input_tokens if output.usage and not synthetic else None
                if case.mode == "tool" and response is None:
                    result = score_niah(case, None, failure_status="environment_error")
                    result["parse_error"] = "raw_response_unavailable"
                else:
                    result = score_niah(
                        case, response, observed_input_tokens=observed,
                        truncation_reported=(output.metadata or {}).get("input_truncated"),
                        context_policy=(output.metadata or {}).get("llmbench_context_policy", "unknown"),
                    )
            except TimeoutError as exc:
                result = score_niah(case, None, failure_status="timeout")
                result["failure_detail"] = str(exc)
            except Exception as exc:
                if getattr(exc, "abort_campaign", False):
                    raise
                result = score_niah(case, None, failure_status="transport_error")
                result["failure_detail"] = f"{type(exc).__name__}: {exc}"
            result.update({"synthetic": synthetic, "model_evaluated": not synthetic})
            state.metadata["llmbench_result"] = result
            state.completed = True
            return state
        return solve

    return api["Task"](dataset=samples, solver=llmbench_retrieval_probe(), scorer=_result_scorer(),
                        **_task_settings("llmbench_niah", config=evaluation_config(generation=generation,
                                                                                 timeout_seconds=timeout_seconds),
                                         time_limit=timeout_seconds))


def stateful_tool_task(
    *,
    max_calls: int = 8,
    repair_budget: int = 1,
    session_lock: SessionLock | None = None,
    mode: RunMode | str = RunMode.OFFLINE,
    generation: GenerationSettings | None = None,
    timeout_seconds: int = 30,
) -> Any:
    """Create an Inspect bounded tool loop using fresh in-memory state per sample."""
    api = _imports()
    config = evaluation_config(generation=generation, timeout_seconds=timeout_seconds)
    prototype = ToolEpisode(max_calls=max_calls, repair_budget=repair_budget)
    sample = api["Sample"](id=prototype.task_id, input=(
        "Read src/config.ts, change retries to 3 while preserving the file's formatting, "
        "then run tests. Use the revision from read_file when applying your patch. "
        "Correct any tool errors within the available call budget."
    ))

    @api["solver"]
    def llmbench_stateful_tools() -> Any:
        async def solve(state: Any, generate: Any) -> Any:
            model = api["get_model"]()
            synthetic = _guard(model, session_lock, mode)
            episode = ToolEpisode(max_calls=max_calls, repair_budget=repair_budget)
            while not episode.terminal_failure and not episode.result()["passed"]:
                try:
                    output = await model.generate(state.messages, tools=_tool_infos(episode.tools), tool_choice="auto",
                                                  config=config, cache=False)
                    state.output = output
                    state.messages.append(output.message)
                    raw = _output_raw(output)
                    if raw is None:
                        episode.terminal_failure = "raw_response_unavailable"
                        break
                    results = episode.consume(raw)
                    for result in results:
                        if "tool_call_id" in result:
                            state.messages.append(api["ChatMessageTool"](
                                content=json.dumps(result), tool_call_id=result["tool_call_id"],
                            ))
                    # No tool call means there is no valid protocol result to pair.
                    # Retrying that malformed conversation would make the agent
                    # integration policy an unrecorded repair, so terminate it.
                    if not any("tool_call_id" in result for result in results):
                        episode.terminal_failure = episode.terminal_failure or "protocol_error"
                except TimeoutError:
                    episode.terminal_failure = "timeout"
                except Exception as exc:
                    if getattr(exc, "abort_campaign", False):
                        raise
                    episode.terminal_failure = "transport_error"
                    episode.trace.append({"error": f"{type(exc).__name__}: {exc}"})
            result = episode.result()
            result.update({"synthetic": synthetic, "model_evaluated": not synthetic})
            state.metadata["llmbench_result"] = result
            state.completed = True
            return state
        return solve

    return api["Task"](dataset=[sample], solver=llmbench_stateful_tools(), scorer=_result_scorer(),
                        **_task_settings("llmbench_stateful_tools", message_limit=max_calls * 3 + 2,
                                         config=config, time_limit=timeout_seconds))


def mock_model_from_messages(messages: Sequence[dict[str, Any]]) -> Any:
    """Build Inspect's built-in MockLLM with preserved raw fixture messages."""
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, ModelUsage, get_model
    from inspect_ai.tool import ToolCall

    from .tools import parse_tool_response

    outputs = []
    for message in messages:
        parsed, _ = parse_tool_response(message)
        # Parsed representation is only for Inspect's transcript. Our scorer uses
        # the unmodified copy retained below, including intentional malformed JSON.
        calls = [ToolCall(id=call.call_id, function=call.name, arguments=call.arguments) for call in parsed]
        outputs.append(ModelOutput(
            model="mockllm/llmbench-offline", choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=message.get("content") or "", tool_calls=calls or None),
                stop_reason="tool_calls" if calls else "stop",
            )], usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            metadata={"llmbench_raw_message": copy.deepcopy(message), "synthetic": True,
                      "usage_source": "synthetic-fixture-not-a-tokenizer"},
        ))
    return get_model("mockllm/llmbench-offline", custom_outputs=outputs,
                     config=evaluation_config(), memoize=False)


def run_inspect_suite(
    model: Any,
    output_dir: str | Path,
    selections: Sequence[TaskSelection],
    session_lock: SessionLock,
    *,
    mode: RunMode | str = RunMode.OFFLINE,
    generation: GenerationSettings | None = None,
    max_output_tokens: int = 256,
    seed: int = 42,
    timeout_seconds: int = 120,
    niah_cases: Sequence[NiahCase] = (),
    artifacts: Any = None,
    on_log_failure: Any = None,
) -> dict[str, Any]:
    """Execute only explicitly selected local tasks and retain every required row.

    Selectors:
      tool-probes / local-tools-v1: tools/nested-exact-v1, tools/no-call-v1
      tool-episodes / local-tools-v1: tools/read-edit-test-v1
      niah / local-niah-v1: exact IDs from supplied ``niah_cases``.

    Static local tool fixtures are development-only. Unsupported suite revisions,
    duplicate IDs, unknown tasks and pretend holdout labels fail before inference.
    Upstream benchmark names are deliberately not aliases for these local tasks.
    """
    from inspect_ai import eval as inspect_eval

    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be a positive integer")
    settings = generation or GenerationSettings(max_output_tokens=max_output_tokens, seed=seed)
    evaluation_config(generation=settings, timeout_seconds=timeout_seconds)
    nested = strict_argument_fixture()
    no_call = ToolExpectation("tools/no-call-v1", nested.tools, expected_text="No tool is needed.")
    probes = {
        nested.task_id: ("Call configure_job for src/CacheConfig.ts with retries 3, enabled true, "
                         "and ordered tags [cpu, gpu].", nested),
        no_call.task_id: ("Do not change anything or call a tool. Reply exactly: No tool is needed.", no_call),
    }
    needle_lookup = {case.task_id: case for case in niah_cases}
    if len(needle_lookup) != len(niah_cases):
        raise ValueError("duplicate provided NIAH case IDs")
    expected_rows: dict[str, dict[str, Any]] = {}
    tasks = []
    common = {"session_lock": session_lock, "mode": mode, "generation": settings,
              "timeout_seconds": timeout_seconds}
    for raw_selection in selections:
        selection = (raw_selection if isinstance(raw_selection, TaskSelection)
                     else TaskSelection.model_validate(raw_selection))
        required_revision = {"tool-probes": "local-tools-v1", "tool-episodes": "local-tools-v1",
                             "niah": "local-niah-v1"}.get(selection.suite)
        if required_revision is None or selection.revision != required_revision:
            raise ValueError(f"unsupported suite/revision: {selection.suite}/{selection.revision}")
        if selection.suite != "niah" and selection.split != "development":
            raise ValueError("static local tool fixtures cannot be labeled unseen holdout")
        if selection.suite == "tool-probes":
            if any(task_id not in probes for task_id in selection.task_ids):
                raise ValueError("unknown tool-probes task ID")
            tasks.append(tool_probe_task([probes[task_id] for task_id in selection.task_ids], **common))
        elif selection.suite == "tool-episodes":
            if selection.task_ids != ("tools/read-edit-test-v1",):
                raise ValueError("unknown tool-episodes task ID")
            tasks.append(stateful_tool_task(repair_budget=settings.repair_attempts, **common))
        else:
            if any(task_id not in needle_lookup for task_id in selection.task_ids):
                raise ValueError("unknown NIAH case ID; supply the exact generated cases")
            selected_cases = [needle_lookup[task_id] for task_id in selection.task_ids]
            if any(case.split != selection.split or case.seed != selection.fixture_seed for case in selected_cases):
                raise ValueError("NIAH split or fixture seed differs from the selection manifest")
            tasks.append(retrieval_task(selected_cases, **common))
        for task_id in selection.task_ids:
            if task_id in expected_rows:
                raise ValueError("duplicate task IDs across suite selections")
            if task_id in probes:
                prompt, expectation = probes[task_id]
                payload = {"prompt": prompt, "tools": [item.openai_schema() for item in expectation.tools],
                           "calls": [{"name": item.name, "arguments": item.arguments} for item in expectation.calls],
                           "expected_text": expectation.expected_text}
            elif task_id in needle_lookup:
                payload = needle_lookup[task_id].to_dict(include_answers=True)
            else:
                episode = ToolEpisode(repair_budget=settings.repair_attempts)
                payload = {"initial": episode.initial_content, "expected": episode.expected_content,
                           "tools": [item.openai_schema() for item in episode.tools],
                           "max_calls": episode.max_calls, "repair_budget": episode.repair_budget}
            expected_rows[task_id] = {
                "task_id": task_id, "category": "retrieval" if selection.suite == "niah" else "tools",
                "suite": selection.suite, "suite_revision": selection.revision, "split": selection.split,
                "fixture_seed": selection.fixture_seed,
                "fixture_hash": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
            }
    if not tasks:
        raise ValueError("select at least one supported evaluation task")
    synthetic = _guard(model, session_lock, mode)
    with inspect_runtime_directory(output_dir):
        if artifacts is not None:
            from .budgeted_logs import bounded_inspect_logs
            if not callable(on_log_failure):
                raise ValueError("metered Inspect logs require a cancellation/admission failure callback")
            logging_context = bounded_inspect_logs(artifacts, on_log_failure)
        else:
            logging_context = nullcontext(None)
        with logging_context as log_session:
            logging_options = ({"log_format": "json", "log_buffer": 1, "log_shared": False}
                               if log_session is not None else {})
            logs = inspect_eval(
                tasks, model=model, log_dir=log_session.url if log_session else str(output_dir),
                display="none", max_samples=1, max_tasks=1, retry_on_error=0,
                fail_on_error=False, score_on_error=True, log_model_api=True,
                ctl_server=False, acp_server=False, **logging_options,
            )
            if log_session is not None:
                log_session.check()
                for log in logs:
                    log.location = log_session.local_path(str(log.location))
    if getattr(model.api, "abort_reason", None):
        from .capture import UnsafeCaptureRuntimeState
        unsafe = UnsafeCaptureRuntimeState(model.api.abort_reason)
        unsafe.logs = [str(log.location) for log in logs]
        raise unsafe
    observed: dict[str, dict[str, Any]] = {}
    for log in logs:
        for sample in log.samples or []:
            result = (sample.metadata or {}).get("llmbench_result")
            if result is not None:
                observed[str(sample.id)] = result
    samples = []
    for task_id, metadata in expected_rows.items():
        result = observed.get(task_id, {"score": 0.0, "passed": False, "status": "environment_error",
                                        "reason": "required sample produced no scored result"})
        original_status = result["status"]
        failures = {"timeout", "transport_error", "environment_error", "unsupported", "cancelled", "invalid_context"}
        normalized_status = ("environment_error" if original_status == "raw_response_unavailable"
                             else original_status if original_status in failures else "completed")
        samples.append({**result, **metadata, "status": normalized_status, "outcome_status": original_status,
                        "raw_response": result.get("raw_response"),
                        "synthetic": synthetic, "model_evaluated": not synthetic})
    return {
        "logs": [str(log.location) for log in logs], "eval_statuses": [log.status for log in logs],
        "samples": samples, "attempted": len(samples), "passed": sum(sample["passed"] for sample in samples),
        "synthetic": synthetic, "model_evaluated": not synthetic,
        "generation": settings.model_dump(mode="json"), "timeout_seconds": timeout_seconds,
    }


def run_inspect_mock_demo(log_dir: str | Path, *, inject_failure: bool = False) -> dict[str, Any]:
    """Run real Inspect scheduling/scoring/logging with its network-free MockLLM.

    This function has no option to select a real provider or inference URL.
    The output is harness acceptance evidence, never model quality evidence.
    """
    from inspect_ai import eval as inspect_eval

    fixture = strict_argument_fixture()
    args = copy.deepcopy(fixture.calls[0].arguments)
    if inject_failure:
        args["options"]["retries"] = "3"
    model = mock_model_from_messages([
        tool_message(fixture.calls[0].name, args),
        {"role": "assistant", "content": "No tool is needed."},
    ])
    with inspect_runtime_directory(log_dir):
        logs = inspect_eval(
            tool_probe_task(), model=model, log_dir=str(log_dir), display="none",
            max_samples=1, max_tasks=1, retry_on_error=0, log_model_api=True,
            fail_on_error=False, score_on_error=True, ctl_server=False, acp_server=False,
        )
    cases = []
    for log in logs:
        for sample in log.samples or []:
            result = (sample.metadata or {}).get("llmbench_result")
            if result is None:
                result = {"task_id": str(sample.id), "suite": "tools", "passed": False,
                          "score": 0.0, "status": "environment_error", "synthetic": True,
                          "model_evaluated": False, "error": str(sample.error)}
            cases.append(result)
    return {"synthetic": True, "model_evaluated": False, "framework": "Inspect AI",
            "cases": cases, "attempted": len(cases), "passed": sum(case["passed"] for case in cases),
            "logs": [str(log.location) for log in logs], "eval_statuses": [log.status for log in logs]}
