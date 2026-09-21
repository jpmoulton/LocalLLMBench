import copy

import pytest

from llmbench.evaluations import run_demo_suite
from llmbench.evaluations.tools import (
    ExpectedCall, ToolEpisode, ToolExpectation, failed_tool_score, score_tool_response,
    strict_argument_fixture, strict_equal, strict_json_loads, summarize_tool_scores, tool_message,
)


def test_exact_nested_fixture_and_raw_preservation():
    fixture = strict_argument_fixture()
    response = tool_message(fixture.calls[0].name, fixture.calls[0].arguments)
    score = score_tool_response(response, fixture)
    assert score.passed and score.score == 1
    response["tool_calls"][0]["function"]["arguments"] = "changed"
    assert score.raw_response["tool_calls"][0]["function"]["arguments"] != "changed"


@pytest.mark.parametrize("wrong", ["3", True, 3.0, None])
def test_json_types_are_never_coerced(wrong):
    fixture = strict_argument_fixture()
    arguments = copy.deepcopy(fixture.calls[0].arguments)
    arguments["options"]["retries"] = wrong
    score = score_tool_response(tool_message("configure_job", arguments), fixture)
    assert not score.passed and not score.schema_valid and not score.arguments_correct
    assert score.status == "schema_error"


@pytest.mark.parametrize("raw", [
    '{"path": "a", "path": "b"}', '{"n": NaN}', '{"n": Infinity}',
    '{"a": 1,}', '```json\n{"a":1}\n```', '[]', {"a": 1},
])
def test_malformed_raw_arguments_are_not_repaired(raw):
    response = tool_message("configure_job", "{}")
    response["tool_calls"][0]["function"]["arguments"] = raw
    score = score_tool_response(response, strict_argument_fixture())
    assert not score.passed and not score.protocol_valid
    assert score.raw_response == response


def test_correct_schema_but_wrong_identifier_is_semantic_failure():
    fixture = strict_argument_fixture()
    arguments = copy.deepcopy(fixture.calls[0].arguments)
    arguments["path"] = "src/cacheconfig.ts"
    score = score_tool_response(tool_message("configure_job", arguments), fixture)
    assert score.schema_valid and score.selection_correct
    assert not score.arguments_correct and not score.passed


def test_invented_nested_field_fails_schema():
    fixture = strict_argument_fixture()
    arguments = copy.deepcopy(fixture.calls[0].arguments)
    arguments["options"]["invented"] = False
    assert not score_tool_response(tool_message("configure_job", arguments), fixture).schema_valid


def test_unknown_tool_and_duplicate_id_fail():
    fixture = strict_argument_fixture()
    assert not score_tool_response(tool_message("invented_tool", fixture.calls[0].arguments), fixture).passed
    response = tool_message("configure_job", fixture.calls[0].arguments)
    response["tool_calls"].append(copy.deepcopy(response["tool_calls"][0]))
    assert not score_tool_response(response, fixture).protocol_valid


def test_no_call_and_missing_required_call():
    fixture = strict_argument_fixture()
    expectation = ToolExpectation("no-call", fixture.tools, expected_text="No tool needed.")
    assert score_tool_response({"content": "No tool needed."}, expectation).passed
    assert not score_tool_response(tool_message("configure_job", fixture.calls[0].arguments), expectation).passed
    assert not score_tool_response({"content": "Done."}, fixture).passed


def test_unordered_parallel_calls_match_complete_pairs():
    fixture = strict_argument_fixture()
    first = fixture.calls[0]
    second_args = copy.deepcopy(first.arguments)
    second_args["path"] = "src/Other.ts"
    second = ExpectedCall(first.name, second_args)
    expectation = ToolExpectation("parallel", fixture.tools, (first, second), ordered=False)
    message = tool_message(second.name, second.arguments, "b")
    message["tool_calls"] += tool_message(first.name, first.arguments, "a")["tool_calls"]
    assert score_tool_response(message, expectation).passed
    ordered = ToolExpectation("ordered", fixture.tools, (first, second), ordered=True)
    assert not score_tool_response(message, ordered).passed


def test_failed_requests_stay_in_required_denominator():
    fixture = strict_argument_fixture()
    passing = score_tool_response(tool_message(fixture.calls[0].name, fixture.calls[0].arguments), fixture)
    summary = summarize_tool_scores([
        passing, failed_tool_score("timeout", "timeout", "late"),
        failed_tool_score("unsupported", "unsupported", "no native tools"),
    ])
    assert summary["success_all_required"] == pytest.approx(1 / 3)
    assert summary["completion_rate"] == pytest.approx(1 / 3)
    assert summary["attempted"] == 3


def patch(episode, *, revision=1, content=None, call_id="patch"):
    return tool_message("apply_patch", {"path": episode.path, "expected_revision": revision,
                                        "content": content if content is not None else episode.expected_content}, call_id)


def test_stateful_clean_chain_and_reset():
    episode = ToolEpisode()
    episode.consume(tool_message("read_file", {"path": episode.path}, "read"))
    episode.consume(patch(episode))
    episode.consume(tool_message("run_tests", {"path": episode.path}, "test"))
    assert episode.result()["first_attempt_success"]
    episode.reset()
    assert episode.content == episode.initial_content
    assert episode.calls == 0 and episode.revision == 1 and not episode.trace
    assert not episode.result()["passed"]


def test_stateful_revision_error_recovery_is_separate():
    episode = ToolEpisode(repair_budget=1)
    episode.consume(tool_message("read_file", {"path": episode.path}, "read"))
    error = episode.consume(patch(episode, revision=0, call_id="stale"))
    assert error[0]["error"] == "revision_conflict"
    episode.consume(patch(episode, call_id="repair"))
    episode.consume(tool_message("run_tests", {"path": episode.path}, "test"))
    result = episode.result()
    assert result["passed"] and result["success_after_repairs"] and not result["first_attempt_success"]
    assert result["errors"] == 1 and result["calls"] == 4


def test_stateful_dependencies_budget_and_known_wrong_patch():
    episode = ToolEpisode(repair_budget=0)
    episode.consume(patch(episode))
    assert episode.result()["status"] == "repair_budget_exhausted"
    assert episode.content == episode.initial_content
    episode = ToolEpisode(max_calls=3)
    episode.consume(tool_message("read_file", {"path": episode.path}, "read"))
    episode.consume(patch(episode, content="export const retries = 30;\n"))
    episode.consume(tool_message("run_tests", {"path": episode.path}, "test"))
    assert not episode.result()["passed"]
    assert episode.result()["status"] == "call_budget_exhausted"


def test_duplicate_episode_ids_and_loop_are_bounded():
    episode = ToolEpisode(max_calls=3)
    episode.consume(tool_message("read_file", {"path": episode.path}, "same"))
    episode.consume(patch(episode, call_id="same"))
    assert episode.errors == 1
    episode.consume(tool_message("read_file", {"path": episode.path}, "last"))
    assert episode.terminal_failure == "call_budget_exhausted"


def test_deep_json_comparison_does_not_confuse_bool_and_int():
    assert not strict_equal({"x": [True]}, {"x": [1]})
    assert not strict_equal({"x": 1.0}, {"x": 1})
    with pytest.raises(ValueError, match="duplicate"):
        strict_json_loads('{"a": {"b": 1, "b": 2}}')


def test_demo_is_synthetic_and_optional_failures_reduce_denominator():
    result = run_demo_suite()
    bad = run_demo_suite(inject_failure=True)
    assert result["synthetic"] and not result["model_evaluated"]
    assert result["summary"]["success_all_required"] == 1
    assert bad["summary"]["attempted"] == result["summary"]["attempted"] + 1
    assert bad["summary"]["success_all_required"] < 1
    assert result["summary"]["context_claims_verified"] == 0


@pytest.mark.parametrize("mutation", ["argument", "schema"])
def test_tool_fixture_mutation_cannot_change_scoring_under_stable_identity(mutation):
    fixture = strict_argument_fixture()
    raw = tool_message(fixture.calls[0].name, fixture.calls[0].arguments)
    if mutation == "argument":
        fixture.calls[0].arguments["options"]["retries"] = 99
    else:
        fixture.tools[0].parameters["additionalProperties"] = True
    with pytest.raises(ValueError, match="stable ID") as caught:
        score_tool_response(raw, fixture)
    assert caught.value.abort_campaign
