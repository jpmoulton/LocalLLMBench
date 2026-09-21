import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from llmbench.evaluations.retrieval import (
    ACCOUNTING_SLACK_TOKENS, build_niah_case, estimated_token_counter, score_niah,
)
from llmbench.evaluations.tools import tool_message


def exact_test_counter(messages, tools):
    # A synthetic tokenizer/template pair for testing the counter boundary, not a
    # claimed real-model tokenizer. Each whitespace word is one fixture token.
    return sum(len(message["content"].split()) + 4 for message in messages) + len(json.dumps(tools).split()) + 2


def test_determinism_split_identity_and_full_budget_accounting():
    arguments = dict(depths=(0.05, 0.5, 0.95), seed=7, target_input_tokens=1024,
                     token_counter=exact_test_counter, tokenizer_verified=True,
                     tokenizer_id="fixture-tokenizer", template_id="fixture-template")
    case = build_niah_case(**arguments)
    repeated = build_niah_case(**arguments)
    holdout = build_niah_case(**arguments, split="holdout")
    assert case == repeated and case.task_id != holdout.task_id
    assert case.accounting.actual_input_tokens == exact_test_counter(list(case.messages), list(case.tools))
    assert case.accounting.actual_input_tokens <= 1024
    assert case.accounting.actual_input_tokens + case.accounting.reserved_output_tokens <= case.accounting.context_capacity
    depths = [needle.actual_depth for needle in case.needles]
    assert depths == sorted(depths) and depths[0] < 0.2 and depths[-1] > 0.8


def test_counting_includes_chat_template_and_tools():
    seen = []

    def counter(messages, tools):
        assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
        assert tools[0]["function"]["name"] == "submit_retrieved_keys"
        seen.append(True)
        return exact_test_counter(messages, tools)

    case = build_niah_case(mode="tool", token_counter=counter, tokenizer_verified=True,
                           tokenizer_id="test", template_id="test")
    assert seen and case.accounting.tool_schema_included


def test_estimates_and_missing_readback_never_verify_runtime():
    case = build_niah_case()
    result = score_niah(case, json.dumps(case.expected), observed_input_tokens=case.accounting.actual_input_tokens)
    assert result["passed"]
    assert not result["eligible_for_context_claim"]
    assert result["context"]["verification_status"] == "estimated_tokenizer"
    case = build_niah_case(token_counter=exact_test_counter, tokenizer_verified=True,
                           tokenizer_id="test", template_id="test")
    result = score_niah(case, json.dumps(case.expected))
    assert result["passed"] and not result["eligible_for_context_claim"]
    assert result["context"]["verification_status"] == "runtime_count_missing"


def test_native_count_mismatch_and_truncation_invalidate_correct_answer():
    case = build_niah_case(token_counter=exact_test_counter, tokenizer_verified=True,
                           tokenizer_id="test", template_id="test")
    correct = json.dumps(case.expected)
    mismatch = score_niah(case, correct, observed_input_tokens=case.accounting.actual_input_tokens - 1)
    assert mismatch["exact_retrieval"] and not mismatch["passed"]
    assert mismatch["status"] == "invalid_context"
    truncated = score_niah(case, correct, observed_input_tokens=case.accounting.actual_input_tokens,
                           truncation_reported=True)
    assert not truncated["passed"]
    verified = score_niah(case, correct, observed_input_tokens=case.accounting.actual_input_tokens,
                          truncation_reported=False, context_policy="reject-overflow-no-shift")
    assert verified["eligible_for_context_claim"]


def test_unknown_truncation_and_context_shift_policy_cannot_certify_context():
    case = build_niah_case(token_counter=exact_test_counter, tokenizer_verified=True,
                           tokenizer_id="test", template_id="test")
    result = score_niah(case, json.dumps(case.expected), observed_input_tokens=case.accounting.actual_input_tokens)
    assert result["context"]["truncation_reported"] is None
    assert result["context"]["verification_status"] == "truncation_evidence_missing"
    assert not result["eligible_for_context_claim"]
    result = score_niah(case, json.dumps(case.expected), observed_input_tokens=case.accounting.actual_input_tokens,
                        truncation_reported=False)
    assert result["context"]["verification_status"] == "context_policy_unverified"
    assert not result["eligible_for_context_claim"]


def test_multi_needle_missing_needle_distractors_and_exact_values():
    case = build_niah_case(depths=(0.1, 0.5, 0.9), missing_indices=(1,), distractors=6)
    missing = case.needles[1]
    assert missing.expected_value is None and missing.actual_depth is None
    assert f"CURRENT {missing.key}:" not in case.messages[1]["content"]
    assert f"OBSOLETE {missing.key}:" in case.messages[1]["content"]
    correct = score_niah(case, json.dumps(case.expected))
    assert correct["passed"] and correct["retrieval_fraction"] == 1
    wrong = dict(case.expected)
    wrong[case.needles[0].key] = "fabricated"
    partial = score_niah(case, json.dumps(wrong))
    assert partial["retrieval_fraction"] == pytest.approx(2 / 3) and not partial["passed"]
    absent_key = dict(case.expected)
    absent_key.pop(missing.key)
    assert not score_niah(case, json.dumps(absent_key))["passed"]


def test_tool_dependent_retrieval_checks_selection_and_args_separately():
    case = build_niah_case(mode="tool", depths=(0.1, 0.9))
    correct = tool_message("submit_retrieved_keys", {"values": case.expected})
    assert score_niah(case, correct)["passed"]
    wrong_tool = tool_message("invented_submit", {"values": case.expected})
    score = score_niah(case, wrong_tool)
    assert score["exact_retrieval"] and not score["passed"]
    assert not score["tool_score"]["selection_correct"]


@pytest.mark.parametrize("response", ["{} extra", "[]", '```json\n{}\n```', '{"x": NaN}', "null"])
def test_retrieval_does_not_extract_or_repair_malformed_answers(response):
    assert not score_niah(build_niah_case(), response)["passed"]


def test_timeout_keeps_zero_scored_case():
    result = score_niah(build_niah_case(), None, failure_status="timeout")
    assert result["status"] == "timeout" and result["score"] == 0 and result["retrieval_fraction"] == 0


def test_invalid_capacity_and_counter_provenance_are_rejected():
    with pytest.raises(ValueError, match="capacity"):
        build_niah_case(target_input_tokens=4096, context_capacity=4096)
    with pytest.raises(ValueError, match="require"):
        build_niah_case(target_input_tokens=10)
    with pytest.raises(ValueError, match="explicit"):
        build_niah_case(tokenizer_verified=True)
    with pytest.raises(ValueError, match="estimator"):
        build_niah_case(token_counter=estimated_token_counter, tokenizer_verified=True)
    with pytest.raises(ValueError, match="counter"):
        build_niah_case(token_counter=lambda messages, tools: True)


def test_safe_metadata_omits_answer_prompt_and_expected_tool_values():
    case = build_niah_case(split="holdout", mode="tool")
    metadata = case.to_dict()
    assert "expected" not in metadata and "messages" not in metadata and "tool_expectation" not in metadata
    assert "expected_value" not in metadata["needles"][0]


@pytest.mark.parametrize("mutation", ["prompt", "schema", "answer", "tool-expectation"])
def test_niah_fixture_drift_rejected_before_scoring_or_export(mutation):
    case = build_niah_case(mode="tool")
    if mutation == "prompt":
        case.messages[1]["content"] += " Changed request."
    elif mutation == "schema":
        case.tools[0]["function"]["description"] = "Changed schema."
    elif mutation == "answer":
        case.expected[case.needles[0].key] = "changed"
    else:
        case.tool_expectation.calls[0].arguments["values"] = {"changed": "answer"}
    with pytest.raises(ValueError) as caught:
        score_niah(case, {})
    assert caught.value.abort_campaign
    with pytest.raises(ValueError):
        case.to_dict()


# LIVE-004 / AM-2: the verbatim raw_response of sample niah/single-middle/development/seed-42 from
# artifacts/container-pilot/suite-q6-8k-1/evaluation.json (scored 0.0, outcome_status=protocol_error).
LIVE_004_RAW_RESPONSE = '```json\n{\n  "key_e2d92bb192": "pass_6fd558f3c212af54f72aa3fa"\n}\n```'
LENIENT_KEYS = ("format_strict_ok", "lenient_parse_used", "lenient_retrieval_fraction", "lenient_per_needle",
                "lenient_outcome_correct")


def _fenced(payload, tag="json"):
    return f"```{tag}\n{payload}\n```"


def test_live_004_fenced_correct_answer_reproduces_strict_failure_and_lenient_success():
    case = build_niah_case(seed=42)
    assert case.expected == {"key_e2d92bb192": "pass_6fd558f3c212af54f72aa3fa"}
    result = score_niah(case, LIVE_004_RAW_RESPONSE)
    # Strict outcome is byte-for-byte what the pilot recorded.
    assert result["score"] == 0.0 and result["passed"] is False and result["status"] == "protocol_error"
    assert result["retrieval_fraction"] == 0.0 and result["exact_retrieval"] is False
    assert result["outcome_correct"] is False and result["per_needle"][0]["correct"] is False
    assert result["parse_error"] == "Expecting value: line 1 column 1 (char 0)"
    assert result["raw_response"] == LIVE_004_RAW_RESPONSE
    # Lenient parse is recorded separately and never touches score/passed.
    assert result["format_strict_ok"] is False and result["lenient_parse_used"] is True
    assert result["lenient_retrieval_fraction"] == 1.0 and result["lenient_outcome_correct"] is True
    assert result["lenient_per_needle"] == [{**result["per_needle"][0], "correct": True}]


@pytest.mark.parametrize("tag", ["", "json"])
def test_fenced_correct_answer_with_bare_or_json_fence(tag):
    case = build_niah_case(depths=(0.1, 0.5, 0.9), missing_indices=(1,))
    result = score_niah(case, _fenced(json.dumps(case.expected, indent=2), tag))
    assert result["score"] == 0.0 and result["status"] == "protocol_error" and result["retrieval_fraction"] == 0.0
    assert result["lenient_parse_used"] and result["lenient_retrieval_fraction"] == 1.0
    assert result["lenient_outcome_correct"] and all(item["correct"] for item in result["lenient_per_needle"])


def test_fenced_wrong_value_fails_strict_and_lenient():
    case = build_niah_case(depths=(0.2, 0.8))
    wrong = {case.needles[0].key: "fabricated", case.needles[1].key: case.expected[case.needles[1].key]}
    result = score_niah(case, _fenced(json.dumps(wrong)))
    assert result["score"] == 0.0 and result["retrieval_fraction"] == 0.0 and result["format_strict_ok"] is False
    assert result["lenient_parse_used"] is True and result["lenient_outcome_correct"] is False
    assert result["lenient_retrieval_fraction"] == pytest.approx(0.5)
    assert [item["correct"] for item in result["lenient_per_needle"]] == [False, True]
    fully_wrong = score_niah(case, _fenced(json.dumps({key: "fabricated" for key in case.expected})))
    assert fully_wrong["score"] == 0.0 and fully_wrong["lenient_retrieval_fraction"] == 0.0
    assert fully_wrong["lenient_outcome_correct"] is False


def test_unfenced_correct_answer_is_strict_ok_and_lenient_unused():
    case = build_niah_case()
    result = score_niah(case, json.dumps(case.expected))
    assert result["score"] == 1.0 and result["passed"] and result["format_strict_ok"] is True
    assert result["lenient_parse_used"] is False and result["lenient_retrieval_fraction"] == 1.0
    assert result["lenient_outcome_correct"] is True and result["lenient_per_needle"] == result["per_needle"]


@pytest.mark.parametrize("response", [
    "```\n```json\n{payload}\n```\n```",           # double fence
    "```json\n```json\n{payload}\n```\n```",       # double tagged fence
    "Here is the answer:\n```json\n{payload}\n```",  # prose before
    "```json\n{payload}\n```\nHope this helps.",   # prose after
    "```json {payload} ```",                       # not a line fence
    "```json\n{payload}",                          # unterminated fence
    "```python\n{payload}\n```",                   # unsupported tag
    "```JSON\n{payload}\n```",                     # tag is exact
    "```json\n{payload}\n``` ```",                 # trailing junk fence
    "```json\n{payload} extra\n```",               # fenced but malformed
    "```json\r\n{payload}\r\n```",                 # CRLF fence: LF-only by decision (REV-D1-01)
])
def test_double_fences_and_prose_fail_both_strict_and_lenient(response):
    case = build_niah_case()
    result = score_niah(case, response.replace("{payload}", json.dumps(case.expected)))
    assert result["score"] == 0.0 and not result["passed"] and result["retrieval_fraction"] == 0.0
    assert result["format_strict_ok"] is False and result["lenient_parse_used"] is False
    assert result["lenient_retrieval_fraction"] == 0.0 and result["lenient_outcome_correct"] is False
    assert not any(item["correct"] for item in result["lenient_per_needle"])


def test_fenced_non_object_and_fenced_malformed_json_fail_lenient():
    case = build_niah_case()
    for response in (_fenced("[]"), _fenced("null"), _fenced('{"x": NaN}'), _fenced(""), "```json\n```"):
        result = score_niah(case, response)
        assert result["score"] == 0.0 and result["lenient_parse_used"] is False
        assert result["lenient_retrieval_fraction"] == 0.0 and result["lenient_outcome_correct"] is False


def test_lenient_never_changes_score_or_context_claims():
    case = build_niah_case(token_counter=exact_test_counter, tokenizer_verified=True,
                           tokenizer_id="test", template_id="test")
    result = score_niah(case, _fenced(json.dumps(case.expected)),
                        observed_input_tokens=case.accounting.actual_input_tokens,
                        truncation_reported=False, context_policy="reject-overflow-no-shift")
    assert result["context"]["actual_context_verified"] is True
    assert result["lenient_outcome_correct"] is True and result["lenient_retrieval_fraction"] == 1.0
    assert result["score"] == 0.0 and not result["passed"] and not result["eligible_for_context_claim"]
    assert result["status"] == "protocol_error"


def test_failure_status_and_tool_mode_record_lenient_fields_without_leniency():
    timeout = score_niah(build_niah_case(), None, failure_status="timeout")
    assert timeout["format_strict_ok"] is False and timeout["lenient_parse_used"] is False
    assert timeout["lenient_retrieval_fraction"] == 0.0 and timeout["lenient_outcome_correct"] is False
    assert timeout["lenient_per_needle"] == timeout["per_needle"]
    case = build_niah_case(mode="tool", depths=(0.1, 0.9))
    correct = score_niah(case, tool_message("submit_retrieved_keys", {"values": case.expected}))
    assert correct["format_strict_ok"] is True and correct["lenient_parse_used"] is False
    assert correct["lenient_retrieval_fraction"] == 1.0 and correct["lenient_outcome_correct"] is True
    # A well-formed native call to the wrong tool is format-valid but a selection failure: no leniency involved.
    wrong_tool = score_niah(case, tool_message("invented_submit", {"values": case.expected}))
    assert wrong_tool["format_strict_ok"] is True and wrong_tool["lenient_parse_used"] is False
    assert wrong_tool["score"] == 0.0 and wrong_tool["lenient_outcome_correct"] is False
    assert wrong_tool["lenient_retrieval_fraction"] == 1.0 and wrong_tool["lenient_per_needle"] == wrong_tool["per_needle"]
    # A fenced text completion in tool mode is a tool protocol error; the text fence path is never applied.
    fenced_text = score_niah(case, {"role": "assistant", "content": _fenced(json.dumps(case.expected))})
    assert fenced_text["score"] == 0.0 and fenced_text["format_strict_ok"] is False
    assert fenced_text["lenient_parse_used"] is False and fenced_text["lenient_retrieval_fraction"] == 0.0


def test_lenient_fields_present_and_json_serialisable_in_every_outcome():
    case = build_niah_case()
    for response, failure in ((json.dumps(case.expected), None), (LIVE_004_RAW_RESPONSE, None),
                              ("{} extra", None), (None, "timeout")):
        result = score_niah(case, response, failure_status=failure)
        assert all(key in result for key in LENIENT_KEYS)
        assert type(result["format_strict_ok"]) is bool and type(result["lenient_parse_used"]) is bool
        assert type(result["lenient_outcome_correct"]) is bool
        assert type(result["lenient_retrieval_fraction"]) is float
        json.dumps(result)


# ---------------------------------------------------------------------------------------------------
# LIVE-008. Sample niah/tool-multi/development/seed-42 in every candidate of
# artifacts/container-campaign/live-smoke-2/candidates/*/evaluation.json scored 0.0 with
# outcome_status="invalid_context" and context.verification_status="count_mismatch" although
# retrieval_fraction was 1.0, every per_needle entry correct and tool_score.passed True: our count was
# one token above the server's usage.prompt_tokens (4086/4085 at ctx 4864; 5113/5112 and 6397/6396 for
# the other two ctx sizes), only on the tool path. The non-tool variants matched exactly.
#
# tests/test_containers_selection.py does not exist in this tree, so per the branch assignment the
# selection-side regressions live here beside the scoring ones.
LIVE_008_ACCOUNTING = {"requested_input_tokens": 4096, "actual_input_tokens": 4086,
                       "context_capacity": 4864, "reserved_output_tokens": 512}
LIVE_008_OBSERVED = 4085
DATA = Path(__file__).parent / "data"


def capture_request_tools(tools, *, strict_tools=False):
    """Rebuild tools exactly as ``evaluations/capture.py`` builds ``request["tools"]`` for Inspect.

    Written out here on purpose: it is the contract the counting path has to match, not a call into the
    implementation under test. Mirrors inspect_tasks._tool_infos + capture.py's request assembly.
    """
    from inspect_ai.tool import ToolInfo, ToolParams

    infos = [ToolInfo(name=tool["function"]["name"], description=tool["function"]["description"],
                      parameters=ToolParams.model_validate(tool["function"]["parameters"])) for tool in tools]
    return [{"type": "function", "function": {
        "name": info.name, "description": info.description,
        "parameters": info.parameters.model_dump(exclude_none=True, by_alias=True),
        "strict": strict_tools,
    }} for info in infos]


def key_order_sensitive_counter(messages, tools):
    """A server that, like llama.cpp, renders the tool JSON in the request's own key order.

    llama.cpp renders a tool as name/description/parameters (``strict`` never reaches the template), so
    the base count here ignores both key order and ``strict``. A ``parameters`` object that is not
    byte-for-byte the one the serving path sends costs exactly one extra token: same schema, permuted
    keys, one token of disagreement, which is the live symptom.
    """
    rendered = [{"name": tool["function"]["name"], "description": tool["function"]["description"],
                 "parameters": tool["function"]["parameters"]} for tool in tools]
    base = (sum(len(message["content"].split()) + 4 for message in messages)
            + len(json.dumps(rendered, sort_keys=True).split()) + 2)
    served = all(json.dumps(tool["function"]["parameters"])
                 == json.dumps(capture_request_tools([tool])[0]["function"]["parameters"]) for tool in tools)
    return {"tokens": base + (0 if served else 1)}


def live_008_tool_case(**changes):
    """The tool-multi fixture carrying the accounting numbers the live campaign recorded."""
    case = build_niah_case(depths=(0.05, 0.5, 0.95), mode="tool", seed=42, token_counter=exact_test_counter,
                           tokenizer_verified=True, tokenizer_id="test", template_id="test")
    return replace(case, accounting=replace(case.accounting, **{**LIVE_008_ACCOUNTING, **changes}))


def live_008_score(case, **evidence):
    return score_niah(case, tool_message("submit_retrieved_keys", {"values": case.expected}), **evidence)


def test_live_008_authored_and_serving_tool_schemas_differ_only_by_key_order():
    case = build_niah_case(depths=(0.05, 0.5, 0.95), mode="tool", seed=42)
    authored, served = json.dumps(list(case.tools)), json.dumps(capture_request_tools(case.tools))
    assert authored != served                       # the live divergence, reproduced offline
    authored_params = json.dumps(case.tools[0]["function"]["parameters"])
    served_params = json.dumps(capture_request_tools(case.tools)[0]["function"]["parameters"])
    assert authored_params != served_params and len(authored_params) == len(served_params)
    assert sorted(authored_params) == sorted(served_params)   # a permutation, not different content
    assert json.loads(authored_params) == json.loads(served_params)


def test_live_008_selection_counts_the_serialisation_the_serving_path_sends():
    from llmbench.config import GenerationSettings, TaskSelection
    from llmbench.evaluations.selection import build_selected_cases

    seen = []

    def counter(messages, tools):
        seen.append(json.dumps(tools))
        return key_order_sensitive_counter(messages, tools)

    benchmarks = [TaskSelection(suite="niah", revision="local-niah-v1", task_ids=("multi", "tool-multi"))]
    cases, _ = build_selected_cases(
        benchmarks, requested_input_tokens=900, ctx_size=2048,
        generation=GenerationSettings(max_output_tokens=128), counter=counter,
        tokenizer_id="gguf-sha256:" + "0" * 64, template_id="sha256:" + "1" * 64, tokenizer_verified=True)
    tool_case = next(case for case in cases if case.mode == "tool")
    with_tools = [body for body in seen if json.loads(body)]
    assert with_tools, "the tool case must be counted with its tool schema"
    # Every counted body is byte-identical to what capture.py will send, so the modelled server's
    # one-token penalty never applies and the recorded count is the count the server will report.
    assert all(json.loads(body) == capture_request_tools(tool_case.tools) for body in with_tools)
    assert all(body == json.dumps(capture_request_tools(tool_case.tools)) for body in with_tools)
    assert tool_case.accounting.actual_input_tokens == key_order_sensitive_counter(
        list(tool_case.messages), capture_request_tools(tool_case.tools))["tokens"]
    # The authoring order (what was counted live) would have been one token higher.
    assert key_order_sensitive_counter(list(tool_case.messages), list(tool_case.tools))["tokens"] == (
        tool_case.accounting.actual_input_tokens + 1)


def test_live_008_one_token_mismatch_keeps_retrieval_and_still_withholds_the_context_claim():
    case = live_008_tool_case()
    result = live_008_score(case, observed_input_tokens=LIVE_008_OBSERVED, truncation_reported=False,
                            context_policy="reject-overflow-no-shift")
    assert case.accounting.actual_input_tokens - LIVE_008_OBSERVED == 1
    # Retrieval axis: exactly what the live sample recorded, no longer thrown away.
    assert result["retrieval_fraction"] == 1.0 and all(item["correct"] for item in result["per_needle"])
    assert result["tool_score"]["passed"] is True and result["outcome_correct"] is True
    assert result["score"] == 1.0 and result["passed"] is True
    assert result["status"] == "passed_context_unverified"
    # Context axis: unchanged, strict and fail-closed.
    assert result["context"]["verification_status"] == "count_mismatch"
    assert result["context"]["observed_count_matches"] is False
    assert result["context"]["actual_context_verified"] is False
    assert result["eligible_for_context_claim"] is False
    assert result["context_accounting_mismatch"] is True and result["context_accounting_only"] is True
    json.dumps(result)


def test_live_008_truncated_prompt_still_scores_zero_and_unverified():
    case = live_008_tool_case()
    for observed in (LIVE_008_OBSERVED, case.accounting.actual_input_tokens):
        result = live_008_score(case, observed_input_tokens=observed, truncation_reported=True,
                                context_policy="reject-overflow-no-shift")
        assert result["score"] == 0.0 and result["passed"] is False
        assert result["status"] == "invalid_context" and result["context_accounting_only"] is False
        assert result["context"]["verification_status"] == "truncated"
        assert result["context"]["actual_context_verified"] is False
        assert result["eligible_for_context_claim"] is False
        # The retrieval evidence is still recorded; it just cannot be scored as a pass.
        assert result["retrieval_fraction"] == 1.0 and result["outcome_correct"] is True


@pytest.mark.parametrize("label,observed,truncation", [
    ("beyond_slack_low", LIVE_008_ACCOUNTING["actual_input_tokens"] - ACCOUNTING_SLACK_TOKENS - 1, False),
    ("beyond_slack_high", LIVE_008_ACCOUNTING["actual_input_tokens"] + ACCOUNTING_SLACK_TOKENS + 1, False),
    ("truncation_unknown", LIVE_008_OBSERVED, None),
])
def test_live_008_large_or_unexplained_mismatch_still_invalidates(label, observed, truncation):
    case = live_008_tool_case()
    result = live_008_score(case, observed_input_tokens=observed, truncation_reported=truncation,
                            context_policy="reject-overflow-no-shift")
    assert result["score"] == 0.0 and result["passed"] is False, label
    assert result["status"] == "invalid_context" and result["context_accounting_only"] is False, label
    assert result["context"]["actual_context_verified"] is False, label
    assert result["eligible_for_context_claim"] is False, label


def test_live_008_served_prompt_that_misses_the_declared_budget_is_never_a_pass():
    # A small disagreement is not benign when the served prompt no longer fits the budget we claimed.
    case = live_008_tool_case(actual_input_tokens=4352)
    observed = case.accounting.context_capacity - case.accounting.reserved_output_tokens + 1
    assert abs(observed - case.accounting.actual_input_tokens) <= ACCOUNTING_SLACK_TOKENS
    result = live_008_score(case, observed_input_tokens=observed, truncation_reported=False,
                            context_policy="reject-overflow-no-shift")
    assert result["score"] == 0.0 and result["passed"] is False
    assert result["status"] == "invalid_context" and result["context_accounting_only"] is False
    assert result["eligible_for_context_claim"] is False


def test_live_008_matching_counts_and_the_non_tool_path_are_unchanged():
    tool_case = live_008_tool_case()
    verified = live_008_score(tool_case, observed_input_tokens=tool_case.accounting.actual_input_tokens,
                              truncation_reported=False, context_policy="reject-overflow-no-shift")
    assert verified["status"] == "passed" and verified["score"] == 1.0
    assert verified["eligible_for_context_claim"] is True
    assert verified["context_accounting_mismatch"] is False and verified["context_accounting_only"] is False
    text_case = build_niah_case(depths=(0.05, 0.5, 0.95), seed=42, token_counter=exact_test_counter,
                                tokenizer_verified=True, tokenizer_id="test", template_id="test")
    correct = json.dumps(text_case.expected)
    exact = score_niah(text_case, correct, observed_input_tokens=text_case.accounting.actual_input_tokens,
                       truncation_reported=False, context_policy="reject-overflow-no-shift")
    assert exact["status"] == "passed" and exact["score"] == 1.0 and exact["eligible_for_context_claim"]
    assert exact["context"]["verification_status"] == "verified"
    assert exact["context_accounting_mismatch"] is False and exact["context_accounting_only"] is False
    wrong = score_niah(text_case, json.dumps({key: "fabricated" for key in text_case.expected}),
                       observed_input_tokens=text_case.accounting.actual_input_tokens - 1,
                       truncation_reported=False, context_policy="reject-overflow-no-shift")
    assert wrong["score"] == 0.0 and wrong["passed"] is False  # a wrong answer never becomes a pass


def _fake_tokens(text):
    """Merge runs of ``}`` into one token, the way a real BPE vocabulary does.

    The count then depends on key order and not merely on length, which is what makes two equal-length
    permutations of the same tool schema tokenize differently on real hardware.
    """
    return len(re.findall(r"\}+|[^}]{1,4}", text))


class _CountRouteTransport:
    """Minimal llama.cpp stand-in whose rendered prompt keeps the request's own tool key order."""

    def __init__(self, *, order_sensitive=True):
        self.order_sensitive = order_sensitive
        self.bodies = []

    def _render(self, payload):
        tools = payload.get("tools", [])
        text = json.dumps(tools) if self.order_sensitive else json.dumps(tools, sort_keys=True)
        return json.dumps(payload.get("messages", [])) + text

    def request_json(self, method, path, payload=None):
        return self.request_any(method, path, payload)

    def request_any(self, method, path, payload=None):
        if path in {"/props", "/v1/models", "/slots"}:
            name = {"/props": "props-b11011.json", "/v1/models": "v1-models-b11011.json",
                    "/slots": "slots-b11011.json"}[path]
            return json.loads((DATA / name).read_text(encoding="utf-8"))["body"]
        if path == "/apply-template":
            self.bodies.append(payload)
            return {"prompt": self._render(payload)}
        if path == "/tokenize":
            return {"tokens": list(range(_fake_tokens(payload["content"])))}
        if path == "/v1/chat/completions":
            prompt = _fake_tokens(self._render(payload))
            return {"model": "qwen3.8-27b-q4_k_m", "usage": {"prompt_tokens": prompt, "completion_tokens": 1},
                    "choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": "r"}}]}
        raise AssertionError(f"unexpected request: {method} {path}")


@pytest.mark.parametrize("order_sensitive", [True, False])
def test_live_008_count_route_reports_tool_schema_key_order_sensitivity(order_sensitive):
    from llmbench.backends.base import LivePermission
    from llmbench.backends.llamacpp import ExpectedServer, LlamaCppBackend
    from llmbench.safety import SessionLock

    transport = _CountRouteTransport(order_sensitive=order_sensitive)
    backend = LlamaCppBackend(
        "http://127.0.0.1:8080", transport=transport, permissions=LivePermission(False, True, False),
        session_lock=SessionLock(False, True, False),
        expected=ExpectedServer(alias="qwen3.8-27b-q4_k_m", n_ctx=8192, build_info="b11011-aa39d7a3e"))
    backend.attach()
    result = backend.verify_count_route()
    assert [case["name"] for case in result["cases"]] == ["no-tools", "tools"]  # unchanged shape
    assert result["exact"] is True
    assert result["tool_schema_key_order_sensitive"] is order_sensitive
    assert type(result["tool_schema_reordered_tokens"]) is int
    json.dumps(result)
