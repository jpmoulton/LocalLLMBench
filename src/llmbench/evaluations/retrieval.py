"""Deterministic long-context fixtures with explicit token-accounting evidence.

The caller supplies the loaded tokenizer's complete chat-template counter. An
estimate never becomes runtime verification just because the answer is correct.
No tokenizer, model, or dataset is downloaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Sequence

from .tools import (
    ExpectedCall, FixtureIntegrityError, ToolDefinition, ToolExpectation,
    score_tool_response, strict_equal, strict_json_loads,
)

TokenCounter = Callable[[list[dict[str, str]], list[dict[str, Any]]], int]
SUITE_REVISION = "local-niah-v1"
NIAH_VARIANTS: dict[str, dict[str, Any]] = {
    "single-early": {"depths": (0.05,)}, "single-middle": {"depths": (0.5,)},
    "single-late": {"depths": (0.95,)}, "multi": {"depths": (0.05, 0.5, 0.95)},
    "missing": {"depths": (0.05, 0.5, 0.95), "missing_indices": (1,)},
    "tool-multi": {"depths": (0.05, 0.5, 0.95), "mode": "tool"},
}
"""Task id -> ``NiahCase`` keyword arguments: where the needles sit and how the answer is asked for."""
# LIVE-008: two renderings of the SAME fixture can disagree by a few tokens purely from prompt-template and
# JSON-serialisation effects (the first live campaign observed exactly one, on the tool path). This ceiling
# bounds how far the served prompt's own count may differ from ours before the served prompt is treated as a
# different prompt rather than the same prompt counted differently. It is far below one needle record
# ("CURRENT <key>: pass_<24 hex>\n"), so a disagreement inside it cannot add or remove fixture content. It
# never licenses a context claim: any disagreement still leaves ``actual_context_verified`` false.
ACCOUNTING_SLACK_TOKENS = 8


@dataclass(frozen=True)
class ContextAccounting:
    requested_input_tokens: int
    actual_input_tokens: int
    context_capacity: int
    reserved_output_tokens: int
    counting_method: Literal["exact-post-template", "estimated"]
    tokenizer_id: str
    template_id: str
    tool_schema_included: bool
    prompt_sha256: str
    tool_schema_sha256: str

    @property
    def tokenizer_verified(self) -> bool:
        return self.counting_method == "exact-post-template"

    def verify_runtime(
        self, observed_input_tokens: int | None, *, truncation_reported: bool | None = None,
        context_policy: Literal["reject-overflow-no-shift", "unknown"] = "unknown",
    ) -> dict[str, Any]:
        if observed_input_tokens is not None and (type(observed_input_tokens) is not int or observed_input_tokens < 0):
            raise ValueError("observed input count must be a nonnegative native integer")
        if truncation_reported is not None and type(truncation_reported) is not bool:
            raise ValueError("truncation evidence must be boolean or unknown")
        if context_policy not in {"reject-overflow-no-shift", "unknown"}:
            raise ValueError("unknown context policy evidence")
        matches = observed_input_tokens == self.actual_input_tokens if observed_input_tokens is not None else None
        verified = (self.tokenizer_verified and matches is True and truncation_reported is False
                    and context_policy == "reject-overflow-no-shift")
        return {
            **asdict(self), "tokenizer_verified": self.tokenizer_verified,
            "observed_input_tokens": observed_input_tokens, "observed_count_matches": matches,
            "truncation_reported": truncation_reported, "context_policy": context_policy,
            "actual_context_verified": verified,
            "verification_status": "verified" if verified else (
                "truncated" if truncation_reported else "count_mismatch" if matches is False
                else "estimated_tokenizer" if not self.tokenizer_verified else "runtime_count_missing" if matches is None
                else "truncation_evidence_missing" if truncation_reported is None else "context_policy_unverified"
            ),
        }


@dataclass(frozen=True)
class Needle:
    key: str
    expected_value: str | None
    requested_depth: float
    actual_depth: float | None
    prefix_input_tokens: int | None
    present: bool


@dataclass(frozen=True)
class NiahCase:
    task_id: str
    messages: tuple[dict[str, str], ...]
    tools: tuple[dict[str, Any], ...]
    expected: dict[str, str | None]
    needles: tuple[Needle, ...]
    accounting: ContextAccounting
    seed: int
    mode: Literal["text", "tool"]
    tool_expectation: ToolExpectation | None
    split: Literal["development", "holdout"] = "development"

    def validate_integrity(self) -> None:
        if _digest(self.messages) != self.accounting.prompt_sha256:
            raise FixtureIntegrityError("NIAH prompt changed under a stable prompt hash/task ID")
        if _digest(self.tools) != self.accounting.tool_schema_sha256:
            raise FixtureIntegrityError("NIAH tool schema changed under a stable schema hash/task ID")
        expected = {needle.key: needle.expected_value for needle in self.needles}
        if not strict_equal(self.expected, expected):
            raise FixtureIntegrityError("NIAH reference answers changed under a stable task ID")
        if self.tool_expectation is not None:
            self.tool_expectation.validate_integrity()

    def to_dict(self, *, include_answers: bool = False) -> dict[str, Any]:
        self.validate_integrity()
        result = asdict(self)
        if not include_answers:
            # Holdout reporting should use this metadata method, not asdict(case).
            result.pop("expected")
            result.pop("tool_expectation")
            result["needles"] = [{k: v for k, v in needle.items() if k != "expected_value"}
                                 for needle in result["needles"]]
            result.pop("messages")
        return result


def estimated_token_counter(messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> int:
    """Dependency-free development estimate; never suitable as tokenizer proof."""
    rendered = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, separators=(",", ":"))
    return (len(rendered.encode("utf-8")) + 3) // 4


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_niah_case(
    *,
    target_input_tokens: int = 2048,
    context_capacity: int = 4096,
    reserved_output_tokens: int = 256,
    depths: Sequence[float] = (0.5,),
    seed: int = 42,
    distractors: int = 2,
    missing_indices: Sequence[int] = (),
    mode: Literal["text", "tool"] = "text",
    split: Literal["development", "holdout"] = "development",
    token_counter: TokenCounter | None = None,
    tokenizer_verified: bool = False,
    tokenizer_id: str = "char-estimate-v1",
    template_id: str = "json-envelope-estimate-v1",
    max_padding_units: int = 524_288,
) -> NiahCase:
    """Fill up to an input budget without cutting a needle or a tool schema.

    Depths specify desired fractions of the filler. Actual token coordinates
    include the system message/template and are saved separately. The selected
    count can be slightly below the target because indivisible fixture records
    are never truncated. The same counter receives complete messages AND tools.
    """
    for name, value, minimum in (
        ("target_input_tokens", target_input_tokens, 1), ("context_capacity", context_capacity, 1),
        ("reserved_output_tokens", reserved_output_tokens, 1), ("distractors", distractors, 0),
        ("max_padding_units", max_padding_units, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if target_input_tokens + reserved_output_tokens > context_capacity:
        raise ValueError("requested input plus output reserve exceeds context capacity")
    if not depths or len(depths) > 64 or any(type(d) not in (int, float) or not 0 <= d <= 1 for d in depths):
        raise ValueError("provide 1..64 finite needle depths between zero and one")
    if mode not in {"text", "tool"} or split not in {"development", "holdout"}:
        raise ValueError("unknown output mode or split")
    missing = set(missing_indices)
    if len(missing) != len(missing_indices) or any(type(i) is not int or i < 0 or i >= len(depths) for i in missing):
        raise ValueError("missing_indices must be distinct valid needle indices")
    if tokenizer_verified and (token_counter is None or not tokenizer_id or not template_id):
        raise ValueError("verified counting requires an explicit full-template counter and provenance")
    if tokenizer_verified and token_counter is estimated_token_counter:
        raise ValueError("the bundled character estimator cannot be marked verified")
    counter = token_counter or estimated_token_counter
    rng = random.Random(f"{SUITE_REVISION}:{split}:{seed}")
    keys = [f"key_{rng.getrandbits(40):010x}" for _ in depths]
    values = [f"pass_{rng.getrandbits(96):024x}" for _ in depths]
    expected = {key: None if index in missing else values[index] for index, key in enumerate(keys)}
    definition = ToolDefinition("submit_retrieved_keys", {
        "type": "object", "required": ["values"], "additionalProperties": False,
        "properties": {"values": {
            "type": "object", "required": keys, "additionalProperties": False,
            "properties": {key: {"type": ["string", "null"]} for key in keys},
        }},
    }, "Submit the exact CURRENT passkeys; use null where no CURRENT record exists.")
    tools = [definition.openai_schema()] if mode == "tool" else []
    system = (
        "You are checking an archive. Only records labeled CURRENT are authoritative. "
        "OBSOLETE records and unrelated archive notes are distractors. For each requested key, "
        "return its exact case-sensitive CURRENT passkey, or null if no CURRENT record exists. "
        "Do not infer missing values. "
        + ("Call submit_retrieved_keys exactly once with the values object." if mode == "tool"
           else "Answer only a JSON object mapping each requested key to its passkey or null.")
    )
    query = "\nEND ARCHIVE\nRequested keys: " + json.dumps(keys) + "\n"
    records = [f"CURRENT {key}: {values[i]}\n" if i not in missing else "" for i, key in enumerate(keys)]
    obsolete = []
    for index in range(distractors):
        key = keys[index % len(keys)]
        obsolete.append(f"OBSOLETE {key}: pass_{rng.getrandbits(96):024x}\n")
    # Seeded notes do not contain any expected passkey. Padding uses stable words,
    # keeping fixture construction linear in the final input's size.
    filler_words = [f"archive_{rng.getrandbits(40):010x}" for _ in range(64)]

    def render(units: int) -> tuple[list[dict[str, str]], list[int | None]]:
        slots: dict[int, list[tuple[int, str]]] = {}
        for index, (depth, record) in enumerate(zip(depths, records)):
            if record:
                slots.setdefault(round(float(depth) * units), []).append((index, record))
        for index, record in enumerate(obsolete):
            slots.setdefault(round((index + 1) / (len(obsolete) + 1) * units), []).append((-1, record))
        chunks, offsets, character_count = ["BEGIN ARCHIVE\n"], [None] * len(keys), len("BEGIN ARCHIVE\n")
        for position in range(units + 1):
            for needle_index, record in slots.get(position, []):
                if needle_index >= 0:
                    offsets[needle_index] = character_count
                chunks.append(record)
                character_count += len(record)
            if position < units:
                word = filler_words[position % len(filler_words)] + ("\n" if position % 8 == 7 else " ")
                chunks.append(word)
                character_count += len(word)
        content = "".join(chunks) + query
        return [{"role": "system", "content": system}, {"role": "user", "content": content}], offsets

    def count(messages: list[dict[str, str]]) -> int:
        result = counter(messages, tools)
        if type(result) is not int or result < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return result

    messages, offsets = render(0)
    actual_count = count(messages)
    if actual_count > target_input_tokens:
        raise ValueError(f"fixture and template require {actual_count} input tokens, exceeding requested target")
    best_units, low, high = 0, 0, min(max_padding_units, max(32, target_input_tokens * 4))
    # Count is normally monotonic at this scale. Retain only proven-fitting
    # candidates so small tokenizer boundary effects cannot overflow the budget.
    while low <= high:
        middle = (low + high) // 2
        candidate, candidate_offsets = render(middle)
        candidate_count = count(candidate)
        if candidate_count <= target_input_tokens:
            best_units = middle
            messages, offsets, actual_count = candidate, candidate_offsets, candidate_count
            low = middle + 1
        else:
            high = middle - 1
    for units in range(best_units + 1, min(best_units + 9, max_padding_units + 1)):
        candidate, candidate_offsets = render(units)
        candidate_count = count(candidate)
        if actual_count <= candidate_count <= target_input_tokens:
            messages, offsets, actual_count = candidate, candidate_offsets, candidate_count
    if actual_count + reserved_output_tokens > context_capacity:
        raise AssertionError("internal context accounting overflow")
    needles = []
    for index, offset in enumerate(offsets):
        prefix_count = None
        if offset is not None:
            prefix = [messages[0], {"role": "user", "content": messages[1]["content"][:offset]}]
            prefix_count = count(prefix)
        needles.append(Needle(keys[index], expected[keys[index]], float(depths[index]),
                              min(1.0, prefix_count / actual_count) if prefix_count is not None and actual_count else None,
                              prefix_count, index not in missing))
    accounting = ContextAccounting(
        target_input_tokens, actual_count, context_capacity, reserved_output_tokens,
        "exact-post-template" if tokenizer_verified else "estimated", tokenizer_id, template_id,
        bool(tools), _digest(messages), _digest(tools),
    )
    identity = _digest({"messages": messages, "tools": tools, "accounting": asdict(accounting),
                        "split": split, "seed": seed})[:16]
    task_id = f"retrieval/{mode}/{len(depths)}-needle/{split}/{identity}"
    expectation = ToolExpectation(task_id, (definition,), (ExpectedCall("submit_retrieved_keys", {"values": expected}),))
    return NiahCase(task_id, tuple(messages), tuple(tools), expected, tuple(needles), accounting, seed,
                    mode, expectation if mode == "tool" else None, split)


def strip_one_markdown_fence(text: str) -> str | None:
    """Return the body of exactly one surrounding ``` or ```json line fence, else None.

    Deterministic LIVE-004 lenient format: surrounding whitespace is ignored, the
    opening line must be exactly ``` or ```json, the closing ``` must start the
    final line, and nothing may precede or follow the fence. No bracket hunting,
    no nested-fence removal, no semantic repair. The body is whitespace-stripped.
    Fence lines are LF-delimited only (a CRLF fence fails closed).
    """
    if type(text) is not str:
        return None
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None
    tag, newline, remainder = stripped[3:].partition("\n")
    if not newline or tag not in ("", "json") or not remainder.endswith("\n```"):
        return None
    return remainder[: -len("\n```")].strip()


def _lenient_parse(response: Any) -> dict[str, Any] | None:
    """Strict JSON object parse after removing exactly one fence; None when not applicable."""
    body = strip_one_markdown_fence(response)
    if body is None:
        return None
    try:
        parsed = strict_json_loads(body)
    except (ValueError, TypeError, RecursionError):
        return None
    return parsed if type(parsed) is dict else None


def score_niah(
    case: NiahCase,
    response: Any,
    *,
    observed_input_tokens: int | None = None,
    truncation_reported: bool | None = None,
    context_policy: Literal["reject-overflow-no-shift", "unknown"] = "unknown",
    failure_status: str | None = None,
) -> dict[str, Any]:
    """Score exact values and keep input verification separate from correctness.

    The strict result (``score``/``passed``/``status``/``retrieval_fraction``) is
    the benchmark outcome. LIVE-004 fields report retrieval correctness under a
    labelled lenient format (text mode only: exactly one surrounding markdown
    fence removed) without ever changing the strict outcome:
    ``format_strict_ok``, ``lenient_parse_used``, ``lenient_retrieval_fraction``,
    ``lenient_per_needle`` and ``lenient_outcome_correct``. When the lenient
    parse is not used the lenient fields mirror the strict ones.

    LIVE-008 separates the context-accounting axis from the retrieval axis.
    Context verification stays strict and fail-closed: any disagreement between
    our count and the server's own prompt count leaves ``actual_context_verified``
    false, ``verification_status`` "count_mismatch" and ``eligible_for_context_claim``
    false, so the candidate can never make a validated context claim. A
    disagreement only destroys the retrieval outcome when the served prompt
    cannot be shown to be this fixture -- reported truncation, unknown truncation,
    a served prompt that did not fit the declared budget, or a disagreement larger
    than ``ACCOUNTING_SLACK_TOKENS``. Otherwise the sample keeps its retrieval
    result under the labelled status "passed_context_unverified", with
    ``context_accounting_mismatch`` and ``context_accounting_only`` recording why.
    """
    case.validate_integrity()
    evidence = case.accounting.verify_runtime(observed_input_tokens, truncation_reported=truncation_reported,
                                              context_policy=context_policy)
    failure_categories = {"timeout", "transport_error", "environment_error", "unsupported", "cancelled"}
    if failure_status is not None and failure_status not in failure_categories:
        raise ValueError("unknown failure status")
    parsed, parse_error, tool_result = None, None, None
    if failure_status is None:
        if case.mode == "tool":
            tool_result = score_tool_response(response, case.tool_expectation).to_dict()
            if tool_result["protocol_valid"]:
                try:
                    parsed = strict_json_loads(response["tool_calls"][0]["function"]["arguments"]).get("values")
                except (KeyError, IndexError, TypeError, ValueError):
                    parse_error = "missing values object"
            else:
                parse_error = "invalid native tool call"
        else:
            try:
                parsed = strict_json_loads(response)
            except (ValueError, TypeError, RecursionError) as exc:
                parse_error = str(exc)
    if parsed is not None and type(parsed) is not dict:
        parse_error = "answer must be a JSON object"
        parsed = None

    def needle_results(answer: Any) -> list[dict[str, Any]]:
        rows = []
        for needle in case.needles:
            correct = (failure_status is None and type(answer) is dict and needle.key in answer
                       and strict_equal(answer[needle.key], needle.expected_value))
            rows.append({"key": needle.key, "present": needle.present, "correct": correct,
                         "requested_depth": needle.requested_depth, "actual_depth": needle.actual_depth})
        return rows

    per_needle = needle_results(parsed)
    exact = failure_status is None and strict_equal(parsed, case.expected)
    outcome = exact and (tool_result is None or tool_result["passed"])
    # LIVE-004: lenient format is text-mode only, attempted only after a strict
    # parse failure, and reported beside (never instead of) the strict outcome.
    format_strict_ok = failure_status is None and parse_error is None
    lenient_parsed = _lenient_parse(response) if (not format_strict_ok and failure_status is None
                                                 and case.mode == "text") else None
    lenient_parse_used = lenient_parsed is not None
    lenient_per_needle = (needle_results(lenient_parsed) if lenient_parse_used
                          else [dict(row) for row in per_needle])  # copy: no aliasing
    lenient_outcome = (strict_equal(lenient_parsed, case.expected) if lenient_parse_used else outcome)
    # LIVE-008: two axes. The context axis stays fail-closed above (a mismatch already forces
    # actual_context_verified false). The retrieval axis is destroyed only by evidence that the served
    # prompt was not this fixture: reported truncation, unknown truncation, a served prompt that did not
    # fit the declared budget, or a disagreement too large to be a rendering artefact. Missing evidence
    # permits diagnostics but no context claim.
    observed = evidence["observed_input_tokens"]
    count_mismatch = evidence["observed_count_matches"] is False
    accounting_only = (count_mismatch and truncation_reported is False and observed is not None
                       and observed + case.accounting.reserved_output_tokens <= case.accounting.context_capacity
                       and abs(observed - case.accounting.actual_input_tokens) <= ACCOUNTING_SLACK_TOKENS)
    invalid_context = truncation_reported is True or (count_mismatch and not accounting_only)
    passed = outcome and not invalid_context
    status = failure_status or ("invalid_context" if invalid_context
                                else "passed_context_unverified" if passed and count_mismatch
                                else "passed" if passed
                                else "protocol_error" if parse_error else "incorrect")
    return {
        "task_id": case.task_id, "suite": "retrieval", "suite_revision": SUITE_REVISION,
        "passed": passed, "score": float(passed), "status": status,
        "retrieval_fraction": sum(item["correct"] for item in per_needle) / len(per_needle),
        "exact_retrieval": exact, "outcome_correct": outcome,
        "eligible_for_context_claim": passed and evidence["actual_context_verified"],
        "context_accounting_mismatch": count_mismatch, "context_accounting_only": accounting_only,
        "per_needle": per_needle, "context": evidence,
        "format_strict_ok": format_strict_ok, "lenient_parse_used": lenient_parse_used,
        "lenient_retrieval_fraction": sum(item["correct"] for item in lenient_per_needle) / len(lenient_per_needle),
        "lenient_per_needle": lenient_per_needle, "lenient_outcome_correct": lenient_outcome,
        "tool_score": tool_result, "parse_error": parse_error, "raw_response": response,
    }
