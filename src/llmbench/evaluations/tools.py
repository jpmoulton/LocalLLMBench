"""Strict, deterministic tool-call checks and an entirely in-memory tool fixture.

These are local regression fixtures, not BFCL or another public benchmark. Raw API
arguments are scored before parsing conveniences, coercion, or repair. Nothing in
this module executes model code, opens files, or contacts an inference backend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

SUITE_REVISION = "local-tools-v1"


class FixtureIntegrityError(ValueError):
    """A fixture changed after its identity was assigned; invalidate the run."""

    abort_campaign = True


def strict_json_loads(text: str) -> Any:
    """Parse JSON without accepting NaN, infinity, or duplicate object keys."""
    if type(text) is not str:
        raise ValueError("raw arguments must be a JSON string")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number: {value}")
        return number

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)


def strict_equal(left: Any, right: Any) -> bool:
    """JSON value equality with no type coercion, including bool versus int."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(strict_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def schema_errors(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Validate the explicit JSON Schema subset used by our frozen fixtures.

    Unsupported keywords raise rather than silently claiming schema validation.
    Arbitrary upstream schemas should use their upstream validator/adapter.
    """
    supported = {
        "type", "properties", "required", "additionalProperties", "items", "enum", "const",
        "minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength", "pattern",
        "description", "title",
    }
    unknown = set(schema) - supported
    if unknown:
        raise ValueError(f"unsupported fixture schema keywords: {sorted(unknown)}")
    expected_types = schema.get("type")
    types = expected_types if isinstance(expected_types, list) else [expected_types]
    type_checks = {
        "object": lambda x: type(x) is dict,
        "array": lambda x: type(x) is list,
        "string": lambda x: type(x) is str,
        "integer": lambda x: type(x) is int,
        "number": lambda x: type(x) is int or (type(x) is float and math.isfinite(x)),
        "boolean": lambda x: type(x) is bool,
        "null": lambda x: x is None,
    }
    if any(t not in type_checks for t in types):
        raise ValueError(f"fixture schema needs supported explicit type: {types}")
    if not any(type_checks[t](value) for t in types):
        return [f"{path}: expected {expected_types}, got {type(value).__name__}"]
    errors = []
    if "enum" in schema and not any(strict_equal(value, x) for x in schema["enum"]):
        errors.append(f"{path}: value not in enum")
    if "const" in schema and not strict_equal(value, schema["const"]):
        errors.append(f"{path}: value differs from const")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        for key, item in value.items():
            if type(key) is not str:
                errors.append(f"{path}: non-string object key")
            elif key in props:
                errors.extend(schema_errors(item, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}.{key}: invented field")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(schema_errors(item, schema["additionalProperties"], f"{path}.{key}"))
    if isinstance(value, list):
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]"))
        for bound, comparator in (("minItems", lambda a, b: a < b), ("maxItems", lambda a, b: a > b)):
            if bound in schema and comparator(len(value), schema[bound]):
                errors.append(f"{path}: violates {bound}")
    if type(value) is str:
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: below minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: above maxLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: pattern mismatch")
    if type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            errors.append(f"{path}: non-finite number")
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    return errors


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    parameters: dict[str, Any]
    description: str = "Deterministic benchmark fixture tool."

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }}


@dataclass(frozen=True)
class ExpectedCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolExpectation:
    task_id: str
    tools: tuple[ToolDefinition, ...]
    calls: tuple[ExpectedCall, ...] = ()
    ordered: bool = True
    expected_text: str | None = None
    _fixture_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_fixture_hash", self._current_hash())

    def _current_hash(self) -> str:
        payload = {"task_id": self.task_id, "tools": [asdict(tool) for tool in self.tools],
                   "calls": [asdict(call) for call in self.calls], "ordered": self.ordered,
                   "expected_text": self.expected_text}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    def validate_integrity(self) -> None:
        if self._current_hash() != self._fixture_hash:
            raise FixtureIntegrityError("tool fixture definitions or expected arguments changed under a stable ID")


@dataclass(frozen=True)
class ParsedCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass(frozen=True)
class ToolScore:
    task_id: str
    passed: bool
    score: float
    status: str
    protocol_valid: bool
    schema_valid: bool
    selection_correct: bool
    arguments_correct: bool
    call_count_correct: bool
    errors: tuple[str, ...]
    raw_response: Any
    suite: str = "tools"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_tool_response(response: Any) -> tuple[list[ParsedCall], list[str]]:
    """Parse an OpenAI *assistant message*, retaining raw argument strings.

    Pass the API's choices[0].message, not a repaired Inspect argument dictionary.
    A transport/HTTP error must instead be recorded via ``failed_tool_score``.
    """
    if type(response) is not dict:
        return [], ["assistant message must be an object"]
    if response.get("role", "assistant") != "assistant":
        return [], ["message role must be assistant"]
    if "error" in response:
        return [], ["API error is not an assistant tool response"]
    if "function_call" in response:
        return [], ["legacy function_call is unsupported in the native tool fixture"]
    if "content" not in response and "tool_calls" not in response:
        return [], ["assistant message must contain content or native tool_calls"]
    if response.get("content") is not None and type(response["content"]) is not str:
        return [], ["assistant text content must be a string or null"]
    raw_calls = response.get("tool_calls", [])
    if raw_calls is None:
        raw_calls = []
    if type(raw_calls) is not list:
        return [], ["tool_calls must be an array"]
    result, errors, seen_ids = [], [], set()
    for index, raw in enumerate(raw_calls):
        prefix = f"call[{index}]"
        if type(raw) is not dict or raw.get("type") != "function":
            errors.append(f"{prefix}: expected a native function call")
            continue
        call_id = raw.get("id")
        if type(call_id) is not str or not call_id:
            errors.append(f"{prefix}: missing call id")
            continue
        if call_id in seen_ids:
            errors.append(f"{prefix}: duplicate call id")
            continue
        seen_ids.add(call_id)
        function = raw.get("function")
        if type(function) is not dict or type(function.get("name")) is not str:
            errors.append(f"{prefix}: missing function name")
            continue
        raw_args = function.get("arguments")
        try:
            arguments = strict_json_loads(raw_args)
        except (ValueError, TypeError, RecursionError) as exc:
            errors.append(f"{prefix}: invalid raw JSON: {exc}")
            continue
        if type(arguments) is not dict:
            errors.append(f"{prefix}: arguments must be an object")
            continue
        result.append(ParsedCall(call_id, function["name"], arguments, raw_args))
    return result, errors


def failed_tool_score(task_id: str, status: str, detail: str, raw_response: Any = None) -> ToolScore:
    if status not in {"timeout", "transport_error", "environment_error", "unsupported", "cancelled"}:
        raise ValueError("use an explicit infrastructure/capability failure category")
    return ToolScore(task_id, False, 0.0, status, False, False, False, False, False,
                     (detail,), copy.deepcopy(raw_response))


def score_tool_response(response: Any, expectation: ToolExpectation) -> ToolScore:
    expectation.validate_integrity()
    calls, protocol_errors = parse_tool_response(response)
    errors = list(protocol_errors)
    schema_ok = not protocol_errors
    definitions = {tool.name: tool for tool in expectation.tools}
    if len(definitions) != len(expectation.tools):
        raise ValueError("fixture contains duplicate tool definitions")
    for call in calls:
        if call.name not in definitions:
            errors.append(f"unknown tool: {call.name}")
            schema_ok = False
        else:
            call_errors = schema_errors(call.arguments, definitions[call.name].parameters)
            errors.extend(call_errors)
            schema_ok = schema_ok and not call_errors
    count_ok = not protocol_errors and len(calls) == len(expectation.calls)
    if not count_ok:
        errors.append(f"expected {len(expectation.calls)} calls, got {len(calls)} parsed calls")
    expected = list(expectation.calls)
    if not expectation.ordered:
        # Match whole name+argument pairs; sorting by name loses duplicate-call semantics.
        remaining = calls[:]
        for wanted in expected:
            for index, actual in enumerate(remaining):
                if actual.name == wanted.name and strict_equal(actual.arguments, wanted.arguments):
                    remaining.pop(index)
                    break
            else:
                break
        args_ok = count_ok and not remaining
        selection_ok = count_ok and sorted(c.name for c in calls) == sorted(c.name for c in expected)
    else:
        selection_ok = count_ok and all(a.name == e.name for a, e in zip(calls, expected))
        args_ok = selection_ok and all(strict_equal(a.arguments, e.arguments) for a, e in zip(calls, expected))
    if not selection_ok:
        errors.append("tool selection or required order is incorrect")
    if not args_ok:
        errors.append("argument values or JSON types are incorrect")
    text_ok = expectation.expected_text is None or (
        type(response) is dict and response.get("content") == expectation.expected_text
    )
    if not text_ok:
        errors.append("required no-call answer differs")
    passed = not protocol_errors and schema_ok and count_ok and selection_ok and args_ok and text_ok
    status = "passed" if passed else (
        "protocol_error" if protocol_errors else "schema_error" if not schema_ok else "incorrect"
    )
    return ToolScore(expectation.task_id, passed, float(passed), status, not protocol_errors,
                     schema_ok, selection_ok, args_ok, count_ok, tuple(errors), copy.deepcopy(response))


def tool_message(name: str, arguments: dict[str, Any] | str, call_id: str = "call_1") -> dict[str, Any]:
    """Build fixture input; string arguments are deliberately left unmodified."""
    encoded = arguments if type(arguments) is str else json.dumps(arguments, allow_nan=False)
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": name, "arguments": encoded},
    }]}


def summarize_tool_scores(scores: Iterable[ToolScore]) -> dict[str, Any]:
    rows = list(scores)
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.status] = statuses.get(row.status, 0) + 1
    passed = sum(row.passed for row in rows)
    completed = sum(row.status in {"passed", "incorrect", "protocol_error", "schema_error"} for row in rows)
    return {"attempted": len(rows), "passed": passed, "completed": completed,
            "success_all_required": passed / len(rows) if rows else None,
            "completion_rate": completed / len(rows) if rows else None, "status_counts": statuses}


def strict_argument_fixture() -> ToolExpectation:
    return ToolExpectation("tools/nested-exact-v1", (ToolDefinition("configure_job", {
        "type": "object", "additionalProperties": False, "required": ["path", "options"],
        "properties": {
            "path": {"type": "string"},
            "options": {"type": "object", "additionalProperties": False,
                        "required": ["retries", "enabled", "tags"], "properties": {
                            "retries": {"type": "integer", "minimum": 0},
                            "enabled": {"type": "boolean"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        }},
        },
    }),), (ExpectedCall("configure_job", {
        "path": "src/CacheConfig.ts", "options": {"retries": 3, "enabled": True, "tags": ["cpu", "gpu"]},
    }),))


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


@dataclass
class ToolEpisode:
    """A bounded read/edit/test chain with revision conflicts and clean resets.

    ``run_tests`` checks evaluator-owned fixture content, and does not invoke an
    interpreter. Actual code-execution suites belong in isolated workers.
    """

    task_id: str = "tools/read-edit-test-v1"
    max_calls: int = 8
    repair_budget: int = 1
    path: str = "src/config.ts"
    initial_content: str = "export const retries = 1;\n"
    expected_content: str = "export const retries = 3;\n"
    trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    content: str = field(default="", init=False)
    revision: int = field(default=1, init=False)
    read_seen: bool = field(default=False, init=False)
    tests_passed: bool = field(default=False, init=False)
    errors: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)
    terminal_failure: str | None = field(default=None, init=False)
    seen_call_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if type(self.repair_budget) is not int or self.repair_budget < 0:
            raise ValueError("repair_budget must be a nonnegative integer")
        self.reset()

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        text = {"type": "string"}
        return (
            ToolDefinition("read_file", _object_schema({"path": text})),
            ToolDefinition("apply_patch", _object_schema({
                "path": text, "expected_revision": {"type": "integer"}, "content": text,
            })),
            ToolDefinition("run_tests", _object_schema({"path": text})),
        )

    def reset(self) -> None:
        self.trace.clear()
        self.content = self.initial_content
        self.revision, self.errors, self.calls = 1, 0, 0
        self.read_seen = self.tests_passed = False
        self.terminal_failure = None
        self.seen_call_ids.clear()

    def consume(self, response: Any) -> list[dict[str, Any]]:
        if self.terminal_failure:
            return [{"error": "episode_terminated", "reason": self.terminal_failure}]
        parsed, parse_errors = parse_tool_response(response)
        self.trace.append({"raw_response": copy.deepcopy(response), "results": []})
        results = self.trace[-1]["results"]
        if parse_errors or len(parsed) != 1:
            # The fixture has dependent tools; parallel submissions are not a valid chain.
            raw_calls = response.get("tool_calls") if type(response) is dict else None
            self.calls += max(1, len(raw_calls) if type(raw_calls) is list else len(parsed))
            self._error(results, "protocol_error", "; ".join(parse_errors) or "exactly one dependent call required")
        else:
            call = parsed[0]
            self.calls += 1
            if self.calls > self.max_calls:
                self.terminal_failure = "call_budget_exhausted"
                results.append({"error": self.terminal_failure, "tool_call_id": call.call_id})
            elif call.call_id in self.seen_call_ids:
                self._error(results, "duplicate_call_id", call.call_id)
            else:
                self.seen_call_ids.add(call.call_id)
                definition = next((tool for tool in self.tools if tool.name == call.name), None)
                if definition is None:
                    self._error(results, "unknown_tool", call.name)
                else:
                    issues = schema_errors(call.arguments, definition.parameters)
                    if issues:
                        self._error(results, "schema_error", "; ".join(issues))
                    else:
                        self._execute(call, results)
            for result in results:
                result.setdefault("tool_call_id", call.call_id)
        if self.errors > self.repair_budget:
            self.terminal_failure = "repair_budget_exhausted"
        if self.calls >= self.max_calls and not self.tests_passed:
            self.terminal_failure = self.terminal_failure or "call_budget_exhausted"
        return copy.deepcopy(results)

    def _error(self, results: list[dict[str, Any]], category: str, detail: str) -> None:
        self.errors += 1
        results.append({"error": category, "detail": detail})

    def _execute(self, call: ParsedCall, results: list[dict[str, Any]]) -> None:
        args = call.arguments
        if args["path"] != self.path:
            self._error(results, "file_not_found", args["path"])
        elif call.name == "read_file":
            self.read_seen = True
            results.append({"content": self.content, "revision": self.revision})
        elif call.name == "apply_patch":
            if not self.read_seen:
                self._error(results, "dependency_error", "read the file before editing")
            elif args["expected_revision"] != self.revision:
                self._error(results, "revision_conflict", f"current revision is {self.revision}")
            else:
                self.content = args["content"]
                self.revision += 1
                self.tests_passed = False
                results.append({"applied": True, "revision": self.revision})
        elif call.name == "run_tests":
            if not self.read_seen or self.revision == 1:
                self._error(results, "dependency_error", "read and apply a patch before testing")
            else:
                self.tests_passed = self.content == self.expected_content
                if self.tests_passed:
                    results.append({"passed": True, "tests": 2, "execution": "in-memory-fixture-check"})
                else:
                    self._error(results, "test_failure", "expected retries = 3")

    def result(self) -> dict[str, Any]:
        passed = self.tests_passed and self.content == self.expected_content and self.terminal_failure is None
        return {
            "task_id": self.task_id, "suite": "tools", "passed": passed, "score": float(passed),
            "status": "passed" if passed else self.terminal_failure or "incomplete",
            "first_attempt_success": passed and self.errors == 0,
            "success_after_repairs": passed and self.errors > 0,
            "calls": self.calls, "errors": self.errors, "repairs": self.errors,
            "execution": "in-memory-fixture-check", "trace": copy.deepcopy(self.trace),
            "final_state": {"content": self.content, "revision": self.revision, "tests_passed": self.tests_passed},
        }
