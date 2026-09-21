"""BFCL (Berkeley Function Calling Leaderboard) adapter: pinned data, offline scoring, no judge.

Upstream is the Berkeley Function Calling Leaderboard (repo ``ShishirPatil/gorilla``, PyPI package
``bfcl-eval``, Apache-2.0). This adapter targets the V4 package line but scores only the single-turn,
deterministic, AST-checked categories (``simple``, ``multiple``, ``parallel``, ``parallel_multiple``,
``irrelevance`` and the ``live_*`` equivalents). ``web_search`` (SerpAPI key + live network), ``memory``
(downloads an embedding model), ``format_sensitivity`` (non-scoring) and ``multi_turn`` (stateful
environment, future work) are out of scope and are rejected if an image bakes them; see
``EXCLUDED_CATEGORIES``.

Three properties matter more than leaderboard parity:

1. **The scorer is an AST matcher, not an LLM judge.** It is reimplemented here rather than imported from
   ``bfcl_eval`` internals: we need a deterministic, unit-testable scorer whose rules we control, and
   upstream's package layout is not a stable API. The rules are documented on :func:`value_matches` and
   :func:`score_calls`, and every divergence from upstream is called out there.
2. **Execution mode is a first-class axis.** ``native`` sends OpenAI ``tools=`` and reads
   ``message.tool_calls``, exercising llama.cpp's Jinja template, its JSON-schema->GBNF grammar and its
   tool-call parser - i.e. what actually ships. ``prompt`` formats the same schemas into a system prompt
   and parses the text back. Comparing the two separates "the model broke" from "llama.cpp's tool
   handling broke". The mode is part of the task id and is recorded on every row.
3. **Honest denominators.** Every declared task id comes back with a row: transport failures, malformed
   output, budget exhaustion and adapter bugs all produce scored-0 rows with an explicit
   ``outcome_status``. Raw response bytes are persisted through ``context.artifacts`` *before* they are
   parsed; evidence that cannot be stored aborts the run instead of being dropped.

The dataset is read from ``<context.dataset_root>/bfcl/<revision>/`` and is never fetched: a missing,
mismatched or tampered bundle makes :meth:`BfclAdapter.available` return ``(False, reason)``. The exact
upstream package version and gorilla commit that produced the bundle live in its ``manifest.json``, are
verified at preflight and are copied onto every row, so a row can always be traced to the bytes scored.

Scores from a subset, and any ``prompt``-mode score (our prompt contract is JSON, not upstream's Python
call syntax), are ours - they are not official BFCL leaderboard numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from . import (
    REQUIRED_SAMPLE_KEYS, BenchmarkAborted, BenchmarkContext, BenchmarkUnavailable, missing_rows,
)
from ..config import RunMode
from ..evaluations.tools import parse_tool_response, strict_json_loads
from ..safety import OperationForbidden

BENCHMARK_ID = "bfcl"
CATEGORY = "tools"
REVISION = "bfcl-v4-ast-subset-v1"
"""Our pin generation. The upstream identity it must be built from is declared, and verified, in the
bundle manifest (``upstream_package``/``upstream_version``/``source_commit``); the adapter refuses a
bundle whose ``adapter_revision`` is not exactly this string."""

UPSTREAM_PACKAGE = "bfcl-eval"
UPSTREAM_REPO = "https://github.com/ShishirPatil/gorilla"
UPSTREAM_LICENSE = "Apache-2.0"

AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance",
                  "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
                  "live_irrelevance")
"""The offline, deterministic, AST-scored categories this adapter supports, in report order."""

EXCLUDED_CATEGORIES = {
    "web_search": "needs a SerpAPI key and live network access at run time",
    "memory": "downloads an embedding model at run time",
    "format_sensitivity": "upstream does not score it",
    "multi_turn": "stateful multi-turn environment; future work, not this adapter",
    "live_relevance": "outside the pinned scope; it is the inverse check of live_irrelevance",
}

MODES = ("native", "prompt")
OUTCOMES = ("passed", "wrong_arguments", "wrong_function", "hallucinated_parameter", "missing_required",
            "unexpected_call", "missing_call", "invalid_output", "timeout", "environment_error")
RUNNABLE_STATUSES = ("completed", "invalid_output", "timeout", "environment_error")

TASK_ID_PATTERN = re.compile(r"^bfcl/(native|prompt)/(?P<record>[A-Za-z0-9][A-Za-z0-9._-]*)$")
MAX_EXPECTED_CALLS = 8
"""A ground-truth variant with more calls than this is a dataset error: the matcher is exponential."""
MIN_ITEM_SECONDS = 5.0
HOLDOUT_SHARE = 64  # of 256: a deterministic quarter of the pinned items.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

_UPSTREAM_TYPES = {"dict": "object", "object": "object", "array": "array", "tuple": "array",
                   "list": "array", "float": "number", "number": "number", "integer": "integer",
                   "int": "integer", "string": "string", "str": "string", "boolean": "boolean",
                   "bool": "boolean", "null": "null", "any": "any"}
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_REASON_RANK = {"hallucinated_parameter": 0, "missing_required": 1, "wrong_arguments": 2,
                "wrong_function": 3, "unexpected_call": 4, "missing_call": 5}

PROMPT_MODE_INSTRUCTIONS = (
    "You are a function-calling assistant. You are given the JSON schemas of the functions you may call.\n"
    "Reply with ONLY a JSON array of calls and nothing else, in this exact form:\n"
    '[{"name": "<function name>", "arguments": {"<parameter>": <value>}}]\n'
    "Use the JSON types the schema declares: true/false for booleans, numbers without quotes, arrays in "
    "the order given.\n"
    "Omit optional parameters you were not given values for. Do not invent parameters.\n"
    "If no available function applies to the request, reply with exactly [] and nothing else.\n"
    "Available functions:\n"
)


# --------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One emitted function call, already decoded from the wire."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ExpectedCall:
    """One ground-truth call: each parameter maps to the list of acceptable values (upstream shape)."""

    name: str
    arguments: dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class BfclRecord:
    record_id: str
    category: str
    messages: tuple[dict[str, str], ...]
    functions: tuple[dict[str, Any], ...]
    """Function schemas converted from upstream's type language to JSON Schema."""
    variants: tuple[tuple[ExpectedCall, ...], ...]
    """Acceptable answers. Each variant is a complete call list; ``()`` means "no call is correct"."""
    identity: str

    @property
    def schemas(self) -> dict[str, dict[str, Any]]:
        return {item["name"]: item.get("parameters", {}) for item in self.functions}

    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": json.loads(json.dumps(item))} for item in self.functions]


@dataclass(frozen=True)
class MatchResult:
    passed: bool
    outcome_status: str
    matched_answer: int | None
    detail: str


@dataclass(frozen=True)
class DatasetBundle:
    root: Path
    manifest: dict[str, Any]
    records: tuple[BfclRecord, ...]
    subset_label: str

    @property
    def by_id(self) -> dict[str, BfclRecord]:
        return {record.record_id: record for record in self.records}


# --------------------------------------------------------------------------------------------------
# The AST matcher
# --------------------------------------------------------------------------------------------------


def normalize_string(value: str) -> str:
    """Upstream-style string leniency: case, punctuation and whitespace are all removed.

    Mirrors upstream's ``standardize_string``, which strips separators outright rather than collapsing
    them: ``"New York"``, ``"new-york"`` and ``"newyork"`` are the same value. It is deliberately
    narrower than a semantic match - ``"NYC"`` is still wrong. Genuine paraphrases have to be listed
    as acceptable values in the ground truth.
    """
    if type(value) is not str:
        raise TypeError("normalize_string expects a string")
    return "".join(value.lower().translate(_PUNCTUATION).split())


def value_matches(actual: Any, expected: Any, schema: Mapping[str, Any] | None = None) -> bool:
    """Typed value comparison. The declared schema type tightens numbers; it never loosens a type.

    Rules, all exercised by the unit tests:

    * ``bool`` only ever matches ``bool``. The string ``"true"`` fails against ``True``, and ``1`` fails
      against ``True`` (Python's ``bool``/``int`` identity is explicitly not honoured).
    * Numbers match numerically, but a schema-declared ``integer`` parameter rejects a float, and no
      number ever matches a string.
    * Strings match after :func:`normalize_string`.
    * Lists are order-sensitive and length-sensitive; elements use the schema's ``items``.
    * Dicts compare by key set with order ignored, values recursively via ``properties``.
    * ``None`` matches only ``None``.
    """
    declared = (schema or {}).get("type")
    if type(expected) is bool or type(actual) is bool:
        return type(actual) is bool and type(expected) is bool and actual is expected
    if expected is None or actual is None:
        return expected is None and actual is None
    if type(expected) is str:
        return type(actual) is str and normalize_string(actual) == normalize_string(expected)
    if type(expected) in (int, float):
        if type(actual) not in (int, float):
            return False
        if declared == "integer" and type(actual) is not int:
            return False
        if type(actual) is float and not math.isfinite(actual):
            return False
        return actual == expected
    if type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            return False
        items = (schema or {}).get("items")
        return all(value_matches(a, e, items) for a, e in zip(actual, expected))
    if type(expected) is dict:
        if type(actual) is not dict or actual.keys() != expected.keys():
            return False
        properties = (schema or {}).get("properties", {})
        return all(value_matches(actual[key], expected[key], properties.get(key)) for key in expected)
    return False


def _omittable(options: Sequence[Any]) -> bool:
    """Upstream marks an optional parameter by listing the empty string as an acceptable value."""
    return any(option == "" for option in options)


def match_call(call: Call, expected: ExpectedCall, schema: Mapping[str, Any]) -> str | None:
    """Score one emitted call against one ground-truth call. Returns ``None`` on a match, else a reason.

    Order of judgement is deliberate: a parameter the schema never declared is a hallucination, a
    declared-but-missing required parameter is a protocol failure, and only then are values compared.
    """
    if call.name != expected.name:
        return "wrong_function"
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    required = schema.get("required", []) if isinstance(schema, Mapping) else []
    for key in call.arguments:
        if key not in properties:
            return "hallucinated_parameter"
    for key in required:
        if key not in call.arguments:
            return "missing_required"
    for key in call.arguments:
        if key not in expected.arguments:
            return "wrong_arguments"  # a declared parameter the acceptable answer does not allow here
    for key, options in expected.arguments.items():
        if key not in call.arguments:
            if key in required:
                return "missing_required"
            if _omittable(options):
                continue
            return "wrong_arguments"
        if not any(value_matches(call.arguments[key], option, properties.get(key)) for option in options):
            return "wrong_arguments"
    return None


def _worse(current: str | None, candidate: str | None) -> str | None:
    """Keep the most specific diagnosis seen so far (lowest rank wins)."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if _REASON_RANK[candidate] < _REASON_RANK[current] else current


def match_variant(calls: Sequence[Call], variant: Sequence[ExpectedCall],
                  schemas: Mapping[str, Mapping[str, Any]]) -> str | None:
    """All-or-nothing, order-independent matching of a whole call list against one acceptable answer.

    Every emitted call must match a distinct ground-truth call and every ground-truth call must be
    matched. One unmatched call on either side fails the case; there is no partial credit. Matching is
    exhaustive rather than greedy so that a bad first pairing cannot fail a case that is satisfiable.
    """
    if len(calls) != len(variant):
        return "missing_call" if len(calls) < len(variant) else "unexpected_call"
    if not variant:
        return None
    if len(variant) > MAX_EXPECTED_CALLS:
        raise ValueError(f"ground truth has {len(variant)} calls; the cap is {MAX_EXPECTED_CALLS}")

    def assign(available: tuple[Call, ...], wanted: tuple[ExpectedCall, ...]) -> tuple[bool, str | None]:
        if not wanted:
            return True, None
        reason: str | None = None
        for index, call in enumerate(available):
            failure = match_call(call, wanted[0], schemas.get(wanted[0].name, {}))
            if failure is None:
                ok, deeper = assign(available[:index] + available[index + 1:], wanted[1:])
                if ok:
                    return True, None
                reason = _worse(reason, deeper)
            else:
                reason = _worse(reason, failure)
        return False, reason

    matched, failure = assign(tuple(calls), tuple(variant))
    return None if matched else (failure or "wrong_arguments")


def score_calls(calls: Sequence[Call], record: BfclRecord) -> MatchResult:
    """Score emitted calls against every acceptable answer; the first variant that matches wins.

    ``record.variants == ((),)`` is the irrelevance case: passing means emitting no call at all.
    """
    schemas = record.schemas
    reason: str | None = None
    for index, variant in enumerate(record.variants):
        failure = match_variant(calls, variant, schemas)
        if failure is None:
            return MatchResult(True, "passed", index, "")
        reason = _worse(reason, failure)
    outcome = reason or "wrong_arguments"
    return MatchResult(False, outcome, None,
                       f"no acceptable answer matched ({len(record.variants)} checked): {outcome}")


# --------------------------------------------------------------------------------------------------
# Pinned dataset
# --------------------------------------------------------------------------------------------------


def json_schema(node: Any, *, path: str = "parameters") -> Any:
    """Convert upstream's Python-flavoured type language into JSON Schema for the OpenAI ``tools`` field.

    Upstream writes ``{"type": "dict", "properties": {"n": {"type": "float"}}}``; llama.cpp needs
    ``object``/``number``. ``any`` drops the ``type`` keyword rather than guessing one.
    """
    if type(node) is not dict:
        raise ValueError(f"{path}: function schema nodes must be objects")
    converted: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type":
            if type(value) is not str or value not in _UPSTREAM_TYPES:
                raise ValueError(f"{path}: unsupported parameter type {value!r}")
            mapped = _UPSTREAM_TYPES[value]
            if mapped != "any":
                converted["type"] = mapped
        elif key == "properties":
            if type(value) is not dict:
                raise ValueError(f"{path}.properties must be an object")
            converted["properties"] = {name: json_schema(item, path=f"{path}.{name}")
                                       for name, item in value.items()}
        elif key == "items":
            converted["items"] = json_schema(value, path=f"{path}.items")
        else:
            converted[key] = value
    return converted


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _confined(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} or ":" in part
                                                   for part in pure.parts):
        raise BenchmarkUnavailable(f"dataset path must be relative and confined: {relative!r}")
    path = (root / Path(*pure.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise BenchmarkUnavailable(f"dataset path escapes the bundle: {relative!r}")
    return path


def _read_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    """Read one pinned JSON-Lines file (upstream's format) or a JSON array, after verifying its bytes."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BenchmarkUnavailable(f"pinned dataset file is unreadable: {path.name}: {exc}") from exc
    actual = _digest(data)
    if actual != expected_sha256:
        raise BenchmarkUnavailable(f"{path.name} sha256 {actual} differs from the manifest pin "
                                   f"{expected_sha256}")
    text = data.decode("utf-8")
    stripped = text.lstrip()
    try:
        if stripped.startswith("["):
            rows = strict_json_loads(text)
        else:
            rows = [strict_json_loads(line) for line in text.splitlines() if line.strip()]
    except (ValueError, TypeError, RecursionError) as exc:
        raise BenchmarkUnavailable(f"{path.name} is not strict JSON: {exc}") from exc
    if not all(type(row) is dict for row in rows):
        raise BenchmarkUnavailable(f"{path.name} must contain JSON objects")
    return rows


def _variants(raw: Any, record_id: str) -> tuple[tuple[ExpectedCall, ...], ...]:
    """Normalize the ground truth into acceptable-answer variants.

    Accepted shapes (both are unambiguous because a call is an object and a variant is an array):
    upstream's ``[{"fn": {"param": [values]}}, ...]`` (one variant), and ``[[...], [...]]`` (several
    acceptable answers). ``[]`` and ``[[]]`` both mean "the correct answer is no call at all".
    """
    if type(raw) is not list:
        raise BenchmarkUnavailable(f"{record_id}: ground truth must be an array")
    if not raw:
        return ((),)
    groups = raw if all(type(item) is list for item in raw) else [raw]
    variants: list[tuple[ExpectedCall, ...]] = []
    for group in groups:
        if type(group) is not list:
            raise BenchmarkUnavailable(f"{record_id}: mixed ground-truth shapes")
        calls: list[ExpectedCall] = []
        for entry in group:
            if type(entry) is not dict or len(entry) != 1:
                raise BenchmarkUnavailable(f"{record_id}: each ground-truth call is a single-key object")
            name, arguments = next(iter(entry.items()))
            if type(arguments) is not dict:
                raise BenchmarkUnavailable(f"{record_id}: ground-truth arguments must be an object")
            options = {}
            for key, value in arguments.items():
                if type(value) is not list or not value:
                    raise BenchmarkUnavailable(
                        f"{record_id}: {name}.{key} must list at least one acceptable value")
                options[key] = tuple(value)
            calls.append(ExpectedCall(name, options))
        if len(calls) > MAX_EXPECTED_CALLS:
            raise BenchmarkUnavailable(f"{record_id}: {len(calls)} ground-truth calls exceeds the cap")
        variants.append(tuple(calls))
    return tuple(variants)


def _messages(raw: Any, record_id: str) -> tuple[dict[str, str], ...]:
    """Upstream stores ``question`` as a list of turns; the AST categories have exactly one."""
    if type(raw) is not list or not raw:
        raise BenchmarkUnavailable(f"{record_id}: question must be a non-empty array")
    turns = raw[0] if type(raw[0]) is list else raw
    if type(turns) is not list or not turns:
        raise BenchmarkUnavailable(f"{record_id}: question turn must be a non-empty array")
    if len(raw) > 1 and type(raw[0]) is list:
        raise BenchmarkUnavailable(f"{record_id}: multi-turn records are out of scope for this adapter")
    messages = []
    for item in turns:
        if type(item) is not dict or type(item.get("role")) is not str \
                or type(item.get("content")) is not str:
            raise BenchmarkUnavailable(f"{record_id}: every message needs a string role and content")
        if item["role"] not in {"system", "user", "assistant"}:
            raise BenchmarkUnavailable(f"{record_id}: unsupported message role {item['role']!r}")
        messages.append({"role": item["role"], "content": item["content"]})
    return tuple(messages)


def _build_record(category: str, raw: Mapping[str, Any], answer: Mapping[str, Any]) -> BfclRecord:
    record_id = raw.get("id")
    if type(record_id) is not str or not TASK_ID_PATTERN.match(f"bfcl/native/{record_id}"):
        raise BenchmarkUnavailable(f"{category}: record id {record_id!r} is missing or unusable")
    functions_raw = raw.get("function")
    if type(functions_raw) is not list or not functions_raw:
        raise BenchmarkUnavailable(f"{record_id}: function must be a non-empty array")
    functions = []
    for item in functions_raw:
        if type(item) is not dict or type(item.get("name")) is not str:
            raise BenchmarkUnavailable(f"{record_id}: every function needs a name")
        converted = {"name": item["name"], "description": str(item.get("description", "")),
                     "parameters": json_schema(item.get("parameters", {"type": "dict", "properties": {}}))}
        functions.append(converted)
    names = [item["name"] for item in functions]
    if len(set(names)) != len(names):
        raise BenchmarkUnavailable(f"{record_id}: duplicate function names")
    ground_truth = answer.get("ground_truth", answer.get("possible_answer"))
    if ground_truth is None:
        raise BenchmarkUnavailable(f"{record_id}: answer row has neither ground_truth nor possible_answer")
    variants = _variants(ground_truth, record_id)
    for variant in variants:
        for call in variant:
            if call.name not in names:
                raise BenchmarkUnavailable(f"{record_id}: ground truth names unknown function {call.name}")
    if category.endswith("irrelevance") and any(variant for variant in variants):
        raise BenchmarkUnavailable(f"{record_id}: an irrelevance record must expect no call")
    messages = _messages(raw.get("question"), record_id)
    identity = _digest(_canonical({"id": record_id, "category": category, "question": list(messages),
                                   "function": functions, "ground_truth": ground_truth}).encode("utf-8"))
    return BfclRecord(record_id, category, messages, tuple(functions), variants, identity)


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    try:
        manifest = strict_json_loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkUnavailable(f"no pinned BFCL bundle at {root}: {exc}") from exc
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise BenchmarkUnavailable(f"{path} is not strict JSON: {exc}") from exc
    if type(manifest) is not dict:
        raise BenchmarkUnavailable("manifest.json must be a JSON object")
    if manifest.get("adapter_revision") != REVISION:
        raise BenchmarkUnavailable(f"bundle was built for {manifest.get('adapter_revision')!r}; this "
                                   f"adapter is pinned to {REVISION!r}")
    if manifest.get("upstream_package") != UPSTREAM_PACKAGE:
        raise BenchmarkUnavailable(f"bundle must come from {UPSTREAM_PACKAGE}, not "
                                   f"{manifest.get('upstream_package')!r}")
    version = manifest.get("upstream_version")
    if type(version) is not str or not version.strip():
        raise BenchmarkUnavailable("manifest must record the exact upstream_version that was installed")
    commit = manifest.get("source_commit")
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise BenchmarkUnavailable("manifest must record the 40-hex gorilla source_commit")
    categories = manifest.get("categories")
    if type(categories) is not dict or not categories:
        raise BenchmarkUnavailable("manifest must list at least one baked category")
    unsupported = sorted(set(categories) - set(AST_CATEGORIES))
    if unsupported:
        reasons = "; ".join(f"{name}: {EXCLUDED_CATEGORIES.get(name, 'not an AST category')}"
                            for name in unsupported)
        raise BenchmarkUnavailable(f"bundle contains out-of-scope categories ({reasons})")
    return manifest


def load_bundle(dataset_root: str | Path, *, items_per_category: int | None = None) -> DatasetBundle:
    """Load and verify the pinned bundle. Raises :class:`BenchmarkUnavailable`; never touches the network.

    ``items_per_category`` takes the first N records of each category *in pinned file order*, which is a
    stable, documented subset - and a subset is labelled as such on every row, never as a full BFCL run.
    """
    if items_per_category is not None and (type(items_per_category) is not int or items_per_category < 1):
        raise BenchmarkUnavailable("items_per_category must be a positive integer when set")
    root = (Path(dataset_root) / BENCHMARK_ID / REVISION)
    manifest = _manifest(root)
    records: list[BfclRecord] = []
    for category in AST_CATEGORIES:
        entry = manifest["categories"].get(category)
        if entry is None:
            continue
        if type(entry) is not dict:
            raise BenchmarkUnavailable(f"{category}: manifest entry must be an object")
        digests = entry.get("sha256")
        if type(digests) is not dict or any(type(digests.get(key)) is not str
                                            for key in ("records", "answers")):
            raise BenchmarkUnavailable(f"{category}: manifest must pin records and answers sha256")
        if any(type(entry.get(key)) is not str for key in ("records", "answers")):
            raise BenchmarkUnavailable(f"{category}: manifest must name the records and answers files")
        rows = _read_rows(_confined(root, entry["records"]), digests["records"])
        answers = _read_rows(_confined(root, entry["answers"]), digests["answers"])
        by_id: dict[str, Mapping[str, Any]] = {}
        for answer in answers:
            key = answer.get("id")
            if type(key) is not str or key in by_id:
                raise BenchmarkUnavailable(f"{category}: answer rows need unique string ids")
            by_id[key] = answer
        selected = rows if items_per_category is None else rows[:items_per_category]
        for row in selected:
            answer = by_id.get(row.get("id"))
            if answer is None:
                raise BenchmarkUnavailable(f"{category}: no ground truth for record {row.get('id')!r}")
            records.append(_build_record(category, row, answer))
        declared = entry.get("count")
        if type(declared) is int and declared != len(rows):
            raise BenchmarkUnavailable(f"{category}: manifest declares {declared} records, file has "
                                       f"{len(rows)}")
    if not records:
        raise BenchmarkUnavailable("the pinned bundle declared categories but contained no records")
    seen = [record.record_id for record in records]
    if len(set(seen)) != len(seen):
        raise BenchmarkUnavailable("record ids are not unique across the baked categories")
    label = "full" if items_per_category is None else f"first-{items_per_category}-per-category"
    return DatasetBundle(root, manifest, tuple(records), label)


def split_of(record_id: str) -> str:
    """Deterministic, disjoint item partition keyed by the pin and the record id.

    This is an item partition, not an unseen holdout: BFCL is public data that any model may have been
    trained on. The registry additionally forbids a non-retrieval suite from claiming ``holdout``.
    """
    marker = hashlib.sha256(f"{REVISION}:{record_id}".encode("utf-8")).digest()[0]
    return "holdout" if marker < HOLDOUT_SHARE else "development"


def task_id_for(mode: str, record_id: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return f"bfcl/{mode}/{record_id}"


def parse_task_id(task_id: str) -> tuple[str, str] | None:
    match = TASK_ID_PATTERN.match(task_id) if type(task_id) is str else None
    return (match.group(1), match.group("record")) if match else None


# --------------------------------------------------------------------------------------------------
# Requests and responses
# --------------------------------------------------------------------------------------------------


def prompt_mode_messages(record: BfclRecord) -> list[dict[str, str]]:
    """``prompt`` mode: the same schemas, rendered into a system prompt, answered as JSON.

    This is our contract, not upstream's Python-call syntax, so prompt-mode numbers are comparable
    across our own candidates but are not BFCL leaderboard scores.
    """
    schemas = json.dumps(list(record.functions), ensure_ascii=False, indent=2, sort_keys=False,
                         allow_nan=False)
    messages = [{"role": "system", "content": PROMPT_MODE_INSTRUCTIONS + schemas}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in record.messages)
    return messages


def build_request(context: BenchmarkContext, record: BfclRecord, mode: str) -> dict[str, Any]:
    """One chat-completions payload. ``native`` advertises tools; ``prompt`` deliberately does not."""
    generation = context.generation
    if mode == "native":
        messages = [dict(item) for item in record.messages]
    else:
        messages = prompt_mode_messages(record)
    payload: dict[str, Any] = {
        "model": context.model_alias, "messages": messages, "stream": False,
        "temperature": float(generation.temperature), "top_p": float(generation.top_p),
        "seed": int(generation.seed), "max_tokens": int(generation.max_output_tokens),
    }
    top_k = getattr(generation, "top_k", None)
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if mode == "native":
        payload["tools"] = record.openai_tools()
        payload["tool_choice"] = "auto"  # irrelevance requires the model to be free to answer in text
    return payload


def parse_native_output(message: Any) -> tuple[list[Call], list[str]]:
    """Read ``message.tool_calls`` through the strict shared parser (no repair, no coercion)."""
    parsed, errors = parse_tool_response(message)
    return [Call(item.name, item.arguments) for item in parsed], list(errors)


def parse_prompt_output(text: Any) -> tuple[list[Call], list[str]]:
    """Parse the JSON array contract from assistant text. Anything else is ``invalid_output``."""
    if type(text) is not str:
        return [], ["assistant content must be a string in prompt mode"]
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = stripped.find("["), stripped.rfind("]")
    if start < 0 or end < start:
        return [], ["no JSON array of calls in the assistant text"]
    try:
        parsed = strict_json_loads(stripped[start:end + 1])
    except (ValueError, TypeError, RecursionError) as exc:
        return [], [f"call array is not strict JSON: {exc}"]
    if type(parsed) is not list:
        return [], ["call array must be a JSON array"]
    calls, errors = [], []
    for index, item in enumerate(parsed):
        if type(item) is not dict or type(item.get("name")) is not str:
            errors.append(f"call[{index}]: needs a string name")
            continue
        arguments = item.get("arguments", {})
        if type(arguments) is not dict:
            errors.append(f"call[{index}]: arguments must be an object")
            continue
        if any(type(key) is not str for key in arguments):
            errors.append(f"call[{index}]: argument keys must be strings")
            continue
        calls.append(Call(item["name"], arguments))
    return (calls, errors) if not errors else ([], errors)


class ChatTransport(Protocol):
    """Minimal seam: returns the response body *as bytes* so evidence is stored before it is parsed."""

    def post_chat(self, payload: Mapping[str, Any], *, timeout: float) -> bytes: ...


class UrllibChatTransport:
    """Stdlib POST to ``<base_url>/chat/completions``. The constructor performs no network activity."""

    def __init__(self, base_url: str, *, max_response_bytes: int = MAX_RESPONSE_BYTES) -> None:
        from urllib.parse import urlsplit
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.max_response_bytes = max_response_bytes

    def post_chat(self, payload: Mapping[str, Any], *, timeout: float) -> bytes:
        from urllib.error import HTTPError, URLError
        from urllib.request import HTTPRedirectHandler, Request, build_opener

        class _NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, *args: Any, **kwargs: Any) -> None:
                return None

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(self.endpoint, data=body, method="POST",
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
                data = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:  # bodies can carry prompt text; the status is enough for a row
            raise RuntimeError(f"inference returned HTTP {exc.code}") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("inference request timed out") from exc
            raise RuntimeError(f"inference request failed: {type(exc.reason).__name__}") from exc
        if len(data) > self.max_response_bytes:
            raise RuntimeError("inference response exceeded the adapter's byte limit")
        return data


# --------------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------------


class BfclAdapter:
    """``BenchmarkAdapter`` for the pinned, offline BFCL AST categories.

    ``transport`` is the injection seam used by the tests; when it is ``None`` the adapter builds a
    stdlib HTTP client from ``context.base_url`` lazily, inside :meth:`run` only, so importing,
    preflighting or enumerating tasks can never touch a socket. ``context.options['transport']``
    overrides both.
    """

    benchmark_id = BENCHMARK_ID
    revision = REVISION
    category = CATEGORY

    def __init__(self, transport: ChatTransport | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 request_timeout_seconds: float = 120.0) -> None:
        self._transport = transport
        self._clock = clock
        self._request_timeout = float(request_timeout_seconds)
        self._cache: dict[tuple[str, int | None], DatasetBundle] = {}

    # -- preflight ---------------------------------------------------------------------------------

    def bundle(self, context: BenchmarkContext) -> DatasetBundle:
        limit = (context.options or {}).get("items_per_category")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise BenchmarkUnavailable("items_per_category must be a positive integer when set")
        key = (str(context.dataset_root), limit)
        if key not in self._cache:
            self._cache[key] = load_bundle(context.dataset_root, items_per_category=limit)
        return self._cache[key]

    def available(self, context: BenchmarkContext) -> tuple[bool, str]:
        try:
            bundle = self.bundle(context)
            modes = self.modes(context)
            declared = self.task_ids(context)
        except Exception as exc:  # preflight never raises; the reason travels to the caller instead
            return False, f"{type(exc).__name__}: {exc}"
        if not declared:
            return False, (f"no {context.split} items in the pinned bundle for modes {list(modes)}")
        return True, (f"{len(declared)} tasks from {UPSTREAM_PACKAGE} "
                      f"{bundle.manifest['upstream_version']} @ {bundle.manifest['source_commit'][:12]} "
                      f"({bundle.subset_label}, modes {list(modes)})")

    def modes(self, context: BenchmarkContext) -> tuple[str, ...]:
        raw = (context.options or {}).get("modes", ("native",))
        if type(raw) is str:
            raw = (raw,)
        modes = tuple(raw)
        if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
            raise BenchmarkUnavailable(f"modes must be a unique non-empty subset of {MODES}")
        return modes

    def task_ids(self, context: BenchmarkContext) -> tuple[str, ...]:
        bundle = self.bundle(context)
        modes = self.modes(context)
        return tuple(task_id_for(mode, record.record_id) for mode in modes
                     for record in bundle.records if split_of(record.record_id) == context.split)

    # -- execution ---------------------------------------------------------------------------------

    def run(self, context: BenchmarkContext) -> list[dict[str, Any]]:
        declared = self._declared(context)
        gap = dict(suite=BENCHMARK_ID, revision=REVISION, category=CATEGORY, split=context.split)
        if context.artifacts is None:
            return _label(missing_rows(declared, [], **gap, status="environment_error",
                                       reason="no artifact writer: raw responses could not be persisted"))
        try:
            bundle = self.bundle(context)
        except Exception as exc:
            return _label(missing_rows(declared, [], **gap, status="environment_error",
                                       reason=f"pinned dataset unavailable: {type(exc).__name__}: {exc}"))
        records = bundle.by_id
        transport = (context.options or {}).get("transport") or self._transport
        if transport is None:
            try:
                transport = UrllibChatTransport(context.base_url)
            except ValueError as exc:
                return _label(missing_rows(declared, [], **gap, status="environment_error",
                                           reason=f"unusable base_url: {exc}"))
        rows: list[dict[str, Any]] = []
        for task_id in declared:
            left = self._remaining(context)
            if left <= MIN_ITEM_SECONDS:
                rows.extend(missing_rows(declared, rows, **gap, status="timeout",
                                         reason="wall budget exhausted before this item was sent"))
                break
            parsed = parse_task_id(task_id)
            record = records.get(parsed[1]) if parsed else None
            if parsed is None or record is None:
                rows.append(self._gap_row(context, task_id, bundle,
                                          f"task id is not in the pinned {REVISION} bundle"))
                continue
            try:
                rows.append(self._run_item(context, task_id, parsed[0], record, bundle, transport, left))
            except BenchmarkAborted as exc:
                exc.rows = rows + missing_rows(declared, rows, **gap, status="environment_error",
                                               reason=f"run aborted: {exc}")
                raise
            except (OperationForbidden, KeyboardInterrupt, SystemExit):
                raise  # a denied permission gate is a campaign-level decision, not a scored 0
            except Exception as exc:
                rows.append(self._gap_row(context, task_id, bundle,
                                          f"adapter failure: {type(exc).__name__}: {exc}", record=record))
        rows.extend(missing_rows(declared, rows, **gap, status="environment_error",
                                 reason="the adapter produced no row for this declared task"))
        order = {task_id: index for index, task_id in enumerate(declared)}
        rows.sort(key=lambda row: order.get(row.get("task_id"), len(order)))
        return _label(rows)

    # -- internals ---------------------------------------------------------------------------------

    def _declared(self, context: BenchmarkContext) -> tuple[str, ...]:
        """Exactly the tasks the caller asked for, de-duplicated; else everything we can run."""
        requested = tuple(context.task_ids or ())
        if not requested:
            try:
                return self.task_ids(context)
            except Exception:
                return ()
        seen: dict[str, None] = {}
        for task_id in requested:
            seen.setdefault(task_id, None)
        return tuple(seen)

    def _remaining(self, context: BenchmarkContext) -> float:
        try:
            left = float(context.remaining_seconds())
        except TimeoutError:
            return 0.0
        return left if math.isfinite(left) else 0.0

    def _base(self, context: BenchmarkContext, task_id: str, bundle: DatasetBundle,
              record: BfclRecord | None) -> dict[str, Any]:
        parsed = parse_task_id(task_id)
        manifest = bundle.manifest
        row = {"task_id": task_id, "suite": BENCHMARK_ID, "suite_revision": REVISION, "category": CATEGORY,
               "split": context.split, "synthetic": False, "mode": parsed[0] if parsed else None,
               "split_kind": "deterministic-item-partition", "subset_label": bundle.subset_label,
               "upstream_package": UPSTREAM_PACKAGE, "upstream_version": manifest["upstream_version"],
               "source_commit": manifest["source_commit"], "scorer": "llmbench-bfcl-ast-v1"}
        if record is not None:
            row.update({"record_id": record.record_id, "bfcl_category": record.category,
                        "fixture_hash": record.identity, "expected_calls": len(record.variants[0]),
                        "acceptable_answers": len(record.variants)})
        return row

    def _gap_row(self, context: BenchmarkContext, task_id: str, bundle: DatasetBundle, reason: str, *,
                 record: BfclRecord | None = None) -> dict[str, Any]:
        return _row(self._base(context, task_id, bundle, record), status="environment_error",
                    outcome_status="environment_error", model_evaluated=False, error=reason)

    def _run_item(self, context: BenchmarkContext, task_id: str, mode: str, record: BfclRecord,
                  bundle: DatasetBundle, transport: ChatTransport, left: float) -> dict[str, Any]:
        base = self._base(context, task_id, bundle, record)
        payload = build_request(context, record, mode)
        base["prompt_sha256"] = _digest(_canonical(payload).encode("utf-8"))
        if context.session_lock is not None:
            context.session_lock.check("inference", RunMode.LIVE)
        started = self._clock()
        try:
            raw = transport.post_chat(payload, timeout=max(1.0, min(self._request_timeout, left)))
        except TimeoutError as exc:
            return _row(base, status="timeout", outcome_status="timeout", model_evaluated=False,
                        error=f"request timed out: {exc}", elapsed_seconds=self._elapsed(started))
        except Exception as exc:
            return _row(base, status="environment_error", outcome_status="environment_error",
                        model_evaluated=False, error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=self._elapsed(started))
        base["elapsed_seconds"] = self._elapsed(started)
        if type(raw) is not bytes:
            raise BenchmarkAborted("transport returned a parsed object; raw bytes are required evidence")
        # Evidence first: nothing below this line may run before the bytes are on disk.
        relative = f"bfcl/{mode}/{_slug(record.record_id)}.json"
        try:
            context.artifacts.write(relative, raw)
        except Exception as exc:
            raise BenchmarkAborted(f"BFCL response evidence could not be persisted: "
                                   f"{type(exc).__name__}: {exc}") from exc
        base.update({"response_artifact": relative, "response_sha256": _digest(raw),
                     "response_bytes": len(raw)})
        try:
            response = strict_json_loads(raw.decode("utf-8"))
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError) as exc:
            return _row(base, status="invalid_output", outcome_status="invalid_output",
                        model_evaluated=True, error=f"response is not strict JSON: {exc}")
        if type(response) is not dict or "error" in response:
            return _row(base, status="environment_error", outcome_status="environment_error",
                        model_evaluated=False, error="inference returned an error envelope")
        served = response.get("model")
        if served != context.model_alias:
            return _row(base, status="environment_error", outcome_status="environment_error",
                        model_evaluated=False,
                        error=f"served model {served!r} is not the candidate {context.model_alias!r}")
        choices = response.get("choices")
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            return _row(base, status="invalid_output", outcome_status="invalid_output",
                        model_evaluated=True, error="response must carry exactly one choice")
        message = choices[0].get("message")
        base["finish_reason"] = choices[0].get("finish_reason")
        calls, errors = (parse_native_output(message) if mode == "native"
                         else parse_prompt_output(message.get("content")
                                                  if type(message) is dict else None))
        if errors:
            return _row(base, status="invalid_output", outcome_status="invalid_output",
                        model_evaluated=True, emitted_calls=0, errors=errors,
                        error="; ".join(errors)[:400])
        result = score_calls(calls, record)
        return _row(base, status="completed", outcome_status=result.outcome_status,
                    model_evaluated=True, passed=result.passed, score=float(result.passed),
                    matched_answer=result.matched_answer, emitted_calls=len(calls),
                    emitted_functions=[call.name for call in calls],
                    error=None if result.passed else result.detail)

    def _elapsed(self, started: float) -> float:
        return max(0.0, self._clock() - started)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", value)[:120]


def _row(base: Mapping[str, Any], *, status: str, outcome_status: str, model_evaluated: bool,
         passed: bool = False, score: float = 0.0, **extra: Any) -> dict[str, Any]:
    """Build one sample row and refuse to emit one that breaks the shared contract."""
    if status not in RUNNABLE_STATUSES or outcome_status not in OUTCOMES:
        raise ValueError(f"unknown status pair: {status}/{outcome_status}")
    if status != "completed" and (passed or score):
        raise ValueError("only a completed item may carry a nonzero score")
    always = {"error", "matched_answer"}  # explicit nulls: "no error" and "no acceptable answer matched"
    row = {**base, "status": status, "outcome_status": outcome_status, "passed": bool(passed),
           "score": float(score), "model_evaluated": bool(model_evaluated),
           **{key: value for key, value in extra.items() if value is not None or key in always}}
    absent = [key for key in REQUIRED_SAMPLE_KEYS if key not in row]
    if absent:
        raise ValueError(f"row is missing required keys: {absent}")
    return row


def _label(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give helper-generated rows (``missing_rows``) the same mode/record labelling as scored rows."""
    labelled = []
    for row in rows:
        parsed = parse_task_id(row.get("task_id", ""))
        if parsed and row.get("mode") is None:
            row = {**row, "mode": parsed[0], "record_id": parsed[1]}
        labelled.append(row)
    return labelled
