"""BFCL adapter: AST matcher rules, mode axis, and denominator/evidence behaviour.

Everything here is offline. The adapter is driven through a fake OpenAI-compatible transport; no
Docker, model, GPU or socket is involved, and the only HTTP class touched is checked for its
constructor validation, which performs no network activity.

The dataset fixtures under ``tests/data/bfcl-*`` are hand-made records that mirror the upstream shape
(``question``/``function`` in upstream's type language plus a ``ground_truth``/``possible_answer``
list). They are not redistributed upstream data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llmbench.benchmarks import REQUIRED_SAMPLE_KEYS, BenchmarkAborted, BenchmarkContext
from llmbench.benchmarks import bfcl
from llmbench.config import GenerationSettings
from llmbench.safety import SessionLock

DATA = Path(__file__).parent / "data"
ALIAS = "candidate-model"
FIXTURES = {
    "simple": ("bfcl-simple.json", "bfcl-simple-answer.json"),
    "parallel": ("bfcl-parallel.json", "bfcl-parallel-answer.json"),
    "irrelevance": ("bfcl-irrelevance.json", "bfcl-irrelevance-answer.json"),
    "live_simple": ("bfcl-live-simple.json", "bfcl-live-simple-answer.json"),
}


# ---------------------------------------------------------------------------------------------
# Offline fixtures
# ---------------------------------------------------------------------------------------------


def build_dataset(root: Path, *, categories=tuple(FIXTURES), mutate=None) -> Path:
    """Materialise a pinned bundle from the hand-made fixtures; return the dataset_root to pass in."""
    dataset_root = root / "datasets"
    bundle = dataset_root / bfcl.BENCHMARK_ID / bfcl.REVISION
    manifest = json.loads((DATA / "bfcl-manifest.json").read_text(encoding="utf-8"))
    manifest["categories"] = {name: entry for name, entry in manifest["categories"].items()
                              if name in categories}
    for name, entry in manifest["categories"].items():
        for kind, source in zip(("records", "answers"), FIXTURES[name]):
            target = bundle / entry[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (DATA / source).read_bytes()
            target.write_bytes(data)
            entry["sha256"][kind] = hashlib.sha256(data).hexdigest()
    if mutate is not None:
        mutate(manifest)
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return dataset_root


class RecordingArtifacts:
    """Bounded-writer stand-in: records every byte string handed to it, in order."""

    def __init__(self, fail: bool = False) -> None:
        self.writes: list[tuple[str, bytes]] = []
        self.fail = fail

    def write(self, rel: str, data: bytes) -> None:
        if self.fail:
            raise OSError("artifact allocation exhausted")
        assert type(data) is bytes
        self.writes.append((rel, data))


class FakeTransport:
    """OpenAI-compatible transport seam. Returns bytes, exactly like the real one, or raises."""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.requests: list[dict] = []

    def post_chat(self, payload, *, timeout):
        self.requests.append(json.loads(json.dumps(payload)))
        assert timeout > 0
        result = self.responder(payload, len(self.requests) - 1)
        if isinstance(result, BaseException):
            raise result
        return result if type(result) is bytes else json.dumps(result).encode("utf-8")


def chat_response(*, tool_calls=None, content=None, model=ALIAS, choices=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = [
            {"id": f"call_{index}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for index, (name, arguments) in enumerate(tool_calls)]
    body = {"model": model, "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
    if choices is not None:
        body["choices"] = choices
    return body


def make_context(dataset_root, *, task_ids=(), split="development", options=None, remaining=None,
                 artifacts=None, alias=ALIAS):
    return BenchmarkContext(
        base_url="http://inference:8080/v1", model_alias=alias, task_ids=tuple(task_ids), split=split,
        seed=42, generation=GenerationSettings(max_output_tokens=256, temperature=0, seed=42),
        remaining_seconds=remaining or (lambda: 600.0), artifacts=artifacts or RecordingArtifacts(),
        session_lock=SessionLock(allow_inference=True), dataset_root=str(dataset_root),
        options=dict(options or {}))


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("bfcl-bundle"))


@pytest.fixture(scope="module")
def bundle(dataset_root):
    return bfcl.load_bundle(dataset_root)


def score(bundle, record_id, calls):
    return bfcl.score_calls([bfcl.Call(name, arguments) for name, arguments in calls],
                            bundle.by_id[record_id])


# ---------------------------------------------------------------------------------------------
# Typed value matching
# ---------------------------------------------------------------------------------------------


BOOL = {"type": "boolean"}
INT = {"type": "integer"}
NUMBER = {"type": "number"}
TEXT = {"type": "string"}
STRINGS = {"type": "array", "items": {"type": "string"}}
PAIR = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}}


@pytest.mark.parametrize("actual,expected,schema,ok", [
    (True, True, BOOL, True),
    (False, False, BOOL, True),
    ("true", True, BOOL, False),            # the string "true" is not the boolean true
    ("True", True, BOOL, False),
    (1, True, BOOL, False),                 # Python's bool/int identity is not honoured
    (True, 1, INT, False),
    (0, False, BOOL, False),
    (3, 3, INT, True),
    (3, 3.0, NUMBER, True),                 # numeric equality when the schema allows a number
    (3.0, 3, INT, False),                   # a declared integer rejects a float
    (3.5, 3.5, NUMBER, True),
    (4, 3, INT, False),
    ("5", 5, INT, False),
    (5, "5", TEXT, False),
    ("New York", "new-york", TEXT, True),   # case, punctuation and whitespace are all stripped
    ("New York", "newyork", TEXT, True),
    ("  Nonna's  ", "nonnas", TEXT, True),
    ("NYC", "New York", TEXT, False),       # not a semantic match
    (["a", "b"], ["a", "b"], STRINGS, True),
    (["b", "a"], ["a", "b"], STRINGS, False),   # list order is significant
    (["a"], ["a", "b"], STRINGS, False),
    ({"a": 1, "b": "x"}, {"b": "x", "a": 1}, PAIR, True),   # dict key order is not significant
    ({"a": 1}, {"a": 1, "b": "x"}, PAIR, False),
    ({"a": 1, "b": "x", "c": 2}, {"a": 1, "b": "x"}, PAIR, False),
    ({"a": True}, {"a": 1}, PAIR, False),
    (None, None, TEXT, True),
    (None, "x", TEXT, False),
    ("x", None, TEXT, False),
])
def test_value_matches_table(actual, expected, schema, ok):
    assert bfcl.value_matches(actual, expected, schema) is ok


def test_normalize_string_rejects_non_strings():
    with pytest.raises(TypeError):
        bfcl.normalize_string(5)


# ---------------------------------------------------------------------------------------------
# Single-call AST rules
# ---------------------------------------------------------------------------------------------


def test_exact_call_passes_and_reports_the_matched_answer(bundle):
    result = score(bundle, "simple_0", [("geometry.circle_area", {"radius": 5, "unit": "cm"})])
    assert (result.passed, result.outcome_status, result.matched_answer) == (True, "passed", 0)


def test_paraphrase_in_the_acceptable_list_passes(bundle):
    assert score(bundle, "simple_0",
                 [("geometry.circle_area", {"radius": 5.0, "unit": "Centimeters"})]).passed


def test_optional_parameter_may_be_omitted(bundle):
    assert score(bundle, "simple_0", [("geometry.circle_area", {"radius": 5})]).passed


def test_string_where_a_boolean_is_required_fails(bundle):
    result = score(bundle, "simple_1",
                   [("job.configure", {"name": "deploy", "verbose": "true", "retries": 3})])
    assert (result.passed, result.outcome_status) == (False, "wrong_arguments")


def test_parameter_outside_the_schema_is_a_hallucination(bundle):
    result = score(bundle, "simple_1", [("job.configure", {
        "name": "deploy", "verbose": True, "retries": 3, "priority": "high"})])
    assert result.outcome_status == "hallucinated_parameter"


def test_declared_parameter_outside_the_acceptable_answer_is_wrong_arguments(bundle):
    result = score(bundle, "simple_2", [("restaurant.book", {
        "restaurant": "Nonna's", "party": 4, "tags": ["outdoor", "quiet"], "note": "window seat"})])
    assert result.outcome_status == "wrong_arguments"


def test_missing_required_parameter(bundle):
    result = score(bundle, "simple_1", [("job.configure", {"name": "deploy", "verbose": True})])
    assert result.outcome_status == "missing_required"


def test_list_order_is_significant(bundle):
    good = score(bundle, "simple_2", [("restaurant.book", {
        "restaurant": "Nonna's", "party": 4, "tags": ["outdoor", "quiet"]})])
    bad = score(bundle, "simple_2", [("restaurant.book", {
        "restaurant": "Nonna's", "party": 4, "tags": ["quiet", "outdoor"]})])
    assert good.passed and not bad.passed and bad.outcome_status == "wrong_arguments"


def test_dict_key_order_is_not_significant(bundle):
    result = score(bundle, "simple_3",
                   [("cache.configure", {"options": {"policy": "lru", "memory_gib": 2}})])
    assert result.passed


def test_dict_key_set_is_significant(bundle):
    result = score(bundle, "simple_3", [("cache.configure", {
        "options": {"memory_gib": 2, "policy": "lru", "ttl": 30}})])
    assert result.outcome_status == "wrong_arguments"


def test_wrong_function_name(bundle):
    result = score(bundle, "simple_0", [("geometry.square_area", {"radius": 5})])
    assert result.outcome_status == "wrong_function"


def test_declared_integer_rejects_a_float(bundle):
    assert score(bundle, "live_simple_1-1-0", [("math.round", {"value": 3.5, "digits": 0})]).passed
    result = score(bundle, "live_simple_1-1-0", [("math.round", {"value": 3.5, "digits": 0.0})])
    assert result.outcome_status == "wrong_arguments"


@pytest.mark.parametrize("trip,index", [("one_way", 0), ("return", 1)])
def test_several_acceptable_answers_each_pass(bundle, trip, index):
    result = score(bundle, "live_simple_0-0-0", [("flights.search", {
        "origin": "Paris", "destination": "Tokyo", "date": "2026-03-01", "trip": trip})])
    assert (result.passed, result.matched_answer) == (True, index)


def test_value_outside_every_acceptable_answer_fails(bundle):
    result = score(bundle, "live_simple_0-0-0", [("flights.search", {
        "origin": "Paris", "destination": "Tokyo", "date": "2026-03-01", "trip": "maybe"})])
    assert (result.passed, result.matched_answer) == (False, None)
    assert result.outcome_status == "wrong_arguments"


# ---------------------------------------------------------------------------------------------
# Parallel: all-or-nothing, order independent
# ---------------------------------------------------------------------------------------------


def test_parallel_matches_in_any_order(bundle):
    forward = score(bundle, "parallel_0", [("weather.current", {"city": "Berlin", "metric": True}),
                                           ("weather.current", {"city": "Lisbon", "metric": True})])
    reverse = score(bundle, "parallel_0", [("weather.current", {"city": "Lisbon", "metric": True}),
                                           ("weather.current", {"city": "Berlin", "metric": True})])
    assert forward.passed and reverse.passed


def test_parallel_one_wrong_argument_fails_the_whole_case(bundle):
    result = score(bundle, "parallel_0", [("weather.current", {"city": "Berlin", "metric": True}),
                                          ("weather.current", {"city": "Madrid", "metric": True})])
    assert (result.passed, result.outcome_status) == (False, "wrong_arguments")


def test_parallel_missing_one_call_fails(bundle):
    result = score(bundle, "parallel_0", [("weather.current", {"city": "Berlin", "metric": True})])
    assert result.outcome_status == "missing_call"


def test_parallel_extra_call_fails(bundle):
    result = score(bundle, "parallel_0", [("weather.current", {"city": "Berlin", "metric": True}),
                                          ("weather.current", {"city": "Lisbon", "metric": True}),
                                          ("weather.current", {"city": "Paris", "metric": True})])
    assert result.outcome_status == "unexpected_call"


def test_parallel_multiple_functions_match_across_the_set(bundle):
    result = score(bundle, "parallel_1", [("weather.current", {"city": "Oslo"}),
                                          ("money.convert", {"amount": 10, "source": "USD",
                                                             "target": "EUR"})])
    assert result.passed


def test_parallel_matching_is_exhaustive_not_greedy(bundle):
    """A first pairing that happens to match must not strand a later ground-truth call."""
    result = score(bundle, "parallel_0", [("weather.current", {"city": "Lisbon"}),
                                          ("weather.current", {"city": "Berlin"})])
    assert result.passed


# ---------------------------------------------------------------------------------------------
# Irrelevance: the correct answer is no call at all
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("record_id", ["irrelevance_0", "irrelevance_1"])
def test_irrelevance_passes_when_no_call_is_emitted(bundle, record_id):
    assert score(bundle, record_id, []).passed


@pytest.mark.parametrize("record_id", ["irrelevance_0", "irrelevance_1"])
def test_irrelevance_fails_when_a_call_is_emitted(bundle, record_id):
    record = bundle.by_id[record_id]
    name = record.functions[0]["name"]
    arguments = {key: "x" for key in record.functions[0]["parameters"].get("required", [])}
    assert score(bundle, record_id, [(name, arguments)]).outcome_status == "unexpected_call"


# ---------------------------------------------------------------------------------------------
# Dataset loading and preflight
# ---------------------------------------------------------------------------------------------


def test_available_is_false_when_the_dataset_directory_is_missing(tmp_path):
    ok, reason = bfcl.BfclAdapter().available(make_context(tmp_path / "nothing-here"))
    assert ok is False and "BenchmarkUnavailable" in reason


def test_available_reports_the_pinned_upstream_identity(dataset_root):
    ok, reason = bfcl.BfclAdapter().available(make_context(dataset_root))
    assert ok is True
    assert bfcl.UPSTREAM_PACKAGE in reason and "0.0.0-test-fixture" in reason


def test_available_is_false_when_a_pinned_file_was_tampered_with(tmp_path):
    root = build_dataset(tmp_path, categories=("simple",))
    target = root / bfcl.BENCHMARK_ID / bfcl.REVISION / "data" / "BFCL_v4_simple.json"
    target.write_bytes(target.read_bytes() + b"\n")
    ok, reason = bfcl.BfclAdapter().available(make_context(root))
    assert ok is False and "sha256" in reason


def test_available_is_false_for_a_revision_mismatch(tmp_path):
    def bump(manifest):
        manifest["adapter_revision"] = "bfcl-v9-something-else"

    root = build_dataset(tmp_path, categories=("simple",), mutate=bump)
    ok, reason = bfcl.BfclAdapter().available(make_context(root))
    assert ok is False and bfcl.REVISION in reason


def test_out_of_scope_categories_are_refused(tmp_path):
    def add_web_search(manifest):
        manifest["categories"]["web_search"] = dict(manifest["categories"]["simple"])

    root = build_dataset(tmp_path, categories=("simple",), mutate=add_web_search)
    ok, reason = bfcl.BfclAdapter().available(make_context(root))
    assert ok is False and "SerpAPI" in reason


def test_excluded_categories_are_documented_and_disjoint_from_the_supported_set():
    assert set(bfcl.EXCLUDED_CATEGORIES).isdisjoint(bfcl.AST_CATEGORIES)
    for name in ("web_search", "memory", "format_sensitivity", "multi_turn"):
        assert bfcl.EXCLUDED_CATEGORIES[name]


def test_upstream_type_language_is_converted_to_json_schema(bundle):
    parameters = bundle.by_id["simple_0"].functions[0]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["radius"]["type"] == "number"
    assert bundle.by_id["simple_3"].functions[0]["parameters"]["properties"]["options"]["type"] \
        == "object"


def test_splits_are_deterministic_and_disjoint(bundle):
    development = {r.record_id for r in bundle.records if bfcl.split_of(r.record_id) == "development"}
    holdout = {r.record_id for r in bundle.records if bfcl.split_of(r.record_id) == "holdout"}
    assert development and holdout and development.isdisjoint(holdout)
    assert bfcl.split_of("simple_0") == bfcl.split_of("simple_0")


def test_task_ids_are_stable_ordered_and_mode_scoped(dataset_root):
    adapter = bfcl.BfclAdapter()
    context = make_context(dataset_root, options={"modes": ("native", "prompt")})
    ids = adapter.task_ids(context)
    assert ids == adapter.task_ids(context)
    assert len(set(ids)) == len(ids)
    assert sum(item.startswith("bfcl/native/") for item in ids) == len(ids) // 2
    assert all(bfcl.parse_task_id(item) for item in ids)


def test_items_per_category_yields_a_labelled_subset(dataset_root):
    adapter = bfcl.BfclAdapter()
    context = make_context(dataset_root, options={"items_per_category": 1})
    assert adapter.bundle(context).subset_label == "first-1-per-category"
    assert len(adapter.bundle(context).records) == len(FIXTURES)


# ---------------------------------------------------------------------------------------------
# Execution: modes, rows, evidence, budget
# ---------------------------------------------------------------------------------------------


def run_adapter(dataset_root, responder, **kwargs):
    adapter = bfcl.BfclAdapter(FakeTransport(responder))
    context = make_context(dataset_root, **kwargs)
    return adapter, context, adapter.run(context)


def test_native_mode_sends_tools_and_reads_tool_calls(dataset_root):
    task_id = bfcl.task_id_for("native", "simple_0")
    adapter, context, rows = run_adapter(
        dataset_root,
        lambda payload, index: chat_response(tool_calls=[("geometry.circle_area",
                                                          {"radius": 5, "unit": "cm"})]),
        task_ids=[task_id])
    request = adapter._transport.requests[0]
    assert request["tools"][0]["function"]["name"] == "geometry.circle_area"
    assert request["tool_choice"] == "auto" and request["model"] == ALIAS
    assert all(message["role"] != "system" for message in request["messages"])
    assert request["temperature"] == 0 and request["seed"] == 42
    row = rows[0]
    assert (row["status"], row["outcome_status"], row["passed"], row["score"]) == \
        ("completed", "passed", True, 1.0)
    assert row["mode"] == "native" and row["bfcl_category"] == "simple"
    assert row["suite"] == "bfcl" and row["suite_revision"] == bfcl.REVISION
    assert row["matched_answer"] == 0 and row["record_id"] == "simple_0"
    assert row["upstream_version"] == "0.0.0-test-fixture" and row["subset_label"] == "full"
    assert row["response_sha256"] and row["prompt_sha256"] and row["emitted_calls"] == 1


def test_prompt_mode_renders_the_schema_and_parses_text(dataset_root):
    task_id = bfcl.task_id_for("prompt", "simple_0")
    body = '```json\n[{"name": "geometry.circle_area", "arguments": {"radius": 5, "unit": "cm"}}]\n```'
    adapter, context, rows = run_adapter(
        dataset_root, lambda payload, index: chat_response(content=body), task_ids=[task_id])
    request = adapter._transport.requests[0]
    assert "tools" not in request and "tool_choice" not in request
    assert request["messages"][0]["role"] == "system"
    assert "geometry.circle_area" in request["messages"][0]["content"]
    assert rows[0]["mode"] == "prompt" and rows[0]["passed"] is True


def test_both_modes_produce_one_row_each_for_the_same_record(dataset_root):
    ids = [bfcl.task_id_for("native", "simple_0"), bfcl.task_id_for("prompt", "simple_0")]

    def responder(payload, index):
        if "tools" in payload:
            return chat_response(tool_calls=[("geometry.circle_area", {"radius": 5})])
        return chat_response(content='[{"name": "geometry.circle_area", "arguments": {"radius": 5}}]')

    _, _, rows = run_adapter(dataset_root, responder, task_ids=ids)
    assert [row["task_id"] for row in rows] == ids
    assert {row["mode"] for row in rows} == {"native", "prompt"}
    assert all(row["passed"] for row in rows)
    assert len({row["response_artifact"] for row in rows}) == 2


def test_irrelevance_is_handled_in_both_modes(dataset_root):
    ids = [bfcl.task_id_for("native", "irrelevance_0"), bfcl.task_id_for("prompt", "irrelevance_0")]

    def responder(payload, index):
        if "tools" in payload:
            return chat_response(content="I cannot help with that using the available functions.")
        return chat_response(content="[]")

    _, _, rows = run_adapter(dataset_root, responder, task_ids=ids)
    assert all(row["outcome_status"] == "passed" for row in rows)


def test_irrelevance_emitting_a_call_is_unexpected_call(dataset_root):
    task_id = bfcl.task_id_for("native", "irrelevance_0")
    _, _, rows = run_adapter(
        dataset_root, lambda payload, index: chat_response(tool_calls=[("weather.current",
                                                                       {"city": "Oslo"})]),
        task_ids=[task_id])
    assert rows[0]["outcome_status"] == "unexpected_call" and rows[0]["status"] == "completed"
    assert rows[0]["score"] == 0.0 and rows[0]["model_evaluated"] is True
    assert rows[0]["matched_answer"] is None


def test_every_row_carries_the_required_sample_keys(dataset_root):
    adapter = bfcl.BfclAdapter(FakeTransport(
        lambda payload, index: chat_response(content="[]")))
    context = make_context(dataset_root, options={"modes": ("native", "prompt")})
    rows = adapter.run(context)
    declared = adapter.task_ids(context)
    assert [row["task_id"] for row in rows] == list(declared)
    for row in rows:
        assert not [key for key in REQUIRED_SAMPLE_KEYS if key not in row]
        assert row["synthetic"] is False and row["category"] == "tools"
        if row["status"] != "completed":
            assert row["score"] == 0.0 and row["passed"] is False


def test_malformed_native_tool_call_is_invalid_output(dataset_root):
    task_id = bfcl.task_id_for("native", "simple_0")
    body = {"model": ALIAS, "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_0", "type": "function",
                        "function": {"name": "geometry.circle_area", "arguments": "{radius: 5}"}}]}}]}
    _, _, rows = run_adapter(dataset_root, lambda payload, index: body, task_ids=[task_id])
    assert rows[0]["status"] == "invalid_output" and rows[0]["outcome_status"] == "invalid_output"
    assert rows[0]["model_evaluated"] is True and rows[0]["score"] == 0.0


def test_prose_instead_of_a_call_array_is_invalid_output_in_prompt_mode(dataset_root):
    task_id = bfcl.task_id_for("prompt", "simple_0")
    _, _, rows = run_adapter(
        dataset_root, lambda payload, index: chat_response(content="Sure, I will do that for you."),
        task_ids=[task_id])
    assert rows[0]["status"] == "invalid_output"


def test_a_different_served_model_is_refused(dataset_root):
    task_id = bfcl.task_id_for("native", "simple_0")
    _, _, rows = run_adapter(
        dataset_root,
        lambda payload, index: chat_response(tool_calls=[("geometry.circle_area", {"radius": 5})],
                                             model="some-other-model"),
        task_ids=[task_id])
    assert rows[0]["status"] == "environment_error" and rows[0]["model_evaluated"] is False
    assert "some-other-model" in rows[0]["error"]


def test_raw_bytes_are_persisted_before_they_are_parsed(dataset_root):
    """Unparseable bytes still land in the artifact store, byte for byte, before scoring sees them."""
    artifacts = RecordingArtifacts()
    task_id = bfcl.task_id_for("native", "simple_0")
    _, _, rows = run_adapter(dataset_root, lambda payload, index: b"{not json at all",
                             task_ids=[task_id], artifacts=artifacts)
    assert artifacts.writes == [(rows[0]["response_artifact"], b"{not json at all")]
    assert rows[0]["status"] == "invalid_output"
    assert rows[0]["response_sha256"] == hashlib.sha256(b"{not json at all").hexdigest()


def test_unpersistable_evidence_aborts_but_keeps_the_denominator(dataset_root):
    adapter = bfcl.BfclAdapter(FakeTransport(
        lambda payload, index: chat_response(tool_calls=[("geometry.circle_area", {"radius": 5})])))
    context = make_context(dataset_root, artifacts=RecordingArtifacts(fail=True))
    declared = adapter.task_ids(context)
    with pytest.raises(BenchmarkAborted) as caught:
        adapter.run(context)
    assert [row["task_id"] for row in caught.value.rows] == list(declared)
    assert all(row["score"] == 0.0 and row["status"] == "environment_error"
               for row in caught.value.rows)


def test_missing_artifact_writer_fails_closed(dataset_root):
    adapter = bfcl.BfclAdapter(FakeTransport(lambda payload, index: chat_response(content="[]")))
    context = make_context(dataset_root)
    context = BenchmarkContext(**{**context.__dict__, "artifacts": None})
    rows = adapter.run(context)
    assert rows and all(row["status"] == "environment_error" for row in rows)
    assert [row["task_id"] for row in rows] == list(adapter.task_ids(context))


def test_a_transport_failure_midway_still_produces_every_row(dataset_root):
    def responder(payload, index):
        if index == 1:
            return RuntimeError("connection reset by peer")
        return chat_response(content="[]")

    adapter = bfcl.BfclAdapter(FakeTransport(responder))
    context = make_context(dataset_root, options={"modes": ("prompt",)})
    declared = adapter.task_ids(context)
    rows = adapter.run(context)
    assert [row["task_id"] for row in rows] == list(declared)
    failed = [row for row in rows if row["status"] == "environment_error"]
    assert len(failed) == 1 and "connection reset" in failed[0]["error"]
    assert failed[0]["model_evaluated"] is False


def test_a_request_timeout_marks_only_that_item(dataset_root):
    def responder(payload, index):
        return TimeoutError("read timed out") if index == 0 else chat_response(content="[]")

    adapter = bfcl.BfclAdapter(FakeTransport(responder))
    context = make_context(dataset_root, options={"modes": ("prompt",)})
    rows = adapter.run(context)
    assert rows[0]["status"] == "timeout" and rows[0]["outcome_status"] == "timeout"
    assert [row["status"] for row in rows[1:]] == ["completed"] * (len(rows) - 1)


def test_budget_exhaustion_marks_the_remainder_timeout_instead_of_dropping_it(dataset_root):
    clock = {"left": 600.0}

    def remaining():
        value = clock["left"]
        clock["left"] = 1.0  # the budget is gone after the first item
        return value

    adapter = bfcl.BfclAdapter(FakeTransport(lambda payload, index: chat_response(content="[]")))
    context = make_context(dataset_root, remaining=remaining, options={"modes": ("prompt",)})
    declared = adapter.task_ids(context)
    rows = adapter.run(context)
    assert [row["task_id"] for row in rows] == list(declared)
    assert rows[0]["status"] == "completed"
    assert all(row["status"] == "timeout" and row["score"] == 0.0 for row in rows[1:])
    assert all(row["mode"] == "prompt" for row in rows)


def test_a_raising_budget_callback_is_treated_as_exhausted(dataset_root):
    def remaining():
        raise TimeoutError("evaluation wall budget exhausted")

    adapter = bfcl.BfclAdapter(FakeTransport(lambda payload, index: chat_response(content="[]")))
    context = make_context(dataset_root, remaining=remaining, options={"modes": ("prompt",)})
    rows = adapter.run(context)
    assert rows and all(row["status"] == "timeout" for row in rows)
    assert [row["task_id"] for row in rows] == list(adapter.task_ids(context))


def test_an_unknown_task_id_still_produces_a_row(dataset_root):
    _, _, rows = run_adapter(dataset_root, lambda payload, index: chat_response(content="[]"),
                             task_ids=["bfcl/native/not_in_the_bundle", "bfcl/native/simple_0"])
    assert [row["task_id"] for row in rows] == ["bfcl/native/not_in_the_bundle",
                                                "bfcl/native/simple_0"]
    assert rows[0]["status"] == "environment_error" and "not in the pinned" in rows[0]["error"]


def test_run_without_a_dataset_returns_scored_zero_rows_for_the_requested_ids(tmp_path):
    adapter = bfcl.BfclAdapter(FakeTransport(lambda payload, index: chat_response(content="[]")))
    context = make_context(tmp_path / "missing", task_ids=["bfcl/native/simple_0"])
    rows = adapter.run(context)
    assert len(rows) == 1 and rows[0]["status"] == "environment_error"
    assert rows[0]["mode"] == "native" and "pinned dataset unavailable" in rows[0]["error"]


def test_holdout_and_development_runs_do_not_share_items(dataset_root):
    adapter = bfcl.BfclAdapter(FakeTransport(lambda payload, index: chat_response(content="[]")))
    development = set(adapter.task_ids(make_context(dataset_root)))
    holdout = set(adapter.task_ids(make_context(dataset_root, split="holdout")))
    assert development and holdout and development.isdisjoint(holdout)


def test_generation_settings_and_session_lock_are_honoured(dataset_root):
    adapter = bfcl.BfclAdapter(FakeTransport(lambda payload, index: chat_response(content="[]")))
    context = make_context(dataset_root, task_ids=[bfcl.task_id_for("prompt", "simple_0")])
    adapter.run(context)
    request = adapter._transport.requests[0]
    assert request["max_tokens"] == 256 and request["top_p"] == 1 and request["stream"] is False


def test_a_denied_session_lock_stops_the_run(dataset_root):
    from llmbench.safety import OperationForbidden
    adapter = bfcl.BfclAdapter(FakeTransport(lambda payload, index: chat_response(content="[]")))
    context = make_context(dataset_root, task_ids=[bfcl.task_id_for("prompt", "simple_0")])
    context = BenchmarkContext(**{**context.__dict__, "session_lock": SessionLock()})
    with pytest.raises(OperationForbidden):
        adapter.run(context)
    assert adapter._transport.requests == []


# ---------------------------------------------------------------------------------------------
# The default transport: constructed, never used, by these tests
# ---------------------------------------------------------------------------------------------


def test_default_transport_validates_the_base_url_without_touching_the_network():
    transport = bfcl.UrllibChatTransport("http://inference:8080/v1")
    assert transport.endpoint == "http://inference:8080/v1/chat/completions"
    for bad in ("ftp://inference/v1", "http://user:pw@inference/v1", "http://inference/v1?x=1", "://"):
        with pytest.raises(ValueError):
            bfcl.UrllibChatTransport(bad)


def test_default_transport_rejects_a_nonpositive_timeout():
    transport = bfcl.UrllibChatTransport("http://inference:8080/v1")
    with pytest.raises(ValueError):
        transport.post_chat({"model": ALIAS}, timeout=0)
