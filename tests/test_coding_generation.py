"""Private coding generation: prompts, deterministic parsing, repair round, raw evidence and broker hand-off."""

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.analysis import _split_rows
from llmbench.coding import generation
from llmbench.coding.broker import default_broker_settings
from llmbench.coding.broker_client import BrokerAborted
from llmbench.coding.fixtures import fixtures
from llmbench.coding.generation import PatchParse, build_prompt, parse_patch, repair_prompt, run_coding_benchmarks
from llmbench.coding.spool import CodingJobRequest, CodingJobResult, SpoolIntegrityError
from llmbench.config import (
    BackendSettings, ExperimentManifest, GenerationSettings, ModelArtifact, RunMode, TaskSelection,
)
from llmbench.containers.config import BenchmarkSelection
from llmbench.evaluations.capture import UnsafeCaptureRuntimeState
from llmbench.safety import SessionLock

from test_coding_broker import FIXTURE, MultiFixtureExecutor, make_broker

LOCK = SessionLock(allow_inference=True)
PYTHON, TYPESCRIPT, JAVASCRIPT = fixtures()
REFERENCE = json.dumps({item.path: item.content for item in PYTHON.reference_files})


def selection(*task_ids):
    return BenchmarkSelection(benchmark_id="coding", revision="private-coding-v1", task_ids=task_ids or (PYTHON.fixture_id,))


def config_for(*task_ids, repair_attempts=0, broker=None):
    return SimpleNamespace(benchmarks=(selection(*task_ids),), broker=broker,
                           generation=GenerationSettings(max_output_tokens=256, repair_attempts=repair_attempts))


def manifest_for(*task_ids):
    """A campaign manifest selecting the coding tasks, for analysis' sample-origin check."""
    return ExperimentManifest(
        model=ModelArtifact(model_key="m:q4", sha256="0" * 64, source_revision="r", quantization="q4",
                            tokenizer_hash="t", template_hash="h"),
        backend=BackendSettings(engine="mock", runtime_revision="v", context_length=8192),
        tasks=(TaskSelection(suite="coding", revision="private-coding-v1", task_ids=task_ids or (PYTHON.fixture_id,)),),
        harness_revision="0.1.0", scorer_revision="v1")


def origin_errors(rows, *task_ids):
    _, errors = _split_rows({"samples": rows}, manifest_for(*task_ids), "development")
    return [item for item in errors if item.startswith("sample_origin_unverified")]


def scripted(*contents):
    """A completion callback answering the scripted contents in order and recording every request."""
    script, calls = list(contents), []

    async def callback(request):
        calls.append(request)
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": item}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    callback.calls = calls
    return callback


class MemoryArtifacts:
    def __init__(self, fail=False):
        self.writes, self.fail = {}, fail

    def write(self, relative, content):
        if self.fail:
            raise OSError("disk full")
        if relative in self.writes:
            raise FileExistsError(relative)
        self.writes[relative] = content


class FakeClient:
    """Answers each submitted request with the next scripted outcome (callable of the request)."""

    def __init__(self, *outcomes):
        self.outcomes, self.submitted, self.clock = list(outcomes), [], lambda: 100.0

    def namespace(self):
        return {"session_id": "session-1", "attempt_id": "a" * 32}

    def submit(self, request):
        assert isinstance(request, CodingJobRequest)
        self.submitted.append(request)
        return request.request_id

    def wait(self, request_id, *, deadline, **_):
        request = next(item for item in self.submitted if item.request_id == request_id)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome(request)


def completed(passed, *, failing=("tail",), synthetic=False):
    """A worker sample as ``run_coding_fixture`` shapes it, including the worker's own ``synthetic`` flag."""
    def build(request):
        cases = [{"case_id": check.case_id, "passed": passed or check.case_id not in failing,
                  "status": "passed" if passed or check.case_id not in failing else "incorrect"}
                 for check in PYTHON.checks]
        sample = {"task_id": PYTHON.fixture_id, "category": "coding", "score": float(passed), "passed": passed,
                  "cases": cases, "required_checks": len(cases), "attempted_checks": len(cases), "synthetic": synthetic,
                  "first_attempt_success": passed and request.attempt_index == 1, "attempt_index": request.attempt_index}
        return CodingJobResult(request_id=request.request_id, request_sha256=request.content_sha256(),
                               status="completed", sample=sample, trace_sha256="1" * 64, cleanup_confirmed=True,
                               abort_campaign=False, finished_utc="t")
    return build


def terminal(status, reason="broker said no"):
    def build(request):
        return CodingJobResult(request_id=request.request_id, request_sha256=request.content_sha256(), status=status,
                               failure_reason=reason, cleanup_confirmed=status != "cleanup-unverified",
                               abort_campaign=status == "cleanup-unverified", finished_utc="t")
    return build


def run(config, callback, client, *, artifacts=None):
    artifacts = artifacts or MemoryArtifacts()
    return run_coding_benchmarks(config, callback, lambda: 50.0, artifacts, LOCK, client=client), artifacts


def test_prompt_contains_fixture_prompt_and_initial_files_only():
    for fixture in fixtures():
        messages = build_prompt(fixture)
        assert [item["role"] for item in messages] == ["system", "user"]
        text = json.dumps(messages)
        assert fixture.prompt in messages[1]["content"]
        for item in fixture.initial_files:
            assert item.path in messages[1]["content"] and item.content in messages[1]["content"]
        assert generation.entry_source(fixture) in messages[0]["content"]
        disclosed = fixture.prompt + "".join(item.content for item in fixture.initial_files)  # model-visible by design
        for check in fixture.checks:
            for secret in (check.expected_json, check.arguments_json):
                assert secret in disclosed or secret not in text, secret
        assert "reference" not in text.lower() and "expected" not in text.lower()
    assert "size must be positive" not in json.dumps(build_prompt(PYTHON))
    assert "new Set(" not in json.dumps(build_prompt(JAVASCRIPT)) and "groups.get" not in json.dumps(build_prompt(TYPESCRIPT))
    assert build_prompt(PYTHON) == build_prompt(PYTHON)


def test_parse_json_object_and_single_fence_deterministically():
    source = PYTHON.reference_files[0].content
    parsed = parse_patch(REFERENCE, PYTHON)
    assert parsed == PatchParse({"batches.py": source}, "json-object", None)
    assert parse_patch("  \n" + REFERENCE + "\n\n", PYTHON) == parsed
    fenced = "Here is the fix:\n```python\n" + source.rstrip("\n") + "\n```\nDone."
    assert parse_patch(fenced, PYTHON) == PatchParse({"batches.py": source}, "single-fence", None)
    assert parse_patch("```\n" + source + "```", PYTHON).patch == {"batches.py": source}
    assert parse_patch(fenced, PYTHON) == parse_patch(fenced, PYTHON)
    assert parse_patch("```ts\nexport const x = 1;\n```", TYPESCRIPT).patch == {"group.ts": "export const x = 1;\n"}
    assert parse_patch("```js\nmodule.exports = {};\n```", JAVASCRIPT).patch == {"unique.cjs": "module.exports = {};\n"}
    # A JSON object wrapped in a fence is one fenced block, never re-parsed as JSON: the rule is positional.
    wrapped = parse_patch("```json\n" + REFERENCE + "\n```", PYTHON)
    assert wrapped.method == "single-fence" and wrapped.patch == {"batches.py": REFERENCE + "\n"}


def test_parse_rejects_multiple_fences_extra_files_and_prose():
    source = PYTHON.reference_files[0].content
    two = "```python\n" + source + "```\nand also\n```python\n" + source + "```"
    assert parse_patch(two, PYTHON).error == "multiple_or_unterminated_fences"
    assert parse_patch("```python\n" + source, PYTHON).error == "multiple_or_unterminated_fences"
    assert parse_patch("```python\n\n```", PYTHON).error == "empty_fence"
    assert parse_patch("I would implement chunked by slicing the list.", PYTHON).error == "no_patch_found"
    assert parse_patch("", PYTHON).error == "no_patch_found"
    assert parse_patch(None, PYTHON).error == "no_text_content"
    extra = parse_patch(json.dumps({"batches.py": source, "helpers.py": "x"}), PYTHON)
    assert extra.patch is None and extra.error == "files_outside_editable_list"
    assert parse_patch('{"batches.py": 1}', PYTHON).error == "json_object_must_map_filenames_to_source_strings"
    assert parse_patch("{}", PYTHON).error == "json_object_must_map_filenames_to_source_strings"
    assert parse_patch('{"batches.py": "a", "batches.py": "b"}', PYTHON).error.startswith("invalid_json")
    assert parse_patch(REFERENCE + " trailing prose", PYTHON).error.startswith("invalid_json")
    assert parse_patch("{ not json ```python\nx\n```", PYTHON).error.startswith("invalid_json")  # no fallback


def test_invalid_patch_scores_zero_and_keeps_denominator():
    client = FakeClient()
    samples, artifacts = run(config_for(), scripted("Sure! Slice the list in a loop."), client)
    assert len(samples) == 1 and client.submitted == []
    sample = samples[0]
    assert sample["task_id"] == PYTHON.fixture_id and sample["category"] == "coding"
    assert sample["score"] == 0.0 and sample["passed"] is False and sample["outcome_status"] == "invalid_patch"
    assert sample["first_attempt_success"] is False and sample["model_evaluated"] is True
    assert sample["fixture_hash"] == PYTHON.identity() and sample["split"] == "development"
    assert sample["generation"]["parse_methods"] == ["no_patch_found"] and sample["generation"]["attempts"] == 1
    assert sample["error"] == "no_patch_found"
    # Measured evidence for the campaign consumers: a real response was persisted and the model scored 0 (REV-B1-01).
    assert sample["synthetic"] is False and sample["model_evaluated"] is True
    assert origin_errors(samples) == []
    unknown, _ = run(config_for(PYTHON.fixture_id, "python/absent"), scripted(REFERENCE), FakeClient(completed(True)))
    assert [row["task_id"] for row in unknown] == [PYTHON.fixture_id, "python/absent"]
    assert unknown[1]["outcome_status"] == "unknown_fixture" and unknown[1]["score"] == 0.0
    assert unknown[1]["model_evaluated"] is False and unknown[1]["synthetic"] is False  # the model was never asked
    harness, _ = run(config_for(), scripted(REFERENCE), FakeClient(TimeoutError("deadline")))
    assert harness[0]["outcome_status"] == "broker_timeout" and harness[0]["model_evaluated"] is False
    assert harness[0]["synthetic"] is False and origin_errors(harness) == ["sample_origin_unverified:python/chunks"]


def test_origin_pair_separates_model_outcomes_from_harness_outcomes():
    """Model-attributable outcomes count as a measured 0; harness failures never masquerade as one (REV-B1-01)."""
    measured = {"invalid_patch": (scripted("prose only"), FakeClient()),
                "broker_rejected:patch_too_large": (scripted(REFERENCE), FakeClient(terminal("rejected", "patch_too_large"))),
                "broker_rejected:patch_outside_allowlist": (
                    scripted(REFERENCE), FakeClient(terminal("rejected", "patch_outside_allowlist")))}
    harness = {"timeout": (scripted(asyncio.TimeoutError()), FakeClient()),
               "broker_timeout": (scripted(REFERENCE), FakeClient(TimeoutError("deadline"))),
               "broker_integrity": (scripted(REFERENCE), FakeClient(SpoolIntegrityError("mismatch"))),
               "broker_rejected:no_time": (scripted(REFERENCE), FakeClient(terminal("rejected", "no_time"))),
               "broker_rejected:too_many_requests": (scripted(REFERENCE), FakeClient(terminal("rejected", "too_many_requests"))),
               "broker_rejected:replay_changed_content": (
                   scripted(REFERENCE), FakeClient(terminal("rejected", "replay_changed_content"))),
               "broker_interrupted": (scripted(REFERENCE), FakeClient(terminal("interrupted", "x"))),
               "broker_cancelled": (scripted(REFERENCE), FakeClient(terminal("cancelled", "x"))),
               "broker_environment_error": (scripted(REFERENCE), FakeClient(terminal("environment-error", "x")))}
    for label, (callback, client) in {**measured, **harness}.items():
        rows, _ = run(config_for(), callback, client)
        row = rows[0]
        expected_outcome = label.split(":")[0]
        assert row["outcome_status"] == expected_outcome and row["score"] == 0.0 and row["passed"] is False, label
        assert row["synthetic"] is False, label
        if label in measured:
            assert row["model_evaluated"] is True and origin_errors(rows) == [], label
        else:
            assert row["model_evaluated"] is False, label
            assert origin_errors(rows) == ["sample_origin_unverified:python/chunks"], label
        if expected_outcome == "timeout":
            assert row["generation"]["responses"] == []  # nothing was persisted: no response ever came back
    # The completed path is untouched: the worker's own synthetic flag is what the row carries, never a default.
    rows, _ = run(config_for(), scripted(REFERENCE), FakeClient(completed(True)))
    assert rows[0]["model_evaluated"] is True and rows[0]["synthetic"] is False and origin_errors(rows) == []
    rows, _ = run(config_for(), scripted(REFERENCE), FakeClient(completed(True, synthetic=True)))
    assert rows[0]["synthetic"] is True and origin_errors(rows) == ["sample_origin_unverified:python/chunks"]


def test_repair_round_broker_failure_keeps_the_measured_first_attempt():
    """A harness failure during the repair round must not erase the real first-attempt measurement."""
    client = FakeClient(completed(False), terminal("interrupted", "broker restarted"))
    samples, _ = run(config_for(repair_attempts=1), scripted(REFERENCE, REFERENCE), client)
    sample = samples[0]
    assert sample["outcome_status"] == "failed" and sample["passed"] is False and sample["score"] == 0.0
    assert sample["first_attempt_success"] is False and sample["model_evaluated"] is True
    assert sample["broker"]["attempt_index"] == 1 and sample["broker"]["status"] == "completed"
    assert sample["repair_error"] == "broker restarted" and sample["repair_broker"]["status"] == "interrupted"
    assert sample["repair_broker"]["attempt_index"] == 2 and len(client.submitted) == 2
    assert sample["generation"]["attempts"] == 2 and sample["generation"]["repair_attempted"] is True
    assert origin_errors(samples) == []


def test_oversize_request_is_scored_invalid_patch_before_submission():
    """A contract-valid request above the broker's request byte cap never reaches the spool (REV-B1-03)."""
    client = FakeClient(completed(True))
    config = config_for(broker=default_broker_settings(max_request_bytes=4096, max_patch_bytes=8_388_608))
    samples, _ = run(config, scripted(json.dumps({"batches.py": "#" * 5000 + "\n"})), client)
    sample = samples[0]
    assert client.submitted == [] and sample["outcome_status"] == "invalid_patch" and sample["score"] == 0.0
    assert sample["error"].startswith("request_too_large") and sample["model_evaluated"] is True
    assert sample["synthetic"] is False and sample["generation"]["parse_method"] == "json-object"
    # Under the cap the identical patch is submitted as before.
    samples, _ = run(config_for(broker=default_broker_settings()), scripted(json.dumps({"batches.py": "#" * 5000 + "\n"})),
                     client)
    assert len(client.submitted) == 1 and samples[0]["outcome_status"] == "passed"


def test_broker_abort_reported_by_the_client_raises_unsafe_state():
    with pytest.raises(UnsafeCaptureRuntimeState, match="campaign abort"):
        run(config_for(), scripted(REFERENCE), FakeClient(BrokerAborted("inherited abort")))


def test_first_attempt_success_flag_and_single_repair_round():
    client = FakeClient(completed(True))
    samples, _ = run(config_for(repair_attempts=1), scripted(REFERENCE), client)
    assert samples[0]["passed"] is True and samples[0]["first_attempt_success"] is True
    assert samples[0]["outcome_status"] == "passed" and samples[0]["generation"]["attempts"] == 1
    assert samples[0]["generation"]["repair_attempted"] is False and len(client.submitted) == 1
    assert client.submitted[0].attempt_index == 1 and client.submitted[0].patch == json.loads(REFERENCE)

    client = FakeClient(completed(False), completed(True))
    callback = scripted(REFERENCE, REFERENCE)
    samples, _ = run(config_for(repair_attempts=1), callback, client)
    sample = samples[0]
    assert sample["passed"] is True and sample["first_attempt_success"] is False
    assert sample["generation"]["attempts"] == 2 and sample["generation"]["repair_attempted"] is True
    assert [item.attempt_index for item in client.submitted] == [1, 2]
    assert len(callback.calls) == 2 and callback.calls[1]["messages"][2]["role"] == "assistant"
    assert '"tail"' in callback.calls[1]["messages"][3]["content"]
    assert sample["broker"]["attempt_index"] == 2 and len(sample["generation"]["responses"]) == 2

    client = FakeClient(completed(False), completed(False))
    samples, _ = run(config_for(repair_attempts=3), scripted(REFERENCE, REFERENCE, REFERENCE), client)
    assert samples[0]["passed"] is False and samples[0]["first_attempt_success"] is False
    assert len(client.submitted) == 2 and samples[0]["generation"]["attempts"] == 2  # exactly one repair round

    client = FakeClient(completed(False))
    callback = scripted(REFERENCE, REFERENCE)
    samples, _ = run(config_for(repair_attempts=0), callback, client)
    assert samples[0]["passed"] is False and len(callback.calls) == 1 and len(client.submitted) == 1
    assert samples[0]["generation"]["repair_attempted"] is False

    client = FakeClient(completed(False))
    samples, _ = run(config_for(repair_attempts=1), scripted(REFERENCE, "no code here"), client)
    assert samples[0]["passed"] is False and samples[0]["repair_error"] == "no_patch_found"
    assert samples[0]["first_attempt_success"] is False and len(client.submitted) == 1


def test_repair_prompt_never_contains_expected_values():
    previous = {"batches.py": PYTHON.initial_files[0].content}
    failures = [{"case_id": check.case_id, "status": "incorrect", "expected_json": check.expected_json,
                 "observation": {"value": json.loads(check.expected_json)}} for check in PYTHON.checks]
    messages = repair_prompt(PYTHON, previous, failures)
    text = json.dumps(messages)
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == json.dumps(previous)
    for check in PYTHON.checks:
        assert check.case_id in messages[3]["content"]
        assert check.expected_json not in text.replace(PYTHON.prompt, "")
        assert check.arguments_json not in text
    assert "observation" not in text and "expected" not in text.lower()
    assert "size must be positive" not in text
    with pytest.raises(ValueError):
        repair_prompt(PYTHON, previous, [{"case_id": "tail"}])
    with pytest.raises(ValueError):
        repair_prompt(PYTHON, {"batches.py": 1}, [])


def test_raw_responses_are_persisted_before_parsing():
    artifacts = MemoryArtifacts()
    samples, _ = run(config_for(), scripted("garbage that will not parse"), FakeClient(), artifacts=artifacts)
    assert list(artifacts.writes) == ["coding/python-chunks/response-1.json"]
    raw = artifacts.writes["coding/python-chunks/response-1.json"]
    assert json.loads(raw)["choices"][0]["message"]["content"] == "garbage that will not parse"
    assert samples[0]["generation"]["responses"] == [hashlib.sha256(raw).hexdigest()]
    with pytest.raises(UnsafeCaptureRuntimeState, match="persisted"):
        run(config_for(), scripted(REFERENCE), FakeClient(completed(True)), artifacts=MemoryArtifacts(fail=True))
    with pytest.raises(Exception):
        run(config_for(), scripted(REFERENCE), FakeClient(completed(True)), artifacts=SimpleNamespace())
    artifacts = MemoryArtifacts()
    run(config_for(repair_attempts=1), scripted(REFERENCE, "still nothing"), FakeClient(completed(False)),
        artifacts=artifacts)
    assert sorted(artifacts.writes) == ["coding/python-chunks/response-1.json", "coding/python-chunks/response-2.json"]


def test_abort_from_broker_raises_unsafe_state():
    with pytest.raises(UnsafeCaptureRuntimeState, match="unverified worker cleanup"):
        run(config_for(), scripted(REFERENCE), FakeClient(terminal("cleanup-unverified", "rm failed")))
    samples, _ = run(config_for(), scripted(REFERENCE), FakeClient(terminal("rejected", "no_time")))
    assert samples[0]["outcome_status"] == "broker_rejected" and samples[0]["error"] == "no_time"
    assert samples[0]["score"] == 0.0 and samples[0]["first_attempt_success"] is False
    samples, _ = run(config_for(), scripted(REFERENCE), FakeClient(TimeoutError("deadline")))
    assert samples[0]["outcome_status"] == "broker_timeout" and samples[0]["score"] == 0.0
    samples, _ = run(config_for(), scripted(REFERENCE), FakeClient(SpoolIntegrityError("mismatch")))
    assert samples[0]["outcome_status"] == "broker_integrity"
    with pytest.raises(UnsafeCaptureRuntimeState):
        run(config_for(), scripted(UnsafeCaptureRuntimeState("callback poisoned")), FakeClient())
    with pytest.raises(Exception):
        run_coding_benchmarks(config_for(), scripted(REFERENCE), lambda: 50.0, MemoryArtifacts(), SessionLock(),
                              client=FakeClient(completed(True)))  # inference not authorized: never asks the model


def test_hook_signature_matches_container_eval_contract():
    parameters = inspect.signature(run_coding_benchmarks).parameters
    assert list(parameters)[:5] == ["config", "callback", "remaining", "artifacts", "session_lock"]
    assert all(parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for name in list(parameters)[:5])
    assert parameters["client"].kind is inspect.Parameter.KEYWORD_ONLY and parameters["client"].default is None
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in list(parameters)[5:])
    assert run_coding_benchmarks(SimpleNamespace(benchmarks=(), generation=None), None, None, None, None) == []
    with pytest.raises(FileNotFoundError):  # client=None resolves the container spool, which is absent here
        run_coding_benchmarks(config_for(), scripted(REFERENCE), lambda: 1.0, MemoryArtifacts(), LOCK)


def test_parsed_patch_refused_by_the_request_contract_scores_zero_without_submission():
    client = FakeClient(completed(True))
    samples, _ = run(config_for(), scripted(json.dumps({"batches.py": "x\u0000y\n"})), client)
    sample = samples[0]
    assert client.submitted == [] and sample["outcome_status"] == "invalid_patch" and sample["score"] == 0.0
    assert sample["error"].startswith("schema:") and sample["first_attempt_success"] is False
    assert sample["generation"]["parse_method"] == "json-object" and sample["generation"]["parse_methods"] == ["json-object"]
    huge = json.dumps({"batches.py": "#" * 8_388_609})  # beyond the contract's hard cap, whatever the broker allows
    samples, _ = run(config_for(), scripted(huge), client)
    assert samples[0]["outcome_status"] == "invalid_patch" and client.submitted == []
    client = FakeClient(completed(False))
    samples, _ = run(config_for(repair_attempts=1), scripted(REFERENCE, json.dumps({"batches.py": "\u0000"})), client)
    assert samples[0]["passed"] is False and samples[0]["repair_error"].startswith("schema:")
    assert len(client.submitted) == 1 and samples[0]["generation"]["parse_method"] == "json-object"
    samples, _ = run(config_for(), scripted("prose only"), FakeClient())
    assert samples[0]["generation"]["parse_method"] is None


class InitialSourceFails(MultiFixtureExecutor):
    """The scripted worker reports a mutated input (a failing check) whenever the staged source is unchanged."""

    def run(self, argv, **limits):
        if argv[1] == "run":
            mount = argv[argv.index("--mount") + 1]
            path = Path(mount.removeprefix("type=bind,source=").removesuffix(",target=/workspace,readonly"))
            source = json.loads(path.joinpath(".llmbench-input.json").read_text())["source"]
            initial = next(item.content for item in FIXTURE.initial_files if item.path == source)
            self.mutate = path.joinpath(source).read_text() == initial
        return super().run(argv, **limits)


def test_end_to_end_through_the_host_broker_with_a_fake_worker(tmp_path):
    broker, executor, _ = make_broker(tmp_path, executor=InitialSourceFails(FIXTURE))
    client = broker.direct_client()
    artifacts = MemoryArtifacts()
    config = SimpleNamespace(benchmarks=(selection(FIXTURE.fixture_id),),
                             generation=GenerationSettings(max_output_tokens=256, repair_attempts=1))
    initial = json.dumps({item.path: item.content for item in FIXTURE.initial_files})
    samples = run_coding_benchmarks(config, scripted(initial, REFERENCE), lambda: 200.0, artifacts,
                                    SessionLock(allow_inference=True, allow_container_execution=True), client=client)
    sample = samples[0]
    assert sample["passed"] is True and sample["first_attempt_success"] is False
    assert sample["generation"]["attempts"] == 2 and sample["broker"]["attempt_index"] == 2
    assert sample["broker"]["cleanup_confirmed"] is True and sample["synthetic"] is True
    assert sum(argv[1] == "run" for argv, _ in executor.calls) == 2 * len(FIXTURE.checks)
    assert sorted(artifacts.writes) == ["coding/python-chunks/response-1.json", "coding/python-chunks/response-2.json"]
    assert sorted(p.name for p in broker.results_dir.iterdir()) == sorted(
        ["namespace.json"] + [f"{rid}.json" for rid in client._submitted])
    assert broker.cancel_all() == {"cancelled": 0, "cleanup_verified": True, "workers_checked": 0}
    assert RunMode.LIVE.value == "live"
