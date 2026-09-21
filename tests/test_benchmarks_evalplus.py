"""EvalPlus MBPP+ adapter: pinned dataset, deterministic extraction, pass@1, splits, and the execution boundary.

Nothing in this file executes model-generated code, and the autouse ``no_host_execution`` fixture proves the
adapter does not either: process spawning (``subprocess``/``os``), ``exec`` and ``eval`` are replaced by guards
that fail the test if the adapter reaches them, and ``compile`` is allowed only with ``PyCF_ONLY_AST`` â€” the
syntax-only mode ``ast.parse`` uses, which yields a tree and never runnable code. The import machinery is the
single documented exception (a lazily imported module legitimately execs its own module body). Execution is
represented by ``FakeExecutor``, a stand-in for the existing host broker / ``DockerWorker`` path that records
requests and replays scripted results.
"""

import ast
import asyncio
import builtins
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llmbench.benchmarks import REQUIRED_SAMPLE_KEYS, BenchmarkAborted, BenchmarkContext
from llmbench.benchmarks import evalplus
from llmbench.benchmarks.evalplus import (
    EVALPLUS_DATASETS, DatasetError, EvalPlusAdapter, build_messages, build_test_module, execution_request,
    extract_solution, load_dataset, split_assignment,
)
from llmbench.config import GenerationSettings
from llmbench.safety import OperationForbidden

DATA = Path(__file__).parent / "data"
SAMPLE = DATA / "evalplus-mbpp-plus-sample.jsonl"
RESPONSES = json.loads((DATA / "evalplus-responses.json").read_text(encoding="utf-8"))
INVALID = json.loads((DATA / "evalplus-invalid-records.json").read_text(encoding="utf-8"))
MBPP = EVALPLUS_DATASETS["mbpp-plus"]
ONE_ITEM = replace(MBPP, item_count=1, filename="one-item.jsonl")
THREE_ITEMS = replace(MBPP, item_count=3, filename="three-items.jsonl")
IMPORT_MACHINERY = ("importlib", "_frozen_importlib", "zipimport")


# ---------------------------------------------------------------------------------------------------------
# The host-execution guards
# ---------------------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_host_execution(monkeypatch):
    """Fail loudly if the adapter ever tries to run anything on this machine."""
    real_exec, real_eval, real_compile = builtins.exec, builtins.eval, builtins.compile

    def from_import_machinery():
        return sys._getframe(2).f_globals.get("__name__", "").startswith(IMPORT_MACHINERY)

    def spawn_guard(label):
        def forbidden(*args, **kwargs):
            raise AssertionError(f"host execution attempted through {label}")
        return forbidden

    def guarded_exec(*args, **kwargs):
        if from_import_machinery():
            return real_exec(*args, **kwargs)
        raise AssertionError("code was exec'd on the host")

    def guarded_eval(*args, **kwargs):
        if from_import_machinery():
            return real_eval(*args, **kwargs)
        raise AssertionError("code was eval'd on the host")

    def guarded_compile(source, filename, mode, flags=0, *args, **kwargs):
        if not flags & ast.PyCF_ONLY_AST and not from_import_machinery():
            raise AssertionError("runnable code was compiled on the host")
        return real_compile(source, filename, mode, flags, *args, **kwargs)

    for module, attribute in ((subprocess, "Popen"), (subprocess, "run"), (subprocess, "call"),
                              (subprocess, "check_output"), (os, "system"), (os, "popen"),
                              (os, "posix_spawn"), (os, "spawnv"), (os, "execv")):
        monkeypatch.setattr(module, attribute, spawn_guard(f"{module.__name__}.{attribute}"), raising=False)
    monkeypatch.setattr(builtins, "exec", guarded_exec)
    monkeypatch.setattr(builtins, "eval", guarded_eval)
    monkeypatch.setattr(builtins, "compile", guarded_compile)


# ---------------------------------------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------------------------------------

class FakeLock:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def check(self, capability, mode):
        self.calls.append((capability, str(mode)))
        if self.error is not None:
            raise self.error


class MemoryArtifacts:
    def __init__(self, fail=False):
        self.writes, self.fail = {}, fail

    def write(self, relative, data):
        if self.fail:
            raise OSError("artifact budget exhausted")
        if relative in self.writes:
            raise FileExistsError(relative)
        self.writes[relative] = data


class FakeExecutor:
    """Stand-in for the isolated worker path: it records requests and replays results, and runs nothing."""

    def __init__(self, result=None, *, results=None, raises=None):
        self.requests, self.result, self.results, self.raises = [], result, list(results or []), raises

    def execute(self, request):
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        item = self.results.pop(0) if self.results else self.result
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(request)
        if item is None:
            raise AssertionError("the adapter submitted an unexpected execution request")
        return item


def worker_result(passed_tests, required, *, status="completed", failure_kind=None, compile_ok=True,
                  failed=(), reason=None, abort=False, cleanup=True, sample=None):
    body = sample
    if status == "completed" and sample is None:
        body = {"passed_tests": passed_tests, "required_tests": required, "failure_kind": failure_kind,
                "compile_ok": compile_ok, "failed_test_ids": list(failed), "failure_detail": None}
    return {"status": status, "sample": body if status == "completed" else None, "failure_reason": reason,
            "cleanup_confirmed": cleanup, "abort_campaign": abort, "trace_sha256": "a" * 64}


def all_tests_pass(request):
    return worker_result(request["required_tests"], request["required_tests"])


def solution_for(entry_point):
    return f"```python\ndef {entry_point}(*args, **kwargs):\n    return None\n```"


def _demanded_entry_point(request):
    """The entry point the prompt insists on; the fake model answers with exactly that name."""
    user = request["messages"][-1]["content"]
    return user.split("under exactly the name `")[1].split("`")[0]


def dispatching_responder(fail_after=None):
    """Answers each request with a fenced solution named after the entry point that prompt demands."""
    calls = []

    def complete(request):
        calls.append(request)
        if fail_after is not None and len(calls) > fail_after:
            raise RuntimeError("connection reset by the inference server")
        return {"model": request.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": solution_for(_demanded_entry_point(request))}}]}

    complete.calls = calls
    return complete


def fixed_responder(content):
    calls = []

    def complete(request):
        calls.append(request)
        return {"model": request.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}]}

    complete.calls = calls
    return complete


def sample_records():
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_lines(count, pin=MBPP):
    base = sample_records()
    lines = []
    for index in range(count):
        record = dict(base[index % len(base)])
        record["task_id"] = f"{pin.upstream_prefix}/{index + 2}"
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


def write_dataset(root, *, pin=MBPP, count=None, lines=None):
    directory = Path(root) / "evalplus"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / pin.filename
    body = lines if lines is not None else dataset_lines(count if count is not None else pin.item_count, pin)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def make_context(root, *, split="development", seed=42, task_ids=(), completion=None, client=None,
                 remaining=None, artifacts=None, options=None, lock=None, model_alias="candidate-alias"):
    choices = dict(options or {})
    if completion is not None:
        choices.setdefault("completion", completion)
    if client is not None:
        choices.setdefault("execution_client", client)
    return BenchmarkContext(
        base_url="http://inference:8080/v1", model_alias=model_alias, task_ids=tuple(task_ids), split=split,
        seed=seed, generation=GenerationSettings(max_output_tokens=256),
        remaining_seconds=remaining or (lambda: 600.0), artifacts=artifacts or MemoryArtifacts(),
        session_lock=lock or FakeLock(), dataset_root=str(root), options=choices)


def catalog(root, pin=MBPP):
    return {item.task_id: item for item in load_dataset(Path(root) / "evalplus" / pin.filename, pin).items}


def pick(root, count=1, *, entry_point="similar_elements", split="development", seed=42):
    """Declared task ids from the given split, restricted to one entry point so test counts are knowable."""
    ids = EvalPlusAdapter().task_ids(make_context(root, split=split, seed=seed))
    items = catalog(root)
    return tuple(task_id for task_id in ids
                 if entry_point is None or items[task_id].entry_point == entry_point)[:count]


def budget(values):
    """A remaining-seconds callable that walks the given values and then holds the last one."""
    remaining = list(values)

    def clock():
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return clock


def run_one(tmp_path, result, *, content=None, artifacts=None, raises=None, lock=None):
    """One similar_elements item (4 pinned tests) through the whole adapter."""
    write_dataset(tmp_path)
    declared = pick(tmp_path)
    executor = FakeExecutor(result, raises=raises)
    completion = dispatching_responder() if content is None else fixed_responder(content)
    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared, completion=completion,
                                              client=executor, artifacts=artifacts, lock=lock))
    assert len(rows) == 1
    return rows[0], executor, completion


# ---------------------------------------------------------------------------------------------------------
# The execution boundary
# ---------------------------------------------------------------------------------------------------------

FORBIDDEN_MODULES = {"subprocess", "multiprocessing", "ctypes", "runpy", "importlib", "pty", "socket",
                     "urllib", "http", "httpx", "requests", "os", "shutil", "docker", "pickle"}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "system", "popen", "subprocess.run",
                   "subprocess.Popen", "os.system", "os.popen", "os.execv", "os.spawnv",
                   "importlib.import_module"}


def test_the_host_execution_guards_are_actually_armed():
    """The harness itself: if these guards were inert, every other test's isolation claim would be empty."""
    for attempt in (lambda: builtins.exec("1 + 1"), lambda: builtins.eval("1 + 1"),
                    lambda: builtins.compile("1 + 1", "<guard>", "exec"),
                    lambda: subprocess.Popen(["python", "-c", "pass"])):
        with pytest.raises(AssertionError):
            attempt()
    assert ast.parse("1 + 1") is not None  # syntax-only compilation stays allowed


def test_adapter_source_contains_no_execution_primitives():
    """A static reading of the module: nothing in it can start a process or run generated code."""
    tree = ast.parse(Path(evalplus.__file__).read_text(encoding="utf-8"))
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                called.add(target.id)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                called.add(f"{target.value.id}.{target.attr}")
    assert not imported & FORBIDDEN_MODULES, sorted(imported & FORBIDDEN_MODULES)
    assert not called & FORBIDDEN_CALLS, sorted(called & FORBIDDEN_CALLS)


def test_without_an_execution_client_every_row_is_an_environment_error(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path, 3, entry_point=None)
    completion = dispatching_responder()
    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared, completion=completion))
    assert [row["task_id"] for row in rows] == list(declared)
    assert {row["outcome_status"] for row in rows} == {"environment_error"}
    assert all(row["score"] == 0.0 and row["passed"] is False and row["model_evaluated"] is False
               for row in rows)
    assert all("never run on the host" in row["error"] for row in rows)
    assert completion.calls == []  # no model is asked when its answer could not be evaluated anyway


def test_a_whole_pass_runs_under_the_host_execution_guards(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path, 4, entry_point=None)
    executor = FakeExecutor(all_tests_pass)
    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared, completion=dispatching_responder(),
                                              client=executor))
    assert len(rows) == 4 and len(executor.requests) == 4
    assert all(row["passed"] is True and row["execution"] == "isolated-worker-broker" for row in rows)


# ---------------------------------------------------------------------------------------------------------
# Pinned dataset
# ---------------------------------------------------------------------------------------------------------

def test_available_is_false_without_the_dataset(tmp_path):
    ok, reason = EvalPlusAdapter().available(make_context(tmp_path))
    assert ok is False and "absent" in reason and "mbpp-plus" in reason


def test_available_is_false_when_the_item_count_does_not_match_the_pin(tmp_path):
    write_dataset(tmp_path, count=12)
    ok, reason = EvalPlusAdapter().available(make_context(tmp_path))
    assert ok is False and "378" in reason and "12" in reason


def test_available_reports_the_pinned_dataset(tmp_path):
    write_dataset(tmp_path)
    adapter = EvalPlusAdapter()
    ok, reason = adapter.available(make_context(tmp_path))
    assert ok is True and "378 items" in reason
    assert adapter.revision == "evalplus/mbpp-plus@v0.2.0" and adapter.category == "coding"
    assert adapter.benchmark_id == "evalplus" and adapter.smoke_test is False


def test_humaneval_plus_is_supported_but_marked_a_smoke_test():
    assert EvalPlusAdapter("humaneval-plus").smoke_test is True
    assert EVALPLUS_DATASETS["humaneval-plus"].item_count == 164
    assert EVALPLUS_DATASETS["mbpp-plus"].item_count == 378


def test_sample_dataset_loads_with_stable_identities(tmp_path):
    path = write_dataset(tmp_path, pin=THREE_ITEMS, lines=SAMPLE.read_text(encoding="utf-8").splitlines())
    dataset = load_dataset(path, THREE_ITEMS)
    assert [item.task_id for item in dataset.items] == ["evalplus/mbpp-plus/Mbpp/2",
                                                        "evalplus/mbpp-plus/Mbpp/3",
                                                        "evalplus/mbpp-plus/Mbpp/4"]
    assert dataset.items[0].entry_point == "similar_elements" and len(dataset.items[0].tests) == 4
    assert len(dataset.sha256) == 64 and all(len(item.item_sha256) == 64 for item in dataset.items)
    again = load_dataset(path, THREE_ITEMS)
    assert [item.item_sha256 for item in again.items] == [item.item_sha256 for item in dataset.items]


def record_line(record):
    """``<NUL>`` stands in for a real NUL, which cannot be stored literally in the fixture file."""
    return json.dumps(record, ensure_ascii=False).replace("<NUL>", "\\u0000")


@pytest.mark.parametrize("name", sorted(key for key in INVALID if not key.startswith("_")))
def test_load_dataset_refuses_a_record_that_does_not_match_the_pin(tmp_path, name):
    path = write_dataset(tmp_path, pin=ONE_ITEM, lines=[record_line(INVALID[name])])
    with pytest.raises(DatasetError):
        load_dataset(path, ONE_ITEM)


def test_checksum_sidecar_mismatch_fails_closed(tmp_path):
    path = write_dataset(tmp_path, pin=ONE_ITEM, lines=[record_line(INVALID["_base"])])
    load_dataset(path, ONE_ITEM)  # without a sidecar the file is accepted on its own validated content
    path.with_name(path.name + ".sha256").write_text("b" * 64 + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="sidecar"):
        load_dataset(path, ONE_ITEM)


# ---------------------------------------------------------------------------------------------------------
# Splits and the paired-comparison guarantee
# ---------------------------------------------------------------------------------------------------------

def test_development_and_holdout_are_disjoint_and_cover_every_item(tmp_path):
    write_dataset(tmp_path)
    adapter = EvalPlusAdapter()
    development = adapter.task_ids(make_context(tmp_path, split="development"))
    holdout = adapter.task_ids(make_context(tmp_path, split="holdout"))
    assert len(development) == 189 and len(holdout) == 189
    assert not set(development) & set(holdout)
    assert len(set(development) | set(holdout)) == 378
    assert list(development) == sorted(development, key=lambda task: int(task.rsplit("/", 1)[1]))


def test_the_same_seed_gives_two_adapter_instances_the_same_items(tmp_path):
    """The paired-comparison guarantee: two candidates are measured on identical items."""
    write_dataset(tmp_path)
    first = EvalPlusAdapter().task_ids(make_context(tmp_path, seed=7))
    second = EvalPlusAdapter().task_ids(make_context(tmp_path, seed=7))
    assert first == second and len(first) == 189
    assert EvalPlusAdapter().task_ids(make_context(tmp_path, seed=8)) != first


def test_split_assignment_depends_only_on_dataset_seed_and_task_id(tmp_path):
    dataset = load_dataset(write_dataset(tmp_path), MBPP)
    one = split_assignment(dataset.items, dataset_key="mbpp-plus", seed=42)
    two = split_assignment(tuple(reversed(dataset.items)), dataset_key="mbpp-plus", seed=42)
    assert one["development"] == two["development"] and one["holdout"] == two["holdout"]
    assert split_assignment(dataset.items, dataset_key="mbpp-plus", seed=43)["development"] != one["development"]


def test_item_limit_selects_a_nested_deterministic_subset(tmp_path):
    write_dataset(tmp_path)
    adapter = EvalPlusAdapter()
    ten = adapter.task_ids(make_context(tmp_path, options={"item_limit": 10}))
    twenty = adapter.task_ids(make_context(tmp_path, options={"item_limit": 20}))
    assert len(ten) == 10 and len(twenty) == 20 and set(ten) <= set(twenty)
    assert set(twenty) <= set(adapter.task_ids(make_context(tmp_path)))
    assert ten == EvalPlusAdapter().task_ids(make_context(tmp_path, options={"item_limit": 10}))


def test_rows_carry_a_stable_pair_key_for_item_by_item_comparison(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path, 2)
    runs = []
    for outcome in (worker_result(4, 4), worker_result(0, 4, failure_kind="assertion", failed=["base/0"])):
        rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared,
                                                  completion=dispatching_responder(),
                                                  client=FakeExecutor(outcome)))
        runs.append(rows)
    assert [row["paired_key"] for row in runs[0]] == [row["paired_key"] for row in runs[1]]
    assert [row["item_sha256"] for row in runs[0]] == [row["item_sha256"] for row in runs[1]]
    assert [row["passed"] for row in runs[0]] == [True, True]
    assert [row["passed"] for row in runs[1]] == [False, False]
    assert all(row["paired_eligible"] is True for row in runs[0] + runs[1])


# ---------------------------------------------------------------------------------------------------------
# Prompting and extraction
# ---------------------------------------------------------------------------------------------------------

def test_prompt_follows_the_evalplus_instruct_convention(tmp_path):
    path = write_dataset(tmp_path, pin=THREE_ITEMS, lines=SAMPLE.read_text(encoding="utf-8").splitlines())
    item = load_dataset(path, THREE_ITEMS).items[0]
    messages = build_messages(item)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "self-contained Python script" in messages[1]["content"]
    assert item.prompt in messages[1]["content"]
    assert "def similar_elements(test_tup1, test_tup2)" in messages[1]["content"]
    assert _demanded_entry_point({"messages": messages}) == "similar_elements"


@pytest.mark.parametrize("name", sorted(key for key in RESPONSES if not key.startswith("_")))
def test_extraction_is_deterministic_and_refuses_ambiguity(name):
    case = RESPONSES[name]
    result = extract_solution(case["content"], "similar_elements")
    assert result.method == case["method"], name
    assert result.error == case["error"], name
    if case["method"] is None:
        assert result.source is None
    else:
        assert "def similar_elements" in result.source and result.source.endswith("\n")
        assert ast.parse(result.source) is not None


def test_extraction_rejects_an_oversize_solution():
    body = "def similar_elements(a, b):\n" + "    x = 1\n" * 20000
    assert extract_solution(body, "similar_elements").error == "solution_too_large"


def test_extraction_keeps_helpers_defined_after_the_entry_point():
    body = ("Here you go:\n\ndef similar_elements(a, b):\n    return _shared(a, b)\n\n"
            "def _shared(a, b):\n    return tuple(set(a) & set(b))\n\nLet me know if that helps.\n")
    result = extract_solution(body, "similar_elements")
    assert result.method == "prose-trimmed" and "_shared" in result.source


# ---------------------------------------------------------------------------------------------------------
# The execution request
# ---------------------------------------------------------------------------------------------------------

def test_execution_request_carries_the_solution_and_the_pinned_tests(tmp_path):
    path = write_dataset(tmp_path, pin=THREE_ITEMS, lines=SAMPLE.read_text(encoding="utf-8").splitlines())
    dataset = load_dataset(path, THREE_ITEMS)
    item = dataset.items[0]
    request = execution_request(item, "def similar_elements(a, b):\n    return a\n", dataset=dataset,
                                split="development", seed=42)
    assert set(request["files"]) == {"solution.py", "evalplus_tests.py"}
    assert request["entrypoint"] == "solution:similar_elements" and request["language"] == "python"
    assert request["required_tests"] == 4 and request["test_ids"] == ["base/0", "base/1", "plus/0", "plus/1"]
    assert request["item_sha256"] == item.item_sha256 and request["dataset_sha256"] == dataset.sha256
    assert request["benchmark_revision"] == THREE_ITEMS.revision and request["kind"] == "benchmark-item"
    assert request["files"]["solution.py"].startswith("def similar_elements")
    assert json.loads(json.dumps(request)) == request  # JSON-only, so it crosses the spool unchanged
    module = request["files"]["evalplus_tests.py"]
    assert "from solution import similar_elements" in module and "LLMBENCH_TESTS" in module
    assert ast.parse(module) is not None


def test_test_module_names_one_function_per_pinned_test(tmp_path):
    path = write_dataset(tmp_path, pin=THREE_ITEMS, lines=SAMPLE.read_text(encoding="utf-8").splitlines())
    module = build_test_module(load_dataset(path, THREE_ITEMS).items[1])  # the record with a test_setup
    tree = ast.parse(module)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert functions == ["llmbench_test_0000", "llmbench_test_0001", "llmbench_test_0002"]
    assert "import math" in module and "('base/0', 'llmbench_test_0000')" in module.replace('"', "'")


def test_the_submitted_solution_is_exactly_what_was_extracted(tmp_path):
    row, executor, _ = run_one(tmp_path, all_tests_pass)
    submitted = executor.requests[0]["files"]["solution.py"]
    assert submitted == "def similar_elements(*args, **kwargs):\n    return None\n"
    assert row["extraction_method"] == "fenced"


# ---------------------------------------------------------------------------------------------------------
# Scoring: pass@1 and the failure modes
# ---------------------------------------------------------------------------------------------------------

def test_pass_at_one_requires_every_expanded_test(tmp_path):
    passed_row, _, _ = run_one(tmp_path, worker_result(4, 4))
    assert passed_row["status"] == "completed" and passed_row["outcome_status"] == "passed"
    assert passed_row["score"] == 1.0 and passed_row["passed"] is True
    assert passed_row["passed_tests"] == 4 and passed_row["required_tests"] == 4
    near, _, _ = run_one(tmp_path, worker_result(3, 4, failure_kind="assertion", failed=["plus/1"]))
    assert near["passed"] is False and near["score"] == 0.0 and near["outcome_status"] == "assertion_failed"
    assert near["passed_tests"] == 3 and near["failed_test_ids"] == ["plus/1"]


@pytest.mark.parametrize("result,outcome,status,model_evaluated", [
    (worker_result(0, 4, failure_kind="assertion", failed=["base/0"]), "assertion_failed", "completed", True),
    (worker_result(0, 4, failure_kind="exception", compile_ok=False), "exception", "completed", True),
    (worker_result(2, 4, failure_kind="timeout"), "timeout", "timeout", True),
    (worker_result(0, 0, status="timeout", reason="worker timeout"), "timeout", "timeout", True),
    (worker_result(0, 0, status="environment-error", reason="docker is unavailable"), "environment_error",
     "environment_error", False),
    (worker_result(0, 0, status="rejected", reason="patch_too_large"), "invalid_output", "invalid_output", True),
    (worker_result(0, 0, status="rejected", reason="spool_full"), "environment_error", "environment_error",
     False),
])
def test_each_failure_mode_maps_to_its_outcome_status(tmp_path, result, outcome, status, model_evaluated):
    row, _, _ = run_one(tmp_path, result)
    assert row["outcome_status"] == outcome and row["status"] == status
    assert row["model_evaluated"] is model_evaluated
    assert row["score"] == 0.0 and row["passed"] is False


def test_an_unusable_response_is_invalid_output_not_a_guess(tmp_path):
    row, executor, _ = run_one(tmp_path, all_tests_pass, content=RESPONSES["two_different_blocks"]["content"])
    assert row["status"] == "invalid_output" and row["outcome_status"] == "invalid_output"
    assert row["model_evaluated"] is True and row["score"] == 0.0
    assert row["extraction_error"] == "ambiguous_code_blocks"
    assert executor.requests == []  # nothing is submitted when nothing could be extracted


@pytest.mark.parametrize("sample", [
    {"passed_tests": 4, "required_tests": 4, "failure_kind": "assertion", "compile_ok": True},
    {"passed_tests": 3, "required_tests": 4, "failure_kind": None, "compile_ok": True},
    {"passed_tests": 9, "required_tests": 4, "failure_kind": None, "compile_ok": True},
    {"passed_tests": 4, "required_tests": 7, "failure_kind": None, "compile_ok": True},
    {"passed_tests": "4", "required_tests": 4, "failure_kind": None, "compile_ok": True},
    {"passed_tests": 4, "required_tests": 4, "failure_kind": None, "compile_ok": False},
    {"passed": True},
])
def test_an_incoherent_worker_result_is_an_environment_error_never_a_pass(tmp_path, sample):
    row, _, _ = run_one(tmp_path, worker_result(0, 0, sample=sample))
    assert row["outcome_status"] == "environment_error" and row["passed"] is False and row["score"] == 0.0
    assert row["model_evaluated"] is False and "contract" in row["error"]


def test_unverified_worker_cleanup_aborts_the_run(tmp_path):
    with pytest.raises(BenchmarkAborted, match="cleanup"):
        run_one(tmp_path, worker_result(0, 0, status="cleanup-unverified", reason="container still present",
                                        cleanup=False, abort=True))


# ---------------------------------------------------------------------------------------------------------
# Denominator discipline
# ---------------------------------------------------------------------------------------------------------

def test_every_declared_task_produces_a_row_when_the_transport_fails_midway(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path, 5, entry_point=None)
    completion = dispatching_responder(fail_after=1)
    executor = FakeExecutor(all_tests_pass)
    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared, completion=completion,
                                              client=executor))
    assert [row["task_id"] for row in rows] == list(declared)
    assert rows[0]["passed"] is True
    assert [row["outcome_status"] for row in rows[1:]] == ["environment_error"] * 4
    assert all(row["model_evaluated"] is False for row in rows[1:])
    assert all(set(REQUIRED_SAMPLE_KEYS) <= set(row) for row in rows)
    assert len(executor.requests) == 1


def test_every_declared_task_produces_a_row_when_the_executor_fails_midway(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path, 6, entry_point=None)
    executor = FakeExecutor(raises=OSError("spool unavailable"))
    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared,
                                              completion=dispatching_responder(), client=executor))
    assert [row["task_id"] for row in rows] == list(declared)
    assert {row["outcome_status"] for row in rows} == {"environment_error"}
    assert all(row["score"] == 0.0 and row["model_evaluated"] is False for row in rows)
    assert len(executor.requests) == 3  # it stops asking a client that failed three times in a row


def test_budget_exhaustion_marks_the_remaining_rows_instead_of_dropping_them(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path, 4, entry_point=None)
    executor = FakeExecutor(all_tests_pass)
    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared,
                                              completion=dispatching_responder(), client=executor,
                                              remaining=budget([600.0, 600.0, 1.0])))
    assert [row["task_id"] for row in rows] == list(declared)
    assert [row["outcome_status"] for row in rows] == ["passed", "passed", "budget_exhausted",
                                                       "budget_exhausted"]
    assert all(row["score"] == 0.0 and row["status"] == "environment_error" for row in rows[2:])
    assert len(executor.requests) == 2


def test_a_task_from_the_other_split_is_refused_not_silently_run(tmp_path):
    write_dataset(tmp_path)
    holdout = pick(tmp_path, 1, entry_point=None, split="holdout")
    executor = FakeExecutor(all_tests_pass)
    rows = EvalPlusAdapter().run(make_context(tmp_path, split="development", task_ids=holdout,
                                              completion=dispatching_responder(), client=executor))
    assert len(rows) == 1 and rows[0]["outcome_status"] == "environment_error"
    assert "other split" in rows[0]["error"] and executor.requests == []


def test_an_unavailable_dataset_still_produces_every_declared_row(tmp_path):
    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=("evalplus/mbpp-plus/Mbpp/2",),
                                              completion=dispatching_responder(),
                                              client=FakeExecutor(all_tests_pass)))
    assert len(rows) == 1 and rows[0]["outcome_status"] == "environment_error"
    assert rows[0]["score"] == 0.0 and rows[0]["model_evaluated"] is False
    assert set(REQUIRED_SAMPLE_KEYS) <= set(rows[0])


def test_an_empty_selection_runs_the_whole_split_and_returns_one_row_each(tmp_path):
    write_dataset(tmp_path)
    executor = FakeExecutor(all_tests_pass)
    rows = EvalPlusAdapter().run(make_context(tmp_path, completion=dispatching_responder(), client=executor))
    expected = EvalPlusAdapter().task_ids(make_context(tmp_path))
    assert [row["task_id"] for row in rows] == list(expected) and len(rows) == 189
    assert len({row["task_id"] for row in rows}) == 189 and len(executor.requests) == 189


def test_rows_carry_every_required_sample_key(tmp_path):
    row, _, _ = run_one(tmp_path, worker_result(4, 4))
    assert set(REQUIRED_SAMPLE_KEYS) <= set(row)
    assert row["suite"] == "evalplus" and row["suite_revision"] == "evalplus/mbpp-plus@v0.2.0"
    assert row["category"] == "coding" and row["split"] == "development" and row["synthetic"] is False
    assert row["metric"] == "pass@1" and row["dataset"] == "mbpp-plus"


# ---------------------------------------------------------------------------------------------------------
# Evidence, permission and transport details
# ---------------------------------------------------------------------------------------------------------

def test_the_raw_response_is_persisted_before_it_is_parsed(tmp_path):
    artifacts = MemoryArtifacts()
    row, _, _ = run_one(tmp_path, all_tests_pass, content="no code at all", artifacts=artifacts)
    assert row["outcome_status"] == "invalid_output"
    assert len(artifacts.writes) == 1 and next(iter(artifacts.writes)).startswith("evalplus/")
    written = json.loads(next(iter(artifacts.writes.values())).decode("utf-8"))
    assert written["choices"][0]["message"]["content"] == "no code at all"


def test_unpersistable_evidence_aborts_rather_than_scoring(tmp_path):
    with pytest.raises(BenchmarkAborted, match="persisted"):
        run_one(tmp_path, all_tests_pass, artifacts=MemoryArtifacts(fail=True))


def test_every_model_call_passes_the_inference_permission_gate(tmp_path):
    lock = FakeLock()
    run_one(tmp_path, all_tests_pass, lock=lock)
    assert lock.calls and all(capability == "inference" for capability, _ in lock.calls)


def test_a_denied_permission_propagates_instead_of_becoming_a_row(tmp_path):
    with pytest.raises(OperationForbidden):
        run_one(tmp_path, all_tests_pass, lock=FakeLock(error=OperationForbidden("policy denies inference")))


def test_the_request_carries_the_candidate_alias_and_greedy_settings(tmp_path):
    _, _, completion = run_one(tmp_path, all_tests_pass)
    request = completion.calls[0]
    assert request["model"] == "candidate-alias" and request["temperature"] == 0
    assert request["max_tokens"] == 256 and request["seed"] == 42 and request["top_p"] == 1


def test_a_different_served_model_is_refused(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path)
    executor = FakeExecutor(all_tests_pass)

    def wrong_model(request):
        return {"model": "some-other-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": solution_for("similar_elements")}}]}

    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared, completion=wrong_model,
                                              client=executor))
    assert rows[0]["outcome_status"] == "environment_error"
    assert rows[0]["served_model"] == "some-other-model" and executor.requests == []


def test_an_async_completion_callable_is_awaited(tmp_path):
    write_dataset(tmp_path)
    declared = pick(tmp_path)
    seen = []

    async def complete(request):
        seen.append(request)
        await asyncio.sleep(0)
        return {"model": request["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": solution_for("similar_elements")}}]}

    rows = EvalPlusAdapter().run(make_context(tmp_path, task_ids=declared, completion=complete,
                                              client=FakeExecutor(all_tests_pass)))
    assert len(seen) == 1 and rows[0]["passed"] is True


@pytest.mark.parametrize("flags", [{"status": "environment-error"}, {"model_evaluated": False}])
def test_completed_envelope_cannot_hide_infrastructure_only_evidence(tmp_path, flags):
    sample = {"passed_tests": 0, "required_tests": 4, "compile_ok": False, "failure_kind": "exception", **flags}
    row, _, _ = run_one(tmp_path, worker_result(0, 4, sample=sample))
    assert row["status"] == "environment_error" and row["model_evaluated"] is False
