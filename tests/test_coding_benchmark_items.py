"""Public-benchmark items through the broker path: host-owned tests, solution-only requests (LIVE-014).

The first live EvalPlus run failed every item with ``'DirectClient' object has no attribute 'execute'``: the
adapter and the broker had been built against different contracts and nothing joined them. These tests pin the
join, and the rule that a result the driver did not earn can never score as a pass.
"""

import json
from types import SimpleNamespace

import pytest

from llmbench.benchmarks.evalplus import _validate_sample
from llmbench.coding.benchmark_items import BenchmarkItemFixture, evalplus_item_fixtures, run_test_module
from llmbench.coding.broker_client import SpoolClient
from llmbench.coding.sandbox import WorkerResult
from llmbench.coding.spool import CodingJobRequest
from llmbench.config import RunMode
from llmbench.safety import SessionLock

TESTS = ("base/0", "base/1", "plus/0")
FIXTURE = BenchmarkItemFixture(fixture_id="evalplus/mbpp-plus/Mbpp/2", revision="evalplus/mbpp-plus@v0.2.0",
                               item_sha256="a" * 64, test_module_source="LLMBENCH_TESTS = ()\n", test_ids=TESTS)


class FakeWorker:
    synthetic = True
    mode = RunMode.LIVE

    def __init__(self, root, result):
        self.allowed_root, self._result = root, result
        self.session_lock = SessionLock(allow_container_execution=True)
        self.staged = None

    def prepare(self, path, *, image, command, limits, collect_result):
        self.staged = {item.name: item.read_text(encoding="utf-8") for item in path.iterdir()}
        return SimpleNamespace(argv=("docker", "run"), name="llmbench-x")

    def run(self, job):
        return self._result


def payload(outcomes, *, compile_ok=True, runner_error=None, error_origin=None, item_id=FIXTURE.fixture_id, **extra):
    return json.dumps({"protocol": 2, "item_id": item_id, "compile_ok": compile_ok, "runner_error": runner_error,
                       "error_origin": error_origin or ("candidate" if runner_error else None),
                       "outcomes": outcomes, **extra}).encode()


def run(tmp_path, result, patch=None):
    worker = FakeWorker(tmp_path, result)
    out = run_test_module(FIXTURE, patch or {"solution.py": "def f():\n    return 1\n"}, worker=worker,
                          image="sha256:" + "0" * 64, timeout_seconds=120)
    return out, worker


def completed(raw):
    return WorkerResult("completed", 0, cleanup_confirmed=True, result_bytes=raw)


def test_every_pinned_test_passing_is_a_pass_the_adapter_accepts(tmp_path):
    out, worker = run(tmp_path, completed(payload([[test, "passed", None] for test in TESTS])))
    sample = out["sample"]
    assert sample["passed"] and sample["score"] == 1.0 and sample["failure_kind"] is None
    assert (sample["passed_tests"], sample["required_tests"]) == (3, 3) and not out["abort_campaign"]
    checked = _validate_sample(sample, required=3)
    assert checked["error"] is None and checked["passed"]
    assert set(worker.staged) == {"solution.py", "evalplus_tests.py", ".llmbench-input.json", ".llmbench-testdriver.py"}
    assert worker.staged["evalplus_tests.py"] == FIXTURE.test_module_source  # the HOST's tests, not the request's


def test_a_failed_assertion_fails_the_item_and_names_its_kind(tmp_path):
    out, _ = run(tmp_path, completed(payload([["base/0", "passed", None], ["base/1", "assertion", "out: 1"],
                                              ["plus/0", "passed", None]])))
    sample = out["sample"]
    assert not sample["passed"] and sample["failure_kind"] == "assertion" and sample["passed_tests"] == 2
    assert sample["failed_test_ids"] == ["base/1"] and sample["failure_detail"] == "out: 1"
    checked = _validate_sample(sample, required=3)
    assert checked["error"] is None and not checked["passed"] and checked["failure_kind"] == "assertion_failed"


@pytest.mark.parametrize("raw", [
    payload([["plus/0", "passed", None], ["base/0", "passed", None], ["base/1", "passed", None]]),   # reordered
    payload([[test, "passed", None] for test in TESTS] + [["plus/1", "passed", None]]),              # invented test
    payload([[test, "ok", None] for test in TESTS]),                                                 # unknown outcome
    payload([[test, "passed", None] for test in TESTS], item_id="evalplus/mbpp-plus/Mbpp/3"),        # another item
    payload([[test, "passed", None] for test in TESTS], passed=True),                                # extra claim
    b"not json", None])
def test_a_result_the_driver_did_not_earn_never_scores_as_a_pass(tmp_path, raw):
    out, _ = run(tmp_path, completed(raw))
    sample = out["sample"]
    assert not sample["passed"] and sample["score"] == 0.0 and sample["passed_tests"] == 0
    assert sample["status"] == "environment-error" and sample["model_evaluated"] is False
    assert _validate_sample(sample, required=3)["error"] is not None


def test_a_prefix_of_passes_is_not_a_pass(tmp_path):
    out, _ = run(tmp_path, completed(payload([["base/0", "passed", None]])))
    assert not out["sample"]["passed"] and out["sample"]["status"] == "environment-error"
    assert out["sample"]["model_evaluated"] is False
    assert "before every pinned test ran" in out["sample"]["failure_reason"]


def test_import_failure_timeout_and_killed_driver_are_failed_items(tmp_path):
    crashed, _ = run(tmp_path / "a", completed(payload([], compile_ok=False, runner_error="SyntaxError: bad")))
    assert crashed["sample"]["failure_kind"] == "exception" and not crashed["sample"]["compile_ok"]
    hung, _ = run(tmp_path / "b", completed(payload([["base/0", "timeout", None]])))
    assert hung["sample"]["failure_kind"] == "timeout" and not hung["sample"]["passed"]
    killed, _ = run(tmp_path / "c", completed(payload([["base/0", "exception", "RuntimeError: broken"]])))
    assert killed["sample"]["failure_kind"] == "exception" and killed["sample"]["cases"][0]["status"] == "ran"
    for out in (crashed, hung, killed):
        assert _validate_sample(out["sample"], required=3)["error"] is None and not out["abort_campaign"]


def test_unverified_cleanup_aborts_the_campaign(tmp_path):
    out, _ = run(tmp_path, WorkerResult("completed", 0, cleanup_confirmed=False, cleanup_error="rm failed",
                                        result_bytes=payload([[test, "passed", None] for test in TESTS])))
    assert out["abort_campaign"] and not out["sample"]["passed"]
    assert out["sample"]["cases"][0]["status"] == "cleanup-unverified"


@pytest.mark.parametrize("patch", [{"solution.py": "x", "evalplus_tests.py": "LLMBENCH_TESTS = ()"},
                                   {"other.py": "x"}, {"solution.py": 1}])
def test_the_patch_is_exactly_the_solution(tmp_path, patch):
    with pytest.raises(ValueError):
        run(tmp_path, completed(b"{}"), patch=patch)


def test_execute_sends_the_solution_and_identity_but_never_the_tests(tmp_path):
    requests, results = tmp_path / "requests", tmp_path / "results"
    requests.mkdir()
    results.mkdir()
    client = SpoolClient(requests, results, clock=lambda: 100.0)
    client.namespace = lambda: {"session_id": "session-a", "attempt_id": "b" * 32}
    seen = {}
    client.wait = lambda request_id, *, deadline, **_: seen.update(request_id=request_id, deadline=deadline) or "done"
    request = {"kind": "benchmark-item", "task_id": FIXTURE.fixture_id, "benchmark_revision": FIXTURE.revision,
               "item_sha256": FIXTURE.item_sha256, "timeout_seconds": 30,
               "files": {"solution.py": "def f():\n    return 1\n", "evalplus_tests.py": "assert False  # SECRET"}}
    assert client.execute(request) == "done" and seen["deadline"] == 100.0 + 30 + 120.0
    written = list(requests.iterdir())
    assert len(written) == 1 and "SECRET" not in written[0].read_text(encoding="utf-8")
    job = CodingJobRequest.model_validate_json(written[0].read_text(encoding="utf-8"))
    assert job.patch == {"solution.py": "def f():\n    return 1\n"} and job.request_id == seen["request_id"]
    assert (job.fixture_id, job.fixture_revision, job.fixture_hash) == (FIXTURE.fixture_id, FIXTURE.revision, "a" * 64)
    for bad in ({**request, "kind": "fixture"}, {**request, "files": {}}, {**request, "timeout_seconds": 0}, "text"):
        with pytest.raises(ValueError):
            client.execute(bad)


def test_item_fixtures_follow_the_selection_and_refuse_a_task_the_pin_lacks(tmp_path):
    assert evalplus_item_fixtures(SimpleNamespace(benchmarks=(), dataset_root=str(tmp_path))) == {}
    other = SimpleNamespace(benchmark_id="bfcl", task_ids=("bfcl/native/simple_python_0",))
    assert evalplus_item_fixtures(SimpleNamespace(benchmarks=(other,), dataset_root=str(tmp_path))) == {}
    selected = SimpleNamespace(benchmark_id="evalplus", task_ids=("evalplus/mbpp-plus/Mbpp/2",))
    with pytest.raises(Exception):  # no pinned dataset under this root: refuse before any model time is spent
        evalplus_item_fixtures(SimpleNamespace(benchmarks=(selected,), dataset_root=str(tmp_path)))


@pytest.mark.parametrize("result", [
    WorkerResult("completed", 125, stderr=b"Docker daemon could not start task", cleanup_confirmed=True),
    WorkerResult("timeout", None, cleanup_confirmed=True),  # no candidate timeout observation
    WorkerResult("protocol-error", 0, cleanup_confirmed=True),
    WorkerResult("result-missing", 1, cleanup_confirmed=True),
    completed(payload([], compile_ok=False, runner_error="ModuleNotFoundError: numpy", error_origin="harness")),
    completed(payload([], compile_ok=False, runner_error="TimeoutError: harness import", error_origin="harness")),
])
def test_infrastructure_failure_is_not_a_measured_evalplus_failure(tmp_path, result):
    out, _ = run(tmp_path, result)
    sample = out["sample"]
    assert sample["status"] == "environment-error" and sample["model_evaluated"] is False
    assert sample["cases"][0]["status"] == "environment-error" and not out["abort_campaign"]
    assert sample["failure_reason"] and _validate_sample(sample, required=3)["error"]


def test_candidate_import_timeout_remains_a_measured_failure(tmp_path):
    out, _ = run(tmp_path, completed(payload([], compile_ok=False, runner_error="TimeoutError: candidate import")))
    assert out["sample"]["status"] == "completed" and out["sample"]["model_evaluated"] is True
    assert out["sample"]["failure_kind"] == "timeout"


def test_error_origin_is_required_and_must_agree_with_the_driver_error(tmp_path):
    for index, changes in enumerate(({"error_origin": "unknown"}, {"error_origin": "harness"},
                                      {"runner_error": "SyntaxError: bad", "error_origin": None})):
        raw = json.loads(payload([]))
        raw.update(changes)
        out, _ = run(tmp_path / str(index), completed(json.dumps(raw).encode()))
        assert out["sample"]["status"] == "environment-error"
