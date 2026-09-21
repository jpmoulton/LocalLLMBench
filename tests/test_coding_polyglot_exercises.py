"""Aider Polyglot exercises through the broker path: host-owned tests, patch-only requests.

Three things found live are pinned here because no model run could have revealed them:
* 832 of the corpus's 881 JavaScript tests ship as ``xtest`` (skipped); upstream un-skips them, and so must this,
  or an exercise "passes" on its first test alone.
* A solution that ends the process early returns exit code 0 with no test summary. Judged on the exit code, as
  upstream does, that is a false pass.
* A fixed ``nofile=128`` made the CORRECT reference for parallel-letter-frequency die with EMFILE.
"""

import json
from types import SimpleNamespace

import pytest

from llmbench.benchmarks.aider_polyglot import ExecutionRequest, normalize_execution_result
from llmbench.coding.benchmark_items import (PolyglotExerciseFixture, polyglot_command, polyglot_exercise_fixtures,
                                             polyglot_verdict, run_polyglot_exercise)
from llmbench.coding.broker_client import SpoolClient
from llmbench.coding.fixtures import SourceFile
from llmbench.coding.sandbox import DockerWorker, SandboxLimits, WorkerResult
from llmbench.coding.spool import CodingJobRequest, CodingJobResult, publish_atomic, request_name, result_bytes
from llmbench.config import RunMode
from llmbench.safety import SessionLock

PYTEST_OK = "................                                                         [100%]\n16 passed in 0.01s\n"
JEST_OK = "Test Suites: 1 passed, 1 total\nTests:       16 passed, 16 total\nSnapshots:   0 total\n"
SPEC = "test('a', () => {});\nxtest('b', () => {});\n  xtest('c', () => {});\nconst nextest = 1;\n"


def fixture(language="javascript"):
    names = {"javascript": ("two-fer.js", "two-fer.spec.js"), "python": ("two_fer.py", "two_fer_test.py")}[language]
    return PolyglotExerciseFixture(
        fixture_id=f"aider-polyglot/{language}/two-fer", revision="aider-polyglot-v1", exercise_sha256="c" * 64,
        language=language, initial_files=(SourceFile(names[0], "// stub\n"),),
        test_files=((names[1], SPEC),), support_files=(("package.json", "{}"), (names[1], "SUPPORT MUST NOT WIN")))


class FakeWorker:
    synthetic = True
    mode = RunMode.LIVE

    def __init__(self, root, result):
        self.allowed_root, self._result, self.staged, self.command = root, result, None, None
        self.session_lock = SessionLock(allow_container_execution=True)

    def prepare(self, path, *, image, command, limits, collect_result):
        self.staged = {item.name: item.read_text(encoding="utf-8") for item in path.iterdir() if item.is_file()}
        self.command = command
        return SimpleNamespace(argv=("docker", "run"), name="llmbench-x")

    def run(self, job):
        return self._result


def observed(output, *, exit_code=0, timed_out=False, runner_error=None, item="aider-polyglot/javascript/two-fer"):
    raw = json.dumps({"protocol": 3, "item_id": item, "exit_code": exit_code, "timed_out": timed_out,
                      "runner_error": runner_error, "output": output}).encode()
    return WorkerResult("completed", 0, cleanup_confirmed=True, result_bytes=raw)


def run(tmp_path, result, *, language="javascript", patch=None):
    item = fixture(language)
    worker = FakeWorker(tmp_path, result)
    editable = item.initial_files[0].path
    return run_polyglot_exercise(item, patch or {editable: "solution\n"}, worker=worker,
                                 image="sha256:" + "0" * 64, timeout_seconds=120), worker


def test_skipped_javascript_tests_are_unskipped_on_the_hosts_copy_exactly_as_upstream_does(tmp_path):
    out, worker = run(tmp_path, observed(JEST_OK))
    spec = worker.staged["two-fer.spec.js"]
    assert "xtest(" not in spec and spec.count("test(") == 3 and "const nextest = 1;" in spec  # \b: nextest untouched
    assert worker.staged["two-fer.js"] == "solution\n" and "SUPPORT MUST NOT WIN" not in spec
    assert out["sample"]["passed"] and out["sample"]["tests_passed"] == 16
    argv = json.loads(worker.staged[".llmbench-input.json"])["argv"]
    assert argv[-1] == "two-fer.spec.js" and "--runTestsByPath" in argv and worker.command[0] == "python3"


def test_python_tests_are_not_rewritten(tmp_path):
    _, worker = run(tmp_path, observed(PYTEST_OK, item="aider-polyglot/python/two-fer"), language="python")
    assert worker.staged["two_fer_test.py"] == SPEC


@pytest.mark.parametrize(("language", "exit_code", "timed_out", "output", "passed"), [
    ("python", 0, False, PYTEST_OK, True),
    ("python", 0, False, "", False),                                   # os._exit(0) at import: no summary
    ("python", 0, False, "no tests ran in 0.00s\n", False),
    ("python", 0, False, "15 passed, 1 failed in 0.02s\n", False),
    ("python", 0, False, "16 passed, 1 error in 0.02s\n", False),
    ("python", 1, False, PYTEST_OK, False),                            # a printed summary cannot beat the exit code
    ("python", 0, True, PYTEST_OK, False),
    ("python", None, False, PYTEST_OK, False),
    ("javascript", 0, False, JEST_OK, True),
    ("javascript", 0, False, "at Object.require (two-fer.spec.js:1:1)\n", False),   # process.exit(0): no summary
    ("javascript", 0, False, "Tests:       1 failed, 15 passed, 16 total\n", False),
    ("javascript", 0, False, "Tests:       16 skipped, 16 total\n", False),
    ("javascript", 1, False, JEST_OK, False),
])
def test_an_exit_code_alone_never_passes_an_exercise(language, exit_code, timed_out, output, passed):
    assert polyglot_verdict(language, exit_code, timed_out, output)[0] is passed


def test_failures_timeouts_and_driver_errors_are_failed_exercises_with_feedback(tmp_path):
    failed, _ = run(tmp_path / "a", observed("Tests:       3 failed, 13 passed, 16 total\n", exit_code=1))
    assert not failed["sample"]["passed"] and "3 failed" in failed["sample"]["test_output"]
    hung, _ = run(tmp_path / "b", observed("candidate execution timed out", exit_code=None, timed_out=True))
    assert not hung["sample"]["passed"] and hung["sample"]["verdict"] == "timeout"
    broken, _ = run(tmp_path / "c", observed("", exit_code=None, runner_error="OSError: copy failed"))
    assert not broken["sample"]["passed"] and broken["sample"]["verdict"].startswith("driver error")
    forged, _ = run(tmp_path / "d", observed(JEST_OK, item="aider-polyglot/javascript/other"))
    assert not forged["sample"]["passed"]
    for out in (failed, hung):
        assert not out["abort_campaign"] and out["sample"]["status"] == "completed"
        assert out["sample"]["model_evaluated"] is True
    for out in (broken, forged):
        assert not out["abort_campaign"] and out["sample"]["status"] == "environment-error"
        assert out["sample"]["model_evaluated"] is False
    unverified, _ = run(tmp_path / "e", WorkerResult("completed", 0, cleanup_confirmed=False, cleanup_error="rm"))
    assert unverified["abort_campaign"] and not unverified["sample"]["passed"]


@pytest.mark.parametrize("patch", [{"two-fer.spec.js": "test('x', () => {});"}, {"package.json": "{}"},
                                   {"two-fer.js": "x", "extra.js": "y"}, {}])
def test_a_patch_may_only_touch_the_editable_solution_files(tmp_path, patch):
    item = fixture()
    with pytest.raises(ValueError):
        run_polyglot_exercise(item, patch, worker=FakeWorker(tmp_path, observed(JEST_OK)),
                              image="sha256:" + "0" * 64, timeout_seconds=120)


def test_commands_name_the_pinned_test_files_and_other_languages_are_refused():
    argv, links = polyglot_command(fixture("python"))
    assert argv[:3] == ("python3", "-m", "pytest") and argv[-1] == "two_fer_test.py" and links == {}
    argv, links = polyglot_command(fixture("javascript"))
    assert argv[0] == "node" and links == {"node_modules": "/opt/llmbench/polyglot-js/node_modules"}
    rust = PolyglotExerciseFixture("aider-polyglot/rust/x", "aider-polyglot-v1", "c" * 64, "rust", (), (), ())
    with pytest.raises(ValueError):
        polyglot_command(rust)
    selection = SimpleNamespace(benchmark_id="aider-polyglot", task_ids=("aider-polyglot/rust/accumulate",))
    with pytest.raises(ValueError):  # no pinned test command: refused before any model time is spent
        polyglot_exercise_fixtures(SimpleNamespace(benchmarks=(selection,), dataset_root="unused"))
    assert polyglot_exercise_fixtures(SimpleNamespace(benchmarks=(), dataset_root="unused")) == {}


def test_the_client_sends_only_the_patch_and_identity_and_rekeys_the_verified_result(tmp_path):
    requests, results = tmp_path / "requests", tmp_path / "results"
    requests.mkdir()
    results.mkdir()
    client = SpoolClient(requests, results)
    request = ExecutionRequest(session_id="session-a", attempt_id="b" * 32, request_id="d" * 32,
                               task_id="aider-polyglot/javascript/two-fer", language="javascript",
                               exercise_sha256="c" * 64, attempt_index=2, workdir="javascript/two-fer",
                               command=("npm", "test"), patch={"two-fer.js": "solution\n"},
                               test_files={"two-fer.spec.js": "SECRET TESTS"}, support_files={"package.json": "{}"},
                               timeout_seconds=120, upstream_commit="7e06", submitted_utc="2026-09-20T00:00:00+00:00")
    assert client.submit(request) == "d" * 32
    text = (requests / request_name("d" * 32)).read_text(encoding="utf-8")
    job = CodingJobRequest.model_validate_json(text)
    assert "SECRET TESTS" not in text and job.patch == {"two-fer.js": "solution\n"} and job.attempt_index == 2
    assert (job.fixture_id, job.fixture_revision, job.fixture_hash) == (request.task_id, "aider-polyglot-v1", "c" * 64)

    sample = {"passed": True, "test_output": JEST_OK, "tests_passed": 16, "exit_code": 0}
    good = CodingJobResult(request_id="d" * 32, request_sha256=job.content_sha256(), status="completed",
                           sample=sample, cleanup_confirmed=True, abort_campaign=False, finished_utc="t")
    publish_atomic(results, request_name("d" * 32), result_bytes(good))
    raw = client.poll("d" * 32)
    assert raw["request_sha256"] == request.content_sha256() != job.content_sha256()
    outcome = normalize_execution_result(raw, request)
    assert outcome.status == "completed" and outcome.passed and outcome.info["tests_passed"] == 16
    with pytest.raises(TypeError):
        client.submit(SimpleNamespace(suite="something-else"))


def test_open_files_is_a_bounded_limit_the_executor_still_verifies(tmp_path):
    from llmbench.containers.executor import BoundedProcessExecutor
    lock = SessionLock(allow_container_execution=True)
    candidate = tmp_path / "workers" / "candidate"
    candidate.mkdir(parents=True)
    worker = DockerWorker(allowed_root=tmp_path / "workers", session_lock=lock, mode=RunMode.LIVE,
                          executor=BoundedProcessExecutor(session_lock=lock, mode=RunMode.LIVE))
    image = "sha256:" + "0" * 64
    assert "--ulimit=nofile=128:128" in worker.prepare(candidate, image=image, command=("python3", "x.py")).argv
    raised = worker.prepare(candidate, image=image, command=("python3", "x.py"),
                            limits=SandboxLimits(open_files=1024)).argv
    assert "--ulimit=nofile=1024:1024" in raised
    for bad in (16, 8192):
        with pytest.raises(ValueError):
            SandboxLimits(open_files=bad)


@pytest.mark.parametrize("result", [
    WorkerResult("completed", 125, stderr=b"Docker daemon could not start task", cleanup_confirmed=True),
    WorkerResult("timeout", None, cleanup_confirmed=True),
    WorkerResult("protocol-error", 0, cleanup_confirmed=True),
    WorkerResult("completed", 0, cleanup_confirmed=True, result_bytes=b"not json"),
    observed("", exit_code=None, runner_error="FileNotFoundError: pytest missing"),
    observed("", exit_code=None, runner_error="OSError: copy failed"),
    observed("", exit_code=None),
])
def test_infrastructure_failure_is_not_a_measured_polyglot_failure(tmp_path, result):
    out, _ = run(tmp_path, result)
    sample = out["sample"]
    assert sample["status"] == "environment-error" and sample["model_evaluated"] is False
    assert sample["cases"][0]["status"] == "environment-error" and not out["abort_campaign"]
    assert sample["failure_reason"]


@pytest.mark.parametrize("exit_code", [3, 4, 5])
def test_pytest_infrastructure_exit_codes_are_not_candidate_failures(tmp_path, exit_code):
    out, _ = run(tmp_path, observed("pytest cannot collect tests", exit_code=exit_code,
                                  item="aider-polyglot/python/two-fer"), language="python")
    assert out["sample"]["status"] == "environment-error" and out["sample"]["model_evaluated"] is False


def test_candidate_syntax_or_runtime_error_remains_a_model_failure(tmp_path):
    out, _ = run(tmp_path, observed("SyntaxError in solution", exit_code=2,
                                  item="aider-polyglot/python/two-fer"), language="python")
    assert out["sample"]["status"] == "completed" and out["sample"]["model_evaluated"] is True
    assert out["sample"]["passed"] is False


def test_runner_preflight_uses_only_the_pinned_toolchain(tmp_path):
    _, worker = run(tmp_path, observed(JEST_OK))
    settings = json.loads(worker.staged[".llmbench-input.json"])
    assert settings["preflight_argv"] == ["node", "/opt/llmbench/polyglot-js/node_modules/jest/bin/jest.js", "--version"]
    assert "two-fer.js" not in settings["preflight_argv"]
