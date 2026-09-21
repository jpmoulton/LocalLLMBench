"""Public-benchmark items the host broker can execute: host-owned tests, evaluator-supplied solution only.

The private fixtures run one container per JSON check case. A public coding benchmark instead ships a test
module per item. The trust split is kept identical: the evaluator submits ``solution.py`` as the patch and names
the item by id and pinned hash; the broker rebuilds the test module from the HOST's pinned dataset and refuses a
request whose hash does not match. Test text therefore never crosses the spool, and an evaluator cannot weaken
the tests it is scored by.
"""

from __future__ import annotations

import base64
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..config import canonical_json
from ..evaluations.tools import strict_json_loads
from .drivers import POLYGLOT_DRIVER, TEST_MODULE_DRIVER
from .fixtures import SourceFile
from .sandbox import DockerWorker, SandboxLimits

SOLUTION_FILE = "solution.py"
TEST_MODULE = "evalplus_tests"
OUTCOMES = frozenset({"passed", "assertion", "exception", "timeout"})
MAX_SOLUTION_BYTES = 262_144


@dataclass(frozen=True)
class BenchmarkItemFixture:
    """One pinned benchmark item, shaped so the broker validates it exactly like a private fixture."""

    fixture_id: str
    revision: str
    item_sha256: str
    test_module_source: str
    test_ids: tuple[str, ...]
    language: str = "python"
    entry_point: str | None = None
    initial_files: tuple[SourceFile, ...] = (SourceFile(SOLUTION_FILE, ""),)

    def identity(self) -> str:
        return self.item_sha256


def evalplus_item_fixtures(config: Any) -> dict[str, BenchmarkItemFixture]:
    """Items for every EvalPlus task the run selected, read from the host's pinned dataset.

    Empty when EvalPlus is not selected. A selected benchmark whose dataset cannot be read raises: the run
    would otherwise start, generate solutions, and only then discover nothing can be scored.
    """
    selections = [item for item in getattr(config, "benchmarks", None) or ()
                  if getattr(item, "benchmark_id", None) == "evalplus"]
    if not selections:
        return {}
    from ..benchmarks import BenchmarkContext
    from ..benchmarks.evalplus import DATASET_SUBDIR, EVALPLUS_DATASETS, build_test_module, load_dataset
    pin = EVALPLUS_DATASETS["mbpp-plus"]
    baked = BenchmarkContext.__dataclass_fields__["dataset_root"].default
    root = Path(getattr(config, "dataset_root", None) or baked)
    dataset = load_dataset(root / DATASET_SUBDIR / pin.filename, pin)
    by_id = dataset.by_task_id()
    wanted = [task_id for selection in selections for task_id in selection.task_ids]
    missing = [task_id for task_id in wanted if task_id not in by_id]
    if missing:
        raise ValueError(f"selected EvalPlus tasks are absent from the pinned dataset: {missing[:5]}")
    return {task_id: BenchmarkItemFixture(
        fixture_id=task_id, revision=pin.revision, item_sha256=by_id[task_id].item_sha256,
        test_module_source=build_test_module(by_id[task_id]), entry_point=by_id[task_id].entry_point,
        test_ids=tuple(test.test_id for test in by_id[task_id].tests)) for task_id in wanted}


# ---------------------------------------------------------------------------------------------------------------
# Aider Polyglot: one Exercism exercise, run by its own pinned test runner
# ---------------------------------------------------------------------------------------------------------------

POLYGLOT_JS_MODULES = "/opt/llmbench/polyglot-js/node_modules"
POLYGLOT_MAX_OUTPUT_CHARS = 6000
_UNSKIP = re.compile(r"\bxtest\(")
"""Exercism ships every JavaScript test after the first as ``xtest`` (skipped); 832 of the 881 tests in the pinned
corpus. Upstream Aider un-skips them with ``sed -i 's/\\bxtest(/test(/g' *.spec.js`` before it runs an exercise,
and so does this, on the HOST's copy of the spec. Without it an exercise "passes" on its first test alone."""
_PYTEST_PASSED = re.compile(r"\b(\d+) passed\b")
_PYTEST_BAD = re.compile(r"\b\d+ (?:failed|errors?)\b")
_JEST_TESTS = re.compile(r"^Tests:\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class PolyglotExerciseFixture:
    """One pinned exercise. ``initial_files`` is the editable allowlist the broker enforces on the patch."""

    fixture_id: str
    revision: str
    exercise_sha256: str
    language: str
    initial_files: tuple[SourceFile, ...]
    test_files: tuple[tuple[str, str], ...]
    support_files: tuple[tuple[str, str], ...]

    def identity(self) -> str:
        return self.exercise_sha256


def polyglot_exercise_fixtures(config: Any) -> dict[str, PolyglotExerciseFixture]:
    """Exercises for every Aider Polyglot task the run selected, loaded from the host's pinned corpus."""
    selections = [item for item in getattr(config, "benchmarks", None) or ()
                  if getattr(item, "benchmark_id", None) == "aider-polyglot"]
    if not selections:
        return {}
    from ..benchmarks import BenchmarkContext
    from ..benchmarks.aider_polyglot import CORPUS_DIRNAME, REVISION, TEST_COMMANDS, load_exercise, practice_root
    baked = BenchmarkContext.__dataclass_fields__["dataset_root"].default
    corpus = Path(getattr(config, "dataset_root", None) or baked) / CORPUS_DIRNAME
    found: dict[str, PolyglotExerciseFixture] = {}
    for task_id in (task_id for selection in selections for task_id in selection.task_ids):
        parts = task_id.split("/")
        if len(parts) != 3 or TEST_COMMANDS.get(parts[1]) is None:
            raise ValueError(f"{task_id}: not an exercise in a language with a pinned test command")
        exercise = load_exercise(task_id, practice_root(corpus, parts[1]) / parts[2])
        found[task_id] = PolyglotExerciseFixture(
            fixture_id=task_id, revision=REVISION, exercise_sha256=exercise.digest, language=exercise.language,
            initial_files=tuple(SourceFile(name, content) for name, content in exercise.solution),
            test_files=exercise.tests, support_files=exercise.support)
    return found


def polyglot_command(fixture: PolyglotExerciseFixture) -> tuple[tuple[str, ...], dict[str, str]]:
    """``(argv, links)`` for the driver. Test files are named explicitly, so a model-created file is never run."""
    tests = [name for name, _ in fixture.test_files]
    if fixture.language == "python":
        return ("python3", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *tests), {}
    if fixture.language == "javascript":
        return (("node", POLYGLOT_JS_MODULES + "/jest/bin/jest.js", "--ci", "--runInBand", "--watchman=false",
                 "--cacheDirectory", "/tmp/jest-cache", "--rootDir", "/tmp/work",
                 "--runTestsByPath", *tests), {"node_modules": POLYGLOT_JS_MODULES})
    raise ValueError(f"no pinned test command for {fixture.language}")


def polyglot_verdict(language: str, exit_code: Any, timed_out: bool, output: str) -> tuple[bool, int | None, str]:
    """``(passed, tests_passed, why)``. An exit code alone never passes: the runner's own summary must agree.

    A solution that ends the process early (``os._exit(0)`` at import, ``process.exit(0)``) yields exit code 0 with
    no summary; requiring the summary line turns that into a failure instead of a pass.
    """
    if timed_out:
        return False, None, "timeout"
    if type(exit_code) is not int or exit_code != 0:
        return False, None, f"exit code {exit_code}"
    if language == "python":
        match = _PYTEST_PASSED.search(output)
        if match is None or int(match[1]) < 1 or _PYTEST_BAD.search(output):
            return False, None, "exit code 0 without a clean pytest summary"
        return True, int(match[1]), ""
    lines = _JEST_TESTS.findall(output)
    summary = lines[-1] if lines else ""
    match = re.search(r"\b(\d+) passed\b", summary)
    if match is None or int(match[1]) < 1 or "failed" in summary:
        return False, None, "exit code 0 without a clean jest summary"
    return True, int(match[1]), ""


def run_polyglot_exercise(fixture: PolyglotExerciseFixture, patch: Mapping[str, str], *, worker: DockerWorker,
                          image: str, limits: SandboxLimits | None = None, split: str = "development",
                          first_attempt: bool = True, timeout_seconds: float | None = None) -> dict:
    """Stage the host's pinned tests beside the submitted solution files and run the pinned test command once."""
    worker.session_lock.check("container", worker.mode)
    allowed = {item.path for item in fixture.initial_files}
    if not isinstance(patch, Mapping) or not patch or set(patch) - allowed:
        raise ValueError("an exercise patch may only contain the exercise's editable solution files")
    if timeout_seconds is not None and (type(timeout_seconds) not in (float, int)
                                        or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Total exercise timeout must be finite and positive")
    resources = limits or SandboxLimits()
    if timeout_seconds is not None:
        resources = SandboxLimits.model_validate({**resources.model_dump(),
                                                  "timeout_seconds": max(1, min(resources.timeout_seconds,
                                                                                int(timeout_seconds)))})
    argv, links = polyglot_command(fixture)
    preflight_argv = (("python3", "-m", "pytest", "--version") if fixture.language == "python"
                      else ("node", POLYGLOT_JS_MODULES + "/jest/bin/jest.js", "--version"))
    sources = {item.path: patch.get(item.path, item.content) for item in fixture.initial_files}
    tests = {name: (_UNSKIP.sub("test(", content) if fixture.language == "javascript" else content)
             for name, content in fixture.test_files}
    staged = {**dict(fixture.support_files), **sources, **tests}  # pinned tests always win a name collision
    root = worker.allowed_root / ("exercise-" + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    started, job, traces = time.monotonic(), None, []
    case: dict[str, Any] = {"case_id": "pinned-test-command", "passed": False}
    observation = None
    try:
        for name, content in staged.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or name.startswith(".llmbench-") or "\\" in name:
                raise ValueError(f"unsafe exercise file name: {name!r}")
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
        root.joinpath(".llmbench-input.json").write_text(canonical_json({
            "item_id": fixture.fixture_id, "argv": list(argv), "links": links,
            "preflight_argv": list(preflight_argv),
            "child_timeout_seconds": max(1, resources.timeout_seconds - 15),
            "max_output_chars": POLYGLOT_MAX_OUTPUT_CHARS}), encoding="utf-8")
        root.joinpath(".llmbench-polyglot-driver.py").write_text(POLYGLOT_DRIVER, encoding="utf-8")
        job = worker.prepare(root, image=image, command=("python3", "/workspace/.llmbench-polyglot-driver.py"),
                             limits=resources, collect_result=True)
        result = worker.run(job)
        raw = asdict(result)
        for key in ("stdout", "stderr", "result_bytes", "result_read_stdout", "result_read_stderr"):
            raw[key + "_base64"] = base64.b64encode(raw.pop(key) or b"").decode("ascii")
        traces.append({"case_id": case["case_id"], "job_argv": job.argv, "worker": raw})
        case.update(cleanup_confirmed=result.cleanup_confirmed, worker_status=result.status)
        if not result.cleanup_confirmed:
            case.update(status="cleanup-unverified", cleanup_error=result.cleanup_error)
        elif result.status != "completed" or result.returncode != 0:
            # This is the Docker/driver envelope, not the candidate test process. Without a valid
            # driver observation, startup/collection/outer-timeout faults cannot be charged to a model.
            case.update(status="environment-error", error=f"worker {result.status}, exit {result.returncode}")
        else:
            observation = _validate_polyglot(result.result_bytes, fixture)
            if observation["error"] is not None:
                case.update(status="environment-error", error=observation["error"])
            elif observation["runner_error"]:
                case.update(status="environment-error", error="driver error: " + observation["runner_error"])
            elif observation["exit_code"] is None and not observation["timed_out"]:
                case.update(status="environment-error", error="driver returned no test exit code")
            elif fixture.language == "python" and observation["exit_code"] in {3, 4, 5}:
                case.update(status="environment-error", error=f"pytest infrastructure exit {observation['exit_code']}")
            else:
                case.update(status="ran")
    except Exception as exc:
        case.update(status="cleanup-unverified" if job is not None else "environment-error", error=str(exc))
        traces.append({"case_id": case["case_id"], "error": type(exc).__name__ + ": " + str(exc)})

    passed, tests_passed, output, exit_code, why = False, None, "", None, case["status"]
    if case["status"] == "environment-error":
        why = case.get("error", "coding worker infrastructure failed")
    elif observation is not None and observation.get("error") is None:
        output, exit_code = observation["output"], observation["exit_code"]
        if observation["runner_error"]:
            why = "driver error: " + str(observation["runner_error"])
        else:
            passed, tests_passed, why = polyglot_verdict(fixture.language, exit_code, observation["timed_out"], output)
    elif observation is not None:
        why = observation["error"]
    case.update(passed=passed, elapsed_seconds=time.monotonic() - started, verdict=why)
    abort_campaign = case["status"] == "cleanup-unverified"
    sample = {"task_id": fixture.fixture_id, "fixture_hash": fixture.identity(), "category": "coding",
              "suite_revision": fixture.revision, "split": split, "language": fixture.language,
              "status": "environment-error" if case["status"] == "environment-error" else "completed",
              "model_evaluated": case["status"] == "ran", "score": float(passed), "passed": passed,
              "failure_reason": case.get("error"), "exit_code": exit_code,
              "tests_passed": tests_passed, "test_output": output[:POLYGLOT_MAX_OUTPUT_CHARS + 64],
              "verdict": why, "cases": [case], "first_attempt_success": first_attempt and passed,
              "elapsed_seconds": time.monotonic() - started, "synthetic": worker.synthetic,
              "execution": "isolated-docker-per-exercise", "abort_campaign": abort_campaign}
    return {"sample": sample, "abort_campaign": abort_campaign, "candidate_path": str(root),
            "raw_bytes": canonical_json({"fixture": fixture.identity(), "traces": traces}).encode()}


def _validate_polyglot(payload: bytes | None, fixture: PolyglotExerciseFixture) -> dict:
    if payload is None:
        return {"error": "result-missing"}
    try:
        parsed = strict_json_loads(payload.decode("utf-8"))
        if type(parsed) is not dict or set(parsed) != {"protocol", "item_id", "exit_code", "timed_out",
                                                       "runner_error", "output"}:
            raise ValueError("unexpected result protocol fields")
        if type(parsed["protocol"]) is not int or parsed["protocol"] != 3 or parsed["item_id"] != fixture.fixture_id:
            raise ValueError("result protocol/item identity mismatch")
        if type(parsed["timed_out"]) is not bool or type(parsed["output"]) is not str:
            raise ValueError("timed_out must be boolean and output a string")
        if parsed["exit_code"] is not None and type(parsed["exit_code"]) is not int:
            raise ValueError("exit_code must be an integer or null")
        if parsed["runner_error"] is not None and type(parsed["runner_error"]) is not str:
            raise ValueError("runner_error must be a string or null")
        return {"error": None, **parsed}
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        return {"error": f"protocol-error: {exc}"}


def _validate(payload: bytes | None, fixture: BenchmarkItemFixture) -> dict:
    """Counts from the driver's result file. Only an in-order prefix of the pinned test ids is believed."""
    if payload is None:
        return {"error": "result-missing"}
    try:
        parsed = strict_json_loads(payload.decode("utf-8"))
        if type(parsed) is not dict or set(parsed) != {"protocol", "item_id", "compile_ok", "runner_error",
                                                       "error_origin", "outcomes"}:
            raise ValueError("unexpected result protocol fields")
        if type(parsed["protocol"]) is not int or parsed["protocol"] != 2 or parsed["item_id"] != fixture.fixture_id:
            raise ValueError("result protocol/item identity mismatch")
        if type(parsed["compile_ok"]) is not bool or type(parsed["outcomes"]) is not list:
            raise ValueError("compile_ok must be boolean and outcomes a list")
        if parsed["runner_error"] is not None and type(parsed["runner_error"]) is not str:
            raise ValueError("runner_error must be a string or null")
        if parsed["error_origin"] not in (None, "candidate", "harness"):
            raise ValueError("error_origin must be candidate, harness or null")
        if bool(parsed["runner_error"]) != (parsed["error_origin"] is not None):
            raise ValueError("runner_error and error_origin must agree")
        outcomes = parsed["outcomes"]
        if len(outcomes) > len(fixture.test_ids):
            raise ValueError("more outcomes than pinned tests")
        for expected, row in zip(fixture.test_ids, outcomes):
            if type(row) is not list or len(row) != 3 or row[0] != expected or row[1] not in OUTCOMES:
                raise ValueError("outcomes must follow the pinned test order")
        if (len(outcomes) < len(fixture.test_ids) and not parsed["runner_error"]
                and all(row[1] == "passed" for row in outcomes)):
            raise ValueError("driver stopped without a failure before every pinned test ran")
        return {"error": None, "compile_ok": parsed["compile_ok"], "runner_error": parsed["runner_error"],
                "error_origin": parsed["error_origin"], "outcomes": outcomes}
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        return {"error": f"protocol-error: {exc}"}


def run_test_module(fixture: BenchmarkItemFixture, patch: Mapping[str, str], *, worker: DockerWorker, image: str,
                    limits: SandboxLimits | None = None, split: str = "development", first_attempt: bool = True,
                    timeout_seconds: float | None = None) -> dict:
    """Run one item's pinned tests against the submitted solution in a single isolated container.

    Returns the same envelope as ``run_coding_fixture`` so the broker publishes it unchanged. pass@1 needs every
    pinned test to pass. Only a validated driver observation earns a model outcome. Missing/malformed results,
    Docker startup/collection faults and harness errors are incomplete infrastructure evidence, not model failures.
    """
    worker.session_lock.check("container", worker.mode)
    solution = patch.get(SOLUTION_FILE) if isinstance(patch, Mapping) else None
    if type(solution) is not str or set(patch) != {SOLUTION_FILE}:
        raise ValueError("a benchmark item patch is exactly solution.py")
    if len(solution.encode("utf-8")) > MAX_SOLUTION_BYTES:
        raise ValueError("solution exceeds the byte budget")
    if timeout_seconds is not None and (type(timeout_seconds) not in (float, int)
                                        or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Total item timeout must be finite and positive")
    resources = limits or SandboxLimits()
    if timeout_seconds is not None:
        resources = SandboxLimits.model_validate({**resources.model_dump(),
                                                  "timeout_seconds": max(1, min(resources.timeout_seconds,
                                                                                int(timeout_seconds)))})
    per_test = max(1, min(10, resources.timeout_seconds // 4))
    root = worker.allowed_root / ("item-" + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    started, job, traces = time.monotonic(), None, []
    required = len(fixture.test_ids)
    case = {"case_id": "pinned-tests", "passed": False}
    try:
        root.joinpath(SOLUTION_FILE).write_text(solution, encoding="utf-8")
        root.joinpath(TEST_MODULE + ".py").write_text(fixture.test_module_source, encoding="utf-8")
        root.joinpath(".llmbench-input.json").write_text(canonical_json({
            "item_id": fixture.fixture_id, "test_module": TEST_MODULE, "entry_point": fixture.entry_point,
            "per_test_timeout_seconds": per_test,
            "import_timeout_seconds": per_test}), encoding="utf-8")
        root.joinpath(".llmbench-testdriver.py").write_text(TEST_MODULE_DRIVER, encoding="utf-8")
        job = worker.prepare(root, image=image, command=("python3", "/workspace/.llmbench-testdriver.py"),
                             limits=resources, collect_result=True)
        result = worker.run(job)
        raw = asdict(result)
        for key in ("stdout", "stderr", "result_bytes", "result_read_stdout", "result_read_stderr"):
            raw[key + "_base64"] = base64.b64encode(raw.pop(key) or b"").decode("ascii")
        traces.append({"case_id": case["case_id"], "job_argv": job.argv, "worker": raw})
        case.update(cleanup_confirmed=result.cleanup_confirmed, worker_status=result.status)
        if not result.cleanup_confirmed:
            case.update(status="cleanup-unverified", cleanup_error=result.cleanup_error)
        elif result.status != "completed" or result.returncode != 0:
            # This is the Docker/driver envelope, not the candidate test process. Without a valid
            # driver observation, startup/collection/outer-timeout faults cannot be charged to a model.
            case.update(status="environment-error", error=f"worker {result.status}, exit {result.returncode}")
        else:
            observation = _validate(result.result_bytes, fixture)
            case["observation"] = observation
            if observation["error"] is not None:
                case.update(status="environment-error", error=observation["error"])
            elif observation["error_origin"] == "harness":
                case.update(status="environment-error", error=observation["runner_error"])
            else:
                case.update(status="ran")
    except Exception as exc:
        case.update(status="cleanup-unverified" if job is not None else "environment-error", error=str(exc))
        traces.append({"case_id": case["case_id"], "error": type(exc).__name__ + ": " + str(exc)})

    passed_tests, compile_ok, failed_ids, kind, detail = 0, False, list(fixture.test_ids), "exception", case["status"]
    observation = case.get("observation")
    if case["status"] == "environment-error":
        detail = case.get("error", "coding worker infrastructure failed")
    elif observation is not None and observation["error"] is None:
        outcomes, compile_ok = observation["outcomes"], observation["compile_ok"]
        passed_ids = {row[0] for row in outcomes if row[1] == "passed"}
        passed_tests = len(passed_ids)
        failed_ids = [test_id for test_id in fixture.test_ids if test_id not in passed_ids]
        first_bad = next((row for row in outcomes if row[1] != "passed"), None)
        if observation["runner_error"]:
            kind = "timeout" if observation["runner_error"].startswith("TimeoutError:") else "exception"
            detail = observation["runner_error"][:500]
        elif first_bad is not None:
            kind, detail = first_bad[1], (first_bad[2] or first_bad[1])
        elif passed_tests == required and compile_ok:
            kind, detail = None, None
        else:
            kind, detail = "exception", "driver stopped before every pinned test ran"
    elif observation is not None:
        detail = observation["error"]
    success = kind is None
    case.update(passed=success, elapsed_seconds=time.monotonic() - started)
    case.pop("observation", None)
    abort_campaign = case["status"] == "cleanup-unverified"
    sample = {"task_id": fixture.fixture_id, "fixture_hash": fixture.identity(), "category": "coding",
              "suite_revision": fixture.revision, "split": split, "language": fixture.language,
              "status": "environment-error" if case["status"] == "environment-error" else "completed",
              "model_evaluated": case["status"] == "ran", "score": float(success), "passed": success,
              "failure_reason": case.get("error"),
              "passed_tests": passed_tests, "required_tests": required, "compile_ok": bool(compile_ok),
              "failure_kind": kind, "failed_test_ids": failed_ids[:64],
              "failure_detail": detail if detail is None else str(detail)[:500],
              "cases": [case], "first_attempt_success": first_attempt and success,
              "elapsed_seconds": time.monotonic() - started, "synthetic": worker.synthetic,
              "execution": "isolated-docker-per-item", "abort_campaign": abort_campaign}
    return {"sample": sample, "abort_campaign": abort_campaign, "candidate_path": str(root),
            "raw_bytes": canonical_json({"fixture": fixture.identity(), "traces": traces}).encode()}
