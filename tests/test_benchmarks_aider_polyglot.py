"""Aider Polyglot adapter: edit-format parser, two-attempt loop, splits, metrics and the isolation boundary.

Everything here is offline. No Docker, no model, no network, no host execution: ``no_host_execution``
replaces every process-spawning entry point with a failing stub for the whole module, so any attempt to
run model-written code on this machine breaks the suite loudly instead of quietly passing.

The corpus under ``tests/data/aider-polyglot`` is hand-made (see its README); no upstream exercise
content is redistributed here.
"""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.benchmarks import BenchmarkAborted, BenchmarkContext, BenchmarkUnavailable, REQUIRED_SAMPLE_KEYS
from llmbench.benchmarks import aider_polyglot as adapter
from llmbench.benchmarks.aider_polyglot import (
    AiderPolyglotBenchmark, CorpusError, EditBlock, ExecutionRequest, aggregate_metrics, apply_edit_blocks,
    build_initial_prompt, build_plan, build_repair_prompt, load_exercise, normalize_execution_result,
    parse_and_apply, parse_edit_blocks, pinned_test_command, read_pin, selected_languages, split_exercises,
)
from llmbench.config import GenerationSettings
from llmbench.safety import SessionLock

DATA_ROOT = str(Path(__file__).parent / "data")
LOCK = SessionLock(allow_inference=True)
WORD_REVERSE = "aider-polyglot/python/word-reverse"
FLATTEN = "aider-polyglot/python/flatten-once"
TITLE_CASE = "aider-polyglot/javascript/title-case"
STUB_LINE = '    raise NotImplementedError("implement reverse")'
SOLUTION_LINE = "    return text[::-1]"


@pytest.fixture(autouse=True)
def no_host_execution(monkeypatch):
    """Model-written code never runs here. Every process API fails loudly for the whole module."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"host execution attempted: {args!r} {kwargs!r}")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, refuse)
    monkeypatch.setattr("os.system", refuse)
    monkeypatch.setattr("os.execv", refuse, raising=False)
    monkeypatch.setattr("os.posix_spawn", refuse, raising=False)


# ---------------------------------------------------------------------------------------------------
# harness doubles
# ---------------------------------------------------------------------------------------------------


class MemoryArtifacts:
    def __init__(self, fail=False):
        self.writes, self.fail = {}, fail

    def write(self, relative, content):
        if self.fail:
            raise OSError("disk full")
        if relative in self.writes:
            raise FileExistsError(relative)
        self.writes[relative] = content


class ScriptExhausted(BaseException):
    """Not an ``Exception``: the adapter's own transport guard must not be able to swallow it."""


def _completion(request, content):
    return {"model": request["model"], "choices": [{"index": 0, "finish_reason": "stop",
                                                    "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7}}


class FakeTransport:
    """Answers scripted assistant contents in order; an exception in the script is raised instead."""

    def __init__(self, *contents):
        self.script, self.requests = list(contents), []

    def __call__(self, request):
        self.requests.append(request)
        if not self.script:
            raise ScriptExhausted("the adapter asked for more completions than the script provides")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _completion(request, item)


# The one-line edit that solves each fixture exercise, keyed by its editable file.
SOLUTIONS = {
    "word_reverse.py": (STUB_LINE, SOLUTION_LINE),
    "flatten_once.py": ("def flatten_once(rows):\n    return []",
                        "def flatten_once(rows):\n    return [item for row in rows for item in row]"),
    "digit_sum.py": ('    raise NotImplementedError("implement digit_sum")',
                     "    return sum(int(digit) for digit in str(number))"),
    "sum_pair.py": ('    raise NotImplementedError("implement sum_pair")', "    return left + right"),
    "title-case.js": ("  throw new Error('Remove this statement and implement this function');",
                      "  return text.replace(/\\b\\w/g, (c) => c.toUpperCase());"),
    "list-total.js": ("  throw new Error('Remove this statement and implement this function');",
                      "  return values.reduce((a, b) => a + b, 0);"),
}


class SolvingTransport:
    """Answers whichever exercise it is shown with that exercise's correct edit block."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        prompt = request["messages"][1]["content"]
        for name, (search, replace) in SOLUTIONS.items():
            if f"--- {name} (current content) ---" in prompt:
                return _completion(request, block(name, search, replace))
        raise ScriptExhausted(f"no scripted solution for this prompt: {prompt[:120]!r}")


class FakeExecutionClient:
    """The isolated worker path, scripted. It never executes anything; it only answers."""

    def __init__(self, *outcomes, namespace=None):
        self.outcomes = list(outcomes)
        self.requests: list[ExecutionRequest] = []
        self._namespace = namespace or {"session_id": "session-1", "attempt_id": "a" * 32}
        self.clock = lambda: 1000.0

    def namespace(self):
        if isinstance(self._namespace, BaseException):
            raise self._namespace
        return self._namespace

    def submit(self, request):
        assert isinstance(request, ExecutionRequest), "the adapter must submit a structured request"
        self.requests.append(request)
        return request.request_id

    def wait(self, request_id, *, deadline, **_):
        request = next(item for item in self.requests if item.request_id == request_id)
        if not self.outcomes:
            raise ScriptExhausted("the adapter submitted more work than the script provides")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome(request)


def completed(passed, output="AssertionError: expected 'cba'"):
    def build(request):
        return {"request_id": request.request_id, "request_sha256": request.content_sha256(),
                "status": "completed", "sample": {"passed": passed, "test_output": output,
                                                  "tests_passed": 1 if passed else 0},
                "failure_reason": None, "trace_sha256": "1" * 64, "cleanup_confirmed": True,
                "abort_campaign": False, "finished_utc": "t"}
    return build


def terminal(status, reason="the worker refused"):
    def build(request):
        return {"request_id": request.request_id, "request_sha256": request.content_sha256(),
                "status": status, "sample": None, "failure_reason": reason, "trace_sha256": None,
                "cleanup_confirmed": status != "cleanup-unverified",
                "abort_campaign": status == "cleanup-unverified", "finished_utc": "t"}
    return build


def context(*, task_ids=(WORD_REVERSE,), split="development", seed=42, remaining=lambda: 300.0,
            artifacts=None, dataset_root=DATA_ROOT, **options):
    return BenchmarkContext(
        base_url="http://inference:8080/v1", model_alias="candidate-q4", task_ids=tuple(task_ids),
        split=split, seed=seed, generation=GenerationSettings(max_output_tokens=256),
        remaining_seconds=remaining, artifacts=artifacts if artifacts is not None else MemoryArtifacts(),
        session_lock=LOCK, dataset_root=dataset_root, options=dict(options))


def block(filename, search, replace):
    return (f"{filename}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n")


def fenced(filename, search, replace, language="python"):
    return (f"{filename}\n```{language}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n"
            f">>>>>>> REPLACE\n```\n")


REVERSE_EDIT = block("word_reverse.py", STUB_LINE, SOLUTION_LINE)


def run_case(*, responses, outcomes, task_ids=(WORD_REVERSE,), artifacts=None, **kwargs):
    transport, client = FakeTransport(*responses), FakeExecutionClient(*outcomes)
    store = artifacts if artifacts is not None else MemoryArtifacts()
    benchmark = AiderPolyglotBenchmark(transport=transport, execution_client=client)
    rows = benchmark.run(context(task_ids=task_ids, artifacts=store, **kwargs))
    return rows, transport, client, store


# ---------------------------------------------------------------------------------------------------
# search/replace parser
# ---------------------------------------------------------------------------------------------------


FILES = {"a.py": "def f():\n    return 1\n\n\ndef g():\n    return 1\n", "b.py": "value = 0\n"}


def test_single_exact_match_applies():
    parsed = parse_and_apply(block("b.py", "value = 0", "value = 42"), FILES)
    assert parsed.well_formed and parsed.error is None
    assert parsed.files == {**FILES, "b.py": "value = 42\n"}
    assert parsed.blocks == (EditBlock("b.py", "value = 0\n", "value = 42\n"),)
    assert FILES["b.py"] == "value = 0\n", "the caller's files must not be mutated"


def test_zero_matches_is_malformed():
    parsed = parse_and_apply(block("b.py", "value = 7", "value = 42"), FILES)
    assert not parsed.well_formed and parsed.error.startswith("search_not_found:b.py#0")
    assert parsed.files is None


def test_multiple_matches_is_malformed():
    parsed = parse_and_apply(block("a.py", "    return 1", "    return 2"), FILES)
    assert parsed.error == "search_not_unique:a.py#0:2"
    assert parsed.files is None


def test_multiple_matches_in_the_real_fixture_stub():
    exercise = load_exercise(FLATTEN, build_plan(context()).directories[FLATTEN])
    parsed = parse_and_apply(block("flatten_once.py", "    return []", "    return sum(rows, [])"),
                             exercise.editable())
    assert parsed.error == "search_not_unique:flatten_once.py#0:2"


def test_missing_divider():
    text = "b.py\n<<<<<<< SEARCH\nvalue = 0\n>>>>>>> REPLACE\n"
    assert parse_and_apply(text, FILES).error == "missing_divider"


def test_unterminated_block():
    text = "b.py\n<<<<<<< SEARCH\nvalue = 0\n=======\nvalue = 1\n"
    assert parse_and_apply(text, FILES).error == "unterminated_block"


def test_missing_search_marker_is_no_edit_blocks():
    text = "b.py\n```python\nvalue = 42\n```\n"
    assert parse_and_apply(text, FILES).error == "no_edit_blocks"


def test_stray_replace_marker():
    assert parse_and_apply("nothing here\n>>>>>>> REPLACE\n", FILES).error == "stray_replace_marker"


def test_nested_search_marker():
    text = "b.py\n<<<<<<< SEARCH\n<<<<<<< SEARCH\n=======\nx\n>>>>>>> REPLACE\n"
    assert parse_and_apply(text, FILES).error == "nested_search_marker"


def test_duplicate_divider():
    text = "b.py\n<<<<<<< SEARCH\nvalue = 0\n=======\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE\n"
    assert parse_and_apply(text, FILES).error == "duplicate_divider"


def test_empty_search_is_malformed():
    text = "b.py\n<<<<<<< SEARCH\n=======\nvalue = 1\n>>>>>>> REPLACE\n"
    assert parse_and_apply(text, FILES).error == "empty_search"


def test_missing_filename():
    text = "<<<<<<< SEARCH\nvalue = 0\n=======\nvalue = 1\n>>>>>>> REPLACE\n"
    assert parse_and_apply(text, FILES).error == "missing_filename"


def test_unlisted_filename_is_rejected():
    parsed = parse_and_apply(block("c.py", "value = 0", "value = 1"), FILES)
    assert parsed.error == "unknown_filename:c.py"


@pytest.mark.parametrize("name", ["../escape.py", "../../etc/passwd", "/etc/passwd", "C:\\windows\\x.py",
                                  "sub/../../a.py", "./a.py"])
def test_traversal_and_absolute_filenames_are_rejected_before_the_allowlist(name):
    parsed = parse_and_apply(block(name, "value = 0", "value = 1"), FILES)
    assert parsed.error.startswith("unsafe_filename:"), parsed.error
    assert parsed.files is None


def test_test_file_is_not_editable():
    exercise = load_exercise(WORD_REVERSE, build_plan(context()).directories[WORD_REVERSE])
    parsed = parse_and_apply(block("word_reverse_test.py", "x", "y"), exercise.editable())
    assert parsed.error == "unknown_filename:word_reverse_test.py"


def test_multiple_blocks_in_one_response_apply_in_order():
    text = block("b.py", "value = 0", "value = 1") + block("b.py", "value = 1", "value = 2")
    parsed = parse_and_apply(text, FILES)
    assert parsed.well_formed and len(parsed.blocks) == 2
    assert parsed.files["b.py"] == "value = 2\n"


def test_multiple_blocks_are_all_or_nothing():
    text = block("b.py", "value = 0", "value = 1") + block("a.py", "    return 1", "    return 9")
    parsed = parse_and_apply(text, FILES)
    assert parsed.error == "search_not_unique:a.py#1:2"
    assert parsed.files is None, "a failing second block must not leave the first one applied"


def test_blocks_across_two_files():
    text = block("b.py", "value = 0", "value = 3") + block("a.py", "def g():\n    return 1",
                                                           "def g():\n    return 4")
    parsed = parse_and_apply(text, FILES)
    assert parsed.files["b.py"] == "value = 3\n" and "return 4" in parsed.files["a.py"]


def test_crlf_response_matches_lf_file():
    parsed = parse_and_apply(block("b.py", "value = 0", "value = 5").replace("\n", "\r\n"), FILES)
    assert parsed.well_formed, parsed.error
    assert parsed.files["b.py"] == "value = 5\n", "the applied text is normalized to LF"


def test_lf_response_matches_crlf_file():
    parsed = parse_and_apply(block("b.py", "value = 0", "value = 6"), {"b.py": "value = 0\r\n"})
    assert parsed.well_formed and parsed.files["b.py"] == "value = 6\n"


def test_fenced_blocks_and_decorated_filenames_are_accepted():
    for text in (fenced("b.py", "value = 0", "value = 8"),
                 block("`b.py`", "value = 0", "value = 8"),
                 block("**b.py**:", "value = 0", "value = 8")):
        parsed = parse_and_apply(text, FILES)
        assert parsed.well_formed, (text, parsed.error)
        assert parsed.files["b.py"] == "value = 8\n"


def test_prose_around_blocks_is_ignored_but_prose_as_a_filename_is_not():
    good = "Here is the fix.\n\n" + block("b.py", "value = 0", "value = 9") + "\nThat should do it.\n"
    assert parse_and_apply(good, FILES).files["b.py"] == "value = 9\n"
    bad = "I will now edit the file:\n```python\n<<<<<<< SEARCH\nvalue = 0\n=======\nx\n>>>>>>> REPLACE\n```"
    assert parse_and_apply(bad, FILES).error.startswith("unknown_filename:")


def test_non_string_content_is_malformed():
    assert parse_edit_blocks(None, FILES).error == "no_text_content"
    assert parse_edit_blocks({"tool_calls": []}, FILES).error == "no_text_content"


def test_apply_rejects_a_block_for_a_file_it_was_not_given():
    updated, error = apply_edit_blocks((EditBlock("zzz.py", "a\n", "b\n"),), FILES)
    assert updated is None and error == "unknown_filename:zzz.py"


# ---------------------------------------------------------------------------------------------------
# corpus, languages and split
# ---------------------------------------------------------------------------------------------------


def test_available_and_task_ids_for_the_fixture_corpus():
    benchmark = AiderPolyglotBenchmark()
    ok, reason = benchmark.available(context())
    assert ok and reason == ""
    ids = benchmark.task_ids(context())
    assert ids == build_plan(context()).development
    assert all(item.startswith("aider-polyglot/") and item.count("/") == 2 for item in ids)


def test_available_is_false_when_the_corpus_is_absent(tmp_path):
    ok, reason = AiderPolyglotBenchmark().available(context(dataset_root=str(tmp_path)))
    assert not ok and "corpus is absent" in reason


def test_task_ids_raises_benchmark_unavailable_without_a_corpus(tmp_path):
    with pytest.raises(BenchmarkUnavailable):
        AiderPolyglotBenchmark().task_ids(context(dataset_root=str(tmp_path)))


def test_available_rejects_an_unpinned_or_damaged_corpus(tmp_path):
    root = tmp_path / "aider-polyglot"
    shutil.copytree(Path(DATA_ROOT) / "aider-polyglot", root)
    (root / "pin.json").unlink()
    ok, reason = AiderPolyglotBenchmark().available(context(dataset_root=str(tmp_path)))
    assert not ok and "pin.json" in reason
    (root / "pin.json").write_text(json.dumps({"schema_version": 1, "repository": adapter.UPSTREAM_REPOSITORY,
                                               "commit": "not-a-commit", "retrieved_utc": "t",
                                               "languages": ["python"], "toolchains": ["python"]}),
                                   encoding="utf-8")
    ok, reason = AiderPolyglotBenchmark().available(context(dataset_root=str(tmp_path)))
    assert not ok and "commit" in reason


def test_available_refuses_a_language_without_a_baked_toolchain(tmp_path):
    root = tmp_path / "aider-polyglot"
    shutil.copytree(Path(DATA_ROOT) / "aider-polyglot", root)
    pin = json.loads((root / "pin.json").read_text(encoding="utf-8"))
    pin["toolchains"] = ["python"]
    (root / "pin.json").write_text(json.dumps(pin), encoding="utf-8")
    ok, reason = AiderPolyglotBenchmark().available(context(dataset_root=str(tmp_path)))
    assert not ok and "toolchain" in reason and "javascript" in reason


def test_expected_commit_mismatch_refuses():
    ok, reason = AiderPolyglotBenchmark().available(context(expected_commit="a" * 40))
    assert not ok and "not the expected" in reason


def test_default_languages_are_python_and_javascript_and_there_is_no_typescript():
    assert selected_languages({}) == ("javascript", "python")
    assert set(adapter.DEFAULT_LANGUAGES) == {"python", "javascript"}
    assert "typescript" not in adapter.UPSTREAM_LANGUAGES
    assert sum(adapter.UPSTREAM_EXERCISE_COUNTS.values()) == 225
    assert adapter.UPSTREAM_EXERCISE_COUNTS["python"] + adapter.UPSTREAM_EXERCISE_COUNTS["javascript"] == 83
    with pytest.raises(CorpusError):
        selected_languages({"languages": "typescript"})
    with pytest.raises(CorpusError):
        selected_languages({"languages": ["python", "python"]})


def test_language_selection_is_deterministic_and_order_independent():
    assert selected_languages({"languages": "python,javascript"}) == ("javascript", "python")
    assert selected_languages({"languages": ["javascript", "python"]}) == ("javascript", "python")
    single = build_plan(context(languages="python"))
    assert single.languages == ("python",)
    assert all("/python/" in task for task in single.development + single.holdout)


def test_split_is_disjoint_exhaustive_and_seed_stable():
    plan = build_plan(context())
    assert set(plan.development) & set(plan.holdout) == set()
    everything = set(plan.development) | set(plan.holdout)
    assert everything == set(plan.directories)
    assert plan.development == build_plan(context()).development
    assert plan.holdout == build_plan(context()).holdout
    other = build_plan(context(seed=1234))
    assert set(other.development) | set(other.holdout) == everything
    assert set(other.development) & set(other.holdout) == set()


def test_split_covers_every_language_and_respects_the_fraction():
    names = [f"task-{index}" for index in range(40)]
    development, holdout = split_exercises(names, seed=7)
    assert len(development) == 20 and len(holdout) == 20
    assert set(development) & set(holdout) == set() and set(development) | set(holdout) == set(names)
    assert (development, holdout) == split_exercises(list(reversed(names)), seed=7)
    assert development != split_exercises(names, seed=8)[0]
    quarter, rest = split_exercises(names, seed=7, development_fraction=0.25)
    assert len(quarter) == 10 and len(rest) == 30
    with pytest.raises(CorpusError):
        split_exercises(names, seed=7, development_fraction=1.0)


def test_item_set_digest_pairs_two_candidates_on_the_same_items():
    left, right = build_plan(context()), build_plan(context())
    assert left.item_set_digest("development") == right.item_set_digest("development")
    assert left.item_set_digest("development") != left.item_set_digest("holdout")
    assert left.item_set_digest("development") != build_plan(context(seed=9)).item_set_digest("development")


def test_task_ids_for_holdout_are_the_holdout_items_only():
    benchmark = AiderPolyglotBenchmark()
    development = set(benchmark.task_ids(context()))
    holdout = set(benchmark.task_ids(context(split="holdout")))
    assert development and holdout and development & holdout == set()


def test_unsupported_option_is_refused():
    ok, reason = AiderPolyglotBenchmark().available(context(shuffle=True))
    assert not ok and "unsupported option" in reason


def test_exercise_loading_excludes_meta_and_normalizes_newlines(tmp_path):
    plan = build_plan(context())
    exercise = load_exercise(WORD_REVERSE, plan.directories[WORD_REVERSE])
    assert dict(exercise.solution) == {"word_reverse.py": "def reverse(text):\n" + STUB_LINE + "\n"}
    assert list(dict(exercise.tests)) == ["word_reverse_test.py"]
    assert "example" not in json.dumps(exercise.support_files())
    root = tmp_path / "aider-polyglot"
    shutil.copytree(Path(DATA_ROOT) / "aider-polyglot", root)
    target = root / "python" / "exercises" / "practice" / "word-reverse" / "word_reverse.py"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    crlf = load_exercise(WORD_REVERSE, build_plan(context(dataset_root=str(tmp_path))).directories[WORD_REVERSE])
    assert dict(crlf.solution) == dict(exercise.solution), "corpus files are normalized to LF on load"


def test_broken_exercise_manifest_becomes_a_row_not_a_silent_drop(tmp_path):
    root = tmp_path / "aider-polyglot"
    shutil.copytree(Path(DATA_ROOT) / "aider-polyglot", root)
    (root / "python" / "exercises" / "practice" / "word-reverse" / ".meta" / "config.json").write_text(
        json.dumps({"files": {"solution": ["word_reverse.py"]}}), encoding="utf-8")
    rows, transport, client, _ = run_case(responses=[], outcomes=[], dataset_root=str(tmp_path))
    assert [row["task_id"] for row in rows] == [WORD_REVERSE]
    assert rows[0]["status"] == "environment_error" and rows[0]["outcome_status"] == "corpus_error"
    assert rows[0]["score"] == 0.0 and rows[0]["model_evaluated"] is False
    assert transport.requests == [] and client.requests == []


def test_meta_example_is_never_shown_to_the_model():
    plan = build_plan(context())
    task = "aider-polyglot/python/sum-pair"
    exercise = load_exercise(task, plan.directories[task])
    text = json.dumps(build_initial_prompt(exercise))
    assert "left + right" not in text and ".meta" not in text
    assert "sum_pair_test.py" in text, "the test filename is named"
    assert "assertEqual" not in text, "the test bodies are not shown"


# ---------------------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------------------


def test_initial_prompt_states_the_edit_format_and_the_editable_files():
    exercise = load_exercise(WORD_REVERSE, build_plan(context()).directories[WORD_REVERSE])
    messages = build_initial_prompt(exercise)
    assert [item["role"] for item in messages] == ["system", "user"]
    assert all(marker in messages[0]["content"] for marker in
               ("<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE", "exactly once"))
    assert "word_reverse.py" in messages[1]["content"] and STUB_LINE in messages[1]["content"]


def test_repair_prompt_feeds_the_failure_and_forbids_editing_the_tests():
    exercise = load_exercise(WORD_REVERSE, build_plan(context()).directories[WORD_REVERSE])
    messages = build_repair_prompt(exercise, exercise.editable(), "previous reply", "E   AssertionError: x",
                                   kind="tests")
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "previous reply"
    assert "tests are correct and must not be changed" in messages[3]["content"]
    assert "AssertionError: x" in messages[3]["content"]
    formatted = build_repair_prompt(exercise, exercise.editable(), "", "search_not_found:x#0", kind="format")
    assert "could not be applied" in formatted[3]["content"]
    assert "tests are correct and must not be changed" in formatted[3]["content"]
    with pytest.raises(ValueError):
        build_repair_prompt(exercise, exercise.editable(), "", "", kind="other")


# ---------------------------------------------------------------------------------------------------
# two-attempt loop
# ---------------------------------------------------------------------------------------------------


def test_attempt_one_passes():
    rows, transport, client, artifacts = run_case(responses=[REVERSE_EDIT], outcomes=[completed(True)])
    row = rows[0]
    assert row["status"] == "completed" and row["passed"] is True and row["score"] == 1.0
    assert row["first_attempt_success"] is True and row["attempt_1_passed"] is True
    assert row["well_formed"] is True and row["malformed_responses"] == 0 and row["responses"] == 1
    assert row["model_evaluated"] is True and row["synthetic"] is False
    assert len(transport.requests) == 1 and len(client.requests) == 1
    assert "aider-polyglot/python/word-reverse/response-1.json" in artifacts.writes
    assert all(key in row for key in REQUIRED_SAMPLE_KEYS)


def test_attempt_one_fails_then_attempt_two_passes():
    rows, transport, client, _ = run_case(
        responses=[block("word_reverse.py", STUB_LINE, "    return text"),
                   block("word_reverse.py", "    return text", SOLUTION_LINE)],
        outcomes=[completed(False, "AssertionError: 'abc' != 'cba'"), completed(True)])
    row = rows[0]
    assert row["passed"] is True and row["score"] == 1.0
    assert row["first_attempt_success"] is False, "a repaired pass is never a first-attempt pass"
    assert row["attempt_1_passed"] is False and row["well_formed"] is True and row["responses"] == 2
    assert len(client.requests) == 2 and [item.attempt_index for item in client.requests] == [1, 2]
    repair = transport.requests[1]["messages"][-1]["content"]
    assert "AssertionError: 'abc' != 'cba'" in repair and "must not be changed" in repair


def test_attempt_two_edits_the_files_attempt_one_already_changed():
    first = block("word_reverse.py", STUB_LINE, "    return text")
    second = block("word_reverse.py", "    return text", SOLUTION_LINE)
    rows, _, client, _ = run_case(responses=[first, second],
                                  outcomes=[completed(False), completed(True)])
    assert rows[0]["passed"] is True
    assert client.requests[1].patch["word_reverse.py"] == "def reverse(text):\n" + SOLUTION_LINE + "\n"


def test_both_attempts_fail():
    rows, _, client, _ = run_case(responses=[REVERSE_EDIT,
                                             block("word_reverse.py", SOLUTION_LINE, "    return text")],
                                  outcomes=[completed(False), completed(False)])
    row = rows[0]
    assert row["passed"] is False and row["score"] == 0.0 and row["status"] == "completed"
    assert row["outcome_status"] == "failed" and row["model_evaluated"] is True
    assert row["first_attempt_success"] is False and row["well_formed"] is True
    assert len(client.requests) == 2


def test_malformed_attempt_one_is_never_executed_and_asks_for_the_format_again():
    rows, transport, client, _ = run_case(responses=["Sure! Here is the whole file:\n\ndef reverse(t): ...",
                                                     REVERSE_EDIT],
                                          outcomes=[completed(True)])
    row = rows[0]
    assert len(client.requests) == 1 and client.requests[0].attempt_index == 2
    assert row["passed"] is True and row["first_attempt_success"] is False
    assert row["well_formed"] is False and row["malformed_responses"] == 1
    assert row["edit_errors"] == ["no_edit_blocks"]
    assert "could not be applied" in transport.requests[1]["messages"][-1]["content"]


def test_both_responses_malformed_is_a_measured_zero():
    rows, _, client, _ = run_case(responses=["no blocks here", "still no blocks"], outcomes=[])
    row = rows[0]
    assert client.requests == []
    assert row["status"] == "completed" and row["outcome_status"] == "invalid_edit_format"
    assert row["score"] == 0.0 and row["passed"] is False and row["model_evaluated"] is True
    assert row["well_formed"] is False and row["malformed_responses"] == 2
    assert row["edit_errors"] == ["no_edit_blocks", "no_edit_blocks"]


def test_a_malformed_repair_after_a_real_failure_keeps_the_measured_zero():
    rows, _, _, _ = run_case(responses=[REVERSE_EDIT, "I give up"], outcomes=[completed(False)])
    row = rows[0]
    assert row["status"] == "completed" and row["outcome_status"] == "failed"
    assert row["model_evaluated"] is True and row["well_formed"] is False
    assert row["repair_error"].startswith("invalid_edit_format: no_edit_blocks")


def test_transport_failure_on_attempt_one_is_an_environment_error():
    rows, _, client, _ = run_case(responses=[RuntimeError("connection reset")], outcomes=[])
    row = rows[0]
    assert row["status"] == "environment_error" and row["outcome_status"] == "transport_error"
    assert row["model_evaluated"] is False and row["score"] == 0.0 and row["responses"] == 0
    assert row["well_formed"] is None, "a case the model never answered is not a format violation"
    assert client.requests == []


def test_transport_timeout_on_attempt_two_preserves_first_attempt_but_does_not_qualify():
    rows, _, _, _ = run_case(responses=[REVERSE_EDIT, TimeoutError("slow")], outcomes=[completed(False)])
    row = rows[0]
    assert row["status"] == "timeout" and row["model_evaluated"] is False and row["passed"] is False
    assert row["attempt_1_passed"] is False and "TimeoutError" in row["repair_error"]


def test_served_model_mismatch_is_refused():
    class WrongModel(FakeTransport):
        def __call__(self, request):
            response = super().__call__(request)
            response["model"] = "some-other-model"
            return response

    transport = WrongModel(REVERSE_EDIT)
    rows = AiderPolyglotBenchmark(transport=transport,
                                  execution_client=FakeExecutionClient()).run(context())
    assert rows[0]["outcome_status"] == "served_model_mismatch" and rows[0]["model_evaluated"] is False


# ---------------------------------------------------------------------------------------------------
# isolation boundary
# ---------------------------------------------------------------------------------------------------


def test_the_host_execution_guard_is_actually_armed():
    """If this ever stops raising, every other test in this module is measuring nothing."""
    with pytest.raises(AssertionError, match="host execution attempted"):
        subprocess.Popen(["python", "-c", "print(1)"])
    with pytest.raises(AssertionError, match="host execution attempted"):
        subprocess.run(["python", "-c", "print(1)"])


def test_module_imports_no_process_api():
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "os.system", "os.popen", "docker", "pty.spawn"):
        assert forbidden not in source, forbidden


def test_without_an_execution_client_every_row_is_an_environment_error():
    benchmark = AiderPolyglotBenchmark(transport=FakeTransport())
    declared = build_plan(context()).development
    rows = benchmark.run(context(task_ids=declared))
    assert [row["task_id"] for row in rows] == list(declared)
    for row in rows:
        assert row["status"] == "environment_error" and row["outcome_status"] == "no_execution_client"
        assert row["score"] == 0.0 and row["passed"] is False and row["model_evaluated"] is False
        assert "never run on the host" in row["error"]


def test_without_a_transport_every_row_is_an_environment_error():
    rows = AiderPolyglotBenchmark(execution_client=FakeExecutionClient()).run(context())
    assert rows[0]["outcome_status"] == "no_transport" and rows[0]["score"] == 0.0


def test_an_unavailable_execution_namespace_fails_closed():
    client = FakeExecutionClient(namespace=RuntimeError("no namespace published"))
    rows = AiderPolyglotBenchmark(transport=FakeTransport(), execution_client=client).run(context())
    assert rows[0]["outcome_status"] == "no_execution_namespace" and rows[0]["model_evaluated"] is False


def test_the_execution_request_carries_the_patch_the_pinned_tests_and_the_command():
    rows, _, client, _ = run_case(responses=[REVERSE_EDIT], outcomes=[completed(True)])
    request = client.requests[0]
    assert isinstance(request, ExecutionRequest)
    assert request.task_id == WORD_REVERSE and request.language == "python" and request.attempt_index == 1
    assert request.patch == {"word_reverse.py": "def reverse(text):\n" + SOLUTION_LINE + "\n"}
    assert list(request.test_files) == ["word_reverse_test.py"]
    assert request.test_files["word_reverse_test.py"].startswith("import unittest")
    assert request.command[:3] == ("python", "-m", "pytest") and request.command[-1] == "word_reverse_test.py"
    assert request.workdir == "python/word-reverse" and request.timeout_seconds >= 1
    assert request.upstream_commit == read_pin(Path(DATA_ROOT) / "aider-polyglot").commit
    assert request.session_id == "session-1" and len(request.request_id) == 32
    payload = json.loads(json.dumps(request.payload()))
    assert payload["suite"] == "aider-polyglot" and payload["exercise_sha256"] == rows[0]["exercise_sha256"]
    assert request.content_sha256() == ExecutionRequest(**{**request.__dict__,
                                                          "submitted_utc": "later"}).content_sha256()


def test_javascript_requests_carry_the_support_files_and_the_npm_command():
    rows, _, client, _ = run_case(
        responses=[block("title-case.js", "  throw new Error('Remove this statement and implement this "
                                          "function');", "  return text;")],
        outcomes=[completed(True)], task_ids=(TITLE_CASE,))
    assert rows[0]["passed"] is True
    request = client.requests[0]
    assert request.command == ("npm", "test", "--silent")
    assert list(request.support_files) == ["package.json"]
    assert list(request.test_files) == ["title-case.spec.js"]


def test_an_execution_result_for_another_request_is_refused():
    def impostor(request):
        return {"request_id": request.request_id, "request_sha256": "0" * 64, "status": "completed",
                "sample": {"passed": True}, "cleanup_confirmed": True, "abort_campaign": False}

    rows, _, _, _ = run_case(responses=[REVERSE_EDIT, REVERSE_EDIT], outcomes=[impostor, impostor])
    assert rows[0]["outcome_status"] == "execution_integrity"
    assert rows[0]["status"] == "environment_error" and rows[0]["model_evaluated"] is False


def test_unverified_worker_cleanup_aborts_the_run():
    with pytest.raises(BenchmarkAborted):
        run_case(responses=[REVERSE_EDIT], outcomes=[terminal("cleanup-unverified", "worker still alive")])


def test_a_worker_rejection_of_the_model_patch_stays_model_attributable():
    rows, _, _, _ = run_case(responses=[REVERSE_EDIT],
                             outcomes=[terminal("rejected", "patch_outside_allowlist")])
    row = rows[0]
    assert row["status"] == "completed" and row["model_evaluated"] is True and row["score"] == 0.0
    assert row["outcome_status"] == "execution_rejected"


def test_a_worker_environment_error_is_not_charged_to_the_model():
    rows, _, _, _ = run_case(responses=[REVERSE_EDIT], outcomes=[terminal("environment-error", "no image")])
    assert rows[0]["status"] == "environment_error" and rows[0]["model_evaluated"] is False
    assert rows[0]["outcome_status"] == "execution_environment_error"


def test_a_completed_result_without_a_boolean_verdict_is_refused():
    request = _sample_request()
    vague = {"request_id": request.request_id, "request_sha256": request.content_sha256(),
             "status": "completed", "sample": {"passed": "yes"}, "cleanup_confirmed": True,
             "abort_campaign": False}
    assert normalize_execution_result(vague, request).status == "invalid_execution_result"
    vague["sample"] = None
    assert normalize_execution_result(vague, request).status == "invalid_execution_result"
    good = {**vague, "sample": {"passed": True}}
    assert normalize_execution_result(good, request).passed is True


def _sample_request():
    return ExecutionRequest(session_id="s", attempt_id="a" * 32, request_id="b" * 32, task_id=WORD_REVERSE,
                            language="python", exercise_sha256="c" * 64, attempt_index=1,
                            workdir="python/word-reverse", command=("python",), patch={"a.py": "x"},
                            test_files={}, support_files={}, timeout_seconds=10, upstream_commit="0" * 40,
                            submitted_utc="t")


def test_normalize_rejects_unknown_statuses_and_missing_results():
    request = _sample_request()
    assert normalize_execution_result(None, request).status == "invalid_execution_result"
    assert normalize_execution_result({"status": "weird"}, request).status == "invalid_execution_result"


# ---------------------------------------------------------------------------------------------------
# denominator, budget and evidence
# ---------------------------------------------------------------------------------------------------


def test_every_declared_task_produces_a_row_when_one_fails_midway():
    declared = build_plan(context()).development
    transport = FakeTransport(RuntimeError("reset"), "no blocks", "no blocks", RuntimeError("reset"),
                              RuntimeError("reset"))
    client = FakeExecutionClient()
    rows = AiderPolyglotBenchmark(transport=transport, execution_client=client).run(
        context(task_ids=declared))
    assert [row["task_id"] for row in rows] == list(declared)
    assert len({row["task_id"] for row in rows}) == len(declared)
    assert all(row["score"] == 0.0 and row["passed"] is False for row in rows)
    assert all(row["suite"] == "aider-polyglot" and row["suite_revision"] == "aider-polyglot-v1"
               for row in rows)
    assert all(isinstance(row["fixture_hash"], str) and row["fixture_hash"] for row in rows)
    assert len({row["fixture_hash"] for row in rows}) == len(rows)


def test_an_empty_selection_runs_the_whole_declared_split():
    declared = build_plan(context()).development
    transport = FakeTransport(*(["no blocks"] * 2 * len(declared)))
    rows = AiderPolyglotBenchmark(transport=transport, execution_client=FakeExecutionClient()).run(
        context(task_ids=()))
    assert [row["task_id"] for row in rows] == list(declared)


def test_two_candidates_see_the_same_items_for_paired_comparison():
    declared = build_plan(context()).development
    runs = []
    for passing in (True, False):
        client = FakeExecutionClient(*([completed(passing)] * 2 * len(declared)))
        runs.append(AiderPolyglotBenchmark(transport=SolvingTransport(), execution_client=client).run(
            context(task_ids=declared)))
    assert [row["passed"] for row in runs[0]] == [True] * len(declared)
    assert [row["passed"] for row in runs[1]] == [False] * len(declared)
    left, right = ([row["task_id"] for row in item] for item in runs)
    assert left == right == list(declared)
    assert {row["item_set_digest"] for row in runs[0] + runs[1]} == {
        build_plan(context()).item_set_digest("development")}
    pairs = list(zip(runs[0], runs[1]))
    assert all(one["task_id"] == two["task_id"] for one, two in pairs), "rows line up item by item"
    assert all(one["fixture_hash"] == two["fixture_hash"] for one, two in pairs)


def test_an_adapter_bug_becomes_a_row_instead_of_shrinking_the_denominator(monkeypatch):
    declared = build_plan(context()).development
    calls = []

    def explode(text, files):
        calls.append(text)
        raise RuntimeError("parser bug")

    monkeypatch.setattr(adapter, "parse_and_apply", explode)
    rows = AiderPolyglotBenchmark(transport=SolvingTransport(),
                                  execution_client=FakeExecutionClient()).run(context(task_ids=declared))
    assert [row["task_id"] for row in rows] == list(declared)
    assert len(calls) == len(declared), "every task was still attempted"
    for item in rows:
        assert item["status"] == "environment_error" and item["outcome_status"] == "adapter_error"
        assert item["model_evaluated"] is False and item["score"] == 0.0


def test_a_policy_refusal_is_never_swallowed_into_a_row(monkeypatch):
    from llmbench.safety import OperationForbidden

    def refuse(*args, **kwargs):
        raise OperationForbidden("live operations are not authorized")

    monkeypatch.setattr(adapter, "parse_and_apply", refuse)
    with pytest.raises(OperationForbidden):
        run_case(responses=[REVERSE_EDIT], outcomes=[completed(True)])


def test_an_unknown_task_id_is_reported_not_silently_dropped():
    rows, _, _, _ = run_case(responses=[], outcomes=[], task_ids=("aider-polyglot/python/nope",))
    assert rows[0]["outcome_status"] == "unknown_task" and rows[0]["status"] == "environment_error"


def test_a_holdout_task_requested_on_the_development_split_is_refused():
    plan = build_plan(context())
    rows, _, client, _ = run_case(responses=[], outcomes=[], task_ids=(plan.holdout[0],))
    assert rows[0]["outcome_status"] == "unknown_task" and client.requests == []


def test_budget_exhaustion_marks_the_remaining_tasks_and_drops_none():
    declared = build_plan(context()).development
    ticks = [400.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    def remaining():
        return ticks.pop(0) if ticks else 5.0

    transport = SolvingTransport()
    client = FakeExecutionClient(completed(True), completed(True), completed(True))
    rows = AiderPolyglotBenchmark(transport=transport, execution_client=client).run(
        context(task_ids=declared, remaining=remaining))
    assert [row["task_id"] for row in rows] == list(declared)
    marked = [row for row in rows if row["outcome_status"] == "budget_exhausted"]
    assert len(marked) == len(declared) - 1
    assert all(row["status"] == "timeout" and row["score"] == 0.0 and row["model_evaluated"] is False
               for row in marked)
    assert len(transport.requests) <= 2


def test_an_exhausted_budget_callable_that_raises_is_treated_as_exhausted():
    def remaining():
        raise TimeoutError("evaluation wall budget exhausted")

    rows, transport, _, _ = run_case(responses=[], outcomes=[], remaining=remaining)
    assert rows[0]["outcome_status"] == "budget_exhausted" and transport.requests == []


def test_raw_responses_are_persisted_before_they_are_parsed():
    rows, _, _, artifacts = run_case(responses=[REVERSE_EDIT, REVERSE_EDIT],
                                     outcomes=[completed(False), completed(False)])
    names = sorted(artifacts.writes)
    assert "aider-polyglot/python/word-reverse/response-1.json" in names
    assert "aider-polyglot/python/word-reverse/response-2.json" in names
    stored = json.loads(artifacts.writes["aider-polyglot/python/word-reverse/response-1.json"])
    assert stored["choices"][0]["message"]["content"] == REVERSE_EDIT
    assert [item["attempt"] for item in rows[0]["evidence"]["responses"]] == [1, 2]


def test_a_response_that_cannot_be_persisted_aborts_rather_than_being_measured():
    with pytest.raises(BenchmarkAborted):
        run_case(responses=[REVERSE_EDIT], outcomes=[completed(True)], artifacts=MemoryArtifacts(fail=True))


def test_metrics_are_persisted_and_recomputable_from_the_rows():
    rows, _, _, artifacts = run_case(responses=[REVERSE_EDIT], outcomes=[completed(True)])
    stored = json.loads(artifacts.writes["aider-polyglot/metrics.json"])
    assert stored == aggregate_metrics(rows)
    assert stored["pass_rate_1"] == 1.0 and stored["percent_cases_well_formed"] == 100.0


# ---------------------------------------------------------------------------------------------------
# metric aggregation
# ---------------------------------------------------------------------------------------------------


def row(**overrides):
    base = {"suite": "aider-polyglot", "responses": 1, "well_formed": True, "malformed_responses": 0,
            "passed": False, "first_attempt_success": False}
    return {**base, **overrides}


def test_percent_cases_well_formed_counts_cases_not_responses():
    rows = [row(malformed_responses=2, well_formed=False), row(malformed_responses=1, well_formed=False),
            row(), row()]
    metrics = aggregate_metrics(rows)
    assert metrics["cases_total"] == 4 and metrics["cases_answered"] == 4
    assert metrics["cases_well_formed"] == 2 and metrics["cases_malformed"] == 2
    assert metrics["percent_cases_well_formed"] == 50.0
    assert metrics["malformed_responses"] == 3


def test_pass_rates_use_the_full_declared_denominator():
    rows = [row(passed=True, first_attempt_success=True), row(passed=True), row(),
            row(responses=0, well_formed=None)]
    metrics = aggregate_metrics(rows)
    assert metrics["cases_total"] == 4 and metrics["cases_unanswered"] == 1
    assert metrics["pass_rate_1"] == 0.25 and metrics["pass_rate_2"] == 0.5
    assert metrics["percent_cases_well_formed"] == 100.0, "unanswered cases are not format evidence"
    assert metrics["cases_answered"] == 3


def test_metrics_ignore_rows_from_other_suites_and_survive_an_empty_set():
    assert aggregate_metrics([row(suite="coding", passed=True)])["cases_total"] == 0
    empty = aggregate_metrics([])
    assert empty["pass_rate_1"] is None and empty["percent_cases_well_formed"] is None


def test_metrics_over_a_real_mixed_run():
    first, _, _, _ = run_case(responses=[REVERSE_EDIT], outcomes=[completed(True)])
    second, _, _, _ = run_case(responses=["no blocks", REVERSE_EDIT], outcomes=[completed(True)])
    third, _, _, _ = run_case(responses=[REVERSE_EDIT,
                                         block("word_reverse.py", SOLUTION_LINE, "    return text")],
                              outcomes=[completed(False), completed(False)])
    metrics = aggregate_metrics(first + second + third)
    assert metrics["cases_total"] == 3 and metrics["cases_passed_attempt_1"] == 1
    assert metrics["cases_passed_either_attempt"] == 2
    assert metrics["pass_rate_1"] == pytest.approx(1 / 3) and metrics["pass_rate_2"] == pytest.approx(2 / 3)
    assert metrics["percent_cases_well_formed"] == pytest.approx(200 / 3)


def test_the_test_command_is_pinned_only_for_languages_with_a_toolchain():
    exercise = load_exercise(WORD_REVERSE, build_plan(context()).directories[WORD_REVERSE])
    assert pinned_test_command(exercise)[0] == "python"
    for language in ("cpp", "go", "java", "rust"):
        assert adapter.TEST_COMMANDS[language] is None
        with pytest.raises(CorpusError):
            pinned_test_command(SimpleNamespace(language=language, tests=()))


def test_worker_failure_on_repair_does_not_qualify_the_two_attempt_metric():
    rows, _, _, _ = run_case(responses=[REVERSE_EDIT, block("word_reverse.py", SOLUTION_LINE, SOLUTION_LINE + "  # retry")],
                             outcomes=[completed(False), terminal("environment-error", "Docker startup exit 125")])
    row = rows[0]
    assert row["status"] == "environment_error" and row["model_evaluated"] is False
    assert row["attempt_1_passed"] is False and row["score"] == 0
    assert "Docker startup exit 125" in row["repair_error"]


def test_model_attributable_repair_rejection_remains_a_measured_failure():
    rows, _, _, _ = run_case(responses=[REVERSE_EDIT, block("word_reverse.py", SOLUTION_LINE, SOLUTION_LINE + "  # retry")],
                             outcomes=[completed(False), terminal("rejected", "patch_too_large")])
    assert rows[0]["status"] == "completed" and rows[0]["model_evaluated"] is True
    assert not rows[0]["passed"]


def test_completed_envelope_cannot_hide_infrastructure_only_evidence():
    request = _sample_request()
    for flags in ({"status": "environment-error"}, {"model_evaluated": False}):
        raw = {"status": "completed", "sample": {"passed": False, **flags}, "cleanup_confirmed": True}
        outcome = normalize_execution_result(raw, request)
        assert outcome.status == "invalid_execution_result" and outcome.model_attributable is False
