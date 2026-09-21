"""EvalPlus adapter (MBPP+ primary, HumanEval+ smoke test) for the public-benchmark protocol.

Upstream: https://github.com/evalplus/evalplus (Apache-2.0). MBPP+ v0.2.0 is the 378 original MBPP problems
with the ~35x expanded EvalPlus test suite; HumanEval+ v0.1.10 is 164 problems where good local models sit near
90%, so it resolves too few failures to compare configurations and is marked ``smoke_test``. The metric is
pass@1 with greedy decoding: an item passes only when every expanded test of that item passes.

THE CODE-EXECUTION BOUNDARY (read this before changing anything here)
--------------------------------------------------------------------
This module NEVER executes model-generated code. It has no ``subprocess``, no ``exec``/``eval``, no
``importlib``, no Docker and no host fallback. The only thing it does with a generated solution is:

  1. parse it with :func:`ast.parse` (syntax analysis only: ``compile(..., PyCF_ONLY_AST)``, nothing runs), and
  2. put it, as text, into an execution request dict that is handed to ``context.options["execution_client"]``,
     which is the existing host broker / ``DockerWorker`` path (``llmbench.coding.spool`` +
     ``llmbench.coding.broker``), the same isolation that ``coding.runner.run_coding_fixture`` uses.

If no execution client is present in the context, every row is ``environment_error`` with score 0.0. There is
deliberately no other branch: a missing sandbox must cost a measurement, never produce one.

Rows
----
One row per declared task id, always, failures included (``missing_rows`` backfills anything the loop never
reached). Every row carries ``passed_tests``/``required_tests``, an ``outcome_status`` drawn from
``passed``/``assertion_failed``/``exception``/``timeout``/``invalid_output``/``environment_error``/
``budget_exhausted``, and a stable ``paired_key`` so two candidates can be compared item by item (McNemar on
the discordant pairs) instead of only by aggregate percentage.

Splits
------
``development`` and ``holdout`` are disjoint halves of the pinned item set, chosen by a seeded hash rank
(see :func:`split_assignment`): stable for a given (dataset, seed), independent of configuration, so the same
items are used for every candidate — which is what makes the paired comparison legitimate and gives coding a
real holdout for the first time.

Dataset
-------
The pinned JSONL is baked into the evaluator image; nothing is downloaded at run time. ``available()`` returns
``(False, reason)`` when the file is absent, unreadable, the wrong size, or does not hold exactly the pinned
number of well-formed records.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from . import BenchmarkAborted, BenchmarkContext, missing_rows
from ..coding.spool import read_bounded
from ..config import RunMode, canonical_json
from ..evaluations.tools import strict_json_loads

# --------------------------------------------------------------------------------------------------------
# Pins and budgets
# --------------------------------------------------------------------------------------------------------

EVALPLUS_SOURCE = "https://github.com/evalplus/evalplus"
DATASET_SUBDIR = "evalplus"
MAX_DATASET_BYTES = 67_108_864
MAX_SOLUTION_BYTES = 65_536
# Mbpp/462 composes a 800,671-byte module from upstream's own 107 tests; it is the only item over
# the old 262,144 cap. Raised rather than trimming real upstream data, since the cap exists to bound
# what we hand a worker, not to edit the benchmark. Still far below the worker's output budget.
MAX_TEST_MODULE_BYTES = 1_048_576
MAX_PROMPT_BYTES = 32_768
MAX_TRIM_LINES = 40
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30
DEFAULT_MIN_ITEM_SECONDS = 5.0
MAX_OUTPUT_BYTES = 1_048_576
MAX_CONSECUTIVE_EXECUTION_ERRORS = 3
DEVELOPMENT_NUMERATOR, DEVELOPMENT_DENOMINATOR = 1, 2  # development = floor(n * 1/2) items, holdout = the rest

FAILURE_MODES = ("assertion_failed", "exception", "timeout", "invalid_output", "environment_error")
"""The failure vocabulary this adapter reports in ``outcome_status`` (plus ``passed`` and ``budget_exhausted``)."""

MODEL_ATTRIBUTABLE_REJECTIONS = frozenset({"patch_too_large", "patch_outside_allowlist", "solution_too_large"})
"""Worker rejections the model is answerable for: a real solution the sandbox contract still refuses."""

_PROPAGATE = frozenset({"UnsafeCaptureRuntimeState", "OperationForbidden", "OperationDenied", "BenchmarkAborted"})
"""Exception type names that must never become a row: they stop the run the way the coding path stops it."""


@dataclass(frozen=True)
class DatasetPin:
    key: str
    version: str
    filename: str
    upstream_prefix: str
    item_count: int
    revision: str
    smoke_test: bool
    note: str


EVALPLUS_DATASETS: dict[str, DatasetPin] = {
    "mbpp-plus": DatasetPin(
        key="mbpp-plus", version="v0.2.0", filename="MbppPlus-v0.2.0.jsonl", upstream_prefix="Mbpp",
        item_count=378, revision="evalplus/mbpp-plus@v0.2.0", smoke_test=False,
        note="378 MBPP problems with the EvalPlus expanded tests; the usable signal for config comparison"),
    "humaneval-plus": DatasetPin(
        key="humaneval-plus", version="v0.1.10", filename="HumanEvalPlus-v0.1.10.jsonl",
        upstream_prefix="HumanEval", item_count=164, revision="evalplus/humaneval-plus@v0.1.10", smoke_test=True,
        note="164 problems; good local models pass ~90%, so ~16 failures cannot separate configurations"),
}

# --------------------------------------------------------------------------------------------------------
# Pinned dataset records
# --------------------------------------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEST_ID = re.compile(r"^(?:base|plus)/[0-9]{1,6}$")
_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")
_RECORD_KEYS = {"task_id", "entry_point", "prompt", "tests", "signature", "test_setup"}
_REQUIRED_RECORD_KEYS = {"task_id", "entry_point", "prompt", "tests"}


@dataclass(frozen=True)
class EvalPlusTest:
    test_id: str
    kind: str
    code: str


@dataclass(frozen=True)
class EvalPlusItem:
    task_id: str
    upstream_id: str
    upstream_index: int
    entry_point: str
    prompt: str
    signature: str
    test_setup: str
    tests: tuple[EvalPlusTest, ...]
    item_sha256: str


@dataclass(frozen=True)
class LoadedDataset:
    pin: DatasetPin
    path: str
    sha256: str
    items: tuple[EvalPlusItem, ...]

    def by_task_id(self) -> dict[str, EvalPlusItem]:
        return {item.task_id: item for item in self.items}


class DatasetError(ValueError):
    """The pinned dataset is absent, unreadable or does not match the pin. Reported by ``available()``."""


class ExecutionClient(Protocol):
    """The evaluator-side handle on the existing isolated worker path. The adapter owns no other way to run code."""

    def execute(self, request: dict[str, Any]) -> Any:
        """Submit one execution request and return a ``CodingJobResult``-shaped mapping or object."""


def _item_identity(record: Mapping[str, Any], pin: DatasetPin) -> str:
    payload = {"dataset": pin.key, "version": pin.version, "task_id": record["task_id"],
               "entry_point": record["entry_point"], "prompt": record["prompt"],
               "signature": record.get("signature", ""), "test_setup": record.get("test_setup", ""),
               "tests": [[test["test_id"], test["kind"], test["code"]] for test in record["tests"]]}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _text(value: Any, *, name: str, max_bytes: int, allow_empty: bool = False) -> str:
    if type(value) is not str or (not value and not allow_empty):
        raise DatasetError(f"{name} must be a nonempty string")
    if "\x00" in value:
        raise DatasetError(f"{name} contains a NUL byte")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:  # lone surrogates cannot cross the spool contract
        raise DatasetError(f"{name} is not encodable UTF-8: {exc}") from exc
    if len(encoded) > max_bytes:
        raise DatasetError(f"{name} exceeds {max_bytes} bytes")
    return value


def _parse_record(raw: Mapping[str, Any], pin: DatasetPin, line: int) -> EvalPlusItem:
    if type(raw) is not dict:
        raise DatasetError(f"line {line}: record must be a JSON object")
    unknown = set(raw) - _RECORD_KEYS
    if unknown or not _REQUIRED_RECORD_KEYS <= set(raw):
        raise DatasetError(f"line {line}: record keys must be exactly {sorted(_REQUIRED_RECORD_KEYS)} "
                           f"(optional: signature, test_setup); unexpected={sorted(unknown)}")
    upstream_id = _text(raw["task_id"], name=f"line {line}: task_id", max_bytes=128)
    prefix, separator, index = upstream_id.partition("/")
    if prefix != pin.upstream_prefix or not separator or not index.isdigit():
        raise DatasetError(f"line {line}: task_id must be {pin.upstream_prefix}/<number>, got {upstream_id!r}")
    entry_point = _text(raw["entry_point"], name=f"line {line}: entry_point", max_bytes=128)
    if not _IDENTIFIER.match(entry_point):
        raise DatasetError(f"line {line}: entry_point must be a plain Python identifier")
    prompt = _text(raw["prompt"], name=f"line {line}: prompt", max_bytes=MAX_PROMPT_BYTES)
    signature = _text(raw.get("signature", ""), name=f"line {line}: signature", max_bytes=512, allow_empty=True)
    setup = _text(raw.get("test_setup", ""), name=f"line {line}: test_setup", max_bytes=8192, allow_empty=True)
    tests = raw["tests"]
    if type(tests) is not list or not tests:
        raise DatasetError(f"line {line}: tests must be a nonempty list")
    parsed, seen = [], set()
    for position, test in enumerate(tests):
        label = f"line {line}: tests[{position}]"
        if type(test) is not dict or set(test) != {"test_id", "kind", "code"}:
            raise DatasetError(f"{label} must be an object with test_id, kind and code")
        test_id = _text(test["test_id"], name=f"{label}.test_id", max_bytes=64)
        if not _TEST_ID.match(test_id) or test_id in seen:
            raise DatasetError(f"{label}.test_id must be a unique base/<n> or plus/<n> identifier")
        seen.add(test_id)
        if test["kind"] not in {"base", "plus"} or not test_id.startswith(test["kind"] + "/"):
            raise DatasetError(f"{label}.kind must be 'base' or 'plus' and agree with test_id")
        parsed.append(EvalPlusTest(test_id, test["kind"],
                                   _text(test["code"], name=f"{label}.code", max_bytes=MAX_TEST_MODULE_BYTES)))
    item = EvalPlusItem(task_id=f"evalplus/{pin.key}/{upstream_id}", upstream_id=upstream_id,
                        upstream_index=int(index), entry_point=entry_point, prompt=prompt, signature=signature,
                        test_setup=setup, tests=tuple(parsed), item_sha256=_item_identity(raw, pin))
    module = build_test_module(item)
    if len(module.encode("utf-8")) > MAX_TEST_MODULE_BYTES:
        raise DatasetError(f"line {line}: composed test module exceeds {MAX_TEST_MODULE_BYTES} bytes")
    if not _parses(module):
        raise DatasetError(f"line {line}: composed test module does not parse; the pinned tests are malformed")
    return item


def load_dataset(path: str | Path, pin: DatasetPin, *, max_bytes: int = MAX_DATASET_BYTES) -> LoadedDataset:
    """Read, hash and fully validate the baked JSONL. Raises ``DatasetError``; never partially accepts a file."""
    target = Path(path)
    try:
        raw = read_bounded(target, max_bytes)
    except FileNotFoundError as exc:
        raise DatasetError(f"pinned dataset is absent: {target}") from exc
    except (OSError, ValueError) as exc:
        raise DatasetError(f"pinned dataset is unusable ({type(exc).__name__}): {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = target.with_name(target.name + ".sha256")
    if sidecar.exists():
        try:
            declared = read_bounded(sidecar, 1024).decode("ascii").split()[0].strip().lower()
        except (OSError, ValueError, UnicodeDecodeError, IndexError) as exc:
            raise DatasetError(f"dataset checksum sidecar is unreadable: {exc}") from exc
        if declared != digest:
            raise DatasetError(f"dataset sha256 {digest} does not match the pinned sidecar {declared}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"pinned dataset is not UTF-8: {exc}") from exc
    items, seen = [], set()
    for line, content in enumerate(text.splitlines(), start=1):
        if not content.strip():
            continue
        try:
            record = strict_json_loads(content)
        except (ValueError, RecursionError) as exc:
            raise DatasetError(f"line {line}: invalid JSON ({exc})") from exc
        item = _parse_record(record, pin, line)
        if item.task_id in seen:
            raise DatasetError(f"line {line}: duplicate task id {item.task_id}")
        seen.add(item.task_id)
        items.append(item)
    if len(items) != pin.item_count:
        raise DatasetError(f"{pin.key} {pin.version} pins {pin.item_count} items; this file holds {len(items)}")
    ordered = tuple(sorted(items, key=lambda entry: (entry.upstream_index, entry.task_id)))
    return LoadedDataset(pin=pin, path=str(target), sha256=digest, items=ordered)


# --------------------------------------------------------------------------------------------------------
# Deterministic split
# --------------------------------------------------------------------------------------------------------

def _rank(task_id: str, *, dataset_key: str, seed: int) -> str:
    material = f"evalplus-split|{dataset_key}|{int(seed)}|{task_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def split_assignment(items: tuple[EvalPlusItem, ...], *, dataset_key: str, seed: int) -> dict[str, tuple[str, ...]]:
    """Disjoint development/holdout halves, ranked by ``sha256(dataset|seed|task_id)``.

    The rank depends only on the pinned task id, the dataset key and the seed, so every candidate in a campaign
    sees the same items in the same split — the precondition for an item-by-item (paired) comparison. The first
    ``floor(n/2)`` ranked ids are ``development``; the remainder is ``holdout``. Each list is returned in the
    canonical upstream order, and the two lists never intersect.
    """
    ranked = sorted(items, key=lambda item: (_rank(item.task_id, dataset_key=dataset_key, seed=seed), item.task_id))
    cut = len(ranked) * DEVELOPMENT_NUMERATOR // DEVELOPMENT_DENOMINATOR
    order = {item.task_id: (item.upstream_index, item.task_id) for item in items}
    development = tuple(sorted((item.task_id for item in ranked[:cut]), key=order.__getitem__))
    holdout = tuple(sorted((item.task_id for item in ranked[cut:]), key=order.__getitem__))
    return {"development": development, "holdout": holdout,
            "rank_order": tuple(item.task_id for item in ranked)}


# --------------------------------------------------------------------------------------------------------
# Prompting and deterministic extraction (no execution anywhere in this section)
# --------------------------------------------------------------------------------------------------------

INSTRUCT_PREFIX = ("Please provide a self-contained Python script that solves the following problem in a "
                   "markdown code block:")
SYSTEM_PROMPT = ("You are an expert Python programmer. Answer with exactly one markdown ```python code block "
                 "holding a complete, self-contained solution. Do not add tests, example calls, explanations "
                 "or a second code block.")

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)\r?\n?[ \t]*```", re.S)
_PYTHON_FENCE_LANGUAGES = frozenset({"", "python", "py", "python3"})
_CODE_START = re.compile(r"^(?:def |async def |class |import |from |@)")


def build_messages(item: EvalPlusItem) -> list[dict[str, str]]:
    """EvalPlus's instruct convention: the task prompt verbatim plus the required function signature."""
    required = item.signature or f"def {item.entry_point}(...)"
    user = (f"{INSTRUCT_PREFIX}\n```\n{item.prompt}\n```\n"
            f"The script must define `{required}` at module level under exactly the name "
            f"`{item.entry_point}`, and may define any helpers it needs. Return only the code block.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Extraction:
    source: str | None
    method: str | None
    error: str | None


def _parses(source: str) -> ast.Module | None:
    """Syntax analysis only. ``ast.parse`` compiles to an AST (``PyCF_ONLY_AST``); nothing is ever executed."""
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def _defines(source: str, entry_point: str) -> str | None:
    """Return the accepted source (raw, else dedented) when it declares ``entry_point`` at module level."""
    candidates = [source]
    dedented = textwrap.dedent(source)
    if dedented != source:
        candidates.append(dedented)
    for candidate in candidates:
        tree = _parses(candidate)
        if tree is not None and any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    and node.name == entry_point for node in tree.body):
            return candidate
    return None


def _finalize(source: str, method: str) -> Extraction:
    body = source if source.endswith("\n") else source + "\n"
    if len(body.encode("utf-8")) > MAX_SOLUTION_BYTES:
        return Extraction(None, None, "solution_too_large")
    return Extraction(body, method, None)


def extract_solution(content: Any, entry_point: str) -> Extraction:
    """Deterministically recover the submitted module. Ambiguity is refused, never resolved by guessing.

    Accepted shapes, in this order:
      * fenced: exactly one python-tagged (or untagged) fenced block that defines ``entry_point`` at module
        level; several identical such blocks count as one, several different ones are ``ambiguous_code_blocks``;
      * bare module: the whole response parses and defines ``entry_point``;
      * prose-wrapped: the text from the first top-level code line onwards parses once a bounded number of
        trailing lines is dropped (the first prefix that parses wins, so nothing before it is discarded).

    Everything else returns an error string; the caller scores that row 0 with ``outcome_status`` ``invalid_output``.
    """
    if type(content) is not str or not content.strip():
        return Extraction(None, None, "no_text_content")
    if "\x00" in content:
        return Extraction(None, None, "invalid_characters")
    text = content.replace("\r\n", "\n")
    if "```" in text:
        return _from_fences(text, entry_point)
    return _from_plain_text(text, entry_point)


def _from_fences(text: str, entry_point: str) -> Extraction:
    blocks = _FENCE.findall(text)
    if not blocks or text.count("```") != 2 * len(blocks):
        return Extraction(None, None, "unterminated_or_nested_fence")
    python_blocks = [body for language, body in blocks if language.lower() in _PYTHON_FENCE_LANGUAGES]
    if not python_blocks:
        return Extraction(None, None, "no_python_code_block")
    accepted = [source for source in (_defines(body, entry_point) for body in python_blocks) if source is not None]
    if not accepted:
        if any(_parses(body) is not None for body in python_blocks):
            return Extraction(None, None, "entry_point_missing")
        return Extraction(None, None, "code_block_does_not_parse")
    if len({source.strip() for source in accepted}) > 1:
        return Extraction(None, None, "ambiguous_code_blocks")
    return _finalize(accepted[0], "fenced")


def _from_plain_text(text: str, entry_point: str) -> Extraction:
    whole = _defines(text, entry_point)
    if whole is not None:
        return _finalize(whole, "bare-module")
    lines = text.split("\n")
    start = next((index for index, line in enumerate(lines) if _CODE_START.match(line)), None)
    if start is None:
        return Extraction(None, None, "no_code_found")
    body = lines[start:]
    for dropped in range(min(len(body), MAX_TRIM_LINES + 1)):
        candidate = "\n".join(body[:len(body) - dropped])
        accepted = _defines(candidate, entry_point)
        if accepted is not None:
            return _finalize(accepted, "prose-trimmed")
    return Extraction(None, None, "code_does_not_parse")


# --------------------------------------------------------------------------------------------------------
# The execution request handed to the existing isolated worker path
# --------------------------------------------------------------------------------------------------------

TEST_MODULE_DOC = ("Pinned EvalPlus tests, composed by the adapter and executed only inside the isolated "
                   "worker container. The evaluator never imports or runs this file.")


def build_test_module(item: EvalPlusItem) -> str:
    """One pinned test per function, plus ``LLMBENCH_TESTS`` so the worker can report per-test outcomes."""
    lines = [f'"""{TEST_MODULE_DOC}"""', f"from solution import {item.entry_point}", ""]
    if item.test_setup:
        lines.extend([item.test_setup, ""])
    names = []
    for index, test in enumerate(item.tests):
        name = f"llmbench_test_{index:04d}"
        names.append((test.test_id, name))
        lines.extend([f"def {name}():", textwrap.indent(test.code, "    "), ""])
    entries = ", ".join(f"({test_id!r}, {name!r})" for test_id, name in names)
    lines.append(f"LLMBENCH_TESTS = ({entries},)")
    return "\n".join(lines) + "\n"


def execution_request(item: EvalPlusItem, solution: str, *, dataset: LoadedDataset, split: str, seed: int,
                      timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS) -> dict[str, Any]:
    """The bounded, JSON-only request the host worker path receives. It carries text, never a command."""
    return {
        "schema_version": 1,
        "kind": "benchmark-item",
        "benchmark_id": "evalplus",
        "benchmark_revision": dataset.pin.revision,
        "dataset": dataset.pin.key,
        "dataset_sha256": dataset.sha256,
        "task_id": item.task_id,
        "upstream_task_id": item.upstream_id,
        "item_sha256": item.item_sha256,
        "language": "python",
        "entrypoint": f"solution:{item.entry_point}",
        "files": {"solution.py": solution, "evalplus_tests.py": build_test_module(item)},
        "test_ids": [test.test_id for test in item.tests],
        "required_tests": len(item.tests),
        "split": split,
        "seed": int(seed),
        "attempt_index": 1,
        "timeout_seconds": int(timeout_seconds),
        "max_output_bytes": MAX_OUTPUT_BYTES,
    }


# --------------------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------------------

def _must_propagate(exc: BaseException) -> bool:
    """Permission denials and capture aborts stop the run; they never become a scored row."""
    return type(exc).__name__ in _PROPAGATE


def _field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=str, indent=1).encode("utf-8")


def _remaining(context: BenchmarkContext) -> float:
    try:
        value = float(context.remaining_seconds())
    except TimeoutError:
        return 0.0
    except Exception:  # a broken budget callable must not silently grant unlimited time
        return 0.0
    return value if math.isfinite(value) else 0.0


class EvalPlusAdapter:
    """Protocol-conforming adapter. ``dataset`` selects the pin; ``benchmark_id`` lets a caller register variants."""

    category = "coding"

    def __init__(self, dataset: str = "mbpp-plus", *, benchmark_id: str = "evalplus") -> None:
        if dataset not in EVALPLUS_DATASETS:
            raise ValueError(f"unknown EvalPlus dataset: {dataset}")
        self.dataset = EVALPLUS_DATASETS[dataset]
        self.benchmark_id = benchmark_id
        self.revision = self.dataset.revision
        self.smoke_test = self.dataset.smoke_test
        self.source = EVALPLUS_SOURCE
        self._cache: dict[tuple[str, int, int], LoadedDataset] = {}

    # -- dataset -------------------------------------------------------------------------------------------

    def dataset_path(self, context: BenchmarkContext) -> Path:
        return Path(context.dataset_root) / DATASET_SUBDIR / self.dataset.filename

    def _load(self, context: BenchmarkContext) -> LoadedDataset:
        path = self.dataset_path(context)
        try:
            info = path.stat()
            key = (str(path), info.st_size, info.st_mtime_ns)
        except OSError as exc:
            raise DatasetError(f"pinned dataset is absent: {path} ({type(exc).__name__})") from exc
        cached = self._cache.get(key)
        if cached is None:
            cached = load_dataset(path, self.dataset)
            self._cache[key] = cached
        return cached

    def available(self, context: BenchmarkContext) -> tuple[bool, str]:
        try:
            dataset = self._load(context)
        except Exception as exc:
            return False, f"{self.dataset.key} {self.dataset.version} unavailable: {exc}"
        return True, (f"{self.dataset.key} {self.dataset.version}: {len(dataset.items)} items at "
                      f"{dataset.path} (sha256 {dataset.sha256[:16]}...)")

    # -- task selection ------------------------------------------------------------------------------------

    def splits(self, context: BenchmarkContext) -> dict[str, tuple[str, ...]]:
        dataset = self._load(context)
        return split_assignment(dataset.items, dataset_key=self.dataset.key, seed=context.seed)

    def task_ids(self, context: BenchmarkContext) -> tuple[str, ...]:
        """The deterministic item set for ``context.split`` and ``context.seed``, in canonical upstream order."""
        if context.split not in {"development", "holdout"}:
            raise ValueError(f"unknown split: {context.split}")
        assignment = self.splits(context)
        selected = assignment[context.split]
        limit = self._option(context, "item_limit", None)
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError("item_limit must be a positive integer")
            # Nested subsets: the first `limit` items of the split's seeded rank, so a smaller budget still
            # gives every candidate the same items.
            members = set(selected)
            kept = set([task_id for task_id in assignment["rank_order"] if task_id in members][:limit])
            selected = tuple(task_id for task_id in selected if task_id in kept)
        return selected

    # -- options -------------------------------------------------------------------------------------------

    @staticmethod
    def _option(context: BenchmarkContext, name: str, default: Any) -> Any:
        options = context.options or {}
        return options.get(name, default)

    # -- run -----------------------------------------------------------------------------------------------

    def run(self, context: BenchmarkContext) -> list[dict[str, Any]]:
        declared = tuple(context.task_ids)
        ok, reason = self.available(context)
        if context.split not in {"development", "holdout"}:
            ok, reason = False, f"unknown split {context.split!r}: only development and holdout exist"
        if not ok:
            return self._unrun(declared, [], "environment_error", "environment_error", context.split, reason)
        dataset = self._load(context)
        assignment = self.splits(context)
        declared = declared or self.task_ids(context)
        allowed = set(assignment[context.split])
        catalog = dataset.by_task_id()
        client = self._option(context, "execution_client", None)
        completion = self._option(context, "completion", None)
        minimum = float(self._option(context, "min_item_seconds", DEFAULT_MIN_ITEM_SECONDS))
        produced: list[dict[str, Any]] = []
        halted: tuple[str, str] | None = None
        consecutive_errors = 0
        for task_id in declared:
            if halted is not None:
                break
            item = catalog.get(task_id)
            if item is None or task_id not in allowed:
                produced.append(self._row(context, dataset, item, task_id, status="environment_error",
                                          outcome="environment_error", model_evaluated=False,
                                          error=("task is not part of this benchmark" if item is None else
                                                 f"task belongs to the other split, not {context.split}")))
                continue
            if completion is None:
                produced.append(self._row(context, dataset, item, task_id, status="environment_error",
                                          outcome="environment_error", model_evaluated=False,
                                          error="no completion callable in context.options['completion']"))
                continue
            if client is None:
                produced.append(self._row(context, dataset, item, task_id, status="environment_error",
                                          outcome="environment_error", model_evaluated=False,
                                          error="no execution client: generated code is never run on the host"))
                continue
            if _remaining(context) <= minimum:
                halted = ("budget_exhausted", f"wall budget below the {minimum:g}s per-item floor")
                break
            row = self._run_item(context, dataset, item, completion, client)
            produced.append(row)
            consecutive_errors = consecutive_errors + 1 if row.get("executor_error") else 0
            if consecutive_errors >= MAX_CONSECUTIVE_EXECUTION_ERRORS:
                halted = ("environment_error",
                          f"the execution client failed {consecutive_errors} times in a row; stopping")
        outcome, reason = halted or ("environment_error", "task was never reached")
        rows = produced + self._unrun(declared, produced, "environment_error", outcome, context.split, reason)
        order = {task_id: index for index, task_id in enumerate(declared)}
        rows.sort(key=lambda row: order.get(row["task_id"], len(order)))
        return rows

    # -- rows ----------------------------------------------------------------------------------------------

    def _unrun(self, declared, produced, status: str, outcome: str, split: str, reason: str) -> list[dict]:
        rows = missing_rows(declared, produced, suite=self.benchmark_id, revision=self.revision,
                            category=self.category, split=split, status=status, reason=reason)
        return [{**row, "outcome_status": outcome, "dataset": self.dataset.key, "metric": "pass@1",
                 "smoke_test": self.smoke_test, "passed_tests": 0, "required_tests": None,
                 "paired_key": _paired_key(row["task_id"]), "paired_eligible": False,
                 "execution": "isolated-worker-broker"} for row in rows]

    def _row(self, context: BenchmarkContext, dataset: LoadedDataset, item: EvalPlusItem | None, task_id: str, *,
             status: str, outcome: str, model_evaluated: bool, score: float = 0.0, passed: bool = False,
             **extra: Any) -> dict[str, Any]:
        row = {"task_id": task_id, "suite": self.benchmark_id, "suite_revision": self.revision,
               "category": self.category, "split": context.split, "seed": context.seed,
               "status": status, "outcome_status": outcome, "score": float(score), "passed": bool(passed),
               "model_evaluated": bool(model_evaluated), "synthetic": False, "metric": "pass@1",
               "dataset": self.dataset.key, "dataset_sha256": dataset.sha256, "smoke_test": self.smoke_test,
               "execution": "isolated-worker-broker", "paired_key": _paired_key(task_id),
               "paired_eligible": bool(model_evaluated), "passed_tests": 0,
               "required_tests": len(item.tests) if item is not None else None}
        if item is not None:
            row.update(upstream_task_id=item.upstream_id, entry_point=item.entry_point,
                       item_sha256=item.item_sha256)
        row.update(extra)
        return row

    # -- one item ------------------------------------------------------------------------------------------

    def _run_item(self, context: BenchmarkContext, dataset: LoadedDataset, item: EvalPlusItem,
                  completion: Any, client: Any) -> dict[str, Any]:
        slug = _SLUG.sub("-", item.task_id)
        try:
            context.session_lock.check("inference", RunMode.LIVE)
        except Exception as exc:
            if _must_propagate(exc):
                raise
            return self._row(context, dataset, item, item.task_id, status="environment_error",
                             outcome="environment_error", model_evaluated=False,
                             error=f"{type(exc).__name__}: {exc}")
        try:
            response = self._ask(context, completion, build_messages(item))
        except (asyncio.TimeoutError, TimeoutError) as exc:
            return self._row(context, dataset, item, item.task_id, status="timeout", outcome="timeout",
                             model_evaluated=False, timeout_stage="generation",
                             error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            if _must_propagate(exc):
                raise
            return self._row(context, dataset, item, item.task_id, status="environment_error",
                             outcome="environment_error", model_evaluated=False,
                             error=f"{type(exc).__name__}: {exc}")
        raw = _dump(response)
        try:
            context.artifacts.write(f"evalplus/{slug}/response.json", raw)
        except Exception as exc:  # evidence that cannot be persisted stops the run, as in the coding path
            raise BenchmarkAborted(f"evalplus response evidence could not be persisted for {item.task_id}: "
                                   f"{type(exc).__name__}: {exc}") from exc
        evidence = {"response_sha256": hashlib.sha256(raw).hexdigest()}
        served = _served_model(response)
        if served is not None and served != context.model_alias:
            return self._row(context, dataset, item, item.task_id, status="environment_error",
                             outcome="environment_error", model_evaluated=False, served_model=served,
                             error=f"served model {served!r} is not the candidate alias {context.model_alias!r}",
                             **evidence)
        extraction = extract_solution(_content(response), item.entry_point)
        evidence["extraction_method"] = extraction.method
        if extraction.source is None:
            return self._row(context, dataset, item, item.task_id, status="invalid_output",
                             outcome="invalid_output", model_evaluated=True,
                             extraction_error=extraction.error, **evidence)
        evidence["solution_sha256"] = hashlib.sha256(extraction.source.encode("utf-8")).hexdigest()
        timeout = _bounded_int(self._option(context, "execution_timeout_seconds",
                                            DEFAULT_EXECUTION_TIMEOUT_SECONDS), 1, 600)
        request = execution_request(item, extraction.source, dataset=dataset, split=context.split,
                                    seed=context.seed, timeout_seconds=timeout)
        try:
            result = client.execute(request)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            return self._row(context, dataset, item, item.task_id, status="timeout", outcome="timeout",
                             model_evaluated=False, timeout_stage="worker", executor_error=True,
                             error=f"{type(exc).__name__}: {exc}", **evidence)
        except Exception as exc:
            if _must_propagate(exc):
                raise
            return self._row(context, dataset, item, item.task_id, status="environment_error",
                             outcome="environment_error", model_evaluated=False, executor_error=True,
                             error=f"{type(exc).__name__}: {exc}", **evidence)
        return self._score(context, dataset, item, result, evidence)

    def _ask(self, context: BenchmarkContext, completion: Any, messages: list[dict[str, str]]) -> Any:
        generation = context.generation
        request: dict[str, Any] = {"model": context.model_alias, "messages": messages,
                                   "max_tokens": getattr(generation, "max_output_tokens", 512),
                                   "temperature": getattr(generation, "temperature", 0),
                                   "top_p": getattr(generation, "top_p", 1),
                                   "seed": getattr(generation, "seed", context.seed)}
        top_k = getattr(generation, "top_k", None)
        if top_k is not None:
            request["top_k"] = top_k
        value = completion(request)
        if inspect.isawaitable(value):
            budget = _remaining(context)
            if budget <= 0:
                close = getattr(value, "close", None)
                if callable(close):
                    close()
                raise TimeoutError("wall budget exhausted before the request could be awaited")
            return asyncio.run(asyncio.wait_for(value, timeout=budget))
        return value

    # -- scoring -------------------------------------------------------------------------------------------

    def _score(self, context: BenchmarkContext, dataset: LoadedDataset, item: EvalPlusItem, result: Any,
               evidence: dict[str, Any]) -> dict[str, Any]:
        """pass@1 from the worker's structured result. A claim is only accepted when the counts support it."""
        status = _field(result, "status")
        reason = _field(result, "failure_reason")
        info = {"worker_status": status, "trace_sha256": _field(result, "trace_sha256"),
                "cleanup_confirmed": bool(_field(result, "cleanup_confirmed", False))}
        if _field(result, "abort_campaign", False) or status == "cleanup-unverified":
            raise BenchmarkAborted(f"evalplus worker reported unverified cleanup for {item.task_id}: {reason}")
        if status != "completed":
            if status == "timeout":
                return self._row(context, dataset, item, item.task_id, status="timeout", outcome="timeout",
                                 model_evaluated=True, timeout_stage="execution", error=reason, **evidence, **info)
            if status == "rejected" and reason in MODEL_ATTRIBUTABLE_REJECTIONS:
                return self._row(context, dataset, item, item.task_id, status="invalid_output",
                                 outcome="invalid_output", model_evaluated=True, error=reason, **evidence, **info)
            return self._row(context, dataset, item, item.task_id, status="environment_error",
                             outcome="environment_error", model_evaluated=False, executor_error=True,
                             error=reason or f"worker status {status!r}", **evidence, **info)
        sample = _field(result, "sample")
        checked = _validate_sample(sample, required=len(item.tests))
        if checked.get("error"):
            return self._row(context, dataset, item, item.task_id, status="environment_error",
                             outcome="environment_error", model_evaluated=False, executor_error=True,
                             error=f"worker result violates the contract: {checked['error']}", **evidence, **info)
        passed_tests, required = checked["passed_tests"], checked["required_tests"]
        detail = {"passed_tests": passed_tests, "required_tests": required,
                  "failed_test_ids": checked["failed_test_ids"], "compile_ok": checked["compile_ok"],
                  "failure_detail": checked["failure_detail"]}
        if checked["passed"]:
            return self._row(context, dataset, item, item.task_id, status="completed", outcome="passed",
                             model_evaluated=True, score=1.0, passed=True, **evidence, **info, **detail)
        outcome = checked["failure_kind"]
        status_value = "timeout" if outcome == "timeout" else "completed"
        return self._row(context, dataset, item, item.task_id, status=status_value, outcome=outcome,
                         model_evaluated=True, **evidence, **info, **detail)


def _paired_key(task_id: str) -> str:
    return f"pair:{task_id}"


def _bounded_int(value: Any, low: int, high: int) -> int:
    if type(value) is not int or value < low or value > high:
        return max(low, min(high, DEFAULT_EXECUTION_TIMEOUT_SECONDS))
    return value


def _content(response: Any) -> Any:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _served_model(response: Any) -> str | None:
    value = response.get("model") if isinstance(response, Mapping) else None
    return value if type(value) is str and value else None


def _validate_sample(sample: Any, *, required: int) -> dict[str, Any]:
    """Accept only a coherent worker sample. Counts decide pass@1; a bare 'passed' claim is never enough."""
    if not isinstance(sample, Mapping):
        return {"error": "a completed result must carry a sample object"}
    if sample.get("status") == "environment-error" or sample.get("model_evaluated") is False:
        return {"error": "completed envelope contains infrastructure-only evidence"}
    passed_tests, declared = sample.get("passed_tests"), sample.get("required_tests", required)
    compile_ok = sample.get("compile_ok", True)
    failure_kind = sample.get("failure_kind")
    if type(passed_tests) is not int or type(declared) is not int or type(compile_ok) is not bool:
        return {"error": "passed_tests/required_tests must be integers and compile_ok a boolean"}
    if declared != required:
        return {"error": f"worker ran {declared} tests; the pinned item declares {required}"}
    if not 0 <= passed_tests <= required:
        return {"error": f"passed_tests {passed_tests} is outside 0..{required}"}
    if failure_kind is not None and failure_kind not in {"assertion", "exception", "timeout"}:
        return {"error": f"unknown failure_kind {failure_kind!r}"}
    complete = passed_tests == required and compile_ok
    if complete and failure_kind is not None:
        return {"error": "every test passed yet a failure_kind was reported"}
    if not complete and failure_kind is None:
        return {"error": "a failed item must name its failure_kind"}
    failed = sample.get("failed_test_ids") or []
    if not isinstance(failed, (list, tuple)) or any(type(value) is not str for value in failed):
        return {"error": "failed_test_ids must be a list of strings"}
    kind = {"assertion": "assertion_failed", "exception": "exception", "timeout": "timeout"}.get(failure_kind)
    detail = sample.get("failure_detail")
    return {"error": None, "passed": complete, "passed_tests": passed_tests, "required_tests": required,
            "failed_test_ids": list(failed)[:64], "compile_ok": compile_ok, "failure_kind": kind,
            "failure_detail": detail if type(detail) is str else None}


def mbpp_plus() -> EvalPlusAdapter:
    """The comparison-grade adapter: MBPP+ v0.2.0, 378 items."""
    return EvalPlusAdapter("mbpp-plus")


def humaneval_plus() -> EvalPlusAdapter:
    """Smoke test only: HumanEval+ v0.1.10 resolves too few failures to compare configurations."""
    return EvalPlusAdapter("humaneval-plus", benchmark_id="evalplus")
