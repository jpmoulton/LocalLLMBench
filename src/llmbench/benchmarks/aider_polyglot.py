"""Aider Polyglot: hard Exercism exercises edited through search/replace blocks, two attempts each.

Upstream is ``github.com/Aider-AI/polyglot-benchmark``: 225 exercises (the ones at most three of seven
leading models solved) across cpp 26, go 39, java 47, javascript 49, python 34, rust 30. There is no
TypeScript. The model sees the exercise instructions and the stub files and must edit them with
search/replace blocks; the pinned upstream unit tests then decide. Attempt 1 is cold; attempt 2 sees the
failing test output with the standing instruction that the tests are correct and must not be changed.

Three metrics come out of the rows: ``pass_rate_1`` (attempt 1), ``pass_rate_2`` (either attempt) and
``percent_cases_well_formed`` -- the share of measured cases in which every model response parsed and
applied as a valid edit block. Format compliance is the degradation canary: a damaged sampling
distribution breaks block structure before it breaks program logic.

This is not aider's harness. Aider's runner pulls in aider + litellm, wants Docker-in-Docker and shuffles
its subset with an unseeded ``random.shuffle``; nothing here is reproducible under that. The loop, the
parser and the split are implemented against ``benchmarks.BenchmarkAdapter`` instead, and every item set
is a pure function of (languages, split, seed, corpus).

Boundaries this module keeps, in order of importance:

* **It never executes model-written code.** It imports no process API at all. Edited files, the pinned
  test files and the exercise's support files are handed to the injected execution client, which is the
  established isolated worker path (``coding.broker_client``-shaped: ``namespace``/``submit``/``wait``).
  With no client every declared row comes back ``environment_error`` with score 0 -- there is no host
  fallback, and its absence is reported, not worked around.
* Raw responses are persisted through ``context.artifacts`` **before** they are parsed; a failed write
  raises ``BenchmarkAborted`` rather than measuring something whose evidence was lost.
* Every declared task id produces exactly one row. Budget exhaustion marks the rest, it never drops them.
* The model never sees ``.meta/`` (the upstream example solution) or the contents of the test files.

Row status convention, matching ``coding.generation``: a response that reached a verdict attributable to
the model -- passed, failed, or malformed edit format -- is ``status="completed"`` with
``model_evaluated=True`` and a 0/1 score, so it stays in the scored denominator. Harness-attributable
outcomes (no client, transport failure, corpus damage, budget) are ``environment_error``/``timeout`` with
``model_evaluated=False``; ``analysis._scores`` counts those as 0 in the denominator without crediting
them to the candidate.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..config import RunMode, canonical_json
from ..evaluations.tools import strict_json_loads
from ..safety import OperationForbidden
from ..store import utc_now
from . import BenchmarkAborted, BenchmarkContext, BenchmarkUnavailable, missing_rows

BENCHMARK_ID = "aider-polyglot"
REVISION = "aider-polyglot-v1"
CATEGORY = "coding"
UPSTREAM_REPOSITORY = "https://github.com/Aider-AI/polyglot-benchmark"
CORPUS_DIRNAME = "aider-polyglot"
PIN_NAME = "pin.json"

UPSTREAM_LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
UPSTREAM_EXERCISE_COUNTS = {"cpp": 26, "go": 39, "java": 47, "javascript": 49, "python": 34, "rust": 30}
DEFAULT_LANGUAGES = ("javascript", "python")
"""The user's stack: 49 + 34 = 83 exercises. Stored in canonical order; selection is never random."""

TEST_COMMANDS: dict[str, tuple[str, ...] | None] = {
    # Only the two languages this adapter has actually pinned a command for. The other four upstream
    # languages fail closed in ``available`` rather than guessing a build invocation.
    "python": ("python", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"),
    "javascript": ("npm", "test", "--silent"),
    "cpp": None, "go": None, "java": None, "rust": None,
}

MAX_META_BYTES = 65_536
MAX_FILE_BYTES = 262_144
MAX_EXERCISE_BYTES = 1_048_576
MAX_EXERCISE_FILES = 64
MAX_FEEDBACK_CHARS = 6_000
DEFAULT_EXERCISE_TIMEOUT_SECONDS = 120
DEFAULT_MIN_EXERCISE_SECONDS = 30.0
DEFAULT_DEVELOPMENT_FRACTION = 0.5

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"
_FENCE = re.compile(r"^`{3,}[A-Za-z0-9_+.#-]*$")
_SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,80}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_TRIM = " \t`*\"'#"
_SKIP_DIRECTORIES = frozenset({".git", "node_modules", "target", "build", "__pycache__", ".pytest_cache"})
_EXECUTION_STATUSES = ("completed", "rejected", "timeout", "environment-error", "interrupted", "cancelled",
                       "cleanup-unverified")
MODEL_ATTRIBUTABLE_REJECTIONS = frozenset({"patch_too_large", "patch_outside_allowlist"})
"""Broker rejections the model answers for: the edit really was produced, it was just beyond the caps."""

EDIT_FORMAT_HELP = (
    "Reply with nothing but search/replace edit blocks. Each block is exactly:\n"
    "\n"
    "path/to/file.ext\n"
    f"{SEARCH_MARKER}\n"
    "the exact lines that are in the file right now\n"
    f"{DIVIDER}\n"
    "the lines that replace them\n"
    f"{REPLACE_MARKER}\n"
    "\n"
    "Rules: the filename is on its own line directly above the block and must be one of the editable "
    "files listed below. The SEARCH text must reproduce the current file byte for byte, including "
    "indentation, and must occur exactly once in that file. Emit several blocks to make several edits; "
    "they are applied in order. Do not write explanations, diffs, or whole files."
)


class CorpusError(ValueError):
    """The exercise corpus on disk is missing, malformed, oversize or not the pinned shape."""


# --------------------------------------------------------------------------------------------------
# Search/replace edit format
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EditBlock:
    filename: str
    search: str
    replace: str


@dataclass(frozen=True)
class EditParse:
    """``error is None`` is exactly ``well_formed``; ``files`` is set only when every block applied."""

    blocks: tuple[EditBlock, ...] = ()
    files: dict[str, str] | None = None
    error: str | None = None

    @property
    def well_formed(self) -> bool:
        return self.error is None


def normalize_newlines(text: str) -> str:
    """CRLF and lone CR become LF. Corpus files and model output are compared in this one form."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _safe_relative(name: str) -> bool:
    if type(name) is not str or not name or len(name) > 255 or "\\" in name or ":" in name:
        return False
    if any(ord(char) < 32 for char in name):
        return False
    pure = PurePosixPath(name)
    return (not pure.is_absolute() and str(pure) == name
            and not any(part in {"", ".", ".."} for part in pure.parts))


def _clean_filename(line: str) -> str:
    """Undo the decoration models put around a path: backticks, bold markers, a trailing colon."""
    name = line.strip()
    for _ in range(4):
        cleaned = name.strip(_TRIM).strip().rstrip(":").strip()
        if cleaned == name:
            break
        name = cleaned
    return name


def _filename_before(lines: Sequence[str], index: int) -> str | None:
    """The nearest preceding line that is neither blank nor a code fence; aider's own convention."""
    for position in range(index - 1, -1, -1):
        stripped = lines[position].strip()
        if not stripped or _FENCE.fullmatch(stripped):
            continue
        return _clean_filename(stripped)
    return None


def _joined(lines: list[str]) -> str:
    return "\n".join(lines) + "\n" if lines else ""


def parse_edit_blocks(text: Any, editable: Iterable[str]) -> EditParse:
    """Strict, line-oriented parse. Returns the blocks, or the first failure reason and no blocks.

    Failure reasons (the exact strings that land in the row's ``edit_errors``):
    ``no_text_content``, ``no_edit_blocks``, ``missing_filename``, ``unsafe_filename:<name>``,
    ``unknown_filename:<name>``, ``missing_divider``, ``duplicate_divider``, ``nested_search_marker``,
    ``unterminated_block``, ``stray_replace_marker``, ``empty_search``.
    """
    if type(text) is not str:
        return EditParse(error="no_text_content")
    allowed = set(editable)
    lines = normalize_newlines(text).split("\n")
    blocks: list[EditBlock] = []
    index, total = 0, len(lines)
    while index < total:
        marker = lines[index].rstrip()
        if marker == REPLACE_MARKER:
            return EditParse(error="stray_replace_marker")
        if marker != SEARCH_MARKER:
            index += 1
            continue
        name = _filename_before(lines, index)
        if not name:
            return EditParse(error="missing_filename")
        if not _safe_relative(name):
            return EditParse(error=f"unsafe_filename:{name[:120]}")
        if name not in allowed:
            return EditParse(error=f"unknown_filename:{name[:120]}")
        cursor, search = index + 1, []
        while cursor < total and lines[cursor].rstrip() != DIVIDER:
            current = lines[cursor].rstrip()
            if current == SEARCH_MARKER:
                return EditParse(error="nested_search_marker")
            if current == REPLACE_MARKER:
                return EditParse(error="missing_divider")
            search.append(lines[cursor])
            cursor += 1
        if cursor >= total:
            return EditParse(error="missing_divider")
        cursor, replace = cursor + 1, []
        while cursor < total and lines[cursor].rstrip() != REPLACE_MARKER:
            current = lines[cursor].rstrip()
            if current == SEARCH_MARKER:
                return EditParse(error="nested_search_marker")
            if current == DIVIDER:
                return EditParse(error="duplicate_divider")
            replace.append(lines[cursor])
            cursor += 1
        if cursor >= total:
            return EditParse(error="unterminated_block")
        if not search:
            return EditParse(error="empty_search")
        blocks.append(EditBlock(name, _joined(search), _joined(replace)))
        index = cursor + 1
    if not blocks:
        return EditParse(error="no_edit_blocks")
    return EditParse(tuple(blocks))


def apply_edit_blocks(blocks: Sequence[EditBlock],
                      files: Mapping[str, str]) -> tuple[dict[str, str] | None, str | None]:
    """Apply in order, all or nothing. Each SEARCH must match its file exactly once, right then."""
    current = {name: normalize_newlines(content) for name, content in files.items()}
    for position, block in enumerate(blocks):
        if block.filename not in current:
            return None, f"unknown_filename:{block.filename[:120]}"
        content = current[block.filename]
        occurrences = content.count(block.search)
        if occurrences == 0:
            return None, f"search_not_found:{block.filename}#{position}"
        if occurrences > 1:
            return None, f"search_not_unique:{block.filename}#{position}:{occurrences}"
        current[block.filename] = content.replace(block.search, block.replace, 1)
    return current, None


def parse_and_apply(text: Any, files: Mapping[str, str]) -> EditParse:
    """Parse the response and apply it to the current editable contents; both must succeed."""
    parsed = parse_edit_blocks(text, files.keys())
    if not parsed.well_formed:
        return parsed
    updated, error = apply_edit_blocks(parsed.blocks, files)
    if error is not None:
        return EditParse(parsed.blocks, None, error)
    return EditParse(parsed.blocks, updated, None)


# --------------------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusPin:
    """``pin.json`` beside the corpus: what the image actually baked, written at bake time.

    The upstream commit is never hardcoded here. The adapter refuses to run against a corpus that does
    not state the commit it came from, so a row can always name the exact exercises it measured.
    """

    repository: str
    commit: str
    retrieved_utc: str
    languages: tuple[str, ...]
    toolchains: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {"repository": self.repository, "commit": self.commit, "retrieved_utc": self.retrieved_utc,
                "languages": list(self.languages), "toolchains": list(self.toolchains)}


@dataclass(frozen=True)
class Exercise:
    task_id: str
    language: str
    slug: str
    instructions: str
    solution: tuple[tuple[str, str], ...]
    tests: tuple[tuple[str, str], ...]
    support: tuple[tuple[str, str], ...]
    digest: str

    def editable(self) -> dict[str, str]:
        return dict(self.solution)

    def test_files(self) -> dict[str, str]:
        return dict(self.tests)

    def support_files(self) -> dict[str, str]:
        return dict(self.support)


@dataclass(frozen=True)
class Plan:
    root: Path
    pin: CorpusPin
    languages: tuple[str, ...]
    seed: int
    development_fraction: float
    directories: dict[str, Path]
    development: tuple[str, ...]
    holdout: tuple[str, ...]

    def ids_for(self, split: str) -> tuple[str, ...]:
        if split not in {"development", "holdout"}:
            raise CorpusError(f"unknown split: {split!r}")
        return self.development if split == "development" else self.holdout

    def item_set_digest(self, split: str) -> str:
        payload = {"suite": BENCHMARK_ID, "revision": REVISION, "commit": self.pin.commit, "split": split,
                   "seed": self.seed, "languages": list(self.languages),
                   "development_fraction": self.development_fraction, "task_ids": list(self.ids_for(split))}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _read_text(path: Path, cap: int) -> str:
    if not path.is_file():
        raise CorpusError(f"missing corpus file: {path.name}")
    size = path.stat().st_size
    if size > cap:
        raise CorpusError(f"{path.name} is {size} bytes; the cap is {cap}")
    try:
        return normalize_newlines(path.read_bytes().decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CorpusError(f"{path.name} is not UTF-8: {exc}") from exc
    except OSError as exc:
        raise CorpusError(f"{path.name} is unreadable: {exc}") from exc


def read_pin(root: str | Path) -> CorpusPin:
    """Strict read of ``<corpus>/pin.json``; anything unexpected is a refusal, never a default."""
    path = Path(root) / PIN_NAME
    raw = _read_text(path, MAX_META_BYTES)
    try:
        parsed = strict_json_loads(raw)
    except (ValueError, RecursionError) as exc:
        raise CorpusError(f"{PIN_NAME} is not strict JSON: {exc}") from exc
    if type(parsed) is not dict:
        raise CorpusError(f"{PIN_NAME} must be a JSON object")
    expected = {"schema_version", "repository", "commit", "retrieved_utc", "languages", "toolchains"}
    if set(parsed) != expected:
        raise CorpusError(f"{PIN_NAME} keys must be exactly {sorted(expected)}")
    if parsed["schema_version"] != 1:
        raise CorpusError(f"{PIN_NAME} schema_version must be 1")
    if parsed["repository"] != UPSTREAM_REPOSITORY:
        raise CorpusError(f"{PIN_NAME} repository is {parsed['repository']!r}, not {UPSTREAM_REPOSITORY}")
    if type(parsed["commit"]) is not str or not _HEX40.fullmatch(parsed["commit"]):
        raise CorpusError(f"{PIN_NAME} commit must be a 40-character lowercase hex commit id")
    if type(parsed["retrieved_utc"]) is not str or not parsed["retrieved_utc"]:
        raise CorpusError(f"{PIN_NAME} retrieved_utc must be a nonempty string")
    for key in ("languages", "toolchains"):
        value = parsed[key]
        if (type(value) is not list or not value or any(item not in UPSTREAM_LANGUAGES for item in value)
                or len(set(value)) != len(value)):
            raise CorpusError(f"{PIN_NAME} {key} must be distinct upstream language names")
    return CorpusPin(parsed["repository"], parsed["commit"], parsed["retrieved_utc"],
                     tuple(parsed["languages"]), tuple(parsed["toolchains"]))


def practice_root(root: str | Path, language: str) -> Path:
    return Path(root) / language / "exercises" / "practice"


def enumerate_exercises(root: str | Path, language: str) -> tuple[tuple[str, Path], ...]:
    """Every practice directory, sorted by slug. A name outside the slug grammar refuses the corpus."""
    base = practice_root(root, language)
    if not base.is_dir():
        raise CorpusError(f"no practice directory for {language}: {base}")
    found: list[tuple[str, Path]] = []
    for entry in sorted(base.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        if not _SLUG.fullmatch(entry.name):
            raise CorpusError(f"{language}: exercise directory {entry.name!r} is outside the slug grammar")
        found.append((f"{BENCHMARK_ID}/{language}/{entry.name}", entry))
    if not found:
        raise CorpusError(f"no exercises for {language} under {base}")
    return tuple(found)


def _manifest_files(directory: Path) -> tuple[list[str], list[str]]:
    meta = directory / ".meta" / "config.json"
    raw = _read_text(meta, MAX_META_BYTES)
    try:
        parsed = strict_json_loads(raw)
    except (ValueError, RecursionError) as exc:
        raise CorpusError(f"{directory.name}/.meta/config.json is not strict JSON: {exc}") from exc
    files = parsed.get("files") if type(parsed) is dict else None
    if type(files) is not dict:
        raise CorpusError(f"{directory.name}/.meta/config.json has no files object")
    solution, tests = files.get("solution"), files.get("test")
    for label, value in (("solution", solution), ("test", tests)):
        if (type(value) is not list or not value or len(set(value)) != len(value)
                or any(not _safe_relative(item) for item in value)):
            raise CorpusError(f"{directory.name}: files.{label} must be distinct safe relative paths")
        if any(PurePosixPath(item).parts[0] in {".meta", ".docs"} for item in value):
            raise CorpusError(f"{directory.name}: files.{label} may not point into .meta or .docs")
    if set(solution) & set(tests):
        raise CorpusError(f"{directory.name}: a file is listed as both solution and test")
    return list(solution), list(tests)


def _support_files(directory: Path, claimed: set[str]) -> list[tuple[str, str]]:
    """Everything else the exercise ships (package.json, jest config, fixtures), never .meta or .docs."""
    support: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        parts = PurePosixPath(relative).parts
        if parts[0] in {".meta", ".docs"} or any(part in _SKIP_DIRECTORIES for part in parts):
            continue
        if relative in claimed:
            continue
        support.append((relative, _read_text(path, MAX_FILE_BYTES)))
    return support


def load_exercise(task_id: str, directory: Path) -> Exercise:
    """Read one exercise exactly as the image baked it. ``.meta`` never reaches the model or the worker."""
    parts = task_id.split("/")
    if len(parts) != 3 or parts[0] != BENCHMARK_ID:
        raise CorpusError(f"malformed task id: {task_id!r}")
    language, slug = parts[1], parts[2]
    solution, tests = _manifest_files(directory)
    instructions = _read_text(directory / ".docs" / "instructions.md", MAX_FILE_BYTES)
    extra = directory / ".docs" / "instructions.append.md"
    if extra.is_file():
        instructions = instructions + "\n\n" + _read_text(extra, MAX_FILE_BYTES)
    solution_files = [(name, _read_text(directory / Path(*PurePosixPath(name).parts), MAX_FILE_BYTES))
                      for name in solution]
    test_files = [(name, _read_text(directory / Path(*PurePosixPath(name).parts), MAX_FILE_BYTES))
                  for name in tests]
    support = _support_files(directory, set(solution) | set(tests))
    everything = solution_files + test_files + support
    if len(everything) > MAX_EXERCISE_FILES:
        raise CorpusError(f"{task_id}: {len(everything)} files exceed the {MAX_EXERCISE_FILES}-file cap")
    total = sum(len(content.encode("utf-8")) for _, content in everything)
    if total > MAX_EXERCISE_BYTES:
        raise CorpusError(f"{task_id}: {total} bytes exceed the {MAX_EXERCISE_BYTES}-byte cap")
    digest = hashlib.sha256(canonical_json({
        "task_id": task_id, "instructions": instructions, "solution": dict(solution_files),
        "tests": dict(test_files), "support": dict(support)}).encode("utf-8")).hexdigest()
    return Exercise(task_id, language, slug, instructions, tuple(solution_files), tuple(test_files),
                    tuple(support), digest)


def selected_languages(options: Mapping[str, Any]) -> tuple[str, ...]:
    """``options['languages']`` as a comma string or sequence; canonical order, never random."""
    raw = options.get("languages", DEFAULT_LANGUAGES)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [str(item).strip() for item in raw]
    else:
        raise CorpusError("languages option must be a comma-separated string or a sequence")
    if not values:
        raise CorpusError("at least one language must be selected")
    unknown = sorted(set(values) - set(UPSTREAM_LANGUAGES))
    if unknown:
        raise CorpusError(f"unknown upstream language(s): {', '.join(unknown)} "
                          f"(upstream has {', '.join(UPSTREAM_LANGUAGES)} and no TypeScript)")
    if len(set(values)) != len(values):
        raise CorpusError("languages must be distinct")
    return tuple(sorted(set(values)))


def _order_key(seed: int, task_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{REVISION}:{seed}:{task_id}".encode("utf-8")).hexdigest()
    return digest, task_id


def split_exercises(task_ids: Sequence[str], *, seed: int,
                    development_fraction: float = DEFAULT_DEVELOPMENT_FRACTION) -> tuple[tuple[str, ...],
                                                                                        tuple[str, ...]]:
    """Deterministic, disjoint, exhaustive split of one language's exercises.

    Order the slugs by ``sha256(revision:seed:task_id)`` (ties by task id -- ids are unique, so there are
    none), take ``ceil(n * fraction)`` for development and the remainder for holdout. Same seed and same
    corpus give the same two sets on any machine; no exercise is in both, and none is lost.
    """
    if not 0 < development_fraction < 1:
        raise CorpusError("development_fraction must be strictly between 0 and 1")
    ordered = sorted(task_ids, key=lambda task_id: _order_key(seed, task_id))
    cut = math.ceil(len(ordered) * development_fraction)
    cut = min(max(cut, 1), len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    return tuple(sorted(ordered[:cut])), tuple(sorted(ordered[cut:]))


def build_plan(context: BenchmarkContext) -> Plan:
    """Resolve the corpus, the pin, the language selection and both splits. Raises ``CorpusError``."""
    options = context.options or {}
    unknown = sorted(set(options) - {"languages", "development_fraction", "exercise_timeout_seconds",
                                     "min_exercise_seconds", "transport", "completion", "callback",
                                     "execution_client", "client", "expected_commit"})
    if unknown:
        raise CorpusError(f"unsupported option(s): {', '.join(unknown)}")
    root = Path(context.dataset_root) / CORPUS_DIRNAME
    if not root.is_dir():
        raise CorpusError(f"exercise corpus is absent: {root}")
    pin = read_pin(root)
    expected = options.get("expected_commit")
    if expected is not None and expected != pin.commit:
        raise CorpusError(f"corpus commit {pin.commit} is not the expected {expected}")
    languages = selected_languages(options)
    missing_toolchain = [name for name in languages if name not in pin.toolchains]
    if missing_toolchain:
        raise CorpusError(f"the worker image declares no toolchain for: {', '.join(missing_toolchain)}")
    unpinned = [name for name in languages if TEST_COMMANDS.get(name) is None]
    if unpinned:
        raise CorpusError(f"no pinned test command for: {', '.join(unpinned)}")
    absent = [name for name in languages if name not in pin.languages]
    if absent:
        raise CorpusError(f"the corpus pin does not carry: {', '.join(absent)}")
    fraction = options.get("development_fraction", DEFAULT_DEVELOPMENT_FRACTION)
    if type(fraction) not in (int, float) or not 0 < float(fraction) < 1:
        raise CorpusError("development_fraction must be a number strictly between 0 and 1")
    seed = int(context.seed)
    directories: dict[str, Path] = {}
    development: list[str] = []
    holdout: list[str] = []
    for language in languages:
        found = enumerate_exercises(root, language)
        directories.update({task_id: path for task_id, path in found})
        left, right = split_exercises([task_id for task_id, _ in found], seed=seed,
                                      development_fraction=float(fraction))
        development.extend(left)
        holdout.extend(right)
    return Plan(root, pin, languages, seed, float(fraction), directories,
                tuple(sorted(development)), tuple(sorted(holdout)))


# --------------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------------


def _file_section(files: Mapping[str, str]) -> str:
    return "".join(f"\n--- {name} (current content) ---\n{content}" for name, content in files.items())


def build_initial_prompt(exercise: Exercise, files: Mapping[str, str] | None = None) -> list[dict]:
    """Attempt 1: instructions, the editable stubs, and the names (never the bodies) of the tests."""
    current = dict(files) if files is not None else exercise.editable()
    system = ("You are completing a programming exercise by editing files in place. " + EDIT_FORMAT_HELP)
    user = (f"# {exercise.slug} ({exercise.language})\n\n{exercise.instructions}\n\n"
            f"Editable files: {', '.join(current)}\n"
            f"Read-only test files that will be run against your code: {', '.join(dict(exercise.tests))}\n"
            "Do not create, rename or modify any other file.\n"
            + _file_section(current))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_repair_prompt(exercise: Exercise, files: Mapping[str, str], previous: str, feedback: str, *,
                        kind: str) -> list[dict]:
    """Attempt 2. ``kind`` is 'tests' (failing output) or 'format' (the edit blocks did not apply)."""
    if kind not in {"tests", "format"}:
        raise ValueError("repair kind must be 'tests' or 'format'")
    messages = build_initial_prompt(exercise, files)
    messages.append({"role": "assistant", "content": previous})
    if kind == "tests":
        instruction = ("The tests are correct and must not be changed. Do not edit any test file. "
                       "Fix the implementation in the editable files so the tests pass.\n\n"
                       "Test output:\n" + feedback[:MAX_FEEDBACK_CHARS])
    else:
        instruction = ("Your reply could not be applied as search/replace edits: " + feedback[:MAX_FEEDBACK_CHARS]
                       + "\nThe tests are correct and must not be changed. Do not edit any test file.")
    messages.append({"role": "user", "content": instruction + "\n\n" + EDIT_FORMAT_HELP})
    return messages


# --------------------------------------------------------------------------------------------------
# Execution request handed to the isolated worker path
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionRequest:
    """What the adapter submits for one attempt. The host stages exactly this and nothing else.

    It mirrors ``coding.spool.CodingJobRequest`` (namespace identity, request id, content hash, attempt
    index, patch) and adds what an Exercism exercise needs: the pinned test files, the support files,
    the working directory and the pinned test command. ``patch`` holds full file contents, already
    normalized to LF -- the worker never applies a diff and never sees the model's raw text.
    """

    session_id: str
    attempt_id: str
    request_id: str
    task_id: str
    language: str
    exercise_sha256: str
    attempt_index: int
    workdir: str
    command: tuple[str, ...]
    patch: dict[str, str]
    test_files: dict[str, str]
    support_files: dict[str, str]
    timeout_seconds: int
    upstream_commit: str
    submitted_utc: str
    schema_version: int = 1
    suite: str = BENCHMARK_ID
    suite_revision: str = REVISION

    def payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "suite": self.suite,
                "suite_revision": self.suite_revision, "session_id": self.session_id,
                "attempt_id": self.attempt_id, "request_id": self.request_id, "task_id": self.task_id,
                "language": self.language, "exercise_sha256": self.exercise_sha256,
                "attempt_index": self.attempt_index, "workdir": self.workdir, "command": list(self.command),
                "patch": dict(self.patch), "test_files": dict(self.test_files),
                "support_files": dict(self.support_files), "timeout_seconds": self.timeout_seconds,
                "upstream_commit": self.upstream_commit, "submitted_utc": self.submitted_utc}

    def content_sha256(self) -> str:
        payload = {key: value for key, value in self.payload().items() if key != "submitted_utc"}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def pinned_test_command(exercise: Exercise) -> tuple[str, ...]:
    """The pinned argv for this language, run in the staged exercise directory.

    Python names the pinned test files explicitly so a model-created file can never be collected;
    javascript runs the exercise's own jest script, whose dependencies the worker image must bake
    (the worker has no network).
    """
    command = TEST_COMMANDS.get(exercise.language)
    if not command:
        raise CorpusError(f"no pinned test command for {exercise.language}")
    if exercise.language == "python":
        return tuple(command) + tuple(name for name, _ in exercise.tests)
    return tuple(command)


@dataclass(frozen=True)
class ExecutionOutcome:
    status: str
    passed: bool
    output: str
    failure_reason: str | None
    model_attributable: bool
    info: dict[str, Any]


def _field(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def normalize_execution_result(raw: Any, request: ExecutionRequest) -> ExecutionOutcome:
    """Accept only a structured, identity-matched result. Anything else is a harness failure, not a 0.

    Raises ``BenchmarkAborted`` when the worker path reports unverified cleanup or a campaign abort:
    the same fail-closed rule the coding broker already imposes.
    """
    if raw is None:
        return ExecutionOutcome("invalid_execution_result", False, "", "no result returned", False, {})
    status = _field(raw, "status")
    if status not in _EXECUTION_STATUSES:
        return ExecutionOutcome("invalid_execution_result", False, "", f"unknown status {status!r}", False, {})
    request_id, content = _field(raw, "request_id"), _field(raw, "request_sha256")
    if request_id is not None and request_id != request.request_id:
        return ExecutionOutcome("execution_integrity", False, "", "result is for another request", False, {})
    if content is not None and content != request.content_sha256():
        return ExecutionOutcome("execution_integrity", False, "",
                                "result does not match the submitted request content", False, {})
    reason = _field(raw, "failure_reason")
    reason = reason if isinstance(reason, str) else None
    info = {"request_id": request.request_id, "attempt_index": request.attempt_index, "status": status,
            "trace_sha256": _field(raw, "trace_sha256"), "cleanup_confirmed": _field(raw, "cleanup_confirmed")}
    if _field(raw, "abort_campaign") is True or status == "cleanup-unverified":
        raise BenchmarkAborted(f"the isolated worker reported unverified cleanup for {request.task_id}: "
                               f"{reason}")
    if status != "completed":
        attributable = status == "rejected" and reason in MODEL_ATTRIBUTABLE_REJECTIONS
        return ExecutionOutcome("execution_" + status.replace("-", "_"), False, "", reason, attributable, info)
    sample = _field(raw, "sample")
    if (_field(sample, "status") == "environment-error" or _field(sample, "model_evaluated") is False):
        return ExecutionOutcome("invalid_execution_result", False, "",
                                "completed envelope contains infrastructure-only evidence", False, info)
    passed = _field(sample, "passed") if sample is not None else None
    if type(passed) is not bool:
        return ExecutionOutcome("invalid_execution_result", False, "",
                                "a completed result must carry sample.passed as a boolean", False, info)
    output = _field(sample, "test_output") or _field(sample, "output") or ""
    info.update({key: _field(sample, key) for key in ("tests_passed", "tests_failed", "exit_code")
                 if _field(sample, key) is not None})
    return ExecutionOutcome("completed", passed, output if isinstance(output, str) else "", None, True, info)


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The suite's three headline numbers, recomputable from the rows alone.

    ``pass_rate_1`` and ``pass_rate_2`` use the full declared denominator: a case the harness could not
    run counts as a failure, exactly like ``analysis._scores``. ``percent_cases_well_formed`` uses the
    cases in which the model actually answered at least once (``responses >= 1``), because a case that
    never reached the model says nothing about edit-format compliance; ``cases_unanswered`` is reported
    beside it so the gap can never be read as compliance.
    """
    considered = [row for row in rows if row.get("suite") == BENCHMARK_ID]
    total = len(considered)
    answered = [row for row in considered if int(row.get("responses") or 0) >= 1]
    well_formed = [row for row in answered if row.get("well_formed") is True]
    first = sum(1 for row in considered if row.get("first_attempt_success") is True)
    either = sum(1 for row in considered if row.get("passed") is True)
    return {"suite": BENCHMARK_ID, "suite_revision": REVISION, "cases_total": total,
            "cases_answered": len(answered), "cases_unanswered": total - len(answered),
            "cases_passed_attempt_1": first, "cases_passed_either_attempt": either,
            "pass_rate_1": (first / total) if total else None,
            "pass_rate_2": (either / total) if total else None,
            "cases_well_formed": len(well_formed), "cases_malformed": len(answered) - len(well_formed),
            "malformed_responses": sum(int(row.get("malformed_responses") or 0) for row in considered),
            "percent_cases_well_formed": (100.0 * len(well_formed) / len(answered)) if answered else None,
            "denominators": {"pass_rate": "all declared cases",
                             "percent_cases_well_formed": "cases with at least one model response"}}


# --------------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------------


def _dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=str, indent=1).encode("utf-8")


def _content_of(response: Any) -> Any:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


async def _awaited(value: Any) -> Any:
    return await value


class AiderPolyglotBenchmark:
    """``BenchmarkAdapter`` for the Aider Polyglot exercises.

    The two collaborators are injected, never constructed here: ``transport`` is the evaluator's
    completion callback (sync or async, OpenAI chat shape), ``execution_client`` is the isolated worker
    client (``namespace()``, ``submit(request)``, ``wait(request_id, deadline=...)``, optional
    ``clock``). Either may instead arrive through ``context.options``.
    """

    benchmark_id = BENCHMARK_ID
    revision = REVISION
    category = CATEGORY

    def __init__(self, *, transport: Callable[[dict], Any] | None = None, execution_client: Any = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._transport = transport
        self._execution_client = execution_client
        self._clock = clock

    # -- preflight ---------------------------------------------------------------------------------

    def available(self, context: BenchmarkContext) -> tuple[bool, str]:
        try:
            plan = build_plan(context)
            if not plan.ids_for(context.split):
                return False, f"no {context.split} exercises for {', '.join(plan.languages)}"
        except CorpusError as exc:
            return False, str(exc)
        except Exception as exc:  # available() never raises: an unexpected failure is still unavailable
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""

    def task_ids(self, context: BenchmarkContext) -> tuple[str, ...]:
        try:
            return build_plan(context).ids_for(context.split)
        except CorpusError as exc:
            raise BenchmarkUnavailable(str(exc)) from exc

    # -- run ---------------------------------------------------------------------------------------

    def run(self, context: BenchmarkContext) -> list[dict[str, Any]]:
        declared = tuple(context.task_ids)
        try:
            plan = build_plan(context)
        except OperationForbidden:
            raise  # a policy refusal is never turned into rows
        except Exception as exc:
            return self._all_failed(context, declared, "environment_error", "corpus_unavailable",
                                    str(exc) if isinstance(exc, CorpusError)
                                    else f"{type(exc).__name__}: {exc}")
        if not declared:
            declared = plan.ids_for(context.split)
        digest = plan.item_set_digest(context.split)
        eligible = set(plan.ids_for(context.split))

        client = self._execution_client or self._option(context, "execution_client", "client")
        if client is None:
            # The loud boundary: no isolated worker, no execution. There is no host fallback to take.
            return self._all_failed(context, declared, "environment_error", "no_execution_client",
                                    "no isolated execution client is wired; model-written code is never "
                                    "run on the host", plan=plan, digest=digest)
        transport = self._transport or self._option(context, "transport", "completion", "callback")
        if transport is None or not callable(transport):
            return self._all_failed(context, declared, "environment_error", "no_transport",
                                    "no completion transport is wired", plan=plan, digest=digest)
        try:
            namespace = client.namespace()
            session_id, attempt_id = namespace["session_id"], namespace["attempt_id"]
        except Exception as exc:
            return self._all_failed(context, declared, "environment_error", "no_execution_namespace",
                                    f"execution namespace unavailable: {type(exc).__name__}: {exc}",
                                    plan=plan, digest=digest)

        options = context.options or {}
        floor = float(options.get("min_exercise_seconds", DEFAULT_MIN_EXERCISE_SECONDS))
        rows: list[dict[str, Any]] = []
        exhausted = False
        for task_id in declared:
            base = self._base_row(context, task_id, plan=plan, digest=digest)
            if task_id not in eligible:
                rows.append(self._failed(base, "environment_error", "unknown_task",
                                         f"{task_id} is not in the {context.split} item set for seed "
                                         f"{plan.seed}"))
                continue
            remaining = _remaining(context)
            if exhausted or remaining is None or remaining < floor:
                exhausted = True
                rows.append(self._failed(base, "timeout", "budget_exhausted",
                                         "the wall budget was exhausted before this exercise ran"))
                continue
            try:
                exercise = load_exercise(task_id, plan.directories[task_id])
            except CorpusError as exc:
                rows.append(self._failed(base, "environment_error", "corpus_error", str(exc)))
                continue
            base["fixture_hash"] = exercise.digest
            base["exercise_sha256"] = exercise.digest
            try:
                rows.append(self._run_case(context, exercise, base, transport=transport, client=client,
                                           session_id=session_id, attempt_id=attempt_id,
                                           upstream_commit=plan.pin.commit))
            except (BenchmarkAborted, OperationForbidden, KeyboardInterrupt, SystemExit):
                raise  # an abort or a policy refusal must stop the run, never become a scored row
            except BaseException as exc:
                # A bug in this adapter must not shrink the denominator, and must not look like a model 0.
                rows.append(self._failed(base, "environment_error", "adapter_error",
                                         f"{type(exc).__name__}: {exc}"))
        rows.extend(missing_rows(declared, rows, suite=BENCHMARK_ID, revision=REVISION, category=CATEGORY,
                                 split=context.split, status="environment_error",
                                 reason="the adapter returned no row for this declared task"))
        self._persist_metrics(context, rows)
        return rows

    # -- internals ---------------------------------------------------------------------------------

    def _option(self, context: BenchmarkContext, *names: str) -> Any:
        options = context.options or {}
        for name in names:
            if options.get(name) is not None:
                return options[name]
        return None

    def _base_row(self, context: BenchmarkContext, task_id: str, *, plan: Plan | None,
                  digest: str | None) -> dict[str, Any]:
        language = task_id.split("/")[1] if task_id.count("/") == 2 else None
        row = {"task_id": task_id, "suite": BENCHMARK_ID, "suite_revision": REVISION, "category": CATEGORY,
               "split": context.split, "seed": context.seed, "language": language,
               "upstream_repository": UPSTREAM_REPOSITORY, "synthetic": False, "model_evaluated": False,
               "responses": 0, "attempts": 0, "well_formed": None, "malformed_responses": 0,
               "edit_errors": [], "first_attempt_success": False, "attempt_1_passed": False,
               "item_set_digest": digest}
        if plan is not None:
            row["upstream_commit"] = plan.pin.commit
            row["languages"] = list(plan.languages)
        # Analysis requires a fixture hash on every row; without a loadable exercise it is the identity
        # of what was declared, which is still unique per task and can never collide with another row.
        row["fixture_hash"] = hashlib.sha256(
            canonical_json({"task_id": task_id, "revision": REVISION,
                            "commit": plan.pin.commit if plan else None}).encode("utf-8")).hexdigest()
        return row

    def _failed(self, base: Mapping[str, Any], status: str, outcome: str, error: str,
                *, model_evaluated: bool = False, **extra: Any) -> dict[str, Any]:
        return {**base, "status": status, "outcome_status": outcome, "score": 0.0, "passed": False,
                "model_evaluated": model_evaluated, "error": error, **extra}

    def _all_failed(self, context: BenchmarkContext, declared: Sequence[str], status: str, outcome: str,
                    reason: str, *, plan: Plan | None = None,
                    digest: str | None = None) -> list[dict[str, Any]]:
        rows = [self._failed(self._base_row(context, task_id, plan=plan, digest=digest), status, outcome,
                             reason) for task_id in declared]
        self._persist_metrics(context, rows)
        return rows

    def _persist_metrics(self, context: BenchmarkContext, rows: list[dict[str, Any]]) -> None:
        """Best effort, loudly marked: the rows already carry everything the metrics are derived from."""
        try:
            context.artifacts.write(f"{BENCHMARK_ID}/metrics.json", _dump(aggregate_metrics(rows)))
        except Exception as exc:
            note = f"metrics.json was not written: {type(exc).__name__}: {exc}"
            for row in rows:
                row["metrics_persist_error"] = note

    def _run_case(self, context: BenchmarkContext, exercise: Exercise, base: dict[str, Any], *,
                  transport: Callable[[dict], Any], client: Any, session_id: str, attempt_id: str,
                  upstream_commit: str) -> dict[str, Any]:
        """One exercise, at most two attempts. Every exit goes through ``finish``, so a row always states
        how many responses it saw, how many were malformed and whether the case was well formed."""
        options = context.options or {}
        timeout = int(options.get("exercise_timeout_seconds", DEFAULT_EXERCISE_TIMEOUT_SECONDS))
        evidence: dict[str, Any] = {"responses": [], "parse": [], "execution": []}
        files = exercise.editable()
        row = dict(base)

        def record(parse: EditParse) -> None:
            evidence["parse"].append(parse.error or "well_formed")
            if parse.error is not None:
                row["malformed_responses"] = int(row["malformed_responses"]) + 1
                row["edit_errors"] = list(row["edit_errors"]) + [parse.error]

        def finish(value: dict[str, Any]) -> dict[str, Any]:
            # ``None`` when the model never answered: that is not compliance and not a violation either.
            responses = int(value.get("responses") or 0)
            value["well_formed"] = None if responses == 0 else int(value["malformed_responses"]) == 0
            value["evidence"] = json.loads(json.dumps(evidence, ensure_ascii=False, default=str))
            value["execution"] = list(evidence["execution"])
            return value

        def verdict(outcome_status: str, passed: bool, *, first: bool = False, **extra: Any) -> dict:
            return finish({**row, "status": "completed", "outcome_status": outcome_status,
                           "score": 1.0 if passed else 0.0, "passed": passed, "model_evaluated": True,
                           "first_attempt_success": first and passed, **extra})

        def ask(messages: list[dict], index: int) -> Any:
            return self._ask(context, transport, exercise, messages, index, evidence, row)

        def execute(updated: Mapping[str, str], index: int) -> ExecutionOutcome:
            return self._execute(context, client, exercise, updated, index, timeout, session_id,
                                 attempt_id, upstream_commit, evidence)

        # ---- attempt 1 (cold)
        try:
            first_response = ask(build_initial_prompt(exercise), 1)
        except _TransportFailure as exc:
            return finish(self._failed(row, exc.status, exc.outcome, exc.message))
        first_text = _content_of(first_response)
        first_parse = parse_and_apply(first_text, files)
        record(first_parse)
        attempt_one: ExecutionOutcome | None = None
        if first_parse.well_formed and first_parse.files is not None:
            attempt_one = execute(first_parse.files, 1)
            if attempt_one.status != "completed":
                # No verdict for a well-formed edit: harness-attributable unless the worker rejected the
                # model's own oversize/out-of-allowlist patch.
                status = "timeout" if attempt_one.status == "execution_timeout" else "environment_error"
                return finish(self._failed(row, "completed" if attempt_one.model_attributable else status,
                                           attempt_one.status,
                                           attempt_one.failure_reason or attempt_one.status,
                                           model_evaluated=attempt_one.model_attributable))
            files = dict(first_parse.files)
            row["attempt_1_passed"] = attempt_one.passed
            if attempt_one.passed:
                return verdict("passed", True, first=True)

        # ---- attempt 2: fed the failure, told the tests are correct and must not be changed
        if attempt_one is not None:
            feedback, kind = attempt_one.output or "the unit tests failed", "tests"
        else:
            feedback, kind = first_parse.error or "unparseable", "format"
        previous = first_text if isinstance(first_text, str) else ""
        try:
            second_response = ask(build_repair_prompt(exercise, files, previous, feedback, kind=kind), 2)
        except _TransportFailure as exc:
            return finish(self._after_attempt_two(row, attempt_one, exc.outcome, exc.message, exc.status))
        second_parse = parse_and_apply(_content_of(second_response), files)
        record(second_parse)
        if not second_parse.well_formed or second_parse.files is None:
            if attempt_one is None:  # both responses malformed: a measured edit-format failure
                return verdict("invalid_edit_format", False, repair_error=second_parse.error)
            return verdict("failed", False, repair_error=f"invalid_edit_format: {second_parse.error}")
        second = execute(second_parse.files, 2)
        if second.status != "completed":
            if second.model_attributable:
                return verdict("failed", False, repair_error=second.failure_reason or second.status)
            return finish(self._after_attempt_two(row, attempt_one, second.status,
                                                  second.failure_reason or second.status,
                                                  "timeout" if second.status == "execution_timeout"
                                                  else "environment_error"))
        return verdict("passed" if second.passed else "failed", second.passed)

    def _after_attempt_two(self, row: Mapping[str, Any], attempt_one: ExecutionOutcome | None, outcome: str,
                           error: str, status: str) -> dict[str, Any]:
        """Keep attempt-one evidence, but an unobserved repair cannot qualify a two-attempt score."""
        detail = {"repair_error": error} if attempt_one is not None else {"error": error}
        prefix = "repair_" if attempt_one is not None else "malformed_then_"
        return {**row, "status": status, "outcome_status": prefix + outcome, "score": 0.0,
                "passed": False, "model_evaluated": False, "first_attempt_success": False, **detail}

    def _ask(self, context: BenchmarkContext, transport: Callable[[dict], Any], exercise: Exercise,
             messages: list[dict], index: int, evidence: dict[str, Any], row: dict[str, Any]) -> Any:
        remaining = _remaining(context)
        if remaining is None:
            raise _TransportFailure("timeout", "budget_exhausted", "no wall budget left for this request")
        context.session_lock.check("inference", RunMode.LIVE)
        generation = context.generation
        request: dict[str, Any] = {
            "model": context.model_alias, "messages": messages,
            "max_tokens": getattr(generation, "max_output_tokens", 1024),
            "temperature": getattr(generation, "temperature", 0.0),
            "top_p": getattr(generation, "top_p", 1.0), "seed": getattr(generation, "seed", 42)}
        if getattr(generation, "top_k", None) is not None:
            request["top_k"] = generation.top_k
        row["attempts"] = int(row["attempts"]) + 1
        try:
            value = transport(request)
            if inspect.isawaitable(value):
                value = asyncio.run(asyncio.wait_for(_awaited(value), timeout=remaining))
        except BenchmarkAborted:
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise _TransportFailure("timeout", "transport_timeout", f"TimeoutError: {exc}") from exc
        except Exception as exc:
            raise _TransportFailure("environment_error", "transport_error",
                                    f"{type(exc).__name__}: {exc}") from exc
        # Evidence before parsing: a response we cannot store is never measured.
        relative = f"{BENCHMARK_ID}/{exercise.language}/{exercise.slug}/response-{index}.json"
        raw = _dump(value)
        try:
            context.artifacts.write(relative, raw)
        except Exception as exc:
            raise BenchmarkAborted(f"aider-polyglot response evidence could not be persisted: "
                                   f"{type(exc).__name__}: {exc}") from exc
        row["responses"] = int(row["responses"]) + 1
        evidence["responses"].append({"attempt": index, "artifact": relative,
                                      "sha256": hashlib.sha256(raw).hexdigest()})
        served = _field(value, "model")
        if isinstance(served, str) and served and served != context.model_alias:
            raise _TransportFailure("environment_error", "served_model_mismatch",
                                    f"the server answered as {served!r}, not {context.model_alias!r}")
        return value

    def _execute(self, context: BenchmarkContext, client: Any, exercise: Exercise, files: Mapping[str, str],
                 attempt_index: int, timeout: int, session_id: str, attempt_id: str, upstream_commit: str,
                 evidence: dict[str, Any]) -> ExecutionOutcome:
        command = TEST_COMMANDS.get(exercise.language)
        if not command:
            return ExecutionOutcome("execution_unsupported_language", False, "",
                                    f"no pinned test command for {exercise.language}", False, {})
        remaining = _remaining(context)
        if remaining is None:
            return ExecutionOutcome("execution_timeout", False, "", "no wall budget left for execution",
                                    False, {})
        request = ExecutionRequest(
            session_id=session_id, attempt_id=attempt_id, request_id=uuid.uuid4().hex,
            task_id=exercise.task_id, language=exercise.language, exercise_sha256=exercise.digest,
            attempt_index=attempt_index, workdir=f"{exercise.language}/{exercise.slug}",
            command=pinned_test_command(exercise),
            patch={name: normalize_newlines(content) for name, content in files.items()},
            test_files=exercise.test_files(), support_files=exercise.support_files(),
            timeout_seconds=max(1, min(timeout, int(remaining))), upstream_commit=upstream_commit,
            submitted_utc=utc_now())
        clock = getattr(client, "clock", None)
        now = clock() if callable(clock) else self._clock()
        try:
            client.submit(request)
            raw = client.wait(request.request_id, deadline=now + remaining)
        except BenchmarkAborted:
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            outcome = ExecutionOutcome("execution_timeout", False, "", f"TimeoutError: {exc}", False, {})
            evidence["execution"].append({"attempt": attempt_index, "status": outcome.status,
                                          "error": outcome.failure_reason})
            return outcome
        except Exception as exc:
            outcome = ExecutionOutcome("execution_client_error", False, "",
                                       f"{type(exc).__name__}: {exc}", False, {})
            evidence["execution"].append({"attempt": attempt_index, "status": outcome.status,
                                          "error": outcome.failure_reason})
            return outcome
        outcome = normalize_execution_result(raw, request)
        evidence["execution"].append({"attempt": attempt_index, "status": outcome.status,
                                      "passed": outcome.passed, "error": outcome.failure_reason,
                                      **outcome.info})
        return outcome


class _TransportFailure(RuntimeError):
    def __init__(self, status: str, outcome: str, message: str) -> None:
        super().__init__(message)
        self.status, self.outcome, self.message = status, outcome, message


def _remaining(context: BenchmarkContext) -> float | None:
    """``None`` when the wall budget is gone; ``container_eval``'s callable raises instead of returning 0."""
    try:
        value = float(context.remaining_seconds())
    except TimeoutError:
        return None
    except Exception:
        return None
    return value if value > 0 and math.isfinite(value) else None


ADAPTER = AiderPolyglotBenchmark
