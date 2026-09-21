#!/usr/bin/env python
"""Fetch, normalize and pin the public benchmark datasets the evaluator image bakes.

This is a ROOT-RUN, NETWORK-USING staging tool. It is the only place in the suite that is allowed to
touch the network for benchmark data: it downloads each dataset from its official source, rewrites it
into the exact on-disk layout the matching adapter reads, records real provenance (source URL, upstream
commit or release tag, retrieval timestamp, per-file sha256, license) and leaves the result in an
artifacts directory that ``docker build`` copies into ``/opt/llmbench/benchmarks``. Nothing downstream
ever fetches anything: the evaluator image has no network at run time, and every adapter fails closed
with the exact path it wanted when a dataset is absent.

Four benchmarks, four layouts - each one is dictated by the adapter, never invented here:

``bfcl``              ``bfcl/<adapter revision>/manifest.json`` plus ``data/`` and
                      ``data/possible_answer/``. Bytes come from the ``bfcl_eval`` wheel on PyPI (so
                      ``upstream_package``/``upstream_version`` are literally what was installed) and
                      every staged file is then proven byte-identical to the same path in
                      ``ShishirPatil/gorilla`` at the resolved ``source_commit``. A file that does not
                      match refuses the whole benchmark rather than recording a commit that did not
                      produce these bytes.
``ruler``             ``ruler/essays/*.txt`` - the haystack corpus only ``niah_multiquery`` needs.
                      Each essay's bytes are verified against the git blob id recorded in the pinned
                      commit's tree, so the corpus is provably the one that commit holds.
``evalplus``          ``evalplus/MbppPlus-v0.2.0.jsonl`` (+ ``.sha256`` sidecar), normalized from two
                      assets of the SAME official ``evalplus/mbppplus_release`` v0.2.0 release.
``aider-polyglot``    ``aider-polyglot/pin.json`` plus ``<lang>/exercises/practice/<slug>/`` trees,
                      extracted from the repository tarball at the resolved commit.

Rules this tool keeps, in order of importance:

1. **Real provenance only.** Every commit is 40-hex, resolved from the source at run time and checked
   against the bytes it is supposed to have produced. There is no placeholder, no zero commit and no
   "unknown" default anywhere: a source that cannot be resolved fails that benchmark with the reason
   and the other three still run.
2. **Normalization is declared, never silent.** Where the adapter's schema needs something upstream
   does not ship as-is (BFCL's irrelevance ground truth, MBPP+'s per-test assertions), the derivation
   is implemented here, named in the manifest and explained in ``docs/benchmarks.md``.
3. **Verifiable and re-runnable.** Every staged file is hashed into a per-benchmark manifest and a
   top-level ``staging-manifest.json``; ``--offline-verify`` re-checks an existing directory with no
   network at all and reports a changed byte, a missing file or an extra file.
4. **Nothing is vendored into the repository.** The output directory is under ``artifacts/`` (gitignored).
5. **A benchmark that cannot be staged honestly is not staged.** Its directory is removed and the
   manifest records the failure, so the image simply lacks it and ``container_eval.self_check()`` says so.

Usage (root)::

    python scripts/stage_benchmark_datasets.py --output artifacts/benchmark-datasets
    python scripts/stage_benchmark_datasets.py --output artifacts/benchmark-datasets --offline-verify

``--benchmark`` (repeatable, comma-separated) restricts the run; the default is every benchmark.
Set ``GITHUB_TOKEN`` (or ``GH_TOKEN``) to raise GitHub's anonymous API rate limit; the header is only
ever sent to github.com hosts.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

SCHEMA_VERSION = 1
TOOL_NAME = "scripts/stage_benchmark_datasets.py"
MANIFEST_NAME = "staging-manifest.json"
IMAGE_DATASET_ROOT = "/opt/llmbench/benchmarks"
DEFAULT_OUTPUT = "artifacts/benchmark-datasets"

BENCHMARKS = ("bfcl", "ruler", "evalplus", "aider-polyglot")

HEX40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
GITHUB_HOSTS = frozenset({"api.github.com", "raw.githubusercontent.com", "codeload.github.com"})

DEFAULT_TIMEOUT_SECONDS = 180.0
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024


# ==================================================================================================
# Errors
# ==================================================================================================


class StagingError(RuntimeError):
    """One benchmark could not be staged honestly. It is reported and the others continue."""


# ==================================================================================================
# Small utilities
# ==================================================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_id(data: bytes) -> str:
    """The git object id of a blob: ``sha1("blob <len>\\0" + content)``, as a tree entry records it."""
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                      separators=(",", ": ")).encode("utf-8") + b"\n"


def require_commit(value: Any, *, source: str) -> str:
    """A commit is 40 lowercase hex and never a placeholder. Anything else refuses the benchmark."""
    if type(value) is not str or HEX40.fullmatch(value) is None:
        raise StagingError(f"{source} did not yield a 40-character lowercase hex commit id: {value!r}")
    if len(set(value)) <= 1:
        raise StagingError(f"{source} yielded the placeholder commit {value!r}")
    return value


def confined(relative: str) -> PurePosixPath:
    """Reject anything that is not a plain, relative, dot-free path before it reaches the filesystem."""
    if type(relative) is not str or not relative or relative != relative.strip():
        raise StagingError(f"staged path must be a clean relative path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts:
        raise StagingError(f"staged path must be relative: {relative!r}")
    if pure.as_posix() != relative:
        raise StagingError(f"staged path must already be in normal form: {relative!r}")
    for part in pure.parts:
        if part in {"", ".", ".."} or ":" in part or "\\" in part or any(ord(ch) < 32 for ch in part):
            raise StagingError(f"staged path segment is unusable: {relative!r}")
    return pure


def split_list(values: Iterable[str]) -> tuple[str, ...]:
    """``--flag a,b --flag c`` and ``--flag a --flag b`` both mean the same ordered, de-duplicated set."""
    out: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return tuple(out)


# ==================================================================================================
# Fetching
# ==================================================================================================


class Fetcher(Protocol):
    """The single network seam. The tests inject a fake and never open a socket."""

    def fetch(self, url: str) -> bytes: ...


class UrllibFetcher:
    """Bounded stdlib HTTPS GET. The constructor performs no network activity."""

    def __init__(self, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                 max_bytes: int = MAX_DOWNLOAD_BYTES, token: str | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.max_bytes = int(max_bytes)
        self.token = token or None
        self._cache: dict[str, bytes] = {}

    def fetch(self, url: str) -> bytes:
        from urllib.error import HTTPError, URLError
        from urllib.parse import urlsplit
        from urllib.request import Request, urlopen

        if url in self._cache:
            return self._cache[url]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise StagingError(f"only credential-free https URLs are fetched: {url!r}")
        headers = {"Accept": "*/*", "User-Agent": "llmbench-dataset-staging/1"}
        if self.token and parsed.hostname in GITHUB_HOSTS:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = response.read(self.max_bytes + 1)
        except HTTPError as exc:
            raise StagingError(f"GET {url} returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise StagingError(f"GET {url} failed: {type(exc.reason).__name__}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise StagingError(f"GET {url} timed out after {self.timeout_seconds:g}s") from exc
        if len(data) > self.max_bytes:
            raise StagingError(f"GET {url} exceeded the {self.max_bytes}-byte download bound")
        self._cache[url] = data
        return data


def fetch_json(fetcher: Fetcher, url: str) -> Any:
    raw = fetcher.fetch(url)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StagingError(f"{url} did not return JSON: {exc}") from exc


def github_commit(fetcher: Fetcher, repository: str, ref: str) -> dict[str, Any]:
    """Resolve ``<owner>/<repo>@<ref>`` to a real commit id and its authoring date."""
    payload = fetch_json(fetcher, f"https://api.github.com/repos/{repository}/commits/{ref}")
    if not isinstance(payload, Mapping):
        raise StagingError(f"unexpected commit payload for {repository}@{ref}")
    commit = require_commit(payload.get("sha"), source=f"{repository}@{ref}")
    detail = payload.get("commit") if isinstance(payload.get("commit"), Mapping) else {}
    author = detail.get("author") if isinstance(detail.get("author"), Mapping) else {}
    return {"repository": repository, "ref": ref, "commit": commit,
            "committed_utc": author.get("date") if type(author.get("date")) is str else None}


def github_tree(fetcher: Fetcher, repository: str, commit: str) -> list[dict[str, Any]]:
    payload = fetch_json(fetcher, f"https://api.github.com/repos/{repository}/git/trees/{commit}"
                                  "?recursive=1")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tree"), list):
        raise StagingError(f"unexpected tree payload for {repository}@{commit}")
    if payload.get("truncated") is True:
        raise StagingError(f"the git tree for {repository}@{commit} came back truncated; "
                           "this tool will not stage a partial listing")
    return [entry for entry in payload["tree"] if isinstance(entry, Mapping)]


def raw_github_url(repository: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"


# ==================================================================================================
# Staging primitives
# ==================================================================================================


class Writer:
    """Writes one benchmark's tree and records every byte it wrote."""

    def __init__(self, root: Path, directory: str) -> None:
        self.root = Path(root)
        self.directory = directory
        self.base = self.root / directory
        self.files: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        if self.base.is_symlink():
            raise StagingError(f"{self.base} is a link; refusing to stage into it")
        if self.base.exists():
            shutil.rmtree(self.base)
        self.base.mkdir(parents=True, exist_ok=False)
        self.files.clear()

    def write(self, relative: str, data: bytes) -> str:
        """``relative`` is relative to the benchmark directory; the manifest key is relative to root."""
        pure = confined(relative)
        target = self.base.joinpath(*pure.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        key = f"{self.directory}/{pure.as_posix()}"
        self.files[key] = {"bytes": len(data), "sha256": sha256_bytes(data)}
        return key

    def drop(self, relative: str) -> None:
        pure = confined(relative)
        target = self.base.joinpath(*pure.parts)
        if target.exists():
            target.unlink()
        self.files.pop(f"{self.directory}/{pure.as_posix()}", None)


class Result:
    """One benchmark's outcome. ``status`` is the only thing the caller has to read."""

    STAGED = "staged"
    BLOCKED = "staged_with_blocker"
    FAILED = "failed"

    def __init__(self, benchmark: str, directory: str) -> None:
        self.benchmark = benchmark
        self.directory = directory
        self.status = Result.FAILED
        self.files: dict[str, dict[str, Any]] = {}
        self.provenance: dict[str, Any] = {}
        self.license: dict[str, Any] = {}
        self.normalization: list[str] = []
        self.verification: dict[str, Any] = {}
        self.notes: list[str] = []
        self.blocker: str | None = None
        self.error: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(entry["bytes"] for entry in self.files.values())

    def as_dict(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "directory": self.directory, "status": self.status,
                "file_count": len(self.files), "total_bytes": self.total_bytes,
                "provenance": self.provenance, "license": self.license,
                "normalization": list(self.normalization), "verification": self.verification,
                "notes": list(self.notes), "blocker": self.blocker, "error": self.error,
                "files": dict(sorted(self.files.items()))}


def _import_llmbench(module: str) -> Any:
    """Import an adapter for validation, adding ``src`` to the path when the tree is not installed."""
    source = Path(__file__).resolve().parent.parent / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import importlib

    return importlib.import_module(module)


# ==================================================================================================
# BFCL
# ==================================================================================================

BFCL_DIRECTORY = "bfcl"
BFCL_PYPI_PACKAGE = "bfcl-eval"
BFCL_WHEEL_DATA_PREFIX = "bfcl_eval/data/"
BFCL_GORILLA_REPOSITORY = "ShishirPatil/gorilla"
BFCL_GORILLA_DATA_PREFIX = "berkeley-function-call-leaderboard/bfcl_eval/data/"

BFCL_CATEGORY_FILES: dict[str, str] = {
    # ``simple`` is BFCL v4's *python* simple split. The java and javascript splits encode their
    # arguments as language-specific strings that this adapter's AST matcher does not model, so they
    # are deliberately not baked; see docs/benchmarks.md.
    "simple": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
    "live_simple": "BFCL_v4_live_simple.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
    "live_parallel": "BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
    "live_irrelevance": "BFCL_v4_live_irrelevance.json",
}
BFCL_DERIVED_ANSWERS = ("irrelevance", "live_irrelevance")
"""Upstream ships no ``possible_answer`` file for these: passing IS emitting no call. The answers file
is derived here as ``{"id": ..., "ground_truth": []}`` per record, which is exactly what the adapter
validates for an irrelevance category, and the derivation is declared in the bundle manifest."""


def _bfcl_rows(data: bytes, what: str) -> list[dict[str, Any]]:
    text = data.decode("utf-8")
    stripped = text.lstrip()
    try:
        rows = json.loads(text) if stripped.startswith("[") else [
            json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError as exc:
        raise StagingError(f"{what} is not JSON or JSON-Lines: {exc}") from exc
    if not all(type(row) is dict for row in rows):
        raise StagingError(f"{what} must hold JSON objects")
    return rows


def stage_bfcl(fetcher: Fetcher, root: Path, *, version: str | None, categories: Sequence[str],
               gorilla_ref: str, validate: bool) -> Result:
    revision_module = _import_llmbench("llmbench.benchmarks.bfcl")
    revision = revision_module.REVISION
    result = Result("bfcl", BFCL_DIRECTORY)
    writer = Writer(root, BFCL_DIRECTORY)
    writer.reset()
    bundle_prefix = f"{revision}/"

    index = fetch_json(fetcher, f"https://pypi.org/pypi/{BFCL_PYPI_PACKAGE}/json")
    resolved = version or (index.get("info") or {}).get("version")
    if type(resolved) is not str or not resolved.strip():
        raise StagingError("PyPI did not report a bfcl-eval version")
    release = fetch_json(fetcher, f"https://pypi.org/pypi/{BFCL_PYPI_PACKAGE}/{resolved}/json")
    wheels = [item for item in (release.get("urls") or []) if item.get("packagetype") == "bdist_wheel"]
    if len(wheels) != 1:
        raise StagingError(f"bfcl-eval {resolved} does not publish exactly one wheel on PyPI")
    wheel_url = wheels[0]["url"]
    declared_sha256 = (wheels[0].get("digests") or {}).get("sha256")
    payload = fetcher.fetch(wheel_url)
    actual_sha256 = sha256_bytes(payload)
    if declared_sha256 and actual_sha256 != declared_sha256:
        raise StagingError(f"the downloaded bfcl-eval wheel hashes {actual_sha256}, PyPI declares "
                           f"{declared_sha256}")

    commit_info = github_commit(fetcher, BFCL_GORILLA_REPOSITORY, gorilla_ref)
    commit = commit_info["commit"]

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise StagingError(f"the bfcl-eval wheel is not a readable zip: {exc}") from exc

    selected = [name for name in categories if name in BFCL_CATEGORY_FILES]
    unknown = [name for name in categories if name not in BFCL_CATEGORY_FILES]
    if unknown:
        raise StagingError(f"not AST categories this adapter can bake: {', '.join(sorted(unknown))}")
    if not selected:
        raise StagingError("no BFCL categories were selected")

    entries: dict[str, dict[str, Any]] = {}
    verified_files: list[str] = []
    for category in selected:
        filename = BFCL_CATEGORY_FILES[category]
        records = _read_member(archive, BFCL_WHEEL_DATA_PREFIX + filename)
        _verify_against_gorilla(fetcher, commit, filename, records, verified_files)
        rows = _bfcl_rows(records, filename)
        writer.write(f"{bundle_prefix}data/{filename}", records)
        answers_relative = f"{bundle_prefix}data/possible_answer/{filename}"
        if category in BFCL_DERIVED_ANSWERS:
            derived = _bfcl_derived_answers(rows, filename)
            writer.write(answers_relative, derived)
            answers_sha256 = sha256_bytes(derived)
            answers_origin = "derived: an irrelevance record's correct answer is no call at all"
        else:
            answers = _read_member(archive, BFCL_WHEEL_DATA_PREFIX + "possible_answer/" + filename)
            _verify_against_gorilla(fetcher, commit, "possible_answer/" + filename, answers,
                                    verified_files)
            writer.write(answers_relative, answers)
            answers_sha256 = sha256_bytes(answers)
            answers_origin = f"{BFCL_WHEEL_DATA_PREFIX}possible_answer/{filename}"
        entries[category] = {
            "records": f"data/{filename}", "answers": f"data/possible_answer/{filename}",
            "count": len(rows),
            "sha256": {"records": sha256_bytes(records), "answers": answers_sha256},
            "upstream_records": BFCL_WHEEL_DATA_PREFIX + filename, "upstream_answers": answers_origin,
        }

    accepted, rejected = entries, {}
    if validate:
        accepted, rejected = _bfcl_accepted_categories(root, writer, revision, resolved, commit, entries)
        for category in rejected:
            filename = BFCL_CATEGORY_FILES[category]
            writer.drop(f"{bundle_prefix}data/{filename}")
            writer.drop(f"{bundle_prefix}data/possible_answer/{filename}")
    if not accepted:
        raise StagingError("every candidate BFCL category was rejected by the adapter's own loader: "
                           + "; ".join(f"{name}: {reason}" for name, reason in rejected.items()))

    manifest = _bfcl_manifest(revision, resolved, commit, accepted)
    writer.write(f"{bundle_prefix}manifest.json", manifest)

    result.files = dict(writer.files)
    result.status = Result.STAGED
    result.provenance = {
        "upstream_package": BFCL_PYPI_PACKAGE, "upstream_version": resolved,
        "bytes_from": wheel_url, "wheel_sha256": actual_sha256,
        "wheel_sha256_declared_by_pypi": declared_sha256,
        "source_repository": f"https://github.com/{BFCL_GORILLA_REPOSITORY}",
        "source_commit": commit, "source_ref": gorilla_ref,
        "source_committed_utc": commit_info["committed_utc"], "retrieved_utc": utc_now(),
        "adapter_revision": revision, "categories": sorted(accepted),
        "rejected_categories": rejected,
    }
    result.license = {"spdx_id": "Apache-2.0",
                      "holder": "The Gorilla / Berkeley Function Calling Leaderboard authors",
                      "source": f"https://github.com/{BFCL_GORILLA_REPOSITORY}"}
    result.normalization = [
        "``simple`` is baked from BFCL_v4_simple_python.json; the java and javascript simple splits are "
        "not baked because this adapter's AST matcher does not model their argument encodings",
        "the irrelevance categories get a derived answers file ({\"id\": ..., \"ground_truth\": []}); "
        "upstream ships none because the correct answer is emitting no call",
    ]
    baked = {BFCL_CATEGORY_FILES[name] for name in accepted}
    result.verification = {
        "wheel_sha256_matches_pypi": bool(declared_sha256) and actual_sha256 == declared_sha256,
        "files_byte_identical_to_source_commit": sorted(
            name for name in verified_files if PurePosixPath(name).name in baked),
    }
    if validate:
        result.verification["adapter_load"] = _bfcl_load_summary(root, revision)
        if rejected:
            result.notes.append("categories rejected by llmbench.benchmarks.bfcl.load_bundle and left "
                                "unbaked: " + "; ".join(f"{k}: {v}" for k, v in rejected.items()))
    return result


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError as exc:
        raise StagingError(f"the bfcl-eval wheel has no member {name}") from exc


def _verify_against_gorilla(fetcher: Fetcher, commit: str, relative: str, payload: bytes,
                            verified: list[str]) -> None:
    url = raw_github_url(BFCL_GORILLA_REPOSITORY, commit, BFCL_GORILLA_DATA_PREFIX + relative)
    upstream = fetcher.fetch(url)
    if sha256_bytes(upstream) != sha256_bytes(payload):
        raise StagingError(
            f"{relative} in the PyPI wheel is not byte-identical to {BFCL_GORILLA_REPOSITORY}@"
            f"{commit[:12]}; refusing to record that commit as the source of these bytes")
    verified.append(relative)


def _bfcl_derived_answers(rows: Sequence[Mapping[str, Any]], filename: str) -> bytes:
    lines = []
    for row in rows:
        identifier = row.get("id")
        if type(identifier) is not str or not identifier:
            raise StagingError(f"{filename}: every record needs a string id to derive its answer row")
        lines.append(json.dumps({"id": identifier, "ground_truth": []}, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bfcl_manifest(revision: str, version: str, commit: str, categories: Mapping[str, Any]) -> bytes:
    return canonical_json({
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": revision,
        "upstream_package": BFCL_PYPI_PACKAGE,
        "upstream_version": version,
        "source_commit": commit,
        "source_repository": f"https://github.com/{BFCL_GORILLA_REPOSITORY}",
        "license": "Apache-2.0",
        "staged_by": TOOL_NAME,
        "staged_utc": utc_now(),
        "categories": {name: dict(entry) for name, entry in sorted(categories.items())},
    })


def _bfcl_accepted_categories(root: Path, writer: Writer, revision: str, version: str, commit: str,
                              entries: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Ask the adapter's own loader, one category at a time, which bundles it will accept."""
    module = _import_llmbench("llmbench.benchmarks.bfcl")
    relative = f"{revision}/manifest.json"
    accepted: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    for name, entry in entries.items():
        writer.write(relative, _bfcl_manifest(revision, version, commit, {name: entry}))
        try:
            module.load_bundle(root)
        except Exception as exc:
            rejected[name] = f"{type(exc).__name__}: {exc}"
        else:
            accepted[name] = entry
    writer.drop(relative)
    return accepted, rejected


def _bfcl_load_summary(root: Path, revision: str) -> dict[str, Any]:
    module = _import_llmbench("llmbench.benchmarks.bfcl")
    bundle = module.load_bundle(root)
    splits = {"development": 0, "holdout": 0}
    for record in bundle.records:
        splits[module.split_of(record.record_id)] += 1
    return {"revision": revision, "records": len(bundle.records),
            "categories": sorted({record.category for record in bundle.records}), "splits": splits}


# ==================================================================================================
# RULER
# ==================================================================================================

RULER_DIRECTORY = "ruler"
RULER_DEFAULT_REPOSITORY = "gkamradt/needle-in-a-haystack"
RULER_DEFAULT_PREFIX = "needlehaystack/PaulGrahamEssays/"
RULER_MIN_ESSAYS = 8


def stage_ruler(fetcher: Fetcher, root: Path, *, repository: str, ref: str, prefix: str,
                validate: bool) -> Result:
    result = Result("ruler", RULER_DIRECTORY)
    writer = Writer(root, RULER_DIRECTORY)
    writer.reset()

    commit_info = github_commit(fetcher, repository, ref)
    commit = commit_info["commit"]
    tree = github_tree(fetcher, repository, commit)
    blobs = sorted((entry for entry in tree
                    if entry.get("type") == "blob"
                    and str(entry.get("path", "")).startswith(prefix)
                    and str(entry.get("path", "")).endswith(".txt")),
                   key=lambda entry: str(entry["path"]))
    if len(blobs) < RULER_MIN_ESSAYS:
        raise StagingError(f"{repository}@{commit[:12]} holds {len(blobs)} essay files under {prefix}; "
                           f"at least {RULER_MIN_ESSAYS} are needed for a haystack")
    total = 0
    for entry in blobs:
        path = str(entry["path"])
        name = PurePosixPath(path).name
        if SAFE_SEGMENT.fullmatch(name) is None:
            raise StagingError(f"essay filename is outside the safe grammar: {name!r}")
        payload = fetcher.fetch(raw_github_url(repository, commit, path))
        blob_id = str(entry.get("sha") or "")
        if HEX40.fullmatch(blob_id) is None or git_blob_id(payload) != blob_id:
            raise StagingError(f"{path} does not match the git blob id recorded in {repository}@"
                               f"{commit[:12]}")
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StagingError(f"{path} is not UTF-8: {exc}") from exc
        writer.write(f"essays/{name}", payload)
        total += len(payload)

    result.files = dict(writer.files)
    result.status = Result.STAGED
    result.provenance = {
        "source_repository": f"https://github.com/{repository}", "source_commit": commit,
        "source_ref": ref, "source_path_prefix": prefix,
        "source_committed_utc": commit_info["committed_utc"], "retrieved_utc": utc_now(),
        "essay_count": len(blobs), "essay_bytes": total,
        "consumed_by": "llmbench.benchmarks.ruler_tasks.corpus (RULER niah_multiquery only)",
    }
    result.license = {
        "spdx_id": None,
        "summary": "Essay text is Copyright Paul Graham (paulgraham.com), redistributed by the "
                   "needle-in-a-haystack benchmark repository; the repository itself carries a "
                   "LICENSE.txt GitHub classifies as NOASSERTION. Staged as benchmark haystack text "
                   "only, not re-licensed and not vendored into this repository.",
        "source": f"https://github.com/{repository}",
    }
    result.normalization = [
        "staged into the adapter's documented ``ruler/essays/*.txt`` fallback rather than synthesising "
        "upstream's single ``PaulGrahamEssays.json`` dump, so the bytes stay exactly what the pinned "
        "commit holds",
    ]
    result.verification = {"git_blob_ids_match_pinned_tree": True}
    if validate:
        result.verification.update(_ruler_corpus_summary(root))
        supported = result.verification.get("max_supported_target_tokens_estimate")
        if isinstance(supported, int) and supported < 131072:
            result.notes.append(
                f"the staged corpus supports RULER lengths up to roughly {supported} tokens without "
                "repetition; longer declared lengths will fail closed as length_target_unmet, which is "
                "the vendored generator's documented refusal to repeat the haystack")
    return result


def _ruler_corpus_summary(root: Path) -> dict[str, Any]:
    corpus = _import_llmbench("llmbench.benchmarks.ruler_tasks.corpus")
    status = corpus.essay_status(str(root))
    if not status.present:
        raise StagingError(f"the staged corpus is not visible to the adapter: {status.detail}")
    words = corpus.essay_words(str(root))
    sentences = corpus.essay_sentences(str(root))
    # RULER grows the haystack in WORDS; English prose on a BPE tokenizer runs near 0.75 words per
    # token, so this is the deliberately conservative length the corpus can reach without repeating.
    return {"corpus_words": len(words), "corpus_sentences": len(sentences),
            "max_supported_target_tokens_estimate": int(len(words) / 0.75)}


# ==================================================================================================
# EvalPlus MBPP+
# ==================================================================================================

EVALPLUS_DIRECTORY = "evalplus"
EVALPLUS_RELEASE_REPOSITORY = "evalplus/mbppplus_release"
EVALPLUS_DEFAULT_TAG = "v0.2.0"
EVALPLUS_CANONICAL_ASSET = "MbppPlus.jsonl.gz"
EVALPLUS_ORIGIN_ASSET = "MbppPlus-OriginFmt.jsonl.gz"


class UpstreamShape(StagingError):
    """One upstream record does not have the shape this normalization understands."""


def stage_evalplus(fetcher: Fetcher, root: Path, *, tag: str, validate: bool,
                   trim_oversize: bool = False) -> Result:
    module = _import_llmbench("llmbench.benchmarks.evalplus")
    pin = module.EVALPLUS_DATASETS["mbpp-plus"]
    result = Result("evalplus", EVALPLUS_DIRECTORY)
    writer = Writer(root, EVALPLUS_DIRECTORY)
    writer.reset()

    release = fetch_json(fetcher, f"https://api.github.com/repos/{EVALPLUS_RELEASE_REPOSITORY}"
                                  f"/releases/tags/{tag}")
    assets = {item.get("name"): item for item in (release.get("assets") or [])
              if isinstance(item, Mapping)}
    for name in (EVALPLUS_CANONICAL_ASSET, EVALPLUS_ORIGIN_ASSET):
        if name not in assets:
            raise StagingError(f"{EVALPLUS_RELEASE_REPOSITORY} {tag} has no asset {name}")
    canonical_raw = fetcher.fetch(assets[EVALPLUS_CANONICAL_ASSET]["browser_download_url"])
    origin_raw = fetcher.fetch(assets[EVALPLUS_ORIGIN_ASSET]["browser_download_url"])
    canonical = _jsonl_gz(canonical_raw, EVALPLUS_CANONICAL_ASSET)
    origin = _jsonl_gz(origin_raw, EVALPLUS_ORIGIN_ASSET)

    by_task = {row.get("task_id"): row for row in canonical}
    origin_by_task = {f"{pin.upstream_prefix}/{row.get('task_id')}": row for row in origin}
    if len(by_task) != len(canonical) or len(origin_by_task) != len(origin):
        raise StagingError("MBPP+ task ids are not unique in the published assets")
    if set(by_task) != set(origin_by_task):
        raise StagingError("the two published MBPP+ v0.2.0 assets do not cover the same task ids")

    records: list[dict[str, Any]] = []
    base_verified = 0
    oversize: list[str] = []
    trimmed: dict[str, dict[str, int]] = {}
    for task_id in sorted(by_task, key=_mbpp_order):
        record, verified = _mbpp_record(by_task[task_id], origin_by_task[task_id], pin)
        base_verified += int(verified)
        module_text = _composed_module_size(module, record, pin)
        if module_text > module.MAX_TEST_MODULE_BYTES:
            if trim_oversize:
                before = len(record["tests"])
                _mbpp_trim(module, record, pin)
                trimmed[task_id] = {"tests_before": before, "tests_after": len(record["tests"]),
                                    "composed_bytes_before": module_text,
                                    "composed_bytes_after": _composed_module_size(module, record, pin)}
            else:
                oversize.append(f"{task_id} ({module_text} bytes)")
        records.append(record)
    if len(records) != pin.item_count:
        raise StagingError(f"MBPP+ {tag} normalized to {len(records)} records; the adapter pins "
                           f"{pin.item_count}")

    payload = ("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                       for record in records)).encode("utf-8")
    writer.write(pin.filename, payload)
    digest = sha256_bytes(payload)
    writer.write(pin.filename + ".sha256", (digest + "\n").encode("ascii"))

    result.files = dict(writer.files)
    result.provenance = {
        "upstream_repository": f"https://github.com/{EVALPLUS_RELEASE_REPOSITORY}",
        "release_tag": tag, "release_published_utc": release.get("published_at"),
        "release_html_url": release.get("html_url"),
        "assets": {
            EVALPLUS_CANONICAL_ASSET: {"sha256": sha256_bytes(canonical_raw),
                                       "bytes": len(canonical_raw),
                                       "url": assets[EVALPLUS_CANONICAL_ASSET]["browser_download_url"]},
            EVALPLUS_ORIGIN_ASSET: {"sha256": sha256_bytes(origin_raw), "bytes": len(origin_raw),
                                    "url": assets[EVALPLUS_ORIGIN_ASSET]["browser_download_url"]},
        },
        "retrieved_utc": utc_now(), "records": len(records), "dataset_sha256": digest,
        "adapter_revision": pin.revision,
    }
    result.license = {"spdx_id": "Apache-2.0", "holder": "The EvalPlus authors",
                      "source": "https://github.com/evalplus/evalplus"}
    result.normalization = [
        "prompt/entry_point come from MbppPlus.jsonl; the per-test assertions come from the same "
        "release's MbppPlus-OriginFmt.jsonl, whose expected outputs are already materialized upstream - "
        "no upstream code is executed by this tool to produce ground truth",
        "each upstream ``for i, (inp, exp) in enumerate(zip(inputs, results))`` iteration becomes one "
        "test record whose code is the loop body with ``inp``/``exp`` replaced by that iteration's "
        "literal AST nodes and re-emitted with ast.unparse",
        "the harness preamble (imports plus the upstream ``assertion``/``is_floats`` helpers, and the "
        "``ref_func`` reference implementation three tasks compare against) is carried verbatim into "
        "``test_setup``",
        "tests are labelled base/<n> for the first len(base_input) iterations and plus/<n> thereafter, "
        "the boundary being the base_input length published in MbppPlus.jsonl",
    ]
    result.verification = {
        "task_ids_agree_across_both_assets": True,
        "base_prefix_matches_base_input": f"{base_verified}/{len(records)}",
        "trimmed_items": trimmed,
    }
    if trimmed:
        result.normalization.append(
            "--evalplus-trim-oversize was used: an item whose composed test module exceeds the "
            "adapter's MAX_TEST_MODULE_BYTES keeps every base test plus as many plus tests, in "
            "upstream order, as fit. The exact before/after test counts are in "
            "verification.trimmed_items; those items are NOT full MBPP+ items.")
        result.notes.append("trimmed to fit the adapter's test-module cap: " + "; ".join(
            f"{task}: {counts['tests_before']} -> {counts['tests_after']} tests"
            for task, counts in sorted(trimmed.items())))
    if oversize:
        result.status = Result.BLOCKED
        result.blocker = (
            f"{len(oversize)} item(s) compose a test module larger than the adapter's "
            f"MAX_TEST_MODULE_BYTES={module.MAX_TEST_MODULE_BYTES}: {', '.join(oversize)}. "
            "load_dataset() is all-or-nothing and the pin demands exactly "
            f"{pin.item_count} items, so the staged file is complete and correct but the adapter will "
            "report evalplus unavailable until that constant is raised in "
            "src/llmbench/benchmarks/evalplus.py. Nothing here is truncated to hide it.")
    else:
        result.status = Result.STAGED
    if validate:
        result.verification["adapter_load"] = _evalplus_load_summary(module, root, pin, result)
    return result


def _mbpp_order(task_id: str) -> tuple[int, str]:
    tail = task_id.rsplit("/", 1)[-1]
    return (int(tail) if tail.isdigit() else 1 << 30, task_id)


def _jsonl_gz(payload: bytes, what: str) -> list[dict[str, Any]]:
    try:
        text = gzip.decompress(payload).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise StagingError(f"{what} is not a readable gzip of UTF-8 text: {exc}") from exc
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise StagingError(f"{what} line {number} is not JSON: {exc}") from exc
        if type(row) is not dict:
            raise StagingError(f"{what} line {number} is not a JSON object")
        rows.append(row)
    return rows


def _quiet_upstream_syntax_warnings():
    """Some upstream MBPP harnesses embed regexes with unescaped backslashes; parsing them is noisy."""
    import contextlib
    import warnings

    @contextlib.contextmanager
    def suppressed():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            yield

    return suppressed()


class _LoopSubstitution(ast.NodeTransformer):
    """Replace the upstream loop's ``inp``/``exp``/``i`` with one iteration's own literal nodes."""

    def __init__(self, replacements: Mapping[str, ast.AST]) -> None:
        self.replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 - ast visitor naming
        if isinstance(node.ctx, ast.Load) and node.id in self.replacements:
            return self.replacements[node.id]
        return node


def _mbpp_record(canonical: Mapping[str, Any], origin: Mapping[str, Any], pin: Any
                 ) -> tuple[dict[str, Any], bool]:
    task_id = canonical.get("task_id")
    entry_point = canonical.get("entry_point")
    prompt = canonical.get("prompt")
    if type(task_id) is not str or type(entry_point) is not str or type(prompt) is not str:
        raise UpstreamShape(f"{task_id!r}: task_id, entry_point and prompt must be strings")
    harness = origin.get("test")
    if type(harness) is not str or not harness.strip():
        raise UpstreamShape(f"{task_id}: the OriginFmt record carries no test harness")
    with _quiet_upstream_syntax_warnings():
        setup, tests_source = _mbpp_split_harness(task_id, harness, entry_point)
    base_input = canonical.get("base_input")
    if type(base_input) is not list:
        raise UpstreamShape(f"{task_id}: base_input must be a list")
    base_count = len(base_input)
    if base_count < 1 or base_count > len(tests_source):
        raise UpstreamShape(f"{task_id}: base_input declares {base_count} cases but the harness has "
                            f"{len(tests_source)}")
    tests = []
    base_seen = plus_seen = 0
    for index, code in enumerate(tests_source):
        if index < base_count:
            tests.append({"test_id": f"base/{base_seen}", "kind": "base", "code": code})
            base_seen += 1
        else:
            tests.append({"test_id": f"plus/{plus_seen}", "kind": "plus", "code": code})
            plus_seen += 1
    record: dict[str, Any] = {"task_id": task_id, "entry_point": entry_point, "prompt": prompt,
                              "tests": tests}
    if setup:
        record["test_setup"] = setup
    signature = _mbpp_signature(canonical.get("canonical_solution"), entry_point)
    if signature:
        record["signature"] = signature
    with _quiet_upstream_syntax_warnings():
        verified = _mbpp_base_prefix_matches(harness, base_input, base_count)
    if not task_id.startswith(pin.upstream_prefix + "/"):
        raise UpstreamShape(f"{task_id}: task id must start with {pin.upstream_prefix}/")
    return record, verified


def _mbpp_split_harness(task_id: str, harness: str, entry_point: str) -> tuple[str, list[str]]:
    """Cut the upstream harness into (preamble, one code string per iteration)."""
    try:
        tree = ast.parse(harness)
    except SyntaxError as exc:
        raise UpstreamShape(f"{task_id}: the upstream harness does not parse: {exc}") from exc
    if not tree.body or not isinstance(tree.body[-1], ast.For):
        raise UpstreamShape(f"{task_id}: the upstream harness must end in the assertion loop")
    loop = tree.body[-1]
    if len(loop.body) != 1 or loop.orelse:
        raise UpstreamShape(f"{task_id}: the assertion loop must have exactly one body statement")
    assigns = {node.targets[0].id: node for node in tree.body
               if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name)}
    if "inputs" not in assigns or not isinstance(assigns["inputs"].value, ast.List):
        raise UpstreamShape(f"{task_id}: the upstream harness must assign a list to ``inputs``")
    inputs = assigns["inputs"].value.elts
    results: list[ast.expr] | None = None
    if "results" in assigns:
        if not isinstance(assigns["results"].value, ast.List):
            raise UpstreamShape(f"{task_id}: ``results`` must be a list")
        results = assigns["results"].value.elts
        if len(results) != len(inputs):
            raise UpstreamShape(f"{task_id}: inputs and results have different lengths")
    names = _mbpp_loop_names(task_id, loop, with_results=results is not None)
    body = loop.body[0]
    if not any(isinstance(node, ast.Name) and node.id == entry_point for node in ast.walk(body)):
        raise UpstreamShape(f"{task_id}: the loop body never calls {entry_point}")
    setup = "\n".join(harness.splitlines()[:assigns["inputs"].lineno - 1]).rstrip() + "\n"
    codes = []
    for index in range(len(inputs)):
        replacements: dict[str, ast.AST] = {names["index"]: ast.Constant(index),
                                            names["input"]: inputs[index]}
        if results is not None:
            replacements[names["expected"]] = results[index]
        clone = _LoopSubstitution(replacements).visit(_deep_copy(body))
        ast.fix_missing_locations(clone)
        codes.append(ast.unparse(clone))
    if not codes:
        raise UpstreamShape(f"{task_id}: the upstream harness declares no inputs")
    return setup, codes


def _mbpp_loop_names(task_id: str, loop: ast.For, *, with_results: bool) -> dict[str, str]:
    target = loop.target
    if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
        raise UpstreamShape(f"{task_id}: the assertion loop target must be ``i, ...``")
    first = target.elts[0]
    if not isinstance(first, ast.Name):
        raise UpstreamShape(f"{task_id}: the assertion loop index must be a plain name")
    second = target.elts[1]
    if with_results:
        if not isinstance(second, ast.Tuple) or len(second.elts) != 2 \
                or not all(isinstance(item, ast.Name) for item in second.elts):
            raise UpstreamShape(f"{task_id}: the assertion loop must unpack ``(inp, exp)``")
        return {"index": first.id, "input": second.elts[0].id, "expected": second.elts[1].id}
    if not isinstance(second, ast.Name):
        raise UpstreamShape(f"{task_id}: the assertion loop must bind a plain input name")
    return {"index": first.id, "input": second.id, "expected": ""}


def _deep_copy(node: ast.AST) -> ast.AST:
    import copy

    return copy.deepcopy(node)


def _mbpp_signature(solution: Any, entry_point: str) -> str:
    if type(solution) is not str:
        return ""
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            try:
                rendered = f"def {entry_point}({ast.unparse(node.args)})"
            except Exception:
                return ""
            return rendered if len(rendered.encode("utf-8")) <= 512 and "\n" not in rendered else ""
    return ""


def _mbpp_literal(node: ast.AST) -> Any:
    """Evaluate an upstream data literal. ``inf``/``nan`` are names imported by the harness preamble."""
    import math

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_mbpp_literal(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [_mbpp_literal(item) for item in node.elts]
    if isinstance(node, ast.Set):
        return ("<set>", frozenset(_mbpp_literal(item) for item in node.elts))
    if isinstance(node, ast.Dict):
        return ("<dict>", tuple((_mbpp_literal(key), _mbpp_literal(value))
                                for key, value in zip(node.keys, node.values)))
    if isinstance(node, ast.Name) and node.id in {"inf", "nan"}:
        return math.inf if node.id == "inf" else math.nan
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _mbpp_literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left, right = _mbpp_literal(node.left), _mbpp_literal(node.right)
        return left + right if isinstance(node.op, ast.Add) else left * right
    raise ValueError(f"unsupported literal node {type(node).__name__}")


def _mbpp_canonical(value: Any) -> Any:
    import math

    if isinstance(value, tuple) and len(value) == 2 and value[0] == "<set>":
        return ["<set>", sorted(repr(_mbpp_canonical(item)) for item in value[1])]
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "<dict>":
        return ["<dict>", sorted([repr(_mbpp_canonical(k)), repr(_mbpp_canonical(v))]
                                 for k, v in value[1])]
    if isinstance(value, (list, tuple)):
        return [_mbpp_canonical(item) for item in value]
    if isinstance(value, dict):
        return ["<dict>", sorted([repr(_mbpp_canonical(k)), repr(_mbpp_canonical(v))]
                                 for k, v in value.items())]
    if isinstance(value, float) and math.isnan(value):
        return "<nan>"
    if isinstance(value, complex):
        return f"<complex>{value}"
    return value


def _mbpp_base_prefix_matches(harness: str, base_input: Sequence[Any], base_count: int) -> bool:
    """Evidence, not a gate: do the first ``base_count`` harness inputs equal the published base_input?"""
    try:
        tree = ast.parse(harness)
        assigns = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)}
        elements = assigns["inputs"].elts[:base_count]
        staged = [_mbpp_canonical(_mbpp_literal(item)) for item in elements]
    except (KeyError, ValueError, AttributeError, SyntaxError):
        return False
    return staged == [_mbpp_canonical(item) for item in base_input]


def _mbpp_trim(module: Any, record: dict[str, Any], pin: Any) -> None:
    """Keep every base test, then plus tests in upstream order while the composed module still fits.

    Only reachable through ``--evalplus-trim-oversize``. It weakens that item's MBPP+ test suite, which
    is why it is opt-in and why the counts it changes are recorded in the manifest.
    """
    tests = list(record["tests"])
    base = [test for test in tests if test["kind"] == "base"]
    plus = [test for test in tests if test["kind"] == "plus"]
    record["tests"] = base or tests[:1]
    if _composed_module_size(module, record, pin) > module.MAX_TEST_MODULE_BYTES:
        raise StagingError(f"{record['task_id']}: even the base tests alone exceed "
                           f"MAX_TEST_MODULE_BYTES={module.MAX_TEST_MODULE_BYTES}")
    kept = list(record["tests"])
    for test in plus:
        record["tests"] = kept + [test]
        if _composed_module_size(module, record, pin) > module.MAX_TEST_MODULE_BYTES:
            break
        kept = list(record["tests"])
    record["tests"] = kept


def _composed_module_size(module: Any, record: Mapping[str, Any], pin: Any) -> int:
    item = module.EvalPlusItem(
        task_id=f"evalplus/{pin.key}/{record['task_id']}", upstream_id=record["task_id"],
        upstream_index=0, entry_point=record["entry_point"], prompt=record["prompt"],
        signature=record.get("signature", ""), test_setup=record.get("test_setup", ""),
        tests=tuple(module.EvalPlusTest(test["test_id"], test["kind"], test["code"])
                    for test in record["tests"]),
        item_sha256="0" * 64)
    return len(module.build_test_module(item).encode("utf-8"))


def _evalplus_load_summary(module: Any, root: Path, pin: Any, result: Result) -> dict[str, Any]:
    path = root / EVALPLUS_DIRECTORY / pin.filename
    try:
        with _quiet_upstream_syntax_warnings():
            dataset = module.load_dataset(path, pin)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    tests = sum(len(item.tests) for item in dataset.items)
    return {"ok": True, "items": len(dataset.items), "tests": tests, "sha256": dataset.sha256}


# ==================================================================================================
# Aider Polyglot
# ==================================================================================================

AIDER_DIRECTORY = "aider-polyglot"
AIDER_REPOSITORY = "Aider-AI/polyglot-benchmark"
AIDER_MAX_MEMBER_BYTES = 1024 * 1024
AIDER_MAX_MEMBERS = 20000


def stage_aider_polyglot(fetcher: Fetcher, root: Path, *, ref: str, languages: Sequence[str],
                         toolchains: Sequence[str], validate: bool) -> Result:
    module = _import_llmbench("llmbench.benchmarks.aider_polyglot")
    result = Result("aider-polyglot", AIDER_DIRECTORY)
    writer = Writer(root, AIDER_DIRECTORY)
    writer.reset()

    unknown = [name for name in languages if name not in module.UPSTREAM_LANGUAGES]
    if unknown:
        raise StagingError(f"upstream has no language(s) {', '.join(unknown)}; it carries "
                           f"{', '.join(module.UPSTREAM_LANGUAGES)} and no TypeScript")
    if not languages:
        raise StagingError("at least one language must be staged")

    commit_info = github_commit(fetcher, AIDER_REPOSITORY, ref)
    commit = commit_info["commit"]
    payload = fetcher.fetch(f"https://codeload.github.com/{AIDER_REPOSITORY}/tar.gz/{commit}")
    wanted = tuple(f"{language}/exercises/practice/" for language in languages)
    counts: dict[str, set[str]] = {language: set() for language in languages}
    members = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            members += 1
            if members > AIDER_MAX_MEMBERS:
                raise StagingError("the upstream tarball holds more members than this tool will read")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise StagingError(f"refusing a non-regular tar member: {member.name!r}")
            if not member.isfile():
                continue
            relative = _aider_relative(member.name)
            if relative is None or not relative.startswith(wanted):
                continue
            if member.size > AIDER_MAX_MEMBER_BYTES:
                raise StagingError(f"{relative} is {member.size} bytes; the per-file bound is "
                                   f"{AIDER_MAX_MEMBER_BYTES}")
            handle = archive.extractfile(member)
            if handle is None:
                raise StagingError(f"{relative} could not be read from the tarball")
            writer.write(relative, handle.read())
            parts = relative.split("/")
            counts[parts[0]].add(parts[3])
    missing = [language for language in languages if not counts[language]]
    if missing:
        raise StagingError(f"the tarball carried no practice exercises for {', '.join(missing)}")
    expected = module.UPSTREAM_EXERCISE_COUNTS
    mismatched = {language: (len(counts[language]), expected[language]) for language in languages
                  if expected.get(language) is not None and len(counts[language]) != expected[language]}

    retrieved = utc_now()
    pin = {"schema_version": 1, "repository": module.UPSTREAM_REPOSITORY, "commit": commit,
           "retrieved_utc": retrieved, "languages": list(languages), "toolchains": list(toolchains)}
    writer.write(module.PIN_NAME, canonical_json(pin))

    result.files = dict(writer.files)
    result.status = Result.STAGED
    result.provenance = {
        "source_repository": module.UPSTREAM_REPOSITORY, "source_commit": commit, "source_ref": ref,
        "source_committed_utc": commit_info["committed_utc"], "retrieved_utc": retrieved,
        "languages": list(languages), "toolchains": list(toolchains),
        "exercises": {language: len(counts[language]) for language in languages},
        "tarball": f"https://codeload.github.com/{AIDER_REPOSITORY}/tar.gz/{commit}",
        "tarball_sha256": sha256_bytes(payload),
    }
    result.license = {
        "spdx_id": None,
        "summary": "The exercises are Exercism practice exercises redistributed by "
                   "Aider-AI/polyglot-benchmark. Exercism's exercise content carries Exercism's own "
                   "terms (per-track LICENSE plus contributor attribution) rather than a single clean "
                   "SPDX grant, so no SPDX id is claimed here. Staged for local benchmarking only and "
                   "not vendored into this repository.",
        "source": module.UPSTREAM_REPOSITORY,
    }
    result.normalization = [
        "only the selected languages' ``<lang>/exercises/practice`` trees are staged; the repository's "
        "other languages and top-level files are not copied",
        "``toolchains`` in pin.json records the toolchains ROOT asserts the worker image provides; the "
        "adapter refuses a language the pin does not list",
    ]
    result.verification = {"exercise_counts_match_upstream": not mismatched}
    if mismatched:
        result.notes.append("exercise counts differ from the counts the adapter documents: "
                            + "; ".join(f"{name}: staged {got}, documented {want}"
                                        for name, (got, want) in mismatched.items()))
    if validate:
        result.verification.update(_aider_corpus_summary(module, root, languages))
    return result


def _aider_relative(name: str) -> str | None:
    """Strip the tarball's ``<repo>-<sha>/`` prefix and refuse anything that escapes the tree."""
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise StagingError(f"refusing a tar member with an unsafe path: {name!r}")
    if len(pure.parts) < 2:
        return None
    return PurePosixPath(*pure.parts[1:]).as_posix()


def _aider_corpus_summary(module: Any, root: Path, languages: Sequence[str]) -> dict[str, Any]:
    corpus = root / AIDER_DIRECTORY
    module.read_pin(corpus)
    loaded, failures = 0, []
    for language in languages:
        for task_id, directory in module.enumerate_exercises(corpus, language):
            try:
                module.load_exercise(task_id, directory)
            except Exception as exc:
                failures.append(f"{task_id}: {type(exc).__name__}: {exc}")
            else:
                loaded += 1
    return {"exercises_loadable": loaded, "exercises_rejected": len(failures),
            "rejection_reasons": failures[:20]}


# ==================================================================================================
# Manifest and offline verification
# ==================================================================================================


def read_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StagingError(f"no staging manifest at {path}: {exc}") from exc
    except ValueError as exc:
        raise StagingError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("benchmarks"), Mapping):
        raise StagingError(f"{path} is not a staging manifest")
    return dict(payload)


def write_manifest(root: Path, results: Mapping[str, Result]) -> dict[str, Any]:
    """Merge this run's benchmarks into the manifest, keeping benchmarks this run did not touch."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if (root / MANIFEST_NAME).is_file():
        try:
            existing = read_manifest(root).get("benchmarks", {})
        except StagingError:
            existing = {}
    benchmarks = {name: entry for name, entry in existing.items() if name not in results}
    for name, result in results.items():
        if result.status != Result.FAILED:
            benchmarks[name] = result.as_dict()
    files: dict[str, dict[str, Any]] = {}
    for entry in benchmarks.values():
        files.update(entry.get("files") or {})
    manifest = {
        "schema_version": SCHEMA_VERSION, "tool": TOOL_NAME, "generated_utc": utc_now(),
        "image_dataset_root": IMAGE_DATASET_ROOT,
        "benchmarks": dict(sorted(benchmarks.items())),
        "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files.values()),
        "failed": {name: result.error for name, result in sorted(results.items())
                   if result.status == Result.FAILED},
    }
    path = root / MANIFEST_NAME
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return manifest


def offline_verify(root: Path) -> list[str]:
    """Re-check a staged directory with no network. Returns one string per problem found."""
    root = Path(root)
    manifest = read_manifest(root)
    problems: list[str] = []
    declared: dict[str, dict[str, Any]] = {}
    for name, entry in sorted((manifest.get("benchmarks") or {}).items()):
        if not isinstance(entry, Mapping):
            problems.append(f"{name}: manifest entry is not an object")
            continue
        for relative, record in sorted((entry.get("files") or {}).items()):
            declared[relative] = {"benchmark": name, **(record if isinstance(record, Mapping) else {})}
    for relative, record in declared.items():
        try:
            pure = confined(relative)
        except StagingError as exc:
            problems.append(f"{relative}: unusable manifest path ({exc})")
            continue
        path = root.joinpath(*pure.parts)
        if not path.is_file():
            problems.append(f"{relative}: MISSING (declared by {record['benchmark']})")
            continue
        data = path.read_bytes()
        if len(data) != record.get("bytes"):
            problems.append(f"{relative}: size {len(data)} differs from the manifest's "
                            f"{record.get('bytes')}")
        digest = sha256_bytes(data)
        if digest != record.get("sha256"):
            problems.append(f"{relative}: sha256 {digest} differs from the manifest's "
                            f"{record.get('sha256')}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME or relative == MANIFEST_NAME + ".tmp":
            continue
        if relative not in declared:
            problems.append(f"{relative}: EXTRA file not declared by the manifest")
    return problems


# ==================================================================================================
# CLI
# ==================================================================================================


def _stagers(args: argparse.Namespace, fetcher: Fetcher, root: Path) -> dict[str, Callable[[], Result]]:
    return {
        "bfcl": lambda: stage_bfcl(fetcher, root, version=args.bfcl_version,
                                   categories=split_list(args.bfcl_categories)
                                   or tuple(BFCL_CATEGORY_FILES),
                                   gorilla_ref=args.bfcl_ref, validate=not args.no_validate),
        "ruler": lambda: stage_ruler(fetcher, root, repository=args.ruler_repository,
                                     ref=args.ruler_ref, prefix=args.ruler_prefix,
                                     validate=not args.no_validate),
        "evalplus": lambda: stage_evalplus(fetcher, root, tag=args.evalplus_tag,
                                           validate=not args.no_validate,
                                           trim_oversize=args.evalplus_trim_oversize),
        "aider-polyglot": lambda: stage_aider_polyglot(
            fetcher, root, ref=args.aider_ref,
            languages=split_list(args.aider_languages) or ("javascript", "python"),
            toolchains=split_list(args.aider_toolchains) or split_list(args.aider_languages)
            or ("javascript", "python"), validate=not args.no_validate),
    }


def run_staging(selected: Sequence[str], stagers: Mapping[str, Callable[[], Result]], root: Path,
                *, log: Callable[[str], None]) -> dict[str, Result]:
    """Stage each selected benchmark. One benchmark's failure never stops the others."""
    results: dict[str, Result] = {}
    for name in selected:
        log(f"[{name}] staging")
        try:
            result = stagers[name]()
        except StagingError as exc:
            result = Result(name, name)
            result.status = Result.FAILED
            result.error = str(exc)
        except Exception as exc:  # an unexpected defect is this benchmark's failure, not the run's
            result = Result(name, name)
            result.status = Result.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
        results[name] = result
        if result.status == Result.FAILED:
            directory = root / result.directory
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)
            log(f"[{name}] FAILED: {result.error}")
        else:
            log(f"[{name}] {result.status}: {len(result.files)} files, {result.total_bytes} bytes")
            if result.blocker:
                log(f"[{name}] BLOCKER: {result.blocker}")
            for note in result.notes:
                log(f"[{name}] note: {note}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stage_benchmark_datasets",
        description="Fetch, normalize and pin the public benchmark datasets the evaluator image bakes.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"staging directory the image build copies (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--benchmark", action="append", default=[],
                        help="benchmark to stage; repeatable or comma-separated (default: all of "
                             + ", ".join(BENCHMARKS) + ")")
    parser.add_argument("--offline-verify", action="store_true",
                        help="re-verify an existing staging directory against its manifest; no network")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help="per-request timeout in seconds")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip loading the staged data through the adapters (not recommended)")
    parser.add_argument("--bfcl-version", default=None,
                        help="exact bfcl-eval version to stage (default: the latest on PyPI)")
    parser.add_argument("--bfcl-ref", default="HEAD",
                        help="gorilla ref whose commit is recorded as source_commit (default: HEAD)")
    parser.add_argument("--bfcl-categories", action="append", default=[],
                        help="BFCL AST categories to bake (default: every category this tool maps)")
    parser.add_argument("--ruler-repository", default=RULER_DEFAULT_REPOSITORY,
                        help="repository holding the Paul Graham essay haystack")
    parser.add_argument("--ruler-ref", default="HEAD", help="ref to pin the essay corpus at")
    parser.add_argument("--ruler-prefix", default=RULER_DEFAULT_PREFIX,
                        help="path prefix of the essay text files inside that repository")
    parser.add_argument("--evalplus-tag", default=EVALPLUS_DEFAULT_TAG,
                        help=f"mbppplus_release tag (default: {EVALPLUS_DEFAULT_TAG})")
    parser.add_argument("--evalplus-trim-oversize", action="store_true",
                        help="keep every base test but drop the plus tests that do not fit the "
                             "adapter's MAX_TEST_MODULE_BYTES, recording the counts in the manifest. "
                             "Off by default: the faithful staging is complete and the adapter's "
                             "constant is the thing that should move.")
    parser.add_argument("--aider-ref", default="HEAD",
                        help="polyglot-benchmark ref whose commit is recorded in pin.json")
    parser.add_argument("--aider-languages", action="append", default=[],
                        help="languages to stage (default: javascript,python - the two with pinned "
                             "test commands)")
    parser.add_argument("--aider-toolchains", action="append", default=[],
                        help="toolchains the worker image provides (default: the staged languages)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output).resolve()

    def log(message: str) -> None:
        print(message, flush=True)

    if args.offline_verify:
        try:
            problems = offline_verify(root)
        except StagingError as exc:
            log(f"offline verification could not run: {exc}")
            return 2
        if problems:
            log(f"offline verification FAILED with {len(problems)} problem(s):")
            for problem in problems:
                log(f"  - {problem}")
            return 1
        manifest = read_manifest(root)
        log(f"offline verification OK: {manifest.get('file_count')} files, "
            f"{manifest.get('total_bytes')} bytes under {root}")
        return 0

    selected = split_list(args.benchmark) or BENCHMARKS
    unknown = [name for name in selected if name not in BENCHMARKS]
    if unknown:
        log(f"unknown benchmark(s): {', '.join(unknown)}; known: {', '.join(BENCHMARKS)}")
        return 2
    root.mkdir(parents=True, exist_ok=True)
    fetcher = UrllibFetcher(timeout_seconds=args.timeout,
                            token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
    results = run_staging(selected, _stagers(args, fetcher, root), root, log=log)
    manifest = write_manifest(root, results)
    log("")
    log(f"staging manifest: {root / MANIFEST_NAME}")
    for name in selected:
        result = results[name]
        log(f"  {name:<16} {result.status:<20} {len(result.files):>5} files "
            f"{result.total_bytes:>12} bytes")
    log(f"  {'TOTAL':<16} {'':<20} {manifest['file_count']:>5} files "
        f"{manifest['total_bytes']:>12} bytes")
    failed = [name for name, result in results.items() if result.status == Result.FAILED]
    blocked = [name for name, result in results.items() if result.status == Result.BLOCKED]
    if blocked:
        log(f"blocked (staged, but an adapter constraint stops it being used): {', '.join(blocked)}")
    if failed:
        log(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
