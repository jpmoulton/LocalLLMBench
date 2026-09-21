"""Offline tests for ``scripts/stage_benchmark_datasets.py``.

Nothing here opens a socket. Every upstream source is served by :class:`FakeFetcher` from bytes built in
this file, which is the same seam the real tool uses, so the normalization, the manifest hashing, the
per-benchmark failure isolation and ``--offline-verify`` are all exercised against real files on disk.

The EvalPlus fixture deliberately carries exactly the 378 items the adapter pins, so the staged JSONL is
fed to the real ``llmbench.benchmarks.evalplus.load_dataset`` with the real ``DatasetPin``: the schema
claim is checked by the adapter itself, not by a restatement of it here.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

from llmbench.benchmarks import aider_polyglot, bfcl, evalplus
from llmbench.benchmarks.ruler_tasks import corpus as ruler_corpus

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "stage_benchmark_datasets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("llmbench_stage_benchmark_datasets", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage = _load_script()

GORILLA_COMMIT = "1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4"
ESSAY_COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3f"
AIDER_COMMIT = "9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c"
BFCL_VERSION = "2026.3.23"
WHEEL_URL = "https://files.pythonhosted.org/packages/ab/cd/bfcl_eval-2026.3.23-py3-none-any.whl"
ESSAY_REPOSITORY = "gkamradt/needle-in-a-haystack"
ESSAY_PREFIX = "needlehaystack/PaulGrahamEssays/"
CANONICAL_URL = ("https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/"
                 "MbppPlus.jsonl.gz")
ORIGIN_URL = ("https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/"
              "MbppPlus-OriginFmt.jsonl.gz")


# ==================================================================================================
# Fake upstream
# ==================================================================================================


class FakeFetcher:
    """Serves a fixed URL -> bytes map. An entry in ``broken`` raises instead, like a dead source."""

    def __init__(self, payloads: dict[str, bytes], broken: dict[str, str] | None = None) -> None:
        self.payloads = dict(payloads)
        self.broken = dict(broken or {})
        self.requested: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.requested.append(url)
        if url in self.broken:
            raise stage.StagingError(self.broken[url])
        if url not in self.payloads:
            raise stage.StagingError(f"fake fetcher has no payload for {url}")
        return self.payloads[url]


def _json(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def _commit_payload(commit: str, when: str = "2026-01-02T03:04:05Z") -> bytes:
    return _json({"sha": commit, "commit": {"author": {"date": when}}})


# ---- BFCL --------------------------------------------------------------------------------------

BFCL_FIXTURE_CATEGORIES = ("simple", "irrelevance")


def _bfcl_function(name: str) -> dict:
    return {"name": name, "description": f"call {name}",
            "parameters": {"type": "dict",
                           "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                           "required": ["city"]}}


def _bfcl_records(prefix: str, count: int) -> bytes:
    lines = []
    for index in range(count):
        lines.append(json.dumps({
            "id": f"{prefix}_{index}",
            "question": [[{"role": "user", "content": f"weather in city {index} please"}]],
            "function": [_bfcl_function("get_weather")],
        }))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bfcl_answers(prefix: str, count: int) -> bytes:
    lines = [json.dumps({"id": f"{prefix}_{index}",
                         "ground_truth": [{"get_weather": {"city": [f"city {index}"], "days": [1, ""]}}]})
             for index in range(count)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bfcl_wheel(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr("bfcl_eval-2026.3.23.dist-info/METADATA", "Name: bfcl-eval\n")
    return buffer.getvalue()


def _bfcl_payloads() -> dict[str, bytes]:
    simple = _bfcl_records("simple", 6)
    simple_answers = _bfcl_answers("simple", 6)
    irrelevance = _bfcl_records("irrelevance", 4)
    data = {
        "BFCL_v4_simple_python.json": simple,
        "possible_answer/BFCL_v4_simple_python.json": simple_answers,
        "BFCL_v4_irrelevance.json": irrelevance,
    }
    wheel = _bfcl_wheel({stage.BFCL_WHEEL_DATA_PREFIX + name: payload for name, payload in data.items()})
    payloads = {
        "https://pypi.org/pypi/bfcl-eval/json": _json({"info": {"version": BFCL_VERSION}}),
        f"https://pypi.org/pypi/bfcl-eval/{BFCL_VERSION}/json": _json({
            "urls": [{"packagetype": "sdist", "url": "https://example.invalid/sdist.tar.gz",
                      "digests": {"sha256": "0" * 64}},
                     {"packagetype": "bdist_wheel", "url": WHEEL_URL,
                      "digests": {"sha256": stage.sha256_bytes(wheel)}}]}),
        WHEEL_URL: wheel,
        f"https://api.github.com/repos/{stage.BFCL_GORILLA_REPOSITORY}/commits/HEAD":
            _commit_payload(GORILLA_COMMIT),
    }
    for name, payload in data.items():
        payloads[stage.raw_github_url(stage.BFCL_GORILLA_REPOSITORY, GORILLA_COMMIT,
                                      stage.BFCL_GORILLA_DATA_PREFIX + name)] = payload
    return payloads


# ---- RULER -------------------------------------------------------------------------------------


def _essay(index: int) -> bytes:
    sentences = [f"Essay {index} sentence {number} says something about startups and taste."
                 for number in range(40)]
    return (" ".join(sentences) + "\n").encode("utf-8")


def _ruler_payloads() -> dict[str, bytes]:
    essays = {f"{ESSAY_PREFIX}essay{index:02d}.txt": _essay(index) for index in range(10)}
    tree = {"truncated": False,
            "tree": [{"path": "README.md", "type": "blob", "sha": "b" * 40, "size": 3},
                     {"path": ESSAY_PREFIX.rstrip("/"), "type": "tree", "sha": "c" * 40}]
            + [{"path": path, "type": "blob", "sha": stage.git_blob_id(payload), "size": len(payload)}
               for path, payload in sorted(essays.items())]}
    payloads = {
        f"https://api.github.com/repos/{ESSAY_REPOSITORY}/commits/HEAD": _commit_payload(ESSAY_COMMIT),
        f"https://api.github.com/repos/{ESSAY_REPOSITORY}/git/trees/{ESSAY_COMMIT}?recursive=1":
            _json(tree),
    }
    for path, payload in essays.items():
        payloads[stage.raw_github_url(ESSAY_REPOSITORY, ESSAY_COMMIT, path)] = payload
    return payloads


# ---- EvalPlus ----------------------------------------------------------------------------------

EVALPLUS_PREAMBLE = """import numpy as np
from math import inf

def is_floats(x) -> bool:
    return isinstance(x, float)


def assertion(out, exp, atol):
    if atol == 0:
        assert out == exp, f'out: {out}, exp: {exp}'
    else:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)

"""
EVALPLUS_REF_PREAMBLE = EVALPLUS_PREAMBLE + """
def ref_func(a, b):
    return a + b

"""
FOUR_BASE_TASK = 4
DIFFERENTIAL_TASKS = (7, 11, 19)
INFINITY_TASK = 23


def _evalplus_fixture() -> tuple[list[dict], list[dict]]:
    canonical: list[dict] = []
    origin: list[dict] = []
    for index in range(evalplus.EVALPLUS_DATASETS["mbpp-plus"].item_count):
        base = [[1, 2], [2, 3], [3, 4]]
        plus: list[list[object]] = [[0, 0], [-1, 1]]
        if index == FOUR_BASE_TASK:
            base = [[1, 2], [2, 3], [3, 4], [5, 6]]
        if index == INFINITY_TASK:
            plus = [[0, 0], ["inf", 1]]
        inputs_source = ", ".join("[" + ", ".join(
            item if item == "inf" else repr(item) for item in row) + "]" for row in base + plus)
        results = [repr(sum(pair)) if "inf" not in pair else "inf" for pair in base + plus]
        if index in DIFFERENTIAL_TASKS:
            harness = (EVALPLUS_REF_PREAMBLE
                       + f"\ninputs = [{inputs_source}]\n"
                       + "for i, inp in enumerate(inputs):\n"
                       + "    assertion(solve(*inp), ref_func(*inp), 0)\n")
        else:
            harness = (EVALPLUS_PREAMBLE
                       + f"\ninputs = [{inputs_source}]\n"
                       + f"results = [{', '.join(results)}]\n"
                       + "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
                       + "    assertion(solve(*inp), exp, 0)\n")
        canonical.append({
            "task_id": f"Mbpp/{index}", "entry_point": "solve",
            "prompt": '"""\nWrite a function to add two numbers.\nassert solve(1, 2) == 3\n"""\n',
            "canonical_solution": "\ndef solve(a, b):\n    return a + b\n",
            "base_input": base, "plus_input": plus, "atol": 0, "contract": "", "assertion": "",
        })
        origin.append({"task_id": index, "code": "\ndef solve(a, b):\n    return a + b\n",
                       "prompt": "Write a function to add two numbers.", "source_file": "fixture",
                       "test_imports": [], "test_list": ["assert solve(1, 2) == 3"], "test": harness})
    return canonical, origin


def _jsonl_gz(rows: list[dict]) -> bytes:
    body = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    return gzip.compress(body)


def _evalplus_payloads() -> dict[str, bytes]:
    canonical, origin = _evalplus_fixture()
    return {
        "https://api.github.com/repos/evalplus/mbppplus_release/releases/tags/v0.2.0": _json({
            "published_at": "2024-04-17T08:53:46Z",
            "html_url": "https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0",
            "assets": [{"name": "MbppPlus.jsonl.gz", "browser_download_url": CANONICAL_URL},
                       {"name": "MbppPlus-OriginFmt.jsonl.gz", "browser_download_url": ORIGIN_URL}]}),
        CANONICAL_URL: _jsonl_gz(canonical),
        ORIGIN_URL: _jsonl_gz(origin),
    }


# ---- Aider Polyglot ----------------------------------------------------------------------------

AIDER_SLUGS = ("acronym", "anagram", "bob")


def _aider_tarball() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(f"polyglot-benchmark-{AIDER_COMMIT}/{name}")
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))

        add("README.md", b"upstream readme\n")
        add("go/exercises/practice/ignored/main.go", b"package main\n")
        for slug in AIDER_SLUGS:
            base = f"python/exercises/practice/{slug}"
            add(f"{base}/.docs/instructions.md", f"# {slug}\n\nSolve {slug}.\n".encode("utf-8"))
            add(f"{base}/.meta/config.json",
                _json({"files": {"solution": [f"{slug}.py"], "test": [f"{slug}_test.py"]},
                       "blurb": slug}))
            add(f"{base}/{slug}.py", f"def {slug}(value):\n    pass\n".encode("utf-8"))
            add(f"{base}/{slug}_test.py",
                f"from {slug} import {slug}\n\n\ndef test_{slug}():\n    assert {slug}('x') is None\n"
                .encode("utf-8"))
    return buffer.getvalue()


def _aider_payloads() -> dict[str, bytes]:
    return {
        f"https://api.github.com/repos/{stage.AIDER_REPOSITORY}/commits/HEAD":
            _commit_payload(AIDER_COMMIT),
        f"https://codeload.github.com/{stage.AIDER_REPOSITORY}/tar.gz/{AIDER_COMMIT}": _aider_tarball(),
    }


def all_payloads() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for part in (_bfcl_payloads(), _ruler_payloads(), _evalplus_payloads(), _aider_payloads()):
        payloads.update(part)
    return payloads


# ==================================================================================================
# Harness
# ==================================================================================================


def stagers(fetcher, root: Path) -> dict:
    return {
        "bfcl": lambda: stage.stage_bfcl(fetcher, root, version=None,
                                         categories=BFCL_FIXTURE_CATEGORIES, gorilla_ref="HEAD",
                                         validate=True),
        "ruler": lambda: stage.stage_ruler(fetcher, root, repository=ESSAY_REPOSITORY, ref="HEAD",
                                           prefix=ESSAY_PREFIX, validate=True),
        "evalplus": lambda: stage.stage_evalplus(fetcher, root, tag="v0.2.0", validate=True),
        "aider-polyglot": lambda: stage.stage_aider_polyglot(fetcher, root, ref="HEAD",
                                                             languages=("python",),
                                                             toolchains=("python",), validate=True),
    }


def run_all(root: Path, *, broken: dict[str, str] | None = None) -> tuple[dict, dict, FakeFetcher]:
    fetcher = FakeFetcher(all_payloads(), broken=broken)
    results = stage.run_staging(stage.BENCHMARKS, stagers(fetcher, root), root, log=lambda _: None)
    manifest = stage.write_manifest(root, results)
    return results, manifest, fetcher


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> tuple[Path, dict, dict]:
    root = tmp_path_factory.mktemp("staged")
    results, manifest, _ = run_all(root)
    return root, results, manifest


# ==================================================================================================
# Every benchmark stages into the layout its adapter reads
# ==================================================================================================


def test_all_four_benchmarks_stage(staged):
    _, results, _ = staged
    assert sorted(results) == sorted(stage.BENCHMARKS)
    assert [results[name].status for name in stage.BENCHMARKS] == [stage.Result.STAGED] * 4
    assert all(result.error is None and result.blocker is None for result in results.values())


def test_bfcl_layout_is_what_load_bundle_reads(staged):
    root, results, _ = staged
    bundle_root = root / "bfcl" / bfcl.REVISION
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_revision"] == bfcl.REVISION
    assert manifest["upstream_package"] == bfcl.UPSTREAM_PACKAGE == "bfcl-eval"
    assert manifest["upstream_version"] == BFCL_VERSION
    assert manifest["source_commit"] == GORILLA_COMMIT
    assert sorted(manifest["categories"]) == ["irrelevance", "simple"]
    for category, entry in manifest["categories"].items():
        records = bundle_root / entry["records"]
        answers = bundle_root / entry["answers"]
        assert entry["records"].startswith("data/") and entry["answers"].startswith(
            "data/possible_answer/")
        assert stage.sha256_bytes(records.read_bytes()) == entry["sha256"]["records"]
        assert stage.sha256_bytes(answers.read_bytes()) == entry["sha256"]["answers"]
        assert entry["count"] == len(records.read_text(encoding="utf-8").strip().splitlines())
        del category
    # The adapter itself is the acceptance test for the layout.
    loaded = bfcl.load_bundle(root)
    assert len(loaded.records) == 10
    assert sorted({record.category for record in loaded.records}) == ["irrelevance", "simple"]
    assert results["bfcl"].verification["adapter_load"]["records"] == 10


def test_bfcl_derives_the_answers_upstream_does_not_ship_for_irrelevance(staged):
    root, results, _ = staged
    derived = (root / "bfcl" / bfcl.REVISION / "data" / "possible_answer"
               / "BFCL_v4_irrelevance.json").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in derived.splitlines() if line.strip()]
    assert rows and all(set(row) == {"id", "ground_truth"} and row["ground_truth"] == [] for row in rows)
    assert any("irrelevance" in note and "ground_truth" in note
               for note in results["bfcl"].normalization)
    # Every irrelevance record must therefore expect no call at all.
    for record in bfcl.load_bundle(root).records:
        if record.category == "irrelevance":
            assert record.variants == ((),)


def test_bfcl_refuses_a_commit_that_did_not_produce_the_staged_bytes(tmp_path):
    payloads = all_payloads()
    url = stage.raw_github_url(stage.BFCL_GORILLA_REPOSITORY, GORILLA_COMMIT,
                               stage.BFCL_GORILLA_DATA_PREFIX + "BFCL_v4_simple_python.json")
    payloads[url] = payloads[url] + b"\n"
    fetcher = FakeFetcher(payloads)
    with pytest.raises(stage.StagingError, match="byte-identical"):
        stage.stage_bfcl(fetcher, tmp_path, version=None, categories=BFCL_FIXTURE_CATEGORIES,
                         gorilla_ref="HEAD", validate=True)


def test_bfcl_refuses_a_wheel_whose_hash_pypi_does_not_declare(tmp_path):
    payloads = all_payloads()
    payloads[WHEEL_URL] = payloads[WHEEL_URL] + b"\x00"
    with pytest.raises(stage.StagingError, match="PyPI declares"):
        stage.stage_bfcl(FakeFetcher(payloads), tmp_path, version=None,
                         categories=BFCL_FIXTURE_CATEGORIES, gorilla_ref="HEAD", validate=True)


def test_ruler_layout_is_the_corpus_fallback_the_adapter_documents(staged):
    root, results, _ = staged
    essays = sorted((root / "ruler" / "essays").glob("*.txt"))
    assert len(essays) == 10
    status = ruler_corpus.essay_status(str(root))
    assert status.present and status.path == str(root / "ruler" / "essays")
    assert not [gap.detail for gap in
                __import__("llmbench.benchmarks.ruler_tasks", fromlist=["corpus_gaps"]).corpus_gaps(
                    str(root), ["niah_multiquery", "vt", "cwe", "niah_multikey_3"])]
    assert results["ruler"].verification["corpus_words"] > 1000
    assert results["ruler"].verification["git_blob_ids_match_pinned_tree"] is True


def test_ruler_refuses_an_essay_that_does_not_match_the_pinned_blob_id(tmp_path):
    payloads = all_payloads()
    url = stage.raw_github_url(ESSAY_REPOSITORY, ESSAY_COMMIT, f"{ESSAY_PREFIX}essay03.txt")
    payloads[url] = payloads[url] + b"tampered"
    with pytest.raises(stage.StagingError, match="git blob id"):
        stage.stage_ruler(FakeFetcher(payloads), tmp_path, repository=ESSAY_REPOSITORY, ref="HEAD",
                          prefix=ESSAY_PREFIX, validate=True)


def test_aider_layout_is_what_build_plan_reads(staged):
    root, results, _ = staged
    corpus = root / aider_polyglot.CORPUS_DIRNAME
    pin = json.loads((corpus / aider_polyglot.PIN_NAME).read_text(encoding="utf-8"))
    assert set(pin) == {"schema_version", "repository", "commit", "retrieved_utc", "languages",
                        "toolchains"}
    read = aider_polyglot.read_pin(corpus)
    assert read.commit == AIDER_COMMIT and read.languages == ("python",)
    found = aider_polyglot.enumerate_exercises(corpus, "python")
    assert [task_id for task_id, _ in found] == [f"aider-polyglot/python/{slug}"
                                                for slug in AIDER_SLUGS]
    for task_id, directory in found:
        exercise = aider_polyglot.load_exercise(task_id, directory)
        assert exercise.instructions.startswith("# ")
        assert list(exercise.editable()) == [f"{exercise.slug}.py"]
        assert list(exercise.test_files()) == [f"{exercise.slug}_test.py"]
    assert not (corpus / "go").exists()  # only the selected languages are staged
    assert results["aider-polyglot"].verification["exercises_loadable"] == len(AIDER_SLUGS)
    assert results["aider-polyglot"].verification["exercises_rejected"] == 0
    # The fixture is a three-exercise subset, so the count note must say so rather than stay silent.
    assert any("staged 3, documented 34" in note for note in results["aider-polyglot"].notes)


def test_aider_records_exercism_licensing_rather_than_a_clean_spdx_id(staged):
    _, results, _ = staged
    license_info = results["aider-polyglot"].license
    assert license_info["spdx_id"] is None
    assert "Exercism" in license_info["summary"]
    assert results["bfcl"].license["spdx_id"] == "Apache-2.0"
    assert results["evalplus"].license["spdx_id"] == "Apache-2.0"
    assert results["ruler"].license["spdx_id"] is None
    assert "Paul Graham" in results["ruler"].license["summary"]


# ==================================================================================================
# EvalPlus: the adapter's own loader is the schema check
# ==================================================================================================


def test_staged_evalplus_records_are_accepted_by_load_dataset(staged):
    root, results, _ = staged
    pin = evalplus.EVALPLUS_DATASETS["mbpp-plus"]
    path = root / "evalplus" / pin.filename
    dataset = evalplus.load_dataset(path, pin)
    assert len(dataset.items) == pin.item_count == 378
    assert dataset.sha256 == stage.sha256_bytes(path.read_bytes())
    assert (path.with_name(path.name + ".sha256")).read_text(encoding="ascii").strip() == dataset.sha256
    item = dataset.by_task_id()["evalplus/mbpp-plus/Mbpp/2"]
    assert item.entry_point == "solve" and item.signature == "def solve(a, b)"
    assert [test.test_id for test in item.tests] == ["base/0", "base/1", "base/2", "plus/0", "plus/1"]
    assert item.tests[0].code == "assertion(solve(*[1, 2]), 3, 0)"
    assert "def assertion(out, exp, atol):" in item.test_setup
    # build_test_module is what the worker runs; load_dataset already refused anything that cannot parse.
    module_text = evalplus.build_test_module(item)
    assert module_text.startswith('"""') and "from solution import solve" in module_text
    assert results["evalplus"].verification["adapter_load"] == {
        "ok": True, "items": 378, "tests": sum(len(entry.tests) for entry in dataset.items),
        "sha256": dataset.sha256}


def test_evalplus_base_plus_boundary_follows_the_published_base_input(staged):
    root, _, _ = staged
    pin = evalplus.EVALPLUS_DATASETS["mbpp-plus"]
    dataset = evalplus.load_dataset(root / "evalplus" / pin.filename, pin)
    four = dataset.by_task_id()[f"evalplus/mbpp-plus/Mbpp/{FOUR_BASE_TASK}"]
    assert [test.kind for test in four.tests] == ["base"] * 4 + ["plus"] * 2
    assert [test.test_id for test in four.tests][:4] == ["base/0", "base/1", "base/2", "base/3"]


def test_evalplus_carries_the_differential_and_infinity_shapes(staged):
    root, _, _ = staged
    pin = evalplus.EVALPLUS_DATASETS["mbpp-plus"]
    catalog = evalplus.load_dataset(root / "evalplus" / pin.filename, pin).by_task_id()
    differential = catalog[f"evalplus/mbpp-plus/Mbpp/{DIFFERENTIAL_TASKS[0]}"]
    assert "def ref_func(a, b):" in differential.test_setup
    assert differential.tests[0].code == "assertion(solve(*[1, 2]), ref_func(*[1, 2]), 0)"
    infinite = catalog[f"evalplus/mbpp-plus/Mbpp/{INFINITY_TASK}"]
    assert infinite.tests[-1].code == "assertion(solve(*[inf, 1]), inf, 0)"
    assert "from math import inf" in infinite.test_setup  # the name the emitted test relies on


def test_evalplus_reports_an_item_that_overflows_the_adapters_module_cap(tmp_path):
    payloads = all_payloads()
    canonical, origin = _evalplus_fixture()
    # Size the synthetic item off the adapter's own cap so raising the constant cannot make this
    # test silently stop exercising the overflow path (it did once, when the cap went to 1 MiB).
    from llmbench.benchmarks.evalplus import MAX_TEST_MODULE_BYTES
    filler = ", ".join(str(number) for number in range(MAX_TEST_MODULE_BYTES // 4))
    origin[5]["test"] = (EVALPLUS_PREAMBLE
                         + f"\ninputs = [[1, 2], [{filler}]]\nresults = [3, 1]\n"
                         + "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
                         + "    assertion(solve(*inp), exp, 0)\n")
    canonical[5]["base_input"] = [[1, 2]]
    canonical[5]["plus_input"] = [list(range(MAX_TEST_MODULE_BYTES // 4))]
    payloads[CANONICAL_URL] = _jsonl_gz(canonical)
    payloads[ORIGIN_URL] = _jsonl_gz(origin)
    result = stage.stage_evalplus(FakeFetcher(payloads), tmp_path, tag="v0.2.0", validate=True)
    assert result.status == stage.Result.BLOCKED
    assert "Mbpp/5" in result.blocker and "MAX_TEST_MODULE_BYTES" in result.blocker
    assert result.verification["adapter_load"]["ok"] is False
    # Opting in trims only the plus tests of the offending item and records exactly what it changed.
    trimmed = stage.stage_evalplus(FakeFetcher(payloads), tmp_path, tag="v0.2.0", validate=True,
                                   trim_oversize=True)
    assert trimmed.status == stage.Result.STAGED
    assert trimmed.verification["trimmed_items"]["Mbpp/5"]["tests_after"] == 1
    assert trimmed.verification["adapter_load"]["ok"] is True
    assert any("trim-oversize" in note for note in trimmed.normalization)


# ==================================================================================================
# Manifest, provenance and offline verification
# ==================================================================================================


def test_manifest_hashes_are_computed_over_the_bytes_on_disk(staged):
    root, results, manifest = staged
    assert manifest["schema_version"] == stage.SCHEMA_VERSION
    assert manifest["image_dataset_root"] == "/opt/llmbench/benchmarks"
    assert manifest["failed"] == {}
    declared = 0
    for name, entry in manifest["benchmarks"].items():
        assert entry["file_count"] == len(entry["files"]) == len(results[name].files)
        for relative, record in entry["files"].items():
            path = root / relative
            payload = path.read_bytes()
            assert record["bytes"] == len(payload)
            assert record["sha256"] == stage.sha256_bytes(payload)
            declared += 1
    assert manifest["file_count"] == declared
    assert manifest["total_bytes"] == sum(
        (root / relative).stat().st_size
        for entry in manifest["benchmarks"].values() for relative in entry["files"])


def test_provenance_never_carries_a_placeholder_commit(staged):
    _, _, manifest = staged
    found = 0
    for entry in manifest["benchmarks"].values():
        for key, value in entry["provenance"].items():
            if not key.endswith("commit") or not isinstance(value, str):
                continue
            assert re.fullmatch(r"[0-9a-f]{40}", value), f"{key}={value!r} is not a real commit id"
            assert len(set(value)) > 1, f"{key}={value!r} is a placeholder"
            found += 1
    assert found >= 3  # bfcl, ruler and aider-polyglot each pin one
    commits = [entry["provenance"].get("source_commit") for entry in manifest["benchmarks"].values()]
    assert "0" * 40 not in commits and None not in commits[:1]


@pytest.mark.parametrize("value", ["", "0" * 40, "f" * 40, "abc", "A" * 40, None, 42,
                                   "1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e"])
def test_require_commit_rejects_anything_that_is_not_a_real_commit(value):
    with pytest.raises(stage.StagingError):
        stage.require_commit(value, source="fixture")


def test_require_commit_accepts_a_real_looking_commit():
    assert stage.require_commit(GORILLA_COMMIT, source="fixture") == GORILLA_COMMIT


def test_a_dead_source_fails_only_its_own_benchmark(tmp_path):
    broken = {"https://api.github.com/repos/evalplus/mbppplus_release/releases/tags/v0.2.0":
              "GET .../releases/tags/v0.2.0 returned HTTP 451"}
    results, manifest, _ = run_all(tmp_path, broken=broken)
    assert results["evalplus"].status == stage.Result.FAILED
    assert "HTTP 451" in results["evalplus"].error
    assert not (tmp_path / "evalplus").exists()  # a failed benchmark leaves nothing to bake
    assert [results[name].status for name in ("bfcl", "ruler", "aider-polyglot")] == \
        [stage.Result.STAGED] * 3
    assert "evalplus" not in manifest["benchmarks"]
    assert "HTTP 451" in manifest["failed"]["evalplus"]
    assert stage.offline_verify(tmp_path) == []


def test_an_unexpected_defect_in_one_stager_does_not_abort_the_others(tmp_path):
    fetcher = FakeFetcher(all_payloads())
    plan = stagers(fetcher, tmp_path)

    def explode():
        raise ZeroDivisionError("a defect, not a StagingError")

    plan["ruler"] = explode
    results = stage.run_staging(stage.BENCHMARKS, plan, tmp_path, log=lambda _: None)
    assert results["ruler"].status == stage.Result.FAILED
    assert results["ruler"].error == "ZeroDivisionError: a defect, not a StagingError"
    assert [results[name].status for name in ("bfcl", "evalplus", "aider-polyglot")] == \
        [stage.Result.STAGED] * 3


def test_offline_verify_is_clean_for_an_untouched_directory(staged):
    root, _, _ = staged
    assert stage.offline_verify(root) == []
    assert stage.main(["--output", str(root), "--offline-verify"]) == 0


def test_offline_verify_catches_a_mutated_byte(tmp_path):
    run_all(tmp_path)
    target = tmp_path / "ruler" / "essays" / "essay04.txt"
    payload = bytearray(target.read_bytes())
    payload[5] ^= 0x01
    target.write_bytes(bytes(payload))
    problems = stage.offline_verify(tmp_path)
    assert len(problems) == 1 and "ruler/essays/essay04.txt" in problems[0]
    assert "sha256" in problems[0]
    assert stage.main(["--output", str(tmp_path), "--offline-verify"]) == 1


def test_offline_verify_catches_a_missing_file(tmp_path):
    run_all(tmp_path)
    (tmp_path / "ruler" / "essays" / "essay02.txt").unlink()
    problems = stage.offline_verify(tmp_path)
    assert problems == ["ruler/essays/essay02.txt: MISSING (declared by ruler)"]


def test_offline_verify_catches_a_truncated_file_by_size_and_hash(tmp_path):
    run_all(tmp_path)
    target = tmp_path / "evalplus" / evalplus.EVALPLUS_DATASETS["mbpp-plus"].filename
    target.write_bytes(target.read_bytes()[:-40])
    problems = stage.offline_verify(tmp_path)
    assert len(problems) == 2
    assert any("size" in problem for problem in problems)
    assert any("sha256" in problem for problem in problems)


def test_offline_verify_catches_an_extra_file(tmp_path):
    run_all(tmp_path)
    (tmp_path / "ruler" / "essays" / "smuggled.txt").write_bytes(b"not staged by the tool\n")
    problems = stage.offline_verify(tmp_path)
    assert problems == ["ruler/essays/smuggled.txt: EXTRA file not declared by the manifest"]


def test_offline_verify_needs_a_manifest(tmp_path):
    assert stage.main(["--output", str(tmp_path / "nothing-here"), "--offline-verify"]) == 2
    with pytest.raises(stage.StagingError, match="no staging manifest"):
        stage.offline_verify(tmp_path / "nothing-here")


def test_re_staging_one_benchmark_keeps_the_others_in_the_manifest(tmp_path):
    run_all(tmp_path)
    fetcher = FakeFetcher(all_payloads())
    again = stage.run_staging(("ruler",), stagers(fetcher, tmp_path), tmp_path, log=lambda _: None)
    merged = stage.write_manifest(tmp_path, again)
    assert sorted(merged["benchmarks"]) == sorted(stage.BENCHMARKS)
    assert stage.offline_verify(tmp_path) == []


def test_staging_is_byte_stable_across_runs_apart_from_the_recorded_timestamps(tmp_path):
    first_root, second_root = tmp_path / "one", tmp_path / "two"
    _, first, _ = run_all(first_root)
    _, second, _ = run_all(second_root)
    timestamped = {f"aider-polyglot/{aider_polyglot.PIN_NAME}",
                   f"bfcl/{bfcl.REVISION}/manifest.json"}
    for name in stage.BENCHMARKS:
        left = first["benchmarks"][name]["files"]
        right = second["benchmarks"][name]["files"]
        assert sorted(left) == sorted(right)
        for relative in left:
            if relative in timestamped:
                continue
            assert left[relative] == right[relative], relative


# ==================================================================================================
# CLI surface
# ==================================================================================================


def test_unknown_benchmark_is_refused_before_any_fetching(capsys):
    assert stage.main(["--benchmark", "nope", "--output", "unused"]) == 2
    assert "unknown benchmark" in capsys.readouterr().out


def test_split_list_accepts_both_spellings():
    assert stage.split_list(["a,b", "b", "c"]) == ("a", "b", "c")
    assert stage.split_list([]) == ()


@pytest.mark.parametrize("relative", ["/etc/passwd", "../escape", "a/../b", "", "  ", "a/./b",
                                       "c:/windows", "back\\slash"])
def test_confined_refuses_unusable_staged_paths(relative):
    with pytest.raises(stage.StagingError):
        stage.confined(relative)


def test_urllib_fetcher_refuses_non_https_and_credentialed_urls():
    fetcher = stage.UrllibFetcher()
    for url in ("http://example.invalid/x", "ftp://example.invalid/x",
                "https://user:pass@example.invalid/x", "file:///etc/passwd"):
        with pytest.raises(stage.StagingError, match="credential-free https"):
            fetcher.fetch(url)
