"""Run every pinned Aider Polyglot exercise's REFERENCE solution through the real worker path. No model.

A harness that fails correct solutions manufactures low scores, and nothing in a model run can reveal it. The
corpus ships a reference for every exercise (``.meta/example.py`` / ``.meta/proof.ci.js``; never staged into a
worker and never shown to a model), so the harness can be calibrated against ground truth: every reference must
pass, and every untouched stub must fail. Root-owned: it starts worker containers.

Usage: python scripts/polyglot_reference_calibration.py --iidfile artifacts/container-prep/worker-image-3/worker-image.id
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmbench.benchmarks.aider_polyglot import enumerate_exercises  # noqa: E402
from llmbench.coding.benchmark_items import polyglot_exercise_fixtures, run_polyglot_exercise  # noqa: E402
from llmbench.coding.image_plan import verified_local_image_id  # noqa: E402
from llmbench.coding.sandbox import DockerWorker, SandboxLimits  # noqa: E402
from llmbench.config import RunMode  # noqa: E402
from llmbench.containers.executor import BoundedProcessExecutor  # noqa: E402
from llmbench.safety import SessionLock  # noqa: E402

REFERENCES = {"python": "example.py", "javascript": "proof.ci.js"}
KNOWN_PASSING_STUBS = frozenset({"aider-polyglot/javascript/ledger"})
"""Exercism's ledger is a REFACTORING exercise: the code it hands out already passes its tests, upstream, by
design. It is therefore a free point for a model that changes nothing - a property of the benchmark, reported
here so it is known, not a harness defect to be repaired."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iidfile", required=True)
    parser.add_argument("--dataset-root", default="artifacts/benchmark-datasets")
    parser.add_argument("--output", default="artifacts/container-pilot/polyglot-reference-calibration.json")
    parser.add_argument("--policy", default="runtime-policy.json")
    args = parser.parse_args()
    image = verified_local_image_id(args.iidfile)
    dataset_root = Path(args.dataset_root).resolve()
    corpus = dataset_root / "aider-polyglot"
    task_ids = tuple(task_id for language in REFERENCES for task_id, _ in enumerate_exercises(corpus, language))
    selection = SimpleNamespace(benchmark_id="aider-polyglot", task_ids=task_ids)
    fixtures = polyglot_exercise_fixtures(SimpleNamespace(benchmarks=(selection,), dataset_root=str(dataset_root)))
    lock = SessionLock.read(args.policy)
    limits = SandboxLimits(timeout_seconds=120, memory_mib=2048, pids=256, open_files=1024)
    rows = []
    Path("artifacts/test-runs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="artifacts/test-runs") as tmp:
        for task_id, fixture in fixtures.items():
            _, language, slug = task_id.split("/")
            meta = corpus / language / "exercises" / "practice" / slug / ".meta"
            reference = (meta / REFERENCES[language]).read_text(encoding="utf-8")
            editable = [item.path for item in fixture.initial_files]
            row = {"task_id": task_id, "editable_files": editable}
            for label, patch in (("reference", {editable[0]: reference}),
                                 ("stub", {editable[0]: fixture.initial_files[0].content})):
                if len(editable) != 1:
                    row[label] = {"skipped": "more than one editable file; the single reference cannot be placed"}
                    continue
                worker = DockerWorker(allowed_root=Path(tmp).resolve(), session_lock=lock, mode=RunMode.LIVE,
                                      executor=BoundedProcessExecutor(session_lock=lock, mode=RunMode.LIVE))
                sample = run_polyglot_exercise(fixture, patch, worker=worker, image=image, limits=limits,
                                               timeout_seconds=300)["sample"]
                row[label] = {"passed": sample["passed"], "tests_passed": sample["tests_passed"],
                              "exit_code": sample["exit_code"], "verdict": sample["verdict"],
                              "seconds": round(sample["elapsed_seconds"], 1),
                              "tail": sample["test_output"].strip()[-400:]}
            rows.append(row)
            print(f"{task_id:52s} reference={row['reference'].get('passed')!s:5} "
                  f"tests={row['reference'].get('tests_passed')!s:4} stub={row['stub'].get('passed')!s:5}", flush=True)
    bad_reference = [r["task_id"] for r in rows if r["reference"].get("passed") is not True]
    bad_stub = [r["task_id"] for r in rows
                if r["stub"].get("passed") is not False and r["task_id"] not in KNOWN_PASSING_STUBS]
    summary = {"image": image, "exercises": len(rows), "references_failing": bad_reference,
               "stubs_passing": bad_stub, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} exercises; references failing: {len(bad_reference)}; stubs passing: {len(bad_stub)}")
    for task_id in bad_reference:
        print("  REFERENCE FAILS:", task_id)
    for task_id in bad_stub:
        print("  STUB PASSES:", task_id)
    return 0 if not bad_reference and not bad_stub else 1


if __name__ == "__main__":
    raise SystemExit(main())

