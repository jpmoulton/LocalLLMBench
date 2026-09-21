"""Root-owned live coding-broker acceptance (plan B-L1); no model, no host candidate execution.

Scripted "model" responses drive the real generation step (``run_coding_benchmarks``) through a
``DirectClient`` into a real ``HostBroker`` that runs the digest-pinned worker image via Docker.
Variants per fixture: reference (must pass), initial (must fail ``incorrect``), hang (bounded candidate
timeout), malformed (no patch: never submitted), oversize (patch above ``max_patch_bytes``: broker rejects),
traversal (file outside the editable list: never submitted), plus two raw spool files the broker itself
must reject: an oversize request file and a traversal patch key. Every result is durably recorded as it
happens; acceptance needs every variant as expected, complete ledgers, verified worker cleanup, verified
absence of every owned container name and no infrastructure error. Exit code 0 only on acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from llmbench.coding.acceptance import acceptance_verdict
from llmbench.coding.broker import HostBroker
from llmbench.coding.fixtures import fixtures
from llmbench.coding.generation import run_coding_benchmarks
from llmbench.coding.image_plan import verified_local_image_id
from llmbench.coding.sandbox import BoundedProcessExecutor, SandboxLimits
from llmbench.coding.spool import CodingJobRequest, publish_atomic, request_bytes, request_name
from llmbench.config import GenerationSettings, RunMode
from llmbench.containers.artifacts import RunArtifacts
from llmbench.containers.config import BenchmarkSelection, ImageRef
from llmbench.evaluations.tools import strict_json_loads
from llmbench.safety import SessionLock
from llmbench.store import utc_now

from llmbench.containers.config import BrokerSettings  # noqa: E402

HANGS = {"python": "def chunked(values, size):\n    while True:\n        pass\n",
         "typescript": "export type RecordItem = { key: string; value: number };\nexport function groupRecords("
                       "records: RecordItem[]): {key:string; values:number[]}[] { for (;;) {} }\n",
         "javascript": "exports.stableUnique = values => { for (;;) {} };\n"}
VARIANTS = ("reference", "initial", "hang", "malformed", "oversize", "traversal", "spool-oversize", "spool-traversal")
LEDGER_TERMINAL = {"finished", "published"}
REMAINING_SECONDS = 600.0  # the scripted evaluator's remaining budget handed to the generation step
# The generation step checks the inference permission before every "model" call. The scripted callback contacts no
# endpoint, so it gets an inference-only lock; the broker keeps the real policy lock for container execution.
SCRIPTED_LOCK = SessionLock(allow_inference=True, reason="scripted model responses; no inference endpoint exists")


def durable_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    temporary = path.with_name(path.name + ".pending") if replace else path
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        os.replace(temporary, path)


def scripted(content: str):
    async def callback(request):
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "scripted": True}
    return callback


class Artifacts:
    """Raw-response sink for the generation step: durable files under ``<variant>/responses``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, relative: str, content: bytes) -> Path:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        durable_write(target, content)
        return target


def response_for(fixture, variant: str, settings: BrokerSettings) -> str:
    editable = fixture.initial_files[0].path
    if variant == "reference":
        return json.dumps({item.path: item.content for item in fixture.reference_files})
    if variant == "initial":
        return json.dumps({item.path: item.content for item in fixture.initial_files})
    if variant == "hang":
        return "```\n" + HANGS[fixture.language] + "```"
    if variant == "malformed":
        return "I would fix this by iterating, but here is no code: {not json"
    if variant == "oversize":
        return json.dumps({editable: "# " + "x" * (settings.max_patch_bytes + 1) + "\n"})
    if variant == "traversal":
        return json.dumps({editable: fixture.reference_files[0].content, "../escape.py": "print('escape')\n"})
    raise ValueError(variant)


def ledger_rows(broker: HostBroker) -> list[dict]:
    rows = []
    for path in sorted(broker.ledger_dir.glob("ledger*.jsonl")):
        rows.extend(strict_json_loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def ledger_complete(rows: list[dict]) -> tuple[bool, list[str]]:
    """Every validated request reached finished+published; every rejected request was published; no gaps."""
    problems = []
    by_request: dict[str, set[str]] = {}
    for row in rows:
        if type(row.get("request_id")) is str:
            by_request.setdefault(row["request_id"], set()).add(row["event"])
    for request_id, events in by_request.items():
        if "validated" in events and not LEDGER_TERMINAL <= events:
            problems.append(f"{request_id}: validated without finished+published")
        if "rejected" in events and "published" not in events and "publish_skipped_existing" not in events:
            problems.append(f"{request_id}: rejected without a published result")
        if "started" in events and "finished" not in events:
            problems.append(f"{request_id}: started without finished")
    if not rows or rows[0].get("event") != "broker" or rows[-1].get("event") != "cancel_all":
        problems.append("ledger must open with the broker header and close with cancel_all")
    return not problems, problems


def worker_names(rows: list[dict]) -> set[str]:
    return {row["name"] for row in rows if row.get("event") == "worker" and type(row.get("name")) is str}


def config_for(fixture, image: str, settings: BrokerSettings):
    worker_image = ImageRef(role="worker", reference=image, image_id=image, entrypoint=("python3",))
    return SimpleNamespace(session_id="broker-acceptance", broker=settings, worker_image=worker_image,
                           generation=GenerationSettings(max_output_tokens=1024, repair_attempts=0),
                           benchmarks=(BenchmarkSelection(benchmark_id="coding", revision=fixture.revision,
                                                          task_ids=(fixture.fixture_id,)),))


def raw_spool_request(broker: HostBroker, fixture, variant: str, settings: BrokerSettings) -> str:
    """Write a request file the way a hostile evaluator would; returns the request id."""
    request_id = uuid.uuid4().hex
    if variant == "spool-oversize":
        payload = b"{" + b" " * (settings.max_request_bytes + 1) + b"}"
    else:
        request = CodingJobRequest(session_id=broker.session_id, attempt_id=broker.attempt_id, request_id=request_id,
                                   fixture_id=fixture.fixture_id, fixture_revision=fixture.revision,
                                   fixture_hash=fixture.identity(), attempt_index=1,
                                   patch={fixture.initial_files[0].path: "x\n"}, submitted_utc=utc_now())
        data = json.loads(request_bytes(request))
        data["patch"] = {"../escape.py": "print('escape')\n"}
        payload = json.dumps(data).encode("utf-8")
    publish_atomic(broker.requests_dir, request_name(request_id), payload)
    return request_id


def spool_verdict(broker: HostBroker, request_id: str, variant: str) -> dict:
    """Two ticks: the broker must publish a rejection for the raw file without launching any worker."""
    before = len(worker_names(ledger_rows(broker)))
    summaries = [broker.tick(600), broker.tick(600)]
    path = broker.results_dir / request_name(request_id)
    result = strict_json_loads(path.read_text(encoding="utf-8")) if path.exists() else None
    expected = "oversize" if variant == "spool-oversize" else "schema: patch"
    reasons = []
    if result is None:
        reasons.append("no rejection result was published for the raw request")
    else:
        if result.get("status") != "rejected" or not str(result.get("failure_reason", "")).startswith(expected):
            reasons.append(f"expected rejected/{expected}, got {result.get('status')}/{result.get('failure_reason')}")
        if result.get("cleanup_confirmed") is not True or result.get("abort_campaign") is not False:
            reasons.append("a rejection must confirm cleanup and never abort")
    if sum(item["processed"] for item in summaries) != 0 or len(worker_names(ledger_rows(broker))) != before:
        reasons.append("a raw rejection must not launch a worker")
    return {"as_expected": not reasons, "reasons": reasons, "result": result, "ticks": summaries}


def generation_verdict(fixture, variant: str, sample: dict, broker: HostBroker, settings: BrokerSettings) -> dict:
    reasons = []
    broker_info = sample.get("broker")
    if variant in {"malformed", "traversal"}:
        expected_error = "no_patch_found" if variant == "malformed" else "files_outside_editable_list"
        if sample.get("outcome_status") != "invalid_patch" or sample.get("error") != expected_error:
            reasons.append(f"expected invalid_patch/{expected_error}, got "
                           f"{sample.get('outcome_status')}/{sample.get('error')}")
        if broker_info is not None:
            reasons.append("an invalid patch must never reach the broker")
    elif variant == "oversize":
        if sample.get("outcome_status") != "broker_rejected" or sample.get("error") != "patch_too_large":
            reasons.append(f"expected broker_rejected/patch_too_large, got "
                           f"{sample.get('outcome_status')}/{sample.get('error')}")
    else:
        if broker_info is None or broker_info.get("status") != "completed":
            reasons.append(f"expected a completed broker result, got {broker_info}")
        else:
            trace = broker.run_dir / "broker" / "traces" / f"{broker_info['request_id']}.json"
            result = {"sample": sample, "raw_bytes": trace.read_bytes(), "abort_campaign": sample.get("abort_campaign")}
            verdict = acceptance_verdict(fixture, variant, result)
            reasons.extend(verdict["reasons"])
            if broker_info.get("cleanup_confirmed") is not True:
                reasons.append("broker result did not confirm cleanup")
    if sample.get("score") not in (0.0, 1.0) or sample.get("model_evaluated") is not True:
        reasons.append("sample must carry a 0/1 score and model_evaluated")
    if (sample.get("first_attempt_success") is True) != (variant == "reference"):
        reasons.append("first_attempt_success must be true exactly for the reference patch")
    if variant == "hang" and settings.worker_limits.timeout_seconds > 60:
        reasons.append("hang variant needs a short worker timeout to be a bounded-timeout proof")
    return {"as_expected": not reasons, "reasons": reasons}


def run_variant(out: Path, fixture, variant: str, image: str, lock: SessionLock, hang_timeout: int, *,
                executor=None) -> dict:
    """One broker per variant under ``<out>/runs``; ``executor`` is injected only by the offline smoke test."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", fixture.fixture_id)
    run_dir = out / "runs" / f"{slug}-{variant}"
    limits = SandboxLimits(timeout_seconds=hang_timeout) if variant == "hang" else SandboxLimits()
    settings = BrokerSettings(worker_limits=limits,
                              fixture_timeout_seconds=min(900, max(180, 6 * limits.timeout_seconds + 60)))
    artifacts = RunArtifacts(run_dir)
    started = time.monotonic()
    broker = HostBroker(run_dir, config_for(fixture, image, settings), lock, artifacts.scoped("broker"),
                        executor=executor)
    broker.reconcile()
    row = {"fixture": fixture.fixture_id, "variant": variant, "run_dir": str(run_dir), "attempt_id": broker.attempt_id}
    try:
        if variant.startswith("spool-"):
            request_id = raw_spool_request(broker, fixture, variant, settings)
            row.update(request_id=request_id, **spool_verdict(broker, request_id, variant))
        else:
            samples = run_coding_benchmarks(config_for(fixture, image, settings),
                                            scripted(response_for(fixture, variant, settings)),
                                            lambda: REMAINING_SECONDS, Artifacts(run_dir / "responses"),
                                            SCRIPTED_LOCK, client=broker.direct_client())
            sample = samples[0]
            durable_write(run_dir / "sample.json", json.dumps(sample, indent=2, default=str).encode("utf-8"))
            row.update(sample=sample, **generation_verdict(fixture, variant, sample, broker, settings))
    finally:
        cancel = broker.cancel_all()
        rows = ledger_rows(broker)
        complete, problems = ledger_complete(rows)
        row.update(cancel_all=cancel, ledger_complete=complete, ledger_problems=problems,
                   worker_names=sorted(worker_names(rows)), abort_campaign=broker.abort_campaign,
                   elapsed_seconds=round(time.monotonic() - started, 2))
        if not complete or cancel.get("cleanup_verified") is not True or broker.abort_campaign:
            row["as_expected"] = False
            row.setdefault("reasons", []).append("ledger incomplete, cleanup unverified or campaign abort")
    return row


def absence_checks(names: set[str], lock: SessionLock) -> list[dict]:
    executor = BoundedProcessExecutor(session_lock=lock, mode=RunMode.LIVE)
    checks = []
    for name in sorted(names):
        try:
            check = executor.run(("docker", "ps", "--all", "--filter", f"name=^/{name}$", "--format", "{{.ID}}"),
                                 timeout_seconds=10, max_output_bytes=4096)
            queried = check.status == "completed" and check.returncode == 0
            checks.append({"name": name, "status": check.status, "returncode": check.returncode,
                           "verified_absent": queried and not check.stdout.strip(),
                           "present": bool(check.stdout.strip()) if queried else None,
                           "stderr": check.stderr.decode("utf-8", errors="replace")})
        except BaseException as exc:
            checks.append({"name": name, "verified_absent": False, "error": type(exc).__name__ + ": " + str(exc)})
    return checks


def summary_of(rows: list[dict], *, expected_rows: int, checks: list[dict], image: str, error: str | None) -> dict:
    complete = len(rows) == expected_rows
    expected = complete and all(row.get("as_expected") is True for row in rows)
    ledgers = bool(rows) and all(row.get("ledger_complete") is True for row in rows)
    cleanup = bool(rows) and all(row.get("cancel_all", {}).get("cleanup_verified") is True for row in rows)
    absent = bool(checks) and all(check.get("verified_absent") is True for check in checks)
    abort = any(row.get("abort_campaign") is not False for row in rows)
    return {"image": image, "accepted": expected and ledgers and cleanup and absent and not abort and error is None,
            "complete": complete, "expected_rows": expected_rows, "all_as_expected": expected,
            "all_ledgers_complete": ledgers, "all_cleanup_verified": cleanup, "absence_query_verified": absent,
            "any_abort": abort, "error": error, "absence_checks": checks,
            "leftover_worker_containers": [check["name"] for check in checks if check.get("present") is True],
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iidfile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hang-timeout", type=int, default=12)
    parser.add_argument("--policy", default="runtime-policy.json")
    args = parser.parse_args()
    image = verified_local_image_id(args.iidfile)
    lock = SessionLock.read(args.policy)
    lock.check("container", RunMode.LIVE)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected = fixtures()
    expected_rows = len(VARIANTS) * len(selected)
    rows, names, checks = [], set(), []
    error = None

    def checkpoint() -> dict:
        summary = summary_of(rows, expected_rows=expected_rows, checks=checks, image=image, error=error)
        durable_write(out / "acceptance-summary.json", json.dumps(summary, indent=2, default=str).encode("utf-8"),
                      replace=True)
        return summary

    checkpoint()
    try:
        halted = False
        for fixture in selected:
            for variant in VARIANTS:
                row = run_variant(out, fixture, variant, image, lock, args.hang_timeout)
                rows.append(row)
                names.update(row.get("worker_names", []))
                checkpoint()
                print(json.dumps({key: row[key] for key in ("fixture", "variant", "as_expected", "reasons",
                                                            "ledger_complete", "elapsed_seconds") if key in row}),
                      flush=True)
                if row.get("abort_campaign") or not row.get("as_expected"):
                    halted = True  # nothing after a failed calibration or cleanup doubt is evidence
                    break
            if halted:
                break
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        checks.extend(absence_checks(names, lock))
        summary = checkpoint()
    print("ACCEPTED:", summary["accepted"], "| complete:", summary["complete"], "| ledgers:",
          summary["all_ledgers_complete"], "| cleanup:", summary["all_cleanup_verified"], "| absence:",
          summary["absence_query_verified"], flush=True)
    raise SystemExit(0 if summary["accepted"] else 1)


if __name__ == "__main__":
    main()
