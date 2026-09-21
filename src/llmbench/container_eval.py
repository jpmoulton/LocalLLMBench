"""Evaluator entry point for one attached llama.cpp candidate. Import has no side effects.

Order: attach+readback -> count-route check -> overflow probe -> warmups (labelled, charged)
-> speed repetitions -> Inspect selections -> coding -> public benchmarks. The host runner owns
the server's lifecycle; this module only attaches to it. Every selected task stays in the
denominator on error or timeout.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from .config import RunMode
from .safety import OperationForbidden, SessionLock

RESULT_KEYS = ("samples", "speed_observations", "readback", "count_route", "overflow_probe",
               "actual_context_verified", "abort_reason", "errors")
CONTAINER_CONFIG_PATH = "/run/llmbench/config.json"
CONTAINER_ARTIFACTS_DIR = "/artifacts"
IMAGE_PROVENANCE_DIR = "/opt/llmbench"
BENCHMARK_DATASET_ROOT = "/opt/llmbench/benchmarks"
BENCHMARK_ARTIFACT_DIR = "benchmarks"


@dataclass(frozen=True)
class EvaluationContext:
    base_url: str
    artifacts_dir: str | Path
    deadline_monotonic: float
    session_lock: SessionLock | None = None
    clock: Callable[[], float] = time.monotonic
    allow_remote: bool = False  # only the evaluator-container path sets this, explicitly
    policy_path: str | Path = "runtime-policy.json"
    artifacts: Any = None  # optional shared writer with write(rel, bytes) and trace(rel)
    coding_hook: Any = None  # run_coding_benchmarks(config, callback, remaining, artifacts, lock, *, client)
    coding_client: Any = None  # host-process mode passes broker.direct_client(); container mode resolves /spool
    grant_seconds: float | None = None  # the typed child grant this deadline was derived from, if any
    sleep: Callable[[float], None] = time.sleep
    dataset_root: str | Path = BENCHMARK_DATASET_ROOT  # where the evaluator image baked the pinned corpora


def wait_ready(backend_or_transport: Any, *, deadline: float, clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep, poll_seconds: float = 1.0) -> dict[str, Any]:
    """Bounded `GET /health` poll through the backend's transport. Never claims readiness it did not observe."""
    if not math.isfinite(deadline) or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("wait_ready needs a finite deadline and a positive poll interval")
    transport = getattr(backend_or_transport, "transport", backend_or_transport)
    started, attempts, last_error = clock(), 0, None
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return {"ready": False, "attempts": attempts, "elapsed_seconds": max(0.0, clock() - started),
                    "last_error": last_error, "reason": "inference did not report healthy before the deadline"}
        if hasattr(transport, "timeout"):
            transport.timeout = max(0.05, min(5.0, remaining))
        attempts += 1
        try:
            healthy = transport.request_json("GET", "/health").get("status") == "ok"
        except Exception as exc:  # every transport failure is a retry until the deadline, then honest failure
            healthy, last_error = False, f"{type(exc).__name__}: {exc}"
        if healthy:
            return {"ready": True, "attempts": attempts, "elapsed_seconds": max(0.0, clock() - started),
                    "last_error": last_error, "reason": None}
        pause = min(poll_seconds, deadline - clock())
        if pause > 0:
            sleep(pause)


def _bounded_read(path: Path, max_bytes: int) -> bytes:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("evaluation.json must be a regular, unlinked file")
    if info.st_size > max_bytes:
        raise ValueError(f"evaluation.json is {info.st_size} bytes; the allocation is {max_bytes}")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("evaluation.json grew past the allocation while being read")
    return data


def read_evaluation(path: str | Path, *, max_bytes: int, expected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Host-side ingestion of a child's evaluation.json: bounded bytes, strict envelope, full denominator."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    try:
        evaluation = json.loads(_bounded_read(Path(path), max_bytes).decode("utf-8"))
    except OSError as exc:
        raise ValueError(f"evaluation.json is unreadable: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"evaluation.json is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation.json must be a JSON object")
    missing = [key for key in RESULT_KEYS if key not in evaluation]
    if missing:
        raise ValueError("evaluation.json lacks required keys: " + ", ".join(missing))
    if not isinstance(evaluation["samples"], list) or not isinstance(evaluation["speed_observations"], list):
        raise ValueError("samples and speed_observations must be lists")
    if not isinstance(evaluation["errors"], list) or not all(isinstance(e, dict) for e in evaluation["errors"]):
        raise ValueError("errors must be a list of objects")
    if evaluation["abort_reason"] is not None and not isinstance(evaluation["abort_reason"], str):
        raise ValueError("abort_reason must be null or a string")
    if type(evaluation["actual_context_verified"]) is not bool:
        raise ValueError("actual_context_verified must be a boolean")
    seen: list[str] = []
    for sample in evaluation["samples"]:
        if (not isinstance(sample, dict) or not isinstance(sample.get("task_id"), str)
                or not isinstance(sample.get("category"), str) or "score" not in sample or "status" not in sample):
            raise ValueError("every sample needs task_id, category, score and status")
        seen.append(sample["task_id"])
    expected = [row["task_id"] for row in expected_rows]
    absent = [task for task in expected if task not in seen]
    duplicates = sorted({task for task in seen if seen.count(task) > 1})
    unknown = sorted(set(seen) - set(expected))
    if absent or duplicates or unknown:
        raise ValueError(f"denominator mismatch: missing={absent} duplicated={duplicates} unexpected={unknown}")
    return evaluation


class PlainArtifacts:
    """Exclusive-create files confined to one directory; fallback when ``containers.artifacts`` is absent."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, rel: str) -> Path:
        pure = PurePosixPath(rel)
        if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} or ":" in part for part in pure.parts):
            raise ValueError(f"artifact path must be relative and confined: {rel!r}")
        path = (self.root / Path(*pure.parts)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"artifact path escapes the artifact directory: {rel!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, rel: str, data: bytes) -> None:
        with self._path(rel).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def trace(self, rel: str) -> Iterator[Callable[[Any], None]]:
        with self._path(rel).open("xb") as handle:
            def sink(event: Any) -> None:
                row = asdict(event) if hasattr(event, "__dataclass_fields__") else event
                handle.write(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8") + b"\n")
                handle.flush()
            yield sink


def _default_artifacts(root: str | Path, max_bytes: int) -> Any:
    """The evaluator requires the budgeted writer; missing support fails closed."""
    from .containers.artifacts import RunArtifacts
    return RunArtifacts(root, max_bytes)


def _dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=str, indent=1).encode("utf-8")


def _error(stage: str, exc: BaseException) -> dict[str, str]:
    return {"stage": stage, "error_type": type(exc).__name__, "error": str(exc)}


def speed_observations_from(evaluation: dict[str, Any]) -> list[Any]:
    """Rebuild ``SpeedObservation`` objects from the JSON rows for ``summarize_speed``."""
    from .measurement import SpeedObservation
    return [SpeedObservation(**{**row, "content_event_times": tuple(row["content_event_times"])})
            for row in evaluation["speed_observations"]]


def _missing_rows(expected: list[dict[str, Any]], samples: list[dict[str, Any]], status: str,
                  reason: str) -> list[dict[str, Any]]:
    present = {sample.get("task_id") for sample in samples}
    return [{**row, "score": 0.0, "passed": False, "status": status, "outcome_status": status, "reason": reason,
             "synthetic": False, "model_evaluated": False} for row in expected if row["task_id"] not in present]


def _bfcl_adapter() -> Any:
    from .benchmarks.bfcl import BfclAdapter
    return BfclAdapter()


def _ruler_adapter() -> Any:
    from .benchmarks.ruler import RulerAdapter
    return RulerAdapter()


def _evalplus_adapter() -> Any:
    from .benchmarks.evalplus import mbpp_plus
    return mbpp_plus()


def _aider_adapter() -> Any:
    from .benchmarks.aider_polyglot import AiderPolyglotBenchmark
    return AiderPolyglotBenchmark()


BENCHMARK_ADAPTERS: dict[str, Callable[[], Any]] = {
    "bfcl": _bfcl_adapter, "ruler": _ruler_adapter,
    "evalplus": _evalplus_adapter, "aider-polyglot": _aider_adapter,
}
"""Registry benchmark ID -> adapter constructor. This is the only place the evaluator knows how to
build a public-benchmark adapter; each builder imports lazily so module import stays inert. It is
also the seam the tests replace (``monkeypatch.setitem``) to drive the dispatch stage with a fake."""


@dataclass(frozen=True)
class _Wiring:
    """How one adapter is actually fed, because the four contracts genuinely differ.

    ``route`` is the ``options`` key the completion route arrives under and ``shape`` says what the
    adapter expects there: ``callback`` for the evaluator's async ``request -> response`` capture
    callback (RULER ``completion``, EvalPlus ``completion``, Aider ``transport``), or ``bytes`` for
    BFCL, whose ``ChatTransport.post_chat`` must hand back the undecoded response body because BFCL
    persists and digests those bytes before parsing them.

    ``options`` maps a *registry* option name onto the name the adapter's ``run()`` actually reads;
    the two are not the same for RULER (``tasks``/``lengths`` -> ``ruler_tasks``/``ruler_lengths``)
    or Aider (``split_fraction`` -> ``development_fraction``). A selected option with no entry here
    is a wiring gap, reported as ``environment_error`` rows - never silently dropped and never
    forwarded under a name the adapter would reject.

    ``context`` names the evaluator-derived options an adapter reads beyond the route; only RULER
    takes any, because only RULER re-counts prompts through the serving tokenizer.
    """

    route: str
    shape: str
    execution: bool
    options: dict[str, str] = field(default_factory=dict)
    context: tuple[str, ...] = ()


BENCHMARK_WIRING: dict[str, _Wiring] = {
    "bfcl": _Wiring("transport", "bytes", False,
                    {"modes": "modes", "items_per_category": "items_per_category"}),
    "ruler": _Wiring("completion", "callback", False,
                     {"tasks": "ruler_tasks", "lengths": "ruler_lengths",
                      "ruler_output_tokens": "ruler_output_tokens"},
                     ("token_counter", "context_capacity", "tokenizer_id", "template_id",
                      "tokenizer_verified")),
    "evalplus": _Wiring("completion", "callback", True,
                        {"item_limit": "item_limit", "min_item_seconds": "min_item_seconds",
                         "execution_timeout_seconds": "execution_timeout_seconds"}),
    "aider-polyglot": _Wiring("transport", "callback", True,
                              {"languages": "languages", "expected_commit": "expected_commit",
                               "split_fraction": "development_fraction"}),
}
"""Only ``execution=True`` adapters are handed the isolated worker client; the others must never see
one, because nothing they produce is ever executed."""


def _await_completion(route: Any, payload: dict[str, Any], timeout: float) -> Any:
    """Call the capture callback from synchronous adapter code, awaiting it when it is a coroutine."""
    import asyncio
    import inspect as inspect_module

    value = route(payload)
    if inspect_module.isawaitable(value):
        return asyncio.run(asyncio.wait_for(value, timeout=max(1.0, float(timeout))))
    return value


def _served_model(response: Any) -> str | None:
    """The model identity the server itself echoed, read back out of the retained raw stream.

    The capture callback assembles deltas and does not re-emit a top-level ``model``; BFCL insists on
    checking the served model itself. Returning the alias we asked for would make that check a
    tautology, so only a value the server actually sent is carried through, and only when every
    chunk that carried one agreed.
    """
    if not isinstance(response, Mapping):
        return None
    names = set()
    for row in response.get("llmbench_raw_stream") or []:
        data = row.get("data") if isinstance(row, Mapping) else None
        if isinstance(data, Mapping) and type(data.get("model")) is str:
            names.add(data["model"])
    return names.pop() if len(names) == 1 else None


class _BytesChatTransport:
    """BFCL's ``ChatTransport`` over the evaluator's guarded completion callback.

    BFCL requires raw bytes so it can store and digest the response before parsing it; this bridge
    re-serialises the assembled response and attaches only the served model the stream reported.
    """

    def __init__(self, route: Any) -> None:
        self._route = route

    def post_chat(self, payload: Mapping[str, Any], *, timeout: float) -> bytes:
        response = _await_completion(self._route, dict(payload), timeout)
        if isinstance(response, Mapping) and "model" not in response:
            served = _served_model(response)
            if served is not None:
                response = {**response, "model": served}
        return json.dumps(response, ensure_ascii=False, default=str).encode("utf-8")


class _ScopedWriter:
    """Prefixing view over a writer that has no ``scoped`` of its own (e.g. ``PlainArtifacts``)."""

    def __init__(self, parent: Any, prefix: str) -> None:
        self.parent, self.prefix = parent, prefix

    def write(self, relative: str, content: bytes) -> Any:
        return self.parent.write(f"{self.prefix}/{relative}", content)

    def trace(self, relative: str) -> Any:
        return self.parent.trace(f"{self.prefix}/{relative}")


def _scoped(artifacts: Any, prefix: str) -> Any:
    scoped = getattr(artifacts, "scoped", None)
    return scoped(prefix) if callable(scoped) else _ScopedWriter(artifacts, prefix)


def _suite_of(selection: Any) -> str:
    return str(getattr(selection, "benchmark_id", None) or getattr(selection, "suite", ""))


def _namespaced_selections(benchmarks: Sequence[Any], registry: Any) -> list[tuple[Any, Any]]:
    """The selections whose registry entry draws its task list from a baked dataset, in order."""
    pairs = []
    for selection in benchmarks:
        entry = registry.get(_suite_of(selection))
        if entry is not None and entry.task_namespace is not None:
            pairs.append((selection, entry))
    return pairs


def _execution_client(ctx: EvaluationContext) -> Any:
    """The same isolated worker the coding stage uses, resolved the same way.

    Absent means the coding-capable adapters are handed nothing and report ``environment_error``.
    There is deliberately no host fallback: model-written code runs in the worker or nowhere.
    """
    if ctx.coding_client is not None:
        return ctx.coding_client
    if not ctx.allow_remote:
        return None
    try:
        from .coding.broker_client import SpoolClient
        return SpoolClient()
    except Exception:
        return None


def _renamed_options(benchmark_id: str, declared: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The selection's own options under the names this adapter reads, or the reason they cannot pass."""
    wiring = BENCHMARK_WIRING.get(benchmark_id)
    if wiring is None:
        return {}, f"no dispatch wiring is declared for {benchmark_id}"
    unwired = sorted(set(declared) - set(wiring.options))
    if unwired:
        return {}, "the evaluator has no wiring for selected option(s): " + ", ".join(unwired)
    return {wiring.options[name]: value for name, value in declared.items()}, None


def _benchmark_options(benchmark_id: str, declared: Mapping[str, Any], *, route: Any, client: Any,
                       supplied: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Translate the selection's validated options into what this adapter's ``run()`` reads."""
    options, problem = _renamed_options(benchmark_id, declared)
    if problem is not None:
        return {}, problem
    wiring = BENCHMARK_WIRING[benchmark_id]
    options[wiring.route] = _BytesChatTransport(route) if wiring.shape == "bytes" else route
    if wiring.execution:
        options["execution_client"] = client
    absent = [name for name in wiring.context if name not in supplied]
    if absent:
        return {}, "the evaluator could not supply: " + ", ".join(absent)
    options.update({name: supplied[name] for name in wiring.context})
    return options, None


def _checked_rows(rows: Any, declared: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only well-formed rows for declared tasks, in declaration order, and name every rejection.

    A row for a task nobody selected is an error, not a bonus: it is dropped and reported, and the
    declared task it failed to cover is backfilled by the caller.
    """
    from .benchmarks import REQUIRED_SAMPLE_KEYS

    if not isinstance(rows, list):
        return [], [f"the adapter returned {type(rows).__name__}, not a list of rows"]
    problems: list[str] = []
    kept: dict[str, dict[str, Any]] = {}
    allowed = set(declared)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index} is {type(row).__name__}, not an object")
        elif row.get("task_id") not in allowed:
            problems.append(f"row {index} reports undeclared task {row.get('task_id')!r}")
        elif [key for key in REQUIRED_SAMPLE_KEYS if key not in row]:
            absent = ", ".join(key for key in REQUIRED_SAMPLE_KEYS if key not in row)
            problems.append(f"{row['task_id']} lacks required key(s): {absent}")
        elif row["task_id"] in kept:
            problems.append(f"{row['task_id']} was reported twice")
        else:
            kept[row["task_id"]] = row
    return [kept[task_id] for task_id in declared if task_id in kept], problems


def _benchmark_backfill(selection: Any, entry: Any, declared: Sequence[str],
                        produced: Sequence[dict[str, Any]], *, status: str, reason: str) -> list[dict[str, Any]]:
    from .benchmarks import missing_rows
    return missing_rows(declared, produced, suite=entry.benchmark_id, revision=selection.revision,
                        category=entry.category, split=selection.split, status=status, reason=reason)


def _benchmark_context(selection: Any, ctx: EvaluationContext, *, alias: str, generation: Any,
                       remaining: Callable[[], float], artifacts: Any, session_lock: Any,
                       options: Mapping[str, Any]) -> Any:
    from .benchmarks import BenchmarkContext
    return BenchmarkContext(base_url=ctx.base_url, model_alias=alias,
                            task_ids=tuple(selection.task_ids), split=selection.split,
                            seed=int(getattr(selection, "seed", 42)), generation=generation,
                            remaining_seconds=remaining, artifacts=artifacts,
                            session_lock=session_lock, dataset_root=str(ctx.dataset_root),
                            options=dict(options))


def _budget_status(remaining: Callable[[], float]) -> str:
    """Rows a benchmark never produced are ``timeout`` once the wall budget is gone, never dropped."""
    try:
        remaining()
    except TimeoutError:
        return "timeout"
    except Exception:
        return "environment_error"
    return "environment_error"


def _run_one_benchmark(selection: Any, entry: Any, adapter: Any, result: dict[str, Any], *,
                       ctx: EvaluationContext, artifacts: Any, backend: Any, alias: str, lock: Any,
                       n_ctx: int, overflow_passed: bool, request_timeout: float,
                       remaining: Callable[[], float], check: Callable[[], None], generation: Any,
                       supplied: Mapping[str, Any], client: Any, preflight_reason: str) -> None:
    """Run one public benchmark and append exactly one row per declared task to ``result["samples"]``.

    Raises ``BenchmarkAborted`` after recording the rows it already has and setting ``abort_reason``,
    so an abort stops the campaign the way ``UnsafeCaptureRuntimeState`` does. Everything else - a
    missing adapter, an unavailable dataset, an adapter that raises, rows that break the shared
    contract - becomes rows for the declared tasks, because the denominator may never shrink.
    """
    from .backends.llamacpp import context_evidence_callback
    from .benchmarks import BenchmarkAborted
    from .evaluations.capture import backend_completion_callback

    benchmark_id = entry.benchmark_id
    declared = tuple(selection.task_ids)
    wiring = BENCHMARK_WIRING.get(benchmark_id)
    record: dict[str, Any] = {"benchmark_id": benchmark_id, "revision": selection.revision,
                              "split": selection.split, "declared": len(declared), "produced": 0,
                              "status": "ok", "reason": None, "problems": [], "backfilled": 0,
                              "execution_client": bool(client) if wiring and wiring.execution else None}
    result["benchmarks"].append(record)

    def finish(rows: Sequence[dict[str, Any]], status: str, reason: str) -> None:
        """Publish this benchmark's rows: what it produced, plus one backfilled row per silent task."""
        filled = _benchmark_backfill(selection, entry, declared, rows, status=status, reason=reason)
        record.update(produced=len(rows), backfilled=len(filled))
        result["samples"].extend(list(rows) + filled)

    def fail(status: str, reason: str, rows: Sequence[dict[str, Any]] = (), *, backfill: str = "") -> None:
        """``status`` describes the benchmark; ``backfill`` is the sample status its silent tasks get."""
        record.update(status=status, reason=reason)
        finish(rows, backfill or status, reason)

    if adapter is None or wiring is None:
        return fail("environment_error", preflight_reason or f"no adapter is wired for {benchmark_id}")
    try:
        remaining()
    except TimeoutError as exc:
        return fail("timeout", f"the wall budget ended before {benchmark_id} ran: {exc}")

    scope = f"{BENCHMARK_ARTIFACT_DIR}/{benchmark_id}"
    requests = [0]

    def save(response: dict[str, Any]) -> None:
        artifacts.write(f"{scope}/request-{requests[0]}.json", _dump(response))
        requests[0] += 1

    with artifacts.trace(f"{scope}/transport.jsonl") as sink:
        evidence = context_evidence_callback(
            backend_completion_callback(backend, alias, session_lock=lock, mode=RunMode.LIVE,
                                        event_sink=sink, timeout_seconds=request_timeout,
                                        deadline_monotonic=ctx.deadline_monotonic, clock=ctx.clock),
            n_ctx_slot=n_ctx, save=save,
            overflow_policy="reject-overflow-no-shift" if overflow_passed else "unknown")
        options, problem = _benchmark_options(benchmark_id, getattr(selection, "options", None) or {},
                                              route=_Guarded(evidence, check), client=client,
                                              supplied=supplied)
        if problem is not None:
            return fail("environment_error", problem)
        context = _benchmark_context(selection, ctx, alias=alias, generation=generation,
                                     remaining=remaining, artifacts=_scoped(artifacts, scope),
                                     session_lock=lock, options=options)
        try:
            ok, reason = adapter.available(context)
        except Exception as exc:  # available() promises not to raise; a broken adapter still may
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            # Never a silent skip: every declared task of an unavailable benchmark stays in the
            # denominator, carrying the adapter's own reason.
            return fail("environment_error", f"{benchmark_id} is unavailable: {reason}")
        try:
            produced = adapter.run(context)
        except BenchmarkAborted as exc:
            kept, _ = _checked_rows(list(getattr(exc, "rows", None) or []), declared)
            result["abort_reason"] = result["abort_reason"] or f"{benchmark_id} aborted: {exc}"
            fail("aborted", f"{benchmark_id} aborted: {exc}", kept, backfill="environment_error")
            raise
        except _passthrough_errors():
            raise
        except Exception as exc:
            # An adapter that raises is a harness defect: it is recorded as an error (so the child's
            # exit code is honest) and its declared tasks are backfilled, never dropped.
            result["errors"].append(_error(f"benchmarks:{benchmark_id}", exc))
            return fail(_budget_status(remaining),
                        f"{benchmark_id} failed: {type(exc).__name__}: {exc}")
        if evidence.abort_reason:
            result["abort_reason"] = evidence.abort_reason
    rows, problems = _checked_rows(produced, declared)
    record["problems"] = problems
    status = _budget_status(remaining)
    reason = "; ".join(problems)[:400] or f"{benchmark_id} returned no row for this declared task"
    finish(rows, status, reason)
    if problems:
        record.update(status="invalid_rows", reason=reason)
    elif record["backfilled"]:
        record.update(status=status, reason=reason)


def _passthrough_errors() -> tuple[type[BaseException], ...]:
    """Refusals and interrupts are campaign-level decisions, never a scored zero."""
    from .backends.base import OperationDenied
    return (OperationDenied, OperationForbidden, KeyboardInterrupt, SystemExit)


def _coding_hook(ctx: EvaluationContext, config: Any) -> Any:
    """The coding generation step is the default hook wherever a
    coding client can exist (host-process direct client, or the container path whose spool is under /spool)."""
    if not any(item.benchmark_id == "coding" for item in config.benchmarks):
        return None
    if ctx.coding_hook is not None:
        return ctx.coding_hook
    if ctx.coding_client is None and not ctx.allow_remote:
        return None  # no broker client anywhere: the rows stay in the denominator as environment_error
    from .coding.generation import run_coding_benchmarks
    return run_coding_benchmarks


def run_evaluation(config: Any, ctx: EvaluationContext, *, backend_factory: Any = None) -> dict[str, Any]:
    lock = ctx.session_lock or SessionLock.read(ctx.policy_path)
    lock.check("inference", RunMode.LIVE)  # before registry work, clients, discovery or artifacts

    def remaining() -> float:
        value = ctx.deadline_monotonic - ctx.clock()
        if value <= 0:
            raise TimeoutError("evaluation wall budget exhausted")
        return value

    from .registry import builtin_registry
    registry = builtin_registry()
    if config.registry_digest is not None and config.registry_digest != registry.digest():
        raise ValueError("registry_digest differs from this harness's benchmark registry")
    # A public benchmark's task list lives in a baked dataset, so the registry can only refuse a
    # selection whose data is absent once the adapters have been asked. This reads files; it starts
    # no client, opens no socket and loads no model, and it happens before either could exist.
    namespaced = _namespaced_selections(config.benchmarks, registry)
    adapters, baked, dataset_reasons = _benchmark_preflight(
        namespaced, ctx, alias=config.alias(), generation=config.generation, remaining=remaining,
        session_lock=lock)
    registry.validate(config.benchmarks, broker_available=config.broker is not None,
                      datasets_available=baked)
    dispatched = {entry.benchmark_id for _, entry in namespaced}

    from .backends.base import BackendError, LivePermission, OperationDenied
    from .backends.llamacpp import (
        ExpectedServer, LlamaCppBackend, context_evidence_callback, observation_with_native_timings, speed_record,
    )
    from .backends.measurement_bridge import run_speed_probe
    from .evaluations.capture import UnsafeCaptureRuntimeState, backend_completion_callback, create_capture_model
    from .evaluations.inspect_tasks import run_inspect_suite
    from .evaluations.selection import build_selected_cases, expected_task_rows
    from .measurement import build_speed_input

    artifacts = ctx.artifacts or _default_artifacts(ctx.artifacts_dir, config.limits.max_artifact_bytes)
    expected_rows = expected_task_rows(config.benchmarks)
    result: dict[str, Any] = {"samples": [], "speed_observations": [], "readback": None, "count_route": None,
                              "overflow_probe": None, "actual_context_verified": False, "abort_reason": None,
                              "errors": [], "warmups": [], "native_timings": [], "tokenizer_verified": False,
                              "inspect_logs": [], "stages": [], "benchmarks": []}
    passthrough = (OperationDenied, OperationForbidden, KeyboardInterrupt, SystemExit)
    request_timeout = float(config.bounds.request_timeout_seconds)

    def authorize() -> None:
        (ctx.session_lock or SessionLock.read(ctx.policy_path)).check("inference", RunMode.LIVE)

    @contextmanager
    def stage(name: str) -> Iterator[None]:
        started = ctx.clock()
        row = {"name": name, "status": "ok", "elapsed_seconds": None}
        result["stages"].append(row)
        try:
            remaining()
            authorize()
            yield
        except BaseException as exc:
            row["status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
            raise
        finally:
            row["elapsed_seconds"] = max(0.0, ctx.clock() - started)

    def persist(rel: str, value: Any) -> bool:
        """Evidence that cannot be stored stops the run; it is never silently dropped."""
        try:
            artifacts.write(rel, _dump(value))
            return True
        except Exception as exc:
            result["errors"].append(_error(f"persist:{rel}", exc))
            result["abort_reason"] = result["abort_reason"] or f"{rel} was not written: {type(exc).__name__}: {exc}"
            return False

    alias = config.alias()
    n_ctx = config.engine.ctx_size
    factory = backend_factory or LlamaCppBackend
    backend = factory(ctx.base_url, expected=ExpectedServer(alias=alias, n_ctx=n_ctx,
                                                            build_info=config.inference_image.build_info),
                      permissions=LivePermission(False, True, False), session_lock=ctx.session_lock,
                      policy_path=ctx.policy_path, timeout=request_timeout, allow_remote=ctx.allow_remote,
                      cache_prompt=config.engine.cache_prompt, deadline_monotonic=ctx.deadline_monotonic, clock=ctx.clock)
    quality_requests = [0]
    try:
        if ctx.allow_remote:
            # Container path: the host proved readiness from logs only; HTTP readiness is proven here, bounded by
            # the grant and the startup bound, before any identity readback.
            try:
                with stage("wait-ready"):
                    readiness = wait_ready(backend, deadline=min(ctx.deadline_monotonic,
                                                                ctx.clock() + config.bounds.startup_seconds),
                                           clock=ctx.clock, sleep=ctx.sleep)
                    result["readiness"] = readiness
                    if not readiness["ready"]:
                        raise TimeoutError(readiness["reason"])
            except passthrough:
                raise
            except Exception as exc:
                result["errors"].append(_error("wait-ready", exc))
                result["abort_reason"] = f"inference never became ready: {type(exc).__name__}: {exc}"
                status = "timeout" if isinstance(exc, TimeoutError) else "environment_error"
                return _finish(result, expected_rows, artifacts, status, result["abort_reason"])
        try:
            with stage("attach"):
                verified = backend.attach()
                result["readback"] = verified.model.raw
        except passthrough:
            raise
        except Exception as exc:
            result["errors"].append(_error("attach", exc))
            result["abort_reason"] = f"attach failed: {type(exc).__name__}: {exc}"
            return _finish(result, expected_rows, artifacts, "environment_error", "server identity was not verified")
        if not persist("readback.json", {"requested": verified.requested, "effective": verified.effective,
                                         "raw": verified.model.raw}):
            return _finish(result, expected_rows, artifacts, "environment_error", result["abort_reason"])

        try:
            with stage("count-route"):
                result["count_route"] = backend.verify_count_route()
        except passthrough:
            raise
        except Exception as exc:
            result["errors"].append(_error("count-route", exc))
            result["count_route"] = {"exact": False, "error": f"{type(exc).__name__}: {exc}"}
        result["tokenizer_verified"] = result["count_route"].get("exact") is True
        if not persist("count-route.json", result["count_route"]):
            return _finish(result, expected_rows, artifacts, "environment_error", result["abort_reason"])
        if getattr(backend, "abort_reason", None):
            result["abort_reason"] = backend.abort_reason
            return _finish(result, expected_rows, artifacts, "timeout", result["abort_reason"])

        try:
            with stage("overflow-probe"):
                result["overflow_probe"] = backend.probe_overflow_rejection()
        except passthrough:
            raise
        except Exception as exc:
            result["errors"].append(_error("overflow-probe", exc))
            result["overflow_probe"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        overflow_passed = result["overflow_probe"].get("passed") is True
        if not persist("overflow-probe.json", result["overflow_probe"]):
            return _finish(result, expected_rows, artifacts, "environment_error", result["abort_reason"])
        if getattr(backend, "abort_reason", None):
            result["abort_reason"] = backend.abort_reason
        if result["abort_reason"]:
            return _finish(result, expected_rows, artifacts, "environment_error", result["abort_reason"])

        def count(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            remaining()
            return backend.count_chat_tokens(messages, tools)

        generation = config.generation
        try:
            with stage("speed"), artifacts.trace("speed/transport.jsonl") as sink:
                prompt = build_speed_input(config.requested_input_tokens, count)
                payload = {"messages": prompt["messages"], "max_tokens": config.speed.output_tokens,
                           "temperature": generation.temperature, "top_p": generation.top_p,
                           "seed": generation.seed, "ignore_eos": config.speed.ignore_eos}
                if generation.top_k is not None:
                    payload["top_k"] = generation.top_k
                plan = ([("warmup", index) for index in range(config.speed.warmup_repetitions)]
                        + [("speed", index) for index in range(config.speed.repetitions)])
                if not persist("speed/input.json", {"expected_input_tokens": prompt["tokens"],
                                                    "method": prompt.get("method"), "payload": payload}):
                    plan = []
                for label, index in plan:
                    authorize()
                    probe = run_speed_probe(backend, verified, payload, expected_input_tokens=prompt["tokens"],
                                            event_sink=sink, timeout_seconds=min(remaining(), request_timeout))
                    observation = observation_with_native_timings(
                        probe, require_cold_prompt=not config.engine.cache_prompt)
                    record = speed_record(probe, observation, label=label, index=index)
                    stored = persist(f"speed/{label}-{index}.json", record)
                    row = json.loads(json.dumps(asdict(observation)))
                    if label == "warmup":
                        result["warmups"].append(row)
                    else:
                        result["speed_observations"].append(row)
                        result["native_timings"].append(record["native_timings"])
                    if probe.worker_terminated is False or probe.cancellation_requested:
                        result["abort_reason"] = (f"{label} probe {index} was cancelled or its worker did not "
                                                  "terminate; server idleness is unverified")
                    if not stored or result["abort_reason"]:
                        break
        except passthrough:
            raise
        except Exception as exc:
            result["errors"].append(_error("speed", exc))
            if getattr(backend, "abort_reason", None):
                result["abort_reason"] = backend.abort_reason
            if getattr(backend, "pending_request_ids", ()):
                result["abort_reason"] = "a speed request is still active after a failure; server idleness is unverified"
        if result["abort_reason"]:
            return _finish(result, expected_rows, artifacts, "environment_error", result["abort_reason"])

        props = verified.model.raw.get("props", {})
        raw_template = props.get("chat_template")
        template_id = ("sha256:" + hashlib.sha256(raw_template.encode("utf-8")).hexdigest()
                       if isinstance(raw_template, str) and raw_template else None)
        tokenizer_id = "gguf-sha256:" + config.assets[0].sha256
        # The public suites run in their own stage; Inspect must never be handed a suite it cannot run.
        inspect_benchmarks = [item for item in config.benchmarks if _suite_of(item) not in dispatched]
        failure_status, failure_reason = "environment_error", "quality stage did not run"
        try:
            with stage("quality"), artifacts.trace("quality/transport.jsonl") as sink:
                if template_id is None:
                    raise BackendError("/props has no chat_template to identify the prompt template")
                cases, selections = build_selected_cases(
                    inspect_benchmarks, requested_input_tokens=config.requested_input_tokens, ctx_size=n_ctx,
                    generation=generation, counter=count, tokenizer_id=tokenizer_id,
                    template_id=template_id, tokenizer_verified=result["tokenizer_verified"])
                callback = backend_completion_callback(backend, alias, session_lock=lock, mode=RunMode.LIVE,
                                                       event_sink=sink, timeout_seconds=request_timeout,
                                                       deadline_monotonic=ctx.deadline_monotonic, clock=ctx.clock)

                def save(response: dict[str, Any]) -> None:
                    artifacts.write(f"quality/quality-request-{quality_requests[0]}.json", _dump(response))
                    quality_requests[0] += 1

                evidence = context_evidence_callback(
                    callback, n_ctx_slot=n_ctx, save=save,
                    overflow_policy="reject-overflow-no-shift" if overflow_passed else "unknown")

                def log_failure(exc: BaseException) -> None:
                    result["abort_reason"] = (result["abort_reason"] or
                        f"Inspect evidence could not be written: {type(exc).__name__}: {exc}")
                    # A log write may fail while a request is active. Cancel only
                    # our reserved requests; cleanup remains the bridge/runner's job.
                    for request_id in getattr(backend, "pending_request_ids", ()):
                        def cancel(identifier: str = request_id) -> None:
                            try:
                                backend.cancel(identifier)
                            except BaseException:
                                pass  # the abort already records unverified runtime state
                        threading.Thread(target=cancel, daemon=True, name="llmbench-log-failure-cancel").start()

                def check() -> None:
                    if result["abort_reason"]:
                        raise UnsafeCaptureRuntimeState(result["abort_reason"])
                    authorize()
                    remaining()
                model = create_capture_model(_Guarded(evidence, check), model_name=alias, session_lock=lock,
                                             mode=RunMode.LIVE, generation=generation)
                if selections:
                    output = run_inspect_suite(model, Path(ctx.artifacts_dir) / "inspect", selections=selections,
                                               session_lock=lock, mode=RunMode.LIVE, generation=generation,
                                               timeout_seconds=max(1, int(min(remaining(), request_timeout))),
                                               niah_cases=cases, artifacts=artifacts, on_log_failure=log_failure)
                    result["samples"].extend(output["samples"])
                    result["inspect_logs"] = output["logs"]
                if evidence.abort_reason:
                    result["abort_reason"] = evidence.abort_reason
            coding_hook = _coding_hook(ctx, config)
            if coding_hook is not None and not result["abort_reason"]:
                # LIVE-007: the coding stage owns its raw sink. Handing it the quality callback published the
                # first coding request into a trace file whose ``with`` block had already closed, and the run
                # aborted on "Raw capture persistence failed". Separate traces also keep the two stages' raw
                # evidence independently identifiable; the guards are otherwise the quality path's own.
                with stage("coding"), artifacts.trace("coding/transport.jsonl") as coding_sink:
                    coding_requests = [0]

                    def save_coding(response: dict[str, Any]) -> None:
                        artifacts.write(f"coding/coding-request-{coding_requests[0]}.json", _dump(response))
                        coding_requests[0] += 1

                    coding_evidence = context_evidence_callback(
                        backend_completion_callback(backend, alias, session_lock=lock, mode=RunMode.LIVE,
                                                    event_sink=coding_sink, timeout_seconds=request_timeout,
                                                    deadline_monotonic=ctx.deadline_monotonic, clock=ctx.clock),
                        n_ctx_slot=n_ctx, save=save_coding,
                        overflow_policy="reject-overflow-no-shift" if overflow_passed else "unknown")
                    rows = coding_hook(config, _Guarded(coding_evidence, check), remaining, artifacts, lock,
                                       client=ctx.coding_client)
                    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                        raise TypeError("the coding hook must return a list of sample dictionaries")
                    result["samples"].extend(rows)
                    if coding_evidence.abort_reason:
                        result["abort_reason"] = coding_evidence.abort_reason
            if namespaced and not result["abort_reason"]:
                # LIVE-007 again, in a third stage: every public benchmark gets its OWN trace sink and
                # its OWN completion callback, both opened here. Nothing below may publish through the
                # quality or coding sinks - those ``with`` blocks have already closed.
                with stage("benchmarks"):
                    supplied = {"token_counter": count, "context_capacity": n_ctx,
                                "tokenizer_id": tokenizer_id, "template_id": template_id,
                                "tokenizer_verified": result["tokenizer_verified"]}
                    worker_client = _execution_client(ctx)
                    for selection, entry in namespaced:
                        _run_one_benchmark(selection, entry, adapters.get(entry.benchmark_id), result,
                                           ctx=ctx, artifacts=artifacts, backend=backend, alias=alias,
                                           lock=lock, n_ctx=n_ctx, overflow_passed=overflow_passed,
                                           request_timeout=request_timeout, remaining=remaining,
                                           check=check, generation=generation, supplied=supplied,
                                           client=worker_client,
                                           preflight_reason=dataset_reasons.get(entry.benchmark_id, ""))
                        if result["abort_reason"]:
                            break
        except passthrough:
            raise
        except Exception as exc:
            result["errors"].append(_error(result["stages"][-1]["name"], exc))
            if getattr(exc, "abort_campaign", False):
                result["abort_reason"] = str(exc)
            if isinstance(exc, TimeoutError):
                failure_status = "timeout"
            failure_reason = f"{type(exc).__name__}: {exc}"
        return _finish(result, expected_rows, artifacts, failure_status, failure_reason)
    finally:
        backend.close()


class _Guarded:
    """Re-checks policy and wall budget before every guarded request; keeps ``abort_reason`` visible."""

    def __init__(self, inner: Any, check: Callable[[], None]) -> None:
        self._inner, self._check = inner, check

    @property
    def abort_reason(self) -> str | None:
        return self._inner.abort_reason

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check()
        return await self._inner(request)


def _finish(result: dict[str, Any], expected_rows: list[dict[str, Any]], artifacts: Any, status: str,
            reason: str) -> dict[str, Any]:
    result["samples"].extend(_missing_rows(expected_rows, result["samples"], status, reason))
    contexts = [sample for sample in result["samples"] if sample.get("category") == "retrieval"]
    # Same rule as live.py: a context claim needs at least one retrieval sample, all of them verified.
    result["actual_context_verified"] = bool(contexts) and all(
        (sample.get("context") or {}).get("actual_context_verified") is True for sample in contexts)
    clean = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    try:
        artifacts.write("evaluation.json", _dump(clean))
    except Exception as exc:  # The caller still receives the result; the lost file is itself recorded.
        clean["errors"].append(_error("persist", exc))
        clean["abort_reason"] = clean["abort_reason"] or f"evaluation.json was not written: {type(exc).__name__}: {exc}"
        # A byte-capped child must still leave its honest verdict for the host: same content, compact encoding,
        # carrying the persist error above. If even that fails the file stays absent and the host fails closed.
        try:
            artifacts.write("evaluation.json", json.dumps(clean, ensure_ascii=False, separators=(",", ":"),
                                                          default=str).encode("utf-8"))
        except Exception as fallback:
            clean["errors"].append(_error("persist-compact", fallback))
    return clean


def _provenance(directory: str | Path) -> dict[str, Any]:
    """Build-time files the evaluator image records; absent on the host, present inside the image."""
    root = Path(directory)
    found: dict[str, Any] = {"python_version": None, "pip_freeze_sha256": None, "provenance_dir": str(root)}
    version, freeze = root / "python-version.txt", root / "pip-freeze.txt"
    if version.is_file():
        found["python_version"] = version.read_bytes()[:200].decode("utf-8", "replace").strip()
    if freeze.is_file():
        found["pip_freeze_sha256"] = hashlib.sha256(freeze.read_bytes()).hexdigest()
    found["runtime_python"] = sys.version.split()[0]
    return found


def _build_adapter(benchmark_id: str) -> Any:
    builder = BENCHMARK_ADAPTERS.get(benchmark_id)
    if builder is None:
        raise LookupError("no adapter is wired for this benchmark")
    return builder()


def baked_datasets(dataset_root=None) -> tuple[str, ...]:
    """Benchmark ids whose pinned corpora are present under ``dataset_root``.

    Shared by the host runner at admission and by the image self-check, so the pre-load refusal and
    the self-check report can never disagree about what this environment actually has.
    """
    from .benchmarks import BenchmarkContext
    root = str(dataset_root or BENCHMARK_DATASET_ROOT)
    present = []
    for benchmark_id, build in BENCHMARK_ADAPTERS.items():
        try:
            context = BenchmarkContext(base_url="", model_alias="", task_ids=(), split="development",
                                       seed=0, generation=None, remaining_seconds=lambda: 0.0,
                                       artifacts=None, session_lock=None, dataset_root=root)
            ok, _ = build().available(context)
        except Exception:
            ok = False
        if ok:
            present.append(benchmark_id)
    return tuple(present)


def _dataset_available(benchmark_id: str) -> tuple[bool, str]:
    """Ask the adapter whether its pinned dataset is baked here. A missing dataset is reported, not
    raised: the self-check describes the image, and the registry refuses the selection later."""
    from .benchmarks import BenchmarkContext
    try:
        adapter = _build_adapter(benchmark_id)
        context = BenchmarkContext(base_url="", model_alias="", task_ids=(), split="development", seed=0,
                                   generation=None, remaining_seconds=lambda: 0.0, artifacts=None,
                                   session_lock=None)
        ok, reason = adapter.available(context)
        return bool(ok), str(reason)
    except LookupError as exc:
        return False, str(exc)
    except Exception as exc:  # a broken adapter must not crash the image self-check
        return False, f"{type(exc).__name__}: {exc}"


def _benchmark_preflight(pairs: Sequence[tuple[Any, Any]], ctx: EvaluationContext, *, alias: str,
                         generation: Any, remaining: Callable[[], float],
                         session_lock: Any) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Build each selected public benchmark's adapter once and ask it whether its dataset is baked.

    The registry refuses a benchmark whose data this image does not have *before* a model is loaded,
    so its ``datasets_available`` argument has to come from the adapters' own ``available()``. The
    adapter instances are kept and reused by the dispatch stage, which asks again with the live
    wiring attached; that second answer is what turns into rows.
    """
    adapters: dict[str, Any] = {}
    baked: list[str] = []
    reasons: dict[str, str] = {}
    for selection, entry in pairs:
        benchmark_id = entry.benchmark_id
        try:
            adapter = adapters[benchmark_id] = _build_adapter(benchmark_id)
            options, problem = _renamed_options(benchmark_id, getattr(selection, "options", None) or {})
            if problem is not None:
                raise ValueError(problem)
            context = _benchmark_context(selection, ctx, alias=alias, generation=generation,
                                         remaining=remaining, artifacts=None, session_lock=session_lock,
                                         options=options)
            ok, reason = adapter.available(context)
        except Exception as exc:  # available() must not raise; a builder still can
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        reasons[benchmark_id] = str(reason)
        if ok:
            baked.append(benchmark_id)
    return adapters, baked, reasons


def self_check(*, provenance_dir: str | Path = IMAGE_PROVENANCE_DIR) -> dict[str, Any]:
    """Imports, registry digest and task availability. No network, model, Docker or policy access."""
    report: dict[str, Any] = {"ok": False, "imports": {}, "registry_digest": None, "tasks": {},
                              "datasets": {}, "coding": {},
                              "errors": [], **_provenance(provenance_dir)}
    for name in ("inspect_ai", "pydantic", "llmbench.backends.llamacpp", "llmbench.evaluations.capture",
                 "llmbench.evaluations.inspect_tasks", "llmbench.evaluations.selection", "llmbench.registry"):
        try:
            __import__(name)
            report["imports"][name] = True
        except Exception as exc:
            report["imports"][name] = False
            report["errors"].append(f"import {name}: {type(exc).__name__}: {exc}")
    try:
        from .evaluations.tools import ToolEpisode, strict_argument_fixture
        from .evaluations.retrieval import NIAH_VARIANTS
        from .registry import builtin_registry
        registry = builtin_registry()
        report["registry_digest"] = registry.digest()
        available = {"tool-probes": {strict_argument_fixture().task_id, "tools/no-call-v1"},
                     "tool-episodes": {ToolEpisode().task_id}, "niah": set(NIAH_VARIANTS)}
        for entry in registry.entries:
            if entry.capability != "runner":
                continue
            if entry.task_namespace is not None:
                # Its tasks come from a baked dataset, so report whether this image actually has it.
                ok, reason = _dataset_available(entry.benchmark_id)
                report["datasets"][entry.benchmark_id] = {"available": ok, "reason": reason,
                                                          "namespace": entry.task_namespace}
                continue
            missing = sorted(set(entry.task_ids) - available.get(entry.benchmark_id, set()))
            report["tasks"][entry.benchmark_id] = {"task_ids": list(entry.task_ids), "missing": missing}
            if missing:
                report["errors"].append(f"{entry.benchmark_id}: no runner for {', '.join(missing)}")
        from .coding.fixtures import fixtures
        coding = registry.get("coding")
        present = {item.fixture_id: item.identity() for item in fixtures()}
        report["coding"] = {"fixtures": present, "task_ids": list(coding.task_ids) if coding else [],
                            "fixtures_present": coding is not None and set(coding.task_ids) == set(present)}
        if not report["coding"]["fixtures_present"]:
            report["errors"].append("coding: registry task ids do not match the private fixtures present")
    except Exception as exc:
        report["errors"].append(f"registry: {type(exc).__name__}: {exc}")
    report["ok"] = not report["errors"]
    return report


def main(argv: list[str] | None = None, *, base_url: str | None = None,
         artifacts_dir: str | Path = CONTAINER_ARTIFACTS_DIR, backend_factory: Any = None) -> int:
    """Exit codes: 0 evaluation written without abort/errors, 2 invalid invocation/config/policy, 3 otherwise.

    ``--grant-seconds`` is mandatory on the container path (the compose-internal inference URL): the host's
    typed grant, not ``bounds.evaluation_seconds`` alone, bounds the child. ``--policy`` defaults to the
    ``runtime-policy.json`` beside the config; ``--artifact-bytes`` caps the child's own artifact tree.
    """
    parser = argparse.ArgumentParser(prog="python -m llmbench.container_eval", allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--self-check", action="store_true")
    parser.add_argument("--grant-seconds", type=int)
    parser.add_argument("--policy")
    parser.add_argument("--artifact-bytes", type=int)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    if args.self_check:
        if args.grant_seconds is not None or args.policy is not None or args.artifact_bytes is not None:
            print("--self-check takes no other arguments", file=sys.stderr)
            return 2
        report = self_check()
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 3
    if args.grant_seconds is not None and args.grant_seconds < 1:
        print("--grant-seconds must be a positive integer", file=sys.stderr)
        return 2
    if args.artifact_bytes is not None and args.artifact_bytes < 1:
        print("--artifact-bytes must be a positive integer", file=sys.stderr)
        return 2
    try:
        from .containers.config import read_run_config
        config = read_run_config(args.config)
    except Exception as exc:
        print(f"invalid container run config: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    from .backends.llamacpp import INFERENCE_URL
    url = base_url or INFERENCE_URL
    remote = url == INFERENCE_URL
    if remote and args.grant_seconds is None:
        print("the container path requires --grant-seconds from the host", file=sys.stderr)
        return 2
    policy = Path(args.policy) if args.policy else Path(args.config).resolve().parent / "runtime-policy.json"
    if not policy.is_file():
        print(f"runtime policy is missing: {policy}", file=sys.stderr)
        return 2
    grant = config.bounds.evaluation_seconds if args.grant_seconds is None else min(args.grant_seconds,
                                                                                    config.bounds.evaluation_seconds)
    artifacts = None
    if args.artifact_bytes is not None:
        try:
            SessionLock.read(policy).check("inference", RunMode.LIVE)  # before any directory is created
            artifacts = _default_artifacts(artifacts_dir, args.artifact_bytes)
        except (OSError, ValueError, OperationForbidden) as exc:
            print(f"evaluation refused before artifacts: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    # The compose-internal service name is not loopback. Opting in is explicit and limited to that one origin.
    ctx = EvaluationContext(base_url=url, artifacts_dir=artifacts_dir, deadline_monotonic=time.monotonic() + grant,
                            allow_remote=remote, policy_path=policy, artifacts=artifacts, grant_seconds=float(grant))
    try:
        evaluation = run_evaluation(config, ctx, backend_factory=backend_factory)
    except (OperationForbidden, ValueError) as exc:
        print(f"evaluation refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    code = 3 if evaluation["abort_reason"] or evaluation["errors"] else 0
    # One summary line on stdout: the host captures the container log, so the verdict survives even when the
    # byte cap left no room for evaluation.json itself.
    print(json.dumps({"exit_code": code, "abort_reason": evaluation["abort_reason"],
                      "samples": len(evaluation["samples"]), "errors": len(evaluation["errors"]),
                      "persist_errors": sum(1 for item in evaluation["errors"]
                                            if str(item.get("stage", "")).startswith("persist")),
                      "stages": [row["name"] for row in evaluation.get("stages", [])]}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
