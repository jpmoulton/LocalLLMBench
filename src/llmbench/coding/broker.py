"""Host-side coding broker: validates spool requests, runs them in the owned Docker worker, publishes results.

Construction performs only host-owned file setup (spool and worker directories, ledger, namespace). Docker is
reached solely through the injected or owned ``BoundedProcessExecutor`` after the session lock authorizes
container execution. Nothing in the spool is trusted; see ``validate`` for the exact order of checks.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from ..config import RunMode, canonical_json
from ..containers.config import NAME
from ..evaluations.tools import strict_json_loads
from ..store import utc_now
from .benchmark_items import (BenchmarkItemFixture, PolyglotExerciseFixture, evalplus_item_fixtures,
                              polyglot_exercise_fixtures, run_polyglot_exercise, run_test_module)
from .fixtures import CodingFixture, fixtures as default_fixtures
from .runner import run_coding_fixture
from .sandbox import BoundedProcessExecutor, DockerWorker
from .spool import (
    HEX32, NAMESPACE_NAME, REQUEST_NAME, CodingJobRequest, CodingJobResult, NotRegularFile, Oversize,
    publish_atomic, read_bounded, request_name, result_bytes,
)

CLEANUP_MARGIN_SECONDS = 20  # owned-container rm + absence query after the last case of a fixture
MAX_SCAN_ENTRIES = 1024
MAX_IGNORED_NAMES = 4096
MAX_REJECTIONS_PER_TICK = 16
LEDGER_MAX_BYTES = 64 * 1024 * 1024
IGNORED_REASONS = frozenset({"bad_name", "not_regular_file", "result_cap"})
_IMAGE = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}")


def default_broker_settings(**overrides):
    """``BrokerSettings`` with defaults. Imported here, not at module load: the config module imports this package."""
    from ..containers.config import BrokerSettings
    return BrokerSettings(**overrides)


class _LedgerWorker:
    """Delegates to the real worker; records every owned container name in the ledger before it can start."""

    def __init__(self, inner, record: Callable[[str], None]) -> None:
        self._inner, self._record = inner, record

    @property
    def session_lock(self):
        return self._inner.session_lock

    @property
    def mode(self):
        return self._inner.mode

    @property
    def allowed_root(self):
        return self._inner.allowed_root

    @property
    def synthetic(self):
        return self._inner.synthetic

    def prepare(self, *args, **kwargs):
        job = self._inner.prepare(*args, **kwargs)
        self._record(job.name)
        return job

    def run(self, job):
        return self._inner.run(job)


class HostBroker:
    def __init__(self, run_dir, config, session_lock, artifacts, *, worker_factory=None, fixtures=default_fixtures,
                 clock=time.monotonic, executor=None, attempt_id: str | None = None, session_id: str | None = None,
                 utc: Callable[[], str] = utc_now) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.config, self.session_lock, self.artifacts = config, session_lock, artifacts
        self.settings = getattr(config, "broker", None)
        if self.settings is None:
            raise ValueError("config.broker (BrokerSettings) is required for coding benchmarks")
        image = getattr(getattr(config, "worker_image", None), "reference", None)
        if type(image) is not str or not _IMAGE.fullmatch(image):
            raise ValueError("config.worker_image must be a digest-pinned worker image")
        self.image = image
        # Results are capped at 2 * max_requests; a legitimate evaluator needs up to two (first + repair) per selected
        # coding task, and a cap below that would leave its later requests without any result (silent starvation).
        selected = sum(len(getattr(item, "task_ids", ())) for item in getattr(config, "benchmarks", None) or ()
                       if getattr(item, "benchmark_id", None) == "coding")
        # Public benchmarks ride on top of the fixtures' two attempts each: an EvalPlus item is single-attempt (one
        # request), an Aider Polyglot exercise gets upstream's second try with the test output (two requests).
        items = evalplus_item_fixtures(config)
        exercises = polyglot_exercise_fixtures(config)
        needed = 2 * selected + len(items) + 2 * len(exercises)
        if needed > self.settings.max_requests:
            raise ValueError(f"broker.max_requests={self.settings.max_requests} cannot admit two attempts for each of "
                             f"the {selected} selected coding tasks and {len(exercises)} Aider Polyglot exercises "
                             f"plus one for each of the {len(items)} EvalPlus items; raise it to at least {needed}")
        items = {**items, **exercises}
        self.worker_factory, self.clock, self.utc = worker_factory, clock, utc
        self.executor = executor or BoundedProcessExecutor(session_lock=session_lock, mode=RunMode.LIVE)
        self._fixtures: dict[str, CodingFixture | BenchmarkItemFixture] = {item.fixture_id: item
                                                                           for item in fixtures()}
        self._fixtures.update(items)
        self.requests_dir = self.run_dir / "spool" / "requests"
        self.results_dir = self.run_dir / "spool" / "results"
        self.workers_root = self.run_dir / "broker" / "workers"
        self.ledger_dir = Path(getattr(artifacts, "root"))
        for directory in (self.requests_dir, self.results_dir, self.workers_root, self.ledger_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.abort_campaign, self.abort_reason = False, None
        self._ignored: set[str] = set()
        self._seen: dict[str, tuple[int, int]] = {}
        self._content: dict[str, str] = {}          # request_id -> content_sha256 of the validated request
        self._terminal: dict[str, str] = {}         # request_id -> published status
        self._results: dict[str, CodingJobResult] = {}
        self._accepted: set[str] = set()
        self._workers: dict[str, list[str]] = {}    # request_id -> owned container names recorded at prepare
        self._unfinished: set[str] = set()
        self._finished: dict[str, dict] = {}        # request_id -> last ledgered `finished` line (previous generations)
        self._results_published = 0
        self._ledger = self._sync_fd = None
        previous = self._read_previous_ledgers()
        self.session_id, self.attempt_id, source = self._resolve_namespace(previous, attempt_id, session_id)
        for event in previous:
            self._absorb(event)
        self._open_ledger()
        self._log("broker", session_id=self.session_id, attempt_id=self.attempt_id, namespace_source=source,
                  image=self.image, settings=self.settings.model_dump(mode="json"),
                  ledger_generation=self._ledger_name, previous_events=len(previous))
        self._publish_namespace()

    # ---- construction helpers ---------------------------------------------------------------------------------
    @classmethod
    def factory(cls, run_dir, config, session_lock, artifacts, *, attempt_id: str | None = None,
                session_id: str | None = None) -> "HostBroker":
        """Construct and reconcile. A runner that knows its attempt identity passes it; the ids are then checked
        against the ledger and against the attempt labels in the run's plan, never silently preferred to them."""
        broker = cls(run_dir, config, session_lock, artifacts, attempt_id=attempt_id, session_id=session_id)
        broker.reconcile()
        return broker

    @property
    def floor_seconds(self) -> float:
        """Least time in which one worker case can run and its owned container be verified absent."""
        return float(self.settings.worker_limits.timeout_seconds + CLEANUP_MARGIN_SECONDS)

    def _ledger_files(self) -> list[Path]:
        """Generations in order: ``ledger.jsonl`` (0), then ``ledger-1.jsonl``, ``ledger-2.jsonl``, ..."""
        def generation(path: Path) -> int:
            match = re.fullmatch(r"ledger(?:-(\d+))?\.jsonl", path.name)
            return int(match[1] or 0) if match else -1
        return sorted((p for p in self.ledger_dir.glob("ledger*.jsonl") if generation(p) >= 0), key=generation)

    def _read_previous_ledgers(self) -> list[dict]:
        events: list[dict] = []
        for path in self._ledger_files():
            raw = read_bounded(path, LEDGER_MAX_BYTES)
            for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                event = strict_json_loads(line)
                if type(event) is not dict or type(event.get("event")) is not str:
                    raise ValueError(f"corrupt broker ledger {path.name} line {number}")
                events.append(event)
        return events

    def _resolve_namespace(self, previous: list[dict], attempt_id, session_id) -> tuple[str, str, str]:
        recorded = next((e for e in previous if e.get("event") == "broker"), None)
        configured = getattr(self.config, "session_id", None)
        if recorded is not None:
            candidate = (recorded.get("session_id"), recorded.get("attempt_id"), "ledger")
            if (attempt_id is not None and attempt_id != candidate[1]) or (session_id is not None
                                                                          and session_id != candidate[0]):
                raise ValueError("broker namespace differs from the recorded ledger namespace")
        elif attempt_id is not None:
            candidate = (session_id or configured or attempt_id[:12], attempt_id, "argument")
            planned = self._namespace_from_plan()
            if planned is not None and planned[:2] != candidate[:2]:
                # Two sources of one identity that disagree mean the caller has the wrong run directory or the
                # wrong attempt; guessing which is right would admit the other attempt's requests.
                raise ValueError("broker namespace arguments differ from the attempt labels in the run's plan")
        else:
            candidate = self._namespace_from_plan()
            if candidate is None:
                minted = uuid.uuid4().hex
                candidate = (session_id or configured or minted[:12], minted, "minted")
            elif session_id is not None and session_id != candidate[0]:
                # A session named by the caller is checked like an attempt id, never silently replaced by the plan's.
                raise ValueError("broker namespace arguments differ from the attempt labels in the run's plan")
        if not (type(candidate[0]) is str and re.fullmatch(NAME, candidate[0])
                and type(candidate[1]) is str and re.fullmatch(HEX32, candidate[1])):
            raise ValueError("broker namespace must be a safe session name and a 32-hex attempt id")
        return candidate

    def _namespace_from_plan(self) -> tuple[str, str, str] | None:
        """The runner writes its plan before the broker stage; the plan's labels carry the attempt identity.

        A container run has ``plan/compose.json`` (labels of the inference service) and it is read exactly as it
        always was. A native run has no compose file: its llama-server launch record ``plan/native-launch.json``
        carries the same two labels under ``labels`` (source ``native-plan``). The native record is consulted only
        when no compose file exists, so a container run can never pick its namespace up from a stray native record.
        A plan file that exists but cannot be read is an error, never a reason to mint a fresh namespace.
        """
        sources = (("compose.json", "plan", lambda plan: plan["services"]["inference"]["labels"]),
                   ("native-launch.json", "native-plan", lambda plan: plan["labels"]))
        for name, source, locate in sources:
            try:
                plan = strict_json_loads(read_bounded(self.run_dir / "plan" / name, 1_048_576).decode("utf-8"))
                labels = locate(plan)
                return (labels["llmbench.session"], labels["llmbench.attempt"], source)
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                raise ValueError(f"plan/{name} exists but its attempt labels are unreadable: {exc}") from exc
        return None

    def _open_ledger(self) -> None:
        existing = self._ledger_files()
        self._ledger_name = "ledger.jsonl" if not existing else f"ledger-{len(existing)}.jsonl"
        self._ledger = self.artifacts.open_external(self._ledger_name, "xb")
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        self._sync_fd = os.open(self.ledger_dir / self._ledger_name, flags)

    def _log(self, event: str, **fields: Any) -> None:
        if self._ledger is None or self._ledger.closed:
            raise RuntimeError("broker ledger is closed")
        line = (canonical_json({"event": event, "utc": self.utc(), **fields}) + "\n").encode("utf-8")
        self._ledger.write(line)  # the shared artifact budget is charged before the bytes reach the disk
        self._ledger.flush()
        os.fsync(self._sync_fd)

    def _absorb(self, event: dict) -> None:
        kind, request_id = event.get("event"), event.get("request_id")
        if kind == "ignored" and type(event.get("name")) is str:
            self._ignored.add(event["name"])
        elif kind == "validated" and type(request_id) is str:
            self._accepted.add(request_id)
            self._content[request_id] = event.get("content_sha256")
        elif kind == "started" and type(request_id) is str:
            self._unfinished.add(request_id)
        elif kind == "worker" and type(request_id) is str and type(event.get("name")) is str:
            self._workers.setdefault(request_id, []).append(event["name"])
        elif kind == "finished" and type(request_id) is str:
            self._unfinished.discard(request_id)
            self._finished[request_id] = event
            if event.get("status") == "cleanup-unverified":
                self.abort_campaign, self.abort_reason = True, event.get("failure_reason") or "cleanup unverified"
        elif kind in {"published", "republished"} and type(request_id) is str:
            self._terminal[request_id] = event.get("status")
            self._results_published += 1

    def _publish_namespace(self) -> None:
        payload = (canonical_json({"schema_version": 1, "session_id": self.session_id,
                                   "attempt_id": self.attempt_id}) + "\n").encode("utf-8")
        try:
            existing = read_bounded(self.results_dir / NAMESPACE_NAME, 4096)
        except FileNotFoundError:
            publish_atomic(self.results_dir, NAMESPACE_NAME, payload)
            return
        if existing != payload:
            raise ValueError("spool namespace.json belongs to another broker attempt")

    @property
    def closed(self) -> bool:
        return self._ledger is None or self._ledger.closed

    def close(self) -> None:
        if self._ledger is not None and not self._ledger.closed:
            self._ledger.close()
        if self._sync_fd is not None:
            os.close(self._sync_fd)
            self._sync_fd = None

    # ---- validation -----------------------------------------------------------------------------------------
    def validate(self, path: Path, *, max_seconds: float | None = None) -> tuple[CodingJobRequest | None, str | None]:
        """Checks in plan order; never raises on spool content.

        Returns ``(request, None)`` for an accepted request, ``(request, "replay")`` for an exact replay of a
        finished request and ``(None, reason)`` otherwise. Reasons in ``IGNORED_REASONS`` produce no result.
        """
        path = Path(path)
        name = path.name
        if not re.fullmatch(REQUEST_NAME, name):
            return None, "bad_name"
        try:
            if path.parent.resolve() != self.requests_dir:  # only entries of the owned spool directory
                return None, "bad_name"
        except OSError:
            return None, "bad_name"
        try:
            raw = read_bounded(path, self.settings.max_request_bytes)
        except Oversize:
            return None, "oversize"
        except (NotRegularFile, OSError):
            return None, "not_regular_file"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "invalid_utf8"
        try:
            parsed = strict_json_loads(text)
        except (ValueError, RecursionError) as exc:
            return None, f"invalid_json: {str(exc)[:120]}"
        if type(parsed) is not dict:
            return None, "invalid_json: not an object"
        try:
            request = CodingJobRequest.model_validate(parsed)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            location = ".".join(str(item) for item in first.get("loc", ()))
            return None, f"schema: {location}: {first.get('msg', '')}"[:200]
        if request.request_id != name[:-5]:
            return None, "request_id_mismatch"
        if (request.session_id, request.attempt_id) != (self.session_id, self.attempt_id):
            return None, "foreign_namespace"
        fixture = self._fixtures.get(request.fixture_id)
        if fixture is None:
            return None, "unknown_fixture"
        if request.fixture_revision != fixture.revision:
            return None, "fixture_revision"
        if request.fixture_hash != fixture.identity():
            return None, "fixture_hash"
        if set(request.patch) - {item.path for item in fixture.initial_files}:
            return None, "patch_outside_allowlist"
        if request.patch_bytes() > self.settings.max_patch_bytes:
            return None, "patch_too_large"
        if request.request_id not in self._accepted and len(self._accepted) >= self.settings.max_requests:
            return None, "too_many_requests"
        sha = request.content_sha256()
        if request.request_id in self._terminal:
            if self._content.get(request.request_id) == sha:
                return request, "replay"
            return None, "replay_changed_content"
        if max_seconds is not None and max_seconds < self.floor_seconds:
            return None, "no_time"
        return request, None

    # ---- execution ------------------------------------------------------------------------------------------
    def _worker(self, request_id: str):
        def record(name: str) -> None:
            self._workers.setdefault(request_id, []).append(name)
            self._log("worker", request_id=request_id, name=name)
        if self.worker_factory is not None:
            inner = self.worker_factory(self.workers_root, self.session_lock, self.executor)
        else:
            inner = DockerWorker(allowed_root=self.workers_root, session_lock=self.session_lock, mode=RunMode.LIVE,
                                 executor=self.executor)
        return _LedgerWorker(inner, record)

    def _result(self, request_id: str, sha: str, status: str, *, sample=None, failure_reason=None, trace=None,
                cleanup_confirmed: bool, abort: bool = False) -> CodingJobResult:
        return CodingJobResult(request_id=request_id, request_sha256=sha, status=status, sample=sample,
                               failure_reason=failure_reason, trace_sha256=trace, cleanup_confirmed=cleanup_confirmed,
                               abort_campaign=abort, finished_utc=self.utc())

    def execute(self, request: CodingJobRequest, *, max_seconds: float) -> CodingJobResult:
        """Run every check of the fixture through ``run_coding_fixture`` under the host-owned worker root."""
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds):
            raise ValueError("max_seconds must be finite")
        if request.fixture_id not in self._fixtures:
            raise ValueError(f"unknown fixture {request.fixture_id}; validate the request first")
        request_id, sha = request.request_id, request.content_sha256()
        fixture = self._fixtures[request.fixture_id]
        if max_seconds < self.floor_seconds:
            return self._result(request_id, sha, "rejected", failure_reason="no_time", cleanup_confirmed=True)
        budget = float(min(self.settings.fixture_timeout_seconds, max_seconds - CLEANUP_MARGIN_SECONDS))
        started = self.clock()
        self._unfinished.add(request_id)
        self._log("started", request_id=request_id, fixture_id=request.fixture_id, attempt_index=request.attempt_index,
                  budget_seconds=budget)
        run = (run_test_module if isinstance(fixture, BenchmarkItemFixture)
               else run_polyglot_exercise if isinstance(fixture, PolyglotExerciseFixture) else run_coding_fixture)
        try:
            outcome = run(fixture, request.patch, worker=self._worker(request_id), image=self.image,
                          limits=self.settings.worker_limits, first_attempt=request.attempt_index == 1,
                          timeout_seconds=budget)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:1000]
            if self._workers.get(request_id):
                result = self._result(request_id, sha, "cleanup-unverified", failure_reason=reason,
                                      cleanup_confirmed=False, abort=True)
            else:
                result = self._result(request_id, sha, "environment-error", failure_reason=reason, cleanup_confirmed=True)
        else:
            result = self._outcome_result(request, sha, outcome)
        self._unfinished.discard(request_id)
        if result.abort_campaign:
            self.abort_campaign, self.abort_reason = True, result.failure_reason
        self._log("finished", request_id=request_id, status=result.status, failure_reason=result.failure_reason,
                  workers=list(self._workers.get(request_id, [])), elapsed_seconds=self.clock() - started,
                  trace_sha256=result.trace_sha256)
        return result

    def _outcome_result(self, request: CodingJobRequest, sha: str, outcome: dict) -> CodingJobResult:
        request_id, raw = request.request_id, outcome["raw_bytes"]
        sample = {**outcome["sample"], "request_id": request_id, "attempt_index": request.attempt_index}
        unverified = [f"{case.get('case_id')}={case.get('cleanup_error') or case.get('status')}"
                      for case in sample["cases"] if case.get("status") == "cleanup-unverified"]
        try:
            self.artifacts.write(f"traces/{request_id}.json", raw)
        except Exception as exc:
            reason = f"trace_not_persisted: {type(exc).__name__}: {exc}"[:1000]
            if outcome["abort_campaign"]:
                return self._result(request_id, sha, "cleanup-unverified", cleanup_confirmed=False, abort=True,
                                    failure_reason=reason)
            return self._result(request_id, sha, "environment-error", failure_reason=reason, cleanup_confirmed=True)
        trace = hashlib.sha256(raw).hexdigest()
        if outcome["abort_campaign"]:
            return self._result(request_id, sha, "cleanup-unverified", trace=trace, cleanup_confirmed=False, abort=True,
                                failure_reason=("worker cleanup unverified: " + "; ".join(unverified))[:1000])
        # A case whose Docker client never started ran nothing. The private-fixture runner still totals such a case
        # as a failed check of a "completed" sample; published as-is it would score a missing sandbox as the model's
        # 0. The whole request is infrastructure-failed instead (nothing to clean up, so no abort). Public items
        # already say environment-error; they carry the worker status per case and get the same named reason.
        unlaunched = [str(case.get("case_id")) for case in sample["cases"]
                      if "sandbox-unavailable" in (case.get("status"), case.get("worker_status"))]
        if unlaunched:
            return self._result(request_id, sha, "environment-error", trace=trace, cleanup_confirmed=True,
                                failure_reason=("sandbox_unavailable: the Docker client could not be started for "
                                                "case(s) " + ", ".join(unlaunched))[:1000])
        if sample.get("status") == "environment-error":
            reason = sample.get("failure_reason") or "public coding worker infrastructure failed"
            return self._result(request_id, sha, "environment-error", failure_reason=str(reason)[:1000],
                                trace=trace, cleanup_confirmed=True)
        return self._result(request_id, sha, "completed", sample=sample, trace=trace, cleanup_confirmed=True)

    # ---- publication ----------------------------------------------------------------------------------------
    def _publish(self, result: CodingJobResult, *, event: str = "published") -> bool:
        payload = result_bytes(result)
        try:
            publish_atomic(self.results_dir, request_name(result.request_id), payload)
        except FileExistsError:  # only this broker writes here: the request is terminal, never executed again
            self._terminal.setdefault(result.request_id, result.status)
            self._results_published += 1
            self._log("publish_skipped_existing", request_id=result.request_id, status=result.status)
            return False
        self._terminal[result.request_id] = result.status
        self._results[result.request_id] = result
        self._results_published += 1
        self._log(event, request_id=result.request_id, status=result.status,
                  result_sha256=hashlib.sha256(payload).hexdigest())
        return True

    def _reject(self, request_id: str, sha: str, reason: str) -> None:
        self._log("rejected", request_id=request_id, reason=reason)
        self._publish(self._result(request_id, sha, "rejected", failure_reason=reason, cleanup_confirmed=True))

    def _ignore(self, name: str, reason: str) -> None:
        if name not in self._ignored and len(self._ignored) < MAX_IGNORED_NAMES:
            self._ignored.add(name)
            self._log("ignored", name=name, reason=reason)

    def _content_sha_of(self, path: Path) -> str:
        """Hash of the validated request content when parseable, else of the raw bytes (bounded)."""
        try:
            raw = read_bounded(path, self.settings.max_request_bytes)
        except (OSError, ValueError):
            return hashlib.sha256(path.name.encode("utf-8")).hexdigest()
        try:
            return CodingJobRequest.model_validate(strict_json_loads(raw.decode("utf-8"))).content_sha256()
        except (ValueError, UnicodeDecodeError, ValidationError, RecursionError):
            return hashlib.sha256(raw).hexdigest()

    def _scan(self) -> list[tuple[str, tuple[int, int] | None]]:
        """Valid-named, non-ignored spool entries with their lstat signature, sorted; bounded in count."""
        names = []
        try:
            with os.scandir(self.requests_dir) as entries:
                for entry in entries:
                    if len(names) >= MAX_SCAN_ENTRIES:
                        break
                    if re.fullmatch(REQUEST_NAME, entry.name) and entry.name not in self._ignored:
                        names.append(entry.name)
        except OSError:
            return []
        rows = []
        for name in sorted(names):
            try:
                st = os.lstat(self.requests_dir / name)
                rows.append((name, (st.st_size, st.st_mtime_ns)))
            except OSError:
                rows.append((name, None))
        return rows

    def _pending(self, rows) -> int:
        return sum(name[:-5] not in self._terminal for name, _ in rows)

    # ---- protocol -------------------------------------------------------------------------------------------
    def tick(self, max_seconds: float) -> dict:
        """Handle at most one execution (plus bounded cheap rejections) and return within ``max_seconds``."""
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds):
            raise ValueError("max_seconds must be finite")
        if self.closed:
            raise RuntimeError("broker is closed")
        summary = {"processed": 0, "rejected": 0, "pending": 0, "abort_campaign": self.abort_campaign}
        if self.abort_campaign:
            summary.update(pending=self._pending(self._scan()), reason=self.abort_reason)
            return summary
        for name, signature in self._scan():
            if summary["processed"] >= 1 or summary["rejected"] >= MAX_REJECTIONS_PER_TICK:
                break
            path, stem = self.requests_dir / name, name[:-5]
            if signature is None or (stem in self._terminal and self._seen.get(name) == signature
                                     and (self.results_dir / name).exists()):
                continue  # finished, unchanged and its result is still published
            if self._results_published >= 2 * self.settings.max_requests:
                self._ignore(name, "result_cap")
                continue
            if self._seen.get(name) != signature:
                self._seen[name] = signature
                self._log("seen", name=name, size=signature[0])
            request, reason = self.validate(path, max_seconds=max_seconds)
            if reason in IGNORED_REASONS:
                self._ignore(name, reason)
            elif reason == "replay":
                self._log("replay", request_id=stem)
                if not (self.results_dir / name).exists():
                    stored = self._results.get(stem)
                    self._publish(stored or self._result(stem, request.content_sha256(), "rejected",
                                                         cleanup_confirmed=True, failure_reason="replay_without_result"),
                                  event="republished")
            elif reason == "replay_changed_content":
                self._log("rejected", request_id=stem, reason=reason)
                if not (self.results_dir / name).exists():
                    self._publish(self._result(stem, self._content_sha_of(path), "rejected", cleanup_confirmed=True,
                                               failure_reason=reason))
                summary["rejected"] += 1
            elif reason is not None:
                self._reject(stem, self._content_sha_of(path), reason)
                summary["rejected"] += 1
            else:
                self._accepted.add(request.request_id)
                self._content[request.request_id] = request.content_sha256()
                self._log("validated", request_id=request.request_id, content_sha256=self._content[request.request_id],
                          fixture_id=request.fixture_id, attempt_index=request.attempt_index)
                self._publish(self.execute(request, max_seconds=max_seconds))
                summary["processed"] += 1
                if self.abort_campaign:
                    break
        summary.update(pending=self._pending(self._scan()), abort_campaign=self.abort_campaign)
        if self.abort_campaign:
            summary["reason"] = self.abort_reason
        return summary

    def _verify_absent(self, names: list[str]) -> list[dict]:
        checks = []
        for name in names:
            row = {"name": name, "removed": False, "verified_absent": False}
            try:
                self.session_lock.check("container", RunMode.LIVE)
                removal = self.executor.run(("docker", "rm", "-f", name), timeout_seconds=10, max_output_bytes=4096)
                row["removed"] = removal.status == "completed"
                absent = self.executor.run(("docker", "ps", "--all", "--filter", f"name=^/{name}$", "--format", "{{.ID}}"),
                                           timeout_seconds=10, max_output_bytes=4096)
                row["verified_absent"] = (absent.status == "completed" and absent.returncode == 0
                                          and not absent.stdout.strip())
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            checks.append(row)
        return checks

    def reconcile(self) -> dict:
        """Remove and absence-verify every recorded-but-unfinished worker; publish ``interrupted`` results."""
        interrupted, checks = [], []
        for request_id in sorted(self._unfinished):
            rows = self._verify_absent(self._workers.get(request_id, []))
            checks.extend(rows)
            verified = all(row["verified_absent"] for row in rows)
            status = "interrupted" if verified else "cleanup-unverified"
            sha = self._content.get(request_id) or hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            reason = "broker was interrupted before the request finished"
            if not verified:  # the reason is ledgered with the result so a later generation inherits the cause
                reason += f"; interrupted worker of {request_id} not verified absent"
                self.abort_campaign, self.abort_reason = True, reason
            result = self._result(request_id, sha, status, cleanup_confirmed=verified, abort=not verified,
                                  failure_reason=reason)
            self._unfinished.discard(request_id)
            self._log("interrupted", request_id=request_id, status=status, workers=rows)
            self._log("finished", request_id=request_id, status=status, failure_reason=result.failure_reason,
                      workers=[row["name"] for row in rows], elapsed_seconds=0.0, trace_sha256=None)
            self._publish(result)
            interrupted.append(request_id)
        # Finished (cleanup already verified or already aborted) but never published: terminal, never re-executed.
        for request_id in sorted(set(self._finished) - set(self._terminal)):
            event = self._finished[request_id]
            sha = self._content.get(request_id) or hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            if event.get("status") == "cleanup-unverified":
                result = self._result(request_id, sha, "cleanup-unverified", cleanup_confirmed=False, abort=True,
                                      failure_reason=str(event.get("failure_reason") or self.abort_reason)[:2000])
            else:
                result = self._result(request_id, sha, "interrupted", cleanup_confirmed=True, failure_reason=(
                    f"broker was interrupted after finishing the request ({event.get('status')}) but before "
                    "publishing its result"))
            self._log("interrupted", request_id=request_id, status=result.status, workers=[],
                      reason="finished_without_published", finished_status=event.get("status"))
            self._publish(result)
            interrupted.append(request_id)
        summary = {"interrupted": len(interrupted), "workers_checked": len(checks),
                   "cleanup_verified": not self.abort_campaign and all(row["verified_absent"] for row in checks),
                   "abort_campaign": self.abort_campaign}
        self._log("reconcile", **summary)
        return summary

    def cancel_all(self) -> dict:
        """Publish ``cancelled`` for every pending request, verify owned workers are gone, close the ledger."""
        cancelled, checks = 0, []
        if self.closed:
            return {"cancelled": 0, "cleanup_verified": not self.abort_campaign, "workers_checked": 0}
        for name, _ in self._scan():
            stem = name[:-5]
            if stem in self._terminal:
                continue
            if self._publish(self._result(stem, self._content_sha_of(self.requests_dir / name), "cancelled",
                                          failure_reason="broker cancelled the request at cleanup",
                                          cleanup_confirmed=True)):
                self._log("cancelled", request_id=stem)
                cancelled += 1
        for request_id in sorted(self._unfinished):
            checks.extend(self._verify_absent(self._workers.get(request_id, [])))
        verified = not self.abort_campaign and all(row["verified_absent"] for row in checks)
        self._log("cancel_all", cancelled=cancelled, workers=checks, cleanup_verified=verified)
        self.close()
        return {"cancelled": cancelled, "cleanup_verified": verified, "workers_checked": len(checks)}

    def direct_client(self):
        from .broker_client import DirectClient
        return DirectClient(self, clock=self.clock)
