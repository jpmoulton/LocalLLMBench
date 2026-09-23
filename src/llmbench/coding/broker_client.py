"""Evaluator-side clients of the coding broker: file spool (container) and direct (host-process).

Both clients submit ``CodingJobRequest`` files atomically and only ever accept a bounded, well-formed
``CodingJobResult`` whose ``request_sha256`` matches the content they submitted. Malformed, oversize or
linked result files are ignored until the deadline; a well-formed result for other content is an integrity
error. ``DirectClient`` drives ``HostBroker.tick`` itself, so host-process mode uses the identical
validate-and-execute path through the private spool under the run directory. ``UnavailableClient`` is what a
host-process evaluation without a broker hands out instead of ``None``: it refuses every call.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from ..containers.config import NAME
from ..evaluations.tools import strict_json_loads
from .spool import (
    HEX32, NAMESPACE_NAME, CodingJobRequest, CodingJobResult, SpoolIntegrityError, publish_atomic, read_bounded,
    request_bytes, request_name,
)

DEFAULT_REQUESTS_DIR = "/spool/requests"
DEFAULT_RESULTS_DIR = "/spool/results"
DEFAULT_RESULT_BYTES = 4_194_304
SOLUTION_FILE = "solution.py"
EXECUTE_MARGIN_SECONDS = 120.0  # container start, result copy and verified removal, on top of the item's own budget


class BrokerAborted(RuntimeError):
    """The ticked broker reports a campaign abort while a request is pending: no result will ever follow."""


class ExecutionUnavailable(RuntimeError):
    """No isolated worker is reachable from this evaluator; generated code runs in the worker or nowhere."""


class UnavailableClient:
    """Stand-in client for a host-process evaluation that has no broker: every operation refuses, loudly.

    The coding hook reads ``client=None`` as "inside the evaluator container" and opens ``SpoolClient`` on
    ``/spool/...``. On the host that path is not a broker's spool: requests would land in a directory nobody
    validates (or another process owns) and the hook would wait out its whole budget for results that cannot
    come. Handing the hook this object instead makes the first call (``namespace()``) fail with the reason, so the
    coding rows stay in the denominator as harness errors and nothing is ever written or executed.
    """

    def __init__(self, reason: str, *, clock=time.monotonic) -> None:
        if type(reason) is not str or not reason.strip():
            raise ValueError("an unavailable client needs the reason execution is unavailable")
        self.reason, self.clock = reason, clock

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise ExecutionUnavailable(self.reason)

    namespace = submit = execute = poll = wait = _refuse


class SpoolClient:
    """Runs inside the evaluator: writes to ``/spool/requests`` and reads ``/spool/results`` (mounted read-only)."""

    def __init__(self, requests_dir: str | Path = DEFAULT_REQUESTS_DIR, results_dir: str | Path = DEFAULT_RESULTS_DIR,
                 *, clock=time.monotonic, sleep=time.sleep) -> None:
        self.requests_dir, self.results_dir = Path(requests_dir), Path(results_dir)
        self.clock, self.sleep = clock, sleep
        self._submitted: dict[str, str] = {}
        self._foreign: dict[str, str] = {}  # request_id -> the adapter-side content hash of a translated request

    def namespace(self) -> dict[str, str]:
        """The host broker publishes the attempt identity next to the results; requests must carry it."""
        raw = read_bounded(self.results_dir / NAMESPACE_NAME, 4096)
        parsed = strict_json_loads(raw.decode("utf-8"))
        if (type(parsed) is not dict or set(parsed) != {"schema_version", "session_id", "attempt_id"}
                or parsed["schema_version"] != 1 or type(parsed["session_id"]) is not str
                or type(parsed["attempt_id"]) is not str or not re.fullmatch(NAME, parsed["session_id"])
                or not re.fullmatch(HEX32, parsed["attempt_id"])):
            raise ValueError("spool namespace.json is not a valid broker namespace")
        return {"session_id": parsed["session_id"], "attempt_id": parsed["attempt_id"]}

    def submit(self, request: Any) -> str:
        """Submit a ``CodingJobRequest``, or an Aider Polyglot ``ExecutionRequest`` translated into one.

        The Polyglot adapter describes an attempt with the exercise's tests and support files attached. None of
        that crosses the spool: only the patch and the exercise's id, revision and pinned hash do, and the broker
        stages ITS OWN copy of the tests and refuses a hash it does not hold.
        """
        if not isinstance(request, CodingJobRequest):
            if getattr(request, "suite", None) != "aider-polyglot" or not callable(getattr(request, "content_sha256",
                                                                                           None)):
                raise TypeError("submit requires a CodingJobRequest or an aider-polyglot ExecutionRequest")
            foreign = request.content_sha256()
            request = CodingJobRequest(session_id=request.session_id, attempt_id=request.attempt_id,
                                       request_id=request.request_id, fixture_id=request.task_id,
                                       fixture_revision=request.suite_revision,
                                       fixture_hash=request.exercise_sha256, attempt_index=request.attempt_index,
                                       patch=dict(request.patch), submitted_utc=request.submitted_utc)
            self._foreign[request.request_id] = foreign
        publish_atomic(self.requests_dir, request_name(request.request_id), request_bytes(request))
        self._submitted[request.request_id] = request.content_sha256()
        return request.request_id

    def execute(self, request: Mapping[str, Any], *, margin_seconds: float = EXECUTE_MARGIN_SECONDS) -> CodingJobResult:
        """Run one public-benchmark item through the broker and return its result.

        A benchmark adapter describes an item as a plain mapping. Only the solution text and the item's identity
        cross the spool: the broker rebuilds the tests from the host's pinned dataset and refuses an id, revision
        or hash it does not hold, so whatever test text the adapter composed for its own records is never sent.
        """
        if not isinstance(request, Mapping) or request.get("kind") != "benchmark-item":
            raise ValueError("execute requires a benchmark-item request mapping")
        files = request.get("files")
        solution = files.get(SOLUTION_FILE) if isinstance(files, Mapping) else None
        timeout = request.get("timeout_seconds")
        if type(solution) is not str or type(timeout) is not int or timeout < 1:
            raise ValueError("a benchmark-item request needs files['solution.py'] and an integer timeout_seconds")
        job = CodingJobRequest(**self.namespace(), request_id=uuid.uuid4().hex, fixture_id=request.get("task_id"),
                               fixture_revision=request.get("benchmark_revision"),
                               fixture_hash=request.get("item_sha256"), attempt_index=1,
                               patch={SOLUTION_FILE: solution},
                               submitted_utc=datetime.now(timezone.utc).isoformat())
        self.submit(job)
        return self.wait(job.request_id, deadline=self.clock() + timeout + margin_seconds)

    def poll(self, request_id: str, *, max_bytes: int = DEFAULT_RESULT_BYTES) -> CodingJobResult | None:
        """One bounded look at the result file; ``None`` when absent, malformed, oversize or linked."""
        expected = self._submitted.get(request_id)
        if expected is None:
            raise ValueError("unknown request_id; submit it through this client first")
        try:
            raw = read_bounded(self.results_dir / request_name(request_id), max_bytes)
        except (OSError, ValueError):
            return None
        try:
            result = CodingJobResult.model_validate(strict_json_loads(raw.decode("utf-8")))
        except (ValueError, UnicodeDecodeError, ValidationError, RecursionError):
            return None
        if result.request_id != request_id:
            return None
        if result.request_sha256 != expected:
            raise SpoolIntegrityError(f"result for {request_id} does not match the submitted request content")
        if request_id in self._foreign:
            # Integrity was just verified against what really crossed the spool. The adapter checks the result
            # against ITS request's hash, so the verified result is re-keyed to that; nothing else changes.
            return {**result.model_dump(mode="json"), "request_sha256": self._foreign[request_id]}
        return result

    def wait(self, request_id: str, *, deadline: float, poll_seconds: float = 0.5,
             max_bytes: int = DEFAULT_RESULT_BYTES) -> CodingJobResult:
        while True:
            result = self.poll(request_id, max_bytes=max_bytes)
            if result is not None:
                return result
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError(f"no broker result for {request_id} before the deadline")
            self.sleep(min(poll_seconds, remaining))


class DirectClient(SpoolClient):
    """Host-process mode: same spool files and validation, with the broker ticked by the waiting caller."""

    def __init__(self, broker: Any, *, clock=time.monotonic, sleep=time.sleep) -> None:
        super().__init__(broker.requests_dir, broker.results_dir, clock=clock, sleep=sleep)
        self.broker = broker

    def namespace(self) -> dict[str, str]:
        return {"session_id": self.broker.session_id, "attempt_id": self.broker.attempt_id}

    def wait(self, request_id: str, *, deadline: float, poll_seconds: float = 0.5,
             max_bytes: int = DEFAULT_RESULT_BYTES) -> CodingJobResult:
        while True:
            result = self.poll(request_id, max_bytes=max_bytes)
            if result is not None:
                return result
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError(f"no broker result for {request_id} before the deadline")
            summary = self.broker.tick(remaining)
            if summary.get("abort_campaign"):
                result = self.poll(request_id, max_bytes=max_bytes)  # the abort may be this request's own result
                if result is not None:
                    return result
                raise BrokerAborted(f"coding broker aborted the campaign while {request_id} was pending: "
                                    f"{summary.get('reason')}")
            if summary["processed"] == 0 and summary["rejected"] == 0:
                self.sleep(min(poll_seconds, max(0.0, deadline - self.clock())))
