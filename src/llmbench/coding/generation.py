"""Private coding generation step: prompt, deterministic patch parsing, one optional repair round, broker hand-off.

The model never sees reference solutions or expected values. Raw responses are persisted before parsing.
An unparseable patch scores 0 and stays in the denominator. Every worker execution goes through the broker
(``SpoolClient`` inside the evaluator container, ``DirectClient`` in host-process mode); the evaluator never
runs Docker itself. An unverified worker cleanup or campaign abort from the broker raises
``UnsafeCaptureRuntimeState`` so the evaluator records ``abort_reason``.

Every row carries the tree's sample-origin pair (``synthetic``, ``model_evaluated``) that ``analysis._split_rows``
and ``reports`` require. Outcomes attributable to the model after a persisted real response (``invalid_patch``,
``broker_rejected`` for ``patch_too_large``/``patch_outside_allowlist``) are measured evidence: ``synthetic False,
model_evaluated True`` (a scored 0). Outcomes attributable to the harness (``timeout``, ``broker_timeout``,
``broker_integrity``, every other ``broker_*`` status or rejection reason, ``unknown_fixture``) carry
``model_evaluated False`` like ``container_eval._missing_rows``: the candidate is not judged on them, and they
never masquerade as a model 0. The completed path keeps the worker's own ``synthetic`` flag.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import ValidationError

from ..config import RunMode
from ..evaluations.capture import UnsafeCaptureRuntimeState
from ..evaluations.tools import strict_json_loads
from ..store import utc_now
from .broker_client import BrokerAborted
from .fixtures import CodingFixture, fixtures as default_fixtures
from .spool import CodingJobRequest, CodingJobResult, SpoolIntegrityError, request_bytes

_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)\r?\n?[ \t]*```", re.S)
_EXTENSIONS = {"python": ".py", "typescript": ".ts", "javascript": ".cjs"}
# Broker rejections the model is answerable for: its parsed patch was real but beyond what the broker admits.
MODEL_ATTRIBUTABLE_REJECTIONS = frozenset({"patch_too_large", "patch_outside_allowlist"})


class InvalidPatch(ValueError):
    """A parsed patch that the strict request contract refuses; scored 0 without any broker submission."""


@dataclass(frozen=True)
class PatchParse:
    patch: dict[str, str] | None
    method: Literal["json-object", "single-fence"] | None
    error: str | None


def entry_source(fixture: CodingFixture) -> str:
    """The editable file that the fixture's entrypoint lives in (same rule as ``runner.stage_case``)."""
    module = fixture.entrypoint.partition(":")[0]
    source = module + _EXTENSIONS[fixture.language]
    if source not in {item.path for item in fixture.initial_files}:
        raise ValueError("fixture entrypoint source is absent from its editable files")
    return source


def build_prompt(fixture: CodingFixture) -> list[dict]:
    """System + user messages carrying only the task statement and the current editable sources."""
    editable = [item.path for item in fixture.initial_files]
    system = ("You are solving a small private coding task. Return the complete replacement source of exactly the "
              "editable file(s) listed by the user: either a JSON object mapping each editable filename to its full "
              "new content, or exactly one fenced code block containing the full new content of "
              f"{entry_source(fixture)}. Do not add explanations, tests, or other files.")
    user = fixture.prompt + "\n\nEditable filenames: " + ", ".join(editable) + "\n"
    for item in fixture.initial_files:
        user += f"\n--- {item.path} (current content) ---\n{item.content}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_patch(content: Any, fixture: CodingFixture) -> PatchParse:
    """Deterministic: text starting with ``{`` must be a strict JSON object of editable file -> source; otherwise
    exactly one fenced block becomes the entry source. Anything else is an error, never repaired."""
    if type(content) is not str:
        return PatchParse(None, None, "no_text_content")
    text = content.strip()
    allowed = {item.path for item in fixture.initial_files}
    if text.startswith("{"):
        try:
            parsed = strict_json_loads(text)
        except (ValueError, RecursionError) as exc:
            return PatchParse(None, None, f"invalid_json: {str(exc)[:200]}")
        if (type(parsed) is not dict or not parsed
                or not all(type(key) is str and type(value) is str for key, value in parsed.items())):
            return PatchParse(None, None, "json_object_must_map_filenames_to_source_strings")
        if set(parsed) - allowed:
            return PatchParse(None, None, "files_outside_editable_list")
        return PatchParse(dict(parsed), "json-object", None)
    opens = text.count("```")
    blocks = _FENCE.findall(text)
    if opens == 0:
        return PatchParse(None, None, "no_patch_found")
    if opens != 2 or len(blocks) != 1:
        return PatchParse(None, None, "multiple_or_unterminated_fences")
    block = blocks[0]
    if not block.strip():
        return PatchParse(None, None, "empty_fence")
    return PatchParse({entry_source(fixture): block if block.endswith("\n") else block + "\n"}, "single-fence", None)


def repair_prompt(fixture: CodingFixture, previous: dict, failures: list[dict]) -> list[dict]:
    """Conversation for one repair round. Only ``case_id`` and ``status`` of each failure are disclosed."""
    if type(previous) is not dict or not all(type(k) is str and type(v) is str for k, v in previous.items()):
        raise ValueError("previous patch must map filenames to source strings")
    rows = []
    for item in failures:
        if not isinstance(item, dict) or "case_id" not in item or "status" not in item:
            raise ValueError("failures must carry case_id and status")
        rows.append({"case_id": str(item["case_id"]), "status": str(item["status"])})
    messages = build_prompt(fixture)
    messages.append({"role": "assistant", "content": json.dumps(previous, ensure_ascii=False)})
    messages.append({"role": "user", "content": "Your submission did not pass these checks: "
                     + json.dumps(rows, ensure_ascii=False)
                     + ". Reply with the complete corrected file(s) in the same format, nothing else."})
    return messages


def _dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=str, indent=1).encode("utf-8")


def _slug(fixture_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", fixture_id)


def run_coding_benchmarks(config, callback, remaining: Callable[[], float], artifacts, session_lock, *,
                          client=None, fixtures=default_fixtures, clock=time.monotonic) -> list[dict]:
    """Coding hook for ``container_eval`` ``client=None`` resolves the container spool client."""
    selections = [item for item in config.benchmarks if item.benchmark_id == "coding"]
    if not selections:
        return []
    if client is None:
        from .broker_client import SpoolClient
        client = SpoolClient(clock=clock)
    namespace = client.namespace()  # fails closed: without the host namespace no request can be admitted
    catalog = {item.fixture_id: item for item in fixtures()}
    generation = config.generation
    samples: list[dict] = []
    for selection in selections:
        for task_id in selection.task_ids:
            remaining()
            base = {"task_id": task_id, "category": "coding", "suite": "coding", "suite_revision": selection.revision,
                    "split": selection.split, "fixture_seed": selection.seed, "model_evaluated": True}
            fixture = catalog.get(task_id)
            if fixture is None:  # the model was never asked: a harness gap, not a measured 0
                samples.append({**base, "score": 0.0, "passed": False, "status": "completed",
                                "outcome_status": "unknown_fixture", "first_attempt_success": False,
                                "synthetic": False, "model_evaluated": False})
                continue
            samples.append(_evaluate_fixture(fixture, base, config, callback, remaining, artifacts, session_lock,
                                             client, namespace, generation))
    return samples


def _evaluate_fixture(fixture, base, config, callback, remaining, artifacts, session_lock, client, namespace,
                      generation) -> dict:
    base = {**base, "fixture_hash": fixture.identity(), "language": fixture.language}
    slug = _slug(fixture.fixture_id)
    evidence: dict[str, Any] = {"responses": [], "parse_methods": [], "parse_method": None, "attempts": 0,
                                "repair_attempted": False}

    def ask(messages: list[dict]) -> Any:
        session_lock.check("inference", RunMode.LIVE)
        request = {"messages": messages, "max_tokens": generation.max_output_tokens,
                   "temperature": generation.temperature, "seed": generation.seed, "top_p": generation.top_p}
        if generation.top_k is not None:
            request["top_k"] = generation.top_k
        response = asyncio.run(asyncio.wait_for(callback(request), timeout=remaining()))
        raw = _dump(response)
        evidence["attempts"] += 1
        try:
            artifacts.write(f"coding/{slug}/response-{evidence['attempts']}.json", raw)
        except Exception as exc:
            raise UnsafeCaptureRuntimeState(
                f"coding response evidence could not be persisted: {type(exc).__name__}: {exc}") from exc
        evidence["responses"].append(hashlib.sha256(raw).hexdigest())
        return response

    def parse(response: Any) -> PatchParse:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        parsed = parse_patch(content, fixture)
        evidence["parse_methods"].append(parsed.method or parsed.error)
        evidence["parse_method"] = parsed.method  # of the latest attempt; None when it did not parse
        return parsed

    def finish(sample: dict) -> dict:
        return {**sample, "generation": json.loads(json.dumps(evidence))}

    def failed(outcome: str, error: str | None = None, *, model_evaluated: bool, **extra) -> dict:
        """A scored-0 row with the origin pair: ``model_evaluated`` only for model-attributable outcomes."""
        row = {**base, "score": 0.0, "passed": False, "status": "completed", "outcome_status": outcome,
               "first_attempt_success": False, "synthetic": False, "model_evaluated": model_evaluated, **extra}
        if error is not None:
            row["error"] = error
        return finish(row)

    def submit(patch: dict[str, str], attempt_index: int) -> CodingJobResult:
        try:
            request = CodingJobRequest(session_id=namespace["session_id"], attempt_id=namespace["attempt_id"],
                                       request_id=uuid.uuid4().hex, fixture_id=fixture.fixture_id,
                                       fixture_revision=fixture.revision, fixture_hash=fixture.identity(),
                                       attempt_index=attempt_index, patch=patch, submitted_utc=utc_now())
        except ValidationError as exc:  # a parsed patch the strict contract still refuses (hard caps, NUL, ...)
            first = exc.errors()[0] if exc.errors() else {}
            raise InvalidPatch(f"schema: {first.get('msg', 'invalid request')}"[:200]) from exc
        settings = getattr(config, "broker", None)
        if settings is not None and len(request_bytes(request)) > settings.max_request_bytes:
            # The broker would answer an oversize file with a rejection the client cannot match (its hash is of the
            # name); the model's patch is real, so it is scored locally as an invalid patch instead.
            raise InvalidPatch(f"request_too_large: {len(request_bytes(request))} bytes exceed the "
                               f"{settings.max_request_bytes}-byte broker request cap")
        client.submit(request)
        try:
            result = client.wait(request.request_id, deadline=client.clock() + remaining())
        except BrokerAborted as exc:
            raise UnsafeCaptureRuntimeState(f"coding broker reported a campaign abort while {fixture.fixture_id} "
                                            f"was pending: {exc}") from exc
        if result.abort_campaign or result.status == "cleanup-unverified":
            raise UnsafeCaptureRuntimeState(f"coding broker reported unverified worker cleanup for "
                                            f"{fixture.fixture_id}: {result.failure_reason}")
        return result

    def broker_info(result: CodingJobResult, attempt_index: int) -> dict:
        return {"request_id": result.request_id, "attempt_index": attempt_index, "status": result.status,
                "trace_sha256": result.trace_sha256, "cleanup_confirmed": result.cleanup_confirmed}

    def sample_of(result: CodingJobResult, attempt_index: int) -> dict:
        broker = broker_info(result, attempt_index)
        if result.status != "completed":
            measured = result.status == "rejected" and result.failure_reason in MODEL_ATTRIBUTABLE_REJECTIONS
            return failed("broker_" + result.status.replace("-", "_"), result.failure_reason, broker=broker,
                          model_evaluated=measured)
        row = {**base, **dict(result.sample), "broker": broker}
        row["outcome_status"] = "passed" if row.get("passed") is True else "failed"
        row["first_attempt_success"] = attempt_index == 1 and row.get("passed") is True
        return row

    try:
        first = parse(ask(build_prompt(fixture)))
    except asyncio.TimeoutError as exc:
        return failed("timeout", f"TimeoutError: {exc}", model_evaluated=False)
    if first.patch is None:
        return failed("invalid_patch", first.error, model_evaluated=True)
    try:
        result = submit(first.patch, 1)
    except InvalidPatch as exc:
        return failed("invalid_patch", str(exc), model_evaluated=True)
    except TimeoutError as exc:
        return failed("broker_timeout", str(exc), model_evaluated=False)
    except SpoolIntegrityError as exc:
        return failed("broker_integrity", str(exc), model_evaluated=False)
    sample = sample_of(result, 1)
    if sample.get("passed") is True or result.status != "completed" or generation.repair_attempts < 1:
        return finish(sample)
    evidence["repair_attempted"] = True
    failures = [{"case_id": case.get("case_id"), "status": case.get("status")}
                for case in sample.get("cases", []) if case.get("passed") is not True]
    try:
        second = parse(ask(repair_prompt(fixture, first.patch, failures)))
    except asyncio.TimeoutError as exc:
        return finish({**sample, "first_attempt_success": False, "repair_error": f"TimeoutError: {exc}"})
    if second.patch is None:
        return finish({**sample, "first_attempt_success": False, "repair_error": second.error})
    try:
        repaired = submit(second.patch, 2)
    except (InvalidPatch, TimeoutError, SpoolIntegrityError) as exc:
        return finish({**sample, "first_attempt_success": False, "repair_error": str(exc)})
    if repaired.status != "completed":  # the measured first attempt stands; the failed repair is recorded beside it
        return finish({**sample, "first_attempt_success": False, "repair_error": repaired.failure_reason,
                       "repair_broker": broker_info(repaired, 2)})
    return finish(sample_of(repaired, 2))
