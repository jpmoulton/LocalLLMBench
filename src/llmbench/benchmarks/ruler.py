"""RULER long-context adapter: per-length rows with honest length and context accounting.

RULER (NVIDIA, Apache-2.0, github.com/NVIDIA/RULER) generates synthetic long-context items at any
requested token length and scores them by substring matching, with no judge. This adapter drives
the vendored generators in ``ruler_tasks`` against the candidate's private llama.cpp endpoint and
returns one row per declared task id, so a report can draw a per-length degradation curve and
derive an effective length.

What this adapter refuses to do:
  * It never claims a length it did not measure. The prompt is grown to the requested token count
    with the counting route the suite already trusts (llama.cpp ``/apply-template`` + ``/tokenize``,
    injected as ``options["token_counter"]``), and an item whose counted prompt lands outside the
    tolerance is reported as ``length_target_unmet`` WITHOUT being sent: a prompt of the wrong size
    is not a measurement at the requested size.
  * It never lets a context-accounting discrepancy silently destroy a correct retrieval result
    (LIVE-004/LIVE-008). Both axes are recorded: the substring score always, and separately whether
    the served prompt can be proven to be this fixture. A mismatch inside
    ``ACCOUNTING_SLACK_TOKENS`` with truncation explicitly denied keeps the retrieval outcome under
    the labelled status ``passed_context_unverified`` and still leaves
    ``eligible_for_context_claim`` false.
  * It never drops a declared task. Budget exhaustion, transport failure and generator failure all
    produce rows; unrun declared tasks come back through ``missing_rows``.
  * It never substitutes filler for a missing corpus: ``available()`` fails closed with the exact
    path the image should have baked.

Wiring contract (registry.py and container_eval.py own the wiring; this module only reads options):
  ``options["token_counter"]``  (required) ``(messages, tools) -> int | {"tokens": int, ...}``
  ``options["completion"]``     (required) ``(payload) -> response`` or an awaitable of one; the
                                callback owns the model alias, exactly as the coding hook's does.
  ``options["context_capacity"]``        served ``n_ctx``; lengths that do not fit are refused.
  ``options["ruler_tasks"]`` / ``options["ruler_lengths"]``  override the declared matrix.
  ``options["tokenizer_id"]`` / ``options["template_id"]`` / ``options["tokenizer_verified"]``
                                provenance for the counting route; without a verified counter no
                                row can ever be eligible for a context claim.
  ``options["ruler_output_tokens"]``     overrides upstream ``tokens_to_generate`` per task.
  ``options["length_tolerance_tokens"]`` / ``["length_tolerance_fraction"]``
  ``options["prefill_tokens_per_second"]`` / ``["decode_tokens_per_second"]`` /
  ``options["tokenize_tokens_per_second"]`` / ``["item_overhead_seconds"]``  budget model only.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..evaluations.retrieval import ACCOUNTING_SLACK_TOKENS, ContextAccounting
from ..safety import OperationForbidden
from . import BenchmarkAborted, BenchmarkContext, missing_rows
from .ruler_tasks import (
    DEFAULT_MAX_COUNTER_CALLS, DEFERRED_TASKS, TASK_SPECS, GeneratedItem, LengthUnreachable,
    corpus_gaps, digest, generate_task, score_prediction,
)

BENCHMARK_ID = "ruler"
REVISION = "ruler-vendored-v1"
CATEGORY = "retrieval"
DEFAULT_TASKS = ("niah_multikey_3", "niah_multiquery", "vt", "cwe")
DEFAULT_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
"""RULER's standard reporting ladder. A campaign selects the subset it can afford."""
EFFECTIVE_LENGTH_THRESHOLD = 0.85
"""NOT upstream's rule. Upstream (README at c3f5e3b4) calls a length "effective" when the 13-task
average clears an ABSOLUTE 85.6, i.e. Llama-2-7B's own 4K score; this is a RELATIVE rule - 85% of
this candidate's own shortest-length score - over a 4-task subset with one sample per point. It is
a useful within-candidate degradation summary and it is NOT the published "effective length".
See docs/benchmarks.md for which comparisons are legitimate."""
EMPTY_TOOL_SCHEMA_SHA256 = digest([])
_FAILED_STATUSES = frozenset({"environment_error", "timeout", "invalid_output"})


def task_id_for(task: str, length: int) -> str:
    """``ruler/<task>/<length>``: the task and the length are both part of the identity."""
    return f"{BENCHMARK_ID}/{task}/{length}"


def parse_task_id(task_id: str) -> tuple[str, int]:
    parts = str(task_id).split("/")
    if len(parts) != 3 or parts[0] != BENCHMARK_ID or not parts[2].isdigit() or int(parts[2]) < 1:
        raise ValueError(f"not a RULER task id (expected ruler/<task>/<length>): {task_id!r}")
    return parts[1], int(parts[2])


def _options(context: BenchmarkContext) -> Mapping[str, Any]:
    options = getattr(context, "options", None)
    return options if isinstance(options, Mapping) else {}


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(options: Mapping[str, Any], key: str, default: float) -> float:
    value = options.get(key, default)
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite positive number")
    return float(value)


def _declared_matrix(context: BenchmarkContext) -> tuple[tuple[str, int], ...]:
    """The (task, length) pairs for this run: the caller's task ids when given, else the matrix."""
    options = _options(context)
    if getattr(context, "task_ids", ()):
        pairs = [parse_task_id(task_id) for task_id in context.task_ids]
    else:
        tasks = tuple(options.get("ruler_tasks") or DEFAULT_TASKS)
        lengths = tuple(options.get("ruler_lengths") or DEFAULT_LENGTHS)
        for task in tasks:
            if type(task) is not str:
                raise ValueError("ruler_tasks must be task-name strings")
        pairs = [(task, _positive_int(length, "ruler_lengths entry"))
                 for length in lengths for task in tasks]
    # Ascending length: the cheap items run first, so a budget that runs out costs the fewest points.
    ordered = sorted(dict.fromkeys(pairs), key=lambda pair: (pair[1], pair[0]))
    if len(ordered) != len(pairs):
        raise ValueError("duplicate RULER task ids were declared")
    return tuple(ordered)


def _capacity(options: Mapping[str, Any], target: int, output_tokens: int) -> tuple[int, str]:
    capacity = options.get("context_capacity")
    if capacity is None:
        return target + output_tokens, "assumed"
    return _positive_int(capacity, "context_capacity"), "served"


def _tolerance(options: Mapping[str, Any], target: int) -> int:
    explicit = options.get("length_tolerance_tokens")
    if explicit is not None:
        return _positive_int(explicit, "length_tolerance_tokens")
    fraction = _positive_float(options, "length_tolerance_fraction", 0.01)
    return max(64, int(math.ceil(target * fraction)))


def estimated_item_seconds(target: int, output_tokens: int, options: Mapping[str, Any]) -> float:
    """Wall cost of one item: the counting search, the prefill, and the decode.

    A 128K prompt is roughly 85 s of prefill on this hardware, and the length search tokenizes a
    growing prompt several times before that. The estimate only has to be honest enough to stop the
    adapter from starting an item it cannot finish.
    """
    prefill = _positive_float(options, "prefill_tokens_per_second", 1500.0)
    decode = _positive_float(options, "decode_tokens_per_second", 20.0)
    tokenize = _positive_float(options, "tokenize_tokens_per_second", 250_000.0)
    overhead = _positive_float(options, "item_overhead_seconds", 5.0)
    calls = _positive_int(options.get("max_counter_calls", DEFAULT_MAX_COUNTER_CALLS), "max_counter_calls")
    return target / prefill + output_tokens / decode + calls * target / tokenize + overhead


def _remaining(context: BenchmarkContext) -> tuple[float, str | None]:
    probe = getattr(context, "remaining_seconds", None)
    if not callable(probe):
        return 0.0, "the benchmark context exposes no remaining_seconds() budget probe"
    try:
        value = probe()
    except TimeoutError as exc:
        return 0.0, f"the evaluation wall budget is exhausted: {exc}"
    except Exception as exc:
        return 0.0, f"the wall budget could not be read: {type(exc).__name__}: {exc}"
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        return 0.0, "remaining_seconds() did not return a finite number"
    return float(value), None


def _is_abort(exc: BaseException) -> bool:
    """Unsafe runtime state propagates; an ordinary transport failure becomes a row."""
    return (isinstance(exc, (BenchmarkAborted, OperationForbidden))
            or bool(getattr(exc, "abort_campaign", False)))


def _dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=str, indent=1).encode("utf-8")


def _slug(task_id: str) -> str:
    return task_id.replace("/", "-")


def _dispatch(completion: Callable[[dict[str, Any]], Any], payload: dict[str, Any],
              timeout: float) -> Any:
    """Call the injected completion route, awaiting it when it is the async capture callback."""
    result = completion(payload)
    if inspect.isawaitable(result):
        return asyncio.run(asyncio.wait_for(result, timeout=max(1.0, timeout)))
    return result


def _response_text(response: Any) -> str | None:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if type(content) is str else None


def _count_field(value: Any) -> int | None:
    return value if type(value) is int and not isinstance(value, bool) and value >= 0 else None


def _runtime_evidence(response: Any) -> dict[str, Any]:
    """Only positive evidence is asserted; anything absent stays unknown, never guessed."""
    if not isinstance(response, Mapping):
        return {"observed_input_tokens": None, "truncation_reported": None, "context_policy": "unknown",
                "finish_reason": None, "served_model": None, "output_tokens": None}
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    truncated = response.get("input_truncated")
    policy = response.get("llmbench_context_policy")
    choices = response.get("choices")
    finish = None
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        finish = choices[0].get("finish_reason")
    return {"observed_input_tokens": _count_field(usage.get("prompt_tokens")),
            "truncation_reported": truncated if type(truncated) is bool else None,
            "context_policy": policy if policy == "reject-overflow-no-shift" else "unknown",
            "finish_reason": finish if type(finish) is str else None,
            "served_model": response.get("model") if type(response.get("model")) is str else None,
            "output_tokens": _count_field(usage.get("completion_tokens"))}


class RulerAdapter:
    """One row per declared ``ruler/<task>/<length>`` id, failures included."""

    benchmark_id = BENCHMARK_ID
    revision = REVISION
    category = CATEGORY

    def __init__(self, *, tasks: Sequence[str] = DEFAULT_TASKS,
                 lengths: Sequence[int] = DEFAULT_LENGTHS) -> None:
        self.tasks, self.lengths = tuple(tasks), tuple(lengths)

    # ---- preflight -------------------------------------------------------------------------

    def available(self, context: BenchmarkContext) -> tuple[bool, str]:
        """Is every corpus the declared tasks need baked into this image? Never raises."""
        try:
            matrix = self._matrix(context)
            names = sorted({task for task, _ in matrix})
            deferred = [name for name in names if name not in TASK_SPECS]
            if deferred:
                reasons = "; ".join(f"{name}: {DEFERRED_TASKS.get(name, 'unknown RULER task')}"
                                    for name in deferred)
                return False, f"RULER cannot run the declared tasks - {reasons}"
            root = str(getattr(context, "dataset_root", "") or "")
            if not root:
                return False, "RULER has no dataset root to look for its haystack corpora in"
            gaps = corpus_gaps(root, names)
            if gaps:
                return False, "RULER corpus missing - " + "; ".join(gap.detail for gap in gaps)
            return True, (f"RULER {REVISION}: {len(matrix)} items over tasks {', '.join(names)}; "
                          f"corpora present under {root}")
        except Exception as exc:
            return False, f"RULER preflight failed: {type(exc).__name__}: {exc}"

    def task_ids(self, context: BenchmarkContext) -> tuple[str, ...]:
        return tuple(task_id_for(task, length) for task, length in self._matrix(context))

    def _matrix(self, context: BenchmarkContext) -> tuple[tuple[str, int], ...]:
        options = _options(context)
        if getattr(context, "task_ids", ()) or options.get("ruler_tasks") or options.get("ruler_lengths"):
            return _declared_matrix(context)
        return tuple(sorted(((task, length) for length in self.lengths for task in self.tasks),
                            key=lambda pair: (pair[1], pair[0])))

    # ---- execution -------------------------------------------------------------------------

    def run(self, context: BenchmarkContext) -> list[dict[str, Any]]:
        options = _options(context)
        # A declaration this adapter cannot even parse is a wiring error, not a measurement: it
        # raises here rather than returning a short list that would quietly shrink the denominator.
        matrix = self._matrix(context)
        declared = [task_id_for(task, length) for task, length in matrix]
        rows: list[dict[str, Any]] = []
        counter, completion = options.get("token_counter"), options.get("completion")
        artifacts = getattr(context, "artifacts", None)
        reason = None
        if not callable(counter) or not callable(completion):
            reason = ("RULER was not wired: options must carry a callable token_counter and a callable "
                      "completion route")
        elif artifacts is None or not callable(getattr(artifacts, "write", None)):
            reason = "RULER has no artifacts writer, so no prompt or response evidence could be kept"
        if reason is not None:
            return self._fill(declared, rows, context, "environment_error", reason)
        stop_status, stop_reason = "timeout", "the evaluation wall budget ended before this item"
        for task, length in matrix:
            identifier = task_id_for(task, length)
            spec = TASK_SPECS.get(task)
            if spec is None:
                rows.append(self._row(context, identifier, task, length, status="environment_error",
                                      outcome_status="task_not_vendored",
                                      error=DEFERRED_TASKS.get(task, f"unknown RULER task {task!r}")))
                continue
            output_tokens = self._output_tokens(options, spec)
            remaining, budget_error = _remaining(context)
            needed = estimated_item_seconds(length, output_tokens, options)
            if budget_error is not None or remaining <= needed:
                stop_status = "environment_error" if budget_error is not None else "timeout"
                stop_reason = budget_error or (
                    f"stopped before {identifier}: {remaining:.1f}s left, about {needed:.1f}s needed")
                break
            try:
                rows.append(self._item(context, identifier, task, length, spec, options, counter,
                                       completion, artifacts, min(remaining, needed * 4)))
            except BaseException as exc:
                # Unsafe runtime state stops everything; any other failure is this item's row, so a
                # defect in one item can never shrink the denominator for the rest.
                if _is_abort(exc) or not isinstance(exc, Exception):
                    raise
                rows.append(self._row(context, identifier, task, length, status="environment_error",
                                      outcome_status="adapter_error",
                                      error=f"{type(exc).__name__}: {exc}"))
        return self._fill(declared, rows, context, stop_status, stop_reason)

    def _output_tokens(self, options: Mapping[str, Any], spec: Any) -> int:
        override = options.get("ruler_output_tokens")
        return spec.max_output_tokens if override is None else _positive_int(override, "ruler_output_tokens")

    def _fill(self, declared: Sequence[str], rows: list[dict[str, Any]], context: BenchmarkContext,
              status: str, reason: str) -> list[dict[str, Any]]:
        """Declared-but-unrun ids become rows, keeping their length so the curve stays honest."""
        filled = missing_rows(declared, rows, suite=BENCHMARK_ID, revision=REVISION, category=CATEGORY,
                              split=getattr(context, "split", "development"), status=status, reason=reason)
        for row in filled:
            task, length = parse_task_id(row["task_id"])
            row.update({"ruler_task": task, "ruler_family": getattr(TASK_SPECS.get(task), "family", None),
                        "target_input_tokens": length, "actual_input_tokens": None,
                        "length_delta_tokens": None, "length_within_tolerance": False,
                        "eligible_for_context_claim": False, "fixture_seed": getattr(context, "seed", None)})
        by_id = {row["task_id"]: row for row in rows + filled}
        return [by_id[identifier] for identifier in declared]

    def _row(self, ctx: BenchmarkContext, task_id: str, task: str, length: int, *, status: str,
             outcome_status: str, score: float = 0.0, passed: bool = False,
             model_evaluated: bool = False, **extra: Any) -> dict[str, Any]:
        """``extra`` carries the row's own keys, including ``context`` - hence the shortened name."""
        row = {"task_id": task_id, "suite": BENCHMARK_ID, "suite_revision": REVISION, "category": CATEGORY,
               "split": getattr(ctx, "split", "development"), "status": status, "score": float(score),
               "passed": bool(passed), "model_evaluated": bool(model_evaluated), "synthetic": False,
               "outcome_status": outcome_status, "ruler_task": task,
               "ruler_family": getattr(TASK_SPECS.get(task), "family", None),
               "ruler_metric": getattr(TASK_SPECS.get(task), "metric", None),
               "target_input_tokens": length, "actual_input_tokens": None, "length_delta_tokens": None,
               "length_within_tolerance": False, "eligible_for_context_claim": False,
               "fixture_seed": getattr(ctx, "seed", None)}
        row.update(extra)
        return row

    def item_seed(self, context: BenchmarkContext, task: str, length: int) -> int:
        """Reproducible per item, and disjoint between development and holdout by construction."""
        material = f"{REVISION}:{getattr(context, 'split', 'development')}:{getattr(context, 'seed', 0)}:" \
                   f"{task}:{length}"
        return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)

    def _item(self, context: BenchmarkContext, task_id: str, task: str, length: int, spec: Any,
              options: Mapping[str, Any], counter: Callable[..., Any], completion: Callable[..., Any],
              artifacts: Any, timeout: float) -> dict[str, Any]:
        seed = self.item_seed(context, task, length)
        output_tokens = self._output_tokens(options, spec)
        capacity, capacity_source = _capacity(options, length, output_tokens)
        tolerance = _tolerance(options, length)
        slug = _slug(task_id)
        base = {"fixture_seed": getattr(context, "seed", None), "item_seed": seed,
                "ruler_output_tokens": output_tokens, "length_tolerance_tokens": tolerance,
                "context_capacity": capacity, "context_capacity_source": capacity_source}
        if length + output_tokens > capacity:
            return self._row(context, task_id, task, length, status="environment_error",
                             outcome_status="length_exceeds_context_window", **base,
                             error=f"a {length}-token prompt plus {output_tokens} output tokens does not fit "
                                   f"the served {capacity}-token window")
        try:
            item = generate_task(task, target_tokens=length, counter=counter, seed=seed,
                                 dataset_root=str(getattr(context, "dataset_root", "") or "") or None,
                                 max_counter_calls=_positive_int(
                                     options.get("max_counter_calls", DEFAULT_MAX_COUNTER_CALLS),
                                     "max_counter_calls"))
        except BaseException as exc:
            if _is_abort(exc) or not isinstance(exc, Exception):
                raise
            status = "environment_error"
            outcome = "generator_length_unreachable" if isinstance(exc, LengthUnreachable) else "generator_failed"
            return self._row(context, task_id, task, length, status=status, outcome_status=outcome, **base,
                             error=f"{type(exc).__name__}: {exc}")
        base.update({"actual_input_tokens": item.actual_tokens,
                     "length_delta_tokens": item.actual_tokens - length,
                     "length_within_tolerance": abs(item.actual_tokens - length) <= tolerance,
                     "counter_calls": item.counter_calls, "prompt_sha256": item.prompt_sha256,
                     "fixture_hash": item.prompt_sha256, "expected_count": len(item.answers),
                     "ruler_metadata": dict(item.metadata)})
        self._persist(artifacts, f"ruler/{slug}/prompt.json", item.summary())
        if not base["length_within_tolerance"]:
            # Refused before any request: a prompt of the wrong size cannot be a data point at this size.
            return self._row(context, task_id, task, length, status="environment_error",
                             outcome_status="length_target_unmet", **base,
                             error=f"the generated prompt counted {item.actual_tokens} tokens against a "
                                   f"{length}-token target, outside the {tolerance}-token tolerance")
        payload = self._payload(context, item, output_tokens)
        started = time.monotonic()
        try:
            response = _dispatch(completion, payload, timeout)
        except BaseException as exc:
            if _is_abort(exc) or not isinstance(exc, Exception):
                raise
            timed_out = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            self._persist(artifacts, f"ruler/{slug}/response.json",
                          {"llmbench_error": f"{type(exc).__name__}: {exc}"})
            return self._row(context, task_id, task, length, status="timeout" if timed_out else "environment_error",
                             outcome_status="transport_timeout" if timed_out else "transport_error", **base,
                             error=f"{type(exc).__name__}: {exc}",
                             elapsed_seconds=time.monotonic() - started)
        self._persist(artifacts, f"ruler/{slug}/response.json", response)  # raw, before any parsing
        base["elapsed_seconds"] = time.monotonic() - started
        return self._score(context, task_id, task, length, item, response, options, base)

    def _payload(self, context: BenchmarkContext, item: GeneratedItem, output_tokens: int) -> dict[str, Any]:
        generation = getattr(context, "generation", None)
        payload: dict[str, Any] = {"messages": [dict(message) for message in item.messages],
                                   "max_tokens": output_tokens}
        for key in ("temperature", "top_p", "seed", "top_k"):
            value = getattr(generation, key, None)
            if value is not None:
                payload[key] = value
        return payload

    def _persist(self, artifacts: Any, relative: str, value: Any) -> None:
        try:
            artifacts.write(relative, _dump(value))
        except Exception as exc:
            raise BenchmarkAborted(
                f"RULER evidence could not be persisted to {relative}: {type(exc).__name__}: {exc}") from exc

    def _score(self, context: BenchmarkContext, task_id: str, task: str, length: int, item: GeneratedItem,
               response: Any, options: Mapping[str, Any], base: dict[str, Any]) -> dict[str, Any]:
        evidence = _runtime_evidence(response)
        alias = getattr(context, "model_alias", None)
        served = evidence["served_model"]
        if alias and served is not None and served != alias:
            return self._row(context, task_id, task, length, status="environment_error",
                             outcome_status="served_model_mismatch", **base,
                             error=f"the endpoint served {served!r} while this campaign evaluates {alias!r}")
        text = _response_text(response)
        accounting = ContextAccounting(
            requested_input_tokens=length, actual_input_tokens=item.actual_tokens,
            context_capacity=base["context_capacity"], reserved_output_tokens=base["ruler_output_tokens"],
            counting_method="exact-post-template" if options.get("tokenizer_verified") is True else "estimated",
            tokenizer_id=str(options.get("tokenizer_id") or "unverified-counter"),
            template_id=str(options.get("template_id") or "unverified-template"),
            tool_schema_included=False, prompt_sha256=item.prompt_sha256,
            tool_schema_sha256=EMPTY_TOOL_SCHEMA_SHA256)
        try:
            context_evidence = accounting.verify_runtime(
                evidence["observed_input_tokens"], truncation_reported=evidence["truncation_reported"],
                context_policy=evidence["context_policy"])
        except ValueError as exc:
            return self._row(context, task_id, task, length, status="invalid_output",
                             outcome_status="invalid_context_evidence", **base,
                             error=f"unusable runtime context evidence: {exc}")
        observed = context_evidence["observed_input_tokens"]
        mismatch = context_evidence["observed_count_matches"] is False
        # LIVE-008: the retrieval axis only dies when the served prompt cannot be shown to be this
        # fixture. A small disagreement with truncation explicitly denied is an accounting fact, and
        # it still blocks the context claim below.
        accounting_only = (mismatch and evidence["truncation_reported"] is False and observed is not None
                           and observed + base["ruler_output_tokens"] <= base["context_capacity"]
                           and abs(observed - item.actual_tokens) <= ACCOUNTING_SLACK_TOKENS)
        invalid_context = evidence["truncation_reported"] is True or (mismatch and not accounting_only)
        if text is None:
            return self._row(context, task_id, task, length, status="invalid_output",
                             outcome_status="no_assistant_content", model_evaluated=True, **base,
                             context=context_evidence, context_accounting_mismatch=mismatch,
                             context_accounting_only=accounting_only,
                             error="the response carried no assistant message content",
                             finish_reason=evidence["finish_reason"], output_tokens=evidence["output_tokens"])
        strict = score_prediction(text, item.answers, metric=item.metric)
        lenient = score_prediction(text, item.answers, metric=item.metric, lenient=True)
        outcome = strict.correct
        passed = outcome and not invalid_context
        outcome_status = ("invalid_context" if invalid_context
                          else "passed_context_unverified" if passed and mismatch
                          else "passed" if passed
                          else "empty_output" if strict.prediction_empty
                          else "partial" if strict.score > 0 else "incorrect")
        raw_limit = _positive_int(options.get("max_recorded_response_characters", 4096),
                                  "max_recorded_response_characters")
        return self._row(
            context, task_id, task, length, status="completed", outcome_status=outcome_status,
            score=0.0 if invalid_context else strict.score, passed=passed, model_evaluated=True, **base,
            string_match_all=strict.string_match_all, string_match_part=strict.string_match_part,
            retrieval_fraction=strict.string_match_all, score_before_context_check=strict.score,
            matched_count=len(strict.matched), per_expected=list(strict.per_expected),
            outcome_correct=outcome, format_strict_ok=not strict.prediction_empty,
            lenient_score=lenient.score, lenient_string_match_all=lenient.string_match_all,
            lenient_per_expected=list(lenient.per_expected), lenient_outcome_correct=lenient.correct,
            lenient_parse_used=lenient.score > strict.score,
            context=context_evidence, context_accounting_mismatch=mismatch,
            context_accounting_only=accounting_only,
            eligible_for_context_claim=passed and context_evidence["actual_context_verified"],
            finish_reason=evidence["finish_reason"], output_tokens=evidence["output_tokens"],
            output_cap_hit=evidence["finish_reason"] == "length",
            response_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            raw_response=text[:raw_limit], raw_response_truncated=len(text) > raw_limit)


def adapter(**options: Any) -> RulerAdapter:
    return RulerAdapter(**options)


# ---- reporting helpers (pure; the report module owns presentation) -----------------------------

def length_curve(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean score per declared length, with the evidence a reader needs to distrust a point."""
    buckets: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("suite") != BENCHMARK_ID:
            continue
        length = row.get("target_input_tokens")
        if type(length) is int:
            buckets.setdefault(length, []).append(row)
    curve = []
    for length in sorted(buckets):
        items = buckets[length]
        scored = [row for row in items if row.get("status") == "completed"]
        curve.append({
            "target_input_tokens": length, "declared": len(items), "completed": len(scored),
            "complete": len(scored) == len(items),
            "mean_score": (sum(float(row.get("score") or 0.0) for row in scored) / len(scored)
                           if scored else None),
            "tasks": sorted({str(row.get("ruler_task")) for row in items}),
            "context_verified": bool(scored) and all(row.get("eligible_for_context_claim") is True
                                                     for row in scored),
            "failed": sorted({str(row.get("outcome_status")) for row in items
                              if row.get("status") in _FAILED_STATUSES}),
        })
    return curve


def effective_length(rows: Iterable[Mapping[str, Any]], *,
                     threshold: float = EFFECTIVE_LENGTH_THRESHOLD) -> dict[str, Any]:
    """Longest length still holding at ``threshold`` of this candidate's shortest-length score.

    This is a within-candidate degradation summary, **not** RULER's published "effective length",
    which is an absolute 85.6 on the 13-task average - see ``EFFECTIVE_LENGTH_THRESHOLD``.

    Fail-closed: the walk stops at the first length whose declared items did not all complete, so a
    length that was never fully measured can never be reported as supported, and the returned claim
    is marked unverified unless every contributing row was eligible for a context claim.
    """
    if type(threshold) not in (int, float) or isinstance(threshold, bool) or not 0 < threshold <= 1:
        raise ValueError("threshold must be a fraction in (0, 1]")
    curve = length_curve(rows)
    if not curve or curve[0]["mean_score"] is None or not curve[0]["complete"]:
        return {"effective_length_tokens": None, "baseline_length_tokens": None, "baseline_score": None,
                "threshold": threshold, "context_verified": False, "curve": curve,
                "reason": "the shortest declared length did not complete, so there is no baseline"}
    baseline = curve[0]["mean_score"]
    floor = baseline * threshold
    effective, verified, reason = curve[0]["target_input_tokens"], curve[0]["context_verified"], "held"
    for point in curve[1:]:
        if not point["complete"] or point["mean_score"] is None:
            reason = f"stopped at {point['target_input_tokens']}: not every declared item completed"
            break
        if point["mean_score"] < floor:
            reason = f"fell below {floor:.3f} at {point['target_input_tokens']}"
            break
        effective, verified = point["target_input_tokens"], verified and point["context_verified"]
    return {"effective_length_tokens": effective, "baseline_length_tokens": curve[0]["target_input_tokens"],
            "baseline_score": baseline, "threshold": threshold, "score_floor": floor,
            "context_verified": verified, "curve": curve, "reason": reason}
