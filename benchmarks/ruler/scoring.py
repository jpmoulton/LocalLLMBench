"""Dependency-free RULER recall and coverage-aware result interpretation.

Scoring protocol: NVIDIA/RULER at revision
db43f92ff634bbce1c33d1fb15858aec4bba509a (Apache-2.0):
https://github.com/NVIDIA/RULER/blob/db43f92ff634bbce1c33d1fb15858aec4bba509a/scripts/eval/synthetic/constants.py
https://github.com/NVIDIA/RULER/blob/db43f92ff634bbce1c33d1fb15858aec4bba509a/scripts/eval/evaluate.py
This is an independent implementation of its case-insensitive substring recall,
not an exact-match metric or a claim of full RULER/paper-score equivalence.
"""

import math
from typing import Any

from .definition import TASKS


SCORER_NAME = "ruler_recall"
TRUNCATION_REASONS = {"max_tokens", "model_length"}
_CONTROL_CHARACTERS = str.maketrans({chr(value): "\n" for value in range(32)})
SCORED_STATUSES = ("pass", "partial", "fail")
SAMPLE_STATUSES = (*SCORED_STATUSES, "error", "truncation", "missing")


def normalize_prediction(prediction: str) -> str:
    """Apply the official strip/control-character normalization, then lowercase."""
    return prediction.strip().translate(_CONTROL_CHARACTERS).strip().lower()


def substring_recall(prediction: str, references: list[str]) -> float:
    """Return fractional reference recall, counting duplicate references separately."""
    if not isinstance(prediction, str):
        raise ValueError("Prediction must be a string")
    if not isinstance(references, list) or not references or any(
        not isinstance(reference, str) or not reference.strip() for reference in references
    ):
        raise ValueError("References must be a nonempty list of nonempty strings")
    normalized = normalize_prediction(prediction)
    return sum(reference.lower() in normalized for reference in references) / len(references)


def output_finish_reason(output: Any) -> str | None:
    choices = getattr(output, "choices", None)
    return getattr(choices[0], "stop_reason", None) if choices else None


def score_output(output: Any, references: list[str]) -> tuple[str, float]:
    """Score only Inspect's final completion; never concatenate reasoning blocks."""
    reason = output_finish_reason(output)
    if reason in TRUNCATION_REASONS:
        raise RuntimeError(f"Generation truncated ({reason}); no score assigned.")
    if getattr(output, "error", None):
        raise RuntimeError(f"Model generation failed: {output.error}")
    if reason not in {"stop", "content_filter"}:
        raise RuntimeError(f"Missing or unexpected generation finish reason: {reason}")
    # Inspect's completion contains text content, not separately represented
    # ContentReasoning/reasoning_content. Do not parse or concatenate raw messages.
    answer = getattr(output, "completion", None)
    if not isinstance(answer, str):
        raise RuntimeError("Model returned no final completion text.")
    return answer, substring_recall(answer, references)


def _json_evidence(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_evidence(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _json_evidence(value.model_dump(mode="json"))


def sample_summary(sample: Any, task: str) -> dict:
    output = getattr(sample, "output", None)
    reason = output_finish_reason(output)
    scores = getattr(sample, "scores", None) or {}
    values = {name: score.value for name, score in scores.items()}
    recall: Any = values.get(SCORER_NAME)
    error = _json_evidence(getattr(sample, "error", None))
    model_error = getattr(output, "error", None)
    limit = _json_evidence(getattr(sample, "limit", None))
    if reason in TRUNCATION_REASONS:
        status = "truncation"
    elif error is not None or model_error or limit is not None:
        status = "error"
    elif (
        reason not in {"stop", "content_filter"}
        or set(values) != {SCORER_NAME}
        or isinstance(recall, bool)
        or not isinstance(recall, (int, float))
        or not math.isfinite(recall)
        or not 0 <= recall <= 1
    ):
        status = "error"
        error = {"message": "Sample has no unambiguous finite final recall score or valid finish reason."}
    else:
        status = "pass" if recall == 1 else "fail" if recall == 0 else "partial"
    return {
        "id": str(sample.id),
        "inspect_id": sample.id,
        "task": task,
        "status": status,
        "recall": recall if status in SCORED_STATUSES else None,
        "finish_reason": reason,
        "finish_reason_format": "Inspect normalized stop reason (max_tokens corresponds to API length)",
        "token_usage": _json_evidence(getattr(output, "usage", None)),
        "scores": _json_evidence(values),
        "error": error,
        "model_error": model_error,
        "limit": limit,
    }


def _counts(samples: list[dict]) -> dict:
    return {status: sum(sample["status"] == status for sample in samples) for status in SAMPLE_STATUSES}


def summarize(logs: list, provenance: dict, run_error: dict | None) -> dict:
    """Withhold every aggregate unless the entire frozen selection is covered."""
    selected_ids = provenance["selected_ids"]
    tasks = provenance["tasks"]
    sample_tasks = provenance["sample_tasks"]
    expected = set(selected_ids)
    samples = {}
    log_errors = []
    for log in logs:
        if log.status != "success" or getattr(log, "error", None) is not None:
            log_errors.append({
                "log": log.location,
                "status": log.status,
                "error": _json_evidence(getattr(log, "error", None)),
            })
        for sample in log.samples or []:
            sample_id = str(sample.id)
            if sample_id not in expected or sample_id in samples:
                log_errors.append({
                    "log": log.location,
                    "message": f"Unexpected or duplicate sample ID: {sample.id}",
                })
                continue
            task = sample_tasks[sample_id]
            summary = sample_summary(sample, task)
            metadata = getattr(sample, "metadata", None) or {}
            if metadata.get("task") != task:
                log_errors.append({
                    "log": log.location,
                    "message": f"Sample task metadata does not match frozen selection: {sample_id}",
                })
            samples[sample_id] = summary
    ordered = [
        samples.get(sample_id, {
            "id": sample_id, "task": sample_tasks[sample_id], "status": "missing", "recall": None,
        })
        for sample_id in selected_ids
    ]
    counts = _counts(ordered)
    scored = sum(counts[status] for status in SCORED_STATUSES)
    complete = (
        bool(selected_ids) and bool(logs) and run_error is None and not log_errors
        and scored == len(selected_ids)
    )
    per_task = {}
    for task in tasks:
        outcomes = [sample for sample in ordered if sample["task"] == task]
        task_counts = _counts(outcomes)
        per_task[task] = {
            "denominator": len(outcomes),
            "scored": sum(task_counts[status] for status in SCORED_STATUSES),
            "counts": task_counts,
            "score": 100 * sum(sample["recall"] for sample in outcomes) / len(outcomes)
            if complete else None,
        }
    return {
        "benchmark": "RULER retrieval subset",
        "protocol": provenance["protocol"],
        "status": "completed" if complete else "incomplete",
        "denominator": len(selected_ids),
        "scored": scored,
        "counts": counts,
        "retrieval_score": sum(result["score"] for result in per_task.values()) / len(tasks)
        if complete else None,
        **{f"{task}_score": per_task.get(task, {}).get("score") for task in TASKS},
        "per_task": per_task,
        "score_policy": "Percent case-insensitive substring reference recall; equal-weight mean of selected task means. All aggregate scores withheld unless every selected sample is scored without errors or truncation.",
        "reference_length_policy": "Frozen input budget measured with cl100k_base, not the tested model tokenizer; input includes the answer prefix.",
        "input_tokens": provenance["input_tokens"],
        "reserved_output_tokens": provenance["reserved_output_tokens"],
        "inspect_logs": [log.location for log in logs],
        "inspect_log_errors": log_errors,
        "run_error": run_error,
        "samples": ordered,
    }
