"""Official public benchmark adapters.

Each adapter drives a pinned upstream benchmark against the candidate's llama.cpp endpoint and returns
sample rows in the same shape the private fixtures produce, so analysis, comparison and reporting are
unchanged. An adapter never scores with an LLM judge, never reaches the network at run time, and never
reports a task it did not actually execute: every declared task id must come back with a row, failures
included, so denominators stay honest.

The upstream dataset and harness live in the evaluator image; an adapter that cannot find them raises
`BenchmarkUnavailable` at preflight rather than silently reducing its task list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class BenchmarkUnavailable(RuntimeError):
    """The pinned dataset, harness or toolchain is absent from this image."""


class BenchmarkAborted(RuntimeError):
    """The run must stop and the campaign must not continue (propagates like UnsafeCaptureRuntimeState)."""


@dataclass(frozen=True)
class BenchmarkContext:
    """Everything an adapter may use. It gets no Docker, no network and no credentials."""

    base_url: str
    """Private llama.cpp OpenAI-compatible endpoint, e.g. http://inference:8080/v1."""
    model_alias: str
    """The exact alias the server serves; an adapter must send this and reject a different served model."""
    task_ids: tuple[str, ...]
    """Exactly the tasks to run. Every one must produce a row."""
    split: str
    """"development" or "holdout"; adapters must keep the two item sets disjoint."""
    seed: int
    generation: Any
    """GenerationSettings: temperature, top_p, seed, max_output_tokens."""
    remaining_seconds: Callable[[], float]
    """Wall budget left. An adapter checks this between items and stops cleanly, marking the rest."""
    artifacts: Any
    """Bounded writer: .write(rel, bytes) and .trace(rel). Raw responses go here BEFORE parsing."""
    session_lock: Any
    """Permission gate; an adapter performs no privileged operation of its own."""
    dataset_root: str = "/opt/llmbench/benchmarks"
    """Where the image baked the pinned datasets."""
    options: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    benchmark_id: str
    revision: str
    category: str

    def available(self, context: BenchmarkContext) -> tuple[bool, str]:
        """Preflight: is the pinned dataset/harness present? Returns (ok, reason). Never raises."""

    def task_ids(self, context: BenchmarkContext) -> tuple[str, ...]:
        """The exact, stable, ordered task ids this adapter can run for the given split."""

    def run(self, context: BenchmarkContext) -> list[dict[str, Any]]:
        """Execute and return one sample row per declared task id, failures included."""


REQUIRED_SAMPLE_KEYS = ("task_id", "suite", "suite_revision", "category", "split", "status", "score",
                        "passed", "model_evaluated", "synthetic")
"""Keys every adapter row must carry. `status` is "completed" only when the item really ran; use
"environment_error", "timeout" or "invalid_output" otherwise, and keep `score` 0.0 for those."""


def missing_rows(task_ids, produced, *, suite: str, revision: str, category: str, split: str,
                 status: str, reason: str) -> list[dict[str, Any]]:
    """Rows for declared tasks an adapter never returned, so a crash cannot shrink the denominator."""
    seen = {row.get("task_id") for row in produced}
    return [{"task_id": task_id, "suite": suite, "suite_revision": revision, "category": category,
             "split": split, "status": status, "score": 0.0, "passed": False, "model_evaluated": False,
             "synthetic": False, "outcome_status": status, "error": reason}
            for task_id in task_ids if task_id not in seen]
