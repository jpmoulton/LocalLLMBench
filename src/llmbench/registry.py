"""Benchmark registry: what a container candidate may select, and what can actually run.

Benchmark IDs are the suite names already used by ``TaskSelection.suite`` in
``evaluations/inspect_tasks.py`` and ``live.py``, so a ``BenchmarkSelection`` maps 1:1.
``runner`` means executable by this harness today. Public upstream suites are importable
results only; a local task is never an alias for an upstream benchmark. Import is inert.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Literal

from pydantic import Field, model_validator

from .config import StrictModel, canonical_json

TOOL_PROBE_TASKS = ("tools/nested-exact-v1", "tools/no-call-v1")
TOOL_EPISODE_TASKS = ("tools/read-edit-test-v1",)
IMPORT_ONLY_REVISION = "upstream-pin-required"
BROKER = "coding-broker"
DATASET_PREFIX = "dataset:"


def dataset_requirement(benchmark_id: str) -> str:
    """What a benchmark needs the evaluator image to have baked, e.g. ``dataset:ruler``."""
    return DATASET_PREFIX + benchmark_id


class RegistryError(ValueError):
    """A benchmark selection cannot be executed; raised before any model or server is touched."""


class RegistryEntry(StrictModel):
    benchmark_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    category: Literal["tools", "retrieval", "coding"]
    capability: Literal["runner", "import-only", "unavailable"]
    task_ids: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    official: bool = False
    task_namespace: str | None = None
    """Set when the task list comes from a baked dataset rather than this file. Selections must then
    name ids under ``<namespace>/``; the adapter's preflight decides which of them actually exist, so
    the registry never pretends to know the contents of an image it cannot read."""
    option_keys: tuple[str, ...] = ()
    """Option names a selection may carry. Anything else is refused; an empty tuple means none."""
    holdout_capable: bool = False
    """True only when the adapter can produce a deterministic split whose halves share no item.
    A handful of static fixtures cannot, and must never be labelled unseen holdout.

    This promises item disjointness, not freshness. ``niah`` generates its items from the seed, so
    its holdout is genuinely unseen. The public suites partition a released dataset the weights may
    already have been trained on, so their holdout guards against overfitting the *tuning* to
    particular items, not against training contamination; reports must not conflate the two."""

    @model_validator(mode="after")
    def coherent(self) -> "RegistryEntry":
        if len(set(self.task_ids)) != len(self.task_ids) or not all(self.task_ids):
            raise ValueError("task_ids must be unique nonempty strings")
        if self.task_namespace is not None and not self.task_namespace.strip("/"):
            raise ValueError("task_namespace must be a nonempty path segment")
        if self.capability != "import-only" and not self.task_ids and self.task_namespace is None:
            raise ValueError("a local benchmark must enumerate its task IDs")
        # A runner may depend on data the image bakes and on the coding broker, because both are
        # verified at preflight before a model loads. It may not depend on an unbuilt component.
        runtime_requirements = {BROKER} | {dataset_requirement(self.benchmark_id)}
        if self.capability == "runner" and set(self.requires) - runtime_requirements:
            raise ValueError("a runner entry cannot depend on a component that is not built in")
        if self.capability == "runner" and BROKER in self.requires and self.category != "coding":
            raise ValueError("only a coding benchmark needs the coding broker")
        return self


class Registry:
    def __init__(self, entries: Iterable[RegistryEntry]) -> None:
        self.entries = tuple(sorted(entries, key=lambda item: item.benchmark_id))
        self._by_id = {item.benchmark_id: item for item in self.entries}
        if len(self._by_id) != len(self.entries):
            raise ValueError("duplicate benchmark ID in registry")

    def get(self, benchmark_id: str) -> RegistryEntry | None:
        return self._by_id.get(benchmark_id)

    def digest(self) -> str:
        payload = [item.model_dump(mode="json") for item in self.entries]
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def validate(self, selections: Iterable[Any], *, broker_available: bool,
                 datasets_available: Iterable[str] = ()) -> None:
        """Reject anything that cannot run now. Accepts BenchmarkSelection-shaped objects.

        ``datasets_available`` names the benchmark ids whose pinned dataset the evaluator image
        actually baked, as reported by each adapter's ``available()``. A benchmark that needs data
        it does not have is refused here, before a model is loaded.
        """
        if type(broker_available) is not bool:
            raise RegistryError("broker_available must be a boolean")
        baked = {str(item) for item in datasets_available}
        if not baked.issubset({entry.benchmark_id for entry in self.entries}):
            raise RegistryError("datasets_available names a benchmark this registry does not list")
        available = ({BROKER} if broker_available else set()) | {dataset_requirement(item) for item in baked}
        seen: set[tuple[str, str, int | None, str]] = set()
        rows = list(selections)
        if not rows:
            raise RegistryError("select at least one benchmark")
        for selection in rows:
            entry = self._by_id.get(selection.benchmark_id)
            if entry is None:
                raise RegistryError(f"unknown benchmark: {selection.benchmark_id}")
            if entry.capability == "import-only":
                raise RegistryError(f"{entry.benchmark_id} is import-only: results can be imported with a verified "
                                    "upstream pin, but this harness cannot execute it")
            if selection.revision != entry.revision:
                raise RegistryError(f"unsupported revision for {entry.benchmark_id}: {selection.revision}")
            missing = [item for item in entry.requires if item not in available]
            if entry.capability == "unavailable" and (missing or not entry.requires):
                raise RegistryError(f"{entry.benchmark_id} requires {', '.join(missing) or 'an unbuilt runner'}")
            # A runner with code but without its baked data or broker is refused here, before load.
            if missing:
                raise RegistryError(f"{entry.benchmark_id} requires {', '.join(missing)}")
            if entry.task_namespace is None:
                unknown = [item for item in selection.task_ids if item not in entry.task_ids]
            else:
                prefix = entry.task_namespace.rstrip("/") + "/"
                unknown = [item for item in selection.task_ids
                           if not item.startswith(prefix) or item == prefix or ".." in item.split("/")]
            if unknown or not selection.task_ids:
                raise RegistryError(f"unknown {entry.benchmark_id} task: {', '.join(unknown) or '(none selected)'}")
            if len(set(selection.task_ids)) != len(selection.task_ids):
                raise RegistryError(f"duplicate {entry.benchmark_id} task IDs")
            if selection.split != "development" and not entry.holdout_capable:
                raise RegistryError(f"{entry.benchmark_id} cannot produce a disjoint unseen holdout split")
            options = dict(getattr(selection, "options", None) or {})
            refused = sorted(set(options) - set(entry.option_keys))
            if refused:
                raise RegistryError(f"{entry.benchmark_id} does not accept option(s): {', '.join(refused)}")
            for task_id in selection.task_ids:
                seeded = entry.category == "retrieval" or entry.task_namespace is not None
                seed = getattr(selection, "seed", 42) if seeded else None
                key = (entry.benchmark_id, selection.split, seed, task_id)
                if key in seen:
                    raise RegistryError(f"task selected twice: {task_id}")
                seen.add(key)


def builtin_registry() -> Registry:
    from .coding.adapters import SUITES
    from .coding.fixtures import fixtures
    from .evaluations.retrieval import NIAH_VARIANTS

    coding = fixtures()
    upstream_category = {"aider-polyglot": "coding", "evalplus": "coding", "bfcl": "tools",
                         "tool-eval-bench": "tools", "ruler": "retrieval"}
    entries = [
        RegistryEntry(benchmark_id="tool-probes", revision="local-tools-v1", category="tools",
                      capability="runner", task_ids=TOOL_PROBE_TASKS),
        RegistryEntry(benchmark_id="tool-episodes", revision="local-tools-v1", category="tools",
                      capability="runner", task_ids=TOOL_EPISODE_TASKS),
        RegistryEntry(benchmark_id="niah", revision="local-niah-v1", category="retrieval",
                      capability="runner", task_ids=tuple(NIAH_VARIANTS), holdout_capable=True),
        RegistryEntry(benchmark_id="coding", revision="private-coding-v1", category="coding",
                      capability="unavailable", task_ids=tuple(item.fixture_id for item in coding),
                      requires=(BROKER,), languages=tuple(sorted({item.language for item in coding}))),
    ]
    entries.extend(_official_entries())
    wired = {entry.benchmark_id for entry in entries}
    entries.extend(RegistryEntry(benchmark_id=name, revision=IMPORT_ONLY_REVISION, category=upstream_category[name],
                                 capability="import-only", official=True)
                   for name in SUITES if name not in wired)
    return Registry(entries)


def _official_entries() -> list[RegistryEntry]:
    """Public benchmarks this harness executes itself, from `llmbench.benchmarks`.

    Their task lists live in the pinned dataset the evaluator image bakes, not here, so each entry
    declares a namespace and requires ``dataset:<id>``. Importing this module stays inert: only the
    adapters' identity constants are read, never their data.
    """
    from .benchmarks import aider_polyglot, bfcl, evalplus, ruler

    mbpp = evalplus.mbpp_plus()
    return [
        RegistryEntry(benchmark_id=bfcl.BENCHMARK_ID, revision=bfcl.REVISION, category=bfcl.CATEGORY,
                      capability="runner", task_namespace=bfcl.BENCHMARK_ID,
                      requires=(dataset_requirement(bfcl.BENCHMARK_ID),),
                      option_keys=("modes", "items_per_category"), official=True,
                      holdout_capable=True),
        RegistryEntry(benchmark_id=ruler.BENCHMARK_ID, revision=ruler.REVISION, category=ruler.CATEGORY,
                      capability="runner", task_namespace=ruler.BENCHMARK_ID,
                      requires=(dataset_requirement(ruler.BENCHMARK_ID),),
                      option_keys=("lengths", "tasks", "ruler_output_tokens"),
                      official=True, holdout_capable=True),
        RegistryEntry(benchmark_id=mbpp.benchmark_id, revision=mbpp.revision, category=mbpp.category,
                      capability="runner", task_namespace=mbpp.benchmark_id,
                      requires=(dataset_requirement(mbpp.benchmark_id), BROKER),
                      option_keys=("item_limit", "execution_timeout_seconds", "min_item_seconds"),
                      languages=("python",), official=True, holdout_capable=True),
        RegistryEntry(benchmark_id=aider_polyglot.BENCHMARK_ID, revision=aider_polyglot.REVISION,
                      category=aider_polyglot.CATEGORY, capability="runner",
                      task_namespace=aider_polyglot.BENCHMARK_ID,
                      requires=(dataset_requirement(aider_polyglot.BENCHMARK_ID), BROKER),
                      option_keys=("languages", "expected_commit", "split_fraction"),
                      languages=aider_polyglot.DEFAULT_LANGUAGES, official=True, holdout_capable=True),
    ]
