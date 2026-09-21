"""Derive a ``ContainerSessionConfig`` from GGUF weights: pure given a hasher and a header reader.

``llmbench tune --model <file or directory>`` needs no hand-written session: every GGUF found is hashed (charged
to the session), its header supplies name, architecture, chat template, training context and MTP layers, and
the default frozen search space (AM-1) is capped by the model's ``n_ctx_train``. The result is written to
``<output>/session-config.json`` by ``tune`` so the run is reproducible from a file.

Derivation also decides WHICH benchmarks the session measures (U18). The four official public suites (BFCL,
RULER, EvalPlus MBPP+, Aider Polyglot) keep their task lists in the pinned corpora rather than in the registry,
so this module asks the environment what is actually staged (``container_eval.baked_datasets``) and each present
adapter what its task ids are (``adapter.task_ids``). Nothing is guessed: a dataset that is not there is simply
not selected, and every offer, selection, bound and skip is recorded with its reason in the plan that
``llmbench tune`` prints and writes beside the session config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from ..config import canonical_json, hash_file
from .config import ContainerRunConfig, ModelAsset
from .gguf import gguf_summary
from .session import (MIN_CONTEXT_INPUT_TOKENS, ContainerSessionConfig, SessionBudgets, output_reserve_tokens,
                      usable_input_tokens)

DEFAULT_KV_PAIRS = (("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0"))
DEFAULT_CTX_TIERS = (8192, 32768, 131072)
DEFAULT_BATCH = 2048
DEFAULT_REASONING = ("off", "on")
# RAM-offload axes. The derived default stays SINGLE-VALUED, so `tune --model ...` derives exactly the session it
# derived before: every layer on the GPU, KV cache on the GPU. A caller that wants to compare CPU/RAM offload
# passes its own members to ``derive_session_config``.
DEFAULT_GPU_LAYERS: tuple[int | str, ...] = ("all",)
DEFAULT_KV_OFFLOAD: tuple[bool, ...] = (True,)
DEFAULT_HOLDOUT_SEED = 7
DEFAULT_HOLDOUT_RESERVE, DEFAULT_CANDIDATE_WALL, DEFAULT_CLEANUP_RESERVE = 2400, 1800, 120
# Context-range derivation. Allocations are rounded UP to a 256-token boundary: llama.cpp allocates the KV
# cache in whole ubatches and the smallest ubatch this schema allows is 32, so 256 is a whole number of them and
# keeps ctx_size readable in the report. Rounding up only ever adds capacity, never removes usable input.
CONTEXT_ALLOCATION_GRANULARITY = 256
DEFAULT_CONTEXT_POINTS = 4  # floor, ceiling and at most two interior input targets
DEFAULT_CONTEXT_FLOOR = 4096  # usable input tokens; the half-fill of the smallest default tier (8192)
QUANT_SUFFIX = re.compile(r"[-_.]((?:IQ|TQ|Q)\d[A-Za-z0-9_]*|F16|F32|BF16)\.gguf$", re.IGNORECASE)
NAME_CHARS = re.compile(r"[^a-z0-9._-]+")

# ---- public benchmark selection (U18) ------------------------------------------------------------------------
IMAGE_DATASET_ROOT = "/opt/llmbench/benchmarks"
"""``benchmarks.BenchmarkContext.dataset_root``'s default: where the evaluator IMAGE bakes the pinned corpora."""
STAGED_DATASET_DIRNAME = "artifacts/benchmark-datasets"
"""Where ``scripts/stage_benchmark_datasets.py`` stages them on the HOST, relative to the working directory."""
PUBLIC_BENCHMARK_ORDER = ("bfcl", "ruler", "evalplus", "aider-polyglot")
"""Selection order, and therefore the order the per-candidate wall allowance is spent in. BFCL first because it
is the only suite that has been measured producing a SPREAD (76.2% with categorised failures, 2026-09-19
``artifacts/container-pilot/suite-q4-bfcl-1``) and is therefore the strongest discriminator; RULER second (live,
cheap at the shortest length); the two broker-gated coding suites last because neither has ever run live."""
PUBLIC_SPLIT, PUBLIC_SEED = "development", 42
"""Public suites contribute development selections only; the session's holdout plan is NIAH's (session.py)."""

RULER_DEFAULT_TASKS = ("niah_multikey_3", "vt")
RULER_SKIPPED_TASKS = {
    "cwe": "output-cap sensitive, so a weak discriminator: cwe/8192 was measured at 0.2 with output_cap_hit "
           "true even at 512 output tokens because the model enumerates word counts verbatim and is cut off "
           "mid-answer (artifacts/container-pilot/suite-q4-ruler-3)",
    "niah_multiquery": "never run live, and the upstream verification rates it the weakest of the four for "
                       "comparability; keeping the default at the two tasks measured clean also halves the "
                       "per-candidate cost (docs/benchmarks.md)",
}
"""Why the other two vendored RULER tasks are not in the derived default. Both stay selectable by hand through a
session config's ``tasks`` option; nothing is removed from the adapter."""
RULER_OUTPUT_TOKENS = 512
"""DELIBERATE DEVIATION from upstream's per-task ``tokens_to_generate`` (128 niah / 30 vt / 120 cwe). Root
measured those caps truncating a CORRECT answer into a 0.2 score - every ``vt`` score at 32K and above in
``artifacts/container-campaign/ruler-kv-1`` is a 30-token cap artefact, and the identical item scored 1.0 with
180 output tokens once the cap was raised. 512 leaves real headroom: the clean tasks answer in 19-33 tokens
(cwe used 216 when it was not truncated), so the cap stops being part of the measurement."""
MAX_RULER_LENGTHS = 4
"""At most four points on RULER's ladder per session, thinned by ``_thin`` so the shortest and the longest that
fit are always kept and a capped ladder is a subset of the full one."""
BFCL_MODES, BFCL_ITEMS_PER_CATEGORY = ("native",), 3
"""Native function-calling mode only (prompt-mode numbers are ours, not leaderboard scores) and the first three
records of each of the ten AST categories in pinned file order - the exact subset that was run live."""
EVALPLUS_ITEM_LIMIT = 20
"""The first 20 items of MBPP+'s seeded development rank order (nested subsets, so every candidate gets the same
items). The full 378-item split cannot fit a 1800-second candidate wall beside speed and the other suites."""
AIDER_LANGUAGES = ("python",)
"""The cheapest single language of the two this adapter has pinned test commands for (34 exercises, 17 in the
development half). Even that rarely fits a derived candidate wall; see ``PUBLIC_ITEM_SECONDS``."""

PUBLIC_ITEM_SECONDS = {
    # Planning estimates only - the adapters' own budget probes still govern what actually runs. Provenance:
    # BFCL: 139 s was measured for 21 items plus the local tool probes (suite-q4-bfcl-1) -> <=7 s per item.
    "bfcl": 7.0,
    # EvalPlus: measured 109 s for 23 items (generation plus one sandboxed container each) at ~70 tok/s,
    # i.e. under 5 s per item; doubled for slower models. Reasoning mode costs several times more per item.
    "evalplus": 10.0,
    # Aider Polyglot: measured 563-754 s for 42 exercises with both attempts (13-18 s each) at 110-140 tok/s
    # with speculation; doubled because the default baseline runs without it.
    "aider-polyglot": 36.0,
}
PUBLIC_BENCHMARK_WALL_SHARE = 0.5
"""How much of one candidate's wall the public suites may be planned to occupy. The rest pays for image start,
the speed probe, the local fixtures and cleanup. Exceeding it does not fail - the item count is tightened, or
the suite is skipped with the estimate and the allowance both named."""
PUBLIC_COUNT_LADDERS: dict[str, tuple[str, tuple[int, ...]]] = {
    "bfcl": ("items_per_category", (BFCL_ITEMS_PER_CATEGORY, 2, 1)),
    "evalplus": ("item_limit", (EVALPLUS_ITEM_LIMIT, 10, 5)),
}
"""Benchmarks whose item count derivation can tighten, richest first. A suite absent from this table has no
bounded-count option in the registry and is skipped rather than started at a size that cannot finish."""


def find_ggufs(model_path: str | Path) -> list[Path]:
    """One file, or every ``*.gguf`` directly inside a directory (sorted); nothing recursive or hidden."""
    target = Path(model_path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(item for item in target.iterdir() if item.is_file() and item.suffix.lower() == ".gguf"
                       and not item.name.startswith("."))
    else:
        raise ValueError(f"{target} is neither a GGUF file nor a directory")
    if not files:
        raise ValueError(f"no .gguf files found in {target}")
    split = [item.name for item in files if re.search(r"-\d{5}-of-\d{5}\.gguf$", item.name, re.IGNORECASE)]
    if split:
        raise ValueError(f"split GGUF files are not supported: {', '.join(split)}")
    return [item.resolve() for item in files]


def quantization_of(path: Path, summary: dict) -> str:
    """Prefer the header's ``general.file_type``; fall back to the filename suffix; never guess."""
    named = summary.get("quantization")
    if isinstance(named, str) and named:
        return named
    match = QUANT_SUFFIX.search(path.name)
    if match:
        return match[1].upper()
    raise ValueError(f"{path.name}: quantization is neither declared in the header nor named in the file")


def session_id_for(name: str) -> str:
    slug = NAME_CHARS.sub("-", name.lower()).strip("-.")[:50] or "model"
    return f"tune-{slug}"


def runner_benchmarks(*, broker_configured: bool) -> list[dict]:
    """The LOCAL registry benchmarks a session can select from the weights alone; coding only behind a broker.

    These are the hand-written fixtures whose task IDs the registry itself carries (tool probes, tool episodes,
    NIAH, the private coding fixtures). The dataset-backed public suites are deliberately not here: their task
    IDs live in the pinned corpora, so they need the environment probe in ``public_benchmark_plan`` rather than a
    guess from a registry entry. All twelve local fixtures score 1.000 on the weights measured so far, so a
    session that carries only these rows ranks nothing - which is exactly why derivation now adds the public
    suites that are actually staged.
    """
    from ..registry import builtin_registry
    rows = []
    for entry in builtin_registry().entries:
        if entry.task_namespace is not None:
            continue
        if entry.capability == "runner" or (broker_configured and entry.benchmark_id == "coding"):
            rows.append({"benchmark_id": entry.benchmark_id, "revision": entry.revision,
                         "task_ids": list(entry.task_ids)})
    return rows


def resolve_dataset_root(base: ContainerRunConfig, *, dataset_root: str | None = None,
                         project_root: str | Path | None = None) -> tuple[str | None, str]:
    """``(dataset_root, source)``: where THIS session's evaluator will look for the pinned public corpora.

    ``None`` means "leave the evaluator image's baked path alone" and is correct only where that path exists.
    A host-process evaluator reads the HOST filesystem, so ``/opt/llmbench/benchmarks`` is not there for it and
    every dataset reads as absent - the live symptom root hit was the runner correctly refusing all four public
    benchmarks at admission. The staged host directory is therefore the default in that mode, but only when it
    really exists: naming a path that is not there would turn an honest "no datasets here" into a wrong reason.

    ``source`` is one of ``explicit`` (``--dataset-root``), ``base-config``, ``staged-host-default``,
    ``evaluator-image`` (container mode) or ``none-staged``.
    """
    if dataset_root is not None:
        resolved = Path(dataset_root).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"dataset root {resolved} is not a directory; stage the corpora with "
                             "scripts/stage_benchmark_datasets.py or omit --dataset-root")
        return str(resolved.resolve()), "explicit"
    if base.dataset_root is not None:
        return base.dataset_root, "base-config"
    if base.evaluator.mode != "host-process":
        return None, "evaluator-image"
    staged = (Path(project_root) if project_root is not None else Path.cwd()) / STAGED_DATASET_DIRNAME
    return (str(staged.resolve()), "staged-host-default") if staged.is_dir() else (None, "none-staged")


def dataset_root_reason(dataset_root: str | None, source: str) -> str:
    """One sentence a user can act on, for the plan record and the console."""
    return {
        "explicit": f"--dataset-root {dataset_root}",
        "base-config": f"the base candidate config names {dataset_root}",
        "staged-host-default": f"the staged host directory {dataset_root} exists and the evaluator is a host "
                               "process",
        "evaluator-image": f"the evaluator runs in a container and reads its baked {IMAGE_DATASET_ROOT}, which "
                           "this host cannot enumerate",
        "none-staged": f"no staged dataset directory exists under {STAGED_DATASET_DIRNAME}; run "
                       "scripts/stage_benchmark_datasets.py or pass --dataset-root",
    }[source]


@dataclass(frozen=True)
class _Defaults:
    """The option set derivation asks a public adapter for, or the reason it asks for nothing."""

    options: dict[str, Any] | None = None
    notes: tuple[str, ...] = ()
    refusal: str = ""


def ruler_lengths_for(usable_input_ceiling: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(kept, dropped)`` RULER lengths: the upstream ladder that fits, thinned to ``MAX_RULER_LENGTHS``.

    Bounded by the SMALLEST context tier, not the largest: one benchmark selection is shared by every candidate
    in the session, and a length that does not fit a candidate's served context is reported as a refused item
    scoring 0.0 - a fake quality finding of exactly the kind already caught twice (docs/lessons.md). To measure retrieval
    at 128K, pin the floor there (``--context-floor 131072``) so every candidate can hold the prompt.
    """
    from ..benchmarks.ruler import DEFAULT_LENGTHS
    kept, dropped = _thin([length for length in DEFAULT_LENGTHS if length <= usable_input_ceiling],
                          MAX_RULER_LENGTHS)
    return tuple(kept), tuple(dropped)


def _defaults_for(benchmark_id: str, *, usable_input_ceiling: int) -> _Defaults:
    if benchmark_id == "ruler":
        from ..benchmarks.ruler import DEFAULT_LENGTHS
        kept, dropped = ruler_lengths_for(usable_input_ceiling)
        if not kept:
            return _Defaults(refusal=f"the smallest context tier holds only {usable_input_ceiling} usable input "
                                     f"tokens and RULER's shortest standard length is {min(DEFAULT_LENGTHS)}")
        notes = [f"tasks limited to {', '.join(RULER_DEFAULT_TASKS)}; "
                 + "; ".join(f"{task} excluded - {why}" for task, why in sorted(RULER_SKIPPED_TASKS.items())),
                 f"ruler_output_tokens {RULER_OUTPUT_TOKENS}: a deliberate deviation from upstream's per-task "
                 "128/30/120 caps, which were measured truncating correct answers into failures",
                 f"lengths bounded by the smallest context tier ({usable_input_ceiling} usable input tokens)"]
        if dropped:
            notes.append(f"lengths {list(dropped)} dropped to keep at most {MAX_RULER_LENGTHS} ladder points")
        return _Defaults({"tasks": list(RULER_DEFAULT_TASKS), "lengths": list(kept),
                          "ruler_output_tokens": RULER_OUTPUT_TOKENS}, tuple(notes))
    if benchmark_id == "bfcl":
        return _Defaults({"modes": list(BFCL_MODES), "items_per_category": BFCL_ITEMS_PER_CATEGORY},
                         (f"modes {list(BFCL_MODES)} (prompt mode doubles the items and its numbers are ours, "
                          "not leaderboard scores)",
                          f"items_per_category {BFCL_ITEMS_PER_CATEGORY}, in pinned file order"))
    if benchmark_id == "evalplus":
        return _Defaults({"item_limit": EVALPLUS_ITEM_LIMIT},
                         (f"item_limit {EVALPLUS_ITEM_LIMIT} of the seeded development rank order (nested "
                          "subsets, so every candidate is scored on the same items)",))
    if benchmark_id == "aider-polyglot":
        return _Defaults({"languages": list(AIDER_LANGUAGES)},
                         (f"languages {list(AIDER_LANGUAGES)}",))
    return _Defaults(refusal="derivation carries no default option set for this benchmark")


def _option_ladder(benchmark_id: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    """The option sets to try, richest first, when the estimate does not fit the wall allowance."""
    if benchmark_id == "ruler":  # drop the longest length first: it costs by far the most wall
        return [{**options, "lengths": list(options["lengths"][:count])}
                for count in range(len(options["lengths"]), 0, -1)]
    if benchmark_id in PUBLIC_COUNT_LADDERS:
        name, values = PUBLIC_COUNT_LADDERS[benchmark_id]
        return [{**options, name: value} for value in values if value <= options[name]]
    return [dict(options)]


def estimated_selection_seconds(benchmark_id: str, *, task_ids: tuple[str, ...], options: dict[str, Any]) -> float:
    """Planning estimate for one candidate's run of this selection. Never a measurement; see
    ``PUBLIC_ITEM_SECONDS`` for each number's provenance. RULER models its own items, because a 131K prompt
    costs two orders of magnitude more than a 4K one."""
    if benchmark_id == "ruler":
        from ..benchmarks.ruler import estimated_item_seconds
        return len(options["tasks"]) * sum(estimated_item_seconds(length, options["ruler_output_tokens"], {})
                                           for length in options["lengths"])
    if benchmark_id not in PUBLIC_ITEM_SECONDS:  # a default option set without a cost model would select blind
        raise KeyError(f"{benchmark_id} has default options in _defaults_for but no per-item planning cost in "
                       "PUBLIC_ITEM_SECONDS")
    return len(task_ids) * PUBLIC_ITEM_SECONDS[benchmark_id]


def _adapter_context(benchmark_id: str, dataset_root: str, options: dict[str, Any]) -> Any:
    """A ``BenchmarkContext`` with no endpoint, no artifacts and no lock: enough to enumerate task ids only.

    The option names are translated through ``container_eval.BENCHMARK_WIRING`` - the same table the evaluator
    uses - because a selection states REGISTRY names (``tasks``, ``lengths``) while ``run()`` reads the adapter's
    (``ruler_tasks``, ``ruler_lengths``). Reproducing that mapping here would be a second source of truth.
    """
    from ..benchmarks import BenchmarkContext
    from ..container_eval import BENCHMARK_WIRING
    names = BENCHMARK_WIRING[benchmark_id].options
    return BenchmarkContext(base_url="", model_alias="", task_ids=(), split=PUBLIC_SPLIT, seed=PUBLIC_SEED,
                            generation=None, remaining_seconds=lambda: 0.0, artifacts=None, session_lock=None,
                            dataset_root=dataset_root,
                            options={names[key]: value for key, value in options.items()})


def _public_row(benchmark_id: str, *, status: str, reason: str, items: int = 0, seconds: float = 0.0,
                options: dict[str, Any] | None = None, notes: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"benchmark_id": benchmark_id, "status": status, "reason": reason, "items": items,
            "estimated_seconds": round(seconds, 1), "options": options, "notes": list(notes)}


def _fit_selection(benchmark_id: str, options: dict[str, Any], *, dataset_root: str,
                   allowance: float) -> tuple[dict[str, Any] | None, tuple[str, ...], float, str]:
    """``(options, task_ids, seconds, problem)``: the richest option set whose estimate fits ``allowance``.

    Every attempt asks the ADAPTER for its task ids, so the recorded task list and the recorded options can
    never disagree. An adapter that refuses (a corrupt pin, an unreadable corpus) becomes ``problem``, never an
    exception: a missing dataset is a recorded skip, not a failed derivation.
    """
    attempts = _option_ladder(benchmark_id, options)
    last = ""
    for options in attempts:
        try:
            task_ids = tuple(_adapter_from(benchmark_id).task_ids(_adapter_context(benchmark_id, dataset_root,
                                                                                  options)))
        except Exception as exc:  # an adapter must not be able to break derivation
            return None, (), 0.0, f"the adapter could not enumerate its tasks: {type(exc).__name__}: {exc}"
        if not task_ids:
            return None, (), 0.0, f"the adapter offers no {PUBLIC_SPLIT} task for this pinned dataset"
        seconds = estimated_selection_seconds(benchmark_id, task_ids=task_ids, options=options)
        if seconds <= allowance:
            return options, task_ids, seconds, ""
        last = (f"an estimated {seconds:.0f} s for {len(task_ids)} items does not fit the "
                f"{allowance:.0f} s left of this candidate's public-benchmark allowance")
    return None, (), 0.0, last


def _adapter_from(benchmark_id: str) -> Any:
    from ..container_eval import BENCHMARK_ADAPTERS
    return BENCHMARK_ADAPTERS[benchmark_id]()


def public_benchmarks_offered(registry: Any) -> list[str]:
    """Every runnable dataset-backed suite the registry lists, in ``PUBLIC_BENCHMARK_ORDER`` then alphabetically.

    Driven off the registry rather than off the constant, so a public suite wired in later cannot vanish from the
    plan: it is offered, and skipped with "derivation carries no default option set for this benchmark" until
    somebody gives it one. A silent absence is the failure mode this whole plan exists to prevent.
    """
    runnable = [entry.benchmark_id for entry in registry.entries
                if entry.capability == "runner" and entry.task_namespace is not None]
    return [item for item in PUBLIC_BENCHMARK_ORDER if item in runnable] + sorted(
        item for item in runnable if item not in PUBLIC_BENCHMARK_ORDER)


def public_benchmark_plan(*, dataset_root: str | None, dataset_root_source: str, broker_configured: bool,
                          usable_input_ceiling: int, candidate_wall_seconds: int) -> dict[str, Any]:
    """Which official public suites this session will measure, which it will not, and why for every one.

    ``selections`` are ready ``BenchmarkSelection`` rows; ``benchmarks`` is the audit a user reads to see why
    something is absent. Availability comes from ``container_eval.baked_datasets`` - the same probe the runner
    uses at admission and the evaluator image uses in its self-check - so derivation can never select a corpus
    the run would then be refused for.
    """
    from ..registry import BROKER, builtin_registry
    registry = builtin_registry()
    present: tuple[str, ...] = ()
    if dataset_root is not None:
        from ..container_eval import baked_datasets
        present = baked_datasets(dataset_root)
    allowance = candidate_wall_seconds * PUBLIC_BENCHMARK_WALL_SHARE
    remaining, rows, selections = allowance, [], []
    for benchmark_id in public_benchmarks_offered(registry):
        entry = registry.get(benchmark_id)
        if entry is None or entry.capability != "runner" or entry.task_namespace is None:
            rows.append(_public_row(benchmark_id, status="skipped",
                                    reason="the registry does not offer it as a runnable public suite"))
            continue
        if BROKER in entry.requires and not broker_configured:
            rows.append(_public_row(benchmark_id, status="skipped",
                                    reason="it executes generated code, which needs the coding worker broker; "
                                           "the base candidate config carries none (set `broker` to enable it)"))
            continue
        if dataset_root is None or benchmark_id not in present:
            why = dataset_root_reason(dataset_root, dataset_root_source)
            rows.append(_public_row(benchmark_id, status="skipped",
                                    reason=f"its pinned dataset is not present: {why}"))
            continue
        defaults = _defaults_for(benchmark_id, usable_input_ceiling=usable_input_ceiling)
        if defaults.options is None:
            rows.append(_public_row(benchmark_id, status="skipped", reason=defaults.refusal))
            continue
        options, task_ids, seconds, problem = _fit_selection(benchmark_id, defaults.options,
                                                             dataset_root=dataset_root, allowance=remaining)
        if options is None:
            rows.append(_public_row(benchmark_id, status="skipped", reason=problem, notes=defaults.notes))
            continue
        notes = list(defaults.notes)
        if options != defaults.options:
            notes.append(f"item count tightened to fit the wall allowance: {options}")
        remaining -= seconds
        selections.append({"benchmark_id": benchmark_id, "revision": entry.revision, "task_ids": list(task_ids),
                           "split": PUBLIC_SPLIT, "seed": PUBLIC_SEED, "options": options})
        rows.append(_public_row(benchmark_id, status="selected",
                                reason=f"its pinned dataset is present under {dataset_root}",
                                items=len(task_ids), seconds=seconds, options=options, notes=tuple(notes)))
    return {"power": quality_power(rows, registry),
            "dataset_root": dataset_root, "dataset_root_source": dataset_root_source,
            "dataset_root_reason": dataset_root_reason(dataset_root, dataset_root_source),
            "datasets_present": list(present), "broker_configured": broker_configured,
            "candidate_wall_seconds": candidate_wall_seconds, "wall_allowance_seconds": round(allowance, 1),
            "wall_planned_seconds": round(allowance - remaining, 1),
            "offered": [row["benchmark_id"] for row in rows],
            "selected": [row["benchmark_id"] for row in rows if row["status"] == "selected"],
            "skipped": [row["benchmark_id"] for row in rows if row["status"] == "skipped"],
            "benchmarks": rows, "selections": selections}


def quality_power(rows: list[dict], registry: Any) -> list[dict[str, Any]]:
    """Per quality category: the public items selected, and the smallest loss that many items can demonstrate.

    Validating a non-baseline candidate needs non-inferiority within the policy's ``max_quality_loss``. With n
    pass/fail items and not one regression, the best that can be shown is a loss below ``1 - 0.025 ** (1 / n)``.
    When that exceeds the tolerance, only the baseline can be validated however good a candidate is - which is
    worth knowing before a multi-hour run rather than after it.
    """
    from ..config import AcceptancePolicy
    from ..search import smallest_demonstrable_loss, tasks_needed_for_loss
    tolerance = AcceptancePolicy().max_quality_loss
    totals: dict[str, int] = {}
    for row in rows:
        entry = registry.get(row["benchmark_id"])
        if row["status"] == "selected" and entry is not None:
            totals[entry.category] = totals.get(entry.category, 0) + row["items"]
    return [{"category": category, "public_items": items,
             "smallest_demonstrable_loss": round(smallest_demonstrable_loss(items), 4),
             "policy_max_quality_loss": tolerance, "items_needed": tasks_needed_for_loss(tolerance),
             "can_validate_a_non_baseline_candidate": smallest_demonstrable_loss(items) <= tolerance}
            for category, items in sorted(totals.items())]


def benchmark_plan_lines(plan: dict[str, Any]) -> list[str]:
    """The plan as console text: one line per benchmark, reasons included. Never silently empty."""
    lines = [f"public benchmark datasets: {plan['dataset_root_reason']}",
             f"present: {', '.join(plan['datasets_present']) or 'none'}; "
             f"selected: {', '.join(plan['selected']) or 'none'} "
             f"({plan['wall_planned_seconds']:.0f} s of a {plan['wall_allowance_seconds']:.0f} s per-candidate "
             "allowance, estimated)"]
    for row in plan["benchmarks"]:
        lines.append(f"  {row['benchmark_id']}: {row['status']} - {row['reason']}"
                     + (f" [{row['items']} items, ~{row['estimated_seconds']:.0f} s]"
                        if row["status"] == "selected" else ""))
        lines.extend(f"      note: {note}" for note in row["notes"])
    for item in plan.get("power", []):
        if not item["can_validate_a_non_baseline_candidate"]:
            lines.append(f"  power: {item['category']} has {item['public_items']} public items, which can demonstrate a "
                         f"quality loss no smaller than {item['smallest_demonstrable_loss']:.0%}; the policy asks for "
                         f"{item['policy_max_quality_loss']:.0%} (about {item['items_needed']} items). Candidates "
                         "will be measured and compared, but only the baseline can be VALIDATED at this size.")
    return lines


def session_budgets_for(budget_seconds: int) -> SessionBudgets:
    """Default reserves scaled down for short budgets; validation still refuses budgets too small to run."""
    holdout = min(DEFAULT_HOLDOUT_RESERVE, budget_seconds // 6)
    candidate = min(DEFAULT_CANDIDATE_WALL, budget_seconds - holdout - DEFAULT_CLEANUP_RESERVE)
    return SessionBudgets(wall_seconds=budget_seconds, holdout_reserve_seconds=holdout,
                          cleanup_reserve_seconds=DEFAULT_CLEANUP_RESERVE, candidate_wall_seconds=candidate)


def _round_up(value: int, step: int = CONTEXT_ALLOCATION_GRANULARITY) -> int:
    return -(-value // step) * step


def _round_down(value: int, step: int = CONTEXT_ALLOCATION_GRANULARITY) -> int:
    return (value // step) * step


def context_allocation_for(input_tokens: int, *, template_reserve_tokens: int, output_tokens: int) -> int:
    """The engine ``ctx_size`` that holds ``input_tokens`` of real INPUT.

    Allocation = input target + template reserve + output capacity, rounded UP to
    ``CONTEXT_ALLOCATION_GRANULARITY``. Rounding up only adds capacity, so the target stays fully usable as
    input with the output still reserved above it.
    """
    if type(input_tokens) is not int or input_tokens < MIN_CONTEXT_INPUT_TOKENS:
        raise ValueError(f"an input target must be at least {MIN_CONTEXT_INPUT_TOKENS} usable input tokens")
    return _round_up(input_tokens + template_reserve_tokens + output_tokens)


def _thin(targets: list[int], keep: int) -> tuple[list[int], list[int]]:
    """Keep at most ``keep`` of ``targets``, always the ceiling and (from two upwards) the floor, the rest
    spread evenly. Returns ``(kept, dropped)``; a capped search is always a subset of the full one."""
    if keep >= len(targets):
        return list(targets), []
    if keep <= 1:
        kept = [targets[-1]]
    else:
        indexes = sorted({round(index * (len(targets) - 1) / (keep - 1)) for index in range(keep)})
        kept = [targets[index] for index in indexes]
    return kept, [item for item in targets if item not in kept]


def context_input_targets(context_floor: int, context_ceiling: int, *,
                          max_points: int = DEFAULT_CONTEXT_POINTS) -> list[int]:
    """Ascending USABLE INPUT targets across ``[context_floor, context_ceiling]``.

    The full layout is ``DEFAULT_CONTEXT_POINTS`` targets: both bounds (the ceiling is the claim an explicit
    range exists to evidence, the floor is the cheap end of the comparison) plus interior targets spaced
    GEOMETRICALLY and rounded down to the allocation granularity. Geometric, not linear: engine behaviour at 8K,
    32K and 128K differs by order of magnitude, and a wide range spaced linearly (512..262144 -> 512, 87K, 175K,
    262K) would never test a small context at all. A smaller ``max_points`` THINS that layout instead of
    re-spacing it, so a capped search is always a subset of the full one and the dropped targets can be named.
    """
    if type(context_floor) is not int or type(context_ceiling) is not int:
        raise ValueError("context_floor and context_ceiling must be integer token counts")
    if min(context_floor, context_ceiling) < MIN_CONTEXT_INPUT_TOKENS:
        raise ValueError(f"context_floor and context_ceiling are usable INPUT tokens and must be at least "
                         f"{MIN_CONTEXT_INPUT_TOKENS}")
    if context_floor > context_ceiling:
        raise ValueError(f"context_floor {context_floor} exceeds context_ceiling {context_ceiling}")
    if max_points < 1:
        raise ValueError("a context range needs at least one input target")
    if context_floor == context_ceiling:
        return [context_ceiling]
    targets = {context_floor, context_ceiling}
    ratio = (context_ceiling / context_floor) ** (1 / (DEFAULT_CONTEXT_POINTS - 1))
    for index in range(1, DEFAULT_CONTEXT_POINTS - 1):
        interior = _round_down(int(context_floor * ratio ** index))
        targets.add(min(max(interior, context_floor), context_ceiling))
    return _thin(sorted(targets), max_points)[0]


def context_tiers_for_range(context_floor: int, context_ceiling: int, *, template_reserve_tokens: int,
                            output_tokens: int, n_ctx_train: int,
                            max_points: int = DEFAULT_CONTEXT_POINTS) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(ctx_tiers, dropped_input_targets)`` for a range of USABLE INPUT tokens.

    Every tier is an allocation (``context_allocation_for``) and never exceeds ``n_ctx_train``; an unreachable
    ceiling is refused here rather than silently shrunk. ``dropped_input_targets`` are the targets the
    full-resolution layout would have measured but ``max_points`` (the candidate-count cap) removed; two targets
    that round to one allocation are merged silently because that single allocation still fills to at least the
    larger target.
    """
    allocation = dict(template_reserve_tokens=template_reserve_tokens, output_tokens=output_tokens)
    ceiling_allocation = context_allocation_for(context_ceiling, **allocation)
    if ceiling_allocation > n_ctx_train:
        raise ValueError(f"context_ceiling {context_ceiling} usable input tokens needs a {ceiling_allocation}-token "
                         f"allocation (input + {template_reserve_tokens} template reserve + {output_tokens} "
                         f"output), but the model's training context is {n_ctx_train} tokens")
    full = context_input_targets(context_floor, context_ceiling, max_points=DEFAULT_CONTEXT_POINTS)
    kept, dropped = _thin(full, max_points)
    tiers = sorted({context_allocation_for(target, **allocation) for target in kept})
    return tuple(tiers), tuple(dropped)


def context_range_bounds(context_floor: int | None, context_ceiling: int | None, *, n_ctx_train: int,
                         template_reserve_tokens: int, output_tokens: int) -> tuple[int, int]:
    """Fill in whichever bound the caller left out (both are USABLE INPUT tokens).

    A missing ceiling becomes the largest input this model can hold once the template reserve and the output
    capacity are subtracted from ``n_ctx_train`` (rounded down to the allocation granularity). A missing floor
    becomes ``DEFAULT_CONTEXT_FLOOR`` (4096), or the ceiling when that is smaller.

    The reserve is subtracted ROUNDED UP, because ``context_allocation_for`` rounds ``input + reserve`` up to the
    same granularity: subtracting the raw reserve would derive a ceiling whose allocation exceeds ``n_ctx_train``
    whenever ``n_ctx_train % 256 >= (reserve + output) % 256 > 0``, and this module would then refuse its own
    derivation. A no-op wherever the reserve is already a whole multiple of the granularity.
    """
    largest = _round_down(n_ctx_train - _round_up(template_reserve_tokens + output_tokens))
    ceiling = largest if context_ceiling is None else context_ceiling
    if ceiling < MIN_CONTEXT_INPUT_TOKENS:
        raise ValueError(f"training context {n_ctx_train} leaves only {ceiling} usable input tokens once the "
                         f"{template_reserve_tokens}-token template reserve and {output_tokens} output tokens are "
                         "reserved; no context range is possible")
    return (min(DEFAULT_CONTEXT_FLOOR, ceiling) if context_floor is None else context_floor), ceiling


def context_points_budget(*, max_candidates: int, quantizations: int, kv_pairs: int, spec_types: int,
                          reasoning: int, batch_sizes: int, gpu_layers: int = 1, kv_offload: int = 1) -> int:
    """How many context tiers the deterministic schedule can carry inside ``max_candidates``.

    The schedule is baseline + one candidate per non-baseline member of every axis, and the context sweep is
    LAST, so an oversized tier list would be truncated away and the ceiling would never run. The first tier is
    the baseline's and costs nothing extra; at least the ceiling is always kept.

    Every non-context axis must be counted here, the two RAM-offload axes (``gpu_layers``, ``kv_offload``)
    included: the slots they consume sit BEFORE the context sweep in ``proposals.SWEEP_ORDER``. ``others`` is an
    UPPER bound on the non-context candidates and can never under-count them -- ``full_schedule`` additionally
    deduplicates by candidate fingerprint, which can only remove candidates -- so the tier count returned can
    never over-state what fits and ``session._coherent_context_schedule`` can never find the ceiling truncated
    away. The offload axes default to 1 (a single-member axis costs no slot), so an existing caller's budget is
    unchanged.
    """
    others = sum(max(0, count - 1) for count in (quantizations, kv_pairs, spec_types, reasoning, batch_sizes,
                                                 gpu_layers, kv_offload))
    return max(1, min(DEFAULT_CONTEXT_POINTS, max_candidates - 1 - others + 1))


@dataclass(frozen=True)
class _Headers:
    """What the GGUF headers alone settle, before any file is hashed."""

    quantizations: list[str]
    model_name: str
    n_ctx_train: int
    spec_types: tuple[str, ...]
    summaries: dict[Path, dict]


def _read_headers(files: list[Path], summaries: dict[Path, dict]) -> _Headers:
    facts, quantizations, names = [], [], set()
    for path in files:
        summary = summaries[path]
        if not summary.get("template_hash"):
            raise ValueError(f"{path.name} has no tokenizer.chat_template; the session cannot identify prompts")
        name = summary.get("name") if isinstance(summary.get("name"), str) and summary.get("name") else path.stem
        names.add(name)
        quantizations.append(quantization_of(path, summary))
        facts.append(summary)
    if len(set(quantizations)) != len(quantizations):
        raise ValueError(f"two GGUF files declare the same quantization: {quantizations}")
    if len(names) != 1:
        raise ValueError("every GGUF in a session must be the same model; found " + ", ".join(sorted(names)))
    trained = [item["n_ctx_train"] for item in facts if type(item.get("n_ctx_train")) is int]
    if len(trained) != len(facts):
        raise ValueError("every GGUF must declare its training context length")
    nextn = all(type(item.get("nextn_predict_layers")) is int and item["nextn_predict_layers"] > 0 for item in facts)
    return _Headers(quantizations, names.pop(), min(trained), ("none", "draft-mtp") if nextn else ("none",),
                    summaries)


def _context_shape(headers: _Headers, *, base: ContainerRunConfig, budgets: SessionBudgets,
                   context_floor: int | None, context_ceiling: int | None,
                   gpu_layers: tuple, kv_offload: tuple) -> tuple[tuple[int, ...], int, dict[str, Any]]:
    """``(ctx_tiers, baseline input fill, recorded context range)``. Independent of the file hashes."""
    reserve, output = base.template_reserve_tokens, output_reserve_tokens(base)
    n_ctx_train = headers.n_ctx_train
    if context_floor is None and context_ceiling is None:
        tiers = tuple(tier for tier in DEFAULT_CTX_TIERS if tier <= n_ctx_train) or ((n_ctx_train,)
                                                                                     if n_ctx_train >= 512 else ())
        if not tiers:
            raise ValueError(f"training context {n_ctx_train} is below the smallest usable tier")
        return tiers, min(base.requested_input_tokens, tiers[0] // 2), {}
    # an explicit range REPLACES the fixed tiers; both bounds are usable INPUT tokens
    floor, ceiling = context_range_bounds(context_floor, context_ceiling, n_ctx_train=n_ctx_train,
                                          template_reserve_tokens=reserve, output_tokens=output)
    tiers, dropped = context_tiers_for_range(
        floor, ceiling, template_reserve_tokens=reserve, output_tokens=output, n_ctx_train=n_ctx_train,
        max_points=context_points_budget(max_candidates=budgets.max_candidates,
                                         quantizations=len(headers.quantizations), kv_pairs=len(DEFAULT_KV_PAIRS),
                                         spec_types=len(headers.spec_types), reasoning=len(DEFAULT_REASONING),
                                         batch_sizes=1, gpu_layers=len(gpu_layers), kv_offload=len(kv_offload)))
    fill = usable_input_tokens(tiers[0], template_reserve_tokens=reserve, output_tokens=output, ceiling=ceiling)
    return tiers, fill, {"context_floor": floor, "context_ceiling": ceiling, "context_n_ctx_train": n_ctx_train,
                         "context_points_dropped": list(dropped)}


def candidate_wall_for_tiers(base: ContainerRunConfig, tiers: tuple[int, ...], budgets: SessionBudgets,
                             *, ceiling: int | None) -> SessionBudgets:
    """Session budgets whose per-candidate wall admits the longest tier's work, when the session can afford it.

    ``proposals.size_long_context_work`` raises a long candidate's own bounds, but the runner is granted the
    smaller of those and the SESSION's ``candidate_wall_seconds``; left at the 1800 s default the long tier would
    still be cut off. When even one such candidate does not fit beside the holdout reserve the budgets are returned
    unchanged: the candidate then times out and says so, which is the honest outcome for a budget that small.
    """
    from .proposals import size_long_context_work
    reserve, output = base.template_reserve_tokens, output_reserve_tokens(base)
    raw = base.model_dump(mode="json")
    raw["requested_input_tokens"] = usable_input_tokens(max(tiers), template_reserve_tokens=reserve,
                                                        output_tokens=output, ceiling=ceiling)
    size_long_context_work(raw)
    needed = raw["bounds"]["candidate_wall_seconds"]
    if needed <= budgets.candidate_wall_seconds or needed + budgets.holdout_reserve_seconds > budgets.wall_seconds:
        return budgets
    return budgets.model_copy(update={"candidate_wall_seconds": needed})


def ruler_input_ceiling(base: ContainerRunConfig, smallest_tier: int) -> int:
    """The longest RULER prompt every candidate in the session can hold, output capacity reserved on top."""
    return usable_input_tokens(smallest_tier, template_reserve_tokens=base.template_reserve_tokens,
                               output_tokens=max(output_reserve_tokens(base), RULER_OUTPUT_TOKENS))


def benchmark_plan_for_model(model_path: str | Path, *, base: ContainerRunConfig, budget_seconds: int = 14400,
                             reader: Callable[[Path], dict] = gguf_summary, context_floor: int | None = None,
                             context_ceiling: int | None = None, dataset_root: str | None = None,
                             project_root: str | Path | None = None,
                             gpu_layers: tuple[int | Literal["all"], ...] | None = None,
                             kv_offload: tuple[bool, ...] | None = None) -> dict[str, Any]:
    """Exactly the benchmark plan ``derive_session_config`` will build, WITHOUT hashing the weights.

    ``llmbench tune --model`` prints and records this before the session starts, so a user or an agent can see what
    will be measured - and why anything is absent - without waiting for a four-hour run to find out. Same
    arguments in, same plan out: the two callers share ``_read_headers``, ``_context_shape`` and
    ``public_benchmark_plan``, so the printed plan cannot drift from the derived one.
    """
    gpu_layers = DEFAULT_GPU_LAYERS if gpu_layers is None else tuple(gpu_layers)
    kv_offload = DEFAULT_KV_OFFLOAD if kv_offload is None else tuple(kv_offload)
    files = find_ggufs(model_path)
    headers = _read_headers(files, {path: reader(path) for path in files})
    budgets = session_budgets_for(budget_seconds)
    tiers, _, _ = _context_shape(headers, base=base, budgets=budgets, context_floor=context_floor,
                                 context_ceiling=context_ceiling, gpu_layers=gpu_layers, kv_offload=kv_offload)
    root, source = resolve_dataset_root(base, dataset_root=dataset_root, project_root=project_root)
    return public_benchmark_plan(dataset_root=root, dataset_root_source=source,
                                 broker_configured=getattr(base, "broker", None) is not None,
                                 usable_input_ceiling=ruler_input_ceiling(base, tiers[0]),
                                 candidate_wall_seconds=budgets.candidate_wall_seconds)


def derive_session_config(model_path: str | Path, *, base: ContainerRunConfig, budget_seconds: int = 14400,
                          image_bundle: str | None = None, inference_image=None,
                          hasher: Callable[[Path], str] = hash_file, reader: Callable[[Path], dict] = gguf_summary,
                          holdout_seed: int = DEFAULT_HOLDOUT_SEED, context_floor: int | None = None,
                          context_ceiling: int | None = None,
                          gpu_layers: tuple[int | Literal["all"], ...] | None = None,
                          kv_offload: tuple[bool, ...] | None = None, dataset_root: str | None = None,
                          project_root: str | Path | None = None) -> ContainerSessionConfig:
    """Pure given ``hasher``, ``reader`` and the staged datasets: the same inputs always derive the same session.

    Coding benchmarks (the private fixtures, EvalPlus and Aider Polyglot) are included only when ``base.broker``
    is configured: they execute generated code, the registry lists them unavailable without the broker, and
    ``ContainerRunConfig`` refuses a coding selection without one.

    The four official public suites are selected from what is actually staged under the resolved
    ``dataset_root`` - see ``resolve_dataset_root`` for how that is chosen and ``public_benchmark_plan`` for what
    is selected, bounded and skipped, with a reason for each. ``dataset_root`` is recorded on the derived base, so
    it round-trips through ``session-config.json`` into every candidate and into ``resume``.

    ``context_floor``/``context_ceiling`` are USABLE INPUT TOKENS (output capacity is reserved on top of them).
    Given either one, the derived ``ctx_tiers`` are the allocations for that range and REPLACE
    ``DEFAULT_CTX_TIERS``; the range, the model's training context and any input target the candidate cap
    dropped are recorded in the session config so ``resume`` reproduces the same search.

    ``gpu_layers``/``kv_offload`` are the CPU/RAM offload axes and default to the single member that matches
    today's behaviour (``("all",)`` and ``(True,)``): the derived session is unchanged unless a caller asks for
    the sweep. Extra members cost schedule slots, so they are counted into ``context_points_budget`` before the
    context tiers are sized -- the declared ceiling keeps priority over an offload candidate.
    """
    gpu_layers = DEFAULT_GPU_LAYERS if gpu_layers is None else tuple(gpu_layers)
    kv_offload = DEFAULT_KV_OFFLOAD if kv_offload is None else tuple(kv_offload)
    files = find_ggufs(model_path)
    summaries = {path: reader(path) for path in files}
    headers = _read_headers(files, summaries)
    assets = [ModelAsset(host_path=str(path), size_bytes=path.stat().st_size, sha256=hasher(path),
                         quantization=quantization, model_name=headers.model_name,
                         source_revision=f"local:{path.parent.name}")
              for path, quantization in zip(files, headers.quantizations)]
    quantizations = list(headers.quantizations)
    spec_types = headers.spec_types
    budgets = session_budgets_for(budget_seconds)
    tiers, fill, context_search = _context_shape(headers, base=base, budgets=budgets, context_floor=context_floor,
                                                 context_ceiling=context_ceiling, gpu_layers=gpu_layers,
                                                 kv_offload=kv_offload)
    broker = getattr(base, "broker", None) is not None
    root, source = resolve_dataset_root(base, dataset_root=dataset_root, project_root=project_root)
    plan = public_benchmark_plan(dataset_root=root, dataset_root_source=source, broker_configured=broker,
                                 usable_input_ceiling=ruler_input_ceiling(base, tiers[0]),
                                 candidate_wall_seconds=budgets.candidate_wall_seconds)
    raw_base = base.model_dump(mode="json")
    raw_base.update(label="session-base", session_id=None, parent_grant_seconds=None, dataset_root=root,
                    assets=[assets[0].model_dump(mode="json")],
                    benchmarks=runner_benchmarks(broker_configured=broker) + plan["selections"])
    if inference_image is not None:
        raw_base["inference_image"] = inference_image.model_dump(mode="json")
    engine = raw_base["engine"]
    # llama.cpp silently clamps a batch above ctx_size, so a small context tier clamps it here instead.
    batch = min(DEFAULT_BATCH, tiers[0])
    # The base candidate is pinned to the baseline member of every axis, the offload axes included, so
    # ``_coherent_offload_axes`` has nothing to reconcile and the baseline runs the configuration it declares.
    engine.update(ctx_size=tiers[0], cache_type_k="f16", cache_type_v="f16", flash_attn="on", batch_size=batch,
                  ubatch_size=min(int(engine.get("ubatch_size", 512)), batch), spec_type="none", reasoning="off",
                  n_gpu_layers=gpu_layers[0], kv_offload=kv_offload[0])
    raw_base["requested_input_tokens"] = fill
    development_seeds = {row.get("seed", 42) for row in raw_base["benchmarks"] if row["benchmark_id"] == "niah"}
    if holdout_seed in development_seeds:
        raise ValueError("holdout seed collides with the development NIAH seed")
    niah = next((row["task_ids"] for row in raw_base["benchmarks"] if row["benchmark_id"] == "niah"), None)
    if not niah:
        raise ValueError("the registry offers no runnable NIAH benchmark; a holdout cannot be planned")
    session: dict[str, Any] = {
        "schema_version": 1, "session_id": session_id_for(assets[0].model_name), "image_bundle": image_bundle,
        "assets": [asset.model_dump(mode="json") for asset in assets], "base": raw_base,
        "search": {"quantizations": quantizations, "kv_pairs": [list(pair) for pair in DEFAULT_KV_PAIRS],
                   "ctx_tiers": list(tiers), "spec_types": list(spec_types), "reasoning": list(DEFAULT_REASONING),
                   "batch_sizes": [batch], "gpu_layers": list(gpu_layers), "kv_offload": list(kv_offload),
                   **context_search},
        "holdout": {"niah_task_ids": niah, "niah_seed": holdout_seed},
        # An explicit context range is a request for long candidates: give the session a per-candidate wall
        # their filled prompts fit in. The public-benchmark plan above was sized on the unraised wall on purpose,
        # so a long ceiling does not quietly enlarge every other candidate's workload.
        "budgets": (candidate_wall_for_tiers(base, tiers, budgets, ceiling=context_search["context_ceiling"])
                    if context_search else budgets).model_dump(mode="json"),
        "proposal_mode": "deterministic", "proposal_file": None,
        "policy": {"schema_version": 1, "name": session_id_for(assets[0].model_name), "mode": "live"},
    }
    return ContainerSessionConfig.model_validate_json(canonical_json(session))
