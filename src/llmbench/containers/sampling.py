"""Bounded, descriptive generation experiments, separate from serving-setting tuning."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from ..config import GenerationSettings, canonical_json
from ..evaluations.selection import expected_task_rows
from ..safety import OperationForbidden, SessionLock
from ..store import utc_now
from .campaign import read_indexed_artifact
from .config import ContainerRunConfig, NativeBundle, read_run_config
from .runner import INGEST_MARGIN_SECONDS
from .session import authorize_config, read_bundle, write_atomic_json, write_exclusive_json

DEFAULT_TEMPERATURES = (0.6, 0.8)
DEFAULT_SEEDS = (42, 43, 44)
MAX_ATTEMPTS = 128
REPORT_NAME = "sampling-report.json"


def add_sampling_commands(subparsers) -> None:
    command = subparsers.add_parser("sample", help="Compare generation sampling on fixed benchmark items")
    command.add_argument("--config", required=True, help="ContainerRunConfig JSON; engine settings are inherited")
    command.add_argument("--output", required=True)
    command.add_argument("--temperature", action="append", type=float,
                         help="Repeatable stochastic temperature in (0, 5]; defaults: 0.6, 0.8")
    command.add_argument("--seed", action="append", type=int, help="Repeatable nonnegative generation seed; defaults: 42, 43, 44")
    command.add_argument("--top-p", type=float, default=0.95)
    command.add_argument("--budget-seconds", required=True, type=float, help="Whole experiment wall budget, including preparation")
    command.add_argument("--policy", default="runtime-policy.json")
    command.add_argument("--capabilities-dir", default="artifacts/container-prep")
    command.add_argument("--image-bundle")
    command.add_argument("--native-bundle", help="native-bundle.json from `prepare --runtime metal-native`; the "
                                                 "config's native server pins must match (metal-native only)")
    command.add_argument("--plan-only", action="store_true", help="Persist an immutable plan without policy authorization or inference")


def _sampling_grid(temperatures, seeds, top_p: float) -> list[dict]:
    temperatures = tuple(DEFAULT_TEMPERATURES if temperatures is None else temperatures)
    seeds = tuple(DEFAULT_SEEDS if seeds is None else seeds)
    if not temperatures or not seeds:
        raise ValueError("sampling needs at least one stochastic temperature and one seed")
    if type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 5
           for value in temperatures):
        raise ValueError("stochastic temperatures must be finite and in (0, 5]; the greedy control is added automatically")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("generation seeds must be nonnegative integers")
    temperatures, seeds = tuple(dict.fromkeys(temperatures)), tuple(dict.fromkeys(seeds))
    if 1 + len(temperatures) * len(seeds) > MAX_ATTEMPTS:
        raise ValueError(f"sampling grid exceeds {MAX_ATTEMPTS} attempts including the greedy control")
    rows = [{"role": "greedy-control", "temperature": 0.0, "top_p": 1.0, "seed": seeds[0]}]
    rows.extend({"role": "stochastic", "temperature": float(temperature), "top_p": float(top_p), "seed": seed}
                for temperature in temperatures for seed in seeds)
    return rows


def _minimum_grant(config: ContainerRunConfig) -> int:
    minimum = config.bounds.minimum_wall_seconds() + 1
    if config.evaluator.mode == "container":
        # Mirror the runner's child grant: both verification passes and ingestion must fit before cleanup.
        minimum += config.bounds.verify_seconds + INGEST_MARGIN_SECONDS
    return minimum


def _identity(row: dict) -> tuple:
    return row.get("suite"), row.get("task_id"), row.get("split")


def _metrics(rows: list[dict], *, measured: bool) -> dict:
    completed = [row for row in rows if row.get("status") == "completed"]
    output_known = [row for row in rows if isinstance(row.get("output_cap_hit"), bool)
                    or isinstance(row.get("finish_reason"), str)]
    input_known = [row for row in rows if isinstance(row.get("truncation_reported"), bool)
                   or isinstance(row.get("input_truncated"), bool)]
    return {"score": sum(float(row["score"]) if row.get("status") == "completed" else 0.0
                          for row in rows) / len(rows) if measured and rows else None,
            "item_count": len(rows), "completed_item_count": len(completed) if measured else None,
            "failed_item_count": len(rows) - len(completed) if measured else None,
            "output_truncation_count": sum(row.get("output_cap_hit") is True or row.get("finish_reason") == "length"
                                           for row in output_known) if measured and output_known else None,
            "output_truncation_observed_items": len(output_known) if measured else 0,
            "input_truncation_count": sum(row.get("truncation_reported") is True or row.get("input_truncated") is True
                                          for row in input_known) if measured and input_known else None,
            "input_truncation_observed_items": len(input_known) if measured else 0}


def _scores(rows: list[dict], *, measured: bool) -> dict:
    return {axis: {key: _metrics([row for row in rows if row[field] == key], measured=measured)
                   for key in sorted({row[field] for row in rows})}
            for axis, field in (("by_suite", "suite"), ("by_category", "category"))}


def _read_scores(config: ContainerRunConfig, run_dir: Path, expected: list[dict]) -> dict:
    evaluation = json.loads(read_indexed_artifact(run_dir, "evaluation.json", max_bytes=config.limits.max_artifact_bytes))
    rows = evaluation.get("samples") if isinstance(evaluation, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("evaluation.json lacks sample rows")
    if evaluation.get("abort_reason"):
        raise ValueError("evaluation aborted: " + str(evaluation["abort_reason"]))
    if Counter(_identity(row) for row in rows) != Counter(_identity(row) for row in expected):
        raise ValueError("evaluation item set differs from the immutable sampling plan")
    categories = {_identity(row): row["category"] for row in expected}
    for row in rows:
        if row.get("category") != categories[_identity(row)]:
            raise ValueError("evaluation item category differs from the sampling plan")
        if row.get("status") == "completed":
            score = row.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("completed item score must be finite and in [0, 1]")
    return _scores(rows, measured=True)


def _seed_summary(attempts: list[dict], grid: list[dict], expected: list[dict]) -> list[dict]:
    groups = {}
    for planned in grid:
        if planned["role"] == "stochastic":
            groups.setdefault((planned["temperature"], planned["top_p"]), []).append(planned["seed"])
    denominators = _scores(expected, measured=False)
    result = []
    for (temperature, top_p), planned_seeds in groups.items():
        attempts_in_group = [row for row in attempts if row["role"] == "stochastic"
                             and row["temperature"] == temperature and row["top_p"] == top_p
                             and row.get("state") != "not-started"]
        measured = [row for row in attempts_in_group if row["evidence"] == "measured-descriptive"]
        group_complete = (len(measured) == len(planned_seeds)
                          and {row["seed"] for row in measured} == set(planned_seeds))
        scores = {}
        for axis in ("by_suite", "by_category"):
            scores[axis] = {}
            for name, denominator in denominators[axis].items():
                values = [row["scores"][axis][name]["score"] for row in measured]
                values = [value for value in values if value is not None]
                complete = group_complete and len(values) == len(planned_seeds)
                scores[axis][name] = {"mean": statistics.fmean(values) if complete else None,
                                      "minimum": min(values) if complete else None,
                                      "maximum": max(values) if complete else None,
                                      "population_stddev": statistics.pstdev(values) if complete else None,
                                      "measured_seed_count": len(values),
                                      "unique_benchmark_items": denominator["item_count"]}
        result.append({"temperature": temperature, "top_p": top_p,
                       "planned_seeds": planned_seeds, "group_complete": group_complete,
                       "attempted_seeds": [row["seed"] for row in attempts_in_group],
                       "measured_seeds": [row["seed"] for row in measured], "scores": scores})
    return result


def _run_sampling(config: ContainerRunConfig, output: str | Path, *, temperatures=None, seeds=None,
                  top_p: float = 0.95, budget_seconds: float, runner=None, plan_only: bool = False,
                  clock=time.monotonic, started_clock: float | None = None, provenance: dict | None = None) -> dict:
    started = clock() if started_clock is None else started_clock
    if type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and positive")
    grid = _sampling_grid(temperatures, seeds, top_p)
    expected = expected_task_rows(config.benchmarks)
    if len({_identity(row) for row in expected}) != len(expected):
        raise ValueError("sampling requires unambiguous suite/task/split identities")
    candidates = []
    for index, row in enumerate(grid):
        raw = config.model_dump(mode="json")
        raw["generation"] = GenerationSettings.model_validate({**raw["generation"],
                                                               **{key: row[key] for key in ("temperature", "top_p", "seed")}}).model_dump(mode="json")
        raw["label"] = f"sample-{index:03d}-{row['role']}"
        candidates.append(ContainerRunConfig.model_validate_json(canonical_json(raw)))
    minimum = _minimum_grant(config)
    root = Path(output).resolve()
    # A new directory prevents overwriting an earlier experiment or claiming an unfinished one was resumed.
    root.mkdir(parents=True, exist_ok=False)
    write_exclusive_json(root / "sampling-config.json", config.model_dump(mode="json"))
    plan = {"schema_version": 1, "experiment": "generation-sampling", "created_utc": utc_now(),
            "base_fingerprint": config.fingerprint(), "budget_seconds": budget_seconds,
            "minimum_candidate_grant_seconds": minimum,
            "benchmark_items": expected, "unique_benchmark_items": len(expected),
            "benchmark_selection_sha256": hashlib.sha256(canonical_json([item.model_dump(mode="json") for item in config.benchmarks]).encode()).hexdigest(),
            "engine_reasoning": config.engine.reasoning,
            "provenance": provenance or {},
            "candidates": [{**row, "index": index, "config": candidate.model_dump(mode="json"),
                            "config_fingerprint": candidate.fingerprint(), "output": f"attempts/{index:03d}"}
                           for index, (row, candidate) in enumerate(zip(grid, candidates))]}
    write_exclusive_json(root / "sampling-plan.json", plan)
    report = {"schema_version": 1, "experiment": "generation-sampling", "state": "planned" if plan_only else "running",
              "evidence": "descriptive-only", "execution_performed": False, "attempts": [],
              "plan": "sampling-plan.json", "plan_sha256": hashlib.sha256(canonical_json(plan).encode()).hexdigest(),
              "stop_reason": None, "elapsed_seconds": max(0.0, clock() - started),
              "unattempted_indices": list(range(len(grid))), "seed_summary": [],
              "interpretation": ["Separate from controlled serving-setting tuning; no winner or validated preset is selected.",
                                 "Generation seeds repeat the same benchmark items, not new independent items.",
                                 "Seed means, ranges and population standard deviations are descriptive; no confidence intervals, significance or noninferiority claims.",
                                 "Seed-group aggregates are null until every planned seed is measured; successful-seed subsets are not averaged.",
                                 "Failed, aborted, synthetic or unverified attempts have no measured scores; successful attempt item failures remain in the denominator as zero.",
                                 "Null truncation counts mean unavailable evidence, not zero truncations.",
                                 "All non-grid generation settings and engine reasoning are inherited from the candidate."]}

    def persist() -> None:
        report["elapsed_seconds"] = max(0.0, clock() - started)
        report["seed_summary"] = _seed_summary(report["attempts"], grid, expected)
        write_atomic_json(root / REPORT_NAME, report)

    persist()
    if plan_only:
        return report
    if runner is None:
        raise ValueError("a policy-authorized ContainerRunner is required for execution")
    for index, (row, candidate) in enumerate(zip(grid, candidates)):
        remaining = budget_seconds - (clock() - started)
        grant = min(remaining, candidate.bounds.candidate_wall_seconds,
                    candidate.parent_grant_seconds or candidate.bounds.candidate_wall_seconds)
        if grant < minimum:
            report.update(state="stopped", stop_reason="insufficient-budget-for-candidate-bounds")
            break
        attempt = {**row, "index": index, "output": f"attempts/{index:03d}",
                   "config_fingerprint": candidate.fingerprint(), "state": "running", "attempt_id": None,
                   "granted_seconds": grant, "evidence": "unmeasured", "failure_reasons": [],
                   "cleanup_verified": None, "abort_campaign": False, "synthetic": None,
                   "scores": _scores(expected, measured=False)}
        report["attempts"].append(attempt)
        report["unattempted_indices"].remove(index)
        previously_executed = report["execution_performed"]
        report["execution_performed"] = True if previously_executed else None
        persist()  # If interrupted, the attempt remains discoverable before any server can be started.
        remaining = budget_seconds - (clock() - started)
        grant = min(grant, remaining)
        if grant < minimum:
            attempt.update(state="not-started", failure_reasons=["budget expired while persisting the attempt"])
            report["execution_performed"] = previously_executed
            report["unattempted_indices"].insert(0, index)
            report.update(state="stopped", stop_reason="insufficient-budget-for-candidate-bounds")
            break
        attempt["granted_seconds"] = grant
        report["execution_performed"] = True
        try:
            result = runner.run(candidate, root / attempt["output"], remaining_budget_seconds=grant)
        except (Exception, KeyboardInterrupt) as exc:
            # No returned terminal cleanup evidence: even a pre-start exception must stop further GPU work.
            attempt.update(state="cancelled" if isinstance(exc, KeyboardInterrupt) else "exception",
                           failure_reasons=[f"{type(exc).__name__}: {exc}"], cleanup_verified=False, abort_campaign=True)
            report.update(state="aborted", stop_reason="runner-exception-cleanup-unverified")
            persist()
            return report
        attempt.update(state=result.state, attempt_id=result.attempt_id, synthetic=result.synthetic,
                       failure_stage=result.failure_stage, failure_reasons=list(result.failure_reasons),
                       warnings=list(result.warnings), cleanup_verified=result.cleanup.verified,
                       abort_campaign=result.abort_campaign, elapsed_seconds=result.elapsed_seconds,
                       effective_settings_verified=result.effective_settings_verified,
                       actual_context_verified=result.actual_context_verified)
        if result.abort_campaign or not result.cleanup.verified:
            report.update(state="aborted", stop_reason="cleanup-uncertain")
            persist()
            return report
        if result.state == "cancelled":
            report.update(state="stopped", stop_reason="candidate-cancelled")
            persist()
            return report
        if (result.state == "completed" and not result.synthetic and result.effective_settings_verified
                and result.actual_context_verified):
            try:
                if result.config_fingerprint != candidate.fingerprint():
                    raise ValueError("runner result configuration differs from the sampling plan")
                attempt["scores"] = _read_scores(candidate, root / attempt["output"], expected)
                attempt["evidence"] = "measured-descriptive"
            except (ValueError, OSError, TypeError, KeyError) as exc:
                attempt["failure_reasons"].append(f"score-evidence-unavailable: {type(exc).__name__}: {exc}")
        if attempt["evidence"] == "unmeasured" and not attempt["failure_reasons"]:
            attempt["failure_reasons"].append("synthetic or settings/context verification missing")
        persist()
    else:
        report.update(state="completed" if all(row["evidence"] == "measured-descriptive" for row in report["attempts"])
                      else "completed-with-unmeasured-attempts", stop_reason="grid-exhausted")
    persist()
    return report


def _checked_bundle(args, config: ContainerRunConfig):
    """The prepared bundle that pins ``config``, compared pin by pin; None when none was given.

    Each runtime is pinned by its own kind of bundle (``--image-bundle`` for NVIDIA images, ``--native-bundle``
    for a metal-native server), and the other kind is refused rather than silently not compared: an image bundle
    has nothing to say about a native server, so comparing would find no disagreement and pass.
    """
    from .cli import check_image_bundle, check_native_bundle
    image_path, native_path = args.image_bundle, getattr(args, "native_bundle", None)
    if image_path and native_path:
        raise ValueError("pass one bundle: --image-bundle (nvidia-container) or --native-bundle (metal-native)")
    if not image_path and not native_path:
        return None
    bundle = read_bundle(native_path or image_path)
    if native_path:
        if not isinstance(bundle, NativeBundle):
            raise ValueError(f"--native-bundle {native_path} is an image bundle (NVIDIA images)")
        problems = check_native_bundle(config, bundle)
        if problems:
            raise ValueError("config native server differs from the prepared bundle: " + "; ".join(problems))
        return bundle
    if isinstance(bundle, NativeBundle):
        raise ValueError(f"--image-bundle {image_path} is a native bundle; pass it as --native-bundle")
    problems = check_image_bundle(config, bundle)
    if problems:
        raise ValueError("config images differ from the prepared bundle: " + "; ".join(problems))
    return bundle


def _default_runner(args):
    """Dispatches on the config's runtime; an NVIDIA config gets exactly
    ``ContainerRunner(capabilities_dir=..., policy_path=...)`` as before, built when its first attempt runs."""
    from .runtime import DispatchRunner
    return DispatchRunner(capabilities_dir=args.capabilities_dir, policy_path=args.policy)


def main_sampling(args, *, runner_factory=None, clock=time.monotonic) -> int:
    started = clock()
    try:
        config = read_run_config(args.config)
        bundle = _checked_bundle(args, config)
        runner = None
        if not args.plan_only:
            lock = SessionLock.read(args.policy)
            authorize_config(lock, config)  # the config's runtime decides; NVIDIA: container, load, inference
            runner = runner_factory(args) if runner_factory else _default_runner(args)
        provenance = {"policy": str(args.policy), "capabilities_dir": str(args.capabilities_dir),
                      "image_bundle": None if bundle is None or isinstance(bundle, NativeBundle)
                      else bundle.model_dump(mode="json")}
        if isinstance(bundle, NativeBundle):  # a key of its own; an NVIDIA plan's provenance is unchanged
            provenance["native_bundle"] = bundle.model_dump(mode="json")
        report = _run_sampling(config, args.output, temperatures=args.temperature, seeds=args.seed,
                               top_p=args.top_p, budget_seconds=args.budget_seconds, runner=runner,
                               plan_only=args.plan_only, clock=clock, started_clock=started,
                               provenance=provenance)
        print(json.dumps({"state": report["state"], "stop_reason": report["stop_reason"],
                          "attempts": len(report["attempts"]), "execution_performed": report["execution_performed"],
                          "output": str(args.output), "report": REPORT_NAME}, indent=2))
        return 0 if report["state"] in ("planned", "completed") else 4 if report["state"] == "aborted" else 3
    except (ValueError, KeyError, OSError, OperationForbidden) as exc:
        print(f"llmbench sample: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
