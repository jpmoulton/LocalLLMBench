"""Bounded screening, measured serving interactions, then a larger finalist comparison.

Runtime-agnostic: every stage is a ``session.tune`` of a derived stage session, so the candidates' own runtime
(NVIDIA container or metal-native) decides which runner serves them (``runtime.DispatchRunner``) and which
``runtime-policy.json`` permissions the whole optimization needs (``session.authorize_session``).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

from ..config import canonical_json
from ..safety import OperationForbidden, SessionLock
from .proposals import (FAMILY_AXES, Proposal, apply, deterministic_schedule, load_proposal_file)
from .session import (ContainerSessionConfig, authorize_session, policy_for, read_bundle, read_session_config, tune,
                      write_atomic_json, write_exclusive_json)


def screening_session(session: ContainerSessionConfig, items: int) -> ContainerSessionConfig:
    """Keep a deterministic nested subset of each declared suite; never invent different fixtures."""
    if items < 1:
        raise ValueError("screen items must be positive")
    raw = session.model_dump(mode="json")
    for selection in raw["base"]["benchmarks"]:
        selection["task_ids"] = selection["task_ids"][:items]
    return ContainerSessionConfig.model_validate_json(canonical_json(raw))


def shortlist(outcomes: list[dict], proposals: list[Proposal], limit: int) -> list[Proposal]:
    """Retain speed and quality extremes, without calling screening differences significant."""
    by_label = {proposal.label(): proposal for proposal in proposals if proposal.family != "baseline"}
    rows = {}
    for outcome in outcomes:
        labels = {row["attempt_id"]: row["label"] for row in outcome["report"]["rows"]
                  if row["split"] == "development"}
        for row in outcome["campaign"].get("results", []):
            label = labels.get(row.get("attempt_id"))
            quality = [q for q in row.get("quality", {}).values()
                       if isinstance(q, dict) and q.get("attempted")]
            speed = row.get("speed", {}).get("minimum_native_tps")
            if (label not in by_label or row.get("status") != "completed" or row.get("synthetic") is not False
                    or row.get("effective_settings_verified") is not True
                    or row.get("actual_context_verified") is not True
                    or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0
                    or not quality or not all(q.get("absolute_passed") is True for q in quality)):
                continue
            scores = [q["score"] for q in quality if isinstance(q.get("score"), (int, float))]
            if len(scores) != len(quality) or not all(math.isfinite(score) for score in scores):
                continue
            rows[label] = (speed, min(scores), sum(scores) / len(scores))
    fastest = sorted(rows, key=lambda label: (-rows[label][0], label))
    quality = sorted(rows, key=lambda label: (-rows[label][1], -rows[label][2], label))
    ordered = []
    for candidates in itertools.zip_longest(fastest, quality):
        for label in candidates:
            if label is not None and label not in ordered:
                ordered.append(label)
    return [by_label[label] for label in ordered[:limit]]


def interaction_proposals(session: ContainerSessionConfig, baseline: Proposal,
                          finalists: list[Proposal], limit: int) -> list[Proposal]:
    """Measure bounded pairwise combinations, not predicted additive speedups.

    Weights and context stay fixed: neither a different model nor an easier input workload is a serving
    interaction. Conflicting values and invalid combinations are rejected before spending GPU time.
    """
    base = apply(session, session.base, baseline)
    seen = {base.fingerprint(), *(apply(session, base, proposal).fingerprint() for proposal in finalists)}
    combinations = []
    for left, right in itertools.combinations(finalists, 2):
        if any(key in right.changes and value != right.changes[key] for key, value in left.changes.items()):
            continue
        changes = {**left.changes, **right.changes}
        changes = {key: value for key, value in changes.items() if value != baseline.changes[key]}
        if len(changes) < 2 or set(changes) - FAMILY_AXES["combination"]:
            continue
        proposal = Proposal(family="combination", changes=changes,
                            note=f"measured interaction of {left.label()} and {right.label()}")
        try:
            fingerprint = apply(session, base, proposal).fingerprint()
        except ValueError:
            continue
        if fingerprint not in seen:
            seen.add(fingerprint)
            combinations.append(proposal)
            if len(combinations) >= limit:
                break
    return combinations if limit > 0 else []


def _stage_session(session, root, name, proposals, wall_seconds, *, confirmation=False):
    raw = session.model_dump(mode="json")
    wall = int(wall_seconds)
    reserve = min(session.budgets.holdout_reserve_seconds, wall // 4) if confirmation else 0
    cleanup = session.budgets.cleanup_reserve_seconds
    candidate = min(session.budgets.candidate_wall_seconds, wall - reserve - cleanup)
    if wall < 600 or candidate <= session.base.bounds.minimum_wall_seconds():
        raise ValueError(f"not enough remaining budget for {name}; no candidate started")
    raw["session_id"] = f"{session.session_id[:45]}-{name}"
    raw["budgets"].update(wall_seconds=wall, holdout_reserve_seconds=reserve,
                          candidate_wall_seconds=candidate, max_candidates=len(proposals))
    raw["base"]["bounds"]["candidate_wall_seconds"] = candidate
    raw["proposal_mode"] = "file"
    raw["proposal_file"] = str(root / f"{name}-proposals.json")
    return ContainerSessionConfig.model_validate_json(canonical_json(raw))


def optimization_plan(session, *, screen_items=8, finalists=2, max_combinations=4, budget_seconds=None):
    if session.base.generation.temperature != 0:
        raise ValueError("controlled setting optimization requires temperature 0; use sample for sampling")
    if not 1 <= screen_items <= 10000 or not 1 <= finalists <= 16 or not 0 <= max_combinations <= 16:
        raise ValueError("screen-items must be 1..10000, finalists 1..16, max-combinations 0..16")
    proposals = (load_proposal_file(session.proposal_file, session) if session.proposal_mode == "file"
                 else deterministic_schedule(session))
    wall = session.budgets.wall_seconds if budget_seconds is None else budget_seconds
    screen_wall = int(wall * (0.4 if max_combinations else 0.5))
    interaction_wall = int(wall * 0.2) if max_combinations else 0
    confirmation_wall = wall - screen_wall - interaction_wall
    screen = screening_session(session, screen_items)
    for name, base, seconds in (("screen", screen, screen_wall),
                                ("confirm", session, confirmation_wall)):
        _stage_session(base, Path("."), name, proposals, seconds, confirmation=name == "confirm")
    if max_combinations:
        _stage_session(screen, Path("."), "interactions", proposals[:1], interaction_wall)
    return {"schema_version": 1, "budget_seconds": wall, "screen_budget_seconds": screen_wall,
            "interaction_budget_seconds": interaction_wall, "finalists": finalists,
            "max_combinations": max_combinations,
            "screen_proposals": [p.model_dump(mode="json") for p in proposals],
            "screen_items": {s.benchmark_id: len(s.task_ids) for s in screen.base.benchmarks},
            "confirmation_items": {s.benchmark_id: len(s.task_ids) for s in session.base.benchmarks},
            "confirmation_is_holdout": False,
            "notes": ["Confirmation repeats finalists on the full declared development task set; it is not an "
                      "independent holdout. Existing non-inferiority and holdout gates remain unchanged.",
                      "Shortlists alternate measured speed and worst-category quality extremes. Small screening "
                      "differences are not claims of significance.",
                      "Interactions keep model, context and sampling fixed. Every combination is actually run.",
                      "If the full workload equals the screening subset, confirmation adds repetitions, not "
                      "independent benchmark items. Supply a larger task set to improve statistical power."]}


def optimize(session, output, *, screen_items=8, finalists=2, max_combinations=4, budget_seconds=None,
             policy_path="runtime-policy.json", capabilities_dir="artifacts/container-prep",
             runner_factory=None, tune_fn=tune, clock=time.monotonic):
    plan = optimization_plan(session, screen_items=screen_items, finalists=finalists,
                             max_combinations=max_combinations, budget_seconds=budget_seconds)
    lock = SessionLock.read(policy_path)
    # Either kind of bundle; read before the output exists so a policy refusal still leaves nothing behind.
    bundle = read_bundle(session.image_bundle) if session.image_bundle else None
    authorize_session(lock, session, bundle)  # NVIDIA: exactly container, load, inference, as before
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_exclusive_json(root / "optimization-plan.json", plan)
    write_exclusive_json(root / "input-session.json", session.model_dump(mode="json"))
    started = clock()
    report: dict = {"schema_version": 1, "state": "running", "stages": {}, "plan": plan,
                    "recommendation": None, "automatic_deployment_performed": False}
    screen = screening_session(session, screen_items)
    proposals = [Proposal.model_validate_json(canonical_json(p)) for p in plan["screen_proposals"]]
    baseline = proposals[0]

    def run_stage(name, base, candidates, allowance, *, confirmation=False):
        remaining = plan["budget_seconds"] - (clock() - started)
        stage = _stage_session(base, root, name, candidates, min(remaining, allowance),
                               confirmation=confirmation)
        write_exclusive_json(Path(stage.proposal_file), [p.model_dump(mode="json") for p in candidates])
        if runner_factory is None:  # an NVIDIA candidate still gets exactly ContainerRunner(<these three>)
            from .runtime import DispatchRunner
            runner = DispatchRunner(capabilities_dir=capabilities_dir, policy_path=policy_path,
                                    policy=policy_for(stage))
        else:
            runner = runner_factory(stage)
        print(f"optimize: {name}: {len(candidates)} candidates, "
              f"{sum(len(s.task_ids) for s in stage.base.benchmarks)} tasks each", flush=True)
        report["stages"][name] = {"state": "running", "output": str(root / name),
                                 "proposals": [p.model_dump(mode="json") for p in candidates]}
        write_atomic_json(root / "optimization-report.json", report)
        outcome = tune_fn(stage, root / name, runner=runner, session_lock=lock, bundle=bundle,
                          bundle_path=session.image_bundle)
        report["stages"][name].update(state=outcome["stop_reason"], summary=outcome["summary"])
        write_atomic_json(root / "optimization-report.json", report)
        if outcome["summary"].get("abort_campaign"):
            raise RuntimeError(f"{name} aborted; refusing further GPU work")
        return outcome

    try:
        screened = run_stage("screen", screen, proposals, plan["screen_budget_seconds"])
        picks = shortlist([screened], proposals, finalists)
        combinations = interaction_proposals(session, baseline, picks, max_combinations)
        outcomes = [screened]
        if combinations:
            interacted = run_stage("interactions", screen, [baseline, *combinations],
                                   plan["interaction_budget_seconds"])
            outcomes.append(interacted)
        all_proposals = [*proposals, *combinations]
        picks = shortlist(outcomes, all_proposals, finalists)
        confirmed = run_stage("confirm", session, [baseline, *picks], plan["budget_seconds"], confirmation=True)
        report["recommendation"] = confirmed["report"]["recommendation"]
        complete = all(stage["state"] in {"complete", "candidates"} for stage in report["stages"].values())
        report["state"] = "completed" if complete else "stopped-early"
    except KeyboardInterrupt:
        report["state"] = "interrupted"
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        report.update(state="stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = clock() - started
        write_atomic_json(root / "optimization-report.json", report)
    return report


def add_optimization_commands(subparsers):
    parser = subparsers.add_parser("optimize", help="Screen settings, test interactions, confirm finalists")
    parser.add_argument("--config", required=True, help="session config carrying the FULL confirmation workload")
    parser.add_argument("--output", required=True)
    parser.add_argument("--screen-items", type=int, default=8, help="maximum tasks per suite during screening")
    parser.add_argument("--finalists", type=int, default=2, help="non-baseline finalists; baseline is always repeated")
    parser.add_argument("--max-combinations", type=int, default=4)
    parser.add_argument("--budget-seconds", type=int)
    parser.add_argument("--policy", default="runtime-policy.json")
    parser.add_argument("--capabilities-dir", default="artifacts/container-prep")
    parser.add_argument("--plan-only", action="store_true")


def main_optimize(args):
    try:
        session = read_session_config(args.config)
        options = dict(screen_items=args.screen_items, finalists=args.finalists,
                       max_combinations=args.max_combinations, budget_seconds=args.budget_seconds)
        if args.plan_only:
            print(json.dumps(optimization_plan(session, **options), indent=2))
            return 0
        result = optimize(session, args.output, policy_path=args.policy, capabilities_dir=args.capabilities_dir,
                          **options)
        print(json.dumps(result, indent=2))
        return 0 if result["state"] == "completed" else 3
    except KeyboardInterrupt:
        print("llmbench optimize: interrupted; see per-stage reports and verify container cleanup", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, OperationForbidden) as exc:
        print(f"llmbench optimize: {exc}", file=sys.stderr)
        return 2
