"""Serial, resumable orchestration. Executors supply observations; they never select winners."""

from __future__ import annotations

import hashlib
import json
import time
import tempfile
from pathlib import Path
from typing import Callable

from .config import CampaignPolicy, ExperimentManifest, RunMode, canonical_json
from .measurement import SpeedObservation, summarize_speed
from .provenance import environment_record
from .safety import SessionLock
from .search import BudgetTracker, assess_candidate, pareto_frontier, propose
from .store import Store


def generate_candidates(base: ExperimentManifest, proposals: list[dict]) -> list[ExperimentManifest]:
    candidates = [base]
    seen = {base.fingerprint()}
    for proposal in proposals:
        if set(proposal) != {"family", "changes"}:
            raise ValueError("proposal requires exactly family and changes")
        candidate = propose(base, proposal["changes"], proposal["family"])
        if candidate.fingerprint() not in seen:
            candidates.append(candidate)
            seen.add(candidate.fingerprint())
    return candidates


def synthetic_executor(manifest: ExperimentManifest, *, timeout_seconds: float) -> dict:
    """Exercise genuine scorers on deterministic fixtures, never a model endpoint."""
    from .evaluations import run_demo_suite
    from .coding.fixtures import fixtures, score_observations
    demo = run_demo_suite()
    cases = []
    failure_statuses = {"timeout", "transport_error", "environment_error", "unsupported", "cancelled"}
    for row in demo["cases"]:
        cases.append({**row, "category": row.get("category", row.get("suite", "tools")),
                      "outcome_status": row["status"],
                      "status": row["status"] if row["status"] in failure_statuses else "completed",
                      "fixture_hash": hashlib.sha256(row["task_id"].encode()).hexdigest(), "split": "development"})
    for fixture in fixtures():
        observations = {case.case_id: json.loads(case.expected_json) for case in fixture.checks}
        score = score_observations(fixture, observations, compile_ok=True, execution_ok=True)
        cases.append({"task_id": fixture.fixture_id, "category": "coding", "score": float(score.success),
                      "status": "completed", "passed": score.success, "fixture_hash": fixture.identity(),
                      "split": "development", "synthetic": True, "model_evaluated": False,
                      "details": "Synthetic evaluator observations only; no candidate programs executed."})
    # Synthetic values are deliberately attractive: eligibility must still reject them.
    rate = {"f16": 65., "q8_0": 90., "q4_0": 110.}.get(manifest.backend.k_cache, 70.)
    length = manifest.generation.max_output_tokens
    duration = length / rate
    observations = [SpeedObservation(
        status="completed", finish_reason="length", output_tokens=length,
        native_generation_seconds=duration, elapsed_seconds=duration + .1, first_event_seconds=.1,
        content_event_times=tuple(.1 + duration * i / 10 for i in range(11)),
        requested_output_tokens=length, input_tokens=manifest.requested_input_tokens,
        expected_input_tokens=manifest.requested_input_tokens, accepted_tokens_verified=False,
        synthetic=True) for _ in range(5)]
    return {"synthetic": True, "model_evaluated": False, "samples": cases, "speed_observations": observations,
            "effective_settings_verified": False, "actual_context_verified": False,
            "holdout_passed": False, "raw": demo, "comparisons": {}}


def execute_attempt(store: Store, manifest: ExperimentManifest, policy: CampaignPolicy,
                    executor: Callable, *, session_lock: SessionLock | None = None,
                    parent_id=None, timeout_seconds=None) -> dict:
    if manifest.mode != policy.mode:
        raise ValueError("manifest and campaign modes must agree")
    if manifest.mode == RunMode.DRY_RUN:
        return {"planned": True, "manifest": manifest.model_dump(mode="json")}
    if manifest.mode == RunMode.LIVE:
        (session_lock or SessionLock()).check("inference", RunMode.LIVE)
        if manifest.environment_hash != environment_record()["sha256"]:
            raise ValueError("live manifest does not match this exact environment")
    elif executor is not synthetic_executor:
        raise ValueError("offline mode only accepts the built-in synthetic executor")
    synthetic = manifest.mode != RunMode.LIVE
    attempt_id = store.create_attempt(manifest, synthetic=synthetic, parent_id=parent_id)
    store.transition(attempt_id, "running")
    started = time.monotonic()
    try:
        store.artifact(attempt_id, "environment.json", canonical_json(environment_record()).encode())
        if hasattr(executor, "bind_artifact_sink"):
            executor.bind_artifact_sink(lambda name, content: store.artifact(attempt_id, name, content))
        if hasattr(executor, "bind_sample_sink"):
            executor.bind_sample_sink(lambda sample: store.add_sample(attempt_id, sample))
        with store.trace(attempt_id) as trace_sink:
            if hasattr(executor, "bind_event_sink"):
                executor.bind_event_sink(trace_sink)
            result = executor(manifest, timeout_seconds=timeout_seconds or policy.budgets.task_timeout_seconds)
        # Persist exact raw evidence before any scoring, normalization or sample insertion.
        store.artifact(attempt_id, "raw.json", canonical_json(result.get("raw", {})).encode())
        if result.get("synthetic") is not synthetic:
            raise ValueError("executor returned inconsistent synthetic provenance")
        samples = result["samples"]
        if result.get("samples_persisted") is not True:
            for sample in samples:
                store.add_sample(attempt_id, sample)
        speed = summarize_speed(result.get("speed_observations", []), policy.acceptance)
        quality = {}
        for category in ("coding", "tools", "retrieval"):
            rows = [sample for sample in samples if sample.get("category") == category]
            scores = [float(row.get("score", 0)) if row.get("status") == "completed" else 0. for row in rows]
            quality[category] = {"attempted": len(rows), "score": sum(scores) / len(scores) if scores else None,
                                 "comparison": result.get("comparisons", {}).get(category, {"status": "unmeasured"})}
        evidence = {"attempt_id": attempt_id, "manifest_hash": manifest.fingerprint(),
                    "policy_hash": hashlib.sha256(canonical_json(policy.model_dump(mode="json")).encode()).hexdigest(),
                    "status": "completed", "synthetic": synthetic, "model_evaluated": not synthetic,
                    "speed": speed, "quality": quality,
                    "actual_input_tokens": manifest.requested_input_tokens,
                    "effective_settings_verified": result.get("effective_settings_verified") is True,
                    "actual_context_verified": result.get("actual_context_verified") is True,
                    "holdout_passed": result.get("holdout_passed") is True,
                    "elapsed_seconds": time.monotonic() - started}
        evidence["eligibility"] = assess_candidate(evidence, policy)
        store.artifact(attempt_id, "evidence.json", canonical_json(evidence).encode())
        store.event(attempt_id, "evidence", evidence)
        store.transition(attempt_id, "completed")
        return evidence
    except (Exception, KeyboardInterrupt) as exc:
        state = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        store.event(attempt_id, "error", {"type": type(exc).__name__, "message": str(exc)})
        store.transition(attempt_id, state, str(exc))
        if isinstance(exc, KeyboardInterrupt):
            raise
        return {"attempt_id": attempt_id, "manifest_hash": manifest.fingerprint(), "status": state,
                "synthetic": synthetic, "error": str(exc), "abort_campaign": getattr(exc, "abort_campaign", False) is True,
                "eligibility": {"eligible": False, "reasons": [state]}}


def _measured(row: dict) -> list[dict]:
    """The quality blocks of the categories this candidate declared tasks for; the rest are not judged."""
    from .search import unmeasured_categories
    skipped = unmeasured_categories(row["quality"])
    return [block for name, block in row["quality"].items() if name not in skipped]


def run_campaign(store: Store, manifests: list[ExperimentManifest], policy: CampaignPolicy,
                 *, executor=synthetic_executor, session_lock=None, resume=False, holdout_selections=()) -> dict:
    if not manifests:
        raise ValueError("campaign needs at least one candidate")
    tracker = BudgetTracker(policy.budgets)
    results, skipped = [], []
    identity = hashlib.sha256(canonical_json({"policy": policy.model_dump(mode="json"),
                                            "manifests": [item.fingerprint() for item in manifests],
                                            "holdout": [item.model_dump(mode="json") for item in holdout_selections]
                                            }).encode()).hexdigest()
    policy_hash = hashlib.sha256(canonical_json(policy.model_dump(mode="json")).encode()).hexdigest()
    resource_lock = Path(tempfile.gettempdir()) / "llmbench-gpu-resource.lock" if policy.mode == RunMode.LIVE else None
    with store.campaign_lock(resource_lock):
        if resume:
            checkpoint = store.campaign_state(identity)
            tracker.started -= checkpoint["elapsed_seconds"]
            tracker.candidates = checkpoint["candidates"]
        for manifest in manifests:
            previous = [row for row in store.attempts() if row["experiment_id"] == manifest.fingerprint()]
            if resume and previous and previous[-1]["state"] == "completed":
                events = store.results(previous[-1]["id"])["events"]
                import json
                saved = [json.loads(row["payload"]) for row in events if row["kind"] == "evidence"]
                if saved and saved[-1].get("policy_hash") == policy_hash:
                    results.append(saved[-1])
                    continue
            if not tracker.admit(policy.budgets.task_timeout_seconds):
                skipped.append({"manifest_hash": manifest.fingerprint(), "reason": "search_budget_reserved_or_exhausted"})
                continue
            # Reserve the full granted duration durably; a crash cannot refund it.
            store.checkpoint_campaign(identity, policy.budgets.wall_seconds - tracker.remaining() + policy.budgets.task_timeout_seconds, tracker.candidates)
            parent = None
            if resume and previous:
                last = previous[-1]
                if last["state"] == "running":
                    store.transition(last["id"], "interrupted", "explicit resume after exclusive campaign lock acquired")
                    last["state"] = "interrupted"
                if last["state"] in {"interrupted", "failed", "cancelled"}:
                    parent = last["id"]
            try:
                result = execute_attempt(store, manifest, policy, executor, session_lock=session_lock, parent_id=parent,
                                         timeout_seconds=min(policy.budgets.task_timeout_seconds, tracker.remaining()))
            finally:
                store.checkpoint_campaign(identity, policy.budgets.wall_seconds - tracker.remaining(), tracker.candidates)
            results.append(result)
            if result.get("abort_campaign") is True:
                break
            if policy.search_mode == "good-enough" and result.get("eligibility", {}).get("eligible") is True:
                break
        if holdout_selections and results and not any(row.get("abort_campaign") is True for row in results):
            from .analysis import analyze_campaign, plan_holdout_candidates
            screening = analyze_campaign(store, {"results": results}, policy)
            survivors = [row for row in screening["results"] if row.get("speed", {}).get("qualifies") is True
                         and row.get("effective_settings_verified") is True
                         and row.get("actual_context_verified") is True
                         and not row.get("analysis_errors") and _measured(row) and all(
                             quality["absolute_passed"] and quality["comparison"]["status"] == "noninferior"
                             for quality in _measured(row))]
            survivors.sort(key=lambda row: (*((row["quality"][category]["score"] or 0.0) for category in
                                               ("coding", "tools", "retrieval")),
                                            row["speed"]["minimum_native_tps"]), reverse=True)
            selected_ids = [row["attempt_id"] for row in survivors[:policy.budgets.max_validation_candidates]]
            if selected_ids:
                selected_ids.insert(0, results[0]["attempt_id"])
                selected = [ExperimentManifest.model_validate_json(canonical_json(store.results(key)["manifest"]))
                            for key in dict.fromkeys(selected_ids)]
                for heldout in plan_holdout_candidates(selected, holdout_selections):
                    # A holdout that already completed under this policy is reused on resume, exactly like a
                    # development attempt. It used to run again every time: unreachable while no candidate could
                    # survive screening, and a wasted GPU-hour per resume once one could.
                    done = [row for row in store.attempts() if row["experiment_id"] == heldout.fingerprint()
                            and row["state"] == "completed"]
                    if resume and done:
                        import json
                        saved = [json.loads(row["payload"]) for row in store.results(done[-1]["id"])["events"]
                                 if row["kind"] == "evidence"]
                        if saved and saved[-1].get("policy_hash") == policy_hash:
                            results.append(saved[-1])
                            continue
                    if not tracker.admit(policy.budgets.task_timeout_seconds, validation=True):
                        skipped.append({"manifest_hash": heldout.fingerprint(), "reason": "validation_budget_exhausted"})
                        break
                    store.checkpoint_campaign(identity, policy.budgets.wall_seconds - tracker.remaining() + policy.budgets.task_timeout_seconds, tracker.candidates)
                    try:
                        validation_result = execute_attempt(store, heldout, policy, executor, session_lock=session_lock,
                            timeout_seconds=min(policy.budgets.task_timeout_seconds, tracker.remaining()))
                    finally:
                        store.checkpoint_campaign(identity, policy.budgets.wall_seconds - tracker.remaining(), tracker.candidates)
                    results.append(validation_result)
                    if validation_result.get("abort_campaign") is True:
                        break
    frontier = pareto_frontier(results)
    campaign = {"campaign_id": identity, "policy": policy.model_dump(mode="json"), "results": results,
            "skipped": skipped, "frontier": [row["attempt_id"] for row in frontier],
            "winner": None, "automatic_deployment_performed": False,
            "selection_note": "Frontier retains quality/speed/context tradeoffs; a preset requires verified holdout evidence.",
            "remaining_wall_seconds": tracker.remaining()}
    if policy.mode != RunMode.DRY_RUN:
        from .analysis import analyze_campaign
        return analyze_campaign(store, campaign, policy)
    return campaign
