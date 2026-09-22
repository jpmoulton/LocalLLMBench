"""Recompute campaign conclusions from stored evidence, never previous verdicts.

Analysis is offline: it reads attempts, appends analysis events, and plans new
manifests. It cannot load a model, execute an evaluation, or deploy a preset.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import math
from typing import Iterable

from .config import CampaignPolicy, ExperimentManifest, TaskSelection, canonical_json
from .measurement import SpeedObservation, summarize_speed
from .provenance import assert_comparable
from .search import assess_candidate, paired_comparison, pareto_frontier


CATEGORIES = ("coding", "tools", "retrieval")


def infer_comparison_family(left: ExperimentManifest, right: ExperimentManifest) -> str:
    a, b = (item.model_dump(mode="json", exclude={"annotations"}) for item in (left, right))
    if a == b:
        assert_comparable(a, b, "kv")  # Still requires recorded environment identity.
        return "reference"
    compatible = []
    for family in ("kv", "weights", "performance", "context", "reasoning"):
        try:
            assert_comparable(a, b, family)
            if family == "weights" and any(a["model"][key] != b["model"][key] for key in
                                             ("source_revision", "tokenizer_hash", "template_hash")):
                raise ValueError("weight comparison changed source/tokenizer/template")
            compatible.append(family)
        except ValueError:
            continue
    if len(compatible) != 1:
        if right.annotations.get("treatment_family") == "combination":
            assert_comparable(a, b, "combination")
            return "combination"
        raise ValueError("comparison must vary exactly one declared family with fixed provenance")
    return compatible[0]


def config_key(manifest: ExperimentManifest) -> str:
    """Identity of a complete configuration, ignoring display annotations and task selections."""
    value = manifest.model_dump(mode="json", exclude={"annotations", "tasks"})
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


_config_key = config_key


def discover_campaign_attempts(store, identifiers: Iterable[str]) -> dict[str, ExperimentManifest]:
    """Resolve declared configurations against all Store attempts in insertion order.

    Task selections do not change configuration identity. Include omitted development
    and holdout attempts for each declared configuration, so reports and export use
    the same contamination and latest-attempt evidence. Unrelated configurations
    remain outside the campaign. Invalid Store manifests fail closed because their
    configuration cannot be established safely.
    """
    requested = set(identifiers)
    if any(not isinstance(key, str) or not key for key in requested):
        raise ValueError("campaign attempt IDs must be nonempty strings")
    stored = {}
    for item in store.attempts_in_insertion_order():
        try:
            stored[item["id"]] = ExperimentManifest.model_validate_json(canonical_json(item["manifest"]))
        except ValueError as exc:
            raise ValueError(f"Store attempt {item['id']} has no valid manifest: {type(exc).__name__}") from exc
    if requested - stored.keys():
        raise ValueError("campaign must name only Store attempts")
    configurations = {config_key(stored[key]) for key in requested}
    return {key: manifest for key, manifest in stored.items()
            if key in requested or config_key(manifest) in configurations}


def _evidence(record: dict) -> dict:
    # Analysis events never become new original evidence, avoiding circular
    # reliance on conclusions from a previous policy or analysis invocation.
    events = [row for row in record["events"] if row["kind"] == "evidence"]
    if len(events) != 1:
        # Evidence is sealed once, before completion. An appended later event never
        # replaces it: the attempt then has no trusted measured evidence (fail closed).
        return {}
    raw = events[-1]["payload"]
    return json.loads(raw) if isinstance(raw, str) else deepcopy(raw)


def _scores(rows: list[dict]) -> tuple[list[float], list[str]]:
    values, errors = [], []
    for row in rows:
        if row.get("status") != "completed":
            values.append(0.0)  # Timeouts and environment failures stay in the denominator.
            continue
        value = row.get("score")
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
            errors.append(f"invalid_completed_score:{row.get('task_id')}")
            continue
        values.append(float(value))
    return values, errors


def _absolute(rows: list[dict], threshold: float) -> dict:
    values, errors = _scores(rows)
    score = sum(values) / len(values) if values and not errors else None
    interval = None
    method = "unmeasured"
    if score is not None:
        n, z = len(values), 1.959963984540054
        if all(value in {0., 1.} for value in values):
            denominator = 1 + z * z / n
            center = (score + z * z / (2 * n)) / denominator
            radius = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denominator
            interval = [max(0., center - radius), min(1., center + radius)]
            method = "wilson-binomial"
        else:
            radius = math.sqrt(math.log(40) / (2 * n))
            interval = [max(0., score - radius), min(1., score + radius)]
            method = "bounded-hoeffding"
    return {"attempted": len(rows), "completed": sum(row.get("status") == "completed" for row in rows),
            "score": score, "threshold": threshold,
            "absolute_passed": score is not None and score >= threshold,
            "absolute_95": interval, "absolute_interval_method": method,
            "absolute_gate": "prespecified point-score threshold; interval reported separately", "errors": errors}


def _split_rows(record: dict, manifest: ExperimentManifest, split: str) -> tuple[list[dict], list[str]]:
    declared = {selection.split for selection in manifest.tasks}
    rows, errors = [], []
    seen = set()
    fixture_hashes = set()
    for row in record["samples"]:
        row_split = row.get("split")
        if row_split not in declared or row_split not in {"development", "holdout"}:
            errors.append(f"undeclared_sample_split:{row.get('task_id')}")
        if row_split != split:
            continue
        identity = row.get("task_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            errors.append("missing_or_duplicate_task_id")
        seen.add(identity)
        if row.get("category") not in CATEGORIES:
            errors.append(f"unknown_category:{identity}")
        if not isinstance(row.get("fixture_hash"), str) or not row["fixture_hash"]:
            errors.append(f"missing_fixture_hash:{identity}")
        elif row["fixture_hash"] in fixture_hashes:
            errors.append(f"replayed_fixture_hash:{identity}")
        fixture_hashes.add(row.get("fixture_hash"))
        if row.get("synthetic") is not False or row.get("model_evaluated") is not True:
            errors.append(f"sample_origin_unverified:{identity}")
        rows.append(deepcopy(row))
    covered = set()
    for selection in manifest.tasks:
        if selection.split != split:
            continue
        for requested in selection.task_ids:
            aliases = {requested}
            if selection.suite == "niah" and not requested.startswith("niah/"):
                aliases.add(f"niah/{requested}/{split}/seed-{selection.fixture_seed}")
            matches = seen & aliases
            if len(matches) != 1 or covered & matches:
                errors.append(f"required_task_missing_or_ambiguous:{selection.suite}:{requested}")
            covered.update(matches)
    if seen - covered:
        errors.append("unexpected_tasks_not_in_manifest")
    return rows, errors


def _compare_categories(reference: list[dict], candidate: list[dict], policy: CampaignPolicy, *,
                        self_reference: bool, comparable: bool, resamples: int) -> dict:
    thresholds = {"coding": policy.acceptance.minimum_coding_score,
                  "tools": policy.acceptance.minimum_tool_score,
                  "retrieval": policy.acceptance.minimum_retrieval_score}
    result = {}
    for category in CATEGORIES:
        left = [row for row in reference if row.get("category") == category]
        right = [row for row in candidate if row.get("category") == category]
        absolute = _absolute(right, thresholds[category])
        comparison = {"status": "inconclusive", "reason": "missing_or_incomparable_reference"}
        if comparable and left and right and not absolute["errors"]:
            try:
                # Always perform pairing/provenance validation, including for
                # the named baseline. Zero relative loss is then definitional.
                comparison = paired_comparison(left, right, maximum_loss=policy.acceptance.max_quality_loss,
                                               resamples=resamples)
                if self_reference:
                    comparison = {"status": "noninferior", "tasks": len(right), "mean_delta": 0.,
                                  "lower_95": 0., "upper_95": 0., "method": "identical-reference-definition",
                                  "maximum_loss": policy.acceptance.max_quality_loss}
            except (ValueError, KeyError, TypeError) as exc:
                comparison = {"status": "inconclusive", "reason": str(exc)}
        result[category] = {**absolute, "comparison": comparison}
    return result


def _speed(evidence: dict, policy: CampaignPolicy) -> dict:
    allowed = {item.name for item in fields(SpeedObservation)}
    raw_rows = evidence.get("speed", {}).get("observations", [])
    if not isinstance(raw_rows, list):
        raw_rows = [None]  # Malformed evidence must fail, not silently become a clean empty run.
    observations = []
    try:
        for row in raw_rows:
            values = {key: deepcopy(value) for key, value in row.items() if key in allowed}
            if "content_event_times" in values:
                values["content_event_times"] = tuple(values["content_event_times"])
            observations.append(SpeedObservation(**values))
        return summarize_speed(observations, policy.acceptance)
    except (TypeError, ValueError, AttributeError) as exc:
        return {"qualifies": False, "attempted": len(raw_rows), "observations": [],
                "minimum_native_tps": None, "median_native_tps": None, "maximum_native_tps": None,
                "reasons": [f"invalid_stored_speed_observations:{exc}"]}


def _holdout_coverage(reference_rows: list[dict], candidate_rows: list[dict],
                      quality: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split the quality space into what a holdout measured, validated, and never covered.

    ``registry.validate`` refuses to label a non-retrieval suite as unseen holdout
    (``registry.py:89-90``), so a holdout attempt legitimately measures only part of the
    quality space. Demanding a passing score in every category would make validation
    impossible by construction, which is exactly how a complete, passing retrieval holdout
    came back as an unexplained refusal (LIVE-011).

    Coverage is read from the rows the holdout pair actually produced, never assumed. A
    category the holdout manifest DECLARED but did not deliver is already a
    ``required_task_missing_or_ambiguous`` error from ``_split_rows``, so an empty category
    here can only mean the holdout never selected it. A category that either side measured
    stays in scope and must pass on both sides, so a one-sided category is a failure rather
    than something that quietly becomes "not covered".
    """
    measured = {row.get("category") for row in [*reference_rows, *candidate_rows]}
    covered = [name for name in CATEGORIES if name in measured]
    uncovered = [name for name in CATEGORIES if name not in measured]
    validated = [name for name in covered if quality[name]["absolute_passed"]
                 and quality[name]["comparison"]["status"] == "noninferior"]
    for name in CATEGORIES:
        # An uncovered category is marked, never scored: absolute_passed stays False so every
        # downstream gate (including reports.export_preset) still refuses to claim it.
        quality[name]["covered_by_holdout"] = name in covered
    return covered, uncovered, validated, [f"holdout_category_failed:{name}" for name in covered
                                           if name not in validated]


def _is_measured(record: dict, evidence: dict) -> bool:
    return (record["attempt"].get("synthetic") in {False, 0}
            and evidence.get("synthetic") is False and evidence.get("model_evaluated") is True
            and record["attempt"].get("state") == "completed")


def _context(rows: list[dict], speed: dict, manifest: ExperimentManifest) -> dict:
    retrieval = [row for row in rows if row.get("category") == "retrieval"]
    counts, errors = [], []
    if not retrieval:
        errors.append("retrieval_context_missing")
    for row in retrieval:
        value = row.get("context", {})
        count = value.get("actual_input_tokens")
        if (type(count) is not int or count < 1 or count > manifest.requested_input_tokens
                or value.get("observed_input_tokens") != count
                or value.get("actual_context_verified") is not True
                or value.get("counting_method") != "exact-post-template"
                or value.get("truncation_reported") is not False
                or value.get("context_policy") != "reject-overflow-no-shift"
                or value.get("tokenizer_id") != manifest.model.tokenizer_hash
                or value.get("template_id") != manifest.model.template_hash
                or count + manifest.generation.max_output_tokens > manifest.backend.context_length):
            errors.append(f"retrieval_context_unverified:{row.get('task_id')}")
        else:
            counts.append(count)
    for observation in speed.get("observations", []):
        count = observation.get("input_tokens")
        if (observation.get("actual_context_verified") is not True or type(count) is not int
                or count < 1 or count > manifest.requested_input_tokens):
            errors.append("speed_context_unverified")
        else:
            counts.append(count)
    return {"verified": not errors and bool(counts), "errors": errors,
            "requested_input_tokens": manifest.requested_input_tokens,
            "minimum_measured_input_tokens": min(counts) if counts else None,
            "maximum_measured_input_tokens": max(counts) if counts else None}


def analyze_campaign(store, campaign: dict, policy: CampaignPolicy, baseline_attempt_id: str | None = None, *,
                     resamples: int = 2000) -> dict:
    """Recompute speed, paired quality and disjoint holdout validation from Store.

    Separate holdout attempts are associated only with the identical complete
    configuration (excluding task selections). Their own provenance, speed,
    configuration and context checks must pass. Original evidence is immutable.

    ``holdout_passed`` is a SCOPED verdict: the holdout covered at least one quality
    category and validated every category it covered. The categories it never measured
    are listed in ``holdout["uncovered_categories"]`` and are never treated as passing -
    they carry ``absolute_passed=False`` and a named ``holdout_category_not_covered``
    reason, so ``reports.export_preset`` still refuses a preset that claims them.
    """
    identifiers = list(dict.fromkeys([*campaign.get("all_attempt_ids", []),
        *(row["attempt_id"] for row in campaign.get("results", []) if row.get("attempt_id"))]))
    if baseline_attempt_id and baseline_attempt_id not in identifiers:
        identifiers.insert(0, baseline_attempt_id)
    requested_ids = identifiers
    manifests = discover_campaign_attempts(store, requested_ids)
    # Keep the current run's requested results first; discovered history still participates
    # in all checks, and holdout freshness independently uses authoritative Store order.
    identifiers = list(dict.fromkeys([*requested_ids, *manifests]))
    records = {key: store.results(key) for key in identifiers}
    development_ids = [key for key in identifiers if any(item.split == "development" for item in manifests[key].tasks)]
    if not development_ids:
        return {**deepcopy(campaign), "results": [], "frontier": [], "winner": None,
                "all_attempt_ids": identifiers, "best_presets": {},
                "analysis_note": "No stored development attempts; nothing has been validated."}
    # Scope expansion must not silently change a caller's original default reference.
    baseline = baseline_attempt_id or next((key for key in requested_ids if key in development_ids),
                                          development_ids[0])
    if baseline not in development_ids:
        raise ValueError("baseline_attempt_id must identify a development reference attempt")
    reference_manifest = manifests[baseline]
    reference_record = records[baseline]
    reference_evidence = _evidence(reference_record)
    reference_rows, reference_errors = _split_rows(reference_record, reference_manifest, "development")
    if not _context(reference_rows, _speed(reference_evidence, policy), reference_manifest)["verified"]:
        reference_errors.append("reference_context_accounting_unverified")
    holdouts = {}
    all_development_hashes = set()
    # Store insertion order is authoritative; caller-owned campaign order cannot cherry-pick a holdout.
    order = {row["id"]: index for index, row in enumerate(store.attempts_in_insertion_order())}
    for key in sorted(identifiers, key=order.__getitem__):
        dev, _ = _split_rows(records[key], manifests[key], "development")
        all_development_hashes.update(row.get("fixture_hash") for row in dev if row.get("fixture_hash"))
        if any(item.split == "holdout" for item in manifests[key].tasks):
            holdouts[_config_key(manifests[key])] = key  # Last recorded attempt; never cherry-pick earlier successes.
    baseline_holdout_id = holdouts.get(_config_key(reference_manifest))
    baseline_holdout_rows, baseline_holdout_errors = [], ["reference_holdout_missing"]
    baseline_holdout_record = None
    if baseline_holdout_id:
        baseline_holdout_record = records[baseline_holdout_id]
        baseline_holdout_rows, baseline_holdout_errors = _split_rows(
            baseline_holdout_record, manifests[baseline_holdout_id], "holdout")
    policy_hash = hashlib.sha256(canonical_json(policy.model_dump(mode="json")).encode()).hexdigest()
    results = []
    for key in development_ids:
        record, manifest = records[key], manifests[key]
        original = _evidence(record)
        rows, errors = _split_rows(record, manifest, "development")
        comparable, family = True, "unverified"
        try:
            family = infer_comparison_family(reference_manifest, manifest)
        except ValueError as exc:
            comparable = False
            errors.append(f"comparison_not_controlled:{exc}")
        errors.extend(f"reference:{value}" for value in reference_errors)
        if sum(row["kind"] == "evidence" for row in record["events"]) != 1:
            errors.append("evidence_events_not_exactly_one")
        if not _is_measured(reference_record, reference_evidence):
            errors.append("reference_not_completed_measured_evidence")
        if reference_evidence.get("effective_settings_verified") is not True:
            errors.append("reference_settings_unverified")
        if reference_evidence.get("actual_context_verified") is not True:
            errors.append("reference_context_unverified")
        quality = _compare_categories(reference_rows, rows, policy, self_reference=key == baseline,
                                      comparable=comparable and not errors, resamples=resamples)
        holdout_id = holdouts.get(_config_key(manifest))
        holdout_rows, holdout_errors = [], ["candidate_holdout_missing"]
        holdout_record = None
        if holdout_id:
            holdout_record = records[holdout_id]
            holdout_rows, holdout_errors = _split_rows(holdout_record, manifests[holdout_id], "holdout")
        holdout_errors.extend(f"reference:{value}" for value in baseline_holdout_errors)
        if {holdout_id, baseline_holdout_id} & {key, baseline}:
            holdout_errors.append("holdout_not_separate_attempt")  # An attempt never validates itself.
        for held in baseline_holdout_rows + holdout_rows:
            if held.get("fixture_hash") in all_development_hashes:
                holdout_errors.append("development_holdout_fixture_overlap")
        holdout_comparable = comparable and not holdout_errors
        if holdout_record and baseline_holdout_record:
            try:
                infer_comparison_family(manifests[baseline_holdout_id], manifests[holdout_id])
            except ValueError as exc:
                holdout_comparable = False
                holdout_errors.append(f"holdout_comparison_not_controlled:{exc}")
            for label, held_record in (("candidate", holdout_record), ("reference", baseline_holdout_record)):
                held_evidence = _evidence(held_record)
                if not _is_measured(held_record, held_evidence):
                    holdout_errors.append(f"{label}_holdout_not_measured")
                if held_evidence.get("effective_settings_verified") is not True:
                    holdout_errors.append(f"{label}_holdout_settings_unverified")
                if held_evidence.get("actual_context_verified") is not True:
                    holdout_errors.append(f"{label}_holdout_context_unverified")
                held_rows = holdout_rows if label == "candidate" else baseline_holdout_rows
                held_manifest = manifests[holdout_id if label == "candidate" else baseline_holdout_id]
                if not _context(held_rows, _speed(held_evidence, policy), held_manifest)["verified"]:
                    holdout_errors.append(f"{label}_holdout_context_accounting_unverified")
                # A slow high-precision reference is valid quality evidence.
                # Only the configuration being selected must meet the speed floor.
                if label == "candidate" and not _speed(held_evidence, policy)["qualifies"]:
                    holdout_errors.append(f"{label}_holdout_speed_failed")
        holdout_quality = _compare_categories(baseline_holdout_rows, holdout_rows, policy,
            self_reference=key == baseline and holdout_id == baseline_holdout_id,
            comparable=holdout_comparable and not holdout_errors, resamples=resamples)
        covered, uncovered, validated_categories, category_failures = _holdout_coverage(
            baseline_holdout_rows, holdout_rows, holdout_quality)
        blocking = [*holdout_errors, *category_failures]
        if not covered:
            blocking.append("holdout_measured_no_category")  # Vacuous truth is not validation evidence.
        # The verdict is scoped: the holdout validated everything it covered, and it covered something.
        holdout_passed = not blocking
        # A refusal always names its cause, and an uncovered category is named even when the verdict is
        # positive, so "not validated" can never render with an empty reason list (LIVE-011).
        holdout_errors = [*blocking, *(f"holdout_category_not_covered:{name}" for name in uncovered)]
        speed = _speed(original, policy)
        context = _context(rows, speed, manifest)
        refreshed = {**deepcopy(original), "attempt_id": key, "manifest_hash": manifest.fingerprint(),
                     "policy_hash": policy_hash, "status": record["attempt"]["state"],
                     "synthetic": not _is_measured(record, original), "model_evaluated": _is_measured(record, original),
                     "actual_input_tokens": context["minimum_measured_input_tokens"],
                     "requested_input_tokens": manifest.requested_input_tokens, "context_accounting": context,
                     "actual_context_verified": context["verified"] and original.get("actual_context_verified") is True,
                     "speed": speed,
                     "quality": quality, "comparison_family": family, "baseline_attempt_id": baseline,
                     "holdout_passed": holdout_passed,
                     "holdout": {"attempt_id": holdout_id, "reference_attempt_id": baseline_holdout_id,
                                 "quality": holdout_quality, "passed": holdout_passed,
                                 "covered_categories": covered, "uncovered_categories": uncovered,
                                 "validated_categories": validated_categories,
                                 "fully_validated": holdout_passed and not uncovered,
                                 "errors": sorted(set(holdout_errors))}, "analysis_errors": errors}
        refreshed["eligibility"] = assess_candidate(refreshed, policy)
        # A validated configuration always needs real unseen holdout evidence,
        # even if a development-only policy elects to relax its screening gate.
        # The same holds for effective settings and measured context: screening flags may relax
        # exploration, but they never produce a validated winner or preset.
        extra = list(errors)
        if not holdout_passed:
            extra.append("holdout_not_validated")
        extra.extend(f"{name}_required_for_validation" for name in
                     ("effective_settings_verified", "actual_context_verified") if refreshed.get(name) is not True)
        if extra:
            refreshed["eligibility"] = {**refreshed["eligibility"], "eligible": False,
                "reasons": list(dict.fromkeys(refreshed["eligibility"]["reasons"] + extra))}
        results.append(refreshed)
    # A later failed re-run of the identical manifest supersedes an earlier success; never cherry-pick.
    failed = [row for row in results if not row["eligibility"]["eligible"]]
    for row in results:
        later = [other["attempt_id"] for other in failed if other["manifest_hash"] == row["manifest_hash"]
                 and order[other["attempt_id"]] > order[row["attempt_id"]]]
        if later:
            row["eligibility"] = {**row["eligibility"], "eligible": False, "reasons": list(dict.fromkeys(
                row["eligibility"]["reasons"] + [f"superseded_by_later_failed_attempt:{key}" for key in later]))}
        store.event(row["attempt_id"], "analysis", row)
    frontier = pareto_frontier(results)
    def quality_rank(row):
        scores = tuple(row["quality"][category]["score"] or 0.0 for category in CATEGORIES)  # unmeasured: no credit
        return (*scores, row["speed"]["minimum_native_tps"], row["actual_input_tokens"])
    ranked = sorted(frontier, key=quality_rank, reverse=True)
    def preset(row):
        # The holdout scope travels with the preset: a preset validated on a retrieval holdout is not
        # a preset validated on coding, and every reader of best_presets must be able to say which.
        held = row["holdout"]
        return {"attempt_id": row["attempt_id"], "manifest_hash": row["manifest_hash"],
                "manifest": manifests[row["attempt_id"]].model_dump(mode="json"),
                "native_tps_minimum": row["speed"]["minimum_native_tps"],
                "validated_input_tokens": row["actual_input_tokens"],
                "holdout_validated_categories": list(held["validated_categories"]),
                "holdout_uncovered_categories": list(held["uncovered_categories"]),
                "unmeasured_categories": list(row["eligibility"].get("unmeasured_categories", [])),
                "deployed": False}
    best = {}
    if ranked:
        best["highest_quality"] = preset(ranked[0])
        best["fastest_within_quality"] = preset(max(frontier, key=lambda row: row["speed"]["minimum_native_tps"]))
        best["largest_validated_context"] = preset(max(frontier, key=lambda row: row["actual_input_tokens"]))
    return {**deepcopy(campaign), "results": results, "frontier": [row["attempt_id"] for row in frontier],
            "all_attempt_ids": identifiers,
            "winner": ranked[0]["attempt_id"] if ranked else None, "best_presets": best,
            "reference": {"attempt_id": baseline, "model_key": reference_manifest.model.model_key,
                          "weight_quantization": reference_manifest.model.quantization,
                          "interpretation": f"Degradation relative to {reference_manifest.model.quantization} reference; "
                                            "no claim of equivalence to higher precision weights."},
            "automatic_deployment_performed": False, "analysis_policy_hash": policy_hash,
            "analysis_note": "Recomputed from stored samples and native speed observations under the current policy. "
                             "Each context block needs its own paired reference; untested lengths are not certified. "
                             "A holdout validates only the categories it measured; every other category is reported "
                             "as not covered by holdout and is never claimed as validated. A category with no "
                             "declared task is unmeasured: it cannot fail a candidate and is never claimed for one."}


def plan_holdout_candidates(manifests: Iterable[ExperimentManifest],
                            task_selections: Iterable[TaskSelection]) -> list[ExperimentManifest]:
    """Return immutable holdout-only manifests; do not execute or load anything.

    New static task IDs or different generated NIAH seeds protect against
    obvious reuse at planning time. Actual fixture-hash disjointness is also
    required after results exist, during analysis.
    """
    bases, selections = list(manifests), tuple(task_selections)
    if not selections or not all(isinstance(item, TaskSelection) and item.split == "holdout" for item in selections):
        raise ValueError("provide nonempty immutable TaskSelections explicitly marked holdout")
    def task_keys(items):
        return [(item.suite, task, item.fixture_seed if item.suite == "niah" else None)
                for item in items for task in item.task_ids]
    pairs = task_keys(selections)
    if len(pairs) != len(set(pairs)):
        raise ValueError("holdout task selections contain duplicate suite/task IDs")
    output, seen = [], set()
    for base in bases:
        if not isinstance(base, ExperimentManifest) or any(item.split != "development" for item in base.tasks):
            raise ValueError("holdout planning requires development-only ExperimentManifests")
        development = set(task_keys(base.tasks))
        development_niah_seeds = {item.fixture_seed for item in base.tasks if item.suite == "niah"}
        if any(item.suite == "niah" and item.fixture_seed in development_niah_seeds for item in selections):
            raise ValueError("generated NIAH holdout requires a different fixture seed from development")
        if development & set(pairs):
            raise ValueError("holdout IDs overlap development tasks; relabeling does not create unseen tasks")
        raw = base.model_dump(mode="json")
        raw["tasks"] = [item.model_dump(mode="json") for item in selections]
        raw["annotations"] = {**raw["annotations"], "holdout_plan": "not executed; fixture disjointness awaits validation"}
        planned = ExperimentManifest.model_validate_json(canonical_json(raw))
        if planned.fingerprint() not in seen:
            output.append(planned)
            seen.add(planned.fingerprint())
    return output
