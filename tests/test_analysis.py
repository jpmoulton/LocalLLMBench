"""Analysis tests populate temporary stores with artificial evidence; no inference."""

from dataclasses import asdict
import json

import pytest

from llmbench.analysis import analyze_campaign, infer_comparison_family, plan_holdout_candidates
from llmbench.config import AcceptancePolicy, CampaignPolicy, ExperimentManifest, RunMode, TaskSelection
from llmbench.measurement import SpeedObservation
from llmbench.search import propose
from llmbench.store import Store


def tasks(split="development", n=20):
    return tuple(f"{split}/{category}/{index}" for category in ("coding", "tools", "retrieval") for index in range(n))


def selection(split="development", n=20):
    return TaskSelection(suite="private-mixed", revision="immutable-v1", task_ids=tasks(split, n), split=split)


def reference(manifest, n=20):
    raw = manifest.model_dump(mode="json")
    raw.update(mode="live", environment_hash="recorded-environment", tasks=[selection(n=n).model_dump(mode="json")])
    raw["backend"]["engine"] = "lm-studio"
    raw["model"]["quantization"] = "Q6_K"
    return ExperimentManifest.model_validate_json(json.dumps(raw))


def policy(**kwargs):
    return CampaignPolicy(name="analysis-test", mode=RunMode.LIVE,
                          acceptance=AcceptancePolicy(max_quality_loss=.2, **kwargs))


def rows_for(names, split):
    return [{"task_id": name, "fixture_hash": f"fixture:{name}", "category": name.split("/")[1],
             "split": split, "status": "completed", "score": 1., "synthetic": False,
             "model_evaluated": True, "context": {"actual_input_tokens": 512, "observed_input_tokens": 512,
                "actual_context_verified": True, "counting_method": "exact-post-template",
                "truncation_reported": False, "context_policy": "reject-overflow-no-shift",
                "tokenizer_id": "fixture-v1", "template_id": "fixture-v1"}} for name in names]


def rows(split="development", n=20):
    return rows_for(tasks(split, n), split)


def insert(store, manifest, *, samples=None, rate=80., synthetic=False, changes=None, state="completed"):
    attempt = store.create_attempt(manifest, synthetic=synthetic)
    store.transition(attempt, "running")
    for sample in samples if samples is not None else rows(manifest.tasks[0].split):
        store.add_sample(attempt, sample)
    duration = 512 / rate
    observation = SpeedObservation(status="completed", finish_reason="length", output_tokens=512,
        native_generation_seconds=duration, elapsed_seconds=duration + .2, first_event_seconds=.1,
        content_event_times=tuple(.1 + duration * index / 10 for index in range(11)), input_tokens=512,
        expected_input_tokens=512, accepted_tokens_verified=True, synthetic=synthetic)
    evidence = {"attempt_id": attempt, "status": "completed", "synthetic": synthetic,
        "model_evaluated": not synthetic, "effective_settings_verified": True,
        "actual_context_verified": True, "holdout_passed": True,
        "speed": {"qualifies": True, "minimum_native_tps": 9999,
                  "observations": [asdict(observation) for _ in range(5)]},
        "eligibility": {"eligible": True, "reasons": [], "unmeasured_categories": []}}
    evidence.update(changes or {})
    store.event(attempt, "evidence", evidence)
    store.transition(attempt, state)
    return attempt


def build(store, base, candidate=None, *, holdout=True, candidate_rows=None, candidate_holdout_rows=None,
          base_rate=80., candidate_rate=160.):
    candidate = candidate or propose(base, {"backend.k_cache": "q8_0"}, "kv")
    base_id = insert(store, base, rate=base_rate)
    candidate_id = insert(store, candidate, samples=candidate_rows, rate=candidate_rate)
    ids = [base_id, candidate_id]
    if holdout:
        held = plan_holdout_candidates([base, candidate], [selection("holdout")])
        ids += [insert(store, held[0], rate=base_rate),
                insert(store, held[1], samples=candidate_holdout_rows, rate=candidate_rate)]
    return {"results": [{"attempt_id": key} for key in ids]}, base_id, candidate_id


def test_missing_holdout_cannot_be_replaced_by_old_boolean(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), holdout=False)
        result = analyze_campaign(store, campaign, policy(require_holdout=False), baseline, resamples=100)
        assert result["winner"] is None
        assert result["frontier"] == []
        assert all(not row["holdout_passed"] for row in result["results"])
        assert all("holdout_not_validated" in row["eligibility"]["reasons"] for row in result["results"])


def test_validated_holdout_and_uncertainty_select_faster_without_ceiling(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest))
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert result["winner"] == candidate
        assert result["best_presets"]["fastest_within_quality"]["native_tps_minimum"] == 160
        assert result["reference"]["weight_quantization"] == "Q6_K"
        assert "Q6_K reference" in result["reference"]["interpretation"]
        assert not result["automatic_deployment_performed"]
        candidate_result = result["results"][1]
        assert candidate_result["comparison_family"] == "kv"
        assert candidate_result["quality"]["coding"]["comparison"]["status"] == "noninferior"
        assert candidate_result["quality"]["coding"]["absolute_95"] is not None


def test_speed_and_eligibility_recomputed_under_current_policy(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest))
        first = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert first["winner"] == candidate
        assert len(first["all_attempt_ids"]) == 4
        second = analyze_campaign(store, first, policy(minimum_tokens_per_second=200.), baseline, resamples=100)
        assert second["winner"] is None
        assert len(second["all_attempt_ids"]) == 4
        assert all(row["holdout"]["attempt_id"] is not None for row in second["results"])
        assert all(not row["speed"]["qualifies"] for row in second["results"])
        events = store.results(candidate)["events"]
        assert len([event for event in events if event["kind"] == "evidence"]) == 1
        assert len([event for event in events if event["kind"] == "analysis"]) == 2


def test_comparability_rejects_mixed_axes_or_changed_environment(manifest):
    base = reference(manifest)
    mixed = propose(propose(base, {"backend.k_cache": "q8_0"}, "kv"), {"backend.batch_size": 256}, "performance")
    with pytest.raises(ValueError):
        infer_comparison_family(base, mixed)
    changed = base.model_dump(mode="json")
    changed["environment_hash"] = "different"
    with pytest.raises(ValueError):
        infer_comparison_family(base, ExperimentManifest.model_validate_json(json.dumps(changed)))
    changed = base.model_dump(mode="json")
    changed["model"]["source_revision"] = "different-base-model"
    with pytest.raises(ValueError):
        infer_comparison_family(base, ExperimentManifest.model_validate_json(json.dumps(changed)))


def test_required_missing_case_cannot_shrink_both_denominators(manifest, tmp_path):
    with Store(tmp_path) as store:
        base = reference(manifest)
        key = insert(store, base, samples=rows()[:-1])
        held = plan_holdout_candidates([base], [selection("holdout")])[0]
        held_id = insert(store, held)
        result = analyze_campaign(store, {"results": [{"attempt_id": key}, {"attempt_id": held_id}]},
                                  policy(), key, resamples=100)
        assert result["winner"] is None
        assert any("required_task_missing" in error for error in result["results"][0]["analysis_errors"])


def test_failures_stay_in_denominator_even_when_raw_score_is_one(manifest, tmp_path):
    samples = rows()
    samples[0]["status"] = "environment_error"
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_rows=samples)
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        coding = result["results"][1]["quality"]["coding"]
        assert coding["attempted"] == 20 and coding["completed"] == 19
        assert coding["score"] == .95
        assert coding["comparison"]["mean_delta"] == -.05


def test_unknown_completed_score_is_inconclusive_not_fabricated(manifest, tmp_path):
    samples = rows()
    samples[0]["score"] = "looks good"
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_rows=samples)
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        coding = result["results"][1]["quality"]["coding"]
        assert coding["score"] is None
        assert coding["comparison"]["status"] == "inconclusive"
        assert not result["results"][1]["eligibility"]["eligible"]


def test_holdout_fixture_reuse_rejected_even_with_new_task_ids(manifest, tmp_path):
    holdout_rows = rows("holdout")
    holdout_rows[0]["fixture_hash"] = rows()[0]["fixture_hash"]
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_holdout_rows=holdout_rows)
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert not result["results"][1]["holdout_passed"]
        assert "development_holdout_fixture_overlap" in result["results"][1]["holdout"]["errors"]


def test_candidate_holdout_missing_tools_cannot_pass(manifest, tmp_path):
    samples = [row for row in rows("holdout") if row["category"] != "tools"]
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_holdout_rows=samples)
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert not result["results"][1]["holdout_passed"]
        assert result["results"][1]["holdout"]["quality"]["tools"]["attempted"] == 0


def test_baseline_relative_zero_still_requires_absolute_quality(manifest, tmp_path):
    base = reference(manifest)
    samples = rows()
    for row in samples:
        row["score"] = 0.
    with Store(tmp_path) as store:
        key = insert(store, base, samples=samples)
        held = plan_holdout_candidates([base], [selection("holdout")])[0]
        held_key = insert(store, held)
        result = analyze_campaign(store, {"results": [{"attempt_id": key}, {"attempt_id": held_key}]},
                                  policy(), key, resamples=100)
        assert result["results"][0]["quality"]["coding"]["comparison"]["mean_delta"] == 0
        assert result["winner"] is None


def test_slow_reference_can_support_faster_candidate_quality(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), base_rate=40.)
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert result["winner"] == candidate
        assert not result["results"][0]["eligibility"]["eligible"]


def test_fixture_pairing_mismatch_does_not_claim_quantization_loss(manifest, tmp_path):
    samples = rows()
    samples[0]["fixture_hash"] = "different-prompt"
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_rows=samples)
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert result["results"][1]["quality"]["coding"]["comparison"]["status"] == "inconclusive"


def test_holdout_planning_returns_new_immutable_manifests_without_execution(manifest):
    base = reference(manifest)
    before = base.fingerprint()
    planned = plan_holdout_candidates([base, base], [selection("holdout")])
    assert len(planned) == 1
    assert base.fingerprint() == before and planned[0].fingerprint() != before
    assert all(item.split == "holdout" for item in planned[0].tasks)
    with pytest.raises((TypeError, ValueError)):
        planned[0].annotations["changed"] = "yes"
    with pytest.raises(ValueError):
        plan_holdout_candidates([base], [selection()])
    reused = TaskSelection(suite="private-mixed", revision="immutable-v1", task_ids=tasks(), split="holdout")
    with pytest.raises(ValueError):
        plan_holdout_candidates([base], [reused])


def test_analysis_does_not_trust_previous_synthetic_winner(manifest, tmp_path):
    with Store(tmp_path) as store:
        base = reference(manifest)
        key = insert(store, base, synthetic=True)
        result = analyze_campaign(store, {"results": [{"attempt_id": key, "eligibility": {"eligible": True}}]},
                                  policy(), key, resamples=100)
        assert result["winner"] is None


def test_context_reports_measured_occupancy_not_requested_tier(manifest, tmp_path):
    samples = rows()
    for row in samples:
        row["context"]["actual_input_tokens"] = 500
        row["context"]["observed_input_tokens"] = 500
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_rows=samples)
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        actual = result["results"][1]
        assert actual["actual_input_tokens"] == 500
        assert actual["requested_input_tokens"] == 512
        assert actual["context_accounting"]["maximum_measured_input_tokens"] == 512


def test_niah_holdout_keeps_variant_names_with_distinct_fixture_seeds(manifest):
    raw = reference(manifest).model_dump(mode="json")
    raw["tasks"] = [TaskSelection(suite="niah", revision="local-niah-v1",
        task_ids=("single-early", "multi"), fixture_seed=42).model_dump(mode="json")]
    base = ExperimentManifest.model_validate_json(json.dumps(raw))
    selections = [TaskSelection(suite="niah", revision="local-niah-v1",
        task_ids=("single-early", "multi"), fixture_seed=seed, split="holdout") for seed in (1001, 1002)]
    planned = plan_holdout_candidates([base], selections)
    assert len(planned[0].tasks) == 2
    assert planned[0].tasks[0].fixture_seed == 1001
    with pytest.raises(ValueError, match="different fixture seed"):
        plan_holdout_candidates([base], [TaskSelection(suite="niah", revision="local-niah-v1",
            task_ids=("single-late",), fixture_seed=42, split="holdout")])


# ---- Review findings REV-033/035/038/041: the analysis verdict must agree with validated export ----


def result_row(result, attempt):
    return next(row for row in result["results"] if row["attempt_id"] == attempt)


def test_store_lists_attempts_in_insertion_order_with_manifests(manifest, tmp_path):
    with Store(tmp_path) as store:
        keys = [insert(store, reference(manifest), samples=[]) for _ in range(5)]
        # Creation timestamps can tie or be edited; insertion order cannot.
        store.db.execute("UPDATE attempts SET created='1999' WHERE id=?", (keys[-1],))
        store.db.commit()
        listed = store.attempts_in_insertion_order()
        assert [row["id"] for row in listed] == keys
        assert [row["id"] for row in store.attempts()][0] == keys[-1]
        assert all(row["manifest"] == store.results(row["id"])["manifest"] for row in listed)
        assert listed[0]["state"] == "completed" and listed[0]["synthetic"] == 0


@pytest.mark.parametrize("flag,key", [("require_effective_settings_verified", "effective_settings_verified"),
                                      ("require_actual_context_verified", "actual_context_verified")])
def test_rev033_relaxed_verification_flags_never_select_a_validated_winner(manifest, tmp_path, flag, key):
    base = reference(manifest)
    candidate_manifest = propose(base, {"backend.k_cache": "q8_0"}, "kv")
    with Store(tmp_path) as store:
        baseline = insert(store, base)
        candidate = insert(store, candidate_manifest, rate=160., changes={key: False})
        held = plan_holdout_candidates([base, candidate_manifest], [selection("holdout")])
        ids = [baseline, candidate, insert(store, held[0]), insert(store, held[1], rate=160.)]
        result = analyze_campaign(store, {"all_attempt_ids": ids}, policy(**{flag: False}), baseline, resamples=100)
        row = result_row(result, candidate)
        assert row["holdout_passed"] is True and row["eligibility"]["eligible"] is False
        assert f"{key}_required_for_validation" in row["eligibility"]["reasons"]
        assert result["winner"] == baseline and candidate not in result["frontier"]


def test_rev035_single_attempt_is_never_its_own_separate_holdout(manifest, tmp_path):
    raw = reference(manifest).model_dump(mode="json")
    raw["tasks"] = [selection().model_dump(mode="json"), selection("holdout").model_dump(mode="json")]
    combined = ExperimentManifest.model_validate_json(json.dumps(raw))
    with Store(tmp_path) as store:
        key = insert(store, combined, samples=rows() + rows("holdout"))
        result = analyze_campaign(store, {"results": [{"attempt_id": key}]}, policy(), key, resamples=100)
        row = result["results"][0]
        assert row["holdout"]["attempt_id"] == key and row["holdout_passed"] is False
        assert "holdout_not_separate_attempt" in row["holdout"]["errors"]
        assert result["winner"] is None and result["best_presets"] == {}


def test_rev035_campaign_order_cannot_cherry_pick_an_earlier_holdout(manifest, tmp_path):
    failed = rows("holdout")
    for sample in failed:
        sample["score"] = 0.
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest))
        first = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        earlier = result_row(first, candidate)["holdout"]["attempt_id"]
        candidate_manifest = ExperimentManifest.model_validate_json(json.dumps(store.results(candidate)["manifest"]))
        held = plan_holdout_candidates([candidate_manifest], [selection("holdout")])[0]
        later = insert(store, held, samples=failed, rate=160.)
        ids = [key for key in first["all_attempt_ids"] if key != earlier] + [later, earlier]  # Success listed last.
        result = analyze_campaign(store, {"all_attempt_ids": ids}, policy(), baseline, resamples=100)
        row = result_row(result, candidate)
        assert row["holdout"]["attempt_id"] == later and row["holdout_passed"] is False
        assert result["winner"] != candidate


def test_rev038_appended_evidence_event_is_not_measured_evidence(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_rate=30.)
        first = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert result_row(first, candidate)["speed"]["qualifies"] is False
        fast = json.loads(next(event["payload"] for event in store.results(baseline)["events"]
                               if event["kind"] == "evidence"))
        for observation in fast["speed"]["observations"]:
            observation["native_generation_seconds"] = 512 / 160.
        store.event(candidate, "evidence", {**fast, "attempt_id": candidate})
        second = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        row = result_row(second, candidate)
        assert row["eligibility"]["eligible"] is False and second["winner"] != candidate
        assert "evidence_events_not_exactly_one" in row["analysis_errors"]


@pytest.mark.parametrize("declared", [True, False])
def test_rev041_later_failed_rerun_of_identical_manifest_supersedes(manifest, tmp_path, declared):
    failed = rows()
    for sample in failed:
        sample["score"] = 0.
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest))
        candidate_manifest = ExperimentManifest.model_validate_json(json.dumps(store.results(candidate)["manifest"]))
        rerun = insert(store, candidate_manifest, samples=failed, rate=160.)
        if declared:
            campaign["results"].insert(0, {"attempt_id": rerun})  # Campaign order is not authoritative.
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert f"superseded_by_later_failed_attempt:{rerun}" in result_row(result, candidate)["eligibility"]["reasons"]
        assert result["winner"] == baseline
        newest = insert(store, candidate_manifest, rate=160.)
        campaign["results"].append({"attempt_id": newest})
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert result["winner"] == newest
        assert result_row(result, candidate)["eligibility"]["eligible"] is False


@pytest.mark.parametrize("victim", [0, 1, 2, 3], ids=["baseline", "candidate", "reference-holdout", "candidate-holdout"])
@pytest.mark.parametrize("origin", [None, (0, 1), ("false", "true")], ids=["missing", "numbers", "strings"])
def test_rev037_direct_analysis_requires_explicit_measured_sample_origin(manifest, tmp_path, victim, origin):
    base = reference(manifest)
    candidate_manifest = propose(base, {"backend.k_cache": "q8_0"}, "kv")
    held = plan_holdout_candidates([base, candidate_manifest], [selection("holdout")])
    manifests = [base, candidate_manifest, *held]
    with Store(tmp_path) as store:
        ids = []
        for index, item in enumerate(manifests):
            samples = rows(item.tasks[0].split)
            if index == victim:
                if origin is None:
                    del samples[0]["synthetic"], samples[0]["model_evaluated"]
                else:
                    samples[0].update(synthetic=origin[0], model_evaluated=origin[1])
            ids.append(insert(store, item, samples=samples, rate=160. if index in {1, 3} else 80.))
        result = analyze_campaign(store, {"all_attempt_ids": ids}, policy(), ids[0], resamples=100)
        row = result_row(result, ids[1])
        assert row["eligibility"]["eligible"] is False
        assert ids[1] not in result["frontier"] and result["winner"] != ids[1]
        assert all(value["attempt_id"] != ids[1] for value in result["best_presets"].values())
        assert "sample_origin_unverified" in json.dumps(row)
        assert row["quality"]["coding"]["attempted"] == 20  # Origin rejection never drops the case.


# ---- LIVE-011: a holdout validates what it measured, and a refusal always names its cause ----
# registry.py:89-90 refuses to label a non-retrieval suite as unseen holdout, so the live campaign's
# holdout could only run NIAH. These reproduce that shape: development measures every category, the
# holdout measures retrieval alone, and the SAME holdout attempt is both candidate and reference
# evidence for the baseline (exactly as in artifacts/container-campaign/live-offload-2).


RETRIEVAL_HOLDOUT = tuple(f"holdout/retrieval/{index}" for index in range(6))


def retrieval_holdout_selection():
    return TaskSelection(suite="private-mixed", revision="immutable-v1", task_ids=RETRIEVAL_HOLDOUT,
                         split="holdout")


def live_shape(store, manifest, *, holdout_samples=None, holdout=True, holdout_state="completed"):
    base = reference(manifest)
    development = insert(store, base)
    ids = [development]
    if holdout:
        held = plan_holdout_candidates([base], [retrieval_holdout_selection()])[0]
        samples = rows_for(RETRIEVAL_HOLDOUT, "holdout") if holdout_samples is None else holdout_samples
        ids.append(insert(store, held, samples=samples, state=holdout_state))
    return {"results": [{"attempt_id": key} for key in ids]}, development


def test_live011_retrieval_only_holdout_validates_retrieval_and_names_what_it_never_covered(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, development = live_shape(store, manifest)
        result = analyze_campaign(store, campaign, policy(), development, resamples=100)
        row = result_row(result, development)
        held = row["holdout"]
        assert held["covered_categories"] == ["retrieval"] and held["validated_categories"] == ["retrieval"]
        assert held["uncovered_categories"] == ["coding", "tools"]
        # The holdout validated everything it measured, but that is not a full validation.
        assert row["holdout_passed"] is True and held["passed"] is True and held["fully_validated"] is False
        # Never silently passing: the uncovered categories are named and still score as unmeasured.
        assert held["errors"] == ["holdout_category_not_covered:coding", "holdout_category_not_covered:tools"]
        assert held["quality"]["retrieval"]["covered_by_holdout"] is True
        assert all(held["quality"][name]["covered_by_holdout"] is False
                   and held["quality"][name]["absolute_passed"] is False for name in ("coding", "tools"))
        # A definite verdict: the live shape no longer dead-ends with an empty reason.
        assert row["eligibility"] == {"eligible": True, "reasons": [], "unmeasured_categories": []}
        assert result["winner"] == development and result["frontier"] == [development]
        for preset in result["best_presets"].values():
            assert preset["holdout_validated_categories"] == ["retrieval"]
            assert preset["holdout_uncovered_categories"] == ["coding", "tools"]


def test_live011_a_retrieval_only_holdout_still_cannot_export_a_preset(manifest, tmp_path):
    """REV-032 holds: a scoped verdict is not a deployable preset, and the refusal names the gap."""
    from llmbench.reports import export_preset, export_provisional_preset
    current = policy()
    with Store(tmp_path) as store:
        campaign, development = live_shape(store, manifest)
        result = analyze_campaign(store, campaign, current, development, resamples=100)
        row = result_row(result, development)
        assert row["eligibility"]["eligible"] is True and row["holdout"]["fully_validated"] is False
        exported = store.results(development)["manifest"]
        for export in (export_preset, export_provisional_preset):
            destination = tmp_path / f"{export.__name__}.json"
            with pytest.raises(ValueError, match="holdout") as refusal:
                export(exported, row, current, destination, store=store, campaign=result,
                       baseline_attempt_id=development, allow_self_reference=True)
            assert "coding" in str(refusal.value) and "tools" in str(refusal.value)
            assert not destination.exists()


def test_live011_failing_retrieval_holdout_still_blocks(manifest, tmp_path):
    failed = rows_for(RETRIEVAL_HOLDOUT, "holdout")
    for sample in failed:
        sample["score"] = 0.
    with Store(tmp_path) as store:
        campaign, development = live_shape(store, manifest, holdout_samples=failed)
        result = analyze_campaign(store, campaign, policy(), development, resamples=100)
        row = result_row(result, development)
        assert row["holdout_passed"] is False and row["holdout"]["validated_categories"] == []
        assert "holdout_category_failed:retrieval" in row["holdout"]["errors"]
        assert "holdout_not_validated" in row["eligibility"]["reasons"]
        assert result["winner"] is None and result["best_presets"] == {}


def test_live011_missing_and_errored_holdouts_still_block(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, development = live_shape(store, manifest, holdout=False)
        row = result_row(analyze_campaign(store, campaign, policy(), development, resamples=100), development)
        assert row["holdout_passed"] is False and row["holdout"]["covered_categories"] == []
        assert "candidate_holdout_missing" in row["holdout"]["errors"]
        assert "holdout_measured_no_category" in row["holdout"]["errors"]
    with Store(tmp_path / "errored") as store:
        campaign, development = live_shape(store, manifest, holdout_state="failed")
        row = result_row(analyze_campaign(store, campaign, policy(), development, resamples=100), development)
        # The holdout ran retrieval and scored 1.000, but the attempt itself never completed.
        assert row["holdout_passed"] is False and row["holdout"]["covered_categories"] == ["retrieval"]
        assert "candidate_holdout_not_measured" in row["holdout"]["errors"]
        assert "holdout_not_validated" in row["eligibility"]["reasons"]


def test_live011_a_holdout_that_measured_nothing_never_validates_vacuously(manifest, tmp_path):
    """An empty holdout satisfies "everything it covered passed" vacuously; that is not evidence."""
    raw = reference(manifest).model_dump(mode="json")
    raw["tasks"] = [retrieval_holdout_selection().model_dump(mode="json")]
    empty_holdout = ExperimentManifest.model_validate_json(json.dumps(raw))
    with Store(tmp_path) as store:
        development = insert(store, reference(manifest))
        held = insert(store, empty_holdout, samples=[])
        result = analyze_campaign(store, {"results": [{"attempt_id": development}, {"attempt_id": held}]},
                                  policy(), development, resamples=100)
        row = result_row(result, development)
        assert row["holdout_passed"] is False and row["holdout"]["covered_categories"] == []
        assert "holdout_measured_no_category" in row["holdout"]["errors"]
        assert result["winner"] is None


@pytest.mark.parametrize("shape", ["missing", "errored", "failing", "empty-samples"])
def test_live011_a_negative_holdout_verdict_always_carries_a_reason(manifest, tmp_path, shape):
    """The empty-reason defect at its source: errors can never be empty while the verdict is negative."""
    failed = rows_for(RETRIEVAL_HOLDOUT, "holdout")
    for sample in failed:
        sample["score"] = 0.
    options = {"missing": {"holdout": False}, "errored": {"holdout_state": "failed"},
               "failing": {"holdout_samples": failed}, "empty-samples": {"holdout_samples": []}}
    with Store(tmp_path) as store:
        campaign, development = live_shape(store, manifest, **options[shape])
        row = result_row(analyze_campaign(store, campaign, policy(), development, resamples=100), development)
        assert row["holdout"]["passed"] is False
        assert row["holdout"]["errors"] and all(item for item in row["holdout"]["errors"])
        assert "holdout_not_validated" in row["eligibility"]["reasons"]


def test_scope_finds_omitted_failed_holdout_without_including_unrelated_configuration(manifest, tmp_path):
    with Store(tmp_path) as store:
        campaign, baseline, candidate = build(store, reference(manifest))
        candidate_manifest = ExperimentManifest.model_validate_json(json.dumps(store.results(candidate)["manifest"]))
        other = propose(candidate_manifest, {"backend.batch_size": 256}, "performance")
        unrelated = insert(store, other, samples=rows("holdout"))  # Deliberately invalid, but outside this scope.
        held = plan_holdout_candidates([candidate_manifest], [selection("holdout")])[0]
        failed = rows("holdout")
        for sample in failed:
            sample["score"] = 0.
        later = insert(store, held, samples=failed, rate=160.)  # Not in caller's campaign snapshot.
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert later in result["all_attempt_ids"] and unrelated not in result["all_attempt_ids"]
        assert result_row(result, candidate)["holdout"]["attempt_id"] == later
        assert result_row(result, candidate)["eligibility"]["eligible"] is False
        assert result["winner"] == baseline
