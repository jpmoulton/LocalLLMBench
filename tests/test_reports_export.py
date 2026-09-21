"""REV-032 export regressions: artificial Store evidence only; no model, service or container."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from llmbench import reports
from llmbench.analysis import analyze_campaign, plan_holdout_candidates
from llmbench.config import BackendSettings, ExperimentManifest, ModelArtifact, TaskSelection, canonical_json
from llmbench.reports import export_preset, export_provisional_preset
from llmbench.search import propose
from llmbench.store import Store
from test_analysis import build, insert, policy, reference, rows, selection, tasks

GOOD = {"score": 1., "absolute_passed": True, "errors": [], "comparison": {"status": "noninferior"}}


def analyzed(store, manifest, current=None, **kwargs):
    """Build an artificial Store, analyze it, and return everything an honest caller would hold."""
    campaign, baseline, candidate = build(store, reference(manifest), **kwargs)
    result = analyze_campaign(store, campaign, current or policy(), baseline, resamples=100)
    row = next(item for item in result["results"] if item["attempt_id"] == candidate)
    return result, baseline, candidate, row, store.results(candidate)["manifest"]


def forge(row):
    """A caller-fabricated fully positive verdict; only fresh Store recomputation can refute it."""
    quality = {category: deepcopy(GOOD) for category in ("coding", "tools", "retrieval")}
    return {**deepcopy(row), "eligibility": {"eligible": True, "reasons": [], "unmeasured_categories": []}, "analysis_errors": [],
            "synthetic": False, "model_evaluated": True, "status": "completed", "holdout_passed": True,
            "effective_settings_verified": True, "actual_context_verified": True,
            "context_accounting": {**deepcopy(row.get("context_accounting") or {}), "verified": True},
            "speed": {**deepcopy(row.get("speed", {})), "qualifies": True}, "quality": deepcopy(quality),
            "holdout": {"attempt_id": "forged-holdout", "reference_attempt_id": "forged-reference-holdout",
                        "quality": quality, "passed": True, "errors": []}}


def context(store, result, baseline):
    return {"store": store, "campaign": result, "baseline_attempt_id": baseline, "resamples": 100}


def manifest_of(store, attempt):
    return ExperimentManifest.model_validate_json(canonical_json(store.results(attempt)["manifest"]))


def test_relaxed_screening_without_holdout_cannot_export_validated_preset(manifest, tmp_path):
    # Exact planning-review reproduction: relaxed screening policy, no holdout attempts at all.
    relaxed = policy(require_holdout=False)
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, relaxed, holdout=False)
        assert row["eligibility"]["eligible"] is False and row["holdout_passed"] is False
        with pytest.raises(ValueError):
            export_preset(exported, row, relaxed, target)
        assert not target.exists()
        with pytest.raises(ValueError, match="supplied:.*holdout_not_validated"):
            export_preset(exported, row, relaxed, target, **context(store, result, baseline))
        assert not target.exists()
        # Forging the nested verdict does not help: the Store is recomputed under the current policy.
        with pytest.raises(ValueError, match="current:.*holdout_not_validated"):
            export_preset(exported, forge(row), relaxed, target, **context(store, result, baseline))
        assert not target.exists()


def test_legacy_call_without_store_context_fails_closed(manifest, tmp_path):
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        assert row["eligibility"] == {"eligible": True, "reasons": [], "unmeasured_categories": []}
        full = context(store, result, baseline)
        for missing in ("store", "campaign", "baseline_attempt_id"):
            partial = {key: value for key, value in full.items() if key != missing}
            with pytest.raises(ValueError, match="authoritative"):
                export_preset(exported, row, policy(), target, **partial)
        with pytest.raises(ValueError, match="authoritative"):
            export_preset(exported, row, policy(), target)
        with pytest.raises(ValueError, match="authoritative"):
            export_preset(exported, row, policy(), target, **{**full, "store": object()})
        with pytest.raises(ValueError, match="authoritative"):
            export_preset(exported, row, policy().model_dump(mode="json"), target, **full)
    assert not target.exists()


@pytest.mark.parametrize("relaxed", [False, True])
def test_validated_store_exports_exact_references_offline(manifest, tmp_path, monkeypatch, relaxed):
    import socket
    import subprocess
    current = policy(require_holdout=not relaxed)
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, current)
        monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("network prohibited"))
        monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("network prohibited"))
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("process prohibited"))
        payload = export_preset(exported, row, current, target, **context(store, result, baseline))
        written = json.loads(target.read_text(encoding="utf-8"))
        assert written == payload
        assert written["kind"] == "validated-preset" and written["applied"] is False
        assert written["evidence_attempt_id"] == candidate and written["baseline_attempt_id"] == baseline
        assert written["holdout_attempt_id"] == row["holdout"]["attempt_id"]
        assert written["reference_holdout_attempt_id"] == row["holdout"]["reference_attempt_id"]
        assert len({candidate, baseline, written["holdout_attempt_id"],
                    written["reference_holdout_attempt_id"]}) == 4
        assert written["manifest"] == exported
        assert written["manifest_hash"] == manifest_of(store, candidate).fingerprint() == row["manifest_hash"]
        assert written["policy_hash"] == result["analysis_policy_hash"] == hashlib.sha256(
            canonical_json(current.model_dump(mode="json")).encode()).hexdigest()
        assert set(written["analyzed_attempt_ids"]) == set(result["all_attempt_ids"])
        assert "blockers" not in written and "row" not in written
        # REV-039: the artifact states what was accepted, not only a policy hash.
        assert written["acceptance"] == current.acceptance.model_dump(mode="json")
        assert written["comparison_family"] == "kv" and written["reference_self_validated"] is False
        assert written["native_tps_minimum"] == 160 and written["validated_input_tokens"] == 512
        for label in ("quality", "holdout_quality"):
            assert {key: value["score"] for key, value in written[label].items()} == {
                "coding": 1., "tools": 1., "retrieval": 1.}
            assert all(value["comparison"]["status"] == "noninferior" for value in written[label].values())
            assert all(value["threshold"] is not None for value in written[label].values())


def test_existing_target_is_never_overwritten(manifest, tmp_path):
    target = tmp_path / "preset.json"
    target.write_text("operator-owned", encoding="utf-8")
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(FileExistsError):
                export(exported, row, policy(), target, **context(store, result, baseline))
            assert target.read_text(encoding="utf-8") == "operator-owned"


def test_failed_write_leaves_no_partial_preset(manifest, tmp_path, monkeypatch):
    target = tmp_path / "preset.json"
    with pytest.raises(TypeError):
        reports._write_exclusive(target, {"unserializable": object()})
    assert not target.exists()
    real_open = Path.open
    class DiskFull:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def write(self, text):
            raise OSError("disk full")
    def failing_open(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        return DiskFull(handle) if mode == "x" else handle
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        monkeypatch.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match="disk full"):
            export_preset(exported, row, policy(), target, **context(store, result, baseline))
        monkeypatch.undo()
    assert not target.exists()


def drop(*path):
    def change(row):
        for key in path[:-1]:
            row = row[key]
        del row[path[-1]]
    return change


def put(value, *path):
    def change(row):
        for key in path[:-1]:
            row = row[key]
        row[path[-1]] = value
    return change


SUPPLIED_DEFECTS = {
    "negative_eligibility": put({"eligible": False, "reasons": ["speed_gate_failed"]}, "eligibility"),
    "negative_without_reasons": put(False, "eligibility", "eligible"),
    "positive_with_reasons": put(["holdout_not_validated"], "eligibility", "reasons"),
    "truthy_not_boolean": put(1, "eligibility", "eligible"),
    "string_verdict": put("true", "eligibility", "eligible"),
    "eligibility_missing": drop("eligibility"),
    "eligibility_not_mapping": put(True, "eligibility"),
    "reasons_missing": drop("eligibility", "reasons"),
    "reasons_not_list": put("", "eligibility", "reasons"),
    "analysis_errors_present": put(["required_task_missing_or_ambiguous:x"], "analysis_errors"),
    "analysis_errors_missing": drop("analysis_errors"),
    "analysis_errors_malformed": put(None, "analysis_errors"),
    "synthetic_origin": put(True, "synthetic"),
    "model_not_evaluated": put(False, "model_evaluated"),
    "incomplete_origin": put("failed", "status"),
    "speed_failed": put(False, "speed", "qualifies"),
    "speed_missing": drop("speed"),
    "stale_policy_hash": put("0" * 64, "policy_hash"),
    "policy_hash_missing": drop("policy_hash"),
    "changed_manifest_hash": put("0" * 64, "manifest_hash"),
    "other_baseline": put("another-baseline", "baseline_attempt_id"),
    "holdout_flag_false": put(False, "holdout_passed"),
    "holdout_flag_truthy": put(1, "holdout_passed"),
    "holdout_flag_missing": drop("holdout_passed"),
    "holdout_missing": drop("holdout"),
    "holdout_not_mapping": put([], "holdout"),
    "holdout_failed": put(False, "holdout", "passed"),
    "holdout_passed_missing": drop("holdout", "passed"),
    "holdout_attempt_none": put(None, "holdout", "attempt_id"),
    "holdout_attempt_empty": put("", "holdout", "attempt_id"),
    "holdout_attempt_missing": drop("holdout", "attempt_id"),
    "reference_holdout_none": put(None, "holdout", "reference_attempt_id"),
    "reference_holdout_missing": drop("holdout", "reference_attempt_id"),
    "holdout_errors_present": put(["development_holdout_fixture_overlap"], "holdout", "errors"),
    "holdout_errors_missing": drop("holdout", "errors"),
    "holdout_quality_missing": drop("holdout", "quality"),
    "holdout_category_missing": drop("holdout", "quality", "tools"),
    "holdout_category_absolute_failed": put(False, "holdout", "quality", "coding", "absolute_passed"),
    "holdout_category_degraded": put({"status": "degraded"}, "holdout", "quality", "retrieval", "comparison"),
    "holdout_category_inconclusive": put("inconclusive", "holdout", "quality", "tools", "comparison",
                                         "status"),
    "holdout_category_score_errors": put(["invalid_completed_score:x"], "holdout", "quality", "tools",
                                         "errors"),
    "development_category_missing": drop("quality", "coding"),
    "development_category_below_threshold": put(.1, "quality", "coding", "score"),
    "development_category_nan": put(float("nan"), "quality", "tools", "score"),
    "development_category_boolean_score": put(True, "quality", "tools", "score"),
    "development_comparison_missing": drop("quality", "retrieval", "comparison"),
    "quality_not_mapping": put(None, "quality"),
    "settings_unverified": put(False, "effective_settings_verified"),
    "settings_truthy": put(1, "effective_settings_verified"),
    "settings_missing": drop("effective_settings_verified"),
    "context_unverified": put(False, "actual_context_verified"),
    "context_missing": drop("actual_context_verified"),
    "context_accounting_unverified": put(False, "context_accounting", "verified"),
    "context_accounting_missing": drop("context_accounting"),
    "context_accounting_not_mapping": put(True, "context_accounting"),
}


@pytest.fixture(scope="module")
def validated_store(tmp_path_factory):
    # One genuinely validated Store shared by read-mostly cases; analysis only appends events to it.
    base = ExperimentManifest(
        model=ModelArtifact(model_key="synthetic-fixture", sha256="0" * 64, source_revision="fixture-v1",
                            quantization="synthetic", tokenizer_hash="fixture-v1",
                            template_hash="fixture-v1"),
        backend=BackendSettings(engine="mock", runtime_revision="mock-v1", context_length=8192),
        tasks=(TaskSelection(suite="offline-demo", revision="v1", task_ids=("tool-1", "needle-1")),),
        harness_revision="0.1.0", scorer_revision="v1")
    with Store(tmp_path_factory.mktemp("validated") / "store") as store:
        yield (store, *analyzed(store, base))


@pytest.mark.parametrize("defect", sorted(SUPPLIED_DEFECTS))
def test_supplied_verdict_defects_reject_for_both_artifacts(validated_store, tmp_path, defect):
    store, result, baseline, candidate, row, exported = validated_store
    broken = deepcopy(row)
    SUPPLIED_DEFECTS[defect](broken)
    target = tmp_path / "preset.json"
    held_only = defect.startswith(("holdout_", "reference_holdout_"))
    with pytest.raises(ValueError, match="supplied:"):
        export_preset(exported, broken, policy(), target, **context(store, result, baseline))
    assert not target.exists()
    if not held_only:  # Holdout-only gaps are the one thing a provisional artifact may record instead.
        with pytest.raises(ValueError, match="supplied:"):
            export_provisional_preset(exported, broken, policy(), target, **context(store, result, baseline))
        assert not target.exists()


@pytest.mark.parametrize("evidence", [None, {}, [], "attempt", {"attempt_id": ""}, {"attempt_id": 7}])
def test_missing_analysis_rejects(validated_store, tmp_path, evidence):
    store, result, baseline, candidate, row, exported = validated_store
    for export in (export_preset, export_provisional_preset):
        with pytest.raises(ValueError, match="analysis_missing"):
            export(exported, evidence, policy(), tmp_path / "preset.json", **context(store, result, baseline))
    assert not (tmp_path / "preset.json").exists()


def test_raw_original_evidence_is_not_an_analysis(validated_store, tmp_path):
    store, result, baseline, candidate, row, exported = validated_store
    raw = json.loads(next(event["payload"] for event in store.results(candidate)["events"]
                          if event["kind"] == "evidence"))
    assert raw["eligibility"] == {"eligible": True, "reasons": [], "unmeasured_categories": []} and raw["holdout_passed"] is True
    with pytest.raises(ValueError, match="supplied:"):
        export_preset(exported, raw, policy(), tmp_path / "preset.json", **context(store, result, baseline))
    assert not (tmp_path / "preset.json").exists()


def test_campaign_must_declare_known_attempt_and_baseline(validated_store, tmp_path):
    store, result, baseline, candidate, row, exported = validated_store
    target = tmp_path / "preset.json"
    campaigns = [{}, {"results": [{"attempt_id": candidate}]}, {"all_attempt_ids": [baseline]},
                 {"all_attempt_ids": [*result["all_attempt_ids"], "unknown-attempt"]},
                 {"results": [None]}, {"all_attempt_ids": 5}]
    for campaign in campaigns:
        with pytest.raises(ValueError, match="preset export refused"):
            export_preset(exported, row, policy(), target,
                          **{**context(store, result, baseline), "campaign": campaign})
    with pytest.raises(ValueError, match="preset export refused"):
        export_preset(exported, row, policy(), target, **context(store, result, "unknown-baseline"))
    assert not target.exists()


def test_failed_holdout_rejects_even_with_forged_verdict(manifest, tmp_path):
    failed = rows("holdout")
    for sample in failed:
        sample["score"] = 0.
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, candidate_holdout_rows=failed)
        assert row["holdout_passed"] is False and row["holdout"]["attempt_id"]
        with pytest.raises(ValueError, match="supplied:"):
            export_preset(exported, row, policy(), target, **context(store, result, baseline))
        with pytest.raises(ValueError, match="current:.*holdout_not_passed"):
            export_preset(exported, forge(row), policy(), target, **context(store, result, baseline))
    assert not target.exists()


def test_holdout_category_failure_rejects(manifest, tmp_path):
    samples = [sample for sample in rows("holdout") if sample["category"] != "tools"]
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, candidate_holdout_rows=samples)
        with pytest.raises(ValueError, match="current:.*holdout_tools_not_validated"):
            export_preset(exported, forge(row), policy(), target, **context(store, result, baseline))
    assert not target.exists()


def test_development_holdout_fixture_overlap_rejects(manifest, tmp_path):
    samples = rows("holdout")
    samples[0]["fixture_hash"] = rows()[0]["fixture_hash"]
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, candidate_holdout_rows=samples)
        with pytest.raises(ValueError, match="current:.*development_holdout_fixture_overlap"):
            export_preset(exported, forge(row), policy(), target, **context(store, result, baseline))
    assert not target.exists()


def test_analysis_errors_reject_both_artifacts(manifest, tmp_path):
    target = tmp_path / "preset.json"
    relaxed = policy(require_holdout=False)
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, relaxed,
                                                              candidate_rows=rows()[:-1])
        assert any("required_task_missing" in error for error in row["analysis_errors"])
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(ValueError, match="supplied:.*analysis_error:required_task_missing"):
                export(exported, row, relaxed, target, **context(store, result, baseline))
            with pytest.raises(ValueError, match="current:.*analysis_error:required_task_missing"):
                export(exported, forge(row), relaxed, target, **context(store, result, baseline))
    assert not target.exists()


def test_missing_reference_holdout_rejects(manifest, tmp_path):
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        base = reference(manifest)
        campaign, baseline, candidate = build(store, base, holdout=False)
        held = plan_holdout_candidates([manifest_of(store, candidate)], [selection("holdout")])[0]
        campaign["results"].append({"attempt_id": insert(store, held, rate=160.)})
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        row = next(item for item in result["results"] if item["attempt_id"] == candidate)
        assert row["holdout"]["attempt_id"] and row["holdout"]["reference_attempt_id"] is None
        exported = store.results(candidate)["manifest"]
        with pytest.raises(ValueError, match="supplied:.*reference_holdout_attempt_missing"):
            export_preset(exported, row, policy(), target, **context(store, result, baseline))
        with pytest.raises(ValueError, match="current:.*reference_holdout_attempt_missing"):
            export_preset(exported, forge(row), policy(), target, **context(store, result, baseline))
    assert not target.exists()


def test_same_development_and_holdout_attempt_is_not_separate_validation(manifest, tmp_path):
    target = tmp_path / "preset.json"
    raw = reference(manifest).model_dump(mode="json")
    raw["tasks"] = [selection().model_dump(mode="json"), selection("holdout").model_dump(mode="json")]
    combined = ExperimentManifest.model_validate_json(json.dumps(raw))
    with Store(tmp_path / "store") as store:
        key = insert(store, combined, samples=rows() + rows("holdout"))
        campaign = {"results": [{"attempt_id": key}]}
        result = analyze_campaign(store, campaign, policy(), key, resamples=100)
        row = result["results"][0]
        assert row["holdout"]["attempt_id"] == key  # Analysis associates the same attempt with itself.
        opt_in = {"allow_self_reference": True}  # Isolate the separate-holdout guard from the self-baseline one.
        exported = store.results(key)["manifest"]
        with pytest.raises(ValueError, match="supplied:.*holdout_not_separate_from_development"):
            export_preset(exported, row, policy(), target, **context(store, result, key), **opt_in)
        forged = forge(row)
        with pytest.raises(ValueError, match="current:.*holdout_not_separate_from_development"):
            export_preset(exported, forged, policy(), target, **context(store, result, key), **opt_in)
        forged["holdout"]["attempt_id"] = key
        with pytest.raises(ValueError, match="supplied:.*holdout_not_separate_from_development"):
            export_preset(exported, forged, policy(), target, **context(store, result, key), **opt_in)
    assert not target.exists()


def test_stale_policy_rejects_after_fresh_recomputation(manifest, tmp_path):
    target = tmp_path / "preset.json"
    stricter = policy(minimum_tokens_per_second=200.)
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        with pytest.raises(ValueError, match="supplied:.*policy_identity_mismatch"):
            export_preset(exported, row, stricter, target, **context(store, result, baseline))
        relabeled = {**deepcopy(row), "policy_hash": hashlib.sha256(
            canonical_json(stricter.model_dump(mode="json")).encode()).hexdigest()}
        with pytest.raises(ValueError, match="current:.*speed_gate_failed"):
            export_preset(exported, relabeled, stricter, target, **context(store, result, baseline))
        assert not target.exists()
        payload = export_preset(exported, row, policy(), target, **context(store, result, baseline))
        assert payload["kind"] == "validated-preset" and payload["applied"] is False


def test_changed_manifest_rejects(manifest, tmp_path):
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        changed = deepcopy(exported)
        changed["backend"]["batch_size"] = 256 if changed["backend"].get("batch_size") != 256 else 128
        with pytest.raises(ValueError, match="supplied:.*manifest_identity_mismatch"):
            export_preset(changed, row, policy(), target, **context(store, result, baseline))
        relabeled = {**deepcopy(row), "manifest_hash": ExperimentManifest.model_validate_json(
            canonical_json(changed)).fingerprint()}
        with pytest.raises(ValueError, match="manifest_identity_mismatch"):
            export_preset(changed, relabeled, policy(), target, **context(store, result, baseline))
        with pytest.raises(ValueError, match="supplied:.*manifest_identity_mismatch"):
            export_preset(store.results(baseline)["manifest"], row, policy(), target,
                          **context(store, result, baseline))
        with pytest.raises(ValueError):
            export_preset({**exported, "unknown_field": 1}, row, policy(), target,
                          **context(store, result, baseline))
    assert not target.exists()


@pytest.mark.parametrize("campaign_shape", ["stale_snapshot", "current", "reordered"])
def test_later_failed_holdout_supersedes_earlier_success(manifest, tmp_path, campaign_shape):
    target = tmp_path / "preset.json"
    failed = rows("holdout")
    for sample in failed:
        sample["score"] = 0.
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        assert row["eligibility"] == {"eligible": True, "reasons": [], "unmeasured_categories": []}
        held = plan_holdout_candidates([manifest_of(store, candidate)], [selection("holdout")])[0]
        later = insert(store, held, samples=failed, rate=160.)
        campaign = deepcopy(result)
        if campaign_shape == "current":
            campaign["all_attempt_ids"].append(later)
        elif campaign_shape == "reordered":  # Listing the earlier success last must not cherry-pick it.
            earlier = row["holdout"]["attempt_id"]
            campaign["all_attempt_ids"] = [key for key in campaign["all_attempt_ids"] if key != earlier]
            campaign["all_attempt_ids"] += [later, earlier]
        with pytest.raises(ValueError, match="current:.*holdout_not_passed"):
            export_preset(exported, row, policy(), target, **context(store, campaign, baseline))
        assert not target.exists()
        # A yet later successful holdout is the current evidence again and is the one referenced.
        newest = insert(store, held, rate=160.)
        payload = export_preset(exported, row, policy(), target, **context(store, campaign, baseline))
        assert payload["holdout_attempt_id"] == newest and newest in payload["analyzed_attempt_ids"]


def test_malformed_current_analysis_rejects(validated_store, tmp_path, monkeypatch):
    store, result, baseline, candidate, row, exported = validated_store
    target = tmp_path / "preset.json"
    hashed = result["analysis_policy_hash"]
    unverdicted = {key: value for key, value in row.items() if key != "eligibility"}
    outputs = [None, {}, [], {"results": "rows", "analysis_policy_hash": hashed},
               {"results": [], "analysis_policy_hash": hashed},
               {"results": [None], "analysis_policy_hash": hashed},
               {"results": [row, row], "analysis_policy_hash": hashed}, {"results": [row]},
               {"results": [row], "analysis_policy_hash": "0" * 64},
               {"results": [unverdicted], "analysis_policy_hash": hashed},
               {"results": [{**row, "eligibility": {"eligible": "yes", "reasons": []}}],
                "analysis_policy_hash": hashed},
               {"results": [{**row, "holdout": None}], "analysis_policy_hash": hashed},
               {"results": [{**row, "holdout": {**row["holdout"], "attempt_id": "not-in-store"}}],
                "analysis_policy_hash": hashed}]
    for output in outputs:
        monkeypatch.setattr(reports.analysis, "analyze_campaign", lambda *a, _output=output, **kw: _output)
        with pytest.raises(ValueError, match="preset export refused"):
            export_preset(exported, row, policy(), target, **context(store, result, baseline))
        assert not target.exists()
    def exploding(*args, **kwargs):
        raise KeyError("missing attempt")
    monkeypatch.setattr(reports.analysis, "analyze_campaign", exploding)
    with pytest.raises(ValueError, match="current analysis failed"):
        export_preset(exported, row, policy(), target, **context(store, result, baseline))
    assert not target.exists()


def test_synthetic_origin_cannot_be_forged_into_a_preset(manifest, tmp_path):
    target = tmp_path / "preset.json"
    relaxed = policy(require_holdout=False)
    with Store(tmp_path / "store") as store:
        key = insert(store, reference(manifest), synthetic=True)
        result = analyze_campaign(store, {"results": [{"attempt_id": key}]}, relaxed, key, resamples=100)
        exported = store.results(key)["manifest"]
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(ValueError, match="current:.*not_completed_measured_evidence"):
                export(exported, forge(result["results"][0]), relaxed, target, **context(store, result, key),
                       allow_self_reference=True)
    assert not target.exists()


def test_provisional_export_is_distinct_and_records_blockers(manifest, tmp_path):
    relaxed = policy(require_holdout=False)
    target = tmp_path / "provisional.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, relaxed, holdout=False)
        with pytest.raises(ValueError, match="authoritative"):
            export_provisional_preset(exported, row, relaxed, target)
        # A policy that requires holdout does not even pass screening for this attempt.
        strict_result = analyze_campaign(store, result, policy(), baseline, resamples=100)
        strict_row = next(item for item in strict_result["results"] if item["attempt_id"] == candidate)
        with pytest.raises(ValueError, match="screening:holdout_unverified"):
            export_provisional_preset(exported, strict_row, policy(), target,
                                      **context(store, strict_result, baseline))
        assert not target.exists()
        payload = export_provisional_preset(exported, row, relaxed, target,
                                            **context(store, result, baseline))
        written = json.loads(target.read_text(encoding="utf-8"))
        assert written == payload
        assert written["kind"] == "provisional-preset" and written["validated"] is False
        assert written["applied"] is False and written["evidence_attempt_id"] == candidate
        assert written["holdout_attempt_id"] is None and written["reference_holdout_attempt_id"] is None
        assert "analysis:holdout_not_validated" in written["validation_blockers"]
        assert "holdout_error:candidate_holdout_missing" in written["validation_blockers"]
        assert written["policy_hash"] == result["analysis_policy_hash"]
        assert written["manifest_hash"] == row["manifest_hash"]
        # The provisional artifact is not evidence: validated export still refuses.
        with pytest.raises(ValueError):
            export_preset(exported, row, relaxed, tmp_path / "preset.json",
                          **context(store, result, baseline))
        assert not (tmp_path / "preset.json").exists()


def test_provisional_export_never_claims_validation_even_when_holdout_passed(manifest, tmp_path):
    target = tmp_path / "provisional.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        payload = export_provisional_preset(exported, row, policy(), target,
                                            **context(store, result, baseline))
    assert payload["kind"] == "provisional-preset" and payload["validated"] is False
    assert payload["validation_blockers"] == ["validated_export_not_performed"]
    assert "validated-preset" not in target.read_text(encoding="utf-8")


def test_provisional_export_refuses_an_executed_failed_holdout(manifest, tmp_path):
    failed = rows("holdout")
    for sample in failed:
        sample["score"] = 0.
    relaxed = policy(require_holdout=False)
    target = tmp_path / "provisional.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, relaxed,
                                                              candidate_holdout_rows=failed)
        assert row["holdout"]["attempt_id"] and row["holdout"]["reference_attempt_id"]
        with pytest.raises(ValueError, match="holdout was executed and did not validate"):
            export_provisional_preset(exported, row, relaxed, target, **context(store, result, baseline))
    assert not target.exists()


# ---- REV-033..REV-041: findings from the independent review of the first repair ----------------


def four_attempts(store, manifest, *, candidate_changes=None, candidate_rate=160.):
    base = reference(manifest)
    candidate_manifest = propose(base, {"backend.k_cache": "q8_0"}, "kv")
    baseline = insert(store, base)
    candidate = insert(store, candidate_manifest, rate=candidate_rate, changes=candidate_changes)
    held = plan_holdout_candidates([base, candidate_manifest], [selection("holdout")])
    ids = [baseline, candidate, insert(store, held[0]), insert(store, held[1], rate=160.)]
    return {"all_attempt_ids": ids}, baseline, candidate


def row_of(result, attempt):
    return next(item for item in result["results"] if item["attempt_id"] == attempt)


@pytest.mark.parametrize("flag,key", [("require_effective_settings_verified", "effective_settings_verified"),
                                      ("require_actual_context_verified", "actual_context_verified")])
def test_rev033_relaxed_settings_or_context_flag_cannot_reach_validated_preset(manifest, tmp_path, flag, key):
    relaxed = policy(**{flag: False})
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        campaign, baseline, candidate = four_attempts(store, manifest, candidate_changes={key: False})
        result = analyze_campaign(store, campaign, relaxed, baseline, resamples=100)
        row = row_of(result, candidate)
        assert row[key] is False and row["holdout_passed"] is True
        exported = store.results(candidate)["manifest"]
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(ValueError, match=f"supplied:{key}_not_true"):
                export(exported, row, relaxed, target, **context(store, result, baseline))
            with pytest.raises(ValueError, match=f"current:.*{key}_not_true"):
                export(exported, forge(row), relaxed, target, **context(store, result, baseline))
    assert not target.exists()


def test_rev033_truncated_development_context_cannot_reach_validated_preset(manifest, tmp_path):
    relaxed = policy(require_actual_context_verified=False)
    truncated = rows()
    for sample in truncated:
        if sample["category"] == "retrieval":
            sample["context"].update(truncation_reported=True, actual_context_verified=False)
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, relaxed, candidate_rows=truncated)
        assert row["context_accounting"]["verified"] is False and row["holdout_passed"] is True
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(ValueError, match="supplied:.*context_accounting_not_verified"):
                export(exported, row, relaxed, target, **context(store, result, baseline))
            with pytest.raises(ValueError, match="current:.*context_accounting_not_verified"):
                export(exported, forge(row), relaxed, target, **context(store, result, baseline))
    assert not target.exists()


def test_rev034_self_baseline_requires_explicit_recorded_opt_in(manifest, tmp_path):
    from llmbench.config import AcceptancePolicy, CampaignPolicy, RunMode
    strict = CampaignPolicy(name="analysis-test", mode=RunMode.LIVE, acceptance=AcceptancePolicy())
    degraded = rows(), rows("holdout")
    for samples in degraded:
        for sample in [item for item in samples if item["category"] == "coding"][:3]:
            sample["score"] = 0.
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        campaign, baseline, candidate = build(store, reference(manifest), candidate_rows=degraded[0],
                                              candidate_holdout_rows=degraded[1])
        honest = analyze_campaign(store, campaign, strict, baseline, resamples=100)
        assert row_of(honest, candidate)["eligibility"]["eligible"] is False
        selfref = analyze_campaign(store, campaign, strict, candidate, resamples=100)
        row = row_of(selfref, candidate)
        assert row["eligibility"] == {"eligible": True, "reasons": [], "unmeasured_categories": []}  # Zero loss is definitional here.
        exported = store.results(candidate)["manifest"]
        for export in (export_preset, export_provisional_preset):
            for opt_in in ({}, {"allow_self_reference": False}, {"allow_self_reference": 1}):
                with pytest.raises(ValueError, match="self_reference_requires_explicit_allow_self_reference"):
                    export(exported, row, strict, target, **context(store, selfref, candidate), **opt_in)
                assert not target.exists()
        payload = export_preset(exported, row, strict, target, **context(store, selfref, candidate),
                                allow_self_reference=True)
        assert payload["reference_self_validated"] is True and payload["comparison_family"] == "reference"
        assert payload["baseline_attempt_id"] == payload["evidence_attempt_id"] == candidate
        assert payload["holdout_attempt_id"] == payload["reference_holdout_attempt_id"] != candidate
        assert json.loads(target.read_text(encoding="utf-8")) == payload
        # The opt-in never relaxes a real comparison: the true baseline still refuses this candidate.
        with pytest.raises(ValueError, match="coding"):
            export_preset(exported, row_of(honest, candidate), strict, tmp_path / "other.json",
                          **context(store, honest, baseline), allow_self_reference=True)
        assert not (tmp_path / "other.json").exists()


@pytest.mark.parametrize("declared", [True, False])
def test_rev036_omitted_development_attempt_cannot_hide_fixture_overlap(manifest, tmp_path, declared):
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        base = reference(manifest)
        candidate_manifest = propose(base, {"backend.k_cache": "q8_0"}, "kv")
        raw = candidate_manifest.model_dump(mode="json")
        raw["tasks"] = [TaskSelection(suite="private-mixed", revision="immutable-v1",
                                      task_ids=tasks("tuning")).model_dump(mode="json")]
        tuning = ExperimentManifest.model_validate_json(canonical_json(raw))
        leaked = rows("holdout")  # An earlier tuning run on fixtures later relabeled as unseen holdout.
        for sample, name in zip(leaked, tasks("tuning")):
            sample.update(task_id=name, split="development")
        contaminating = insert(store, tuning, samples=leaked, rate=160.)
        campaign, baseline, candidate = build(store, base, candidate_manifest)
        if declared:
            campaign["results"].append({"attempt_id": contaminating})
        result = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        row = row_of(result, candidate)
        assert row["eligibility"]["eligible"] is False
        assert contaminating in result["all_attempt_ids"]
        assert result["winner"] is None and result["best_presets"] == {}
        with pytest.raises(ValueError, match="current:.*development_holdout_fixture_overlap"):
            export_preset(store.results(candidate)["manifest"], forge(row), policy(), target,
                          **context(store, result, baseline))
    assert not target.exists()


@pytest.mark.parametrize("stripped", ["candidate", "candidate_holdout", "baseline", "reference_holdout"])
def test_rev037_samples_of_unknown_origin_fail_closed(manifest, tmp_path, stripped):
    def unknown(split="development"):
        samples = rows(split)
        for sample in samples:
            del sample["synthetic"], sample["model_evaluated"]
        return samples
    target = tmp_path / "preset.json"
    base = reference(manifest)
    candidate_manifest = propose(base, {"backend.k_cache": "q8_0"}, "kv")
    held = plan_holdout_candidates([base, candidate_manifest], [selection("holdout")])
    with Store(tmp_path / "store") as store:
        ids = [insert(store, base, samples=unknown() if stripped == "baseline" else None),
               insert(store, candidate_manifest, rate=160., samples=unknown() if stripped == "candidate" else None),
               insert(store, held[0], samples=unknown("holdout") if stripped == "reference_holdout" else None),
               insert(store, held[1], rate=160.,
                      samples=unknown("holdout") if stripped == "candidate_holdout" else None)]
        result = analyze_campaign(store, {"all_attempt_ids": ids}, policy(), ids[0], resamples=100)
        row = row_of(result, ids[1])
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(ValueError, match="current:.*sample_origin_unverified"):
                export(store.results(ids[1])["manifest"], forge(row), policy(), target,
                       **context(store, result, ids[0]))
    assert not target.exists()


@pytest.mark.parametrize("tampered", ["candidate", "candidate_holdout", "baseline", "reference_holdout"])
def test_rev038_later_evidence_event_never_replaces_sealed_evidence(manifest, tmp_path, tampered):
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        campaign, baseline, candidate = four_attempts(
            store, manifest, candidate_rate=30. if tampered == "candidate" else 160.)
        index = ["baseline", "candidate", "reference_holdout", "candidate_holdout"].index(tampered)
        victim = campaign["all_attempt_ids"][index]
        first = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        assert row_of(first, candidate)["eligibility"]["eligible"] is (tampered != "candidate")
        fast = json.loads(next(event["payload"] for event in store.results(campaign["all_attempt_ids"][3])["events"]
                               if event["kind"] == "evidence"))
        store.event(victim, "evidence", {**fast, "attempt_id": victim})  # Appended after completion.
        second = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        row = row_of(second, candidate)
        assert row["eligibility"]["eligible"] is False
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(ValueError, match="current:.*evidence_events_not_exactly_one"):
                export(store.results(candidate)["manifest"], forge(row), policy(), target,
                       **context(store, second, baseline))
    assert not target.exists()


def test_rev039_exclusive_create_is_the_guard_and_the_file_is_synced(manifest, tmp_path, monkeypatch):
    target = tmp_path / "preset.json"
    real = reports.analysis.analyze_campaign
    def racing(*args, **kwargs):  # The destination appears after the early exists() check, during analysis.
        value = real(*args, **kwargs)
        target.write_text("racer-owned", encoding="utf-8")
        return value
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        monkeypatch.setattr(reports.analysis, "analyze_campaign", racing)
        for export in (export_preset, export_provisional_preset):
            with pytest.raises(FileExistsError):
                export(exported, row, policy(), target, **context(store, result, baseline))
            assert target.read_text(encoding="utf-8") == "racer-owned"
            target.unlink()
        monkeypatch.undo()
        synced = []
        real_fsync = reports.os.fsync
        monkeypatch.setattr(reports.os, "fsync", lambda descriptor: (synced.append(descriptor), real_fsync(descriptor)))
        export_preset(exported, row, policy(), target, **context(store, result, baseline))
    assert len(synced) == 1 and json.loads(target.read_text(encoding="utf-8"))["kind"] == "validated-preset"


def test_rev039_lax_policy_is_visible_in_the_payload(manifest, tmp_path):
    lax = policy(minimum_tokens_per_second=1., minimum_coding_score=0., minimum_tool_score=0.,
                 minimum_retrieval_score=0.)
    lax = lax.model_copy(update={"acceptance": lax.acceptance.model_copy(update={"max_quality_loss": 1.})})
    bad = rows(), rows("holdout")
    for samples in bad:
        for sample in samples:
            if sample["category"] == "coding":
                sample["score"] = 0.
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest, lax, candidate_rows=bad[0],
                                                              candidate_holdout_rows=bad[1])
        payload = export_preset(exported, row, lax, tmp_path / "preset.json", **context(store, result, baseline))
    assert payload["acceptance"]["minimum_coding_score"] == 0 and payload["acceptance"]["max_quality_loss"] == 1
    assert payload["quality"]["coding"]["score"] == 0 == payload["holdout_quality"]["coding"]["score"]
    assert payload["quality"]["coding"]["comparison"]["mean_delta"] == -1


@pytest.mark.parametrize("holdout_scores_fail", [True, False])
def test_rev040_provisional_refuses_failed_candidate_holdout_without_reference(manifest, tmp_path,
                                                                               holdout_scores_fail):
    relaxed = policy(require_holdout=False)
    samples = rows("holdout")
    for sample in samples:
        sample["score"] = 0. if holdout_scores_fail else 1.
    target = tmp_path / "provisional.json"
    with Store(tmp_path / "store") as store:
        campaign, baseline, candidate = build(store, reference(manifest), holdout=False)
        held = plan_holdout_candidates([manifest_of(store, candidate)], [selection("holdout")])[0]
        campaign["results"].append({"attempt_id": insert(store, held, samples=samples, rate=160.)})
        result = analyze_campaign(store, campaign, relaxed, baseline, resamples=100)
        row = row_of(result, candidate)
        assert row["holdout"]["attempt_id"] and row["holdout"]["reference_attempt_id"] is None
        exported = store.results(candidate)["manifest"]
        if holdout_scores_fail:
            with pytest.raises(ValueError, match="holdout was executed and did not validate"):
                export_provisional_preset(exported, row, relaxed, target, **context(store, result, baseline))
            assert not target.exists()
        else:  # Only the reference comparison is outstanding: still provisional, with the gap recorded.
            payload = export_provisional_preset(exported, row, relaxed, target,
                                                **context(store, result, baseline))
            assert payload["kind"] == "provisional-preset"
            assert "reference_holdout_attempt_missing" in payload["validation_blockers"]


@pytest.mark.parametrize("declared", [True, False])
def test_rev041_later_failed_development_rerun_supersedes(manifest, tmp_path, declared):
    failed = rows()
    for sample in failed:
        sample["score"] = 0.
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        rerun = insert(store, manifest_of(store, candidate), samples=failed, rate=160.)
        campaign = deepcopy(result)
        if declared:
            campaign["all_attempt_ids"].append(rerun)
        with pytest.raises(ValueError, match=f"current:.*superseded_by_later_failed_attempt:{rerun}"):
            export_preset(exported, row, policy(), target, **context(store, campaign, baseline))
        assert not target.exists()
        newest = insert(store, manifest_of(store, candidate), rate=160.)  # A later good re-run is exportable.
        campaign["all_attempt_ids"].append(newest)
        current = analyze_campaign(store, campaign, policy(), baseline, resamples=100)
        payload = export_preset(exported, row_of(current, newest), policy(), target,
                                **context(store, current, baseline))
        assert payload["evidence_attempt_id"] == newest


def test_store_manifest_that_no_longer_validates_refuses_with_export_error(manifest, tmp_path):
    target = tmp_path / "preset.json"
    with Store(tmp_path / "store") as store:
        result, baseline, candidate, row, exported = analyzed(store, manifest)
        store.db.execute("INSERT INTO experiments VALUES ('legacy', '{\"schema_version\": 0}', 'then')")
        store.db.execute("INSERT INTO attempts VALUES ('legacy-attempt','legacy','completed',0,'then','then',NULL)")
        store.db.commit()
        with pytest.raises(ValueError, match="preset export refused: .*legacy-attempt"):
            export_preset(exported, row, policy(), target, **context(store, result, baseline))
    assert not target.exists()
