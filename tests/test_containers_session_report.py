"""Session report (AM-1 item 2): one row per attempt, evidence-labelled recommendation, escaped HTML. Fake runner only."""

import html
import json
import re

import pytest

from llmbench.analysis import plan_holdout_candidates
from llmbench.containers.campaign import ContainerExecutor
from llmbench.containers.session import holdout_selections, policy_for, read_ledger
from llmbench.containers.session_report import (CATEGORIES, COLUMNS, CONTEXT_CLAIM_NOTE, FILL_VERIFIED_NOTE,
                                                HEADERS, NO_MEASURED_FILL, NONE, NOT_COVERED, NOT_OBSERVED,
                                                OFFLOAD_NOTE, SCOPE_UNKNOWN, SCOPE_UNKNOWN_SUFFIX, SCREENING,
                                                UNEXPLAINED, UNVERIFIED_FILL, VALIDATED, _holdout_suffix,
                                                attempt_row, build_session_report, recommendation, session_html,
                                                session_markdown, write_session_report)
from llmbench.controller import execute_attempt
from llmbench.store import Store
from test_containers_campaign import candidates, run_tune, small_session
from test_containers_session import LOCK, isolated_gpu_lock

_ = isolated_gpu_lock  # fixture re-export for pytest collection


def test_columns_and_headers_cover_every_required_field():
    assert len(COLUMNS) == len(HEADERS) == len(set(COLUMNS))
    for name in ("label", "quantization", "ctx", "kv", "flash_attn", "speculation", "reasoning", "batch_ubatch",
                 "state", "tps_min", "tps_median", "tps_max", "prefill_tps", "ttft_s", "load_s", "vram_mib", "tools",
                 "retrieval", "coding", "first_attempt_vs_repaired", "context_verified", "settings_verified",
                 "eligibility_reasons", "holdout", "artifact_dir"):
        assert name in COLUMNS, name


def test_report_lists_one_row_per_attempt_including_failures_and_holdouts(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path, quantizations=("Q4_K_M", "Q6_K"))  # baseline, kv, weights
    reason = "inference was not healthy within 300s"
    outcome, runner, clock = run_tune(tmp_path, session, script={
        "kv-q8_0-q8_0": {"state": "rejected", "stage": "admit", "reasons": ["foreign VRAM 8000 MiB"]},
        "weights-q6_k": {"state": "timeout", "stage": "ready", "reasons": [reason]}})
    out = tmp_path / "out"
    report = outcome["report"]
    assert report["schema_version"] == 1 and report["synthetic"] is False and report["columns"] == list(COLUMNS)
    assert [row["state"] for row in report["rows"]] == ["completed", "failed (container: rejected at admit)",
                                                       "failed (container: timeout at ready)",
                                                       "completed"]  # the surviving baseline's holdout attempt
    assert all(set(COLUMNS) <= set(row) for row in report["rows"])
    baseline = report["rows"][0]
    assert (baseline["label"], baseline["quantization"], baseline["ctx"], baseline["kv"]) == ("baseline", "Q4_K_M",
                                                                                              8192, "f16/f16")
    assert baseline["flash_attn"] is True and baseline["speculation"] == "none" and baseline["reasoning"] == "off"
    assert baseline["batch_ubatch"] == "2048/512" and baseline["split"] == "development"
    assert baseline["tps_min"] == baseline["tps_median"] == baseline["tps_max"] == 69.0
    assert baseline["prefill_tps"] == 4096 / 1.2 and baseline["ttft_s"] == 0.4
    assert baseline["load_s"] == 63.6 and baseline["vram_mib"] == 21179.0
    assert baseline["tools"] == "1.000 (2 attempted)" and baseline["retrieval"] == "1.000 (2 attempted)"
    assert baseline["coding"] == "n/a (0 attempted)" and baseline["first_attempt_vs_repaired"] == "n/a"
    assert baseline["context_verified"] is True and baseline["settings_verified"] is True
    # The baseline declared tools and retrieval, passed both, and therefore reached the holdout: it is validated
    # on what that holdout covers, and the report says exactly which categories that is.
    assert "holdout_not_validated" not in baseline["eligibility_reasons"]
    assert "retrieval" in baseline["holdout"] and not baseline["holdout"].startswith("not validated")
    assert baseline["artifact_dir"] == str(out / "candidates" / "000-baseline")
    assert baseline["candidate_report"] == "reports/candidate.md" and baseline["comparison_family"] == "reference"
    rejected, timed_out = report["rows"][1], report["rows"][2]
    assert "foreign VRAM 8000 MiB" in rejected["eligibility_reasons"] and reason in timed_out["eligibility_reasons"]
    assert rejected["tps_min"] is None and rejected["tools"] == "n/a (0 attempted)"
    assert rejected["context_verified"] is False and rejected["settings_verified"] is False
    assert rejected["artifact_dir"] == str(out / "candidates" / "001-kv-q8_0-q8_0")
    assert timed_out["quantization"] == "Q6_K" and timed_out["holdout"].startswith("not validated: candidate_holdout_missing")
    assert timed_out["artifact_dir"] == str(out / "candidates" / "002-weights-q6_k")
    # Recommendation: the baseline passed its holdout, so it is validated - on retrieval only, and says so.
    rec = report["recommendation"]
    for name in ("best_quality", "fastest_eligible", "largest_verified_context"):
        assert rec[name]["label"] == "baseline" and rec[name]["evidence"] != SCREENING
        assert rec[name]["evidence"].startswith("validated on retrieval holdout only")
        assert rec[name]["holdout_uncovered_categories"] == ["coding", "tools"]
        assert "better_unvalidated" not in rec[name]  # both other candidates failed; nothing measured better
    assert "No validated preset" in rec["preset_note"]
    # Budget accounting, reproduction commands and links.
    budget = report["budget"]
    assert budget["wall_seconds"] == 3600 and budget["stop_reason"] == "complete"
    # The controller's BudgetTracker charges the process clock (not injectable); a fake campaign can finish
    # inside one Windows clock tick, so only the sign is asserted here.
    assert budget["charged_seconds"] >= 0
    assert (budget["candidates_attempted"], budget["holdouts_attempted"], budget["max_candidates"]) == (3, 1, 4)
    assert budget["remaining_seconds"] > 0 and budget["skipped"] == []
    assert report["reproduction"][0] == f"llmbench resume --output {out}"
    assert any("candidates" in command and "001-kv-q8_0-q8_0" in command for command in report["reproduction"])
    assert set(report["campaign_reports"]) == {"json", "markdown", "html"}
    assert (out / "reports" / "report.md").is_file()
    assert set(report["candidate_reports"]) == {row["attempt_id"] for row in report["rows"]}
    assert all(path.endswith("candidate.md") for path in report["candidate_reports"].values())
    # The family is a property of the manifests, so a failed kv attempt keeps "kv". The weights sweep reads
    # "unverified": plan 4.2 maps tokenizer_hash to the GGUF sha256 (the evaluator labels samples the same way),
    # so analysis.infer_comparison_family refuses the weight comparison as uncontrolled (recorded gap, branch log).
    assert [entry["family"] for entry in report["comparisons"]] == ["reference", "kv", "unverified"]
    # Files: JSON round-trips, markdown has one table row per attempt, HTML has one <tr> per attempt.
    stored = json.loads((out / "reports" / "session.json").read_text(encoding="utf-8"))
    assert [row["state"] for row in stored["rows"]] == [row["state"] for row in report["rows"]]
    markdown = (out / "reports" / "session.md").read_text(encoding="utf-8")
    table = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert len(table) == 1 + 4 and table[0] == "| " + " | ".join(HEADERS) + " |"  # 3 development + 1 holdout
    assert "## Recommendation" in markdown and "## Budget" in markdown and "## Reproduction" in markdown
    html = (out / "reports" / "session.html").read_text(encoding="utf-8")
    assert html.count("<tr>") == 1 + 4 and "session.md" not in html
    assert "validated on retrieval holdout only" in html
    # A holdout attempt (validation evidence) is its own row, never merged into the development row.
    ledger = read_ledger(out)
    pairs = candidates(session)
    executor = ContainerExecutor(session, None, runner=runner, output_root=out, deadline_epoch=clock.wall() + 3600,
                                 template_hashes=dict(ledger["template_hashes"]), clock=clock, wall=clock.wall)
    for config, manifest in pairs:
        executor.register(config, manifest)
    held = plan_holdout_candidates([pairs[0][1]], holdout_selections(session))[0]
    with Store(out) as store:
        result = execute_attempt(store, held, policy_for(session), executor, session_lock=LOCK)
        assert result["status"] == "completed"
        again = build_session_report(store, outcome["campaign"], session, ledger, output_root=out,
                                     stop_reason="complete", remaining_seconds=1.0)
    # The session already ran the surviving baseline's holdout; this second one is the attempt executed just above.
    assert len(again["rows"]) == 5 and [row["split"] for row in again["rows"][3:]] == ["holdout", "holdout"]
    for row in again["rows"][3:]:
        assert row["label"] == "baseline-holdout" and row["holdout"].startswith("holdout attempt")
    assert again["budget"]["holdouts_attempted"] == 2 and again["budget"]["candidates_attempted"] == 3


def test_rows_show_the_context_allocation_beside_the_measured_input_fill(tmp_path, isolated_gpu_lock):
    """U16: allocation and measured input stand side by side; the claim is expressed in usable input tokens."""
    session = small_session(tmp_path, ctx_tiers=(2816, 6912), kv_pairs=(("f16", "f16"),), context_floor=2048,
                            context_ceiling=6000, context_n_ctx_train=262144)
    outcome, runner, clock = run_tune(tmp_path, session)
    report = outcome["report"]
    assert [row["label"] for row in report["rows"]] == ["baseline", "ctx-6912", "baseline-holdout"]
    assert report["context_range"] == {"context_floor": 2048, "context_ceiling": 6000, "n_ctx_train": 262144,
                                       "unit": "usable input tokens", "dropped_input_targets": [],
                                       "scheduled_input_targets": [2048, 6000], "unscheduled_input_targets": []}
    baseline, largest = report["rows"][:2]
    assert (baseline["ctx"], baseline["input_requested"], baseline["input_actual"]) == (2816, 2048, 2043)
    assert (largest["ctx"], largest["input_requested"], largest["input_actual"]) == (6912, 6000, 5995)
    # The measured fill is a measurement, never the requested count the controller stores under the same name.
    assert all(row["input_actual"] < row["input_requested"] < row["ctx"] for row in report["rows"][:2])
    assert largest["input_measured_maximum"] == 6000 and largest["input_fill_verified"] is True
    # The context claim names usable INPUT tokens and points at the attempt that ran at that fill.
    # The VALIDATED context claim is the baseline's, because only the baseline can pass a non-inferiority test at
    # this item count. The larger run is not hidden behind it: it is named as measured-but-not-validated.
    claim = report["recommendation"]["largest_verified_context"]
    assert claim["label"] == "baseline" and claim["usable_input_tokens"] == 2043
    assert claim["context_allocation"] == 2816 and claim["evidence"].startswith("validated on retrieval holdout only")
    assert claim["claim_note"] == CONTEXT_CLAIM_NOTE
    larger = claim["better_unvalidated"]
    assert larger["label"] == "ctx-6912" and larger["measured_input_tokens"] == 5995
    assert larger["evidence"] == SCREENING and larger["blockers"]
    markdown = (tmp_path / "out" / "reports" / "session.md").read_text(encoding="utf-8")
    assert "| Ctx allocation | Input requested | Input measured |" in markdown.replace("Label | Quant | ", "")
    assert "| 6912 | 6000 | 5995 |" in markdown and "2043 usable input tokens measured" in markdown
    assert "measured better but NOT validated: `ctx-6912`" in markdown and "5995 input tokens" in markdown
    assert "## Context range (usable input tokens)" in markdown and "2048..6000 usable INPUT tokens" in markdown
    # Both ways a declared target can vanish are reported: derivation thinning and schedule truncation.
    assert "- Input targets the pinned schedule carries: 2048, 6000" in markdown
    assert "- Input targets dropped for the candidate cap when the tiers were derived: none" in markdown
    assert ("- Declared input targets the candidate cap cut out of the schedule (never measured): none"
            in markdown)
    html = (tmp_path / "out" / "reports" / "session.html").read_text(encoding="utf-8")
    assert "<td>6912</td><td>6000</td><td>5995</td>" in html and "usable INPUT tokens" in html
    assert "Input targets the pinned schedule carries: 2048, 6000" in html
    # Without an analysed row the fill still comes from the attempt's own samples, labelled unverified.
    with Store(tmp_path / "out") as store:
        row = attempt_row(store, store.attempts_in_insertion_order()[0], None)
    assert (row["input_actual"], row["input_requested"], row["input_fill_verified"]) == (2043, 2048, False)
    # REV-LIVE-04: a schedule that omits a declared target can never be reported as "dropped: none". The pinned
    # schedule, not the declared tier list, decides what was searched (the live-smoke failure mode).
    ledger = read_ledger(tmp_path / "out")
    thinned = {**ledger, "proposals": [item for item in ledger["proposals"]
                                       if (item.get("changes") or {}).get("ctx_tier") != 6912]}
    with Store(tmp_path / "out") as store:
        cut = build_session_report(store, outcome["campaign"], session, thinned, output_root=tmp_path / "out",
                                   stop_reason="candidates", remaining_seconds=1.0)
    assert cut["context_range"]["scheduled_input_targets"] == [2048]
    assert cut["context_range"]["unscheduled_input_targets"] == [6000]
    text = session_markdown(cut)
    assert "- Input targets the pinned schedule carries: 2048" in text
    assert "- Declared input targets the candidate cap cut out of the schedule (never measured): 6000" in text


def test_an_unverified_input_fill_is_marked_wherever_it_is_rendered(tmp_path, isolated_gpu_lock):
    """REV-LIVE-07: an unverified fallback count must never render like a verified measurement.

    Every row of the first live campaign carried ``input_fill_verified: false`` while `Input measured` rendered
    a bare number, and the skill tells the reader to quote that column.
    """
    import copy
    session = small_session(tmp_path, ctx_tiers=(2816, 6912), kv_pairs=(("f16", "f16"),), context_floor=2048,
                            context_ceiling=6000, context_n_ctx_train=262144)
    outcome, runner, clock = run_tune(tmp_path, session)
    report = outcome["report"]
    assert all(row["input_fill_verified"] is True for row in report["rows"]
               if not row["label"].endswith("-holdout"))  # the fake runner's holdout evidence carries no fill
    verified = session_markdown(report)
    assert all(UNVERIFIED_FILL not in line for line in verified.splitlines() if line.startswith("| 2816 |")
               or line.startswith("| 6912 |"))
    unverified = copy.deepcopy(report)  # exactly the live-smoke shape: measured numbers, none of them verified
    for row in unverified["rows"]:
        row["input_fill_verified"] = False
    markdown = session_markdown(unverified)
    assert "| 6912 | 6000 | 5995 (unverified) |" in markdown and "| 2816 | 2048 | 2043 (unverified) |" in markdown
    assert FILL_VERIFIED_NOTE in markdown and "verified" in FILL_VERIFIED_NOTE
    html = session_html(unverified)
    assert "<td>5995 (unverified)</td>" in html and "input_fill_verified" in html
    # An absent measurement is not an unverified one: nothing measured still renders "-", never "- (unverified)".
    unverified["rows"][0]["input_actual"] = None
    assert "| 2816 | 2048 | - |" in session_markdown(unverified)


def test_rows_show_the_requested_offload_beside_what_the_readback_observed(tmp_path, isolated_gpu_lock):
    """U17: GPU layers and KV placement are columns, and an effective value is never invented."""
    session = small_session(tmp_path)
    outcome, runner, clock = run_tune(tmp_path, session)
    report = outcome["report"]
    assert COLUMNS.index("kv_placement") < COLUMNS.index("gpu_layers") < COLUMNS.index("flash_attn")
    baseline = report["rows"][0]
    # The fake runner stores an EMPTY settings evidence list: the request is shown, nothing is claimed observed.
    assert baseline["gpu_layers"] == f"all ({NOT_OBSERVED})" and baseline["kv_placement"] == f"gpu ({NOT_OBSERVED})"
    assert baseline["gpu_layers_effective"] is None and baseline["kv_placement_effective"] is None
    assert baseline["gpu_layers_status"] == "no-evidence" and baseline["kv_placement_status"] == "no-evidence"
    markdown = (tmp_path / "out" / "reports" / "session.md").read_text(encoding="utf-8")
    assert "| KV placement | GPU layers |" in markdown and OFFLOAD_NOTE in markdown
    assert f"| gpu ({NOT_OBSERVED}) | all ({NOT_OBSERVED}) |" in markdown
    assert "inference_memory_mib" in OFFLOAD_NOTE  # the container's limit, not host RAM, bounds a RAM offload
    assert OFFLOAD_NOTE in session_html(report) or html.escape(OFFLOAD_NOTE) in session_html(report)

    def rebuilt(*rows) -> dict:
        with Store(tmp_path / "out") as store:
            attempt = store.attempts_in_insertion_order()[0]
            (store.root / "raw" / attempt["id"] / "settings-evidence.json").write_text(
                json.dumps({"settings": list(rows), "unverified_required": []}), encoding="utf-8")
            return attempt_row(store, attempt, None)

    def evidence(control, requested, effective, status) -> dict:
        return {"control": control, "requested": requested, "effective": effective, "status": status,
                "evidence": "startup log", "required": True}

    observed = rebuilt(evidence("n_gpu_layers", 24, "24/48", "verified-log"),
                       evidence("kv_offload", False, False, "verified-log"))
    assert observed["gpu_layers"] == "24 (24/48 offloaded to GPU)" and observed["kv_placement"] == "ram (observed ram)"
    assert observed["gpu_layers_effective"] == "24/48" and observed["kv_placement_effective"] is False
    # A mismatch is labelled as one; the engine offloading fewer layers than asked must not read as agreement.
    mismatched = rebuilt(evidence("n_gpu_layers", 24, "20/48", "mismatch"),
                         evidence("kv_offload", False, True, "mismatch"))
    assert mismatched["gpu_layers"] == "24 (mismatch: 20/48 offloaded to GPU)"
    assert mismatched["kv_placement"] == "ram (mismatch: observed gpu)"
    # An unobserved control keeps the request and says nothing was observed; no count is ever fabricated.
    silent = rebuilt(evidence("n_gpu_layers", 24, None, "unobserved"),
                     evidence("kv_offload", False, None, "unobserved"))
    assert silent["gpu_layers"] == f"24 ({NOT_OBSERVED})" and silent["kv_placement"] == f"ram ({NOT_OBSERVED})"
    assert "48" not in silent["gpu_layers"] and silent["gpu_layers_effective"] is None
    # Without any settings evidence the manifest still names the fraction that was requested, never a count.
    assert rebuilt()["gpu_layers"] == f"all ({NOT_OBSERVED})"


def test_a_split_kv_cache_is_never_reported_as_all_on_the_gpu(tmp_path, isolated_gpu_lock):
    """REV-OFF-01: the cell for the shape U17 was written to sweep, which no test exercised.

    Both live partial-offload candidates rendered `KV placement: gpu (observed gpu)` while a quarter and a
    half of their KV cache respectively sat in system RAM: `_offload_columns` evaluated the row's `effective`
    for truth, and LIVE-010 had made a split a non-empty (therefore truthy) value. The placement is now read,
    not re-derived, and the cell names both sides of a split.
    """
    session = small_session(tmp_path)
    run_tune(tmp_path, session)

    def rebuilt(*rows) -> dict:
        with Store(tmp_path / "out") as store:
            attempt = store.attempts_in_insertion_order()[0]
            (store.root / "raw" / attempt["id"] / "settings-evidence.json").write_text(
                json.dumps({"settings": list(rows), "unverified_required": []}), encoding="utf-8")
            return attempt_row(store, attempt, None)

    def kv_row(effective, status="verified-log", **extra) -> dict:
        return {"control": "kv_offload", "requested": True, "effective": effective, "status": status,
                "evidence": "startup log", "required": True, **extra}

    # The live `001-offload-48` row, as `readback` seals it now: state in `effective`, MiB in `detail`.
    split = rebuilt(kv_row("split", detail="ram 84 MiB on CPU / gpu 252 MiB on CUDA0"))
    assert split["kv_placement"] == "gpu (observed split: ram 84 MiB on CPU / gpu 252 MiB on CUDA0)"
    assert split["kv_placement_state"] == "split" and "84 MiB" in split["kv_placement_detail"]
    assert split["kv_placement_requested"] == "gpu" and split["kv_placement_status"] == "verified-log"
    for text in (session_markdown(build_report(tmp_path)), session_html(build_report(tmp_path))):
        assert "observed split" in text and "84 MiB" in text and "252 MiB" in text
    # Evidence the LIVE-010 build sealed carries the split as a string (it is in the stored live campaigns,
    # and a report is built from what the attempt sealed): the same cell, from the older shape.
    legacy = rebuilt(kv_row("split (ram 168 MiB / gpu 168 MiB on CUDA0)"))
    assert legacy["kv_placement"] == "gpu (observed split: ram 168 MiB / gpu 168 MiB on CUDA0)"
    assert legacy["kv_placement_state"] == "split"
    # A split that contradicts the request is still labelled a mismatch, and still names both sides.
    bad = rebuilt(kv_row("split", status="mismatch", detail="ram 84 MiB on CPU / gpu 252 MiB on CUDA0"))
    assert bad["kv_placement"] == "gpu (mismatch: observed split: ram 84 MiB on CPU / gpu 252 MiB on CUDA0)"
    # An unrecognised value is not a placement: the cell says nothing was observed rather than guessing.
    assert rebuilt(kv_row("somewhere else"))["kv_placement"] == f"gpu ({NOT_OBSERVED})"
    assert rebuilt(kv_row(1))["kv_placement"] == f"gpu ({NOT_OBSERVED})"


def build_report(tmp_path) -> dict:
    """The session report as `llmbench resume` would rebuild it from the Store on disk."""
    session = small_session(tmp_path)
    ledger = read_ledger(tmp_path / "out")
    with Store(tmp_path / "out") as store:
        return build_session_report(store, {"results": []}, session, ledger, output_root=tmp_path / "out",
                                    stop_reason="complete", remaining_seconds=1.0)


def test_a_context_claim_without_a_measured_fill_is_withdrawn(tmp_path):
    """No claim from the allocation alone: a pick that never measured an input fill is not a context claim."""
    labels = {"a1": "baseline", "b2": "kv-q8_0-q8_0"}
    row = {"attempt_id": "a1", "status": "completed", "synthetic": False, "effective_settings_verified": True,
           "actual_context_verified": True, "speed": {"qualifies": True, "minimum_native_tps": 61.0},
           "quality": {"tools": {"attempted": 2, "score": 1.0, "absolute_passed": True},
                       "retrieval": {"attempted": 2, "score": 1.0, "absolute_passed": True}},
           "eligibility": {"eligible": False, "reasons": ["holdout_not_validated"]}}  # no actual_input_tokens
    block = recommendation({"best_presets": {}, "results": [row]}, labels, {"a1": 131072})
    assert block["largest_verified_context"] == {"attempt_id": None, "label": None, "evidence": NO_MEASURED_FILL,
                                                 "claim_note": CONTEXT_CLAIM_NOTE}
    assert block["best_quality"]["evidence"] == SCREENING and block["best_quality"]["attempt_id"] == "a1"
    # A validated preset with no measured input is withdrawn the same way; an allocation is not evidence.
    # The preset states its holdout scope, as `analysis.preset()` always does: a payload that does not is an
    # unknown scope and never reaches the flat validated wording (REV-OFF-03, asserted below).
    hollow = {"attempt_id": "b2", "native_tps_minimum": 88.0, "validated_input_tokens": None,
              "holdout_validated_categories": list(CATEGORIES), "holdout_uncovered_categories": []}
    validated = recommendation({"best_presets": {"largest_validated_context": hollow}, "results": []}, labels,
                               {"b2": 131072})
    assert validated["largest_verified_context"]["evidence"] == NO_MEASURED_FILL
    measured = recommendation({"best_presets": {"largest_validated_context": {**hollow, "validated_input_tokens":
                                                                             130560}}, "results": []}, labels,
                              {"b2": 131072})
    claim = measured["largest_verified_context"]
    assert claim["evidence"] == VALIDATED and claim["usable_input_tokens"] == 130560
    assert claim["claim"] == "130560 usable input tokens measured on attempt b2 in a 131072-token allocation " \
                             "(holdout validated)"
    # An unknown allocation still yields a claim: the measured fill is what is being claimed.
    assert recommendation({"best_presets": {"largest_validated_context": {**hollow, "validated_input_tokens": 4091}},
                           "results": []}, labels)["largest_verified_context"]["claim"].endswith(
        "4091 usable input tokens measured on attempt b2 (holdout validated)")


def test_recommendation_labels_validated_screening_and_none():
    labels = {"a1": "baseline", "b2": "kv-q8_0-q8_0"}
    validated = {"attempt_id": "b2", "native_tps_minimum": 88.0, "validated_input_tokens": 4091,
                 "holdout_validated_categories": list(CATEGORIES), "holdout_uncovered_categories": []}
    campaign = {"best_presets": {"highest_quality": validated, "fastest_within_quality": validated,
                                 "largest_validated_context": validated}, "results": []}
    block = recommendation(campaign, labels)
    assert all(block[name]["evidence"] == VALIDATED and block[name]["label"] == "kv-q8_0-q8_0"
               for name in ("best_quality", "fastest_eligible", "largest_verified_context"))
    row = {"attempt_id": "a1", "status": "completed", "synthetic": False, "effective_settings_verified": True,
           "actual_context_verified": True, "actual_input_tokens": 4091,
           "speed": {"qualifies": True, "minimum_native_tps": 61.0},
           "quality": {"tools": {"attempted": 2, "score": 1.0, "absolute_passed": True},
                       "retrieval": {"attempted": 2, "score": 1.0, "absolute_passed": True},
                       "coding": {"attempted": 0, "score": None}},
           "eligibility": {"eligible": False, "reasons": ["holdout_not_validated"]}}
    screening = recommendation({"best_presets": {}, "results": [row]}, labels)
    assert screening["best_quality"] == {"attempt_id": "a1", "label": "baseline", "evidence": SCREENING,
                                         "native_tps_minimum": 61.0, "measured_input_tokens": 4091,
                                         "blockers": ["holdout_not_validated"]}
    slow = {**row, "speed": {"qualifies": False, "minimum_native_tps": 12.0}}
    unverified = {**row, "actual_context_verified": False}
    failed = {**row, "quality": {**row["quality"], "tools": {"attempted": 2, "score": 0.5, "absolute_passed": False}}}
    for candidate in (slow, unverified, failed, {**row, "synthetic": True}, {**row, "status": "failed"}):
        none = recommendation({"best_presets": {}, "results": [candidate]}, labels)
        assert none["best_quality"] == {"attempt_id": None, "label": None, "evidence": NONE}
        assert none["fastest_eligible"]["evidence"] == NONE and none["largest_verified_context"]["evidence"] == NONE


def test_html_and_markdown_escape_model_and_runtime_text():
    hostile = "<script>alert('x')</script> | pipe\nnewline"
    row = {column: hostile for column in COLUMNS}
    row.update(tps_min=1.0, tps_median=2.0, tps_max=3.0, prefill_tps=None, ttft_s=None, load_s=None, vram_mib=None,
               eligibility_reasons=[hostile, "second"], context_verified=False, settings_verified=False)
    report = {"session_id": "s", "campaign_identity": "c", "synthetic": True, "columns": list(COLUMNS), "rows": [row],
              "recommendation": {"best_quality": {"attempt_id": "1", "label": hostile, "evidence": NONE},
                                 "fastest_eligible": {"attempt_id": None, "label": None, "evidence": NONE},
                                 "largest_verified_context": {"attempt_id": None, "label": None, "evidence": NONE},
                                 "preset_note": hostile},
              "comparisons": [{"attempt_id": "1", "label": hostile, "family": "kv", "tools": {"status": "x"},
                               "retrieval": {"status": "y", "lower_95": -0.1, "upper_95": 0.2}, "coding": {}}],
              "budget": {"wall_seconds": 1, "charged_seconds": 0.5, "holdout_reserve_seconds": 0,
                         "remaining_seconds": 0.5, "stop_reason": hostile, "started_utc": "t", "deadline_utc": "t",
                         "candidates_attempted": 1, "holdouts_attempted": 0, "skipped": []},
              "reproduction": [hostile], "campaign_reports": {"json": hostile}, "candidate_reports": {"1": hostile}}
    html = session_html(report)
    assert hostile not in html and "<script>alert" not in html and "&lt;script&gt;alert" in html
    assert "SYNTHETIC HARNESS TEST" in html
    assert html.count("<tr>") == 2
    markdown = session_markdown(report)
    assert "SYNTHETIC HARNESS TEST" in markdown
    table_rows = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert len(table_rows) == 2 and len(table_rows[1].split(" | ")) == len(HEADERS)  # pipes/newlines never split cells
    assert "\\|" in table_rows[1] and "newline" in table_rows[1]
    assert re.search(r"retrieval: y \[-0\.100, 0\.200\]", markdown)


def test_write_session_report_writes_three_files_atomically(tmp_path):
    report = {"session_id": "s", "campaign_identity": "c", "synthetic": True, "columns": list(COLUMNS), "rows": [],
              "recommendation": {"best_quality": {"attempt_id": None, "label": None, "evidence": NONE},
                                 "fastest_eligible": {"attempt_id": None, "label": None, "evidence": NONE},
                                 "largest_verified_context": {"attempt_id": None, "label": None, "evidence": NONE},
                                 "preset_note": "n"},
              "comparisons": [], "budget": {"wall_seconds": 1, "charged_seconds": 0, "holdout_reserve_seconds": 0,
                                            "remaining_seconds": 0, "stop_reason": "complete", "started_utc": "t",
                                            "deadline_utc": "t", "candidates_attempted": 0, "holdouts_attempted": 0,
                                            "skipped": []},
              "reproduction": [], "campaign_reports": {}, "candidate_reports": {}}
    files = write_session_report(report, tmp_path / "reports")
    assert set(files) == {"session_json", "session_markdown", "session_html"}
    assert json.loads((tmp_path / "reports" / "session.json").read_text(encoding="utf-8"))["rows"] == []
    assert not list((tmp_path / "reports").glob("*.tmp*"))
    files_again = write_session_report(report, tmp_path / "reports")  # resume rewrites in place
    assert files_again == files


def test_attempt_row_for_an_attempt_without_any_artifacts(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path)
    (config, manifest), *_ = candidates(session)
    with Store(tmp_path / "store") as store:
        attempt_id = store.create_attempt(manifest, synthetic=False)
        store.transition(attempt_id, "running")
        store.event(attempt_id, "error", {"type": "KeyboardInterrupt", "message": ""})
        store.transition(attempt_id, "cancelled", "interrupted")
        row = attempt_row(store, store.attempt(attempt_id), None)
    assert row["state"].startswith("cancelled (KeyboardInterrupt") and row["label"] == "baseline"
    assert row["eligibility_reasons"] == ["KeyboardInterrupt: "] and row["artifact_dir"] == "-"
    assert row["tps_min"] is None and row["holdout"] == "not attempted" and row["split"] == "development"
    assert row["quantization"] == config.model.quantization and row["manifest_hash"] == manifest.fingerprint()


# ---- LIVE-011: the holdout cell names its scope, and a refusal never renders a bare trailing colon ----


def _holdout_row(tmp_path, block, isolated_gpu_lock):
    session = small_session(tmp_path)
    (_, manifest), *_ = candidates(session)
    with Store(tmp_path / "store") as store:
        attempt_id = store.create_attempt(manifest, synthetic=False)
        store.transition(attempt_id, "running")
        store.transition(attempt_id, "completed")
        return attempt_row(store, store.attempt(attempt_id), {"attempt_id": attempt_id, "holdout": block})


def test_live011_holdout_cell_names_the_categories_the_holdout_could_not_cover(tmp_path, isolated_gpu_lock):
    """A retrieval-only holdout reads as validated ON retrieval, with coding and tools named as uncovered."""
    row = _holdout_row(tmp_path, {"passed": True, "fully_validated": False, "errors":
                                  ["holdout_category_not_covered:coding", "holdout_category_not_covered:tools"],
                                  "covered_categories": ["retrieval"], "validated_categories": ["retrieval"],
                                  "uncovered_categories": ["coding", "tools"]}, isolated_gpu_lock)
    assert row["holdout"] == f"passed on retrieval | {NOT_COVERED}: coding, tools"
    assert row["holdout_validated_categories"] == ["retrieval"]
    assert row["holdout_uncovered_categories"] == ["coding", "tools"]


@pytest.mark.parametrize("errors,expected", [
    (["holdout_category_failed:retrieval"], "not validated: holdout_category_failed:retrieval"),
    ([], f"not validated: {UNEXPLAINED}")])  # a foreign payload still says something, never ": "
def test_live011_a_refused_holdout_always_prints_a_reason(tmp_path, isolated_gpu_lock, errors, expected):
    row = _holdout_row(tmp_path, {"passed": False, "errors": errors, "covered_categories": ["retrieval"],
                                  "validated_categories": [], "uncovered_categories": []}, isolated_gpu_lock)
    assert row["holdout"] == expected
    assert not row["holdout"].rstrip().endswith(":") and row["holdout"].split(": ", 1)[1].strip()


def test_live011_a_partially_covered_preset_is_never_called_simply_validated():
    labels = {"b2": "baseline"}
    preset = {"attempt_id": "b2", "native_tps_minimum": 68.8, "validated_input_tokens": 4085,
              "holdout_validated_categories": ["retrieval"], "holdout_uncovered_categories": ["coding", "tools"]}
    block = recommendation({"best_presets": {"highest_quality": preset, "fastest_within_quality": preset,
                                             "largest_validated_context": preset}, "results": []}, labels,
                           {"b2": 5376})
    for name in ("best_quality", "fastest_eligible", "largest_verified_context"):
        assert block[name]["evidence"] != VALIDATED
        assert block[name]["evidence"].startswith("validated on retrieval holdout only")
        assert f"coding, tools {NOT_COVERED}" in block[name]["evidence"]
        assert block[name]["holdout_uncovered_categories"] == ["coding", "tools"]
    assert block["largest_verified_context"]["claim"] == (
        f"4085 usable input tokens measured on attempt b2 in a 5376-token allocation (holdout validated on "
        f"retrieval; {NOT_COVERED}: coding, tools)")
    assert NOT_COVERED in block["preset_note"]
    # Full coverage keeps the plain validated wording and the plain claim.
    full = {**preset, "holdout_validated_categories": list(CATEGORIES), "holdout_uncovered_categories": []}
    covered = recommendation({"best_presets": {"largest_validated_context": full}, "results": []}, labels)
    assert covered["largest_verified_context"]["evidence"] == VALIDATED
    assert covered["largest_verified_context"]["claim"].endswith("(holdout validated)")


@pytest.mark.parametrize("scope", [
    {},  # no coverage keys at all: the payload says nothing about what the holdout measured
    {"holdout_validated_categories": ["retrieval"]},  # one half of the pair is not a scope
    {"holdout_uncovered_categories": ["coding"]},
    {"holdout_validated_categories": [], "holdout_uncovered_categories": []},  # covered nothing
    {"holdout_validated_categories": "retrieval", "holdout_uncovered_categories": []},  # not a list
])
def test_a_preset_that_does_not_state_its_holdout_scope_is_not_called_validated(scope):
    """REV-OFF-03: the one renderer in this file that failed OPEN now fails closed, like `_holdout_cell`.

    An unknown scope used to take the `VALIDATED` branch -- "validated: separate holdout passed ..." plus a
    context claim ending "(holdout validated)" -- because a payload with no coverage keys has an empty
    `uncovered` list, which read as full coverage. Unknown coverage is the weakest evidence available, not
    the strongest claim available.
    """
    labels = {"b2": "baseline"}
    preset = {"attempt_id": "b2", "native_tps_minimum": 70.0, "validated_input_tokens": 4085, **scope}
    block = recommendation({"best_presets": {"highest_quality": preset, "fastest_within_quality": preset,
                                             "largest_validated_context": preset}, "results": []}, labels,
                           {"b2": 5376})
    for name in ("best_quality", "fastest_eligible", "largest_verified_context"):
        assert block[name]["evidence"] == SCOPE_UNKNOWN and block[name]["attempt_id"] == "b2"
        assert not block[name]["evidence"].startswith("validated")
    claim = block["largest_verified_context"]["claim"]
    assert claim.endswith(SCOPE_UNKNOWN_SUFFIX) and "(holdout validated)" not in claim
    assert claim.startswith("4085 usable input tokens measured on attempt b2")  # the measurement still stands
    # A screening pick is unaffected: its own weaker wording already says where its evidence came from.
    screening = {"attempt_id": "b2", "label": "baseline", "measured_input_tokens": 4085}
    assert _holdout_suffix(screening) == " (screening evidence only)"


def test_unreadable_stored_artifacts_are_named_in_the_row_warnings(tmp_path, isolated_gpu_lock):
    """REV-C1-09: a malformed or oversize candidate.json/raw.json says so in the row instead of a bare '-'."""
    session = small_session(tmp_path)
    outcome, runner, clock = run_tune(tmp_path, session)
    with Store(tmp_path / "out") as store:
        attempt = store.attempts_in_insertion_order()[0]
        target = store.root / "raw" / attempt["id"] / "candidate.json"
        assert target.is_file()
        target.write_text("{not json", encoding="utf-8")
        (store.root / "raw" / attempt["id"] / "raw.json").write_text("[]", encoding="utf-8")
        row = attempt_row(store, attempt, outcome["campaign"]["results"][0])
    assert row["artifact_dir"] == "-" and row["label"] == "baseline"
    assert [":".join(item.split(":")[:2]) for item in row["warnings"]] == [
        "artifact_unreadable:raw.json", "artifact_unreadable:candidate.json"]


def test_a_failed_candidate_never_shows_its_artefact_zeros_as_scores():
    """A candidate that died mid-stage stores score 0.0 for every declared task. That zero is the crash, not the
    model: it once read as `tools 0.000, retrieval 0.000` for a run whose quality stage had simply timed out."""
    from llmbench.containers.session_report import _quality_text
    quality = {"tools": {"attempted": 24, "score": 0.0}, "coding": {"attempted": 0, "score": None}}
    assert _quality_text(quality, "tools") == "0.000 (24 attempted)"          # a completed run really scored zero
    assert _quality_text(quality, "tools", failed=True) == "unmeasured: run failed (24 declared)"
    assert _quality_text(quality, "coding", failed=True) == "n/a (0 attempted)"
