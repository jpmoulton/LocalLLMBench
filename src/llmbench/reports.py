"""Readable reports and evidence-gated preset export; no server configuration changes."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path

from . import analysis
from .config import CampaignPolicy, ExperimentManifest, canonical_json
from .search import assess_candidate
from .store import Store


def markdown_report(campaign: dict) -> str:
    rows = campaign["results"]
    synthetic = any(row.get("synthetic") is not False for row in rows)
    lines = ["# Local inference benchmark report", "",
             "**SYNTHETIC HARNESS TEST — no model performance measured.**" if synthetic else
             "Measured experiment report. Eligibility depends on complete configuration, context and holdout evidence.",
             "", f"Campaign: `{campaign['campaign_id']}`", "",
             "| Attempt | Status | Native minimum tok/s | Coding | Tools | Retrieval | Eligible |",
             "|---|---|---:|---:|---:|---:|---|"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")
    def number(value):
        return "unmeasured" if value is None else f"{value:.3f}"
    for row in rows:
        values = [row.get("attempt_id", "planned"), row.get("status", "planned"),
                  number(row.get("speed", {}).get("minimum_native_tps")),
                  *(number(row.get("quality", {}).get(category, {}).get("score"))
                    for category in ("coding", "tools", "retrieval")),
                  str(row.get("eligibility", {}).get("eligible", False))]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.extend(["", "50 tok/s is a floor, with no speed ceiling. Scores retain failed attempts in the denominator.",
                  "", "## Eligibility details", ""])
    for row in rows:
        reasons = row.get("eligibility", {}).get("reasons", ["planned_only"])
        lines.append(f"- `{cell(row.get('attempt_id', 'planned'))}`: " + ", ".join(cell(x) for x in reasons))
    lines.extend(["", f"Qualifying frontier: {', '.join(campaign.get('frontier', [])) or 'none'}.",
                  "No model was deployed by report generation.", "",
                  "Context capacity, filled prompt tokens and tested usable context are different measurements.",
                  "No unmeasured context length is certified.", ""])
    return "\n".join(lines)


def html_report(campaign: dict) -> str:
    # A self-contained report with filtering; escape all model/runtime text before display.
    report = html.escape(markdown_report(campaign))
    data = html.escape(json.dumps(campaign, indent=2, ensure_ascii=False))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Local inference benchmark</title>
<style>body{{font:16px system-ui;background:#111820;color:#e7edf3;max-width:1100px;margin:40px auto;padding:0 24px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;background:#17222c;padding:24px;border-radius:8px}}
input{{padding:10px;width:95%;background:#fff;color:#111}} summary{{cursor:pointer;margin:20px 0}}</style>
<h1>Local inference benchmark</h1><pre>{report}</pre>
<details><summary>Complete evidence and failure reasons</summary>
<label>Find evidence <input id="filter" placeholder="Filter evidence lines"></label><pre id="data">{data}</pre></details>
<script>const p=document.getElementById('data'),all=p.textContent;
document.getElementById('filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();
p.textContent=q?all.split('\\n').filter(x=>x.toLowerCase().includes(q)).join('\\n'):all;}});</script></html>"""


def write_reports(campaign: dict, output: str | Path) -> dict:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    # Reports are reproducible views; raw artifacts in Store remain immutable.
    files = {"json": directory / "campaign.json", "markdown": directory / "report.md", "html": directory / "report.html"}
    files["json"].write_text(json.dumps(campaign, indent=2, ensure_ascii=False), encoding="utf-8")
    files["markdown"].write_text(markdown_report(campaign), encoding="utf-8")
    files["html"].write_text(html_report(campaign), encoding="utf-8")
    return {key: str(path.resolve()) for key, path in files.items()}


HOLDOUT_REASONS = frozenset({"holdout_not_validated", "holdout_unverified"})


def _text(value) -> bool:
    return isinstance(value, str) and bool(value)


def _refuse(reasons) -> ValueError:
    return ValueError("preset export refused: " + ", ".join(dict.fromkeys(str(item) for item in reasons)))


def _quality_reasons(quality, label: str, policy: CampaignPolicy) -> list[str]:
    thresholds = {"coding": policy.acceptance.minimum_coding_score,
                  "tools": policy.acceptance.minimum_tool_score,
                  "retrieval": policy.acceptance.minimum_retrieval_score}
    reasons = []
    for category in analysis.CATEGORIES:
        row = quality.get(category) if isinstance(quality, dict) else None
        row = row if isinstance(row, dict) else {}
        score, comparison = row.get("score"), row.get("comparison")
        if (row.get("absolute_passed") is not True or row.get("errors") != []
                or type(score) not in {int, float} or not math.isfinite(score)
                or not thresholds[category] <= score <= 1
                or not isinstance(comparison, dict) or comparison.get("status") != "noninferior"):
            reasons.append(f"{label}_{category}_not_validated")
    return reasons


def _verdict(row, policy: CampaignPolicy, *, attempt_id: str, baseline: str, manifest_hash: str,
             policy_hash: str) -> tuple[list[str], list[str]]:
    """Return (core, holdout) rejection reasons; malformed or missing structures reject."""
    if not isinstance(row, dict):
        return ["analysis_missing"], ["holdout_verdict_missing_or_malformed"]
    core, held = [], []
    eligibility = row.get("eligibility")
    listed = eligibility.get("reasons") if isinstance(eligibility, dict) else None
    # A positive verdict has no reasons; a negative verdict must say why. Anything else is malformed.
    if (not isinstance(eligibility, dict) or type(eligibility.get("eligible")) is not bool
            or not isinstance(listed, list) or not all(_text(item) for item in listed)
            or eligibility["eligible"] is bool(listed)):
        core.append("eligibility_verdict_missing_or_malformed")
    else:
        for reason in listed:
            (held if reason in HOLDOUT_REASONS else core).append(f"analysis:{reason}")
    errors = row.get("analysis_errors")
    core.extend([f"analysis_error:{item}" for item in errors] if isinstance(errors, list)
                else ["analysis_errors_missing_or_malformed"])
    if row.get("attempt_id") != attempt_id:
        core.append("attempt_identity_mismatch")
    if row.get("baseline_attempt_id") != baseline:
        core.append("baseline_identity_mismatch")
    if (row.get("synthetic") is not False or row.get("model_evaluated") is not True
            or row.get("status") != "completed"):
        core.append("not_completed_measured_evidence")
    if row.get("manifest_hash") != manifest_hash:
        core.append("manifest_identity_mismatch")
    if row.get("policy_hash") != policy_hash:
        core.append("policy_identity_mismatch")
    if not isinstance(row.get("speed"), dict) or row["speed"].get("qualifies") is not True:
        core.append("speed_gate_failed")
    # Validation always needs verified effective settings and measured context on the exported
    # development evidence; relaxed screening flags in the policy never waive them.
    core.extend(f"{name}_not_true" for name in ("effective_settings_verified", "actual_context_verified")
                if row.get(name) is not True)
    if not isinstance(row.get("context_accounting"), dict) or row["context_accounting"].get("verified") is not True:
        core.append("context_accounting_not_verified")
    core.extend(_quality_reasons(row.get("quality"), "development", policy))
    holdout = row.get("holdout")
    if not isinstance(holdout, dict):
        return core, held + ["holdout_verdict_missing_or_malformed"]
    if row.get("holdout_passed") is not True or holdout.get("passed") is not True:
        held.append("holdout_not_passed")
    for label, value in (("holdout", holdout.get("attempt_id")),
                         ("reference_holdout", holdout.get("reference_attempt_id"))):
        if not _text(value):
            held.append(f"{label}_attempt_missing")
        elif value in {attempt_id, baseline}:
            held.append(f"{label}_not_separate_from_development")
    errors = holdout.get("errors")
    held.extend([f"holdout_error:{item}" for item in errors] if isinstance(errors, list)
                else ["holdout_errors_missing_or_malformed"])
    held.extend(_quality_reasons(holdout.get("quality"), "holdout", policy))
    return core, held


def _stored_origin_reasons(store: Store, attempts) -> list[str]:
    """Sealed evidence and per-sample origin for every attempt the verdict rests on; unknown is not measured."""
    reasons = []
    for key in dict.fromkeys(item for item in attempts if _text(item)):
        record = store.results(key)
        if sum(event["kind"] == "evidence" for event in record["events"]) != 1:
            reasons.append(f"evidence_events_not_exactly_one:{key}")
        if not record["samples"] or not all(isinstance(sample, dict) and sample.get("synthetic") is False
                                            and sample.get("model_evaluated") is True for sample in record["samples"]):
            reasons.append(f"sample_origin_unverified:{key}")
    return reasons


def _current_verdict(manifest, evidence, policy, destination, *, store, campaign, baseline_attempt_id,
                     resamples: int, allow_self_reference: bool, provisional: bool) -> dict:
    """Recompute the named development attempt's verdict from the authoritative Store.

    The caller-supplied evidence only names the attempt and must itself be the
    matching conclusion; it is never trusted. Every Store attempt sharing a
    configuration with the attempt or baseline joins the analysis in insertion
    order, so a stale snapshot cannot hide a later failed holdout, a later
    failed re-run, or a development run that already saw the holdout fixtures.
    Analysis appends its usual analysis events; it cannot load or deploy anything.
    """
    if (not isinstance(store, Store) or not isinstance(campaign, dict) or not _text(baseline_attempt_id)
            or not isinstance(policy, CampaignPolicy)):
        raise _refuse(["authoritative store, campaign, baseline_attempt_id and current policy are required"])
    if Path(destination).exists():
        raise FileExistsError(f"preset export never overwrites: {destination}")
    try:
        policy = CampaignPolicy.model_validate_json(canonical_json(policy.model_dump(mode="json")))
        supplied = ExperimentManifest.model_validate_json(canonical_json(manifest))
        declared = {*campaign.get("all_attempt_ids", []),
                    *(row["attempt_id"] for row in campaign.get("results", []) if row.get("attempt_id"))}
    except (TypeError, AttributeError, KeyError) as exc:
        raise _refuse([f"malformed manifest, policy or campaign: {exc}"]) from exc
    attempt_id = evidence.get("attempt_id") if isinstance(evidence, dict) else None
    if not _text(attempt_id):
        raise _refuse(["analysis_missing"])
    # Noninferiority is definitional against oneself, so a reference preset is an explicit decision.
    self_reference = attempt_id == baseline_attempt_id
    if self_reference and allow_self_reference is not True:
        raise _refuse(["self_reference_requires_explicit_allow_self_reference"])
    policy_hash = hashlib.sha256(canonical_json(policy.model_dump(mode="json")).encode()).hexdigest()
    identity = {"attempt_id": attempt_id, "baseline": baseline_attempt_id,
                "manifest_hash": supplied.fingerprint(), "policy_hash": policy_hash}
    core, held = _verdict(evidence, policy, **identity)
    if core or (held and not provisional):
        raise _refuse(f"supplied:{item}" for item in core + held)
    try:
        stored = analysis.discover_campaign_attempts(store, declared)
    except ValueError as exc:
        raise _refuse([str(exc)]) from exc
    if not {attempt_id, baseline_attempt_id} <= declared:
        raise _refuse(["campaign must declare the attempt and baseline and name only Store attempts"])
    if stored[attempt_id].fingerprint() != supplied.fingerprint():
        raise _refuse(["manifest_identity_mismatch"])
    identifiers = list(stored)
    try:
        current = analysis.analyze_campaign(store, {"all_attempt_ids": identifiers}, policy,
                                            baseline_attempt_id, resamples=resamples)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _refuse([f"current analysis failed: {exc}"]) from exc
    results = current.get("results") if isinstance(current, dict) else None
    rows = [row for row in results if isinstance(row, dict) and row.get("attempt_id") == attempt_id
            ] if isinstance(results, list) else []
    if len(rows) != 1 or current.get("analysis_policy_hash") != policy_hash:
        raise _refuse(["current analysis has no single verdict for this development attempt and policy"])
    core, held = _verdict(rows[0], policy, **identity)
    holdout = rows[0].get("holdout") if isinstance(rows[0].get("holdout"), dict) else {}
    references = {"holdout_attempt_id": holdout.get("attempt_id"),
                  "reference_holdout_attempt_id": holdout.get("reference_attempt_id")}
    if not held and not set(references.values()) <= set(identifiers):
        held.append("holdout_attempt_not_in_store")
    core.extend(_stored_origin_reasons(store, [attempt_id, baseline_attempt_id, *(
        value for value in references.values() if value in stored)]))
    if core or (held and not provisional):
        raise _refuse(f"current:{item}" for item in core + held)
    row = rows[0]
    speed = row["speed"]
    return {"manifest": stored[attempt_id].model_dump(mode="json"), "manifest_hash": supplied.fingerprint(),
            "policy_hash": policy_hash, "acceptance": policy.acceptance.model_dump(mode="json"),
            "evidence_attempt_id": attempt_id, "baseline_attempt_id": baseline_attempt_id,
            "reference_self_validated": self_reference, "comparison_family": row.get("comparison_family"),
            **references, "analyzed_attempt_ids": identifiers,
            "native_tps_minimum": speed.get("minimum_native_tps"),
            "validated_input_tokens": row.get("actual_input_tokens"), "quality": row.get("quality"),
            "holdout_quality": holdout.get("quality"), "blockers": list(dict.fromkeys(held)), "row": row}


def _write_exclusive(destination: str | Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)  # Serialize before the file can exist.
    target = Path(destination)
    handle = target.open("x", encoding="utf-8")  # Exclusive creation never overwrites a preset.
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())  # A crash must not leave a truncated file at a validated path.
    except BaseException:
        target.unlink(missing_ok=True)  # Only reached for the file created above.
        raise


def export_preset(manifest: dict, evidence: dict, policy: CampaignPolicy, destination: str | Path, *,
                  store: Store | None = None, campaign: dict | None = None,
                  baseline_attempt_id: str | None = None, resamples: int = 2000,
                  allow_self_reference: bool = False) -> dict:
    """Write kind=validated-preset only for a fresh positive Store analysis with separate holdout.

    Calls without authoritative Store context fail closed. Relaxed screening
    flags (holdout, effective settings, measured context) never substitute for
    validation. Exporting the baseline itself needs allow_self_reference=True
    and is recorded as reference_self_validated. All checks finish before the
    exclusive destination is created.
    """
    verdict = _current_verdict(manifest, evidence, policy, destination, store=store, campaign=campaign,
                               baseline_attempt_id=baseline_attempt_id, resamples=resamples,
                               allow_self_reference=allow_self_reference, provisional=False)
    row = verdict.pop("row")
    eligibility = row.get("eligibility")
    if (verdict.pop("blockers") or not isinstance(eligibility, dict)
            or eligibility.get("eligible") is not True or eligibility.get("reasons") != []
            or row.get("analysis_errors") != []
            or row.get("holdout_passed") is not True):
        raise _refuse(["current analysis is not a positive validated conclusion"])
    payload = {"schema_version": 2, "kind": "validated-preset", **verdict, "applied": False}
    _write_exclusive(destination, payload)
    return payload


def export_provisional_preset(manifest: dict, evidence: dict, policy: CampaignPolicy,
                              destination: str | Path, *, store: Store | None = None,
                              campaign: dict | None = None, baseline_attempt_id: str | None = None,
                              resamples: int = 2000, allow_self_reference: bool = False) -> dict:
    """Write an explicitly unvalidated screening artifact; never kind=validated-preset.

    Requires the same fresh Store analysis, identities and development
    conclusion under the current screening policy. Only outstanding holdout
    validation is tolerated, with every reason recorded. A candidate holdout
    that was executed and failed its absolute gates, or failed against an
    executed reference holdout, is a rejection, not a preset.
    """
    verdict = _current_verdict(manifest, evidence, policy, destination, store=store, campaign=campaign,
                               baseline_attempt_id=baseline_attempt_id, resamples=resamples,
                               allow_self_reference=allow_self_reference, provisional=True)
    row = verdict.pop("row")
    screening = assess_candidate(row, policy)
    if not screening["eligible"]:
        raise _refuse(f"screening:{item}" for item in screening["reasons"])
    blockers = verdict.pop("blockers")
    executed = verdict["holdout_quality"] if isinstance(verdict["holdout_quality"], dict) else {}
    scored_failing = any(not isinstance(executed.get(category), dict) or executed[category].get(
        "absolute_passed") is not True for category in analysis.CATEGORIES)
    if blockers and _text(verdict["holdout_attempt_id"]) and (
            scored_failing or _text(verdict["reference_holdout_attempt_id"])):
        raise _refuse(["holdout was executed and did not validate", *blockers])
    blockers = blockers or ["validated_export_not_performed"]
    payload = {"schema_version": 2, "kind": "provisional-preset", **verdict, "validated": False,
               "validation_blockers": blockers, "applied": False}
    _write_exclusive(destination, payload)
    return payload
