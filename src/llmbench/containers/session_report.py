"""Session report: every attempt the campaign made, one row each, plus an evidence-labelled recommendation.

Built only from the Store (attempt states, sealed evidence, samples, copied container results) and the
controller's analysed campaign dictionary. Nothing here loads a model or changes a server. All model and
runtime text is escaped before it reaches HTML.
"""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path
from typing import Any

from ..config import ExperimentManifest, canonical_json
from .config import kv_placement_reading

CATEGORIES = ("coding", "tools", "retrieval")
COLUMNS = ("label", "quantization", "ctx", "input_requested", "input_actual", "kv", "kv_placement", "gpu_layers",
           "flash_attn", "speculation", "reasoning", "batch_ubatch", "state", "tps_min", "tps_median", "tps_max",
           "prefill_tps", "ttft_s", "load_s", "vram_mib", "tools", "retrieval", "coding",
           "first_attempt_vs_repaired", "context_verified", "settings_verified", "eligibility_reasons", "holdout",
           "artifact_dir")
HEADERS = ("Label", "Quant", "Ctx allocation", "Input requested", "Input measured", "K/V", "KV placement",
           "GPU layers", "Flash", "Spec", "Reasoning", "Batch/ubatch", "State", "tok/s min", "tok/s median",
           "tok/s max", "Prefill tok/s", "TTFT s", "Load s", "VRAM MiB", "Tools", "Retrieval", "Coding",
           "First/repaired", "Context ok", "Settings ok", "Eligibility reasons", "Holdout", "Artifact dir")
# Whole-token columns: "-" when nothing was measured, never a formatted float and never a silent zero.
TOKEN_COLUMNS = ("input_requested", "input_actual")
# An "Input measured" cell that the analysis did not verify carries this marker, so a fallback count can never be
# read as a verified measurement.
UNVERIFIED_FILL = "(unverified)"
FILL_VERIFIED_NOTE = ("An `Input measured` value is a VERIFIED fill only when the analysis rebuilt it from samples "
                      "that passed tokenizer and template identity, exact post-template counting, no reported "
                      "truncation and the reject-overflow context policy; every other value is the attempt's own "
                      f"sample count and is marked {UNVERIFIED_FILL}. Never quote an {UNVERIFIED_FILL} number, or "
                      "one whose `Context ok` is false, as a tested context.")
# RAM-offload columns. The requested placement comes from the manifest (always present); the effective value is
# read from the attempt's copied ``settings-evidence.json`` and is printed ONLY when that evidence carries one.
VERIFIED_EVIDENCE = frozenset({"verified-api", "verified-log", "verified-behavior"})
NOT_OBSERVED = "effective not observed"
OFFLOAD_NOTE = ("`GPU layers` is the requested `--n-gpu-layers` beside the `offloaded/total` layer count the "
                "startup log reported, and `KV placement` is `gpu` (`--kv-offload`) or `ram` (`--no-kv-offload`) "
                "beside the placement that was observed: `gpu`, `ram`, or `split` with the MiB on each device, "
                "because a partial offload leaves the KV of the blocks still on the CPU in system RAM; "
                f"`({NOT_OBSERVED})` means the attempt produced no readback for that control, never that the "
                "request was met. Layers and KV cache pushed off the "
                "GPU move into the CONTAINER's memory, so a RAM-offloaded configuration is bounded by "
                "`limits.inference_memory_mib`, not by host RAM; read those rows beside `VRAM MiB`, which only "
                "ever falls as work moves to system RAM.")
MAX_RAW_BYTES = 16 * 1024 * 1024
VALIDATED = "validated: separate holdout passed with verified settings and measured context"
# A holdout only ever measures the categories the registry lets it run (``registry.py:89-90`` forbids a
# non-retrieval holdout), so a validated preset is validated ON the categories its holdout covered. The
# rest are named here and are never spoken of as validated: a preset validated on a retrieval holdout is
# not a preset validated on coding (LIVE-011).
NOT_COVERED = "not covered by holdout"
# A preset payload that names neither the categories its holdout covered nor the ones it did not is an UNKNOWN
# scope, and an unknown scope is never rendered as the strongest claim available. `_holdout_cell`
# already fails closed this way ("passed on no category"); these two say the same thing for a recommendation.
SCOPE_UNKNOWN = ("holdout scope not stated: a holdout attempt passed for this candidate, but the analysis named "
                 "no category it covered, so no category is claimed validated here")
SCOPE_UNKNOWN_SUFFIX = " (holdout scope not stated)"
UNEXPLAINED = "holdout_verdict_unexplained"
SCREENING = "screening only: development evidence, holdout not validated; no preset can be exported"
NONE = "no qualifying candidate"
NO_MEASURED_FILL = ("no qualifying candidate: no candidate has a measured input fill, and a context claim is "
                    "never made from the allocation alone")
CONTEXT_CLAIM_NOTE = ("A context claim names USABLE INPUT tokens a candidate actually ran at; the allocation "
                      "(ctx_size) additionally reserves the template reserve and the output capacity above them.")


def _number(value, digits=1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _read_store_json(store, attempt_id: str, name: str, warnings: list[str]) -> dict:
    """A JSON artifact the controller or executor stored for the attempt. Absent reads as {} (the attempt never
    produced it); present but oversize, malformed or not an object also reads as {} and is named in ``warnings``
    (``artifact_unreadable:<name>: <why>``) so the report row says why a column is missing."""
    path = store.root / "raw" / attempt_id / name
    if not path.is_file():
        return {}
    if path.stat().st_size > MAX_RAW_BYTES:
        warnings.append(f"artifact_unreadable:{name}: exceeds {MAX_RAW_BYTES} bytes")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        warnings.append(f"artifact_unreadable:{name}: not JSON ({type(exc).__name__})")
        return {}
    if not isinstance(raw, dict):
        warnings.append(f"artifact_unreadable:{name}: not a JSON object")
        return {}
    return raw


def _events(record: dict, kind: str) -> list[dict]:
    rows = []
    for event in record["events"]:
        if event["kind"] != kind:
            continue
        payload = event["payload"]
        rows.append(json.loads(payload) if isinstance(payload, str) else payload)
    return rows


def _speed_columns(speed: dict) -> dict:
    observations = speed.get("observations") if isinstance(speed, dict) else None
    prefill, ttft = [], []
    for row in observations or []:
        seconds, tokens = row.get("prompt_processing_seconds"), row.get("input_tokens")
        if isinstance(seconds, (int, float)) and seconds > 0 and type(tokens) is int and tokens > 0:
            prefill.append(tokens / seconds)
        first = row.get("first_event_seconds")
        if isinstance(first, (int, float)):
            ttft.append(float(first))
    return {"tps_min": speed.get("minimum_native_tps") if isinstance(speed, dict) else None,
            "tps_median": speed.get("median_native_tps") if isinstance(speed, dict) else None,
            "tps_max": speed.get("maximum_native_tps") if isinstance(speed, dict) else None,
            "prefill_tps": _median(prefill), "ttft_s": _median(ttft)}


def _quality_text(quality: dict, category: str, *, failed: bool = False) -> str:
    """A score, or why there is none. A candidate that died mid-stage still stores 0.0 for every declared task;
    that zero is an artefact of the failure, and printing it would turn a crash into a quality result."""
    row = quality.get(category) if isinstance(quality, dict) else None
    if not isinstance(row, dict) or not row.get("attempted"):
        return "n/a (0 attempted)"
    if failed:
        return f"unmeasured: run failed ({row.get('attempted')} declared)"
    score = row.get("score")
    return f"{_number(score, 3)} ({row.get('attempted')} attempted)"


def _measured_input_tokens(samples: list[dict], accounting: dict) -> tuple[int | None, int | None, bool]:
    """The input fill a candidate actually ran at: ``(minimum, maximum, verified)``; ``(None, None, False)``
    when nothing was measured.

    Only measurements count. The controller stores the REQUESTED count under the name ``actual_input_tokens``
    in its evidence event (``controller.py``), so a context claim must never read that field: the analysis
    rebuilds the real counts from the samples into ``context_accounting`` (tokenizer, template, counting method
    and overflow all checked), and when no analysed row exists the attempt's own samples are read directly and
    the result is reported unverified.
    """
    minimum, maximum = accounting.get("minimum_measured_input_tokens"), accounting.get("maximum_measured_input_tokens")
    if type(minimum) is int:
        return minimum, maximum if type(maximum) is int else minimum, accounting.get("verified") is True
    counts = []
    for sample in samples:
        context = sample.get("context")
        if not isinstance(context, dict):
            continue
        count = context.get("actual_input_tokens")
        if type(count) is int and count > 0 and context.get("observed_input_tokens") == count:
            counts.append(count)
    return (min(counts), max(counts), False) if counts else (None, None, False)


def _evidence_row(rows: list, control: str) -> dict:
    return next((row for row in rows if isinstance(row, dict) and row.get("control") == control), {})


def _offload_columns(backend, evidence_rows: list) -> dict:
    """Requested GPU/RAM placement beside what the readback actually observed.

    ``readback.build_settings_evidence`` records ``n_gpu_layers`` as the ``offloaded/total`` pair the startup log
    printed and ``kv_offload`` as the KV buffer's device; both are absent for an attempt that never reached the
    verify stage, and the KV device is also unobservable when nothing was offloaded to the GPU at all. An
    effective value is rendered only when the evidence carries one, and a mismatch is labelled as such: a
    requested setting is never repeated as though it had been measured.

    The REQUESTED value is the exact engine setting the run recorded in its settings evidence. An attempt that
    produced none falls back to the manifest, which carries the offload FRACTION rather than a layer count, so it
    reads e.g. ``50% of layers``; the layer numbers themselves only ever come from an observation.
    """
    layers, placement = _evidence_row(evidence_rows, "n_gpu_layers"), _evidence_row(evidence_rows, "kv_offload")
    requested = layers.get("requested")
    if requested is None:  # no settings evidence; the manifest still carries the offload FRACTION that was asked
        requested = "all" if backend.gpu_offload == 1.0 else f"{backend.gpu_offload:.0%} of layers"
    effective = layers.get("effective")
    if effective is None:
        gpu_cell = f"{requested} ({NOT_OBSERVED})"
    elif layers.get("status") in VERIFIED_EVIDENCE:
        gpu_cell = f"{requested} ({effective} offloaded to GPU)"
    else:
        gpu_cell = f"{requested} (mismatch: {effective} offloaded to GPU)"
    asked = placement.get("requested")  # the engine flag as the run recorded it; the manifest is the fallback
    requested_kv = "gpu" if (backend.kv_on_gpu if asked is None else asked) else "ram"
    observed = placement.get("effective")
    # The placement is READ, never re-derived: `kv_placement_reading` turns whatever the attempt sealed into
    # the three-state value and its MiB detail, and the cell prints that verbatim. Deciding it here by the
    # truth of `effective` printed every split cache as `gpu` -- half a KV cache in RAM, reported as none of
    # it -- and an unrecognised value reads as "not observed" rather than as a placement.
    state, detail = kv_placement_reading(observed, placement.get("detail"))
    if state is None:
        kv_cell = f"{requested_kv} ({NOT_OBSERVED})"
    else:
        seen = f"{state}: {detail}" if detail else state
        kv_cell = (f"{requested_kv} (observed {seen})" if placement.get("status") in VERIFIED_EVIDENCE
                   else f"{requested_kv} (mismatch: observed {seen})")
    return {"gpu_layers": gpu_cell, "kv_placement": kv_cell, "gpu_layers_requested": requested,
            "gpu_layers_effective": effective, "gpu_layers_status": layers.get("status") or "no-evidence",
            "kv_placement_requested": requested_kv, "kv_placement_effective": observed,
            "kv_placement_state": state, "kv_placement_detail": detail,
            "kv_placement_status": placement.get("status") or "no-evidence"}


def _category_list(block: dict, key: str) -> list[str]:
    value = block.get(key)
    return [str(item) for item in value] if isinstance(value, list) else []


def _partial_validated(covered: list[str], uncovered: list[str]) -> str:
    """Evidence wording for a preset whose holdout covered only part of the quality space."""
    return (f"validated on {', '.join(covered) or 'no category'} holdout only: separate holdout passed with "
            f"verified settings and measured context, but {', '.join(uncovered)} {NOT_COVERED} and NOT validated")


def _holdout_cell(block: dict) -> str:
    """The holdout verdict for one development row; a refusal always names at least one cause.

    ``analysis`` guarantees a non-empty ``errors`` list whenever ``passed`` is not true, so ``UNEXPLAINED``
    can only appear for a foreign or truncated analysis payload -- and even then the cell says something
    instead of rendering a bare ``not validated:`` with nothing after the colon (LIVE-011). The categories
    the holdout could not measure are printed beside the verdict, never folded into it.
    """
    covered, uncovered = _category_list(block, "validated_categories"), _category_list(block, "uncovered_categories")
    if block.get("passed") is True:
        text = "passed on " + (", ".join(covered) if covered else "no category")
    else:
        text = "not validated: " + "; ".join(_category_list(block, "errors")[:4] or [UNEXPLAINED])
    return text + (f" | {NOT_COVERED}: " + ", ".join(uncovered) if uncovered else "")


def _coding_text(samples: list[dict]) -> str:
    coding = [row for row in samples if row.get("category") == "coding"]
    if not coding:
        return "n/a"
    first = sum(row.get("first_attempt_success") is True for row in coding)
    passed = sum(row.get("status") == "completed" and row.get("passed") is True for row in coding)
    return f"{first} first-attempt / {max(0, passed - first)} repaired / {len(coding)} attempted"


def attempt_row(store, attempt: dict, analysis_row: dict | None) -> dict:
    """One report row for one Store attempt, including failed, rejected, timeout and holdout attempts."""
    record = store.results(attempt["id"])
    manifest = ExperimentManifest.model_validate_json(canonical_json(record["manifest"]))
    evidence = _events(record, "evidence")
    evidence = evidence[0] if len(evidence) == 1 else {}
    errors = _events(record, "error")
    unreadable: list[str] = []
    raw = _read_store_json(store, attempt["id"], "raw.json", unreadable)  # written by the controller on success
    candidate = _read_store_json(store, attempt["id"], "candidate.json", unreadable)  # executor, every attempt
    # Copied by the executor for every attempt that reached the verify stage; absent reads as no evidence at all.
    settings = _read_store_json(store, attempt["id"], "settings-evidence.json", unreadable).get("settings")
    settings = settings if isinstance(settings, list) else []
    result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    if not result:  # the executor copies result.json before a failed candidate is raised
        result = _read_store_json(store, attempt["id"], "result.json", unreadable)
    backend, split = manifest.backend, {item.split for item in manifest.tasks}
    holdout_attempt = "holdout" in split
    label = raw.get("label") or candidate.get("label") or manifest.annotations.get("label", "unlabelled")
    run_dir = raw.get("run_dir") or candidate.get("run_dir") or "-"
    candidate_report = raw.get("candidate_report") or (result.get("reports") or {}).get("candidate")
    state = attempt["state"]
    if result and (result.get("state") != "completed" or state != "completed"):
        state = f"{state} (container: {result.get('state')} at {result.get('failure_stage') or 'unknown'})"
    elif errors:
        state = f"{state} ({errors[-1].get('type')}: {str(errors[-1].get('message'))[:160]})"
    reasons = list(((analysis_row or evidence or {}).get("eligibility") or {}).get("reasons") or [])
    reasons += [f"{item.get('type')}: {item.get('message')}" for item in errors]
    reasons += [str(item) for item in result.get("failure_reasons") or []]
    reasons = list(dict.fromkeys(reasons or [state]))
    holdout = "holdout attempt (validation evidence for its configuration)" if holdout_attempt else "not attempted"
    holdout_block = analysis_row.get("holdout") if analysis_row else None
    holdout_block = holdout_block if isinstance(holdout_block, dict) else {}
    if holdout_block:
        holdout = _holdout_cell(holdout_block)
    warnings = list(raw.get("warnings") or []) + list(result.get("warnings") or []) + unreadable
    context_ok = (analysis_row or evidence or {}).get("actual_context_verified") is True
    settings_ok = (analysis_row or evidence or {}).get("effective_settings_verified") is True
    accounting = (analysis_row or {}).get("context_accounting") or {}
    measured_min, measured_max, fill_verified = _measured_input_tokens(record["samples"], accounting)
    return {"attempt_id": attempt["id"], "parent_id": attempt.get("parent_id"), "split": "holdout" if holdout_attempt
            else "development", "label": label, "quantization": manifest.model.quantization,
            # Allocation and measured fill stand side by side: the allocation reserves output capacity above the
            # input, so it is never itself evidence that the model ran at that many input tokens.
            "ctx": backend.context_length, "input_requested": manifest.requested_input_tokens,
            "input_actual": measured_min, "input_measured_maximum": measured_max,
            "input_fill_verified": fill_verified,
            "kv": f"{backend.k_cache}/{backend.v_cache}", **_offload_columns(backend, settings),
            "flash_attn": backend.flash_attention, "speculation": "draft-mtp" if backend.mtp else "none",
            "reasoning": backend.reasoning or "engine-default",
            "batch_ubatch": f"{backend.batch_size}/{backend.ubatch_size if backend.ubatch_size else '-'}",
            "state": state, **_speed_columns(evidence.get("speed", {})),
            "load_s": result.get("load_seconds"), "vram_mib": result.get("vram_used_mib_after_load"),
            "tools": _quality_text(evidence.get("quality", {}), "tools", failed=not str(state).startswith("completed")),
            "retrieval": _quality_text(evidence.get("quality", {}), "retrieval",
                                       failed=not str(state).startswith("completed")),
            "coding": _quality_text(evidence.get("quality", {}), "coding", failed=not str(state).startswith("completed")),
            "first_attempt_vs_repaired": _coding_text(record["samples"]), "context_verified": context_ok,
            "settings_verified": settings_ok, "eligibility_reasons": list(reasons), "holdout": holdout,
            "holdout_validated_categories": _category_list(holdout_block, "validated_categories"),
            "holdout_uncovered_categories": _category_list(holdout_block, "uncovered_categories"),
            "artifact_dir": run_dir, "candidate_report": candidate_report,
            "warnings": warnings, "manifest_hash": manifest.fingerprint(), "synthetic": attempt.get("synthetic"),
            "comparison_family": (analysis_row or {}).get("comparison_family"),
            "template_check": raw.get("template_check")}


def _screening_candidates(results: list[dict]) -> list[dict]:
    keep = []
    for row in results:
        speed, quality = row.get("speed", {}), row.get("quality", {})
        measured = [quality[name] for name in CATEGORIES if isinstance(quality.get(name), dict)
                    and quality[name].get("attempted")]
        if (row.get("status") == "completed" and row.get("synthetic") is False and speed.get("qualifies") is True
                and row.get("effective_settings_verified") is True and row.get("actual_context_verified") is True
                and measured and all(item.get("absolute_passed") is True for item in measured)):
            keep.append(row)
    return keep


def _is_validated_pick(item: dict) -> bool:
    """A preset-backed pick. Only the validated branch of ``recommendation`` carries a validated fill field,
    so the claim wording never depends on matching an evidence sentence that now names the holdout scope."""
    return "validated_input_tokens" in item


def _holdout_scope(block: dict) -> tuple[list[str], list[str], bool]:
    """``(validated, uncovered, stated)`` for a preset payload; ``stated`` decides how strong a claim is allowed.

    The scope is STATED only when the payload carries both category lists AND names at least one validated
    category. ``analysis.preset()`` always sets both, so this is about the payload no in-repo path produces --
    a stale, truncated or foreign one -- and about which way that unknown should fall. It falls to "nothing is
    claimed": the empty ``uncovered`` of a payload that simply lacks the keys used to read as full coverage and
    print the flat ``VALIDATED`` wording, which is the strongest sentence this module can emit.
    """
    covered, uncovered = (_category_list(block, "holdout_validated_categories"),
                          _category_list(block, "holdout_uncovered_categories"))
    stated = (isinstance(block.get("holdout_validated_categories"), list)
              and isinstance(block.get("holdout_uncovered_categories"), list) and bool(covered))
    if "holdout_scope_stated" in block:
        # A block this module already built carries the verdict it reached, because building it NORMALIZES
        # the two lists: judging those normalized lists again would quietly promote an unknown scope.
        stated = block["holdout_scope_stated"] is True
    return covered, uncovered, stated


def _holdout_suffix(item: dict) -> str:
    if not _is_validated_pick(item):
        return " (screening evidence only)"
    covered, uncovered, stated = _holdout_scope(item)
    if not stated:
        return SCOPE_UNKNOWN_SUFFIX
    if not uncovered:
        return " (holdout validated)"
    return f" (holdout validated on {', '.join(covered)}; {NOT_COVERED}: {', '.join(uncovered)})"


def _context_claim(block: dict, allocations: dict[str, int]) -> None:
    """Express the largest-context pick in USABLE INPUT tokens, or withdraw it.

    The claim is the input fill a candidate was measured at (``validated_input_tokens`` on a validated preset,
    ``measured_input_tokens`` on a screening pick -- both are ``analysis``' minimum measured count). Without
    such a measurement the pick is withdrawn: an allocation on its own is not evidence that the model ran at
    that many input tokens.
    """
    item = block["largest_verified_context"]
    fill = item.get("validated_input_tokens") if _is_validated_pick(item) else item.get("measured_input_tokens")
    item["claim_note"] = CONTEXT_CLAIM_NOTE
    if item.get("attempt_id") is None:
        return
    if type(fill) is not int or fill <= 0:
        block["largest_verified_context"] = {"attempt_id": None, "label": None, "evidence": NO_MEASURED_FILL,
                                             "claim_note": CONTEXT_CLAIM_NOTE}
        return
    allocation = allocations.get(item["attempt_id"])
    item["usable_input_tokens"] = fill
    item["context_allocation"] = allocation
    item["claim"] = (f"{fill} usable input tokens measured on attempt {item['attempt_id']}"
                     + (f" in a {allocation}-token allocation" if type(allocation) is int else "")
                     + _holdout_suffix(item))


def recommendation(campaign: dict, labels: dict[str, str], allocations: dict[str, int] | None = None) -> dict:
    """Three recommendations, each labelled by evidence status; validated presets only from the analysis."""
    best, results = campaign.get("best_presets") or {}, campaign.get("results") or []
    screening = _screening_candidates(results)

    def quality_key(row):
        return tuple((row["quality"].get(name) or {}).get("score") or 0. for name in CATEGORIES)

    picks = {"best_quality": ("highest_quality", lambda row: (quality_key(row), row["speed"]["minimum_native_tps"])),
             "fastest_eligible": ("fastest_within_quality", lambda row: row["speed"]["minimum_native_tps"]),
             "largest_verified_context": ("largest_validated_context",
                                          lambda row: (row.get("actual_input_tokens") or 0,
                                                       row["speed"]["minimum_native_tps"]))}
    block = {}
    for name, (preset_key, order) in picks.items():
        if preset_key in best:
            attempt = best[preset_key]["attempt_id"]
            covered, uncovered, stated = _holdout_scope(best[preset_key])
            evidence = SCOPE_UNKNOWN if not stated else (_partial_validated(covered, uncovered) if uncovered
                                                         else VALIDATED)
            block[name] = {"attempt_id": attempt, "label": labels.get(attempt), "evidence": evidence,
                           "holdout_validated_categories": covered, "holdout_uncovered_categories": uncovered,
                           "holdout_scope_stated": stated,
                           "native_tps_minimum": best[preset_key].get("native_tps_minimum"),
                           "validated_input_tokens": best[preset_key].get("validated_input_tokens")}
            # Validation needs a demonstrated non-inferiority, which small item counts cannot give anyone but the
            # baseline. A screening candidate that beats the validated pick on this slot's own measure is named
            # beside it, with what blocks it, so a validated baseline can never hide a faster or larger run.
            chosen = next((row for row in results if row.get("attempt_id") == attempt), None)
            better = [row for row in screening if chosen is not None and row["attempt_id"] != attempt
                      and order(row) > order(chosen)]
            if better:
                row = max(better, key=order)
                block[name]["better_unvalidated"] = {
                    "attempt_id": row["attempt_id"], "label": labels.get(row["attempt_id"]), "evidence": SCREENING,
                    "native_tps_minimum": row["speed"].get("minimum_native_tps"),
                    "measured_input_tokens": row.get("actual_input_tokens"),
                    "blockers": list(row.get("eligibility", {}).get("reasons", []))}
        elif screening:
            row = max(screening, key=order)
            block[name] = {"attempt_id": row["attempt_id"], "label": labels.get(row["attempt_id"]),
                           "evidence": SCREENING, "native_tps_minimum": row["speed"].get("minimum_native_tps"),
                           "measured_input_tokens": row.get("actual_input_tokens"),
                           "blockers": list(row.get("eligibility", {}).get("reasons", []))}
        else:
            block[name] = {"attempt_id": None, "label": None, "evidence": NONE}
    _context_claim(block, allocations or {})
    block["preset_note"] = ("No validated preset exists until a separate holdout attempt passes for a candidate; "
                            "screening picks are not deployable. A preset is validated only ON the categories "
                            f"its holdout measured; a category marked `{NOT_COVERED}` was never re-tested on "
                            "unseen fixtures and nothing here claims it.")
    return block


def _comparisons(campaign: dict, labels: dict[str, str]) -> list[dict]:
    rows = []
    for row in campaign.get("results") or []:
        entry = {"attempt_id": row.get("attempt_id"), "label": labels.get(row.get("attempt_id")),
                 "family": row.get("comparison_family"), "baseline_attempt_id": row.get("baseline_attempt_id")}
        for name in CATEGORIES:
            comparison = (row.get("quality", {}).get(name) or {}).get("comparison") or {}
            entry[name] = {key: comparison.get(key) for key in ("status", "mean_delta", "lower_95", "upper_95",
                                                                "method", "reason") if key in comparison}
        rows.append(entry)
    return rows


def _context_range(session, ledger: dict) -> dict | None:
    """The declared range plus the input targets the schedule the ledger pinned actually carries.

    ``search.context_points_dropped`` names only the targets the derivation thinned away. The schedule can drop a
    declared tier too -- ``deterministic_schedule`` truncates at ``max_candidates`` and the context sweep is last
    -- so the targets that really reach a candidate, and the declared ones that do not, are reported beside it
   . The ledger's proposals are the pinned schedule, so this describes the run, not the intent.
    """
    span = session.search.context_range()
    if span is None:
        return None
    from .session import output_reserve_tokens, usable_input_tokens
    reserve, output = session.base.template_reserve_tokens, output_reserve_tokens(session.base)
    tiers, baseline_tier = session.search.ctx_tiers, session.search.ctx_tiers[0]
    scheduled = {(row.get("changes") or {}).get("ctx_tier", baseline_tier)  # no ctx_tier change: the baseline tier
                 for row in ledger.get("proposals") or []}
    targets = {tier: usable_input_tokens(tier, template_reserve_tokens=reserve, output_tokens=output,
                                         ceiling=session.search.context_ceiling) for tier in tiers}
    span["scheduled_input_targets"] = sorted({value for tier, value in targets.items() if tier in scheduled})
    span["unscheduled_input_targets"] = sorted({value for tier, value in targets.items() if tier not in scheduled})
    return span


def build_session_report(store, campaign: dict, session, ledger: dict, *, output_root: str | Path,
                         stop_reason: str, remaining_seconds: float, campaign_reports: dict | None = None) -> dict:
    analysed = {row.get("attempt_id"): row for row in campaign.get("results") or [] if row.get("attempt_id")}
    rows = [attempt_row(store, attempt, analysed.get(attempt["id"]))
            for attempt in store.attempts_in_insertion_order()]
    labels = {row["attempt_id"]: row["label"] for row in rows}
    charged = store.campaign_state(ledger["campaign_identity"])["elapsed_seconds"]
    wall = ledger["wall_seconds"]
    root = Path(output_root)
    reproduction = [f"llmbench resume --output {root}",
                    f"llmbench tune --config {root / 'session-config.json'} --output <new directory>"]
    reproduction += [f"llmbench candidate --config {Path(row['artifact_dir']) / 'config.json'} "
                     f"--output <new directory>" for row in rows if row["artifact_dir"] != "-"]
    allocations = {row["attempt_id"]: row["ctx"] for row in rows}
    return {"schema_version": 1, "session_id": session.session_id, "campaign_identity": ledger["campaign_identity"],
            "synthetic": any(row.get("synthetic") not in {0, False} for row in rows),
            "context_range": _context_range(session, ledger),
            "columns": list(COLUMNS), "rows": rows,
            "recommendation": recommendation(campaign, labels, allocations),
            "comparisons": _comparisons(campaign, labels), "frontier": campaign.get("frontier", []),
            "winner": campaign.get("winner"), "analysis_note": campaign.get("analysis_note"),
            "budget": {"wall_seconds": wall, "charged_seconds": charged,
                       "holdout_reserve_seconds": session.budgets.holdout_reserve_seconds,
                       "remaining_seconds": remaining_seconds, "stop_reason": stop_reason,
                       "started_utc": ledger["started_utc"], "deadline_utc": ledger["deadline_utc"],
                       "candidates_attempted": sum(row["split"] == "development" for row in rows),
                       "holdouts_attempted": sum(row["split"] == "holdout" for row in rows),
                       "max_candidates": session.budgets.max_candidates,
                       "skipped": campaign.get("skipped", [])},
            "reproduction": reproduction, "campaign_reports": dict(campaign_reports or {}),
            "candidate_reports": {row["attempt_id"]: str(Path(row["artifact_dir"]) / row["candidate_report"])
                                  for row in rows if row["artifact_dir"] != "-" and row.get("candidate_report")}}


def _cell(value) -> str:
    text = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _row_values(row: dict) -> list[str]:
    values = []
    for column in COLUMNS:
        value = row[column]
        if column in {"tps_min", "tps_median", "tps_max", "prefill_tps", "ttft_s", "load_s", "vram_mib"}:
            value = _number(value, 2 if column == "ttft_s" else 1)
        elif column in TOKEN_COLUMNS and value is None:
            value = "-"  # nothing measured; never a zero that reads like a measurement
        elif column == "input_actual" and row.get("input_fill_verified") is not True:
            value = f"{value} {UNVERIFIED_FILL}"  # a fallback count must never render like a verified measurement
        values.append(_cell(value))
    return values


def _target_line(label: str, span: dict, key: str) -> str:
    if key not in span:
        return f"- {label}: not recorded"
    values = span.get(key) or []
    return f"- {label}: " + (", ".join(str(item) for item in values) if values else "none")


def _context_range_lines(report: dict) -> list[str]:
    """The declared search range, in usable input tokens, or nothing when the session declared none.

    Two different things can remove a declared input target: the DERIVATION thins the layout to fit the candidate
    cap (``search.context_points_dropped``), and the SCHEDULE itself is truncated at ``max_candidates``. Both are
    printed, so "dropped: none" can never be stated while the schedule omits a declared target.
    """
    span = report.get("context_range")
    if not span:
        return []
    return ["", "## Context range (usable input tokens)", "",
            f"- Searched {span['context_floor']}..{span['context_ceiling']} usable INPUT tokens; the template "
            f"reserve and the output capacity are reserved on top, so each tier's allocation is larger.",
            f"- Model training context: {span['n_ctx_train']} tokens.",
            _target_line("Input targets the pinned schedule carries", span, "scheduled_input_targets"),
            _target_line("Input targets dropped for the candidate cap when the tiers were derived", span,
                         "dropped_input_targets"),
            _target_line("Declared input targets the candidate cap cut out of the schedule (never measured)", span,
                         "unscheduled_input_targets")]


def session_markdown(report: dict) -> str:
    banner = ("**SYNTHETIC HARNESS TEST - no model performance measured.**" if report["synthetic"] else
              "Measured llama.cpp container session. Every attempt is listed; failures stay in the table.")
    budget, rec = report["budget"], report["recommendation"]
    lines = [f"# Tuning session `{report['session_id']}`", "", banner, "",
             f"Campaign `{report['campaign_identity']}`; stop reason **{budget['stop_reason']}**; "
             f"{budget['candidates_attempted']} development and {budget['holdouts_attempted']} holdout attempts.", "",
             "## Recommendation", ""]
    for name in ("best_quality", "fastest_eligible", "largest_verified_context"):
        item = rec[name]
        target = f"`{item['label']}` (attempt `{item['attempt_id']}`)" if item.get("attempt_id") else "none"
        lines.append(f"- {name.replace('_', ' ')}: {target} - {item['evidence']}")
        if item.get("claim"):
            lines.append(f"  - largest measured usable context: {_cell(item['claim'])}")
        if item.get("claim_note"):
            lines.append(f"  - {_cell(item['claim_note'])}")
        if item.get("blockers"):
            lines.append("  - blockers: " + ", ".join(_cell(x) for x in item["blockers"]))
        other = item.get("better_unvalidated")
        if other:
            lines.append(f"  - measured better but NOT validated: `{other['label']}` (attempt `{other['attempt_id']}`, "
                         f"{other['native_tps_minimum']} tok/s minimum, {other['measured_input_tokens']} input "
                         f"tokens) - {other['evidence']}; blocked by: "
                         + ", ".join(_cell(x) for x in other["blockers"]))
    lines += ["", rec["preset_note"], *_context_range_lines(report), "", "## Every attempt", "",
              "| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    lines += ["| " + " | ".join(_row_values(row)) + " |" for row in report["rows"]]
    lines += ["", "## Paired comparisons (development attempts against the baseline)", ""]
    for entry in report["comparisons"]:
        parts = [f"{name}: {entry[name].get('status')}"
                 + (f" [{entry[name]['lower_95']:.3f}, {entry[name]['upper_95']:.3f}]"
                    if isinstance(entry[name].get("lower_95"), (int, float)) else "") for name in CATEGORIES]
        lines.append(f"- `{entry['label']}` ({entry['family']}): " + "; ".join(parts))
    lines += ["", "## Budget", "",
              f"- Wall budget {budget['wall_seconds']} s; charged {budget['charged_seconds']:.0f} s; holdout reserve "
              f"{budget['holdout_reserve_seconds']} s; remaining {budget['remaining_seconds']:.0f} s",
              f"- Started {budget['started_utc']}; deadline {budget['deadline_utc']}; stop reason {budget['stop_reason']}",
              f"- Skipped: {len(budget['skipped'])}", "", "## Reproduction", ""]
    lines += [f"- `{_cell(command)}`" for command in report["reproduction"]]
    lines += ["", "## Reports", ""]
    lines += [f"- campaign {key}: `{_cell(path)}`" for key, path in report["campaign_reports"].items()]
    lines += [f"- candidate `{_cell(attempt)}`: `{_cell(path)}`" for attempt, path in report["candidate_reports"].items()]
    lines += ["", "Context capacity, filled prompt tokens and tested usable context are different measurements.",
              CONTEXT_CLAIM_NOTE, FILL_VERIFIED_NOTE, OFFLOAD_NOTE,
              "Scores keep every failed, timed-out or rejected task in the denominator.", ""]
    return "\n".join(lines)


def session_html(report: dict) -> str:
    def esc(value) -> str:
        return html.escape(_cell(value), quote=True)
    head = "".join(f"<th>{esc(item)}</th>" for item in HEADERS)
    body = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in _row_values(row)) + "</tr>"
                   for row in report["rows"])
    rec = "".join(f"<li><b>{esc(name)}</b>: {esc(item.get('label') or 'none')} - {esc(item['evidence'])}"
                  + (f"<br>{esc(item['claim'])}" if item.get("claim") else "") + "</li>"
                  for name, item in report["recommendation"].items() if isinstance(item, dict))
    span = report.get("context_range")
    range_html = "" if not span else (
        f"<h2>Context range</h2><p>{esc(span['context_floor'])} to {esc(span['context_ceiling'])} usable INPUT "
        f"tokens (model training context {esc(span['n_ctx_train'])}).</p><ul>"
        + "".join(f"<li>{esc(_target_line(label, span, key)[2:])}</li>" for label, key in (
            ("Input targets the pinned schedule carries", "scheduled_input_targets"),
            ("Input targets dropped for the candidate cap when the tiers were derived", "dropped_input_targets"),
            ("Declared input targets the candidate cap cut out of the schedule (never measured)",
             "unscheduled_input_targets"))) + "</ul>")
    data = html.escape(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tuning session {esc(report['session_id'])}</title>
<style>body{{font:15px system-ui;background:#111820;color:#e7edf3;margin:32px auto;max-width:1600px;padding:0 24px}}
table{{border-collapse:collapse;font-size:13px}} th,td{{border:1px solid #2a3642;padding:4px 8px;text-align:left}}
th{{background:#17222c;position:sticky;top:0}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#17222c;padding:16px}}
input{{padding:8px;width:95%;background:#fff;color:#111}} .wrap{{overflow-x:auto}}</style>
<h1>Tuning session {esc(report['session_id'])}</h1>
<p>{esc('SYNTHETIC HARNESS TEST - no model performance measured.' if report['synthetic'] else
         'Measured llama.cpp container session; one row per attempt, failures included.')}</p>
<h2>Recommendation</h2><ul>{rec}</ul><p>{esc(report['recommendation']['preset_note'])}</p>
<p>{esc(CONTEXT_CLAIM_NOTE)}</p><p>{esc(FILL_VERIFIED_NOTE)}</p><p>{esc(OFFLOAD_NOTE)}</p>{range_html}
<h2>Every attempt</h2><div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<h2>Budget</h2><p>{esc(json.dumps(report['budget'], default=str))}</p>
<h2>Reproduction</h2><ul>{''.join(f'<li><code>{esc(item)}</code></li>' for item in report['reproduction'])}</ul>
<details><summary>Complete report data</summary><label>Find <input id="filter" placeholder="Filter lines"></label>
<pre id="data">{data}</pre></details>
<script>const p=document.getElementById('data'),all=p.textContent;
document.getElementById('filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();
p.textContent=q?all.split('\\n').filter(x=>x.toLowerCase().includes(q)).join('\\n'):all;}});</script></html>"""


def write_session_report(report: dict, reports_dir: str | Path) -> dict[str, str]:
    from .session import write_atomic_json, write_atomic_text
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic_json(directory / "session.json", json.loads(json.dumps(report, default=str)))
    for name, text in (("session.md", session_markdown(report)), ("session.html", session_html(report))):
        write_atomic_text(directory / name, text)  # tmp + fsync + replace, like session.json
    return {"session_json": str((directory / "session.json").resolve()),
            "session_markdown": str((directory / "session.md").resolve()),
            "session_html": str((directory / "session.html").resolve())}


__all__ = ["attempt_row", "recommendation", "build_session_report", "session_markdown", "session_html",
           "write_session_report", "COLUMNS", "HEADERS", "CONTEXT_CLAIM_NOTE", "FILL_VERIFIED_NOTE",
           "NO_MEASURED_FILL", "UNVERIFIED_FILL", "OFFLOAD_NOTE", "NOT_OBSERVED", "NOT_COVERED", "UNEXPLAINED",
           "SCOPE_UNKNOWN", "SCOPE_UNKNOWN_SUFFIX", "Any"]
