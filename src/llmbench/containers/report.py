"""Candidate evidence row and human-readable reports. Pure file output; nothing is loaded or deployed.

A `metal-native` candidate gets the same report with its own identity and memory lines: the pinned llama-server
executable (sha256, libraries digest, build) instead of an image, and the unified-memory evidence of its result
instead of a VRAM figure. An Apple Silicon Mac has no dedicated VRAM, so the report says "not applicable" there
rather than "unmeasured", which would claim a measurement was missed. An NVIDIA candidate's report is unchanged.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from ..config import AcceptancePolicy, CampaignPolicy, RunMode, canonical_json
from ..measurement import SpeedObservation, summarize_speed
from ..reports import markdown_report, html_report
from ..search import assess_candidate
from .config import ContainerRunConfig, ContainerRunResult
from .readback import unverified_required
from .session_report import NATIVE_RUNTIME, SANDBOX_BLOCKED, pressure_text, unified_memory_evidence

CATEGORIES = ("coding", "tools", "retrieval")
CANDIDATE_UNIFIED_NOTE = (
    "Unified memory: on Apple Silicon the CPU, the GPU and every other application share one physical pool, so "
    "this candidate has no VRAM figure. The server footprint is the llama-server process's phys_footprint (what "
    "Activity Monitor calls Memory), which includes its Metal allocations; the Metal buffers are the model, KV, "
    "compute and recurrent buffers the server logged on the Metal device, against Metal's recommended working set "
    "(the budget). Host available memory, swap and pressure are host-wide and include other applications; swap "
    "growth runs from the admission sample taken before the load (or the watchdog's first sample, when that one "
    "had no swap reading) to the highest sample during evaluation.")


def default_policy(config: ContainerRunConfig) -> CampaignPolicy:
    return CampaignPolicy(name="container-candidate", mode=RunMode.LIVE,
                          acceptance=AcceptancePolicy(speed_repetitions=config.speed.repetitions))


def summarize_evaluation(config: ContainerRunConfig, evaluation: dict, policy: CampaignPolicy | None) -> dict:
    policy = policy or default_policy(config)
    samples = [row for row in evaluation.get("samples") or [] if isinstance(row, dict)]
    observations = [_observation(row) for row in evaluation.get("speed_observations") or []]
    quality = {}
    for category in CATEGORIES:  # Same denominator rule as controller.execute_attempt: failures score zero.
        rows = [sample for sample in samples if sample.get("category") == category]
        scores = [float(row.get("score", 0)) if row.get("status") == "completed" else 0. for row in rows]
        quality[category] = {"attempted": len(rows), "score": sum(scores) / len(scores) if scores else None,
                             "comparison": {"status": "unmeasured"}}
    return {"speed": summarize_speed(observations, policy.acceptance), "quality": quality,
            "samples_total": len(samples)}


def _observation(row) -> SpeedObservation:
    if isinstance(row, SpeedObservation):
        return row
    if not isinstance(row, dict):
        raise TypeError("speed observations must be SpeedObservation records")
    names = SpeedObservation.__dataclass_fields__
    values = {key: value for key, value in row.items() if key in names}
    values["content_event_times"] = tuple(values.get("content_event_times") or ())
    return SpeedObservation(**values)


def evidence_row(config: ContainerRunConfig, result: ContainerRunResult, policy: CampaignPolicy) -> dict:
    dumped = result.model_dump(mode="json")
    row = {"attempt_id": result.attempt_id, "manifest_hash": result.config_fingerprint,
           "policy_hash": hashlib.sha256(canonical_json(policy.model_dump(mode="json")).encode()).hexdigest(),
           "status": result.state, "synthetic": result.synthetic, "model_evaluated": not result.synthetic,
           "speed": dumped["speed"], "quality": dumped["quality"],
           "actual_input_tokens": config.requested_input_tokens,
           "effective_settings_verified": result.effective_settings_verified,
           "actual_context_verified": result.actual_context_verified, "holdout_passed": False,
           "elapsed_seconds": result.elapsed_seconds, "engine": "llama.cpp", "label": config.label,
           "failure_stage": result.failure_stage, "failure_reasons": list(result.failure_reasons),
           "warnings": list(result.warnings), "cleanup_verified": result.cleanup.verified,
           "abort_campaign": result.abort_campaign,
           "unverified_required_controls": unverified_required(result.settings)}
    row["eligibility"] = assess_candidate(row, policy)
    return row


def _number(value) -> str:
    return "unmeasured" if value is None else f"{value:.2f}"


def _cell(value) -> str:
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(header: tuple[str, ...], rows, *, numeric: tuple[int, ...] = ()) -> list[str]:
    rule = "|" + "|".join("---:" if index in numeric else "---" for index in range(len(header))) + "|"
    return ["| " + " | ".join(header) + " |", rule,
            *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows), ""]


def _mib_text(value) -> str:
    return "unmeasured" if value is None else f"{value:.0f} MiB"


def _engine_line(engine) -> str:
    return (f"- Context {engine.ctx_size}, K/V {engine.cache_type_k}/{engine.cache_type_v}, flash attention "
            f"{engine.flash_attn}, GPU layers {engine.n_gpu_layers}, speculation {engine.spec_type}, "
            f"reasoning {engine.reasoning}")


def _native_lines(config: ContainerRunConfig, result: ContainerRunResult, memory: dict, load: str) -> list[str]:
    """Identity and unified-memory lines for a metal-native candidate, from the config's pin and the result's
    ``memory`` block. Every figure that was not measured reads ``unmeasured``; none is derived from another."""
    server = config.native_server
    pressure = pressure_text(memory["pressure_level"]) or "unmeasured"
    lines = [f"- llama-server: `{_cell(server.executable)}` sha256 `{server.executable_sha256}`, libraries "
             f"`{server.libraries_sha256}`, build {_cell(server.build_info)}, source "
             f"{_cell(server.source or 'not recorded')}",
             _engine_line(config.engine),
             f"- Elapsed {result.elapsed_seconds:.1f}s; model load {load}; VRAM after load not applicable: "
             "unified memory",
             f"- Unified memory after load: server footprint {_mib_text(memory['server_footprint_mib'])} (RSS "
             f"{_mib_text(memory['server_rss_mib'])}); Metal buffers {_mib_text(memory['metal_mib'])} of "
             f"{_mib_text(memory['metal_budget_mib'])} budget; host available "
             f"{_mib_text(memory['host_available_mib'])}; swap used {_mib_text(memory['swap_used_mib'])} (growth "
             f"{_mib_text(memory['swap_growth_mib'])}); pressure level {pressure}; power: "
             f"{_cell(memory['power'] or 'unmeasured')}"]
    violations = memory["watchdog_violations"]
    if violations is not None:  # the watchdog ran: its summary exists even when nothing was violated
        peak = _mib_text(memory["server_footprint_peak_mib"])
        lines.append(f"- During evaluation: peak server footprint {peak}; "
                     f"lowest host available {_mib_text(memory['min_host_available_mib'])}; highest pressure level "
                     f"{pressure_text(memory['max_pressure_level']) or 'unmeasured'}; watchdog violations: "
                     + (", ".join(_cell(item) for item in violations) or "none"))
    if str(memory["power"] or "").startswith("battery"):
        lines.append("- Measured on battery power: macOS may lower CPU and GPU clocks on battery, so speed on AC "
                     "can differ.")
    if memory["coding_sandbox_status"] is not None:
        lines.append(f"- Coding sandbox: {_cell(memory['coding_sandbox_status'])}"
                     + (f" ({_cell(memory['coding_sandbox_reason'])})" if memory["coding_sandbox_reason"] else "")
                     + ("" if memory["coding_sandbox_status"] == "available" else
                        "; the suites that execute generated code were not run, and generated code is never "
                        "executed on the host"))
    return lines


def _coding_blocked(memory: dict | None) -> bool:
    return memory is not None and memory["coding_sandbox_status"] not in (None, "available")


def candidate_markdown(config: ContainerRunConfig, result: ContainerRunResult, row: dict,
                       policy: CampaignPolicy) -> str:
    engine, model, image, speed = config.engine, config.model, config.inference_image, row["speed"]
    native = config.runtime == NATIVE_RUNTIME
    memory = unified_memory_evidence(result.memory) if native else None
    banner = ("**SYNTHETIC HARNESS TEST - no model performance measured.**" if result.synthetic else
              "Measured llama.cpp native (Metal) candidate. One candidate never establishes a validated preset."
              if native else
              "Measured llama.cpp container candidate. One candidate never establishes a validated preset.")
    stage = f" at stage `{_cell(result.failure_stage)}`" if result.failure_stage else ""
    load = "unmeasured" if result.load_seconds is None else f"{result.load_seconds:.1f}s"
    vram = result.vram_used_mib_after_load
    vram = "unmeasured" if vram is None else f"{vram:.0f} MiB"
    rates = " / ".join(_number(speed.get(key)) for key in ("minimum_native_tps", "median_native_tps",
                                                           "maximum_native_tps"))
    lines = [f"# {'Native (Metal)' if native else 'Container'} candidate `{_cell(config.label)}`", "", banner, "",
             f"- State: **{_cell(result.state)}**{stage}",
             f"- Attempt `{result.attempt_id}`, project `{_cell(result.project_name)}`",
             f"- Configuration `{result.config_fingerprint}`",
             f"- Model: {_cell(model.model_name)} {_cell(model.quantization)} (`{model.sha256}`)"]
    if native:
        lines += _native_lines(config, result, memory, load)
    else:
        lines += [f"- Image: `{_cell(image.reference)}` build {_cell(image.build_info)}", _engine_line(engine),
                  f"- Elapsed {result.elapsed_seconds:.1f}s; model load {load}; VRAM after load {vram}"]
    lines += [f"- Cleanup verified: **{result.cleanup.verified}**; abort campaign: {result.abort_campaign}"
              + (f"; server exit {_cell(result.cleanup.server_exit)}" if native and result.cleanup.server_exit
                 else "")
              + (f"; still present: {', '.join(_cell(item) for item in result.cleanup.processes_remaining)}"
                 if native and result.cleanup.processes_remaining else ""), ""]
    for title, items in (("Failure reasons", result.failure_reasons), ("Warnings", result.warnings)):
        if items:
            lines += [f"## {title}", "", *(f"- {_cell(item)}" for item in items), ""]
    if native:
        from .runtime import not_enforced_settings
        lines += ["## Settings this runtime does not enforce", "",
                  *(f"- {_cell(item)}" for item in not_enforced_settings(config)), ""]

    def score(name: str) -> str:
        # Rows a blocked sandbox backfilled score 0.0 in the denominator; printed, that zero would read as a model
        # that cannot code. It is a harness that could not run the code, so the cell says that instead.
        if name == "coding" and _coding_blocked(memory):
            return SANDBOX_BLOCKED
        return _number(row["quality"][name]["score"])

    floor = policy.acceptance.minimum_tokens_per_second
    lines += ["## Speed", "",
              f"- Meets the {floor:g} tok/s floor (a floor, never a ceiling): {speed.get('qualifies')}",
              f"- Native tok/s minimum / median / maximum: {rates} over {speed.get('attempted')} repetitions "
              f"at {config.requested_input_tokens} requested input tokens",
              *(f"- {_cell(item)}" for item in speed.get("reasons", [])), "",
              "## Quality (failures stay in the denominator)", "",
              *_table(("Category", "Attempted", "Score"),
                      [(name, row["quality"][name]["attempted"], score(name)) for name in CATEGORIES],
                      numeric=(1, 2)),
              f"Eligible: {row['eligibility']['eligible']} - "
              + ", ".join(_cell(item) for item in row["eligibility"]["reasons"]), "",
              "## Requested versus effective settings", "",
              *_table(("Control", "Required", "Requested", "Effective", "Status", "Evidence"),
                      [(item.control, item.required, item.requested, item.effective, item.status, item.evidence)
                       for item in result.settings]),
              "Required controls without API, log or behaviour evidence: "
              + (", ".join(unverified_required(result.settings)) or "none") + ".", "",
              "## Stages", "",
              *_table(("Stage", "Status", "Offset s", "Elapsed s", "Detail"),
                      [(item.name, item.status, f"{item.started_offset_seconds:.1f}",
                        f"{item.elapsed_seconds:.1f}", item.detail) for item in result.stages], numeric=(2, 3)),
              "Context capacity, filled prompt tokens and tested usable context are different measurements.",
              *([CANDIDATE_UNIFIED_NOTE] if native else []), ""]
    return "\n".join(lines)


def write_candidate_reports(config: ContainerRunConfig, result: ContainerRunResult, evaluation: dict,
                            policy: CampaignPolicy | None, run_dir: str | Path, *, artifacts=None) -> dict:
    policy = policy or default_policy(config)
    row = evidence_row(config, result, policy)
    campaign = {"campaign_id": f"container-candidate-{result.attempt_id}",
                "policy": policy.model_dump(mode="json"), "results": [row], "skipped": [], "frontier": [], "winner": None,
                "automatic_deployment_performed": False,
                "evaluation_errors": [str(item) for item in evaluation.get("errors") or []]}
    from .artifacts import RunArtifacts
    writer = artifacts or RunArtifacts(run_dir, config.limits.max_artifact_bytes)
    files = {"json": ("reports/campaign.json", json.dumps(campaign, indent=2, ensure_ascii=False)),
             "markdown": ("reports/report.md", markdown_report(campaign)),
             "html": ("reports/report.html", html_report(campaign)),
             "candidate": ("reports/candidate.md", candidate_markdown(config, result, row, policy))}
    encoded = {key: (name, text.encode("utf-8")) for key, (name, text) in files.items()}
    if sum(len(content) for _, content in encoded.values()) > writer.remaining_bytes:
        raise ValueError("artifact budget exhausted before report generation")
    for name, content in encoded.values():
        writer.write(name, content)
    return {**{key: name for key, (name, _) in encoded.items()}, "eligibility": row["eligibility"]}
