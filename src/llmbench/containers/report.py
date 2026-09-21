"""Candidate evidence row and human-readable reports. Pure file output; nothing is loaded or deployed."""

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

CATEGORIES = ("coding", "tools", "retrieval")


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


def candidate_markdown(config: ContainerRunConfig, result: ContainerRunResult, row: dict,
                       policy: CampaignPolicy) -> str:
    engine, model, image, speed = config.engine, config.model, config.inference_image, row["speed"]
    banner = ("**SYNTHETIC HARNESS TEST - no model performance measured.**" if result.synthetic else
              "Measured llama.cpp container candidate. One candidate never establishes a validated preset.")
    stage = f" at stage `{_cell(result.failure_stage)}`" if result.failure_stage else ""
    load = "unmeasured" if result.load_seconds is None else f"{result.load_seconds:.1f}s"
    vram = result.vram_used_mib_after_load
    vram = "unmeasured" if vram is None else f"{vram:.0f} MiB"
    rates = " / ".join(_number(speed.get(key)) for key in ("minimum_native_tps", "median_native_tps",
                                                           "maximum_native_tps"))
    lines = [f"# Container candidate `{_cell(config.label)}`", "", banner, "",
             f"- State: **{_cell(result.state)}**{stage}",
             f"- Attempt `{result.attempt_id}`, project `{_cell(result.project_name)}`",
             f"- Configuration `{result.config_fingerprint}`",
             f"- Model: {_cell(model.model_name)} {_cell(model.quantization)} (`{model.sha256}`)",
             f"- Image: `{_cell(image.reference)}` build {_cell(image.build_info)}",
             f"- Context {engine.ctx_size}, K/V {engine.cache_type_k}/{engine.cache_type_v}, flash attention "
             f"{engine.flash_attn}, GPU layers {engine.n_gpu_layers}, speculation {engine.spec_type}, "
             f"reasoning {engine.reasoning}",
             f"- Elapsed {result.elapsed_seconds:.1f}s; model load {load}; VRAM after load {vram}",
             f"- Cleanup verified: **{result.cleanup.verified}**; abort campaign: {result.abort_campaign}", ""]
    for title, items in (("Failure reasons", result.failure_reasons), ("Warnings", result.warnings)):
        if items:
            lines += [f"## {title}", "", *(f"- {_cell(item)}" for item in items), ""]
    floor = policy.acceptance.minimum_tokens_per_second
    lines += ["## Speed", "",
              f"- Meets the {floor:g} tok/s floor (a floor, never a ceiling): {speed.get('qualifies')}",
              f"- Native tok/s minimum / median / maximum: {rates} over {speed.get('attempted')} repetitions "
              f"at {config.requested_input_tokens} requested input tokens",
              *(f"- {_cell(item)}" for item in speed.get("reasons", [])), "",
              "## Quality (failures stay in the denominator)", "",
              *_table(("Category", "Attempted", "Score"),
                      [(name, row["quality"][name]["attempted"], _number(row["quality"][name]["score"]))
                       for name in CATEGORIES], numeric=(1, 2)),
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
              "Context capacity, filled prompt tokens and tested usable context are different measurements.", ""]
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
