"""Pure planning of the exact llama-server argv and the Compose project. No I/O, no Docker."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .capabilities import ALLOWED_FLAGS, DENIED_FLAGS
from .config import MODEL_CONTAINER_PATH, NAME, ChildGrant, ContainerRunConfig

__all__ = ["ALLOWED_FLAGS", "DENIED_FLAGS", "ComposePlan", "build_compose_plan", "build_server_argv",
           "evaluator_argv"]

INFERENCE_PORT = 8080
CONFIG_CONTAINER_PATH = "/run/llmbench/config.json"
POLICY_CONTAINER_PATH = "/run/llmbench/runtime-policy.json"
ARTIFACTS_CONTAINER_PATH = "/artifacts"
SPOOL_REQUESTS_PATH = "/spool/requests"
SPOOL_RESULTS_PATH = "/spool/results"
EVALUATOR_STOP_GRACE = "10s"


def build_server_argv(config: ContainerRunConfig) -> tuple[str, ...]:
    engine = config.engine
    argv = ["--model", MODEL_CONTAINER_PATH, "--host", "0.0.0.0", "--port", str(INFERENCE_PORT),
            "--alias", config.alias(), "--ctx-size", str(engine.ctx_size),
            "--n-gpu-layers", str(engine.n_gpu_layers),
            "--cache-type-k", engine.cache_type_k, "--cache-type-v", engine.cache_type_v,
            "--kv-offload" if engine.kv_offload else "--no-kv-offload",
            "--flash-attn", engine.flash_attn, "--load-mode", engine.load_mode,
            "--batch-size", str(engine.batch_size), "--ubatch-size", str(engine.ubatch_size),
            "--threads", str(engine.threads), "--threads-batch", str(engine.threads_batch),
            "--parallel", str(engine.parallel),
            "--cache-prompt" if engine.cache_prompt else "--no-cache-prompt",
            "--cache-reuse", str(engine.cache_reuse), "--cache-ram", str(engine.cache_ram_mib),
            "--no-context-shift", "--fit", engine.fit]
    if engine.swa_full:
        argv.append("--swa-full")
    argv += ["--jinja", "--reasoning", engine.reasoning, "--reasoning-format", engine.reasoning_format,
             "--reasoning-budget", str(engine.reasoning_budget), "--spec-type", engine.spec_type]
    if engine.spec_type != "none":
        argv += ["--spec-draft-n-max", str(engine.spec_draft_n_max)]
    argv += ["--warmup" if engine.warmup else "--no-warmup", "--no-ui", "--metrics", "--slots", "--offline",
             "--log-verbosity", str(engine.log_verbosity)]
    return tuple(argv)


def evaluator_argv(grant: ChildGrant) -> tuple[str, ...]:
    """Arguments appended to the evaluator image entrypoint (`python -m llmbench.container_eval`)."""
    return ("--config", CONFIG_CONTAINER_PATH, "--policy", POLICY_CONTAINER_PATH,
            "--grant-seconds", str(grant.grant_seconds), "--artifact-bytes", str(grant.artifact_bytes))


@dataclass(frozen=True)
class ComposePlan:
    project_name: str
    compose: Mapping[str, Any]
    server_argv: tuple[str, ...]
    labels: Mapping[str, str]
    compose_path: Path

    def compose_json(self) -> str:
        return json.dumps(self.compose, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _hardened(image, labels: dict, role: str) -> dict:
    return {"image": image.reference, "pull_policy": "never", "entrypoint": list(image.entrypoint),
            "restart": "no", "init": True, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"],
            "healthcheck": {"disable": True}, "labels": {**labels, "llmbench.role": role}}


def _bind(source: Path | str, target: str, *, read_only: bool) -> dict:
    mount = {"type": "bind", "source": str(source), "target": target, "bind": {"create_host_path": False}}
    if read_only:
        mount["read_only"] = True
    return mount


def _evaluator_service(config: ContainerRunConfig, run_dir: Path, labels: dict, grant: ChildGrant,
                       logging: dict) -> dict:
    limits, plan_dir = config.limits, run_dir / "plan"
    volumes = [_bind(plan_dir / "evaluator-config.json", CONFIG_CONTAINER_PATH, read_only=True),
               _bind(plan_dir / "runtime-policy.json", POLICY_CONTAINER_PATH, read_only=True),
               _bind(run_dir / "evaluator", ARTIFACTS_CONTAINER_PATH, read_only=False)]
    if config.broker is not None:
        volumes += [_bind(run_dir / "spool" / "requests", SPOOL_REQUESTS_PATH, read_only=False),
                    _bind(run_dir / "spool" / "results", SPOOL_RESULTS_PATH, read_only=True)]
    return {
        **_hardened(config.evaluator.image, labels, "evaluator"), "read_only": True, "user": "10001:10001",
        "command": list(evaluator_argv(grant)), "networks": ["bench"],
        "tmpfs": [f"/tmp:size={limits.evaluator_tmpfs_mib}m,mode=1777"], "stop_grace_period": EVALUATOR_STOP_GRACE,
        "logging": logging,
        # LIVE-006: the pids ceiling is declared once, under deploy.resources.limits beside cpus/memory. A
        # top-level `pids_limit` next to a deploy limits block fails compose-go's consistency check
        # ("can't set distinct values on 'pids_limit' and 'deploy.resources.limits.pids'").
        "deploy": {"resources": {"limits": {"cpus": str(limits.evaluator_cpus),
                                            "memory": str(limits.evaluator_memory_mib * 1048576),
                                            "pids": limits.evaluator_pids}}},
        "volumes": volumes}


def build_compose_plan(config: ContainerRunConfig, run_dir: str | Path, *, attempt_id: str,
                       session_id: str, grant: ChildGrant | None = None) -> ComposePlan:
    if not re.fullmatch(r"[0-9a-f]{32}", attempt_id) or not re.fullmatch(NAME, session_id):
        raise ValueError("attempt_id must be 32 lowercase hex characters and session_id a safe name")
    host_mode = config.evaluator.mode == "host-process"
    if host_mode is not (grant is None):
        raise ValueError("a ChildGrant is required exactly when the evaluator runs in container mode")
    if grant is not None and grant.artifact_bytes != config.limits.evaluator_artifact_bytes:
        raise ValueError("the grant's artifact allocation must equal limits.evaluator_artifact_bytes")
    project = "llmbench-" + attempt_id[:12]
    labels = {"llmbench.owner": "llmbench", "llmbench.session": session_id, "llmbench.attempt": attempt_id}
    limits, argv = config.limits, build_server_argv(config)
    maximum = limits.log_max_bytes // (1024 * 1024)
    logging = {"driver": "local", "options": {"max-size": f"{max(1, maximum)}m", "max-file": "2"}}
    inference = {
        **_hardened(config.inference_image, labels, "inference"), "command": list(argv),
        "stop_grace_period": "15s",
        "logging": logging,
        "deploy": {"resources": {
            "limits": {"cpus": str(limits.inference_cpus), "memory": str(limits.inference_memory_mib * 1048576)},
            "reservations": {"devices": [{"driver": "nvidia", "device_ids": [limits.gpu_device_id],
                                          "capabilities": ["gpu"]}]}}},
        "volumes": [{"type": "bind", "source": config.model.host_path, "target": config.model.container_path,
                     "read_only": True, "bind": {"create_host_path": False}}],
        "networks": ["bench", "hostlink"] if host_mode else ["bench"]}
    networks = {"bench": {"internal": True, "labels": {**labels, "llmbench.role": "network"}}}
    services = {"inference": inference}
    if host_mode:
        # An internal network cannot publish ports; the host evaluator reaches a loopback-only ephemeral port.
        inference["ports"] = [{"target": INFERENCE_PORT, "host_ip": "127.0.0.1", "protocol": "tcp"}]
        networks["hostlink"] = {"labels": {**labels, "llmbench.role": "network"}}
    else:
        services["evaluator"] = _evaluator_service(config, Path(run_dir), labels, grant, logging)
    compose = {"name": project, "services": services, "networks": networks}
    return ComposePlan(project, compose, argv, labels, Path(run_dir) / "plan" / "compose.json")
