"""Persistent multi-hour tuning session over llama.cpp container candidates.

A session is a frozen search space, a holdout plan and budgets around a ``ContainerRunConfig`` template.
``tune`` derives every candidate, writes an exclusive ``session.json`` ledger with the wall-clock deadline,
runs the shared campaign controller with the container executor and writes reports. ``resume`` rebuilds the
identical manifests, honours the persisted deadline (downtime is never refunded) and continues. Import is
inert; nothing here talks to Docker or a model.

The runtime that serves the candidates (``nvidia-container`` or ``metal-native``, see ``config.RUNTIMES``) is a
property of the candidate configs, not of the session machinery: the same ledger, campaign and reports drive
both. What differs is which prepared bundle pins the server (``image-bundle.json`` or ``native-bundle.json``,
told apart by ``read_bundle``), which permissions a run needs (``runtime.required_operations``) and which runner
executes it (``runtime.DispatchRunner``). A session that names no runtime is exactly the NVIDIA session it was
before runtimes existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_serializer, model_validator

from ..config import (BackendSettings, CampaignPolicy, ExperimentManifest, ModelArtifact, RunMode,
                      StrictModel, TaskSelection, canonical_json)
from ..safety import OperationForbidden, SessionLock
from .config import (NAME, RUNTIMES, CacheType, ContainerRunConfig, ImageBundle, LlamaCppSettings, ModelAsset,
                     NativeBundle, NativeLimits, read_image_bundle)

DEFAULT_CAPABILITIES = "artifacts/container-prep"
DEFAULT_BASE_CONFIG = str(Path(__file__).with_name("data") / "base-candidate.json")
"""The candidate template ``tune --model`` starts from: pinned inference image, engine defaults and the
local fixtures. Its model asset is a placeholder that derivation replaces with the GGUFs it is given.
Packaged, so the command works from any directory; ``examples/candidate.json`` is the same file."""
LEDGER_NAME = "session.json"
SESSION_CONFIG_NAME = "session-config.json"
SUMMARY_NAME = "session-summary.json"
MAX_JSON_BYTES = 16 * 1024 * 1024
MIN_CONTEXT_INPUT_TOKENS = 512  # the engine's smallest ctx_size; a usable-input bound below it is meaningless
# Engine fields carried into BackendSettings; every other LlamaCppSettings field is hashed into runtime_revision.
MAPPED_ENGINE_FIELDS = frozenset({"engine", "ctx_size", "n_gpu_layers", "cache_type_k", "cache_type_v", "kv_offload",
                                  "flash_attn", "spec_type", "parallel", "batch_size", "ubatch_size", "threads",
                                  "reasoning"})
NVIDIA_OPERATIONS = ("container", "load", "inference")
"""What an NVIDIA container session has always needed from ``runtime-policy.json``. The runtime registry
(``runtime.required_operations``) returns exactly this tuple for an NVIDIA config; it is spelled out here only
for the one check that runs before any config exists (``main_tune --model`` authorizes before hashing GGUFs)."""
DEFAULT_PLANNING_SLOWDOWN = 1.0
METAL_REVISION_PREFIX = "metal:"
"""Prefixed to a metal-native candidate's ``runtime_revision`` so a CUDA and a Metal result of the same llama.cpp
commit are never the same backend to the controller's comparison rules."""


class SessionBudgets(StrictModel):
    wall_seconds: int = Field(default=14400, ge=600, le=86400)
    holdout_reserve_seconds: int = Field(default=2400, ge=0)
    cleanup_reserve_seconds: int = Field(default=120, ge=15)
    max_candidates: int = Field(default=12, ge=1, le=64)
    candidate_wall_seconds: int = Field(default=1800, ge=60, le=14400)
    max_validation_candidates: int = Field(default=2, ge=1, le=10)

    @model_validator(mode="after")
    def coherent(self) -> "SessionBudgets":
        if self.holdout_reserve_seconds + self.cleanup_reserve_seconds >= self.wall_seconds:
            raise ValueError("holdout and cleanup reserves must leave wall budget for development candidates")
        if self.candidate_wall_seconds + self.holdout_reserve_seconds > self.wall_seconds:
            raise ValueError("not even one development candidate fits beside the holdout reserve")
        return self


def output_reserve_tokens(config: ContainerRunConfig) -> int:
    """The output capacity a candidate reserves above its input: the larger of the generation and speed budgets.

    This is the same quantity ``ContainerRunConfig.coherent`` uses for its capacity rule, so a fill computed
    from it can never produce a configuration the candidate schema refuses.
    """
    return max(config.generation.max_output_tokens, config.speed.output_tokens)


def usable_input_tokens(allocation: int, *, template_reserve_tokens: int, output_tokens: int,
                        ceiling: int | None = None) -> int:
    """Usable INPUT tokens inside a context ``allocation`` (the engine's ``ctx_size``).

    ``context_floor``/``context_ceiling`` are USABLE INPUT TOKENS: the template reserve and the output capacity
    are reserved ON TOP of them and are never counted as input. A range ``ceiling`` clamps the fill so rounding
    the allocation up never makes a candidate claim more input than the session asked for.
    """
    usable = allocation - template_reserve_tokens - output_tokens
    return min(usable, ceiling) if ceiling is not None else usable


class SearchSpace(StrictModel):
    quantizations: tuple[str, ...] = Field(min_length=1)
    kv_pairs: tuple[tuple[CacheType, CacheType], ...] = Field(min_length=1)
    ctx_tiers: tuple[int, ...] = Field(min_length=1)
    spec_types: tuple[Literal["none", "draft-mtp"], ...] = Field(min_length=1)
    reasoning: tuple[Literal["on", "off", "auto"], ...] = Field(min_length=1)
    batch_sizes: tuple[int, ...] = Field(min_length=1)
    # ---- optional explicit context range. Both bounds are USABLE INPUT TOKENS; the template reserve and
    # the output capacity are reserved on top of them. When present the range REPLACES the fixed default tiers:
    # ``ctx_tiers`` are then the allocations derived from it (containers/derive.py) and every candidate's
    # ``requested_input_tokens`` is filled to its tier's input target instead of the base fill ratio.
    context_floor: int | None = Field(default=None, ge=MIN_CONTEXT_INPUT_TOKENS)
    context_ceiling: int | None = Field(default=None, ge=MIN_CONTEXT_INPUT_TOKENS)
    context_n_ctx_train: int | None = Field(default=None, ge=MIN_CONTEXT_INPUT_TOKENS)
    context_points_dropped: tuple[int, ...] = ()
    # ---- optional CPU/RAM offload axes. Pushing work to system RAM is a first-class comparison, not a
    # fallback: ``gpu_layers`` moves whole model layers off the GPU (``--n-gpu-layers``; anything below the
    # model's block count leaves the remaining layers in system RAM) and ``kv_offload`` places the KV cache on
    # the GPU (True, ``--kv-offload``) or in system RAM (False, ``--no-kv-offload``). Both default to a single
    # member equal to today's behaviour, so every session config written before these axes existed validates and schedules
    # exactly as it did: all layers on the GPU, KV cache on the GPU.
    gpu_layers: tuple[Annotated[int, Field(ge=0, le=999)] | Literal["all"], ...] = Field(default=("all",),
                                                                                        min_length=1)
    kv_offload: tuple[bool, ...] = Field(default=(True,), min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> "SearchSpace":
        for name in ("quantizations", "kv_pairs", "ctx_tiers", "spec_types", "reasoning", "batch_sizes"):
            values = getattr(self, name)
            if len(set(values)) != len(values) or not all(values):
                raise ValueError(f"{name} must be unique nonempty values")
        # ``0`` GPU layers and ``False`` (KV cache in RAM) are legitimate members, so the offload axes are checked
        # for uniqueness only; emptiness is the field's own min_length.
        for name in ("gpu_layers", "kv_offload"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique values")
        if tuple(sorted(self.ctx_tiers)) != self.ctx_tiers:
            raise ValueError("ctx_tiers must ascend")
        for tier in self.ctx_tiers:
            LlamaCppSettings(ctx_size=tier, batch_size=32, ubatch_size=32)  # engine bounds for the tier
        for batch in self.batch_sizes:
            LlamaCppSettings(ctx_size=1_048_576, batch_size=batch, ubatch_size=32)  # engine bounds for the batch
        for key, value in self.kv_pairs:  # the engine's flash-attention K/V kernel rule, no exceptions
            LlamaCppSettings(ctx_size=self.ctx_tiers[0], cache_type_k=key, cache_type_v=value, flash_attn="on",
                             batch_size=32, ubatch_size=32)
        for layers in self.gpu_layers:  # engine bounds for the layer count (0..999 or "all")
            LlamaCppSettings(ctx_size=self.ctx_tiers[0], n_gpu_layers=layers, batch_size=32, ubatch_size=32)
        for placement in self.kv_offload:  # engine bounds for the KV placement flag
            LlamaCppSettings(ctx_size=self.ctx_tiers[0], kv_offload=placement, batch_size=32, ubatch_size=32)
        return self._coherent_context_range()

    def _coherent_context_range(self) -> "SearchSpace":
        """A range is given whole or not at all, ascends, and names the training context it was checked against.

        The bound-versus-reserve arithmetic needs the candidate template's reserve and output budgets, so the
        exact reachability rule lives in ``ContainerSessionConfig.coherent``; only model-independent facts are
        checked here.
        """
        given = [name for name in ("context_floor", "context_ceiling") if getattr(self, name) is not None]
        if len(given) == 1:
            raise ValueError("context_floor and context_ceiling are usable INPUT token bounds and must be given "
                             f"together; only {given[0]} was set")
        if not given:
            if self.context_n_ctx_train is not None or self.context_points_dropped:
                raise ValueError("context_n_ctx_train and context_points_dropped belong to a context range; set "
                                 "context_floor and context_ceiling or leave all four unset")
            return self
        if self.context_floor > self.context_ceiling:
            raise ValueError(f"context_floor {self.context_floor} exceeds context_ceiling {self.context_ceiling} "
                             "(both are usable INPUT tokens)")
        if self.context_n_ctx_train is None:
            raise ValueError("a context range must record the model's training context as context_n_ctx_train so "
                             "the ceiling can be checked against it; fail closed rather than guess")
        if self.context_ceiling >= self.context_n_ctx_train:
            raise ValueError(f"context_ceiling {self.context_ceiling} usable input tokens is unreachable: the "
                             f"model's training context is {self.context_n_ctx_train} tokens and the template "
                             "reserve plus the output capacity are reserved on top of the input")
        if max(self.ctx_tiers) > self.context_n_ctx_train:
            raise ValueError(f"context tier allocation {max(self.ctx_tiers)} exceeds the model's training context "
                             f"{self.context_n_ctx_train}")
        outside = [item for item in self.context_points_dropped
                   if not self.context_floor <= item <= self.context_ceiling]
        if outside or len(set(self.context_points_dropped)) != len(self.context_points_dropped):
            raise ValueError(f"dropped context points must be distinct input targets inside the range; {outside} "
                             "lie outside it")
        return self

    def context_range(self) -> dict | None:
        """The declared range as plain JSON (usable input tokens), or None when no range was given."""
        if self.context_floor is None:
            return None
        return {"context_floor": self.context_floor, "context_ceiling": self.context_ceiling,
                "n_ctx_train": self.context_n_ctx_train, "unit": "usable input tokens",
                "dropped_input_targets": list(self.context_points_dropped)}


class HoldoutPlan(StrictModel):
    niah_task_ids: tuple[str, ...] = Field(min_length=1)
    niah_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def known_variants(self) -> "HoldoutPlan":
        from ..evaluations.retrieval import NIAH_VARIANTS
        unknown = [item for item in self.niah_task_ids if item not in NIAH_VARIANTS]
        if unknown or len(set(self.niah_task_ids)) != len(self.niah_task_ids):
            raise ValueError(f"holdout NIAH variants must be unique known names; unknown: {unknown}")
        return self


def _default_policy() -> CampaignPolicy:
    return CampaignPolicy(name="container-session", mode=RunMode.LIVE)


class ContainerSessionConfig(StrictModel):
    schema_version: Literal[1] = 1
    session_id: str = Field(pattern=NAME)
    image_bundle: str | None = Field(default=None, min_length=1)
    assets: tuple[ModelAsset, ...] = Field(min_length=1)
    base: ContainerRunConfig
    search: SearchSpace
    holdout: HoldoutPlan
    budgets: SessionBudgets = Field(default_factory=SessionBudgets)
    proposal_mode: Literal["deterministic", "file"] = "deterministic"
    proposal_file: str | None = Field(default=None, min_length=1)
    policy: CampaignPolicy = Field(default_factory=_default_policy)
    planning_slowdown: float = Field(default=DEFAULT_PLANNING_SLOWDOWN, gt=0, le=100)
    """The factor derivation multiplied every per-item public-benchmark cost estimate by when it chose what fits
    a candidate's wall (``derive.PLANNING_SLOWDOWN``): 1.0 for the NVIDIA host the estimates were measured on, more
    for a slower runtime. Frozen here so the record says what the selection was sized for; nothing at run time
    reads it, because the adapters' own budget probes still govern what actually runs. Left out of the JSON at its
    default, so an NVIDIA ``session-config.json`` (and the sha256 its ledger pins) is byte-identical to before."""

    @model_serializer(mode="wrap")
    def _omit_default_slowdown(self, handler):
        data = handler(self)
        if isinstance(data, dict) and data.get("planning_slowdown") == DEFAULT_PLANNING_SLOWDOWN:
            data.pop("planning_slowdown")
        return data

    @model_validator(mode="after")
    def coherent(self) -> "ContainerSessionConfig":
        quantizations = [item.quantization for item in self.assets]
        if len(set(quantizations)) != len(quantizations):
            raise ValueError("each session asset must carry a distinct quantization")
        if len({item.sha256 for item in self.assets}) != len(self.assets):
            raise ValueError("each session asset must be a distinct file")
        missing = [item for item in self.search.quantizations if item not in quantizations]
        if missing:
            raise ValueError(f"search quantizations without an asset: {missing}")
        if any(item.split != "development" for item in self.base.benchmarks):
            raise ValueError("base benchmarks must all be development selections; holdout comes from the plan")
        development_seeds = {item.seed for item in self.base.benchmarks if item.benchmark_id == "niah"}
        if self.holdout.niah_seed in development_seeds:
            raise ValueError("holdout NIAH seed must differ from every development seed")
        if self.policy.mode != RunMode.LIVE:
            raise ValueError("a container session is a live campaign; policy.mode must be live")
        if (self.proposal_mode == "file") is (self.proposal_file is None):
            raise ValueError("proposal_file is required exactly when proposal_mode is file")
        if self.budgets.candidate_wall_seconds <= self.base.bounds.minimum_wall_seconds():
            raise ValueError("candidate_wall_seconds must exceed the base stage bounds")
        self._coherent_offload_axes()
        self._coherent_context_tiers()
        from .proposals import AXES, Proposal, apply, baseline_proposal
        baseline = apply(self, self.base, baseline_proposal(self))
        family = {"quantization": "weights", "kv_pair": "kv", "ctx_tier": "context", "spec_type": "speculation",
                  "reasoning": "reasoning", "batch_size": "performance", "gpu_layers": "offload",
                  "kv_offload": "kv-placement"}
        members = {"quantization": self.search.quantizations, "kv_pair": self.search.kv_pairs,
                   "ctx_tier": self.search.ctx_tiers, "spec_type": self.search.spec_types,
                   "reasoning": self.search.reasoning, "batch_size": self.search.batch_sizes,
                   "gpu_layers": self.search.gpu_layers, "kv_offload": self.search.kv_offload}
        for axis in AXES:  # every single-axis variation must be an engine-valid, capacity-valid candidate
            for value in members[axis]:
                try:
                    apply(self, baseline, Proposal(family=family[axis], changes={axis: value}))
                except ValueError as exc:
                    raise ValueError(f"search axis {axis}={value!r} is invalid against the base: {exc}") from exc
        self._coherent_context_schedule()  # after the axis loop: it needs a schedule of valid candidates
        return self

    def _coherent_offload_axes(self) -> None:
        """The base candidate's offload settings must be members of the axes that will sweep them.

        ``baseline_proposal`` pins EVERY axis to its first member, so a base candidate that asks for partial
        offload (or the KV cache in RAM) while the search space still carries the single default member would be
        silently re-tuned to "all layers on the GPU" and the session would measure a configuration nobody asked
        for. Name the conflict instead of resolving it.
        """
        for axis, declared, actual in (("gpu_layers", self.search.gpu_layers, self.base.engine.n_gpu_layers),
                                       ("kv_offload", self.search.kv_offload, self.base.engine.kv_offload)):
            if actual not in declared:
                raise ValueError(f"the base candidate's engine sets {axis}={actual!r}, which search.{axis} "
                                 f"{list(declared)} does not list; the baseline pins every axis to its first "
                                 f"member, so list {actual!r} in search.{axis} or change the base candidate "
                                 "rather than let the session silently re-tune the offload configuration")

    def _coherent_context_tiers(self) -> None:
        """A declared range must be reachable on this model and actually covered by the tiers that will run.

        ``context_ceiling`` is USABLE INPUT: the allocation the engine needs for it is
        ``ceiling + template_reserve_tokens + max(generation.max_output_tokens, speed.output_tokens)``. If that
        exceeds the model's training context the session is refused (naming the training context); if no tier
        allocates it, the ceiling would never be measured and "supports N tokens" could never be evidenced.
        """
        search = self.search
        if search.context_ceiling is None:
            return
        reserve, output = self.base.template_reserve_tokens, output_reserve_tokens(self.base)
        needed = search.context_ceiling + reserve + output
        if needed > search.context_n_ctx_train:
            raise ValueError(
                f"context_ceiling {search.context_ceiling} usable input tokens needs a {needed}-token allocation "
                f"(input + {reserve} template reserve + {output} output), but the model's training context is "
                f"{search.context_n_ctx_train} tokens")
        if max(search.ctx_tiers) < needed:
            raise ValueError(
                f"no context tier allocates the {needed} tokens the ceiling needs (largest tier "
                f"{max(search.ctx_tiers)}), so context_ceiling {search.context_ceiling} would never be measured")
        for tier in search.ctx_tiers:
            usable = usable_input_tokens(tier, template_reserve_tokens=reserve, output_tokens=output,
                                         ceiling=search.context_ceiling)
            if usable < search.context_floor:
                raise ValueError(f"context tier allocation {tier} carries only {usable} usable input tokens, below "
                                 f"context_floor {search.context_floor}")

    def _coherent_context_schedule(self) -> None:
        """A declared ceiling must survive the candidate cap, not merely be allocated by some tier.

        ``proposals.deterministic_schedule`` cuts ``full_schedule`` at ``budgets.max_candidates`` and the context
        sweep is LAST and ascending, so an oversized search loses its ceiling tier first: the session would
        declare a range whose ceiling never runs. ``derive`` sizes the tier list to the cap
        (``derive.context_points_budget``); a session config written any other way is protected only here. A
        proposal FILE names its own candidates and is checked against the frozen space when it is read.
        """
        search = self.search
        if search.context_ceiling is None or self.proposal_mode != "deterministic":
            return
        from .proposals import deterministic_schedule, full_schedule  # local: proposals imports this module
        ceiling_tier, baseline_tier = max(search.ctx_tiers), search.ctx_tiers[0]

        def runs_the_ceiling(proposal) -> bool:  # a proposal that does not move ctx_tier keeps the baseline's
            return proposal.changes.get("ctx_tier", baseline_tier) == ceiling_tier

        if any(runs_the_ceiling(item) for item in deterministic_schedule(self)):
            return
        schedule = full_schedule(self)
        needed = next((index + 1 for index, item in enumerate(schedule) if runs_the_ceiling(item)), len(schedule))
        raise ValueError(
            f"context_ceiling {search.context_ceiling} usable input tokens would never be measured: the "
            f"deterministic schedule is cut to max_candidates {self.budgets.max_candidates} and no candidate left "
            f"in it allocates the {ceiling_tier}-token ceiling tier ({len(search.ctx_tiers)} context tiers "
            f"declared); raise max_candidates to at least {needed} or narrow the context range")


def read_session_config(path: str | Path) -> ContainerSessionConfig:
    return ContainerSessionConfig.model_validate_json(_read_bounded(path))


def _read_bounded(path: str | Path, limit: int = MAX_JSON_BYTES) -> str:
    target = Path(path)
    if not target.is_file() or target.stat().st_size > limit:
        raise ValueError(f"{target} must be a regular file of at most {limit} bytes")
    return target.read_text(encoding="utf-8")



def image_ids(bundle) -> dict[str, str | None] | None:
    """What a bundle pins, as the ledger's ``image_ids``; None when the session runs without a bundle.

    An ``ImageBundle`` pins image IDs (inference, evaluator, worker), exactly as before native runtimes existed. A
    ``NativeBundle`` pins the llama-server executable and its libraries by SHA-256 instead, plus the optional
    sandbox worker image. The ledger key keeps its old name so no schema 1 ledger changes shape; the two kinds of
    value have disjoint keys, so a bundle of the other kind can never compare equal on ``resume``.
    """
    if bundle is None:
        return None
    worker = bundle.worker.image_id if bundle.worker is not None else None
    if isinstance(bundle, NativeBundle):
        return {"native_server": bundle.native_server.executable_sha256,
                "libraries": bundle.native_server.libraries_sha256, "worker": worker}
    return {"inference": bundle.inference.image_id, "evaluator": bundle.evaluator.image_id, "worker": worker}


def read_bundle(path: str | Path) -> ImageBundle | NativeBundle:
    """Either prepared bundle, told apart by what the file declares rather than by what it is called.

    ``llmbench prepare`` writes ``image-bundle.json`` (NVIDIA images); ``prepare --runtime metal-native`` writes
    ``native-bundle.json``, which declares ``"runtime": "metal-native"`` and a ``native_server``. Each is a strict
    model that refuses the other's keys, so a file that is neither (or a hybrid of both) fails validation instead
    of being read as the wrong kind. Bounded like every other JSON this module reads.
    """
    text = _read_bounded(path)
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{path} is not a JSON bundle: {exc}") from exc
    if isinstance(raw, dict) and (raw.get("runtime") == "metal-native" or "native_server" in raw):
        return NativeBundle.model_validate_json(text)
    return ImageBundle.model_validate_json(text)


def native_overlay(raw: dict, bundle: NativeBundle) -> dict:
    """In place: a candidate's JSON rewritten to be served by ``bundle``'s pinned native server. Returns ``raw``.

    Sets the runtime, the server reference and the watchdog limits, and removes the inference image (a native
    candidate runs none; the config validator refuses one). The config's own ``native_limits`` are kept when it
    has them, so limits a session was tuned under survive a moved bundle; otherwise the documented defaults
    apply, which the config then records. The sandbox worker image comes from the bundle when it pins one. The
    evaluator is left as the config states it: a container evaluator cannot reach a host server, and the config
    validator names that conflict instead of this function quietly changing what was asked for.
    """
    raw["runtime"] = "metal-native"
    raw["native_server"] = bundle.native_server.model_dump(mode="json")
    raw["native_limits"] = raw.get("native_limits") or NativeLimits().model_dump(mode="json")
    raw.pop("inference_image", None)
    if bundle.worker is not None:
        raw["worker_image"] = bundle.worker.model_dump(mode="json")
    return raw


def _overlay_bundle(raw: dict, bundle) -> dict:
    """In place: the images or native server ``bundle`` pins, laid over a candidate's JSON. Returns ``raw``."""
    if isinstance(bundle, NativeBundle):
        return native_overlay(raw, bundle)
    if raw.get("runtime", "nvidia-container") != "nvidia-container":
        raise ValueError(f"an image bundle pins a CUDA inference image, but this session runs the {raw['runtime']} "
                         "runtime; pass the native-bundle.json it was prepared with instead")
    raw["inference_image"] = bundle.inference.model_dump(mode="json")
    if raw["evaluator"]["mode"] == "container":
        raw["evaluator"]["image"] = bundle.evaluator.model_dump(mode="json")
    if bundle.worker is not None:
        raw["worker_image"] = bundle.worker.model_dump(mode="json")
    return raw


def bundled_base(session: "ContainerSessionConfig", bundle) -> ContainerRunConfig:
    """The session's base candidate as the bundle will actually serve it (the base itself without a bundle)."""
    if bundle is None:
        return session.base
    raw = _overlay_bundle(session.base.model_dump(mode="json"), bundle)
    return ContainerRunConfig.model_validate_json(canonical_json(raw))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_for(session: ContainerSessionConfig) -> CampaignPolicy:
    budgets = session.budgets
    limits = session.policy.budgets.model_dump(mode="json")
    limits.update(wall_seconds=budgets.wall_seconds, reserve_validation_seconds=budgets.holdout_reserve_seconds,
                  task_timeout_seconds=budgets.candidate_wall_seconds, max_candidates=budgets.max_candidates,
                  max_validation_candidates=budgets.max_validation_candidates)
    raw = session.policy.model_dump(mode="json")
    # The probe count is the candidate's; acceptance must count exactly those repetitions.
    raw["acceptance"]["speed_repetitions"] = session.base.speed.repetitions
    raw.update(name=session.session_id, mode="live", budgets=limits)
    return CampaignPolicy.model_validate_json(canonical_json(raw))


def holdout_selections(session: ContainerSessionConfig) -> tuple[TaskSelection, ...]:
    return (TaskSelection(suite="niah", revision="local-niah-v1", task_ids=session.holdout.niah_task_ids,
                          split="holdout", fixture_seed=session.holdout.niah_seed),)


def run_config_for(session: ContainerSessionConfig, bundle, proposal, *, benchmarks=None,
                   baseline=None) -> ContainerRunConfig:
    """Candidate configuration for a proposal; images (or the native server) come from the bundle, labels from
    the proposal. See ``_overlay_bundle`` for what each kind of bundle lays over the candidate."""
    from .proposals import apply
    if proposal.family == "baseline":
        base = session.base
    else:
        if baseline is None or baseline.family != "baseline":
            raise ValueError("a single-axis proposal needs the session's baseline proposal")
        base = apply(session, session.base, baseline)
    raw = apply(session, base, proposal).model_dump(mode="json")
    raw.update(label=proposal.label(), session_id=session.session_id,
               benchmarks=[item.model_dump(mode="json") for item in (benchmarks or session.base.benchmarks)])
    raw["bounds"]["candidate_wall_seconds"] = session.budgets.candidate_wall_seconds
    if bundle is not None:
        _overlay_bundle(raw, bundle)
    return ContainerRunConfig.model_validate_json(canonical_json(raw))


def _task_selection(selection) -> TaskSelection:
    """The manifest's view of one benchmark selection, carrying its options.

    LIVE-013: ``BenchmarkSelection.to_task_selection`` predates ``TaskSelection.options`` and drops them, so a
    selection's options never reached the manifest and the executor rebuilt the candidate without them. The
    options are revalidated through JSON here, which is the same trip the executor's rebuild makes, so an
    option that cannot survive it is refused at ``tune`` rather than quietly changing a measurement.
    """
    from .campaign import BenchmarkOptionsNotCarried

    base = selection.to_task_selection()
    if not selection.options:
        return base
    try:
        raw = canonical_json({**base.model_dump(mode="json"), "options": selection.options})
        carried = TaskSelection.model_validate_json(raw)
        restored = canonical_json(carried.options)
        declared = canonical_json(selection.options)
    except (TypeError, ValueError) as exc:
        raise BenchmarkOptionsNotCarried(
            f"benchmark {selection.benchmark_id} declares options that cannot be carried into the manifest, so "
            f"a candidate could not be shown to run what the session asked for: {type(exc).__name__}: {exc}"
        ) from exc
    if restored != declared:
        raise BenchmarkOptionsNotCarried(
            f"benchmark {selection.benchmark_id} declared options {declared} but the manifest would record "
            f"{restored}; refusing to record a configuration the session did not ask for")
    return carried


def to_manifest(config: ContainerRunConfig, *, template_hash: str, scorer_revision: str, environment_hash: str,
                block_count: int | None = None) -> ExperimentManifest:
    """Controller manifest for a candidate; unmapped engine fields hash into runtime_revision.

    ``runtime_revision`` is ``<build_info>+<sha12 of the unmapped engine fields>`` for an NVIDIA candidate, as it
    always was. A metal-native candidate's is prefixed ``metal:``: the Metal and CUDA backends of one llama.cpp
    commit are different kernels on different hardware, and the controller treats equal ``BackendSettings`` as the
    same backend. The prefix, rather than a new ``BackendSettings`` field, keeps every NVIDIA manifest fingerprint
    byte-identical.
    """
    engine, asset = config.engine, config.model
    if config.runtime == "metal-native":
        build, prefix = config.native_server.build_info, METAL_REVISION_PREFIX
    else:
        image = config.inference_image
        if not image.build_info:
            raise ValueError("inference image build_info is required for runtime_revision")
        build, prefix = image.build_info, ""
    unmapped = {key: value for key, value in engine.model_dump(mode="json").items()
                if key not in MAPPED_ENGINE_FIELDS}
    revision = f"{prefix}{build}+{hashlib.sha256(canonical_json(unmapped).encode()).hexdigest()[:12]}"
    if engine.n_gpu_layers == "all":
        offload = 1.0
    elif engine.n_gpu_layers == 0:
        offload = 0.0  # nothing on the GPU is a fraction the block count cannot change; every layer is in RAM
    elif type(block_count) is int and 0 < engine.n_gpu_layers <= block_count:
        offload = engine.n_gpu_layers / block_count
    else:
        raise ValueError("a numeric n_gpu_layers needs the model's block_count to express gpu_offload")
    model = ModelArtifact(model_key=f"{asset.model_name}:{asset.quantization}", sha256=asset.sha256,
                          source_revision=asset.source_revision, quantization=asset.quantization,
                          tokenizer_hash="gguf-sha256:" + asset.sha256, template_hash=template_hash,
                          provenance={"size_bytes": asset.size_bytes, "format": asset.format})
    backend = BackendSettings(engine="llama.cpp", runtime_revision=revision, context_length=engine.ctx_size,
                              gpu_offload=offload, k_cache=engine.cache_type_k, v_cache=engine.cache_type_v,
                              kv_on_gpu=engine.kv_offload, flash_attention=engine.flash_attn == "on",
                              mtp=engine.spec_type == "draft-mtp", parallel=engine.parallel,
                              batch_size=engine.batch_size, ubatch_size=engine.ubatch_size,
                              cpu_threads=engine.threads, reasoning=engine.reasoning)
    return ExperimentManifest(
        model=model, backend=backend, generation=config.generation,
        tasks=tuple(_task_selection(item) for item in config.benchmarks),
        harness_revision=config.harness_revision, scorer_revision=scorer_revision,
        environment_hash=environment_hash, mode=RunMode.LIVE,
        requested_input_tokens=config.requested_input_tokens, template_reserve_tokens=config.template_reserve_tokens,
        annotations={"label": config.label, "session_id": config.session_id or "",
                     "config_fingerprint": config.fingerprint()})


def asset_metadata(session: ContainerSessionConfig, *, reader=None) -> dict[str, dict]:
    """GGUF facts per asset sha256 (template hash, block count); a missing chat template fails closed."""
    from .gguf import gguf_summary
    read = reader or gguf_summary
    facts = {}
    for asset in session.assets:
        summary = read(asset.host_path)
        if not summary.get("template_hash"):
            raise ValueError(f"{asset.host_path} declares no tokenizer.chat_template; the prompt template "
                             "identity cannot be established")
        facts[asset.sha256] = {"template_hash": summary["template_hash"], "block_count": summary.get("block_count"),
                               "architecture": summary.get("architecture"), "name": summary.get("name"),
                               "n_ctx_train": summary.get("n_ctx_train")}
    return facts


def check_context_range_against_headers(session: ContainerSessionConfig, facts: dict[str, dict]) -> None:
    """A declared ``context_n_ctx_train`` must not exceed what the weights themselves declare.

    The reachability rule is only as good as the training context it is checked against, so the GGUF headers
    read at ``tune`` get the last word: a session config claiming a larger training context than the weights
    declare (or weights that declare none) is refused before any candidate runs.
    """
    declared = session.search.context_n_ctx_train
    if declared is None:
        return
    for asset in session.assets:
        trained = (facts.get(asset.sha256) or {}).get("n_ctx_train")
        if type(trained) is not int:
            raise ValueError(f"{asset.host_path} declares no training context length, so the session's context "
                             f"range (ceiling {session.search.context_ceiling} usable input tokens) cannot be "
                             "checked against the weights")
        if declared > trained:
            raise ValueError(f"the session declares context_n_ctx_train {declared} but {asset.host_path} declares "
                             f"{trained}; the context range was validated against a training context the weights "
                             "do not have")


def build_candidates(session: ContainerSessionConfig, bundle, proposals, facts: dict[str, dict], *,
                     scorer_revision: str, environment_hash: str) -> list[tuple[ContainerRunConfig, ExperimentManifest]]:
    baseline = proposals[0]
    if baseline.family != "baseline":
        raise ValueError("the first proposal must be the baseline")
    pairs = []
    for proposal in proposals:
        config = run_config_for(session, bundle, proposal, baseline=baseline)
        meta = facts[config.model.sha256]
        manifest = to_manifest(config, template_hash=meta["template_hash"], scorer_revision=scorer_revision,
                               environment_hash=environment_hash, block_count=meta.get("block_count"))
        if proposal.family == "combination":
            manifest = manifest.model_copy(update={"annotations": {
                **manifest.annotations, "treatment_family": "combination"}})
        pairs.append((config, manifest))
    return pairs


# ---- durable files ------------------------------------------------------------------------------------------

def write_exclusive_json(path: str | Path, payload: dict | list) -> Path:
    """Exclusive creation; a crash leaves no truncated file at the path."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    target = Path(path)
    handle = target.open("x", encoding="utf-8")
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def write_atomic_text(path: str | Path, text: str) -> Path:
    """Replace-in-place with fsync for files that resume legitimately rewrites (summary, reports)."""
    target = Path(path)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def write_atomic_json(path: str | Path, payload: dict) -> Path:
    return write_atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def read_ledger(output: str | Path) -> dict:
    ledger = json.loads(_read_bounded(Path(output) / LEDGER_NAME))
    required = {"schema_version", "session_id", "campaign_identity", "started_utc", "deadline_utc", "wall_seconds",
                "policy_hash", "proposals", "manifests", "template_hashes", "scorer_revision", "environment_hash",
                "image_bundle", "image_ids", "config_fingerprints", "session_config_sha256"}
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1 or required - ledger.keys():
        raise ValueError("session.json is not a schema 1 session ledger")
    return ledger


def _epoch(utc: str) -> float:
    return datetime.fromisoformat(utc).timestamp()


def _authorize(lock: SessionLock, operations: tuple[str, ...] = NVIDIA_OPERATIONS) -> None:
    for operation in operations:
        lock.check(operation, RunMode.LIVE)


def required_operations_for(config: ContainerRunConfig) -> tuple[str, ...]:
    """The ``runtime-policy.json`` operations running ``config`` needs, from the runtime registry.

    One source of truth (``runtime.required_operations``): the NVIDIA tuple is exactly ``NVIDIA_OPERATIONS``;
    a metal-native config needs ``native`` instead of ``container``, plus ``container`` again only when it
    carries a coding broker (whose sandbox workers are containers). Imported lazily so importing this module
    stays inert.
    """
    from .runtime import required_operations
    return tuple(required_operations(config))


def authorize_config(lock: SessionLock, config: ContainerRunConfig) -> None:
    """Refuse a single candidate (``sample``, ``candidate``) this policy does not allow for its runtime."""
    _authorize(lock, required_operations_for(config))


def authorize_session(lock: SessionLock, session: ContainerSessionConfig, bundle=None) -> None:
    """Refuse, before anything is written or started, a session this policy does not allow.

    Checked for the session's base candidate and, when a bundle is given, for the base as that bundle will serve
    it: a native bundle laid over a session turns its candidates into host processes, and the permission that
    matters is the one for what will actually run, not for what the session file was first written for.
    """
    _authorize(lock, required_operations_for(session.base))
    if bundle is not None:
        _authorize(lock, required_operations_for(bundled_base(session, bundle)))


# ---- orchestration ------------------------------------------------------------------------------------------

def _tune_bundle(bundle, bundle_path) -> tuple[Any, str | None]:
    """The effective bundle for tune: read from ``bundle_path`` (recorded in the ledger) or given in memory.

    Either kind of bundle (``read_bundle``); the ledger keeps its ``image_bundle`` key for both."""
    if bundle_path is None:
        return bundle, None
    path = Path(bundle_path).resolve()
    loaded = read_bundle(path)
    if bundle is not None and image_ids(bundle) != image_ids(loaded):
        raise ValueError(f"the given bundle and {path} pin different images or native servers "
                         f"({image_ids(bundle)} != {image_ids(loaded)})")
    return loaded, str(path)


def _pin_session_config(root: Path, session: ContainerSessionConfig) -> str:
    """Write ``session-config.json`` or accept an identical pre-written one; any other content is refused."""
    target = root / SESSION_CONFIG_NAME
    if target.exists():
        try:
            existing = json.loads(_read_bounded(target))
        except ValueError as exc:
            raise ValueError(f"{target} exists and is not readable JSON: {exc}") from exc
        if existing != session.model_dump(mode="json"):
            raise ValueError(f"{target} exists and describes a different session; use another --output")
    else:
        write_exclusive_json(target, session.model_dump(mode="json"))
    return _file_sha256(target)


def _refuse_while_gpu_locked() -> None:
    """A visible GPU lock stops a NEW session before it writes anything.

    The campaign takes the lock exclusively later, which stays the real guard; this early look exists because a
    launch that died on the lock used to leave a ledger behind, and the next attempt was then told to `resume`
    a session that had never started.
    """
    from ..locks import describe_lock
    from .lease import default_lease_path
    path = default_lease_path()
    if path.exists():
        raise FileExistsError(f"the GPU is in use: {describe_lock(path)}")


def tune(session: ContainerSessionConfig, output: str | Path, *, runner, session_lock: SessionLock,
         bundle=None, bundle_path: str | Path | None = None, clock=time.monotonic, wall=time.time,
         started_wall: float | None = None, started_clock: float | None = None, metadata_reader=None,
         proposals=None) -> dict:
    """Start a new session in ``output``: exclusive ledger, campaign, reports. Setup time is charged.

    ``bundle_path`` (the effective ``--image-bundle``, or the native bundle of a metal-native session) is read here
    and pinned in the ledger with what it pins so ``resume`` rebuilds the identical candidates; an in-memory
    ``bundle`` alone pins the IDs but no path.
    ``started_wall``/``started_clock`` let the caller charge work done before this call (GGUF hashing and
    session derivation in ``main_tune``) to the session wall and the durable campaign checkpoint.
    """
    from ..provenance import environment_record
    from ..registry import builtin_registry
    from .proposals import deterministic_schedule, load_proposal_file
    _authorize(session_lock, required_operations_for(session.base))
    _refuse_while_gpu_locked()
    started_wall = wall() if started_wall is None else started_wall
    started_clock = clock() if started_clock is None else started_clock
    bundle, bundle_location = _tune_bundle(bundle, bundle_path)
    authorize_session(session_lock, session, bundle)  # what the bundle will actually run, before anything is written
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / LEDGER_NAME).exists():
        raise ValueError(f"{root / LEDGER_NAME} exists; use resume for a started session")
    config_sha256 = _pin_session_config(root, session)
    facts = asset_metadata(session, reader=metadata_reader)  # ~1 s per GGUF header, charged to the session
    check_context_range_against_headers(session, facts)
    policy = policy_for(session)
    if proposals is None:
        proposals = (load_proposal_file(session.proposal_file, session) if session.proposal_mode == "file"
                     else deterministic_schedule(session))
    scorer, environment = builtin_registry().digest(), environment_record()["sha256"]
    pairs = build_candidates(session, bundle, proposals, facts, scorer_revision=scorer, environment_hash=environment)
    from .campaign import campaign_identity
    holdout = holdout_selections(session)
    manifests = [manifest for _, manifest in pairs]
    identity = campaign_identity(manifests, policy, holdout)
    started_utc = datetime.fromtimestamp(started_wall, timezone.utc).isoformat()
    deadline_utc = (datetime.fromtimestamp(started_wall, timezone.utc)
                    + timedelta(seconds=session.budgets.wall_seconds)).isoformat()
    ledger = {"schema_version": 1, "session_id": session.session_id, "campaign_identity": identity,
              "started_utc": started_utc, "deadline_utc": deadline_utc, "wall_seconds": session.budgets.wall_seconds,
              "policy_hash": _policy_hash(policy), "policy": policy.model_dump(mode="json"),
              "proposals": [item.model_dump(mode="json") for item in proposals],
              "manifests": [item.fingerprint() for item in manifests],
              "labels": [config.label for config, _ in pairs],
              "template_hashes": {sha: meta["template_hash"] for sha, meta in facts.items()},
              "block_counts": {sha: meta.get("block_count") for sha, meta in facts.items()},
              "scorer_revision": scorer, "environment_hash": environment,
              "image_bundle": bundle_location, "image_ids": image_ids(bundle),
              "config_fingerprints": [config.fingerprint() for config, _ in pairs],
              "session_config_sha256": config_sha256,
              "holdout": [item.model_dump(mode="json") for item in holdout]}
    write_exclusive_json(root / LEDGER_NAME, ledger)
    setup_seconds = max(0.0, clock() - started_clock)
    return _run(session, root, ledger, pairs, policy, runner=runner, session_lock=session_lock, clock=clock,
                wall=wall, setup_seconds=setup_seconds, resume=False)


def _resume_bundle(ledger: dict, bundle):
    """The bundle tune pinned: the ledger path, or an explicit bundle carrying the same pins.

    For a native session the pins are the executable and library digests (``image_ids``), so a rebuilt or
    upgraded llama-server is refused exactly like a different inference image, and a moved one is accepted."""
    pinned = ledger["image_ids"]
    if pinned is None:
        if bundle is not None:
            raise ValueError("this session ran without an image bundle; an image bundle on resume would change "
                             "the inference image, start a new session instead")
        return None
    native = "native_server" in pinned
    if bundle is None:
        path = ledger["image_bundle"]
        if not path or not Path(path).is_file():
            if native:  # `resume --image-bundle` reads either kind (`read_bundle`); resume has no --native-bundle
                raise ValueError(f"the native bundle recorded at tune ({path!r}) is missing; pass the moved "
                                 f"native-bundle.json as --image-bundle with the same pins {pinned} to continue")
            raise ValueError(f"the image bundle recorded at tune ({path!r}) is missing; pass --image-bundle with "
                             f"the same image IDs {pinned} to continue")
        bundle = read_bundle(path)
    if image_ids(bundle) != pinned:
        if native:
            raise ValueError(f"bundle pins {image_ids(bundle)} differ from the ledger's native server pins {pinned}; "
                             "resume refuses to change the llama-server, start a new session instead")
        raise ValueError(f"image bundle image IDs {image_ids(bundle)} differ from the ledger {pinned}; resume "
                         "refuses to change images, start a new session instead")
    return bundle


def resume(output: str | Path, *, runner, session_lock: SessionLock, bundle=None, clock=time.monotonic,
           wall=time.time) -> dict:
    """Continue a started session under its persisted deadline; any drift fails closed before a candidate runs.

    Checked against the ledger: ``session-config.json`` bytes (sha256), policy hash, the proposal schedule, the
    image bundle's image IDs, every rebuilt ``ContainerRunConfig`` fingerprint and every manifest fingerprint
    (which carries ``environment_hash``, so a harness source change between tune and resume also refuses).
    """
    from ..provenance import environment_record
    from ..registry import builtin_registry
    from .proposals import Proposal, deterministic_schedule
    root = Path(output).resolve()
    ledger = read_ledger(root)
    config_path = root / SESSION_CONFIG_NAME
    session = read_session_config(config_path)
    _authorize(session_lock, required_operations_for(session.base))  # the runtime is the session's; read first
    if session.session_id != ledger["session_id"]:
        raise ValueError("session-config.json does not belong to session.json")
    policy = policy_for(session)
    if _policy_hash(policy) != ledger["policy_hash"]:
        raise ValueError("the session policy changed since tune; resume refuses to mix policies")
    if _file_sha256(config_path) != ledger["session_config_sha256"]:
        raise ValueError(f"{SESSION_CONFIG_NAME} was edited since tune (sha256 differs from the ledger); restore "
                         "it or start a new session")
    proposals = [Proposal.model_validate_json(canonical_json(row)) for row in ledger["proposals"]]
    if session.proposal_mode == "deterministic" and proposals != deterministic_schedule(session):
        raise ValueError("the deterministic schedule no longer matches the ledger")
    bundle = _resume_bundle(ledger, bundle)
    authorize_session(session_lock, session, bundle)
    facts = {sha: {"template_hash": value, "block_count": (ledger.get("block_counts") or {}).get(sha)}
             for sha, value in ledger["template_hashes"].items()}
    scorer, environment = builtin_registry().digest(), environment_record()["sha256"]
    pairs = build_candidates(session, bundle, proposals, facts, scorer_revision=scorer, environment_hash=environment)
    if [config.fingerprint() for config, _ in pairs] != ledger["config_fingerprints"]:
        raise ValueError("rebuilt candidate configurations differ from the ledger (config_fingerprints): the "
                         "session config or image bundle changed since tune; start a new session")
    if [manifest.fingerprint() for _, manifest in pairs] != ledger["manifests"]:
        raise ValueError("rebuilt manifests differ from the ledger: the harness source, image bundle or registry "
                         "changed since tune (environment_hash is part of every live manifest); start a new session")
    return _run(session, root, ledger, pairs, policy, runner=runner, session_lock=session_lock, clock=clock,
                wall=wall, setup_seconds=0.0, resume=True)


def _policy_hash(policy: CampaignPolicy) -> str:
    return hashlib.sha256(canonical_json(policy.model_dump(mode="json")).encode()).hexdigest()


def _run(session, root: Path, ledger: dict, pairs, policy: CampaignPolicy, *, runner, session_lock, clock, wall,
         setup_seconds: float, resume: bool) -> dict:
    from ..controller import run_campaign
    from ..reports import write_reports
    from ..store import Store
    from .campaign import ContainerExecutor
    from .proposals import full_schedule
    from .session_report import build_session_report, write_session_report
    holdout = holdout_selections(session)
    manifests = [manifest for _, manifest in pairs]
    deadline_epoch = _epoch(ledger["deadline_utc"])
    identity = ledger["campaign_identity"]
    # The count cap cuts the deterministic schedule before the controller sees it (proposals.deterministic_schedule).
    truncated = session.proposal_mode == "deterministic" and len(full_schedule(session)) > len(pairs)
    with Store(root, max_artifact_bytes=policy.budgets.max_artifact_bytes) as store:
        executor = ContainerExecutor(session, None, runner=runner, output_root=root, deadline_epoch=deadline_epoch,
                                     template_hashes=dict(ledger["template_hashes"]), clock=clock, wall=wall)
        for config, manifest in pairs:
            executor.register(config, manifest)
        expired = wall() >= deadline_epoch
        if expired:
            from ..analysis import analyze_campaign
            identifiers = [row["id"] for row in store.attempts_in_insertion_order()]
            campaign = {"campaign_id": identity, "policy": policy.model_dump(mode="json"), "results": [],
                        "skipped": [], "frontier": [], "winner": None, "automatic_deployment_performed": False,
                        "all_attempt_ids": identifiers, "remaining_wall_seconds": 0.0,
                        "selection_note": "Deadline passed before resume; reports recomputed from stored evidence."}
            campaign = analyze_campaign(store, campaign, policy) if identifiers else campaign
        else:
            if not resume and setup_seconds > 0:
                # Derivation and GGUF hashing happened inside the session wall; reserve it durably so the
                # controller's tracker and the executor's wall-clock deadline agree.
                store.checkpoint_campaign(identity, setup_seconds, 0)
            campaign = run_campaign(store, manifests, policy, executor=executor, session_lock=session_lock,
                                    resume=resume or setup_seconds > 0, holdout_selections=holdout)
            if campaign["campaign_id"] != identity:
                raise ValueError("controller campaign identity differs from the ledger")
        _restore_failure_fields(store, campaign)
        stop = _stop_reason(campaign, expired=expired, deadline_hits=executor.deadline_hits, policy=policy,
                            truncated=truncated, admitted=store.campaign_state(identity)["candidates"])
        reports = write_reports(campaign, root / "reports")
        report = build_session_report(store, campaign, session, ledger, output_root=root, stop_reason=stop,
                                      remaining_seconds=max(0.0, deadline_epoch - wall()),
                                      campaign_reports=reports)
        report_files = write_session_report(report, root / "reports")
    summary = {"session_id": session.session_id, "campaign_identity": identity, "stop_reason": stop,
               "deadline_utc": ledger["deadline_utc"], "remaining_seconds": report["budget"]["remaining_seconds"],
               "attempts": len(report["rows"]), "frontier": campaign.get("frontier", []),
               "winner": campaign.get("winner"), "best_presets": sorted(campaign.get("best_presets", {})),
               "abort_campaign": stop == "abort",  # a deadline stop is "stopped early", not cleanup uncertainty
               "reports": {**reports, **report_files}, "recommendation": report["recommendation"]}
    write_atomic_json(root / SUMMARY_NAME, summary)
    return {"campaign": campaign, "summary": summary, "report": report, "stop_reason": stop}


def _restore_failure_fields(store, campaign: dict) -> None:
    """analyze_campaign rebuilds rows from stored evidence and drops the controller's error/abort fields;
    put them back from the Store's error events so the outcome and stop reason tell what happened."""
    from .campaign import ABORTING_ERROR_TYPES
    for row in campaign.get("results", []):
        if row.get("status") == "completed" or "error" in row or not row.get("attempt_id"):
            continue
        errors = [json.loads(event["payload"]) if isinstance(event["payload"], str) else event["payload"]
                  for event in store.results(row["attempt_id"])["events"] if event["kind"] == "error"]
        if not errors:
            continue
        last = errors[-1]
        row["error"] = str(last.get("message", ""))
        row["error_type"] = last.get("type")
        row["abort_campaign"] = last.get("type") in ABORTING_ERROR_TYPES


def _stop_reason(campaign: dict, *, expired: bool, deadline_hits: int, policy: CampaignPolicy, truncated: bool,
                 admitted: int) -> str:
    """abort > deadline > the controller refused a candidate (count cap reached: ``candidates``, else time:
    ``budget``) > the schedule itself was cut by the count cap (``candidates``) > ``complete``.
    ``admitted`` is the controller's durable admit count (retries consume cap slots too)."""
    results = campaign.get("results", [])
    if any(row.get("abort_campaign") is True and row.get("error_type") != "SessionDeadlineExceeded"
           for row in results):
        return "abort"  # cleanup uncertain or another runner abort: the session must not continue
    if expired or deadline_hits:
        return "deadline"
    if any(row.get("reason") == "search_budget_reserved_or_exhausted" for row in campaign.get("skipped", [])):
        return "candidates" if admitted >= policy.budgets.max_candidates else "budget"
    return "candidates" if truncated else "complete"



# ---- CLI -----------------------------------------------------------------------------------------------------

def add_session_commands(subparsers) -> None:
    tune_parser = subparsers.add_parser("tune", help="Run a persistent tuning session (needs runtime-policy.json)")
    source = tune_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="a ContainerSessionConfig JSON file")
    source.add_argument("--model", help="a GGUF file or a directory of GGUFs; the session is derived from it")
    tune_parser.add_argument("--output", required=True)
    tune_parser.add_argument("--budget-seconds", type=int, help="override the session wall budget")
    tune_parser.add_argument("--image-bundle", help="image-bundle.json from `llmbench prepare`")
    tune_parser.add_argument("--search", default="default", help="search preset for --model (only: default)")
    tune_parser.add_argument("--context-floor", type=int, metavar="N", default=None,
                             help="smallest USABLE INPUT tokens to search (>=512); --model only. Output capacity "
                                  "is reserved on top of it. Missing while --context-ceiling is given: 4096, or "
                                  "the ceiling when that is smaller.")
    tune_parser.add_argument("--context-ceiling", type=int, metavar="N", default=None,
                             help="largest USABLE INPUT tokens to search (>=512); --model only. Missing while "
                                  "--context-floor is given: the largest input the model's training context holds "
                                  "once the template reserve and the output capacity are subtracted.")
    tune_parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG,
                             help="candidate config used as the label template for --model")
    resume_parser = subparsers.add_parser("resume", help="Continue a started session under its persisted deadline")
    resume_parser.add_argument("--output", required=True)
    resume_parser.add_argument("--image-bundle", help="only when the bundle tune used has moved; its image IDs "
                                                      "(or, for a metal-native session's native-bundle.json, its "
                                                      "executable and library digests) must equal the ones pinned "
                                                      "in session.json")
    for command in (tune_parser, resume_parser):
        command.add_argument("--policy", default="runtime-policy.json")
        command.add_argument("--capabilities-dir", default=DEFAULT_CAPABILITIES)


def _default_runner_factory(args, policy: CampaignPolicy):
    """The runner for whichever runtime each candidate names. For an NVIDIA candidate the dispatcher builds
    exactly ``ContainerRunner(capabilities_dir=..., policy_path=..., policy=...)``, lazily, as this factory did."""
    from .runtime import DispatchRunner
    return DispatchRunner(capabilities_dir=args.capabilities_dir, policy_path=args.policy, policy=policy)


def _requested_runtime(args) -> tuple[str | None, str | None]:
    """``(runtime, native_bundle)`` as asked on the command line, refusing contradictions; never a guess.

    ``--runtime``/``--native-bundle`` are read with ``getattr`` because the top-level CLI adds them to this
    parser (``cli.py``), not ``add_session_commands``; a parser without them asks for the NVIDIA session this
    command always ran. Both flags are explicit, as ``cli._run_tune`` also requires: a native bundle never switches
    the runtime by itself, and ``--runtime metal-native`` never runs without the pinned server it describes.
    """
    runtime, native = getattr(args, "runtime", None), getattr(args, "native_bundle", None)
    if runtime is not None and runtime not in RUNTIMES:
        raise ValueError(f"unknown runtime {runtime!r}; expected one of {', '.join(RUNTIMES)}")
    if native is not None and runtime != "metal-native":
        raise ValueError("--native-bundle pins a metal-native llama-server; add --runtime metal-native (the "
                         "nvidia-container runtime takes --image-bundle)")
    if runtime == "metal-native" and native is None:
        raise ValueError("--runtime metal-native needs --native-bundle (native-bundle.json from `llmbench prepare "
                         "--runtime metal-native`): it is what pins the llama-server executable that will run")
    return runtime, native


def _native_bundle_at(path: str | Path, flag: str) -> NativeBundle:
    bundle = read_bundle(path)
    if not isinstance(bundle, NativeBundle):
        raise ValueError(f"{flag} {path} is an image bundle (NVIDIA images), not a native-bundle.json")
    return bundle


def _exit_code(outcome: dict) -> int:
    if outcome["summary"]["abort_campaign"]:
        return 4
    return 0 if outcome["stop_reason"] in {"complete", "candidates"} else 3


def _print(outcome: dict) -> None:
    print(json.dumps(outcome["summary"], indent=2, ensure_ascii=False))


def main_tune(args, *, runner_factory=None, clock=time.monotonic, wall=time.time) -> int:
    """Exit codes: 0 finished, 2 invalid input or policy, 3 stopped early or interrupted, 4 campaign aborted."""
    started, started_clock = wall(), clock()  # derivation and GGUF hashing below are charged to the session
    try:
        lock = SessionLock.read(args.policy)
        floor, ceiling = getattr(args, "context_floor", None), getattr(args, "context_ceiling", None)
        if args.config:
            if floor is not None or ceiling is not None:
                raise ValueError("--context-floor/--context-ceiling derive a search space from --model; a --config "
                                 "session already fixes its own ctx_tiers and context range")
            if getattr(args, "native_bundle", None) is not None:
                raise ValueError("--native-bundle derives a metal-native session from --model; a --config session "
                                 "already pins its runtime in `base` and its bundle in `image_bundle` (a moved "
                                 "bundle goes in --image-bundle)")
            session = read_session_config(args.config)  # a bounded read: the runtime is the session's own
            runtime = getattr(args, "runtime", None)
            if runtime is not None and runtime != session.base.runtime:
                raise ValueError(f"--runtime {runtime} differs from the {session.base.runtime} runtime this session "
                                 "config's base candidate names; a --config session is run as it is written")
            _authorize(lock, required_operations_for(session.base))
        else:
            from .derive import derive_session_config
            from .config import read_run_config
            native_path = _requested_runtime(args)[1]  # metal-native exactly when a native bundle is given
            if native_path is None:
                _authorize(lock)  # the NVIDIA session: refused before any GGUF is hashed, as it always was
            if args.search != "default":
                raise ValueError(f"unknown search preset {args.search!r}; only 'default' exists")
            if native_path is None:
                bundle = read_bundle(args.image_bundle) if args.image_bundle else None
                if isinstance(bundle, NativeBundle):
                    raise ValueError(f"--image-bundle {args.image_bundle} is a native bundle; pass it as "
                                     "--native-bundle with --runtime metal-native")
                session = derive_session_config(args.model, base=read_run_config(args.base_config),
                                                budget_seconds=args.budget_seconds or 14400,
                                                image_bundle=args.image_bundle,
                                                inference_image=bundle.inference if bundle else None,
                                                context_floor=floor, context_ceiling=ceiling)
            else:
                from .derive import planned_operations
                if args.image_bundle:
                    raise ValueError("--image-bundle pins NVIDIA container images; a metal-native session is pinned "
                                     "by --native-bundle alone")
                native = _native_bundle_at(native_path, "--native-bundle")
                base = read_run_config(args.base_config)
                # Refused before any GGUF is hashed, for the base as the derivation will serve it: the native
                # server, and the template's broker only when the bundle's sandbox verdict keeps it.
                _authorize(lock, planned_operations(base, native_bundle=native))
                session = derive_session_config(args.model, base=base, budget_seconds=args.budget_seconds or 14400,
                                                image_bundle=native_path, native_bundle=native,
                                                context_floor=floor, context_ceiling=ceiling)
        if args.budget_seconds:
            raw = session.model_dump(mode="json")
            raw["budgets"]["wall_seconds"] = args.budget_seconds
            session = ContainerSessionConfig.model_validate_json(canonical_json(raw))
        # The effective bundle (an image bundle, or a metal-native session's native bundle); tune pins it.
        bundle_path = args.image_bundle or session.image_bundle
        policy = policy_for(session)
        runner = (runner_factory or _default_runner_factory)(args, policy)
        outcome = tune(session, args.output, runner=runner, session_lock=lock, bundle_path=bundle_path, clock=clock,
                       wall=wall, started_wall=started, started_clock=started_clock)
    except (ValidationError, ValueError, KeyError, OSError, OperationForbidden) as exc:
        print(f"llmbench tune: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"llmbench tune: interrupted; continue with: llmbench resume --output {args.output}", file=sys.stderr)
        return 3
    _print(outcome)
    return _exit_code(outcome)


def main_resume(args, *, runner_factory=None, clock=time.monotonic, wall=time.time) -> int:
    try:
        lock = SessionLock.read(args.policy)
        session = read_session_config(Path(args.output) / SESSION_CONFIG_NAME)  # the runtime is the session's
        _authorize(lock, required_operations_for(session.base))
        runtime = getattr(args, "runtime", None)
        if runtime is not None and runtime != session.base.runtime:
            raise ValueError(f"--runtime {runtime} differs from the {session.base.runtime} runtime this session was "
                             "tuned with; resume continues the session as it was started")
        image_path, native_path = args.image_bundle, getattr(args, "native_bundle", None)
        if image_path and native_path:
            raise ValueError("pass a moved bundle once: --image-bundle or --native-bundle, not both")
        bundle = (_native_bundle_at(native_path, "--native-bundle") if native_path
                  else read_bundle(image_path) if image_path else None)
        policy = policy_for(session)
        runner = (runner_factory or _default_runner_factory)(args, policy)
        outcome = resume(args.output, runner=runner, session_lock=lock, bundle=bundle, clock=clock, wall=wall)
    except (ValidationError, ValueError, KeyError, OSError, OperationForbidden) as exc:
        print(f"llmbench resume: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"llmbench resume: interrupted; continue with: llmbench resume --output {args.output}", file=sys.stderr)
        return 3
    _print(outcome)
    return _exit_code(outcome)


def main(argv=None, *, runner_factory=None) -> int:
    """Standalone parser for the session commands alone (tests drive ``main_tune``/``main_resume`` through it)."""
    root = argparse.ArgumentParser(prog="llmbench")
    add_session_commands(root.add_subparsers(dest="command", required=True))
    args = root.parse_args(argv)
    handler = main_tune if args.command == "tune" else main_resume
    return handler(args, runner_factory=runner_factory)


__all__ = ["SessionBudgets", "SearchSpace", "HoldoutPlan", "ContainerSessionConfig", "read_session_config",
           "read_image_bundle", "read_bundle", "image_ids", "native_overlay", "bundled_base",
           "required_operations_for", "authorize_config", "authorize_session", "policy_for", "holdout_selections",
           "run_config_for",
           "to_manifest", "asset_metadata", "check_context_range_against_headers", "build_candidates",
           "output_reserve_tokens", "usable_input_tokens", "tune", "resume", "add_session_commands", "main_tune",
           "main_resume", "main"]


if __name__ == "__main__":  # python -m llmbench.containers.session tune|resume ... (until cli.py wires them)
    raise SystemExit(main())
