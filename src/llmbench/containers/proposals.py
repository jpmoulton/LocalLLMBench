"""Typed, bounded candidate proposals over a frozen search space. No LLM proposes anything here.

A proposal names one causal family and the axis values it changes relative to the session baseline.
``deterministic_schedule`` is the built-in proposer; ``load_proposal_file`` is the typed hook for an
external proposer. ``apply`` turns a proposal into a fully re-validated ``ContainerRunConfig``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from ..config import StrictModel, canonical_json, freeze_json
from .config import NAME, ContainerRunConfig
from .session import output_reserve_tokens, usable_input_tokens

Family = Literal["baseline", "kv", "weights", "context", "performance", "reasoning", "speculation", "offload",
                 "kv-placement"]
AXES = ("quantization", "kv_pair", "ctx_tier", "spec_type", "reasoning", "batch_size", "gpu_layers", "kv_offload")
FAMILY_AXES: dict[str, frozenset[str]] = {
    "baseline": frozenset(AXES), "kv": frozenset({"kv_pair"}), "weights": frozenset({"quantization"}),
    "context": frozenset({"ctx_tier"}), "performance": frozenset({"batch_size"}),
    "reasoning": frozenset({"reasoning"}), "speculation": frozenset({"spec_type"}),
    # CPU/RAM offload is a first-class axis. "offload" moves model layers off the GPU into system RAM
    # (``--n-gpu-layers``); "kv-placement" puts the KV cache on the GPU or in system RAM (``--kv-offload`` /
    # ``--no-kv-offload``).
    "offload": frozenset({"gpu_layers"}), "kv-placement": frozenset({"kv_offload"}),
}
# ``context`` stays LAST and ascending: ``deterministic_schedule`` cuts the schedule at ``max_candidates`` and
# ``session._coherent_context_schedule`` depends on the ceiling tier being the last thing a cap can
# take away. The offload families are therefore inserted BEFORE it, never after.
SWEEP_ORDER = ("kv", "weights", "speculation", "reasoning", "performance", "offload", "kv-placement", "context")
MAX_PROPOSAL_FILE_BYTES = 1_048_576


class Proposal(StrictModel):
    family: Family
    changes: dict[str, Any]
    note: str = ""

    @model_validator(mode="after")
    def bounded(self) -> "Proposal":
        keys = set(self.changes)
        if not keys or keys - FAMILY_AXES[self.family]:
            raise ValueError(f"{self.family} proposal may only change {sorted(FAMILY_AXES[self.family])}")
        if self.family == "baseline" and keys != set(AXES):
            raise ValueError("a baseline proposal must fix every axis")
        normalized = {}
        for key, value in self.changes.items():
            if key == "kv_pair":
                if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(
                        type(item) is str for item in value):
                    raise ValueError("kv_pair must be a pair of cache type names")
                value = tuple(value)
            elif key in {"ctx_tier", "batch_size"}:
                if type(value) is not int:
                    raise ValueError(f"{key} must be an integer")
            elif key == "gpu_layers":  # a layer count, or "all"; ``bool`` is not an acceptable layer count
                if type(value) is not int and value != "all":
                    raise ValueError("gpu_layers must be an integer layer count or \"all\"")
            elif key == "kv_offload":  # True: KV cache on the GPU; False: KV cache in system RAM
                if type(value) is not bool:
                    raise ValueError("kv_offload must be a boolean")
            elif type(value) is not str or not value:
                raise ValueError(f"{key} must be a nonempty string")
            normalized[key] = value
        object.__setattr__(self, "changes", freeze_json(normalized))
        if len(self.note) > 500 or any(ord(char) < 32 for char in self.note):
            raise ValueError("note must be a short single line")
        return self

    def label(self) -> str:
        if self.family == "baseline":
            return "baseline"
        (axis, value), = self.changes.items()
        if axis == "kv_offload":  # the label names the PLACEMENT, not the flag: "kv-cache-ram" reads as it runs
            text = "gpu" if value else "ram"
        else:
            text = "-".join(value) if isinstance(value, tuple) else str(value)
        short = {"kv_pair": "kv", "quantization": "weights", "ctx_tier": "ctx", "spec_type": "spec",
                 "reasoning": "reasoning", "batch_size": "batch", "gpu_layers": "offload",
                 "kv_offload": "kv-cache"}[axis]
        slug = re.sub(r"[^a-z0-9._-]+", "-", f"{short}-{text}".lower()).strip("-.")[:40]
        if not re.fullmatch(NAME, slug):
            raise ValueError(f"proposal label is not a safe name: {slug!r}")
        return slug


def _space_members(session) -> dict[str, tuple]:
    search = session.search
    return {"quantization": search.quantizations, "kv_pair": search.kv_pairs, "ctx_tier": search.ctx_tiers,
            "spec_type": search.spec_types, "reasoning": search.reasoning, "batch_size": search.batch_sizes,
            "gpu_layers": search.gpu_layers, "kv_offload": search.kv_offload}


def check_membership(session, proposal: Proposal) -> None:
    members = _space_members(session)
    for axis, value in proposal.changes.items():
        if value not in members[axis]:
            raise ValueError(f"{axis} value {value!r} is outside the frozen search space")


def baseline_proposal(session) -> Proposal:
    search = session.search
    spec = "none" if "none" in search.spec_types else search.spec_types[0]
    return Proposal(family="baseline", changes={
        "quantization": search.quantizations[0], "kv_pair": search.kv_pairs[0], "ctx_tier": search.ctx_tiers[0],
        "spec_type": spec, "reasoning": search.reasoning[0], "batch_size": search.batch_sizes[0],
        "gpu_layers": search.gpu_layers[0], "kv_offload": search.kv_offload[0]},
        note="deterministic baseline: first member of every axis, no speculation")


def context_fill_for(session, base: ContainerRunConfig, tier: int) -> int:
    """``requested_input_tokens`` for a candidate that moves to context tier ``tier``.

    With an explicit context range the tiers are allocations derived from USABLE INPUT bounds, so the
    candidate is filled to that tier's input target -- the allocation minus the template reserve and the output
    capacity, never above ``context_ceiling``. "Supports N tokens" then means a candidate really ran with about
    N tokens of input and the requested output still reserved above it.

    Without a range the legacy behaviour is unchanged: the base fill ratio is kept, so a larger tier tests a
    proportionally larger filled prompt. ``ContainerRunConfig``'s capacity rule re-checks both paths.
    """
    ceiling = getattr(session.search, "context_ceiling", None)
    if ceiling is None:
        return base.requested_input_tokens * tier // base.engine.ctx_size
    return usable_input_tokens(tier, template_reserve_tokens=base.template_reserve_tokens,
                               output_tokens=output_reserve_tokens(base), ceiling=ceiling)


PLANNING_PREFILL_TOKENS_PER_SECOND = 1000.0
"""Deliberately below what was measured (1080-1280 tok/s at 262144 on a 27B model), so the plan errs long."""
LONG_REQUEST_SECONDS = 120.0
"""A filled request slower than this makes the default workload unaffordable; below it nothing changes."""
LONG_RETRIEVAL_SECONDS = 900.0
"""How much of a long candidate may go to filled retrieval prompts."""
NIAH_PRIORITY = ("multi", "single-late", "single-early", "single-middle", "missing", "tool-multi")
"""Which variants survive thinning: `multi` plants needles at 5%, 50% and 95% of ONE prompt, so it buys the most
evidence per filled request; the late needle is the classic long-context failure."""
MAX_CANDIDATE_WALL_SECONDS = 14400


def filled_request_seconds(input_tokens: int) -> float:
    """Planning cost of one request filled to ``input_tokens``: prefill at the planning rate plus fixed overhead."""
    return input_tokens / PLANNING_PREFILL_TOKENS_PER_SECOND + 60.0


def size_long_context_work(raw: dict) -> None:
    """Fit a context candidate's filled-prompt work into bounds it can actually finish in. In place.

    Every speed request and every NIAH variant of a context candidate is filled to the candidate's input target.
    At a few thousand tokens that is free; at a quarter of a million it is four minutes each, and the default set
    cannot finish inside the default evaluation bound - so the context was never verified. Above
    ``LONG_REQUEST_SECONDS`` per filled request this keeps the warm-up and at most two speed repetitions, keeps
    the NIAH variants that fit ``LONG_RETRIEVAL_SECONDS`` (always at least one, in ``NIAH_PRIORITY`` order), and
    raises the evaluation, request and candidate bounds to hold what is left. It never lowers a bound.
    """
    cost = filled_request_seconds(raw["requested_input_tokens"])
    if cost <= LONG_REQUEST_SECONDS:
        return
    speed, bounds = raw["speed"], raw.setdefault("bounds", {})
    speed["repetitions"] = min(speed.get("repetitions", 3), 2)
    keep = max(1, int(LONG_RETRIEVAL_SECONDS // cost))
    retrieval_requests = 0
    for selection in raw.get("benchmarks", []):
        if selection.get("benchmark_id") != "niah":
            continue
        ranked = sorted(selection["task_ids"], key=lambda name: (NIAH_PRIORITY.index(name)
                                                                 if name in NIAH_PRIORITY else len(NIAH_PRIORITY)))
        kept = set(ranked[:keep])
        selection["task_ids"] = [name for name in selection["task_ids"] if name in kept]  # original order
        retrieval_requests += len(selection["task_ids"])
    requests = speed.get("warmup_repetitions", 1) + speed["repetitions"] + retrieval_requests
    evaluation = int(1.3 * requests * cost) + 600  # 600 s for the short suites, which do not grow with context
    defaults = {"evaluation_seconds": 1200, "request_timeout_seconds": 300, "candidate_wall_seconds": 1800,
                "hash_seconds": 300, "startup_seconds": 300, "verify_seconds": 60, "cleanup_reserve_seconds": 60}
    current = {name: bounds.get(name, value) for name, value in defaults.items()}
    bounds["evaluation_seconds"] = max(current["evaluation_seconds"], evaluation)
    bounds["request_timeout_seconds"] = max(current["request_timeout_seconds"], int(2 * cost))
    wall = (bounds["evaluation_seconds"] + current["hash_seconds"] + current["startup_seconds"]
            + 2 * current["verify_seconds"] + current["cleanup_reserve_seconds"] + 120)
    bounds["candidate_wall_seconds"] = min(MAX_CANDIDATE_WALL_SECONDS, max(current["candidate_wall_seconds"], wall))


def apply(session, base: ContainerRunConfig, proposal: Proposal) -> ContainerRunConfig:
    """Re-validated copy of ``base`` with the proposal's axis values; the capacity rule is re-checked."""
    check_membership(session, proposal)
    raw = base.model_dump(mode="json")
    engine = raw["engine"]
    for axis, value in proposal.changes.items():
        if axis == "quantization":
            asset = next((item for item in session.assets if item.quantization == value), None)
            if asset is None:
                raise ValueError(f"no session asset carries quantization {value!r}")
            raw["assets"] = [asset.model_dump(mode="json")]
        elif axis == "kv_pair":
            engine["cache_type_k"], engine["cache_type_v"] = value
        elif axis == "ctx_tier":
            raw["requested_input_tokens"] = context_fill_for(session, base, value)
            engine["ctx_size"] = value
            size_long_context_work(raw)
        elif axis == "spec_type":
            engine["spec_type"] = value
        elif axis == "reasoning":
            engine["reasoning"] = value
        elif axis == "batch_size":
            engine["batch_size"] = value
        elif axis == "gpu_layers":  # layers above this count stay in system RAM (--n-gpu-layers)
            engine["n_gpu_layers"] = value
        elif axis == "kv_offload":  # False puts the KV cache in system RAM (--no-kv-offload)
            engine["kv_offload"] = value
    return ContainerRunConfig.model_validate_json(canonical_json(raw))


def deterministic_schedule(session) -> list[Proposal]:
    """The full schedule cut to ``budgets.max_candidates`` (the session's count cap)."""
    return full_schedule(session)[:session.budgets.max_candidates]


def full_schedule(session) -> list[Proposal]:
    """Baseline, then single-axis sweeps kv -> weights -> speculation -> reasoning -> batch -> offload ->
    kv placement -> context tiers, deduplicated by candidate fingerprint and not capped;
    ``deterministic_schedule`` applies the cap. Context stays last so a cap takes the ceiling tier away last."""
    base = baseline_proposal(session)
    baseline_config = apply(session, session.base, base)
    members = _space_members(session)
    axis_of = {"kv": "kv_pair", "weights": "quantization", "speculation": "spec_type", "reasoning": "reasoning",
               "performance": "batch_size", "offload": "gpu_layers", "kv-placement": "kv_offload",
               "context": "ctx_tier"}
    schedule, seen = [base], {baseline_config.fingerprint()}
    for family in SWEEP_ORDER:
        axis = axis_of[family]
        values = members[axis]
        if axis == "ctx_tier":
            values = tuple(sorted(values))
        for value in values:
            if value == base.changes[axis]:
                continue
            proposal = Proposal(family=family, changes={axis: value}, note=f"single-axis sweep of {axis}")
            fingerprint = apply(session, baseline_config, proposal).fingerprint()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            schedule.append(proposal)
    return schedule


def load_proposal_file(path: str | Path, session) -> list[Proposal]:
    """Strict JSON list of proposals: baseline first, members of the frozen space only, bounded count."""
    target = Path(path)
    if not target.is_file() or target.stat().st_size > MAX_PROPOSAL_FILE_BYTES:
        raise ValueError("proposal file must be a regular file of at most 1 MiB")
    try:
        rows = json.loads(target.read_text(encoding="utf-8"))
    except RecursionError as exc:
        raise ValueError("proposal file nests too deeply") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"proposal file is not UTF-8 JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("proposal file must be a nonempty JSON list")
    if len(rows) > session.budgets.max_candidates:
        raise ValueError(f"proposal file lists {len(rows)} candidates; the session allows "
                         f"{session.budgets.max_candidates}")
    proposals = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"proposal {index} is not an object")
        proposal = Proposal.model_validate_json(canonical_json(row))  # unknown fields/axes rejected
        check_membership(session, proposal)
        proposals.append(proposal)
    if proposals[0].family != "baseline" or any(item.family == "baseline" for item in proposals[1:]):
        raise ValueError("the first proposal must be the only baseline proposal")
    baseline_config = apply(session, session.base, proposals[0])
    seen = {baseline_config.fingerprint()}
    for proposal in proposals[1:]:
        fingerprint = apply(session, baseline_config, proposal).fingerprint()
        if fingerprint in seen:
            raise ValueError(f"proposal {proposal.label()} duplicates an earlier candidate")
        seen.add(fingerprint)
    return proposals
