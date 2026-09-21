"""Deterministic schedule, bounded apply and the typed proposal file. Pure; nothing runs."""

import json

import pytest
from pydantic import ValidationError

from llmbench.config import canonical_json
from llmbench.containers.config import ContainerRunConfig
from llmbench.containers.proposals import (Proposal, apply, baseline_proposal, context_fill_for,
                                           deterministic_schedule, load_proposal_file)
from test_containers_session import make_session


def test_deterministic_schedule_is_baseline_then_single_axis_sweeps_bounded_and_deduplicated(tmp_path):
    session = make_session(tmp_path, budgets={"wall_seconds": 14400, "max_candidates": 12})
    schedule = deterministic_schedule(session)
    assert [item.family for item in schedule] == ["baseline", "kv", "weights", "speculation", "reasoning", "context"]
    assert [item.label() for item in schedule] == ["baseline", "kv-q8_0-q8_0", "weights-q6_k", "spec-draft-mtp",
                                                   "reasoning-on", "ctx-32768"]
    assert schedule[0].changes == {"quantization": "Q4_K_M", "kv_pair": ("f16", "f16"), "ctx_tier": 8192,
                                   "spec_type": "none", "reasoning": "off", "batch_size": 2048,
                                   "gpu_layers": "all", "kv_offload": True}  # U17: the baseline fixes every axis
    assert schedule == deterministic_schedule(session)  # stable across calls
    bounded = deterministic_schedule(make_session(tmp_path / "b", budgets={"wall_seconds": 3600,
                                                                          "holdout_reserve_seconds": 900,
                                                                          "candidate_wall_seconds": 900,
                                                                          "max_candidates": 3}))
    assert [item.label() for item in bounded] == ["baseline", "kv-q8_0-q8_0", "weights-q6_k"]
    # Baseline prefers "none" even when it is not the first listed speculation type; the sweep skips it.
    reordered = make_session(tmp_path / "c", spec_types=("draft-mtp", "none"), budgets={"wall_seconds": 14400})
    labels = [item.label() for item in deterministic_schedule(reordered)]
    assert deterministic_schedule(reordered)[0].changes["spec_type"] == "none"
    assert labels.count("spec-draft-mtp") == 1 and "spec-none" not in labels
    # Fingerprints are all distinct: no candidate is measured twice.
    baseline = apply(session, session.base, schedule[0])
    fingerprints = {apply(session, baseline, item).fingerprint() for item in schedule[1:]} | {baseline.fingerprint()}
    assert len(fingerprints) == len(schedule)
    # Context tiers sweep in ascending order regardless of listing order.
    tiers = make_session(tmp_path / "d", ctx_tiers=(8192, 16384, 32768), budgets={"wall_seconds": 14400})
    assert [item.label() for item in deterministic_schedule(tiers) if item.family == "context"] == ["ctx-16384",
                                                                                                    "ctx-32768"]


def test_offload_and_kv_placement_are_single_axis_families_swept_before_context(tmp_path):
    """U17: the two offload families follow the single-axis pattern and are deduplicated by fingerprint."""
    session = make_session(tmp_path, quantizations=("Q4_K_M",), kv_pairs=(("f16", "f16"),), ctx_tiers=(8192,),
                           spec_types=("none",), reasoning=("off",), gpu_layers=("all", 24, 0),
                           kv_offload=(True, False), budgets={"wall_seconds": 14400, "max_candidates": 12})
    schedule = deterministic_schedule(session)
    assert [item.family for item in schedule] == ["baseline", "offload", "offload", "kv-placement"]
    assert [item.label() for item in schedule] == ["baseline", "offload-24", "offload-0", "kv-cache-ram"]
    assert "offload-all" not in [item.label() for item in schedule]  # the baseline member is never re-run
    assert all(len(item.changes) == 1 for item in schedule[1:])  # one axis per proposal, exactly as the others
    baseline = apply(session, session.base, schedule[0])
    assert (baseline.engine.n_gpu_layers, baseline.engine.kv_offload) == ("all", True)
    moved = {item.label(): apply(session, baseline, item) for item in schedule[1:]}
    assert moved["offload-24"].engine.n_gpu_layers == 24 and moved["offload-0"].engine.n_gpu_layers == 0
    assert moved["kv-cache-ram"].engine.kv_offload is False
    assert moved["offload-24"].engine.kv_offload is True  # a single-axis proposal moves nothing else
    assert moved["kv-cache-ram"].engine.n_gpu_layers == "all"
    fingerprints = {config.fingerprint() for config in moved.values()} | {baseline.fingerprint()}
    assert len(fingerprints) == len(schedule)  # every candidate is a distinct configuration
    # A family never reaches another family's axis, and each value is typed at the proposal boundary.
    with pytest.raises(ValidationError, match="may only change"):
        Proposal(family="offload", changes={"kv_offload": False})
    with pytest.raises(ValidationError, match="may only change"):
        Proposal(family="kv-placement", changes={"gpu_layers": 24})
    for changes, match in (({"gpu_layers": "most"}, "integer layer count"), ({"gpu_layers": True}, "layer count"),
                           ({"gpu_layers": 1.5}, "layer count")):
        with pytest.raises(ValidationError, match=match):
            Proposal(family="offload", changes=changes)
    for value in ("ram", 0, None):
        with pytest.raises(ValidationError, match="kv_offload must be a boolean"):
            Proposal(family="kv-placement", changes={"kv_offload": value})
    with pytest.raises(ValueError, match="outside the frozen search space"):
        apply(session, baseline, Proposal(family="offload", changes={"gpu_layers": 12}))
    with pytest.raises(ValueError, match="outside the frozen search space"):
        apply(make_session(tmp_path / "plain"), baseline, Proposal(family="kv-placement",
                                                                   changes={"kv_offload": False}))
    # The same fingerprint deduplication guards an external proposer: one candidate is never measured twice.
    path = tmp_path / "offload-proposals.json"
    rows = [{"family": "baseline", "changes": dict(schedule[0].changes, kv_pair=["f16", "f16"])},
            {"family": "offload", "changes": {"gpu_layers": 24}}]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert [item.label() for item in load_proposal_file(path, session)] == ["baseline", "offload-24"]
    path.write_text(json.dumps(rows + [rows[1]]), encoding="utf-8")
    with pytest.raises(ValueError, match="offload-24 duplicates an earlier candidate"):
        load_proposal_file(path, session)


def test_default_offload_axes_leave_every_pre_u17_candidate_untouched(tmp_path):
    """Backward compatibility: at their defaults the axes add no candidate and change no engine field."""
    session = make_session(tmp_path, budgets={"wall_seconds": 14400, "max_candidates": 12})
    assert (session.search.gpu_layers, session.search.kv_offload) == (("all",), (True,))
    schedule = deterministic_schedule(session)
    assert [item.label() for item in schedule] == ["baseline", "kv-q8_0-q8_0", "weights-q6_k", "spec-draft-mtp",
                                                   "reasoning-on", "ctx-32768"]  # the pre-U17 schedule, unchanged
    assert not [item for item in schedule if item.family in {"offload", "kv-placement"}]
    baseline = apply(session, session.base, schedule[0])
    for proposal in schedule:
        config = apply(session, session.base if proposal.family == "baseline" else baseline, proposal)
        assert config.engine.n_gpu_layers == session.base.engine.n_gpu_layers == "all"
        assert config.engine.kv_offload is True and session.base.engine.kv_offload is True
    # Spelling the defaults out in the search space produces the identical schedule, proposal for proposal.
    explicit = make_session(tmp_path / "x", gpu_layers=("all",), kv_offload=(True,),
                            budgets={"wall_seconds": 14400, "max_candidates": 12})
    assert deterministic_schedule(explicit) == schedule
    assert apply(explicit, explicit.base, schedule[0]).fingerprint() == baseline.fingerprint()


def test_apply_rejects_values_outside_frozen_space_and_capacity(tmp_path):
    session = make_session(tmp_path)
    baseline = apply(session, session.base, baseline_proposal(session))
    with pytest.raises(ValueError, match="outside the frozen search space"):
        apply(session, baseline, Proposal(family="kv", changes={"kv_pair": ("q4_0", "q4_0")}))
    with pytest.raises(ValueError, match="outside the frozen search space"):
        apply(session, baseline, Proposal(family="weights", changes={"quantization": "Q8_0"}))
    with pytest.raises(ValueError, match="outside the frozen search space"):
        apply(session, baseline, Proposal(family="context", changes={"ctx_tier": 16384}))
    with pytest.raises(ValueError):  # a family never reaches another family's axis
        Proposal(family="kv", changes={"ctx_tier": 8192})
    with pytest.raises(ValueError):
        Proposal(family="kv", changes={"kv_pair": ("q8_0", "q8_0"), "quantization": "Q6_K"})
    with pytest.raises(ValueError, match="every axis"):
        Proposal(family="baseline", changes={"kv_pair": ("f16", "f16")})
    with pytest.raises(ValueError):
        Proposal(family="kv", changes={"kv_pair": "q8_0"})
    # Capacity is re-checked on the resulting configuration: a smaller tier with a fat output budget fails closed.
    raw = session.base.model_dump(mode="json")
    raw["engine"]["ctx_size"], raw["requested_input_tokens"], raw["template_reserve_tokens"] = 32768, 16384, 4000
    raw["speed"]["output_tokens"] = 4096
    fat = ContainerRunConfig.model_validate_json(canonical_json(raw))
    with pytest.raises(ValueError, match="exceeds context capacity"):
        apply(session, fat, Proposal(family="context", changes={"ctx_tier": 8192}))
    assert apply(session, fat, Proposal(family="context", changes={"ctx_tier": 32768})).requested_input_tokens == 16384
    # Engine coherence rules apply too (ubatch above batch).
    raw = session.base.model_dump(mode="json")
    raw["engine"]["ubatch_size"] = 2048
    wide = ContainerRunConfig.model_validate_json(canonical_json(raw))
    narrow = make_session(tmp_path / "n", batch_sizes=(2048, 512))
    with pytest.raises(ValueError, match="ubatch_size cannot exceed"):
        apply(narrow, wide, Proposal(family="performance", changes={"batch_size": 512}))


def test_context_proposal_fills_to_the_tier_input_target_under_an_explicit_range(tmp_path):
    """U16 fill rule: with a range, requested_input_tokens is the tier's INPUT target, not a fill ratio."""
    session = make_session(tmp_path, ctx_tiers=(2816, 4864, 6912), context_floor=2048, context_ceiling=6000,
                           context_n_ctx_train=262144)
    baseline = apply(session, session.base, baseline_proposal(session))
    assert (baseline.engine.ctx_size, baseline.requested_input_tokens) == (2816, 2048)  # the floor, exactly
    for tier, expected in ((4864, 4096), (6912, 6000)):  # 4864-768 fits; 6912-768=6144 is clamped to the ceiling
        moved = apply(session, baseline, Proposal(family="context", changes={"ctx_tier": tier}))
        assert (moved.engine.ctx_size, moved.requested_input_tokens) == (tier, expected)
        assert context_fill_for(session, baseline, tier) == expected
        # The output capacity is reserved ON TOP of the input, never taken out of it.
        assert moved.requested_input_tokens + moved.template_reserve_tokens + 512 <= moved.engine.ctx_size
        assert session.search.context_floor <= moved.requested_input_tokens <= session.search.context_ceiling
    # Every other family keeps the baseline's fill, so a context claim is never moved by an unrelated axis.
    kv = apply(session, baseline, Proposal(family="kv", changes={"kv_pair": ("q8_0", "q8_0")}))
    assert kv.requested_input_tokens == baseline.requested_input_tokens == 2048


def test_context_proposal_without_a_range_keeps_the_legacy_proportional_fill(tmp_path):
    """Backward compatibility: no context range -> the base fill ratio, byte for byte as before."""
    session = make_session(tmp_path)  # ctx_tiers (8192, 32768), no range
    assert session.search.context_floor is None and session.search.context_ceiling is None
    baseline = apply(session, session.base, baseline_proposal(session))
    assert (baseline.engine.ctx_size, baseline.requested_input_tokens) == (8192, 4096)  # half of the tier
    moved = apply(session, baseline, Proposal(family="context", changes={"ctx_tier": 32768}))
    assert moved.requested_input_tokens == 4096 * 32768 // 8192 == 16384  # proportional, not 32768 - 768
    assert context_fill_for(session, baseline, 32768) == 16384


def test_proposal_file_is_typed_and_rejects_unknown_axes_or_commands(tmp_path):
    session = make_session(tmp_path)
    good = [{"family": "baseline", "changes": {"quantization": "Q4_K_M", "kv_pair": ["f16", "f16"], "ctx_tier": 8192,
                                               "spec_type": "none", "reasoning": "off", "batch_size": 2048,
                                               "gpu_layers": "all", "kv_offload": True}},
            {"family": "kv", "changes": {"kv_pair": ["q8_0", "q8_0"]}, "note": "external proposer"}]
    path = tmp_path / "proposals.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    proposals = load_proposal_file(path, session)
    assert [item.label() for item in proposals] == ["baseline", "kv-q8_0-q8_0"]
    assert proposals[1].changes == {"kv_pair": ("q8_0", "q8_0")} and proposals[1].note == "external proposer"

    def refuse(rows, match):
        path.write_text(json.dumps(rows), encoding="utf-8")
        with pytest.raises((ValueError, ValidationError), match=match):
            load_proposal_file(path, session)

    refuse({"family": "kv"}, "nonempty JSON list")
    refuse([], "nonempty JSON list")
    refuse([good[1], good[0]], "first proposal must be the only baseline")
    refuse([good[0], {"family": "kv", "changes": {"kv_pair": ["q8_0", "q8_0"]}, "command": "docker rm -f"}], "extra")
    refuse([good[0], {"family": "kv", "changes": {"threads": 16}}], "may only change")
    refuse([good[0], {"family": "performance", "changes": {"batch_size": 4096}}], "outside the frozen search space")
    refuse([good[0], {"family": "kv", "changes": {"kv_pair": ["q8_0", "q8_0"]}},
            {"family": "kv", "changes": {"kv_pair": ["q8_0", "q8_0"]}}], "duplicates")
    refuse([good[0], {"family": "tuning", "changes": {"kv_pair": ["q8_0", "q8_0"]}}], "family")
    refuse(good + [{"family": "reasoning", "changes": {"reasoning": "on"}}] * 4, "allows 4")
    path.write_text("[{", encoding="utf-8")
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        load_proposal_file(path, session)
    with pytest.raises(ValueError, match="regular file"):
        load_proposal_file(tmp_path / "missing.json", session)
    session_with_file = make_session(tmp_path / "s", proposal_mode="file", proposal_file=str(path))
    assert session_with_file.proposal_file == str(path)


def test_proposal_file_nesting_too_deep_is_a_value_error(tmp_path):
    """REV-C1-04: a 100 000-deep JSON array maps to ValueError, never an escaping RecursionError."""
    session = make_session(tmp_path)
    path = tmp_path / "deep.json"
    path.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    with pytest.raises(ValueError, match="nests too deeply"):
        load_proposal_file(path, session)
