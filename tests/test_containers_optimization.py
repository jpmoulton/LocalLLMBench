"""Optimization boundaries that could otherwise promote invalid experimental evidence."""
import json

import pytest

from llmbench.analysis import infer_comparison_family
from llmbench.config import ExperimentManifest
from llmbench.containers.optimization import (interaction_proposals, optimization_plan, screening_session, shortlist)
from llmbench.containers.proposals import Proposal, apply, baseline_proposal
from llmbench.search import propose
from test_containers_session import make_session


def test_interactions_combine_distinct_axes_without_changing_workload(tmp_path):
    session = make_session(tmp_path, budgets={"wall_seconds": 14400})
    baseline = baseline_proposal(session)
    kv = Proposal(family="kv", changes={"kv_pair": ("q8_0", "q8_0")})
    mtp = Proposal(family="speculation", changes={"spec_type": "draft-mtp"})
    context = Proposal(family="context", changes={"ctx_tier": 32768})
    combos = interaction_proposals(session, baseline, [kv, mtp, context], 4)
    assert len(combos) == 1
    measured = apply(session, apply(session, session.base, baseline), combos[0])
    assert measured.engine.cache_type_k == "q8_0" and measured.engine.spec_type == "draft-mtp"
    assert measured.requested_input_tokens == session.base.requested_input_tokens
    assert measured.engine.ctx_size == session.base.engine.ctx_size
    assert interaction_proposals(session, baseline, [kv, mtp], 0) == []
    reverse = Proposal(family="combination", changes=dict(reversed(list(combos[0].changes.items()))))
    assert reverse.label() == combos[0].label()
    with pytest.raises(ValueError):
        Proposal(family="combination", changes={"ctx_tier": 32768, "spec_type": "draft-mtp"})


def test_combined_comparison_requires_explicit_treatment_and_fixed_provenance(manifest):
    base = manifest.model_copy(update={"environment_hash": "recorded"})
    candidate = propose(base, {"backend.k_cache": "q8_0", "backend.mtp": True}, "combination")
    with pytest.raises(ValueError, match="exactly one"):
        infer_comparison_family(base, candidate)
    candidate = candidate.model_copy(update={"annotations": {"treatment_family": "combination"}})
    assert infer_comparison_family(base, candidate) == "combination"
    for key, value in (("environment_hash", "different"), ("requested_input_tokens", 8192)):
        raw = candidate.model_dump(mode="json")
        raw[key] = value
        with pytest.raises(ValueError):
            infer_comparison_family(base, ExperimentManifest.model_validate_json(json.dumps(raw)))


def test_failed_or_unverified_screening_cannot_win_shortlist():
    proposals = [Proposal(family="kv", changes={"kv_pair": ("q8_0", "q8_0")}),
                 Proposal(family="speculation", changes={"spec_type": "draft-mtp"}),
                 Proposal(family="reasoning", changes={"reasoning": "on"})]
    evidence = []
    report = []
    for i, proposal in enumerate(proposals):
        evidence.append({"attempt_id": str(i), "status": "completed", "synthetic": False,
                         "effective_settings_verified": True, "actual_context_verified": True,
                         "speed": {"minimum_native_tps": 100 * (i + 1)},
                         "quality": {"coding": {"attempted": 8, "absolute_passed": True, "score": 1.0}}})
        report.append({"attempt_id": str(i), "label": proposal.label(), "split": "development"})
    evidence[1]["status"] = "failed"
    evidence[2]["actual_context_verified"] = False
    outcomes = [{"campaign": {"results": evidence}, "report": {"rows": report}}]
    assert shortlist(outcomes, proposals, 3) == [proposals[0]]


def test_screening_is_nested_and_tiny_budget_refused_before_work(tmp_path):
    session = make_session(tmp_path, budgets={"wall_seconds": 14400})
    screened = screening_session(session, 1)
    for original, small in zip(session.base.benchmarks, screened.base.benchmarks):
        assert small.task_ids == original.task_ids[:1]
        assert small.options == original.options
        assert small.seed == original.seed
    with pytest.raises(ValueError, match="not enough remaining budget"):
        optimization_plan(session, budget_seconds=600)
    stochastic = session.model_copy(update={"base": session.base.model_copy(update={
        "generation": session.base.generation.model_copy(update={"temperature": 0.6})})})
    with pytest.raises(ValueError, match="use sample"):
        optimization_plan(stochastic)


def test_optimize_authorizes_the_sessions_own_runtime_and_dispatches_its_runner(tmp_path):
    """A native session needs native permission, not container permission, and each stage's runner dispatches on
    the candidates' runtime; an NVIDIA policy is refused before the output directory exists."""
    from llmbench.containers.optimization import optimize
    from llmbench.containers.runtime import DispatchRunner
    from llmbench.containers.session import ContainerSessionConfig, policy_for
    from llmbench.safety import OperationForbidden
    from test_containers_session import make_native_bundle, native_session
    bundle_path = tmp_path / "native-bundle.json"
    bundle_path.write_text(make_native_bundle().model_dump_json(), encoding="utf-8")
    raw = json.loads(native_session(tmp_path / "s", budgets={"wall_seconds": 14400}).model_dump_json())
    raw["image_bundle"] = str(bundle_path)
    session = ContainerSessionConfig.model_validate_json(json.dumps(raw))
    policies = {}
    for name, grants in (("nvidia", {"allow_container_execution": True}), ("mac", {"allow_native_execution": True})):
        policies[name] = tmp_path / f"{name}.json"
        policies[name].write_text(json.dumps({"allow_model_operations": True, "allow_inference": True, **grants}),
                                  encoding="utf-8")
    with pytest.raises(OperationForbidden, match="native"):
        optimize(session, tmp_path / "refused", policy_path=policies["nvidia"], max_combinations=0,
                 tune_fn=lambda *a, **k: pytest.fail("no stage may start"))
    assert not (tmp_path / "refused").exists()
    stages = []

    def fake_tune(stage, output, *, runner, session_lock, bundle, bundle_path):
        stages.append({"stage": stage, "runner": runner, "bundle": bundle, "bundle_path": bundle_path})
        return {"stop_reason": "complete", "summary": {"abort_campaign": False},
                "report": {"rows": [], "recommendation": None}, "campaign": {"results": []}}
    report = optimize(session, tmp_path / "out", policy_path=policies["mac"], capabilities_dir="caps",
                      max_combinations=0, tune_fn=fake_tune)
    assert report["state"] == "completed"
    assert [item["stage"].session_id.rsplit("-", 1)[-1] for item in stages] == ["screen", "confirm"]
    for item in stages:
        runner = item["runner"]
        assert isinstance(runner, DispatchRunner) and runner.built == ()
        assert (runner.capabilities_dir, runner.policy_path) == ("caps", policies["mac"])
        assert runner.policy == policy_for(item["stage"])
        assert item["bundle"].native_server.executable_sha256 == "d" * 64 and item["bundle_path"] == str(bundle_path)
        assert item["stage"].base.runtime == "metal-native"
