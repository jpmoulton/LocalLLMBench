import copy

import pytest

from llmbench.config import AcceptancePolicy, Budgets, CampaignPolicy
from llmbench.measurement import SpeedObservation, context_envelope, context_schedule, summarize_speed
from llmbench.search import BudgetTracker, assess_candidate, paired_comparison, pareto_frontier, propose


def observation(**changes):
    fields = dict(status="completed", finish_reason="length", output_tokens=512,
                  native_generation_seconds=6.4, elapsed_seconds=7, first_event_seconds=.5,
                  content_event_times=(.5, 1., 2., 3., 4., 5., 6., 7.),
                  input_tokens=1024, expected_input_tokens=1024, accepted_tokens_verified=True)
    fields.update(changes)
    return SpeedObservation(**fields)


def test_speed_above_floor_has_no_ceiling():
    result = summarize_speed([observation()] * 5, AcceptancePolicy())
    assert result["qualifies"]
    assert result["minimum_native_tps"] == 80


@pytest.mark.parametrize("changes", [
    {"synthetic": True}, {"status": "cancelled"}, {"output_tokens": 30, "finish_reason": "stop"},
    {"content_event_times": (7.,)}, {"native_generation_seconds": None},
    {"accepted_tokens_verified": False}, {"expected_input_tokens": 1025},
    {"content_event_times": (.5, 7.)}, {"native_generation_seconds": 12., "elapsed_seconds": 12.},
    {"content_event_times": (6.999, 7.)},
])
def test_speed_does_not_discard_bad_repetition(changes):
    result = summarize_speed([observation()] * 4 + [observation(**changes)], AcceptancePolicy())
    assert result["attempted"] == 5
    assert not result["qualifies"]


def test_event_chunks_are_not_output_tokens():
    row = observation().metrics()
    assert row["output_tokens"] == 512
    assert len(row["content_event_times"]) == 8


def test_unmeasured_context_not_certified():
    assert context_schedule() == (8192, 16384, 32768, 65536, 131072, 262144)
    row = {"actual_input_tokens": 8192, "context_verified": True,
           "speed_passed": True, "quality_passed": True, "synthetic": True}
    assert context_envelope([row])["largest_tested_usable_input_tokens"] is None


def tasks(score=1., status="completed"):
    return [dict(task_id=f"t{i}", fixture_hash=f"hash{i}", category="tools", split="development",
                 score=score, status=status) for i in range(20)]


def test_paired_comparison_failure_denominator():
    baseline = tasks()
    candidate = tasks(1., "environment_error")
    result = paired_comparison(baseline, candidate, resamples=100)
    assert result["status"] == "degraded"
    assert result["mean_delta"] == -1
    assert paired_comparison(baseline, baseline, resamples=100)["status"] == "inconclusive"
    with pytest.raises(ValueError):
        paired_comparison(baseline, candidate[:-1])
    candidate[0]["fixture_hash"] = "mismatch"
    with pytest.raises(ValueError):
        paired_comparison(baseline, candidate)


def test_small_sample_inconclusive():
    assert paired_comparison(tasks()[:2], tasks()[:2], resamples=100)["status"] == "inconclusive"


def test_renamed_fixture_replays_do_not_inflate_certainty():
    replayed = [{**tasks()[0], "task_id": f"alias-{index}"} for index in range(200)]
    with pytest.raises(ValueError, match="duplicate fixture"):
        paired_comparison(replayed, replayed, resamples=100)


def test_unseen_regressions_bound_does_not_collapse():
    rows = [dict(task_id=f"t{i}", fixture_hash=f"hash{i}", category="tools", split="development",
                 score=1., status="completed") for i in range(200)]
    result = paired_comparison(rows, rows, resamples=100)
    assert result["status"] == "noninferior"
    assert -.02 < result["lower_95"] < 0


def test_budget_reserves_validation_and_limits_candidates():
    now = [0]
    tracker = BudgetTracker(Budgets(wall_seconds=100, reserve_validation_seconds=20, max_candidates=1),
                            clock=lambda: now[0])
    assert not tracker.admit(81)
    assert tracker.admit(60)
    assert not tracker.admit(1)
    now[0] = 85
    assert tracker.admit(10, validation=True)
    assert not tracker.admit(20, validation=True)


def test_proposal_causal_axis(manifest):
    candidate = propose(manifest, {"backend.k_cache": "q8_0"}, "kv")
    assert candidate.backend.k_cache == "q8_0"
    assert candidate.fingerprint() != manifest.fingerprint()
    with pytest.raises(ValueError):
        propose(manifest, {"generation.repair_attempts": 3}, "kv")


def test_reasoning_and_ubatch_axes_are_proposable_and_exclusive(manifest):
    reasoning = propose(manifest, {"backend.reasoning": "on"}, "reasoning")
    assert reasoning.backend.reasoning == "on" and reasoning.fingerprint() != manifest.fingerprint()
    ubatch = propose(manifest, {"backend.ubatch_size": 256}, "performance")
    assert ubatch.backend.ubatch_size == 256
    with pytest.raises(ValueError):  # reasoning is its own family, never a performance knob
        propose(manifest, {"backend.reasoning": "on"}, "performance")
    with pytest.raises(ValueError):  # and the reasoning family carries nothing else
        propose(manifest, {"backend.ubatch_size": 256}, "reasoning")
    with pytest.raises(ValueError):
        propose(manifest, {"backend.reasoning": "on", "backend.k_cache": "q8_0"}, "reasoning")


def evidence(speed=80):
    return dict(synthetic=False, status="completed", effective_settings_verified=True,
                actual_context_verified=True, actual_input_tokens=8192, holdout_passed=True,
                speed={"qualifies": True, "minimum_native_tps": speed},
                quality={key: {"score": 1., "comparison": {"status": "noninferior"}}
                         for key in ("coding", "tools", "retrieval")})


def test_winner_requires_live_verified_holdout():
    row = evidence()
    assert assess_candidate(row, CampaignPolicy(name="test"))["eligible"]
    for key in ("effective_settings_verified", "actual_context_verified", "holdout_passed"):
        bad = copy.deepcopy(row)
        bad[key] = False
        assert not assess_candidate(bad, CampaignPolicy(name="test"))["eligible"]
    row["synthetic"] = True
    assert not assess_candidate(row, CampaignPolicy(name="test"))["eligible"]


def test_frontier_keeps_higher_speed():
    rows = [evidence(50), evidence(80)]
    for row in rows:
        row["eligibility"] = assess_candidate(row, CampaignPolicy(name="test"))
    assert pareto_frontier(rows) == [rows[1]]
