"""The non-inferiority gate is exact; these pin that its reach is REPORTED, because at default item counts it
cannot be met by anything but the baseline and nothing used to say so."""

import pytest

from llmbench.config import CampaignPolicy
from llmbench.search import (assess_candidate, paired_comparison, smallest_demonstrable_loss, tasks_needed_for_loss,
                             unmeasured_categories)


def rows(count, failing=()):
    return [{"task_id": f"t{index}", "fixture_hash": f"{index:064x}", "category": "tools", "split": "development",
             "status": "completed", "score": 0.0 if index in failing else 1.0} for index in range(count)]


@pytest.mark.parametrize(("tasks", "bound"), [(24, 0.1425), (60, 0.0596), (145, 0.0251), (183, 0.02)])
def test_the_best_demonstrable_loss_shrinks_only_with_more_items(tasks, bound):
    assert smallest_demonstrable_loss(tasks) == pytest.approx(bound, abs=5e-4)


def test_how_many_items_a_tolerance_needs():
    assert tasks_needed_for_loss(0.02) == 183 and tasks_needed_for_loss(0.05) == 72 and tasks_needed_for_loss(0.15) == 23
    assert tasks_needed_for_loss(0) is None and smallest_demonstrable_loss(0) is None


def test_an_identical_candidate_is_inconclusive_at_24_items_and_the_comparison_says_why():
    same = paired_comparison(rows(24), rows(24), maximum_loss=0.02, resamples=200)
    assert same["status"] == "inconclusive" and same["mean_delta"] == 0
    assert same["smallest_demonstrable_loss"] == pytest.approx(0.1425, abs=5e-4)
    assert same["smallest_demonstrable_loss"] > same["maximum_loss"]  # unreachable however good the candidate
    assert same["tasks_needed_for_maximum_loss"] == 183
    enough = paired_comparison(rows(200), rows(200), maximum_loss=0.02, resamples=200)
    assert enough["status"] == "noninferior"


def test_an_undeclared_category_is_unmeasured_not_failed_and_is_always_reported():
    policy = CampaignPolicy(name="t")
    passing = {"attempted": 24, "score": 0.8, "comparison": {"status": "noninferior"}}
    evidence = {"synthetic": False, "status": "completed", "effective_settings_verified": True,
                "actual_context_verified": True, "holdout_passed": True, "speed": {"qualifies": True},
                "quality": {"tools": passing, "retrieval": passing,
                            "coding": {"attempted": 0, "score": None, "comparison": {"status": "unmeasured"}}}}
    assert assess_candidate(evidence, policy) == {"eligible": True, "reasons": [], "unmeasured_categories": ["coding"]}
    # a DECLARED category that produced no score is attempted, stays in the denominator, and fails the candidate
    evidence["quality"]["coding"] = {"attempted": 3, "score": None, "comparison": {"status": "inconclusive"}}
    verdict = assess_candidate(evidence, policy)
    assert not verdict["eligible"] and "coding_absolute_quality_failed" in verdict["reasons"]
    assert verdict["unmeasured_categories"] == []
    # speed alone is never enough
    nothing = {name: {"attempted": 0, "score": None} for name in ("coding", "tools", "retrieval")}
    assert unmeasured_categories(nothing) == ["coding", "tools", "retrieval"]
    verdict = assess_candidate({**evidence, "quality": nothing}, policy)
    assert not verdict["eligible"] and "no_quality_measured" in verdict["reasons"]


def test_the_plan_warns_before_the_run_when_nothing_but_the_baseline_can_be_validated():
    from llmbench.containers.derive import benchmark_plan_lines, quality_power
    from llmbench.registry import builtin_registry
    power = quality_power([{"benchmark_id": "bfcl", "status": "selected", "items": 21},
                           {"benchmark_id": "evalplus", "status": "skipped", "items": 0}], builtin_registry())
    assert power == [{"category": "tools", "public_items": 21, "smallest_demonstrable_loss": 0.1611,
                      "policy_max_quality_loss": 0.02, "items_needed": 183,
                      "can_validate_a_non_baseline_candidate": False}]
    lines = benchmark_plan_lines({"dataset_root_reason": "x", "datasets_present": [], "selected": [],
                                  "wall_planned_seconds": 0.0, "wall_allowance_seconds": 0.0, "benchmarks": [],
                                  "power": power})
    assert any("only the baseline can be VALIDATED" in line and "183 items" in line for line in lines)
