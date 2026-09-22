"""Sampling repeats fixed items; failures and time limits must never become quality evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.containers.artifacts import RunArtifacts
from llmbench.containers.config import ContainerRunConfig, read_run_config
from llmbench.containers.sampling import (
    MAX_ATTEMPTS, REPORT_NAME, _minimum_grant, _read_scores, _run_sampling, _sampling_grid, _scores, _seed_summary,
)
from llmbench.evaluations.selection import expected_task_rows

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.json"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Runner:
    def __init__(self, clock, outcomes, *, elapsed=1):
        self.clock, self.outcomes, self.elapsed = clock, iter(outcomes), elapsed
        self.calls = []

    def run(self, config, output, *, remaining_budget_seconds):
        self.calls.append((config, remaining_budget_seconds))
        self.clock.now += self.elapsed
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        state, cleanup = outcome
        return SimpleNamespace(state=state, config_fingerprint=config.fingerprint(), attempt_id=str(len(self.calls)),
                               synthetic=False, effective_settings_verified=True, actual_context_verified=True,
                               cleanup=SimpleNamespace(verified=cleanup), abort_campaign=not cleanup,
                               failure_stage=None, failure_reasons=() if state == "completed" else (state,),
                               warnings=(), elapsed_seconds=self.elapsed)


def test_sampling_grid_deduplicates_without_repeating_greedy():
    grid = _sampling_grid([0.8, 0.6, 0.8], [43, 42, 43], 0.9)
    assert grid[0] == {"role": "greedy-control", "temperature": 0.0, "top_p": 1.0, "seed": 43}
    assert [(row["temperature"], row["seed"]) for row in grid[1:]] == [(0.8, 43), (0.8, 42), (0.6, 43), (0.6, 42)]
    assert all(row["role"] == "stochastic" and row["temperature"] > 0 for row in grid[1:])


@pytest.mark.parametrize("temperatures,seeds,top_p", [
    ([0], [42], 0.95), ([float("nan")], [42], 0.95), ([5.01], [42], 0.95),
    ([0.8], [-1], 0.95), ([0.8], [True], 0.95), ([0.8], [42], 0),
    ([0.8], [42], float("inf")), ([], [42], 0.95), ([0.8], [], 0.95),
    ([0.8], list(range(MAX_ATTEMPTS)), 0.95),
])
def test_invalid_or_unbounded_grids_are_refused(temperatures, seeds, top_p):
    with pytest.raises(ValueError):
        _sampling_grid(temperatures, seeds, top_p)


def test_budget_stops_dispatch_without_refunding_fast_or_failed_attempts(tmp_path):
    config = read_run_config(EXAMPLE)
    clock = Clock()
    floor = _minimum_grant(config)
    runner = Runner(clock, [("failed", True)], elapsed=2)
    report = _run_sampling(config, tmp_path / "out", runner=runner, clock=clock, temperatures=[0.8], seeds=[42],
                           budget_seconds=floor + 1)
    assert len(runner.calls) == 1
    assert runner.calls[0][1] == floor + 1
    assert report["state"] == "stopped" and report["unattempted_indices"] == [1]
    assert report["attempts"][0]["evidence"] == "unmeasured"
    assert all(row["score"] is None for row in report["attempts"][0]["scores"]["by_suite"].values())


def test_parent_grant_too_small_prevents_even_first_runner_call(tmp_path):
    raw = read_run_config(EXAMPLE).model_dump(mode="json")
    raw["parent_grant_seconds"] = 1
    clock = Clock()
    runner = Runner(clock, [])
    report = _run_sampling(ContainerRunConfig.model_validate_json(json.dumps(raw)), tmp_path / "out", runner=runner,
                           clock=clock, budget_seconds=10000)
    assert runner.calls == []
    assert report["execution_performed"] is False and report["attempts"] == []
    assert report["stop_reason"] == "insufficient-budget-for-candidate-bounds"


@pytest.mark.parametrize("outcome", [("cleanup-uncertain", False), RuntimeError("lost runner"), KeyboardInterrupt()])
def test_uncertain_cleanup_or_escaped_exception_aborts_and_persists(tmp_path, outcome):
    clock = Clock()
    runner = Runner(clock, [outcome, ("completed", True)])
    report = _run_sampling(read_run_config(EXAMPLE), tmp_path / "out", runner=runner, clock=clock, budget_seconds=10000)
    assert len(runner.calls) == 1
    assert report["state"] == "aborted"
    assert report["attempts"][0]["evidence"] == "unmeasured"
    assert report["attempts"][0]["abort_campaign"] is True
    assert json.loads((tmp_path / "out" / REPORT_NAME).read_text()) == report
    assert report["unattempted_indices"] == list(range(1, 7))


def test_cancelled_candidate_does_not_start_another_even_after_verified_cleanup(tmp_path):
    clock = Clock()
    runner = Runner(clock, [("cancelled", True), ("completed", True)])
    report = _run_sampling(read_run_config(EXAMPLE), tmp_path / "out", runner=runner, clock=clock, budget_seconds=10000)
    assert len(runner.calls) == 1
    assert report["stop_reason"] == "candidate-cancelled"
    assert report["attempts"][0]["evidence"] == "unmeasured"


def test_missing_evidence_cannot_promote_completed_runs_to_scores(tmp_path):
    clock = Clock()
    runner = Runner(clock, [("completed", True), ("failed", True)])
    report = _run_sampling(read_run_config(EXAMPLE), tmp_path / "out", runner=runner, clock=clock,
                           temperatures=[0.8], seeds=[42], budget_seconds=10000)
    assert len(runner.calls) == 2
    assert report["state"] == "completed-with-unmeasured-attempts"
    assert all(row["evidence"] == "unmeasured" for row in report["attempts"])
    assert report["seed_summary"][0]["measured_seeds"] == []
    assert all(row["mean"] is None for row in report["seed_summary"][0]["scores"]["by_suite"].values())


def test_plan_is_inert_and_existing_experiment_is_never_overwritten(tmp_path):
    config = read_run_config(EXAMPLE)
    output = tmp_path / "out"
    report = _run_sampling(config, output, plan_only=True, budget_seconds=1)
    original = (output / "sampling-plan.json").read_bytes()
    assert report["execution_performed"] is False and report["attempts"] == []
    with pytest.raises(FileExistsError):
        _run_sampling(config, output, plan_only=True, budget_seconds=1)
    assert (output / "sampling-plan.json").read_bytes() == original


def test_changed_benchmark_item_set_with_valid_index_is_not_comparable(tmp_path):
    config = read_run_config(EXAMPLE)
    expected = expected_task_rows(config.benchmarks)
    samples = [{**row, "status": "completed", "score": 1.0} for row in expected[:-1]]
    artifacts = RunArtifacts(tmp_path / "attempt")
    artifacts.write_json("evaluation.json", {"samples": samples})
    artifacts.write_index()
    with pytest.raises(ValueError, match="item set"):
        _read_scores(config, tmp_path / "attempt", expected)


def test_seed_summary_does_not_pool_repetitions_as_new_items_and_preserves_truncation_unknowns():
    samples = [{"suite": "ruler", "category": "retrieval", "status": "completed", "score": 1.0,
                "finish_reason": "length"},
               {"suite": "ruler", "category": "retrieval", "status": "timeout", "score": 1.0}]
    first = _scores(samples, measured=True)
    assert first["by_suite"]["ruler"]["score"] == 0.5  # failed item stays in denominator, not a fabricated pass
    assert first["by_suite"]["ruler"]["output_truncation_count"] == 1
    assert first["by_suite"]["ruler"]["output_truncation_observed_items"] == 1
    assert first["by_suite"]["ruler"]["input_truncation_count"] is None
    second = _scores([{**row, "status": "completed", "score": 1.0} for row in samples], measured=True)
    summary = _seed_summary([{"role": "stochastic", "temperature": 0.8, "top_p": 0.95, "seed": seed,
                             "evidence": "measured-descriptive", "scores": scores}
                            for seed, scores in [(42, first), (43, second)]],
                            _sampling_grid([0.8], [42, 43], 0.95), samples)[0]
    suite = summary["scores"]["by_suite"]["ruler"]
    assert suite["unique_benchmark_items"] == 2 and suite["measured_seed_count"] == 2
    assert suite["mean"] == 0.75 and suite["population_stddev"] == 0.25
    assert summary["planned_seeds"] == [42, 43] and summary["group_complete"] is True


@pytest.mark.parametrize("failed_seed_attempted", [False, True])
def test_seed_group_never_averages_only_surviving_seeds(failed_seed_attempted):
    samples = [{"suite": "ruler", "category": "retrieval", "status": "completed", "score": 1.0}]
    attempts = [{"role": "stochastic", "temperature": 0.8, "top_p": 0.95, "seed": 42,
                 "evidence": "measured-descriptive", "scores": _scores(samples, measured=True)}]
    if failed_seed_attempted:
        attempts.append({"role": "stochastic", "temperature": 0.8, "top_p": 0.95, "seed": 43,
                         "evidence": "unmeasured", "scores": _scores(samples, measured=False)})
    summaries = _seed_summary(attempts, _sampling_grid([0.8, 0.6], [42, 43], 0.95), samples)
    for summary in summaries:
        assert summary["planned_seeds"] == [42, 43] and summary["group_complete"] is False
        suite = summary["scores"]["by_suite"]["ruler"]
        assert all(suite[key] is None for key in ("mean", "minimum", "maximum", "population_stddev"))
    assert summaries[0]["measured_seeds"] == [42]
    assert summaries[0]["attempted_seeds"] == ([42, 43] if failed_seed_attempted else [42])
    assert summaries[1]["attempted_seeds"] == [] and summaries[1]["measured_seeds"] == []
