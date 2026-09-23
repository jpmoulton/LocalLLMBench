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


BLOCKED_REASON = "sandbox_unavailable: the Docker daemon is not reachable"


def _backfilled(row):
    """What container_eval._execution_block writes for an item the unavailable Docker sandbox could not run."""
    return {**row, "score": 0.0, "passed": False, "status": "environment_error",
            "outcome_status": "environment_error", "reason": BLOCKED_REASON, "synthetic": False,
            "model_evaluated": False}


def test_sandbox_blocked_items_are_counted_and_never_scored_as_a_measured_zero():
    coding = [{"suite": "evalplus", "task_id": f"Mbpp/{index}", "split": "development", "category": "coding"}
              for index in range(3)]
    ruler = [{"suite": "ruler", "task_id": "r/1", "split": "development", "category": "retrieval",
              "status": "completed", "score": 1.0},
             {"suite": "ruler", "task_id": "r/2", "split": "development", "category": "retrieval",
              "status": "environment_error", "score": 1.0, "model_evaluated": False, "reason": "server gone"}]
    scores = _scores([*map(_backfilled, coding), *ruler], measured=True)
    for block in (scores["by_suite"]["evalplus"], scores["by_category"]["coding"]):
        assert block["score"] is None and block["item_count"] == 0 and block["failed_item_count"] == 0
        assert (block["blocked_item_count"], block["blocked_reason"]) == (3, BLOCKED_REASON)
    # Every other failure, another environment error included, stays in the denominator as zero.
    assert scores["by_suite"]["ruler"]["score"] == 0.5 and "blocked_item_count" not in scores["by_suite"]["ruler"]
    summary = _seed_summary([{"role": "stochastic", "temperature": 0.8, "top_p": 0.95, "seed": seed,
                              "evidence": "measured-descriptive", "scores": scores} for seed in (42, 43)],
                            _sampling_grid([0.8], [42, 43], 0.95), [*coding, *ruler])[0]
    blocked = summary["scores"]["by_suite"]["evalplus"]
    assert all(blocked[key] is None for key in ("mean", "minimum", "maximum", "population_stddev"))
    assert (blocked["measured_seed_count"], blocked["unique_benchmark_items"], blocked["blocked_seed_count"],
            blocked["blocked_reason"]) == (0, 3, 2, BLOCKED_REASON)
    assert summary["scores"]["by_suite"]["ruler"]["mean"] == 0.5
    assert "blocked_seed_count" not in summary["scores"]["by_suite"]["ruler"]
    # Two seeds that lost the sandbox for different causes: the group names both, not only the first seed's.
    other = "sandbox_unavailable: docker CLI not found"
    later = _scores([*({**_backfilled(row), "reason": other} for row in coding), *ruler], measured=True)
    mixed = _seed_summary([{"role": "stochastic", "temperature": 0.8, "top_p": 0.95, "seed": seed,
                            "evidence": "measured-descriptive", "scores": block}
                           for seed, block in ((42, scores), (43, later))],
                          _sampling_grid([0.8], [42, 43], 0.95), [*coding, *ruler])[0]
    assert mixed["scores"]["by_suite"]["evalplus"]["blocked_reason"] == f"{other}; {BLOCKED_REASON}"


class EvaluatingRunner(Runner):
    """Completes every attempt and writes the evaluation a sandbox-less evaluator would: the items of the
    ``blocked`` suites backfilled, every other item answered."""

    def __init__(self, clock, expected, blocked):
        super().__init__(clock, [("completed", True)] * 8)
        self.expected, self.blocked = expected, blocked

    def run(self, config, output, *, remaining_budget_seconds):
        result = super().run(config, output, remaining_budget_seconds=remaining_budget_seconds)
        samples = [_backfilled(row) if row["suite"] in self.blocked else {**row, "status": "completed", "score": 1.0}
                   for row in self.expected]
        artifacts = RunArtifacts(output)
        artifacts.write_json("evaluation.json", {"samples": samples})
        artifacts.write_index()
        return result


def test_a_sampling_run_reports_a_sandbox_blocked_suite_as_blocked_never_as_a_score(tmp_path):
    config = read_run_config(EXAMPLE)
    expected = expected_task_rows(config.benchmarks)
    suites = {row["suite"]: row["category"] for row in expected}
    blocked, answered = sorted(suites)[:1], sorted(suites)[1:]
    assert blocked and answered
    report = _run_sampling(config, tmp_path / "out", runner=EvaluatingRunner(Clock(), expected, set(blocked)),
                           clock=Clock(), temperatures=[0.8], seeds=[42], budget_seconds=10000)
    assert report["state"] == "completed"
    assert all(attempt["evidence"] == "measured-descriptive" for attempt in report["attempts"])
    for attempt in report["attempts"]:
        for axis, name in (("by_suite", blocked[0]), ("by_category", suites[blocked[0]])):
            block = attempt["scores"][axis][name]
            assert block["score"] is None and block["blocked_reason"] == BLOCKED_REASON
        assert attempt["scores"]["by_suite"][answered[0]]["score"] == 1.0
    seed = report["seed_summary"][0]["scores"]["by_suite"]
    assert seed[blocked[0]]["mean"] is None and seed[blocked[0]]["blocked_seed_count"] == 1
    assert seed[answered[0]]["mean"] == 1.0
    assert any("blocked_item_count" in line for line in report["interpretation"])
    assert json.loads((tmp_path / "out" / REPORT_NAME).read_text()) == report
    # Nothing but blocked items: nothing was put to the model, so no attempt counts as measured.
    nothing = _run_sampling(config, tmp_path / "none", runner=EvaluatingRunner(Clock(), expected, set(suites)),
                            clock=Clock(), temperatures=[0.8], seeds=[42], budget_seconds=10000)
    assert nothing["state"] == "completed-with-unmeasured-attempts"
    assert all(attempt["evidence"] == "unmeasured" and "every benchmark item was blocked: " + BLOCKED_REASON
               in attempt["failure_reasons"][-1] for attempt in nothing["attempts"])
    # No blocked item anywhere: the report carries neither the keys nor the note.
    plain = _run_sampling(config, tmp_path / "plain", runner=EvaluatingRunner(Clock(), expected, set()),
                          clock=Clock(), temperatures=[0.8], seeds=[42], budget_seconds=10000)
    assert set(plain["attempts"][1]["scores"]["by_suite"][blocked[0]]) == {
        "score", "item_count", "completed_item_count", "failed_item_count", "output_truncation_count",
        "output_truncation_observed_items", "input_truncation_count", "input_truncation_observed_items"}
    assert not any("blocked" in line for line in plain["interpretation"])
    assert "blocked_seed_count" not in json.dumps(plain["seed_summary"])


def _native_config_and_bundle(tmp_path):
    from llmbench.containers.session import native_overlay
    from test_containers_session import make_native_bundle
    bundle = make_native_bundle()
    config = tmp_path / "native-config.json"
    config.write_text(json.dumps(native_overlay(json.loads(EXAMPLE.read_text(encoding="utf-8")), bundle)),
                      encoding="utf-8")
    path = tmp_path / "native-bundle.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    return config, path


def _sample_args(tmp_path, config, *, output="out", **overrides):
    values = dict(config=str(config), output=str(tmp_path / output), temperature=[0.8], seed=[42], top_p=0.95,
                  budget_seconds=10000.0, policy=str(tmp_path / "policy.json"), capabilities_dir="caps",
                  image_bundle=None, native_bundle=None, plan_only=False)
    return SimpleNamespace(**{**values, **overrides})


def test_native_sampling_is_pinned_by_its_bundle_and_authorized_for_its_runtime(tmp_path, capsys):
    from llmbench.containers.sampling import main_sampling
    from test_containers_session import make_native_bundle
    config, bundle = _native_config_and_bundle(tmp_path)
    assert main_sampling(_sample_args(tmp_path, config, native_bundle=str(bundle), plan_only=True)) == 0
    provenance = json.loads((tmp_path / "out" / "sampling-plan.json").read_text())["provenance"]
    assert provenance["image_bundle"] is None and provenance["native_bundle"]["native_server"]["executable_sha256"] \
        == "d" * 64
    capsys.readouterr()
    rebuilt = tmp_path / "rebuilt.json"
    rebuilt.write_text(make_native_bundle(executable_sha256="0" * 64).model_dump_json(), encoding="utf-8")
    for overrides, message in (({"native_bundle": str(rebuilt)}, "native_server.executable_sha256"),
                               ({"image_bundle": str(bundle)}, "is a native bundle"),
                               ({"image_bundle": str(bundle), "native_bundle": str(bundle)}, "pass one bundle")):
        assert main_sampling(_sample_args(tmp_path, config, output="refused", plan_only=True, **overrides)) == 2
        assert message in capsys.readouterr().err and not (tmp_path / "refused").exists()
    # A live run needs native permission; the NVIDIA policy is refused before any runner or output exists.
    (tmp_path / "policy.json").write_text(json.dumps({"allow_model_operations": True, "allow_inference": True,
                                                      "allow_container_execution": True}), encoding="utf-8")
    assert main_sampling(_sample_args(tmp_path, config, output="denied"),
                         runner_factory=lambda args: pytest.fail("no runner without permission")) == 2
    assert "native forbidden" in capsys.readouterr().err and not (tmp_path / "denied").exists()
    (tmp_path / "policy.json").write_text(json.dumps({"allow_model_operations": True, "allow_inference": True,
                                                      "allow_native_execution": True}), encoding="utf-8")
    clock = Clock()
    runner = Runner(clock, [("completed", True), ("completed", True)])
    assert main_sampling(_sample_args(tmp_path, config, output="ran"), runner_factory=lambda args: runner,
                         clock=clock) == 3  # fake results carry no evaluation: completed-with-unmeasured-attempts
    assert [call[0].runtime for call in runner.calls] == ["metal-native", "metal-native"]


def test_the_default_sampling_runner_dispatches_on_the_configs_runtime(tmp_path):
    from llmbench.containers.runtime import DispatchRunner
    from llmbench.containers.sampling import _default_runner
    runner = _default_runner(SimpleNamespace(capabilities_dir="caps", policy="p.json"))
    assert isinstance(runner, DispatchRunner) and runner.built == ()
    assert (runner.capabilities_dir, runner.policy_path, runner.policy) == ("caps", "p.json", None)


def test_nvidia_sampling_provenance_and_bundle_check_are_unchanged(tmp_path, capsys):
    from llmbench.containers.sampling import main_sampling
    from test_containers_session import make_bundle
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    bundle = make_bundle(raw["inference_image"]["image_id"].split(":")[1])
    raw["inference_image"] = json.loads(bundle.inference.model_dump_json())
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    path = tmp_path / "image-bundle.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    assert main_sampling(_sample_args(tmp_path, config, image_bundle=str(path), plan_only=True)) == 0
    provenance = json.loads((tmp_path / "out" / "sampling-plan.json").read_text())["provenance"]
    assert set(provenance) == {"policy", "capabilities_dir", "image_bundle"}
    assert provenance["image_bundle"]["inference"]["image_id"] == bundle.inference.image_id
    other = tmp_path / "other.json"
    other.write_text(make_bundle("2" * 64).model_dump_json(), encoding="utf-8")
    assert main_sampling(_sample_args(tmp_path, config, output="x", image_bundle=str(other), plan_only=True)) == 2
    assert "config images differ from the prepared bundle" in capsys.readouterr().err
