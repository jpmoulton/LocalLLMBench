"""Container executor and the tune/resume session paths against a fake runner. No Docker, model or GPU."""

import hashlib
import json
from pathlib import Path

import pytest

from llmbench.analysis import config_key, plan_holdout_candidates
from llmbench.config import RunMode, TaskSelection, canonical_json
from llmbench.containers import campaign as campaign_module
from llmbench.containers import session as session_module
from llmbench.containers.campaign import (BenchmarkOptionsNotCarried, CampaignAbort, ContainerExecutor,
                                          SessionDeadlineExceeded, campaign_identity, read_indexed_artifact)
from llmbench.containers.config import BenchmarkSelection
from llmbench.containers.proposals import deterministic_schedule
from llmbench.containers.session import (asset_metadata, build_candidates, holdout_selections, policy_for,
                                         read_ledger, resume, tune)
from llmbench.controller import execute_attempt
from llmbench.measurement import SpeedObservation
from llmbench.provenance import environment_record
from llmbench.registry import builtin_registry
from llmbench.reports import export_preset, export_provisional_preset
from llmbench.safety import SessionLock
from llmbench.store import Store
from test_containers_session import (development_labels, LIVE_RULER_BENCHMARK, LOCK, TEMPLATE, FakeClock, FakeRunner, isolated_gpu_lock,
                                     live_ruler_selection, make_session, metadata_reader)

_ = isolated_gpu_lock  # fixture re-export for pytest collection


def small_session(tmp_path, **changes):
    options = {"quantizations": ("Q4_K_M",), "spec_types": ("none",), "reasoning": ("off",), "ctx_tiers": (8192,)}
    options.update(changes)
    return make_session(tmp_path, **options)


def candidates(session, template=TEMPLATE):
    facts = asset_metadata(session, reader=metadata_reader(template))
    return build_candidates(session, None, deterministic_schedule(session), facts,
                            scorer_revision=builtin_registry().digest(),
                            environment_hash=environment_record()["sha256"])


def run_tune(tmp_path, session, *, script=None, clock=None, template=TEMPLATE, output=None):
    clock = clock or FakeClock()
    runner = FakeRunner(clock, template=template, script=script, policy=policy_for(session))
    outcome = tune(session, output or tmp_path / "out", runner=runner, session_lock=LOCK, clock=clock,
                   wall=clock.wall, metadata_reader=metadata_reader())
    return outcome, runner, clock


def test_executor_maps_manifest_to_registered_config_and_rebuilds_holdout_benchmarks(tmp_path):
    session = small_session(tmp_path)
    pairs = candidates(session)
    executor = ContainerExecutor(session, None, runner=None, output_root=tmp_path / "out", deadline_epoch=1e12,
                                 template_hashes={})
    for config, manifest in pairs:
        executor.register(config, manifest)
    config, manifest = pairs[1]
    assert executor.config_for(manifest) == config
    held = plan_holdout_candidates([manifest], holdout_selections(session))[0]
    holdout_config = executor.config_for(held)
    assert holdout_config.label == "kv-q8_0-q8_0-holdout" and holdout_config.session_id == "session-test"
    assert [(item.benchmark_id, item.split, item.seed, item.task_ids) for item in holdout_config.benchmarks] == [
        ("niah", "holdout", 7, ("single-middle", "multi"))]
    assert holdout_config.engine == config.engine and holdout_config.assets == config.assets
    assert holdout_config.fingerprint() != config.fingerprint()  # task selection is part of the run config
    foreign = manifest.model_copy(update={"backend": manifest.backend.model_copy(update={"batch_size": 1024})})
    with pytest.raises(ValueError, match="registered"):
        executor.config_for(foreign)
    other = pairs[0][0]
    with pytest.raises(ValueError, match="different container config"):
        executor.register(other, manifest)


def test_executor_returns_controller_compatible_evidence_from_fake_runner_artifacts(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path)
    (config, manifest), *_ = candidates(session)
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session))
    executor = ContainerExecutor(session, None, runner=runner, output_root=tmp_path / "out",
                                 deadline_epoch=clock.wall() + 3600, template_hashes={
                                     sha: meta["template_hash"] for sha, meta in
                                     asset_metadata(session, reader=metadata_reader()).items()},
                                 clock=clock, wall=clock.wall)
    executor.register(config, manifest)
    events, artifacts = [], {}
    executor.bind_event_sink(events.append)
    executor.bind_artifact_sink(lambda name, content: artifacts.__setitem__(name, content))
    evidence = executor(manifest, timeout_seconds=900)
    assert runner.calls[0]["budget"] == 900 and runner.calls[0]["run_dir"] == tmp_path / "out" / "candidates" / "000-baseline"
    assert evidence["synthetic"] is False and evidence["model_evaluated"] is True
    assert evidence["effective_settings_verified"] is True and evidence["actual_context_verified"] is True
    assert evidence["holdout_passed"] is False and evidence["comparisons"] == {} and evidence["samples_persisted"] is False
    assert all(isinstance(item, SpeedObservation) for item in evidence["speed_observations"])
    assert {row["task_id"] for row in evidence["samples"]} == {"tools/nested-exact-v1", "tools/no-call-v1",
                                                               "niah/single-middle/development/seed-42",
                                                               "niah/multi/development/seed-42"}
    assert evidence["raw"]["template_check"]["matches"] is True and evidence["raw"]["label"] == "baseline"
    assert evidence["raw"]["candidate_report"] == "reports/candidate.md"
    assert set(artifacts) == {"candidate.json", "result.json", "settings-evidence.json", "stages.jsonl"}
    assert json.loads(artifacts["candidate.json"])["run_dir"] == str(tmp_path / "out" / "candidates" / "000-baseline")
    assert [event["event"] for event in events] == ["candidate-start", "candidate-end"]
    # The controller accepts it as measured live evidence.
    with Store(tmp_path / "store") as store:
        result = execute_attempt(store, manifest, policy_for(session), executor, session_lock=LOCK)
        assert result["status"] == "completed" and result["synthetic"] is False
        assert result["speed"]["qualifies"] is True and result["quality"]["tools"]["score"] == 1.0
        record = store.results(result["attempt_id"])
        assert len(record["samples"]) == 4 and record["attempt"]["state"] == "completed"
    # The next call uses the next candidate index, never an existing directory.
    assert executor._next == 2
    fresh = ContainerExecutor(session, None, runner=runner, output_root=tmp_path / "out", deadline_epoch=1e12,
                              template_hashes={})
    assert fresh._next == 2


def ruler_candidates(session):
    """Candidate pairs whose only benchmark is the RULER selection ruler-kv-1 declared (LIVE-013)."""
    facts = asset_metadata(session, reader=metadata_reader(TEMPLATE))
    baseline = deterministic_schedule(session)[0]
    config = session_module.run_config_for(session, None, baseline, benchmarks=(live_ruler_selection(),))
    meta = facts[config.model.sha256]
    return config, session_module.to_manifest(config, template_hash=meta["template_hash"],
                                              scorer_revision="s" * 8, environment_hash="e" * 8,
                                              block_count=meta.get("block_count"))


def test_executor_rebuild_restores_benchmark_options_byte_for_byte(tmp_path):
    """LIVE-013: the candidate the executor hands the runner must carry the options the session declared.

    The campaign in artifacts/container-campaign/ruler-kv-1 declared ``ruler_output_tokens: 512`` and its
    candidates/000-baseline/config.json recorded ``options: {}``; RULER then capped ``vt`` at upstream's 30
    tokens, so the identical item (item_seed 7683617250397783352) scored 0.2 there and 1.0 with 180 output
    tokens in the standalone run artifacts/container-pilot/suite-q4-ruler-long-1.
    """
    session = small_session(tmp_path)
    config, manifest = ruler_candidates(session)
    executor = ContainerExecutor(session, None, runner=None, output_root=tmp_path / "out", deadline_epoch=1e12,
                                 template_hashes={})
    executor.register(config, manifest)
    rebuilt = executor.config_for(manifest)
    assert rebuilt == config
    assert [item.benchmark_id for item in rebuilt.benchmarks] == ["ruler"]
    assert dict(rebuilt.benchmarks[0].options)["ruler_output_tokens"] == 512
    # Byte-for-byte: session selection -> manifest task -> executor rebuild -> the config.json the runner writes.
    declared = canonical_json(LIVE_RULER_BENCHMARK["options"])
    assert (canonical_json(config.benchmarks[0].options) == canonical_json(manifest.tasks[0].options)
            == canonical_json(rebuilt.benchmarks[0].options) == declared)
    written = json.loads(canonical_json(rebuilt.model_dump(mode="json")))["benchmarks"][0]["options"]
    assert written == {"tasks": ["niah_multikey_3", "vt"], "lengths": [16384, 32768, 65536, 131072],
                       "ruler_output_tokens": 512}
    # A holdout rebuild of the same manifest carries the holdout selection's own (empty) options, unchanged.
    held = plan_holdout_candidates([manifest], holdout_selections(session))[0]
    assert dict(executor.config_for(held).benchmarks[0].options) == {}


def test_executor_refuses_a_candidate_whose_benchmark_options_were_dropped(tmp_path, monkeypatch):
    """The guard that makes LIVE-013 unrepeatable: a rebuild that loses options aborts, it never runs."""
    session = small_session(tmp_path)
    config, manifest = ruler_candidates(session)
    executor = ContainerExecutor(session, None, runner=None, output_root=tmp_path / "out", deadline_epoch=1e12,
                                 template_hashes={})
    executor.register(config, manifest)
    assert executor.config_for(manifest).benchmarks[0].options  # sane before the regression is injected

    def pre_live_013(tasks):
        """The rebuild exactly as it was at 87f4965: every option silently dropped."""
        return tuple(BenchmarkSelection(benchmark_id=item.suite, revision=item.revision, task_ids=item.task_ids,
                                        split=item.split, seed=item.fixture_seed) for item in tasks)

    monkeypatch.setattr(campaign_module, "_benchmarks_from", pre_live_013)
    with pytest.raises(BenchmarkOptionsNotCarried, match="refusing to measure a configuration") as raised:
        executor.config_for(manifest)
    assert raised.value.abort_campaign is True
    assert type(raised.value).__name__ in campaign_module.ABORTING_ERROR_TYPES
    assert "ruler_output_tokens" in str(raised.value)  # the report says which setting differed


def test_options_carried_guard_names_a_selection_it_cannot_serialise():
    """A future option type that JSON cannot express is a named abort, never a silent difference."""
    class Task:
        suite, options = "ruler", {"lengths": {16384, 32768}}  # a set: JSON cannot carry it

    with pytest.raises(BenchmarkOptionsNotCarried, match="ruler declares options that cannot be serialised"):
        campaign_module._assert_options_carried((Task(),), (Task(),))
    with pytest.raises(BenchmarkOptionsNotCarried, match="cannot be compared"):
        campaign_module._assert_options_carried((Task(),), ())


def test_benchmarks_rebuild_names_options_the_candidate_cannot_accept():
    """If the two option validators ever drift apart, the campaign says which benchmark it could not carry."""
    class Task:
        suite, revision, split, fixture_seed = "ruler", "r", "development", 42
        task_ids = ("ruler/vt/32768",)
        options = {f"k{index}": index for index in range(17)}  # more options than a selection may carry

    with pytest.raises(BenchmarkOptionsNotCarried, match="ruler declares options the candidate cannot accept"):
        campaign_module._benchmarks_from((Task(),))


def test_manifests_differing_only_in_a_benchmark_option_are_never_pooled(tmp_path):
    """Two candidates differing only in an option are different experiments and cannot share a registration."""
    session = small_session(tmp_path)
    capped_config, capped = ruler_candidates(session)
    raw = json.loads(canonical_json(LIVE_RULER_BENCHMARK))
    raw["options"]["ruler_output_tokens"] = 30
    facts = asset_metadata(session, reader=metadata_reader(TEMPLATE))
    uncapped_config = session_module.run_config_for(
        session, None, deterministic_schedule(session)[0],
        benchmarks=(BenchmarkSelection.model_validate_json(canonical_json(raw)),))
    uncapped = session_module.to_manifest(
        uncapped_config, template_hash=facts[uncapped_config.model.sha256]["template_hash"],
        scorer_revision="s" * 8, environment_hash="e" * 8,
        block_count=facts[uncapped_config.model.sha256].get("block_count"))
    assert capped.fingerprint() != uncapped.fingerprint()
    assert capped_config.fingerprint() != uncapped_config.fingerprint()
    executor = ContainerExecutor(session, None, runner=None, output_root=tmp_path / "out", deadline_epoch=1e12,
                                 template_hashes={})
    executor.register(capped_config, capped)
    # analysis.config_key deliberately ignores task selections, so the two share a key; the executor refuses
    # to pool them rather than run one candidate's options under the other's registration.
    assert config_key(capped) == config_key(uncapped)
    with pytest.raises(ValueError, match="different container config"):
        executor.register(uncapped_config, uncapped)
    assert canonical_json(executor.config_for(capped).benchmarks[0].options) == canonical_json(capped.tasks[0].options)


def test_indexed_artifact_read_is_hash_checked_and_bounded(tmp_path):
    session = small_session(tmp_path)
    (config, _), *_ = candidates(session)
    runner = FakeRunner(FakeClock(), policy=policy_for(session))
    run_dir = tmp_path / "run"
    runner.run(config, run_dir, remaining_budget_seconds=100, lease_held=True)
    content = read_indexed_artifact(run_dir, "evaluation.json", max_bytes=1 << 20)
    assert json.loads(content)["samples"]
    with pytest.raises(ValueError, match="exceeds"):
        read_indexed_artifact(run_dir, "evaluation.json", max_bytes=16)
    with pytest.raises(ValueError, match="not in the artifact index"):
        read_indexed_artifact(run_dir, "secrets.json", max_bytes=1 << 20)
    (run_dir / "evaluation.json").write_bytes(content + b" ")
    with pytest.raises(ValueError, match="does not match"):
        read_indexed_artifact(run_dir, "evaluation.json", max_bytes=1 << 20)
    (run_dir / "artifact-index.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="schema 1"):
        read_indexed_artifact(run_dir, "result.json", max_bytes=1 << 20)


def test_run_campaign_with_container_executor_persists_attempts_and_analysis(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path)
    outcome, runner, clock = run_tune(tmp_path, session)
    out = tmp_path / "out"
    campaign = outcome["campaign"]
    ledger = read_ledger(out)
    assert campaign["campaign_id"] == ledger["campaign_identity"] == campaign_identity(
        [manifest for _, manifest in candidates(session)], policy_for(session), holdout_selections(session))
    assert development_labels(runner) == ["baseline", "kv-q8_0-q8_0"]
    assert [row["status"] for row in campaign["results"]] == ["completed", "completed"]
    assert campaign["results"][1]["comparison_family"] == "kv" and campaign["results"][0]["comparison_family"] == "reference"
    assert all(row["synthetic"] is False and row["model_evaluated"] is True for row in campaign["results"])
    assert campaign["results"][1]["quality"]["tools"]["comparison"]["status"] in {"inconclusive", "noninferior"}
    assert outcome["stop_reason"] == "complete" and outcome["summary"]["abort_campaign"] is False
    with Store(out) as store:
        attempts = store.attempts_in_insertion_order()
        assert [row["state"] for row in attempts] == ["completed", "completed", "completed"]  # 2 + the holdout
        for row in attempts[:2]:  # a holdout attempt is analysed as part of its development row, not on its own
            record = store.results(row["id"])
            assert sum(event["kind"] == "evidence" for event in record["events"]) == 1
            assert any(event["kind"] == "analysis" for event in record["events"])
            names = {name for (name,) in store.db.execute("SELECT name FROM artifacts WHERE attempt_id=?", (row["id"],))}
            assert {"result.json", "settings-evidence.json", "stages.jsonl", "raw.json", "evidence.json"} <= names
            assert record["manifest"]["backend"]["engine"] == "llama.cpp" and record["manifest"]["mode"] == "live"
        # Coding is not declared in this session, so it is UNMEASURED rather than failed: the baseline survives
        # screening on the categories it did declare and is re-run on the held-out items. (This used to be a
        # recorded gap: an undeclared category failed every candidate, so no session could reach a holdout.)
        held = [record for record in (store.results(row["id"]) for row in attempts)
                if "holdout" in json.dumps(record["manifest"]["tasks"])]
        assert len(held) == 1 and development_labels(runner) == ["baseline", "kv-q8_0-q8_0"]
        assert [call["label"] for call in runner.calls][-1] == "baseline-holdout"
        assert campaign["results"][0]["holdout_passed"] is True
        assert campaign["results"][0]["eligibility"] == {"eligible": True, "reasons": [],
                                                         "unmeasured_categories": ["coding"]}
    for name in ("session.json", "session-config.json", "session-summary.json", "reports/campaign.json",
                 "reports/report.md", "reports/report.html", "reports/session.json", "reports/session.md",
                 "reports/session.html", "candidates/000-baseline/result.json", "candidates/001-kv-q8_0-q8_0/result.json"):
        assert (out / name).is_file(), name
    summary = json.loads((out / "session-summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "complete" and summary["attempts"] == 3  # 2 development + 1 holdout
    assert summary["winner"] == campaign["results"][0]["attempt_id"]  # validated on what its holdout covered
    assert summary["recommendation"]["best_quality"]["evidence"].startswith("validated on retrieval holdout only")
    # Setup time (metadata reads) was charged: the durable checkpoint includes it and the deadline is fixed.
    assert ledger["deadline_utc"] > ledger["started_utc"] and ledger["wall_seconds"] == 3600
    with pytest.raises(ValueError, match="use resume"):
        run_tune(tmp_path, session)


def test_session_deadline_is_enforced_with_wall_clock_and_aborts_dispatch(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path, quantizations=("Q4_K_M", "Q6_K"))
    (config, manifest), *_ = candidates(session)
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session))
    executor = ContainerExecutor(session, None, runner=runner, output_root=tmp_path / "x",
                                 deadline_epoch=clock.wall() + 700, template_hashes={}, clock=clock, wall=clock.wall)
    executor.register(config, manifest)
    with pytest.raises(SessionDeadlineExceeded) as info:
        executor(manifest, timeout_seconds=900)
    assert info.value.abort_campaign is True and runner.calls == [] and executor.deadline_hits == 1
    # Through the controller: the first candidate consumes the wall, the second is refused, nothing else runs.
    outcome, runner, clock = run_tune(tmp_path, session, script={"baseline": {"seconds": 3000.0}})
    assert development_labels(runner) == ["baseline"]
    assert outcome["stop_reason"] == "deadline"
    statuses = [(row["status"], row.get("abort_campaign", False)) for row in outcome["campaign"]["results"]]
    assert statuses == [("completed", False), ("failed", True)]
    with Store(tmp_path / "out") as store:
        attempts = store.attempts_in_insertion_order()
        assert [row["state"] for row in attempts] == ["completed", "failed"]
        errors = [json.loads(event["payload"]) for event in store.results(attempts[1]["id"])["events"]
                  if event["kind"] == "error"]
        assert errors[0]["type"] == "SessionDeadlineExceeded"
    assert len(outcome["report"]["rows"]) == 2 and "SessionDeadlineExceeded" in outcome["report"]["rows"][1]["state"]


def test_resume_honors_persisted_deadline_and_does_not_refund_in_flight_reservation(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path, quantizations=("Q4_K_M", "Q6_K"))  # baseline, kv, weights
    clock = FakeClock()
    with pytest.raises(KeyboardInterrupt):
        run_tune(tmp_path, session, clock=clock, script={"kv-q8_0-q8_0": {"raise": KeyboardInterrupt()}})
    out = tmp_path / "out"
    ledger = read_ledger(out)
    with Store(out) as store:
        attempts = store.attempts_in_insertion_order()
        assert [row["state"] for row in attempts] == ["completed", "cancelled"]
        charged_before = store.campaign_state(ledger["campaign_identity"])["elapsed_seconds"]
        assert charged_before >= 0  # controller.BudgetTracker charges the process clock, not the fake clock
    clock.now += 2600  # downtime between interrupt and resume is never refunded
    runner = FakeRunner(clock, policy=policy_for(session))
    outcome = resume(out, runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall)
    deadline_epoch = session_module._epoch(ledger["deadline_utc"])
    assert development_labels(runner) == ["kv-q8_0-q8_0"]
    assert runner.calls[0]["budget"] == pytest.approx(deadline_epoch - (clock.wall() - 120), abs=1)
    assert runner.calls[0]["budget"] < 900  # the persisted deadline, not the candidate wall, bounded the run
    assert outcome["stop_reason"] == "deadline"  # the third candidate no longer fits
    with Store(out) as store:
        attempts = store.attempts_in_insertion_order()
        assert [row["state"] for row in attempts] == ["completed", "cancelled", "completed", "failed"]
        assert attempts[2]["parent_id"] == attempts[1]["id"]
        assert store.campaign_state(ledger["campaign_identity"])["elapsed_seconds"] >= charged_before
        assert store.results(attempts[0]["id"])["events"][-1]["kind"] == "analysis"
    assert read_ledger(out) == ledger  # the ledger is never rewritten
    # After the deadline a resume only writes reports.
    clock.now += 10_000
    quiet = FakeRunner(clock, policy=policy_for(session))
    again = resume(out, runner=quiet, session_lock=LOCK, clock=clock, wall=clock.wall)
    assert quiet.calls == [] and again["stop_reason"] == "deadline"
    assert len(again["report"]["rows"]) == 4 and (out / "reports" / "session.md").is_file()
    # Policy drift or a different session config fail closed instead of mixing evidence.
    raw = json.loads((out / "session-config.json").read_text(encoding="utf-8"))
    raw["budgets"]["candidate_wall_seconds"] = 800
    (out / "session-config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="policy changed"):
        resume(out, runner=quiet, session_lock=LOCK, clock=clock, wall=clock.wall)


def test_abort_campaign_from_runner_stops_the_campaign(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path, quantizations=("Q4_K_M", "Q6_K"))
    outcome, runner, clock = run_tune(tmp_path, session, script={"baseline": {"state": "cleanup-uncertain",
                                                                             "stage": "cleanup",
                                                                             "reasons": ["owned container remains"]}})
    assert development_labels(runner) == ["baseline"]
    assert outcome["stop_reason"] == "abort" and outcome["summary"]["abort_campaign"] is True
    assert session_module._exit_code(outcome) == 4
    results = outcome["campaign"]["results"]
    assert len(results) == 1 and results[0]["status"] == "failed" and results[0]["abort_campaign"] is True
    assert "cleanup uncertain" in results[0]["error"]
    with Store(tmp_path / "out") as store:
        assert [row["state"] for row in store.attempts_in_insertion_order()] == ["failed"]
    with pytest.raises(CampaignAbort):
        raise CampaignAbort("x")
    assert CampaignAbort.abort_campaign is True


def test_template_hash_mismatch_withholds_context_claim(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path, kv_pairs=(("f16", "f16"),))
    served = "{{ messages | tojson }}"  # what the server actually applies differs from the GGUF header
    outcome, runner, clock = run_tune(tmp_path, session, template=served)
    row = outcome["campaign"]["results"][0]
    assert row["actual_context_verified"] is False and row["status"] == "completed"
    assert any(item.startswith("retrieval_context_unverified") for item in row["context_accounting"]["errors"])
    assert "actual_context_verified_required_for_validation" in row["eligibility"]["reasons"]
    with Store(tmp_path / "out") as store:
        attempt = store.attempts_in_insertion_order()[0]["id"]
        raw = json.loads((store.root / "raw" / attempt / "raw.json").read_text(encoding="utf-8"))
        assert raw["template_check"]["matches"] is False and raw["warnings"][0].startswith("template_hash_mismatch")
        evidence = [json.loads(e["payload"]) for e in store.results(attempt)["events"] if e["kind"] == "evidence"][0]
        assert evidence["actual_context_verified"] is False
    report_row = outcome["report"]["rows"][0]
    assert report_row["context_verified"] is False and report_row["template_check"]["matches"] is False
    assert outcome["report"]["recommendation"]["best_quality"]["evidence"] == "no qualifying candidate"


def test_failed_candidate_is_recorded_failed_not_skipped(tmp_path, isolated_gpu_lock):
    session = small_session(tmp_path, quantizations=("Q4_K_M", "Q6_K"))
    reason = "<script>alert('rejected')</script>"
    outcome, runner, clock = run_tune(tmp_path, session, script={
        "kv-q8_0-q8_0": {"state": "rejected", "stage": "admit", "reasons": [reason]},
        "weights-q6_k": {"state": "timeout", "stage": "ready", "reasons": ["inference was not healthy within 300s"]}})
    assert development_labels(runner) == ["baseline", "kv-q8_0-q8_0", "weights-q6_k"]
    results = outcome["campaign"]["results"]
    assert [row["status"] for row in results] == ["completed", "failed", "failed"]
    assert outcome["campaign"]["skipped"] == [] and outcome["stop_reason"] == "complete"
    assert "rejected at stage admit" in results[1]["error"] and reason in results[1]["error"]
    with Store(tmp_path / "out") as store:
        attempts = store.attempts_in_insertion_order()
        assert [row["state"] for row in attempts] == ["completed", "failed", "failed", "completed"]  # + holdout
        names = {name for (name,) in store.db.execute("SELECT name FROM artifacts WHERE attempt_id=?",
                                                       (attempts[1]["id"],))}
        assert "result.json" in names and "raw.json" not in names  # copied before the failure was raised
    rows = outcome["report"]["rows"]
    assert [row["state"] for row in rows] == ["completed", "failed (container: rejected at admit)",
                                             "failed (container: timeout at ready)", "completed"]  # + holdout
    assert reason in rows[1]["eligibility_reasons"]
    html = (tmp_path / "out" / "reports" / "session.html").read_text(encoding="utf-8")
    assert reason not in html and "&lt;script&gt;" in html


def test_tune_and_resume_cli_paths_are_inert_without_policy(tmp_path, capsys):
    session = small_session(tmp_path)
    config_path = tmp_path / "session.json"
    config_path.write_text(session.model_dump_json(), encoding="utf-8")
    out = tmp_path / "out"
    calls = []

    def factory(args, policy):
        calls.append(args)
        pytest.fail("a runner must not be constructed without a live policy")

    missing = str(tmp_path / "no-policy.json")
    assert session_module.main(["tune", "--config", str(config_path), "--output", str(out), "--policy", missing],
                               runner_factory=factory) == 2
    assert "forbidden" in capsys.readouterr().err
    assert not out.exists() and calls == []
    assert session_module.main(["resume", "--output", str(out), "--policy", missing], runner_factory=factory) == 2
    assert not out.exists() and calls == []
    denied = tmp_path / "denied.json"
    denied.write_text(json.dumps({"allow_inference": True}), encoding="utf-8")  # containers still forbidden
    assert session_module.main(["tune", "--config", str(config_path), "--output", str(out), "--policy", str(denied)],
                               runner_factory=factory) == 2
    assert not out.exists()
    with pytest.raises(SystemExit):
        session_module.main(["tune", "--output", str(out)])  # --config or --model is required


def test_tune_cli_runs_with_an_injected_runner_and_writes_the_session(tmp_path, isolated_gpu_lock, capsys):
    session = small_session(tmp_path)
    config_path = tmp_path / "session.json"
    config_path.write_text(session.model_dump_json(), encoding="utf-8")
    policy_path = tmp_path / "runtime-policy.json"
    policy_path.write_text(json.dumps({"allow_model_operations": True, "allow_inference": True,
                                       "allow_container_execution": True}), encoding="utf-8")
    clock = FakeClock()
    runners = []

    def factory(args, policy):
        runner = FakeRunner(clock, policy=policy)
        runners.append(runner)
        return runner

    out = tmp_path / "cli-out"
    code = session_module.main(["tune", "--config", str(config_path), "--output", str(out), "--policy",
                                str(policy_path), "--budget-seconds", "3000"], runner_factory=factory)
    printed = json.loads(capsys.readouterr().out)
    assert code == 0 and printed["stop_reason"] == "complete" and printed["attempts"] == 3  # 2 + the holdout
    assert read_ledger(out)["wall_seconds"] == 3000 and development_labels(runners[0]) == ["baseline",
                                                                                                    "kv-q8_0-q8_0"]
    assert session_module.main(["resume", "--output", str(out), "--policy", str(policy_path)],
                               runner_factory=factory) == 0
    assert runners[1].calls == []  # both candidates already completed under the same policy
    # REV-C1-01: resume --image-bundle on a session that ran without one is refused before any candidate runs.
    from test_containers_session import make_bundle
    bundle_file = tmp_path / "image-bundle.json"
    bundle_file.write_text(make_bundle().model_dump_json(), encoding="utf-8")
    assert session_module.main(["resume", "--output", str(out), "--policy", str(policy_path), "--image-bundle",
                                str(bundle_file)], runner_factory=factory) == 2
    assert "image bundle" in capsys.readouterr().err and runners[2].calls == []


def test_export_preset_refuses_a_campaign_with_an_unmeasured_category(tmp_path, isolated_gpu_lock):
    """The session names a winner on what it measured and says what it did not; a validated-preset EXPORT still
    demands every category, so an unmeasured one can never be certified."""
    session = small_session(tmp_path)
    outcome, runner, clock = run_tune(tmp_path, session)
    campaign = outcome["campaign"]
    baseline = campaign["results"][0]["attempt_id"]
    policy = policy_for(session)
    with Store(tmp_path / "out") as store:
        for row in campaign["results"]:
            manifest = store.results(row["attempt_id"])["manifest"]
            destination = tmp_path / f"preset-{row['attempt_id']}.json"
            with pytest.raises(ValueError, match="preset export refused"):
                export_preset(manifest, row, policy, destination, store=store, campaign=campaign,
                              baseline_attempt_id=baseline, allow_self_reference=True, resamples=100)
            with pytest.raises(ValueError, match="preset export refused"):
                export_provisional_preset(manifest, row, policy, destination, store=store, campaign=campaign,
                                          baseline_attempt_id=baseline, allow_self_reference=True, resamples=100)
            assert not destination.exists()
    assert campaign["winner"] == baseline
    for preset in campaign["best_presets"].values():
        assert preset["unmeasured_categories"] == ["coding"]
        assert preset["holdout_validated_categories"] == ["retrieval"]
        assert preset["holdout_uncovered_categories"] == ["coding", "tools"]


def test_registered_manifests_carry_live_mode_and_environment(tmp_path):
    session = small_session(tmp_path)
    for config, manifest in candidates(session):
        assert manifest.mode == RunMode.LIVE and manifest.environment_hash == environment_record()["sha256"]
        assert manifest.annotations["config_fingerprint"] == config.fingerprint()
        assert isinstance(manifest.tasks[0], TaskSelection)
    assert hashlib.sha256(canonical_json({"a": 1}).encode()).hexdigest()  # canonical_json stays the shared hasher
    assert isinstance(SessionLock(), SessionLock) and Path(tmp_path).exists()


def test_tune_cli_derives_a_session_from_weights_and_writes_session_config(tmp_path, isolated_gpu_lock, capsys):
    """AM-1 item 1: `tune --model <dir>` needs no hand-written session; the derived one is saved for reproduction."""
    from test_containers_session import gguf_bytes
    weights = tmp_path / "Qwen3.8-27B-GGUF"
    weights.mkdir()
    for quant in ("Q4_K_M", "Q6_K"):
        (weights / f"Qwen3.8-27B-{quant}.gguf").write_bytes(gguf_bytes(quant, n_ctx_train=32768, nextn=0))
    policy_path = tmp_path / "runtime-policy.json"
    policy_path.write_text(json.dumps({"allow_model_operations": True, "allow_inference": True,
                                       "allow_container_execution": True}), encoding="utf-8")
    clock = FakeClock()
    runners = []

    def factory(args, policy):
        runners.append(FakeRunner(clock, policy=policy))
        return runners[-1]

    out = tmp_path / "derived"
    code = session_module.main(["tune", "--model", str(weights), "--output", str(out), "--policy", str(policy_path),
                                "--budget-seconds", "3000"], runner_factory=factory)
    printed = json.loads(capsys.readouterr().out)
    assert code == 0 and printed["stop_reason"] == "complete"
    derived = session_module.read_session_config(out / "session-config.json")
    assert derived.session_id == "tune-qwen3.8-27b" and derived.budgets.wall_seconds == 3000
    assert derived.search.quantizations == ("Q4_K_M", "Q6_K") and derived.search.ctx_tiers == (8192, 32768)
    assert development_labels(runners[0]) == ["baseline", "kv-q8_0-q8_0", "kv-q4_0-q4_0", "weights-q6_k",
                                                            "reasoning-on", "ctx-32768"]
    ledger = read_ledger(out)
    assert ledger["wall_seconds"] == 3000 and len(ledger["template_hashes"]) == 2
    assert printed["attempts"] == 7 and (out / "reports" / "session.md").is_file()  # 6 + the baseline's holdout
    with pytest.raises(SystemExit):  # --search accepts only the default preset
        session_module.main(["tune", "--model", str(weights), "--output", str(tmp_path / "x"), "--policy",
                             str(policy_path), "--search", "wide", "--unknown"], runner_factory=factory)
    assert session_module.main(["tune", "--model", str(weights), "--output", str(tmp_path / "x"), "--policy",
                                str(policy_path), "--search", "wide"], runner_factory=factory) == 2
    assert "unknown search preset" in capsys.readouterr().err and not (tmp_path / "x").exists()


def test_resume_rebuilds_candidates_with_the_tune_image_bundle_and_refuses_a_missing_or_different_one(
        tmp_path, isolated_gpu_lock):
    """REV-C1-01: the bundle given to tune is pinned in the ledger (path + image IDs + config fingerprints);
    resume rebuilds byte-identical candidates from it and fails closed when it is missing or differs."""
    from test_containers_session import make_bundle
    session = small_session(tmp_path, quantizations=("Q4_K_M", "Q6_K"))  # baseline, kv, weights
    bundle_path = tmp_path / "prep" / "image-bundle.json"
    bundle_path.parent.mkdir()
    bundle_path.write_text(make_bundle("1" * 64).model_dump_json(), encoding="utf-8")
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session), script={"kv-q8_0-q8_0": {"raise": KeyboardInterrupt()}})
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        tune(session, out, runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall,
             metadata_reader=metadata_reader(), bundle_path=bundle_path)
    tune_image = runner.calls[0]["config"].inference_image.image_id
    assert tune_image == "sha256:" + "1" * 64
    ledger = read_ledger(out)
    assert ledger["image_bundle"] == str(bundle_path.resolve())
    assert ledger["image_ids"] == {"inference": tune_image, "evaluator": "sha256:" + "b" * 64, "worker": None}
    assert ledger["config_fingerprints"] == [pair[0].fingerprint() for pair in build_candidates(
        session, session_module.read_image_bundle(bundle_path), deterministic_schedule(session),
        asset_metadata(session, reader=metadata_reader()), scorer_revision=builtin_registry().digest(),
        environment_hash=environment_record()["sha256"])]
    assert len(ledger["config_fingerprints"]) == 3
    # A different bundle on resume is refused before any candidate runs.
    other = FakeRunner(clock, policy=policy_for(session))
    with pytest.raises(ValueError, match="image"):
        resume(out, runner=other, session_lock=LOCK, clock=clock, wall=clock.wall, bundle=make_bundle("2" * 64))
    assert other.calls == []
    # Without --image-bundle the ledger path is used and the resumed candidate carries the same inference image.
    clock.now += 10
    second = FakeRunner(clock, policy=policy_for(session))
    outcome = resume(out, runner=second, session_lock=LOCK, clock=clock, wall=clock.wall)
    assert development_labels(second) == ["kv-q8_0-q8_0", "weights-q6_k"]
    assert second.calls[0]["config"].inference_image.image_id == tune_image
    assert second.calls[0]["config"].fingerprint() == ledger["config_fingerprints"][1]
    assert outcome["stop_reason"] == "complete"
    # The bundle file removed: resume refuses and runs nothing; an explicit bundle with the same IDs still works.
    bundle_path.unlink()
    third = FakeRunner(clock, policy=policy_for(session))
    with pytest.raises(ValueError, match="image bundle"):
        resume(out, runner=third, session_lock=LOCK, clock=clock, wall=clock.wall)
    assert third.calls == []
    again = resume(out, runner=third, session_lock=LOCK, clock=clock, wall=clock.wall, bundle=make_bundle("1" * 64))
    assert third.calls == [] and again["stop_reason"] == "complete"  # everything already ran
    # A session that used no bundle refuses one on resume (the inference image would change).
    plain = small_session(tmp_path / "plain")
    run_tune(tmp_path / "plain", plain, clock=clock)
    with pytest.raises(ValueError, match="image"):
        resume(tmp_path / "plain" / "out", runner=third, session_lock=LOCK, clock=clock, wall=clock.wall,
               bundle=make_bundle("1" * 64))
    assert read_ledger(tmp_path / "plain" / "out")["image_ids"] is None


def test_resume_refuses_an_edited_session_config_and_tune_refuses_a_stale_one(tmp_path, isolated_gpu_lock):
    """REV-C1-02/05: session-config.json is pinned by the ledger (sha256 + config fingerprints)."""
    session = small_session(tmp_path, quantizations=("Q4_K_M", "Q6_K"))
    clock = FakeClock()
    with pytest.raises(KeyboardInterrupt):
        run_tune(tmp_path, session, clock=clock, script={"kv-q8_0-q8_0": {"raise": KeyboardInterrupt()}})
    out = tmp_path / "out"
    ledger = read_ledger(out)
    config_path = out / "session-config.json"
    assert ledger["session_config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    original = config_path.read_bytes()
    raw = json.loads(original)
    raw["base"]["limits"]["max_foreign_vram_mib"] = raw["base"]["limits"]["max_foreign_vram_mib"] + 1  # not in policy
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    quiet = FakeRunner(clock, policy=policy_for(session))
    with pytest.raises(ValueError, match="session-config.json"):
        resume(out, runner=quiet, session_lock=LOCK, clock=clock, wall=clock.wall)
    assert quiet.calls == []
    config_path.write_bytes(original)
    outcome = resume(out, runner=quiet, session_lock=LOCK, clock=clock, wall=clock.wall)
    assert development_labels(quiet) == ["kv-q8_0-q8_0", "weights-q6_k"]
    assert outcome["stop_reason"] == "complete"
    # tune into a directory holding another session's config refuses instead of adopting the foreign file.
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "session-config.json").write_bytes(original)
    foreign = small_session(tmp_path / "f", session_id="session-other")
    runner = FakeRunner(clock, policy=policy_for(foreign))
    with pytest.raises(ValueError, match="session-config.json"):
        tune(foreign, stale, runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall,
             metadata_reader=metadata_reader())
    assert runner.calls == [] and not (stale / "session.json").exists()
    assert (stale / "session-config.json").read_bytes() == original
    # An identical pre-written config (the --model path writes it before tune) is accepted.
    same = tmp_path / "same"
    same.mkdir()
    session_module.write_exclusive_json(same / "session-config.json", session.model_dump(mode="json"))
    run_tune(tmp_path, session, clock=clock, output=same)
    assert read_ledger(same)["session_config_sha256"] == hashlib.sha256(
        (same / "session-config.json").read_bytes()).hexdigest()


def test_stop_reason_is_candidates_only_when_the_cap_truncates_the_schedule(tmp_path, isolated_gpu_lock):
    """`candidates` = the count cap cut the schedule (or the controller's admit count reached it); a schedule
    exactly at the cap, and history rows of an interrupted-then-retried candidate, are not truncations."""
    from llmbench.containers.proposals import full_schedule
    budgets = {"wall_seconds": 3600, "holdout_reserve_seconds": 900, "cleanup_reserve_seconds": 60,
               "max_candidates": 3, "candidate_wall_seconds": 900, "max_validation_candidates": 2}
    four = small_session(tmp_path / "four", quantizations=("Q4_K_M", "Q6_K"), reasoning=("off", "on"),
                         budgets=budgets)  # baseline, kv, weights, reasoning-on
    assert len(full_schedule(four)) == 4 and len(deterministic_schedule(four)) == 3
    outcome, runner, _ = run_tune(tmp_path / "four", four)
    assert development_labels(runner) == ["baseline", "kv-q8_0-q8_0", "weights-q6_k"]
    assert outcome["campaign"]["skipped"] == []  # the schedule was cut before the controller saw it
    assert outcome["stop_reason"] == "candidates" and session_module._exit_code(outcome) == 0
    three = small_session(tmp_path / "three", quantizations=("Q4_K_M", "Q6_K"), budgets=budgets)
    outcome, runner, _ = run_tune(tmp_path / "three", three)
    assert len(development_labels(runner)) == 3 and outcome["campaign"]["skipped"] == []
    assert outcome["stop_reason"] == "complete"
    # Interrupted then resumed under a cap of 4: the cancelled row plus its retry do not make it `candidates`.
    loose = dict(budgets, max_candidates=4)
    retry = small_session(tmp_path / "retry", quantizations=("Q4_K_M", "Q6_K"), budgets=loose)
    clock = FakeClock()
    with pytest.raises(KeyboardInterrupt):
        run_tune(tmp_path / "retry", retry, clock=clock, script={"kv-q8_0-q8_0": {"raise": KeyboardInterrupt()}})
    outcome = resume(tmp_path / "retry" / "out", runner=FakeRunner(clock, policy=policy_for(retry)),
                     session_lock=LOCK, clock=clock, wall=clock.wall)
    rows = outcome["report"]["rows"]
    assert len([row for row in rows if not row["label"].endswith("-holdout")]) == 4
    assert outcome["stop_reason"] == "complete"


def test_executor_propagates_the_original_error_when_the_candidate_record_cannot_be_stored(tmp_path):
    """REV-C1-11: a failing artifact sink on the exception path never masks the runner's exception."""
    session = small_session(tmp_path)
    (config, manifest), *_ = candidates(session)
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session), script={"baseline": {"raise": KeyboardInterrupt()}})
    executor = ContainerExecutor(session, None, runner=runner, output_root=tmp_path / "out",
                                 deadline_epoch=clock.wall() + 3600, template_hashes={}, clock=clock, wall=clock.wall)
    executor.register(config, manifest)
    events = []
    executor.bind_event_sink(events.append)

    def broken_sink(name, content):
        raise RuntimeError("artifact budget exhausted")

    executor.bind_artifact_sink(broken_sink)
    with pytest.raises(KeyboardInterrupt):
        executor(manifest, timeout_seconds=900)
    assert [event["event"] for event in events] == ["candidate-start", "candidate-record-not-stored", "candidate-end"]
    assert "artifact budget exhausted" in events[1]["reason"]
