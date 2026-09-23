import json
import os
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.containers import cli
from test_containers_session import isolated_gpu_lock

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "candidate.json"
DATA = Path(__file__).parent / "data"
_ = isolated_gpu_lock  # fixture re-export for pytest collection


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("pure commands must not start a process")
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)


@pytest.fixture
def prep(tmp_path):
    directory = tmp_path / "prep"
    directory.mkdir()
    (directory / "llama-server-help.txt").write_bytes((DATA / "llama-server-help-b11011.txt").read_bytes())
    (directory / "llama-server-version.txt").write_bytes((DATA / "llama-server-version-b11011.txt").read_bytes())
    return directory


def output_of(capsys):
    return json.loads(capsys.readouterr().out)


def test_validate_is_pure_and_reports_identity(capsys, tmp_path):
    assert cli.main(["validate", "--config", str(EXAMPLE)]) == 0
    data = output_of(capsys)
    assert data["valid"] and data["label"] == "q4-8k" and data["alias"] == "llmbench-" + data["fingerprint"][:12]
    assert data["execution_performed"] is False
    broken = tmp_path / "broken.json"
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["engine"]["extra_args"] = ["--tools", "all"]
    broken.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.main(["validate", "--config", str(broken)]) == 2
    assert "extra_args" in capsys.readouterr().err
    assert cli.main(["validate", "--config", str(tmp_path / "missing.json")]) == 2
    broken.write_text("{not json", encoding="utf-8")
    assert cli.main(["validate", "--config", str(broken)]) == 2


def test_capabilities_reports_the_saved_help(capsys, prep, tmp_path):
    assert cli.main(["capabilities", "--capabilities-dir", str(prep)]) == 0
    data = output_of(capsys)
    assert data["flags"] == 411 and data["build"] == "b11011-aa39d7a3e" and data["allowed_flags_missing"] == []
    assert cli.main(["capabilities", "--capabilities-dir", str(prep), "--config", str(EXAMPLE)]) == 0
    data = output_of(capsys)
    assert data["help_sha256_matches"] is True and data["findings"] == []
    help_file = prep / "llama-server-help.txt"
    help_file.write_bytes(help_file.read_bytes().replace(b"--fit [on|off]", b"--fut [on|off]"))
    assert cli.main(["capabilities", "--capabilities-dir", str(prep), "--config", str(EXAMPLE)]) == 2
    data = output_of(capsys)
    assert data["allowed_flags_missing"] == ["--fit"] and data["help_sha256_matches"] is False
    assert cli.main(["capabilities", "--capabilities-dir", str(tmp_path / "missing")]) == 2


def test_plan_prints_argv_and_compose_without_writing(capsys, prep, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["plan", "--config", str(EXAMPLE), "--capabilities-dir", str(prep)]) == 0
    data = output_of(capsys)
    assert data["execution_performed"] is False and data["server_argv"][:2] == ["--model", "/models/model.gguf"]
    assert data["compose"]["services"]["inference"]["command"] == data["server_argv"]
    assert data["compose"]["name"] == "llmbench-000000000000"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prep"]
    help_file = prep / "llama-server-help.txt"
    help_file.write_bytes(help_file.read_bytes() + b"\n--extra   changed help\n")
    assert cli.main(["plan", "--config", str(EXAMPLE), "--capabilities-dir", str(prep)]) == 2
    assert "help_sha256" in capsys.readouterr().err


@pytest.mark.parametrize("state, code", [("completed", 0), ("failed", 3), ("timeout", 3), ("cancelled", 3),
                                         ("rejected", 3), ("cleanup-uncertain", 4)])
def test_candidate_maps_terminal_states_to_exit_codes(capsys, tmp_path, state, code):
    seen = {}

    class Runner:
        def run(self, config, output, *, remaining_budget_seconds=None):
            seen.update(config=config, output=output, budget=remaining_budget_seconds)
            return SimpleNamespace(
                state=state, synthetic=True, attempt_id="a" * 32, failure_stage=None, failure_reasons=(),
                warnings=(), cleanup=SimpleNamespace(verified=state != "cleanup-uncertain"),
                abort_campaign=state == "cleanup-uncertain", elapsed_seconds=1.5, effective_settings_verified=False,
                actual_context_verified=False, speed={"minimum_native_tps": 69.2}, samples_total=4,
                model_dump=lambda mode: {"reports": {"candidate": "reports/candidate.md"}})

    def factory(args):
        seen["args"] = args
        return Runner()
    arguments = ["candidate", "--config", str(EXAMPLE), "--output", str(tmp_path / "out"), "--budget-seconds", "900"]
    assert cli.main(arguments, runner_factory=factory) == code
    data = output_of(capsys)
    assert data["state"] == state and data["abort_campaign"] is (code == 4) and data["minimum_native_tps"] == 69.2
    assert seen["budget"] == 900.0 and seen["config"].label == "q4-8k" and seen["output"] == str(tmp_path / "out")
    assert seen["args"].capabilities_dir == "artifacts/container-prep" and seen["args"].policy == "runtime-policy.json"


def test_candidate_input_errors_exit_two_without_a_runner(capsys, tmp_path):
    def factory(args):
        raise AssertionError("an invalid configuration must not construct a runner")
    assert cli.main(["candidate", "--config", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out")],
                    runner_factory=factory) == 2

    class Refusing:
        def run(self, config, output, **options):
            raise ValueError("candidate output directory must be new or empty")
    assert cli.main(["candidate", "--config", str(EXAMPLE), "--output", str(tmp_path)],
                    runner_factory=lambda args: Refusing()) == 2
    assert "new or empty" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["tune", "--config", str(EXAMPLE)])
    with pytest.raises(SystemExit):
        cli.main(["candidate", "--config", str(EXAMPLE)])


INFERENCE_REF = "ghcr.io/ggml-org/llama.cpp@sha256:" + "d" * 64
BASE_REF = "python:3.12-slim-bookworm@sha256:" + "a" * 64


def bundle_for(config, **changes):
    from llmbench.containers.config import ImageBundle
    inference = config.inference_image.model_dump()
    evaluator = {"role": "evaluator", "reference": "sha256:" + "c" * 64, "image_id": "sha256:" + "c" * 64,
                 "entrypoint": ["python", "-m", "llmbench.container_eval"]}
    return ImageBundle.model_validate_json(json.dumps({
        "prepared_utc": "2026-09-18T00:00:00+00:00", "inference": inference, "evaluator": evaluator,
        "help_sha256": inference["help_sha256"], "registry_digest": "1" * 64, **changes}))


def test_prepare_is_gated_by_allow_preparation_and_writes_bundle(capsys, tmp_path, monkeypatch):
    from llmbench.config import RunMode
    from llmbench.containers import image_plan
    from llmbench.containers.config import read_run_config
    from llmbench.containers.executor import ComposeExecutor
    bundle = bundle_for(read_run_config(EXAMPLE))
    seen = {}

    def preparer(args):
        seen["args"] = args
        return bundle
    base = ["prepare", "--inference", INFERENCE_REF, "--evaluator-base", BASE_REF, "--output", str(tmp_path / "prep")]
    assert cli.main(base, preparer=preparer) == 0
    data = output_of(capsys)
    assert data["execution_performed"] is True and data["evaluator"] == bundle.evaluator.image_id
    assert data["worker"] is None and data["registry_digest"] == "1" * 64 and data["help_sha256"] == bundle.help_sha256
    assert seen["args"].policy == "runtime-policy.json" and seen["args"].worker_iidfile is None
    with pytest.raises(SystemExit):  # both digests are mandatory
        cli.main(["prepare", "--inference", INFERENCE_REF, "--output", str(tmp_path / "x")], preparer=preparer)
    # The default preparer is gated by the persistent policy before any executor or wheel build exists.
    denied = tmp_path / "denied.json"
    denied.write_text(json.dumps({"allow_inference": True, "allow_model_operations": True}), encoding="utf-8")
    assert cli.main([*base, "--policy", str(denied)]) == 2
    assert "container" in capsys.readouterr().err and not (tmp_path / "prep").exists()
    # With container permission it constructs a live, preparation-enabled, non-synthetic executor and hands the
    # validated inputs to prepare_images (scripted here); --wheel skips the wheel build entirely.
    allowed = tmp_path / "allowed.json"
    allowed.write_text(json.dumps({"allow_container_execution": True}), encoding="utf-8")
    captured = {}

    def fake_prepare(inputs, *, executor, artifact_dir, session_lock=None, **options):
        captured.update(inputs=inputs, executor=executor, artifact_dir=artifact_dir, lock=session_lock)
        return bundle

    def no_build(*args, **kwargs):
        raise AssertionError("the wheel must not be built when --wheel is given")
    monkeypatch.setattr(image_plan, "prepare_images", fake_prepare)
    monkeypatch.setattr(image_plan, "build_wheel", no_build)
    wheel = tmp_path / "localllmbench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"PK\x03\x04")
    assert cli.main([*base, "--policy", str(allowed), "--wheel", str(wheel)]) == 0
    assert output_of(capsys)["evaluator"] == bundle.evaluator.image_id
    executor = captured["executor"]
    assert isinstance(executor, ComposeExecutor) and executor.allow_preparation is True
    assert executor.synthetic is False and executor.mode == RunMode.LIVE and executor.session_lock == captured["lock"]
    assert captured["lock"].allow_container_execution and captured["artifact_dir"] == Path(tmp_path / "prep")
    assert captured["inputs"].inference == INFERENCE_REF and captured["inputs"].evaluator_base == BASE_REF
    assert captured["inputs"].wheel_path == wheel and captured["inputs"].lock_path is None
    executor._validate(("docker", "pull", INFERENCE_REF))  # preparation vocabulary is on for this executor only
    with pytest.raises(ValueError):
        ComposeExecutor(session_lock=captured["lock"], mode=RunMode.LIVE)._validate(("docker", "pull", INFERENCE_REF))
    # Tagged or malformed references never reach Docker: PrepareInputs refuses them and the CLI exits 2.
    for inference, evaluator_base in (("ghcr.io/ggml-org/llama.cpp:latest", BASE_REF),
                                      (INFERENCE_REF, "python:3.12-slim-bookworm"),
                                      (INFERENCE_REF, "node:24-bookworm-slim@sha256:" + "a" * 64)):
        captured.clear()
        assert cli.main(["prepare", "--inference", inference, "--evaluator-base", evaluator_base, "--output",
                         str(tmp_path / "never"), "--policy", str(allowed), "--wheel", str(wheel)]) == 2
        assert captured == {} and "digest" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


def test_candidate_image_bundle_mismatch_exits_two(capsys, tmp_path):
    from llmbench.containers.config import read_run_config
    config = read_run_config(EXAMPLE)
    calls = []

    class Runner:
        def run(self, config, output, *, remaining_budget_seconds=None):
            calls.append(output)
            return SimpleNamespace(
                state="completed", synthetic=True, attempt_id="a" * 32, failure_stage=None, failure_reasons=(),
                warnings=(), cleanup=SimpleNamespace(verified=True), abort_campaign=False, elapsed_seconds=1.0,
                effective_settings_verified=False, actual_context_verified=False, speed={}, samples_total=0,
                model_dump=lambda mode: {"reports": {}})

    def factory(args):
        return Runner()

    def write(name, bundle):
        path = tmp_path / name
        path.write_text(bundle.model_dump_json(), encoding="utf-8")
        return path
    matching = write("match.json", bundle_for(config))
    arguments = ["candidate", "--config", str(EXAMPLE), "--output", str(tmp_path / "out")]
    assert cli.main([*arguments, "--image-bundle", str(matching)], runner_factory=factory) == 0
    assert calls == [str(tmp_path / "out")] and output_of(capsys)["state"] == "completed"
    inference = {**config.inference_image.model_dump(), "image_id": "sha256:" + "9" * 64,
                 "reference": "sha256:" + "9" * 64}
    mismatched = write("mismatch.json", bundle_for(config, inference=inference))
    assert cli.main([*arguments, "--image-bundle", str(mismatched)], runner_factory=factory) == 2
    err = capsys.readouterr().err
    assert "differ from the prepared bundle" in err and "inference.image_id" in err and len(calls) == 1
    # A container-mode config must name exactly the bundle's evaluator image; the registry digest must agree too.
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["evaluator"] = {"mode": "container", "image": {"role": "evaluator", "reference": "sha256:" + "b" * 64,
                                                       "image_id": "sha256:" + "b" * 64,
                                                       "entrypoint": ["python", "-m", "llmbench.container_eval"]}}
    container = tmp_path / "container.json"
    container.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.main(["candidate", "--config", str(container), "--output", str(tmp_path / "c"), "--image-bundle",
                     str(matching)], runner_factory=factory) == 2
    assert "evaluator.image_id" in capsys.readouterr().err and len(calls) == 1
    raw["registry_digest"] = "0" * 64
    container.write_text(json.dumps(raw), encoding="utf-8")
    raw_bundle = json.loads(matching.read_text(encoding="utf-8"))
    raw_bundle["evaluator"] = raw["evaluator"]["image"]
    exact = tmp_path / "exact.json"
    exact.write_text(json.dumps(raw_bundle), encoding="utf-8")
    assert cli.main(["candidate", "--config", str(container), "--output", str(tmp_path / "c"), "--image-bundle",
                     str(exact)], runner_factory=factory) == 2
    assert "registry_digest" in capsys.readouterr().err and len(calls) == 1
    raw_bundle["registry_digest"] = "0" * 64
    exact.write_text(json.dumps(raw_bundle), encoding="utf-8")
    assert cli.main(["candidate", "--config", str(container), "--output", str(tmp_path / "c"), "--image-bundle",
                     str(exact)], runner_factory=factory) == 0
    assert len(calls) == 2
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert cli.main([*arguments, "--image-bundle", str(tmp_path / "broken.json")], runner_factory=factory) == 2
    assert cli.main([*arguments, "--image-bundle", str(tmp_path / "absent.json")], runner_factory=factory) == 2
    assert len(calls) == 2 and cli.main(arguments, runner_factory=factory) == 0  # the bundle stays optional
    capsys.readouterr()


def test_default_runner_is_constructed_inertly(tmp_path):
    # The candidate runner dispatches on the config's runtime and builds nothing until a candidate arrives; the
    # NVIDIA delegate it then builds is today's inert ContainerRunner, unchanged.
    from llmbench.containers.config import read_run_config
    from llmbench.containers.runner import ContainerRunner
    dispatch = cli._default_runner(SimpleNamespace(capabilities_dir=str(tmp_path), policy=str(tmp_path / "p.json")))
    assert dispatch.synthetic is False and dispatch.built == ()
    runner = dispatch.runner_for(read_run_config(EXAMPLE))
    assert type(runner) is ContainerRunner and dispatch.built == ("nvidia-container",)
    assert runner.synthetic is False and runner.executor is None and runner.policy_path == tmp_path / "p.json"
    assert runner.capabilities_dir == tmp_path and runner.policy is None


def test_run_py_is_a_thin_shim(capsys, monkeypatch):
    # Delegation behavior defines the shim; Windows interruption handling may add a few lines.
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setattr("sys.argv", ["run.py", "validate", "--config", str(EXAMPLE)])
    with pytest.raises(SystemExit) as finished:
        runpy.run_path(str(ROOT / "run.py"), run_name="__main__")
    assert finished.value.code == 0 and output_of(capsys)["valid"] is True


# ---- stage-2 integration wiring (tune/resume registration, broker factory and coding hook defaults) -------------

def test_tune_and_resume_are_registered_and_dispatch_to_session(capsys, tmp_path, monkeypatch):
    from llmbench.containers import session
    import argparse
    commands = next(a for a in cli.parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
    assert {"validate", "capabilities", "plan", "candidate", "prepare", "tune", "resume"} <= set(commands)
    seen = []

    def fake_tune(args, **options):
        seen.append(("tune", args, options))
        return 3

    def fake_resume(args, **options):
        seen.append(("resume", args, options))
        return 4

    real_reader = cli.read_run_config

    def never(path):
        """A session config must never be read as a run config; the --model path legitimately reads its base."""
        if str(path) == str(tmp_path / "session.json"):
            raise AssertionError("tune/resume take a session config; the run-config reader must not see it")
        return real_reader(path)
    monkeypatch.setattr(session, "main_tune", fake_tune)
    monkeypatch.setattr(session, "main_resume", fake_resume)
    monkeypatch.setattr(cli, "read_run_config", never)
    weights = tmp_path / "weights"
    weights.mkdir()
    from test_containers_session import gguf_bytes
    (weights / "tiny-Q4_K_M.gguf").write_bytes(gguf_bytes("Q4_K_M", name="tiny-model", n_ctx_train=32768, nextn=0))
    assert cli.main(["tune", "--config", str(tmp_path / "session.json"), "--output", str(tmp_path / "s1")]) == 3
    assert cli.main(["tune", "--model", str(weights), "--output", str(tmp_path / "s2"),
                     "--budget-seconds", "900", "--image-bundle", "bundle.json", "--policy", "p.json",
                     "--base-config", str(EXAMPLE), "--capabilities-dir", "caps"]) == 3
    assert cli.main(["resume", "--output", str(tmp_path / "s1")]) == 4
    assert [row[0] for row in seen] == ["tune", "tune", "resume"] and all(row[2] == {} for row in seen)
    first, second, third = (row[1] for row in seen)
    assert first.command == "tune" and first.config == str(tmp_path / "session.json") and first.model is None
    assert first.output == str(tmp_path / "s1") and first.budget_seconds is None and first.image_bundle is None
    assert first.policy == "runtime-policy.json" and first.capabilities_dir == "artifacts/container-prep"
    assert first.search == "default" and first.base_config == session.DEFAULT_BASE_CONFIG
    assert second.model == str(weights) and second.config is None and second.budget_seconds == 900
    assert second.image_bundle == "bundle.json" and second.policy == "p.json" and second.capabilities_dir == "caps"
    assert third.command == "resume" and third.output == str(tmp_path / "s1") and third.image_bundle is None
    assert third.policy == "runtime-policy.json" and not hasattr(third, "config")
    assert capsys.readouterr().out == ""  # the session mains own all output
    for argv in (["tune", "--output", str(tmp_path / "x")],  # a source is mandatory
                 ["tune", "--config", "a.json", "--model", "b", "--output", str(tmp_path / "x")],  # and exclusive
                 ["tune", "--config", "a.json"],  # --output is mandatory
                 ["resume"], ["resume", "--config", "a.json", "--output", str(tmp_path / "x")]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    assert seen[3:] == []
    with pytest.raises(SystemExit) as finished:
        cli.main(["--help"])
    assert finished.value.code == 0
    text = capsys.readouterr().out
    assert all(name in text for name in ("tune", "resume", "prepare", "candidate"))


def test_tune_and_resume_are_gated_by_the_policy_before_any_runner_exists(capsys, tmp_path, monkeypatch):
    from llmbench.containers import session

    def no_runner(args, policy):
        raise AssertionError("a denied policy must not construct a runner")
    monkeypatch.setattr(session, "_default_runner_factory", no_runner)
    denied = tmp_path / "denied.json"
    denied.write_text(json.dumps({"allow_inference": True, "allow_model_operations": True}), encoding="utf-8")
    output = tmp_path / "session"
    assert cli.main(["tune", "--config", str(tmp_path / "session.json"), "--output", str(output), "--policy",
                     str(denied)]) == 2
    assert "llmbench tune" in capsys.readouterr().err and not output.exists()
    assert cli.main(["resume", "--output", str(output), "--policy", str(denied)]) == 2
    assert "llmbench resume" in capsys.readouterr().err and not output.exists()
    assert cli.main(["tune", "--config", str(tmp_path / "session.json"), "--output", str(output), "--policy",
                     str(tmp_path / "absent.json")]) == 2
    assert not output.exists()


def test_context_range_flags_round_trip_into_the_session_config_and_resume(tmp_path, capsys, monkeypatch,
                                                                           isolated_gpu_lock):
    """U16 CLI: --context-floor/--context-ceiling are usable INPUT tokens, recorded so resume reproduces them."""
    from llmbench.containers import session as session_module
    from llmbench.containers.session import read_session_config
    from test_containers_session import FakeClock, FakeRunner, gguf_bytes
    policy = tmp_path / "runtime-policy.json"
    policy.write_text(json.dumps({"allow_container_execution": True, "allow_model_operations": True,
                                  "allow_inference": True}), encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "tiny-Q4_K_M.gguf").write_bytes(gguf_bytes("Q4_K_M", name="tiny-model", n_ctx_train=32768, nextn=0))
    clock, runners = FakeClock(), []

    def factory(args, campaign_policy):
        runners.append(FakeRunner(clock, policy=campaign_policy))
        return runners[-1]
    monkeypatch.setattr(session_module, "_default_runner_factory", factory)
    output = tmp_path / "s1"
    assert cli.main(["tune", "--model", str(weights), "--output", str(output), "--policy", str(policy),
                     "--base-config", str(EXAMPLE), "--budget-seconds", "3600", "--context-floor", "2048",
                     "--context-ceiling", "8000"]) == 0
    capsys.readouterr()
    written = json.loads((output / "session-config.json").read_text(encoding="utf-8"))["search"]
    assert (written["context_floor"], written["context_ceiling"]) == (2048, 8000)
    assert written["context_n_ctx_train"] == 32768 and written["context_points_dropped"] == []
    assert written["ctx_tiers"] == [2816, 3840, 5632, 8960]  # input target + 256 reserve + 512 output, rounded up
    session = read_session_config(output / "session-config.json")
    assert session.search.context_range() == {"context_floor": 2048, "context_ceiling": 8000, "n_ctx_train": 32768,
                                              "unit": "usable input tokens", "dropped_input_targets": []}
    # Every candidate really ran at its tier's input target, with the output capacity reserved above it.
    fills = {call["config"].engine.ctx_size: call["config"].requested_input_tokens for call in runners[0].calls}
    assert fills == {2816: 2048, 3840: 3072, 5632: 4864, 8960: 8000}
    assert all(fill + 256 + 512 <= tier for tier, fill in fills.items())
    # resume rebuilds the identical candidates from the recorded range (any drift refuses before a candidate runs).
    assert cli.main(["resume", "--output", str(output), "--policy", str(policy)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stop_reason"] in {"complete", "candidates"} and len(runners) == 2
    assert read_session_config(output / "session-config.json").search == session.search
    # The flags derive a search space from --model; a hand-written session already fixes its own.
    assert cli.main(["tune", "--config", str(output / "session-config.json"), "--output", str(tmp_path / "s2"),
                     "--policy", str(policy), "--context-floor", "2048"]) == 2
    assert "--context-floor" in capsys.readouterr().err and not (tmp_path / "s2").exists()


def test_dataset_root_round_trips_into_the_session_config_and_back_through_resume(tmp_path, capsys, monkeypatch,
                                                                                 isolated_gpu_lock):
    """U18 CLI: --dataset-root decides which official public benchmarks a derived session can even see.

    It is recorded on the derived base, so every candidate reads the same corpora and ``resume`` rebuilds the
    identical selections; and the plan that explains what was selected and skipped is printed before the run
    starts and written into the session directory.
    """
    from llmbench.containers import session as session_module
    from llmbench.containers.session import SESSION_CONFIG_NAME, read_session_config
    from test_containers_derive import stage_datasets
    from test_containers_session import FakeClock, FakeRunner, gguf_bytes
    policy = tmp_path / "runtime-policy.json"
    policy.write_text(json.dumps({"allow_container_execution": True, "allow_model_operations": True,
                                  "allow_inference": True}), encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "tiny-Q4_K_M.gguf").write_bytes(gguf_bytes("Q4_K_M", name="tiny-model", n_ctx_train=32768, nextn=0))
    datasets = stage_datasets(tmp_path / "corpora")  # BFCL + RULER staged outside the default location
    resolved = str(Path(datasets).resolve())
    clock, runners = FakeClock(), []

    def factory(args, campaign_policy):
        runners.append(FakeRunner(clock, policy=campaign_policy))
        return runners[-1]
    monkeypatch.setattr(session_module, "_default_runner_factory", factory)
    output = tmp_path / "s1"

    def tune_argv(target, *extra):
        return ["tune", "--model", str(weights), "--output", str(target), "--policy", str(policy),
                "--base-config", str(EXAMPLE), "--budget-seconds", "3600", *extra]
    assert cli.main(tune_argv(output, "--dataset-root", datasets)) == 0
    printed = capsys.readouterr().err
    assert "public benchmark datasets" in printed and datasets in printed  # shown BEFORE the session ran
    assert "bfcl: selected" in printed and "evalplus: skipped" in printed and "broker" in printed
    session = read_session_config(output / SESSION_CONFIG_NAME)
    assert session.base.dataset_root == resolved
    selected = {item.benchmark_id: item for item in session.base.benchmarks}
    assert {"bfcl", "ruler"} <= set(selected) and "evalplus" not in selected  # no broker in the base config
    assert selected["ruler"].task_ids == ("ruler/niah_multikey_3/4096", "ruler/vt/4096")
    assert selected["ruler"].options["ruler_output_tokens"] == 512
    # Every candidate really carried the root, so the host-process evaluator reads the staged corpora.
    assert {call["config"].dataset_root for call in runners[0].calls} == {resolved}
    plan = json.loads((output / cli.SELECTION_PLAN_NAME).read_text(encoding="utf-8"))
    assert plan["dataset_root"] == resolved and plan["selected"] == ["bfcl", "ruler"]
    assert all(row["reason"] for row in plan["benchmarks"]) and "aider-polyglot" in plan["skipped"]
    # resume rebuilds the identical selections from the recorded root; nothing is re-resolved from the host.
    assert cli.main(["resume", "--output", str(output), "--policy", str(policy)]) == 0
    assert json.loads(capsys.readouterr().out)["stop_reason"] in {"complete", "candidates"}
    # resume rebuilds the identical candidates from the pinned session config (any drift refuses first), so the
    # root and the selections it carries are the ones tune derived - nothing is re-resolved from the host.
    reloaded = read_session_config(output / SESSION_CONFIG_NAME)
    assert len(runners) == 2 and reloaded.base.dataset_root == resolved
    assert reloaded.base.benchmarks == session.base.benchmarks
    # A path that is not a directory is refused, never guessed into the session.
    assert cli.main(tune_argv(tmp_path / "s2", "--dataset-root", str(tmp_path / "absent"))) == 2
    assert "is not a directory" in capsys.readouterr().err and not (tmp_path / "s2").exists()
    # And a hand-written session already states its own root, so the flag is refused rather than overriding it.
    assert cli.main(["tune", "--config", str(output / SESSION_CONFIG_NAME), "--output", str(tmp_path / "s3"),
                     "--policy", str(policy), "--dataset-root", datasets]) == 2
    assert "--dataset-root" in capsys.readouterr().err and not (tmp_path / "s3").exists()


def test_the_default_dataset_root_is_the_staged_directory_only_when_it_exists(tmp_path, capsys, monkeypatch,
                                                                             isolated_gpu_lock):
    """Without the flag, `tune --model` finds artifacts/benchmark-datasets under the working directory - and
    when it is not there, says so and selects nothing rather than naming a path that does not exist."""
    from llmbench.containers import session as session_module
    from llmbench.containers.derive import STAGED_DATASET_DIRNAME
    from llmbench.containers.session import SESSION_CONFIG_NAME, read_session_config
    from test_containers_derive import stage_datasets
    from test_containers_session import FakeClock, FakeRunner, gguf_bytes
    monkeypatch.chdir(tmp_path)
    policy = tmp_path / "runtime-policy.json"
    policy.write_text(json.dumps({"allow_container_execution": True, "allow_model_operations": True,
                                  "allow_inference": True}), encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "tiny-Q4_K_M.gguf").write_bytes(gguf_bytes("Q4_K_M", name="tiny-model", n_ctx_train=32768, nextn=0))
    clock = FakeClock()
    monkeypatch.setattr(session_module, "_default_runner_factory",
                        lambda args, campaign_policy: FakeRunner(clock, policy=campaign_policy))
    arguments = ["tune", "--model", str(weights), "--policy", str(policy), "--base-config", str(EXAMPLE),
                 "--budget-seconds", "3600", "--output"]
    assert cli.main([*arguments, str(tmp_path / "bare")]) == 0
    assert "no staged dataset directory exists" in capsys.readouterr().err
    bare = read_session_config(tmp_path / "bare" / SESSION_CONFIG_NAME)
    assert bare.base.dataset_root is None
    assert not any(item.benchmark_id in {"bfcl", "ruler", "evalplus", "aider-polyglot"}
                   for item in bare.base.benchmarks)
    stage_datasets(tmp_path / STAGED_DATASET_DIRNAME)
    assert cli.main([*arguments, str(tmp_path / "staged")]) == 0
    assert "the staged host directory" in capsys.readouterr().err
    staged = read_session_config(tmp_path / "staged" / SESSION_CONFIG_NAME)
    assert staged.base.dataset_root == str((tmp_path / STAGED_DATASET_DIRNAME).resolve())
    assert {"bfcl", "ruler"} <= {item.benchmark_id for item in staged.base.benchmarks}


def test_real_runner_defaults_to_the_host_broker_factory_and_fakes_never_do(tmp_path, monkeypatch):
    from llmbench.containers import runner as runner_module
    from llmbench.containers.runner import ContainerRunner
    from llmbench.coding.broker import HostBroker
    real = ContainerRunner(capabilities_dir=tmp_path)
    assert real.synthetic is False and real.broker_factory is runner_module._default_broker_factory
    assert real.coding_hook is None  # container_eval resolves the coding hook itself
    injected = ContainerRunner(capabilities_dir=tmp_path, broker_factory=lambda *args: None)
    assert injected.broker_factory is not runner_module._default_broker_factory
    fake = ContainerRunner(capabilities_dir=tmp_path, executor=object())
    assert fake.synthetic is True and fake.broker_factory is None
    for boundary in ("http_factory", "evaluator", "telemetry", "hasher"):
        assert ContainerRunner(capabilities_dir=tmp_path, **{boundary: object()}).broker_factory is None
    calls = []

    def fake_factory(run_dir, config, session_lock, artifacts):
        calls.append((run_dir, config, session_lock, artifacts))
        return "broker"
    monkeypatch.setattr(HostBroker, "factory", fake_factory)
    assert runner_module._default_broker_factory(tmp_path, "config", "lock", "artifacts") == "broker"
    assert calls == [(tmp_path, "config", "lock", "artifacts")]


def test_container_eval_resolves_the_generation_hook_only_where_a_client_can_exist(tmp_path):
    from llmbench import container_eval
    from llmbench.coding.generation import run_coding_benchmarks
    coding = SimpleNamespace(benchmarks=[SimpleNamespace(benchmark_id="tool-probes"),
                                         SimpleNamespace(benchmark_id="coding")])
    without = SimpleNamespace(benchmarks=[SimpleNamespace(benchmark_id="tool-probes")])

    def context(**fields):
        return container_eval.EvaluationContext(base_url="http://127.0.0.1:18080", artifacts_dir=tmp_path,
                                                deadline_monotonic=0.0, **fields)
    injected = object()
    assert container_eval._coding_hook(context(coding_hook=injected), without) is None
    assert container_eval._coding_hook(context(coding_hook=injected, coding_client="direct"), coding) is injected
    assert container_eval._coding_hook(context(), coding) is None  # host process without a broker: no client
    assert container_eval._coding_hook(context(coding_client="direct"), coding) is run_coding_benchmarks
    assert container_eval._coding_hook(context(allow_remote=True), coding) is run_coding_benchmarks
    assert container_eval._coding_hook(context(allow_remote=True), without) is None


def test_host_registry_pin_is_not_compared_to_unused_evaluator_image_registry():
    from llmbench.containers.config import ContainerRunConfig
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["registry_digest"] = "f" * 64
    host = ContainerRunConfig.model_validate_json(json.dumps(raw))
    bundle = bundle_for(host)
    assert host.evaluator.mode == "host-process"
    assert host.registry_digest != bundle.registry_digest
    assert cli.check_image_bundle(host, bundle) == []
    # The inference image is still pinned even though the unused evaluator registry can differ.
    other = bundle_for(host, inference={**host.inference_image.model_dump(), "image_id": "sha256:" + "9" * 64})
    assert any("inference.image_id" in problem for problem in cli.check_image_bundle(host, other))


def test_container_registry_pin_still_must_match_bundled_evaluator():
    from llmbench.containers.config import ContainerRunConfig, read_run_config
    base = read_run_config(EXAMPLE)
    bundle = bundle_for(base)
    raw = base.model_dump(mode="json")
    raw.update(registry_digest="f" * 64,
               evaluator={"mode": "container", "image": bundle.evaluator.model_dump(mode="json")})
    container = ContainerRunConfig.model_validate_json(json.dumps(raw))
    assert cli.check_image_bundle(container, bundle) == [
        "registry_digest: config differs from the bundle's evaluator registry digest"]


def test_host_preflight_still_rejects_wrong_installed_registry_pin():
    from llmbench.containers.config import ContainerRunConfig
    from llmbench.containers.runner import _default_registry_validator
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["registry_digest"] = "f" * 64
    host = ContainerRunConfig.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError, match="does not match the installed benchmark registry"):
        _default_registry_validator(host)


# ---- runtimes: metal-native beside nvidia-container (the NVIDIA command lines above are unchanged) ---------------

METAL_VERSION = ("version: 0.4.1-dev (build 11011, commit aa39d7a3e)\n"
                 "built with AppleClang 21.0.0.21000101 for Darwin arm64\n")
NATIVE_EXECUTABLE = "/Users/example/llama-b11011/llama-server"


def metal_help() -> str:
    """The b11011 macOS build's --help: the CUDA capture plus `--rpc SERVERS` (that build has GGML_RPC=ON)."""
    lines = (DATA / "llama-server-help-b11011.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    at = next(index for index, line in enumerate(lines) if line.startswith("--list-devices")) + 1
    rpc = ["--rpc SERVERS                           comma separated list of RPC servers (host:port)\n",
           "                                        (env: LLAMA_ARG_RPC)\n"]
    return "".join(lines[:at] + rpc + lines[at:])


def metal_help_sha256() -> str:
    from llmbench.containers.capabilities import help_sha256
    return help_sha256(metal_help())


@pytest.fixture
def native_prep(tmp_path):
    directory = tmp_path / "native-prep"
    directory.mkdir()
    (directory / "llama-server-help.txt").write_text(metal_help(), encoding="utf-8")
    (directory / "llama-server-version.txt").write_text(METAL_VERSION, encoding="utf-8")
    return directory


def write_native_config(directory, name="native.json", **options):
    from test_runtime_registry import native_raw
    path = directory / name
    path.write_text(json.dumps(native_raw(**options)), encoding="utf-8")
    return path


def native_bundle_for(config, *, worker=None, sandbox=None, libraries=None, **server_changes):
    from llmbench.containers.config import NativeBundle
    server = {**config.native_server.model_dump(mode="json"), **server_changes}
    return NativeBundle.model_validate_json(json.dumps({
        "prepared_utc": "2026-09-23T00:00:00+00:00", "native_server": server, "libraries": libraries or {},
        "worker": worker, "help_sha256": server["help_sha256"], "registry_digest": "1" * 64,
        "host": {"machine": "arm64", "chip": "Apple M1"},
        "sandbox": sandbox or {"status": "blocked", "reason": "docker CLI not found"}}))


def write_bundle(directory, name, bundle):
    path = directory / name
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    return path


def without_endpoint(argv) -> list[str]:
    """An argv minus the four flags whose values legitimately differ between runtimes (and between configs)."""
    items, tokens = [], iter(argv)
    for token in tokens:
        if token in ("--model", "--host", "--port", "--alias"):
            next(tokens)
            continue
        items.append(token)
    return items


def test_validate_reports_a_native_configs_runtime_and_refuses_what_metal_cannot_honour(capsys, tmp_path):
    from llmbench.containers.config import read_run_config
    assert cli.main(["validate", "--config", str(EXAMPLE)]) == 0
    assert list(output_of(capsys)) == ["valid", "label", "fingerprint", "alias", "execution_performed"]  # unchanged
    path = write_native_config(tmp_path)
    assert cli.main(["validate", "--config", str(path)]) == 0
    data = output_of(capsys)
    config = read_run_config(path)
    assert data["valid"] is True and data["execution_performed"] is False and data["runtime"] == "metal-native"
    assert data["fingerprint"] == config.fingerprint() and data["alias"] == config.alias()
    assert data["required_operations"] == ["native", "load", "inference"] and data["unsupported_settings"] == []
    assert [note.split(" ", 1)[0] for note in data["not_enforced_settings"]] == [
        "limits.inference_memory_mib", "limits.inference_cpus", "limits.max_foreign_vram_mib"]
    assert cli.main(["validate", "--config", str(path), "--runtime", "metal-native"]) == 0
    capsys.readouterr()
    # --runtime on a config command only confirms the config's own runtime.
    for config_path, runtime in ((path, "nvidia-container"), (EXAMPLE, "metal-native")):
        assert cli.main(["validate", "--config", str(config_path), "--runtime", runtime]) == 2
        assert "disagrees with the config" in capsys.readouterr().err
    second_gpu = write_native_config(tmp_path, "second-gpu.json", limits={"gpu_device_id": "1"})
    assert cli.main(["validate", "--config", str(second_gpu)]) == 2
    data = output_of(capsys)
    assert data["valid"] is False and "MTL0" in data["unsupported_settings"][0]


def test_capabilities_default_directory_follows_the_runtime_and_checks_the_native_pin(capsys, tmp_path, prep,
                                                                                     native_prep, monkeypatch):
    from llmbench.containers.config import read_run_config
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()
    prep.rename(tmp_path / "artifacts" / "container-prep")
    native_prep.rename(tmp_path / "artifacts" / "native-prep")
    assert cli.main(["capabilities"]) == 0
    nvidia = output_of(capsys)
    assert list(nvidia) == ["flags", "help_sha256", "version", "build", "allowed_flags_missing",
                            "execution_performed"]  # the NVIDIA default and output are unchanged
    assert nvidia["help_sha256"] == read_run_config(EXAMPLE).help_sha256
    assert cli.main(["capabilities", "--runtime", "metal-native"]) == 0
    metal = output_of(capsys)
    assert metal["runtime"] == "metal-native" and metal["capabilities_dir"] == "artifacts/native-prep"
    assert metal["help_sha256"] == metal_help_sha256() != nvidia["help_sha256"]
    assert metal["flags"] == nvidia["flags"] + 1 and metal["build"] == "b11011-aa39d7a3e"
    assert metal["allowed_flags_missing"] == []
    pinned = write_native_config(tmp_path, help_sha256=metal_help_sha256())
    assert cli.main(["capabilities", "--config", str(pinned)]) == 0  # the config's runtime picks the directory
    data = output_of(capsys)
    assert data["help_sha256_matches"] is True and data["findings"] == [] and data["unsupported_settings"] == []
    # The CUDA capture is not this server's help, and the native pin says so.
    assert cli.main(["capabilities", "--config", str(pinned), "--capabilities-dir", "artifacts/container-prep"]) == 2
    assert output_of(capsys)["help_sha256_matches"] is False
    assert cli.main(["capabilities", "--config", str(EXAMPLE)]) == 0
    assert output_of(capsys)["help_sha256_matches"] is True
    assert cli.main(["capabilities", "--config", str(pinned), "--runtime", "nvidia-container"]) == 2
    assert "disagrees" in capsys.readouterr().err


def test_plan_for_a_native_config_is_the_loopback_argv_with_identical_settings_and_no_compose(capsys, tmp_path,
                                                                                             prep, native_prep,
                                                                                             monkeypatch):
    from llmbench.containers.config import read_run_config
    from llmbench.containers.plan import build_server_argv
    monkeypatch.chdir(tmp_path)
    path = write_native_config(tmp_path, help_sha256=metal_help_sha256())
    before = sorted(item.name for item in tmp_path.iterdir())
    assert cli.main(["plan", "--config", str(path), "--capabilities-dir", str(native_prep)]) == 0
    data = output_of(capsys)
    config = read_run_config(path)
    assert "compose" not in data and data["execution_performed"] is False and data["runtime"] == "metal-native"
    assert data["executable"] == NATIVE_EXECUTABLE and data["fingerprint"] == config.fingerprint()
    argv = data["server_argv"]
    assert argv[:6] == ["--model", config.model.host_path, "--host", "127.0.0.1", "--port", "0"]
    assert argv[argv.index("--alias") + 1] == config.alias()
    # Every setting flag is the one the NVIDIA container gets for the same engine settings.
    assert without_endpoint(argv) == without_endpoint(build_server_argv(read_run_config(EXAMPLE)))
    assert data["required_operations"] == ["native", "load", "inference"] and len(data["not_enforced_settings"]) == 3
    assert sorted(item.name for item in tmp_path.iterdir()) == before  # pure: nothing written
    # The saved help must be this server's own, exactly as for NVIDIA.
    assert cli.main(["plan", "--config", str(path), "--capabilities-dir", str(prep)]) == 2
    assert "help_sha256" in capsys.readouterr().err
    second_gpu = write_native_config(tmp_path, "second-gpu.json", help_sha256=metal_help_sha256(),
                                     limits={"gpu_device_id": "1"})
    assert cli.main(["plan", "--config", str(second_gpu), "--capabilities-dir", str(native_prep)]) == 2
    assert "MTL0" in output_of(capsys)["unsupported_settings"][0]


CANDIDATE_KEYS = ["state", "synthetic", "attempt_id", "failure_stage", "failure_reasons", "warnings",
                  "cleanup_verified", "abort_campaign", "elapsed_seconds", "effective_settings_verified",
                  "actual_context_verified", "minimum_native_tps", "samples_total", "reports", "output"]


def test_candidate_dispatches_a_native_config_and_pins_it_to_its_own_bundle_kind(capsys, tmp_path):
    from llmbench.containers.config import read_run_config
    path = write_native_config(tmp_path)
    config = read_run_config(path)
    memory = {"kind": "apple-unified", "server_phys_footprint_mib": 1905}
    seen = []

    class Runner:
        def run(self, config, output, *, remaining_budget_seconds=None):
            seen.append(("run", config.runtime, output))
            return SimpleNamespace(
                state="completed", synthetic=True, attempt_id="a" * 32, failure_stage=None, failure_reasons=(),
                warnings=(), cleanup=SimpleNamespace(verified=True), abort_campaign=False, elapsed_seconds=1.0,
                effective_settings_verified=True, actual_context_verified=True, speed={}, samples_total=0,
                model_dump=lambda mode: {"reports": {}, **({"memory": memory} if config.runtime != "nvidia-container"
                                                             else {})})

    def factory(args):
        seen.append(("args", args))
        return Runner()

    def runs():
        return sum(1 for row in seen if row[0] == "run")
    arguments = ["candidate", "--config", str(path), "--output", str(tmp_path / "out")]
    matching = write_bundle(tmp_path, "native-bundle.json", native_bundle_for(config))
    assert cli.main([*arguments, "--native-bundle", str(matching)], runner_factory=factory) == 0
    data = output_of(capsys)
    assert list(data) == [*CANDIDATE_KEYS, "runtime", "memory"]
    assert data["runtime"] == "metal-native" and data["memory"] == memory
    args = seen[0][1]
    assert args.capabilities_dir == "artifacts/native-prep" and args.native_bundle == str(matching)
    assert seen[1] == ("run", "metal-native", str(tmp_path / "out"))
    # Each identity pin must be the prepared one; a refusal happens before any runner exists.
    for field, value in (("executable_sha256", "1" * 64), ("libraries_sha256", "2" * 64),
                         ("build_info", "b11012-0123456"), ("help_sha256", "3" * 64)):
        bundle = write_bundle(tmp_path, f"{field}.json", native_bundle_for(config, **{field: value}))
        assert cli.main([*arguments, "--native-bundle", str(bundle)], runner_factory=factory) == 2
        assert f"native_server.{field}" in capsys.readouterr().err
    # Where the executable lives on this host, and its display source, are not identity.
    moved = write_bundle(tmp_path, "moved.json", native_bundle_for(config, executable="/opt/llama/llama-server",
                                                                   source="rebuilt elsewhere"))
    assert cli.main([*arguments, "--native-bundle", str(moved)], runner_factory=factory) == 0
    capsys.readouterr()
    # A worker image the config names must be the bundle's.
    worker = {"role": "worker", "reference": "sha256:" + "4" * 64, "image_id": "sha256:" + "4" * 64,
              "platform": "linux/arm64", "entrypoint": ["/usr/local/bin/llmbench-worker"]}
    with_worker = write_native_config(tmp_path, "with-worker.json", worker_image=worker)
    assert cli.main(["candidate", "--config", str(with_worker), "--output", str(tmp_path / "w"), "--native-bundle",
                     str(matching)], runner_factory=factory) == 2
    assert "worker: the bundle carries no worker image" in capsys.readouterr().err
    assert runs() == 2
    # Each runtime is pinned only by its own bundle kind: the other kind is refused, never silently not compared.
    image_bundle = write_bundle(tmp_path, "image-bundle.json", bundle_for(read_run_config(EXAMPLE)))
    assert cli.main([*arguments, "--image-bundle", str(image_bundle)], runner_factory=factory) == 2
    assert "--native-bundle" in capsys.readouterr().err
    assert cli.main(["candidate", "--config", str(EXAMPLE), "--output", str(tmp_path / "n"), "--native-bundle",
                     str(matching)], runner_factory=factory) == 2
    assert "use --image-bundle" in capsys.readouterr().err
    # A capabilities directory means nothing to the native runner, so naming one is refused rather than ignored.
    assert cli.main([*arguments, "--capabilities-dir", "artifacts/native-prep"], runner_factory=factory) == 2
    assert "does not apply to a metal-native candidate" in capsys.readouterr().err
    unsupported = write_native_config(tmp_path, "second-gpu.json", limits={"gpu_device_id": "1"})
    assert cli.main(["candidate", "--config", str(unsupported), "--output", str(tmp_path / "u")],
                    runner_factory=factory) == 2
    err = capsys.readouterr().err
    assert "cannot honour" in err and "MTL0" in err
    assert cli.main([*arguments, "--runtime", "nvidia-container"], runner_factory=factory) == 2
    assert runs() == 2
    # The NVIDIA candidate output keeps exactly its keys, and its default capabilities directory.
    assert cli.main(["candidate", "--config", str(EXAMPLE), "--output", str(tmp_path / "nv")],
                    runner_factory=factory) == 0
    assert list(output_of(capsys)) == CANDIDATE_KEYS
    assert seen[-2][1].capabilities_dir == "artifacts/container-prep" and seen[-2][1].native_bundle is None


def test_bundle_checks_never_compare_across_runtimes():
    from llmbench.containers.config import read_run_config
    from test_runtime_registry import native_config
    nvidia, native = read_run_config(EXAMPLE), native_config()
    image, pinned = bundle_for(nvidia), native_bundle_for(native)
    assert cli.check_image_bundle(nvidia, image) == [] == cli.check_bundle(nvidia, image)
    assert cli.check_native_bundle(native, pinned) == [] == cli.check_bundle(native, pinned)
    # An image bundle has nothing to compare for a native config; that is a problem, not a pass (sampling and the
    # prepared sweep call check_image_bundle).
    [problem] = cli.check_image_bundle(native, image)
    assert problem.startswith("runtime: the config runs metal-native") and "native-bundle.json" in problem
    assert cli.check_bundle(nvidia, pinned) == [
        "runtime: the config runs nvidia-container; a native bundle pins only metal-native candidates"]


def test_prepare_native_is_gated_by_the_native_policy_and_keeps_the_nvidia_command_line(capsys, tmp_path,
                                                                                         monkeypatch):
    import sys
    import types
    from llmbench.containers.config import read_run_config
    from test_runtime_registry import native_config
    worker = {"role": "worker", "reference": "sha256:" + "4" * 64, "image_id": "sha256:" + "4" * 64,
              "platform": "linux/arm64", "entrypoint": ["/usr/local/bin/llmbench-worker"]}
    bundle = native_bundle_for(native_config(), worker=worker, sandbox={"status": "available", "reason": None})
    seen = []

    def preparer(args):
        seen.append(args)
        return bundle
    output = tmp_path / "native-prep"
    base = ["prepare", "--runtime", "metal-native", "--llama-server", NATIVE_EXECUTABLE, "--output", str(output)]
    assert cli.main(base, preparer=preparer) == 0
    assert output_of(capsys) == {
        "execution_performed": True, "runtime": "metal-native", "output": str(output),
        "executable": NATIVE_EXECUTABLE, "native_server": "e" * 64, "libraries_sha256": "f" * 64,
        "build_info": "b11011-aa39d7a3e", "worker": "sha256:" + "4" * 64, "worker_platform": "linux/arm64",
        "sandbox": "available", "sandbox_reason": None, "help_sha256": "a" * 64, "registry_digest": "1" * 64,
        "prepared_utc": "2026-09-23T00:00:00+00:00"}
    assert seen[0].llama_server == NATIVE_EXECUTABLE and seen[0].inference is None and seen[0].worker_iidfile is None
    # Requiredness follows the runtime, and a flag of the other runtime is refused, both the argparse way.
    for argv in (["prepare", "--runtime", "metal-native", "--output", str(output)],
                 [*base, "--inference", INFERENCE_REF], [*base, "--evaluator-base", BASE_REF],
                 [*base, "--wheel", "localllmbench.whl"], [*base, "--lock", "requirements.linux.lock"],
                 ["prepare", "--inference", INFERENCE_REF, "--evaluator-base", BASE_REF, "--llama-server",
                  NATIVE_EXECUTABLE, "--output", str(output)],
                 ["prepare", "--runtime", "nvidia-container", "--inference", INFERENCE_REF, "--output", str(output)],
                 ["prepare", "--evaluator-base", BASE_REF, "--output", str(output)]):
        with pytest.raises(SystemExit) as refused:
            cli.main(argv, preparer=preparer)
        assert refused.value.code == 2
    assert "the following arguments are required: --inference" in capsys.readouterr().err
    assert len(seen) == 1
    # A preparer that hands back an NVIDIA bundle for a native prepare is refused, not summarised.
    assert cli.main(base, preparer=lambda args: bundle_for(read_run_config(EXAMPLE))) == 2
    assert "not a NativeBundle" in capsys.readouterr().err
    # The default preparer checks the native permission before native_prep (which runs the executable) is used.
    calls = []

    def prepare_native(executable, output, *, session_lock, worker_iidfile=None, **options):
        calls.append((executable, output, session_lock, worker_iidfile, options))
        return bundle
    fake = types.ModuleType("llmbench.containers.native_prep")
    fake.prepare_native = prepare_native
    monkeypatch.setitem(sys.modules, "llmbench.containers.native_prep", fake)
    container_only = tmp_path / "container-policy.json"
    container_only.write_text(json.dumps({"allow_container_execution": True, "allow_model_operations": True,
                                          "allow_inference": True}), encoding="utf-8")
    assert cli.main([*base, "--policy", str(container_only)]) == 2
    assert "native forbidden" in capsys.readouterr().err and calls == [] and not output.exists()
    native_policy = tmp_path / "native-policy.json"
    native_policy.write_text(json.dumps({"allow_native_execution": True}), encoding="utf-8")
    assert cli.main([*base, "--policy", str(native_policy), "--worker-iidfile", "worker-image.id"]) == 0
    assert output_of(capsys)["worker"] == "sha256:" + "4" * 64
    [(executable, destination, lock, iidfile, options)] = calls
    assert executable == Path(NATIVE_EXECUTABLE) and destination == output and iidfile == "worker-image.id"
    assert lock.allow_native_execution is True and lock.allow_container_execution is False and options == {}


def test_tune_forwards_the_native_runtime_to_the_plan_and_the_session(capsys, tmp_path, monkeypatch):
    from llmbench.containers import derive, session as session_module
    from llmbench.containers.config import read_native_bundle
    from test_runtime_registry import native_config
    plans, tunes = [], []

    def fake_plan(model, **options):
        plans.append(options)
        return {"dataset_root_source": "none", "dataset_root": None}

    def fake_tune(args, **options):
        tunes.append(args)
        return 0
    monkeypatch.setattr(derive, "benchmark_plan_for_model", fake_plan)
    monkeypatch.setattr(derive, "benchmark_plan_lines", lambda plan: ["plan line"])
    monkeypatch.setattr(session_module, "main_tune", fake_tune)
    bundle_path = write_bundle(tmp_path, "native-bundle.json", native_bundle_for(native_config()))
    common = ["tune", "--model", str(tmp_path / "weights"), "--output", str(tmp_path / "s"), "--base-config",
              str(EXAMPLE)]
    assert cli.main([*common, "--runtime", "metal-native", "--native-bundle", str(bundle_path)]) == 0
    err = capsys.readouterr().err
    assert ("llmbench tune: runtime metal-native: llama-server eeeeeeeeeeee (b11011-aa39d7a3e); coding sandbox "
            "blocked (docker CLI not found)") in err and "llmbench tune: plan line" in err
    # The plan sees the bundle exactly as the derivation will; the session reads both flags from args.
    assert plans[0]["native_bundle"] == read_native_bundle(bundle_path)
    assert tunes[0].runtime == "metal-native" and tunes[0].native_bundle == str(bundle_path)
    # NVIDIA: the plan gets exactly the arguments it always did, and the session sees no runtime.
    assert cli.main(common) == 0
    assert set(plans[1]) == {"base", "budget_seconds", "context_floor", "context_ceiling", "dataset_root"}
    assert tunes[1].runtime is None and tunes[1].native_bundle is None
    assert "runtime metal-native" not in capsys.readouterr().err
    # Both flags are explicit and refused, before any plan or session, when they contradict each other.
    for extra, message in ((["--runtime", "metal-native"], "requires --native-bundle"),
                           (["--native-bundle", str(bundle_path)], "add --runtime metal-native"),
                           (["--runtime", "nvidia-container", "--native-bundle", str(bundle_path)], "add --runtime"),
                           (["--runtime", "metal-native", "--native-bundle", str(bundle_path), "--image-bundle",
                             "image-bundle.json"], "--image-bundle pins container images"),
                           (["--runtime", "metal-native", "--native-bundle", str(tmp_path / "absent.json")],
                            "absent.json")):
        assert cli.main([*common, *extra]) == 2
        assert message in capsys.readouterr().err
    assert len(plans) == 2 and len(tunes) == 2
    # A --config session already states its runtime and pins, like --context-floor and --dataset-root.
    for extra in (["--runtime", "metal-native"], ["--runtime", "nvidia-container"], ["--native-bundle",
                                                                                    str(bundle_path)]):
        assert cli.main(["tune", "--config", str(tmp_path / "session.json"), "--output", str(tmp_path / "s2"),
                         *extra]) == 2
        assert "--runtime/--native-bundle" in capsys.readouterr().err
    assert len(tunes) == 2


def test_doctor_reports_static_runtime_facts_without_running_anything(capsys, tmp_path, monkeypatch,
                                                                     isolated_gpu_lock):
    import platform
    import shutil
    from llmbench.containers.config import RUNTIMES
    looked_up = []
    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: looked_up.append(name))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"allow_native_execution": True}), encoding="utf-8")
    doctor = ["doctor", "--policy", str(policy)]
    assert cli.main(doctor) == 0
    data = output_of(capsys)
    facts = data["runtime"]
    assert data["server_contacted"] is False and data["model_operations_performed"] is False
    assert facts["execution_performed"] is False and facts["native_bundle"] is None
    assert data["session_policy"]["allow_native_execution"] is True
    assert facts["system"] == platform.system() and facts["machine"] == platform.machine()
    assert facts["docker_cli"] is None and looked_up == ["docker"]
    if hasattr(os, "sysconf"):
        assert type(facts["memory_total_bytes"]) is int and facts["memory_total_bytes"] > 0
    lease = isolated_gpu_lock / "llmbench-gpu-resource.lock"
    assert facts["lease"] == {"path": str(lease), "exists": False, "holder": None}
    assert list(facts["runtimes"]) == list(RUNTIMES)
    assert facts["runtimes"]["metal-native"]["memory_kind"] == "apple-unified"
    # A held lease is named, with the native advice when a native run holds it.
    lease.write_text(json.dumps({"pid": os.getpid(), "owner": "native-run:abc", "runtime": "metal-native",
                                 "server_pid": os.getpid()}), encoding="utf-8")
    assert cli.main(doctor) == 0
    facts = output_of(capsys)["runtime"]
    assert facts["lease"]["exists"] is True and f"its llama-server is pid {os.getpid()}" in facts["lease"]["holder"]
    assert lease.exists()  # doctor describes a lock; it never removes one


@pytest.mark.skipif(os.name == "nt", reason="a native server is pinned by a POSIX absolute path; metal-native is "
                                            "macOS-only")
def test_doctor_checks_a_native_bundle_by_the_runners_own_rule_without_running_it(capsys, tmp_path,
                                                                                  isolated_gpu_lock):
    import hashlib
    from llmbench.config import canonical_json
    from llmbench.containers.native_prep import hash_native_server
    from test_runtime_registry import native_config

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
    install = tmp_path / "llama-b11011"
    install.mkdir()
    executable = install / "llama-server"
    executable.write_bytes(b"\xcf\xfa\xed\xfe llama-server b11011")
    (install / "libggml-metal.0.dylib").write_bytes(b"metal backend")
    (install / "libllama.0.dylib").write_bytes(b"llama")
    (install / "libllama.dylib").symlink_to("libllama.0.dylib")  # a release's soname link, inside the directory
    (install / "llama-cli").write_bytes(b"not a library, never part of the pin")
    libraries = {"libggml-metal.0.dylib": sha(b"metal backend"), "libllama.0.dylib": sha(b"llama"),
                 "libllama.dylib": sha(b"llama")}
    libraries_sha256 = sha(canonical_json(libraries).encode("utf-8"))
    bundle = native_bundle_for(native_config(), executable=str(executable),
                               executable_sha256=sha(executable.read_bytes()), libraries=libraries,
                               libraries_sha256=libraries_sha256)
    bundle_path = write_bundle(tmp_path, "native-bundle.json", bundle)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"allow_native_execution": True}), encoding="utf-8")
    doctor = ["doctor", "--policy", str(policy)]
    assert cli.main([*doctor, "--native-bundle", str(bundle_path)]) == 0
    state = output_of(capsys)["runtime"]["native_bundle"]
    assert state["executable_exists"] is True and state["executable_sha256_matches"] is True
    assert state["libraries"] == {"libggml-metal.0.dylib": "matches", "libllama.0.dylib": "matches",
                                  "libllama.dylib": "matches"}
    assert state["libraries_match"] is True and state["sandbox"] == "blocked"
    # The set digest is the native runner's own admission comparison, computed by the same rule.
    assert state["libraries_sha256_observed"] == libraries_sha256 == hash_native_server(executable)["libraries_sha256"]
    assert state["libraries_sha256_matches"] is True
    # A rebuilt Metal backend is a different server even when the executable is unchanged.
    (install / "libggml-metal.0.dylib").write_bytes(b"another metal backend")
    assert cli.main([*doctor, "--native-bundle", str(bundle_path)]) == 0
    facts = output_of(capsys)["runtime"]
    assert facts["native_bundle"]["executable_sha256_matches"] is True
    assert facts["native_bundle"]["libraries"]["libggml-metal.0.dylib"] == "differs"
    assert facts["native_bundle"]["libraries_match"] is False
    assert facts["native_bundle"]["libraries_sha256_matches"] is False
    # A library added beside the server changes what dyld can load and what the runner pins: it is never
    # overlooked just because the bundle did not list it.
    (install / "libggml-metal.0.dylib").write_bytes(b"metal backend")
    (install / "libggml-blas.0.dylib").write_bytes(b"planted")
    assert cli.main([*doctor, "--native-bundle", str(bundle_path)]) == 0
    state = output_of(capsys)["runtime"]["native_bundle"]
    assert state["libraries"] == {"libggml-blas.0.dylib": "not in the bundle", "libggml-metal.0.dylib": "matches",
                                  "libllama.0.dylib": "matches", "libllama.dylib": "matches"}
    assert state["libraries_match"] is False and state["libraries_sha256_matches"] is False
    assert state["libraries_sha256_observed"] == hash_native_server(executable)["libraries_sha256"]
    (install / "libggml-blas.0.dylib").unlink()
    # A library link may point only inside the directory, as the runner requires; elsewhere it is not followed.
    outside = tmp_path / "outside.dylib"
    outside.write_bytes(b"llama")
    (install / "libllama.0.dylib").unlink()
    (install / "libllama.0.dylib").symlink_to(outside)
    assert cli.main([*doctor, "--native-bundle", str(bundle_path)]) == 0
    state = output_of(capsys)["runtime"]["native_bundle"]
    assert "outside the executable's directory" in state["libraries"]["libllama.0.dylib"]
    assert state["libraries_match"] is False and state["libraries_sha256_matches"] is None
    (install / "libllama.0.dylib").unlink()
    (install / "libllama.0.dylib").write_bytes(b"llama")
    # The runner pins a regular executable, never a link that could be repointed; the same bytes behind a link
    # are reported, not matched.
    real = install / "llama-server.real"
    executable.rename(real)
    executable.symlink_to(real)
    assert cli.main([*doctor, "--native-bundle", str(bundle_path)]) == 0
    state = output_of(capsys)["runtime"]["native_bundle"]
    assert state["executable_sha256_matches"] is None and "never a link" in state["executable_problem"]
    executable.unlink()
    real.unlink()
    assert cli.main([*doctor, "--native-bundle", str(bundle_path)]) == 0
    state = output_of(capsys)["runtime"]["native_bundle"]
    assert state["executable_exists"] is False and state["executable_sha256_matches"] is None
    assert state["executable_problem"] == "missing"
    # A library name that is not a plain file name is never opened.
    escaping = write_bundle(tmp_path, "escaping.json", native_bundle_for(
        native_config(), executable=str(executable), libraries={"../outside.dylib": "0" * 64}))
    assert cli.main([*doctor, "--native-bundle", str(escaping)]) == 0
    state = output_of(capsys)["runtime"]["native_bundle"]
    assert state["libraries"]["../outside.dylib"] == "invalid name; not read"
    assert state["libraries_match"] is False
    assert cli.main([*doctor, "--native-bundle", str(tmp_path / "absent.json")]) == 2
