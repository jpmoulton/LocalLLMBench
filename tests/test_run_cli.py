import json
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
    runner = cli._default_runner(SimpleNamespace(capabilities_dir=str(tmp_path), policy=str(tmp_path / "p.json")))
    assert runner.synthetic is False and runner.executor is None and runner.policy_path == tmp_path / "p.json"


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
