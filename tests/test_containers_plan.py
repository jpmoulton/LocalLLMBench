import json
import re
from pathlib import Path

import pytest

from llmbench.containers.config import ChildGrant, ContainerRunConfig, read_run_config
from llmbench.containers.plan import build_compose_plan, build_server_argv, evaluator_argv

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DATA = Path(__file__).parent / "data"
ATTEMPT = "0123456789abcdef0123456789abcdef"
IMAGE = "sha256:" + "b" * 64
EVALUATOR = {"role": "evaluator", "reference": IMAGE, "image_id": IMAGE,
             "entrypoint": ["python", "-m", "llmbench.container_eval"]}


def example(**engine):
    raw = json.loads((EXAMPLES / "candidate.json").read_text(encoding="utf-8"))
    raw["engine"].update(engine)
    return ContainerRunConfig.model_validate_json(json.dumps(raw))


def plan_for(config, tmp_path):
    return build_compose_plan(config, tmp_path / "run", attempt_id=ATTEMPT, session_id="session-1")


def container_config(*, broker=False, host_path=None, **top):
    raw = json.loads((EXAMPLES / "candidate.json").read_text(encoding="utf-8"))
    raw["evaluator"] = {"mode": "container", "image": EVALUATOR}
    if broker:
        raw["broker"] = {}
    if host_path is not None:
        raw["assets"][0]["host_path"] = host_path
    raw.update(top)
    return ContainerRunConfig.model_validate_json(json.dumps(raw))


def grant_for(config, seconds=900):
    return ChildGrant(grant_seconds=seconds, artifact_bytes=config.limits.evaluator_artifact_bytes,
                      issued_utc="2026-09-18T00:00:00+00:00", issued_host_offset_seconds=12.5,
                      watchdog_slack_seconds=config.evaluator.watchdog_slack_seconds)


def posix_sources(compose):
    compose = json.loads(json.dumps(compose))
    for service in compose["services"].values():
        for mount in service["volumes"]:
            mount["source"] = Path(mount["source"]).as_posix()
    return compose


def test_golden_argv_matches_the_live_verified_order():
    config = example()
    assert build_server_argv(config) == (
        "--model", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "--alias", config.alias(),
        "--ctx-size", "8192", "--n-gpu-layers", "all", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--kv-offload", "--flash-attn", "on", "--load-mode", "none", "--batch-size", "2048",
        "--ubatch-size", "512", "--threads", "8", "--threads-batch", "8", "--parallel", "1",
        "--no-cache-prompt", "--cache-reuse", "0", "--cache-ram", "0", "--no-context-shift", "--fit", "off",
        "--jinja", "--reasoning", "off", "--reasoning-format", "deepseek", "--reasoning-budget", "-1",
        "--spec-type", "none", "--warmup", "--no-ui", "--metrics", "--slots", "--offline",
        "--log-verbosity", "4")


def test_argv_variants():
    argv = build_server_argv(example(n_gpu_layers=40, kv_offload=False, cache_prompt=True, swa_full=True,
                                     warmup=False, spec_type="draft-mtp", spec_draft_n_max=5,
                                     cache_type_k="q8_0", cache_type_v="q8_0", load_mode="mmap"))
    joined = " ".join(argv)
    for fragment in ("--n-gpu-layers 40", "--no-kv-offload", " --cache-prompt", "--fit off --swa-full --jinja",
                     "--spec-type draft-mtp --spec-draft-n-max 5 --no-warmup", "--cache-type-k q8_0",
                     "--load-mode mmap"):
        assert fragment in joined
    assert "--spec-draft-n-max" not in build_server_argv(example())
    assert not {"--log-colors", "--no-webui", "--no-mmap", "--seed"} & set(argv)


def test_compose_plan_is_deterministic_and_hardened(tmp_path):
    config = example()
    plan = plan_for(config, tmp_path)
    assert plan.compose_json() == plan_for(config, tmp_path).compose_json()
    assert json.loads(plan.compose_json()) == plan.compose
    assert plan.project_name == "llmbench-0123456789ab" and plan.compose["name"] == plan.project_name
    assert re.fullmatch(r"llmbench-[0-9a-f]{12}", plan.project_name)
    assert plan.compose_path == tmp_path / "run" / "plan" / "compose.json"
    assert set(plan.compose["services"]) == {"inference"}
    service = plan.compose["services"]["inference"]
    assert service["image"] == config.inference_image.reference and service["pull_policy"] == "never"
    assert service["entrypoint"] == ["/app/llama-server"] and tuple(service["command"]) == plan.server_argv
    assert service["restart"] == "no" and service["init"] is True and service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"] and service["healthcheck"] == {"disable": True}
    assert service["logging"] == {"driver": "local", "options": {"max-size": "16m", "max-file": "2"}}
    resources = service["deploy"]["resources"]
    assert resources["limits"] == {"cpus": "12.0", "memory": str(24576 * 1048576)}
    assert resources["reservations"]["devices"] == [
        {"driver": "nvidia", "device_ids": ["0"], "capabilities": ["gpu"]}]
    for key in ("environment", "env_file", "privileged", "cap_add", "devices", "network_mode", "pid", "ipc"):
        assert key not in service
    text = plan.compose_json()
    assert "docker.sock" not in text and "privileged" not in text and ":latest" not in text


def test_model_mount_is_long_syntax_read_only_and_round_trips_windows_paths(tmp_path):
    config = example()
    service = plan_for(config, tmp_path).compose["services"]["inference"]
    assert service["volumes"] == [{"type": "bind", "source": config.model.host_path,
                                   "target": "/models/model.gguf", "read_only": True,
                                   "bind": {"create_host_path": False}}]
    raw = json.loads((EXAMPLES / "candidate.json").read_text(encoding="utf-8"))
    raw["assets"][0]["host_path"] = "C:\\Model Files\\Qwen 27B (Q4)\\model file.gguf"
    spaced = ContainerRunConfig.model_validate_json(json.dumps(raw))
    reloaded = json.loads(plan_for(spaced, tmp_path).compose_json())
    assert reloaded["services"]["inference"]["volumes"][0]["source"] == "C:\\Model Files\\Qwen 27B (Q4)\\model file.gguf"


def test_networks_labels_and_loopback_port_in_host_process_mode(tmp_path):
    plan = plan_for(example(), tmp_path)
    service, networks = plan.compose["services"]["inference"], plan.compose["networks"]
    assert set(networks) == {"bench", "hostlink"} and networks["bench"]["internal"] is True
    assert "internal" not in networks["hostlink"] and service["networks"] == ["bench", "hostlink"]
    assert service["ports"] == [{"target": 8080, "host_ip": "127.0.0.1", "protocol": "tcp"}]
    expected = {"llmbench.owner": "llmbench", "llmbench.session": "session-1", "llmbench.attempt": ATTEMPT}
    assert plan.labels == expected
    assert service["labels"] == {**expected, "llmbench.role": "inference"}
    for network in networks.values():
        assert network["labels"] == {**expected, "llmbench.role": "network"}


def test_container_mode_has_no_published_port_and_gpu_only_on_inference(tmp_path):
    config = container_config()
    plan = build_compose_plan(config, tmp_path / "run", attempt_id=ATTEMPT, session_id="session-1",
                              grant=grant_for(config))
    inference, evaluator = plan.compose["services"]["inference"], plan.compose["services"]["evaluator"]
    assert set(plan.compose["networks"]) == {"bench"} and "ports" not in inference and "ports" not in evaluator
    assert inference["networks"] == evaluator["networks"] == ["bench"]
    assert "reservations" not in evaluator["deploy"]["resources"] and "devices" not in json.dumps(evaluator)
    assert evaluator["read_only"] is True and evaluator["user"] == "10001:10001"
    assert evaluator["pull_policy"] == "never" and evaluator["cap_drop"] == ["ALL"]
    assert evaluator["command"][:2] == ["--config", "/run/llmbench/config.json"]
    assert [item["target"] for item in evaluator["volumes"]] == ["/run/llmbench/config.json",
                                                                  "/run/llmbench/runtime-policy.json", "/artifacts"]
    assert evaluator["volumes"][0]["read_only"] is True and "/models" not in json.dumps(evaluator)


def test_container_mode_golden_compose_matches_fixture():
    config = container_config(broker=True, host_path="/srv/models/model.gguf")
    plan = build_compose_plan(config, "/runs/attempt", attempt_id=ATTEMPT, session_id="session-1",
                              grant=grant_for(config))
    golden = json.loads((DATA / "evaluator-compose-golden.json").read_text(encoding="utf-8"))
    assert posix_sources(json.loads(plan.compose_json())) == golden
    evaluator = golden["services"]["evaluator"]
    assert evaluator["tmpfs"] == ["/tmp:size=256m,mode=1777"] and "pids_limit" not in evaluator
    assert evaluator["deploy"]["resources"]["limits"]["pids"] == 256
    assert evaluator["init"] is True and evaluator["stop_grace_period"] == "10s"
    assert evaluator["security_opt"] == ["no-new-privileges:true"] and evaluator["read_only"] is True
    assert evaluator["logging"] == {"driver": "local", "options": {"max-size": "16m", "max-file": "2"}}
    # The spool results mount is read-only for the evaluator: the client can never forge a broker result.
    results = next(item for item in evaluator["volumes"] if item["target"] == "/spool/results")
    assert results["read_only"] is True and results["bind"] == {"create_host_path": False}


def test_evaluator_pids_ceiling_is_declared_once_under_deploy_limits(tmp_path):
    # LIVE-006: compose-go compares `pids_limit` with `deploy.resources.limits.pids` whenever a
    # `deploy.resources.limits` block exists, and an absent `pids` counts as 0, so `pids_limit: 256` beside
    # the cpus/memory limits fails `docker compose config` with "can't set distinct values on 'pids_limit'
    # and 'deploy.resources.limits.pids'". The ceiling is declared exactly once, as
    # `deploy.resources.limits.pids`, next to the cpus and memory limits; no service carries `pids_limit`.
    for pids in (256, 64):
        config = container_config(limits={"evaluator_pids": pids})
        assert config.limits.evaluator_pids == pids
        plan = build_compose_plan(config, tmp_path, attempt_id=ATTEMPT, session_id="s", grant=grant_for(config))
        for name, service in plan.compose["services"].items():
            assert "pids_limit" not in service, name
        limits = plan.compose["services"]["evaluator"]["deploy"]["resources"]["limits"]
        assert limits == {"cpus": "2.0", "memory": str(4096 * 1048576), "pids": pids}


def test_container_mode_mounts_only_config_policy_artifacts_and_spool(tmp_path):
    for broker in (False, True):
        config = container_config(broker=broker)
        run_dir = tmp_path / ("with-broker" if broker else "plain")
        plan = build_compose_plan(config, run_dir, attempt_id=ATTEMPT, session_id="s", grant=grant_for(config))
        mounts = {item["target"]: item for item in plan.compose["services"]["evaluator"]["volumes"]}
        expected = {"/run/llmbench/config.json": (run_dir / "plan" / "evaluator-config.json", True),
                    "/run/llmbench/runtime-policy.json": (run_dir / "plan" / "runtime-policy.json", True),
                    "/artifacts": (run_dir / "evaluator", False)}
        if broker:
            expected.update({"/spool/requests": (run_dir / "spool" / "requests", False),
                             "/spool/results": (run_dir / "spool" / "results", True)})
        assert set(mounts) == set(expected)
        for target, (source, read_only) in expected.items():
            mount = mounts[target]
            assert mount["type"] == "bind" and Path(mount["source"]) == source
            assert mount.get("read_only", False) is read_only and mount["bind"] == {"create_host_path": False}
        assert "docker.sock" not in json.dumps(plan.compose)
        assert "/models" not in json.dumps(plan.compose["services"]["evaluator"])


def test_container_mode_has_no_hostlink_port_gpu_or_environment(tmp_path):
    config = container_config(broker=True)
    plan = build_compose_plan(config, tmp_path, attempt_id=ATTEMPT, session_id="s", grant=grant_for(config))
    compose = plan.compose
    assert set(compose["networks"]) == {"bench"} and compose["networks"]["bench"]["internal"] is True
    assert "hostlink" not in json.dumps(compose)
    for service in compose["services"].values():
        assert "ports" not in service and "environment" not in service and "env_file" not in service
        for key in ("privileged", "cap_add", "devices", "network_mode", "pid", "ipc", "userns_mode"):
            assert key not in service
    evaluator = compose["services"]["evaluator"]
    assert "reservations" not in evaluator["deploy"]["resources"] and "nvidia" not in json.dumps(evaluator)
    assert compose["services"]["inference"]["deploy"]["resources"]["reservations"]["devices"][0]["driver"] == "nvidia"


def test_grant_is_required_exactly_in_container_mode(tmp_path):
    config = container_config()
    with pytest.raises(ValueError, match="ChildGrant is required"):
        build_compose_plan(config, tmp_path, attempt_id=ATTEMPT, session_id="s")
    host = read_run_config(EXAMPLES / "candidate.json")
    with pytest.raises(ValueError, match="ChildGrant is required"):
        build_compose_plan(host, tmp_path, attempt_id=ATTEMPT, session_id="s", grant=grant_for(host))
    assert "evaluator" not in build_compose_plan(host, tmp_path, attempt_id=ATTEMPT, session_id="s").compose["services"]
    wrong = grant_for(config).model_copy(update={"artifact_bytes": 4096})
    with pytest.raises(ValueError, match="artifact allocation"):
        build_compose_plan(config, tmp_path, attempt_id=ATTEMPT, session_id="s", grant=wrong)


def test_evaluator_argv_carries_grant_policy_and_artifact_bytes(tmp_path):
    config = container_config()
    grant = grant_for(config, seconds=613)
    plan = build_compose_plan(config, tmp_path, attempt_id=ATTEMPT, session_id="s", grant=grant)
    evaluator = plan.compose["services"]["evaluator"]
    assert evaluator["entrypoint"] == ["python", "-m", "llmbench.container_eval"]
    assert evaluator["command"] == list(evaluator_argv(grant)) == [
        "--config", "/run/llmbench/config.json", "--policy", "/run/llmbench/runtime-policy.json",
        "--grant-seconds", "613", "--artifact-bytes", str(config.limits.evaluator_artifact_bytes)]
    assert "--self-check" not in evaluator["command"] and "--base-url" not in json.dumps(evaluator)
    # The evaluator container itself runs the same argv shape that container_eval.main accepts.
    from llmbench.container_eval import main
    assert main(["--self-check", "--grant-seconds", "1"]) == 2  # grant only accompanies --config


@pytest.mark.parametrize("attempt, session", [("short", "s"), ("G" * 32, "s"), (ATTEMPT, "bad session"),
                                              (ATTEMPT, "")])
def test_unsafe_identities_are_rejected(tmp_path, attempt, session):
    with pytest.raises(ValueError):
        build_compose_plan(read_run_config(EXAMPLES / "candidate.json"), tmp_path,
                           attempt_id=attempt, session_id=session)
