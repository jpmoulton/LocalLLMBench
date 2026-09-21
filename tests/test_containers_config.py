import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmbench.containers.capabilities import help_sha256
from llmbench.containers.config import (KV_PLACEMENTS, ContainerRunConfig, ContainerRunResult,
                                        kv_placement_reading, read_run_config)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "candidate.json"
DATA = Path(__file__).parent / "data"


def payload(**changes):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    for path, value in changes.items():
        target = raw
        *parents, leaf = path.split("__")
        for key in parents:
            target = target[int(key) if key.isdigit() else key]
        target[leaf] = value
    return raw


def build(**changes) -> ContainerRunConfig:
    return ContainerRunConfig.model_validate_json(json.dumps(payload(**changes)))


def test_examples_validate_and_carry_the_real_help_hash():
    config = read_run_config(EXAMPLE)
    help_text = (DATA / "llama-server-help-b11011.txt").read_bytes().decode("utf-8")
    assert config.inference_image.help_sha256 == help_sha256(help_text)
    assert config.engine.reasoning == "off" and config.engine.log_verbosity == 4
    assert [item.to_task_selection().suite for item in config.benchmarks] == ["tool-probes", "niah"]
    mtp = read_run_config(EXAMPLE.with_name("candidate-mtp.json"))
    assert mtp.engine.spec_type == "draft-mtp" and mtp.fingerprint() != config.fingerprint()
    assert config.alias() == "llmbench-" + config.fingerprint()[:12]


@pytest.mark.parametrize("changes", [
    {"unknown": 1}, {"engine__extra_args": ["--tools", "all"]},
    {"inference_image__reference": "ghcr.io/ggml-org/llama.cpp:server-cuda"},
    {"inference_image__image_id": "sha256:abc"}, {"inference_image__help_sha256": None},
    {"inference_image__role": "worker"}, {"inference_image__entrypoint": []},
    {"assets__0__sha256": "E" * 64}, {"assets__0__host_path": "models\\model.gguf"},
    {"assets__0__host_path": "C:\\models\\..\\secret.gguf"}, {"assets__0__host_path": "C:\\models\\a,b.gguf"},
    {"assets__0__host_path": "C:\\models\\model-00001-of-00002.gguf"},
    {"assets__0__host_path": "C:\\models\\a\nb.gguf"}, {"assets__0__container_path": "/models/other.gguf"},
    {"assets": []}, {"engine__n_gpu_layers": "auto"}, {"engine__flash_attn": "auto"},
    {"engine__n_gpu_layers": True}, {"engine__threads": -1}, {"engine__parallel": 2},
    {"engine__fit": "on"}, {"engine__context_shift": True}, {"engine__cache_ram_mib": 8192},
    {"engine__flash_attn": "off", "engine__cache_type_v": "q8_0"},
    {"engine__cache_type_k": "q8_0"}, {"engine__cache_type_k": "q5_1", "engine__cache_type_v": "q5_1"},
    {"engine__ubatch_size": 4096}, {"engine__spec_draft_n_max": 5}, {"engine__log_verbosity": 5},
    {"engine__spec_type": "ngram-cache"}, {"requested_input_tokens": 7600},
    {"speed__output_tokens": 4096}, {"generation__reasoning": "off"}, {"generation__tool_emulation": True},
    {"benchmarks__0__benchmark_id": "coding"},
    # Which option KEYS are legal is the registry's call; the schema refuses unbounded/unsafe values.
    {"benchmarks__0__options": {"bad name!": 1}}, {"benchmarks__0__options": {"x": ["a"] * 65}},
    {"benchmarks__0__options": {"x": "y" * 257}}, {"benchmarks__0__options": {"x": {"nested": 1}}},
    {"benchmarks__0__task_ids": ["a", "a"]}, {"benchmarks": []}, {"label": "Has Space"},
    {"evaluator__mode": "container"}, {"bounds": {"candidate_wall_seconds": 600}}, {"broker": {"max_requests": 0}},
    {"evaluator__watchdog_slack_seconds": 4},
])
def test_invalid_configurations_are_rejected(changes):
    with pytest.raises(ValidationError):
        build(**changes)


def test_a_partial_offload_may_keep_the_kv_cache_on_the_gpu():
    """LIVE-010 option (b): `kv_offload=True` with a partial `n_gpu_layers` is accepted, not refused.

    Refusing it would make every point of the U17 `gpu_layers` sweep unconfigurable (`session.py` validates
    each single-axis variation of the baseline, which pins `kv_offload` to its first member), and these
    settings do not know the model's block count that would decide what "partial" means.
    """
    engine = build(engine__n_gpu_layers=48, engine__kv_offload=True).engine
    assert engine.n_gpu_layers == 48 and engine.kv_offload is True
    assert engine.consistent_kv_placements(66) == ("gpu", "split")  # KV follows the layers
    assert engine.consistent_kv_placements(48) == ("gpu",)  # an int at or above the block count is a full offload
    assert engine.consistent_kv_placements(None) == ("gpu", "split")  # an unknown block count decides nothing
    assert build(engine__n_gpu_layers="all").engine.consistent_kv_placements(66) == ("gpu",)
    assert build(engine__n_gpu_layers="all").engine.consistent_kv_placements(None) == ("gpu",)
    assert build(engine__n_gpu_layers=0).engine.consistent_kv_placements(66) == ("ram",)
    for layers in ("all", 48, 0):  # `--no-kv-offload` means system RAM whatever the layers do
        assert build(engine__n_gpu_layers=layers, engine__kv_offload=False).engine.consistent_kv_placements(66) \
            == ("ram",)


def test_a_short_partial_offload_can_leave_every_kv_layer_in_system_ram():
    """REV-OFF-02: the rule admitted the hybrid case in one direction only.

    The live target carries a KV cache on 16 of its 66 offloadable blocks (`llama_kv_cache: size = 336.00 MiB
    (5376 cells, 16 layers, 1/1 seqs)`), roughly one in four, so an offload shorter than that spacing can land
    entirely between two KV-bearing blocks and leave the whole cache in system RAM. Refusing `ram` there hard-
    failed a legitimate candidate at `verify` and threw its measured throughput away; the judgement is made on
    the layer counts the log printed, not on a guess about which blocks carry a cache.
    """
    for layers in (1, 2, 3, 4):  # 4 * 16 = 64 < 66: the offloaded tail can fall between two KV-bearing blocks
        engine = build(engine__n_gpu_layers=layers).engine
        assert engine.consistent_kv_placements(66, 16) == ("gpu", "split", "ram"), layers
    for layers in (5, 16, 32, 48, 65):  # 5 * 16 = 80 > 66: some KV-bearing block must have reached the GPU
        assert build(engine__n_gpu_layers=layers).engine.consistent_kv_placements(66, 16) == ("gpu", "split")
    # A DENSE model has a KV cache on every block but the output entry, so only a single offloaded layer can
    # miss them all -- the same arithmetic, and it keeps a dense all-RAM cache the contradiction it is.
    assert build(engine__n_gpu_layers=1).engine.consistent_kv_placements(65, 64) == ("gpu", "split", "ram")
    assert build(engine__n_gpu_layers=2).engine.consistent_kv_placements(65, 64) == ("gpu", "split")
    # An absent, zero or incoherent layer count admits nothing extra: the pre-repair rule stands.
    for counts in ((66, None), (None, 16), (66, 0), (66, 67), (None, None)):
        assert build(engine__n_gpu_layers=2).engine.consistent_kv_placements(*counts) == ("gpu", "split"), counts
    # The other branches are untouched by the layer count.
    assert build(engine__n_gpu_layers=0).engine.consistent_kv_placements(66, 16) == ("ram",)
    assert build(engine__n_gpu_layers="all").engine.consistent_kv_placements(66, 16) == ("gpu",)
    assert build(engine__n_gpu_layers=2, engine__kv_offload=False).engine.consistent_kv_placements(66, 16) \
        == ("ram",)


@pytest.mark.parametrize("effective, detail, expected", [
    ("split", "ram 84 MiB on CPU / gpu 252 MiB on CUDA0", ("split", "ram 84 MiB on CPU / gpu 252 MiB on CUDA0")),
    ("gpu", "336 MiB on CUDA0", ("gpu", "336 MiB on CUDA0")),
    ("ram", "", ("ram", "")),
    # Rows sealed by earlier builds: LIVE-010's split string, and the bool that predates it.
    ("split (ram 84 MiB / gpu 252 MiB on CUDA0)", "", ("split", "ram 84 MiB / gpu 252 MiB on CUDA0")),
    (True, "", ("gpu", "")), (False, "", ("ram", "")),
    # Anything else is "no placement observed", never a placement: the reading fails closed.
    (None, "", (None, "")), ("", "", (None, "")), ("gpu-ish", "", (None, "")), (1, "", (None, "")),
    ({"placement": "split"}, "", (None, "")),
])
def test_a_kv_placement_is_read_from_the_row_never_inferred_from_truthiness(effective, detail, expected):
    """REV-OFF-01: one reading for every shape a stored `kv_offload` row can carry."""
    assert kv_placement_reading(effective, detail) == expected
    assert expected[0] is None or expected[0] in KV_PLACEMENTS


def test_accepted_variants():
    assert build(engine__flash_attn="off", engine__cache_type_k="q8_0").engine.cache_type_k == "q8_0"
    assert build(engine__cache_type_k="q8_0", engine__cache_type_v="q8_0").engine.cache_type_v == "q8_0"
    assert build(engine__n_gpu_layers=40).engine.n_gpu_layers == 40
    assert build(engine__spec_type="draft-mtp", engine__spec_draft_n_max=5).engine.spec_draft_n_max == 5
    assert build(assets__0__host_path="/srv/models/model.gguf").model.host_path.startswith("/srv")


def test_fingerprint_ignores_display_fields_and_tracks_measured_ones():
    base = build()
    same = build(label="other-name", session_id="s-2", parent_grant_seconds=900,
                 assets__0__host_path="D:\\elsewhere\\model.gguf", inference_image__source="different note",
                 bounds={"candidate_wall_seconds": 3600})
    assert same.fingerprint() == base.fingerprint()
    for changes in ({"engine__cache_type_k": "q8_0", "engine__cache_type_v": "q8_0"},
                    {"engine__spec_type": "draft-mtp"}, {"assets__0__sha256": "1" * 64},
                    {"engine__threads": 6}, {"benchmarks__0__seed": 7}, {"requested_input_tokens": 1024}):
        assert build(**changes).fingerprint() != base.fingerprint()


def test_configuration_is_immutable():
    config = build()
    with pytest.raises((ValidationError, TypeError)):
        config.engine.ctx_size = 4096
    with pytest.raises((ValidationError, TypeError)):
        config.label = "changed"
    with pytest.raises(TypeError):
        config.benchmarks[0].options["shots"] = 1


def test_result_invariants():
    base = {"session_id": "s", "attempt_id": "a" * 32, "config_fingerprint": "0" * 64, "project_name": "p",
            "state": "completed", "synthetic": True, "started_utc": "t", "finished_utc": "t",
            "elapsed_seconds": 1.0, "budget_charged_seconds": 1.0}
    assert ContainerRunResult.model_validate(base).abort_campaign is False
    for changes in ({"abort_campaign": True}, {"state": "cleanup-uncertain"},
                    {"failure_reasons": ("speed",)}, {"speed": {"value": float("nan")}}):
        with pytest.raises((ValidationError, ValueError)):
            ContainerRunResult.model_validate({**copy.deepcopy(base), **changes})
    uncertain = ContainerRunResult.model_validate({**base, "state": "cleanup-uncertain", "abort_campaign": True})
    assert uncertain.abort_campaign is True


def container(**changes):
    image = {"role": "evaluator", "reference": "sha256:" + "b" * 64, "image_id": "sha256:" + "b" * 64,
             "entrypoint": ["python", "-m", "llmbench.container_eval"]}
    return build(evaluator={"mode": "container", "image": image}, **changes)


def test_broker_settings_bounds_and_coding_requires_broker():
    from llmbench.coding.sandbox import SandboxLimits
    from llmbench.containers.config import BrokerSettings
    defaults = BrokerSettings()
    assert (defaults.max_requests, defaults.max_request_bytes, defaults.max_patch_bytes,
            defaults.fixture_timeout_seconds) == (8, 2_097_152, 1_048_576, 180)
    assert defaults.worker_limits == SandboxLimits()
    for bad in ({"max_requests": 0}, {"max_requests": 1025}, {"max_request_bytes": 4095}, {"max_patch_bytes": 1023},
                {"fixture_timeout_seconds": 29}, {"fixture_timeout_seconds": 901}, {"unknown": 1},
                {"worker_limits": {"pids": 257}}):
        with pytest.raises(ValidationError):
            BrokerSettings(**bad)
    coding = {"benchmark_id": "coding", "revision": "private-coding-v1", "task_ids": ["python/chunks"]}
    with pytest.raises(ValidationError, match="broker"):
        build(benchmarks=[coding])
    with_broker = build(benchmarks=[coding], broker={"max_requests": 2})
    assert with_broker.broker.max_requests == 2 and build().broker is None
    assert with_broker.fingerprint() != build(benchmarks=[coding], broker={"max_requests": 3}).fingerprint()


def test_child_grant_and_image_bundle_are_strict():
    from llmbench.containers.config import ChildGrant, ImageBundle, read_image_bundle
    grant = ChildGrant(grant_seconds=900, artifact_bytes=1024, issued_utc="t", issued_host_offset_seconds=1.5,
                       watchdog_slack_seconds=30)
    assert grant.grant_seconds == 900
    for bad in ({"grant_seconds": 0}, {"grant_seconds": "900"}, {"artifact_bytes": 0}, {"issued_utc": ""},
                {"issued_host_offset_seconds": -1.0}, {"watchdog_slack_seconds": 4}, {"extra": 1}):
        with pytest.raises(ValidationError):
            ChildGrant(**{**grant.model_dump(), **bad})
    config = read_run_config(EXAMPLE)
    inference = config.inference_image.model_dump()
    evaluator = {"role": "evaluator", "reference": "sha256:" + "c" * 64, "image_id": "sha256:" + "c" * 64,
                 "entrypoint": ["python", "-m", "llmbench.container_eval"]}
    base = {"prepared_utc": "2026-09-18T00:00:00+00:00", "inference": inference, "evaluator": evaluator,
            "help_sha256": inference["help_sha256"], "registry_digest": "0" * 64,
            "evaluator_build": {"plan": {"base_image": "python:3.12-slim-bookworm@sha256:" + "a" * 64}}}
    bundle = ImageBundle.model_validate_json(json.dumps(base))
    assert bundle.worker is None and bundle.schema_version == 1
    with pytest.raises(TypeError):
        bundle.evaluator_build["plan"]["base_image"] = "changed"
    for changes in ({"schema_version": 2}, {"help_sha256": "1" * 64}, {"registry_digest": "xyz"},
                    {"evaluator": {**evaluator, "entrypoint": ["python"]}}, {"evaluator": {**evaluator, "role": "worker"}},
                    {"inference": {**inference, "role": "evaluator"}}, {"worker": {**evaluator, "role": "evaluator"}},
                    {"evaluator_build": {"nan": float("nan")}}, {"unknown": True}):
        with pytest.raises((ValidationError, ValueError)):
            ImageBundle.model_validate_json(json.dumps({**copy.deepcopy(base), **changes}))
    worker = {**evaluator, "role": "worker", "entrypoint": ["python3"]}
    with_worker = ImageBundle.model_validate_json(json.dumps({**base, "worker": worker}))
    assert with_worker.worker.role == "worker"
    path = Path(__file__).resolve().parent / "data" / "evaluator-bundle-tmp.json"
    try:
        path.write_text(with_worker.model_dump_json(), encoding="utf-8")
        assert read_image_bundle(path) == with_worker
    finally:
        path.unlink(missing_ok=True)


def test_evaluator_artifact_allocation_must_fit_under_cap():
    from llmbench.containers.config import ResourceLimits, terminal_reserve_bytes
    limits = ResourceLimits()
    assert (limits.evaluator_artifact_bytes, limits.evaluator_pids, limits.evaluator_tmpfs_mib) == (268_435_456, 256, 256)
    assert terminal_reserve_bytes(1_073_741_824) == 262144 and terminal_reserve_bytes(1_048_576) == 262144
    assert container().limits.child_allocation_fits()
    for bad in ({"max_artifact_bytes": 268_435_456 + 262144}, {"max_artifact_bytes": 1_048_576},
                {"evaluator_artifact_bytes": 1_073_741_824 - 262144}):
        with pytest.raises(ValidationError, match="terminal reserve"):
            container(limits=bad)
    small = container(limits={"max_artifact_bytes": 1_048_576, "evaluator_artifact_bytes": 524288})
    assert small.limits.evaluator_artifact_bytes == 524288
    # Host-process candidates do not allocate a child tree; a small cap alone stays valid there.
    assert build(limits={"max_artifact_bytes": 1_048_576}).limits.max_artifact_bytes == 1_048_576
    for bad in ({"evaluator_artifact_bytes": 65535}, {"evaluator_pids": 15}, {"evaluator_tmpfs_mib": 4097}):
        with pytest.raises(ValidationError):
            ResourceLimits(**bad)


def test_fingerprint_ignores_watchdog_slack():
    base = container()
    assert base.evaluator.watchdog_slack_seconds == 30
    image = base.evaluator.image.model_dump()
    slack = build(evaluator={"mode": "container", "image": image, "watchdog_slack_seconds": 120})
    assert slack.fingerprint() == base.fingerprint()
    assert build(evaluator={"mode": "container", "image": {**image, "image_id": "sha256:" + "d" * 64,
                                                            "reference": "sha256:" + "d" * 64}}).fingerprint() != base.fingerprint()
    assert container(limits={"evaluator_artifact_bytes": 134_217_728}).fingerprint() != base.fingerprint()
    with pytest.raises(ValidationError):
        build(evaluator={"mode": "host-process", "watchdog_slack_seconds": 301})
