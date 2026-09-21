import hashlib
import json
import os
from pathlib import Path

import pytest

from llmbench.coding.sandbox import WorkerResult
from llmbench.config import hash_file
from llmbench.containers.config import (KV_PLACEMENTS, ContainerRunConfig, ImageRef, ModelAsset,
                                        kv_placement_reading)
from llmbench.containers.plan import build_server_argv
from llmbench.containers.preflight import (PreflightError, PreflightTimeout, check_gpu_headroom,
                                           check_path_lengths, reverify_identity, verify_image,
                                           verify_model_asset)
from llmbench.containers.readback import (REQUIRED_CONTROLS, build_settings_evidence, parse_startup_log,
                                          settings_verified, unverified_required)

DATA = Path(__file__).parent / "data"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.json"
IMAGE_ID = "sha256:" + "d" * 64
REFERENCE = "ghcr.io/ggml-org/llama.cpp@" + IMAGE_ID


def asset_for(path: Path, content: bytes = b"GGUF-bytes", **changes) -> ModelAsset:
    path.write_bytes(content)
    values = {"host_path": str(path), "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
              "quantization": "Q4_K_M", "model_name": "fixture", "source_revision": "fixture-v1", **changes}
    return ModelAsset(**values)


def test_model_asset_is_hashed_and_identified(tmp_path):
    asset = asset_for(tmp_path / "model.gguf")
    evidence = verify_model_asset(asset)
    assert evidence["sha256"] == asset.sha256 == hash_file(asset.host_path)
    assert evidence["size"] == asset.size_bytes and evidence["nlink"] == 1 and evidence["file_id"]
    assert verify_model_asset(asset, hasher=hash_file)["sha256"] == asset.sha256
    assert reverify_identity(asset, evidence)["unchanged"] is True


def test_model_asset_mismatches_are_rejected(tmp_path):
    with pytest.raises(PreflightError, match="size"):
        verify_model_asset(asset_for(tmp_path / "a.gguf", size_bytes=3))
    with pytest.raises(PreflightError, match="SHA256"):
        verify_model_asset(asset_for(tmp_path / "b.gguf", sha256="0" * 64))
    missing = asset_for(tmp_path / "c.gguf")
    Path(missing.host_path).unlink()
    with pytest.raises(PreflightError, match="not readable"):
        verify_model_asset(missing)
    directory = tmp_path / "directory.gguf"
    directory.mkdir()
    with pytest.raises(PreflightError, match="regular file"):
        verify_model_asset(ModelAsset(host_path=str(directory), size_bytes=1, sha256="0" * 64, quantization="q",
                                      model_name="m", source_revision="r"))


def test_links_are_rejected(tmp_path):
    asset = asset_for(tmp_path / "model.gguf")
    os.link(asset.host_path, tmp_path / "second-name.gguf")
    with pytest.raises(PreflightError, match="hard links"):
        verify_model_asset(asset)
    real = asset_for(tmp_path / "real.gguf")
    link = tmp_path / "link.gguf"
    try:
        os.symlink(real.host_path, link)
    except OSError:
        pytest.skip("symlinks are unavailable to this account")
    with pytest.raises(PreflightError, match="regular file"):
        verify_model_asset(real.model_copy(update={"host_path": str(link)}))


def test_replacement_during_or_after_hashing_is_detected(tmp_path):
    asset = asset_for(tmp_path / "model.gguf")

    def replacing_hasher(path):
        digest = hash_file(path)
        Path(path).write_bytes(b"GGUF-BYTES")  # Same size, different bytes and timestamp.
        os.utime(path, ns=(1, 1))
        return digest
    with pytest.raises(PreflightError, match="changed while"):
        verify_model_asset(asset, hasher=replacing_hasher)
    stable = asset_for(tmp_path / "stable.gguf")
    evidence = verify_model_asset(stable)
    os.utime(stable.host_path, ns=(5, 5))
    assert reverify_identity(stable, evidence)["unchanged"] is False
    Path(stable.host_path).unlink()
    assert reverify_identity(stable, evidence)["unchanged"] is False


def test_hash_deadline_is_enforced(tmp_path):
    asset = asset_for(tmp_path / "model.gguf")
    ticks = iter(range(0, 1000, 10))
    with pytest.raises(PreflightTimeout):
        verify_model_asset(asset, deadline=5, clock=lambda: next(ticks))
    ticks = iter(range(0, 1000, 10))
    with pytest.raises(PreflightTimeout):
        verify_model_asset(asset, hasher=hash_file, deadline=5, clock=lambda: next(ticks))


class InspectExecutor:
    def __init__(self, payload=None, result=None):
        self.payload, self.result, self.calls = payload, result, []

    def run(self, argv, **limits):
        self.calls.append((argv, limits))
        return self.result or WorkerResult("completed", 0, json.dumps(self.payload).encode())


def inspection(**changes):
    raw = {"Id": IMAGE_ID, "RepoDigests": [REFERENCE], "Os": "linux", "Architecture": "amd64", "Size": 1,
           "Created": "2026-09-17T07:25:06Z",
           "Config": {"Entrypoint": ["/app/llama-server"], "Env": ["PATH=/usr/bin", "LLAMA_ARG_HOST=0.0.0.0"]}}
    raw.update(changes)
    return raw


def image_ref(**changes):
    values = {"role": "inference", "reference": REFERENCE, "image_id": IMAGE_ID,
              "entrypoint": ("/app/llama-server",), "build_info": "b11011-aa39d7a3e", "help_sha256": "0" * 64}
    return ImageRef(**{**values, **changes})


def test_image_identity_is_verified_with_one_exact_inspect():
    executor = InspectExecutor(inspection())
    evidence = verify_image(executor, image_ref())
    assert executor.calls[0][0] == ("docker", "image", "inspect", "--format", "{{json .}}", REFERENCE)
    assert evidence["id"] == IMAGE_ID and evidence["platform"] == "linux/amd64"
    assert evidence["environment_names"] == ["LLAMA_ARG_HOST", "PATH"] and "0.0.0.0" not in json.dumps(evidence)
    assert verify_image(InspectExecutor(inspection(RepoDigests=[])), image_ref(reference=IMAGE_ID))["id"] == IMAGE_ID


@pytest.mark.parametrize("payload, fragment", [
    (inspection(Id="sha256:" + "e" * 64), "does not equal configured"),
    (inspection(RepoDigests=["other/repo@" + IMAGE_ID]), "RepoDigests"),
    (inspection(Architecture="arm64"), "platform"),
    (inspection(Config={"Entrypoint": ["/bin/sh", "-c"], "Env": []}), "entrypoint"),
    (inspection(Config={"Entrypoint": ["/app/llama-server"], "Env": ["LLAMA_ARG_CTX_SIZE=4096"]}),
     "LLAMA_ARG_CTX_SIZE"),
    ([inspection()], "unreadable"), ({"Config": {}}, "unreadable"),
])
def test_image_mismatches_are_rejected(payload, fragment):
    with pytest.raises(PreflightError, match=fragment):
        verify_image(InspectExecutor(payload), image_ref())


def test_missing_image_is_rejected():
    for result in (WorkerResult("completed", 1, b"", b"No such image"), WorkerResult("timeout", None),
                   WorkerResult("completed", 0, b"not json")):
        with pytest.raises(PreflightError):
            verify_image(InspectExecutor(result=result), image_ref())


def test_gpu_headroom_is_a_conflict_not_a_kill():
    sample = {"gpus": [{"index": 0.0, "name": "RTX 5090", "memory_used_mib": 1800.0, "memory_total_mib": 32607.0}],
              "gpu_error": None}
    assert check_gpu_headroom(sample, 4096)["memory_used_mib"] == 1800.0
    with pytest.raises(PreflightError, match="already has 1800 MiB"):
        check_gpu_headroom(sample, 1024)
    for broken in ({"gpus": [], "gpu_error": "nvidia-smi missing"}, {"gpus": sample["gpus"], "gpu_error": "x"},
                   {"gpus": [{"index": 0.0, "memory_used_mib": None}], "gpu_error": None}, {}):
        with pytest.raises(PreflightError, match="unavailable"):
            check_gpu_headroom(broken, 4096)
    with pytest.raises(PreflightError, match="GPU 1"):
        check_gpu_headroom(sample, 4096, device_id="1")


def test_long_run_directories_are_rejected(tmp_path):
    check_path_lengths(tmp_path / "run")
    with pytest.raises(PreflightError, match="180"):
        check_path_lengths(tmp_path / ("x" * 60) / ("y" * 60) / ("z" * 60))


def log(name: str) -> str:
    return (DATA / f"startup-b11011-{name}.log").read_text(encoding="utf-8")


def test_verbosity_three_log_has_no_memory_evidence():
    parsed = parse_startup_log(log("lv3"))
    assert parsed["verbosity"] == 3 and parsed["n_slots"] == 1 and parsed["n_ctx_slot"] == 8192
    assert parsed["model_loaded_seconds"] == pytest.approx(56.192921)
    for key in ("kv_cache", "offloaded_layers", "flash_attn", "n_batch", "speculative", "n_threads"):
        assert key not in parsed


def test_verbosity_four_log_is_parsed():
    parsed = parse_startup_log(log("lv4"))
    assert parsed["build"] == "b11011-aa39d7a3e" and parsed["verbosity"] == 4 and parsed["load_mode"] == "none"
    assert (parsed["offloaded_layers"], parsed["total_layers"]) == (66, 66)
    assert (parsed["n_ctx"], parsed["n_batch"], parsed["n_ubatch"], parsed["n_seq_max"]) == (131072, 2048, 512, 1)
    assert parsed["flash_attn"] == "enabled" and parsed["kv_unified"] == "false"
    assert parsed["kv_cache"] == {"size_mib": 8192.0, "cells": 131072, "layers": 16, "k_type": "f16",
                                  "k_mib": 4096.0, "v_type": "f16", "v_mib": 4096.0}
    assert parsed["kv_buffers"] == [{"device": "CUDA0", "mib": 8192.0}] and parsed["draft_kv_caches"] == []
    assert (parsed["n_threads"], parsed["n_threads_batch"]) == (8, 8)
    assert parsed["fa_quants"] == [["q4_0", "q4_0"], ["q8_0", "q8_0"], ["f16", "f16"], ["bf16", "bf16"]]
    assert parsed["speculative"] == {"implementations": []} and parsed["prompt_cache_mib"] == 8192
    assert (parsed["n_slots"], parsed["n_ctx_slot"]) == (1, 131072) and parsed["warmup"] is True
    assert parsed["model_loaded_seconds"] == pytest.approx(56.395523) and parsed["thinking"] == 1
    assert parsed["listening"] == "http://0.0.0.0:8080" and parsed["ui_disabled"] is True


def test_speculative_log_keeps_target_and_draft_apart():
    parsed = parse_startup_log(log("lv4-mtp"))
    assert parsed["speculative"] == {"implementations": ["draft-mtp"], "n_max": 3, "n_min": 0}
    assert parsed["kv_cache"]["layers"] == 16 and parsed["kv_cache"]["size_mib"] == 8192.0
    assert parsed["draft_kv_caches"] == [{"size_mib": 512.0, "cells": 131072, "layers": 1, "k_type": "f16",
                                          "k_mib": 256.0, "v_type": "f16", "v_mib": 256.0}]
    assert parsed["model_loaded_seconds"] == pytest.approx(56.783198)
    assert parse_startup_log("") == {"parser_version": "llama-server-startup-v1"}


def config_for(**engine) -> ContainerRunConfig:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["engine"].update(engine)
    raw["requested_input_tokens"] = 512
    return ContainerRunConfig.model_validate_json(json.dumps(raw))


def readback_for(config, **props):
    body = json.loads((DATA / "props-b11011.json").read_text(encoding="utf-8"))["body"]
    body.update({"model_alias": config.alias(), **props})
    body["default_generation_settings"]["n_ctx"] = props.get("n_ctx", config.engine.ctx_size)
    slots = json.loads((DATA / "slots-b11011.json").read_text(encoding="utf-8"))["body"]
    slots[0]["n_ctx"] = body["default_generation_settings"]["n_ctx"]
    return {"props": body, "slots": slots, "models": {}}


def startup_for(name="lv4", **changes):
    parsed = parse_startup_log(log(name).replace("prompt cache is enabled, size limit: 8192 MiB",
                                                 "prompt cache is disabled - use `--cache-ram N` to enable it")
                               .replace("thinking = 1", "thinking = 0"))
    return {**parsed, **changes}


def evidence_for(config, startup, readback=None, overflow=None):
    rows = build_settings_evidence(config.engine, build_server_argv(config), readback, startup, overflow,
                                   alias=config.alias(), build_info=config.inference_image.build_info)
    return rows, {row.control: row for row in rows}


def test_evidence_statuses_for_a_matching_run():
    config = config_for(ctx_size=131072)
    rows, by_name = evidence_for(config, startup_for(), readback_for(config), {"passed": True})
    assert by_name["ctx_size"].status == "verified-api" and by_name["parallel"].status == "verified-api"
    assert by_name["alias"].status == "verified-api" and by_name["build_info"].status == "verified-api"
    for control in ("n_gpu_layers", "cache_type_k", "cache_type_v", "kv_offload", "flash_attn", "load_mode",
                    "batch_size", "ubatch_size", "threads", "threads_batch", "cache_ram_mib", "spec_type",
                    "log_verbosity", "warmup", "reasoning"):
        assert by_name[control].status == "verified-log", control
    assert by_name["n_gpu_layers"].effective == "66/66" and by_name["flash_attn"].effective == "on"
    assert by_name["context_shift"].status == "verified-behavior"
    assert by_name["fit"].status == "verified-behavior" and by_name["fit"].effective == "off"
    for control in ("cache_prompt", "cache_reuse", "jinja", "reasoning_format", "reasoning_budget"):
        assert by_name[control].status == "argv-accepted", control
    assert "spec_draft_n_max" not in by_name and "swa_full" not in by_name
    assert {row.control for row in rows if row.required} == set(REQUIRED_CONTROLS) - {"spec_draft_n_max"}
    assert unverified_required(rows) == [] and settings_verified(rows) is True
    blocked = [row if row.control != "fit" else row.model_copy(update={"status": "argv-accepted"}) for row in rows]
    assert unverified_required(blocked) == ["fit"] and settings_verified(blocked) is False


def test_live_compose_log_verifies_the_example_candidate():
    config = ContainerRunConfig.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))
    startup = parse_startup_log(log("lv4-compose"))
    assert startup["prompt_cache_mib"] == 0 and startup["thinking"] == 0 and startup["n_ctx"] == 8192
    assert startup["kv_cache"]["cells"] == 8192 and startup["model_loaded_seconds"] == pytest.approx(55.969794)
    rows, by_name = evidence_for(config, startup, readback_for(config), {"passed": True})
    assert not [row.control for row in rows if row.status in {"mismatch", "unobserved"}]
    assert unverified_required(rows) == [] and by_name["reasoning"].status == "verified-log"
    assert by_name["flash_attn_kernel"].status == "verified-log" and not by_name["flash_attn_kernel"].required
    assert by_name["flash_attn_kernel"].effective == [["q4_0", "q4_0"], ["q8_0", "q8_0"], ["f16", "f16"],
                                                      ["bf16", "bf16"]]
    unsupported = config_for(cache_type_k="q8_0", cache_type_v="q8_0")
    _, by_name = evidence_for(unsupported, {**startup, "fa_quants": [["f16", "f16"]]}, readback_for(unsupported))
    assert by_name["flash_attn_kernel"].status == "mismatch"
    assert "flash_attn_kernel" not in evidence_for(config_for(flash_attn="off"), startup)[1]


def test_speculative_and_partial_offload_evidence():
    config = config_for(ctx_size=131072, spec_type="draft-mtp", n_gpu_layers=40, reasoning="auto")
    startup = startup_for("lv4-mtp", offloaded_layers=40)
    _, by_name = evidence_for(config, startup, readback_for(config))
    assert by_name["spec_type"].effective == "draft-mtp" and by_name["spec_type"].status == "verified-log"
    assert by_name["spec_draft_n_max"].status == "verified-log" and by_name["spec_draft_n_max"].required
    assert by_name["n_gpu_layers"].status == "verified-log" and by_name["reasoning"].status == "argv-accepted"
    assert by_name["context_shift"].status == "argv-accepted"
    assert by_name["fit"].status == "verified-behavior"  # A partial offload is still the requested offload.


def test_mismatches_and_missing_evidence_block_verification():
    config = config_for(ctx_size=131072)
    startup = startup_for(flash_attn="disabled", n_batch=1024, offloaded_layers=30, prompt_cache_mib=8192,
                          kv_buffers=[{"device": "CPU", "mib": 1.0}], n_threads=16,
                          kv_placement={"placement": "ram", "ram_mib": 1.0, "gpu_mib": 0.0,
                                        "ram_devices": ["CPU"], "gpu_devices": []},
                          speculative={"implementations": ["draft-mtp"], "n_max": 3, "n_min": 0})
    rows, by_name = evidence_for(config, startup, readback_for(config, n_ctx=4096, total_slots=4,
                                                               model_alias="other", build_info="b1-x"))
    for control in ("ctx_size", "parallel", "n_gpu_layers", "kv_offload", "flash_attn", "batch_size", "threads",
                    "cache_ram_mib", "spec_type", "alias", "build_info"):
        assert by_name[control].status == "mismatch", control
    assert settings_verified(rows) is False
    rows, by_name = evidence_for(config, parse_startup_log(log("lv3")))
    assert by_name["cache_type_k"].status == "unobserved" and by_name["ctx_size"].status == "unobserved"
    assert by_name["parallel"].status == "verified-log" and by_name["warmup"].status == "unobserved"
    assert "cache_type_k" in unverified_required(rows) and settings_verified(rows) is False
    assert settings_verified(()) is False
    disagreeing = startup_for(n_ctx=8192)
    assert evidence_for(config, disagreeing, readback_for(config))[1]["ctx_size"].status == "mismatch"


def live_baseline(drop=(), readback=True, argv_fit=("--fit", "off"), **startup_changes):
    """`artifacts/container-campaign/live-full-1/candidates/000-baseline`, from its own saved artifacts.

    Its engine is the q4-8k example at ctx 5376; the startup log and the `/props`+`/slots` readback in
    `tests/data/` are copies of that candidate's files and the overflow probe repeats its saved result.
    """
    config = config_for(ctx_size=5376)
    startup = {key: value for key, value in parse_startup_log(log("live-full-1-baseline")).items()
               if key not in drop}
    startup.update(startup_changes)
    saved = json.loads((DATA / "readback-b11011-live-full-1-baseline.json").read_text(encoding="utf-8"))
    probe = {"passed": True, "sent_tokens": 5440, "expected_n_ctx": 5376, "status": 400,
             "error_type": "exceed_context_size_error", "reported_n_ctx": 5376}
    argv = list(build_server_argv(config))
    argv[argv.index("--fit"):argv.index("--fit") + 2] = list(argv_fit)
    rows = build_settings_evidence(config.engine, tuple(argv), saved if readback else None, startup, probe,
                                   alias=saved["props"]["model_alias"],
                                   build_info=saved["props"]["build_info"])
    return rows, {row.control: row for row in rows}


def test_the_live_startup_log_carries_no_fit_line():
    """Step 1 of the `fit` investigation: b11011 says nothing about fitting, even at verbosity 4."""
    text = log("live-full-1-baseline")
    assert "verbosity = 4" in text and not [line for line in text.splitlines() if "fit" in line.lower()]
    parsed = parse_startup_log(text)
    assert parsed["build"] == "b11011-aa39d7a3e" and parsed["n_ctx"] == 5376 and parsed["n_batch"] == 2048
    assert parsed["n_ubatch"] == 512 and (parsed["offloaded_layers"], parsed["total_layers"]) == (66, 66)


def test_fit_is_verified_from_the_live_campaign_baseline_artifacts():
    rows, by_name = live_baseline()
    fit = by_name["fit"]
    assert fit.status == "verified-behavior" and fit.effective == "off" and fit.required
    for observation in ("ctx_size=5376 (verified-api)", "n_gpu_layers=66/66 (verified-log)",
                        "batch_size=2048 (verified-log)", "ubatch_size=512 (verified-log)"):
        assert observation in fit.evidence, observation
    assert by_name["ctx_size"].status == "verified-api" and by_name["context_shift"].effective is False
    assert by_name["kv_offload"].effective == "gpu" and by_name["cache_type_v"].effective == "f16"
    assert not [row.control for row in rows if row.status in {"mismatch", "unobserved"}]
    # The saved settings-evidence.json of that candidate listed `fit` as its only unverified required control.
    assert unverified_required(rows) == [] and settings_verified(rows) is True


@pytest.mark.parametrize("control, changes", [
    ("ctx_size", {"readback": False, "drop": ("n_ctx",)}),
    ("n_gpu_layers", {"drop": ("offloaded_layers", "total_layers")}),
    ("batch_size", {"n_batch": 1024}),
    ("ubatch_size", {"n_ubatch": 256}),
])
def test_fit_is_not_verified_when_an_adjustable_control_is_not_verified(control, changes):
    rows, by_name = live_baseline(**changes)
    assert by_name[control].status in {"unobserved", "mismatch"}, by_name[control]
    assert by_name["fit"].status == "argv-accepted" and by_name["fit"].effective is None
    assert control in by_name["fit"].evidence and "not verified" in by_name["fit"].evidence
    assert "fit" in unverified_required(rows) and settings_verified(rows) is False


def live_offload(name, n_gpu_layers="all", kv_offload=True, **startup_changes):
    """`artifacts/container-campaign/live-offload-2/candidates/...`, from its own saved startup log.

    Those candidates run the q4-8k example at ctx 5376 -- the same configuration as the `live-full-1` baseline
    apart from the two U17 offload axes, line for line (`n_ctx`, `n_batch`, threads and cache types are
    identical) -- so that baseline's saved `/props` + `/slots` readback and overflow probe stand in for theirs.
    Only the startup log, which is the sole evidence for the KV placement, comes from the offload campaign.
    """
    config = config_for(ctx_size=5376, n_gpu_layers=n_gpu_layers, kv_offload=kv_offload)
    startup = {**parse_startup_log(log(name)), **startup_changes}
    saved = json.loads((DATA / "readback-b11011-live-full-1-baseline.json").read_text(encoding="utf-8"))
    probe = {"passed": True, "sent_tokens": 5440, "expected_n_ctx": 5376, "status": 400,
             "error_type": "exceed_context_size_error", "reported_n_ctx": 5376}
    rows = build_settings_evidence(config.engine, build_server_argv(config), saved, startup, probe,
                                   alias=saved["props"]["model_alias"], build_info=saved["props"]["build_info"])
    return rows, {row.control: row for row in rows}


def test_partial_offload_splits_the_kv_cache_in_the_live_startup_logs():
    """LIVE-010 step 1: the three real logs show the KV cache is not a two-state thing."""
    partial = parse_startup_log(log("live-offload-2-partial-48"))
    assert (partial["offloaded_layers"], partial["total_layers"]) == (48, 66)
    assert partial["kv_placement"] == {"placement": "split", "ram_mib": 84.0, "gpu_mib": 252.0,
                                       "ram_devices": ["CPU"], "gpu_devices": ["CUDA0"]}
    assert partial["kv_cache"]["size_mib"] == 84.0 + 252.0 and partial["kv_cache"]["layers"] == 16
    full = parse_startup_log(log("live-offload-2-baseline"))
    assert (full["offloaded_layers"], full["total_layers"]) == (66, 66)
    assert full["kv_placement"] == {"placement": "gpu", "ram_mib": 0.0, "gpu_mib": 336.0,
                                    "ram_devices": [], "gpu_devices": ["CUDA0"]}
    ram = parse_startup_log(log("live-offload-2-kv-ram"))  # `--no-kv-offload` with every block on the GPU
    assert (ram["offloaded_layers"], ram["total_layers"]) == (66, 66)
    assert ram["kv_placement"] == {"placement": "ram", "ram_mib": 336.0, "gpu_mib": 0.0,
                                   "ram_devices": ["CPU"], "gpu_devices": []}
    assert "kv_placement" not in parse_startup_log(log("lv3"))  # no KV buffer line, so no placement claim


def test_a_draft_model_keeps_its_own_kv_buffers_out_of_the_target_placement():
    parsed = parse_startup_log(log("lv4-mtp"))
    assert parsed["kv_buffers"] == [{"device": "CUDA0", "mib": 8192.0}, {"device": "CUDA0", "mib": 512.0}]
    assert parsed["kv_placement"]["gpu_mib"] == 8192.0 == parsed["kv_cache"]["size_mib"]


def test_a_split_kv_cache_is_the_requested_partial_offload_not_a_mismatch():
    """LIVE-010: `offload-48` must verify and go on to report tok/s, not fail at `verify`."""
    rows, by_name = live_offload("live-offload-2-partial-48", n_gpu_layers=48)
    kv = by_name["kv_offload"]
    assert kv.status == "verified-log" and kv.requested is True
    # REV-OFF-01: the placement is the VALUE, always one of the three states, with the MiB beside it in
    # `detail`. A consumer that evaluates `effective` for truth can no longer read a split as "all on the GPU".
    assert kv.effective == "split" and kv.effective in KV_PLACEMENTS
    assert kv.detail == "ram 84 MiB on CPU / gpu 252 MiB on CUDA0"
    assert kv_placement_reading(kv.effective, kv.detail) == ("split", "ram 84 MiB on CPU / gpu 252 MiB on CUDA0")
    assert "blocks left on the CPU keep their KV in system RAM" in kv.evidence
    assert by_name["n_gpu_layers"].status == "verified-log" and by_name["n_gpu_layers"].effective == "48/66"
    assert not [row.control for row in rows if row.status in {"mismatch", "unobserved"}]
    assert unverified_required(rows) == [] and settings_verified(rows) is True


def test_full_gpu_and_forced_ram_placements_read_the_same_as_before():
    rows, by_name = live_offload("live-offload-2-baseline")
    assert by_name["kv_offload"].status == "verified-log" and by_name["kv_offload"].effective == "gpu"
    assert by_name["kv_offload"].detail == "336 MiB on CUDA0" and settings_verified(rows) is True
    rows, by_name = live_offload("live-offload-2-kv-ram", kv_offload=False)
    assert by_name["kv_offload"].status == "verified-log" and by_name["kv_offload"].effective == "ram"
    assert by_name["kv_offload"].detail == "336 MiB on CPU"
    assert by_name["n_gpu_layers"].effective == "66/66" and settings_verified(rows) is True


SPLIT = "split"


@pytest.mark.parametrize("name, engine, startup, effective", [
    # `--no-kv-offload` was asked for, yet a CUDA0 KV buffer appeared: RAM-only is the whole of that request.
    ("live-offload-2-partial-48", {"n_gpu_layers": 48, "kv_offload": False}, {}, SPLIT),
    ("live-offload-2-baseline", {"kv_offload": False}, {}, "gpu"),
    # Every block was offloaded, yet part of the cache stayed in system RAM.
    ("live-offload-2-partial-48", {"n_gpu_layers": "all"}, {"offloaded_layers": 66}, SPLIT),
    ("live-offload-2-partial-48", {"n_gpu_layers": 66}, {"offloaded_layers": 66}, SPLIT),
    # A partial offload with no device KV buffer at all, long enough to have reached a KV-bearing block
    # (48 blocks against a cache that spans 16 of 66): the cache was pushed off the GPU regardless.
    ("live-offload-2-kv-ram", {"n_gpu_layers": 48}, {"offloaded_layers": 48}, "ram"),
    # Nothing was offloaded, yet a device KV buffer appeared: no block on the GPU can hold a cache there.
    ("live-offload-2-baseline", {"n_gpu_layers": 0}, {"offloaded_layers": 0}, "gpu"),
])
def test_a_kv_placement_that_contradicts_the_request_is_still_a_mismatch(name, engine, startup, effective):
    rows, by_name = live_offload(name, **engine, **startup)
    assert by_name["kv_offload"].status == "mismatch" and by_name["kv_offload"].effective == effective
    assert ("contradicts this request" in by_name["kv_offload"].evidence) is (effective == SPLIT)
    assert by_name["kv_offload"].detail  # a mismatch still names the MiB it saw, on whichever device
    assert by_name["n_gpu_layers"].status == "verified-log"  # the contradiction is the placement, not the count
    assert "kv_offload" in unverified_required(rows) and settings_verified(rows) is False


def test_a_short_partial_offload_that_reaches_no_kv_layer_is_not_accused():
    """REV-OFF-02: the mirror of the hybrid case the `gpu` branch already admitted.

    The target is hybrid -- `llama_kv_cache: ... 16 layers` out of 66 offloadable blocks, about one in four --
    so a few offloaded blocks can land entirely between two KV-bearing ones and print a CPU-only KV buffer.
    Scoring that a mismatch hard-fails the candidate at `verify` (`runner._settings_evidence`) and discards
    its measured tok/s: the exact failure LIVE-010 repaired, moved to the low end of the U17 sweep.
    """
    for layers in (1, 2, 3, 4):  # 66 blocks / 16 KV layers: an offload of four or fewer can miss them all
        rows, by_name = live_offload("live-offload-2-kv-ram", n_gpu_layers=layers, offloaded_layers=layers)
        kv = by_name["kv_offload"]
        assert kv.status == "verified-log" and kv.effective == "ram" and kv.requested is True
        assert kv.detail == "336 MiB on CPU"
        assert f"no KV-bearing block reached the GPU -- {layers} of 66 blocks" in kv.evidence
        assert "cannot tell the flag's two settings apart" in kv.evidence  # verified, but not discriminating
        assert by_name["n_gpu_layers"].status == "verified-log"
        assert unverified_required(rows) == [] and settings_verified(rows) is True
    # One block further and the offload must have reached a KV-bearing block, so an all-RAM cache is a
    # contradiction again: the repair opens the low end of the sweep, it does not disarm the check.
    _, by_name = live_offload("live-offload-2-kv-ram", n_gpu_layers=5, offloaded_layers=5)
    assert by_name["kv_offload"].status == "mismatch"
    assert "an offload of this size cannot account for" in by_name["kv_offload"].evidence
    # A log that never printed the cache's layer count decides nothing, so the old rule stands and the
    # observation is refused: the admission is earned by a counted layer span, never assumed.
    rows, by_name = live_offload("live-offload-2-kv-ram", n_gpu_layers=1, offloaded_layers=1,
                                 kv_cache={"k_type": "f16", "v_type": "f16"})
    assert by_name["cache_type_k"].status == "verified-log"  # only the layer count is missing
    assert by_name["kv_offload"].status == "mismatch" and settings_verified(rows) is False


def test_a_zero_layer_offload_verifies_the_only_placement_it_can_produce():
    """REV-OFF-05: `offload-0` is a sweep point, and it could never be eligible while this row said nothing.

    `consistent_kv_placements` calls `ram` the single honest outcome with nothing on the GPU; the row used to
    observe exactly that and decline to verify it, which left `kv_offload` -- a REQUIRED control -- unverified
    forever, so the extreme full-RAM point could run, be measured, and never reach the frontier. The two agree
    now: the placement is verified and the evidence says in words that it cannot discriminate the flag.
    """
    rows, by_name = live_offload("live-offload-2-kv-ram", n_gpu_layers=0, offloaded_layers=0)
    kv = by_name["kv_offload"]
    assert kv.status == "verified-log" and kv.effective == "ram" and kv.detail == "336 MiB on CPU"
    assert "nothing was offloaded, so the cache had nowhere but system RAM" in kv.evidence
    assert "cannot tell the flag's two settings apart" in kv.evidence
    assert by_name["n_gpu_layers"].effective == "0/66" and by_name["n_gpu_layers"].status == "verified-log"
    assert unverified_required(rows) == [] and settings_verified(rows) is True
    # `--no-kv-offload` at zero layers reads the same way: the request and the observation still agree.
    rows, by_name = live_offload("live-offload-2-kv-ram", n_gpu_layers=0, kv_offload=False, offloaded_layers=0)
    assert by_name["kv_offload"].status == "verified-log" and settings_verified(rows) is True


def test_fit_is_not_verified_unless_the_argv_asked_for_off():
    _, asked_on = live_baseline(argv_fit=("--fit", "on"))
    assert asked_on["fit"].status == "argv-accepted" and asked_on["fit"].effective is None
    _, absent = live_baseline(argv_fit=())
    assert absent["fit"].status == "unobserved" and "was not passed" in absent["fit"].evidence
    assert "fit" in unverified_required(list(asked_on.values())) and not settings_verified(list(absent.values()))
