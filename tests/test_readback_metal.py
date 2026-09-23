"""Metal readback: the real b11011 Metal startup log, `/props` and `/slots` from a native smoke run on an M1.

The fixtures are the smoke run's own files (Qwen3-1.7B-Q4_K_M, full offload, `llama-server` b11011 run directly
on the host) with only the machine-specific model directory replaced by `/Users/example/models`; the ephemeral
port stays as captured. `props-b11011-metal.json` and `slots-b11011-metal.json` are the raw response bodies the
smoke run saved -- unlike the CUDA fixtures there is no `{"status", "body"}` wrapper, because no status was
recorded and none is invented here.
"""

import hashlib
import json
from pathlib import Path

import pytest

from llmbench.backends.llamacpp import OVERFLOW_ERROR_TYPE, OVERFLOW_MARGIN_TOKENS
from llmbench.config import canonical_json
from llmbench.containers.config import ContainerRunConfig
from llmbench.containers.plan import build_server_argv
from llmbench.containers.readback import (NATIVE_PARSER_VERSION, PARSER_VERSION, REQUIRED_CONTROLS,
                                          build_settings_evidence, parse_memory_breakdown, parse_native_startup,
                                          parse_startup_log, settings_verified, unverified_required)

DATA = Path(__file__).parent / "data"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.json"
METAL_LOG = DATA / "startup-b11011-metal-qwen3-1.7b.log"
MODEL_PATH = "/Users/example/models/Qwen3-1.7B-Q4_K_M.gguf"
PORT = 50363  # the smoke run's ephemeral loopback port, as captured
BUILD = "b11011-aa39d7a3e"
# The smoke run recorded no overflow probe. This is `LlamaCppBackend.probe_overflow_rejection`'s passing result
# for the 4864-token slot, key for key; `build_settings_evidence` reads only its `passed`.
PROBE = {"passed": True, "sent_tokens": 4864 + OVERFLOW_MARGIN_TOKENS, "expected_n_ctx": 4864,
         "expected_error_type": OVERFLOW_ERROR_TYPE, "status": 400, "error_type": OVERFLOW_ERROR_TYPE,
         "reported_n_ctx": 4864, "reported_n_prompt_tokens": 4864 + OVERFLOW_MARGIN_TOKENS, "detail": None}


def metal_text() -> str:
    return METAL_LOG.read_text(encoding="utf-8")


def cuda_text(name: str) -> str:
    return (DATA / f"startup-b11011-{name}.log").read_text(encoding="utf-8")


def metal_readback() -> dict:
    return {"props": json.loads((DATA / "props-b11011-metal.json").read_text(encoding="utf-8")),
            "slots": json.loads((DATA / "slots-b11011-metal.json").read_text(encoding="utf-8")), "models": {}}


def metal_config(**engine) -> ContainerRunConfig:
    """The example candidate moved onto the native runtime with the smoke run's engine settings.

    The smoke server ran `--ctx-size 4864` (4096 input + 256 template reserve + 512 output) with reasoning off
    (its log says `thinking = 0`); every other engine value is the example's and matches the log line for line.
    """
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.pop("inference_image")
    raw.update(runtime="metal-native", native_limits={},
               native_server={"executable": "/Users/example/llama-b11011/llama-server",
                              "executable_sha256": "1" * 64, "libraries_sha256": "2" * 64,
                              "build_info": BUILD, "help_sha256": "3" * 64,
                              "source": "llama.cpp b11011 macOS arm64"})
    raw["assets"][0].update(host_path=MODEL_PATH, model_name="Qwen3-1.7B")
    raw["engine"].update({"ctx_size": 4864, "reasoning": "off", **engine})
    return ContainerRunConfig.model_validate_json(json.dumps(raw))


def native_argv(config: ContainerRunConfig) -> tuple[str, ...]:
    return build_server_argv(config, model_path=config.server_model_path, host="127.0.0.1", port=PORT)


def metal_evidence(config, *, overflow=PROBE, startup=None):
    readback = metal_readback()
    rows = build_settings_evidence(config.engine, native_argv(config), readback,
                                   parse_startup_log(metal_text()) if startup is None else startup, overflow,
                                   alias=readback["props"]["model_alias"], build_info=config.build_info)
    return rows, {row.control: row for row in rows}


# ---- fixtures ----------------------------------------------------------------------------------------------

def test_metal_fixtures_are_the_smoke_capture_with_only_the_model_directory_replaced():
    for path in DATA.glob("*metal*"):
        text = path.read_text(encoding="utf-8")
        assert "josephmoulton" not in text and "Codex" not in text, path.name
    log, readback = metal_text(), metal_readback()
    assert f"load_model: loading model '{MODEL_PATH}'" in log
    assert readback["props"]["model_path"] == MODEL_PATH and readback["props"]["build_info"] == BUILD
    assert readback["props"]["default_generation_settings"]["n_ctx"] == 4864
    assert readback["props"]["total_slots"] == 1 and readback["slots"] == [
        {"id": 0, "n_ctx": 4864, "speculative": False, "is_processing": False}]


# ---- parse_startup_log: the shared settings parser reads Metal unchanged ---------------------------------

def test_parse_startup_log_reads_the_metal_log_from_the_same_lines_as_cuda():
    parsed = parse_startup_log(metal_text())
    assert parsed["parser_version"] == PARSER_VERSION and parsed["build"] == BUILD and parsed["verbosity"] == 4
    assert (parsed["offloaded_layers"], parsed["total_layers"]) == (29, 29)
    assert parsed["flash_attn"] == "enabled" and parsed["load_mode"] == "none" and parsed["kv_unified"] == "false"
    assert (parsed["n_ctx"], parsed["n_ctx_seq"], parsed["n_batch"], parsed["n_ubatch"]) == (4864, 4864, 2048, 512)
    assert (parsed["n_threads"], parsed["n_threads_batch"], parsed["n_seq_max"]) == (8, 8, 1)
    assert parsed["kv_cache"] == {"size_mib": 532.0, "cells": 4864, "layers": 28, "k_type": "f16", "k_mib": 266.0,
                                  "v_type": "f16", "v_mib": 266.0}
    assert parsed["draft_kv_caches"] == [] and parsed["kv_buffers"] == [{"device": "MTL0", "mib": 532.0}]
    # MTL0 is a device under the unchanged `_host_buffer` rule: the whole cache sits in Metal's working set.
    assert parsed["kv_placement"] == {"placement": "gpu", "ram_mib": 0.0, "gpu_mib": 532.0, "ram_devices": [],
                                      "gpu_devices": ["MTL0"]}
    assert parsed["model_buffers"] == [{"device": "CPU", "mib": 243.43}, {"device": "MTL0", "mib": 1050.43}]
    assert (parsed["n_slots"], parsed["n_ctx_slot"], parsed["thinking"]) == (1, 4864, 0)
    assert parsed["speculative"] == {"implementations": []} and parsed["prompt_cache_mib"] == 0
    assert parsed["warmup"] is True and parsed["ui_disabled"] is True
    assert parsed["listening"] == f"http://127.0.0.1:{PORT}"
    assert parsed["model_loaded_seconds"] == pytest.approx(2.556011)
    # Metal's system_info has no FA_QUANTS entry, and no recurrent state exists in a dense model.
    assert "fa_quants" not in parsed and "recurrent_mib" not in parsed


# Digests of `parse_startup_log` over every CUDA fixture, taken before the native parsers were added. The
# container runtime's evidence must stay byte-identical: a new key or a changed value here changes a sealed
# settings-evidence.json and every report built from it.
CUDA_GOLDEN = {
    "live-full-1-baseline": "4a04ad461ebaac285f0368a3cae5a73610972bca004938c484dc8e69236ce389",
    "live-offload-2-baseline": "cb7fc19b859eac8314b95f3953f2a674039207dbfdc1b05edcdc1908f81d625b",
    "live-offload-2-kv-ram": "1011397b59faf5efa29cf12c614853060a3831f46fc953008f726ce0b0975231",
    "live-offload-2-partial-48": "bdb9d18244102312c46f62c95254f9cd6469b1e1801c2c4b7f516846dadf3f8c",
    "lv3": "547be7a6313cabe486ce50e5200baa576756e115897721c866983faf9595bbad",
    "lv4-compose": "94e1b3c190d3d58e412ea4b4e3e1ea3646cd1840359bb55d95394138cfe71fd3",
    "lv4-mtp": "25e27db7d027ffad4c58a378ec8e7a0983090a04c63da5c97fbba0c44a793c09",
    "lv4": "2a7486136b3c883f3f13a12ee7c8d2b3a30dc19e8d4d34f732207a6e732523d3",
}


@pytest.mark.parametrize("name", sorted(CUDA_GOLDEN))
def test_cuda_startup_parsing_is_byte_identical(name):
    parsed = parse_startup_log(cuda_text(name))
    assert hashlib.sha256(canonical_json(parsed).encode("utf-8")).hexdigest() == CUDA_GOLDEN[name]
    assert not {"devices", "metal_budget_mib", "metal_resident_mib", "compute_buffers"} & set(parsed)


# ---- parse_native_startup ----------------------------------------------------------------------------------

def test_native_startup_facts_from_the_metal_log():
    native = parse_native_startup(metal_text())
    assert native["parser_version"] == NATIVE_PARSER_VERSION
    assert native["devices"] == [
        {"name": "MTL0", "description": "Apple M1", "total_mib": 5461, "free_mib": 5460},
        {"name": "BLAS", "description": "Accelerate", "total_mib": 0, "free_mib": 0},
        {"name": "CPU", "description": "Apple M1", "total_mib": 8192, "free_mib": 8192}]
    assert native["model_device"] == {"name": "MTL0", "description": "Apple M1", "device_id": "unknown id",
                                      "free_mib": 5460}
    assert native["model_devices"] == [native["model_device"]]
    assert native["system_info_backends"] == ["MTL", "CPU"] and native["metal_embed_library"] is True
    assert native["system_info_features"]["MTL"] == {"EMBED_LIBRARY": "1"}
    assert "FA_QUANTS" not in native["system_info_features"]["CPU"]
    assert native["metal_device"] == "Apple M1" and native["metal_budget_mib"] == 5461
    assert native["model_buffers"] == [{"device": "CPU", "mib": 243.43}, {"device": "MTL0", "mib": 1050.43}]
    assert native["kv_buffers"] == [{"device": "MTL0", "mib": 532.0}] and "recurrent_buffers" not in native
    assert native["compute_buffers"] == [{"device": "MTL0", "mib": 48.76, "context": 0},
                                         {"device": "CPU", "mib": 12.76, "context": 0}]
    assert native["resident_by_device"] == {"CPU": 256.19, "MTL0": 1631.19}
    assert native["metal_resident_mib"] == 1631.19 and native["host_resident_mib"] == 256.19
    assert (native["offloaded_layers"], native["total_layers"]) == (29, 29)
    json.dumps(native, allow_nan=False)  # result.json `memory` must accept it as-is


@pytest.mark.parametrize("name, metal_self, host_self", [
    ("qwen3-1.7b", 1631, 256),  # dense: KV only
    ("qwen3.5-2b", 1340, 410),  # hybrid: 6 KV layers plus a 19.27 MiB MTL0 recurrent-state buffer
])
def test_native_residency_reconciles_with_llama_cpp_own_memory_breakdown(name, metal_self, host_self):
    """The startup buffer lines and the exit table are two independent prints of the same allocations.

    llama.cpp truncates each table cell to whole MiB, so the parsed sums must truncate to exactly the table's
    numbers: if a buffer were dropped, double-counted or put on the wrong side, this would not hold. The table's
    `context` column is the KV cache plus the recurrent state, which is why both belong in the resident sum.
    """
    text = (DATA / f"startup-b11011-metal-{name}.log").read_text(encoding="utf-8")
    native, table = parse_native_startup(text), parse_memory_breakdown(text)
    by_device = {row["device"]: row for row in table}
    metal, host = by_device["MTL0"], by_device["Host"]

    def total(keys, device):
        return int(sum(row["mib"] for key in keys for row in native.get(key, ()) if row["device"] == device))

    assert int(native["metal_resident_mib"]) == metal["self"] == metal_self
    assert total(("model_buffers",), "MTL0") == metal["model"]
    assert total(("kv_buffers", "recurrent_buffers"), "MTL0") == metal["context"]
    assert total(("compute_buffers",), "MTL0") == metal["compute"]
    assert int(native["host_resident_mib"]) == host["self"] == host_self
    assert total(("model_buffers",), "CPU") == host["model"]
    assert total(("kv_buffers", "recurrent_buffers"), "CPU") == host["context"] == 0
    assert total(("compute_buffers",), "CPU") == host["compute"]
    assert metal["total"] == native["metal_budget_mib"] == 5461  # the budget the watchdog falls back to
    assert set(native["resident_by_device"]) == {"MTL0", "CPU"}  # BLAS is listed but holds nothing


def test_a_hybrid_model_on_metal_keeps_its_recurrent_state_in_the_working_set():
    text = (DATA / "startup-b11011-metal-qwen3.5-2b.log").read_text(encoding="utf-8")
    native, parsed = parse_native_startup(text), parse_startup_log(text)
    assert native["recurrent_buffers"] == [{"device": "MTL0", "mib": 19.27}]
    assert native["metal_resident_mib"] == round(1211.05 + 57.0 + 19.27 + 52.84, 3)
    assert parsed["recurrent_mib"] == 19.27 and parsed["kv_cache"]["layers"] == 6
    assert (parsed["offloaded_layers"], parsed["total_layers"]) == (25, 25)
    assert parsed["kv_placement"]["placement"] == "gpu" and parsed["kv_placement"]["gpu_devices"] == ["MTL0"]


def test_memory_breakdown_rows_from_the_metal_log():
    assert parse_memory_breakdown(metal_text()) == [
        {"device": "MTL0", "description": "Apple M1", "total": 5461, "free": 3829, "self": 1631, "model": 1050,
         "context": 532, "compute": 48, "unaccounted": 0, "table": 0},
        {"device": "Host", "description": None, "total": None, "free": None, "self": 256, "model": 243,
         "context": 0, "compute": 12, "unaccounted": None, "table": 0}]


def test_memory_breakdown_other_shapes_and_absence():
    """The row shapes llama.cpp prints besides the Metal fixture's, in its own spacing: a device row whose
    parts exceed its total (negative unaccounted), a host buffer type other than `Host`, a row in neither shape,
    and a second table."""
    prefix = "0.03.023.146 I common_memory_breakdown_print: "
    header = prefix + "| memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |"
    text = "\n".join(prefix + row if row.startswith("|  ") else row for row in [
        header,
        "|   - CUDA0 (RTX 5090)     | 32579 =  100 + (32600 = 30000 +    2000 +     600) +    -121 |",
        "|   - CPU_REPACK           |                   50 =    50 +       0 +       0                |",
        "|   - MTL0 (Apple M1)    |  5461 = broken |",
        header,
        "|   - Host               |                  256 =   243 +       0 +      12                |",
    ])
    rows = parse_memory_breakdown(text)
    assert [(row["device"], row["table"]) for row in rows] == [("CUDA0", 0), ("CPU_REPACK", 0), ("Host", 1)]
    assert rows[0]["unaccounted"] == -121 and rows[0]["description"] == "RTX 5090"
    assert rows[1] == {"device": "CPU_REPACK", "description": None, "total": None, "free": None, "self": 50,
                       "model": 50, "context": 0, "compute": 0, "unaccounted": None, "table": 0}
    # The startup logs of the container campaign end before shutdown: no table is no rows, not zero rows' worth.
    for name in CUDA_GOLDEN:
        assert parse_memory_breakdown(cuda_text(name)) == []
    assert parse_memory_breakdown("") == []


def test_native_parser_on_cuda_logs_claims_nothing_about_metal():
    native = parse_native_startup(cuda_text("lv4-mtp"))
    assert [item["name"] for item in native["devices"]] == ["CUDA0", "CPU"]
    assert native["model_device"]["device_id"] == "0000:01:00.0"
    assert native["system_info_backends"] == ["CUDA", "CPU"]
    assert native["system_info_features"]["CUDA"]["FA_QUANTS"] == "q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16"
    for key in ("metal_embed_library", "metal_device", "metal_budget_mib", "metal_resident_mib"):
        assert key not in native, key
    # The draft context is reserved twice (graph nodes 50, then 56); only its latest reservation is resident.
    assert native["compute_buffers"] == [
        {"device": "CUDA0", "mib": 250.02, "context": 0}, {"device": "CUDA_Host", "mib": 148.02, "context": 0},
        {"device": "CUDA0", "mib": 196.02, "context": 1}, {"device": "CUDA_Host", "mib": 148.02, "context": 1}]
    assert native["kv_buffers"] == [{"device": "CUDA0", "mib": 8192.0}, {"device": "CUDA0", "mib": 512.0}]
    assert native["resident_by_device"] == {"CUDA0": 25087.98, "CUDA_Host": 978.07}
    assert native["host_resident_mib"] == 978.07  # CUDA's pinned host pool is host memory, as `_host_buffer` says
    partial = parse_native_startup(cuda_text("live-offload-2-partial-48"))
    assert partial["recurrent_buffers"] == [{"device": "CPU", "mib": 43.64}, {"device": "CUDA0", "mib": 105.98}]
    assert partial["resident_by_device"]["CPU"] == 84.0 + 43.64  # KV + RS of the blocks left on the CPU
    assert (partial["offloaded_layers"], partial["total_layers"]) == (48, 66)
    assert parse_native_startup(cuda_text("lv3")) == {"parser_version": NATIVE_PARSER_VERSION}


def test_an_incomplete_metal_log_claims_no_residency():
    text = metal_text()
    cut = parse_native_startup(text[:text.index("sched_reserve: reserving")])
    assert cut["model_buffers"] and cut["kv_buffers"] and "compute_buffers" not in cut
    for key in ("resident_by_device", "metal_resident_mib", "host_resident_mib"):
        assert key not in cut, key  # a partial sum would read as the whole allocation
    assert cut["metal_budget_mib"] == 5461 and cut["metal_device"] == "Apple M1"
    assert parse_native_startup("") == {"parser_version": NATIVE_PARSER_VERSION}


def test_residency_needs_every_constructed_context_reserved():
    """A draft context allocates its KV cache before its compute buffers: a log cut between the two holds a
    buffer set that looks complete for the target alone. Every constructed context must have reserved."""
    text = cuda_text("lv4-mtp")
    full, draft = parse_native_startup(text), text.index("creating MTP draft context")
    first, second = (text.index("sched_reserve: reserving", draft),
                     text.index("sched_reserve: reserving", text.index("adding speculative implementation")))
    # Cut just after the draft context is constructed, and again after its KV cache but before it reserves.
    for cut in (text.index("n_seq_max", text.index("llama_context: constructing", draft)), first):
        partial = parse_native_startup(text[:cut])
        assert [row["context"] for row in partial["compute_buffers"]] == [0, 0]
        for key in ("resident_by_device", "host_resident_mib"):
            assert key not in partial, (cut, key)
    # Cut between the draft's two reservations: every context has reserved, and the re-reservation printed
    # the same sizes, so the partial log already holds the whole residency.
    early = parse_native_startup(text[:second])
    assert early["resident_by_device"] == full["resident_by_device"]
    assert early["host_resident_mib"] == full["host_resident_mib"]
    placement = ("llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 5090) (0000:01:00.0) - "
                 "30927 MiB free")
    other = placement.replace("CUDA0", "CUDA1").replace("0000:01:00.0", "0000:02:00.0")
    two = parse_native_startup(text.replace(placement, other + "\n0.00.625.775 I " + placement, 1))
    assert "model_device" not in two and [item["name"] for item in two["model_devices"]] == ["CUDA1", "CUDA0"]


def test_mapped_metal_weights_stay_in_the_metal_working_set():
    """`--load-mode mmap` prints the offloaded weights as `MTL0_Mapped` (b11011 names the mapped buffer type
    `MTL<n>_Mapped` and reports it as not host): still Metal residency, never host memory."""
    text = metal_text().replace("MTL0 model buffer size", "MTL0_Mapped model buffer size")
    native, parsed = parse_native_startup(text), parse_startup_log(text)
    assert native["resident_by_device"] == {"CPU": 256.19, "MTL0_Mapped": 1050.43, "MTL0": 580.76}
    assert native["metal_resident_mib"] == 1631.19 and native["host_resident_mib"] == 256.19
    assert parsed["model_buffers"][1] == {"device": "MTL0_Mapped", "mib": 1050.43}
    assert parsed["kv_placement"]["placement"] == "gpu"


def test_metal_budget_and_embed_flag_are_never_guessed():
    text = metal_text()
    mtl0 = "common_param:   - MTL0    : Apple M1 (5461 MiB, 5460 MiB free)"
    two = parse_native_startup(text.replace(mtl0, mtl0 + "\n0.00.179.094 I cmn  common_param:   - MTL1    : "
                                                         "Apple M1 (4096 MiB, 4096 MiB free)"))
    assert [item["name"] for item in two["devices"]] == ["MTL0", "MTL1", "BLAS", "CPU"]
    assert "metal_budget_mib" not in two  # two Metal devices: which budget applies is not the log's to say
    appended = parse_native_startup(text + "0.09.000.000 I cmn  common_param:   - MTL1    : Apple M2 "
                                           "(9999 MiB, 9999 MiB free)\n")
    assert len(appended["devices"]) == 3 and appended["metal_budget_mib"] == 5461  # first device_info block only
    featureless = parse_native_startup(text.replace("| MTL : EMBED_LIBRARY = 1 | CPU :", "| MTL : CPU :"))
    assert featureless["system_info_backends"] == ["MTL", "CPU"] and featureless["metal_embed_library"] is False
    assert featureless["system_info_features"]["CPU"]["NEON"] == "1"
    disabled = parse_native_startup(text.replace("EMBED_LIBRARY = 1", "EMBED_LIBRARY = 0"))
    assert disabled["metal_embed_library"] is False and disabled["system_info_features"]["MTL"] == {
        "EMBED_LIBRARY": "0"}
    no_info = parse_native_startup("\n".join(line for line in text.splitlines() if "system_info:" not in line))
    assert "metal_embed_library" not in no_info and "system_info_backends" not in no_info


# ---- settings evidence on Metal ----------------------------------------------------------------------------

def test_native_argv_is_the_container_argv_with_only_the_endpoint_changed():
    config = metal_config()
    argv, container = native_argv(config), build_server_argv(config)
    changed = {index for index, (left, right) in enumerate(zip(argv, container)) if left != right}
    assert len(argv) == len(container) and {argv[index - 1] for index in changed} == {"--model", "--host", "--port"}
    assert argv[argv.index("--model") + 1] == metal_readback()["props"]["model_path"] == config.server_model_path
    assert parse_startup_log(metal_text())["listening"] == "http://{}:{}".format(
        argv[argv.index("--host") + 1], argv[argv.index("--port") + 1])


def test_metal_run_verifies_every_required_control_from_the_same_lines():
    config = metal_config()
    rows, by_name = metal_evidence(config)
    assert not [row.control for row in rows if row.status == "mismatch"]
    # FA_QUANTS is a CUDA system_info entry; on Metal the kernel row is unobserved and, as before, not required.
    assert [row.control for row in rows if row.status == "unobserved"] == ["flash_attn_kernel"]
    kernel = by_name["flash_attn_kernel"]
    assert kernel.required is False
    assert kernel.evidence == f"startup log ({PARSER_VERSION}): FA_QUANTS: not reported"
    for control in ("ctx_size", "parallel", "alias", "build_info"):
        assert by_name[control].status == "verified-api", control
    for control in ("n_gpu_layers", "cache_type_k", "cache_type_v", "kv_offload", "flash_attn", "load_mode",
                    "batch_size", "ubatch_size", "threads", "threads_batch", "cache_ram_mib", "spec_type",
                    "log_verbosity", "warmup", "reasoning"):
        assert by_name[control].status == "verified-log", control
    assert by_name["n_gpu_layers"].effective == "29/29" and by_name["flash_attn"].effective == "on"
    assert by_name["kv_offload"].effective == "gpu" and by_name["kv_offload"].detail == "532 MiB on MTL0"
    assert by_name["cache_type_k"].effective == by_name["cache_type_v"].effective == "f16"
    assert by_name["reasoning"].effective == "off" and by_name["cache_ram_mib"].effective == 0
    assert by_name["build_info"].effective == BUILD == config.build_info
    assert by_name["fit"].status == "verified-behavior"
    assert "n_gpu_layers=29/29 (verified-log)" in by_name["fit"].evidence
    assert by_name["context_shift"].status == "verified-behavior"
    assert {row.control for row in rows if row.required} == set(REQUIRED_CONTROLS) - {"spec_draft_n_max"}
    assert unverified_required(rows) == [] and settings_verified(rows) is True


def test_without_the_overflow_probe_only_context_shift_stays_unverified():
    rows, by_name = metal_evidence(metal_config(), overflow=None)
    assert not [row.control for row in rows if row.status == "mismatch"]
    assert by_name["context_shift"].status == "argv-accepted" and by_name["context_shift"].required
    assert by_name["fit"].status == "verified-behavior"  # fit is witnessed by readback, not by the probe
    assert unverified_required(rows) == ["context_shift"] and settings_verified(rows) is False


@pytest.mark.parametrize("engine, control, fragment", [
    ({"n_gpu_layers": 20}, "n_gpu_layers", "offloaded layers"),
    # MTL0 is a device buffer: a Metal KV cache under --no-kv-offload is the same contradiction as on CUDA0.
    ({"kv_offload": False}, "kv_offload", "a device KV buffer contradicts --no-kv-offload"),
    ({"threads": 4}, "threads", PARSER_VERSION),
    ({"batch_size": 1024}, "batch_size", PARSER_VERSION),
])
def test_a_metal_log_that_contradicts_the_request_is_a_mismatch(engine, control, fragment):
    rows, by_name = metal_evidence(metal_config(**engine))
    assert by_name[control].status == "mismatch" and fragment in by_name[control].evidence
    assert settings_verified(rows) is False


def test_a_different_pinned_build_or_context_is_a_mismatch():
    config = metal_config()
    other = config.model_copy(update={"native_server": config.native_server.model_copy(
        update={"build_info": "b11010-aa39d7a3e"})})
    assert metal_evidence(other)[1]["build_info"].status == "mismatch"
    startup = {**parse_startup_log(metal_text()), "n_ctx": 8192}  # log and /props disagree about the context
    _, by_name = metal_evidence(config, startup=startup)
    assert by_name["ctx_size"].status == "mismatch" and by_name["fit"].status == "argv-accepted"
