"""Effective-setting evidence from the llama-server startup log (build b11011 wording) and API readback.

A missing line is never treated as a default. Argv acceptance alone stays `argv-accepted`.

`parse_startup_log` reads the settings every runtime shares: a Metal build prints the same `llama_context`,
`load_tensors`, `llama_kv_cache` and `system_info` lines as the CUDA image, so one parser verifies both.
`parse_native_startup` and `parse_memory_breakdown` add what only matters on a unified-memory host -- which
devices the build saw, how much Metal may hold, and how much it does hold -- without changing a byte of what
`parse_startup_log` returns for a container run.
"""

from __future__ import annotations

import re
from bisect import bisect_right

from .config import LlamaCppSettings, SettingEvidence

PARSER_VERSION = "llama-server-startup-v1"
NATIVE_PARSER_VERSION = "llama-server-native-startup-v1"
VERIFIED = frozenset({"verified-api", "verified-log", "verified-behavior"})
# Memory, scheduling, context and speculation controls that change what is measured. `fit` has no b11011
# log line and no /props field; it is verified from behaviour instead, see `_fit`.
REQUIRED_CONTROLS = ("ctx_size", "parallel", "n_gpu_layers", "cache_type_k", "cache_type_v", "kv_offload",
                     "flash_attn", "load_mode", "batch_size", "ubatch_size", "threads", "threads_batch",
                     "cache_ram_mib", "context_shift", "fit", "spec_type", "spec_draft_n_max")
# What b11011's `--fit` can move: "whether to adjust unset arguments to fit in device memory" with `--fit-ctx`
# as the floor of the context it may set (`artifacts/container-prep/llama-server-help.txt`). `plan.py` passes
# every one of them explicitly and each is read back individually, so an adjustment surfaces as a mismatch.
FIT_ADJUSTABLE = ("ctx_size", "n_gpu_layers", "batch_size", "ubatch_size")

_MIB = r"(\d+(?:\.\d+)?) MiB"
_FIRST = {  # key -> (pattern, converters); the first match is the target context, later ones are drafts.
    "build": (r"build (\d+) \(([0-9a-f]+)\)", None),
    "verbosity": (r"verbosity = (\d+)", (int,)),
    "threads": (r"system_info: n_threads = (\d+) \(n_threads_batch = (\d+)\)", None),
    "fa_quants": (r"FA_QUANTS = ([a-z0-9_,-]+)", None),
    "load_mode": (r"load_tensors: .*\(load_mode = ([a-z+]+)\)", (str,)),
    "offloaded": (r"load_tensors: offloaded (\d+)/(\d+) layers to GPU", None),
    "n_seq_max": (r"llama_context: n_seq_max\s+= (\d+)", (int,)),
    "n_ctx": (r"llama_context: n_ctx\s+= (\d+)", (int,)),
    "n_ctx_seq": (r"llama_context: n_ctx_seq\s+= (\d+)", (int,)),
    "n_batch": (r"llama_context: n_batch\s+= (\d+)", (int,)),
    "n_ubatch": (r"llama_context: n_ubatch\s+= (\d+)", (int,)),
    "flash_attn": (r"llama_context: flash_attn\s+= (\w+)", (str,)),
    "kv_unified": (r"llama_context: kv_unified\s+= (\w+)", (str,)),
    "slots": (r"initializing, n_slots = (\d+), n_ctx_slot = (\d+)", None),
    "thinking": (r"chat template, thinking = (\d+)", (int,)),
    "listening": (r"llama_server: listening on (\S+)", (str,)),
}
_KV_CACHE = re.compile(r"llama_kv_cache: size =\s*" + _MIB + r" \(\s*(\d+) cells,\s*(\d+) layers,[^)]*\), "
                       r"K \((\w+)\):\s*" + _MIB + r", V \((\w+)\):\s*" + _MIB)
_KV_BUFFER = re.compile(r"llama_kv_cache:\s+(\S+) KV buffer size =\s*" + _MIB)
_MODEL_BUFFER = re.compile(r"load_tensors:\s+(\S+) model buffer size =\s*" + _MIB)
_RS_BUFFER = re.compile(r"llama_memory_recurrent:\s+(\S+) RS buffer size =\s*" + _MIB)
_COMPUTE_BUFFER = re.compile(r"sched_reserve:\s+(\S+) compute buffer size =\s*" + _MIB)
_CONTEXT = re.compile(r"llama_context: constructing llama_context")
_STAMP = r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) "
# `common_param:   - MTL0    : Apple M1 (5461 MiB, 5460 MiB free)`, one line per device under `device_info:`.
_DEVICE_LINE = re.compile(r"common_param:\s+- (\S+)\s+: (.*) \((\d+) MiB, (\d+) MiB free\)\s*$")
# `llama_prepare_model_devices: using device MTL0 (Apple M1) (unknown id) - 5460 MiB free`; the CUDA image
# prints its PCI bus id where Metal prints `unknown id`.
_MODEL_DEVICE = re.compile(r"llama_prepare_model_devices: using device (\S+) \((.*)\) \(([^()]*)\) - "
                           r"(\d+) MiB free")
# `common_memory_breakdown_print` at exit. A device row is `total = free + (self = model + context + compute) +
# unaccounted`; the Host row and other host buffer types have no total/free/unaccounted. Spaces only, never
# `\s`, so a pattern cannot run across into the next log line.
_BREAKDOWN = "common_memory_breakdown_print: "
_BREAKDOWN_HEADER = re.compile(_BREAKDOWN + r"\| memory breakdown \[MiB\]")
_BREAKDOWN_DEVICE = re.compile(_BREAKDOWN + r"\| *- (.+?) *\| *(\d+) *= *(\d+) *\+ *"
                               r"\( *(\d+) *= *(\d+) *\+ *(\d+) *\+ *(\d+) *\) *\+ *(-?\d+) *\|")
_BREAKDOWN_HOST = re.compile(_BREAKDOWN + r"\| *- (.+?) *\| *(\d+) *= *(\d+) *\+ *(\d+) *\+ *(\d+) *\|")
_DEVICE_COLUMNS = ("total", "free", "self", "model", "context", "compute", "unaccounted")
_HOST_COLUMNS = ("self", "model", "context", "compute")


def _host_buffer(device: str) -> bool:
    """Whether a b11011 buffer device is host RAM rather than an accelerator.

    The host allocators print as `CPU` and `CPU_Mapped`; a backend's pinned host pool prints as `CUDA_Host`
    (the same live log shows `CUDA_Host model buffer size` for weights that stayed in system RAM). Everything
    else -- `CUDA0`, `ROCm0`, `Vulkan0`, `SYCL0`, and on Metal `MTL0` (`MTL0_Mapped` for mapped weights) -- is
    a device. On Apple Silicon both sides are the same physical memory: "device" there means a buffer in
    Metal's working set, never separate VRAM (`tests/data/startup-b11011-metal-qwen3-1.7b.log`).
    """
    return device.startswith("CPU") or device.endswith("_Host")


def _metal_buffer(device: str) -> bool:
    """Whether a device name is Metal's: b11011 registers the Apple GPU as `MTL0` (live `--list-devices`)."""
    return device.startswith("MTL")


def _kv_placement(buffers: list[dict]) -> dict:
    """Three-state residency of one KV cache: "gpu", "ram" or "split", with the MiB on each side.

    A partial layer offload splits the cache -- llama.cpp prints one buffer line per device, so 48 of 66
    blocks on the GPU prints an 84 MiB `CPU` line and a 252 MiB `CUDA0` line. Sizes decide the state so that a
    device named with a 0.00 MiB allocation is not read as residency; if every line is zero, the device names
    decide instead.
    """
    host = [item for item in buffers if _host_buffer(item["device"])]
    accelerator = [item for item in buffers if not _host_buffer(item["device"])]
    ram_mib = round(float(sum(item["mib"] for item in host)), 3)
    gpu_mib = round(float(sum(item["mib"] for item in accelerator)), 3)
    on_ram, on_gpu = ram_mib > 0, gpu_mib > 0
    if not on_ram and not on_gpu:
        on_ram, on_gpu = bool(host), bool(accelerator)
    return {"placement": "split" if on_ram and on_gpu else "ram" if on_ram else "gpu",
            "ram_mib": ram_mib, "gpu_mib": gpu_mib,
            "ram_devices": [item["device"] for item in host],
            "gpu_devices": [item["device"] for item in accelerator]}


def parse_startup_log(text: str) -> dict:
    found: dict = {"parser_version": PARSER_VERSION}
    for key, (pattern, convert) in _FIRST.items():
        match = re.search(pattern, text)
        if match and convert:
            found[key] = convert[0](match[1])
        elif match:
            found[key] = match.groups()
    if "build" in found:
        found["build"] = "b{}-{}".format(*found["build"])
    if "threads" in found:
        found["n_threads"], found["n_threads_batch"] = (int(item) for item in found.pop("threads"))
    if "fa_quants" in found:
        found["fa_quants"] = [item.split("-") for item in found["fa_quants"][0].strip(",").split(",")]
    if "offloaded" in found:
        found["offloaded_layers"], found["total_layers"] = (int(item) for item in found.pop("offloaded"))
    if "slots" in found:
        found["n_slots"], found["n_ctx_slot"] = (int(item) for item in found.pop("slots"))
    caches = [{"size_mib": float(m[1]), "cells": int(m[2]), "layers": int(m[3]), "k_type": m[4],
               "k_mib": float(m[5]), "v_type": m[6], "v_mib": float(m[7])} for m in _KV_CACHE.finditer(text)]
    if caches:
        found["kv_cache"], found["draft_kv_caches"] = caches[0], caches[1:]
    buffers = list(_KV_BUFFER.finditer(text))
    if buffers:
        found["kv_buffers"] = [{"device": match[1], "mib": float(match[2])} for match in buffers]
        # Buffer lines precede the `size =` line of the cache they belong to, so the target model's buffers are
        # the ones before the first `size =` line; a draft model's buffers follow it and are its own placement.
        first_size = re.search(r"llama_kv_cache: size =", text)
        end = first_size.start() if first_size else len(text)
        target = [row for row, match in zip(found["kv_buffers"], buffers) if match.start() < end]
        found["kv_placement"] = _kv_placement(target or found["kv_buffers"])
    models = _MODEL_BUFFER.findall(text)
    if models:
        found["model_buffers"] = [{"device": device, "mib": float(size)} for device, size in models]
    recurrent = re.search(r"llama_memory_recurrent: size =\s*" + _MIB, text)
    if recurrent:
        found["recurrent_mib"] = float(recurrent[1])
    implementations = re.findall(r"adding speculative implementation '([^']+)'", text)
    if implementations or "no implementations specified for speculative decoding" in text:
        found["speculative"] = {"implementations": implementations}
        limits = re.search(r"common_specu: - n_max=(\d+), n_min=(\d+)", text)
        if implementations and limits:
            found["speculative"].update(n_max=int(limits[1]), n_min=int(limits[2]))
    cache = re.search(r"prompt cache is enabled, size limit: (\d+) MiB", text)
    if cache or "prompt cache is disabled" in text:
        found["prompt_cache_mib"] = int(cache[1]) if cache else 0
    if "warming up the model with an empty run" in text:
        found["warmup"] = True
    if "The UI is disabled" in text:
        found["ui_disabled"] = True
    loaded = re.search(_STAMP + r".*llama_server: model loaded", text, re.MULTILINE)
    if loaded:
        found["model_loaded_seconds"] = (int(loaded[1]) * 60 + int(loaded[2]) + int(loaded[3]) / 1e3
                                         + int(loaded[4]) / 1e6)
    return found


def _device_info(text: str) -> list[dict]:
    """The devices the build enumerated, from the `device_info:` block that opens every b11011 log.

    Only the lines directly under the FIRST `device_info:` header are read, so a device line quoted anywhere
    else (a second server's log appended to the same file, say) is never taken for this server's hardware.
    """
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if "common_param: device_info:" in line), None)
    devices: list[dict] = []
    for line in lines[start + 1:] if start is not None else ():
        match = _DEVICE_LINE.search(line)
        if not match:
            break
        devices.append({"name": match[1], "description": match[2], "total_mib": int(match[3]),
                        "free_mib": int(match[4])})
    return devices


def _system_info(rest: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Backends and their feature flags from the `| NAME : KEY = VALUE | KEY = VALUE | NAME : ...` tail.

    llama.cpp prints `NAME : ` before each backend's features and nothing between backends, so a backend with
    no features runs straight into the next (`MTL : CPU : NEON = 1`); every leading `NAME : ` is a backend.
    """
    backends: list[str] = []
    features: dict[str, dict[str, str]] = {}
    current = None
    for piece in rest.split("|"):
        piece = piece.strip()
        while prefix := re.match(r"(\w+) :(?: |$)", piece):
            current = prefix[1]
            if current not in features:
                backends.append(current)
                features[current] = {}
            piece = piece[prefix.end():].lstrip()
        feature = re.fullmatch(r"(\w+) = (.*)", piece)
        if feature and current is not None:
            features[current][feature[1]] = feature[2]
    return backends, features


def _compute_buffers(text: str) -> list[dict]:
    """The compute buffers each llama_context holds once loading is done: its LAST reservation, not every line.

    `sched_reserve` re-reserves a context whose graph changed and prints the whole new allocation, which
    replaces the old one -- the live MTP log reserves its draft context twice (`graph nodes = 50`, then `= 56`
    once speculative decoding attaches), each time printing `CUDA0 compute buffer size = 196.02 MiB`. Summing
    every line would count that memory twice. A line belongs to the context most recently constructed before
    it and to the reservation most recently opened before it; each context keeps its latest reservation.
    """
    contexts = [match.start() for match in _CONTEXT.finditer(text)]
    reservations = [match.start() for match in re.finditer(r"sched_reserve: reserving", text)]
    lines = [(max(bisect_right(contexts, match.start()) - 1, 0), bisect_right(reservations, match.start()),
              {"device": match[1], "mib": float(match[2])}) for match in _COMPUTE_BUFFER.finditer(text)]
    latest: dict[int, int] = {}
    for context, reservation, _ in lines:
        latest[context] = max(latest.get(context, reservation), reservation)
    return [{**row, "context": context} for context, reservation, row in lines if reservation == latest[context]]


def parse_native_startup(text: str) -> dict:
    """Unified-memory facts from a llama-server startup log: which devices exist and what Metal holds.

    The settings a candidate requested are read by `parse_startup_log`, from the same lines on every runtime.
    This reads what only a unified-memory host needs, and like that parser it never fills a missing line
    with a default -- an absent key means the log did not say:

    * `devices` -- every device under `device_info:` with its total and free MiB at startup. For `MTL0` the
      total is Metal's recommended working-set size (5461 MiB on an 8 GB M1), which is `metal_budget_mib`
      when the log lists exactly one Metal device. It is a budget carved from host memory, not VRAM.
    * `model_device` -- the one device `llama_prepare_model_devices` placed the model on (`model_devices`
      lists every such line, so a multi-device placement is visible rather than truncated to its first).
    * `system_info_backends` / `system_info_features` -- the `system_info` backends in order (`["MTL",
      "CPU"]`), and `metal_embed_library` when a Metal backend is listed: True only when it printed
      `EMBED_LIBRARY = 1`, since a backend's printed feature list is complete.
    * `model_buffers`, `kv_buffers`, `recurrent_buffers`, `compute_buffers` -- every `{device, mib}` line;
      compute buffers keep only each context's latest reservation, see `_compute_buffers`.
    * `metal_device` -- what `ggml_metal_init: found device:` named.
    * `resident_by_device`, `host_resident_mib`, `metal_resident_mib` -- model + KV + recurrent + compute
      MiB per device, then summed over host (`_host_buffer`) and Metal (`MTL*`, which includes the
      `MTL0_Mapped` weights of `--load-mode mmap`) devices. They are written only when model buffers were
      printed and EVERY llama_context the log constructed has its compute reservation, which every completed
      load does: a log cut off before a context's `sched_reserve` -- the target's, or a draft context's after
      its KV cache was already allocated -- would otherwise under-report residency as if it were the whole.
      Context output buffers are left out, as llama.cpp's `llama_context::memory_breakdown` leaves them out.
      For a single-context run the sums therefore reconcile with `parse_memory_breakdown` (host: the `Host` row
      plus any other host-side buffer-type row such as `CPU_REPACK`); llama-server prints that table for its
      target context only, so a draft context's KV and compute appear here and not there.
      `metal_resident_mib` is written only when the log shows a Metal device.
    * `offloaded_layers` / `total_layers` -- as in `parse_startup_log`.
    """
    found: dict = {"parser_version": NATIVE_PARSER_VERSION}
    devices = _device_info(text)
    if devices:
        found["devices"] = devices
    placed = [{"name": match[1], "description": match[2], "device_id": match[3], "free_mib": int(match[4])}
              for match in _MODEL_DEVICE.finditer(text)]
    if placed:
        found["model_devices"] = placed
        if len(placed) == 1:
            found["model_device"] = placed[0]
    info = re.search(r"system_info: [^|\n]*\|(.*)$", text, re.MULTILINE)
    if info:
        found["system_info_backends"], found["system_info_features"] = _system_info(info[1])
        if "MTL" in found["system_info_features"]:
            found["metal_embed_library"] = found["system_info_features"]["MTL"].get("EMBED_LIBRARY") == "1"
    metal = re.search(r"ggml_metal_init: found device: (.+?)\s*$", text, re.MULTILINE)
    if metal:
        found["metal_device"] = metal[1]
    metal_devices = [item for item in devices if _metal_buffer(item["name"])]
    if len(metal_devices) == 1:
        found["metal_budget_mib"] = metal_devices[0]["total_mib"]
    for key, pattern in (("model_buffers", _MODEL_BUFFER), ("kv_buffers", _KV_BUFFER),
                         ("recurrent_buffers", _RS_BUFFER)):
        rows = [{"device": match[1], "mib": float(match[2])} for match in pattern.finditer(text)]
        if rows:
            found[key] = rows
    compute = _compute_buffers(text)
    if compute:
        found["compute_buffers"] = compute
    constructed = range(len(_CONTEXT.findall(text)))
    if "model_buffers" in found and constructed and {row["context"] for row in compute} == set(constructed):
        per_device: dict[str, float] = {}
        for key in ("model_buffers", "kv_buffers", "recurrent_buffers", "compute_buffers"):
            for row in found.get(key, ()):
                per_device[row["device"]] = per_device.get(row["device"], 0.0) + row["mib"]
        found["resident_by_device"] = {device: round(mib, 3) for device, mib in per_device.items()}
        found["host_resident_mib"] = round(sum(mib for device, mib in per_device.items()
                                               if _host_buffer(device)), 3)
        if metal_devices or metal or any(_metal_buffer(device) for device in per_device):
            found["metal_resident_mib"] = round(sum(mib for device, mib in per_device.items()
                                                    if _metal_buffer(device)), 3)
    offloaded = re.search(_FIRST["offloaded"][0], text)
    if offloaded:
        found["offloaded_layers"], found["total_layers"] = int(offloaded[1]), int(offloaded[2])
    return found


def parse_memory_breakdown(text: str) -> list[dict]:
    """The `common_memory_breakdown_print` table llama-server prints as it exits, one dict per row.

    A device row carries `total = free + (self = model + context + compute) + unaccounted` (MiB, truncated by
    llama.cpp, so a row's parts can sum one below its total; `unaccounted` is signed, and Metal clamps `free`
    at 0 once allocations pass the working-set size, so an over-committed MTL0 row prints it negative); the
    `Host` row and each buffer type that is neither host nor a model device (`CPU_REPACK`) carry only
    `self = model + context + compute`, and their `total`, `free` and `unaccounted` are None rather than zero.
    The table covers the one llama_context llama-server passes it, the target's: never a draft context. `device` is the row's first word (`MTL0`, `Host`), `description` its parenthesised name or None,
    and `table` counts the tables printed before it, so a log with more than one never merges their rows. A row
    in neither shape is left out: this reads what llama.cpp printed, it does not reconstruct what it meant.
    """
    headers = [match.start() for match in _BREAKDOWN_HEADER.finditer(text)]
    parsed = [(match, _DEVICE_COLUMNS) for match in _BREAKDOWN_DEVICE.finditer(text)]
    parsed += [(match, _HOST_COLUMNS) for match in _BREAKDOWN_HOST.finditer(text)]
    rows = []
    for match, columns in sorted(parsed, key=lambda item: item[0].start()):
        values = dict(zip(columns, (int(item) for item in match.groups()[1:])))
        name = re.fullmatch(r"(\S+)(?: \((.*)\))?", match[1])
        rows.append({"device": name[1] if name else match[1], "description": name[2] if name else None,
                     **{column: values.get(column) for column in _DEVICE_COLUMNS},
                     "table": max(bisect_right(headers, match.start()) - 1, 0)})
    return rows


def _compare(control, requested, effective, status, source, *, equal=None) -> SettingEvidence:
    if effective is None:
        return SettingEvidence(control=control, requested=requested, status="unobserved",
                               evidence=f"{source}: not reported", required=control in REQUIRED_CONTROLS)
    matches = requested == effective if equal is None else equal
    return SettingEvidence(control=control, requested=requested, effective=effective,
                           status=status if matches else "mismatch", evidence=source,
                           required=control in REQUIRED_CONTROLS)


def _accepted(control, requested, argv, flag) -> SettingEvidence:
    present = flag in argv
    return SettingEvidence(control=control, requested=requested, status="argv-accepted" if present else "unobserved",
                           evidence=f"{flag} accepted at startup; no API or log readback in this build"
                           if present else f"{flag} was not passed", required=control in REQUIRED_CONTROLS)


def _placement_detail(placement: dict) -> str:
    """The MiB on each device behind an observed placement, as display text: never a verdict, only a witness."""
    def side(name: str) -> str:
        devices = ", ".join(placement.get(f"{name}_devices") or []) or "an unnamed device"
        return "{:g} MiB on {}".format(placement.get(f"{name}_mib") or 0.0, devices)

    if placement.get("placement") == "split":
        return f"ram {side('ram')} / gpu {side('gpu')}"
    return side("ram" if placement.get("placement") == "ram" else "gpu")


def _kv_note(settings: LlamaCppSettings, observed: str, allowed, total, kv_layers) -> str:
    """Why the observed placement does or does not answer the request, in the row's own words."""
    if observed not in allowed:
        if observed == "split":
            return "; a split cache contradicts this request"
        if observed == "gpu":
            return "; a device KV buffer contradicts --no-kv-offload"
        # Deliberately not "the offload reached a KV-bearing block": with no layer count in the log that is
        # the rule's default, not an observation. What IS observed is a whole cache this request cannot place.
        return "; the whole cache is in system RAM, which an offload of this size cannot account for"
    if observed == "split":
        return "; a partial offload splits the cache: the blocks left on the CPU keep their KV in system RAM"
    if observed == "ram" and settings.kv_offload:
        if settings.n_gpu_layers == 0:
            return ("; nothing was offloaded, so the cache had nowhere but system RAM: this is the only "
                    "placement the request can produce, and it cannot tell the flag's two settings apart")
        return (f"; no KV-bearing block reached the GPU -- {settings.n_gpu_layers} of {total} blocks were "
                f"offloaded and only {kv_layers} of them carry a cache -- so this is a placement the request "
                "can produce, but it cannot tell the flag's two settings apart")
    return ""


def _kv_offload(settings: LlamaCppSettings, startup: dict, total, log: str) -> SettingEvidence:
    """Judge the observed KV placement against what the request can physically mean.

    The old binary reading -- "the first KV buffer is a CPU buffer, so the cache is in RAM" -- failed every
    partial-offload candidate: `n_gpu_layers=48` on a 66-block model with `kv_offload=True` prints a CPU AND a
    CUDA0 KV buffer, and that split IS the requested configuration honoured, so `offload-48` and `offload-32`
    were failed at `verify` and their measured tok/s was discarded. `LlamaCppSettings.consistent_kv_placements`
    owns the rule; this function only reports what was seen.

    `effective` is the observed placement itself -- one of `KV_PLACEMENTS`, in every case -- and the MiB on
    each device go to `detail`. The value no longer changes type by case, so no reader can turn a split into
    "all on the GPU" by evaluating it for truth. A placement that contradicts the request -- a
    device buffer under `--no-kv-offload`, a CPU buffer under a full offload, or an all-RAM cache under an
    offload long enough to have reached a KV-bearing block -- is still a mismatch.

    An all-RAM cache under a very short offload, or under `n_gpu_layers=0`, is a placement the request itself
    produces, so it is recorded as observed and verified rather than accused or ignored; what it cannot do is
    tell the flag's two settings apart, and the evidence string says exactly that.
    """
    placement = startup.get("kv_placement") or {}
    observed, source = placement.get("placement"), log + ": KV buffer device"
    if observed is None:
        return _compare("kv_offload", settings.kv_offload, None, "verified-log", source)
    kv_layers = (startup.get("kv_cache") or {}).get("layers")
    allowed = settings.consistent_kv_placements(total, kv_layers)
    row = _compare("kv_offload", settings.kv_offload, observed, "verified-log",
                   source + _kv_note(settings, observed, allowed, total, kv_layers), equal=observed in allowed)
    return row.model_copy(update={"detail": _placement_detail(placement)})


def _fit(settings: LlamaCppSettings, argv, rows) -> SettingEvidence:
    """`--fit off` is a prohibition, and what it forbids is observable even though the flag is not.

    b11011 prints no fit line at verbosity 4 and carries no fit field on `/props`, so the flag's own value
    cannot be read back. Its whole documented effect is to adjust arguments to fit in device memory, and the
    arguments it can move -- context size, GPU layer count, logical and physical batch size -- are all passed
    explicitly by `build_server_argv` and read back here from `/props` and the startup log. When every one of
    them came back equal to the request, no device-memory adjustment happened to the measured configuration;
    that absence, not the flag register, is what `verified-behavior` records. If any of those observations is
    missing or mismatched the row stays `argv-accepted`: an unseen control could have been adjusted.
    """
    row = _accepted("fit", settings.fit, argv, "--fit")
    requested = [value for flag, value in zip(argv, argv[1:]) if flag == "--fit"]
    if row.status != "argv-accepted" or settings.fit != "off" or requested != ["off"]:
        return row  # `--fit` absent from argv, or argv asking for something other than the settings' `off`.
    witnesses = [next((item for item in rows if item.control == control), None) for control in FIT_ADJUSTABLE]
    missing = [control for control, item in zip(FIT_ADJUSTABLE, witnesses)
               if item is None or item.status not in VERIFIED]
    if missing:
        return row.model_copy(update={"evidence": row.evidence + "; no adjustment check either: "
                                      + ", ".join(missing) + " not verified"})
    observed = ", ".join(f"{item.control}={item.effective} ({item.status})" for item in witnesses)
    return SettingEvidence(control="fit", requested=settings.fit, effective="off", status="verified-behavior",
                           required="fit" in REQUIRED_CONTROLS,
                           evidence="--fit off: no device-memory adjustment observed; the fit-adjustable "
                                    f"arguments were each read back equal to the request: {observed}")


def build_settings_evidence(settings: LlamaCppSettings, argv, readback: dict | None, startup: dict | None,
                            overflow_probe: dict | None, *, alias: str | None = None,
                            build_info: str | None = None) -> tuple[SettingEvidence, ...]:
    readback, startup, overflow_probe = readback or {}, startup or {}, overflow_probe or {}
    props = readback.get("props") if isinstance(readback.get("props"), dict) else {}
    slots = readback.get("slots") if isinstance(readback.get("slots"), list) else []
    log = "startup log (" + PARSER_VERSION + ")"
    api_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    slot_ctx = {slot.get("n_ctx") for slot in slots if isinstance(slot, dict)}
    rows = []
    if api_ctx is not None:
        consistent = api_ctx == settings.ctx_size and slot_ctx <= {api_ctx} and startup.get("n_ctx", api_ctx) == api_ctx
        rows.append(_compare("ctx_size", settings.ctx_size, api_ctx, "verified-api",
                             "/props default_generation_settings.n_ctx, /slots and startup log", equal=consistent))
    else:
        rows.append(_compare("ctx_size", settings.ctx_size, startup.get("n_ctx"), "verified-log", log))
    rows.append(_compare("parallel", settings.parallel, props.get("total_slots"), "verified-api", "/props total_slots")
                if props.get("total_slots") is not None else
                _compare("parallel", settings.parallel, startup.get("n_slots"), "verified-log", log))
    offloaded, total = startup.get("offloaded_layers"), startup.get("total_layers")
    if offloaded is None:
        rows.append(_compare("n_gpu_layers", settings.n_gpu_layers, None, "verified-log", log))
    else:
        wanted = total if settings.n_gpu_layers == "all" else min(settings.n_gpu_layers, total)
        rows.append(_compare("n_gpu_layers", settings.n_gpu_layers, f"{offloaded}/{total}", "verified-log",
                             log + ": offloaded layers", equal=offloaded == wanted))
    cache = startup.get("kv_cache") or {}
    rows.append(_compare("cache_type_k", settings.cache_type_k, cache.get("k_type"), "verified-log", log))
    rows.append(_compare("cache_type_v", settings.cache_type_v, cache.get("v_type"), "verified-log", log))
    rows.append(_kv_offload(settings, startup, total, log))
    flash = {"enabled": "on", "disabled": "off"}.get(startup.get("flash_attn"), startup.get("flash_attn"))
    rows.append(_compare("flash_attn", settings.flash_attn, flash, "verified-log", log))
    if settings.flash_attn == "on":
        pair, kernels = [settings.cache_type_k, settings.cache_type_v], startup.get("fa_quants")
        rows.append(_compare("flash_attn_kernel", pair, kernels, "verified-log", log + ": FA_QUANTS",
                             equal=None if kernels is None else pair in kernels))
    for control, key in (("load_mode", "load_mode"), ("batch_size", "n_batch"), ("ubatch_size", "n_ubatch"),
                         ("threads", "n_threads"), ("threads_batch", "n_threads_batch"),
                         ("cache_ram_mib", "prompt_cache_mib"), ("log_verbosity", "verbosity")):
        rows.append(_compare(control, getattr(settings, control), startup.get(key), "verified-log", log))
    rejected = overflow_probe.get("passed") is True or overflow_probe.get("rejected") is True
    rows.append(SettingEvidence(control="context_shift", requested=False, effective=False, required=True,
                                status="verified-behavior", evidence="over-capacity request was rejected")
                if rejected else _accepted("context_shift", False, argv, "--no-context-shift"))
    rows.append(_fit(settings, argv, rows))
    speculative = startup.get("speculative")
    effective = None if speculative is None else ",".join(speculative["implementations"]) or "none"
    rows.append(_compare("spec_type", settings.spec_type, effective, "verified-log", log))
    if settings.spec_type != "none":
        rows.append(_compare("spec_draft_n_max", settings.spec_draft_n_max, (speculative or {}).get("n_max"),
                             "verified-log", log))
    thinking = {1: "on", 0: "off"}.get(startup.get("thinking"))
    rows.append(_accepted("reasoning", settings.reasoning, argv, "--reasoning") if settings.reasoning == "auto"
                else _compare("reasoning", settings.reasoning, thinking, "verified-log", log + ": thinking flag"))
    rows.append(_compare("warmup", True, startup.get("warmup"), "verified-log", log) if settings.warmup
                else _accepted("warmup", False, argv, "--no-warmup"))
    for control, flag in (("cache_prompt", "--cache-prompt" if settings.cache_prompt else "--no-cache-prompt"),
                          ("cache_reuse", "--cache-reuse"), ("jinja", "--jinja"),
                          ("reasoning_format", "--reasoning-format"), ("reasoning_budget", "--reasoning-budget"),
                          *((("swa_full", "--swa-full"),) if settings.swa_full else ())):
        rows.append(_accepted(control, getattr(settings, control), argv, flag))
    if alias is not None:
        rows.append(_compare("alias", alias, props.get("model_alias"), "verified-api", "/props model_alias"))
    if build_info is not None:
        rows.append(_compare("build_info", build_info, props.get("build_info") or startup.get("build"),
                             "verified-api" if props.get("build_info") else "verified-log",
                             "/props build_info" if props.get("build_info") else log))
    return tuple(rows)


def unverified_required(evidence) -> list[str]:
    return [row.control for row in evidence if row.required and row.status not in VERIFIED]


def settings_verified(evidence) -> bool:
    controls = {row.control for row in evidence if row.required}
    expected = {name for name in REQUIRED_CONTROLS if name != "spec_draft_n_max"}
    return expected <= controls and not unverified_required(evidence)
