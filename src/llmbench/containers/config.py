"""Strict, versioned contracts for one llama.cpp container candidate. Importing performs no I/O."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from ..coding.sandbox import SandboxLimits
from ..config import GenerationSettings, StrictModel, TaskSelection, canonical_json, freeze_json

SHA256 = r"^[0-9a-f]{64}$"
IMAGE_ID = r"^sha256:[0-9a-f]{64}$"
DIGEST_REF = r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$"
NAME = r"^[a-z0-9][a-z0-9._-]{0,62}$"
BUILD_INFO = r"^b\d+-[0-9a-f]{7,40}$"
MODEL_CONTAINER_PATH = "/models/model.gguf"
EVALUATOR_ENTRYPOINT = ("python", "-m", "llmbench.container_eval")
# The inference runtimes a candidate can name. `nvidia-container` is the original pinned CUDA image run under
# Docker Compose and stays the default: a config that does not name a runtime serialises, fingerprints and runs
# exactly as it did before runtimes existed. `metal-native` runs a pinned llama-server executable directly on an
# Apple Silicon host with Metal offload; generated code still only ever runs in the sandboxed worker containers.
RUNTIMES = ("nvidia-container", "metal-native")
DEFAULT_RUNTIME = "nvidia-container"
CacheType = Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
# b11011 system_info reports FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 (observed live).
FLASH_ATTN_KV_PAIRS = frozenset({("f16", "f16"), ("bf16", "bf16"), ("q8_0", "q8_0"), ("q4_0", "q4_0")})
UNQUANTIZED_CACHE = frozenset({"f32", "f16", "bf16"})
# The three states a KV cache can be observed in. This is the whole vocabulary: a `kv_offload` evidence row's
# `effective` value is one of these and nothing else, so no reader has to infer a placement from truthiness
#. The MiB on each device live in that row's `detail`, which is display text only.
KV_PLACEMENTS = ("gpu", "ram", "split")


class ImageRef(StrictModel):
    role: Literal["inference", "evaluator", "worker"]
    reference: str
    image_id: str = Field(pattern=IMAGE_ID)
    # linux/arm64 exists for sandbox WORKER images built natively on an Apple Silicon host's Linux VM; the
    # inference and evaluator images this project pins are linux/amd64 and a stored value is never guessed.
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64"
    entrypoint: tuple[str, ...] = Field(min_length=1)
    build_info: str | None = None
    help_sha256: str | None = Field(default=None, pattern=SHA256)
    source: str | None = None

    @model_validator(mode="after")
    def immutable_reference(self) -> "ImageRef":
        if not (re.fullmatch(DIGEST_REF, self.reference) or re.fullmatch(IMAGE_ID, self.reference)):
            raise ValueError("image reference must be a repository digest or a local image ID; tags are rejected")
        if any(not item or "\x00" in item for item in self.entrypoint):
            raise ValueError("entrypoint items must be nonempty strings")
        if self.role == "inference" and (not self.help_sha256 or not self.build_info):
            raise ValueError("an inference image requires build_info and the normalized help_sha256")
        if self.role != "worker" and self.platform != "linux/amd64":
            raise ValueError("only a sandbox worker image may be linux/arm64")
        return self


class NativeServerRef(StrictModel):
    """A pinned llama-server executable run directly on the host: the native analogue of an inference `ImageRef`.

    Identity is the executable's SHA-256 plus `libraries_sha256`, a digest over every llama.cpp/ggml library
    shipped beside it (the Metal backend lives in `libggml-metal`, so hashing the executable alone would not pin
    what runs). `build_info` and `help_sha256` are read from the executable's own `--version` and `--help` and
    re-checked at admission. `executable` is where it lives on THIS host and, like `ModelAsset.host_path`, is not
    part of the fingerprint; `source` is display text only.
    """

    role: Literal["inference"] = "inference"
    backend: Literal["metal"] = "metal"
    platform: Literal["darwin/arm64"] = "darwin/arm64"
    executable: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=SHA256)
    libraries_sha256: str = Field(pattern=SHA256)
    build_info: str = Field(pattern=BUILD_INFO)
    help_sha256: str = Field(pattern=SHA256)
    source: str | None = None

    @field_validator("executable")
    @classmethod
    def absolute_executable(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or any(char in value for char in ("\r", "\n", "\x00")):
            raise ValueError("native executable must be an absolute path without parent traversal")
        return value


class NativeLimits(StrictModel):
    """Admission and watchdog limits for a native (unified-memory) server. There is no cgroup on a macOS host
    process, so these are checked by the runner instead of enforced by a kernel: admission refuses to start when
    they already fail, and the watchdog stops the server when they fail during the run.

    Memory on Apple Silicon is one pool shared by the CPU, the GPU and every other application, so every number
    here is labelled for what it is: process memory is the server's `phys_footprint` (what Activity Monitor calls
    Memory, including its Metal allocations), `available` is the host's available memory, and swap is host-wide.
    None of them is VRAM.
    """

    # Admission: the host must have at least the model file plus this reserve available before the load.
    memory_reserve_mib: int = Field(default=1024, ge=0, le=1_048_576)
    # Admission: macOS kern.memorystatus_vm_pressure_level (1 normal, 2 warn, 4 critical) at or below this.
    max_memory_pressure_level: Literal[1, 2, 4] = 2
    # Admission: the GPU's own utilisation (IOAccelerator "Device Utilization %") sampled before the load. A game
    # or another inference server shares the GPU with the measurement; that is a conflict to report, never a
    # process to stop.
    max_foreign_gpu_utilization_percent: int = Field(default=50, ge=0, le=100)
    # Watchdog: stop the server when its phys_footprint exceeds this. None means Metal's own
    # recommendedMaxWorkingSetSize as the server logged it at startup.
    max_server_footprint_mib: int | None = Field(default=None, ge=256, le=2_097_152)
    # Watchdog: stop the server when host swap in use grows by more than this during the candidate.
    max_swap_growth_mib: int = Field(default=2048, ge=0, le=1_048_576)
    # How often the watchdog samples memory while the evaluator runs.
    sample_interval_seconds: float = Field(default=1.0, ge=0.2, le=30.0)
    # Seconds between SIGTERM and SIGKILL when stopping the server.
    stop_grace_seconds: int = Field(default=15, ge=1, le=120)


class ModelAsset(StrictModel):
    role: Literal["main"] = "main"
    host_path: str = Field(min_length=1)
    container_path: Literal["/models/model.gguf"] = MODEL_CONTAINER_PATH
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=SHA256)
    format: Literal["gguf"] = "gguf"
    quantization: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)

    @field_validator("host_path")
    @classmethod
    def safe_host_path(cls, value: str) -> str:
        if any(char in value for char in (",", "\r", "\n", "\x00")):
            raise ValueError("host_path cannot contain commas or control characters")
        windows, posix = PureWindowsPath(value), PurePosixPath(value)
        if not ((windows.drive and windows.root) or posix.is_absolute()):
            raise ValueError("host_path must be absolute")
        if ".." in windows.parts or ".." in posix.parts:
            raise ValueError("host_path cannot contain parent traversal")
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", value, re.IGNORECASE):
            raise ValueError("split GGUF files are not supported by this version")
        return value


class LlamaCppSettings(StrictModel):
    """One candidate's llama-server configuration.

    ``kv_offload`` is a PROHIBITION, not a destination: True (``--kv-offload``) means "do not force the KV
    cache off the GPU", False (``--no-kv-offload``) means "keep all of it in system RAM". The cache follows
    the layers, so ``kv_offload=True`` with a partial ``n_gpu_layers`` is coherent and is accepted here: the
    blocks left on the CPU keep their KV in system RAM and the cache is honestly split across both devices.
    ``consistent_kv_placements`` states which observations that leaves room for, and `readback.py` records the
    split (with the MiB on each device) rather than calling it a mismatch.
    """

    engine: Literal["llama.cpp"] = "llama.cpp"
    ctx_size: int = Field(ge=512, le=1_048_576)
    n_gpu_layers: Annotated[int, Field(ge=0, le=999)] | Literal["all"] = "all"
    cache_type_k: CacheType = "f16"
    cache_type_v: CacheType = "f16"
    kv_offload: bool = True
    flash_attn: Literal["on", "off"] = "on"
    load_mode: Literal["none", "mmap"] = "none"
    batch_size: int = Field(default=2048, ge=32, le=16384)
    ubatch_size: int = Field(default=512, ge=32, le=8192)
    threads: int = Field(default=8, ge=1, le=64)
    threads_batch: int = Field(default=8, ge=1, le=64)
    parallel: Literal[1] = 1
    cache_prompt: bool = False
    cache_reuse: Literal[0] = 0
    cache_ram_mib: Literal[0] = 0
    context_shift: Literal[False] = False
    fit: Literal["off"] = "off"
    swa_full: bool = False
    jinja: Literal[True] = True
    reasoning: Literal["on", "off", "auto"] = "auto"
    reasoning_format: Literal["deepseek", "none"] = "deepseek"
    reasoning_budget: int = Field(default=-1, ge=-1, le=32768)
    spec_type: Literal["none", "draft-mtp"] = "none"
    spec_draft_n_max: int = Field(default=3, ge=1, le=16)
    warmup: bool = True
    log_verbosity: int = Field(default=4, ge=3, le=4)

    @model_validator(mode="after")
    def coherent(self) -> "LlamaCppSettings":
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        if self.batch_size > self.ctx_size:
            raise ValueError("batch_size cannot exceed ctx_size; llama.cpp would silently clamp it")
        if self.flash_attn == "off" and self.cache_type_v not in UNQUANTIZED_CACHE:
            raise ValueError("a quantized cache_type_v requires flash_attn on")
        if self.flash_attn == "on" and (self.cache_type_k, self.cache_type_v) not in FLASH_ATTN_KV_PAIRS:
            raise ValueError("with flash_attn on this build only has kernels for matching K/V cache types "
                             "f16, bf16, q8_0 or q4_0")
        if self.spec_type == "none" and self.spec_draft_n_max != 3:
            raise ValueError("spec_draft_n_max requires a speculative spec_type")
        # `kv_offload=True` with a partial `n_gpu_layers` is NOT refused here (LIVE-010): it is the ordinary
        # meaning of a partial offload, the block count that would decide "partial" is a property of the model
        # and not of these settings, and refusing it would make every point of the `gpu_layers` sweep
        # unconfigurable. See the class docstring and `consistent_kv_placements`.
        return self

    def consistent_kv_placements(self, total_layers: int | None = None,
                                 kv_layers: int | None = None) -> tuple[str, ...]:
        """Which observed KV-cache placements -- "gpu", "ram", "split" -- this request can honestly produce.

        Live evidence (`artifacts/container-campaign/live-offload-2/candidates/001-offload-48`): asking for 48
        of 66 blocks on the GPU with the KV cache left on the GPU printed BOTH an 84 MiB `CPU` and a 252 MiB
        `CUDA0` KV buffer. A partial offload necessarily splits the cache, so "all KV on the GPU" is not a
        thing a partial offload can mean; what the flag rules out is the cache being pushed off the GPU.

        A partial request also admits "gpu": on a hybrid model only some blocks carry a KV cache, so an
        offload can cover every one of them and leave nothing in RAM. It admits "ram" for the mirror image --
        an offload short enough that no KV-bearing block reached the GPU -- whenever `kv_layers` says that is
        arithmetically possible; see `_offload_can_miss_every_kv_layer`, which is the quantitative half of
        this rule. `--no-kv-offload` admits only "ram" -- any device KV buffer contradicts it --
        and a full offload admits only "gpu", because with every block on the GPU a `CPU` KV buffer means the
        cache was pushed off it. When the model's block count is unknown a partial request cannot be told from
        a full one, so both survive; the `n_gpu_layers` evidence row reports that gap on its own rather than
        having this rule guess.

        `n_gpu_layers == 0` admits exactly "ram" -- the cache has nowhere else to be -- which is a single
        honest outcome, so `readback._kv_offload` verifies an observed `ram` there instead of declining to
        look. What such an observation cannot do is tell the flag's two settings apart, and the
        evidence row says so in words rather than leaving the reader to work it out.
        """
        if not self.kv_offload:
            return ("ram",)
        if self.n_gpu_layers == "all" or (total_layers is not None and self.n_gpu_layers >= total_layers):
            return ("gpu",)
        if self.n_gpu_layers == 0:
            return ("ram",)  # nothing on the GPU: the cache has nowhere else to be, whatever the flag asked.
        if self._offload_can_miss_every_kv_layer(total_layers, kv_layers):
            return ("gpu", "split", "ram")
        return ("gpu", "split")

    def _offload_can_miss_every_kv_layer(self, total_layers: int | None, kv_layers: int | None) -> bool:
        """Whether this partial offload can leave EVERY KV-bearing block in system RAM.

        llama.cpp offloads the LAST `n_gpu_layers` of `total_layers` entries, and only `kv_layers` of those
        entries carry a KV cache -- 16 of 66 on the live hybrid target (`llama_kv_cache: size = 336.00 MiB
        (5376 cells, 16 layers, ...)`), about one in four. The offloaded tail therefore falls entirely between
        two KV-bearing blocks exactly when it is shorter than their spacing `total_layers / kv_layers`, i.e.
        when `n_gpu_layers * kv_layers < total_layers`. On that hybrid model four blocks or fewer qualify; on
        a dense model (`kv_layers` one below `total_layers`, the non-KV output entry) only a single block
        does, so a dense model's all-RAM cache under any real offload stays the mismatch it should be.

        This is deliberately a possibility test on the layer counts the log itself printed, not a guess at
        which blocks carry a cache: it never admits an observation the counts forbid, and never accuses one
        they allow. An unknown or incoherent count admits nothing, which keeps the old rule.
        """
        if not total_layers or not kv_layers or kv_layers > total_layers:
            return False
        return self.n_gpu_layers * kv_layers < total_layers


def kv_placement_reading(effective: Any, detail: Any = "") -> tuple[str | None, str]:
    """The three-state placement a `kv_offload` evidence row observed, plus its display detail.

    Returns one of `KV_PLACEMENTS` (or `None` when the row observed nothing) and the MiB-per-device text to
    print beside it. Every consumer of a stored row goes through here, because a report is built from whatever
    the attempt SEALED and older builds wrote other shapes: before LIVE-010 the row was a bare bool (`True`
    meaning the cache was on the GPU), and LIVE-010 wrote the string
    `"split (ram 84 MiB / gpu 252 MiB on CUDA0)"` for a split while keeping bools for the other two states --
    which is what made `session_report` print a split cache as `gpu`. Anything unrecognised reads
    as "no placement observed", never as a placement.
    """
    text = detail.strip() if isinstance(detail, str) else ""
    if isinstance(effective, bool):  # rows sealed before LIVE-010: True is the cache on the GPU, False in RAM
        return ("gpu" if effective else "ram"), text
    if isinstance(effective, str):
        state, _, rest = effective.partition(" ")
        if state in KV_PLACEMENTS:
            return state, text or rest.strip().strip("()").strip()
    return None, ""


class BenchmarkSelection(StrictModel):
    benchmark_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    task_ids: tuple[str, ...] = Field(min_length=1)
    split: Literal["development", "holdout"] = "development"
    seed: int = Field(default=42, ge=0)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_ids")
    @classmethod
    def unique_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or not all(value):
            raise ValueError("task_ids must be unique nonempty strings")
        return value

    @field_validator("options")
    @classmethod
    def bounded_json_options(cls, value):
        """Options are typed, bounded, frozen JSON. Which KEYS a benchmark accepts is the registry's
        call (``RegistryEntry.option_keys``), checked at preflight; this validator only guarantees the
        value cannot smuggle a command, a path or an unbounded structure into an adapter."""
        if len(value) > 16:
            raise ValueError("a benchmark selection may carry at most 16 options")
        scalar = (str, int, float, bool)
        for key, item in value.items():
            if type(key) is not str or not key or len(key) > 64 or not key.replace("_", "").isalnum():
                raise ValueError(f"invalid benchmark option name: {key!r}")
            items = item if isinstance(item, (list, tuple)) else [item]
            if isinstance(item, (list, tuple)) and len(items) > 64:
                raise ValueError(f"benchmark option {key} has too many values")
            for element in items:
                if isinstance(element, bool) or not isinstance(element, scalar):
                    if not isinstance(element, scalar):
                        raise ValueError(f"benchmark option {key} must hold scalars or a list of them")
                if isinstance(element, str) and (len(element) > 256 or chr(0) in element):
                    raise ValueError(f"benchmark option {key} has an oversized or invalid string")
        return freeze_json(value)

    def to_task_selection(self) -> TaskSelection:
        return TaskSelection(suite=self.benchmark_id, revision=self.revision, task_ids=self.task_ids,
                             fixture_seed=self.seed, split=self.split, options=dict(self.options))


def terminal_reserve_bytes(max_artifact_bytes: int) -> int:
    """The runner's reserved terminal capacity (result.json + index); child allocations must stay below it."""
    return min(262144, max_artifact_bytes // 4)


class ResourceLimits(StrictModel):
    gpu_device_id: str = Field(default="0", pattern=r"^\d{1,2}$")
    # Upper bounds only reject nonsense: what a host can really give a container is its own business, and a
    # RAM-offloaded candidate legitimately needs far more than a GPU-resident one. Docker refuses a limit the
    # machine cannot honour, which is a clearer failure than a schema tuned to one workstation.
    inference_memory_mib: int = Field(default=24576, ge=4096, le=2_097_152)
    inference_cpus: float = Field(default=12.0, gt=0, le=512)
    evaluator_memory_mib: int = Field(default=4096, ge=512, le=16384)
    evaluator_cpus: float = Field(default=2.0, gt=0, le=8)
    evaluator_artifact_bytes: int = Field(default=268_435_456, ge=65536)
    evaluator_pids: int = Field(default=256, ge=16, le=4096)
    evaluator_tmpfs_mib: int = Field(default=256, ge=16, le=4096)
    log_max_bytes: int = Field(default=16 * 1024 * 1024, ge=65536, le=256 * 1024 * 1024)
    max_artifact_bytes: int = Field(default=1_073_741_824, ge=1_048_576)
    max_foreign_vram_mib: int = Field(default=4096, ge=0)

    def child_allocation_fits(self) -> bool:
        return self.evaluator_artifact_bytes < self.max_artifact_bytes - terminal_reserve_bytes(self.max_artifact_bytes)


class BrokerSettings(StrictModel):
    """Host coding-broker limits. Defined here; the coding broker imports it."""
    # 64 covered the private fixtures; a public coding suite needs one request per EvalPlus item and two per
    # Aider Polyglot exercise (83 exercises in the python+javascript corpus), so the ceiling is sized for those.
    max_requests: int = Field(default=8, ge=1, le=1024)
    max_request_bytes: int = Field(default=2_097_152, ge=4096, le=16_777_216)
    max_patch_bytes: int = Field(default=1_048_576, ge=1024, le=8_388_608)
    fixture_timeout_seconds: int = Field(default=180, ge=30, le=900)
    worker_limits: SandboxLimits = Field(default_factory=SandboxLimits)


class StageBounds(StrictModel):
    candidate_wall_seconds: int = Field(default=1800, ge=60, le=14400)
    hash_seconds: int = Field(default=300, ge=1)
    startup_seconds: int = Field(default=300, ge=10, le=1800)
    readiness_poll_seconds: float = Field(default=1.0, gt=0, le=30)
    verify_seconds: int = Field(default=60, ge=1)
    request_timeout_seconds: int = Field(default=300, ge=1)
    evaluation_seconds: int = Field(default=1200, ge=1)
    cleanup_reserve_seconds: int = Field(default=60, ge=15)

    @model_validator(mode="after")
    def stages_fit(self) -> "StageBounds":
        if self.minimum_wall_seconds() >= self.candidate_wall_seconds:
            raise ValueError("hash + startup + verify + cleanup reserve must be smaller than candidate wall")
        return self

    def minimum_wall_seconds(self) -> int:
        return self.hash_seconds + self.startup_seconds + self.verify_seconds + self.cleanup_reserve_seconds


class SpeedProbe(StrictModel):
    repetitions: int = Field(default=3, ge=1, le=20)
    warmup_repetitions: int = Field(default=1, ge=0, le=5)
    output_tokens: int = Field(default=512, ge=16, le=4096)
    ignore_eos: bool = True


class EvaluatorSettings(StrictModel):
    mode: Literal["host-process", "container"] = "host-process"
    image: ImageRef | None = None
    watchdog_slack_seconds: int = Field(default=30, ge=5, le=300)

    @model_validator(mode="after")
    def image_matches_mode(self) -> "EvaluatorSettings":
        if (self.mode == "container") is (self.image is None):
            raise ValueError("an evaluator image is required exactly when mode is container")
        if self.image is not None and self.image.role != "evaluator":
            raise ValueError("evaluator image must have role evaluator")
        return self


class ContainerRunConfig(StrictModel):
    """One candidate. The name predates native runtimes: `runtime` says what serves the model.

    `nvidia-container` (the default) requires `inference_image`; `metal-native` requires `native_server` and
    `native_limits` and has no inference image. Keys that hold the NVIDIA defaults (`runtime`, `native_server`,
    `native_limits`) are left out of every serialisation, so an NVIDIA config's JSON, fingerprint, alias and
    Compose project are byte-identical to what they were before runtimes existed, and an evaluator image built
    from an older wheel still accepts it.
    """

    schema_version: Literal[1] = 1
    label: str = Field(pattern=NAME)
    session_id: str | None = Field(default=None, pattern=NAME)
    parent_grant_seconds: int | None = Field(default=None, ge=1)
    assets: tuple[ModelAsset, ...] = Field(min_length=1, max_length=1)
    runtime: Literal["nvidia-container", "metal-native"] = DEFAULT_RUNTIME
    inference_image: ImageRef | None = None
    native_server: NativeServerRef | None = None
    native_limits: NativeLimits | None = None
    evaluator: EvaluatorSettings = Field(default_factory=EvaluatorSettings)
    worker_image: ImageRef | None = None
    broker: BrokerSettings | None = None
    engine: LlamaCppSettings
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    speed: SpeedProbe = Field(default_factory=SpeedProbe)
    requested_input_tokens: int = Field(default=512, ge=0)
    template_reserve_tokens: int = Field(default=256, ge=0)
    benchmarks: tuple[BenchmarkSelection, ...] = Field(min_length=1)
    registry_digest: str | None = Field(default=None, pattern=SHA256)
    dataset_root: str | None = None
    """Where the pinned public-benchmark corpora live for THIS run. ``None`` means the evaluator
    image's baked path. A host-process evaluator must point at the host staging directory instead,
    because the container path does not exist on the host and every dataset reads as absent."""
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    bounds: StageBounds = Field(default_factory=StageBounds)
    harness_revision: str = Field(min_length=1)

    @model_serializer(mode="wrap")
    def _omit_default_runtime(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            if data.get("runtime") == DEFAULT_RUNTIME:
                data.pop("runtime")
            for key in ("inference_image", "native_server", "native_limits"):
                if key in data and data[key] is None:
                    data.pop(key)
        return data

    @model_validator(mode="after")
    def coherent(self) -> "ContainerRunConfig":
        if self.runtime == "nvidia-container":
            if self.inference_image is None:
                raise ValueError("the nvidia-container runtime requires inference_image")
            if self.native_server is not None or self.native_limits is not None:
                raise ValueError("native_server and native_limits belong to the metal-native runtime only")
        else:
            if self.native_server is None or self.native_limits is None:
                raise ValueError("the metal-native runtime requires native_server and native_limits")
            if self.inference_image is not None:
                raise ValueError("the metal-native runtime runs no inference image; remove inference_image")
            if self.evaluator.mode != "host-process":
                raise ValueError("the metal-native runtime supports only the host-process evaluator: a native "
                                 "server is not reachable from the evaluator container's internal network")
        if self.inference_image is not None and self.inference_image.role != "inference":
            raise ValueError("inference_image must have role inference")
        if self.worker_image is not None and self.worker_image.role != "worker":
            raise ValueError("worker_image must have role worker")
        output = max(self.generation.max_output_tokens, self.speed.output_tokens)
        if self.requested_input_tokens + self.template_reserve_tokens + output > self.engine.ctx_size:
            raise ValueError("input + template reserve + output exceeds context capacity")
        if self.generation.reasoning != "model-default":
            raise ValueError("reasoning is an engine flag here; generation.reasoning must stay model-default")
        if self.generation.tool_emulation:
            raise ValueError("native tool profile cannot emulate tool calls")
        if self.evaluator.mode == "container" and not self.limits.child_allocation_fits():
            raise ValueError("limits.evaluator_artifact_bytes must be smaller than max_artifact_bytes minus the "
                             "terminal reserve")
        if self.broker is None and any(item.benchmark_id == "coding" for item in self.benchmarks):
            raise ValueError("coding benchmarks need the worker broker: set `broker` to enable them")
        keys = [(item.benchmark_id, item.split, item.seed) for item in self.benchmarks]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate benchmark selection")
        return self

    @property
    def model(self) -> ModelAsset:
        return self.assets[0]

    @property
    def build_info(self) -> str:
        """The llama.cpp build serving this candidate, whichever runtime pins it."""
        server = self.native_server if self.runtime == "metal-native" else self.inference_image
        return server.build_info

    @property
    def help_sha256(self) -> str:
        """The pinned normalized `llama-server --help` hash, whichever runtime pins it."""
        server = self.native_server if self.runtime == "metal-native" else self.inference_image
        return server.help_sha256

    @property
    def server_model_path(self) -> str:
        """The exact path the server is given with `--model`: the bind-mount target inside the inference
        container, or the host file itself for a native server."""
        return self.model.host_path if self.runtime == "metal-native" else MODEL_CONTAINER_PATH

    def fingerprint(self) -> str:
        # Labels, grants, stage bounds, host locations and display sources do not change what is measured.
        payload = self.model_dump(mode="json", exclude={"label", "session_id", "parent_grant_seconds",
                                                       "bounds", "dataset_root"})
        payload["evaluator"].pop("watchdog_slack_seconds")
        for asset in payload["assets"]:
            asset.pop("host_path")
        for image in (payload.get("inference_image"), payload["worker_image"], payload["evaluator"]["image"]):
            if image is not None:
                image.pop("source")
        if payload.get("native_server") is not None:
            payload["native_server"].pop("executable")
            payload["native_server"].pop("source")
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def alias(self) -> str:
        return "llmbench-" + self.fingerprint()[:12]


class StageRecord(StrictModel):
    name: Literal["admit", "hash", "plan", "start", "ready", "verify", "warmup", "speed", "quality", "report",
                  "cleanup", "broker", "evaluator"]
    status: Literal["ok", "failed", "timeout", "skipped"]
    started_offset_seconds: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    detail: str = ""


class SettingEvidence(StrictModel):
    """One control, what was asked for, what was observed, and how strongly that observation was made.

    `effective` is the observed VALUE and is what a renderer prints; `detail` is optional display text about
    that value (the MiB on each KV device, say) and is never load-bearing for a verdict. A consumer that needs
    the state of a KV placement reads `effective` through `kv_placement_reading`, never by truthiness.
    """

    control: str = Field(min_length=1)
    requested: Any = None
    effective: Any = None
    status: Literal["verified-api", "verified-log", "verified-behavior", "argv-accepted", "mismatch", "unobserved"]
    evidence: str = ""
    detail: str = ""
    required: bool = False


class CleanupEvidence(StrictModel):
    attempted: bool = False
    compose_down_returncode: int | None = None
    containers_remaining: tuple[str, ...] = ()
    networks_remaining: tuple[str, ...] = ()
    verified: bool = False
    error: str | None = None
    lease_retained: bool = False
    # Native runtime only (omitted from the JSON when unset, so container results are unchanged): how the owned
    # server process ended ("returncode=0", "signal=SIGTERM", ...) and the owned processes still present after
    # the stop, as "pid:<n>" entries. A non-empty list is unverified cleanup, exactly like a leftover container.
    server_exit: str | None = None
    processes_remaining: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_native_defaults(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            if data.get("server_exit") is None:
                data.pop("server_exit", None)
            if not data.get("processes_remaining"):
                data.pop("processes_remaining", None)
        return data


class ArtifactEntry(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256)
    size: int = Field(ge=0)


class ContainerRunResult(StrictModel):
    schema_version: Literal[1] = 1
    session_id: str
    attempt_id: str
    config_fingerprint: str = Field(pattern=SHA256)
    project_name: str
    state: Literal["completed", "failed", "timeout", "cancelled", "rejected", "cleanup-uncertain"]
    synthetic: bool
    failure_stage: str | None = None
    failure_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    started_utc: str
    finished_utc: str
    elapsed_seconds: float = Field(ge=0)
    budget_charged_seconds: float = Field(ge=0)
    stages: tuple[StageRecord, ...] = ()
    container_ids: tuple[str, ...] = ()
    image_evidence: dict[str, Any] = Field(default_factory=dict)
    model_evidence: dict[str, Any] = Field(default_factory=dict)
    settings: tuple[SettingEvidence, ...] = ()
    effective_settings_verified: bool = False
    actual_context_verified: bool = False
    speed: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    samples_total: int = Field(default=0, ge=0)
    load_seconds: float | None = None
    vram_used_mib_after_load: float | None = None
    cleanup: CleanupEvidence = Field(default_factory=CleanupEvidence)
    abort_campaign: bool = False
    artifacts: tuple[ArtifactEntry, ...] = ()
    reports: dict[str, Any] = Field(default_factory=dict)
    # Native runtime only; omitted from the JSON for the default runtime so container results are unchanged.
    runtime: Literal["nvidia-container", "metal-native"] = DEFAULT_RUNTIME
    # Unified-memory evidence of a native run (kind "apple-unified"): server phys_footprint and RSS, Metal buffer
    # sizes from the startup log, host available memory, swap and memory pressure. Never VRAM; see NativeLimits.
    memory: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_evidence", "model_evidence", "speed", "quality", "reports", "memory")
    @classmethod
    def immutable_json(cls, value):
        canonical_json(value)
        return freeze_json(value)

    @model_serializer(mode="wrap")
    def _omit_native_defaults(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            if data.get("runtime") == DEFAULT_RUNTIME:
                data.pop("runtime")
            if not data.get("memory"):
                data.pop("memory", None)
        return data

    @model_validator(mode="after")
    def abort_tracks_cleanup(self) -> "ContainerRunResult":
        if self.abort_campaign is not (self.state == "cleanup-uncertain"):
            raise ValueError("abort_campaign is true exactly when cleanup is uncertain")
        if self.state == "completed" and self.failure_reasons:
            raise ValueError("a completed run cannot carry failure reasons")
        return self


class ChildGrant(StrictModel):
    """Typed wall-clock grant handed to the evaluator container; clocks are never shared across the boundary."""
    grant_seconds: int = Field(ge=1)
    artifact_bytes: int = Field(ge=1)
    issued_utc: str = Field(min_length=1)
    issued_host_offset_seconds: float = Field(ge=0)
    watchdog_slack_seconds: int = Field(ge=5, le=300)


class ImageBundle(StrictModel):
    """Output of `llmbench prepare`: every image a candidate may use, pinned by ID, plus build provenance."""
    schema_version: Literal[1] = 1
    prepared_utc: str = Field(min_length=1)
    inference: ImageRef
    evaluator: ImageRef
    worker: ImageRef | None = None
    evaluator_build: dict[str, Any] = Field(default_factory=dict)
    help_sha256: str = Field(pattern=SHA256)
    registry_digest: str = Field(pattern=SHA256)

    @field_validator("evaluator_build")
    @classmethod
    def immutable_build(cls, value):
        canonical_json(value)
        return freeze_json(value)

    @model_validator(mode="after")
    def roles(self) -> "ImageBundle":
        if self.inference.role != "inference" or self.evaluator.role != "evaluator":
            raise ValueError("bundle images must carry their roles")
        if self.worker is not None and self.worker.role != "worker":
            raise ValueError("bundle worker image must have role worker")
        if self.evaluator.entrypoint != EVALUATOR_ENTRYPOINT:
            raise ValueError("evaluator image entrypoint must be python -m llmbench.container_eval")
        if self.help_sha256 != self.inference.help_sha256:
            raise ValueError("bundle help_sha256 must equal the inference image help_sha256")
        return self


class NativeBundle(StrictModel):
    """Output of `llmbench prepare --runtime metal-native`: the pinned native server, what its own `--version`,
    `--help` and `--list-devices` said, the host it was prepared on, and whether the coding sandbox was available.

    The native analogue of `ImageBundle`. There is no evaluator image (a native server is evaluated by the
    host-process evaluator) and the sandbox worker image is optional: without it the coding suites are recorded
    as blocked, never run on the host.
    """

    schema_version: Literal[1] = 1
    prepared_utc: str = Field(min_length=1)
    runtime: Literal["metal-native"] = "metal-native"
    native_server: NativeServerRef
    libraries: dict[str, str] = Field(default_factory=dict)
    worker: ImageRef | None = None
    help_sha256: str = Field(pattern=SHA256)
    registry_digest: str = Field(pattern=SHA256)
    host: dict[str, Any] = Field(default_factory=dict)
    build: dict[str, Any] = Field(default_factory=dict)
    sandbox: dict[str, Any] = Field(default_factory=dict)

    @field_validator("libraries", "host", "build", "sandbox")
    @classmethod
    def immutable_json(cls, value):
        canonical_json(value)
        return freeze_json(value)

    @model_validator(mode="after")
    def coherent(self) -> "NativeBundle":
        if self.help_sha256 != self.native_server.help_sha256:
            raise ValueError("bundle help_sha256 must equal the native server help_sha256")
        if self.worker is not None and self.worker.role != "worker":
            raise ValueError("bundle worker image must have role worker")
        return self


def read_native_bundle(path: str | Path) -> NativeBundle:
    return NativeBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))


def read_run_config(path: str | Path) -> ContainerRunConfig:
    return ContainerRunConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def read_image_bundle(path: str | Path) -> ImageBundle:
    return ImageBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))
