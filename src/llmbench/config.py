"""Strict, versioned experiment contracts shared by the controller and evaluations."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True, validate_default=True)


class FrozenDict(dict):
    """JSON-compatible mapping that rejects ordinary mutation, including nested data."""

    def _deny(self, *args, **kwargs):
        raise TypeError("manifest mappings are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny

    def __deepcopy__(self, memo):
        return self


def freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


class RunMode(str, Enum):
    OFFLINE = "offline"
    DRY_RUN = "dry-run"
    LIVE = "live"


class ModelArtifact(StrictModel):
    model_key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    tokenizer_hash: str = Field(min_length=1)
    template_hash: str = Field(min_length=1)
    path: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def immutable_provenance(cls, value):
        canonical_json(value)  # Reject non-JSON objects and non-finite values before hashing.
        return freeze_json(value)


class BackendSettings(StrictModel):
    engine: Literal["lm-studio", "mock", "llama.cpp"] = "lm-studio"
    runtime_revision: str = Field(min_length=1)
    context_length: int = Field(ge=128, le=4_194_304)
    gpu_offload: float = Field(default=1.0, ge=0, le=1)
    k_cache: str = "f16"
    v_cache: str = "f16"
    kv_on_gpu: bool = True
    flash_attention: bool = True
    mtp: bool | None = None
    parallel: int | None = Field(default=None, ge=1, le=64)
    batch_size: int = Field(default=512, ge=1)
    cpu_threads: int | None = Field(default=None, ge=1)
    reasoning: Literal["on", "off", "auto"] | None = None
    ubatch_size: int | None = Field(default=None, ge=32)


class GenerationSettings(StrictModel):
    temperature: float = Field(default=0, ge=0, le=5)
    top_p: float = Field(default=1, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    seed: int = Field(default=42, ge=0)
    max_output_tokens: int = Field(default=512, ge=1)
    reasoning: str = "model-default"
    tool_emulation: bool = False
    strict_tools: bool = False
    repair_attempts: int = Field(default=0, ge=0, le=10)


class TaskSelection(StrictModel):
    suite: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    task_ids: tuple[str, ...] = Field(min_length=1)
    fixture_seed: int = Field(default=42, ge=0)
    split: Literal["development", "holdout"] = "development"
    # LIVE-013: benchmark options change what is measured (a dropped ``ruler_output_tokens`` truncated every
    # RULER ``vt`` answer at 30 tokens and scored the truncation as a retrieval failure), so the manifest --
    # the controller's record of what a candidate ran -- must carry them, not just the container config.
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_ids")
    @classmethod
    def unique_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("task_ids must be unique")
        return value

    @field_validator("options")
    @classmethod
    def bounded_json_options(cls, value):
        """Typed, bounded, frozen JSON, accepting exactly what ``BenchmarkSelection.options`` accepts.

        The shape is deliberately identical to ``containers.config.BenchmarkSelection.bounded_json_options``
        (tests/test_foundation.py pins the two against each other): which KEYS a benchmark accepts is the
        registry's call, checked at preflight; this validator only guarantees the value cannot smuggle a
        command, a path or an unbounded structure into an adapter.
        """
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


class ExperimentManifest(StrictModel):
    schema_version: Literal[1] = 1
    model: ModelArtifact
    backend: BackendSettings
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    tasks: tuple[TaskSelection, ...] = Field(min_length=1)
    harness_revision: str = Field(min_length=1)
    scorer_revision: str = Field(min_length=1)
    environment_hash: str = "unrecorded"
    tool_schema_hash: str = "none"
    mode: RunMode = RunMode.OFFLINE
    requested_input_tokens: int = Field(default=512, ge=0)
    template_reserve_tokens: int = Field(default=256, ge=0)
    annotations: dict[str, str] = Field(default_factory=dict)

    @field_validator("annotations")
    @classmethod
    def immutable_annotations(cls, value):
        return freeze_json(value)

    @model_validator(mode="after")
    def capacity_fits(self) -> "ExperimentManifest":
        needed = self.requested_input_tokens + self.template_reserve_tokens + self.generation.max_output_tokens
        if needed > self.backend.context_length:
            raise ValueError("input + template reserve + output exceeds context capacity")
        if self.mode == RunMode.LIVE and self.backend.engine == "mock":
            raise ValueError("mock backend cannot be labeled a live model experiment")
        return self

    def fingerprint(self) -> str:
        # Annotation labels are display-only; every measurement-affecting field is hashed, benchmark
        # options included: two candidates differing only in an option are different experiments.
        # An empty options mapping is canonicalised away because "declares no options" and "declares an
        # empty set of options" are the same experiment; that also keeps every manifest recorded before
        # options existed on its original fingerprint, so already-written ledgers still verify.
        payload = self.model_dump(mode="json", exclude={"annotations"})
        for task in payload.get("tasks", ()):
            if not task.get("options"):
                task.pop("options", None)
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class Budgets(StrictModel):
    wall_seconds: int = Field(default=14400, ge=1)
    max_candidates: int = Field(default=12, ge=1)
    max_validation_candidates: int = Field(default=2, ge=1, le=10)
    reserve_validation_seconds: int = Field(default=600, ge=0)
    task_timeout_seconds: int = Field(default=120, ge=1)
    max_download_bytes: int = Field(default=0, ge=0)
    max_artifact_bytes: int = Field(default=1_073_741_824, ge=1)

    @model_validator(mode="after")
    def valid_reserve(self) -> "Budgets":
        if self.reserve_validation_seconds >= self.wall_seconds:
            raise ValueError("validation reserve must be smaller than total wall budget")
        return self


class AcceptancePolicy(StrictModel):
    minimum_tokens_per_second: float = Field(default=50, gt=0)
    speed_repetitions: int = Field(default=5, ge=1)
    maximum_stream_gap_seconds: float = Field(default=2, gt=0)
    # Absolute score FLOORS. They exist to reject a broken configuration (a chat template that breaks tool calls
    # scores near zero), not to rank healthy ones: that is the paired comparison against the session's own baseline
    # (`max_quality_loss`). The earlier 0.80 / 0.95 / 0.95 were set when "tools" meant three hand-written probes a
    # good model aces; against public suites they rejected every candidate ever measured. Each floor is the lowest
    # healthy baseline measured on a 27B-class model minus two standard errors at the default item count:
    #   tools      BFCL 0.76-0.83            -> 0.76 - 2*0.087 (n=24)
    #   retrieval  RULER 1.0, NIAH-8 >= 0.63 -> the same 0.60; eight items cannot support a tighter floor
    #   coding     MBPP+ 0.70-0.75, Aider Polyglot 0.43-0.50, mixed by default -> 0.40
    # Calibrate them to your own model with `llmbench analyze`; nothing has to be re-run.
    minimum_coding_score: float = Field(default=0.40, ge=0, le=1)
    minimum_tool_score: float = Field(default=0.60, ge=0, le=1)
    minimum_retrieval_score: float = Field(default=0.60, ge=0, le=1)
    max_quality_loss: float = Field(default=0.02, ge=0, le=1)
    require_effective_settings_verified: bool = True
    require_actual_context_verified: bool = True
    require_holdout: bool = True
    # No speed ceiling: faster qualifying configurations remain candidates.


class CampaignPolicy(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    mode: RunMode = RunMode.OFFLINE
    search_mode: Literal["optimize-within-budget", "good-enough"] = "optimize-within-budget"
    budgets: Budgets = Field(default_factory=Budgets)
    acceptance: AcceptancePolicy = Field(default_factory=AcceptancePolicy)
    deploy_winner: bool = False
    languages: tuple[str, ...] = ("python", "typescript", "javascript")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def read_manifest(path: str | Path) -> ExperimentManifest:
    return ExperimentManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
