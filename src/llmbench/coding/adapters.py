"""Pinned upstream suite plans and provenance-verified normalized result imports.

An import preserves upstream values; it never invents an official score from the
custom case denominator. Import support is distinct from validated execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from ..config import RunMode, StrictModel, canonical_json
from ..safety import SessionLock


SUITES = {
    "aider-polyglot": ("https://github.com/Aider-AI/aider/blob/main/benchmark/README.md",
                       "import-only: exact exercise mapping and isolated agent split require a pinned integration"),
    "evalplus": ("https://github.com/evalplus/evalplus", "evaluation argv for pre-generated samples; no model load"),
    "bfcl": ("https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
             "import-only: registered native FC handler must be verified; generic local handler is insufficient"),
    "tool-eval-bench": ("https://github.com/SeraphimSerapis/tool-eval-bench", "native endpoint command plan"),
    "ruler": ("https://github.com/NVIDIA/RULER",
              "import-only: upstream shell workflow/tokenizer/server configuration needs pinned integration"),
}


class UpstreamPin(StrictModel):
    suite: Literal["aider-polyglot", "evalplus", "bfcl", "tool-eval-bench", "ruler"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    case_ids: tuple[str, ...] = Field(min_length=1)
    languages: tuple[str, ...] = Field(min_length=1)
    subset_label: str = Field(min_length=1)
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def case_ids_unique(self) -> "UpstreamPin":
        if any(not value for value in self.case_ids) or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Case IDs must be nonempty and unique")
        return self


class CaseResult(StrictModel):
    case_id: str = Field(min_length=1)
    status: Literal["passed", "partial", "failed", "timeout", "environment-error", "unsupported", "cancelled"]
    eligible_for_upstream_score: bool
    first_attempt_success: bool = False
    repairs: int = Field(default=0, ge=0)
    duration_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def consistent_attempt(self) -> "CaseResult":
        if self.first_attempt_success and (self.status != "passed" or self.repairs != 0):
            raise ValueError("First-attempt success requires passed status with no repairs")
        return self


class ResultEnvelope(StrictModel):
    schema_version: Literal[1] = 1
    pin: UpstreamPin
    run_id: str = Field(min_length=1)
    model_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_artifact: str = Field(min_length=1)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    official_metrics_json: str = Field(min_length=2)
    cases: tuple[CaseResult, ...] = Field(min_length=1)
    source_format: str = Field(min_length=1)

    @model_validator(mode="after")
    def exact_denominator(self) -> "ResultEnvelope":
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids) or set(ids) != set(self.pin.case_ids):
            raise ValueError("Every planned case must have exactly one result, including failures")
        metrics = json.loads(self.official_metrics_json)
        if not isinstance(metrics, dict):
            raise ValueError("Official metrics must preserve a JSON object")
        canonical_json(metrics)  # Reject NaN/Infinity rather than importing impossible scores.
        return self


@dataclass(frozen=True)
class ImportedResult:
    envelope: ResultEnvelope
    artifact_verified: bool
    attempted: int
    completed: int
    successful: int
    all_required_success_rate: float
    upstream_eligibility_ids: tuple[str, ...]
    official_metrics_json: str
    normalization_verified: bool = False
    interpretation: str = "Verified artifact import; upstream execution and leaderboard comparability not certified"


def import_result(envelope: ResultEnvelope, *, artifact_root: str | Path,
                  max_artifact_bytes: int = 16_777_216) -> ImportedResult:
    root = Path(artifact_root).resolve(strict=True)
    path = (root / envelope.raw_artifact).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Raw artifact must be a file within artifact_root")
    if max_artifact_bytes <= 0 or path.stat().st_size > max_artifact_bytes:
        raise ValueError("Raw artifact exceeds import size budget")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != envelope.raw_sha256:
        raise ValueError("Raw artifact hash mismatch")
    completed = sum(case.status in {"passed", "partial", "failed"} for case in envelope.cases)
    successful = sum(case.status == "passed" for case in envelope.cases)
    return ImportedResult(envelope, True, len(envelope.cases), completed, successful,
                          successful / len(envelope.cases),
                          tuple(case.case_id for case in envelope.cases if case.eligible_for_upstream_score),
                          envelope.official_metrics_json)


@dataclass(frozen=True)
class UpstreamCommand:
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    requires_inference: bool
    requires_isolation: bool
    pin: UpstreamPin
    verification: str = "Documentation-verified argv only; pinned installation/live execution must be verified"


def evaluation_command(pin: UpstreamPin, *, dataset: str = "humaneval",
                       samples: str = "/workspace/samples.jsonl", base_url: str | None = None,
                       model: str | None = None, seed: int = 42) -> UpstreamCommand:
    if pin.suite == "evalplus":
        if dataset not in {"humaneval", "mbpp"} or samples != "/workspace/samples.jsonl":
            raise ValueError("Use a supported dataset and the fixed isolated sample path")
        return UpstreamCommand(("evalplus.evaluate", "--dataset", dataset, "--samples", samples),
                               (), False, True, pin)
    if pin.suite == "tool-eval-bench":
        parsed = urlsplit(base_url or "")
        if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("An explicit local HTTP origin is required; endpoint autodiscovery is forbidden")
        if not model or not isinstance(model, str) or "\x00" in model or type(seed) is not int or seed < 0:
            raise ValueError("An explicit model identifier and nonnegative integer seed are required")
        if any(not case.startswith("TC-") or not case[3:].isdigit() for case in pin.case_ids):
            raise ValueError("Tool-eval command requires exact TC-number scenario IDs")
        return UpstreamCommand(("tool-eval-bench", "run", "--base-url", base_url or "", "--seed", str(seed),
                                "--scenarios", *pin.case_ids, "--json-file", "result.json"),
                               (("TOOL_EVAL_MODEL", model),), True, False, pin)
    raise NotImplementedError(SUITES[pin.suite][1])


def authorize_command(command: UpstreamCommand, *, session_lock: SessionLock, mode: RunMode) -> None:
    """Called immediately before an external runner; plans/imports require no permission."""
    # Infer required capabilities from the supported suite, never caller-supplied flags alone.
    if command.pin.suite not in {"tool-eval-bench", "evalplus"}:
        raise NotImplementedError("This suite does not yet have an executable adapter")
    if command.requires_inference or command.pin.suite == "tool-eval-bench":
        session_lock.check("inference", mode)
    if command.requires_isolation or command.pin.suite == "evalplus":
        session_lock.check("container", mode)
