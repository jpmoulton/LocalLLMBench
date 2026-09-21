"""Small, dependency-free contracts for inference backends.

Importing this module never connects to an inference server. Model operations
require separate, explicit capabilities supplied by the campaign controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol


class BackendError(RuntimeError):
    """A backend failed without producing a valid benchmark result."""


class OperationDenied(BackendError):
    """An operation is outside the current live-operation authorization."""


class UnsupportedSetting(BackendError):
    """A requested setting cannot be applied and verified by this adapter."""


class OwnershipError(BackendError):
    """An operation would affect a model not owned by this adapter."""


class VerificationError(BackendError):
    def __init__(self, message: str, *, mismatches: Mapping[str, Any] | None = None,
                 instance_id: str | None = None) -> None:
        super().__init__(message)
        self.mismatches = dict(mismatches or {})
        self.instance_id = instance_id


@dataclass(frozen=True)
class LivePermission:
    allow_model_load: bool = False
    allow_inference: bool = False
    allow_model_unload: bool = False

    def __post_init__(self) -> None:
        for field_name in ("allow_model_load", "allow_inference", "allow_model_unload"):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")

    def require(self, operation: str) -> None:
        permissions = {
            "load": self.allow_model_load,
            "inference": self.allow_inference,
            "unload": self.allow_model_unload,
            "restore": self.allow_model_unload,
        }
        if operation not in permissions or not permissions[operation]:
            raise OperationDenied(f"Live {operation} is disabled by the operation policy")


@dataclass(frozen=True)
class LoadedModel:
    instance_id: str
    model_key: str
    config: dict[str, Any]
    owned: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendSnapshot:
    backend: str
    models: tuple[LoadedModel, ...]
    captured_at: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    available: bool
    api: str
    load_settings: tuple[str, ...]
    observable_settings: tuple[str, ...]
    notes: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadRequest:
    model_key: str
    settings: dict[str, Any] = field(default_factory=dict)
    instance_identifier: str | None = None


@dataclass(frozen=True)
class VerifiedModel:
    model: LoadedModel
    requested: dict[str, Any]
    effective: dict[str, Any]
    evidence: str


@dataclass(frozen=True)
class StreamEvent:
    request_id: str
    monotonic_seconds: float
    event: str
    data: dict[str, Any] | str


class Backend(Protocol):
    def probe(self) -> BackendCapabilities: ...
    def snapshot(self) -> BackendSnapshot: ...
    def load(self, request: LoadRequest) -> VerifiedModel: ...
    def effective_config(self, instance_id: str) -> dict[str, Any]: ...
    def stream(self, instance_id: str, payload: Mapping[str, Any], *,
               request_id: str | None = None) -> Iterator[StreamEvent]: ...
    def cancel(self, request_id: str) -> bool: ...
    def unload(self, instance_id: str) -> None: ...
    def restore(self, snapshot: BackendSnapshot) -> BackendSnapshot: ...
