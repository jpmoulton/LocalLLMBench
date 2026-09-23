"""Persistent session lock. No CLI flag silently overrides a model-operation prohibition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import RunMode


class OperationForbidden(RuntimeError):
    pass


PERMISSIONS = ("allow_model_operations", "allow_inference", "allow_container_execution", "allow_native_execution")


@dataclass(frozen=True)
class SessionLock:
    allow_model_operations: bool = False
    allow_inference: bool = False
    allow_container_execution: bool = False
    # Launching a pinned llama-server executable directly on this host (the metal-native runtime), outside any
    # container. Separate from container execution on purpose: a policy written for the NVIDIA containers never
    # silently authorizes running a host binary. Absent from a policy file means False.
    allow_native_execution: bool = False
    reason: str = "Live operations have not been authorized in this session."

    def __post_init__(self) -> None:
        for name in PERMISSIONS:
            if type(getattr(self, name)) is not bool:
                raise OperationForbidden(f"{name} must be a boolean")
        if not isinstance(self.reason, str):
            raise OperationForbidden("Session lock reason must be a string")

    @classmethod
    def read(cls, path: str | Path) -> "SessionLock":
        lock_path = Path(path)
        if not lock_path.exists():
            return cls()
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
        allowed = {*PERMISSIONS, "reason"}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise OperationForbidden("Invalid session lock; refusing live operations")
        for key in allowed - {"reason"}:
            if key in raw and type(raw[key]) is not bool:
                raise OperationForbidden(f"{key} must be a boolean")
        if "reason" in raw and not isinstance(raw["reason"], str):
            raise OperationForbidden("Session lock reason must be a string")
        return cls(**raw)

    def check(self, operation: str, mode: RunMode | str) -> None:
        if mode != RunMode.LIVE:
            raise OperationForbidden(f"{operation} requires live mode; current mode is {mode}")
        mapping = {
            "load": self.allow_model_operations,
            "unload": self.allow_model_operations,
            "restore": self.allow_model_operations,
            "configure": self.allow_model_operations,
            "inference": self.allow_inference,
            "container": self.allow_container_execution,
            "native": self.allow_native_execution,
        }
        if not mapping.get(operation, False):
            raise OperationForbidden(f"{operation} forbidden: {self.reason}")

    def to_json(self) -> dict:
        """The policy as a runtime-policy.json object. `allow_native_execution` is written only when granted, so
        the policy handed to an evaluator container built from an older wheel is exactly what it always was."""
        data = {name: getattr(self, name) for name in PERMISSIONS}
        if not data["allow_native_execution"]:
            data.pop("allow_native_execution")
        return {**data, "reason": self.reason}
