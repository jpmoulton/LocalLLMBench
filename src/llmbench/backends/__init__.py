"""Inference backend contracts and the llama.cpp server adapter. Live operations need an explicit grant."""

from .base import (
    Backend, BackendCapabilities, BackendError, BackendSnapshot, LivePermission,
    LoadedModel, LoadRequest, OperationDenied, OwnershipError, StreamEvent,
    UnsupportedSetting, VerificationError, VerifiedModel,
)

__all__ = [
    "Backend", "BackendCapabilities", "BackendError", "BackendSnapshot", "LivePermission",
    "LoadedModel", "LoadRequest", "OperationDenied", "OwnershipError", "StreamEvent",
    "UnsupportedSetting", "VerificationError", "VerifiedModel",
]
