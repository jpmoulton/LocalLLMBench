"""Exclusive GPU lease: one GPU workload at a time on this machine. A stale lease is never broken automatically."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..config import canonical_json
from ..store import utc_now


class LeaseHeld(RuntimeError):
    pass


def default_lease_path() -> Path:
    # Identical to controller.run_campaign's live resource lock so both execution paths exclude each other.
    return Path(tempfile.gettempdir()) / "llmbench-gpu-resource.lock"


class GpuLease:
    def __init__(self, path: str | Path | None = None, *, owner: str = "container-run") -> None:
        self.path = Path(path) if path is not None else default_lease_path()
        self.owner = owner
        self.held = False

    def acquire(self) -> "GpuLease":
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            from ..locks import describe_lock
            raise LeaseHeld(f"GPU lease is held: {describe_lock(self.path)}") from exc
        try:
            os.write(descriptor, canonical_json({"pid": os.getpid(), "created": utc_now(),
                                                 "owner": self.owner}).encode("utf-8"))
        finally:
            os.close(descriptor)
        self.held = True
        return self

    def release(self) -> None:
        if self.held:  # Only the lease this object created; never another owner's file.
            self.path.unlink(missing_ok=True)
            self.held = False


@contextmanager
def gpu_lease(path: str | Path | None = None, *, owner: str = "container-run"):
    lease = GpuLease(path, owner=owner).acquire()
    try:
        yield lease
    finally:
        lease.release()
