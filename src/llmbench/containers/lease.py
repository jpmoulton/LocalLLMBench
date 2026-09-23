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


# Holder fields `acquire` writes; `annotate` may add facts beside them but never rewrite who holds the lease.
HOLDER_FIELDS = frozenset({"pid", "created", "owner"})


class GpuLease:
    def __init__(self, path: str | Path | None = None, *, owner: str = "container-run") -> None:
        self.path = Path(path) if path is not None else default_lease_path()
        self.owner = owner
        self.held = False
        self._holder: dict | None = None
        self._file: tuple[int, int] | None = None  # (st_dev, st_ino) of the file this object created
        self._written = b""  # what this object last wrote into it

    def acquire(self) -> "GpuLease":
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            from ..locks import describe_lock
            raise LeaseHeld(f"GPU lease is held: {describe_lock(self.path)}") from exc
        holder = {"pid": os.getpid(), "created": utc_now(), "owner": self.owner}
        data = canonical_json(holder).encode("utf-8")
        try:
            info = os.fstat(descriptor)
            os.write(descriptor, data)
        finally:
            os.close(descriptor)
        self.held, self._holder, self._file, self._written = True, holder, (info.st_dev, info.st_ino), data
        return self

    def annotate(self, **facts) -> None:
        """Add facts to the holder record of the lease this object holds, e.g. the llama-server pid once it exists.

        A lease left behind by a killed run is described to a human who must decide whether anything it started
        still runs (`locks.describe_lock`); a native run's leftover is a host process, so its pid is the most useful
        thing the file can say. Only the file this object created is ever written: opened without following a link
        and never created, it must be the same file (device and inode) and still hold exactly what this object last
        wrote into it -- a reused inode cannot also carry this run's pid, timestamp and attempt id -- so a lease
        someone deleted and another run re-took is left alone (`LeaseHeld`). It is rewritten in place: the path never
        disappears, so exclusion does not lapse.
        """
        if not self.held or self._holder is None:
            raise LeaseHeld("only a held lease can be annotated")
        if HOLDER_FIELDS & set(facts):
            raise ValueError(f"annotations cannot replace {', '.join(sorted(HOLDER_FIELDS & set(facts)))}")
        holder = {**self._holder, **facts}
        data = canonical_json(holder).encode("utf-8")
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(self.path, flags)
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != self._file or os.read(descriptor, len(self._written) + 1) != self._written:
                raise LeaseHeld(f"{self.path} is no longer the lease this run created; left untouched")
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.write(descriptor, data) != len(data):
                raise OSError(f"short write annotating {self.path}")
            os.ftruncate(descriptor, len(data))
        finally:
            os.close(descriptor)
        self._holder, self._written = holder, data

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
