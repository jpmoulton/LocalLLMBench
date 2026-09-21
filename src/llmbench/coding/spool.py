"""Strict spool contracts between the evaluator and the host coding broker. Import performs no I/O.

The evaluator (container or in-process) writes ``CodingJobRequest`` files into ``spool/requests`` and reads
``CodingJobResult`` files from ``spool/results``. The host trusts nothing in the spool: every name, link
state, size, byte, JSON value and identity is validated before a patch is staged for a worker container.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..config import StrictModel, canonical_json, freeze_json
from ..containers.config import NAME, SHA256

REQUEST_NAME = r"^[0-9a-f]{32}\.json$"
HEX32 = r"^[0-9a-f]{32}$"
FIXTURE_REVISION = "private-coding-v1"
NAMESPACE_NAME = "namespace.json"
MAX_PATCH_FILES = 32
MAX_PATCH_BYTES_HARD = 8_388_608  # BrokerSettings.max_patch_bytes upper bound; the broker enforces its own cap
RESULT_STATUSES = ("completed", "rejected", "interrupted", "cancelled", "environment-error", "cleanup-unverified")
_PUBLISH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.json")
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _safe_source_path(name: str) -> bool:
    pure = PurePosixPath(name)
    return (type(name) is str and 0 < len(name) <= 255 and not pure.is_absolute() and str(pure) == name
            and not any(part in {"", ".", ".."} for part in pure.parts)
            and not any(char in name for char in ("\\", ":", "\x00"))
            and not any(ord(char) < 32 for char in name))


def _text_ok(value: str) -> bool:
    if type(value) is not str or "\x00" in value:
        return False
    try:
        value.encode("utf-8")  # lone surrogates cannot be encoded
    except UnicodeEncodeError:
        return False
    return True


class CodingJobRequest(StrictModel):
    schema_version: Literal[1] = 1
    session_id: str = Field(pattern=NAME)
    attempt_id: str = Field(pattern=HEX32)
    request_id: str = Field(pattern=HEX32)
    fixture_id: str = Field(min_length=1, max_length=128)
    # The private fixtures' revision, or a pinned public benchmark's (e.g. ``evalplus/mbpp-plus@v0.2.0``). The
    # broker still refuses any revision that is not the one its own host-side fixture declares.
    fixture_revision: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._@/+-]*$")
    fixture_hash: str = Field(pattern=SHA256)
    attempt_index: Literal[1, 2] = 1  # 1 = first attempt, 2 = repaired
    patch: dict[str, str]
    submitted_utc: str = Field(min_length=1, max_length=64)

    @field_validator("patch")
    @classmethod
    def safe_patch(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or len(value) > MAX_PATCH_FILES:
            raise ValueError(f"patch must contain 1..{MAX_PATCH_FILES} files")
        total = 0
        for name, content in value.items():
            if not _safe_source_path(name):
                raise ValueError("patch file names must be safe relative POSIX paths")
            if not _text_ok(content):
                raise ValueError("patch content must be UTF-8 text without NUL or lone surrogates")
            total += len(content.encode("utf-8"))
        if total > MAX_PATCH_BYTES_HARD:
            raise ValueError("patch exceeds the hard byte cap")
        return dict(value)

    @field_validator("fixture_id", "submitted_utc")
    @classmethod
    def plain_text(cls, value: str) -> str:
        if not _text_ok(value) or any(ord(char) < 32 for char in value):
            raise ValueError("control characters are not allowed")
        return value

    def patch_bytes(self) -> int:
        return sum(len(content.encode("utf-8")) for content in self.patch.values())

    def content_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"submitted_utc"})
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class CodingJobResult(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=HEX32)
    request_sha256: str = Field(pattern=SHA256)
    status: Literal["completed", "rejected", "interrupted", "cancelled", "environment-error", "cleanup-unverified"]
    sample: dict[str, Any] | None = None
    failure_reason: str | None = Field(default=None, max_length=2000)
    trace_sha256: str | None = Field(default=None, pattern=SHA256)
    cleanup_confirmed: bool
    abort_campaign: bool
    finished_utc: str = Field(min_length=1, max_length=64)

    @field_validator("sample")
    @classmethod
    def immutable_sample(cls, value):
        if value is None:
            return None
        canonical_json(value)  # JSON-only, finite numbers
        return freeze_json(value)

    @model_validator(mode="after")
    def coherent(self) -> "CodingJobResult":
        if (self.status == "completed") != (self.sample is not None):
            raise ValueError("a completed result carries exactly one sample; other statuses carry none")
        if self.status != "completed" and not self.failure_reason:
            raise ValueError("a non-completed result must state its failure_reason")
        if self.abort_campaign != (self.status == "cleanup-unverified"):
            raise ValueError("abort_campaign is true exactly when cleanup is unverified")
        if self.status == "cleanup-unverified" and self.cleanup_confirmed:
            raise ValueError("cleanup-unverified cannot confirm cleanup")
        return self


class SpoolIntegrityError(ValueError):
    """A well-formed result does not belong to the submitted request content."""


class NotRegularFile(ValueError):
    """The path is missing, a link, a reparse point, a directory or hard-linked: it is ignored."""


class Oversize(ValueError):
    """The regular file exceeds the byte cap; nothing beyond the cap was read."""


def request_name(request_id: str) -> str:
    if not re.fullmatch(HEX32, request_id or ""):
        raise ValueError("request_id must be 32 lowercase hex characters")
    return request_id + ".json"


def publish_atomic(directory: str | Path, name: str, payload: bytes) -> Path:
    """Exclusive, durable publication: temp file in the same directory, fsync, then rename into place.

    An existing target is never replaced. On any failure the temporary file is removed again.
    """
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    if not _PUBLISH_NAME.fullmatch(name or ""):
        raise ValueError("spool file names are plain .json basenames")
    base = Path(directory)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("spool directory must be an existing real directory")
    target = base / name
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(str(target))
    temporary = base / f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # the original failure is the one to report
        raise
    if os.name != "nt":  # directory entries are durable only after the directory itself is synced
        fd = os.open(base, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return target


def _check_regular(st: os.stat_result, max_bytes: int, *, label: str) -> None:
    if not stat.S_ISREG(st.st_mode) or getattr(st, "st_file_attributes", 0) & _REPARSE:
        raise NotRegularFile(f"{label}: not a regular file (link, reparse point or directory)")
    if st.st_nlink != 1:
        raise NotRegularFile(f"{label}: hard-linked file ({st.st_nlink} links)")
    if st.st_size > max_bytes:
        raise Oversize(f"{label}: {st.st_size} bytes exceed the {max_bytes}-byte cap")


def read_bounded(path: str | Path, max_bytes: int) -> bytes:
    """Read one regular, unlinked, size-capped file without following links; never reads more than the cap.

    Order: lstat (regular, no reparse point, nlink == 1, size) -> open without following -> fstat identity and
    size recheck -> bounded read. ``FileNotFoundError`` propagates; other failures are ``NotRegularFile`` or
    ``Oversize`` (both ``ValueError``).
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    target = Path(path)
    before = os.lstat(target)  # never follows a symlink; junctions/reparse points show their attributes
    _check_regular(before, max_bytes, label="lstat")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NOINHERIT", 0)
    descriptor = _open_file(target, flags)
    try:
        after = os.fstat(descriptor)
        _check_regular(after, max_bytes, label="open")
        if (before.st_ino, before.st_dev) != (after.st_ino, after.st_dev) or before.st_size != after.st_size:
            raise NotRegularFile("file changed between inspection and open")
        chunks, total = [], 0
        while True:
            block = os.read(descriptor, min(65536, max_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise Oversize(f"read exceeded the {max_bytes}-byte cap")
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


_open_file = os.open  # indirection so tests can prove that an oversize file is never opened


def result_bytes(result: CodingJobResult) -> bytes:
    return (canonical_json(result.model_dump(mode="json")) + "\n").encode("utf-8")


def request_bytes(request: CodingJobRequest) -> bytes:
    return (canonical_json(request.model_dump(mode="json")) + "\n").encode("utf-8")
