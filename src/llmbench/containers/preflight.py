"""Admission checks: model bytes/identity, exact image identity, GPU headroom and path lengths."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path

from .config import ImageRef, ModelAsset

MAX_RUN_DIR_CHARS = 180
_REPARSE_POINT = 0x400


class PreflightError(ValueError):
    pass


class PreflightTimeout(PreflightError):
    pass


def _identity(path: Path) -> dict:
    info = os.lstat(path)
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
            or not stat.S_ISREG(info.st_mode)):
        raise PreflightError("model asset must be a regular file, not a link or reparse point")
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "file_id": info.st_ino, "device": info.st_dev,
            "nlink": info.st_nlink}


def _bounded_hash(path: Path, deadline: float | None, clock) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            if deadline is not None and clock() > deadline:
                raise PreflightTimeout("model hashing exceeded its stage bound")
    return digest.hexdigest()


def verify_model_asset(asset: ModelAsset, *, hasher=None, deadline: float | None = None,
                       clock=time.monotonic) -> dict:
    """Hash the declared file stably. hasher=None uses a deadline-aware SHA256 equal to config.hash_file."""
    path = Path(asset.host_path)
    try:
        for parent in path.parents:
            if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                raise PreflightError("model asset path crosses a linked directory")
        before = _identity(path)
        if before["nlink"] != 1:
            raise PreflightError("model asset has additional hard links")
        if before["size"] != asset.size_bytes:
            raise PreflightError(f"model asset size {before['size']} does not match declared {asset.size_bytes}")
        started = clock()
        digest = _bounded_hash(path, deadline, clock) if hasher is None else hasher(path)
        elapsed = clock() - started
        if deadline is not None and clock() > deadline:
            raise PreflightTimeout("model hashing exceeded its stage bound")
        if _identity(path) != before:
            raise PreflightError("model asset changed while it was being hashed")
    except OSError as exc:
        raise PreflightError(f"model asset is not readable: {exc}") from exc
    if digest != asset.sha256:
        raise PreflightError("model asset SHA256 does not match the declared value")
    return {"path": str(path), **before, "sha256": digest, "hash_seconds": elapsed}


def reverify_identity(asset: ModelAsset, evidence: dict) -> dict:
    """Cheap post-run check that the mounted file was not replaced; the bytes are not rehashed."""
    try:
        after = _identity(Path(asset.host_path))
    except (OSError, PreflightError) as exc:
        return {"unchanged": False, "error": str(exc)}
    return {"unchanged": all(after[key] == evidence.get(key) for key in after), **after}


def verify_image(executor, ref: ImageRef, *, timeout_seconds: int = 30) -> dict:
    result = executor.run(("docker", "image", "inspect", "--format", "{{json .}}", ref.reference),
                          timeout_seconds=timeout_seconds, max_output_bytes=1_048_576)
    if result.status != "completed" or result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:500]
        raise PreflightError(f"image {ref.reference} is not available locally ({result.status}): {detail}")
    try:
        raw = json.loads(result.stdout)
        config = raw.get("Config") or {}
        evidence = {"id": raw["Id"], "repo_digests": list(raw.get("RepoDigests") or []),
                    "platform": f"{raw.get('Os')}/{raw.get('Architecture')}",
                    "entrypoint": list(config.get("Entrypoint") or []),
                    "environment_names": sorted(item.split("=", 1)[0] for item in config.get("Env") or []),
                    "created": raw.get("Created"), "size": raw.get("Size")}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PreflightError(f"unreadable image inspection: {exc}") from exc
    problems = []
    if evidence["id"] != ref.image_id:
        problems.append(f"image Id {evidence['id']} does not equal configured {ref.image_id}")
    if ref.reference != evidence["id"] and ref.reference not in evidence["repo_digests"]:
        problems.append("configured digest reference is not among the image RepoDigests")
    if evidence["platform"] != ref.platform:
        problems.append(f"image platform {evidence['platform']} is not {ref.platform}")
    if tuple(evidence["entrypoint"]) != ref.entrypoint:
        problems.append(f"image entrypoint {evidence['entrypoint']} does not equal the configured entrypoint")
    ambient = [name for name in evidence["environment_names"]
               if name.startswith("LLAMA_ARG_") and name != "LLAMA_ARG_HOST"]
    if ambient:
        problems.append("image environment would set server arguments: " + ", ".join(ambient))
    if problems:
        raise PreflightError("; ".join(problems))
    return evidence


def check_gpu_headroom(sample: dict, limit_mib: int, *, device_id: str = "0") -> dict:
    """A competing GPU workload is a conflict to report, never a process to stop."""
    gpus = [row for row in sample.get("gpus") or [] if row.get("index") == float(device_id)]
    if sample.get("gpu_error") or len(gpus) != 1 or gpus[0].get("memory_used_mib") is None:
        raise PreflightError(f"GPU {device_id} telemetry is unavailable: {sample.get('gpu_error') or 'no such device'}")
    used, total = gpus[0]["memory_used_mib"], gpus[0].get("memory_total_mib")
    if used > limit_mib:
        raise PreflightError(f"GPU {device_id} already has {used:.0f} MiB in use; limit for foreign load is {limit_mib}")
    return {"device": device_id, "name": gpus[0].get("name"), "memory_used_mib": used, "memory_total_mib": total,
            "limit_mib": limit_mib}


def check_path_lengths(run_dir: str | Path, *, limit: int = MAX_RUN_DIR_CHARS) -> None:
    length = len(str(Path(run_dir).resolve()))
    if length > limit:
        raise PreflightError(f"run directory path has {length} characters; the limit is {limit}")
