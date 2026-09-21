"""Pure, shell-free plans for the narrowly scoped trusted coding worker build.

Nothing in this module calls Docker, resolves tags, downloads files, or changes
session permissions. The caller must authorize and execute a concrete plan.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


_BASE = r"(?:docker\.io/library/)?node:24-bookworm-slim@sha256:[0-9a-f]{64}"


def worker_image_build_plan(*, base_image: str, artifact_dir: str | Path) -> dict:
    """Return an unexecuted build manifest requiring an exact official base digest.

    Runtime uses the immutable image ID from --iidfile; no registry publishing is
    necessary. A mutable staging tag is never passed to the scoring worker.
    """
    if type(base_image) is not str or not re.fullmatch(_BASE, base_image):
        raise ValueError("An official node:24-bookworm-slim@sha256 digest is required")
    artifacts = Path(artifact_dir).resolve()
    context = Path(__file__).with_name("worker_image").resolve(strict=True)
    files = {name: hashlib.sha256((context / name).read_bytes()).hexdigest()
             for name in ("Dockerfile", ".dockerignore", "polyglot-js/package.json")}
    iidfile = artifacts / "worker-image.id"
    return {
        "schema_version": 1,
        "status": "planned-not-executed",
        "purpose": "Trusted Python/Node/TypeScript toolchain for the isolated coding fixtures and public coding suites",
        "base_image": base_image,
        "platform": "linux/amd64",
        "typescript_version": "5.9.3",
        "context": str(context),
        "source_sha256": files,
        "build_argv": ["docker", "build", "--platform=linux/amd64", "--pull=false", "--no-cache",
                       "--build-arg", "BASE_IMAGE=" + base_image, "--iidfile", str(iidfile),
                       "--tag", "llmbench-worker:staged", "--file", str(context / "Dockerfile"), str(context)],
        "build_requires_network": True,
        "build_downloads": ["Exact digest-pinned official Node base image if absent",
                            "Debian python3, python3-numpy and python3-pytest", "npm typescript@5.9.3",
                            "the pinned jest/babel toolchain in polyglot-js/package.json"],
        "iidfile": str(iidfile),
        "final_image_id": None,
        "final_repository_digest": None,
        "runtime": {"image_source": "verified iidfile sha256 image ID", "pull": "never",
                    "network": "none", "gpu_devices": [], "read_only": True,
                    "user": "10001:10001", "memory_mib": 512, "memory_swap_mib": 512,
                    "cpus": 1.0, "pids": 64, "temporary_mib": 128,
                    "candidate_mount": "dedicated read-only source and input-only driver directory",
                    "case_timeout_seconds": 30, "result_keepalive_seconds": 30},
        "required_post_build_evidence": ["Exact build stdout/stderr and exit status",
                                         "worker-image.id", "docker image inspect <exact image ID> JSON",
                                         "/opt/llmbench/* toolchain/package version files",
                                         "Reference and broken-fixture live acceptance results",
                                         "Owned-container absence verification"],
        "rebuild_guarantee": "Final image content is pinned; Debian resolution can change across rebuilds",
    }


def verified_local_image_id(iidfile: str | Path) -> str:
    """Read only a bounded Docker build output ID, without querying a daemon."""
    path = Path(iidfile)
    with path.open("rb") as stream:
        data = stream.read(80)
    if len(data) == 80:
        raise ValueError("Image ID output exceeds its expected length")
    try:
        value = data.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Image ID must be ASCII") from exc
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Build output must be one complete sha256 image ID")
    return value
