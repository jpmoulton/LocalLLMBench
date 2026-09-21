"""Evaluator image preparation: pure build plans and context staging, plus scripted Docker preparation.

Nothing here resolves tags or runs Docker by itself. `prepare_images` only issues the preparation-vocabulary
argv (pull by digest, help/version capture, build with --iidfile, self-check run) through the injected bounded
executor and refuses to write a bundle unless every check passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ..config import canonical_json
from ..store import utc_now
from .capabilities import help_sha256, parse_server_help, parse_version
from .config import DIGEST_REF, EVALUATOR_ENTRYPOINT, IMAGE_ID, ImageBundle, ImageRef
from .executor import BUILD_SHAPE, SELF_CHECK_SHAPE

EVALUATOR_BASE = r"(?:docker\.io/library/)?python:3\.12-slim-bookworm@sha256:[0-9a-f]{64}"
CONTEXT_DIR = "evaluator-context"
BUNDLE_NAME = "image-bundle.json"
LOCK_NAME = "requirements.linux.lock"
_HASHED_LINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,_-]+\])?==[^\s]+ --hash=sha256:[0-9a-f]{64}$")
_MAX_TEXT = 4 * 1024 * 1024


def _regular(path: Path, what: str) -> Path:
    """lstat the path exactly as given (resolving first would follow the very link this rejects)."""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"{what} must be a regular file, not a link or reparse point: {path}")
    return path.resolve()


def pull_reference(reference: str) -> str:
    """`name[:tag]@sha256:<digest>` -> `name@sha256:<digest>`; the digest alone identifies what is pulled."""
    name, digest = reference.rsplit("@", 1)
    return name.split(":", 1)[0] + "@" + digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_dir() -> Path:
    return Path(__file__).with_name("evaluator_image").resolve(strict=True)


def lock_is_hash_pinned(text: str) -> tuple[bool, list[str]]:
    """Every requirement line carries exactly one sha256 hash; comments and blanks are the only other lines."""
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not _HASHED_LINE.match(stripped):
            problems.append(f"line {number}: not a `name==version --hash=sha256:<hex>` pin")
    return not problems, problems


def evaluator_image_build_plan(*, base_image: str, artifact_dir: str | Path, wheel_path: str | Path,
                               lock_path: str | Path) -> dict:
    """An unexecuted build manifest. The base must be the official digest; wheel and lock are hashed here."""
    if type(base_image) is not str or not re.fullmatch(EVALUATOR_BASE, base_image):
        raise ValueError("An official python:3.12-slim-bookworm@sha256 digest is required")
    artifacts = Path(artifact_dir).resolve()
    wheel, lock = _regular(Path(wheel_path), "wheel"), _regular(Path(lock_path), "lock")
    if wheel.suffix != ".whl" or not re.fullmatch(r"localllmbench-[0-9A-Za-z.+!_-]+\.whl", wheel.name):
        raise ValueError("the harness wheel must be a localllmbench-*.whl built from this tree")
    pinned, problems = lock_is_hash_pinned(lock.read_text(encoding="utf-8"))
    if not pinned:
        raise ValueError("the Linux lock is not fully hash-pinned: " + "; ".join(problems[:5]))
    source = source_dir()
    files = {name: _sha256(_regular(source / name, name)) for name in ("Dockerfile", ".dockerignore")}
    files[LOCK_NAME], files[wheel.name] = _sha256(lock), _sha256(wheel)
    context, iidfile = artifacts / CONTEXT_DIR, artifacts / "evaluator-image.id"
    build_argv = ["docker", *BUILD_SHAPE[:5], str(iidfile), "--file", str(context / "Dockerfile"), str(context)]
    return {
        "schema_version": 1, "status": "planned-not-executed",
        "purpose": "Inspect-driven evaluator for one attached llama.cpp candidate; no Docker, GPU or model access",
        "base_image": base_image, "platform": "linux/amd64", "python": "3.12",
        "source_dir": str(source), "context": str(context), "wheel": str(wheel), "lock": str(lock),
        "source_sha256": files, "build_argv": build_argv, "iidfile": str(iidfile),
        # Honest provenance: the lock install has no --no-index, so every wheel it names is downloaded from PyPI
        # during `docker build` and verified against the recorded sha256. The base is pulled by digest before.
        "build_requires_network": True,
        "build_downloads": ["Every wheel named in requirements.linux.lock, from PyPI, verified against the lock's "
                            "sha256 hashes (pip install --require-hashes --no-deps)"],
        "base_image_pulled_before_build": True,  # `docker pull name@digest` at prepare time; build uses --pull=false
        "install": ["pip install --require-hashes --no-deps -r requirements.linux.lock",
                    "pip install --no-index --no-deps <wheel>"],
        "runtime": {"image_source": "verified iidfile sha256 image ID", "pull": "never", "network": "bench only",
                    "gpu_devices": [], "read_only": True, "user": "10001:10001", "tmpfs": "/tmp",
                    "entrypoint": list(EVALUATOR_ENTRYPOINT), "cmd": None},
        "required_post_build_evidence": ["evaluator-image.id", "docker image inspect <exact image ID> JSON",
                                         "self-check JSON with ok=true and the host registry digest",
                                         "/opt/llmbench/{pip-freeze.txt,python-version.txt}"],
    }


def build_evaluator_context(staging_dir: str | Path, *, wheel_path: str | Path, base_image: str,
                            lock_path: str | Path | None = None) -> dict[str, str]:
    """Copy exactly Dockerfile (digest rendered into the ARG default), .dockerignore, lock and wheel."""
    if type(base_image) is not str or not re.fullmatch(EVALUATOR_BASE, base_image):
        raise ValueError("An official python:3.12-slim-bookworm@sha256 digest is required")
    staging = Path(staging_dir)
    if staging.is_symlink() or (staging.exists() and (not staging.is_dir() or any(staging.iterdir()))):
        raise ValueError("staging directory must be new or empty and never a link")
    source = source_dir()
    wheel = _regular(Path(wheel_path), "wheel")
    lock = _regular(Path(lock_path) if lock_path is not None else source / LOCK_NAME, "lock")
    if wheel.suffix != ".whl":
        raise ValueError("the harness wheel must end in .whl")
    pinned, problems = lock_is_hash_pinned(lock.read_text(encoding="utf-8"))
    if not pinned:
        raise ValueError("the Linux lock is not fully hash-pinned: " + "; ".join(problems[:5]))
    dockerfile = _regular(source / "Dockerfile", "Dockerfile").read_text(encoding="utf-8")
    if dockerfile.count("\nARG BASE_IMAGE\n") != 1:
        raise ValueError("the source Dockerfile must declare exactly one bare `ARG BASE_IMAGE`")
    rendered = dockerfile.replace("\nARG BASE_IMAGE\n", f"\nARG BASE_IMAGE={base_image}\n", 1)
    staging.mkdir(parents=True, exist_ok=True)
    payloads = {"Dockerfile": rendered.encode("utf-8"),
                ".dockerignore": _regular(source / ".dockerignore", ".dockerignore").read_bytes(),
                LOCK_NAME: lock.read_bytes(), wheel.name: wheel.read_bytes()}
    digests = {}
    for name, content in payloads.items():
        target = staging / name
        with target.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        digests[name] = hashlib.sha256(content).hexdigest()
    return digests


class PrepareInputs:
    """Plain, validated inputs for `prepare_images`; no I/O in the constructor beyond path normalisation."""

    def __init__(self, *, inference: str, evaluator_base: str, wheel_path: str | Path,
                 lock_path: str | Path | None = None, worker_iidfile: str | Path | None = None) -> None:
        if type(inference) is not str or not re.fullmatch(DIGEST_REF, inference):
            raise ValueError("the inference image must be a repository@sha256 digest reference")
        if type(evaluator_base) is not str or not re.fullmatch(EVALUATOR_BASE, evaluator_base):
            raise ValueError("An official python:3.12-slim-bookworm@sha256 digest is required")
        self.inference, self.evaluator_base = inference, evaluator_base
        self.wheel_path = Path(wheel_path)
        self.lock_path = Path(lock_path) if lock_path is not None else None
        self.worker_iidfile = Path(worker_iidfile) if worker_iidfile is not None else None


def read_iidfile(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        data = stream.read(80)
    if len(data) == 80:
        raise ValueError("image ID output exceeds its expected length")
    value = data.decode("ascii", "strict").strip() if data else ""
    if not re.fullmatch(IMAGE_ID, value):
        raise ValueError("iidfile must hold one complete sha256 image ID")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _inspect(executor, reference: str, *, bound: int) -> dict:
    result = executor.run(("docker", "image", "inspect", "--format", "{{json .}}", reference),
                          timeout_seconds=bound, max_output_bytes=_MAX_TEXT)
    if result.status != "completed" or result.returncode != 0:
        raise ValueError(f"image {reference} is not available locally ({result.status})")
    raw = json.loads(result.stdout)
    if not isinstance(raw, dict) or not re.fullmatch(IMAGE_ID, str(raw.get("Id"))):
        raise ValueError(f"unreadable image inspection for {reference}")
    return raw


def prepare_images(inputs: PrepareInputs, *, executor, artifact_dir: str | Path, clock: Callable[[], float] = time.monotonic,
                   session_lock=None, registry_digest: str | None = None, bound_seconds: int = 900) -> ImageBundle:
    """Pull, build, self-check and record every image a candidate may use. Any failed check refuses the bundle."""
    if session_lock is not None:
        from ..config import RunMode
        session_lock.check("container", RunMode.LIVE)
    artifacts = Path(artifact_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    if (artifacts / BUNDLE_NAME).exists():
        raise ValueError(f"{BUNDLE_NAME} already exists in {artifacts}; use a new output directory")
    started, log = clock(), []

    def call(argv: tuple[str, ...], *, bound: int = 300, max_output_bytes: int = _MAX_TEXT):
        result = executor.run(argv, timeout_seconds=bound, max_output_bytes=max_output_bytes)
        log.append({"argv": list(argv), "status": result.status, "returncode": result.returncode,
                    "offset_seconds": max(0.0, clock() - started)})
        return result

    def require(result, what: str):
        if result.status != "completed" or result.returncode != 0:
            raise ValueError(f"{what} failed ({result.status}, rc={result.returncode}): "
                             + (result.stdout + result.stderr).decode("utf-8", "replace")[-1500:])
        return result

    # 1. Inference image by digest, with its help/version captured from the exact image.
    require(call(("docker", "pull", inputs.inference), bound=bound_seconds), "pull inference")
    inference_raw = _inspect(executor, inputs.inference, bound=60)
    help_text = require(call(("docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint",
                              "/app/llama-server", inputs.inference, "--help"), bound=120),
                        "llama-server --help").stdout.decode("utf-8", "replace")
    version_text = require(call(("docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint",
                                 "/app/llama-server", inputs.inference, "--version"), bound=120), "llama-server --version")
    version_text = (version_text.stdout + version_text.stderr).decode("utf-8", "replace")
    caps = parse_server_help(help_text, version_text=version_text)
    build_info = parse_version(version_text)["build_info"]
    (artifacts / "llama-server-help.txt").write_bytes(help_text.encode("utf-8"))
    (artifacts / "llama-server-version.txt").write_bytes(version_text.encode("utf-8"))
    inference = ImageRef(role="inference", reference=inputs.inference, image_id=inference_raw["Id"],
                         entrypoint=tuple((inference_raw.get("Config") or {}).get("Entrypoint") or ()),
                         build_info=build_info, help_sha256=help_sha256(help_text),
                         source=f"prepare: pulled by digest into {artifacts.name}")
    if caps.help_sha256 != inference.help_sha256:
        raise ValueError("normalized help hash disagrees with the parsed capabilities")

    # 2. Evaluator base by digest, context staging, build with --iidfile, identity readback.
    require(call(("docker", "pull", pull_reference(inputs.evaluator_base)), bound=bound_seconds),
            "pull evaluator base")
    plan = evaluator_image_build_plan(base_image=inputs.evaluator_base, artifact_dir=artifacts,
                                      wheel_path=inputs.wheel_path,
                                      lock_path=inputs.lock_path or source_dir() / LOCK_NAME)
    staged = build_evaluator_context(artifacts / CONTEXT_DIR, wheel_path=inputs.wheel_path,
                                     base_image=inputs.evaluator_base, lock_path=inputs.lock_path)
    build = call(tuple(plan["build_argv"]), bound=bound_seconds, max_output_bytes=16 * 1024 * 1024)
    (artifacts / "evaluator-build.log").write_bytes(build.stdout + build.stderr)
    require(build, "docker build evaluator")
    evaluator_id = read_iidfile(plan["iidfile"])
    evaluator_raw = _inspect(executor, evaluator_id, bound=60)
    config = evaluator_raw.get("Config") or {}
    if tuple(config.get("Entrypoint") or ()) != EVALUATOR_ENTRYPOINT or config.get("Cmd"):
        raise ValueError("built evaluator image must expose the container_eval entrypoint and no CMD")
    if str(config.get("User")) != "10001:10001":
        raise ValueError("built evaluator image must run as 10001:10001")
    if f"{evaluator_raw.get('Os')}/{evaluator_raw.get('Architecture')}" != "linux/amd64":
        raise ValueError("built evaluator image is not linux/amd64")

    # 3. Self-check inside the built image: imports, registry digest, task availability, provenance.
    check = require(call(("docker", *SELF_CHECK_SHAPE[:6], evaluator_id, SELF_CHECK_SHAPE[7]), bound=300),
                    "evaluator self-check")
    try:
        report = json.loads(check.stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"self-check output is not JSON: {exc}") from exc
    (artifacts / "evaluator-self-check.json").write_bytes(check.stdout)
    expected_digest = registry_digest or _host_registry_digest()
    if not isinstance(report, dict) or report.get("ok") is not True:
        raise ValueError("evaluator self-check did not report ok=true: "
                         + json.dumps((report or {}).get("errors") if isinstance(report, dict) else report)[:1500])
    if report.get("registry_digest") != expected_digest:
        raise ValueError("evaluator image registry digest differs from the host registry digest")
    if not report.get("python_version") or not report.get("pip_freeze_sha256"):
        raise ValueError("evaluator image lacks /opt/llmbench provenance (python-version.txt, pip-freeze.txt)")
    evaluator = ImageRef(role="evaluator", reference=evaluator_id, image_id=evaluator_id,
                         entrypoint=EVALUATOR_ENTRYPOINT, build_info=report.get("python_version"),
                         source=f"prepare: built from {plan['base_image']} with wheel {inputs.wheel_path.name}")

    # 4. Optional worker image from a verified iidfile.
    worker = None
    if inputs.worker_iidfile is not None:
        worker_id = read_iidfile(inputs.worker_iidfile)
        worker_raw = _inspect(executor, worker_id, bound=60)
        worker = ImageRef(role="worker", reference=worker_id, image_id=worker_id,
                          entrypoint=tuple((worker_raw.get("Config") or {}).get("Entrypoint") or ()),
                          source=f"prepare: verified iidfile {inputs.worker_iidfile.name}")

    bundle = ImageBundle(prepared_utc=utc_now(), inference=inference, evaluator=evaluator, worker=worker,
                         evaluator_build={"plan": plan, "staged_sha256": staged, "self_check": report,
                                          "docker_log": log, "elapsed_seconds": max(0.0, clock() - started)},
                         help_sha256=inference.help_sha256, registry_digest=expected_digest)
    _atomic_write(artifacts / BUNDLE_NAME, (canonical_json(bundle.model_dump(mode="json")) + "\n").encode("utf-8"))
    return bundle


def _host_registry_digest() -> str:
    from ..registry import builtin_registry
    return builtin_registry().digest()


def build_wheel(project_root: str | Path, output_dir: str | Path, *, python: str | None = None,
                run: Callable[..., Any] = subprocess.run) -> Path:
    """`pip wheel . --no-deps --no-build-isolation` from the working tree; returns the single wheel path."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in output.glob("*.whl")}
    argv = [python or sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation",
            "--wheel-dir", str(output)]
    done = run(argv, cwd=str(Path(project_root).resolve()), capture_output=True, timeout=900)
    if done.returncode != 0:
        raise ValueError("pip wheel failed: " + (done.stdout + done.stderr).decode("utf-8", "replace")[-1500:])
    created = sorted(path for path in output.glob("*.whl") if path.name not in before)
    if len(created) != 1:
        raise ValueError(f"expected exactly one new wheel in {output}, found {len(created)}")
    return created[0]
