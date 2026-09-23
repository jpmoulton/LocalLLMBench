"""Build the sandboxed coding worker image and record its immutable id. Needs network; never touches the GPU.

    python scripts/build_worker_image.py --output artifacts/container-prep/worker-image \\
        --base-image node:24-bookworm-slim@sha256:<digest>

On an Apple Silicon Mac (Docker in an arm64 Linux VM such as Colima) build the native arm64 worker instead, so
coding cases are not timed under emulation; the base digest must then be the official multi-platform index digest
(or its arm64 manifest):

    python scripts/build_worker_image.py --platform linux/arm64 --output artifacts/native-prep/worker-image \\
        --base-image node:24-bookworm-slim@sha256:<index digest>

The build plan comes from ``llmbench.coding.image_plan`` (an exact official base digest is required), the log and
``docker image inspect`` output are kept beside ``worker-image.id``, and that id - not a tag - is what runs. The
built image's own inspected Os/Architecture must equal ``--platform`` (default linux/amd64, the original worker);
it is recorded in ``build-plan.json`` and printed. Pass the iidfile to ``llmbench prepare --worker-iidfile``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmbench.coding.image_plan import (WORKER_PLATFORMS, verified_local_image_id,  # noqa: E402
                                        worker_image_build_plan)
from llmbench.config import RunMode  # noqa: E402
from llmbench.safety import SessionLock  # noqa: E402

INSPECT_TIMEOUT_SECONDS = 60


def inspected_platform(stdout: str, image: str) -> str:
    """`Os/Architecture` from `docker image inspect <id>` (a one-element JSON array), refused when unreadable or
    when it describes another image: the platform recorded is what was built, not what was asked for."""
    try:
        (raw,) = json.loads(stdout)
        if raw["Id"] != image:
            raise ValueError(f"inspection returned Id {raw['Id']!r}")
        return f"{raw['Os']}/{raw['Architecture']}"
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"unreadable docker image inspect output for {image}: {exc}") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-image", required=True, help="node:24-bookworm-slim@sha256:<64 hex>")
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", choices=WORKER_PLATFORMS, default="linux/amd64",
                        help="worker image platform (default linux/amd64; linux/arm64 for Docker on Apple Silicon)")
    parser.add_argument("--policy", default="runtime-policy.json")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    plan = worker_image_build_plan(base_image=args.base_image, artifact_dir=output, platform=args.platform)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    SessionLock.read(args.policy).check("container", RunMode.LIVE)
    output.mkdir(parents=True, exist_ok=True)
    (output / "build-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    with (output / "build.log").open("wb") as log:
        code = subprocess.run(plan["build_argv"], stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    if code != 0:
        print(f"docker build failed (exit {code}); see {output / 'build.log'}", file=sys.stderr)
        return 3
    image = verified_local_image_id(plan["iidfile"])
    try:
        inspected = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True,
                                   check=False, timeout=INSPECT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"docker image inspect {image} did not answer within {INSPECT_TIMEOUT_SECONDS} s", file=sys.stderr)
        return 3
    (output / "image-inspect.json").write_text(inspected.stdout, encoding="utf-8")
    try:
        if inspected.returncode != 0:
            raise ValueError(f"docker image inspect {image} failed (exit {inspected.returncode})")
        built = inspected_platform(inspected.stdout, image)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if built != args.platform:
        print(f"the built image is {built}, not the requested {args.platform}; it is not a usable worker for this "
              f"host (see {output / 'image-inspect.json'})", file=sys.stderr)
        return 3
    print(json.dumps({"image_id": image, "iidfile": plan["iidfile"], "platform": built}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
