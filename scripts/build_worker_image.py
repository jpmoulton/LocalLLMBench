"""Build the sandboxed coding worker image and record its immutable id. Needs network; never touches the GPU.

    python scripts/build_worker_image.py --output artifacts/container-prep/worker-image \\
        --base-image node:24-bookworm-slim@sha256:<digest>

The build plan comes from ``llmbench.coding.image_plan`` (an exact official base digest is required), the log and
``docker image inspect`` output are kept beside ``worker-image.id``, and that id - not a tag - is what runs.
Pass the iidfile to ``llmbench prepare --worker-iidfile``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmbench.coding.image_plan import verified_local_image_id, worker_image_build_plan  # noqa: E402
from llmbench.config import RunMode  # noqa: E402
from llmbench.safety import SessionLock  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-image", required=True, help="node:24-bookworm-slim@sha256:<64 hex>")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default="runtime-policy.json")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    plan = worker_image_build_plan(base_image=args.base_image, artifact_dir=output)
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
    inspected = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, check=False)
    (output / "image-inspect.json").write_text(inspected.stdout, encoding="utf-8")
    print(json.dumps({"image_id": image, "iidfile": plan["iidfile"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
