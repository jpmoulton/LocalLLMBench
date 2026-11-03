"""Run selected benchmarks, or all registered benchmarks when selection is omitted."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import uuid

from benchmarks.config import load_config
from benchmarks.registry import BENCHMARKS, get_definition, resolve_benchmarks


IMAGE = "localllmbench-benchmark:local"
NETWORK = "localllmbench"


def docker_json(*args: str):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("benchmark_config.json"))
    parser.add_argument("--benchmarks", nargs="+", help="Benchmark names; overrides config selection")
    parser.add_argument("--list-benchmarks", action="store_true", help="List all registered benchmarks")
    parser.add_argument("--select-only", action="store_true", help="Preview selected benchmark inputs without Docker or inference")
    args = parser.parse_args()
    if args.list_benchmarks:
        print("\n".join(BENCHMARKS))
        return 0
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.benchmarks is not None:
        config["benchmarks"] = resolve_benchmarks(args.benchmarks)

    container_config = copy.deepcopy(config)
    mounts = []
    previews = {}
    needs_docker = False
    for name in config["benchmarks"]:
        definition = get_definition(name)
        options = config["benchmark_options"][name]
        needs_docker = needs_docker or definition.NEEDS_DOCKER
        for field in definition.PATH_OPTIONS:
            path = Path(options[field]).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            path = path.resolve()
            if not path.exists():
                parser.error(f"{name}.{field} does not exist: {path}")
            options[field] = str(path)
            destination = f"/datasets/{name}/{field}"
            container_config["benchmark_options"][name][field] = destination
            mounts.extend(["--mount", f"type=bind,source={path},target={destination},readonly"])
        if args.select_only:
            _, previews[name] = definition.prepare(options)
    if args.select_only:
        print(json.dumps(previews, indent=2))
        return 0

    inference = config["inference"]["container"]
    details = docker_json("inspect", inference)[0]
    if not details["State"]["Running"]:
        parser.error(f"Inference container is not running: {inference}")
    if NETWORK not in details["NetworkSettings"]["Networks"]:
        existing = subprocess.run(
            ["docker", "network", "inspect", NETWORK], capture_output=True, check=False,
        )
        if existing.returncode != 0:
            subprocess.run(["docker", "network", "create", NETWORK], check=True)
        subprocess.run(["docker", "network", "connect", NETWORK, inference], check=True)

    root = Path(config["output_dir"]).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = (root / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "container_config.json").write_text(json.dumps(container_config, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "inference.json").write_text(json.dumps({
        "container": inference,
        "image_id": details["Image"],
        "image": docker_json("image", "inspect", details["Image"])[0],
        "command": details["Config"]["Cmd"],
        "environment": details["Config"]["Env"],
        "mounts": details["Mounts"],
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Results: {output}", flush=True)
    socket_mount = ["--mount", "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock"] if needs_docker else []
    command = [
        "docker", "run", "--rm", "--init",
        "--name", f"localllmbench-benchmark-{run_id}",
        "--network", NETWORK,
        *socket_mount,
        *mounts,
        "--mount", f"type=bind,source={output},target=/results",
        IMAGE,
        "--config", "/results/container_config.json",
        "--output", "/results",
        "--base-url", f"http://{inference}:8080/v1",
    ]
    result = subprocess.run(command, check=False)
    print(f"Results retained in: {output}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
