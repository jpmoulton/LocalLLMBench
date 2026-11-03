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


def resolve_inputs(config: dict, base: Path) -> dict:
    """Resolve only selected benchmark paths; never mutate the caller's settings."""
    resolved = copy.deepcopy(config)
    for name in resolved["benchmarks"]:
        definition = get_definition(name)
        options = resolved["benchmark_options"][name]
        for field in definition.PATH_OPTIONS:
            path = Path(options[field]).expanduser()
            path = (path if path.is_absolute() else base / path).resolve()
            if not path.exists():
                raise ValueError(f"{name}.{field} does not exist: {path}")
            options[field] = str(path)
    return resolved


def build_command(
    config: dict, output: Path, *, name: str,
    labels: dict[str, str] | None = None, expected_manifests: dict | None = None,
) -> list[str]:
    """Prepare mounts/configuration for an already connected inference container."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = resolve_inputs(config, Path.cwd())
    container_config = copy.deepcopy(config)
    mounts = []
    needs_docker = False
    for benchmark_name in config["benchmarks"]:
        definition = get_definition(benchmark_name)
        needs_docker = needs_docker or definition.NEEDS_DOCKER
        for field in definition.PATH_OPTIONS:
            path = config["benchmark_options"][benchmark_name][field]
            destination = f"/datasets/{benchmark_name}/{field}"
            container_config["benchmark_options"][benchmark_name][field] = destination
            mounts.extend(["--mount", f"type=bind,source={path},target={destination},readonly"])
    for filename, data in (("config.json", config), ("container_config.json", container_config)):
        (output / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    command = ["docker", "run", "--rm", "--init", "--name", name, "--network", NETWORK]
    for key, value in (labels or {}).items():
        command.extend(["--label", f"{key}={value}"])
    # Compose propagates these labels to sibling code sandboxes, not secrets or host access.
    for key, variable in (("localllmbench.run", "LOCALLLMBENCH_RUN_ID"),
                          ("localllmbench.trial", "LOCALLLMBENCH_TRIAL_ID")):
        if key in (labels or {}):
            command.extend(["--env", f"{variable}={labels[key]}"])
    if needs_docker:
        command.extend(["--mount", "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock"])
    command.extend([
        *mounts, "--mount", f"type=bind,source={output},target=/results", IMAGE,
        "--config", "/results/container_config.json", "--output", "/results",
        "--base-url", f"http://{config['inference']['container']}:8080/v1",
    ])
    if expected_manifests is not None:
        (output / "expected_manifests.json").write_text(
            json.dumps(expected_manifests, indent=2) + "\n", encoding="utf-8",
        )
        command.extend(["--expected-manifests", "/results/expected_manifests.json"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("benchmark_config.json"))
    parser.add_argument("--benchmarks", nargs="+", help="Benchmark names; overrides config selection")
    parser.add_argument("--list-benchmarks", action="store_true", help="List all registered benchmarks")
    parser.add_argument("--select-only", action="store_true", help="Preview selected benchmark inputs without Docker or inference")
    parser.add_argument("--output", type=Path, help="Use this exact output directory instead of generating a run directory")
    args = parser.parse_args()
    if args.list_benchmarks:
        print("\n".join(BENCHMARKS))
        return 0
    config_path = args.config.resolve()
    try:
        config = load_config(config_path)
        if args.benchmarks is not None:
            config["benchmarks"] = resolve_benchmarks(args.benchmarks)
        config = resolve_inputs(config, config_path.parent)
        if args.select_only:
            previews = {name: get_definition(name).prepare(config["benchmark_options"][name])[1]
                        for name in config["benchmarks"]}
            print(json.dumps(previews, indent=2))
            return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    inference = config["inference"]["container"]
    details = docker_json("inspect", inference)[0]
    if not details["State"]["Running"]:
        parser.error(f"Inference container is not running: {inference}")
    if NETWORK not in details["NetworkSettings"]["Networks"]:
        existing = subprocess.run(["docker", "network", "inspect", NETWORK], capture_output=True)
        if existing.returncode:
            subprocess.run(["docker", "network", "create", NETWORK], check=True)
        subprocess.run(["docker", "network", "connect", NETWORK, inference], check=True)
    root = Path(config["output_dir"]).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output.resolve() if args.output else (root / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inference.json").write_text(json.dumps({
        "container": inference, "image_id": details["Image"],
        "image": docker_json("image", "inspect", details["Image"])[0],
        "command": details["Config"]["Cmd"], "environment": details["Config"]["Env"],
        "mounts": details["Mounts"],
    }, indent=2) + "\n", encoding="utf-8")
    command = build_command(config, output, name=f"localllmbench-benchmark-{run_id}")
    print(f"Results: {output}", flush=True)
    result = subprocess.run(command, check=False)
    print(f"Results retained in: {output}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
