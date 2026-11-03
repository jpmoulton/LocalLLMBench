"""Launch one llama.cpp container; inference options pass through unchanged."""

import argparse
from pathlib import Path
import subprocess


IMAGE = "localllmbench-inference:local"


def build_command(
    model: Path, *, name: str, port: int = 8080, bind_address: str = "127.0.0.1",
    server_args: list[str] | None = None, labels: dict[str, str] | None = None,
    network: str | None = None,
) -> list[str]:
    """Build the same inference launch for standalone and orchestrated use."""
    model = model.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"Model file does not exist: {model}")
    address = bind_address
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"
    command = ["docker", "run", "--detach", "--init", "--name", name, "--gpus", "all"]
    for key, value in (labels or {}).items():
        command.extend(["--label", f"{key}={value}"])
    if network:
        command.extend(["--network", network])
    command.extend([
        "--mount", f"type=bind,source={model.parent},target=/models,readonly",
        "--publish", f"{address}:{port}:8080",
        IMAGE, *(server_args or []),
        "--model", f"/models/{model.name}", "--host", "0.0.0.0", "--port", "8080",
    ])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Put llama-server options after --, for example: "
            "-- --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q8_0"
        ),
    )
    parser.add_argument("--model", required=True, type=Path, help="Host GGUF file path")
    parser.add_argument("--port", type=int, default=8080, help="Published host port (default: 8080)")
    parser.add_argument(
        "--bind-address", default="127.0.0.1",
        help="Host bind IP (default: 127.0.0.1; 0.0.0.0 exposes the server to the network)",
    )
    parser.add_argument("--name", default="localllmbench-inference", help="Docker container name")
    parser.add_argument("server_args", nargs=argparse.REMAINDER, help="Native llama-server options after --")
    args = parser.parse_args()

    model = args.model

    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]

    try:
        command = build_command(
            model, name=args.name, port=args.port, bind_address=args.bind_address,
            server_args=server_args,
        )
    except ValueError as exc:
        parser.error(str(exc))
    address = args.bind_address
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print(f"Container started; model loading may still be in progress.", flush=True)
        print(f"Readiness: http://{address}:{args.port}/health", flush=True)
        print(f"Logs: docker logs -f {args.name}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
