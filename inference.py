"""Launch one llama.cpp container; inference options pass through unchanged."""

import argparse
from pathlib import Path
import subprocess


IMAGE = "localllmbench-inference:local"


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

    model = args.model.expanduser().resolve()
    if not model.is_file():
        parser.error(f"Model file does not exist: {model}")

    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]

    address = args.bind_address
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"

    # Mount the directory so adjacent GGUF shards are also available, read-only.
    # No shell: spaces and native inference arguments remain separate argv entries.
    command = [
        "docker", "run", "--detach", "--init",
        "--name", args.name,
        "--gpus", "all",
        "--mount", f"type=bind,source={model.parent},target=/models,readonly",
        "--publish", f"{address}:{args.port}:8080",
        IMAGE,
        *server_args,
        # Transport and model selection belong to the launcher, not the passthrough.
        "--model", f"/models/{model.name}",
        "--host", "0.0.0.0", "--port", "8080",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print(f"Container started; model loading may still be in progress.", flush=True)
        print(f"Readiness: http://{address}:{args.port}/health", flush=True)
        print(f"Logs: docker logs -f {args.name}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
