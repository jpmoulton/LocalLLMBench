"""Run registered benchmarks through shared model, logging, and execution services."""

import argparse
import importlib.metadata
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog
    from inspect_ai.model import GenerateConfig

from .config import load_config
from .performance import collect_generation_metrics
from .registry import get_definition, get_runtime




def save_json(path: Path, data: Any) -> None:
    """Replace atomically so interrupted writes cannot resemble completed results."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()




def server_provenance(base_url: str, output: Path, timeout: int) -> dict:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError("--base-url must end in /v1")
    server: dict[str, Any] = {"base_url": base_url, "captured_at": now()}
    # Save each response immediately: a later failed endpoint still leaves evidence.
    for key, url in (
        ("props", base_url[:-3] + "/props"),
        ("models", base_url + "/models"),
    ):
        with urlopen(url, timeout=timeout) as response:
            server[key] = json.load(response)
        save_json(output / "server.json", server)
    context = server["props"].get("default_generation_settings", {}).get("n_ctx")
    if isinstance(context, bool) or not isinstance(context, int):
        raise RuntimeError("Server /props does not report default_generation_settings.n_ctx; deployed context cannot be verified.")
    server["deployment_context"] = context
    save_json(output / "server.json", server)
    return server




def run_benchmark(name: str, options: dict, model: Any, generation: "GenerateConfig",
                  execution: dict, output: Path, expected_manifests: dict | None = None) -> dict:
    """Keep one benchmark's data, scoring, and failures inside its own result directory."""
    output.mkdir(parents=True, exist_ok=False)
    started = now()
    save_json(output / "summary.json", {
        "status": "incomplete", "phase": "preparing", "started_at": started,
    })
    logs: list["EvalLog"] = []
    provenance: dict = {}
    prepared = False
    runtime = None
    score_metrics = {}
    error = None
    interrupted = False
    log_dir = output / "inspect"
    try:
        definition = get_definition(name)
        score_metrics = getattr(definition, "SCORE_METRICS", {})
        # Freeze identity before loading runtime code or allowing any generation.
        payload, provenance = definition.prepare(options)
        prepared = True
        save_json(output / "manifest.json", provenance)
        if expected_manifests is not None and (
            name not in expected_manifests or provenance != expected_manifests[name]
        ):
            raise ValueError(f"{name} prepared manifest does not match the expected plan identity")
        runtime = get_runtime(name)
        task = runtime.make_task(payload, provenance, options, generation)
        save_json(output / "summary.json", {
            "status": "incomplete", "phase": "evaluating", "started_at": started,
        })
        from inspect_ai import eval as inspect_eval

        logs = inspect_eval(
            task,
            model=model,
            fail_on_error=False,
            retry_on_error=0,
            score_on_error=False,
            sandbox_cleanup=True,
            max_samples=execution["concurrency"],
            max_sandboxes=execution["concurrency"],
            epochs=1,
            log_dir=str(log_dir),
            log_format="eval",
            log_samples=True,
            log_model_api=True,
            log_buffer=1,
            log_header_only=False,
            score_display=False,
            display="plain",
        )
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(error["traceback"], file=sys.stderr)
        if not logs:
            for path in sorted(log_dir.glob("*.eval")):
                try:
                    from inspect_ai.log import read_eval_log

                    logs.append(read_eval_log(str(path)))
                except Exception as read_error:
                    error.setdefault("unreadable_logs", []).append({"path": str(path), "error": str(read_error)})
    summary = {"status": "incomplete", "run_error": error}
    if prepared and runtime is not None:
        try:
            summary = runtime.summarize(logs, provenance, error)
            if summary.get("status") not in ("completed", "incomplete"):
                raise ValueError(f"{name} returned an invalid summary status")
        except Exception as exc:
            summary = {
                "status": "incomplete", "run_error": error,
                "summary_error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
            }
    # Runtime-owned summaries cannot turn a preparation/evaluation error into success.
    if error is not None:
        summary["status"] = "incomplete"
        summary["run_error"] = error
    summary.update(collect_generation_metrics(logs))
    summary["score_metrics"] = score_metrics
    summary.update(started_at=started, finished_at=now())
    if interrupted:
        summary["interrupted"] = True
    save_json(output / "summary.json", summary)
    print(f"{name}: {summary['status']} ({output / 'summary.json'})", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-manifests", type=Path,
                        help="JSON map of frozen benchmark preparation identities")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    suite: dict[str, Any] = {
        "status": "incomplete", "started_at": now(), "benchmarks": {}, "run_error": None,
    }
    save_json(args.output / "summary.json", suite)
    try:
        config = load_config(args.config)
        names = config["benchmarks"]
        expected_manifests = None
        if args.expected_manifests is not None:
            expected_manifests = json.loads(args.expected_manifests.read_text(encoding="utf-8"))
            if not isinstance(expected_manifests, dict) or any(
                not isinstance(value, dict) for value in expected_manifests.values()
            ):
                raise ValueError("--expected-manifests must contain a JSON object of benchmark manifests")
        suite["benchmarks"] = {
            name: {"status": "pending", "summary": f"{name}/summary.json"} for name in names
        }
        save_json(args.output / "summary.json", suite)
        save_json(args.output / "effective_config.json", config)
        save_json(args.output / "versions.json", {
            "python": platform.python_version(),
            "packages": {distribution.metadata["Name"]: distribution.version
                         for distribution in importlib.metadata.distributions()},
        })
        execution = config["execution"]
        server_provenance(args.base_url, args.output, execution["request_timeout"])
        from inspect_ai.model import GenerateConfig, get_model

        generation = config["generation"]
        generate_config = GenerateConfig(
            max_tokens=generation["max_tokens"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            top_k=generation["top_k"],
            seed=generation["seed"],
            max_retries=0,
            timeout=execution["request_timeout"],
            max_connections=execution["concurrency"],
            cache=False,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": generation["thinking"]},
                "top_k": generation["top_k"],
                "min_p": generation["min_p"],
            },
        )
        model = get_model(
            f"openai-api/local/{config['inference']['model']}",
            base_url=args.base_url,
            api_key="local-no-key",
            config=generate_config,
            max_retries=0,
            client_timeout=execution["request_timeout"],
            responses_api=False,
            stream=False,
        )
        for name in names:
            result = run_benchmark(
                name, config["benchmark_options"][name], model, generate_config,
                execution, args.output / name, expected_manifests,
            )
            suite["benchmarks"][name]["status"] = result["status"]
            save_json(args.output / "summary.json", suite)
            if result.get("interrupted"):
                break
        if all(result["status"] == "completed" for result in suite["benchmarks"].values()):
            suite["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        suite["run_error"] = {
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
        }
        print(suite["run_error"]["traceback"], file=sys.stderr)
    suite["finished_at"] = now()
    save_json(args.output / "summary.json", suite)
    print(json.dumps(suite, indent=2))
    return 0 if suite["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
