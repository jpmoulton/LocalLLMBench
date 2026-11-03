"""Run a LocalLLMBench recipe, or benchmark model paths with shared defaults."""

import argparse
import json
from pathlib import Path

from orchestrator.recipe import build_plan
from orchestrator.runner import DockerBackend, execute


BANNER = r"""
 _                    _ _     _     __  __ ____                  _
| |    ___   ___ __ _| | |   | |   |  \/  | __ )  ___ _ __   ___| |__
| |   / _ \ / __/ _` | | |   | |   | |\/| |  _ \ / _ \ '_ \ / __| '_ \
| |__| (_) | (_| (_| | | |___| |___| |  | | |_) |  __/ | | | (__| | | |
|_____\___/ \___\__,_|_|_____|_____|_|  |_|____/ \___|_| |_|\___|_| |_|
"""


def preview(plan: dict) -> None:
    print(BANNER, flush=True)
    print(f"Run: {plan['name']}")
    print(f"Trials: {len(plan['trials'])} (sequential)")
    print(f"Benchmarks: {', '.join(plan['benchmarks'])}")
    print(f"Server binding: {plan['inference']['bind_address']}:{plan['inference']['port']}")
    print(f"Results root: {plan['output_dir']}")
    print("\nInference settings across trials:")
    keys = sorted({key for trial in plan["trials"] for key in trial["parameters"]})
    for key in keys:
        values = list(dict.fromkeys(
            json.dumps(trial["parameters"][key], sort_keys=True) if key in trial["parameters"] else "<native default>"
            for trial in plan["trials"]
        ))
        print(f"  {key}: {' -> '.join(values)}{' [varies]' if len(values) > 1 else ''}")
    print("  Unspecified native flags use the inference image's defaults.")
    print(f"\nFixed generation settings: {json.dumps(plan['generation'], sort_keys=True)}")
    print(f"Fixed benchmark execution settings: {json.dumps(plan['execution'], sort_keys=True)}")
    print("Benchmark options and prepared task identities are frozen in plan.json for every trial.")
    for name in plan["benchmarks"]:
        print(f"  {name}: {json.dumps(plan['benchmark_options'][name], sort_keys=True)}")
    print("\nTrial schedule:")
    for trial in plan["trials"]:
        print(f"  {trial['id']} | {trial['model']} | {trial['group']} | "
              f"{json.dumps(trial['parameters'], sort_keys=True)}")
    print(flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--recipe", type=Path, help="YAML recipe; relative paths resolve beside the recipe")
    inputs.add_argument("--models", nargs="+", help="GGUF model paths; use defaults and all registered benchmarks")
    parser.add_argument("--plan-only", action="store_true", help="Resolve and preview without Docker, inference, or output files")
    parser.add_argument("--verbose_logging", action="store_true",
                        help="Stream llama.cpp logs alongside source-labelled benchmark output")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(recipe_path=args.recipe, models=args.models)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    preview(plan)
    if args.plan_only:
        print("Plan only: no containers started and no model loaded.")
        return 0
    output, run, report = execute(plan, backend=DockerBackend(verbose_logging=args.verbose_logging))
    print(f"\nRun {run['status']} in {run['elapsed_seconds']:.1f}s: {json.dumps(run['counts'])}")
    if report:
        for name, metrics in report.get("benchmark_leaders", {}).items():
            for metric, leader in metrics.items():
                print(f"  {name} best {leader.get('label', metric)}: {leader['value']} "
                      f"(trials {', '.join(leader['trial_ids'])})")
    print(f"Results and status: {output}")
    if (output / "results.xlsx").is_file():
        print(f"Spreadsheet: {output / 'results.xlsx'}")
    print("Generation metrics are in the benchmark summaries and spreadsheet; no speed ranking is applied.")
    return 130 if run["status"] == "cancelled" else (0 if run["status"] == "completed" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
