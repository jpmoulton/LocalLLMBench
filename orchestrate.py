"""Run a LocalLLMBench recipe, or benchmark model paths with shared defaults."""

import argparse
import json
from pathlib import Path

from orchestrator.recipe import build_plan
from orchestrator.logging import RunLogs
from orchestrator.runner import execute


BANNER = r"""
      __    ____  _________    __       __    __    __  ___
     / /   / __ \/ ____/   |  / /      / /   / /   /  |/  /
    / /   / / / / /   / /| | / /      / /   / /   / /|_/ /
   / /___/ /_/ / /___/ ___ |/ /___   / /___/ /___/ /  / /
  /_____/\____/\____/_/  |_/_____/  /_____/_____/_/  /_/

      ____  _______   __________  __         _|_|_|_|_
     / __ )/ ____/ | / / ____/ / / /        |  [] []  |
    / __  / __/ /  |/ / /   / /_/ /       ==|  [] []  |==
   / /_/ / /___/ /|  / /___/ __  /          |_________|
  /_____/_____/_/ |_/\____/_/ /_/             | | | |
"""


def preview(plan: dict, logs: RunLogs | None = None) -> None:
    logs = logs if logs is not None else RunLogs()
    logs.print(BANNER)
    logs.print(f"Run: {plan['name']}")
    logs.print(f"Trials: {len(plan['trials'])} (sequential)")
    logs.print(f"Benchmarks: {', '.join(plan['benchmarks'])}")
    logs.print(f"Server binding: {plan['inference']['bind_address']}:{plan['inference']['port']}")
    logs.print(f"Results root: {plan['output_dir']}")
    logs.print("\nInference settings across trials:")
    keys = sorted({key for trial in plan["trials"] for key in trial["parameters"]})
    for key in keys:
        values = list(dict.fromkeys(
            json.dumps(trial["parameters"][key], sort_keys=True) if key in trial["parameters"] else "<native default>"
            for trial in plan["trials"]
        ))
        logs.print(f"  {key}: {' -> '.join(values)}{' [varies]' if len(values) > 1 else ''}")
    logs.print("  Unspecified native flags use the inference image's defaults.")
    logs.print(f"\nFixed generation settings: {json.dumps(plan['generation'], sort_keys=True)}")
    logs.print(f"Fixed benchmark execution settings: {json.dumps(plan['execution'], sort_keys=True)}")
    logs.print("Benchmark options and prepared task identities are frozen in execution_plan.json for every trial.")
    for name in plan["benchmarks"]:
        logs.print(f"  {name}: {json.dumps(plan['benchmark_options'][name], sort_keys=True)}")
    logs.print("\nTrial schedule:")
    for trial in plan["trials"]:
        logs.print(f"  {trial['id']} | {trial['model']} | {trial['group']} | "
                   f"{json.dumps(trial['parameters'], sort_keys=True)}")
    logs.print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--recipe", type=Path, help="YAML recipe; relative paths resolve beside the recipe")
    inputs.add_argument("--models", nargs="+", help="GGUF model paths; use defaults and all registered benchmarks")
    parser.add_argument("--plan-only", action="store_true", help="Resolve and preview without Docker, inference, or output files")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(recipe_path=args.recipe, models=args.models)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.plan_only:
        preview(plan)
        print("Plan only: no containers started and no model loaded.")
        return 0
    with RunLogs() as logs:
        preview(plan, logs)
        output, run, report = execute(plan, logs=logs)
        logs.print(f"\nRun {run['status']} in {run['elapsed_seconds']:.1f}s: {json.dumps(run['counts'])}")
        if report:
            for name, metrics in report.get("benchmark_leaders", {}).items():
                for metric, leader in metrics.items():
                    logs.print(f"  {name} best {leader.get('label', metric)}: {leader['value']} "
                               f"(trials {', '.join(leader['trial_ids'])})")
        logs.print(f"Results and status: {output}")
        if (output / "results.xlsx").is_file():
            logs.print(f"Spreadsheet: {output / 'results.xlsx'}")
        logs.print("Comparison scores and generation metrics are in results.json and results.xlsx; no speed ranking is applied.")
        return 130 if run["status"] == "cancelled" else (0 if run["status"] == "completed" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
