#!/usr/bin/env python
"""Compact, evidence-labelled summary of finished candidates, per suite, from run directories. Read-only.

    python scripts/native_results_summary.py runs/mac/sweep-main runs/mac/sessions --json out.json --markdown out.md

Every ``candidates/*/result.json`` under the given roots becomes one row. Scores are split by SUITE rather than the
campaign's categories, because a category mixes suites (the tools category is BFCL plus the local tool fixtures;
retrieval is RULER plus the local NIAH fixtures) and a single mixed number hides which suite moved. The rules the
rest of the project uses hold here too:

* an ``environment_error`` row is a harness failure: it is counted separately and never enters a score;
* a response cut off by its output cap is counted as ``truncated`` beside the score, never folded into it silently;
* the local NIAH fixtures carry a strict (exact protocol) and a labelled lenient score; both are printed;
* speed comes only from the server's own timings; a candidate whose probes failed has no tok/s, not a zero;
* memory on a ``metal-native`` row is unified memory (server phys_footprint, Metal buffers, host swap); it is never
  called VRAM.

Nothing here re-scores, ranks or picks a winner; the session reports remain the record.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MIB = 1024 * 1024


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mib(value):
    return None if not isinstance(value, (int, float)) or isinstance(value, bool) else round(value / MIB)


def _suite_rows(samples: list[dict]) -> dict:
    suites: dict[str, dict] = {}
    for row in samples:
        if not isinstance(row, dict):
            continue
        suite = row.get("suite") or str(row.get("task_id", "")).split("/")[0] or "unknown"
        entry = suites.setdefault(suite, {"items": 0, "scored": 0, "score_sum": 0.0, "harness_errors": 0,
                                          "failures": 0, "truncated": 0, "lenient_correct": 0})
        entry["items"] += 1
        status = row.get("status")
        if status == "environment_error":
            entry["harness_errors"] += 1
            continue
        entry["scored"] += 1
        cut_off = row.get("finish_reason") == "length" or row.get("output_cap_hit") is True
        if status == "completed":
            entry["score_sum"] += float(row.get("score") or 0)
        elif not cut_off:
            entry["failures"] += 1  # invalid output, timeout, transport error without an output cap: scores 0
        if cut_off:  # a response the output cap cut off, whatever status its scorer gave it: reported as such
            entry["truncated"] += 1
        if row.get("lenient_outcome_correct") is True:
            entry["lenient_correct"] += 1
    return {name: {"items": value["items"], "scored": value["scored"],
                   "passed": round(value["score_sum"], 3),
                   "score": round(value["score_sum"] / value["scored"], 3) if value["scored"] else None,
                   "harness_errors": value["harness_errors"], "failed_not_cut_off": value["failures"],
                   "truncated": value["truncated"], "lenient_correct": value["lenient_correct"]}
            for name, value in sorted(suites.items())}


def _responses_cut_off(run_dir: Path) -> dict:
    """Per suite, how many recorded model RESPONSES hit the output cap, read from the raw response files the
    evaluator persists (``evaluator/benchmarks/<suite>/...`` and ``evaluator/coding/...``). Multi-attempt suites
    (Aider Polyglot, the coding fixtures) record the finish reason per response, not on the scored row, so a row-level
    count alone would miss a capped attempt. An adapter's own ``response*.json`` and the transport's ``request-*.json``
    hold the same completion, so a suite's adapter files are used when it wrote any and its transport files otherwise
    (never both, which would count every response twice)."""
    found: dict[str, dict[str, list[str | None]]] = {}
    root = run_dir / "evaluator"
    for path in sorted(root.rglob("*.json")) if root.is_dir() else ():
        parts = path.relative_to(root).parts
        if len(parts) < 2 or parts[0] not in ("benchmarks", "coding"):
            continue
        body = _load(path)
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        suite = parts[1] if parts[0] == "benchmarks" else "coding"
        kind = "adapter" if path.name.startswith("response") else "transport"
        found.setdefault(suite, {"adapter": [], "transport": []})[kind].append(choices[0].get("finish_reason"))
    result = {}
    for suite, kinds in found.items():
        reasons = kinds["adapter"] or kinds["transport"]
        result[suite] = {"cut_off": sum(reason == "length" for reason in reasons), "responses": len(reasons)}
    return result


def candidate(run_dir: Path) -> dict:
    result = _load(run_dir / "result.json") or {}
    config = _load(run_dir / "config.json") or {}
    evaluation = _load(run_dir / "evaluation.json") or {}
    engine, memory = config.get("engine") or {}, result.get("memory") or {}
    after, during, log = memory.get("after_load") or {}, memory.get("during_evaluation") or {}, \
        memory.get("server_log") or {}
    speed = result.get("speed") or {}
    timings = [item for item in evaluation.get("native_timings") or [] if isinstance(item, dict)]
    prefill = [item["prompt_n"] / (item["prompt_ms"] / 1000) for item in timings
               if item.get("prompt_ms") and item.get("prompt_n")]
    first = [row.get("first_event_seconds") for row in speed.get("observations") or []
             if isinstance(row, dict) and isinstance(row.get("first_event_seconds"), (int, float))]
    asset = (config.get("assets") or [{}])[0]
    server = config.get("native_server") or config.get("inference_image") or {}
    return {
        "run_dir": str(run_dir), "model": asset.get("model_name"), "model_sha256": asset.get("sha256"),
        "label": config.get("label"), "runtime": config.get("runtime", "nvidia-container"),
        "build_info": server.get("build_info"), "state": result.get("state"),
        "failure_stage": result.get("failure_stage"), "failure_reasons": list(result.get("failure_reasons") or []),
        "ctx_size": engine.get("ctx_size"), "requested_input_tokens": config.get("requested_input_tokens"),
        "kv": f"{engine.get('cache_type_k')}/{engine.get('cache_type_v')}", "reasoning": engine.get("reasoning"),
        "flash_attn": engine.get("flash_attn"), "batch": f"{engine.get('batch_size')}/{engine.get('ubatch_size')}",
        "n_gpu_layers": engine.get("n_gpu_layers"), "offloaded": (f"{log.get('offloaded_layers')}/"
                                                                  f"{log.get('total_layers')}" if log else None),
        "memory_reserve_mib": (config.get("native_limits") or {}).get("memory_reserve_mib"),
        "tps_min": speed.get("minimum_native_tps"), "tps_median": speed.get("median_native_tps"),
        "tps_max": speed.get("maximum_native_tps"), "speed_reasons": list(speed.get("reasons") or [])[:6],
        "prefill_tps_median": statistics.median(prefill) if prefill else None,
        "first_token_s_median": statistics.median(first) if first else None,
        "load_s": result.get("load_seconds"), "settings_verified": result.get("effective_settings_verified"),
        "context_verified": result.get("actual_context_verified"),
        "server_footprint_after_load_mib": after.get("server_phys_footprint_mib"),
        "server_footprint_peak_mib": _mib(during.get("peak_phys_footprint_bytes")),
        "metal_buffers_mib": log.get("metal_resident_mib"), "metal_budget_mib": log.get("metal_budget_mib"),
        "host_swap_growth_mib": _mib(during.get("swap_growth_bytes")),
        "host_available_min_mib": _mib(during.get("min_available_bytes")),
        "memory_pressure_max": during.get("max_pressure_level"),
        "gpu_utilization_median": (during.get("gpu_utilization_percent") or {}).get("median"),
        "watchdog_violations": [item.get("kind") for item in during.get("violations") or []
                                if isinstance(item, dict)],
        "power": ((memory.get("admission") or {}).get("power") or {}).get("power_source"),
        "vram_mib": result.get("vram_used_mib_after_load"),
        "suites": _suite_rows(evaluation.get("samples") or []),
        "responses_cut_off": _responses_cut_off(run_dir),
        "cleanup_verified": (result.get("cleanup") or {}).get("verified"),
        "server_exit": (result.get("cleanup") or {}).get("server_exit"),
    }


def collect(roots) -> list[dict]:
    rows = []
    for root in roots:
        for path in sorted(Path(root).rglob("result.json")):
            if path.parent.parent.name == "candidates":
                rows.append(candidate(path.parent))
    return rows


def _fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _suite_cell(suites: dict, name: str) -> str:
    entry = suites.get(name)
    if not entry:
        return "-"
    text = f"{_fmt(entry['passed'], 0) if float(entry['passed']).is_integer() else _fmt(entry['passed'], 2)}/{entry['scored']}"
    notes = []
    if entry["harness_errors"]:
        notes.append(f"{entry['harness_errors']} harness err")
    if entry["truncated"]:
        notes.append(f"{entry['truncated']} cut off")
    if entry["failed_not_cut_off"]:
        notes.append(f"{entry['failed_not_cut_off']} invalid")
    return text + (f" ({', '.join(notes)})" if notes else "")


def markdown(rows: list[dict]) -> str:
    out = ["| Model | Candidate | ctx | KV | reasoning | state | tok/s median (min-max) | prefill tok/s | first token s "
           "| footprint MiB after load / peak | Metal MiB | swap growth MiB | pressure max | settings ok | ctx ok "
           "| BFCL | RULER | NIAH strict (lenient) | tool fixtures |",
           "|---|---|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|"]
    for row in rows:
        suites = row["suites"]
        niah = suites.get("niah")
        niah_cell = "-" if not niah else f"{_fmt(niah['passed'], 0)}/{niah['scored']} ({niah['lenient_correct']}/{niah['scored']})"
        tools = [suites.get(name) for name in ("tool-probes", "tool-episodes") if suites.get(name)]
        tool_cell = "-" if not tools else f"{_fmt(sum(t['passed'] for t in tools), 0)}/{sum(t['scored'] for t in tools)}"
        speed = "-" if row["tps_median"] is None else (f"{row['tps_median']:.1f} ({row['tps_min']:.1f}-"
                                                        f"{row['tps_max']:.1f})")
        out.append("| " + " | ".join([
            str(row["model"]), str(row["label"]), _fmt(row["ctx_size"]), row["kv"], str(row["reasoning"]),
            str(row["state"]), speed, _fmt(row["prefill_tps_median"], 0), _fmt(row["first_token_s_median"], 2),
            f"{_fmt(row['server_footprint_after_load_mib'], 0)} / {_fmt(row['server_footprint_peak_mib'], 0)}",
            _fmt(row["metal_buffers_mib"], 0), _fmt(row["host_swap_growth_mib"], 0),
            _fmt(row["memory_pressure_max"]), _fmt(row["settings_verified"]), _fmt(row["context_verified"]),
            _suite_cell(suites, "bfcl"), _suite_cell(suites, "ruler"), niah_cell, tool_cell]) + " |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", help="run directories (sweeps or sessions) to summarise")
    parser.add_argument("--json", help="write the rows as JSON here")
    parser.add_argument("--markdown", help="write the table as Markdown here")
    args = parser.parse_args(argv)
    rows = collect(args.roots)
    if not rows:
        print("no candidates/*/result.json under the given roots", file=sys.stderr)
        return 2
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    table = markdown(rows)
    if args.markdown:
        Path(args.markdown).write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
