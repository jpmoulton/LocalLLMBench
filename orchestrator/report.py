"""Export persisted trial evidence without running benchmarks or ranking speed."""

from datetime import datetime
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


PERFORMANCE_FIELDS = (
    "model_calls", "latency_calls", "decode_calls", "prompt_calls",
    "request_latency_seconds_mean", "request_latency_seconds_total",
    "prompt_tokens", "prompt_seconds", "prompt_tokens_per_second",
    "generated_tokens", "decode_seconds", "decode_tokens_per_second",
    "decode_ms_per_token",
)


def _json_value(value: Any) -> Any:
    """Keep JSON strict; nonfinite measurements are unavailable, never winners."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append(sheet: Any, values: list[Any]) -> None:
    """Explicit string cells prevent Excel interpreting external text as formulas."""
    row = []
    for value in values:
        value = _json_value(value)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, str):
            value = ILLEGAL_CHARACTERS_RE.sub(lambda match: f"\\x{ord(match[0]):02x}", value)
        row.append(value)
    sheet.append(row)
    for cell, value in zip(sheet[sheet.max_row], row):
        if isinstance(value, str):
            cell.data_type = "s"


def _table(sheet: Any, rows: list[dict], first_columns: tuple[str, ...]) -> None:
    columns = list(first_columns)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    _append(sheet, columns)
    for row in rows:
        _append(sheet, [row.get(key) for key in columns])
        for index, key in enumerate(columns, 1):
            value = row.get(key)
            if (key in {"summary_path", "manifest_path", "inspect_directory", "trial_path", "benchmark_summary"}
                    and isinstance(value, str) and value.startswith("trials/")
                    and ".." not in value.replace("\\", "/").split("/")):
                cell = sheet.cell(sheet.max_row, index)
                cell.hyperlink = value
                cell.style = "Hyperlink"
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for index, column in enumerate(columns, 1):
        sheet.column_dimensions[get_column_letter(index)].width = min(48, max(16, len(column) + 2))


def _leaders(results: list[dict]) -> dict:
    """Module declarations, not benchmark-specific names, determine score leaders."""
    candidates: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    leaders = {result["benchmark"]: {} for result in results}
    for result in results:
        summary = result["summary"]
        metadata = summary.get("score_metrics", {})
        if not isinstance(metadata, dict):
            continue
        for metric, definition in metadata.items():
            if isinstance(definition, dict) and definition.get("direction") in ("higher", "lower"):
                candidates.setdefault((result["benchmark"], metric), []).append((result, definition))
    for (benchmark, metric), entries in candidates.items():
        # Conflicting module declarations cannot define a meaningful comparison.
        directions = {definition["direction"] for _, definition in entries}
        units = {json.dumps(definition.get("unit"), sort_keys=True) for _, definition in entries}
        if len(directions) != 1 or len(units) != 1:
            continue
        eligible = []
        for result, definition in entries:
            score = result["summary"].get(metric)
            if (result["status"] == "completed" and isinstance(score, (int, float))
                    and not isinstance(score, bool) and math.isfinite(score)):
                eligible.append((result, definition, score))
        if not eligible:
            continue
        direction = entries[0][1]["direction"]
        best = (max if direction == "higher" else min)(entry[2] for entry in eligible)
        definition = eligible[0][1]
        leaders[benchmark][metric] = {
            "label": definition.get("label", metric), "direction": direction,
            "unit": definition.get("unit"), "value": best,
            "trial_ids": [result["trial_id"] for result, _, score in eligible if score == best],
        }
    return leaders


def _collect(output: Path, plan: dict, run: dict) -> dict:
    states = {str(trial["id"]): trial for trial in run.get("trials", [])}
    trials = []
    results = []
    samples = []
    errors = []
    for planned in plan.get("trials", []):
        trial_id = str(planned["id"])
        trial = {**planned, "status": "pending", **states.get(trial_id, {})}
        trial.setdefault("benchmark_summary", f"trials/{trial_id}/benchmarks/summary.json")
        trial["trial_path"] = f"trials/{trial_id}/trial.json"
        trials.append(trial)
        for benchmark in plan.get("benchmarks", []):
            directory = Path("trials") / trial_id / "benchmarks" / benchmark
            summary_path = (directory / "summary.json").as_posix()
            status = "pending" if trial["status"] == "pending" else "incomplete"
            summary = {}
            try:
                summary = json.loads((output / summary_path).read_text(encoding="utf-8"))
                if not isinstance(summary, dict):
                    raise ValueError("Benchmark summary must be a JSON object")
            except FileNotFoundError:
                pass
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append({"path": summary_path, "type": type(exc).__name__, "message": str(exc)})
                summary = {}
            if summary:
                status = summary.get("status", "incomplete")
            try:
                duration = (
                    datetime.fromisoformat(summary.get("finished_at"))
                    - datetime.fromisoformat(summary.get("started_at"))
                ).total_seconds()
                duration = duration if duration >= 0 else None
            except (TypeError, ValueError):
                duration = None
            result = {
                "trial_id": trial_id, "benchmark": benchmark, "model": trial.get("model"),
                "group": trial.get("group"), "parameters": trial.get("parameters", {}),
                "trial_status": trial["status"], "status": status,
                "summary_path": summary_path,
                "manifest_path": (directory / "manifest.json").as_posix(),
                "inspect_directory": (directory / "inspect").as_posix(),
                "inspect_logs": summary.get("inspect_logs", []),
                "expected_manifest": plan.get("manifests", {}).get(benchmark),
                "request_concurrency": plan.get("execution", {}).get("concurrency"),
                "parallel_slots": trial.get("parameters", {}).get("--parallel"),
                "benchmark_duration_seconds": duration,
                "summary": summary,
            }
            results.append(result)
            for field, row_type in (("samples", "outcome"), ("generation_samples", "generation")):
                rows = summary.get(field, [])
                if not isinstance(rows, list):
                    continue
                for sample in rows:
                    samples.append({
                        "trial_id": trial_id, "benchmark": benchmark, "row_type": row_type,
                        "summary_path": summary_path, "manifest_path": result["manifest_path"],
                        "inspect_directory": result["inspect_directory"],
                        "inspect_logs": result["inspect_logs"], "sample": sample,
                    })
    return _json_value({
        "schema_version": 1, "run": {key: value for key, value in run.items() if key != "trials"},
        "trials": trials, "benchmark_results": results, "samples": samples,
        "benchmark_leaders": _leaders(results), "report_errors": errors,
    })


def _workbook(report: dict, plan: dict) -> Workbook:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Run"
    run_rows = [{"Field": key, "Value": value} for key, value in report["run"].items()]
    for key in ("name", "source", "output_dir", "inference", "benchmarks", "benchmark_options",
                "generation", "execution", "manifests"):
        run_rows.append({"Field": f"plan.{key}", "Value": plan.get(key)})
    run_rows.extend([
        {"Field": "plan_path", "Value": "plan.json"},
        {"Field": "run_path", "Value": "run.json"},
        {"Field": "report_path", "Value": "report.json"},
        {"Field": "report_errors", "Value": report["report_errors"]},
        {"Field": "generation_measurements", "Value": (
            "Server decode rate is summed generated tokens / summed measured decode seconds; "
            "not total system throughput. Timing counters show coverage; missing is unavailable. "
            "No speed ranking. Probe timing is excluded."
        )},
        {"Field": "sample_rows", "Value": (
            "Outcome rows preserve benchmark-owned sample identities and fields; generation rows "
            "are individual model calls with native Inspect IDs. Do not count both as samples."
        )},
    ])
    _table(sheet, run_rows, ("Field", "Value"))
    sheet.auto_filter.ref = None
    _append(sheet, [])
    _append(sheet, ["Benchmark", "Metric", "Label", "Direction", "Value", "Trial IDs", "Unit"])
    for benchmark, metrics in report["benchmark_leaders"].items():
        if not metrics:
            _append(sheet, [benchmark, None, "No eligible completed scores"])
        for metric, leader in metrics.items():
            _append(sheet, [benchmark, metric, leader["label"], leader["direction"],
                            leader["value"], leader["trial_ids"], leader["unit"]])
    sheet.column_dimensions["A"].width = 32
    sheet.column_dimensions["B"].width = 80

    trial_rows = []
    for trial in report["trials"]:
        row = {key: value for key, value in trial.items() if key != "durations"}
        for phase in ("load", "probe", "benchmark", "cleanup"):
            row[f"duration.{phase}_seconds"] = (trial.get("durations") or {}).get(phase)
        trial_rows.append(row)
    _table(workbook.create_sheet("Trials"), trial_rows,
           ("id", "model", "group", "parameters", "status", "phase", "started_at", "finished_at",
            "duration_seconds", "error", "benchmark_summary", "trial_path"))

    result_rows = []
    for result in report["benchmark_results"]:
        row = {key: value for key, value in result.items() if key != "summary"}
        summary = result["summary"]
        for key, value in summary.items():
            if key not in ("samples", "generation_samples", "performance"):
                row[f"result.{key}"] = value
        performance = summary.get("performance") or {}
        for key in dict.fromkeys((*PERFORMANCE_FIELDS, *performance)):
            row[f"performance.{key}"] = performance.get(key)
        result_rows.append(row)
    _table(workbook.create_sheet("Benchmark results"), result_rows,
           ("trial_id", "benchmark", "model", "group", "status", "trial_status", "summary_path"))

    sample_rows = []
    for sample in report["samples"]:
        row = {key: value for key, value in sample.items() if key != "sample"}
        if isinstance(sample["sample"], dict):
            row.update({f"sample.{key}": value for key, value in sample["sample"].items()})
        else:
            row["sample"] = sample["sample"]
        sample_rows.append(row)
    _table(workbook.create_sheet("Samples"), sample_rows,
           ("trial_id", "benchmark", "row_type", "sample.id", "sample.epoch", "sample.call_index",
            "summary_path", "manifest_path", "inspect_directory", "inspect_logs"))
    return workbook


def write_report(output: Path, plan: dict, run: dict) -> dict:
    """Atomically save JSON first, then XLSX; leave authoritative JSON on export failure.

    Completed benchmark evaluations alone may provide declared quality leaders,
    even if a different benchmark made their trial incomplete. Generation metrics
    are evidence only. Pending trials and absent outputs never imply zero scores.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = _collect(output, plan, run)
    _atomic_json(output / "report.json", report)
    workbook = _workbook(report, plan)
    with tempfile.NamedTemporaryFile(dir=output, suffix=".xlsx", delete=False) as file:
        temporary = Path(file.name)
    try:
        workbook.save(temporary)
        temporary.replace(output / "results.xlsx")
    finally:
        workbook.close()
        temporary.unlink(missing_ok=True)
    return report
