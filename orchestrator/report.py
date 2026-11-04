"""Aggregate comparison exports and lightweight execution status; no sample exports."""

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
        # Rows contain only scalar comparison data, never serialized objects.
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


def _cell(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub(lambda match: f"\\x{ord(match[0]):02x}", value)
    return value


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _settings(row: dict, prefix: str, value: Any) -> None:
    """Flatten requested settings into scalar columns, including repeated native flags."""
    if isinstance(value, dict):
        for key, item in value.items():
            _settings(row, f"{prefix}.{key}", item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _settings(row, f"{prefix}.{index}", item)
    else:
        row[prefix] = _cell(value)


def _error_text(error: Any) -> str | None:
    if isinstance(error, dict):
        error = error.get("message") or error.get("type")
    return _cell(" ".join(error.split())[:500]) if isinstance(error, str) and error else None


def _collect(output: Path, plan: dict, run: dict) -> tuple[list[dict], dict]:
    states = {str(trial["id"]): trial for trial in run.get("trials", [])}
    rows = []
    leader_inputs = []
    for planned in plan.get("trials", []):
        trial_id = str(planned["id"])
        trial: dict[str, Any] = {**planned, "status": "pending", **states.get(trial_id, {})}
        for benchmark in plan.get("benchmarks", []):
            summary_path = output / "trials" / trial_id / "benchmarks" / benchmark / "summary.json"
            summary = {}
            read_error = None
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(summary, dict):
                    raise ValueError("Benchmark summary must be a JSON object")
            except FileNotFoundError:
                pass
            except (OSError, UnicodeError, ValueError) as exc:
                read_error = f"{type(exc).__name__}: {exc}"
                summary = {}
            status = summary.get("status") or (
                trial["status"] if trial["status"] in ("pending", "running", "cancelled") else "incomplete"
            )
            try:
                duration = (
                    datetime.fromisoformat(summary["finished_at"])
                    - datetime.fromisoformat(summary["started_at"])
                ).total_seconds()
                duration = duration if duration >= 0 else None
            except (KeyError, TypeError, ValueError):
                duration = None
            row = {
                "trial_id": trial_id,
                "model": _cell(Path(trial["model"]).name) if trial.get("model") else None,
                "model_path": _cell(trial.get("model")),
                "configuration": _cell(trial.get("group")),
                "benchmark": _cell(benchmark),
                "trial_status": trial["status"],
                "benchmark_status": status,
            }
            _settings(row, "inference", trial.get("parameters", {}))
            _settings(row, "generation", plan.get("generation", {}))
            _settings(row, "benchmark_options", plan.get("benchmark_options", {}).get(benchmark, {}))
            metadata = summary.get("score_metrics", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            for metric in metadata:
                row[f"score.{metric}"] = _number(summary.get(metric))
            performance = summary.get("performance") or {}
            performance = performance if isinstance(performance, dict) else {}
            for field in dict.fromkeys((*PERFORMANCE_FIELDS, *performance)):
                row[f"performance.{field}"] = _number(performance.get(field))
            durations = trial.get("durations") or {}
            row.update({
                "load_seconds": _number(durations.get("load")),
                "probe_seconds": _number(durations.get("probe")),
                "benchmark_seconds": duration,
                "trial_benchmark_seconds": _number(durations.get("benchmark")),
                "cleanup_seconds": _number(durations.get("cleanup")),
                "trial_seconds": _number(trial.get("duration_seconds")),
                "request_concurrency": plan.get("execution", {}).get("concurrency"),
                "sample_count": _number(summary.get("denominator")),
                "scored_count": _number(summary.get("scored")),
            })
            counts = summary.get("counts", {})
            if isinstance(counts, dict):
                row.update({f"count.{key}": _number(value) for key, value in counts.items()})
            row["trial_error"] = _error_text(trial.get("error"))
            row["benchmark_error"] = _error_text(
                read_error or summary.get("run_error") or summary.get("summary_error")
            )
            rows.append(row)
            # Keep only aggregate score evidence for console leaders, not sample arrays.
            leader_inputs.append({
                "trial_id": trial_id, "benchmark": benchmark, "status": status,
                "summary": {"score_metrics": metadata, **{key: summary.get(key) for key in metadata}},
            })
    columns = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{column: row.get(column) for column in columns} for row in rows]
    return rows, _leaders(leader_inputs)


def write_status(output: Path, run: dict) -> None:
    """Persist progress without collecting benchmark data or regenerating exports."""
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 3,
        "run": {key: value for key, value in run.items() if key != "trials"},
        "trials": [
            {key: trial.get(key) for key in ("id", "status", "phase", "started_at", "finished_at", "error")}
            for trial in run.get("trials", [])
        ],
    }
    _atomic_json(output / "execution_report.json", _json_value(report))


def write_results(output: Path, plan: dict, run: dict) -> dict:
    """Write the same scalar rows to JSON and XLSX, retaining JSON if XLSX fails."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, leaders = _collect(output, plan, run)
    results = {"schema_version": 1, "results": rows}
    _atomic_json(output / "results.json", results)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    _table(sheet, rows, ("trial_id", "model", "model_path", "configuration", "benchmark", "trial_status", "benchmark_status"))
    with tempfile.NamedTemporaryFile(dir=output, suffix=".xlsx", delete=False) as file:
        temporary = Path(file.name)
    try:
        workbook.save(temporary)
        temporary.replace(output / "results.xlsx")
    finally:
        workbook.close()
        temporary.unlink(missing_ok=True)
    return {**results, "benchmark_leaders": leaders}
