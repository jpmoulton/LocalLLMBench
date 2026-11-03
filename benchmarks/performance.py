"""Extract generation evidence without importing Inspect or benchmark-specific code.

Server prompt/decode aggregates include only calls with both a token count and
positive elapsed time. Their coverage counters are the denominators for these
measurements; untimed usage is retained on individual rows, never substituted for
server timing. Summed per-call decode times are not concurrent wall throughput.
"""

from collections.abc import Mapping
import math
from typing import Any


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _number(value: Any) -> int | float | None:
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return value
    return None


def _tokens(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _seconds(value: Any) -> float | None:
    milliseconds = _number(value)
    return milliseconds / 1000 if milliseconds is not None else None


def _json_value(value: Any) -> Any:
    """Keep native score/error details JSON-safe, including Pydantic records."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _row(sample: Any, event: Any, call_index: int) -> dict:
    call = _get(event, "call")
    output = _get(event, "output") if event is not None else _get(sample, "output")
    response = _get(call, "response")
    timings = _get(response, "timings")
    usage = _get(output, "usage")
    response_usage = _get(response, "usage")
    choices = _get(output, "choices") or _get(response, "choices") or []
    finish_reason = None
    if choices:
        finish_reason = _get(choices[0], "stop_reason") or _get(choices[0], "finish_reason")
    generated_tokens = _tokens(_get(timings, "predicted_n"))
    decode_seconds = _seconds(_get(timings, "predicted_ms"))
    # An event's missing call time must not be replaced by the final sample's time.
    latency = _number(_get(call, "time") if event is not None else _get(output, "time"))
    input_tokens = _tokens(_get(usage, "input_tokens"))
    output_tokens = _tokens(_get(usage, "output_tokens"))
    if input_tokens is None:
        input_tokens = _tokens(_get(response_usage, "prompt_tokens"))
    if output_tokens is None:
        output_tokens = _tokens(_get(response_usage, "completion_tokens"))
    scores = _get(sample, "scores") or {}
    error = _get(event, "error") or _get(response, "error") or _get(sample, "error")
    return {
        "id": str(_get(sample, "id", "")),
        "epoch": _get(sample, "epoch", 1),
        "call_index": call_index,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_latency_seconds": latency,
        "prompt_tokens": _tokens(_get(timings, "prompt_n")),
        "prompt_cached_tokens": _tokens(_get(timings, "cache_n")),
        "prompt_seconds": _seconds(_get(timings, "prompt_ms")),
        "generated_tokens": generated_tokens,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            generated_tokens / decode_seconds
            if generated_tokens is not None and decode_seconds is not None and decode_seconds > 0
            else None
        ),
        "finish_reason": str(finish_reason) if finish_reason is not None else None,
        "error": _json_value(error),
        "scores": {str(name): _json_value(_get(score, "value", score))
                   for name, score in scores.items()},
        "timing_source": "model_event" if event is not None else "sample_output",
    }


def collect_generation_metrics(logs: list[Any]) -> dict:
    """Return common summary fields for native EvalLogs or equivalent mappings.

    Model events are canonical. If no event has a call, use a nonempty sample
    output once, with usage/latency only; its server timing remains unavailable.
    If no output exists, retain model events even without calls so failed calls
    are visible. Indices are one-based within each sample/epoch. Errors and
    truncated responses participate in measurements just like successful calls.
    """
    rows = []
    events_without_call = 0
    for log in logs:
        for sample in _get(log, "samples") or []:
            events = [event for event in _get(sample, "events") or []
                      if _get(event, "event") == "model"]
            events_without_call += sum(_get(event, "call") is None for event in events)
            has_calls = any(_get(event, "call") is not None for event in events)
            output = _get(sample, "output")
            has_output = output is not None and any(
                _get(output, field) is not None and _get(output, field) != []
                for field in ("time", "usage", "choices")
            )
            if has_calls or not has_output:
                rows.extend(_row(sample, event, index) for index, event in enumerate(events, 1))
            else:
                rows.append(_row(sample, None, 1))

    latencies = [row["request_latency_seconds"] for row in rows
                 if row["request_latency_seconds"] is not None]
    performance = {
        "model_calls": len(rows),
        "latency_calls": len(latencies),
        "request_latency_seconds_mean": sum(latencies) / len(latencies) if latencies else None,
        "request_latency_seconds_total": sum(latencies) if latencies else None,
        "output_fallback_calls": sum(row["timing_source"] == "sample_output" for row in rows),
        "model_events_without_call": events_without_call,
    }
    for prefix, token_key, seconds_key in (
        ("prompt", "prompt_tokens", "prompt_seconds"),
        ("decode", "generated_tokens", "decode_seconds"),
    ):
        measured = [row for row in rows if row[token_key] is not None
                    and row[seconds_key] is not None and row[seconds_key] > 0]
        tokens = sum(row[token_key] for row in measured) if measured else None
        seconds = sum(row[seconds_key] for row in measured) if measured else None
        performance[f"{prefix}_calls"] = len(measured)
        performance[token_key] = tokens
        performance[seconds_key] = seconds
        performance[f"{prefix}_tokens_per_second"] = tokens / seconds if seconds else None
    performance["decode_ms_per_token"] = (
        performance["decode_seconds"] * 1000 / performance["generated_tokens"]
        if performance["generated_tokens"] else None
    )
    return {"performance": performance, "generation_samples": rows}
