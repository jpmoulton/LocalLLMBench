"""Conservative speed accounting: native tokens and stream delivery are distinct."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from .config import AcceptancePolicy


@dataclass(frozen=True)
class SpeedObservation:
    status: str
    finish_reason: str | None
    output_tokens: int | None
    native_generation_seconds: float | None
    elapsed_seconds: float
    first_event_seconds: float | None
    content_event_times: tuple[float, ...] = ()
    requested_output_tokens: int = 512
    input_tokens: int | None = None
    expected_input_tokens: int | None = None
    accepted_tokens_verified: bool = False
    synthetic: bool = False
    cancellation_acknowledged: bool = False
    prompt_processing_seconds: float | None = None
    native_tokens_per_second_reported: float | None = None
    native_timing_source: str | None = None

    def __post_init__(self):
        for name in ("accepted_tokens_verified", "synthetic", "cancellation_acknowledged"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        for name in ("output_tokens", "input_tokens", "expected_input_tokens", "requested_output_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} requires a nonnegative integer")
        for value in (self.elapsed_seconds, self.native_generation_seconds, self.first_event_seconds,
                      self.prompt_processing_seconds, self.native_tokens_per_second_reported,
                      *self.content_event_times):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError("timings must be finite nonnegative seconds")
        if tuple(sorted(self.content_event_times)) != self.content_event_times:
            raise ValueError("event times must be monotonically ordered")
        if self.content_event_times and self.content_event_times[-1] > self.elapsed_seconds:
            raise ValueError("event cannot occur after request completion")
        if self.native_generation_seconds is not None and self.native_generation_seconds > self.elapsed_seconds:
            raise ValueError("native generation duration cannot exceed request duration")

    def metrics(self) -> dict:
        rate = None
        if self.output_tokens is not None and self.native_generation_seconds:
            rate = self.output_tokens / self.native_generation_seconds
        elif self.native_tokens_per_second_reported is not None and self.native_timing_source:
            rate = self.native_tokens_per_second_reported
        end_to_end = self.output_tokens / self.elapsed_seconds if self.output_tokens is not None and self.elapsed_seconds else None
        gaps = [b - a for a, b in zip(self.content_event_times, self.content_event_times[1:])]
        # Include a silent tail after the last chunk; TTFT is reported separately.
        if self.content_event_times:
            gaps.append(self.elapsed_seconds - self.content_event_times[-1])
        context_verified = (self.input_tokens is not None and self.expected_input_tokens is not None
                            and self.input_tokens == self.expected_input_tokens)
        span = self.content_event_times[-1] - self.content_event_times[0] if len(self.content_event_times) > 1 else 0.
        # A pair of chunks emitted only at completion is still a buffered response.
        expected_span = self.output_tokens / rate if self.output_tokens and rate else None
        delivery_observable = (len(self.content_event_times) >= 2 and expected_span is not None
                               and span >= .5 * expected_span)
        return {
            **asdict(self), "native_tokens_per_second": rate,
            "end_to_end_tokens_per_second": end_to_end,
            "maximum_delivery_gap_seconds": max(gaps) if gaps else None,
            "stream_delivery_observable": delivery_observable,
            "content_delivery_span_seconds": span,
            "actual_context_verified": context_verified,
            "short_output": self.output_tokens is None or self.output_tokens < self.requested_output_tokens,
        }


def summarize_speed(observations: list[SpeedObservation], policy: AcceptancePolicy) -> dict:
    reasons = []
    if len(observations) < policy.speed_repetitions:
        reasons.append("insufficient_repetitions")
    rows = [item.metrics() for item in observations]
    valid_rates = []
    for index, row in enumerate(rows):
        faults = []
        if row["synthetic"]:
            faults.append("synthetic")
        if row["status"] != "completed":
            faults.append("incomplete")
        if row["finish_reason"] not in {"length", "stop", "max_tokens"}:
            faults.append("unknown_finish")
        if row["short_output"]:
            faults.append("short_output")
        if not row["accepted_tokens_verified"] or row["native_tokens_per_second"] is None:
            faults.append("native_token_evidence_missing")
        if not row["stream_delivery_observable"]:
            faults.append("buffered_or_unobservable_stream")
        if policy.require_actual_context_verified and not row["actual_context_verified"]:
            faults.append("input_count_unverified")
        if row["maximum_delivery_gap_seconds"] is not None and (
            row["maximum_delivery_gap_seconds"] > policy.maximum_stream_gap_seconds
        ):
            faults.append("delivery_stall")
        rate = row["native_tokens_per_second"]
        if rate is not None and rate < policy.minimum_tokens_per_second:
            faults.append("below_speed_floor")
        if rate is not None:
            valid_rates.append(rate)
        reasons.extend(f"repetition_{index}:{fault}" for fault in faults)
    return {"attempted": len(rows), "qualifies": bool(rows) and not reasons, "reasons": reasons,
            "minimum_native_tps": min(valid_rates) if valid_rates else None,
            "median_native_tps": statistics.median(valid_rates) if valid_rates else None,
            "maximum_native_tps": max(valid_rates) if valid_rates else None, "observations": rows}


def context_schedule(maximum=262144, minimum=8192) -> tuple[int, ...]:
    if type(maximum) is not int or type(minimum) is not int or minimum < 1 or maximum < minimum:
        raise ValueError("invalid context range")
    points = []
    current = minimum
    while current < maximum:
        points.append(current)
        current *= 2
    points.append(maximum)
    return tuple(points)


def context_envelope(rows: list[dict]) -> dict:
    """Never extrapolate over untested lengths; retain holes and failures in the curve."""
    ordered = sorted(rows, key=lambda row: row["actual_input_tokens"])
    usable = [row["actual_input_tokens"] for row in ordered
              if row.get("context_verified") is True and row.get("quality_passed") is True
              and row.get("speed_passed") is True and row.get("synthetic") is False]
    return {"largest_tested_usable_input_tokens": max(usable) if usable else None,
            "tested_lengths": ordered, "unmeasured_lengths_certified": False}


def build_speed_input(target_tokens, counter):
    """The largest filler prompt that fits ``target_tokens`` under the served chat template (bounded search).

    ``counter(messages, tools)`` must return the served tokenizer's count as ``{"tokens": int, ...}``; the best
    fitting render is returned unchanged so its actual occupancy is recorded, never an estimate.
    """
    def render(units):
        messages = [{"role": "system", "content": "You are a programming tutor. Produce a long detailed answer."},
                    {"role": "user", "content": "Reference notes:\n" +
                     "Python lists preserve insertion order. TypeScript checks types before runtime.\n" * units +
                     "\nWrite a comprehensive tutorial comparing Python and TypeScript data structures. "
                     "Give many examples and continue until the output limit. Do not conclude early."}]
        return counter(messages, [])
    baseline = render(0)
    if baseline["tokens"] > target_tokens:
        raise ValueError("requested prompt size is smaller than the speed task template")
    lo, hi = 0, max(1, target_tokens)
    best = baseline
    for _ in range(24):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        result = render(mid)
        if result["tokens"] <= target_tokens:
            best, lo = result, mid + 1
        else:
            hi = mid - 1
    return best
