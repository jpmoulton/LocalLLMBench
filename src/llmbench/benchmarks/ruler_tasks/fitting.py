"""Length targeting: grow a generated prompt to a requested token count with a real counter.

Mirrors the intent of NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER)
``scripts/data/synthetic/*.py``, where each generator loops over a haystack size and keeps the
largest size whose tokenized prompt still fits ``max_sequence_length``. Upstream walks the size
with a coarse incremental loop and its own tokenizer; this module binary-searches the same
quantity with the counter the caller injects, so the count comes from the route the suite already
trusts (llama.cpp ``/apply-template`` + ``/tokenize``) rather than from tiktoken.

The search only ever accepts a candidate it actually counted, and never accepts one above the
target, so an indivisible unit (one noise sentence, one numbered word) can leave the prompt a few
tokens short. That shortfall is reported, never hidden: the caller compares it against a tolerance
and refuses to call an item a measurement at the requested length when it does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

DEFAULT_MAX_COUNTER_CALLS = 24
"""Each call is one HTTP round trip against a prompt that can be 128K tokens long."""


class LengthUnreachable(ValueError):
    """The requested token target cannot be produced (template too large, or units exhausted)."""


@dataclass(frozen=True)
class Fit:
    units: int
    tokens: int
    counter_calls: int
    base_tokens: int
    """Tokens of the unit-free prompt (template plus the item's own indivisible content)."""


def normalize_count(value: Any) -> int:
    """Accept either a plain integer or the suite's ``{"tokens": int, ...}`` counter envelope."""
    if isinstance(value, Mapping):
        value = value.get("tokens")
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError("the token counter must return a nonnegative integer or {'tokens': int}")
    return value


def fit_units(count_units: Callable[[int], Any], *, target_tokens: int, max_units: int,
              max_counter_calls: int = DEFAULT_MAX_COUNTER_CALLS) -> Fit:
    """Largest unit count whose counted prompt is <= ``target_tokens``.

    ``count_units(units)`` renders and counts; it is assumed nondecreasing in ``units`` (every
    unit only adds text). Results are cached so a repeated probe costs nothing. The call budget is
    hard: the search returns the best proven-fitting candidate found within it.
    """
    for name, value, minimum in (("target_tokens", target_tokens, 1), ("max_units", max_units, 1),
                                 ("max_counter_calls", max_counter_calls, 3)):
        if type(value) is not int or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    seen: dict[int, int] = {}
    calls = 0

    def count(units: int) -> int:
        nonlocal calls
        if units not in seen:
            if calls >= max_counter_calls:
                raise LengthUnreachable(
                    f"the length search exhausted its {max_counter_calls}-call counting budget")
            calls += 1
            seen[units] = normalize_count(count_units(units))
        return seen[units]

    base = count(0)
    if base > target_tokens:
        raise LengthUnreachable(
            f"the task template alone needs {base} tokens, above the {target_tokens}-token target")
    probe_units = min(max_units, 32)
    probed = count(probe_units)
    per_unit = max((probed - base) / probe_units, 1e-9) if probe_units else 1e-9
    best_units, best_tokens = (probe_units, probed) if probed <= target_tokens else (0, base)
    # Bracket around the linear estimate, then widen until a unit count is proven to overshoot.
    high = min(max_units, max(probe_units * 2, int((target_tokens - base) / per_unit * 1.25) + 8))
    try:
        while count(high) <= target_tokens:
            best_units, best_tokens = high, seen[high]
            if high >= max_units:
                return Fit(best_units, best_tokens, calls, base)
            high = min(max_units, max(high * 2, high + 8))
        low = best_units
        while low < high - 1:
            middle = (low + high) // 2
            tokens = count(middle)
            if tokens <= target_tokens:
                low, best_units, best_tokens = middle, middle, tokens
            else:
                high = middle
    except LengthUnreachable:
        pass  # the call budget ended the search; only a proven-fitting candidate is returned
    return Fit(best_units, best_tokens, calls, base)
