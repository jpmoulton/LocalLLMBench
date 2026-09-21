"""RULER's judge-free substring scorers.

Mirrors NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER) at commit
``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``: ``scripts/eval/synthetic/constants.py``
(``string_match_all``, ``string_match_part`` and the per-task metric assignment) and
``scripts/eval/evaluate.py`` (``postprocess_pred`` - note it lives there, not in the constants
module). Every task vendored here scores with ``string_match_all``; ``string_match_part`` is the
QA-family metric and is kept so a later QA task scores the way upstream does.

VERIFIED, NO DIVERGENCE. ``tests/data/ruler-upstream-scoring.py`` holds upstream's own three
functions verbatim and ``test_vendored_scorer_agrees_with_upstreams_own_implementation`` runs them
side by side with these. The only difference is shape: upstream's metrics take a *batch* and return
``round(mean * 100, 2)``, these take one sample and return a fraction in ``[0, 1]``; the adapter
does the averaging.

Upstream semantics, reproduced exactly:
  * ``postprocess_pred`` strips the prediction and replaces every C0 control character with a
    newline, then strips again.
  * ``string_match_all`` = fraction of reference strings that occur as a case-insensitive
    substring of the prediction. Partial credit is the point: a 4-needle item that recovers 3
    needles scores 0.75.
  * ``string_match_part`` = 1.0 when ANY reference occurs, else 0.0.
No judge, no model, no fuzzy matching, no network.

The lenient pass is this suite's own LIVE-004 addition, never upstream's: it is reported beside
the strict score and can never replace it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

STRING_MATCH_ALL = "string_match_all"
STRING_MATCH_PART = "string_match_part"
METRICS = (STRING_MATCH_ALL, STRING_MATCH_PART)

_CONTROL = re.compile(r"[\x00-\x1f]")
_WHITESPACE = re.compile(r"\s+")


def postprocess_pred(prediction: Any) -> str:
    """Upstream ``postprocess_pred``: strip, control characters to newlines, strip again."""
    if type(prediction) is not str:
        return ""
    return _CONTROL.sub("\n", prediction.strip()).strip()


def _present(prediction: str, reference: str) -> bool:
    return reference.lower() in prediction.lower()


def string_match_all(prediction: str, references: Sequence[str]) -> float:
    """Fraction of references present in the prediction (upstream's per-sample value / 100)."""
    if not references:
        raise ValueError("a RULER item must declare at least one reference string")
    return sum(_present(prediction, reference) for reference in references) / len(references)


def string_match_part(prediction: str, references: Sequence[str]) -> float:
    """1.0 when any reference is present, else 0.0 (upstream's per-sample value / 100)."""
    if not references:
        raise ValueError("a RULER item must declare at least one reference string")
    return float(any(_present(prediction, reference) for reference in references))


def lenient_text(prediction: Any) -> str:
    """LIVE-004-style labelled lenient rendering: one markdown fence removed, whitespace collapsed.

    This never changes which strings are being looked for and never repairs semantics; it only
    removes formatting that a chat model wraps an otherwise correct answer in. The strict score is
    always computed and reported separately.
    """
    from ...evaluations.retrieval import strip_one_markdown_fence

    text = postprocess_pred(prediction)
    body = strip_one_markdown_fence(text)
    return _WHITESPACE.sub(" ", (body if body is not None else text)).strip()


@dataclass(frozen=True)
class ItemScore:
    metric: str
    score: float
    string_match_all: float
    string_match_part: float
    matched: tuple[str, ...]
    per_expected: tuple[dict[str, Any], ...]
    prediction_empty: bool

    @property
    def correct(self) -> bool:
        return self.score >= 1.0


def score_prediction(prediction: Any, references: Sequence[str], *, metric: str = STRING_MATCH_ALL,
                     lenient: bool = False) -> ItemScore:
    """Score one RULER item. ``lenient`` only changes how the prediction text is rendered."""
    if metric not in METRICS:
        raise ValueError(f"unknown RULER metric: {metric!r}")
    if not references or any(type(reference) is not str or not reference for reference in references):
        raise ValueError("references must be a non-empty sequence of non-empty strings")
    text = lenient_text(prediction) if lenient else postprocess_pred(prediction)
    rows = tuple({"expected": reference, "present": _present(text, reference)} for reference in references)
    every = sum(row["present"] for row in rows) / len(rows)
    part = float(any(row["present"] for row in rows))
    return ItemScore(metric, every if metric == STRING_MATCH_ALL else part, every, part,
                     tuple(row["expected"] for row in rows if row["present"]), rows, not text)
