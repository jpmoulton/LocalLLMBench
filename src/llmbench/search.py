"""Budgeted candidate search and paired uncertainty, independent of an LLM supervisor."""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field

from .config import Budgets, CampaignPolicy, ExperimentManifest, canonical_json


def paired_comparison(baseline: list[dict], candidate: list[dict], *, maximum_loss=.02,
                      resamples=2000, seed=42, minimum_tasks=20) -> dict:
    """Bootstrap whole tasks. Failed attempts score zero and never disappear."""
    def index(rows):
        result = {}
        fixtures_seen = set()
        for row in rows:
            key = row["task_id"]
            if key in result:
                raise ValueError("duplicate task IDs: aggregate seeded replicates at task level first")
            fixture_hash = row.get("fixture_hash")
            if fixture_hash in fixtures_seen:
                raise ValueError("duplicate fixture identity: renamed/replayed tasks are not independent evidence")
            fixtures_seen.add(fixture_hash)
            score = row.get("score", 0.0) if row.get("status") == "completed" else 0.0
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("scores must be finite numbers in [0,1]")
            result[key] = (score, row.get("fixture_hash"), row.get("category"), row.get("split"))
        return result
    if type(resamples) is not int or resamples < 100 or not 0 <= maximum_loss <= 1:
        raise ValueError("at least 100 resamples and a valid loss tolerance are required")
    left, right = index(baseline), index(candidate)
    if not left or set(left) != set(right):
        raise ValueError("paired comparisons require identical nonempty task IDs, including failures")
    for key in left:
        if left[key][1:] != right[key][1:] or not left[key][1]:
            raise ValueError("task fixture/category/split provenance does not match")
    deltas = [right[key][0] - left[key][0] for key in sorted(left)]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples))
    bootstrap_interval = [boot[int(.025 * resamples)], boot[min(resamples - 1, int(.975 * resamples))]]
    if all(delta in {-1, 0, 1} for delta in deltas):
        # A collapsed empirical bootstrap cannot exclude rare unseen regressions.
        # Bound gain/loss probabilities separately with exact binomial bounds.
        # Bonferroni allocation gives each directional delta bound >=95% coverage.
        gains, losses = deltas.count(1), deltas.count(-1)
        n = len(deltas)
        lower = _binomial_lower(gains, n, .025) - _binomial_upper(losses, n, .025)
        upper = _binomial_upper(gains, n, .025) - _binomial_lower(losses, n, .025)
        method = "paired-discordance-exact-binomial-bounds"
    else:
        # For arbitrary bounded partial-credit deltas, use distribution-free bounds.
        radius = math.sqrt(2 * math.log(2 / .05) / len(deltas))
        lower = max(-1., statistics.mean(deltas) - radius)
        upper = min(1., statistics.mean(deltas) + radius)
        method = "paired-bounded-hoeffding"
    status = "inconclusive"
    if len(deltas) >= minimum_tasks:
        if lower >= -maximum_loss:
            status = "noninferior"
        elif upper < -maximum_loss:
            status = "degraded"
    return {"tasks": len(deltas), "mean_delta": statistics.mean(deltas), "lower_95": lower,
            "upper_95": upper, "maximum_loss": maximum_loss, "status": status,
            "method": method, "bootstrap_diagnostic_95": bootstrap_interval,
            "seed": seed, "resamples": resamples,
            "smallest_demonstrable_loss": smallest_demonstrable_loss(len(deltas)),
            "tasks_needed_for_maximum_loss": tasks_needed_for_loss(maximum_loss)}


def smallest_demonstrable_loss(tasks: int, alpha: float = .025) -> float | None:
    """The best case: with NO regression on ``tasks`` pass/fail items, the largest loss that still cannot be
    excluded. Non-inferiority within a tolerance below this is unreachable at this sample size, whatever the
    candidate does - 0.14 at 24 items, 0.06 at 60, 0.02 at 183."""
    return None if tasks < 1 else 1 - alpha ** (1 / tasks)


def tasks_needed_for_loss(maximum_loss: float, alpha: float = .025) -> int | None:
    """How many regression-free pass/fail items it takes before ``maximum_loss`` can be demonstrated at all."""
    if not 0 < maximum_loss < 1:
        return None
    return math.ceil(math.log(alpha) / math.log(1 - maximum_loss))


def _binomial_upper(successes, total, alpha):
    if successes == total:
        return 1.
    if successes == 0:
        return 1 - alpha ** (1 / total)
    lo, hi = 0., 1.
    for _ in range(55):
        p = (lo + hi) / 2
        terms = [math.lgamma(total + 1) - math.lgamma(i + 1) - math.lgamma(total - i + 1)
                 + i * math.log(p) + (total - i) * math.log1p(-p) for i in range(successes + 1)]
        peak = max(terms)
        cdf = math.exp(peak) * sum(math.exp(term - peak) for term in terms)
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return hi


def _binomial_lower(successes, total, alpha):
    return 1 - _binomial_upper(total - successes, total, alpha)


@dataclass
class BudgetTracker:
    limits: Budgets
    clock: object = time.monotonic
    started: float = field(init=False)
    candidates: int = 0

    def __post_init__(self):
        self.started = self.clock()

    def remaining(self):
        return max(0, self.limits.wall_seconds - (self.clock() - self.started))

    def admit(self, estimated_seconds: float, *, validation=False):
        if not math.isfinite(estimated_seconds) or estimated_seconds <= 0:
            raise ValueError("positive bounded duration estimate required")
        reserve = 0 if validation else self.limits.reserve_validation_seconds
        if estimated_seconds > self.remaining() - reserve:
            return False
        if not validation:
            if self.candidates >= self.limits.max_candidates:
                return False
            self.candidates += 1
        return True


def propose(base: ExperimentManifest, changes: dict, family: str) -> ExperimentManifest:
    """Bound supervisor changes to a declared causal axis; never accept arbitrary commands."""
    axes = {
        "kv": {"backend.k_cache", "backend.v_cache"},
        "context": {"backend.context_length", "requested_input_tokens"},
        "performance": {"backend.gpu_offload", "backend.batch_size", "backend.cpu_threads",
                        "backend.flash_attention", "backend.mtp", "backend.parallel", "backend.kv_on_gpu",
                        "backend.ubatch_size"},
        "weights": {"model"},
        "reasoning": {"backend.reasoning"},
    }
    # A combined serving treatment keeps weights, context, generation, tasks and provenance fixed.
    axes["combination"] = axes["kv"] | axes["performance"] | axes["reasoning"]
    if family not in axes or not changes or set(changes) - axes[family]:
        raise ValueError("proposal changes fields outside the declared experiment family")
    payload = base.model_dump(mode="json")
    for path, value in changes.items():
        parts = path.split(".")
        if len(parts) == 1:
            payload[path] = value
        else:
            payload[parts[0]][parts[1]] = value
    proposed = ExperimentManifest.model_validate_json(canonical_json(payload))
    if proposed.fingerprint() == base.fingerprint():
        raise ValueError("proposal duplicates the baseline")
    return proposed


QUALITY_CATEGORIES = ("coding", "tools", "retrieval")


def unmeasured_categories(quality: dict) -> list[str]:
    """Categories for which the run declared no task at all (``attempted`` is 0 or absent).

    A declared task that crashed or was cut off is still ATTEMPTED: it stays in the denominator and can fail the
    candidate. Only a category nobody asked for is unmeasured. It is never treated as passed - it is reported, and
    a validated-preset export still refuses a candidate that has one (``reports._quality_reasons``).
    """
    def measured(block) -> bool:  # a task count, or a score someone actually computed
        return isinstance(block, dict) and (bool(block.get("attempted")) or (
            isinstance(block.get("score"), (int, float)) and not isinstance(block.get("score"), bool)))
    return [name for name in QUALITY_CATEGORIES if not measured(quality.get(name))]


def assess_candidate(evidence: dict, policy: CampaignPolicy) -> dict:
    reasons = []
    if evidence.get("synthetic") is not False:
        reasons.append("synthetic_or_unknown_origin")
    if evidence.get("status") != "completed":
        reasons.append("incomplete")
    if policy.acceptance.require_effective_settings_verified and evidence.get("effective_settings_verified") is not True:
        reasons.append("settings_unverified")
    if policy.acceptance.require_actual_context_verified and evidence.get("actual_context_verified") is not True:
        reasons.append("context_unverified")
    if policy.acceptance.require_holdout and evidence.get("holdout_passed") is not True:
        reasons.append("holdout_unverified")
    speed = evidence.get("speed", {})
    if speed.get("qualifies") is not True:
        reasons.append("speed_gate_failed")
    thresholds = {"coding": policy.acceptance.minimum_coding_score,
                  "tools": policy.acceptance.minimum_tool_score,
                  "retrieval": policy.acceptance.minimum_retrieval_score}
    quality = evidence.get("quality", {})
    unmeasured = unmeasured_categories(quality)
    for category, threshold in thresholds.items():
        if category in unmeasured:  # nothing was declared for it: not measured, so not judged - and said so below
            continue
        row = quality.get(category, {})
        score = row.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not threshold <= score <= 1:
            reasons.append(f"{category}_absolute_quality_failed")
        if row.get("comparison", {}).get("status") != "noninferior":
            reasons.append(f"{category}_quantization_loss_unverified")
    if len(unmeasured) == len(QUALITY_CATEGORIES):
        reasons.append("no_quality_measured")  # speed alone never makes a candidate eligible
    return {"eligible": not reasons, "reasons": reasons, "unmeasured_categories": unmeasured}


def pareto_frontier(candidates: list[dict]) -> list[dict]:
    """Keep tradeoffs across speed, tested context and each independent quality category."""
    eligible = [row for row in candidates if row.get("eligibility", {}).get("eligible") is True]
    def values(row):  # an unmeasured category is not a dimension anything can win or lose on
        return (row["speed"]["minimum_native_tps"], row["actual_input_tokens"],
                *(row["quality"][key]["score"] for key in QUALITY_CATEGORIES
                  if key not in unmeasured_categories(row["quality"])))
    result = []
    for row in eligible:
        own = values(row)
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in own):
            raise ValueError("frontier requires complete finite measurements")
        dominated = any(all(b >= a for a, b in zip(own, values(other)))
                        and any(b > a for a, b in zip(own, values(other))) for other in eligible)
        if not dominated:
            result.append(row)
    return result
