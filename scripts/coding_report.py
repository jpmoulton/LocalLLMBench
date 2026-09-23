"""Coding comparison across the sessions of one sweep: pass@1 per candidate, truncation, and PAIRED differences.

Two rules learned the hard way in this project are built in rather than left to the reader:

* A pass rate is printed next to its ``finish_reason`` counts. An item cut off at the output cap is reported as
  ``cut off``, never silently folded into "failed": twice a cap has manufactured a quality finding here.
* Configurations and models are compared on the IDENTICAL item set with an exact McNemar test on the discordant
  pairs. "Pass rate among the items that finished" is never used - that is survivorship bias, because the items a
  configuration fails to finish are the hard ones.
* A suite that could not run because the Docker sandbox was unavailable (a Mac without Colima, say) is printed as
  ``blocked`` with its reason, never as a pass rate of zero: its rows are backfilled ``environment_error``s, not
  answers the model got wrong, and they never enter a paired comparison. The reason comes from the session's plan
  (``benchmark-selection.json``, when derivation already knew) or from the candidate's own evaluation record
  (when the sandbox probe failed at run time). Generated code is never executed on the host instead.
* A candidate with no items for the suite, or with no timed quality stage, prints ``-`` there rather than
  dividing by zero or formatting a missing number.

Usage: python scripts/coding_report.py --root <sweep directory> [--suite evalplus|aider-polyglot]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# container_eval's reason prefix for the rows and benchmark record of a suite whose sandbox was unavailable.
SANDBOX_UNAVAILABLE = "sandbox_unavailable"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_candidate(directory: Path, suite: str) -> dict | None:
    result, evaluation = directory / "result.json", directory / "evaluator" / "evaluation.json"
    if not result.is_file() or not evaluation.is_file():
        return None
    outcome = json.loads(result.read_text(encoding="utf-8"))
    evaluated = json.loads(evaluation.read_text(encoding="utf-8"))
    samples = evaluated.get("samples") or []
    items = {row["task_id"]: row for row in samples if row.get("suite") == suite}
    finish: dict[str, int] = {}
    tokens = []
    # The adapter's own per-item response files only (``response.json`` / ``response-<attempt>.json``). The
    # ``request-N.json`` transport captures beside them hold the same completions, so a broader glob counts every
    # response twice - which doubled the cut-off count the first time this was generalised.
    for path in (directory / "evaluator" / "benchmarks" / suite).rglob("response*.json"):
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(response, dict) or not response.get("choices"):
            continue
        reason = response["choices"][0].get("finish_reason")
        finish[reason] = finish.get(reason, 0) + 1
        tokens.append((response.get("usage") or {}).get("completion_tokens") or 0)
    stages = {stage["name"]: stage["elapsed_seconds"] for stage in outcome.get("stages", [])}
    speed = (outcome.get("speed") or {}).get("median_native_tps")
    return {"label": directory.name.split("-", 1)[1], "state": outcome.get("state"), "items": items,
            "finish": finish, "tokens": tokens, "quality_seconds": stages.get("quality"), "tps": speed,
            "vram": outcome.get("vram_used_mib_after_load"),
            "environment_errors": sum(1 for row in items.values() if row.get("status") == "environment_error"),
            "runtime": outcome.get("runtime", "nvidia-container"),
            "blocked": sandbox_blocked(evaluated, suite, items)}


def sandbox_blocked(evaluation: dict, suite: str, items: dict) -> str | None:
    """The reason ``suite`` could not run in this candidate because its Docker sandbox was unavailable, or None.

    Evidence only: the evaluator's benchmark record for the suite says ``environment_error`` with a
    ``sandbox_unavailable: ...`` reason, or every one of the suite's rows does. A suite with some real answers
    is not blocked, however many environment errors it also has."""
    for record in evaluation.get("benchmarks") or []:
        if (isinstance(record, dict) and record.get("benchmark_id") == suite
                and record.get("status") == "environment_error"
                and str(record.get("reason") or "").startswith(SANDBOX_UNAVAILABLE)):
            return str(record["reason"])
    reasons = {str(row.get("reason") or "") for row in items.values()}
    if items and all(row.get("status") == "environment_error" for row in items.values()) \
            and all(reason.startswith(SANDBOX_UNAVAILABLE) for reason in reasons):
        return sorted(reasons)[0]
    return None


def plan_blocked(root: Path, slug: str, suite: str) -> str | None:
    """The reason the session's own plan blocked ``suite`` (``derive`` found no usable sandbox), or None. A plan
    that is absent or unreadable blocks nothing: this reports the plan's words, it never infers them."""
    try:
        plan = json.loads((root / slug / "run" / "benchmark-selection.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for row in (plan.get("benchmarks") or []) if isinstance(plan, dict) else []:
        if isinstance(row, dict) and row.get("benchmark_id") == suite and row.get("status") == "blocked":
            return str(row.get("reason") or "blocked by the session plan")
    return None


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def paired(left: dict, right: dict) -> dict:
    shared = sorted(set(left["items"]) & set(right["items"]))
    b = sum(1 for t in shared if left["items"][t]["passed"] and not right["items"][t]["passed"])
    c = sum(1 for t in shared if not left["items"][t]["passed"] and right["items"][t]["passed"])
    return {"n": len(shared), "left_only": b, "right_only": c, "p": mcnemar_exact(b, c)}


def collect(root: Path, suite: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """``(candidates per model, plan-blocked reason per model)`` for one sweep directory."""
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    models, blocked = {}, {}
    for slug in index:
        candidates = [load_candidate(d, suite) for d in sorted((root / slug / "run" / "candidates").glob("*"))
                      if d.is_dir()] if (root / slug / "run" / "candidates").is_dir() else []
        models[slug] = [c for c in candidates if c]
        reason = plan_blocked(root, slug, suite)
        if reason is not None:
            blocked[slug] = reason
    return models, blocked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--suite", default="evalplus", choices=("evalplus", "aider-polyglot"))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    models, blocked = collect(root, args.suite)
    text = render(models, args.suite, blocked)
    (root / f"coding-report-{args.suite}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def render(models: dict[str, list[dict]], suite: str, blocked: dict[str, str] | None = None) -> str:
    """The report text. ``blocked`` maps a model to the reason its plan blocked ``suite``."""
    blocked = blocked or {}
    polyglot = suite == "aider-polyglot"
    title = ("Aider Polyglot, python + javascript development split, pass within two attempts" if polyglot
             else "EvalPlus MBPP+, pass@1")
    out = [f"# Coding comparison ({title})", ""]
    if polyglot:
        out += ["Upstream's protocol: the model edits the stub with search/replace blocks; on failure it sees the test "
                "output once and tries again. `first try` is upstream's pass_rate_1, `pass` its pass_rate_2. "
                "`javascript/ledger` is a refactoring exercise whose stub already passes - a free point by design.", ""]
    extra = " first try | well-formed edits | python | javascript |" if polyglot else ""
    out += [f"| Model | Candidate | pass | 95% CI |{extra} cut off at cap | env errors | median tokens | quality stage |",
            "|---|---|---|---|" + ("---|---|---|---|" if polyglot else "") + "---|---|---|---|"]
    trailing = 9 if polyglot else 5  # cells after Model, Candidate and pass
    for slug, candidates in models.items():
        if not candidates:
            out.append(f"| `{slug}` | *(not run)* |" + (f" blocked: {_cell(blocked[slug])} |" + " - |" * trailing
                                                         if slug in blocked else " - |" * (10 if polyglot else 6)))
        for c in candidates:
            reason = c.get("blocked") or (blocked.get(slug) if not c["items"] else None)
            if reason:  # the sandbox could not run the suite: no pass rate exists, not even zero
                out.append(f"| `{slug}` | {c['label']} | blocked: {_cell(reason)} |" + " - |" * trailing)
                continue
            rows = list(c["items"].values())
            n = len(rows)
            quality = "-" if c["quality_seconds"] is None else f"{c['quality_seconds']:.0f} s"
            if n == 0:  # nothing of this suite ran: no rate and no interval, never a division by zero
                out.append(f"| `{slug}` | {c['label']} | 0/0 (no items) |" + " - |" * (trailing - 1)
                           + f" {quality} |")
                continue
            passed = sum(1 for row in rows if row["passed"])
            low, high = wilson(passed, n)
            tokens = sorted(c["tokens"])
            cells = ""
            if polyglot:
                def language(name: str) -> str:
                    sub = [row for row in rows if row["task_id"].split("/")[1] == name]
                    return f"{sum(1 for row in sub if row['passed'])}/{len(sub)}"
                first = sum(1 for row in rows if row.get("attempt_1_passed"))
                formed = sum(1 for row in rows if row.get("well_formed"))
                cells = f" {first}/{n} = {first / n:.1%} | {formed}/{n} | {language('python')} | {language('javascript')} |"
            out.append(f"| `{slug}` | {c['label']} | **{passed}/{n} = {passed / n:.1%}** | {low:.0%}-{high:.0%} |{cells} "
                       f"{c['finish'].get('length', 0)} | {c['environment_errors']} | "
                       f"{tokens[len(tokens) // 2] if tokens else '-'} | {quality} |")
    # Only candidates that actually answered items are paired: a blocked or empty suite has nothing to compare.
    comparable = {slug: [c for c in candidates if c["items"] and not c.get("blocked")]
                  for slug, candidates in models.items()}
    out += ["", "## Paired comparisons on identical items (exact McNemar)", "",
            "| Comparison | n | left only | right only | p |", "|---|---|---|---|---|"]
    for slug, candidates in comparable.items():
        by = {c["label"]: c for c in candidates}
        if "baseline" in by and "reasoning-on" in by:
            r = paired(by["baseline"], by["reasoning-on"])
            out.append(f"| `{slug}`: reasoning off vs on | {r['n']} | {r['left_only']} | {r['right_only']} | {r['p']:.3f} |")
    slugs = [s for s, c in comparable.items() if any(x["label"] == "baseline" for x in c)]
    for i, left in enumerate(slugs):
        for right in slugs[i + 1:]:
            a = next(x for x in comparable[left] if x["label"] == "baseline")
            b = next(x for x in comparable[right] if x["label"] == "baseline")
            r = paired(a, b)
            out.append(f"| `{left}` vs `{right}` (reasoning off) | {r['n']} | {r['left_only']} | {r['right_only']} | "
                       f"{r['p']:.3f} |")
    out += ["", "A p-value near 1 means the two columns are indistinguishable on this item set; with a few dozen items "
                "a true difference smaller than roughly 15-20 points will usually not reach significance.", ""]
    if blocked or any(c.get("blocked") for candidates in models.values() for c in candidates):
        out += ["`blocked` means the Docker sandbox that runs generated code was unavailable, so the suite was not "
                "run: it is not a pass rate of zero and it is left out of every paired comparison. Generated code is "
                "never executed on the host instead.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
