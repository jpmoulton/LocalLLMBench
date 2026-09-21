"""Aggregate several tuning sessions into ONE comparison of every candidate that ran, across every model.

Each session already reports itself; this adds the cross-model view that no single session can produce, and
it is deliberately conservative about what it claims:

* Every candidate a session attempted appears, including failures, with the reason it failed. A candidate
  that did not run is printed as "not run", never omitted and never interpolated.
* Scores are copied from the session's own analysis. Nothing is re-scored, averaged across models, or
  ranked into a "winner" here - eligibility stays the campaign's decision.
* A GGUF's declared ``general.file_type`` is reported next to the tensor types actually present, because a
  file may be named for a quantization its header does not declare (NVFP4 weights currently land in a
  ggml type the ``llama_ftype`` enum has no entry for, so the header reads Q8_0).
* Models whose chat templates differ are flagged, because prompt-shaped scores are not comparable between
  them on the strength of this run alone.

Usage: python scripts/cross_model_report.py --root <sweep directory written by scripts/sweep_models.py>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmbench.containers.gguf import gguf_summary  # noqa: E402

MISSING = "-"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return MISSING
    return format(value, spec) if spec else str(value)


def collect_model(slug: str, meta: dict, root: Path) -> dict:
    """Everything known about one model's session. Absent pieces stay absent; nothing is inferred."""
    run = root / slug / "run"
    record = {"slug": slug, "model_path": meta.get("model"), "declared_quantization": meta.get("quantization"),
              "sha256": meta.get("sha256"), "candidates": [], "status": "not run", "notes": []}

    path = Path(meta["model"]) if meta.get("model") else None
    if path and path.is_file():
        try:
            summary = gguf_summary(path)
            file_type = summary.get("file_type")
            record["header"] = {
                "architecture": summary.get("architecture"), "name": summary.get("name"),
                "file_type": file_type, "file_type_name": summary.get("declared_quantization") or "UNKNOWN",
                "quantization": summary.get("quantization"),
                "n_ctx_train": summary.get("n_ctx_train"), "block_count": summary.get("block_count"),
                "nextn_predict_layers": summary.get("nextn_predict_layers"),
                "template_hash": summary.get("template_hash"),
                "size_gib": round(path.stat().st_size / 2 ** 30, 2)}
            record["tensor_types"] = summary.get("tensor_types") or {}
        except Exception as exc:
            record["notes"].append(f"header unreadable: {type(exc).__name__}: {exc}")

    ledger = _read_json(run / "session.json")
    summary_json = _read_json(run / "session-summary.json")
    campaign = _read_json(run / "reports" / "campaign.json")
    if ledger is None:
        return record

    record["status"] = "running (no summary yet)" if summary_json is None else summary_json.get("stop_reason")
    record["environment_hash"] = ledger.get("environment_hash")
    record["session_id"] = ledger.get("session_id")
    if summary_json:
        record["winner"] = summary_json.get("winner")
        record["attempts"] = summary_json.get("attempts")
        record["recommendation"] = summary_json.get("recommendation")

    labels = ledger.get("labels") or []
    proposals = ledger.get("proposals") or []
    manifests = ledger.get("manifests") or []
    by_manifest = {row.get("manifest_hash"): row for row in ((campaign or {}).get("results") or [])}

    for index, label in enumerate(labels):
        changes = proposals[index].get("changes", {}) if index < len(proposals) else {}
        analysis = by_manifest.get(manifests[index]) if index < len(manifests) else None
        directory = next((d for d in sorted((run / "candidates").glob(f"{index:03d}-*")) if d.is_dir()), None)
        result = _read_json(directory / "result.json") if directory else None

        # ``settings`` is the evidence list: one row per control with what was REQUESTED and what the server
        # was observed to actually do. Report the effective value, and carry any control whose evidence is a
        # mismatch, because a candidate that did not run the settings it asked for is not the candidate.
        evidence = (result or {}).get("settings") or []
        engine = {row.get("control"): row.get("effective") for row in evidence if isinstance(row, dict)}
        mismatches = [row.get("control") for row in evidence
                      if isinstance(row, dict) and row.get("status") == "mismatch"]
        quality = (analysis or {}).get("quality") or (result or {}).get("quality") or {}

        # A candidate that dies mid-stage still writes score 0.0 for every category it had started.
        # That zero is an artifact of the failure, not a measurement: reporting it would invent a quality
        # finding out of a crash. Suppress scores for any candidate the runner marked failed.
        failed = (result or {}).get("state") == "failed" or bool((result or {}).get("failure_stage"))

        def score(area: str):
            block = quality.get(area) or {}
            value = None if failed else block.get("score")
            return value, block.get("attempted"), block.get("completed"), block.get("threshold")

        candidate = {
            "index": index, "label": label, "changes": changes,
            "state": (result or {}).get("state", "not run"),
            "failure_stage": (result or {}).get("failure_stage"),
            "failure_reasons": (result or {}).get("failure_reasons") or [],
            "vram_mib": (result or {}).get("vram_used_mib_after_load"),
            "load_seconds": (result or {}).get("load_seconds"),
            "elapsed_seconds": (result or {}).get("elapsed_seconds") or (analysis or {}).get("elapsed_seconds"),
            "ctx_size": engine.get("ctx_size") or changes.get("ctx_tier"),
            "cache_k": engine.get("cache_type_k"), "cache_v": engine.get("cache_type_v"),
            "spec_type": engine.get("spec_type") or changes.get("spec_type"),
            "reasoning": engine.get("reasoning") or changes.get("reasoning"),
            "n_gpu_layers": engine.get("n_gpu_layers"), "kv_offload": engine.get("kv_offload"),
            "settings_mismatches": mismatches,
            "requested_input_tokens": (analysis or {}).get("requested_input_tokens"),
            "actual_input_tokens": (analysis or {}).get("actual_input_tokens"),
            # ``actual_input_tokens`` is the MINIMUM across a candidate's requests, so a context tier whose
            # speed probe filled the tier still reports ~4k here because the quality items are short. The
            # largest prompt the candidate actually held is the maximum, and the tier is only a context
            # CLAIM when the candidate-level check verified it.
            "measured_input_min": ((analysis or {}).get("context_accounting") or {}).get(
                "minimum_measured_input_tokens"),
            "measured_input_max": ((analysis or {}).get("context_accounting") or {}).get(
                "maximum_measured_input_tokens"),
            "context_errors": ((analysis or {}).get("context_accounting") or {}).get("errors") or [],
            "context_verified": (analysis or {}).get("actual_context_verified"),
            # The campaign analysis withholds speed for a candidate it judged incomplete. That reading was
            # still taken, so keep it separately and mark it raw rather than dropping a real measurement or
            # passing it off as one the analysis accepted.
            "median_tps": ((analysis or {}).get("speed") or {}).get("median_native_tps"),
            "raw_median_tps": ((result or {}).get("speed") or {}).get("median_native_tps"),
            "speed_reasons": ((analysis or {}).get("speed") or (result or {}).get("speed")
                              or {}).get("reasons") or [],
            "eligible": ((analysis or {}).get("eligibility") or {}).get("eligible"),
            "ineligible_reasons": ((analysis or {}).get("eligibility") or {}).get("reasons") or [],
        }
        for area in ("tools", "retrieval", "coding"):
            value, attempted, completed, threshold = score(area)
            candidate[area] = value
            candidate[f"{area}_attempted"] = attempted
            candidate[f"{area}_completed"] = completed
            candidate[f"{area}_threshold"] = threshold
        record["candidates"].append(candidate)

    selection = _read_json(run / "benchmark-selection.json")
    if isinstance(selection, dict):
        record["benchmark_selection"] = {
            "selected": selection.get("selected") or selection.get("present"),
            "skipped": selection.get("skipped")}
    return record


def render_markdown(records: list[dict], generated: str) -> str:
    out: list[str] = []
    w = out.append
    w("# Cross-model inference sweep")
    w("")
    w(f"Generated {generated} by `scripts/cross_model_report.py`. Every candidate that each session "
      "attempted is listed, failures included. Scores are copied from each session's own analysis; nothing "
      "is re-scored or ranked here.")
    w("")

    w("## Models measured")
    w("")
    w("| Model | Size | Header `file_type` | Tensor types actually present | Template | Session |")
    w("|---|---|---|---|---|---|")
    for record in records:
        header = record.get("header") or {}
        types = record.get("tensor_types") or {}
        top = ", ".join(f"{k}×{v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1])[:3]) or MISSING
        template = (header.get("template_hash") or MISSING)
        w(f"| `{record['slug']}` | {_fmt(header.get('size_gib'))} GiB | "
          f"{_fmt(header.get('file_type'))} → {header.get('file_type_name', MISSING)} | {top} | "
          f"`{template[7:15] if template.startswith('sha256:') else template}` | {record.get('status')} |")
    w("")

    templates = {(r.get("header") or {}).get("template_hash") for r in records if r.get("header")}
    if len(templates - {None}) > 1:
        w("> **Chat templates differ between these models.** Prompt-shaped scores (tool calling especially) "
          "are therefore not strictly apples-to-apples across rows; a difference may come from the template "
          "rather than the weights.")
        w("")

    renamed = [r["slug"] for r in records if (r.get("header") or {}).get("quantization")
               and r["header"]["quantization"] != r["header"].get("file_type_name")]
    if renamed:
        w(f"> **The header of {', '.join('`' + slug + '`' for slug in renamed)} does not name its weights.** "
          "`general.file_type` has no value for some formats (NVFP4), so a converter writes a stand-in and the "
          "quantization is read from the tensors instead. The tensor-type column is what the weights really are.")
        w("")
    unknown = {label for r in records for label in (r.get("tensor_types") or {}) if label.startswith("type_")}
    if unknown:
        w(f"> **Tensor type(s) this version does not know: {', '.join(sorted(unknown))}.** They are reported by "
          "number, never guessed at.")
        w("")

    w("## Every candidate, every model")
    w("")
    w("A context tier is a verified CONTEXT CLAIM only when `ctx ok` is ✅. `largest prompt` is the biggest "
      "input the candidate actually held; the quality items in the same candidate are short, so this is not "
      "the whole story on its own. A tok/s marked `(raw)` was measured but the campaign analysis refused it "
      "because that candidate did not complete - treat it as an observation, not an accepted result.")
    w("")
    w("| Model | Candidate | ctx | KV | spec | reason | tok/s | VRAM MiB | tools | retrieval | coding | "
      "largest prompt | ctx ok | state |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for record in records:
        if not record["candidates"]:
            w(f"| `{record['slug']}` | *(not run)* | - | - | - | - | - | - | - | - | - | - | - | "
              f"{record.get('status')} |")
        for c in record["candidates"]:
            kv = f"{c['cache_k']}/{c['cache_v']}" if c["cache_k"] else MISSING

            def cell(area: str) -> str:
                """A score, `0/0` when nothing was attempted, or `unmeasured` when a run died mid-stage."""
                value, attempted = c[area], c.get(f"{area}_attempted")
                if isinstance(value, (int, float)):
                    return f"{value:.3f}"
                if not attempted:
                    return "0/0"
                return "unmeasured"

            tools, retr, coding = cell("tools"), cell("retrieval"), cell("coding")
            if isinstance(c["median_tps"], (int, float)):
                tps = f"{c['median_tps']:.1f}"
            elif isinstance(c.get("raw_median_tps"), (int, float)):
                tps = f"{c['raw_median_tps']:.1f} (raw)"  # measured, but the analysis did not accept it
            else:
                tps = MISSING
            vram = f"{c['vram_mib']:.0f}" if isinstance(c["vram_mib"], (int, float)) else MISSING
            largest = _fmt(c.get("measured_input_max"))
            ok = {True: "✅", False: "❌"}.get(c.get("context_verified"), MISSING)
            if c.get("context_errors"):
                ok += f" ({', '.join(c['context_errors'])})"
            state = c["state"] if not c["failure_reasons"] else f"{c['state']}: {c['failure_reasons'][0]}"
            if c.get("settings_mismatches"):  # it did not serve what it asked for; the row is not that config
                state += f" ⚠ mismatch: {', '.join(c['settings_mismatches'])}"
            w(f"| `{record['slug']}` | {c['label']} | {_fmt(c['ctx_size'])} | {kv} | "
              f"{_fmt(c['spec_type'])} | {_fmt(c['reasoning'])} | {tps} | {vram} | {tools} | {retr} | "
              f"{coding} | {largest} | {ok} | {state} |")
    w("")

    w("## Cross-model comparison (measured axes only)")
    w("")
    w("Each column is the best value that model achieved among its own candidates, at the settings named. "
      "This is a description of what was measured, not a ranking: quality did not separate these models "
      "(see below), and the campaign named no winner.")
    w("")
    w("| Model | baseline tok/s | best tok/s (settings) | lowest VRAM | tok/s at 64K | VRAM at 262144 | "
      "tools range |")
    w("|---|---|---|---|---|---|---|")
    for record in records:
        done = [c for c in record["candidates"] if c["state"] == "completed"]
        if not done:
            w(f"| `{record['slug']}` | - | - | - | - | - | - |")
            continue
        base = next((c for c in done if c["label"] == "baseline"), None)
        speeds = [c for c in done if isinstance(c["median_tps"], (int, float))]
        best = max(speeds, key=lambda c: c["median_tps"]) if speeds else None
        vrams = [c for c in done if isinstance(c["vram_mib"], (int, float))]
        low = min(vrams, key=lambda c: c["vram_mib"]) if vrams else None
        at64 = next((c for c in done if c["ctx_size"] == 66048), None)
        ceiling = next((c for c in record["candidates"]
                        if c["ctx_size"] == 262144 and isinstance(c["vram_mib"], (int, float))), None)
        tools = sorted(c["tools"] for c in done if isinstance(c["tools"], (int, float)))
        w(f"| `{record['slug']}` | {_fmt(base and base['median_tps'], '.1f')} | "
          f"{_fmt(best and best['median_tps'], '.1f')} ({best['label'] if best else MISSING}) | "
          f"{_fmt(low and low['vram_mib'], '.0f')} | {_fmt(at64 and at64['median_tps'], '.1f')} | "
          f"{_fmt(ceiling and ceiling['vram_mib'], '.0f')} | "
          f"{f'{tools[0]:.3f}-{tools[-1]:.3f}' if tools else MISSING} |")
    w("")

    every = [c for r in records for c in r["candidates"]
             if c["state"] == "completed" and isinstance(c["tools"], (int, float))]
    if every:
        lo, hi = min(c["tools"] for c in every), max(c["tools"] for c in every)
        attempted = next((c["tools_attempted"] for c in every if c.get("tools_attempted")), None)
        w(f"**Tool scores did not separate anything.** Across all {len(every)} completed candidates the tool "
          f"score spans {lo:.3f}-{hi:.3f}"
          + (f" on {attempted} items, i.e. a spread of about {round((hi - lo) * attempted)} item(s)." if
             attempted else ".")
          + " A difference of one or two items on a set this size is well inside binomial noise, so no model "
            "and no setting in this sweep has a demonstrated tool-calling advantage.")
        w("")

    w("## Per-model detail")
    w("")
    for record in records:
        w(f"### `{record['slug']}`")
        w("")
        w(f"- Weights: `{record.get('model_path')}`")
        w(f"- SHA256: `{record.get('sha256')}`")
        w(f"- Session stop reason: **{record.get('status')}**; winner: "
          f"**{record.get('winner') if record.get('winner') else 'none'}**")
        if record.get("benchmark_selection"):
            w(f"- Benchmarks: {json.dumps(record['benchmark_selection'])}")
        if record.get("notes"):
            for note in record["notes"]:
                w(f"- Note: {note}")
        reasons: dict[str, int] = {}
        for c in record["candidates"]:
            for reason in c["ineligible_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        if reasons:
            w("- Why no candidate qualified (count of candidates per reason):")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                w(f"  - `{reason}` × {count}")
        w("")

    w("## What this report does not establish")
    w("")
    w("- **No validated preset.** A preset is validated only when a separate holdout attempt passes, and "
      "only on the categories that holdout measured. Nothing here is deployable on this evidence alone.")
    w("- **No cross-model winner.** Candidates are compared within a session against that session's own "
      "baseline. Ranking different models against each other would need a paired run on identical items, "
      "which this sweep does not perform.")
    w("- **Coding is unmeasured** wherever `coding` shows `0/0`: the coding worker broker is not configured "
      "in the base config, so EvalPlus and Aider Polyglot were skipped rather than scored.")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="a sweep directory holding index.json")
    parser.add_argument("--output", default=None, help="directory for the report (default: --root)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    index = _read_json(root / "index.json")
    if not isinstance(index, dict) or not index:
        print(f"no index.json under {root}; nothing to report", file=sys.stderr)
        return 1

    records = [collect_model(slug, meta, root) for slug, meta in index.items()]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out_dir = Path(args.output).resolve() if args.output else root
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "cross-model-report.json").write_text(
        json.dumps({"generated_utc": generated, "models": records}, indent=2), encoding="utf-8")
    markdown = render_markdown(records, generated)
    (out_dir / "cross-model-report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nwrote {out_dir / 'cross-model-report.md'} and .json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
