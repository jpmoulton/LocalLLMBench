# Development

```bash
pip install -e ".[eval,dev]"
pytest                      # ~1,500 tests, about 3 minutes; no GPU, Docker, model or network is touched
ruff check src tests scripts
```

The whole suite runs against fakes and recorded fixtures (`tests/data/` holds real llama.cpp `/props`, `/slots`,
route and help captures), so it is safe on any machine and runs in CI on Linux, macOS and Windows. Anything that
needs a GPU is a script under `scripts/`, refuses to run without `runtime-policy.json`, and is never part of
`pytest`.

## Layout

```text
src/llmbench/
  containers/        the product: config, plan (Compose + server argv), runner, session (tune/resume), derive
                     (model -> search space + benchmark plan), proposals, readback (settings evidence), reports, cli
  container_eval.py  the evaluator: speed probes, local fixtures, public benchmark dispatch
  backends/          llama.cpp server adapter, bounded HTTP/SSE transport, speed-probe bridge
  benchmarks/        bfcl, ruler (+ ruler_tasks/), evalplus, aider_polyglot: one adapter each, one shared contract
  coding/            sandbox (DockerWorker), host broker, spool protocol, container drivers, benchmark bridge
  evaluations/       local tool / retrieval fixtures, strict tool-call capture, Inspect tasks
  analysis.py, search.py, controller.py, store.py, reports.py   campaign loop, eligibility, SQLite evidence, reports
  registry.py        which benchmarks exist, what each needs (dataset, broker), which options each accepts
scripts/             dataset staging, image build, multi-model sweeps, reports, calibration, server diagnostics
examples/            candidate and session configs (examples/candidate.json is the packaged tune template)
```

## Conventions that matter

* **Strict models everywhere.** Configs are frozen pydantic models that reject unknown fields. Round-trip them with
  `canonical_json(...)` / `model_validate_json(...)`; plain `model_validate(dict)` rejects JSON lists for tuples.
* **Fingerprints are identity.** Anything that changes a measurement belongs in the candidate fingerprint. Adding a
  field with a default to a fingerprinted model changes every fingerprint that serialises it (and the golden
  Compose fixture).
* **Fail closed, and say why.** A refusal names its reason; a missing dataset is a recorded skip, not a crash and
  not a silent omission.
* **Never turn a harness failure into a score.** Rows that did not really run are `environment_error`.
* Do not edit `src/` while a session is running: the source hash is part of `resume`'s identity check.

## Adding a benchmark

1. Write an adapter in `benchmarks/` implementing `BenchmarkAdapter` (`available`, `task_ids`, `run`): one row per
   declared task, raw responses persisted before parsing, no network, no LLM judge.
2. Register it in `registry.py` (category, `requires`, `option_keys`, `holdout_capable`) and wire its options in
   `container_eval.BENCHMARK_WIRING`.
3. Give derivation a default option set and a per-item cost in `containers/derive.py`, or it is offered and skipped
   with a stated reason.
4. Pin its data in `scripts/stage_benchmark_datasets.py`.
5. If it executes code, route it through the broker (`coding/benchmark_items.py`) with host-owned tests, and
   calibrate the harness against known-good and known-bad solutions **in the real sandbox** before trusting a score.

## Releasing the container images

`llmbench prepare` builds the evaluator image from a wheel of this package plus the hash-pinned
`requirements.linux.lock` (regenerate it with `scripts/lock_evaluator_requirements.py`). The worker image is built
by `scripts/build_worker_image.py`. Both are referenced by immutable image id, never by tag.
