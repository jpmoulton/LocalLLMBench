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
    runtime.py       the runtime registry: permissions, refused / unenforced settings, DispatchRunner per config
    native.py        the metal-native runner: the container stage machine with a host llama-server process
    native_prep.py   `prepare --runtime metal-native`: hash, Mach-O linkage proof, capability capture, native bundle
  apple.py           Apple Silicon unified-memory telemetry (phys_footprint, vm_stat, swap, pressure, ioreg GPU,
                     pmset power), the admission check and the memory watchdog
  container_eval.py  the evaluator: speed probes, local fixtures, public benchmark dispatch
  backends/          llama.cpp server adapter, bounded HTTP/SSE transport, speed-probe bridge
  benchmarks/        bfcl, ruler (+ ruler_tasks/), evalplus, aider_polyglot: one adapter each, one shared contract
  coding/            sandbox (DockerWorker), host broker, spool protocol, container drivers, benchmark bridge
  evaluations/       local tool / retrieval fixtures, strict tool-call capture, Inspect tasks
  analysis.py, search.py, controller.py, store.py, reports.py   campaign loop, eligibility, SQLite evidence, reports
  registry.py        which benchmarks exist, what each needs (dataset, broker), which options each accepts
scripts/             dataset staging, image build, the macOS llama.cpp installer, multi-model sweeps, reports,
                     calibration, server diagnostics
examples/            candidate and session configs (examples/candidate.json is the packaged tune template)
```

## What is tested live and what only against fakes

The two runtimes are covered differently, and a change should say which kind of evidence it has:

* **`nvidia-container`** was exercised live end to end (RTX 5090, Windows 11, Docker Desktop on WSL2); the unit
  suite runs it against a fake Docker executor and recorded CUDA fixtures. Its configs, fingerprints, Compose
  goldens, `result.json`, candidate, cross-model and coding reports must stay byte-identical when the native
  runtime changes (the session table only gains appended columns): the native fields are omitted from every
  serialisation at their defaults, and tests pin that.
* **`metal-native`** is unit-tested against fakes (a scripted server process, telemetry and evaluator) and against
  real captures from an Apple M1 running llama.cpp `b11011`: the Metal startup logs, `/props` and `/slots`
  (`tests/data/*metal*`) and the `sysctl`, `vm_stat`, `ioreg`, `pmset` outputs (`tests/data/apple-*.txt`). Two
  things in `pytest` are real on purpose: darwin-only reads of the test process's own `phys_footprint`, and a
  harmless Python child standing in for `llama-server` so process groups, signals and the proof of absence are
  tested for real (POSIX only). No test starts `llama-server`, Metal, Docker or Colima.
On an M1 Mac (8 GB) these native paths have been exercised live: the pinned installer, native `prepare` with the
worker image, `candidate`, `tune` from `--config` and from a derived session, `scripts/sweep_models.py` with the
cross-model report, admission refusals on memory, the memory watchdog (sampling only: it never had to stop a
server), and all four public suites. The
`linux/arm64` coding worker was built in Colima and calibrated at 2 GiB and again at 1 GiB of VM memory (83 of 83
Aider Polyglot references pass, 83 of 83 stubs fail), and EvalPlus, Aider Polyglot and the private coding fixtures
then ran through the host broker in that sandbox. A session was interrupted with SIGINT while its first candidate
was evaluating (cancelled, server stopped, absence proven, lease released) and `resume` then completed it, and
`doctor --native-bundle` re-checked the pinned server. Not exercised live: `optimize`, `sample`,
`run_prepared_sweep.py`, partial offload and KV-in-RAM candidates. The NVIDIA runtime was not run live on
the Mac; its coverage there is the unit suite. See
[the measured Mac results](results/apple-m1-2026-09-23.md).

When a real run disagrees with a fixture, capture the new output into `tests/data/` (with machine-specific paths
replaced) and fix the parser against it; do not edit a capture to fit the parser.

## Conventions that matter

* **Strict models everywhere.** Configs are frozen pydantic models that reject unknown fields. Round-trip them with
  `canonical_json(...)` / `model_validate_json(...)`; plain `model_validate(dict)` rejects JSON lists for tuples.
* **Fingerprints are identity.** Anything that changes a measurement belongs in the candidate fingerprint. Adding a
  field with a default to a fingerprinted model changes every fingerprint that serialises it (and the golden
  Compose fixture).
* **Fail closed, and say why.** A refusal names its reason; a missing dataset is a recorded skip, not a crash and
  not a silent omission.
* **Never turn a harness failure into a score.** Rows that did not really run are `environment_error`, and a
  suite the sandbox could not run is reported as `blocked`, never as a zero.
* **Label memory by what it measures.** VRAM (NVIDIA) and unified memory (Apple Silicon) are different physical
  quantities; they never share a column, a minimum or a comparison.
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
