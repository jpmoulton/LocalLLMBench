# Host orchestrator

The host CLI connects the existing inference and benchmark containers into a sequential **Run → Trial → Benchmark evaluation** workflow. See [terminology](GLOSSARY.md), [recipe semantics](recipe.md), and the [example recipe](../recipes/qwen-kv-sweep.yaml).

**Verification status:** CPU regressions, an isolated Docker/Inspect workflow smoke, and one real two-trial Q4_K_M GPU run passed. The live run preceded verbose logging; that addition was verified separately with CPU regression and log-streaming checks.

## Setup and commands

Use Python 3.10+ and Docker with Linux containers. Windows/Docker Desktop is the exercised host; Linux-host execution is not yet verified. Inspect and benchmark dependencies stay in the benchmark image; the host adds YAML/schema/spreadsheet dependencies:

```powershell
python -m pip install -r requirements.txt
docker build --tag localllmbench-inference:local .
docker build --tag localllmbench-benchmark:local benchmarks
```

Build images before running; the orchestrator does not rebuild them inside a run. Rebuild the benchmark image after changing shared/module code. GPU-enabled Docker and enough available VRAM are required for real inference.

Preview without Docker calls, inference, or output files:

```powershell
python orchestrate.py --recipe recipes/qwen-kv-sweep.yaml --plan-only
python orchestrate.py --models "F:/Models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf" --plan-only
```

Remove `--plan-only` to execute. `--recipe` and `--models` are mutually exclusive; multiple model paths follow `--models`. Normal execution prints the same preview and starts without interactive confirmation.

For a small live check, [qwen-smoke.yaml](../recipes/qwen-smoke.yaml) selects Q4_K_M at 128K context with q8_0 and q4_0 K/V caches, two fixed BigCodeBench problems each, and the normal reasoning/output budget. This recipe completed successfully in the live smoke recorded below. On this Windows checkout, use `.\.venv\Scripts\python.exe orchestrate.py --recipe recipes/qwen-smoke.yaml` to select the installed dependencies without activating the environment. If the pre-existing `localllmbench-inference-128k` is running, stop it first to release its GPU allocation and port 8080. Ctrl+C requests cleanup and retains partial results; allow the process to finish cleanup before reclaiming the GPU.

Model-only mode creates one configuration per model: context 8192, otherwise native image defaults (including f16 K/V), all registered benchmarks, and shared generation/module defaults. It is not a recommended 128K deployment preset. Use a recipe to select larger contexts or other overrides. Output-token budget is a cap, not a guarantee that prompt plus output fits.

## Implementation map

| File | Responsibility |
| --- | --- |
| `orchestrate.py` | Argument parsing, large plain-ASCII `LocalLLMBench` wordmark and plan preview, final results, exit code |
| `orchestrator/recipe.py` | Safe YAML/schema validation, defaults, paths, deterministic expansion, frozen input manifests |
| `orchestrator/runner.py` | Sequential lifecycle, readiness/probe, live controller output, cleanup and progress |
| `orchestrator/state.py` | Atomic JSON state and status counts |
| `orchestrator/report.py` | Persisted-evidence reporting and real Excel workbook |
| `inference.py`, `benchmark.py` | Shared launch construction; standalone CLIs remain available |
| `benchmarks/performance.py` | Common native model-call timing extraction |

Both input modes use the same resolved plan and execution path. Recipe-relative paths resolve beside the recipe; model-only paths/output resolve against the working directory. The host prepares selected benchmark identities once. Each controller compares its prepared manifest exactly against that plan before constructing the task or generating benchmark answers. Changed input/selection fails that evaluation instead of silently changing the workload.

## Trial lifecycle

1. Persist the plan/run state; show total trials, models, varying native arguments, and fixed benchmark/generation options.
2. Start a uniquely named inference container on the shared `localllmbench` Docker network. Mount only its model directory read-only and publish the requested address/port.
3. Poll `/health`, also checking that this container is still running, up to `execution.startup_timeout`.
4. Send one short Hello world probe: reasoning disabled, temperature 0, 32-token cap. Require usable assistant text, not an exact phrase match. Save it separately; exclude it from benchmark metrics.
5. Start a fresh benchmark controller with the same options and frozen identities used by every trial. It talks to inference over Docker networking, not the published host address.
6. Stream controller output; save elapsed status about every 15 seconds while it runs. On completion, read suite/per-benchmark summaries.
7. Capture inference logs and phase durations, clean owned containers, print scores/generation metrics and remaining trials, and refresh reports.
8. Continue sequentially, then finalize reports/status and exit.

Native launch arguments/image identity, reported server context, effective configuration, selected IDs/provenance, and Inspect logs are retained. `--ctx-size` is the context control; the old `minimum_context` setting/check has been removed. The model file is identified by path, not a multi-gigabyte content hash: do not overwrite it during a run.

## Failure and ownership rules

Model startup, probe, or benchmark failures mark the trial incomplete; retain available evidence, clean up, and continue. Incorrect benchmark answers are completed scored outcomes. There are no trial retries, automatic retuning, or GPU-fit prediction.

Every owned container receives `localllmbench.run` and `localllmbench.trial` labels. The benchmark controller passes ownership into its Compose sandbox configuration. Cleanup filters on both labels and includes leftover sibling sandboxes. Unrelated containers and occupied ports are never taken over. The shared Docker network is retained.

Ctrl+C stops scheduling, cleans the active trial, retains pending trials and partial reports, and exits 130. Interrupts during cleanup request cancellation without interrupting removal. If cleanup cannot establish that owned containers are gone, stop rather than overlap the next GPU trial. Status-write failures during cleanup do not bypass removal. Forced process termination, power loss, or Docker-daemon failure cannot guarantee cleanup; there is no resume mechanism yet.

Exit codes: `0` all requested evaluations completed; `1` incomplete execution/reporting; `2` invalid CLI/recipe input; `130` controlled interruption. JSON remains authoritative if spreadsheet export fails (for example, Excel has locked the workbook).

## Generation measurements

Generation speed is displayed, **not ranked as “fastest.”** Each benchmark may declare quality metrics/directions for independent high/low-score leaders and ties. There is no combined winner or average of unrelated benchmark scores.

| Measurement | Definition |
| --- | --- |
| Request latency | Inspect's model API call duration, excluding reference tests/scoring |
| Prompt throughput | Sum of paired llama.cpp prompt tokens divided by summed prompt time |
| Decode throughput | Sum of paired `predicted_n` divided by summed `predicted_ms / 1000` |
| Average decode ms/token | Summed decode milliseconds divided by paired generated tokens |
| Benchmark/trial/run duration | Wall time, kept separate from model-generation timing |

Timing coverage counters and per-call rows accompany aggregates. Missing/unusable measurements are null, not zero. Reasoning tokens are included when counted by the server. At concurrency greater than one, summed per-request decode-time throughput is not total system throughput.

The current non-streaming calls do not measure client TTFT or streaming TPOT. Prompt time is not TTFT; average server decode ms/token is not client TPOT. Streaming measurements, GPU memory monitoring, and long-input benchmarks remain future work. High context allocation with short prompts does not establish filled-context performance.

## Console and saved logs

Enable live inference output on the next launch:

```powershell
.\.venv\Scripts\python.exe orchestrate.py --recipe recipes/qwen-smoke.yaml --verbose_logging
```

This follows the current trial's inference stdout/stderr from startup with `[llama.cpp]` labels and Docker timestamps, alongside `[benchmark]`-labelled controller/Inspect console output. Without the flag, the existing progress and benchmark output remain unchanged. The flag changes console visibility, not native logging levels, generation settings, or which benchmark records are saved. Verbose terminal I/O can add overhead; prefer normal mode for timing comparisons.

Each trial's `benchmark.log` is written live, with unprefixed controller stdout/stderr. `inference.log` is a snapshot collected at cleanup, not a live file. Detailed prompts, reasoning, answers, test/scoring output, and errors stay in `benchmarks/<name>/inspect/*.eval`; verbose mode does not expand those records into the console. The log-follower process is stopped during cleanup even if Docker removal fails. An already-running orchestrator does not pick up a newly added CLI flag.

## Results and future read-only UI

```text
results/<run-id>/
  plan.json
  run.json
  report.json
  results.xlsx
  trials/0001/
    trial.json
    inference.json
    inference.log
    probe.json
    benchmark.log
    benchmarks/
      config.json
      container_config.json
      expected_manifests.json
      effective_config.json
      server.json
      versions.json
      summary.json
      bigcodebench/
        manifest.json
        summary.json
        inspect/*.eval
```

Artifacts absent after an early failure are expected. `run.json` records current phase/trial, timestamps, counts, all planned trial states, errors, and result paths. `trial.json` records exact model/settings and load/probe/benchmark/cleanup durations. JSON state uses atomic replacement; a future read-only localhost UI can consume it without scraping terminal output. No web server or control UI is implemented.

Workbook sheets: **Run**, **Trials**, **Benchmark results**, **Samples**. Include incomplete/unstarted trials, per-benchmark quality leaders, generation measurements/coverage, sample outcomes/model-call rows, and links to evidence. External strings are literal spreadsheet cells, not formulas. Full reasoning/answers remain in native Inspect logs.

## Exercised verification

- `python -m unittest discover -s tests -q`: 62 tests passed, including expansion, manifest mismatch rejection, timing coverage, report safety, fail-and-continue, interruption, cleanup failure, status/export failures, and log-reader cleanup after Docker failure.
- Both real-path CLI previews passed: the checked-in recipe resolves eight trials; model-only input resolves one default trial. Neither loads a model.
- Rebuilt the benchmark image successfully.
- Disposable CPU-only Docker smoke: one deliberately exited startup followed by two completed real Inspect/BigCodeBench sandbox evaluations of a synthetic one-problem corpus. Verified fixed manifests, probe versus benchmark reasoning settings, timing propagation, spreadsheet readback, and no remaining owned containers. The response service had no GPU device requests. These were workflow fixtures, not model scores or speed measurements; scripts/data/results were removed afterward.
- Ran the new collector over the retained native evaluation at `results/20261103T184944Z-95d186f9/`: two measured calls, 9,880 generated tokens, 141.95804 summed decode seconds, approximately 69.598 server decode tokens/s. This reads existing evidence; no new inference request.
- Verbose logging: isolated CPU Docker stdout/stderr streamed with source labels alongside a controller-process fixture; raw benchmark logs and nonzero exit code were preserved, and the follower/container were removed. CLI help exposes the flag. No GPU operations or changes to the user's active run were made.

### Real GPU integration smoke — 2026-11-03

The user executed `recipes/qwen-smoke.yaml` on Windows/RTX 5090. Retained local evidence: `results/20261103T210559-31669ea5/` (generated results are excluded from Git). The terminal output, persisted run state, per-benchmark summaries, per-call timing records, and inference logs were reviewed.

- One Qwen3.8-27B Q4_K_M model, two sequential configurations: q8_0 K/V followed by q4_0 K/V.
- Both servers reported 131072 tokens per slot. Both evaluated `BigCodeBench/19` and `BigCodeBench/578`, with reasoning enabled, selection/generation seed 42, and an 8192-token output cap.
- Both probes succeeded; both evaluations completed with 2/2 passes, two measured model calls each, and no recorded generation errors or truncation.
- Run status: completed, two completed trials, zero incomplete/cancelled trials, 500.664 seconds total. Recorded lifecycle includes cleanup between trials and final reporting.

| Measurement | q8_0 K/V | q4_0 K/V |
| --- | --- | --- |
| Decode tokens/second | 73.236 | 66.672 |
| Generated tokens | 9880 | 9482 |
| Mean request seconds | 68.113 | 71.737 |
| Load/readiness seconds | 105.415 | 84.486 |
| Benchmark-controller seconds | 149.292 | 152.364 |
| Entire trial seconds | 259.432 | 241.069 |

This is a manually exercised integration smoke, not a statistically useful quality study or an automated GPU test. Lower total trial time did not imply faster decoding: startup was shorter in the second trial. Output lengths changed substantially between configurations. Two passing problems do not establish quality equivalence; short prompts at a 128K allocation do not establish filled-context performance. Total VRAM savings, Linux-host behavior, and the full eight-trial example remain unverified. No GPU run is triggered by the unit suite.
