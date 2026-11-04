# Host orchestrator

The host CLI connects the existing inference and benchmark containers into a sequential **Run → Trial → Benchmark evaluation** workflow. See [terminology](GLOSSARY.md), [recipe semantics](recipe.md), and the [example recipe](../recipes/qwen-kv-sweep.yaml).

**Verification status:** CPU regressions, an isolated Docker/Inspect workflow smoke, and one real two-trial Q4_K_M GPU run passed. The live GPU run preceded the logging-mode changes; subsequent logging verification uses isolated CPU fixtures.

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

For a longer deployment/workflow stress run, [qwen-stress.yaml](../recipes/qwen-stress.yaml) expands both Q4_K_M and Q6_K across contexts 131072/262144, paired q8_0/q4_0 K/V caches, and micro-batches 256/512. This produces 16 sequential trials, each using the same 64 seed-42 BigCodeBench problems (1,024 benchmark generations if all trials reach evaluation). Reasoning and the normal 8,192-token output cap remain enabled. Logical batch size stays 2048, Flash Attention stays on, all layers are requested on the GPU, and auto-fit is off. Use `.\.venv\Scripts\python.exe orchestrate.py --recipe recipes/qwen-stress.yaml --plan-only` to preview, then remove `--plan-only` to run. Planning was exercised without loading models; GPU fit and full execution are unverified, especially Q6_K with 256K/q8_0. This allocates large contexts but does not fill them or test concurrent serving. For a shorter 16-trial workflow check, change `benchmark_options.bigcodebench.selection.count` to 8 (128 benchmark generations). Ctrl+C retains partial results and requests cleanup; there is no resume support.

For long-input retrieval, [qwen-ruler.yaml](../recipes/qwen-ruler.yaml) explicitly selects RULER: Q4_K_M, 128K server context, q8_0/q4_0 K/V, eight variants with eight frozen prompts each. This is two trials and 128 generation requests. The 64K input budget is measured using a fixed reference tokenizer, not Qwen's tokenizer. Preparation, scoring, and CPU-only verification are documented in [benchmark_container.md](benchmark_container.md#ruler-retrieval-subset). Real model execution of this recipe remains unverified.

For the verified smaller retrieval smoke, [qwen-ruler-smoke.yaml](../recipes/qwen-ruler-smoke.yaml) uses only q8_0 K/V and one prompt per variant (eight requests total). It completed in 6m 38s with all references retrieved and no errors or truncation. See the [live GPU verification record](benchmark_container.md#ruler-live-gpu-smoke--2026-11-04); this does not establish the full recipe's quality or runtime.

Model-only mode creates one configuration per model: context 8192, otherwise native image defaults (including f16 K/V), all registered benchmarks, and shared generation/module defaults. It is not a recommended 128K deployment preset. Use a recipe to select larger contexts or other overrides. Output-token budget is a cap, not a guarantee that prompt plus output fits.

Default-all now includes the 64K-reference-input RULER corpus as well as BigCodeBench. The model-only shortcut's 8192 context cannot accommodate that workload; use a suitable recipe. Selecting a longer context does not regenerate RULER prompts, and no automatic context resizing is performed.

## Implementation map

| File | Responsibility |
| --- | --- |
| `orchestrate.py` | Argument parsing, stacked 59-column plain-ASCII wordmark with chip motif (no tagline), plan preview, final results, exit code |
| `orchestrator/recipe.py` | Safe YAML/schema validation, defaults, paths, deterministic expansion, frozen input manifests |
| `orchestrator/runner.py` | Sequential lifecycle, readiness/probe, live controller output, cleanup and progress |
| `orchestrator/logging.py` | Run-wide stream routing, live file writes, source/trial labels, and logging-mode ownership |
| `orchestrator/state.py` | Atomic JSON state and status counts |
| `orchestrator/report.py` | Persisted-evidence reporting and real Excel workbook |
| `inference.py`, `benchmark.py` | Shared launch construction; standalone CLIs remain available |
| `benchmarks/performance.py` | Common native model-call timing extraction |

Both input modes use the same resolved plan and execution path. Recipe-relative paths resolve beside the recipe; model-only paths/output resolve against the working directory. The host prepares selected benchmark identities once. Each controller compares its prepared manifest exactly against that plan before constructing the task or generating benchmark answers. Changed input/selection fails that evaluation instead of silently changing the workload.

## Trial lifecycle

1. Persist the plan/run state; show total trials, models, varying native arguments, and fixed benchmark/generation options.
2. Start a uniquely named inference container on the shared `localllmbench` Docker network. Mount only its model directory read-only and publish the requested address/port.
3. Poll `/health`, also checking that this container is still running, up to `execution.startup_timeout`. Connection refusals, resets/closures before an HTTP response, and loading responses such as HTTP 503 stay inside this readiness loop; HTTP error responses are closed to release their resources. Container exit still fails startup. This does not retry the Hello world probe or benchmark generation.
4. Send one short Hello world probe: reasoning disabled, temperature 0, 32-token cap. Require usable assistant text, not an exact phrase match. Save it separately; exclude it from benchmark metrics.
5. Start a fresh benchmark controller with the same options and frozen identities used by every trial. It talks to inference over Docker networking, not the published host address.
6. Save controller text output to its live log and run archive; atomically refresh `execution_report.json` about every 15 seconds while benchmarks run. On completion, read suite/per-benchmark summaries.
7. Retain phase durations and logs, clean owned containers, print scores/generation metrics and remaining trials, and refresh matching comparison exports in `results.json` and `results.xlsx`.
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

Logging is automatic. The main terminal shows only LocalLLMBench's plan, progress, results, and failure notices. There is no logging flag or mode to select.

Every executing run writes all three streams to two locations:

| Stream | Stable live file, relative to the project directory | This run's archived copy |
| --- | --- | --- |
| LocalLLMBench | `logs/localllmbench.log` | `results/<run-id>/logging/localllmbench.log` |
| llama.cpp stdout/stderr | `logs/llamacpp.log` | `results/<run-id>/logging/llamacpp.log` |
| Benchmark controller / Inspect stdout/stderr | `logs/benchmarks.log` | `results/<run-id>/logging/benchmarks.log` |

The stable files append across executions: they are not truncated or replaced, so a tail window can stay open between runs. Their directory is anchored to the project, independent of the current working directory or recipe's `output_dir`. Archived files contain only their own run and are written at the same time as the live files, not copied at shutdown. Both destinations are flushed after each entry so logs are available during execution and remain useful after an interrupted run.

Directories and files are created automatically when execution begins, before model startup. `--plan-only` creates no run directory or log files. Start normally:

```powershell
.\.venv\Scripts\python.exe orchestrate.py --recipe recipes/qwen-smoke.yaml
```

From the project directory, use separate PowerShell windows for any streams you want to follow:

```powershell
Get-Content .\logs\localllmbench.log -Tail 50 -Wait
```

```powershell
Get-Content .\logs\llamacpp.log -Tail 50 -Wait
```

```powershell
Get-Content .\logs\benchmarks.log -Tail 50 -Wait
```

Start tailing after the first execution has created the files. The CLI prints both the stable log directory and the current run's archive directory. Entries carry UTC timestamps, a run ID, a source label, and a trial ID or run-level marker. LocalLLMBench's files include the initial preview and final summary; inference output is followed live from startup through cleanup.

Files are closed after final reporting. Inference followers are stopped during cleanup even if Docker removal fails. Live files are ignored by Git and currently have no automatic rotation; they grow across runs. Existing result archives are not modified or backfilled into the stable files.

JSON status/scores, spreadsheets, and native Inspect `.eval` evidence remain unchanged. Inspect evidence includes prompts, reasoning, answers, test/scoring output, and errors; it is separate from the text streams. This behavior applies to the orchestrator, not standalone launchers, and does not change Docker's internal log retention or engine log levels.

## Results and future read-only UI

```text
results/<run-id>/
  execution_plan.json
  execution_report.json
  results.json
  results.xlsx
  logging/
    localllmbench.log
    llamacpp.log
    benchmarks.log
  trials/0001/
    trial.json
    inference.json
    probe.json
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

### Artifact responsibilities

| File | Contents |
| --- | --- |
| `execution_plan.json` | Resolved configuration, fixed benchmark identities, and planned trials |
| `execution_report.json` | Lightweight live execution status: run ID/name, phase/current trial, timestamps, elapsed time, counts, errors, and each trial's status/phase/timestamps/error |
| `results.json` | A `results` array containing one flat comparison row per trial × benchmark |
| `results.xlsx` | A single **Results** sheet with the same rows, field names, and scalar values as `results.json` |

The execution report is written before model startup and atomically updated at phase changes, benchmark heartbeats, trial completion, and finalization. It does not collect benchmark results on every heartbeat. Comparison exports refresh after each trial and at finalization. JSON is saved before XLSX; if Excel locks the spreadsheet, the newest JSON remains available and the execution report records the export failure. The older XLSX may then be stale.

### Comparison rows

Every row is self-contained: the model/settings combination is repeated for each benchmark it ran. Filter the sheet by benchmark, model, or settings without joining another sheet. There are no Run, Trials, or Samples sheets.

| Fields | Meaning |
| --- | --- |
| `trial_id`, `model`, `model_path`, `configuration`, `benchmark` | Trial identity, model filename and full path, configuration-group name, and benchmark |
| `inference.*` | Requested native parameters, for example `inference.--ctx-size` and `inference.--cache-type-k` |
| `generation.*` | Fixed answer-generation settings, including thinking, token limit, sampling settings, and seed |
| `benchmark_options.*`, `request_concurrency` | Benchmark options and concurrent requests |
| `score.*` | Declared benchmark scores, such as `score.pass_rate`; unavailable/invalid values are null |
| `performance.*` | Aggregate decode/prompt speed, request latency, token totals, measured durations, and coverage counters |
| `load_seconds`, `probe_seconds`, `cleanup_seconds`, `trial_seconds` | Timings for the enclosing trial |
| `benchmark_seconds`, `trial_benchmark_seconds` | This benchmark's wall time versus the trial's total benchmark-controller time |
| `sample_count`, `scored_count`, `count.*` | Aggregate sample counts, never individual records |
| `trial_status`, `benchmark_status`, `trial_error`, `benchmark_error` | Outcomes and concise failure information |

Column names are JSON keys verbatim: for example `results[0]["score.pass_rate"]` or `results[0]["performance.decode_tokens_per_second"]`. Both exports are built from the same normalized rows. All rows have the same keys; benchmark-specific columns are null/blank where unavailable or inapplicable, not zero. Nested settings flatten into separate columns; list values use numeric indexes such as `inference.--override.args.0`. No JSON objects are embedded in spreadsheet cells.

Generation metrics belong to each benchmark evaluation; unrelated benchmarks are not averaged into an overall speed. Trial timings repeat across that trial's benchmark rows: do not sum repeated load or trial durations. Missing native arguments use image defaults; the comparison does not invent effective values for unspecified flags. Actual startup/server evidence remains under each trial.

Neither comparison export nor the execution report includes individual samples, prompts, reasoning, answers, or full tracebacks. Error cells contain at most 500 characters; full evidence stays in benchmark summaries, native Inspect logs, trial artifacts, and text logs. External spreadsheet strings remain literal text rather than formulas. Quality leaders still appear in the CLI; no speed ranking or extra leader table is exported.

A future read-only localhost UI can combine the execution plan/status and comparison rows; no UI is implemented. New runs do not create `plan.json`, `run.json`, or `report.json`. Existing result folders are not renamed or rewritten.

## Exercised verification

- `python -m unittest discover -s tests -q`: 73 tests passed, including JSON/XLSX row parity, scalar settings, aggregate-only exports, renamed execution artifacts, live status transitions, failed exports, score eligibility, lifecycle failures/cancellation, logging, and readiness.
- Both real-path CLI previews passed: the checked-in recipe resolves eight trials; model-only input resolves one default trial. Neither loads a model.
- Rebuilt the benchmark image successfully.
- Disposable CPU-only Docker smoke: one deliberately exited startup followed by two completed real Inspect/BigCodeBench sandbox evaluations of a synthetic one-problem corpus. Verified fixed manifests, probe versus benchmark reasoning settings, timing propagation, spreadsheet readback, and no remaining owned containers. The response service had no GPU device requests. These were workflow fixtures, not model scores or speed measurements; scripts/data/results were removed afterward.
- Ran the new collector over the retained native evaluation at `results/20261103T184944Z-95d186f9/`: two measured calls, 9,880 generated tokens, 141.95804 summed decode seconds, approximately 69.598 server decode tokens/s. This reads existing evidence; no new inference request.
- Automatic logging smoke (2026-11-04): two executions of the actual CLI with a disposable CPU response service and real Inspect/BigCodeBench sandbox evaluation, one fixture problem per execution. An unchanged open tail handle saw the second execution's inference output. All three stable live files exactly matched the concatenated per-run archives, child streams stayed out of the main console, and native `.eval`/JSON/XLSX results were retained. Also verified file-free plan-only, rejection of the removed logging flag, and owned-container cleanup. Temporary fixtures/results were removed; no GPU/model was loaded and no image rebuild was required. The 72-test CPU suite covers live flushes, stream/trial isolation, append/archive boundaries, concurrent writers, subprocess failures, and follower cleanup.
- Startup disconnect fix (2026-11-04): reproduced the reported `RemoteDisconnected` using a real local HTTP server before editing. The readiness poll now catches connection-level failures within its existing deadline. Verified with a CPU-only Docker service that closes the first health connection, returns HTTP 503 next, then becomes healthy and answers the probe. Owned fixture cleanup passed; no competing GPU/model work or interference with the user's active second trial. The running orchestrator retains its old code; the fix applies on the next launch.
- Comparison replay (2026-11-04): exported saved evidence from `results/20261103T210559-31669ea5/` into a temporary directory. Verified two trial/benchmark rows with 56 matching scalar columns in JSON and the single Results sheet. Scores, generation speeds, model/settings, load times, and trial durations matched the original evidence. Only the four approved top-level artifacts were generated; execution status contains no result/sample tables. Original artifacts were hash-verified unchanged and temporary exports removed. No Docker/model/GPU work.

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
