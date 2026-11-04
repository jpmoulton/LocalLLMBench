# Benchmark container

## Scope

Shared Inspect AI controller, with one directory per benchmark. Registered modules are BigCodeBench and the RULER retrieval subset. Both use the same host orchestrator, frozen-input checks, generation settings, logging, and comparison exports.

## BigCodeBench local dataset

User-selected asset directory:

`C:\Users\hocke\Documents\LLM-Reusable-Assets\benchmarks\datasets\current\bigcodebench`

Observed assets: `corpus.jsonl`, `sample.jsonl`, `manifest.json`, and `task_domains.json`.

The supplied manifest identifies the corpus as BigCodeBench v0.1.4, with 1,140 problems and source dataset revision `b74c0d0bf70d2c0bc459be537895cca163007f1a`. Its existing sample contains 64 problems selected using seed 42 and a domain-stratified policy. That pre-existing sample policy is not the policy selected for this new project.

## RULER retrieval subset

The local `ruler` asset directory contains 49 essays, not ready-made benchmark prompts. `benchmarks/ruler/prepare.py` uses NVIDIA's original NIAH generator to create a frozen corpus once, on CPU. Host and container preparation subsequently only read/select that corpus; no prompts are regenerated between trials.

This is **eight classic RULER retrieval variants**, not the full 13-task suite or a paper-comparable leaderboard score:

| Variant | Retrieval task |
| --- | --- |
| `niah_single_1` | One word-key/numeric-value needle in repeated noise |
| `niah_single_2` | One word-key/numeric-value needle in essays |
| `niah_single_3` | One word-key/UUID-value needle in essays |
| `niah_multikey_1` | Four keyed needles in essays; query one key |
| `niah_multikey_2` | Retrieve a word-key/numeric-value target among distractor needles |
| `niah_multikey_3` | Retrieve a UUID-key/UUID-value target among distractor needles |
| `niah_multivalue` | Retrieve four numeric values for one word key |
| `niah_multiquery` | Retrieve the values for four queried word keys |

### Prepare once, without a GPU

The default corpus is already prepared on this PC at:

`C:\Users\hocke\Documents\LLM-Reusable-Assets\benchmarks\datasets\current\ruler\prepared`

For a fresh output directory, from the project root:

```powershell
docker build --tag localllmbench-benchmark:local benchmarks
docker run --rm --cpus 2 --memory 4g --mount "type=bind,source=C:/Users/hocke/Documents/LLM-Reusable-Assets/benchmarks/datasets/current/ruler,target=/ruler" --entrypoint python localllmbench-benchmark:local -m benchmarks.ruler.prepare --essays /ruler/essays --output /ruler/prepared
```

The parent directory is mounted because preparation creates the output directory itself. Existing output is never overwritten; choose another output directory for a different corpus. Original essays are only read. Preparation needs network access to pinned source/tokenizer resources, but never downloads model weights or accesses an inference service. Upstream builds a large adjective/noun vocabulary, so CPU preparation uses appreciable RAM; the command above completed within its 4 GiB limit.

Preparation options:

| CLI option | Default and meaning |
| --- | --- |
| `--input-tokens` | `65536`: maximum input plus answer-prefix tokens, measured with the fixed reference tokenizer; minimum `4096` |
| `--samples-per-task` | `8`: number generated for each variant |
| `--seed` | `42`: synthetic problem-generation seed, from `0` through `2**32 - 1` |
| `--reserved-output-tokens` | `8192`: upstream sequence-sizing reservation; input ceiling remains `--input-tokens` |
| `--tasks` | All eight variants; optionally supply a space-separated subset |

Artifacts are `corpus.jsonl`, `manifest.json`, and attribution notices. The manifest records source hashes, dependency versions, essay identities/order, tokenizer resources, preparation settings, and the corpus hash. Do not commit or redistribute generated essay prompts without appropriate permission.

The generator is pinned to [NVIDIA/RULER revision db43f92](https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a). Downloaded source, NLTK English sentence tables, and the `cl100k_base` vocabulary are SHA256-verified. Direct preparation dependencies are pinned in `benchmarks/ruler/requirements.txt`; attribution is in `benchmarks/ruler/THIRD_PARTY_NOTICES.txt`.

### Length, selection, and reasoning

**64K is a reference-token input budget, not a claim of exactly 64K Qwen tokens or a server allocation.** The fixed tokenizer is `tiktoken`'s `cl100k_base`, one of upstream's supported adapters. The actual prepared inputs range from 64,100 to 65,536 reference tokens. Chat-wrapper overhead is not included. Model-native counts and output must still fit the deployed context.

`recipes/qwen-ruler.yaml` requests 131072 server context, one slot, and the shared 8192-token reasoning-plus-final output cap. Changing serving context or KV precision does not alter the prompts. To evaluate a different input length, prepare another corpus with `--input-tokens` and point `dataset_path` at it. Preparation's output reservation does not override `generation.max_tokens` or allocate server context.

The official base template and answer prefix are sent together as one user message. The local essays, fixed reference tokenizer, chat transport, larger output budget, and reasoning-enabled/final-only scoring are explicit protocol adaptations. They make comparisons within a frozen LocalLLMBench run useful without claiming equivalence to the [original RULER evaluation](https://arxiv.org/abs/2404.06654).

Module defaults:

```yaml
benchmarks: [ruler]
benchmark_options:
  ruler:
    dataset_path: C:/Users/hocke/Documents/LLM-Reusable-Assets/benchmarks/datasets/current/ruler/prepared
    # Omit tasks to use all eight variants; an explicit list selects a subset.
    selection:
      count_per_task: 8
      seed: 42
```

Thus the default evaluation contains **64 prompts per trial**. Per-task selection ranks stable sample IDs by a versioned SHA256 of the selection seed and ID. Paths, models, and inference/generation settings do not participate. Selected IDs and the full source identity are frozen in `execution_plan.json` and checked inside every controller.

There are three independent seeds: preparation, subset selection, and model generation. The initial pool contains eight examples per variant, so selecting all eight with another selection seed changes order, not membership. To sample different subsets, first generate a larger pool using `--samples-per-task`, then select fewer records per task. To synthesize different keys/values/positions, prepare a new corpus with another preparation `--seed`.

### Scores and execution

The scorer follows upstream's case-insensitive substring **reference recall**, not exact-match accuracy. Finding two of four requested values scores 50%; extra output is not penalized. Separately represented reasoning is retained by Inspect but excluded from scoring. The overall percentage is the equal-weight mean of selected variant scores.

`score.retrieval_score` and eight `<variant>_score` fields appear in the existing `results.json` and Results worksheet, alongside model/settings, timings, and generation metrics. Unselected variant scores are null. Missing/duplicate/unexpected outcomes, request errors, or truncation make the evaluation incomplete and withhold all aggregate scores; an ordinary wrong answer remains a completed scored outcome. Per-sample evidence stays in the benchmark summary and native Inspect logs, not comparison exports.

RULER has no generated-code execution or sandbox. A RULER-only controller does not mount the Docker socket. If BigCodeBench is selected too, its existing sandbox requirements still apply.

```powershell
# CPU-only preview; does not launch inference.
.\.venv\Scripts\python.exe orchestrate.py --recipe recipes/qwen-ruler.yaml --plan-only

# Starts the GPU model: run only when the GPU is available.
.\.venv\Scripts\python.exe orchestrate.py --recipe recipes/qwen-ruler.yaml
```

The recipe runs Q4_K_M at q8_0 then q4_0 K/V: two trials, 64 identical prompts each, 128 total generation requests if both complete. Explicit BigCodeBench smoke/stress recipes remain unchanged. **Default-all now includes both modules**, so prepare both inputs and use sufficient serving context. The model-only CLI shortcut's 8192 context is not suitable for this default RULER corpus; use a high-context recipe instead. Context is never silently resized.

## BigCodeBench decisions

- Read the full local corpus rather than silently using the existing `sample.jsonl`.
- Start with simple deterministic seeded selection: default 64 problems, seed 42, both configurable. Rank stable task IDs using a documented hash of the seed and task ID so selection does not depend on row ordering.
- Persist selected task IDs, selection settings, and dataset identity with each run. Identical corpus contents and selection settings must select identical problems across all model/settings runs.
- Use Complete prompts with the instruction prefix from Inspect's established BigCodeBench implementation. The local dataset version is v0.1.4; do not mislabel this as an unchanged run of Inspect's default v0.1.2 dataset.
- Keep Inspect in the benchmark container and permit that trusted controller to access the host Docker daemon.
- Execute generated Python in separate per-sample sibling Docker sandboxes, without network, host mounts, GPU, or Docker socket. This is not a nested Docker daemon. Host Docker access grants the controller broad privileges; generated code must not execute in that controller.
- Continue after sample execution errors where possible, retain diagnostics, and mark runs with such errors incomplete. Ordinary incorrect solutions are scored failures, not infrastructure errors.
- Dataset/asset downloads and caching are allowed; external model/API calls require separate approval.

## BigCodeBench selection policy to revisit

The user explicitly wants to revisit broad, guaranteed multi-category coverage. Simple seeded selection is acceptable now, but it does not guarantee domain or library coverage and does not select easier problems. Reconsider stratification or another coverage-aware policy later, with the user choosing the design.

A 64-problem unweighted pass rate changes by about 1.56 percentage points per problem. It is a practical initial subset, not a guarantee that small quantization effects will be detectable. Keep per-problem outcomes for comparisons.

## Generation and deployment configuration

Standalone settings live in `benchmark_config.json`; [orchestrator recipes](recipe.md) use the same defaults and benchmark-owned options. Shared defaults/validation live in `benchmarks/config.py`. Unknown keys fail rather than silently being ignored. Standalone host input/output paths are relative to the configuration file unless absolute.

| Section | Settings and initial defaults |
| --- | --- |
| `benchmarks` | Omitted or `null`: all registered benchmarks, in registry order. Explicit nonempty list: only those benchmarks, in the supplied order. Unknown names and duplicates fail. |
| `benchmark_options.bigcodebench` | `dataset_path`: local asset directory above; `selection: {"count": 64, "seed": 42}`; `test_timeout: 60` seconds |
| `benchmark_options.ruler` | Prepared dataset path above; all eight retrieval variants; `selection: {"count_per_task": 8, "seed": 42}` |
| `inference` | `container: "localllmbench-inference-128k"`, `model: "local-model"` for standalone attachment |
| `generation` | `thinking: true`, `max_tokens: 8192`, `temperature: 1.0`, `top_p: 0.95`, `top_k: 20`, `min_p: 0.05`, `seed: 42` |
| `execution` | `concurrency: 1`, `request_timeout: 900` seconds |
| `output_dir` | `"results"` |

The checked-in configuration omits `benchmarks`, so adding a registered benchmark includes it in default runs. A benchmark's options do not select it. Selected benchmarks execute sequentially; concurrency controls samples within an evaluation.

The former root-level `dataset_path`/`selection` and `execution.test_timeout` have moved into `benchmark_options.bigcodebench`. There are no compatibility aliases or global `--count`/`--seed` flags; sampling is benchmark-specific.

The user approved reasoning-enabled benchmarking at intended high deployment contexts, rather than extrapolating from small-context configurations. The first verified server uses Q4_K_M weights, context 131072, q8_0 K/V caches, Flash Attention, and full layer offload. Inference image defaults remain unchanged.

The benchmark controller attaches to its configured server; it does not reconfigure inference. It records server-reported context from `/props`, with no separate minimum-context threshold. The host orchestrator starts each trial's server with the requested `--ctx-size`. `generation.max_tokens` caps reasoning plus final code; prompt and output still need to fit the deployed context.

Sampling values were made explicit from the model's observed defaults. They are a starting operating profile, not a universal best preset or an official BigCodeBench leaderboard protocol. Temperature zero remains configurable. A fixed generation seed does not guarantee identical outputs across quantizations, runtimes, hardware, or concurrency.

The generation seed is independent of the subset seed. Changing generation or inference settings cannot change task selection.

## Files and pinned dependencies

- `benchmark.py`: generic host launcher, benchmark selection, input mounts, and preparation preview. Standard-library Python plus Docker CLI.
- `benchmark_config.json`: editable suite configuration.
- `benchmarks/registry.py`: explicit benchmark-name-to-module mapping; lazy definition/runtime imports.
- `benchmarks/config.py`: shared configuration validation and merging of benchmark-owned defaults.
- `benchmarks/run.py`: shared server checks, Inspect execution, per-benchmark output isolation, and suite status. No BigCodeBench prompt/scoring/selection logic.
- `benchmarks/bigcodebench/definition.py`: host-safe defaults, input paths, validation, local corpus loading, deterministic selection, and provenance.
- `benchmarks/bigcodebench/task.py`: Complete prompts, reference preflight, sandboxed scoring, and coverage-aware result accounting.
- `benchmarks/bigcodebench/sandbox.yaml`: this benchmark's code-execution environment.
- `benchmarks/bigcodebench/requirements.txt`: benchmark-specific controller dependencies.
- `benchmarks/ruler/prepare.py`: explicit CPU-only synthesis using pinned upstream code/resources.
- `benchmarks/ruler/definition.py`: frozen-corpus validation, selection, and provenance.
- `benchmarks/ruler/task.py`, `scoring.py`: Inspect adapter, reference recall, and completeness accounting.
- `benchmarks/ruler/requirements.txt`: pinned CPU preparation dependencies.
- `benchmarks/Dockerfile`, `benchmarks/requirements.txt`: shared controller image/dependencies. The build installs each benchmark directory's optional requirements file.
- `tests/test_benchmark_config.py`, `tests/test_bigcodebench.py`, `tests/test_ruler.py`: shared selection/configuration and benchmark-specific selection/scoring/coverage regressions.

Controller dependencies: Inspect AI `0.3.287`, Inspect Evals `0.27.0`, OpenAI SDK `3.4.0`. Python and Docker client base images are digest-pinned in the Dockerfile. Direct Python dependencies are version-pinned; transitive versions are recorded per run, but are not a full dependency lockfile. Keep the built image for exact repeat runs.

Code sandbox: `ghcr.io/generality-labs/inspect-eval-bigcodebench:2026-07-28@sha256:d7c88bf92782a7486d64bc1d07cd38ba9e735b5f58235fc0c0f36f8f692a1cfe`, the pinned image referenced by Inspect Evals.

## Adding a benchmark

When adding/registering a benchmark or changing a module interface, follow [Adding a benchmark](add_benchmark_instruct.md). It covers the definition/runtime contract, configuration and path handling, dependencies/sandboxes, registration, and end-to-end verification.

Benchmarks own their preparation, Inspect task, and result interpretation. The shared runner owns suite execution and status; another benchmark fitting the existing contract requires no shared-runner or Dockerfile change.


## Build and run

From the project directory:

```powershell
docker build --tag localllmbench-benchmark:local benchmarks
```

For a fresh high-context inference deployment (stop any existing container publishing port 8080 first):

```powershell
python inference.py --model "F:\Models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" --name localllmbench-inference-128k -- --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0
```

That container already exists after development; use `docker start localllmbench-inference-128k` if it is stopped. Wait for `/health` to return `status: ok`.

List registered benchmarks and preview preparation without Docker or model execution:

```powershell
python benchmark.py --list-benchmarks
python benchmark.py --select-only
python benchmark.py --benchmarks bigcodebench --select-only
python benchmark.py --benchmarks ruler --select-only
```

Run all configured benchmarks, or explicitly select only BigCodeBench:

```powershell
python benchmark.py
python benchmark.py --benchmarks bigcodebench
```

The CLI benchmark list overrides the configuration's list. Do not launch these commands simultaneously against the same single-slot server.

For a two-problem smoke run, save this as `my_benchmark_config.json`:

```json
{
  "benchmarks": ["bigcodebench"],
  "benchmark_options": {
    "bigcodebench": {
      "selection": {"count": 2, "seed": 42}
    }
  }
}
```

Then run:

```powershell
python benchmark.py --config my_benchmark_config.json
```

A configuration may contain just changed fields. Omit `benchmarks` to run all registered benchmarks with their respective options. Change BigCodeBench's count/seed inside its options; shared generation overrides remain under `generation`.

Larger output budgets still require adequate prompt-plus-output context. Increasing request concurrency also requires an intentionally configured inference server; it does not create more server slots.

The launcher creates/reuses the `localllmbench` Docker bridge network and attaches the selected inference container. The controller calls `http://<inference-container>:8080/v1`, not its own localhost. Host publication of inference remains localhost-only. The launcher requires a local Linux Docker engine context, as tested with Docker Desktop on Windows; remote Docker daemons and Linux-host execution are not verified.

## Reproducible selection

Algorithm identifier: `localllmbench.bigcodebench.sample.v1`.

For each task ID, compute SHA256 over UTF-8:

```text
localllmbench.bigcodebench.sample.v1
<decimal seed>
<task ID>
```

There is one newline between fields and no trailing newline. Sort ascending by digest bytes, then task ID as a collision tie-break; take the first `count` tasks. Smaller counts are prefixes of larger ones for the same seed/corpus. Selection never uses model identity, generation settings, outcomes, or source row position.

The manifest records both raw corpus SHA256 and a canonical content fingerprint (compact, sorted-key JSON of all records sorted by task ID). Reordering JSONL rows leaves selection and the canonical fingerprint unchanged. Changing task contents changes the fingerprint even if IDs stay the same. A supplied asset manifest's raw checksum must match the actual corpus; update provenance deliberately if changing that corpus.

For the supplied corpus, default seed 42 begins with `BigCodeBench/19`, `BigCodeBench/578`, and `BigCodeBench/800`. The pre-existing domain-stratified `sample.jsonl` is not used.

## Execution and scoring

1. Capture shared server properties/model information and check the required deployment context.
2. For each selected benchmark, prepare its inputs and save its provenance in its own output directory. BigCodeBench reads the local corpus and selects its task IDs here.
3. For each sample, run its canonical solution against its tests in a sandbox before requesting a model answer. A failed reference preflight is an execution/dataset error, not evidence against the model.
4. Send the Complete prompt through Inspect's native converter and solver. `chat_template_kwargs.enable_thinking` selects reasoning mode for the tested model.
5. Keep reasoning in Inspect's reasoning blocks. Extract code only from the final completion using Inspect Evals' `find_code`, then execute that code plus the dataset tests.
6. Retain the native Inspect log and write a coverage-aware summary.

One response per task, one epoch; no automatic model/sample/timeout retries. Canonical and generated code execute in separate child processes and temporary working directories. Reference solutions are not sent to the model.

This is a local v0.1.4 Complete-prompt, sampled evaluation, not an official full-corpus leaderboard result. The custom scorer preserves the upstream test-execution/code-extraction approach but adds reference preflight and explicit execution-error/truncation handling.

Scoring policy:

- Passing tests: `pass`.
- Ordinary generated-code/test failure: `fail`.
- Reference-preflight failure, sandbox timeout/OOM/abnormal process exit, model/API error, or missing result: incomplete evaluation.
- Exhausted output/context budget: `truncation`, no score, incomplete evaluation.
- Continue other samples where possible. Never increase the budget, retry a failed answer, or replace a failed task with another task silently.
- `bigcodebench/summary.json` publishes `pass_rate = passes / all selected tasks` only when every selected task is scored without execution errors or truncation. Otherwise `pass_rate` is null and per-task outcomes/counts remain available.
- An incomplete benchmark does not prevent later selected benchmarks from running. An interrupt stops the suite; shared startup failures prevent evaluation.
- Exit code 0 means every selected benchmark completed, not that every solution passed. Exit code 1 indicates an incomplete suite.

The native Inspect log may still contain metrics computed over its successfully scored samples. Use the project summary's coverage/status before treating those metrics as a full-run score.

## Viewing results locally in VS Code

Install the shared and benchmark-specific controller dependencies into the project's Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r benchmarks/requirements.txt -r benchmarks/bigcodebench/requirements.txt
```

The workspace default interpreter is `.venv/Scripts/python.exe`. If VS Code has already selected another interpreter, use **Python: Select Interpreter** from the Command Palette (`Ctrl+Shift+P`), choose **Enter interpreter path...**, and select `.venv\Scripts\python.exe` in this project. The workspace default does not override an existing interpreter selection.

Close and reopen the `.eval` file using the Inspect AI extension. If Inspect View still reports a missing or outdated package, run **Developer: Reload Window** and reopen the file. Activating a terminal environment alone does not select the interpreter used by the extension.

Verify the local environment without activating it:

```powershell
.\.venv\Scripts\inspect.exe --version
.\.venv\Scripts\python.exe -m pip check
```

Viewing saved logs does not require starting the inference server or benchmark containers.

## Isolation and resource limits

The trusted controller mounts selected benchmark inputs read-only and a unique output directory read/write. For BigCodeBench it also mounts the host Docker socket. Docker-daemon access is powerful; only trusted project/controller code may run there.

Generated code runs in per-sample siblings: user `1000:1000`, network disabled, no host mounts/socket/GPU, all Linux capabilities dropped, `no-new-privileges`, 1 CPU, 2 GiB memory, and 256-process limit. Its writable filesystem is disposable. Inspect removes sample sandboxes; the launcher removes the controller when it exits. Orchestrated controllers/sandboxes carry run/trial ownership labels so the host can remove leftovers after an interrupted controller.

These are Docker isolation measures, not a hardened VM security boundary or protection against container/kernel exploits. No secrets are intentionally passed into the execution sandboxes. Some benchmark tasks can fail due to dependency/version assumptions or resource limits; reference preflight exposes these rather than quietly treating them as model errors.

## Results

Each run gets a unique `results/<UTC timestamp>-<suffix>/` directory, printed by the launcher. Nothing is overwritten when rerunning the same seed.

| Artifact | Content |
| --- | --- |
| `config.json` | Resolved host settings, selected names, and host input paths |
| `container_config.json`, `effective_config.json` | Remapped container settings and runner-effective configuration |
| `inference.json` | Docker image identity, launch arguments/environment, and model-directory mount |
| `server.json` | Server properties, model metadata, verified deployment context |
| `versions.json` | Python and installed package versions |
| `summary.json` | Suite status and each selected benchmark's status/summary location; no cross-benchmark aggregate score |
| `<benchmark>/manifest.json` | Benchmark-owned preparation provenance; BigCodeBench records selected IDs, count/seed, algorithm, and dataset fingerprints |
| `<benchmark>/inspect/*.eval` | Native Inspect samples, reasoning, final answers, test results, errors, model API records |
| `<benchmark>/summary.json` | Benchmark-owned scores/coverage plus shared `score_metrics`, `performance`, and per-call `generation_samples` |

Paths beginning `/results` inside logs refer to that run's host output directory. Startup failures also retain an incomplete summary when the controller reached its runner. An interrupted container can leave an unfinished/incomplete run; there is no resume mechanism yet.

The run records the model path and inference image, not a cryptographic hash of the multi-gigabyte model artifact. Do not overwrite a model file in place between comparisons and assume its filename proves identity. A stronger model-artifact identity policy is still open.

The [host orchestrator](orchestrator.md) now manages sweeps and spreadsheet export. Shared performance extraction retains request latency, native prompt/decode timings, and coverage; missing values remain null. Client TTFT/streaming TPOT, comparison UI, and a results database remain unimplemented. Short coding prompts at a 128K allocation do not establish long-input performance or quality.

## Verification — 2026-11-03

- Rebuilt and exercised the modular controller image against the existing 131072-context Q4_K_M/q8_0 server.
- `--list-benchmarks`, default `--select-only`, explicit `--benchmarks bigcodebench --select-only`, and unknown-name rejection were exercised. Default and explicit previews retained identical 64 selected IDs and the original canonical corpus fingerprint.
- Twenty-three standard-library regression tests passed, covering multi-entry registry selection/default-all behavior, configuration validation/isolation, and deterministic BigCodeBench selection/fingerprints.
- A default-all run with BigCodeBench count 2 completed successfully: `BigCodeBench/19` and `/578` both passed. Output totals remained 3,605 and 6,275 tokens; native Inspect logs retained separate reasoning and final-text blocks.
- Retained modular-run output: `results/20261103T184944Z-95d186f9/`, with root suite summary and nested `bigcodebench/` artifacts.
- A throwaway in-container check temporarily registered the real BigCodeBench module under a second name. Default-all visited both names; a deliberate missing-corpus failure in the first did not prevent real inference in the second. A deliberate one-token cap produced incomplete/null-score output. Explicit selection excluded the second registration. No additional benchmark or temporary outputs were retained.
- Earlier live sandbox inspection verified user, disabled network, absent mounts/GPU, dropped capabilities, memory/process limits, and no-new-privileges.
- Earlier pre-refactor success and intentional two-truncation outputs remain at `results/20261103T183309Z-01608898/` and `results/20261103T183600Z-12fc9b40/`. These historical runs retain their older flat artifact layout.

The full 64-task generation run has not been executed. Two passes establish the integration path, not a useful quality estimate. The high-context server was exercised with short prompts, not a filled 128K context. No Linux-host or other-model run has been tested.

The orchestrator integration subsequently passed 61 CPU regression tests and a CPU-only Docker/Inspect smoke; see [its verification record](orchestrator.md#exercised-verification). That smoke used synthetic responses/tasks, not new model inference. Shared runtime changes include frozen manifest checks before generation, timing extraction, and removal of the legacy minimum-context guard.

### RULER CPU verification — 2026-11-04

- Rebuilt the shared benchmark image; 99 host regressions passed. The 26 RULER regressions also passed after the final numeric-score guard adjustment. New module LSP diagnostics are clean.
- Executed the pinned generator under `--cpus 2 --memory 4g`: 64 real prompts across eight variants, 17,344,560 corpus bytes, reference inputs 64,100–65,536 tokens. Corpus SHA256: `e419d27d77b27385de4e4d4fbdde0682808d2d647204ac0a9f28dc53fc604fb3`. Original essay files match the recorded preparation hashes.
- Actual CLI default-all preparation selected 64 BigCodeBench plus 64 RULER records. The RULER recipe's plan-only preview resolved both KV configurations without starting containers or loading a model.
- Executed two real CPU Docker/Inspect evaluations with controlled HTTP responses and one real frozen prompt per retrieval variant. Both controllers ran without network access beyond loopback, GPU requests, or Docker socket mounts. Host/container manifests matched exactly.
- The first response fixture deliberately returned wrong, partial, and correct final answers while placing all references in separate reasoning. Expected aggregate recall was 75%; Inspect and the module summary agreed. This is a scoring/transport fixture, not a model-quality result.
- The second fixture truncated one response. The benchmark became incomplete and all overall/per-variant comparison scores were null; timing evidence remained available. Native Inspect's own mean over scored samples is not the authoritative complete-run score.
- Both exports were read back: one Results row with 66 matching JSON/XLSX columns, correct sample counts, status, and score availability. Temporary fixture scripts/results were removed.
- No inference server/model was started. Qwen-native token lengths, VRAM fit, retrieval quality, and performance remain to be checked when the user permits a GPU run.

## RULER live GPU smoke — 2026-11-04

After the user explicitly made the GPU available, ran [qwen-ruler-smoke.yaml](../recipes/qwen-ruler-smoke.yaml) through the real orchestrator: Qwen3.8-27B Q4_K_M weights, q8_0 K/V caches, 131072 context, one slot, reasoning enabled, and one frozen prompt per retrieval variant. Retained local evidence: `results/20261104T190827-6ab3ec66/`.

| Measurement | Observed result |
| --- | ---: |
| Total run time | 397.8 seconds |
| Load/readiness time | 97.767 seconds |
| Benchmark evaluation time | 289.331 seconds |
| Samples scored | 8 / 8 |
| Overall and every variant's retrieval recall | 100% |
| Errors / truncations / missing outcomes | 0 / 0 / 0 |
| Server-reported prompt tokens, total | 567,305 |
| Server prompt throughput | 2,582.023 tokens/second |
| Generated tokens, total | 3,103 |
| Server decode throughput | 55.958 tokens/second |
| Mean request latency | 35.962 seconds |

The single Results row and `results.json` matched across 73 columns, allowing ordinary floating-point serialization rounding in XLSX. Inference cleanup completed; `docker ps` showed no running containers afterward.

This proves the live long-input integration at this configuration, not a quantization comparison or full-context capability. The two-trial, 64-prompts-per-trial RULER recipe remains untested. No total/peak VRAM measurement was collected, and no Linux-host run was exercised. Generated results remain local and are not committed.
