# LocalLLMBench — Progress

## Project intent

Build a local-model testing and benchmarking environment that measures both inference speed and quality. Compare quantizations and inference settings for one or several models so the user can make task-specific quality/speed tradeoffs from quantitative evidence.

## Working agreement

- This is a fresh project. Earlier projects or leftover files named LocalLLMBench are not part of this version; do not reuse them without the user's direction.
- The user has relevant experience building a similar system at Amazon and wants an active role in development.
- Work slowly and iteratively. The user decides designs and build order; discuss a bounded component before implementing it.
- Do not autonomously expand scope, choose the overall architecture, or build a framework ahead of agreed requirements.
- Record agreed decisions, actual progress, and open questions here. Keep proposals distinct from decisions.

## Initial environment and direction

User-reported environment:

- Development OS: Windows.
- Initial GPU: NVIDIA RTX 5090.
- Longer-term goal: support other operating systems and GPUs.
- Use Docker for inference and benchmark workloads. llama.cpp is selected and its initial CUDA inference container is implemented; benchmark runtime details remain undecided.
- Docker does not by itself guarantee GPU portability: host drivers, GPU passthrough, inference backend support, and hardware-specific kernels still matter.

## Initial model inventory

Root: `F:\Models\lmstudio-community`

| Model directory | Reported variants |
| --- | --- |
| `Qwen3.8-27B-GGUF` | Q4_K_M, Q6_K |
| `Qwen3.8-27B-NVFP4-MTP-GGUF` | NVFP4 MTP MID-HIGH |
| `Qwen3.8-27B-Ultra-Uncensored-Heretic-Native-MTP-Preserved-NVFP4-GGUF` | Heretic NVFP4 |

Inspected `Qwen3.8-27B-GGUF`: Q4_K_M, Q6_K, and a BF16 multimodal projector are present. Only Q4_K_M text inference has been loaded and exercised. The other directories remain user-reported and uninspected.

Do not assume model identity, lineage, backend compatibility, or MTP support from names alone. Verify only when relevant to an agreed task.

## Agreed high-level workflow

- **Inference container:** loads a model and starts an inference server.
- **Benchmark container:** uses Inspect AI to execute benchmarks, call the inference server, and obtain scores. Benchmark selection and whether different benchmark categories need separate containers remain undecided.
- **Host-side orchestrator:** runs directly on the host and manages launching inference and benchmark containers, with room for additional workflow components later. No orchestrator container or Docker-in-Docker.
- The orchestrator accepts a list of model locations on disk and a list of inference settings, then sweeps models and settings through benchmark execution.
- Multiple inference containers will probably run sequentially, not concurrently. This is the user's current preference, not a finalized scheduling design.
- After the sweep, the user reviews results to choose models/settings for their use cases.
- Broad inference measurement is required: TTFT, TPOT, latency, generation throughput, and other relevant inference metrics. Exact definitions, collection mechanisms, and aggregation remain open.
- Results handling, storage, reporting, and derived metrics will be designed later.
- Container lifetime/reuse, sweep combination semantics, configuration schemas, and component interfaces are not yet specified.
- Develop orchestration for Windows first. Linux support remains a goal, but OS-specific implementation details will be addressed when needed; no portability framework is selected.

## Planned LocalLLMBench UI — deferred

- Add a project-specific web UI served on localhost, separate from llama.cpp's bundled chat UI.
- Show the current sweep status, scheduled runs, active runs, and completed runs.
- Allow viewing results for completed runs while the rest of the sweep continues.
- The UI is strictly read-only: status and results only. No starting, stopping, cancelling, retrying, or editing runs. This is a scope decision, not merely an initial-iteration limitation.
- The orchestrator owns run state independently of the UI; closing or refreshing the browser must not affect sweep execution or results.
- The orchestrator supplies and updates the relevant run status and results for the UI.
- Design and implementation are deferred until the user chooses to work on this component. UI framework, hosting arrangement, update mechanism, and results storage remain undecided.

## Benchmark execution policies

- Decide evaluation protocol per benchmark with the user: review its reference recipe before choosing reference settings or an explicitly labeled custom variant.
- Decide thinking mode and generation budgets per benchmark. Do not silently inherit server reasoning behavior.
- On execution errors, record failed samples and continue where possible. Mark affected runs incomplete and report failures and completion coverage; do not present partial results as a fully successful benchmark. Ordinary wrong answers are scored outcomes, not execution failures.
- Allow required dataset/asset downloads and cache them. Keep downloads outside inference timing. Paid or external model/API calls require separate approval.
- First benchmark selected and integrated: local BigCodeBench v0.1.4, Complete prompts, configurable seeded subset (default 64 tasks / seed 42), reasoning enabled. See `benchmark_container.md` for protocol, usage, isolation, and verification.

## Measurement considerations — proposals, not selected designs

- Keep quantization comparisons distinct from comparisons involving changed model weights, fine-tuning, or speculative/MTP inference. Otherwise quality differences cannot reliably be attributed to quantization alone.
- Distinguish client-observed timing from server execution timing, and streaming chunks from actual tokens. Inspect AI's available measurements and any additional instrumentation must be verified when designing metrics.
- Initial benchmark runs record dataset selection/fingerprints, effective configuration, server properties, inference Docker image/arguments/mounts, package versions, native Inspect logs, and a coverage-aware summary. Model-file hashing and broader results design remain open.
- BigCodeBench is the first quality benchmark; the rest of the suite and task coverage remain user-directed.
- Prefer sequential GPU runs initially to avoid cross-run resource contention; benchmark request concurrency is a separate workload setting.
- Keep host model paths distinct from container paths and avoid unnecessary shell-specific dependencies. The initial launcher uses host Python's standard library and the Docker CLI; this does not select a language or architecture for the future orchestrator.
- Define readiness and teardown boundaries when designing lifecycle management so benchmarks do not begin before the model is ready or overlap with the previous run.

## Inference backend — llama.cpp selected

- User selected llama.cpp as the initial inference backend. Fine-grained setup control remains a priority; the user is already familiar with vLLM from work.
- Official llama-server startup reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#usage . Its option tables are generated from the CLI. The implemented image pins build b12063 / commit 2cbca389f and was checked using its own `--help` and `--version`.
- vLLM engine argument reference: https://docs.vllm.ai/en/latest/configuration/engine_args/ . This URL tracks developer-preview documentation; match documentation to the selected release before implementation.
- vLLM's stable GGUF documentation currently labels support highly experimental and under-optimized and directs users to an out-of-tree plugin: https://docs.vllm.ai/en/stable/features/quantization/gguf/ .
- Implement llama.cpp first. Adding vLLM later remains possible, but is not current implementation scope; no multi-backend framework is authorized or needed upfront.
- Do not infer that the local NVFP4/MTP files work in either backend from generic GGUF, NVFP4, or speculative-decoding support. Exact artifact, architecture, kernels, and backend build compatibility remain unverified.
- Preserve backend-specific controls rather than reducing both engines to a lowest-common-denominator settings schema. The initial launcher passes native llama-server arguments through; image environment variables supply agreed defaults.
- llama.cpp now runs in Docker and has passed real Q4_K_M load/API/configuration smoke checks. vLLM remains unimplemented; no quality or speed comparison has been measured.

## Initial inference container

- User approved the proposed startup settings/defaults and explicitly prioritized context size and KV-cache precision as sweep variables.
- Implemented `Dockerfile`, `.dockerignore`, and `inference.py`. Detailed usage, pinned runtime identity, defaults, caveats, and verification are in `inference_container.md`.
- Official CUDA 12.8.1 server image pinned by digest, with defaults: context 8192, full GPU offload, Flash Attention on, f16 K/V cache, GPU KV offload, batch 2048, microbatch 512, one slot, continuous batching on, runtime-selected CPU threads, fit off, prompt caching off, context shifting off, warmup on, speculative decoding off, metrics on.
- Launcher accepts a host GGUF path, host bind IP/port, container name, and native inference arguments after `--`. Uses a read-only model-directory mount and localhost-only publication by default. No third-party Python dependencies.
- Verified Docker build, model readiness, non-streaming and streaming chat, metrics, zero prompt-cache reuse on a repeated request, and visible rejection of an invalid cache type.
- Verified a sequential override run on port 8081: context 16384, q8_0 K/V, no warmup. Runtime logs confirmed actual q8_0 KV buffers, Flash Attention, and 66/66 layers offloaded to RTX 5090.
- Model metadata reports `qwen35` architecture despite the Qwen3.8 filename. Loading warns about unused blk.64/nextn tensors. Preserve this observation; MTP and model provenance are not established.
- The original 8192-context `localllmbench-inference` container is now stopped and retained. Active container: `localllmbench-inference-128k` on localhost port 8080, Q4_K_M weights, context 131072, q8_0 K/V, Flash Attention, full offload. Image defaults unchanged.
- High-context logs confirmed 66/66 layers offloaded, 4352 MiB attention KV buffer, and separate recurrent state. Short reasoning-enabled requests and BigCodeBench samples ran successfully; actual filled-128K workloads and paging behavior remain unverified.
- No sweep orchestrator, automatic retries, or other backend integration added.

## First benchmark integration

- User approved simple seeded task selection, Complete prompts, reasoning-enabled evaluation, and host-Docker access for the trusted benchmark controller to create sibling execution sandboxes.
- User specifically wants testing at intended deployment contexts (128K+) rather than extrapolating from small-context configurations. The benchmark verifies a configurable minimum server context, initially 131072, without altering the inference server.
- Implemented `benchmark.py`, `benchmark_config.json`, and `benchmarks/` controller/configuration/scoring/sandbox files. Configurable count, subset seed, thinking mode, output budget, sampling, generation seed, concurrency, and timeouts.
- Defaults use 64 tasks / subset seed 42, thinking on, 8192 total output tokens, temperature 1.0, top-p 0.95, top-k 20, min-p 0.05, generation seed 42, concurrency 1. Sampling reflects the observed model defaults and can be changed; not a universal preset.
- Hash-ranked task IDs make selection independent of row order, model, and generation settings. Selection manifests retain exact IDs and raw/canonical dataset fingerprints. Do not use the pre-existing stratified sample as this project's sample.
- Revisit guaranteed multi-category coverage with the user later. Simple selection does not guarantee domain coverage or prefer easier tasks.
- Generated Python runs only in per-sample non-root Docker sandboxes with no network, host mounts, GPU, or Docker socket; CPU/memory/process limits and capability restrictions are configured. Live sandbox inspection verified these properties. Controller has broad host-Docker privileges by design; this is not Docker-in-Docker.
- Canonical-solution preflight detects dataset/environment problems before model generation. Final code alone is scored; reasoning is retained separately. Errors/truncation mark runs incomplete while other samples continue.
- Verified a real two-task run (`BigCodeBench/19`, `/578`): both passed, separate reasoning/final-text blocks retained. A deliberate one-token-budget run continued through two truncations and produced incomplete/null-score output with exit 1.
- Full 64-task generation has not been run; two-task success is integration evidence only.
- Results retained under `results/20261103T183309Z-01608898/` (successful smoke) and `results/20261103T183600Z-12fc9b40/` (intentional truncation smoke).

## Modular benchmark suite

- User explicitly required one folder per benchmark, configurable benchmark selection, and all registered benchmarks by default. BigCodeBench must not dictate the shared controller's data, prompts, scoring, or results.
- `benchmarks/registry.py` is the explicit registry. Each benchmark has a host-safe `definition.py` (options, declared input paths, preparation/provenance) and container-only `task.py` (Inspect task and benchmark-specific summary). The contract is documented in `benchmark_container.md`.
- BigCodeBench implementation, sandbox, and benchmark dependencies now live under `benchmarks/bigcodebench/`. Shared host/configuration/runner code contains no BigCodeBench protocol logic.
- Optional `benchmarks` configuration list selects names in order; omitted/null runs all registered entries. CLI `--benchmarks NAME ...` overrides selection; `--list-benchmarks` lists entries. Unknown, empty, or duplicate selections fail.
- Benchmark-specific settings live under `benchmark_options.<name>`. BigCodeBench owns `dataset_path`, `selection.count`, `selection.seed`, and `test_timeout`. Old root settings and global count/seed CLI flags were removed, not aliased.
- Shared settings remain inference connection, generation, request execution, and output location. Only selected benchmark inputs are mounted/prepared. Docker socket access is conditional on selected benchmark needs.
- Benchmarks run sequentially with separate `results/<run>/<benchmark>/manifest.json`, `summary.json`, and `inspect/` outputs. Root summary records suite/benchmark statuses and child-summary locations; it does not average unrelated metrics. An incomplete benchmark does not stop later benchmarks; interruption/shared startup failure does.
- Twenty-three configuration/registry/selection regression tests passed. CLI default-all/explicit previews retained the same 64-task selection and canonical dataset fingerprint.
- Rebuilt the image and ran two real reasoning-enabled BigCodeBench tasks through the modular path: both passed, native reasoning/final text retained. Output: `results/20261103T184944Z-95d186f9/`.
- A throwaway second registration of the real module verified multi-benchmark dispatch, isolated outputs, continuation after a missing-corpus failure, explicit-selection exclusion, and incomplete/null-score handling after a deliberate one-token cap. No extra benchmark was added.
- BigCodeBench is still the only registered benchmark. No sweep manager, UI, new metrics pipeline, or additional benchmark was implemented.

## Comparison design — confirmed

- Settled terminology is recorded in [GLOSSARY.md](GLOSSARY.md). The user confirmed shared understanding at the end of the interview; these are agreed design decisions, not implemented orchestration behavior.
- Runs may vary models and multiple serving settings freely. No controlled/deployment comparison modes or restrictions are required. Preserve each combination's exact model/settings and outcomes without attributing differences to an isolated variable.
- Each benchmark's protocol and selected problems remain the same across every model/serving combination in a run. Sampling parameters, generation seed, reasoning mode, and output-token budget also remain fixed across those combinations.
- Evaluate current workloads at intended deployment context capacity. Short-prompt measurements at a high allocation are useful for those workloads, but do not establish long-input speed or retrieval quality. The user plans separate long-context benchmarks later; those do not block the first end-to-end workflow.
- Feasibility classification, VRAM monitoring, and automated resource safeguards are deferred. Existing execution failures remain visible as incomplete results; successful startup alone does not establish full-context feasibility.
- Canonical hierarchy: **Run** is the complete end-to-end workflow; **Trial** is one model/serving-configuration execution across all selected benchmarks; **Benchmark evaluation** is one benchmark's execution/results within a trial. Earlier notes and current implementation use run/sweep more loosely; code/output terminology has not yet been migrated.
- Run planning evaluates every model against every requested serving configuration. The user subsequently requested per-setting value arrays and automatic configuration expansion in the recipe, superseding the earlier decision to defer parameter grids. The proposed recipe preserves explicit configurations as all-scalar blocks. Do not automatically remove combinations based on predicted GPU fit.
- User deferred additional validation, fast-fail checks, and complicated retry/error-handling machinery. On execution failure, retain available diagnostics/results, mark the affected work incomplete, and continue with the remaining work. No automatic retries.
- Each trial uses fresh inference and benchmark-controller containers, reusing built images. Keep that trial's inference server alive across its selected benchmarks, then clean up trial containers before starting the next trial. Trials execute sequentially.
- The user confirmed the comparison model and minimal trial lifecycle. The design interview is complete; the user will choose the next implementation step.

## YAML recipe specification

- User selected YAML for readable, commented recipes and approved named configuration groups. References: [recipe format and defaults](recipe.md), [JSON Schema](../recipes/recipe.schema.json), and [YAML example](../recipes/qwen-kv-sweep.yaml). The obsolete JSON example was removed.
- Configuration groups accept scalar native llama.cpp settings or lists of alternatives. Expand each group independently, concatenate the resulting configurations, then evaluate every model against them. Groups do not inherit from each other; benchmark/generation settings remain fixed.
- `kv_quantization` pairs K/V by default, with optional atomic asymmetric pairs. Native independent K/V axes are also possible, but cannot be mixed with that shorthand in the same group.
- Checked the pinned image's actual server help: cache type names include `q8_0`, `q4_0`, and `bf16`; the recipe uses exact names rather than `q8`, `q4`, or uppercase aliases. The two example GGUF filenames were verified on disk.
- Schema/default checks passed against safely parsed YAML, existing benchmark defaults, and eight native image defaults. The example expands into eight ordered trials: two models times three 128K cache choices plus one 256K/q4 configuration per model. Atomic K/V and multi-argument mappings were verified; ten malformed/conflicting variants were rejected, including YAML on/off boolean coercion.
- Defaults are specified, not implemented by a recipe loader: schema annotations do not populate fields. Shared defaults mirror existing benchmark behavior; native defaults remain image-owned rather than being blindly injected as CLI arguments. Context size stays explicit, and KV shorthand has no injected default that could override native K/V settings.
- The example selects all registered benchmarks by default; currently this means 64 BigCodeBench problems per trial. No model trials were launched and feasibility of the example combinations is not established. No orchestrator, new runtime dependency, or changes to existing launchers/image defaults were added.
- User rejected a separate recipe minimum-context threshold as redundant. The YAML example and schema now use only each group's `--ctx-size` for context configuration. Existing standalone benchmark `inference.minimum_context` code/configuration is intentionally unchanged during schema review; remove that legacy guard during orchestrator integration so smaller requested contexts are not rejected by a hidden default.

## Progress

### 2026-11-03

- Read the initial requirements and working constraints.
- Inspected the current project folder; it was empty.
- Created this progress document as requested.
- Recorded the user's inference/benchmark separation, Inspect AI choice, model/settings sweep workflow, and broad inference-metrics requirement.
- Updated the architecture decision: the orchestrator runs directly on the host, while inference and benchmarks remain containerized. Windows first; Linux implementation details deferred.
- Selected llama.cpp as the first inference backend; deferred any vLLM integration.
- Built and exercised the initial inference container against the real Q4_K_M model; documented the agreed configuration surface and observed limitations.
- Recorded the requested future localhost sweep-status/results UI; no UI implementation started.
- Implemented and smoke-tested the first benchmark container; moved the active inference deployment to 131072 context with q8_0 caches and documented the new workflow.
- Refactored benchmarks into independent modules and added configurable/default-all suite selection; verified shared dispatch and the real BigCodeBench path.

## Next step

Use the YAML recipe specification for the next orchestrator CLI iteration when the user directs implementation. Current launchers do not consume recipes yet. Keep deferred monitoring, UI, advanced compatibility validation, and retry machinery outside the initial implementation.
