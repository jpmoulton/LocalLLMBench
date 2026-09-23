# Usage

All commands are `llmbench <command>` (or `python run.py <command>` from a checkout). Commands that start
containers, or a native `llama-server` on a Mac, need `runtime-policy.json` in the working directory; without it
they exit 2 and do nothing.

## Preparation and retries

Stage datasets before preparing the evaluator image when using container evaluation. `prepare` discovers
`artifacts/benchmark-datasets/`, copies its regular files into the build context, rejects links, and records
SHA-256 hashes. An empty benchmark directory is created when no datasets are staged.

The wheel builder checks the chosen interpreter for `pip`, `setuptools` and `wheel` before Docker work. Missing
dependencies produce an installation command; nothing is installed automatically. Repeating a failed preparation
in the same output directory builds a fresh private context and preserves earlier contexts and unrelated files.
A same-version wheel is replaced only after a successful fresh build. Completed `image-bundle.json` outputs are
not overwritten: use a new output directory to prepare another completed bundle.

## Tuning one model

```bash
llmbench tune --model /models/my-model.gguf --output runs/my-model --budget-seconds 7200
```

The session is derived from the file: it is hashed, its GGUF header is read (architecture, training context,
whether it carries MTP draft layers, chat template), and a search space is built:

| Axis | Default members |
|---|---|
| KV cache | `f16`, `q8_0`, `q4_0` |
| Speculation | `none`, plus `draft-mtp` when the model has MTP layers |
| Reasoning | `off`, `on` |
| Context | 8192, 32768, 131072 (those the model supports) |
| Quantization | one per GGUF when `--model` is a directory of files of the *same* model |
| Offload | all layers on GPU, KV on GPU (extra members are opt-in, below) |

Candidates vary **one axis at a time** from the baseline, in a fixed order with context last. Before anything
starts, the benchmark plan is printed and written to `benchmark-selection.json`: which public suites were found,
which were selected, which were skipped and why. Read it — it is how you learn a suite is missing *before* a
two-hour run rather than after.

`llmbench resume --output runs/my-model` continues an interrupted session under its original deadline.

The files in `examples/` are templates for everything *except* the model: their model block is a placeholder
(`/models/your-model.gguf`, a zero hash) that `tune --model` replaces with what it reads from your file. To run one
directly with `candidate`, fill in `host_path`, `size_bytes` and `sha256` first; a mismatch is refused.

### A context range

```bash
llmbench tune --model m.gguf --output runs/long --context-floor 131072 --context-ceiling 261376
```

Both bounds are **usable input tokens**; the template and output reserves are added on top, so "supports 128K"
means about 128K tokens of real input ran. A long tier is given a workload it can finish: above about 60K input
tokens the candidate keeps the warm-up and two speed repetitions, the NIAH variants that fit (the multi-needle one
first), and evaluation bounds sized to the prompt; the session's per-candidate wall is raised to match when the
budget allows. A 262144-token candidate is bounded at about an hour; its five filled requests measured about four
minutes each on a 27B model. A ceiling the model's training context cannot hold is refused, not shrunk. Long tiers
need quantized KV on most GPUs — see "Choosing the candidates yourself".

### Choosing the candidates yourself

A session config (`session-config.json` in any run directory, or `examples/session.json`) can name a proposal
file instead of the built-in schedule: set `"proposal_mode": "file"` and `"proposal_file": "<path>"`. The file is a
JSON list. The first entry is the baseline and fixes every axis; each later entry changes one family:

```json
[
  {"family": "baseline", "changes": {"quantization": "Q4_K_M", "kv_pair": ["q4_0", "q4_0"], "ctx_tier": 4864,
     "spec_type": "none", "reasoning": "off", "batch_size": 2048, "gpu_layers": "all", "kv_offload": true}},
  {"family": "speculation", "changes": {"spec_type": "draft-mtp"}},
  {"family": "context", "changes": {"ctx_tier": 262144}}
]
```

Two reasons to do this. Context candidates inherit the **baseline's** KV precision, so a `q4_0` baseline is what
lets a 262144-token tier fit in VRAM. And candidates run in file order, so put what you care about first: a budget
cuts from the end. Run it with `llmbench tune --config my-session.json --output runs/x`.

### CPU / RAM offload

`search.gpu_layers` (e.g. `["all", 48]`) and `search.kv_offload` (`[true, false]`) add offload candidates to a
session config. The report shows where the KV cache really ended up — `gpu`, `ram` or `split` — from the server
log. Some quantizations (NVFP4, for one) have GPU kernels only and cannot be offloaded.

## Several models

```bash
python scripts/sweep_models.py --models-dir /models/candidates --output runs/compare --budget-seconds 7200
```

One session per file, sequentially, then `cross-model-report.md`: every candidate of every model, failures
included, with the tensor types each file really contains and a warning when chat templates differ (prompt-shaped
scores are then not strictly comparable). Interrupting it forwards the interrupt and waits, so the running
session removes its own containers.

## Coding benchmarks

EvalPlus and Aider Polyglot execute generated code and therefore need the sandboxed worker image and a broker.

```bash
# 1. Build the worker image (Python + numpy + pytest, Node + TypeScript + a pinned jest toolchain).
python scripts/build_worker_image.py --output artifacts/container-prep/worker-image \
  --base-image node:24-bookworm-slim@sha256:<digest>

# 2. Record it in the image bundle.
llmbench prepare --output artifacts/container-prep --worker-iidfile artifacts/container-prep/worker-image/worker-image.id \
  --inference <llama.cpp image@digest> --evaluator-base <python image@digest>

# 3. Check the harness against ground truth: every reference solution must pass, every untouched stub must fail.
python scripts/polyglot_reference_calibration.py --iidfile artifacts/container-prep/worker-image/worker-image.id
```

Then use a base config that carries a `broker` block and the worker image: copy `examples/candidate-coding.json`,
set `worker_image.image_id` and `worker_image.reference` to the id in `worker-image.id`, and pass it with
`--base-config`. `broker.max_requests` must cover one request per EvalPlus item and two per Polyglot exercise.

On an Apple Silicon Mac the worker is built for `linux/arm64` and pinned in the native bundle instead; see
[Apple Silicon (Metal) native runtime](#apple-silicon-metal-native-runtime).

Give reasoning models a real output budget (`generation.max_output_tokens`, 8192 or more) or their thinking is
cut off and scored as failure; `scripts/coding_report.py --root <sweep> [--suite aider-polyglot]` prints how many
responses hit the cap beside every score, and paired significance tests between candidates and models.

## Apple Silicon (Metal) native runtime

On an Apple Silicon Mac there is no NVIDIA GPU for the inference container, so the `metal-native` runtime runs a
**pinned llama.cpp `llama-server` executable directly on the Mac with Metal offload**. Everything else is the same
as on NVIDIA: the benchmark definitions, datasets, scorers, the stage machine, the settings evidence, the analysis
and the reports. Model-written code still runs only in the sandboxed Docker worker, never on the Mac itself; without
Docker the coding suites are recorded as **blocked**.

A config or a session states its own runtime. A candidate config says `"runtime": "metal-native"` (with a
`native_server` and `native_limits`, no `inference_image`), a session carries it in its base config, and a config
that names no runtime is an NVIDIA config, unchanged. `--runtime metal-native` *chooses* the runtime only where
no config exists yet; everywhere else it is at most a check, and several commands have no such flag:

| Command | `--runtime` | `--native-bundle` |
|---|---|---|
| `tune --model`, `scripts/sweep_models.py` | chooses; `metal-native` requires `--native-bundle` | required with `--runtime metal-native`, refused without it |
| `prepare` | chooses what to prepare | none (prepare writes the bundle) |
| `capabilities` | without `--config`, chooses which prepared capture to read; with it, must agree with the config | none |
| `validate`, `plan`, `candidate` | must agree with the config, which decides | `candidate` only, optional: its pins must equal the config's |
| `sample` | no such flag | optional: its pins must equal the config's |
| `tune --config` | refused: the session config states its runtime | refused |
| `resume` | no such flag | no such flag: a moved native bundle is passed as `--image-bundle` |
| `optimize` | no such flag | no such flag |
| `doctor` | no such flag | optional: whether the bundle's executable and libraries are still the pinned files |

`candidate` and `sample` run a metal-native config without a bundle too: before the server starts, admission checks
the config's own `native_server` pins against the files on disk (the executable and library hashes, the linkage
proof, and the build and help the executable's own `--version` and `--help` report). A bundle adds the comparison
with what `prepare` recorded.

### 1. Authorise native execution

Running a host executable is a separate permission from running containers, so a policy written for the NVIDIA
containers never silently authorises it. The machine's owner adds it to `runtime-policy.json`:

```json
{
  "allow_model_operations": true,
  "allow_inference": true,
  "allow_container_execution": true,
  "allow_native_execution": true,
  "reason": "Allow the pinned llama-server on this Mac, and the coding sandbox containers."
}
```

A native candidate needs `allow_native_execution`, `allow_model_operations` and `allow_inference`.
`allow_container_execution` is needed only for the coding sandbox: inspecting the worker image at prepare time,
and a session whose base config carries a `broker`. An absent `allow_native_execution` means false.

### 2. Install the pinned llama.cpp release

```bash
# From a copy you downloaded yourself (no network is used):
python scripts/install_llamacpp_macos.py --archive ~/Downloads/llama-b11011-bin-macos-arm64.tar.gz
# Or let the script fetch the pinned release asset (explicit opt-in):
python scripts/install_llamacpp_macos.py --download
```

The pin is llama.cpp `b11011` (commit `aa39d7a3e145a88202793a89462d65e94a5fc25f`), release asset
`llama-b11011-bin-macos-arm64.tar.gz`, 11156605 bytes, SHA-256
`9f88854d8216454a883f6d970e52a888d85c1ff321086c08c3d2f69362e0154d`, built upstream by the release workflow's
macOS arm64 job with `-DGGML_METAL_EMBED_LIBRARY=ON` and `-DGGML_RPC=ON` among its flags. The size and hash are
checked **before** anything is extracted; unsafe members (absolute paths, `..`, links leaving the directory) are
refused; the `com.apple.quarantine` attribute is removed; and the executable's own `--version` must report build
11011, commit `aa39d7a3e`. The result is `artifacts/native-runtime/llama-b11011/` (`--output` changes the root) with
an `install-manifest.json` recording the asset, its hash, the upstream build flags and every file's SHA-256. An
existing install directory is reused only when its files match; it is never overwritten, and one that holds
anything the release does not (a model file kept beside `llama-server`, say) is refused without that file being
read. `--download` is bounded in time as well as size: every socket read gets at most 120 s and never more than
what is left of 30 minutes for the whole transfer, however slowly the server sends. Running the executable, even
as `--version`, is native execution, so the installer needs `allow_native_execution` too.

### 3. Prepare: pin the executable

```bash
llmbench prepare --runtime metal-native \
  --llama-server artifacts/native-runtime/llama-b11011/llama-server --output artifacts/native-prep
```

`prepare` refuses anything but macOS on Apple Silicon. It hashes the executable and every `lib*.dylib`/`lib*.so`
beside it (the Metal backend lives in `libggml-metal`, so the executable alone would not pin what runs), plus any
Metal shader library a build without `GGML_METAL_EMBED_LIBRARY` loads from that directory (`*.metallib`,
`*.metal` and the `ggml-common.h`/`ggml-metal-impl.h` headers; the pinned release embeds its shaders and has
none). It proves from the Mach-O load commands, read at the arm64 slice an arm64 process loads, that those hashed
files are the only non-system code the executable loads (a Homebrew-style `../lib` layout is refused), checks the
directory against its `install-manifest.json` when there is one, runs the executable **only** as `--version`,
`--help` and `--list-devices` (with a scrubbed environment and a timeout), refuses unless a Metal (`MTL`) device is
listed, checks that every flag the harness can pass exists in this build, and writes `native-bundle.json` last. An
existing bundle is never overwritten: prepare again into a new directory. The saved `llama-server-help.txt`,
`llama-server-version.txt` and `llama-server-devices.txt` are what `llmbench capabilities` reads for a Metal
config (default directory `artifacts/native-prep`).

The Metal build's `--help` differs from the CUDA image's (it lists the `--rpc` option), so its `help_sha256`
differs too; the two runtimes are never mistaken for each other.

### 4. Stage the datasets

```bash
python scripts/stage_benchmark_datasets.py      # once, with network; writes artifacts/benchmark-datasets/
```

The Mac evaluator is always a host process, so it reads the staged corpora directly; pass the directory with
`--dataset-root` (the default is `artifacts/benchmark-datasets` when it exists).

### 5. Optional: the coding sandbox (Docker in a Linux VM)

EvalPlus, Aider Polyglot and the local coding fixtures execute generated code, which only ever happens in the
sandboxed worker container. On a Mac that needs Docker in a Linux VM, for example Colima:

```bash
brew install colima docker
colima start --cpu 2 --memory 2              # the VM's memory comes out of the same unified pool as the model

# Build the worker natively for arm64 (no emulation); use the official multi-platform index digest.
python scripts/build_worker_image.py --platform linux/arm64 --output artifacts/native-prep/worker-image \
  --base-image node:24-bookworm-slim@sha256:<index digest>

# Check the harness against ground truth in THIS sandbox: every reference passes, every stub fails.
python scripts/polyglot_reference_calibration.py --iidfile artifacts/native-prep/worker-image/worker-image.id

# Pin the worker together with the server. A bundle is never overwritten, so after step 3 this is a new
# directory; pass its native-bundle.json to tune from then on.
llmbench prepare --runtime metal-native \
  --llama-server artifacts/native-runtime/llama-b11011/llama-server --output artifacts/native-prep-coding \
  --worker-iidfile artifacts/native-prep/worker-image/worker-image.id
```

`prepare` probes the sandbox without side effects (Docker CLI present, daemon reachable, the pinned worker image
present locally, container execution allowed) and records the verdict in the bundle's `sandbox` block. Derivation
plans from that verdict: when it is not `available`, the three code-executing suites are planned as `blocked` with
the reason (`the Docker sandbox is unavailable (...); generated code is never executed on the host`), not
selected, and the base config's `broker` is dropped. Starting Colima later does not change a bundle: prepare again
into a new directory. To actually run the coding suites, also give `tune` a base config with a `broker` block
(copy `examples/candidate-coding.json`, point `worker_image` at the arm64 image, pass it with `--base-config`). A
candidate whose config carries a broker probes the sandbox again before evaluating; if it has gone away, the coding
rows are recorded as `environment_error` with reason `sandbox_unavailable: ...`; the candidate, session,
cross-model and coding reports print the category as `blocked`, never as a score of zero (the rows still count as
failures in the denominator, so such a candidate cannot meet a coding floor).

### 6. Check before running (free)

```bash
llmbench validate --config my-metal-candidate.json
llmbench plan --config my-metal-candidate.json           # runtime, native server argv, unsupported/not-enforced
llmbench capabilities --config my-metal-candidate.json   # reads artifacts/native-prep by default
llmbench doctor --native-bundle artifacts/native-prep/native-bundle.json
```

For a Metal config `plan` prints the native server argv (the model's host path, `--host 127.0.0.1`, and a port
placeholder; the real port is chosen when the server starts) and no Compose project. `doctor` adds a static
runtime block: platform, total memory, whether `docker` is on the PATH, the GPU lease (see
[A run that was killed](#a-run-that-was-killed)), and for a native bundle whether its executable and the library
and shader files beside it are still exactly the pinned ones. None of them starts anything.

### 7. Tune, sweep, resume

```bash
llmbench tune --model /path/to/model.gguf --runtime metal-native \
  --native-bundle artifacts/native-prep/native-bundle.json \
  --dataset-root artifacts/benchmark-datasets --output runs/mac-model --budget-seconds 7200

python scripts/sweep_models.py --models a.gguf b.gguf --output runs/mac-compare --runtime metal-native \
  --native-bundle artifacts/native-prep/native-bundle.json --dataset-root artifacts/benchmark-datasets

llmbench resume --output runs/mac-model

llmbench candidate --config my-metal-candidate.json \
  --native-bundle artifacts/native-prep/native-bundle.json --output runs/one-candidate
```

On `tune --model` and `scripts/sweep_models.py`, `--runtime metal-native` and `--native-bundle` go together and
either one alone is refused. `candidate` takes the bundle alone, because its config already names the runtime,
and `resume` needs neither: the session recorded its runtime and bundle (see the table in
[Apple Silicon (Metal) native runtime](#apple-silicon-metal-native-runtime)). Without
`--context-floor/--context-ceiling`, a Mac session is derived at 4096 usable input tokens (RULER's shortest
length, so every suite can be planned); the NVIDIA defaults are unchanged. The per-item time estimates the plan
uses were measured on the NVIDIA host, so a Mac plan multiplies them by a `planning_slowdown` of 3.0, recorded in
`session-config.json` and printed with the plan. `candidate --native-bundle` refuses a bundle whose executable,
library, build, help or worker-image pins differ from the config's, and `resume --image-bundle` (given a moved
native bundle) one whose executable, library or worker-image pins differ from the session's: a rebuilt server (or
sandbox) is a new session. Before its first model and after each one, a native sweep checks that the GPU lease is
released and that no process is still running the bundle's executable, and an interrupted one names any such
leftover before it exits; it never stops one itself. A Metal
candidate's `runtime_revision` is prefixed `metal:`, so a CUDA and a Metal result of the same llama.cpp commit are
never treated as the same backend.

### How memory is measured and labelled

Apple Silicon has one physical memory pool shared by the CPU, the GPU and every other application, so a Metal
candidate has **no VRAM figure** (`vram_used_mib_after_load` stays null) and reports never put its numbers in the
`VRAM MiB` column. Instead, `result.json → memory` (`"kind": "apple-unified"`) records:

| Where | What it is |
|---|---|
| `admission` | three samples before the load: host available memory, macOS memory pressure, GPU utilisation, swap, power |
| `after_load.server_phys_footprint_mib` | the llama-server process's `phys_footprint` (what Activity Monitor calls Memory); **includes** its Metal allocations |
| `after_load.server_rss_mib` | its resident set size, for comparison; the footprint, not the RSS, is the figure that counts the Metal allocations |
| `after_load.host_memory_available_mib`, `swap_used_mib`, `memory_pressure_level` | host-wide: other applications included |
| `after_load.power` | `pmset -g batt`: AC or battery and the battery percentage |
| `server_log.metal_resident_mib`, `metal_budget_mib` | the model, KV, compute and recurrent buffers the server logged on the Metal device, and the Metal working-set budget it reported |
| `during_evaluation` | the memory watchdog's summary: peak footprint, lowest available memory, highest pressure, swap growth, GPU utilisation, violations |
| `memory_breakdown_at_exit` | llama.cpp's own memory table printed when the server stopped |
| `coding_sandbox` | the sandbox probe for this candidate |

The raw samples are kept in `memory-samples.jsonl` and `verify.json`. In reports: the candidate report prints
`VRAM after load not applicable: unified memory` and a `Unified memory after load` line; the session table appends
`Runtime`, `Server footprint MiB (unified)`, `Metal buffers MiB`, `Swap growth MiB` and `Power` columns (`-` on
NVIDIA rows); the cross-model report adds the same columns when a Mac session is present and never takes a
lowest-VRAM figure across the two kinds of memory. `KV placement` `gpu`/`ram` on a Mac means Metal buffers versus
CPU buffers in the same memory.

### Admission, the memory watchdog and cleanup

There is no cgroup around a macOS process, so `native_limits` are checked by the runner instead:

| `native_limits` field | Default | Meaning |
|---|---|---|
| `memory_reserve_mib` | 1024 | admission needs the model file plus this much host memory available |
| `max_memory_pressure_level` | 2 | admission refuses above this macOS pressure level (1 normal, 2 warn, 4 critical) |
| `max_foreign_gpu_utilization_percent` | 50 | admission refuses when the GPU is already busier than this (median of the samples) |
| `max_server_footprint_mib` | none | the watchdog stops the server above this footprint; none = the Metal budget from the startup log |
| `max_swap_growth_mib` | 2048 | the watchdog stops the server when host swap grows more than this since admission |
| `sample_interval_seconds` | 1.0 | how often the watchdog samples during evaluation |
| `stop_grace_seconds` | 15 | SIGTERM to SIGKILL grace when stopping the server |

Admission refuses rather than guesses: missing telemetry is a refusal, and a busy GPU or a Mac under memory
pressure is a conflict to report, never a process to stop. The watchdog also stops the server at critical pressure
(level 4); the candidate then fails with `native_memory_watchdog: <reason>`. Nothing but the candidate's own
server is ever stopped.

The server runs in its own process group, bound to `127.0.0.1` on a port chosen when it starts, with every
`LLAMA_*`, `GGML_*`, `HF_*`/`HUGGINGFACE_*`, `DYLD_*` (dynamic-loader overrides such as `DYLD_INSERT_LIBRARIES`),
`MTL_*`/`METAL_*` (Metal debug and validation layers) and `*_proxy` variable removed from its environment, so
nothing but the recorded argv configures it and no unpinned code is loaded into it. Cleanup sends SIGTERM to that
group, then SIGKILL after the grace period, and is verified only when the child was reaped, its process group is
empty, no process carries this candidate's `--alias` and `--port`, and the port refuses connections. Anything less is
`cleanup-uncertain` (exit code 4) and stops the campaign, exactly like a container that would not go away. A
server that died of a SIGKILL the harness did not send is reported as such (macOS can do that under memory
pressure; the report does not claim it did).

### What is unsupported or not enforced

* **Refused:** `evaluator.mode: container` (the evaluator container's network cannot reach a server on the Mac's
  loopback; the evaluator is a host process) and `limits.gpu_device_id` other than `"0"` (Metal exposes one
  device, `MTL0`).
* **Accepted but not enforced, and listed in the candidate report:** `limits.inference_memory_mib` and
  `limits.inference_cpus` are Docker cgroup limits a host process does not have (the admission and the watchdog
  apply instead; CPU use follows `engine.threads`/`engine.threads_batch`), and `limits.max_foreign_vram_mib` is
  NVIDIA-only (the GPU-utilisation admission replaces it).
* **One Mac, one server at a time.** The same GPU lease as the NVIDIA runtime serialises every live run; see
  [A run that was killed](#a-run-that-was-killed) for a lease left behind.

### A run that was killed

Every run removes the lease when it ends, Ctrl-C included, with one deliberate exception: a `candidate` or
`sample` attempt whose cleanup could not be verified (`cleanup-uncertain`, exit code 4) keeps it, so nothing starts
beside what it may have left. A run that was killed (SIGKILL, a closed terminal, macOS ending it under memory
pressure) leaves the lease file behind too, and can leave its `llama-server` running, because the server has a
process group of its own. Every later live run then refuses to start, and both that refusal and `llmbench doctor`
name the lease's holder. Once the holder's process is gone, they say what to check before you delete the file.
That depends on what the lease records, which depends on the command that wrote it:

| Written by | Records | Check before deleting it |
|---|---|---|
| a session: `tune`, `resume`, each `optimize` stage, `scripts/sweep_models.py` | its pid and start time only: no runtime | both: `pgrep -fl llama-server` and `docker ps -a --filter name=llmbench-` |
| a metal-native `candidate`, or one `sample` attempt | `native-run:<attempt>`, and the llama-server's `server_pid` once it started | `ps -p <server_pid>`, and stop it only if it is that llama-server (a pid can be reused); `pgrep -fl llama-server` if no pid was recorded |
| an NVIDIA `candidate`, or one `sample` attempt | `container-run:<attempt>` | `docker ps -a --filter name=llmbench-` |

`scripts/run_prepared_sweep.py --run` starts each entry as a `candidate`, so its lease is a candidate's, not a
session's. While the holder's process still exists, the advice is to wait for that run or stop it first. Nothing
ever deletes the lease for you.

### Results on a laptop

A Mac shares its memory, its GPU and its power budget with everything else running on it. Before measuring, close
memory-heavy applications (the machine's owner decides what to close; the harness never stops anything of theirs),
and prefer AC power: every Metal result records the power source and the battery percentage, and a result measured
on battery or under memory pressure should be read as such. Swap growth is host-wide, so it can come from another
application. Compare Mac results with Mac results measured the same way; a Mac tok/s and an NVIDIA tok/s measure
different hardware, and a unified-memory footprint is never a VRAM figure.

On September 23, 2026, a MacBook Pro (M1, 8 GB unified memory, macOS 14.2.1, AC power) ran the pinned llama.cpp
`b11011` Metal server with two Q4_K_M GGUFs. The local run is at `runs/mac/sweep-main/` (ignored by Git); these
figures are from its cross-model report. Each model completed four candidates: baseline, q8_0 KV, q4_0 KV and
reasoning on. Each completed candidate received 24 tool-calling and 8 retrieval items, plus three speed probes.

| Model | GGUF SHA-256 | Baseline tok/s | Fastest measured tok/s | Baseline server footprint | BFCL baseline | RULER baseline |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B Q4_K_M | `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897` | 27.2 | 28.1 (q8_0 KV) | 1902 MiB | 0.708 | 0.250 |
| Qwen3.5-2B Q4_K_M | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` | 29.3 | 30.7 (reasoning on) | 1815 MiB | 0.500 | 0.750 |

The server footprint includes Metal allocations in unified memory. The sweep did not validate a preset: the
default 50 tok/s floor exceeded every measured speed, the item counts could not establish close quality
differences, and no holdout passed. Some candidates verified a 4096-token usable input; higher context tiers were
planned but did not run before the budget expired. EvalPlus and Aider Polyglot were unmeasured because the sweep's
base config omitted the broker, even though a Linux/arm64 coding worker was separately built and calibrated. These
results show a functioning native runtime and its observed tradeoffs, not a cross-model winner.

A later attempt to add one EvalPlus and one Aider Polyglot coding item used the already calibrated worker in a
1 GiB Colima VM, with no downloads. Native admission refused the first Qwen3-1.7B candidate before model load:
1250 MiB of host unified memory was available, below the model-plus-reserve requirement of 1568 MiB. The candidate
cleaned up and the VM was stopped. No Mac coding score resulted from that attempt; reducing the reserve to force a
run would discard the memory guard that kept the laptop stable.

## Screen, measure interactions, then confirm

`optimize` takes a **session configuration containing the full confirmation workload**, with real model hashes
and the desired frozen search space. Use a saved `session-config.json` from `tune` or a fully populated session
config; increase its benchmark selections before running if stronger quality evidence is needed.

```bash
llmbench optimize --config my-session.json --output runs/optimized --plan-only
llmbench optimize --config my-session.json --output runs/optimized \
  --screen-items 8 --finalists 2 --max-combinations 4 --budget-seconds 14400
```

The command runs sequential stages under one wall budget:

1. **Screen:** baseline plus the configured proposals, using the first `--screen-items` declared tasks per suite.
   Benchmark options and fixture seeds are unchanged, so screening is a nested subset of confirmation.
2. **Interactions:** valid pairwise combinations of shortlisted changes, each actually measured alongside the
   original baseline. Conflicting axis values are excluded. Model weights, context and generation settings stay
   fixed; combinations may change KV cache, MTP, reasoning, batching and offload. Set `--max-combinations 0` to
   disable this stage.
3. **Confirm:** the original baseline and up to `--finalists` non-baseline picks, using every task in the supplied
   configuration. Shortlisting alternates measured speed and worst-category quality extremes, not claims of
   statistical significance. Failed, synthetic, unverified and below-floor measurements cannot enter it.

By default, screening can use 40% of the wall budget and interactions 20%; confirmation gets the remaining time.
Without interactions, screening can use 50%. Stage bounds are checked before execution; an insufficient budget
is refused rather than silently shrinking the workload. Unused stage time remains available for confirmation.
Cleanup uncertainty stops all later stages.

`optimization-plan.json` records item counts and candidates; `optimization-report.json` links each stage's
standard session reports and carries only the confirmation stage's recommendation. `--plan-only` prints the plan
without creating output or requiring live authorization. Execution requires a new output directory. If interrupted,
the per-stage session artifacts remain available to `resume`; the outer optimization is not automatically resumed.

**Confirmation is not a new holdout.** It repeats the selected configurations on the full declared development
set. Existing non-inferiority and separate holdout requirements still decide what can be called validated.
If the configured full workload is no larger than the screening subset, the second stage adds repetitions,
not independent items. Supply enough distinct tasks for the quality tolerance you want to establish.

For manual proposal files, an explicit `combination` family accepts two or more serving axes, for example
`{"family": "combination", "changes": {"kv_pair": ["q8_0", "q8_0"], "spec_type": "draft-mtp"}}`.
It does not authorize changes to weights, input length, sampling, benchmark tasks, or provenance.

## Separate generation-sampling experiments

```bash
llmbench sample --config candidate.json --output runs/sampling-plan \
  --temperature 0.6 --temperature 0.8 --seed 42 --seed 43 --seed 44 \
  --top-p 0.95 --budget-seconds 7200 --plan-only
llmbench sample --config candidate.json --output runs/sampling \
  --temperature 0.6 --temperature 0.8 --seed 42 --seed 43 --seed 44 \
  --top-p 0.95 --budget-seconds 7200
```

The input is a complete **candidate** configuration, not a session. Defaults are temperatures 0.6/0.8, seeds
42/43/44, and top-p 0.95. One greedy control (`temperature=0`, `top_p=1`, first seed) precedes the deduplicated
stochastic grid. All other generation settings, engine reasoning, and benchmark fixtures are inherited unchanged.
Give reasoning enough output tokens in the candidate; this command does not increase its cap.

Both planning and execution require a new output directory. Planning records an immutable config and grid but
does not authorize or run inference. Live attempts run sequentially and stop when the remaining wall budget cannot
cover the next candidate's minimum stage bounds. `sampling-report.json` records per-suite/category scores, item
counts, available truncation evidence, failures, cleanup evidence and unattempted grid members.

Seed means/ranges/standard deviations are descriptive only. Replaying an item with a new generation seed does
not create an independent benchmark item. A group receives aggregate scores only when **every planned seed**
has verified measured evidence; incomplete groups keep null aggregates and explicit planned/measured seed lists.
Failed attempts have no scores. No winner, non-inferiority conclusion or validated preset is exported.

## Re-scoring without the GPU

```bash
llmbench analyze runs/my-model/reports/campaign.json --store runs/my-model --policy my-policy.json --output runs/my-model/rescored
```

The default score floors (tools 0.60, retrieval 0.60, coding 0.40) only reject broken configurations; ranking is
done by the paired comparison against the session's own baseline. Calibrate them to your model and re-analyse;
nothing is re-run.

**Why the winner is usually the baseline.** Validating any *other* candidate needs non-inferiority within
`max_quality_loss` (2%), and about 183 regression-free items per category before that can be shown at all; the
default plan uses about 24. The plan prints this before it starts. The report still names the faster or larger
candidate it measured but could not validate. To validate one, raise the item counts (`items_per_category`,
`item_limit`) or set a `max_quality_loss` your item counts can demonstrate.

## Driving it from a coding agent

The intended workflow is to give an agent a goal ("best coding quality at 50 tok/s or more, with 64K of context")
and let it work: check configurations for free with `validate`, `plan` and `capabilities`, run `tune`, read the
JSON reports, write a proposal file for the next question, and run again.
`.claude/skills/benchmark-model/SKILL.md` tells an agent how to run a session and, more importantly, how to
report one without overstating it. The same rules apply to people.

## Diagnostics

`scripts/llamacpp_manual_pilot.py`, `llamacpp_filled_context_probe.py` and `llamacpp_route_checks.py` probe an
already running llama.cpp server by URL. They are for investigating a server, not for producing results.
