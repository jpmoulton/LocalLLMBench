---
name: benchmark-model
description: Find the best llama.cpp serving settings for a local GGUF model with this repository's benchmarking suite (NVIDIA containers, or native Metal on an Apple Silicon Mac), and report the result without overstating it. Use when asked to benchmark, tune, compare or "get the best inference settings for" a model.
---

# Benchmark a model

## Before anything

1. `llmbench doctor` — all three `allow_*` flags in the runtime policy must be true (on a Mac, see below), or
   every live command exits 2.
   Never create or edit `runtime-policy.json` yourself; it is the machine owner's authorisation.
2. The GPU must be idle. Another inference server or a game holding VRAM makes admission refuse every candidate
   (`limits.max_foreign_vram_mib`). Ask the user before stopping anything of theirs.
3. One GPU workload at a time. Never start a second session while one runs.
4. Prerequisites exist: `artifacts/container-prep/` (from `llmbench prepare`) and `artifacts/benchmark-datasets/`
   (from `scripts/stage_benchmark_datasets.py`). See `docs/usage.md`.

## Run

```bash
llmbench tune --model /path/to/model.gguf --output runs/<name> --budget-seconds 7200
python scripts/sweep_models.py --models a.gguf b.gguf --output runs/<name>        # several models
llmbench resume --output runs/<name>                                              # after an interruption
```

Read the benchmark plan it prints first: which suites were selected, which were skipped, and why. A skipped suite
is a gap in the result — say so.

Need long context, RAM offload, more benchmark items, or coding benchmarks? Those are session-config and
proposal-file changes: `docs/usage.md`. Context candidates inherit the baseline's KV precision, so very long tiers
usually need a `q4_0` or `q8_0` baseline to fit.

To stop a run, interrupt the session (Ctrl-C) and wait. Do not kill it: a killed session leaves its container
holding VRAM. Afterwards `docker ps -a --filter name=llmbench-` must be empty.

Do not edit anything under `src/` while a session is running.

## On an Apple Silicon Mac

The Mac runs the pinned llama.cpp server natively with Metal (`docs/usage.md`, "Apple Silicon (Metal) native
runtime"). The rules above still apply; these are added.

* **Always pass the runtime explicitly:** `--runtime metal-native --native-bundle artifacts/native-prep/native-bundle.json`
  on `tune --model`, and the same pair on `scripts/sweep_models.py`; either flag alone is refused. There is no
  automatic switch. `resume` takes neither (the session remembers its runtime and bundle), `tune --config` refuses
  both (the session config states its runtime), and a `candidate` config names its own runtime (`--native-bundle`
  is optional there and must match it).
* **Permissions:** `llmbench doctor` must show `allow_native_execution` (plus the model and inference flags) as
  true; `allow_container_execution` only matters for the coding sandbox. Never add or change a permission
  yourself.
* **Prerequisites:** `artifacts/native-runtime/llama-b11011/` (from `scripts/install_llamacpp_macos.py`) and
  `artifacts/native-prep/native-bundle.json` (from `llmbench prepare --runtime metal-native`). Downloading the
  release (`--download`) and staging datasets use the network: ask first.
* **Coding suites need Docker** (for example Colima). Without it the plan lists EvalPlus, Aider Polyglot and the
  coding fixtures as `blocked`. Report them as blocked, with the reason: never as a score, never as a failure, and
  never run generated code on the Mac to fill the gap.
* **Memory is unified. Never call a footprint "VRAM"** and never compare it with an NVIDIA VRAM figure. Quote the
  server footprint (`phys_footprint`, which includes the Metal buffers), the Metal buffers and the Metal budget as
  what they are, and say that swap and available memory are host-wide.
* **Battery and memory pressure change results.** Every Metal result records the power source and the macOS
  memory pressure. If it ran on battery, under memory pressure (level 2 or higher), with swap growing, or beside
  other heavy applications, say so next to any speed you quote. Ask the user before closing anything of theirs;
  never stop their processes.
* **A Mac is slow at long context.** The plan sizes its work at three times the NVIDIA estimates and a Mac session
  defaults to 4096 usable input tokens; do not promise long-context results the plan did not schedule.
* **Stopping:** interrupt the session (Ctrl-C) and wait. Afterwards no `llama-server` from the bundle may be left
  running (`pgrep -fl llama-server`) and the GPU lease must be gone. If a lease is left, `llmbench doctor` names
  its holder and what to check: a session's lease records no runtime, so it asks for both `pgrep -fl llama-server`
  and `docker ps -a --filter name=llmbench-`. Report what it says; deleting the lease, or stopping a leftover
  server, is the user's decision.
* Mac numbers describe that Mac. Never rank a Mac tok/s against an NVIDIA tok/s as if it were a model difference.

## Report — the rules that matter

* **Check `finish_reason` before quoting any score below 1.0.** A response cut off at the output cap is not a wrong
  answer. `scripts/coding_report.py` prints cut-off counts; for other suites look at the transport logs.
* **A failed candidate has no scores.** Its zeros are artefacts of the failure; report it as unmeasured and give the
  failure reason.
* **Do not rank on noise.** State the item count. 24 items resolve about ±8 points, 60 about ±6. Candidates one or
  two items apart are tied. Use the paired tests in `scripts/coding_report.py` when comparing.
* **Never quote a pass rate "among the items that finished".**
* **The validated winner is usually just the baseline**, because validating anything else needs about 183 items per
  category (the plan prints the exact figure). The report's "measured better but NOT validated" line is screening
  evidence: present it as that, and your reading of it as *your reading*. Do not invent a winner, and do not call
  anything validated that the report does not. List a preset's `unmeasured_categories` whenever you quote it.
* **A context size is a claim only when `ctx ok` is true** — a prompt that long actually ran.
* Different chat templates between models make tool-calling scores not strictly comparable; the cross-model
  report flags this.
* Everything runs at temperature 0, which is unfair to reasoning modes. Say so if you report on them.

`docs/lessons.md` explains where each of these rules came from.
