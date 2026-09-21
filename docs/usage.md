# Usage

All commands are `llmbench <command>` (or `python run.py <command>` from a checkout). Commands that start
containers need `runtime-policy.json` in the working directory; without it they exit 2 and do nothing.

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

Give reasoning models a real output budget (`generation.max_output_tokens`, 8192 or more) or their thinking is
cut off and scored as failure; `scripts/coding_report.py --root <sweep> [--suite aider-polyglot]` prints how many
responses hit the cap beside every score, and paired significance tests between candidates and models.

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

`.claude/skills/benchmark-model/SKILL.md` tells an agent how to run a session and, more importantly, how to
report one without overstating it. The same rules apply to people.

## Diagnostics

`scripts/llamacpp_manual_pilot.py`, `llamacpp_filled_context_probe.py` and `llamacpp_route_checks.py` probe an
already running llama.cpp server by URL. They are for investigating a server, not for producing results.
