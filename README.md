# LocalLLMBench

**LocalLLMBench finds the best inference settings for a local model - both speed *and* quality - across different
quantizations and different serving settings, so you can pick the best trade-off for your situation.**

Is the Q4 fast enough, or is the Q6 worth the speed it costs? Does a quantized KV cache hurt tool calling? How much
context fits on your GPU, and how fast is it when it is actually full? Does speculative decoding double your speed
for free? The answers depend on your model, your hardware and what you need from them, so LocalLLMBench measures
them on your machine instead of guessing.

LocalLLMBench (the command is `llmbench`) runs **any GGUF model you have on disk** in a pinned llama.cpp container,
sweeps the settings that matter (quantization, KV-cache precision, speculative decoding, reasoning, context size,
CPU/RAM offload), and scores every variation on speed, VRAM, verified context and **official public benchmarks**:
BFCL (tool calling), RULER (long-context retrieval), EvalPlus MBPP+ and Aider Polyglot (coding). You get a report
of everything that was tried - the whole trade-off curve, not just a winner. What matters to you goes in as
input: a speed floor, minimum scores and a quality tolerance as the acceptance policy, and the context range you
need as part of the search.

Nothing is tied to a particular model. Point `--model` at a file: its size, hash, architecture, quantization,
training context and speculative-decoding support are read from the file itself, and the search space is derived
from them.

It is deliberately sceptical. A setting counts as *effective* only when the server was observed using it. A
context size is a claim only when a prompt that long actually ran. A benchmark item cut off by an output cap is
reported as cut off, never as a failure. Model-written code runs only inside a network-less, read-only container.

Abridged from a real session (a 27B model on one 32 GB GPU; yours will differ):

```text
candidate        tok/s   VRAM MiB   tools   retrieval   ctx     KV     speculation
baseline          70.0     21042    0.792     1.000      4864   q4_0   none
spec-draft-mtp   139.7     21895    0.792     1.000      4864   q4_0   draft-mtp    <- 2x speed, same scores
kv-f16            68.6     21263    0.792     1.000      4864   f16    none
ctx-66048         60.1     22388    0.792     1.000     66048   q4_0   none
```

## Requirements

| | |
|---|---|
| Python | 3.11+ on Linux, macOS or Windows |
| To run a sweep | Docker with an NVIDIA GPU visible to containers: Linux with the NVIDIA Container Toolkit, or Windows with Docker Desktop on WSL2. macOS runs everything *except* GPU containers: validation, planning, analysis, reports and the test suite. |
| Disk | about 6 GB for container images and 20 MB for benchmark datasets |

## Install

```bash
git clone https://github.com/jpmoulton/LocalLLMBench.git && cd LocalLLMBench
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[eval]"                            # use ".[eval,dev]" to run the tests
```

## Quick start

```bash
# 1. Allow GPU and container work on THIS machine. Without this file every live command refuses to run.
cp runtime-policy.example.json runtime-policy.json

# 2. Once: pull the pinned llama.cpp image, build the evaluator image, record what the server supports.
llmbench prepare --output artifacts/container-prep \
  --inference ghcr.io/ggml-org/llama.cpp@sha256:d84303da604d44b7656058d91728f623f87b24b3223abd8016b7cbadeadc8d5e \
  --evaluator-base python:3.12-slim-bookworm@sha256:<digest>

# 3. Once: fetch and pin the public benchmark datasets (checksummed; nothing is fetched at run time).
python scripts/stage_benchmark_datasets.py

# 4. Tune a model. It prints which benchmarks will run, and why any are skipped, before it starts.
llmbench tune --model /path/to/model.gguf --output runs/my-model --budget-seconds 7200

# Several models, one after another, with a cross-model report at the end:
python scripts/sweep_models.py --models a.gguf b.gguf --output runs/compare
```

`runs/my-model/reports/` holds the report as Markdown, HTML and JSON. An interrupted session continues with
`llmbench resume --output runs/my-model`; its deadline keeps running while it is down.

Coding benchmarks execute generated code, so they also need the sandboxed worker image:
[docs/usage.md](docs/usage.md#coding-benchmarks).

## Point an agent at it

LocalLLMBench is built to be driven by a coding agent (Claude Code, or anything that can run shell commands). The
idea: you say *"find the best settings for this model on this machine - I need at least 50 tok/s and 64K of
context"*, and the agent has the tools to get there and the guardrails to report it honestly.

What the agent gets:

* **One command that derives a search from a model file** (`tune`), and a **proposal file** format for choosing its
  own candidates when the default search is not what the question needs - long context on a quantized KV baseline,
  RAM offload, a quantization ladder ([docs/usage.md](docs/usage.md#choosing-the-candidates-yourself)).
* **Free checks before spending GPU time:** `validate`, `plan` and `capabilities` print the exact server argv and
  what the pinned llama.cpp build supports, without starting anything.
* **Machine-readable evidence:** every report is also JSON, every candidate records which settings the server was
  *observed* using, and `analyze` re-scores a finished run under a different policy without the GPU. An agent can
  iterate: run a session, read the results, write the next proposal file, run again. Every session has a
  wall-clock budget, and `resume` continues an interrupted one.
* **A skill that teaches the rules:** [.claude/skills/benchmark-model/SKILL.md](.claude/skills/benchmark-model/SKILL.md)
  covers how to run a session and how to report one: check for cut-off responses before quoting a score, never
  rank on noise, never call a context size supported unless a prompt that long ran, never invent a winner.

What the agent does not get: `runtime-policy.json` is the machine owner's authorisation and the agent is told never
to write it; one GPU workload runs at a time behind a lock; and model-written code only ever runs in the sandbox.

## Commands

| Command | What it does | Starts containers |
|---|---|---|
| `tune`, `resume` | Run or continue a tuning session for one model | yes |
| `candidate` | Run exactly one configuration | yes |
| `prepare` | Pull/build the pinned images and record server capabilities | yes |
| `validate`, `plan`, `capabilities` | Check a configuration; print the exact server argv and Compose project | no |
| `analyze` | Re-score a finished run under a different acceptance policy, without the GPU | no |
| `list`, `show`, `report`, `doctor` | Read recorded evidence, rebuild reports, show the environment | no |

Without installing: `python run.py <command>` is the same program.

## Reading the results honestly

* **Small item counts cannot separate close configurations.** 24 tool-calling items resolve roughly ±8 points, so
  two candidates one item apart are tied. `scripts/coding_report.py` prints exact paired (McNemar) tests for this
  reason. Raise the item counts before believing a small difference.
* **The validated winner is usually the baseline.** A winner must pass the acceptance policy *and* a holdout run,
  and any candidate other than the baseline must also be shown non-inferior to it - which takes about 183 items per
  category at the default 2% tolerance. The plan says so before it starts, and the report names the faster or
  larger candidate it measured but could not validate. A category no benchmark covered is `unmeasured`, never
  failed. `llmbench analyze` re-scores a finished run under a different policy without re-running anything.
* **Everything runs at temperature 0.** That is right for comparing settings and unfair to reasoning modes, which
  can loop under greedy decoding. Treat "reasoning did not help" as a statement about greedy decoding.

[docs/lessons.md](docs/lessons.md) lists the measurement traps this project fell into, so you can avoid them.

## Documentation

* [docs/usage.md](docs/usage.md) — sessions, context ranges, RAM offload, proposal files, several models (one per
  quantization), coding benchmarks, agents
* [docs/architecture.md](docs/architecture.md) — containers, trust boundaries, how evidence is recorded
* [docs/benchmarks.md](docs/benchmarks.md) — the four public suites, what is pinned, where we deviate from upstream
* [docs/lessons.md](docs/lessons.md) — measurement traps and what guards against each
* [docs/development.md](docs/development.md) — tests, layout, adding a benchmark

## Status

Beta. Exercised live on one machine (RTX 5090, Windows 11, Docker Desktop on WSL2) with llama.cpp build `b11011`;
the unit suite runs on Linux, macOS and Windows. Single-GPU NVIDIA and single-file GGUF models only.

MIT licensed. Vendored third-party data and its licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
