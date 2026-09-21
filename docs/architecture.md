# Architecture

## One candidate = one immutable configuration

A **candidate** is a model file plus a complete set of serving settings. It is never changed while it runs;
a different setting is a different candidate with a different fingerprint. A **session** (`llmbench tune`) is an
outer loop: derive a search space from the model, run candidates inside a wall-clock budget, analyse, optionally
validate the best ones on held-out items, and report everything.

```text
host (llmbench)  ── owns Docker, the GPU lock, the result store and the coding broker
  │
  ├─ inference container   llama.cpp server, pinned by image digest, model mounted read-only, private network
  ├─ evaluator             drives the benchmarks against the server (host process, or its own container)
  └─ coding workers        one short-lived container per test run: no network, read-only root, no capabilities,
                           non-root user, bounded memory / CPU / pids / open files / time
```

No container is ever given the Docker socket. The host starts and removes everything, and a candidate is not
complete until its containers and networks have been verified absent. When cleanup cannot be verified the run ends
with exit code 4 and the campaign stops, because the next measurement would share the GPU with a leftover.

## Stages of a candidate

`admit → hash → plan → start → ready → verify → evaluate (speed probes, then benchmarks) → cleanup → report`

Each stage is bounded and recorded in `stages.jsonl`. *Admit* refuses to start if other processes already hold
more VRAM than `limits.max_foreign_vram_mib`: a measurement taken beside a game or another server is not a
measurement. *Verify* reads the server's own `/props`, `/slots` and startup log.

## Evidence, not intent

Every requested setting gets one row in `result.json → settings`:

| status | meaning |
|---|---|
| `verified-api` | the server's API reports the value |
| `verified-log` | the startup log states it |
| `verified-behavior` | confirmed by a probe (for example an overflow request) |
| `argv-accepted` | the flag was accepted; its effect was not observed |
| `mismatch` | the server did something else — the candidate is not the configuration it claims to be |
| `unobserved` | nothing confirms it |

Context works the same way. `--context-floor/--context-ceiling` mean **usable input tokens**; template and output
reserves are added on top. A tier is a context claim only when a prompt of that size really ran
(`actual_context_verified`), and the report shows the largest prompt each candidate actually held.

## Coding benchmarks and the trust split

Generated code is never run on the host or in the evaluator. The evaluator writes a request into a spool
directory; the **host broker** validates it, stages files and runs a worker container.

The evaluator may send only a *patch* (the files a task allows it to edit) plus the task's id, revision and pinned
hash. Tests always come from the **host's** pinned copy of the dataset, and a hash the host does not hold is
refused — so an evaluator, or a model steering it, cannot weaken the tests it is scored by. A pass needs the test
runner's own summary as well as exit code 0, because a solution that calls `os._exit(0)` returns 0 with no tests
run.

## Storage and analysis

Each run directory holds a SQLite result store (`results.sqlite3`), raw request/response transport logs, server
logs, the exact Compose project and server argv, and reports. Reports are views; the store is the record.
`llmbench analyze` re-scores a finished run from the store under a different acceptance policy.

A candidate is *eligible* when it meets the policy (speed floor, per-category score floors, verified settings and
context, non-inferiority to the baseline). Only categories a benchmark actually covered are judged; the rest are
recorded as `unmeasured_categories`, and a candidate with nothing measured is never eligible. A *winner*
additionally needs a passing holdout run; completed holdouts are reused on `resume`.
Public suites partition released data, so a holdout pass shows item-disjoint agreement, not freshness.

## Reproducibility

Images are pinned by digest, datasets by per-file SHA-256, model files by SHA-256. A session records the hash of
the package source (`environment_hash`); editing the source mid-session invalidates `resume` by design. The engine
runs at temperature 0 with a fixed seed.

## Exit codes

`0` ok · `2` invalid input or refused by policy · `3` the run did not complete · `4` cleanup could not be verified
