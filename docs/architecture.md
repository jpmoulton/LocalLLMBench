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

That is the `nvidia-container` runtime, the default. On an Apple Silicon Mac the `metal-native` runtime replaces
only the inference container; see [Runtimes](#runtimes) below.

## Stages of a candidate

`admit → hash → plan → start → ready → verify → evaluate (speed probes, then benchmarks) → cleanup → report`

Each stage is bounded and recorded in `stages.jsonl`. *Admit* refuses to start if other processes already hold
more VRAM than `limits.max_foreign_vram_mib` (on a Mac: if the GPU is already busy, memory pressure is high or
there is not enough unified memory available): a measurement taken beside a game or another server is not a
measurement. *Verify* reads the server's own `/props`, `/slots` and startup log. Both runtimes run exactly these
stages.

## Runtimes

`ContainerRunConfig.runtime` names what serves the model, and `containers/runtime.py` is the one place that turns
that name into consequences (required permissions, refused and unenforced settings, which runner runs it), so the
CLI, sessions and sweeps cannot disagree about them. A config that names no runtime is `nvidia-container` and
serialises, fingerprints and runs byte-for-byte as before runtimes existed.

| | `nvidia-container` | `metal-native` |
|---|---|---|
| Inference server | pinned CUDA `llama-server` image under Docker Compose | pinned `llama-server` executable run directly on the Mac, Metal offload |
| Pinned by | image digest, `help_sha256`, `build_info` (`image-bundle.json`) | executable SHA-256, a digest over every `lib*.dylib`/`lib*.so` and Metal shader file (`*.metallib`, `*.metal` and its headers) beside it, `help_sha256`, `build_info` (`native-bundle.json`) |
| Evaluator | host process or its own container | host process only |
| Coding sandbox | Docker worker containers via the host broker | the same, Docker in a Linux VM (e.g. Colima); blocked without it |
| Memory evidence | VRAM in use after load (`nvidia-smi`) | unified memory: server `phys_footprint`, Metal buffers, host available memory, swap, pressure, power |
| Admission | foreign VRAM | host available memory vs model + reserve, macOS memory pressure, GPU utilisation |
| Limits | Docker cgroup memory / CPU / pids | `native_limits` checked by the runner (admission and a memory watchdog) |
| Permission | `allow_container_execution` | `allow_native_execution` (plus `allow_container_execution` for the coding sandbox) |
| `runtime_revision` | `<build_info>+<sha12>` | `metal:<build_info>+<sha12>` |

The native runner (`containers/native.py`) subclasses the container runner and replaces only the primitives that
touched Docker, so both runtimes produce the same run directory (`config.json`, `stages.jsonl`, `preflight.json`,
`plan/server-argv.json`, `verify.json`, `evaluation.json`, `settings-evidence.json`, `cleanup.json`,
`result.json`, `reports/`) under the same rules; a native run adds `plan/native-launch.json`, `logs/server.log` and
the unified-memory evidence. The `metal:` prefix on `runtime_revision` means a CUDA and a Metal result of the same
llama.cpp commit are never the same backend to the analysis, so they are never pooled or paired.

### Process model

```text
host (llmbench)  ── owns the GPU lease, the result store, the coding broker, and the one llama-server it starts
  │
  ├─ llama-server   child process in its OWN session / process group, bound to 127.0.0.1 on a port chosen at start,
  │                 scrubbed environment (no LLAMA_*, GGML_*, HF_*, DYLD_*, MTL_*/METAL_* or proxy
  │                 variable can change what the argv says or what code is loaded),
  │                 output drained into a bounded log; the model file is read in place from the host path
  ├─ evaluator      host process, talks to the server over loopback
  ├─ memory watchdog  samples phys_footprint, swap and pressure while the evaluator runs; reports, never kills
  └─ coding workers the same sandboxed Docker containers as on NVIDIA, through the same host broker
```

Starting the server in its own session means a Ctrl-C in the terminal reaches the harness, which then stops the
server deliberately, instead of reaching the server directly. The watchdog only reports a violation; the runner
decides, and the only thing it ever signals is the process group of the child it started and has not yet reaped,
so a recycled pid can never be hit.

### Trust boundaries

* **The executable is identity, not a path.** `prepare --runtime metal-native` hashes the executable and every
  library beside it, and any Metal shader library a build without an embedded one loads from there, proves from
  the Mach-O load commands that those hashed files are the only non-system code dyld will load for it (each image
  read at the slice an arm64 process loads, never an arm64e slice listed first), checks them against the
  installer's `install-manifest.json` when there is one, and runs it only as `--version`, `--help` and
  `--list-devices`. Admission re-hashes all of it, proves the linkage again and re-reads `--version` and `--help`
  before a single flag is trusted; a changed file is a refusal, not a new baseline.
* **A separate permission.** `allow_native_execution` is its own key in `runtime-policy.json`: a policy written for
  the NVIDIA containers never authorises starting a host binary.
* **Generated code never runs on the host.** The evaluator is a host process on a Mac, but code a model writes is
  still only ever executed by the broker in a network-less, read-only worker container. When the Docker sandbox is
  unavailable the code-executing suites are planned or recorded as `blocked` with the reason; there is no host
  fallback.
* **Loopback only.** The server is told to listen on `127.0.0.1`, never on every interface of a laptop, and a
  server whose own `listening on` line names any other address is refused.

### Cleanup proof

Cleanup sends SIGTERM to the server's process group, waits `native_limits.stop_grace_seconds`, then SIGKILL, and
reaps the child. It is verified only by positive proof of absence: the child reaped, its process group empty, no
process whose command line carries this attempt's `--alias` and `--port`, and the port refusing connections
(`cleanup.json`: `server_exit`, `processes_remaining`). Anything less is `cleanup-uncertain`, exit code 4, and the
campaign stops, exactly as for a container that would not go away. A SIGKILL the harness did not send is recorded
as such.

### One lease for both

Both runtimes take the same GPU lease file before admission, so an NVIDIA and a native run, or two native runs,
can never overlap on one machine. What the file records depends on who takes it. A session (`tune`, `resume`,
each `optimize` stage, and `scripts/sweep_models.py`, which runs `tune`) takes it once as its campaign lock and
records only its pid and start time; its candidates run under that lock and never rewrite it. `llmbench
candidate` (also each entry `scripts/run_prepared_sweep.py` runs), and each `sample` attempt, takes it per
candidate, as `container-run:<attempt>` or `native-run:<attempt>`, and a native one adds `runtime` and the
llama-server's `server_pid` once the server has started. A stale lease left by a killed
run is described by `llmbench doctor`, and by the refusal of the next run, with the check that fits what it
records: `ps -p <server_pid>` when the server's pid is there, `pgrep -fl llama-server` for a native run without
one, `docker ps -a --filter name=llmbench-` for a container run, and both of the last two for a session's lock,
which names no runtime ([A run that was killed](usage.md#a-run-that-was-killed)). Nothing ever deletes a lease
for you.

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
logs, the exact Compose project (or native launch record) and server argv, and reports. Reports are views; the
store is the record.
`llmbench analyze` re-scores a finished run from the store under a different acceptance policy.

A candidate is *eligible* when it meets the policy (speed floor, per-category score floors, verified settings and
context, non-inferiority to the baseline). Only categories a benchmark actually covered are judged; the rest are
recorded as `unmeasured_categories`, and a candidate with nothing measured is never eligible. A *winner*
additionally needs a passing holdout run; completed holdouts are reused on `resume`.
Public suites partition released data, so a holdout pass shows item-disjoint agreement, not freshness.

## Reproducibility

Images are pinned by digest, a native server by the SHA-256 of its executable and libraries, datasets by per-file
SHA-256, model files by SHA-256. A session records the hash of
the package source (`environment_hash`); editing the source mid-session invalidates `resume` by design. The engine
runs at temperature 0 with a fixed seed.

## Exit codes

`0` ok · `2` invalid input or refused by policy · `3` the run did not complete · `4` cleanup could not be verified
