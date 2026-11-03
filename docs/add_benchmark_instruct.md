# Adding a benchmark to LocalLLMBench

Use this guide when adding/registering a benchmark or changing a benchmark module's interface. It is intended for both developers and AI agents. Work on the requested benchmark only; benchmark choice and evaluation protocol remain user-directed.

## How the pieces fit

The host orchestrator, `orchestrate.py`, resolves a recipe and starts fresh inference/benchmark containers for each trial. The standalone launcher, `benchmark.py`, attaches a suite to an existing inference server. Both reuse launch construction. Inside the controller, `benchmarks/run.py` executes benchmarks sequentially through Inspect AI. Each benchmark supplies preparation, an Inspect `Task`, and result interpretation.

`benchmarks/registry.py` maps public names to Python import paths. The loader imports `<module>.definition` for configuration/preparation and `<module>.task` for Inspect execution. These are ordinary Python modules with agreed function names, not subclasses or separate services. Missing interface members fail at runtime; registration alone does not prove an implementation works.

Read the existing implementation for patterns, not as a universal benchmark protocol:

- [BigCodeBench definition](../benchmarks/bigcodebench/definition.py): options, local inputs, deterministic selection, provenance.
- [BigCodeBench runtime](../benchmarks/bigcodebench/task.py): Inspect task and coverage-aware summary.
- [Shared configuration](../benchmarks/config.py), [registry](../benchmarks/registry.py), and [runner](../benchmarks/run.py): actual shared contracts.
- [Container documentation](benchmark_container.md): inference prerequisites, build/run setup, security limits, and existing verification.

## 1. Establish the evaluation protocol

Before implementing, establish the dataset/version, prompt variant, selection policy, scoring rules, generation requirements, dependencies, and execution limits. Resolve material unknowns with the user rather than inventing a benchmark protocol.

Keep dataset selection independent of model identity, quantization, inference settings, and previous scores. If sampling is supported, define a deterministic default and preserve selected IDs and dataset identity so comparisons use the same problems. BigCodeBench's hash-ranking algorithm is one implementation, not a requirement that every benchmark use its schema or algorithm.

**Ready to implement:** inputs, expected outcomes, sampling behavior, and the distinction between incorrect answers and execution failures are defined.

## 2. Create the benchmark directory and definition

Choose a stable lowercase Python identifier, using underscores if needed. Create `benchmarks/<name>/definition.py` and `benchmarks/<name>/task.py`. Keep benchmark-specific helpers, optional `requirements.txt`, and optional sandbox configuration in that directory.

`definition.py` must be importable on the host with the standard library alone. It exposes:

| Member | Required behavior |
| --- | --- |
| `DEFAULT_OPTIONS` | Dictionary of benchmark-owned defaults. Shared configuration merges user overrides recursively into this dictionary and rejects keys absent from it. Declare every supported option here. |
| `PATH_OPTIONS` | Tuple of flat option names identifying local input files/directories. Use an empty tuple when none are needed. Each declared field must contain a path string. |
| `NEEDS_DOCKER` | Boolean indicating whether the controller needs host Docker-daemon access, usually to create execution sandboxes. All benchmarks still run inside the controller container. |
| `validate_options(options)` | Validate merged values and raise useful errors. Do not read datasets or require their paths to exist here. |
| `prepare(options)` | Return `(payload, provenance)` after read-only input preparation. |
| `SCORE_METRICS` (optional) | Map top-level numeric summary keys to `{label, direction, unit}` metadata. `direction` is `"higher"` or `"lower"`; only declared, finite scores from completed evaluations are eligible for per-benchmark leaders. |

All registered definitions and their options are loaded/validated during configuration loading, including unselected benchmarks. Keep imports and validation free of dataset access, runtime imports, and side effects.

Only selected input paths are checked and mounted. The launcher resolves relative paths against the configuration file's directory, mounts inputs read-only at `/datasets/<name>/<option>`, and rewrites those options for the controller. Use the supplied paths; do not reconstruct Windows paths inside the container. Nested path fields are not supported by `PATH_OPTIONS`.

`prepare()` runs on the host for standalone `--select-only` and orchestrator planning, and inside each controller against mounted paths.

The **payload** is benchmark-owned Python data passed to `make_task()`. The shared runner does not inspect or serialize it. **Provenance** must be a JSON-serializable, path-independent dictionary recording selection, input identity/version, and protocol details. The orchestrator freezes it in `plan.json`; each controller writes `manifest.json` and compares it exactly before generation. Exclude host/container paths, timestamps, and other incidental values. A mismatch makes the evaluation incomplete rather than selecting replacement tasks. Preparation is read-only; acquire assets separately.

**Definition complete:** it imports without Inspect installed, validates options without input access, and prepares reproducible inputs/provenance on host and container paths.

## 3. Implement the Inspect runtime

`task.py` may import Inspect and benchmark-specific dependencies. Implement both functions:

### `make_task(payload, provenance, options, generate_config) -> inspect_ai.Task`

Construct and return one Inspect `Task`; do not execute the evaluation here. The shared runner calls `inspect_eval()` and owns model connection, request concurrency/timeouts, log directories, and suite sequencing.

The benchmark owns sample conversion, prompts, setup/solvers, scorers, and any sandbox configuration. Use the supplied generation configuration. Keep reference answers/test data out of model-visible prompts unless the approved protocol intentionally includes them. Reasoning handling and answer extraction must match the benchmark's scoring protocol; BigCodeBench scores final code while retaining reasoning in logs.

### `summarize(logs, provenance, run_error) -> dict`

Return a JSON-serializable dictionary containing `status`, exactly `"completed"` or `"incomplete"`. Metrics, sample representation, denominators, and diagnostics are benchmark-owned.

`logs` contains Inspect `EvalLog` objects and may be empty or partial. `run_error` is either null or a dictionary with exception type, message, and traceback; it may include log-recovery errors. The runner calls the summarizer after successful preparation even if task construction or execution failed.

Account for every expected sample, including missing/duplicate outcomes and execution errors. Ordinary incorrect answers are scored outcomes; infrastructure failures, exhausted budgets, or missing results must not silently improve an aggregate by shrinking its denominator. Report incomplete coverage explicitly and withhold any metric that would misleadingly look like a complete-run score. Preserve diagnostics and never silently substitute tasks or retry answers.

Declare comparable numeric summary fields in `definition.SCORE_METRICS`, for example:

```python
SCORE_METRICS = {
    "accuracy": {"label": "Accuracy", "direction": "higher", "unit": "fraction"},
}
```

Return `accuracy` at the summary's top level and withhold it when incomplete coverage would make it misleading. Metadata is copied into `score_metrics`; reporting remains benchmark-agnostic. There is no cross-benchmark score average or generation-speed ranking.

The shared runner adds timestamps, `score_metrics`, `performance`, and `generation_samples`; reserve those fields for shared code. Common performance comes from native model-call records, not benchmark wall time; see [measurement definitions](orchestrator.md#generation-measurements). It saves the summary and continues after an incomplete benchmark, stopping on interruption. Shared startup/configuration failures can prevent the suite from starting. The root summary records statuses and child-summary locations.

**Runtime complete:** a real Inspect task can execute, and its summary distinguishes complete scoring from partial/error results without relying on BigCodeBench-specific assumptions.

## 4. Add dependencies and isolation when needed

Put benchmark-specific controller dependencies in `benchmarks/<name>/requirements.txt`, with pinned direct versions. The Dockerfile installs the shared requirements plus each benchmark directory's requirements in one environment. A requirements file is optional; shared dependencies need not be repeated.

For generated-code execution, use a benchmark-owned sandbox configuration and set `NEEDS_DOCKER = True`. Follow the existing isolation pattern: generated code runs in sibling non-root containers with no host mounts, Docker socket, GPU, or network unless explicitly approved, and with resource limits. Run only trusted benchmark implementation code in the controller; its Docker socket grants broad host privileges.

Each Compose sandbox service must propagate orchestration ownership so cancellation can clean up siblings after the controller exits:

```yaml
labels:
  localllmbench.run: "${LOCALLLMBENCH_RUN_ID:-standalone}"
  localllmbench.trial: "${LOCALLLMBENCH_TRIAL_ID:-standalone}"
```

The host passes these environment variables to orchestrated controllers. Standalone sandboxes use the fallback labels; orchestrator cleanup matches only its unique run/trial values.

For a benchmark without Docker sandbox needs, use `NEEDS_DOCKER = False` and omit sandbox configuration. The socket is mounted if any selected benchmark needs it.

Modules are not security boundaries: all selected implementations share one controller process and installed dependency environment. If a benchmark requires incompatible dependencies, multiple independent Inspect tasks, or a different execution lifecycle, discuss that contract change before extending the shared runner.

**Environment complete:** dependencies resolve together and any untrusted execution has an explicit, verified sandbox boundary.

## 5. Register and configure it

Add one entry to `BENCHMARKS` in [registry.py](../benchmarks/registry.py): the key is `<name>` and the value is `"benchmarks.<name>"`. Keep existing entries. Folder creation alone does not register a benchmark; there is no directory auto-discovery.

User overrides belong under `benchmark_options.<name>` in run configuration. Keep shared model/generation/execution settings in their existing sections rather than adding benchmark branches to shared code.

Selection behavior:

- Omitted or null `benchmarks`: every registered name, in registry insertion order.
- Explicit nonempty `benchmarks` list: only those names, in the supplied order.
- CLI `--benchmarks NAME ...`: overrides the configured selection after configuration validation.
- Unknown names, duplicates, and empty selections fail.

**Registering a benchmark therefore adds it to default-all runs**, including the checked-in configuration. Make its required inputs clear. Input availability for all selected benchmarks is checked before container launch. Users can select a subset when not all datasets are available.

Rebuild the controller image after adding/changing module code or dependencies. The host reads the working tree, but the container uses its built-in copy; a stale image can have an older registry or implementation.

**Registration complete:** the host and rebuilt image contain the same registered implementation and configuration options.

## 6. Verify the integration

From the project root, use these existing commands:

```powershell
python -m unittest discover -s tests -v
python benchmark.py --list-benchmarks
docker build --tag localllmbench-benchmark:local benchmarks
```

Create a temporary smoke configuration selecting only the new benchmark, with its real input path and a small supported sample count. Then run preparation and actual inference against the existing, ready inference server:

```powershell
python benchmark.py --config smoke_benchmark_config.json --select-only
python benchmark.py --config smoke_benchmark_config.json
```

Completion checklist:

- The new name is listed; explicit selection excludes other benchmarks, while omitted selection includes all registered names. Preview default-all only when the required inputs are available; a full default-all evaluation is not necessary for a small integration smoke.
- Selection/provenance are stable across repeated preparation and model/generation changes where sampling is supported. Input changes are distinguishable in provenance.
- The same provenance compares equal after host paths are remapped into the container; deliberate input/protocol changes fail before generation.
- Declared score directions/units describe the returned top-level metrics. Incomplete results retain diagnostics without becoming score leaders; missing timing is unavailable rather than zero.
- A real model response reaches the intended scorer. Inspect logs and per-sample outcomes support the reported metrics; tests alone are insufficient evidence.
- Exercise an appropriate failure/truncation case and verify incomplete coverage is visible rather than reported as a complete score. Add deterministic regression tests for meaningful selection/scoring/error boundaries.
- Confirm `results/<run>/<name>/manifest.json`, `inspect/`, and `summary.json`, plus the root suite summary, describe the actual run. Preparation failure may produce only a diagnostic summary.
- For code execution, inspect the actual sandbox's permissions, mounts, network, and limits; a YAML declaration alone is not verification.
- Keep smoke outputs as evidence, remove temporary configurations/scripts, and update [progress.md](progress.md) with the protocol, exercised checks, and remaining limits. Update the container documentation if shared behavior changed.

Adding a benchmark does not authorize implementing a sweep manager, UI, new inference backend, or unrelated benchmark. Stop after the requested integration and its verification.
