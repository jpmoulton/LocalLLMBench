# YAML recipes

**Status: recipe format and defaults specified; orchestrator CLI not implemented yet.** The current `inference.py` and `benchmark.py` launchers do not consume recipes. This iteration changes the schema, example, and documentation only.

The existing standalone `benchmark_config.json` remains unchanged; this YAML format is the input for the future orchestrator, not a replacement parser for the current benchmark launcher.

- [JSON Schema](../recipes/recipe.schema.json): version 1, JSON Schema Draft 2020-12, validating the data parsed from YAML.
- [Complete YAML example](../recipes/qwen-kv-sweep.yaml): Q4_K_M and Q6_K models, with a three-choice 128K group and a single-configuration 256K group.
- [Terminology](GLOSSARY.md): run, trial, and benchmark evaluation.

Recipes use YAML for readability, comments, and named groups. The example's `# yaml-language-server: $schema=./recipe.schema.json` comment provides editor schema association; `schema_version` identifies the recipe format. The schema remains JSON because it describes the parsed data, not the source syntax.

Use safe, single-document YAML parsing. The future loader must reject duplicate mapping keys rather than silently overwrite settings. Quote native string values such as `"on"` and `"off"`: some YAML parsers otherwise turn them into booleans. Use literal `true`/`false` for actual booleans, and `null` for default-all benchmark selection. No executable YAML tags, includes, or environment interpolation are part of this format.

JSON Schema `default` annotations do not insert values. Shared recipe defaults must be resolved by the future loader; benchmark defaults come from each module; unspecified native server settings continue to use image defaults. These are specified defaults, not an implemented recipe-loading pipeline.

## Sections and ownership

| Field | Meaning |
| --- | --- |
| `schema_version` | Required format version; currently `1`. |
| `name` | Optional human-readable run label. Reusing it must not overwrite prior results. |
| `models` | Required ordered list of local GGUF paths. Different weight quants are different entries, not a weight-quantization setting. |
| `inference` | Fixed connection settings and a list of serving-configuration blocks to expand. |
| `benchmarks` | Optional ordered list of benchmark names. Omitted/null means every registered benchmark, resolved once for the run. An empty list does not mean all. |
| `benchmark_options` | Fixed options keyed by benchmark name, using each module's existing defaults and validator. Options do not select a benchmark. |
| `generation` | Existing thinking, output budget, sampling, and seed settings, fixed across trials. |
| `execution` | Fixed benchmark request concurrency/timeouts and inference startup timeout. Trials themselves remain sequential. |
| `output_dir` | Parent directory for unique run outputs. The example uses `../results` because it lives in `recipes/`. |

All relative model, declared benchmark-input, and output paths resolve against the recipe's directory, not the shell's current directory. Absolute Windows paths can use forward slashes, as in the example. Environment-variable expansion, includes, and templating are not proposed.

There are no per-model benchmark or generation overrides. A run resolves those settings and the benchmark selection once and uses them unchanged for every trial. Benchmark-specific arrays remain ordinary option data; they are never interpreted as sweep axes.

## Serving configurations and expansion

`inference.configurations` is a nonempty list of independent configuration blocks. Each block contains `parameters` and optionally a `name` and `kv_quantization`.

1. Within one block, a scalar parameter is fixed. An array means try each listed value.
2. Expand those arrays into their Cartesian product, including the block's KV-cache choices.
3. Concatenate the configurations produced by separate blocks; do not multiply blocks together or inherit settings between them.
4. Evaluate every model against every resulting configuration.

This retains the agreed explicit-configuration approach: a block containing only scalar settings produces one configuration. Lists enable parameter-grid expansion. Separate groups describe particular combinations without generating unwanted cross-pairs.

The supplied example produces three 128K configurations (q8_0, q4_0, bf16) plus one 256K configuration (q4_0 only). Two models therefore produce **eight trials**. With the currently registered BigCodeBench and its default 64 problems, that is 512 attempted model answers if every trial reaches generation. Canonical preflight does not request model answers. The example uses default-all benchmark selection, so additional registered benchmarks will also be included.

Deterministic order: models in list order, then configuration groups in list order. Within a group, sort parameter names lexicographically, treating `kv_quantization` as another axis name; preserve each value list's order, with the last axis varying fastest. Reordering YAML mapping keys must not change trial order. Keep requested combinations even if different groups describe equivalent settings; there is no automatic deduplication or repeats mechanism.

## Native llama.cpp settings

`parameters` uses actual long option names, including `--`, rather than maintaining a renamed copy of llama.cpp's configuration surface. Examples are `--ctx-size`, `--gpu-layers`, `--flash-attn`, `--batch-size`, `--ubatch-size`, `--parallel`, and `--threads`.

Value rules:

- String/number: one argument value. `--flash-attn: "off"` becomes `--flash-attn off`.
- List: alternative values. `--ctx-size: [131072, 262144]` is a sweep axis, not a multi-argument flag.
- `true`: emit a valueless flag. `--no-warmup: true` emits `--no-warmup`.
- `false`: omit that flag. This does **not** emit its inverse or override an image default. To disable warmup, use `--no-warmup: true`, not `--warmup: false`.
- For one option requiring several tokens, use an atomic mapping with an `args` list. For example:

```yaml
--control-vector-layer-range:
  args: ["1", "3"]
```

The inner list is literal arguments, not a sweep axis. This illustrates encoding only; an actual control-vector experiment also needs its assets configured.

Pass the eventual arguments as tokens, not through a shell. String values containing commas or spaces remain one argument. Host-path translation applies only to model paths and benchmark-declared input paths; arbitrary native file arguments are container paths. The existing inference launcher mounts the selected model's parent directory at `/models`; this proposal does not add arbitrary extra asset mounts.

The orchestrator owns model selection, container identity, transport, and the API alias `local-model`. Sampling/reasoning controls belong in `generation`. The schema rejects common conflicting native options, but is not a complete llama.cpp grammar or semantic validator: additional sampling or prompt-protocol overrides must not be hidden in serving-setting axes. Unknown native options are ultimately the pinned server's responsibility. We are not adding comprehensive compatibility validation here.

Unspecified serving settings retain the existing inference image/server defaults. Explicit native values override image defaults. Record the requested configuration, launch arguments/image identity, and reported server settings with results rather than treating the recipe alone as proof of the effective configuration. Read [the inference defaults](inference_container.md) for the current profile.

## KV-cache precision

The one convenience field is `kv_quantization`, because K and V are commonly changed together:

```yaml
kv_quantization: [q8_0, q4_0, bf16]
```

This is three paired choices, not nine independent K/V combinations. Each string produces both `--cache-type-k VALUE` and `--cache-type-v VALUE`.

For asymmetric pairs, use atomic mappings:

```yaml
kv_quantization:
  - k: q8_0
    v: q8_0
  - k: q8_0
    v: q4_0
```

Alternatively, omit `kv_quantization` and supply native `--cache-type-k` / `--cache-type-v` options in `parameters`. Arrays on both native options intentionally generate every K/V pair. Mixing the convenience field and either native cache option within one block is rejected rather than resolved through hidden precedence.

The pinned b12063 server advertises `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, and `q5_1`. Use those exact names: `q8`, `q4`, and uppercase `BF16` are not recipe aliases. Although named `kv_quantization` for familiarity, the field also accepts unquantized floating-point cache types.

Advertised types do not establish kernel/model compatibility or GPU fit. The example's expanded combinations have not been load-tested; failures should be recorded and execution should continue under the agreed trial policy.

## Context, concurrency, and defaults

Every configuration group explicitly specifies `--ctx-size`: this is the requested context allocation passed to llama.cpp. For example, `131072` requests 128K tokens and `262144` requests 256K. There is no separate minimum-context setting in the recipe.

`--ctx-size` retains native server semantics. With multiple parallel slots, do not assume every request gets the full allocation. Record the server-reported context with results rather than adding another user-configured threshold.

Integration note: the current standalone benchmark still has its earlier `inference.minimum_context` guard. Remove that legacy setting/check when implementing the recipe-consuming orchestrator so it cannot silently reject intentionally smaller context configurations. This schema change does not modify the working benchmark runtime.

`--parallel` controls server slots. `execution.concurrency` controls benchmark sample/request concurrency. Neither is trial concurrency, and increasing one does not automatically change the other. Keep request concurrency fixed across a run; varying slot count is still a valid serving experiment but must be interpreted alongside actual request load.

### Default values and precedence

| Setting | Default when omitted |
| --- | --- |
| `inference.bind_address`, `inference.port` | `127.0.0.1`, `8080` |
| Native `--gpu-layers`, `--flash-attn`, `--parallel` | `all`, `"on"`, `1` |
| Native `--batch-size`, `--ubatch-size`, `--threads` | `2048`, `512`, `-1` (runtime-selected CPU threads) |
| Native K/V cache types | `f16` / `f16` when neither shorthand nor native cache option overrides them |
| Native `--fit`, `--spec-type` | `"off"`, `none` |
| Other existing image behavior | KV offload on, continuous batching on, warmup on, prompt caching off, context shifting off, metrics on |
| `benchmarks` | All registered benchmarks, resolved once for the run |
| `benchmark_options` | Each benchmark's module defaults; BigCodeBench currently uses count `64`, subset seed `42`, test timeout `60` seconds, and its configured local dataset path |
| `generation.thinking`, `generation.max_tokens` | `true`, `8192` (reasoning plus final output) |
| `generation.temperature`, `top_p`, `top_k`, `min_p`, `seed` | `1.0`, `0.95`, `20`, `0.05`, `42` |
| `execution.concurrency`, `execution.request_timeout` | `1`, `900` seconds |
| `execution.startup_timeout` | `300` seconds; bounds readiness waiting, not a retry policy |
| `output_dir` | `results`, relative to the recipe directory |

Models, format version, configuration groups, and each group's context size are required rather than silently invented. Group/run names are optional labels.

Explicit shared recipe values override their defaults. Benchmark overrides merge into the selected modules' existing defaults. Groups do not inherit from earlier groups.

Native schema defaults document the existing image behavior: **do not blindly fill `parameters` with them and emit extra CLI options**. Emit only explicitly requested native options plus the K/V arguments derived from an explicit `kv_quantization`. This avoids overriding an alternate native flag or colliding with the KV shorthand. Omitted native options remain the image/server's responsibility; neither the image defaults nor standalone benchmark defaults were changed.

In particular, `kv_quantization` itself has no injected default. With it omitted, an explicit native K or V setting wins, and an unspecified side retains `f16`. With it supplied, both sides come from that choice and native K/V overrides in the same group are rejected. The example explicitly chooses q8_0/q4_0/bf16; none of those choices replaces the image's default cache precision.

Bind address defaults to `127.0.0.1`, port to 8080. The published host address is not the controller-to-server address: containers communicate over the existing Docker network. An existing service using that host port is not owned by the new run; it must not be stopped silently.

## Validation and implementation boundary

The JSON Schema checks document structure, scalar/list distinctions, nonempty axes, obvious duplicates within an axis, known cache names, and a few ownership conflicts. It deliberately leaves registered benchmark names and options to the existing registry/modules so a new benchmark does not require editing this schema.

It does not check GPU memory, model architecture support, native option combinations, whether files exist, benchmark population limits, or whether expanded trials can successfully run. Additional validation and fast-fail behavior remain deferred. No execution, retry, resource monitoring, or UI code is included.

Verification exercised safe YAML parsing, schema validation, eight-trial group expansion, multi-axis expansion with atomic K/V/argument mappings, and defaults against the existing Dockerfile and benchmark configuration. Ten malformed/conflicting variants were rejected, including YAML-coerced boolean values for native on/off options. No model trials were executed.

A first real orchestrator smoke should use one model, one configuration, and two problems—not the full example run. The next implementation step is the recipe-consuming CLI; it is not part of this schema/documentation change.
