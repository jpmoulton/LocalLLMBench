# Inference container

## Scope and design

One llama.cpp inference server, with configurable startup settings. The separate benchmark controller is documented in [benchmark_container.md](benchmark_container.md); the host sweep CLI is documented in [orchestrator.md](orchestrator.md). Multiple inference backends remain future work.

- `Dockerfile`: a thin image based on the official CUDA server image, with our agreed defaults and a readiness health check. No model weights or application framework are copied into it.
- `inference.py`: a standard-library Python launcher. It mounts the model directory read-only, publishes a host port, and launches one detached container. Native llama-server arguments after `--` pass through unchanged.
- `.dockerignore`: excludes everything except the Dockerfile from the build context.
- Defaults live in the image, not in a second Python configuration schema. Native command-line options override image environment defaults. Request-level generation options can override server generation defaults.

Standalone `inference.py` leaves lifecycle management to Docker commands. A successful launcher exit means Docker started the container, not that the model is ready. The orchestrator reuses its launch construction and additionally owns readiness, probing, sequential execution, and cleanup.

## Pinned runtime

- Upstream image: `ghcr.io/ggml-org/llama.cpp:server-cuda`, resolved and pinned by digest in the Dockerfile.
- Digest: `sha256:d5ffba7d096ab0236c65dc147d797b4b7658a7e5425634593373635d661bccda`.
- llama.cpp: `b12063`, commit `2cbca389f515df455412dafa07e2b1359cc72a6d`.
- CUDA runtime: 12.8.1; tested image platform: Linux AMD64.
- Local image name: `localllmbench-inference:local`.

Pinning avoids silently changing engine behavior when rebuilding. Updating llama.cpp is a deliberate Dockerfile change followed by a new load/inference check.

References: [pinned server documentation](https://github.com/ggml-org/llama.cpp/blob/2cbca389f515df455412dafa07e2b1359cc72a6d/tools/server/README.md), [upstream Docker documentation](https://github.com/ggml-org/llama.cpp/blob/2cbca389f515df455412dafa07e2b1359cc72a6d/docs/docker.md).

## Requirements

- Docker with Linux containers and NVIDIA GPU passthrough. Verified on Docker Desktop 4.76.0 / Engine 29.5.2 on Windows.
- NVIDIA driver compatible with the CUDA image. Verified with RTX 5090 and driver 617.42.
- Host Python, verified with Python 3.12.10; no pip dependencies.
- A supported local GGUF file. Linux hosts will also need NVIDIA Container Toolkit; Linux host operation has not been tested here.

## Build and launch

Run from this project directory:

```powershell
docker build --tag localllmbench-inference:local .
python inference.py --model "F:\Models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf"
```

The default container name is `localllmbench-inference`. The launcher does not replace an existing container. If it already exists, start it with `docker start localllmbench-inference`, or stop/remove it before launching a different configuration.

Inspect readiness and logs:

```powershell
docker logs -f localllmbench-inference
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

Wait for `status: ok` before sending inference requests. The Docker health check calls the same endpoint every 10 seconds with a five-minute startup grace period. It reports readiness; it does not restart unhealthy containers.

Endpoints:

- OpenAI-compatible API base: `http://127.0.0.1:8080/v1`
- Model identifier: `local-model` (override with native `--alias` if needed)
- Model information: `/v1/models`
- Effective server/generation properties: `/props`
- Prometheus metrics: `/metrics`
- Upstream llama.cpp web UI: `/` (not a project-built UI)

A small PowerShell request:

```powershell
$body = @{
    model = "local-model"
    messages = @(@{ role = "user"; content = "What is 2 + 2? Reply with only the number." })
    temperature = 0
    max_tokens = 32
    chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType "application/json" -Body $body
```

The thinking override above was verified with this model's embedded template. It is a per-request smoke-test choice, not a universal setting for other models or a server default.

## Changing context, caches, and other options

Stop/remove the current container before replacing its configuration:

```powershell
docker stop localllmbench-inference
docker rm localllmbench-inference
python inference.py --model "F:\Models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" --port 8080 -- --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q8_0
```

`docker start` reuses the old configuration; it does not apply new arguments or a rebuilt image. Recreate the container to change either.

Launcher inputs before `--`:

| Input | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | Host path to the GGUF file; its parent directory is mounted at `/models`, read-only. Adjacent shards remain accessible. |
| `--port` | `8080` | Published host port. Internal server port remains `8080`. |
| `--bind-address` | `127.0.0.1` | Host IP on which Docker publishes the port. |
| `--name` | `localllmbench-inference` | Docker container name. |

The launcher owns model selection and internal `--host`/`--port`; its values are appended after passthrough options. Use the launcher inputs to change model or host networking, not conflicting native arguments.

Inside the container, the server listens on `0.0.0.0:8080`. Host publication defaults to localhost only. `--bind-address 0.0.0.0` exposes the published port on all host IPv4 interfaces: there is no API authentication configured by default, and upstream CORS defaults are permissive. Keep localhost binding for the initial workflow. A shared Docker network for the benchmark container will be designed later.

## Inference defaults

| Setting | Default | Native override after `--` |
| --- | --- | --- |
| Context size | `8192` | `--ctx-size N` |
| GPU layers | `all` | `--gpu-layers N` |
| Flash Attention | `on` | `--flash-attn on/off/auto` |
| Key-cache precision | `f16` | `--cache-type-k TYPE` |
| Value-cache precision | `f16` | `--cache-type-v TYPE` |
| KV-cache GPU offload | Enabled | `--no-kv-offload` / `--kv-offload` |
| Logical batch size | `2048` | `--batch-size N` |
| Physical microbatch size | `512` | `--ubatch-size N` |
| Parallel slots | `1` | `--parallel N` |
| Continuous batching | Enabled | `--no-cont-batching` / `--cont-batching` |
| CPU generation threads | Runtime-selected (`-1`) | `--threads N` |
| CPU prompt threads | Same as generation threads | `--threads-batch N` |
| Automatic memory fitting | Disabled | `--fit on/off` |
| Prompt caching | Disabled | `--cache-prompt` / `--no-cache-prompt` |
| Context shifting | Disabled | `--context-shift` / `--no-context-shift` |
| Startup warmup | Enabled | `--no-warmup` / `--warmup` |
| Speculative decoding | `none` | `--spec-type TYPE` (other required options depend on method/model) |
| Metrics endpoint | Enabled | Native environment `LLAMA_ARG_ENDPOINT_METRICS=0` when using Docker directly |

Use one value for slash-separated choices, for example `--flash-attn off`, not the literal string `on/off/auto`.

All native llama-server options are available through passthrough; the table is the supported starting surface, not a new whitelist. The pinned runtime validates arguments. Unsupported combinations fail rather than being translated into another configuration. Model-specific behavior must still be checked in logs; selecting a flag is not proof of the effective execution path.

For Docker-native invocation, the image also accepts llama.cpp's `LLAMA_ARG_*` environment variables. For example, `docker run -e LLAMA_ARG_CTX_SIZE=16384 ...` overrides the image environment; explicit native CLI values take precedence. The Python launcher deliberately does not duplicate every environment variable as a Python option.

Read the exact build's option list:

```powershell
docker run --rm localllmbench-inference:local --help
```

The embedded chat template and sampling metadata are retained. On the tested file, `/props` reported temperature `1.0`, top-k `20`, and top-p approximately `0.95`; do not assume the generic CLI help defaults describe effective model sampling. Send explicit request-level sampling and output limits for controlled evaluations. Prompt caching is a server default, not a prohibition on request-level overrides.

Context allocation interacts with parallel slots and KV layout. Do not interpret a context change as a per-request guarantee when also increasing concurrency. KV quantization is separate from GGUF weight quantization and can affect quality; supported combinations depend on model, build, and attention backend.

## Verification — 2026-11-03

Executed with the real `Qwen3.8-27B-Q4_K_M.gguf`, not a mock:

- Built the pinned image and launched it through `inference.py` using default inference settings.
- Docker health became healthy; host `/health` returned `{"status":"ok"}`.
- `/v1/models` reported context `8192`, alias `local-model`, and Q4_K - Medium weights.
- Non-streaming chat returned `4` for the arithmetic prompt above.
- Streaming chat returned `4`, a final usage event, and `[DONE]`.
- Repeating the prompt reported zero cached prompt tokens.
- `/metrics` returned inference counters.
- Docker inspection confirmed localhost-only publication and a read-only model-directory mount.
- Stopped the baseline, then launched a separate container on host port `8081` with context `16384`, K/V `q8_0`, `--no-warmup`, and verbose startup logging. `/v1/models` confirmed the new context and chat again returned `4`.
- Override startup logs confirmed Flash Attention enabled, `66/66` layers offloaded to the RTX 5090, and K/V buffers actually using `q8_0`. KV buffer size was `544 MiB`; a separate recurrent-state buffer was `149.62 MiB`.
- An invalid cache type exited with code `1` and an explicit unsupported-cache-type error.
- Removed the temporary override container and restarted the original default container.

These are load/API/configuration smoke checks, not quality or speed benchmark results. Other GPU applications were running, and the short responses are unsuitable for throughput comparisons. No Linux-host, LAN-access, multimodal, Q6_K, NVFP4, or MTP run has been validated.

### Model-specific observations

The file's GGUF metadata declares architecture `qwen35`, name `Qwen_Qwen3.8 27B`, and training context `262144`. These are observed metadata, not a claim about model provenance.

The loader reports unused tensors in `blk.64`, including `nextn` tensors. The baseline completes inference but does not exercise these extra tensors or establish MTP support. Keep this warning visible for later model-specific investigation.

This hybrid model allocates both attention KV buffers and recurrent state. Changing K/V cache precision does not quantize all model state or weights. Account for those separate allocations when later measuring VRAM/context tradeoffs.

## High-context benchmark deployment

The initial 8192-context container is now stopped, not deleted. The active benchmark server is `localllmbench-inference-128k`, published at the same `http://127.0.0.1:8080` address:

```powershell
python inference.py --model "F:\Models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf" --name localllmbench-inference-128k -- --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 --log-verbosity 4
```

This is the creation command; the container already exists. Use `docker start localllmbench-inference-128k` if stopped, or stop/remove it before changing configuration. The two containers cannot publish port 8080 simultaneously.

Observed startup: 66/66 layers offloaded, CUDA model buffer 15088.32 MiB, attention KV buffer 4352 MiB, recurrent-state buffer 149.62 MiB, and CUDA compute buffer 740.02 MiB. A host GPU snapshot reported 27678 MiB used out of 32607 MiB, including other applications. The server completed reasoning-enabled requests and two real BigCodeBench problems.

These measurements establish this configured deployment and short-prompt execution, not a guarantee about full-context memory pressure, absence of Windows memory paging, or performance with an actual 128K input.

The benchmark launcher connects this container to the `localllmbench` Docker network so the controller can use its container name. Host publication remains localhost-only. The image defaults above remain unchanged; context and cache precision are explicit launch overrides.

## Lifecycle

```powershell
docker stop localllmbench-inference
docker start localllmbench-inference
docker logs -f localllmbench-inference
```

To remove the container, stop it and run `docker rm localllmbench-inference`. The model files remain untouched. Failed containers are retained for log inspection; the launcher does not automatically remove, replace, retry, or restart them.

For the currently active benchmark server, substitute `localllmbench-inference-128k` in those lifecycle commands. Keep the original container stopped while the high-context one owns port 8080.
