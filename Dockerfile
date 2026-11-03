# llama.cpp b12063 (2cbca389f), official CUDA 12.8.1 server image.
FROM ghcr.io/ggml-org/llama.cpp@sha256:d5ffba7d096ab0236c65dc147d797b4b7658a7e5425634593373635d661bccda

# Native llama.cpp environment defaults; explicit CLI arguments take precedence.
ENV LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_ALIAS=local-model \
    LLAMA_ARG_CTX_SIZE=8192 \
    LLAMA_ARG_N_GPU_LAYERS=all \
    LLAMA_ARG_FLASH_ATTN=on \
    LLAMA_ARG_CACHE_TYPE_K=f16 \
    LLAMA_ARG_CACHE_TYPE_V=f16 \
    LLAMA_ARG_KV_OFFLOAD=1 \
    LLAMA_ARG_BATCH=2048 \
    LLAMA_ARG_UBATCH=512 \
    LLAMA_ARG_N_PARALLEL=1 \
    LLAMA_ARG_CONT_BATCHING=1 \
    LLAMA_ARG_THREADS=-1 \
    LLAMA_ARG_FIT=off \
    LLAMA_ARG_CACHE_PROMPT=0 \
    LLAMA_ARG_CONTEXT_SHIFT=0 \
    LLAMA_ARG_SPEC_TYPE=none \
    LLAMA_ARG_ENDPOINT_METRICS=1

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=5m --retries=3 \
    CMD ["curl", "--fail", "--silent", "http://127.0.0.1:8080/health"]

# Warmup has no environment option. A later --no-warmup overrides this.
ENTRYPOINT ["/app/llama-server", "--warmup"]
