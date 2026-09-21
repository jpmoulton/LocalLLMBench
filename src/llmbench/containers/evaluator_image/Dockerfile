# llmbench evaluator image. The staging step (containers/image_plan.build_evaluator_context) writes the exact
# official python:3.12-slim-bookworm digest into the ARG default; the source file carries no default on purpose.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="llmbench evaluator"
LABEL org.opencontainers.image.description="Inspect-driven evaluator for one attached llama.cpp candidate"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/cache

# Every dependency wheel named in the hash-pinned Linux lock (derived from requirements.lock.txt) is downloaded from PyPI
# at build time and verified against its recorded sha256 (--require-hashes, no resolution, no dependencies); the
# harness itself comes from a wheel built on the host from the working tree and is installed offline (--no-index).
COPY requirements.linux.lock /opt/llmbench/requirements.linux.lock
COPY *.whl /opt/llmbench/wheels/
RUN pip install --require-hashes --no-deps -r /opt/llmbench/requirements.linux.lock \
    && pip install --no-index --no-deps /opt/llmbench/wheels/*.whl \
    && pip freeze > /opt/llmbench/pip-freeze.txt \
    && python --version > /opt/llmbench/python-version.txt \
    && python -c "import llmbench.container_eval, inspect_ai" \
    && mkdir -p /artifacts /spool/requests /spool/results /run/llmbench \
    && rm -rf /root/.cache/pip /tmp/*

# Pinned benchmark data. scripts/stage_benchmark_datasets.py stages the datasets on the host (the only place in
# the suite allowed to reach the network for them) and root copies that staging directory into the build context
# as `benchmarks/` before building. This COPY is unconditional and explicit: it bakes exactly what the staging
# directory holds. A dataset that was not staged is simply not in the image and container_eval.self_check()
# reports that benchmark unavailable - no data is ever retrieved while the container runs, which has no network.
# staging-manifest.json travels with the data, so /opt/llmbench/benchmarks/staging-manifest.json inside the image
# records the source, upstream commit or release tag, retrieval time, license and sha256 of every byte scored.
COPY benchmarks /opt/llmbench/benchmarks

WORKDIR /opt/llmbench
USER 10001:10001
# Compose supplies the argv (--config/--policy/--grant-seconds/--artifact-bytes); there is deliberately no CMD.
ENTRYPOINT ["python", "-m", "llmbench.container_eval"]
