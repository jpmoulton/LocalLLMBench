# The planning helper requires an official Node 24 Bookworm slim digest.
# No default tag: callers must supply BASE_IMAGE=...@sha256:<64 hex>.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="llmbench private coding worker"
LABEL org.opencontainers.image.description="Python, JavaScript and TypeScript fixture toolchain"

# Debian packages resolve once during this trusted build. Their exact versions
# and the final image ID are retained; rebuilding is not claimed byte-identical.
# python3-numpy: EvalPlus's official assertion harness (`np.allclose`) is imported by every MBPP+ test
# module. Workers run with --network=none, so it has to be baked here; it cannot be installed at run time.
# python3-pytest: the pinned test command for Aider Polyglot's Python exercises.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-numpy python3-pytest \
    && rm -rf /var/lib/apt/lists/*
RUN npm install --global --ignore-scripts --no-audit --no-fund typescript@5.9.3 \
    && npm cache clean --force
# The jest/babel toolchain every JavaScript exercise in the pinned Aider Polyglot corpus declares. The
# exercise directory is mounted read-only, so the driver links this node_modules next to a /tmp copy.
COPY polyglot-js/package.json /opt/llmbench/polyglot-js/package.json
RUN cd /opt/llmbench/polyglot-js \
    && npm install --ignore-scripts --no-audit --no-fund --omit=dev \
    && npm cache clean --force
RUN mkdir -p /opt/llmbench /workspace \
    && python3 --version > /opt/llmbench/python-version.txt \
    && python3 -c "import numpy; print(numpy.__version__)" > /opt/llmbench/numpy-version.txt \
    && python3 -m pytest --version > /opt/llmbench/pytest-version.txt 2>&1 \
    && (cd /opt/llmbench/polyglot-js && npm ls --all --json > /opt/llmbench/polyglot-js-packages.json) \
    && test -x /opt/llmbench/polyglot-js/node_modules/.bin/jest \
    && node --version > /opt/llmbench/node-version.txt \
    && tsc --version > /opt/llmbench/typescript-version.txt \
    && dpkg-query -W > /opt/llmbench/debian-packages.txt \
    && npm ls --global --json > /opt/llmbench/npm-packages.json \
    && test "$(tsc --version)" = "Version 5.9.3"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NODE_ENV=production
WORKDIR /workspace
USER 10001:10001
# DockerWorker always overrides the entrypoint; no service is started by default.
ENTRYPOINT ["python3"]
CMD ["--version"]
