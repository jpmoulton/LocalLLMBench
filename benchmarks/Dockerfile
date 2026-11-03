FROM docker:29.5.2-cli@sha256:9ba8e32bfc35a2c7ae2feb1e3241b2778ae21dee80f4dcd31d04e1cfdea86ea2 AS docker_tools
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS dependencies

# Docker client only: Inspect creates sibling sandboxes through the host daemon.
COPY --from=docker_tools /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker_tools /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
WORKDIR /app
COPY . /app/benchmarks/
# Resolve shared and all benchmark-specific dependencies in one pip invocation.
RUN set -- -r /app/benchmarks/requirements.txt; \
    for requirements in /app/benchmarks/*/requirements.txt; do \
        if [ -f "$requirements" ]; then set -- "$@" -r "$requirements"; fi; \
    done; \
    pip install --no-cache-dir "$@"
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

FROM dependencies AS benchmark
ENTRYPOINT ["python", "-m", "benchmarks.run"]
