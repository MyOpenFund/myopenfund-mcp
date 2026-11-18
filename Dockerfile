# Host-agnostic image: everything deployment-specific comes from MOF_* environment variables.
# Pinned base image: tag AND digest, so a rebuild never picks up another Python or Debian silently.
# Changing it is a deliberate bump (README, "Pinned versions"): same value in compose.example.yaml.
ARG PYTHON_IMAGE=python:3.13.18-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
# Build stage: git is needed only to fetch the pinned git dependencies of the extras, never at runtime.
FROM ${PYTHON_IMAGE} AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md LICENSE constraints.txt build-constraints.txt ./
COPY src ./src
# Toolset extras installed in the image, comma-separated: vault (default), companies, rag.
ARG MOF_EXTRAS=vault
# -c constraints.txt: every PyPI runtime dependency at the exact version the CI tested; --build-constraint:
# the build backends of this package and of the git dependencies, built from source (README, "Pinned versions").
RUN pip install --no-cache-dir -c constraints.txt --build-constraint build-constraints.txt ".[${MOF_EXTRAS}]"

FROM ${PYTHON_IMAGE}
RUN groupadd --gid 10001 mcp && useradd --uid 10001 --gid 10001 --create-home mcp
COPY --from=build /opt/venv /opt/venv
# Mount point of the rag toolset's model cache (MOF_RAG_MODEL_CACHE=/models). It exists and belongs to
# the service user so that a new named volume mounted here is writable for the one-off fill.
RUN mkdir /models && chown 10001:10001 /models
# MOF_MCP_HOST=0.0.0.0 is inside the container only: what is published on the host is the
# deployment's choice (compose.example.yaml binds 127.0.0.1 by default).
# ORT_DISABLE_TELEMETRY=1: the ONNX runtime of the rag extra sends usage events to its publisher
# unless told not to. The server sets it for itself too; here it also covers `docker run ... python`.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ORT_DISABLE_TELEMETRY=1 \
    MOF_MCP_TRANSPORT=http \
    MOF_MCP_HOST=0.0.0.0 \
    MOF_MCP_PORT=8000
USER 10001:10001
WORKDIR /home/mcp
EXPOSE 8000
# Liveness via the unauthenticated GET /health (python stdlib: no curl in the image). Proxies are
# bypassed on purpose: an inherited HTTP(S)_PROXY must never capture a loopback probe.
# start-period 60 s: with the rag toolset the port opens once the embedding model is loaded, which
# takes 5 to 7 s from a warm disk and has taken more than 30 s from a cold one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request as u; port = os.environ.get('MOF_MCP_PORT', '').strip() or '8000'; u.build_opener(u.ProxyHandler({})).open(f'http://127.0.0.1:{port}/health', timeout=4)"]
CMD ["myopenfund-mcp", "serve"]
