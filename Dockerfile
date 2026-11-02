# Host-agnostic image: everything deployment-specific comes from MOF_* environment variables.
# Build stage: git is needed only to fetch the pinned vault-query dependency, never at runtime.
FROM python:3.13-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[vault]"

FROM python:3.13-slim
RUN groupadd --gid 10001 mcp && useradd --uid 10001 --gid 10001 --create-home mcp
COPY --from=build /opt/venv /opt/venv
# MOF_MCP_HOST=0.0.0.0 is inside the container only: what is published on the host is the
# deployment's choice (compose.example.yaml binds 127.0.0.1 by default).
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MOF_MCP_TRANSPORT=http \
    MOF_MCP_HOST=0.0.0.0 \
    MOF_MCP_PORT=8000
USER 10001:10001
WORKDIR /home/mcp
EXPOSE 8000
# Liveness via the unauthenticated GET /health (python stdlib: no curl in the image). Proxies are
# bypassed on purpose: an inherited HTTP(S)_PROXY must never capture a loopback probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request as u; port = os.environ.get('MOF_MCP_PORT', '').strip() or '8000'; u.build_opener(u.ProxyHandler({})).open(f'http://127.0.0.1:{port}/health', timeout=4)"]
CMD ["myopenfund-mcp", "serve"]
