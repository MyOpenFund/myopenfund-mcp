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
CMD ["myopenfund-mcp", "serve"]
