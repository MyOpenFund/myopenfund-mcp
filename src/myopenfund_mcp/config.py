"""Server settings, from environment variables only.

Every variable and its default is documented in .env.example. A bad value stops
the server at start with the variable named — never a silent fallback.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


class ConfigError(ValueError):
    pass


def _get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return None if value is None or value.strip() == "" else value.strip()


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    transport: str = "stdio"          # MOF_MCP_TRANSPORT: stdio | http
    host: str = "127.0.0.1"           # MOF_MCP_HOST (http): loopback unless the deployment says otherwise
    port: int = 8000                  # MOF_MCP_PORT
    path: str = "/mcp"                # MOF_MCP_PATH
    tokens: str | None = None         # MOF_MCP_TOKENS: client:sha256[:tier],... (see auth.py)
    vault_database_url: str | None = None  # MOF_VAULT_DATABASE_URL: as vault_readonly; unset = no vault toolset
    query_timeout_s: int = 20         # MOF_MCP_QUERY_TIMEOUT_S: Postgres statement_timeout per call
    default_rows: int = 50            # MOF_MCP_DEFAULT_ROWS
    max_rows: int = 200               # MOF_MCP_MAX_ROWS
    max_result_chars: int = 200_000   # MOF_MCP_MAX_RESULT_CHARS: larger results are refused, not cut
    max_error_samples: int = 5        # MOF_MCP_MAX_ERROR_SAMPLES: error samples kept per runs_status row
    max_text_chars: int = 2000        # MOF_MCP_MAX_TEXT_CHARS: each error text cut to this length

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        transport = _get(env, "MOF_MCP_TRANSPORT") or "stdio"
        if transport not in ("stdio", "http"):
            raise ConfigError(f"MOF_MCP_TRANSPORT must be 'stdio' or 'http', got {transport!r}")
        settings = cls(
            transport=transport,
            host=_get(env, "MOF_MCP_HOST") or "127.0.0.1",
            port=_int(env, "MOF_MCP_PORT", 8000),
            path=_get(env, "MOF_MCP_PATH") or "/mcp",
            tokens=_get(env, "MOF_MCP_TOKENS"),
            vault_database_url=_get(env, "MOF_VAULT_DATABASE_URL"),
            query_timeout_s=_int(env, "MOF_MCP_QUERY_TIMEOUT_S", 20),
            default_rows=_int(env, "MOF_MCP_DEFAULT_ROWS", 50),
            max_rows=_int(env, "MOF_MCP_MAX_ROWS", 200),
            max_result_chars=_int(env, "MOF_MCP_MAX_RESULT_CHARS", 200_000, minimum=1_000),
            max_error_samples=_int(env, "MOF_MCP_MAX_ERROR_SAMPLES", 5),
            max_text_chars=_int(env, "MOF_MCP_MAX_TEXT_CHARS", 2000),
        )
        if settings.default_rows > settings.max_rows:
            raise ConfigError("MOF_MCP_DEFAULT_ROWS must not exceed MOF_MCP_MAX_ROWS")
        if settings.transport == "http" and not settings.tokens:
            raise ConfigError(
                "MOF_MCP_TRANSPORT=http requires MOF_MCP_TOKENS: the HTTP server never runs unauthenticated"
            )
        return settings
