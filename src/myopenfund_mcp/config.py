"""Server settings, from environment variables only.

Every variable and its default is documented in .env.example. A bad value stops
the server at start with the variable named — never a silent fallback.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field


class ConfigError(ValueError):
    pass


def _get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return None if value is None or value.strip() == "" else value.strip()


def _int(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    # ASCII digits only: int() would also take "+5", "1_000" or non-ASCII digits.
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ConfigError(f"{name} must be an integer, got {raw!r}")
    value = int(raw)
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    transport: str = "stdio"  # MOF_MCP_TRANSPORT: stdio | http
    host: str = "127.0.0.1"  # MOF_MCP_HOST (http): loopback unless the deployment says otherwise
    port: int = 8000  # MOF_MCP_PORT
    path: str = "/mcp"  # MOF_MCP_PATH
    # MOF_MCP_TOKENS: client:sha256[:tier],... (see auth.py); kept out of repr
    tokens: str | None = field(default=None, repr=False)
    # MOF_VAULT_DATABASE_URL: as vault_readonly; unset = no vault toolset; kept out of repr
    vault_database_url: str | None = field(default=None, repr=False)
    query_timeout_s: int = 20  # MOF_MCP_QUERY_TIMEOUT_S: Postgres statement_timeout per call
    connect_timeout_s: int = 10  # MOF_MCP_CONNECT_TIMEOUT_S: libpq connect_timeout per call
    default_rows: int = 50  # MOF_MCP_DEFAULT_ROWS
    max_rows: int = 200  # MOF_MCP_MAX_ROWS
    max_result_chars: int = 200_000  # MOF_MCP_MAX_RESULT_CHARS: larger results are refused, not cut
    max_error_samples: int = 5  # MOF_MCP_MAX_ERROR_SAMPLES: error samples kept per runs_status row
    max_text_chars: int = 2000  # MOF_MCP_MAX_TEXT_CHARS: each error text cut to this length
    # MOF_COMPANIES_CATALOG: my-open-data catalogue as lake_reader; unset = no companies/datasets toolsets
    companies_catalog: str | None = field(default=None, repr=False)
    companies_data_root: str | None = None  # MOF_COMPANIES_DATA_ROOT: where this host sees the store files
    companies_query_timeout_s: int = 20  # MOF_COMPANIES_QUERY_TIMEOUT_S: time budget per store call
    companies_max_companies: int = 10  # MOF_COMPANIES_MAX_COMPANIES: companies per call
    companies_max_concurrency: int = 2  # MOF_COMPANIES_MAX_CONCURRENCY: store reads running at once

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        transport = _get(env, "MOF_MCP_TRANSPORT") or "stdio"
        if transport not in ("stdio", "http"):
            raise ConfigError(f"MOF_MCP_TRANSPORT must be 'stdio' or 'http', got {transport!r}")
        settings = cls(
            transport=transport,
            host=_get(env, "MOF_MCP_HOST") or "127.0.0.1",
            port=_int(env, "MOF_MCP_PORT", 8000, maximum=65535),
            path=_get(env, "MOF_MCP_PATH") or "/mcp",
            tokens=_get(env, "MOF_MCP_TOKENS"),
            vault_database_url=_get(env, "MOF_VAULT_DATABASE_URL"),
            # statement_timeout is an int32 in milliseconds.
            query_timeout_s=_int(env, "MOF_MCP_QUERY_TIMEOUT_S", 20, maximum=2_147_483),
            # libpq silently raises a connect_timeout below 2 to 2: refused here rather than misread.
            connect_timeout_s=_int(env, "MOF_MCP_CONNECT_TIMEOUT_S", 10, minimum=2, maximum=300),
            default_rows=_int(env, "MOF_MCP_DEFAULT_ROWS", 50),
            max_rows=_int(env, "MOF_MCP_MAX_ROWS", 200),
            max_result_chars=_int(env, "MOF_MCP_MAX_RESULT_CHARS", 200_000, minimum=1_000),
            max_error_samples=_int(env, "MOF_MCP_MAX_ERROR_SAMPLES", 5),
            max_text_chars=_int(env, "MOF_MCP_MAX_TEXT_CHARS", 2000),
            companies_catalog=_get(env, "MOF_COMPANIES_CATALOG"),
            companies_data_root=_get(env, "MOF_COMPANIES_DATA_ROOT"),
            companies_query_timeout_s=_int(env, "MOF_COMPANIES_QUERY_TIMEOUT_S", 20, maximum=3600),
            companies_max_companies=_int(env, "MOF_COMPANIES_MAX_COMPANIES", 10, maximum=100),
            companies_max_concurrency=_int(env, "MOF_COMPANIES_MAX_CONCURRENCY", 2, maximum=16),
        )
        if bool(settings.companies_catalog) != bool(settings.companies_data_root):
            missing = "MOF_COMPANIES_DATA_ROOT" if settings.companies_catalog else "MOF_COMPANIES_CATALOG"
            raise ConfigError(f"{missing} is required with the other MOF_COMPANIES_* store variable")
        if not settings.path.startswith("/"):
            raise ConfigError(f"MOF_MCP_PATH must start with '/', got {settings.path!r}")
        if settings.default_rows > settings.max_rows:
            raise ConfigError(
                f"MOF_MCP_DEFAULT_ROWS ({settings.default_rows}) must not exceed "
                f"MOF_MCP_MAX_ROWS ({settings.max_rows})"
            )
        if settings.transport == "http" and not settings.tokens:
            raise ConfigError(
                "MOF_MCP_TRANSPORT=http requires MOF_MCP_TOKENS: the HTTP server never runs unauthenticated"
            )
        return settings
