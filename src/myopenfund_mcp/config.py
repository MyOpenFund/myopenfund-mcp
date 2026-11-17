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


# company-corpus's TTM Q4 rules (company_corpus.financials.Q4Rule)
Q4_RULES = ("ytd9", "sum_of_quarters")


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


def _optional_int(env: Mapping[str, str], name: str, minimum: int = 1, maximum: int | None = None) -> int | None:
    """An integer with no default: unset or empty stays None (the library's own choice applies)."""
    return None if _get(env, name) is None else _int(env, name, minimum, minimum, maximum)


_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name)
    if raw is None:
        return default
    if raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


# A host name or an address (an IPv6 literal bare or in brackets): no scheme, no port, no path.
_HOST = re.compile(r"[A-Za-z0-9._:\[\]-]{1,253}")
# Qdrant refuses a collection name holding '/' or longer than 255 characters.
_COLLECTION = re.compile(r"[^/\s]{1,255}")


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
    # MOF_COMPANIES_Q4_RULE: how TTM derives an unreported Q4 — ytd9 (FY - 9-month YTD, the rule of
    # company-corpus's published outputs) |
    # sum_of_quarters (FY - Q1 - Q2 - Q3)
    companies_q4_rule: str = "ytd9"
    # MOF_RAG_QDRANT_HOST: the Qdrant server holding the embedded corpus; unset = no rag toolset
    rag_qdrant_host: str | None = None
    rag_qdrant_port: int = 6333  # MOF_RAG_QDRANT_PORT: Qdrant HTTP port
    # MOF_RAG_QDRANT_API_KEY: a read-only key is enough; kept out of repr, never logged or echoed
    rag_qdrant_api_key: str | None = field(default=None, repr=False)
    rag_qdrant_https: bool = False  # MOF_RAG_QDRANT_HTTPS: connect to Qdrant over TLS
    rag_qdrant_timeout: int = 10  # MOF_RAG_QDRANT_TIMEOUT: seconds one request to Qdrant may take
    rag_collection: str | None = None  # MOF_RAG_COLLECTION: the collection searched; needed with the host
    # MOF_RAG_MODEL_CACHE: directory of the pinned embedding model files; needed with the host
    rag_model_cache: str | None = None
    rag_onnx_threads: int | None = None  # MOF_RAG_ONNX_THREADS: unset = the ONNX runtime decides
    rag_default_limit: int = 8  # MOF_RAG_DEFAULT_LIMIT: passages returned when the caller gives no limit
    rag_max_limit: int = 20  # MOF_RAG_MAX_LIMIT: largest limit a caller may ask for
    # MOF_RAG_CODES_TTL_S: how long the collection's filter values (source codes, doc types, languages)
    # are kept before being read again
    rag_codes_ttl_s: int = 300

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
            companies_q4_rule=_get(env, "MOF_COMPANIES_Q4_RULE") or "ytd9",
            rag_qdrant_host=_get(env, "MOF_RAG_QDRANT_HOST"),
            rag_qdrant_port=_int(env, "MOF_RAG_QDRANT_PORT", 6333, maximum=65535),
            rag_qdrant_api_key=_get(env, "MOF_RAG_QDRANT_API_KEY"),
            rag_qdrant_https=_bool(env, "MOF_RAG_QDRANT_HTTPS", False),
            rag_qdrant_timeout=_int(env, "MOF_RAG_QDRANT_TIMEOUT", 10, maximum=300),
            rag_collection=_get(env, "MOF_RAG_COLLECTION"),
            rag_model_cache=_get(env, "MOF_RAG_MODEL_CACHE"),
            rag_onnx_threads=_optional_int(env, "MOF_RAG_ONNX_THREADS", maximum=256),
            rag_default_limit=_int(env, "MOF_RAG_DEFAULT_LIMIT", 8),
            rag_max_limit=_int(env, "MOF_RAG_MAX_LIMIT", 20, maximum=1000),
            rag_codes_ttl_s=_int(env, "MOF_RAG_CODES_TTL_S", 300, maximum=86_400),
        )
        if settings.companies_q4_rule not in Q4_RULES:
            raise ConfigError(f"MOF_COMPANIES_Q4_RULE must be one of {', '.join(Q4_RULES)}, "
                              f"got {settings.companies_q4_rule!r}")
        if bool(settings.companies_catalog) != bool(settings.companies_data_root):
            missing = "MOF_COMPANIES_DATA_ROOT" if settings.companies_catalog else "MOF_COMPANIES_CATALOG"
            raise ConfigError(f"{missing} is required with the other MOF_COMPANIES_* store variable")
        if bool(settings.rag_qdrant_host) != bool(settings.rag_collection):
            missing = "MOF_RAG_COLLECTION" if settings.rag_qdrant_host else "MOF_RAG_QDRANT_HOST"
            raise ConfigError(f"{missing} is required with the other of MOF_RAG_QDRANT_HOST / MOF_RAG_COLLECTION")
        if settings.rag_qdrant_host:
            # Neither value is echoed: a pasted URL may carry credentials.
            if _HOST.fullmatch(settings.rag_qdrant_host) is None:
                raise ConfigError("MOF_RAG_QDRANT_HOST must be a host name or an address, without scheme, "
                                  "port or path (the port is MOF_RAG_QDRANT_PORT, TLS is MOF_RAG_QDRANT_HTTPS)")
            if _COLLECTION.fullmatch(settings.rag_collection) is None:
                raise ConfigError("MOF_RAG_COLLECTION must be a collection name: 1-255 characters, "
                                  "no '/' and no whitespace")
            if not settings.rag_model_cache:
                raise ConfigError(
                    "MOF_RAG_MODEL_CACHE is required with MOF_RAG_QDRANT_HOST: the directory holding the "
                    "embedding model files (about 1.1 GB, never a temporary directory)"
                )
        if settings.rag_default_limit > settings.rag_max_limit:
            raise ConfigError(
                f"MOF_RAG_DEFAULT_LIMIT ({settings.rag_default_limit}) must not exceed "
                f"MOF_RAG_MAX_LIMIT ({settings.rag_max_limit})"
            )
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
