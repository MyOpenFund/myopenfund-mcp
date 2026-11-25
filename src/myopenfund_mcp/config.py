"""Server settings, from environment variables only.

Every variable and its default is documented in .env.example. A bad value stops
the server at start with the variable named — never a silent fallback.
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


class ConfigError(ValueError):
    pass


# company-corpus's TTM Q4 rules (company_corpus.financials.Q4Rule)
Q4_RULES = ("ytd9", "sum_of_quarters")
# rag-engine's ways of applying a date bound (rag_engine.config.DATE_FILTER_MODES), the default first.
# Written out here: this module never imports the [rag] extra.
RAG_DATE_FILTER_MODES = ("auto", "interval", "legacy")
COMPANIES_SOURCES = ("auto", "store", "sec")  # MOF_COMPANIES_SOURCE: auto = the store when configured, else SEC
CACHE_MODES = ("on", "off")  # MOF_CACHE
# MOF_MARKET_SOURCE: where market_prices reads (MCP chantier 3 Q9): auto = the store when it holds the rows,
# else a live fetch with the caller's own key | store | live
MARKET_SOURCES = ("auto", "store", "live")
# rag-engine's ONNX execution providers (OnnxEmbedder `provider`); written out, like the modes above.
ONNX_PROVIDERS = ("cpu", "coreml")
#: Largest MOF_SEC_MAX_RPS: SEC's fair-access limit is 10 requests per second.
SEC_MAX_RPS_LIMIT = 10.0
#: Most characters of MOF_SEC_CONTACT (it travels in a User-Agent header).
SEC_CONTACT_MAX_CHARS = 200
_FORM = re.compile(r"[A-Za-z0-9][A-Za-z0-9./-]{0,19}")


def default_cache_dir() -> str:
    """The per-user cache folder of the platform plus ``myopenfund-mcp``: ``$XDG_CACHE_HOME`` or
    ``~/.cache`` on Linux, ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%`` on Windows. A relative
    ``XDG_CACHE_HOME`` / ``LOCALAPPDATA`` is ignored (the XDG specification says so). Raises ConfigError
    when no absolute folder can be found (no home directory): the caller then sets MOF_CACHE_DIR."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Caches")
    elif sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA", "")
        base = local if os.path.isabs(local) else os.path.join(home, "AppData", "Local")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "")
        base = xdg if os.path.isabs(xdg) else os.path.join(home, ".cache")
    if not os.path.isabs(base):
        raise ConfigError("no home directory found to place the cache in: set MOF_CACHE_DIR to an "
                          "absolute path")
    return os.path.join(base, "myopenfund-mcp")


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
        allowed = f"from {minimum} to {maximum}" if maximum is not None else f">= {minimum}"
        raise ConfigError(f"{name} must be an integer {allowed}, got {raw!r}")
    value = int(raw)
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _optional_int(env: Mapping[str, str], name: str, minimum: int = 1, maximum: int | None = None) -> int | None:
    """An integer with no default: unset or empty stays None (the library's own choice applies)."""
    return None if _get(env, name) is None else _int(env, name, minimum, minimum, maximum)


def _float(env: Mapping[str, str], name: str, default: float, maximum: float) -> float:
    """A decimal number in (0, maximum]; the raw value is named in the error only when it is not a secret
    (none of the callers' values is)."""
    raw = _get(env, name)
    if raw is None:
        return default
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", raw) is None:
        raise ConfigError(f"{name} must be a number above 0 and at most {maximum:g}, got {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or value <= 0 or value > maximum:
        raise ConfigError(f"{name} must be above 0 and at most {maximum:g}, got {raw!r}")
    return value


def _forms(raw: str) -> tuple[str, ...]:
    """Comma-separated form types, stripped, upper-cased, without duplicates, in order."""
    forms = [f.strip().upper() for f in raw.split(",")]
    if not all(_FORM.fullmatch(f) for f in forms):
        raise ConfigError("MOF_FILINGS_DEFAULT_FORMS must be comma-separated form types such as "
                          f"10-K,10-Q, got {raw!r}")
    return tuple(dict.fromkeys(forms))


#: Largest poll interval of `warm` (seconds), for MOF_WARM_POLL_INTERVAL_S and --poll-interval.
WARM_POLL_MAX_S = 3600

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


# A host name or an IPv4 address: no scheme, no port, no path. IPv6 literals are checked apart.
_HOST = re.compile(r"[A-Za-z0-9._-]{1,253}")
#: Most characters of MOF_RAG_DESCRIPTION_NOTES: the text is sent to every client with the tool list.
RAG_NOTES_MAX_CHARS = 1500
# Qdrant refuses a collection name holding '/' or longer than 255 characters.
_COLLECTION = re.compile(r"[^/\s]{1,255}")


def _is_ipv6(text: str) -> bool:
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        return False
    return True


def _check_rag_host(host: str) -> None:
    """Refuse a MOF_RAG_QDRANT_HOST that is not a bare host. The value is never echoed: a pasted URL
    may carry credentials."""
    name = "MOF_RAG_QDRANT_HOST"
    if "://" in host or "/" in host:
        raise ConfigError(f"{name} must be a host name or an address, without scheme or path "
                          "(TLS is MOF_RAG_QDRANT_HTTPS, the port is MOF_RAG_QDRANT_PORT)")
    if host.startswith("["):
        # A bracketed IPv6 literal, alone: "[::1]". "[::1]:6333" carries a port.
        if host.endswith("]") and _is_ipv6(host[1:-1]):
            return
        if "]:" in host and _is_ipv6(host[1:host.index("]:")]):
            raise ConfigError(f"{name} must not carry a port: give the address alone and the port in "
                              "MOF_RAG_QDRANT_PORT")
        raise ConfigError(f"{name} is not an IPv6 address in brackets")
    if ":" in host:
        if _is_ipv6(host):
            return
        raise ConfigError(f"{name} must not carry a port: give the host alone and the port in "
                          "MOF_RAG_QDRANT_PORT")
    if _HOST.fullmatch(host) is None:
        raise ConfigError(f"{name} must be a host name or an address, without scheme, port or path "
                          "(the port is MOF_RAG_QDRANT_PORT, TLS is MOF_RAG_QDRANT_HTTPS)")


#: Default MOF_MARKET_NO_RESTRICTED_CLIENTS: the agents' client (spec §13: it may post publicly).
NO_RESTRICTED_CLIENTS = ("agents",)


def _no_restricted_clients(env: Mapping[str, str]) -> tuple[str, ...]:
    """MOF_MARKET_NO_RESTRICTED_CLIENTS: client ids, comma-separated, without duplicates, in order. An id is
    never echoed (a token pasted here must not reach a log): errors name the entry by position."""
    from . import auth  # auth imports this module

    raw = _get(env, "MOF_MARKET_NO_RESTRICTED_CLIENTS")
    if raw is None:
        return NO_RESTRICTED_CLIENTS
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not ids:  # "," would otherwise disable the guard silently
        raise ConfigError("MOF_MARKET_NO_RESTRICTED_CLIENTS is set but holds no client id: give at least one "
                          "(an empty value keeps the default, agents)")
    for position, client in enumerate(ids, start=1):
        if not auth.is_valid_client_id(client):
            raise ConfigError(f"MOF_MARKET_NO_RESTRICTED_CLIENTS entry {position}: a client id is 1-32 "
                              "characters of A-Z a-z 0-9 _ . -")
    return tuple(dict.fromkeys(ids))


#: Default MOF_MARKET_TOOL_ONLY_SOURCES: sources whose rows only market_prices serves (instrument_id only):
#: the private export's rows hold private tokens and labels (market-data spec §9.3, §13).
TOOL_ONLY_SOURCES = ("private_history",)
_SOURCE_CODE = re.compile(r"[a-z][a-z0-9_]{0,62}")  # open-data-store's source-code rule, written out


def _tool_only_sources(env: Mapping[str, str]) -> tuple[str, ...]:
    """MOF_MARKET_TOOL_ONLY_SOURCES: source codes, comma-separated, without duplicates, in order; each in the
    registry's code format. Errors name the entry by position."""
    name = "MOF_MARKET_TOOL_ONLY_SOURCES"
    raw = _get(env, name)
    if raw is None:
        return TOOL_ONLY_SOURCES
    codes = [item.strip() for item in raw.split(",") if item.strip()]
    if not codes:
        raise ConfigError(f"{name} is set but holds no source code (an empty value keeps the default, "
                          "private_history)")
    for position, code in enumerate(codes, start=1):
        if not _SOURCE_CODE.fullmatch(code):
            raise ConfigError(f"{name} entry {position}: a source code is 1-63 characters of a-z 0-9 _, "
                              "starting with a letter")
    return tuple(dict.fromkeys(codes))


def _restricted_allow(env: Mapping[str, str], never: tuple[str, ...]) -> Mapping[str, frozenset[str]]:
    """MOF_MARKET_RESTRICTED_ALLOW, checked against the clients of MOF_MCP_TOKENS, the restricted codes of
    the source registry and the clients that never get one (`never`). Unset = empty without importing the
    [market] extra."""
    spec = _get(env, "MOF_MARKET_RESTRICTED_ALLOW")
    if spec is None:
        return {}
    from . import auth  # auth imports this module

    tokens = _get(env, "MOF_MCP_TOKENS")
    if tokens is None:
        raise ConfigError("MOF_MARKET_RESTRICTED_ALLOW names clients of MOF_MCP_TOKENS, which is not set: "
                          "without tokens no client can be given restricted sources")
    client_ids = frozenset(e.client_id for e in auth.parse_tokens(tokens))
    return auth.parse_restricted_allow(spec, client_ids, auth.registered_restricted_sources(), frozenset(never))


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
    # MOF_COMPANIES_CATALOG: open-data-store catalogue as lake_reader; unset = no companies/datasets toolsets
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
    # MOF_RAG_QDRANT_TIMEOUT: seconds ONE request to Qdrant may take (a call makes several)
    rag_qdrant_timeout: int = 10
    rag_collection: str | None = None  # MOF_RAG_COLLECTION: the collection searched; needed with the host
    # MOF_RAG_MODEL_CACHE: directory of the pinned embedding model files; needed with the host
    rag_model_cache: str | None = None
    rag_onnx_threads: int | None = None  # MOF_RAG_ONNX_THREADS: unset = the ONNX runtime decides
    rag_default_limit: int = 8  # MOF_RAG_DEFAULT_LIMIT: passages returned when the caller gives no limit
    rag_max_limit: int = 20  # MOF_RAG_MAX_LIMIT: largest limit a caller may ask for
    # MOF_RAG_CODES_TTL_S: how long the collection's filter values (source codes, doc types, languages)
    # are kept before being read again
    rag_codes_ttl_s: int = 300
    # MOF_RAG_CODES_RETRY_S: least seconds between two reads of the filter values that the TTL did not
    # ask for (before a value is refused; after a read that failed)
    rag_codes_retry_s: int = 5
    # MOF_RAG_CALL_TIMEOUT_S: time budget of one rag_search call, its wait for a free slot included;
    # never below MOF_RAG_QDRANT_TIMEOUT
    rag_call_timeout_s: int = 15
    rag_max_concurrency: int = 4  # MOF_RAG_MAX_CONCURRENCY: searches running at once (worker threads)
    # MOF_RAG_CHECK_RETRY_S: seconds a failed collection check is remembered: calls made meanwhile are
    # answered at once with its reason instead of asking Qdrant again (a failed search is not
    # remembered: it makes the next call run the check)
    rag_check_retry_s: int = 5
    # MOF_RAG_DESCRIPTION_NOTES: text appended to the description of rag_search (what the collection
    # holds, what its codes mean); empty = nothing appended
    rag_description_notes: str = ""
    # MOF_RAG_DATE_FILTER_MODE: how rag-engine applies a date bound -- auto (the intervals on a collection
    # where date_start and date_end are indexed, the date field elsewhere) | interval | legacy
    rag_date_filter_mode: str = "auto"
    # MOF_RAG_DATE_FILTER_REFRESH_S: in auto mode, seconds the answer "are the intervals indexed?" is kept
    # before the collection is asked again (0 = before every date-bounded search)
    rag_date_filter_refresh_s: int = 300
    # MOF_RAG_DATE_FILTER_RETRY_S: in auto mode, seconds before a collection whose payload schema could not
    # be read is asked again (meanwhile the last answer, or legacy, is used)
    rag_date_filter_retry_s: int = 30
    # --- no-vault mode: SEC data fetched live, filings indexed on this machine ---
    # MOF_SEC_CONTACT: e-mail or name sent in the User-Agent, as SEC's fair-access rules ask; kept out of
    # repr, never logged or echoed
    sec_contact: str | None = field(default=None, repr=False)
    sec_max_rps: float = 5.0  # MOF_SEC_MAX_RPS: requests per second to SEC, above 0 and at most 10
    sec_timeout_s: int = 30  # MOF_SEC_TIMEOUT_S: seconds one request to SEC may take
    sec_retries: int = 4  # MOF_SEC_RETRIES: retries of a failed request (0 = none)
    sec_call_budget_s: int = 50  # MOF_SEC_CALL_BUDGET_S: time budget of one tool call that fetches from SEC
    # MOF_SEC_POSITIONS_LOOKBACK_Y: years before now whose filings get statement positions when a call
    # names no period start (positions cost three requests per filing); MOF_SEC_TTM_LOOKBACK_M moves
    # that start further back, as it does a call's period_from
    sec_positions_lookback_y: int = 5
    # MOF_SEC_MEMO_COMPANIES: companies whose live tables are kept in memory (the oldest built is dropped)
    sec_memo_companies: int = 50
    # MOF_SEC_POSITIONS_DONOR_DEPTH: earlier report filings looked through for the positions a filing
    # without its own borrows, as the store's selection does (0 = none)
    sec_positions_donor_depth: int = 4
    # MOF_SEC_TTM_LOOKBACK_M: months before the period's start (period_from, or the start the lookback
    # above gives a call without one) whose facts' filings also get positions, so the
    # period's TTM and ratios read the store's values. 20 covers both Q4 rules: a quarterly TTM spans
    # four quarter ends up to 300 days back; its earliest quarter, when derived, reads a 9-month YTD up
    # to 100 days earlier (ytd9: < 400 days, 14 months) or three quarters up to 300 days earlier
    # (sum_of_quarters: < 600 days, 20 months); the prior-year averages read up to 385 days back
    sec_ttm_lookback_m: int = 20
    # MOF_SEC_FAILURE_TTL_S: seconds a failed fetch of a company is remembered: calls meanwhile get its
    # error at once instead of starting the fetch again
    sec_failure_ttl_s: int = 60
    # MOF_SEC_FRESHNESS_BUDGET_S: seconds a call gives the auto-mode check of the store against SEC (the
    # companies whose SEC filing list did not come by then are answered by the store, freshness_unchecked)
    sec_freshness_budget_s: int = 5
    # MOF_COMPANIES_SOURCE: auto (the store when MOF_COMPANIES_CATALOG is set, SEC when MOF_SEC_CONTACT is,
    # neither = no companies toolset) | store | sec
    companies_source: str = "auto"
    # MOF_CACHE: on | off (off = entries live for the process only, in a temporary folder removed at exit)
    cache: str = "on"
    # MOF_CACHE_DIR: absolute path; None = the platform's cache folder, found by cache_path() when needed
    cache_dir: str | None = None
    # MOF_CACHE_MAX_GB: size of the cached HTTP bodies above which the oldest-written are removed (the
    # model files and the local index live under the same folder, on top, and are never removed)
    cache_max_gb: int = 20
    # MOF_CACHE_EVICT_TO: fraction of MOF_CACHE_MAX_GB that eviction shrinks the cache down to
    cache_evict_to: float = 0.9
    # MOF_CACHE_TMP_GRACE_S: seconds a temporary file of an interrupted write is left before eviction
    # removes it
    cache_tmp_grace_s: int = 3600
    cache_ttl_submissions_h: int = 24  # MOF_CACHE_TTL_SUBMISSIONS_H: hours a filing list is reused
    cache_ttl_companyfacts_h: int = 24  # MOF_CACHE_TTL_COMPANYFACTS_H: hours a company-facts file is reused
    filings_max_docs: int = 40  # MOF_FILINGS_MAX_DOCS: filings downloaded and indexed by one call
    # MOF_FILINGS_SEARCH_DEFAULT_LIMIT: passages companies_search_filings returns when the caller gives no limit
    filings_search_default_limit: int = 8
    # MOF_FILINGS_SEARCH_MAX_LIMIT: largest limit a caller may ask companies_search_filings for
    filings_search_max_limit: int = 20
    # MOF_FILINGS_SEARCH_MAX_DOCS: fetched filings a search without `companies` may cover (more: pass companies)
    filings_search_max_docs: int = 40
    # MOF_FILINGS_NOT_READY_LISTED: not-yet-ready documents listed in a search result (the rest are counted)
    filings_not_ready_listed: int = 10
    index_workers: int = 1  # MOF_INDEX_WORKERS: filings indexed in parallel
    # MOF_INDEX_CLOSE_TIMEOUT_S: seconds closing the filings indexer waits for its workers' current step
    # (a batch, a fetch, a model load); past it the index is released and the workers die with the process
    index_close_timeout_s: int = 10
    # MOF_WARM_POLL_INTERVAL_S: seconds `myopenfund-mcp warm` waits between two looks at the SEC fetch
    # (still warming) and at the indexing progress; the --poll-interval option overrides it
    warm_poll_interval_s: float = 5.0
    # MOF_WARM_STALL_S: seconds without any progress after which `warm` stops waiting for the documents
    # still queued or indexing, lists them and exits 1
    warm_stall_s: int = 1800
    rag_onnx_provider: str = "cpu"  # MOF_RAG_ONNX_PROVIDER: cpu | coreml
    search_rrf_k: int = 60  # MOF_SEARCH_RRF_K: constant k of the reciprocal-rank fusion of a search
    search_candidates: int = 50  # MOF_SEARCH_CANDIDATES: passages each retriever hands to the fusion
    search_eigenmind: bool = False  # MOF_SEARCH_EIGENMIND: add the graph retriever to a search (off until evaluated)
    # MOF_FILINGS_DEFAULT_FORMS: comma-separated form types, stripped and without duplicates
    filings_default_forms: tuple[str, ...] = ("10-K",)
    chunk_max_tokens: int = 480  # MOF_CHUNK_MAX_TOKENS: largest passage, in tokens
    chunk_overlap_tokens: int = 32  # MOF_CHUNK_OVERLAP_TOKENS: tokens shared by neighbouring passages
    market_source: str = "auto"  # MOF_MARKET_SOURCE: auto | store | live
    # MOF_MARKET_RESTRICTED_ALLOW: client:src1|src2,... restricted sources whose stored rows each client of
    # MOF_MCP_TOKENS may read (auth.allowed_restricted_for); default none; kept out of repr. A read-only
    # mapping (a dict given in code is copied into one): no runtime code can widen the gate.
    market_restricted_allow: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False)
    # MOF_MARKET_NO_RESTRICTED_CLIENTS: clients the allow-list may never name (spec §13: the agents' token)
    market_no_restricted_clients: tuple[str, ...] = NO_RESTRICTED_CLIENTS
    market_max_instruments: int = 10  # MOF_MARKET_MAX_INSTRUMENTS: instruments per market_prices call
    market_max_rows: int = 5000  # MOF_MARKET_MAX_ROWS: rows per market_prices result (refused above, never cut)
    market_live_timeout_s: int = 60  # MOF_MARKET_LIVE_TIMEOUT_S: time budget of a live market-data fetch
    # MOF_MARKET_LIVE_CONCURRENCY: live fetches running at once (an abandoned one keeps its slot until it ends);
    # one more is answered [market_busy], never queued
    market_live_concurrency: int = 1
    # MOF_MARKET_TOOL_ONLY_SOURCES: sources whose rows the dataset tools never serve, whatever the allow-list
    market_tool_only_sources: tuple[str, ...] = field(default=TOOL_ONLY_SOURCES, repr=False)

    def __post_init__(self) -> None:
        allow = self.market_restricted_allow
        if not isinstance(allow, MappingProxyType):
            object.__setattr__(self, "market_restricted_allow",
                               MappingProxyType({k: frozenset(v) for k, v in dict(allow).items()}))

    def cache_path(self) -> str:
        """The cache folder: MOF_CACHE_DIR, or the platform default (ConfigError when there is none)."""
        return self.cache_dir or default_cache_dir()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        transport = _get(env, "MOF_MCP_TRANSPORT") or "stdio"
        if transport not in ("stdio", "http"):
            raise ConfigError(f"MOF_MCP_TRANSPORT must be 'stdio' or 'http', got {transport!r}")
        no_restricted = _no_restricted_clients(env)
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
            rag_codes_retry_s=_int(env, "MOF_RAG_CODES_RETRY_S", 5, maximum=3600),
            rag_call_timeout_s=_int(env, "MOF_RAG_CALL_TIMEOUT_S", 15, maximum=3600),
            rag_max_concurrency=_int(env, "MOF_RAG_MAX_CONCURRENCY", 4, maximum=64),
            rag_check_retry_s=_int(env, "MOF_RAG_CHECK_RETRY_S", 5, maximum=3600),
            rag_description_notes=_get(env, "MOF_RAG_DESCRIPTION_NOTES") or "",
            rag_date_filter_mode=_get(env, "MOF_RAG_DATE_FILTER_MODE") or "auto",
            rag_date_filter_refresh_s=_int(env, "MOF_RAG_DATE_FILTER_REFRESH_S", 300, minimum=0, maximum=86_400),
            rag_date_filter_retry_s=_int(env, "MOF_RAG_DATE_FILTER_RETRY_S", 30, maximum=3600),
            sec_contact=_get(env, "MOF_SEC_CONTACT"),
            sec_max_rps=_float(env, "MOF_SEC_MAX_RPS", 5.0, SEC_MAX_RPS_LIMIT),
            sec_timeout_s=_int(env, "MOF_SEC_TIMEOUT_S", 30, maximum=600),
            sec_retries=_int(env, "MOF_SEC_RETRIES", 4, minimum=0, maximum=20),
            sec_call_budget_s=_int(env, "MOF_SEC_CALL_BUDGET_S", 50, maximum=3600),
            sec_positions_lookback_y=_int(env, "MOF_SEC_POSITIONS_LOOKBACK_Y", 5, maximum=50),
            sec_memo_companies=_int(env, "MOF_SEC_MEMO_COMPANIES", 50, maximum=10000),
            sec_positions_donor_depth=_int(env, "MOF_SEC_POSITIONS_DONOR_DEPTH", 4, minimum=0, maximum=40),
            sec_ttm_lookback_m=_int(env, "MOF_SEC_TTM_LOOKBACK_M", 20, minimum=0, maximum=120),
            sec_failure_ttl_s=_int(env, "MOF_SEC_FAILURE_TTL_S", 60, maximum=3600),
            sec_freshness_budget_s=_int(env, "MOF_SEC_FRESHNESS_BUDGET_S", 5, maximum=600),
            companies_source=_get(env, "MOF_COMPANIES_SOURCE") or "auto",
            cache=(_get(env, "MOF_CACHE") or "on").lower(),
            cache_dir=_get(env, "MOF_CACHE_DIR"),
            cache_max_gb=_int(env, "MOF_CACHE_MAX_GB", 20, maximum=100_000),
            cache_evict_to=_float(env, "MOF_CACHE_EVICT_TO", 0.9, 1.0),
            cache_tmp_grace_s=_int(env, "MOF_CACHE_TMP_GRACE_S", 3600, minimum=60, maximum=86_400),
            cache_ttl_submissions_h=_int(env, "MOF_CACHE_TTL_SUBMISSIONS_H", 24, maximum=8760),
            cache_ttl_companyfacts_h=_int(env, "MOF_CACHE_TTL_COMPANYFACTS_H", 24, maximum=8760),
            filings_max_docs=_int(env, "MOF_FILINGS_MAX_DOCS", 40, maximum=1000),
            filings_search_default_limit=_int(env, "MOF_FILINGS_SEARCH_DEFAULT_LIMIT", 8, maximum=1000),
            filings_search_max_limit=_int(env, "MOF_FILINGS_SEARCH_MAX_LIMIT", 20, maximum=1000),
            filings_search_max_docs=_int(env, "MOF_FILINGS_SEARCH_MAX_DOCS", 40, maximum=10000),
            filings_not_ready_listed=_int(env, "MOF_FILINGS_NOT_READY_LISTED", 10, minimum=0, maximum=1000),
            index_workers=_int(env, "MOF_INDEX_WORKERS", 1, maximum=32),
            index_close_timeout_s=_int(env, "MOF_INDEX_CLOSE_TIMEOUT_S", 10, minimum=0, maximum=3600),
            warm_poll_interval_s=_float(env, "MOF_WARM_POLL_INTERVAL_S", 5.0, WARM_POLL_MAX_S),
            warm_stall_s=_int(env, "MOF_WARM_STALL_S", 1800, maximum=604800),
            rag_onnx_provider=_get(env, "MOF_RAG_ONNX_PROVIDER") or "cpu",
            search_rrf_k=_int(env, "MOF_SEARCH_RRF_K", 60, maximum=1000),
            search_candidates=_int(env, "MOF_SEARCH_CANDIDATES", 50, maximum=1000),
            search_eigenmind=_bool(env, "MOF_SEARCH_EIGENMIND", False),
            filings_default_forms=_forms(_get(env, "MOF_FILINGS_DEFAULT_FORMS") or "10-K"),
            chunk_max_tokens=_int(env, "MOF_CHUNK_MAX_TOKENS", 480, minimum=16, maximum=8192),
            chunk_overlap_tokens=_int(env, "MOF_CHUNK_OVERLAP_TOKENS", 32, minimum=0, maximum=2048),
            market_source=_get(env, "MOF_MARKET_SOURCE") or "auto",
            market_restricted_allow=_restricted_allow(env, no_restricted),
            market_no_restricted_clients=no_restricted,
            market_max_instruments=_int(env, "MOF_MARKET_MAX_INSTRUMENTS", 10, maximum=100),
            market_max_rows=_int(env, "MOF_MARKET_MAX_ROWS", 5000, maximum=1_000_000),
            market_live_timeout_s=_int(env, "MOF_MARKET_LIVE_TIMEOUT_S", 60, maximum=3600),
            market_live_concurrency=_int(env, "MOF_MARKET_LIVE_CONCURRENCY", 1, maximum=16),
            market_tool_only_sources=_tool_only_sources(env),
        )
        if settings.market_source not in MARKET_SOURCES:
            raise ConfigError(f"MOF_MARKET_SOURCE must be one of {', '.join(MARKET_SOURCES)}, "
                              f"got {settings.market_source!r}")
        if settings.market_source == "store" and not settings.companies_catalog:
            raise ConfigError("MOF_MARKET_SOURCE=store requires MOF_COMPANIES_CATALOG "
                              "(and MOF_COMPANIES_DATA_ROOT): market prices are read from the same store")
        if settings.companies_source not in COMPANIES_SOURCES:
            raise ConfigError(f"MOF_COMPANIES_SOURCE must be one of {', '.join(COMPANIES_SOURCES)}, "
                              f"got {settings.companies_source!r}")
        if settings.companies_source == "store" and not settings.companies_catalog:
            raise ConfigError("MOF_COMPANIES_SOURCE=store requires MOF_COMPANIES_CATALOG "
                              "(and MOF_COMPANIES_DATA_ROOT)")
        if settings.companies_source == "sec" and not settings.sec_contact:
            raise ConfigError("MOF_COMPANIES_SOURCE=sec requires MOF_SEC_CONTACT: SEC asks every client "
                              "to identify itself")
        if settings.cache not in CACHE_MODES:
            raise ConfigError(f"MOF_CACHE must be one of {', '.join(CACHE_MODES)}, got {settings.cache!r}")
        if settings.cache_dir is not None and not os.path.isabs(settings.cache_dir):
            raise ConfigError("MOF_CACHE_DIR must be an absolute path ('~' is not expanded): a relative one "
                              "would depend on the directory the server is started from")
        if settings.sec_contact is not None and (
            len(settings.sec_contact) > SEC_CONTACT_MAX_CHARS
            or not all(" " <= c <= "~" for c in settings.sec_contact)
        ):
            # The value is never echoed: it is the secret itself.
            raise ConfigError(f"MOF_SEC_CONTACT must be printable ASCII (no line break, no accent), at most "
                              f"{SEC_CONTACT_MAX_CHARS} characters")
        if settings.rag_onnx_provider not in ONNX_PROVIDERS:
            raise ConfigError(f"MOF_RAG_ONNX_PROVIDER must be one of {', '.join(ONNX_PROVIDERS)}, "
                              f"got {settings.rag_onnx_provider!r}")
        if settings.chunk_overlap_tokens >= settings.chunk_max_tokens:
            raise ConfigError(
                f"MOF_CHUNK_OVERLAP_TOKENS ({settings.chunk_overlap_tokens}) must be below "
                f"MOF_CHUNK_MAX_TOKENS ({settings.chunk_max_tokens})"
            )
        if settings.companies_q4_rule not in Q4_RULES:
            raise ConfigError(f"MOF_COMPANIES_Q4_RULE must be one of {', '.join(Q4_RULES)}, "
                              f"got {settings.companies_q4_rule!r}")
        if settings.rag_date_filter_mode not in RAG_DATE_FILTER_MODES:
            raise ConfigError(f"MOF_RAG_DATE_FILTER_MODE must be one of {', '.join(RAG_DATE_FILTER_MODES)}, "
                              f"got {settings.rag_date_filter_mode!r}")
        if bool(settings.companies_catalog) != bool(settings.companies_data_root):
            missing = "MOF_COMPANIES_DATA_ROOT" if settings.companies_catalog else "MOF_COMPANIES_CATALOG"
            raise ConfigError(f"{missing} is required with the other MOF_COMPANIES_* store variable")
        if bool(settings.rag_qdrant_host) != bool(settings.rag_collection):
            missing = "MOF_RAG_COLLECTION" if settings.rag_qdrant_host else "MOF_RAG_QDRANT_HOST"
            raise ConfigError(f"{missing} is required with the other of MOF_RAG_QDRANT_HOST / MOF_RAG_COLLECTION")
        if settings.rag_qdrant_host:
            _check_rag_host(settings.rag_qdrant_host)
            if _COLLECTION.fullmatch(settings.rag_collection) is None:
                raise ConfigError("MOF_RAG_COLLECTION must be a collection name: 1-255 characters, "
                                  "no '/' and no whitespace")
            if not settings.rag_model_cache:
                raise ConfigError(
                    "MOF_RAG_MODEL_CACHE is required with MOF_RAG_QDRANT_HOST: the directory holding the "
                    "embedding model files (about 1.1 GB, never a temporary directory)"
                )
            if not os.path.isabs(settings.rag_model_cache):
                raise ConfigError("MOF_RAG_MODEL_CACHE must be an absolute path: a relative one would depend "
                                  "on the directory the server is started from")
        if settings.rag_call_timeout_s < settings.rag_qdrant_timeout:
            raise ConfigError(
                f"MOF_RAG_CALL_TIMEOUT_S ({settings.rag_call_timeout_s}) must not be below "
                f"MOF_RAG_QDRANT_TIMEOUT ({settings.rag_qdrant_timeout}): a call must outlast one request to Qdrant"
            )
        if len(settings.rag_description_notes) > RAG_NOTES_MAX_CHARS:
            raise ConfigError(f"MOF_RAG_DESCRIPTION_NOTES must hold at most {RAG_NOTES_MAX_CHARS} characters, "
                              f"got {len(settings.rag_description_notes)}")
        if settings.rag_default_limit > settings.rag_max_limit:
            raise ConfigError(
                f"MOF_RAG_DEFAULT_LIMIT ({settings.rag_default_limit}) must not exceed "
                f"MOF_RAG_MAX_LIMIT ({settings.rag_max_limit})"
            )
        if settings.filings_search_default_limit > settings.filings_search_max_limit:
            raise ConfigError(
                f"MOF_FILINGS_SEARCH_DEFAULT_LIMIT ({settings.filings_search_default_limit}) must not exceed "
                f"MOF_FILINGS_SEARCH_MAX_LIMIT ({settings.filings_search_max_limit})"
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
