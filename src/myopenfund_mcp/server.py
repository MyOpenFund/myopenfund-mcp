"""Assemble the server: core + every toolset whose stack is installed and configured."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import partial

import anyio
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from . import __version__
from .audit import AuditMiddleware
from .auth import HashedTokenVerifier, caller_tier, parse_tokens
from .config import ConfigError, Settings
from .errors import BACKEND_UNAVAILABLE, error_code
from .filings import GRAPH_NOT_AVAILABLE
from .paging import jsonable

log = logging.getLogger("myopenfund_mcp.server")

MISSING_VAULT_EXTRA = ("MOF_VAULT_DATABASE_URL is set but the [vault] extra is not installed: "
                       "pip install 'myopenfund-mcp[vault]'")
MISSING_COMPANIES_EXTRA = ("MOF_COMPANIES_CATALOG is set but the [companies] extra is not installed: "
                           "pip install 'myopenfund-mcp[companies]'")
MISSING_LIVE_EXTRA = ("MOF_SEC_CONTACT is set but the [companies] extra is not installed: "
                      "pip install 'myopenfund-mcp[companies]'")

MISSING_MARKET_EXTRA = ("MOF_MARKET_SOURCE is set to {mode} but the [market] extra is not installed: "
                        "pip install 'myopenfund-mcp[market]'")
#: the distributions of the [market] extra: a missing one means the extra is not installed
_MARKET_MODULES = ("market_data", "open_data_store")

MISSING_RAG_EXTRA = ("MOF_RAG_QDRANT_HOST is set but the [rag] extra is not installed: "
                     "pip install 'myopenfund-mcp[rag]'")

FILINGS_NO_SEC = ("not available: the filings tools download documents from SEC, which needs MOF_SEC_CONTACT "
                  "(e.g. an email address)")
FILINGS_STORE_MODE = ("not available: MOF_COMPANIES_SOURCE=store reads nothing from SEC; the filings tools need "
                      "MOF_COMPANIES_SOURCE=auto or sec with MOF_SEC_CONTACT")
FILINGS_NO_CLIENT = "not available: this server has no SEC client (its live source was given to it)"
FILINGS_MISSING_EXTRA = ("not available: the [filings] extra is not installed: "
                         "pip install 'myopenfund-mcp[companies,filings]'")

RAG_IMPORT_FAILED = ("MOF_RAG_QDRANT_HOST is set but the rag toolset could not be imported, although the "
                     "[rag] extra seems installed: see the error logged just before this one")

INSTRUCTIONS = (
    "MyOpenFund: official primary sources only (central banks and SEC company filings). "
    "Call server_info first to see which toolsets this deployment enables. Results carry doc_id, "
    "dates and source URLs: cite them. Company data is point-in-time: pass as_of to see what was "
    "known then. Errors start with a stable [code]."
)


def _cache_dir(settings: Settings, sec) -> str | None:
    """The SEC cache folder: the opened one, else the configured one (None when off or not found)."""
    if sec is not None:
        return str(sec.cache.root)
    if settings.cache == "off":
        return None
    try:
        return settings.cache_path()
    except ConfigError:
        return None


def _close(owned: list) -> None:
    """Close the sources a server built, newest first (the live source before the store)."""
    for source in reversed(owned):
        try:
            source.close()
        except Exception as exc:  # noqa: BLE001 -- the others still close; the class only is logged
            log.warning("closing %s failed (%s)", type(source).__name__, type(exc).__name__)


def build_server(settings: Settings, *, vault_reader=None, companies_store=None, companies_live=None,
                 rag_backend=None, filings_indexer=None, market_adapter=None) -> FastMCP:
    """``companies_store`` / ``companies_live`` / ``filings_indexer``: injected sources (tests); the server
    closes only the ones it builds itself: at shutdown, or at once when a later step refuses to start.
    ``market_adapter``: a ``(source_code, budget) -> Adapter`` factory for market_prices' live fetches
    (tests)."""
    owned: list = []
    try:
        return _build_server(settings, owned, vault_reader=vault_reader, companies_store=companies_store,
                             companies_live=companies_live, rag_backend=rag_backend,
                             filings_indexer=filings_indexer, market_adapter=market_adapter)
    except BaseException:
        _close(owned)
        raise


def _filings(settings: Settings, sec, injected, owned: list):
    """``(lazy indexer, None)`` when the filings tools can be mounted, else ``(None, reason)``.

    They need the ``[filings]`` extra and an SEC client (a contact, and a mode that reads SEC). Nothing is
    built, opened or loaded here: the indexer is made by its first use, and closed at shutdown."""
    if injected is not None:
        from .filings.tools import LazyIndexer

        return LazyIndexer(lambda: injected), None
    if sec is None:
        if not settings.sec_contact:
            return None, FILINGS_NO_SEC
        return None, FILINGS_STORE_MODE if settings.companies_source == "store" else FILINGS_NO_CLIENT
    try:
        from .filings.indexer import FilingsIndexer
        from .filings.tools import LazyIndexer
    except ImportError as exc:  # optional: the other toolsets run without it
        log.info("filings: the [filings] extra is not importable (%s)", type(exc).__name__)
        return None, FILINGS_MISSING_EXTRA
    handle = LazyIndexer(partial(FilingsIndexer, settings, sec))
    owned.append(handle)  # after the live source and the router: closed before them
    return handle, None


def _market(root: FastMCP, settings: Settings, store, adapter, *, auth_enabled: bool) -> str:
    """Mount market_prices (namespace market) when the [market] extra is installed; its server_info status.
    Without the extra the server runs without it, unless MOF_MARKET_SOURCE asks for a mode explicitly."""
    try:
        from .market.tools import build_market_toolset
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in _MARKET_MODULES:
            raise  # a broken install, not a missing extra
        if settings.market_source != "auto":
            raise ConfigError(MISSING_MARKET_EXTRA.format(mode=settings.market_source)) from None
        log.info("market: the [market] extra is not installed (%s)", exc.name)
        return "not installed"
    root.mount(build_market_toolset(store, settings, auth_enabled=auth_enabled, adapter_factory=adapter),
               namespace="market")
    return "enabled"


def _build_server(settings: Settings, owned: list, *, vault_reader, companies_store, companies_live,
                  rag_backend, filings_indexer, market_adapter) -> FastMCP:
    auth_enabled = settings.transport == "http"
    if auth_enabled and not settings.tokens:
        raise ConfigError(
            "MOF_MCP_TRANSPORT=http requires MOF_MCP_TOKENS: the HTTP server never runs unauthenticated"
        )
    auth = HashedTokenVerifier(parse_tokens(settings.tokens)) if auth_enabled else None
    # mask_error_details: an unexpected exception (driver error, DSN in a message) reaches the
    # client as a generic error; our ToolError / ResourceError keep their `[code] message`.
    @asynccontextmanager
    async def lifespan(_server):  # `owned`: the sources this server built, closed at shutdown
        try:
            yield {}
        finally:
            _close(owned)

    root = FastMCP("myopenfund", instructions=INSTRUCTIONS, version=__version__, auth=auth,
                   middleware=[AuditMiddleware()], mask_error_details=True, lifespan=lifespan)

    if vault_reader is None and settings.vault_database_url:
        from .vault.reader import PostgresVaultReader

        # Configured but not installed is a deployment mistake: refuse to start rather than run without
        # the toolset the operator asked for. The ConfigError is raised outside the handler (no chaining).
        try:
            vault_reader = PostgresVaultReader(settings.vault_database_url,
                                               query_timeout_s=settings.query_timeout_s,
                                               connect_timeout_s=settings.connect_timeout_s)
        except ImportError:
            vault_reader = None
        if vault_reader is None:
            raise ConfigError(MISSING_VAULT_EXTRA)
    if vault_reader is not None:
        from .vault.tools import build_vault_toolset, register_vault_resources

        root.mount(build_vault_toolset(vault_reader, settings, auth_enabled=auth_enabled), namespace="vault")
        register_vault_resources(root, vault_reader, settings, auth_enabled=auth_enabled)

    if companies_store is None and settings.companies_catalog:
        # Same rule as the vault: configured but not installed refuses to start.
        try:
            from .companies.store import CompaniesStore

            companies_store = CompaniesStore(settings.companies_data_root, settings.companies_catalog,
                                             max_concurrency=settings.companies_max_concurrency,
                                             q4_rule=settings.companies_q4_rule)
        except ImportError:
            companies_store = None
        if companies_store is None:
            raise ConfigError(MISSING_COMPANIES_EXTRA)
        owned.append(companies_store)
    sec = None
    filings, filings_reason = None, FILINGS_NO_SEC  # set with the companies toolset
    if companies_live is None and settings.sec_contact and settings.companies_source != "store":
        # MOF_COMPANIES_SOURCE=store never reads SEC: no live source (nor cache folder) then.
        try:
            from .companies.build import build_live
        except ImportError:
            companies_live = None
        else:
            try:
                sec, companies_live = build_live(settings)
            except ImportError:
                companies_live = None
        if companies_live is None:
            raise ConfigError(MISSING_LIVE_EXTRA)
        owned.append(companies_live)
    if companies_store is not None or companies_live is not None:
        from .companies.build import build_router
        from .companies.common import dict_to_utc
        from .companies.datasets import build_datasets_toolset
        from .companies.tools import build_companies_toolset

        try:
            router = build_router(companies_store, companies_live, settings)
        except ValueError as exc:  # a source the mode needs is missing (from_env refuses it earlier)
            raise ConfigError(str(exc)) from None
        owned.append(router)  # its check threads: closed first at shutdown, then the sources
        if companies_store is not None:  # without a store, datasets_* are not mounted
            root.mount(build_datasets_toolset(companies_store, settings, auth_enabled=auth_enabled),
                       namespace="datasets")
        root.mount(build_companies_toolset(router, settings, auth_enabled=auth_enabled),
                   namespace="companies")
        filings, filings_reason = _filings(settings, sec, filings_indexer, owned)
        if filings is not None:
            from .filings.tools import build_filings_toolset

            root.mount(build_filings_toolset(filings.get, router, settings, auth_enabled=auth_enabled),
                       namespace="companies")

    if rag_backend is None and settings.rag_qdrant_host:
        # Same rule as the vault: configured but not installed refuses to start. A missing module is the
        # rag-engine package itself, or its ONNX backend when the embedding model is loaded in start()
        # (the engine then raises an ImportError naming its extra, caused by the missing module).
        # Any other ImportError (a broken install, a library that cannot be loaded) is not a missing
        # extra: it is reported as what it is, with its cause in the log.
        failure = None
        try:
            from .rag.backend import EngineRagBackend

            rag_backend = EngineRagBackend(settings)
            rag_backend.start()
        except ImportError as exc:
            rag_backend = None
            missing = isinstance(exc, ModuleNotFoundError) or isinstance(exc.__cause__, ModuleNotFoundError)
            if missing:
                log.error("rag: a module of the [rag] extra is missing (%s)", exc)
                failure = MISSING_RAG_EXTRA
            else:
                log.error("rag: the toolset could not be imported (%s: %s)", type(exc).__name__, exc,
                          exc_info=exc)
                failure = RAG_IMPORT_FAILED
        if failure is not None:
            raise ConfigError(failure)
    elif rag_backend is not None:
        rag_backend.start()
    if rag_backend is not None:
        from .rag.tools import build_rag_toolset

        root.mount(build_rag_toolset(rag_backend, settings, auth_enabled=auth_enabled), namespace="rag")

    market_status = _market(root, settings, companies_store, market_adapter, auth_enabled=auth_enabled)
    companies_status = ("enabled" if companies_store is not None or companies_live is not None
                        else "not configured")
    datasets_status = "enabled" if companies_store is not None else "not configured"

    # Open to any authenticated caller (no tier check): it is how a client learns its tier.
    @root.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
    async def server_info() -> dict:
        """What this deployment offers: enabled toolsets, your tier, version, data freshness.

        If the vault cannot be read, `freshness` is null and `freshness_error` holds the error code (e.g.
        backend_unavailable, timeout, schema_outdated); the rest still answers.
        With the rag toolset: `rag` (collection, embedding model, engine version, default and maximum
        limit, `status`: ready, or unavailable with a `reason`; and `source_codes`, `doc_types`,
        `languages`: the exact values rag_search accepts for these filters, null while not known).
        With the company store: `companies_freshness` (last publication, UTC; batch count), or
        `companies_freshness_error` if the store cannot be read.
        `filings`: the filing-text tools (companies_fetch_filings, companies_search_filings,
        companies_read_filing): `status` enabled with the embedding `model`, its `provider`, the
        `chunker_version` and `documents` (`ready`, `total`; null until the first filings call opens the
        index), or `status` with the reason they are not available; `graph`: not available yet.
        `toolsets.market`: market_prices, enabled or not installed (the [market] extra).
        `companies_source` (auto, store or sec: where company data comes from), `sec_contact_configured`
        (whether SEC can be reached; never the contact itself), `cache` (`dir`: the SEC cache folder,
        `bytes`: its size, null while no cache is open).
        """
        info = {
            "version": __version__,
            "tier": caller_tier(get_access_token(), auth_enabled=auth_enabled),
            "toolsets": {"vault": "enabled" if vault_reader is not None else "not configured",
                         "rag": "enabled" if rag_backend is not None else "not configured",
                         "companies": companies_status, "datasets": datasets_status,
                         "market": market_status},
            "freshness": None,
            "companies_source": settings.companies_source,
            "sec_contact_configured": bool(settings.sec_contact),
            "cache": {"dir": _cache_dir(settings, sec),
                      "bytes": sec.cache.size() if sec is not None else None},
        }
        if rag_backend is not None:
            # The state recorded at start and by the last calls: no request to Qdrant from here.
            info["rag"] = {**rag_backend.info(), "default_limit": settings.rag_default_limit,
                           "max_limit": settings.rag_max_limit}
        if vault_reader is not None:
            try:
                info["freshness"] = jsonable(await anyio.to_thread.run_sync(partial(vault_reader.freshness)))
            except Exception as exc:  # noqa: BLE001 -- an outage must not hide the tier and toolsets
                # The class only: an exception's text can hold the DSN.
                log.warning("server_info: freshness read failed (%s)", type(exc).__name__)
                info["freshness_error"] = error_code(exc, BACKEND_UNAVAILABLE)
        if filings is not None:
            from .filings.tools import filings_info

            try:
                indexer = filings.get()
            except Exception as exc:  # noqa: BLE001 -- closed (shutting down): the rest still answers
                log.warning("server_info: filings indexer unavailable (%s)", type(exc).__name__)
                info["filings"] = {"status": "unavailable", "graph": GRAPH_NOT_AVAILABLE}
            else:
                info["filings"] = await anyio.to_thread.run_sync(partial(filings_info, indexer, settings))
        else:
            info["filings"] = {"status": filings_reason, "graph": GRAPH_NOT_AVAILABLE}
        if companies_store is not None:
            try:
                with anyio.fail_after(settings.companies_query_timeout_s):
                    fresh = await anyio.to_thread.run_sync(partial(companies_store.freshness),
                                                           abandon_on_cancel=True)
                info["companies_freshness"] = jsonable(dict_to_utc(fresh))
            except Exception as exc:  # noqa: BLE001 -- an outage (or a timeout) must not hide the rest
                log.warning("server_info: companies freshness read failed (%s)", type(exc).__name__)
                info["companies_freshness_error"] = BACKEND_UNAVAILABLE
        return info

    # HTTP only: an unauthenticated liveness probe for load balancers and the image HEALTHCHECK.
    # It answers "ok" and nothing else (no version, no data); /mcp keeps requiring a token.
    @root.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return root
