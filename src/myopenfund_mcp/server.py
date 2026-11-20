"""Assemble the server: core + every toolset whose stack is installed and configured."""
from __future__ import annotations

import logging
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
from .paging import jsonable

log = logging.getLogger("myopenfund_mcp.server")

MISSING_VAULT_EXTRA = ("MOF_VAULT_DATABASE_URL is set but the [vault] extra is not installed: "
                       "pip install 'myopenfund-mcp[vault]'")
MISSING_COMPANIES_EXTRA = ("MOF_COMPANIES_CATALOG is set but the [companies] extra is not installed: "
                           "pip install 'myopenfund-mcp[companies]'")

MISSING_RAG_EXTRA = ("MOF_RAG_QDRANT_HOST is set but the [rag] extra is not installed: "
                     "pip install 'myopenfund-mcp[rag]'")

RAG_IMPORT_FAILED = ("MOF_RAG_QDRANT_HOST is set but the rag toolset could not be imported, although the "
                     "[rag] extra seems installed: see the error logged just before this one")

INSTRUCTIONS = (
    "MyOpenFund: official primary sources only (central banks and SEC company filings). "
    "Call server_info first to see which toolsets this deployment enables. Results carry doc_id, "
    "dates and source URLs: cite them. Company data is point-in-time: pass as_of to see what was "
    "known then. Errors start with a stable [code]."
)


def build_server(settings: Settings, *, vault_reader=None, companies_store=None, rag_backend=None) -> FastMCP:
    auth_enabled = settings.transport == "http"
    if auth_enabled and not settings.tokens:
        raise ConfigError(
            "MOF_MCP_TRANSPORT=http requires MOF_MCP_TOKENS: the HTTP server never runs unauthenticated"
        )
    auth = HashedTokenVerifier(parse_tokens(settings.tokens)) if auth_enabled else None
    # mask_error_details: an unexpected exception (driver error, DSN in a message) reaches the
    # client as a generic error; our ToolError / ResourceError keep their `[code] message`.
    root = FastMCP("myopenfund", instructions=INSTRUCTIONS, version=__version__, auth=auth,
                   middleware=[AuditMiddleware()], mask_error_details=True)

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
    if companies_store is not None:
        from .companies.common import dict_to_utc
        from .companies.datasets import build_datasets_toolset
        from .companies.tools import build_companies_toolset

        root.mount(build_datasets_toolset(companies_store, settings, auth_enabled=auth_enabled),
                   namespace="datasets")
        root.mount(build_companies_toolset(companies_store, settings, auth_enabled=auth_enabled),
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

    companies_status = "enabled" if companies_store is not None else "not configured"

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
        """
        info = {
            "version": __version__,
            "tier": caller_tier(get_access_token(), auth_enabled=auth_enabled),
            "toolsets": {"vault": "enabled" if vault_reader is not None else "not configured",
                         "rag": "enabled" if rag_backend is not None else "not configured",
                         "companies": companies_status, "datasets": companies_status},
            "freshness": None,
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
