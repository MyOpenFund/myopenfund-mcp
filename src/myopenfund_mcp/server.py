"""Assemble the server: core + every toolset whose stack is installed and configured."""
from __future__ import annotations

from functools import partial

import anyio
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token

from . import __version__
from .audit import AuditMiddleware
from .auth import HashedTokenVerifier, caller_tier, parse_tokens
from .config import Settings
from .paging import jsonable

INSTRUCTIONS = (
    "MyOpenFund: official primary sources only (central banks today; company filings next). "
    "Call server_info first to see which toolsets this deployment enables. Results carry doc_id, "
    "dates and source URLs: cite them. Errors start with a stable [code]."
)


def build_server(settings: Settings, *, vault_reader=None) -> FastMCP:
    auth_enabled = settings.transport == "http"
    auth = HashedTokenVerifier(parse_tokens(settings.tokens)) if auth_enabled else None
    root = FastMCP("myopenfund", instructions=INSTRUCTIONS, version=__version__, auth=auth,
                   middleware=[AuditMiddleware()])

    if vault_reader is None and settings.vault_database_url:
        from .vault.reader import PostgresVaultReader

        vault_reader = PostgresVaultReader(settings.vault_database_url,
                                           query_timeout_s=settings.query_timeout_s)
    if vault_reader is not None:
        from .vault.tools import build_vault_toolset, register_vault_resources

        root.mount(build_vault_toolset(vault_reader, settings, auth_enabled=auth_enabled), namespace="vault")
        register_vault_resources(root, vault_reader, settings, auth_enabled=auth_enabled)

    # Open to any authenticated caller (no tier check): it is how a client learns its tier.
    @root.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
    async def server_info() -> dict:
        """What this deployment offers: enabled toolsets, your tier, version, data freshness."""
        freshness = None
        if vault_reader is not None:
            freshness = jsonable(await anyio.to_thread.run_sync(partial(vault_reader.freshness)))
        return {
            "version": __version__,
            "tier": caller_tier(get_access_token(), auth_enabled=auth_enabled),
            "toolsets": {"vault": "enabled" if vault_reader is not None else "not configured",
                         "rag": "not configured", "companies": "not configured"},
            "freshness": freshness,
        }

    return root
