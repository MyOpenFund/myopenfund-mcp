"""Opt-in: the whole read path against a real vault database.

Run as `vault_readonly` (the deployment's role):
    MOF_MCP_IT_DATABASE_URL=postgresql://vault_readonly:<password>@<host>:5432/documents \
        python -m pytest -m integration
Assertions hold on any non-empty vault: no count is assumed beyond "more than zero".
"""
import os

import pytest
from fastmcp import Client

from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

URL = os.environ.get("MOF_MCP_IT_DATABASE_URL")

pytestmark = [pytest.mark.integration, pytest.mark.anyio,
              pytest.mark.skipif(not URL, reason="MOF_MCP_IT_DATABASE_URL not set")]


async def test_real_vault_read_path():
    async with Client(build_server(Settings(vault_database_url=URL))) as c:
        info = (await c.call_tool("server_info", {})).data
        assert info["toolsets"]["vault"] == "enabled"
        live = info["freshness"]["live_documents"]
        assert live > 0

        page = (await c.call_tool("vault_search_documents", {"limit": 5})).data
        assert page["total"] == live
        assert page["returned"] == min(5, page["total"]) == len(page["documents"])
        if page["total"] > 5:
            assert page["next_cursor"]
            nxt = (await c.call_tool("vault_search_documents",
                                     {"limit": 5, "cursor": page["next_cursor"]})).data
            first_ids = {d["doc_id"] for d in page["documents"]}
            assert nxt["returned"] > 0 and first_ids.isdisjoint(d["doc_id"] for d in nxt["documents"])
        else:
            assert page["next_cursor"] is None

        doc_id = page["documents"][0]["doc_id"]
        doc = (await c.call_tool("vault_get_document", {"doc_id": doc_id})).data
        assert doc["doc_id"] == doc_id

        stats = (await c.call_tool("vault_corpus_stats", {"group_by": "source_code"})).data
        assert stats["total"] == live
        assert stats["groups"] and all(g["count"] > 0 for g in stats["groups"])

        runs = (await c.call_tool("vault_runs_status", {"limit": 5})).data
        assert set(runs) == {"runs", "source_runs", "open_discovery_errors", "source_health"}
        assert all(isinstance(v, list) for v in runs.values())
        assert all(len(runs[k]) <= 5 for k in ("runs", "source_runs", "open_discovery_errors"))


def test_connection_is_read_only():
    """The server-side guard (default_transaction_read_only) holds whatever the role may do."""
    import psycopg2.errors

    from myopenfund_mcp.vault.reader import PostgresVaultReader

    reader = PostgresVaultReader(URL, query_timeout_s=10)
    with (reader._connection() as conn, conn.cursor() as cur,
          pytest.raises(psycopg2.errors.ReadOnlySqlTransaction)):
        # Matches no row: harmless even if the guard were missing and the role could write.
        cur.execute("UPDATE documents SET title = title WHERE false")
