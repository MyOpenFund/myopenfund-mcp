"""Opt-in: the companies and datasets read path against a real my-open-data store.

    MOF_MCP_IT_COMPANIES_CATALOG=<catalogue connection string> \
    MOF_MCP_IT_COMPANIES_DATA_ROOT=<where this host sees the store files> \
        python -m pytest -m integration tests/integration/test_real_store.py

Optional: MOF_MCP_IT_COMPANIES_TIMEOUT_S (per store call, default 120: over SMB one read can take
minutes) and MOF_MCP_IT_ASSERT_LATENCY=1 (enforce the < 5 s fundamentals criterion; meaningful on the NAS only).
The catalogue string is never printed or put in a skip reason.
"""
import os
import time

import pytest
from fastmcp import Client

from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

CATALOG = os.environ.get("MOF_MCP_IT_COMPANIES_CATALOG")
DATA_ROOT = os.environ.get("MOF_MCP_IT_COMPANIES_DATA_ROOT")
TIMEOUT_S = int(os.environ.get("MOF_MCP_IT_COMPANIES_TIMEOUT_S") or 120)

pytestmark = [pytest.mark.integration, pytest.mark.anyio,
              pytest.mark.skipif(not (CATALOG and DATA_ROOT),
                                 reason="MOF_MCP_IT_COMPANIES_CATALOG / _DATA_ROOT not set")]

APPLE_CIK = "0000320193"
REVENUE_TAG = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"


def _client() -> Client:
    return Client(build_server(Settings(companies_catalog=CATALOG, companies_data_root=DATA_ROOT,
                                        companies_query_timeout_s=TIMEOUT_S)))


async def test_apple_fy2024_revenue_as_known_in_december_2024():
    async with _client() as c:
        res = (await c.call_tool("companies_facts", {
            "companies": ["AAPL"], "tags": [REVENUE_TAG], "frequency": "annual",
            "period_from": "2024-09-28", "period_to": "2024-09-28",
            "as_of": "2024-12-01T00:00:00Z"})).data
    assert [r["value"] for r in res["rows"]] == [391035000000]


async def test_fundamentals_one_company():
    async with _client() as c:
        await c.call_tool("server_info", {})  # warm-up: not part of the measured call
        t0 = time.monotonic()
        res = (await c.call_tool("companies_fundamentals", {"companies": ["AAPL"], "metrics": "all"})).data
        elapsed = time.monotonic() - t0
        # the revenue row is looked up in its own period: the page above is newest-first and may not reach it
        fy = (await c.call_tool("companies_fundamentals", {
            "companies": ["AAPL"], "concepts": ["revenue"], "period_from": "2024-09-28",
            "period_to": "2024-09-28"})).data
    print(f"companies_fundamentals AAPL metrics=all: {elapsed:.1f} s")
    assert res["total"] > 0
    assert any(r["kind"] == "metric" for r in res["rows"])
    revenue = [r for r in fy["rows"] if r["key"] == "revenue" and str(r["period_end"])[:10] == "2024-09-28"]
    assert revenue and revenue[0]["value"] == 391035000000 and revenue[0]["filing_ref"]
    if os.environ.get("MOF_MCP_IT_ASSERT_LATENCY") == "1":  # only on the NAS (spec: < 5 s there)
        assert elapsed < 5, f"{elapsed:.1f} s"


async def test_server_info_reports_the_store():
    async with _client() as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["toolsets"]["companies"] == "enabled" and info["toolsets"]["datasets"] == "enabled"
    assert "companies_freshness_error" not in info
    assert info["companies_freshness"]["batches"] > 0


async def test_resolve_apple_by_name():
    async with _client() as c:
        res = (await c.call_tool("companies_resolve", {"query": "apple"})).data
    assert len(res["candidates"]) > 1
    assert APPLE_CIK in {h["cik"] for h in res["candidates"]}
