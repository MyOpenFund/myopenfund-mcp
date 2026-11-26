"""Opt-in: the same companies read from a real store and from SEC live give the same values (spec §8, G2).

    MOF_MCP_IT_COMPANIES_CATALOG=<catalogue connection string> \
    MOF_MCP_IT_COMPANIES_DATA_ROOT=<where this host sees the store files> \
    MOF_SEC_CONTACT="Your Name your.email@example.com" \
        python -m pytest -m integration -s tests/integration/test_store_vs_live.py

Two servers over the same tools: one in `store` mode, one in `sec` mode (live). For each company, the
`companies_fundamentals` rows (concepts and every metric) as known at the same `as_of` are compared by
(kind, key, period end, frequency, ttm), a key held by two rows of one side being an error. A
difference is a row whose value, `original_tag` or `filing_ref` differs, or that one side only has;
every difference is printed with both sides' tag, location and filing. Criterion: at least
MOF_MCP_IT_PARITY_MIN_SHARE of the rows equal (default 0.99, the spec's parity threshold), every
difference listed.

Optional: MOF_MCP_IT_PARITY_COMPANIES (comma-separated, default the scenario's five companies),
MOF_MCP_IT_PARITY_AS_OF (default 2025-06-30T00:00:00Z: a date the store surely covers),
MOF_MCP_IT_PARITY_FROM (period ends on or after, default 2022-01-01), MOF_MCP_IT_PARITY_MIN_SHARE,
MOF_MCP_IT_COMPANIES_TIMEOUT_S (per store call, default 120), MOF_MCP_IT_CACHE_DIR (absolute; the live
side's cache, default a temporary folder). The catalogue string and the contact are never printed.
"""
import asyncio
import os
import time

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

CATALOG = os.environ.get("MOF_MCP_IT_COMPANIES_CATALOG")
DATA_ROOT = os.environ.get("MOF_MCP_IT_COMPANIES_DATA_ROOT")
CONTACT = os.environ.get("MOF_SEC_CONTACT")

pytestmark = [pytest.mark.integration, pytest.mark.anyio,
              pytest.mark.skipif(not (CATALOG and DATA_ROOT and CONTACT),
                                 reason="MOF_MCP_IT_COMPANIES_CATALOG / _DATA_ROOT or MOF_SEC_CONTACT not set")]

COMPANIES = [c.strip() for c in (os.environ.get("MOF_MCP_IT_PARITY_COMPANIES") or "MSFT,AMZN,GOOGL,META,ORCL")
             .split(",") if c.strip()]
AS_OF = os.environ.get("MOF_MCP_IT_PARITY_AS_OF") or "2025-06-30T00:00:00Z"
PERIOD_FROM = os.environ.get("MOF_MCP_IT_PARITY_FROM") or "2022-01-01"


def _min_share() -> float:
    raw = os.environ.get("MOF_MCP_IT_PARITY_MIN_SHARE") or "0.99"
    try:
        value = float(raw)
    except ValueError:
        pytest.fail(f"MOF_MCP_IT_PARITY_MIN_SHARE must be a number between 0 and 1, got {raw!r}")
    if not 0 < value <= 1:
        pytest.fail(f"MOF_MCP_IT_PARITY_MIN_SHARE must be in (0, 1], got {value}")
    return value


def _settings(source: str, cache_dir: str) -> Settings:
    env = {**os.environ, "MOF_COMPANIES_SOURCE": source, "MOF_CACHE_DIR": cache_dir,
           "MOF_COMPANIES_QUERY_TIMEOUT_S": os.environ.get("MOF_MCP_IT_COMPANIES_TIMEOUT_S") or "120"}
    if source == "store":
        env.update(MOF_COMPANIES_CATALOG=CATALOG, MOF_COMPANIES_DATA_ROOT=DATA_ROOT)
        env.pop("MOF_SEC_CONTACT", None)  # store mode: SEC is never asked
    else:
        env.pop("MOF_COMPANIES_CATALOG", None)
        env.pop("MOF_COMPANIES_DATA_ROOT", None)
    return Settings.from_env(env)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> str:
    given = os.environ.get("MOF_MCP_IT_CACHE_DIR")
    if given:
        if not os.path.isabs(given):
            pytest.fail("MOF_MCP_IT_CACHE_DIR must be an absolute path")
        return given
    return str(tmp_path_factory.mktemp("mof-cache"))


async def _rows(client: Client, company: str) -> list[dict]:
    rows, cursor, deadline = [], None, time.monotonic() + 900
    while True:
        args = {"companies": [company], "metrics": "all", "period_from": PERIOD_FROM, "as_of": AS_OF, "limit": 200}
        if cursor:
            args["cursor"] = cursor
        try:
            page = (await client.call_tool("companies_fundamentals", args)).data
        except ToolError as exc:  # live: SEC still being read; ask again as a model would
            if str(exc).startswith(("[warming]", "[timeout]")) and time.monotonic() < deadline:
                await asyncio.sleep(2)
                continue
            raise
        rows += page["rows"]
        cursor = page["next_cursor"]
        if not cursor:
            return rows


def _key(r: dict) -> tuple:
    return r["kind"], r["key"], str(r["period_end"])[:10], r["frequency"], bool(r.get("ttm"))


def _side(r: dict | None) -> str:
    if r is None:
        return "absent"
    return (f"{r['value']} ({r.get('original_tag') or 'metric'}, {r.get('location') or '-'}, "
            f"{r.get('filing_ref') or '-'})")


def _by_key(company: str, side: str, rows: list[dict]) -> dict[tuple, dict]:
    """Rows by `_key`; a key seen twice would hide a row from the comparison: refused."""
    out: dict[tuple, dict] = {}
    for r in rows:
        k = _key(r)
        assert k not in out, f"{company} {side}: two rows for {k}: {_side(out[k])} and {_side(r)}"
        out[k] = r
    return out


def _same(a: dict, b: dict) -> bool:
    """Equal value, from the same tag and the same filing (a metric has neither: value only)."""
    return all(a.get(f) == b.get(f) for f in ("value", "original_tag", "filing_ref"))


async def test_store_and_live_agree(cache_dir):
    compared = equal = 0
    differences: list[str] = []
    async with (Client(build_server(_settings("store", cache_dir))) as store,
                Client(build_server(_settings("sec", cache_dir))) as live):
        for company in COMPANIES:
            s_rows, l_rows = await _rows(store, company), await _rows(live, company)
            assert {r["source"] for r in s_rows} <= {"store"} and {r["source"] for r in l_rows} <= {"sec_live"}
            assert s_rows, f"{company}: the store has no row as of {AS_OF}"
            s_by, l_by = _by_key(company, "store", s_rows), _by_key(company, "live", l_rows)
            n_eq = 0
            for k in sorted(set(s_by) | set(l_by)):
                a, b = s_by.get(k), l_by.get(k)
                if a is not None and b is not None and _same(a, b):
                    n_eq += 1
                else:
                    differences.append(f"{company} {k}: store {_side(a)} | live {_side(b)}")
            total = len(set(s_by) | set(l_by))
            print(f"{company}: {n_eq}/{total} rows equal")
            compared += total
            equal += n_eq
    print(f"Differences: {len(differences)}")
    for line in differences:
        print("-", line)
    assert compared
    share = equal / compared
    print(f"equal {equal}/{compared} = {share:.2%} (criterion >= {_min_share():.0%})")
    assert share >= _min_share(), f"{share:.2%} of rows equal; differences listed above"
