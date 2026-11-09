import time

import pytest
from companies_fixture import AAPL, AAPL_10K_2025, JPM, NWG, REV, seed
from fastmcp import Client
from fastmcp.exceptions import ToolError

from myopenfund_mcp.companies.store import CompaniesStore
from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def store(tmp_path):
    cfg = seed(tmp_path)
    s = CompaniesStore(str(cfg.data_root), cfg.catalog)
    yield s
    s.close()


def _client(store, **kw):
    return Client(build_server(Settings(**kw), companies_store=store))


async def _call(store, tool, args, **kw):
    async with _client(store, **kw) as c:
        return (await c.call_tool(tool, args)).data


# --- resolve ------------------------------------------------------------------------------------

async def test_resolve_by_ticker_name_and_ambiguity(store):
    async with _client(store) as c:
        r = (await c.call_tool("companies_resolve", {"query": "AAPL"})).data
        both = (await c.call_tool("companies_resolve", {"query": "apple"})).data
    assert [x["cik"] for x in r["candidates"]] == [AAPL]
    assert len(both["candidates"]) == 2 and both["total"] == 2
    apple = r["candidates"][0]
    assert [n["name"] for n in apple["names"]] == ["APPLE COMPUTER INC", "Apple Inc."]
    assert apple["names"][0]["since"] == "1994-01-26" and apple["tickers"] == ["AAPL"]


async def test_resolve_over_padded_cik_and_repeated_ticker_name(store):
    r = await _call(store, "companies_resolve", {"query": "000000320193"})
    assert [x["cik"] for x in r["candidates"]] == [AAPL]
    jpm = (await _call(store, "companies_resolve", {"query": "jpm"}))["candidates"][0]
    assert [n["name"] for n in jpm["names"]] == ["JPMORGAN CHASE & CO"]  # the ticker row repeats it


async def test_resolve_limit(store):
    r = await _call(store, "companies_resolve", {"query": "apple", "limit": 1})
    assert len(r["candidates"]) == 1 and r["total"] == 2


# --- facts --------------------------------------------------------------------------------------

async def test_facts_point_in_time(store):
    async with _client(store) as c:
        before = (await c.call_tool("companies_facts", {"companies": ["AAPL"], "tags": [REV],
                                                         "as_of": "2025-01-01T00:00:00Z"})).data
        after = (await c.call_tool("companies_facts", {"companies": ["AAPL"], "tags": [REV]})).data
    assert [r["value"] for r in before["rows"]] == [391035000000]
    assert [r["value"] for r in after["rows"]][-1] == 391036000000  # FY2024 restated, after FY2025
    assert after["rows"][0]["name"] == "Apple Inc."
    assert before["as_of"] == "2025-01-01T00:00:00+00:00"


async def test_facts_rows_shape_sort_and_utc(store):
    res = await _call(store, "companies_facts", {"companies": [JPM, "AAPL"]})
    rows = res["rows"]
    assert all("batch_id" not in r and r["cik"] == r["entity_value"] for r in rows)
    assert all(r["known_at"].endswith("+00:00") for r in rows)
    keys = [(r["cik"], r["valid_date"]) for r in rows]
    assert [k[0] for k in keys] == sorted(k[0] for k in keys)  # by cik first
    aapl = [r["valid_date"] for r in rows if r["cik"] == AAPL]
    assert aapl == sorted(aapl, reverse=True)  # newest period first
    assert res["total"] == len(rows) == 8 and res["returned"] == 8


async def test_facts_frequency_period_and_paging(store):
    q = await _call(store, "companies_facts", {"companies": [JPM], "frequency": "quarterly"})
    assert [r["valid_date"] for r in q["rows"]] == ["2025-03-31"]
    p = await _call(store, "companies_facts", {"companies": [AAPL], "period_from": "2025-01-01"})
    assert {r["valid_date"] for r in p["rows"]} == {"2025-09-27"}
    async with _client(store) as c:
        one = (await c.call_tool("companies_facts", {"companies": [AAPL], "limit": 3})).data
        two = (await c.call_tool("companies_facts", {"companies": [AAPL], "limit": 3,
                                                      "cursor": one["next_cursor"]})).data
    assert one["returned"] == 3 and one["next_cursor"]
    assert two["returned"] == 2 and two["next_cursor"] is None


async def test_facts_same_company_twice_is_read_once(store):
    res = await _call(store, "companies_facts", {"companies": [AAPL, "aapl", "320193"]})
    assert res["total"] == 5


# --- fundamentals -------------------------------------------------------------------------------

async def test_fundamentals_concepts_and_metrics(store):
    res = await _call(store, "companies_fundamentals", {
        "companies": [AAPL], "concepts": ["revenue", "net_income"], "metrics": ["net_margin", "net_margin_ttm"],
        "frequency": "annual"})
    kinds = {(r["kind"], r["key"]) for r in res["rows"]}
    assert kinds == {("concept", "revenue"), ("concept", "net_income"), ("metric", "net_margin"),
                     ("metric", "net_margin_ttm")}
    rev = next(r for r in res["rows"] if r["key"] == "revenue" and r["period_end"] == "2024-09-28")
    assert rev["filing_ref"] == AAPL_10K_2025 and rev["original_tag"] == REV
    assert rev["value"] == 391036000000 and rev["label"] == "Revenue" and rev["unit"] == "USD"
    assert rev["known_at"] == "2025-10-31T20:00:00+00:00"
    assert set(rev) == {"cik", "name", "period_end", "frequency", "kind", "key", "label", "value", "unit",
                        "original_tag", "filing_ref", "source_doc_id", "known_at", "location", "conflict"}
    nm = next(r for r in res["rows"] if r["key"] == "net_margin")
    assert set(nm) == {"cik", "name", "period_end", "frequency", "kind", "key", "label", "value", "unit",
                       "ttm", "sector_relevant"}
    assert nm["ttm"] is False and nm["unit"] == "%" and abs(nm["value"] - 23.97) < 0.01
    assert next(r for r in res["rows"] if r["key"] == "net_margin_ttm")["ttm"] is True
    assert res["sector"].startswith("unknown") and res["dictionary_version"] == "1.0.0"


async def test_fundamentals_sort_concepts_before_metrics_newest_first(store):
    res = await _call(store, "companies_fundamentals", {"companies": [JPM, AAPL], "metrics": "all"})
    order = [(r["cik"], r["period_end"], r["frequency"], r["kind"], r["key"]) for r in res["rows"]]
    expected = sorted(order, key=lambda k: (k[0], [-ord(ch) for ch in k[1]], k[2], k[3] != "concept", k[4]))
    assert order == expected
    assert order[0][0] == JPM and order[-1][0] == AAPL  # 0000019617 < 0000320193


async def test_fundamentals_metrics_all_as_string_or_list(store):
    a = await _call(store, "companies_fundamentals", {"companies": [AAPL], "metrics": "all"})
    b = await _call(store, "companies_fundamentals", {"companies": [AAPL], "metrics": ["all"]})
    keys = {r["key"] for r in a["rows"] if r["kind"] == "metric"}
    assert keys == {r["key"] for r in b["rows"] if r["kind"] == "metric"}
    assert {"net_margin", "roe", "roa", "net_margin_ttm"} <= keys


async def test_fundamentals_point_in_time_and_period_range(store):
    res = await _call(store, "companies_fundamentals", {
        "companies": [AAPL], "concepts": ["revenue"], "as_of": "2025-01-01T00:00:00Z"})
    assert [(r["period_end"], r["value"]) for r in res["rows"]] == [("2024-09-28", 391035000000)]
    rng = await _call(store, "companies_fundamentals", {
        "companies": [AAPL], "metrics": "all", "period_to": "2024-12-31"})
    assert {r["period_end"] for r in rng["rows"]} == {"2024-09-28"}


async def test_fundamentals_without_metrics_returns_concepts_only(store):
    res = await _call(store, "companies_fundamentals", {"companies": [AAPL]})
    assert {r["kind"] for r in res["rows"]} == {"concept"}


async def test_ifrs_only_issuer_gives_empty_rows_not_an_error(store):
    res = await _call(store, "companies_fundamentals", {"companies": ["NWG"], "metrics": "all"})
    assert res["rows"] == [] and res["dictionary_version"] is None


@pytest.mark.parametrize("args, code", [
    ({"companies": ["apple"]}, "ambiguous_company"),
    ({"companies": ["nothing-like-this"]}, "unknown_company"),
    ({"companies": [AAPL], "concepts": ["revenu"]}, "unknown_concept"),
    ({"companies": [AAPL], "metrics": ["roe_ttm2"]}, "unknown_metric"),
    ({"companies": [str(i) for i in range(11)]}, "too_many_companies"),
    ({"companies": [AAPL], "period_from": "2025-01-01", "period_to": "2024-01-01"}, "invalid_argument"),
    ({"companies": [AAPL], "limit": 0}, "invalid_argument"),
    ({"companies": [AAPL], "cursor": "junk"}, "invalid_argument"),
])
async def test_errors_are_coded(store, args, code):
    async with _client(store) as c:
        with pytest.raises(ToolError, match=rf"^\[{code}\]"):
            await c.call_tool("companies_fundamentals", args)


@pytest.mark.parametrize("args", [
    {"companies": [AAPL], "concepts": ["revenu"]},
    {"companies": [AAPL], "metrics": ["nope"]},
    {"companies": [str(i) for i in range(11)]},
    {"companies": [AAPL], "period_from": "2025-01-01", "period_to": "2024-01-01"},
    {"companies": [AAPL], "limit": 0},
    {"companies": [AAPL], "cursor": "junk"},
])
async def test_bad_input_is_refused_before_the_store_is_touched(store, args):
    calls = []
    real = store.entities
    store.entities = lambda: calls.append(1) or real()
    async with _client(store) as c:
        with pytest.raises(ToolError):
            await c.call_tool("companies_fundamentals", args)
    assert calls == []


async def test_facts_validates_before_the_store_too(store):
    calls = []
    store.entities = lambda: calls.append(1) or []
    async with _client(store) as c:
        for args in ({"companies": [str(i) for i in range(11)]},
                     {"companies": [AAPL], "period_from": "2025-01-01", "period_to": "2024-01-01"}):
            with pytest.raises(ToolError, match=r"^\[(too_many_companies|invalid_argument)\]"):
                await c.call_tool("companies_facts", args)
    assert calls == []


async def test_ambiguity_lists_candidates_with_ciks(store):
    async with _client(store) as c:
        with pytest.raises(ToolError, match=AAPL):
            await c.call_tool("companies_facts", {"companies": ["apple"]})


async def test_unknown_concept_lists_valid_ones(store):
    async with _client(store) as c:
        with pytest.raises(ToolError, match="revenue"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL], "concepts": ["revenu"]})


async def test_timeout_is_coded(store):
    store.concept_rows = lambda cik, as_of: time.sleep(3)
    async with _client(store, companies_query_timeout_s=1) as c:
        with pytest.raises(ToolError, match=r"^\[timeout\].*narrow the request"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL]})


async def test_one_time_budget_for_the_whole_call(store):
    """Each read alone fits the budget; three of them do not: the call times out within ~1 budget."""
    real = store.concept_rows

    def slow(cik, as_of):
        time.sleep(0.6)
        return real(cik, as_of)

    store.concept_rows = slow
    async with _client(store, companies_query_timeout_s=1) as c:
        t0 = time.monotonic()
        with pytest.raises(ToolError, match=r"^\[timeout\]"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL, JPM, NWG]})
        assert time.monotonic() - t0 < 1.6


async def test_facts_shares_one_budget_with_the_lookup(store):
    real_entities, real_read = store.entities, store.read

    def slow_entities():
        time.sleep(0.6)
        return real_entities()

    def slow_read(*a, **kw):
        time.sleep(0.6)
        return real_read(*a, **kw)

    store.entities, store.read = slow_entities, slow_read
    async with _client(store, companies_query_timeout_s=1) as c:
        t0 = time.monotonic()
        with pytest.raises(ToolError, match=r"^\[timeout\]"):
            await c.call_tool("companies_facts", {"companies": [AAPL]})
        assert time.monotonic() - t0 < 1.6


async def test_descriptions_are_directive(store):
    async with _client(store) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "companies_resolve" in tools["companies_fundamentals"].description
    assert "companies_resolve" in tools["companies_facts"].description
    assert "as_of" in tools["companies_facts"].description
    assert "companies_fundamentals" in tools["companies_facts"].description
