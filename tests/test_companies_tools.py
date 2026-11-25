import threading
import time

import anyio
import pytest
from companies_fixture import AAPL, AAPL_10K_2025, JPM, NWG, REV, seed
from fastmcp import Client
from fastmcp.exceptions import ToolError

from myopenfund_mcp.companies.common import parse_as_of
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


def _tracked(fn):
    """``fn`` wrapped so a test can wait for the calls its tool abandoned on timeout: they keep
    running in their threads, and the fixture's teardown must not race them."""
    running, idle = [0], threading.Condition()

    def wrapper(*a, **kw):
        with idle:
            running[0] += 1
        try:
            return fn(*a, **kw)
        finally:
            with idle:
                running[0] -= 1
                idle.notify_all()

    async def settled():
        def wait():
            with idle:
                assert idle.wait_for(lambda: running[0] == 0, timeout=10)
        await anyio.to_thread.run_sync(wait)

    return wrapper, settled


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
    assert res["total"] == len(rows) == 10 and res["returned"] == 10


async def test_facts_frequency_period_and_paging(store):
    # facts stay raw as filed: frequency is the raw column, no instants added
    q = await _call(store, "companies_facts", {"companies": [JPM], "frequency": "quarterly"})
    assert [(r["valid_date"], r["frequency"]) for r in q["rows"]] == [("2025-03-31", "quarterly")]
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
                        "original_tag", "filing_ref", "source_doc_id", "known_at", "location", "conflict",
                        "source", "reason"}
    nm = next(r for r in res["rows"] if r["key"] == "net_margin")
    assert set(nm) == {"cik", "name", "period_end", "frequency", "kind", "key", "label", "value", "unit",
                       "ttm", "sector_relevant", "source", "reason"}
    # no contact: the store answers, its freshness unchecked (R27)
    assert (rev["source"], rev["reason"]) == (nm["source"], nm["reason"]) == ("store", "freshness_unchecked")
    assert nm["ttm"] is False and nm["unit"] == "%" and abs(nm["value"] - 23.97) < 0.01
    assert next(r for r in res["rows"] if r["key"] == "net_margin_ttm")["ttm"] is True
    # the dictionary version company-corpus 06c619d ships
    assert res["sector"].startswith("unknown") and res["dictionary_version"] == "2.1.0"


async def test_fundamentals_sort_concepts_before_metrics_newest_first(store):
    res = await _call(store, "companies_fundamentals", {"companies": [JPM, AAPL], "metrics": "all"})
    order = [(r["cik"], r["period_end"], r["frequency"], r["kind"], r["key"]) for r in res["rows"]]
    expected = sorted(order, key=lambda k: (k[0], [-ord(ch) for ch in k[1]], k[2], k[3] != "concept", k[4]))
    assert order == expected
    assert order[0][0] == JPM and order[-1][0] == AAPL  # 0000019617 < 0000320193


async def test_fundamentals_annual_includes_balance_sheet_at_annual_period_ends(store):
    res = await _call(store, "companies_fundamentals", {"companies": [AAPL, JPM], "frequency": "annual"})
    inst = {(r["cik"], r["key"], r["period_end"]) for r in res["rows"] if r["frequency"] == "instant"}
    assert inst == {(AAPL, "assets", "2024-09-28"), (AAPL, "equity", "2024-09-28")}  # not JPM's Q1 assets
    assert {r["frequency"] for r in res["rows"]} == {"annual", "instant"}
    # the instants sort with their period: cik, period_end desc, then frequency
    aapl = [(r["period_end"], r["frequency"]) for r in res["rows"] if r["cik"] == AAPL]
    assert aapl == sorted(aapl, key=lambda k: ([-ord(c) for c in k[0]], k[1]))


async def test_fundamentals_quarterly_includes_balance_sheet_at_quarter_ends(store):
    res = await _call(store, "companies_fundamentals", {"companies": [JPM], "frequency": "quarterly"})
    assert {(r["frequency"], r["key"], r["period_end"]) for r in res["rows"]} == {
        ("quarterly", "revenue", "2025-03-31"), ("instant", "assets", "2025-03-31")}


async def test_fundamentals_balance_sheet_follows_the_period_range_and_concepts(store):
    res = await _call(store, "companies_fundamentals", {
        "companies": [AAPL], "frequency": "annual", "concepts": ["assets"], "period_from": "2025-01-01"})
    assert res["rows"] == []  # FY2025 has no balance sheet in the store; FY2024 is out of range
    res = await _call(store, "companies_fundamentals", {
        "companies": [AAPL], "frequency": "annual", "concepts": ["assets"]})
    assert [(r["key"], r["period_end"]) for r in res["rows"]] == [("assets", "2024-09-28")]


async def test_fundamentals_instant_returns_instants_only(store):
    res = await _call(store, "companies_fundamentals", {"companies": [AAPL, JPM], "frequency": "instant",
                                                        "metrics": "all"})
    assert {r["frequency"] for r in res["rows"]} == {"instant"}
    assert {(r["cik"], r["key"], r["period_end"]) for r in res["rows"]} == {
        (AAPL, "assets", "2024-09-28"), (AAPL, "equity", "2024-09-28"), (JPM, "assets", "2025-03-31")}


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
    ({"companies": [AAPL], "metrics": ["all", "roe"]}, "invalid_argument"),
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
    {"companies": [AAPL], "metrics": ["all", "roe"]},
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


async def test_all_cannot_be_combined_with_other_metrics(store):
    async with _client(store) as c:
        with pytest.raises(ToolError, match=r"'all' cannot be combined with other metrics"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL], "metrics": ["all", "roe"]})


@pytest.mark.parametrize("tool", ["companies_facts", "companies_fundamentals"])
async def test_company_identifiers_are_length_bounded(store, tool):
    calls = []
    store.entities = lambda: calls.append(1) or []
    async with _client(store) as c:
        with pytest.raises(ToolError) as err:
            await c.call_tool(tool, {"companies": ["x" * 201]})
    assert calls == [] and not str(err.value).startswith("[unknown_company]")


async def test_ambiguity_lists_candidates_with_ciks(store):
    async with _client(store) as c:
        with pytest.raises(ToolError, match=AAPL):
            await c.call_tool("companies_facts", {"companies": ["apple"]})


async def test_unknown_concept_lists_valid_ones(store):
    async with _client(store) as c:
        with pytest.raises(ToolError, match="revenue"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL], "concepts": ["revenu"]})


async def test_timeout_is_coded(store):
    store.concept_rows, settled = _tracked(lambda cik, as_of: time.sleep(1.5))
    async with _client(store, companies_query_timeout_s=1) as c:
        with pytest.raises(ToolError, match=r"^\[timeout\].*narrow the request"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL]})
    await settled()


async def test_one_time_budget_for_the_whole_call(store):
    """Each read alone fits the budget; three of them do not: the call times out within ~1 budget."""
    real = store.concept_rows

    def slow(cik, as_of):
        time.sleep(0.6)
        return real(cik, as_of)

    store.concept_rows, settled = _tracked(slow)
    async with _client(store, companies_query_timeout_s=1) as c:
        t0 = time.monotonic()
        with pytest.raises(ToolError, match=r"^\[timeout\]"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL, JPM, NWG]})
        assert time.monotonic() - t0 < 1.7
    await settled()


async def test_facts_shares_one_budget_with_the_lookup(store):
    real_entities, real_read = store.entities, store.read

    # 0.8 s each: one shared 1 s budget times out at ~1.0 s; a budget per call would not time out
    # at all (each fits), and the call would end at ~1.6 s. The window between is wide on purpose.
    def slow_entities():
        time.sleep(0.8)
        return real_entities()

    def slow_read(*a, **kw):
        time.sleep(0.8)
        return real_read(*a, **kw)

    store.entities = slow_entities
    store.read, settled = _tracked(slow_read)   # the abandoned read is still running after the call
    async with _client(store, companies_query_timeout_s=1) as c:
        t0 = time.monotonic()
        with pytest.raises(ToolError, match=r"^\[timeout\]"):
            await c.call_tool("companies_facts", {"companies": [AAPL]})
        assert time.monotonic() - t0 < 1.5
    await settled()


async def test_descriptions_are_directive(store):
    async with _client(store) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "companies_resolve" in tools["companies_fundamentals"].description
    assert "companies_resolve" in tools["companies_facts"].description
    assert "as_of" in tools["companies_facts"].description
    assert "companies_fundamentals" in tools["companies_facts"].description


# --- an empty published store (contracts approved, nothing published) ---------------------------

@pytest.fixture
def empty_store(tmp_path):
    from open_data_store.config import StoreConfig
    from open_data_store.contracts.standard import ENTITY_OBSERVATIONS, FILING_STATEMENTS, FUNDAMENTALS
    from open_data_store.publisher import approve_contract

    cfg = StoreConfig(data_root=tmp_path / "store", catalog=str(tmp_path / "meta.ducklake"))
    for contract in (ENTITY_OBSERVATIONS, FUNDAMENTALS, FILING_STATEMENTS):
        approve_contract(cfg, contract, "test")
    s = CompaniesStore(str(cfg.data_root), cfg.catalog)
    yield s
    s.close()


async def test_an_empty_store_answers_without_errors(empty_store):
    async with _client(empty_store) as c:
        names = {d["name"] for d in (await c.call_tool("datasets_list", {})).data["datasets"]}
        resolved = (await c.call_tool("companies_resolve", {"query": "apple"})).data
        with pytest.raises(ToolError, match=r"^\[unknown_company\]"):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL]})
        info = (await c.call_tool("server_info", {})).data
    assert names == set()  # approved but unpublished datasets are not listed
    assert resolved["candidates"] == [] and resolved["total"] == 0
    assert "companies_freshness_error" not in info
    assert info["companies_freshness"]["batches"] == 0


async def test_an_unpublished_dataset_reads_as_empty(empty_store):
    async with _client(empty_store) as c:
        res = (await c.call_tool("datasets_read_asof", {"dataset": "fundamentals"})).data
    assert res["rows"] == [] and res["next_cursor"] is None


async def test_an_empty_store_gives_no_entities_and_no_concept_rows(empty_store):
    assert empty_store.entities() == []
    assert empty_store.concept_rows(AAPL, parse_as_of(None)) == ([], [])


def test_server_passes_the_configured_q4_rule_to_the_store(tmp_path, monkeypatch):
    import myopenfund_mcp.companies.store as store_mod

    made = []

    class Recording(store_mod.CompaniesStore):
        def __init__(self, *args, **kwargs):
            made.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(store_mod, "CompaniesStore", Recording)
    build_server(Settings(companies_catalog="x", companies_data_root=str(tmp_path),
                          companies_q4_rule="sum_of_quarters"))
    assert made and made[0]["q4_rule"] == "sum_of_quarters"


async def test_facts_return_the_nine_month_row_as_filed(store):
    # fundamentals 3.1.0 stores 9-month year-to-date facts as 'ytd9'; facts expose them raw
    res = await _call(store, "companies_facts", {"companies": [JPM], "frequency": "ytd9"})
    assert [(r["frequency"], r["valid_date"], r["original_tag"], r["value"]) for r in res["rows"]] == [
        ("ytd9", "2025-09-30", "us-gaap:Revenues", 135000000000)]


async def test_facts_without_frequency_include_the_nine_month_row(store):
    res = await _call(store, "companies_facts", {"companies": [JPM]})
    assert "ytd9" in {r["frequency"] for r in res["rows"]}


async def test_fundamentals_never_output_a_nine_month_row(store):
    for args in ({"companies": [JPM]}, {"companies": [JPM], "metrics": "all"},
                 {"companies": [JPM], "concepts": ["revenue"], "metrics": "all"}):
        res = await _call(store, "companies_fundamentals", args)
        assert res["rows"] and "ytd9" not in {r["frequency"] for r in res["rows"]}


async def test_fundamentals_have_no_nine_month_periods(store):
    # 9-month values only derive an unreported Q4 (as in company-corpus); asking for them is refused
    async with _client(store) as c:
        with pytest.raises(ToolError):
            await c.call_tool("companies_fundamentals", {"companies": [AAPL], "frequency": "ytd9"})


# --- source routing (fake store + fake live SEC source) -----------------------------------------

SENTINEL = "sentinel-contact-7f3a@example.invalid"


def _routed(store, live, **kw):
    kw.setdefault("sec_contact", SENTINEL if live is not None else None)
    return Client(build_server(Settings(**kw), companies_store=store, companies_live=live))


async def _routed_call(store, live, tool, args, **kw):
    async with _routed(store, live, **kw) as c:
        return (await c.call_tool(tool, args)).data


async def _routed_error(store, live, tool, args, **kw) -> str:
    async with _routed(store, live, **kw) as c:
        with pytest.raises(ToolError) as err:
            await c.call_tool(tool, args)
    return str(err.value)


async def test_mixed_call_tags_rows_per_company():
    from source_fakes import APPLE, MSFT, FakeLive, FakeStore

    store, live = FakeStore(), FakeLive()
    facts = await _routed_call(store, live, "companies_facts", {"companies": [APPLE, MSFT]})
    fund = await _routed_call(store, live, "companies_fundamentals", {"companies": ["AAPL", "MSFT"]})
    for res in (facts, fund):
        by_cik = {r["cik"]: r for r in res["rows"]}
        assert (by_cik[APPLE]["source"], by_cik[APPLE]["value"]) == ("store", 100)
        assert "reason" not in by_cik[APPLE]
        assert (by_cik[MSFT]["source"], by_cik[MSFT]["reason"], by_cik[MSFT]["value"]) == (
            "sec_live", "not_in_store", 200)
    assert live.calls.count("concept_rows") == 1  # Microsoft only: Apple came from the store


async def test_sec_mode_rows_carry_no_reason_and_the_store_is_not_read():
    from source_fakes import APPLE, FakeLive, FakeStore

    store = FakeStore()
    res = await _routed_call(store, FakeLive(), "companies_fundamentals", {"companies": [APPLE]},
                             companies_source="sec")
    assert [(r["source"], "reason" in r) for r in res["rows"]] == [("sec_live", False)]
    assert store.calls == []


async def test_auto_without_a_store_rows_say_no_store():
    from source_fakes import FakeLive

    res = await _routed_call(None, FakeLive(), "companies_facts", {"companies": ["MSFT"]})
    assert [(r["source"], r["reason"]) for r in res["rows"]] == [("sec_live", "no_store")]


async def test_a_store_failure_answers_live_with_store_unavailable():
    from source_fakes import APPLE, FakeLive, FakeStore

    store = FakeStore(fail={"read:fundamentals": RuntimeError("down")})
    res = await _routed_call(store, FakeLive(), "companies_facts", {"companies": [APPLE]})
    assert [(r["source"], r["reason"], r["value"]) for r in res["rows"]] == [
        ("sec_live", "store_unavailable", 200)]


async def test_without_a_store_datasets_are_not_mounted():
    from source_fakes import FakeLive

    async with _routed(None, FakeLive()) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert "companies_fundamentals" in names and not any(n.startswith("datasets_") for n in names)
    assert info["toolsets"]["companies"] == "enabled" and info["toolsets"]["datasets"] == "not configured"
    assert "companies_freshness" not in info


async def test_without_contact_and_store_the_companies_toolset_is_absent():
    async with Client(build_server(Settings())) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert not any(n.startswith(("companies_", "datasets_")) for n in names)
    assert info["toolsets"]["companies"] == "not configured" and info["sec_contact_configured"] is False


async def test_contact_missing_with_a_store_names_the_setting_and_store_companies_still_answer():
    from source_fakes import APPLE, MSFT, FakeStore

    store = FakeStore()
    # without a contact the resolver is the store's: a company outside it is not even found
    msg = await _routed_error(store, None, "companies_fundamentals", {"companies": [APPLE, MSFT]})
    assert msg.startswith("[unknown_company]") and "MOF_SEC_CONTACT" in msg and MSFT in msg
    ok = await _routed_call(store, None, "companies_fundamentals", {"companies": [APPLE]})
    assert [r["source"] for r in ok["rows"]] == ["store"]


async def test_warming_lists_progress_per_company_and_says_call_again():
    from source_fakes import APPLE, MSFT, FakeLive

    from myopenfund_mcp.companies.live import Warming

    live = FakeLive(fail={"concept_rows:" + APPLE: Warming(3, 12), "concept_rows:" + MSFT: Warming(1, 9)})
    msg = await _routed_error(None, live, "companies_fundamentals", {"companies": [APPLE, MSFT]})
    assert msg.startswith("[warming]") and "call again" in msg
    assert f"CIK {APPLE}): 3/12" in msg and f"CIK {MSFT}): 1/9" in msg


async def test_warming_for_one_company_only_is_still_an_error():
    from source_fakes import APPLE, MSFT, FakeLive

    from myopenfund_mcp.companies.live import Warming

    live = FakeLive(fail={"read:fundamentals:" + MSFT: Warming(2, 5)})
    msg = await _routed_error(None, live, "companies_facts", {"companies": [APPLE, MSFT]})
    assert msg.startswith("[warming]") and f"CIK {MSFT}): 2/5" in msg and APPLE not in msg


async def test_a_definitive_error_wins_over_warming():
    from source_fakes import APPLE, MSFT, FakeLive

    from myopenfund_mcp.companies.live import NoFacts, Warming

    live = FakeLive(fail={"concept_rows:" + APPLE: Warming(1, 2), "concept_rows:" + MSFT: NoFacts(MSFT)})
    msg = await _routed_error(None, live, "companies_fundamentals", {"companies": [APPLE, MSFT]})
    assert msg.startswith("[no_facts]") and MSFT in msg


def _live_errors():
    from company_corpus.live.sec import SecContactMissing, SecRefused, SecUnavailable

    from myopenfund_mcp.companies.live import LiveClosed, LiveDataError, NoFacts

    leak = f"contact {SENTINEL}"
    return [(NoFacts("0000320193"), "no_facts"), (SecRefused(leak), "sec_refused"),
            (SecUnavailable(leak), "sec_unavailable"), (SecContactMissing(leak), "sec_contact_missing"),
            (LiveDataError(leak), "backend_unavailable"), (LiveClosed(), "backend_unavailable")]


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("tool", ["companies_facts", "companies_fundamentals"])
async def test_live_errors_are_coded_and_never_hold_the_contact(index, tool, caplog):
    from source_fakes import APPLE, FakeLive

    exc, code = _live_errors()[index]
    method = "read:fundamentals" if tool == "companies_facts" else "concept_rows"
    msg = await _routed_error(None, FakeLive(fail={method: exc}), tool, {"companies": [APPLE]})
    assert msg.startswith(f"[{code}]")
    if code.startswith("sec_contact"):
        assert "MOF_SEC_CONTACT" in msg
    assert SENTINEL not in msg and SENTINEL not in caplog.text


async def test_results_and_server_info_never_hold_the_contact(caplog):
    from source_fakes import FakeLive, FakeStore

    async with _routed(FakeStore(), FakeLive()) as c:
        texts = [str((await c.call_tool(name, args)).data) for name, args in (
            ("server_info", {}), ("companies_resolve", {"query": "microsoft"}),
            ("companies_facts", {"companies": ["AAPL", "MSFT"]}),
            ("companies_fundamentals", {"companies": ["AAPL", "MSFT"], "metrics": "all"}))]
    assert texts and all(SENTINEL not in t for t in texts) and SENTINEL not in caplog.text


async def test_resolve_unions_store_and_live_companies():
    from source_fakes import APPLE, MSFT, FakeLive, FakeStore

    res = await _routed_call(FakeStore(), FakeLive(), "companies_resolve", {"query": "c"})  # Inc., CORP, CO
    assert {c["cik"] for c in res["candidates"]} >= {APPLE, MSFT}


async def test_server_info_reports_the_source_contact_and_cache(tmp_path):
    from source_fakes import FakeLive

    async with _routed(None, FakeLive(), cache_dir=str(tmp_path / "c"), companies_source="sec") as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["companies_source"] == "sec" and info["sec_contact_configured"] is True
    assert info["cache"] == {"dir": str(tmp_path / "c"), "bytes": None}  # injected source: no cache opened


async def test_the_server_builds_the_live_source_and_closes_it_at_shutdown(tmp_path):
    settings = Settings(sec_contact=SENTINEL, cache_dir=str(tmp_path / "c"))
    server = build_server(settings)
    async with Client(server) as c:
        info = (await c.call_tool("server_info", {})).data
        names = {t.name for t in await c.list_tools()}
    assert info["cache"] == {"dir": str(tmp_path / "c"), "bytes": 0}
    assert "companies_facts" in names and "datasets_list" not in names

    from unittest import mock

    from myopenfund_mcp.companies import live as live_mod

    made = []

    class Recording(live_mod.LiveSource):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    with mock.patch.object(live_mod, "LiveSource", Recording):
        server = build_server(settings)
    async with Client(server):
        assert made and not made[0]._closed.is_set()
    assert made[0]._closed.is_set()


def test_store_mode_builds_no_live_source(tmp_path, monkeypatch):
    import myopenfund_mcp.companies.live as live_mod

    monkeypatch.setattr(live_mod, "LiveSource", lambda *a, **kw: pytest.fail("built"))
    build_server(Settings(sec_contact=SENTINEL, companies_source="store", companies_catalog="x",
                          companies_data_root=str(tmp_path), cache_dir=str(tmp_path / "c")))
    assert not (tmp_path / "c").exists()


async def test_live_calls_get_the_sec_budget_on_top_of_the_store_budget():
    from source_fakes import APPLE, FakeLive

    live = FakeLive()
    real = live.concept_rows

    def slow(cik, as_of, **kw):
        time.sleep(1.3)
        return real(cik, as_of, **kw)

    live.concept_rows = slow
    res = await _routed_call(None, live, "companies_fundamentals", {"companies": [APPLE]},
                             companies_query_timeout_s=1, sec_call_budget_s=2)
    assert res["rows"]


async def test_a_company_needing_sec_without_contact_is_sec_contact_missing():
    """A company the resolver knows but the router must read live, with no live source (defensive:
    the store resolver lists only store companies, so the tools reach this through a store listing a
    CIK whose data the router sends to SEC)."""
    from source_fakes import APPLE, FakeStore

    from myopenfund_mcp.companies.common import store_call
    from myopenfund_mcp.companies.router import SourceRouter

    router = SourceRouter(FakeStore(ciks=()), None, "auto")
    with pytest.raises(ToolError) as err:
        await store_call(router.call, settings=Settings(), company=APPLE, route=router.route(APPLE),
                         method="concept_rows", cik=APPLE, as_of=parse_as_of(None))
    msg = str(err.value)
    assert msg.startswith("[sec_contact_missing]") and "MOF_SEC_CONTACT" in msg and APPLE in msg


async def test_rows_say_freshness_unchecked_when_sec_fails_during_the_check():
    from company_corpus.live.sec import SecUnavailable
    from source_fakes import APPLE, FakeLive, FakeStore

    res = await _routed_call(FakeStore(), FakeLive(fail={"filings": SecUnavailable("x")}), "companies_facts",
                             {"companies": [APPLE]})
    assert [(r["source"], r["reason"], r["value"]) for r in res["rows"]] == [("store", "freshness_unchecked", 100)]


async def test_store_companies_are_read_in_one_store_read():
    from source_fakes import APPLE, APPLE_REFS, JPM, MSFT, FakeLive, FakeStore

    store = FakeStore(ciks=(APPLE, JPM))
    store.filings[JPM] = list(store.filings[APPLE])
    store.positions[JPM] = set(store.positions[APPLE])
    live = FakeLive(refs={APPLE: list(APPLE_REFS), JPM: list(APPLE_REFS)})
    res = await _routed_call(store, live, "companies_facts", {"companies": [APPLE, JPM, MSFT]})
    assert {(r["cik"], r["source"]) for r in res["rows"]} == {(APPLE, "store"), (JPM, "store"), (MSFT, "sec_live")}
    assert store.calls.count("read:fundamentals") == 1
    assert live.calls.count("read:fundamentals") == 1
    assert store.calls.count("freshness") <= 2 and store.calls.count("datasets") == 1  # not per company


async def test_a_pure_store_call_keeps_the_store_budget():
    from source_fakes import APPLE, FakeLive, FakeStore

    store = FakeStore()
    real = store.read

    def slow(dataset, **kw):
        if dataset == "fundamentals":
            time.sleep(1.5)
        return real(dataset, **kw)

    store.read, settled = _tracked(slow)
    t0 = time.monotonic()
    msg = await _routed_error(store, FakeLive(), "companies_facts", {"companies": [APPLE]},
                              companies_query_timeout_s=1, sec_call_budget_s=30)
    assert msg.startswith("[timeout]") and time.monotonic() - t0 < 1.45
    await settled()


async def test_routing_time_counts_in_the_live_wait():
    from source_fakes import APPLE, FakeLive, FakeStore

    live = FakeLive()
    real = live.filings

    def slow_check(cik, **kw):
        time.sleep(0.5)
        return real(cik, **kw)

    live.filings = slow_check
    store = FakeStore(filings={})  # behind: Apple goes live after a 0.5 s freshness check
    res = await _routed_call(store, live, "companies_fundamentals", {"companies": [APPLE]}, sec_call_budget_s=2)
    assert [(r["source"], r["reason"]) for r in res["rows"]] == [("sec_live", "store_behind")]
    assert live.budgets and live.budgets[0] <= 1.55


async def test_a_resolver_sec_failure_is_reported_when_the_store_cannot_resolve():
    from company_corpus.live.sec import SecUnavailable
    from source_fakes import APPLE, FakeLive, FakeStore

    live = FakeLive(fail={"entities": SecUnavailable("x")})
    msg = await _routed_error(FakeStore(), live, "companies_facts", {"companies": ["MSFT"]})
    assert msg.startswith("[sec_unavailable]")
    msg = await _routed_error(FakeStore(), live, "companies_resolve", {"query": "microsoft"})
    assert msg.startswith("[sec_unavailable]")
    ok = await _routed_call(FakeStore(), live, "companies_resolve", {"query": "AAPL"})  # the store resolves it
    assert [c["cik"] for c in ok["candidates"]] == [APPLE]


async def test_a_store_company_and_a_live_company_sharing_a_name_are_ambiguous():
    from source_fakes import APPLE, FakeLive, FakeStore, entity

    other = "0000000777"
    live = FakeLive(known=[entity(other, name="Apple Inc.", ticker="APPL.X")])
    res = await _routed_call(FakeStore(), live, "companies_resolve", {"query": "apple inc"})
    assert {c["cik"] for c in res["candidates"]} == {APPLE, other}
    msg = await _routed_error(FakeStore(), live, "companies_facts", {"companies": ["apple inc"]})
    assert msg.startswith("[ambiguous_company]") and APPLE in msg and other in msg


async def test_a_new_live_ticker_of_a_store_company_resolves_to_it():
    from source_fakes import APPLE, FakeLive, FakeStore, entity

    live = FakeLive(known=[entity(APPLE, name="Apple Inc. (new name at SEC)", ticker="APLE.NEW")])
    res = await _routed_call(FakeStore(), live, "companies_resolve", {"query": "APLE.NEW"})
    assert [(c["cik"], c["current_name"]) for c in res["candidates"]] == [(APPLE, "Apple Inc.")]


def test_sources_built_before_a_refusal_are_closed(tmp_path, monkeypatch):
    import myopenfund_mcp.companies.store as store_mod
    from myopenfund_mcp.config import ConfigError

    made = []

    class Recording(store_mod.CompaniesStore):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)
            self.closed = False

        def close(self):
            self.closed = True
            super().close()

    monkeypatch.setattr(store_mod, "CompaniesStore", Recording)
    with pytest.raises(ConfigError):  # the store is built, then the live source's mode check refuses
        build_server(Settings(companies_catalog="x", companies_data_root=str(tmp_path), companies_source="sec"))
    assert made and made[0].closed


async def test_shutdown_closes_the_live_source_before_the_store(tmp_path, monkeypatch):
    import myopenfund_mcp.companies.live as live_mod
    import myopenfund_mcp.companies.store as store_mod

    order = []

    class Store(store_mod.CompaniesStore):
        def close(self):
            order.append("store")
            super().close()

    class Live(live_mod.LiveSource):
        def close(self):
            order.append("live")
            super().close()

    monkeypatch.setattr(store_mod, "CompaniesStore", Store)
    monkeypatch.setattr(live_mod, "LiveSource", Live)
    server = build_server(Settings(companies_catalog="x", companies_data_root=str(tmp_path), sec_contact=SENTINEL,
                                   cache_dir=str(tmp_path / "c")))
    async with Client(server):
        pass
    assert order == ["live", "store"]


async def test_a_slow_failing_sec_leaves_store_companies_answered_unchecked():
    """Three store companies, SEC failing after 0.8 s, 1 s + 1 s budgets: store rows, never [timeout]."""
    from company_corpus.live.sec import SecUnavailable
    from source_fakes import APPLE, APPLE_REFS, JPM, MSFT, FakeLive, FakeStore

    store = FakeStore(ciks=(APPLE, MSFT, JPM))
    for c in (MSFT, JPM):
        store.filings[c] = list(store.filings[APPLE])
        store.positions[c] = set(store.positions[APPLE])
    live = FakeLive(refs={c: list(APPLE_REFS) for c in (APPLE, MSFT, JPM)})

    def failing(cik, **kw):
        time.sleep(0.8)
        raise SecUnavailable("x")

    live.filings = failing
    for tool in ("companies_facts", "companies_fundamentals"):
        res = await _routed_call(store, live, tool, {"companies": [APPLE, MSFT, JPM]},
                                 companies_query_timeout_s=1, sec_call_budget_s=1)
        assert {(r["cik"], r["source"], r["reason"]) for r in res["rows"]} == {
            (c, "store", "freshness_unchecked") for c in (APPLE, MSFT, JPM)}
