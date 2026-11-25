"""The source router: each `auto` branch, the fixed modes, the store_unavailable fallback, the route memo."""

import datetime as dt

import pytest
from company_corpus.live.sec import SecUnavailable
from source_fakes import ACCEPTED_E24, APPLE, JPM, K24, MSFT, FakeLive, FakeStore, Ref

from myopenfund_mcp.companies.router import LiveNotConfigured, Route, SourceRouter
from myopenfund_mcp.companies.store import StoreBadArgument

NOW = dt.datetime(2026, 11, 25, tzinfo=dt.UTC)


def _router(store, live, mode="auto", **kw):
    return SourceRouter(store, live, mode, now=lambda: NOW, **kw)


# --- auto: each branch --------------------------------------------------------------------------


def test_auto_store_current_with_positions_answers_from_the_store():
    assert _router(FakeStore(), FakeLive()).route(APPLE) == Route("store")


def test_auto_without_a_store_is_live_no_store():
    live = FakeLive()
    assert _router(None, live).route(APPLE) == Route("sec_live", "no_store")
    assert live.calls == []  # nothing to compare: no SEC request for the route


def test_auto_company_not_in_store():
    live = FakeLive()
    assert _router(FakeStore(), live).route(MSFT) == Route("sec_live", "not_in_store")
    assert live.calls == []


def test_auto_store_behind_sec():
    later = Ref("0000320193-25-000001", ACCEPTED_E24 + dt.timedelta(days=90), "10-Q")
    live = FakeLive(latest={APPLE: (later, later)})
    assert _router(FakeStore(), live).route(APPLE) == Route("sec_live", "store_behind")


def test_auto_store_without_filings_rows_is_behind():
    assert _router(FakeStore(filings={}), FakeLive()).route(APPLE) == Route("sec_live", "store_behind")


def test_auto_store_without_positions_for_the_latest_report():
    store = FakeStore(positions={APPLE: set()})
    assert _router(store, FakeLive()).route(APPLE) == Route("sec_live", "store_no_positions")


def test_auto_newest_8k_held_by_the_store_is_current():
    # SEC's newest report-form filing is an 8-K (never positions): the positions check reads the 10-K
    store = FakeStore()
    assert _router(store, FakeLive()).route(APPLE) == Route("store")
    assert "read:filing_statements" in store.calls


def test_auto_same_accession_is_current_without_an_acceptance_time():
    live = FakeLive(latest={APPLE: (Ref(K24, None), Ref(K24, None))})
    assert _router(FakeStore(filings={APPLE: [(K24, None)]}), live).route(APPLE) == Route("store")


def test_auto_store_failure_is_store_unavailable_and_not_remembered():
    store = FakeStore(fail={"entities": RuntimeError("down")})
    router = _router(store, FakeLive())
    assert router.route(APPLE) == Route("sec_live", "store_unavailable")
    store.fail.clear()
    assert router.route(APPLE) == Route("store")


def test_auto_store_failure_during_the_freshness_reads():
    store = FakeStore(fail={"read:filings": RuntimeError("down")})
    assert _router(store, FakeLive()).route(APPLE) == Route("sec_live", "store_unavailable")


def test_auto_sec_check_failure_serves_the_store_and_checks_again_next_call():
    live = FakeLive(fail={"latest_filing": SecUnavailable("SEC unavailable")})
    router = _router(FakeStore(), live)
    assert router.route(APPLE) == Route("store")
    live.fail.clear()
    live.latest = {APPLE: (Ref("x", ACCEPTED_E24 + dt.timedelta(days=1)),) * 2}
    assert router.route(APPLE) == Route("sec_live", "store_behind")


def test_auto_without_contact_serves_store_companies_and_refuses_the_others():
    router = _router(FakeStore(), None)
    assert router.route(APPLE) == Route("store")
    assert router.call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW)[0] == Route("store")
    with pytest.raises(LiveNotConfigured) as err:
        router.call(company=MSFT, method="concept_rows", cik=MSFT, as_of=NOW)
    assert err.value.reason == "not_in_store"


def test_routes_are_remembered_for_the_ttl():
    now = [0.0]
    live = FakeLive()
    router = _router(FakeStore(), live, route_ttl_s=3600, clock=lambda: now[0])
    router.route(APPLE)
    router.route(APPLE)
    assert live.calls.count("latest_filing") == 2  # one check: report forms + positions forms
    now[0] = 3601
    router.route(APPLE)
    assert live.calls.count("latest_filing") == 4


# --- fixed modes --------------------------------------------------------------------------------


def test_store_mode_never_calls_live():
    live = FakeLive()
    router = _router(FakeStore(), live, "store")
    assert router.route(MSFT) == Route("store")
    router.call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW)
    router.entities()
    assert live.calls == []


def test_sec_mode_never_calls_the_store():
    store = FakeStore()
    router = _router(store, FakeLive(), "sec")
    assert router.route(APPLE) == Route("sec_live")
    router.call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW)
    router.read("fundamentals", as_of=NOW, filters={"entity_value": [APPLE]}, columns=None, limit=None)
    router.entities()
    assert store.calls == []


def test_modes_need_their_source():
    with pytest.raises(ValueError, match="store"):
        SourceRouter(None, FakeLive(), "store")
    with pytest.raises(ValueError, match="MOF_SEC_CONTACT"):
        SourceRouter(FakeStore(), None, "sec")
    with pytest.raises(ValueError):
        SourceRouter(None, None, "auto")
    with pytest.raises(ValueError, match="mode"):
        SourceRouter(FakeStore(), None, "both")


# --- call: dispatch and fallback ----------------------------------------------------------------


def test_call_dispatches_per_company_and_passes_the_period_to_live_only():
    store, live = FakeStore(), FakeLive()
    router = _router(store, live)
    p = {"period_from": dt.date(2023, 1, 1), "period_to": dt.date(2024, 12, 31)}
    a_route, (a_rows, _) = router.call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW, **p)
    m_route, (m_rows, _) = router.call(company=MSFT, method="concept_rows", cik=MSFT, as_of=NOW, **p)
    assert (a_route.source, a_rows[0]["value"]) == ("store", 100)
    assert (m_route, m_rows[0]["value"]) == (Route("sec_live", "not_in_store"), 200)
    assert live.periods == [(p["period_from"], p["period_to"])]


def test_a_store_failure_in_the_data_call_answers_live_for_that_call():
    store = FakeStore(fail={"concept_rows": RuntimeError("down")})
    router = _router(store, FakeLive())
    route, (rows, _) = router.call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW)
    assert route == Route("sec_live", "store_unavailable") and rows[0]["value"] == 200
    store.fail.clear()
    assert router.call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW)[0] == Route("store")


def test_an_argument_error_is_not_a_store_failure():
    store = FakeStore(fail={"concept_rows": StoreBadArgument("bad")})
    live = FakeLive()
    with pytest.raises(StoreBadArgument):
        _router(store, live).call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW)
    assert "concept_rows" not in live.calls


def test_store_failure_without_fallback_in_store_mode():
    store = FakeStore(fail={"concept_rows": RuntimeError("down")})
    with pytest.raises(RuntimeError):
        _router(store, FakeLive(), "store").call(company=APPLE, method="concept_rows", cik=APPLE, as_of=NOW)


def test_read_of_several_companies_is_merged_per_source():
    router = _router(FakeStore(), FakeLive())
    rows = router.read(
        "fundamentals",
        as_of=NOW,
        filters={"entity_scheme": ["cik"], "entity_value": [APPLE, MSFT]},
        columns=None,
        limit=None,
    )
    assert {(r["entity_value"], r["value"]) for r in rows} == {(APPLE, 100), (MSFT, 200)}


# --- entities -----------------------------------------------------------------------------------


def test_auto_entities_union_store_rows_win():
    store, live = FakeStore(), FakeLive()
    store_rows = store.entities()
    store_rows[0]["name"] = "APPLE FROM STORE"
    store.entities = lambda: [dict(r) for r in store_rows]
    rows = _router(store, live).entities()
    assert {r["subject_value"] for r in rows} == {APPLE, MSFT, JPM}
    assert [r["name"] for r in rows if r["subject_value"] == APPLE] == ["APPLE FROM STORE"]


def test_auto_entities_when_one_side_fails():
    live_down = FakeLive(fail={"entities": SecUnavailable("x")})
    assert {r["subject_value"] for r in _router(FakeStore(), live_down).entities()} == {APPLE}
    store_down = FakeStore(fail={"entities": RuntimeError("x")})
    assert {r["subject_value"] for r in _router(store_down, FakeLive()).entities()} == {APPLE, MSFT, JPM}


def test_auto_entities_without_contact_or_store():
    assert {r["subject_value"] for r in _router(FakeStore(), None).entities()} == {APPLE}
    assert {r["subject_value"] for r in _router(None, FakeLive()).entities()} == {APPLE, MSFT, JPM}


def test_close_closes_both_sources():
    store, live = FakeStore(), FakeLive()
    _router(store, live).close()
    assert store.closed and live.closed


def test_the_live_source_exposes_the_forms_the_router_compares():
    from myopenfund_mcp.companies.live import LiveSource
    from myopenfund_mcp.config import Settings

    src = LiveSource(object(), Settings())
    try:
        assert "8-K" in src.report_forms and "10-K" in src.report_forms
        assert set(src.positions_forms) == {"10-K", "10-Q", "20-F", "40-F"}
    finally:
        src.close()
