"""The source router: each `auto` branch for a call's `as_of`, the fixed modes, the store_unavailable
fallback, what is remembered per store publication; on fakes and on the real seeded store."""

import datetime as dt
import logging

import pyarrow as pa
import pytest
from companies_fixture import AAPL_10K_2024, AAPL_10K_2025, NOW, fact, position, seed
from company_corpus.live.sec import SecUnavailable
from source_fakes import (
    ACCEPTED_E24,
    ACCEPTED_K24,
    APPLE,
    APPLE_REFS,
    E24,
    JPM,
    K24,
    MSFT,
    FakeLive,
    FakeStore,
    Ref,
    entity,
)

from myopenfund_mcp.companies.router import LiveNotConfigured, Route, SourceRouter
from myopenfund_mcp.companies.store import CompaniesStore, StoreBadArgument

LATER = dt.datetime(2026, 11, 25, tzinfo=dt.UTC)
STORE, UNCHECKED = Route("store"), Route("store", "freshness_unchecked")


def _router(store, live, mode="auto", **kw):
    return SourceRouter(store, live, mode, now=lambda: LATER, **kw)


def _q(accn: str, accepted: dt.datetime, form: str = "10-Q") -> Ref:
    return Ref(accn, accepted, form)


# --- auto: each branch --------------------------------------------------------------------------


def test_auto_store_current_with_positions_answers_from_the_store():
    assert _router(FakeStore(), FakeLive()).route(APPLE) == STORE


def test_auto_without_a_store_is_live_no_store():
    live = FakeLive()
    assert _router(None, live).route(APPLE) == Route("sec_live", "no_store")
    assert live.calls == []  # nothing to compare: no SEC request for the route


def test_auto_company_not_in_store():
    live = FakeLive()
    assert _router(FakeStore(), live).route(MSFT) == Route("sec_live", "not_in_store")
    assert live.calls == []


def test_auto_store_behind_sec():
    later = _q("0000320193-25-000001", ACCEPTED_E24 + dt.timedelta(days=90))
    live = FakeLive(refs={APPLE: [*APPLE_REFS, later]})
    assert _router(FakeStore(), live).route(APPLE) == Route("sec_live", "store_behind")


def test_auto_store_without_filings_rows_for_the_company_is_behind():
    assert _router(FakeStore(filings={}), FakeLive()).route(APPLE) == Route("sec_live", "store_behind")


def test_auto_store_without_positions_for_the_latest_report():
    store = FakeStore(positions={APPLE: {"0000320193-23-000106"}})  # FY2023 only
    assert _router(store, FakeLive()).route(APPLE) == Route("sec_live", "store_no_positions")


def test_auto_newest_8k_held_by_the_store_is_current():
    # SEC's newest report-form filing is an 8-K (never positions): the positions check reads the 10-K
    store = FakeStore()
    assert _router(store, FakeLive()).route(APPLE) == STORE
    assert "read:filing_statements" in store.calls


def test_auto_same_accession_is_current_without_comparing_times():
    store = FakeStore(filings={APPLE: [(K24, None), (E24, None)]})
    assert _router(store, FakeLive()).route(APPLE) == STORE


def test_a_filing_without_acceptance_time_is_not_compared():
    live = FakeLive(refs={APPLE: [*APPLE_REFS, Ref("0000320193-26-000001", None, "10-Q")]})
    assert _router(FakeStore(), live).route(APPLE) == STORE


# --- auto: as_of (R30) --------------------------------------------------------------------------


def test_as_of_in_the_past_store_holds_everything_known_then():
    """SEC has yesterday's 10-Q and 8-K, the store has not: behind now, current for an earlier as_of."""
    yesterday = LATER - dt.timedelta(days=1)
    live = FakeLive(
        refs={
            APPLE: [
                *APPLE_REFS,
                _q("0000320193-26-000010", yesterday),
                _q("0000320193-26-000011", yesterday, "8-K"),
            ]
        }
    )
    router = _router(FakeStore(), live)
    assert router.route(APPLE) == Route("sec_live", "store_behind")
    assert router.route(APPLE, ACCEPTED_E24 + dt.timedelta(days=30)) == STORE
    assert router.route(APPLE, LATER - dt.timedelta(days=2)) == STORE


def test_as_of_between_a_report_and_its_positions():
    """At an as_of before the 10-K, the positions checked are the previous report's."""
    store = FakeStore(positions={APPLE: {"0000320193-23-000106"}})
    router = _router(store, FakeLive())
    assert router.route(APPLE) == Route("sec_live", "store_no_positions")
    assert router.route(APPLE, ACCEPTED_K24 - dt.timedelta(days=1)) == STORE


# --- auto: the freshness check cannot run (R27) -------------------------------------------------


def test_without_contact_store_companies_are_unchecked_and_the_others_need_sec():
    router = _router(FakeStore(), None)
    assert router.route(APPLE) == UNCHECKED
    assert (
        router.call(company=APPLE, route=UNCHECKED, method="concept_rows", cik=APPLE, as_of=LATER)[0]
        == UNCHECKED
    )
    route = router.route(MSFT)
    with pytest.raises(LiveNotConfigured) as err:
        router.call(company=MSFT, route=route, method="concept_rows", cik=MSFT, as_of=LATER)
    assert err.value.reason == "not_in_store"


def test_sec_failure_during_the_check_is_unchecked_and_checked_again_next_call():
    live = FakeLive(fail={"filings": SecUnavailable("SEC unavailable")})
    router = _router(FakeStore(), live)
    assert router.route(APPLE) == UNCHECKED
    live.fail.clear()
    live.refs[APPLE].append(_q("x", ACCEPTED_E24 + dt.timedelta(days=1)))
    assert router.route(APPLE) == Route("sec_live", "store_behind")


# --- the store failing --------------------------------------------------------------------------


def test_auto_store_failure_is_store_unavailable_and_not_remembered():
    store = FakeStore(fail={"freshness": RuntimeError("down")})
    router = _router(store, FakeLive())
    assert router.route(APPLE) == Route("sec_live", "store_unavailable")
    store.fail.clear()
    assert router.route(APPLE) == STORE


def test_auto_store_failure_during_the_freshness_reads():
    store = FakeStore(fail={"read:filings": RuntimeError("down")})
    assert _router(store, FakeLive()).route(APPLE) == Route("sec_live", "store_unavailable")


def test_a_store_failure_without_contact_is_raised():
    with pytest.raises(RuntimeError):
        _router(FakeStore(fail={"freshness": RuntimeError("down")}), None).route(APPLE)


# --- memory: per publication, one freshness read per plan ---------------------------------------


def test_a_plan_reads_the_store_state_once_whatever_the_number_of_companies():
    store = FakeStore(ciks=(APPLE, MSFT, JPM))
    live = FakeLive(refs={c: list(APPLE_REFS) for c in (APPLE, MSFT, JPM)})
    _router(store, live).plan([APPLE, MSFT, JPM])
    assert (store.calls.count("freshness"), store.calls.count("entities"), store.calls.count("datasets")) == (
        1,
        1,
        1,
    )


def test_what_the_store_holds_is_remembered_until_a_publication():
    store = FakeStore(filings={APPLE: [(K24, ACCEPTED_K24)]})  # the 8-K is not in the store yet
    router = _router(store, FakeLive())
    assert router.route(APPLE) == Route("sec_live", "store_behind")
    store.filings[APPLE].append((E24, ACCEPTED_E24))
    assert router.route(APPLE) == Route("sec_live", "store_behind")  # same publication: remembered
    assert store.calls.count("read:filings") == 1
    store.batches += 1  # a publication drops it
    assert router.route(APPLE) == STORE
    assert store.calls.count("read:filings") == 2


def test_sec_filings_are_read_on_every_plan_through_the_live_memo():
    live = FakeLive()
    router = _router(FakeStore(), live)
    router.route(APPLE)
    router.route(APPLE)
    assert live.calls.count("filings") == 2  # the live source's submissions memo answers the second


# --- fixed modes --------------------------------------------------------------------------------


def test_store_mode_never_calls_live():
    live = FakeLive()
    router = _router(FakeStore(), live, "store")
    assert router.route(MSFT) == STORE
    router.call(company=APPLE, route=STORE, method="concept_rows", cik=APPLE, as_of=LATER)
    router.entities()
    assert live.calls == []


def test_sec_mode_never_calls_the_store():
    store = FakeStore()
    router = _router(store, FakeLive(), "sec")
    assert router.route(APPLE) == Route("sec_live")
    router.concept_rows(APPLE, LATER)
    router.read("fundamentals", as_of=LATER, filters={"entity_value": [APPLE]}, columns=None, limit=None)
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


def test_call_passes_the_period_and_budget_to_live_only():
    store, live = FakeStore(), FakeLive()
    router = _router(store, live)
    p = {"period_from": dt.date(2023, 1, 1), "period_to": dt.date(2024, 12, 31)}
    a_route, (a_rows, _) = router.call(
        company=APPLE, route=STORE, method="concept_rows", cik=APPLE, as_of=LATER, budget_s=7, **p
    )
    live_route = Route("sec_live", "not_in_store")
    m_route, (m_rows, _) = router.call(
        company=MSFT, route=live_route, method="concept_rows", cik=MSFT, as_of=LATER, budget_s=7, **p
    )
    assert (a_route, a_rows[0]["value"]) == (STORE, 100)
    assert (m_route, m_rows[0]["value"]) == (live_route, 200)
    assert live.periods == [(p["period_from"], p["period_to"])] and live.budgets == [7]


def test_a_store_failure_in_the_data_call_answers_live_for_that_call():
    store = FakeStore(fail={"concept_rows": RuntimeError("down")})
    router = _router(store, FakeLive())
    route, (rows, _) = router.call(company=APPLE, route=STORE, method="concept_rows", cik=APPLE, as_of=LATER)
    assert route == Route("sec_live", "store_unavailable") and rows[0]["value"] == 200
    assert router.try_store(method="concept_rows", cik=APPLE, as_of=LATER) is None


def test_an_argument_error_is_not_a_store_failure():
    store = FakeStore(fail={"concept_rows": StoreBadArgument("bad")})
    live = FakeLive()
    with pytest.raises(StoreBadArgument):
        _router(store, live).call(company=APPLE, route=STORE, method="concept_rows", cik=APPLE, as_of=LATER)
    assert "concept_rows" not in live.calls


def test_store_failure_without_fallback_in_store_mode():
    store = FakeStore(fail={"concept_rows": RuntimeError("down")})
    with pytest.raises(RuntimeError):
        _router(store, FakeLive(), "store").try_store(method="concept_rows", cik=APPLE, as_of=LATER)


def test_read_of_several_companies_reads_the_store_once():
    store = FakeStore()
    router = _router(store, FakeLive())
    rows = router.read(
        "fundamentals",
        as_of=LATER,
        filters={"entity_scheme": ["cik"], "entity_value": [APPLE, MSFT]},
        columns=None,
        limit=None,
    )
    assert {(r["entity_value"], r["value"]) for r in rows} == {(APPLE, 100), (MSFT, 200)}
    assert store.calls.count("read:fundamentals") == 1


# --- entities -----------------------------------------------------------------------------------


def test_auto_entities_union_store_names_win_and_live_tickers_add():
    store = FakeStore()
    store.entities = lambda: [{**entity(APPLE), "name": "APPLE FROM STORE", "id_value": "AAPL"}]
    live = FakeLive(known=[entity(APPLE, ticker="APLE.NEW")])  # a ticker the store does not have yet
    rows = _router(store, live).entities()
    assert {r["subject_value"] for r in rows} == {APPLE, MSFT, JPM}
    assert {r["name"] for r in rows if r["subject_value"] == APPLE} == {"APPLE FROM STORE", None}
    assert {r["id_value"] for r in rows if r["subject_value"] == APPLE} == {"AAPL", "APLE.NEW"}
    assert all(r["entity_id"] in (None, f"e-{APPLE}") for r in rows if r["subject_value"] == APPLE)


def test_auto_entities_when_one_side_fails():
    failure = SecUnavailable("x")
    rows, err = _router(FakeStore(), FakeLive(fail={"entities": failure})).resolver_rows()
    assert {r["subject_value"] for r in rows} == {APPLE} and err is failure
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


# --- the real seeded store (no `filings` dataset published: R29) --------------------------------

ACCEPTED_10K_2024 = dt.datetime(2024, 11, 1, 20, 0, tzinfo=dt.UTC)
ACCEPTED_10K_2025 = dt.datetime(2025, 10, 31, 20, 0, tzinfo=dt.UTC)
Q1_2026 = "0000320193-26-000006"
ACCEPTED_Q1_2026 = dt.datetime(2026, 1, 30, 21, 0, tzinfo=dt.UTC)


@pytest.fixture
def real(tmp_path):
    cfg = seed(tmp_path)
    s = CompaniesStore(str(cfg.data_root), cfg.catalog)
    yield s, cfg
    s.close()


def _sec(*refs) -> FakeLive:
    return FakeLive(refs={APPLE: list(refs)})


def test_real_store_current_company_is_served_by_the_store(real, caplog):
    store, _ = real
    live = _sec(
        Ref(AAPL_10K_2024, ACCEPTED_10K_2024),
        Ref(AAPL_10K_2025, ACCEPTED_10K_2025),
        Ref("0000320193-25-000090", ACCEPTED_10K_2025 + dt.timedelta(days=1), "8-K"),
    )
    with caplog.at_level(logging.WARNING, logger="myopenfund_mcp.companies"):
        router = _router(store, live)
        # before the FY2025 10-K: the store holds the 2024 10-K and its positions
        assert router.route(APPLE, dt.datetime(2025, 6, 1, tzinfo=dt.UTC)) == STORE
        router.route(APPLE)
    assert caplog.text.count("no published 'filings' dataset") == 1  # logged once


def test_real_store_behind_sec(real):
    store, _ = real
    live = _sec(
        Ref(AAPL_10K_2024, ACCEPTED_10K_2024),
        Ref(AAPL_10K_2025, ACCEPTED_10K_2025),
        Ref(Q1_2026, ACCEPTED_Q1_2026, "10-Q"),
    )
    assert _router(store, live).route(APPLE) == Route("sec_live", "store_behind")


def test_real_store_latest_filing_without_positions(real):
    store, _ = real  # the FY2025 10-K has facts in the store but no filing_statements rows
    live = _sec(Ref(AAPL_10K_2024, ACCEPTED_10K_2024), Ref(AAPL_10K_2025, ACCEPTED_10K_2025))
    assert _router(store, live).route(APPLE) == Route("sec_live", "store_no_positions")


def test_real_store_without_filings_compares_the_latest_filing_with_facts(real):
    """R29: without `filings`, an 8-K the store's facts cannot show does not put the company behind."""
    store, _ = real
    live = _sec(
        Ref(AAPL_10K_2024, ACCEPTED_10K_2024),
        Ref("0000320193-24-000130", ACCEPTED_10K_2024 + dt.timedelta(days=4), "8-K"),
    )
    assert _router(store, live).route(APPLE) == STORE
    assert _router(store, live).route(JPM) == STORE  # SEC lists nothing newer for it
    assert _router(store, live).route("0000789019") == Route("sec_live", "not_in_store")


def test_real_store_publication_invalidates_what_was_remembered(real):
    from open_data_store.contracts.standard import FILING_STATEMENTS, FUNDAMENTALS
    from open_data_store.publisher import publish_pending
    from open_data_store.writer import write_batch

    store, cfg = real
    live = _sec(Ref(AAPL_10K_2024, ACCEPTED_10K_2024), Ref(Q1_2026, ACCEPTED_Q1_2026, "10-Q"))
    router = _router(store, live)
    assert router.route(APPLE) == Route("sec_live", "store_behind")
    later = NOW + dt.timedelta(days=1)
    write_batch(
        FUNDAMENTALS,
        pa.Table.from_pylist(
            [
                fact(
                    APPLE,
                    "us-gaap:Revenues",
                    dt.date(2025, 12, 27),
                    1,
                    start=dt.date(2025, 9, 28),
                    freq="quarterly",
                    accn=Q1_2026,
                    known="2026-01-30T21:00:00",
                )
            ],
            schema=FUNDAMENTALS.schema,
        ),
        landing_root=cfg.landing,
        run_id="r2",
        producer="test",
        now=later,
    )
    write_batch(
        FILING_STATEMENTS,
        pa.Table.from_pylist(
            [
                position(
                    APPLE,
                    "Revenues",
                    "IS",
                    accn=Q1_2026,
                    end=dt.date(2025, 12, 27),
                    known="2026-01-30T21:00:00",
                )
            ],
            schema=FILING_STATEMENTS.schema,
        ),
        landing_root=cfg.landing,
        run_id="r2",
        producer="test",
        now=later,
    )
    store.close()  # the publisher attaches the catalogue itself; the store reopens lazily
    assert publish_pending(cfg, now=later).exit_code == 0
    assert router.route(APPLE) == STORE
