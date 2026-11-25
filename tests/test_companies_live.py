"""The live source: the store's method set computed from SEC (a fake one), offline."""

from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime

import pytest
from company_corpus.live.sec import SecRefused, SecUnavailable
from live_fixture import (
    AAPL,
    AAPL_E24,
    AAPL_K23,
    AAPL_K24,
    AAPL_Q24,
    AAPL_REVENUE_FY2024,
    ARCHIVES,
    COMPANYFACTS_URL,
    MSFT,
    MSFT_K24,
    MSFT_REVENUE_FY2024,
    NOFACTS,
    REV,
    TICKERS_URL,
    FakeSec,
)

from myopenfund_mcp.companies import resolve
from myopenfund_mcp.companies.live import LiveDataError, LiveSource, NoFacts, Warming
from myopenfund_mcp.companies.store import StoreBadArgument, StoreUnknownDataset
from myopenfund_mcp.config import Settings

NOW = datetime(2026, 11, 25, 12, tzinfo=UTC)
TAG = f"us-gaap:{REV}"  # original_tag as stored
LATER = datetime(2026, 1, 1, tzinfo=UTC)
#: Apple's 10-K/10-Q filings in the default 5-year window (the 8-K gets no positions)
AAPL_POSITIONS_FILINGS = 3


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _settings(**kw) -> Settings:
    return Settings(**{"sec_call_budget_s": 5, "companies_max_concurrency": 2, **kw})


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make(clock):
    made = []

    def _make(sec=None, **kw):
        src = LiveSource(sec or FakeSec(), _settings(**kw), clock=clock, now=lambda: NOW)
        made.append(src)
        return src

    yield _make
    for src in made:
        src.close()


def _folder(cik: str, accn: str) -> str:
    return ARCHIVES.format(cik_int=int(cik), acc_nodash=accn.replace("-", ""))


def _annual_revenue(rows, day):
    return [
        r for r in rows if r["concept"] == "revenue" and r["frequency"] == "annual" and r["valid_date"] == day
    ]


# --- the corpus's own computation ----------------------------------------------------------------
def test_apple_fy2024_revenue_through_concept_rows_and_summaries(make):
    src = make()
    rows, vintages = src.concept_rows(AAPL, LATER)
    rev = _annual_revenue(rows, date(2024, 9, 28))
    assert len(rev) == 1 and rev[0]["value"] == AAPL_REVENUE_FY2024
    assert rev[0]["original_tag"] == TAG and rev[0]["filing_ref"] == AAPL_K24
    assert all(v["entity_value"] == AAPL for v in vintages)
    assert len([v for v in vintages if v["original_tag"] == TAG]) == 4  # re-reported FY2023 kept
    summaries = src.summaries(AAPL, "Apple Inc.", rows, vintages)
    fy = [s for s in summaries if s.period_end == date(2024, 9, 28) and s.frequency == "annual"]
    assert len(fy) == 1 and fy[0].values["revenue"]["value"] == AAPL_REVENUE_FY2024
    assert fy[0].company == "Apple Inc."
    net_margin = fy[0].derived["net_margin"]
    assert net_margin["value"] == pytest.approx(100 * 93736000000 / AAPL_REVENUE_FY2024)  # percent
    assert net_margin["sector_relevant"] is True  # not sector-sensitive (sector unknown, as in the store)


def test_recorded_positions_place_microsoft_revenue_on_its_statement(make):
    rows, _ = make().concept_rows(MSFT, LATER)
    rev = _annual_revenue(rows, date(2024, 6, 30))
    assert len(rev) == 1 and rev[0]["value"] == MSFT_REVENUE_FY2024
    assert rev[0]["location"] == "statement" and rev[0]["filing_ref"] == MSFT_K24


def test_concept_rows_as_of_before_a_filing_ignore_it(make):
    rows, vintages = make().concept_rows(AAPL, datetime(2024, 1, 1, tzinfo=UTC))
    assert _annual_revenue(rows, date(2024, 9, 28)) == []
    assert {v["filing_ref"] for v in vintages} == {AAPL_K23}


def test_q4_rule_is_the_settings(make):
    with pytest.raises(ValueError, match="q4_rule"):
        make(companies_q4_rule="guess")


def test_catalogs_are_company_corpus_copies(make):
    src = make()
    concepts = src.concept_catalog()
    assert "revenue" in {c["key"] for c in concepts}
    concepts[0]["key"] = "tampered"
    assert src.concept_catalog()[0]["key"] != "tampered"
    assert "net_margin" in {m["key"] for m in src.metric_catalog()}


# --- read --------------------------------------------------------------------------------------------
def _read(src, dataset="fundamentals", *, as_of=LATER, filters=None, **kw):
    kw = {"columns": None, "limit": None, **kw}
    flt = {"entity_scheme": "cik", "entity_value": [AAPL], **(filters or {})}
    return src.read(dataset, as_of=as_of, filters=flt, **kw)


def test_read_as_of_excludes_rows_known_later(make):
    rows = _read(make(), as_of=datetime(2024, 1, 1, tzinfo=UTC))
    assert rows and {r["filing_ref"] for r in rows} == {AAPL_K23}
    assert all(r["known_at"] <= datetime(2024, 1, 1, tzinfo=UTC) for r in rows)


def test_read_latest_per_key_unless_vintages(make):
    src = make()
    flt = {"original_tag": TAG, "valid_date": "2023-09-30"}
    latest = _read(src, filters=flt)
    assert [r["filing_ref"] for r in latest] == [AAPL_K24]  # the later vintage of the key
    every = _read(src, filters=flt, vintages=True)
    assert [r["filing_ref"] for r in every] == [AAPL_K23, AAPL_K24]  # natural key, then known_at


def test_read_sort_limit_and_columns(make):
    src = make()
    rows = _read(src)
    keys = [(r["original_tag"], r["valid_date"], r["frequency"]) for r in rows]
    assert len(rows) == 7 and keys == sorted(keys)  # 8 facts, FY2023 revenue's two vintages once
    top = _read(src, limit=2, columns=["original_tag", "value"])
    assert top == [{k: r[k] for k in ("original_tag", "value")} for r in rows[:2]]


def test_read_filings(make):
    rows = _read(make(), "filings")
    assert {r["filing_ref"] for r in rows} >= {AAPL_K24, AAPL_K23, AAPL_E24}


@pytest.mark.parametrize("dataset", ["entity_observations", "filing_statements", "nope"])
def test_read_other_datasets_is_unknown(make, dataset):
    with pytest.raises(StoreUnknownDataset):
        _read(make(), dataset)


@pytest.mark.parametrize(
    ("filters", "columns", "limit", "match"),
    [
        ({"entity_value": [AAPL], "nope": 1}, None, None, "unknown column"),
        ({"entity_value": [AAPL]}, ["nope"], None, "unknown column"),
        ({"entity_value": [AAPL], "valid_date": "soon"}, None, None, "valid_date"),
        ({"entity_value": [AAPL]}, None, -1, "limit"),
        ({"original_tag": TAG}, None, None, "entity_value"),
    ],
)
def test_read_bad_arguments(make, filters, columns, limit, match):
    sec = FakeSec()
    with pytest.raises(StoreBadArgument, match=match):
        make(sec).read("fundamentals", as_of=LATER, filters=filters, columns=columns, limit=limit)
    assert sec.calls == []  # checked before any request


def test_read_of_another_scheme_or_a_non_cik_value_fetches_nothing(make):
    sec = FakeSec()
    src = make(sec)
    assert _read(src, filters={"entity_scheme": "lei"}) == []
    assert _read(src, filters={"entity_value": "320193"}) == []
    assert sec.calls == []


def test_read_too_many_companies(make):
    flt = {"entity_value": [str(i).zfill(10) for i in range(3)]}
    with pytest.raises(StoreBadArgument, match="companies"):
        make(companies_max_companies=2).read(
            "fundamentals", as_of=LATER, filters=flt, columns=None, limit=None
        )


def test_naive_as_of_is_a_bad_argument(make):
    with pytest.raises(StoreBadArgument, match="as_of"):
        make().concept_rows(AAPL, datetime(2026, 1, 1))


# --- memo, window, warming ------------------------------------------------------------------------
def test_tables_are_memoised_for_the_companyfacts_ttl(make, clock):
    sec = FakeSec()
    src = make(sec, cache_ttl_companyfacts_h=1)
    src.concept_rows(AAPL, LATER)
    n = len(sec.calls)
    _read(src)
    assert len(sec.calls) == n
    clock.t += 3601
    src.concept_rows(AAPL, LATER)
    assert len(sec.calls) > n
    facts = COMPANYFACTS_URL.format(cik=AAPL)
    assert dict(sec.calls)[facts] == 3600  # the TTL reaches the cache as max_age_s


def test_positions_only_for_the_window(make):
    sec = FakeSec()
    src = make(sec)
    src.concept_rows(AAPL, LATER, period_from=date(2024, 1, 1), period_to=date(2024, 12, 31))
    urls = sec.urls()
    assert any(u.startswith(_folder(AAPL, AAPL_K24)) for u in urls)
    assert any(u.startswith(_folder(AAPL, AAPL_Q24)) for u in urls)
    assert not any(u.startswith(_folder(AAPL, AAPL_K23)) for u in urls)
    assert not any(u.startswith(_folder(AAPL, AAPL_E24)) for u in urls)  # not a report form
    n = len(urls)
    src.concept_rows(AAPL, LATER, period_from=date(2024, 6, 1), period_to=date(2024, 9, 30))
    assert len(sec.calls) == n  # a window inside the built one reuses it
    src.concept_rows(AAPL, LATER, period_from=date(2023, 1, 1))
    assert any(u.startswith(_folder(AAPL, AAPL_K23)) for u in sec.urls()[n:])


def test_default_window_is_the_lookback(make):
    sec = FakeSec()
    make(sec, sec_positions_lookback_y=2).concept_rows(AAPL, LATER)  # 2024-11-25 onwards
    assert not any("/Archives/" in u for u in sec.urls())


def test_slow_sec_answers_warming_then_serves(make):
    sec = FakeSec(hold="companyfacts")
    src = make(sec, sec_call_budget_s=1)
    with pytest.raises(Warming) as info:
        src.concept_rows(AAPL, LATER)
    # submissions fetched; companyfacts held; 3 requests per 10-K/10-Q in the window
    assert (info.value.done, info.value.total) == (1, 2 + 3 * AAPL_POSITIONS_FILINGS)
    assert "1/11" in str(info.value)
    sec.gate.set()
    rows, _ = src.concept_rows(AAPL, LATER)
    assert _annual_revenue(rows, date(2024, 9, 28))[0]["value"] == AAPL_REVENUE_FY2024
    assert sec.urls().count(COMPANYFACTS_URL.format(cik=AAPL)) == 1  # the same build, finished


def test_concurrent_callers_share_one_build(make):
    sec = FakeSec(hold="companyfacts")
    src = make(sec)
    results, errors = [], []

    def call():
        try:
            results.append(src.concept_rows(AAPL, LATER))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    sec.gate.set()
    for t in threads:
        t.join(10)
    assert errors == [] and len(results) == 4
    assert sec.urls().count(COMPANYFACTS_URL.format(cik=AAPL)) == 1


def test_memo_keeps_at_most_the_configured_companies(make):
    sec = FakeSec()
    src = make(sec, sec_memo_companies=1)
    src.concept_rows(AAPL, LATER)
    src.concept_rows(MSFT, LATER)
    n = len(sec.calls)
    src.concept_rows(AAPL, LATER)  # dropped for Microsoft: built again
    assert len(sec.calls) > n


# --- errors ------------------------------------------------------------------------------------------
def test_companyfacts_404_is_no_facts_and_remembered(make):
    sec = FakeSec()
    src = make(sec)
    with pytest.raises(NoFacts) as info:
        src.concept_rows(NOFACTS, LATER)
    assert info.value.cik == NOFACTS
    n = len(sec.calls)
    with pytest.raises(NoFacts):
        _read(src, filters={"entity_value": NOFACTS})
    assert len(sec.calls) == n


def test_unknown_cik_is_no_facts(make):
    with pytest.raises(NoFacts):
        make().concept_rows("0000000007", LATER)


def test_malformed_companyfacts_is_a_data_fault_not_a_bad_argument(make):
    sec = FakeSec()
    sec.routes[COMPANYFACTS_URL.format(cik=AAPL)] = b"{not json"
    src = make(sec)
    with pytest.raises(LiveDataError) as info:
        src.concept_rows(AAPL, LATER)
    assert not isinstance(info.value, ValueError)
    sec.routes[COMPANYFACTS_URL.format(cik=AAPL)] = FakeSec().routes[COMPANYFACTS_URL.format(cik=AAPL)]
    assert src.concept_rows(AAPL, LATER)[0]  # a failed build is not remembered


@pytest.mark.parametrize("exc", [SecRefused("undeclared"), SecUnavailable("throttled")])
def test_sec_errors_keep_their_type(make, exc):
    class Failing(FakeSec):
        def get_bytes(self, url, *, max_age_s=None):
            raise exc

    with pytest.raises(type(exc)):
        make(Failing()).concept_rows(AAPL, LATER)
    with pytest.raises(type(exc)):
        make(Failing()).entities()


# --- resolution, filings ---------------------------------------------------------------------------
def test_entities_hold_the_tickers_then_the_former_names(make):
    sec = FakeSec()
    src = make(sec)
    idx = resolve.index(src.entities())
    assert idx[AAPL]["tickers"] == ["AAPL"] and idx[AAPL]["names"] == [{"name": "Apple Inc.", "since": None}]
    assert [h["cik"] for h in resolve.resolve(idx, "msft")] == [MSFT]
    src.concept_rows(AAPL, LATER)
    idx = resolve.index(src.entities())
    assert idx[AAPL]["names"] == [
        {"name": "APPLE COMPUTER INC", "since": date(1994, 1, 26)},
        {"name": "Apple Inc.", "since": date(2007, 1, 4)},
    ]  # no stray ticker-file name dated None
    assert idx[AAPL]["current_name"] == "Apple Inc." and idx[AAPL]["tickers"] == ["AAPL"]
    assert [h["cik"] for h in resolve.resolve(idx, "apple computer")] == [AAPL]
    assert idx[MSFT]["names"] == [{"name": "MICROSOFT CORP", "since": None}]
    assert sec.urls().count(TICKERS_URL) == 1  # the tickers file is memoised too


def test_entities_are_copies(make):
    src = make()
    rows = src.entities()
    rows[0]["name"] = "tampered"
    assert src.entities()[0]["name"] != "tampered"


def test_latest_filing_and_filings_read_the_submissions_only(make):
    sec = FakeSec()
    src = make(sec)
    assert src.latest_filing(AAPL).accession == AAPL_E24
    tenk = src.filings(AAPL, forms=["10-K"], period_from=None, period_to=None)
    assert [f.accession for f in tenk] == [AAPL_K24, AAPL_K23]
    dated = src.filings(AAPL, forms=None, period_from=date(2024, 1, 1), period_to=date(2024, 7, 1))
    assert [f.accession for f in dated] == [AAPL_Q24]
    assert sec.urls() == ["https://data.sec.gov/submissions/CIK0000320193.json"]
    assert src.latest_filing("0000000007") is None
    assert src.filings("0000000007", forms=None, period_from=None, period_to=None) == []


def test_the_build_reuses_the_submissions_already_read(make):
    sec = FakeSec()
    src = make(sec)
    src.latest_filing(AAPL)
    src.concept_rows(AAPL, LATER)
    sub = "https://data.sec.gov/submissions/CIK0000320193.json"
    assert sec.urls().count(sub) <= 2  # once for the listing, at most once for the build


def test_malformed_tickers_file_is_a_data_fault(make):
    sec = FakeSec()
    sec.routes[TICKERS_URL] = json.dumps([1, 2]).encode()
    with pytest.raises(LiveDataError):
        make(sec).entities()
