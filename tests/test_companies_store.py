import threading
from datetime import UTC, date, datetime

import pytest
from companies_fixture import AAPL, AAPL_10K_2024, JPM, N_FUNDAMENTALS_LATEST, NWG, REV, seed

from myopenfund_mcp.companies.store import CompaniesStore, StoreBadArgument, StoreUnknownDataset

LATER = datetime(2026, 1, 1, tzinfo=UTC)
BEFORE_RESTATEMENT = datetime(2025, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return seed(tmp_path_factory.mktemp("companies"))


@pytest.fixture
def store(cfg):
    s = CompaniesStore(str(cfg.data_root), cfg.catalog)
    yield s
    s.close()


def test_freshness(store):
    f = store.freshness()
    assert f["batches"] == 3 and f["last_published_at"] is not None


def test_entities_carry_names_and_tickers(store):
    rows = store.entities()
    assert {r["id_value"] for r in rows if r["subject_value"] == AAPL} == {AAPL, "AAPL"}
    assert {r["subject_value"] for r in rows if "APPLE" in (r["name"] or "").upper()} >= {AAPL}
    assert len({r["subject_value"] for r in rows if "APPLE" in (r["name"] or "").upper()}) == 2


def test_entities_are_cached_until_a_new_batch(store):
    assert store.entities() is store.entities()


def _rev(store, as_of):
    rows = store.read("fundamentals", as_of=as_of,
                      filters={"entity_value": [AAPL], "original_tag": [REV],
                               "valid_date": [date(2024, 9, 28)]},
                      columns=None, limit=None)
    return [r["value"] for r in rows]


def test_read_is_point_in_time(store):
    assert _rev(store, BEFORE_RESTATEMENT) == [391035000000]
    assert _rev(store, LATER) == [391036000000]


def test_read_vintages_keep_both_values(store):
    rows = store.read("fundamentals", as_of=LATER,
                      filters={"entity_value": [AAPL], "original_tag": [REV],
                               "valid_date": [date(2024, 9, 28)]},
                      columns=["value", "known_at"], limit=None, vintages=True)
    assert [r["value"] for r in rows] == [391035000000, 391036000000]  # sorted by known_at
    assert set(rows[0]) == {"value", "known_at"}


def test_read_order_is_stable(store):
    a = store.read("fundamentals", as_of=LATER, filters=None, columns=None, limit=None)
    b = store.read("fundamentals", as_of=LATER, filters=None, columns=None, limit=None)
    assert a == b and len(a) == N_FUNDAMENTALS_LATEST
    key = [(r["entity_scheme"], r["entity_value"], r["original_tag"], r["valid_date"],
            r["frequency"], r["basis"], r["source_code"]) for r in a]
    assert key == sorted(key)


def test_limit_and_columns_apply_after_the_sort(store):
    full = store.read("fundamentals", as_of=LATER, filters=None, columns=None, limit=None)
    page = store.read("fundamentals", as_of=LATER, filters=None, columns=["value"], limit=3)
    assert page == [{"value": r["value"]} for r in full[:3]]
    assert store.read("fundamentals", as_of=LATER, filters=None, columns=None, limit=0) == []


def test_unknown_column_is_a_bad_argument(store):
    with pytest.raises(StoreBadArgument, match="unknown column 'x' for fundamentals"):
        store.read("fundamentals", as_of=LATER, filters=None, columns=["value", "x"], limit=None)


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_bad_limit_is_a_bad_argument(store, limit):
    with pytest.raises(StoreBadArgument):
        store.read("fundamentals", as_of=LATER, filters=None, columns=None, limit=limit)


def test_concept_rows_give_revenue_and_vintages(store):
    rows, vint = store.concept_rows(AAPL, LATER)
    rev = [r for r in rows if r["concept"] == "revenue" and r["frequency"] == "annual"
           and r["valid_date"] == date(2024, 9, 28)]
    assert rev and rev[0]["original_tag"] == REV and rev[0]["value"] == 391036000000
    assert all(v["entity_value"] == AAPL for v in vint)
    assert len([v for v in vint if v["original_tag"] == REV]) == 3  # restatement kept


def test_concept_rows_of_a_bank_without_positions(store):
    rows, vint = store.concept_rows(JPM, LATER)
    assert {r["concept"] for r in rows} >= {"revenue", "net_income"}
    assert len(vint) == 3


def test_an_issuer_without_fundamentals_gives_no_rows(store):
    assert store.concept_rows(NWG, LATER) == ([], [])
    assert store.summaries(NWG, "NatWest Group plc", [], []) == []


def test_summaries_give_ratios_with_unknown_sector(store):
    rows, vint = store.concept_rows(AAPL, LATER)
    s = store.summaries(AAPL, "Apple Inc.", rows, vint)
    annual = next(x for x in s if x.frequency == "annual" and x.period_end == date(2024, 9, 28))
    assert annual.derived["net_margin"]["sector_relevant"] is True  # not sector-sensitive
    # sector-sensitive metric, but no SIC in the store: relevance unknown, never asserted
    assert annual.derived["asset_turnover"]["sector_relevant"] is None


def test_summaries_date_a_restated_period_by_its_first_filing(store):
    rows, vint = store.concept_rows(AAPL, LATER)
    s = store.summaries(AAPL, "Apple Inc.", rows, vint)
    fy24 = next(x for x in s if x.frequency == "annual" and x.period_end == date(2024, 9, 28))
    assert fy24.publication_date == date(2024, 11, 1)  # not the 2025 restatement
    assert fy24.accession == AAPL_10K_2024


def test_summaries_name_falls_back_to_the_cik(store):
    rows, vint = store.concept_rows(AAPL, LATER)
    assert store.summaries(AAPL, "", rows, vint)[0].company == AAPL


def test_summaries_refuse_foreign_vintages(store):
    rows, _ = store.concept_rows(AAPL, LATER)
    _, jpm_vint = store.concept_rows(JPM, LATER)
    with pytest.raises(StoreBadArgument):
        store.summaries(AAPL, "Apple Inc.", rows, jpm_vint)


def test_errors_are_typed(store):
    with pytest.raises(StoreUnknownDataset, match="unknown dataset 'nope'"):
        store.describe("nope")
    with pytest.raises(StoreUnknownDataset):
        store.read("nope", as_of=LATER, filters=None, columns=None, limit=None)
    with pytest.raises(StoreBadArgument):
        store.read("fundamentals", as_of=LATER, filters={"not_a_column": "x"}, columns=None,
                   limit=None)
    with pytest.raises(StoreBadArgument):  # naive as_of
        store.read("fundamentals", as_of=datetime(2026, 1, 1), filters=None, columns=None,
                   limit=None)
    first = store._store
    store.freshness()
    assert store._store is first  # argument errors keep the Store open


def test_datasets_and_describe(store):
    assert {d["name"] for d in store.datasets()} == {
        "entity_observations", "fundamentals", "filing_statements"}
    assert store.describe("fundamentals")["natural_key"][:2] == ["entity_scheme", "entity_value"]


def test_catalogs(store):
    assert "revenue" in {c["key"] for c in store.concept_catalog()}
    assert "roe_ttm" in {m["key"] for m in store.metric_catalog()}


def test_reopens_after_a_backend_failure(cfg):
    opened = []

    def factory(c):
        from myopendata.reader import Store
        opened.append(1)
        return Store.open(c)

    s = CompaniesStore(str(cfg.data_root), cfg.catalog, store_factory=factory)
    try:
        s.freshness()
        s._store.con.close()          # simulate a dropped connection
        with pytest.raises(Exception):  # noqa: B017 — whatever the backend raises propagates
            s.freshness()
        assert s._store is None
        s.freshness()                 # reopened
        assert len(opened) == 2
    finally:
        s.close()


def test_a_backend_error_never_carries_the_catalogue(tmp_path):
    secret = "postgres:dbname=lake host=db user=lake_reader password=hunter2"

    def factory(c):
        raise RuntimeError(f"could not connect to {c.catalog}")

    s = CompaniesStore(str(tmp_path), secret, store_factory=factory)
    with pytest.raises(Exception) as info:
        s.freshness()
    assert "hunter2" not in str(info.value) and info.value.__cause__ is None
    assert info.value.__suppress_context__
    assert "hunter2" not in repr(s)


class _SlowStore:
    def __init__(self, gate, inside):
        self.gate, self.inside = gate, inside

    def freshness(self):
        self.inside.append(1)
        self.gate.wait(5)
        return {"batches": 0, "last_published_at": None}

    def close(self):
        pass


def test_concurrency_is_bounded(tmp_path):
    gate, inside = threading.Event(), []
    s = CompaniesStore(str(tmp_path), "x", max_concurrency=1,
                       store_factory=lambda c: _SlowStore(gate, inside))
    threads = [threading.Thread(target=s.freshness) for _ in range(2)]
    for t in threads:
        t.start()
    threading.Event().wait(0.3)
    assert len(inside) == 1           # the second call waits for the slot
    gate.set()
    for t in threads:
        t.join(5)
    assert len(inside) == 2
