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
    first = store.entities()
    cached = store._entities[1]
    assert store.entities() == first and store._entities[1] is cached  # no re-read
    store._entities = (-1, cached)                                    # batches changed
    store.entities()
    assert store._entities[1] is not cached                          # re-read


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
    assert len(vint) == 4


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


# --- fix round 1: argument validation before the query, deep scrubbing ---

@pytest.mark.parametrize("filters", [
    {"valid_date": ["not-a-date"]},
    {"value": ["abc"]},
    {"valid_date": [[1, 2]]},
    {"known_at": ["2025-01-01T00:00:00"]},   # naive timestamp
    {"value": [True]},
    {"value": [float("nan")]},
    {"entity_value": [320193]},              # string column, int value
    {"entity_value": [None]},
    {"entity_value": [{"a": 1}]},
])
def test_wrong_typed_filter_values_are_bad_arguments_and_keep_the_store(store, filters):
    store.freshness()
    first = store._store
    with pytest.raises(StoreBadArgument, match="filter"):
        store.read("fundamentals", as_of=LATER, filters=filters, columns=None, limit=None)
    assert store._store is first


def test_filters_accept_iso_strings_and_numeric_strings(store):
    by_str = store.read("fundamentals", as_of=LATER,
                        filters={"valid_date": "2024-09-28", "value": ["391036000000"],
                                 "known_at": ["2025-10-31T20:00:00+00:00"]},
                        columns=["entity_value", "value"], limit=None)
    assert by_str == [{"entity_value": AAPL, "value": 391036000000}]


def test_filters_on_publisher_columns(store):
    eid = store.read("fundamentals", as_of=LATER, filters={"entity_value": AAPL},
                     columns=["entity_id"], limit=1)[0]["entity_id"]
    rows = store.read("fundamentals", as_of=LATER, filters={"entity_id": [eid]},
                      columns=["entity_value"], limit=None)
    assert {r["entity_value"] for r in rows} == {AAPL}


@pytest.mark.parametrize("kwargs", [
    {"filters": {"nope": ["x"]}},
    {"filters": ["entity_value"]},
    {"columns": "value"},
    {"columns": ["nope"]},
])
def test_names_and_shapes_are_checked_before_the_query(store, kwargs):
    args = {"filters": None, "columns": None, "limit": None, **kwargs}
    with pytest.raises(StoreBadArgument):
        store.read("fundamentals", as_of=LATER, **args)


def test_a_duckdb_data_error_is_a_bad_argument_without_discarding(store):
    import duckdb

    store.freshness()
    first = store._store

    def boom(s):
        raise duckdb.ConversionException("Could not convert string 'x' to DATE")

    with pytest.raises(StoreBadArgument, match="^a filter value does not match its column type$"):
        store._run(boom)
    assert store._store is first


def test_concept_rows_reject_a_non_string_cik(store):
    with pytest.raises(StoreBadArgument):
        store.concept_rows(320193, LATER)


def test_entities_cannot_be_mutated_by_callers(store):
    store.entities()[0]["name"] = "tampered"
    store.entities().clear()
    assert store.entities() and "tampered" not in {r["name"] for r in store.entities()}


SECRET = "postgres:dbname=lake host=db user=lake_reader password=hunter2"


@pytest.mark.parametrize("make", [
    lambda: _chained(RuntimeError("boom"), RuntimeError(f"ATTACH {SECRET}")),
    lambda: RuntimeError("auth failed for password hunter2"),
    lambda: RuntimeError("x", {"dsn": SECRET}),
    lambda: _context(RuntimeError("outer"), OSError("password=hunter2 rejected")),
])
def test_scrubbing_walks_the_whole_chain(tmp_path, make):
    def factory(c):
        raise make()

    s = CompaniesStore(str(tmp_path), SECRET, store_factory=factory)
    with pytest.raises(Exception) as info:
        s.freshness()
    from myopenfund_mcp.companies.store import StoreBackendError

    assert type(info.value) is StoreBackendError
    assert "hunter2" not in str(info.value) and info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_an_unrelated_backend_error_propagates_unchanged(tmp_path):
    def factory(c):
        raise _chained(RuntimeError("boom"), OSError("connection refused"))

    s = CompaniesStore(str(tmp_path), SECRET, store_factory=factory)
    with pytest.raises(RuntimeError, match="^boom$"):
        s.freshness()


def _chained(outer, cause):
    try:
        raise outer from cause
    except Exception as exc:  # noqa: BLE001
        return exc


def _context(outer, inner):
    try:
        try:
            raise inner
        except Exception:  # noqa: BLE001
            raise outer  # implicit __context__ on purpose
    except Exception as exc:  # noqa: BLE001
        return exc


# --- final-review fixes: Arrow-side sort/slice/project, Arrow errors are backend faults ---

class _TableStore:
    """A minimal Store serving one in-memory table (natural key ``k``, nullable on purpose)."""

    def __init__(self, table):
        self.table, self.reads = table, 0

    def describe(self, dataset):
        return {"natural_key": ["k"], "added_by_publisher": [],
                "columns": [{"name": f.name, "type": str(f.type)} for f in self.table.schema]}

    def read_asof(self, dataset, as_of, *, filters=None, vintages=False):
        self.reads += 1
        return self.table

    def close(self):
        pass


def test_nulls_in_a_sort_key_go_last_and_ties_break_on_known_at(tmp_path):
    import pyarrow as pa

    t = pa.table({
        "k": ["b", None, "a", "b", None, "a"],
        "known_at": [datetime(2025, 1, 2, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
                     datetime(2025, 1, 3, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
                     None, datetime(2025, 1, 1, tzinfo=UTC)],
        "v": [1, 2, 3, 4, 5, 6],
    })
    s = CompaniesStore(str(tmp_path), "x", store_factory=lambda c: _TableStore(t))
    rows = s.read("d", as_of=LATER, filters=None, columns=None, limit=None)
    assert [r["v"] for r in rows] == [6, 3, 4, 1, 2, 5]  # None keys last; None known_at last
    page = s.read("d", as_of=LATER, filters=None, columns=["v"], limit=4)
    assert page == [{"v": 6}, {"v": 3}, {"v": 4}, {"v": 1}]


ARROW_SECRET = "postgres:dbname=lake host=db user=lake_reader password=hunter2"


@pytest.mark.parametrize("message", [f"cannot read {ARROW_SECRET}", "password=hunter2 rejected"])
def test_an_arrow_error_naming_the_catalogue_is_withheld(tmp_path, message):
    import pyarrow as pa

    from myopenfund_mcp.companies.store import StoreBackendError

    def factory(c):
        raise pa.lib.ArrowInvalid(message)

    s = CompaniesStore(str(tmp_path), ARROW_SECRET, store_factory=factory)
    with pytest.raises(StoreBackendError) as info:
        s.freshness()
    assert "hunter2" not in str(info.value) and "lake_reader" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_a_plain_arrow_error_is_a_backend_fault_not_a_bad_argument(tmp_path):
    import pyarrow as pa

    class Broken(_TableStore):
        def read_asof(self, dataset, as_of, *, filters=None, vintages=False):
            raise pa.lib.ArrowInvalid("Parquet magic bytes not found")

    opened = []

    def factory(c):
        opened.append(1)
        return Broken(pa.table({"k": ["a"], "known_at": [LATER]}))

    s = CompaniesStore(str(tmp_path), "x", store_factory=factory)
    with pytest.raises(pa.lib.ArrowInvalid) as info:
        s.read("d", as_of=LATER, filters=None, columns=None, limit=None)
    assert not isinstance(info.value, StoreBadArgument)
    assert s._store is None                              # discarded: the next call reopens
    with pytest.raises(pa.lib.ArrowInvalid):
        s.read("d", as_of=LATER, filters=None, columns=None, limit=None)
    assert len(opened) == 2


@pytest.mark.parametrize("error", [ValueError("inconsistent units in stored facts"),
                                   "duckdb-data-error"])
def test_a_selection_failure_is_not_the_callers(store, monkeypatch, error):
    import duckdb
    from company_corpus.concepts import selection

    def boom(facts, positions):
        raise duckdb.ConversionException("Could not convert") if error == "duckdb-data-error" else error

    monkeypatch.setattr(selection, "run_selection", boom)
    with pytest.raises(Exception) as info:
        store.concept_rows(AAPL, LATER)
    assert not isinstance(info.value, StoreBadArgument)


def test_an_argument_value_error_naming_the_catalogue_is_withheld(tmp_path):
    from myopenfund_mcp.companies.store import StoreBackendError

    def factory(c):
        raise ValueError(f"bad option in {ARROW_SECRET}")

    s = CompaniesStore(str(tmp_path), ARROW_SECRET, store_factory=factory)
    with pytest.raises(StoreBackendError) as info:
        s.freshness()
    assert "hunter2" not in str(info.value)


def test_missing_company_corpus_is_an_import_error_at_construction(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "company_corpus.concepts.store_summaries", None)
    with pytest.raises(ImportError):
        CompaniesStore(str(tmp_path), "x")


class _CountingStore:
    """Delegates to a real store, counting describe() calls; ``hidden`` datasets read as
    unpublished (no columns) until removed."""

    def __init__(self, inner, calls, hidden):
        self._inner, self._calls, self._hidden = inner, calls, hidden

    def describe(self, dataset):
        self._calls.append(dataset)
        d = self._inner.describe(dataset)
        return {**d, "columns": []} if dataset in self._hidden else d

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _counting(cfg, calls, hidden=frozenset()):
    from myopendata.reader import Store
    return CompaniesStore(str(cfg.data_root), cfg.catalog,
                          store_factory=lambda c: _CountingStore(Store.open(c), calls, hidden))


def test_a_published_dataset_is_described_once(cfg):
    calls = []
    s = _counting(cfg, calls)
    try:
        first = s.concept_rows(AAPL, LATER)
        assert len(s.concept_rows(AAPL, LATER)[0]) == len(first[0]) > 0
        s.entities()
        s.entities()
        assert calls.count("fundamentals") == 1
        assert calls.count("entity_observations") == 1
    finally:
        s.close()


def test_an_unpublished_dataset_is_rechecked_so_publication_is_seen(cfg):
    calls, hidden = [], {"fundamentals"}
    s = _counting(cfg, calls, hidden)
    try:
        assert s.concept_rows(AAPL, LATER) == ([], [])
        assert s.concept_rows(AAPL, LATER) == ([], [])
        assert calls.count("fundamentals") == 2
        hidden.clear()  # the dataset is published now
        rows, _ = s.concept_rows(AAPL, LATER)
        assert rows
        s.concept_rows(AAPL, LATER)
        assert calls.count("fundamentals") == 3  # described until seen published, then remembered
    finally:
        s.close()


# --- a Store is never closed under a call in flight (DuckDB refuses, or the call's next query fails) ---

class _BusyStore:
    """Like a DuckDB connection: ``close`` refuses while a call runs, a closed Store fails calls."""

    def __init__(self, gate, inside):
        self.gate, self.inside = gate, inside
        self.active = 0
        self.closed = 0

    def _check(self):
        if self.closed:
            raise RuntimeError("Connection already closed")

    def freshness(self):                  # blocks between two queries until the gate opens
        self._check()
        self.active += 1
        try:
            self.inside.set()
            self.gate.wait(5)
            self._check()
            return {"batches": 0, "last_published_at": None}
        finally:
            self.active -= 1

    def datasets(self):                   # a backend failure
        raise RuntimeError("backend down")

    def close(self):
        if self.active:
            raise RuntimeError("connection is busy with another call")
        self.closed += 1


def _busy_store(tmp_path, gate, inside):
    opened = []

    def factory(c):
        opened.append(_BusyStore(gate, inside))
        return opened[-1]

    return CompaniesStore(str(tmp_path), "x", max_concurrency=2, store_factory=factory), opened


def _in_flight(s, inside):
    out = {}

    def call():
        try:
            out["result"] = s.freshness()
        except Exception as exc:  # noqa: BLE001 — the assertion below shows it
            out["result"] = exc

    t = threading.Thread(target=call)
    t.start()
    assert inside.wait(5)
    return t, out


def test_a_discard_leaves_calls_in_flight_their_store(tmp_path):
    gate, inside = threading.Event(), threading.Event()
    s, opened = _busy_store(tmp_path, gate, inside)
    t, out = _in_flight(s, inside)
    with pytest.raises(RuntimeError, match="backend down"):
        s.datasets()                      # another call fails: its Store is discarded...
    old = opened[0]
    assert s._store is None and old.closed == 0   # ...detached, not closed under the call
    gate.set()
    t.join(5)
    assert out["result"] == {"batches": 0, "last_published_at": None}  # the call completes
    assert old.closed == 1                # closed by the last call to leave it
    assert s.freshness() == {"batches": 0, "last_published_at": None}
    assert len(opened) == 2 and opened[1].closed == 0   # the next call opened a new Store
    s.close()
    assert opened[1].closed == 1


def test_close_leaves_calls_in_flight_their_store(tmp_path):
    gate, inside = threading.Event(), threading.Event()
    s, opened = _busy_store(tmp_path, gate, inside)
    t, out = _in_flight(s, inside)
    s.close()                             # shutdown with a call in flight (e.g. an abandoned one)
    assert s._store is None and opened[0].closed == 0
    gate.set()
    t.join(5)
    assert out["result"] == {"batches": 0, "last_published_at": None}
    assert opened[0].closed == 1


def test_close_when_idle_closes_at_once(tmp_path):
    gate, inside = threading.Event(), threading.Event()
    gate.set()
    s, opened = _busy_store(tmp_path, gate, inside)
    s.freshness()
    s.close()
    assert opened[0].closed == 1
    s.close()                             # idempotent
    assert opened[0].closed == 1


def test_a_discard_mid_call_on_the_real_store(cfg):
    """my-open-data's Store: a discard between two queries of another call must not close the
    DuckDB connection that call still uses."""
    s = CompaniesStore(str(cfg.data_root), cfg.catalog)
    between, go_on, out = threading.Event(), threading.Event(), {}

    def two_queries(st):
        st.datasets()
        between.set()
        go_on.wait(5)
        return st.datasets()

    def call():
        try:
            out["result"] = s._run(two_queries)
        except Exception as exc:  # noqa: BLE001
            out["result"] = exc

    t = threading.Thread(target=call)
    t.start()
    try:
        assert between.wait(5)
        s._discard(s._store)              # as after another thread's backend failure
        go_on.set()
        t.join(5)
        assert isinstance(out["result"], list) and out["result"]
    finally:
        go_on.set()
        s.close()


def _spy_summaries(monkeypatch):
    import company_corpus.concepts.store_summaries as mod

    seen = []
    real = mod.summaries_from_rows

    def spy(*args, **kwargs):
        seen.append(kwargs.get("q4_rule"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "summaries_from_rows", spy)
    return seen


def test_summaries_use_the_corpus_q4_rule_by_default(cfg, monkeypatch):
    seen = _spy_summaries(monkeypatch)
    s = CompaniesStore(str(cfg.data_root), cfg.catalog)
    try:
        rows, vint = s.concept_rows(AAPL, LATER)
        s.summaries(AAPL, "Apple Inc.", rows, vint)
    finally:
        s.close()
    assert seen == ["ytd9"]


def test_summaries_use_the_configured_q4_rule(cfg, monkeypatch):
    seen = _spy_summaries(monkeypatch)
    s = CompaniesStore(str(cfg.data_root), cfg.catalog, q4_rule="sum_of_quarters")
    try:
        rows, vint = s.concept_rows(AAPL, LATER)
        s.summaries(AAPL, "Apple Inc.", rows, vint)
    finally:
        s.close()
    assert seen == ["sum_of_quarters"]


def test_an_unknown_q4_rule_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="q4_rule"):
        CompaniesStore(str(tmp_path), "x", q4_rule="annual")
