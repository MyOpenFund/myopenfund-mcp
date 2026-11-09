"""The only module that imports my-open-data and company-corpus (layer rule: public interfaces only).

Blocking (DuckDB): the tools call it in a worker thread under a time budget. One my-open-data
``Store`` per process, opened lazily and reopened after a backend failure (``Store.read_asof``
takes a cursor per call, so threads share it safely). A semaphore bounds concurrent calls, and an
abandoned (timed-out) call keeps its slot until it ends, so calls never pile up.

Errors: my-open-data's ``UnknownDataset`` -> :class:`StoreUnknownDataset`, ``ValueError`` (bad
filter, column, limit, naive ``as_of``...) -> :class:`StoreBadArgument`; both keep the Store open.
Anything else closes the Store and propagates (the tools map it to ``[backend_unavailable]``),
except that an error whose message names the catalogue (which may hold credentials) is replaced
by a :class:`StoreBackendError` without it. The catalogue string never goes in a message or a log.
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path


class StoreUnknownDataset(LookupError):
    pass


class StoreBadArgument(ValueError):
    pass


class StoreBackendError(RuntimeError):
    """A backend failure whose original message was withheld (it named the catalogue)."""


def _open(cfg):
    from myopendata.reader import Store

    return Store.open(cfg)


def _sort_key(key: list[str]):
    # None last within a column; a column's non-null values share one type, so they compare.
    return lambda r: tuple((r[k] is None, "" if r[k] is None else r[k]) for k in key)


class CompaniesStore:
    def __init__(self, data_root: str, catalog: str, *, max_concurrency: int = 2,
                 store_factory=None):
        from myopendata.config import StoreConfig  # ImportError here = [companies] extra missing

        self._cfg = StoreConfig(data_root=Path(data_root), catalog=catalog)
        self._catalog = catalog
        self._factory = store_factory or _open
        self._store = None
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._entities: tuple[int, list[dict]] | None = None

    def __repr__(self) -> str:
        return f"CompaniesStore(data_root={str(self._cfg.data_root)!r})"

    def close(self) -> None:
        with self._lock:
            store, self._store = self._store, None
        if store is not None:
            store.close()

    def _discard(self, store) -> None:
        """Close the Store that failed, unless another thread already replaced it."""
        with self._lock:
            if self._store is not store:
                return
            self._store = None
        try:
            store.close()
        except Exception:  # noqa: BLE001, S110 — it is already broken; the next call reopens
            pass

    def _scrub(self, exc: BaseException) -> BaseException | None:
        if self._catalog and self._catalog in str(exc):
            return StoreBackendError(f"store backend error ({type(exc).__name__}; "
                                     "message withheld: it named the catalogue)")
        return None

    def _run(self, fn):
        from myopendata.reader.api import UnknownDataset

        with self._slots:
            store = None
            try:
                with self._lock:
                    if self._store is None:
                        self._store = self._factory(self._cfg)
                    store = self._store
                return fn(store)
            except UnknownDataset as exc:
                raise StoreUnknownDataset(exc.args[0] if exc.args else "unknown dataset") from None
            except ValueError as exc:
                raise StoreBadArgument(str(exc)) from None
            except Exception as exc:
                if store is not None:
                    self._discard(store)
                clean = self._scrub(exc)
                if clean is not None:
                    raise clean from None
                raise

    def freshness(self) -> dict:
        return self._run(lambda s: s.freshness())

    def entities(self) -> list[dict]:
        """Every ``entity_observations`` row (latest per key), cached until a batch is published."""
        batches = self.freshness()["batches"]
        cached = self._entities
        if cached is not None and cached[0] == batches:
            return cached[1]
        rows = self._run(lambda s: s.read_asof("entity_observations").to_pylist())
        self._entities = (batches, rows)
        return rows

    def read(self, dataset: str, *, as_of: datetime, filters: dict | None,
             columns: list[str] | None, limit: int | None, vintages: bool = False) -> list[dict]:
        """Rows knowable at ``as_of``, sorted by the dataset's natural key then ``known_at`` so
        pages are stable across calls; ``limit`` and ``columns`` apply after the sort."""
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                  or limit < 0):
            raise StoreBadArgument("limit must be a non-negative int")

        def go(s):
            key = [*s.describe(dataset)["natural_key"], "known_at"]
            table = s.read_asof(dataset, as_of, filters=filters, vintages=vintages)
            unknown = [c for c in columns or () if c not in table.column_names]
            if unknown:
                raise ValueError(f"unknown column {unknown[0]!r} for {dataset}")
            rows = table.to_pylist()
            rows.sort(key=_sort_key(key))
            if limit is not None:
                rows = rows[:limit]
            if columns:
                rows = [{c: r[c] for c in columns} for r in rows]
            return rows

        return self._run(go)

    def datasets(self) -> list[dict]:
        return self._run(lambda s: s.datasets())

    def describe(self, dataset: str) -> dict:
        return self._run(lambda s: s.describe(dataset))

    def concept_rows(self, cik: str, as_of: datetime) -> tuple[list[dict], list[dict]]:
        """``(selection_rows, vintages)`` for one issuer: company-corpus's concept selection on
        its ``fundamentals`` vintages and ``filing_statements`` knowable at ``as_of``, and those
        vintages (``summaries`` needs them for publication dates). ``([], [])`` without
        fundamentals."""
        from company_corpus.concepts.selection import run_selection

        key = {"entity_scheme": ["cik"], "entity_value": [cik]}

        def go(s):
            facts = s.read_asof("fundamentals", as_of, filters=key, vintages=True)
            if facts.num_rows == 0:
                return [], []
            positions = s.read_asof("filing_statements", as_of, filters=key)
            return run_selection(facts, positions), facts.to_pylist()

        return self._run(go)

    def summaries(self, cik: str, name: str, rows: list[dict], vintages: list[dict]) -> list:
        """company-corpus period summaries (values, ratios, TTM) of ``concept_rows``' output.

        The store has no SIC, so the sector is unknown (every ``sector_relevant`` is None)."""
        from company_corpus.concepts.store_summaries import summaries_from_rows

        with self._slots:
            try:
                return summaries_from_rows(rows, company=name or cik, sector_known=False,
                                           vintages=vintages)
            except ValueError as exc:
                raise StoreBadArgument(str(exc)) from None

    def concept_catalog(self) -> list[dict]:
        from company_corpus.concepts.store_summaries import concept_catalog

        return concept_catalog()

    def metric_catalog(self) -> list[dict]:
        from company_corpus.concepts.store_summaries import metric_catalog

        return metric_catalog()
