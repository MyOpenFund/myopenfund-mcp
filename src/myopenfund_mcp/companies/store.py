"""The only module that imports my-open-data and company-corpus (layer rule: public interfaces only).

Blocking (DuckDB): the tools call it in a worker thread under a time budget. One my-open-data
``Store`` per process, opened lazily and reopened after a backend failure (``Store.read_asof``
takes a cursor per call, so threads share it safely). A semaphore bounds concurrent calls, and an
abandoned (timed-out) call keeps its slot until it ends, so calls never pile up.

Arguments are checked before any query: filter and column names against ``describe``, each
filter value against its column's declared type (wrong values would otherwise surface as DuckDB
conversion errors, i.e. as a backend failure).

Errors: my-open-data's ``UnknownDataset`` -> :class:`StoreUnknownDataset`, ``ValueError`` (bad
filter, column, limit, naive ``as_of``...) and DuckDB ``DataError`` (a value the checks missed)
-> :class:`StoreBadArgument`; all keep the Store open. Anything else closes the Store and
propagates (the tools map it to ``[backend_unavailable]``), except that an error whose chain
(causes, contexts, args) names the catalogue or its password is replaced by a
:class:`StoreBackendError` with a fixed message. The catalogue never goes in a message or a log.
"""
from __future__ import annotations

import math
import re
import threading
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


class StoreUnknownDataset(LookupError):
    pass


class StoreBadArgument(ValueError):
    pass


class StoreBackendError(RuntimeError):
    """A backend failure whose original message was withheld (it named the catalogue)."""


_PASSWORD_RES = (
    re.compile(r"password\s*=\s*'((?:[^'\\]|\\.)*)'"),  # libpq key='value'
    re.compile(r"password\s*=\s*([^\s']+)"),           # libpq key=value
    re.compile(r"://[^:/@\s]+:([^@\s]+)@"),            # URI user:password@host
)
_DATA_ERROR_MESSAGE = "a filter value does not match its column type"


def _password(catalog: str) -> str | None:
    for rx in _PASSWORD_RES:
        m = rx.search(catalog or "")
        if m and m.group(1):
            return m.group(1)
    return None


def _kind(arrow_type: str) -> str:
    t = arrow_type.lower()
    if t in ("string", "large_string", "utf8", "large_utf8"):
        return "string"
    if t.startswith("date"):
        return "date"
    if t.startswith("timestamp"):
        return "timestamp"
    if t.startswith(("int", "uint")):
        return "integer"
    if t.startswith(("decimal", "double", "float", "halffloat")):
        return "number"
    if t in ("bool", "boolean"):
        return "bool"
    return "other"


_EXPECTS = {"string": "string", "date": "ISO date (YYYY-MM-DD)",
            "timestamp": "timezone-aware ISO datetime", "integer": "integer",
            "number": "numeric", "bool": "boolean", "other": "scalar"}


def _coerce(kind: str, v):
    """The value as the column's type, or raise ValueError (message built by the caller)."""
    if v is None or isinstance(v, (list, tuple, set, frozenset, dict, bytes)):
        raise ValueError
    if kind == "string":
        if not isinstance(v, str):
            raise ValueError
        return v
    if kind == "date":
        if isinstance(v, datetime):
            raise ValueError
        return v if isinstance(v, date) else date.fromisoformat(v)
    if kind == "timestamp":
        d = v if isinstance(v, datetime) else datetime.fromisoformat(v)
        if d.tzinfo is None:
            raise ValueError
        return d
    if kind == "integer":
        if isinstance(v, bool):
            raise ValueError
        if isinstance(v, str) and re.fullmatch(r"[+-]?\d+", v.strip()):
            return int(v)
        if not isinstance(v, int):
            raise ValueError
        return v
    if kind == "number":
        if isinstance(v, bool):
            raise ValueError
        if isinstance(v, str):
            try:
                v = Decimal(v.strip())
            except InvalidOperation:
                raise ValueError from None
        if not isinstance(v, (int, float, Decimal)):
            raise ValueError
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError
        if isinstance(v, Decimal) and not v.is_finite():
            raise ValueError
        return v
    if kind == "bool":
        if not isinstance(v, bool):
            raise ValueError
        return v
    if not isinstance(v, (str, int, float, Decimal, date)):
        raise TypeError
    return v


def _checked(description: dict, dataset: str, filters, columns) -> dict | None:
    """Filters with values coerced to their columns' types; raises StoreBadArgument."""
    types = {c["name"]: c["type"] for c in description["columns"]}
    for name in description.get("added_by_publisher", ()):
        types.setdefault(name, "string")
    if columns is not None:
        if not isinstance(columns, (list, tuple)) or not all(isinstance(c, str) for c in columns):
            raise StoreBadArgument("columns must be a list of column names")
        for c in columns:
            if c not in types:
                raise StoreBadArgument(f"unknown column {c!r} for {dataset}")
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise StoreBadArgument("filters must be an object of column -> value or list of values")
    out = {}
    for col, value in filters.items():
        if not isinstance(col, str) or col not in types:
            raise StoreBadArgument(f"unknown column {col!r} for {dataset}")
        values = list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
        if not values:
            raise StoreBadArgument(f"filter {col!r} needs at least one value")
        kind = _kind(types[col])
        try:
            out[col] = [_coerce(kind, v) for v in values]
        except (ValueError, TypeError):
            raise StoreBadArgument(f"filter {col!r} expects {_EXPECTS[kind]} values") from None
    return out


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
        self._secrets = tuple(x for x in (catalog, _password(catalog)) if x)
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

    def _names_the_catalogue(self, exc: BaseException) -> bool:
        """Whether the catalogue or its password appears anywhere in exc's chain or args."""
        seen, todo = set(), [exc]
        while todo:
            e = todo.pop()
            if e is None or id(e) in seen:
                continue
            seen.add(id(e))
            texts = [str(e), repr(e.args)]
            if any(secret in t for secret in self._secrets for t in texts):
                return True
            todo += [e.__cause__, e.__context__]
        return False

    def _run(self, fn):
        from duckdb import DataError
        from myopendata.reader.api import UnknownDataset

        with self._slots:
            store = None
            try:
                # The factory (a Postgres connect in production) runs under the lock: a slow
                # connect stalls the other callers, each within its own time budget.
                with self._lock:
                    if self._store is None:
                        self._store = self._factory(self._cfg)
                    store = self._store
                return fn(store)
            except UnknownDataset as exc:
                raise StoreUnknownDataset(exc.args[0] if exc.args else "unknown dataset") from None
            except StoreBadArgument:
                raise
            except ValueError as exc:
                raise StoreBadArgument(str(exc)) from None
            except DataError:  # a value the checks missed: the caller's mistake, Store is fine
                raise StoreBadArgument(_DATA_ERROR_MESSAGE) from None
            except Exception as exc:
                if store is not None:
                    self._discard(store)
                if self._names_the_catalogue(exc):
                    raise StoreBackendError(
                        f"store backend error ({type(exc).__name__}; message withheld)") from None
                raise

    def freshness(self) -> dict:
        return self._run(lambda s: s.freshness())

    def entities(self) -> list[dict]:
        """Every ``entity_observations`` row (latest per key), cached until a batch is published.
        Callers get copies: the cache cannot be mutated through them."""
        batches = self.freshness()["batches"]
        cached = self._entities
        if cached is None or cached[0] != batches:
            rows = self._run(lambda s: s.read_asof("entity_observations").to_pylist())
            cached = self._entities = (batches, rows)
        return [dict(r) for r in cached[1]]

    def read(self, dataset: str, *, as_of: datetime, filters: dict | None,
             columns: list[str] | None, limit: int | None, vintages: bool = False) -> list[dict]:
        """Rows knowable at ``as_of``, sorted by the dataset's natural key then ``known_at`` so
        pages are stable across calls; ``limit`` and ``columns`` apply after the sort."""
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                  or limit < 0):
            raise StoreBadArgument("limit must be a non-negative int")

        def go(s):
            description = s.describe(dataset)
            checked = _checked(description, dataset, filters, columns)
            key = [*description["natural_key"], "known_at"]
            rows = s.read_asof(dataset, as_of, filters=checked, vintages=vintages).to_pylist()
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

        if not isinstance(cik, str) or not cik:
            raise StoreBadArgument("cik must be a non-empty string")
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
