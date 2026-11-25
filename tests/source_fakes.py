"""In-memory fakes of the company store and the live SEC source, for the source router and the tools.

Both record every call (``calls``), so a test can assert which source was (never) touched; ``fail``
makes a method raise. Rows carry only the columns the tools read.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass

UTC = dt.UTC
APPLE, MSFT, JPM = "0000320193", "0000789019", "0000019617"
NAMES = {APPLE: ("Apple Inc.", "AAPL"), MSFT: ("MICROSOFT CORP", "MSFT"), JPM: ("JPMORGAN CHASE & CO", "JPM")}
K24, E24, K23 = "0000320193-24-000123", "0000320193-24-000130", "0000320193-23-000106"
ACCEPTED_K24 = dt.datetime(2024, 11, 1, 20, 6, 22, tzinfo=UTC)
ACCEPTED_E24 = dt.datetime(2024, 11, 5, 21, 0, tzinfo=UTC)
ACCEPTED_K23 = dt.datetime(2023, 11, 3, 20, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Ref:
    """The FilingRef fields the router reads."""

    accession: str
    accepted: dt.datetime | None
    form: str = "10-K"


#: Apple's filings at SEC, newest first: an 8-K after the FY2024 10-K
APPLE_REFS = [Ref(E24, ACCEPTED_E24, "8-K"), Ref(K24, ACCEPTED_K24), Ref(K23, ACCEPTED_K23)]


def entity(cik: str, name: str | None = None, ticker: str | None = None) -> dict:
    default_name, default_ticker = NAMES.get(cik, (None, None))
    return {
        "subject_scheme": "cik",
        "subject_value": cik,
        "entity_id": f"e-{cik}",
        "name": name or default_name,
        "id_scheme": "ticker",
        "id_value": ticker or default_ticker,
        "country": "US",
        "valid_date": "2020-01-01",
    }


def fact(cik: str, value: int, *, known=ACCEPTED_K24, accn=K24) -> dict:
    return {
        "entity_scheme": "cik",
        "entity_value": cik,
        "original_tag": "us-gaap:Revenues",
        "frequency": "annual",
        "valid_date": dt.date(2024, 9, 28),
        "value": value,
        "unit": "USD",
        "filing_ref": accn,
        "known_at": known,
        "batch_id": "b1",
    }


def concept_row(cik: str, value: int) -> dict:
    return {
        "valid_date": dt.date(2024, 9, 28),
        "frequency": "annual",
        "concept": "revenue",
        "value": value,
        "unit": "USD",
        "original_tag": "us-gaap:Revenues",
        "filing_ref": K24,
        "source_doc_id": None,
        "known_at": ACCEPTED_K24,
        "location": "statement",
        "conflict": False,
        "dictionary_version": "2.1.0",
    }


class _Recording:
    def __init__(self, fail: dict | None = None):
        self.calls: list[str] = []
        self.fail = dict(fail or {})  # method name (or "method:cik") -> exception to raise
        self.closed = False
        self._lock = threading.Lock()

    def _enter(self, method: str, cik: str | None = None) -> None:
        with self._lock:
            self.calls.append(method)
        exc = self.fail.get(f"{method}:{cik}") or self.fail.get(method)
        if exc is not None:
            raise exc

    def concept_catalog(self) -> list[dict]:
        return [{"key": "revenue", "label": "Revenue"}]

    def metric_catalog(self) -> list[dict]:
        return [{"key": "net_margin"}]

    def summaries(self, cik, name, rows, vintages) -> list:
        self._enter("summaries", cik)
        return []

    def close(self) -> None:
        self.closed = True


class FakeStore(_Recording):
    """``ciks``: the companies held. ``filings``: cik -> [(accession, known_at)] (None = the
    ``filings`` dataset is not published). ``positions``: cik -> accessions with filing_statements
    rows. ``value``: the revenue the store holds. ``batches``: the publication count."""

    def __init__(self, ciks=(APPLE,), *, filings=..., positions=None, value=100, fail=None):
        super().__init__(fail)
        self.ciks = list(ciks)
        self.filings = (
            {APPLE: [(K23, ACCEPTED_K23), (K24, ACCEPTED_K24), (E24, ACCEPTED_E24)]}
            if filings is ...
            else filings
        )
        self.positions = {APPLE: {K24, K23}} if positions is None else positions
        self.value = value
        self.batches = 1

    def freshness(self) -> dict:
        self._enter("freshness")
        return {"last_published": ACCEPTED_E24, "batches": self.batches}

    def datasets(self) -> list[dict]:
        self._enter("datasets")
        names = ["entity_observations", "fundamentals", "filing_statements"]
        return [{"name": n} for n in names + (["filings"] if self.filings is not None else [])]

    def entities(self) -> list[dict]:
        self._enter("entities")
        return [entity(c) for c in self.ciks]

    def read(self, dataset, *, as_of, filters, columns, limit, vintages=False) -> list[dict]:
        ciks = (filters or {}).get("entity_value", [None])
        self._enter(f"read:{dataset}", ciks[0] if len(ciks) == 1 else None)
        if dataset == "filing_statements":
            wanted = set(filters.get("filing_ref", []))
            return [{"filing_ref": a} for a in sorted(self.positions.get(ciks[0], set()) & wanted)][:limit]
        return [fact(c, self.value) for c in ciks if c in self.ciks]

    def known_filings(self, cik, *, dataset="filings") -> list[dict]:
        self._enter(f"known_filings:{dataset}", cik)
        if dataset == "filings":
            return [{"filing_ref": a, "known_at": k} for a, k in (self.filings or {}).get(cik, [])]
        return [{"filing_ref": K24, "known_at": ACCEPTED_K24}] if cik in self.ciks else []

    def concept_rows(self, cik, as_of):  # the store's signature as the existing tests patch it
        self._enter("concept_rows", cik)
        return [concept_row(cik, self.value)], []


class FakeLive(_Recording):
    """``refs``: cik -> its filings at SEC (``Ref``, any order). ``known``: extra resolver rows.
    ``value``: the revenue SEC gives. ``budgets``: the ``budget_s`` each read received."""

    report_forms = ("10-K", "10-Q", "8-K")
    positions_forms = ("10-K", "10-Q", "20-F", "40-F")

    def __init__(self, *, refs=None, known=(), value=200, fail=None):
        super().__init__(fail)
        self.refs = {APPLE: list(APPLE_REFS)} if refs is None else refs
        self.known = list(known)
        self.value = value
        self.periods: list[tuple] = []
        self.budgets: list[float | None] = []

    def entities(self) -> list[dict]:
        self._enter("entities")
        return [entity(c) for c in NAMES] + [dict(r) for r in self.known]

    def filings(self, cik, *, forms, period_from, period_to) -> list:
        self._enter("filings", cik)
        wanted = {f.upper() for f in forms} if forms is not None else None
        return [
            r
            for r in self.refs.get(cik, [])
            if (wanted is None or r.form.upper() in wanted or r.form.upper().removesuffix("/A") in wanted)
            and (period_from is None or (getattr(r, "report_date", None) or dt.date.min) >= period_from)
            and (period_to is None or (getattr(r, "report_date", None) or dt.date.max) <= period_to)
        ]

    def latest_filing(self, cik, *, forms=None):
        found = self.filings(cik, forms=forms, period_from=None, period_to=None)
        return max(found, key=lambda r: r.accepted) if found else None

    def read(self, dataset, *, as_of, filters, columns, limit, vintages=False, budget_s=None) -> list[dict]:
        cik = filters["entity_value"][0]
        self._enter(f"read:{dataset}", cik)
        self.budgets.append(budget_s)
        return [fact(c, self.value) for c in filters["entity_value"]]

    def concept_rows(self, cik, as_of, *, period_from=None, period_to=None, budget_s=None):
        self._enter("concept_rows", cik)
        self.periods.append((period_from, period_to))
        self.budgets.append(budget_s)
        return [concept_row(cik, self.value)], []
