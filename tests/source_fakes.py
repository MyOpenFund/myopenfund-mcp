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


@dataclass(frozen=True)
class Ref:
    """The FilingRef fields the router reads."""

    accession: str
    accepted: dt.datetime | None
    form: str = "10-K"


def entity(cik: str) -> dict:
    name, ticker = NAMES[cik]
    return {
        "subject_scheme": "cik",
        "subject_value": cik,
        "entity_id": f"e-{cik}",
        "name": name,
        "id_scheme": "ticker",
        "id_value": ticker,
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
    """``ciks``: the companies held. ``filings``: cik -> [(accession, known_at)]. ``positions``: cik ->
    accessions with filing_statements rows. ``value``: the revenue the store holds."""

    def __init__(self, ciks=(APPLE,), *, filings=None, positions=None, value=100, fail=None):
        super().__init__(fail)
        self.ciks = list(ciks)
        self.filings = {APPLE: [(K24, ACCEPTED_K24), (E24, ACCEPTED_E24)]} if filings is None else filings
        self.positions = {APPLE: {K24}} if positions is None else positions
        self.value = value

    def freshness(self) -> dict:
        self._enter("freshness")
        return {"last_published": ACCEPTED_E24, "batches": 1}

    def entities(self) -> list[dict]:
        self._enter("entities")
        return [entity(c) for c in self.ciks]

    def read(self, dataset, *, as_of, filters, columns, limit, vintages=False) -> list[dict]:
        cik = (filters or {}).get("entity_value", [None])[0]
        self._enter(f"read:{dataset}", cik)
        if dataset == "filings":
            return [{"filing_ref": a, "known_at": k} for a, k in self.filings.get(cik, [])]
        if dataset == "filing_statements":
            wanted = set(filters.get("filing_ref", []))
            return [{"filing_ref": a} for a in sorted(self.positions.get(cik, set()) & wanted)][:limit]
        return [fact(cik, self.value)] if cik in self.ciks else []

    def concept_rows(self, cik, as_of):  # the store's signature as the existing tests patch it
        self._enter("concept_rows", cik)
        return [concept_row(cik, self.value)], []


class FakeLive(_Recording):
    """``latest``: cik -> (newest report-form filing, newest positions-form filing). ``value``: the
    revenue SEC gives."""

    report_forms = ("10-K", "10-Q", "8-K")
    positions_forms = ("10-K", "10-Q")

    def __init__(self, *, latest=None, value=200, fail=None):
        super().__init__(fail)
        self.latest = (
            {APPLE: (Ref(E24, ACCEPTED_E24, "8-K"), Ref(K24, ACCEPTED_K24))} if latest is None else latest
        )
        self.value = value
        self.periods: list[tuple] = []

    def entities(self) -> list[dict]:
        self._enter("entities")
        return [entity(c) for c in NAMES]

    def latest_filing(self, cik, *, forms=None):
        self._enter("latest_filing", cik)
        refs = self.latest.get(cik)
        if refs is None:
            return None
        return refs[1] if tuple(forms or ()) == self.positions_forms else refs[0]

    def filings(self, cik, *, forms, period_from, period_to) -> list:
        self._enter("filings", cik)
        return []

    def read(self, dataset, *, as_of, filters, columns, limit, vintages=False) -> list[dict]:
        cik = filters["entity_value"][0]
        self._enter(f"read:{dataset}", cik)
        return [fact(cik, self.value)]

    def concept_rows(self, cik, as_of, *, period_from=None, period_to=None):
        self._enter("concept_rows", cik)
        self.periods.append((period_from, period_to))
        return [concept_row(cik, self.value)], []
