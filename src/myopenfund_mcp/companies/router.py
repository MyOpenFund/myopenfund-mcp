"""Per-company source routing for the ``companies_*`` tools: the store, live SEC, or whichever is current.

``MOF_COMPANIES_SOURCE``:

- ``store``: every company from the store; live SEC is never called.
- ``sec``: every company live from SEC; the store is never called.
- ``auto`` (default), per company: the store when it is configured **and** holds the company **and**
  its latest filing for it (max ``known_at`` of its ``filings`` rows, or that accession itself) is
  not older than SEC's latest (one cached submissions request) **and** it holds
  ``filing_statements`` for SEC's latest filing that carries positions; else live, with a
  ``reason``: ``no_store``, ``not_in_store``, ``store_behind``, ``store_no_positions``, or
  ``store_unavailable`` (the store failed: live for that call only).

Two readings of the spec's rule, each forced by the producer's tables (documented here, not hidden):
the store lists only report forms in ``filings`` (8-K included), so SEC's latest is taken among
those same forms (``LiveSource.report_forms``), not among every form (a Form 4 would put every
store behind); and an 8-K never has statement positions, so the positions check reads SEC's latest
filing of the forms that carry them (``LiveSource.positions_forms``: 10-K, 10-Q, 20-F, 40-F).

Without a contact (no live source) under ``auto``: a company in the store is served from it
unchecked (nothing to compare with), any other one raises :class:`LiveNotConfigured`. When the
freshness check itself fails on the SEC side (SEC unreachable, refused, closed), a company in the
store is served from it, and the route is not remembered, so the next call checks again.

A route is remembered per CIK for ``route_ttl_s`` (``MOF_CACHE_TTL_SUBMISSIONS_H``, the age of the
submissions it was decided on); a ``store_unavailable`` route never is. Blocking: the tools call
it in a worker thread, like the store and the live source.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from .store import StoreBadArgument, StoreUnknownDataset

log = logging.getLogger("myopenfund_mcp.companies")

MODES = ("auto", "store", "sec")
STORE, LIVE = "store", "sec_live"
NOT_IN_STORE, STORE_BEHIND, STORE_NO_POSITIONS = "not_in_store", "store_behind", "store_no_positions"
STORE_UNAVAILABLE, NO_STORE = "store_unavailable", "no_store"
#: the caller's mistakes, never a store failure: they propagate, no fallback
_ARGUMENT_ERRORS = (StoreBadArgument, StoreUnknownDataset)


@dataclass(frozen=True)
class Route:
    source: Literal["store", "sec_live"]
    reason: str | None = None


class LiveNotConfigured(RuntimeError):
    """``auto`` with a store and no ``MOF_SEC_CONTACT``: this company needs SEC."""

    def __init__(self, cik: str, reason: str):
        super().__init__(f"CIK {cik} needs SEC ({reason}) and no SEC contact is configured")
        self.cik, self.reason = cik, reason


def _store_failure(exc: BaseException) -> bool:
    return not isinstance(exc, _ARGUMENT_ERRORS)


class SourceRouter:
    """The store's / live source's method set, dispatched per CIK, plus :meth:`route` and :meth:`call`.

    ``clock`` (monotonic seconds) and ``now`` (aware datetime) are injectable for tests."""

    def __init__(
        self,
        store,
        live,
        mode: str,
        *,
        route_ttl_s: float = 24 * 3600,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        if mode == "store" and store is None:
            raise ValueError("MOF_COMPANIES_SOURCE=store needs a company store")
        if mode == "sec" and live is None:
            raise ValueError("MOF_COMPANIES_SOURCE=sec needs MOF_SEC_CONTACT")
        if store is None and live is None:
            raise ValueError("a company store or MOF_SEC_CONTACT is needed")
        self.store, self.live, self.mode = store, live, mode
        self._ttl_s = route_ttl_s
        self._clock = clock
        self._now = now or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self._routes: dict[str, tuple[float, Route]] = {}

    def __repr__(self) -> str:
        return (
            f"SourceRouter(mode={self.mode!r}, store={self.store is not None}, live={self.live is not None})"
        )

    @property
    def live_enabled(self) -> bool:
        """Whether a call may reach SEC (its time budget then includes ``MOF_SEC_CALL_BUDGET_S``)."""
        return self.mode != "store" and self.live is not None

    # --- routing ---------------------------------------------------------------------------------
    def route(self, cik: str) -> Route:
        if self.mode == "store":
            return Route(STORE)
        if self.mode == "sec":
            return Route(LIVE)
        if self.store is None:
            return Route(LIVE, NO_STORE)
        with self._lock:
            kept = self._routes.get(cik)
        if kept is not None and self._clock() - kept[0] < self._ttl_s:
            return kept[1]
        route, keep = self._check(cik)
        if keep:
            with self._lock:
                self._routes[cik] = (self._clock(), route)
        return route

    def _store_ciks(self) -> set[str]:
        return {r.get("subject_value") for r in self.store.entities()}

    def _unavailable(self, exc: BaseException) -> tuple[Route, bool]:
        log.warning("companies: store failed, answering from SEC (%s)", type(exc).__name__)
        if self.live is None:
            raise exc
        return Route(LIVE, STORE_UNAVAILABLE), False

    def _check(self, cik: str) -> tuple[Route, bool]:
        """``auto`` with a store: (route, whether to remember it)."""
        try:
            in_store = cik in self._store_ciks()
        except Exception as exc:  # any store failure falls back to SEC
            if not _store_failure(exc):
                raise
            return self._unavailable(exc)
        if not in_store:
            return Route(LIVE, NOT_IN_STORE), True
        if self.live is None:
            return Route(STORE), True
        try:
            latest = self.live.latest_filing(cik, forms=self.live.report_forms)
            report = self.live.latest_filing(cik, forms=self.live.positions_forms)
        except Exception as exc:  # noqa: BLE001 -- SEC cannot tell: the store answers, checked next call
            log.warning(
                "companies: SEC freshness check failed, answering from the store (%s)", type(exc).__name__
            )
            return Route(STORE), False
        key = {"entity_scheme": ["cik"], "entity_value": [cik]}
        try:
            if latest is not None:
                filed = self._read_or_empty("filings", key, ["filing_ref", "known_at"], None)
                refs = {r["filing_ref"] for r in filed}
                newest = max((r["known_at"] for r in filed if r["known_at"] is not None), default=None)
                current = latest.accession in refs or (
                    latest.accepted is not None and newest is not None and newest >= latest.accepted
                )
                if not current:
                    return Route(LIVE, STORE_BEHIND), True
            if report is not None:
                located = self._read_or_empty(
                    "filing_statements", {**key, "filing_ref": [report.accession]}, ["filing_ref"], 1
                )
                if not located:
                    return Route(LIVE, STORE_NO_POSITIONS), True
        except Exception as exc:  # noqa: BLE001 -- any store failure falls back to SEC
            return self._unavailable(exc)
        return Route(STORE), True

    def _read_or_empty(
        self, dataset: str, filters: dict, columns: list[str], limit: int | None
    ) -> list[dict]:
        """A store read for routing; a dataset the store does not have reads as empty."""
        try:
            return self.store.read(dataset, as_of=self._now(), filters=filters, columns=columns, limit=limit)
        except StoreUnknownDataset:
            return []

    def source(self, route: Route):
        """The object that serves ``route``."""
        return self.store if route.source == STORE else self.live

    def call(self, *, company: str, method: str, **kwargs) -> tuple[Route, object]:
        """``(route, result)`` of ``method(**kwargs)`` on the source of the CIK ``company``. Under
        ``auto``, a store failure (anything but an argument error) answers from SEC for this call,
        ``store_unavailable``."""
        route = self.route(company)
        if route.source == LIVE and self.live is None:
            raise LiveNotConfigured(company, route.reason or NO_STORE)
        if route.source == STORE:
            try:
                return route, self._on_store(method, kwargs)
            except Exception as exc:  # a store failure, unless an argument error
                if self.mode != "auto" or self.live is None or not _store_failure(exc):
                    raise
                log.warning("companies: store failed, answering from SEC (%s)", type(exc).__name__)
                route = Route(LIVE, STORE_UNAVAILABLE)
        return route, getattr(self.live, method)(**kwargs)

    def _on_store(self, method: str, kwargs: dict):
        if method == "concept_rows":
            # the store holds every position already: the period only steers what live fetches
            kwargs = {k: v for k, v in kwargs.items() if k not in ("period_from", "period_to")}
        return getattr(self.store, method)(**kwargs)

    # --- the store's / live source's method set --------------------------------------------------
    def entities(self) -> list[dict]:
        """``store``/``sec``: that source's rows. ``auto``: the store's rows plus the live resolver's
        rows of the CIKs the store does not hold (store rows win); either alone when the other fails."""
        if self.mode == "store" or (self.mode == "auto" and self.live is None):
            return self.store.entities()
        if self.mode == "sec" or self.store is None:
            return self.live.entities()
        try:
            stored = self.store.entities()
        except Exception as exc:  # noqa: BLE001 -- a store outage leaves the live resolver
            log.warning("companies: store entities failed, resolving from SEC (%s)", type(exc).__name__)
            return self.live.entities()
        try:
            live = self.live.entities()
        except Exception as exc:  # noqa: BLE001 -- an SEC outage leaves the store's companies
            log.warning("companies: SEC resolver failed, resolving from the store (%s)", type(exc).__name__)
            return stored
        held = {r.get("subject_value") for r in stored}
        return stored + [r for r in live if r.get("subject_value") not in held]

    def read(
        self,
        dataset: str,
        *,
        as_of: datetime,
        filters: dict | None,
        columns: list[str] | None,
        limit: int | None,
        vintages: bool = False,
    ) -> list[dict]:
        """One company's rows (the ``entity_value`` filter names one CIK) from its source; several
        CIKs are read per source and concatenated (``limit`` applies to each source's part)."""
        kwargs = {
            "dataset": dataset,
            "as_of": as_of,
            "columns": columns,
            "limit": limit,
            "vintages": vintages,
        }
        ciks = list(dict.fromkeys((filters or {}).get("entity_value") or []))
        if not ciks:
            src = self.store if self.mode != "sec" and self.store is not None else self.live
            return src.read(filters=filters, **kwargs)
        rows: list[dict] = []
        for cik in ciks:
            _, part = self.call(
                company=cik, method="read", filters={**filters, "entity_value": [cik]}, **kwargs
            )
            rows += part
        return rows

    def concept_rows(self, cik: str, as_of: datetime, *, period_from=None, period_to=None):
        return self.call(
            company=cik,
            method="concept_rows",
            cik=cik,
            as_of=as_of,
            period_from=period_from,
            period_to=period_to,
        )[1]

    def summaries(self, cik: str, name: str, rows: list[dict], vintages: list[dict] | None) -> list:
        return self.source(self.route(cik)).summaries(cik=cik, name=name, rows=rows, vintages=vintages)

    def concept_catalog(self) -> list[dict]:
        return (self.store or self.live).concept_catalog()

    def metric_catalog(self) -> list[dict]:
        return (self.store or self.live).metric_catalog()

    def latest_filing(self, cik: str, **kwargs):
        """SEC's newest filing of the CIK (the live source's; None without one)."""
        return None if self.live is None else self.live.latest_filing(cik, **kwargs)

    def filings(self, cik: str, **kwargs) -> list:
        """SEC's filings of the CIK (the live source's; ``[]`` without one)."""
        return [] if self.live is None else self.live.filings(cik, **kwargs)

    def close(self) -> None:
        for src in (self.live, self.store):
            if src is not None:
                src.close()
