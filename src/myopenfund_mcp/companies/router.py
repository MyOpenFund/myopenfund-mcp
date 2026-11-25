"""Per-company source routing for the ``companies_*`` tools: the store, live SEC, or whichever is current.

``MOF_COMPANIES_SOURCE``:

- ``store``: every company from the store; live SEC is never called.
- ``sec``: every company live from SEC; the store is never called.
- ``auto`` (default), per company and for the call's ``as_of``: the store when it is configured
  **and** holds the company **and** holds SEC's latest report filing accepted at or before
  ``as_of`` **and** holds ``filing_statements`` for SEC's latest filing carrying positions accepted
  at or before ``as_of``; else live, with a ``reason``: ``no_store``, ``not_in_store``,
  ``store_behind``, ``store_no_positions`` (the store's positions lag its facts), or
  ``store_unavailable`` (the store failed: live for that call only).

Readings of the spec's rule (rulings R26, R29, R30):

- SEC's latest is taken among the forms the producer lists in ``filings``
  (``LiveSource.report_forms``, 8-K included; a Form 4 would put every store behind), and the
  positions check among the forms that carry positions (``LiveSource.positions_forms``: 10-K, 10-Q,
  20-F, 40-F; an 8-K never has any). A filing SEC lists without an acceptance time cannot be placed
  against ``as_of`` and is not compared.
- The store holds a filing when its accession is among the store's filings of the company, or the
  store's newest such filing is known no earlier than that filing's acceptance.
- A store without a published ``filings`` dataset (logged once, at warning level): its filings are
  read from ``fundamentals`` (the ``filing_ref`` and ``known_at`` of its facts), which see only
  filings with XBRL facts, so the comparison then uses SEC's latest filing carrying positions (an
  8-K without facts would otherwise put every company behind).

When the freshness check cannot run (``auto`` with a store and no ``MOF_SEC_CONTACT``, or SEC failing
during the check) a company in the store is served from it with ``reason: freshness_unchecked``
(ruling R27). A company that must be read live while no SEC contact is configured raises
:class:`LiveNotConfigured` (``[sec_contact_missing]``).

Memory: nothing is remembered as a route. SEC's filing list per company is the live source's
submissions memo (stamped when the submissions were fetched, kept ``MOF_CACHE_TTL_SUBMISSIONS_H``);
what the store holds (its companies, whether ``filings`` is published, each company's filings, the
positions found per filing) is remembered per store publication (``freshness()["batches"]``: any
publication drops it). One :meth:`plan` reads ``freshness`` once and ``entities``/``datasets`` only
after a publication, whatever the number of companies. Blocking: the tools call it in a worker
thread, like the store and the live source.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from .store import StoreBadArgument, StoreUnknownDataset

log = logging.getLogger("myopenfund_mcp.companies")

MODES = ("auto", "store", "sec")
STORE, LIVE = "store", "sec_live"
NOT_IN_STORE, STORE_BEHIND, STORE_NO_POSITIONS = "not_in_store", "store_behind", "store_no_positions"
STORE_UNAVAILABLE, NO_STORE = "store_unavailable", "no_store"
FRESHNESS_UNCHECKED = "freshness_unchecked"
#: the caller's mistakes, never a store failure: they propagate, no fallback
_ARGUMENT_ERRORS = (StoreBadArgument, StoreUnknownDataset)
#: the live source's methods that take a per-call wait (``budget_s``)
_BUDGETED = frozenset({"read", "concept_rows"})


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


def _base(form: str | None) -> str:
    upper = (form or "").upper()
    return upper.removesuffix("/A")


@dataclass
class _StoreState:
    """What the store holds, for one publication (``batches``)."""

    batches: object
    ciks: frozenset
    filings_published: bool
    filings: dict = field(default_factory=dict)  # cik -> (accessions, newest known_at)
    positions: dict = field(default_factory=dict)  # (cik, accession) -> bool


class SourceRouter:
    """The store's / live source's method set, dispatched per CIK, plus :meth:`plan`.

    ``clock`` (monotonic seconds) and ``now`` (aware datetime) are injectable for tests."""

    def __init__(
        self,
        store,
        live,
        mode: str,
        *,
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
        self._clock = clock
        self._now = now or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self._state: _StoreState | None = None
        self._warned_no_filings = False

    def __repr__(self) -> str:
        return (
            f"SourceRouter(mode={self.mode!r}, store={self.store is not None}, live={self.live is not None})"
        )

    @property
    def live_enabled(self) -> bool:
        """Whether a call may reach SEC (its time budget then includes ``MOF_SEC_CALL_BUDGET_S``)."""
        return self.mode != "store" and self.live is not None

    # --- routing ---------------------------------------------------------------------------------
    def plan(self, ciks: Iterable[str], as_of: datetime | None = None) -> dict[str, Route]:
        """Each CIK's route for a call at ``as_of`` (None = now)."""
        ciks = list(dict.fromkeys(ciks))
        if self.mode == "store":
            return dict.fromkeys(ciks, Route(STORE))
        if self.mode == "sec":
            return dict.fromkeys(ciks, Route(LIVE))
        if self.store is None:
            return dict.fromkeys(ciks, Route(LIVE, NO_STORE))
        as_of = as_of or self._now()
        try:
            state = self._store_state()
        except Exception as exc:  # any store failure falls back to SEC
            if not _store_failure(exc) or self.live is None:
                raise
            log.warning("companies: store failed, answering from SEC (%s)", type(exc).__name__)
            return dict.fromkeys(ciks, Route(LIVE, STORE_UNAVAILABLE))
        return {cik: self._decide(cik, as_of, state) for cik in ciks}

    def route(self, cik: str, as_of: datetime | None = None) -> Route:
        return self.plan([cik], as_of)[cik]

    def _store_state(self) -> _StoreState:
        batches = self.store.freshness()["batches"]
        with self._lock:
            state = self._state
        if state is not None and state.batches == batches:
            return state
        ciks = frozenset(r.get("subject_value") for r in self.store.entities())
        published = {d.get("name") for d in self.store.datasets()}
        state = _StoreState(batches, ciks, "filings" in published)
        if not state.filings_published and not self._warned_no_filings:
            self._warned_no_filings = True
            log.warning(
                "companies: the store has no published 'filings' dataset; its filings are read "
                "from 'fundamentals' to route companies"
            )
        with self._lock:
            self._state = state
        return state

    def _decide(self, cik: str, as_of: datetime, state: _StoreState) -> Route:
        if cik not in state.ciks:
            return Route(LIVE, NOT_IN_STORE)
        if self.live is None:
            return Route(STORE, FRESHNESS_UNCHECKED)
        try:
            refs = self.live.filings(cik, forms=self.live.report_forms, period_from=None, period_to=None)
        except Exception as exc:  # noqa: BLE001 -- SEC cannot tell: the store answers, unchecked
            log.warning(
                "companies: SEC freshness check failed, answering from the store (%s)", type(exc).__name__
            )
            return Route(STORE, FRESHNESS_UNCHECKED)
        known = sorted(
            (r for r in refs if r.accepted is not None and r.accepted <= as_of),
            key=lambda r: r.accepted,
            reverse=True,
        )
        positioned = [r for r in known if _base(r.form) in self.live.positions_forms]
        latest = known[0] if state.filings_published else (positioned[0] if positioned else None)
        try:
            if latest is not None and not self._holds(cik, latest, state):
                return Route(LIVE, STORE_BEHIND)
            if positioned and not self._has_positions(cik, positioned[0].accession, state):
                return Route(LIVE, STORE_NO_POSITIONS)
        except Exception as exc:  # any store failure falls back to SEC
            if not _store_failure(exc):
                raise
            log.warning("companies: store failed, answering from SEC (%s)", type(exc).__name__)
            return Route(LIVE, STORE_UNAVAILABLE)
        return Route(STORE)

    def _holds(self, cik: str, ref, state: _StoreState) -> bool:
        with self._lock:
            held = state.filings.get(cik)
        if held is None:
            key = {"entity_scheme": ["cik"], "entity_value": [cik]}
            dataset = "filings" if state.filings_published else "fundamentals"
            rows = self.store.read(
                dataset, as_of=self._now(), filters=key, columns=["filing_ref", "known_at"], limit=None
            )
            accessions = frozenset(r["filing_ref"] for r in rows if r["filing_ref"])
            newest = max((r["known_at"] for r in rows if r["known_at"] is not None), default=None)
            held = (accessions, newest)
            with self._lock:
                state.filings[cik] = held
        accessions, newest = held
        return ref.accession in accessions or (newest is not None and newest >= ref.accepted)

    def _has_positions(self, cik: str, accession: str, state: _StoreState) -> bool:
        with self._lock:
            found = state.positions.get((cik, accession))
        if found is None:
            rows = self.store.read(
                "filing_statements",
                as_of=self._now(),
                filters={"entity_scheme": ["cik"], "entity_value": [cik], "filing_ref": [accession]},
                columns=["filing_ref"],
                limit=1,
            )
            found = bool(rows)
            with self._lock:
                state.positions[(cik, accession)] = found
        return found

    # --- running a call on a route ---------------------------------------------------------------
    def source(self, route: Route):
        """The object that serves ``route``."""
        return self.store if route.source == STORE else self.live

    def _can_fall_back(self, exc: BaseException) -> bool:
        return self.mode == "auto" and self.live is not None and _store_failure(exc)

    def try_store(self, *, method: str, **kwargs):
        """``method(**kwargs)`` on the store; None when it failed and ``auto`` can answer from SEC
        instead (the caller then routes those companies ``store_unavailable``)."""
        try:
            return self._on_store(method, kwargs)
        except Exception as exc:
            if not self._can_fall_back(exc):
                raise
            log.warning("companies: store failed, answering from SEC (%s)", type(exc).__name__)
            return None

    def call(
        self, *, company: str, route: Route, method: str, budget_s: float | None = None, **kwargs
    ) -> tuple[Route, object]:
        """``(route, result)`` of ``method(**kwargs)`` on ``route``'s source for the CIK ``company``.
        A store failure under ``auto`` answers from SEC (``store_unavailable``). ``budget_s``: the live
        source's wait for this call (the router's freshness check counts in the same budget)."""
        if route.source == STORE:
            result = self.try_store(method=method, **kwargs)
            if result is not None:
                return route, result
            route = Route(LIVE, STORE_UNAVAILABLE)
        if self.live is None:
            raise LiveNotConfigured(company, route.reason or NO_STORE)
        if budget_s is not None and method in _BUDGETED:
            kwargs["budget_s"] = budget_s
        return route, getattr(self.live, method)(**kwargs)

    def _on_store(self, method: str, kwargs: dict):
        if method == "concept_rows":
            # the store holds every position already: the period only steers what live fetches
            kwargs = {k: v for k, v in kwargs.items() if k not in ("period_from", "period_to")}
        return getattr(self.store, method)(**kwargs)

    # --- the store's / live source's method set --------------------------------------------------
    def resolver_rows(self) -> tuple[list[dict], BaseException | None]:
        """``(entity rows, SEC failure)``. ``store``/``sec``: that source's rows. ``auto``: the store's
        rows, plus the live resolver's: every row of the CIKs the store does not hold, and the
        tickers of those it holds as additional identifiers (store rows keep the names); either
        alone when the other fails, the SEC failure returned so an unresolved query can report it."""
        if self.mode == "store" or (self.mode == "auto" and self.live is None):
            return self.store.entities(), None
        if self.mode == "sec" or self.store is None:
            return self.live.entities(), None
        try:
            stored = self.store.entities()
        except Exception as exc:  # noqa: BLE001 -- a store outage leaves the live resolver
            log.warning("companies: store entities failed, resolving from SEC (%s)", type(exc).__name__)
            return self.live.entities(), None
        try:
            live = self.live.entities()
        except Exception as exc:  # noqa: BLE001 -- an SEC outage leaves the store's companies
            log.warning("companies: SEC resolver failed, resolving from the store (%s)", type(exc).__name__)
            return stored, exc
        held = {r.get("subject_value") for r in stored}
        extra = []
        for r in live:
            if r.get("subject_value") not in held:
                extra.append(r)
            elif r.get("id_scheme") == "ticker" and r.get("id_value"):
                extra.append({**r, "name": None, "entity_id": None, "country": None})
        return stored + extra, None

    def entities(self) -> list[dict]:
        return self.resolver_rows()[0]

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
        """Rows of the CIKs named by ``entity_value``, each from its source (store ones in one read);
        concatenated, ``limit`` applying to each part."""
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
        routes = self.plan(ciks, as_of)
        held = [c for c in ciks if routes[c].source == STORE]
        rows: list[dict] = []
        if held:
            got = self.try_store(method="read", filters={**filters, "entity_value": held}, **kwargs)
            if got is None:
                routes.update(dict.fromkeys(held, Route(LIVE, STORE_UNAVAILABLE)))
            else:
                rows += got
        for cik in ciks:
            if routes[cik].source == LIVE:
                rows += self.call(
                    company=cik,
                    route=routes[cik],
                    method="read",
                    filters={**filters, "entity_value": [cik]},
                    **kwargs,
                )[1]
        return rows

    def concept_rows(self, cik: str, as_of: datetime, *, period_from=None, period_to=None):
        return self.call(
            company=cik,
            route=self.route(cik, as_of),
            method="concept_rows",
            cik=cik,
            as_of=as_of,
            period_from=period_from,
            period_to=period_to,
        )[1]

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
        """The live source first (its builds stop), then the store."""
        for src in (self.live, self.store):
            if src is not None:
                src.close()
