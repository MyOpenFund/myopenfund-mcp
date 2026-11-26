"""The `market` toolset: `market_prices`, daily prices (raw or total return) through market-data's Python API
(`market_data.api.prices`, the one code path of the tool and a notebook: market-data spec §10). Nothing is
computed here.

Where a call reads (`mode`, default MOF_MARKET_SOURCE), per input, as the companies tools route (MCP
chantier 3 Q9):

- ``store``: the store only (market-data's ``server`` profile on the store of MOF_COMPANIES_CATALOG);
  never a live fetch.
- ``live``: a live fetch only (market-data's ``none`` profile: in memory, nothing stored); never the store.
- ``auto``: the store when it is configured and holds the input over the range (at least one row the
  caller may read), else a live fetch, with a ``reason``: ``no_store``, ``not_in_store`` (nothing the
  caller may read names it), ``store_no_data`` (no row in the range), ``store_source_restricted``
  (``source`` is a restricted source this caller may not read from the store) or ``store_unavailable``
  (the store failed or ran out of time).

The store is read in one market-data call for all the inputs (and one more, raw, for a total return: it
gives each session's provider). When several sources the caller may read price one session, the first of
MOF_MARKET_PRECEDENCE wins (codes the caller may not read are dropped from it; readable ones it leaves out
come after it, alphabetically); ``source`` reads one source only.

A live fetch never reads the store. An input only the store resolves by its form (a ``mof:`` instrument
id, a ``source:symbol`` string, a CIK) is refused with ``[market_unknown_instrument]`` before any provider
request; any other name goes to the provider as typed. A client not allowed the live source gets
``[market_source_restricted]`` first, whatever the names (no answer depends on what the store holds).

Every row carries ``source`` (store or live), ``reason`` when ``auto`` went live, and ``source_code`` (the
provider of that price). Stored rows carry ``instrument_id`` only, never a label, a private_history token
or a source symbol (spec §9.3); live rows also carry ``input`` (the caller's own words). The ``inputs``
list says where each input was read, without instrument ids for stored ones.

Restricted sources (spec §13): stored rows of a restricted source are served only to a client of
MOF_MARKET_RESTRICTED_ALLOW that lists it (`auth.allowed_restricted_for`; none over stdio). A live fetch
over stdio uses the caller's own key (the process environment: TIINGO_API_KEY, MASSIVE_API_KEY), so it is
the caller's own data: no gate. Over HTTP it uses the deployment's key, and serving it is redistribution:
only the client's listed sources (`auth.live_restricted_for`). A key is never a tool argument and never
echoed.

Bounds: at most MOF_MARKET_MAX_INSTRUMENTS inputs and MOF_MARKET_MAX_ROWS rows per call, and the server's
result cap; larger requests are refused, never cut. Store reads run in a worker thread within
MOF_COMPANIES_QUERY_TIMEOUT_S, live fetches within MOF_MARKET_LIVE_TIMEOUT_S. At most
MOF_MARKET_LIVE_CONCURRENCY live fetches run at once (they spend the key's quota): one more is answered
``[market_busy]`` at once, never queued. A live fetch never sleeps beyond its time budget: a wait the
adapter asks for (Tiingo's hourly budget, Massive's 5 requests a minute, a ``Retry-After``) that would end
after it is ``[limit_exceeded]`` with the wait, at once. A fetch past its time budget all the same (a slow
answer) is abandoned, not stopped (a worker thread cannot be): it may still finish and spend its quota, and
it keeps its slot until it ends.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal

import anyio
from fastmcp import FastMCP
from market_data import api
from market_data.adapters import massive, tiingo
from pydantic import Field

from ..auth import (
    allowed_restricted_for,
    caller_client_id,
    live_restricted_for,
    registered_restricted_sources,
    registered_sources,
    tier_check,
)
from ..config import Settings
from ..errors import (
    BACKEND_UNAVAILABLE,
    INVALID_ARGUMENT,
    LIMIT_EXCEEDED,
    MARKET_AMBIGUOUS_INSTRUMENT,
    MARKET_BUSY,
    MARKET_KEY_MISSING,
    MARKET_NO_DATA,
    MARKET_SOURCE_RESTRICTED,
    MARKET_UNKNOWN_INSTRUMENT,
    RESULT_TOO_LARGE,
    TIMEOUT,
    TOO_MANY_INSTRUMENTS,
    tool_error,
)
from ..paging import cap_result

log = logging.getLogger("myopenfund_mcp.market")

STORE, LIVE = "store", "live"
NO_STORE, NOT_IN_STORE, STORE_NO_DATA = "no_store", "not_in_store", "store_no_data"
STORE_SOURCE_RESTRICTED, STORE_UNAVAILABLE = "store_source_restricted", "store_unavailable"
#: the environment variable holding each live source's key (market-data's adapters)
KEY_ENVS = {tiingo.SOURCE_CODE: tiingo.API_KEY_ENV, massive.SOURCE_CODE: massive.API_KEY_ENV}
_KEY_ERRORS = (tiingo.TiingoKeyError, massive.MassiveKeyError)
_QUOTA_ERRORS = (tiingo.TiingoQuotaError,)
MAX_INSTRUMENT_CHARS = 64
_SOURCE_CODE = re.compile(r"[a-z][a-z0-9_]{0,62}")  # open-data-store's source-code rule
_CIK = re.compile(r"[0-9]+")
#: the adapters' own text when a retry would wait longer than ``max_wait_s`` (numbers only)
_ADAPTER_WAIT = re.compile(r"asked to wait ([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?) s, beyond max_wait_s")
_LISTED = 20  # inputs quoted in a message
TOOLSET = "the market toolset (market_prices)"  # what a registry error names
ANNOTATIONS = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True}
SIZE_HINT = "narrow the date range or ask for fewer instruments"
BACKEND_MESSAGE = "the market data store cannot be read right now; try again later"
LIVE_MESSAGE = "the live market data source cannot be read right now; try again later"
INSTRUMENTS_DESCRIPTION = (f"Tickers (store and live); FIGIs, CIKs and mof: instrument ids resolve through the "
                           f"store only. At most the server maximum per call, each at most {MAX_INSTRUMENT_CHARS} "
                           "characters")
MODE_DESCRIPTION = ("auto: the store when it holds the instrument over the range, else a live fetch; store: never "
                    "live; live: never the store. Omitted = the server's MOF_MARKET_SOURCE")
SOURCE_DESCRIPTION = ("A provider code to read from (e.g. tiingo, massive, private_history); live fetches take "
                      "tiingo (default) or massive. Omitted = any source this client may read")

#: ``(source_code, budget) -> Adapter``: the live adapter of a fetch (tests inject one); ``budget`` is the
#: fetch's `_Budget` (its ``sleep`` and ``max_wait_s`` keep every wait inside the call's time budget)
AdapterFactory = Callable[[str, "_Budget"], object]


def _labels(instruments: list[str], settings: Settings) -> list[str]:
    labels = list(dict.fromkeys(i.strip() for i in instruments))
    if not labels or not all(labels):
        raise tool_error(INVALID_ARGUMENT, "pass at least one instrument; none may be blank")
    if len(labels) > settings.market_max_instruments:
        raise tool_error(TOO_MANY_INSTRUMENTS, f"{len(labels)} instruments; at most "
                                               f"{settings.market_max_instruments} per call")
    return labels


def _date(value: str, name: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        raise tool_error(INVALID_ARGUMENT, f"{name} must be an ISO date, YYYY-MM-DD") from None


def _listed(labels) -> list[str]:
    return list(labels)[:_LISTED]


async def _off_loop(fn, seconds: float):
    """`fn()` in a worker thread within `seconds` (`[timeout]` after it; the thread is abandoned)."""
    try:
        with anyio.fail_after(seconds):
            return await anyio.to_thread.run_sync(fn, abandon_on_cancel=True)
    except TimeoutError:
        raise tool_error(TIMEOUT, f"the market data did not come within {seconds} s; {SIZE_HINT}") from None


# --- the store (market-data's server profile) ---------------------------------------------------------
_KINDS = ((api.UnknownInstrumentError, "unknown"), (api.NoDataError, "no_data"),
          (api.RestrictedSourceError, "restricted"))
_REASONS = {"unknown": NOT_IN_STORE, "no_data": STORE_NO_DATA, "restricted": STORE_SOURCE_RESTRICTED,
            "unavailable": STORE_UNAVAILABLE}


def precedence_for(settings: Settings, allowed: frozenset[str], restricted: frozenset[str],
                   registered: frozenset[str]) -> tuple[str, ...]:
    """MOF_MARKET_PRECEDENCE for a caller who may read the restricted sources ``allowed``: its codes this
    caller may read, in order, then the other sources it may read, alphabetically (market-data ignores the
    sources a precedence leaves out). A restricted code outside ``allowed`` is never passed."""
    readable = {c for c in registered if c not in restricted or c in allowed}
    listed = [c for c in settings.market_precedence if c in readable]
    return (*listed, *sorted(readable - set(listed)))


def _store_answers(store, labels, start, end, source, adjust, allowed, precedence, how) -> dict:
    """One store call: ``{"table", "raw", "failed": {input: kind}, "refusal"}``.

    All the inputs are read in one market-data call (one more, raw, for a total return: each session's
    provider). An error naming some inputs (unknown, no data from ``source``) sets them aside and the others
    are read again; an error about all of them (a restricted ``source``) sets them all aside; an ambiguous
    input or a conflict refuses the call (the caller's to fix, whatever the mode). In ``auto``, when fewer
    instruments come back than inputs remain (two inputs naming one instrument, or one without a row in the
    range), each remaining input is read alone to know which has no row: the server profile keys its rows
    by instrument_id only."""

    def go(s):
        kw = {"source": source, "profile": "server", "store": s, "allowed_restricted": allowed,
              "precedence": precedence}
        pending, failed = list(labels), {}
        table = raw = None
        while pending:
            try:
                table = api.prices(pending, start, end, adjust=adjust, **kw)
                raw = api.prices(pending, start, end, adjust="raw", **kw) if adjust == "total_return" else None
                break
            except api.AmbiguousInstrumentError as exc:  # handed back: a ValueError would read as a bad argument
                return {"refusal": ("ambiguous", list(exc.inputs))}
            except api.ApiError as exc:
                kind = next((name for cls, name in _KINDS if isinstance(exc, cls)), None)
                if kind is None:  # several sources, or series, disagree on one session
                    return {"refusal": ("conflict", pending)}
                named = set(getattr(exc, "inputs", ()) or ()) if kind != "restricted" else set(pending)
                hit = [label for label in pending if label in named] or pending
                failed.update(dict.fromkeys(hit, kind))
                pending = [label for label in pending if label not in hit]
        if table is not None and how != STORE:
            ids = (raw if raw is not None else table)["instrument_id"].to_pylist()
            if len(set(ids)) < len(pending):
                for label in pending:
                    one = api.prices([label], start, end, adjust="raw", **kw)
                    if not one.num_rows:
                        failed[label] = "no_data"
        return {"table": table, "raw": raw, "failed": failed, "refusal": None}

    return store.call(go)


def _store_rows(table, raw) -> list[dict]:
    if table is None:
        return []
    rows = table.to_pylist()
    if raw is not None:  # total return: the provider of each session's price
        provider = {(r["instrument_id"], r["session_date"]): r["source_code"] for r in raw.to_pylist()}
        for r in rows:
            r["source_code"] = provider.get((r["instrument_id"], r["session_date"]))
    return [{**r, "source": STORE} for r in rows]


def _store_refusal(kind: str, labels: list[str], source: str | None):
    """The error of a store-mode (or no-live-fallback) call for `labels` that the store cannot answer."""
    if kind in ("unknown", NOT_IN_STORE):
        return tool_error(MARKET_UNKNOWN_INSTRUMENT,
                          f"{len(labels)} instrument(s) nothing this client may read names in the store: "
                          f"{_listed(labels)}")
    if kind in ("no_data", STORE_NO_DATA):
        return tool_error(MARKET_NO_DATA, f"no stored price this client may read for {_listed(labels)} in the range"
                                          + (f" from {source}" if source else ""))
    if kind in ("restricted", STORE_SOURCE_RESTRICTED):
        return tool_error(MARKET_SOURCE_RESTRICTED,
                          f"{source!r} is a restricted source this client may not read from the store "
                          "(MOF_MARKET_RESTRICTED_ALLOW); omit source to read the sources it may")
    if kind == "ambiguous":
        return tool_error(MARKET_AMBIGUOUS_INSTRUMENT, f"{_listed(labels)} name several instruments: pass a "
                                                       "FIGI or an instrument id")
    if kind == "conflict":  # the server's own words: the caller's inputs only, and the argument it has
        return tool_error(INVALID_ARGUMENT, f"several sources (or two series of one source) give {_listed(labels)} "
                                            "different prices on one session: pass source to read one source")
    return tool_error(BACKEND_UNAVAILABLE, BACKEND_MESSAGE)  # store_unavailable


def _route(answers: dict, labels: list[str], how: str, source: str | None):
    """``(routes, store_rows)``: each input the store cannot answer goes live in ``auto`` (with its reason)
    or fails the call in ``store``."""
    if answers.get("refusal"):
        kind, hit = answers["refusal"]
        raise _store_refusal(kind, hit, source)
    routes = {label: _REASONS[answers["failed"][label]] for label in labels if label in answers["failed"]}
    if how == STORE and routes:
        reason = next(iter(routes.values()))
        raise _store_refusal(reason, [k for k, v in routes.items() if v == reason], source)
    return routes, _store_rows(answers["table"], answers["raw"])


# --- live (market-data's none profile) ----------------------------------------------------------------
class _OverBudget(Exception):
    """A wait the adapter asked for would end after the call's time budget (`_Budget.sleep`)."""


class _Budget:
    """The time budget of one live fetch (MOF_MARKET_LIVE_TIMEOUT_S from the call's start). Its ``sleep`` is
    the adapter's: a wait that fits is slept, a longer one is refused at once (recorded in ``over``, then
    ``_OverBudget``, which the adapter wraps in its own error). ``max_wait_s``: the adapter's cap on a retry
    wait, the time left (at most the adapter's own default)."""

    def __init__(self, seconds: float, *, default_max_wait_s: float):
        self.seconds = seconds
        self._deadline = time.monotonic() + seconds
        self._default = default_max_wait_s
        self.over: float | None = None

    def left(self) -> float:
        return self._deadline - time.monotonic()

    @property
    def max_wait_s(self) -> float:
        return max(min(self.left(), self._default), 0.001)

    def sleep(self, seconds: float) -> None:
        if seconds > self.left():
            self.over = seconds
            raise _OverBudget
        time.sleep(seconds)


def _live_adapter(code: str, budget: _Budget):
    """The live adapter of ``code`` keyed from this process's environment (the caller's own key over stdio),
    its waits bounded by ``budget``."""
    if not os.environ.get(KEY_ENVS[code], "").strip():
        raise api.MissingKeyError(f"no {code} key")  # the message is not forwarded (`_live_error`)
    cls = tiingo.TiingoAdapter if code == tiingo.SOURCE_CODE else massive.MassiveAdapter
    return cls.from_env(sleep=budget.sleep, max_wait_s=budget.max_wait_s)


class _OutsidePlan(Exception):
    def __init__(self, days: int, oldest: dt.date):
        super().__init__(days, oldest)
        self.days, self.oldest = days, oldest


def _fetch_live(labels, start, end, code, adjust, factory: AdapterFactory, budget: _Budget):
    adapter = factory(code, budget)
    try:
        days = getattr(adapter, "max_history_days", None)  # Massive's plan window: refused before any request
        if isinstance(days, int) and not isinstance(days, bool):
            oldest = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=days)
            if start < oldest:
                raise _OutsidePlan(days, oldest)
        return api.prices(labels, start, end, source=code, adjust=adjust, profile="none", adapter=adapter)
    finally:
        if hasattr(adapter, "close"):
            adapter.close()


class _LiveSlots:
    """MOF_MARKET_LIVE_CONCURRENCY slots for live fetches. A slot is taken before the worker thread is asked
    and released when the fetch itself ends, so an abandoned (timed-out) fetch keeps it: no new fetch runs
    beside one still spending the key. A fetch cancelled before its thread started never runs."""

    def __init__(self, n: int):
        self._slots = threading.BoundedSemaphore(n)

    def take(self) -> bool:
        return self._slots.acquire(blocking=False)

    async def run(self, fn, seconds: float):
        """``fn()`` off the event loop within ``seconds``, in a slot taken by :meth:`take`."""
        lock, state = threading.Lock(), {"started": False, "skipped": False}

        def guarded():
            with lock:
                if state["skipped"]:
                    return None
                state["started"] = True
            try:
                return fn()
            finally:
                self._slots.release()

        try:
            return await _off_loop(guarded, seconds)
        finally:
            with lock:
                if not state["started"]:  # cancelled before its thread ran: it never will
                    state["skipped"] = True
                    self._slots.release()


def _wait_error(code: str, seconds: float, budget: _Budget):
    return tool_error(LIMIT_EXCEEDED, f"{code} asks to wait about {round(seconds)} s before its next request "
                                      f"(its request budget), beyond this call's {budget.seconds:g} s "
                                      "(MOF_MARKET_LIVE_TIMEOUT_S): try again later, or ask for fewer "
                                      "instruments")


def _live_error(exc: BaseException, code: str, budget: _Budget):
    env = KEY_ENVS.get(code, "the source's key variable")
    if budget.over is not None:  # a wait refused by `_Budget.sleep`, whatever the adapter wrapped it in
        return _wait_error(code, budget.over, budget)
    if isinstance(exc, _OutsidePlan):
        return tool_error(INVALID_ARGUMENT, f"{code}'s plan serves the last {exc.days} days only (from "
                                            f"{exc.oldest.isoformat()}): start on or after it, or use source "
                                            "tiingo (the deep free history)")
    if isinstance(exc, api.MissingKeyError):
        return tool_error(MARKET_KEY_MISSING, f"no {code} API key in this server's environment: set {env} "
                                              "(your own key; it is never a tool argument)")
    if isinstance(exc, massive.MassiveKeyError) and str(exc).startswith("Massive answered HTTP 403"):
        return tool_error(LIMIT_EXCEEDED, f"{code} refused the request (HTTP 403): it is outside the plan of "
                                          f"this server's key (its history window, or a session not published "
                                          f"yet); a refused key is HTTP 401 (check {env})")
    if isinstance(exc, _KEY_ERRORS):
        return tool_error(MARKET_KEY_MISSING, f"{code} refused the API key in {env}: check it")
    if isinstance(exc, api.UnknownInstrumentError):
        return tool_error(MARKET_UNKNOWN_INSTRUMENT, f"{code} knows no such symbol: {_listed(exc.inputs)}")
    if isinstance(exc, api.NoDataError):
        return tool_error(MARKET_NO_DATA, str(exc))  # names the caller's inputs and the range only
    if isinstance(exc, api.AmbiguousInstrumentError):
        return tool_error(MARKET_AMBIGUOUS_INSTRUMENT, str(exc))
    if isinstance(exc, _QUOTA_ERRORS):
        return tool_error(LIMIT_EXCEEDED, f"the request cap of this server's {code} key is reached; try later")
    wait = _ADAPTER_WAIT.search(str(exc)) if isinstance(exc, (tiingo.TiingoError, massive.MassiveError)) else None
    if wait is not None:  # a Retry-After beyond the time left (the adapter's max_wait_s)
        return _wait_error(code, float(wait.group(1)), budget)
    log.warning("market: live fetch failed (%s)", type(exc).__name__)  # the class only
    return tool_error(BACKEND_UNAVAILABLE, LIVE_MESSAGE)


def _live_rows(table, code: str) -> list[dict]:
    return [{**r, "source_code": r.get("source_code", code), "source": LIVE} for r in table.to_pylist()]


def _store_only(labels: list[str]) -> list[str]:
    """Inputs only the store resolves, by their form: an instrument id (``mof:``) or a ``source:symbol``."""
    return [label for label in labels if ":" in label]


def build_market_toolset(store, settings: Settings, *, auth_enabled: bool,
                         adapter_factory: AdapterFactory | None = None) -> FastMCP:
    """``store``: the CompaniesStore (its open-data-store holds the market datasets) or None.
    ``adapter_factory``: tests only. Reads the source registry once (a ConfigError if it is broken)."""
    restricted = registered_restricted_sources(TOOLSET)
    registered = registered_sources(TOOLSET)
    factory = adapter_factory or _live_adapter
    slots = _LiveSlots(settings.market_live_concurrency)
    mk = FastMCP("market", mask_error_details=True)
    read = tier_check("read", auth_enabled=auth_enabled)

    @mk.tool(name="prices", auth=read, annotations=ANNOTATIONS)
    async def market_prices(
        instruments: Annotated[list[Annotated[str, Field(max_length=MAX_INSTRUMENT_CHARS)]],
                               Field(description=INSTRUMENTS_DESCRIPTION)],
        start: Annotated[str, Field(description="First session, ISO date YYYY-MM-DD", max_length=10)],
        end: Annotated[str, Field(description="Last session (inclusive), ISO date YYYY-MM-DD", max_length=10)],
        source: Annotated[str | None, Field(description=SOURCE_DESCRIPTION, max_length=63)] = None,
        adjust: Annotated[Literal["raw", "total_return"], Field(
            description="raw: closes, volumes, currency; total_return: an index based 1.0 on the first "
                        "session of the range with a price (dividends reinvested, splits applied)")] = "raw",
        mode: Annotated[Literal["auto", "store", "live"] | None, Field(description=MODE_DESCRIPTION)] = None,
    ) -> dict:
        """Daily prices of a few instruments over a date range, raw or as a total-return index, from the
        store or fetched live (see `mode`). Every row says where it comes from (`source`: store or live,
        `reason` when auto fetched live) and its provider (`source_code`). Stored rows are keyed by
        instrument_id only; live rows also carry the `input` they answer. Restricted providers are served
        only to the clients allowed them. Refused, never cut, above the server's instrument, row and size
        caps."""
        labels = _labels(instruments, settings)
        first, last = _date(start, "start"), _date(end, "end")
        if first > last:
            raise tool_error(INVALID_ARGUMENT, "start is after end")
        if source is not None and not _SOURCE_CODE.fullmatch(source):
            raise tool_error(INVALID_ARGUMENT, "source must be a provider code: a-z 0-9 _, starting with a letter")
        how = mode or settings.market_source
        if how == STORE and store is None:
            raise tool_error(INVALID_ARGUMENT, "this server has no market store (MOF_COMPANIES_CATALOG unset): "
                                               "use mode auto or live")
        live_code = source or api.DEFAULT_LIVE_SOURCE
        live_possible = live_code in api.LIVE_SOURCES
        if how == LIVE and not live_possible:
            raise tool_error(INVALID_ARGUMENT, f"live fetches take source {' or '.join(api.LIVE_SOURCES)} "
                                               f"(default {api.DEFAULT_LIVE_SOURCE})")
        client = caller_client_id(auth_enabled=auth_enabled)

        routes: dict[str, str | None] = {}  # input -> live reason (None: mode live); absent: the store
        rows: list[dict] = []
        if how == LIVE:
            routes = dict.fromkeys(labels)
        elif store is None:
            routes = dict.fromkeys(labels, NO_STORE)
        else:
            allowed = allowed_restricted_for(client, settings)
            order = None if source else precedence_for(settings, allowed, restricted, registered)
            fn = partial(_store_answers, store, labels, first, last, source, adjust, allowed, order, how)
            try:
                answers = await _off_loop(fn, settings.companies_query_timeout_s)
            except Exception as exc:  # noqa: BLE001 -- the store failed: its text is never forwarded
                log.warning("market: store read failed (%s)", type(exc).__name__)
                if how == STORE:
                    raise (exc if _is_timeout(exc) else _store_refusal(STORE_UNAVAILABLE, labels, source)) from None
                answers = {"table": None, "raw": None, "failed": dict.fromkeys(labels, "unavailable"),
                           "refusal": None}
            routes, rows = _route(answers, labels, how, source)

        if routes:
            live_labels = list(routes)
            if not live_possible:  # auto with a provider no live source is: the store's answer stands
                reason = next(iter(routes.values()))
                if reason == NO_STORE:
                    raise tool_error(INVALID_ARGUMENT, f"this server has no market store, and live fetches take "
                                                       f"source {' or '.join(api.LIVE_SOURCES)}")
                raise _store_refusal(reason, [k for k, v in routes.items() if v == reason], source)
            if live_code in restricted and live_code not in live_restricted_for(client, settings, restricted):
                # first, whatever the names: no answer of a client not allowed live depends on them
                raise tool_error(MARKET_SOURCE_RESTRICTED,
                                 f"a live fetch from {live_code} uses this server's key: serving it is "
                                 "redistribution, and this client is not allowed it (MOF_MARKET_RESTRICTED_ALLOW)")
            ciks = [label for label in live_labels if _CIK.fullmatch(label)]
            if ciks:
                raise tool_error(MARKET_UNKNOWN_INSTRUMENT,
                                 f"{_listed(ciks)}: a CIK names an instrument through the store's identifiers "
                                 "only; live sources take tickers")
            only_store = _store_only(live_labels)
            if only_store:
                raise tool_error(MARKET_UNKNOWN_INSTRUMENT,
                                 f"{_listed(only_store)}: an instrument id or a source:symbol names an instrument "
                                 "through the store only; live sources take tickers")
            budget = _Budget(settings.market_live_timeout_s,
                             default_max_wait_s=(tiingo if live_code == tiingo.SOURCE_CODE
                                                 else massive).DEFAULT_MAX_WAIT_S)
            fetch = partial(_fetch_live, live_labels, first, last, live_code, adjust, factory, budget)
            if not slots.take():
                raise tool_error(MARKET_BUSY, f"{settings.market_live_concurrency} live fetch(es) already running "
                                              "(MOF_MARKET_LIVE_CONCURRENCY, an abandoned one included until it "
                                              "ends); try again in a moment")
            try:
                table = await slots.run(fetch, settings.market_live_timeout_s)
            except Exception as exc:  # noqa: BLE001 -- mapped to a code; a fixed message for the unexpected
                raise (exc if _is_timeout(exc) else _live_error(exc, live_code, budget)) from None
            by_input: dict[str, list[dict]] = {}
            for row in _live_rows(table, live_code):
                by_input.setdefault(row["input"], []).append(row)
            for label in live_labels:
                reason = routes[label]
                rows += [{**r, **({"reason": reason} if reason else {})} for r in by_input.get(label, [])]

        if len(rows) > settings.market_max_rows:
            raise tool_error(RESULT_TOO_LARGE, f"{len(rows)} rows (cap {settings.market_max_rows}); {SIZE_HINT}")
        inputs = [{"input": label, "source": LIVE if label in routes else STORE,
                   **({"reason": routes[label]} if routes.get(label) else {})} for label in labels]
        return cap_result({"mode": how, "adjust": adjust, "start": first.isoformat(), "end": last.isoformat(),
                           "returned": len(rows), "inputs": inputs, "rows": rows}, settings, hint=SIZE_HINT)

    return mk


def _is_timeout(exc: BaseException) -> bool:
    return str(exc).startswith(f"[{TIMEOUT}]")
