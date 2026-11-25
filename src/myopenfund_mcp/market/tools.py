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
``[market_busy]`` at once, never queued. A fetch past its time budget is abandoned, not stopped (a worker
thread cannot be): it may still finish and spend its quota, and it keeps its slot until it ends.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import threading
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

MODES = ("auto", "store", "live")
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
_LISTED = 20  # inputs quoted in a message
ANNOTATIONS = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True}
SIZE_HINT = "narrow the date range or ask for fewer instruments"
BACKEND_MESSAGE = "the market data store cannot be read right now; try again later"
LIVE_MESSAGE = "the live market data source cannot be read right now; try again later"
INSTRUMENTS_DESCRIPTION = (f"Tickers or FIGIs (CIKs and mof: instrument ids through the store only), at most the "
                           f"server maximum per call, each at most {MAX_INSTRUMENT_CHARS} characters")
MODE_DESCRIPTION = ("auto: the store when it holds the instrument over the range, else a live fetch; store: never "
                    "live; live: never the store. Omitted = the server's MOF_MARKET_SOURCE")
SOURCE_DESCRIPTION = ("A provider code to read from (e.g. tiingo, massive, private_history); live fetches take "
                      "tiingo (default) or massive. Omitted = any source this client may read")

#: `source_code -> Adapter`: an adapter instead of the one market-data builds from the environment (tests)
AdapterFactory = Callable[[str], object]


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
def _store_answers(store, labels, start, end, source, adjust, allowed) -> dict[str, tuple]:
    """Per input: ``("rows", table, raw_or_None)`` or ``(kind, ApiError)``; one store call, one input at a
    time (the server profile returns instrument ids only: per input is how a route is known per input).
    With a total return, the raw rows of the same read give each session's provider."""

    def kind(exc: api.ApiError) -> str:
        for cls, name in ((api.UnknownInstrumentError, "unknown"), (api.NoDataError, "no_data"),
                          (api.RestrictedSourceError, "restricted"),
                          (api.AmbiguousInstrumentError, "ambiguous")):
            if isinstance(exc, cls):
                return name
        return "conflict"

    def go(s):
        out: dict[str, tuple] = {}
        for label in labels:
            read = partial(api.prices, [label], start, end, source=source, profile="server", store=s,
                           allowed_restricted=allowed)
            try:
                table = read(adjust=adjust)
                raw = read(adjust="raw") if adjust == "total_return" else None
            except api.ApiError as exc:  # handed back: a ValueError would read as the store's bad argument
                out[label] = (kind(exc), exc)
            else:
                out[label] = ("rows", table, raw)
        return out

    return store.call(go)


def _store_rows(answer: tuple) -> list[dict]:
    table, raw = answer[1], answer[2]
    rows = table.to_pylist()
    if raw is not None:  # total return: the provider of each session's price
        provider = {(r["instrument_id"], r["session_date"]): r["source_code"] for r in raw.to_pylist()}
        for r in rows:
            r["source_code"] = provider.get((r["instrument_id"], r["session_date"]))
    return [{**r, "source": STORE} for r in rows]


def _store_refusal(kind: str, labels: list[str], exc: api.ApiError | None, source: str | None):
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
    if kind == "conflict":  # market-data's message names the caller's inputs only
        return tool_error(INVALID_ARGUMENT, str(exc))
    return tool_error(BACKEND_UNAVAILABLE, BACKEND_MESSAGE)  # store_unavailable


# --- live (market-data's none profile) ----------------------------------------------------------------
def _fetch_live(labels, start, end, code, adjust, factory: AdapterFactory | None):
    adapter = factory(code) if factory is not None else None
    try:
        return api.prices(labels, start, end, source=code, adjust=adjust, profile="none", adapter=adapter)
    finally:
        if adapter is not None and hasattr(adapter, "close"):
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


def _live_error(exc: BaseException, code: str):
    env = KEY_ENVS.get(code, "the source's key variable")
    if isinstance(exc, api.MissingKeyError):
        return tool_error(MARKET_KEY_MISSING, f"no {code} API key in this server's environment: set {env} "
                                              "(your own key; it is never a tool argument)")
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
    log.warning("market: live fetch failed (%s)", type(exc).__name__)  # the class only
    return tool_error(BACKEND_UNAVAILABLE, LIVE_MESSAGE)


def _live_rows(table, code: str, reason: str | None) -> list[dict]:
    extra = {"source": LIVE, **({"reason": reason} if reason else {})}
    return [{**r, "source_code": r.get("source_code", code), **extra} for r in table.to_pylist()]


def build_market_toolset(store, settings: Settings, *, auth_enabled: bool,
                         adapter_factory: AdapterFactory | None = None) -> FastMCP:
    """``store``: the CompaniesStore (its open-data-store holds the market datasets) or None.
    ``adapter_factory``: tests only. Reads the source registry once (a ConfigError if it is broken)."""
    restricted = registered_restricted_sources()
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
            fn = partial(_store_answers, store, labels, first, last, source, adjust, allowed)
            try:
                answers = await _off_loop(fn, settings.companies_query_timeout_s)
            except Exception as exc:  # noqa: BLE001 -- the store failed: its text is never forwarded
                log.warning("market: store read failed (%s)", type(exc).__name__)
                if how == STORE:
                    raise (exc if _is_timeout(exc) else _store_refusal(STORE_UNAVAILABLE, labels, None,
                                                                       source)) from None
                answers = dict.fromkeys(labels, ("unavailable", None))
            routes, rows = _route(answers, labels, how, source)

        if routes:
            live_labels = list(routes)
            if not live_possible:  # auto with a provider no live source is: the store's answer stands
                reason = next(iter(routes.values()))
                if reason == NO_STORE:
                    raise tool_error(INVALID_ARGUMENT, f"this server has no market store, and live fetches take "
                                                       f"source {' or '.join(api.LIVE_SOURCES)}")
                raise _store_refusal(reason, [k for k, v in routes.items() if v == reason], None, source)
            ciks = [label for label in live_labels if _CIK.fullmatch(label)]
            if ciks:
                raise tool_error(MARKET_UNKNOWN_INSTRUMENT,
                                 f"{_listed(ciks)}: a CIK names an instrument through the store's identifiers "
                                 "only; live sources take tickers")
            if live_code in restricted and live_code not in live_restricted_for(client, settings, restricted):
                raise tool_error(MARKET_SOURCE_RESTRICTED,
                                 f"a live fetch from {live_code} uses this server's key: serving it is "
                                 "redistribution, and this client is not allowed it (MOF_MARKET_RESTRICTED_ALLOW)")
            fetch = partial(_fetch_live, live_labels, first, last, live_code, adjust, adapter_factory)
            if not slots.take():
                raise tool_error(MARKET_BUSY, f"{settings.market_live_concurrency} live fetch(es) already running "
                                              "(MOF_MARKET_LIVE_CONCURRENCY, an abandoned one included until it "
                                              "ends); try again in a moment")
            try:
                table = await slots.run(fetch, settings.market_live_timeout_s)
            except Exception as exc:  # noqa: BLE001 -- mapped to a code; a fixed message for the unexpected
                raise (exc if _is_timeout(exc) else _live_error(exc, live_code)) from None
            by_input: dict[str, list[dict]] = {}
            for row in _live_rows(table, live_code, None):
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


def _route(answers: dict[str, tuple], labels: list[str], how: str, source: str | None):
    """``(routes, store_rows)``: each input the store cannot answer goes live in ``auto`` (with its reason)
    or fails the call in ``store``; stored rows of one instrument named by two inputs are kept once."""
    reasons = {"unknown": NOT_IN_STORE, "no_data": STORE_NO_DATA, "restricted": STORE_SOURCE_RESTRICTED,
               "unavailable": STORE_UNAVAILABLE}
    routes: dict[str, str] = {}
    rows: list[dict] = []
    seen: set = set()
    for kind in ("ambiguous", "conflict"):  # the caller's to fix, whatever the mode
        hit = [label for label in labels if answers[label][0] == kind]
        if hit:
            raise _store_refusal(kind, hit, answers[hit[0]][1], source)
    for label in labels:
        answer = answers[label]
        if answer[0] == "rows" and (answer[1].num_rows or how == STORE):
            for r in _store_rows(answer):
                key = (r["instrument_id"], r["session_date"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
            continue
        routes[label] = STORE_NO_DATA if answer[0] == "rows" else reasons[answer[0]]
    if how == STORE and routes:
        reason = next(iter(routes.values()))
        raise _store_refusal(reason, [k for k, v in routes.items() if v == reason],
                             None, source)
    return routes, rows
