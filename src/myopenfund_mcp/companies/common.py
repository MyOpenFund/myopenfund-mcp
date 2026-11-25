"""Shared by the companies and datasets toolsets: time budget, error mapping, as_of, UTC rows."""
from __future__ import annotations

import datetime as dt
import logging
from contextlib import contextmanager
from functools import partial

import anyio
from fastmcp.exceptions import ToolError

from ..errors import (
    BACKEND_UNAVAILABLE,
    INVALID_ARGUMENT,
    NO_FACTS,
    SEC_CONTACT_MISSING,
    SEC_REFUSED,
    SEC_UNAVAILABLE,
    TIMEOUT,
    UNKNOWN_COLUMN,
    UNKNOWN_DATASET,
    WARMING,
    tool_error,
)
from .live import LiveClosed, LiveDataError, NoFacts, Warming, sec_error_code
from .router import LiveNotConfigured
from .store import StoreBadArgument, StoreUnknownDataset

log = logging.getLogger("myopenfund_mcp.companies")
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
AS_OF_DESCRIPTION = ("Point in time, ISO 8601 (UTC if no offset); only what was publicly known then "
                     "is returned; omitted = now")
BACKEND_MESSAGE = "the company data store cannot be read right now; try again later"
LIVE_BACKEND_MESSAGE = "the SEC data cannot be read right now; try again later"
#: Fixed messages: an SEC error's own text is never forwarded (nor is the contact, ever).
SEC_MESSAGES = {
    SEC_CONTACT_MISSING: "SEC access needs a contact: set MOF_SEC_CONTACT (e.g. an email address)",
    SEC_REFUSED: "SEC refused the requests as an undeclared automated tool; check MOF_SEC_CONTACT",
    SEC_UNAVAILABLE: "SEC is unreachable or throttling after the retries; try again later",
}


class WarmingError(ToolError):
    """``[warming]`` for one company: SEC requests ``done`` of ``total``; the fetch goes on."""

    def __init__(self, done: int, total: int):
        super().__init__(f"[{WARMING}] still fetching from SEC: {done}/{total} requests done; the fetch "
                         "goes on in the background: call again in a few seconds")
        self.done, self.total = done, total


def parse_as_of(value: dt.datetime | None) -> dt.datetime:
    """The as_of instant in UTC: None = now, naive = UTC, aware = converted."""
    if value is None:
        return dt.datetime.now(dt.UTC)
    return value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value.astimezone(dt.UTC)


def _utc(value):
    """`value` with every tz-aware datetime in UTC, recursively through dicts, lists and tuples."""
    if isinstance(value, dt.datetime) and value.tzinfo is not None:
        return value.astimezone(dt.UTC)
    if isinstance(value, dict):
        return {k: _utc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_utc(v) for v in value)
    return value


def rows_to_utc(rows: list[dict]) -> list[dict]:
    """New rows with every tz-aware datetime (nested ones too) in UTC.

    The store's DuckDB cursors return instants in the host's time zone; results are always UTC.
    Naive datetimes and dates are left as they are; the input is never modified.
    """
    return [_utc(row) for row in rows]


def dict_to_utc(value: dict) -> dict:
    """One dict (e.g. freshness) with its tz-aware datetimes in UTC."""
    return _utc(value)


def _timeout(seconds: float) -> Exception:
    return tool_error(TIMEOUT, f"the company data did not come within {seconds} s; "
                               "narrow the request (fewer companies, a period range)")


@contextmanager
def time_budget(settings, seconds: float | None = None):
    """One time budget (`seconds`, default `companies_query_timeout_s`) for everything inside: a tool
    wraps all its store calls in it, so N companies cannot take N budgets. Its expiry is `[timeout]`.
    Yields the cancel scope: a tool may move its deadline once it knows which sources a call needs."""
    seconds = settings.companies_query_timeout_s if seconds is None else seconds
    start, scope = anyio.current_time(), None
    try:
        with anyio.fail_after(seconds) as scope:
            yield scope
    except TimeoutError:
        raise _timeout(round(scope.deadline - start, 1)) from None


def call_budget_s(settings, *, live: bool) -> float:
    """A call's largest time budget: `companies_query_timeout_s`, plus `sec_call_budget_s` when it may
    reach SEC (the live source answers `[warming]` after that budget, so the call must outlast it)."""
    return settings.companies_query_timeout_s + (settings.sec_call_budget_s if live else 0)


async def store_call(fn, /, *, settings, timeout_s: float | None = None, **kwargs):
    """Run a blocking store (or router, or live source) method off the event loop within the time
    budget (and within any enclosing `time_budget`); map its errors to codes (:func:`coded`)."""
    try:
        with time_budget(settings, timeout_s):  # nested in a tool's own budget, the earlier deadline wins
            return await anyio.to_thread.run_sync(partial(fn, **kwargs), abandon_on_cancel=True)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 -- every other error gets its code
        raise coded(exc) from None


def coded(exc: BaseException) -> ToolError:
    """The `[code]` ToolError of a store, router or live-source error.

    Backend failures never forward their message (it may hold a DSN; an SEC error never holds the
    contact, but its text is not forwarded either): the class only is logged.
    """
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, Warming):
        return WarmingError(exc.done, exc.total)
    if isinstance(exc, NoFacts):
        return tool_error(NO_FACTS, f"SEC has no XBRL financial facts for CIK {exc.cik} (it files no "
                                    "financial statements in XBRL, e.g. a fund or a foreign private issuer "
                                    "before XBRL)")
    if isinstance(exc, LiveNotConfigured):
        return tool_error(SEC_CONTACT_MISSING, f"CIK {exc.cik} must be read from SEC ({exc.reason}) and SEC "
                                               "access is not configured: set MOF_SEC_CONTACT (e.g. an email "
                                               "address)")
    if isinstance(exc, (LiveDataError, LiveClosed)):
        log.warning("live SEC call failed (%s)", type(exc).__name__)
        return tool_error(BACKEND_UNAVAILABLE, LIVE_BACKEND_MESSAGE)
    if isinstance(exc, StoreUnknownDataset):
        return tool_error(UNKNOWN_DATASET, f"{exc}; see datasets_list")
    if isinstance(exc, StoreBadArgument):
        msg = str(exc)
        if msg.startswith("unknown column"):
            return tool_error(UNKNOWN_COLUMN, f"{msg}; see datasets_describe")
        return tool_error(INVALID_ARGUMENT, msg)
    code = sec_error_code(exc)
    if code is not None:
        log.warning("SEC call failed (%s)", type(exc).__name__)
        return tool_error(code, SEC_MESSAGES[code])
    log.warning("store call failed (%s)", type(exc).__name__)  # StoreBackendError and anything else
    return tool_error(BACKEND_UNAVAILABLE, BACKEND_MESSAGE)
