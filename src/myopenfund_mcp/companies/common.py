"""Shared by the companies and datasets toolsets: time budget, error mapping, as_of, UTC rows."""
from __future__ import annotations

import datetime as dt
import logging
from functools import partial

import anyio

from ..errors import (
    BACKEND_UNAVAILABLE,
    INVALID_ARGUMENT,
    TIMEOUT,
    UNKNOWN_COLUMN,
    UNKNOWN_DATASET,
    tool_error,
)
from .store import StoreBadArgument, StoreUnknownDataset

log = logging.getLogger("myopenfund_mcp.companies")
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
AS_OF_DESCRIPTION = ("Point in time, ISO 8601 (UTC if no offset); only what was publicly known then "
                     "is returned; omitted = now")
BACKEND_MESSAGE = "the company data store cannot be read right now; try again later"


def parse_as_of(value: dt.datetime | None) -> dt.datetime:
    """The as_of instant in UTC: None = now, naive = UTC, aware = converted."""
    if value is None:
        return dt.datetime.now(dt.UTC)
    return value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value.astimezone(dt.UTC)


def _utc(value):
    if isinstance(value, dt.datetime) and value.tzinfo is not None:
        return value.astimezone(dt.UTC)
    return value


def rows_to_utc(rows: list[dict]) -> list[dict]:
    """New rows with every tz-aware datetime (top-level value) in UTC.

    The store's DuckDB cursors return instants in the host's time zone; results are always UTC.
    Naive datetimes and dates are left as they are.
    """
    return [{k: _utc(v) for k, v in row.items()} for row in rows]


def dict_to_utc(value: dict) -> dict:
    """One dict (e.g. freshness) with its top-level tz-aware datetimes in UTC."""
    return rows_to_utc([value])[0]


async def store_call(fn, /, *, settings, **kwargs):
    """Run a blocking store method off the event loop within the time budget; map its errors to codes.

    Backend failures never forward their message (it may hold a DSN): the class only is logged.
    """
    try:
        with anyio.fail_after(settings.companies_query_timeout_s):
            return await anyio.to_thread.run_sync(partial(fn, **kwargs), abandon_on_cancel=True)
    except TimeoutError:
        raise tool_error(TIMEOUT, f"the store did not answer within {settings.companies_query_timeout_s} s; "
                                  "narrow the request (fewer companies, a period range)") from None
    except StoreUnknownDataset as exc:
        raise tool_error(UNKNOWN_DATASET, f"{exc.args[0] if exc.args else exc}; see datasets_list") from None
    except StoreBadArgument as exc:
        msg = str(exc)
        if msg.startswith("unknown column"):
            raise tool_error(UNKNOWN_COLUMN, f"{msg}; see datasets_describe") from None
        raise tool_error(INVALID_ARGUMENT, msg) from None
    except Exception as exc:  # noqa: BLE001 -- StoreBackendError and anything else: fixed message
        log.warning("store call failed (%s)", type(exc).__name__)
        raise tool_error(BACKEND_UNAVAILABLE, BACKEND_MESSAGE) from None
