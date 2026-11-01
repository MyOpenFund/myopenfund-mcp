"""Cursor paging, row limits, result-size cap, JSON-safe values."""
from __future__ import annotations

import base64
import binascii
import datetime as dt
import json
import math
from decimal import Decimal
from typing import Any

from .config import Settings
from .errors import INVALID_ARGUMENT, RESULT_TOO_LARGE, tool_error

# Largest offset a cursor may carry: the int4 bound vault-query's check_int applies to offsets.
MAX_OFFSET = 2**31 - 1

_BAD_CURSOR = "cursor is not one this server issued; restart without it"


def _is_int(value: Any) -> bool:
    """A plain int: a bool is an int subclass but never a count."""
    return isinstance(value, int) and not isinstance(value, bool)


def encode_cursor(offset: int) -> str:
    if not _is_int(offset) or not 0 <= offset <= MAX_OFFSET:
        raise ValueError(f"cursor offset must be an int in [0, {MAX_OFFSET}], got {offset!r}")
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).decode()


def decode_cursor(cursor: str | None) -> int:
    """Offset carried by a cursor this server issued; anything else is refused, never guessed.

    Only the exact string `encode_cursor` produces is accepted (re-encoded and compared), so
    stripped padding, junk the base64 decoder would silently skip, or a hand-built payload fail.
    """
    if cursor is None:
        return 0
    try:
        offset = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))["o"]
    except (binascii.Error, ValueError, KeyError, TypeError, IndexError, RecursionError):
        offset = None
    if not _is_int(offset) or not 0 <= offset <= MAX_OFFSET or encode_cursor(offset) != cursor:
        raise tool_error(INVALID_ARGUMENT, _BAD_CURSOR)
    return offset


def clamp_limit(limit: int | None, settings: Settings) -> int:
    if limit is None:
        return settings.default_rows
    if not _is_int(limit) or not 1 <= limit <= settings.max_rows:
        raise tool_error(INVALID_ARGUMENT, f"limit must be an integer between 1 and {settings.max_rows}")
    return limit


def jsonable(value: Any) -> Any:
    """Plain JSON values: dates/datetimes as ISO 8601 (naive stays naive), Decimal as float,
    tuples as lists, NaN/Infinity as null (what the MCP serializer would send)."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dt.date):  # datetime is a date subclass
        return value.isoformat()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def result_chars(payload: Any) -> int:
    """Characters of the text block the client receives: compact JSON, non-ASCII unescaped
    (the same string fastmcp's default serializer emits for a JSON-safe payload)."""
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def cap_result(payload: dict, settings: Settings) -> dict:
    """Refuse — never truncate — a result larger than the configured cap."""
    payload = jsonable(payload)
    size = result_chars(payload)
    if size > settings.max_result_chars:
        raise tool_error(
            RESULT_TOO_LARGE,
            f"result is {size} characters (cap {settings.max_result_chars}); ask for a smaller limit",
        )
    return payload
