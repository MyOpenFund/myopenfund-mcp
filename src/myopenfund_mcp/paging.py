"""Cursor paging, row limits, result-size cap, JSON-safe values."""
from __future__ import annotations

import base64
import binascii
import datetime as dt
import json
import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .config import Settings
from .errors import INVALID_ARGUMENT, RESULT_TOO_LARGE, tool_error

# Largest offset a cursor may carry: the int4 bound vault-query's check_int applies to offsets.
MAX_OFFSET = 2**31 - 1

_BAD_CURSOR = (
    "cursor is not one this server issued; pass next_cursor from the previous page unchanged, "
    "or omit it to start over"
)


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
    """Validated row limit: `None` means `settings.default_rows`.

    Despite its name it refuses, never clamps: a non-integer or a value outside
    [1, settings.max_rows] raises `[invalid_argument]` rather than being silently brought into range.
    """
    if limit is None:
        return settings.default_rows
    if not _is_int(limit) or not 1 <= limit <= settings.max_rows:
        raise tool_error(INVALID_ARGUMENT, f"limit must be an integer between 1 and {settings.max_rows}")
    return limit


def _decimal(value: Decimal) -> int | float | None:
    if not value.is_finite():
        return None
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def jsonable(value: Any) -> Any:
    """Plain JSON values (dict / list / str / int / float / bool / None), whatever the driver returns.

    - Mappings (dict subclasses such as RealDictRow or OrderedDict) become plain dicts; lists and
      tuples become lists; both recursively.
    - dates/datetimes/times: ISO 8601 (naive stays naive).
    - timedelta: total seconds as a number (`timedelta(days=1)` -> 86400.0).
    - Decimal: integral and finite -> int (`Decimal("12345")` -> 12345), otherwise float.
    - bytes / bytearray / memoryview: standard base64, ASCII.
    - NaN / Infinity (float or Decimal): null, as the MCP serializer would send.
    - str / int / float / bool / None: unchanged.
    - anything else (UUID, ipaddress, psycopg Range...): `str(value)`.
    """
    if isinstance(value, Mapping):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int)):  # bool is an int subclass
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return _decimal(value)
    if isinstance(value, (dt.date, dt.time)):  # datetime is a date subclass
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(value).decode("ascii")
    return str(value)


def result_chars(payload: Any) -> int:
    """Characters of the text block the client receives: compact JSON, non-ASCII unescaped
    (the same string fastmcp's default serializer emits for a JSON-safe payload).

    Known divergence: floats with a negative exponent. Python writes `1e-07`, fastmcp's
    pydantic serializer writes `1e-7`, so the count may overstate by one character per such
    float; it never understates. Every other value (ints, positive exponents, strings,
    escapes, non-ASCII) matches exactly (tested through a real FastMCP tool call).
    """
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def cap_result(payload: dict, settings: Settings, hint: str = "ask for a smaller limit") -> dict:
    """Refuse — never truncate — a result larger than the configured cap.

    `hint` ends the `[result_too_large]` message: tell the caller how to get a smaller result
    for this particular tool.
    """
    payload = jsonable(payload)
    size = result_chars(payload)
    if size > settings.max_result_chars:
        raise tool_error(
            RESULT_TOO_LARGE,
            f"result is {size} characters (cap {settings.max_result_chars}); {hint}",
        )
    return payload
