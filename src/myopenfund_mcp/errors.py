"""Tool errors with a stable code first, so agents can branch on it: `[code] message`."""
import re

from fastmcp.exceptions import ToolError

INVALID_ARGUMENT = "invalid_argument"
NOT_FOUND = "not_found"
BACKEND_UNAVAILABLE = "backend_unavailable"
TIMEOUT = "timeout"
SCHEMA_OUTDATED = "schema_outdated"
RESULT_TOO_LARGE = "result_too_large"
AMBIGUOUS_COMPANY = "ambiguous_company"
UNKNOWN_COMPANY = "unknown_company"
UNKNOWN_CONCEPT = "unknown_concept"
UNKNOWN_METRIC = "unknown_metric"
UNKNOWN_DATASET = "unknown_dataset"
UNKNOWN_COLUMN = "unknown_column"
TOO_MANY_COMPANIES = "too_many_companies"
RAG_UNAVAILABLE = "rag_unavailable"
LIMIT_EXCEEDED = "limit_exceeded"
SEC_CONTACT_MISSING = "sec_contact_missing"
SEC_REFUSED = "sec_refused"
SEC_UNAVAILABLE = "sec_unavailable"
WARMING = "warming"
NOT_FETCHED = "not_fetched"
INDEX_NOT_READY = "index_not_ready"
INDEX_UNAVAILABLE = "index_unavailable"
NO_FACTS = "no_facts"
# market_prices (market-data spec §10, §13)
MARKET_KEY_MISSING = "market_key_missing"
MARKET_SOURCE_RESTRICTED = "market_source_restricted"
MARKET_UNKNOWN_INSTRUMENT = "market_unknown_instrument"
MARKET_AMBIGUOUS_INSTRUMENT = "market_ambiguous_instrument"
MARKET_NO_DATA = "market_no_data"
MARKET_BUSY = "market_busy"
TOO_MANY_INSTRUMENTS = "too_many_instruments"


def tool_error(code: str, message: str) -> ToolError:
    return ToolError(f"[{code}] {message}")


_CODE = re.compile(r"\[([a-z_]+)\] ")


def error_code(exc: BaseException, default: str) -> str:
    """The `[code]` of one of our ToolErrors; `default` for any other exception (never its text)."""
    match = _CODE.match(str(exc)) if isinstance(exc, ToolError) else None
    return match.group(1) if match else default
