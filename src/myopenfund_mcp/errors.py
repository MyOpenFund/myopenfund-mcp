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


def tool_error(code: str, message: str) -> ToolError:
    return ToolError(f"[{code}] {message}")


_CODE = re.compile(r"\[([a-z_]+)\] ")


def error_code(exc: BaseException, default: str) -> str:
    """The `[code]` of one of our ToolErrors; `default` for any other exception (never its text)."""
    match = _CODE.match(str(exc)) if isinstance(exc, ToolError) else None
    return match.group(1) if match else default
