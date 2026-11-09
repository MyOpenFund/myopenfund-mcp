"""Tool errors with a stable code first, so agents can branch on it: `[code] message`."""
from fastmcp.exceptions import ToolError

INVALID_ARGUMENT = "invalid_argument"
NOT_FOUND = "not_found"
BACKEND_UNAVAILABLE = "backend_unavailable"
TIMEOUT = "timeout"
RESULT_TOO_LARGE = "result_too_large"
AMBIGUOUS_COMPANY = "ambiguous_company"
UNKNOWN_COMPANY = "unknown_company"
UNKNOWN_CONCEPT = "unknown_concept"
UNKNOWN_METRIC = "unknown_metric"
UNKNOWN_DATASET = "unknown_dataset"
UNKNOWN_COLUMN = "unknown_column"
TOO_MANY_COMPANIES = "too_many_companies"


def tool_error(code: str, message: str) -> ToolError:
    return ToolError(f"[{code}] {message}")
