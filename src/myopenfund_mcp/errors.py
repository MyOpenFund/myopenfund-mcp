"""Tool errors with a stable code first, so agents can branch on it: `[code] message`."""
from fastmcp.exceptions import ToolError

INVALID_ARGUMENT = "invalid_argument"
NOT_FOUND = "not_found"
BACKEND_UNAVAILABLE = "backend_unavailable"
TIMEOUT = "timeout"
RESULT_TOO_LARGE = "result_too_large"


def tool_error(code: str, message: str) -> ToolError:
    return ToolError(f"[{code}] {message}")
