"""One log line per tool call: who, what, how long, how it ended. Never arguments (they may hold data)."""
import logging
import time

from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware

log = logging.getLogger("myopenfund_mcp.audit")


class AuditMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        token = get_access_token()
        client = token.client_id if token else "local"
        started = time.monotonic()
        outcome = "error"
        try:
            result = await call_next(context)
            outcome = "ok"
            return result
        finally:
            log.info("tool=%s client=%s outcome=%s ms=%d", context.message.name, client, outcome,
                     (time.monotonic() - started) * 1000)
