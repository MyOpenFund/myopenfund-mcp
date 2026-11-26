"""A built server on a real uvicorn, on a free loopback port: bearer tokens reach the tools as in production."""
from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

import uvicorn

START_TIMEOUT_S = 10.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve(server, port: int, path: str = "/mcp"):
    """Run `server` (built with transport http on `port`) until the block ends; yields its MCP URL."""
    uv = uvicorn.Server(uvicorn.Config(server.http_app(path=path), host="127.0.0.1", port=port,
                                       log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + START_TIMEOUT_S
    while not uv.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            uv.should_exit = True
            thread.join(timeout=5)
            raise RuntimeError(f"uvicorn did not start on 127.0.0.1:{port}")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}{path}"
    finally:
        uv.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "uvicorn did not shut down"
