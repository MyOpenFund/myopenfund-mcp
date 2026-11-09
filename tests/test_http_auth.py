"""Bearer tokens and tiers end to end: a real uvicorn server on a free loopback port."""
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import anyio
import pytest
import uvicorn
from fakes import FakeVaultReader
from fastmcp import Client

from myopenfund_mcp.auth import hash_token, tier_check
from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio

ALICE_TOKEN = "t-alice-" + "a" * 40
AGENT_TOKEN = "t-agent-" + "a" * 40
START_TIMEOUT_S = 10.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ListingOnlyStore:
    """Enough of a CompaniesStore to mount the companies and datasets toolsets (listing tools
    never reads the store)."""

    def freshness(self):
        return {"batches": 0, "last_published_at": None}


@pytest.fixture
def http_url():
    port = _free_port()
    settings = Settings(transport="http", port=port,
                        tokens=f"alice:{hash_token(ALICE_TOKEN)}:act,agent:{hash_token(AGENT_TOKEN)}:read")
    server = build_server(settings, vault_reader=FakeVaultReader(), companies_store=_ListingOnlyStore())

    # v1 ships no act-only tool: a throwaway one proves tier enforcement over HTTP.
    @server.tool(auth=tier_check("act", auth_enabled=True))
    async def act_only_probe() -> str:
        return "acted"

    app = server.http_app(path=settings.path)
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + START_TIMEOUT_S
    while not uv.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            uv.should_exit = True
            thread.join(timeout=5)
            pytest.fail(f"uvicorn did not start on 127.0.0.1:{port}")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}{settings.path}"
    finally:
        uv.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "uvicorn did not shut down"


async def test_read_token_gets_read_tier_and_vault_tools(http_url):
    async with Client(http_url, auth=AGENT_TOKEN) as c:
        assert (await c.call_tool("server_info", {})).data["tier"] == "read"
        assert (await c.call_tool("vault_search_documents", {})).data["total"] == 4


async def test_read_token_lists_the_companies_and_datasets_tools(http_url):
    async with Client(http_url, auth=AGENT_TOKEN) as c:
        names = {t.name for t in await c.list_tools()}
    assert {"companies_resolve", "companies_facts", "companies_fundamentals"} <= names
    assert {"datasets_list", "datasets_describe", "datasets_read_asof", "datasets_export_csv"} <= names


async def test_act_token_gets_act_tier(http_url):
    async with Client(http_url, auth=ALICE_TOKEN) as c:
        assert (await c.call_tool("server_info", {})).data["tier"] == "act"
        assert "act_only_probe" in {t.name for t in await c.list_tools()}
        assert (await c.call_tool("act_only_probe", {})).data == "acted"


async def test_read_tier_cannot_see_or_call_an_act_tool(http_url):
    async with Client(http_url, auth=AGENT_TOKEN) as c:
        names = {t.name for t in await c.list_tools()}
        assert "server_info" in names and "vault_search_documents" in names
        assert "act_only_probe" not in names
        with pytest.raises(Exception, match="act_only_probe"):  # hidden: unknown to this caller
            await c.call_tool("act_only_probe", {})


async def test_document_resource_readable_with_read_token(http_url):
    async with Client(http_url, auth=AGENT_TOKEN) as c:
        contents = await c.read_resource("vault://documents/us1")
    assert json.loads(contents[0].text)["doc_id"] == "us1"


async def test_wrong_token_is_refused(http_url):
    with pytest.raises(Exception):  # noqa: B017 - transport-level refusal, type is fastmcp's choice
        async with Client(http_url, auth="wrong-" + "w" * 40) as c:
            await c.list_tools()


async def test_missing_token_is_refused(http_url):
    with pytest.raises(Exception):  # noqa: B017
        async with Client(http_url) as c:
            await c.list_tools()


def _post_status(url: str, body: dict, headers: dict) -> int:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream", **headers})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # loopback: never via a proxy
    try:
        with opener.open(request, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic eDp5"}])
async def test_refusals_are_http_401(http_url, headers):
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "probe", "version": "0"}}}
    assert await anyio.to_thread.run_sync(_post_status, http_url, body, headers) == 401


def _get(url: str) -> tuple[int, str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # loopback: never via a proxy
    try:
        with opener.open(url, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


async def test_health_answers_ok_without_a_token(http_url):
    health = http_url.rsplit("/", 1)[0] + "/health"
    assert await anyio.to_thread.run_sync(_get, health) == (200, "ok")


async def test_mcp_still_refuses_without_a_token_next_to_health(http_url):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    assert await anyio.to_thread.run_sync(_post_status, http_url, body, {}) == 401
