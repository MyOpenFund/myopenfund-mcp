"""The server with or without the [market] extra (CI runs this module in a job without it): it starts, the
market toolset is mounted exactly when the extra is installed, and the datasets tools keep working."""
import pytest
from companies_fixture import seed
from extras import MARKET_INSTALLED, MARKET_STATUS, MARKET_TOOLS
from fastmcp import Client

from myopenfund_mcp.companies.store import CompaniesStore
from myopenfund_mcp.config import ConfigError, Settings
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio


async def test_the_server_starts_and_mounts_market_prices_only_with_the_extra():
    async with Client(build_server(Settings())) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert {"market_prices"} & names == MARKET_TOOLS
    assert info["toolsets"]["market"] == MARKET_STATUS


def test_a_market_mode_without_the_extra_refuses_to_start():
    if MARKET_INSTALLED:
        pytest.skip("the [market] extra is installed")
    with pytest.raises(ConfigError, match=r"MOF_MARKET_SOURCE is set to live but the \[market\] extra"):
        build_server(Settings(market_source="live"))


async def test_the_datasets_tools_start_and_read(tmp_path):
    cfg = seed(tmp_path)
    store = CompaniesStore(str(cfg.data_root), cfg.catalog)
    try:
        async with Client(build_server(Settings(), companies_store=store)) as c:
            info = (await c.call_tool("server_info", {})).data
            listed = (await c.call_tool("datasets_list", {})).data
            read = (await c.call_tool("datasets_read_asof", {"dataset": "fundamentals", "limit": 2})).data
    finally:
        store.close()
    assert info["toolsets"]["datasets"] == "enabled" and info["toolsets"]["market"] == MARKET_STATUS
    assert {"fundamentals"} <= {d["name"] for d in listed["datasets"]}
    assert read["returned"] == 2  # SEC rows: no restricted source, served with or without the registry
