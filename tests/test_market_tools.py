"""The `market_prices` tool: one code path (market-data's `prices`), restricted sources gated per token.

Stored rows of a restricted source are served only to a client of MOF_MARKET_RESTRICTED_ALLOW; a live
fetch with the caller's own key (stdio) is the caller's own data; over HTTP a live fetch uses the
deployment's key and is gated like stored rows. Synthetic store and fake live source: tests/market_fixture.py.
"""
import json
import sys
import time

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from http_server import free_port, serve

pytest.importorskip("market_data", reason="the [market] extra is not installed")

from market_fixture import (
    CIK1,
    END,
    FIGI1,
    I1,
    I2,
    KEY,
    PRIV_LABEL,
    PRIV_TOKEN,
    START,
    fake_tiingo,
    seed_market,
)

from myopenfund_mcp.auth import hash_token
from myopenfund_mcp.companies.store import CompaniesStore
from myopenfund_mcp.config import ConfigError, Settings
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio

RANGE = {"start": START.isoformat(), "end": END.isoformat()}
NOTEBOOK_TOKEN = "t-notebook-" + "n" * 40
AGENTS_TOKEN = "t-agents-" + "a" * 40
RAW_STORE_KEYS = {"instrument_id", "session_date", "close", "volume", "currency", "source_code", "source"}


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch, tmp_path):
    # Never a real key or quota file from the developer's environment.
    monkeypatch.setenv("MOF_MARKET_STATE_DIR", str(tmp_path / "quota"))  # market-data's quota files
    for var in ("TIINGO_API_KEY", "MASSIVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def market_store(tmp_path):
    cfg = seed_market(tmp_path)
    store = CompaniesStore(str(cfg.data_root), cfg.catalog)
    yield store
    store.close()


def _factory(requests: list | None = None):
    def make(code: str):
        assert code == "tiingo", code
        return fake_tiingo(requests)
    return make


def _client(settings=None, *, store=None, requests=None, adapter=True):
    return Client(build_server(settings or Settings(), companies_store=store,
                               market_adapter=_factory(requests) if adapter else None))


async def _call(client, **args):
    return (await client.call_tool("market_prices", {**RANGE, **args})).data


# --- listing ----------------------------------------------------------------------------------------
async def test_tool_is_listed_without_a_key_argument():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    tool = tools["market_prices"]
    assert tool.annotations.read_only_hint is True
    assert set(tool.input_schema["properties"]) == {"instruments", "start", "end", "source", "adjust", "mode"}
    assert info["toolsets"]["market"] == "enabled"


async def test_without_the_market_extra_the_server_starts_without_the_tool(monkeypatch):
    monkeypatch.delitem(sys.modules, "myopenfund_mcp.market.tools", raising=False)
    monkeypatch.setitem(sys.modules, "market_data", None)  # import market_data -> ModuleNotFoundError
    monkeypatch.setitem(sys.modules, "market_data.api", None)
    async with Client(build_server(Settings())) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert "market_prices" not in names
    assert info["toolsets"]["market"] == "not installed"

    # A mode asked for explicitly without the extra is a deployment mistake: refuse to start.
    with pytest.raises(ConfigError, match=r"MOF_MARKET_SOURCE is set to live but the \[market\] extra"):
        build_server(Settings(market_source="live"))


# --- live fetches over stdio: the caller's own key, no gate -----------------------------------------
async def test_live_fetch_uses_caller_key_no_gate():
    requests: list = []
    async with _client(requests=requests) as c:
        out = await _call(c, instruments=["AAA", "BBB"])
        live = await _call(c, instruments=["AAA"], mode="live")
    assert len(requests) == 3  # nothing gated: tiingo is restricted, yet the caller's own fetch is served
    assert all(r.headers["Authorization"] == f"Token {KEY}" for r in requests)
    assert out["returned"] == 10 and out["mode"] == "auto"
    assert {(r["source"], r["reason"], r["source_code"]) for r in out["rows"]} == {("live", "no_store", "tiingo")}
    assert {r["input"] for r in out["rows"]} == {"AAA", "BBB"}
    assert out["inputs"] == [{"input": "AAA", "source": "live", "reason": "no_store"},
                             {"input": "BBB", "source": "live", "reason": "no_store"}]
    assert all("reason" not in r and r["source"] == "live" for r in live["rows"])  # mode live: no fallback


async def test_live_key_comes_from_the_environment_and_is_never_echoed(monkeypatch):
    async with _client(adapter=False) as c:
        with pytest.raises(ToolError, match=r"^\[market_key_missing\].*TIINGO_API_KEY"):
            await _call(c, instruments=["AAA"], mode="live")
        with pytest.raises(ToolError, match=r"^\[market_key_missing\].*MASSIVE_API_KEY"):
            await _call(c, instruments=["AAA"], mode="live", source="massive")


async def test_live_unknown_symbol_and_cik_are_unknown_instruments():
    async with _client() as c:
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\].*ZZZ"):
            await _call(c, instruments=["AAA", "ZZZ"], mode="live")
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\].*CIK"):
            await _call(c, instruments=[CIK1], mode="live")


async def test_live_total_return_rows_carry_their_provider():
    async with _client() as c:
        out = await _call(c, instruments=["AAA"], mode="live", adjust="total_return")
    assert out["returned"] == 5 and out["rows"][0]["tr_index"] == 1.0
    assert {r["source_code"] for r in out["rows"]} == {"tiingo"}


# --- stored rows over stdio: restricted rows are not the caller's own data --------------------------
async def test_stdio_never_reads_stored_restricted_rows(market_store):
    async with _client(store=market_store) as c:
        with pytest.raises(ToolError, match=r"^\[market_source_restricted\]"):
            await _call(c, instruments=["AAA"], mode="store", source="tiingo")
        stored = await _call(c, instruments=["AAA"], mode="store")
        auto = await _call(c, instruments=["AAA", "BBB"])
    assert stored["returned"] == 0
    assert stored["inputs"] == [{"input": "AAA", "source": "store"}]
    # auto: the store holds no row the caller may read for AAA, and nothing names BBB: both live
    assert auto["inputs"] == [{"input": "AAA", "source": "live", "reason": "store_no_data"},
                              {"input": "BBB", "source": "live", "reason": "not_in_store"}]
    assert {r["close"] for r in auto["rows"] if r["input"] == "AAA"} == {300.0, 301.0, 302.0, 303.0, 304.0}


async def test_store_mode_never_goes_live(market_store):
    requests: list = []
    async with _client(store=market_store, requests=requests) as c:
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\].*BBB"):
            await _call(c, instruments=["BBB"], mode="store")
    assert requests == []
    async with _client() as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\].*store"):
            await _call(c, instruments=["AAA"], mode="store")


async def test_store_failure_goes_live_in_auto_and_is_masked_in_store_mode(market_store, caplog):
    def boom(fn):
        raise OSError("host=secret-db password=hunter2")

    market_store.call = boom
    async with _client(store=market_store) as c:
        auto = await _call(c, instruments=["AAA"])
        with pytest.raises(ToolError, match=r"^\[backend_unavailable\]") as err:
            await _call(c, instruments=["AAA"], mode="store")
    assert {r["reason"] for r in auto["rows"]} == {"store_unavailable"}
    assert "hunter2" not in str(err.value) and "hunter2" not in caplog.text


# --- bounds and arguments ---------------------------------------------------------------------------
async def test_bounds_are_refused_never_cut():
    async with _client(Settings(market_max_instruments=1, market_max_rows=4)) as c:
        with pytest.raises(ToolError, match=r"^\[too_many_instruments\].*2 instruments.*at most 1"):
            await _call(c, instruments=["AAA", "BBB"])
        with pytest.raises(ToolError, match=r"^\[result_too_large\].*5 rows.*4"):
            await _call(c, instruments=["AAA"])


@pytest.mark.parametrize("args", [
    {"start": "2025-06-06", "end": "2025-06-02"},
    {"start": "June 2"},
    {"instruments": []},
    {"instruments": [" "]},
    {"source": "Not A Code"},
    {"source": "private_history", "mode": "live"},
])
async def test_bad_arguments_are_refused(args):
    async with _client() as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\]"):
            await _call(c, **{"instruments": ["AAA"], **args})


async def test_live_timeout(monkeypatch):
    class Slow:
        source_code = "tiingo"

        def fetch(self, *a, **k):
            time.sleep(3)
            return iter(())

    async with Client(build_server(Settings(market_live_timeout_s=1), market_adapter=lambda code: Slow())) as c:
        with pytest.raises(ToolError, match=r"^\[timeout\]"):
            await _call(c, instruments=["AAA"], mode="live")


# --- over HTTP: stored rows and live fetches per token ----------------------------------------------
@pytest.fixture
def http_url(market_store):
    port = free_port()
    settings = Settings(transport="http", port=port,
                        tokens=f"notebook:{hash_token(NOTEBOOK_TOKEN)}:read,agents:{hash_token(AGENTS_TOKEN)}:read",
                        market_restricted_allow={"notebook": frozenset({"tiingo", "private_history"})})
    server = build_server(settings, companies_store=market_store, market_adapter=_factory())
    with serve(server, port) as url:
        yield url


async def test_restricted_rows_not_served_to_read_token(http_url):
    async with Client(http_url, auth=AGENTS_TOKEN) as c:
        stored = await _call(c, instruments=["AAA"], mode="store")
        with pytest.raises(ToolError, match=r"^\[market_source_restricted\]"):
            await _call(c, instruments=["AAA"], mode="store", source="tiingo")
        # auto: nothing readable in the store, and a live fetch would serve the deployment's key
        with pytest.raises(ToolError, match=r"^\[market_source_restricted\].*tiingo"):
            await _call(c, instruments=["AAA"])
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\]"):
            await _call(c, instruments=[PRIV_LABEL], mode="store")
    assert stored["returned"] == 0 and stored["rows"] == []
    async with Client(http_url, auth=NOTEBOOK_TOKEN) as c:
        allowed = await _call(c, instruments=["AAA"], mode="store")
        live = await _call(c, instruments=["BBB"])
    assert allowed["returned"] == 5
    assert {(r["instrument_id"], r["source"], r["source_code"]) for r in allowed["rows"]} == {(I1, "store", "tiingo")}
    assert {(r["source"], r["reason"]) for r in live["rows"]} == {("live", "not_in_store")}


async def test_as_of_output_has_no_labels(http_url):
    async with Client(http_url, auth=NOTEBOOK_TOKEN) as c:
        raw = await _call(c, instruments=[PRIV_LABEL, FIGI1, CIK1], mode="store")
        tr = await _call(c, instruments=[PRIV_LABEL, "AAA"], mode="store", adjust="total_return")
    assert {r["instrument_id"] for r in raw["rows"]} == {I1, I2}
    assert all(set(r) == RAW_STORE_KEYS for r in raw["rows"])
    assert {r["source_code"] for r in raw["rows"] if r["instrument_id"] == I2} == {"private_history"}
    assert all(set(r) == {"instrument_id", "session_date", "tr_index", "flags", "source_code", "source"}
               for r in tr["rows"])
    assert {r["source_code"] for r in tr["rows"] if r["instrument_id"] == I1} == {"tiingo"}
    for out in (raw, tr):
        rows = json.dumps(out["rows"])
        assert PRIV_TOKEN not in json.dumps(out)
        for label in (PRIV_LABEL, FIGI1, CIK1, "AAA", "tiingo:", "private_history:"):
            assert label not in rows


class _Slow:
    """A live adapter whose fetch takes `seconds`, then answers as the fake Tiingo does."""

    source_code = "tiingo"

    def __init__(self, seconds: float, done: list):
        self._seconds, self._done = seconds, done
        self._inner = fake_tiingo()

    def __getattr__(self, name):  # everything else as the fake Tiingo
        return getattr(self._inner, name)

    def fetch(self, *args, **kwargs):
        time.sleep(self._seconds)
        out = list(self._inner.fetch(*args, **kwargs))
        self._done.append(True)
        return iter(out)

    def close(self):
        self._inner.close()


async def test_live_fetches_are_bounded_and_busy_is_answered_not_queued():
    import anyio

    done: list = []
    server = build_server(Settings(market_live_concurrency=1),
                          market_adapter=lambda code: _Slow(1.5, done))
    async with Client(server) as c:
        results: list = []

        async def first():
            results.append(await _call(c, instruments=["AAA"], mode="live"))

        async with anyio.create_task_group() as tg:
            tg.start_soon(first)
            await anyio.sleep(0.3)
            t0 = time.monotonic()
            with pytest.raises(ToolError, match=r"^\[market_busy\].*MOF_MARKET_LIVE_CONCURRENCY"):
                await _call(c, instruments=["BBB"], mode="live")
            assert time.monotonic() - t0 < 1.0  # answered at once, not queued
        assert results[0]["returned"] == 5
        assert (await _call(c, instruments=["BBB"], mode="live"))["returned"] == 5  # the slot is free again


async def test_an_abandoned_live_fetch_keeps_its_slot_until_it_ends():
    done: list = []
    made: list = []

    def factory(code):  # the first fetch outlasts the budget, later ones are quick
        made.append(code)
        return _Slow(2.5 if len(made) == 1 else 0, done)

    server = build_server(Settings(market_live_concurrency=1, market_live_timeout_s=1), market_adapter=factory)
    async with Client(server) as c:
        with pytest.raises(ToolError, match=r"^\[timeout\]"):
            await _call(c, instruments=["AAA"], mode="live")
        # the abandoned thread still fetches with the key: no second fetch beside it
        with pytest.raises(ToolError, match=r"^\[market_busy\]"):
            await _call(c, instruments=["AAA"], mode="live")
        import anyio

        deadline = time.monotonic() + 5
        while not done and time.monotonic() < deadline:
            await anyio.sleep(0.05)
        await anyio.sleep(0.1)  # its slot is released just after the fetch returns
        assert done == [True]  # it finished its fetch: the quota is spent
        out = await _call(c, instruments=["AAA"], mode="live")
    assert out["returned"] == 5 and len(made) == 2  # the busy call built no adapter
