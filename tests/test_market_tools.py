"""The `market_prices` tool: one code path (market-data's `prices`), restricted sources gated per token.

Stored rows of a restricted source are served only to a client of MOF_MARKET_RESTRICTED_ALLOW; a live
fetch with the caller's own key (stdio) is the caller's own data; over HTTP a live fetch uses the
deployment's key and is gated like stored rows. Synthetic store and fake live source: tests/market_fixture.py.
"""
import json
import sys
import threading
import time

import anyio
import anyio.lowlevel
import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from http_server import free_port, serve

pytest.importorskip("market_data", reason="the [market] extra is not installed")

from market_fixture import (
    CIK1,
    DAYS,
    END,
    FIGI1,
    I1,
    I2,
    KEY,
    MASSIVE_CLOSES,
    NOW,
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
    def make(code: str, budget=None):
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

    async with Client(build_server(Settings(market_live_timeout_s=1), market_adapter=lambda code, budget: Slow())) as c:
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
    """A live adapter whose fetch waits for `release` (set at once: no wait), then answers as the fake Tiingo
    does. `started` is set when a fetch begins, `closed` when the adapter is closed (the fetch has ended)."""

    source_code = "tiingo"

    def __init__(self, release: threading.Event | None = None):
        self.started, self.closed = threading.Event(), threading.Event()
        self._release = release
        self._inner = fake_tiingo()

    def __getattr__(self, name):  # everything else as the fake Tiingo
        return getattr(self._inner, name)

    def fetch(self, *args, **kwargs):
        self.started.set()
        if self._release is not None:
            assert self._release.wait(10), "the test never released the fetch"
        return iter(list(self._inner.fetch(*args, **kwargs)))

    def close(self):
        self._inner.close()
        self.closed.set()


async def _wait(event: threading.Event):
    assert await anyio.to_thread.run_sync(event.wait, 10), "the event never came"


async def _call_when_free(client, **args):
    """A call retried while the slot is still held by a fetch that has just ended (released after close)."""
    for _ in range(200):
        try:
            return await _call(client, **args)
        except ToolError as exc:
            if not str(exc).startswith("[market_busy]"):
                raise
        await anyio.lowlevel.checkpoint()
    raise AssertionError("the slot was never released")


async def test_live_fetches_are_bounded_and_busy_is_answered_not_queued():
    release = threading.Event()
    made: list[_Slow] = []

    def factory(code, budget):
        made.append(_Slow(release if not made else None))
        return made[-1]

    server = build_server(Settings(market_live_concurrency=1), market_adapter=factory)
    async with Client(server) as c:
        results: list = []

        async def first():
            results.append(await _call(c, instruments=["AAA"], mode="live"))

        async with anyio.create_task_group() as tg:
            tg.start_soon(first)
            await _wait_for_adapter(made)
            await _wait(made[0].started)
            t0 = time.monotonic()
            with pytest.raises(ToolError, match=r"^\[market_busy\].*MOF_MARKET_LIVE_CONCURRENCY"):
                await _call(c, instruments=["BBB"], mode="live")
            assert time.monotonic() - t0 < 1.0  # answered at once, not queued
            release.set()
        assert results[0]["returned"] == 5
        assert (await _call_when_free(c, instruments=["BBB"], mode="live"))["returned"] == 5  # the slot is free


async def _wait_for_adapter(made: list):
    with anyio.fail_after(10):
        while not made:
            await anyio.lowlevel.checkpoint()


async def test_an_abandoned_live_fetch_keeps_its_slot_until_it_ends():
    release = threading.Event()
    made: list[_Slow] = []

    def factory(code, budget):  # the first fetch outlasts the budget (until released), later ones are quick
        made.append(_Slow(release if not made else None))
        return made[-1]

    server = build_server(Settings(market_live_concurrency=1, market_live_timeout_s=1), market_adapter=factory)
    async with Client(server) as c:
        with pytest.raises(ToolError, match=r"^\[timeout\]"):
            await _call(c, instruments=["AAA"], mode="live")
        # the abandoned thread still fetches with the key: no second fetch beside it
        with pytest.raises(ToolError, match=r"^\[market_busy\]"):
            await _call(c, instruments=["AAA"], mode="live")
        assert len(made) == 1  # the busy call built no adapter
        release.set()
        await _wait(made[0].closed)  # it finished its fetch: the quota is spent
        out = await _call_when_free(c, instruments=["AAA"], mode="live")
    assert out["returned"] == 5 and len(made) == 2


# --- several sources on one instrument: MOF_MARKET_PRECEDENCE ---------------------------------------
def _overlap_server(tmp_path, port, **settings):
    cfg = seed_market(tmp_path, overlap=True, conflict=settings.pop("conflict", False))
    store = CompaniesStore(str(cfg.data_root), cfg.catalog)
    tokens = f"notebook:{hash_token(NOTEBOOK_TOKEN)}:read,agents:{hash_token(AGENTS_TOKEN)}:read"
    allow = {"notebook": frozenset({"tiingo", "massive"})}
    s = Settings(transport="http", port=port, tokens=tokens, market_restricted_allow=allow, **settings)
    return store, build_server(s, companies_store=store, market_adapter=_factory())


async def test_two_allowed_sources_on_one_instrument_follow_the_precedence(tmp_path):
    port = free_port()
    store, server = _overlap_server(tmp_path, port)
    try:
        with serve(server, port) as url:
            async with Client(url, auth=NOTEBOOK_TOKEN) as c:
                out = await _call(c, instruments=["AAA"], mode="store")
                tr = await _call(c, instruments=["AAA"], mode="store", adjust="total_return")
                one = await _call(c, instruments=["AAA"], mode="store", source="tiingo")
    finally:
        store.close()
    # default private_history,massive,tiingo: massive wins where it has a price, tiingo elsewhere
    by_day = {r["session_date"]: (r["source_code"], r["close"]) for r in out["rows"]}
    assert len(out["rows"]) == 5 and {r["instrument_id"] for r in out["rows"]} == {I1}
    assert [by_day[d.isoformat()] for d in DAYS] == [
        ("tiingo", 100.0), ("tiingo", 101.0), *[("massive", c) for c in MASSIVE_CLOSES]]
    assert [r["source_code"] for r in tr["rows"]] == ["tiingo", "tiingo", "massive", "massive", "massive"]
    assert {r["source_code"] for r in one["rows"]} == {"tiingo"} and len(one["rows"]) == 5


async def test_precedence_is_configurable_and_never_names_an_unreadable_source(tmp_path, monkeypatch):
    from myopenfund_mcp.market import tools

    seen: list = []
    real = tools.api.prices

    def spy(*args, **kwargs):
        seen.append(kwargs.get("precedence"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tools.api, "prices", spy)
    port = free_port()
    store, server = _overlap_server(tmp_path, port, market_precedence=("tiingo", "massive"))
    try:
        with serve(server, port) as url:
            async with Client(url, auth=NOTEBOOK_TOKEN) as c:
                out = await _call(c, instruments=["AAA"], mode="store")
            async with Client(url, auth=AGENTS_TOKEN) as c:
                none = await _call(c, instruments=["AAA"], mode="store")
    finally:
        store.close()
    assert {r["source_code"] for r in out["rows"]} == {"tiingo"} and len(out["rows"]) == 5
    assert none["rows"] == []
    notebook, agents = seen[0], seen[-1]
    assert list(notebook[:2]) == ["tiingo", "massive"]  # the setting's order first
    assert "private_history" not in notebook and "manual" not in notebook  # not allowed: never passed
    assert not {"tiingo", "massive", "private_history", "manual"} & set(agents)


async def test_conflicting_series_get_the_servers_own_message(tmp_path):
    port = free_port()
    store, server = _overlap_server(tmp_path, port, conflict=True)
    try:
        with serve(server, port) as url:
            async with Client(url, auth=NOTEBOOK_TOKEN) as c:
                with pytest.raises(ToolError, match=r"^\[invalid_argument\].*AAA.*pass source") as err:
                    await _call(c, instruments=["AAA"], mode="store")
    finally:
        store.close()
    assert "precedence" not in str(err.value) and "AAZ" not in str(err.value)


# --- one store read per call ------------------------------------------------------------------------
async def test_store_inputs_are_read_in_one_call(http_url, monkeypatch):
    from myopenfund_mcp.market import tools

    calls: list = []
    real = tools.api.prices

    def spy(instruments, *args, **kwargs):
        calls.append((list(instruments), kwargs.get("adjust")))
        return real(instruments, *args, **kwargs)

    monkeypatch.setattr(tools.api, "prices", spy)
    async with Client(http_url, auth=NOTEBOOK_TOKEN) as c:
        raw = await _call(c, instruments=[PRIV_LABEL, "AAA"], mode="store")
        n_raw = len(calls)
        tr = await _call(c, instruments=[PRIV_LABEL, "AAA"], mode="auto", adjust="total_return")
    assert n_raw == 1 and calls[0][0] == [PRIV_LABEL, "AAA"]
    assert [a for _, a in calls[1:]] == ["total_return", "raw"]  # the raw read gives each session's provider
    assert {r["instrument_id"] for r in raw["rows"]} == {I1, I2} and raw["returned"] == 10
    assert {(r["instrument_id"], r["source_code"]) for r in tr["rows"]} == {(I1, "tiingo"), (I2, "private_history")}
    assert tr["inputs"] == [{"input": PRIV_LABEL, "source": "store"}, {"input": "AAA", "source": "store"}]


async def test_one_store_call_still_routes_each_input(http_url):
    async with Client(http_url, auth=NOTEBOOK_TOKEN) as c:
        out = await _call(c, instruments=["AAA", "BBB", FIGI1])
    # AAA and its FIGI name one instrument (its rows once); BBB is not in the store: live
    assert out["inputs"] == [{"input": "AAA", "source": "store"},
                             {"input": "BBB", "source": "live", "reason": "not_in_store"},
                             {"input": FIGI1, "source": "store"}]
    assert sum(r["source"] == "store" for r in out["rows"]) == 5
    assert {r["input"] for r in out["rows"] if r["source"] == "live"} == {"BBB"}


# --- store-only names never reach a live source -----------------------------------------------------
@pytest.mark.parametrize("label", [I2, f"private_history:{PRIV_TOKEN}", "tiingo:AAA"])
async def test_live_refuses_store_only_names_before_any_request(label):
    requests: list = []
    async with _client(requests=requests) as c:
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\]"):
            await _call(c, instruments=["AAA", label], mode="live")
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\]"):
            await _call(c, instruments=[label])  # auto without a store
    assert requests == []


@pytest.mark.parametrize("mode", ["auto", "live"])
async def test_a_private_label_is_never_sent_to_a_live_source(market_store, mode):
    requests: list = []
    async with _client(store=market_store, requests=requests) as c:
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\]") as err:
            await _call(c, instruments=["AAA", PRIV_LABEL], mode=mode)
        with pytest.raises(ToolError, match=r"^\[market_unknown_instrument\]"):
            await _call(c, instruments=[PRIV_TOKEN], mode=mode)  # the private token typed as is
        ok = await _call(c, instruments=["AAA"], mode=mode)  # an open label still goes live
    assert requests and all(PRIV_LABEL not in str(r.url) and PRIV_TOKEN not in str(r.url) for r in requests)
    assert len(requests) == 1 and ok["returned"] == 5
    assert PRIV_TOKEN not in str(err.value)


# --- live waits never outlast the call's budget -----------------------------------------------------
def _budgeted_tiingo(tmp_path, requests: list, **kw):
    def make(code, budget):
        return fake_tiingo(requests, state_dir=tmp_path / "quota", sleep=budget.sleep,
                           max_wait_s=budget.max_wait_s, **kw)
    return make


async def test_tiingo_hourly_budget_wait_is_limit_exceeded_not_a_sleep(tmp_path):
    requests: list = []
    server = build_server(Settings(market_live_timeout_s=30),
                          market_adapter=_budgeted_tiingo(tmp_path, requests, per_hour=1))
    async with Client(server) as c:
        assert (await _call(c, instruments=["AAA"], mode="live"))["returned"] == 5
        t0 = time.monotonic()
        with pytest.raises(ToolError, match=r"^\[limit_exceeded\].*tiingo.*wait about (3[45][0-9]{2}|3600) s") as err:
            await _call(c, instruments=["BBB"], mode="live")
        assert time.monotonic() - t0 < 5  # refused at once, never slept
    assert len(requests) == 1 and "MOF_MARKET_LIVE_TIMEOUT_S" in str(err.value)


async def test_a_retry_wait_beyond_the_budget_is_limit_exceeded(tmp_path):
    def answer(request):
        return httpx.Response(429, headers={"Retry-After": "600"})

    def make(code, budget):
        from market_data.adapters.tiingo import TiingoAdapter

        return TiingoAdapter(KEY, transport=httpx.MockTransport(answer), state_dir=None, min_spacing_s=0,
                             sleep=budget.sleep, max_wait_s=budget.max_wait_s, now=lambda: NOW)

    async with Client(build_server(Settings(market_live_timeout_s=30), market_adapter=make)) as c:
        with pytest.raises(ToolError, match=r"^\[limit_exceeded\].*wait about 600 s"):
            await _call(c, instruments=["AAA"], mode="live")


def _fake_massive(status: int, requests: list):
    from market_data.adapters.massive import MassiveAdapter

    def answer(request):
        requests.append(request)
        return httpx.Response(status, json={"status": "NOT_AUTHORIZED", "request_id": "r"})

    def make(code, budget):
        assert code == "massive"
        return MassiveAdapter(KEY, transport=httpx.MockTransport(answer), sleep=budget.sleep,
                              max_wait_s=budget.max_wait_s, now=lambda: NOW)
    return make


async def test_massive_plan_window_and_403_get_their_own_codes():
    requests: list = []
    async with Client(build_server(Settings(), market_adapter=_fake_massive(403, requests))) as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\].*massive.*730 days") as old:
            await _call(c, instruments=["AAA"], mode="live", source="massive", start="2001-01-02",
                        end="2001-01-31")
        assert requests == []  # refused before any request
        with pytest.raises(ToolError, match=r"^\[limit_exceeded\].*massive.*HTTP 403.*plan"):
            await _call(c, instruments=["AAA"], mode="live", source="massive")
    assert "tiingo" in str(old.value)  # the deep free history


async def test_massive_401_is_a_refused_key():
    async with Client(build_server(Settings(), market_adapter=_fake_massive(401, []))) as c:
        with pytest.raises(ToolError, match=r"^\[market_key_missing\].*massive.*MASSIVE_API_KEY"):
            await _call(c, instruments=["AAA"], mode="live", source="massive")


# --- start-up checks --------------------------------------------------------------------------------
def test_tokens_without_a_guarded_client_log_one_warning(caplog):
    import logging

    tokens = f"agent-1:{hash_token(AGENTS_TOKEN)}:read,notebook:{hash_token(NOTEBOOK_TOKEN)}:read"
    with caplog.at_level(logging.WARNING, logger="myopenfund_mcp"):
        build_server(Settings(transport="http", tokens=tokens))
    warned = [r.getMessage() for r in caplog.records if "MOF_MARKET_NO_RESTRICTED_CLIENTS" in r.getMessage()]
    assert len(warned) == 1 and "2 client(s)" in warned[0]
    assert "agent-1" not in caplog.text and "notebook" not in caplog.text  # no client id in the log
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="myopenfund_mcp"):
        build_server(Settings(transport="http", tokens=tokens, market_no_restricted_clients=("Agent-1",)))
        build_server(Settings())  # stdio: no token, nothing to guard
    assert "MOF_MARKET_NO_RESTRICTED_CLIENTS" not in caplog.text


def test_an_open_data_store_without_the_source_filter_refuses_to_start(monkeypatch, market_store):
    from open_data_store import sources

    monkeypatch.delattr(sources, "filter_sources")  # what open-data-store 0.1 lacks
    with pytest.raises(ConfigError, match=r"myopenfund-open-data-store>=0\.2") as err:
        build_server(Settings(), companies_store=market_store)
    assert err.value.__cause__ is None
    with pytest.raises(ConfigError, match=r"myopenfund-open-data-store>=0\.2"):
        build_server(Settings())  # the market toolset alone needs it too


def test_a_market_import_error_is_not_a_missing_extra(monkeypatch, caplog):
    import logging

    monkeypatch.delitem(sys.modules, "myopenfund_mcp.market.tools", raising=False)
    import myopenfund_mcp.market as pkg

    monkeypatch.delattr(pkg, "tools", raising=False)
    real = __import__("builtins").__import__

    def broken(name, *args, **kwargs):
        if name == "market_data":
            raise ImportError("libsomething.so: cannot open shared object file")
        return real(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", broken)
    with caplog.at_level(logging.ERROR, logger="myopenfund_mcp"), \
            pytest.raises(ConfigError, match=r"market toolset could not be imported"):
        build_server(Settings())
    assert "ImportError" in caplog.text
