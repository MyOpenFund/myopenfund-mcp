import datetime as dt
import sys
import time

import pytest
from companies_fixture import AAPL, seed
from fastmcp import Client
from fastmcp.exceptions import ToolError

from myopenfund_mcp.companies.common import parse_as_of, rows_to_utc, store_call
from myopenfund_mcp.companies.datasets import rows_to_csv
from myopenfund_mcp.companies.store import (
    CompaniesStore,
    StoreBackendError,
    StoreBadArgument,
    StoreUnknownDataset,
)
from myopenfund_mcp.config import ConfigError, Settings
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def store(tmp_path):
    cfg = seed(tmp_path)
    s = CompaniesStore(str(cfg.data_root), cfg.catalog)
    yield s
    s.close()


def _client(store, **kw):
    return Client(build_server(Settings(**kw), companies_store=store))


async def test_tools_listed_read_only(store):
    async with _client(store) as c:
        tools = {t.name: t for t in await c.list_tools()}
    for n in ("datasets_list", "datasets_describe", "datasets_read_asof", "datasets_export_csv"):
        assert tools[n].annotations.read_only_hint is True
    for n in ("companies_resolve", "companies_facts", "companies_fundamentals"):
        assert tools[n].annotations.read_only_hint is True


async def test_list_and_describe(store):
    async with _client(store) as c:
        names = {d["name"] for d in (await c.call_tool("datasets_list", {})).data["datasets"]}
        desc = (await c.call_tool("datasets_describe", {"dataset": "fundamentals"})).data
    assert {"fundamentals", "entity_observations", "filing_statements"} <= names
    assert "entity_value" in desc["natural_key"]


async def test_read_asof_filters_pages_and_echoes_as_of(store):
    # Apple, latest per key now: 5 facts (FY2024 revenue restated counts once)
    async with _client(store) as c:
        first = (await c.call_tool("datasets_read_asof", {
            "dataset": "fundamentals", "filters": {"entity_value": AAPL}, "limit": 3})).data
        second = (await c.call_tool("datasets_read_asof", {
            "dataset": "fundamentals", "filters": {"entity_value": AAPL}, "limit": 3,
            "cursor": first["next_cursor"]})).data
    assert first["returned"] == 3 and first["as_of"].endswith("+00:00") and first["next_cursor"]
    assert second["returned"] == 2 and second["next_cursor"] is None
    keys = [(r["original_tag"], r["valid_date"]) for r in first["rows"] + second["rows"]]
    assert len(set(keys)) == 5
    assert all(r["known_at"].endswith("+00:00") for r in first["rows"] + second["rows"])


async def test_read_asof_datetimes_are_utc_and_as_of_is_respected(store):
    async with _client(store) as c:
        old = (await c.call_tool("datasets_read_asof", {
            "dataset": "fundamentals", "filters": {"entity_value": AAPL},
            "as_of": "2025-01-01T01:00:00+01:00", "columns": ["original_tag", "valid_date", "value", "known_at"],
        })).data
    assert old["as_of"] == "2025-01-01T00:00:00+00:00"
    assert old["rows"] and all(r["known_at"] == "2024-11-01T20:00:00+00:00" for r in old["rows"])
    assert set(old["rows"][0]) == {"original_tag", "valid_date", "value", "known_at"}


async def test_read_asof_vintages(store):
    async with _client(store) as c:
        args = {"dataset": "fundamentals", "filters": {"entity_value": AAPL, "valid_date": "2024-09-28",
                                                       "original_tag": [
                                                           "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"]}}
        latest = (await c.call_tool("datasets_read_asof", args)).data
        every = (await c.call_tool("datasets_read_asof", {**args, "vintages": True})).data
    assert latest["returned"] == 1 and every["returned"] == 2


async def test_unknown_dataset_and_column_are_coded(store):
    async with _client(store) as c:
        with pytest.raises(ToolError, match=r"^\[unknown_dataset\].*datasets_list"):
            await c.call_tool("datasets_describe", {"dataset": "nope"})
        with pytest.raises(ToolError, match=r"^\[unknown_column\].*datasets_describe"):
            await c.call_tool("datasets_read_asof", {"dataset": "fundamentals", "filters": {"x": "1"}})
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] filter 'valid_date'"):
            await c.call_tool("datasets_read_asof", {"dataset": "fundamentals",
                                                     "filters": {"valid_date": "not a date"}})
        with pytest.raises(ToolError, match=r"^\[invalid_argument\]"):
            await c.call_tool("datasets_read_asof", {"dataset": "fundamentals", "limit": 0})
        with pytest.raises(ToolError, match=r"^\[invalid_argument\]"):
            await c.call_tool("datasets_read_asof", {"dataset": "fundamentals", "cursor": "junk"})


async def test_backend_failure_is_masked(store, caplog):
    def boom(**_):
        raise StoreBackendError("store backend error (OperationalError; message withheld)")

    def leak(**_):
        raise OSError("host=secret-db password=hunter2")

    async with _client(store) as c:
        store.datasets = boom
        with pytest.raises(ToolError, match=r"^\[backend_unavailable\]") as e1:
            await c.call_tool("datasets_list", {})
        store.datasets = leak
        with pytest.raises(ToolError, match=r"^\[backend_unavailable\]") as e2:
            await c.call_tool("datasets_list", {})
    for e in (e1, e2):
        assert "OperationalError" not in str(e.value) and "hunter2" not in str(e.value)
    assert "hunter2" not in caplog.text and "OSError" in caplog.text


async def test_server_info_reports_store_freshness(store):
    async with _client(store) as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["toolsets"]["datasets"] == "enabled" and info["toolsets"]["companies"] == "enabled"
    assert info["companies_freshness"]["batches"] == 3
    assert info["companies_freshness"]["last_published_at"].endswith("+00:00")


async def test_server_info_survives_a_store_outage(store):
    store.freshness = lambda: (_ for _ in ()).throw(OSError("down"))
    async with _client(store) as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["companies_freshness_error"] == "backend_unavailable"
    assert "companies_freshness" not in info and info["toolsets"]["datasets"] == "enabled"


async def test_without_a_store_datasets_is_not_configured():
    async with Client(build_server(Settings())) as c:
        info = (await c.call_tool("server_info", {})).data
        names = {t.name for t in await c.list_tools()}
    assert info["toolsets"]["datasets"] == info["toolsets"]["companies"] == "not configured"
    assert "companies_freshness" not in info and not any(n.startswith("datasets_") for n in names)


def test_configured_without_the_extra_refuses_to_start(monkeypatch):
    monkeypatch.setitem(sys.modules, "myopendata.config", None)  # import -> ImportError
    with pytest.raises(ConfigError, match=r"^MOF_COMPANIES_CATALOG is set but the \[companies\] extra"):
        build_server(Settings(companies_catalog="cat", companies_data_root="/x"))


async def test_configured_store_is_built_from_settings(tmp_path):
    cfg = seed(tmp_path)
    server = build_server(Settings(companies_catalog=cfg.catalog, companies_data_root=str(cfg.data_root)))
    async with Client(server) as c:
        names = {d["name"] for d in (await c.call_tool("datasets_list", {})).data["datasets"]}
    assert "fundamentals" in names


# --- common.py helpers -------------------------------------------------------------------------

def test_parse_as_of():
    assert parse_as_of(dt.datetime(2025, 1, 1)) == dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    paris = dt.timezone(dt.timedelta(hours=1))
    assert parse_as_of(dt.datetime(2025, 1, 1, 1, tzinfo=paris)).tzinfo == dt.UTC
    now = parse_as_of(None)
    assert now.tzinfo == dt.UTC and abs((dt.datetime.now(dt.UTC) - now).total_seconds()) < 5


def test_rows_to_utc_converts_aware_datetimes_only():
    paris = dt.timezone(dt.timedelta(hours=1))
    rows = [{"t": dt.datetime(2025, 1, 1, 1, tzinfo=paris), "d": dt.date(2025, 1, 1),
             "n": dt.datetime(2025, 1, 1), "s": "x", "v": None}]
    out = rows_to_utc(rows)
    assert out[0]["t"] == dt.datetime(2025, 1, 1, tzinfo=dt.UTC) and out[0]["t"].tzinfo == dt.UTC
    assert out[0]["d"] == dt.date(2025, 1, 1) and out[0]["n"] == dt.datetime(2025, 1, 1)
    assert out[0]["s"] == "x" and out[0]["v"] is None
    assert rows[0]["t"].tzinfo == paris  # input untouched


def test_rows_to_utc_is_recursive():
    paris = dt.timezone(dt.timedelta(hours=1))
    t = dt.datetime(2025, 1, 1, 1, tzinfo=paris)
    rows = [{"nested": {"t": t, "l": [t, {"t": t}]}, "tuple": (t,)}]
    out = rows_to_utc(rows)[0]
    for v in (out["nested"]["t"], out["nested"]["l"][0], out["nested"]["l"][1]["t"], out["tuple"][0]):
        assert v.tzinfo == dt.UTC and v == t
    assert rows[0]["nested"]["t"].tzinfo == paris and rows[0]["nested"]["l"][1]["t"].tzinfo == paris


async def test_store_call_maps_errors():
    s = Settings(companies_query_timeout_s=1)

    def raise_(exc):
        def f(**_):
            raise exc
        return f

    with pytest.raises(ToolError, match=r"^\[unknown_dataset\] unknown dataset 'x'; see datasets_list$"):
        await store_call(raise_(StoreUnknownDataset("unknown dataset 'x'")), settings=s)
    with pytest.raises(ToolError, match=r"^\[unknown_column\] unknown column 'c'; see datasets_describe$"):
        await store_call(raise_(StoreBadArgument("unknown column 'c'")), settings=s)
    with pytest.raises(ToolError, match=r"^\[invalid_argument\] limit must be positive$"):
        await store_call(raise_(StoreBadArgument("limit must be positive")), settings=s)
    with pytest.raises(ToolError, match=r"^\[backend_unavailable\]"):
        await store_call(raise_(RuntimeError("dsn=secret")), settings=s)
    assert await store_call(lambda *, a: a + 1, settings=s, a=1) == 2


async def test_store_call_times_out():
    s = Settings(companies_query_timeout_s=1)
    t0 = time.monotonic()
    with pytest.raises(ToolError, match=r"^\[timeout\].*narrow the request"):
        await store_call(lambda: time.sleep(3), settings=s)
    assert time.monotonic() - t0 < 2.5


async def test_export_csv_is_inline_text(store):
    async with _client(store) as c:
        res = (await c.call_tool("datasets_export_csv", {
            "dataset": "entity_observations", "filters": {"subject_value": AAPL}})).data
    lines = res["csv"].splitlines()
    assert lines[0].startswith('"entity_id"') or lines[0].startswith("entity_id") or "subject_value" in lines[0]
    assert len(lines) == 1 + res["returned"] and res["returned"] > 0


async def test_export_csv_too_large_is_refused_not_cut(store):
    async with _client(store, max_result_chars=1000) as c:
        with pytest.raises(ToolError, match=r"^\[result_too_large\]"):
            await c.call_tool("datasets_export_csv", {"dataset": "filing_statements"})


async def test_export_csv_timestamps_are_utc_and_decimals_exact(store):
    import csv
    import io
    from decimal import Decimal

    cols = ["original_tag", "valid_date", "value", "known_at"]
    async with _client(store) as c:
        args = {"dataset": "fundamentals", "filters": {"entity_value": AAPL}, "columns": cols}
        res = (await c.call_tool("datasets_export_csv", args)).data
        rows = (await c.call_tool("datasets_read_asof", args)).data["rows"]
    parsed = list(csv.DictReader(io.StringIO(res["csv"])))
    assert len(parsed) == len(rows) == res["returned"]
    assert res["as_of"].endswith("+00:00")
    for got, want in zip(parsed, rows, strict=True):
        assert Decimal(got["value"]) == Decimal(str(want["value"]))  # exact, no float rounding
        assert got["known_at"].startswith(want["known_at"][:10])
        assert want["known_at"].endswith("+00:00")


async def test_export_csv_pages_like_read_asof(store):
    async with _client(store) as c:
        args = {"dataset": "fundamentals", "filters": {"entity_value": AAPL}, "limit": 3}
        first = (await c.call_tool("datasets_export_csv", args)).data
        second = (await c.call_tool("datasets_export_csv", {**args, "cursor": first["next_cursor"]})).data
    assert first["returned"] == 3 and second["returned"] == 2 and second["next_cursor"] is None


async def test_export_csv_empty_result(store):
    async with _client(store) as c:
        base = {"dataset": "fundamentals", "filters": {"entity_value": "0000000000"}}
        # columns known: header only; columns not given: nothing to name, so an empty string
        with_cols = (await c.call_tool("datasets_export_csv", {**base, "columns": ["original_tag", "value"]})).data
        without = (await c.call_tool("datasets_export_csv", base)).data
    assert with_cols["returned"] == 0 and with_cols["csv"].strip() == '"original_tag","value"'
    assert without["returned"] == 0 and without["csv"] == ""


def test_rows_to_csv_writes_decimals_exactly():
    from decimal import Decimal

    # Arrow keeps Decimal as decimal128: digits are written as they are, no float round trip
    # (all values of a column share the widest scale: 1.5 is written 1.500000, the same number).
    text = rows_to_csv([{"v": Decimal("123456789012345678.123456"), "n": "a,b"},
                        {"v": Decimal("0.1"), "n": None}], None)
    lines = text.splitlines()
    assert lines[0] == '"v","n"'
    assert lines[1] == '123456789012345678.123456,"a,b"'
    assert Decimal(lines[2].split(",")[0]) == Decimal("0.1")
    assert "e" not in lines[2].lower().split(",")[0]
