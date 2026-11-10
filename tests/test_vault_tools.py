import datetime as dt
import json
import logging
import sys

import pytest
from fakes import DOCS, FakeVaultReader, invalid_query_reader
from fastmcp import Client
from fastmcp.exceptions import ToolError

from myopenfund_mcp.config import ConfigError, Settings
from myopenfund_mcp.errors import BACKEND_UNAVAILABLE, tool_error
from myopenfund_mcp.paging import MAX_OFFSET, encode_cursor
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio

VAULT_TOOLS = {"vault_search_documents", "vault_get_document", "vault_corpus_stats", "vault_runs_status"}


def _client(reader=None, **settings):
    return Client(build_server(Settings(**settings), vault_reader=reader or FakeVaultReader()))


async def test_tools_are_namespaced_and_read_only():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert set(tools) == VAULT_TOOLS | {"server_info"}
    for name in VAULT_TOOLS | {"server_info"}:
        assert tools[name].annotations.read_only_hint is True, name


async def test_descriptions_carry_the_caveats():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    runs = tools["vault_runs_status"].description
    assert "seen_at_is_ingestion_time" in runs and "last_seen_at" in runs and "approximat" in runs
    assert "`runs`" in runs and "source_runs" in runs
    search = tools["vault_search_documents"].description
    assert "doc_id" in search and "reverse" not in search.lower()


async def test_search_pages_with_a_cursor_and_coerces_string_arguments():
    reader = FakeVaultReader()
    async with _client(reader) as c:
        first = (await c.call_tool("vault_search_documents", {"source_code": "us", "limit": "2"})).data
        assert (first["total"], first["returned"]) == (3, 2)
        assert reader.calls[-1][4] == 2  # "2" reached the reader as an int
        second = (await c.call_tool("vault_search_documents",
                                    {"source_code": "us", "limit": 2, "cursor": first["next_cursor"]})).data
    assert reader.calls[-1][5] == 2  # the cursor carried offset 2
    assert [d["doc_id"] for d in second["documents"]] == ["us3"]
    assert second["next_cursor"] is None
    assert first["documents"][0]["date"] == "2010-01-13"
    assert second["documents"][0]["date"] is None


async def test_search_includes_tombstones_only_on_request():
    async with _client() as c:
        live = (await c.call_tool("vault_search_documents", {"source_code": "us"})).data
        every = (await c.call_tool("vault_search_documents", {"source_code": "us", "include_deleted": True})).data
    assert live["total"] == 3 and every["total"] == 4
    assert next(d for d in every["documents"] if d["doc_id"] == "us9")["deleted_at"].startswith("2026-09-01")


async def test_search_maps_text_to_the_title_filter_and_rejects_bad_arguments():
    reader = FakeVaultReader()
    async with _client(reader) as c:
        await c.call_tool("vault_search_documents", {"text": "FOMC", "date_from": "2010-01-01"})
        filters = reader.calls[-1][1]
        assert filters["q"] == "FOMC" and str(filters["date_from"]) == "2010-01-01"
        with pytest.raises(ToolError):
            await c.call_tool("vault_search_documents", {"date_from": "13/01/2010"})
        with pytest.raises(ToolError, match=r"\[invalid_argument\].*limit"):
            await c.call_tool("vault_search_documents", {"limit": 10_000})
        with pytest.raises(ToolError, match=r"\[invalid_argument\].*cursor"):
            await c.call_tool("vault_search_documents", {"cursor": "not-a-cursor"})
        with pytest.raises(ToolError):
            await c.call_tool("vault_search_documents", {"sort_by": "sha256"})


async def test_reader_invalid_query_surfaces_as_invalid_argument():
    async with _client(invalid_query_reader()) as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] unknown sort field"):
            await c.call_tool("vault_search_documents", {})


async def test_a_result_over_the_cap_is_refused_not_cut():
    big = {**DOCS[0], "doc_id": "big", "title": "x" * 5000}
    async with _client(FakeVaultReader(docs=[big]), max_result_chars=1000) as c:
        with pytest.raises(ToolError, match=r"\[result_too_large\]"):
            await c.call_tool("vault_get_document", {"doc_id": "big"})
        with pytest.raises(ToolError, match=r"\[result_too_large\]"):
            await c.call_tool("vault_search_documents", {})


async def test_get_document_unknown_is_not_found():
    async with _client() as c:
        assert (await c.call_tool("vault_get_document", {"doc_id": "fr1"})).data["mime_type"] == "text/html"
        with pytest.raises(ToolError, match=r"\[not_found\]"):
            await c.call_tool("vault_get_document", {"doc_id": "nope"})


async def test_corpus_stats_counts_live_documents_with_null_as_a_group():
    reader = FakeVaultReader()
    async with _client(reader) as c:
        data = (await c.call_tool("vault_corpus_stats", {"group_by": "year"})).data
        with pytest.raises(ToolError):
            await c.call_tool("vault_corpus_stats", {"group_by": "sha256"})
    assert data["total"] == 4
    assert {"key": None, "count": 1} in data["groups"]
    assert reader.calls[0] == ("stats", "year", None, 50)


async def test_document_resource_is_on_the_root_uri():
    async with _client() as c:
        templates = [t.uri_template for t in await c.list_resource_templates()]
        contents = await c.read_resource("vault://documents/us1")
        assert templates == ["vault://documents/{doc_id}"]
        with pytest.raises(Exception, match=r"^\[not_found\] "):
            await c.read_resource("vault://documents/nope")
    doc = json.loads(contents[0].text)
    assert doc["doc_id"] == "us1" and doc["date"] == "2010-01-13"


async def test_runs_status_bundles_runs_source_runs_errors_and_health():
    reader = FakeVaultReader()
    async with _client(reader, max_error_samples=2, max_text_chars=8) as c:
        data = (await c.call_tool("vault_runs_status", {"source_code": "xx", "limit": "5"})).data
    # Run grain: the crashed run with no source detail shows only in `runs`.
    assert [r["run_id"] for r in data["runs"]] == ["r1", "r0"]
    assert data["runs"][1]["started_at"] == "2026-10-01T05:00:00"  # naive stays naive
    assert [r["run_id"] for r in data["source_runs"]] == ["r1"]
    assert data["source_runs"][0]["error_samples"] == ["HTTP 404"] * 2
    assert data["open_discovery_errors"][0]["error_class"] == "HTTP404"
    assert data["open_discovery_errors"][0]["seen_at_is_ingestion_time"] is True
    assert data["open_discovery_errors"][0]["error"] == "not foun"
    assert data["source_health"] == []
    assert ("runs_status", None, "xx", None, 5, 2, 8) in reader.calls
    assert ("errors", None, "xx", 5, 8) in reader.calls
    assert ("recent_runs", None, None, 5) in reader.calls


async def test_runs_status_values_are_json_safe():
    async with _client() as c:
        data = (await c.call_tool("vault_runs_status", {"since": "2026-09-01T00:00:00Z"})).data
    health = data["source_health"][0]
    assert health["expected_every_days"] == 45.5 and health["observed_every_days"] is None


async def test_server_info_reports_toolsets_tier_and_freshness():
    async with _client() as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["toolsets"] == {"vault": "enabled", "rag": "not configured", "companies": "not configured",
                                "datasets": "not configured"}
    assert info["tier"] == "act"  # stdio: local user
    assert info["version"]
    assert info["freshness"]["live_documents"] == 4
    assert info["freshness"]["last_ingest_seen_at"].startswith("2026-10-01")


async def test_without_a_vault_only_server_info_is_mounted():
    async with Client(build_server(Settings(), vault_reader=None)) as c:
        names = [t.name for t in await c.list_tools()]
        templates = await c.list_resource_templates()
        info = (await c.call_tool("server_info", {})).data
    assert names == ["server_info"]
    assert templates == []
    assert info["toolsets"]["vault"] == "not configured" and info["freshness"] is None


async def test_audit_logs_one_line_per_call_without_arguments(caplog):
    caplog.set_level(logging.INFO, logger="myopenfund_mcp.audit")
    async with _client() as c:
        await c.call_tool("vault_search_documents", {"text": "secret-term"})
        with pytest.raises(ToolError):
            await c.call_tool("vault_get_document", {"doc_id": "nope"})
    lines = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.audit"]
    assert len(lines) == 2
    assert "tool=vault_search_documents client=local outcome=ok" in lines[0]
    assert "tool=vault_get_document client=local outcome=error" in lines[1]
    assert not any("secret-term" in line or "nope" in line for line in lines)


LEAK = RuntimeError("postgres://u:hunter2@h/db failed")


async def test_unexpected_errors_are_masked_for_tools_server_info_and_the_resource(caplog):
    caplog.set_level(logging.INFO, logger="myopenfund_mcp")
    async with _client(FakeVaultReader(fail_with=LEAK)) as c:
        messages = []
        for name, args in [("vault_search_documents", {}), ("vault_get_document", {"doc_id": "us1"}),
                           ("vault_corpus_stats", {}), ("vault_runs_status", {})]:
            with pytest.raises(ToolError) as caught:
                await c.call_tool(name, args)
            messages.append(str(caught.value))
        info = (await c.call_tool("server_info", {})).data
        messages.append(json.dumps(info))
        with pytest.raises(Exception) as caught:
            await c.read_resource("vault://documents/us1")
        messages.append(str(caught.value))
    messages += [r.getMessage() for r in caplog.records]
    for message in messages:
        assert "hunter2" not in message and "postgres://" not in message, message
    assert info["freshness"] is None and info["freshness_error"] == "backend_unavailable"
    lines = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.audit"]
    assert len(lines) == 5
    assert sum("outcome=error" in line for line in lines) == 4
    assert "tool=server_info client=local outcome=ok" in lines[4]
    logged = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.server"]
    assert any("RuntimeError" in line for line in logged), logged


async def test_server_info_survives_a_vault_outage():
    reader = FakeVaultReader(fail_with=tool_error(BACKEND_UNAVAILABLE, "the vault database is unreachable"))
    async with _client(reader) as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["version"] and info["tier"] == "act"
    assert info["toolsets"]["vault"] == "enabled"
    assert info["freshness"] is None and info["freshness_error"] == "backend_unavailable"


async def test_server_info_has_no_freshness_error_when_the_read_succeeds():
    async with _client() as c:
        info = (await c.call_tool("server_info", {})).data
    assert "freshness_error" not in info


async def test_masking_keeps_our_coded_errors():
    async with _client() as c:
        with pytest.raises(ToolError, match=r"^\[not_found\] "):
            await c.call_tool("vault_get_document", {"doc_id": "nope"})
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] "):
            await c.call_tool("vault_search_documents", {"limit": 0})


async def test_since_description_states_finished_and_in_flight_semantics():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    since = tools["vault_runs_status"].input_schema["properties"]["since"]["description"]
    assert "FINISHED" in since and "in flight" in since and "source_runs" in since


async def test_parameter_descriptions_are_present():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    search = tools["vault_search_documents"].input_schema["properties"]
    for name in ("doc_type", "language", "year", "mime_type", "include_deleted", "limit",
                 "sort_by", "sort_dir"):
        assert search[name].get("description"), name
    assert "refused" in search["limit"]["description"]
    assert search["text"]["description"] == (
        "Case-insensitive substring of the title, matched literally (% and _ are ordinary characters)")
    stats = tools["vault_corpus_stats"].input_schema["properties"]
    assert "maximum number of groups returned" in stats["limit"]["description"]
    assert "refused" in stats["limit"]["description"] and stats["corpus"].get("description")
    assert stats["group_by"].get("description")
    runs = tools["vault_runs_status"].input_schema["properties"]
    assert "each" in runs["limit"]["description"] and "refused" in runs["limit"]["description"]
    assert runs["corpus"]["description"] == search["corpus"]["description"]
    assert runs["source_code"]["description"] == search["source_code"]["description"]


async def test_search_description_warns_that_pages_can_shift_while_ingesting():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    search = tools["vault_search_documents"].description
    assert "shift" in search and "ingesting" in search


@pytest.mark.parametrize(("given", "expected"), [
    ("2026-09-01T00:00:00", dt.datetime(2026, 9, 1, tzinfo=dt.UTC)),
    ("2026-09-01T02:00:00+02:00", dt.datetime(2026, 9, 1, 2, tzinfo=dt.timezone(dt.timedelta(hours=2)))),
])
async def test_runs_status_since_is_utc_when_naive_and_kept_otherwise(given, expected):
    reader = FakeVaultReader()
    async with _client(reader) as c:
        await c.call_tool("vault_runs_status", {"since": given})
    sinces = [call[3] for call in reader.calls if call[0] == "runs_status"]
    sinces += [call[2] for call in reader.calls if call[0] == "recent_runs"]
    assert len(sinces) == 2
    for since in sinces:
        assert since == expected and since.utcoffset() == expected.utcoffset()


@pytest.mark.parametrize("missing", ["vault_query", "psycopg2"])
def test_vault_url_without_the_extra_is_a_config_error(monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ConfigError) as caught:
        build_server(Settings(vault_database_url="postgresql://u:hunter2@h/db"))
    assert str(caught.value) == ("MOF_VAULT_DATABASE_URL is set but the [vault] extra is not installed: "
                                 "pip install 'myopenfund-mcp[vault]'")
    assert caught.value.__cause__ is None and caught.value.__context__ is None


class _HugeTotalReader(FakeVaultReader):
    def search_documents(self, **kwargs):
        _, rows = super().search_documents(**{**kwargs, "offset": 0})
        return 10 * MAX_OFFSET, rows[:1]


async def test_no_cursor_is_issued_past_the_offset_bound():
    async with _client(_HugeTotalReader()) as c:
        data = (await c.call_tool("vault_search_documents",
                                  {"limit": 1, "cursor": encode_cursor(MAX_OFFSET - 1)})).data
        assert data["returned"] == 1 and data["next_cursor"] is not None  # end == MAX_OFFSET
        data = (await c.call_tool("vault_search_documents",
                                  {"limit": 1, "cursor": encode_cursor(MAX_OFFSET)})).data
    assert data["returned"] == 1 and data["next_cursor"] is None  # end would be MAX_OFFSET + 1


def test_http_without_tokens_is_a_config_error():
    with pytest.raises(ConfigError):
        build_server(Settings(transport="http"), vault_reader=FakeVaultReader())


async def test_date_from_after_date_to_is_refused():
    async with _client() as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] date_from is after date_to"):
            await c.call_tool("vault_search_documents", {"date_from": "2011-01-02", "date_to": "2011-01-01"})


@pytest.mark.parametrize(("tool", "arg"), [("vault_get_document", "doc_id"),
                                           ("vault_search_documents", "text"),
                                           ("vault_search_documents", "cursor")])
async def test_string_arguments_are_length_capped(tool, arg):
    async with _client() as c:
        with pytest.raises(ToolError, match="at most 200 characters"):
            await c.call_tool(tool, {arg: "a" * 201})
        if arg != "cursor":
            await c.call_tool(tool, {arg: "a" * 200}, raise_on_error=False)


async def test_resource_body_is_the_compact_json_the_cap_measures():
    async with _client() as c:
        contents = await c.read_resource("vault://documents/fr1")
    text = contents[0].text
    assert text == json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    assert "Décision" in text
