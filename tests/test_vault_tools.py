import datetime as dt
import json
import logging
import sys

import pytest
from fakes import (
    DOC_TYPE_NEEDS_CORPUS,
    DOCS,
    MANY_HEALTH,
    TWINS,
    FakeVaultReader,
    health_row,
    invalid_query_reader,
)
from fastmcp import Client
from fastmcp.exceptions import ToolError

from myopenfund_mcp.config import ConfigError, Settings
from myopenfund_mcp.errors import BACKEND_UNAVAILABLE, SCHEMA_OUTDATED, tool_error
from myopenfund_mcp.paging import MAX_OFFSET, encode_cursor
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio

VAULT_TOOLS = {"vault_search_documents", "vault_get_document", "vault_corpus_stats", "vault_runs_status",
               "vault_vocabulary"}


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
    assert "`source_health`" in runs.split("`limit`", 1)[1] and "overdue" in runs
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


async def test_doc_type_without_corpus_carries_vault_querys_message_whole():
    async with _client() as c:
        for tool, args in [("vault_search_documents", {"doc_type": "A1"}),
                           ("vault_corpus_stats", {"group_by": "doc_type"})]:
            with pytest.raises(ToolError) as caught:
                await c.call_tool(tool, args)
            assert str(caught.value) == f"[invalid_argument] {DOC_TYPE_NEEDS_CORPUS}", tool


async def test_doc_type_with_corpus_is_answered():
    async with _client() as c:
        filters = {"doc_type": "A1", "corpus": "central-bank"}
        page = (await c.call_tool("vault_search_documents", filters)).data
        grouping = {"group_by": "doc_type", "corpus": "central-bank"}
        stats = (await c.call_tool("vault_corpus_stats", grouping)).data
    assert [d["doc_id"] for d in page["documents"]] == ["fr1"]
    assert {g["key"] for g in stats["groups"]} == {"A1", "C1"}


async def test_descriptions_tell_the_model_doc_type_needs_a_corpus():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    doc_type = tools["vault_search_documents"].input_schema["properties"]["doc_type"]["description"]
    group_by = tools["vault_corpus_stats"].input_schema["properties"]["group_by"]["description"]
    for text in (doc_type, group_by, tools["vault_search_documents"].description,
                 tools["vault_corpus_stats"].description):
        assert "corpus" in text and "ask the user" in text, text
    assert "invalid_argument" in doc_type and "invalid_argument" in group_by


async def test_vocabulary_lists_codes_with_labels_per_corpus():
    reader = FakeVaultReader()
    async with _client(reader) as c:
        every = (await c.call_tool("vault_vocabulary", {})).data
        company = (await c.call_tool("vault_vocabulary", {"corpus": "company", "kind": "doc_type"})).data
    assert [(v["corpus"], v["kind"], v["code"]) for v in every["codes"]] == [
        ("central-bank", "doc_type", "A1"), ("central-bank", "doc_type", "C1"),
        ("central-bank", "source_code", "us"), ("company", "doc_type", "A1"), ("company", "doc_type", "A2"),
        ("company", "entity_scheme", "cik")]
    assert (every["returned"], every["truncated"]) == (6, False)
    assert [(v["code"], v["label"], v["family"]) for v in company["codes"]] == [
        ("A1", "Annual report (10-K)", "A"), ("A2", "Quarterly report (10-Q)", "A")]
    # One row more than the page is read, to tell a full page from a cut one.
    assert reader.calls == [("vocabulary", None, None, 51), ("vocabulary", "company", "doc_type", 51)]


async def test_vocabulary_says_when_limit_cut_the_list():
    async with _client() as c:
        cut = (await c.call_tool("vault_vocabulary", {"limit": 2})).data
        exact = (await c.call_tool("vault_vocabulary", {"limit": 6})).data
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] limit"):
            await c.call_tool("vault_vocabulary", {"limit": 0})
        with pytest.raises(ToolError):
            await c.call_tool("vault_vocabulary", {"kind": "language"})
    assert (cut["returned"], cut["truncated"], len(cut["codes"])) == (2, True, 2)
    assert (exact["returned"], exact["truncated"]) == (6, False)


async def test_vocabulary_description_and_doc_type_point_to_each_other():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    vocabulary = tools["vault_vocabulary"]
    assert vocabulary.annotations.read_only_hint is True
    assert "within its corpus" in vocabulary.description and "vault_corpus_stats" in vocabulary.description
    assert "truncated" in vocabulary.description
    doc_type = tools["vault_search_documents"].input_schema["properties"]["doc_type"]["description"]
    assert "vault_vocabulary" in doc_type


async def test_get_document_over_the_cap_says_to_pass_corpus_not_a_limit():
    """Two corpora each under the cap, together over it: corpus is the way to a smaller result."""
    twins = [{**t, "title": "x" * 600} for t in TWINS]
    async with _client(FakeVaultReader(docs=twins), max_result_chars=1300) as c:
        with pytest.raises(ToolError) as caught:
            await c.call_tool("vault_get_document", {"doc_id": "twin"})
        one = (await c.call_tool("vault_get_document", {"doc_id": "twin", "corpus": "company"})).data
        with pytest.raises(Exception) as resource:
            await c.read_resource("vault://documents/twin")
    assert str(caught.value).startswith("[result_too_large] ")
    assert "pass `corpus`" in str(caught.value) and "limit" not in str(caught.value)
    assert [d["corpus"] for d in one] == ["company"]
    assert "vault_get_document" in str(resource.value) and "corpus" in str(resource.value)


async def test_one_document_over_the_cap_does_not_suggest_corpus():
    big = {**DOCS[0], "doc_id": "big", "title": "x" * 5000}
    async with _client(FakeVaultReader(docs=[big]), max_result_chars=1000) as c:
        with pytest.raises(ToolError) as caught:
            await c.call_tool("vault_get_document", {"doc_id": "big", "corpus": "central-bank"})
    assert "MOF_MCP_MAX_RESULT_CHARS" in str(caught.value) and "limit" not in str(caught.value)


async def test_a_result_over_the_cap_is_refused_not_cut():
    big = {**DOCS[0], "doc_id": "big", "title": "x" * 5000}
    async with _client(FakeVaultReader(docs=[big]), max_result_chars=1000) as c:
        with pytest.raises(ToolError, match=r"\[result_too_large\]"):
            await c.call_tool("vault_get_document", {"doc_id": "big"})
        with pytest.raises(ToolError, match=r"\[result_too_large\]"):
            await c.call_tool("vault_search_documents", {})


async def test_get_document_unknown_is_not_found():
    async with _client() as c:
        fr1 = (await c.call_tool("vault_get_document", {"doc_id": "fr1"})).data
        assert fr1[0]["mime_type"] == "text/html"
        with pytest.raises(ToolError, match=r"\[not_found\]"):
            await c.call_tool("vault_get_document", {"doc_id": "nope"})


async def test_get_document_without_corpus_returns_one_row_per_corpus():
    reader = FakeVaultReader(docs=DOCS + TWINS)
    async with _client(reader) as c:
        result = await c.call_tool("vault_get_document", {"doc_id": "twin"})
    assert [(d["corpus"], d["title"]) for d in result.data] == [("central-bank", "Rate decision"),
                                                                ("company", "Form 10-K")]
    assert reader.calls == [("get", "twin", None)]
    # The text block the model reads is the list itself, not a wrapper.
    assert json.loads(result.content[0].text) == result.data
    # MCP structured output must be an object: the list is wrapped as {"result": [...]} (README says so).
    assert result.structured_content == {"result": json.loads(result.content[0].text)}


async def test_get_document_with_corpus_returns_that_corpus_only():
    reader = FakeVaultReader(docs=DOCS + TWINS)
    async with _client(reader) as c:
        rows = (await c.call_tool("vault_get_document", {"doc_id": "twin", "corpus": "company"})).data
        single = (await c.call_tool("vault_get_document", {"doc_id": "us1", "corpus": "central-bank"})).data
        with pytest.raises(ToolError, match=r"^\[not_found\] no document 'us1' in corpus 'company'; omit "
                                            r"corpus to look in every corpus"):
            await c.call_tool("vault_get_document", {"doc_id": "us1", "corpus": "company"})
    assert [(d["corpus"], d["entity_value"], d["date_precision"]) for d in rows] == [
        ("company", "0000320193", "day")]
    assert [d["doc_id"] for d in single] == ["us1"]
    assert ("get", "twin", "company") in reader.calls


async def test_get_document_description_says_it_returns_a_list():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    get = tools["vault_get_document"]
    assert "one per corpus" in get.description and "at most one" in get.description
    assert get.input_schema["properties"]["corpus"].get("description")
    assert get.input_schema["required"] == ["doc_id"]


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
    docs = json.loads(contents[0].text)
    assert [(d["doc_id"], d["date"]) for d in docs] == [("us1", "2010-01-13")]


async def test_document_resource_lists_every_corpus_holding_the_doc_id():
    async with _client(FakeVaultReader(docs=DOCS + TWINS)) as c:
        contents = await c.read_resource("vault://documents/twin")
    assert [d["corpus"] for d in json.loads(contents[0].text)] == ["central-bank", "company"]


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


async def test_runs_status_keeps_bad_content_null_apart_from_a_count():
    async with _client() as c:
        data = (await c.call_tool("vault_runs_status", {})).data
    assert [(r["source_code"], r["bad_content"]) for r in data["source_runs"]] == [("xx", None), ("us", 2)]
    health = data["source_health"][0]
    assert health["source_bad_content_7d"] is None and health["last_run_bad_content"] == 0


async def test_runs_status_description_explains_bad_content():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    runs = tools["vault_runs_status"].description
    assert "bad_content" in runs and "docs_failed" in runs and "not reported" in runs
    assert "source_bad_content_7d" in runs and "last_run_bad_content" in runs


async def test_runs_status_values_are_json_safe():
    async with _client() as c:
        data = (await c.call_tool("vault_runs_status", {"since": "2026-09-01T00:00:00Z"})).data
    health = data["source_health"][0]
    assert health["expected_interval_days"] == 45.5 and health["source_docs_new_median_per_run_90d"] is None


async def test_runs_status_limit_bounds_source_health_too():
    """79 series (the production vault's size): at limit=1 the result was 66,000 characters."""
    reader = FakeVaultReader(health=MANY_HEALTH)
    async with _client(reader) as c:
        result = await c.call_tool("vault_runs_status", {"limit": 1})
    data = result.data
    assert len(data["source_health"]) == 1
    assert data["source_health_total"] == 79
    assert len(result.content[0].text) < 5_000


async def test_runs_status_keeps_the_unhealthy_series_first():
    rows = [health_row("a"), health_row("b", cadence_status=None, days_late=None),
            health_row("c", cadence_status="soon"), health_row("d", cadence_status="overdue", days_late=5),
            health_row("e", cadence_status="overdue", days_late=None),
            health_row("f", cadence_status="overdue", days_late=40), health_row("g", cadence_status="soon"),
            health_row("h", cadence_status="overdue", days_late=40)]
    async with _client(FakeVaultReader(health=rows)) as c:
        top = (await c.call_tool("vault_runs_status", {"limit": 3})).data
        every = (await c.call_tool("vault_runs_status", {"limit": 200})).data
    # overdue (most days late first, NULL last), then soon, then the rest; ties keep the view's order.
    assert [h["source_code"] for h in every["source_health"]] == ["f", "h", "d", "e", "c", "g", "a", "b"]
    assert [h["source_code"] for h in top["source_health"]] == ["f", "h", "d"]
    assert top["source_health_total"] == every["source_health_total"] == 8


async def test_runs_status_small_health_is_unchanged():
    async with _client() as c:
        data = (await c.call_tool("vault_runs_status", {})).data
    assert [h["source_code"] for h in data["source_health"]] == ["us"]
    assert data["source_health_total"] == 1


async def test_runs_status_too_large_is_refused_not_cut():
    async with _client(FakeVaultReader(health=MANY_HEALTH), max_result_chars=5_000) as c:
        with pytest.raises(ToolError, match=r"^\[result_too_large\] .*smaller limit"):
            await c.call_tool("vault_runs_status", {"limit": 50})
        data = (await c.call_tool("vault_runs_status", {"limit": 2})).data
    assert len(data["source_health"]) == 2


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
                           ("vault_corpus_stats", {}), ("vault_runs_status", {}), ("vault_vocabulary", {})]:
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
    assert len(lines) == 6
    assert sum("outcome=error" in line for line in lines) == 5
    assert "tool=server_info client=local outcome=ok" in lines[5]
    logged = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.server"]
    assert any("RuntimeError" in line for line in logged), logged


async def test_server_info_freshness_error_is_the_readers_code():
    reader = FakeVaultReader(fail_with=tool_error(SCHEMA_OUTDATED, "the vault lacks a column or table"))
    async with _client(reader) as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["freshness"] is None and info["freshness_error"] == "schema_outdated"


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
    assert "source_health" in runs["limit"]["description"]
    assert runs["corpus"]["description"] == search["corpus"]["description"]
    assert runs["source_code"]["description"] == search["source_code"]["description"]


async def test_search_rows_carry_the_contract_columns_and_the_description_says_how_to_read_them():
    async with _client(FakeVaultReader(docs=TWINS)) as c:
        tools = {t.name: t for t in await c.list_tools()}
        rows = (await c.call_tool("vault_search_documents", {"corpus": "company"})).data["documents"]
    assert [(r["date_precision"], r["date_kind"], r["entity_scheme"], r["entity_value"]) for r in rows] == [
        ("day", "publication", "cik", "0000320193")]
    search = tools["vault_search_documents"].description
    for column in ("date_precision", "date_kind", "entity_scheme", "entity_value"):
        assert column in search, column


async def test_search_description_warns_that_pages_can_shift_while_ingesting():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    search = tools["vault_search_documents"].description
    assert "shift" in search and "ingesting" in search


async def test_search_description_states_the_total_order():
    """vault-query 0.5.1 breaks ties by doc_id, then corpus: the description says pages are a
    total order (no repeat, no skip) and no longer tells the model to filter by corpus for it."""
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    search = " ".join(tools["vault_search_documents"].description.split())
    assert "doc_id, then corpus" in search and "never repeat or skip" in search
    assert "within a corpus only" not in search


def test_installed_vault_query_breaks_ties_by_doc_id_then_corpus():
    """The pin is myopenfund-vault-query 0.1.0 (the former vault-query 0.6.0): its search ORDER BY ends with doc_id, corpus."""
    import inspect
    from importlib.metadata import version

    from vault_query import documents

    assert version("myopenfund-vault-query") == "0.1.0"
    assert "NULLS LAST, doc_id, corpus" in inspect.getsource(documents.search_documents)


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
                                           ("vault_get_document", "corpus"),
                                           ("vault_vocabulary", "corpus"),
                                           ("vault_search_documents", "text"),
                                           ("vault_search_documents", "cursor")])
async def test_string_arguments_are_length_capped(tool, arg):
    required = {"doc_id": "us1"} if tool == "vault_get_document" else {}
    async with _client() as c:
        with pytest.raises(ToolError, match="at most 200 characters"):
            await c.call_tool(tool, {**required, arg: "a" * 201})
        if arg != "cursor":
            await c.call_tool(tool, {**required, arg: "a" * 200}, raise_on_error=False)


async def test_resource_body_is_the_compact_json_the_cap_measures():
    async with _client() as c:
        contents = await c.read_resource("vault://documents/fr1")
    text = contents[0].text
    assert text == json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    assert "Décision" in text


async def test_a_nul_in_text_is_refused_naming_text_not_the_library_name():
    reader = FakeVaultReader()
    async with _client(reader) as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] text must not contain a NUL"):
            await c.call_tool("vault_search_documents", {"text": "a\x00b"})
    assert not reader.calls


@pytest.mark.parametrize("since", ["2026-10-01T00:00:00+16:00", "2026-10-01T00:00:00-23:59"])
async def test_runs_status_refuses_an_offset_postgres_cannot_take(since):
    reader = FakeVaultReader()
    async with _client(reader) as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] since: the UTC offset"):
            await c.call_tool("vault_runs_status", {"since": since})
    assert not reader.calls


@pytest.mark.parametrize("since", ["2026-10-01T00:00:00+15:59", "2026-10-01T00:00:00-15:59"])
async def test_runs_status_accepts_the_largest_offsets(since):
    reader = FakeVaultReader()
    async with _client(reader) as c:
        await c.call_tool("vault_runs_status", {"since": since})
    assert any(call[0] == "recent_runs" for call in reader.calls)


async def test_since_and_text_descriptions_state_the_bound_and_literal_match():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    since = tools["vault_runs_status"].input_schema["properties"]["since"]["description"]
    assert "15:59" in since
    text = tools["vault_search_documents"].input_schema["properties"]["text"]["description"]
    assert "literally" in text


async def test_runs_status_refuses_an_offset_with_seconds():
    """Refused before any read, by the argument parser or by the offset check."""
    reader = FakeVaultReader()
    async with _client(reader) as c:
        with pytest.raises(ToolError):
            await c.call_tool("vault_runs_status", {"since": "2026-10-01T00:00:00+05:00:30"})
    assert not reader.calls


@pytest.mark.parametrize(("given", "expected"), [
    ("2026-09-01T06:00:00", dt.datetime(2026, 9, 1, 6, tzinfo=dt.UTC)),
    ("2026-09-01T08:00:00+02:00", dt.datetime(2026, 9, 1, 8, tzinfo=dt.timezone(dt.timedelta(hours=2)))),
])
async def test_runs_status_until_reaches_both_run_reads(given, expected):
    reader = FakeVaultReader()
    async with _client(reader) as c:
        await c.call_tool("vault_runs_status", {"until": given})
    assert reader.untils == [expected, expected]
    assert all(u.utcoffset() == expected.utcoffset() for u in reader.untils)


async def test_runs_status_refuses_an_until_offset_postgres_cannot_take():
    reader = FakeVaultReader()
    async with _client(reader) as c:
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] until: the UTC offset"):
            await c.call_tool("vault_runs_status", {"until": "2026-10-01T00:00:00+16:00"})
    assert not reader.calls


async def test_until_description_states_the_inclusive_bound():
    async with _client() as c:
        tools = {t.name: t for t in await c.list_tools()}
    until = tools["vault_runs_status"].input_schema["properties"]["until"]["description"]
    assert "at or before" in until and "in flight" in until and "15:59" in until
