"""Opt-in: rag_search end to end against a real Qdrant, through the MCP client.

    MOF_MCP_IT_QDRANT_URL=http://127.0.0.1:6333 python -m pytest -m integration tests/integration/test_real_qdrant.py

Use a THROWAWAY Qdrant, NEVER a production one: the test creates collections named
`mof-mcp-it-<random>`, seeds them through rag-engine (store, schema, payload indexes) and deletes them at
the end. The embedder is rag-engine's
FakeEmbedder, injected into the backend: the real model (1.1 GB) is not needed.

Optional, for a Qdrant started with API keys: MOF_MCP_IT_QDRANT_API_KEY (a key that can write: seeds and
cleans up) and MOF_MCP_IT_QDRANT_READ_ONLY_API_KEY (the key the server is then given, as in a
deployment). No key is ever printed or put in a skip reason.
"""
import asyncio
import os
import uuid
from urllib.parse import urlsplit

import pytest
from fastmcp import Client

from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

URL = os.environ.get("MOF_MCP_IT_QDRANT_URL")
WRITE_KEY = os.environ.get("MOF_MCP_IT_QDRANT_API_KEY") or None
READ_KEY = os.environ.get("MOF_MCP_IT_QDRANT_READ_ONLY_API_KEY") or WRITE_KEY

pytestmark = [pytest.mark.integration, pytest.mark.anyio,
              pytest.mark.skipif(not URL, reason="MOF_MCP_IT_QDRANT_URL not set")]


def _settings(collection: str, **overrides) -> Settings:
    parts = urlsplit(URL)
    return Settings(**{"rag_qdrant_host": parts.hostname, "rag_qdrant_port": parts.port or 6333,
                       "rag_qdrant_https": parts.scheme == "https", "rag_qdrant_api_key": READ_KEY,
                       "rag_qdrant_timeout": 5, "rag_collection": collection, "rag_model_cache": "/unused",
                       **overrides})


def _client(collection: str, fake, **overrides) -> Client:
    from myopenfund_mcp.rag.backend import EngineRagBackend

    settings = _settings(collection, **overrides)
    # The store is the real one, built from the settings as in a deployment; only the embedder is injected.
    return Client(build_server(settings, rag_backend=EngineRagBackend(settings, embedder=fake)))


async def _error(client, arguments) -> str:
    result = await client.call_tool("rag_search", arguments, raise_on_error=False)
    assert result.is_error, result.data
    return result.content[0].text


def _quoted(values) -> str:
    return ", ".join(f'"{value}"' for value in values)


async def _rag_info(client, *, until=lambda info: True) -> dict:
    """`server_info`'s rag object, asked again (for 10 s at most) until `until` holds."""
    for _ in range(100):
        info = (await client.call_tool("server_info", {})).data["rag"]
        if until(info):
            break
        await asyncio.sleep(0.1)
    return info


@pytest.fixture(scope="module")
def writer():
    from rag_engine.store import QdrantStore

    return QdrantStore.from_url(URL, WRITE_KEY, 10)


@pytest.fixture(scope="module")
def seeded(writer):
    """(collection, embedder): the fixture corpus in a collection of its own, deleted afterwards."""
    from rag_fixture import seed

    collection = f"mof-mcp-it-{uuid.uuid4().hex[:12]}"
    try:
        yield collection, seed(writer, collection)
    finally:
        writer.client.delete_collection(collection)


async def test_rag_search_end_to_end(seeded):
    from rag_fixture import CHUNKS, DOC_TYPES, ECB_RATES, FOMC, MODEL, SOURCE_CODES

    collection, fake = seeded
    async with _client(collection, fake) as c:
        info = (await c.call_tool("server_info", {})).data
        assert info["toolsets"]["rag"] == "enabled"
        assert info["rag"] == {"status": "ready", "collection": collection, "model": MODEL,
                               "engine_version": info["rag"]["engine_version"], "source_codes": SOURCE_CODES,
                               "doc_types": DOC_TYPES, "languages": ["en", "ja"], "default_limit": 8,
                               "max_limit": 20}

        data = (await c.call_tool("rag_search", {"query": ECB_RATES})).data
        assert data["count"] == CHUNKS
        assert data["hits"][0] == {
            "rank": 1, "score": 1.0, "text": ECB_RATES, "doc_id": "ecb/A1/2022-07-21",
            "title": "Monetary policy decisions", "date": "2022-07-21", "source_code": "ecb", "doc_type": "A1",
            "language": "en", "page": 1, "chunk_number": 0,
            "citation": "Monetary policy decisions (ecb A1, 2022-07-21), p. 1"}
        scores = [h["score"] for h in data["hits"]]
        assert scores == sorted(scores, reverse=True)

        # Filters on the indexed payload fields; both date bounds inclusive; the undated document left out.
        data = (await c.call_tool("rag_search", {"query": FOMC, "source_codes": ["us", "ecb"], "doc_types": ["A3"],
                                                 "languages": ["en"], "date_from": "2022-06-15",
                                                 "date_to": "2022-06-15", "limit": 5})).data
        assert [(h["doc_id"], h["page"], h["citation"]) for h in data["hits"]] == [
            ("us/A3/2022-06-15", 5, "Minutes of the Federal Open Market Committee (us A3, 2022-06-15), p. 5")]
        assert data["filters"] == {"source_codes": ["us", "ecb"], "doc_types": ["A3"], "languages": ["en"],
                                   "date_from": "2022-06-15", "date_to": "2022-06-15"}
        dated = (await c.call_tool("rag_search", {"query": "rates", "date_from": "1900-01-01"})).data
        assert "gb/B1/undated" not in {h["doc_id"] for h in dated["hits"]} and dated["count"] == CHUNKS - 1
        undated = (await c.call_tool("rag_search", {"query": "rates", "source_codes": ["gb"]})).data
        assert [(h["date"], h["citation"]) for h in undated["hits"]] == [(None, "gb/B1/undated (gb B1), p. 3")]

        # The valid values are the collection's own (facet API on the keyword indexes).
        assert await _error(c, {"query": "rates", "source_codes": ["ECB"]}) == (
            '[invalid_argument] source_codes: unknown value "ECB" (did you mean "ecb"? values are matched '
            "exactly); valid values: " + _quoted(SOURCE_CODES))
        assert await _error(c, {"query": "rates", "doc_types": ["minutes"]}) == (
            '[invalid_argument] doc_types: unknown value "minutes"; valid values: ' + _quoted(DOC_TYPES))

        assert await _error(c, {"query": "rates", "source_codes": '["ecb"]'}) == (
            '[invalid_argument] source_codes must be a JSON array of strings, e.g. ["ecb"]; omit it for no '
            "restriction")
        assert (await _error(c, {"query": "rates", "date_from": "2016"})).startswith("[invalid_argument] date_from")
        assert (await _error(c, {"query": "rates", "limit": 21})).startswith("[limit_exceeded] limit 21 is above")
        assert (await _error(c, {"query": " "})).startswith("[invalid_argument] query")

    async with _client(collection, fake, max_result_chars=1000) as c:
        message = await _error(c, {"query": "rates"})
        assert message.startswith("[result_too_large]") and "limit=" in message


async def test_the_values_are_read_by_a_scroll_when_the_collection_has_no_keyword_index(writer):
    from rag_fixture import DOC_TYPES, SOURCE_CODES, embedder, seed

    collection = f"mof-mcp-it-{uuid.uuid4().hex[:12]}"
    try:
        seed(writer, collection)
        for field in ("source_code", "doc_type", "language"):
            writer.client.delete_payload_index(collection, field)
        async with _client(collection, embedder()) as c:
            # The scroll runs in the background from the start: a search never waits for it.
            assert (await c.call_tool("rag_search", {"query": "rates", "doc_types": ["E1"]})).data["count"] == 1
            info = await _rag_info(c, until=lambda info: info["source_codes"] is not None)
            assert (info["source_codes"], info["doc_types"]) == (SOURCE_CODES, DOC_TYPES)
            assert (await _error(c, {"query": "rates", "source_codes": ["ECB"]})).endswith(
                "valid values: " + _quoted(SOURCE_CODES))
            assert (await c.call_tool("rag_search", {"query": "rates", "doc_types": ["E1"]})).data["count"] == 1
    finally:
        writer.client.delete_collection(collection)


async def test_a_code_ingested_while_the_server_runs_is_accepted_without_waiting_for_the_ttl(writer):
    from qdrant_client import models
    from rag_engine.schema import build_payload, page_for_id, point_id
    from rag_fixture import SOURCE_CODES, seed

    collection = f"mof-mcp-it-{uuid.uuid4().hex[:12]}"
    try:
        fake = seed(writer, collection)
        # rag_codes_retry_s=0: the read made before a refusal is not held back (5 s by default).
        async with _client(collection, fake, rag_codes_ttl_s=3600, rag_codes_retry_s=0) as c:
            assert (await _error(c, {"query": "rates", "source_codes": ["zz"]})).endswith(
                "valid values: " + _quoted(SOURCE_CODES))
            text = "A document of a source ingested after the server started."
            payload = build_payload("zz/A1/new", {"corpus": "central-bank", "source_code": "zz", "doc_type": "A1",
                                                  "title": "New", "date": "2026-01-01", "language": "en"},
                                    filename="new.pdf", chunk_number=0, ingestion_date="2026-11-17T00:00:00",
                                    text=text, page=0)
            writer.upsert(collection, [models.PointStruct(id=point_id("zz/A1/new", page_for_id(0), 0),
                                                          vector=fake.encode_passage([text])[0].tolist(),
                                                          payload=payload)])
            data = (await c.call_tool("rag_search", {"query": text, "source_codes": ["zz"]})).data
            assert [h["doc_id"] for h in data["hits"]] == ["zz/A1/new"]
            assert "zz" in (await _rag_info(c))["source_codes"]
    finally:
        writer.client.delete_collection(collection)


async def test_a_collection_that_disappears_is_rag_unavailable_then_served_again(writer):
    """What the engine lets through raw (HTTP 404 on a search) is a coded error; the toolset recovers."""
    from rag_fixture import CHUNKS, SOURCE_CODES, embedder, seed

    collection = f"mof-mcp-it-{uuid.uuid4().hex[:12]}"
    try:
        fake = embedder()
        # rag_check_retry_s=0: a failure is not remembered, so every call asks Qdrant (5 s by default).
        async with _client(collection, fake, rag_check_retry_s=0) as c:  # no collection yet: the server starts
            info = (await c.call_tool("server_info", {})).data["rag"]
            assert (info["status"], info["reason"]) == ("unavailable", "collection_missing")
            assert (await _error(c, {"query": "rates"})).startswith("[rag_unavailable] collection_missing: ")
            seed(writer, collection)
            assert (await c.call_tool("rag_search", {"query": "rates"})).data["count"] == CHUNKS
            writer.client.delete_collection(collection)
            assert (await _error(c, {"query": "rates"})).startswith("[rag_unavailable] collection_missing: ")
            seed(writer, collection)
            assert (await c.call_tool("rag_search", {"query": "rates"})).data["count"] == CHUNKS
            assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"
            # The values of the collection as it is now, read after it came back.
            assert (await _error(c, {"query": "rates", "source_codes": ["ECB"]})).endswith(
                "valid values: " + _quoted(SOURCE_CODES))
    finally:
        writer.client.delete_collection(collection)


async def test_a_filter_the_collection_cannot_serve_is_the_error_of_that_search_alone(writer):
    """Strict mode refuses (HTTP 400) a filter on a field without a payload index: here `date`. The other
    searches are served, by the same server, before and after."""
    from qdrant_client import models
    from rag_fixture import CHUNKS, embedder, seed

    collection = f"mof-mcp-it-{uuid.uuid4().hex[:12]}"
    try:
        seed(writer, collection)
        writer.client.delete_payload_index(collection, "date")
        writer.client.update_collection(collection, strict_mode_config=models.StrictModeConfig(
            enabled=True, unindexed_filtering_retrieve=False))
        async with _client(collection, embedder(), rag_check_retry_s=3600) as c:
            for _ in range(2):
                assert await _error(c, {"query": "rates", "date_from": "2022-01-01"}) == (
                    "[rag_unavailable] search_refused: the vector database refused this search, and only this "
                    "one: the collection, as it is indexed, cannot serve it (most often a filter on a field "
                    "without a payload index); retry with fewer filters, without date_from and date_to first")
                assert (await c.call_tool("rag_search", {"query": "rates"})).data["count"] == CHUNKS
                assert (await c.call_tool("rag_search", {"query": "rates", "source_codes": ["ecb"]})).data["count"] == 2
                assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"
    finally:
        writer.client.delete_collection(collection)


async def test_a_failure_is_remembered_and_the_next_calls_answer_at_once(seeded):
    """An address that drops packets would cost every call a whole timeout: only the first one pays."""
    import socket
    import time

    collection, fake = seeded
    with socket.socket() as silent:  # accepts connections and never answers, as a frozen Qdrant does
        silent.bind(("127.0.0.1", 0))
        silent.listen(64)
        async with _client(collection, fake, rag_qdrant_host="127.0.0.1", rag_qdrant_port=silent.getsockname()[1],
                           rag_qdrant_https=False, rag_qdrant_timeout=1, rag_check_retry_s=30) as c:
            started = time.monotonic()
            answers = await asyncio.gather(*[_error(c, {"query": "rates", "source_codes": ["ecb"]})
                                             for _ in range(30)])
            assert time.monotonic() - started < 1  # 30 calls, no request: the check of the start is remembered
            assert all(a.startswith("[rag_unavailable] qdrant_unreachable: ") for a in answers)


async def test_a_collection_of_another_dimension_is_rag_unavailable(seeded):
    from rag_engine.embedding import FakeEmbedder, ModelSpec

    collection, _ = seeded
    other = FakeEmbedder(ModelSpec(name="fake/other", dim=16, passage_prefix="", query_prefix=""))
    async with _client(collection, other) as c:
        assert (await _error(c, {"query": "rates"})).startswith("[rag_unavailable] collection_mismatch: ")


async def test_an_unreachable_qdrant_is_rag_unavailable_and_names_no_host(seeded):
    import socket

    collection, fake = seeded
    with socket.socket() as s:  # a loopback port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    async with _client(collection, fake, rag_qdrant_host="127.0.0.1", rag_qdrant_port=port,
                       rag_qdrant_https=False) as c:
        message = await _error(c, {"query": "rates"})
    assert message.startswith("[rag_unavailable] qdrant_unreachable: ")
    assert "127.0.0.1" not in message and str(port) not in message


@pytest.mark.skipif(not WRITE_KEY, reason="MOF_MCP_IT_QDRANT_API_KEY not set: the Qdrant has no API key")
async def test_a_refused_key_is_rag_unavailable_and_never_echoed(seeded):
    collection, fake = seeded
    wrong = "wrong-" + uuid.uuid4().hex
    async with _client(collection, fake, rag_qdrant_api_key=wrong) as c:
        info = (await c.call_tool("server_info", {})).data
        message = await _error(c, {"query": "rates"})
    assert message.startswith("[rag_unavailable] qdrant_auth: ")
    assert info["rag"]["reason"] == "qdrant_auth" and wrong not in message and wrong not in str(info)


@pytest.mark.skipif(not os.environ.get("MOF_MCP_IT_QDRANT_READ_ONLY_API_KEY"),
                    reason="MOF_MCP_IT_QDRANT_READ_ONLY_API_KEY not set")
def test_the_server_key_cannot_write(seeded):
    """The deployment's guarantee: with the read-only key, Qdrant itself refuses a write."""
    from rag_engine.store import QdrantAuthError, QdrantStore

    reader = QdrantStore.from_url(URL, READ_KEY, 10)
    with pytest.raises(QdrantAuthError):
        reader.ensure_collection(f"mof-mcp-it-{uuid.uuid4().hex[:12]}", 8)
