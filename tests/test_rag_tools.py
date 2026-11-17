"""The rag toolset on a fake engine: rag-engine's own store in qdrant-client's in-process mode, its
FakeEmbedder (no model), and a monkeypatched `search` where an outage is needed."""
import json
import logging
import sys
import threading
import unicodedata

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from rag_engine.embedding import ModelDownloadError, ModelIntegrityError
from rag_engine.store import CollectionMismatchError, CollectionNotFoundError, QdrantAuthError
from rag_fixture import (
    BOE,
    BOJ,
    CHUNKS,
    COLLECTION,
    DIM,
    DOC_TYPES,
    ECB_RATES,
    FOMC,
    MODEL,
    SOURCE_CODES,
    embedder,
    local_store,
    seed,
)

from myopenfund_mcp.config import ConfigError, Settings
from myopenfund_mcp.rag import backend as backend_module
from myopenfund_mcp.rag.backend import EngineRagBackend
from myopenfund_mcp.rag.tools import citation, human_page
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio

HOST, KEY = "qdrant.secret-host.example", "k-hunter2"
HIT_KEYS = ["rank", "score", "text", "doc_id", "title", "date", "source_code", "doc_type", "language", "page",
            "chunk_number", "citation"]


def _settings(**overrides) -> Settings:
    return Settings(**{"rag_qdrant_host": HOST, "rag_qdrant_api_key": KEY, "rag_collection": COLLECTION,
                       "rag_model_cache": "/models", **overrides})


@pytest.fixture(scope="module")
def seeded():
    store = local_store()
    return store, seed(store)


@pytest.fixture
def backend(seeded):
    store, fake = seeded
    return EngineRagBackend(_settings(), store=store, embedder=fake)


def _client(backend, **overrides) -> Client:
    return Client(build_server(_settings(**overrides), rag_backend=backend))


async def _error(client, arguments) -> str:
    result = await client.call_tool("rag_search", arguments, raise_on_error=False)
    assert result.is_error, result.data
    return result.content[0].text


# -- surface -----------------------------------------------------------------------------------------


async def test_the_tool_is_namespaced_read_only_and_described_for_a_model(backend):
    async with _client(backend) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert set(tools) == {"rag_search", "server_info"}
    tool = tools["rag_search"]
    assert tool.annotations.read_only_hint is True
    for needle in ('"ecb"', '"us"', '"gb"', '"au"', '"ca"', '"jp"', '"A1"', '"A2"', '"A3"', '"B1"', '"E1"',
                   '"F1"', "YYYY-MM-DD", "inclusive", "does not write an answer", "counted from 1", "citation",
                   "PASSAGES", "never corrected"):
        assert needle in tool.description, needle
    schema = tool.input_schema
    assert list(schema["properties"]) == ["query", "source_codes", "doc_types", "date_from", "date_to",
                                         "languages", "limit"]
    assert schema["required"] == ["query"] and schema["additionalProperties"] is False
    for name, prop in schema["properties"].items():
        assert prop.get("description"), name
    # Lists are published as arrays of strings only (no string alternative to tempt a model).
    for name in ("source_codes", "doc_types", "languages"):
        assert {"array", "null"} == {branch["type"] for branch in schema["properties"][name]["anyOf"]}, name
    assert "refused" in schema["properties"]["limit"]["description"]


async def test_a_hit_carries_the_passage_its_document_and_a_citation_and_nothing_else(backend):
    async with _client(backend) as c:
        data = (await c.call_tool("rag_search", {"query": ECB_RATES})).data
    assert list(data) == ["query", "filters", "count", "hits"]
    assert data["query"] == ECB_RATES and data["count"] == len(data["hits"]) == CHUNKS
    assert data["filters"] == {"source_codes": None, "doc_types": None, "languages": None,
                               "date_from": None, "date_to": None}
    assert [h["rank"] for h in data["hits"]] == list(range(1, CHUNKS + 1))
    scores = [h["score"] for h in data["hits"]]
    assert scores == sorted(scores, reverse=True) and all(round(s, 4) == s for s in scores)
    first = data["hits"][0]
    assert list(first) == HIT_KEYS  # no point_id (64-bit), no payload, no corpus
    assert first == {"rank": 1, "score": 1.0, "text": ECB_RATES, "doc_id": "ecb/A1/2022-07-21",
                     "title": "Monetary policy decisions", "date": "2022-07-21", "source_code": "ecb",
                     "doc_type": "A1", "language": "en", "page": 1, "chunk_number": 0,
                     "citation": "Monetary policy decisions (ecb A1, 2022-07-21), p. 1"}


async def test_pages_are_counted_from_one_and_absent_pages_stay_null(backend):
    async with _client(backend) as c:
        by_text = {}
        for text in (ECB_RATES, FOMC, BOJ, BOE):
            by_text[text] = (await c.call_tool("rag_search", {"query": text, "limit": 1})).data["hits"][0]
    assert by_text[ECB_RATES]["page"] == 1  # stored 0: the first page
    assert by_text[FOMC]["page"] == 5  # stored 4
    assert by_text[FOMC]["citation"] == "Minutes of the Federal Open Market Committee (us A3, 2022-06-15), p. 5"
    assert by_text[BOJ]["page"] is None  # a document without pages: no page part
    assert by_text[BOJ]["score"] == 1.0 and by_text[BOJ]["text"] == unicodedata.normalize("NFKD", BOJ) != BOJ
    assert by_text[BOJ]["citation"] == "Outlook for Economic Activity and Prices (jp E1, 2023-01-18)"
    # No title and no date: the doc_id stands for the title, the date is left out.
    assert by_text[BOE]["title"] is None and by_text[BOE]["date"] is None
    assert by_text[BOE]["citation"] == "gb/B1/undated (gb B1), p. 3"


@pytest.mark.parametrize(("stored", "shown"), [(0, 1), (1, 2), (41, 42), (None, None), (-1, None), ("3", None),
                                              (True, None), (2.0, None)])
def test_human_page_is_the_stored_page_plus_one_or_nothing(stored, shown):
    assert human_page(stored) == shown


@pytest.mark.parametrize(("hit", "expected"), [
    ({"title": "T", "source_code": "ecb", "doc_type": "A1", "date": "2022-07-21", "page": 0},
     "T (ecb A1, 2022-07-21), p. 1"),
    ({"title": "T", "source_code": "ecb", "doc_type": "A1", "date": "2022-07-21", "page": None},
     "T (ecb A1, 2022-07-21)"),
    ({"title": "T", "source_code": "ecb", "doc_type": "A1", "date": "2022-07-21"}, "T (ecb A1, 2022-07-21)"),
    ({"title": "T", "source_code": None, "doc_type": None, "date": "2022-07-21", "page": 9}, "T (2022-07-21), p. 10"),
    ({"title": "T", "doc_type": "A1", "page": 1}, "T (A1), p. 2"),
    ({"title": "", "doc_id": "d1"}, "d1"),
    ({}, "untitled document"),
])
def test_citation_leaves_out_what_is_missing(hit, expected):
    assert citation(hit) == expected


# -- arguments ---------------------------------------------------------------------------------------


async def test_filters_narrow_the_search_and_are_echoed_as_applied(backend):
    async with _client(backend) as c:
        async def ids(**arguments):
            data = (await c.call_tool("rag_search", {"query": "rates", **arguments})).data
            return sorted({h["doc_id"] for h in data["hits"]}), data["filters"]

        assert (await ids(source_codes=["ecb", "us", "ecb"])) == (
            ["ecb/A1/2022-07-21", "us/A3/2022-06-15"],
            {"source_codes": ["ecb", "us"], "doc_types": None, "languages": None, "date_from": None,
             "date_to": None})
        assert (await ids(doc_types=["E1"]))[0] == ["jp/E1/2023-01-18"]  # a type held by one chunk
        assert (await ids(languages=["ja"]))[0] == ["jp/E1/2023-01-18"]
        assert (await ids(source_codes=["ecb"], doc_types=["A3"]))[0] == []  # AND between arguments
        # Both bounds inclusive; with a bound set the undated document is left out.
        found, applied = await ids(date_from="2022-06-15", date_to="2022-07-21")
        assert found == ["ecb/A1/2022-07-21", "us/A3/2022-06-15"]
        assert (applied["date_from"], applied["date_to"]) == ("2022-06-15", "2022-07-21")
        assert "gb/B1/undated" not in (await ids(date_from="1900-01-01"))[0]
        assert "gb/B1/undated" in (await ids())[0]


async def test_limit_defaults_to_the_setting_and_is_refused_above_the_maximum(seeded):
    store, fake = seeded
    backend = EngineRagBackend(_settings(), store=store, embedder=fake)
    async with _client(backend, rag_default_limit=2, rag_max_limit=4) as c:
        assert (await c.call_tool("rag_search", {"query": "rates"})).data["count"] == 2
        assert (await c.call_tool("rag_search", {"query": "rates", "limit": "3"})).data["count"] == 3  # coerced
        assert (await c.call_tool("rag_search", {"query": "rates", "limit": 4})).data["count"] == 4
        assert await _error(c, {"query": "rates", "limit": 5}) == (
            "[limit_exceeded] limit 5 is above the maximum of 4: ask for at most 4 passages")
        for bad in (0, -3):
            assert await _error(c, {"query": "rates", "limit": bad}) == (
                "[invalid_argument] limit must be an integer between 1 and 4")


@pytest.mark.parametrize(("arguments", "starts"), [
    ({"query": "   "}, "[invalid_argument] query must be a non-blank string"),
    ({"query": "\u200b\u0301"}, "[invalid_argument] the query has no searchable character"),
    ({"query": "x", "source_codes": []}, "[invalid_argument] source_codes is an empty list"),
    ({"query": "x", "doc_types": [" "]}, "[invalid_argument] doc_types must hold non-blank strings"),
    ({"query": "x", "source_codes": "ecb"},
     "[invalid_argument] source_codes must be a list of strings, got the single string 'ecb': write ['ecb']"),
    ({"query": "x", "date_from": "2016"}, "[invalid_argument] date_from must be"),
    ({"query": "x", "date_to": "10/03/2016"}, "[invalid_argument] date_to must be"),
    ({"query": "x", "date_to": "2016-02-30"}, "[invalid_argument] date_to is not a date of the calendar"),
    ({"query": "x", "date_from": "2016-03-10T00:00:00"}, "[invalid_argument] date_from must be"),
    ({"query": "x", "date_from": "2022-07-22", "date_to": "2022-07-21"},
     "[invalid_argument] date_from (2022-07-22) is after date_to (2022-07-21)"),
])
async def test_what_the_engine_refuses_is_an_invalid_argument_with_its_message(backend, arguments, starts):
    async with _client(backend) as c:
        assert (await _error(c, arguments)).startswith(starts)


@pytest.mark.parametrize("arguments", [
    {},  # query is required
    {"query": "x" * 2001},
    {"query": "x", "source_codes": [7]},
    {"query": "x", "doc_types": ["A" * 65]},
    {"query": "x", "limit": "many"},
    {"query": "x", "sources": ["ecb"]},  # an invented argument name
    {"query": "x", "year": 2022},
])
async def test_what_the_schema_refuses_never_reaches_the_backend(backend, monkeypatch, arguments):
    monkeypatch.setattr(backend, "search", lambda **_: pytest.fail("the backend was called"))
    async with _client(backend) as c:
        with pytest.raises(ToolError):
            await c.call_tool("rag_search", arguments)


# -- unknown filter values ---------------------------------------------------------------------------


async def test_an_unknown_code_is_refused_with_the_valid_values_and_never_corrected(backend, monkeypatch):
    calls = []
    real = backend_module.search
    monkeypatch.setattr(backend_module, "search", lambda *a, **k: calls.append(k) or real(*a, **k))
    async with _client(backend) as c:
        assert await _error(c, {"query": "x", "source_codes": ["ECB"]}) == (
            "[invalid_argument] source_codes: unknown value 'ECB' (did you mean 'ecb'? values are matched "
            "exactly); valid values: " + ", ".join(SOURCE_CODES))
        assert await _error(c, {"query": "x", "source_codes": ["ecb", "fed", " us "]}) == (
            "[invalid_argument] source_codes: unknown values 'fed', ' us ' (did you mean 'us'? values are "
            "matched exactly); valid values: " + ", ".join(SOURCE_CODES))
        assert await _error(c, {"query": "x", "doc_types": ["a3"]}) == (
            "[invalid_argument] doc_types: unknown value 'a3' (did you mean 'A3'? values are matched exactly); "
            "valid values: " + ", ".join(DOC_TYPES))
        assert await _error(c, {"query": "x", "languages": ["english"]}) == (
            "[invalid_argument] languages: unknown value 'english'; valid values: en, ja")
        assert calls == []  # nothing was searched with a corrected or an unknown value
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 2
    assert len(calls) == 1


async def test_valid_values_come_from_the_collection_and_are_read_again_after_the_ttl(seeded, monkeypatch):
    store, fake = seeded
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=60), store=store, embedder=fake)
    reads, now = [], [1000.0]
    real = store.client.facet
    monkeypatch.setattr(store.client, "facet", lambda *a, **k: reads.append(a[1]) or real(*a, **k))
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: now[0])
    async with _client(backend) as c:
        await c.call_tool("rag_search", {"query": "x"})
        assert reads == []  # no filter value to check: nothing is read
        await c.call_tool("rag_search", {"query": "x", "source_codes": ["us"]})
        await c.call_tool("rag_search", {"query": "x", "doc_types": ["A1"]})
        assert reads == ["source_code", "doc_type", "language"]  # read once, kept
        now[0] += 61
        await c.call_tool("rag_search", {"query": "x", "source_codes": ["us"]})
        assert reads == ["source_code", "doc_type", "language"] * 2
        # A value that is not a string (the fixture's numeric source code) is never a valid value.
        assert "7" not in await _error(c, {"query": "x", "source_codes": ["nope"]})


@pytest.mark.parametrize("failure", [
    AttributeError("facet"),  # a client without the facet API
    UnexpectedResponse(400, "Bad Request", b"no index for facet", httpx.Headers()),  # no keyword index
    UnexpectedResponse(404, "Not Found", b"", httpx.Headers()),  # a server without the facet API
])
async def test_without_the_facet_api_the_values_are_read_by_a_full_scroll(seeded, monkeypatch, failure):
    store, fake = seeded
    backend = EngineRagBackend(_settings(), store=store, embedder=fake)

    def no_facet(*args, **kwargs):
        raise failure

    monkeypatch.setattr(store.client, "facet", no_facet)
    monkeypatch.setattr(backend_module, "SCROLL_PAGE", 2)  # several pages
    async with _client(backend) as c:
        assert (await _error(c, {"query": "x", "source_codes": ["ECB"]})).endswith(
            "valid values: " + ", ".join(SOURCE_CODES))
        assert (await c.call_tool("rag_search", {"query": "x", "doc_types": ["E1"]})).data["count"] == 1


@pytest.mark.parametrize("limits", [{"MAX_FACET_VALUES": 4}, {"MAX_SCROLL_POINTS": 2, "SCROLL_PAGE": 2}])
async def test_a_list_of_values_that_may_be_cut_is_not_used_to_refuse(seeded, monkeypatch, limits):
    store, fake = seeded
    backend = EngineRagBackend(_settings(), store=store, embedder=fake)
    for name, value in limits.items():
        monkeypatch.setattr(backend_module, name, value)
    if "MAX_SCROLL_POINTS" in limits:
        monkeypatch.delattr(type(store.client), "facet")
    async with _client(backend) as c:
        # source_code has 4 values (the cap): not checked, so an unknown value finds nothing, as in the engine.
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ECB"]})).data["count"] == 0


async def test_values_that_cannot_be_read_are_not_checked_and_the_search_still_answers(seeded, monkeypatch, caplog):
    store, fake = seeded
    backend = EngineRagBackend(_settings(), store=store, embedder=fake)

    def broken(*args, **kwargs):
        raise RuntimeError(f"http://{HOST}:6333 api-key={KEY}")

    monkeypatch.setattr(store.client, "facet", broken)
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        for _ in range(3):
            data = (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ECB"]})).data
            assert data["count"] == 0 and data["filters"]["source_codes"] == ["ECB"]
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 2
    warnings = [r.getMessage() for r in caplog.records if "filter values could not be read" in r.getMessage()]
    assert len(warnings) == 1 and "RuntimeError" in warnings[0]  # logged once
    assert not any(HOST in r.getMessage() or KEY in r.getMessage() for r in caplog.records)


# -- outages -----------------------------------------------------------------------------------------


def _response(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status, "reason", f"body from http://{HOST}:6333 key {KEY}".encode(), httpx.Headers())


OUTAGES = [
    (_response(404), "collection_missing"),
    (_response(400), "collection_mismatch"),
    (_response(500), "qdrant_error"),
    (_response(503), "qdrant_error"),
    (ResponseHandlingException(httpx.ConnectError(f"cannot connect to {HOST}")), "qdrant_unreachable"),
    (ResponseHandlingException(httpx.ReadTimeout(f"timed out reading {HOST}")), "qdrant_unreachable"),
    (QdrantAuthError("Qdrant refused the request (HTTP 401): check QDRANT_API_KEY"), "qdrant_auth"),
]


@pytest.mark.parametrize(("failure", "reason"), OUTAGES, ids=[r + "-" + type(f).__name__ for f, r in OUTAGES])
async def test_an_outage_is_rag_unavailable_with_a_reason_and_leaks_nothing(backend, monkeypatch, caplog,
                                                                             failure, reason):
    def failing(*args, **kwargs):
        raise failure

    monkeypatch.setattr(backend_module, "search", failing)
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        message = await _error(c, {"query": "what did the council decide", "source_codes": ["ecb"]})
        info = (await c.call_tool("server_info", {})).data
    assert message == f"[rag_unavailable] {reason}: {backend_module.REASONS[reason]}"
    assert info["rag"]["status"] == "unavailable" and info["rag"]["reason"] == reason
    everything = [message, json.dumps(info)] + [r.getMessage() for r in caplog.records]
    for text in everything:
        assert HOST not in text and KEY not in text and "QDRANT_API_KEY" not in text, text
    assert any(reason in r.getMessage() and type(failure).__name__ in r.getMessage()
               for r in caplog.records if r.name == "myopenfund_mcp.rag")


async def test_after_an_outage_the_collection_is_checked_again_and_the_toolset_recovers(backend, monkeypatch):
    real = backend_module.search
    monkeypatch.setattr(backend_module, "search", lambda *a, **k: (_ for _ in ()).throw(_response(404)))
    async with _client(backend) as c:
        assert (await _error(c, {"query": "x"})).startswith("[rag_unavailable] collection_missing")
        monkeypatch.setattr(backend_module, "search", real)
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"


async def test_an_unexpected_error_is_masked(backend, monkeypatch, caplog):
    def failing(*args, **kwargs):
        raise RuntimeError(f"http://{HOST}:6333 api-key={KEY}")

    monkeypatch.setattr(backend_module, "search", failing)
    caplog.set_level(logging.INFO, logger="myopenfund_mcp")
    async with _client(backend) as c:
        message = await _error(c, {"query": "x"})
    assert HOST not in message and KEY not in message and not message.startswith("[")
    assert not any(HOST in r.getMessage() or KEY in r.getMessage() for r in caplog.records)


# -- start-up ----------------------------------------------------------------------------------------


class _DownStore:
    """A store whose collection check fails until `failure` is cleared."""

    def __init__(self, store, failure):
        self._store, self.failure, self.checks = store, failure, 0

    def check_collection(self, name, dim):
        self.checks += 1
        if self.failure is not None:
            raise self.failure
        return self._store.check_collection(name, dim)

    def __getattr__(self, name):
        return getattr(self._store, name)


START_FAILURES = [
    (ResponseHandlingException(httpx.ConnectError(f"cannot connect to {HOST}")), "qdrant_unreachable"),
    (QdrantAuthError("Qdrant refused the request (HTTP 403): check QDRANT_API_KEY"), "qdrant_auth"),
    (CollectionNotFoundError(f"collection {COLLECTION!r} does not exist"), "collection_missing"),
    (CollectionMismatchError("another dimension"), "collection_mismatch"),
    (_response(502), "qdrant_error"),
    (RuntimeError(f"http://{HOST}"), "qdrant_error"),
]


@pytest.mark.parametrize(("failure", "reason"), START_FAILURES, ids=[r + "-" + type(f).__name__
                                                                      for f, r in START_FAILURES])
async def test_qdrant_down_at_start_keeps_the_server_up_and_the_toolset_recovers(seeded, caplog, failure, reason):
    store, fake = seeded
    down = _DownStore(store, failure)
    caplog.set_level(logging.DEBUG)
    backend = EngineRagBackend(_settings(), store=down, embedder=fake)
    async with _client(backend) as c:  # the server starts
        assert down.checks == 1  # checked at start, not at the first call
        info = (await c.call_tool("server_info", {})).data
        assert info["toolsets"]["rag"] == "enabled"
        assert info["rag"] == {"status": "unavailable", "reason": reason, "collection": COLLECTION, "model": MODEL,
                               "engine_version": backend_module.ENGINE_VERSION, "default_limit": 8, "max_limit": 20}
        for _ in range(2):
            assert await _error(c, {"query": "x"}) == f"[rag_unavailable] {reason}: {backend_module.REASONS[reason]}"
        assert down.checks == 3  # every call asks again, so the toolset recovers without a restart
        down.failure = None
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"
        await c.call_tool("rag_search", {"query": "x"})
        assert down.checks == 4  # once it passed, a search costs one request, not three
    logged = [r.getMessage() for r in caplog.records]
    assert sum("collection check failed" in line for line in logged) == 1  # once per change of state
    assert not any(HOST in line or KEY in line for line in logged)


@pytest.mark.parametrize(("failure", "reason"), [
    (ModelIntegrityError("onnx/model.onnx: expected sha256 84a4, found 0000"), "model_integrity"),
    (ModelDownloadError("could not download the model files into '/models'"), "model_unavailable"),
    (OSError("read-only file system"), "model_unavailable"),
])
async def test_a_model_that_cannot_be_loaded_at_start_is_never_loaded_by_a_call(seeded, monkeypatch, caplog,
                                                                                failure, reason):
    store, _ = seeded
    loads = []

    def load(engine_settings, **kwargs):
        loads.append(engine_settings)
        raise failure

    monkeypatch.setattr(backend_module, "make_embedder", load)
    caplog.set_level(logging.INFO, logger="myopenfund_mcp.rag")
    backend = EngineRagBackend(_settings(), store=store)
    async with _client(backend) as c:
        info = (await c.call_tool("server_info", {})).data
        assert (info["rag"]["status"], info["rag"]["reason"]) == ("unavailable", reason)
        assert info["rag"]["model"] == "intfloat/multilingual-e5-base"
        for _ in range(2):
            assert await _error(c, {"query": "x"}) == f"[rag_unavailable] {reason}: {backend_module.REASONS[reason]}"
    assert len(loads) == 1  # at start only: no load, hence no download, at call time
    assert any(r.levelno == logging.ERROR and type(failure).__name__ in r.getMessage() for r in caplog.records)


async def test_the_model_is_loaded_at_start_from_mof_settings_only(seeded, monkeypatch):
    """The engine's own variables in the process environment are never read."""
    store, fake = seeded
    for name, value in {"QDRANT_HOST": "elsewhere", "QDRANT_API_KEY": "other", "RAG_ENGINE_BACKEND": "torch",
                        "RAG_ENGINE_MODEL_CACHE": "/elsewhere", "RAG_ENGINE_QDRANT_TIMEOUT": "60",
                        "RAG_ENGINE_ONNX_THREADS": "64", "RAG_ENGINE_MODEL": "other/model"}.items():
        monkeypatch.setenv(name, value)
    seen = []
    monkeypatch.setattr(backend_module, "make_embedder", lambda s, **k: seen.append(("embedder", s)) or fake)
    monkeypatch.setattr(backend_module, "QdrantStore", lambda **k: seen.append(("store", k["settings"])) or store)
    settings = _settings(rag_qdrant_port=6334, rag_qdrant_https=True, rag_qdrant_timeout=7, rag_onnx_threads=3,
                         rag_default_limit=5, rag_max_limit=9)
    backend = EngineRagBackend(settings)
    assert [kind for kind, _ in seen] == ["store"]  # nothing is loaded before start()
    build_server(settings, rag_backend=backend)
    assert [kind for kind, _ in seen] == ["store", "embedder"]
    engine = seen[1][1]
    assert engine is seen[0][1]
    assert (engine.qdrant_host, engine.qdrant_port, engine.qdrant_api_key, engine.qdrant_https) == (
        HOST, 6334, KEY, True)
    assert (engine.qdrant_timeout, engine.backend, engine.model_cache, engine.onnx_threads) == (
        7, "onnx", "/models", 3)
    assert (engine.search_limit, engine.search_max_limit) == (5, 9)
    assert (engine.model_name, engine.model_dim) == ("intfloat/multilingual-e5-base", 768)


def test_the_real_store_is_built_from_the_settings_with_the_configured_timeout():
    # Port 1 on loopback: nothing listens, and nothing leaves the host.
    settings = _settings(rag_qdrant_host="127.0.0.1", rag_qdrant_port=1, rag_qdrant_timeout=3)
    backend = EngineRagBackend(settings, embedder=embedder())
    assert (backend._store.host, backend._store.port) == ("127.0.0.1", 1)
    assert backend._store.client._client._timeout == 3
    assert KEY not in repr(backend._store) and backend._dim == DIM
    build_server(settings, rag_backend=backend)  # an unreachable Qdrant does not stop the start
    assert backend.info()["reason"] == "qdrant_unreachable"


@pytest.mark.parametrize("missing", ["rag_engine", "rag_engine.search", "qdrant_client.http.exceptions"])
def test_rag_configured_without_the_extra_is_a_config_error(monkeypatch, missing):
    for name in [n for n in sys.modules if n.startswith("myopenfund_mcp.rag")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ConfigError) as caught:
        build_server(_settings())
    assert str(caught.value) == ("MOF_RAG_QDRANT_HOST is set but the [rag] extra is not installed: "
                                 "pip install 'myopenfund-mcp[rag]'")
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_rag_configured_without_the_onnx_backend_is_a_config_error(monkeypatch, seeded):
    monkeypatch.setattr(backend_module, "QdrantStore", lambda **_: seeded[0])

    def no_onnx(engine_settings, **kwargs):
        raise ImportError("the onnx backend needs rag-engine[onnx]")

    monkeypatch.setattr(backend_module, "make_embedder", no_onnx)
    with pytest.raises(ConfigError, match=r"\[rag\] extra is not installed"):
        build_server(_settings())


async def test_without_rag_settings_the_toolset_is_not_mounted():
    async with Client(build_server(Settings())) as c:
        names = [t.name for t in await c.list_tools()]
        info = (await c.call_tool("server_info", {})).data
    assert names == ["server_info"]
    assert info["toolsets"]["rag"] == "not configured" and "rag" not in info


async def test_server_info_lists_the_rag_toolset_and_nothing_secret(backend):
    async with _client(backend, rag_default_limit=3, rag_max_limit=7) as c:
        info = (await c.call_tool("server_info", {})).data
    assert info["toolsets"] == {"vault": "not configured", "rag": "enabled", "companies": "not configured",
                                "datasets": "not configured"}
    assert info["rag"] == {"status": "ready", "collection": COLLECTION, "model": MODEL,
                           "engine_version": backend_module.ENGINE_VERSION, "default_limit": 3, "max_limit": 7}
    assert HOST not in json.dumps(info) and KEY not in json.dumps(info)


# -- result cap, audit, threads ----------------------------------------------------------------------


class _CannedBackend:
    """A backend returning fixed hits: what a RagBackend owes the toolset, and nothing of the engine."""

    def __init__(self, texts):
        self.texts, self.threads = texts, []

    def start(self):
        pass

    def info(self):
        return {"status": "ready", "collection": "canned", "model": "none", "engine_version": "0"}

    def search(self, *, query, filters, limit, max_limit):
        self.threads.append(threading.current_thread())
        hits = [{"score": 0.123456 - n / 100, "text": text, "doc_id": f"d{n}", "title": f"T{n}", "page": n,
                 "date": "2022-01-01", "source_code": "ecb", "doc_type": "A1", "language": "en",
                 "chunk_number": n, "corpus": "central-bank", "point_id": 2**62 + n}
                for n, text in enumerate(self.texts)]
        return {name: None for name in ("source_codes", "doc_types", "languages", "date_from", "date_to")}, hits[:limit]


async def test_a_result_over_the_cap_is_refused_naming_the_largest_limit_that_fits():
    backend = _CannedBackend(["x" * 400] * 8)
    async with _client(backend, max_result_chars=2000) as c:
        message = await _error(c, {"query": "q"})
        assert message.startswith("[result_too_large] result is ") and "(cap 2000)" in message
        fits = int(message.rsplit("limit=", 1)[1].split()[0])
        assert message.endswith(f"ask again with limit={fits} or less") and 1 <= fits < 8
        assert (await c.call_tool("rag_search", {"query": "q", "limit": fits})).data["count"] == fits  # it fits
        assert (await _error(c, {"query": "q", "limit": fits + 1})).startswith("[result_too_large]")  # the largest


async def test_a_single_passage_over_the_cap_is_refused_not_cut():
    async with _client(_CannedBackend(["x" * 5000]), max_result_chars=1000) as c:
        message = await _error(c, {"query": "q"})
    assert message.startswith("[result_too_large]") and "even the first passage does not fit" in message


async def test_scores_are_rounded_and_point_ids_never_leave_the_server():
    async with _client(_CannedBackend(["a", "b"])) as c:
        result = await c.call_tool("rag_search", {"query": "q"})
    assert [h["score"] for h in result.data["hits"]] == [0.1235, 0.1135]
    assert "point_id" not in result.content[0].text and str(2**62) not in result.content[0].text
    assert "corpus" not in result.data["hits"][0]


async def test_audit_logs_one_line_per_call_and_no_log_holds_the_query(backend, caplog):
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        await c.call_tool("rag_search", {"query": "secret-term about rates", "source_codes": ["ecb"]})
        await _error(c, {"query": "secret-term about rates", "source_codes": ["fed"]})
        await _error(c, {"query": "secret-term about rates", "limit": 99})
    lines = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.audit"]
    assert len(lines) == 3
    assert "tool=rag_search client=local outcome=ok" in lines[0]
    assert all("tool=rag_search client=local outcome=error" in line for line in lines[1:])
    assert not any("secret-term" in r.getMessage() for r in caplog.records)
    assert not any("ecb" in line or "fed" in line for line in lines)  # arguments are never logged


async def test_the_blocking_search_runs_off_the_event_loop():
    backend = _CannedBackend(["a"])
    async with _client(backend) as c:
        await c.call_tool("rag_search", {"query": "q"})
    assert backend.threads and backend.threads[0] is not threading.main_thread()


def test_the_readme_documents_the_codes_every_reason_and_every_argument():
    from pathlib import Path

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    for needle in ("`rag_unavailable`", "`limit_exceeded`", "`rag_search`", *(f"`{r}`" for r in backend_module.REASONS),
                   "`source_codes`", "`doc_types`", "`languages`", "`date_from`, `date_to`", "`limit`", "`citation`",
                   "the first page is 1"):
        assert needle in readme, needle
