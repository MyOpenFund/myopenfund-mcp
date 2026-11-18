"""The rag toolset on a fake engine: rag-engine's own store in qdrant-client's in-process mode, its
FakeEmbedder (no model), and a monkeypatched `search` where an outage is needed."""
import asyncio
import json
import logging
import socket
import sys
import threading
import time
import unicodedata
from pathlib import Path

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
from myopenfund_mcp.rag.tools import build_rag_toolset, citation, human_page
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio

HOST, KEY = "qdrant.secret-host.example", "k-hunter2"
DATES_SENTENCE = ("Date precision varies by bank (some documents carry only the year or the month): if a date "
                  "range returns nothing, retry with whole-year bounds or without dates.")
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


def _settled(backend) -> None:
    """Wait for the read of the filter values that is running, if any (the background scroll)."""
    with backend._read_done:
        assert backend._read_done.wait_for(lambda: not backend._reading, timeout=10)


@pytest.fixture
def clock(monkeypatch):
    """The clock of the backend's caches (failed check, filter values), moved by hand: `clock[0] += 61`."""
    now = [1000.0]
    monkeypatch.setattr(backend_module, "monotonic", lambda: now[0])
    return now


# -- surface -----------------------------------------------------------------------------------------


async def test_the_tool_is_namespaced_read_only_and_described_for_a_model(backend):
    async with _client(backend) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert set(tools) == {"rag_search", "server_info"}
    tool = tools["rag_search"]
    assert tool.annotations.read_only_hint is True
    for needle in ("YYYY-MM-DD", "inclusive", "does not write an answer", "counted from 1", "citation",
                   "PASSAGES", "never corrected", "JSON arrays", "server_info", "rag.source_codes",
                   "rag.doc_types", "rag.languages", "never send an empty array",
                   DATES_SENTENCE, "limit: number of passages, from 1 to 20; omitted = 8."):
        assert needle in tool.description, needle
    # The default text is corpus-neutral: no bank, no code and no meaning of a code that could go stale.
    for absent in ("ecb", "ECB", "European", "Federal", "England", "Japan", "A1", "A2", "A3", "B1", "E1", "F1",
                   "press", "minutes", "About this collection"):
        assert absent not in tool.description, absent
    assert len(tool.description) < 1800
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
    for name in ("source_codes", "doc_types", "languages"):
        assert f"server_info's rag.{name}" in schema["properties"][name]["description"]


async def test_the_description_states_the_deployments_limits_and_appends_its_notes(backend):
    notes = 'Corpus: test documents. doc_types: "A1" rate-decision press release.'
    async with _client(backend, rag_default_limit=3, rag_max_limit=7, rag_description_notes=notes) as c:
        description = {t.name: t for t in await c.list_tools()}["rag_search"].description
    assert "limit: number of passages, from 1 to 7; omitted = 3." in description
    assert description.endswith("\n\nAbout this collection: " + notes)


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


ARRAY = "must be a JSON array of strings, e.g. "
EMPTY = (' is an empty array, which could mean "match nothing" as well as "no restriction": omit the argument '
         "for no restriction")
DATE = ' must be "YYYY-MM-DD", e.g. "2016-03-10"'
UNSEARCHABLE = ("query has no searchable character: it must hold at least one letter, number, symbol or "
                "punctuation mark")


@pytest.mark.parametrize(("arguments", "message"), [
    ({"query": "   "}, "query must be a non-blank string: a question or a topic"),
    ({"query": "\u200b\u0301"}, UNSEARCHABLE),
    ({"query": "\u00a8"}, UNSEARCHABLE),  # a spacing accent: nothing is left of it once normalised
    # A string where an array is expected: the frequent mistake of a small model.
    ({"query": "x", "source_codes": "ecb"}, f'source_codes {ARRAY}["ecb"]; omit it for no restriction'),
    # The array written as a string: the example is the array meant, never ['["ecb"]'].
    ({"query": "x", "source_codes": '["ecb"]'}, f'source_codes {ARRAY}["ecb"]; omit it for no restriction'),
    ({"query": "x", "doc_types": '["A1", "A3"]'}, f'doc_types {ARRAY}["A1", "A3"]; omit it for no restriction'),
    ({"query": "x", "doc_types": "['A1']"}, f'doc_types {ARRAY}["A1"]; omit it for no restriction'),
    ({"query": "x", "source_codes": "ecb,us"}, f'source_codes {ARRAY}["ecb", "us"]; omit it for no restriction'),
    ({"query": "x", "languages": "a b c d e"}, f'languages {ARRAY}["a", "b", "c"]; omit it for no restriction'),
    ({"query": "x", "languages": " "}, f'languages {ARRAY}["code"]; omit it for no restriction'),
    ({"query": "x", "languages": "[]"}, f'languages {ARRAY}["code"]; omit it for no restriction'),
    ({"query": "x", "source_codes": []}, "source_codes" + EMPTY),
    ({"query": "x", "languages": []}, "languages" + EMPTY),
    ({"query": "x", "doc_types": [" "]}, "doc_types must hold non-blank strings"),
    ({"query": "x", "date_from": "2016"}, "date_from" + DATE),
    ({"query": "x", "date_to": "10/03/2016"}, "date_to" + DATE),
    ({"query": "x", "date_from": "2016-03-10T00:00:00"}, "date_from" + DATE),
    ({"query": "x", "date_to": "None"}, "date_to" + DATE),
    ({"query": "x", "date_to": "2016-02-30"}, 'date_to is not a date of the calendar: "2016-02-30"'),
    ({"query": "x", "date_from": "2022-07-22", "date_to": "2022-07-21"},
     "date_from (2022-07-22) is after date_to (2022-07-21)"),
])
async def test_a_bad_argument_is_refused_by_the_tool_in_json_terms(backend, monkeypatch, arguments, message):
    """The tool owns these messages: nothing of the engine's wording (Python lists, None, datetime.date)."""
    monkeypatch.setattr(backend, "search", lambda **_: pytest.fail("the backend was called"))
    async with _client(backend) as c:
        text = await _error(c, arguments)
    assert text == "[invalid_argument] " + message
    for python in ("None", "datetime", "['", "']", "got "):
        assert python not in text.replace('"None"', ""), python


def test_what_only_the_engine_refuses_is_still_an_invalid_argument(backend):
    """The second line: an argument the tool would let through and rag-engine refuses keeps its code."""
    with pytest.raises(ToolError, match=r"^\[invalid_argument\] source_codes holds a string that cannot be encoded"):
        backend.search(query="x", filters={"source_codes": ["\ud800"]}, limit=1, max_limit=5)
    with pytest.raises(ToolError, match=r"^\[invalid_argument\] query must be a non-blank string"):
        backend.search(query=" ", filters={}, limit=1, max_limit=5)


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

VALID_SOURCES = ", ".join(f'"{code}"' for code in SOURCE_CODES)
VALID_TYPES = ", ".join(f'"{code}"' for code in DOC_TYPES)


async def test_an_unknown_code_is_refused_with_the_valid_values_and_never_corrected(backend, monkeypatch):
    calls = []
    real = backend_module.search
    monkeypatch.setattr(backend_module, "search", lambda *a, **k: calls.append(k) or real(*a, **k))
    async with _client(backend) as c:
        assert await _error(c, {"query": "x", "source_codes": ["ECB"]}) == (
            '[invalid_argument] source_codes: unknown value "ECB" (did you mean "ecb"? values are matched '
            "exactly); valid values: " + VALID_SOURCES)
        assert await _error(c, {"query": "x", "source_codes": ["ecb", "fed", " us "]}) == (
            '[invalid_argument] source_codes: unknown values "fed", " us " (did you mean "us"? values are '
            "matched exactly); valid values: " + VALID_SOURCES)
        assert await _error(c, {"query": "x", "doc_types": ["a3"]}) == (
            '[invalid_argument] doc_types: unknown value "a3" (did you mean "A3"? values are matched exactly); '
            "valid values: " + VALID_TYPES)
        assert await _error(c, {"query": "x", "languages": ["english"]}) == (
            '[invalid_argument] languages: unknown value "english"; valid values: "en", "ja"')
        assert calls == []  # nothing was searched with a corrected or an unknown value
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 2
    assert len(calls) == 1


async def test_valid_values_come_from_the_collection_and_are_read_again_after_the_ttl(seeded, monkeypatch, clock):
    store, fake = seeded
    reads = []
    real = store.client.facet
    monkeypatch.setattr(store.client, "facet", lambda *a, **k: reads.append(a[1]) or real(*a, **k))
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=60), store=store, embedder=fake)
    async with _client(backend) as c:
        assert reads == ["source_code", "doc_type", "language"]  # read at start, once the collection is checked
        await c.call_tool("rag_search", {"query": "x"})
        await c.call_tool("rag_search", {"query": "x", "source_codes": ["us"]})
        await c.call_tool("rag_search", {"query": "x", "doc_types": ["A1"]})
        assert len(reads) == 3  # kept
        clock[0] += 61
        await c.call_tool("rag_search", {"query": "x"})
        assert len(reads) == 3  # no filter value to check: nothing is read
        await c.call_tool("rag_search", {"query": "x", "source_codes": ["us"]})
        assert reads == ["source_code", "doc_type", "language"] * 2
        # A value that is not a string (the fixture's numeric source code) is never a valid value.
        assert "7" not in await _error(c, {"query": "x", "source_codes": ["nope"]})


class _Codes:
    """A Qdrant client whose stored codes the test changes: facet answers from `values`, or fails."""

    def __init__(self, store, **values):
        self._client, self.values, self.reads, self.failure = store.client, values, 0, None

    def facet(self, collection, field, **kwargs):
        from types import SimpleNamespace as NS

        self.reads += field == "source_code"
        if self.failure is not None:
            raise self.failure
        return NS(hits=[NS(value=value) for value in self.values.get(field, [])])

    def __getattr__(self, name):
        return getattr(self._client, name)


class _Store:
    """rag-engine's store with another client, and a collection check the test can fail."""

    def __init__(self, store, client):
        self._store, self.client, self.failure, self.checks = store, client, None, 0

    def check_collection(self, name, dim):
        self.checks += 1
        if self.failure is not None:
            raise self.failure
        return self._store.check_collection(name, dim)

    def __getattr__(self, name):
        return getattr(self._store, name)


async def test_a_code_ingested_after_the_last_read_is_accepted_at_once(seeded, clock):
    """Before a value is refused the codes are read once more: a new code is not refused for a whole TTL."""
    store, fake = seeded
    codes = _Codes(store, source_code=["ecb"], doc_type=["A1"], language=["en"])
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=300, rag_codes_retry_s=5), store=_Store(store, codes),
                               embedder=fake)
    async with _client(backend) as c:
        assert codes.reads == 1
        clock[0] += 10
        codes.values["source_code"] = ["ecb", "us"]  # documents of a new source arrive
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["us"]})).data["count"] == 1
        assert codes.reads == 2  # read again before refusing, and "us" was there
        # Still unknown after the second read: refused, with the values just read.
        clock[0] += 10
        assert (await _error(c, {"query": "x", "source_codes": ["fed"]})).endswith('valid values: "ecb", "us"')
        assert codes.reads == 3
        # Not more often than every rag_codes_retry_s seconds, however many wrong codes are sent.
        clock[0] += 4
        for _ in range(5):
            await _error(c, {"query": "x", "source_codes": ["fed"]})
        assert codes.reads == 3
        clock[0] += 2
        await _error(c, {"query": "x", "source_codes": ["fed"]})
        assert codes.reads == 4
        assert (await c.call_tool("server_info", {})).data["rag"]["source_codes"] == ["ecb", "us"]


async def test_the_values_are_dropped_when_the_collection_comes_back(seeded, clock):
    """A collection recreated during an outage: its old codes must not refuse the new ones."""
    store, fake = seeded
    codes = _Codes(store, source_code=["old"], doc_type=["A1"], language=["en"])
    down = _Store(store, codes)
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=3600, rag_codes_retry_s=3600, rag_check_retry_s=5),
                               store=down, embedder=fake)
    async with _client(backend) as c:
        assert (await _error(c, {"query": "x", "source_codes": ["ecb"]})).endswith('valid values: "old"')
        down.failure = CollectionNotFoundError("gone")
        backend._failed("collection_missing", down.failure, "search failed")  # what a failed search records
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "unavailable"
        down.failure, codes.values["source_code"] = None, ["ecb", "us"]
        clock[0] += 6
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 2
        assert (await c.call_tool("server_info", {})).data["rag"]["source_codes"] == ["ecb", "us"]


async def test_an_empty_list_of_values_refuses_nothing(seeded):
    """An empty collection (or a field no point carries): no check, and never "valid values: " with nothing."""
    store, fake = seeded
    codes = _Codes(store, source_code=[], doc_type=["A1"], language=["en"])
    backend = EngineRagBackend(_settings(), store=_Store(store, codes), embedder=fake)
    async with _client(backend) as c:
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ECB"]})).data["count"] == 0
        assert (await _error(c, {"query": "x", "source_codes": ["ECB"], "doc_types": ["a1"]})).startswith(
            '[invalid_argument] doc_types: unknown value "a1"')
        assert (await c.call_tool("server_info", {})).data["rag"]["source_codes"] == []


async def test_valid_values_are_quoted_as_json_whatever_they_hold(seeded):
    store, fake = seeded
    codes = _Codes(store, source_code=['é b/"c\n,', "ecb"], doc_type=["A1"], language=["en"])
    backend = EngineRagBackend(_settings(), store=_Store(store, codes), embedder=fake)
    async with _client(backend) as c:
        assert await _error(c, {"query": "x", "source_codes": ['É B/"C\n,']}) == (
            '[invalid_argument] source_codes: unknown value "É B/\\"C\\n," (did you mean "é b/\\"c\\n,"? values are '
            'matched exactly); valid values: "ecb", "é b/\\"c\\n,"')


async def test_a_failed_read_of_the_values_is_tried_again_soon_not_after_the_ttl(seeded, clock, caplog):
    store, fake = seeded
    codes = _Codes(store, source_code=SOURCE_CODES, doc_type=DOC_TYPES, language=["en", "ja"])
    codes.failure = RuntimeError(f"http://{HOST}:6333 api-key={KEY}")
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=3600, rag_codes_retry_s=5), store=_Store(store, codes),
                               embedder=fake)
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        assert codes.reads == 1  # at start: failed
        for _ in range(3):  # not checked meanwhile, and not read again at every call
            data = (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ECB"]})).data
            assert data["count"] == 0 and data["filters"]["source_codes"] == ["ECB"]
        assert codes.reads == 1
        assert (await c.call_tool("server_info", {})).data["rag"]["source_codes"] is None
        codes.failure = None
        clock[0] += 6  # 6 s later, not 3600: the hint is back
        assert (await _error(c, {"query": "x", "source_codes": ["ECB"]})).startswith(
            '[invalid_argument] source_codes: unknown value "ECB" (did you mean "ecb"?')
        assert codes.reads == 2
    warnings = [r.getMessage() for r in caplog.records if "filter values could not be read" in r.getMessage()]
    assert len(warnings) == 1 and "RuntimeError" in warnings[0]  # logged once
    assert not any(HOST in r.getMessage() or KEY in r.getMessage() for r in caplog.records)


# -- unknown filter values: a collection without the facet API or the keyword indexes -----------------


class _Scrolled:
    """A client without facets: counts the scrolls, and can hold one until the test lets it go."""

    def __init__(self, store, failure=None):
        self._client, self._failure = store.client, failure or AttributeError("facet")
        self.scrolls, self.fields, self.hold, self.release = 0, [], False, threading.Event()

    def facet(self, *args, **kwargs):
        raise self._failure

    def scroll(self, collection, *, offset=None, with_payload=None, **kwargs):
        if offset is None:
            self.scrolls += 1
            self.fields.append(with_payload)
            if self.hold:
                assert self.release.wait(10)
        return self._client.scroll(collection, offset=offset, with_payload=with_payload, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


@pytest.mark.parametrize("failure", [
    AttributeError("facet"),  # a client without the facet API
    UnexpectedResponse(400, "Bad Request", b"no index for facet", httpx.Headers()),  # no keyword index
    UnexpectedResponse(404, "Not Found", b"", httpx.Headers()),  # a server without the facet API
])
async def test_without_the_facet_api_the_values_are_read_by_one_scroll_of_the_three_fields(seeded, monkeypatch,
                                                                                           failure):
    store, fake = seeded
    client = _Scrolled(store, failure)
    monkeypatch.setattr(backend_module, "SCROLL_PAGE", 2)  # several pages
    backend = EngineRagBackend(_settings(), store=_Store(store, client), embedder=fake)
    async with _client(backend) as c:
        _settled(backend)
        assert (await _error(c, {"query": "x", "source_codes": ["ECB"]})).endswith("valid values: " + VALID_SOURCES)
        assert (await _error(c, {"query": "x", "doc_types": ["a1"]})).endswith("valid values: " + VALID_TYPES)
        assert (await c.call_tool("rag_search", {"query": "x", "doc_types": ["E1"]})).data["count"] == 1
        info = (await c.call_tool("server_info", {})).data["rag"]
    assert (info["source_codes"], info["doc_types"], info["languages"]) == (SOURCE_CODES, DOC_TYPES, ["en", "ja"])
    # One pass over the collection for the three fields, not one pass per field.
    assert client.scrolls == 1 and client.fields == [["source_code", "doc_type", "language"]]


async def test_the_scroll_runs_in_the_background_and_no_search_waits_for_it(seeded, clock):
    store, fake = seeded
    client = _Scrolled(store)
    client.hold = True
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=60, rag_codes_retry_s=5), store=_Store(store, client),
                               embedder=fake)
    async with _client(backend) as c:  # the start does not wait for the scroll either
        assert client.scrolls == 1  # started at start-up, once the collection is checked
        # No value is known yet: a filtered search is served at once, unchecked.
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 2
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ECB"]})).data["count"] == 0
        assert (await c.call_tool("server_info", {})).data["rag"]["source_codes"] is None
        client.release.set()
        _settled(backend)
        client.release.clear()
        clock[0] += 1
        assert (await _error(c, {"query": "x", "source_codes": ["ECB"]})).endswith("valid values: " + VALID_SOURCES)
        assert client.scrolls == 1
        # At the end of the TTL the values are read again in the background: the searches made meanwhile
        # are served, and checked against the values already known.
        clock[0] += 61
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 2
        assert client.scrolls == 2 and backend._reading  # running, and the call above did not wait for it
        assert (await c.call_tool("rag_search", {"query": "x", "doc_types": ["E1"]})).data["count"] == 1
        assert client.scrolls == 2  # one at a time
        client.release.set()
        _settled(backend)
        assert (await _error(c, {"query": "x", "source_codes": ["ECB"]})).endswith("valid values: " + VALID_SOURCES)


async def test_a_refusal_waits_for_the_scroll_within_the_call_budget(seeded, clock):
    """Before a refusal the values are read once more: with a scroll, the call waits for it, but never
    beyond its own time budget."""
    store, fake = seeded
    client = _Scrolled(store)
    backend = EngineRagBackend(_settings(rag_call_timeout_s=0.4, rag_codes_retry_s=5), store=_Store(store, client),
                               embedder=fake)
    async with _client(backend, rag_call_timeout_s=0.4) as c:
        _settled(backend)
        clock[0] += 6
        client.hold = True
        started = time.monotonic()
        text = await _error(c, {"query": "x", "source_codes": ["ECB"]})
        assert time.monotonic() - started < 5
        assert client.scrolls == 2
        # The scroll did not end in time: refused against the values known, or the call's budget ended.
        assert text.endswith("valid values: " + VALID_SOURCES) or text.startswith("[rag_unavailable] timeout")
        client.release.set()
        _settled(backend)


@pytest.mark.parametrize("limits", [{"MAX_FACET_VALUES": 4}, {"MAX_SCROLL_POINTS": 2, "SCROLL_PAGE": 2}])
async def test_a_list_of_values_that_may_be_cut_is_not_used_to_refuse(seeded, monkeypatch, limits):
    store, fake = seeded
    for name, value in limits.items():
        monkeypatch.setattr(backend_module, name, value)
    client = _Scrolled(store) if "MAX_SCROLL_POINTS" in limits else store.client
    backend = EngineRagBackend(_settings(), store=_Store(store, client), embedder=fake)
    async with _client(backend) as c:
        _settled(backend)
        # source_code has 4 values (the cap): not checked, so an unknown value finds nothing, as in the engine.
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ECB"]})).data["count"] == 0
        assert (await c.call_tool("server_info", {})).data["rag"]["source_codes"] is None


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


QUERY = "secret-term what did the council decide"


@pytest.mark.parametrize(("failure", "reason"), OUTAGES, ids=[r + "-" + type(f).__name__ for f, r in OUTAGES])
async def test_an_outage_is_rag_unavailable_with_a_reason_and_leaks_nothing(backend, monkeypatch, caplog,
                                                                             failure, reason):
    def failing(*args, **kwargs):
        raise failure

    monkeypatch.setattr(backend_module, "search", failing)
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        message = await _error(c, {"query": QUERY, "source_codes": ["ecb"]})
        again = await _error(c, {"query": QUERY})  # answered from the recorded failure
        info = (await c.call_tool("server_info", {})).data
    assert message == again == f"[rag_unavailable] {reason}: {backend_module.REASONS[reason]}"
    assert info["rag"]["status"] == "unavailable" and info["rag"]["reason"] == reason
    everything = [message, json.dumps(info)] + [r.getMessage() for r in caplog.records]
    for text in everything:
        assert HOST not in text and KEY not in text and "QDRANT_API_KEY" not in text, text
        assert "secret-term" not in text and "council" not in text, text  # nor the query, on this path either
    assert any(reason in r.getMessage() and type(failure).__name__ in r.getMessage()
               for r in caplog.records if r.name == "myopenfund_mcp.rag")


async def test_after_an_outage_the_collection_is_checked_again_and_the_toolset_recovers(backend, monkeypatch, clock):
    real = backend_module.search
    monkeypatch.setattr(backend_module, "search", lambda *a, **k: (_ for _ in ()).throw(_response(404)))
    async with _client(backend) as c:
        assert (await _error(c, {"query": "x"})).startswith("[rag_unavailable] collection_missing")
        monkeypatch.setattr(backend_module, "search", real)
        clock[0] += 6  # past rag_check_retry_s
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"


async def test_a_failed_check_is_remembered_a_few_seconds_and_not_run_by_every_call(seeded, clock):
    """With Qdrant hanging, each check costs a whole timeout: the calls made meanwhile answer at once."""
    store, fake = seeded
    down = _Store(store, store.client)
    down.failure = ResponseHandlingException(httpx.ReadTimeout("timed out"))
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    expected = "[rag_unavailable] qdrant_unreachable: " + backend_module.REASONS["qdrant_unreachable"]
    async with _client(backend) as c:
        assert down.checks == 1  # at start
        for _ in range(20):
            assert await _error(c, {"query": "x"}) == expected
        assert down.checks == 1  # 20 calls, no request
        clock[0] += 4.9
        await _error(c, {"query": "x"})
        assert down.checks == 1
        clock[0] += 0.2  # 5.1 s after the failure: one call checks again
        await _error(c, {"query": "x"})
        assert down.checks == 2
        await _error(c, {"query": "x"})
        assert down.checks == 2  # and that failure is remembered in its turn
        down.failure = None
        clock[0] += 5.1
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        assert down.checks == 3


async def test_calls_arriving_while_a_check_runs_are_answered_at_once(seeded, clock):
    store, fake = seeded
    entered, release = threading.Event(), threading.Event()

    class Hanging(_Store):
        def check_collection(self, name, dim):
            if self.failure is not None and self.checks >= 1:
                self.checks += 1
                entered.set()
                assert release.wait(10)
                raise self.failure
            return super().check_collection(name, dim)

    down = Hanging(store, store.client)
    down.failure = ResponseHandlingException(httpx.ReadTimeout("timed out"))
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    async with _client(backend) as c:
        clock[0] += 6  # the failure of the start is forgotten: the next call checks
        first = asyncio.create_task(_error(c, {"query": "x"}))
        assert await asyncio.to_thread(entered.wait, 10)
        started = time.monotonic()
        for _ in range(10):  # the check is running: these do not wait for it, and do not run another
            assert (await _error(c, {"query": "x"})).startswith("[rag_unavailable] qdrant_unreachable")
        assert time.monotonic() - started < 2 and down.checks == 2
        release.set()
        assert (await first).startswith("[rag_unavailable] qdrant_unreachable")


async def test_a_filtered_call_does_not_pay_for_two_timeouts(seeded, monkeypatch, clock):
    """The read of the filter values failed for an outage: the search is not tried after it."""
    store, fake = seeded
    codes = _Codes(store, source_code=SOURCE_CODES, doc_type=DOC_TYPES, language=["en", "ja"])
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=60), store=_Store(store, codes), embedder=fake)
    searched = []
    monkeypatch.setattr(backend_module, "search", lambda *a, **k: searched.append(a) or [])
    async with _client(backend) as c:
        clock[0] += 61  # the values are due
        codes.failure = ResponseHandlingException(httpx.ReadTimeout(f"timed out reading {HOST}"))
        assert await _error(c, {"query": "x", "source_codes": ["ecb"]}) == (
            "[rag_unavailable] qdrant_unreachable: " + backend_module.REASONS["qdrant_unreachable"])
        assert searched == [] and codes.reads == 2
        # ... and the outage is recorded: the next calls ask nothing.
        await _error(c, {"query": "x", "source_codes": ["ecb"]})
        await _error(c, {"query": "x"})
        assert searched == [] and codes.reads == 2
        assert (await c.call_tool("server_info", {})).data["rag"]["reason"] == "qdrant_unreachable"


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
async def test_qdrant_down_at_start_keeps_the_server_up_and_the_toolset_recovers(seeded, caplog, clock, failure,
                                                                                 reason):
    store, fake = seeded
    down = _DownStore(store, failure)
    caplog.set_level(logging.DEBUG)
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    async with _client(backend) as c:  # the server starts
        assert down.checks == 1  # checked at start, not at the first call
        info = (await c.call_tool("server_info", {})).data
        assert info["toolsets"]["rag"] == "enabled"
        assert info["rag"] == {"status": "unavailable", "reason": reason, "collection": COLLECTION, "model": MODEL,
                               "engine_version": backend_module.ENGINE_VERSION, "source_codes": None,
                               "doc_types": None, "languages": None, "default_limit": 8, "max_limit": 20}
        for _ in range(2):
            clock[0] += 6  # past rag_check_retry_s: the call asks again, so the toolset recovers without a restart
            assert await _error(c, {"query": "x"}) == f"[rag_unavailable] {reason}: {backend_module.REASONS[reason]}"
        assert down.checks == 3
        down.failure = None
        clock[0] += 6
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        info = (await c.call_tool("server_info", {})).data["rag"]
        assert info["status"] == "ready" and info["source_codes"] is None  # read by the first filtered call
        await c.call_tool("rag_search", {"query": "x"})
        assert down.checks == 4  # once it passed, a search costs one request, not three
        assert (await _error(c, {"query": "x", "source_codes": ["ECB"]})).endswith("valid values: " + VALID_SOURCES)
        assert (await c.call_tool("server_info", {})).data["rag"]["source_codes"] == SOURCE_CODES
    logged = [r.getMessage() for r in caplog.records]
    assert sum("collection check failed" in line for line in logged) == 1  # once per change of state
    assert not any(HOST in line or KEY in line for line in logged)


ENGINE_TEXT = ("could not download the model files into '/models': set RAG_ENGINE_MODEL_CACHE to a directory "
               "already filled with these files")
# Written out here, not read from the module: the client must get these words and no others.
MODEL_MESSAGES = {
    "model_integrity": "[rag_unavailable] model_integrity: an embedding model file is not the pinned one; "
                       "the deployment must be fixed",
    "model_unavailable": "[rag_unavailable] model_unavailable: the embedding model could not be loaded when the "
                         "server started; the deployment must be fixed",
}


@pytest.mark.parametrize(("failure", "reason"), [
    (ModelIntegrityError("onnx/model.onnx: expected sha256 84a4, found 0000"), "model_integrity"),
    (ModelDownloadError(ENGINE_TEXT), "model_unavailable"),
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
            message = await _error(c, {"query": "x"})
            # The fixed text of the reason: nothing of the engine's own message reaches the client.
            assert message == MODEL_MESSAGES[reason]
            assert str(failure) not in message and "sha256" not in message and "/models" not in message
    assert len(loads) == 1  # at start only: no load, hence no download, at call time
    errors = [r.getMessage() for r in caplog.records
              if r.levelno == logging.ERROR and r.name == "myopenfund_mcp.rag"]
    assert len(errors) == 1 and type(failure).__name__ in errors[0]
    # The operator's log names the variable of THIS server; the engine's own one is never read here.
    assert "MOF_RAG_MODEL_CACHE" in errors[0] and "RAG_ENGINE_MODEL_CACHE" not in errors[0]


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


def test_the_real_store_is_built_from_the_settings_and_gives_up_after_the_configured_timeout():
    """A listener that accepts and never answers, as a frozen Qdrant does: the request ends at the timeout."""
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(5)
        port = silent.getsockname()[1]
        settings = _settings(rag_qdrant_host="127.0.0.1", rag_qdrant_port=port, rag_qdrant_timeout=1)
        backend = EngineRagBackend(settings, embedder=embedder())
        assert (backend._store.host, backend._store.port) == ("127.0.0.1", port)
        assert KEY not in repr(backend._store) and backend._dim == DIM
        started = time.monotonic()
        build_server(settings, rag_backend=backend)  # a Qdrant that does not answer does not stop the start
        assert 0.9 <= time.monotonic() - started < 5
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

    def no_onnx(engine_settings, **kwargs):  # as rag-engine raises it
        raise ImportError("the onnx backend needs rag-engine[onnx]") from ModuleNotFoundError(
            "No module named 'fastembed'", name="fastembed")

    monkeypatch.setattr(backend_module, "make_embedder", no_onnx)
    with pytest.raises(ConfigError, match=r"\[rag\] extra is not installed"):
        build_server(_settings())


def test_an_import_error_that_is_not_a_missing_module_is_reported_as_such(monkeypatch, seeded, caplog):
    """A broken install (a shared library that cannot be loaded, a name gone from a dependency) is not
    "the extra is not installed": the operator would reinstall it in vain."""
    monkeypatch.setattr(backend_module, "QdrantStore", lambda **_: seeded[0])

    def broken(engine_settings, **kwargs):
        raise ImportError("libgomp.so.1: cannot open shared object file: No such file or directory")

    monkeypatch.setattr(backend_module, "make_embedder", broken)
    caplog.set_level(logging.INFO, logger="myopenfund_mcp")
    with pytest.raises(ConfigError) as caught:
        build_server(_settings())
    assert "is not installed" not in str(caught.value) and "could not be imported" in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    logged = [r for r in caplog.records if r.levelno == logging.ERROR and r.name.startswith("myopenfund_mcp")]
    assert len(logged) == 1 and "libgomp.so.1" in logged[0].getMessage() and logged[0].exc_info is not None


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
                           "engine_version": backend_module.ENGINE_VERSION, "source_codes": SOURCE_CODES,
                           "doc_types": DOC_TYPES, "languages": ["en", "ja"], "default_limit": 3, "max_limit": 7}
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

    def search(self, *, query, filters, limit, max_limit, deadline=None):
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


async def test_the_blocking_search_runs_off_the_event_loop_in_a_thread_of_the_toolset():
    backend = _CannedBackend(["a"])
    async with _client(backend) as c:
        await c.call_tool("rag_search", {"query": "q"})
    assert backend.threads and backend.threads[0] is not threading.main_thread()
    assert backend.threads[0].name.startswith("mof-rag")  # not one of the pool the other toolsets share


# -- time budget and concurrency ---------------------------------------------------------------------

# Written out here, not read from the module: what a client is told.
TIMEOUT = ("[rag_unavailable] timeout: the search did not finish within the time allowed for one call; "
           "retry later")
BUSY = ("[rag_unavailable] busy: too many searches are running and none finished within the time allowed for "
        "one call; retry later")


class _SlowBackend(_CannedBackend):
    """A backend that hangs like a Qdrant that accepts connections and never answers, until `release`."""

    def __init__(self):
        super().__init__(["a"])
        self.release, self._count = threading.Event(), threading.Lock()
        self.started = self.running = self.peak = 0
        self.deadlines = []

    def search(self, *, deadline=None, **kwargs):
        with self._count:
            self.started += 1
            self.running += 1
            self.peak = max(self.peak, self.running)
            self.deadlines.append(deadline)
        try:
            assert self.release.wait(20)
            return super().search(**kwargs)
        finally:
            with self._count:
                self.running -= 1


async def _timed(coroutine):
    started = time.monotonic()
    return await coroutine, time.monotonic() - started


async def test_a_search_that_outlasts_the_call_budget_answers_timeout(caplog):
    backend = _SlowBackend()
    caplog.set_level(logging.DEBUG)
    try:
        async with _client(backend, rag_call_timeout_s=0.3) as c:
            before = time.monotonic()
            message, took = await _timed(_error(c, {"query": "secret-term"}))
            assert message == TIMEOUT and 0.25 <= took < 3
            # The backend is told when its caller stops waiting.
            assert before + 0.3 <= backend.deadlines[0] <= time.monotonic()
    finally:
        backend.release.set()
    assert not any("secret-term" in r.getMessage() for r in caplog.records)


async def test_calls_in_excess_wait_in_the_event_loop_and_answer_busy_and_threads_never_pile_up():
    backend = _SlowBackend()
    try:
        async with _client(backend, rag_call_timeout_s=0.5, rag_max_concurrency=2) as c:
            calls = [asyncio.create_task(_timed(_error(c, {"query": "q"}))) for _ in range(8)]
            await asyncio.sleep(0.1)
            # The event loop is free while they wait, and so is every thread but the toolset's two.
            info, took = await _timed(c.call_tool("server_info", {}))
            assert info.data["toolsets"]["rag"] == "enabled" and took < 0.3
            answers = await asyncio.gather(*calls)
            assert sorted(message for message, _ in answers) == [BUSY] * 6 + [TIMEOUT] * 2
            assert all(0.4 <= took < 3 for _, took in answers)  # each one at the end of ITS budget, not in turn
            assert (backend.started, backend.peak) == (2, 2)
            # The two searches still hang, abandoned by their callers: they keep their slots. New calls
            # start no thread on top of them, however many come.
            for _ in range(3):
                assert await _error(c, {"query": "q"}) == BUSY
            assert (backend.started, backend.running) == (2, 2)
            backend.release.set()  # the vector database answers again
            for _ in range(100):
                if backend.running == 0:
                    break
                await asyncio.sleep(0.02)
            assert (await c.call_tool("rag_search", {"query": "q"})).data["count"] == 1
            assert backend.started == 3  # the calls answered `busy` were dropped: they never ran afterwards
    finally:
        backend.release.set()


async def test_a_waiting_call_is_served_when_a_slot_frees_within_its_budget():
    backend = _SlowBackend()
    try:
        async with _client(backend, rag_call_timeout_s=10, rag_max_concurrency=1) as c:
            calls = [asyncio.create_task(c.call_tool("rag_search", {"query": "q"})) for _ in range(3)]
            await asyncio.sleep(0.2)
            assert (backend.started, backend.running) == (1, 1)
            backend.release.set()
            assert [result.data["count"] for result in await asyncio.gather(*calls)] == [1, 1, 1]
            assert (backend.started, backend.peak) == (3, 1)
    finally:
        backend.release.set()


async def test_a_refused_argument_takes_no_slot():
    backend = _SlowBackend()
    try:
        async with _client(backend, rag_call_timeout_s=5, rag_max_concurrency=1) as c:
            hanging = asyncio.create_task(c.call_tool("rag_search", {"query": "q"}))
            await asyncio.sleep(0.1)
            message, took = await _timed(_error(c, {"query": "q", "source_codes": "ecb"}))
            assert message.startswith("[invalid_argument] source_codes must be a JSON array") and took < 1
            message, took = await _timed(_error(c, {"query": "q", "limit": 99}))
            assert message.startswith("[limit_exceeded]") and took < 1
            backend.release.set()
            await hanging
    finally:
        backend.release.set()


# -- tier ---------------------------------------------------------------------------------------------


async def test_rag_search_is_registered_for_the_read_tier():
    """Fails if the tool loses its tier check, or asks for another tier."""
    from fastmcp.server.auth import AccessToken, AuthContext

    toolset = build_rag_toolset(_CannedBackend(["a"]), _settings(), auth_enabled=True)
    tool = await toolset.local_provider.get_tool("search")
    assert tool.auth is not None
    assert await toolset.get_tool("search") is None  # hidden from a caller without a token

    def allowed(scopes) -> bool:
        token = None if scopes is None else AccessToken(token="t", client_id="c", scopes=scopes, expires_at=None)
        return tool.auth(AuthContext(token=token, component=tool))

    assert allowed(["read"]) and allowed(["read", "act"])
    assert not allowed([]) and not allowed(["other"]) and not allowed(None)  # a caller below the read tier


def test_the_readme_documents_the_codes_every_reason_and_every_argument():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    for needle in ("`rag_unavailable`", "`limit_exceeded`", "`rag_search`", *(f"`{r}`" for r in backend_module.REASONS),
                   "`source_codes`", "`doc_types`", "`languages`", "`date_from`, `date_to`", "`limit`", "`citation`",
                   "the first page is 1", "`ORT_DISABLE_TELEMETRY`", "`HF_HUB_OFFLINE=1`", "`mem_limit: 3g`",
                   "no outbound connection", "MOF_RAG_DESCRIPTION_NOTES="):
        assert needle in readme, needle


def test_the_suggested_notes_of_the_readme_fit_the_setting():
    """The value the README suggests for the central-bank collection is accepted as it is written."""
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    line = next(line for line in readme.splitlines() if line.startswith("MOF_RAG_DESCRIPTION_NOTES="))
    settings = Settings.from_env({"MOF_RAG_DESCRIPTION_NOTES": line.split("=", 1)[1]})
    for needle in ('"A2" policy statement (for the ECB: the press-conference page, with its questions and answers)',
                   ('"B1" press-conference material (Federal Reserve: transcript; ECB: the combined decision and '
                    "statement, without the Q&A)"), '"E1" monetary policy report or staff projections',
                   '"F1" economic projections', '"A1" rate-decision press release',
                   '"A3" minutes or accounts of the meeting'):
        assert needle in settings.rag_description_notes, needle
