"""The rag toolset on a fake engine: rag-engine's own store in qdrant-client's in-process mode, its
FakeEmbedder (no model), and a monkeypatched `search` where an outage is needed."""
import asyncio
import json
import logging
import re
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
from rag_engine.store import (
    CollectionMismatchError,
    CollectionNotFoundError,
    MissingIndexError,
    QdrantAuthError,
    QdrantStore,
)
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
    MPC,
    PROJECTIONS,
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
DATES_SENTENCE = ("With a bound set, a document known only to the month or the year matches any range overlapping "
                  "its period; documents without a date are left out unless include_undated is true.")
UNDATED_SENTENCE = "- include_undated: true also keeps the documents without a date under a date bound; default false."
CITATION_SENTENCE = ("cite each passage with its `citation`, a ready-made reference: title, then source and "
                     "document-type codes and date in brackets, then the page, each part the document lacks left out.")
HIT_KEYS = ["rank", "score", "text", "doc_id", "title", "date", "date_precision", "date_kind", "source_code",
            "doc_type", "language", "page", "chunk_number", "citation"]
PRECISION_SENTENCE = ("`date_precision` (day, month, year or null) says how much of `date` is known: a month or a "
                      "year is written as its first day. `date_kind` (such as publication or meeting, or null) "
                      "says what the date marks.")


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
                   "rag.doc_types", "rag.languages", "never send an empty array", "an array even for a single code",
                   DATES_SENTENCE, UNDATED_SENTENCE, CITATION_SENTENCE, PRECISION_SENTENCE,
                   "title, date, date_precision, date_kind, source_code,",
                   "limit: number of passages, from 1 to 20; omitted = 8."):
        assert needle in tool.description, needle
    # The default text is corpus-neutral: no bank, no code and no meaning of a code that could go stale.
    for absent in ("ecb", "ECB", "European", "Federal", "England", "Japan", "A1", "A2", "A3", "B1", "E1", "F1",
                   "press", "minutes", "About this collection", "bank", "whole-year"):
        assert absent not in tool.description, absent
    assert len(tool.description) < 1800
    # As a client receives it: no markup, no escaped character, and no placeholder a model would send as it
    # is written (an example value in brackets, a made-up code).
    served = tool.description + " ".join(p["description"] for p in tool.input_schema["properties"].values())
    for absent in ("<", ">", "&lt;", "&gt;", "&amp;", "&quot;", "&#", "e.g.", '["'):
        assert absent not in served, absent
    schema = tool.input_schema
    assert list(schema["properties"]) == ["query", "source_codes", "doc_types", "date_from", "date_to",
                                         "include_undated", "languages", "limit"]
    # A JSON boolean and nothing else, false by default; the strings "true" and "false" are refused.
    flag = schema["properties"]["include_undated"]
    assert (flag["type"], flag["default"]) == ("boolean", False) and "not a string" in flag["description"]
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


@pytest.mark.parametrize(("mode", "overlap"), [("auto", True), ("interval", True), ("legacy", False)])
async def test_the_description_says_how_a_bound_treats_a_month_or_a_year_in_the_configured_mode(backend, mode,
                                                                                               overlap):
    async with _client(backend, rag_date_filter_mode=mode) as c:
        description = {t.name: t for t in await c.list_tools()}["rag_search"].description
    assert (DATES_SENTENCE in description) is overlap
    legacy = ("With a bound set, a document known only to the month or the year is found only on its period's first "
              "day; documents without a date are left out unless include_undated is true.")
    assert (legacy in description) is not overlap


@pytest.mark.parametrize("mode", ["auto", "interval", "legacy"])
def test_the_description_stays_under_its_cap_in_the_worst_case(mode):
    """The longest limits the settings accept (MOF_RAG_MAX_LIMIT and MOF_RAG_DEFAULT_LIMIT at 1000), in every
    date filter mode, without notes: under the 1800 characters the README states."""
    from myopenfund_mcp.rag.tools import search_description

    worst = Settings.from_env({"MOF_RAG_MAX_LIMIT": "1000", "MOF_RAG_DEFAULT_LIMIT": "1000",
                               "MOF_RAG_DATE_FILTER_MODE": mode})
    assert "from 1 to 1000; omitted = 1000." in search_description(worst)
    assert len(search_description(worst)) < 1800


async def test_a_hit_carries_the_passage_its_document_and_a_citation_and_nothing_else(backend):
    async with _client(backend) as c:
        data = (await c.call_tool("rag_search", {"query": ECB_RATES})).data
    assert list(data) == ["query", "filters", "count", "hits"]
    assert data["query"] == ECB_RATES and data["count"] == len(data["hits"]) == CHUNKS
    assert data["filters"] == {"source_codes": None, "doc_types": None, "languages": None,
                               "date_from": None, "date_to": None, "include_undated": False}
    assert [h["rank"] for h in data["hits"]] == list(range(1, CHUNKS + 1))
    scores = [h["score"] for h in data["hits"]]
    assert scores == sorted(scores, reverse=True) and all(round(s, 4) == s for s in scores)
    first = data["hits"][0]
    assert list(first) == HIT_KEYS  # no point_id (64-bit), no payload, no corpus
    assert first == {"rank": 1, "score": 1.0, "text": ECB_RATES, "doc_id": "ecb/A1/2022-07-21",
                     "title": "Monetary policy decisions", "date": "2022-07-21", "date_precision": "day",
                     "date_kind": "meeting", "source_code": "ecb",
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
    """What the description announces: title, then codes and date in brackets, then the page."""
    assert citation(hit) == expected


@pytest.mark.parametrize(("date", "precision", "shown"), [
    ("2016-07-01", "month", "July 2016"),
    ("2016-01-01", "month", "January 2016"),
    ("2016-12-01", "month", "December 2016"),
    ("2016-07-14", "month", "July 2016"),  # not the first day: the month all the same
    ("2016-07-01T00:00:00Z", "month", "July 2016"),
    ("2016-01-01", "year", "2016"),
    ("2016-07-01T00:00:00Z", "year", "2016"),
    ("2016-07-01", "day", "2016-07-01"),
    ("2016-07-01", None, "2016-07-01"),  # unknown: the stored date, as before the producers stated it
    ("2016-07-01", "week", "2016-07-01"),  # a precision the server does not know: the stored date
    ("2016-07-01", "", "2016-07-01"),
    ("2016-07-01", ["month"], "2016-07-01"),
    ("2016-13-01", "month", "2016-13-01"),  # not a month: written as stored, never guessed
    ("2016-00-01", "month", "2016-00-01"),
    ("2016-07", "month", "2016-07"),  # not a full date: as stored
    ("2016", "year", "2016"),
    ("20160701", "month", "20160701"),
    ("2016-07-011", "month", "2016-07-011"),
    (None, "month", None),
    ("", "year", None),
])
def test_the_citation_writes_the_date_at_its_precision(date, precision, shown):
    hit = {"title": "T", "source_code": "gb", "doc_type": "A3", "date": date, "date_precision": precision, "page": 0}
    assert citation(hit) == (f"T (gb A3, {shown}), p. 1" if shown else "T (gb A3), p. 1")


async def test_a_hit_says_how_precise_its_date_is_and_what_it_marks(backend):
    """Null when the payload has neither key: the production collection until the producers' backfill
    reaches it. A month or a year is cited as such, a day or an unknown precision as the stored date."""
    async with _client(backend) as c:
        async def first(text):
            hit = (await c.call_tool("rag_search", {"query": text, "limit": 1})).data["hits"][0]
            return hit["doc_id"], hit["date"], hit["date_precision"], hit["date_kind"], hit["citation"]

        assert await first(MPC) == ("gb/A3/2016-07", "2016-07-01", "month", "meeting",
                                    "Monetary Policy Summary (gb A3, July 2016), p. 2")
        assert await first(PROJECTIONS) == ("ecb/E1/2015", "2015-01-01", "year", "period",
                                            "Staff projections (ecb E1, 2015)")
        assert await first(ECB_RATES) == ("ecb/A1/2022-07-21", "2022-07-21", "day", "meeting",
                                          "Monetary policy decisions (ecb A1, 2022-07-21), p. 1")
        assert await first(FOMC) == ("us/A3/2022-06-15", "2022-06-15", None, None,  # neither key stored
                                     "Minutes of the Federal Open Market Committee (us A3, 2022-06-15), p. 5")
        assert await first(BOE) == ("gb/B1/undated", None, None, None, "gb/B1/undated (gb B1), p. 3")  # nulls


# -- arguments ---------------------------------------------------------------------------------------


async def test_filters_narrow_the_search_and_are_echoed_as_applied(backend):
    async with _client(backend) as c:
        async def ids(**arguments):
            data = (await c.call_tool("rag_search", {"query": "rates", **arguments})).data
            return sorted({h["doc_id"] for h in data["hits"]}), data["filters"]

        assert (await ids(source_codes=["ecb", "us", "ecb"])) == (
            ["ecb/A1/2022-07-21", "ecb/E1/2015", "us/A3/2022-06-15"],
            {"source_codes": ["ecb", "us"], "doc_types": None, "languages": None, "date_from": None,
             "date_to": None, "include_undated": False})
        assert (await ids(doc_types=["B1"]))[0] == ["gb/B1/undated"]  # a type held by one chunk
        assert (await ids(languages=["ja"]))[0] == ["jp/E1/2023-01-18"]
        assert (await ids(source_codes=["ecb"], doc_types=["A3"]))[0] == []  # AND between arguments
        # Both bounds inclusive; with a bound set the undated document is left out.
        found, applied = await ids(date_from="2022-06-15", date_to="2022-07-21")
        assert found == ["ecb/A1/2022-07-21", "us/A3/2022-06-15"]
        assert (applied["date_from"], applied["date_to"]) == ("2022-06-15", "2022-07-21")
        assert "gb/B1/undated" not in (await ids(date_from="1900-01-01"))[0]
        assert "gb/B1/undated" in (await ids())[0]


async def test_include_undated_lets_the_documents_without_a_date_through_a_date_bound(backend):
    async with _client(backend) as c:
        async def found(**arguments):
            data = (await c.call_tool("rag_search", {"query": "rates", **arguments})).data
            return {h["doc_id"] for h in data["hits"]}, data["filters"]["include_undated"]

        dated, flag = await found(date_from="1900-01-01")
        assert "gb/B1/undated" not in dated and flag is False and len(dated) == 6
        everything, flag = await found(date_from="1900-01-01", include_undated=True)
        assert everything == dated | {"gb/B1/undated"} and flag is True
        # Only the undated ones are added: the dated documents outside the range stay out.
        assert (await found(date_to="2015-12-31", include_undated=True)) == ({"ecb/E1/2015", "gb/B1/undated"}, True)
        late, flag = await found(date_from="2023-01-01", include_undated=True)
        assert late == {"jp/E1/2023-01-18", "gb/B1/undated"} and flag is True
        # Without a date bound it changes nothing, and is accepted rather than refused; null is false.
        assert (await found(include_undated=True))[0] == (await found())[0]
        assert (await found(include_undated=None)) == (await found())


@pytest.mark.parametrize("mode", ["interval", "auto", "legacy"])
async def test_a_month_or_a_year_matches_a_range_overlapping_its_period_in_interval_mode(seeded, mode):
    """The fixture's intervals as rag-engine's build_payload writes them. In-process Qdrant keeps no payload
    index: `auto` reads the collection, finds no index, and applies the bound to `date` (legacy)."""
    store, fake = seeded
    backend = EngineRagBackend(_settings(rag_date_filter_mode=mode), store=store, embedder=fake)
    intervals = mode == "interval"
    async with _client(backend, rag_date_filter_mode=mode) as c:
        async def found(date_from, date_to):
            data = (await c.call_tool("rag_search", {"query": "rates", "date_from": date_from,
                                                     "date_to": date_to})).data
            return {h["doc_id"] for h in data["hits"]}

        mid_july = await found("2016-07-14", "2016-07-14")
        assert mid_july == ({"gb/A3/2016-07"} if intervals else set())
        assert await found("2016-07-31", "2016-08-31") == ({"gb/A3/2016-07"} if intervals else set())
        assert await found("2016-06-01", "2016-06-30") == set()  # inclusive bounds, nothing outside the month
        assert await found("2016-07-01", "2016-07-01") == {"gb/A3/2016-07"}  # the stored first day: every mode
        assert await found("2015-06-01", "2015-06-30") == ({"ecb/E1/2015"} if intervals else set())
        assert await found("2015-12-31", "2016-01-31") == ({"ecb/E1/2015"} if intervals else set())
        # A day is a day in every mode.
        assert await found("2022-07-21", "2022-07-21") == {"ecb/A1/2022-07-21"}
        assert await found("2022-07-22", "2022-12-31") == set()


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
FLAG = "include_undated must be a JSON boolean, true or false without quotes, not "
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
    # A boolean is a JSON true or false: never guessed from a string or a number.
    ({"query": "x", "date_from": "2016-01-01", "include_undated": "true"}, FLAG + '"true"; omit it for false'),
    ({"query": "x", "include_undated": "false"}, FLAG + '"false"; omit it for false'),
    ({"query": "x", "include_undated": "yes"}, FLAG + '"yes"; omit it for false'),
    ({"query": "x", "include_undated": 1}, FLAG + "1; omit it for false"),
    ({"query": "x", "include_undated": 0}, FLAG + "0; omit it for false"),
    ({"query": "x", "include_undated": [True]}, FLAG + "a JSON array; omit it for false"),
    ({"query": "x", "include_undated": {"value": True}}, FLAG + "a JSON object; omit it for false"),
    ({"query": "x", "include_undated": "x" * 200}, FLAG + '"' + "x" * 36 + "...; omit it for false"),
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
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 3
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


@pytest.mark.parametrize("seen_by_a_check", [True, False])
async def test_the_values_are_dropped_when_the_collection_comes_back(seeded, monkeypatch, clock, seen_by_a_check):
    """A collection recreated during an outage: its old codes must not refuse the new ones."""
    store, fake = seeded
    codes = _Codes(store, source_code=["old"], doc_type=["A1"], language=["en"])
    down = _Store(store, codes)
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=3600, rag_codes_retry_s=3600, rag_check_retry_s=5),
                               store=down, embedder=fake)
    async with _client(backend) as c:
        assert (await _error(c, {"query": "x", "source_codes": ["ecb"]})).endswith('valid values: "old"')
        with monkeypatch.context() as patched:  # a search finds the collection gone ...
            patched.setattr(backend_module, "search", lambda *a, **k: (_ for _ in ()).throw(_response(404)))
            assert (await _error(c, {"query": "x"})).startswith("[rag_unavailable] collection_missing")
        if seen_by_a_check:  # ... and so does the check of the next call
            down.failure = CollectionNotFoundError("gone")
            assert (await _error(c, {"query": "x"})).startswith("[rag_unavailable] collection_missing")
            assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "unavailable"
            clock[0] += 6
        down.failure, codes.values["source_code"] = None, ["ecb", "us"]  # recreated
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 3
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
        assert (await c.call_tool("rag_search", {"query": "x", "doc_types": ["B1"]})).data["count"] == 1
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
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 3
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
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 3
        assert client.scrolls == 2 and backend._reading  # running, and the call above did not wait for it
        assert (await c.call_tool("rag_search", {"query": "x", "doc_types": ["B1"]})).data["count"] == 1
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
    (_response(500), "qdrant_error"),
    (_response(503), "qdrant_error"),
    (_response(429), "qdrant_error"),
    (ResponseHandlingException(httpx.ConnectError(f"cannot connect to {HOST}")), "qdrant_unreachable"),
    (ResponseHandlingException(httpx.ReadTimeout(f"timed out reading {HOST}")), "qdrant_unreachable"),
    (QdrantAuthError("Qdrant refused the request (HTTP 401): check QDRANT_API_KEY"), "qdrant_auth"),
]


QUERY = "secret-term what did the council decide"
# Written out here, not read from the module: what a client is told.
UNREACHABLE = ("[rag_unavailable] qdrant_unreachable: the vector database cannot be reached or did not answer in "
               "time; retry later")
REFUSED = ("[rag_unavailable] search_refused: the vector database refused this search, and only this one: the "
           "collection, as it is indexed, cannot serve it (most often a filter on a field without a payload "
           "index); retry with fewer filters, without date_from and date_to first")


def _timed_out() -> ResponseHandlingException:
    return ResponseHandlingException(httpx.ReadTimeout(f"timed out reading {HOST}"))


def _failing_for(monkeypatch, failures: dict):
    """`search` fails with `failures[query]` for the queries listed there; the others are really served."""
    real = backend_module.search

    def search(query, **kwargs):
        if query in failures:
            raise failures[query]
        return real(query, **kwargs)

    monkeypatch.setattr(backend_module, "search", search)


@pytest.mark.parametrize(("failure", "reason"), OUTAGES, ids=[r + "-" + type(f).__name__ for f, r in OUTAGES])
async def test_an_outage_is_rag_unavailable_with_a_reason_and_leaks_nothing(seeded, monkeypatch, caplog,
                                                                             failure, reason):
    """The search fails, then the collection check does: Qdrant is down, and said so to every caller."""
    store, fake = seeded
    down = _Store(store, store.client)
    backend = EngineRagBackend(_settings(), store=down, embedder=fake)
    _failing_for(monkeypatch, {QUERY: failure})
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        message = await _error(c, {"query": QUERY, "source_codes": ["ecb"]})
        # That failure alone refuses nobody: server_info still shows the outcome of the last check.
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"
        down.failure = failure
        again = await _error(c, {"query": "another caller"})  # checks the collection: it fails too
        info = (await c.call_tool("server_info", {})).data
    assert message == again == f"[rag_unavailable] {reason}: {backend_module.REASONS[reason]}"
    assert info["rag"]["status"] == "unavailable" and info["rag"]["reason"] == reason
    everything = [message, json.dumps(info)] + [r.getMessage() for r in caplog.records]
    for text in everything:
        assert HOST not in text and KEY not in text and "QDRANT_API_KEY" not in text, text
        assert "secret-term" not in text and "council" not in text, text  # nor the query, on this path either
    logged = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.rag"]
    for what in ("search failed", "collection check failed"):
        assert any(what in line and reason in line and type(failure).__name__ in line for line in logged), what


async def test_one_search_that_times_out_refuses_no_other_caller(seeded, monkeypatch):
    """Qdrant is healthy and one request of one caller is answered too late: the next call checks the
    collection (one cheap request), is served, and the calls after it ask nothing more."""
    store, fake = seeded
    healthy = _Store(store, store.client)
    backend = EngineRagBackend(_settings(rag_check_retry_s=3600), store=healthy, embedder=fake)
    _failing_for(monkeypatch, {"heavy": _timed_out()})
    async with _client(backend) as c:
        assert healthy.checks == 1  # at start
        for expected_checks in (2, 3):  # as often as it happens
            assert await _error(c, {"query": "heavy"}) == UNREACHABLE
            for _ in range(5):
                assert (await c.call_tool("rag_search", {"query": "light"})).data["count"] == CHUNKS
            assert healthy.checks == expected_checks  # one check after the failure, not one per call
            assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"


async def test_a_search_that_succeeds_clears_the_failure_of_another(seeded, monkeypatch):
    """A search that ends well after another failed is the proof that Qdrant answers: no check follows."""
    store, fake = seeded
    healthy = _Store(store, store.client)
    backend = EngineRagBackend(_settings(), store=healthy, embedder=fake)
    real, entered, release = backend_module.search, threading.Event(), threading.Event()

    def search(query, **kwargs):
        if query == "heavy":
            raise _timed_out()
        if query == "slow":
            entered.set()
            assert release.wait(10)
        return real(query, **kwargs)

    monkeypatch.setattr(backend_module, "search", search)
    async with _client(backend) as c:
        slow = asyncio.create_task(c.call_tool("rag_search", {"query": "slow"}))
        assert await asyncio.to_thread(entered.wait, 10)
        assert await _error(c, {"query": "heavy"}) == UNREACHABLE
        assert backend._suspected == "qdrant_unreachable"
        release.set()
        assert (await slow).data["count"] == CHUNKS
        assert backend._suspected is None
        assert (await c.call_tool("rag_search", {"query": "light"})).data["count"] == CHUNKS
        assert healthy.checks == 1  # the one of the start


@pytest.mark.parametrize("status", [400, 422, 409])
async def test_a_search_qdrant_refuses_is_the_error_of_that_call_alone(seeded, monkeypatch, caplog, status):
    """HTTP 400 is what Qdrant in strict mode answers to a filter on a field without an index: nothing is
    wrong with the deployment for the other callers, and nobody is refused. The next call only checks
    the collection before its search (one cheap request), in case it is the collection that changed."""
    store, fake = seeded
    healthy = _Store(store, store.client)
    backend = EngineRagBackend(_settings(), store=healthy, embedder=fake)
    _failing_for(monkeypatch, {QUERY: _response(status)})
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        for _ in range(3):
            message = await _error(c, {"query": QUERY, "date_from": "2016-01-01"})
            assert message == REFUSED
            assert (await c.call_tool("rag_search", {"query": "another caller"})).data["count"] == CHUNKS
            info = (await c.call_tool("server_info", {})).data
            assert info["rag"]["status"] == "ready" and "reason" not in info["rag"]
        # One check after each refusal, by the next call, and no failure recorded.
        assert healthy.checks == 4 and backend._suspected is None and backend._collection_reason is None
    for text in [message] + [r.getMessage() for r in caplog.records]:
        assert HOST not in text and KEY not in text and "secret-term" not in text, text
    assert not any(r.levelno >= logging.WARNING for r in caplog.records if r.name == "myopenfund_mcp.rag")


async def test_a_collection_that_no_longer_matches_the_model_is_found_after_a_refused_search(seeded, monkeypatch):
    """A collection recreated with another vector size refuses EVERY search with HTTP 400: the call
    after the first refusal checks the collection and says what is wrong, to everyone."""
    store, fake = seeded
    changed = _Store(store, store.client)
    backend = EngineRagBackend(_settings(), store=changed, embedder=fake)
    monkeypatch.setattr(backend_module, "search", lambda query, **kwargs: (_ for _ in ()).throw(_response(400)))
    async with _client(backend) as c:
        changed.failure = CollectionMismatchError("vector size 4, expected 32")
        assert await _error(c, {"query": "x"}) == REFUSED
        expected = "[rag_unavailable] collection_mismatch: " + backend_module.REASONS["collection_mismatch"]
        for _ in range(3):
            assert await _error(c, {"query": "y"}) == expected
        assert changed.checks == 2  # the failed check is remembered
        info = (await c.call_tool("server_info", {})).data["rag"]
        assert info["status"] == "unavailable" and info["reason"] == "collection_mismatch"


async def test_searches_that_fail_while_the_check_passes_are_remembered_like_a_failed_check(
        seeded, monkeypatch, clock, caplog):
    """Qdrant answers the collection check and no search: without a memory every call would pay a whole
    request timeout. One call per retry period pays; the others are answered at once."""
    store, fake = seeded
    half = _Store(store, store.client)
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=half, embedder=fake)
    real, searches, hung = backend_module.search, [0], [True]

    def search(query, **kwargs):
        searches[0] += 1
        if hung[0]:
            raise _timed_out()
        return real(query, **kwargs)

    monkeypatch.setattr(backend_module, "search", search)
    caplog.set_level(logging.WARNING, logger="myopenfund_mcp.rag")
    async with _client(backend) as c:
        assert await _error(c, {"query": "a"}) == UNREACHABLE  # fails: the next call checks
        assert await _error(c, {"query": "b"}) == UNREACHABLE  # check passes, search fails again
        assert (half.checks, searches[0]) == (2, 2)
        for _ in range(20):  # remembered: no request at all
            assert await _error(c, {"query": "c"}) == UNREACHABLE
        assert (half.checks, searches[0]) == (2, 2)
        info = (await c.call_tool("server_info", {})).data["rag"]
        assert info["status"] == "unavailable" and info["reason"] == "qdrant_unreachable"
        for cycle in (1, 2):  # each retry period: one check, one search that pays, the rest at once
            clock[0] += 6
            for _ in range(10):
                assert await _error(c, {"query": "d"}) == UNREACHABLE
            assert (half.checks, searches[0]) == (2 + cycle, 2 + cycle)
        hung[0] = False
        clock[0] += 6
        assert (await c.call_tool("rag_search", {"query": "e"})).data["count"] == CHUNKS
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"
        # Served again: a later single failure is a suspicion once more, not a remembered outage.
        hung[0] = True
        assert await _error(c, {"query": "f"}) == UNREACHABLE
        assert backend._collection_reason is None and backend._suspected == "qdrant_unreachable"
    warnings = [r.getMessage() for r in caplog.records if "search failed" in r.getMessage()]
    assert len(warnings) == 2, warnings  # once for the outage, once after the recovery: not once per call


def test_a_search_served_disproves_a_failure_remembered_from_a_search(seeded, monkeypatch):
    """Two slow requests of a healthy Qdrant time out around a check that passes: the second is
    remembered like a failed check, and the search that is served just after clears it for everyone."""
    store, fake = seeded
    healthy = _Store(store, store.client)
    backend = EngineRagBackend(_settings(rag_check_retry_s=3600), store=healthy, embedder=fake)
    backend.start()
    _failing_for(monkeypatch, {"heavy": _timed_out()})
    call = {"filters": {}, "limit": 3, "max_limit": 20}
    with pytest.raises(ToolError):
        backend.search(query="heavy", **call)  # suspected
    backend._require_collection()  # a third caller's check passes; its search is still running
    with pytest.raises(ToolError):
        backend.search(query="heavy", **call)  # the second slow request: remembered
    assert backend._collection_reason == "qdrant_unreachable" and backend.info()["status"] == "unavailable"
    backend._served()  # the third caller's search ends well
    assert backend._collection_reason is None and backend.info()["status"] == "ready"
    assert backend.search(query="light", **call)[1]
    # A failure remembered from a CHECK is not cleared by a late search: only a check that passes does.
    healthy.failure = CollectionNotFoundError("gone")
    with pytest.raises(ToolError):
        backend.search(query="heavy", **call)
    with pytest.raises(ToolError):
        backend.search(query="light", **call)  # the check fails
    backend._served()
    assert backend._collection_reason == "collection_missing"


async def test_after_an_outage_the_collection_is_checked_again_and_the_toolset_recovers(seeded, monkeypatch, clock):
    store, fake = seeded
    down = _Store(store, store.client)
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    _failing_for(monkeypatch, {"x": _response(404)})
    async with _client(backend) as c:
        down.failure = CollectionNotFoundError("gone")
        expected = "[rag_unavailable] collection_missing: " + backend_module.REASONS["collection_missing"]
        assert await _error(c, {"query": "x"}) == expected  # the search fails
        assert await _error(c, {"query": "y"}) == expected  # then the check of the next call
        assert down.checks == 2
        for _ in range(20):  # that failed check is remembered: no request
            assert await _error(c, {"query": "y"}) == expected
        assert down.checks == 2
        down.failure = None
        clock[0] += 6  # past rag_check_retry_s
        assert (await c.call_tool("rag_search", {"query": "y"})).data["count"] == CHUNKS
        assert down.checks == 3
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"


async def test_a_failed_check_is_remembered_a_few_seconds_and_not_run_by_every_call(seeded, clock):
    """With Qdrant hanging, each check costs a whole timeout: the calls made meanwhile answer at once."""
    store, fake = seeded
    down = _Store(store, store.client)
    down.failure = ResponseHandlingException(httpx.ReadTimeout("timed out"))
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    async with _client(backend) as c:
        assert down.checks == 1  # at start
        for _ in range(20):
            assert await _error(c, {"query": "x"}) == UNREACHABLE
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


class _Hanging(_Store):
    """A store whose collection check, while `failure` is set, hangs until `release` and then fails."""

    def __init__(self, store, client):
        super().__init__(store, client)
        self.entered, self.release = threading.Event(), threading.Event()

    def check_collection(self, name, dim):
        if self.failure is None:
            return super().check_collection(name, dim)
        self.checks += 1
        self.entered.set()
        assert self.release.wait(10)
        raise self.failure


@pytest.mark.parametrize("after", ["a failed check", "a failed search"])
async def test_calls_arriving_while_a_check_runs_are_answered_at_once(seeded, monkeypatch, clock, after):
    store, fake = seeded
    down = _Hanging(store, store.client)
    if after == "a failed check":  # the one of the start
        down.failure = _timed_out()
        down.release.set()
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    _failing_for(monkeypatch, {"heavy": _timed_out()})
    async with _client(backend) as c:
        if after == "a failed check":
            clock[0] += 6  # that failure is forgotten: the next call checks
        else:
            assert await _error(c, {"query": "heavy"}) == UNREACHABLE  # the next call checks, at once
            down.failure = _timed_out()
        down.release.clear()
        checks = down.checks
        down.entered.clear()
        first = asyncio.create_task(_error(c, {"query": "x"}))
        assert await asyncio.to_thread(down.entered.wait, 10)
        started = time.monotonic()
        for _ in range(10):  # the check is running: these do not wait for it, and do not run another
            assert await _error(c, {"query": "x"}) == UNREACHABLE
        assert time.monotonic() - started < 2 and down.checks == checks + 1
        down.release.set()
        assert await first == UNREACHABLE
        for _ in range(10):  # and it failed: remembered
            assert await _error(c, {"query": "x"}) == UNREACHABLE
        assert down.checks == checks + 1


async def test_qdrant_hanging_after_a_healthy_start_answers_a_burst_within_two_request_timeouts(seeded,
                                                                                                monkeypatch):
    """Every request hangs for the request timeout (0.5 s here), then fails. Fifty calls at once, four
    slots: four searches time out, ONE call runs the check, every other call is answered at once."""
    store, fake = seeded
    request_s = 0.5

    class Frozen(_Store):
        def check_collection(self, name, dim):
            if self.failure is None:
                return super().check_collection(name, dim)
            self.checks += 1
            time.sleep(request_s)
            raise self.failure

    down = Frozen(store, store.client)
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    searches = []

    def search(query, **kwargs):
        searches.append(query)
        time.sleep(request_s)
        raise _timed_out()

    monkeypatch.setattr(backend_module, "search", search)
    async with _client(backend, rag_call_timeout_s=5, rag_max_concurrency=4) as c:
        down.failure = _timed_out()
        started = time.monotonic()
        answers = await asyncio.gather(*[_timed(_error(c, {"query": f"q{n}"})) for n in range(50)])
        assert time.monotonic() - started < 2 * request_s + 1.5
        assert {message for message, _ in answers} == {UNREACHABLE}
        assert len(searches) == 4 and down.checks == 2  # the start, then one check for the whole burst
        message, took = await _timed(_error(c, {"query": "later"}))
        assert message == UNREACHABLE and took < 0.3 and down.checks == 2  # from the recorded failure


async def test_a_filtered_call_does_not_pay_for_two_timeouts(seeded, monkeypatch, clock):
    """The read of the filter values failed for an outage: the search is not tried after it, and the next
    call checks the collection before anything else."""
    store, fake = seeded
    codes = _Codes(store, source_code=SOURCE_CODES, doc_type=DOC_TYPES, language=["en", "ja"])
    down = _Store(store, codes)
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=60), store=down, embedder=fake)
    searched = []
    monkeypatch.setattr(backend_module, "search", lambda *a, **k: searched.append(a) or [])
    async with _client(backend) as c:
        clock[0] += 61  # the values are due
        codes.failure = down.failure = ResponseHandlingException(httpx.ReadTimeout(f"timed out reading {HOST}"))
        assert await _error(c, {"query": "x", "source_codes": ["ecb"]}) == UNREACHABLE
        assert searched == [] and codes.reads == 2 and down.checks == 1
        # The next call checks the collection, which fails too: one request, and it is remembered.
        assert await _error(c, {"query": "x", "source_codes": ["ecb"]}) == UNREACHABLE
        assert searched == [] and codes.reads == 2 and down.checks == 2
        await _error(c, {"query": "x", "source_codes": ["ecb"]})
        await _error(c, {"query": "x"})
        assert searched == [] and codes.reads == 2 and down.checks == 2
        assert (await c.call_tool("server_info", {})).data["rag"]["reason"] == "qdrant_unreachable"


async def test_a_failed_read_of_the_values_refuses_no_other_caller(seeded, monkeypatch, clock):
    """One facet request answered too late by a healthy Qdrant: the next call is served."""
    store, fake = seeded
    codes = _Codes(store, source_code=SOURCE_CODES, doc_type=DOC_TYPES, language=["en", "ja"])
    healthy = _Store(store, codes)
    backend = EngineRagBackend(_settings(rag_codes_ttl_s=60), store=healthy, embedder=fake)
    async with _client(backend) as c:
        clock[0] += 61  # the values are due
        codes.failure = _timed_out()
        assert await _error(c, {"query": "x", "source_codes": ["ecb"]}) == UNREACHABLE
        codes.failure = None
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        assert (await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})).data["count"] == 3
        assert healthy.checks == 2  # the start, then the call that followed the failure


# -- flags that must not stay set ----------------------------------------------------------------------


class _Interrupt(BaseException):
    """Not an `Exception`: what a handler of `Exception` lets through."""


def test_a_check_interrupted_by_anything_leaves_no_check_running(seeded, clock):
    store, fake = seeded
    down = _Store(store, store.client)
    down.failure = _timed_out()
    backend = EngineRagBackend(_settings(rag_check_retry_s=5), store=down, embedder=fake)
    backend.start()
    clock[0] += 6
    down.failure = _Interrupt()
    with pytest.raises(_Interrupt):
        backend._require_collection()
    assert backend._checking is False
    down.failure = None
    backend._require_collection()  # the next call checks, and is served: nothing is stuck
    assert down.checks == 3 and backend.info()["status"] == "ready"


@pytest.mark.parametrize("failure", [RuntimeError("can't start new thread"), _Interrupt()],
                         ids=["no-thread", "interrupted"])
def test_a_read_of_the_values_that_cannot_be_handed_over_leaves_no_read_running(seeded, monkeypatch, clock,
                                                                                failure):
    """The scroll could not be given to its thread: `_reading` must not stay set, or no value would ever
    be read again and every refusal would wait for the end of its call."""
    store, fake = seeded
    client = _Scrolled(store)
    backend = EngineRagBackend(_settings(rag_codes_retry_s=5), store=_Store(store, client), embedder=fake)

    def start(thread):
        raise failure

    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", start)
        if isinstance(failure, Exception):
            backend.start()  # recorded as a failed read: the search is served unchecked
        else:
            with pytest.raises(_Interrupt):
                backend.start()
    assert backend._reading is False and client.scrolls == 0
    clock[0] += 6  # past rag_codes_retry_s: the read is tried again, by the next filtered call
    with pytest.raises(ToolError) as refused:
        _backend_search(backend, time.monotonic() + 10, source_codes=["ECB"])
    assert str(refused.value).endswith("valid values: " + VALID_SOURCES) and client.scrolls == 1


def test_a_scroll_interrupted_by_anything_leaves_no_read_running(seeded, monkeypatch):
    store, fake = seeded
    client = _Scrolled(store)
    backend = EngineRagBackend(_settings(), store=_Store(store, client), embedder=fake)
    monkeypatch.setattr(backend, "_scroll_values", lambda: (_ for _ in ()).throw(_Interrupt()))
    monkeypatch.setattr(threading, "excepthook", lambda args: None)  # the thread ends on it: not this test's matter
    backend.start()
    _settled(backend)
    assert backend._reading is False and backend._values is None


# -- the call's deadline, inside the backend ------------------------------------------------------------


def _backend_search(backend, deadline, **filters):
    return backend.search(query="x", filters={"source_codes": None, "doc_types": None, "languages": None,
                                              "date_from": None, "date_to": None, **filters},
                          limit=3, max_limit=20, deadline=deadline)


def test_no_search_is_started_after_the_calls_deadline(backend, monkeypatch):
    """A call that got its slot when its caller had already gone asks Qdrant nothing."""
    backend.start()
    searched = []
    monkeypatch.setattr(backend_module, "search", lambda *a, **k: searched.append(a) or [])
    with pytest.raises(ToolError) as refused:
        _backend_search(backend, time.monotonic() - 0.001)
    assert str(refused.value) == TIMEOUT and searched == []
    assert _backend_search(backend, time.monotonic() + 60) == (
        {"source_codes": None, "doc_types": None, "languages": None, "date_from": None, "date_to": None,
         "include_undated": False}, [])
    assert len(searched) == 1
    assert _backend_search(backend, None)[1] == [] and len(searched) == 2  # no deadline: no limit


def test_the_read_before_a_refusal_is_not_waited_for_beyond_the_calls_deadline(seeded, clock):
    """The backend alone, without the tool's own budget around it: the scroll hangs, the call does not."""
    store, fake = seeded
    client = _Scrolled(store)
    backend = EngineRagBackend(_settings(rag_codes_retry_s=5), store=_Store(store, client), embedder=fake)
    backend.start()
    _settled(backend)
    clock[0] += 6  # the values may be read again
    client.hold = True
    try:
        started = time.monotonic()
        with pytest.raises(ToolError) as refused:
            _backend_search(backend, started + 0.3, source_codes=["ECB"])
        took = time.monotonic() - started
        # Refused against the values already known, at the deadline, while the scroll still runs.
        assert str(refused.value).endswith("valid values: " + VALID_SOURCES)
        assert 0.25 <= took < 3 and client.scrolls == 2 and backend._reading
    finally:
        client.release.set()
        _settled(backend)


# -- date bounds: the engine's date filter mode -----------------------------------------------------


class _SchemaClient:
    """The fixture's Qdrant client, but the payload schema it reports (are `date_start` and `date_end`
    indexed?) is the test's: `indexed`, or a `failure` raised after waiting for `release` while `hold`
    is set. The collection check of the store sees the real collection."""

    def __init__(self, client, indexed):
        self._client, self.indexed, self.failure, self.hold = client, indexed, None, False
        self.release, self.reads, self.threads, self.checking = threading.Event(), 0, [], threading.local()

    def get_collection(self, collection_name):
        info = self._client.get_collection(collection_name=collection_name)
        if getattr(self.checking, "active", False):
            return info
        self.reads += 1
        self.threads.append(threading.current_thread().name)
        if self.hold:
            assert self.release.wait(10)
        if self.failure is not None:
            raise self.failure
        from types import SimpleNamespace as NS

        schema = {field: NS(data_type="datetime") for field in ("date_start", "date_end")} if self.indexed else {}
        return info.model_copy(update={"payload_schema": schema})

    def __getattr__(self, name):
        return getattr(self._client, name)


class _Schema(QdrantStore):
    """rag-engine's own store (its cache of the answer, its fallback and retry delay) over that client."""

    def __init__(self, client, indexed=False):
        self.schema = _SchemaClient(client, indexed)
        super().__init__(client=self.schema)
        self.asked, self.checks, self.done = [], 0, threading.Event()

    def check_collection(self, name, dim):
        self.checks += 1
        self.schema.checking.active = True
        try:
            return super().check_collection(name, dim)
        finally:
            self.schema.checking.active = False

    def uses_date_intervals(self, collection, *, max_age=None, retry_after=None):
        self.asked.append((max_age, retry_after))
        try:
            return super().uses_date_intervals(collection, max_age=max_age, retry_after=retry_after)
        finally:
            self.done.set()


def _spied(monkeypatch) -> list[dict]:
    """The keyword arguments of every search sent to the engine (which is really run)."""
    real, seen = backend_module.search, []

    def search(query, **kwargs):
        seen.append({"query": query, **kwargs})
        return real(query, **kwargs)

    monkeypatch.setattr(backend_module, "search", search)
    return seen


@pytest.mark.parametrize("indexed", [True, False])
async def test_auto_mode_reads_the_collection_for_a_date_bound_and_hands_its_answer_to_the_engine(
        seeded, monkeypatch, indexed):
    store, fake = seeded
    for refresh, reads in ((0, 3), (300, 1)):  # 0: read before every date-bounded search
        schema = _Schema(store.client, indexed=indexed)
        backend = EngineRagBackend(_settings(rag_date_filter_refresh_s=refresh, rag_date_filter_retry_s=11),
                                   store=schema, embedder=fake)
        sent = _spied(monkeypatch)
        async with _client(backend) as c:
            assert schema.schema.reads == 1  # once at start, after the collection check passed
            await c.call_tool("rag_search", {"query": "x", "source_codes": ["ecb"]})
            assert schema.schema.reads == 1  # no date bound: nothing to ask
            await c.call_tool("rag_search", {"query": "x", "date_from": "2022-01-01"})
            await c.call_tool("rag_search", {"query": "x", "date_to": "2022-12-31"})
            assert schema.schema.reads == reads
        # The settings, never the engine's defaults or its environment.
        assert schema.asked == [(refresh, 11)] * 3
        assert [(s["date_filter_mode"], s["date_filter_refresh"], s["date_filter_retry"]) for s in sent] == [
            ("auto", refresh, 11)] + [("interval" if indexed else "legacy", refresh, 11)] * 2


async def test_no_schema_read_at_start_after_a_failed_read_of_the_values(seeded, monkeypatch):
    """The read of the filter values failed for an outage at start: the schema read could only add one more
    request timeout to the start. It is made by the first date-bounded call instead."""
    store, fake = seeded
    schema = _Schema(store.client, indexed=True)
    backend = EngineRagBackend(_settings(), store=schema, embedder=fake)
    monkeypatch.setattr(backend, "_facet_values", lambda: (_ for _ in ()).throw(_timed_out()))
    backend.start()
    assert schema.schema.reads == 0 and backend._suspected == "qdrant_unreachable"
    monkeypatch.undo()  # Qdrant answers again
    sent = _spied(monkeypatch)
    async with _client(backend) as c:
        assert (await c.call_tool("rag_search", {"query": "x", "date_from": "2022-01-01"})).data["count"] > 0
    assert schema.schema.reads == 1 and sent[-1]["date_filter_mode"] == "interval"


@pytest.mark.parametrize("mode", ["interval", "legacy"])
async def test_a_fixed_mode_never_reads_the_collection_and_is_passed_as_it_is(seeded, monkeypatch, mode):
    store, fake = seeded
    schema = _Schema(store.client, indexed=True)
    backend = EngineRagBackend(_settings(rag_date_filter_mode=mode), store=schema, embedder=fake)
    sent = _spied(monkeypatch)
    async with _client(backend) as c:
        found = (await c.call_tool("rag_search", {"query": "x", "date_from": "2022-06-15",
                                                  "date_to": "2022-07-21"})).data
    assert schema.schema.reads == 0 and schema.asked == [] and [s["date_filter_mode"] for s in sent] == [mode]
    assert sorted({h["doc_id"] for h in found["hits"]}) == ["ecb/A1/2022-07-21", "us/A3/2022-06-15"]


async def test_the_schema_read_runs_in_the_calls_slot_and_within_its_budget(seeded, monkeypatch):
    """The read hangs like a frozen Qdrant: the call answers `timeout` at the end of its budget, the slot
    stays taken (another call is `busy`), and no search is started once the budget is spent."""
    store, fake = seeded
    schema = _Schema(store.client, indexed=True)
    backend = EngineRagBackend(_settings(rag_date_filter_refresh_s=0), store=schema, embedder=fake)
    sent = _spied(monkeypatch)
    async with _client(backend, rag_call_timeout_s=0.4, rag_max_concurrency=1) as c:
        reads = schema.schema.reads
        schema.schema.hold, schema.schema.failure = True, _timed_out()
        schema.done.clear()
        try:
            bounded = asyncio.create_task(_timed(_error(c, {"query": "x", "date_from": "2022-01-01"})))
            await asyncio.sleep(0.1)
            assert schema.schema.reads == reads + 1 and schema.schema.threads[-1].startswith("mof-rag")
            assert await _error(c, {"query": "x"}) == BUSY  # the read holds the only slot
            message, took = await bounded
            assert message == TIMEOUT and 0.3 <= took < 3
        finally:
            schema.schema.release.set()
        assert await asyncio.to_thread(schema.done.wait, 10)
        await asyncio.sleep(0.05)  # the worker ends its call (deadline check) after the read
        assert sent == []  # the read failed late: the deadline had passed, no search was started
        # A failed read is not an outage: no failure is remembered, no collection check follows.
        assert (backend._collection_reason, backend._suspected, backend._recheck) == (None, None, False)
        checks, schema.schema.hold = schema.checks, False
        data = (await c.call_tool("rag_search", {"query": "x", "date_from": "2022-01-01"})).data
        assert data["count"] > 0 and schema.checks == checks
        assert schema.schema.reads == reads + 1  # within the retry delay the engine does not read again
        assert sent[-1]["date_filter_mode"] == "interval"  # the last answer it read (at start)


@pytest.mark.parametrize("failure", [_timed_out(), _response(500), _response(403),
                                     QdrantAuthError("Qdrant refused the request (HTTP 401)")],
                         ids=["timeout", "500", "403", "auth"])
async def test_a_failed_schema_read_refuses_no_call_and_is_not_remembered_as_an_outage(seeded, monkeypatch,
                                                                                       caplog, failure):
    store, fake = seeded
    schema = _Schema(store.client, indexed=True)
    schema.schema.failure = failure  # from the start: no answer was ever read
    backend = EngineRagBackend(_settings(), store=schema, embedder=fake)
    sent = _spied(monkeypatch)
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        for _ in range(3):
            data = (await c.call_tool("rag_search", {"query": FOMC, "date_from": "2022-06-15",
                                                     "date_to": "2022-06-15"})).data
            assert [h["doc_id"] for h in data["hits"]] == ["us/A3/2022-06-15"]
        info = (await c.call_tool("server_info", {})).data["rag"]
    assert info["status"] == "ready" and schema.checks == 1  # the start's check only
    assert schema.schema.reads == 1  # the start's read; the calls within the retry delay ask nothing
    assert [s["date_filter_mode"] for s in sent] == ["legacy"] * 3  # never read: the filter on `date`
    assert (backend._collection_reason, backend._suspected) == (None, None)
    assert not [r for r in caplog.records if r.name == "myopenfund_mcp.rag" and r.levelno >= logging.WARNING]


# What a proxy or a TLS layer in front of Qdrant answers when the schema read fails: each names a host,
# an address or a hint that must reach no log record, whatever the logger.
PROXY_BODY = b'{"status":{"error":"upstream qdrant-prod.internal:6333 refused the connection"}}'
NGINX_502 = b"<html><head><title>502 Bad Gateway</title></head><body>upstream 10.0.4.17:6333</body></html>"
SCHEMA_FAILURES = [
    UnexpectedResponse(500, "Internal Server Error", PROXY_BODY, httpx.Headers()),
    UnexpectedResponse(502, "Bad Gateway", NGINX_502, httpx.Headers()),
    ResponseHandlingException(httpx.ConnectError(
        f"[SSL: CERTIFICATE_VERIFY_FAILED] hostname '{HOST}' doesn't match 'qdrant-prod.internal'")),
    UnexpectedResponse(401, "Unauthorized", b'{"status":{"error":"Invalid api-key k-hunter2"}}', httpx.Headers()),
]
LEAKS = (HOST, KEY, "qdrant-prod.internal", "10.0.4.17", "6333", "upstream", "<html", "Raw response", "api-key",
         "QDRANT_API_KEY", "secret-term", "council")


def _record_texts(records) -> list[str]:
    """Everything a handler could write of each record: the message, the exception and the stack."""
    texts = []
    for record in records:
        texts.append(record.getMessage())
        if record.exc_info:
            texts.append(logging.Formatter().formatException(record.exc_info))
        texts.extend(filter(None, (record.exc_text, record.stack_info)))
    return texts


@pytest.mark.parametrize("failure", SCHEMA_FAILURES, ids=["proxy-500", "nginx-502", "tls", "auth-401"])
async def test_a_failed_schema_read_leaves_no_host_body_or_key_in_any_log_record(seeded, monkeypatch, caplog,
                                                                                 failure):
    store, fake = seeded
    schema = _Schema(store.client, indexed=True)
    schema.schema.failure = failure
    caplog.set_level(logging.DEBUG)
    backend = EngineRagBackend(_settings(), store=schema, embedder=fake)
    async with _client(backend) as c:
        data = (await c.call_tool("rag_search", {"query": "secret-term what did the council decide",
                                                 "date_from": "2022-01-01"})).data
    assert data["count"] > 0
    engine = [r for r in caplog.records if r.name.startswith("rag_engine")]
    warned = [r for r in engine if r.levelno == logging.WARNING]
    assert len(warned) == 1  # the engine's warning is kept, at its level, with the class of the error
    assert type(failure).__name__ in warned[0].getMessage() or "QdrantAuthError" in warned[0].getMessage()
    assert COLLECTION in warned[0].getMessage() and "legacy" in warned[0].getMessage()
    for text in _record_texts(caplog.records):
        for leak in LEAKS:
            assert leak not in text, (leak, text)


def test_any_engine_record_carrying_an_error_is_reduced_to_its_class(seeded, caplog):
    """Not only the known warning: any record of an engine logger with an exception among its arguments, an
    attached exception, or the server's host or key in its text, is written without them."""
    import rag_engine.embedding

    store, fake = seeded
    EngineRagBackend(_settings(), store=store, embedder=fake)  # installs the filter
    caplog.set_level(logging.DEBUG)
    error = UnexpectedResponse(502, "Bad Gateway", NGINX_502, httpx.Headers())
    for logger in (logging.getLogger("rag_engine.store"), logging.getLogger(rag_engine.embedding.__name__),
                   logging.getLogger("rag_engine"), logging.getLogger("rag_engine.some_future_module")):
        logger.error("something failed: %s", error)
        try:
            raise error
        except UnexpectedResponse:
            logger.exception("something failed")
        logger.warning("could not reach %s with %s", HOST, KEY)
        logger.info("Qdrant collection %r has no datetime index", COLLECTION)  # nothing to hide: kept as it is
    records = [r for r in caplog.records if r.name.startswith("rag_engine")]
    assert len(records) == 16 and {r.levelno for r in records} == {logging.ERROR, logging.WARNING, logging.INFO}
    for text in _record_texts(records):
        for leak in LEAKS:
            assert leak not in text, (leak, text)
    assert all("UnexpectedResponse" in r.getMessage() for r in records if r.levelno == logging.ERROR)
    kept = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert kept == [f"Qdrant collection {COLLECTION!r} has no datetime index"] * 4


def test_an_engine_record_reaches_the_root_handlers_once_reduced_whatever_its_logger(seeded):
    """As in a deployment: one handler on the root logger (the CLI's). A logger of the engine created after
    the filter was installed is covered too, and nothing is written twice or dropped."""
    store, fake = seeded
    EngineRagBackend(_settings(), store=store, embedder=fake)
    EngineRagBackend(_settings(), store=store, embedder=fake)  # built twice: still one relay

    class Collect(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    collect, root = Collect(), logging.getLogger()
    root.addHandler(collect)
    try:
        logging.getLogger("rag_engine.created_later").warning("failed: %s", _response(502))
        logging.getLogger("rag_engine.created_later").warning("plain line")
    finally:
        root.removeHandler(collect)
    assert [r.getMessage() for r in collect.records] == [
        ("rag-engine logged a message about an error (UnexpectedResponse); its text is not logged: it may hold "
         "the vector database's address, a response body or a credential hint"), "plain line"]
    relays = [h for h in logging.getLogger("rag_engine").handlers if type(h).__name__ == "_EngineLogRelay"]
    assert len(relays) == 1 and logging.getLogger("rag_engine").handlers[0] is relays[0]


@pytest.mark.parametrize("collection", ["qdrant-docs", "qdrant"])
def test_the_host_and_the_key_are_matched_as_whole_tokens_never_the_collection(seeded, caplog, collection):
    """A host named `qdrant` and a collection `qdrant-docs` (or `qdrant` itself): the engine's ordinary
    lines naming the collection are kept; the host itself, alone or with a port, is not written. The
    collection is exempt as the engine writes it, quoted (`%r`): `'qdrant'`."""
    store, fake = seeded
    EngineRagBackend(_settings(rag_qdrant_host="qdrant", rag_qdrant_api_key="k1", rag_collection=collection),
                     store=store, embedder=fake)
    caplog.set_level(logging.DEBUG)
    engine = logging.getLogger("rag_engine.store")
    kept = [(f"Qdrant collection {collection!r} has datetime indexes on date_start and date_end: date filters "
             "in auto mode use the intervals"), "talking to qdrant.example.org and qdrant-backup", "token k10 or xk1"]
    for line in kept:
        engine.info(line)
    for line in ("could not reach qdrant", "could not reach qdrant:6333", "url http://qdrant:6333/collections",
                 'host "qdrant" refused', "api-key=k1", "key k1."):
        engine.warning(line)
    messages = [r.getMessage() for r in caplog.records if r.name == "rag_engine.store"]
    assert messages[:3] == kept
    assert all(m.startswith("rag-engine logged a message about an error") for m in messages[3:]), messages[3:]
    assert len(messages) == 9


async def test_a_filter_the_collection_refuses_for_a_missing_index_is_search_refused(seeded, monkeypatch, caplog):
    """rag-engine raises its own MissingIndexError for the HTTP 400 of a collection in strict mode."""
    store, fake = seeded
    healthy = _Store(store, store.client)
    backend = EngineRagBackend(_settings(rag_date_filter_mode="interval"), store=healthy, embedder=fake)
    try:
        raise _response(400)
    except UnexpectedResponse as cause:
        refusal = MissingIndexError(f"collection {COLLECTION!r} refuses a filter on 'date_start'")
        refusal.__cause__ = cause
    _failing_for(monkeypatch, {QUERY: refusal})
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        assert await _error(c, {"query": QUERY, "date_from": "2016-01-01"}) == REFUSED
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        assert healthy.checks == 2  # the next call checked the collection first, and was served
    logged = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.rag"]
    assert "rag: a search was refused by the vector database (HTTP 400, a filtered field has no index)" in logged


INTERNAL = ("[rag_unavailable] internal_error: the server failed on this search for a reason of its own (its "
            "log names the error's class); retry, and tell the operator if it lasts")


@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING])
async def test_an_unexpected_error_is_coded_and_only_its_class_is_logged(seeded, monkeypatch, caplog, capfd,
                                                                          level):
    """A programming error, or anything the backend does not know: nothing of its text reaches the client, a
    log record of any logger, or stderr (FastMCP prints the traceback of an exception it receives)."""
    store, fake = seeded
    backend = EngineRagBackend(_settings(), store=_Store(store, store.client), embedder=fake)

    def failing(*args, **kwargs):
        raise RuntimeError(f"http://{HOST}:6333 api-key={KEY}")

    monkeypatch.setattr(backend_module, "search", failing)
    caplog.set_level(level)
    async with _client(backend) as c:
        message = await _error(c, {"query": "secret-term what did the council decide"})
        assert (await c.call_tool("server_info", {})).data["rag"]["status"] == "ready"  # not an outage
    assert message == INTERNAL
    texts = _record_texts(caplog.records) + list(capfd.readouterr())
    for text in texts:
        for leak in LEAKS:
            assert leak not in text, (leak, text)
    # FastMCP's own line for any tool error ("Error calling tool ...") holds the coded message only.
    errors = [r.getMessage() for r in caplog.records
              if r.levelno >= logging.ERROR and r.name.startswith("myopenfund")]
    assert errors == [("rag: a search failed with an unexpected error (RuntimeError); rag_search answered "
                       "[rag_unavailable] internal_error. Its text and traceback, which may hold the vector "
                       "database's address or a response body, are logged at DEBUG level only")]


def test_at_debug_level_a_developer_gets_the_traceback_of_an_unexpected_error(backend, monkeypatch, caplog):
    def failing(*args, **kwargs):
        raise RuntimeError("a programming error")

    monkeypatch.setattr(backend_module, "search", failing)
    caplog.set_level(logging.DEBUG, logger="myopenfund_mcp.rag")
    with pytest.raises(ToolError, match="internal_error"):
        backend.search(query="x", filters={}, limit=1, max_limit=5)
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info]
    assert len(debug) == 1 and "a programming error" in logging.Formatter().formatException(debug[0].exc_info)


async def test_a_reply_that_is_not_json_is_qdrant_error(seeded, monkeypatch, caplog, capfd):
    """A proxy in front of Qdrant answering a search with 200 and an HTML page: the client fails to decode
    it. Qdrant cannot be read, as with an error status: `qdrant_error`, and the next call checks the
    collection first."""
    store, fake = seeded
    healthy = _Store(store, store.client)
    backend = EngineRagBackend(_settings(), store=healthy, embedder=fake)
    _failing_for(monkeypatch, {QUERY: json.JSONDecodeError("Expecting value", NGINX_502.decode(), 0)})
    caplog.set_level(logging.DEBUG)
    async with _client(backend) as c:
        assert await _error(c, {"query": QUERY}) == (
            "[rag_unavailable] qdrant_error: the vector database answered with an error; retry later")
        assert backend._suspected == "qdrant_error"
        assert (await c.call_tool("rag_search", {"query": "x"})).data["count"] == CHUNKS
        assert healthy.checks == 2  # the next call checked the collection, and was served
    for text in _record_texts(caplog.records) + list(capfd.readouterr()):
        for leak in LEAKS:
            assert leak not in text, (leak, text)


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
                        "RAG_ENGINE_ONNX_THREADS": "64", "RAG_ENGINE_MODEL": "other/model",
                        "RAG_ENGINE_DATE_FILTER_MODE": "legacy", "RAG_ENGINE_DATE_FILTER_REFRESH": "1",
                        "RAG_ENGINE_DATE_FILTER_RETRY": "2"}.items():
        monkeypatch.setenv(name, value)
    seen = []
    monkeypatch.setattr(backend_module, "make_embedder", lambda s, **k: seen.append(("embedder", s)) or fake)
    monkeypatch.setattr(backend_module, "QdrantStore", lambda **k: seen.append(("store", k["settings"])) or store)
    settings = _settings(rag_qdrant_port=6334, rag_qdrant_https=True, rag_qdrant_timeout=7, rag_onnx_threads=3,
                         rag_default_limit=5, rag_max_limit=9, rag_date_filter_mode="interval",
                         rag_date_filter_refresh_s=60, rag_date_filter_retry_s=9)
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
    assert (engine.date_filter_mode, engine.date_filter_refresh, engine.date_filter_retry) == ("interval", 60, 9)
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
                 "date": "2022-01-01", "date_precision": "month", "date_kind": "publication",
                 "source_code": "ecb", "doc_type": "A1", "language": "en",
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


async def test_the_cap_counts_the_date_fields_and_the_citation_as_served():
    """The limit named by the refusal is the largest that fits with every field of a hit, the two date
    fields and the citation written to its precision included."""
    backend = _CannedBackend(["x" * 300] * 6)
    async with _client(backend, max_result_chars=2000) as c:
        fits = int((await _error(c, {"query": "q"})).rsplit("limit=", 1)[1].split()[0])
        served = await c.call_tool("rag_search", {"query": "q", "limit": fits})
        assert len(served.content[0].text) <= 2000
        hit = served.data["hits"][0]
        assert (hit["date_precision"], hit["date_kind"]) == ("month", "publication")
        assert hit["citation"] == "T0 (ecb A1, January 2022), p. 1"
        assert (await _error(c, {"query": "q", "limit": fits + 1})).startswith("[result_too_large]")


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
                   "no outbound connection", "MOF_RAG_DESCRIPTION_NOTES=", "`include_undated`",
                   "`date_precision`", "`date_kind`", "`July 2016`", "`sync-metadata`", "`date_start`",
                   "`date_end`", "#### Dates in `rag_search`", "No new volume"):
        assert needle in readme, needle


def test_the_readme_pins_rag_engine_to_the_commit_the_server_is_built_on():
    root = Path(__file__).resolve().parent.parent
    pins = re.findall(r"MyOpenFund/rag-engine@([0-9a-f]{40})", (root / "pyproject.toml").read_text())
    assert len(pins) == 1
    readme = (root / "README.md").read_text()
    assert re.findall(r"MyOpenFund/rag-engine(?:@([0-9a-f]+))?[\"`'\s]", readme) == pins  # one install line, pinned


def test_the_fill_command_says_when_it_is_done_in_the_readme_and_in_the_compose_example():
    root = Path(__file__).resolve().parent.parent
    readme, compose = (root / "README.md").read_text(), (root / "compose.example.yaml").read_text()
    assert ".release(); print('model cache ready')\"" in readme
    assert "release();\n#         print('model cache ready')\n" in compose
    assert "ONNX embedding model ready" not in readme + compose  # a line the fill never prints


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
