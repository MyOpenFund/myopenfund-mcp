"""The `rag` toolset: search the passages of the embedded corpus.

One call has ONE time budget (`rag_call_timeout_s`), its wait for a free slot included. At most
`rag_max_concurrency` searches run at once, in threads of the toolset's own: the calls in excess wait
in the event loop, and a hanging Qdrant takes no thread from the other toolsets. A call whose
budget ends answers `[rag_unavailable]`: `busy` when it never got a slot, `timeout` when its search
was still running. That search keeps its slot until it really ends (one request to Qdrant never
outlasts `rag_qdrant_timeout`), so abandoned threads cannot pile up.

The arguments a client gets wrong most often are checked here, before anything else, and refused
in the terms of the JSON the client must send.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Annotated

import anyio
from fastmcp import FastMCP
from pydantic import Field, WithJsonSchema

from ..auth import tier_check
from ..config import Settings
from ..errors import INVALID_ARGUMENT, LIMIT_EXCEEDED, tool_error
from ..paging import cap_result, jsonable, result_chars
from .backend import RagBackend, unavailable

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
MAX_QUERY_CHARS = 2000  # longer queries are refused by schema validation (the model reads 512 tokens)
MAX_CODE_CHARS = 64
MAX_CODES = 50
MAX_DATE_CHARS = 40  # room for a malformed date to reach the coded error instead of the schema's
CODE_ARGUMENTS = ("source_codes", "doc_types", "languages")
DATE_ARGUMENTS = ("date_from", "date_to")

# Corpus-neutral on purpose: what the collection holds and what its codes mean is the deployment's to
# say, in MOF_RAG_DESCRIPTION_NOTES (appended below) -- the codes themselves are read from the collection.
SEARCH_DESCRIPTION = """Search the embedded document collection by meaning and get the most relevant \
PASSAGES (text excerpts), best first. It returns source text only; it does not write an answer.

Arguments:
- query (required): a question or topic in plain language.
- source_codes, doc_types, languages: JSON arrays of exact codes (case matters), e.g. \
source_codes=["<code>"]. The valid codes are listed in server_info (rag.source_codes, rag.doc_types, \
rag.languages) and in the refusal message: a code the collection does not hold is refused with the valid \
ones, never corrected.
- date_from, date_to: publication date bounds, "YYYY-MM-DD", both inclusive. With a bound set, undated \
documents are left out. Date precision varies by bank (some documents carry only the year or the month): \
if a date range returns nothing, retry with whole-year bounds or without dates.
- limit: number of passages, from 1 to {max_limit}; omitted = {default_limit}.
Values of one array are alternatives (OR); different arguments combine (AND). Omit an argument to leave it \
unrestricted; never send an empty array.

Result: {{query, filters, count, hits}}. Each hit: rank, score (cosine, higher is closer), text, doc_id, \
title, date, source_code, doc_type, language, page, chunk_number, citation. `page` is the page the passage \
starts on, counted from 1 (null for a document without pages). Quote `text` and cite each passage with \
its `citation` (title, date, page); several hits may come from the same document."""

QUERY_DESCRIPTION = "What to look for, in plain language (a question or a topic). Required."
SOURCE_CODES_DESCRIPTION = ('Sources to keep: JSON array of exact codes among server_info\'s rag.source_codes, '
                            'e.g. ["<code>"]; omitted = all')
DOC_TYPES_DESCRIPTION = ('Document types to keep: JSON array of exact codes among server_info\'s rag.doc_types, '
                         'e.g. ["<code>"]; omitted = all')
LANGUAGES_DESCRIPTION = ('Languages to keep: JSON array of exact codes among server_info\'s rag.languages, '
                         'e.g. ["en"]; omitted = all')
DATE_FROM_DESCRIPTION = "Earliest publication date, YYYY-MM-DD, inclusive"
DATE_TO_DESCRIPTION = "Latest publication date, YYYY-MM-DD, inclusive"
LIMIT_DESCRIPTION = ("Number of passages; omitted = server default; a value above the server maximum is "
                     "refused, not lowered")

Code = Annotated[str, Field(max_length=MAX_CODE_CHARS)]
# Published as an array of strings and nothing else. A bare string (a frequent mistake of small models:
# "ecb" or '["ecb"]' for ["ecb"]) is still let through to the tool, which refuses it with a coded message
# saying what to write, instead of the framework's uncoded schema error.
Codes = Annotated[
    Annotated[list[Code], Field(max_length=MAX_CODES)] | Annotated[str, Field(max_length=MAX_QUERY_CHARS)],
    WithJsonSchema({"type": "array", "items": {"type": "string", "maxLength": MAX_CODE_CHARS},
                    "maxItems": MAX_CODES}),
]
Day = Annotated[str, Field(max_length=MAX_DATE_CHARS)]

_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_CODE_IN_A_STRING = re.compile(r"""[^\s,\[\]'"]+""")
MAX_EXAMPLE_CODES = 3


def search_description(settings: Settings) -> str:
    """The tool's description: the limits of this deployment, then its notes on the collection."""
    text = SEARCH_DESCRIPTION.format(max_limit=settings.rag_max_limit, default_limit=settings.rag_default_limit)
    notes = settings.rag_description_notes.strip()
    return f"{text}\n\nAbout this collection: {notes}" if notes else text


def _array_example(value: str) -> str:
    """The JSON array a client that sent the string `value` meant, as an example of what to write:
    '["ecb"]' for `ecb` as well as for the array already written as a string, `["ecb"]`."""
    codes = [code for code in _CODE_IN_A_STRING.findall(value) if len(code) <= MAX_CODE_CHARS]
    return json.dumps(codes[:MAX_EXAMPLE_CODES] or ["code"], ensure_ascii=False)


def _searchable(text: str) -> bool:
    """rag-engine's rule: at least one letter, number, symbol or punctuation mark."""
    return any(unicodedata.category(character)[0] in "LNSP" for character in text)


def check_arguments(query: str, codes: dict, dates: dict) -> None:
    """Refuse, in the client's terms (JSON, never Python), what rag-engine would refuse in its own."""
    if not query.strip():
        raise tool_error(INVALID_ARGUMENT, "query must be a non-blank string: a question or a topic")
    if not (_searchable(query) and _searchable(unicodedata.normalize("NFKD", query))):
        raise tool_error(INVALID_ARGUMENT, "query has no searchable character: it must hold at least one "
                                           "letter, number, symbol or punctuation mark")
    for name, value in codes.items():
        if value is None:
            continue
        if isinstance(value, str):
            raise tool_error(INVALID_ARGUMENT, f"{name} must be a JSON array of strings, e.g. "
                                               f"{_array_example(value)}; omit it for no restriction")
        if not value:
            raise tool_error(INVALID_ARGUMENT, f'{name} is an empty array, which could mean "match nothing" as '
                                               'well as "no restriction": omit the argument for no restriction')
        if not all(item.strip() for item in value):
            raise tool_error(INVALID_ARGUMENT, f"{name} must hold non-blank strings")
    days = {}
    for name, value in dates.items():
        if value is None:
            continue
        if _DATE.fullmatch(value) is None:
            raise tool_error(INVALID_ARGUMENT, f'{name} must be "YYYY-MM-DD", e.g. "2016-03-10"')
        try:
            days[name] = dt.date.fromisoformat(value)
        except ValueError:
            raise tool_error(INVALID_ARGUMENT, f"{name} is not a date of the calendar: "
                                               f"{json.dumps(value)}") from None
    if len(days) == 2 and days["date_from"] > days["date_to"]:
        raise tool_error(INVALID_ARGUMENT, f"date_from ({dates['date_from']}) is after date_to ({dates['date_to']})")


def human_page(page) -> int | None:
    """The page as a reader counts it: stored pages are counted from 0. None without a usable page."""
    return page + 1 if type(page) is int and page >= 0 else None


def citation(hit: dict) -> str:
    """`<title> (<source_code> <doc_type>, <date>), p. <page>`, each missing part left out."""
    label = hit.get("title") or hit.get("doc_id") or "untitled document"
    origin = " ".join(str(part) for part in (hit.get("source_code"), hit.get("doc_type")) if part)
    details = ", ".join(str(part) for part in (origin, hit.get("date")) if part)
    page = human_page(hit.get("page"))
    return f"{label}{f' ({details})' if details else ''}{f', p. {page}' if page is not None else ''}"


def shape_hit(rank: int, hit: dict) -> dict:
    """One hit as the client sees it. Never `point_id` (a 64-bit integer, which JSON clients in
    JavaScript cannot hold) nor the whole payload."""
    return {"rank": rank, "score": round(float(hit["score"]), 4), "text": hit.get("text"),
            "doc_id": hit.get("doc_id"), "title": hit.get("title"), "date": hit.get("date"),
            "source_code": hit.get("source_code"), "doc_type": hit.get("doc_type"),
            "language": hit.get("language"), "page": human_page(hit.get("page")),
            "chunk_number": hit.get("chunk_number"), "citation": citation(hit)}


def _result(query: str, filters: dict, hits: list[dict]) -> dict:
    return {"query": query, "filters": filters, "count": len(hits), "hits": hits}


def capped(query: str, filters: dict, hits: list[dict], settings: Settings) -> dict:
    """The result, or `[result_too_large]` naming the largest limit that fits: refused, never cut."""
    payload = jsonable(_result(query, filters, hits))
    hint = "ask for a smaller limit"
    if result_chars(payload) > settings.max_result_chars:
        fits = next((n for n in range(len(hits) - 1, 0, -1)
                     if result_chars(_result(query, filters, payload["hits"][:n])) <= settings.max_result_chars), 0)
        hint = (f"ask again with limit={fits} or less" if fits else
                "even the first passage does not fit; narrow the query or ask the operator to raise the cap")
    return cap_result(payload, settings, hint=hint)


def build_rag_toolset(backend: RagBackend, settings: Settings, *, auth_enabled: bool) -> FastMCP:
    rag = FastMCP("rag", mask_error_details=True)
    read = tier_check("read", auth_enabled=auth_enabled)
    budget = settings.rag_call_timeout_s
    # The toolset's own threads, never more than the limit: a search that outlives its call (a hanging
    # Qdrant) keeps one until it ends, and the calls waiting for one wait in the event loop.
    workers = ThreadPoolExecutor(max_workers=settings.rag_max_concurrency, thread_name_prefix="mof-rag")

    @rag.tool(auth=read, annotations=READ_ONLY, description=search_description(settings))
    async def search(
        query: Annotated[str, Field(description=QUERY_DESCRIPTION, max_length=MAX_QUERY_CHARS)],
        source_codes: Annotated[Codes | None, Field(description=SOURCE_CODES_DESCRIPTION)] = None,
        doc_types: Annotated[Codes | None, Field(description=DOC_TYPES_DESCRIPTION)] = None,
        date_from: Annotated[Day | None, Field(description=DATE_FROM_DESCRIPTION)] = None,
        date_to: Annotated[Day | None, Field(description=DATE_TO_DESCRIPTION)] = None,
        languages: Annotated[Codes | None, Field(description=LANGUAGES_DESCRIPTION)] = None,
        limit: Annotated[int | None, Field(description=LIMIT_DESCRIPTION)] = None,
    ) -> dict:
        maximum = settings.rag_max_limit
        wanted = settings.rag_default_limit if limit is None else limit
        if wanted < 1:
            raise tool_error(INVALID_ARGUMENT, f"limit must be an integer between 1 and {maximum}")
        if wanted > maximum:
            raise tool_error(LIMIT_EXCEEDED, f"limit {wanted} is above the maximum of {maximum}: "
                                             f"ask for at most {maximum} passages")
        codes = {"source_codes": source_codes, "doc_types": doc_types, "languages": languages}
        dates = {"date_from": date_from, "date_to": date_to}
        check_arguments(query, codes, dates)
        # The backend is blocking (model inference, HTTP to Qdrant): it runs off the event loop.
        job = workers.submit(partial(backend.search, query=query, filters={**codes, **dates}, limit=wanted,
                                     max_limit=maximum, deadline=time.monotonic() + budget))
        with anyio.move_on_after(budget) as scope:
            applied, hits = await asyncio.wrap_future(job)
        if scope.cancelled_caught:
            # cancel() succeeds only for a search that has not started: it never will.
            raise unavailable("busy" if job.cancel() else "timeout")
        return capped(query, applied, [shape_hit(rank, hit) for rank, hit in enumerate(hits, start=1)], settings)

    return rag
