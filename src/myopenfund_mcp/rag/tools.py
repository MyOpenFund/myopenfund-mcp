"""The `rag` toolset: search the passages of the embedded central-bank corpus."""
from __future__ import annotations

from functools import partial
from typing import Annotated

import anyio
from fastmcp import FastMCP
from pydantic import Field, WithJsonSchema

from ..auth import tier_check
from ..config import Settings
from ..errors import INVALID_ARGUMENT, LIMIT_EXCEEDED, tool_error
from ..paging import cap_result, jsonable, result_chars
from .backend import RagBackend

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
MAX_QUERY_CHARS = 2000  # longer queries are refused by schema validation (the model reads 512 tokens)
MAX_CODE_CHARS = 64
MAX_CODES = 50
MAX_DATE_CHARS = 40  # room for a malformed date to reach the coded error instead of the schema's

SEARCH_DESCRIPTION = """Search central-bank monetary-policy documents by meaning and get the most relevant \
PASSAGES (text excerpts), best first. It returns source text only; it does not write an answer.

Arguments:
- query (required): a question or topic in plain language, e.g. "why did the ECB raise rates in July 2022".
- source_codes: list of central banks, exact lower-case codes: "ecb" (European Central Bank), "us" (Federal \
Reserve), "gb" (Bank of England), "au" (Reserve Bank of Australia), "ca" (Bank of Canada), "jp" (Bank of Japan).
- doc_types: list of document types, exact upper-case codes: "A1" (rate-decision press release), "A2" (policy \
statement), "A3" (minutes or accounts of the meeting), "B1" (press conference), "E1" (monetary policy report), \
"F1" (projections).
- languages: list of language codes as stored, e.g. ["en"].
- date_from, date_to: publication date bounds, "YYYY-MM-DD", both inclusive. With a bound set, undated \
documents are left out.
- limit: number of passages; omit for the server default.
Lists are JSON arrays, e.g. source_codes=["ecb","us"]. Values of one list are alternatives (OR); different \
arguments combine (AND). Omit an argument to leave it unrestricted; never send an empty list. A code the \
collection does not hold is refused with the list of valid codes, never corrected.

Result: {query, filters, count, hits}. Each hit: rank, score (cosine, higher is closer), text, doc_id, \
title, date, source_code, doc_type, language, page, chunk_number, citation. `page` is the page the passage \
starts on, counted from 1 (null for a document without pages). Quote `text` and cite each passage with \
its `citation` (title, date, page); several hits may come from the same document."""

QUERY_DESCRIPTION = "What to look for, in plain language (a question or a topic). Required."
SOURCE_CODES_DESCRIPTION = ('Central banks to keep, exact lower-case codes among "ecb", "us", "gb", "au", "ca", '
                            '"jp", e.g. ["ecb", "us"]; omitted = all')
DOC_TYPES_DESCRIPTION = ('Document types to keep, exact upper-case codes among "A1", "A2", "A3", "B1", "E1", '
                         '"F1", e.g. ["A3"]; omitted = all')
LANGUAGES_DESCRIPTION = 'Language codes to keep, exact, e.g. ["en"]; omitted = all'
DATE_FROM_DESCRIPTION = "Earliest publication date, YYYY-MM-DD, inclusive"
DATE_TO_DESCRIPTION = "Latest publication date, YYYY-MM-DD, inclusive"
LIMIT_DESCRIPTION = ("Number of passages; omitted = server default; a value above the server maximum is "
                     "refused, not lowered")

Code = Annotated[str, Field(max_length=MAX_CODE_CHARS)]
# Published as an array of strings and nothing else. A bare string (a frequent mistake of small models:
# "ecb" for ["ecb"]) is still let through to the engine, which refuses it with a coded message saying
# what to write, instead of the framework's uncoded schema error.
Codes = Annotated[
    Annotated[list[Code], Field(max_length=MAX_CODES)] | Annotated[str, Field(max_length=MAX_QUERY_CHARS)],
    WithJsonSchema({"type": "array", "items": {"type": "string", "maxLength": MAX_CODE_CHARS},
                    "maxItems": MAX_CODES}),
]
Day = Annotated[str, Field(max_length=MAX_DATE_CHARS)]


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

    @rag.tool(auth=read, annotations=READ_ONLY, description=SEARCH_DESCRIPTION)
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
        filters = {"source_codes": source_codes, "doc_types": doc_types, "languages": languages,
                   "date_from": date_from, "date_to": date_to}
        # The backend is blocking (model inference, HTTP to Qdrant): run it off the event loop.
        applied, hits = await anyio.to_thread.run_sync(
            partial(backend.search, query=query, filters=filters, limit=wanted, max_limit=maximum))
        return capped(query, applied, [shape_hit(rank, hit) for rank, hit in enumerate(hits, start=1)], settings)

    return rag
