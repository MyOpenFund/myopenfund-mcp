"""The `vault` toolset: find and read the official documents, check the pipeline's health."""
from __future__ import annotations

import datetime as dt
import json
from functools import partial
from typing import Annotated, Literal

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError
from pydantic import Field

from ..auth import tier_check
from ..config import Settings
from ..errors import INVALID_ARGUMENT, NOT_FOUND, tool_error
from ..paging import MAX_OFFSET, cap_result, clamp_limit, decode_cursor, encode_cursor
from .reader import VaultReader

SortField = Literal["date", "year", "corpus", "source_code", "doc_type", "title", "created_at"]
GroupField = Literal["corpus", "source_code", "doc_type", "language", "year", "provenance", "mime_type"]
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
MAX_ARG_CHARS = 200  # doc_id, text and cursor: longer strings are refused by schema validation

_REFUSED = "a value outside [1, server maximum] is refused, not clamped"
LIMIT_DESCRIPTION = f"Rows per page; omitted = server default; {_REFUSED}"
STATS_LIMIT_DESCRIPTION = f"The maximum number of groups returned; omitted = server default; {_REFUSED}"
RUNS_LIMIT_DESCRIPTION = (f"Applies to each list (runs, source_runs, open_discovery_errors); "
                          f"omitted = server default; {_REFUSED}")
CORPUS_DESCRIPTION = "e.g. 'central-bank'"
SOURCE_CODE_DESCRIPTION = "Publisher code, e.g. 'us' (Fed), 'fr'"
TEXT_DESCRIPTION = "Case-insensitive substring of the title, matched literally (% and _ are ordinary characters)"
SORT_BY_DESCRIPTION = "Field the rows are ordered by"
SORT_DIR_DESCRIPTION = "asc (oldest / smallest first) or desc (newest / largest first)"
GROUP_BY_DESCRIPTION = "Field the live documents are counted by"
SINCE_DESCRIPTION = ("ISO 8601 timestamp (UTC if no offset; an offset is whole minutes, at most +/-15:59); "
                     "keeps runs FINISHED at or after it; runs still "
                     "in flight are left out when since is given; applies to `runs` and `source_runs` only")
# The largest UTC offset Postgres's timestamptz accepts (16:00 and beyond raise
# InvalidTimeZoneDisplacementValue); vault-query refuses the rest with a message naming its own grammar.
MAX_SINCE_OFFSET = dt.timedelta(hours=15, minutes=59)

RUNS_STATUS_DESCRIPTION = """Is the pipeline healthy? Four lists, newest first:
- `runs`: one row per run (run grain), with its outcome and totals. A run with no per-source detail
  (e.g. it crashed before reaching any source) appears only here.
- `source_runs`: one row per (run, source): counts, outcome and bounded error samples.
- `open_discovery_errors`: unresolved discovery failures, most recently seen first. While an error's
  `seen_at_is_ingestion_time` is true, its `last_seen_at` is approximate (the time it was ingested,
  not the time it was last observed).
- `source_health`: expected-vs-observed publication cadence per series.
`limit` applies to each of `runs`, `source_runs` and `open_discovery_errors`."""


async def _call(fn, /, **kwargs):
    """Readers are blocking (psycopg2): run them off the event loop."""
    return await anyio.to_thread.run_sync(partial(fn, **kwargs))


def build_vault_toolset(reader: VaultReader, settings: Settings, *, auth_enabled: bool) -> FastMCP:
    vault = FastMCP("vault", mask_error_details=True)
    read = tier_check("read", auth_enabled=auth_enabled)

    @vault.tool(auth=read, annotations=READ_ONLY)
    async def search_documents(
        corpus: Annotated[str | None, Field(description=CORPUS_DESCRIPTION)] = None,
        source_code: Annotated[str | None, Field(description=SOURCE_CODE_DESCRIPTION)] = None,
        doc_type: Annotated[str | None, Field(
            description="Publisher's document-type code, e.g. 'C1'; exact match")] = None,
        language: Annotated[str | None, Field(description="Language code, e.g. 'en', 'fr'; exact match")] = None,
        year: Annotated[int | None, Field(description="Publication year, e.g. 2010")] = None,
        mime_type: Annotated[str | None, Field(
            description="e.g. 'application/pdf', 'text/html'; exact match")] = None,
        date_from: Annotated[dt.date | None, Field(description="YYYY-MM-DD, inclusive")] = None,
        date_to: Annotated[dt.date | None, Field(description="YYYY-MM-DD, inclusive")] = None,
        text: Annotated[str | None, Field(description=TEXT_DESCRIPTION, max_length=MAX_ARG_CHARS)] = None,
        sort_by: Annotated[SortField, Field(description=SORT_BY_DESCRIPTION)] = "date",
        sort_dir: Annotated[Literal["asc", "desc"], Field(description=SORT_DIR_DESCRIPTION)] = "desc",
        limit: Annotated[int | None, Field(description=LIMIT_DESCRIPTION)] = None,
        cursor: Annotated[str | None, Field(description="next_cursor of the previous page, unchanged",
                                            max_length=MAX_ARG_CHARS)] = None,
        include_deleted: Annotated[bool, Field(
            description="Also return withdrawn documents (tombstones, with `deleted_at` set)")] = False,
    ) -> dict:
        """Find official documents (metadata, not text). Start here: filter, then open one with
        vault_get_document. Rows are ordered by `sort_by` in `sort_dir`; documents with equal sort
        values are always ordered by doc_id ascending, whatever `sort_dir`. `total` counts the
        whole filtered set; pass `next_cursor` back (same filters) to get the next page. Cursors are
        offsets: if the vault is ingesting between two pages, a row can shift (seen twice or skipped)."""
        page = clamp_limit(limit, settings)
        offset = decode_cursor(cursor)
        if text is not None and "\x00" in text:  # vault-query would name it `q`, its own name
            raise tool_error(INVALID_ARGUMENT, "text must not contain a NUL character")
        if date_from is not None and date_to is not None and date_from > date_to:
            raise tool_error(INVALID_ARGUMENT, "date_from is after date_to")
        filters = {"corpus": corpus, "source_code": source_code, "doc_type": doc_type,
                   "language": language, "year": year, "mime_type": mime_type,
                   "date_from": date_from, "date_to": date_to, "q": text}
        total, rows = await _call(reader.search_documents, filters=filters, sort_by=sort_by,
                                  sort_dir=sort_dir, limit=page, offset=offset,
                                  include_deleted=include_deleted)
        end = offset + len(rows)
        # A cursor never carries an offset the next call would refuse (int4 bound).
        next_cursor = encode_cursor(end) if rows and end < total and end <= MAX_OFFSET else None
        return cap_result({"total": total, "returned": len(rows), "next_cursor": next_cursor,
                           "documents": rows}, settings)

    @vault.tool(auth=read, annotations=READ_ONLY)
    async def get_document(doc_id: Annotated[str, Field(
            description="doc_id from vault_search_documents", max_length=MAX_ARG_CHARS)]) -> dict:
        """Every stored fact of one document: dates, source and PDF URLs, publisher, type, `extra`."""
        row = await _call(reader.get_document, doc_id=doc_id)
        if row is None:
            raise tool_error(NOT_FOUND, f"no document {doc_id!r}; find ids with vault_search_documents")
        return cap_result(row, settings)

    @vault.tool(auth=read, annotations=READ_ONLY)
    async def corpus_stats(
        group_by: Annotated[GroupField, Field(description=GROUP_BY_DESCRIPTION)] = "source_code",
        corpus: Annotated[str | None, Field(
            description="Count only this corpus, e.g. 'central-bank'; omitted = every corpus")] = None,
        limit: Annotated[int | None, Field(description=STATS_LIMIT_DESCRIPTION)] = None,
    ) -> dict:
        """Live-document counts grouped by one field, largest first (NULL is a real group)."""
        return cap_result(await _call(reader.corpus_stats, group_by=group_by, corpus=corpus,
                                      limit=clamp_limit(limit, settings)), settings)

    @vault.tool(auth=read, annotations=READ_ONLY, description=RUNS_STATUS_DESCRIPTION)
    async def runs_status(
        corpus: Annotated[str | None, Field(description=CORPUS_DESCRIPTION)] = None,
        source_code: Annotated[str | None, Field(description=SOURCE_CODE_DESCRIPTION)] = None,
        since: Annotated[dt.datetime | None, Field(description=SINCE_DESCRIPTION)] = None,
        limit: Annotated[int | None, Field(description=RUNS_LIMIT_DESCRIPTION)] = None,
    ) -> dict:
        page = clamp_limit(limit, settings)
        if since is not None and since.tzinfo is None:
            since = since.replace(tzinfo=dt.UTC)  # a naive since is UTC (SINCE_DESCRIPTION)
        if since is not None:
            utc_offset = since.utcoffset()
            if abs(utc_offset) > MAX_SINCE_OFFSET or utc_offset % dt.timedelta(minutes=1):
                raise tool_error(INVALID_ARGUMENT, "since: the UTC offset must be whole minutes, at most "
                                                   "+/-15:59 (the database's limit)")
        runs = await _call(reader.recent_runs, corpus=corpus, since=since, limit=page)
        source_runs = await _call(reader.runs_status, corpus=corpus, source_code=source_code, since=since,
                                  limit=page, max_error_samples=settings.max_error_samples,
                                  max_text_chars=settings.max_text_chars)
        errors = await _call(reader.open_discovery_errors, corpus=corpus, source_code=source_code,
                             limit=page, max_text_chars=settings.max_text_chars)
        health = await _call(reader.source_health, corpus=corpus, source_code=source_code)
        return cap_result({"runs": runs, "source_runs": source_runs, "open_discovery_errors": errors,
                           "source_health": health}, settings)

    return vault


def register_vault_resources(root: FastMCP, reader: VaultReader, settings: Settings, *,
                             auth_enabled: bool) -> None:
    """On the root server: a mounted namespace would rewrite the URI to vault://vault/..."""

    @root.resource("vault://documents/{doc_id}", mime_type="application/json",
                   auth=tier_check("read", auth_enabled=auth_enabled))
    async def document(doc_id: str) -> str:
        """One document's metadata as JSON (same content as vault_get_document)."""
        row = await _call(reader.get_document, doc_id=doc_id)
        if row is None:
            raise ResourceError(f"[{NOT_FOUND}] no document {doc_id!r}")
        # The exact string the cap measured (paging.result_chars basis).
        return json.dumps(cap_result(row, settings), ensure_ascii=False, separators=(",", ":"))
