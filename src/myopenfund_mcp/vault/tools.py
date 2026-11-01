"""The `vault` toolset: find and read the official documents, check the pipeline's health."""
from __future__ import annotations

import datetime as dt
import json
from functools import partial
from typing import Annotated, Literal

import anyio
from fastmcp import FastMCP
from pydantic import Field

from ..auth import tier_check
from ..config import Settings
from ..errors import NOT_FOUND, tool_error
from ..paging import cap_result, clamp_limit, decode_cursor, encode_cursor
from .reader import VaultReader

SortField = Literal["date", "year", "corpus", "source_code", "doc_type", "title", "created_at"]
GroupField = Literal["corpus", "source_code", "doc_type", "language", "year", "provenance", "mime_type"]
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}

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
    vault = FastMCP("vault")
    read = tier_check("read", auth_enabled=auth_enabled)

    @vault.tool(auth=read, annotations=READ_ONLY)
    async def search_documents(
        corpus: Annotated[str | None, Field(description="e.g. 'central-bank'")] = None,
        source_code: Annotated[str | None, Field(description="Publisher code, e.g. 'us' (Fed), 'fr'")] = None,
        doc_type: str | None = None,
        language: str | None = None,
        year: int | None = None,
        mime_type: str | None = None,
        date_from: Annotated[dt.date | None, Field(description="YYYY-MM-DD, inclusive")] = None,
        date_to: Annotated[dt.date | None, Field(description="YYYY-MM-DD, inclusive")] = None,
        text: Annotated[str | None, Field(description="Case-insensitive match on the title")] = None,
        sort_by: SortField = "date",
        sort_dir: Literal["asc", "desc"] = "desc",
        limit: Annotated[int | None, Field(description="Rows per page (server default and maximum apply)")] = None,
        cursor: Annotated[str | None, Field(description="next_cursor of the previous page")] = None,
        include_deleted: bool = False,
    ) -> dict:
        """Find official documents (metadata, not text). Start here: filter, then open one with
        vault_get_document. Rows are ordered by `sort_by` in `sort_dir`; documents with equal sort
        values are always ordered by doc_id ascending, whatever `sort_dir`. `total` counts the
        whole filtered set; pass `next_cursor` back (same filters) to get the next page."""
        page = clamp_limit(limit, settings)
        offset = decode_cursor(cursor)
        filters = {"corpus": corpus, "source_code": source_code, "doc_type": doc_type,
                   "language": language, "year": year, "mime_type": mime_type,
                   "date_from": date_from, "date_to": date_to, "q": text}
        total, rows = await _call(reader.search_documents, filters=filters, sort_by=sort_by,
                                  sort_dir=sort_dir, limit=page, offset=offset,
                                  include_deleted=include_deleted)
        end = offset + len(rows)
        return cap_result({"total": total, "returned": len(rows),
                           "next_cursor": encode_cursor(end) if rows and end < total else None,
                           "documents": rows}, settings)

    @vault.tool(auth=read, annotations=READ_ONLY)
    async def get_document(doc_id: str) -> dict:
        """Every stored fact of one document: dates, source and PDF URLs, publisher, type, `extra`."""
        row = await _call(reader.get_document, doc_id=doc_id)
        if row is None:
            raise tool_error(NOT_FOUND, f"no document {doc_id!r}; find ids with vault_search_documents")
        return cap_result(row, settings)

    @vault.tool(auth=read, annotations=READ_ONLY)
    async def corpus_stats(group_by: GroupField = "source_code", corpus: str | None = None,
                           limit: int | None = None) -> dict:
        """Live-document counts grouped by one field, largest first (NULL is a real group)."""
        return cap_result(await _call(reader.corpus_stats, group_by=group_by, corpus=corpus,
                                      limit=clamp_limit(limit, settings)), settings)

    @vault.tool(auth=read, annotations=READ_ONLY, description=RUNS_STATUS_DESCRIPTION)
    async def runs_status(corpus: str | None = None, source_code: str | None = None,
                          since: Annotated[dt.datetime | None, Field(
                              description="ISO 8601; only runs started at or after it")] = None,
                          limit: int | None = None) -> dict:
        page = clamp_limit(limit, settings)
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
            raise tool_error(NOT_FOUND, f"no document {doc_id!r}")
        return json.dumps(cap_result(row, settings), ensure_ascii=False)
