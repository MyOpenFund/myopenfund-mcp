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
# vault_query.VOCABULARY_KINDS (a test keeps them equal): the column each declared code is for.
VocabularyKind = Literal["source_code", "doc_type", "provenance", "entity_scheme"]
GroupField = Literal["corpus", "source_code", "doc_type", "language", "year", "provenance", "mime_type"]
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
MAX_ARG_CHARS = 200  # doc_id, text and cursor: longer strings are refused by schema validation

_REFUSED = "a value outside [1, server maximum] is refused, not clamped"
LIMIT_DESCRIPTION = f"Rows per page; omitted = server default; {_REFUSED}"
STATS_LIMIT_DESCRIPTION = f"The maximum number of groups returned; omitted = server default; {_REFUSED}"
RUNS_LIMIT_DESCRIPTION = (f"Applies to each list (runs, source_runs, open_discovery_errors, source_health); "
                          f"omitted = server default; {_REFUSED}")
CORPUS_DESCRIPTION = "e.g. 'central-bank'"
SOURCE_CODE_DESCRIPTION = "Publisher code, e.g. 'us' (Fed), 'fr'"
TEXT_DESCRIPTION = "Case-insensitive substring of the title, matched literally (% and _ are ordinary characters)"
SORT_BY_DESCRIPTION = "Field the rows are ordered by"
SORT_DIR_DESCRIPTION = "asc (oldest / smallest first) or desc (newest / largest first)"
# doc_type codes belong to their corpus (documents contract): central-bank `A1` and company `A1` share a
# name, not a meaning. vault-query refuses doc_type without corpus with a message listing each corpus's codes.
DOC_TYPE_NEEDS_CORPUS = (
    "needs `corpus`: a code means something only within its corpus. When the corpus is not clear from "
    "the conversation, ask the user which corpus they mean (or infer it from context: a 10-K is a company "
    "filing, a rate decision a central-bank document); without corpus the call is refused with "
    "[invalid_argument] listing each corpus's codes and labels (vault_vocabulary lists them too)")
DOC_TYPE_DESCRIPTION = (f"Document-type code within its corpus, e.g. 'C1'; exact match; "
                        f"{DOC_TYPE_NEEDS_CORPUS}")
GROUP_BY_DESCRIPTION = f"Field the live documents are counted by; doc_type {DOC_TYPE_NEEDS_CORPUS}"
SINCE_DESCRIPTION = ("ISO 8601 timestamp (UTC if no offset; an offset is whole minutes, at most +/-15:59); "
                     "keeps runs FINISHED at or after it; runs still "
                     "in flight are left out when since is given; applies to `runs` and `source_runs` only")
# The largest UTC offset Postgres's timestamptz accepts (16:00 and beyond raise
# InvalidTimeZoneDisplacementValue); vault-query refuses the rest with a message naming its own grammar.
MAX_SINCE_OFFSET = dt.timedelta(hours=15, minutes=59)

RUNS_STATUS_DESCRIPTION = """Is the pipeline healthy? Four lists, newest first unless stated:
- `runs`: one row per run (run grain), with its outcome and totals. A run with no per-source detail
  (e.g. it crashed before reaching any source) appears only here.
- `source_runs`: one row per (run, source): counts, outcome and bounded error samples.
  `bad_content` is the part of `docs_failed` where the server sent the wrong bytes (e.g. an HTML page
  for a PDF), as opposed to a dead URL; NULL means not reported (another producer, an older run), not 0.
- `open_discovery_errors`: unresolved discovery failures, most recently seen first. While an error's
  `seen_at_is_ingestion_time` is true, its `last_seen_at` is approximate (the time it was ingested,
  not the time it was last observed).
- `source_health`: expected-vs-observed publication cadence per series, the least healthy first:
  `cadence_status` overdue, then soon, then the rest; within each, most `days_late` first (unknown
  last). Each row also carries `source_bad_content_7d` (sum over 7 days, NULL when no run of the window
  reported it) and `last_run_bad_content` (the last run's value). `source_health_total` counts
  every series matching corpus/source_code, so a smaller list means `limit` left some out.
`limit` applies to each of `runs`, `source_runs`, `open_discovery_errors` and `source_health`.
A result over the server's size cap is refused, never cut: ask for a smaller `limit`, or filter by
`source_code`."""
VOCABULARY_DESCRIPTION = """What the codes mean: the codes each corpus declares, with their labels, as rows
{corpus, kind, code, label, family} ordered by corpus, kind, code. `kind` is the column a code is for
(doc_type, source_code, provenance, entity_scheme); `family` groups doc_types when the corpus declares
families. A code means something only within its corpus (central-bank A1 and company A1 are different
document types): use this to turn the user's words (e.g. "10-K", "rate decision") into the doc_type of
a corpus before filtering vault_search_documents. A corpus that has not declared its codes yet has no
rows: vault_corpus_stats(group_by="doc_type", corpus=...) shows the codes its documents use.
`truncated` true means `limit` cut the list: filter by corpus and kind."""
GET_DOCUMENT_DESCRIPTION = """Every stored fact of a document: dates (`date`, with `date_precision` and
`date_kind`), the document's URL and the page it was found on, publisher, type, issuing entity
(`entity_scheme`, `entity_value`, e.g. cik), `extra`.
A doc_id is unique within its corpus only, so this returns a LIST: with `corpus`, at most one
document (the one of that corpus); without `corpus`, one per corpus holding this doc_id, ordered by
corpus, each carrying its `corpus`. Pass the `corpus` of the search row you came from.
No document: [not_found]."""
# Rank of a `source_health.cadence_status` in the list (lower first); any other value, NULL included,
# ranks after these.
HEALTH_STATUS_RANK = {"overdue": 0, "soon": 1}


def _least_healthy_first(rows: list[dict]) -> list[dict]:
    """`source_health` rows ordered as RUNS_STATUS_DESCRIPTION says; a stable sort, so ties keep the
    view's own order (corpus, source_code, doc_type)."""
    def key(row: dict) -> tuple:
        days_late = row.get("days_late")
        return (HEALTH_STATUS_RANK.get(row.get("cadence_status"), len(HEALTH_STATUS_RANK)),
                days_late is None, -(days_late or 0))
    return sorted(rows, key=key)


async def _documents(reader: VaultReader, doc_id: str, corpus: str | None) -> list[dict]:
    """vault-query's list (one row per corpus holding doc_id, or the one in `corpus`); none is not_found."""
    rows = await _call(reader.get_document, doc_id=doc_id, corpus=corpus)
    if not rows:
        where = "" if corpus is None else f" in corpus {corpus!r}; omit corpus to look in every corpus"
        raise tool_error(NOT_FOUND, f"no document {doc_id!r}{where}; find ids with vault_search_documents")
    return rows


def _documents_cap_hint(rows: list[dict], *, via: str) -> str:
    """How to get a smaller vault_get_document result: one corpus, when the list holds several; else
    nothing smaller exists (the tool has no limit)."""
    if len(rows) > 1:
        return f"{via} pass `corpus` (one of {sorted({r.get('corpus') for r in rows})}) to get one document"
    return "this one document is larger than MOF_MCP_MAX_RESULT_CHARS; the operator can raise it"


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
        doc_type: Annotated[str | None, Field(description=DOC_TYPE_DESCRIPTION)] = None,
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
        vault_get_document. Read `date` with `date_precision` (day, month or year; null = not stated:
        a year-precision date is the period's first day, not a known day) and `date_kind` (publication,
        meeting or period); `entity_scheme` / `entity_value` name the issuing entity (e.g. cik).
        Rows are ordered by `sort_by` in `sort_dir`; documents with equal sort values are ordered by
        doc_id ascending, whatever `sort_dir`. That order is fixed within a corpus only: two corpora
        can hold the same doc_id, and two such rows with equal sort values may swap between calls, so
        a page boundary between them can repeat one and skip the other; filter by corpus when paging
        matters. `total` counts the whole filtered set; pass `next_cursor` back (same filters) to get
        the next page. Cursors are offsets: if the vault is ingesting between two pages, a row can
        shift (seen twice or skipped).
        A `doc_type` filter needs `corpus`: ask the user which corpus when the context does not say.
        Open a row with vault_get_document(doc_id, corpus) using the row's own `corpus`."""
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

    @vault.tool(auth=read, annotations=READ_ONLY, description=GET_DOCUMENT_DESCRIPTION)
    async def get_document(
        doc_id: Annotated[str, Field(description="doc_id from vault_search_documents",
                                     max_length=MAX_ARG_CHARS)],
        corpus: Annotated[str | None, Field(
            description="The document's corpus (the `corpus` of its search row), e.g. 'central-bank'; "
                        "omitted = every corpus holding this doc_id", max_length=MAX_ARG_CHARS)] = None,
    ) -> list[dict]:
        rows = await _documents(reader, doc_id, corpus)
        return cap_result(rows, settings, hint=_documents_cap_hint(rows, via="call again and"))

    @vault.tool(auth=read, annotations=READ_ONLY)
    async def corpus_stats(
        group_by: Annotated[GroupField, Field(description=GROUP_BY_DESCRIPTION)] = "source_code",
        corpus: Annotated[str | None, Field(
            description="Count only this corpus, e.g. 'central-bank'; omitted = every corpus")] = None,
        limit: Annotated[int | None, Field(description=STATS_LIMIT_DESCRIPTION)] = None,
    ) -> dict:
        """Live-document counts grouped by one field, largest first (NULL is a real group).
        Grouping by doc_type needs `corpus`: ask the user which corpus when the context does not say."""
        return cap_result(await _call(reader.corpus_stats, group_by=group_by, corpus=corpus,
                                      limit=clamp_limit(limit, settings)), settings)

    @vault.tool(auth=read, annotations=READ_ONLY, description=VOCABULARY_DESCRIPTION)
    async def vocabulary(
        corpus: Annotated[str | None, Field(description=f"{CORPUS_DESCRIPTION}; omitted = every corpus",
                                            max_length=MAX_ARG_CHARS)] = None,
        kind: Annotated[VocabularyKind | None, Field(
            description="Only the codes of this column; omitted = every kind")] = None,
        limit: Annotated[int | None, Field(description=LIMIT_DESCRIPTION)] = None,
    ) -> dict:
        page = clamp_limit(limit, settings)
        # One row more than the page tells a full list from a cut one (vault-query has no total).
        rows = await _call(reader.list_vocabulary, corpus=corpus, kind=kind, limit=page + 1)
        return cap_result({"returned": min(len(rows), page), "truncated": len(rows) > page,
                           "codes": rows[:page]}, settings, hint="filter by corpus and kind")

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
        # The pinned vault-query has no limit on source_health: the view is small (one row per series),
        # so it is read whole, ordered here and cut to `limit`, with its total.
        health = await _call(reader.source_health, corpus=corpus, source_code=source_code)
        return cap_result({"runs": runs, "source_runs": source_runs, "open_discovery_errors": errors,
                           "source_health": _least_healthy_first(health)[:page],
                           "source_health_total": len(health)},
                          settings, hint="ask for a smaller limit, or filter by source_code")

    return vault


def register_vault_resources(root: FastMCP, reader: VaultReader, settings: Settings, *,
                             auth_enabled: bool) -> None:
    """On the root server: a mounted namespace would rewrite the URI to vault://vault/..."""

    @root.resource("vault://documents/{doc_id}", mime_type="application/json",
                   auth=tier_check("read", auth_enabled=auth_enabled))
    async def document(doc_id: str) -> str:
        """The metadata of the documents with this doc_id, one per corpus holding it, as a JSON list
        (same content as vault_get_document without corpus)."""
        rows = await _call(reader.get_document, doc_id=doc_id, corpus=None)
        if not rows:
            raise ResourceError(f"[{NOT_FOUND}] no document {doc_id!r}")
        # The exact string the cap measured (paging.result_chars basis).
        hint = _documents_cap_hint(rows, via="call vault_get_document and")
        return json.dumps(cap_result(rows, settings, hint=hint), ensure_ascii=False, separators=(",", ":"))
