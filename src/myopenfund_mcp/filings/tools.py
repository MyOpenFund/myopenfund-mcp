"""The filing-text tools, mounted in the `companies` namespace: fetch, search and read SEC filings.

`companies_fetch_filings` lists a company's filings at SEC (the source router's live source) and queues
them in the background indexer (`filings.indexer`); `companies_search_filings` and
`companies_read_filing` read what the indexer holds. Nothing here waits for indexing: a search covers
the passages indexed so far and says how far each document is.

Every argument is checked before any source is touched; each call has ONE time budget
(`companies_query_timeout_s` + `sec_call_budget_s`, as a live companies call). Errors keep the
`[code] message` form; an indexer or SEC error's own text is never forwarded where it could hold a
path or the SEC contact (fixed messages, the exception class only in the log).
"""

import datetime as dt
import logging
import threading
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import tier_check
from ..companies.common import READ_ONLY, call_budget_s, store_call, time_budget
from ..companies.router import SourceRouter
from ..companies.tools import (
    COMPANIES_DESCRIPTION,
    CompanyId,
    check_companies,
    check_period,
    lookup_companies,
    per_company,
    unknown_company_hint,
)
from ..config import Settings
from ..errors import (
    BACKEND_UNAVAILABLE,
    INDEX_NOT_READY,
    INDEX_UNAVAILABLE,
    INVALID_ARGUMENT,
    LIMIT_EXCEEDED,
    NOT_FETCHED,
    NOT_FOUND,
    tool_error,
)
from ..paging import cap_result, clamp_limit
from . import GRAPH_NOT_AVAILABLE
from .indexer import (
    DocStatus,
    DocumentNotFetched,
    FilingsIndexer,
    IndexUnavailable,
    SectionNotFound,
    chunk_id,
    filing_doc_id,
)

log = logging.getLogger("myopenfund_mcp.filings")

FETCH_TOOL = "companies_fetch_filings"
#: schema bounds (as `rag_search`'s query and the companies tools' identifiers), not thresholds
MAX_QUERY_CHARS = 2000
MAX_ID_CHARS = 200
MAX_CHUNK_ID_CHARS = 20
ACT = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
INDEX_LOCKED_MESSAGE = (
    "the local filings index cannot be opened: another process holds the index folder (another MCP "
    "server, or `myopenfund-mcp warm`, on the same MOF_CACHE_DIR); stop it, or give this server another "
    "MOF_CACHE_DIR, then try again"
)
INDEX_CLOSED_MESSAGE = "the local filings index is closed (the server is shutting down)"
INDEX_FAILED_MESSAGE = "the local filings index failed; try again later"
FORMS_DESCRIPTION = "Forms, e.g. ['10-K', '10-Q']; a base form includes its amendments ('10-K/A')"


class LazyIndexer:
    """The indexer, built by `factory` on first use: nothing is built, opened or loaded at start-up.
    `close()` closes it if it was built (and later uses raise `IndexUnavailable`)."""

    def __init__(self, factory: Callable[[], FilingsIndexer]):
        self._factory = factory
        self._lock = threading.Lock()
        self._indexer: FilingsIndexer | None = None
        self._closed = False

    def get(self) -> FilingsIndexer:
        with self._lock:
            if self._closed:
                raise IndexUnavailable("the filings index is closed")
            if self._indexer is None:
                self._indexer = self._factory()
            return self._indexer

    def close(self) -> None:
        with self._lock:
            self._closed = True
            indexer, self._indexer = self._indexer, None
        if indexer is not None:
            indexer.close()


def filings_info(indexer: FilingsIndexer, settings: Settings) -> dict:
    """`server_info.filings`. The document counts come from the status record (no reconciliation)
    and only once this process has opened the index: asking never takes the index folder's lock."""
    info: dict[str, Any] = {
        "status": "enabled",
        "model": indexer.model_name,
        "provider": settings.rag_onnx_provider,
        "chunker_version": indexer.chunker_version,
        "documents": None,
        "graph": GRAPH_NOT_AVAILABLE,
    }
    if not indexer.opened:
        info["documents_note"] = "the index is opened by the first filings call of this server"
        return info
    try:
        statuses = indexer.status()
    except Exception as exc:  # noqa: BLE001 -- server_info still answers
        log.warning("server_info: filings status failed (%s)", type(exc).__name__)
        info["documents_error"] = (
            INDEX_UNAVAILABLE if isinstance(exc, IndexUnavailable) else BACKEND_UNAVAILABLE
        )
        return info
    info["documents"] = {"ready": sum(s.state == "ready" for s in statuses), "total": len(statuses)}
    return info


def _index_error(exc: BaseException, *, arguments: bool) -> ToolError:
    """The `[code]` of an indexer error. `arguments`: a ValueError is the caller's (a read's offset,
    chunk id or window); elsewhere it is a fault."""
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, IndexUnavailable):
        closed = "closed" in str(exc)
        return tool_error(INDEX_UNAVAILABLE, INDEX_CLOSED_MESSAGE if closed else INDEX_LOCKED_MESSAGE)
    if isinstance(exc, DocumentNotFetched):
        return tool_error(
            NOT_FETCHED,
            f"{exc}; call {FETCH_TOOL} first (its result lists the doc_id of each "
            "document), or take a doc_id from companies_search_filings",
        )
    if isinstance(exc, (SectionNotFound, LookupError)):
        return tool_error(NOT_FOUND, str(exc))  # doc ids, section names and chunk numbers only
    if arguments and isinstance(exc, ValueError):
        return tool_error(INVALID_ARGUMENT, str(exc))
    log.warning("filings index call failed (%s)", type(exc).__name__)
    return tool_error(BACKEND_UNAVAILABLE, INDEX_FAILED_MESSAGE)


def _forms(forms: list[str] | None) -> list[str] | None:
    wanted = [f.strip().upper() for f in forms] if forms is not None else None
    if wanted is not None and (not wanted or not all(wanted)):
        raise tool_error(INVALID_ARGUMENT, "forms must name at least one form, e.g. ['10-K']")
    return wanted


def _company(hit, names: dict[str, str]) -> str | None:
    """The company of a hit: its resolved name, else the one in the chunk title (`<name> <form> <filed>`)."""
    if names.get(hit.source_code):
        return names[hit.source_code]
    title, suffix = hit.title or "", f" {hit.doc_type} {hit.date or 'undated'}"
    return title[: -len(suffix)] if title.endswith(suffix) and len(title) > len(suffix) else None


def _status_row(status: DocStatus, names: dict[str, str]) -> dict:
    return {"company": names.get(status.cik), **status.to_dict()}


def _progress(matched: list[DocStatus]) -> str:
    """`ready n/m`, then the count per other state; passages done of those indexing."""
    ready = sum(s.state == "ready" for s in matched)
    parts = [f"ready {ready}/{len(matched)}"]
    for state in ("queued", "fetching", "indexing", "partial", "failed"):
        docs = [s for s in matched if s.state == state]
        if not docs:
            continue
        text = f"{state} {len(docs)}"
        if state == "indexing":
            done = sum(s.chunks_done for s in docs)
            total = sum(s.chunks_total or 0 for s in docs)
            text += f" ({done}/{total} passages)"
        parts.append(text)
    return ", ".join(parts)


def build_filings_toolset(
    indexer: Callable[[], FilingsIndexer], router: SourceRouter, settings: Settings, *, auth_enabled: bool
) -> FastMCP:
    """The three tools; `indexer` returns the (lazily built) indexer."""
    fs = FastMCP("filings", mask_error_details=True)
    read = tier_check("read", auth_enabled=auth_enabled)
    act = tier_check("act", auth_enabled=auth_enabled)
    budget_s = call_budget_s(settings, live=True)
    unknown_hint = unknown_company_hint(router)
    defaults = ", ".join(settings.filings_default_forms)

    async def call(fn, /, **kwargs):
        """A router call (company lookup, SEC filing list): errors coded as the companies tools do."""
        return await store_call(fn, settings=settings, timeout_s=budget_s, **kwargs)

    async def run(method: str, /, *args, arguments: bool = False, **kwargs):
        """An indexer method off the event loop, within the tool's budget; its errors coded."""
        try:
            target = await anyio.to_thread.run_sync(indexer, abandon_on_cancel=True)
            return await anyio.to_thread.run_sync(
                partial(getattr(target, method), *args, **kwargs), abandon_on_cancel=True
            )
        except Exception as exc:  # noqa: BLE001 -- every error gets its code
            raise _index_error(exc, arguments=arguments) from None

    async def lookup(companies: list[str]) -> list[dict]:
        return await lookup_companies(router, companies, call=call, unknown_hint=unknown_hint)

    fetch_description = (
        "Download companies' SEC filings and index their text on this machine, so that "
        "companies_search_filings can search them: call companies_fetch_filings first, before any "
        f"search. `forms` default: {defaults}. Returns at once with each document's `doc_id` and `state` "
        "(queued, fetching, indexing, ready, partial, failed): indexing goes on in the background, the "
        "MD&A (Item 7) first, and companies_search_filings finds passages as soon as they are indexed. "
        f"At most {settings.filings_max_docs} documents per call (MOF_FILINGS_MAX_DOCS): narrow `forms` or "
        "the period. `model_download` true = the embedding model (about 1.1 GB) is downloaded once first, "
        "so the first passages take a few minutes. Fetching a document again is harmless (a ready one "
        "stays ready; a partial or failed one is resumed)."
    )

    @fs.tool(auth=act, annotations=ACT, description=fetch_description)
    async def fetch_filings(
        companies: Annotated[list[CompanyId], Field(description=COMPANIES_DESCRIPTION, min_length=1)],
        forms: Annotated[
            list[str] | None,
            Field(description=f"{FORMS_DESCRIPTION}; omitted = {defaults} (MOF_FILINGS_DEFAULT_FORMS)"),
        ] = None,
        period_from: Annotated[
            dt.date | None, Field(description="Report period end on or after, YYYY-MM-DD")
        ] = None,
        period_to: Annotated[
            dt.date | None, Field(description="Report period end on or before, YYYY-MM-DD")
        ] = None,
    ) -> dict:
        check_companies(companies, settings)
        check_period(period_from, period_to)
        wanted = _forms(forms) or list(settings.filings_default_forms)
        with time_budget(settings, budget_s):
            hits = await lookup(companies)
            found = await per_company(
                hits,
                lambda h: call(
                    router.filings, cik=h["cik"], forms=wanted, period_from=period_from, period_to=period_to
                ),
            )
            names = {h["cik"]: h["current_name"] for h in hits}
            refs = list({filing_doc_id(r): r for rs in found for r in rs}.values())
            if len(refs) > settings.filings_max_docs:
                raise tool_error(
                    LIMIT_EXCEEDED,
                    f"{len(refs)} filings match; at most {settings.filings_max_docs} "
                    "documents per call (MOF_FILINGS_MAX_DOCS): narrow forms or the "
                    "period (period_from, period_to), or fetch fewer companies at once",
                )
            statuses = await run("enqueue", refs, names=names) if refs else []
        download = bool(getattr(statuses, "model_download", False))
        out = {
            "forms": wanted,
            "total": len(statuses),
            "model_download": download,
            "documents": [_status_row(s, names) for s in statuses],
            "next": (
                "call companies_search_filings now: passages are searchable as they are indexed"
                if statuses
                else "no filing of these forms in this period at SEC; try other forms or another period"
            ),
        }
        return cap_result(out, settings, hint="fetch fewer companies, forms or a shorter period at once")

    @fs.tool(auth=read, annotations=READ_ONLY)
    async def search_filings(
        query: Annotated[
            str,
            Field(
                description="What to find, in English, with synonyms, e.g. 'capital expenditures purchases of "
                "property and equipment'",
                max_length=MAX_QUERY_CHARS,
            ),
        ],
        companies: Annotated[
            list[CompanyId] | None,
            Field(description="CIKs, tickers or entity_ids; omitted = every fetched filing", min_length=1),
        ] = None,
        forms: Annotated[list[str] | None, Field(description=f"{FORMS_DESCRIPTION}; omitted = all")] = None,
        period_from: Annotated[
            dt.date | None, Field(description="Report period end on or after, YYYY-MM-DD")
        ] = None,
        period_to: Annotated[
            dt.date | None, Field(description="Report period end on or before, YYYY-MM-DD")
        ] = None,
        limit: Annotated[
            int | None, Field(description="Passages to return; omitted = server default")
        ] = None,
    ) -> dict:
        """Search the text of filings fetched with companies_fetch_filings: call companies_fetch_filings
        first (`[not_fetched]` = nothing fetched for these companies, forms or period).
        Search in English (the filings' language) and add synonyms, e.g.
        "capital expenditures, purchases of property and equipment".
        Each hit is a passage: company, cik, form, filed, section (e.g. 'Item 7'), url, doc_id,
        chunk_id, text. Before quoting, read it in context with companies_read_filing(doc_id,
        chunk_id); cite `url` and `filed`. `indexed` = documents fully indexed out of those searched
        ("n/m"); a document still indexing is searched as far as it goes and listed in `not_ready`.
        `[index_not_ready]` = nothing searchable yet: call again in a minute."""
        text = query.strip()
        if not text:
            raise tool_error(INVALID_ARGUMENT, "query is empty: say what to find, in English")
        page = clamp_limit(limit, settings)
        if companies is not None:
            check_companies(companies, settings)
        check_period(period_from, period_to)
        wanted = _forms(forms)
        names: dict[str, str] = {}
        with time_budget(settings, budget_s):
            if companies is not None:
                names = {h["cik"]: h["current_name"] for h in await lookup(companies)}
            known = await run("status")
            scope = [s.doc_id for s in known if companies is None or s.cik in names]
            if not scope:
                whose = "for these companies" if companies is not None else "yet"
                raise tool_error(
                    NOT_FETCHED, f"no filing was fetched {whose}; call {FETCH_TOOL} first, then search again"
                )
            found = await run(
                "search",
                text,
                doc_ids=scope,
                forms=wanted,
                period_from=period_from,
                period_to=period_to,
                limit=page,
            )
        matched = found.matched
        if not matched:
            raise tool_error(
                NOT_FETCHED,
                f"no fetched filing matches these forms and period; call {FETCH_TOOL} "
                "with them first, then search again",
            )
        ready = sum(s.state == "ready" for s in matched)
        if not found.hits and ready == 0:
            raise tool_error(
                INDEX_NOT_READY,
                f"nothing is indexed yet among the {len(matched)} fetched filings "
                f"in scope ({_progress(matched)}); indexing goes on in the "
                "background: call again in a minute"
                + (
                    f"; failed documents: fetch them again with {FETCH_TOOL}"
                    if any(s.state == "failed" for s in matched)
                    else ""
                ),
            )
        hits = [
            {
                "company": _company(h, names),
                "cik": h.source_code,
                "form": h.doc_type,
                "filed": h.date,
                "section": h.payload.get("section"),
                "url": h.payload.get("url"),
                "doc_id": h.doc_id,
                "chunk_id": chunk_id(h),
                "text": h.text,
            }
            for h in found.hits
        ]
        out = {
            "query": text,
            "indexed": f"{ready}/{len(matched)}",
            "not_ready": [_status_row(s, names) for s in found.not_ready],
            "returned": len(hits),
            "hits": hits,
        }
        return cap_result(
            out, settings, hint="ask for a smaller limit, or fewer companies, forms or a shorter period"
        )

    @fs.tool(auth=read, annotations=READ_ONLY)
    async def read_filing(
        doc_id: Annotated[
            str,
            Field(
                description="doc_id from companies_search_filings or companies_fetch_filings",
                max_length=MAX_ID_CHARS,
            ),
        ],
        section: Annotated[
            str | None,
            Field(
                description="A section, e.g. 'Item 7' or 'Item 1A': read it from its start (or from chunk_id)",
                max_length=MAX_ID_CHARS,
            ),
        ] = None,
        chunk_id: Annotated[
            str | None,
            Field(
                description="A hit's chunk_id: read forward from that passage", max_length=MAX_CHUNK_ID_CHARS
            ),
        ] = None,
        around: Annotated[
            int | None,
            Field(
                description="With chunk_id (and no section): that many passages before and after it instead",
                ge=0,
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(description="next_offset of the previous read (with chunk_id = its next_chunk_id)", ge=0),
        ] = 0,
    ) -> dict:
        """Read a fetched filing's text, to quote it in context: pass a hit's `doc_id` and `chunk_id`
        (reads forward from that passage; `around`=k gives the k passages on each side instead), or a
        `section` ('Item 7' = MD&A, 'Item 1A' = risk factors, 'Item 8' = financial statements);
        neither = from the start. The result is bounded: while `more` is true, call again with the same
        `doc_id` and `section`, `chunk_id`=`next_chunk_id` and `offset`=`next_offset`. Quote the text
        as it is and cite `url` and `filed`. `[not_fetched]`: call companies_fetch_filings first."""
        with time_budget(settings, budget_s):
            result = await run(
                "read",
                doc_id,
                section=section,
                chunk_id=chunk_id,
                around=around,
                offset=offset,
                arguments=True,
            )
        return cap_result(
            result, settings, hint="read one section, or from a chunk_id, and follow next_chunk_id"
        )

    return fs
