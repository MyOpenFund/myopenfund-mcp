"""The filing-text tools (companies_fetch_filings, companies_search_filings, companies_read_filing).

Tool tests run on an in-memory FastMCP client with a fake indexer and the fake live source; one test
runs the whole path on the real indexer with FakeEmbedder; one runs the tiers over real HTTP auth."""

from __future__ import annotations

import datetime as dt
import logging
import socket
import threading
import time

import anyio
import pytest
from company_corpus.live.filings import FilingRef
from fastmcp import Client
from fastmcp.exceptions import ToolError
from rag_engine.search import Hit
from source_fakes import APPLE, MSFT, FakeLive

from myopenfund_mcp.auth import hash_token
from myopenfund_mcp.config import Settings
from myopenfund_mcp.filings.indexer import (
    DocStatus,
    DocumentNotFetched,
    EnqueueResult,
    IndexUnavailable,
    SearchResult,
    SectionNotFound,
    StatusResult,
    filing_title,
)
from myopenfund_mcp.server import build_server

pytestmark = pytest.mark.anyio

SENTINEL = "sentinel-contact@example.invalid"
TOOLS = {"companies_fetch_filings", "companies_search_filings", "companies_read_filing"}


def filing(
    cik: str = MSFT,
    accession: str = "0000789019-24-000001",
    *,
    form: str = "10-K",
    filed=dt.date(2024, 7, 30),
    report=dt.date(2024, 6, 30),
) -> FilingRef:
    return FilingRef(
        cik=cik,
        accession=accession,
        form=form,
        report_date=report,
        filed=filed,
        accepted=None,
        primary_document="doc.htm",
        url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}",
        doc_id=f"sec-{accession}",
    )


MSFT_K24, MSFT_K23 = (
    filing(),
    filing(accession="0000789019-23-000001", filed=dt.date(2023, 7, 27), report=dt.date(2023, 6, 30)),
)
MSFT_Q = filing(
    accession="0000789019-24-000009", form="10-Q", filed=dt.date(2024, 4, 25), report=dt.date(2024, 3, 31)
)
APPLE_K24 = filing(APPLE, "0000320193-24-000123", filed=dt.date(2024, 11, 1), report=dt.date(2024, 9, 28))


def status(ref: FilingRef, state: str = "queued", done: int = 0, total: int | None = None) -> DocStatus:
    return DocStatus(
        doc_id=ref.doc_id,
        cik=ref.cik,
        form=ref.form,
        accession=ref.accession,
        filed=ref.filed,
        url=ref.url,
        state=state,
        chunks_done=done,
        chunks_total=total,
        error=None,
    )


def hit(
    ref: FilingRef,
    number: int,
    text: str = "Capital expenditures were $44.5 billion.",
    name: str = "MICROSOFT CORP",
) -> Hit:
    return Hit(
        text=text,
        score=0.9,
        doc_id=ref.doc_id,
        title=filing_title(name, ref.form, ref.report_date, ref.filed),
        date=ref.filed.isoformat(),
        source_code=ref.cik,
        doc_type=ref.form,
        language=None,
        page=None,
        chunk_number=number,
        corpus=None,
        point_id=number,
        payload={"section": "Item 7", "url": ref.url, "chunk_number": number},
    )


class FakeIndexer:
    """The FilingsIndexer API over a dict of statuses; records every call; `fail`: method -> exception."""

    model_name = "nomic-ai/nomic-embed-text-v1.5"
    chunker_version = "v2-t480-o32-ttl2"

    def __init__(self, docs=(), *, hits=(), download=False, fail=None, read_result=None, companies=None):
        self.docs = {d.doc_id: d for d in docs}
        self.companies = dict(companies or {})  # cik -> (name, tickers), as recorded at enqueue
        self.hits = list(hits)
        self.download = download
        self.fail = dict(fail or {})
        self.read_result = read_result
        self.calls: list[tuple] = []
        self.opened = False
        self.closed = False

    def _enter(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if method in self.fail:
            raise self.fail[method]
        self.opened = True

    def enqueue(self, refs, *, names=None, tickers=None):
        self._enter("enqueue", refs=list(refs), names=names, tickers=tickers)
        for r in refs:
            self.companies[r.cik] = ((names or {}).get(r.cik), list((tickers or {}).get(r.cik) or []))
        out = EnqueueResult()
        for r in refs:
            self.docs.setdefault(r.doc_id, status(r))
            out.append(self.docs[r.doc_id])
        out.model_download = self.download
        return out

    def fetched_companies(self):
        self._enter("fetched_companies")
        ciks = sorted({d.cik for d in self.docs.values()})
        return [
            {
                "cik": c,
                "name": self.companies.get(c, (None, []))[0],
                "tickers": self.companies.get(c, (None, []))[1],
            }
            for c in ciks
        ]

    def status(self, doc_ids=None):
        self._enter("status", doc_ids=doc_ids)
        return StatusResult([d for d in self.docs.values() if doc_ids is None or d.doc_id in doc_ids])

    def search(self, query, *, doc_ids, forms=None, period_from=None, period_to=None, limit=10):
        self._enter(
            "search",
            query=query,
            doc_ids=doc_ids,
            forms=forms,
            period_from=period_from,
            period_to=period_to,
            limit=limit,
        )
        wanted = None if forms is None else {f.upper() for f in forms}
        matched = [
            d
            for d in self.docs.values()
            if d.doc_id in doc_ids and (wanted is None or d.form.upper().removesuffix("/A") in wanted)
        ]
        ids = {d.doc_id for d in matched}
        hits = [h for h in self.hits if h.doc_id in ids][:limit]
        return SearchResult(hits, [d for d in matched if d.state != "ready"], [], matched)

    def read(self, doc_id, *, section=None, chunk_id=None, around=None, offset=0):
        self._enter("read", doc_id=doc_id, section=section, chunk_id=chunk_id, around=around, offset=offset)
        if doc_id not in self.docs:
            raise DocumentNotFetched(f"document {doc_id!r} was never fetched")
        return self.read_result or {
            **self.docs[doc_id].to_dict(),
            "section": section,
            "chunks": [],
            "more": False,
            "next_chunk_id": None,
            "next_offset": 0,
        }

    def close(self):
        self.closed = True


def _server(indexer, live=None, **kw):
    kw.setdefault("sec_contact", SENTINEL)
    live = (
        live if live is not None else FakeLive(refs={MSFT: [MSFT_K24, MSFT_K23, MSFT_Q], APPLE: [APPLE_K24]})
    )
    return build_server(Settings(**kw), companies_live=live, filings_indexer=indexer)


async def _call(indexer, tool, args, live=None, **kw):
    async with Client(_server(indexer, live, **kw)) as c:
        return (await c.call_tool(tool, args)).data


async def _error(indexer, tool, args, live=None, **kw) -> str:
    async with Client(_server(indexer, live, **kw)) as c:
        with pytest.raises(ToolError) as err:
            await c.call_tool(tool, args)
    return str(err.value)


# --- mounting ----------------------------------------------------------------------------------------


async def test_stdio_lists_and_calls_the_three_tools():
    indexer = FakeIndexer()
    async with Client(_server(indexer)) as c:
        tools = {t.name: t for t in await c.list_tools()}
        assert TOOLS <= set(tools)
        assert (await c.call_tool("companies_fetch_filings", {"companies": ["MSFT"]})).data["total"] == 2


async def test_descriptions_are_directive_for_small_models():
    async with Client(_server(FakeIndexer())) as c:
        tools = {t.name: t for t in await c.list_tools()}
    search = tools["companies_search_filings"].description
    assert "companies_fetch_filings" in search and "English" in search
    assert "capital expenditures" in search and "purchases of property and equipment" in search
    assert "companies_read_filing" in search and "`url`" in search and "`filed`" in search
    fetch = tools["companies_fetch_filings"].description
    assert "call companies_fetch_filings first" in fetch and "MOF_FILINGS_MAX_DOCS" in fetch
    read = tools["companies_read_filing"].description
    assert "quote" in read and "`url`" in read and "`filed`" in read and "companies_fetch_filings" in read
    assert tools["companies_search_filings"].annotations.read_only_hint is True
    assert tools["companies_fetch_filings"].annotations.read_only_hint is False


async def test_without_a_contact_the_tools_are_absent_and_server_info_says_why():
    async with Client(build_server(Settings())) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert not names & TOOLS
    assert "MOF_SEC_CONTACT" in info["filings"]["status"]
    assert info["filings"]["graph"] == "not available (OLAF after the V0)"


async def test_store_mode_and_an_injected_live_source_without_indexer_do_not_mount():
    from source_fakes import FakeStore

    server = build_server(
        Settings(sec_contact=SENTINEL, companies_source="store", companies_catalog="x"),
        companies_store=FakeStore(),
    )
    async with Client(server) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert not names & TOOLS and "MOF_COMPANIES_SOURCE=store" in info["filings"]["status"]
    async with Client(build_server(Settings(sec_contact=SENTINEL), companies_live=FakeLive())) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert not names & TOOLS and "SEC client" in info["filings"]["status"]


async def test_without_the_filings_extra_the_tools_are_absent(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "myopenfund_mcp.filings.indexer", None)  # import -> ImportError
    monkeypatch.delitem(sys.modules, "myopenfund_mcp.filings.tools", raising=False)
    server = build_server(Settings(sec_contact=SENTINEL, cache_dir=str(tmp_path / "c")))
    async with Client(server) as c:
        names = {t.name for t in await c.list_tools()}
        info = (await c.call_tool("server_info", {})).data
    assert "companies_facts" in names and not names & TOOLS
    assert "[filings]" in info["filings"]["status"]


async def test_the_indexer_is_built_lazily_and_closed_at_shutdown(tmp_path, monkeypatch):
    from myopenfund_mcp.filings import indexer as indexer_module

    made = []

    class Recording(FakeIndexer):
        def __init__(self, settings, sec):
            super().__init__()
            self.sec = sec
            made.append(self)

    monkeypatch.setattr(indexer_module, "FilingsIndexer", Recording)
    settings = Settings(sec_contact=SENTINEL, cache_dir=str(tmp_path / "c"))
    server = build_server(settings)
    assert made == []  # nothing built at start-up
    async with Client(server) as c:
        assert made == []
        info = (await c.call_tool("server_info", {})).data["filings"]
        assert info["status"] == "enabled" and info["documents"] is None  # asking never opens the index
        assert (info["model"], info["provider"], info["chunker_version"]) == (
            FakeIndexer.model_name,
            "cpu",
            FakeIndexer.chunker_version,
        )
        assert len(made) == 1 and made[0].calls == [] and made[0].sec is not None
        with pytest.raises(ToolError, match=r"^\[not_fetched\]"):
            await c.call_tool("companies_search_filings", {"query": "debt"})
        info = (await c.call_tool("server_info", {})).data["filings"]
        assert info["documents"] == {"ready": 0, "total": 0} and not made[0].closed
    assert made[0].closed


async def test_server_info_counts_ready_documents_without_reconciliation():
    indexer = FakeIndexer([status(MSFT_K24, "ready", 5, 5), status(MSFT_K23, "indexing", 1, 5)])
    indexer.opened = True
    async with Client(_server(indexer)) as c:
        info = (await c.call_tool("server_info", {})).data["filings"]
    assert info["documents"] == {"ready": 1, "total": 2}
    assert indexer.calls == [("status", {"doc_ids": None})]  # status only: never search or read


# --- companies_fetch_filings -------------------------------------------------------------------------


async def test_fetch_uses_the_default_forms_and_returns_statuses_at_once():
    live = FakeLive(refs={MSFT: [MSFT_K24, MSFT_K23, MSFT_Q]})
    indexer = FakeIndexer(download=True)
    res = await _call(
        indexer,
        "companies_fetch_filings",
        {"companies": ["MSFT"]},
        live=live,
        filings_default_forms=("10-K",),
    )
    assert res["forms"] == ["10-K"] and res["total"] == 2 and res["model_download"] is True
    assert [d["doc_id"] for d in res["documents"]] == [MSFT_K24.doc_id, MSFT_K23.doc_id]
    first = res["documents"][0]
    assert (first["company"], first["cik"], first["state"], first["filed"], first["url"]) == (
        "MICROSOFT CORP",
        MSFT,
        "queued",
        "2024-07-30",
        MSFT_K24.url,
    )
    ((method, kwargs),) = indexer.calls
    assert method == "enqueue" and kwargs["names"] == {MSFT: "MICROSOFT CORP"}


async def test_fetch_passes_forms_and_period_and_dedupes():
    live = FakeLive(refs={MSFT: [MSFT_K24, MSFT_K23, MSFT_Q]})
    indexer = FakeIndexer()
    res = await _call(
        indexer,
        "companies_fetch_filings",
        {"companies": ["MSFT", MSFT], "forms": [" 10-q", "10-K"], "period_from": "2024-01-01"},
        live=live,
    )
    assert [d["doc_id"] for d in res["documents"]] == [MSFT_K24.doc_id, MSFT_Q.doc_id]
    assert live.calls.count("filings") == 1


async def test_fetch_above_the_document_cap_is_refused_naming_the_setting():
    indexer = FakeIndexer()
    msg = await _error(
        indexer, "companies_fetch_filings", {"companies": ["MSFT", "AAPL"]}, filings_max_docs=2
    )
    assert msg.startswith("[limit_exceeded]") and "MOF_FILINGS_MAX_DOCS" in msg and "3 filings" in msg
    assert indexer.calls == []  # nothing queued


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"companies": ["nope-nothing"]}, "unknown_company"),
        ({"companies": ["MSFT"], "forms": []}, "invalid_argument"),
        ({"companies": ["MSFT"], "period_from": "2024-01-01", "period_to": "2023-01-01"}, "invalid_argument"),
        ({"companies": [str(n) for n in range(11)]}, "too_many_companies"),
    ],
)
async def test_fetch_errors_are_coded(args, code):
    indexer = FakeIndexer()
    msg = await _error(indexer, "companies_fetch_filings", args)
    assert msg.startswith(f"[{code}]") and indexer.calls == []


async def test_fetch_with_nothing_at_sec_says_so():
    res = await _call(FakeIndexer(), "companies_fetch_filings", {"companies": ["JPM"]})
    assert res["total"] == 0 and res["documents"] == [] and "no filing" in res["next"]


async def test_a_held_index_is_index_unavailable_saying_another_process():
    locked = IndexUnavailable("the local filings index '/somewhere' is locked")
    for tool, args in (
        ("companies_fetch_filings", {"companies": ["MSFT"]}),
        ("companies_search_filings", {"query": "debt"}),
        ("companies_read_filing", {"doc_id": MSFT_K24.doc_id}),
    ):
        indexer = FakeIndexer(fail={"enqueue": locked, "status": locked, "read": locked})
        msg = await _error(indexer, tool, args)
        assert msg.startswith("[index_unavailable]") and "another process" in msg and "/somewhere" not in msg


async def test_sec_errors_are_coded_without_the_contact(caplog):
    from company_corpus.live.sec import SecUnavailable

    live = FakeLive(refs={MSFT: [MSFT_K24]}, fail={"filings": SecUnavailable(f"for {SENTINEL}")})
    msg = await _error(FakeIndexer(), "companies_fetch_filings", {"companies": ["MSFT"]}, live=live)
    assert msg.startswith("[sec_unavailable]") and SENTINEL not in msg and SENTINEL not in caplog.text


# --- companies_search_filings ------------------------------------------------------------------------


async def test_search_needs_a_query():
    for query in ("", "   "):
        assert (await _error(FakeIndexer(), "companies_search_filings", {"query": query})).startswith(
            "[invalid_argument]"
        )


async def test_search_with_nothing_fetched_names_the_fetch_tool():
    msg = await _error(FakeIndexer(), "companies_search_filings", {"query": "capex"})
    assert msg.startswith("[not_fetched]") and "companies_fetch_filings" in msg
    # fetched for another company only
    indexer = FakeIndexer([status(APPLE_K24, "ready", 3, 3)])
    msg = await _error(indexer, "companies_search_filings", {"query": "capex", "companies": ["MSFT"]})
    assert msg.startswith("[not_fetched]") and "companies_fetch_filings" in msg
    assert not any(m == "search" for m, _ in indexer.calls)
    # fetched, but not these forms
    indexer = FakeIndexer([status(MSFT_K24, "ready", 3, 3)])
    msg = await _error(indexer, "companies_search_filings", {"query": "capex", "forms": ["10-Q"]})
    assert msg.startswith("[not_fetched]") and "companies_fetch_filings" in msg


async def test_search_before_anything_is_indexed_gives_progress():
    indexer = FakeIndexer([status(MSFT_K24, "indexing", 0, 40), status(MSFT_K23, "queued")])
    msg = await _error(indexer, "companies_search_filings", {"query": "capex", "companies": ["MSFT"]})
    assert msg.startswith("[index_not_ready]") and "ready 0/2" in msg and "call again" in msg
    assert "queued 1" in msg and "indexing 1 (0/40 passages)" in msg


@pytest.mark.parametrize("states", [("partial", "partial"), ("failed", "failed"), ("partial", "failed")])
async def test_index_not_ready_without_work_in_flight_says_fetch_again(states):
    indexer = FakeIndexer([status(MSFT_K24, states[0], 0, 40), status(MSFT_K23, states[1])])
    msg = await _error(indexer, "companies_search_filings", {"query": "capex"})
    assert msg.startswith("[index_not_ready]") and "call again in a minute" not in msg
    assert "call companies_fetch_filings again" in msg and "partial" in msg and "failed" in msg


async def test_index_not_ready_with_work_in_flight_says_call_again_and_names_the_rest():
    indexer = FakeIndexer([status(MSFT_K24, "queued"), status(MSFT_K23, "failed")])
    msg = await _error(indexer, "companies_search_filings", {"query": "capex"})
    assert "call again in a minute" in msg and "call companies_fetch_filings again" in msg


async def test_search_hits_carry_the_citation_fields_and_the_progress():
    indexer = FakeIndexer(
        [
            status(MSFT_K24, "ready", 5, 5),
            status(MSFT_K23, "indexing", 2, 5),
            status(APPLE_K24, "ready", 3, 3),
        ],
        hits=[hit(MSFT_K24, 4), hit(MSFT_K23, 1), hit(APPLE_K24, 0, name="Apple Inc.")],
    )
    res = await _call(
        indexer,
        "companies_search_filings",
        {"query": "  capital expenditures ", "companies": ["MSFT"], "limit": 5},
    )
    assert res["indexed"] == "1/2" and res["returned"] == 2 and res["query"] == "capital expenditures"
    assert [n["doc_id"] for n in res["not_ready"]] == [MSFT_K23.doc_id]
    assert res["not_ready"][0]["state"] == "indexing" and res["not_ready"][0]["company"] == "MICROSOFT CORP"
    assert res["hits"][0] == {
        "company": "MICROSOFT CORP",
        "cik": MSFT,
        "form": "10-K",
        "filed": "2024-07-30",
        "section": "Item 7",
        "url": MSFT_K24.url,
        "doc_id": MSFT_K24.doc_id,
        "chunk_id": "4",
        "text": "Capital expenditures were $44.5 billion.",
    }
    (search,) = [kw for m, kw in indexer.calls if m == "search"]
    assert sorted(search["doc_ids"]) == sorted([MSFT_K24.doc_id, MSFT_K23.doc_id]) and search["limit"] == 5


async def test_search_without_companies_passes_every_fetched_doc_id():
    indexer = FakeIndexer(
        [status(MSFT_K24, "ready", 5, 5), status(APPLE_K24, "ready", 3, 3)],
        hits=[hit(APPLE_K24, 0, name="Apple Inc.")],
    )
    res = await _call(indexer, "companies_search_filings", {"query": "capex"})
    (search,) = [kw for m, kw in indexer.calls if m == "search"]
    assert search["doc_ids"] is not None and set(search["doc_ids"]) == {MSFT_K24.doc_id, APPLE_K24.doc_id}
    assert res["indexed"] == "2/2" and res["hits"][0]["company"] == "Apple Inc."  # from the chunk title


async def test_search_limit_uses_its_own_settings():
    indexer = FakeIndexer([status(MSFT_K24, "ready", 5, 5)])
    await _call(indexer, "companies_search_filings", {"query": "capex"})
    await _call(indexer, "companies_search_filings", {"query": "capex"}, filings_search_default_limit=3)
    assert [kw["limit"] for m, kw in indexer.calls if m == "search"] == [8, 3]
    msg = await _error(
        FakeIndexer([status(MSFT_K24, "ready", 5, 5)]),
        "companies_search_filings",
        {"query": "capex", "limit": 21},
    )
    assert msg.startswith("[limit_exceeded]") and "MOF_FILINGS_SEARCH_MAX_LIMIT" in msg and "20" in msg
    msg = await _error(
        FakeIndexer([status(MSFT_K24, "ready", 5, 5)]),
        "companies_search_filings",
        {"query": "capex", "limit": 0},
    )
    assert msg.startswith("[invalid_argument]")


async def test_a_ready_scope_without_hits_is_an_empty_answer():
    res = await _call(
        FakeIndexer([status(MSFT_K24, "ready", 5, 5)]), "companies_search_filings", {"query": "capex"}
    )
    assert res["hits"] == [] and res["indexed"] == "1/1"


async def test_a_too_large_result_is_refused_not_truncated():
    indexer = FakeIndexer(
        [status(MSFT_K24, "ready", 5, 5)], hits=[hit(MSFT_K24, n, text="word " * 200) for n in range(5)]
    )
    msg = await _error(indexer, "companies_search_filings", {"query": "capex"}, max_result_chars=2000)
    assert msg.startswith("[result_too_large]") and "smaller limit" in msg
    big = {**status(MSFT_K24).to_dict(), "chunks": [{"chunk_id": "0", "text": "x" * 5000}], "more": False}
    msg = await _error(
        FakeIndexer([status(MSFT_K24)], read_result=big),
        "companies_read_filing",
        {"doc_id": MSFT_K24.doc_id},
        max_result_chars=2000,
    )
    assert msg.startswith("[result_too_large]")


async def test_an_unexpected_index_error_is_masked_and_never_holds_the_contact(caplog):
    caplog.set_level(logging.DEBUG, logger="myopenfund_mcp")
    indexer = FakeIndexer(
        [status(MSFT_K24, "ready", 5, 5)], fail={"search": RuntimeError(f"boom {SENTINEL}")}
    )
    msg = await _error(indexer, "companies_search_filings", {"query": "capex"})
    assert msg.startswith("[backend_unavailable]") and SENTINEL not in msg
    assert "RuntimeError" in caplog.text and SENTINEL not in caplog.text


# --- companies_read_filing ---------------------------------------------------------------------------


async def test_read_passes_its_arguments_and_returns_the_indexer_result():
    indexer = FakeIndexer([status(MSFT_K24, "ready", 5, 5)])
    res = await _call(
        indexer,
        "companies_read_filing",
        {"doc_id": MSFT_K24.doc_id, "section": "Item 7", "chunk_id": "3", "offset": 10},
    )
    assert res["doc_id"] == MSFT_K24.doc_id and res["url"] == MSFT_K24.url and res["filed"] == "2024-07-30"
    assert indexer.calls == [
        (
            "read",
            {"doc_id": MSFT_K24.doc_id, "section": "Item 7", "chunk_id": "3", "around": None, "offset": 10},
        )
    ]


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (None, "not_fetched"),
        (
            SectionNotFound("document 'd' has no stored chunk in section 'Item 9'; sections: Item 7"),
            "not_found",
        ),
        (LookupError("document 'd' has no stored chunk 99"), "not_found"),
        (ValueError("offset 50 is past the end of chunk 3"), "invalid_argument"),
    ],
)
async def test_read_errors_are_coded(exc, code):
    indexer = FakeIndexer([status(MSFT_K24, "ready", 5, 5)], fail={"read": exc} if exc else None)
    doc = MSFT_K24.doc_id if exc else "sec-never"
    msg = await _error(indexer, "companies_read_filing", {"doc_id": doc})
    assert msg.startswith(f"[{code}]")
    if code == "not_fetched":
        assert "companies_fetch_filings" in msg
    if code == "not_found" and "section" in str(exc):
        assert "Item 7" in msg


async def test_read_refuses_a_negative_offset_or_window():
    for args in ({"offset": -1}, {"around": -2}):
        with pytest.raises(ToolError):
            await _call(
                FakeIndexer([status(MSFT_K24)]), "companies_read_filing", {"doc_id": MSFT_K24.doc_id, **args}
            )


# --- the contact never leaves ------------------------------------------------------------------------


async def test_results_and_server_info_never_hold_the_contact(caplog):
    caplog.set_level(logging.DEBUG, logger="myopenfund_mcp")
    indexer = FakeIndexer(hits=[hit(MSFT_K24, 0)])
    async with Client(_server(indexer)) as c:
        texts = [str((await c.call_tool("companies_fetch_filings", {"companies": ["MSFT"]})).data)]
        indexer.docs[MSFT_K24.doc_id] = status(MSFT_K24, "ready", 1, 1)
        for name, args in (
            ("companies_search_filings", {"query": "capex"}),
            ("companies_read_filing", {"doc_id": MSFT_K24.doc_id}),
            ("server_info", {}),
        ):
            texts.append(str((await c.call_tool(name, args)).data))
    assert all(SENTINEL not in t for t in texts) and SENTINEL not in caplog.text


# --- tiers over HTTP ---------------------------------------------------------------------------------

ACT_TOKEN = "t-act-" + "a" * 40
READ_TOKEN = "t-read-" + "b" * 40


@pytest.fixture
def http_url():
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    settings = Settings(
        transport="http",
        port=port,
        sec_contact=SENTINEL,
        tokens=f"alice:{hash_token(ACT_TOKEN)}:act,agent:{hash_token(READ_TOKEN)}:read",
    )
    server = build_server(
        settings,
        companies_live=FakeLive(refs={MSFT: [MSFT_K24]}),
        filings_indexer=FakeIndexer([status(MSFT_K24, "ready", 1, 1)]),
    )
    uv = uvicorn.Server(
        uvicorn.Config(server.http_app(path=settings.path), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uv.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            uv.should_exit = True
            pytest.fail("uvicorn did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}{settings.path}"
    finally:
        uv.should_exit = True
        thread.join(timeout=10)


async def test_read_tier_cannot_see_or_call_fetch_over_http(http_url):
    async with Client(http_url, auth=READ_TOKEN) as c:
        names = {t.name for t in await c.list_tools()}
        assert {"companies_search_filings", "companies_read_filing"} <= names
        assert "companies_fetch_filings" not in names
        with pytest.raises(Exception, match="companies_fetch_filings"):
            await c.call_tool("companies_fetch_filings", {"companies": ["MSFT"]})
        assert (await c.call_tool("companies_search_filings", {"query": "capex"})).data["indexed"] == "1/1"


async def test_act_tier_can_fetch_over_http(http_url):
    async with Client(http_url, auth=ACT_TOKEN) as c:
        assert (await c.call_tool("companies_fetch_filings", {"companies": ["MSFT"]})).data["total"] == 1


# --- end to end on the real indexer ------------------------------------------------------------------


async def test_end_to_end_on_the_real_indexer(tmp_path):
    from test_filings_indexer import GatedEmbedder, WordTokenizer, load

    from myopenfund_mcp.filings.indexer import FilingsIndexer

    settings = Settings(
        sec_contact=SENTINEL, cache_dir=str(tmp_path / "cache"), chunk_max_tokens=60, chunk_overlap_tokens=8
    )
    embedder = GatedEmbedder(batch_size=8)
    indexer = FilingsIndexer(
        settings,
        sec=object(),
        embedder_factory=lambda: embedder,
        tokenizer_factory=WordTokenizer,
        document_loader=load,
    )
    live = FakeLive(refs={MSFT: [MSFT_K24]})
    try:
        async with Client(build_server(settings, companies_live=live, filings_indexer=indexer)) as c:
            fetched = (await c.call_tool("companies_fetch_filings", {"companies": ["MSFT"]})).data
            assert fetched["total"] == 1 and fetched["model_download"] is False
            deadline = time.monotonic() + 20
            while indexer.status([MSFT_K24.doc_id])[0].state != "ready":
                assert time.monotonic() < deadline, "not indexed in time"
                await anyio.sleep(0.02)
            found = (
                await c.call_tool(
                    "companies_search_filings",
                    {"query": "capital expenditures", "companies": ["MSFT"], "limit": 3},
                )
            ).data
            assert found["indexed"] == "1/1" and found["not_ready"] == [] and found["returned"] == 3
            top = found["hits"][0]
            assert (top["company"], top["cik"], top["form"], top["filed"], top["url"]) == (
                "MICROSOFT CORP",
                MSFT,
                "10-K",
                "2024-07-30",
                MSFT_K24.url,
            )
            assert top["section"] and top["text"]
            read = (
                await c.call_tool(
                    "companies_read_filing",
                    {"doc_id": top["doc_id"], "chunk_id": top["chunk_id"], "around": 1},
                )
            ).data
            assert top["text"] in [chunk["text"] for chunk in read["chunks"]]
            section = (
                await c.call_tool("companies_read_filing", {"doc_id": top["doc_id"], "section": "Item 7"})
            ).data
            assert section["chunks"] and all(ch["section"] == "Item 7" for ch in section["chunks"])
            info = (await c.call_tool("server_info", {})).data["filings"]
            assert info["documents"] == {"ready": 1, "total": 1}
            assert info["chunker_version"] == "v2-t60-o8-ttl2" and info["model"] == indexer.model_name
    finally:
        indexer.close()


async def test_a_document_error_is_shown_bounded_and_masked():
    """I5: a status recorded by an earlier build kept the first line of the message: shown by its class."""
    leaky = status(MSFT_K24, "failed")
    leaky.error = f"OSError: [Errno 13] /home/someone/.cache/x for {SENTINEL.upper()} " + "x" * 5000
    kept = status(MSFT_K23, "failed")
    kept.error = "SecUnavailable: SEC stayed unreachable or throttled; fetch again later"
    res = await _call(FakeIndexer([leaky, kept]), "companies_fetch_filings", {"companies": ["MSFT"]})
    errors = {d["doc_id"]: d["error"] for d in res["documents"]}
    assert errors == {MSFT_K24.doc_id: "OSError", MSFT_K23.doc_id: kept.error}


# --- fix round 1 -------------------------------------------------------------------------------------


async def test_search_by_fetched_cik_name_or_ticker_works_with_sec_down():
    from company_corpus.live.sec import SecUnavailable

    live = FakeLive(fail={"entities": SecUnavailable("down")})
    docs = [status(MSFT_K24, "ready", 5, 5), status(APPLE_K24, "ready", 3, 3)]
    companies = {MSFT: ("MICROSOFT CORP", ["MSFT"]), APPLE: ("Apple Inc.", ["AAPL"])}
    for query in ("789019", MSFT, "msft", "microsoft"):
        indexer = FakeIndexer(docs, hits=[hit(MSFT_K24, 0), hit(APPLE_K24, 1)], companies=companies)
        res = await _call(
            indexer, "companies_search_filings", {"query": "capex", "companies": [query]}, live=live
        )
        assert [h["cik"] for h in res["hits"]] == [MSFT], query
        (search,) = [kw for m, kw in indexer.calls if m == "search"]
        assert search["doc_ids"] == [MSFT_K24.doc_id]
    assert "entities" not in live.calls  # never asked SEC
    # a CIK never fetched: [not_fetched], still without SEC
    msg = await _error(
        FakeIndexer(docs, companies=companies),
        "companies_search_filings",
        {"query": "capex", "companies": ["0000019617"]},
        live=live,
    )
    assert msg.startswith("[not_fetched]")
    # a name not fetched falls back to SEC, whose failure is coded
    msg = await _error(
        FakeIndexer(docs, companies=companies),
        "companies_search_filings",
        {"query": "capex", "companies": ["JPMORGAN"]},
        live=live,
    )
    assert msg.startswith("[sec_unavailable]")


async def test_fetch_records_names_and_tickers_for_local_resolution():
    indexer = FakeIndexer()
    async with Client(_server(indexer)) as c:
        await c.call_tool("companies_fetch_filings", {"companies": ["MSFT"]})
        indexer.docs = {k: status(MSFT_K24, "ready", 1, 1) for k in [MSFT_K24.doc_id]}
        indexer.hits = [hit(MSFT_K24, 0)]
        (enq,) = [kw for m, kw in indexer.calls if m == "enqueue"]
        assert enq["tickers"] == {MSFT: ["MSFT"]} and enq["names"] == {MSFT: "MICROSOFT CORP"}
        res = (await c.call_tool("companies_search_filings", {"query": "capex", "companies": ["MSFT"]})).data
    assert res["returned"] == 1


async def test_ambiguous_among_fetched_companies():
    docs = [status(MSFT_K24, "ready", 5, 5), status(APPLE_K24, "ready", 3, 3)]
    companies = {MSFT: ("MICROSOFT CORP", []), APPLE: ("APPLE CORP", [])}
    msg = await _error(
        FakeIndexer(docs, companies=companies),
        "companies_search_filings",
        {"query": "capex", "companies": ["corp"]},
    )
    assert msg.startswith("[ambiguous_company]") and MSFT in msg and APPLE in msg


@pytest.mark.parametrize(
    ("method", "exc", "tool", "args"),
    [
        ("search", KeyError("chunk_number"), "companies_search_filings", {"query": "capex"}),
        (
            "enqueue",
            IndexError("list index out of range"),
            "companies_fetch_filings",
            {"companies": ["MSFT"]},
        ),
        ("search", LookupError("internal"), "companies_search_filings", {"query": "capex"}),
    ],
)
async def test_internal_lookup_errors_are_masked(method, exc, tool, args, caplog):
    indexer = FakeIndexer([status(MSFT_K24, "ready", 5, 5)], fail={method: exc})
    msg = await _error(indexer, tool, args)
    assert msg.startswith("[backend_unavailable]") and str(exc.args[0]) not in msg
    assert type(exc).__name__ in caplog.text


async def test_search_without_companies_is_bounded():
    docs = [
        status(MSFT_K24, "ready", 5, 5),
        status(MSFT_K23, "ready", 5, 5),
        status(APPLE_K24, "ready", 3, 3),
    ]
    msg = await _error(
        FakeIndexer(docs), "companies_search_filings", {"query": "capex"}, filings_search_max_docs=2
    )
    assert (
        msg.startswith("[invalid_argument]") and "MOF_FILINGS_SEARCH_MAX_DOCS" in msg and "companies" in msg
    )
    # naming the companies is not bounded by it
    res = await _call(
        FakeIndexer(docs),
        "companies_search_filings",
        {"query": "capex", "companies": [MSFT]},
        filings_search_max_docs=1,
    )
    assert res["indexed"] == "2/2"


async def test_not_ready_is_capped_and_counted():
    refs = [filing(accession=f"0000789019-24-{n:06d}") for n in range(10, 15)]
    docs = [status(MSFT_K24, "ready", 5, 5)] + [status(r, "indexing", 1, 5) for r in refs]
    res = await _call(
        FakeIndexer(docs, hits=[hit(MSFT_K24, 0)]),
        "companies_search_filings",
        {"query": "capex"},
        filings_not_ready_listed=2,
    )
    assert res["not_ready_total"] == 5 and len(res["not_ready"]) == 2 and res["indexed"] == "1/6"


@pytest.mark.parametrize("tool", ["companies_fetch_filings", "companies_search_filings"])
async def test_an_empty_companies_list_is_invalid(tool):
    args = {"companies": []} if tool == "companies_fetch_filings" else {"query": "capex", "companies": []}
    msg = await _error(FakeIndexer([status(MSFT_K24, "ready", 1, 1)]), tool, args)
    assert msg.startswith("[invalid_argument]")


async def test_closed_and_locked_are_told_apart_by_type():
    from myopenfund_mcp.filings.indexer import IndexClosed

    msg = await _error(
        FakeIndexer(fail={"status": IndexClosed("locked? no: closed")}),
        "companies_search_filings",
        {"query": "x"},
    )
    assert msg.startswith("[index_unavailable]") and "shutting down" in msg
    msg = await _error(
        FakeIndexer(fail={"status": IndexUnavailable("closed folder")}),
        "companies_search_filings",
        {"query": "x"},
    )
    assert "another process" in msg and "shutting down" not in msg


async def test_fetch_answers_a_compact_list_when_the_full_one_is_too_large():
    refs = [filing(accession=f"0000789019-24-{n:06d}") for n in range(12)]
    live = FakeLive(refs={MSFT: refs})
    res = await _call(
        FakeIndexer(), "companies_fetch_filings", {"companies": ["MSFT"]}, live=live, max_result_chars=1500
    )
    assert res["total"] == 12 and set(res["documents"][0]) == {"doc_id", "state"}
    assert res["next"].startswith("the documents are queued")
    msg = await _error(
        FakeIndexer(),
        "companies_fetch_filings",
        {"companies": ["MSFT"]},
        live=FakeLive(refs={MSFT: [filing(accession=f"0000789019-24-{n:06d}") for n in range(40)]}),
        max_result_chars=1000,
    )
    assert msg.startswith("[result_too_large]") and "were queued" in msg


async def test_a_timeout_while_queuing_says_the_documents_may_be_queued():
    class Slow(FakeIndexer):
        def enqueue(self, refs, **kw):
            time.sleep(1.5)
            return super().enqueue(refs, **kw)

    msg = await _error(
        Slow(),
        "companies_fetch_filings",
        {"companies": ["MSFT"]},
        companies_query_timeout_s=1,
        sec_call_budget_s=0,
    )
    assert msg.startswith("[timeout]") and "may have been queued" in msg and "companies_fetch_filings" in msg


async def test_each_tool_call_writes_one_audit_line(caplog):
    caplog.set_level(logging.INFO, logger="myopenfund_mcp.audit")
    indexer = FakeIndexer(hits=[hit(MSFT_K24, 0)])
    async with Client(_server(indexer)) as c:
        await c.call_tool("companies_fetch_filings", {"companies": ["MSFT"]})
        indexer.docs[MSFT_K24.doc_id] = status(MSFT_K24, "ready", 1, 1)
        await c.call_tool("companies_search_filings", {"query": "capex secret-query"})
        await c.call_tool("companies_read_filing", {"doc_id": MSFT_K24.doc_id})
    lines = [r.getMessage() for r in caplog.records if r.name == "myopenfund_mcp.audit"]
    for tool in sorted(TOOLS):
        assert sum(f"tool={tool} " in line and "outcome=ok" in line for line in lines) == 1, (tool, lines)
    assert not any("secret-query" in line for line in lines)
