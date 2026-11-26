"""The filings indexer: offline, with FakeEmbedder, a word tokenizer and a hand-written 10-K in Markdown."""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from company_corpus.live.documents import FilingDocument, find_sections
from company_corpus.live.filings import FilingRef
from rag_engine.config import Settings as EngineSettings
from rag_engine.embedding import ONNX_MODELS, FakeEmbedder, ModelSpec
from rag_engine.local import LocalIndex

from myopenfund_mcp.config import Settings
from myopenfund_mcp.filings import indexer as indexer_module
from myopenfund_mcp.filings.indexer import (
    DocumentNotFetched,
    FilingsIndexer,
    IndexUnavailable,
    SectionNotFound,
    chunk_options,
    chunker_version,
    document_chunks,
    filing_title,
    model_files_cached,
)
from myopenfund_mcp.paging import result_chars

MARKDOWN = (Path(__file__).parent / "fixtures" / "filings" / "example_10k.md").read_text(encoding="utf-8")
CIK = "0000789019"
CONTACT = "someone@example.com"
SPEC = ModelSpec.from_settings(EngineSettings())
TITLE = "MICROSOFT CORP 10-K for the period ended 2024-06-30, filed 2024-07-30"


class WordTokenizer:
    """One token per word: counts close enough to a real tokenizer's for chunk sizes."""

    def encode(self, text: str) -> list[str]:
        return text.split()


class GatedEmbedder(FakeEmbedder):
    """FakeEmbedder that embeds passages in batches of `batch_size`, and blocks every passage call
    after the first `allowed` ones until `gate` is set."""

    def __init__(self, *, allowed: int | None = None, batch_size: int = 2):
        super().__init__(SPEC)
        self.batch_size = batch_size
        self.allowed = allowed
        self.gate = threading.Event()
        self.passage_calls = 0
        self.passages: list[str] = []
        self.sizes: list[int] = []
        self._count = threading.Lock()

    def encode_passage(self, texts):
        with self._count:
            self.passage_calls += 1
            number = self.passage_calls
        if self.allowed is not None and number > self.allowed:
            self.gate.wait(30)
        self.passages.extend(texts)
        self.sizes.append(len(texts))
        return super().encode_passage(texts)


def ref(
    accession: str = "0000789019-24-000001",
    *,
    form: str = "10-K",
    filed=dt.date(2024, 7, 30),
    report=dt.date(2024, 6, 30),
):
    return FilingRef(
        cik=CIK,
        accession=accession,
        form=form,
        report_date=report,
        filed=filed,
        accepted=None,
        primary_document="example-10k.htm",
        url=f"https://www.sec.gov/Archives/edgar/data/789019/{accession.replace('-', '')}",
        doc_id=f"sec-{accession}",
    )


def load(filing: FilingRef, sec) -> FilingDocument:
    return FilingDocument(ref=filing, markdown=MARKDOWN, sections=find_sections(MARKDOWN, filing.form))


def settings(tmp_path: Path, **overrides) -> Settings:
    values = {"cache_dir": str(tmp_path / "cache"), "chunk_max_tokens": 60, "chunk_overlap_tokens": 8}
    values.update(overrides)
    return Settings(**values)


def make(tmp_path: Path, embedder: FakeEmbedder | None = None, *, loader=load, **overrides) -> FilingsIndexer:
    embedder = embedder if embedder is not None else GatedEmbedder()
    overrides.setdefault("index_batch_size", getattr(embedder, "batch_size", 2))
    return FilingsIndexer(
        settings(tmp_path, **overrides),
        sec=object(),
        embedder_factory=lambda: embedder,
        tokenizer_factory=WordTokenizer,
        document_loader=loader,
    )


def wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


def state_of(indexer: FilingsIndexer, doc_id: str):
    (found,) = indexer.status([doc_id])
    return found


def pieces(tmp_path: Path):
    """The chunks the indexer cuts the fixture into, with their section."""
    filing = ref()
    return document_chunks(
        load(filing, None),
        options=chunk_options(settings(tmp_path)),
        tokenizer=WordTokenizer(),
        title=TITLE,
    )


@pytest.fixture
def item7(tmp_path):
    chunks = pieces(tmp_path)
    numbers = [n for n, (section, _) in enumerate(chunks) if section == "Item 7"]
    assert len(numbers) >= 4 and numbers == list(range(numbers[0], numbers[-1] + 1))
    return len(chunks), len(numbers)


def test_enqueue_returns_before_any_embedding_and_loads_the_model_lazily(tmp_path):
    embedder = GatedEmbedder(allowed=0)
    built = []

    def factory():
        built.append(1)
        return embedder

    indexer = FilingsIndexer(
        settings(tmp_path),
        sec=object(),
        embedder_factory=factory,
        tokenizer_factory=WordTokenizer,
        document_loader=load,
    )
    try:
        assert built == []  # never at start-up
        result = indexer.enqueue([ref()], names={CIK: "MICROSOFT CORP"})
        assert [s.doc_id for s in result] == ["sec-0000789019-24-000001"]
        assert result[0].state in ("queued", "fetching", "indexing")
        assert result[0].chunks_done == 0
        assert result.model_download is False  # an injected embedder: nothing to download
        assert embedder.passages == []
        embedder.gate.set()
        wait_until(lambda: state_of(indexer, "sec-0000789019-24-000001").state == "ready")
        assert built == [1]
    finally:
        embedder.gate.set()
        indexer.close()


def test_priority_section_is_searchable_before_the_document_is_ready(tmp_path, item7):
    total, n7 = item7
    embedder = GatedEmbedder(allowed=-(-n7 // 2), batch_size=2)  # exactly the Item 7 batches
    indexer = make(tmp_path, embedder)
    doc_id = "sec-0000789019-24-000001"
    try:
        indexer.enqueue([ref()], names={CIK: "MICROSOFT CORP"})
        wait_until(lambda: state_of(indexer, doc_id).chunks_done == n7)
        status = state_of(indexer, doc_id)
        assert status.state == "indexing" and status.chunks_total == total

        hits, not_ready = indexer.search("capital expenditures datacenter", doc_ids=[doc_id], limit=3)
        assert hits and all(hit.payload["section"] == "Item 7" for hit in hits)
        assert [s.doc_id for s in not_ready] == [doc_id]
        payload = hits[0].payload
        assert payload["doc_type"] == "10-K"
        assert payload["date"] == "2024-07-30"
        assert payload["title"] == TITLE
        assert payload["source_code"] == CIK
        assert payload["accession"] == "0000789019-24-000001"
        assert payload["url"].startswith("https://www.sec.gov/Archives/")
        assert payload["chunker_version"] == indexer.chunker_version

        embedder.gate.set()
        wait_until(lambda: state_of(indexer, doc_id).state == "ready")
        ready = state_of(indexer, doc_id)
        assert ready.chunks_done == ready.chunks_total == total
        hits, not_ready = indexer.search("Revenue Operating income Net income", doc_ids=[doc_id], limit=10)
        assert not_ready == []
        assert any(hit.payload["section"] == "Item 8" for hit in hits)
    finally:
        embedder.gate.set()
        indexer.close()


def company_ref(cik: str, year: int) -> FilingRef:
    """A 10-K of company `cik` for fiscal `year` (filed the next July)."""
    accession = f"{cik[-4:]}-{year % 100:02d}-000001"
    return FilingRef(
        cik=cik,
        accession=accession,
        form="10-K",
        report_date=dt.date(year, 6, 30),
        filed=dt.date(year, 7, 30),
        accepted=None,
        primary_document="example-10k.htm",
        url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}",
        doc_id=f"sec-{cik}-{year}",
    )


def record_units(indexer: FilingsIndexer, monkeypatch) -> list[tuple[str, str | None, int]]:
    """(doc_id, section, first chunk number) of every batch, in the order the batches are stored."""
    units: list[tuple[str, str | None, int]] = []
    real = indexer._record_batch

    def recording(doc_id, batch, sections, chunks, done):
        units.append((doc_id, sections[batch[0]], batch[0]))
        return real(doc_id, batch, sections, chunks, done)

    monkeypatch.setattr(indexer, "_record_batch", recording)
    return units


def run_in_order(tmp_path, monkeypatch, refs, **overrides):
    """Enqueue `refs` with the first batch held until every document is fetched and cut, so the order
    of the units does not depend on fetch timing; the units in the order they were stored."""
    embedder = GatedEmbedder(allowed=0, batch_size=2)
    indexer = make(tmp_path, embedder, **overrides)
    units = record_units(indexer, monkeypatch)
    try:
        indexer.enqueue(refs)
        wait_until(lambda: all(s.state == "indexing" for s in indexer.status()))
        embedder.gate.set()
        wait_until(lambda: all(s.state == "ready" for s in indexer.status()))
    finally:
        embedder.gate.set()
        indexer.close()
    return units


A, B = "0000000001", "0000000002"


def test_order_interleaves_companies_newest_document_first(tmp_path, monkeypatch, item7):
    """R41: two companies, two 10-Ks each, fetched in one call (oldest first, company by company): the
    first Item 7 batch of each company's newest document, round robin; then the rest of the Item 7
    sections (newest documents first, company by company); then the remainders in the same order."""
    _, n7 = item7
    refs = [company_ref(A, 2022), company_ref(A, 2023), company_ref(B, 2022), company_ref(B, 2023)]
    units = run_in_order(tmp_path, monkeypatch, refs, index_first_batches=1)
    new_a, old_a, new_b, old_b = (f"sec-{A}-2023", f"sec-{A}-2022", f"sec-{B}-2023", f"sec-{B}-2022")
    item7_batches = -(-n7 // 2)
    rest_batches = sum(1 for doc, section, _ in units if doc == new_a and section != "Item 7")
    assert rest_batches > 1 and len(units) == 4 * (item7_batches + rest_batches)
    order = [(doc, section == "Item 7") for doc, section, _ in units]
    expected = (
        [(new_a, True), (new_b, True)]  # pass 1: one batch each
        + [(new_a, True)] * (item7_batches - 1) + [(new_b, True)] * (item7_batches - 1)  # pass 2, rank 0
        + [(old_a, True)] * item7_batches + [(old_b, True)] * item7_batches  # pass 2, rank 1
        + [(new_a, False)] * rest_batches + [(new_b, False)] * rest_batches  # pass 3, rank 0
        + [(old_a, False)] * rest_batches + [(old_b, False)] * rest_batches  # pass 3, rank 1
    )
    assert order == expected
    # within a document, chunks still go in document order inside each pass
    for doc in (new_a, old_b):
        numbers = [n for d, s, n in units if d == doc and s == "Item 7"]
        assert numbers == sorted(numbers)


def test_order_first_unit_of_several_batches_round_robin(tmp_path, monkeypatch, item7):
    assert item7[1] > 2  # two Item 7 batches of 2 at least
    refs = [company_ref(A, 2023), company_ref(B, 2023)]
    units = run_in_order(tmp_path, monkeypatch, refs, index_first_batches=2)
    head = [(doc, section) for doc, section, _ in units[:4]]
    assert head == [(f"sec-{A}-2023", "Item 7"), (f"sec-{B}-2023", "Item 7")] * 2


def test_order_five_companies_each_get_a_first_passage_before_any_remainder(tmp_path, monkeypatch, item7):
    """R41, the failing shape of C11: five companies fetched in one call (their 10-Ks oldest first)."""
    ciks = [f"00000000{n:02d}" for n in range(1, 6)]
    refs = [company_ref(cik, year) for cik in ciks for year in (2022, 2023, 2024)]
    units = run_in_order(tmp_path, monkeypatch, refs, index_first_batches=1)
    assert [(doc, section) for doc, section, _ in units[:5]] == [
        (f"sec-{cik}-2024", "Item 7") for cik in ciks
    ]
    first_rest = next(i for i, (_, section, _) in enumerate(units) if section != "Item 7")
    assert all(section == "Item 7" for _, section, _ in units[:first_rest])
    assert len(units[:first_rest]) == 15 * -(-item7[1] // 2)  # every Item 7 before any remainder


def test_first_batches_zero_goes_document_by_document(tmp_path, monkeypatch, item7):
    refs = [company_ref(A, 2023), company_ref(B, 2023)]
    units = run_in_order(tmp_path, monkeypatch, refs, index_first_batches=0)
    n7b = -(-item7[1] // 2)
    assert [d for d, _, _ in units[:n7b]] == [f"sec-{A}-2023"] * n7b


def test_fetch_ahead_bounds_the_documents_cut_but_not_a_companys_first_round(tmp_path, monkeypatch):
    embedder = GatedEmbedder(allowed=0, batch_size=2)
    indexer = make(tmp_path, embedder, index_fetch_ahead=1, index_first_batches=1)
    units = record_units(indexer, monkeypatch)
    new_a, old_a, new_b = f"sec-{A}-2023", f"sec-{A}-2022", f"sec-{B}-2023"
    try:
        indexer.enqueue([company_ref(A, 2022), company_ref(A, 2023), company_ref(B, 2023)])
        wait_until(lambda: state_of(indexer, new_b).state == "indexing")  # past the bound: a first round
        time.sleep(0.3)
        assert state_of(indexer, old_a).state == "queued"  # held back by MOF_INDEX_FETCH_AHEAD
        embedder.gate.set()
        wait_until(lambda: all(s.state == "ready" for s in indexer.status()))
    finally:
        embedder.gate.set()
        indexer.close()
    assert [doc for doc, _, _ in units[:2]] == [new_a, new_b]


def test_beyond_the_fetch_ahead_bound_a_document_waits_for_a_cut_one_to_finish(tmp_path, monkeypatch):
    """More documents than MOF_INDEX_FETCH_AHEAD (2): the newest of each company is cut and gets its
    Item 7 first; each later document is fetched only once a cut one is finished, remainder included,
    so its Item 7 comes after that remainder (the documented limit of the bound)."""
    embedder = GatedEmbedder(allowed=0, batch_size=2)
    indexer = make(tmp_path, embedder, index_fetch_ahead=2, index_first_batches=1)
    units = record_units(indexer, monkeypatch)
    newest = [f"sec-{A}-2024", f"sec-{B}-2024"]
    held = [f"sec-{A}-2023", f"sec-{B}-2023", f"sec-{A}-2022", f"sec-{B}-2022"]  # their fetch order
    try:
        indexer.enqueue([company_ref(cik, year) for cik in (A, B) for year in (2022, 2023, 2024)])
        wait_until(lambda: all(state_of(indexer, doc).state == "indexing" for doc in newest))
        time.sleep(0.3)
        assert {state_of(indexer, doc).state for doc in held} == {"queued"}  # held back by the bound
        embedder.gate.set()
        wait_until(lambda: all(s.state == "ready" for s in indexer.status()))
    finally:
        embedder.gate.set()
        indexer.close()
    assert [doc for doc, _, _ in units[:2]] == newest  # pass 1
    first_rest = next(i for i, (_, section, _) in enumerate(units) if section != "Item 7")
    assert {doc for doc, _, _ in units[:first_rest]} == set(newest)  # the cut documents' Item 7 first
    last = {doc: max(i for i, (d, _, _) in enumerate(units) if d == doc) for doc in newest + held}
    for k, doc in enumerate(held):
        start = min(i for i, (d, _, _) in enumerate(units) if d == doc)
        assert units[start][1] == "Item 7"
        finished_before = sum(1 for other, end in last.items() if end < start)
        assert finished_before >= k + 1, (doc, finished_before)  # at most 2 documents cut and waiting


def test_a_company_whose_newest_document_fails_at_fetch_keeps_its_first_round(tmp_path, monkeypatch):
    """A, B, C x 2 10-Ks; B's newest fails at fetch: B's next document takes its first-round slot."""
    C = "0000000003"

    def loader(filing, sec):
        if filing.cik == B and filing.filed.year == 2024:
            raise RuntimeError("boom")
        return load(filing, sec)

    embedder = GatedEmbedder(allowed=0, batch_size=2)
    indexer = make(tmp_path, embedder, loader=loader, index_first_batches=1)
    units = record_units(indexer, monkeypatch)
    try:
        indexer.enqueue([company_ref(cik, year) for cik in (A, B, C) for year in (2023, 2024)])
        wait_until(
            lambda: state_of(indexer, f"sec-{B}-2024").state == "failed"
            and all(s.state == "indexing" for s in indexer.status() if s.state != "failed")
        )
        embedder.gate.set()
        wait_until(lambda: all(s.state in ("ready", "failed") for s in indexer.status()))
    finally:
        embedder.gate.set()
        indexer.close()
    assert [(doc, section) for doc, section, _ in units[:3]] == [
        (f"sec-{A}-2024", "Item 7"), (f"sec-{B}-2023", "Item 7"), (f"sec-{C}-2024", "Item 7")
    ]


def test_fetching_again_does_not_rerank_documents_in_flight(tmp_path, monkeypatch):
    """Polling by fetching the same filings again queues nothing new and changes no rank: once A's
    newest is ready, A's older remainder still comes after B's newest remainder (rank 1 after rank 0)."""
    refs = [company_ref(cik, year) for cik in (A, B) for year in (2023, 2024)]
    probe = run_in_order(tmp_path / "probe", monkeypatch, refs, index_first_batches=1)
    new_a, old_a, new_b = f"sec-{A}-2024", f"sec-{A}-2023", f"sec-{B}-2024"
    through_new_a = max(i for i, (d, _, _) in enumerate(probe) if d == new_a) + 1
    assert through_new_a < len(probe)  # A's newest is ready before the end

    embedder = GatedEmbedder(allowed=through_new_a, batch_size=2)
    indexer = make(tmp_path, embedder, index_first_batches=1)
    units = record_units(indexer, monkeypatch)
    try:
        indexer.enqueue(refs)
        wait_until(lambda: state_of(indexer, new_a).state == "ready")
        again = indexer.enqueue(refs)  # nothing new: every document is ready or in flight
        assert {s.state for s in again} <= {"ready", "indexing"}
        embedder.gate.set()
        wait_until(lambda: all(s.state == "ready" for s in indexer.status()))
    finally:
        embedder.gate.set()
        indexer.close()
    rest_b = [i for i, (d, s, _) in enumerate(units) if d == new_b and s != "Item 7"]
    rest_old_a = [i for i, (d, s, _) in enumerate(units) if d == old_a and s != "Item 7"]
    assert max(rest_b) < min(rest_old_a)


def test_default_batch_size_is_the_onnx_batch_size():
    from rag_engine.embedding import ONNX_BATCH_SIZE

    assert Settings().index_batch_size == Settings.from_env({}).index_batch_size == ONNX_BATCH_SIZE


def test_index_batch_size_setting_sizes_the_batches(tmp_path, item7):
    total, n7 = item7
    embedder = GatedEmbedder(batch_size=2)
    indexer = make(tmp_path, embedder, index_batch_size=3)
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "ready")
    finally:
        indexer.close()
    assert len(embedder.passages) == total
    assert max(embedder.sizes) == 3  # MOF_INDEX_BATCH_SIZE, not the embedder's own batch size
    assert embedder.sizes[: -(-n7 // 3)] == [3] * (n7 // 3) + ([n7 % 3] if n7 % 3 else [])


def test_restart_mid_document_shows_partial_then_resumes_to_ready(tmp_path, item7):
    """Review Focus 3: never `ready` after a restart mid-document; the next enqueue finishes it."""
    total, n7 = item7
    doc_id = "sec-0000789019-24-000001"
    first = GatedEmbedder(allowed=-(-n7 // 2), batch_size=2)
    a = make(tmp_path, first)
    a.enqueue([ref()], names={CIK: "MICROSOFT CORP"})
    wait_until(lambda: state_of(a, doc_id).chunks_done == n7)
    # The process "dies" here: the worker stops after its current batch, nothing marks the document.
    closer = threading.Thread(target=a.close)
    closer.start()
    wait_until(a._stop.is_set)
    first.gate.set()
    closer.join(30)
    assert not closer.is_alive()

    second = GatedEmbedder(batch_size=3)
    b = make(tmp_path, second)
    try:
        partial = state_of(b, doc_id)
        assert partial.state == "partial"
        assert 0 < partial.chunks_done < total
        done_before = partial.chunks_done

        result = b.enqueue([ref()], names={CIK: "MICROSOFT CORP"})
        assert result[0].state in ("queued", "fetching", "indexing", "ready")
        wait_until(lambda: state_of(b, doc_id).state == "ready")
        ready = state_of(b, doc_id)
        assert ready.chunks_done == ready.chunks_total == total
        assert len(second.passages) == total - done_before  # resumed, not redone
        collection = b.collection
        index_dir = b.index_dir
    finally:
        b.close()
    with LocalIndex(index_dir, collection=collection, dim=SPEC.dim) as index:
        assert index.count(doc_id) == total


def test_second_indexer_on_the_same_cache_is_unavailable(tmp_path):
    """Review Focus 1: two MCP processes on one cache; the second answers IndexUnavailable, no crash."""
    first = make(tmp_path)
    second = make(tmp_path)
    try:
        assert first.status() == []
        for call in (
            lambda: second.status(),
            lambda: second.enqueue([ref()]),
            lambda: second.search("revenue", doc_ids=None, limit=3),
            lambda: second.read("sec-0000789019-24-000001"),
        ):
            with pytest.raises(IndexUnavailable, match="another process"):
                call()
    finally:
        first.close()
    assert second.status() == []  # the lock is released: the second one recovers
    second.close()


def walk(indexer: FilingsIndexer, doc_id: str, section: str | None = None) -> list[dict]:
    """Every page of a read, following `next_chunk_id` / `next_offset`."""
    pages = [indexer.read(doc_id, section=section)]
    while pages[-1]["more"]:
        last = pages[-1]
        pages.append(
            indexer.read(
                doc_id,
                section=section,
                chunk_id=last["next_chunk_id"],
                offset=last["next_offset"],
                around=None,
            )
        )
        assert len(pages) < 500
    return pages


def rebuilt(pages: list[dict]) -> list[tuple[str, str]]:
    """(chunk_id, whole text) in order, joining the pieces of cut chunks."""
    out: list[tuple[str, str]] = []
    for page in pages:
        for piece in page["chunks"]:
            if out and out[-1][0] == piece["chunk_id"]:
                assert piece["offset"] == len(out[-1][1])
                out[-1] = (piece["chunk_id"], out[-1][1] + piece["text"])
            else:
                assert "offset" not in piece
                out.append((piece["chunk_id"], piece["text"]))
    return out


@pytest.mark.parametrize("max_tokens", [60, 400])
def test_read_walks_a_section_and_the_document_within_the_budget(tmp_path, max_tokens):
    indexer = make(tmp_path, GatedEmbedder(batch_size=4), max_result_chars=1_000, chunk_max_tokens=max_tokens)
    doc_id = "sec-0000789019-24-000001"
    expected = document_chunks(
        load(ref(), None),
        options=chunk_options(settings(tmp_path, chunk_max_tokens=max_tokens)),
        tokenizer=WordTokenizer(),
        title=TITLE,
    )
    try:
        indexer.enqueue([ref()], names={CIK: "MICROSOFT CORP"})
        wait_until(lambda: state_of(indexer, doc_id).state == "ready")

        pages = walk(indexer, doc_id, "Item 7")
        assert all(result_chars(page) <= 1_000 for page in pages)
        assert pages[0]["chunks"][0]["text"].startswith("Revenue increased sixteen percent")
        assert {chunk["section"] for page in pages for chunk in page["chunks"]} == {"Item 7"}
        item7 = [(str(n), chunk.text) for n, (section, chunk) in enumerate(expected) if section == "Item 7"]
        assert rebuilt(pages) == item7
        assert pages[-1]["more"] is False and pages[-1]["next_chunk_id"] is None

        pages = walk(indexer, doc_id)
        assert len(pages) > 1 and all(result_chars(page) <= 1_000 for page in pages)
        assert rebuilt(pages) == [(str(n), chunk.text) for n, (_, chunk) in enumerate(expected)]
        cut = any(piece.get("text_cut") for page in pages for piece in page["chunks"])
        assert cut is (max_tokens == 400)  # 400-token chunks are longer than the 1000-character budget
    finally:
        indexer.close()


def test_read_window_and_errors(tmp_path):
    indexer = make(tmp_path, GatedEmbedder(batch_size=4))
    doc_id = "sec-0000789019-24-000001"
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, doc_id).state == "ready")
        around = indexer.read(doc_id, chunk_id="5", around=1)
        assert [c["chunk_id"] for c in around["chunks"]] == ["4", "5", "6"]
        assert around["more"] is False
        assert [c["chunk_id"] for c in indexer.read(doc_id, chunk_id=0, around=2)["chunks"]] == [
            "0",
            "1",
            "2",
        ]
        first7 = indexer.read(doc_id, section="Item 7")["chunks"][0]["chunk_id"]
        with pytest.raises(LookupError, match="in section 'Item 7'"):
            indexer.read(doc_id, section="Item 7", chunk_id=int(first7) - 1)
        with pytest.raises(SectionNotFound, match="Item 7"):
            indexer.read(doc_id, section="Item 99")
        with pytest.raises(DocumentNotFetched):
            indexer.read("sec-never-fetched")
        with pytest.raises(ValueError, match="offset"):
            indexer.read(doc_id, chunk_id="5", around=1, offset=3)  # window mode
        with pytest.raises(ValueError, match="past the end"):
            indexer.read(doc_id, chunk_id="5", offset=10_000, around=None)
    finally:
        indexer.close()


@pytest.mark.parametrize("eigenmind", [False, True])
def test_search_returns_limit_hits_with_and_without_eigenmind(tmp_path, monkeypatch, eigenmind):
    calls = []
    real = indexer_module.select_diverse

    def spy(hits, vectors, limit):
        calls.append(len(hits))
        return real(hits, vectors, limit)

    monkeypatch.setattr(indexer_module, "select_diverse", spy)
    indexer = make(tmp_path, GatedEmbedder(batch_size=8), search_eigenmind=eigenmind)
    doc_id = "sec-0000789019-24-000001"
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, doc_id).state == "ready")
        hits, not_ready = indexer.search("cash investments revenue", doc_ids=[doc_id], limit=4)
        assert len(hits) == 4 and not_ready == []
        assert bool(calls) is eigenmind
    finally:
        indexer.close()


def test_search_filters_on_forms_and_period_and_skips_unindexed_documents(tmp_path):
    embedder = GatedEmbedder(batch_size=8)
    indexer = make(tmp_path, embedder)
    old = ref("0000789019-23-000001", filed=dt.date(2023, 7, 27), report=dt.date(2023, 6, 30))
    new = ref()
    try:
        indexer.enqueue([old, new])
        wait_until(lambda: all(s.state == "ready" for s in indexer.status()))
        hits, _ = indexer.search("revenue", doc_ids=None, period_from=dt.date(2024, 1, 1), limit=20)
        assert hits and {hit.doc_id for hit in hits} == {new.doc_id}
        hits, _ = indexer.search("revenue", doc_ids=None, forms=["10-Q"], limit=5)
        assert hits == []
        hits, _ = indexer.search("revenue", doc_ids=None, forms=["10-K"], limit=20)
        assert {hit.doc_id for hit in hits} == {old.doc_id, new.doc_id}
    finally:
        indexer.close()


def test_failed_document_is_marked_and_the_worker_continues(tmp_path):
    def loader(filing, sec):
        if filing.accession.endswith("000002"):
            raise RuntimeError(f"SEC refused the request (User-Agent {CONTACT})\nsecond line")
        return load(filing, sec)

    indexer = make(tmp_path, GatedEmbedder(batch_size=8), loader=loader)
    bad, good = ref("0000789019-24-000002"), ref()
    try:
        indexer.enqueue([bad, good])
        wait_until(lambda: {s.state for s in indexer.status()} == {"failed", "ready"})
        failed = state_of(indexer, bad.doc_id)
        assert failed.state == "failed"
        assert failed.error == "RuntimeError"  # an unknown class: its name only, never its message
    finally:
        indexer.close()


SENTINEL = "Jane Sentinel jane.sentinel@example.org"


@pytest.mark.parametrize(
    "message",
    [
        "/home/someone/.cache/myopenfund-mcp/sec/0000789019/secret.htm not readable",
        "GET https://user:p4ssw0rd@proxy.example.com/Archives failed",
        "x" * 5000,
        f"refused for {SENTINEL.upper()}",
    ],
)
@pytest.mark.parametrize(
    ("exc_type", "expected"),
    [
        (RuntimeError, "RuntimeError"),
        ("UnsupportedDocument", "UnsupportedDocument: the primary document is not an HTML file"),
        ("DocumentTooLarge", "DocumentTooLarge: the primary document is larger than the conversion limit"),
        (
            "PrimaryDocumentNotFound",
            "PrimaryDocumentNotFound: the filing index lists no HTML document of the filing's form",
        ),
        ("SecUnavailable", "SecUnavailable: SEC stayed unreachable or throttled; fetch again later"),
        (
            "SecRefused",
            "SecRefused: SEC refused the request as an undeclared automated tool (check MOF_SEC_CONTACT)",
        ),
        ("SecNotFound", "SecNotFound: SEC answered 404 for a file of the filing"),
    ],
)
def test_document_error_text_is_fixed_and_bounded(tmp_path, message, exc_type, expected):
    """I5: the recorded error is a fixed text per known class, else the class name: no path, no
    credential URL, no long text, no contact (whatever its case)."""
    from company_corpus.live import documents, sec

    cls = exc_type if isinstance(exc_type, type) else getattr(documents, exc_type, None) or getattr(sec, exc_type)

    def loader(filing, _sec):
        raise cls(message)

    indexer = make(tmp_path, loader=loader, sec_contact=SENTINEL)
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "failed")
        error = state_of(indexer, DOC).error
    finally:
        indexer.close()
    assert error == expected
    assert len(error) <= 120
    for leak in ("/home", "p4ssw0rd", "xxxxx", "SENTINEL", "sentinel"):
        assert leak not in error


def test_sec_http_error_keeps_its_status_only(tmp_path):
    from company_corpus.live.sec import SecHttpError

    def loader(filing, _sec):
        raise SecHttpError(401, f"401 for https://{SENTINEL}@sec.gov/x")

    indexer = make(tmp_path, loader=loader)
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "failed")
        assert state_of(indexer, DOC).error == "SecHttpError: SEC answered HTTP 401 for a file of the filing"
    finally:
        indexer.close()


def test_public_error_masks_records_of_an_earlier_build():
    from myopenfund_mcp.filings.indexer import public_error

    assert public_error(None) is None
    assert public_error("RuntimeError") == "RuntimeError"
    assert public_error("SecUnavailable: SEC stayed unreachable or throttled; fetch again later") == (
        "SecUnavailable: SEC stayed unreachable or throttled; fetch again later"
    )
    assert public_error("OSError: [Errno 13] /home/someone/x") == "OSError"
    assert public_error("x" * 5000) == "error"


def test_failed_error_never_holds_the_sec_contact(tmp_path):
    def loader(filing, sec):
        raise RuntimeError(f"refused for {CONTACT}")

    indexer = FilingsIndexer(
        settings(tmp_path, sec_contact=CONTACT),
        sec=object(),
        embedder_factory=lambda: GatedEmbedder(),
        tokenizer_factory=WordTokenizer,
        document_loader=loader,
    )
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, ref().doc_id).state == "failed")
        assert CONTACT not in state_of(indexer, ref().doc_id).error
    finally:
        indexer.close()


def test_ready_document_stays_ready_and_is_not_embedded_again(tmp_path):
    embedder = GatedEmbedder(batch_size=8)
    indexer = make(tmp_path, embedder)
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, ref().doc_id).state == "ready")
        embedded = len(embedder.passages)
        again = indexer.enqueue([ref()])
        assert again[0].state == "ready"
        time.sleep(0.1)
        assert len(embedder.passages) == embedded
    finally:
        indexer.close()


def test_default_factory_reports_the_model_download_once(tmp_path, monkeypatch):
    built = {}

    class FakeOnnx(GatedEmbedder):
        def __init__(self, spec, *, cache_dir, threads, provider):
            super().__init__(batch_size=8)
            built.update(spec=spec, cache_dir=cache_dir, threads=threads, provider=provider)

    def fake_tokenizer(cache_dir, model_name):
        built["tokenizer"] = (cache_dir, model_name)
        return WordTokenizer()

    monkeypatch.setattr(indexer_module, "OnnxEmbedder", FakeOnnx)
    monkeypatch.setattr(indexer_module, "load_tokenizer", fake_tokenizer)
    indexer = FilingsIndexer(settings(tmp_path, rag_onnx_threads=2), sec=object(), document_loader=load)
    try:
        assert indexer.enqueue([]).model_download is False  # nothing queued: nothing reported
        assert indexer.enqueue([ref()]).model_download is True  # nothing in <cache>/models yet
        wait_until(lambda: state_of(indexer, ref().doc_id).state == "ready")
        assert indexer.enqueue([ref("0000789019-23-000001")]).model_download is False  # loaded now
        models = str(Path(tmp_path, "cache", "models"))
        assert built["cache_dir"] == models and built["tokenizer"] == (models, SPEC.name)
        assert built["threads"] == 2 and built["provider"] == "cpu"
    finally:
        indexer.close()


def test_model_files_cached_looks_for_every_pinned_file(tmp_path):
    known = ONNX_MODELS[SPEC.name]
    assert model_files_cached(tmp_path, SPEC.name) is False
    snapshot = tmp_path / f"models--{SPEC.name.replace('/', '--')}" / "snapshots" / known.revision
    for name in known.files:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(b"x")
    assert model_files_cached(tmp_path, SPEC.name) is True
    (snapshot / "tokenizer.json").unlink()
    assert model_files_cached(tmp_path, SPEC.name) is False
    assert model_files_cached(tmp_path, "some/unpinned-model") is False


def test_document_chunks_label_each_chunk_with_its_section(tmp_path):
    chunks = pieces(tmp_path)
    sections = [section for section, _ in chunks]
    assert sections[0] is None  # the cover page, before Item 1
    assert sections.index("Item 1") < sections.index("Item 7") < sections.index("Item 8")
    item8 = [chunk.text for section, chunk in chunks if section == "Item 8"]
    assert any("Net income" in text for text in item8)


def test_closed_indexer_refuses_calls(tmp_path):
    indexer = make(tmp_path)
    indexer.close()
    indexer.close()  # twice is harmless
    with pytest.raises(IndexUnavailable, match="closed"):
        indexer.status()


DOC = "sec-0000789019-24-000001"


def stop_midway(tmp_path: Path, n7: int, *, names=None, loader=load, **overrides) -> None:
    """Index the Item 7 batches, then close the indexer as a dying process would leave it."""
    embedder = GatedEmbedder(allowed=-(-n7 // 2), batch_size=2)
    indexer = make(tmp_path, embedder, loader=loader, **overrides)
    indexer.enqueue([ref()], names=names)
    wait_until(lambda: state_of(indexer, DOC).chunks_done == n7)
    closer = threading.Thread(target=indexer.close)
    closer.start()
    wait_until(indexer._stop.is_set)
    embedder.gate.set()
    closer.join(30)
    assert not closer.is_alive()


def stored_count(indexer: FilingsIndexer, doc_id: str = DOC) -> int:
    with LocalIndex(indexer.index_dir, collection=indexer.collection, dim=SPEC.dim) as index:
        return index.count(doc_id)


def test_resume_keeps_the_title_of_the_first_indexing(tmp_path, item7):
    total, n7 = item7
    stop_midway(tmp_path, n7, names={CIK: "MICROSOFT CORP"})
    second = GatedEmbedder(batch_size=8)
    b = make(tmp_path, second)
    try:
        b.enqueue([ref()], names={CIK: "Microsoft Corporation (renamed)"})
        wait_until(lambda: state_of(b, DOC).state == "ready")
        hits, _ = b.search("revenue cash debt risk", doc_ids=[DOC], limit=total)
        assert len(hits) == total
        assert {hit.payload["title"] for hit in hits} == {TITLE}
        assert len(second.passages) < total  # resumed, not recut
    finally:
        b.close()


def test_resume_recuts_when_the_cut_changed(tmp_path, item7):
    total, n7 = item7
    stop_midway(tmp_path, n7)
    changed = MARKDOWN.replace("Net income | 88,136", "Net income | 88,137")

    def new_loader(filing, sec):
        return FilingDocument(ref=filing, markdown=changed, sections=find_sections(changed, filing.form))

    second = GatedEmbedder(batch_size=8)
    b = make(tmp_path, second, loader=new_loader)
    try:
        b.enqueue([ref()])
        wait_until(lambda: state_of(b, DOC).state == "ready")
        assert len(second.passages) == total  # same count, another cut: indexed again from scratch
        texts = [c["text"] for c in b.read(DOC, section="Item 8")["chunks"]]
        assert any("88,137" in text for text in texts) and not any("88,136" in text for text in texts)
    finally:
        b.close()
    assert stored_count(b) == total


def test_crash_between_index_add_and_status_write_is_repaired(tmp_path, item7, monkeypatch):
    total, _ = item7
    embedder = GatedEmbedder(batch_size=2)
    indexer = make(tmp_path, embedder)
    real = indexer._record_batch
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise sqlite3.OperationalError("disk I/O error")
        return real(*args, **kwargs)

    monkeypatch.setattr(indexer, "_record_batch", flaky)
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "failed")
        monkeypatch.setattr(indexer, "_record_batch", real)
        before = len(embedder.passages)
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "ready")
        assert len(embedder.passages) - before == total  # the index held a batch without its record: redone
        assert state_of(indexer, DOC).chunks_done == total
    finally:
        indexer.close()
    assert stored_count(indexer) == total


def test_ready_document_missing_from_the_index_is_partial_when_touched(tmp_path):
    a = make(tmp_path, GatedEmbedder(batch_size=8))
    a.enqueue([ref()])
    wait_until(lambda: state_of(a, DOC).state == "ready")
    a.close()
    with LocalIndex(a.index_dir, collection=a.collection, dim=SPEC.dim) as index:
        index.delete(DOC)
    b = make(tmp_path, GatedEmbedder(batch_size=8))
    try:
        assert state_of(b, DOC).state == "ready"  # not checked at open (lazy)
        assert b.read(DOC)["state"] == "partial"  # first touch: the index lacks its chunks
        b.enqueue([ref()])
        wait_until(lambda: state_of(b, DOC).state == "ready")
        total_chunks = state_of(b, DOC).chunks_total
    finally:
        b.close()
    assert stored_count(b) == total_chunks


def test_several_workers_index_documents_in_parallel(tmp_path):
    refs = [ref(f"0000789019-24-00000{n}") for n in range(1, 5)]
    indexer = make(tmp_path, GatedEmbedder(batch_size=3), index_workers=3)
    try:
        indexer.enqueue(refs)
        wait_until(lambda: [s.state for s in indexer.status()] == ["ready"] * 4)
        assert len(indexer._workers) == 3
        assert all(s.chunks_done == s.chunks_total for s in indexer.status())
    finally:
        indexer.close()
    with LocalIndex(indexer.index_dir, collection=indexer.collection, dim=SPEC.dim) as index:
        counts = {index.count(r.doc_id) for r in refs}
    assert len(counts) == 1 and counts.pop() > 0


def test_cache_off_keeps_the_model_in_the_persistent_cache(tmp_path):
    indexer = make(tmp_path, GatedEmbedder(batch_size=8), cache="off")
    try:
        assert indexer.models_dir == tmp_path / "cache" / "models"  # R36: software, not data
        assert not str(indexer.root).startswith(str(tmp_path))
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "ready")
        assert indexer.index_dir.is_relative_to(indexer.root)
        assert not (tmp_path / "cache" / "index").exists()
    finally:
        indexer.close()


def test_close_does_not_wait_past_its_timeout(tmp_path):
    loading = threading.Event()
    release = threading.Event()

    def slow_model():  # a model download in progress
        loading.set()
        release.wait(30)
        return GatedEmbedder()

    indexer = FilingsIndexer(
        settings(tmp_path, index_close_timeout_s=0),
        sec=object(),
        embedder_factory=slow_model,
        tokenizer_factory=WordTokenizer,
        document_loader=load,
    )
    try:
        indexer.enqueue([ref()])
        assert loading.wait(10)
        started = time.monotonic()
        indexer.close()
        assert time.monotonic() - started < 2
        with pytest.raises(IndexUnavailable):
            indexer.status()
    finally:
        release.set()
    for worker in [*indexer._workers, indexer._fetcher]:
        worker.join(10)
        assert not worker.is_alive()
    b = make(tmp_path)
    try:
        assert state_of(b, DOC).state == "partial"  # the worker stopped after the model load
    finally:
        b.close()


def test_search_does_not_wait_for_a_model_load_once_loaded(tmp_path):
    embedder = GatedEmbedder(batch_size=8)
    indexer = make(tmp_path, embedder)
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "ready")
        with indexer._model_lock:  # a worker would hold it during a load
            hits, _ = indexer.search("revenue", doc_ids=[DOC], limit=2)
        assert len(hits) == 2
    finally:
        indexer.close()


def test_empty_document_fails_with_no_text(tmp_path):
    def empty(filing, sec):
        return FilingDocument(ref=filing, markdown="", sections=())

    indexer = make(tmp_path, loader=empty)
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "failed")
        assert "no text" in state_of(indexer, DOC).error
    finally:
        indexer.close()


def test_unknown_doc_ids_are_reported(tmp_path):
    indexer = make(tmp_path, GatedEmbedder(batch_size=8))
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "ready")
        found = indexer.status(["sec-nope", DOC])
        assert [s.doc_id for s in found] == [DOC] and found.unknown == ["sec-nope"]
        result = indexer.search("revenue", doc_ids=[DOC, "sec-nope"], limit=2)
        hits, not_ready = result
        assert len(hits) == 2 and not_ready == [] and result.unknown == ["sec-nope"]
        assert indexer.search("revenue", doc_ids=["sec-nope"], limit=2).unknown == ["sec-nope"]
    finally:
        indexer.close()


def test_the_sec_contact_never_reaches_the_log(tmp_path, monkeypatch, caplog):
    def loader(filing, sec):
        try:
            raise OSError(f"connection refused for User-Agent {CONTACT}")
        except OSError as exc:
            raise RuntimeError("download failed") from exc

    indexer = FilingsIndexer(
        settings(tmp_path, sec_contact=CONTACT),
        sec=object(),
        embedder_factory=lambda: GatedEmbedder(),
        tokenizer_factory=WordTokenizer,
        document_loader=loader,
    )
    real_update = indexer._update
    failures = []

    def update(doc_id, **values):
        if values.get("state") == "failed":
            failures.append(1)
            try:
                raise ValueError(f"cannot write for {CONTACT}")
            except ValueError as exc:
                raise sqlite3.OperationalError("database is locked") from exc
        return real_update(doc_id, **values)

    monkeypatch.setattr(indexer, "_update", update)
    caplog.set_level(logging.DEBUG, logger="myopenfund_mcp")
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: failures and DOC not in indexer._jobs)
        assert "RuntimeError" in caplog.text and "OperationalError" in caplog.text
        assert CONTACT not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    finally:
        indexer.close()


def test_open_counts_no_document(tmp_path, monkeypatch):
    """N1: reconciliation is lazy; opening the index never counts a document's chunks."""
    a = make(tmp_path, GatedEmbedder(batch_size=8))
    a.enqueue([ref(), ref("0000789019-23-000001")])
    wait_until(lambda: [s.state for s in a.status()] == ["ready", "ready"])
    a.close()
    counted = []
    real = LocalIndex.count
    monkeypatch.setattr(
        LocalIndex, "count", lambda self, doc_id: counted.append(doc_id) or real(self, doc_id)
    )
    b = make(tmp_path, GatedEmbedder(batch_size=8))
    try:
        assert len(b.status()) == 2
        assert counted == []
        b.search("revenue", doc_ids=[DOC], limit=2)
        b.search("revenue", doc_ids=[DOC], limit=2)
        assert counted == [DOC]  # once, when first touched
        b.search("revenue", doc_ids=None, limit=2)
        assert counted == [DOC]  # doc_ids None: no reconciliation (the tools always pass doc_ids)
    finally:
        b.close()


def test_search_reports_the_matched_scope(tmp_path):
    indexer = make(tmp_path, GatedEmbedder(batch_size=8))
    try:
        indexer.enqueue([ref(), ref("0000789019-24-000002", form="10-Q", report=dt.date(2024, 3, 31))])
        wait_until(lambda: [s.state for s in indexer.status()] == ["ready", "ready"])
        both = indexer.search("revenue", doc_ids=[DOC, "sec-0000789019-24-000002"], limit=2)
        assert {s.doc_id for s in both.matched} == {DOC, "sec-0000789019-24-000002"}
        only_k = indexer.search("revenue", doc_ids=[DOC, "sec-0000789019-24-000002"], forms=["10-K"], limit=2)
        assert [s.doc_id for s in only_k.matched] == [DOC]
        assert indexer.search("revenue", doc_ids=[], limit=2).matched == []
        assert indexer.opened and indexer.model_name == SPEC.name
    finally:
        indexer.close()
    assert not indexer.opened


def test_fill_never_exceeds_the_budget_with_the_next_fields(tmp_path):
    """N2: exact fits with a long next chunk id, over every budget from tiny to roomy."""
    rows = [(1000 + n, "Item 7", f"word{n} " * 30) for n in range(3)]
    base = {
        "doc_id": DOC,
        "section": None,
        "chunks": [],
        "more": False,
        "next_chunk_id": None,
        "next_offset": 0,
    }
    for budget in range(result_chars(base) + 100, 1_200):
        indexer = FilingsIndexer(Settings(cache_dir=str(tmp_path), max_result_chars=budget), sec=object())
        result = {**base, "chunks": []}
        indexer._fill(result, iter(rows), 0)
        assert result_chars(result) <= budget, budget
        returned = "".join(c["text"] for c in result["chunks"])
        if result["more"]:
            number = int(result["next_chunk_id"])
            texts = {r[0]: r[2] for r in rows}
            assert 0 <= result["next_offset"] < len(texts[number])
        else:
            assert returned == "".join(r[2] for r in rows)


def test_chunk_id_without_around_reads_forward(tmp_path):
    indexer = make(tmp_path, GatedEmbedder(batch_size=8))
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "ready")
        total = state_of(indexer, DOC).chunks_total
        page = indexer.read(DOC, chunk_id="3")
        assert [c["chunk_id"] for c in page["chunks"]] == [str(n) for n in range(3, total)]
    finally:
        indexer.close()


def test_model_load_creates_a_private_cache_folder(tmp_path):
    indexer = make(tmp_path, GatedEmbedder(batch_size=8), cache="off")
    try:
        indexer.enqueue([ref()])
        wait_until(lambda: state_of(indexer, DOC).state == "ready")
        assert (tmp_path / "cache").stat().st_mode & 0o777 == 0o700
    finally:
        indexer.close()


EXIT_SCRIPT = """
import os, sys, threading, time
sys.path.insert(0, {tests!r})
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
import test_filings_indexer as t
from myopenfund_mcp.filings.indexer import FilingsIndexer
embedder = t.GatedEmbedder(allowed=1, batch_size=2)  # the second batch blocks forever
indexer = FilingsIndexer(
    t.settings(__import__("pathlib").Path({tmp!r}), cache="off", index_close_timeout_s=1),
    sec=object(), embedder_factory=lambda: embedder, tokenizer_factory=t.WordTokenizer,
    document_loader=t.load,
)
indexer.enqueue([t.ref()])
deadline = time.monotonic() + 30
while embedder.passage_calls < 2:
    assert time.monotonic() < deadline
    time.sleep(0.01)
print(indexer.root, flush=True)
sys.exit(0)  # atexit: close (bounded), then the temporary root is removed
"""


def test_cache_off_exit_mid_batch_removes_the_temporary_root(tmp_path):
    """N7: a process with MOF_CACHE=off exits while a worker is inside a batch."""
    import subprocess
    import sys

    script = EXIT_SCRIPT.format(tests=str(Path(__file__).parent), tmp=str(tmp_path))
    done = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    root = Path(done.stdout.strip().splitlines()[-1])
    assert root.name.startswith("myopenfund-mcp-filings-")
    assert not root.exists()


def test_fetched_companies_keep_names_and_tickers(tmp_path):
    indexer = make(tmp_path, GatedEmbedder(batch_size=8))
    try:
        assert indexer.fetched_companies() == []
        indexer.enqueue([ref()], names={CIK: "MICROSOFT CORP"}, tickers={CIK: ["MSFT"]})
        indexer.enqueue([ref("0000789019-23-000001")])  # no names given: the recorded ones stay
        assert indexer.fetched_companies() == [{"cik": CIK, "name": "MICROSOFT CORP", "tickers": ["MSFT"]}]
    finally:
        indexer.close()


def test_reconciliation_keeps_each_document_checked_as_it_goes(tmp_path, monkeypatch):
    a = make(tmp_path, GatedEmbedder(batch_size=8))
    a.enqueue([ref(), ref("0000789019-23-000001")])
    wait_until(lambda: [s.state for s in a.status()] == ["ready", "ready"])
    a.close()
    real, seen = LocalIndex.count, []

    def count(self, doc_id):
        seen.append(doc_id)
        if len(seen) == 2:
            raise RuntimeError("abandoned midway")
        return real(self, doc_id)

    monkeypatch.setattr(LocalIndex, "count", count)
    b = make(tmp_path, GatedEmbedder(batch_size=8))
    try:
        b.status()  # opens
        with pytest.raises(RuntimeError):
            b._reconcile([DOC, "sec-0000789019-23-000001"])
        assert seen[0] in b._reconciled and seen[1] not in b._reconciled
    finally:
        b.close()


def test_closed_is_its_own_type(tmp_path):
    from myopenfund_mcp.filings.indexer import IndexClosed

    indexer = make(tmp_path)
    indexer.close()
    with pytest.raises(IndexClosed):
        indexer.status()


# --- the chunk title (ruling R40) --------------------------------------------------------------


def test_filing_title_states_the_period_and_the_filing_date():
    assert filing_title("MICROSOFT CORP", "10-K", dt.date(2024, 6, 30), dt.date(2024, 7, 30)) == TITLE


def test_filing_title_omits_the_period_when_the_report_date_is_missing():
    assert filing_title("AMAZON COM INC", "8-K", None, dt.date(2024, 2, 1)) == "AMAZON COM INC 8-K, filed 2024-02-01"


def test_filing_title_omits_the_filing_date_when_it_is_missing():
    assert filing_title("CIK 0000789019", "10-K", dt.date(2024, 6, 30), None) == (
        "CIK 0000789019 10-K for the period ended 2024-06-30"
    )
    assert filing_title("CIK 0000789019", "10-K", None, None) == "CIK 0000789019 10-K"


def test_enqueue_titles_chunks_with_the_period(tmp_path):
    indexer = make(tmp_path)
    doc_id = "sec-0000789019-24-000001"
    try:
        indexer.enqueue([ref()], names={CIK: "MICROSOFT CORP"})
        rows = indexer._query(
            "SELECT title FROM documents WHERE collection = ? AND doc_id = ?", (indexer.collection, doc_id)
        )
        assert rows == [(TITLE,)]
    finally:
        indexer.close()


def test_the_chunker_version_names_the_title_format():
    """The title is the chunk context (embedded and FTS-indexed): a new title format is a new
    collection, so a document indexed under the old titles is never served as ready."""
    assert chunker_version(chunk_options(Settings())) == "v2-t480-o32-ttl2"
