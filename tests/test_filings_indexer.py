"""The filings indexer: offline, with FakeEmbedder, a word tokenizer and a hand-written 10-K in Markdown."""

from __future__ import annotations

import datetime as dt
import json
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
    document_chunks,
    model_files_cached,
)

MARKDOWN = (Path(__file__).parent / "fixtures" / "filings" / "example_10k.md").read_text(encoding="utf-8")
CIK = "0000789019"
CONTACT = "someone@example.com"
SPEC = ModelSpec.from_settings(EngineSettings())


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
        self._count = threading.Lock()

    def encode_passage(self, texts):
        with self._count:
            self.passage_calls += 1
            number = self.passage_calls
        if self.allowed is not None and number > self.allowed:
            self.gate.wait(30)
        self.passages.extend(texts)
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
        title="MICROSOFT CORP 10-K 2024-07-30",
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
        assert payload["title"] == "MICROSOFT CORP 10-K 2024-07-30"
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


def test_read_section_is_bounded_and_starts_at_the_section(tmp_path):
    embedder = GatedEmbedder(batch_size=4)
    indexer = make(tmp_path, embedder, max_result_chars=1_000)
    doc_id = "sec-0000789019-24-000001"
    try:
        indexer.enqueue([ref()], names={CIK: "MICROSOFT CORP"})
        wait_until(lambda: state_of(indexer, doc_id).state == "ready")

        page = indexer.read(doc_id, section="Item 7")
        assert len(json.dumps(page)) <= 1_000
        assert page["chunks"][0]["text"].startswith("Revenue increased sixteen percent")
        assert {chunk["section"] for chunk in page["chunks"]} == {"Item 7"}
        assert page["more"] is True and page["next_chunk_id"] is not None

        following = indexer.read(doc_id, chunk_id=page["next_chunk_id"], around=0)
        assert following["chunks"][0]["chunk_id"] == page["next_chunk_id"]

        middle = int(page["chunks"][0]["chunk_id"]) + 1
        around = indexer.read(doc_id, chunk_id=str(middle), around=1)
        assert [c["chunk_id"] for c in around["chunks"]][:2] == [str(middle - 1), str(middle)]
        assert len(json.dumps(around)) <= 1_000

        with pytest.raises(SectionNotFound, match="Item 7"):
            indexer.read(doc_id, section="Item 99")
        with pytest.raises(DocumentNotFetched):
            indexer.read("sec-never-fetched")
        with pytest.raises(ValueError):
            indexer.read(doc_id, section="Item 7", chunk_id="1")
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
        assert failed.error.startswith("RuntimeError: SEC refused the request")
        assert "\n" not in failed.error
    finally:
        indexer.close()


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
        first = indexer.enqueue([ref()])
        assert first.model_download is True  # nothing in <cache>/models yet
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
