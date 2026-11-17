"""A small embedded corpus for the rag toolset, written through rag-engine's own store and schema.

Adversarial on purpose: a chunk on the first page (stored page 0), a document without pages, a
document without date and without title, a doc type held by a single chunk, a Japanese text, and
one point whose `source_code` is not a string. Texts are stored in NFKD, as the chunker stores them. The embedder is rag-engine's FakeEmbedder with no
prefix: a query equal to a chunk's text gets that chunk first, with a score of 1.
"""
from __future__ import annotations

import unicodedata

from qdrant_client import QdrantClient, models
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.schema import build_payload, page_for_id, point_id
from rag_engine.store import QdrantStore

COLLECTION = "rag-toolset-test"
DIM = 32
MODEL = "fake/embedder"
ECB_RATES = "The Governing Council decided to raise the three key ECB interest rates by 50 basis points."
ECB_TPI = "The Governing Council approved the Transmission Protection Instrument."
FOMC = "Participants judged that an increase of 75 basis points in the target range was appropriate."
# As typed (NFC). The chunker stores text in NFKD, and the engine puts the query in NFKD as well.
BOJ = "デフレ脱却に向けて、日本銀行は金融緩和を継続する。"
BOE = "The Committee voted by a majority of 7-2 to increase Bank Rate."

# (doc_id, metadata, [(stored page or None, text), ...])
DOCUMENTS = [
    ("ecb/A1/2022-07-21",
     {"corpus": "central-bank", "source_code": "ecb", "doc_type": "A1", "title": "Monetary policy decisions",
      "date": "2022-07-21", "year": 2022, "language": "en"},
     [(0, ECB_RATES), (1, ECB_TPI)]),
    ("us/A3/2022-06-15",
     {"corpus": "central-bank", "source_code": "us", "doc_type": "A3",
      "title": "Minutes of the Federal Open Market Committee", "date": "2022-06-15", "year": 2022,
      "language": "en"},
     [(4, FOMC)]),
    ("jp/E1/2023-01-18",
     {"corpus": "central-bank", "source_code": "jp", "doc_type": "E1",
      "title": "Outlook for Economic Activity and Prices", "date": "2023-01-18", "year": 2023, "language": "ja"},
     [(None, unicodedata.normalize("NFKD", BOJ))]),
    ("gb/B1/undated",
     {"corpus": "central-bank", "source_code": "gb", "doc_type": "B1", "language": "en"},
     [(2, BOE)]),
    ("odd/numeric-source",
     {"corpus": "central-bank", "source_code": 7, "doc_type": "A1", "title": "Odd", "date": "2021-01-01",
      "language": "en"},
     [(0, "A point whose source code is a number.")]),
]
SOURCE_CODES = ["ecb", "gb", "jp", "us"]
DOC_TYPES = ["A1", "A3", "B1", "E1"]
CHUNKS = sum(len(chunks) for _, _, chunks in DOCUMENTS)


def embedder() -> FakeEmbedder:
    return FakeEmbedder(ModelSpec(name=MODEL, dim=DIM, passage_prefix="", query_prefix=""))


def seed(store: QdrantStore, collection: str = COLLECTION) -> FakeEmbedder:
    """Create `collection` with its payload indexes and write the fixture; returns the embedder used."""
    fake = embedder()
    store.ensure_collection(collection, fake.dim)
    store.ensure_payload_indexes(collection)
    points = []
    for doc_id, metadata, chunks in DOCUMENTS:
        vectors = fake.encode_passage([text for _, text in chunks])
        for number, ((page, text), vector) in enumerate(zip(chunks, vectors, strict=True)):
            payload = build_payload(doc_id, metadata, filename=f"{doc_id}.pdf", chunk_number=number,
                                    ingestion_date="2026-11-17T00:00:00", text=text, page=page)
            points.append(models.PointStruct(id=point_id(doc_id, page_for_id(page), number),
                                             vector=vector.tolist(), payload=payload))
    store.upsert(collection, points)
    return fake


def local_store() -> QdrantStore:
    """A store on qdrant-client's in-process mode: no server."""
    return QdrantStore(client=QdrantClient(":memory:"))
