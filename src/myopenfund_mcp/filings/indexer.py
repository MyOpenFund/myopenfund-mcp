"""Background indexer of SEC filing documents over rag-engine's local index (no-vault mode).

A document goes through: `queued` (enqueued) -> `fetching` (its primary document is downloaded,
through the SEC client and its disk cache, and converted to Markdown by company-corpus) ->
`indexing` (chunked, embedded, added) -> `ready`; or `failed` (with a one-line error), or `partial`
(see "Restart"). `enqueue` returns at once; daemon worker threads (`MOF_INDEX_WORKERS`) do the work.

**Where things live** (never evicted by the HTTP cache):
- `<cache_path()>/models/`: the embedding model files (a Hugging Face cache layout; 1.1 GB once).
  Software, not data: they stay in the persistent cache folder even with `MOF_CACHE=off`;
- `<root>/index/<collection>/`: rag-engine's `LocalIndex` (Qdrant local mode + SQLite FTS5). The
  collection name (`rag_engine.local.collection_name`) holds the model, its pinned revision, the
  chunker version (`v2-t<MOF_CHUNK_MAX_TOKENS>-o<MOF_CHUNK_OVERLAP_TOKENS>`) and the ONNX provider,
  so vectors of two setups are never mixed;
- `<root>/index/status.sqlite3`: this module's record, keyed by collection: one row per document
  (state, chunks done/total, error, metadata, title, fingerprint of its cut) and one row per stored
  chunk (number, section, text), which `read` serves and a resume reads.
`<root>` is `cache_path()`, or with `MOF_CACHE=off` a temporary folder removed at exit (after the
indexer is closed, so no worker writes into a removed folder).

**Chunks.** The Markdown is cut at the boundaries of its Item sections (`FilingDocument.sections`;
the text before, between and after them has no section), and each piece is chunked with
`rag_engine.chunking.chunk_markdown` under the v2 profile, whose token cap and overlap come from
`MOF_CHUNK_MAX_TOKENS` / `MOF_CHUNK_OVERLAP_TOKENS` (the context cap of v2, 120 tokens, is lowered to
half the token cap when the cap is below 240, the most rag-engine allows). So every chunk lies in one
section, and its `section` payload is the name of the section it starts in, exactly. Chunk numbers
run in document order; they are the `first_index` positions given to `LocalIndex.add`, so the point
ids do not depend on the order chunks are embedded in. Payload of each chunk: `doc_id`, `doc_type`
(the form), `date` (filed, `YYYY-MM-DD`), `title` (`<company> <form> <filed>`), `source_code` (the
CIK), `section`, `chunker_version`, `accession`, `url`. A document without any chunk is `failed`.

**Order.** The chunks of the form's priority sections (`company_corpus.live.documents.
priority_sections`: Item 7 of a 10-K) are embedded and added first, then the others in document
order, in batches of the embedder's batch size, each batch within one section. A search serves what
is added so far: the MD&A of a document is searchable before the document is `ready`.

**Model.** Loaded in a worker when the first document is taken, never at start-up; a `search` of
documents indexed by an earlier process, made before any `enqueue`, loads it (and waits for a load in
progress). Once loaded, searches never wait on it. The first `enqueue` result of the process has
`model_download=True` when the pinned model files are not yet in `models/` (reported once).

**Restart.** A document left `queued`, `fetching` or `indexing` by a process that stopped is shown as
`partial` (never `ready`) by the next process that opens the index; so is a `ready` one whose chunk
count in the local index is not its total. Enqueuing it again resumes it: the document is chunked
again with the title stored at its first indexing (the Markdown comes from the immutable disk cache)
and the cut is compared with the stored fingerprint (SHA-256 of the chunker version, the title and
every chunk's section, text and context). Same cut, and the local index holds exactly the stored
chunks: the stored chunk numbers are skipped. Otherwise (another cut, or chunks added to the index
whose record was not written) the document's points and chunks are removed and it is indexed again.

**One process per cache folder.** Qdrant local mode locks its folder. The index is opened by the
first call that needs it; when another process holds it, every method raises `IndexUnavailable`
(naming the lock), and a later call tries again. Nothing else of the server depends on it.

**Errors.** A document that fails is marked `failed` with `<ExceptionType>: <first line of the
message>` (the SEC contact is never written) and the worker continues with the next one. Log lines
carry exception type names only.

`close()` stops the workers after their current step (the model load, the fetch, the chunking or a
batch), waiting at most `MOF_INDEX_CLOSE_TIMEOUT_S`, then releases the index; a worker still busy
then (a model download) finds the index closed and ends with the process. A document cut that way
is `partial` for the next process. Calls after `close()` raise `IndexUnavailable`.
"""

from __future__ import annotations

import atexit
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import queue
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from company_corpus.live.documents import FilingDocument, filing_document, priority_sections
from company_corpus.live.filings import FilingRef
from company_corpus.live.sec import SecGet
from rag_engine.chunking import Chunk, ChunkingOptions, chunk_markdown, load_tokenizer
from rag_engine.config import Settings as EngineSettings
from rag_engine.embedding import ONNX_BATCH_SIZE, ONNX_MODELS, Embedder, ModelSpec, OnnxEmbedder
from rag_engine.local import LocalIndex, collection_name, hybrid_search
from rag_engine.search import Hit, SearchFilters
from rag_engine.select import select_diverse
from rag_engine.store import IndexLocked

from ..config import Settings
from ..paging import result_chars

__all__ = [
    "STATES",
    "DocStatus",
    "DocumentNotFetched",
    "EnqueueResult",
    "FilingsIndexer",
    "IndexUnavailable",
    "SearchResult",
    "SectionNotFound",
    "StatusResult",
    "chunk_id",
    "chunk_options",
    "chunker_version",
    "document_chunks",
    "model_files_cached",
]

log = logging.getLogger("myopenfund_mcp.filings")

State = Literal["queued", "fetching", "indexing", "partial", "ready", "failed"]
STATES: tuple[str, ...] = ("queued", "fetching", "indexing", "partial", "ready", "failed")
_IN_FLIGHT = ("queued", "fetching", "indexing")

#: The v2 profile's own context cap (`ChunkingOptions.v2().context_max_tokens`), lowered to half the
#: token cap when MOF_CHUNK_MAX_TOKENS is below twice it (rag-engine refuses more than half).
_V2 = ChunkingOptions.v2()


class IndexUnavailable(RuntimeError):
    """The local filings index cannot be used: another process holds it, or the indexer is closed."""


class DocumentNotFetched(LookupError):
    """`read` of a document that was never enqueued."""


class SectionNotFound(LookupError):
    """`read(section=...)` of a section the document's stored chunks do not have."""


class _Stopped(Exception):
    """The indexer is closing: the worker leaves the document as it is (partial for the next process)."""


@dataclass
class DocStatus:
    doc_id: str
    cik: str
    form: str
    accession: str
    filed: dt.date | None
    url: str
    state: State
    chunks_done: int
    chunks_total: int | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["filed"] = self.filed.isoformat() if self.filed else None
        return out


class EnqueueResult(list):
    """The statuses `enqueue` returns (a `list[DocStatus]`), plus `model_download`: True on the first
    `enqueue` of the process that starts the first load of the embedding model while its pinned files
    are not in the cache yet (1.1 GB to download); False on every later call."""

    model_download: bool = False


class StatusResult(list):
    """The statuses `status` returns (a `list[DocStatus]`), plus `unknown`: the requested doc_ids that
    were never enqueued, in the order given."""

    unknown: list[str]

    def __init__(self, items: Iterable[DocStatus] = (), unknown: Sequence[str] = ()) -> None:
        super().__init__(items)
        self.unknown = list(unknown)


class SearchResult(tuple):
    """`(hits, not_ready)` as `search` returns it, plus `unknown`: the requested doc_ids that were
    never enqueued, in the order given."""

    unknown: list[str]

    def __new__(cls, hits: list[Hit], not_ready: list[DocStatus], unknown: Sequence[str] = ()) -> Self:
        result = super().__new__(cls, (hits, not_ready))
        result.unknown = list(unknown)
        return result

    @property
    def hits(self) -> list[Hit]:
        return self[0]

    @property
    def not_ready(self) -> list[DocStatus]:
        return self[1]


# --- pure helpers ---------------------------------------------------------------------------------


def chunk_options(settings: Settings) -> ChunkingOptions:
    """The v2 profile with MOF_CHUNK_MAX_TOKENS and MOF_CHUNK_OVERLAP_TOKENS."""
    cap = settings.chunk_max_tokens
    context = min(_V2.context_max_tokens or cap // 2, cap // 2)
    return dataclasses.replace(
        _V2,
        max_chunk_token_count=cap,
        overlap_tokens=settings.chunk_overlap_tokens,
        context_max_tokens=context,
    )


def chunker_version(options: ChunkingOptions) -> str:
    """The label of a cut: the profile and the two sizes the settings change (`v2-t480-o32`)."""
    return f"{options.version}-t{options.max_chunk_token_count}-o{options.overlap_tokens}"


def document_chunks(
    document: FilingDocument, *, options: ChunkingOptions, tokenizer: Any, title: str | None
) -> list[tuple[str | None, Chunk]]:
    """The chunks of `document` in document order, each with the name of its section (or None).

    The Markdown is cut at section boundaries (the text outside every section is cut at them too)
    and each piece is chunked on its own, so a chunk never spans two sections."""
    markdown = document.markdown
    pieces: list[tuple[str | None, int, int]] = []
    cursor = 0
    for section in sorted(document.sections, key=lambda s: s.start):
        start, end = max(section.start, cursor), min(section.end, len(markdown))
        if start >= end:
            continue
        if start > cursor:
            pieces.append((None, cursor, start))
        pieces.append((section.name, start, end))
        cursor = end
    if cursor < len(markdown):
        pieces.append((None, cursor, len(markdown)))
    out: list[tuple[str | None, Chunk]] = []
    for name, start, end in pieces:
        text = markdown[start:end]
        if not text.strip():
            continue
        out.extend(
            (name, chunk) for chunk in chunk_markdown(text, options=options, title=title, tokenizer=tokenizer)
        )
    return out


def model_files_cached(models_dir: str | Path, model_name: str) -> bool:
    """Whether every pinned file of `model_name` (rag-engine's `ONNX_MODELS`) is in `models_dir`
    at its pinned revision (Hugging Face cache layout). False for a model that is not pinned."""
    from huggingface_hub import try_to_load_from_cache

    known = ONNX_MODELS.get(model_name)
    if known is None:
        return False
    for name in known.files:
        found = try_to_load_from_cache(
            known.hf or model_name, name, cache_dir=str(models_dir), revision=known.revision
        )
        if not isinstance(found, str):
            return False
    return True


def chunk_id(hit: Hit) -> str:
    """The id `read(chunk_id=...)` takes for a hit: its chunk number within its document."""
    return str(hit.chunk_number if hit.chunk_number is not None else hit.payload.get("chunk_number"))


def _batches(
    sections: Sequence[str | None], done: Collection[int], priority: Collection[str], size: int
) -> list[list[int]]:
    """Chunk numbers to embed, in batches of at most `size` consecutive numbers of one section:
    the priority sections first, then the others, each in document order; `done` left out."""
    runs: list[list[int]] = []
    for number, section in enumerate(sections):
        if number in done:
            continue
        run = runs[-1] if runs else None
        if run and run[-1] == number - 1 and sections[run[-1]] == section and len(run) < size:
            run.append(number)
        else:
            runs.append([number])
    first = [run for run in runs if sections[run[0]] in priority]
    return first + [run for run in runs if sections[run[0]] not in priority]


def _short_error(exc: BaseException, contact: str | None) -> str:
    lines = str(exc).strip().splitlines()
    text = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    if contact:
        text = text.replace(contact, "<contact>")
    return text


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _default_doc_id(filing: FilingRef) -> str:
    return filing.doc_id or f"sec-{filing.cik}-{filing.accession}"


def _fingerprint(version: str, title: str, pieces: Sequence[tuple[str | None, Chunk]]) -> str:
    """SHA-256 of a document's cut: chunker version, title, and every chunk's section, text, context."""
    digest = hashlib.sha256()
    for part in (version, title):
        digest.update(json.dumps(part).encode() + b"\n")
    for section, chunk in pieces:
        digest.update(json.dumps([section, chunk.text, chunk.context], ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


_SCHEMA = (
    (
        "CREATE TABLE IF NOT EXISTS documents ("
        " collection TEXT NOT NULL, doc_id TEXT NOT NULL, cik TEXT NOT NULL, form TEXT NOT NULL,"
        " accession TEXT NOT NULL, filed TEXT, report_date TEXT, url TEXT NOT NULL, title TEXT NOT NULL,"
        " state TEXT NOT NULL, chunks_done INTEGER NOT NULL DEFAULT 0, chunks_total INTEGER, error TEXT,"
        " updated_at TEXT NOT NULL, fingerprint TEXT, PRIMARY KEY (collection, doc_id))"
    ),
    (
        "CREATE TABLE IF NOT EXISTS chunks ("
        " collection TEXT NOT NULL, doc_id TEXT NOT NULL, chunk_number INTEGER NOT NULL, section TEXT,"
        " text TEXT NOT NULL, context TEXT, PRIMARY KEY (collection, doc_id, chunk_number))"
    ),
)
_COLUMNS = "doc_id, cik, form, accession, filed, url, state, chunks_done, chunks_total, error, report_date"


def _status(row: Sequence[Any]) -> DocStatus:
    filed = dt.date.fromisoformat(row[4]) if row[4] else None
    return DocStatus(
        doc_id=row[0],
        cik=row[1],
        form=row[2],
        accession=row[3],
        filed=filed,
        url=row[5],
        state=row[6],
        chunks_done=int(row[7]),
        chunks_total=row[8],
        error=row[9],
    )


# --- the indexer ----------------------------------------------------------------------------------


class FilingsIndexer:
    """See the module docstring. `embedder_factory`, `tokenizer_factory` and `document_loader` are
    for tests (FakeEmbedder, a word tokenizer, recorded Markdown); a deployment builds the pinned
    ONNX model, its tokenizer and company-corpus's `filing_document` from the settings."""

    def __init__(
        self,
        settings: Settings,
        sec: SecGet,
        *,
        embedder_factory: Callable[[], Embedder] | None = None,
        tokenizer_factory: Callable[[], Any] | None = None,
        document_loader: Callable[[FilingRef, SecGet], FilingDocument] | None = None,
    ) -> None:
        self._settings = settings
        self._sec = sec
        self._spec = ModelSpec.from_settings(EngineSettings())
        known = ONNX_MODELS.get(self._spec.name)
        self.options = chunk_options(settings)
        self.chunker_version = chunker_version(self.options)
        self.collection = collection_name(
            self._spec.name,
            known.revision if known else "unpinned",
            self.chunker_version,
            settings.rag_onnx_provider,
        )
        self._embedder_factory = embedder_factory
        self._tokenizer_factory = tokenizer_factory
        self._loader = document_loader or filing_document
        self._root: Path | None = None
        self._lock = threading.RLock()  # index opening, the status DB, the in-flight set
        self._model_lock = threading.Lock()
        self._embedder: Embedder | None = None
        self._tokenizer: Any = None
        self._download_reported = False
        self._reconciled: set[str] = set()  # documents checked against the index in this process
        self._index: LocalIndex | None = None
        self._db: sqlite3.Connection | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._refs: dict[str, FilingRef] = {}
        self._active: set[str] = set()
        self._stop = threading.Event()
        self._closed = False

    # --- paths ---------------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The folder holding `index/` (a temporary one with MOF_CACHE=off)."""
        with self._lock:
            if self._root is None:
                if self._settings.cache == "off":
                    tmp = tempfile.TemporaryDirectory(prefix="myopenfund-mcp-filings-")
                    atexit.register(tmp.cleanup)
                    atexit.register(self.close)  # atexit runs last-registered first: close, then remove
                    self._root = Path(tmp.name)
                else:
                    self._root = Path(self._settings.cache_path())
            return self._root

    @property
    def models_dir(self) -> Path:
        """`<cache_path()>/models`, whatever MOF_CACHE says (R36: the model is software, not data)."""
        return Path(self._settings.cache_path()) / "models"

    @property
    def index_dir(self) -> Path:
        return self.root / "index" / self.collection

    # --- opening -------------------------------------------------------------------------------

    def _open(self) -> None:
        with self._lock:
            if self._closed:
                raise IndexUnavailable("the filings index is closed")
            if self._index is not None:
                return
            # Private to the user, as the HTTP cache creates it (when the index comes first).
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                index = LocalIndex(self.index_dir, collection=self.collection, dim=self._spec.dim)
            except IndexLocked as exc:
                raise IndexUnavailable(
                    f"the local filings index {str(self.index_dir)!r} is locked: it is open in another "
                    "process (another MCP server or `warm` on the same MOF_CACHE_DIR); stop it, or give "
                    "this one another MOF_CACHE_DIR"
                ) from exc
            db = None
            try:
                db = sqlite3.connect(
                    self.root / "index" / "status.sqlite3", check_same_thread=False, isolation_level=None
                )
                db.execute("PRAGMA journal_mode=WAL")
                for statement in _SCHEMA:
                    db.execute(statement)
                columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
                if "fingerprint" not in columns:  # a status file of an earlier build
                    db.execute("ALTER TABLE documents ADD COLUMN fingerprint TEXT")
                # We hold the index lock: no other process is indexing. What was in flight is partial.
                # (A ready document whose chunks are not all in the index is found lazily: _reconcile.)
                db.execute(
                    f"UPDATE documents SET state = 'partial', updated_at = ? WHERE collection = ? "
                    f"AND state IN ({', '.join('?' * len(_IN_FLIGHT))})",
                    (_now(), self.collection, *_IN_FLIGHT),
                )
            except Exception:
                if db is not None:
                    db.close()
                index.close()
                raise
            self._index, self._db = index, db

    def _start_workers(self) -> None:
        with self._lock:
            if self._workers:
                return
            for number in range(self._settings.index_workers):
                worker = threading.Thread(target=self._work, name=f"filings-indexer-{number}", daemon=True)
                worker.start()
                self._workers.append(worker)

    # --- the model -----------------------------------------------------------------------------

    def _model(self) -> tuple[Embedder, Any]:
        """The embedder and the tokenizer, loaded on first use (one load for every thread). Once
        loaded, no lock is taken: a search never waits for a worker here."""
        embedder = self._embedder
        if embedder is not None:
            return embedder, self._tokenizer
        with self._model_lock:
            if self._embedder is None:
                models = str(self.models_dir)
                # The persistent cache folder, private as the HTTP cache wants it (also with MOF_CACHE=off,
                # so that a later MOF_CACHE=on start does not refuse a folder made with the umask).
                Path(self._settings.cache_path()).mkdir(mode=0o700, parents=True, exist_ok=True)
                if self._embedder_factory is not None:
                    embedder = self._embedder_factory()
                else:
                    embedder = OnnxEmbedder(
                        self._spec,
                        cache_dir=models,
                        threads=self._settings.rag_onnx_threads,
                        provider=self._settings.rag_onnx_provider,
                    )
                if self._tokenizer_factory is not None:
                    tokenizer = self._tokenizer_factory()
                else:
                    tokenizer = load_tokenizer(models, self._spec.name)
                if embedder.dim != self._spec.dim:
                    raise ValueError(
                        f"the embedder gives {embedder.dim}-dimensional vectors, the index expects {self._spec.dim}"
                    )
                self._tokenizer = tokenizer  # before the embedder: the fast path reads the embedder first
                self._embedder = embedder
            return self._embedder, self._tokenizer

    def _model_download_needed(self) -> bool:
        """Whether the model is not loaded, not yet reported, and its files are not in `models/` (the
        default factory only). `enqueue` reports it once, on the first call that queues a document."""
        with self._lock:
            if self._download_reported or self._embedder is not None or self._embedder_factory is not None:
                return False
        return not model_files_cached(self.models_dir, self._spec.name)

    def _reconcile(self, doc_ids: Iterable[str]) -> None:
        """A `ready` document whose chunk count in the local index is not its total becomes `partial`.
        Checked once per document and process, when it is first touched (never at open: a count
        scans the FTS table)."""
        with self._lock:
            pending = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id not in self._reconciled]
            index = self._index
        if not pending or index is None:
            return
        for row in self._rows(pending):
            status = _status(row)
            if status.state == "ready" and index.count(status.doc_id) != status.chunks_total:
                with self._lock:
                    if status.doc_id not in self._active:
                        self._update(status.doc_id, state="partial")
        with self._lock:
            self._reconciled.update(pending)

    # --- the status database -------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """The status database; call with `self._lock` held. Closed -> IndexUnavailable."""
        if self._db is None:
            raise IndexUnavailable("the filings index is closed")
        return self._db

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
        with self._lock:
            return self._conn().execute(sql, params).fetchall()

    def _rows(self, doc_ids: Collection[str] | None) -> list[tuple]:
        if doc_ids is None:
            return self._query(
                f"SELECT {_COLUMNS} FROM documents WHERE collection = ? ORDER BY filed DESC, doc_id",
                (self.collection,),
            )
        return self._query(
            f"SELECT {_COLUMNS} FROM documents WHERE collection = ? AND doc_id IN "
            "(SELECT value FROM json_each(?)) ORDER BY filed DESC, doc_id",
            (self.collection, json.dumps(list(doc_ids))),
        )

    def _get(self, doc_id: str) -> DocStatus | None:
        rows = self._rows([doc_id])
        return _status(rows[0]) if rows else None

    def _update(self, doc_id: str, **values: Any) -> None:
        values["updated_at"] = _now()
        assignments = ", ".join(f"{name} = ?" for name in values)
        self._query(
            f"UPDATE documents SET {assignments} WHERE collection = ? AND doc_id = ?",
            (*values.values(), self.collection, doc_id),
        )

    def _done(self, doc_id: str) -> set[int]:
        rows = self._query(
            "SELECT chunk_number FROM chunks WHERE collection = ? AND doc_id = ?", (self.collection, doc_id)
        )
        return {int(row[0]) for row in rows}

    def _record_batch(
        self,
        doc_id: str,
        batch: Sequence[int],
        sections: Sequence[str | None],
        chunks: Sequence[Chunk],
        done: int,
    ) -> None:
        """The chunks of a batch just added to the index, and the new count, in one transaction."""
        with self._lock:
            db = self._conn()
            with db:  # rolled back if a write fails
                db.execute("BEGIN")
                db.executemany(
                    "INSERT OR REPLACE INTO chunks (collection, doc_id, chunk_number, section, text, context)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (self.collection, doc_id, number, sections[number], chunk.text, chunk.context)
                        for number, chunk in zip(batch, chunks)
                    ],
                )
                db.execute(
                    "UPDATE documents SET chunks_done = ?, updated_at = ? WHERE collection = ? AND doc_id = ?",
                    (done, _now(), self.collection, doc_id),
                )

    def _unknown(self, doc_ids: Collection[str] | None, rows: Sequence[tuple]) -> list[str]:
        if doc_ids is None:
            return []
        known = {row[0] for row in rows}
        return [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id not in known]

    # --- public API ----------------------------------------------------------------------------

    def enqueue(self, refs: Sequence[FilingRef], *, names: Mapping[str, str] | None = None) -> EnqueueResult:
        """Queue `refs` for indexing and return their statuses at once. A `ready` document stays
        ready; one already queued or being indexed by this process is left as it is; a `partial` or
        `failed` one is queued again (a partial one resumes). `names` maps a CIK to the company name
        used in the chunks' `title` (default `CIK <cik>`); a document with stored chunks keeps the
        title it was first indexed with."""
        self._open()
        result = EnqueueResult()
        download = self._model_download_needed()
        self._start_workers()
        names = names or {}
        self._reconcile(_default_doc_id(filing) for filing in refs)
        queued = False
        for filing in refs:
            doc_id = _default_doc_id(filing)
            filed = filing.filed.isoformat() if filing.filed else None
            title = f"{names.get(filing.cik) or f'CIK {filing.cik}'} {filing.form} {filed or 'undated'}"
            with self._lock:
                current = self._get(doc_id)
                if current is not None and (current.state == "ready" or doc_id in self._active):
                    result.append(current)
                    continue
                self._conn().execute(
                    "INSERT INTO documents (collection, doc_id, cik, form, accession, filed, report_date, url,"
                    " title, state, chunks_done, chunks_total, error, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, NULL, NULL, ?)"
                    " ON CONFLICT (collection, doc_id) DO UPDATE SET cik = excluded.cik, form = excluded.form,"
                    " accession = excluded.accession, filed = excluded.filed,"
                    " report_date = excluded.report_date, url = excluded.url,"
                    " title = CASE WHEN documents.chunks_done > 0 THEN documents.title ELSE excluded.title END,"
                    " state = 'queued', error = NULL, updated_at = excluded.updated_at",
                    (
                        self.collection,
                        doc_id,
                        filing.cik,
                        filing.form,
                        filing.accession,
                        filed,
                        filing.report_date.isoformat() if filing.report_date else None,
                        filing.url,
                        title,
                        _now(),
                    ),
                )
                self._refs[doc_id] = filing
                self._active.add(doc_id)
                self._queue.put(doc_id)
                queued = True
                status = self._get(doc_id)
                assert status is not None
                result.append(status)
        if queued and download:
            with self._lock:
                result.model_download = not self._download_reported
                self._download_reported = True
        return result

    def status(self, doc_ids: Collection[str] | None = None) -> StatusResult:
        """The statuses of `doc_ids` (every known document when None), newest filing first; the
        requested documents never enqueued are listed in `.unknown`."""
        self._open()
        rows = self._rows(doc_ids)
        return StatusResult((_status(row) for row in rows), self._unknown(doc_ids, rows))

    def search(
        self,
        query: str,
        *,
        doc_ids: Collection[str] | None = None,
        forms: Collection[str] | None = None,
        period_from: dt.date | None = None,
        period_to: dt.date | None = None,
        limit: int = 10,
    ) -> SearchResult:
        """Hybrid search (dense + BM25, RRF; eigenmind selection when MOF_SEARCH_EIGENMIND) over the
        chunks indexed so far of the requested documents: `(hits, not_ready)`, the statuses of the
        requested documents that are not `ready`, and `.unknown`, those never enqueued.

        Documents: `doc_ids` (every known document when None), restricted to `forms` (a form
        includes its `/A` amendments) and to a report period overlapping `period_from`..`period_to`
        (the report date, or the filing date when the filing has none), as `companies_filings`."""
        self._open()
        wanted = None if forms is None else {f.upper() for f in forms} | {f"{f.upper()}/A" for f in forms}
        all_rows = self._rows(doc_ids)
        if any(row[6] == "ready" and row[0] not in self._reconciled for row in all_rows):
            self._reconcile(row[0] for row in all_rows if row[6] == "ready")
            all_rows = self._rows(doc_ids)
        rows = []
        for row in all_rows:
            period = row[10] or row[4]
            day = dt.date.fromisoformat(period) if period else None
            if wanted is not None and row[2].upper() not in wanted:
                continue
            if (period_from or period_to) and day is None:
                continue
            if period_from and day < period_from or period_to and day > period_to:
                continue
            rows.append(_status(row))
        unknown = self._unknown(doc_ids, all_rows)
        not_ready = [status for status in rows if status.state != "ready"]
        searchable = [status.doc_id for status in rows if status.chunks_done > 0]
        if not searchable:
            return SearchResult([], not_ready, unknown)
        embedder, _ = self._model()
        with self._lock:
            index = self._index
            if index is None or self._closed:
                raise IndexUnavailable("the filings index is closed")
        try:
            hits = hybrid_search(
                query,
                index=index,
                embedder=embedder,
                filters=SearchFilters(doc_ids=searchable),
                limit=limit,
                candidates=self._settings.search_candidates,
                rrf_k=self._settings.search_rrf_k,
                selector=select_diverse if self._settings.search_eigenmind else None,
            )
        except Exception:
            if self._closed:  # the index was closed under the search
                raise IndexUnavailable("the filings index is closed") from None
            raise
        return SearchResult(hits, not_ready, unknown)

    def read(
        self,
        doc_id: str,
        *,
        section: str | None = None,
        chunk_id: str | int | None = None,
        around: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """The stored chunks of `doc_id`, in order, as a result whose JSON (counted as the tools count
        it) stays within MOF_MCP_MAX_RESULT_CHARS. What is read:

        - `section`: that section, from its start (or from `chunk_id` when given) to its end
          (`around` is ignored);
        - `chunk_id` alone (`around` None, the default): from it to the end of the document;
        - `chunk_id` with `around` an integer: a window of `around` stored chunks on each side of it;
        - neither: the document from its first chunk.
        `offset` (with `chunk_id`, outside the window mode) skips that many characters of the first
        chunk's text (the remainder of a chunk cut by the budget). The result's `more` says whether
        the read stopped before the end of what was asked; `next_chunk_id` / `next_offset` then give
        where to go on, always with the same call shape: `read(doc_id, section=<same>,
        chunk_id=next_chunk_id, offset=next_offset)`. A window cut by the budget continues that way
        too, so the continuation reads on to the end of the document, not of the window. A chunk
        larger than what is left of the budget when it comes first is cut (`text_cut: true`) and its
        remainder comes next."""
        if type(offset) is not int or offset < 0:
            raise ValueError(f"offset must be an integer >= 0, got {offset!r}")
        if around is not None and (type(around) is not int or around < 0):
            raise ValueError(f"around must be None or an integer >= 0, got {around!r}")
        number: int | None = None
        if chunk_id is not None:
            try:
                number = int(chunk_id)
            except (TypeError, ValueError):
                raise ValueError(f"chunk_id must be a chunk number, got {chunk_id!r}") from None
        window = number is not None and section is None and around is not None
        if offset and (number is None or window):
            raise ValueError("offset needs chunk_id and a forward read (section, or around=None)")
        self._open()
        self._reconcile([doc_id])
        status = self._get(doc_id)
        if status is None:
            raise DocumentNotFetched(f"document {doc_id!r} was never fetched")
        select = "SELECT chunk_number, section, text FROM chunks WHERE collection = ? AND doc_id = ?"
        key = (self.collection, doc_id)
        result: dict[str, Any] = {
            **status.to_dict(),
            "section": section,
            "chunks": [],
            "more": False,
            "next_chunk_id": None,
            "next_offset": 0,
        }
        with self._lock:
            db = self._conn()
            if section is not None:
                found = db.execute(select + " AND section = ? LIMIT 1", (*key, section)).fetchall()
                if not found:
                    available = [
                        row[0]
                        for row in db.execute(
                            "SELECT section FROM chunks WHERE collection = ? AND doc_id = ? AND section IS NOT NULL"
                            " GROUP BY section ORDER BY min(chunk_number)",
                            key,
                        )
                    ]
                    raise SectionNotFound(
                        f"document {doc_id!r} has no stored chunk in section {section!r}; sections: "
                        f"{', '.join(available) or 'none yet'}"
                    )
            if number is not None:
                anchor = db.execute(select + " AND chunk_number = ?", (*key, number)).fetchall()
                if not anchor or (section is not None and anchor[0][1] != section):
                    raise LookupError(
                        f"document {doc_id!r} has no stored chunk {number}"
                        + (f" in section {section!r}" if section is not None else "")
                    )
                if offset >= len(anchor[0][2]):
                    raise ValueError(f"offset {offset} is past the end of chunk {number}")
            if window:
                assert number is not None and around is not None
                before = db.execute(
                    select + " AND chunk_number < ? ORDER BY chunk_number DESC LIMIT ?",
                    (*key, number, around),
                ).fetchall()
                after = db.execute(
                    select + " AND chunk_number >= ? ORDER BY chunk_number LIMIT ?",
                    (*key, number, around + 1),
                ).fetchall()
                rows: Iterable[tuple] = [*reversed(before), *after]
            elif section is not None:
                rows = db.execute(
                    select + " AND section = ? AND chunk_number >= ? ORDER BY chunk_number",
                    (*key, section, -1 if number is None else number),
                )
            else:
                rows = db.execute(
                    select + " AND chunk_number >= ? ORDER BY chunk_number",
                    (*key, -1 if number is None else number),
                )
            self._fill(result, rows, offset)
        return result

    def _fill(self, result: dict[str, Any], rows: Iterable[tuple], offset: int) -> None:
        """Append chunks of `rows` to `result` while its size (as the tools count it), `more` and
        `next_*` included, stays within the budget; `next_*` point at the first text not returned."""
        budget = self._settings.max_result_chars
        chunks: list[dict[str, Any]] = result["chunks"]
        size = result_chars(result)
        pending: dict[str, Any] | None = None  # the first entry that did not fit
        first = True
        for number, name, text in rows:
            skip, first = (offset if first else 0), False
            entry: dict[str, Any] = {"chunk_id": str(number), "section": name, "text": text[skip:]}
            if skip:
                entry["offset"] = skip
            cost = result_chars(entry) + (1 if chunks else 0)  # the comma between list items
            if size + cost <= budget:
                chunks.append(entry)
                size += cost
                continue
            pending = entry
            break
        if pending is None:
            return
        self._point_at(result, pending)
        # `more` and `next_*` changed the size: give back trailing chunks while more than one is left.
        while result_chars(result) > budget and len(chunks) > 1:
            self._point_at(result, chunks.pop())
        if not chunks or result_chars(result) > budget:
            # The first chunk does not fit whole: its text is cut, the remainder comes next.
            entry = chunks.pop() if chunks else pending
            whole, skip = entry["text"], entry.get("offset", 0)

            def fits(length: int) -> bool:
                piece = {**entry, "text": whole[:length], "text_cut": True}
                trial = {**result, "chunks": [piece], "more": True}
                trial.update(next_chunk_id=entry["chunk_id"], next_offset=skip + length)
                return result_chars(trial) <= budget

            low, high = 0, len(whole) - 1  # a remainder of at least one character comes next
            while low < high:
                middle = (low + high + 1) // 2
                low, high = (middle, high) if fits(middle) else (low, middle - 1)
            if low == 0:
                raise ValueError(f"MOF_MCP_MAX_RESULT_CHARS ({budget}) leaves no room for any chunk text")
            chunks.append({**entry, "text": whole[:low], "text_cut": True})
            result.update(more=True, next_chunk_id=entry["chunk_id"], next_offset=skip + low)

    @staticmethod
    def _point_at(result: dict[str, Any], entry: dict[str, Any]) -> None:
        result.update(more=True, next_chunk_id=entry["chunk_id"], next_offset=entry.get("offset", 0))

    def close(self) -> None:
        """Stop the workers after their current step (at most MOF_INDEX_CLOSE_TIMEOUT_S) and release
        the index. Twice is harmless."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            workers = list(self._workers)
        for _ in workers:
            self._queue.put(None)
        deadline = time.monotonic() + self._settings.index_close_timeout_s
        for worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        busy = sum(worker.is_alive() for worker in workers)
        if busy:
            log.warning(
                "%d filings indexing worker(s) still busy after MOF_INDEX_CLOSE_TIMEOUT_S: the index is "
                "released, their documents are resumed next time",
                busy,
            )
        with self._lock:
            index, db = self._index, self._db
            self._index, self._db = None, None
            if index is not None:
                index.close()
            if db is not None:
                db.close()
        if self._embedder is not None and not busy:
            self._embedder.release()

    # --- the worker ----------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            doc_id = self._queue.get()
            if doc_id is None or self._stop.is_set():
                return
            final: tuple[str, str | None] | None = None
            try:
                self._index_one(doc_id)
                final = ("ready", None)
            except _Stopped:
                pass  # left as it is: partial for the next process
            except Exception as exc:  # noqa: BLE001 - one document fails, the worker goes on
                # Type names only: a message, or its chained context, may hold the SEC contact.
                log.warning("filing %s failed: %s", doc_id, type(exc).__name__)
                final = ("failed", _short_error(exc, self._settings.sec_contact))
            # The final state and leaving the in-flight set together: enqueue sees both or neither.
            with self._lock:
                try:
                    if final is not None:
                        self._update(doc_id, state=final[0], error=final[1])
                except Exception as exc:  # noqa: BLE001 - the index may be closing
                    log.warning("could not record the state of filing %s: %s", doc_id, type(exc).__name__)
                finally:
                    self._active.discard(doc_id)

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise _Stopped

    def _index_one(self, doc_id: str) -> None:
        filing = self._refs[doc_id]
        self._update(doc_id, state="fetching", error=None)
        self._check_stop()
        embedder, tokenizer = self._model()
        self._check_stop()
        document = self._loader(filing, self._sec)
        self._check_stop()
        stored = self._query(
            "SELECT title, fingerprint FROM documents WHERE collection = ? AND doc_id = ?",
            (self.collection, doc_id),
        )
        title, fingerprint = stored[0]
        pieces = document_chunks(document, options=self.options, tokenizer=tokenizer, title=title)
        if not pieces:
            raise ValueError("no text to index in the primary document")
        self._check_stop()
        cut = _fingerprint(self.chunker_version, title, pieces)
        with self._lock:
            index = self._index
            if index is None:
                raise IndexUnavailable("the filings index is closed")
        done = self._done(doc_id)
        if fingerprint != cut and (done or fingerprint is not None) or index.count(doc_id) != len(done):
            # Another cut than the stored one, or chunks in the index without their record: again.
            index.delete(doc_id)
            self._query("DELETE FROM chunks WHERE collection = ? AND doc_id = ?", (self.collection, doc_id))
            done = set()
        self._update(
            doc_id, state="indexing", chunks_total=len(pieces), chunks_done=len(done), fingerprint=cut
        )
        sections = [name for name, _ in pieces]
        size = getattr(embedder, "batch_size", ONNX_BATCH_SIZE)
        base = {
            "doc_type": filing.form,
            "date": filing.filed.isoformat() if filing.filed else None,
            "title": title,
            "source_code": filing.cik,
            "chunker_version": self.chunker_version,
            "accession": filing.accession,
            "url": filing.url,
        }
        for batch in _batches(sections, done, set(priority_sections(filing.form)), size):
            self._check_stop()
            section = sections[batch[0]]
            if any(sections[number] != section for number in batch):  # one payload per batch
                raise RuntimeError("a batch spans two sections")
            chunks = [pieces[number][1] for number in batch]
            vectors = embedder.encode_passage([chunk.embedded_text for chunk in chunks])
            index.add(doc_id, chunks, vectors, {**base, "section": section}, first_index=batch[0])
            self._record_batch(doc_id, batch, sections, chunks, len(done) + len(batch))
            done.update(batch)
