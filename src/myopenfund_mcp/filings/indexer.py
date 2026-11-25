"""Background indexer of SEC filing documents over rag-engine's local index (no-vault mode).

A document goes through: `queued` (enqueued) -> `fetching` (its primary document is downloaded,
through the SEC client and its disk cache, and converted to Markdown by company-corpus) ->
`indexing` (chunked, embedded, added) -> `ready`; or `failed` (with a one-line error), or `partial`
(see "Restart"). `enqueue` returns at once; daemon worker threads (`MOF_INDEX_WORKERS`) do the work.

**Where things live** (under `Settings.cache_path()`; never evicted by the HTTP cache):
- `models/`: the embedding model files (a Hugging Face cache layout; 1.1 GB once);
- `index/<collection>/`: rag-engine's `LocalIndex` (Qdrant local mode + SQLite FTS5). The collection
  name (`rag_engine.local.collection_name`) holds the model, its pinned revision, the chunker
  version (`v2-t<MOF_CHUNK_MAX_TOKENS>-o<MOF_CHUNK_OVERLAP_TOKENS>`) and the ONNX provider, so
  vectors of two setups are never mixed;
- `index/status.sqlite3`: this module's record, keyed by collection: one row per document (state,
  chunks done/total, error, metadata) and one row per stored chunk (number, section, text), which
  `read` serves and a resume reads.
With `MOF_CACHE=off` both folders are in a temporary folder removed at exit (the model is then
downloaded again by every process).

**Chunks.** The Markdown is cut at the boundaries of its Item sections (`FilingDocument.sections`;
the text before, between and after them has no section), and each piece is chunked with
`rag_engine.chunking.chunk_markdown` under the v2 profile, whose token cap and overlap come from
`MOF_CHUNK_MAX_TOKENS` / `MOF_CHUNK_OVERLAP_TOKENS` (the context cap of v2, 120 tokens, is lowered to
half the token cap when the cap is below 240, the most rag-engine allows). So every chunk lies in one
section, and its `section` payload is the name of the section it starts in, exactly. Chunk numbers
run in document order; they are the `first_index` positions given to `LocalIndex.add`, so the point
ids do not depend on the order chunks are embedded in. Payload of each chunk: `doc_id`, `doc_type`
(the form), `date` (filed, `YYYY-MM-DD`), `title` (`<company> <form> <filed>`), `source_code` (the
CIK), `section`, `chunker_version`, `accession`, `url`.

**Order.** The chunks of the form's priority sections (`company_corpus.live.documents.
priority_sections`: Item 7 of a 10-K) are embedded and added first, then the others in document
order, in batches of the embedder's batch size, each batch within one section. A search serves what
is added so far: the MD&A of a document is searchable before the document is `ready`.

**Model.** Loaded in a worker when the first document is taken, never at start-up (and by `search`
when documents indexed by an earlier process are searched first). The first `enqueue` result has
`model_download=True` when the pinned model files are not yet in `models/`.

**Restart.** A document left `queued`, `fetching` or `indexing` by a process that stopped is shown as
`partial` (never `ready`) by the next process that opens the index. Enqueuing it again resumes it:
the document is chunked again (deterministic: the Markdown comes from the immutable disk cache), the
chunk numbers already stored are skipped, and it ends `ready` with the same chunk count. Should the
new chunking give another count, the document is removed and indexed again from scratch.

**One process per cache folder.** Qdrant local mode locks its folder. The index is opened by the
first call that needs it; when another process holds it, every method raises `IndexUnavailable`
(naming the lock), and a later call tries again. Nothing else of the server depends on it.

**Errors.** A document that fails is marked `failed` with `<ExceptionType>: <first line of the
message>` (the SEC contact is never written) and the worker continues with the next one.

`close()` stops the workers after their current step (a batch, a fetch or the model load) and
releases the index. A document cut that way is `partial` for the next process.
"""

from __future__ import annotations

import atexit
import dataclasses
import datetime as dt
import json
import logging
import queue
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

__all__ = [
    "STATES",
    "DocStatus",
    "DocumentNotFetched",
    "EnqueueResult",
    "FilingsIndexer",
    "IndexUnavailable",
    "SectionNotFound",
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
    """The statuses `enqueue` returns (a `list[DocStatus]`), plus `model_download`: True when this
    call starts the first load of the embedding model and its pinned files are not in the cache yet
    (1.1 GB to download)."""

    model_download: bool = False


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


_SCHEMA = (
    (
        "CREATE TABLE IF NOT EXISTS documents ("
        " collection TEXT NOT NULL, doc_id TEXT NOT NULL, cik TEXT NOT NULL, form TEXT NOT NULL,"
        " accession TEXT NOT NULL, filed TEXT, report_date TEXT, url TEXT NOT NULL, title TEXT NOT NULL,"
        " state TEXT NOT NULL, chunks_done INTEGER NOT NULL DEFAULT 0, chunks_total INTEGER, error TEXT,"
        " updated_at TEXT NOT NULL, PRIMARY KEY (collection, doc_id))"
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
        self._index: LocalIndex | None = None
        self._db: sqlite3.Connection | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._refs: dict[str, tuple[FilingRef, str]] = {}
        self._active: set[str] = set()
        self._stop = threading.Event()
        self._closed = False

    # --- paths ---------------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The folder holding `models/` and `index/` (a temporary one with MOF_CACHE=off)."""
        with self._lock:
            if self._root is None:
                if self._settings.cache == "off":
                    tmp = tempfile.TemporaryDirectory(prefix="myopenfund-mcp-filings-")
                    atexit.register(tmp.cleanup)
                    self._root = Path(tmp.name)
                else:
                    self._root = Path(self._settings.cache_path())
            return self._root

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

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
            try:
                db = sqlite3.connect(
                    self.root / "index" / "status.sqlite3", check_same_thread=False, isolation_level=None
                )
                db.execute("PRAGMA journal_mode=WAL")
                for statement in _SCHEMA:
                    db.execute(statement)
                # We hold the index lock: no other process is indexing. What was in flight is partial.
                db.execute(
                    f"UPDATE documents SET state = 'partial', updated_at = ? WHERE collection = ? "
                    f"AND state IN ({', '.join('?' * len(_IN_FLIGHT))})",
                    (_now(), self.collection, *_IN_FLIGHT),
                )
            except Exception:
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
        """The embedder and the tokenizer, loaded on first use (one load for every thread)."""
        with self._model_lock:
            if self._embedder is None:
                models = str(self.models_dir)
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
                self._embedder, self._tokenizer = embedder, tokenizer
            return self._embedder, self._tokenizer

    def _model_download_needed(self) -> bool:
        if self._embedder is not None or self._embedder_factory is not None:
            return False
        return not model_files_cached(self.models_dir, self._spec.name)

    # --- the status database -------------------------------------------------------------------

    def _rows(self, doc_ids: Collection[str] | None) -> list[tuple]:
        assert self._db is not None
        with self._lock:
            if doc_ids is None:
                return self._db.execute(
                    f"SELECT {_COLUMNS} FROM documents WHERE collection = ? ORDER BY filed DESC, doc_id",
                    (self.collection,),
                ).fetchall()
            return self._db.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE collection = ? AND doc_id IN "
                "(SELECT value FROM json_each(?)) ORDER BY filed DESC, doc_id",
                (self.collection, json.dumps(list(doc_ids))),
            ).fetchall()

    def _get(self, doc_id: str) -> DocStatus | None:
        rows = self._rows([doc_id])
        return _status(rows[0]) if rows else None

    def _update(self, doc_id: str, **values: Any) -> None:
        assert self._db is not None
        values["updated_at"] = _now()
        assignments = ", ".join(f"{name} = ?" for name in values)
        with self._lock:
            self._db.execute(
                f"UPDATE documents SET {assignments} WHERE collection = ? AND doc_id = ?",
                (*values.values(), self.collection, doc_id),
            )

    def _done(self, doc_id: str) -> set[int]:
        assert self._db is not None
        with self._lock:
            rows = self._db.execute(
                "SELECT chunk_number FROM chunks WHERE collection = ? AND doc_id = ?",
                (self.collection, doc_id),
            ).fetchall()
        return {int(row[0]) for row in rows}

    # --- public API ----------------------------------------------------------------------------

    def enqueue(self, refs: Sequence[FilingRef], *, names: Mapping[str, str] | None = None) -> EnqueueResult:
        """Queue `refs` for indexing and return their statuses at once. A `ready` document stays
        ready; one already queued or being indexed by this process is left as it is; a `partial` or
        `failed` one is queued again (a partial one resumes). `names` maps a CIK to the company name
        used in the chunks' `title` (default `CIK <cik>`)."""
        self._open()
        result = EnqueueResult()
        result.model_download = self._model_download_needed()
        self._start_workers()
        names = names or {}
        for filing in refs:
            doc_id = _default_doc_id(filing)
            filed = filing.filed.isoformat() if filing.filed else None
            title = f"{names.get(filing.cik) or f'CIK {filing.cik}'} {filing.form} {filed or 'undated'}"
            with self._lock:
                current = self._get(doc_id)
                if current is not None and (current.state == "ready" or doc_id in self._active):
                    result.append(current)
                    continue
                assert self._db is not None
                self._db.execute(
                    "INSERT INTO documents (collection, doc_id, cik, form, accession, filed, report_date, url,"
                    " title, state, chunks_done, chunks_total, error, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, NULL, NULL, ?)"
                    " ON CONFLICT (collection, doc_id) DO UPDATE SET cik = excluded.cik, form = excluded.form,"
                    " accession = excluded.accession, filed = excluded.filed,"
                    " report_date = excluded.report_date, url = excluded.url, title = excluded.title,"
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
                self._refs[doc_id] = (filing, title)
                self._active.add(doc_id)
                self._queue.put(doc_id)
                status = self._get(doc_id)
                assert status is not None
                result.append(status)
        return result

    def status(self, doc_ids: Collection[str] | None = None) -> list[DocStatus]:
        """The statuses of `doc_ids` (every known document when None), newest filing first; a
        document never enqueued is left out."""
        self._open()
        return [_status(row) for row in self._rows(doc_ids)]

    def search(
        self,
        query: str,
        *,
        doc_ids: Collection[str] | None = None,
        forms: Collection[str] | None = None,
        period_from: dt.date | None = None,
        period_to: dt.date | None = None,
        limit: int = 10,
    ) -> tuple[list[Hit], list[DocStatus]]:
        """Hybrid search (dense + BM25, RRF; eigenmind selection when MOF_SEARCH_EIGENMIND) over the
        chunks indexed so far of the requested documents, and the statuses of those not `ready`.

        Documents: `doc_ids` (every known document when None), restricted to `forms` (a form
        includes its `/A` amendments) and to a report period overlapping `period_from`..`period_to`
        (the report date, or the filing date when the filing has none), as `companies_filings`."""
        self._open()
        wanted = None if forms is None else {f.upper() for f in forms} | {f"{f.upper()}/A" for f in forms}
        rows = []
        for row in self._rows(doc_ids):
            period = row[10] or row[4]
            day = dt.date.fromisoformat(period) if period else None
            if wanted is not None and row[2].upper() not in wanted:
                continue
            if (period_from or period_to) and day is None:
                continue
            if period_from and day < period_from or period_to and day > period_to:
                continue
            rows.append(_status(row))
        not_ready = [status for status in rows if status.state != "ready"]
        searchable = [status.doc_id for status in rows if status.chunks_done > 0]
        if not searchable:
            return [], not_ready
        embedder, _ = self._model()
        assert self._index is not None
        hits = hybrid_search(
            query,
            index=self._index,
            embedder=embedder,
            filters=SearchFilters(doc_ids=searchable),
            limit=limit,
            candidates=self._settings.search_candidates,
            rrf_k=self._settings.search_rrf_k,
            selector=select_diverse if self._settings.search_eigenmind else None,
        )
        return hits, not_ready

    def read(
        self,
        doc_id: str,
        *,
        section: str | None = None,
        chunk_id: str | int | None = None,
        around: int = 1,
    ) -> dict[str, Any]:
        """The stored chunks of `doc_id`, in order, as a result whose JSON stays within
        MOF_MCP_MAX_RESULT_CHARS: from the start of `section` to its end, or `around` chunks on each
        side of `chunk_id`, or from the first chunk. `more`/`next_chunk_id` give where to go on
        (`read(doc_id, chunk_id=next_chunk_id, around=0)` then reads on from there)."""
        if section is not None and chunk_id is not None:
            raise ValueError("give section or chunk_id, not both")
        if type(around) is not int or around < 0:
            raise ValueError(f"around must be an integer >= 0, got {around!r}")
        self._open()
        status = self._get(doc_id)
        if status is None:
            raise DocumentNotFetched(f"document {doc_id!r} was never fetched")
        assert self._db is not None
        with self._lock:
            rows = self._db.execute(
                "SELECT chunk_number, section, text FROM chunks WHERE collection = ? AND doc_id = ?"
                " ORDER BY chunk_number",
                (self.collection, doc_id),
            ).fetchall()
        start, stop = 0, len(rows)
        if section is not None:
            numbers = [i for i, row in enumerate(rows) if row[1] == section]
            if not numbers:
                available = list(dict.fromkeys(row[1] for row in rows if row[1] is not None))
                raise SectionNotFound(
                    f"document {doc_id!r} has no stored chunk in section {section!r}; sections: "
                    f"{', '.join(available) or 'none yet'}"
                )
            start = numbers[0]
            stop = start
            while stop < len(rows) and rows[stop][1] == section:
                stop += 1
        elif chunk_id is not None:
            try:
                number = int(chunk_id)
            except (TypeError, ValueError):
                raise ValueError(f"chunk_id must be a chunk number, got {chunk_id!r}") from None
            positions = [i for i, row in enumerate(rows) if row[0] == number]
            if not positions:
                raise LookupError(f"document {doc_id!r} has no stored chunk {chunk_id!r}")
            start, stop = max(0, positions[0] - around), min(len(rows), positions[0] + around + 1)
        result: dict[str, Any] = {
            **status.to_dict(),
            "section": section,
            "chunks": [],
            "more": False,
            "next_chunk_id": None,
        }
        budget = self._settings.max_result_chars
        size = len(json.dumps(result))
        for i in range(start, stop):
            number, name, text = rows[i]
            entry = {"chunk_id": str(number), "section": name, "text": text}
            cost = len(json.dumps(entry)) + 2  # ", " between list items
            if size + cost > budget:
                if not result["chunks"]:
                    # One chunk larger than the whole budget: its text is cut, and said so.
                    room = max(
                        0, budget - size - len(json.dumps({**entry, "text": "", "text_cut": True})) - 2
                    )
                    entry = {**entry, "text": text[:room], "text_cut": True}
                    while len(json.dumps(entry)) + 2 + size > budget and entry["text"]:
                        entry["text"] = entry["text"][:-1]
                    result["chunks"].append(entry)
                    i += 1
                result["more"] = i < len(rows)
                result["next_chunk_id"] = str(rows[i][0]) if i < len(rows) else None
                break
            result["chunks"].append(entry)
            size += cost
        return result

    def close(self) -> None:
        """Stop the workers after their current step and release the index. Twice is harmless."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            workers = list(self._workers)
        for _ in workers:
            self._queue.put(None)
        for worker in workers:
            worker.join()
        with self._lock:
            if self._index is not None:
                self._index.close()
            if self._db is not None:
                self._db.close()
        if self._embedder is not None:
            self._embedder.release()

    # --- the worker ----------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            doc_id = self._queue.get()
            if doc_id is None or self._stop.is_set():
                return
            try:
                self._index_one(doc_id)
            except Exception as exc:  # noqa: BLE001 - one document fails, the worker goes on
                log.warning("filing %s failed: %s", doc_id, type(exc).__name__)
                try:
                    self._update(doc_id, state="failed", error=_short_error(exc, self._settings.sec_contact))
                except Exception:  # the database may be closing with the process
                    log.exception("could not record the failure of filing %s", doc_id)
            finally:
                with self._lock:
                    self._active.discard(doc_id)

    def _index_one(self, doc_id: str) -> None:
        filing, title = self._refs[doc_id]
        index = self._index
        assert index is not None
        self._update(doc_id, state="fetching", error=None)
        embedder, tokenizer = self._model()
        document = self._loader(filing, self._sec)
        pieces = document_chunks(document, options=self.options, tokenizer=tokenizer, title=title)
        current = self._get(doc_id)
        done = self._done(doc_id)
        if current is not None and current.chunks_total is not None and current.chunks_total != len(pieces):
            # Another cut than the one stored: start this document again.
            index.delete(doc_id)
            assert self._db is not None
            with self._lock:
                self._db.execute(
                    "DELETE FROM chunks WHERE collection = ? AND doc_id = ?", (self.collection, doc_id)
                )
            done = set()
        self._update(doc_id, state="indexing", chunks_total=len(pieces), chunks_done=len(done))
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
            if self._stop.is_set():
                return  # left `indexing`: the next process shows it `partial`
            chunks = [pieces[number][1] for number in batch]
            vectors = embedder.encode_passage([chunk.embedded_text for chunk in chunks])
            index.add(doc_id, chunks, vectors, {**base, "section": sections[batch[0]]}, first_index=batch[0])
            assert self._db is not None
            with self._lock, self._db:  # one transaction: rolled back if a write fails
                self._db.execute("BEGIN")
                self._db.executemany(
                    "INSERT OR REPLACE INTO chunks (collection, doc_id, chunk_number, section, text, context)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (self.collection, doc_id, number, sections[number], chunk.text, chunk.context)
                        for number, chunk in zip(batch, chunks)
                    ],
                )
                self._db.execute(
                    "UPDATE documents SET chunks_done = ?, updated_at = ? WHERE collection = ? AND doc_id = ?",
                    (len(done | set(batch)), _now(), self.collection, doc_id),
                )
            done.update(batch)
        self._update(doc_id, state="ready", error=None)
