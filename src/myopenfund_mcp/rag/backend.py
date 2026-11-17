"""The rag toolset's only dependency: one search over the embedded corpus, behind a protocol.

EngineRagBackend runs the `rag_engine` package (the rag-engine repo's one implementation) with one
shared Qdrant store and one shared ONNX embedder for the process. Its configuration is built from
the server's `MOF_RAG_*` settings only: the engine never reads `QDRANT_*` / `RAG_ENGINE_*` from the
environment here.

Start-up (`start()`, called once by `build_server`):
- the embedding model is loaded, which verifies every pinned model file. It is loaded there and
  nowhere else: a tool call never loads or downloads a model. A model that cannot be loaded leaves
  the toolset mounted and answering `[rag_unavailable]` until the server is restarted;
- the collection is checked (read-only). An unreachable Qdrant, a refused key or a missing
  collection also leaves the toolset mounted and answering `[rag_unavailable]`; that check is run
  again at the next call, so the toolset recovers by itself once Qdrant is back.
The other toolsets behave the same way when their backend is down at start: the server starts,
their tools answer a coded error.

No error message, log line or `info()` ever holds the Qdrant host, the API key, a raw response body
or the text of a query.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from rag_engine import __version__ as ENGINE_VERSION
from rag_engine.config import ConfigError as EngineConfigError
from rag_engine.config import Settings as EngineSettings
from rag_engine.embedding import ModelDownloadError, ModelIntegrityError, make_embedder
from rag_engine.search import SearchError, SearchFilters, search
from rag_engine.store import CollectionMismatchError, CollectionNotFoundError, QdrantAuthError, QdrantStore

from ..config import ConfigError, Settings
from ..errors import INVALID_ARGUMENT, RAG_UNAVAILABLE, tool_error

log = logging.getLogger("myopenfund_mcp.rag")

#: filter argument -> payload field whose stored values a requested value is checked against
CODE_FIELDS = {"source_codes": "source_code", "doc_types": "doc_type", "languages": "language"}
#: Most distinct values read per field. A field with as many (or more) is not checked: its list
#: may be cut, and a cut list would refuse valid values.
MAX_FACET_VALUES = 1000
#: Most points the fallback scroll reads (a server or a collection without the facet API or the
#: keyword indexes). A larger collection is not checked rather than checked against a partial list.
MAX_SCROLL_POINTS = 100_000
SCROLL_PAGE = 2000
#: Most valid values listed in an error message.
MAX_LISTED_VALUES = 60

# reason -> what the caller is told. Fixed texts: never the host, the key or a response body.
REASONS = {
    "qdrant_unreachable": "the vector database cannot be reached or did not answer in time; retry later",
    "qdrant_auth": "the vector database refused the server's API key; the deployment must be fixed",
    "qdrant_error": "the vector database answered with an error; retry later",
    "collection_missing": "the configured collection does not exist in the vector database; "
                          "the deployment must be fixed",
    "collection_mismatch": "the configured collection does not match the embedding model (vector size or "
                           "distance); the deployment must be fixed",
    "model_integrity": "an embedding model file is not the pinned one; the deployment must be fixed",
    "model_unavailable": "the embedding model could not be loaded when the server started; "
                         "the deployment must be fixed",
}


class RagBackend(Protocol):
    def start(self) -> None: ...
    def info(self) -> dict: ...
    def search(self, *, query: str, filters: dict, limit: int, max_limit: int) -> tuple[dict, list[dict]]: ...


def unavailable(reason: str) -> Exception:
    return tool_error(RAG_UNAVAILABLE, f"{reason}: {REASONS[reason]}")


def _reason(exc: BaseException) -> str | None:
    """The `[rag_unavailable]` reason of an engine or Qdrant-client error; None for anything else."""
    if isinstance(exc, QdrantAuthError):
        return "qdrant_auth"
    if isinstance(exc, CollectionNotFoundError):
        return "collection_missing"
    if isinstance(exc, CollectionMismatchError):
        return "collection_mismatch"
    if isinstance(exc, ResponseHandlingException):  # connection refused, DNS, TLS, timeout
        return "qdrant_unreachable"
    if isinstance(exc, UnexpectedResponse):
        # On a search: 404 = the collection is gone, 400 = a vector of another dimension.
        return {404: "collection_missing", 400: "collection_mismatch"}.get(exc.status_code, "qdrant_error")
    if isinstance(exc, ModelIntegrityError):
        return "model_integrity"
    if isinstance(exc, ModelDownloadError):
        return "model_unavailable"
    return None


def engine_settings(settings: Settings) -> EngineSettings:
    """The engine's settings, from the server's `MOF_RAG_*` values and nothing else."""
    try:
        return EngineSettings(
            qdrant_host=settings.rag_qdrant_host,
            qdrant_port=settings.rag_qdrant_port,
            qdrant_api_key=settings.rag_qdrant_api_key,
            qdrant_https=settings.rag_qdrant_https,
            qdrant_timeout=settings.rag_qdrant_timeout,
            backend="onnx",  # queries are embedded without PyTorch (CPU only)
            model_cache=settings.rag_model_cache,
            onnx_threads=settings.rag_onnx_threads,
            search_limit=settings.rag_default_limit,
            search_max_limit=settings.rag_max_limit,
        )
    except EngineConfigError as exc:  # the engine names its own variables; it never echoes the key
        error = ConfigError(f"the MOF_RAG_* settings were refused by rag-engine ({exc})")
    raise error


class EngineRagBackend:
    """`store` and `embedder` are for tests only (a seeded store, an embedder without a model):
    a deployment builds both from the settings."""

    def __init__(self, settings: Settings, *, store: Any = None, embedder: Any = None):
        self._collection = settings.rag_collection
        self._ttl_s = settings.rag_codes_ttl_s
        self._embedder = embedder
        self._engine_settings = None
        if store is None or embedder is None:
            self._engine_settings = engine_settings(settings)
        self._store = store if store is not None else QdrantStore(settings=self._engine_settings)
        spec = embedder.spec if embedder is not None else None
        self._model = spec.name if spec else self._engine_settings.model_name
        self._dim = spec.dim if spec else self._engine_settings.model_dim
        self._lock = threading.Lock()
        self._started = False
        self._model_reason: str | None = None  # set at start, kept until restart
        self._collection_reason: str | None = "qdrant_unreachable"  # until the first check passes
        self._logged_reason: str | None = None
        self._codes: dict[str, frozenset[str] | None] | None = None
        self._codes_read_at = 0.0
        self._codes_warned = False

    # -- start-up and state ------------------------------------------------------------------

    def start(self) -> None:
        """Load the model (once, here only) and check the collection. Never raises for an outage;
        an ImportError (the [rag] extra is incomplete) is left to the caller."""
        with self._lock:
            if self._started:
                return
            self._started = True
            if self._embedder is None:
                started = time.monotonic()
                try:
                    self._embedder = make_embedder(self._engine_settings)
                except ImportError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- the toolset answers [rag_unavailable] instead
                    self._model_reason = _reason(exc) or "model_unavailable"
                    # The engine's text names the file, the directory and the hashes: no secret in it.
                    log.error("rag: the embedding model could not be loaded (%s): %s: %s; rag_search answers "
                              "[rag_unavailable] until the server is restarted",
                              self._model_reason, type(exc).__name__, exc)
                else:
                    log.info("rag: embedding model %s ready in %.1f s", self._model, time.monotonic() - started)
            self._check_collection()

    def _check_collection(self) -> None:
        """Run the engine's read-only collection check and record the outcome. Lock held."""
        try:
            self._store.check_collection(self._collection, self._dim)
        except Exception as exc:  # noqa: BLE001 -- recorded; the class only is logged (no host, no body)
            reason = _reason(exc)
            if reason is None:
                reason = "qdrant_error"
            if reason != self._logged_reason:  # once per change of state, not once per call
                log.warning("rag: collection check failed (%s, %s); rag_search answers [rag_unavailable] "
                            "until a check passes", reason, type(exc).__name__)
            self._collection_reason = self._logged_reason = reason
        else:
            log.info("rag: collection %s checked, searches are served", self._collection)
            self._collection_reason = self._logged_reason = None

    def _status(self) -> str | None:
        return self._model_reason or self._collection_reason

    def info(self) -> dict:
        """What `server_info` shows. Reads the recorded state only: no request, nothing secret."""
        reason = self._status()
        info = {"status": "ready" if reason is None else "unavailable", "collection": self._collection,
                "model": self._model, "engine_version": ENGINE_VERSION}
        if reason is not None:
            info["reason"] = reason
        return info

    # -- valid filter values -----------------------------------------------------------------

    def _read_values(self, field: str) -> frozenset[str] | None:
        """Distinct stored values of one payload field; None when they cannot be listed in full."""
        client = self._store.client
        try:
            hits = client.facet(self._collection, field, limit=MAX_FACET_VALUES).hits
        except (AttributeError, UnexpectedResponse):
            # A client or a server without the facet API, or a field without its keyword index.
            return self._scroll_values(field)
        if len(hits) >= MAX_FACET_VALUES:
            return None
        return frozenset(hit.value for hit in hits if isinstance(hit.value, str))

    def _scroll_values(self, field: str) -> frozenset[str] | None:
        client, values, offset, seen = self._store.client, set(), None, 0
        while True:
            points, offset = client.scroll(self._collection, limit=SCROLL_PAGE, offset=offset,
                                           with_payload=[field], with_vectors=False)
            for point in points:
                value = (point.payload or {}).get(field)
                values.update(v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
            seen += len(points)
            if offset is None:
                return frozenset(values)
            if seen >= MAX_SCROLL_POINTS:
                return None

    def _known_values(self) -> dict[str, frozenset[str] | None]:
        """Stored values per checked field, read at most every `rag_codes_ttl_s` seconds.

        A read that fails degrades to no check (None for every field, logged once), never to a failed
        search; it is tried again after the same delay.
        """
        with self._lock:
            now = time.monotonic()
            if self._codes is not None and now - self._codes_read_at < self._ttl_s:
                return self._codes
            try:
                codes = {field: self._read_values(field) for field in CODE_FIELDS.values()}
                self._codes_warned = False
            except Exception as exc:  # noqa: BLE001 -- the search itself reports an outage
                if not self._codes_warned:
                    log.warning("rag: the collection's filter values could not be read (%s); filter values "
                                "are not checked until they can", type(exc).__name__)
                    self._codes_warned = True
                codes = dict.fromkeys(CODE_FIELDS.values())
            self._codes, self._codes_read_at = codes, now
            return codes

    def _check_values(self, filters: SearchFilters) -> None:
        known = None
        for argument, field in CODE_FIELDS.items():
            asked = getattr(filters, argument)
            if asked is None:
                continue
            known = self._known_values() if known is None else known
            valid = known[field]
            unknown = [value for value in asked if value not in valid] if valid is not None else []
            if not unknown:
                continue
            folded = {value.casefold(): value for value in sorted(valid)}
            parts = []
            for value in unknown:
                near = folded.get(value.strip().casefold())
                parts.append(f"{value!r} (did you mean {near!r}? values are matched exactly)" if near
                             else repr(value))
            listed = sorted(valid)
            more = f", ... ({len(listed)} values)" if len(listed) > MAX_LISTED_VALUES else ""
            raise tool_error(INVALID_ARGUMENT,
                             f"{argument}: unknown value{'' if len(parts) == 1 else 's'} {', '.join(parts)}; "
                             f"valid values: {', '.join(listed[:MAX_LISTED_VALUES])}{more}")

    # -- search ------------------------------------------------------------------------------

    def search(self, *, query: str, filters: dict, limit: int, max_limit: int) -> tuple[dict, list[dict]]:
        """(filters as applied, hits as plain dicts, best first). Blocking: call it off the event loop.

        Raises the tool errors itself: `[invalid_argument]` with the engine's own message for an
        argument the engine refuses or a filter value the collection does not hold,
        `[rag_unavailable]` with a reason for an outage. Anything else is left to the server's masking.
        """
        error = None
        try:
            applied = SearchFilters.from_mapping(filters)
            if self._model_reason is not None:
                raise unavailable(self._model_reason)
            if self._collection_reason is not None:
                with self._lock:
                    self._check_collection()
                if self._collection_reason is not None:
                    raise unavailable(self._collection_reason)
            self._check_values(applied)
            hits = search(query, collection=self._collection, store=self._store, embedder=self._embedder,
                          filters=applied, limit=limit, max_limit=max_limit)
        except SearchError as exc:
            error = tool_error(INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # mapped below, or re-raised as it is
            reason = _reason(exc)
            if reason is None:
                raise
            # The class only: a client error's text holds the URL and the response body.
            log.warning("rag: search failed (%s, %s)", reason, type(exc).__name__)
            self._collection_reason = reason  # the next call checks the collection again first
            error = unavailable(reason)
        if error is not None:
            # Raised outside the handler so the client error (URL, body) is not chained.
            raise error
        return (
            {"source_codes": _list(applied.source_codes), "doc_types": _list(applied.doc_types),
             "languages": _list(applied.languages),
             "date_from": applied.date_from.isoformat() if applied.date_from else None,
             "date_to": applied.date_to.isoformat() if applied.date_to else None},
            [hit.to_dict() for hit in hits],
        )


def _list(values: tuple[str, ...] | None) -> list[str] | None:
    return None if values is None else list(values)
