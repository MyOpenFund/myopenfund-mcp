"""The rag toolset's only dependency: one search over the embedded corpus, behind a protocol.

EngineRagBackend runs the `rag_engine` package (the rag-engine repo's one implementation) with one
shared Qdrant store and one shared ONNX embedder for the process. Its configuration is built from
the server's `MOF_RAG_*` settings only: the engine never reads `QDRANT_*` / `RAG_ENGINE_*` from the
environment here. The ONNX runtime's telemetry is turned off by the package's `__init__`, which runs
before the imports below.

Start-up (`start()`, called once by `build_server`):
- the embedding model is loaded, which verifies every pinned model file. It is loaded there and
  nowhere else: a tool call never loads or downloads a model. A model that cannot be loaded leaves
  the toolset mounted and answering `[rag_unavailable]` until the server is restarted;
- the collection is checked (read-only). An unreachable Qdrant, a refused key or a missing
  collection also leaves the toolset mounted and answering `[rag_unavailable]`; the check is run
  again by a later call, so the toolset recovers by itself once Qdrant is back;
- the collection's filter values are read (see below).
The other toolsets behave the same way when their backend is down at start: the server starts,
their tools answer a coded error.

When Qdrant does not answer, a call pays for ONE request timeout at most:
- a failed collection check is remembered `rag_check_retry_s` seconds: calls made meanwhile, and
  calls made while another one is running the check, are answered at once with the recorded reason;
- a search, or a read of the filter values, that fails for an outage (no answer, key refused,
  collection gone, server error) ends its own call and refuses nobody else: it only makes the NEXT
  call run the collection check before its search. Qdrant is healthy: the check passes in a few
  milliseconds and that call is served. Qdrant is down: the check fails and is remembered as above.
  A search that succeeds meanwhile cancels that check;
- a read of the filter values that fails for an outage ends the call: the search is not tried after it.
A search that Qdrant refuses as a request (HTTP 4xx other than a refused key, a missing collection,
408 and 429: a filter on a field without a payload index under strict mode, for instance) is the
matter of that call alone: it answers `[rag_unavailable] search_refused`, and nothing is recorded.
No lock is held during a request to Qdrant.

Filter values (`source_code`, `doc_type`, `language`), used to refuse an unknown code:
- with Qdrant's facet API and the keyword indexes, they are read in the calling thread (a few
  milliseconds): at start, then by one call every `rag_codes_ttl_s` seconds;
- without them, ONE scroll of the collection reads the three fields, always in a background thread:
  searches are served meanwhile with the values already known, or without any check while none is;
- before a value is refused the values are read once more (not more often than every
  `rag_codes_retry_s` seconds), so a code that has just been ingested is not refused for a whole TTL;
- they are dropped when the collection check passes after having failed (the collection may have
  been recreated); an empty list refuses nothing; a failed read is tried again after
  `rag_codes_retry_s` seconds.

No error message and no `info()` holds the Qdrant host, the API key, a raw response body or the text
of a query, and neither does a log line of this module (logger `myopenfund_mcp.rag`). The HTTP
libraries under the Qdrant client are another matter: `httpx` logs one INFO line per request with
the server's address, which is why the CLI raises that logger to WARNING, and `httpcore` logs the
host and port of each connection at DEBUG, a level the CLI never turns on.
"""
from __future__ import annotations

import json
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

#: The clock of the caches (failed check, filter values). Deadlines use `time.monotonic` itself.
monotonic = time.monotonic

#: filter argument -> payload field whose stored values a requested value is checked against
CODE_FIELDS = {"source_codes": "source_code", "doc_types": "doc_type", "languages": "language"}
#: Most distinct values read per field. A field with as many (or more) is not checked: its list
#: may be cut, and a cut list would refuse valid values.
MAX_FACET_VALUES = 1000
#: Most points the fallback scroll reads (a server or a collection without the facet API or the
#: keyword indexes). A larger collection is not checked rather than checked against a partial list.
MAX_SCROLL_POINTS = 100_000
SCROLL_PAGE = 2000
#: Most valid values listed in an error message and in `info()`.
MAX_LISTED_VALUES = 60
#: Answers to a facet request that mean "not here": no keyword index on the field (400), a server
#: without the facet API (404, 405, 501) or one that does not accept the request as written (422).
#: A missing collection is a 404 too: the scroll then says so.
NO_FACET_STATUSES = (400, 404, 405, 422, 501)
#: 4xx answers to a search that say something about Qdrant or the deployment, not about the request:
#: a refused key, a missing collection, a request timeout, a rate limit. Any other 4xx is a refusal
#: of that one search.
NOT_A_REFUSAL_STATUSES = (401, 403, 404, 408, 429)

# reason -> what the caller is told. Fixed texts: never the host, the key, a response body or the
# text of an engine error.
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
    "search_refused": "the vector database refused this search, and only this one: the collection, as it "
                      "is indexed, cannot serve it (most often a filter on a field without a payload "
                      "index); retry with fewer filters, without date_from and date_to first",
    "timeout": "the search did not finish within the time allowed for one call; retry later",
    "busy": "too many searches are running and none finished within the time allowed for one call; "
            "retry later",
}


class RagBackend(Protocol):
    def start(self) -> None: ...
    def info(self) -> dict: ...
    def search(self, *, query: str, filters: dict, limit: int, max_limit: int,
               deadline: float | None = None) -> tuple[dict, list[dict]]: ...


def unavailable(reason: str) -> Exception:
    return tool_error(RAG_UNAVAILABLE, f"{reason}: {REASONS[reason]}")


class _Outage(Exception):
    """The call cannot be served, for the `[rag_unavailable]` reason given. Internal."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def quoted(value: str) -> str:
    """`value` as a JSON string: what the client must send, whatever characters it holds."""
    return json.dumps(value, ensure_ascii=False)


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
        # 404 on a search or a scroll: the collection is gone.
        return {401: "qdrant_auth", 403: "qdrant_auth",
                404: "collection_missing"}.get(exc.status_code, "qdrant_error")
    if isinstance(exc, ModelIntegrityError):
        return "model_integrity"
    if isinstance(exc, ModelDownloadError):
        return "model_unavailable"
    return None


def _refused(exc: BaseException) -> bool:
    """Whether a search failed because Qdrant refused that request: an error of one call, which says
    nothing of the next one (see NOT_A_REFUSAL_STATUSES)."""
    return (isinstance(exc, UnexpectedResponse) and 400 <= exc.status_code < 500
            and exc.status_code not in NOT_A_REFUSAL_STATUSES)


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
        self._codes_retry_s = settings.rag_codes_retry_s
        self._check_retry_s = settings.rag_check_retry_s
        self._embedder = embedder
        self._engine_settings = None
        if store is None or embedder is None:
            self._engine_settings = engine_settings(settings)
        self._store = store if store is not None else QdrantStore(settings=self._engine_settings)
        spec = embedder.spec if embedder is not None else None
        self._model = spec.name if spec else self._engine_settings.model_name
        self._dim = spec.dim if spec else self._engine_settings.model_dim
        self._start_lock = threading.Lock()
        self._started = False
        self._model_reason: str | None = None  # set at start, kept until restart
        # Guards everything below. Never held during a request to Qdrant.
        self._lock = threading.Lock()
        self._collection_reason: str | None = "qdrant_unreachable"  # until the first check passes
        self._failed_at: float | None = None  # when the last check failed
        # Why a search or a read of the values failed since the last check or successful search:
        # the next call checks the collection before it searches. It refuses nobody by itself.
        self._suspected: str | None = None
        self._checking = False  # a call is running the collection check
        self._logged: tuple[str, str] | None = None  # the failure last logged: (what failed, reason)
        self._values: dict[str, frozenset[str] | None] | None = None  # None: not known (yet)
        self._values_tried_at: float | None = None  # when the last read started
        self._values_due_at: float | None = None  # when the next read is due; None: at the next filtered call
        self._reading = False  # a read of the values is running (here or in the background thread)
        self._read_done = threading.Condition(self._lock)
        self._values_warned = False

    # -- start-up and state ------------------------------------------------------------------

    def start(self) -> None:
        """Load the model (once, here only), check the collection, read its filter values. Never
        raises for an outage; an ImportError (the [rag] extra is incomplete) is left to the caller."""
        with self._start_lock:
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
                    # It also names its own cache variable, which this server never reads.
                    text = str(exc).replace("RAG_ENGINE_MODEL_CACHE", "MOF_RAG_MODEL_CACHE")
                    log.error("rag: the embedding model could not be loaded from the directory of "
                              "MOF_RAG_MODEL_CACHE (%s): %s: %s; rag_search answers [rag_unavailable] until "
                              "the server is restarted", self._model_reason, type(exc).__name__, text)
                else:
                    log.info("rag: embedding model %s ready in %.1f s", self._model, time.monotonic() - started)
            with self._lock:
                self._checking = True
            if self._check_collection() is None:
                try:
                    self._read_values()
                except _Outage:
                    pass  # recorded: the first call checks the collection again

    def _check_collection(self) -> str | None:
        """Run the engine's read-only collection check and record the outcome: None, or the reason
        it failed for. The caller has set `_checking`; it is cleared here, whatever happens."""
        try:
            try:
                self._store.check_collection(self._collection, self._dim)
            except Exception as exc:  # noqa: BLE001 -- recorded; the class only is logged (no host, no body)
                reason = _reason(exc) or "qdrant_error"
                with self._lock:
                    changed = ("check", reason) != self._logged
                    self._collection_reason, self._logged = reason, ("check", reason)
                    self._failed_at, self._suspected = monotonic(), None
                if changed:  # once per change of state, not once per call
                    log.warning("rag: collection check failed (%s, %s); rag_search answers [rag_unavailable] "
                                "until a collection check passes", reason, type(exc).__name__)
                return reason
            with self._lock:
                back = self._collection_reason is not None or self._suspected == "collection_missing"
                if back:
                    # Back after a failure: the collection may have been recreated with other codes.
                    self._values = self._values_due_at = self._values_tried_at = None
                self._collection_reason = self._logged = self._failed_at = self._suspected = None
            if back:  # not one line per check that follows a failed search
                log.info("rag: collection %s checked, searches are served", self._collection)
            return None
        finally:
            with self._lock:
                self._checking = False

    def _suspect(self, reason: str, exc: BaseException, what: str) -> None:
        """Record that a search or a read of the values failed for an outage: the next call checks
        the collection before it searches. Nobody is refused on this record alone."""
        with self._lock:
            changed = False
            if self._collection_reason is None:  # else a failed check already says more
                changed = (what, reason) != self._logged
                self._suspected, self._logged = reason, (what, reason)
        if changed:  # once per change of state, not once per call
            log.warning("rag: %s (%s, %s); the next call checks the collection before it searches",
                        what, reason, type(exc).__name__)

    def _served(self) -> None:
        """A search succeeded: Qdrant answers, whatever a search that failed meanwhile recorded."""
        with self._lock:
            if self._suspected is not None:
                self._suspected = self._logged = None

    def _require_collection(self) -> None:
        """Return when searches are served; `_Outage` otherwise.

        After a failed check, the check is run again by one call at a time, and not before
        `rag_check_retry_s` seconds: the other calls are answered at once with the recorded reason.
        After a failed search (or read of the values), the first call to come runs the check at
        once; the calls that arrive while it runs are answered at once with the reason of that failure."""
        with self._lock:
            reason = self._collection_reason or self._suspected
            if reason is None:
                return
            # `_failed_at` is set by a failed check only, and cleared by the check that passes.
            recent = self._failed_at is not None and monotonic() - self._failed_at < self._check_retry_s
            if self._checking or recent:
                raise _Outage(reason)
            self._checking = True
        reason = self._check_collection()
        if reason is not None:
            raise _Outage(reason)

    def info(self) -> dict:
        """What `server_info` shows. Reads the recorded state only: no request, nothing secret."""
        with self._lock:
            reason, values = self._model_reason or self._collection_reason, self._values or {}
        info = {"status": "ready" if reason is None else "unavailable", "collection": self._collection,
                "model": self._model, "engine_version": ENGINE_VERSION}
        if reason is not None:
            info["reason"] = reason
        for argument, field in CODE_FIELDS.items():
            # The valid values of each filter as last read (at most MAX_LISTED_VALUES); null when unknown.
            known = values.get(field)
            info[argument] = None if known is None else sorted(known)[:MAX_LISTED_VALUES]
        return info

    # -- valid filter values -----------------------------------------------------------------

    def _facet_values(self) -> dict[str, frozenset[str] | None] | None:
        """Distinct stored values of each checked field, by the facet API; None when the server or
        the collection cannot answer a facet request (the scroll is then the way)."""
        client, values = self._store.client, {}
        for field in CODE_FIELDS.values():
            try:
                hits = client.facet(self._collection, field, limit=MAX_FACET_VALUES).hits
            except AttributeError:  # a client without the facet API
                return None
            except UnexpectedResponse as exc:
                if exc.status_code in NO_FACET_STATUSES:
                    return None
                raise
            values[field] = (None if len(hits) >= MAX_FACET_VALUES
                             else frozenset(hit.value for hit in hits if isinstance(hit.value, str)))
        return values

    def _scroll_values(self) -> dict[str, frozenset[str] | None]:
        """The same values by ONE scroll of the collection reading the three fields; None for every
        field when the collection is too large to be read in full."""
        client, fields = self._store.client, list(CODE_FIELDS.values())
        values: dict[str, set[str]] = {field: set() for field in fields}
        offset, seen = None, 0
        while True:
            points, offset = client.scroll(self._collection, limit=SCROLL_PAGE, offset=offset,
                                           with_payload=fields, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                for field in fields:
                    value = payload.get(field)
                    values[field].update(v for v in (value if isinstance(value, list) else [value])
                                         if isinstance(v, str))
            seen += len(points)
            if offset is None:
                return {field: frozenset(found) for field, found in values.items()}
            if seen >= MAX_SCROLL_POINTS:
                return dict.fromkeys(fields)

    def _read_values(self) -> None:
        """Read the filter values unless a read is already running. The facet request is made here,
        in the calling thread; when it cannot be answered, the scroll is handed to a background
        thread and this returns at once. `_Outage` when the facet request fails for an outage."""
        with self._lock:
            if self._reading:
                return
            self._reading, self._values_tried_at = True, monotonic()
        ended = False  # the end of the read is recorded, or left to the background thread
        try:
            try:
                values = self._facet_values()
                if values is None:
                    threading.Thread(target=self._scroll_in_background, name="mof-rag-values",
                                     daemon=True).start()
                else:
                    self._values_read(values, None)
                ended = True
                return
            except Exception as exc:  # noqa: BLE001 -- recorded; the search is served unchecked or not at all
                self._values_read(None, exc)
                ended = True
                reason = _reason(exc)
                if reason is None:
                    return
                error = _Outage(reason)
        finally:
            if not ended:
                self._read_abandoned()
        raise error  # outside the handler: the client error (URL, body) is not chained

    def _scroll_in_background(self) -> None:
        ended = False
        try:
            try:
                values = self._scroll_values()
            except Exception as exc:  # noqa: BLE001 -- recorded; nothing to answer from a background thread
                self._values_read(None, exc)
            else:
                self._values_read(values, None)
            ended = True
        finally:
            if not ended:
                self._read_abandoned()

    def _read_abandoned(self) -> None:
        """A read ended without a word (an exception that is not an `Exception`): `_reading` must not
        stay set, or no read would ever run again and a refusal would wait for nothing."""
        with self._lock:
            self._values_due_at = monotonic() + self._codes_retry_s
            self._reading = False
            self._read_done.notify_all()

    def _values_read(self, values: dict | None, exc: BaseException | None) -> None:
        """Record the end of a read: the values, or the failure (the values already known are kept,
        and the read is tried again after `rag_codes_retry_s` seconds, not after the whole TTL)."""
        reason = None if exc is None else _reason(exc)
        with self._lock:
            now = monotonic()
            changed = exc is None and values != self._values
            if exc is None:
                self._values, self._values_due_at, self._values_warned = values, now + self._ttl_s, False
            else:
                self._values_due_at = now + self._codes_retry_s
            warn = exc is not None and not self._values_warned
            self._values_warned = exc is not None
            self._reading = False
            self._read_done.notify_all()
        if changed:  # the counts only, and only when they change: not one line per TTL
            log.info("rag: filter values read from the collection (%s)", ", ".join(
                f"{argument}: {'not listed, too many' if values[field] is None else len(values[field])}"
                for argument, field in CODE_FIELDS.items()))
        if warn:
            log.warning("rag: the collection's filter values could not be read (%s); filter values are "
                        "checked against the last values read, if any, until they can", type(exc).__name__)
        if reason is not None:
            self._suspect(reason, exc, "reading the filter values failed")

    def _known_values(self) -> dict[str, frozenset[str] | None] | None:
        """The stored values per checked field, read again when due; None while none is known."""
        with self._lock:
            due = not self._reading and (self._values_due_at is None or monotonic() >= self._values_due_at)
        if due:
            self._read_values()
        with self._lock:
            return self._values

    def _reread_values(self, deadline: float | None) -> dict[str, frozenset[str] | None] | None:
        """Before a value is refused: the values read once more, unless they were read less than
        `rag_codes_retry_s` seconds ago. Waits for a read that is running (the background scroll),
        within the call's deadline."""
        with self._lock:
            fresh = (self._values_tried_at is not None
                     and monotonic() - self._values_tried_at < self._codes_retry_s)
            reading = self._reading
        if not fresh and not reading:
            self._read_values()
        with self._read_done:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            self._read_done.wait_for(lambda: not self._reading, timeout=wait)
            return self._values

    @staticmethod
    def _unknown(asked: dict[str, tuple[str, ...]], known: dict | None) -> dict[str, list[str]]:
        """argument -> the asked values its field does not hold. A field whose values are not known,
        or that holds none (an empty collection), refuses nothing."""
        unknown = {}
        for argument, wanted in asked.items():
            valid = (known or {}).get(CODE_FIELDS[argument])
            missing = [value for value in wanted if value not in valid] if valid else []
            if missing:
                unknown[argument] = missing
        return unknown

    def _check_values(self, filters: SearchFilters, deadline: float | None) -> None:
        asked = {argument: getattr(filters, argument) for argument in CODE_FIELDS
                 if getattr(filters, argument) is not None}
        if not asked:
            return
        known = self._known_values()
        unknown = self._unknown(asked, known)
        if unknown:
            known = self._reread_values(deadline)
            unknown = self._unknown(asked, known)
        for argument, values in unknown.items():
            valid = known[CODE_FIELDS[argument]]
            folded = {value.casefold(): value for value in sorted(valid)}
            parts = []
            for value in values:
                near = folded.get(value.strip().casefold())
                parts.append(f"{quoted(value)} (did you mean {quoted(near)}? values are matched exactly)" if near
                             else quoted(value))
            listed = sorted(valid)
            more = f", ... ({len(listed)} values)" if len(listed) > MAX_LISTED_VALUES else ""
            raise tool_error(INVALID_ARGUMENT,
                             f"{argument}: unknown value{'' if len(parts) == 1 else 's'} {', '.join(parts)}; "
                             f"valid values: {', '.join(quoted(v) for v in listed[:MAX_LISTED_VALUES])}{more}")

    # -- search ------------------------------------------------------------------------------

    def search(self, *, query: str, filters: dict, limit: int, max_limit: int,
               deadline: float | None = None) -> tuple[dict, list[dict]]:
        """(filters as applied, hits as plain dicts, best first). Blocking: call it off the event loop.

        `deadline` (a `time.monotonic()` value) is when the caller stops waiting: the search is not
        started after it.

        Raises the tool errors itself: `[invalid_argument]` with the engine's own message for an
        argument the engine refuses or a filter value the collection does not hold,
        `[rag_unavailable]` with a reason for an outage or for a search Qdrant refuses (`search_refused`:
        the error of this call alone, never recorded). Anything else is left to the server's masking.
        """
        error = None
        try:
            applied = SearchFilters.from_mapping(filters)
            if self._model_reason is not None:
                raise _Outage(self._model_reason)
            self._require_collection()
            self._check_values(applied, deadline)
            if deadline is not None and time.monotonic() >= deadline:
                raise _Outage("timeout")
            hits = search(query, collection=self._collection, store=self._store, embedder=self._embedder,
                          filters=applied, limit=limit, max_limit=max_limit)
        except SearchError as exc:
            error = tool_error(INVALID_ARGUMENT, str(exc))
        except _Outage as exc:
            error = unavailable(exc.reason)
        except Exception as exc:  # mapped below, or re-raised as it is
            if _refused(exc):
                # The request is refused, not the service: nothing is recorded, the other calls are served.
                log.info("rag: a search was refused by the vector database (HTTP %d)", exc.status_code)
                error = unavailable("search_refused")
            else:
                reason = _reason(exc)
                if reason is None:
                    raise
                # The class only: a client error's text holds the URL and the response body.
                self._suspect(reason, exc, "search failed")
                error = unavailable(reason)
        if error is not None:
            # Raised outside the handler so the client error (URL, body) is not chained.
            raise error
        self._served()
        return (
            {"source_codes": _list(applied.source_codes), "doc_types": _list(applied.doc_types),
             "languages": _list(applied.languages),
             "date_from": applied.date_from.isoformat() if applied.date_from else None,
             "date_to": applied.date_to.isoformat() if applied.date_to else None},
            [hit.to_dict() for hit in hits],
        )


def _list(values: tuple[str, ...] | None) -> list[str] | None:
    return None if values is None else list(values)
