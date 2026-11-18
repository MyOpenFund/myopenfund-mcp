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
matter of that call alone: it answers `[rag_unavailable] search_refused`, and refuses nobody else (the next call checks the collection first).
No lock is held during a request to Qdrant.

Date bounds (`rag_date_filter_mode`, passed to the engine's `search` with its refresh and retry
delays: the engine's own `RAG_ENGINE_DATE_FILTER_*` variables are never read). In `auto` mode a call
with a date bound may need one more request: the read of the collection's payload schema (are
`date_start` and `date_end` indexed?), at most once every `rag_date_filter_refresh_s` seconds. It is
made here, in the call's own thread (its slot) and before the deadline check, by the engine's
`QdrantStore.uses_date_intervals`, and the mode it gives is handed to `search`: a search is never
started after the call's deadline because of that read. The read never fails a call: when it fails,
the engine answers with the last mode it read, or `legacy`, and reads again after
`rag_date_filter_retry_s` seconds. It is therefore not an outage for the toolset either: nothing of it
reaches the collection check's memory, and only the search that follows says whether Qdrant answers.
It is also made once at start, after a collection check and a read of the filter values that passed,
so the first date-bounded call does not pay for it and the log says which filter is in use.

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
of a query, and neither does a log line of this module (logger `myopenfund_mcp.rag`) nor of rag-engine
(loggers `rag_engine` and `rag_engine.*`): building the backend routes the engine's records through
`EngineLogFilter`, which writes those that carry an error without its text (see there). The HTTP
libraries under the Qdrant client are another matter: `httpx` logs one INFO line per request with
the server's address, which is why the CLI raises that logger to WARNING, and `httpcore` logs the
host and port of each connection at DEBUG, a level the CLI never turns on.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Protocol

from fastmcp.exceptions import ToolError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from rag_engine import __version__ as ENGINE_VERSION
from rag_engine.config import ConfigError as EngineConfigError
from rag_engine.config import Settings as EngineSettings
from rag_engine.embedding import ModelDownloadError, ModelIntegrityError, make_embedder
from rag_engine.search import SearchError, SearchFilters, search
from rag_engine.store import (
    CollectionMismatchError,
    CollectionNotFoundError,
    MissingIndexError,
    QdrantAuthError,
    QdrantStore,
)

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
    "internal_error": "the server failed on this search for a reason of its own (its log names the error's "
                      "class); retry, and tell the operator if it lasts",
}


#: The engine's warning when the payload schema of a collection cannot be read (rag_engine.store,
#: `uses_date_intervals`): its arguments are (collection, error class, error, mode used meanwhile,
#: seconds before the next try). Rewritten without the error itself.
_SCHEMA_WARNING = ("The payload schema of Qdrant collection %r could not be read (%s: %s): date filters in auto "
                   "mode use %s until it can; next try in %g seconds at the earliest")
class EngineLogFilter(logging.Filter):
    """Keeps the server's log free of what rag-engine's records may carry from a failed request: the
    Qdrant host (in the client's error, a TLS error), a response body (a proxy's error page naming an
    upstream host or an internal address), the engine's hint naming `QDRANT_API_KEY`.

    A record of an engine logger is reduced when an exception is among its arguments, when it carries
    an exception (`logger.exception`, `exc_info`), or when its text holds the configured Qdrant host
    (as a whole token, as configured or lowercased, with or without a port) or API key (anywhere) (`_sensitive`; the
    configured collection, as the engine quotes it, never counts unless it is also the host). What is kept: the logger, the level, and a fixed message naming the class of each
    exception. The engine's warning on a schema read that failed keeps, besides, its collection, the
    filter used meanwhile and the delay before the next try. A record without any of these (the INFO
    lines naming a collection and the filter in use) is written as the engine wrote it. Never drops a
    record."""

    def __init__(self):
        super().__init__()
        self._protecting = threading.Lock()  # one change at a time
        # (hosts, keys, collections, the pattern of the hosts): replaced whole, never changed, so a thread
        # that logs reads one consistent state without a lock while another backend is built.
        self._state: tuple[frozenset[str], frozenset[str], frozenset[str], re.Pattern | None] = (
            frozenset(), frozenset(), frozenset(), None)

    def protect(self, host: str | None, key: str | None, collection: str | None = None) -> None:
        """The configured host and API key must never be written; the collection is not sensitive."""
        with self._protecting:
            hosts, keys, collections, _ = self._state
            if host:
                # As configured and lowercased (the HTTP client writes it lowercased in its errors), bare
                # too when it was configured as a bracketed IPv6 literal. Not matched without case: the
                # engine's own lines start with the word "Qdrant", a usual host name.
                hosts = hosts | {host, host.lower(), host.strip("[]"), host.strip("[]").lower()}
            keys = keys | ({key} if key else set())
            collections = collections | ({collection} if collection else set())
            # A host is not inside a longer name: no name character before, none after (a dot ending a
            # sentence is no part of a name, a dot followed by a name character is). Longest first.
            alternatives = "|".join(re.escape(value) for value in sorted(hosts, key=len, reverse=True))
            pattern = (re.compile(rf"(?<![A-Za-z0-9._-])(?:{alternatives})(?::[0-9]+)?(?![A-Za-z0-9_-]|\.[A-Za-z0-9])")
                       if hosts else None)
            self._state = (frozenset(hosts), frozenset(keys), frozenset(collections), pattern)

    def _sensitive(self, text: str) -> bool:
        """Whether `text` holds a protected value. The key anywhere, as a plain substring (a false alarm
        on a key costs one reduced line). The host as a whole token, as configured or lowercased, with or
        without `:port`: `Qdrant-Prod` is found in `http://qdrant-prod:6333/`, `qdrant` is not found in
        `qdrant-docs`, `qdrant.example.org` nor the engine's word "Qdrant". A configured collection written as the engine writes it, quoted
        (`'qdrant-docs'`), is taken out first, unless it is also a protected host."""
        hosts, keys, collections, pattern = self._state
        if any(key in text for key in keys):
            return True
        if pattern is None:
            return False
        lowered = {host.lower() for host in hosts}
        for collection in collections:
            if collection.lower() not in lowered:
                text = text.replace(repr(collection), "")
        return pattern.search(text) is not None

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        errors = [arg for arg in args if isinstance(arg, BaseException)]
        if record.exc_info and isinstance(record.exc_info, tuple) and record.exc_info[1] is not None:
            errors.append(record.exc_info[1])
        if not errors:
            try:
                text = record.getMessage()
            except Exception:  # noqa: BLE001 -- a malformed record: reduced rather than written
                text = None
            if text is not None and not self._sensitive(text):
                return True
        classes = ", ".join(dict.fromkeys(type(error).__name__ for error in errors)) or "none"
        if record.msg == _SCHEMA_WARNING and len(args) == 5:
            record.msg = ("The payload schema of Qdrant collection %r could not be read (%s; its text is "
                          "not logged): date filters in auto mode use %s until it can; next try in %g "
                          "seconds at the earliest")
            record.args = (args[0], classes, args[3] if isinstance(args[3], str) else "the last known answer",
                           args[4] if isinstance(args[4], (int, float)) else 0)
        else:
            record.msg = ("rag-engine logged a message about an error (%s); its text is not logged: it may "
                          "hold the vector database's address, a response body or a credential hint")
            record.args = (classes,)
        record.exc_info = record.exc_text = record.stack_info = None
        return True


ENGINE_LOG_FILTER = EngineLogFilter()


class _EngineLogRelay(logging.Handler):
    """The first handler of the `rag_engine` logger, which no longer propagates: every record of the
    engine, from `rag_engine` or any `rag_engine.*` logger, existing or created later, goes through
    `ENGINE_LOG_FILTER` here, then on to the root logger's handlers, as it would have propagated (a
    handler attached to both loggers, as pytest attaches its own, is left to get it from `rag_engine`).
    When no handler takes the record, `logging.lastResort` writes it (WARNING and above), as the
    standard library does.

    What `propagate = False` costs, accepted for the privacy it buys: the change is made on the process's
    `rag_engine` logger and lasts as long as the process (another user of rag-engine in the same process
    gets the filter too); handlers of loggers between `rag_engine` and the root are skipped (there are
    none: the engine's loggers sit directly under the root); and a later `logging.config.dictConfig` that
    names `rag_engine` replaces its handlers, the relay included, and so gives the records back unfiltered
    (the filters put on the engine's existing loggers stay unless the configuration replaces those too)."""

    def __init__(self):
        super().__init__(level=logging.NOTSET)
        self.addFilter(ENGINE_LOG_FILTER)

    def emit(self, record: logging.LogRecord) -> None:
        engine = logging.getLogger("rag_engine").handlers
        found = sum(1 for handler in engine if handler is not self)
        for handler in logging.getLogger().handlers:
            if handler not in engine:
                found += 1
                if record.levelno >= handler.level:
                    handler.handle(record)
        # No handler anywhere on the way: what the standard library does when a record finds none.
        last = logging.lastResort
        if not found and last is not None and record.levelno >= last.level:
            last.handle(record)


def install_engine_log_filter(host: str | None, key: str | None, collection: str | None = None) -> None:
    """Route the engine's records through `ENGINE_LOG_FILTER` (once), and protect `host` and `key`. The
    filter is also put on the engine's loggers that exist already: a record is reduced before any handler
    sees it, whatever the order of the handlers."""
    ENGINE_LOG_FILTER.protect(host, key, collection)
    engine = logging.getLogger("rag_engine")
    # By name: a module imported again (tests do) defines the class again, and one relay is enough.
    if not any(type(handler).__name__ == _EngineLogRelay.__name__ for handler in engine.handlers):
        engine.handlers.insert(0, _EngineLogRelay())
    engine.propagate = False
    for name in [name for name in list(logging.root.manager.loggerDict) if name.startswith("rag_engine.")]:
        logger = logging.getLogger(name)
        if ENGINE_LOG_FILTER not in logger.filters:
            logger.addFilter(ENGINE_LOG_FILTER)
    if ENGINE_LOG_FILTER not in engine.filters:
        engine.addFilter(ENGINE_LOG_FILTER)


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
    if isinstance(exc, json.JSONDecodeError):
        # An answer that is not JSON, such as a proxy's HTML page sent with status 200: the client does
        # not wrap that one. Qdrant cannot be read, as with an error status.
        return "qdrant_error"
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
    nothing of the next one (see NOT_A_REFUSAL_STATUSES). The engine raises its own `MissingIndexError`
    for the 400 of a collection in strict mode asked to filter on a field without an index."""
    if isinstance(exc, MissingIndexError):
        return True
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
            date_filter_mode=settings.rag_date_filter_mode,
            date_filter_refresh=settings.rag_date_filter_refresh_s,
            date_filter_retry=settings.rag_date_filter_retry_s,
        )
    except EngineConfigError as exc:  # the engine names its own variables; it never echoes the key
        error = ConfigError(f"the MOF_RAG_* settings were refused by rag-engine ({exc})")
    raise error


class EngineRagBackend:
    """`store` and `embedder` are for tests only (a seeded store, an embedder without a model):
    a deployment builds both from the settings."""

    def __init__(self, settings: Settings, *, store: Any = None, embedder: Any = None):
        install_engine_log_filter(settings.rag_qdrant_host, settings.rag_qdrant_api_key, settings.rag_collection)
        self._collection = settings.rag_collection
        self._ttl_s = settings.rag_codes_ttl_s
        self._codes_retry_s = settings.rag_codes_retry_s
        self._check_retry_s = settings.rag_check_retry_s
        self._date_mode = settings.rag_date_filter_mode
        self._date_refresh = settings.rag_date_filter_refresh_s
        self._date_retry = settings.rag_date_filter_retry_s
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
        # When a check passed after such a failure, until a search is served: a search that fails
        # again meanwhile is remembered like a failed check (searches hang while the check answers).
        self._passed_at: float | None = None
        self._from_search = False  # the remembered failure is such a search, not a check
        # A search was refused as a request: the next call checks the collection before it searches
        # (a collection recreated with another vector size refuses every search). Refuses nobody.
        self._recheck = False
        # The class of the last unexpected search error logged at ERROR: one line per new class, not per
        # call, until a search is served again.
        self._unexpected_logged: str | None = None
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
                else:
                    # After an outage the read could only add a request timeout to the start: the first
                    # date-bounded call makes it instead.
                    if self._date_mode == "auto":
                        self._date_filter_mode()  # never raises: the engine falls back by itself

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
                    self._passed_at, self._from_search = None, False
                if changed:  # once per change of state, not once per call
                    log.warning("rag: collection check failed (%s, %s); rag_search answers [rag_unavailable] "
                                "until a collection check passes", reason, type(exc).__name__)
                return reason
            with self._lock:
                doubted = self._suspected is not None or self._from_search
                back = ((self._collection_reason is not None and not self._from_search)
                        or self._suspected == "collection_missing")
                if back:
                    # Back after a failure: the collection may have been recreated with other codes.
                    self._values = self._values_due_at = self._values_tried_at = None
                    self._logged = None
                # After a failed search the check proves little: `_logged` is kept until a search is
                # served, so a search that keeps failing is logged once, not once per check.
                self._collection_reason = self._failed_at = self._suspected = None
                self._passed_at, self._from_search = (monotonic() if doubted else None), False
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
                self._logged = (what, reason)
                if self._passed_at is not None:
                    # The check passed and nothing was served since: the check answers, searches do
                    # not. Remembered like a failed check, so one call per retry period pays for it.
                    self._collection_reason, self._failed_at = reason, monotonic()
                    self._suspected, self._passed_at, self._from_search = None, None, True
                else:
                    self._suspected = reason
        if changed:  # once per change of state, not once per call
            log.warning("rag: %s (%s, %s); the next call checks the collection before it searches",
                        what, reason, type(exc).__name__)

    def _served(self) -> None:
        """A search succeeded: Qdrant answers, whatever a search that failed meanwhile recorded."""
        with self._lock:
            self._unexpected_logged = None
            if self._collection_reason is None or self._from_search:
                # A failure remembered from a search (not from a check) is disproved the same way.
                self._collection_reason = self._failed_at = None
                self._suspected = self._logged = self._passed_at = None
                self._from_search = False

    def _require_collection(self) -> None:
        """Return when searches are served; `_Outage` otherwise.

        After a failed check, the check is run again by one call at a time, and not before
        `rag_check_retry_s` seconds: the other calls are answered at once with the recorded reason.
        After a failed search (or read of the values), the first call to come runs the check at
        once; the calls that arrive while it runs are answered at once with the reason of that failure.
        After a refused search the first call to come runs the check too, and the others are served."""
        with self._lock:
            reason = self._collection_reason or self._suspected
            if reason is None and (not self._recheck or self._checking):
                return
            # `_failed_at` is set by a failed check, or by a search that failed right after a check
            # passed; it is cleared by the check that passes.
            recent = self._failed_at is not None and monotonic() - self._failed_at < self._check_retry_s
            if self._checking or recent:
                raise _Outage(reason)
            self._checking, self._recheck = True, False
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

    # -- date bounds -------------------------------------------------------------------------

    def _date_filter_mode(self) -> str:
        """`interval` or `legacy` for `auto` mode: the engine's own reading of the collection, at most
        one request (none while its answer is fresh or a failed read is recent). It never raises for
        Qdrant: a failed read gives the last answer, or `legacy`, and is the engine's to log (one
        warning, written without the error's text by `EngineLogFilter`)."""
        uses = self._store.uses_date_intervals(self._collection, max_age=self._date_refresh,
                                               retry_after=self._date_retry)
        return "interval" if uses else "legacy"

    # -- search ------------------------------------------------------------------------------

    def search(self, *, query: str, filters: dict, limit: int, max_limit: int,
               deadline: float | None = None) -> tuple[dict, list[dict]]:
        """(filters as applied, hits as plain dicts, best first). Blocking: call it off the event loop.

        `deadline` (a `time.monotonic()` value) is when the caller stops waiting: the search is not
        started after it.

        Raises the tool errors itself: `[invalid_argument]` with the engine's own message for an
        argument the engine refuses or a filter value the collection does not hold,
        `[rag_unavailable]` with a reason for an outage or for a search Qdrant refuses (`search_refused`:
        the error of this call alone; the next call checks the collection before it searches), and
        `internal_error` for any other exception, whose class only is logged (its text and traceback at
        DEBUG level): nothing of an exception reaches the server's masking.
        """
        error = None
        try:
            applied = SearchFilters.from_mapping(filters)
            if self._model_reason is not None:
                raise _Outage(self._model_reason)
            self._require_collection()
            self._check_values(applied, deadline)
            mode = self._date_mode
            if mode == "auto" and (applied.date_from is not None or applied.date_to is not None):
                mode = self._date_filter_mode()  # before the deadline check: no search after it
            if deadline is not None and time.monotonic() >= deadline:
                raise _Outage("timeout")
            hits = search(query, collection=self._collection, store=self._store, embedder=self._embedder,
                          filters=applied, limit=limit, max_limit=max_limit, date_filter_mode=mode,
                          date_filter_refresh=self._date_refresh, date_filter_retry=self._date_retry)
        except SearchError as exc:
            error = tool_error(INVALID_ARGUMENT, str(exc))
        except _Outage as exc:
            error = unavailable(exc.reason)
        except ToolError:
            raise  # a coded error of the toolset's own (a filter value the collection does not hold)
        except Exception as exc:  # mapped below, never re-raised
            if _refused(exc):
                # The request is refused, not the service: the other calls are served. The next one
                # checks the collection first, in case it is the collection that changed.
                with self._lock:
                    self._recheck = True
                status = getattr(exc, "status_code", None) or getattr(exc.__cause__, "status_code", 400)
                log.info("rag: a search was refused by the vector database (HTTP %d%s)", status,
                         ", a filtered field has no index" if isinstance(exc, MissingIndexError) else "")
                error = unavailable("search_refused")
            else:
                reason = _reason(exc)
                if reason is None:
                    # Not raised on: FastMCP would log the text and the traceback of the exception, which
                    # may hold the host, the key or a response body. A developer gets them at DEBUG level,
                    # which the CLI never turns on.
                    with self._lock:
                        first = self._unexpected_logged != type(exc).__name__
                        self._unexpected_logged = type(exc).__name__
                    log.log(logging.ERROR if first else logging.DEBUG,
                            "rag: a search failed with an unexpected error (%s); rag_search answered "
                            "[rag_unavailable] internal_error. Its text and traceback, which may hold the "
                            "vector database's address or a response body, are logged at DEBUG level only",
                            type(exc).__name__)
                    log.debug("rag: the unexpected error of a search", exc_info=exc)
                    error = unavailable("internal_error")
                else:
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
             "date_to": applied.date_to.isoformat() if applied.date_to else None,
             "include_undated": applied.include_undated},
            # The engine's hit has no field for these two: they are read from the stored payload.
            [{**hit.to_dict(), "date_precision": hit.payload.get("date_precision"),
              "date_kind": hit.payload.get("date_kind")} for hit in hits],
        )


def _list(values: tuple[str, ...] | None) -> list[str] | None:
    return None if values is None else list(values)
