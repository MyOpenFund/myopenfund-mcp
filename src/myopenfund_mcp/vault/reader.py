"""The vault toolset's only dependency: read functions, behind a protocol.

PostgresVaultReader runs the `vault_query` package (the vault repo's one
implementation) on a short-lived connection that is read-only and time-boxed at
the server: a runaway query is cancelled by Postgres, not by hoping.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Protocol

from ..errors import BACKEND_UNAVAILABLE, INVALID_ARGUMENT, TIMEOUT, tool_error


class VaultReader(Protocol):
    def search_documents(self, *, filters: dict, sort_by: str, sort_dir: str, limit: int,
                         offset: int, include_deleted: bool) -> tuple[int, list[dict]]: ...
    def get_document(self, doc_id: str) -> dict | None: ...
    def corpus_stats(self, *, group_by: str, corpus: str | None, limit: int) -> dict: ...
    def runs_status(self, *, corpus: str | None, source_code: str | None, since: Any, limit: int,
                    max_error_samples: int, max_text_chars: int) -> list[dict]: ...
    def recent_runs(self, *, corpus: str | None, since: Any, limit: int) -> list[dict]: ...
    def source_health(self, *, corpus: str | None, source_code: str | None) -> list[dict]: ...
    def open_discovery_errors(self, *, corpus: str | None, source_code: str | None, limit: int,
                              max_text_chars: int) -> list[dict]: ...
    def freshness(self) -> dict: ...


class PostgresVaultReader:
    def __init__(self, database_url: str, *, query_timeout_s: int, connect_timeout_s: int = 10):
        import psycopg2  # optional extra [vault]: imported only when the toolset is enabled
        import psycopg2.extensions
        import vault_query

        self._pg, self._vq = psycopg2, vault_query
        self._url, self._timeout_ms = database_url, query_timeout_s * 1000
        self._connect_timeout_s = connect_timeout_s

    def _options(self) -> str:
        # The DSN's own options (e.g. search_path) are kept; ours are appended so that, for the same GUC,
        # ours win (the last -c wins): a DSN cannot lift the read-only guard or the statement timeout.
        # Parsed per call, inside _run: a malformed DSN raises a driver error whose text holds the DSN.
        theirs = self._pg.extensions.parse_dsn(self._url).get("options", "")
        # timezone=UTC: timestamptz values come back in UTC, whatever the server or role default.
        ours = f"-c timezone=UTC -c statement_timeout={self._timeout_ms} -c default_transaction_read_only=on"
        return f"{theirs} {ours}" if theirs.strip() else ours

    @contextmanager
    def _connection(self):
        conn = self._pg.connect(
            self._url,
            connect_timeout=self._connect_timeout_s,
            application_name="myopenfund-mcp",
            options=self._options(),
        )
        try:
            # Autocommit (review A2): a read on a non-autocommit connection leaves it "idle in transaction",
            # holding ACCESS SHARE locks that block the vault's DDL train. Short-lived + autocommit = no hazard.
            conn.autocommit = True
            yield conn
        finally:
            conn.close()

    def _run(self, fn, *args, **kwargs):
        error = None
        try:
            with self._connection() as conn:
                return fn(conn, *args, **kwargs)
        except self._vq.InvalidQuery as exc:
            error = tool_error(INVALID_ARGUMENT, str(exc))
        except self._pg.errors.QueryCanceled:
            error = tool_error(TIMEOUT, "the vault query exceeded MOF_MCP_QUERY_TIMEOUT_S (slow query, or the "
                                        "vault is being migrated); retry later or narrow the filters")
        except self._pg.DataError:
            # Fixed text: the driver's message can echo the refused value.
            error = tool_error(INVALID_ARGUMENT, "a value was refused by the database; check its format")
        except self._pg.Error:
            # Never the driver's text: a connection error can carry DSN fragments (vault api/db.py::_redact).
            error = tool_error(BACKEND_UNAVAILABLE, "the vault database is unreachable or refused the query")
        # Raised outside the handler so the driver error (and its DSN-holding frames) is not chained.
        raise error

    def search_documents(self, *, filters, sort_by, sort_dir, limit, offset, include_deleted):
        return self._run(self._vq.search_documents, filters, sort_by=sort_by, sort_dir=sort_dir,
                         limit=limit, offset=offset, include_deleted=include_deleted,
                         columns=self._vq.SEARCH_COLUMNS)

    def get_document(self, doc_id):
        return self._run(self._vq.get_document, doc_id)

    def corpus_stats(self, *, group_by, corpus, limit):
        return self._run(self._vq.corpus_stats, group_by, corpus=corpus, limit=limit)

    def runs_status(self, *, corpus, source_code, since, limit, max_error_samples, max_text_chars):
        return self._run(self._vq.runs_status, corpus=corpus, source_code=source_code, since=since,
                         limit=limit, max_error_samples=max_error_samples, max_text_chars=max_text_chars)

    def recent_runs(self, *, corpus, since, limit):
        return self._run(self._vq.recent_runs, corpus=corpus, since=since, limit=limit)

    def source_health(self, *, corpus, source_code):
        return self._run(self._vq.source_health, corpus=corpus, source_code=source_code)

    def open_discovery_errors(self, *, corpus, source_code, limit, max_text_chars):
        return self._run(self._vq.open_discovery_errors, corpus=corpus, source_code=source_code,
                         limit=limit, max_text_chars=max_text_chars)

    def freshness(self):
        return self._run(self._vq.freshness)
