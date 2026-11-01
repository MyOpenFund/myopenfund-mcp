"""An in-memory VaultReader with adversarial data.

NULL date, an html row, a tombstone, a same-date pair, a run that crashed before any source
(run grain only), an error-only source, an approximate `last_seen_at`, Decimal and naive values.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

from myopenfund_mcp.errors import INVALID_ARGUMENT, tool_error

DOCS = [
    {"doc_id": "us1", "corpus": "central-bank", "source_code": "us", "doc_type": "C1", "title": "FOMC statement",
     "date": dt.date(2010, 1, 13), "year": 2010, "mime_type": "application/pdf", "deleted_at": None},
    {"doc_id": "us2", "corpus": "central-bank", "source_code": "us", "doc_type": "C1", "title": "FOMC minutes",
     "date": dt.date(2010, 1, 13), "year": 2010, "mime_type": "application/pdf", "deleted_at": None},
    {"doc_id": "us3", "corpus": "central-bank", "source_code": "us", "doc_type": "C1", "title": "Undated",
     "date": None, "year": None, "mime_type": "application/pdf", "deleted_at": None},
    {"doc_id": "fr1", "corpus": "central-bank", "source_code": "fr", "doc_type": "A1", "title": "Décision",
     "date": dt.date(2011, 5, 1), "year": 2011, "mime_type": "text/html", "deleted_at": None},
    {"doc_id": "us9", "corpus": "central-bank", "source_code": "us", "doc_type": "C1", "title": "Withdrawn",
     "date": dt.date(2009, 3, 1), "year": 2009, "mime_type": "application/pdf",
     "deleted_at": dt.datetime(2026, 9, 1, tzinfo=dt.UTC)},
]

# Run grain: r0 crashed before reaching any source, so it has no runs_status row.
RECENT_RUNS = [
    {"run_id": "r1", "tool": "vault", "command": "ingest", "corpus": "central-bank",
     "started_at": dt.datetime(2026, 10, 2, 5, 0, tzinfo=dt.UTC),
     "finished_at": dt.datetime(2026, 10, 2, 5, 9, tzinfo=dt.UTC), "outcome": "failed", "exit_code": 1,
     "totals": {"docs_seen": 10, "docs_new": None}},
    {"run_id": "r0", "tool": "vault", "command": "ingest", "corpus": None,
     "started_at": dt.datetime(2026, 10, 1, 5, 0), "finished_at": None, "outcome": "crashed",
     "exit_code": None, "totals": None},
]

SOURCE_RUNS = [
    {"run_id": "r1", "tool": "vault", "command": "ingest", "corpus": "central-bank", "source_code": "xx",
     "started_at": dt.datetime(2026, 10, 2, 5, 0, tzinfo=dt.UTC),
     "finished_at": dt.datetime(2026, 10, 2, 5, 9, tzinfo=dt.UTC), "outcome": "failed", "exit_code": 1,
     "docs_seen": 10, "docs_new": None, "docs_failed": 3, "fetch_errors": 3, "truncated": True,
     "error_samples": ["HTTP 404 on " + "x" * 50] * 9},
]

# Error-only source "xx": no document, no health row. Its last_seen_at is an ingestion-time approximation.
ERRORS = [
    {"fingerprint": "f1", "corpus": "central-bank", "source_code": "xx", "doc_type": None, "context": "listing",
     "url": "https://example.org/x", "error_class": "HTTP404", "error": "not found " + "y" * 50,
     "http_status": 404, "first_seen_at": dt.datetime(2026, 9, 1, tzinfo=dt.UTC),
     "last_seen_at": dt.datetime(2026, 10, 2, tzinfo=dt.UTC), "seen_at_is_ingestion_time": True,
     "last_run_id": None, "occurrences": 2},
]

HEALTH = [
    {"corpus": "central-bank", "source_code": "us", "doc_type": "C1", "expected_every_days": Decimal("45.5"),
     "observed_every_days": Decimal("NaN"), "last_date": dt.date(2010, 1, 13), "status": "late"},
]


class FakeVaultReader:
    def __init__(self, docs=DOCS, *, fail_with: Exception | None = None):
        self.docs, self.calls, self.fail_with = list(docs), [], fail_with

    def _maybe_fail(self):
        if self.fail_with is not None:
            raise self.fail_with

    def search_documents(self, *, filters, sort_by, sort_dir, limit, offset, include_deleted):
        self.calls.append(("search", filters, sort_by, sort_dir, limit, offset, include_deleted))
        self._maybe_fail()
        rows = [d for d in self.docs if include_deleted or d["deleted_at"] is None]
        rows = [d for d in rows if all(d.get(k) == v for k, v in filters.items() if v is not None and k in d)]
        if filters.get("q"):
            rows = [d for d in rows if filters["q"].lower() in (d["title"] or "").lower()]
        return len(rows), rows[offset:offset + limit]

    def get_document(self, doc_id):
        self.calls.append(("get", doc_id))
        self._maybe_fail()
        return next((d for d in self.docs if d["doc_id"] == doc_id), None)

    def corpus_stats(self, *, group_by, corpus, limit):
        self.calls.append(("stats", group_by, corpus, limit))
        self._maybe_fail()
        groups: dict = {}
        live = [d for d in self.docs if d["deleted_at"] is None and corpus in (None, d["corpus"])]
        for d in live:
            groups[d.get(group_by)] = groups.get(d.get(group_by), 0) + 1
        return {"total": len(live), "group_by": group_by,
                "groups": [{"key": k, "count": v} for k, v in groups.items()][:limit]}

    def runs_status(self, *, corpus, source_code, since, limit, max_error_samples, max_text_chars):
        self.calls.append(("runs_status", corpus, source_code, since, limit, max_error_samples, max_text_chars))
        self._maybe_fail()
        rows = [r for r in SOURCE_RUNS if source_code in (None, r["source_code"])]
        return [{**r, "error_samples": [s[:max_text_chars] for s in r["error_samples"][:max_error_samples]]}
                for r in rows][:limit]

    def recent_runs(self, *, corpus, since, limit):
        self.calls.append(("recent_runs", corpus, since, limit))
        self._maybe_fail()
        return [r for r in RECENT_RUNS if corpus in (None, r["corpus"])][:limit]

    def source_health(self, *, corpus, source_code):
        self.calls.append(("health", corpus, source_code))
        self._maybe_fail()
        return [h for h in HEALTH if source_code in (None, h["source_code"])]

    def open_discovery_errors(self, *, corpus, source_code, limit, max_text_chars):
        self.calls.append(("errors", corpus, source_code, limit, max_text_chars))
        self._maybe_fail()
        rows = [e for e in ERRORS if source_code in (None, e["source_code"])]
        return [{**e, "error": e["error"][:max_text_chars]} for e in rows][:limit]

    def freshness(self):
        self._maybe_fail()
        return {"live_documents": 4, "last_ingest_seen_at": dt.datetime(2026, 10, 1, tzinfo=dt.UTC),
                "last_document_update": None, "last_run_finished_at": None}


def invalid_query_reader() -> FakeVaultReader:
    """What PostgresVaultReader raises when vault_query refuses an argument (InvalidQuery)."""
    return FakeVaultReader(fail_with=tool_error(INVALID_ARGUMENT, "unknown sort field"))
