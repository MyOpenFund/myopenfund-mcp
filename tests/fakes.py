"""An in-memory VaultReader with adversarial data.

NULL date, an html row, a tombstone, a same-date pair, a run that crashed before any source
(run grain only), an error-only source, an approximate `last_seen_at`, Decimal and naive values;
TWINS: one doc_id held by two corpora (a doc_id is unique within its corpus only).
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

# The same doc_id in two corpora, listed out of corpus order (vault-query orders by corpus).
TWINS = [
    {"doc_id": "twin", "corpus": "company", "source_code": "sec", "doc_type": "A1", "title": "Form 10-K",
     "date": dt.date(2024, 2, 2), "year": 2024, "mime_type": "text/plain", "deleted_at": None,
     "date_precision": "day", "date_kind": "publication", "entity_scheme": "cik", "entity_value": "0000320193"},
    {"doc_id": "twin", "corpus": "central-bank", "source_code": "us", "doc_type": "A1", "title": "Rate decision",
     "date": dt.date(2024, 1, 1), "year": 2024, "mime_type": "application/pdf", "deleted_at": None,
     "date_precision": "year", "date_kind": None, "entity_scheme": None, "entity_value": None},
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


def health_row(source_code: str, doc_type: str = "C1", *, cadence_status: str | None = "on-track",
               days_late: int | None = 0, corpus: str = "central-bank") -> dict:
    """One row of the vault's `source_health` view, with its 28 columns."""
    return {
        "corpus": corpus, "source_code": source_code, "doc_type": doc_type,
        "expected_interval_days": Decimal("45.5"), "expected_per_year": Decimal("8.0"), "n_3y": 24,
        "last_document_date": dt.date(2010, 1, 13), "next_expected": dt.date(2010, 2, 27),
        "days_until": None if days_late is None else -days_late, "days_late": days_late,
        "cadence_status": cadence_status, "cadence_updated_at": dt.datetime(2026, 11, 3, 4, 0, tzinfo=dt.UTC),
        "last_run_id": "2026-11-03T04:00:00Z-cb-refresh-abcdef",
        "last_run_at": dt.datetime(2026, 11, 3, 4, 12, tzinfo=dt.UTC), "last_run_outcome": "degraded",
        "last_run_truncated": False, "source_runs_7d": 7, "source_degraded_runs_7d": 1,
        "source_truncated_runs_7d": 0, "source_zero_yield_runs_7d": 6, "source_docs_new_7d": 1,
        "source_docs_seen_7d": 4123, "source_fetch_errors_7d": 2, "source_runs_90d": 90,
        "source_docs_new_median_per_run_90d": Decimal("NaN"), "source_open_discovery_errors": 3,
        "source_open_discovery_attempts": 12,
        "source_last_discovery_error_at": dt.datetime(2026, 11, 2, 4, 3, tzinfo=dt.UTC),
    }


HEALTH = [health_row("us", cadence_status="overdue", days_late=33)]
# The production vault's size (79 series, 2026-11-04): about 870 characters a row, the whole list
# about 66,000 characters, too large for a client to read inline.
MANY_HEALTH = [health_row(f"s{i:02d}") for i in range(79)]


class FakeVaultReader:
    def __init__(self, docs=DOCS, *, health=HEALTH, fail_with: Exception | None = None):
        self.docs, self.health, self.calls, self.fail_with = list(docs), list(health), [], fail_with

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

    def get_document(self, doc_id, corpus=None):
        self.calls.append(("get", doc_id, corpus))
        self._maybe_fail()
        rows = [d for d in self.docs if d["doc_id"] == doc_id and corpus in (None, d["corpus"])]
        return sorted(rows, key=lambda d: d["corpus"])

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
        return [h for h in self.health if source_code in (None, h["source_code"])]

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
