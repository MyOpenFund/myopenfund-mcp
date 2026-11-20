"""The installed vault-query is the pinned one: what this server relies on exists.

A pin bump to an older vault, or a vault-query that drops a name, fails here and not in a deployment.
"""
import inspect

import pytest

vault_query = pytest.importorskip("vault_query")


def test_runs_status_rows_carry_bad_content():  # 0.4.0
    from vault_query import health

    assert "bad_content" in health.RUN_COLUMNS


def test_search_rows_carry_document_url():  # 0.3.0 (was pdf_url)
    assert "document_url" in vault_query.SEARCH_COLUMNS and "pdf_url" not in vault_query.ALL_COLUMNS


def test_documents_contract_api():  # 0.5.0
    assert {"date_precision", "date_kind", "entity_scheme", "entity_value"} <= vault_query.ALL_COLUMNS
    assert list(inspect.signature(vault_query.get_document).parameters) == ["conn", "doc_id", "corpus"]
    assert issubclass(vault_query.DocTypeNeedsCorpus, vault_query.InvalidQuery)
    vocabulary = inspect.signature(vault_query.list_vocabulary).parameters
    assert {"corpus", "kind", "limit"} <= set(vocabulary)
    assert vault_query.VOCABULARY_KINDS == {"source_code", "doc_type", "provenance", "entity_scheme"}


class _Cursor:
    def __init__(self, rows):
        self.rows, self.executed = rows, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, rows):
        self.cur = _Cursor(rows)

    def cursor(self, **kwargs):
        return self.cur


def test_get_document_returns_a_list_and_filters_on_corpus():
    rows = [{"doc_id": "d", "corpus": "central-bank"}, {"doc_id": "d", "corpus": "company"}]
    conn = _Conn(rows)
    assert vault_query.get_document(conn, "d") == rows
    one = _Conn(rows[1:])
    assert vault_query.get_document(one, "d", corpus="company") == rows[1:]
    assert one.cur.executed[0][1] == ("d", "company")
    assert vault_query.get_document(_Conn([]), "nope") == []
