"""PostgresVaultReader against the real driver and query package: only connect() and the
vault_query functions are replaced, so the except clauses meet the real exception hierarchy."""
import inspect

import psycopg2
import psycopg2.errors
import pytest
import vault_query
from fastmcp.exceptions import ToolError

from myopenfund_mcp.vault.reader import PostgresVaultReader

URL = "postgresql://u:hunter2@h/db"


class FakeConn:
    def __init__(self):
        self.autocommit = False
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def pg(monkeypatch):
    """The real psycopg2 module with connect() recorded, failing on demand."""
    state = {"calls": [], "conns": [], "exc": None}

    def connect(*args, **kwargs):
        state["calls"].append((args, kwargs))
        if state["exc"] is not None:
            raise state["exc"]
        conn = FakeConn()
        state["conns"].append(conn)
        return conn

    monkeypatch.setattr(psycopg2, "connect", connect)
    return state


def _reader(url=URL, **kwargs):
    return PostgresVaultReader(url, query_timeout_s=1, **kwargs)


def _assert_clean(info, code):
    assert str(info.value).startswith(f"[{code}]")
    assert "hunter2" not in str(info.value)
    assert info.value.__context__ is None
    assert info.value.__cause__ is None


@pytest.mark.parametrize("exc,code", [
    (psycopg2.errors.QueryCanceled("canceling statement due to statement timeout"), "timeout"),
    (psycopg2.errors.InvalidDatetimeFormat('invalid input syntax for type date: "hunter2"'),
     "invalid_argument"),
    (psycopg2.ProgrammingError('invalid dsn: missing "=" after "postgresql://u:hunter2@h/db"'),
     "backend_unavailable"),
    (psycopg2.OperationalError('FATAL: password authentication failed for "u" hunter2'),
     "backend_unavailable"),
])
def test_driver_errors_become_codes_and_never_leak(pg, exc, code):
    pg["exc"] = exc
    with pytest.raises(ToolError) as info:
        _reader().freshness()
    _assert_clean(info, code)


def test_timeout_message_names_the_setting_and_the_migration_case(pg):
    pg["exc"] = psycopg2.errors.QueryCanceled("canceling statement")
    with pytest.raises(ToolError) as info:
        _reader().freshness()
    assert str(info.value) == (
        "[timeout] the vault query exceeded MOF_MCP_QUERY_TIMEOUT_S (slow query, or the vault is being "
        "migrated); retry later or narrow the filters"
    )


def test_database_refused_value_is_invalid_argument_with_a_fixed_text(pg):
    pg["exc"] = psycopg2.errors.InvalidDatetimeFormat('invalid input syntax for type date: "hunter2"')
    with pytest.raises(ToolError) as info:
        _reader().freshness()
    assert str(info.value) == "[invalid_argument] a value was refused by the database; check its format"


def test_invalid_dsn_raised_while_parsing_never_leaks(pg):
    # A malformed DSN fails in the driver's own DSN parser, before connect().
    with pytest.raises(ToolError) as info:
        _reader("u:hunter2 garbage").freshness()
    _assert_clean(info, "backend_unavailable")


def test_invalid_query_becomes_invalid_argument_and_closes_the_connection(pg, monkeypatch):
    def recent_runs(conn, **kwargs):
        raise vault_query.InvalidQuery("sort_by must be one of [...]")

    monkeypatch.setattr(vault_query, "recent_runs", recent_runs)
    with pytest.raises(ToolError) as info:
        _reader().recent_runs(corpus=None, since="yesterday", limit=5)
    assert str(info.value) == "[invalid_argument] sort_by must be one of [...]"
    assert info.value.__context__ is None
    assert pg["conns"][0].closed


def test_query_failure_closes_the_connection(pg, monkeypatch):
    def freshness(conn):
        raise psycopg2.errors.QueryCanceled("canceling statement")

    monkeypatch.setattr(vault_query, "freshness", freshness)
    with pytest.raises(ToolError) as info:
        _reader().freshness()
    _assert_clean(info, "timeout")
    assert pg["conns"][0].closed


def test_connection_is_autocommit_read_only_and_time_boxed(pg, monkeypatch):
    seen = {}

    def freshness(conn):
        seen["autocommit"] = conn.autocommit
        return {"live": 1}

    monkeypatch.setattr(vault_query, "freshness", freshness)
    assert _reader(connect_timeout_s=7).freshness() == {"live": 1}
    assert seen["autocommit"] is True
    (args, kwargs), = pg["calls"]
    assert args == (URL,)
    assert kwargs["application_name"] == "myopenfund-mcp"
    assert kwargs["connect_timeout"] == 7
    assert "-c statement_timeout=1000" in kwargs["options"]
    assert "-c default_transaction_read_only=on" in kwargs["options"]
    assert "-c timezone=UTC" in kwargs["options"]
    assert pg["conns"][0].closed


def test_connect_timeout_defaults_to_ten(pg, monkeypatch):
    monkeypatch.setattr(vault_query, "freshness", lambda conn: {})
    _reader().freshness()
    assert pg["calls"][0][1]["connect_timeout"] == 10


def test_dsn_options_are_kept_and_ours_come_last(pg, monkeypatch):
    monkeypatch.setattr(vault_query, "freshness", lambda conn: {})
    _reader("postgresql://u:p@h/db?options=-c%20search_path%3Dfoo").freshness()
    options = pg["calls"][0][1]["options"]
    assert options.startswith("-c search_path=foo ")
    # Ours appended: for the same GUC, the last -c wins, so the DSN cannot lift the read-only guard.
    assert options.endswith("-c statement_timeout=1000 -c default_transaction_read_only=on")


def test_dsn_options_cannot_override_ours(pg, monkeypatch):
    monkeypatch.setattr(vault_query, "freshness", lambda conn: {})
    _reader("postgresql://u:p@h/db?options=-c%20default_transaction_read_only%3Doff").freshness()
    options = pg["calls"][0][1]["options"]
    assert options.rindex("default_transaction_read_only=on") > options.index("default_transaction_read_only=off")


def test_session_timezone_is_utc_and_wins_over_the_dsn(pg, monkeypatch):
    monkeypatch.setattr(vault_query, "freshness", lambda conn: {})
    _reader("postgresql://u:p@h/db?options=-c%20timezone%3DEurope/Paris").freshness()
    options = pg["calls"][0][1]["options"]
    assert options.startswith("-c timezone=Europe/Paris ")
    assert options.rindex("timezone=UTC") > options.index("timezone=Europe/Paris")


# Each reader method, its call, and the vault_query call it must make (args after conn, kwargs).
PASS_THROUGH = [
    ("search_documents",
     {"filters": {"corpus": "amf"}, "sort_by": "date", "sort_dir": "asc", "limit": 9, "offset": 3, "include_deleted": True},
     (({"corpus": "amf"},), {"sort_by": "date", "sort_dir": "asc", "limit": 9, "offset": 3, "include_deleted": True,
                                 "columns": vault_query.SEARCH_COLUMNS})),
    ("get_document", {"doc_id": "d1"}, (("d1",), {})),
    ("corpus_stats", {"group_by": "year", "corpus": "amf", "limit": 12},
     (("year",), {"corpus": "amf", "limit": 12})),
    ("runs_status",
     {"corpus": "amf", "source_code": "s", "since": "2026-10-01T00:00:00+00:00", "limit": 7,
          "max_error_samples": 3, "max_text_chars": 100},
     ((), {"corpus": "amf", "source_code": "s", "since": "2026-10-01T00:00:00+00:00", "limit": 7,
               "max_error_samples": 3, "max_text_chars": 100})),
    ("recent_runs", {"corpus": "amf", "since": None, "limit": 2}, ((), {"corpus": "amf", "since": None, "limit": 2})),
    ("source_health", {"corpus": "amf", "source_code": None}, ((), {"corpus": "amf", "source_code": None})),
    ("open_discovery_errors", {"corpus": "amf", "source_code": None, "limit": 4, "max_text_chars": 50},
     ((), {"corpus": "amf", "source_code": None, "limit": 4, "max_text_chars": 50})),
    ("freshness", {}, ((), {})),
]


def test_pass_through_covers_every_reader_method():
    from myopenfund_mcp.vault.reader import VaultReader

    protocol = {n for n, v in vars(VaultReader).items() if callable(v) and not n.startswith("_")}
    assert {name for name, _, _ in PASS_THROUGH} == protocol


@pytest.mark.parametrize("name,call,expected", PASS_THROUGH, ids=[p[0] for p in PASS_THROUGH])
def test_arguments_pass_through_unchanged(pg, monkeypatch, name, call, expected):
    real = getattr(vault_query, name)
    got = {}

    def fake(conn, *args, **kwargs):
        # The real signature must accept the call as made (positional filters / group_by included).
        inspect.signature(real).bind(conn, *args, **kwargs)
        got["conn"], got["args"], got["kwargs"] = conn, args, kwargs
        return "result"

    monkeypatch.setattr(vault_query, name, fake)
    assert getattr(_reader(), name)(**call) == "result"
    assert got["conn"] is pg["conns"][0]
    assert (got["args"], got["kwargs"]) == expected
    assert pg["conns"][0].closed
