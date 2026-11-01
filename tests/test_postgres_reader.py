import pytest
from fastmcp.exceptions import ToolError

from myopenfund_mcp.vault.reader import PostgresVaultReader


class FakeErrors:
    class QueryCanceled(Exception):
        pass


class FakeConn:
    def __init__(self):
        self.autocommit = False
        self.closed = False

    def close(self):
        self.closed = True


class FakePg:
    Error = Exception
    errors = FakeErrors

    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []
        self.conns = []

    def connect(self, *a, **k):
        self.calls.append((a, k))
        if self.exc is not None:
            raise self.exc
        conn = FakeConn()
        self.conns.append(conn)
        return conn


class FakeVq:
    class InvalidQuery(ValueError):
        pass

    SEARCH_COLUMNS = ("doc_id",)

    @staticmethod
    def freshness(conn):
        raise AssertionError("never reached: connect() fails first")


def _reader(exc=None, vq=FakeVq):
    r = PostgresVaultReader.__new__(PostgresVaultReader)
    r._pg, r._vq, r._url, r._timeout_ms = FakePg(exc), vq, "postgresql://u:hunter2@h/db", 1000
    return r


@pytest.mark.parametrize("exc,code", [
    (FakeErrors.QueryCanceled(), "timeout"),
    (Exception('FATAL: password authentication failed for "u" hunter2'), "backend_unavailable"),
])
def test_driver_errors_become_codes_and_never_leak(exc, code):
    with pytest.raises(ToolError) as info:
        _reader(exc).freshness()
    assert str(info.value).startswith(f"[{code}]")
    assert "hunter2" not in str(info.value)
    assert info.value.__context__ is None
    assert info.value.__cause__ is None


def test_invalid_query_becomes_invalid_argument_with_its_message():
    class Vq(FakeVq):
        @staticmethod
        def recent_runs(conn, **kwargs):
            raise FakeVq.InvalidQuery("since must be an ISO-8601 timestamp")

    r = _reader(vq=Vq)
    with pytest.raises(ToolError) as info:
        r.recent_runs(corpus=None, since="yesterday", limit=5)
    assert str(info.value) == "[invalid_argument] since must be an ISO-8601 timestamp"
    assert info.value.__context__ is None
    assert r._pg.conns[0].closed


def test_connection_is_autocommit_read_only_and_time_boxed():
    seen = {}

    class Vq(FakeVq):
        @staticmethod
        def freshness(conn):
            seen["autocommit"] = conn.autocommit
            return {"live": 1}

    r = _reader(vq=Vq)
    assert r.freshness() == {"live": 1}
    assert seen["autocommit"] is True
    (args, kwargs), = r._pg.calls
    assert args == ("postgresql://u:hunter2@h/db",)
    assert kwargs["application_name"] == "myopenfund-mcp"
    assert kwargs["connect_timeout"] == 10
    assert "-c statement_timeout=1000" in kwargs["options"]
    assert "-c default_transaction_read_only=on" in kwargs["options"]
    assert r._pg.conns[0].closed


def test_bounds_and_arguments_pass_through():
    got = {}

    class Vq(FakeVq):
        @staticmethod
        def runs_status(conn, **kwargs):
            got["runs_status"] = kwargs
            return []

        @staticmethod
        def open_discovery_errors(conn, **kwargs):
            got["open_discovery_errors"] = kwargs
            return []

        @staticmethod
        def recent_runs(conn, **kwargs):
            got["recent_runs"] = kwargs
            return []

    r = _reader(vq=Vq)
    r.runs_status(corpus="amf", source_code="s", since="2026-10-01T00:00:00+00:00", limit=7,
                  max_error_samples=3, max_text_chars=100)
    r.open_discovery_errors(corpus="amf", source_code=None, limit=4, max_text_chars=50)
    r.recent_runs(corpus="amf", since=None, limit=2)
    assert got["runs_status"] == {"corpus": "amf", "source_code": "s",
                                  "since": "2026-10-01T00:00:00+00:00", "limit": 7,
                                  "max_error_samples": 3, "max_text_chars": 100}
    assert got["open_discovery_errors"] == {"corpus": "amf", "source_code": None, "limit": 4,
                                            "max_text_chars": 50}
    assert got["recent_runs"] == {"corpus": "amf", "since": None, "limit": 2}
