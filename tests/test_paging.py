import base64
import datetime as dt
import ipaddress
import json
import uuid
from collections import OrderedDict
from decimal import Decimal

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import default_serializer

from myopenfund_mcp.config import Settings
from myopenfund_mcp.errors import INVALID_ARGUMENT, tool_error
from myopenfund_mcp.paging import (
    MAX_OFFSET,
    cap_result,
    clamp_limit,
    decode_cursor,
    encode_cursor,
    jsonable,
    result_chars,
)

S = Settings()


def _b64(obj) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode()


def test_tool_error_puts_the_code_first():
    err = tool_error(INVALID_ARGUMENT, "bad")
    assert isinstance(err, ToolError)
    assert str(err) == "[invalid_argument] bad"


def test_cursor_roundtrip():
    for offset in (0, 1, 150, MAX_OFFSET):
        assert decode_cursor(encode_cursor(offset)) == offset
    assert decode_cursor(None) == 0


def test_max_offset_matches_vault_query_bound():
    assert MAX_OFFSET == 2**31 - 1


@pytest.mark.parametrize("bad", [
    "",                                   # empty string is not "no cursor"
    "garbage",                            # not base64 of JSON
    "eyJvIjoiMSJ9",                       # {"o":"1"}: string offset
    _b64({"o": -1}),                      # negative
    _b64({"o": True}),                    # bool is an int subclass
    _b64({"o": 1.0}),                     # float
    _b64({"o": None}),
    _b64({"x": 1}),                       # wrong key
    _b64([1]),                            # not an object
    _b64("o"),
    _b64({"o": 1, "extra": 2}),           # not the shape this server issues
    _b64({"o": MAX_OFFSET + 1}),          # beyond the int4 bound
    base64.urlsafe_b64encode(b'{"o": ' + b"9" * 5000 + b"}").decode(),  # beyond int-str digit limit
    encode_cursor(150).rstrip("="),       # padding stripped
    encode_cursor(150) + "\n",            # trailing junk the decoder would skip
    encode_cursor(150) + "é",             # non-ASCII
    "é" + encode_cursor(150),
    base64.urlsafe_b64encode(b"\xff\xfe").decode(),  # not UTF-8
    base64.urlsafe_b64encode(b'{"o":150}').decode(),    # same value, not the bytes this server issues
])
def test_cursor_tampering_refused(bad):
    with pytest.raises(ToolError, match=r"^\[invalid_argument\] cursor"):
        decode_cursor(bad)


def test_cursor_error_tells_the_caller_what_to_do():
    with pytest.raises(ToolError) as exc:
        decode_cursor("garbage")
    assert str(exc.value) == (
        "[invalid_argument] cursor is not one this server issued; pass next_cursor from the "
        "previous page unchanged, or omit it to start over")


def test_encode_cursor_refuses_out_of_range():
    for bad in (-1, MAX_OFFSET + 1, True):
        with pytest.raises(ValueError):
            encode_cursor(bad)


def test_clamp_limit():
    assert clamp_limit(None, S) == 50
    assert clamp_limit(1, S) == 1
    assert clamp_limit(200, S) == 200
    for bad in (0, -1, 201, True, False, 50.0, "50"):
        with pytest.raises(ToolError, match=r"^\[invalid_argument\] limit"):
            clamp_limit(bad, S)


def test_clamp_limit_follows_settings():
    s = Settings(default_rows=5, max_rows=10)
    assert clamp_limit(None, s) == 5
    assert clamp_limit(10, s) == 10
    with pytest.raises(ToolError, match="between 1 and 10"):
        clamp_limit(11, s)


def test_jsonable():
    assert jsonable({"d": dt.date(2020, 1, 2), "t": dt.datetime(2020, 1, 2, 3, 4, tzinfo=dt.UTC),
                     "x": Decimal("1.5"), "l": [dt.date(2021, 1, 1)]}) == {
        "d": "2020-01-02", "t": "2020-01-02T03:04:00+00:00", "x": 1.5, "l": ["2021-01-01"]}


def test_jsonable_naive_datetime_keeps_no_offset():
    assert jsonable(dt.datetime(2020, 1, 2, 3, 4, 5)) == "2020-01-02T03:04:05"


def test_jsonable_nested_tuples_become_lists():
    assert jsonable({"a": ((1, Decimal("2.5")), [(dt.date(2020, 1, 1),)])}) == {
        "a": [[1, 2.5], [["2020-01-01"]]]}


def test_jsonable_non_finite_numbers_become_null():
    assert jsonable([Decimal("NaN"), float("inf"), Decimal("-Infinity")]) == [None, None, None]


def test_jsonable_leaves_plain_values_alone():
    v = {"s": "é", "i": 3, "f": 0.25, "b": True, "n": None}
    assert jsonable(v) == v


def test_result_chars_is_what_fastmcp_sends():
    payload = jsonable({"rows": [{"title": "Société Exemple — rapport", "n": 1, "x": 1.5, "ok": None,
                                  "d": dt.date(2020, 1, 2), "nested": {"q": 'quote " and \\ and \n'}}],
                        "next_cursor": encode_cursor(50)})
    assert result_chars(payload) == len(default_serializer(payload))


def test_cap_result_refuses_never_truncates():
    with pytest.raises(ToolError, match=r"^\[result_too_large\]"):
        cap_result({"blob": "x" * 300_000}, S)
    small = {"d": dt.date(2020, 1, 2)}
    assert cap_result(small, S) == {"d": "2020-01-02"}


def test_cap_result_boundary_counts_compact_json():
    s = Settings(max_result_chars=1_000)
    overhead = len('{"b":""}')
    exact = {"b": "x" * (1_000 - overhead)}
    assert cap_result(exact, s) == exact
    with pytest.raises(ToolError, match=r"^\[result_too_large\] result is 1001 characters"):
        cap_result({"b": "x" * (1_001 - overhead)}, s)


def test_cap_result_counts_characters_not_escapes():
    s = Settings(max_result_chars=1_000)
    overhead = len('{"b":""}')
    payload = {"b": "é" * (1_000 - overhead)}  # would be 6x larger with \\u escapes
    assert cap_result(payload, s) == payload


def test_cap_result_hint_is_contextual():
    s = Settings(max_result_chars=10)
    with pytest.raises(ToolError, match=r"; ask for a smaller limit$"):
        cap_result({"b": "x" * 50}, s)
    with pytest.raises(ToolError, match=r"^\[result_too_large\] result is \d+ characters \(cap 10\); "
                       r"read the document by section$"):
        cap_result({"b": "x" * 50}, s, hint="read the document by section")


def test_clamp_limit_refuses_rather_than_clamps():
    assert "refuse" in clamp_limit.__doc__.lower()
    with pytest.raises(ToolError):
        clamp_limit(S.max_rows + 1, S)


def test_jsonable_uuid_and_other_types_become_strings():
    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert jsonable({"u": u, "ip": ipaddress.ip_address("10.0.0.1")}) == {
        "u": "12345678-1234-5678-1234-567812345678", "ip": "10.0.0.1"}


def test_jsonable_time_and_timedelta():
    assert jsonable(dt.time(3, 4, 5)) == "03:04:05"
    assert jsonable(dt.timedelta(days=1, seconds=1, microseconds=500_000)) == 86401.5


def test_jsonable_binary_becomes_base64():
    expected = base64.b64encode(b"\x00\xffab").decode("ascii")
    assert jsonable(memoryview(b"\x00\xffab")) == expected
    assert jsonable(b"\x00\xffab") == expected
    assert jsonable(bytearray(b"\x00\xffab")) == expected


def test_jsonable_integral_decimal_becomes_int():
    out = jsonable([Decimal(12345), Decimal("1.2E+3"), Decimal("1.5"), Decimal("-0")])
    assert out == [12345, 1200, 1.5, 0]
    assert [type(v) for v in out] == [int, int, float, int]


def test_jsonable_mapping_subclasses_become_plain_dicts():
    out = jsonable(OrderedDict([("a", OrderedDict([("b", Decimal(2))]))]))
    assert type(out) is dict and type(out["a"]) is dict
    assert out == {"a": {"b": 2}}


def test_jsonable_output_is_plain_json():
    payload = {"u": uuid.uuid4(), "b": b"x", "t": dt.time(1), "td": dt.timedelta(1), "o": object()}
    json.dumps(jsonable(payload), allow_nan=False)


@pytest.mark.anyio
async def test_result_chars_matches_a_real_fastmcp_tool_on_the_wire():
    payload = jsonable({
        "rows": [{"title": "Société Exemple — résumé « été »", "d": dt.date(2020, 1, 2),
                  "t": dt.datetime(2020, 1, 2, 3, 4, 5, tzinfo=dt.UTC), "x": Decimal(12345),
                  "y": Decimal("1.25"), "nested": [[1, [2.5, None]], {"q": 'quote " \\ \n'}]}],
        "next_cursor": encode_cursor(50),
    })
    mcp = FastMCP("t")

    @mcp.tool
    def probe() -> dict:
        return payload

    async with Client(mcp) as c:
        res = await c.call_tool("probe", {})
    assert len(res.content[0].text) == result_chars(payload)
