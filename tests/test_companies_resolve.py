from myopenfund_mcp.companies.resolve import index, resolve

E = [
    {"entity_id": "e1", "subject_value": "0000320193", "id_scheme": "cik", "id_value": "0000320193",
     "name": "APPLE COMPUTER INC", "country": None, "valid_date": "1994-01-26"},
    {"entity_id": "e1", "subject_value": "0000320193", "id_scheme": "cik", "id_value": "0000320193",
     "name": "Apple Inc.", "country": "US", "valid_date": "2019-08-05"},
    {"entity_id": "e1", "subject_value": "0000320193", "id_scheme": "ticker", "id_value": "AAPL",
     "name": None, "country": None, "valid_date": "2026-11-02"},
    {"entity_id": "e2", "subject_value": "0000999999", "id_scheme": "cik", "id_value": "0000999999",
     "name": "APPLE HOSPITALITY REIT INC", "country": "US", "valid_date": "2015-01-01"},
]


def test_index_groups_names_and_tickers():
    i = index(E)["0000320193"]
    assert i["current_name"] == "Apple Inc." and i["tickers"] == ["AAPL"]
    assert [n["name"] for n in i["names"]] == ["APPLE COMPUTER INC", "Apple Inc."]
    assert i["entity_id"] == "e1" and i["country"] == "US"


def test_index_is_order_independent():
    assert index(list(reversed(E))) == index(E)


def test_cik_any_padding_ticker_and_entity_id():
    i = index(E)
    assert [r["cik"] for r in resolve(i, "320193")] == ["0000320193"]
    assert [r["cik"] for r in resolve(i, " 0000320193 ")] == ["0000320193"]
    assert [r["cik"] for r in resolve(i, "000000320193")] == ["0000320193"]  # over-padded
    assert [r["cik"] for r in resolve(i, "aapl")] == ["0000320193"]
    assert [r["cik"] for r in resolve(i, "e2")] == ["0000999999"]


def test_name_substring_hits_former_names_and_is_ambiguous_when_it_should_be():
    i = index(E)
    # sorted by current name: "APPLE HOSPITALITY..." before "Apple Inc." (case-insensitive)
    assert [r["cik"] for r in resolve(i, "apple")] == ["0000999999", "0000320193"]
    assert [r["cik"] for r in resolve(i, "computer")] == ["0000320193"]


def test_nothing_found():
    assert resolve(index(E), "zzz") == []
    assert resolve(index(E), "   ") == []
    assert resolve(index(E), "12345678901") == []  # digits but not a known CIK, no substring fallback


def test_entity_without_any_name():
    i = index([{"entity_id": "e9", "subject_value": "0000000009", "id_scheme": "ticker",
                "id_value": "ZZ", "name": None, "country": None, "valid_date": "2026-01-01"}])
    assert i["0000000009"]["current_name"] is None and i["0000000009"]["names"] == []
    assert [r["cik"] for r in resolve(i, "zz")] == ["0000000009"]


def test_repeated_name_on_a_ticker_row_is_not_duplicated():
    rows = [*E, {"entity_id": "e1", "subject_value": "0000320193", "id_scheme": "ticker",
                 "id_value": "AAPL", "name": "Apple Inc.", "country": "US", "valid_date": "2026-11-03"}]
    i = index(rows)["0000320193"]
    assert [n["name"] for n in i["names"]] == ["APPLE COMPUTER INC", "Apple Inc."]
    assert i["names"][-1]["since"] == "2019-08-05"  # the first time the name was seen


def test_a_name_used_again_after_another_is_kept():
    rows = [{"entity_id": "e", "subject_value": "0000000001", "id_scheme": "cik", "id_value": "1",
             "name": n, "country": None, "valid_date": d}
            for n, d in (("A", "2001-01-01"), ("B", "2002-01-01"), ("A", "2003-01-01"))]
    assert [n["name"] for n in index(rows)["0000000001"]["names"]] == ["A", "B", "A"]
