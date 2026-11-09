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
