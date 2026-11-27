"""A real open-data-store in tmp_path, written by the real writer and publisher.

Adversarial on purpose (no perfect-world fixture):
- Apple's FY2024 revenue is restated by the FY2025 10-K (a later vintage of the same key), and
  that 10-K has no filing_statements rows (its positions are borrowed from the 2024 filing);
- NatWest (IFRS filer) has entity rows but no fundamentals at all;
- APPLE HOSPITALITY REIT shares the "APPLE" substring with Apple, without any fact;
- JPMorgan is a bank (Revenues, no positions, no gross profit), with a quarterly fact and a
  balance-sheet value dated at that quarter end only (none at its annual end);
- Apple's ticker row has no name, JPMorgan's repeats the current name (as production ticker
  rows may), and Apple has two names over time.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pyarrow as pa
from open_data_store.config import StoreConfig
from open_data_store.contracts.standard import ENTITY_OBSERVATIONS, FILING_STATEMENTS, FILINGS, FUNDAMENTALS
from open_data_store.publisher import approve_contract, publish_pending
from open_data_store.writer import write_batch

NOW = datetime(2026, 11, 9, 8, 0, tzinfo=UTC)
AAPL, JPM, NWG, APPLE_LIKE = "0000320193", "0000019617", "0000844150", "0000999999"
REV = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
AAPL_10K_2024, AAPL_10K_2025 = "0000320193-24-000123", "0000320193-25-000079"
#: fundamentals rows latest per key at 2026-01-01: 11 facts, the restated one counts once
N_FUNDAMENTALS_LATEST = 10


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def obs(cik, scheme, value, name=None, valid=date(2020, 1, 1), country="US"):
    return {"subject_scheme": "cik", "subject_value": cik, "id_scheme": scheme, "id_value": value,
                "name": name, "country": country, "valid_date": valid, "known_at": _ts("2026-11-02T00:00:00"),
                "known_at_kind": "published", "source_code": "sec", "source_doc_id": None}


def fact(cik, tag, end, value, *, start=None, freq="annual", known="2024-11-01T20:00:00",
         accn=AAPL_10K_2024, unit="USD"):
    return {"entity_scheme": "cik", "entity_value": cik, "original_tag": tag, "period_start": start,
                "frequency": freq, "fiscal_year": None, "basis": "consolidated", "value": Decimal(value),
                "unit": unit, "extraction_method": "xbrl_filed", "confidence": None, "source_page": None,
                "filing_ref": accn, "valid_date": end, "known_at": _ts(known), "known_at_kind": "published",
                "source_code": "sec", "source_doc_id": None}


def position(cik, tag, statement, accn=AAPL_10K_2024, end=date(2024, 9, 28),
             known="2024-11-01T20:00:00"):
    return {"entity_scheme": "cik", "entity_value": cik, "filing_ref": accn, "statement": statement,
                "tag": tag, "tag_version": "us-gaap/2024", "plabel": tag, "valid_date": end,
                "known_at": _ts(known), "known_at_kind": "published", "source_code": "sec",
                "source_doc_id": None}


def filing(cik, accn, form, report, filed, known):
    return {"entity_scheme": "cik", "entity_value": cik, "filing_ref": accn, "form": form,
            "report_date": report, "valid_date": filed, "known_at": _ts(known), "known_at_kind": "published", "source_code": "sec",
            "source_doc_id": None}


#: Apple's filings in the store's `filings` dataset (``seed(..., filings=True)``): the two 10-Ks and
#: an 8-K after the FY2025 one
AAPL_8K_2025 = "0000320193-25-000090"
AAPL_FILINGS = [
    filing(AAPL, AAPL_10K_2024, "10-K", date(2024, 9, 28), date(2024, 11, 1), "2024-11-01T20:00:00"),
    filing(AAPL, AAPL_10K_2025, "10-K", date(2025, 9, 27), date(2025, 10, 31), "2025-10-31T20:00:00"),
    filing(AAPL, AAPL_8K_2025, "8-K", date(2025, 11, 3), date(2025, 11, 3), "2025-11-03T21:00:00"),
]


def seed(tmp_path, *, filings: bool = False) -> StoreConfig:
    """The store; with ``filings``, the ``filings`` dataset is published too (Apple's only)."""
    cfg = StoreConfig(data_root=tmp_path / "store", catalog=str(tmp_path / "meta.ducklake"))
    for c in (ENTITY_OBSERVATIONS, FUNDAMENTALS, FILING_STATEMENTS, *((FILINGS,) if filings else ())):
        approve_contract(cfg, c, "test")
    land = cfg.landing
    write_batch(ENTITY_OBSERVATIONS, pa.Table.from_pylist([
        obs(AAPL, "cik", AAPL, "APPLE COMPUTER INC", date(1994, 1, 26)),
        obs(AAPL, "cik", AAPL, "Apple Inc.", date(2019, 8, 5)),
        obs(AAPL, "ticker", "AAPL", valid=date(2026, 11, 2)),
        obs(APPLE_LIKE, "cik", APPLE_LIKE, "APPLE HOSPITALITY REIT INC", date(2015, 1, 1)),
        obs(JPM, "cik", JPM, "JPMORGAN CHASE & CO", date(2005, 1, 1)),
        obs(JPM, "ticker", "JPM", "JPMORGAN CHASE & CO", valid=date(2026, 11, 2)),
        obs(NWG, "cik", NWG, "NatWest Group plc", date(2020, 7, 22), country="GB"),
        obs(NWG, "ticker", "NWG", valid=date(2026, 11, 2)),
    ], schema=ENTITY_OBSERVATIONS.schema), landing_root=land, run_id="r1", producer="test",
        now=NOW)
    fy24, fy25 = date(2024, 9, 28), date(2025, 9, 27)
    jpm_10k, jpm_10q = "0000019617-25-000270", "0000019617-25-000400"
    rows = [
        fact(AAPL, REV, fy24, 391035000000, start=date(2023, 10, 1)),
        # FY2025 10-K re-reports FY2024 with a different value: seen only from 2025-10-31
        fact(AAPL, REV, fy24, 391036000000, start=date(2023, 10, 1),
             known="2025-10-31T20:00:00", accn=AAPL_10K_2025),
        fact(AAPL, REV, fy25, 416161000000, start=date(2024, 9, 29),
             known="2025-10-31T20:00:00", accn=AAPL_10K_2025),
        fact(AAPL, "us-gaap:NetIncomeLoss", fy24, 93736000000, start=date(2023, 10, 1)),
        fact(AAPL, "us-gaap:Assets", fy24, 364980000000, freq="instant"),
        fact(AAPL, "us-gaap:StockholdersEquity", fy24, 56950000000, freq="instant"),
        fact(JPM, "us-gaap:Revenues", date(2024, 12, 31), 177556000000, start=date(2024, 1, 1),
             accn=jpm_10k, known="2025-02-14T20:00:00"),
        fact(JPM, "us-gaap:NetIncomeLoss", date(2024, 12, 31), 58471000000,
             start=date(2024, 1, 1), accn=jpm_10k, known="2025-02-14T20:00:00"),
        fact(JPM, "us-gaap:Revenues", date(2025, 3, 31), 45310000000, start=date(2025, 1, 1),
             freq="quarterly", accn=jpm_10q, known="2025-05-01T20:00:00"),
        # the 9-month year-to-date of the Q3 10-Q: stored as 'ytd9', only there to derive a Q4
        fact(JPM, "us-gaap:Revenues", date(2025, 9, 30), 135000000000, start=date(2025, 1, 1),
             freq="ytd9", accn="0000019617-25-000700", known="2025-11-04T20:00:00"),
        fact(JPM, "us-gaap:Assets", date(2025, 3, 31), 4357856000000, freq="instant",
             accn=jpm_10q, known="2025-05-01T20:00:00"),
    ]
    write_batch(FUNDAMENTALS, pa.Table.from_pylist(rows, schema=FUNDAMENTALS.schema),
                landing_root=land, run_id="r1", producer="test", now=NOW)
    write_batch(FILING_STATEMENTS, pa.Table.from_pylist([
        position(AAPL, REV.split(":")[1], "IS"), position(AAPL, "NetIncomeLoss", "IS"),
        position(AAPL, "Assets", "BS"), position(AAPL, "StockholdersEquity", "BS"),
    ], schema=FILING_STATEMENTS.schema), landing_root=land, run_id="r1", producer="test",
        now=NOW)
    if filings:
        write_batch(FILINGS, pa.Table.from_pylist(AAPL_FILINGS, schema=FILINGS.schema), landing_root=land,
                    run_id="r1", producer="test", now=NOW)
    rep = publish_pending(cfg, now=NOW)
    assert rep.exit_code == 0, rep
    return cfg


#: a year-to-date cash-flow filer (Alphabet-like), a capex-only year-to-date filer and semiannual reporters
GOOGL, SEMI, SEMI9, CAPXO = "0001652044", "0000777777", "0000777778", "0000777779"
CAPEX = "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
GOOGL_Q1, GOOGL_Q2, GOOGL_Q3 = "0001652044-24-000053", "0001652044-24-000079", "0001652044-24-000118"
GOOGL_10K, GOOGL_Q2_2025 = "0001652044-25-000014", "0001652044-25-000062"
SEMI_H1, SEMI_FY = "0000777777-24-000010", "0000777777-25-000003"
SEMI9_H1, SEMI9_N9, SEMI9_FY = "0000777778-24-000010", "0000777778-24-000020", "0000777778-25-000003"
CAPXO_Q1, CAPXO_Q2, CAPXO_Q3, CAPXO_FY = ("0000777779-24-000010", "0000777779-24-000020",
                                          "0000777779-24-000030", "0000777779-25-000003")


def seed_ytd(tmp_path) -> StoreConfig:
    """Alphabet's capex 2024 as filed (3M 12,012; 6M 25,198; 9M 38,259; FY 52,535, $M),
    with its standalone quarterly revenue (Q1 and Q2 filed, so no revenue quarter is derived there)
    and its 6-month revenue. Adversarial: the Q2 2025 10-Q re-reports the 6-month 2024 capex as a
    comparative, so the selection keeps that later vintage (a derived Q2 2024's filing_ref is the
    2025 10-Q while the period was first published by the Q2 2024 10-Q). SEMI files no quarters
    (H1 and fiscal year only): its 6-month rows stay. Adversarial: SEMI9 files no quarters either but
    also a 9-month year-to-date value and an H2 6-month row, so company-corpus derives its Q3 and Q4:
    those derived quarters must not hide its 6-month rows. Adversarial: CAPXO files capex only (Q1,
    6M, 9M, FY) and no other concept, so no filed quarter ends at its 6-month date: its derived Q2
    (6M - Q1) proves the filed Q1 and must hide that 6-month row; its balance sheet at the 6-month
    end is a quarter-end value, not a semiannual one."""
    cfg = StoreConfig(data_root=tmp_path / "store", catalog=str(tmp_path / "meta.ducklake"))
    for c in (ENTITY_OBSERVATIONS, FUNDAMENTALS, FILING_STATEMENTS):
        approve_contract(cfg, c, "test")
    land = cfg.landing
    write_batch(ENTITY_OBSERVATIONS, pa.Table.from_pylist([
        obs(GOOGL, "cik", GOOGL, "Alphabet Inc.", date(2015, 10, 2)),
        obs(GOOGL, "ticker", "GOOGL", valid=date(2026, 11, 2)),
        obs(SEMI, "cik", SEMI, "SEMIANNUAL REPORTER PLC", date(2015, 1, 1), country="GB"),
        obs(SEMI9, "cik", SEMI9, "SEMIANNUAL NINE-MONTH REPORTER PLC", date(2015, 1, 1), country="GB"),
        obs(CAPXO, "cik", CAPXO, "CAPEX ONLY FILER INC", date(2015, 1, 1)),
    ], schema=ENTITY_OBSERVATIONS.schema), landing_root=land, run_id="r1", producer="test", now=NOW)
    jan1 = date(2024, 1, 1)
    q1, h1, n9, fy = date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2024, 12, 31)
    k_q1, k_q2, k_q3, k_fy = ("2024-04-25T20:00:00", "2024-07-23T20:00:00", "2024-10-29T20:00:00",
                              "2025-02-04T21:00:00")
    m = 1_000_000
    rows = [
        fact(GOOGL, CAPEX, q1, 12012 * m, start=jan1, freq="quarterly", accn=GOOGL_Q1, known=k_q1),
        fact(GOOGL, CAPEX, h1, 25198 * m, start=jan1, freq="semiannual", accn=GOOGL_Q2, known=k_q2),
        fact(GOOGL, CAPEX, h1, 25198 * m, start=jan1, freq="semiannual", accn=GOOGL_Q2_2025,
             known="2025-07-23T20:00:00"),
        fact(GOOGL, CAPEX, n9, 38259 * m, start=jan1, freq="ytd9", accn=GOOGL_Q3, known=k_q3),
        fact(GOOGL, CAPEX, fy, 52535 * m, start=jan1, accn=GOOGL_10K, known=k_fy),
        fact(GOOGL, REV, q1, 80539 * m, start=jan1, freq="quarterly", accn=GOOGL_Q1, known=k_q1),
        fact(GOOGL, REV, h1, 84742 * m, start=date(2024, 4, 1), freq="quarterly", accn=GOOGL_Q2,
             known=k_q2),
        fact(GOOGL, REV, h1, 165281 * m, start=jan1, freq="semiannual", accn=GOOGL_Q2, known=k_q2),
        fact(SEMI, REV, h1, 500 * m, start=jan1, freq="semiannual", accn=SEMI_H1,
             known="2024-08-30T08:00:00"),
        fact(SEMI, REV, fy, 1100 * m, start=jan1, accn=SEMI_FY, known="2025-03-20T08:00:00"),
        fact(SEMI9, REV, h1, 500 * m, start=jan1, freq="semiannual", accn=SEMI9_H1,
             known="2024-08-30T08:00:00"),
        fact(SEMI9, REV, n9, 800 * m, start=jan1, freq="ytd9", accn=SEMI9_N9, known="2024-11-29T08:00:00"),
        fact(SEMI9, REV, fy, 1100 * m, start=jan1, accn=SEMI9_FY, known="2025-03-20T08:00:00"),
        fact(SEMI9, REV, fy, 600 * m, start=date(2024, 7, 1), freq="semiannual", accn=SEMI9_FY,
             known="2025-03-20T08:00:00"),
        fact(CAPXO, CAPEX, q1, 10 * m, start=jan1, freq="quarterly", accn=CAPXO_Q1, known=k_q1),
        fact(CAPXO, CAPEX, h1, 25 * m, start=jan1, freq="semiannual", accn=CAPXO_Q2, known=k_q2),
        fact(CAPXO, CAPEX, n9, 45 * m, start=jan1, freq="ytd9", accn=CAPXO_Q3, known=k_q3),
        fact(CAPXO, CAPEX, fy, 70 * m, start=jan1, accn=CAPXO_FY, known=k_fy),
        fact(CAPXO, "us-gaap:Assets", h1, 900 * m, freq="instant", accn=CAPXO_Q2, known=k_q2),
    ]
    write_batch(FUNDAMENTALS, pa.Table.from_pylist(rows, schema=FUNDAMENTALS.schema),
                landing_root=land, run_id="r1", producer="test", now=NOW)
    write_batch(FILING_STATEMENTS, pa.Table.from_pylist([
        position(GOOGL, CAPEX.split(":")[1], "CF", accn=GOOGL_Q1, end=q1, known=k_q1),
        position(GOOGL, REV.split(":")[1], "IS", accn=GOOGL_Q1, end=q1, known=k_q1),
    ], schema=FILING_STATEMENTS.schema),
                landing_root=land, run_id="r1", producer="test", now=NOW)
    rep = publish_pending(cfg, now=NOW)
    assert rep.exit_code == 0, rep
    return cfg
