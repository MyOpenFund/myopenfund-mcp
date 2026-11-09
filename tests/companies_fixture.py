"""A real my-open-data store in tmp_path, written by the real writer and publisher.

Adversarial on purpose (no perfect-world fixture):
- Apple's FY2024 revenue is restated by the FY2025 10-K (a later vintage of the same key), and
  that 10-K has no filing_statements rows (its positions are borrowed from the 2024 filing);
- NatWest (IFRS filer) has entity rows but no fundamentals at all;
- APPLE HOSPITALITY REIT shares the "APPLE" substring with Apple, without any fact;
- JPMorgan is a bank (Revenues, no positions, no gross profit), with a quarterly fact;
- Apple's ticker row has no name, JPMorgan's repeats the current name (as production ticker
  rows may), and Apple has two names over time.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pyarrow as pa
from myopendata.config import StoreConfig
from myopendata.contracts.standard import ENTITY_OBSERVATIONS, FILING_STATEMENTS, FUNDAMENTALS
from myopendata.publisher import approve_contract, publish_pending
from myopendata.writer import write_batch

NOW = datetime(2026, 11, 9, 8, 0, tzinfo=UTC)
AAPL, JPM, NWG, APPLE_LIKE = "0000320193", "0000019617", "0000844150", "0000999999"
REV = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
AAPL_10K_2024, AAPL_10K_2025 = "0000320193-24-000123", "0000320193-25-000079"
#: fundamentals rows latest per key at 2026-01-01: 9 facts, the restated one counts once
N_FUNDAMENTALS_LATEST = 8


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


def seed(tmp_path) -> StoreConfig:
    cfg = StoreConfig(data_root=tmp_path / "store", catalog=str(tmp_path / "meta.ducklake"))
    for c in (ENTITY_OBSERVATIONS, FUNDAMENTALS, FILING_STATEMENTS):
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
    ]
    write_batch(FUNDAMENTALS, pa.Table.from_pylist(rows, schema=FUNDAMENTALS.schema),
                landing_root=land, run_id="r1", producer="test", now=NOW)
    write_batch(FILING_STATEMENTS, pa.Table.from_pylist([
        position(AAPL, REV.split(":")[1], "IS"), position(AAPL, "NetIncomeLoss", "IS"),
        position(AAPL, "Assets", "BS"), position(AAPL, "StockholdersEquity", "BS"),
    ], schema=FILING_STATEMENTS.schema), landing_root=land, run_id="r1", producer="test",
        now=NOW)
    rep = publish_pending(cfg, now=NOW)
    assert rep.exit_code == 0, rep
    return cfg
