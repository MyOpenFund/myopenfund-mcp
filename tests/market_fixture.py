"""A synthetic market store in tmp_path (the real open-data-store writer and publisher), and a fake live
adapter (market-data's Tiingo adapter on an httpx mock transport). Nothing here is real data.

Adversarial on purpose:
- every price comes from a restricted source (tiingo, private_history): a caller without the
  allow-list must get none of them, from any tool;
- instrument I2 is named only by a private_history label (``PRIV_LABEL``) and priced by a
  private_history series (``PRIV_TOKEN``): neither string may reach a caller;
- one price row comes from a source no registry declares (``UNREGISTERED``): never served, even to an
  allowed client (the market datasets are read in strict mode, as market-data's own binding reads them);
- I1 has a ticker and a FIGI from openfigi and a CIK from sec (open sources), and one dividend;
- ``seed_market(overlap=True)`` adds a massive series of I1 overlapping tiingo's on the last three sessions,
  at other closes: two sources a client may both read, on one instrument (MOF_MARKET_PRECEDENCE decides);
- ``seed_market(conflict=True)`` adds a second tiingo series of I1 (``tiingo:AAZ``) with another close on
  the first session: two series of one source disagree, which no precedence resolves;
- ``seed_market(public_alias=True)``: the private history also names I2 by ``BBB``, a public ticker the fake
  live source serves.
"""
from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pyarrow as pa
from open_data_store.config import StoreConfig
from open_data_store.contracts.standard.market import (
    CORPORATE_ACTIONS,
    INSTRUMENT_IDENTIFIERS,
    MARKET_CONTRACTS,
    PRICES,
)
from open_data_store.market.asof import sessions
from open_data_store.publisher import approve_contract, publish_pending
from open_data_store.writer import write_batch

START, END = date(2025, 6, 2), date(2025, 6, 6)
NOW = datetime(2025, 6, 10, 12, 0, tzinfo=UTC)
LOADED = datetime(2025, 6, 8, 0, 0, tzinfo=UTC)
I1, I2 = "mof:" + "1" * 32, "mof:" + "2" * 32
FIGI1, CIK1 = "BBG000FAKE01", "0000000042"
PRIV_LABEL, PRIV_TOKEN = "PRVLABELQ", "priv_tok_9f3a"
KEY = "synthetic-tiingo-key-0000"
UNREGISTERED = "unregistered_feed"
#: prices rows per source in the store
N_TIINGO, N_PRIVATE = 5, 5
_SESS = sessions("XNYS", START, END)
DAYS = _SESS["session_date"].to_pylist()
_CLOSES = dict(zip(DAYS, _SESS["close_utc"].to_pylist(), strict=True))


def _price(source: str, symbol: str, day: date, close: float) -> dict:
    return {"source_symbol": symbol, "open": None, "high": None, "low": None, "close": close, "volume": 10,
            "currency": "USD", "price_basis": "raw", "known_at_basis": "session_close", "valid_date": day,
            "known_at": _CLOSES[day], "known_at_kind": "published", "source_code": source, "source_doc_id": None}


def _ident(instrument: str, scheme: str, value: str, source: str) -> dict:
    return {"instrument_id": instrument, "id_scheme": scheme, "id_value": value, "valid_to": None,
            "link_reason": None, "known_at_basis": "load_time", "valid_date": date(2000, 1, 1), "known_at": LOADED,
            "known_at_kind": "collected", "source_code": source, "source_doc_id": None}


#: closes of the massive series of I1 (``seed_market(overlap=True)``), on DAYS[2:]
MASSIVE_CLOSES = (500.0, 501.0, 502.0)


def seed_market(tmp_path, *, overlap: bool = False, conflict: bool = False,
                public_alias: bool = False) -> StoreConfig:
    cfg = StoreConfig(data_root=tmp_path / "store", catalog=str(tmp_path / "meta.ducklake"))
    for contract in MARKET_CONTRACTS:
        approve_contract(cfg, contract, "test")
    prices = [_price("tiingo", "AAA", d, 100.0 + i) for i, d in enumerate(DAYS)]
    prices += [_price("private_history", PRIV_TOKEN, d, 20.0 + i) for i, d in enumerate(DAYS)]
    prices += [_price(UNREGISTERED, "UNR", DAYS[0], 1.0)]  # a source no registry declares: denied (strict)
    if overlap:
        prices += [_price("massive", "AAA", d, c) for d, c in zip(DAYS[2:], MASSIVE_CLOSES, strict=True)]
    if conflict:
        prices.append(_price("tiingo", "AAZ", DAYS[0], 999.0))
    dividend = {"source_symbol": "AAA", "action_type": "cash_dividend_regular", "action_seq": 0,
                "source_action_type": "divCash", "announce_date": None, "record_date": None, "pay_date": None,
                "amount": 1.0, "currency": "USD", "amount_basis": None, "per_share_basis": None,
                "ratio_old": None, "ratio_new": None, "spinoff_symbol": None, "known_at_basis": "ex_date_assumed",
                "valid_date": DAYS[2], "known_at": _CLOSES[DAYS[1]], "known_at_kind": "published",
                "source_code": "tiingo", "source_doc_id": None}
    ids = [
        _ident(I1, "source_symbol", "tiingo:AAA", "tiingo"),
        _ident(I1, "ticker", "AAA", "openfigi"),
        _ident(I1, "composite_figi", FIGI1, "openfigi"),
        _ident(I1, "cik", CIK1, "sec"),
        _ident(I2, "source_symbol", f"private_history:{PRIV_TOKEN}", "private_history"),
        _ident(I2, "ticker", PRIV_LABEL, "private_history"),
    ]
    if overlap:
        ids.append(_ident(I1, "source_symbol", "massive:AAA", "massive"))
    if conflict:
        ids.append(_ident(I1, "source_symbol", "tiingo:AAZ", "tiingo"))
    if public_alias:
        ids.append(_ident(I2, "ticker", "BBB", "private_history"))
    for contract, rows in ((PRICES, prices), (CORPORATE_ACTIONS, [dividend]), (INSTRUMENT_IDENTIFIERS, ids)):
        write_batch(contract, pa.Table.from_pylist(rows, schema=contract.schema), landing_root=cfg.landing,
                    run_id="t", producer="test")
    report = publish_pending(cfg)
    assert report.exit_code == 0, report
    return cfg


# --- the fake live source ---------------------------------------------------------------------------
def _bar(day: str, close: float) -> dict:
    return {"date": f"{day}T00:00:00.000Z", "open": close, "high": close + 1, "low": close - 1, "close": close,
            "volume": 1000, "adjClose": close, "divCash": 0.0, "splitFactor": 1.0}


#: what the fake Tiingo knows: AAA and BBB (another ticker), nothing else
PAYLOADS = {
    "AAA": [_bar(d.isoformat(), 300.0 + i) for i, d in enumerate(DAYS)],
    "BBB": [_bar(d.isoformat(), 40.0 + i) for i, d in enumerate(DAYS)],
}


def fake_tiingo(requests: list | None = None, **adapter):
    """market-data's TiingoAdapter on a mock transport: no network, no quota file (``adapter``: other
    TiingoAdapter arguments, e.g. a state directory, ``per_hour``, ``sleep``)."""
    import httpx
    from market_data.adapters.tiingo import TiingoAdapter

    def answer(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        ticker = request.url.path.split("/")[3]
        if ticker not in PAYLOADS:
            return httpx.Response(404, json={"detail": "Error: Ticker not found"})
        return httpx.Response(200, content=json.dumps(PAYLOADS[ticker]).encode())

    kw = {"state_dir": None, "min_spacing_s": 0, "sleep": lambda s: None, "now": lambda: NOW, **adapter}
    return TiingoAdapter(KEY, transport=httpx.MockTransport(answer), **kw)
