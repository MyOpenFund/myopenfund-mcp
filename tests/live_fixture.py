"""A fake SEC for the live source: synthetic companyfacts and submissions for two issuers.

Shapes follow company-corpus's ``tests/opendata/factories.py`` (small helpers copied, not
imported). Microsoft's FY2024 10-K serves its recorded positions files (index.json,
FilingSummary.xml, presentation linkbase, copied from company-corpus's A6 fixtures), so its
selection reads real statement positions; every other filing folder answers 404 (positions
borrowed). Adversarial on purpose: Apple's FY2023 revenue is re-reported by the FY2024 10-K (two
vintages of one key), Apple has a former name, an 8-K is the newest filing, and an unknown CIK
has no companyfacts.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from company_corpus.live.sec import SecNotFound

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live"
AAPL, MSFT, NOFACTS = "0000320193", "0000789019", "0000000042"
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"
AAPL_REVENUE_FY2024 = 391035000000
MSFT_REVENUE_FY2024 = 245122000000

AAPL_K24, AAPL_Q24, AAPL_K23, AAPL_E24 = (
    "0000320193-24-000123",
    "0000320193-24-000081",
    "0000320193-23-000106",
    "0000320193-24-000130",
)
MSFT_K24, MSFT_K23 = "0000950170-24-087843", "0000950170-23-035122"
#: Microsoft's Q1 FY2025 10-Q: its folder answers 404, so it borrows the FY2024 10-K's positions
MSFT_Q25 = "0000950170-24-118967"
#: Microsoft's Q2 and Q3 FY2024 10-Qs. Q3 files revenue under two tags with different values (the
#: dictionary's first, Revenues, is not on its statement); its folder serves the FY2024 10-K's
#: recorded positions files, which put RevenueFromContract... on the income statement, so the
#: selection picks that tag only with Q3's positions. Q3 feeds the TTM of Q1 FY2025 (2024-09-30):
#: a call on that quarter must read Q3's positions to give the store's TTM. Q2's folder is 404.
MSFT_Q24_2, MSFT_Q24_3 = "0000950170-24-008814", "0000950170-24-048288"
MSFT_Q3_REVENUE, MSFT_Q3_OTHER_REVENUES = 61858000000, 61000000000

AAPL_FILINGS = [  # (accession, acceptance, form, filingDate, reportDate), newest first
    (AAPL_E24, "2024-11-05T21:00:00.000Z", "8-K", "2024-11-05", "2024-10-31"),
    (AAPL_K24, "2024-11-01T20:06:22.000Z", "10-K", "2024-11-01", "2024-09-28"),
    (AAPL_Q24, "2024-08-02T20:00:00.000Z", "10-Q", "2024-08-02", "2024-06-29"),
    (AAPL_K23, "2023-11-03T20:00:00.000Z", "10-K", "2023-11-03", "2023-09-30"),
]
MSFT_FILINGS = [
    (MSFT_Q25, "2024-10-30T20:05:00.000Z", "10-Q", "2024-10-30", "2024-09-30"),
    (MSFT_K24, "2024-07-30T20:06:22.000Z", "10-K", "2024-07-30", "2024-06-30"),
    (MSFT_Q24_3, "2024-04-25T20:05:00.000Z", "10-Q", "2024-04-25", "2024-03-31"),
    (MSFT_Q24_2, "2024-01-30T21:05:00.000Z", "10-Q", "2024-01-30", "2023-12-31"),
    (MSFT_K23, "2023-07-27T20:00:00.000Z", "10-K", "2023-07-27", "2023-06-30"),
]
AAPL_FORMER = [("APPLE COMPUTER INC", "1994-01-26", "2007-01-04")]

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}"


def filings_block(filings) -> dict:
    return {
        "accessionNumber": [f[0] for f in filings],
        "acceptanceDateTime": [f[1] for f in filings],
        "form": [f[2] for f in filings],
        "filingDate": [f[3] for f in filings],
        "reportDate": [f[4] for f in filings],
    }


def submissions_main(cik, *, name, tickers, former=(), filings=()) -> dict:
    business = {"isForeignLocation": 0, "stateOrCountry": "WA"}
    return {
        "cik": str(int(cik)),
        "name": name,
        "tickers": list(tickers),
        "formerNames": [{"name": n, "from": f, "to": t} for n, f, t in former],
        "addresses": {"business": business, "mailing": business},
        "filings": {"recent": filings_block(filings), "files": []},
    }


def point(val, end, *, start=None, accn, form="10-K", filed, fy=2024, fp="FY") -> dict:
    p = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start is not None:
        p["start"] = start
    return p


def companyfacts(cik, name, us_gaap: dict) -> dict:
    ns = {t: {"label": t, "units": units} for t, units in us_gaap.items()}
    return {"cik": int(cik), "entityName": name, "facts": {"us-gaap": ns, "dei": {}}}


def _aapl_facts() -> dict:
    return companyfacts(
        AAPL,
        "Apple Inc.",
        {
            REV: {
                "USD": [
                    point(
                        AAPL_REVENUE_FY2024,
                        "2024-09-28",
                        start="2023-10-01",
                        accn=AAPL_K24,
                        filed="2024-11-01",
                    ),
                    point(
                        383285000000,
                        "2023-09-30",
                        start="2022-10-02",
                        accn=AAPL_K24,
                        filed="2024-11-01",
                    ),
                    point(
                        383285000000,
                        "2023-09-30",
                        start="2022-10-02",
                        accn=AAPL_K23,
                        filed="2023-11-03",
                        fy=2023,
                    ),
                    point(
                        85777000000,
                        "2024-06-29",
                        start="2024-03-31",
                        accn=AAPL_Q24,
                        form="10-Q",
                        filed="2024-08-02",
                        fp="Q3",
                    ),
                ]
            },
            "NetIncomeLoss": {
                "USD": [
                    point(
                        93736000000,
                        "2024-09-28",
                        start="2023-10-01",
                        accn=AAPL_K24,
                        filed="2024-11-01",
                    ),
                    point(
                        96995000000,
                        "2023-09-30",
                        start="2022-10-02",
                        accn=AAPL_K23,
                        filed="2023-11-03",
                        fy=2023,
                    ),
                ]
            },
            "Assets": {
                "USD": [
                    point(364980000000, "2024-09-28", accn=AAPL_K24, filed="2024-11-01"),
                    point(352583000000, "2023-09-30", accn=AAPL_K23, filed="2023-11-03", fy=2023),
                ]
            },
        },
    )


def _msft_facts() -> dict:
    return companyfacts(
        MSFT,
        "MICROSOFT CORPORATION",
        {
            "NetIncomeLoss": {
                "USD": [
                    point(
                        21870000000,
                        "2023-12-31",
                        start="2023-10-01",
                        accn=MSFT_Q24_2,
                        form="10-Q",
                        filed="2024-01-30",
                        fy=2024,
                        fp="Q2",
                    ),
                    point(
                        21939000000,
                        "2024-03-31",
                        start="2024-01-01",
                        accn=MSFT_Q24_3,
                        form="10-Q",
                        filed="2024-04-25",
                        fy=2024,
                        fp="Q3",
                    ),
                    point(
                        66100000000,
                        "2024-03-31",
                        start="2023-07-01",
                        accn=MSFT_Q24_3,
                        form="10-Q",
                        filed="2024-04-25",
                        fy=2024,
                        fp="Q3",
                    ),
                    point(
                        88136000000,
                        "2024-06-30",
                        start="2023-07-01",
                        accn=MSFT_K24,
                        form="10-K",
                        filed="2024-07-30",
                        fy=2024,
                        fp="FY",
                    ),
                    point(
                        24667000000,
                        "2024-09-30",
                        start="2024-07-01",
                        accn=MSFT_Q25,
                        form="10-Q",
                        filed="2024-10-30",
                        fy=2025,
                        fp="Q1",
                    ),
                ]
            },
            "Revenues": {
                "USD": [
                    point(
                        MSFT_Q3_OTHER_REVENUES,
                        "2024-03-31",
                        start="2024-01-01",
                        accn=MSFT_Q24_3,
                        form="10-Q",
                        filed="2024-04-25",
                        fp="Q3",
                    ),
                ]
            },
            REV: {
                "USD": [
                    point(
                        MSFT_Q3_REVENUE,
                        "2024-03-31",
                        start="2024-01-01",
                        accn=MSFT_Q24_3,
                        form="10-Q",
                        filed="2024-04-25",
                        fp="Q3",
                    ),
                    point(  # 9-month YTD: derives Q4 FY2024 (ytd9 rule)
                        180395000000,
                        "2024-03-31",
                        start="2023-07-01",
                        accn=MSFT_Q24_3,
                        form="10-Q",
                        filed="2024-04-25",
                        fp="Q3",
                    ),
                    point(
                        62020000000,
                        "2023-12-31",
                        start="2023-10-01",
                        accn=MSFT_Q24_2,
                        form="10-Q",
                        filed="2024-01-30",
                        fp="Q2",
                    ),
                    point(
                        MSFT_REVENUE_FY2024,
                        "2024-06-30",
                        start="2023-07-01",
                        accn=MSFT_K24,
                        filed="2024-07-30",
                    ),
                    point(
                        65585000000,
                        "2024-09-30",
                        start="2024-07-01",
                        accn=MSFT_Q25,
                        form="10-Q",
                        filed="2024-10-30",
                        fy=2025,
                        fp="Q1",
                    ),
                    point(  # FY2023 re-reported by the FY2024 10-K (whose positions are recorded)
                        211915000000,
                        "2023-06-30",
                        start="2022-07-01",
                        accn=MSFT_K24,
                        filed="2024-07-30",
                    ),
                    point(
                        211915000000,
                        "2023-06-30",
                        start="2022-07-01",
                        accn=MSFT_K23,
                        filed="2023-07-27",
                        fy=2023,
                    ),
                ]
            },
        },
    )


def tickers_file() -> dict:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
        "2": {"cik_str": 42, "ticker": "NOFX", "title": "NO FACTS CO"},
    }


def routes() -> dict[str, bytes]:
    """URL -> body the fake SEC serves (anything else answers 404)."""
    out = {
        TICKERS_URL: json.dumps(tickers_file()).encode(),
        SUBMISSIONS_URL.format(cik=AAPL): json.dumps(
            submissions_main(
                AAPL,
                name="Apple Inc.",
                tickers=("AAPL",),
                former=AAPL_FORMER,
                filings=AAPL_FILINGS,
            )
        ).encode(),
        COMPANYFACTS_URL.format(cik=AAPL): json.dumps(_aapl_facts()).encode(),
        SUBMISSIONS_URL.format(cik=MSFT): json.dumps(
            submissions_main(MSFT, name="MICROSOFT CORP", tickers=("MSFT",), filings=MSFT_FILINGS)
        ).encode(),
        COMPANYFACTS_URL.format(cik=MSFT): json.dumps(_msft_facts()).encode(),
        SUBMISSIONS_URL.format(cik=NOFACTS): json.dumps(
            submissions_main(
                NOFACTS,
                name="NO FACTS CO",
                tickers=("NOFX",),
                filings=[("0000000042-24-000001", "2024-03-01T12:00:00.000Z", "8-K", "2024-03-01", "")],
            )
        ).encode(),
    }
    for accn in (MSFT_K24, MSFT_Q24_3):  # Q3 serves the 10-K's recorded files (see above)
        folder = ARCHIVES.format(cik_int=int(MSFT), acc_nodash=accn.replace("-", ""))
        for f in (FIXTURES / MSFT / MSFT_K24).iterdir():
            out[f"{folder}/{f.name}"] = f.read_bytes()
    return out


class FakeSec:
    """A SecGet over ``routes``: unknown URLs raise SecNotFound. ``gate`` (an Event) holds every
    request whose URL contains ``hold`` until it is set: a slow SEC."""

    def __init__(self, served: dict[str, bytes] | None = None, *, hold: str | None = None):
        self.routes = routes() if served is None else served
        self.calls: list[tuple[str, float | None]] = []
        self.hold, self.gate = hold, threading.Event()
        self._lock = threading.Lock()

    def get_bytes(self, url, *, max_age_s=None):
        if self.hold and self.hold in url:
            self.gate.wait(30)
        with self._lock:
            self.calls.append((url, max_age_s))
        if url not in self.routes:
            raise SecNotFound(url)
        return self.routes[url]

    def urls(self) -> list[str]:
        with self._lock:
            return [u for u, _ in self.calls]
