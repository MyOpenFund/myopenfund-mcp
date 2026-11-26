"""Opt-in: the no-vault mode against the real SEC (no store, no Qdrant): the scenario's companies.

    MOF_SEC_CONTACT="Your Name your.email@example.com" \
        python -m pytest -m integration -s tests/integration/test_no_vault_real.py
    MOF_SEC_CONTACT=... python -m pytest -m models -s tests/integration/test_no_vault_real.py

`-m integration`: quarterly capex and total debt of Microsoft, Amazon, Alphabet, Meta and Oracle since 2022,
and values checked by hand against their 10-K text (the pinned values below). `-m models` (downloads the
embedding model once per run, 1.1 GB, unless MOF_MCP_IT_MODELS_DIR gives it — before any timing — and
indexes 10-Ks: minutes on 4 CPU cores):
- the first quotable passage of each company (spec §8): seconds from `companies_fetch_filings` (its 10-Ks
  since 2022, in a fresh cache folder) to the first `companies_search_filings` hit in Item 7 about capex or
  financing; criterion MOF_MCP_IT_FIRST_PASSAGE_S (default 120, the spec's 2 minutes); once per company
  fetched alone, and once for the five fetched in ONE call (22 10-Ks: each company within the criterion);
- a French question over the Microsoft FY2024 10-K finds a passage of its Item 7 (Review Focus 5).

Optional: MOF_MCP_IT_CACHE_DIR (absolute; the cache folder of the facts tests and the French one, default a
temporary folder per run), MOF_MCP_IT_MODELS_DIR (absolute; a folder holding the model files, the `models/` of
an earlier cache folder: linked into every fresh cache folder instead of downloading),
MOF_SEC_MAX_RPS, MOF_RAG_ONNX_* and MOF_INDEX_* (passed to the server; no other variable of the shell is),
MOF_MCP_IT_INDEX_TIMEOUT_S (seconds a models test waits for indexing, default 1800). Set
MOF_RAG_ONNX_PROVIDER=coreml to measure the Mac's accelerator. The contact is read from the environment only,
never printed.
"""
import asyncio
import datetime as dt
import os
import re
import time

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

CONTACT = os.environ.get("MOF_SEC_CONTACT")

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not CONTACT, reason="MOF_SEC_CONTACT not set")]

TICKERS = ["MSFT", "AMZN", "GOOGL", "META", "ORCL"]
SINCE = "2022-01-01"
#: as_of of the pinned values: after every FY2024 10-K, before any FY2025 one could restate them
PINNED_AS_OF = "2025-04-30T00:00:00Z"
#: (ticker, kind, key, period_end) -> value in USD, read in the 10-K text (C11, 2026-11-26; the 10-K
#: sentence or table line and its URL are in the chantier's C11 report)
PINNED: dict[tuple[str, str, str, str], int] = {
    # msft-20240630.htm: "Additions to property and equipment (44,477) (28,107)"; balance sheet "Short-term
    # debt 6,693", "Current portion of long-term debt 2,249", "Long-term debt 42,688" (sum 51,630)
    ("MSFT", "concept", "capex", "2024-06-30"): 44_477_000_000,
    ("MSFT", "concept", "capex", "2023-06-30"): 28_107_000_000,
    ("MSFT", "metric", "total_debt", "2024-06-30"): 51_630_000_000,
    # amzn-20241231.htm: "Purchases of property and equipment (63,645) (52,729) (82,999)"; Note 6 "Total face
    # value of long-term debt 67,182 58,000"
    ("AMZN", "concept", "capex", "2024-12-31"): 82_999_000_000,
    ("AMZN", "concept", "capex", "2023-12-31"): 52_729_000_000,
    ("AMZN", "metric", "total_debt", "2024-12-31"): 58_000_000_000,
    # goog-20241231.htm: "Purchases of property and equipment (31,485) (32,251) (52,535)"; Note 6 "Total face
    # value of long-term debt 13,000 12,000" and "$2.3 billion of commercial paper outstanding" (sum 14,300)
    ("GOOGL", "concept", "capex", "2024-12-31"): 52_535_000_000,
    ("GOOGL", "concept", "capex", "2023-12-31"): 32_251_000_000,
    ("GOOGL", "metric", "total_debt", "2024-12-31"): 14_300_000_000,
    # meta-20241231.htm: "Purchases of property and equipment (37,256) (27,045) (31,186)"; balance sheet
    # "Long-term debt 28,826 18,385"
    ("META", "concept", "capex", "2024-12-31"): 37_256_000_000,
    ("META", "concept", "capex", "2023-12-31"): 27_045_000_000,
    ("META", "metric", "total_debt", "2024-12-31"): 28_826_000_000,
    # orcl-20240531.htm: "Capital expenditures (6,866) (8,695) (4,511)"; Note 8 "Total notes payable and other
    # borrowings $ 86,869 $ 90,481"
    ("ORCL", "concept", "capex", "2024-05-31"): 6_866_000_000,
    ("ORCL", "concept", "capex", "2023-05-31"): 8_695_000_000,
    ("ORCL", "metric", "total_debt", "2024-05-31"): 86_869_000_000,
}
#: an Item 7 passage worth quoting for the scenario: it speaks of capex or financing
QUOTABLE = re.compile(r"capital expenditure|property and equipment|capex|financ|debt|borrow", re.IGNORECASE)
SCENARIO_QUERY = "capital expenditures purchases of property and equipment financing debt"
FRENCH_QUERY = "Comment Microsoft finance-t-il ses dépenses d'investissement ?"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name) or str(default)
    try:
        value = int(raw)
    except ValueError:
        pytest.fail(f"{name} must be an integer number of seconds, got {raw!r}")
    if value < 1:
        pytest.fail(f"{name} must be >= 1, got {value}")
    return value


def _fresh_cache(path, models: str | None = None) -> str:
    """A new cache folder; the model files of `models` (a `models/` folder) linked into it when given."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    if models:
        os.symlink(models, os.path.join(path, "models"))
    return str(path)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory) -> str:
    """The embedding model files, once per module: MOF_MCP_IT_MODELS_DIR, else downloaded here (1.1 GB)
    before any timed test, so that no timing counts the download."""
    given = os.environ.get("MOF_MCP_IT_MODELS_DIR")
    if given:
        if not os.path.isabs(given) or not os.path.isdir(given):
            pytest.fail("MOF_MCP_IT_MODELS_DIR must be the absolute path of an existing folder")
        return given
    from rag_engine.chunking import load_tokenizer
    from rag_engine.config import Settings as EngineSettings
    from rag_engine.embedding import ModelSpec, OnnxEmbedder

    folder = str(tmp_path_factory.mktemp("models"))
    spec = ModelSpec.from_settings(EngineSettings())
    OnnxEmbedder(spec, cache_dir=folder).encode_passage(["warm up"])
    load_tokenizer(folder, spec.name)
    return folder


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> str:
    given = os.environ.get("MOF_MCP_IT_CACHE_DIR")
    if given:
        if not os.path.isabs(given):
            pytest.fail("MOF_MCP_IT_CACHE_DIR must be an absolute path")
        return given
    return _fresh_cache(tmp_path_factory.mktemp("mof") / "cache", os.environ.get("MOF_MCP_IT_MODELS_DIR"))


#: what the server reads from this process's environment: the contact, the ONNX knobs, the SEC rate
PASSED_ENV = ("MOF_SEC_CONTACT", "MOF_SEC_MAX_RPS", "MOF_RAG_ONNX_PROVIDER", "MOF_RAG_ONNX_THREADS",
              "MOF_INDEX_WORKERS", "MOF_INDEX_BATCH_SIZE", "MOF_INDEX_FIRST_BATCHES", "MOF_INDEX_FETCH_AHEAD")


def _client(cache_dir: str) -> Client:
    """A server on `cache_dir` with a clean environment: only the variables above (no other MOF_* of
    the shell, which could point the test at a store or another cache)."""
    env = {k: os.environ[k] for k in PASSED_ENV if k in os.environ}
    env.update({"MOF_COMPANIES_SOURCE": "sec", "MOF_CACHE_DIR": cache_dir})
    return Client(build_server(Settings.from_env(env)))


async def _call(client: Client, tool: str, args: dict, *, retry=("[warming]", "[timeout]"), wait_s: int = 600):
    """The tool's data; `[warming]` (SEC still being read) and `[timeout]` are asked again, as the
    descriptions tell a model to."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return (await client.call_tool(tool, args)).data
        except ToolError as exc:
            if not str(exc).startswith(retry) or time.monotonic() > deadline:
                raise
        await asyncio.sleep(2)


async def _fundamentals(client: Client, args: dict) -> list[dict]:
    rows, cursor = [], None
    while True:
        page = await _call(client, "companies_fundamentals", {**args, "limit": 200, **({"cursor": cursor} if cursor
                                                                                           else {})})
        rows += page["rows"]
        cursor = page["next_cursor"]
        if not cursor:
            return rows


@pytest.mark.integration
async def test_quarterly_capex_and_total_debt_since_2022(cache_dir):
    async with _client(cache_dir) as c:
        for ticker in TICKERS:
            rows = await _fundamentals(c, {"companies": [ticker], "concepts": ["capex"], "metrics": ["total_debt"],
                                           "frequency": "quarterly", "period_from": SINCE})
            capex = {str(r["period_end"])[:10]: r["value"] for r in rows if r["key"] == "capex"}
            debt = {str(r["period_end"])[:10]: r["value"] for r in rows if r["key"] == "total_debt"}
            print(f"{ticker}: {len(capex)} quarters of capex, {len(debt)} of total debt")
            assert {r["source"] for r in rows} == {"sec_live"}
            # as filed: a cash-flow item has a quarterly row only where the filing has a three-month column
            # (Microsoft and Amazon: Q1-Q3; Alphabet, Meta, Oracle: Q1 only, then 6- and 9-month YTD), so at
            # least one per fiscal year 2022-2024; total debt (balance sheet) at the first three quarter ends of
            # a fiscal year at least, the fiscal year end often being under `annual` only (Meta: from Q3 2022)
            assert sum(day < "2025-01-01" for day in capex) >= 3, (ticker, sorted(capex))
            assert sum(day < "2025-01-01" for day in debt) >= 8, (ticker, sorted(debt))
            assert all(v > 0 for v in capex.values()) and all(v > 0 for v in debt.values())


@pytest.mark.integration
@pytest.mark.parametrize("ticker", TICKERS)
async def test_values_read_in_the_10k_text(cache_dir, ticker):
    got, want = {}, {k: v for k, v in PINNED.items() if k[0] == ticker}
    async with _client(cache_dir) as c:
        for key in want:
            _, kind, name, period_end = key
            args = {"companies": [ticker], "frequency": "annual", "period_from": period_end,
                    "period_to": period_end, "as_of": PINNED_AS_OF}
            args.update({"concepts": [name]} if kind == "concept" else {"concepts": ["capex"], "metrics": [name]})
            rows = await _fundamentals(c, args)
            got[key] = [r["value"] for r in rows if r["key"] == name and str(r["period_end"])[:10] == period_end]
    assert got == {k: [v] for k, v in want.items()}


def _quotable(hits: list[dict]) -> dict | None:
    return next((h for h in hits if re.match(r"Item 7\b", h["section"] or "") and QUOTABLE.search(h["text"])),
                None)


async def _first_passages(c: Client, tickers: list[str], t0: float, wait_s: int) -> dict[str, float]:
    """Seconds from `t0` to each company's first quotable Item 7 hit (searched company by company,
    every 2 s); a company still without one after `wait_s` is missing from the result."""
    found: dict[str, float] = {}
    while len(found) < len(tickers) and time.monotonic() - t0 < wait_s:
        for ticker in tickers:
            if ticker in found:
                continue
            try:
                res = await _call(c, "companies_search_filings",
                                  {"query": SCENARIO_QUERY, "companies": [ticker], "limit": 10})
            except ToolError as exc:
                if not str(exc).startswith("[index_not_ready]"):
                    raise
                continue
            hit = _quotable(res["hits"])
            if hit is not None:
                assert hit["url"].startswith("https://www.sec.gov/")
                found[ticker] = time.monotonic() - t0
        if len(found) < len(tickers):
            await asyncio.sleep(2)
    return found


@pytest.mark.models
@pytest.mark.parametrize("ticker", TICKERS)
async def test_first_quotable_passage(tmp_path, models_dir, ticker):
    limit_s = _int_env("MOF_MCP_IT_FIRST_PASSAGE_S", 120)
    wait_s = _int_env("MOF_MCP_IT_INDEX_TIMEOUT_S", 1800)
    async with _client(_fresh_cache(tmp_path / "cache", models_dir)) as c:
        t0 = time.monotonic()
        fetched = await _call(c, "companies_fetch_filings",
                              {"companies": [ticker], "forms": ["10-K"], "period_from": SINCE})
        assert len(fetched["documents"]) >= 4
        assert fetched["model_download"] is False
        found = await _first_passages(c, [ticker], t0, wait_s)
    print(f"{ticker}: first quotable Item 7 passage after {found.get(ticker, float('nan')):.0f} s "
          f"({len(fetched['documents'])} 10-Ks queued)")
    assert ticker in found, f"{ticker}: no quotable Item 7 passage after {wait_s} s"
    assert found[ticker] <= limit_s, f"{ticker}: {found[ticker]:.0f} s > {limit_s} s (MOF_MCP_IT_FIRST_PASSAGE_S)"


@pytest.mark.models
async def test_first_quotable_passage_of_five_companies_fetched_in_one_call(tmp_path, models_dir):
    """R41, the failing shape of C11: the five companies' 10-Ks since 2022 (22 documents) in ONE
    fetch; each company's first quotable Item 7 passage within the criterion of the fetch call."""
    limit_s = _int_env("MOF_MCP_IT_FIRST_PASSAGE_S", 120)
    wait_s = _int_env("MOF_MCP_IT_INDEX_TIMEOUT_S", 1800)
    async with _client(_fresh_cache(tmp_path / "cache", models_dir)) as c:
        t0 = time.monotonic()
        fetched = await _call(c, "companies_fetch_filings",
                              {"companies": TICKERS, "forms": ["10-K"], "period_from": SINCE})
        fetch_s = time.monotonic() - t0
        assert len(fetched["documents"]) >= 20
        assert fetched["model_download"] is False
        found = await _first_passages(c, TICKERS, t0, min(wait_s, 3 * limit_s))
    total = time.monotonic() - t0
    for ticker in TICKERS:
        print(f"{ticker}: first quotable Item 7 passage after {found.get(ticker, float('nan')):.0f} s")
    print(f"fetch call {fetch_s:.1f} s, {len(fetched['documents'])} 10-Ks queued; last first passage "
          f"{max(found.values(), default=float('nan')):.0f} s; measured for {total:.0f} s")
    assert set(found) == set(TICKERS), f"no quotable Item 7 passage for {sorted(set(TICKERS) - set(found))}"
    late = {t: round(s) for t, s in found.items() if s > limit_s}
    assert not late, f"first passage above {limit_s} s (MOF_MCP_IT_FIRST_PASSAGE_S): {late}"


@pytest.mark.models
async def test_french_query_finds_passages(cache_dir):
    """Review Focus 5: dense retrieval (multilingual e5) still finds the MD&A for a French question."""
    async with _client(cache_dir) as c:
        fetched = await _call(c, "companies_fetch_filings", {
            "companies": ["MSFT"], "forms": ["10-K"], "period_from": "2024-06-30", "period_to": "2024-06-30"})
        assert [d["filed"] for d in fetched["documents"]] == ["2024-07-30"]
        deadline = time.monotonic() + _int_env("MOF_MCP_IT_INDEX_TIMEOUT_S", 1800)
        while True:
            state = (await _call(c, "companies_fetch_filings", {
                "companies": ["MSFT"], "forms": ["10-K"], "period_from": "2024-06-30",
                "period_to": "2024-06-30"}))["documents"][0]
            if state["state"] in ("ready", "failed") or time.monotonic() > deadline:
                break
            await asyncio.sleep(5)
        assert state["state"] == "ready", state
        res = await _call(c, "companies_search_filings", {
            "query": FRENCH_QUERY, "companies": ["MSFT"], "forms": ["10-K"], "period_from": "2024-06-30",
            "period_to": "2024-06-30", "limit": 10})
    sections = [h["section"] for h in res["hits"]]
    print("sections of the top 10:", sections)
    assert len(res["hits"]) == 10
    item7 = [h for h in res["hits"] if re.match(r"Item 7\b", h["section"] or "")]
    assert item7, sections
    print("first Item 7 hit:", item7[0]["url"], item7[0]["text"][:300])
    assert all(h["url"].startswith("https://www.sec.gov/") for h in item7)
    assert dt.date.fromisoformat(str(item7[0]["filed"])[:10]) == dt.date(2024, 7, 30)
