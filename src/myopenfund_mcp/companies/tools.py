"""The `companies` toolset: identify companies, read their filed facts, concepts and ratios as of a date.

Concepts and ratios come from company-corpus through the store adapter or the live SEC source, chosen
per company by the source router (`MOF_COMPANIES_SOURCE`); nothing is computed here. Every row says
where it comes from (`source`, and `reason` when `auto` answered live). Every argument is checked
before any source is touched, and each call has ONE time budget (`companies_query_timeout_s`, plus
`sec_call_budget_s` when SEC may be reached) shared by all its reads; the companies of a call are
read concurrently.
"""
from __future__ import annotations

import datetime as dt
from functools import cache
from typing import Annotated, Literal, get_args

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import tier_check
from ..config import Settings
from ..errors import (
    AMBIGUOUS_COMPANY,
    INVALID_ARGUMENT,
    TOO_MANY_COMPANIES,
    UNKNOWN_COMPANY,
    UNKNOWN_CONCEPT,
    UNKNOWN_METRIC,
    WARMING,
    tool_error,
)
from ..paging import MAX_OFFSET, cap_result, clamp_limit, decode_cursor, encode_cursor
from . import resolve as res
from .common import (
    AS_OF_DESCRIPTION,
    READ_ONLY,
    WarmingError,
    call_budget_s,
    coded,
    parse_as_of,
    rows_to_utc,
    store_call,
    time_budget,
)
from .router import LIVE, STORE, STORE_UNAVAILABLE, Route, SourceRouter

Frequency = Literal["annual", "semiannual", "quarterly", "instant"]
#: as filed in the store; ytd9 = the 9-month year-to-date, which only derives an unreported Q4
FactsFrequency = Literal[*get_args(Frequency), "ytd9"]
#: stored frequencies that exist only to derive other values (TTM): never output by companies_fundamentals
_DERIVATION_ONLY = frozenset({"ytd9"})
SECTOR = ("unknown (SIC not in the store): sector_relevant is null for the ratios that depend on the "
          "sector (banks, insurers)")
COMPANIES_DESCRIPTION = ("CIKs, tickers or entity_ids, at most the server maximum per call; with only a "
                         "name, call companies_resolve first and pass the CIK")
FUNDAMENTALS_FREQUENCY_DESCRIPTION = (
    "Period covered: annual/quarterly/semiannual include the balance-sheet values (frequency 'instant') "
    "dated at those period ends, as the ratios use them; 'instant' = balance-sheet values only; "
    "omitted = all")
FACTS_FREQUENCY_DESCRIPTION = ("The fact's own period as filed (balance-sheet values are 'instant'; "
                               "'ytd9' = 9-month year-to-date); omitted = all")
MAX_QUERY_CHARS = 200
#: one company identifier (CIK, ticker, entity_id); same bound as companies_resolve's query
CompanyId = Annotated[str, Field(max_length=MAX_QUERY_CHARS)]
MAX_CURSOR_CHARS = 200
#: company-corpus's PeriodSummary frequency labels -> the store's
_SUMMARY_FREQUENCY = {"semi-annual": "semiannual"}


def _in_range(day: dt.date, period_from: dt.date | None, period_to: dt.date | None) -> bool:
    return (period_from is None or day >= period_from) and (period_to is None or day <= period_to)


def _newest_first(day: dt.date) -> int:
    return -day.toordinal()


def _origin(route: Route) -> dict:
    """The row fields saying where a company's rows come from."""
    return {"source": route.source, **({"reason": route.reason} if route.reason else {})}


async def _per_company(hits: list[dict], fn) -> list:
    """``fn(hit)`` for every company at once, results in the order of ``hits``. All or error: the first
    company (in that order) that failed otherwise decides the error; else, when some are still being
    fetched from SEC, one ``[warming]`` lists each of them with its progress."""
    results: list = [None] * len(hits)
    errors: list[ToolError | None] = [None] * len(hits)

    async def run(i: int, hit: dict) -> None:
        try:
            results[i] = await fn(hit)
        except ToolError as exc:
            errors[i] = exc

    async with anyio.create_task_group() as tg:
        for i, hit in enumerate(hits):
            tg.start_soon(run, i, hit)
    failed = next((e for e in errors if e is not None and not isinstance(e, WarmingError)), None)
    if failed is not None:
        raise failed
    warming = [(h, e) for h, e in zip(hits, errors, strict=True) if isinstance(e, WarmingError)]
    if warming:
        progress = "; ".join(f"{h['current_name'] or h['cik']} (CIK {h['cik']}): {e.done}/{e.total} requests "
                             "done" for h, e in warming)
        raise tool_error(WARMING, f"still fetching from SEC: {progress}; the fetch goes on in the "
                                  "background: call again in a few seconds")
    return results


class _Budget:
    """One call's time budget, in the cancel scope of its `time_budget`.

    It starts at `companies_query_timeout_s`, plus `sec_call_budget_s` when SEC may be reached. Once
    routing says no company goes live it narrows to the store budget plus the routing time, at most
    `sec_freshness_budget_s` (the SEC freshness check's own bound); a company that goes live (or falls back to SEC) restores the full budget. A
    live read waits `sec_call_budget_s` counted from the start of the call (routing included), so a
    cold call answers `[warming]` before the call budget ends in `[timeout]`."""

    def __init__(self, settings: Settings, scope, *, live_enabled: bool):
        self._settings, self._scope, self._live_enabled = settings, scope, live_enabled
        self._t0 = anyio.current_time()

    def routed(self, routes: dict[str, Route], routing_s: float) -> None:
        if any(r.source == LIVE for r in routes.values()):
            self.go_live()
        elif self._live_enabled:
            self._scope.deadline = self._t0 + self._settings.companies_query_timeout_s + routing_s

    def go_live(self) -> None:
        s = self._settings
        self._scope.deadline = self._t0 + s.companies_query_timeout_s + s.sec_call_budget_s

    def live_wait(self) -> float:
        return max(0.0, self._t0 + self._settings.sec_call_budget_s - anyio.current_time())


def build_companies_toolset(router: SourceRouter, settings: Settings, *, auth_enabled: bool) -> FastMCP:
    co = FastMCP("companies", mask_error_details=True)
    read = tier_check("read", auth_enabled=auth_enabled)
    budget_s = call_budget_s(settings, live=router.live_enabled)

    # auto with a store and no contact: a company outside the store needs SEC, which is not configured
    unknown_hint = ("not in the company store, and companies outside it are read from SEC, which needs "
                    "MOF_SEC_CONTACT (sec_contact_missing); or try companies_resolve"
                    if router.mode == "auto" and router.live is None else "try companies_resolve")

    async def call(fn, /, **kwargs):
        return await store_call(fn, settings=settings, timeout_s=budget_s, **kwargs)

    async def plan(hits: list[dict], when: dt.datetime, budget: _Budget) -> dict[str, Route]:
        t = anyio.current_time()
        routes = await call(router.plan, ciks=[h["cik"] for h in hits], as_of=when)
        # credit the SEC check only (bounded by MOF_SEC_FRESHNESS_BUDGET_S): a slow store read during
        # routing must not stretch a pure-store call past its budget
        budget.routed(routes, min(anyio.current_time() - t, settings.sec_freshness_budget_s))
        return routes

    async def on_live(h: dict, route: Route, budget: _Budget, method: str, **kwargs):
        """One company's read from SEC (or the error naming what is missing)."""
        return await call(router.call, company=h["cik"], route=route, method=method,
                          budget_s=budget.live_wait(), **kwargs)

    @cache
    def catalogs() -> tuple[dict[str, str], frozenset[str]]:
        """Concept labels by key and metric keys: company-corpus's dictionary, read once."""
        return ({c["key"]: c["label"] for c in router.concept_catalog()},
                frozenset(m["key"] for m in router.metric_catalog()))

    def check(companies, period_from, period_to, limit, cursor) -> tuple[int, int]:
        """Arguments every multi-company tool shares, checked before any store work: (page, offset)."""
        if len(companies) > settings.companies_max_companies:
            raise tool_error(TOO_MANY_COMPANIES, f"{len(companies)} companies; at most "
                                                 f"{settings.companies_max_companies} per call")
        if period_from and period_to and period_from > period_to:
            raise tool_error(INVALID_ARGUMENT, "period_from is after period_to")
        return clamp_limit(limit, settings), decode_cursor(cursor)

    def page_of(rows, page, offset):
        chunk = rows[offset:offset + page]
        end = offset + len(chunk)
        return chunk, (encode_cursor(end) if end < len(rows) and end <= MAX_OFFSET else None)

    async def lookup(companies: list[str]) -> list[dict]:
        """One entity per distinct company (in the order given); unknown or ambiguous is an error (the
        SEC resolver's own error when it failed and the store alone cannot resolve the query)."""
        rows, sec_failure = await call(router.resolver_rows)
        idx = res.index(rows)
        out: dict[str, dict] = {}
        for q in companies:
            hits = res.resolve(idx, q)
            if not hits:
                if sec_failure is not None:
                    raise coded(sec_failure)
                raise tool_error(UNKNOWN_COMPANY, f"no company matches {q!r}; {unknown_hint}")
            if len(hits) > 1:
                names = ", ".join(f"{h['current_name']} (CIK {h['cik']})" for h in hits[:10])
                more = f" and {len(hits) - 10} more" if len(hits) > 10 else ""
                raise tool_error(AMBIGUOUS_COMPANY,
                                 f"{q!r} matches {len(hits)} companies: {names}{more}; pass a CIK")
            out.setdefault(hits[0]["cik"], hits[0])
        return list(out.values())

    @co.tool(auth=read, annotations=READ_ONLY)
    async def resolve(
        query: Annotated[str, Field(description="Name (any part, former names too), ticker, CIK or entity_id",
                                    max_length=MAX_QUERY_CHARS)],
        limit: Annotated[int | None, Field(description="Candidates to return; omitted = server default")] = None,
    ) -> dict:
        """Find a company's identifiers (CIK, entity_id, tickers, names over time). Call this first
        when you only have a name; several candidates = ambiguous: pick one by CIK. `names` lists
        former names oldest first with the date each was first seen (`since`)."""
        page = clamp_limit(limit, settings)
        with time_budget(settings, budget_s):
            rows, sec_failure = await call(router.resolver_rows)
        hits = res.resolve(res.index(rows), query)
        if not hits and sec_failure is not None:
            raise coded(sec_failure)
        return cap_result({"candidates": hits[:page], "total": len(hits)}, settings,
                          hint="ask for a smaller limit or a more precise query")

    @co.tool(auth=read, annotations=READ_ONLY)
    async def facts(
        companies: Annotated[list[CompanyId], Field(description=COMPANIES_DESCRIPTION, min_length=1)],
        tags: Annotated[list[str] | None, Field(
            description="Tags as filed, 'namespace:Tag', e.g. 'us-gaap:Revenues'; omitted = all")] = None,
        frequency: Annotated[FactsFrequency | None, Field(description=FACTS_FREQUENCY_DESCRIPTION)] = None,
        period_from: Annotated[dt.date | None, Field(description="Period end on or after, YYYY-MM-DD")] = None,
        period_to: Annotated[dt.date | None, Field(description="Period end on or before, YYYY-MM-DD")] = None,
        as_of: Annotated[dt.datetime | None, Field(description=AS_OF_DESCRIPTION)] = None,
        limit: Annotated[int | None, Field(description="Rows per page; omitted = server default")] = None,
        cursor: Annotated[str | None, Field(description="next_cursor of the previous page, unchanged",
                                            max_length=MAX_CURSOR_CHARS)] = None,
    ) -> dict:
        """Facts exactly as filed (raw XBRL tags) of companies, the latest value publicly known at
        `as_of` per tag and period (a later restatement replaces the original from its filing date).
        For standard items (revenue, net income, debt...) and ratios prefer companies_fundamentals.
        With only a name, call companies_resolve first. Raw: `frequency` filters the filed column
        as is (no balance-sheet values added to a period). Rows: newest period first per company;
        `valid_date` is the period end; datetimes are UTC; `source` says where a company's rows
        come from (store or sec_live), `reason` why SEC answered when the store could have.
        `[warming]` = still fetching from SEC: call again."""
        page, offset = check(companies, period_from, period_to, limit, cursor)
        when = parse_as_of(as_of)
        flt: dict = {"entity_scheme": ["cik"]}
        if tags:
            flt["original_tag"] = tags
        if frequency:
            flt["frequency"] = [frequency]

        read = {"dataset": "fundamentals", "as_of": when, "columns": None, "limit": None}

        with time_budget(settings, budget_s) as scope:
            budget = _Budget(settings, scope, live_enabled=router.live_enabled)
            hits = await lookup(companies)
            routes = await plan(hits, when, budget)
            # the store's companies in ONE read, as before routing; then each live company
            held = [h["cik"] for h in hits if routes[h["cik"]].source == STORE]
            by_cik: dict[str, list[dict]] = {}
            if held:
                got = await call(router.try_store, method="read", filters={**flt, "entity_value": held},
                                 **read)
                if got is None:  # the store failed: SEC answers those companies for this call
                    routes.update(dict.fromkeys(held, Route(LIVE, STORE_UNAVAILABLE)))
                    budget.go_live()
                else:
                    for r in got:
                        by_cik.setdefault(r["entity_value"], []).append(r)
            live_hits = [h for h in hits if routes[h["cik"]].source == LIVE]

            async def one(h):
                route, part = await on_live(h, routes[h["cik"]], budget, "read",
                                            filters={**flt, "entity_value": [h["cik"]]}, **read)
                routes[h["cik"]] = route
                return part

            for h, part in zip(live_hits, await _per_company(live_hits, one), strict=True):
                by_cik[h["cik"]] = part
        rows = [{"cik": h["cik"], "name": h["current_name"],
                 **{k: v for k, v in r.items() if k != "batch_id"}, **_origin(routes[h["cik"]])}
                for h in hits for r in by_cik.get(h["cik"], [])
                if _in_range(r["valid_date"], period_from, period_to)]
        rows.sort(key=lambda r: (r["cik"], _newest_first(r["valid_date"]), r["original_tag"], r["frequency"]))
        chunk, nxt = page_of(rows, page, offset)
        return cap_result({"as_of": when.isoformat(), "returned": len(chunk), "total": len(rows),
                           "next_cursor": nxt, "rows": rows_to_utc(chunk)}, settings,
                          hint="filter tags, a frequency or a period range, or ask for a smaller limit")

    @co.tool(auth=read, annotations=READ_ONLY)
    async def fundamentals(
        companies: Annotated[list[CompanyId], Field(description=COMPANIES_DESCRIPTION, min_length=1)],
        concepts: Annotated[list[str] | None, Field(
            description="Standard concepts, e.g. 'revenue', 'net_income', 'long_term_debt', 'cash' "
                        "(total debt is the metric 'total_debt'); omitted = all")] = None,
        metrics: Annotated[list[str] | Literal["all"] | None, Field(
            description="Ratios computed by company-corpus, e.g. 'net_margin', 'roe', 'roe_ttm', "
                        "'net_debt_to_ebitda_ttm'; 'all' (or ['all']) = every one; omitted = none")] = None,
        frequency: Annotated[Frequency | None, Field(description=FUNDAMENTALS_FREQUENCY_DESCRIPTION)] = None,
        period_from: Annotated[dt.date | None, Field(description="Period end on or after, YYYY-MM-DD")] = None,
        period_to: Annotated[dt.date | None, Field(description="Period end on or before, YYYY-MM-DD")] = None,
        as_of: Annotated[dt.datetime | None, Field(description=AS_OF_DESCRIPTION)] = None,
        limit: Annotated[int | None, Field(description="Rows per page; omitted = server default")] = None,
        cursor: Annotated[str | None, Field(description="next_cursor of the previous page, unchanged",
                                            max_length=MAX_CURSOR_CHARS)] = None,
    ) -> dict:
        """Standard financials of companies as publicly known at `as_of`: one row per company,
        period and concept (with the filing it comes from), plus ratios on request (`metrics`,
        computed by company-corpus, TTM included; `ttm` says which; a quarterly TTM that would need
        a fourth quarter the company did not report on its own may be absent). An annual, quarterly or
        semiannual `frequency` includes the balance-sheet values ('instant') dated at those period
        ends, the ones the ratios use. Call companies_resolve first
        if you only have a name. Banks and insurers: the sector is unknown to the store, so judge
        debt and margin ratios accordingly. Rows: newest period first per company, concepts before
        metrics; datetimes are UTC; `source` says where a company's rows come from (store or
        sec_live), `reason` why SEC answered when the store could have. `[warming]` = still fetching
        from SEC: call again."""
        concept_labels, metric_keys = catalogs()
        bad = sorted(set(concepts or []) - set(concept_labels))
        if bad:
            raise tool_error(UNKNOWN_CONCEPT, f"unknown {bad}; valid: {', '.join(sorted(concept_labels))}")
        if isinstance(metrics, list) and "all" in metrics and len(metrics) > 1:
            raise tool_error(INVALID_ARGUMENT, "'all' cannot be combined with other metrics")
        wanted = set(metric_keys) if metrics in ("all", ["all"]) else set(metrics or [])
        bad = sorted(wanted - metric_keys)
        if bad:
            raise tool_error(UNKNOWN_METRIC, f"unknown {bad}; valid: 'all' or {', '.join(sorted(metric_keys))}")
        page, offset = check(companies, period_from, period_to, limit, cursor)
        when = parse_as_of(as_of)

        def keep(day, freq):  # a duration period (or, for frequency='instant', a balance-sheet date)
            # output filter only: store.summaries below still receives the whole selection (ytd9 included)
            return (freq not in _DERIVATION_ONLY and (frequency is None or freq == frequency)
                    and _in_range(day, period_from, period_to))

        def keep_concept(r, period_ends):
            """F1's period model: a duration frequency brings the instants dated at its period ends."""
            if frequency not in (None, "instant") and r["frequency"] == "instant":
                return r["valid_date"] in period_ends
            return keep(r["valid_date"], r["frequency"])

        async def one(h, routes, budget):
            route, got = routes[h["cik"]], None
            if route.source == STORE:
                got = await call(router.try_store, method="concept_rows", cik=h["cik"], as_of=when)
                if got is None:  # the store failed: SEC answers this company for this call
                    route = Route(LIVE, STORE_UNAVAILABLE)
                    budget.go_live()
            if got is None:
                route, got = await on_live(h, route, budget, "concept_rows", cik=h["cik"], as_of=when,
                                           period_from=period_from, period_to=period_to)
            sel, vintages = got
            summaries = []
            if wanted and sel:
                summaries = await call(router.source(route).summaries, cik=h["cik"], name=h["current_name"],
                                       rows=sel, vintages=vintages)
            return route, sel, summaries

        rows, version = [], None
        with time_budget(settings, budget_s) as scope:
            budget = _Budget(settings, scope, live_enabled=router.live_enabled)
            hits = await lookup(companies)
            routes = await plan(hits, when, budget)
            found = await _per_company(hits, lambda h: one(h, routes, budget))
            for h, (route, sel, summaries) in zip(hits, found, strict=True):
                cik, name, origin = h["cik"], h["current_name"], _origin(route)
                # ends of this company's included duration periods (any concept, before the concept filter)
                period_ends = {r["valid_date"] for r in sel if keep(r["valid_date"], r["frequency"])}
                for r in sel:
                    version = version or r.get("dictionary_version")
                    if (concepts and r["concept"] not in concepts) or not keep_concept(r, period_ends):
                        continue
                    rows.append({"cik": cik, "name": name, "period_end": r["valid_date"],
                                 "frequency": r["frequency"], "kind": "concept", "key": r["concept"],
                                 "label": concept_labels.get(r["concept"], r["concept"]), "value": r["value"],
                                 "unit": r["unit"], "original_tag": r["original_tag"],
                                 "filing_ref": r["filing_ref"], "source_doc_id": r["source_doc_id"],
                                 "known_at": r["known_at"], "location": r["location"],
                                 "conflict": r["conflict"], **origin})
                for s in summaries:
                    freq = _SUMMARY_FREQUENCY.get(s.frequency, s.frequency)
                    if s.period_end is None or not keep(s.period_end, freq):
                        continue
                    for is_ttm, block in ((False, s.derived), (True, s.ttm)):
                        for key, m in block.items():
                            if key in wanted and m.get("value") is not None:
                                rows.append({"cik": cik, "name": name, "period_end": s.period_end,
                                             "frequency": freq, "kind": "metric", "key": key,
                                             "label": m.get("label"), "value": m["value"],
                                             "unit": m.get("unit"), "ttm": is_ttm,
                                             "sector_relevant": m.get("sector_relevant"), **origin})
        rows.sort(key=lambda r: (r["cik"], _newest_first(r["period_end"]), r["frequency"],
                                 r["kind"] != "concept", r["key"]))
        chunk, nxt = page_of(rows, page, offset)
        return cap_result({"as_of": when.isoformat(), "dictionary_version": version, "sector": SECTOR,
                           "returned": len(chunk), "total": len(rows), "next_cursor": nxt,
                           "rows": rows_to_utc(chunk)}, settings,
                          hint="ask for fewer concepts or metrics, a frequency or a period range")

    return co
