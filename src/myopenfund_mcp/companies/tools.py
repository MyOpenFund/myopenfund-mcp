"""The `companies` toolset: identify companies, read their filed facts, concepts and ratios as of a date.

Concepts and ratios come from company-corpus through the store adapter; nothing is computed here.
Every argument is checked before the store is touched, and each call has ONE time budget
(`companies_query_timeout_s`) shared by all its store reads.
"""
from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal

from fastmcp import FastMCP
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
    tool_error,
)
from ..paging import MAX_OFFSET, cap_result, clamp_limit, decode_cursor, encode_cursor
from . import resolve as res
from .common import AS_OF_DESCRIPTION, READ_ONLY, parse_as_of, rows_to_utc, store_call, time_budget

Frequency = Literal["annual", "semiannual", "quarterly", "instant"]
SECTOR = ("unknown (SIC not in the store): sector_relevant is null for the ratios that depend on the "
          "sector (banks, insurers)")
COMPANIES_DESCRIPTION = ("CIKs, tickers or entity_ids, at most the server maximum per call; with only a "
                         "name, call companies_resolve first and pass the CIK")
FREQUENCY_DESCRIPTION = ("Period covered; balance-sheet items are 'instant', so 'annual' alone drops them. "
                         "Omitted = all")
MAX_QUERY_CHARS = 200
MAX_CURSOR_CHARS = 200
#: company-corpus's PeriodSummary frequency labels -> the store's
_SUMMARY_FREQUENCY = {"semi-annual": "semiannual"}


def _in_range(day: dt.date, period_from: dt.date | None, period_to: dt.date | None) -> bool:
    return (period_from is None or day >= period_from) and (period_to is None or day <= period_to)


def _newest_first(day: dt.date) -> int:
    return -day.toordinal()


def build_companies_toolset(store, settings: Settings, *, auth_enabled: bool) -> FastMCP:
    co = FastMCP("companies", mask_error_details=True)
    read = tier_check("read", auth_enabled=auth_enabled)

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
        """One entity per distinct company (in the order given); unknown or ambiguous is an error."""
        idx = res.index(await store_call(store.entities, settings=settings))
        out: dict[str, dict] = {}
        for q in companies:
            hits = res.resolve(idx, q)
            if not hits:
                raise tool_error(UNKNOWN_COMPANY, f"no company matches {q!r}; try companies_resolve")
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
        with time_budget(settings):
            idx = res.index(await store_call(store.entities, settings=settings))
        hits = res.resolve(idx, query)
        return cap_result({"candidates": hits[:page], "total": len(hits)}, settings,
                          hint="ask for a smaller limit or a more precise query")

    @co.tool(auth=read, annotations=READ_ONLY)
    async def facts(
        companies: Annotated[list[str], Field(description=COMPANIES_DESCRIPTION, min_length=1)],
        tags: Annotated[list[str] | None, Field(
            description="Tags as filed, 'namespace:Tag', e.g. 'us-gaap:Revenues'; omitted = all")] = None,
        frequency: Annotated[Frequency | None, Field(description=FREQUENCY_DESCRIPTION)] = None,
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
        With only a name, call companies_resolve first. Rows: newest period first per company;
        `valid_date` is the period end; datetimes are UTC."""
        page, offset = check(companies, period_from, period_to, limit, cursor)
        when = parse_as_of(as_of)
        with time_budget(settings):
            hits = await lookup(companies)
            names = {h["cik"]: h["current_name"] for h in hits}
            flt: dict = {"entity_scheme": ["cik"], "entity_value": list(names)}
            if tags:
                flt["original_tag"] = tags
            if frequency:
                flt["frequency"] = [frequency]
            found = await store_call(store.read, settings=settings, dataset="fundamentals", as_of=when,
                                     filters=flt, columns=None, limit=None)
        rows = [{"cik": r["entity_value"], "name": names[r["entity_value"]],
                 **{k: v for k, v in r.items() if k != "batch_id"}}
                for r in found if _in_range(r["valid_date"], period_from, period_to)]
        rows.sort(key=lambda r: (r["cik"], _newest_first(r["valid_date"]), r["original_tag"], r["frequency"]))
        chunk, nxt = page_of(rows, page, offset)
        return cap_result({"as_of": when.isoformat(), "returned": len(chunk), "total": len(rows),
                           "next_cursor": nxt, "rows": rows_to_utc(chunk)}, settings,
                          hint="filter tags, a frequency or a period range, or ask for a smaller limit")

    @co.tool(auth=read, annotations=READ_ONLY)
    async def fundamentals(
        companies: Annotated[list[str], Field(description=COMPANIES_DESCRIPTION, min_length=1)],
        concepts: Annotated[list[str] | None, Field(
            description="Standard concepts, e.g. 'revenue', 'net_income', 'long_term_debt', 'cash' "
                        "(total debt is the metric 'total_debt'); omitted = all")] = None,
        metrics: Annotated[list[str] | Literal["all"] | None, Field(
            description="Ratios computed by company-corpus, e.g. 'net_margin', 'roe', 'roe_ttm', "
                        "'net_debt_to_ebitda_ttm'; 'all' (or ['all']) = every one; omitted = none")] = None,
        frequency: Annotated[Frequency | None, Field(description=FREQUENCY_DESCRIPTION)] = None,
        period_from: Annotated[dt.date | None, Field(description="Period end on or after, YYYY-MM-DD")] = None,
        period_to: Annotated[dt.date | None, Field(description="Period end on or before, YYYY-MM-DD")] = None,
        as_of: Annotated[dt.datetime | None, Field(description=AS_OF_DESCRIPTION)] = None,
        limit: Annotated[int | None, Field(description="Rows per page; omitted = server default")] = None,
        cursor: Annotated[str | None, Field(description="next_cursor of the previous page, unchanged",
                                            max_length=MAX_CURSOR_CHARS)] = None,
    ) -> dict:
        """Standard financials of companies as publicly known at `as_of`: one row per company,
        period and concept (with the filing it comes from), plus ratios on request (`metrics`,
        computed by company-corpus, TTM included; `ttm` says which). Call companies_resolve first
        if you only have a name. Banks and insurers: the sector is unknown to the store, so judge
        debt and margin ratios accordingly. Rows: newest period first per company, concepts before
        metrics; datetimes are UTC."""
        concept_labels = {c["key"]: c["label"] for c in store.concept_catalog()}
        metric_keys = {m["key"] for m in store.metric_catalog()}
        bad = sorted(set(concepts or []) - set(concept_labels))
        if bad:
            raise tool_error(UNKNOWN_CONCEPT, f"unknown {bad}; valid: {', '.join(sorted(concept_labels))}")
        all_metrics = metrics == "all" or metrics == ["all"]
        wanted = metric_keys if all_metrics else set(metrics or [])
        bad = sorted(wanted - metric_keys)
        if bad:
            raise tool_error(UNKNOWN_METRIC, f"unknown {bad}; valid: 'all' or {', '.join(sorted(metric_keys))}")
        page, offset = check(companies, period_from, period_to, limit, cursor)
        when = parse_as_of(as_of)

        def keep(day, freq):
            return (frequency is None or freq == frequency) and _in_range(day, period_from, period_to)

        rows, version = [], None
        with time_budget(settings):
            for h in await lookup(companies):
                cik, name = h["cik"], h["current_name"]
                sel, vintages = await store_call(store.concept_rows, settings=settings, cik=cik, as_of=when)
                for r in sel:
                    version = version or r.get("dictionary_version")
                    if (concepts and r["concept"] not in concepts) or not keep(r["valid_date"], r["frequency"]):
                        continue
                    rows.append({"cik": cik, "name": name, "period_end": r["valid_date"],
                                 "frequency": r["frequency"], "kind": "concept", "key": r["concept"],
                                 "label": concept_labels.get(r["concept"], r["concept"]), "value": r["value"],
                                 "unit": r["unit"], "original_tag": r["original_tag"],
                                 "filing_ref": r["filing_ref"], "source_doc_id": r["source_doc_id"],
                                 "known_at": r["known_at"], "location": r["location"],
                                 "conflict": r["conflict"]})
                if not (wanted and sel):
                    continue
                summaries = await store_call(store.summaries, settings=settings, cik=cik, name=name,
                                             rows=sel, vintages=vintages)
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
                                             "sector_relevant": m.get("sector_relevant")})
        rows.sort(key=lambda r: (r["cik"], _newest_first(r["period_end"]), r["frequency"],
                                 r["kind"] != "concept", r["key"]))
        chunk, nxt = page_of(rows, page, offset)
        return cap_result({"as_of": when.isoformat(), "dictionary_version": version, "sector": SECTOR,
                           "returned": len(chunk), "total": len(rows), "next_cursor": nxt,
                           "rows": rows_to_utc(chunk)}, settings,
                          hint="ask for fewer concepts or metrics, a frequency or a period range")

    return co
