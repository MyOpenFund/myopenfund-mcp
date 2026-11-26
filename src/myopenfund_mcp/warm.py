"""`myopenfund-mcp warm`: fetch facts and filings ahead of any question, index them to `ready`.

The same objects the server builds (the cached SEC client, the live source, the source router, the
filings indexer) and the same helpers the tools call (`lookup_companies`, `check_companies`,
`router.concept_rows`, `router.filings`, `indexer.enqueue`/`status`): what `warm` leaves in the cache
and the index is what the tools find later. Resumable: documents already `ready` are not enqueued
again, and the SEC cache answers the rest of a second run.

Progress goes to stderr, one line per step, never the SEC contact. Exit codes: 0 all ready, 1 a step
or a document failed, 2 bad arguments or missing contact, 130 interrupted (Ctrl-C).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import logging
import math
import sys
import time
from collections.abc import Callable
from typing import TextIO

from fastmcp.exceptions import ToolError

from .companies.common import call_budget_s, store_call
from .companies.tools import check_companies, lookup_companies, unknown_company_hint
from .config import WARM_POLL_MAX_S, ConfigError, Settings, _forms
from .errors import (
    AMBIGUOUS_COMPANY,
    INVALID_ARGUMENT,
    LIMIT_EXCEEDED,
    NO_FACTS,
    TOO_MANY_COMPANIES,
    UNKNOWN_COMPANY,
    WARMING,
    error_code,
)

log = logging.getLogger("myopenfund_mcp.warm")

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
#: States a document leaves by itself; every other one (queued, fetching, indexing, partial) is waiting.
DONE_STATES = ("ready", "failed")
#: Error codes that mean the arguments, not SEC or the machine, are at fault.
USAGE_CODES = (UNKNOWN_COMPANY, AMBIGUOUS_COMPANY, TOO_MANY_COMPANIES, INVALID_ARGUMENT, LIMIT_EXCEEDED)


class UsageError(Exception):
    """Bad arguments or configuration: exit 2, the message goes to stderr."""


@dataclasses.dataclass
class Stack:
    """What a warm run uses; `close` releases it (the indexer first, bounded by its own timeout)."""

    router: object
    indexer: object
    closers: tuple = ()

    def close(self) -> None:
        for closer in (self.indexer.close, *self.closers):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 -- the others still close; the class only is logged
                log.warning("closing failed (%s)", type(exc).__name__)


def build_stack(settings: Settings) -> Stack:
    """The server's own construction (`server._build_server`) through the same helpers, without the MCP
    layer and without a company store: a warm run exists to fill the cache from SEC."""
    from .companies.build import build_live, build_router
    from .filings.indexer import FilingsIndexer

    sec, live = build_live(settings)
    router = build_router(None, live, settings)
    return Stack(router, FilingsIndexer(settings, sec), closers=(router.close, live.close))


def parse_companies(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",")]
    if not raw.strip() or not all(names):
        raise UsageError("--companies must be comma-separated tickers, CIKs or names, none empty")
    return list(dict.fromkeys(names))


def parse_forms(raw: str | None, settings: Settings) -> list[str]:
    if raw is None:
        return list(settings.filings_default_forms)
    try:
        return list(_forms(raw))
    except ConfigError:
        raise UsageError(
            f"--forms must be comma-separated form types such as 10-K,10-Q, got {raw!r}"
        ) from None


class Warmer:
    def __init__(
        self,
        settings: Settings,
        stack: Stack,
        *,
        poll_interval_s: float,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] | None = None,
        err: TextIO | None = None,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ):
        self.settings, self.stack = settings, stack
        self.poll_interval_s, self._sleep, self._clock = poll_interval_s, sleep, clock or time.monotonic
        self._err = err
        self._now = now
        self.failed_steps: list[str] = []
        self.stalled: list = []

    def say(self, line: str) -> None:
        print(f"[warm] {line}", file=self._err or sys.stderr, flush=True)

    async def _acall(self, fn, /, **kwargs):
        """The tools' own call path: off the loop, within the call budget, errors mapped to codes."""
        return await store_call(
            fn, settings=self.settings, timeout_s=call_budget_s(self.settings, live=True), **kwargs
        )

    def call(self, fn, /, **kwargs):
        return asyncio.run(self._acall(fn, **kwargs))

    def resolve(self, companies: list[str]) -> list[dict]:
        router = self.stack.router
        try:
            check_companies(companies, self.settings)
            return asyncio.run(
                lookup_companies(
                    router, companies, call=self._acall, unknown_hint=unknown_company_hint(router)
                )
            )
        except ToolError as exc:
            if error_code(exc, "") in USAGE_CODES:
                raise UsageError(str(exc)) from None
            self.say(f"cannot resolve the companies: {exc}")
            raise

    def facts(self, hit: dict, since: dt.date, until: dt.date | None) -> None:
        """Build the issuer's tables and statement positions over the window (cached for later calls);
        `[warming]` = still fetching: look again after the poll interval."""
        label = (hit.get("tickers") or [hit["cik"]])[0]
        while True:
            try:
                self.call(
                    self.stack.router.concept_rows,
                    cik=hit["cik"],
                    as_of=self._now(),
                    period_from=since,
                    period_to=until,
                )
            except ToolError as exc:
                code = error_code(exc, "error")
                if code == WARMING:
                    self.say(
                        f"{label} facts {getattr(exc, 'done', '?')}/{getattr(exc, 'total', '?')} requests"
                    )
                    self._sleep(self.poll_interval_s)
                    continue
                if code == NO_FACTS:
                    self.say(f"{label} facts none (SEC has no XBRL facts for it)")
                    return
                self.say(f"{label} facts failed ({code})")
                self.failed_steps.append(f"{label} facts")
                return
            self.say(f"{label} facts ok")
            return

    def refs(self, hits: list[dict], forms: list[str], since: dt.date, until: dt.date | None) -> list:
        from .filings.indexer import filing_doc_id

        found: dict[str, object] = {}
        for hit in hits:
            label = (hit.get("tickers") or [hit["cik"]])[0]
            try:
                listed = self.call(
                    self.stack.router.filings,
                    cik=hit["cik"],
                    forms=forms,
                    period_from=since,
                    period_to=until,
                )
            except ToolError as exc:
                self.say(f"{label} filings failed ({error_code(exc, 'error')})")
                self.failed_steps.append(f"{label} filings")
                continue
            self.say(f"{label} filings {len(listed)}")
            for ref in listed:
                found.setdefault(filing_doc_id(ref), ref)
        return list(found.values())

    def index(self, refs: list, hits: list[dict]) -> list:
        """Enqueue every document (batches of MOF_FILINGS_MAX_DOCS; the indexer leaves a `ready` one
        alone, without any SEC call), then wait until each is `ready` or `failed`, or nothing moves for
        MOF_WARM_STALL_S; the statuses of all of them."""
        from .filings.indexer import filing_doc_id

        indexer = self.stack.indexer
        names = {h["cik"]: h["current_name"] for h in hits}
        tickers = {h["cik"]: h.get("tickers") or [] for h in hits}
        ids = [filing_doc_id(r) for r in refs]
        step = self.settings.filings_max_docs
        already = 0
        for i in range(0, len(refs), step):
            queued = indexer.enqueue(refs[i : i + step], names=names, tickers=tickers)
            already += sum(s.state == "ready" for s in queued)
        if already:
            self.say(f"{already}/{len(refs)} documents already ready")
        last, seen, moved_at = None, None, self._clock()
        while True:
            statuses = list(indexer.status(ids))
            ready_n = sum(s.state == "ready" for s in statuses)
            waiting = [s for s in statuses if s.state not in DONE_STATES]
            line = (
                f"{ready_n}/{len(ids)} documents ready ("
                f"{sum(s.state == 'indexing' for s in statuses)} indexing)"
            )
            if line != last:
                self.say(line)
                last = line
            if not waiting:
                return statuses
            signature = tuple((s.doc_id, s.state, s.chunks_done) for s in statuses)
            if signature != seen:
                seen, moved_at = signature, self._clock()
            elif self._clock() - moved_at >= self.settings.warm_stall_s:
                self.stalled = waiting
                self.say(f"no progress for {self.settings.warm_stall_s}s (MOF_WARM_STALL_S): stopping")
                return statuses
            self._sleep(self.poll_interval_s)

    def run(self, companies: list[str], forms: list[str], since: dt.date, until: dt.date | None) -> int:
        hits = self.resolve(companies)
        for hit in hits:
            self.facts(hit, since, until)
        refs = self.refs(hits, forms, since, until)
        statuses = self.index(refs, hits) if refs else []
        if not refs:
            self.say("no filing of these forms in this window")
        failed = [s for s in statuses if s.state == "failed"]
        if failed:
            from .filings.indexer import public_error
        for s in failed:
            self.say(f"failed {s.doc_id} ({s.form} {s.filed}): {public_error(s.error) or 'no detail'}")
        for s in self.stalled:
            self.say(f"stuck {s.doc_id} ({s.form} {s.filed}): {s.state}, {s.chunks_done} chunk(s) done")
        for step in self.failed_steps:
            self.say(f"failed step: {step}")
        if failed or self.stalled or self.failed_steps:
            self.say(
                f"done with failures: {len(failed)} failed and {len(self.stalled)} stuck document(s), "
                f"{len(self.failed_steps)} step(s)"
            )
            return EXIT_FAILED
        self.say(f"done: {len(statuses)} document(s) ready")
        return EXIT_OK


def run_warm(
    settings: Settings,
    *,
    companies: str,
    forms: str | None,
    since: dt.date,
    until: dt.date | None,
    poll_interval_s: float | None,
    build: Callable[[Settings], Stack] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    err: TextIO | None = None,
) -> int:
    """The `warm` command. `build`, `sleep` and `clock` are for tests."""
    err = err or sys.stderr
    try:
        names = parse_companies(companies)
        wanted = parse_forms(forms, settings)
        if until is not None and until < since:
            raise UsageError("--until is before --since")
        interval = settings.warm_poll_interval_s if poll_interval_s is None else poll_interval_s
        if not (math.isfinite(interval) and 0 < interval <= WARM_POLL_MAX_S):
            raise UsageError(f"--poll-interval must be above 0 and at most {WARM_POLL_MAX_S} seconds")
        if not settings.sec_contact:
            raise UsageError("warm downloads from SEC, which needs MOF_SEC_CONTACT (e.g. an email address)")
        if settings.companies_source == "store":
            raise UsageError("MOF_COMPANIES_SOURCE=store reads nothing from SEC: warm needs auto or sec")
        if settings.cache == "off":
            raise UsageError("MOF_CACHE=off keeps nothing: warm needs the persistent cache (MOF_CACHE=on)")
    except UsageError as exc:
        print(f"myopenfund-mcp: {exc}", file=err)
        return EXIT_USAGE
    try:
        stack = (build or build_stack)(settings)
    except ConfigError as exc:
        print(f"myopenfund-mcp: configuration error: {exc}", file=err)
        return EXIT_USAGE
    except ImportError:
        print(
            "myopenfund-mcp: warm needs the extras: pip install 'myopenfund-mcp[companies,filings]'", file=err
        )
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("[warm] interrupted", file=err)
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 -- the class only: a message may hold a path or a host
        print(f"myopenfund-mcp: warm could not start ({type(exc).__name__})", file=err)
        return EXIT_FAILED
    code = _run(settings, stack, names, wanted, since, until, interval, sleep, clock, err)
    try:
        stack.close()  # the indexer waits at most MOF_INDEX_CLOSE_TIMEOUT_S for its workers
    except KeyboardInterrupt:
        print("[warm] interrupted while closing", file=err)
        return EXIT_INTERRUPTED
    return code


def _run(settings, stack, names, wanted, since, until, interval, sleep, clock, err) -> int:
    """The steps, every failure mapped to an exit code and a message that never holds the contact."""
    try:
        return Warmer(
            settings, stack, poll_interval_s=interval, sleep=sleep or time.sleep, clock=clock, err=err
        ).run(names, wanted, since, until)
    except UsageError as exc:
        print(f"myopenfund-mcp: {exc}", file=err)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("[warm] interrupted: closing the index (a document in progress resumes next run)", file=err)
        return EXIT_INTERRUPTED
    except ToolError:  # the companies could not be resolved: already said
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 -- last resort: the class only, never a message
        if any(c.__name__ == "IndexUnavailable" for c in type(exc).__mro__):
            print(
                "myopenfund-mcp: another process holds the filings index: stop the MCP server (or another "
                "warm) first, or use another MOF_CACHE_DIR",
                file=err,
            )
        else:
            print(f"myopenfund-mcp: warm failed ({type(exc).__name__})", file=err)
        return EXIT_FAILED
