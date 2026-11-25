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
import sys
import time
from collections.abc import Callable
from typing import TextIO

from fastmcp.exceptions import ToolError

from .companies.tools import check_companies, lookup_companies, unknown_company_hint
from .config import ConfigError, Settings, _forms
from .errors import (
    AMBIGUOUS_COMPANY,
    INVALID_ARGUMENT,
    LIMIT_EXCEEDED,
    TOO_MANY_COMPANIES,
    UNKNOWN_COMPANY,
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
    """The server's own construction (`server._build_server`), without the MCP layer: SEC only, since a
    warm run exists to read SEC (a company store needs no warming)."""
    from .companies.live import LiveSource
    from .companies.router import SourceRouter
    from .filings.indexer import FilingsIndexer
    from .live.sec import build_sec

    sec = build_sec(settings)
    live = LiveSource(sec, settings)
    router = SourceRouter(
        None,
        live,
        "sec",
        freshness_budget_s=settings.sec_freshness_budget_s,
        failure_ttl_s=settings.sec_failure_ttl_s,
        check_workers=settings.companies_max_concurrency,
    )
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
        err: TextIO | None = None,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ):
        self.settings, self.stack = settings, stack
        self.poll_interval_s, self._sleep = poll_interval_s, sleep
        self._err = err
        self._now = now
        self.failed_steps: list[str] = []

    def say(self, line: str) -> None:
        print(f"[warm] {line}", file=self._err or sys.stderr, flush=True)

    def resolve(self, companies: list[str]) -> list[dict]:
        router = self.stack.router

        async def call(fn, /, **kwargs):
            return fn(**kwargs)

        try:
            check_companies(companies, self.settings)
            return asyncio.run(
                lookup_companies(router, companies, call=call, unknown_hint=unknown_company_hint(router))
            )
        except ToolError as exc:
            if error_code(exc, "") in USAGE_CODES:
                raise UsageError(str(exc)) from None
            self.say(f"cannot resolve the companies: {exc}")
            raise

    def facts(self, hit: dict, since: dt.date, until: dt.date | None) -> None:
        """Build the issuer's tables and statement positions over the window (cached for later calls);
        `Warming` = still fetching: look again after the poll interval."""
        from .companies.live import NoFacts, Warming

        label = (hit.get("tickers") or [hit["cik"]])[0]
        while True:
            try:
                self.stack.router.concept_rows(hit["cik"], self._now(), period_from=since, period_to=until)
            except Warming as warming:
                self.say(f"{label} facts {warming.done}/{warming.total} requests")
                self._sleep(self.poll_interval_s)
            except NoFacts:
                self.say(f"{label} facts none (SEC has no XBRL facts for it)")
                return
            except Exception as exc:  # noqa: BLE001 -- one company's failure must not stop the others
                self.say(f"{label} facts failed ({error_code(exc, type(exc).__name__)})")
                self.failed_steps.append(f"{label} facts")
                return
            else:
                self.say(f"{label} facts ok")
                return

    def refs(self, hits: list[dict], forms: list[str], since: dt.date, until: dt.date | None) -> list:
        from .companies.live import Warming
        from .filings.indexer import filing_doc_id

        found: dict[str, object] = {}
        for hit in hits:
            label = (hit.get("tickers") or [hit["cik"]])[0]
            while True:
                try:
                    listed = self.stack.router.filings(
                        hit["cik"], forms=forms, period_from=since, period_to=until
                    )
                except Warming:
                    self._sleep(self.poll_interval_s)
                    continue
                except Exception as exc:  # noqa: BLE001
                    self.say(f"{label} filings failed ({error_code(exc, type(exc).__name__)})")
                    self.failed_steps.append(f"{label} filings")
                    break
                self.say(f"{label} filings {len(listed)}")
                for ref in listed:
                    found.setdefault(filing_doc_id(ref), ref)
                break
        return list(found.values())

    def index(self, refs: list, hits: list[dict]) -> list:
        """Enqueue what is not `ready` yet (batches of MOF_FILINGS_MAX_DOCS), then wait until every
        document is `ready` or `failed`; the statuses of all of them."""
        indexer = self.stack.indexer
        names = {h["cik"]: h["current_name"] for h in hits}
        tickers = {h["cik"]: h.get("tickers") or [] for h in hits}
        ids = [r.doc_id for r in refs]
        ready = {s.doc_id for s in indexer.status(ids) if s.state == "ready"}
        todo = [r for r in refs if r.doc_id not in ready]
        if ready:
            self.say(f"{len(ready)}/{len(refs)} documents already ready, skipped")
        step = self.settings.filings_max_docs
        for i in range(0, len(todo), step):
            indexer.enqueue(todo[i : i + step], names=names, tickers=tickers)
        last = None
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
        for s in failed:
            self.say(f"failed {s.doc_id} ({s.form} {s.filed}): {s.error or 'no detail'}")
        for step in self.failed_steps:
            self.say(f"failed step: {step}")
        if failed or self.failed_steps:
            self.say(f"done with failures: {len(failed)} document(s), {len(self.failed_steps)} step(s)")
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
    err: TextIO | None = None,
) -> int:
    """The `warm` command. `build` and `sleep` are for tests."""
    err = err or sys.stderr
    try:
        names = parse_companies(companies)
        wanted = parse_forms(forms, settings)
        if until is not None and until < since:
            raise UsageError("--until is before --since")
        interval = settings.warm_poll_interval_s if poll_interval_s is None else poll_interval_s
        if not interval > 0:
            raise UsageError("--poll-interval must be above 0 seconds")
        if not settings.sec_contact:
            raise UsageError("warm downloads from SEC, which needs MOF_SEC_CONTACT (e.g. an email address)")
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
    try:
        return Warmer(settings, stack, poll_interval_s=interval, sleep=sleep or time.sleep, err=err).run(
            names, wanted, since, until
        )
    except UsageError as exc:
        print(f"myopenfund-mcp: {exc}", file=err)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("[warm] interrupted: closing the index (a document in progress resumes next run)", file=err)
        return EXIT_INTERRUPTED
    except ToolError:  # the companies could not be resolved: already said
        return EXIT_FAILED
    finally:
        stack.close()  # the indexer waits at most MOF_INDEX_CLOSE_TIMEOUT_S for its workers
