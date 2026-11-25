"""The companies data computed live from SEC: the store's method set, without a store.

company-corpus builds one issuer's ``entity_observations``, ``fundamentals``, ``filings`` and
``filing_statements`` tables from SEC responses with the producer's own row functions
(``company_corpus.live.issuer_tables``); the concept selection, the summaries and the Q4 rule
are then exactly the store adapter's (``run_selection``, ``summaries_from_rows`` with
``sector_known=False`` and ``MOF_COMPANIES_Q4_RULE``). Nothing is computed here but the as-of
reading of those tables, with open-data-store's semantics: rows with ``known_at <= as_of``, the
latest per contract natural key (filters on natural-key columns before that choice, the others
after), or every row with ``vintages``.

Fetching: one issuer's tables are built in a worker thread (at most
``MOF_COMPANIES_MAX_CONCURRENCY`` at once, one build per CIK: concurrent callers wait on the same
one) and kept in memory for ``MOF_CACHE_TTL_COMPANYFACTS_H`` (at most ``MOF_SEC_MEMO_COMPANIES``
issuers, the oldest built dropped first). A caller waits up to ``MOF_SEC_CALL_BUDGET_S``, then
gets :class:`Warming` with the requests done and expected while the build goes on; a later call
gets the finished tables. Statement positions are fetched for the 10-K/10-Q/20-F/40-F filings
whose report date lies in a window: the caller's period, or ``MOF_SEC_POSITIONS_LOOKBACK_Y``
years before its end (before now when it has none). Tables built for a window serve any window
inside it (a wider one rebuilds over both), but a result never depends on what was built before:
the selection reads only the positions of the filings in the window asked for.

Errors: a 404 on the companyfacts file, or on the main submissions file (a CIK unknown to SEC) ->
:class:`NoFacts` (remembered like tables; each caller gets its own instance); a 404 on any other
non-positions file is a :class:`LiveDataError`. ``SecRefused``, ``SecContactMissing`` and
``SecUnavailable`` propagate as they are; an SEC response that cannot be read (company-corpus raises
a bare ``ValueError``, or a wrong-shape JSON a ``TypeError``/``KeyError``/``AttributeError``) or a
selection that fails on it -> :class:`LiveDataError`, a backend fault, never the caller's. A build
that fails is remembered for ``MOF_SEC_FAILURE_TTL_S``: callers meanwhile get its error (even one
that came after their budget, while they were answered :class:`Warming`), not a new build. After
:meth:`LiveSource.close` every call raises :class:`LiveClosed`, and a running build stops at its next
request. Argument errors use the store adapter's types (:class:`StoreBadArgument`,
:class:`StoreUnknownDataset`), checked before any request. The SEC contact is never logged here:
no message names it.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Collection
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime

from ..config import Q4_RULES, Settings
from .store import (
    StoreBadArgument,
    StoreUnknownDataset,
    _checked,
    _concept_catalog,
    _metric_catalog,
)

#: Requests company-corpus's ``filing_positions`` makes per filing (index.json, FilingSummary.xml,
#: the presentation linkbase): a progress estimate only, for :class:`Warming`.
REQUESTS_PER_FILING = 3
#: Requests known before the submissions are read: companyfacts and the main submissions file.
FIRST_REQUESTS = 2


class Warming(Exception):
    """The issuer is still being fetched; call again. ``done``/``total``: SEC requests."""

    def __init__(self, done: int, total: int):
        super().__init__(f"still fetching from SEC: {done}/{total} requests done")
        self.done, self.total = done, total


class NoFacts(LookupError):
    """SEC has no XBRL company facts for this CIK."""

    def __init__(self, cik: str):
        super().__init__(f"SEC has no XBRL facts for CIK {cik}")
        self.cik = cik


class LiveDataError(RuntimeError):
    """An SEC response that cannot be read, or a selection that failed on it: a backend fault."""


class LiveClosed(RuntimeError):
    """The live source was closed (server shutting down)."""

    def __init__(self):
        super().__init__("the live SEC source is closed")


@dataclass(frozen=True)
class _Window:
    """Report dates whose filings get statement positions: ``[start, end]``, ``end`` None = open."""

    start: date
    end: date | None

    def covers(self, other: _Window) -> bool:
        return self.start <= other.start and (
            self.end is None or (other.end is not None and other.end <= self.end)
        )

    def union(self, other: _Window) -> _Window:
        end = None if self.end is None or other.end is None else max(self.end, other.end)
        return _Window(min(self.start, other.start), end)


_EVERY_WINDOW = _Window(date.min, None)


class _Counting:
    """A SecGet counting the requests of one build. Bodies read while ``recording`` are kept for
    the build (the submissions, read once to size the work and again by ``issuer_tables``)."""

    def __init__(self, sec, closed: threading.Event):
        self._sec = sec
        self._closed = closed
        self._kept: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self.recording = True
        self.done = 0
        self.total = FIRST_REQUESTS
        self.requested: list[str] = []  # URLs actually requested, in order
        self.missing: str | None = None  # the URL of the last 404

    def get_bytes(self, url: str, *, max_age_s: float | None = None) -> bytes:
        from company_corpus.live.sec import SecNotFound

        if self._closed.is_set():  # close() stops the build at its next request
            raise LiveClosed
        with self._lock:
            kept = self._kept.get(url)
        if kept is not None:
            return kept
        with self._lock:
            self.requested.append(url)
        try:
            body = self._sec.get_bytes(url, max_age_s=max_age_s)
        except SecNotFound:
            with self._lock:
                self.missing = url
            raise
        finally:  # a request that failed was made all the same
            with self._lock:
                self.done += 1
        if self._closed.is_set():
            raise LiveClosed
        if self.recording:
            with self._lock:
                self._kept[url] = body
        return body


@dataclass
class _Job:
    window: _Window
    counting: _Counting
    future: Future | None = None


@dataclass
class _Entry:
    window: _Window
    at: float
    tables: object | None = None  # company_corpus.live.IssuerTables
    archive: object | None = None  # the submissions the tables were built from
    no_facts: bool = False


@contextmanager
def _data_faults():
    """company-corpus's parsing errors (malformed or wrong-shape JSON: ValueError, TypeError,
    KeyError, AttributeError) -> LiveDataError; SEC errors as they are."""
    from company_corpus.live.sec import SecContactMissing

    try:
        yield
    except SecContactMissing:  # a ValueError, but a configuration fault with its own code
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise LiveDataError(f"an SEC response could not be read ({type(exc).__name__})") from exc


def _years_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 February
        return day.replace(year=day.year - years, day=28)


def _is_cik(value) -> bool:
    return isinstance(value, str) and len(value) == 10 and value.isdigit()


def _aware(as_of, now: Callable[[], datetime]) -> datetime:
    """as_of checked; None = now, as open-data-store reads it."""
    if as_of is None:
        return now()
    if not isinstance(as_of, datetime) or as_of.tzinfo is None:
        raise StoreBadArgument("as_of must be a timezone-aware datetime")
    return as_of


class LiveSource:
    """The :class:`~myopenfund_mcp.companies.store.CompaniesStore` method set, from SEC.

    ``clock`` (monotonic seconds) and ``now`` (aware datetime) are injectable for tests."""

    def __init__(
        self,
        sec,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ):
        # ImportError here = company-corpus's live extra or open-data-store missing
        import company_corpus.concepts.store_summaries  # noqa: F401
        from company_corpus.config import Config
        from open_data_store.contracts.standard import FILING_STATEMENTS, FILINGS, FUNDAMENTALS

        if settings.companies_q4_rule not in Q4_RULES:
            raise ValueError(
                f"q4_rule must be one of {', '.join(Q4_RULES)}, got {settings.companies_q4_rule!r}"
            )
        self._sec = sec
        self._settings = settings
        self._clock = clock
        self._now = now or (lambda: datetime.now(UTC))
        # contact="" explicitly: the corpus Config would otherwise read COMPANY_CORPUS_CONTACT;
        # the row functions read only its store_* rules, and the SEC client carries the contact.
        self._config = Config(contact="")
        self._contracts = {FUNDAMENTALS.name: FUNDAMENTALS, FILINGS.name: FILINGS}
        self._contracts.update({c.table_name: c for c in (FUNDAMENTALS, FILINGS)})
        self._statements = FILING_STATEMENTS
        self._failure_ttl_s = settings.sec_failure_ttl_s
        self._facts_ttl_s = settings.cache_ttl_companyfacts_h * 3600
        self._submissions_ttl_s = settings.cache_ttl_submissions_h * 3600
        self._pool = ThreadPoolExecutor(
            max_workers=settings.companies_max_concurrency, thread_name_prefix="mof-live"
        )
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()  # oldest built first
        self._jobs: dict[str, _Job] = {}
        self._failures: dict[str, tuple[float, BaseException]] = {}  # last failed build per CIK
        # Submissions read per CIK (former names for entities(), filings, latest_filing): their own
        # memo, bounded like the tables (MOF_SEC_MEMO_COMPANIES, oldest read dropped) and kept
        # MOF_CACHE_TTL_SUBMISSIONS_H. A CIK dropped from it loses its former names in entities()
        # until its submissions are read again (resolution by ticker or current name still works).
        self._archives: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._tickers: tuple[float, list[dict]] | None = None
        self._closed = threading.Event()

    def __repr__(self) -> str:
        return "LiveSource()"

    def close(self) -> None:
        """Cancel queued builds, stop a running one at its next request; later calls raise
        :class:`LiveClosed`. Returns at once."""
        self._closed.set()
        with self._lock:
            self._jobs.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _check_open(self) -> None:
        if self._closed.is_set():
            raise LiveClosed

    # --- memo ------------------------------------------------------------------------------------
    def _fresh(self, at: float, ttl_s: float) -> bool:
        return self._clock() - at < ttl_s

    def _keep(self, memo: OrderedDict, key: str, value) -> None:
        """Insert as newest; drop the oldest beyond ``MOF_SEC_MEMO_COMPANIES``. Under the lock."""
        memo.pop(key, None)
        memo[key] = value
        while len(memo) > self._settings.sec_memo_companies:
            memo.popitem(last=False)

    def _remember_archive(self, cik: str, archive) -> None:
        with self._lock:
            self._keep(self._archives, cik, (self._clock(), archive))

    def _window(self, period_from: date | None, period_to: date | None) -> _Window:
        end = period_to
        start = period_from or _years_before(
            end or self._now().date(), self._settings.sec_positions_lookback_y
        )
        return _Window(start, end)

    def _build(self, cik: str, job: _Job):
        """Worker thread: fetch and build one issuer's tables, record the entry (or the failure),
        end the job."""
        from company_corpus.live.filings import fetch_submissions, list_filings
        from company_corpus.live.issuer import POSITIONS_FORMS, issuer_tables
        from company_corpus.live.sec import SecNotFound

        counting, w = job.counting, job.window
        first = None
        try:
            now = self._now()
            try:
                with _data_faults():
                    archive = fetch_submissions(cik, counting, max_age_s=self._submissions_ttl_s, now=now)
                    self._remember_archive(cik, archive)
                    window_filings = list_filings(
                        archive, forms=POSITIONS_FORMS, period_from=w.start, period_to=w.end
                    )
                    counting.total = 1 + counting.done + REQUESTS_PER_FILING * len(window_filings)
                    counting.recording = False
                    first = len(counting.requested)  # issuer_tables requests companyfacts first
                    tables = issuer_tables(
                        cik,
                        counting,
                        config=self._config,
                        now=now,
                        positions_from=w.start,
                        positions_to=w.end,
                        facts_max_age_s=self._facts_ttl_s,
                        submissions_max_age_s=self._submissions_ttl_s,
                        positions_forms=POSITIONS_FORMS,
                    )
            except SecNotFound:
                # Only the main submissions file (the build's first request: the CIK is unknown to
                # SEC) or the companyfacts file means "no facts"; positions 404s never get here.
                companyfacts = (
                    first is not None and len(counting.requested) > first and counting.requested[first]
                )
                if counting.missing not in (counting.requested[0], companyfacts):
                    raise LiveDataError("an SEC submissions page is missing (404)") from None
                with self._lock:
                    self._keep(self._entries, cik, _Entry(_EVERY_WINDOW, self._clock(), no_facts=True))
                    self._failures.pop(cik, None)
                raise NoFacts(cik) from None
            with self._lock:
                self._keep(self._entries, cik, _Entry(w, self._clock(), tables=tables, archive=archive))
                self._failures.pop(cik, None)
            return tables
        except NoFacts:
            raise
        except BaseException as exc:
            if not isinstance(exc, LiveClosed):
                with self._lock:
                    self._failures[cik] = (self._clock(), exc)
            raise
        finally:
            with self._lock:
                if self._jobs.get(cik) is job:
                    del self._jobs[cik]

    def _start(self, cik: str, window: _Window) -> _Job:
        """Under the lock: a build of ``cik`` over ``window``."""
        job = _Job(window, _Counting(self._sec, self._closed))
        self._jobs[cik] = job
        # set before the lock is released: the build's own clean-up takes the lock first
        job.future = self._pool.submit(self._build, cik, job)
        return job

    def _tables(self, ciks: Collection[str], window: _Window) -> dict[str, _Entry]:
        """Each CIK's entry (tables over at least ``window``), within the call budget."""
        deadline = self._clock() + self._settings.sec_call_budget_s
        out: dict[str, _Entry] = {}
        while True:
            waiting: list[_Job] = []
            with self._lock:
                self._check_open()
                for cik in ciks:
                    if cik in out:
                        continue
                    entry = self._entries.get(cik)
                    if entry is not None and not self._fresh(entry.at, self._facts_ttl_s):
                        entry = None
                    if entry is not None and entry.window.covers(window):
                        if entry.no_facts:
                            raise NoFacts(cik)
                        out[cik] = entry
                        continue
                    job = self._jobs.get(cik)
                    if job is None:
                        failed = self._failures.get(cik)
                        if failed is not None and self._fresh(failed[0], self._failure_ttl_s):
                            raise failed[1]  # nobody may have seen it: it came after their budget
                        wanted = window if entry is None else entry.window.union(window)
                        job = self._start(cik, wanted)
                    waiting.append(job)
            if not waiting:
                return out
            remaining = deadline - self._clock()
            pending = set()
            if remaining > 0:
                _, pending = wait([j.future for j in waiting], timeout=remaining)
            else:
                pending = {j.future for j in waiting if not j.future.done()}
            if pending:
                late = [j for j in waiting if j.future in pending]
                raise Warming(sum(j.counting.done for j in late), sum(j.counting.total for j in late))
            for j in waiting:
                if j.future.cancelled():
                    raise LiveClosed
                exc = j.future.exception()
                if isinstance(exc, NoFacts):
                    raise NoFacts(exc.cik)  # each caller its own instance
                if exc is not None:
                    raise exc
            # finished: the loop reads the entries (or rebuilds over a wider window)

    # --- as-of reading ---------------------------------------------------------------------------
    def _asof(self, table, contract, as_of: datetime, filters: dict, vintages: bool):
        """open-data-store's read_asof on one Arrow table: known_at <= as_of; natural-key filters,
        then the latest row per natural key (per key and known_at with vintages), then the
        other filters."""
        import pyarrow as pa
        import pyarrow.compute as pc

        known = pa.scalar(as_of, type=table.schema.field("known_at").type)
        table = table.filter(pc.less_equal(table["known_at"], known))
        nk = list(contract.natural_key)
        table = _apply(table, {c: v for c, v in filters.items() if c in nk})
        key = [*nk, "known_at"] if vintages else nk
        table = _latest(table, key)
        return _apply(table, {c: v for c, v in filters.items() if c not in nk})

    def _contract(self, dataset):
        contract = self._contracts.get(dataset) if isinstance(dataset, str) else None
        if contract is None:
            raise StoreUnknownDataset(f"unknown dataset {dataset!r} (live from SEC: fundamentals, filings)")
        return contract

    # --- the store's method set ------------------------------------------------------------------
    def entities(self) -> list[dict]:
        """Ticker rows of SEC's tickers file (kept ``MOF_CACHE_TTL_SUBMISSIONS_H``) and, for each
        issuer whose submissions were read, its dated names (former, then current); the ticker
        file's name of such an issuer is dropped (its ticker kept), so ``resolve.index`` does not
        list an undated first name. Copies."""
        from company_corpus.live.resolver import former_name_rows, resolver_rows

        self._check_open()
        with self._lock:
            tickers = self._tickers
        if tickers is None or not self._fresh(tickers[0], self._submissions_ttl_s):
            with _data_faults():
                rows = resolver_rows(self._sec, max_age_s=self._submissions_ttl_s)
            tickers = (self._clock(), rows)
            with self._lock:
                self._tickers = tickers
        with self._lock:
            archives = {
                cik: archive
                for cik, (at, archive) in self._archives.items()
                if self._fresh(at, self._submissions_ttl_s)
            }
        out = [{**r, "name": None} if r["subject_value"] in archives else dict(r) for r in tickers[1]]
        with _data_faults():
            for archive in archives.values():
                out += former_name_rows(archive)
        return out

    def read(
        self,
        dataset: str,
        *,
        as_of: datetime,
        filters: dict | None,
        columns: list[str] | None,
        limit: int | None,
        vintages: bool = False,
    ) -> list[dict]:
        """``CompaniesStore.read`` on the live tables of the CIKs named by the ``entity_value``
        filter (required: a live read is per issuer): rows knowable at ``as_of``, sorted by
        natural key then ``known_at``; ``limit`` and ``columns`` apply after the sort."""
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise StoreBadArgument("limit must be a non-negative int")
        if not isinstance(vintages, bool):
            raise StoreBadArgument("vintages must be a bool")
        contract = self._contract(dataset)
        description = {"columns": [{"name": f.name, "type": str(f.type)} for f in contract.schema]}
        checked = _checked(description, dataset, filters, columns) or {}
        as_of = _aware(as_of, self._now)
        if "entity_value" not in checked:
            raise StoreBadArgument(
                "a live read needs an entity_value filter (10-digit CIKs): data is fetched per company"
            )
        if "cik" not in checked.get("entity_scheme", ["cik"]):
            return []
        ciks = list(dict.fromkeys(v for v in checked["entity_value"] if _is_cik(v)))
        if len(ciks) > self._settings.companies_max_companies:
            raise StoreBadArgument(
                f"{len(ciks)} companies; at most {self._settings.companies_max_companies} per read"
            )
        if not ciks:
            return []
        import pyarrow as pa

        built = self._tables(ciks, self._window(None, None))
        parts = [
            self._asof(getattr(built[c].tables, contract.name), contract, as_of, checked, vintages)
            for c in ciks
        ]
        t = pa.concat_tables(parts)
        t = t.sort_by([(k, "ascending") for k in [*contract.natural_key, "known_at"]])
        if limit is not None:
            t = t.slice(0, limit)
        if columns:
            t = t.select(list(columns))
        return t.to_pylist()

    def concept_rows(
        self,
        cik: str,
        as_of: datetime,
        *,
        period_from: date | None = None,
        period_to: date | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """``(selection_rows, vintages)`` as ``CompaniesStore.concept_rows`` gives them, from the
        live tables; the period bounds the filings whose statement positions are fetched and read
        (only those: tables built earlier over a wider window give the same result)."""
        from company_corpus.concepts.selection import run_selection
        from company_corpus.live.filings import list_filings
        from company_corpus.live.issuer import POSITIONS_FORMS

        if not _is_cik(cik):
            raise StoreBadArgument("cik must be a 10-digit string")
        as_of = _aware(as_of, self._now)
        window = self._window(period_from, period_to)
        entry = self._tables([cik], window)[cik]
        t = entry.tables
        fundamentals = self._contracts["fundamentals"]
        facts = self._asof(t.fundamentals, fundamentals, as_of, {}, vintages=True)
        if facts.num_rows == 0:
            return [], []
        with _data_faults():
            in_window = [
                f.accession
                for f in list_filings(
                    entry.archive, forms=POSITIONS_FORMS, period_from=window.start, period_to=window.end
                )
            ]
        positions = self._asof(
            t.filing_statements, self._statements, as_of, {"filing_ref": in_window}, vintages=False
        )
        try:
            selected = run_selection(facts, positions)
        except Exception as exc:  # the SEC data, never the caller's arguments
            raise LiveDataError(f"concept selection failed ({type(exc).__name__})") from exc
        return selected, facts.to_pylist()

    def summaries(self, cik: str, name: str, rows: list[dict], vintages: list[dict] | None) -> list:
        """company-corpus period summaries, as ``CompaniesStore.summaries`` computes them (sector
        unknown, ``MOF_COMPANIES_Q4_RULE``)."""
        from company_corpus.concepts.store_summaries import summaries_from_rows

        try:
            return summaries_from_rows(
                rows,
                company=name or cik,
                sector_known=False,
                vintages=vintages,
                q4_rule=self._settings.companies_q4_rule,
            )
        except ValueError as exc:
            raise StoreBadArgument(str(exc)) from None

    def concept_catalog(self) -> list[dict]:
        return [dict(c) for c in _concept_catalog()]

    def metric_catalog(self) -> list[dict]:
        return [dict(m) for m in _metric_catalog()]

    # --- filings (submissions only, no tables) ---------------------------------------------------
    def _archive(self, cik: str):
        """The issuer's submissions (kept ``MOF_CACHE_TTL_SUBMISSIONS_H``); None when SEC does not
        know the CIK (404 on the main file)."""
        from company_corpus.live.filings import fetch_submissions
        from company_corpus.live.sec import SecNotFound

        if not _is_cik(cik):
            raise StoreBadArgument("cik must be a 10-digit string")
        self._check_open()
        with self._lock:
            kept = self._archives.get(cik)
        if kept is not None and self._fresh(kept[0], self._submissions_ttl_s):
            return kept[1]
        counting = _Counting(self._sec, self._closed)
        try:
            with _data_faults():
                archive = fetch_submissions(cik, counting, max_age_s=self._submissions_ttl_s, now=self._now())
        except SecNotFound:
            if counting.missing == counting.requested[0]:  # the main file: a CIK unknown to SEC
                return None
            raise LiveDataError("an SEC submissions page is missing (404)") from None
        self._remember_archive(cik, archive)
        return archive

    def filings(
        self,
        cik: str,
        *,
        forms: Collection[str] | None,
        period_from: date | None,
        period_to: date | None,
    ) -> list:
        """The issuer's filings (``FilingRef``), newest accepted first; ``[]`` for an unknown CIK."""
        from company_corpus.live.filings import list_filings

        archive = self._archive(cik)
        if archive is None:
            return []
        with _data_faults():
            return list_filings(archive, forms=forms, period_from=period_from, period_to=period_to)

    def latest_filing(self, cik: str):
        """The newest filing of any form (``FilingRef``), or None."""
        found = self.filings(cik, forms=None, period_from=None, period_to=None)
        return found[0] if found else None


def _apply(table, filters: dict):
    """Rows whose every filtered column is in its values (exact match, as SQL IN)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    for col, values in filters.items():
        column = table[col]
        try:
            mask = pc.is_in(column, value_set=pa.array(values, type=column.type))
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, OverflowError, TypeError):
            mask = pa.array([v is not None and v in values for v in column.to_pylist()])
        table = table.filter(mask)
    return table


def _latest(table, key: list[str]):
    """The row of highest ``known_at`` per ``key`` (nulls group together, as SQL PARTITION BY)."""
    if table.num_rows == 0:
        return table
    order = [(k, "ascending") for k in key if k != "known_at"]
    table = table.sort_by([*order, ("known_at", "descending")])
    cols = [table[k].to_pylist() for k in key]
    keep, previous = [], object()
    for i, current in enumerate(zip(*cols, strict=True)):
        if current != previous:
            keep.append(i)
            previous = current
    return table.take(keep)
