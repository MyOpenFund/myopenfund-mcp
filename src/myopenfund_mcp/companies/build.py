"""Construction shared by the server and `myopenfund-mcp warm`: one place for the parameters, so the
objects a warm run fills are built exactly like the ones the tools read. Owners close what they build."""

from __future__ import annotations

from ..config import Settings


def build_live(settings: Settings):
    """``(sec, live)``: the cached SEC client and the live source over it. ImportError = the
    ``[companies]`` extra is missing. Needs ``settings.sec_contact``."""
    from ..live.sec import build_sec
    from .live import LiveSource

    sec = build_sec(settings)
    return sec, LiveSource(sec, settings)


def build_router(store, live, settings: Settings):
    """The source router of these settings. ValueError = a source the mode needs is missing."""
    from .router import SourceRouter

    return SourceRouter(
        store,
        live,
        settings.companies_source,
        freshness_budget_s=settings.sec_freshness_budget_s,
        failure_ttl_s=settings.sec_failure_ttl_s,
        check_workers=settings.companies_max_concurrency,
    )
