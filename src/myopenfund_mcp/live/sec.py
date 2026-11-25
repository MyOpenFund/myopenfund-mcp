"""Cached SEC access: only successful bodies are kept; errors always pass through."""

from __future__ import annotations

import logging

from company_corpus.live.sec import SecClient, SecGet

from ..config import ConfigError, Settings
from .cache import DiskCache, cache_key, open_cache

log = logging.getLogger(__name__)


class CachedSec:
    """A :class:`SecGet` in front of another one. ``max_age_s=None`` = immutable: never refetched."""

    def __init__(self, client: SecGet, cache: DiskCache):
        self._client = client
        self._cache = cache

    @property
    def cache(self) -> DiskCache:
        """The disk cache in use (its folder and size are reported by server_info)."""
        return self._cache

    def get_bytes(self, url: str, *, max_age_s: float | None = None) -> bytes:
        key = cache_key(url)
        hit = self._cache.get(key, max_age_s)
        if hit is not None:
            return hit
        body = self._client.get_bytes(url, max_age_s=max_age_s)
        try:
            self._cache.put(key, body)
        except OSError as exc:  # a full or read-only disk must not fail a fetch that succeeded
            log.warning("cache write failed: %s", type(exc).__name__)
        return body


def build_sec(settings: Settings) -> CachedSec | None:
    """None without a contact (no SEC access); otherwise the cached client."""
    if not settings.sec_contact:
        return None
    client = SecClient(
        settings.sec_contact,
        max_rps=settings.sec_max_rps,
        timeout_s=settings.sec_timeout_s,
        retries=settings.sec_retries,
    )
    try:
        cache = open_cache(settings)
    except (
        OSError
    ) as exc:  # the folder cannot be created or read; the error text may name a path, not the contact
        raise ConfigError(
            f"the cache folder cannot be used ({type(exc).__name__}): set MOF_CACHE_DIR to a "
            "folder you can write to, or MOF_CACHE=off"
        ) from exc
    return CachedSec(client, cache)
