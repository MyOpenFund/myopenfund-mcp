"""Disk cache and cached SEC access: offline, with a fake SecGet."""
from __future__ import annotations

import atexit
import os
import stat
import threading
from pathlib import Path

import pytest
from company_corpus.live.sec import SecNotFound

from myopenfund_mcp.config import Settings
from myopenfund_mcp.live.cache import DiskCache, cache_key, open_cache
from myopenfund_mcp.live.sec import CachedSec, build_sec

URL = "https://data.sec.gov/submissions/CIK0000320193.json"


class FakeSec:
    def __init__(self, body: bytes = b"body"):
        self.body, self.calls = body, []

    def get_bytes(self, url, *, max_age_s=None):
        self.calls.append((url, max_age_s))
        if "missing" in url:
            raise SecNotFound(url)
        return self.body


def _entry(cache: DiskCache, url: str) -> Path:
    key = cache_key(url)
    return cache.root / "http" / key[:2] / key


def test_hit_within_age_and_refetch_after_age(tmp_path):
    cache, fake = DiskCache(tmp_path, max_bytes=10**6), FakeSec()
    sec = CachedSec(fake, cache)
    assert sec.get_bytes(URL, max_age_s=3600) == b"body"
    assert sec.get_bytes(URL, max_age_s=3600) == b"body"
    assert len(fake.calls) == 1
    old = _entry(cache, URL)
    os.utime(old, (1, 1))  # an hour-plus old
    fake.body = b"new"
    assert sec.get_bytes(URL, max_age_s=3600) == b"new"
    assert len(fake.calls) == 2
    assert sec.get_bytes(URL, max_age_s=3600) == b"new"


def test_immutable_entry_never_refetched(tmp_path):
    cache, fake = DiskCache(tmp_path, max_bytes=10**6), FakeSec()
    sec = CachedSec(fake, cache)
    sec.get_bytes(URL, max_age_s=None)
    os.utime(_entry(cache, URL), (1, 1))
    sec.get_bytes(URL, max_age_s=None)
    assert len(fake.calls) == 1


def test_errors_are_not_cached(tmp_path):
    cache, fake = DiskCache(tmp_path, max_bytes=10**6), FakeSec()
    sec = CachedSec(fake, cache)
    for _ in range(2):
        with pytest.raises(SecNotFound):
            sec.get_bytes("https://x/missing", max_age_s=None)
    assert len(fake.calls) == 2
    assert not _entry(cache, "https://x/missing").exists()


def test_stray_temp_ignored_and_unreadable_entry_refetched(tmp_path):
    cache, fake = DiskCache(tmp_path, max_bytes=10**6), FakeSec()
    sec = CachedSec(fake, cache)
    entry = _entry(cache, URL)
    entry.parent.mkdir(parents=True)
    (entry.parent / ".tmp-truncated").write_bytes(b"bo")
    assert cache.get(cache_key(URL), None) is None
    entry.mkdir()  # unreadable as a file: read raises OSError
    assert sec.get_bytes(URL, max_age_s=None) == b"body"
    assert len(fake.calls) == 1


def test_eviction_oldest_first_keeps_models_and_index(tmp_path):
    for sub in ("models", "index"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "big").write_bytes(b"x" * 1000)
    cache = DiskCache(tmp_path, max_bytes=25)
    for i, name in enumerate(("a", "b", "c")):
        cache.put(cache_key(name), b"y" * 10)
        os.utime(_entry(cache, name), (100 + i, 100 + i))
    cache.evict()
    assert cache.size() <= 25
    assert cache.get(cache_key("a"), None) is None
    assert cache.get(cache_key("c"), None) == b"y" * 10
    assert (tmp_path / "models" / "big").exists() and (tmp_path / "index" / "big").exists()


def test_put_triggers_eviction_and_stray_temp_is_evictable(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=15)
    stray = tmp_path / "http" / "zz" / ".tmp-stray"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"s" * 10)
    os.utime(stray, (1, 1))
    cache.put(cache_key("a"), b"y" * 10)
    assert not stray.exists()
    assert cache.get(cache_key("a"), None) == b"y" * 10


def test_cache_off_dir_removed_at_exit(tmp_path, monkeypatch):
    hooks = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: hooks.append((fn, a, k)))
    cache = open_cache(Settings(cache="off", cache_dir=str(tmp_path / "never")))
    cache.put(cache_key("a"), b"x")
    assert not (tmp_path / "never").exists()
    assert cache.root.exists()
    for fn, a, k in hooks:
        fn(*a, **k)
    assert not cache.root.exists()


def test_open_cache_creates_private_dir(tmp_path):
    target = tmp_path / "c"
    cache = open_cache(Settings(cache_dir=str(target), cache_max_gb=3))
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert cache.root == target and cache.max_bytes == 3 * 1024**3


def test_two_threads_put_same_key(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10**6)
    data = b"z" * 100_000
    barrier = threading.Barrier(2)

    def work():
        barrier.wait()
        for _ in range(20):
            cache.put("k" * 64, data)
            assert cache.get("k" * 64, None) == data

    threads = [threading.Thread(target=work) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert cache.get("k" * 64, None) == data


def test_build_sec_none_without_contact_and_wired_with_it(tmp_path):
    assert build_sec(Settings(cache_dir=str(tmp_path))) is None
    sec = build_sec(Settings(sec_contact="me@example.com", cache_dir=str(tmp_path)))
    assert isinstance(sec, CachedSec)
