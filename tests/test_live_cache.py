"""Disk cache and cached SEC access: offline, with a fake SecGet."""

from __future__ import annotations

import atexit
import os
import stat
import threading
from pathlib import Path

import pytest
from company_corpus.live.sec import SecNotFound

from myopenfund_mcp.config import ConfigError, Settings
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
    cache = DiskCache(tmp_path, max_bytes=120)
    for i, name in enumerate(("a", "b", "c")):
        cache.put(cache_key(name), b"y" * 10)
        os.utime(_entry(cache, name), (100 + i, 100 + i))
    cache.evict()
    assert cache.size() <= 120
    assert cache.get(cache_key("a"), None) is None
    assert cache.get(cache_key("c"), None) == b"y" * 10
    assert (tmp_path / "models" / "big").exists() and (tmp_path / "index" / "big").exists()


def test_put_triggers_eviction_and_stray_temp_is_evictable(tmp_path):
    stray = tmp_path / "http" / "zz" / ".tmp-stray"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"s" * 10)
    os.utime(stray, (1, 1))
    cache = DiskCache(tmp_path, max_bytes=60, evict_to=1.0)
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
            cache.put("a" * 64, data)
            assert cache.get("a" * 64, None) == data

    threads = [threading.Thread(target=work) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert cache.get("a" * 64, None) == data


def test_build_sec_none_without_contact_and_wired_with_it(tmp_path):
    assert build_sec(Settings(cache_dir=str(tmp_path))) is None
    sec = build_sec(Settings(sec_contact="me@example.com", cache_dir=str(tmp_path)))
    assert isinstance(sec, CachedSec)


def _fill(cache, n, size=10):
    for i in range(n):
        cache.put(cache_key(str(i)), b"y" * size)
        os.utime(_entry(cache, str(i)), (100 + i, 100 + i))


def test_put_uses_running_total_without_walking(tmp_path, monkeypatch):
    cache = DiskCache(tmp_path, max_bytes=10**6)
    _fill(cache, 3)
    monkeypatch.setattr(cache, "_files", lambda: pytest.fail("tree walked"))
    cache.put(cache_key("new"), b"x")
    assert cache.size() > 0


def test_eviction_goes_down_to_low_water_mark(tmp_path):
    entry = len(b"y" * 10) + 6 + 8 + 32
    cache = DiskCache(tmp_path, max_bytes=entry * 10, evict_to=0.5)
    _fill(cache, 11)
    assert cache.size() <= entry * 5
    assert cache.get(cache_key("10"), None) == b"y" * 10  # newest kept
    assert cache.get(cache_key("0"), None) is None


def test_failed_unlink_of_vanished_file_still_reduces_total(tmp_path):
    entry = 10 + 6 + 8 + 32
    cache = DiskCache(tmp_path, max_bytes=entry * 4, evict_to=0.5)
    _fill(cache, 3)
    for i in range(3):  # another evictor already removed everything
        _entry(cache, str(i)).unlink()
    cache.put(cache_key("late"), b"y" * 10)
    cache.put(cache_key("late2"), b"y" * 10)
    cache.put(cache_key("late3"), b"y" * 10)
    assert cache.get(cache_key("late3"), None) == b"y" * 10  # nothing over-evicted


@pytest.mark.parametrize("damage", ["empty", "truncated", "garbage", "flipped"])
def test_corrupt_entry_is_dropped_and_heals(tmp_path, damage):
    cache, fake = DiskCache(tmp_path, max_bytes=10**6), FakeSec(b"payload")
    sec = CachedSec(fake, cache)
    sec.get_bytes(URL, max_age_s=None)
    entry = _entry(cache, URL)
    blob = entry.read_bytes()
    entry.write_bytes(
        {
            "empty": b"",
            "truncated": blob[:-3],
            "garbage": b"x" * 80,
            "flipped": blob[:-1] + bytes([blob[-1] ^ 1]),
        }[damage]
    )
    assert cache.get(cache_key(URL), None) is None
    assert not entry.exists()
    assert sec.get_bytes(URL, max_age_s=None) == b"payload"
    assert len(fake.calls) == 2
    assert sec.get_bytes(URL, max_age_s=None) == b"payload"
    assert len(fake.calls) == 2


def test_directory_at_entry_path_is_repaired(tmp_path):
    cache, fake = DiskCache(tmp_path, max_bytes=10**6), FakeSec()
    sec = CachedSec(fake, cache)
    entry = _entry(cache, URL)
    entry.mkdir(parents=True)
    assert sec.get_bytes(URL, max_age_s=None) == b"body"
    sec.get_bytes(URL, max_age_s=None)
    assert len(fake.calls) == 1 and entry.is_file()


def test_symlinked_entry_outside_root_is_not_followed(tmp_path):
    root = tmp_path / "root"
    cache = DiskCache(root, max_bytes=10**6)
    outside = tmp_path / "secret"
    outside.write_bytes(b"outside")
    entry = _entry(cache, URL)
    entry.parent.mkdir(parents=True)
    entry.symlink_to(outside)
    assert cache.get(cache_key(URL), None) is None
    assert outside.read_bytes() == b"outside" and not entry.is_symlink()


def test_symlinked_prefix_folder_is_not_followed(tmp_path):
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    cache = DiskCache(root, max_bytes=10**6)
    elsewhere.mkdir()
    (root / "http").mkdir(parents=True)
    (root / "http" / cache_key(URL)[:2]).symlink_to(elsewhere)
    (elsewhere / cache_key(URL)).write_bytes(_encode_for_test(b"x"))
    assert cache.get(cache_key(URL), None) is None


def _encode_for_test(data):
    from myopenfund_mcp.live.cache import _encode

    return _encode(data)


def test_future_mtime_is_stale_but_immutable_ignores_age(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10**6)
    cache.put(cache_key("a"), b"v")
    os.utime(_entry(cache, "a"), (4_000_000_000, 4_000_000_000))
    assert cache.get(cache_key("a"), 3600) is None
    assert cache.get(cache_key("a"), None) == b"v"


def test_fresh_temp_of_another_writer_survives_eviction(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10, evict_to=0.5, tmp_grace_s=3600)
    fresh = tmp_path / "http" / "zz" / ".tmp-fresh"
    fresh.parent.mkdir(parents=True)
    fresh.write_bytes(b"s" * 50)
    cache.put(cache_key("a"), b"y" * 10)
    assert fresh.exists()
    assert cache.get(cache_key("a"), None) is None  # evicted: it was the only evictable file


def test_enospc_evicts_once_and_retries(tmp_path, monkeypatch):
    cache = DiskCache(tmp_path, max_bytes=10**6)
    real, calls = cache._write, []

    def flaky(path, blob):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(28, "No space left on device")
        return real(path, blob)

    monkeypatch.setattr(cache, "_write", flaky)
    cache.put(cache_key("a"), b"v")
    assert len(calls) == 2 and cache.get(cache_key("a"), None) == b"v"


def test_bad_key_refused(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10)
    for bad in ("../x", "abc", "A" * 64):
        with pytest.raises(ValueError):
            cache.get(bad, None)
        with pytest.raises(ValueError):
            cache.put(bad, b"")


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX permissions")
def test_existing_group_writable_root_refused(tmp_path):
    target = tmp_path / "c"
    target.mkdir()
    target.chmod(0o770)
    with pytest.raises(ConfigError, match="MOF_CACHE_DIR"):
        open_cache(Settings(cache_dir=str(target)))
    target.chmod(0o700)
    open_cache(Settings(cache_dir=str(target)))


def test_build_sec_unusable_folder_is_a_config_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ConfigError) as err:
        build_sec(Settings(sec_contact="me@example.com", cache_dir=str(blocker / "sub")))
    assert "me@example.com" not in str(err.value)
