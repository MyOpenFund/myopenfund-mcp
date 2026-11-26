"""Technical disk cache of SEC responses.

Entries live under ``root/http/<2-char prefix>/<sha256 of the URL>``; an entry's age is its file
mtime. An entry is a small header (magic, body length, sha256 of the body) followed by the body, so a
truncated or zeroed file is detected and dropped. Writes are atomic (temp file in the same folder,
fsync, ``os.replace``). Eviction removes the oldest-written files, only under ``root/http``: the model
files (``root/models``) and the local index (``root/index``) are never evicted and are not counted
against ``MOF_CACHE_MAX_GB``, which covers the HTTP bodies only.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from pathlib import Path

from ..config import ConfigError, Settings

_TMP_PREFIX = ".tmp-"
_GB = 1024**3
_MAGIC = b"MOFC1\n"
_HEADER = len(_MAGIC) + 8 + 32
_KEY = re.compile(r"[0-9a-f]{64}")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _check_key(key: str) -> None:
    if _KEY.fullmatch(key) is None:
        raise ValueError("a cache key is the 64-hex sha256 of a URL")


def _encode(data: bytes) -> bytes:
    return _MAGIC + len(data).to_bytes(8, "big") + hashlib.sha256(data).digest() + data


def _decode(blob: bytes) -> bytes | None:
    if len(blob) < _HEADER or not blob.startswith(_MAGIC):
        return None
    start = len(_MAGIC)
    length = int.from_bytes(blob[start : start + 8], "big")
    body = blob[_HEADER:]
    if len(body) != length or hashlib.sha256(body).digest() != blob[start + 8 : _HEADER]:
        return None
    return body


class DiskCache:
    def __init__(self, root: Path, *, max_bytes: int, evict_to: float = 0.9, tmp_grace_s: float = 3600):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.evict_to = evict_to
        self.tmp_grace_s = tmp_grace_s
        self._http = self.root / "http"
        self._lock = threading.Lock()  # guards _total and serialises evictions
        self._total = sum(size for _m, size, _p in self._files())  # one walk; re-walked when evicting

    def _path(self, key: str) -> Path:
        return self._http / key[:2] / key

    def _add(self, delta: int) -> None:
        with self._lock:
            self._total = max(0, self._total + delta)

    def _drop(self, path: Path) -> None:
        """Remove whatever sits at an entry path (file, symlink or folder) so the entry heals."""
        try:
            st = path.lstat()
            if stat.S_ISDIR(st.st_mode):
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
                self._add(-st.st_size)
        except OSError:
            pass

    def get(self, key: str, max_age_s: float | None) -> bytes | None:
        """The cached body, or None (absent, stale, or corrupt: a corrupt entry is removed).
        A negative age (mtime in the future) is stale; ``max_age_s=None`` never expires."""
        _check_key(key)
        path = self._path(key)
        try:
            if path.parent.resolve() != self.root.resolve() / "http" / key[:2]:
                return None  # a symlinked folder leading elsewhere: not followed
            fd = os.open(path, os.O_RDONLY | _NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError:
            self._drop(path)
            return None
        try:
            with os.fdopen(fd, "rb") as handle:
                st = os.fstat(handle.fileno())
                if not stat.S_ISREG(st.st_mode):
                    blob = None
                elif max_age_s is not None and not 0 <= time.time() - st.st_mtime <= max_age_s:
                    return None
                else:
                    blob = handle.read()
        except OSError:
            blob = None
        body = None if blob is None else _decode(blob)
        if body is None:
            self._drop(path)
        return body

    def put(self, key: str, data: bytes) -> None:
        _check_key(key)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = _encode(data)
        try:
            self._write(path, blob)
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            self.evict()
            self._write(path, blob)
        with self._lock:
            over = self._total > self.max_bytes
        if over:
            self.evict()

    def _write(self, path: Path, blob: bytes) -> None:
        fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                old = path.lstat().st_size if path.is_file() else 0
            except OSError:
                old = 0
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._add(len(blob) - old)

    def _files(self) -> list[tuple[float, int, Path]]:
        found = []
        for dirpath, _dirs, names in os.walk(self._http):
            for name in names:
                path = Path(dirpath) / name
                try:
                    st = path.lstat()
                except OSError:
                    continue
                found.append((st.st_mtime, st.st_size, path))
        return found

    def size(self) -> int:
        """Bytes held under ``root/http`` (temp files included), from the running total."""
        with self._lock:
            return self._total

    def evict(self) -> None:
        """Re-walk ``root/http`` and remove the oldest-written files until the total is at most
        ``max_bytes * evict_to``. A temp file younger than ``tmp_grace_s`` (another writer's) is kept."""
        with self._lock:
            files = sorted(self._files())
            total = sum(size for _m, size, _p in files)
            target = self.max_bytes * self.evict_to
            now = time.time()
            for _mtime, size, path in files:
                if total <= target:
                    break
                if path.name.startswith(_TMP_PREFIX) and now - _mtime <= self.tmp_grace_s:
                    continue
                try:
                    st = path.lstat()  # re-stat: another evictor may have removed or replaced it
                    path.unlink()
                    total -= st.st_size
                except FileNotFoundError:
                    total -= size
                except OSError:
                    continue
            self._total = max(0, total)


def _check_existing_root(root: Path) -> None:
    """Refuse a cache folder another user owns or others can write to (POSIX)."""
    if not hasattr(os, "getuid"):
        return
    st = root.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise ConfigError(
            "the cache folder must belong to the user running the server and not be writable by group or "
            "others: fix its owner and permissions (chmod 700) or set MOF_CACHE_DIR to another folder"
        )


def open_cache(settings: Settings) -> DiskCache:
    """The cache of these settings. ``MOF_CACHE=off``: a temporary folder removed at exit."""
    options = {
        "max_bytes": settings.cache_max_gb * _GB,
        "evict_to": settings.cache_evict_to,
        "tmp_grace_s": settings.cache_tmp_grace_s,
    }
    if settings.cache == "off":
        tmp = tempfile.TemporaryDirectory(prefix="myopenfund-mcp-")
        atexit.register(tmp.cleanup)
        return DiskCache(Path(tmp.name), **options)
    root = Path(settings.cache_path())
    if root.exists():
        _check_existing_root(root)
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(mode=0o700, exist_ok=True)
    return DiskCache(root, **options)
