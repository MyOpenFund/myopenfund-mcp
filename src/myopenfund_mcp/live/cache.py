"""Technical disk cache of SEC responses.

Entries live under ``root/http/<2-char prefix>/<sha256 of the URL>``; an entry's age is its file
mtime. Writes are atomic (temp file in the same folder + ``os.replace``). Eviction only ever touches
``root/http``: the model files (``root/models``) and the local index (``root/index``) are never evicted.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path

from ..config import Settings

_TMP_PREFIX = ".tmp-"
_GB = 1024**3


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path, *, max_bytes: int):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._http = self.root / "http"
        self._evict_lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self._http / key[:2] / key

    def get(self, key: str, max_age_s: float | None) -> bytes | None:
        """The cached body, or None (absent, older than ``max_age_s``, or unreadable)."""
        path = self._path(key)
        try:
            if max_age_s is not None and time.time() - path.stat().st_mtime > max_age_s:
                return None
            return path.read_bytes()
        except OSError:
            return None

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        if self.size() > self.max_bytes:
            self.evict()

    def _files(self) -> list[tuple[float, int, Path]]:
        found = []
        for dirpath, _dirs, names in os.walk(self._http):
            for name in names:
                path = Path(dirpath) / name
                try:
                    st = path.stat()
                except OSError:
                    continue
                found.append((st.st_mtime, st.st_size, path))
        return found

    def size(self) -> int:
        """Bytes held under ``root/http`` (temp files included)."""
        return sum(size for _m, size, _p in self._files())

    def evict(self) -> None:
        """Remove the oldest files (by mtime) until the total is within ``max_bytes``."""
        with self._evict_lock:
            files = sorted(self._files())
            total = sum(size for _m, size, _p in files)
            for _mtime, size, path in files:
                if total <= self.max_bytes:
                    break
                try:
                    path.unlink()
                except OSError:
                    continue
                total -= size


def open_cache(settings: Settings) -> DiskCache:
    """The cache of these settings. ``MOF_CACHE=off``: a temporary folder removed at exit."""
    max_bytes = settings.cache_max_gb * _GB
    if settings.cache == "off":
        tmp = tempfile.TemporaryDirectory(prefix="myopenfund-mcp-")
        atexit.register(tmp.cleanup)
        return DiskCache(Path(tmp.name), max_bytes=max_bytes)
    root = Path(settings.cache_path())
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(mode=0o700, exist_ok=True)
    return DiskCache(root, max_bytes=max_bytes)
