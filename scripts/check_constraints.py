"""Check that the running environment matches constraints.txt for every dependency of the server.

    python scripts/check_constraints.py [--extras vault,companies,rag]

Walks the installed requirement tree of myopenfund-mcp[extras] (markers evaluated for this interpreter)
and fails when a PyPI dependency is not listed in constraints.txt or is installed at another version.
`pip install -c` already refuses a listed package at another version; what it cannot see is a NEW
dependency (a release that adds one), which would float. Git dependencies are pinned by commit in
pyproject.toml and only walked through. Dev tools (pytest, ruff) are not part of the tree and float.
"""
from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
SELF = "myopenfund-mcp"


def read_constraints(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep or not version:
            raise SystemExit(f"constraints.txt: not an exact pin: {line!r}")
        pins[canonicalize_name(name)] = version.strip()
    return pins


def is_direct_url(dist: metadata.Distribution) -> bool:
    return dist.read_text("direct_url.json") is not None


def walk(root: str, extras: set[str], get=metadata.distribution) -> dict[str, metadata.Distribution]:
    """Every installed distribution reached from root[extras], keyed by canonical name.

    An extra of `root` that it does not declare (`Provides-Extra`) is an error: a renamed extra would
    otherwise make the check cover less than is deployed and still pass. `get` is injectable for tests.
    """
    declared = {canonicalize_name(e) for e in (get(root).metadata.get_all("Provides-Extra") or [])}
    unknown = sorted(e for e in extras if canonicalize_name(e) not in declared)
    if unknown:
        raise SystemExit(f"{root} declares no extra {', '.join(unknown)} (declared: {', '.join(sorted(declared))})")
    seen: dict[str, metadata.Distribution] = {}
    todo: list[tuple[str, set[str]]] = [(root, extras)]
    done: set[tuple[str, frozenset[str]]] = set()
    while todo:
        name, wanted = todo.pop()
        key = (canonicalize_name(name), frozenset(wanted))
        if key in done:
            continue
        done.add(key)
        try:
            dist = get(name)
        except metadata.PackageNotFoundError:
            raise SystemExit(f"{name} is required but not installed")
        seen[canonicalize_name(name)] = dist
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker is not None:
                envs = [dict(default_environment(), extra=e) for e in (wanted or {""})]
                if not any(req.marker.evaluate(env) for env in envs):
                    continue
            todo.append((req.name, set(req.extras)))
    return seen


def check(pins: dict[str, str], tree: dict[str, metadata.Distribution]) -> list[str]:
    """One line per dependency of the tree missing from the pins or installed at another version."""
    problems = []
    for name, dist in sorted(tree.items()):
        if name == canonicalize_name(SELF) or is_direct_url(dist):
            continue
        if name not in pins:
            problems.append(f"{name} {dist.version}: not in constraints.txt")
        elif pins[name] != dist.version:
            problems.append(f"{name}: installed {dist.version}, constraints.txt says {pins[name]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extras", default="vault,companies,rag")
    parser.add_argument("--constraints", default=str(ROOT / "constraints.txt"))
    args = parser.parse_args(argv)
    pins = read_constraints(Path(args.constraints))
    tree = walk(SELF, {e.strip() for e in args.extras.split(",") if e.strip()})
    problems = check(pins, tree)
    for line in problems:
        print(line, file=sys.stderr)
    print(f"{len(tree)} distributions checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
