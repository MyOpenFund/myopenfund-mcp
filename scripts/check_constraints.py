"""Check that the running environment matches constraints.txt for every dependency of the server.

    python scripts/check_constraints.py [--extras vault,companies,rag,filings,market]
    python scripts/check_constraints.py --build


Walks the installed requirement tree of myopenfund-mcp[extras] (markers evaluated for this interpreter)
and fails when a PyPI dependency is not listed in constraints.txt or is installed at another version.
`pip install -c` already refuses a listed package at another version; what it cannot see is a NEW
dependency (a release that adds one), which would float. Git dependencies are pinned by commit in
pyproject.toml and only walked through. Dev tools (pytest, ruff) are not part of the tree and float.

`--build` checks build-constraints.txt instead: every `[build-system] requires` of this package and of
each git dependency at its pinned commit (read from GitHub, as scripts/update_constraints.py does) must
be pinned there, at a version its specifier allows. A git dependency bumped to a commit that wants
another backend, or a newer one, fails here instead of floating in the isolated build environment.
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


def check_build(pins: dict[str, str], requirements: list[str]) -> list[str]:
    """One line per build requirement missing from the pins or pinned outside its specifier; a requirement
    whose environment marker is false on this interpreter (the image's Python) is skipped."""
    problems = []
    for raw in requirements:
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate():  # not installed on this interpreter
            continue
        name = canonicalize_name(req.name)
        if name not in pins:
            problems.append(f"{name} ({raw}): not in build-constraints.txt")
        elif not req.specifier.contains(pins[name], prereleases=True):
            problems.append(f"{name}: build-constraints.txt says {pins[name]}, outside {raw}")
    return sorted(problems)


def _build_requires() -> list[str]:
    """scripts/update_constraints.py's build_requires(): one source for what the generator pins."""
    import importlib.util

    path = ROOT / "scripts" / "update_constraints.py"
    spec = importlib.util.spec_from_file_location("update_constraints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_requires()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extras", default="vault,companies,rag,filings,market")
    parser.add_argument("--constraints", default=str(ROOT / "constraints.txt"))
    parser.add_argument("--build", action="store_true",
                        help="check build-constraints.txt against the build requirements instead")
    args = parser.parse_args(argv)
    if args.build:
        requirements = _build_requires()
        problems = check_build(read_constraints(ROOT / "build-constraints.txt"), requirements)
        for line in problems:
            print(line, file=sys.stderr)
        print(f"{len(requirements)} build requirements checked, {len(problems)} problem(s)")
        return 1 if problems else 0
    pins = read_constraints(Path(args.constraints))
    tree = walk(SELF, {e.strip() for e in args.extras.split(",") if e.strip()})
    problems = check(pins, tree)
    for line in problems:
        print(line, file=sys.stderr)
    print(f"{len(tree)} distributions checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
