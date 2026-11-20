"""The deployment is reproducible: one pinned base image, exact PyPI versions, git deps by commit."""
import importlib.util
import re
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "compose.example.yaml").read_text()
CONSTRAINTS = (ROOT / "constraints.txt").read_text()
PYPROJECT = (ROOT / "pyproject.toml").read_text()
IMAGE = re.search(r"^ARG PYTHON_IMAGE=(\S+)$", DOCKERFILE, re.MULTILINE).group(1)


def test_base_image_is_pinned_by_tag_and_digest():
    assert re.fullmatch(r"python:3\.13\.\d+-slim-[a-z]+@sha256:[0-9a-f]{64}", IMAGE)


def test_locked_ci_job_runs_the_images_python():
    """The `locked` job stands for the deployed image: same Python, to the patch."""
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    locked = workflow.split("\n  locked:\n", 1)[1]
    version = re.search(r'python-version: "([^"]+)"', locked).group(1)
    assert version == re.match(r"python:(\d+\.\d+\.\d+)-", IMAGE).group(1)


def test_every_stage_uses_the_pinned_image():
    froms = re.findall(r"^FROM (\S+)", DOCKERFILE, re.MULTILINE)
    assert froms == ["${PYTHON_IMAGE}", "${PYTHON_IMAGE}"]


def test_compose_variant_uses_the_same_image():
    images = re.findall(r"^#?\s*image: (python\S*)$", COMPOSE, re.MULTILINE)
    assert images and set(images) == {IMAGE}


def test_image_and_install_line_use_the_constraints():
    assert "-c constraints.txt" in DOCKERFILE and "COPY pyproject.toml README.md LICENSE constraints.txt" in DOCKERFILE
    assert "-c https://raw.githubusercontent.com/MyOpenFund/myopenfund-mcp/<commit>/constraints.txt" in COMPOSE


def test_constraints_are_exact_pins_without_urls():
    lines = [line for line in CONSTRAINTS.splitlines() if line and not line.startswith("#")]
    assert lines
    bad = [line for line in lines if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+!-]+", line)]
    assert bad == []
    names = [line.split("==")[0].lower() for line in lines]
    assert names == sorted(names) and len(names) == len(set(names))
    assert not {"myopenfund-mcp", "vault-query", "my-open-data", "company-corpus", "rag-engine"} & set(names)


def test_git_dependencies_are_pinned_by_full_commit():
    refs = re.findall(r"git\+https://github\.com/[^\s\"']+", PYPROJECT)
    assert refs
    assert [r for r in refs if not re.search(r"@[0-9a-f]{40}(#|$)", r)] == []


BUILD_CONSTRAINTS = (ROOT / "build-constraints.txt").read_text()


def test_build_backends_are_pinned_and_used():
    lines = [line for line in BUILD_CONSTRAINTS.splitlines() if line and not line.startswith("#")]
    names = {line.split("==")[0].lower() for line in lines}
    assert all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==\S+", line) for line in lines)
    assert {"hatchling", "setuptools"} <= names  # this package and rag-engine/my-open-data; vault-query, company-corpus
    assert "--build-constraint build-constraints.txt" in DOCKERFILE and "build-constraints.txt ./" in DOCKERFILE
    assert ("--build-constraint https://raw.githubusercontent.com/MyOpenFund/myopenfund-mcp/<commit>/"
            "build-constraints.txt") in COMPOSE


def test_build_constraints_were_generated_from_the_current_git_pins():
    """A git dependency bumped without regenerating build-constraints.txt could bring another build
    backend that would then float: the file records the pins it was built from."""
    pins = re.findall(r"git\+https://github\.com/([^/@\s]+/[^/@\s]+)@([0-9a-f]{40})(?:#subdirectory=([^\s\"']+))?",
                      PYPROJECT)
    expected = sorted(f"{r}@{c}" + (f"#{sub}" if sub else "") for r, c, sub in set(pins))
    recorded = sorted(re.findall(r"^#   (\S+)$", BUILD_CONSTRAINTS, re.MULTILINE))
    assert recorded == expected, "git pins changed: run python3.13 scripts/update_constraints.py"


def test_editable_build_requirement_is_pinned():
    assert re.search(r"^editables==", BUILD_CONSTRAINTS, re.MULTILINE)


# --- the checker's own logic, on fake distributions ---------------------------------------------------
_spec = importlib.util.spec_from_file_location("check_constraints", ROOT / "scripts" / "check_constraints.py")
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


class FakeMetadata:
    def __init__(self, extras):
        self._extras = list(extras)

    def get_all(self, key):
        return self._extras if key == "Provides-Extra" else None


class FakeDist:
    def __init__(self, name, version, requires=(), extras=(), direct=False):
        self.name, self.version, self.requires, self._direct = name, version, list(requires), direct
        self.metadata = FakeMetadata(extras)

    def read_text(self, name):
        return "{}" if (name == "direct_url.json" and self._direct) else None


def _getter(dists):
    def get(name):
        try:
            return dists[name.lower().replace("_", "-")]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None
    return get


DISTS = {
    "myopenfund-mcp": FakeDist("myopenfund-mcp", "0.1.0",
                               ["fastmcp>=4", 'rag-engine; extra == "rag"', 'vault-query; extra == "vault"'],
                               extras=["rag", "vault"]),
    "fastmcp": FakeDist("fastmcp", "4.2.1", ["anyio>=4", 'pywin32; sys_platform == "win32"']),
    "anyio": FakeDist("anyio", "4.16.0"),
    "rag-engine": FakeDist("rag-engine", "0.1", ["onnxruntime[extra-a]"], direct=True),
    "onnxruntime": FakeDist("onnxruntime", "1.32.0", ['coloredlogs; extra == "extra-a"'], extras=["extra-a"]),
    "coloredlogs": FakeDist("coloredlogs", "15.0"),
    "vault-query": FakeDist("vault-query", "0.2", direct=True),
}


def test_walk_follows_extras_and_markers():
    tree = checker.walk("myopenfund-mcp", {"rag"}, get=_getter(DISTS))
    # reached only through the extra `rag` and, below it, through onnxruntime's own extra
    assert set(tree) == {"myopenfund-mcp", "fastmcp", "anyio", "rag-engine", "onnxruntime", "coloredlogs"}
    assert "vault-query" not in tree and "pywin32" not in tree


def test_check_reports_missing_and_mismatched_pins_and_skips_direct_urls():
    tree = checker.walk("myopenfund-mcp", {"rag", "vault"}, get=_getter(DISTS))
    pins = {"fastmcp": "4.2.1", "anyio": "4.15.0", "onnxruntime": "1.32.0"}
    assert checker.check(pins, tree) == [
        "anyio: installed 4.16.0, constraints.txt says 4.15.0",
        "coloredlogs 15.0: not in constraints.txt",
    ]


def test_unknown_extra_is_an_error():
    with pytest.raises(SystemExit, match="declares no extra nosuch"):
        checker.walk("myopenfund-mcp", {"nosuch"}, get=_getter(DISTS))


def test_read_constraints_refuses_a_range(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("# header\nFoo_Bar==1.0\n")
    assert checker.read_constraints(f) == {"foo-bar": "1.0"}
    f.write_text("foo>=1.0\n")
    with pytest.raises(SystemExit):
        checker.read_constraints(f)
