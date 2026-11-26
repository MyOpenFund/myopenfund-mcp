"""The deployment is reproducible: one pinned base image and exact PyPI versions (siblings included)."""
import importlib.util
import re
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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
    assert "-c https://raw.githubusercontent.com/MyOpenFund/myopenfund-mcp/v<version>/constraints.txt" in COMPOSE


def test_constraints_are_exact_pins_without_urls():
    lines = [line for line in CONSTRAINTS.splitlines() if line and not line.startswith("#")]
    assert lines
    bad = [line for line in lines if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+!-]+", line)]
    assert bad == []
    names = [line.split("==")[0].lower() for line in lines]
    assert names == sorted(names) and len(names) == len(set(names))
    assert "myopenfund-mcp" not in names
    # the old, unprefixed names are not the published distributions
    assert not {"vault-query", "open-data-store", "company-corpus", "rag-engine"} & set(names)


def test_no_git_urls_anywhere():
    """Every package of the deployment, ours included, is an ordinary PyPI pin."""
    readme_install_lines = [line for line in (ROOT / "README.md").read_text().splitlines()
                            if re.search(r"pip install|uvx|pip uninstall", line)]
    # The one documented exception: the install line for the latest `main`, of this package only
    # (its dependencies still come from PyPI).
    latest_main = 'pip install "myopenfund-mcp @ git+https://github.com/MyOpenFund/myopenfund-mcp"'
    readme_install_lines = [line for line in readme_install_lines if latest_main not in line]
    texts = {
        "pyproject.toml": PYPROJECT,
        "constraints.txt": CONSTRAINTS,
        "build-constraints.txt": (ROOT / "build-constraints.txt").read_text(),
        "Dockerfile": DOCKERFILE,
        "compose.example.yaml": COMPOSE,
        "README install lines": "\n".join(readme_install_lines),
    }
    for name, text in texts.items():
        assert not re.search(r"git\+", text), name
    assert "allow-direct-references" not in PYPROJECT


def _specifiers():
    """The sibling requirements of pyproject.toml (every extra), by canonical name."""
    import tomllib

    extras = tomllib.loads(PYPROJECT)["project"]["optional-dependencies"]
    reqs = [Requirement(d) for deps in extras.values() for d in deps]
    return {canonicalize_name(r.name): r for r in reqs if r.name.startswith("myopenfund-")}


def test_constraints_pin_siblings_inside_ranges():
    siblings = _specifiers()
    assert set(siblings) == {"myopenfund-vault-query", "myopenfund-open-data-store",
                             "myopenfund-company-corpus", "myopenfund-rag-engine"}
    pins = checker.read_constraints(ROOT / "constraints.txt")
    for name, req in siblings.items():
        assert name in pins, f"{name} is not pinned in constraints.txt"
        assert req.specifier.contains(pins[name]), f"{name}=={pins[name]} is outside {req.specifier}"
    for line in CONSTRAINTS.splitlines():
        if line.startswith("myopenfund-"):
            assert re.fullmatch(r"myopenfund-[a-z-]+==\d+(\.\d+)*", line), line


def test_constraints_cover_every_runtime_dependency():
    """Every direct dependency of the server (core and runtime extras, siblings included) is pinned.
    The transitive closure is `scripts/check_constraints.py`'s job, run by the `locked` CI job on the
    image's Python (it depends on the interpreter's markers)."""
    import tomllib

    project = tomllib.loads(PYPROJECT)["project"]
    deps = list(project["dependencies"])
    for extra, extra_deps in project["optional-dependencies"].items():
        if extra != "dev":
            deps += extra_deps
    pins = checker.read_constraints(ROOT / "constraints.txt")
    missing = sorted({canonicalize_name(Requirement(d).name) for d in deps} - set(pins))
    assert missing == []


BUILD_CONSTRAINTS = (ROOT / "build-constraints.txt").read_text()


def test_build_backends_are_pinned_and_used():
    lines = [line for line in BUILD_CONSTRAINTS.splitlines() if line and not line.startswith("#")]
    names = {line.split("==")[0].lower() for line in lines}
    assert all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==\S+", line) for line in lines)
    assert "hatchling" in names  # this package's only build backend
    assert "--build-constraint build-constraints.txt" in DOCKERFILE and "build-constraints.txt ./" in DOCKERFILE
    assert ("--build-constraint https://raw.githubusercontent.com/MyOpenFund/myopenfund-mcp/v<version>/"
            "build-constraints.txt") in COMPOSE


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


updater = _script("update_constraints")
checker = _script("check_constraints")


def test_build_constraints_cover_build_requires():
    """The backends of this package's `[build-system] requires` (plus `editables`, which hatchling asks
    for in an editable install) are pinned in build-constraints.txt, within their specifiers."""
    requirements = updater.build_requires()
    assert "hatchling>=1.27" in requirements and "editables" in requirements
    pins = checker.read_constraints(ROOT / "build-constraints.txt")
    assert checker.check_build(pins, requirements) == []


def test_editable_build_requirement_is_pinned():
    assert re.search(r"^editables==", BUILD_CONSTRAINTS, re.MULTILINE)


# --- the checker's own logic, on fake distributions ---------------------------------------------------


class FakeMetadata:
    def __init__(self, extras):
        self._extras = list(extras)

    def get_all(self, key):
        return self._extras if key == "Provides-Extra" else None


class FakeDist:
    def __init__(self, name, version, requires=(), extras=()):
        self.name, self.version, self.requires = name, version, list(requires)
        self.metadata = FakeMetadata(extras)


def _getter(dists):
    def get(name):
        try:
            return dists[name.lower().replace("_", "-")]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None
    return get


DISTS = {
    "myopenfund-mcp": FakeDist("myopenfund-mcp", "0.1.0",
                               ["fastmcp>=4", 'myopenfund-rag-engine; extra == "rag"',
                                'myopenfund-vault-query; extra == "vault"'],
                               extras=["rag", "vault"]),
    "fastmcp": FakeDist("fastmcp", "4.2.1", ["anyio>=4", 'pywin32; sys_platform == "win32"']),
    "anyio": FakeDist("anyio", "4.16.0"),
    "myopenfund-rag-engine": FakeDist("myopenfund-rag-engine", "0.1.0", ["onnxruntime[extra-a]"]),
    "onnxruntime": FakeDist("onnxruntime", "1.32.0", ['coloredlogs; extra == "extra-a"'], extras=["extra-a"]),
    "coloredlogs": FakeDist("coloredlogs", "15.0"),
    "myopenfund-vault-query": FakeDist("myopenfund-vault-query", "0.1.0"),
}


def test_walk_follows_extras_and_markers():
    tree = checker.walk("myopenfund-mcp", {"rag"}, get=_getter(DISTS))
    # reached only through the extra `rag` and, below it, through onnxruntime's own extra
    assert set(tree) == {"myopenfund-mcp", "fastmcp", "anyio", "myopenfund-rag-engine", "onnxruntime",
                        "coloredlogs"}
    assert "myopenfund-vault-query" not in tree and "pywin32" not in tree


def test_check_reports_missing_and_mismatched_pins():
    tree = checker.walk("myopenfund-mcp", {"rag", "vault"}, get=_getter(DISTS))
    pins = {"fastmcp": "4.2.1", "anyio": "4.15.0", "onnxruntime": "1.32.0",
            "myopenfund-rag-engine": "0.1.0"}
    assert checker.check(pins, tree) == [
        "anyio: installed 4.16.0, constraints.txt says 4.15.0",
        "coloredlogs 15.0: not in constraints.txt",
        "myopenfund-vault-query 0.1.0: not in constraints.txt",
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


def test_build_check_wants_every_build_requirement_pinned_within_its_specifier():
    pins = {"hatchling": "1.33.0", "setuptools": "84.2.1", "editables": "0.6"}
    assert checker.check_build(pins, ["hatchling>=1.25", "setuptools>=77", "editables"]) == []
    assert checker.check_build(pins, ["hatchling>=2", "flit_core>=3.2", "Setuptools<80"]) == [
        "flit-core (flit_core>=3.2): not in build-constraints.txt",
        "hatchling: build-constraints.txt says 1.33.0, outside hatchling>=2",
        "setuptools: build-constraints.txt says 84.2.1, outside Setuptools<80",
    ]


def test_build_check_runs_on_the_committed_files(monkeypatch):
    """The `--build` mode as the locked job runs it, with build_requires replaced."""
    requirements = ["editables", "hatchling>=1.27"]
    monkeypatch.setattr(checker, "_build_requires", lambda: requirements)
    assert checker.main(["--build"]) == 0
    monkeypatch.setattr(checker, "_build_requires", lambda: ["hatchling>=99"])
    assert checker.main(["--build"]) == 1


def test_build_check_skips_a_requirement_whose_marker_does_not_apply():
    pins = {"hatchling": "1.33.0"}
    assert checker.check_build(pins, ['hatchling>=1.25', 'tomli>=1; python_version < "3.11"']) == []
    assert checker.check_build(pins, ['tomli>=1; python_version >= "3.11"']) == [
        'tomli (tomli>=1; python_version >= "3.11"): not in build-constraints.txt']
