"""The deployment is reproducible: one pinned base image, exact PyPI versions, git deps by commit."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "compose.example.yaml").read_text()
CONSTRAINTS = (ROOT / "constraints.txt").read_text()
PYPROJECT = (ROOT / "pyproject.toml").read_text()
IMAGE = re.search(r"^ARG PYTHON_IMAGE=(\S+)$", DOCKERFILE, re.MULTILINE).group(1)


def test_base_image_is_pinned_by_tag_and_digest():
    assert re.fullmatch(r"python:3\.13\.\d+-slim-[a-z]+@sha256:[0-9a-f]{64}", IMAGE)


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
