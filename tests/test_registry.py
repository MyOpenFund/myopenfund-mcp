"""MCP Registry listing: server.json and the workflow that publishes it after the PyPI release."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from myopenfund_mcp import __version__

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "server.json"
README = REPO / "README.md"
REGISTRY = REPO / ".github" / "workflows" / "mcp-registry.yml"
RELEASE = REPO / ".github" / "workflows" / "release.yml"
NAME = "io.github.MyOpenFund/myopenfund-mcp"


def _workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))
    return data


def _server() -> dict:
    return json.loads(SERVER.read_text(encoding="utf-8"))


def test_server_json_identity():
    server = _server()
    assert server["$schema"] == "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"
    assert server["name"] == NAME
    assert server["title"] == "MyOpenFund MCP"
    assert 0 < len(server["description"]) <= 100
    assert server["repository"] == {"url": "https://github.com/MyOpenFund/myopenfund-mcp", "source": "github"}


def test_server_json_versions_follow_the_package():
    server = _server()
    assert server["version"] == __version__
    assert len(server["packages"]) == 1
    package = server["packages"][0]
    assert package["version"] == __version__
    assert package["registryType"] == "pypi"
    assert package["identifier"] == "myopenfund-mcp"
    assert package["transport"] == {"type": "stdio"}


def test_readme_carries_the_registry_ownership_marker():
    text = README.read_text(encoding="utf-8")
    assert f"<!-- mcp-name: {_server()['name']} -->" in text
    assert f"`{NAME}`" in text


def test_registry_workflow_triggers_and_permissions():
    wf = _workflow(REGISTRY)
    assert set(wf["on"]) == {"workflow_call", "workflow_dispatch"}
    assert not (wf["on"]["workflow_call"] or {}).get("inputs")
    assert wf["permissions"] == {}
    assert list(wf["jobs"]) == ["publish-registry"]
    job = wf["jobs"]["publish-registry"]
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    assert job["concurrency"]["group"] == "mcp-registry"
    assert "environment" not in job  # keeps the id-token away from the PyPI environment


def test_registry_workflow_pins_and_verifies_the_publisher():
    text = REGISTRY.read_text(encoding="utf-8")
    assert re.search(r"MCP_PUBLISHER_VERSION: v\d+\.\d+\.\d+\b", text)
    assert "releases/latest" not in text and "@latest" not in text
    assert "cosign verify-blob" in text
    assert "--certificate-identity" in text and "--certificate-oidc-issuer" in text
    assert "login github-oidc" in text
    assert text.index("cosign verify-blob") < text.index("tar xzf")
    assert "modelcontextprotocol/registry/.github/workflows/release.yml@refs/tags/" in text
    assert "run: |" in text and "${{" not in "".join(
        s["run"] for s in _workflow(REGISTRY)["jobs"]["publish-registry"]["steps"] if "run" in s
    )


def test_registry_workflow_checks_pypi_before_publishing():
    steps = _workflow(REGISTRY)["jobs"]["publish-registry"]["steps"]
    runs = [s.get("run", "") for s in steps]
    pypi = next(i for i, r in enumerate(runs) if "pypi.org/pypi/myopenfund-mcp/" in r)
    publish = next(i for i, r in enumerate(runs) if "mcp-publisher publish" in r)
    assert pypi < publish
    assert "publish to PyPI first" in runs[pypi]
    assert "--retry 5 --retry-all-errors --retry-delay 10" in runs[pypi] and '"404"' in runs[pypi]
    guard = steps[0]
    assert guard["if"] == "github.event_name == 'workflow_dispatch'"
    assert "main" in guard["run"] and "exit 1" in guard["run"]


def test_registry_workflow_actions_are_pinned_by_sha():
    # release.yml is covered by test_packaging; this file is not.
    uses = re.findall(r"uses:\s*(\S+)", REGISTRY.read_text(encoding="utf-8"))
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), ref


def test_release_calls_the_registry_after_the_github_release():
    jobs = _workflow(RELEASE)["jobs"]
    job = jobs["mcp-registry"]
    assert job["needs"] == "github-release" or job["needs"] == ["github-release"]
    assert job["if"] == jobs["github-release"]["if"]
    assert "environment" not in job
    assert job["uses"] == "./.github/workflows/mcp-registry.yml"
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    assert list(jobs)[-1] == "mcp-registry"
