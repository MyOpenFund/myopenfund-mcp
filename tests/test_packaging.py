"""Packaging identity of myopenfund-mcp: published on PyPI, siblings from PyPI within capped ranges."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[1]
DIST = "myopenfund-mcp"
CREDIT = (
    "Created and maintained by [Marc Jeulin](https://github.com/jeulinmarc), "
    "part of [MyOpenFund](https://github.com/MyOpenFund)."
)
RELEASE = REPO / ".github" / "workflows" / "release.yml"
TESTS = REPO / ".github" / "workflows" / "tests.yml"
BLOB = "https://github.com/MyOpenFund/myopenfund-mcp/blob/main/"
SIBLINGS = {
    "vault": ["myopenfund-vault-query>=0.1.0,<0.2"],
    "companies": ["myopenfund-open-data-store[reader]>=0.1.0,<0.2", "myopenfund-company-corpus>=0.1.0,<0.2"],
    "rag": ["myopenfund-rag-engine[onnx]>=0.1.0,<0.2"],
}


def _pyproject() -> dict:
    with (REPO / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _workflow(path: Path) -> dict:
    # PyYAML reads the bare key `on` as the boolean True.
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))
    return data


def _upload_index(steps: list) -> list:
    return [i for i, s in enumerate(steps) if s.get("uses", "").startswith("actions/upload-artifact@")]


def test_distribution_and_script_names():
    p = _pyproject()["project"]
    assert p["name"] == DIST
    assert p["scripts"] == {"myopenfund-mcp": "myopenfund_mcp.cli:main"}


def test_dynamic_version_from_init():
    import myopenfund_mcp

    cfg = _pyproject()
    assert "version" in cfg["project"]["dynamic"]
    assert "version" not in cfg["project"]
    assert cfg["tool"]["hatch"]["version"] == {"path": "src/myopenfund_mcp/__init__.py"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", myopenfund_mcp.__version__)


def test_installed_version_matches_init():
    import importlib.metadata

    import myopenfund_mcp

    assert importlib.metadata.version(DIST) == myopenfund_mcp.__version__


def test_credit_metadata():
    project = _pyproject()["project"]
    assert project["authors"] == [{"name": "Marc Jeulin"}]
    assert project["maintainers"] == [{"name": "Marc Jeulin"}]
    assert "myopenfund" in project["keywords"]
    urls = project["urls"]
    assert urls["Organization"] == "https://github.com/MyOpenFund"
    assert urls["Homepage"] == "https://github.com/MyOpenFund/myopenfund-mcp"
    assert urls["Source"] == "https://github.com/MyOpenFund/myopenfund-mcp"
    assert urls["Issues"] == "https://github.com/MyOpenFund/myopenfund-mcp/issues"
    assert "Changelog" not in urls
    assert project["readme"] == "README.md"
    assert project["license"] == "MIT"
    assert not [c for c in project.get("classifiers", []) if c.startswith("License ::")]


def test_license_copyright():
    assert "Copyright (c) 2026 Marc Jeulin" in (REPO / "LICENSE").read_text(encoding="utf-8")


def test_no_personal_address_in_tracked_metadata():
    for name in ("pyproject.toml", "LICENSE", "README.md", "glama.json"):
        assert "gmail" not in (REPO / name).read_text(encoding="utf-8").lower(), name


def test_glama_json():
    import json

    data = json.loads((REPO / "glama.json").read_text(encoding="utf-8"))
    assert data == {"$schema": "https://glama.ai/mcp/schemas/server.json", "maintainers": ["jeulinmarc"]}


def test_readme_credit_install_and_migration():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# myopenfund-mcp"
    assert next(line for line in lines[1:] if line.strip()) == CREDIT
    assert "<!-- mcp-name: io.github.myopenfund/myopenfund-mcp -->" in text
    assert 'uvx --from "myopenfund-mcp[vault]" myopenfund-mcp serve' in text
    assert f'pip install "{DIST}[companies]"' in text
    assert "pip uninstall -y vault-query open-data-store company-corpus rag-engine" in text
    assert "**Working across repos**" in text
    assert 'pip install "myopenfund-mcp @ git+https://github.com/MyOpenFund/myopenfund-mcp"' in text


def test_readme_links_are_absolute():
    # README.md is the PyPI description: relative links and anchors would 404 there.
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "](#" not in text
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    assert targets
    relative = [t for t in targets if not t.startswith("https://")]
    assert not relative, relative
    assert f"{BLOB}LICENSE" in text
    assert f"{BLOB}Dockerfile" in text
    assert "https://github.com/MyOpenFund/myopenfund-mcp#tools" in text


def test_extras_name_the_published_siblings_capped():
    extras = _pyproject()["project"]["optional-dependencies"]
    for name, expected in SIBLINGS.items():
        assert extras[name] == expected, name


def test_sibling_dependencies_are_capped():
    project = _pyproject()["project"]
    reqs = [Requirement(d) for d in project["dependencies"]]
    for extra in project["optional-dependencies"].values():
        reqs += [Requirement(d) for d in extra]
    siblings = [r for r in reqs if r.name.startswith("myopenfund-")]
    assert {r.name for r in siblings} == {
        "myopenfund-vault-query", "myopenfund-open-data-store", "myopenfund-company-corpus",
        "myopenfund-rag-engine",
    }
    for r in siblings:
        assert {(s.operator, s.version) for s in r.specifier} == {(">=", "0.1.0"), ("<", "0.2")}, r


def test_no_git_urls_or_direct_references_in_metadata():
    cfg = _pyproject()
    project = cfg["project"]
    deps = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        deps += extra
    assert not [d for d in deps if "git+" in d or "@" in d]
    assert "allow-direct-references" not in str(cfg.get("tool", {}).get("hatch", {}).get("metadata", {}))
    assert "metadata" not in cfg["tool"]["hatch"]


def test_dev_extra_has_pyyaml_for_the_workflow_tests():
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    assert any(Requirement(d).name.lower() == "pyyaml" for d in dev)


def test_dockerfile_does_not_install_git():
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "apt-get" not in text
    assert not re.search(r"(?<![\w.-])git(?![\w.-])", text)


def test_dependabot_covers_pip_and_github_actions_weekly():
    config = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    assert sorted((u["package-ecosystem"], u["schedule"]["interval"]) for u in config["updates"]) == [
        ("github-actions", "weekly"), ("pip", "weekly")
    ]


def test_release_workflow_publishes_through_environment():
    workflow = _workflow(RELEASE)
    assert workflow["permissions"] == {}
    assert workflow["on"]["push"]["tags"] == ["v*"]
    assert list(workflow["on"]["workflow_dispatch"]["inputs"]) == ["target"]
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == (
        "${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi')"
        " && 'testpypi' || 'pypi' }}"
    )
    assert publish["permissions"] == {"id-token": "write"}
    with_id_token = [
        name for name, job in workflow["jobs"].items() if "id-token" in job.get("permissions", {})
    ]
    assert with_id_token == ["publish"]
    assert publish["needs"] in ("build", ["build"])


def test_actions_pinned_by_sha():
    package = yaml.safe_dump(_workflow(TESTS)["jobs"]["package"])
    for text in (RELEASE.read_text(encoding="utf-8"), package):
        uses = re.findall(r"^\s*(?:-\s+)?uses:\s*(\S+)", text, re.MULTILINE)
        assert uses
        for ref in uses:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_tag_check_present_with_prefix_v():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    checks = [i for i, s in enumerate(steps) if "tag_version" in s.get("run", "") and "exit 1" in s["run"]]
    assert len(checks) == 1
    assert steps[checks[0]]["if"] == "github.event_name == 'push'"
    assert "${GITHUB_REF_NAME#v}" in steps[checks[0]]["run"]
    assert DIST in steps[checks[0]]["run"]
    upload = _upload_index(steps)
    assert upload and checks[0] < upload[0]


def test_build_checks_tag_is_on_default_branch():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    checks = [i for i, s in enumerate(steps) if "merge-base --is-ancestor" in s.get("run", "")]
    assert len(checks) == 1
    assert "origin/main" in steps[checks[0]]["run"]
    assert steps[checks[0]]["if"] == "github.event_name == 'push'"
    checkout = [s for s in steps if s.get("uses", "").startswith("actions/checkout@")]
    assert checkout[0]["with"]["fetch-depth"] == 0
    upload = _upload_index(steps)
    assert upload and checks[0] < upload[0]


def test_testpypi_publish_skips_existing_but_pypi_does_not():
    steps = {s["name"]: s for s in _workflow(RELEASE)["jobs"]["publish"]["steps"] if "name" in s}
    assert steps["Publish to TestPyPI"]["with"]["skip-existing"] is True
    assert "skip-existing" not in steps["Publish to PyPI"].get("with", {})


def test_concurrency_group():
    assert _workflow(RELEASE)["concurrency"]["group"] == "release-${{ github.ref }}"


def test_dispatch_inputs():
    inputs = _workflow(RELEASE)["on"]["workflow_dispatch"]["inputs"]
    assert list(inputs) == ["target"]
    assert inputs["target"]["type"] == "choice"
    assert inputs["target"]["options"] == ["pypi", "testpypi"]
    assert inputs["target"]["default"] == "pypi"
    assert inputs["target"]["required"] is True


def test_build_dispatch_runs_on_default_branch_only():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    guard = [s for s in steps if "refs/heads/main" in s.get("run", "")]
    assert len(guard) == 1
    assert guard[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert "exit 1" in guard[0]["run"]
    assert steps.index(guard[0]) == 0


def test_build_computes_tag_and_refuses_an_existing_pypi_tag():
    job = _workflow(RELEASE)["jobs"]["build"]
    step = [s for s in job["steps"] if s.get("id") == "tag"]
    assert len(step) == 1
    run = step[0]["run"]
    assert step[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert "importlib.metadata" in run and DIST in run
    assert 'tag="v$version"' in run
    assert 'git ls-remote --tags origin "refs/tags/$tag"' in run
    assert '"$INPUT_TARGET" = "pypi"' in run
    assert "already exists" in run and "exit 1" in run
    upload = _upload_index(job["steps"])
    assert job["steps"].index(step[0]) < upload[0]
    assert job["outputs"]["tag"] == "${{ steps.tag.outputs.tag || github.ref_name }}"
    assert job["outputs"]["sha"] == "${{ github.sha }}"


def test_testpypi_only_for_testpypi_target():
    steps = {s["name"]: s for s in _workflow(RELEASE)["jobs"]["publish"]["steps"] if "name" in s}
    assert "inputs.target == 'testpypi'" in steps["Publish to TestPyPI"]["if"]
    assert "inputs.target == 'testpypi'" in steps["Publish to PyPI"]["if"]
    assert steps["Publish to PyPI"]["if"].startswith("${{ !(")


def _release_job():
    job = _workflow(RELEASE)["jobs"]["github-release"]
    assert job["if"] == (
        "github.event_name == 'push' || "
        "(github.event_name == 'workflow_dispatch' && inputs.target == 'pypi')"
    )
    assert job["permissions"] == {"contents": "write"}
    assert set(job["needs"]) == {"build", "publish"}
    return job


def test_github_release_creates_and_pushes_the_tag_on_dispatch_not_testpypi():
    job = _release_job()
    checkout = [s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkout) == 1
    assert checkout[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert checkout[0]["with"]["persist-credentials"] is True
    assert checkout[0]["with"]["ref"] == "${{ needs.build.outputs.sha }}"
    tag = [s for s in job["steps"] if "git push" in s.get("run", "")]
    assert len(tag) == 1
    assert tag[0]["if"] == "github.event_name == 'workflow_dispatch'"
    run = tag[0]["run"]
    # the publication gate forbids a personal address in tracked files: the tagger is the actor
    assert 'user.name="Marc Jeulin"' in run
    assert 'user.email="${GITHUB_ACTOR_ID}+${GITHUB_ACTOR}@users.noreply.github.com"' in run
    assert 'tag -a "$TAG"' in run and '"$SHA"' in run
    assert 'git push origin "refs/tags/$TAG"' in run


def test_dispatch_tag_step_is_idempotent():
    run = next(s for s in _release_job()["steps"] if "git push" in s.get("run", ""))["run"]
    lookup = run.index('git ls-remote origin "refs/tags/$TAG" "refs/tags/$TAG^{}"')
    assert lookup < run.index("git push")
    assert '"$remote_sha" = "$SHA"' in run  # already there at the built commit: skip
    assert "already exists" in run and "exit 1" in run  # there, but elsewhere: fail
    assert '[ -z "$remote_sha" ]' in run  # absent: create


def test_github_release_is_created_or_completed_on_both_paths():
    job = _release_job()
    release = [s for s in job["steps"] if "gh release create" in s.get("run", "")]
    assert len(release) == 1
    assert "if" not in release[0]
    run = release[0]["run"]
    assert 'gh release view "$TAG"' in run
    assert 'gh release upload "$TAG" dist/* --repo "$GITHUB_REPOSITORY" --clobber' in run
    assert '--target "$SHA"' in run and '--title "$TAG"' in run
    assert "--verify-tag" in run
    assert "--generate-notes" in run


def test_smoke_installs_the_extras_and_runs_the_cli():
    # The CLI has no --version flag (src/ is not touched by the packaging change): the smoke runs --help
    # and compares the installed distribution's version with the package's __version__.
    for path in (RELEASE, TESTS):
        text = path.read_text(encoding="utf-8")
        assert 'dist/*.whl)[vault,companies,rag]"' in text, path
        assert "import myopenfund_mcp" in text, path
        assert '/bin/myopenfund-mcp" --help' in text, path
        assert "myopenfund_mcp.__version__" in text, path
        assert "ORT_DISABLE_TELEMETRY" in text, path


def test_tests_workflow_keeps_its_jobs_adds_package_and_runs_weekly():
    workflow = _workflow(TESTS)
    assert {"tests", "locked", "package"} <= set(workflow["jobs"])
    (entry,) = workflow["on"]["schedule"]
    _minute, _hour, day_of_month, month, day_of_week = entry["cron"].split()
    assert day_of_month == "*" and month == "*"
    assert day_of_week.isdigit() and 0 <= int(day_of_week) <= 7  # one weekday: weekly
    runs = " ".join(s.get("run", "") for s in workflow["jobs"]["package"]["steps"])
    assert "build==1.6.1 twine==7.0.0" in runs
    assert "twine check --strict" in runs
    locked = " ".join(s.get("run", "") for s in workflow["jobs"]["locked"]["steps"])
    assert "--build-constraint build-constraints.txt" in locked
