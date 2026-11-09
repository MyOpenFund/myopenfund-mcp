"""Every environment variable the server reads is documented for operators."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVER_VARS = sorted(set(re.findall(r'"(MOF_[A-Z_]+)"', (ROOT / "src/myopenfund_mcp/config.py").read_text())))


def test_config_reads_variables():
    assert "MOF_MCP_TRANSPORT" in SERVER_VARS and "MOF_VAULT_DATABASE_URL" in SERVER_VARS


@pytest.mark.parametrize("doc", [".env.example", "README.md"])
def test_every_server_variable_is_documented(doc):
    text = (ROOT / doc).read_text()
    missing = [name for name in SERVER_VARS if name not in text]
    assert missing == []


ENV_EXAMPLE = (ROOT / ".env.example").read_text()
DEPLOYMENT_ONLY = {"MOF_MCP_BIND_ADDR", "MOF_MCP_BIND_PORT", "MOF_VAULT_DOCKER_NETWORK"}
# Settings field for each server variable, to compare the documented default.
FIELDS = {
    "MOF_MCP_TRANSPORT": "transport", "MOF_MCP_HOST": "host", "MOF_MCP_PORT": "port",
    "MOF_MCP_PATH": "path", "MOF_MCP_TOKENS": "tokens", "MOF_VAULT_DATABASE_URL": "vault_database_url",
    "MOF_MCP_QUERY_TIMEOUT_S": "query_timeout_s", "MOF_MCP_CONNECT_TIMEOUT_S": "connect_timeout_s",
    "MOF_MCP_DEFAULT_ROWS": "default_rows", "MOF_MCP_MAX_ROWS": "max_rows",
    "MOF_MCP_MAX_RESULT_CHARS": "max_result_chars", "MOF_MCP_MAX_ERROR_SAMPLES": "max_error_samples",
    "MOF_MCP_MAX_TEXT_CHARS": "max_text_chars",
    "MOF_COMPANIES_CATALOG": "companies_catalog", "MOF_COMPANIES_DATA_ROOT": "companies_data_root",
    "MOF_COMPANIES_QUERY_TIMEOUT_S": "companies_query_timeout_s",
    "MOF_COMPANIES_MAX_COMPANIES": "companies_max_companies",
    "MOF_COMPANIES_MAX_CONCURRENCY": "companies_max_concurrency",
}
COMMENTED = re.findall(r"^# (MOF_[A-Z_]+)=(\S+(?: \w+=\S+)*)$", ENV_EXAMPLE, re.MULTILINE)


def test_every_env_example_variable_is_read_by_the_server():
    named = set(re.findall(r"\bMOF_[A-Z_]+\b", ENV_EXAMPLE)) - DEPLOYMENT_ONLY
    assert sorted(named - set(SERVER_VARS)) == []


def test_env_example_has_no_uncommented_assignment():
    active = [line for line in ENV_EXAMPLE.splitlines() if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=", line)]
    assert active == []


def test_field_map_covers_every_server_variable():
    assert sorted(FIELDS) == SERVER_VARS


@pytest.mark.parametrize(("name", "value"), [(n, v) for n, v in COMMENTED if n not in DEPLOYMENT_ONLY])
def test_env_example_defaults_match_settings(name, value):
    from myopenfund_mcp.config import Settings

    default = getattr(Settings(), FIELDS[name])
    if default is None:  # no default: the example shows a placeholder value
        assert "<" in value, f"{name} has no default; its example must be a <placeholder>"
    else:
        assert value == str(default)


def test_every_server_variable_has_a_commented_example():
    assert sorted({n for n, _ in COMMENTED} - DEPLOYMENT_ONLY) == SERVER_VARS
