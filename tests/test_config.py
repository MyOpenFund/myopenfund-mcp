import pytest

from myopenfund_mcp.config import ConfigError, Settings


def test_defaults_are_local_stdio():
    s = Settings.from_env({})
    assert (s.transport, s.host, s.port, s.path) == ("stdio", "127.0.0.1", 8000, "/mcp")
    assert (s.default_rows, s.max_rows, s.query_timeout_s) == (50, 200, 20)
    assert (s.max_error_samples, s.max_text_chars) == (5, 2000)
    assert s.vault_database_url is None


def test_http_never_runs_without_tokens():
    with pytest.raises(ConfigError, match="MOF_MCP_TOKENS"):
        Settings.from_env({"MOF_MCP_TRANSPORT": "http"})


@pytest.mark.parametrize("env,match", [
    ({"MOF_MCP_TRANSPORT": "sse"}, "MOF_MCP_TRANSPORT"),
    ({"MOF_MCP_PORT": "eighty"}, "MOF_MCP_PORT"),
    ({"MOF_MCP_MAX_ROWS": "0"}, "MOF_MCP_MAX_ROWS"),
    ({"MOF_MCP_DEFAULT_ROWS": "500"}, "MOF_MCP_DEFAULT_ROWS"),
    ({"MOF_MCP_MAX_ERROR_SAMPLES": "0"}, "MOF_MCP_MAX_ERROR_SAMPLES"),
    ({"MOF_MCP_MAX_TEXT_CHARS": "many"}, "MOF_MCP_MAX_TEXT_CHARS"),
])
def test_bad_values_fail_at_start_with_the_variable_named(env, match):
    with pytest.raises(ConfigError, match=match):
        Settings.from_env(env)


def test_empty_strings_mean_unset():
    s = Settings.from_env({"MOF_VAULT_DATABASE_URL": "", "MOF_MCP_PORT": ""})
    assert s.vault_database_url is None and s.port == 8000
