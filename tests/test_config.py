import pytest

from myopenfund_mcp.config import ConfigError, Settings


def test_defaults_are_local_stdio():
    s = Settings.from_env({})
    assert (s.transport, s.host, s.port, s.path) == ("stdio", "127.0.0.1", 8000, "/mcp")
    assert (s.default_rows, s.max_rows, s.query_timeout_s) == (50, 200, 20)
    assert (s.max_error_samples, s.max_text_chars) == (5, 2000)
    assert s.connect_timeout_s == 10
    assert s.vault_database_url is None


def test_http_never_runs_without_tokens():
    with pytest.raises(ConfigError, match="MOF_MCP_TOKENS"):
        Settings.from_env({"MOF_MCP_TRANSPORT": "http"})


@pytest.mark.parametrize(
    "env,match",
    [
        ({"MOF_MCP_TRANSPORT": "sse"}, "MOF_MCP_TRANSPORT"),
        ({"MOF_MCP_PORT": "eighty"}, "MOF_MCP_PORT"),
        ({"MOF_MCP_MAX_ROWS": "0"}, "MOF_MCP_MAX_ROWS"),
        ({"MOF_MCP_DEFAULT_ROWS": "500"}, "MOF_MCP_DEFAULT_ROWS"),
        ({"MOF_MCP_MAX_ERROR_SAMPLES": "0"}, "MOF_MCP_MAX_ERROR_SAMPLES"),
        ({"MOF_MCP_MAX_TEXT_CHARS": "many"}, "MOF_MCP_MAX_TEXT_CHARS"),
    ],
)
def test_bad_values_fail_at_start_with_the_variable_named(env, match):
    with pytest.raises(ConfigError, match=match):
        Settings.from_env(env)


def test_empty_strings_mean_unset():
    s = Settings.from_env({"MOF_VAULT_DATABASE_URL": "", "MOF_MCP_PORT": ""})
    assert s.vault_database_url is None and s.port == 8000


@pytest.mark.parametrize(
    "env,match",
    [
        ({"MOF_MCP_PORT": "65536"}, "MOF_MCP_PORT"),
        ({"MOF_MCP_QUERY_TIMEOUT_S": "2147484"}, "MOF_MCP_QUERY_TIMEOUT_S"),
        ({"MOF_MCP_QUERY_TIMEOUT_S": "0"}, "MOF_MCP_QUERY_TIMEOUT_S"),
        ({"MOF_MCP_MAX_RESULT_CHARS": "999"}, "MOF_MCP_MAX_RESULT_CHARS"),
        ({"MOF_MCP_PATH": "mcp"}, "MOF_MCP_PATH"),
        ({"MOF_MCP_MAX_ROWS": "10"}, "MOF_MCP_MAX_ROWS"),
        ({"MOF_MCP_PORT": "+5"}, "MOF_MCP_PORT"),
        ({"MOF_MCP_MAX_ROWS": "1_000"}, "MOF_MCP_MAX_ROWS"),
        ({"MOF_MCP_DEFAULT_ROWS": "٣٤"}, "MOF_MCP_DEFAULT_ROWS"),
        # libpq silently raises a connect_timeout below 2 to 2: refused rather than misread.
        ({"MOF_MCP_CONNECT_TIMEOUT_S": "1"}, "MOF_MCP_CONNECT_TIMEOUT_S"),
        ({"MOF_MCP_CONNECT_TIMEOUT_S": "301"}, "MOF_MCP_CONNECT_TIMEOUT_S"),
    ],
)
def test_bounds_path_and_strict_integers_fail_at_start(env, match):
    with pytest.raises(ConfigError, match=match):
        Settings.from_env(env)


def test_upper_bounds_are_inclusive():
    s = Settings.from_env({"MOF_MCP_PORT": "65535", "MOF_MCP_QUERY_TIMEOUT_S": "2147483"})
    assert (s.port, s.query_timeout_s) == (65535, 2_147_483)


def test_default_rows_over_max_rows_names_both_values():
    with pytest.raises(ConfigError) as exc:
        Settings.from_env({"MOF_MCP_DEFAULT_ROWS": "60", "MOF_MCP_MAX_ROWS": "40"})
    assert "MOF_MCP_DEFAULT_ROWS (60) must not exceed MOF_MCP_MAX_ROWS (40)" in str(exc.value)


def test_default_rows_equal_to_max_rows_is_accepted():
    s = Settings.from_env({"MOF_MCP_DEFAULT_ROWS": "30", "MOF_MCP_MAX_ROWS": "30"})
    assert s.default_rows == s.max_rows == 30


def test_integers_are_stripped():
    assert Settings.from_env({"MOF_MCP_PORT": " 8080 "}).port == 8080


def test_http_with_tokens_starts():
    spec = "alice:" + "a" * 64
    s = Settings.from_env({"MOF_MCP_TRANSPORT": "http", "MOF_MCP_TOKENS": spec})
    assert s.transport == "http" and s.tokens == spec


def test_http_with_blank_tokens_is_refused():
    with pytest.raises(ConfigError, match="MOF_MCP_TOKENS"):
        Settings.from_env({"MOF_MCP_TRANSPORT": "http", "MOF_MCP_TOKENS": "   "})


def test_secrets_never_appear_in_repr():
    spec = "alice:" + "b" * 64
    s = Settings.from_env(
        {
            "MOF_MCP_TRANSPORT": "http",
            "MOF_MCP_TOKENS": spec,
            "MOF_VAULT_DATABASE_URL": "postgresql://vault_readonly:pw@localhost/vault",
        }
    )
    text = repr(s)
    assert "postgresql://" not in text
    assert spec not in text and "b" * 64 not in text


def test_from_env_without_argument_reads_os_environ(monkeypatch):
    monkeypatch.setenv("MOF_MCP_PORT", "9123")
    assert Settings.from_env().port == 9123


def test_connect_timeout_bounds_are_inclusive():
    assert Settings.from_env({"MOF_MCP_CONNECT_TIMEOUT_S": "2"}).connect_timeout_s == 2
    assert Settings.from_env({"MOF_MCP_CONNECT_TIMEOUT_S": "300"}).connect_timeout_s == 300


def test_companies_settings_from_env():
    s = Settings.from_env({"MOF_COMPANIES_CATALOG": "postgres:dbname=lake_catalog host=h user=lake_reader password=p",
                           "MOF_COMPANIES_DATA_ROOT": "/data/my_open_data",
                           "MOF_COMPANIES_MAX_COMPANIES": "5"})
    assert s.companies_data_root == "/data/my_open_data" and s.companies_max_companies == 5
    assert s.companies_query_timeout_s == 20 and s.companies_max_concurrency == 2
    assert "password" not in repr(s)


def test_companies_catalog_needs_a_data_root():
    with pytest.raises(ConfigError, match="MOF_COMPANIES_DATA_ROOT"):
        Settings.from_env({"MOF_COMPANIES_CATALOG": "x"})


def test_companies_data_root_alone_is_refused():
    with pytest.raises(ConfigError, match="MOF_COMPANIES_CATALOG"):
        Settings.from_env({"MOF_COMPANIES_DATA_ROOT": "/d"})


def test_companies_q4_rule_defaults_to_the_corpus_rule():
    assert Settings.from_env({}).companies_q4_rule == "ytd9"


def test_companies_q4_rule_accepts_sum_of_quarters():
    assert Settings.from_env({"MOF_COMPANIES_Q4_RULE": "sum_of_quarters"}).companies_q4_rule == "sum_of_quarters"


@pytest.mark.parametrize("bad", ["YTD9", "sum", "ytd9,sum_of_quarters", "annual"])
def test_companies_q4_rule_refuses_other_values(bad):
    with pytest.raises(ConfigError, match="MOF_COMPANIES_Q4_RULE"):
        Settings.from_env({"MOF_COMPANIES_Q4_RULE": bad})
