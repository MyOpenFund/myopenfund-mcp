import os

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
                           "MOF_COMPANIES_DATA_ROOT": "/data/open_data_store",
                           "MOF_COMPANIES_MAX_COMPANIES": "5"})
    assert s.companies_data_root == "/data/open_data_store" and s.companies_max_companies == 5
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


RAG_ENV = {"MOF_RAG_QDRANT_HOST": "qdrant", "MOF_RAG_COLLECTION": "central-bank-e5b-v1",
           "MOF_RAG_MODEL_CACHE": "/models"}


def test_rag_defaults():
    s = Settings.from_env({})
    assert s.rag_qdrant_host is None and s.rag_collection is None and s.rag_model_cache is None
    assert (s.rag_qdrant_port, s.rag_qdrant_https, s.rag_qdrant_timeout) == (6333, False, 10)
    assert (s.rag_default_limit, s.rag_max_limit, s.rag_onnx_threads, s.rag_codes_ttl_s) == (8, 20, None, 300)
    assert (s.rag_call_timeout_s, s.rag_max_concurrency, s.rag_check_retry_s, s.rag_codes_retry_s) == (15, 4, 5, 5)
    assert s.rag_description_notes == ""
    assert (s.rag_date_filter_mode, s.rag_date_filter_refresh_s, s.rag_date_filter_retry_s) == ("auto", 300, 30)


def test_rag_date_filter_settings_from_env():
    for mode in ("auto", "interval", "legacy"):
        assert Settings.from_env({**RAG_ENV, "MOF_RAG_DATE_FILTER_MODE": mode}).rag_date_filter_mode == mode
    s = Settings.from_env({**RAG_ENV, "MOF_RAG_DATE_FILTER_REFRESH_S": "0", "MOF_RAG_DATE_FILTER_RETRY_S": "1"})
    assert (s.rag_date_filter_refresh_s, s.rag_date_filter_retry_s) == (0, 1)  # 0: the collection is asked every time
    s = Settings.from_env({**RAG_ENV, "MOF_RAG_DATE_FILTER_REFRESH_S": "86400", "MOF_RAG_DATE_FILTER_RETRY_S": "3600"})
    assert (s.rag_date_filter_refresh_s, s.rag_date_filter_retry_s) == (86400, 3600)


def test_the_date_filter_modes_are_rag_engines():
    engine = pytest.importorskip("rag_engine.config")
    from myopenfund_mcp.config import RAG_DATE_FILTER_MODES

    assert RAG_DATE_FILTER_MODES == engine.DATE_FILTER_MODES
    assert Settings().rag_date_filter_mode == engine.Settings().date_filter_mode
    assert Settings().rag_date_filter_refresh_s == engine.Settings().date_filter_refresh
    assert Settings().rag_date_filter_retry_s == engine.Settings().date_filter_retry


def test_rag_settings_from_env():
    s = Settings.from_env({**RAG_ENV, "MOF_RAG_QDRANT_PORT": "6334", "MOF_RAG_QDRANT_API_KEY": "k-secret",
                           "MOF_RAG_QDRANT_HTTPS": "TRUE", "MOF_RAG_QDRANT_TIMEOUT": "5",
                           "MOF_RAG_ONNX_THREADS": "4", "MOF_RAG_DEFAULT_LIMIT": "5", "MOF_RAG_MAX_LIMIT": "30",
                           "MOF_RAG_CODES_TTL_S": "60", "MOF_RAG_CALL_TIMEOUT_S": "9",
                           "MOF_RAG_MAX_CONCURRENCY": "2", "MOF_RAG_CHECK_RETRY_S": "3",
                           "MOF_RAG_CODES_RETRY_S": "7", "MOF_RAG_DESCRIPTION_NOTES": "  Corpus: test.  "})
    assert (s.rag_call_timeout_s, s.rag_max_concurrency, s.rag_check_retry_s, s.rag_codes_retry_s) == (9, 2, 3, 7)
    assert s.rag_description_notes == "Corpus: test."
    assert (s.rag_qdrant_host, s.rag_qdrant_port, s.rag_collection) == ("qdrant", 6334, "central-bank-e5b-v1")
    assert (s.rag_qdrant_https, s.rag_qdrant_timeout, s.rag_model_cache) == (True, 5, "/models")
    assert (s.rag_onnx_threads, s.rag_default_limit, s.rag_max_limit, s.rag_codes_ttl_s) == (4, 5, 30, 60)
    assert s.rag_qdrant_api_key == "k-secret" and "k-secret" not in repr(s)


@pytest.mark.parametrize(("env", "match"), [
    ({"MOF_RAG_QDRANT_HOST": "qdrant"}, "MOF_RAG_COLLECTION"),
    ({"MOF_RAG_COLLECTION": "c"}, "MOF_RAG_QDRANT_HOST"),
    ({"MOF_RAG_QDRANT_HOST": "qdrant", "MOF_RAG_COLLECTION": "c"}, "MOF_RAG_MODEL_CACHE"),
    ({**RAG_ENV, "MOF_RAG_QDRANT_PORT": "0"}, "MOF_RAG_QDRANT_PORT"),
    ({**RAG_ENV, "MOF_RAG_QDRANT_PORT": "65536"}, "MOF_RAG_QDRANT_PORT"),
    ({**RAG_ENV, "MOF_RAG_QDRANT_HTTPS": "maybe"}, "MOF_RAG_QDRANT_HTTPS"),
    ({**RAG_ENV, "MOF_RAG_QDRANT_TIMEOUT": "0"}, "MOF_RAG_QDRANT_TIMEOUT"),
    ({**RAG_ENV, "MOF_RAG_QDRANT_TIMEOUT": "301"}, "MOF_RAG_QDRANT_TIMEOUT"),
    ({**RAG_ENV, "MOF_RAG_QDRANT_TIMEOUT": "2.5"}, "MOF_RAG_QDRANT_TIMEOUT"),
    ({**RAG_ENV, "MOF_RAG_ONNX_THREADS": "0"}, "MOF_RAG_ONNX_THREADS"),
    ({**RAG_ENV, "MOF_RAG_ONNX_THREADS": "many"}, "MOF_RAG_ONNX_THREADS"),
    ({**RAG_ENV, "MOF_RAG_MAX_LIMIT": "0"}, "MOF_RAG_MAX_LIMIT"),
    ({**RAG_ENV, "MOF_RAG_DEFAULT_LIMIT": "0"}, "MOF_RAG_DEFAULT_LIMIT"),
    ({**RAG_ENV, "MOF_RAG_DEFAULT_LIMIT": "21"}, r"MOF_RAG_DEFAULT_LIMIT \(21\) must not exceed MOF_RAG_MAX_LIMIT \(20\)"),
    ({**RAG_ENV, "MOF_RAG_CODES_TTL_S": "0"}, "MOF_RAG_CODES_TTL_S"),
    ({**RAG_ENV, "MOF_RAG_COLLECTION": "a/b"}, "MOF_RAG_COLLECTION"),
    ({**RAG_ENV, "MOF_RAG_CALL_TIMEOUT_S": "0"}, "MOF_RAG_CALL_TIMEOUT_S"),
    # A call must outlast one request to Qdrant, or the Qdrant timeout would never be the one reported.
    ({**RAG_ENV, "MOF_RAG_CALL_TIMEOUT_S": "9"},
     r"MOF_RAG_CALL_TIMEOUT_S \(9\) must not be below MOF_RAG_QDRANT_TIMEOUT \(10\)"),
    ({**RAG_ENV, "MOF_RAG_QDRANT_TIMEOUT": "20"}, "MOF_RAG_CALL_TIMEOUT_S"),
    ({**RAG_ENV, "MOF_RAG_MAX_CONCURRENCY": "0"}, "MOF_RAG_MAX_CONCURRENCY"),
    ({**RAG_ENV, "MOF_RAG_MAX_CONCURRENCY": "65"}, "MOF_RAG_MAX_CONCURRENCY"),
    ({**RAG_ENV, "MOF_RAG_CHECK_RETRY_S": "0"}, "MOF_RAG_CHECK_RETRY_S"),
    ({**RAG_ENV, "MOF_RAG_CODES_RETRY_S": "soon"}, "MOF_RAG_CODES_RETRY_S"),
    ({**RAG_ENV, "MOF_RAG_DESCRIPTION_NOTES": "x" * 1501}, "MOF_RAG_DESCRIPTION_NOTES must hold at most 1500"),
    ({**RAG_ENV, "MOF_RAG_MODEL_CACHE": "models"}, "MOF_RAG_MODEL_CACHE must be an absolute path"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_MODE": "AUTO"},
     "MOF_RAG_DATE_FILTER_MODE must be one of auto, interval, legacy, got 'AUTO'"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_MODE": "intervals"}, "MOF_RAG_DATE_FILTER_MODE must be one of"),
    ({"MOF_RAG_DATE_FILTER_MODE": "off"}, "MOF_RAG_DATE_FILTER_MODE must be one of"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_REFRESH_S": "-1"},
     r"MOF_RAG_DATE_FILTER_REFRESH_S must be an integer from 0 to 86400, got '-1'"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_REFRESH_S": "1.5"},
     r"MOF_RAG_DATE_FILTER_REFRESH_S must be an integer from 0 to 86400, got '1.5'"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_RETRY_S": "-5"}, r"MOF_RAG_DATE_FILTER_RETRY_S must be an integer from 1 to 3600"),
    ({**RAG_ENV, "MOF_RAG_DEFAULT_LIMIT": "-5"}, r"MOF_RAG_DEFAULT_LIMIT must be an integer >= 1, got '-5'"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_REFRESH_S": "86401"}, "MOF_RAG_DATE_FILTER_REFRESH_S must be <= 86400"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_RETRY_S": "0"}, "MOF_RAG_DATE_FILTER_RETRY_S must be >= 1"),
    ({**RAG_ENV, "MOF_RAG_DATE_FILTER_RETRY_S": "3601"}, "MOF_RAG_DATE_FILTER_RETRY_S must be <= 3600"),
    ({**RAG_ENV, "MOF_RAG_MODEL_CACHE": "./models"}, "MOF_RAG_MODEL_CACHE must be an absolute path"),
])
def test_rag_bad_values_fail_at_start_with_the_variable_named(env, match):
    with pytest.raises(ConfigError, match=match):
        Settings.from_env(env)


@pytest.mark.parametrize("host", ["http://qdrant:6333", "user:pw@qdrant/x", "qdrant host", "qdrant/", "[::1",
                                  "[qdrant]", "::1]", "https://[::1]"])
def test_rag_host_is_a_bare_host_and_a_bad_one_is_never_echoed(host):
    with pytest.raises(ConfigError, match="MOF_RAG_QDRANT_HOST") as caught:
        Settings.from_env({**RAG_ENV, "MOF_RAG_QDRANT_HOST": host})
    assert host not in str(caught.value) and "pw" not in str(caught.value)


@pytest.mark.parametrize("host", ["qdrant:6333", "10.0.0.5:6333", "[::1]:6333", "[2001:db8::5]:6333", "user:pw@qdrant",
                                  "qdrant:"])
def test_rag_host_with_a_port_is_refused_naming_the_port_variable(host):
    """It used to pass, then every request failed for ever with `qdrant_error`."""
    with pytest.raises(ConfigError, match="MOF_RAG_QDRANT_HOST must not carry a port.*MOF_RAG_QDRANT_PORT") as caught:
        Settings.from_env({**RAG_ENV, "MOF_RAG_QDRANT_HOST": host})
    assert host not in str(caught.value) and "pw" not in str(caught.value)


@pytest.mark.parametrize("host", ["qdrant", "10.0.0.5", "nas.example.net", "::1", "[::1]", "2001:db8::5",
                                  "[2001:db8::5]", "fe80::1"])
def test_rag_host_accepts_names_and_addresses(host):
    assert Settings.from_env({**RAG_ENV, "MOF_RAG_QDRANT_HOST": host}).rag_qdrant_host == host


@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("yes", True), ("On", True), ("0", False),
                                               ("no", False), ("OFF", False), ("", False)])
def test_rag_https_reads_the_usual_booleans(raw, expected):
    assert Settings.from_env({**RAG_ENV, "MOF_RAG_QDRANT_HTTPS": raw}).rag_qdrant_https is expected


def test_rag_tuning_variables_alone_do_not_configure_the_toolset():
    s = Settings.from_env({"MOF_RAG_MAX_LIMIT": "30", "MOF_RAG_MODEL_CACHE": "/models"})
    assert s.rag_qdrant_host is None and s.rag_max_limit == 30


# --- no-vault mode (live SEC, local filings index) ------------------------------------------------------
def test_no_vault_defaults():
    s = Settings.from_env({})
    assert s.sec_contact is None and s.companies_source == "auto"
    assert (s.sec_max_rps, s.sec_timeout_s, s.sec_retries, s.sec_call_budget_s) == (5.0, 30, 4, 50)
    assert (s.sec_positions_lookback_y, s.sec_memo_companies, s.sec_failure_ttl_s) == (5, 50, 60)
    assert (s.sec_positions_donor_depth, s.sec_ttm_lookback_m) == (4, 20)
    assert (s.cache, s.cache_max_gb) == ("on", 20)
    assert (s.cache_evict_to, s.cache_tmp_grace_s) == (0.9, 3600)
    assert s.cache_dir is None and s.cache_path().endswith("myopenfund-mcp")
    assert (s.cache_ttl_submissions_h, s.cache_ttl_companyfacts_h) == (24, 24)
    assert (s.filings_max_docs, s.index_workers, s.rag_onnx_provider) == (40, 1, "cpu")
    assert (s.filings_search_default_limit, s.filings_search_max_limit) == (8, 20)
    assert (s.filings_search_max_docs, s.filings_not_ready_listed) == (40, 10)
    assert (s.search_rrf_k, s.search_candidates, s.search_eigenmind) == (60, 50, False)
    assert s.filings_default_forms == ("10-K",)
    assert (s.chunk_max_tokens, s.chunk_overlap_tokens) == (480, 32)


def test_cache_dir_default_follows_the_platform(monkeypatch, tmp_path):
    from myopenfund_mcp import config

    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert config.default_cache_dir() == str(tmp_path / "myopenfund-mcp")
    monkeypatch.setattr(config.sys, "platform", "darwin")
    assert config.default_cache_dir().endswith("Library/Caches/myopenfund-mcp")
    monkeypatch.setattr(config.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert config.default_cache_dir() == str(tmp_path / "myopenfund-mcp")


def test_cache_dir_override_and_cache_off():
    s = Settings.from_env({"MOF_CACHE_DIR": "/data/c", "MOF_CACHE": "off"})
    assert (s.cache_dir, s.cache) == ("/data/c", "off")


def test_sources_and_their_requirements():
    assert Settings.from_env({"MOF_COMPANIES_SOURCE": "auto"}).companies_source == "auto"
    with pytest.raises(ConfigError, match="MOF_COMPANIES_CATALOG"):
        Settings.from_env({"MOF_COMPANIES_SOURCE": "store"})
    with pytest.raises(ConfigError, match="MOF_SEC_CONTACT"):
        Settings.from_env({"MOF_COMPANIES_SOURCE": "sec"})
    assert Settings.from_env({"MOF_COMPANIES_SOURCE": "sec", "MOF_SEC_CONTACT": "a@b.c"}).sec_contact == "a@b.c"
    s = Settings.from_env({"MOF_COMPANIES_SOURCE": "store", "MOF_COMPANIES_CATALOG": "postgres:x",
                           "MOF_COMPANIES_DATA_ROOT": "/d"})
    assert s.companies_source == "store"


def test_contact_is_never_shown_or_echoed():
    secret = "someone.private@example.org"
    s = Settings.from_env({"MOF_SEC_CONTACT": secret})
    assert secret not in repr(s)
    with pytest.raises(ConfigError) as err:
        Settings.from_env({"MOF_SEC_CONTACT": secret, "MOF_SEC_MAX_RPS": "11"})
    assert secret not in str(err.value)


@pytest.mark.parametrize(
    "env",
    [
        {"MOF_SEC_MAX_RPS": "11"}, {"MOF_SEC_MAX_RPS": "0"}, {"MOF_SEC_MAX_RPS": "-1"},
        {"MOF_SEC_MAX_RPS": "fast"}, {"MOF_SEC_MAX_RPS": "nan"}, {"MOF_SEC_MAX_RPS": "inf"},
        {"MOF_COMPANIES_SOURCE": "web"}, {"MOF_CACHE": "maybe"}, {"MOF_RAG_ONNX_PROVIDER": "tpu"},
        {"MOF_SEC_TIMEOUT_S": "0"}, {"MOF_SEC_RETRIES": "-1"}, {"MOF_SEC_CALL_BUDGET_S": "0"},
        {"MOF_CACHE_MAX_GB": "0"}, {"MOF_CACHE_EVICT_TO": "0"}, {"MOF_CACHE_EVICT_TO": "1.1"},
        {"MOF_CACHE_TMP_GRACE_S": "59"}, {"MOF_CACHE_TTL_SUBMISSIONS_H": "0"}, {"MOF_FILINGS_MAX_DOCS": "0"},
        {"MOF_INDEX_WORKERS": "0"}, {"MOF_SEARCH_RRF_K": "0"}, {"MOF_SEARCH_CANDIDATES": "0"},
        {"MOF_SEARCH_EIGENMIND": "perhaps"}, {"MOF_FILINGS_DEFAULT_FORMS": ","},
        {"MOF_CHUNK_MAX_TOKENS": "0"}, {"MOF_CHUNK_OVERLAP_TOKENS": "480"},
        {"MOF_CACHE_DIR": "relative/dir"}, {"MOF_CACHE_DIR": "~/cache"},
        {"MOF_SEC_CONTACT": "a@b.c\r\nX: y"}, {"MOF_SEC_CONTACT": "\u00e9@b.c"},
        {"MOF_SEC_CONTACT": "a" * 201}, {"MOF_SEC_MAX_RPS": "10.5"}, {"MOF_SEC_TIMEOUT_S": "601"},
        {"MOF_SEC_RETRIES": "21"}, {"MOF_SEC_CALL_BUDGET_S": "3601"}, {"MOF_CACHE_MAX_GB": "100001"},
        {"MOF_SEC_POSITIONS_LOOKBACK_Y": "0"}, {"MOF_SEC_POSITIONS_LOOKBACK_Y": "51"},
        {"MOF_SEC_MEMO_COMPANIES": "0"}, {"MOF_SEC_MEMO_COMPANIES": "10001"},
        {"MOF_SEC_FAILURE_TTL_S": "0"}, {"MOF_SEC_FAILURE_TTL_S": "3601"},
        {"MOF_SEC_POSITIONS_DONOR_DEPTH": "-1"}, {"MOF_SEC_POSITIONS_DONOR_DEPTH": "41"},
        {"MOF_SEC_TTM_LOOKBACK_M": "-1"}, {"MOF_SEC_TTM_LOOKBACK_M": "121"},
        {"MOF_CACHE_TTL_COMPANYFACTS_H": "0"}, {"MOF_CACHE_TTL_COMPANYFACTS_H": "8761"},
        {"MOF_CACHE_TTL_SUBMISSIONS_H": "8761"}, {"MOF_FILINGS_MAX_DOCS": "1001"}, {"MOF_INDEX_WORKERS": "33"}, {"MOF_INDEX_CLOSE_TIMEOUT_S": "3601"},
        {"MOF_SEARCH_RRF_K": "1001"}, {"MOF_SEARCH_CANDIDATES": "1001"}, {"MOF_CHUNK_MAX_TOKENS": "8193"},
        {"MOF_FILINGS_SEARCH_DEFAULT_LIMIT": "0"}, {"MOF_FILINGS_SEARCH_MAX_LIMIT": "1001"},
        {"MOF_FILINGS_SEARCH_MAX_DOCS": "0"}, {"MOF_FILINGS_NOT_READY_LISTED": "1001"},
        {"MOF_FILINGS_SEARCH_DEFAULT_LIMIT": "21"},
        {"MOF_INDEX_BATCH_SIZE": "0"}, {"MOF_INDEX_BATCH_SIZE": "1025"}, {"MOF_INDEX_FIRST_BATCHES": "-1"},
        {"MOF_INDEX_FIRST_BATCHES": "101"}, {"MOF_INDEX_FETCH_AHEAD": "0"}, {"MOF_INDEX_FETCH_AHEAD": "1001"},
    ],
)
def test_bad_no_vault_values_are_refused(env):
    with pytest.raises(ConfigError, match=next(iter(env))):
        Settings.from_env(env)


def test_rps_is_a_float_up_to_ten_and_retries_may_be_zero():
    assert Settings.from_env({"MOF_SEC_MAX_RPS": "2.5"}).sec_max_rps == 2.5
    assert Settings.from_env({"MOF_SEC_MAX_RPS": "10"}).sec_max_rps == 10.0
    assert Settings.from_env({"MOF_SEC_RETRIES": "0"}).sec_retries == 0
    assert Settings.from_env({"MOF_SEARCH_EIGENMIND": "true"}).search_eigenmind is True


def test_indexing_order_settings():
    default = Settings.from_env({})
    assert (default.index_batch_size, default.index_first_batches, default.index_fetch_ahead) == (32, 1, 24)
    assert default.index_close_timeout_s == 30  # one batch (8-12 s on 4 CPU) fits
    got = Settings.from_env({"MOF_INDEX_BATCH_SIZE": "8", "MOF_INDEX_FIRST_BATCHES": "0", "MOF_INDEX_FETCH_AHEAD": "3"})
    assert (got.index_batch_size, got.index_first_batches, got.index_fetch_ahead) == (8, 0, 3)


def test_default_cache_dir_ignores_a_relative_base_and_names_a_missing_home(monkeypatch, tmp_path):
    from myopenfund_mcp import config

    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", "rel")
    assert config.default_cache_dir() == str(tmp_path / ".cache" / "myopenfund-mcp")
    monkeypatch.setattr(config.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "rel")
    assert config.default_cache_dir().endswith("myopenfund-mcp") and os.path.isabs(config.default_cache_dir())
    monkeypatch.setattr(config.os.path, "expanduser", lambda p: p)  # no home directory
    with pytest.raises(ConfigError, match="MOF_CACHE_DIR"):
        config.default_cache_dir()


def test_an_unusable_default_cache_dir_stops_no_deployment_that_never_asked_for_it(monkeypatch):
    from myopenfund_mcp import config

    monkeypatch.setattr(config.os.path, "expanduser", lambda p: p)
    s = Settings.from_env({})  # vault-only: starts
    with pytest.raises(ConfigError, match="MOF_CACHE_DIR"):
        s.cache_path()


def test_contact_error_never_echoes_the_value():
    with pytest.raises(ConfigError) as err:
        Settings.from_env({"MOF_SEC_CONTACT": "secret\u00e9@b.c"})
    assert "secret" not in str(err.value)


def test_forms_are_normalised_and_cache_is_case_insensitive():
    s = Settings.from_env({"MOF_FILINGS_DEFAULT_FORMS": " 10-k, 10-Q ,10-K", "MOF_CACHE": "OFF"})
    assert s.filings_default_forms == ("10-K", "10-Q") and s.cache == "off"


def test_source_sec_with_a_catalogue_set_still_needs_the_contact():
    with pytest.raises(ConfigError, match="MOF_SEC_CONTACT"):
        Settings.from_env({"MOF_COMPANIES_SOURCE": "sec", "MOF_COMPANIES_CATALOG": "postgres:x",
                           "MOF_COMPANIES_DATA_ROOT": "/d"})
