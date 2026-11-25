"""Every environment variable the server reads is documented for operators."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVER_VARS = sorted(set(re.findall(r'"(MOF_[A-Z0-9_]+)"', (ROOT / "src/myopenfund_mcp/config.py").read_text())))


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
    "MOF_COMPANIES_Q4_RULE": "companies_q4_rule",
    "MOF_RAG_QDRANT_HOST": "rag_qdrant_host", "MOF_RAG_QDRANT_PORT": "rag_qdrant_port",
    "MOF_RAG_QDRANT_API_KEY": "rag_qdrant_api_key", "MOF_RAG_QDRANT_HTTPS": "rag_qdrant_https",
    "MOF_RAG_QDRANT_TIMEOUT": "rag_qdrant_timeout", "MOF_RAG_COLLECTION": "rag_collection",
    "MOF_RAG_MODEL_CACHE": "rag_model_cache", "MOF_RAG_ONNX_THREADS": "rag_onnx_threads",
    "MOF_RAG_DEFAULT_LIMIT": "rag_default_limit", "MOF_RAG_MAX_LIMIT": "rag_max_limit",
    "MOF_RAG_CODES_TTL_S": "rag_codes_ttl_s", "MOF_RAG_CODES_RETRY_S": "rag_codes_retry_s",
    "MOF_RAG_CALL_TIMEOUT_S": "rag_call_timeout_s", "MOF_RAG_MAX_CONCURRENCY": "rag_max_concurrency",
    "MOF_RAG_CHECK_RETRY_S": "rag_check_retry_s", "MOF_RAG_DESCRIPTION_NOTES": "rag_description_notes",
    "MOF_RAG_DATE_FILTER_MODE": "rag_date_filter_mode", "MOF_RAG_DATE_FILTER_REFRESH_S": "rag_date_filter_refresh_s",
    "MOF_RAG_DATE_FILTER_RETRY_S": "rag_date_filter_retry_s",
    "MOF_SEC_CONTACT": "sec_contact", "MOF_SEC_MAX_RPS": "sec_max_rps", "MOF_SEC_TIMEOUT_S": "sec_timeout_s",
    "MOF_SEC_RETRIES": "sec_retries", "MOF_SEC_CALL_BUDGET_S": "sec_call_budget_s",
    "MOF_SEC_POSITIONS_LOOKBACK_Y": "sec_positions_lookback_y", "MOF_SEC_MEMO_COMPANIES": "sec_memo_companies",
    "MOF_SEC_FAILURE_TTL_S": "sec_failure_ttl_s", "MOF_SEC_POSITIONS_DONOR_DEPTH": "sec_positions_donor_depth",
    "MOF_SEC_TTM_LOOKBACK_M": "sec_ttm_lookback_m",
    "MOF_COMPANIES_SOURCE": "companies_source", "MOF_CACHE": "cache", "MOF_CACHE_DIR": "cache_dir",
    "MOF_CACHE_MAX_GB": "cache_max_gb", "MOF_CACHE_EVICT_TO": "cache_evict_to",
    "MOF_CACHE_TMP_GRACE_S": "cache_tmp_grace_s", "MOF_CACHE_TTL_SUBMISSIONS_H": "cache_ttl_submissions_h",
    "MOF_CACHE_TTL_COMPANYFACTS_H": "cache_ttl_companyfacts_h", "MOF_FILINGS_MAX_DOCS": "filings_max_docs",
    "MOF_INDEX_WORKERS": "index_workers", "MOF_RAG_ONNX_PROVIDER": "rag_onnx_provider",
    "MOF_SEARCH_RRF_K": "search_rrf_k", "MOF_SEARCH_CANDIDATES": "search_candidates",
    "MOF_SEARCH_EIGENMIND": "search_eigenmind", "MOF_FILINGS_DEFAULT_FORMS": "filings_default_forms",
    "MOF_CHUNK_MAX_TOKENS": "chunk_max_tokens", "MOF_CHUNK_OVERLAP_TOKENS": "chunk_overlap_tokens",
}

COMMENTED = re.findall(r"^# (MOF_[A-Z0-9_]+)=(\S+(?: \w+=\S+)*)$", ENV_EXAMPLE, re.MULTILINE)


def test_every_env_example_variable_is_read_by_the_server():
    named = set(re.findall(r"\bMOF_[A-Z0-9_]+\b", ENV_EXAMPLE)) - DEPLOYMENT_ONLY
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
    if isinstance(default, tuple):
        assert value == ",".join(default)
    elif default is None or default == "":  # no default: the example shows a placeholder value
        assert "<" in value, f"{name} has no default; its example must be a <placeholder>"
    elif isinstance(default, bool):  # written as the server reads it: true / false
        assert value == str(default).lower()
    else:
        assert value == str(default)


def test_every_server_variable_has_a_commented_example():
    assert sorted({n for n, _ in COMMENTED} - DEPLOYMENT_ONLY) == SERVER_VARS
