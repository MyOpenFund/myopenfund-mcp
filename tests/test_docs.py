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
