"""The command line: `hash-token` makes a token pair, `serve` refuses a bad configuration."""
import json
import logging
import os
import re
import subprocess
import sys

import pytest

from myopenfund_mcp import cli
from myopenfund_mcp.auth import hash_token, parse_tokens

ENTRY = re.compile(r"^MOF_MCP_TOKENS entry: (\S+)$", re.MULTILINE)
TOKEN = re.compile(r"^token \(.*\): (\S+)$", re.MULTILINE)


@pytest.fixture
def clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith("MOF_")]:
        monkeypatch.delenv(name)
    return monkeypatch


@pytest.fixture
def no_server(monkeypatch):
    """Record build_server calls instead of starting anything."""
    calls = []

    class FakeServer:
        def run(self, **kwargs):
            calls.append(("run", kwargs))

    def fake_build(settings, **kwargs):
        calls.append(("build", settings))
        return FakeServer()

    monkeypatch.setattr(cli, "build_server", fake_build)
    return calls


def _hash_token(capsys, *argv):
    """The token goes to stderr (for the human), only the entry to stdout (safe for `>> .env`)."""
    code = cli.main(["hash-token", *argv])
    captured = capsys.readouterr()
    token = TOKEN.search(captured.err).group(1)
    assert token not in captured.out
    assert ENTRY.search(captured.err) is None
    return code, token, ENTRY.search(captured.out).group(1)


def test_hash_token_stdout_is_only_the_entry(capsys):
    assert cli.main(["hash-token", "agent"]) == 0
    captured = capsys.readouterr()
    [line] = captured.out.splitlines()
    assert ENTRY.fullmatch(line)
    assert TOKEN.search(captured.err)


def test_hash_token_prints_a_token_and_its_entry(capsys):
    code, token, entry = _hash_token(capsys, "agent", "--tier", "read")
    assert code == 0
    assert entry == f"agent:{hash_token(token)}:read"
    assert len(token) >= 43  # 32 random bytes, urlsafe base64
    [parsed] = parse_tokens(entry)  # the printed entry is accepted as is
    assert (parsed.client_id, parsed.tier) == ("agent", "read")


def test_hash_token_defaults_to_act_and_never_repeats(capsys):
    _, first, entry = _hash_token(capsys, "alice")
    _, second, _ = _hash_token(capsys, "alice")
    assert entry.endswith(":act")
    assert first != second


@pytest.mark.parametrize("client", ["", "a" * 33, "bad id", "a:b", "a,b", "é", " agent"])
def test_hash_token_refuses_a_client_id_auth_would_refuse(capsys, client):
    assert cli.main(["hash-token", client]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # no token printed for a refused client
    assert "client id" in captured.err


def test_hash_token_takes_no_user_token(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["hash-token", "agent", "my-own-token"])
    assert exc.value.code == 2


def test_hash_token_refuses_an_unknown_tier(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["hash-token", "agent", "--tier", "admin"])
    assert exc.value.code == 2


def test_http_without_tokens_exits_2_and_starts_nothing(clean_env, no_server, capsys):
    clean_env.setenv("MOF_MCP_TRANSPORT", "http")
    assert cli.main([]) == 2
    assert no_server == []
    err = capsys.readouterr().err
    assert "configuration error" in err and "MOF_MCP_TOKENS" in err


def test_bad_token_entry_exits_2_without_echoing_it(clean_env, capsys):
    secret = "s3cr3t-token-" + "x" * 40
    clean_env.setenv("MOF_MCP_TRANSPORT", "http")
    clean_env.setenv("MOF_MCP_TOKENS", f"agent:{secret}")
    assert cli.main(["serve"]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.err and secret not in captured.out


def test_serve_stdio_by_default(clean_env, no_server):
    assert cli.main([]) == 0
    assert no_server[-1] == ("run", {"transport": "stdio", "show_banner": False})


def test_serve_stdio_warns_once_that_tokens_are_ignored(clean_env, no_server, caplog):
    clean_env.setenv("MOF_MCP_TOKENS", f"agent:{'0' * 64}")
    with caplog.at_level(logging.WARNING, logger="myopenfund_mcp"):
        assert cli.main(["serve"]) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == ["MOF_MCP_TOKENS is ignored over stdio"]
    assert "0" * 64 not in caplog.text


def test_serve_stdio_without_tokens_does_not_warn(clean_env, no_server, caplog):
    with caplog.at_level(logging.WARNING, logger="myopenfund_mcp"):
        assert cli.main(["serve"]) == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_serve_http_does_not_warn_about_tokens(clean_env, no_server, caplog):
    clean_env.setenv("MOF_MCP_TRANSPORT", "http")
    clean_env.setenv("MOF_MCP_TOKENS", f"agent:{'0' * 64}")
    with caplog.at_level(logging.WARNING, logger="myopenfund_mcp"):
        assert cli.main(["serve"]) == 0
    assert "ignored" not in caplog.text


def test_serve_refuses_a_vault_url_without_the_extra(clean_env, capsys):
    clean_env.setenv("MOF_VAULT_DATABASE_URL", "postgresql://u:hunter2@h/db")
    clean_env.setitem(sys.modules, "vault_query", None)
    assert cli.main(["serve"]) == 2
    err = capsys.readouterr().err
    assert "[vault] extra is not installed" in err and "hunter2" not in err


def test_serve_http_uses_the_configured_address(clean_env, no_server):
    clean_env.setenv("MOF_MCP_TRANSPORT", "http")
    clean_env.setenv("MOF_MCP_TOKENS", f"agent:{'0' * 64}")
    clean_env.setenv("MOF_MCP_HOST", "0.0.0.0")
    clean_env.setenv("MOF_MCP_PORT", "9123")
    clean_env.setenv("MOF_MCP_PATH", "/x/mcp")
    assert cli.main(["serve"]) == 0
    assert no_server[-1] == ("run", {"transport": "http", "host": "0.0.0.0", "port": 9123,
                                     "path": "/x/mcp", "show_banner": False})


def test_serve_stdio_audit_lines_go_to_stderr_never_stdout(tmp_path):
    """A real `serve` over stdio: stdout holds only JSON-RPC, the audit line lands on stderr."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MOF_", "FASTMCP_"))}
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "server_info", "arguments": {}}},
    ]
    proc = subprocess.Popen([sys.executable, "-m", "myopenfund_mcp.cli", "serve"], env=env, cwd=tmp_path,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    responses, stderr = [], ""
    try:
        for request in requests:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            if "id" in request:
                responses.append(json.loads(proc.stdout.readline()))
        rest, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert [r["id"] for r in responses] == [1, 2]
    assert "result" in responses[1] and not responses[1]["result"].get("isError")
    assert rest.strip() == ""  # nothing but protocol messages on stdout
    assert "tool=server_info client=local outcome=ok" in stderr
