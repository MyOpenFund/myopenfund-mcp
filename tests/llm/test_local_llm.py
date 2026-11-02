"""The local-LLM smoke check (tests/llm/llm_smoke.py).

`test_local_model_answers` is opt-in (`-m llm`, needs MOF_MCP_IT_OLLAMA_URL). The others run by
default: they exercise the schema conversion, the HTTP call and the tool loop against stand-ins
for Ollama, so the opt-in check cannot rot unnoticed.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import llm_smoke  # sibling module: pytest puts this directory on sys.path (rootdir-less tests/)
import pytest
from fakes import FakeVaultReader
from fastmcp import Client

from myopenfund_mcp.config import Settings
from myopenfund_mcp.server import build_server

OLLAMA_URL = os.environ.get("MOF_MCP_IT_OLLAMA_URL")


def client():
    return Client(build_server(Settings(), vault_reader=FakeVaultReader()))


def tool_call(name, arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


class ScriptedChat:
    """Plays fixed assistant replies; records what the model would have been sent."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, tools):
        self.seen.append(([dict(m) for m in messages], tools))
        return self.replies.pop(0)


@pytest.mark.anyio
async def test_ollama_tools_mirror_list_tools():
    async with client() as c:
        listed = await c.list_tools()
    tools = llm_smoke.ollama_tools(listed)
    assert [t["function"]["name"] for t in tools] == [t.name for t in listed]
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in tools)
    by_name = {t["function"]["name"]: t["function"] for t in tools}
    assert "source_code" in by_name["vault_search_documents"]["parameters"]["properties"]
    assert by_name["server_info"]["parameters"]["properties"] == {}
    assert by_name["vault_runs_status"]["description"]  # set through description=, not the docstring
    json.dumps(tools)  # what goes on the wire


@pytest.mark.anyio
async def test_session_feeds_tool_results_back_and_passes():
    chat = ScriptedChat(tool_call("vault_search_documents", {"source_code": "us"}),
                        {"role": "assistant", "content": "The Federal Reserve has 3 documents."})
    async with client() as c:
        calls, answer = await llm_smoke.run_session(c, chat)
    assert calls == [("vault_search_documents", {"source_code": "us"})]
    assert llm_smoke.passed(calls, answer)
    last_messages, _ = chat.seen[-1]
    result = last_messages[-1]
    assert result["role"] == "tool" and result["tool_name"] == "vault_search_documents"
    assert json.loads(result["content"])["total"] == 3  # the fake's live 'us' documents


@pytest.mark.anyio
async def test_string_arguments_and_tool_errors_reach_the_model():
    chat = ScriptedChat(tool_call("vault_get_document", json.dumps({"doc_id": "nope"})),
                        tool_call("vault_corpus_stats", {}),
                        {"role": "assistant", "content": "us: 3"})
    async with client() as c:
        calls, answer = await llm_smoke.run_session(c, chat)
    assert calls == [("vault_get_document", {"doc_id": "nope"}), ("vault_corpus_stats", {})]
    error = chat.seen[1][0][-1]
    assert "[not_found]" in error["content"]
    assert llm_smoke.passed(calls, answer)


@pytest.mark.anyio
async def test_rounds_are_bounded():
    chat = ScriptedChat(*[tool_call("server_info", {})] * 10)
    async with client() as c:
        calls, answer = await llm_smoke.run_session(c, chat, max_rounds=2)
    assert len(chat.seen) == 3 and len(calls) == 2 and answer == ""


@pytest.mark.parametrize(("calls", "answer", "ok"), [
    ([("vault_corpus_stats", {"group_by": "source_code"})], "3", True),
    ([("vault_search_documents", {"source_code": "us"})], "There are 3.", True),
    ([], "3", False),  # guessed without a tool
    ([("vault_search_documents", {"source_code": "fr"})], "3", False),
    ([("vault_corpus_stats", {"group_by": "year"})], "3", False),
    ([("vault_search_documents", {"source_code": "us"})], "4 documents", False),
    ([("vault_search_documents", {"source_code": "us"})], "13 documents", False),
])
def test_pass_criterion(calls, answer, ok):
    assert llm_smoke.passed(calls, answer) is ok


def test_ollama_chat_posts_a_non_streaming_request(monkeypatch):
    for var in ("no_proxy", "NO_PROXY"):  # a developer's http_proxy must not intercept the local stub
        monkeypatch.setenv(var, "*")
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.dumps({"message": {"role": "assistant", "content": "3"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        chat = llm_smoke.ollama_chat(f"http://127.0.0.1:{server.server_port}/", "some-model", timeout_s=5)
        tools = [{"type": "function", "function": {"name": "t", "description": "", "parameters": {}}}]
        reply = chat([{"role": "user", "content": "q"}], tools)
    finally:
        server.shutdown()
        server.server_close()
    assert reply == {"role": "assistant", "content": "3"}
    assert received["path"] == "/api/chat"
    body = received["body"]
    assert body["model"] == "some-model" and body["stream"] is False and body["tools"] == tools
    assert body["messages"] == [{"role": "user", "content": "q"}]


@pytest.mark.llm
@pytest.mark.anyio
@pytest.mark.skipif(not OLLAMA_URL, reason="MOF_MCP_IT_OLLAMA_URL not set")
async def test_local_model_answers():
    model = os.environ.get("MOF_MCP_IT_OLLAMA_MODEL") or llm_smoke.DEFAULT_MODEL
    calls, answer = await llm_smoke.smoke(llm_smoke.ollama_chat(OLLAMA_URL, model))
    assert llm_smoke.passed(calls, answer), f"{model}: calls={calls!r} answer={answer!r}"
