"""Can a small local model use the server? A scripted session against a local Ollama.

The model gets the server's tools (converted from list_tools() input schemas to Ollama's
`tools` JSON), one question whose answer is fixed by the test data (tests/fakes.py: the
source 'us' has 3 live documents and one withdrawn), and up to MAX_ROUNDS tool rounds
through the in-memory client over FakeVaultReader.
Pass = it called vault_corpus_stats (grouped by source_code) or vault_search_documents with
source_code='us', and its final answer contains 3.

    MOF_MCP_IT_OLLAMA_URL=http://127.0.0.1:11434 [MOF_MCP_IT_OLLAMA_MODEL=qwen2.5:7b] \
        python tests/llm/llm_smoke.py

Stdlib HTTP only (urllib): no dependency beyond the server's own.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, for fakes.py when run directly

DEFAULT_MODEL = "qwen2.5:7b"
MAX_ROUNDS = 5
HTTP_TIMEOUT_S = 300  # a cold 7B model on CPU can take minutes for its first answer
QUESTION = ("How many documents does the Federal Reserve (source code 'us') have in the vault? "
            "Count only live documents, not withdrawn ones. Use the tools, then answer with the number.")
EXPECTED = "3"

Chat = Callable[[list[dict], list[dict]], dict]  # (messages, tools) -> assistant message


def ollama_tools(tools) -> list[dict]:
    """MCP tool definitions -> Ollama /api/chat `tools` entries (JSON Schema passed through)."""
    out = []
    for tool in tools:
        params = dict(tool.input_schema or {})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        out.append({"type": "function", "function": {
            "name": tool.name, "description": tool.description or "", "parameters": params}})
    return out


def ollama_chat(base_url: str, model: str, *, timeout_s: float = HTTP_TIMEOUT_S) -> Chat:
    """A Chat over POST {base_url}/api/chat (non-streaming)."""
    endpoint = base_url.rstrip("/") + "/api/chat"

    def chat(messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": False,
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())["message"]

    return chat


def _result_text(result) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False, default=str)
    return "\n".join(getattr(block, "text", "") for block in result.content)


async def run_session(client, chat: Chat, question: str = QUESTION,
                      *, max_rounds: int = MAX_ROUNDS) -> tuple[list[tuple[str, dict]], str]:
    """Drive the model through the client; returns (tool calls made, final answer text)."""
    tools = ollama_tools(await client.list_tools())
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    calls: list[tuple[str, dict]] = []
    for round_no in range(max_rounds + 1):  # max_rounds tool rounds, then the answer
        reply = chat(messages, tools)  # blocking: fine in a one-session smoke run
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return calls, reply.get("content") or ""
        if round_no == max_rounds:
            break  # still calling tools after max_rounds: no answer
        for tc in tool_calls:
            name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            calls.append((name, args))
            result = await client.call_tool(name, args, raise_on_error=False)
            messages.append({"role": "tool", "tool_name": name, "content": _result_text(result)})
    return calls, ""


def passed(calls: list[tuple[str, dict]], answer: str) -> bool:
    used_right_tool = any(
        (name == "vault_corpus_stats" and args.get("group_by", "source_code") == "source_code")
        or (name == "vault_search_documents" and args.get("source_code") == "us")
        for name, args in calls)
    return used_right_tool and re.search(rf"(?<!\d){EXPECTED}(?!\d)", answer) is not None


async def smoke(chat: Chat) -> tuple[list[tuple[str, dict]], str]:
    from fakes import FakeVaultReader
    from fastmcp import Client

    from myopenfund_mcp.config import Settings
    from myopenfund_mcp.server import build_server

    async with Client(build_server(Settings(), vault_reader=FakeVaultReader())) as client:
        return await run_session(client, chat)


def main() -> int:
    url = os.environ.get("MOF_MCP_IT_OLLAMA_URL")
    if not url:
        print("MOF_MCP_IT_OLLAMA_URL is not set (e.g. http://127.0.0.1:11434)", file=sys.stderr)
        return 2
    model = os.environ.get("MOF_MCP_IT_OLLAMA_MODEL") or DEFAULT_MODEL
    calls, answer = asyncio.run(smoke(ollama_chat(url, model)))
    for name, args in calls:
        print(f"tool: {name} {json.dumps(args, ensure_ascii=False)}")
    print(f"answer: {answer}")
    ok = passed(calls, answer)
    print(f"{'PASS' if ok else 'FAIL'} ({model})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
