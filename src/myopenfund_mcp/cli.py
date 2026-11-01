"""`myopenfund-mcp serve` (default) and `myopenfund-mcp hash-token`."""
from __future__ import annotations

import argparse
import logging
import secrets
import sys

from .auth import hash_token, is_valid_client_id
from .config import ConfigError, Settings
from .server import build_server

TOKEN_BYTES = 32  # 256 bits: the unsalted SHA-256 digest is only safe for high-entropy tokens


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myopenfund-mcp",
        description="One MCP server over the MyOpenFund bricks. Configured by MOF_* environment "
                    "variables (see .env.example).")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the server (default; transport from MOF_MCP_TRANSPORT)")
    hash_cmd = sub.add_parser(
        "hash-token", help="generate a new random client token and its MOF_MCP_TOKENS entry",
        description="Generate a new 256-bit random token for one client and print it with the "
                    "MOF_MCP_TOKENS entry (client:sha256:tier) to configure on the server. "
                    "It never takes a token of yours: a chosen secret is too weak for an unsalted digest.")
    hash_cmd.add_argument("client_id", help="1-32 characters of A-Z a-z 0-9 _ . -")
    hash_cmd.add_argument("--tier", choices=["read", "act"], default="act",
                          help="read: read-only tools; act: every tool (default)")
    return parser


def _hash_token(client_id: str, tier: str) -> int:
    if not is_valid_client_id(client_id):
        print("myopenfund-mcp: client id must be 1-32 characters of A-Z a-z 0-9 _ . -", file=sys.stderr)
        return 2
    token = secrets.token_urlsafe(TOKEN_BYTES)
    print(f"token (give it to the client, store nowhere else): {token}")
    print(f"MOF_MCP_TOKENS entry: {client_id}:{hash_token(token)}:{tier}")
    return 0


def _serve() -> int:
    # stdout is the stdio transport: every log line goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = Settings.from_env()
        server = build_server(settings)
    except ConfigError as exc:  # messages never echo a token (auth.parse_tokens)
        print(f"myopenfund-mcp: configuration error: {exc}", file=sys.stderr)
        return 2
    if settings.transport == "http":
        server.run(transport="http", host=settings.host, port=settings.port, path=settings.path,
                   show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "hash-token":
        return _hash_token(args.client_id, args.tier)
    return _serve()


if __name__ == "__main__":
    sys.exit(main())
