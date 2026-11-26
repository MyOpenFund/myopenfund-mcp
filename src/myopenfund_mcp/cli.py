"""`myopenfund-mcp serve` (default), `myopenfund-mcp hash-token` and `myopenfund-mcp warm`."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import secrets
import sys

from .auth import hash_token, is_valid_client_id
from .config import ConfigError, Settings
from .server import build_server
from .warm import EXIT_USAGE, run_warm

log = logging.getLogger("myopenfund_mcp.cli")

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
        description="Generate a new 256-bit random token for one client. The token is printed on "
                    "stderr (give it to the client), the MOF_MCP_TOKENS entry (client:sha256:tier) "
                    "alone on stdout (configure it on the server). "
                    "It never takes a token of yours: a chosen secret is too weak for an unsalted digest.")
    hash_cmd.add_argument("client_id", help="1-32 characters of A-Z a-z 0-9 _ . -")
    hash_cmd.add_argument("--tier", choices=["read", "act"], default="act",
                          help="read: read-only tools; act: every tool (default)")
    warm_cmd = sub.add_parser(
        "warm", help="fetch facts and filings ahead of any question and index them to ready",
        description="Fetch the facts and the filings of the companies over a window and index every "
                    "document to ready in the cache, so later questions are answered from it. Progress "
                    "on stderr. Exit 0 all ready, 1 a failure, 2 bad arguments or no MOF_SEC_CONTACT, "
                    "130 interrupted. Run it again to resume: documents already ready are skipped.")
    warm_cmd.add_argument("--companies", required=True,
                          help="comma-separated tickers, CIKs or names (at most MOF_COMPANIES_MAX_COMPANIES)")
    warm_cmd.add_argument("--forms", default=None,
                          help="comma-separated forms, e.g. 10-K,10-Q (default MOF_FILINGS_DEFAULT_FORMS)")
    warm_cmd.add_argument("--since", required=True, type=_date, help="report period end on or after, YYYY-MM-DD")
    warm_cmd.add_argument("--until", type=_date, default=None,
                          help="report period end on or before, YYYY-MM-DD (default: no limit)")
    warm_cmd.add_argument("--poll-interval", type=float, default=None, metavar="SECONDS",
                          help="seconds between two looks at the progress (default MOF_WARM_POLL_INTERVAL_S)")
    return parser


def _date(raw: str) -> dt.date:
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date YYYY-MM-DD: {raw!r}") from None


def _warm(args) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"myopenfund-mcp: configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return run_warm(settings, companies=args.companies, forms=args.forms, since=args.since,
                    until=args.until, poll_interval_s=args.poll_interval)


def _hash_token(client_id: str, tier: str) -> int:
    if not is_valid_client_id(client_id):
        print("myopenfund-mcp: client id must be 1-32 characters of A-Z a-z 0-9 _ . -", file=sys.stderr)
        return 2
    token = secrets.token_urlsafe(TOKEN_BYTES)
    # The token goes to stderr, for the human only; stdout holds just the entry, so
    # `hash-token alice >> .env` never writes the clear token into a file.
    print(f"token (give it to the client, store nowhere else): {token}", file=sys.stderr)
    print(f"MOF_MCP_TOKENS entry: {client_id}:{hash_token(token)}:{tier}")
    return 0


def _serve() -> int:
    # stdout is the stdio transport: every log line goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # The Qdrant client's HTTP library logs one INFO line per request, with the server's address: the
    # audit line already says a call happened. With this, no line holds the address at the INFO level
    # set above (at DEBUG, the `httpcore` logger would still print the host and port it connects to).
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
        if settings.tokens:  # stdio has no token: the local user owns the process
            log.warning("MOF_MCP_TOKENS is ignored over stdio")
        server.run(transport="stdio", show_banner=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "hash-token":
        return _hash_token(args.client_id, args.tier)
    if args.command == "warm":
        return _warm(args)
    return _serve()


if __name__ == "__main__":
    sys.exit(main())
