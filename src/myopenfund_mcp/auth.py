"""Bearer tokens for the HTTP transport, and the read/act tiers.

MOF_MCP_TOKENS lists one entry per client: `client:sha256hex[:tier]`, comma-separated.
Only the SHA-256 digest of a token is configured; the clear token lives with its
client (`myopenfund-mcp hash-token` makes a pair). Tier `act` (the default, spec Q2)
implies `read`. Over stdio there is no token: the local user owns the process and
gets every tier.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

from fastmcp.server.auth import AccessToken, AuthContext, TokenVerifier

from .config import ConfigError

TIERS = ("read", "act")
SCOPES = {"read": ["read"], "act": ["read", "act"]}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TokenEntry:
    client_id: str
    digest: str
    tier: str


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_tokens(spec: str) -> list[TokenEntry]:
    """Entries of MOF_MCP_TOKENS. Errors name the entry by position, never by content."""
    entries: list[TokenEntry] = []
    items = [item.strip() for item in spec.split(",") if item.strip()]
    for position, item in enumerate(items, start=1):
        parts = [p.strip() for p in item.split(":")]
        if len(parts) not in (2, 3) or not parts[0]:
            raise ConfigError(f"MOF_MCP_TOKENS entry {position}: expected client:sha256[:tier]")
        client, digest = parts[0], parts[1].lower()
        tier = parts[2] if len(parts) == 3 else "act"
        if not _HEX64.match(digest):
            raise ConfigError(
                f"MOF_MCP_TOKENS entry {position} ({client}): give the token's SHA-256 hex digest, "
                "not the token (myopenfund-mcp hash-token)"
            )
        if tier not in TIERS:
            raise ConfigError(f"MOF_MCP_TOKENS entry {position} ({client}): tier must be read or act")
        if any(e.client_id == client for e in entries):
            raise ConfigError(f"MOF_MCP_TOKENS entry {position}: client {client!r} appears twice")
        if any(e.digest == digest for e in entries):
            raise ConfigError(f"MOF_MCP_TOKENS entry {position} ({client}): token already given to another client")
        entries.append(TokenEntry(client, digest, tier))
    if not entries:
        raise ConfigError("MOF_MCP_TOKENS is set but holds no entry")
    return entries


class HashedTokenVerifier(TokenVerifier):
    def __init__(self, entries: list[TokenEntry]):
        super().__init__()
        self._entries = tuple(entries)

    async def verify_token(self, token: str) -> AccessToken | None:
        digest = hash_token(token)
        match = None
        for entry in self._entries:  # no early exit: constant work whatever the position
            if hmac.compare_digest(entry.digest, digest):
                match = entry
        if match is None:
            return None
        return AccessToken(token=token, client_id=match.client_id, scopes=list(SCOPES[match.tier]),
                           expires_at=None)


def tier_check(tier: str, *, auth_enabled: bool):
    """A FastMCP auth check: the caller holds `tier` (always true without auth)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    def check(ctx: AuthContext) -> bool:
        if not auth_enabled:
            return True
        return ctx.token is not None and tier in ctx.token.scopes

    return check


def caller_tier(token: AccessToken | None, *, auth_enabled: bool) -> str:
    if not auth_enabled:
        return "act"
    if token is None:
        return "none"
    return "act" if "act" in token.scopes else "read"
