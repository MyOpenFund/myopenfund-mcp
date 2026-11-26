"""Bearer tokens for the HTTP transport, and the read/act tiers.

MOF_MCP_TOKENS lists one entry per client: `client:sha256hex[:tier]`, comma-separated.
Only the SHA-256 digest of a token is configured; the clear token lives with its
client (`myopenfund-mcp hash-token` makes a pair). Tier `act` (the default, spec Q2)
implies `read`; tiers are case-insensitive. Client ids are 1-32 characters of
A-Z a-z 0-9 _ . - (a urlsafe token is at least 43 characters, so it fails the rule).

Digests are unsalted SHA-256: safe only for high-entropy tokens such as the 256-bit
random ones `myopenfund-mcp hash-token` generates. Never hash a human passphrase: an
unsalted fast hash of a guessable secret falls to a dictionary attack.

Configuration errors never echo an entry's content, not even its client id (a token
of the allow-listed alphabet given in the client field would pass for one): they name
the entry by position only.

Over stdio there is no token: the local user owns the process and gets every tier.

Restricted market sources (market-data spec §13) are two notions, never one set:
- STORED restricted rows a deployment serves: `allowed_restricted_for`, the client's entry of
  MOF_MARKET_RESTRICTED_ALLOW, empty by default and always empty without a client id (stdio included). The
  clients of MOF_MARKET_NO_RESTRICTED_CLIENTS (default `agents`: the agents' token, which may post publicly)
  can never be given an entry.
- LIVE fetches: `live_restricted_for`. Over stdio the fetch uses the caller's own key and returns to that
  caller: their own data, every restricted source. Over HTTP it would use the deployment's key, so serving
  it is redistribution: the client's list again.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastmcp.server.auth import AccessToken, AuthContext, TokenVerifier

from .config import ConfigError

if TYPE_CHECKING:
    from .config import Settings

TIERS = ("read", "act")
SCOPES = {"read": ["read"], "act": ["read", "act"]}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CLIENT_ID = re.compile(r"[A-Za-z0-9_.-]{1,32}")
# open-data-store's source-code rule (open_data_store.sources.SOURCE_CODE_PATTERN), written out: this
# module never imports the [market] extra unless MOF_MARKET_RESTRICTED_ALLOW is set.
_SOURCE_CODE = re.compile(r"[a-z][a-z0-9_]{0,62}")
_ALLOW = "MOF_MARKET_RESTRICTED_ALLOW"


@dataclass(frozen=True)
class TokenEntry:
    client_id: str
    digest: str
    tier: str


def is_valid_client_id(client_id: str) -> bool:
    """The client-id allow-list, shared by MOF_MCP_TOKENS parsing and `hash-token`."""
    return _CLIENT_ID.fullmatch(client_id) is not None


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_tokens(spec: str) -> list[TokenEntry]:
    """Entries of MOF_MCP_TOKENS. Errors name the entry by position, never by content."""
    entries: list[TokenEntry] = []
    items = [item.strip() for item in spec.split(",") if item.strip()]
    for position, item in enumerate(items, start=1):
        parts = [p.strip() for p in item.split(":")]
        if len(parts) not in (2, 3):
            raise ConfigError(f"MOF_MCP_TOKENS entry {position}: expected client:sha256[:tier]")
        client, digest = parts[0], parts[1].lower()
        tier = parts[2].lower() if len(parts) == 3 else "act"
        if not is_valid_client_id(client):  # never echoed: a token may sit here
            raise ConfigError(
                f"MOF_MCP_TOKENS entry {position}: client id must be 1-32 characters of A-Z a-z 0-9 _ . -"
            )
        if not _HEX64.fullmatch(digest):
            raise ConfigError(
                f"MOF_MCP_TOKENS entry {position}: give the token's SHA-256 hex digest, "
                "not the token (myopenfund-mcp hash-token)"
            )
        if tier not in TIERS:
            raise ConfigError(f"MOF_MCP_TOKENS entry {position}: tier must be read or act")
        if any(e.client_id == client for e in entries):
            raise ConfigError(f"MOF_MCP_TOKENS entry {position}: client id already given to an earlier entry")
        if any(e.digest == digest for e in entries):
            raise ConfigError(f"MOF_MCP_TOKENS entry {position}: token already given to another client")
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


def _registry_module(variable: str):
    """open_data_store.sources (>= 0.2.0), else a ConfigError naming `variable` and the [market] extra. Any
    other ImportError (a broken install) propagates with its cause."""
    try:
        import open_data_store.sources  # the dotted import: a missing submodule names itself
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("open_data_store"):
            raise
        sources = None
    else:
        sources = open_data_store.sources
    if sources is None or not hasattr(sources, "restricted_codes"):
        raise ConfigError(f"{variable} needs the [market] extra: the source registry of open-data-store "
                          ">= 0.2.0 is not installed")
    return sources


def _read_registry(variable: str, read):
    try:
        return read()
    except Exception as exc:  # noqa: BLE001 -- a plugin's own text may name private paths or symbols
        raise ConfigError(f"{variable}: the source registry could not be read ({type(exc).__name__}); check "
                          "the installed source plugins (open_data_store.sources entry points)") from None


def registered_restricted_sources(variable: str = _ALLOW) -> frozenset[str]:
    """The `restricted` source codes of the installed open-data-store source registry (its entry points:
    market-data declares massive, tiingo, private_history, manual). Read only when a market variable that
    names sources is set or the market toolset is mounted, so a deployment without the market feature never
    imports it. Errors name `variable` (the setting that needed the registry).

    ConfigError when open-data-store >= 0.2.0 is missing, when the registry cannot be read (a broken plugin
    entry point, a source declared twice: the class only, never the plugin's text) and when it declares no
    restricted source (market-data, which declares them, is not installed). Any other ImportError (a broken
    install) propagates with its cause."""
    sources = _registry_module(variable)
    codes = _read_registry(variable, lambda: sources.restricted_codes())
    if not codes:
        raise ConfigError(f"{variable}: the source registry declares no restricted source: install the [market] "
                          "extra (market-data declares them)")
    return frozenset(codes)


def registered_sources(variable: str) -> frozenset[str]:
    """Every source code of the installed registry, open and restricted: what a list of source codes in
    `variable` may name (a typo is refused, never ignored). The same ConfigErrors as
    `registered_restricted_sources`."""
    sources = _registry_module(variable)
    return frozenset(_read_registry(variable, lambda: sources.discover_sources()))


def parse_restricted_allow(spec: str, client_ids: frozenset[str], restricted: frozenset[str],
                           never: frozenset[str] = frozenset()) -> dict[str, frozenset[str]]:
    """Entries of MOF_MARKET_RESTRICTED_ALLOW, `client:src1|src2,client2:src3`. Each client must be one of
    MOF_MCP_TOKENS (`client_ids`) and not one of `never` (MOF_MARKET_NO_RESTRICTED_CLIENTS: spec §13, the
    agents' token never gets restricted sources), each source a `restricted` code of the registry
    (`restricted`): opening an open source means nothing, and a typo must not pass silently. Errors name the
    entry by position, never by content (a token or a private symbol may sit in it)."""
    allow: dict[str, frozenset[str]] = {}
    never_folded = {n.casefold() for n in never}  # `Agents` is the agents' client too
    items = [item.strip() for item in spec.split(",") if item.strip()]
    for position, item in enumerate(items, start=1):
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 2:
            raise ConfigError(f"{_ALLOW} entry {position}: expected client:source[|source...]")
        client, sources = parts[0], [s.strip() for s in parts[1].split("|")]
        if client not in client_ids:
            raise ConfigError(f"{_ALLOW} entry {position}: the client is not a client of MOF_MCP_TOKENS")
        if client.casefold() in never_folded:
            raise ConfigError(f"{_ALLOW} entry {position}: the client is one of MOF_MARKET_NO_RESTRICTED_CLIENTS, "
                              "which never get restricted sources (spec §13)")
        if client in allow:
            raise ConfigError(f"{_ALLOW} entry {position}: client already given to an earlier entry")
        if sources == [""]:
            raise ConfigError(f"{_ALLOW} entry {position}: give at least one source")
        if not all(_SOURCE_CODE.fullmatch(s) for s in sources):
            raise ConfigError(f"{_ALLOW} entry {position}: a source code must be 1-63 characters of a-z 0-9 _, "
                              "starting with a letter, separated by |")
        if not set(sources) <= restricted:
            raise ConfigError(f"{_ALLOW} entry {position}: a source is not a restricted source of the registry")
        allow[client] = frozenset(sources)
    if not allow:
        raise ConfigError(f"{_ALLOW} is set but holds no entry")
    return allow


def allowed_restricted_for(client_id: str | None, settings: Settings) -> frozenset[str]:
    """Restricted sources whose STORED rows may be served to this caller: its entry of
    MOF_MARKET_RESTRICTED_ALLOW, empty by default. No client id (stdio) gets none: the store's restricted
    rows are not the caller's own data. A client of MOF_MARKET_NO_RESTRICTED_CLIENTS (the agents' token,
    spec §13) has no entry: the configuration refuses one."""
    if client_id is None:
        return frozenset()
    return settings.market_restricted_allow.get(client_id, frozenset())


def live_restricted_for(client_id: str | None, settings: Settings,
                        restricted: frozenset[str]) -> frozenset[str]:
    """Restricted sources this caller may fetch LIVE. Over stdio (no token) the fetch uses the caller's own
    key and returns to that caller: every one of `restricted` (the registry's restricted codes). Over HTTP
    the key is the deployment's, so serving the fetch is redistribution: `allowed_restricted_for`."""
    if settings.transport == "stdio" and client_id is None:
        return frozenset(restricted)
    return allowed_restricted_for(client_id, settings)


def caller_client_id(*, auth_enabled: bool) -> str | None:
    """The client id of the request's bearer token over HTTP; None over stdio (no token)."""
    if not auth_enabled:
        return None
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    return token.client_id if token is not None else None
