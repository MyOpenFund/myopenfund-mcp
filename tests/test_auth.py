import secrets

import pytest
from fastmcp.server.auth import AccessToken, AuthContext
from fastmcp.tools import Tool

from myopenfund_mcp.auth import (
    HashedTokenVerifier,
    caller_tier,
    hash_token,
    parse_tokens,
    tier_check,
)
from myopenfund_mcp.config import ConfigError

D1, D2 = hash_token("secret-1"), hash_token("secret-2")


def test_parse_tokens_defaults_tier_to_act():
    entries = parse_tokens(f"marc:{D1}, agent:{D2}:read")
    assert [(e.client_id, e.tier) for e in entries] == [("marc", "act"), ("agent", "read")]


def test_parse_tokens_normalises_uppercase_digest_and_whitespace():
    entries = parse_tokens(f"  marc : {D1.upper()} ,\n agent:{D2} : read  ")
    assert [(e.client_id, e.digest, e.tier) for e in entries] == [
        ("marc", D1, "act"),
        ("agent", D2, "read"),
    ]


@pytest.mark.parametrize("spec", [
    "marc",                       # no digest
    "marc:secret-1",              # clear token instead of a digest
    f"marc:{D1}:admin",           # unknown tier
    f"marc:{D1},marc:{D2}",       # duplicate client
    f"a:{D1},b:{D1}",             # duplicate token
    f"a:{D1},b:{D1.upper()}",     # duplicate token, other case
    f":{D1}",                     # empty client id
    f"  :{D1}:read",              # blank client id
    " , ",                        # nothing
    "secret-1:marc",              # clear token first, client second
    "secret-1:marc,secret-1:bob", # clear tokens first, twice
    f"secret-1 x:{D1}",           # clear token with a space in the client field
])
def test_parse_tokens_rejects_bad_specs_without_echoing_them(spec):
    with pytest.raises(ConfigError) as exc:
        parse_tokens(spec)
    assert "secret-1" not in str(exc.value)


def test_parse_tokens_never_echoes_a_realistic_token_given_as_client():
    token = secrets.token_urlsafe(32)
    with pytest.raises(ConfigError) as exc:
        parse_tokens(f"marc:{D1},{token}:marc")
    assert token not in str(exc.value)
    assert str(exc.value) == (
        "MOF_MCP_TOKENS entry 2: client id must be 1-32 characters of A-Z a-z 0-9 _ . -"
    )


@pytest.mark.parametrize("client", ["marc", "agent-1", "ops.bot-1", "a_b", "x" * 32])
def test_parse_tokens_accepts_allow_listed_client_ids(client):
    assert parse_tokens(f"{client}:{D1}")[0].client_id == client


@pytest.mark.parametrize("client", ["x" * 33, "marc smith", "marc/1", "é", "ma\nrc"])
def test_parse_tokens_rejects_client_ids_off_the_allow_list(client):
    with pytest.raises(ConfigError, match=r"^MOF_MCP_TOKENS entry 1: client id must be"):
        parse_tokens(f"{client}:{D1}")


@pytest.mark.parametrize("digest", [f"{D1}0", f"{D1[:-1]}\n{D1[-1]}"])
def test_parse_tokens_rejects_digest_that_is_not_exactly_64_hex(digest):
    with pytest.raises(ConfigError, match="entry 1"):
        parse_tokens(f"marc:{digest}:read")


def test_parse_tokens_accepts_tier_case_insensitively():
    entries = parse_tokens(f"marc:{D1}:READ,agent:{D2}:Act")
    assert [e.tier for e in entries] == ["read", "act"]


@pytest.mark.anyio
async def test_verifier_maps_tokens_to_scopes():
    v = HashedTokenVerifier(parse_tokens(f"marc:{D1},agent:{D2}:read"))
    marc = await v.verify_token("secret-1")
    assert (marc.client_id, marc.scopes) == ("marc", ["read", "act"])
    agent = await v.verify_token("secret-2")
    assert (agent.client_id, agent.scopes) == ("agent", ["read"])
    assert await v.verify_token("wrong") is None


@pytest.mark.anyio
async def test_verifier_accepts_uppercase_configured_digest():
    v = HashedTokenVerifier(parse_tokens(f"marc:{D1.upper()}"))
    assert (await v.verify_token("secret-1")).client_id == "marc"


def _tool_fn() -> int:
    return 1


_TOOL = Tool.from_function(_tool_fn, name="t")


def _tok(scopes):
    return AccessToken(token="t", client_id="c", scopes=scopes, expires_at=None)


def _ctx(token):
    return AuthContext(token=token, component=_TOOL)


def test_tier_check():
    assert tier_check("act", auth_enabled=False)(_ctx(None))      # stdio: the local user owns the process
    assert not tier_check("read", auth_enabled=True)(_ctx(None))
    assert tier_check("read", auth_enabled=True)(_ctx(_tok(["read"])))
    assert not tier_check("act", auth_enabled=True)(_ctx(_tok(["read"])))
    assert tier_check("act", auth_enabled=True)(_ctx(_tok(["read", "act"])))


def test_tier_check_rejects_unknown_tier():
    with pytest.raises(ValueError, match="admin"):
        tier_check("admin", auth_enabled=True)


def test_caller_tier():
    assert caller_tier(None, auth_enabled=False) == "act"
    assert caller_tier(None, auth_enabled=True) == "none"
    assert caller_tier(_tok(["read"]), auth_enabled=True) == "read"
    assert caller_tier(_tok(["read", "act"]), auth_enabled=True) == "act"
