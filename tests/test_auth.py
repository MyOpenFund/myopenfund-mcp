import pytest

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
])
def test_parse_tokens_rejects_bad_specs_without_echoing_them(spec):
    with pytest.raises(ConfigError) as exc:
        parse_tokens(spec)
    assert "secret-1" not in str(exc.value)


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


class _Ctx:
    def __init__(self, token):
        self.token = token


class _Tok:
    def __init__(self, scopes):
        self.scopes = scopes


def test_tier_check():
    assert tier_check("act", auth_enabled=False)(_Ctx(None))      # stdio: the local user owns the process
    assert not tier_check("read", auth_enabled=True)(_Ctx(None))
    assert tier_check("read", auth_enabled=True)(_Ctx(_Tok(["read"])))
    assert not tier_check("act", auth_enabled=True)(_Ctx(_Tok(["read"])))
    assert tier_check("act", auth_enabled=True)(_Ctx(_Tok(["read", "act"])))


def test_tier_check_rejects_unknown_tier():
    with pytest.raises(ValueError, match="admin"):
        tier_check("admin", auth_enabled=True)


def test_caller_tier():
    assert caller_tier(None, auth_enabled=False) == "act"
    assert caller_tier(None, auth_enabled=True) == "none"
    assert caller_tier(_Tok(["read"]), auth_enabled=True) == "read"
    assert caller_tier(_Tok(["read", "act"]), auth_enabled=True) == "act"
