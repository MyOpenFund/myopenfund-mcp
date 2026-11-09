"""Company resolution over entity_observations: CIK, ticker, entity_id, or any (former) name."""
from __future__ import annotations

from typing import Any


def _when(row: dict) -> str:
    v = row.get("valid_date")
    return "" if v is None else str(v)  # ISO dates (str or date) sort chronologically as text


def index(entities: list[dict]) -> dict[str, dict]:
    """Entities keyed by CIK (subject_value): names oldest first (consecutive repeats once, `since` = first
    seen), current name, tickers, country."""
    out: dict[str, dict[str, Any]] = {}
    for row in sorted(entities, key=lambda r: (_when(r), str(r.get("name") or ""), str(r.get("id_value") or ""))):
        cik = row["subject_value"]
        e = out.setdefault(cik, {"cik": cik, "entity_id": row.get("entity_id"), "names": [],
                                 "current_name": None, "tickers": [], "country": None})
        if e["entity_id"] is None:
            e["entity_id"] = row.get("entity_id")
        # A name observed again (ticker rows may repeat the current name) is not a new name;
        # a name used again after another one is (A, B, A).
        if row.get("name") and row["name"] != e["current_name"]:
            e["names"].append({"name": row["name"], "since": row.get("valid_date")})
            e["current_name"] = row["name"]
        if row.get("country"):
            e["country"] = row["country"]
        if row.get("id_scheme") == "ticker" and row.get("id_value") and row["id_value"] not in e["tickers"]:
            e["tickers"].append(row["id_value"])
    return out


def _by_name(e: dict) -> tuple:
    return ((e["current_name"] or "").casefold(), e["cik"])


def resolve(idx: dict[str, dict], query: str) -> list[dict]:
    """Matching entities sorted by current name: CIK (any zero padding), ticker, entity_id, then name substring."""
    q = query.strip()
    if not q:
        return []
    if q.isdigit():
        hit = idx.get(q.lstrip("0").zfill(10))  # "320193", "0000320193", "000000320193"
        return [hit] if hit else []
    folded = q.casefold()
    for match in (lambda e: any(t.casefold() == folded for t in e["tickers"]),
                  lambda e: e["entity_id"] == q,
                  lambda e: any(folded in n["name"].casefold() for n in e["names"])):
        hits = [e for e in idx.values() if match(e)]
        if hits:
            return sorted(hits, key=_by_name)
    return []
