"""Pairwise contradiction detection + arbitration suggestions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .facts import Fact, extract_facts
from .hooks import get_judge

_KIND_LABELS = {
    "numeric": "numeric conflict",
    "date": "date conflict",
    "text": "factual conflict",
    "polarity": "polarity conflict",
    "semantic": "semantic conflict",
}

# Attributes that almost certainly reflect a *change* (people move, tastes
# shift) rather than a recording error -- the arbitration note says so.
_CHANGE_PRONE = {"person.location"} | set()  # preferences handled via pref:


def _facts_conflict(a: Fact, b: Fact) -> dict[str, str] | None:
    """Compare two facts on the same attribute; return conflict info or None."""
    if a.attribute != b.attribute or a.kind != b.kind:
        return None

    if a.kind == "numeric":
        if abs(float(a.value) - float(b.value)) > 1e-9:
            return {
                "kind": "numeric",
                "explanation": (
                    f"'{a.raw.strip()}' vs '{b.raw.strip()}': "
                    f"{a.attribute} cannot be both {a.value} and {b.value}."
                ),
            }
    elif a.kind == "date":
        if a.value != b.value:
            return {
                "kind": "date",
                "explanation": (
                    f"'{a.raw.strip()}' vs '{b.raw.strip()}': "
                    f"{a.attribute} cannot be both {a.value} and {b.value}."
                ),
            }
    elif a.kind == "text":
        if a.value != b.value:
            change_note = ""
            if a.attribute in _CHANGE_PRONE or a.attribute.startswith("pref:"):
                change_note = " This may reflect a genuine change over time."
            return {
                "kind": "text",
                "explanation": (
                    f"'{a.raw.strip()}' vs '{b.raw.strip()}': "
                    f"{a.attribute} recorded as both '{a.value}' and '{b.value}'."
                    + change_note
                ),
            }
    elif a.kind == "polarity":
        if a.value != b.value:
            if a.attribute.startswith("pref:"):
                topic = a.attribute[len("pref:"):]
                desc = f"preference toward '{topic}'"
            else:
                desc = f"status '{a.attribute[len('role:'):]}'"
            return {
                "kind": "polarity",
                "explanation": (
                    f"'{a.raw.strip()}' vs '{b.raw.strip()}': "
                    f"opposite {desc}. Both cannot hold at once."
                ),
            }
    return None


def _parse_time(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def arbitrate(entry_a: dict, entry_b: dict, kind: str) -> dict[str, str]:
    """Suggest how to resolve the contradiction.

    Default policy: recency wins when both entries are timestamped (facts
    change; the newer record is more likely current). Without timestamps,
    flag for manual review.
    """
    ta = _parse_time(entry_a.get("created_at"))
    tb = _parse_time(entry_b.get("created_at"))
    aid = str(entry_a.get("id", "?"))
    bid = str(entry_b.get("id", "?"))

    if ta and tb:
        keep = aid if ta >= tb else bid
        drop = bid if ta >= tb else aid
        return {
            "strategy": "recency",
            "recommendation": f"keep {keep}, drop or correct {drop}",
            "reason": (
                f"Both entries are timestamped; the newer record ({keep}) is "
                f"more likely to be current. Verify before deleting."
            ),
        }
    return {
        "strategy": "manual",
        "recommendation": f"review {aid} vs {bid} manually",
        "reason": (
            "At least one entry has no timestamp, so recency cannot decide. "
            "Ask the user which fact is current, then correct the stale entry."
        ),
    }


def check_memories(entries: list[dict]) -> dict:
    """Check memory entries for contradictions.

    Each entry: {"id": ..., "text": "...", "created_at": "ISO8601" (optional)}.

    Returns a report dict with "entries", "pairs_checked", "skipped" and
    "contradictions" (each with entry_a, entry_b, kind, engine, facts,
    explanation and arbitration).
    """
    parsed = []
    skipped = []
    for entry in entries:
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            skipped.append(entry.get("id", "?"))
            continue
        parsed.append({"entry": entry, "facts": extract_facts(text)})

    contradictions: list[dict] = []
    judge = get_judge()

    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            ea = parsed[i]["entry"]
            eb = parsed[j]["entry"]
            aid = str(ea.get("id", i))
            bid = str(eb.get("id", j))
            fa, fb = parsed[i]["facts"], parsed[j]["facts"]

            # 1) rule-based pairwise fact comparison
            for x in fa:
                for y in fb:
                    conflict = _facts_conflict(x, y)
                    if conflict:
                        contradictions.append(
                            {
                                "entry_a": aid,
                                "entry_b": bid,
                                "kind": conflict["kind"],
                                "engine": "rules",
                                "facts": [
                                    {"attribute": x.attribute,
                                     "value": x.value, "raw": x.raw},
                                    {"attribute": y.attribute,
                                     "value": y.value, "raw": y.raw},
                                ],
                                "explanation": conflict["explanation"],
                                "arbitration": arbitrate(ea, eb, conflict["kind"]),
                            }
                        )

            # 2) pluggable LLM judge: only when rules found nothing for the pair
            pair_had_rules = any(
                c["entry_a"] == aid and c["entry_b"] == bid
                for c in contradictions
            )
            if judge and not pair_had_rules:
                verdict = judge(ea.get("text", ""), eb.get("text", ""))
                if isinstance(verdict, dict) and verdict.get("contradiction"):
                    contradictions.append(
                        {
                            "entry_a": aid,
                            "entry_b": bid,
                            "kind": "semantic",
                            "engine": "llm",
                            "facts": [],
                            "explanation": verdict.get(
                                "explanation",
                                "The LLM judge flagged these entries as contradictory.",
                            ),
                            "arbitration": arbitrate(ea, eb, "semantic"),
                        }
                    )

    return {
        "entries": len(parsed),
        "pairs_checked": len(parsed) * (len(parsed) - 1) // 2,
        "skipped": skipped,
        "contradictions": contradictions,
    }
