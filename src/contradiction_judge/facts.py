"""Rule-based fact extraction from free-text memory entries.

Each memory entry's text is scanned for simple factual claims. Every claim
becomes a Fact: an (attribute, value, kind) triple that can be compared
across entries. This is intentionally small and conservative -- anything
it cannot confidently parse is left to the pluggable LLM judge hook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fact model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    """One normalized factual claim extracted from a memory entry's text."""
    attribute: str   # e.g. "person.age", "person.birthday", "pref:<topic>"
    kind: str        # "numeric" | "date" | "text" | "polarity"
    value: object    # normalized value (float / ISO-date str / str / +1|-1)
    raw: str         # the exact matched substring


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# "I am 20 years old" / "my age is 20" / "I'm 240 months old"
_AGE_RE = re.compile(
    r"\b(?:i am|i'm|my age is)\s+(\d+(?:\.\d+)?)\s*"
    r"(years?\s*old|yrs?\s*old|y\.?o\.?|months?\s*old|days?\s*old)?",
    re.IGNORECASE,
)

# "My birthday is May 5, 2006" / "born on 2006-05-05" / "date of birth 5/5/2006"
_BIRTHDAY_RE = re.compile(
    r"\b(?:birthday(?:\s+is)?|born(?:\s+on)?|date of birth(?:\s+is)?)\s*"
    r"([^.;,]+?)(?=[.;,]|$)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_US_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_MONTH_DAY_YEAR_RE = re.compile(
    r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
    re.IGNORECASE,
)
_DAY_MONTH_YEAR_RE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})",
    re.IGNORECASE,
)
_MONTH_DAY_RE = re.compile(
    r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?!\s*\d)",
    re.IGNORECASE,
)

# "My name is hao"
_NAME_RE = re.compile(
    r"\bmy name is\s+([A-Z][A-Za-z .'\-]{1,40}?)(?=[.;,]|$)",
)

# "I live in Pasadena" / "I'm from Arcadia" / "my address is 2426 ..."
_LOCATION_RE = re.compile(
    r"\b(?:i live in|i'm from|i am from|my hometown is|i moved to)\s+"
    r"([^.;,]+?)(?=[.;,]|$)",
    re.IGNORECASE,
)

# Preferences: "I like pizza" vs "I hate pizza" / "I don't like pizza"
_PREF_POS_RE = re.compile(
    r"\bi (?:really )?(?:like|love|enjoy)\s+([^.;,]+?)(?=[.;,]|$)",
    re.IGNORECASE,
)
_PREF_NEG_RE = re.compile(
    r"\bi (?:(?:really )?(?:do not|don't|doesn't)|dislike|hate|can't stand|detest)"
    r"(?:\s+like|\s+enjoy)?\s+([^.;,]+?)(?=[.;,]|$)",
    re.IGNORECASE,
)

# Roles/status: "I am a student" vs "I am not a student"
_ROLE_RE = re.compile(
    r"\bi(?:'m| am)\s+(not\s+)?an?\s+([a-z][a-z ]{0,40}?)(?=[.;,]|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _parse_date(text: str) -> str | None:
    """Normalize a date mention to ISO 'YYYY-MM-DD' or 'MM-DD' (no year)."""
    text = text.strip()
    m = _ISO_DATE_RE.search(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _US_DATE_RE.search(text)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _MONTH_DAY_YEAR_RE.search(text)
    if m:
        mo = _MONTHS[m.group(1).lower()]
        return f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(2)):02d}"
    m = _DAY_MONTH_YEAR_RE.search(text)
    if m:
        mo = _MONTHS[m.group(2).lower()]
        return f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(1)):02d}"
    m = _MONTH_DAY_RE.search(text)
    if m:
        mo = _MONTHS[m.group(1).lower()]
        return f"{mo:02d}-{int(m.group(2)):02d}"
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_facts(text: str) -> list[Fact]:
    """Extract normalized factual claims from a memory entry's text."""
    if not text:
        return []
    facts: list[Fact] = []

    for m in _AGE_RE.finditer(text):
        amount = float(m.group(1))
        unit = (m.group(2) or "years old").lower()
        if unit.startswith("month"):
            years = amount / 12.0
        elif unit.startswith("day"):
            years = amount / 365.25
        else:
            years = amount
        facts.append(Fact("person.age", "numeric", round(years, 6), m.group(0)))

    for m in _BIRTHDAY_RE.finditer(text):
        parsed = _parse_date(m.group(1))
        if parsed:
            facts.append(Fact("person.birthday", "date", parsed, m.group(0)))

    for m in _NAME_RE.finditer(text):
        facts.append(Fact("person.name", "text", _norm(m.group(1)), m.group(0)))

    for m in _LOCATION_RE.finditer(text):
        facts.append(
            Fact("person.location", "text", _norm(m.group(1)), m.group(0))
        )

    for m in _PREF_POS_RE.finditer(text):
        facts.append(
            Fact("pref:" + _norm(m.group(1)), "polarity", +1, m.group(0))
        )
    for m in _PREF_NEG_RE.finditer(text):
        facts.append(
            Fact("pref:" + _norm(m.group(1)), "polarity", -1, m.group(0))
        )

    for m in _ROLE_RE.finditer(text):
        negated, predicate = m.group(1), _norm(m.group(2))
        # The age pattern already claims "i am <number>"; don't let the role
        # pattern swallow numeric predicates.
        if re.match(r"^\d", predicate):
            continue
        facts.append(
            Fact("role:" + predicate, "polarity", -1 if negated else +1, m.group(0))
        )

    return facts
