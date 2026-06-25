"""Aspect / topic tagging — which travel themes a record talks about.

Rule-based for now (no API cost): each aspect has a keyword list and a record is
tagged with every aspect whose keywords appear in its text. The result is stored
as a comma-separated `aspects` string (blank if nothing matches).

SWAPPABLE BY DESIGN: the public contract is `tag(records) -> records`, which sets
the `aspects` column. A future LLM-based tagger can implement the same `tag()`
(returning the same comma-separated `aspects` values) and be dropped in without
changing the CSV columns or any caller.
"""

import re

# aspect -> trigger keywords/phrases (matched case-insensitively, word-bounded).
ASPECTS = {
    "safety": [
        "safe", "safety", "unsafe", "crime", "criminal", "dangerous", "danger",
        "mugged", "mugging", "robbed", "robbery", "theft", "hijack", "hijacking",
    ],
    "cost": [
        "expensive", "cheap", "price", "prices", "pricey", "affordable", "cost",
        "costs", "budget", "value for money", "worth the money",
    ],
    "scenery": [
        "beautiful", "view", "views", "mountain", "mountains", "beach", "beaches",
        "scenery", "scenic", "landscape", "stunning", "gorgeous", "sunset", "nature",
    ],
    "food": [
        "food", "eat", "eating", "restaurant", "restaurants", "biltong", "wine",
        "cuisine", "meal", "meals", "braai", "dish", "dishes", "tasty", "delicious",
    ],
    "wildlife": [
        "safari", "animal", "animals", "kruger", "big five", "big 5", "wildlife",
        "elephant", "elephants", "lion", "lions", "game drive", "rhino", "leopard",
        "gorilla", "gorillas", "gorilla trekking", "chimpanzee", "chimpanzees",
        "chimp", "golden monkey", "primate", "primates",
    ],
    "hospitality": [
        "friendly", "welcoming", "people", "rude", "hospitable", "hospitality",
        "locals", "warm", "kind", "polite",
    ],
    "transport": [
        "uber", "taxi", "drive", "driving", "airport", "road", "roads", "flight",
        "flights", "car", "cars", "transport", "traffic", "train", "bus",
    ],
}

# Pre-compile one word-bounded regex per aspect so e.g. "great" doesn't match "eat".
_PATTERNS = {
    aspect: re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
    )
    for aspect, keywords in ASPECTS.items()
}


def classify(text: str):
    """Return the list of aspects mentioned in `text` (possibly empty)."""
    lowered = (text or "").lower()
    return [aspect for aspect, pattern in _PATTERNS.items() if pattern.search(lowered)]


def tag(records):
    """Set the comma-separated `aspects` column on each record in place."""
    for rec in records:
        rec["aspects"] = ",".join(classify(rec.get("text") or ""))
    return records
