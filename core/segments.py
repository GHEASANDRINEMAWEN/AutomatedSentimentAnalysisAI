"""Visitor segment tagging — what KIND of traveller a record sounds like.

Rule-based for now (no API cost): each segment has a keyword list and a record
is scored by how many DISTINCT keywords of each segment it uses. The best-scoring
segment wins; ties break on PRIORITY. Records that match nothing are
"unclassified" — most comments genuinely do not signal a travel style, and
guessing one would poison every segment average downstream.

Result is stored as a single `segment` string, one of:
    adventure | luxury | business | budget | unclassified

Scoring by distinct-keyword count rather than first-match-wins (the approach in
emotion.py) matters here because the vocabularies overlap: "we hiked to the ruins
then stayed at a cheap hostel" is a budget trip that mentions adventure, and a
priority-ordered scan would silently lose whichever segment it checked second.

SWAPPABLE BY DESIGN: the public contract is `tag(records) -> records`, which sets
the `segment` column. An LLM-based classifier can implement the same `tag()`
(returning the same labels) without changing the CSV columns or any caller.
"""

import re

UNCLASSIFIED = "unclassified"

# segment -> trigger keywords/phrases (matched case-insensitively, word-bounded).
SEGMENTS = {
    "adventure": [
        "safari", "wildlife", "hiking", "hike", "hiked", "trek", "trekking",
        "remote", "ruins", "nature", "mountain", "mountains", "climb",
        "climbing", "adventure", "adventurous", "camping", "camp", "trail",
        "trails", "wilderness", "national park", "game drive", "gorilla",
        "kayak", "rafting", "diving", "dive", "snorkel", "snorkelling",
        "snorkeling", "off the beaten", "backcountry", "summit", "canyon",
        "waterfall", "waterfalls", "jungle", "rainforest", "desert trek",
    ],
    "luxury": [
        "resort", "spa", "fine dining", "luxury", "luxurious", "comfort",
        "comfortable", "five star", "5 star", "five-star", "boutique",
        "upscale", "high end", "high-end", "premium", "villa", "suite",
        "suites", "concierge", "butler", "champagne", "gourmet", "michelin",
        "first class", "business class", "private guide", "private tour",
        "infinity pool", "lodge", "pampered", "indulgent", "opulent",
    ],
    "business": [
        "conference", "meeting", "meetings", "work trip", "business trip",
        "wifi", "wi-fi", "internet speed", "reliable", "airport", "layover",
        "transit", "convention", "summit meeting", "client", "colleague",
        "colleagues", "co-working", "coworking", "workspace", "office",
        "expo", "trade show", "seminar", "networking", "per diem",
        "corporate", "on business", "for work",
    ],
    "budget": [
        "cheap", "cheaper", "hostel", "hostels", "guesthouse", "guest house",
        "budget", "backpack", "backpacking", "backpacker", "backpackers",
        "affordable", "party", "partying", "shoestring", "dorm", "dorms",
        "couchsurf", "couchsurfing", "street food", "local bus", "matatu",
        "cheap eats", "bargain", "haggle", "free walking tour", "low cost",
        "low-cost", "save money", "on a budget",
    ],
}

# Tie-break order when two segments score equally. Business and budget lead
# because their cues are the most specific — "conference", "hostel" and
# "layover" say what the trip IS, whereas "nature" and "comfort" are adjectives
# any traveller reaches for.
PRIORITY = ("business", "budget", "luxury", "adventure")

# One word-bounded regex per segment, so "camp" cannot fire inside "campaign"
# and "dive" cannot fire inside "diverse". Longest-first alternation so
# "business trip" is preferred over a bare overlap.
_PATTERNS = {
    segment: re.compile(
        r"\b(?:" + "|".join(
            re.escape(kw) for kw in sorted(keywords, key=len, reverse=True)
        ) + r")\b"
    )
    for segment, keywords in SEGMENTS.items()
}


def scores(text: str) -> dict:
    """Distinct keyword hits per segment (segments with no hits are omitted)."""
    lowered = (text or "").lower()
    out = {}
    for segment, pattern in _PATTERNS.items():
        hits = set(pattern.findall(lowered))
        if hits:
            out[segment] = len(hits)
    return out


def classify(text: str) -> str:
    """Return one segment label for `text`, or "unclassified" if nothing fires."""
    found = scores(text)
    if not found:
        return UNCLASSIFIED
    best = max(found.values())
    leaders = [s for s, v in found.items() if v == best]
    if len(leaders) == 1:
        return leaders[0]
    for segment in PRIORITY:          # deterministic tie-break
        if segment in leaders:
            return segment
    return UNCLASSIFIED


def tag(records):
    """Set the `segment` column on each record in place."""
    for rec in records:
        rec["segment"] = classify(rec.get("text") or "")
    return records
