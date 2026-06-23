"""Emotion tagging — a single lightweight emotion label per record.

Rule-based for now (no API cost): keyword cues map to one of
    excited | disappointed | fearful | longing | neutral
A record can trip several cues; we resolve to ONE label by priority
(fearful > disappointed > longing > excited), falling back to "neutral".

SWAPPABLE BY DESIGN: the public contract is `tag(records) -> records`, which sets
the `emotion` column. A future LLM-based tagger can implement the same `tag()`
(returning one of the same labels) without changing the CSV columns or callers.
"""

import re

# emotion -> cue keywords/phrases. "longing" captures want-to-go intent.
CUES = {
    "fearful": [
        "scared", "afraid", "fear", "dangerous", "unsafe", "crime", "worried",
        "nervous", "risky", "terrifying", "frightening",
    ],
    "disappointed": [
        "disappointed", "disappointing", "not worth", "waste", "boring",
        "overrated", "terrible", "awful", "worst", "sad", "unfortunately",
        "rude", "let down", "regret",
    ],
    "longing": [
        "want to go", "wanna go", "i wish", "would love", "hope to", "dream",
        "dreaming", "one day", "planning to", "bucket list", "take me",
        "need to visit", "can't wait", "cant wait", "miss", "someday",
    ],
    "excited": [
        "excited", "amazing", "awesome", "incredible", "love it", "loved it",
        "best", "wow", "stunning", "beautiful", "gorgeous", "fantastic",
        "can't wait to", "so good",
    ],
}

# Resolution priority when multiple cues fire (first match wins).
PRIORITY = ("fearful", "disappointed", "longing", "excited")

_PATTERNS = {
    emotion: re.compile(
        r"\b(?:" + "|".join(re.escape(cue) for cue in cues) + r")\b"
    )
    for emotion, cues in CUES.items()
}


def classify(text: str) -> str:
    """Return a single emotion label for `text` ("neutral" if no cue matches)."""
    lowered = (text or "").lower()
    for emotion in PRIORITY:
        if _PATTERNS[emotion].search(lowered):
            return emotion
    return "neutral"


def tag(records):
    """Set the `emotion` column on each record in place."""
    for rec in records:
        rec["emotion"] = classify(rec.get("text") or "")
    return records
