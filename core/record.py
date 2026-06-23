"""The shared common record every provider maps its results into.

Keeping one flat schema means sentiment scoring, storage, and the dashboard
never need to know which platform a row came from.

Required for traceability: every record MUST carry a `timestamp` (the item's
publish date, ISO 8601 UTC) and a `url` (direct link to the comment/post), so
sources can be traced and rows filtered/grouped by date later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

# The canonical field order of the common record.
FIELDS = (
    "source",
    "source_id",
    "country",
    "text",
    "author",
    "timestamp",
    "url",
    "engagement",
    "sentiment_label",
    "sentiment_score",
    "aspects",
    "emotion",
    "relevance_kept",
)


@dataclass
class Record:
    source: str                         # platform name, e.g. "youtube", "reddit"
    source_id: str                      # platform-unique id (used for dedupe)
    country: str                        # country the item is about
    text: str                           # the comment/post text
    author: str = ""                    # display name of the author
    timestamp: str = ""                 # publish date, ISO 8601 UTC
    url: str = ""                       # direct link to the comment/post
    engagement: int = 0                 # likes/score/upvotes
    sentiment_label: Optional[str] = None   # filled in later by core.sentiment
    sentiment_score: Optional[float] = None
    aspects: str = ""                        # filled in later by core.aspects
    emotion: str = ""                        # filled in later by core.emotion
    relevance_kept: Optional[bool] = None    # filled in later by core.relevance

    def to_dict(self) -> dict:
        return asdict(self)


def to_iso8601(value) -> str:
    """Normalize a publish date into ISO 8601 UTC.

    Accepts epoch seconds (Reddit's created_utc) or an ISO string such as
    YouTube's "2021-05-01T12:34:56Z". Returns "" for empty input and passes
    through anything it cannot parse rather than raising.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return text
