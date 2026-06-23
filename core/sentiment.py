"""Source-agnostic sentiment scoring with VADER.

VADER is a lightweight, rule-based scorer tuned for short social-media text,
which is exactly what every provider feeds us. It runs on a record's `text`
field and knows nothing about which platform produced it.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Standard VADER cutoffs for the compound score.
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05

_analyzer = SentimentIntensityAnalyzer()


def _label(compound: float) -> str:
    if compound >= POS_THRESHOLD:
        return "positive"
    if compound <= NEG_THRESHOLD:
        return "negative"
    return "neutral"


def score(records):
    """Attach `sentiment_label` and `sentiment_score` to each record in place.

    `records` is a list of common-record dicts. Returns the same list for
    convenience.
    """
    for rec in records:
        text = (rec.get("text") or "").strip()
        compound = _analyzer.polarity_scores(text)["compound"] if text else 0.0
        rec["sentiment_score"] = compound
        rec["sentiment_label"] = _label(compound)
    return records
