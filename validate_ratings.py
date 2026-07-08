"""Validate our sentiment model against review star ratings (ANALYSIS ONLY).

Reviews carry a 1-5 star `rating` from the source, which is an independent
ground-truth-ish signal for sentiment. This compares each review's model
`sentiment_label` to the sentiment implied by its stars:

    4-5 stars -> positive     3 stars -> neutral     1-2 stars -> negative

(fractional aggregate ratings use 3.5 / 2.5 cutoffs). It prints the overall
agreement rate, a confusion matrix, per-star accuracy, and a few of the
sharpest disagreements so we can eyeball where the model is wrong on reviews.

Reads data/records.jsonl and NEVER writes — the stored data is untouched.

    python validate_ratings.py
"""

import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from core import store

REVIEW_SOURCES = ("google_hotels", "tripadvisor")
CLASSES = ("positive", "neutral", "negative")
# Ordinal position for measuring how far apart two labels are.
_ORDINAL = {"positive": 1, "neutral": 0, "negative": -1}


def expected_from_rating(rating: float) -> str:
    """Sentiment implied by a 1-5 star rating (3.5 / 2.5 cutoffs)."""
    if rating >= 3.5:
        return "positive"
    if rating >= 2.5:
        return "neutral"
    return "negative"


def _rating_of(rec):
    r = rec.get("rating")
    try:
        return float(r) if r is not None and r != "" else None
    except (TypeError, ValueError):
        return None


def _clip(text, width=88):
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main():
    records = store.read_all()
    reviews = [
        r for r in records
        if r.get("source") in REVIEW_SOURCES and _rating_of(r) is not None
        and r.get("sentiment_label") in CLASSES
    ]

    print("=" * 74)
    print("Sentiment model vs. review star rating  (analysis only — no data changed)")
    print("-" * 74)
    print(f"Review records with a star rating: {len(reviews)}")
    if not reviews:
        print("No rated review records found. Pull reviews first "
              "(python run.py --provider serpapi_reviews ...).")
        print("=" * 74)
        return

    agree = 0
    # confusion[expected][predicted] = count
    confusion = defaultdict(lambda: defaultdict(int))
    per_star_total = defaultdict(int)
    per_star_agree = defaultdict(int)
    disagreements = []

    for rec in reviews:
        rating = _rating_of(rec)
        exp = expected_from_rating(rating)
        pred = rec["sentiment_label"]
        confusion[exp][pred] += 1
        star_key = round(rating)
        per_star_total[star_key] += 1
        if exp == pred:
            agree += 1
            per_star_agree[star_key] += 1
        else:
            severity = abs(_ORDINAL[exp] - _ORDINAL[pred])
            score = rec.get("sentiment_score")
            conf = abs(score) if isinstance(score, (int, float)) else 0.0
            disagreements.append((severity, conf, rating, exp, pred, rec))

    rate = 100 * agree / len(reviews)
    print(f"Agreement (model label == star-implied label): "
          f"{agree}/{len(reviews)} = {rate:.1f}%")

    # --- Confusion matrix -------------------------------------------------- #
    print("-" * 74)
    print("Confusion matrix  (rows = star-implied, cols = model prediction):")
    header = " " * 14 + "".join(f"{c:>10}" for c in CLASSES) + f"{'total':>10}"
    print(header)
    for exp in CLASSES:
        row_total = sum(confusion[exp].values())
        cells = "".join(f"{confusion[exp].get(p, 0):>10}" for p in CLASSES)
        print(f"  {exp:<12}{cells}{row_total:>10}")
    col_line = "".join(
        f"{sum(confusion[e].get(p, 0) for e in CLASSES):>10}" for p in CLASSES)
    print(f"  {'total':<12}{col_line}{len(reviews):>10}")

    # --- Per-star accuracy ------------------------------------------------- #
    print("-" * 74)
    print("Accuracy by star bucket:")
    for star in sorted(per_star_total, reverse=True):
        tot = per_star_total[star]
        acc = 100 * per_star_agree[star] / tot if tot else 0
        bar = "#" * int(acc / 5)
        print(f"  {star}★  n={tot:<4} agree {acc:5.1f}%  {bar}")

    # --- Sample disagreements (sharpest first) ----------------------------- #
    # Sort by how far apart (pos<->neg worst), then by model confidence.
    disagreements.sort(key=lambda t: (t[0], t[1]), reverse=True)
    show = min(8, len(disagreements))
    print("-" * 74)
    print(f"{show} sharpest disagreements (model is confident but stars say otherwise):")
    for severity, conf, rating, exp, pred, rec in disagreements[:show]:
        score = rec.get("sentiment_score")
        score_str = f"{score:+.3f}" if isinstance(score, (int, float)) else "n/a"
        print(f"\n  {rating:.1f}★ -> expected {exp:<8} | model said {pred:<8} "
              f"(score {score_str}, {rec.get('source')})")
        print(f"    {_clip(rec.get('text'))}")
        if rec.get("url"):
            print(f"    {rec.get('url')}")

    print("=" * 74)
    print(f"Summary: {rate:.1f}% agreement on {len(reviews)} rated reviews. "
          f"{len(disagreements)} disagreements.")
    print("Note: models rarely emit 'neutral', so 3★ reviews often count as "
          "disagreements — read the matrix, not just the headline rate.")
    print("=" * 74)


if __name__ == "__main__":
    main()
