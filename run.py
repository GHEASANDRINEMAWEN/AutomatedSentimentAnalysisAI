"""Orchestrate a sentiment pull: pick a provider + country -> fetch -> score -> store.

Usage:
    python run.py                                              # youtube, South Africa
    python run.py --provider youtube --country "South Africa"
    python run.py --provider youtube --country "South Africa" --max-results 200

Adapters only pull + map; scoring and storage are source-agnostic and live in
core/. Add a provider by dropping a new package under providers/ with a
fetch(country, queries, max_results) function and registering it below.
"""

import argparse
import importlib
import sys

# Comments are full of emoji; make stdout UTF-8 so printing the summary never
# dies on a legacy console codepage (e.g. Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from core import sentiment, store

PROVIDERS = {
    "youtube": "providers.youtube.adapter",
    "reddit": "providers.reddit.adapter",
}


def load_provider(name: str):
    if name not in PROVIDERS:
        raise SystemExit(f"Unknown provider {name!r}. Choose from: {', '.join(PROVIDERS)}")
    return importlib.import_module(PROVIDERS[name])


def date_range(records):
    stamps = sorted(r["timestamp"] for r in records if r.get("timestamp"))
    if not stamps:
        return ("n/a", "n/a")
    return (stamps[0], stamps[-1])


def print_summary(provider: str, country: str, records, added: int):
    print()
    print("=" * 72)
    print(f"Provider: {provider}    Country: {country}")
    print(f"Comments pulled : {len(records)}")
    print(f"New stored      : {added}   (duplicates skipped: {len(records) - added})")
    lo, hi = date_range(records)
    print(f"Date range      : {lo}  ->  {hi}")
    print("-" * 72)
    print("Sample rows (sentiment / text / url):")
    for rec in records[:5]:
        text = " ".join((rec.get("text") or "").split())
        if len(text) > 90:
            text = text[:87] + "..."
        label = rec.get("sentiment_label") or "n/a"
        value = rec.get("sentiment_score")
        value_str = f"{value:+.3f}" if isinstance(value, (int, float)) else "  n/a "
        print(f"\n  - [{label:>8} {value_str}] {text}")
        print(f"    {rec.get('url')}")
    print("=" * 72)


def _clip(value, width: int) -> str:
    """Truncate a value to `width` chars with an ellipsis if needed."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def print_table(records, limit: int = 20):
    """Print the top `limit` records as an aligned console table.

    Shows the eyeball-friendly columns; the full set (including url) is in
    data/records.csv.
    """
    # (header, width, value-fn)
    columns = [
        ("#",     3,  lambda r, i: str(i + 1)),
        ("date",  10, lambda r, i: (r.get("timestamp") or "")[:10]),
        ("source", 8, lambda r, i: r.get("source") or ""),
        ("country", 12, lambda r, i: r.get("country") or ""),
        ("sentiment", 9, lambda r, i: r.get("sentiment_label") or ""),
        ("score", 7, lambda r, i: (
            f"{r['sentiment_score']:+.3f}"
            if isinstance(r.get("sentiment_score"), (int, float)) else "")),
        ("eng", 5, lambda r, i: str(r.get("engagement", ""))),
        ("author", 18, lambda r, i: r.get("author") or ""),
        ("text", 60, lambda r, i: r.get("text") or ""),
    ]

    header = "  ".join(name.ljust(w) for name, w, _ in columns)
    print()
    print(f"Top {min(limit, len(records))} of {len(records)} rows "
          f"(full table with url -> data/records.csv):")
    print(header)
    print("-" * len(header))
    for i, rec in enumerate(records[:limit]):
        row = "  ".join(_clip(fn(rec, i), w).ljust(w) for _, w, fn in columns)
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Tourism sentiment pull (provider + country)."
    )
    parser.add_argument("--provider", default="youtube", choices=list(PROVIDERS))
    parser.add_argument("--country", default="South Africa")
    parser.add_argument(
        "--max-results", type=int, default=None,
        help="Overall cap on items pulled (defaults to the provider's config).",
    )
    args = parser.parse_args()

    provider = load_provider(args.provider)
    print(f"Fetching {args.provider} for {args.country!r} ...")
    records = provider.fetch(args.country, max_results=args.max_results)
    print(f"Fetched {len(records)} records. Scoring with VADER ...")
    sentiment.score(records)
    added = store.append(records)
    rows = store.export_csv()  # regenerate the CSV mirror from the full store
    print_summary(args.provider, args.country, records, added)
    print(f"\nCSV updated: {store.CSV_FILE}  ({rows} rows total)")
    print_table(records, limit=20)


if __name__ == "__main__":
    main()
