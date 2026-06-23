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
    print_summary(args.provider, args.country, records, added)


if __name__ == "__main__":
    main()
