"""Orchestrate a pull: pick a provider + country -> fetch -> analyze -> store.

Usage:
    python run.py                                              # youtube, South Africa
    python run.py --provider youtube --country "South Africa"
    python run.py --provider youtube --country "South Africa" --max-results 200
    python run.py --reprocess                                  # re-analyze stored data

The analysis layer (all source-agnostic, in core/):
    relevance.mark  -> is each record about travel/tourism?      (relevance_kept)
    sentiment.score -> transformer sentiment                     (sentiment_label/score)
    aspects.tag     -> travel topics mentioned                   (aspects)
    emotion.tag     -> a single emotion cue                      (emotion)

Adapters only pull + map; analysis and storage live in core/. Add a provider by
dropping a new package under providers/ with a fetch(country, queries,
max_results) function and registering it below.
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

from core import aspects, emotion, relevance, sentiment, store

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


def year_counts(records, year_start=2015, year_end=2026):
    """Return per-year (total, kept) counts, covering year_start..year_end plus
    any other years that show up in the data."""
    from collections import defaultdict

    total = defaultdict(int)
    kept = defaultdict(int)
    for rec in records:
        stamp = rec.get("timestamp") or ""
        year = stamp[:4] if stamp[:4].isdigit() else "????"
        total[year] += 1
        if rec.get("relevance_kept"):
            kept[year] += 1

    years = {str(y) for y in range(year_start, year_end + 1)} | set(total)
    return total, kept, sorted(years)


def print_summary(provider: str, country: str, records, added: int):
    keep = relevance.kept(records)
    print()
    print("=" * 72)
    print(f"Provider: {provider}    Country: {country}")
    print(f"Records pulled         : {len(records)}")
    print(f"Passed relevance filter: {len(keep)}   "
          f"(filtered out: {len(records) - len(keep)})")
    if added is None:
        print(f"Re-processed in place   : {len(records)} records")
    else:
        print(f"New stored             : {added}   "
              f"(duplicates skipped: {len(records) - added})")
    lo, hi = date_range(records)
    print(f"Full date range        : {lo}  ->  {hi}")

    print("-" * 72)
    print("Comments per year (total / kept):")
    total, kept_by_year, years = year_counts(records)
    for year in years:
        bar = "#" * min(total[year] // 10, 50)
        print(f"  {year}: {total[year]:5d} / {kept_by_year[year]:5d} kept  {bar}")

    print("-" * 72)
    print("10 sample KEPT rows (sentiment / aspects / emotion / text / url):")
    for rec in keep[:10]:
        text = " ".join((rec.get("text") or "").split())
        if len(text) > 80:
            text = text[:77] + "..."
        label = rec.get("sentiment_label") or "n/a"
        value = rec.get("sentiment_score")
        value_str = f"{value:+.3f}" if isinstance(value, (int, float)) else "  n/a "
        asp = rec.get("aspects") or "-"
        emo = rec.get("emotion") or "-"
        print(f"\n  - [{label:>8} {value_str}] aspects=[{asp}] emotion={emo}")
        print(f"    {text}")
        print(f"    {rec.get('url')}")
    print("=" * 72)


def print_aspect_report(records):
    """Aspect distribution + per-aspect sentiment breakdown over KEPT records."""
    from collections import defaultdict

    keep = relevance.kept(records)
    counts = defaultdict(int)
    by_sentiment = defaultdict(lambda: defaultdict(int))
    none_count = 0
    for rec in keep:
        tags = [a for a in (rec.get("aspects") or "").split(",") if a]
        if not tags:
            none_count += 1
            continue
        label = rec.get("sentiment_label") or "neutral"
        for aspect in tags:
            counts[aspect] += 1
            by_sentiment[aspect][label] += 1

    print()
    print("=" * 72)
    print(f"Aspect analysis over {len(keep)} relevant records")
    print("-" * 72)
    print("Aspect distribution (records mentioning each aspect):")
    order = sorted(counts, key=counts.get, reverse=True)
    for aspect in order:
        bar = "#" * min(counts[aspect] // 5, 50)
        print(f"  {aspect:<12} {counts[aspect]:5d}  {bar}")
    print(f"  {'(none)':<12} {none_count:5d}")

    print("-" * 72)
    print("Sentiment breakdown per aspect (% positive / neutral / negative):")
    print(f"  {'aspect':<12} {'n':>5}   {'pos':>5} {'neu':>5} {'neg':>5}")
    for aspect in order:
        total = counts[aspect]
        pos = by_sentiment[aspect].get("positive", 0)
        neu = by_sentiment[aspect].get("neutral", 0)
        neg = by_sentiment[aspect].get("negative", 0)
        pct = lambda x: f"{(100 * x / total):4.0f}%" if total else "   0%"
        print(f"  {aspect:<12} {total:>5}   {pct(pos)} {pct(neu)} {pct(neg)}")
    print("=" * 72)


def _video_id_from_url(url: str) -> str:
    """Extract the YouTube video id from a watch URL (…watch?v=ID[&…])."""
    if "v=" not in url:
        return url
    return url.split("v=", 1)[1].split("&", 1)[0]


def print_transcript_summary(records, sample: int = 5):
    """Report the transcript signal: videos covered, chunks, and samples."""
    transcripts = [r for r in records if r.get("source") == "youtube_transcript"]
    videos = {_video_id_from_url(r.get("url") or "") for r in transcripts}
    comments = sum(1 for r in records if r.get("source") == "youtube")

    print()
    print("=" * 72)
    print("Transcript signal")
    print(f"  Comment records      : {comments}")
    print(f"  Videos w/ transcript : {len(videos)}")
    print(f"  Transcript chunks    : {len(transcripts)}")
    print("-" * 72)
    print(f"{min(sample, len(transcripts))} sample transcript chunks (sentiment / text / url):")
    for rec in transcripts[:sample]:
        text = " ".join((rec.get("text") or "").split())
        if len(text) > 90:
            text = text[:87] + "..."
        label = rec.get("sentiment_label") or "n/a"
        value = rec.get("sentiment_score")
        value_str = f"{value:+.3f}" if isinstance(value, (int, float)) else "  n/a "
        kept = "kept" if rec.get("relevance_kept") else "filtered"
        print(f"\n  - [{label:>8} {value_str}] ({kept}) {text}")
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
        ("source", 18, lambda r, i: r.get("source") or ""),
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


def analyze(records):
    """Run the full source-agnostic analysis layer over records (in place)."""
    relevance.mark(records)   # mark, don't drop — filtered rows stay for review
    print("  scoring sentiment with transformer (first run downloads the model) ...")
    sentiment.score(records)
    aspects.tag(records)
    emotion.tag(records)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Tourism sentiment + aspect/emotion analysis (provider + country)."
    )
    parser.add_argument("--provider", default="youtube", choices=list(PROVIDERS))
    parser.add_argument("--country", default="South Africa")
    parser.add_argument(
        "--max-results", type=int, default=None,
        help="Overall cap on items pulled (defaults to the provider's config).",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Skip fetching; re-analyze every record already in the store.",
    )
    args = parser.parse_args()

    if args.reprocess:
        records = store.read_all()
        print(f"Re-processing {len(records)} stored records ...")
        analyze(records)
        store.rewrite(records)          # persist new analysis columns
        added = None
    else:
        provider = load_provider(args.provider)
        print(f"Fetching {args.provider} for {args.country!r} ...")
        records = provider.fetch(args.country, max_results=args.max_results)
        print(f"Fetched {len(records)} records. Analyzing ...")
        analyze(records)
        added = store.append(records)

    rows = store.export_csv()  # regenerate the CSV mirror from the full store
    print_summary(args.provider if not args.reprocess else "reprocess",
                  args.country, records, added)
    if any(r.get("source") == "youtube_transcript" for r in records):
        print_transcript_summary(records, sample=5)
    print_aspect_report(records)
    print(f"\nCSV updated: {store.CSV_FILE}  ({rows} rows total)")
    # Aligned view of the KEPT (relevant) rows for quick eyeballing.
    print_table(relevance.kept(records), limit=20)


if __name__ == "__main__":
    main()
