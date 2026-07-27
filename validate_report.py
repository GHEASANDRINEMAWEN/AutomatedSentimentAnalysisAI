"""Smoke-test the PDF report end to end, through the dashboard's own filters.

Builds a report the same way the "Generate report" button does — dashboard
load_data -> filter_data -> core.report.build_pdf — so a green run means the
button works, not merely that the report module imports.

Usage:
    python validate_report.py                          # South Africa, all years
    python validate_report.py --country Ghana
    python validate_report.py --country Kenya --aspects safety cost --out k.pdf
    python validate_report.py --all-countries          # one report per country
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import dashboard
from core import report

OUT_DIR = Path(__file__).parent / "data" / "reports"


def build(df, country, *, year_range, aspects, sentiments, kept_only, granularity):
    """Filter exactly as the dashboard does, then render the PDF."""
    sources = sorted(df["source"].dropna().unique().tolist())
    filtered = dashboard.filter_data(
        df, country=country, year_range=year_range, aspects=aspects,
        sentiments=sentiments, sources=sources, kept_only=kept_only,
    )
    pdf = report.build_pdf(
        filtered, country=country, granularity=granularity,
        filters={
            "Country": country,
            "Year range": f"{year_range[0]}–{year_range[1]}",
            "Aspect (any of)": ", ".join(aspects) if aspects else "all aspects",
            "Sentiment": ", ".join(sentiments),
            "Data source": "all",
            "Relevant records only": "yes" if kept_only else "no",
        },
    )
    return filtered, pdf


def check(filtered, pdf, country) -> bool:
    """Assert the report is non-empty and its numbers match the frame."""
    metrics = report.summarize(filtered)
    problems = []
    if len(pdf) < 5_000:
        problems.append(f"PDF suspiciously small ({len(pdf)} bytes)")
    if not pdf.startswith(b"%PDF"):
        problems.append("output is not a PDF")
    if metrics["n"] != len(filtered):
        problems.append("record count disagrees with the frame")
    paragraphs = report.narrative(filtered, country=country)
    if len(paragraphs) < 2 and not filtered.empty:
        problems.append("narrative did not generate")
    if metrics["n"] and f"{metrics['n']:,}" not in paragraphs[0][1]:
        problems.append("headline sentence is missing the record count")
    for problem in problems:
        print(f"    FAIL  {problem}")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="South Africa")
    parser.add_argument("--all-countries", action="store_true")
    parser.add_argument("--years", type=int, nargs=2, default=(2015, 2026))
    parser.add_argument("--aspects", nargs="*", default=[])
    parser.add_argument("--sentiments", nargs="*", default=list(dashboard.SENTIMENTS))
    parser.add_argument("--granularity", default="Year", choices=["Year", "Month"])
    parser.add_argument("--include-filtered", action="store_true",
                        help="Include records that failed the relevance filter.")
    parser.add_argument("--out", default=None, help="Output path (single country).")
    args = parser.parse_args()

    df = dashboard.load_data()
    print(f"Loaded {len(df):,} records "
          f"({df['country'].nunique()} countries, {df['source'].nunique()} sources)")

    countries = (sorted(df["country"].dropna().unique().tolist())
                 if args.all_countries else [args.country])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    for country in countries:
        filtered, pdf = build(
            df, country, year_range=tuple(args.years), aspects=args.aspects,
            sentiments=args.sentiments, kept_only=not args.include_filtered,
            granularity=args.granularity,
        )
        path = (Path(args.out) if args.out and len(countries) == 1
                else OUT_DIR / dashboard.report_filename(country, tuple(args.years)))
        path.write_bytes(pdf)
        metrics = report.summarize(filtered)
        print(f"\n  {country:<16} {metrics['n']:>6,} records  "
              f"{metrics['pos']:>3.0f}% pos / {metrics['neg']:>3.0f}% neg  "
              f"perception {metrics['perception']:>3}  ->  {path.name} "
              f"({len(pdf) / 1024:,.0f} KB)")
        passed = check(filtered, pdf, country)
        ok = ok and passed
        if passed and metrics["n"]:
            headline = report.narrative(filtered, country=country)[0][1]
            print(f"    {headline.replace('<b>', '').replace('</b>', '')}")

    print("\n" + ("All reports generated and validated." if ok
                  else "Some reports FAILED validation."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
