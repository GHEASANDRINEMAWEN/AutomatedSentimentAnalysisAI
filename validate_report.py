"""Smoke-test the PDF report end to end, through the dashboard's own filters.

Builds a report the same way the "Generate report" button does — dashboard
load_data -> filter_data -> core.report.build_pdf — so a green run means the
button works, not merely that the report module imports.

By default the narrative comes from the deterministic template, so the run costs
nothing and needs no credentials. Pass `--live` to exercise the real Claude path
(requires ANTHROPIC_API_KEY); the check that every published figure is one the
pipeline computed runs either way.

For the engine's own guarantees — verification, repair, fallback — see
validate_narrative.py, which tests them against a stubbed client.

Usage:
    python validate_report.py                          # South Africa, all years
    python validate_report.py --country Zimbabwe --live
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
from core import facts, narrative, report

OUT_DIR = Path(__file__).parent / "data" / "reports"


def build(df, country, *, year_range, aspects, sentiments, kept_only, granularity,
          live=False):
    """Filter exactly as the dashboard does, then render the PDF."""
    sources = sorted(df["source"].dropna().unique().tolist())
    common = dict(year_range=year_range, aspects=aspects, sentiments=sentiments,
                  sources=sources, kept_only=kept_only)
    filtered = dashboard.filter_data(df, country=country, **common)
    # The same view across every country — drives competitive benchmarking.
    benchmark = dashboard.filter_data(df, country="All", **common)
    pdf, diagnostics = report.build_pdf(
        filtered, country=country, granularity=granularity,
        benchmark_df=benchmark, use_claude=live, return_diagnostics=True,
        filters={
            "Country": country,
            "Year range": f"{year_range[0]}–{year_range[1]}",
            "Aspect (any of)": ", ".join(aspects) if aspects else "all aspects",
            "Sentiment": ", ".join(sentiments),
            "Data source": "all",
            "Relevant records only": "yes" if kept_only else "no",
        },
    )
    return filtered, benchmark, pdf, diagnostics


def check(filtered, benchmark, pdf, diagnostics, country, granularity) -> bool:
    """Assert the PDF is real and that every figure in it was computed."""
    metrics = report.summarize(filtered)
    problems = []
    if len(pdf) < 5_000:
        problems.append(f"PDF suspiciously small ({len(pdf)} bytes)")
    if not pdf.startswith(b"%PDF"):
        problems.append("output is not a PDF")
    if metrics["n"] != len(filtered):
        problems.append("record count disagrees with the frame")

    sections = report.narrative(filtered, country=country, granularity=granularity)
    if metrics["n"]:
        if len(sections) != 7:
            problems.append(f"expected 7 sections, got {len(sections)}")
        if f"{metrics['n']:,}" not in sections[0][1]:
            problems.append("the executive summary is missing the record count")

        # The report's central claim: no number reaches the page that the
        # pipeline did not compute. Re-derive the pack and re-check the prose.
        pack = facts.build(filtered, country=country, granularity=granularity,
                           benchmark_df=benchmark)
        allowed = facts.allowed_numbers(pack)
        for heading, text in sections:
            bad = narrative.unverified(text, allowed)
            if bad:
                problems.append(
                    f"{heading} cites uncomputed figures: "
                    + ", ".join(f"{b:g}" for b in bad))
        # Print the engine's log whenever Claude did not write the whole thing.
        # A silent fall back to the template still produces a valid PDF and a
        # green run, so without this a broken key, a truncated draft or a failed
        # verification all look exactly like success.
        if diagnostics["engine"] != "claude":
            print(f"    NOTE  narrative engine: {diagnostics['engine']}")
            for line in diagnostics["log"]:
                print(f"          {line}")
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
    parser.add_argument("--live", action="store_true",
                        help="Write the narrative with Claude (needs "
                             "ANTHROPIC_API_KEY). Default is the free, "
                             "deterministic template.")
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
        filtered, benchmark, pdf, diagnostics = build(
            df, country, year_range=tuple(args.years), aspects=args.aspects,
            sentiments=args.sentiments, kept_only=not args.include_filtered,
            granularity=args.granularity, live=args.live,
        )
        path = (Path(args.out) if args.out and len(countries) == 1
                else OUT_DIR / dashboard.report_filename(country, tuple(args.years)))
        path.write_bytes(pdf)
        metrics = report.summarize(filtered)
        print(f"\n  {country:<16} {metrics['n']:>6,} records  "
              f"{metrics['pos']:>3.0f}% pos / {metrics['neg']:>3.0f}% neg  "
              f"perception {metrics['perception']:>3}  ->  {path.name} "
              f"({len(pdf) / 1024:,.0f} KB, {diagnostics['engine']})")
        passed = check(filtered, benchmark, pdf, diagnostics, country,
                       args.granularity)
        ok = ok and passed
        if passed and diagnostics.get("headline"):
            print(f"    {diagnostics['headline']}")

    print("\n" + ("All reports generated and validated." if ok
                  else "Some reports FAILED validation."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
